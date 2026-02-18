import math
import os
import random
import sys
from collections.abc import Mapping

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

try:
    from .diff_gnn_utils_schedule import (
        _build_torchgeo_data,
        _enable_determinism,
        _relaxed_binary_assignment,
        _repair_candidate,
        _set_global_seeds,
        get_device,
    )
except Exception:
    from diff_gnn_utils_schedule import (  # type: ignore
        _build_torchgeo_data,
        _enable_determinism,
        _relaxed_binary_assignment,
        _repair_candidate,
        _set_global_seeds,
        get_device,
    )
try:
    from .diff_gnn_models import build_order_model
except Exception:
    from diff_gnn_models import build_order_model  # type: ignore
try:
    from .lssp_postprocess import improve_with_lssp_local_search
except Exception:
    from lssp_postprocess import improve_with_lssp_local_search  # type: ignore

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/diff_gnn_ordering.log")

logger = LogManager.get_logger(__name__)

_FAST_MODE_DEFAULTS = {
    "iter": 600,
    "verbose": 600,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "sinkhorn_iters": 8,
    "order_refine_steps": 1,
    "hard_eval_every": 120,
    "pairwise_mode": "rank_sigmoid",
    "pairwise_temp": 0.7,
}


def _sample_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(x).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def _sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    z = log_alpha
    for _ in range(max(1, int(n_iters))):
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
        z = z - torch.logsumexp(z, dim=0, keepdim=True)
    return torch.exp(z)


def _soft_permutation_from_priority(
    priorities: torch.Tensor,
    temperature: float,
    sinkhorn_iters: int,
    add_gumbel: bool,
    gumbel_scale: float,
) -> torch.Tensor:
    n = int(priorities.shape[0])
    if n <= 0:
        return priorities.new_zeros((0, 0))
    if n == 1:
        return priorities.new_ones((1, 1))

    scores = priorities
    if add_gumbel and gumbel_scale > 0:
        scores = scores + float(gumbel_scale) * _sample_gumbel_like(priorities)

    # Earlier positions correspond to higher priority values.
    pos_targets = torch.linspace(1.0, -1.0, steps=n, device=priorities.device, dtype=priorities.dtype)
    tau = max(float(temperature), 1e-6)
    log_alpha = -((scores.unsqueeze(0) - pos_targets.unsqueeze(1)) ** 2) / tau
    return _sinkhorn(log_alpha, n_iters=sinkhorn_iters)


def _pairwise_before_from_perm(P: torch.Tensor) -> torch.Tensor:
    n = int(P.shape[0])
    if n <= 0:
        return P.new_zeros((0, 0))
    if n == 1:
        return P.new_zeros((1, 1))

    upper = torch.triu(torch.ones((n, n), dtype=P.dtype, device=P.device), diagonal=1)
    before = P.transpose(0, 1) @ upper @ P
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    return (before * (1.0 - eye)).clamp(0.0, 1.0)


def _pairwise_before_from_expected_rank(P: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    O(N^2) approximation of pairwise-before probabilities.

    1) Compute expected position (rank) for each task from soft permutation P.
    2) Convert rank differences to before-probabilities with a sigmoid.
    """
    n = int(P.shape[0])
    if n <= 0:
        return P.new_zeros((0, 0))
    if n == 1:
        return P.new_zeros((1, 1))

    pos = torch.arange(n, device=P.device, dtype=P.dtype)
    exp_rank = (P * pos.unsqueeze(1)).sum(dim=0)  # shape (N,)
    t = max(float(temperature), 1e-6)
    # before[j, i] should be high when rank(j) < rank(i)
    before = torch.sigmoid((exp_rank.unsqueeze(0) - exp_rank.unsqueeze(1)) / t)
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    return (before * (1.0 - eye)).clamp(0.0, 1.0)


def _softmax_beta(vals: torch.Tensor, beta: float) -> torch.Tensor:
    b = max(float(beta), 1e-6)
    return (1.0 / b) * torch.logsumexp(b * vals, dim=0)


def _doubly_stochastic_penalty(P: torch.Tensor) -> torch.Tensor:
    if P.numel() == 0:
        return P.new_tensor(0.0)
    row = P.sum(dim=1) - 1.0
    col = P.sum(dim=0) - 1.0
    return torch.mean(row * row) + torch.mean(col * col)


def _entropy_rows_cols(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if P.numel() == 0:
        return P.new_tensor(0.0)
    row_h = -(P * torch.log(P + eps)).sum(dim=1).mean()
    col_h = -(P * torch.log(P + eps)).sum(dim=0).mean()
    return row_h + col_h


def _order_aware_repair_scores(hard_probs: np.ndarray, prio_hw: np.ndarray, prio_sw: np.ndarray, weight: float) -> np.ndarray:
    """
    Blend hard placement probabilities with ordering-head preference so decode uses
    ordering signals too. Higher score means stronger preference to keep node on HW.
    """
    w = float(max(0.0, min(1.0, weight)))
    prio_delta = np.asarray(prio_hw, dtype=float) - np.asarray(prio_sw, dtype=float)
    prio_delta = np.clip(prio_delta, -50.0, 50.0)
    order_pref = 1.0 / (1.0 + np.exp(-prio_delta))
    base = np.asarray(hard_probs, dtype=float)
    return (1.0 - w) * base + w * order_pref


def _differentiable_makespan_loss_with_order(
    TG,
    probs_tensor,
    prio_hw,
    prio_sw,
    node_list,
    beta_softmax=20.0,
    area_penalty_coeff=1e5,
    entropy_coeff=1e-3,
    usage_balance_coeff=0.5,
    target_hw_frac=None,
    partition_cost_coeff=0.0,
    order_tau=0.5,
    sinkhorn_iters=20,
    gumbel_noise=True,
    gumbel_scale=1.0,
    resource_logit_alpha=2.0,
    order_refine_steps=2,
    order_eps=1e-6,
    perm_reg_coeff=1e-2,
    perm_entropy_coeff=1e-3,
    pairwise_mode="rank_sigmoid",
    pairwise_temp=0.5,
):
    """
    Differentiable loss with soft placement + soft ordering:
      - Placement uses relaxed p_hw.
      - Ordering uses Gumbel-Sinkhorn soft permutations for HW and SW lanes.
      - Makespan surrogate combines DAG precedence and resource precedence.
    """
    device = probs_tensor.device
    dtype = probs_tensor.dtype
    N = probs_tensor.shape[0]

    hw_times = torch.tensor([TG.hardware_costs[n] for n in node_list], dtype=dtype, device=device)
    sw_times = torch.tensor([TG.software_costs[n] for n in node_list], dtype=dtype, device=device)
    areas = torch.tensor([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=dtype, device=device)

    total_area = float(TG.total_area) if getattr(TG, "total_area", 0.0) != 0 else float(areas.sum().item())
    if total_area <= 0:
        total_area = 1.0

    exec_time = probs_tensor * hw_times + (1.0 - probs_tensor) * sw_times

    # Soft permutations and pairwise "before" probabilities per resource lane.
    P_hw = _soft_permutation_from_priority(
        prio_hw,
        temperature=order_tau,
        sinkhorn_iters=sinkhorn_iters,
        add_gumbel=bool(gumbel_noise),
        gumbel_scale=gumbel_scale,
    )
    P_sw = _soft_permutation_from_priority(
        prio_sw,
        temperature=order_tau,
        sinkhorn_iters=sinkhorn_iters,
        add_gumbel=bool(gumbel_noise),
        gumbel_scale=gumbel_scale,
    )

    mode = str(pairwise_mode).lower()
    if mode == "exact":
        before_hw = _pairwise_before_from_perm(P_hw)
        before_sw = _pairwise_before_from_perm(P_sw)
    elif mode in ("rank", "rank_sigmoid", "fast"):
        before_hw = _pairwise_before_from_expected_rank(P_hw, temperature=pairwise_temp)
        before_sw = _pairwise_before_from_expected_rank(P_sw, temperature=pairwise_temp)
    else:
        raise ValueError(f"Unsupported pairwise_mode '{pairwise_mode}'. Use 'rank_sigmoid' or 'exact'.")

    gate_hw = probs_tensor.unsqueeze(1) * probs_tensor.unsqueeze(0)
    p_sw = 1.0 - probs_tensor
    gate_sw = p_sw.unsqueeze(1) * p_sw.unsqueeze(0)

    before_resource = (before_hw * gate_hw) + (before_sw * gate_sw)
    if N > 0:
        eye = torch.eye(N, dtype=dtype, device=device)
        before_resource = before_resource * (1.0 - eye)

    G = TG.graph
    topo_nodes = list(nx.topological_sort(G))
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    topo_idx = [node_to_idx[n] for n in topo_nodes]

    preds = [[] for _ in range(N)]
    for node in topo_nodes:
        i = node_to_idx[node]
        for p in G.predecessors(node):
            pidx = node_to_idx[p]
            c = float(TG.communication_costs.get((p, node), 0.0))
            preds[i].append((pidx, c))

    all_idx = torch.arange(N, device=device)
    zero = exec_time.new_tensor(0.0)

    # Iterative refinement: DAG DP pass + resource-precedence pass.
    F_prev = exec_time.clone()
    refine_steps = max(1, int(order_refine_steps))
    for _ in range(refine_steps):
        F_new = torch.zeros_like(exec_time)
        for i in topo_idx:
            if preds[i]:
                dag_terms = [zero]
                for pidx, comm_cost in preds[i]:
                    comm = torch.abs(probs_tensor[pidx] - probs_tensor[i]) * comm_cost
                    dag_terms.append(F_new[pidx] + comm)
                t_dag = _softmax_beta(torch.stack(dag_terms), beta_softmax)
            else:
                t_dag = zero

            if N <= 1:
                t_res = zero
            else:
                mask = all_idx != i
                res_terms = F_prev[mask] + float(resource_logit_alpha) * torch.log(before_resource[mask, i] + float(order_eps))
                t_res = _softmax_beta(res_terms, beta_softmax)

            start_i = _softmax_beta(torch.stack([t_dag, t_res]), beta_softmax)
            F_new[i] = start_i + exec_time[i]

        F_prev = F_new

    makespan_soft = _softmax_beta(F_prev, beta_softmax)

    area_used = torch.dot(probs_tensor, areas)
    area_frac = area_used / float(total_area)
    area_violation = F.relu(area_frac - float(TG.area_constraint))
    area_penalty = area_penalty_coeff * area_violation

    if target_hw_frac is None:
        target_hw_frac = float(TG.area_constraint)
    usage_balance = usage_balance_coeff * (area_frac - float(target_hw_frac)) ** 2

    entropy_like = torch.mean(probs_tensor * (1.0 - probs_tensor))

    exec_cost = torch.sum(exec_time)
    comm_cost = torch.tensor(0.0, dtype=dtype, device=device)
    for (u, v), c in TG.communication_costs.items():
        if u in node_to_idx and v in node_to_idx:
            pu = probs_tensor[node_to_idx[u]]
            pv = probs_tensor[node_to_idx[v]]
            comm_cost = comm_cost + torch.abs(pu - pv) * float(c)
    expected_partition_cost = exec_cost + comm_cost

    perm_reg = _doubly_stochastic_penalty(P_hw) + _doubly_stochastic_penalty(P_sw)
    perm_entropy = _entropy_rows_cols(P_hw) + _entropy_rows_cols(P_sw)

    loss = (
        makespan_soft
        + area_penalty
        + usage_balance
        + entropy_coeff * entropy_like
        + partition_cost_coeff * expected_partition_cost
        + perm_reg_coeff * perm_reg
        + perm_entropy_coeff * perm_entropy
    )

    return loss, {
        "makespan_surrogate": makespan_soft.item(),
        "area_frac": area_frac.item(),
        "area_penalty": area_penalty.item(),
        "usage_balance": usage_balance.item() if isinstance(usage_balance, torch.Tensor) else usage_balance,
        "entropy_like": entropy_like.item(),
        "expected_partition_cost": expected_partition_cost.item(),
        "perm_reg": perm_reg.item(),
        "perm_entropy": perm_entropy.item(),
        "loss": loss.item(),
    }


def _train_with_relaxed_binary_order(TG, model, data, node_list, config, device):
    """
    Differentiable training loop for placement + ordering.
    """
    lr = float(config.get("lr", 1e-3))
    epochs = int(config.get("epochs", 1500))

    tau_start = float(config.get("tau_start", 1.0))
    tau_final = float(config.get("tau_final", 0.1))

    order_tau_start = float(config.get("order_tau_start", 1.0))
    order_tau_final = float(config.get("order_tau_final", 0.2))
    sinkhorn_iters = int(config.get("sinkhorn_iters", 20))
    gumbel_noise = bool(config.get("gumbel_noise", True))
    gumbel_scale = float(config.get("gumbel_scale", 1.0))
    resource_logit_alpha = float(config.get("resource_logit_alpha", 2.0))
    order_refine_steps = int(config.get("order_refine_steps", 2))

    beta_softmax = float(config.get("beta_softmax", 20.0))
    area_penalty_coeff = float(config.get("area_penalty_coeff", 1e5))
    entropy_coeff = float(config.get("entropy_coeff", 1e-3))
    usage_balance_coeff = float(config.get("usage_balance_coeff", 0.5))
    target_hw_frac = config.get("target_hw_frac", None)
    partition_cost_coeff = float(config.get("partition_cost_coeff", 1e-2))

    perm_reg_coeff = float(config.get("perm_reg_coeff", 1e-2))
    perm_entropy_coeff = float(config.get("perm_entropy_coeff", 1e-3))
    pairwise_mode = str(config.get("pairwise_mode", "rank_sigmoid")).lower()
    pairwise_temp = float(config.get("pairwise_temp", 0.5))

    seed = int(config.get("seed", 42))
    hard_eval_every = int(config.get("hard_eval_every", max(1, epochs // 10)))
    sampler = (config.get("sampling") or config.get("sampler") or "soft").lower()
    logit_scale = float(config.get("logit_scale", 8.0))
    center_logits = bool(config.get("center_logits", True))
    hard_train_outputs = bool(config.get("hard_train_outputs", sampler != "soft"))
    order_decode_weight = float(config.get("order_decode_weight", 0.25))
    post_cfg_raw = config.get("postprocess", {})
    post_cfg = dict(post_cfg_raw) if isinstance(post_cfg_raw, Mapping) else {}
    post_enabled = bool(post_cfg.get("enabled", config.get("lssp_postprocess_enabled", False)))
    post_during_eval = bool(post_cfg.get("during_eval", config.get("lssp_postprocess_during_eval", False)))
    post_eval_mode = str(post_cfg.get("eval_mode", config.get("lssp_postprocess_eval", "taskgraph"))).lower()
    post_max_iters = int(post_cfg.get("max_iters", config.get("lssp_postprocess_max_iters", 64)))
    post_enable_area_fill = bool(post_cfg.get("enable_area_fill", config.get("lssp_postprocess_area_fill", True)))
    post_fill_allow_worsen = float(post_cfg.get("fill_allow_worsen", config.get("lssp_postprocess_fill_allow_worsen", 0.0)))
    post_enable_swap = bool(post_cfg.get("enable_swap", config.get("lssp_postprocess_enable_swap", True)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "DiffGNNOrder training: sampler=%s epochs=%d lr=%.2e tau=%.2f->%.2f order_tau=%.2f->%.2f sinkhorn=%d gumbel=%s alpha=%.2f pairwise_mode=%s pairwise_temp=%.2f",
        sampler,
        epochs,
        lr,
        tau_start,
        tau_final,
        order_tau_start,
        order_tau_final,
        sinkhorn_iters,
        str(gumbel_noise),
        resource_logit_alpha,
        pairwise_mode,
        pairwise_temp,
    )
    if post_enabled:
        logger.info(
            "DiffGNNOrder postprocess enabled: eval_mode=%s during_eval=%s max_iters=%d area_fill=%s fill_allow_worsen=%.3f swap=%s",
            post_eval_mode,
            str(post_during_eval),
            post_max_iters,
            str(post_enable_area_fill),
            post_fill_allow_worsen,
            str(post_enable_swap),
        )

    best_sched_cost = float("inf")
    best_assign = None
    best_probs = None

    tau = tau_start
    order_tau = order_tau_start

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits2, prio_hw, prio_sw = model(data.x, data.edge_index)
        _, probs = _relaxed_binary_assignment(
            logits2,
            temperature=tau,
            hard=hard_train_outputs,
            sampler=sampler,
            logit_scale=logit_scale,
            center_logits=center_logits,
        )

        loss, info = _differentiable_makespan_loss_with_order(
            TG,
            probs,
            prio_hw,
            prio_sw,
            node_list,
            beta_softmax=beta_softmax,
            area_penalty_coeff=area_penalty_coeff,
            entropy_coeff=entropy_coeff,
            usage_balance_coeff=usage_balance_coeff,
            target_hw_frac=target_hw_frac,
            partition_cost_coeff=partition_cost_coeff,
            order_tau=order_tau,
            sinkhorn_iters=sinkhorn_iters,
            gumbel_noise=gumbel_noise,
            gumbel_scale=gumbel_scale,
            resource_logit_alpha=resource_logit_alpha,
            order_refine_steps=order_refine_steps,
            perm_reg_coeff=perm_reg_coeff,
            perm_entropy_coeff=perm_entropy_coeff,
            pairwise_mode=pairwise_mode,
            pairwise_temp=pairwise_temp,
        )

        loss.backward()
        optimizer.step()

        tau = max(tau_final, tau_start - (ep / max(1, epochs)) * (tau_start - tau_final))
        order_tau = max(order_tau_final, order_tau_start - (ep / max(1, epochs)) * (order_tau_start - order_tau_final))

        if (ep % hard_eval_every == 0) or (ep == epochs):
            model.eval()
            with torch.no_grad():
                logits2_eval, prio_hw_eval, prio_sw_eval = model(data.x, data.edge_index)
                _, hard_probs_t = _relaxed_binary_assignment(
                    logits2_eval,
                    temperature=max(tau, 1e-6),
                    hard=True,
                    sampler=sampler,
                    logit_scale=logit_scale,
                    center_logits=center_logits,
                )

                hard_probs = hard_probs_t.cpu().numpy().astype(float)
                decode_scores = _order_aware_repair_scores(
                    hard_probs=hard_probs,
                    prio_hw=prio_hw_eval.detach().cpu().numpy().astype(float),
                    prio_sw=prio_sw_eval.detach().cpu().numpy().astype(float),
                    weight=order_decode_weight,
                )
                hard_probs_repaired = _repair_candidate(TG, hard_probs, node_list, prefer_by_score=decode_scores)
                solution = {node_list[i]: int(1 if hard_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
                if post_enabled and post_during_eval:
                    solution, _ = improve_with_lssp_local_search(
                        TG,
                        solution,
                        max_iters=post_max_iters,
                        eval_mode=post_eval_mode,
                        enable_area_fill=post_enable_area_fill,
                        fill_allow_worsen=post_fill_allow_worsen,
                        enable_swap=post_enable_swap,
                    )

                violation = TG.violates(solution)
                if violation:
                    current_sched_cost = TG.violation_cost
                else:
                    try:
                        current_sched_cost = float(TG.evaluate_makespan(solution)["makespan"])
                    except Exception as e:
                        logger.warning("evaluate_makespan failed during training eval: %s", str(e))
                        current_sched_cost = float("inf")

                if current_sched_cost < best_sched_cost:
                    best_sched_cost = current_sched_cost
                    best_assign = solution.copy()
                    best_probs = hard_probs.copy()

        if ep % max(1, epochs // 10) == 0 or ep <= 5:
            logger.info(
                "Epoch %d/%d loss=%.6f surrogate=%.6f area_frac=%.4f area_pen=%.3f perm_reg=%.4f perm_H=%.4f sched_best=%.6f",
                ep,
                epochs,
                info["loss"],
                info["makespan_surrogate"],
                info["area_frac"],
                info["area_penalty"],
                info["perm_reg"],
                info["perm_entropy"],
                best_sched_cost,
            )

    model.eval()
    with torch.no_grad():
        logits2, prio_hw, prio_sw = model(data.x, data.edge_index)
        _, hard_probs_t = _relaxed_binary_assignment(
            logits2,
            temperature=0.1,
            hard=True,
            sampler=sampler,
            logit_scale=logit_scale,
            center_logits=center_logits,
        )
        final_probs = hard_probs_t.cpu().numpy().astype(float)
        final_decode_scores = _order_aware_repair_scores(
            hard_probs=final_probs,
            prio_hw=prio_hw.detach().cpu().numpy().astype(float),
            prio_sw=prio_sw.detach().cpu().numpy().astype(float),
            weight=order_decode_weight,
        )
        final_probs_repaired = _repair_candidate(TG, final_probs, node_list, prefer_by_score=final_decode_scores)
        final_solution = {node_list[i]: int(1 if final_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
        if post_enabled:
            final_solution, post_info = improve_with_lssp_local_search(
                TG,
                final_solution,
                max_iters=post_max_iters,
                eval_mode=post_eval_mode,
                enable_area_fill=post_enable_area_fill,
                fill_allow_worsen=post_fill_allow_worsen,
                enable_swap=post_enable_swap,
            )
            logger.info(
                "DiffGNNOrder final postprocess: improved=%s cost=%.3f hw_area=%.3f/%.3f (%s)",
                str(post_info["improved"]),
                post_info["cost"],
                post_info["hw_area"],
                post_info["budget"],
                post_info["eval_mode"],
            )

        violation = TG.violates(final_solution)
        if violation:
            final_sched_cost = TG.violation_cost
        else:
            try:
                final_sched_cost = float(TG.evaluate_makespan(final_solution)["makespan"])
            except Exception as e:
                logger.warning("evaluate_makespan failed at final eval: %s", str(e))
                final_sched_cost = float("inf")

    if best_assign is None or final_sched_cost < best_sched_cost:
        best_assign = final_solution
        best_probs = final_probs
        best_sched_cost = final_sched_cost

    logger.info("DiffGNNOrder training finished. Best queue makespan: %.6f", best_sched_cost)
    return {
        "best_assign": best_assign,
        "best_probs": np.asarray(best_probs),
        "best_mip_cost": float(best_sched_cost),
        "model": model,
    }


def optimize_diff_gnn_order(TG, config=None, device="cpu"):
    """
    Public entry-point for differentiable GNN optimization with ordering.
    """
    if config is None:
        config = {}

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    if deterministic:
        _enable_determinism(seed)
        logger.info("Determinism enabled (seed=%d).", seed)
    else:
        _set_global_seeds(seed)

    device = torch.device(device)
    data, node_list = _build_torchgeo_data(TG)
    data = data.to(device)

    num_layers = int(config.get("num_layers", 3))
    dropout = float(config.get("dropout", 0.5))
    hidden_dim = int(config.get("hidden_dim", 256))
    model_name = str(config.get("model", config.get("model_name", "default"))).lower()
    model_extra_cfg = {
        k: v
        for k, v in config.items()
        if k not in {"model", "model_name", "hidden_dim", "num_layers", "dropout"}
    }

    model = build_order_model(
        model_name=model_name,
        in_channels=data.num_node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **model_extra_cfg,
    )

    logger.info("DiffGNNOrder model selected: %s", model_name)
    return _train_with_relaxed_binary_order(TG, model, data, node_list, config, device)


def simulate_diff_GNN_order(dim, func_to_optimize, config):
    """
    MethodRegistry-compatible wrapper for ordered diff-GNN.

    Returns:
      (best_cost, best_solution_array)
    """
    logger.info("Starting simulate_diff_GNN_order")

    TG = getattr(func_to_optimize, "__self__", None)
    if TG is None:
        msg = "func_to_optimize must be a bound TaskGraph method so the graph can be accessed."
        logger.error(msg)
        raise ValueError(msg)

    diff_cfg = dict(config.get("diffgnn_order", {}))
    if not diff_cfg:
        # Fallback to diffgnn block for convenience.
        diff_cfg = dict(config.get("diffgnn", {}))

    fast_mode = bool(diff_cfg.get("fast_mode", False))
    if fast_mode:
        for key, value in _FAST_MODE_DEFAULTS.items():
            diff_cfg.setdefault(key, value)
        logger.info("diff_gnn_order fast_mode enabled. Applied speed defaults for unset keys.")

    if "iter" not in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["iter"] = 1500
    if "verbose" not in diff_cfg:
        diff_cfg["verbose"] = 1500
    if "hidden_dim" not in diff_cfg:
        diff_cfg["hidden_dim"] = 256
    if "num_layers" not in diff_cfg:
        diff_cfg["num_layers"] = 3
    if "dropout" not in diff_cfg:
        diff_cfg["dropout"] = 0.5
    if "model" not in diff_cfg and "model_name" not in diff_cfg:
        diff_cfg["model"] = "default"

    if "iter" in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["epochs"] = diff_cfg["iter"]

    # Ordering defaults
    diff_cfg.setdefault("order_tau_start", 1.0)
    diff_cfg.setdefault("order_tau_final", 0.2)
    diff_cfg.setdefault("sinkhorn_iters", 20)
    diff_cfg.setdefault("gumbel_noise", True)
    diff_cfg.setdefault("gumbel_scale", 1.0)
    diff_cfg.setdefault("resource_logit_alpha", 2.0)
    diff_cfg.setdefault("order_refine_steps", 2)
    diff_cfg.setdefault("perm_reg_coeff", 1e-2)
    diff_cfg.setdefault("perm_entropy_coeff", 1e-3)
    diff_cfg.setdefault("pairwise_mode", "rank_sigmoid")
    diff_cfg.setdefault("pairwise_temp", 0.5)

    if "seed" not in diff_cfg and "seed" in config:
        diff_cfg["seed"] = config.get("seed")

    if "device" in diff_cfg:
        config["device"] = diff_cfg.get("device")
    elif "device" not in config:
        config["device"] = "gpu"

    device = get_device(config)
    logger.info("Using device: %s", device)
    print(f"[diff_gnn_order] device={device}")

    result = optimize_diff_gnn_order(TG, config=diff_cfg, device=device)
    best_assign = result.get("best_assign", {})

    sol_arr = np.zeros(len(TG.graph.nodes()), dtype=np.float64)
    for node, idx in TG.node_to_num.items():
        sol_arr[idx] = float(best_assign.get(node, 0))

    node_list = list(TG.graph.nodes())
    sol_arr = _repair_candidate(TG, sol_arr, node_list, prefer_by_score=sol_arr)

    assert sol_arr.shape[0] == dim, f"dim ({dim}) != number of nodes ({sol_arr.shape[0]})"

    solution = {node_list[i]: int(sol_arr[i] > 0.5) for i in range(len(node_list))}
    if TG.violates(solution):
        eval_cost = float(TG.violation_cost)
    else:
        eval_cost = float(TG.evaluate_makespan(solution)["makespan"])

    best_cost = float(result.get("best_mip_cost", eval_cost))
    if not math.isfinite(best_cost):
        best_cost = eval_cost

    logger.info(
        "simulate_diff_GNN_order finished: best_cost=%.6f (eval_cost=%.6f, train_best_cost=%.6f)",
        best_cost,
        eval_cost,
        result.get("best_mip_cost", float("nan")),
    )
    return best_cost, sol_arr
