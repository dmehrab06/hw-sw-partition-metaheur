import math
import os
import random
import sys
import time
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
        _dls_refine_probs,
        _enable_determinism,
        _evaluate_discrete_solution,
        _fill_hw_area_by_score,
        _relaxed_binary_assignment,
        _repair_candidate,
        _set_global_seeds,
        get_device,
    )
except Exception:
    from diff_gnn_utils_schedule import (  # type: ignore
        _build_torchgeo_data,
        _dls_refine_probs,
        _enable_determinism,
        _evaluate_discrete_solution,
        _fill_hw_area_by_score,
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
    "iter": 250,
    "verbose": 250,
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.2,
    "sinkhorn_iters": 8,
    "order_refine_steps": 1,
    "hard_eval_every": 50,
    "gumbel_noise": False,
    "gumbel_scale": 0.0,
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
    paper_sigma = float(config.get("paper_sigma", 0.0))
    paper_sigma = max(0.0, min(1.0, paper_sigma))

    seed = int(config.get("seed", 42))
    hard_eval_every = int(config.get("hard_eval_every", max(1, epochs // 5)))
    selection_metric_train = str(config.get("selection_metric_train", config.get("selection_metric", "queue"))).lower()
    selection_metric_final = str(config.get("selection_metric_final", selection_metric_train)).lower()
    sampler = (config.get("sampling") or config.get("sampler") or "soft").lower()
    logit_scale = float(config.get("logit_scale", 8.0))
    center_logits = bool(config.get("center_logits", True))
    hard_train_outputs = bool(config.get("hard_train_outputs", sampler != "soft"))
    order_decode_weight = float(config.get("order_decode_weight", 0.25))
    post_cfg_raw = config.get("postprocess", {})
    post_cfg = dict(post_cfg_raw) if isinstance(post_cfg_raw, Mapping) else {}
    post_enabled = bool(post_cfg.get("enabled", config.get("lssp_postprocess_enabled", False)))
    post_mode = str(post_cfg.get("mode", config.get("postprocess_mode", "none"))).lower()
    if post_mode == "none" and post_enabled:
        post_mode = "lssp"
    if post_mode not in {"none", "dls", "lssp", "hybrid"}:
        raise ValueError(f"Unsupported postprocess_mode '{post_mode}'. Use none|dls|lssp|hybrid.")
    post_during_train = bool(post_cfg.get("during_train", config.get("postprocess_during_train", False)))
    post_during_eval = bool(post_cfg.get("during_eval", config.get("lssp_postprocess_during_eval", False)))
    use_dls_train = post_mode in {"dls", "hybrid"} and post_during_train
    use_dls_final = post_mode in {"dls", "hybrid"}
    use_lssp_eval = post_mode in {"lssp", "hybrid"} and post_during_train and post_during_eval
    use_lssp_final = post_mode in {"lssp", "hybrid"}
    post_eval_mode = str(post_cfg.get("eval_mode", config.get("lssp_postprocess_eval", "taskgraph"))).lower()
    post_max_iters = int(post_cfg.get("max_iters", config.get("lssp_postprocess_max_iters", 64)))
    post_enable_area_fill = bool(post_cfg.get("enable_area_fill", config.get("lssp_postprocess_area_fill", True)))
    post_fill_allow_worsen = float(post_cfg.get("fill_allow_worsen", config.get("lssp_postprocess_fill_allow_worsen", 0.0)))
    post_enable_swap = bool(post_cfg.get("enable_swap", config.get("lssp_postprocess_enable_swap", True)))
    dls_steps = int(post_cfg.get("dls_steps", config.get("dls_steps", 2 if use_dls_final else 0)))
    dls_flip_eta = float(post_cfg.get("dls_flip_eta", config.get("dls_flip_eta", 0.35)))
    dls_swap_eta = float(post_cfg.get("dls_swap_eta", config.get("dls_swap_eta", 0.18)))
    dls_score_temp = float(post_cfg.get("dls_score_temp", config.get("dls_score_temp", 0.7)))
    dls_comm_coeff = float(post_cfg.get("dls_comm_coeff", config.get("dls_comm_coeff", 0.02)))
    dls_area_proj_iters = int(post_cfg.get("dls_area_proj_iters", config.get("dls_area_proj_iters", 4)))
    dls_area_proj_strength = float(post_cfg.get("dls_area_proj_strength", config.get("dls_area_proj_strength", 6.0)))
    dls_fill_decode = bool(post_cfg.get("dls_fill_decode", config.get("dls_fill_decode", True)))
    dls_lssp_like = bool(post_cfg.get("dls_lssp_like", config.get("dls_lssp_like", False)))
    dls_lssp_pri_coeff = float(post_cfg.get("dls_lssp_pri_coeff", config.get("dls_lssp_pri_coeff", 0.35)))
    dls_lssp_beta = float(post_cfg.get("dls_lssp_beta", config.get("dls_lssp_beta", 8.0)))
    dls_lssp_fill_eta = float(post_cfg.get("dls_lssp_fill_eta", config.get("dls_lssp_fill_eta", 0.20)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "DiffGNNOrder training: sampler=%s epochs=%d lr=%.2e tau=%.2f->%.2f order_tau=%.2f->%.2f sinkhorn=%d gumbel=%s alpha=%.2f pairwise_mode=%s pairwise_temp=%.2f post_mode=%s post_during_train=%s feature_profile=%s edge_weight_mode=%s paper_sigma=%.2f",
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
        post_mode,
        str(post_during_train),
        str(config.get("feature_profile", "default")),
        str(config.get("edge_weight_mode", "auto")),
        paper_sigma,
    )
    logger.info(
        "DiffGNNOrder metrics: train_metric=%s final_metric=%s",
        selection_metric_train,
        selection_metric_final,
    )
    edge_weight_learner = str(
        config.get(
            "edge_weight_learner",
            "per_edge" if bool(config.get("learn_edge_weight", False)) else "none",
        )
    ).lower()
    if edge_weight_learner != "none":
        logger.info(
            "DiffGNNOrder learned edge weight enabled: learner=%s min_scale=%.3f max_scale=%.3f num_edges=%d edge_attr_dim=%d",
            edge_weight_learner,
            float(config.get("edge_weight_min_scale", 0.5)),
            float(config.get("edge_weight_max_scale", 1.5)),
            int(data.edge_index.shape[1]),
            int(getattr(data, "edge_attr", torch.zeros((0, 0))).shape[1]),
        )
    if use_dls_train:
        logger.info(
            "DiffGNNOrder DLS enabled during training: steps=%d flip_eta=%.2f swap_eta=%.2f temp=%.2f comm_coeff=%.3f area_proj_iters=%d area_proj_strength=%.2f lssp_like=%s pri_coeff=%.2f pri_beta=%.2f fill_eta=%.2f",
            dls_steps,
            dls_flip_eta,
            dls_swap_eta,
            dls_score_temp,
            dls_comm_coeff,
            dls_area_proj_iters,
            dls_area_proj_strength,
            str(dls_lssp_like),
            dls_lssp_pri_coeff,
            dls_lssp_beta,
            dls_lssp_fill_eta,
        )
    if use_lssp_final:
        logger.info(
            "DiffGNNOrder final postprocess enabled: eval_mode=%s during_train=%s during_eval=%s max_iters=%d area_fill=%s fill_allow_worsen=%.3f swap=%s",
            post_eval_mode,
            str(post_during_train),
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
    edge_weight = getattr(data, "edge_weight", None)
    edge_attr = getattr(data, "edge_attr", None)
    paper_hgp = getattr(data, "paper_hgp", None)

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits2, prio_hw, prio_sw = model(
            data.x,
            data.edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
        )
        _, probs = _relaxed_binary_assignment(
            logits2,
            temperature=tau,
            hard=hard_train_outputs,
            sampler=sampler,
            logit_scale=logit_scale,
            center_logits=center_logits,
        )
        if paper_sigma > 0 and paper_hgp is not None:
            probs = (1.0 - paper_sigma) * probs + paper_sigma * paper_hgp.to(probs.device, probs.dtype)
            probs = probs.clamp(0.0, 1.0)
        if use_dls_train and dls_steps > 0:
            probs = _dls_refine_probs(
                TG,
                probs,
                node_list=node_list,
                steps=dls_steps,
                flip_eta=dls_flip_eta,
                swap_eta=dls_swap_eta,
                score_temp=dls_score_temp,
                comm_coeff=dls_comm_coeff,
                area_proj_iters=dls_area_proj_iters,
                area_proj_strength=dls_area_proj_strength,
                lssp_like=dls_lssp_like,
                lssp_pri_coeff=dls_lssp_pri_coeff,
                lssp_beta=dls_lssp_beta,
                lssp_fill_eta=dls_lssp_fill_eta,
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
                logits2_eval, prio_hw_eval, prio_sw_eval = model(
                    data.x,
                    data.edge_index,
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                )
                _, hard_probs_t = _relaxed_binary_assignment(
                    logits2_eval,
                    temperature=max(tau, 1e-6),
                    hard=True,
                    sampler=sampler,
                    logit_scale=logit_scale,
                    center_logits=center_logits,
                )
                if paper_sigma > 0 and paper_hgp is not None:
                    hard_probs_t = (1.0 - paper_sigma) * hard_probs_t + paper_sigma * paper_hgp.to(hard_probs_t.device, hard_probs_t.dtype)
                    hard_probs_t = hard_probs_t.clamp(0.0, 1.0)

                hard_probs = hard_probs_t.cpu().numpy().astype(float)
                if use_dls_train and dls_steps > 0:
                    hard_probs_t_refined = _dls_refine_probs(
                        TG,
                        hard_probs_t.float(),
                        node_list=node_list,
                        steps=dls_steps,
                        flip_eta=dls_flip_eta,
                        swap_eta=dls_swap_eta,
                        score_temp=dls_score_temp,
                        comm_coeff=dls_comm_coeff,
                        area_proj_iters=dls_area_proj_iters,
                        area_proj_strength=dls_area_proj_strength,
                        lssp_like=dls_lssp_like,
                        lssp_pri_coeff=dls_lssp_pri_coeff,
                        lssp_beta=dls_lssp_beta,
                        lssp_fill_eta=dls_lssp_fill_eta,
                    )
                    hard_probs = hard_probs_t_refined.detach().cpu().numpy().astype(float)
                decode_scores = _order_aware_repair_scores(
                    hard_probs=hard_probs,
                    prio_hw=prio_hw_eval.detach().cpu().numpy().astype(float),
                    prio_sw=prio_sw_eval.detach().cpu().numpy().astype(float),
                    weight=order_decode_weight,
                )
                hard_probs_repaired = _repair_candidate(TG, hard_probs, node_list, prefer_by_score=decode_scores)
                if use_dls_train and dls_fill_decode:
                    hard_probs_repaired = _fill_hw_area_by_score(
                        TG,
                        hard_probs_repaired,
                        node_list=node_list,
                        prefer_by_score=decode_scores,
                    )
                solution = {node_list[i]: int(1 if hard_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
                if use_lssp_eval:
                    solution, _ = improve_with_lssp_local_search(
                        TG,
                        solution,
                        max_iters=post_max_iters,
                        eval_mode=post_eval_mode,
                        enable_area_fill=post_enable_area_fill,
                        fill_allow_worsen=post_fill_allow_worsen,
                        enable_swap=post_enable_swap,
                    )

                current_sched_cost = _evaluate_discrete_solution(
                    TG,
                    solution,
                    metric=selection_metric_train,
                )

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

    logger.info("DiffGNNOrder final decode started.")
    model.eval()
    with torch.no_grad():
        logits2, prio_hw, prio_sw = model(
            data.x,
            data.edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
        )
        _, hard_probs_t = _relaxed_binary_assignment(
            logits2,
            temperature=0.1,
            hard=True,
            sampler=sampler,
            logit_scale=logit_scale,
            center_logits=center_logits,
        )
        if paper_sigma > 0 and paper_hgp is not None:
            hard_probs_t = (1.0 - paper_sigma) * hard_probs_t + paper_sigma * paper_hgp.to(hard_probs_t.device, hard_probs_t.dtype)
            hard_probs_t = hard_probs_t.clamp(0.0, 1.0)
        final_probs = hard_probs_t.cpu().numpy().astype(float)
        if use_dls_final and dls_steps > 0:
            final_probs_t_refined = _dls_refine_probs(
                TG,
                hard_probs_t.float(),
                node_list=node_list,
                steps=dls_steps,
                flip_eta=dls_flip_eta,
                swap_eta=dls_swap_eta,
                score_temp=dls_score_temp,
                comm_coeff=dls_comm_coeff,
                area_proj_iters=dls_area_proj_iters,
                area_proj_strength=dls_area_proj_strength,
                lssp_like=dls_lssp_like,
                lssp_pri_coeff=dls_lssp_pri_coeff,
                lssp_beta=dls_lssp_beta,
                lssp_fill_eta=dls_lssp_fill_eta,
            )
            final_probs = final_probs_t_refined.detach().cpu().numpy().astype(float)
        final_decode_scores = _order_aware_repair_scores(
            hard_probs=final_probs,
            prio_hw=prio_hw.detach().cpu().numpy().astype(float),
            prio_sw=prio_sw.detach().cpu().numpy().astype(float),
            weight=order_decode_weight,
        )
        final_probs_repaired = _repair_candidate(TG, final_probs, node_list, prefer_by_score=final_decode_scores)
        if use_dls_final and dls_fill_decode:
            final_probs_repaired = _fill_hw_area_by_score(
                TG,
                final_probs_repaired,
                node_list=node_list,
                prefer_by_score=final_decode_scores,
            )
        final_solution = {node_list[i]: int(1 if final_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
        if use_lssp_final:
            post_t0 = time.perf_counter()
            logger.info(
                "DiffGNNOrder final postprocess started: mode=%s eval_mode=%s max_iters=%d swap=%s",
                post_mode,
                post_eval_mode,
                post_max_iters,
                str(post_enable_swap),
            )
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
            logger.info("DiffGNNOrder final postprocess elapsed: %.3fs", time.perf_counter() - post_t0)

        final_sched_cost_train = _evaluate_discrete_solution(
            TG,
            final_solution,
            metric=selection_metric_train,
        )

    if best_assign is None or final_sched_cost_train < best_sched_cost:
        best_assign = final_solution
        best_probs = final_probs
        best_sched_cost = final_sched_cost_train

    if selection_metric_final == selection_metric_train:
        best_final_cost = float(best_sched_cost)
    else:
        metric_t0 = time.perf_counter()
        logger.info(
            "DiffGNNOrder final metric eval started: metric=%s train_metric=%s",
            selection_metric_final,
            selection_metric_train,
        )
        best_final_cost = _evaluate_discrete_solution(
            TG,
            best_assign,
            metric=selection_metric_final,
        )
        logger.info("DiffGNNOrder final metric eval elapsed: %.3fs", time.perf_counter() - metric_t0)

    logger.info(
        "DiffGNNOrder training finished. Best %s makespan: %.6f; selected assignment %s makespan: %.6f",
        selection_metric_train,
        best_sched_cost,
        selection_metric_final,
        best_final_cost,
    )
    return {
        "best_assign": best_assign,
        "best_probs": np.asarray(best_probs),
        "best_mip_cost": float(best_final_cost),
        "best_train_cost": float(best_sched_cost),
        "selection_metric_train": selection_metric_train,
        "selection_metric_final": selection_metric_final,
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
    data, node_list = _build_torchgeo_data(TG, config=config)
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
    if model_name in {"mpnns", "mpnn"} and bool(config.get("mpnns_edge_aware", False)):
        model_extra_cfg.setdefault(
            "mpnns_edge_attr_dim",
            int(getattr(data, "edge_attr", torch.zeros((0, 0))).shape[1]),
        )
    edge_weight_learner = str(
        config.get(
            "edge_weight_learner",
            "per_edge" if bool(config.get("learn_edge_weight", False)) else "none",
        )
    ).lower()
    if edge_weight_learner != "none":
        model_extra_cfg.setdefault("edge_weight_learner", edge_weight_learner)
        model_extra_cfg.setdefault("num_edges", int(data.edge_index.shape[1]))
        model_extra_cfg.setdefault("edge_attr_dim", int(getattr(data, "edge_attr", torch.zeros((0, 0))).shape[1]))

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

    fast_mode = bool(diff_cfg.get("fast_mode", True))
    if fast_mode:
        for key, value in _FAST_MODE_DEFAULTS.items():
            diff_cfg.setdefault(key, value)
        logger.info("diff_gnn_order fast_mode enabled. Applied speed defaults for unset keys.")

    if "iter" not in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["iter"] = 250
    if "verbose" not in diff_cfg:
        diff_cfg["verbose"] = 250
    if "hidden_dim" not in diff_cfg:
        diff_cfg["hidden_dim"] = 64
    if "num_layers" not in diff_cfg:
        diff_cfg["num_layers"] = 3
    if "dropout" not in diff_cfg:
        diff_cfg["dropout"] = 0.2
    if "model" not in diff_cfg and "model_name" not in diff_cfg:
        diff_cfg["model"] = "default"

    if "iter" in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["epochs"] = diff_cfg["iter"]
    if "learn_edge_weight" not in diff_cfg and "learned_edge_weight" in diff_cfg:
        diff_cfg["learn_edge_weight"] = bool(diff_cfg.get("learned_edge_weight"))
    if "edge_weight_learner" not in diff_cfg and bool(diff_cfg.get("learn_edge_weight", False)):
        diff_cfg["edge_weight_learner"] = "per_edge"
    epochs = int(diff_cfg.get("epochs", 250))

    optimize_name = str(getattr(func_to_optimize, "__name__", "") or "").lower()
    uses_mip_blackbox = optimize_name.endswith("_mip") or "_mip" in optimize_name
    train_metric = str(diff_cfg.get("selection_metric_train", diff_cfg.get("selection_metric", "queue"))).lower()
    if "selection_metric_train" not in diff_cfg:
        diff_cfg["selection_metric_train"] = train_metric
    if "selection_metric" not in diff_cfg:
        diff_cfg["selection_metric"] = train_metric
    if "selection_metric_final" not in diff_cfg:
        if bool(diff_cfg.get("final_legacy_lp_if_mip", True)) and uses_mip_blackbox:
            diff_cfg["selection_metric_final"] = "legacy_lp"
        else:
            diff_cfg["selection_metric_final"] = train_metric

    # Speed patch defaults for ordering path (applies even when fast_mode is
    # explicitly disabled, unless the user already overrides each knob).
    if bool(diff_cfg.get("speed_patch", True)):
        diff_cfg.setdefault("sinkhorn_iters", 8)
        diff_cfg.setdefault("order_refine_steps", 1)
        diff_cfg.setdefault("gumbel_noise", False)
        diff_cfg.setdefault("gumbel_scale", 0.0)

    speed_patch_enabled = bool(diff_cfg.get("speed_patch", True))
    default_hard_eval_every = max(1, epochs // 5)
    if "hard_eval_every" not in diff_cfg:
        diff_cfg["hard_eval_every"] = default_hard_eval_every
    elif speed_patch_enabled:
        diff_cfg["hard_eval_every"] = max(int(diff_cfg["hard_eval_every"]), default_hard_eval_every)

    post_cfg_raw = diff_cfg.get("postprocess", {})
    post_cfg = dict(post_cfg_raw) if isinstance(post_cfg_raw, Mapping) else {}
    post_cfg.setdefault("mode", "none")
    post_cfg.setdefault("during_train", False)
    diff_cfg["postprocess"] = post_cfg

    # Lightweight defaults when users keep configs minimal.
    if not any(
        k in diff_cfg
        for k in (
            "entropy_coeff",
            "usage_balance_coeff",
            "partition_cost_coeff",
            "perm_reg_coeff",
            "perm_entropy_coeff",
            "target_hw_frac",
        )
    ):
        diff_cfg["entropy_coeff"] = 0.0
        diff_cfg["usage_balance_coeff"] = 0.0
        diff_cfg["partition_cost_coeff"] = 1e-2
        diff_cfg["perm_reg_coeff"] = 0.0
        diff_cfg["perm_entropy_coeff"] = 0.0

    # Ordering defaults
    diff_cfg.setdefault("order_tau_start", 1.0)
    diff_cfg.setdefault("order_tau_final", 0.2)
    diff_cfg.setdefault("sinkhorn_iters", 8)
    diff_cfg.setdefault("gumbel_noise", False)
    diff_cfg.setdefault("gumbel_scale", 0.0)
    diff_cfg.setdefault("resource_logit_alpha", 2.0)
    diff_cfg.setdefault("order_refine_steps", 1)
    diff_cfg.setdefault("perm_reg_coeff", 1e-2)
    diff_cfg.setdefault("perm_entropy_coeff", 1e-3)
    diff_cfg.setdefault("pairwise_mode", "rank_sigmoid")
    diff_cfg.setdefault("pairwise_temp", 0.7)

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

    selection_metric = str(
        diff_cfg.get(
            "selection_metric_final",
            diff_cfg.get("selection_metric_train", diff_cfg.get("selection_metric", "queue")),
        )
    ).lower()
    eval_cost = _evaluate_discrete_solution(TG, solution, metric=selection_metric)

    best_cost = float(result.get("best_mip_cost", eval_cost))
    if not math.isfinite(best_cost):
        best_cost = eval_cost

    logger.info(
        "simulate_diff_GNN_order finished: best_cost=%.6f (eval_cost=%.6f, train_best_cost=%.6f, metric=%s, train_metric=%s)",
        best_cost,
        eval_cost,
        result.get("best_train_cost", float("nan")),
        selection_metric,
        result.get("selection_metric_train", "unknown"),
    )
    return best_cost, sol_arr
