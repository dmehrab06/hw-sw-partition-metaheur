from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path

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
        _apply_large_graph_cheap_dag_policy,
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
        _apply_large_graph_cheap_dag_policy,
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
try:
    from .partition_schedule_evaluator import evaluate_partition_lssp
except Exception:
    from partition_schedule_evaluator import evaluate_partition_lssp  # type: ignore

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/diff_gnn_ordering.log")

logger = LogManager.get_logger(__name__)

_MKSPAN_DIFFGNN_ORDER_DEFAULTS = {
    "iter": 2500,
    "verbose": 2500,
    "device": "gpu",
    "hidden_dim": 256, #256,
    "num_layers": 3, #3,
    "dropout": 0.5,
    "model": "default",
    "speed_patch": True,
    "hard_eval_every": 100,
    "hard_eval_only_final": True,
    "checkpoint_eval_when_final_only": False,
    "fast_mode": True,
    "feature_profile": "default_plus_paper",
    "edge_weight_mode": "paper2_cosine",
    "edge_weight_learner": "mlp",
    "edge_mlp_hidden_dim": 32,
    "edge_weight_min_scale": 0.5,
    "edge_weight_max_scale": 1.5,
    "sinkhorn_iters": 2,
    "large_graph_order_approx_enabled": True,
    "large_graph_order_approx_threshold": 900,
    "large_graph_order_topk": 64,
    "large_graph_order_candidate_mode": "rank_window",
    "large_graph_order_candidate_pool_factor": 2.0,
    "order_refine_steps": 2,
    "use_hw_ordering": False,
    "gumbel_noise": False,
    "gumbel_scale": 0.0,
    "pairwise_mode": "rank_sigmoid",
    "pairwise_temp": 0.30,
    "soft_makespan_mode": "jacobi",
    "jacobi_iters": 10,
    "soft_makespan_exact_mode": "sequential",
    "soft_makespan_exact_every": 2, #5 okay for mobile
    "soft_makespan_exact_first_epoch": False,
    "resource_candidate_topk": 128,
    "resource_candidate_min_prob": 1e-5,
    "order_decode_weight": 0.55,
    "entropy_coeff": 0.0,
    "usage_balance_coeff": 0.0,
    "partition_cost_coeff": 0.0833333333,
    "perm_reg_coeff": 0.0,
    "perm_entropy_coeff": 0.0,
    # Candidate-selection metric used by diff_gnn decode / train-time hard eval.
    # Options:
    #   "queue"     : fastest; TaskGraph FIFO/event-queue simulator.
    #   "legacy_lp" : DAG/LP-style evaluator (also accepts "legacy", "lp", "dag_lp", "cvxpy").
    # Note:
    #   "lssp" is NOT selected here. Full LSSP is controlled by postprocess["eval_mode"] below.
    "selection_metric": "queue",        # fallback metric if train/final keys are not set.
    "selection_metric_train": "queue",  # metric used while training / ranking decode candidates.
    "selection_metric_final": "queue",  # metric used for final reported selection.
    "final_legacy_lp_if_mip": True,     # when optimizing against a MIP blackbox, final metric can auto-switch to legacy_lp.
    "early_stop_enabled": True,
    "early_stop_min_epochs": 2000,
    "early_stop_patience": 10,
    "early_stop_min_delta": 1e-4,
    "progress_log_every": 50,
}

_MKSPAN_POSTPROCESS_DEFAULTS = {
    # Postprocess mode after decode.
    # Options:
    #   "none"   : disable all postprocess.
    #   "dls"    : differentiable local search only.
    #   "lssp"   : discrete local search only.
    #   "hybrid" : run DLS refinement and then LSSP local search.
    "mode": "hybrid",
    "during_train": False,  # if True, also run postprocess during train-time hard-eval/checkpoints.
    # Cost model used inside the discrete LSSP local-search accept/reject loop.
    # Options:
    #   "lssp"     : slower, more faithful final scheduler with serialized bus + priorities.
    #   "taskgraph": faster, older queue-style TaskGraph evaluator.
    "eval_mode": "lssp",
    "use_dual_lssp_postprocess": False,  # if True, run postprocess with both static and learned SW-order LSSP and keep the better result.
    "max_iters": 120,                    # max local-search iterations in the LSSP stage.
    "adaptive_max_iters": False,         # if True, cap iterations on large graphs.
    "adaptive_large_n": 128,             # graph-size threshold for adaptive cap.
    "adaptive_large_cap": 10,            # max_iters cap applied once graph is large.
    "enable_area_fill": True,            # greedy SW->HW fill step before local search.
    "fill_allow_worsen": 0.0,            # allow small temporary cost increase during area fill.
    "enable_swap": True,                 # allow HW<->SW pair swaps in stage 2.
    # Neighborhood selection:
    #   "critical" : search near critical/cut nodes only (faster).
    #   "all"      : scan all nodes (slower, broader).
    "search_strategy": "critical",
    "candidate_top_k": 128,              # how many ranked nodes to keep for local-search candidates.
    "critical_slack_frac": 0.10,         # larger keeps more near-critical nodes.
    "candidate_include_neighbors": True, # also include graph neighbors of critical nodes.
    "candidate_include_cut_endpoints": True,  # also include endpoints of active cut/comm edges.
    "final_all_decode_candidates": True, # if True, postprocess every decode candidate, not just the current best one.
    "print_progress": True,              # print stage1/stage2 progress logs.
    "print_every": 10,                   # progress log interval.
    # DLS-only knobs used when mode is "dls" or "hybrid".
    "dls_steps": 2,
    "dls_flip_eta": 0.35,                # soft flip step size.
    "dls_swap_eta": 0.18,                # soft swap step size.
    "dls_score_temp": 0.70,              # softmax/sharpen temperature for DLS scoring.
    "dls_comm_coeff": 0.02,              # communication penalty weight in DLS score.
    "dls_area_proj_iters": 4,            # soft area projection refinement steps.
    "dls_area_proj_strength": 6.0,       # strength of the area projection.
    "dls_fill_decode": True,             # greedily fill unused HW area before postprocess candidate evaluation.
}

_FAST_MODE_DEFAULTS = {
    "iter": 500,
    "verbose": 500,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.5,
    "feature_profile": "default_plus_paper",
    "edge_weight_mode": "paper2_cosine",
    "edge_weight_learner": "mlp",
    "edge_mlp_hidden_dim": 32,
    "edge_weight_min_scale": 0.5,
    "edge_weight_max_scale": 1.5,
    "sinkhorn_iters": 10,
    "order_refine_steps": 4,
    "use_hw_ordering": False,
    "hard_eval_only_final": False,
    "checkpoint_eval_when_final_only": False,
    "gumbel_noise": False,
    "gumbel_scale": 0.0,
    "pairwise_mode": "rank_sigmoid",
    "pairwise_temp": 0.30,
    "soft_makespan_mode": "jacobi",
    "jacobi_iters": 4,
    "soft_makespan_exact_mode": "sequential",
    "soft_makespan_exact_every": 10,
    "soft_makespan_exact_first_epoch": True,
    "resource_candidate_topk": 128,
    "resource_candidate_min_prob": 1e-5,
    "order_decode_weight": 0.55,
    "early_stop_enabled": True,
    "early_stop_min_epochs": 250,
    "early_stop_patience": 8,
    "early_stop_min_delta": 5e-4,
    "progress_log_every": 50,
}

# Dataset-specific DiffGNN ordering defaults. These only fill in keys that are
# still unset after reading YAML, so per-config values can override them.
_DIFFGNN_ORDER_DATASET_OVERRIDES = {
    "paper_fig3_11node": {},
    # "mobile_net_tosa": {
    #     "iter": 500,
    #     "verbose": 500,
    #     "early_stop_min_epochs": 500,
    #     "soft_makespan_exact_every": 5,
    #     "large_graph_order_approx_enabled": True,
    #     "large_graph_order_approx_threshold": 800,
    #     "large_graph_order_topk": 64,
    #     "postprocess": {
    #         "candidate_top_k": 64,
    #     },
    # },
    #for runtime
    "mobile_net_tosa": {
        "iter": 1500,
        "verbose": 50,
        "early_stop_min_epochs": 1000,
        "soft_makespan_exact_every": 2,
        "large_graph_order_approx_enabled": True,
        "large_graph_order_approx_threshold": 800,
        "large_graph_order_topk": 64,
        "large_graph_order_candidate_mode": "rank_window",
        "large_graph_order_candidate_pool_factor": 2.0,
        "postprocess": {
            "candidate_top_k": 64,
        },
    },
    "squeezenet_like_1000": {
        "iter": 750,
        "verbose": 750,
        "early_stop_min_epochs": 500,
        "soft_makespan_exact_every": 2,
        "postprocess": {
            "candidate_top_k": 32,
        },
    },
    "squeezenet_like_10000": {
        # "iter": 2500,
        # "verbose": 2500,
        # "early_stop_min_epochs": 2500,

        "iter": 500,
        "verbose": 500,
        "early_stop_min_epochs": 500,

        "soft_makespan_exact_every": 5,
        "postprocess": {
            "candidate_top_k": 16,
        },
    },
        
    "rez_net_tosa": {
        # "iter": 750,
        # "candidate_top_k": 12,
    },
    "squeeze_net_tosa": {
        'iter': 2500,
        'early_stop_min_epochs': 2500,
    },

    # "squeeze_net_tosa": {
    #     'iter': 500,
    #     'early_stop_min_epochs': 500,
    # },

    "anomaly_detection_tosa": {},
    "image_classification_tosa": {},
    "keyword_spotting_tosa": {},
    "visual_wake_words_tosa": {},
    "paper_fig3_11node": {
        "iter": 100
    },
}


def _resolve_diffgnn_order_dataset_name(config):
    if not isinstance(config, Mapping):
        return None

    graph_file = str(config.get("graph-file", "") or "").strip()
    if graph_file:
        return Path(graph_file).stem or None

    taskgraph_pickle = str(config.get("taskgraph-pickle", "") or "").strip()
    if not taskgraph_pickle:
        return None

    stem = Path(taskgraph_pickle).stem
    if stem.startswith("taskgraph-"):
        stem = stem[len("taskgraph-"):]
    if "_area-" in stem:
        stem = stem.split("_area-", 1)[0]
    return stem or None


def _apply_recursive_defaults(target: dict, defaults: Mapping) -> None:
    for key, value in defaults.items():
        if isinstance(value, Mapping):
            current = target.get(key, None)
            if isinstance(current, Mapping):
                merged = dict(current)
                _apply_recursive_defaults(merged, value)
                target[key] = merged
            elif key not in target:
                nested = {}
                _apply_recursive_defaults(nested, value)
                target[key] = nested
            continue
        target.setdefault(key, value)


def _apply_dataset_specific_diffgnn_defaults(diff_cfg: dict, config):
    dataset_name = _resolve_diffgnn_order_dataset_name(config)
    if not dataset_name:
        return None

    overrides = _DIFFGNN_ORDER_DATASET_OVERRIDES.get(dataset_name, None)
    if not isinstance(overrides, Mapping) or not overrides:
        return dataset_name

    _apply_recursive_defaults(diff_cfg, overrides)
    logger.info(
        "Applied dataset-specific diff_gnn_order defaults for %s: %s",
        dataset_name,
        overrides,
    )
    return dataset_name


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


def _topk_pairwise_before_from_priority_logits(
    priorities: torch.Tensor,
    topk: int,
    temperature: float = 0.5,
    candidate_mode: str = "rank_window",
    candidate_pool_factor: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse O(NK) approximation of pairwise-before probabilities.

    Candidate source tasks are selected without materializing a dense NxN
    pairwise matrix. The default `rank_window` mode builds a target-specific
    shortlist around each task's position in the learned priority order, which
    avoids reusing nearly the same global source set for every target.

    Modes:
      - rank_window: immediate higher-ranked neighbors first, then nearby lower-
        ranked neighbors as fallback; pool size is scaled by
        `candidate_pool_factor` before the exact top-k prune.
      - global_topk: legacy behavior that reuses the globally highest-priority
        tasks (excluding self) for every target.
    """
    priorities = priorities.reshape(-1)
    n = int(priorities.shape[0])
    if n <= 1:
        empty_idx = torch.empty((0, n), dtype=torch.long, device=priorities.device)
        empty_prob = priorities.new_empty((0, n))
        return empty_idx, empty_prob

    k_eff = min(max(int(topk), 1), n - 1)
    sorted_idx = torch.argsort(priorities.reshape(-1), descending=True)
    inv_rank = torch.empty((n,), dtype=torch.long, device=priorities.device)
    inv_rank.scatter_(0, sorted_idx, torch.arange(n, device=priorities.device, dtype=torch.long))
    mode = str(candidate_mode or "rank_window").lower()

    if mode in {"global_topk", "global_head", "legacy"}:
        rows = torch.arange(k_eff, device=priorities.device, dtype=torch.long).unsqueeze(1)
        candidate_pos = rows + (rows >= inv_rank.unsqueeze(0)).to(torch.long)
        candidate_idx = sorted_idx[candidate_pos.reshape(-1)].reshape(k_eff, n)
    elif mode in {"rank_window", "window", "local_rank"}:
        pool_k = min(
            max(k_eff, int(math.ceil(float(max(candidate_pool_factor, 1.0)) * k_eff))),
            n - 1,
        )
        before_offsets = -torch.arange(1, pool_k + 1, device=priorities.device, dtype=torch.long)
        after_offsets = torch.arange(1, pool_k + 1, device=priorities.device, dtype=torch.long)
        offsets = torch.cat((before_offsets, after_offsets), dim=0)

        candidate_pos = inv_rank.unsqueeze(0) + offsets.unsqueeze(1)
        valid = (candidate_pos >= 0) & (candidate_pos < n)
        candidate_pos = candidate_pos.clamp(0, n - 1)
        gathered_idx = sorted_idx[candidate_pos.reshape(-1)].reshape(offsets.shape[0], n)

        # Keep the first `pool_k` valid offsets per target according to the
        # offset order above: closer higher-ranked tasks first, then nearby
        # lower-ranked tasks when the target is already near the front.
        valid_rank = torch.cumsum(valid.to(torch.long), dim=0) - 1
        keep = valid & (valid_rank < pool_k)
        col_idx = torch.arange(n, device=priorities.device, dtype=torch.long).unsqueeze(0).expand_as(valid_rank)
        candidate_idx = torch.empty((pool_k, n), dtype=torch.long, device=priorities.device)
        candidate_idx[valid_rank[keep], col_idx[keep]] = gathered_idx[keep]
    else:
        raise ValueError(
            f"Unsupported large-graph candidate_mode '{candidate_mode}'. "
            "Use rank_window or global_topk."
        )

    src_prio = priorities[candidate_idx]
    tgt_prio = priorities.unsqueeze(0)
    tau = max(float(temperature), 1e-6)
    before_prob = torch.sigmoid((src_prio - tgt_prio) / tau)
    return candidate_idx, before_prob


def _sparse_resource_logits_from_priority(
    F_source: torch.Tensor,
    lane_prob: torch.Tensor,
    priorities: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    resource_logit_alpha: float,
    order_eps: float,
    min_prob: float,
    candidate_mode: str = "rank_window",
    candidate_pool_factor: float = 2.0,
) -> torch.Tensor:
    """
    Build sparse resource-precedence logits for the top-k likely predecessors of
    each target node on a shared resource lane.
    """
    candidate_idx, before_prob = _topk_pairwise_before_from_priority_logits(
        priorities,
        topk=topk,
        temperature=temperature,
        candidate_mode=candidate_mode,
        candidate_pool_factor=candidate_pool_factor,
    )
    if candidate_idx.numel() == 0:
        return F_source.new_empty((0, F_source.shape[0]))

    resource_prob = before_prob * lane_prob[candidate_idx] * lane_prob.unsqueeze(0)
    resource_logits = F_source[candidate_idx] + float(resource_logit_alpha) * torch.log(
        resource_prob + float(order_eps)
    )
    if float(min_prob) > 0.0:
        neg_inf = torch.full_like(resource_logits, -1e9)
        resource_logits = torch.where(resource_prob >= float(min_prob), resource_logits, neg_inf)
    k_eff = min(max(int(topk), 1), int(resource_logits.shape[0]))
    if resource_logits.shape[0] > k_eff:
        resource_logits, _ = torch.topk(resource_logits, k=k_eff, dim=0, sorted=False)
    return resource_logits


def _softmax_beta(vals: torch.Tensor, beta: float, dim: int = 0) -> torch.Tensor:
    b = max(float(beta), 1e-6)
    return (1.0 / b) * torch.logsumexp(b * vals, dim=dim)


def _scatter_softmax_beta(
    messages: torch.Tensor,
    index: torch.Tensor,
    size: int,
    beta: float,
    base: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """
    Grouped log-sum-exp reduction over segments identified by `index`.
    This is used by the Jacobi soft-makespan path to parallelize the DAG pass.
    """
    if size <= 0:
        return messages.new_zeros((0,))

    b = max(float(beta), 1e-6)
    device = messages.device
    dtype = messages.dtype

    if base is None:
        base_tensor = torch.full((size,), -1e9, dtype=dtype, device=device)
    elif torch.is_tensor(base):
        base_tensor = base.to(device=device, dtype=dtype)
    else:
        base_tensor = torch.full((size,), float(base), dtype=dtype, device=device)

    max_per = base_tensor.clone()
    if messages.numel() > 0:
        max_per.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)

    base_exp = torch.exp(b * (base_tensor - max_per))
    if messages.numel() > 0:
        msg_scaled = torch.exp(b * (messages - max_per.index_select(0, index)))
        msg_exp = torch.zeros((size,), dtype=dtype, device=device)
        msg_exp.index_add_(0, index, msg_scaled)
        sumexp = base_exp + msg_exp
    else:
        sumexp = base_exp

    return max_per + torch.log(sumexp.clamp_min(1e-30)) / b


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


def _build_order_loss_cache(TG, node_list, device, dtype):
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    topo_nodes = list(nx.topological_sort(TG.graph))
    topo_idx = [node_to_idx[node] for node in topo_nodes]
    topo_depths = {}
    topo_level_groups = {}
    max_topo_depth = 0
    for node in topo_nodes:
        pred_nodes = list(TG.graph.predecessors(node))
        depth = 0 if not pred_nodes else 1 + max(topo_depths[p] for p in pred_nodes)
        topo_depths[node] = depth
        max_topo_depth = max(max_topo_depth, depth)
        topo_level_groups.setdefault(depth, []).append(node_to_idx[node])

    hw_times = torch.tensor([TG.hardware_costs[node] for node in node_list], dtype=dtype, device=device)
    sw_times = torch.tensor([TG.software_costs[node] for node in node_list], dtype=dtype, device=device)
    areas = torch.tensor([TG.hardware_area.get(node, 0.0) for node in node_list], dtype=dtype, device=device)

    pred_idx_tensors = []
    pred_comm_tensors = []
    other_idx_tensors = []
    arange_n = torch.arange(n, device=device)
    for idx, node in enumerate(node_list):
        pred_nodes = list(TG.graph.predecessors(node))
        if pred_nodes:
            pred_idx = torch.tensor([node_to_idx[p] for p in pred_nodes], dtype=torch.long, device=device)
            pred_comm = torch.tensor(
                [float(TG.communication_costs.get((p, node), 0.0)) for p in pred_nodes],
                dtype=dtype,
                device=device,
            )
        else:
            pred_idx = torch.empty((0,), dtype=torch.long, device=device)
            pred_comm = torch.empty((0,), dtype=dtype, device=device)
        pred_idx_tensors.append(pred_idx)
        pred_comm_tensors.append(pred_comm)
        other_idx_tensors.append(arange_n[arange_n != idx])

    comm_src = []
    comm_dst = []
    comm_cost = []
    for (u, v), c in TG.communication_costs.items():
        if u in node_to_idx and v in node_to_idx:
            comm_src.append(node_to_idx[u])
            comm_dst.append(node_to_idx[v])
            comm_cost.append(float(c))

    if comm_src:
        comm_src_idx = torch.tensor(comm_src, dtype=torch.long, device=device)
        comm_dst_idx = torch.tensor(comm_dst, dtype=torch.long, device=device)
        comm_costs = torch.tensor(comm_cost, dtype=dtype, device=device)
    else:
        comm_src_idx = torch.empty((0,), dtype=torch.long, device=device)
        comm_dst_idx = torch.empty((0,), dtype=torch.long, device=device)
        comm_costs = torch.empty((0,), dtype=dtype, device=device)

    total_area = float(getattr(TG, "total_area", 0.0))
    if total_area == 0.0:
        total_area = float(areas.sum().item())
    if total_area <= 0.0:
        total_area = 1.0

    return {
        "hw_times": hw_times,
        "sw_times": sw_times,
        "areas": areas,
        "total_area": total_area,
        "topo_idx": topo_idx,
        "topo_depth": max_topo_depth + 1 if n > 0 else 0,
        "topo_level_tensors": [
            torch.tensor(level_nodes, dtype=torch.long, device=device)
            for _, level_nodes in sorted(topo_level_groups.items())
        ],
        "pred_idx_tensors": pred_idx_tensors,
        "pred_comm_tensors": pred_comm_tensors,
        "other_idx_tensors": other_idx_tensors,
        "comm_src_idx": comm_src_idx,
        "comm_dst_idx": comm_dst_idx,
        "comm_costs": comm_costs,
        "eye": torch.eye(n, dtype=dtype, device=device),
        "zero": torch.zeros((), dtype=dtype, device=device),
    }


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


def _solution_to_array(solution: dict, node_list) -> np.ndarray:
    return np.asarray([float(solution.get(n, 0)) for n in node_list], dtype=float)


def _priority_array_to_node_scores(priorities: np.ndarray | None, node_list) -> dict | None:
    if priorities is None:
        return None
    arr = np.asarray(priorities, dtype=float).ravel()
    if arr.shape[0] != len(node_list):
        raise ValueError(
            f"Priority vector length mismatch: got {arr.shape[0]} scores for {len(node_list)} nodes."
        )
    return {node_list[i]: float(arr[i]) for i in range(len(node_list))}


def _partition_from_thresholded_probs(node_list, probs, threshold: float = 0.5) -> dict[str, int]:
    arr = np.asarray(probs, dtype=float).ravel()
    return {
        node_list[i]: int(float(arr[i]) > float(threshold))
        for i in range(len(node_list))
    }


def _partition_hw_area(TG, partition: Mapping[str, int]) -> float:
    return float(
        sum(float(TG.hardware_area.get(node, 0.0)) for node, assign in partition.items() if int(assign) == 1)
    )


def _evaluate_partition_lssp_safe(
    TG,
    partition: Mapping[str, int],
    *,
    software_priority_scores: Mapping | None = None,
) -> tuple[float, bool]:
    try:
        result = evaluate_partition_lssp(
            TG,
            dict(partition),
            auto_repair=False,
            software_priority_scores=software_priority_scores,
        )
        finish_times = result.get("finish_times", {}) or {}
        raw_makespan = max((float(v) for v in finish_times.values()), default=0.0)
        is_valid = bool(result.get("is_valid", not TG.violates(partition)))
        return float(raw_makespan), is_valid
    except Exception as exc:
        logger.warning("Ablation LSSP evaluation failed: %s", str(exc))
        return float("inf"), False


def _append_ablation_trace_row(rows: list[dict] | None, **payload) -> None:
    if rows is None:
        return
    rows.append(dict(payload))


def _write_ablation_trace_csv(rows: list[dict], output_csv: str | os.PathLike | None) -> None:
    if not output_csv or not rows:
        return
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "graph_name",
        "graph_file",
        "source_config",
        "phase",
        "event",
        "stage",
        "candidate_label",
        "candidate_mode",
        "epoch",
        "global_step",
        "operation_index",
        "iteration",
        "accepted",
        "training_end",
        "soft_seq_makespan",
        "threshold_lssp_static",
        "threshold_lssp_learned_swprio",
        "postprocess_lssp_cost",
        "delta_from_prev",
        "threshold_partition_valid",
        "threshold_hw_nodes",
        "threshold_hw_area",
        "threshold_budget",
        "tau",
        "order_tau",
        "loss",
        "area_frac",
        "area_penalty",
        "selection_metric_train",
        "selection_metric_final",
        "notes",
    ]
    seen = set()
    fieldnames = []
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _decode_repair_candidates(
    TG,
    base_probs: np.ndarray,
    node_list,
    prio_hw: np.ndarray | None,
    prio_sw: np.ndarray | None,
    order_decode_weight: float,
    fill_decode: bool,
):
    """
    Build multiple discrete decode candidates and keep unique assignments.
    This prevents ordered decode from missing better partition-first repairs.
    """
    probs = np.asarray(base_probs, dtype=float).ravel()
    score_candidates = [("placement", probs)]

    if prio_hw is not None and prio_sw is not None:
        w = float(max(0.0, min(1.0, order_decode_weight)))
        if w > 0:
            score_candidates.append((f"order_w{w:.2f}", _order_aware_repair_scores(probs, prio_hw, prio_sw, w)))
            if w > 0.2:
                w_half = 0.5 * w
                score_candidates.append((f"order_w{w_half:.2f}", _order_aware_repair_scores(probs, prio_hw, prio_sw, w_half)))

    candidates = []
    seen = set()
    n_nodes = len(node_list)
    for label, score in score_candidates:
        repaired = _repair_candidate(TG, probs, node_list, prefer_by_score=score)
        if fill_decode:
            repaired = _fill_hw_area_by_score(
                TG,
                repaired,
                node_list=node_list,
                prefer_by_score=score,
            )
        binary = (np.asarray(repaired, dtype=float).ravel() > 0.5).astype(int)
        key = tuple(binary.tolist())
        if key in seen:
            continue
        seen.add(key)
        solution = {node_list[i]: int(binary[i]) for i in range(n_nodes)}
        candidates.append((label, solution, binary.astype(float)))

    if not candidates:
        binary = (probs > 0.5).astype(int)
        solution = {node_list[i]: int(binary[i]) for i in range(len(node_list))}
        candidates.append(("fallback", solution, binary.astype(float)))

    return candidates


def _pick_best_candidate_by_metric(TG, candidates, metric: str):
    best_label = "none"
    best_solution = None
    best_probs = None
    best_cost = float("inf")

    for label, solution, probs in candidates:
        cost = _evaluate_discrete_solution(TG, solution, metric=metric)
        if cost < best_cost:
            best_cost = cost
            best_label = label
            best_solution = solution
            best_probs = probs

    return best_label, best_solution, best_probs, float(best_cost)


def _dual_lssp_postprocess(
    TG,
    solution: dict,
    *,
    max_iters: int,
    eval_mode: str,
    enable_area_fill: bool,
    fill_allow_worsen: float,
    enable_swap: bool,
    search_strategy: str,
    candidate_top_k: int,
    critical_slack_frac: float,
    candidate_include_neighbors: bool,
    candidate_include_cut_endpoints: bool,
    sw_priority_scores: Mapping | None,
    prefer_sw_priority: bool = False,
    use_dual_mode: bool = False,
    print_progress: bool = False,
    print_every: int = 10,
    print_prefix: str = "[diff_gnn_order][postprocess]",
    trace_rows: list[dict] | None = None,
):
    """
    Run LSSP local search on candidate partition.
    
    If use_dual_mode=True and SW scores available, evaluate each move with BOTH modes
    (static-priority and learned SW-priority) and select based on minimum cost.
    Otherwise, use only the specified (or static) priority mode.
    """
    common_kwargs = dict(
        max_iters=max_iters,
        eval_mode=eval_mode,
        enable_area_fill=enable_area_fill,
        fill_allow_worsen=fill_allow_worsen,
        enable_swap=enable_swap,
        search_strategy=search_strategy,
        candidate_top_k=candidate_top_k,
        critical_slack_frac=critical_slack_frac,
        candidate_include_neighbors=candidate_include_neighbors,
        candidate_include_cut_endpoints=candidate_include_cut_endpoints,
    )

    has_sw_priority = isinstance(sw_priority_scores, Mapping) and len(sw_priority_scores) > 0

    if use_dual_mode and has_sw_priority:
        static_trace_rows = [] if trace_rows is not None else None
        swprio_trace_rows = [] if trace_rows is not None else None

        sol_static, info_static = improve_with_lssp_local_search(
            TG,
            solution,
            software_priority_scores=None,
            eval_both_modes=False,
            progress=print_progress,
            progress_every=print_every,
            progress_prefix=f"{print_prefix}[mode=static]",
            trace_rows=static_trace_rows,
            **common_kwargs,
        )
        sol_swprio, info_swprio = improve_with_lssp_local_search(
            TG,
            solution,
            software_priority_scores=sw_priority_scores,
            eval_both_modes=False,
            progress=print_progress,
            progress_every=print_every,
            progress_prefix=f"{print_prefix}[mode=sw_priority]",
            trace_rows=swprio_trace_rows,
            **common_kwargs,
        )

        cost_static = float(info_static.get("cost", float("inf")))
        cost_swprio = float(info_swprio.get("cost", float("inf")))
        if cost_swprio + 1e-9 < cost_static:
            chosen_mode = "sw_priority"
            chosen_sol = sol_swprio
            chosen_info = info_swprio
            chosen_trace_rows = swprio_trace_rows
        else:
            chosen_mode = "static"
            chosen_sol = sol_static
            chosen_info = info_static
            chosen_trace_rows = static_trace_rows

        if trace_rows is not None and chosen_trace_rows:
            trace_rows.extend(chosen_trace_rows)

        return chosen_mode, chosen_sol, chosen_info, [
            ("static", sol_static, info_static),
            ("sw_priority", sol_swprio, info_swprio),
        ]

    single_use_sw_priority = bool(prefer_sw_priority) and has_sw_priority
    sol_best, info_best = improve_with_lssp_local_search(
        TG,
        solution,
        software_priority_scores=sw_priority_scores if single_use_sw_priority else None,
        eval_both_modes=False,
        progress=print_progress,
        progress_every=print_every,
        progress_prefix=f"{print_prefix}[mode={'sw_priority' if single_use_sw_priority else 'static'}]",
        trace_rows=trace_rows,
        **common_kwargs,
    )

    mode = "sw_priority" if single_use_sw_priority else "static"
    return mode, sol_best, info_best, [(mode, sol_best, info_best)]


def _differentiable_makespan_loss_with_order(
    TG,
    probs_tensor,
    prio_hw,
    prio_sw,
    node_list,
    beta_softmax=20.0,
    area_penalty_coeff=1e3,
    entropy_coeff=0.0,
    usage_balance_coeff=0.0,
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
    perm_entropy_coeff=0.0,
    pairwise_mode="rank_sigmoid",
    pairwise_temp=0.5,
    soft_makespan_mode="jacobi",
    soft_makespan_coeff=1.0,
    jacobi_iters=1,
    use_hw_ordering=False,
    resource_candidate_topk=0,
    resource_candidate_min_prob=0.0,
    large_graph_order_approx=False,
    large_graph_order_topk=64,
    large_graph_order_candidate_mode="rank_window",
    large_graph_order_candidate_pool_factor=2.0,
    loss_cache=None,
):
    """
    Differentiable loss with soft placement + soft ordering:
      - Placement uses relaxed p_hw.
      - Ordering uses Gumbel-Sinkhorn soft permutations for SW lane.
      - HW lane ordering is optional and disabled by default (use_hw_ordering=False).
      - Makespan surrogate combines DAG precedence and resource precedence.
      - Large graphs can skip Sinkhorn entirely and use sparse top-k direct-logit
        precedence for O(NK) resource interactions with target-specific shortlist
        selection.
    """
    device = probs_tensor.device
    dtype = probs_tensor.dtype
    N = probs_tensor.shape[0]

    if loss_cache is None:
        loss_cache = _build_order_loss_cache(TG, node_list, device=device, dtype=dtype)

    hw_times = loss_cache["hw_times"]
    sw_times = loss_cache["sw_times"]
    areas = loss_cache["areas"]
    total_area = float(loss_cache["total_area"])

    exec_time = probs_tensor * hw_times + (1.0 - probs_tensor) * sw_times

    approx_active = bool(large_graph_order_approx) and N > 1
    P_sw = None
    P_hw = None
    before_resource = None
    p_sw = 1.0 - probs_tensor
    gate_hw = None

    if not approx_active:
        # Soft permutation and pairwise "before" probabilities.
        # SW ordering stays active; HW ordering is optional.
        P_sw = _soft_permutation_from_priority(
            prio_sw,
            temperature=order_tau,
            sinkhorn_iters=sinkhorn_iters,
            add_gumbel=bool(gumbel_noise),
            gumbel_scale=gumbel_scale,
        )
        if bool(use_hw_ordering):
            P_hw = _soft_permutation_from_priority(
                prio_hw,
                temperature=order_tau,
                sinkhorn_iters=sinkhorn_iters,
                add_gumbel=bool(gumbel_noise),
                gumbel_scale=gumbel_scale,
            )

        mode = str(pairwise_mode).lower()
        if mode == "exact":
            before_sw = _pairwise_before_from_perm(P_sw)
            before_hw = _pairwise_before_from_perm(P_hw) if P_hw is not None else torch.zeros_like(before_sw)
        elif mode in ("rank", "rank_sigmoid", "fast"):
            before_sw = _pairwise_before_from_expected_rank(P_sw, temperature=pairwise_temp)
            before_hw = (
                _pairwise_before_from_expected_rank(P_hw, temperature=pairwise_temp)
                if P_hw is not None
                else torch.zeros_like(before_sw)
            )
        else:
            raise ValueError(f"Unsupported pairwise_mode '{pairwise_mode}'. Use 'rank_sigmoid' or 'exact'.")

        gate_hw = probs_tensor.unsqueeze(1) * probs_tensor.unsqueeze(0)
        gate_sw = p_sw.unsqueeze(1) * p_sw.unsqueeze(0)
        before_resource = before_sw * gate_sw
        if bool(use_hw_ordering):
            before_resource = before_resource + (before_hw * gate_hw)
        if N > 0:
            before_resource = before_resource * (1.0 - loss_cache["eye"])

    topo_idx = loss_cache["topo_idx"]
    topo_depth = int(loss_cache.get("topo_depth", max(1, len(topo_idx))))
    pred_idx_tensors = loss_cache["pred_idx_tensors"]
    pred_comm_tensors = loss_cache["pred_comm_tensors"]
    comm_src_idx = loss_cache["comm_src_idx"]
    comm_dst_idx = loss_cache["comm_dst_idx"]
    comm_costs = loss_cache["comm_costs"]
    zero = loss_cache["zero"]

    # Iterative refinement: DAG DP pass + resource-precedence pass.
    F_prev = exec_time.clone()
    refine_steps = max(0, int(order_refine_steps))
    soft_mode = str(soft_makespan_mode).lower()
    jacobi_steps = max(1, int(jacobi_iters))

    def _resource_start_times(F_source: torch.Tensor) -> torch.Tensor:
        if N <= 1:
            return exec_time.new_zeros((N,))

        if approx_active:
            min_prob = float(resource_candidate_min_prob)
            topk = min(max(int(large_graph_order_topk), 1), N - 1)
            logits_parts = [
                _sparse_resource_logits_from_priority(
                    F_source,
                    p_sw,
                    prio_sw,
                    topk=topk,
                    temperature=pairwise_temp,
                    resource_logit_alpha=resource_logit_alpha,
                    order_eps=order_eps,
                    min_prob=min_prob,
                    candidate_mode=large_graph_order_candidate_mode,
                    candidate_pool_factor=large_graph_order_candidate_pool_factor,
                )
            ]
            if bool(use_hw_ordering):
                p_hw = probs_tensor
                logits_parts.append(
                    _sparse_resource_logits_from_priority(
                        F_source,
                        p_hw,
                        prio_hw,
                        topk=topk,
                        temperature=pairwise_temp,
                        resource_logit_alpha=resource_logit_alpha,
                        order_eps=order_eps,
                        min_prob=min_prob,
                        candidate_mode=large_graph_order_candidate_mode,
                        candidate_pool_factor=large_graph_order_candidate_pool_factor,
                    )
                )
            logits_parts = [part for part in logits_parts if part.numel() > 0]
            if not logits_parts:
                return exec_time.new_zeros((N,))
            resource_logits = (
                torch.cat(logits_parts, dim=0)
                if len(logits_parts) > 1
                else logits_parts[0]
            )
            return _softmax_beta(resource_logits, beta_softmax, dim=0)

        resource_prob = before_resource
        resource_logits = F_source.unsqueeze(1) + float(resource_logit_alpha) * torch.log(
            resource_prob + float(order_eps)
        )
        min_prob = float(resource_candidate_min_prob)
        if min_prob > 0.0:
            neg_inf = torch.full_like(resource_logits, -1e9)
            resource_logits = torch.where(resource_prob >= min_prob, resource_logits, neg_inf)
        topk = int(resource_candidate_topk)
        if topk > 0 and resource_logits.shape[0] > topk:
            resource_logits, _ = torch.topk(resource_logits, k=topk, dim=0, sorted=False)
        return _softmax_beta(resource_logits, beta_softmax, dim=0)

    for _ in range(refine_steps):
        t_res_all = _resource_start_times(F_prev)

        if soft_mode == "jacobi" and N > 1:
            F_new = F_prev
            n_jacobi = min(jacobi_steps, max(1, topo_depth))
            zero_baseline = exec_time.new_zeros((N,))
            for _ in range(n_jacobi):
                t_res_all = _resource_start_times(F_new)
                if comm_src_idx.numel() > 0:
                    comm = torch.abs(probs_tensor[comm_src_idx] - probs_tensor[comm_dst_idx]) * comm_costs
                    dag_msgs = F_new[comm_src_idx] + comm
                    t_dag_all = _scatter_softmax_beta(
                        dag_msgs,
                        comm_dst_idx,
                        N,
                        beta_softmax,
                        base=zero_baseline,
                    )
                else:
                    t_dag_all = zero_baseline

                start_all = _softmax_beta(
                    torch.stack([t_dag_all, t_res_all], dim=0),
                    beta_softmax,
                    dim=0,
                )
                F_new = start_all + exec_time
        else:
            F_new = torch.zeros_like(exec_time)
            for i in topo_idx:
                pred_idx = pred_idx_tensors[i]
                if pred_idx.numel() > 0:
                    pred_comm = pred_comm_tensors[i]
                    comm = torch.abs(probs_tensor[pred_idx] - probs_tensor[i]) * pred_comm
                    dag_terms = torch.cat((zero.reshape(1), F_new[pred_idx] + comm))
                    t_dag = _softmax_beta(dag_terms, beta_softmax)
                else:
                    t_dag = zero

                if N <= 1:
                    t_res = zero
                else:
                    t_res = t_res_all[i]

                start_i = _softmax_beta(torch.stack([t_dag, t_res]), beta_softmax)
                F_new[i] = start_i + exec_time[i]

        F_prev = F_new

    makespan_soft = _softmax_beta(F_prev, beta_softmax)

    
    # Previous area-violation implementations (kept for reference):
    area_used = torch.dot(probs_tensor, areas)
    area_frac = area_used / float(total_area)
    budget = float(TG.area_constraint)
    #area_violation = F.relu(area_frac - float(TG.area_constraint))
    # area_violation = F.relu(area_frac - float(TG.area_constraint)) ** 2
    area_violation = torch.square(area_frac - float(TG.area_constraint))
    # area_violation = F.softplus(area_frac - float(TG.area_constraint))**2
    # area_violation = (area_frac - float(TG.area_constraint))**2
    area_penalty = area_penalty_coeff * area_violation

    # # Compute soft (expected) area usage for monitoring and downstream soft penalties
    # area_used = torch.dot(probs_tensor, areas)
    # area_frac = area_used / float(total_area)
    # budget = float(TG.area_constraint)
    
    # # Straight-through (ST) binary mask at threshold=0.5
    # # Forward pass uses hard 0/1 assignment; backward pass uses gradients of the
    # # original soft `probs_tensor` (standard ST reparameterization trick).
    # threshold = 0.5
    # hard_mask = (probs_tensor > threshold).float()
    # # ST trick: hard forward, soft backward
    # hard_st = hard_mask.detach() + (probs_tensor - probs_tensor.detach())

    # # Hard-area computed from ST mask (used for discrete constraint evaluation)
    # area_used_hard = torch.dot(hard_st, areas)
    # area_frac_hard = area_used_hard / float(total_area)

    # # Violation computed on the discrete fraction (squared-hinge gives stronger
    # # gradients when violated but zero when under budget).
    # area_violation_hard = F.relu(area_frac_hard - budget) ** 2
    # area_penalty = area_penalty_coeff * area_violation_hard

    # Keep soft metrics for logging/monitoring alongside hard metrics
    area_used_soft = area_used
    area_frac_soft = area_frac

    print(area_penalty.item(), area_penalty_coeff)

    if target_hw_frac is None:
        target_hw_frac = float(TG.area_constraint)
    usage_balance = usage_balance_coeff * (area_frac - float(target_hw_frac)) ** 2

    entropy_like = torch.mean(probs_tensor * (1.0 - probs_tensor))

    exec_cost = torch.sum(exec_time)
    comm_src_idx = loss_cache["comm_src_idx"]
    comm_dst_idx = loss_cache["comm_dst_idx"]
    comm_costs = loss_cache["comm_costs"]
    if comm_src_idx.numel() > 0:
        comm_cost = torch.sum(torch.abs(probs_tensor[comm_src_idx] - probs_tensor[comm_dst_idx]) * comm_costs)
    else:
        comm_cost = torch.tensor(0.0, dtype=dtype, device=device)
    expected_partition_cost = exec_cost + comm_cost

    if approx_active:
        perm_reg = zero
        perm_entropy = zero
    elif P_hw is None:
        perm_reg = _doubly_stochastic_penalty(P_sw)
        perm_entropy = _entropy_rows_cols(P_sw)
    else:
        perm_reg = _doubly_stochastic_penalty(P_hw) + _doubly_stochastic_penalty(P_sw)
        perm_entropy = _entropy_rows_cols(P_hw) + _entropy_rows_cols(P_sw)

    # Scale the makespan surrogate to allow stronger emphasis when desired.
    makespan_term = float(soft_makespan_coeff) * makespan_soft

    loss = (
        makespan_term
        + area_penalty
        + usage_balance
        + entropy_coeff * entropy_like
        + partition_cost_coeff * expected_partition_cost
        + perm_reg_coeff * perm_reg
        + perm_entropy_coeff * perm_entropy
    )

    # Ensure hard-area fields exist even if ST block was commented out.
    if "area_frac_hard" not in locals():
        area_frac_hard = area_frac_soft
    if "area_violation_hard" not in locals():
        try:
            area_violation_hard = area_violation
        except Exception:
            area_violation_hard = torch.tensor(0.0, dtype=dtype, device=device)

    return loss, {
        "makespan_surrogate": makespan_soft.item(),
        # soft (expected) area fraction
        "area_frac": float(area_frac_soft.item()),
        "area_frac_soft": float(area_frac_soft.item()),
        # hard (thresholded via ST) area fraction and its violation
        "area_frac_hard": float(area_frac_hard.item()),
        "area_violation_hard": float(area_violation_hard.item()),
        "area_penalty": area_penalty.item(),
        "usage_balance": usage_balance.item() if isinstance(usage_balance, torch.Tensor) else usage_balance,
        "entropy_like": entropy_like.item(),
        "expected_partition_cost": expected_partition_cost.item(),
        "perm_reg": perm_reg.item(),
        "perm_entropy": perm_entropy.item(),
        "order_approx_active": bool(approx_active),
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
    use_hw_ordering = bool(config.get("use_hw_ordering", False))

    beta_softmax = float(config.get("beta_softmax", 20.0))
    
    area_penalty_coeff = float(config.get("area_penalty_coeff", 1e9))
    soft_makespan_coeff = float(config.get("soft_makespan_coeff", 1e3))


    entropy_coeff = float(config.get("entropy_coeff", 0.0))

    usage_balance_coeff = float(config.get("usage_balance_coeff", 0.0))
    target_hw_frac = config.get("target_hw_frac", None)
    partition_cost_coeff = float(config.get("partition_cost_coeff", 0.0833333333))

    perm_reg_coeff = float(config.get("perm_reg_coeff", 0.0))
    perm_entropy_coeff = float(config.get("perm_entropy_coeff", 0.0))
    pairwise_mode = str(config.get("pairwise_mode", "rank_sigmoid")).lower()
    pairwise_temp = float(config.get("pairwise_temp", 0.5))
    soft_makespan_mode = str(config.get("soft_makespan_mode", "jacobi")).lower()
    jacobi_iters = max(1, int(config.get("jacobi_iters", 10)))
    
    soft_makespan_exact_mode = str(config.get("soft_makespan_exact_mode", "sequential")).lower()
    soft_makespan_exact_every = max(0, int(config.get("soft_makespan_exact_every", 10)))
    soft_makespan_exact_first_epoch = bool(config.get("soft_makespan_exact_first_epoch", False))
    resource_candidate_topk = max(0, int(config.get("resource_candidate_topk", 0)))
    resource_candidate_min_prob = max(0.0, float(config.get("resource_candidate_min_prob", 0.0)))
    large_graph_order_approx_enabled = bool(config.get("large_graph_order_approx_enabled", True))
    large_graph_order_approx_threshold = int(config.get("large_graph_order_approx_threshold", 900))
    large_graph_order_topk = max(1, int(config.get("large_graph_order_topk", 64)))
    large_graph_order_candidate_mode = str(
        config.get("large_graph_order_candidate_mode", "rank_window")
    ).lower()
    large_graph_order_candidate_pool_factor = max(
        1.0,
        float(config.get("large_graph_order_candidate_pool_factor", 2.0)),
    )
    large_graph_order_approx = (
        large_graph_order_approx_enabled and len(node_list) > large_graph_order_approx_threshold
    )
    # Paper-prior blending is opt-in. Keep disabled by default unless explicitly enabled.
    paper_sigma_enabled = bool(config.get("paper_sigma_enabled", config.get("paper_blend_enabled", False)))
    paper_sigma = float(config.get("paper_sigma", 0.0)) if paper_sigma_enabled else 0.0
    paper_sigma = max(0.0, min(1.0, paper_sigma))

    seed = int(config.get("seed", 42))
    hard_eval_every = int(config.get("hard_eval_every", max(1, epochs // 5)))
    hard_eval_only_final = bool(config.get("hard_eval_only_final", True))
    checkpoint_eval_when_final_only = bool(config.get("checkpoint_eval_when_final_only", False))
    early_stop_enabled = bool(config.get("early_stop_enabled", True))
    early_stop_min_epochs = max(1, min(int(config.get("early_stop_min_epochs", 250)), epochs))
    early_stop_patience = max(0, int(config.get("early_stop_patience", 5)))
    early_stop_min_delta = max(0.0, float(config.get("early_stop_min_delta", 1e-4)))
    progress_log_every = max(1, int(config.get("progress_log_every", 50)))
    selection_metric_train = str(config.get("selection_metric_train", config.get("selection_metric", "queue"))).lower()
    selection_metric_final = str(config.get("selection_metric_final", selection_metric_train)).lower()
    sampler = (config.get("sampling") or config.get("sampler") or "soft").lower()
    logit_scale = float(config.get("logit_scale", 8.0))
    center_logits = bool(config.get("center_logits", True))
    hard_train_outputs = bool(config.get("hard_train_outputs", sampler != "soft"))

    # Warn if area penalty coefficient is zero — the loss will not include area penalty.
    if float(area_penalty_coeff) == 0.0:
        logger.warning(
            "area_penalty_coeff is 0.0; area penalty will NOT be included in the training loss."
        )
    order_decode_weight = float(config.get("order_decode_weight", 0.25))
    if not use_hw_ordering:
        # Avoid using untrained HW-order head in decode-time scoring.
        order_decode_weight = 0.0
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
    adaptive_post_max_iters = bool(post_cfg.get("adaptive_max_iters", config.get("adaptive_post_max_iters", True)))
    adaptive_post_large_n = int(post_cfg.get("adaptive_large_n", config.get("adaptive_post_large_n", 128)))
    adaptive_post_large_cap = int(post_cfg.get("adaptive_large_cap", config.get("adaptive_post_large_cap", 48)))
    post_enable_area_fill = bool(post_cfg.get("enable_area_fill", config.get("lssp_postprocess_area_fill", True)))
    post_fill_allow_worsen = float(post_cfg.get("fill_allow_worsen", config.get("lssp_postprocess_fill_allow_worsen", 0.0)))
    post_enable_swap = bool(post_cfg.get("enable_swap", config.get("lssp_postprocess_enable_swap", True)))
    post_search_strategy = str(post_cfg.get("search_strategy", config.get("lssp_postprocess_search_strategy", "critical"))).lower()
    post_candidate_top_k = int(post_cfg.get("candidate_top_k", config.get("lssp_postprocess_candidate_top_k", 16)))
    post_use_sw_priority = bool(post_cfg.get("use_sw_priority", config.get("lssp_use_sw_priority", False)))
    post_use_dual_lssp = bool(
        post_cfg.get(
            "use_dual_lssp_postprocess",
            post_cfg.get("best_of_lssp_and_sw_order", config.get("lssp_use_dual_postprocess", False)),
        )
    )
    post_critical_slack_frac = float(post_cfg.get("critical_slack_frac", config.get("lssp_postprocess_critical_slack_frac", 0.05)))
    post_final_all_decode_candidates = bool(
        post_cfg.get("final_all_decode_candidates", config.get("lssp_postprocess_final_all_decode_candidates", True))
    )
    post_print_progress = bool(
        post_cfg.get("print_progress", config.get("lssp_postprocess_print_progress", True))
    )
    post_print_every = max(1, int(post_cfg.get("print_every", config.get("lssp_postprocess_print_every", 10))))
    post_candidate_include_neighbors = bool(
        post_cfg.get("candidate_include_neighbors", config.get("lssp_postprocess_candidate_include_neighbors", True))
    )
    post_candidate_include_cut_endpoints = bool(
        post_cfg.get("candidate_include_cut_endpoints", config.get("lssp_postprocess_candidate_include_cut_endpoints", True))
    )
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
    ablation_cfg_raw = config.get("ablation_trace", {})
    ablation_cfg = dict(ablation_cfg_raw) if isinstance(ablation_cfg_raw, Mapping) else {}
    ablation_enabled = bool(ablation_cfg.get("enabled", False))
    ablation_output_csv = str(ablation_cfg.get("output_csv", "") or "").strip()
    ablation_every = max(1, int(ablation_cfg.get("compute_every", 1)))
    ablation_threshold = float(ablation_cfg.get("discrete_threshold", 0.5))
    ablation_include_static = bool(ablation_cfg.get("include_static_lssp", True))
    ablation_include_swprio = bool(ablation_cfg.get("include_learned_swprio_lssp", True))
    ablation_soft_mode = str(ablation_cfg.get("soft_mode", "sequential")).lower()
    ablation_graph_name = str(
        config.get("_graph_name", config.get("_dataset_name", config.get("graph_name", ""))) or ""
    )
    ablation_graph_file = str(config.get("_graph_file", config.get("graph-file", "")) or "")
    ablation_source_config = str(config.get("_source_config_path", config.get("config", "")) or "")
    if adaptive_post_max_iters and len(node_list) >= adaptive_post_large_n and post_max_iters > adaptive_post_large_cap:
        logger.info(
            "DiffGNNOrder adaptive postprocess cap: max_iters %d -> %d for N=%d",
            post_max_iters,
            adaptive_post_large_cap,
            len(node_list),
        )
        post_max_iters = adaptive_post_large_cap

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "DiffGNNOrder training: sampler=%s epochs=%d lr=%.2e tau=%.2f->%.2f order_tau=%.2f->%.2f sinkhorn=%d gumbel=%s alpha=%.2f use_hw_ordering=%s pairwise_mode=%s pairwise_temp=%.2f soft_mode=%s jacobi_iters=%d exact_mode=%s exact_every=%d exact_first=%s resource_topk=%d resource_min_prob=%.2e post_mode=%s post_during_train=%s feature_profile=%s edge_weight_mode=%s paper_sigma_enabled=%s paper_sigma=%.2f",
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
        str(use_hw_ordering),
        pairwise_mode,
        pairwise_temp,
        soft_makespan_mode,
        jacobi_iters,
        soft_makespan_exact_mode,
        soft_makespan_exact_every,
        str(soft_makespan_exact_first_epoch),
        resource_candidate_topk,
        resource_candidate_min_prob,
        post_mode,
        str(post_during_train),
        str(config.get("feature_profile", "default")),
        str(config.get("edge_weight_mode", "auto")),
        str(paper_sigma_enabled),
        paper_sigma,
    )
    logger.info(
        "DiffGNNOrder large-graph order approx: enabled=%s active=%s N=%d threshold=%d topk=%d candidate_mode=%s pool_factor=%.2f skip_sinkhorn=%s",
        str(large_graph_order_approx_enabled),
        str(large_graph_order_approx),
        len(node_list),
        large_graph_order_approx_threshold,
        large_graph_order_topk,
        large_graph_order_candidate_mode,
        large_graph_order_candidate_pool_factor,
        str(large_graph_order_approx),
    )
    if large_graph_order_approx:
        print(
            "[diff_gnn_order] "
            f"large_graph_order_approx=enabled "
            f"nodes={len(node_list)} "
            f"threshold={large_graph_order_approx_threshold} "
            f"topk={large_graph_order_topk} "
            f"candidate_mode={large_graph_order_candidate_mode} "
            f"pool_factor={large_graph_order_candidate_pool_factor:.2f} "
            f"skip_sinkhorn=True",
            flush=True,
        )
    logger.info(
        "DiffGNNOrder metrics: train_metric=%s final_metric=%s",
        selection_metric_train,
        selection_metric_final,
    )
    logger.info(
        "DiffGNNOrder hard eval: every=%d hard_eval_only_final=%s checkpoint_eval_when_final_only=%s",
        hard_eval_every,
        str(hard_eval_only_final),
        str(checkpoint_eval_when_final_only),
    )
    logger.info(
        "DiffGNNOrder early stop: enabled=%s min_epochs=%d patience=%d min_delta=%.2e",
        str(early_stop_enabled),
        early_stop_min_epochs,
        early_stop_patience,
        early_stop_min_delta,
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
            "DiffGNNOrder final postprocess enabled: eval_mode=%s during_train=%s during_eval=%s max_iters=%d area_fill=%s fill_allow_worsen=%.3f swap=%s search=%s top_k=%d slack_frac=%.3f all_decode_candidates=%s print_progress=%s print_every=%d dual_lssp_compare=%s (legacy use_sw_priority=%s)",
            post_eval_mode,
            str(post_during_train),
            str(post_during_eval),
            post_max_iters,
            str(post_enable_area_fill),
            post_fill_allow_worsen,
            str(post_enable_swap),
            post_search_strategy,
            post_candidate_top_k,
            post_critical_slack_frac,
            str(post_final_all_decode_candidates),
            str(post_print_progress),
            post_print_every,
            str(post_use_dual_lssp),
            str(post_use_sw_priority),
        )
    if ablation_enabled:
        logger.info(
            "DiffGNNOrder ablation trace enabled: output=%s every=%d threshold=%.2f soft_mode=%s include_static=%s include_swprio=%s",
            ablation_output_csv,
            ablation_every,
            ablation_threshold,
            ablation_soft_mode,
            str(ablation_include_static),
            str(ablation_include_swprio),
        )

    best_sched_cost = float("inf")
    best_assign = None
    best_probs = None
    best_sw_priority_scores = None
    best_loss_value = float("inf")
    best_loss_epoch = 0
    stagnant_epochs = 0
    early_stop_monitoring_started = False
    completed_epochs = 0
    ablation_trace_rows: list[dict] | None = [] if ablation_enabled else None

    tau = tau_start
    order_tau = order_tau_start
    edge_weight = getattr(data, "edge_weight", None)
    edge_attr = getattr(data, "edge_attr", None)
    paper_hgp = getattr(data, "paper_hgp", None)
    loss_cache = _build_order_loss_cache(TG, node_list, device=device, dtype=data.x.dtype)
    train_t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        completed_epochs = ep
        model.train()
        optimizer.zero_grad()

        is_progress_epoch = (ep == 1 or ep % progress_log_every == 0 or ep == epochs)
        run_exact_soft_makespan = (
            (soft_makespan_exact_first_epoch and ep == 1)
            or (soft_makespan_exact_every > 0 and ep % soft_makespan_exact_every == 0)
        )
        epoch_soft_makespan_mode = (
            soft_makespan_exact_mode if run_exact_soft_makespan else soft_makespan_mode
        )
        epoch_jacobi_iters = jacobi_iters if epoch_soft_makespan_mode == "jacobi" else 1
        should_print_epoch = is_progress_epoch or run_exact_soft_makespan or (
            epoch_soft_makespan_mode == soft_makespan_mode
        )

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
            soft_makespan_mode=epoch_soft_makespan_mode,
            jacobi_iters=epoch_jacobi_iters,
            soft_makespan_coeff=soft_makespan_coeff,
            use_hw_ordering=use_hw_ordering,
            resource_candidate_topk=resource_candidate_topk,
            resource_candidate_min_prob=resource_candidate_min_prob,
            large_graph_order_approx=large_graph_order_approx,
            large_graph_order_topk=large_graph_order_topk,
            large_graph_order_candidate_mode=large_graph_order_candidate_mode,
            large_graph_order_candidate_pool_factor=large_graph_order_candidate_pool_factor,
            loss_cache=loss_cache,
        )

        loss.backward()
        optimizer.step()

        current_loss_value = float(info["loss"])
        if early_stop_enabled and ep < early_stop_min_epochs:
            if current_loss_value < best_loss_value:
                best_loss_value = current_loss_value
                best_loss_epoch = ep
        elif early_stop_enabled and not early_stop_monitoring_started:
            # Start patience tracking only after the warmup window ends.
            best_loss_value = current_loss_value
            best_loss_epoch = ep
            stagnant_epochs = 0
            early_stop_monitoring_started = True
        elif current_loss_value < (best_loss_value - early_stop_min_delta):
            best_loss_value = current_loss_value
            best_loss_epoch = ep
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        tau = max(tau_final, tau_start - (ep / max(1, epochs)) * (tau_start - tau_final))
        order_tau = max(order_tau_final, order_tau_start - (ep / max(1, epochs)) * (order_tau_start - order_tau_final))

        run_hard_eval = (
            (ep == epochs)
            or ((not hard_eval_only_final) and (ep % hard_eval_every == 0))
            or (hard_eval_only_final and checkpoint_eval_when_final_only and (ep % hard_eval_every == 0))
        )
        if run_hard_eval:
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
                if use_hw_ordering:
                    prio_hw_np = prio_hw_eval.detach().cpu().numpy().astype(float)
                    prio_sw_np = prio_sw_eval.detach().cpu().numpy().astype(float)
                else:
                    prio_hw_np = None
                    prio_sw_np = None
                sw_priority_scores_eval = _priority_array_to_node_scores(
                    prio_sw_eval.detach().cpu().numpy().astype(float),
                    node_list,
                )
                decode_candidates = _decode_repair_candidates(
                    TG,
                    hard_probs,
                    node_list=node_list,
                    prio_hw=prio_hw_np,
                    prio_sw=prio_sw_np,
                    order_decode_weight=order_decode_weight,
                    fill_decode=bool(use_dls_train and dls_fill_decode),
                )
                _, solution, decoded_probs, current_sched_cost = _pick_best_candidate_by_metric(
                    TG,
                    decode_candidates,
                    metric=selection_metric_train,
                )
                if use_lssp_eval:
                    _, solution_post, _, _ = _dual_lssp_postprocess(
                        TG,
                        solution,
                        max_iters=post_max_iters,
                        eval_mode=post_eval_mode,
                        enable_area_fill=post_enable_area_fill,
                        fill_allow_worsen=post_fill_allow_worsen,
                        enable_swap=post_enable_swap,
                        search_strategy=post_search_strategy,
                        candidate_top_k=post_candidate_top_k,
                        critical_slack_frac=post_critical_slack_frac,
                        candidate_include_neighbors=post_candidate_include_neighbors,
                        candidate_include_cut_endpoints=post_candidate_include_cut_endpoints,
                        sw_priority_scores=sw_priority_scores_eval,
                        use_dual_mode=post_use_dual_lssp,
                    )
                    post_cost = _evaluate_discrete_solution(
                        TG,
                        solution_post,
                        metric=selection_metric_train,
                    )
                    if post_cost <= current_sched_cost:
                        solution = solution_post
                        decoded_probs = _solution_to_array(solution_post, node_list)
                        current_sched_cost = post_cost

                if current_sched_cost < best_sched_cost:
                    best_sched_cost = current_sched_cost
                    best_assign = solution.copy()
                    best_probs = np.asarray(decoded_probs, dtype=float).copy()
                    best_sw_priority_scores = sw_priority_scores_eval

        if ablation_trace_rows is not None and (ep % ablation_every == 0 or ep == 1 or ep == epochs):
            with torch.no_grad():
                _, ablation_soft_info = _differentiable_makespan_loss_with_order(
                    TG,
                    probs.detach(),
                    prio_hw.detach(),
                    prio_sw.detach(),
                    node_list,
                    beta_softmax=beta_softmax,
                    area_penalty_coeff=area_penalty_coeff,
                    entropy_coeff=entropy_coeff,
                    usage_balance_coeff=usage_balance_coeff,
                    target_hw_frac=target_hw_frac,
                    partition_cost_coeff=partition_cost_coeff,
                    order_tau=order_tau,
                    sinkhorn_iters=sinkhorn_iters,
                    gumbel_noise=False,
                    gumbel_scale=0.0,
                    resource_logit_alpha=resource_logit_alpha,
                    order_refine_steps=order_refine_steps,
                    perm_reg_coeff=perm_reg_coeff,
                    perm_entropy_coeff=perm_entropy_coeff,
                    pairwise_mode=pairwise_mode,
                    pairwise_temp=pairwise_temp,
                    soft_makespan_mode=ablation_soft_mode,
                    jacobi_iters=1,
                    soft_makespan_coeff=soft_makespan_coeff,
                    use_hw_ordering=use_hw_ordering,
                    resource_candidate_topk=resource_candidate_topk,
                    resource_candidate_min_prob=resource_candidate_min_prob,
                    large_graph_order_approx=large_graph_order_approx,
                    large_graph_order_topk=large_graph_order_topk,
                    large_graph_order_candidate_mode=large_graph_order_candidate_mode,
                    large_graph_order_candidate_pool_factor=large_graph_order_candidate_pool_factor,
                    loss_cache=loss_cache,
                )
                threshold_partition = _partition_from_thresholded_probs(
                    node_list,
                    probs.detach().cpu().numpy(),
                    threshold=ablation_threshold,
                )
                threshold_valid = not TG.violates(threshold_partition)
                static_lssp = float("nan")
                learned_lssp = float("nan")
                if ablation_include_static:
                    static_lssp, _ = _evaluate_partition_lssp_safe(TG, threshold_partition)
                if ablation_include_swprio:
                    sw_priority_scores_epoch = _priority_array_to_node_scores(
                        prio_sw.detach().cpu().numpy().astype(float),
                        node_list,
                    )
                    learned_lssp, _ = _evaluate_partition_lssp_safe(
                        TG,
                        threshold_partition,
                        software_priority_scores=sw_priority_scores_epoch,
                    )
                _append_ablation_trace_row(
                    ablation_trace_rows,
                    graph_name=ablation_graph_name,
                    graph_file=ablation_graph_file,
                    source_config=ablation_source_config,
                    phase="train",
                    event="epoch",
                    stage="train",
                    candidate_label="thresholded_partition",
                    candidate_mode="threshold",
                    epoch=int(ep),
                    global_step=float(ep),
                    operation_index=0,
                    iteration=int(ep),
                    accepted=False,
                    training_end=False,
                    soft_seq_makespan=float(ablation_soft_info["makespan_surrogate"]),
                    threshold_lssp_static=static_lssp,
                    threshold_lssp_learned_swprio=learned_lssp,
                    postprocess_lssp_cost=float("nan"),
                    delta_from_prev=float("nan"),
                    threshold_partition_valid=bool(threshold_valid),
                    threshold_hw_nodes=int(sum(int(v) for v in threshold_partition.values())),
                    threshold_hw_area=float(_partition_hw_area(TG, threshold_partition)),
                    threshold_budget=float(TG.area_constraint * TG.total_area),
                    tau=float(tau),
                    order_tau=float(order_tau),
                    loss=float(info["loss"]),
                    area_frac=float(info["area_frac"]),
                    # For ablation traces we do not record the computed area penalty;
                    # keep loss computation unchanged so training still uses the penalty.
                    area_penalty=float("nan"),
                    selection_metric_train=selection_metric_train,
                    selection_metric_final=selection_metric_final,
                    notes="per_epoch_trace",
                )

        if should_print_epoch:
            elapsed_sec = time.perf_counter() - train_t0
            avg_epoch_sec = elapsed_sec / max(1, ep)
            eta_sec = avg_epoch_sec * max(0, epochs - ep)
            # compute expected (continuous) area and budget for logging
            try:
                expected_area = float(info.get("area_frac", float("nan"))) * float(TG.total_area)
            except Exception:
                expected_area = float("nan")
            try:
                area_budget = float(TG.area_constraint) * float(TG.total_area)
            except Exception:
                area_budget = float("nan")
            logger.info(
                "Epoch %d/%d mode=%s loss=%.6f soft_makespan=%.6f area_frac=%.4f expected_area=%.3f area_budget=%.3f area_pen=%.3f perm_reg=%.4f perm_H=%.4f sched_best=%.6f elapsed=%.2fs avg_epoch=%.3fs eta=%.2fs",
                ep,
                epochs,
                epoch_soft_makespan_mode,
                info["loss"],
                info["makespan_surrogate"],
                info["area_frac"],
                expected_area,
                area_budget,
                info["area_penalty"],
                info["perm_reg"],
                info["perm_entropy"],
                best_sched_cost,
                elapsed_sec,
                avg_epoch_sec,
                eta_sec,
            )
            print(
                "[diff_gnn_order] "
                f"epoch={ep}/{epochs} "
                f"mode={epoch_soft_makespan_mode} "
                f"loss={info['loss']:.6f} "
                f"soft_makespan={info['makespan_surrogate']:.6f} "
                f"expected_area={expected_area:.3f} "
                f"area_budget={area_budget:.3f} "
                f"best_sched={best_sched_cost:.6f} "
                f"elapsed={elapsed_sec:.2f}s "
                f"avg_epoch={avg_epoch_sec:.3f}s "
                f"eta={eta_sec:.2f}s",
                flush=True,
            )

        if early_stop_enabled and ep >= early_stop_min_epochs and stagnant_epochs >= early_stop_patience:
            logger.info(
                "DiffGNNOrder early stop triggered at epoch %d/%d: best_loss=%.6f at epoch %d, no improvement above min_delta=%.2e for %d epochs.",
                ep,
                epochs,
                best_loss_value,
                best_loss_epoch,
                early_stop_min_delta,
                stagnant_epochs,
            )
            break

    optimization_elapsed_sec = time.perf_counter() - train_t0
    postprocess_phase_t0 = time.perf_counter()

    logger.info("DiffGNNOrder final decode started after %d/%d training epochs.", completed_epochs, epochs)
    if ablation_trace_rows:
        for row in reversed(ablation_trace_rows):
            if row.get("phase") == "train":
                row["training_end"] = True
                row["notes"] = "training_end"
                break
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
        if use_hw_ordering:
            prio_hw_np = prio_hw.detach().cpu().numpy().astype(float)
            prio_sw_np = prio_sw.detach().cpu().numpy().astype(float)
        else:
            prio_hw_np = None
            prio_sw_np = None
        final_sw_priority_scores = _priority_array_to_node_scores(
            prio_sw.detach().cpu().numpy().astype(float),
            node_list,
        )
        final_decode_candidates = _decode_repair_candidates(
            TG,
            final_probs,
            node_list=node_list,
            prio_hw=prio_hw_np,
            prio_sw=prio_sw_np,
            order_decode_weight=order_decode_weight,
            fill_decode=bool(use_dls_final and dls_fill_decode),
        )
        final_choice_label, final_solution, final_probs_repaired, final_sched_cost_train = _pick_best_candidate_by_metric(
            TG,
            final_decode_candidates,
            metric=selection_metric_train,
        )
        logger.info(
            "DiffGNNOrder final decode selected candidate=%s from %d candidates (metric=%s cost=%.6f)",
            final_choice_label,
            len(final_decode_candidates),
            selection_metric_train,
            final_sched_cost_train,
        )
        if ablation_trace_rows is not None:
            decode_static_lssp, _ = _evaluate_partition_lssp_safe(TG, final_solution)
            decode_learned_lssp = float("nan")
            if ablation_include_swprio:
                decode_learned_lssp, _ = _evaluate_partition_lssp_safe(
                    TG,
                    final_solution,
                    software_priority_scores=final_sw_priority_scores,
                )
            _append_ablation_trace_row(
                ablation_trace_rows,
                graph_name=ablation_graph_name,
                graph_file=ablation_graph_file,
                source_config=ablation_source_config,
                phase="postprocess",
                event="decode_selected",
                stage="decode",
                candidate_label=str(final_choice_label),
                candidate_mode="decode",
                epoch=int(completed_epochs),
                global_step=float(completed_epochs) + 0.25,
                operation_index=0,
                iteration=0,
                accepted=False,
                training_end=False,
                soft_seq_makespan=float("nan"),
                threshold_lssp_static=float(decode_static_lssp),
                threshold_lssp_learned_swprio=float(decode_learned_lssp),
                postprocess_lssp_cost=float("nan"),
                delta_from_prev=float("nan"),
                threshold_partition_valid=bool(not TG.violates(final_solution)),
                threshold_hw_nodes=int(sum(int(v) for v in final_solution.values())),
                threshold_hw_area=float(_partition_hw_area(TG, final_solution)),
                threshold_budget=float(TG.area_constraint * TG.total_area),
                tau=float("nan"),
                order_tau=float("nan"),
                loss=float("nan"),
                area_frac=float("nan"),
                area_penalty=float("nan"),
                selection_metric_train=selection_metric_train,
                selection_metric_final=selection_metric_final,
                notes="decode_before_postprocess",
            )
        if use_lssp_final:
            post_t0 = time.perf_counter()
            logger.info(
                "DiffGNNOrder final postprocess started: mode=%s eval_mode=%s max_iters=%d swap=%s all_candidates=%s",
                post_mode,
                post_eval_mode,
                post_max_iters,
                str(post_enable_swap),
                str(post_final_all_decode_candidates),
            )
            post_seed_candidates = final_decode_candidates
            if not post_final_all_decode_candidates:
                post_seed_candidates = [
                    (
                        final_choice_label,
                        final_solution,
                        np.asarray(final_probs_repaired, dtype=float).copy(),
                    )
                ]
            print(
                "[diff_gnn_order] "
                f"postprocess_start mode={post_mode} "
                f"eval_mode={post_eval_mode} "
                f"max_iters={post_max_iters} "
                f"all_candidates={str(post_final_all_decode_candidates)} "
                f"print_every={post_print_every} "
                f"candidates={len(post_seed_candidates)}",
                flush=True,
            )

            selected_post_label = final_choice_label
            selected_post_mode = "decode"
            selected_post_info = None
            selected_post_static_cost = float("nan")
            selected_post_swprio_cost = float("nan")

            for cand_idx, (cand_label, cand_solution, _) in enumerate(post_seed_candidates, start=1):
                candidate_trace_rows = [] if ablation_trace_rows is not None else None
                print(
                    "[diff_gnn_order] "
                    f"postprocess_candidate_start idx={cand_idx}/{len(post_seed_candidates)} "
                    f"label={cand_label} "
                    f"current_best={final_sched_cost_train:.6f}",
                    flush=True,
                )
                post_choice_mode, candidate_solution, post_info, post_candidates = _dual_lssp_postprocess(
                    TG,
                    cand_solution,
                    max_iters=post_max_iters,
                    eval_mode=post_eval_mode,
                    enable_area_fill=post_enable_area_fill,
                    fill_allow_worsen=post_fill_allow_worsen,
                    enable_swap=post_enable_swap,
                    search_strategy=post_search_strategy,
                    candidate_top_k=post_candidate_top_k,
                    critical_slack_frac=post_critical_slack_frac,
                    candidate_include_neighbors=post_candidate_include_neighbors,
                    candidate_include_cut_endpoints=post_candidate_include_cut_endpoints,
                    sw_priority_scores=final_sw_priority_scores,
                    prefer_sw_priority=post_use_sw_priority,
                    print_progress=post_print_progress,
                    print_every=post_print_every,
                    print_prefix=f"[diff_gnn_order][postprocess][{cand_idx}/{len(post_seed_candidates)}:{cand_label}]",
                    use_dual_mode=post_use_dual_lssp,
                    trace_rows=candidate_trace_rows,
                )
                static_cost = float("nan")
                swprio_cost = float("nan")
                for mode_name, _, info in post_candidates:
                    if mode_name == "static":
                        static_cost = float(info.get("cost", float("nan")))
                    elif mode_name == "sw_priority":
                        swprio_cost = float(info.get("cost", float("nan")))
                post_cost = _evaluate_discrete_solution(
                    TG,
                    candidate_solution,
                    metric=selection_metric_train,
                )
                print(
                    "[diff_gnn_order] "
                    f"postprocess_candidate_done idx={cand_idx}/{len(post_seed_candidates)} "
                    f"label={cand_label} "
                    f"mode={post_choice_mode} "
                    f"post_cost={post_cost:.6f} "
                    f"static_cost={static_cost:.6f} "
                    f"swprio_cost={swprio_cost:.6f}",
                    flush=True,
                )
                if ablation_trace_rows is not None and candidate_trace_rows:
                    for op_idx, trace_row in enumerate(candidate_trace_rows, start=1):
                        _append_ablation_trace_row(
                            ablation_trace_rows,
                            graph_name=ablation_graph_name,
                            graph_file=ablation_graph_file,
                            source_config=ablation_source_config,
                            phase="postprocess",
                            event=str(trace_row.get("event", "")),
                            stage=str(trace_row.get("stage", "")),
                            candidate_label=str(cand_label),
                            candidate_mode=str(post_choice_mode),
                            epoch=int(completed_epochs),
                            global_step=float(completed_epochs) + 0.25 + float(op_idx),
                            operation_index=int(op_idx),
                            iteration=int(trace_row.get("iteration", 0)),
                            accepted=bool(trace_row.get("accepted", False)),
                            training_end=False,
                            soft_seq_makespan=float("nan"),
                            threshold_lssp_static=float("nan"),
                            threshold_lssp_learned_swprio=float("nan"),
                            postprocess_lssp_cost=float(trace_row.get("postprocess_lssp_cost", float("nan"))),
                            delta_from_prev=float(trace_row.get("delta_from_prev", float("nan"))),
                            threshold_partition_valid=True,
                            threshold_hw_nodes=int(trace_row.get("threshold_hw_nodes", sum(int(v) for v in candidate_solution.values()))),
                            threshold_hw_area=float(trace_row.get("threshold_hw_area", _partition_hw_area(TG, candidate_solution))),
                            threshold_budget=float(trace_row.get("threshold_budget", TG.area_constraint * TG.total_area)),
                            tau=float("nan"),
                            order_tau=float("nan"),
                            loss=float("nan"),
                            area_frac=float("nan"),
                            area_penalty=float("nan"),
                            selection_metric_train=selection_metric_train,
                            selection_metric_final=selection_metric_final,
                            notes="postprocess_trace",
                        )
                if post_cost <= final_sched_cost_train:
                    final_choice_label = cand_label
                    final_solution = candidate_solution
                    final_sched_cost_train = post_cost
                    final_probs_repaired = _solution_to_array(candidate_solution, node_list)
                    selected_post_label = cand_label
                    selected_post_mode = post_choice_mode
                    selected_post_info = post_info
                    selected_post_static_cost = static_cost
                    selected_post_swprio_cost = swprio_cost

            if selected_post_info is not None:
                logger.info(
                    "DiffGNNOrder final postprocess: decode_candidate=%s mode=%s tried=%d static_cost=%.3f swprio_cost=%.3f improved=%s cost=%.3f hw_area=%.3f/%.3f (%s) elapsed=%.3fs eval_calls=%d stage1_iters=%d stage2_iters=%d avg_eval=%.3fms avg_iter=%.3fms search=%s avg_pool=%.1f avg_selected=%.1f",
                    selected_post_label,
                    selected_post_mode,
                    len(post_seed_candidates),
                    selected_post_static_cost,
                    selected_post_swprio_cost,
                    str(selected_post_info["improved"]),
                    selected_post_info["cost"],
                    selected_post_info["hw_area"],
                    selected_post_info["budget"],
                    selected_post_info["eval_mode"],
                    float(selected_post_info.get("elapsed_sec", 0.0)),
                    int(selected_post_info.get("eval_calls", 0)),
                    int(selected_post_info.get("stage1_iters", 0)),
                    int(selected_post_info.get("stage2_iters", 0)),
                    float(selected_post_info.get("avg_eval_ms", 0.0)),
                    float(selected_post_info.get("avg_iter_ms", 0.0)),
                    str(selected_post_info.get("search_strategy", post_search_strategy)),
                    float(selected_post_info.get("avg_candidate_pool", 0.0)),
                    float(selected_post_info.get("avg_selected_candidates", 0.0)),
                )
            else:
                logger.info(
                    "DiffGNNOrder final postprocess: no candidate improved over decode baseline (candidate=%s tried=%d metric=%s cost=%.6f)",
                    final_choice_label,
                    len(post_seed_candidates),
                    selection_metric_train,
                    final_sched_cost_train,
                )
            logger.info("DiffGNNOrder final postprocess elapsed: %.3fs", time.perf_counter() - post_t0)
            print(
                "[diff_gnn_order] "
                f"postprocess_done best_label={final_choice_label} "
                f"best_mode={selected_post_mode} "
                f"best_cost={final_sched_cost_train:.6f} "
                f"elapsed={time.perf_counter() - post_t0:.3f}s",
                flush=True,
            )

    if best_assign is None or final_sched_cost_train < best_sched_cost:
        best_assign = final_solution
        best_probs = np.asarray(final_probs_repaired, dtype=float)
        best_sched_cost = final_sched_cost_train
        best_sw_priority_scores = final_sw_priority_scores

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

    postprocess_elapsed_sec = time.perf_counter() - postprocess_phase_t0

    logger.info(
        "DiffGNNOrder training finished after %d/%d epochs. Best %s makespan: %.6f; selected assignment %s makespan: %.6f",
        completed_epochs,
        epochs,
        selection_metric_train,
        best_sched_cost,
        selection_metric_final,
        best_final_cost,
    )
    print(
        "[diff_gnn_order] "
        f"final_train_metric={selection_metric_train} "
        f"best_train_makespan={best_sched_cost:.6f} "
        f"final_metric={selection_metric_final} "
        f"final_makespan={best_final_cost:.6f}",
        flush=True,
    )
    if ablation_trace_rows is not None:
        final_global_step = (
            max(float(row.get("global_step", 0.0)) for row in ablation_trace_rows) + 1.0
            if ablation_trace_rows
            else float(completed_epochs) + 1.0
        )
        _append_ablation_trace_row(
            ablation_trace_rows,
            graph_name=ablation_graph_name,
            graph_file=ablation_graph_file,
            source_config=ablation_source_config,
            phase="final",
            event="selected_final",
            stage="final",
            candidate_label=str(final_choice_label),
            candidate_mode="final",
            epoch=int(completed_epochs),
            global_step=float(final_global_step),
            operation_index=int(max(1, final_global_step - float(completed_epochs))),
            iteration=0,
            accepted=False,
            training_end=False,
            soft_seq_makespan=float("nan"),
            threshold_lssp_static=float(best_final_cost),
            threshold_lssp_learned_swprio=float("nan"),
            postprocess_lssp_cost=float(best_sched_cost),
            delta_from_prev=float("nan"),
            threshold_partition_valid=bool(best_assign is not None and not TG.violates(best_assign)),
            threshold_hw_nodes=int(sum(int(v) for v in best_assign.values())) if isinstance(best_assign, Mapping) else 0,
            threshold_hw_area=float(_partition_hw_area(TG, best_assign or {})),
            threshold_budget=float(TG.area_constraint * TG.total_area),
            tau=float("nan"),
            order_tau=float("nan"),
            loss=float("nan"),
            area_frac=float("nan"),
            area_penalty=float("nan"),
            selection_metric_train=selection_metric_train,
            selection_metric_final=selection_metric_final,
            notes="selected_final_solution",
        )
        _write_ablation_trace_csv(ablation_trace_rows, ablation_output_csv)
    return {
        "best_assign": best_assign,
        "best_probs": np.asarray(best_probs),
        "best_mip_cost": float(best_final_cost),
        "best_train_cost": float(best_sched_cost),
        "best_sw_priority_scores": best_sw_priority_scores,
        "selection_metric_train": selection_metric_train,
        "selection_metric_final": selection_metric_final,
        "completed_epochs": int(completed_epochs),
        "configured_epochs": int(epochs),
        "optimization_time_sec": float(optimization_elapsed_sec),
        "postprocess_time_sec": float(postprocess_elapsed_sec),
        "total_core_time_sec": float(optimization_elapsed_sec + postprocess_elapsed_sec),
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
    simulate_diff_GNN_order.last_run_meta = None

    TG = getattr(func_to_optimize, "__self__", None)
    if TG is None:
        msg = "func_to_optimize must be a bound TaskGraph method so the graph can be accessed."
        logger.error(msg)
        raise ValueError(msg)

    # diff_gnn_order should use its own explicit YAML block when present;
    # otherwise it falls back to the runtime Python defaults below.
    diff_cfg = dict(config.get("diffgnn_order", {}))
    dataset_name = _apply_dataset_specific_diffgnn_defaults(diff_cfg, config)
    if dataset_name:
        logger.info("diff_gnn_order dataset context resolved as: %s", dataset_name)
    diff_cfg.setdefault("_dataset_name", dataset_name or "")
    diff_cfg.setdefault("_graph_name", dataset_name or Path(str(config.get("graph-file", "") or "")).stem)
    diff_cfg.setdefault("_graph_file", str(config.get("graph-file", "") or ""))
    diff_cfg.setdefault("_source_config_path", str(config.get("config", "") or ""))

    for key, value in _MKSPAN_DIFFGNN_ORDER_DEFAULTS.items():
        diff_cfg.setdefault(key, value)

    fast_mode = bool(diff_cfg.get("fast_mode", True))
    if fast_mode:
        for key, value in _FAST_MODE_DEFAULTS.items():
            diff_cfg.setdefault(key, value)
        logger.info("diff_gnn_order fast_mode enabled. Applied speed defaults for unset keys.")

    if "iter" in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["epochs"] = diff_cfg["iter"]
    if "learn_edge_weight" not in diff_cfg and "learned_edge_weight" in diff_cfg:
        diff_cfg["learn_edge_weight"] = bool(diff_cfg.get("learned_edge_weight"))
    if "edge_weight_learner" not in diff_cfg and bool(diff_cfg.get("learn_edge_weight", False)):
        diff_cfg["edge_weight_learner"] = "per_edge"
    epochs = int(diff_cfg.get("epochs", diff_cfg.get("iter", _MKSPAN_DIFFGNN_ORDER_DEFAULTS["iter"])))

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
        diff_cfg.setdefault("sinkhorn_iters", 4)
        diff_cfg.setdefault("order_refine_steps", 4)
        diff_cfg.setdefault("gumbel_noise", False)
        diff_cfg.setdefault("gumbel_scale", 0.0)

    if "hard_eval_every" not in diff_cfg:
        diff_cfg["hard_eval_every"] = max(1, int(epochs) // 5)

    post_cfg_raw = diff_cfg.get("postprocess", {})
    post_cfg = dict(post_cfg_raw) if isinstance(post_cfg_raw, Mapping) else {}
    for key, value in _MKSPAN_POSTPROCESS_DEFAULTS.items():
        post_cfg.setdefault(key, value)
    diff_cfg["postprocess"] = post_cfg
    large_graph_dag_policy = _apply_large_graph_cheap_dag_policy(
        TG,
        diff_cfg,
        method_label="diff_gnn_order",
    )

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
        diff_cfg["partition_cost_coeff"] = 0.0833333333
        diff_cfg["perm_reg_coeff"] = 0.0
        diff_cfg["perm_entropy_coeff"] = 0.0

    # Ordering defaults
    diff_cfg.setdefault("order_tau_start", 1.0)
    diff_cfg.setdefault("order_tau_final", 0.2)
    diff_cfg["sinkhorn_iters"] = int(
        os.getenv("HWSW_DIFFGNN_ORDER_SINKHORN_ITERS", str(diff_cfg.get("sinkhorn_iters", 4)))
    )
    diff_cfg.setdefault("gumbel_noise", False)
    diff_cfg.setdefault("gumbel_scale", 0.0)
    diff_cfg.setdefault("resource_logit_alpha", 2.0)
    diff_cfg.setdefault("order_refine_steps", 4)
    diff_cfg.setdefault("perm_reg_coeff", 0.0)
    diff_cfg.setdefault("perm_entropy_coeff", 0.0)
    diff_cfg.setdefault("pairwise_mode", "rank_sigmoid")
    diff_cfg.setdefault("pairwise_temp", 0.30)
    diff_cfg.setdefault("soft_makespan_mode", "jacobi")
    diff_cfg.setdefault("jacobi_iters", 10)
    diff_cfg.setdefault("soft_makespan_exact_mode", "sequential")
    diff_cfg.setdefault("soft_makespan_exact_every", 10)
    diff_cfg.setdefault("soft_makespan_exact_first_epoch", False)
    diff_cfg["soft_makespan_mode"] = str(
        os.getenv("HWSW_DIFFGNN_ORDER_SOFT_MAKESPAN_MODE", diff_cfg.get("soft_makespan_mode", "jacobi"))
    ).lower()
    diff_cfg["jacobi_iters"] = int(
        os.getenv("HWSW_DIFFGNN_ORDER_JACOBI_ITERS", str(diff_cfg.get("jacobi_iters", 10)))
    )
    diff_cfg["soft_makespan_exact_mode"] = str(
        os.getenv(
            "HWSW_DIFFGNN_ORDER_SOFT_MAKESPAN_EXACT_MODE",
            diff_cfg.get("soft_makespan_exact_mode", "sequential"),
        )
    ).lower()
    diff_cfg["soft_makespan_exact_every"] = int(
        os.getenv(
            "HWSW_DIFFGNN_ORDER_SOFT_MAKESPAN_EXACT_EVERY",
            str(diff_cfg.get("soft_makespan_exact_every", 10)),
        )
    )
    diff_cfg["soft_makespan_exact_first_epoch"] = str(
        os.getenv(
            "HWSW_DIFFGNN_ORDER_SOFT_MAKESPAN_EXACT_FIRST_EPOCH",
            str(diff_cfg.get("soft_makespan_exact_first_epoch", False)),
        )
    ).strip().lower() in {"1", "true", "yes", "on"}

    if "seed" not in diff_cfg and "seed" in config:
        diff_cfg["seed"] = config.get("seed")

    if "device" in diff_cfg:
        config["device"] = diff_cfg.get("device")
    elif "device" not in config:
        config["device"] = "gpu"

    device = get_device(config)
    logger.info("Using device: %s", device)
    print(f"[diff_gnn_order] device={device}", flush=True)

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
    if large_graph_dag_policy:
        best_cost = float(result.get("best_mip_cost", eval_cost))
        if not math.isfinite(best_cost):
            best_cost = float(eval_cost)
        logger.info(
            "simulate_diff_GNN_order large-graph policy active: skipping wrapper-level LSSP "
            "evaluation; final MethodRegistry reporting will evaluate the selected partition with LSSP."
        )
        print(
            "[diff_gnn_order] "
            f"eval_cost={eval_cost:.6f} "
            "lssp_cost=deferred "
            "lssp_swprio_cost=deferred "
            f"best_cost={best_cost:.6f}",
            flush=True,
        )
        simulate_diff_GNN_order.last_run_meta = {
            "eval_cost": float(eval_cost),
            "lssp_cost": None,
            "lssp_swprio_cost": None,
            "sw_priority_scores": (
                dict(result.get("best_sw_priority_scores", {}))
                if isinstance(result.get("best_sw_priority_scores", None), Mapping)
                else None
            ),
            "selection_metric": selection_metric,
            "selection_metric_train": str(result.get("selection_metric_train", "unknown")),
            "completed_epochs": int(result.get("completed_epochs", epochs)),
            "configured_epochs": int(result.get("configured_epochs", epochs)),
            "optimization_time_sec": float(result.get("optimization_time_sec", 0.0)),
            "postprocess_time_sec": float(result.get("postprocess_time_sec", 0.0)),
            "total_core_time_sec": float(result.get("total_core_time_sec", 0.0)),
            "large_graph_dag_policy": True,
        }
        return best_cost, sol_arr

    lssp_cost = float(evaluate_partition_lssp(TG, solution)["makespan"])
    sw_priority_scores = result.get("best_sw_priority_scores", None)
    lssp_swprio_cost = float("inf")
    if isinstance(sw_priority_scores, Mapping):
        try:
            lssp_swprio_cost = float(
                evaluate_partition_lssp(
                    TG,
                    solution,
                    software_priority_scores=sw_priority_scores,
                )["makespan"]
            )
        except Exception as e:
            logger.warning(
                "DiffGNNOrder sw-priority LSSP evaluation failed; ignoring this score: %s",
                str(e),
            )

    best_cost = float(result.get("best_mip_cost", eval_cost))
    if not math.isfinite(best_cost):
        best_cost = eval_cost
    best_cost = float(min(best_cost, eval_cost, lssp_cost, lssp_swprio_cost))

    logger.info(
        "simulate_diff_GNN_order finished: best_cost=%.6f (eval_cost=%.6f, lssp_cost=%.6f, lssp_swprio_cost=%.6f, train_best_cost=%.6f, metric=%s, train_metric=%s)",
        best_cost,
        eval_cost,
        lssp_cost,
        lssp_swprio_cost if math.isfinite(lssp_swprio_cost) else float("nan"),
        result.get("best_train_cost", float("nan")),
        selection_metric,
        result.get("selection_metric_train", "unknown"),
    )
    print(
        "[diff_gnn_order] "
        f"eval_cost={eval_cost:.6f} "
        f"lssp_cost={lssp_cost:.6f} "
        f"lssp_swprio_cost={lssp_swprio_cost:.6f} "
        f"best_cost={best_cost:.6f}",
        flush=True,
    )
    simulate_diff_GNN_order.last_run_meta = {
        "eval_cost": float(eval_cost),
        "lssp_cost": float(lssp_cost),
        "lssp_swprio_cost": (float(lssp_swprio_cost) if math.isfinite(lssp_swprio_cost) else None),
        "sw_priority_scores": (dict(sw_priority_scores) if isinstance(sw_priority_scores, Mapping) else None),
        "selection_metric": selection_metric,
        "selection_metric_train": str(result.get("selection_metric_train", "unknown")),
        "completed_epochs": int(result.get("completed_epochs", epochs)),
        "configured_epochs": int(result.get("configured_epochs", epochs)),
        "optimization_time_sec": float(result.get("optimization_time_sec", 0.0)),
        "postprocess_time_sec": float(result.get("postprocess_time_sec", 0.0)),
        "total_core_time_sec": float(result.get("total_core_time_sec", 0.0)),
    }
    return best_cost, sol_arr
