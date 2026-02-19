import math
import random
from typing import Tuple

import networkx as nx
import numpy as np

from utils.logging_utils import LogManager

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch should exist in normal runs
    torch = None
    nn = None


logger = LogManager.get_logger(__name__)
_EPS = 1e-12


def _as_float(x) -> float:
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).reshape(-1)[0])
    if np.isscalar(x):
        return float(x)
    return float(np.asarray(x).reshape(-1)[0])


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < _EPS, 1.0, std)
    return (x - mean) / std


def _minmax01(x: np.ndarray) -> np.ndarray:
    lo = np.min(x)
    hi = np.max(x)
    if hi - lo < _EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _build_gcps_inputs(task_graph, node_list):
    graph_undirected = task_graph.graph.to_undirected()
    n_nodes = len(node_list)

    sw = np.array([float(task_graph.software_costs[n]) for n in node_list], dtype=np.float64)
    hw = np.array([float(task_graph.hardware_costs[n]) for n in node_list], dtype=np.float64)
    area = np.array([float(task_graph.hardware_area[n]) for n in node_list], dtype=np.float64)

    hg = sw - hw
    hgp = hg / np.maximum(area, _EPS)

    h0_raw = np.stack([hg, hgp], axis=1)
    h0 = _standardize_columns(h0_raw)

    a = nx.to_numpy_array(graph_undirected, nodelist=node_list, dtype=np.float64)
    a_tilde = a + np.eye(n_nodes, dtype=np.float64)
    degree = np.sum(a_tilde, axis=1)
    d_inv_sqrt = np.power(np.maximum(degree, _EPS), -0.5)
    a_hat = (d_inv_sqrt[:, None] * a_tilde) * d_inv_sqrt[None, :]

    # Paper update: replace non-zero off-diagonal entries with cosine similarity.
    h_norm = h0 / np.maximum(np.linalg.norm(h0, axis=1, keepdims=True), _EPS)
    cosine = np.clip(h_norm @ h_norm.T, -1.0, 1.0)
    cosine01 = 0.5 * (cosine + 1.0)

    a0 = a_hat.copy()
    off_diag = ~np.eye(n_nodes, dtype=bool)
    edge_mask = (a_hat > 0.0) & off_diag
    a0[edge_mask] = cosine01[edge_mask]
    a0 = 0.5 * (a0 + a0.T)

    return {
        "sw": sw,
        "hw": hw,
        "area": area,
        "h0": h0.astype(np.float32),
        "a0": a0.astype(np.float32),
        "hgp01": _minmax01(hgp).astype(np.float32),
    }


def _decode_greedy_area(
    scores: np.ndarray,
    area: np.ndarray,
    area_budget: float,
    quick_search: bool = True,
) -> np.ndarray:
    n_nodes = scores.shape[0]
    labels = np.zeros(n_nodes, dtype=np.float64)
    if area_budget <= 0.0:
        return labels

    order = np.argsort(-scores)
    selected = []
    used_area = 0.0

    if quick_search:
        area_mean = float(np.mean(area)) if area.size else 1.0
        if area_mean <= _EPS:
            area_mean = 1.0
        skip_num = int(max(0.0, math.floor(area_budget / area_mean)))
        for idx in order[:skip_num]:
            selected.append(int(idx))
            used_area += float(area[idx])

        # Dynamic adjustment when initial quick assignment exceeds budget.
        if used_area > area_budget + _EPS:
            selected.sort(key=lambda i: float(scores[i]))
            while used_area > area_budget + _EPS and selected:
                pop_idx = selected.pop(0)
                used_area -= float(area[pop_idx])
        remaining = order[skip_num:]
    else:
        remaining = order

    for idx in remaining:
        idx = int(idx)
        next_area = used_area + float(area[idx])
        if next_area <= area_budget + _EPS:
            selected.append(idx)
            used_area = next_area

    if selected:
        labels[np.array(selected, dtype=int)] = 1.0
    return labels


def _evaluate_candidate(task_graph, func_to_optimize, node_list, x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    blackbox_cost = _as_float(func_to_optimize(x))
    partition = {node_list[i]: int(x[i] > 0.5) for i in range(len(node_list))}
    schedule = float(task_graph.evaluate_makespan(partition)["makespan"])
    return blackbox_cost, schedule


def _get_device(config_value: str):
    if torch is None:
        return "cpu"
    desired = str(config_value or "auto").lower()
    if desired == "cpu":
        return "cpu"
    if desired in ("gpu", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class _GCPSNet(nn.Module):
    """Two hidden layer GCN used by GCPS."""

    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden1) * 0.1)
        self.w2 = nn.Parameter(torch.randn(hidden1, hidden2) * 0.1)
        self.w_out = nn.Parameter(torch.randn(hidden2, 2) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, adj: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(adj @ (h0 @ self.w1))
        h1 = self.dropout(h1)
        h2 = torch.relu(adj @ (h1 @ self.w2))
        h2 = self.dropout(h2)
        logits = h2 @ self.w_out + self.b_out
        probs = torch.softmax(logits, dim=1)[:, 1]
        return torch.clamp(probs, min=0.0, max=1.0)


def simulate_gcps(dim, func_to_optimize, config):
    """
    Graph Convolution Partitioning and Scheduling (GCPS).
    Returns:
        (best_cost, best_solution_vector)
    """
    logger.info("Starting simulate_gcps")

    task_graph = getattr(func_to_optimize, "__self__", None)
    if task_graph is None:
        raise ValueError("func_to_optimize must be a bound TaskGraph method.")

    node_list = list(task_graph.graph.nodes())
    if len(node_list) != int(dim):
        raise ValueError(f"dim ({dim}) does not match node count ({len(node_list)}).")

    gcps_cfg = dict(config.get("gcps", {}))
    seed = int(gcps_cfg.get("seed", config.get("seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    lr = float(gcps_cfg.get("lr", 1e-3))
    dropout = float(gcps_cfg.get("dropout", 0.2))
    pretrain_iter = int(gcps_cfg.get("pretrain_iter", gcps_cfg.get("pretrain_epochs", 100)))
    max_epochs = int(gcps_cfg.get("max_epochs", gcps_cfg.get("iter", 800)))
    schedule_skip = max(1, int(gcps_cfg.get("schedule_skip", gcps_cfg.get("s", 5))))
    sigma = float(gcps_cfg.get("sigma", 0.3))
    hidden1 = int(gcps_cfg.get("hidden_dim_1", gcps_cfg.get("h1", 10)))
    hidden2 = int(gcps_cfg.get("hidden_dim_2", gcps_cfg.get("h2", 5)))
    weight_decay = float(gcps_cfg.get("weight_decay", 0.0))
    quick_search = bool(gcps_cfg.get("quick_search", True))
    area_penalty_coeff = float(gcps_cfg.get("area_penalty_coeff", 0.0))
    verbose_every = int(gcps_cfg.get("verbose", 0))

    alpha = float(gcps_cfg.get("alpha", 5.0))
    if "early_stop_k" in gcps_cfg:
        early_stop_k = max(1, int(gcps_cfg["early_stop_k"]))
    else:
        if dim <= 200:
            early_stop_k = max(1, int(round(dim / max(alpha, _EPS))))
        else:
            early_stop_k = max(1, int(round(dim / max(2.0 * alpha, _EPS))))

    prep = _build_gcps_inputs(task_graph, node_list)
    sw = prep["sw"]
    hw = prep["hw"]
    area = prep["area"]
    h0 = prep["h0"]
    a0 = prep["a0"]
    hgp01 = prep["hgp01"]

    area_budget = float(task_graph.area_constraint) * float(task_graph.total_area)

    # CPU fallback if torch is unavailable.
    if torch is None:
        logger.warning("PyTorch not available; using heuristic-only GCPS fallback.")
        base_scores = np.clip(0.5 * _minmax01(sw - hw) + 0.5 * hgp01, 0.0, 1.0)
        best_x = _decode_greedy_area(base_scores, area, area_budget, quick_search=quick_search)
        best_cost, _ = _evaluate_candidate(task_graph, func_to_optimize, node_list, best_x)
        return best_cost, best_x

    device = _get_device(gcps_cfg.get("device", config.get("device", "auto")))
    logger.info("GCPS device=%s", device)

    adj_t = torch.as_tensor(a0, dtype=torch.float32, device=device)
    h0_t = torch.as_tensor(h0, dtype=torch.float32, device=device)
    sw_t = torch.as_tensor(sw, dtype=torch.float32, device=device)
    hw_t = torch.as_tensor(hw, dtype=torch.float32, device=device)
    area_t = torch.as_tensor(area, dtype=torch.float32, device=device)
    area_budget_t = torch.tensor(float(area_budget), dtype=torch.float32, device=device)

    model = _GCPSNet(in_dim=h0.shape[1], hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _train_step():
        optimizer.zero_grad(set_to_none=True)
        probs = model(adj_t, h0_t)
        tcost = torch.sum(hw_t * probs + sw_t * (1.0 - probs))
        if area_penalty_coeff > 0.0:
            area_violation = torch.relu(torch.sum(area_t * probs) - area_budget_t)
            tcost = tcost + area_penalty_coeff * (area_violation ** 2)
        tcost.backward()
        optimizer.step()
        return float(tcost.detach().cpu())

    # Start from a deterministic score baseline.
    init_scores = np.clip(0.5 * _minmax01(sw - hw) + 0.5 * hgp01, 0.0, 1.0)
    best_x = _decode_greedy_area(init_scores, area, area_budget, quick_search=quick_search)
    best_cost, best_schedule = _evaluate_candidate(task_graph, func_to_optimize, node_list, best_x)
    best_epoch = 0

    metric_name = str(getattr(func_to_optimize, "__name__", "")).lower()
    prefer_schedule = ("makespan" in metric_name) or ("_mip" in metric_name)

    # Pre-training (Algorithm 2 in the paper).
    for _ in range(max(0, pretrain_iter)):
        _train_step()

    for epoch in range(1, max_epochs + 1):
        loss_val = _train_step()

        if epoch % schedule_skip != 0 and epoch != max_epochs:
            continue

        model.eval()
        with torch.no_grad():
            probs = model(adj_t, h0_t).detach().cpu().numpy()
        model.train()

        # Feature aggregation trick from the paper: X'=(1-sigma)X + sigma*F1.
        blended = np.clip((1.0 - sigma) * probs + sigma * hgp01, 0.0, 1.0)
        cand_x = _decode_greedy_area(blended, area, area_budget, quick_search=quick_search)
        cand_cost, cand_schedule = _evaluate_candidate(task_graph, func_to_optimize, node_list, cand_x)

        if prefer_schedule:
            improved = (
                cand_schedule < best_schedule - 1e-9
                or (abs(cand_schedule - best_schedule) <= 1e-9 and cand_cost < best_cost - 1e-9)
            )
        else:
            improved = (
                cand_cost < best_cost - 1e-9
                or (abs(cand_cost - best_cost) <= 1e-9 and cand_schedule < best_schedule - 1e-9)
            )

        if improved:
            best_x = cand_x
            best_cost = cand_cost
            best_schedule = cand_schedule
            best_epoch = epoch

        if verbose_every > 0 and epoch % verbose_every == 0:
            logger.info(
                "GCPS epoch=%d loss=%.6f best_cost=%.6f best_schedule=%.6f",
                epoch,
                loss_val,
                best_cost,
                best_schedule,
            )

        # Early termination from the paper (epoch - opt_epoch < k).
        if epoch - best_epoch >= early_stop_k:
            break

    logger.info(
        "simulate_gcps finished: best_cost=%.6f best_schedule=%.6f best_epoch=%d",
        best_cost,
        best_schedule,
        best_epoch,
    )
    return float(best_cost), np.asarray(best_x, dtype=np.float64)
