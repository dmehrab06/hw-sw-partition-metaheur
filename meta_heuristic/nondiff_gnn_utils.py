import os
import logging
import numpy as np
import networkx as nx
from utils.logging_utils import LogManager
import sys

# # initialize logger for this module
# LogManager.initialize("logs/gnn_utils.log")
# logger = LogManager.get_logger(__name__)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/gnn_utils.log")

logger = LogManager.get_logger(__name__)
# ...existing code...

def _build_features_and_adj(TG):
    """
    Build a simple node feature matrix and normalized adjacency from a TaskGraph instance TG.
    Features: [software_cost, hardware_cost, hardware_area, degree]
    Returns (features: np.ndarray (N,F), adj_norm: np.ndarray (N,N), node_list)
    """
    G = TG.graph
    node_list = list(G.nodes())
    N = len(node_list)

    # raw features
    sw = np.array([TG.software_costs[n] for n in node_list], dtype=np.float64)
    hw = np.array([TG.hardware_costs.get(n, 0.0) for n in node_list], dtype=np.float64)
    area = np.array([TG.hardware_area.get(n, 1.0) for n in node_list], dtype=np.float64)
    deg = np.array([G.degree(n) for n in node_list], dtype=np.float64)

    X = np.vstack([sw, hw, area, deg]).T  # shape (N,4)

    # simple min-max normalization per column (avoid div0)
    mins = X.min(axis=0, keepdims=True)
    maxs = X.max(axis=0, keepdims=True)
    ranges = np.where(maxs - mins <= 1e-9, 1.0, (maxs - mins))
    X_norm = (X - mins) / ranges

    # adjacency matrix (undirected/symmetric)
    A = nx.to_numpy_array(G, nodelist=node_list, dtype=np.float64)
    # add self loops for propagation
    A = A + np.eye(N)
    # row-normalize A
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A / row_sums

    return X_norm, A_norm, node_list


def _simple_gnn_forward(X, A_norm, W1, b1, W2, b2):
    """
    Simple 2-layer GNN-like forward pass using adjacency mixing:
      H = relu(A_norm @ X @ W1 + b1)
      out = sigmoid(A_norm @ H @ W2 + b2)
    Returns vector of shape (N,) with values in (0,1).
    """
    H = A_norm @ X @ W1 + b1  # (N,hidden)
    H = np.maximum(H, 0.0)    # relu
    out = A_norm @ H @ W2 + b2  # (N,1)
    out = 1.0 / (1.0 + np.exp(-out))  # sigmoid
    return out.ravel()


def _repair_candidate(TG, x_continuous, node_list=None, prefer_by_score=None, max_iters=500):
    """
    Repair a continuous candidate into a feasible binary assignment where possible.
    Strategy:
      - Binarize with threshold 0.5
      - Enforce area budget (if TaskGraph provides total_area and area_constraint)
      - Use TG.violates (if present) to check validity and try simple local flips
    Returns binary numpy array shape (N,) of 0.0/1.0 values.
    """
    x = np.asarray(x_continuous).astype(float).ravel().copy()
    N = x.shape[0]
    if node_list is None:
        node_list = list(getattr(TG, "graph").nodes()) if hasattr(TG, "graph") else list(range(N))

    # initial binarization
    x_bin = (x >= 0.5).astype(float)

    # hardware area per node
    hw_area_map = getattr(TG, "hardware_area", None)
    if hw_area_map is not None:
        areas = np.array([hw_area_map.get(n, 0.0) for n in node_list], dtype=float)
    else:
        areas = np.zeros(N, dtype=float)

    # detect area budget from TaskGraph: area_constraint * total_area if possible
    area_budget = None
    if hasattr(TG, "area_constraint") and hasattr(TG, "total_area"):
        try:
            area_budget = float(TG.area_constraint) * float(TG.total_area)
        except Exception:
            area_budget = None

    # prefer scores (higher => prefer to keep on hardware)
    if prefer_by_score is None:
        prefer_by_score = x

    # enforce area budget greedily (remove lowest-score hardware nodes)
    if area_budget is not None:
        total_area = float(np.dot(areas, x_bin))
        if total_area > area_budget + 1e-12:
            hw_idx = np.where(x_bin > 0.5)[0]
            if hw_idx.size > 0:
                order = hw_idx[np.argsort(prefer_by_score[hw_idx])]
                for i in order:
                    x_bin[i] = 0.0
                    total_area -= areas[i]
                    if total_area <= area_budget + 1e-12:
                        break

    # build validator wrapper
    validator = None
    if hasattr(TG, "violates"):
        # TG.violates expects dict {node:placement}, returns 1 if violates, 0 otherwise
        def _validator_arr(arr):
            sol = {node_list[i]: (1 if float(arr[i]) > 0.5 else 0) for i in range(len(node_list))}
            try:
                return (TG.violates(sol) == 0)
            except Exception:
                return False
        validator = _validator_arr
    else:
        # try generic boolean validators if any
        for name in ("is_valid_assignment", "satisfies_constraints", "check_constraints", "validate_assignment", "constraints_ok"):
            if hasattr(TG, name):
                fn = getattr(TG, name)
                def _generic_validator(arr, fn=fn):
                    sol = {node_list[i]: (1 if float(arr[i]) > 0.5 else 0) for i in range(len(node_list))}
                    try:
                        res = fn(sol)
                        if isinstance(res, bool):
                            return res
                        if isinstance(res, (int, float)):
                            return bool(res)
                        return True
                    except Exception:
                        return False
                validator = _generic_validator
                break

    def _is_valid_arr(arr):
        if validator is None:
            return True
        return validator(arr)

    if _is_valid_arr(x_bin):
        return x_bin

    # local repair: try flipping low-score hardware nodes first, occasionally try adding high-score software nodes if area allows
    best = x_bin.copy()
    rng = np.random.RandomState(0)
    for it in range(max_iters):
        cand = best.copy()
        hw_idx = np.where(cand > 0.5)[0]
        sw_idx = np.where(cand <= 0.5)[0]

        # prioritize removing low-score hardware nodes if over budget or by probability
        if hw_idx.size > 0 and (area_budget is not None and np.dot(areas, cand) > area_budget + 1e-12 or rng.rand() < 0.8):
            # remove lowest-score hardware node
            i = hw_idx[np.argmin(prefer_by_score[hw_idx])]
            cand[i] = 0.0
        elif sw_idx.size > 0:
            # try adding highest-score software node if area permits
            j = sw_idx[np.argmax(prefer_by_score[sw_idx])]
            if area_budget is None or (np.dot(areas, cand) + areas[j]) <= area_budget + 1e-12:
                cand[j] = 1.0
            else:
                continue

        if _is_valid_arr(cand):
            return cand

        # occasionally move starting point
        if rng.rand() < 0.05:
            best = cand

    # fallback to all-software if valid
    fallback = np.zeros_like(x_bin)
    if _is_valid_arr(fallback):
        return fallback

    return x_bin


def simulate_nondiff_GNN(dim, func_to_optimize, config):
    """
    Lightweight GNN-based candidate generator / scorer.
    - Expects func_to_optimize to be a bound method of TaskGraph (so we can access TG via func_to_optimize.__self__).
    - Generates multiple candidate assignment vectors x in [0,1] using a tiny MLP/GNN (pure numpy),
      evaluates them via func_to_optimize (which should accept a batch array shape (n_candidates, N))
      and returns the best found solution and its cost.

    Returns:
        (best_cost: float, best_x: np.ndarray) with best_x shape (N,)
    """
    logger.info("Starting simulate_nondiff_GNN")
    # determine TG from bound method if possible
    TG = getattr(func_to_optimize, "__self__", None)
    if TG is None:
        msg = "func_to_optimize must be a bound TaskGraph method (so the TaskGraph instance can be accessed)."
        logger.error(msg)
        raise ValueError(msg)

    # config parameters with sensible defaults
    n_samples = int(config.get('gcomopt', {}).get('n_samples', 128))
    hidden_dim = int(config.get('gcomopt', {}).get('hidden_dim', 16))
    seed = int(config.get('seed', 42))

    np.random.seed(seed)

    X, A_norm, node_list = _build_features_and_adj(TG)
    N = X.shape[0]
    assert N == dim, f"dim ({dim}) != number of nodes ({N})"

    # generate candidate assignments by sampling random small-weight GNNs
    candidates = np.zeros((n_samples, N), dtype=np.float64)
    repaired_candidates = np.zeros((n_samples, N), dtype=np.float64)

    for i in range(n_samples):
        # small random weights, small biases
        W1 = 0.1 * np.random.randn(X.shape[1], hidden_dim)
        b1 = 0.01 * np.random.randn(1, hidden_dim)
        W2 = 0.1 * np.random.randn(hidden_dim, 1)
        b2 = 0.01 * np.random.randn(1, 1)

        out = _simple_gnn_forward(X, A_norm, W1, b1, W2, b2)  # (N,)
        out = np.clip(out, 0.0, 1.0)
        candidates[i, :] = out

        try:
            repaired = _repair_candidate(TG, out, node_list=node_list, prefer_by_score=out)
            repaired_candidates[i, :] = repaired
        except Exception as e:
            logger.warning("Candidate repair failed (sample %d): %s — using raw output", i, str(e))
            repaired_candidates[i, :] = out

    # evaluate repaired candidates using provided function (expects shape (n_samples, N))
    try:
        costs = func_to_optimize(repaired_candidates)
    except Exception as e:
        logger.error("Error while calling func_to_optimize on GNN candidates: %s", str(e))
        raise

    # func_to_optimize may return array-like with structured elements (e.g., dict) in some variants;
    # handle common cases: numeric array, or array of scalars, or list of dicts with 'makespan' key
    if isinstance(costs, np.ndarray) and costs.dtype == object:
        processed = []
        for v in costs:
            if isinstance(v, dict) and 'makespan' in v:
                processed.append(float(v['makespan']))
            else:
                processed.append(float(v))
        costs = np.array(processed, dtype=np.float64)
    elif isinstance(costs, list):
        try:
            costs = np.array(costs, dtype=np.float64)
        except Exception:
            processed = []
            for v in costs:
                if isinstance(v, dict) and 'makespan' in v:
                    processed.append(float(v['makespan']))
                else:
                    processed.append(float(v))
            costs = np.array(processed, dtype=np.float64)

    # find best (min cost)
    best_idx = int(np.argmin(costs))
    best_cost = float(costs[best_idx])
    best_x = repaired_candidates[best_idx, :].astype(np.float64)

    logger.info("nondiff_GNN finished: best_cost=%.6f (sample %d/%d)", best_cost, best_idx + 1, n_samples)
    return best_cost, best_x