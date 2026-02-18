import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import os
import random
import math
import sys
from collections.abc import Mapping

# torch-geometric imports
try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("This module requires torch_geometric. Install with 'pip install torch-geometric'.") from e

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/diff_gnn_utils.log")

logger = LogManager.get_logger(__name__)

try:
    from .diff_gnn_models import build_placement_model
except Exception:
    from diff_gnn_models import build_placement_model  # type: ignore
try:
    from .lssp_postprocess import improve_with_lssp_local_search
except Exception:
    from lssp_postprocess import improve_with_lssp_local_search  # type: ignore


def _set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _enable_determinism(seed: int) -> None:
    _set_global_seeds(seed)
    # cuBLAS determinism (only takes effect for CUDA ops)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Torch deterministic algorithms (warn_only avoids hard failures on unsupported ops)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Avoid TF32 variability on Ampere/Hopper
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _relaxed_binary_assignment(logits2, temperature, hard, sampler="soft", logit_scale=1.0, center_logits=False):
    """
    Flexible relaxed binary sampler:
      - sampler='soft' or 'sigmoid': independent Bernoulli via sigmoid (probabilities in (0,1))
      - sampler='hard': deterministic 0/1 with straight-through gradients
    logit_scale rescales logits before sampling; center_logits subtracts mean (per-batch).
    """
    sampler_in = (sampler or "soft").lower()
    if sampler_in not in ("soft", "sigmoid", "hard"):
        raise ValueError(f"Unsupported sampler '{sampler_in}'. Use 'soft' or 'hard'.")

    use_hard = bool(hard) or sampler_in == "hard"
    logit_hw = logits2[:, 1]
    if center_logits:
        logit_hw = logit_hw - logit_hw.mean()
    logit_hw = logit_hw * float(logit_scale)
    prob_hw = torch.sigmoid(logit_hw / max(temperature, 1e-6))
    if use_hard:
        # Straight-through: forward is 0/1, gradients follow the sigmoid
        hard_mask = (prob_hw > 0.5).float()
        prob_hw = hard_mask.detach() + prob_hw - prob_hw.detach()
    probs2 = torch.stack([1.0 - prob_hw, prob_hw], dim=1)
    return probs2, prob_hw


def _build_torchgeo_data(TG, config=None):
    """
    Convert TaskGraph instance into torch_geometric Data with configurable
    node features and edge weights.

    feature_profile:
      - default: [sw, hw, area, degree, speedup, area_constraint, hw_scale] (min-max)
      - paper:   [HG, HGP] where HG=sw-hw, HGP=(sw-hw)/area (z-score), from Zheng et al.

    edge_weight_mode:
      - none
      - comm: normalized communication cost
      - paper_cosine: cosine similarity on node features, mapped to [0,1]
      - comm_cosine: average of normalized comm and cosine
      - auto: paper_cosine for feature_profile=paper, else none
    """
    cfg = dict(config) if isinstance(config, Mapping) else {}
    feature_profile = str(cfg.get("feature_profile", "default")).lower()
    edge_weight_mode = str(cfg.get("edge_weight_mode", "auto")).lower()
    if edge_weight_mode == "auto":
        edge_weight_mode = "paper_cosine" if feature_profile == "paper" else "none"
    make_undirected = bool(
        cfg.get(
            "make_undirected",
            feature_profile == "paper" or edge_weight_mode in {"paper_cosine", "comm_cosine"},
        )
    )

    G = TG.graph
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    sw = np.array([TG.software_costs[n] for n in node_list], dtype=np.float32)
    hw = np.array([TG.hardware_costs.get(n, 0.0) for n in node_list], dtype=np.float32)
    area = np.array([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=np.float32)
    area_safe = np.maximum(area, 1e-6)

    paper_hgp = None
    if feature_profile == "paper":
        hg = sw - hw
        hgp = hg / area_safe
        F = np.vstack([hg, hgp]).T  # (N,2)
        mean = F.mean(axis=0, keepdims=True)
        std = F.std(axis=0, keepdims=True)
        std = np.where(std <= 1e-9, 1.0, std)
        X_norm = (F - mean) / std
        # keep a [0,1] normalized HGP for optional output blending trick (paper-inspired).
        hgp_min, hgp_max = float(hgp.min()), float(hgp.max())
        hgp_range = max(hgp_max - hgp_min, 1e-6)
        paper_hgp = ((hgp - hgp_min) / hgp_range).astype(np.float32)
    else:
        deg = np.array([G.degree(n) for n in node_list], dtype=np.float32)
        speedup = hw / np.maximum(sw, 1e-6)
        area_constraint = float(getattr(TG, "area_constraint", 0.0))
        hw_scale = float(getattr(TG, "hw_scale_factor", getattr(TG, "HW_Scale_Factor", 1.0)))
        area_constraint_col = np.full_like(sw, area_constraint, dtype=np.float32)
        hw_scale_col = np.full_like(sw, hw_scale, dtype=np.float32)
        X = np.vstack([sw, hw, area, deg, speedup, area_constraint_col, hw_scale_col]).T
        mins = X.min(axis=0, keepdims=True)
        maxs = X.max(axis=0, keepdims=True)
        ranges = np.where(maxs - mins <= 1e-9, 1.0, (maxs - mins))
        X_norm = (X - mins) / ranges

    x = torch.tensor(X_norm, dtype=torch.float32)

    # Build edge list (optionally symmetrized for paper-style graph conv).
    if make_undirected:
        e_set = set()
        for u, v in G.edges():
            e_set.add((u, v))
            e_set.add((v, u))
        edges = sorted(e_set, key=lambda t: (node_to_idx[t[0]], node_to_idx[t[1]]))
    else:
        edges = list(G.edges())

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight_t = None
    else:
        src = [node_to_idx[u] for u, _ in edges]
        dst = [node_to_idx[v] for _, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_weight_t = None
        mode = edge_weight_mode
        if mode != "none":
            comm_raw = []
            for u, v in edges:
                c = float(TG.communication_costs.get((u, v), TG.communication_costs.get((v, u), 0.0)))
                comm_raw.append(max(c, 0.0))
            comm_raw = np.array(comm_raw, dtype=np.float32)
            comm_norm = comm_raw / max(float(comm_raw.max()), 1.0)

            if mode == "comm":
                ew = comm_norm
            else:
                # cosine similarity on selected node features
                vec = X_norm.astype(np.float32)
                norms = np.linalg.norm(vec, axis=1) + 1e-6
                cos_vals = []
                for u, v in edges:
                    i = node_to_idx[u]
                    j = node_to_idx[v]
                    cos = float(np.dot(vec[i], vec[j]) / (norms[i] * norms[j]))
                    cos_vals.append((cos + 1.0) * 0.5)  # map [-1,1] -> [0,1]
                cos_norm = np.clip(np.array(cos_vals, dtype=np.float32), 0.0, 1.0)
                if mode == "paper_cosine":
                    ew = cos_norm
                elif mode == "comm_cosine":
                    ew = 0.5 * comm_norm + 0.5 * cos_norm
                else:
                    raise ValueError(f"Unsupported edge_weight_mode '{edge_weight_mode}'.")

            # Keep tiny epsilon so message passing is not completely disabled on an edge.
            ew = np.clip(ew, 1e-4, None)
            edge_weight_t = torch.tensor(ew, dtype=torch.float32)

    data_kwargs = {"x": x, "edge_index": edge_index}
    if edge_weight_t is not None:
        data_kwargs["edge_weight"] = edge_weight_t
    data = Data(**data_kwargs)
    if paper_hgp is not None:
        data.paper_hgp = torch.tensor(paper_hgp, dtype=torch.float32)

    return data, node_list


def _repair_candidate(TG, x_continuous, node_list, prefer_by_score=None, max_iters=400):
    """
    Lightweight repair to enforce area budget / TG.violates on a continuous assignment.
    Returns a repaired binary numpy array.
    """
    x = np.asarray(x_continuous, dtype=float).ravel().copy()
    N = x.shape[0]
    x_bin = (x >= 0.5).astype(float)

    areas = np.array([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=float)
    area_budget = float(TG.area_constraint) * float(TG.total_area) if hasattr(TG, "total_area") else None
    if prefer_by_score is None:
        prefer_by_score = x

    # enforce area greedily
    if area_budget is not None and area_budget > 0:
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

    def _is_valid(arr):
        try:
            sol = {node_list[i]: int(arr[i] > 0.5) for i in range(N)}
            return TG.violates(sol) == 0
        except Exception:
            return True

    if _is_valid(x_bin):
        return x_bin

    # local search: try removing lowest-score hardware nodes first
    best = x_bin.copy()
    for _ in range(max_iters):
        cand = best.copy()
        hw_idx = np.where(cand > 0.5)[0]
        if hw_idx.size == 0:
            break
        i = hw_idx[np.argmin(prefer_by_score[hw_idx])]
        cand[i] = 0.0
        if _is_valid(cand):
            return cand
        best = cand

    return x_bin


def _fill_hw_area_by_score(TG, x_binary, node_list, prefer_by_score=None):
    """
    Greedily fill unused HW area budget by turning on high-score SW nodes when feasible.
    Used as a decode-time complement to _repair_candidate (which mainly removes HW nodes).
    """
    arr = np.asarray(x_binary, dtype=float).ravel().copy()
    N = arr.shape[0]
    if prefer_by_score is None:
        prefer_by_score = arr
    scores = np.asarray(prefer_by_score, dtype=float).ravel()
    if scores.shape[0] != N:
        scores = np.resize(scores, N)

    areas = np.array([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=float)
    if not hasattr(TG, "total_area"):
        return arr
    area_budget = float(TG.area_constraint) * float(TG.total_area)
    if area_budget <= 0:
        return np.zeros_like(arr)

    def _is_valid(a):
        try:
            sol = {node_list[i]: int(a[i] > 0.5) for i in range(N)}
            return TG.violates(sol) == 0
        except Exception:
            return True

    used = float(np.dot(areas, arr))
    if used >= area_budget - 1e-9:
        return arr

    sw_idx = np.where(arr <= 0.5)[0]
    if sw_idx.size == 0:
        return arr
    order = sw_idx[np.argsort(-scores[sw_idx])]
    for i in order:
        ai = float(areas[i])
        if ai <= 0:
            continue
        if used + ai > area_budget + 1e-12:
            continue
        cand = arr.copy()
        cand[i] = 1.0
        if _is_valid(cand):
            arr = cand
            used += ai
            if used >= area_budget - 1e-9:
                break

    return arr


def _resolve_regularizer_config(config, TG):
    """
    Resolve regularizer weights from config.
    Explicit coefficients in config always override profile defaults.

    Profiles:
      - legacy (default): old diff_gnn behavior
      - modern: lighter usage-balance, stronger partition-cost scaling
      - minimal: only makespan/area terms unless explicitly overridden
    """
    profile = str(config.get("regularizer_profile", "legacy")).lower()
    area_constraint = float(getattr(TG, "area_constraint", 0.0))

    if profile == "modern":
        defaults = {
            "entropy_coeff": 1e-3,
            "usage_balance_coeff": 0.0,
            "target_hw_frac": min(area_constraint, 0.3),
            "partition_cost_coeff": 5e-2 * max(1.0, area_constraint / 0.3),
        }
    elif profile == "minimal":
        defaults = {
            "entropy_coeff": 0.0,
            "usage_balance_coeff": 0.0,
            "target_hw_frac": min(area_constraint, 0.3),
            "partition_cost_coeff": 0.0,
        }
    else:
        defaults = {
            "entropy_coeff": 1e-3,
            "usage_balance_coeff": 0.5,
            "target_hw_frac": area_constraint,
            "partition_cost_coeff": 1e-2,
        }
        profile = "legacy"

    entropy_coeff = float(config.get("entropy_coeff", defaults["entropy_coeff"]))
    usage_balance_coeff = float(config.get("usage_balance_coeff", defaults["usage_balance_coeff"]))
    partition_cost_coeff = float(config.get("partition_cost_coeff", defaults["partition_cost_coeff"]))
    target_hw_frac = config.get("target_hw_frac", defaults["target_hw_frac"])
    if target_hw_frac is not None:
        target_hw_frac = float(target_hw_frac)

    return {
        "profile": profile,
        "entropy_coeff": entropy_coeff,
        "usage_balance_coeff": usage_balance_coeff,
        "partition_cost_coeff": partition_cost_coeff,
        "target_hw_frac": target_hw_frac,
    }


def _evaluate_discrete_solution(TG, solution, metric="legacy_lp"):
    """
    Evaluate a binary solution with selectable metric:
      - legacy_lp: historical diff_gnn evaluation via compute_dag_makespan
      - queue: TaskGraph queue simulator
    """
    if TG.violates(solution):
        return float(TG.violation_cost)

    metric_key = str(metric or "legacy_lp").lower()
    if metric_key in {"legacy_lp", "legacy", "lp", "dag_lp", "cvxpy"}:
        try:
            from utils.scheduler_utils import compute_dag_makespan
            graph = getattr(TG, "rounak_graph", None) or TG.graph
            lp_assignment = [1 - int(solution[n]) for n in graph.nodes()]
            makespan, _ = compute_dag_makespan(graph, lp_assignment)
            return float(makespan)
        except Exception as e:
            logger.warning("Legacy LP evaluation failed; falling back to queue metric: %s", str(e))

    try:
        return float(TG.evaluate_makespan(solution)["makespan"])
    except Exception as e:
        logger.warning("Queue evaluation failed: %s", str(e))
        return float("inf")


def _soft_project_area(
    probs: torch.Tensor,
    areas: torch.Tensor,
    area_budget: float,
    proj_iters: int = 4,
    proj_strength: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable-ish projection toward area-feasible assignments.
    Uses unrolled dual-style updates in probability/logit space, then a final
    smooth scaling cap to avoid large budget violations.
    """
    if area_budget <= 0:
        return torch.zeros_like(probs)

    p = probs.clamp(eps, 1.0 - eps)
    a = areas.clamp_min(0.0)
    a_mean = a.mean().clamp_min(eps)
    a_norm = a / a_mean

    for _ in range(max(1, int(proj_iters))):
        area_used = torch.dot(p, a)
        violation = F.relu((area_used - float(area_budget)) / float(max(area_budget, eps)))
        lam = float(proj_strength) * violation
        logits = torch.logit(p, eps=eps)
        p = torch.sigmoid(logits - lam * a_norm).clamp(eps, 1.0 - eps)

    area_used = torch.dot(p, a)
    scale = torch.clamp(float(area_budget) / (area_used + eps), max=1.0)
    return (p * scale).clamp(0.0, 1.0)


def _dls_refine_probs(
    TG,
    probs: torch.Tensor,
    node_list,
    steps: int = 2,
    flip_eta: float = 0.35,
    swap_eta: float = 0.18,
    score_temp: float = 0.7,
    comm_coeff: float = 0.02,
    area_proj_iters: int = 4,
    area_proj_strength: float = 6.0,
) -> torch.Tensor:
    """
    Differentiable local-search style refinement:
      1) soft-flip toward hardware desirability score
      2) soft-swap mass from low-value HW nodes to high-value SW nodes
      3) differentiable area projection each step
    """
    if steps <= 0:
        return probs

    device = probs.device
    dtype = probs.dtype
    p = probs.clamp(0.0, 1.0)

    sw_times = torch.tensor([TG.software_costs[n] for n in node_list], dtype=dtype, device=device)
    hw_times = torch.tensor([TG.hardware_costs[n] for n in node_list], dtype=dtype, device=device)
    areas = torch.tensor([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=dtype, device=device)
    area_budget = float(getattr(TG, "area_constraint", 0.0)) * float(max(getattr(TG, "total_area", 1.0), 1e-6))

    # Base desirability: positive means moving to HW is attractive.
    base_score = (sw_times - hw_times) / (sw_times.abs() + hw_times.abs() + 1e-6)

    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edge_pairs = [
        (node_to_idx[u], node_to_idx[v], float(c))
        for (u, v), c in TG.communication_costs.items()
        if u in node_to_idx and v in node_to_idx
    ]
    mean_comm = float(np.mean([c for _, _, c in edge_pairs])) if edge_pairs else 1.0
    mean_comm = max(mean_comm, 1e-6)

    eta_flip = float(max(0.0, min(1.0, flip_eta)))
    eta_swap = float(max(0.0, min(1.0, swap_eta)))
    temp = float(max(score_temp, 1e-6))

    for _ in range(max(1, int(steps))):
        comm_pressure = torch.zeros_like(p)
        for i, j, c in edge_pairs:
            w = float(comm_coeff) * (c / mean_comm)
            diff_ij = p[j] - p[i]
            comm_pressure[i] = comm_pressure[i] + w * diff_ij
            comm_pressure[j] = comm_pressure[j] - w * diff_ij

        score = base_score + comm_pressure
        target_hw = torch.sigmoid(score / temp)
        p = torch.lerp(p, target_hw, eta_flip)

        # Soft swap: reallocate HW "mass" from low-score HW nodes to high-score SW nodes.
        want_hw = (1.0 - p) * torch.sigmoid(score / temp)
        release_hw = p * torch.sigmoid(-score / temp)
        gain_weights = F.softmax(want_hw / temp, dim=0)
        release_weights = F.softmax(release_hw / temp, dim=0)
        transfer = eta_swap * 0.5 * (want_hw.mean() + release_hw.mean())
        p = (p + transfer * (gain_weights - release_weights)).clamp(0.0, 1.0)

        p = _soft_project_area(
            p,
            areas=areas,
            area_budget=area_budget,
            proj_iters=area_proj_iters,
            proj_strength=area_proj_strength,
        )

    return p.clamp(0.0, 1.0)


def _differentiable_makespan_loss(
    TG,
    probs_tensor,
    node_list,
    beta_softmax=20.0,
    area_penalty_coeff=1e5,
    entropy_coeff=1e-3,
    usage_balance_coeff=0.0,
    target_hw_frac=None,
    partition_cost_coeff=0.0,
):
    """
    Compute differentiable surrogate loss approximating makespan + penalties.
    - probs_tensor: (N,) values in [0,1] (probability of hardware)
    - Execution time per node = p*hw + (1-p)*sw (differentiable)
    - Comm delay between u->v approximated as |p_u - p_v| * comm_cost
    - Longest-path computed with DP over topo order, using soft-max (logsumexp with beta) to approximate max
    """
    device = probs_tensor.device
    N = probs_tensor.shape[0]
    # tensors of hw/sw/area/comm
    hw_times = torch.tensor([TG.hardware_costs[n] for n in node_list], dtype=torch.float32, device=device)
    sw_times = torch.tensor([TG.software_costs[n] for n in node_list], dtype=torch.float32, device=device)
    areas = torch.tensor([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=torch.float32, device=device)
    total_area = float(TG.total_area) if TG.total_area != 0 else float(areas.sum().item())
    if total_area <= 0:
        total_area = 1.0

    exec_time = probs_tensor * hw_times + (1.0 - probs_tensor) * sw_times  # (N,)
    # print(probs_tensor)

    # prepare predecessors list according to TG.graph topological order
    G = TG.graph
    topo = list(nx.topological_sort(G))
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    # Build adjacency of preds with comm costs as tensor
    preds = []
    for node in topo:
        pn = list(G.predecessors(node))
        if len(pn) == 0:
            preds.append([])
        else:
            preds.append([(node_to_idx[p], float(TG.communication_costs.get((p, node), 0.0))) for p in pn])

    # DP: dp[i] = exec_time[i] + softmax_beta( dp[pred] + comm_delay(pred,i) )
    dp = torch.zeros((len(topo),), dtype=torch.float32, device=device)
    for i, node in enumerate(topo):
        if len(preds[i]) == 0:
            dp[i] = exec_time[node_to_idx[node]]
        else:
            vals = []
            for (pidx, comm_cost) in preds[i]:
                # comm delay proportional to difference in prob
                comm = torch.abs(probs_tensor[pidx] - probs_tensor[node_to_idx[node]]) * comm_cost
                vals.append(dp[pidx] + comm)
            vals_t = torch.stack(vals)  # (k,)
            # soft maximum via logsumexp
            sm = (1.0 / beta_softmax) * torch.logsumexp(beta_softmax * vals_t, dim=0)
            dp[i] = exec_time[node_to_idx[node]] + sm

    # makespan approximate: soft-maximum across dp
    makespan_soft = (1.0 / beta_softmax) * torch.logsumexp(beta_softmax * dp, dim=0)

    # area penalty (differentiable): normalized area used
    area_used = torch.dot(probs_tensor, areas)
    area_frac = area_used / float(total_area)
    area_violation = F.relu(area_frac - float(TG.area_constraint))
    area_penalty = area_penalty_coeff * area_violation

    # optional balance penalty to push hardware usage toward a target fraction
    if target_hw_frac is None:
        # cap target to avoid pushing toward very large HW usage when area is loose
        target_hw_frac = min(float(TG.area_constraint), 0.3)
    usage_balance = usage_balance_coeff * (area_frac - float(target_hw_frac)) ** 2

    # entropy-like term to push probabilities to binary (smooth)
    entropy_like = torch.mean(probs_tensor * (1.0 - probs_tensor))

    # optional expected partition cost (execution + comm) to reduce total partition cost
    exec_cost = torch.sum(exec_time)
    comm_cost = torch.tensor(0.0, device=device)
    for (u, v), c in TG.communication_costs.items():
        if u in node_to_idx and v in node_to_idx:
            pu = probs_tensor[node_to_idx[u]]
            pv = probs_tensor[node_to_idx[v]]
            comm_cost = comm_cost + torch.abs(pu - pv) * float(c)
    expected_partition_cost = exec_cost + comm_cost

    loss = (
        makespan_soft
        + area_penalty
        + usage_balance
        + entropy_coeff * entropy_like
        + partition_cost_coeff * expected_partition_cost
    )
    return loss, {
        "makespan_surrogate": makespan_soft.item(),
        "area_frac": area_frac.item(),
        "area_penalty": area_penalty.item(),
        "usage_balance": usage_balance.item() if isinstance(usage_balance, torch.Tensor) else usage_balance,
        "entropy_like": entropy_like.item(),
        "expected_partition_cost": expected_partition_cost.item(),
        "loss": loss.item()
    }


def _train_with_relaxed_binary(TG, model, data, node_list, config, device):
    """
    Differentiable training loop:
        - GNN produces continuous logits
        - Relaxed sampler produces differentiable assignment (sigmoid by default, straight-through optional)
        - Surrogate loss penalizes area, communication and makespan
        - Adam optimizer updates the model
    """
    import random
    lr = float(config.get("lr", 1e-3))
    epochs = int(config.get("epochs", 1000))
    tau_start = float(config.get("tau_start", 1.0))
    tau_final = float(config.get("tau_final", 0.1))
    beta_softmax = float(config.get("beta_softmax", 20.0))
    area_penalty_coeff = float(config.get("area_penalty_coeff", 1e5))
    reg_cfg = _resolve_regularizer_config(config, TG)
    entropy_coeff = reg_cfg["entropy_coeff"]
    usage_balance_coeff = reg_cfg["usage_balance_coeff"]
    target_hw_frac = reg_cfg["target_hw_frac"]
    partition_cost_coeff = reg_cfg["partition_cost_coeff"]
    regularizer_profile = reg_cfg["profile"]
    selection_metric = str(config.get("selection_metric", "legacy_lp")).lower()
    seed = int(config.get("seed", 42))
    hard_eval_every = int(config.get("hard_eval_every", max(1, epochs // 10)))
    sampler = (config.get("sampling") or config.get("sampler") or "soft").lower()
    logit_scale = float(config.get("logit_scale", 8.0 if regularizer_profile == "legacy" else 3.0))
    center_logits = bool(config.get("center_logits", True))
    paper_sigma = float(config.get("paper_sigma", 0.0))
    paper_sigma = max(0.0, min(1.0, paper_sigma))
    # choose whether to make training assignments hard 0/1 (straight-through) or soft (sigmoid)
    hard_train_outputs = bool(config.get("hard_train_outputs", sampler != "soft"))
    post_cfg_raw = config.get("postprocess", {})
    post_cfg = dict(post_cfg_raw) if isinstance(post_cfg_raw, Mapping) else {}
    post_enabled = bool(post_cfg.get("enabled", config.get("lssp_postprocess_enabled", False)))
    post_mode = str(post_cfg.get("mode", config.get("postprocess_mode", "none"))).lower()
    if post_mode == "none" and post_enabled:
        post_mode = "lssp"
    if post_mode not in {"none", "dls", "lssp", "hybrid"}:
        raise ValueError(f"Unsupported postprocess_mode '{post_mode}'. Use none|dls|lssp|hybrid.")
    use_dls = post_mode in {"dls", "hybrid"}
    use_lssp = post_mode in {"lssp", "hybrid"}
    post_during_eval = bool(post_cfg.get("during_eval", config.get("lssp_postprocess_during_eval", False)))
    post_eval_mode = str(post_cfg.get("eval_mode", config.get("lssp_postprocess_eval", "taskgraph"))).lower()
    post_max_iters = int(post_cfg.get("max_iters", config.get("lssp_postprocess_max_iters", 64)))
    post_enable_area_fill = bool(post_cfg.get("enable_area_fill", config.get("lssp_postprocess_area_fill", True)))
    post_fill_allow_worsen = float(post_cfg.get("fill_allow_worsen", config.get("lssp_postprocess_fill_allow_worsen", 0.0)))
    post_enable_swap = bool(post_cfg.get("enable_swap", config.get("lssp_postprocess_enable_swap", True)))
    dls_steps = int(post_cfg.get("dls_steps", config.get("dls_steps", 2 if use_dls else 0)))
    dls_flip_eta = float(post_cfg.get("dls_flip_eta", config.get("dls_flip_eta", 0.35)))
    dls_swap_eta = float(post_cfg.get("dls_swap_eta", config.get("dls_swap_eta", 0.18)))
    dls_score_temp = float(post_cfg.get("dls_score_temp", config.get("dls_score_temp", 0.7)))
    dls_comm_coeff = float(post_cfg.get("dls_comm_coeff", config.get("dls_comm_coeff", 0.02)))
    dls_area_proj_iters = int(post_cfg.get("dls_area_proj_iters", config.get("dls_area_proj_iters", 4)))
    dls_area_proj_strength = float(post_cfg.get("dls_area_proj_strength", config.get("dls_area_proj_strength", 6.0)))
    dls_fill_decode = bool(post_cfg.get("dls_fill_decode", config.get("dls_fill_decode", True)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(
        "DiffGNN training: sampler=%s, epochs=%d, lr=%.2e, tau_start=%.2f->%.2f, reg_profile=%s, selection_metric=%s, post_mode=%s, feature_profile=%s, edge_weight_mode=%s, paper_sigma=%.2f, entropy_coeff=%.2e, usage_balance_coeff=%.2e, partition_cost_coeff=%.2e, logit_scale=%.2f, center_logits=%s, hard_train_outputs=%s",
        sampler,
        epochs,
        lr,
        tau_start,
        tau_final,
        regularizer_profile,
        selection_metric,
        post_mode,
        str(config.get("feature_profile", "default")),
        str(config.get("edge_weight_mode", "auto")),
        paper_sigma,
        entropy_coeff,
        usage_balance_coeff,
        partition_cost_coeff,
        logit_scale,
        str(center_logits),
        str(hard_train_outputs),
    )
    if use_dls:
        logger.info(
            "DiffGNN DLS enabled: steps=%d flip_eta=%.2f swap_eta=%.2f temp=%.2f comm_coeff=%.3f area_proj_iters=%d area_proj_strength=%.2f",
            dls_steps,
            dls_flip_eta,
            dls_swap_eta,
            dls_score_temp,
            dls_comm_coeff,
            dls_area_proj_iters,
            dls_area_proj_strength,
        )
    if use_lssp:
        logger.info(
            "DiffGNN postprocess enabled: eval_mode=%s during_eval=%s max_iters=%d area_fill=%s fill_allow_worsen=%.3f swap=%s",
            post_eval_mode,
            str(post_during_eval),
            post_max_iters,
            str(post_enable_area_fill),
            post_fill_allow_worsen,
            str(post_enable_swap),
        )

    best_sched_cost = float('inf')
    best_assign = None
    best_probs = None

    edge_weight = getattr(data, "edge_weight", None)
    paper_hgp = getattr(data, "paper_hgp", None)

    tau = tau_start
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits2 = model(data.x, data.edge_index, edge_weight=edge_weight)
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
        if use_dls and dls_steps > 0:
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
            )

        loss, info = _differentiable_makespan_loss(
            TG,
            probs,
            node_list,
            beta_softmax=beta_softmax,
            area_penalty_coeff=area_penalty_coeff,
            entropy_coeff=entropy_coeff,
            usage_balance_coeff=usage_balance_coeff,
            target_hw_frac=target_hw_frac,
            partition_cost_coeff=partition_cost_coeff,
        )
        loss.backward()
        optimizer.step()

        # anneal tau linearly
        tau = max(tau_final, tau_start - (ep / max(1, epochs)) * (tau_start - tau_final))

        # hard_eval_every = 1 # sid: remove this later
        # occasional hard evaluation to get discrete assignment
        if (ep % hard_eval_every == 0) or (ep == epochs):
            model.eval()
            with torch.no_grad():
                logits2_eval = model(data.x, data.edge_index, edge_weight=edge_weight)
                _, hard_probs_t = _relaxed_binary_assignment(
                    logits2_eval, temperature=max(tau, 1e-6), hard=True, sampler=sampler, logit_scale=logit_scale, center_logits=center_logits
                )
                if paper_sigma > 0 and paper_hgp is not None:
                    hard_probs_t = (1.0 - paper_sigma) * hard_probs_t + paper_sigma * paper_hgp.to(hard_probs_t.device, hard_probs_t.dtype)
                    hard_probs_t = hard_probs_t.clamp(0.0, 1.0)

                hard_probs = hard_probs_t.cpu().numpy().astype(float)  # 0/1
                if use_dls and dls_steps > 0:
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
                    )
                    hard_probs = hard_probs_t_refined.detach().cpu().numpy().astype(float)
                hard_probs_repaired = _repair_candidate(TG, hard_probs, node_list, prefer_by_score=hard_probs)
                if use_dls and dls_fill_decode:
                    hard_probs_repaired = _fill_hw_area_by_score(
                        TG,
                        hard_probs_repaired,
                        node_list=node_list,
                        prefer_by_score=hard_probs,
                    )
                solution = {node_list[i]: int(1 if hard_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
                if use_lssp and post_during_eval:
                    solution, _ = improve_with_lssp_local_search(
                        TG,
                        solution,
                        max_iters=post_max_iters,
                        eval_mode=post_eval_mode,
                        enable_area_fill=post_enable_area_fill,
                        fill_allow_worsen=post_fill_allow_worsen,
                        enable_swap=post_enable_swap,
                    )
                current_sched_cost = _evaluate_discrete_solution(TG, solution, metric=selection_metric)

                if current_sched_cost < best_sched_cost:
                    best_sched_cost = current_sched_cost
                    best_assign = solution.copy()
                    best_probs = hard_probs.copy()

        if ep % max(1, epochs // 10) == 0 or ep <= 5:
            logger.info(
                "Epoch %d/%d loss=%.6f surrogate_makespan=%.6f area_frac=%.4f area_pen=%.3f sched_best=%.6f",
                ep,
                epochs,
                info["loss"],
                info["makespan_surrogate"],
                info["area_frac"],
                info["area_penalty"],
                best_sched_cost,
            )

    # Final deterministic prediction using low-temperature hard sampling
    model.eval()
    with torch.no_grad():
        logits2 = model(data.x, data.edge_index, edge_weight=edge_weight)
        _, hard_probs_t = _relaxed_binary_assignment(
            logits2, temperature=0.1, hard=True, sampler=sampler, logit_scale=logit_scale, center_logits=center_logits
        )
        if paper_sigma > 0 and paper_hgp is not None:
            hard_probs_t = (1.0 - paper_sigma) * hard_probs_t + paper_sigma * paper_hgp.to(hard_probs_t.device, hard_probs_t.dtype)
            hard_probs_t = hard_probs_t.clamp(0.0, 1.0)
        final_probs = hard_probs_t.cpu().numpy().astype(float)
        if use_dls and dls_steps > 0:
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
            )
            final_probs = final_probs_t_refined.detach().cpu().numpy().astype(float)
        final_probs_repaired = _repair_candidate(TG, final_probs, node_list, prefer_by_score=final_probs)
        if use_dls and dls_fill_decode:
            final_probs_repaired = _fill_hw_area_by_score(
                TG,
                final_probs_repaired,
                node_list=node_list,
                prefer_by_score=final_probs,
            )
        final_solution = {node_list[i]: int(1 if final_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
        if use_lssp:
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
                "DiffGNN final postprocess: improved=%s cost=%.3f hw_area=%.3f/%.3f (%s)",
                str(post_info["improved"]),
                post_info["cost"],
                post_info["hw_area"],
                post_info["budget"],
                post_info["eval_mode"],
            )
        final_sched_cost = _evaluate_discrete_solution(TG, final_solution, metric=selection_metric)

    # choose best between tracked best and final
    if best_assign is None or final_sched_cost < best_sched_cost:
        best_assign = final_solution
        best_probs = final_probs
        best_sched_cost = final_sched_cost

    logger.info("Training finished. Best %s makespan: %.6f", selection_metric, best_sched_cost)
    return {
        "best_assign": best_assign,            # dict node -> {0,1}
        "best_probs": np.asarray(best_probs),  # numpy array 0/1
        "best_mip_cost": float(best_sched_cost),
        "model": model,
    }


def optimize_diff_gnn(TG, config=None, device='cpu'):
    """
    Public entry-point for differentiable GNN optimization.

    Implements the requested pipeline:
        - GNN produces continuous values in [0,1] (via relaxed sigmoid by default; set config['sampling']='hard' for straight-through 0/1)
        - Loss is fully differentiable, combining makespan surrogate, area, and entropy penalties
        - Adam performs gradient-based updates
        - Hard assignments are produced only for evaluation; training uses relaxed values

    Config options (defaults provided):
      - hidden_dim: 64
      - lr: 1e-3
      - epochs: 500
      - tau_start: 1.0
      - tau_final: 0.1
      - beta_softmax: 20.0
      - area_penalty_coeff: 1e5
      - entropy_coeff: 1e-3
      - usage_balance_coeff: 0.0 (set >0 to push hardware usage toward target_hw_frac)
      - target_hw_frac: defaults to min(area_constraint, 0.3)
      - partition_cost_coeff: 5e-2 (scaled up automatically when area_constraint is large)
      - regularizer_profile: legacy|modern|minimal (default: legacy)
      - selection_metric: legacy_lp|queue (default: legacy_lp)
      - postprocess_mode: none|dls|lssp|hybrid
      - dls_* knobs for differentiable soft flip/swap + area projection
      - seed: 42
      - hard_eval_every: epochs//10
      - sampling: 'soft' (default) or 'hard'
      - logit_scale: 3.0 (increase to sharpen sigmoid probabilities)
      - center_logits: False (set True to subtract batch mean before sigmoid)
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

    model = build_placement_model(
        model_name=model_name,
        in_channels=data.num_node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **model_extra_cfg,
    )

    logger.info("DiffGNN model selected: %s", model_name)
    print(model)

    return _train_with_relaxed_binary(TG, model, data, node_list, config, device)

def get_device(config):
    """
    Determine the correct device based on user config and PyTorch availability.

    Rules:
    - If config["device"] == "cpu": always use CPU.
    - If config["device"] in ["gpu", "cuda"]:
          use GPU only if torch.cuda.is_available(), else use CPU.
    - If config["device"] == "auto" or missing:
          use GPU if available, else CPU.
    """

    cfg_device = config.get("device", "auto").lower()

    if cfg_device == "cpu":
        return "cpu"

    if cfg_device in ("gpu", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"

    # default: auto-detect
    return "cuda" if torch.cuda.is_available() else "cpu"


def simulate_diff_GNN(dim, func_to_optimize, config):
    """
    Entry point compatible with gnn_main/MethodRegistry.
    Args:
        dim: number of nodes in the task graph (sanity check)
        func_to_optimize: bound TaskGraph method used as black-box evaluator
        config: full experiment config dict (expects optional 'diffgnn' sub-dict)

    Returns:
        (best_cost, best_solution_array) where best_solution_array has shape (dim,)
    """
    logger.info("Starting simulate_diff_GNN")

    TG = getattr(func_to_optimize, "__self__", None)
    
    print("Sid:::------Task graph: ", TG)

    if TG is None:
        msg = "func_to_optimize must be a bound TaskGraph method so the graph can be accessed."
        logger.error(msg)
        raise ValueError(msg)

    # pull diff GNN specific config (fallback to top-level keys for convenience)
    diff_cfg = dict(config.get("diffgnn", {}))
    # default diffgnn settings
    if "iter" not in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["iter"] = 1000
    if "verbose" not in diff_cfg:
        diff_cfg["verbose"] = 1000
    if "hidden_dim" not in diff_cfg:
        diff_cfg["hidden_dim"] = 256
    if "num_layers" not in diff_cfg:
        diff_cfg["num_layers"] = 3
    if "dropout" not in diff_cfg:
        diff_cfg["dropout"] = 0.5
    if "model" not in diff_cfg and "model_name" not in diff_cfg:
        diff_cfg["model"] = "default"
    # map common aliases
    if "iter" in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["epochs"] = diff_cfg["iter"]
    # ensure seed/device pass-through
    if "seed" not in diff_cfg and "seed" in config:
        diff_cfg["seed"] = config.get("seed")
    if "device" in diff_cfg:
        config["device"] = diff_cfg.get("device")
    elif "device" not in config:
        config["device"] = "gpu"

    # device = config.get("device", "cpu")
    device = get_device(config)
    logger.info("Using device: %s", device)
    print(f"[diff_gnn] device={device}")

    # Train differentiable GNN and retrieve best assignment
    result = optimize_diff_gnn(TG, config=diff_cfg, device=device)
    best_assign = result.get("best_assign", {})

    # Convert assignment dict -> array ordered by TG.node_to_num
    sol_arr = np.zeros(len(TG.graph.nodes()), dtype=np.float64)
    for node, idx in TG.node_to_num.items():
        sol_arr[idx] = float(best_assign.get(node, 0))

    # optional repair to align with constraint handling and evaluation
    node_list = list(TG.graph.nodes())
    sol_arr = _repair_candidate(TG, sol_arr, node_list, prefer_by_score=sol_arr)

    assert sol_arr.shape[0] == dim, f"dim ({dim}) != number of nodes ({sol_arr.shape[0]})"

    # Evaluate with the same metric used during training selection.
    solution = {node_list[i]: int(sol_arr[i] > 0.5) for i in range(len(node_list))}
    selection_metric = str(diff_cfg.get("selection_metric", "legacy_lp")).lower()
    eval_cost = _evaluate_discrete_solution(TG, solution, metric=selection_metric)

    # Prefer the MIP/LP makespan we already computed during training if available
    best_cost = float(result.get("best_mip_cost", eval_cost))
    if not math.isfinite(best_cost):
        best_cost = eval_cost

    logger.info(
        "simulate_diff_GNN finished: best_cost=%.6f (eval_cost=%.6f, train_best_cost=%.6f, metric=%s)",
        best_cost,
        eval_cost,
        result.get("best_mip_cost", float("nan")),
        selection_metric,
    )
    return best_cost, sol_arr
