import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import sys

# torch-geometric imports
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
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


# # Lightweight differentiable GNN for binary partitioning (hardware/software)
# class DiffGNN(nn.Module):
#     def __init__(self, in_channels=4, hidden_dim=64):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
#         self.lin = nn.Linear(hidden_dim // 2, 1)

#     def forward(self, x, edge_index):
#         """Standard two-layer GCN encoder followed by a single logit per node."""
#         h = F.relu(self.conv1(x, edge_index))
#         h = F.relu(self.conv2(h, edge_index))
#         logits = self.lin(h).squeeze(-1)  # (N,)
#         # produce two-class logits (class0=software, class1=hardware)
#         logits2 = torch.stack([-logits, logits], dim=1)  # (N,2)
#         return logits2

class DiffGNN(nn.Module):
    """
    Lightweight differentiable GNN for binary partitioning (software vs hardware).

    - num_layers controls how many GCNConv layers are used before the final linear head.
    - Output is (N, 2) logits: [software_logit, hardware_logit] per node.
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.dropout = dropout

        # Build GCN stack
        convs = []
        if num_layers == 1:
            # Single GCN layer directly into a smaller embedding size (like your original conv2 output)
            convs.append(GCNConv(in_channels, hidden_dim // 2))
            last_dim = hidden_dim // 2
        else:
            # First layer: in -> hidden_dim
            convs.append(GCNConv(in_channels, hidden_dim))
            # Middle/last GCN layers: keep same dim until final one reduces to hidden_dim//2
            for _ in range(num_layers - 2):
                convs.append(GCNConv(hidden_dim, hidden_dim))
            convs.append(GCNConv(hidden_dim, hidden_dim // 2))
            last_dim = hidden_dim // 2

        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(last_dim, 1)

    def forward(self, x, edge_index):
        """
        Returns:
            logits2: (N, 2) where [:,0]=software logit, [:,1]=hardware logit
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.lin(h).squeeze(-1)              # (N,)
        logits2 = torch.stack([-logits, logits], 1)   # (N,2)
        return logits2



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


def _build_torchgeo_data(TG):
    """
    Convert TaskGraph instance into torch and torch_geometric Data.
    Node features: [sw_time, hw_time, area, degree] normalized per-column.
    Returns Data(x=edge_index, ...) and node_list order.
    """
    G = TG.graph
    node_list = list(G.nodes())

    sw = np.array([TG.software_costs[n] for n in node_list], dtype=np.float32)
    hw = np.array([TG.hardware_costs.get(n, 0.0) for n in node_list], dtype=np.float32)
    area = np.array([TG.hardware_area.get(n, 0.0) for n in node_list], dtype=np.float32)
    deg = np.array([G.degree(n) for n in node_list], dtype=np.float32)

    X = np.vstack([sw, hw, area, deg]).T  # (N,4)
    mins = X.min(axis=0, keepdims=True)
    maxs = X.max(axis=0, keepdims=True)
    ranges = np.where(maxs - mins <= 1e-9, 1.0, (maxs - mins))
    X_norm = (X - mins) / ranges
    x = torch.tensor(X_norm, dtype=torch.float32)

    # build edge_index (directed edges as in NX DiGraph)
    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = [node_list.index(u) for u, v in edges]
        dst = [node_list.index(v) for u, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index), node_list


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
        target_hw_frac = float(TG.area_constraint)
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
    from utils.scheduler_utils import compute_dag_makespan

    lr = float(config.get("lr", 1e-3))
    epochs = int(config.get("epochs", 400))
    tau_start = float(config.get("tau_start", 1.0))
    tau_final = float(config.get("tau_final", 0.1))
    beta_softmax = float(config.get("beta_softmax", 20.0))
    area_penalty_coeff = float(config.get("area_penalty_coeff", 1e5))
    entropy_coeff = float(config.get("entropy_coeff", 1e-3))
    usage_balance_coeff = float(config.get("usage_balance_coeff", 0.5))
    target_hw_frac = config.get("target_hw_frac", None)
    partition_cost_coeff = float(config.get("partition_cost_coeff", 1e-2))
    seed = int(config.get("seed", 42))
    hard_eval_every = int(config.get("hard_eval_every", max(1, epochs // 10)))
    sampler = (config.get("sampling") or config.get("sampler") or "soft").lower()
    logit_scale = float(config.get("logit_scale", 8.0))
    center_logits = bool(config.get("center_logits", True))
    # choose whether to make training assignments hard 0/1 (straight-through) or soft (sigmoid)
    hard_train_outputs = bool(config.get("hard_train_outputs", sampler != "soft"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(
        "DiffGNN training: sampler=%s, epochs=%d, lr=%.2e, tau_start=%.2f->%.2f, entropy_coeff=%.2e, usage_balance_coeff=%.2e, partition_cost_coeff=%.2e, logit_scale=%.2f, center_logits=%s, hard_train_outputs=%s",
        sampler,
        epochs,
        lr,
        tau_start,
        tau_final,
        entropy_coeff,
        usage_balance_coeff,
        partition_cost_coeff,
        logit_scale,
        str(center_logits),
        str(hard_train_outputs),
    )

    best_mip_cost = float('inf')
    best_assign = None
    best_probs = None

    tau = tau_start
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits2 = model(data.x, data.edge_index)
        _, probs = _relaxed_binary_assignment(
            logits2,
            temperature=tau,
            hard=hard_train_outputs,
            sampler=sampler,
            logit_scale=logit_scale,
            center_logits=center_logits,
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
                logits2_eval = model(data.x, data.edge_index)
                _, hard_probs_t = _relaxed_binary_assignment(
                    logits2_eval, temperature=max(tau, 1e-6), hard=True, sampler=sampler, logit_scale=logit_scale, center_logits=center_logits
                )

                hard_probs = hard_probs_t.cpu().numpy().astype(float)  # 0/1
                hard_probs_repaired = _repair_candidate(TG, hard_probs, node_list, prefer_by_score=hard_probs)
                solution = {node_list[i]: int(1 if hard_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
                violation = TG.violates(solution)
                if violation:
                    current_mip_cost = TG.violation_cost
                else:
                    mip_assignment = [1 - solution[k] for k in TG.rounak_graph]
                    try:
                        makespan, _ = compute_dag_makespan(TG.rounak_graph, mip_assignment)
                        current_mip_cost = makespan
                    except Exception as e:
                        logger.warning("compute_dag_makespan failed during training eval: %s", str(e))
                        current_mip_cost = float('inf')

                if current_mip_cost < best_mip_cost:
                    best_mip_cost = current_mip_cost
                    best_assign = solution.copy()
                    best_probs = hard_probs.copy()

        if ep % max(1, epochs // 10) == 0 or ep <= 5:
            logger.info(
                "Epoch %d/%d loss=%.6f surrogate_makespan=%.6f area_frac=%.4f area_pen=%.3f mip_best=%.6f",
                ep,
                epochs,
                info["loss"],
                info["makespan_surrogate"],
                info["area_frac"],
                info["area_penalty"],
                best_mip_cost,
            )

    # Final deterministic prediction using low-temperature hard sampling
    model.eval()
    with torch.no_grad():
        logits2 = model(data.x, data.edge_index)
        _, hard_probs_t = _relaxed_binary_assignment(
            logits2, temperature=0.1, hard=True, sampler=sampler, logit_scale=logit_scale, center_logits=center_logits
        )
        final_probs = hard_probs_t.cpu().numpy().astype(float)
        final_probs_repaired = _repair_candidate(TG, final_probs, node_list, prefer_by_score=final_probs)
        final_solution = {node_list[i]: int(1 if final_probs_repaired[i] > 0.5 else 0) for i in range(len(node_list))}
        violation = TG.violates(final_solution)
        if violation:
            final_mip_cost = TG.violation_cost
        else:
            mip_assignment = [1 - final_solution[k] for k in TG.rounak_graph]
            try:
                final_makespan, _ = compute_dag_makespan(TG.rounak_graph, mip_assignment)
                final_mip_cost = final_makespan
            except Exception as e:
                logger.warning("compute_dag_makespan failed at final eval: %s", str(e))
                final_mip_cost = float('inf')

    # choose best between tracked best and final
    if best_assign is None or final_mip_cost < best_mip_cost:
        best_assign = final_solution
        best_probs = final_probs
        best_mip_cost = final_mip_cost

    logger.info("Training finished. Best MIP makespan: %.6f", best_mip_cost)
    return {
        "best_assign": best_assign,            # dict node -> {0,1}
        "best_probs": np.asarray(best_probs),  # numpy array 0/1
        "best_mip_cost": float(best_mip_cost),
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
      - target_hw_frac: defaults to area_constraint
      - partition_cost_coeff: 0.0 (set >0 to penalize expected partition cost directly)
      - seed: 42
      - hard_eval_every: epochs//10
      - sampling: 'soft' (default) or 'hard'
      - logit_scale: 1.0 (increase to sharpen sigmoid probabilities)
      - center_logits: False (set True to subtract batch mean before sigmoid)
    """
    if config is None:
        config = {}

    device = torch.device(device)
    data, node_list = _build_torchgeo_data(TG)

    data = data.to(device)
    num_layers = int(config.get("num_layers", 3))
    dropout = float(config.get("dropout", 0.5))
    hidden_dim = int(config.get("hidden_dim", 256))

    model = DiffGNN(
        in_channels=data.num_node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

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
    # map common aliases
    if "iter" in diff_cfg and "epochs" not in diff_cfg:
        diff_cfg["epochs"] = diff_cfg["iter"]
    # ensure seed/device pass-through
    if "seed" not in diff_cfg and "seed" in config:
        diff_cfg["seed"] = config.get("seed")

    # device = config.get("device", "cpu")
    device = get_device(config)
    logger.info("Using device: %s", device)

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

    # Evaluate with provided black-box function to align with registry expectations
    raw_cost = func_to_optimize(sol_arr)
    if isinstance(raw_cost, dict) and "makespan" in raw_cost:
        eval_cost = float(raw_cost["makespan"])
    elif isinstance(raw_cost, (list, tuple, np.ndarray)):
        eval_cost = float(np.asarray(raw_cost).ravel()[0])
    else:
        eval_cost = float(raw_cost)

    # Prefer the MIP/LP makespan we already computed during training if available
    best_cost = float(result.get("best_mip_cost", eval_cost))
    if not math.isfinite(best_cost):
        best_cost = eval_cost

    logger.info(
        "simulate_diff_GNN finished: best_cost=%.6f (eval_cost=%.6f, mip_cost=%.6f)",
        best_cost,
        eval_cost,
        result.get("best_mip_cost", float("nan")),
    )
    return best_cost, sol_arr
