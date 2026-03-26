import math
import time
from typing import Dict, Tuple

import networkx as nx
from meta_heuristic.partition_schedule_evaluator import compute_static_priorities, evaluate_partition_lssp


def _node_exec_time(TG, node: str, partition: Dict[str, int]) -> float:
    return float(TG.hardware_costs[node] if int(partition[node]) == 1 else TG.software_costs[node])


def _raw_edge_comm_time(TG, u: str, v: str) -> float:
    return float(TG.communication_costs.get((u, v), 0.0))


def _schedule_detail(TG, partition: Dict[str, int], eval_mode: str) -> Dict:
    mode = str(eval_mode).lower()
    if mode == "lssp":
        return evaluate_partition_lssp(TG, partition)
    return TG.evaluate_makespan(partition)


def _flip_exec_gain(TG, node: str, partition: Dict[str, int]) -> float:
    if int(partition[node]) == 1:
        return float(TG.hardware_costs[node] - TG.software_costs[node])
    return float(TG.software_costs[node] - TG.hardware_costs[node])


def _flip_comm_gain(TG, node: str, partition: Dict[str, int]) -> float:
    """
    Positive means the flip is expected to reduce boundary communication around the node.
    """
    gain = 0.0
    for pred in TG.graph.predecessors(node):
        c = _raw_edge_comm_time(TG, pred, node)
        if c <= 0:
            continue
        gain += c if int(partition[pred]) != int(partition[node]) else -c
    for succ in TG.graph.successors(node):
        c = _raw_edge_comm_time(TG, node, succ)
        if c <= 0:
            continue
        gain += c if int(partition[node]) != int(partition[succ]) else -c
    return gain


def _rank_local_search_nodes(
    TG,
    partition: Dict[str, int],
    eval_mode: str,
    search_strategy: str,
    candidate_top_k: int,
    critical_slack_frac: float,
    include_neighbors: bool,
    include_cut_endpoints: bool,
) -> Tuple[list[str], Dict[str, float]]:
    topo = list(nx.topological_sort(TG.graph))
    topo_idx = {n: i for i, n in enumerate(topo)}
    strategy = str(search_strategy).lower()
    if strategy == "all":
        return list(topo), {
            "candidate_pool_size": float(len(topo)),
            "selected_candidates": float(len(topo)),
            "schedule_makespan": 0.0,
        }

    schedule = _schedule_detail(TG, partition, eval_mode)
    makespan = float(schedule.get("makespan", 0.0))
    start_times = schedule.get("start_times", {}) or {}
    pri = compute_static_priorities(TG, partition)
    slack_thresh = max(1e-9, float(max(0.0, critical_slack_frac)) * max(makespan, 1.0))

    candidate_pool = set()
    for node in topo:
        through_time = float(start_times.get(node, 0.0)) + float(pri.get(node, _node_exec_time(TG, node, partition)))
        slack = max(0.0, makespan - through_time)
        if slack <= slack_thresh:
            candidate_pool.add(node)

    if include_cut_endpoints:
        for u, v in TG.graph.edges():
            if int(partition[u]) != int(partition[v]):
                candidate_pool.add(u)
                candidate_pool.add(v)

    if include_neighbors and candidate_pool:
        frontier = list(candidate_pool)
        for node in frontier:
            candidate_pool.update(TG.graph.predecessors(node))
            candidate_pool.update(TG.graph.successors(node))

    if not candidate_pool:
        candidate_pool = set(topo)

    ranked = []
    max_area = max((float(TG.hardware_area.get(n, 0.0)) for n in topo), default=1.0)
    max_area = max(max_area, 1e-6)
    for node in candidate_pool:
        through_time = float(start_times.get(node, 0.0)) + float(pri.get(node, _node_exec_time(TG, node, partition)))
        slack = max(0.0, makespan - through_time)
        flip_gain = _flip_exec_gain(TG, node, partition) + _flip_comm_gain(TG, node, partition)
        area_up = float(TG.hardware_area.get(node, 0.0)) if int(partition[node]) == 0 else 0.0
        ranked.append(((slack, -flip_gain, area_up / max_area, topo_idx[node]), node))

    ranked.sort(key=lambda item: item[0])
    ordered = [node for _, node in ranked]
    if candidate_top_k > 0:
        ordered = ordered[: max(1, int(candidate_top_k))]

    return ordered, {
        "candidate_pool_size": float(len(candidate_pool)),
        "selected_candidates": float(len(ordered)),
        "schedule_makespan": float(makespan),
    }


def _hardware_area(TG, partition: Dict[str, int]) -> float:
    return float(sum(TG.hardware_area[n] for n, a in partition.items() if int(a) == 1))


def _budget(TG) -> float:
    return float(TG.area_constraint) * float(TG.total_area)


def _cost(TG, partition: Dict[str, int], eval_mode: str) -> float:
    if TG.violates(partition):
        return float(TG.violation_cost)
    return float(_schedule_detail(TG, partition, eval_mode)["makespan"])


def _normalize_partition(partition: Dict[str, int]) -> Dict[str, int]:
    return {n: int(a) for n, a in partition.items()}


def improve_with_lssp_local_search(
    TG,
    partition: Dict[str, int],
    max_iters: int = 64,
    eval_mode: str = "taskgraph",
    enable_area_fill: bool = True,
    fill_allow_worsen: float = 0.0,
    enable_swap: bool = True,
    search_strategy: str = "critical",
    candidate_top_k: int = 16,
    critical_slack_frac: float = 0.05,
    candidate_include_neighbors: bool = True,
    candidate_include_cut_endpoints: bool = True,
) -> Tuple[Dict[str, int], Dict]:
    """
    Optional post-process:
    1) Greedy area-fill by SW->HW flips (uses cost + speedup tie-breakers).
    2) Local 1-flip / swap improvement under area constraint.

    The local neighborhood can be restricted to near-critical nodes instead of
    exhaustively scanning all flips. This keeps the search focused on moves that
    are more likely to change the makespan.
    """
    t0 = time.perf_counter()
    part = _normalize_partition(partition)
    budget = _budget(TG)
    eval_calls = 0
    stage1_iters = 0
    stage2_iters = 0
    candidate_builds = 0
    candidate_pool_total = 0.0
    selected_candidate_total = 0.0

    def _flip(base: Dict[str, int], node: str, value: int) -> Dict[str, int]:
        out = dict(base)
        out[node] = int(value)
        return out

    def _cost_count(p: Dict[str, int]) -> float:
        nonlocal eval_calls
        eval_calls += 1
        return _cost(TG, p, eval_mode)

    cur_cost = _cost_count(part)
    improved = False

    # Stage 1: consume unused HW area with useful flips when possible.
    if enable_area_fill:
        for _ in range(max(1, max_iters // 2)):
            stage1_iters += 1
            cur_area = _hardware_area(TG, part)
            remain = budget - cur_area
            if remain <= 1e-9:
                break

            best = None
            best_key = None
            for n in TG.graph.nodes():
                if part[n] == 1:
                    continue
                add_area = float(TG.hardware_area[n])
                if add_area > remain + 1e-9:
                    continue
                cand = _flip(part, n, 1)
                if TG.violates(cand):
                    continue
                c = _cost_count(cand)
                speedup = float(TG.software_costs[n] - TG.hardware_costs[n])
                delta = c - cur_cost
                # Prefer lower cost delta, then better speedup.
                key = (delta, -speedup)
                if best is None or key < best_key:
                    best = cand
                    best_key = key

            if best is None:
                break
            if best_key[0] > float(fill_allow_worsen):
                break
            part = best
            cur_cost = _cost_count(part)
            improved = True

    strategy = str(search_strategy).lower()
    if strategy not in {"all", "critical"}:
        raise ValueError(f"Unsupported local-search strategy '{search_strategy}'. Use all|critical.")

    # Stage 2: local search by 1-flip and swap.
    for _ in range(max_iters):
        stage2_iters += 1
        best_part = None
        best_cost = cur_cost

        nodes, candidate_info = _rank_local_search_nodes(
            TG,
            part,
            eval_mode=eval_mode,
            search_strategy=strategy,
            candidate_top_k=int(candidate_top_k),
            critical_slack_frac=float(critical_slack_frac),
            include_neighbors=bool(candidate_include_neighbors),
            include_cut_endpoints=bool(candidate_include_cut_endpoints),
        )
        candidate_builds += 1
        candidate_pool_total += float(candidate_info.get("candidate_pool_size", 0.0))
        selected_candidate_total += float(candidate_info.get("selected_candidates", 0.0))

        for n in nodes:
            cand = _flip(part, n, 1 - part[n])
            if TG.violates(cand):
                continue
            c = _cost_count(cand)
            if c + 1e-9 < best_cost:
                best_cost = c
                best_part = cand

        if enable_swap:
            hw_nodes = [n for n in nodes if part[n] == 1]
            sw_nodes = [n for n in nodes if part[n] == 0]
            for h in hw_nodes:
                for s in sw_nodes:
                    cand = dict(part)
                    cand[h] = 0
                    cand[s] = 1
                    if TG.violates(cand):
                        continue
                    c = _cost_count(cand)
                    if c + 1e-9 < best_cost:
                        best_cost = c
                        best_part = cand

        if best_part is None:
            break
        part = best_part
        cur_cost = best_cost
        improved = True

    elapsed = time.perf_counter() - t0
    total_iters = stage1_iters + stage2_iters
    return part, {
        "cost": float(cur_cost),
        "hw_area": float(_hardware_area(TG, part)),
        "budget": float(budget),
        "improved": bool(improved),
        "eval_mode": str(eval_mode),
        "elapsed_sec": float(elapsed),
        "eval_calls": int(eval_calls),
        "stage1_iters": int(stage1_iters),
        "stage2_iters": int(stage2_iters),
        "search_strategy": strategy,
        "candidate_top_k": int(candidate_top_k),
        "critical_slack_frac": float(critical_slack_frac),
        "candidate_builds": int(candidate_builds),
        "avg_candidate_pool": float(candidate_pool_total / candidate_builds) if candidate_builds > 0 else 0.0,
        "avg_selected_candidates": float(selected_candidate_total / candidate_builds) if candidate_builds > 0 else 0.0,
        "avg_eval_ms": float((elapsed / eval_calls) * 1000.0) if eval_calls > 0 else 0.0,
        "avg_iter_ms": float((elapsed / total_iters) * 1000.0) if total_iters > 0 else 0.0,
    }
