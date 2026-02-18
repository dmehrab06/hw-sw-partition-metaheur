import math
from typing import Dict, Tuple

import networkx as nx


def _node_exec_time(TG, node: str, partition: Dict[str, int]) -> float:
    return float(TG.hardware_costs[node] if int(partition[node]) == 1 else TG.software_costs[node])


def _edge_comm_time(TG, u: str, v: str, partition: Dict[str, int]) -> float:
    if int(partition[u]) == int(partition[v]):
        return 0.0
    return float(TG.communication_costs.get((u, v), 0.0))


def compute_static_priorities(TG, partition: Dict[str, int]) -> Dict[str, float]:
    """
    LSSP-style static priorities:
    Pri(node) = longest path length from node to any sink, including node execution
    and cross-partition communication costs on edges.
    """
    G = TG.graph
    topo = list(nx.topological_sort(G))
    pri: Dict[str, float] = {}

    for node in reversed(topo):
        exec_t = _node_exec_time(TG, node, partition)
        succs = list(G.successors(node))
        if not succs:
            pri[node] = exec_t
            continue
        best_tail = -1.0
        for succ in succs:
            comm = _edge_comm_time(TG, node, succ, partition)
            cand = comm + pri[succ]
            if cand > best_tail:
                best_tail = cand
        pri[node] = exec_t + max(0.0, best_tail)

    return pri


def evaluate_makespan_lssp(TG, partition: Dict[str, int]) -> Dict:
    """
    Priority-list scheduling with static priorities (LSSP-style):
    - hardware tasks: run in parallel once dependencies are satisfied
    - software tasks: single processor (serialized), priority-ordered among ready tasks
    - communication: charged only across HW/SW boundary
    """
    G = TG.graph
    pri = compute_static_priorities(TG, partition)
    topo = list(nx.topological_sort(G))
    topo_idx = {n: i for i, n in enumerate(topo)}

    start_times: Dict[str, float] = {}
    finish_times: Dict[str, float] = {}
    scheduled = set()
    sw_available = 0.0

    def earliest_start(node: str) -> float:
        t = 0.0
        for p in G.predecessors(node):
            tf = finish_times[p]
            tf += _edge_comm_time(TG, p, node, partition)
            if tf > t:
                t = tf
        return t

    while len(scheduled) < len(topo):
        ready = [
            n for n in topo
            if n not in scheduled and all(p in scheduled for p in G.predecessors(n))
        ]
        if not ready:
            break

        # Higher priority first; topo index stabilizes tie-breaks.
        ready.sort(key=lambda n: (-pri[n], topo_idx[n]))

        for node in ready:
            est = earliest_start(node)
            if int(partition[node]) == 0:
                st = max(est, sw_available)
                ft = st + float(TG.software_costs[node])
                sw_available = ft
            else:
                st = est
                ft = st + float(TG.hardware_costs[node])
            start_times[node] = st
            finish_times[node] = ft
            scheduled.add(node)

    makespan = max(finish_times.values()) if finish_times else 0.0
    return {"makespan": makespan, "start_times": start_times, "finish_times": finish_times}


def _hardware_area(TG, partition: Dict[str, int]) -> float:
    return float(sum(TG.hardware_area[n] for n, a in partition.items() if int(a) == 1))


def _budget(TG) -> float:
    return float(TG.area_constraint) * float(TG.total_area)


def _cost(TG, partition: Dict[str, int], eval_mode: str) -> float:
    if TG.violates(partition):
        return float(TG.violation_cost)
    mode = str(eval_mode).lower()
    if mode == "lssp":
        return float(evaluate_makespan_lssp(TG, partition)["makespan"])
    return float(TG.evaluate_makespan(partition)["makespan"])


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
) -> Tuple[Dict[str, int], Dict]:
    """
    Optional post-process:
    1) Greedy area-fill by SW->HW flips (uses cost + speedup tie-breakers).
    2) Local 1-flip / swap improvement under area constraint.
    """
    part = _normalize_partition(partition)
    budget = _budget(TG)

    def _flip(base: Dict[str, int], node: str, value: int) -> Dict[str, int]:
        out = dict(base)
        out[node] = int(value)
        return out

    cur_cost = _cost(TG, part, eval_mode)
    improved = False

    # Stage 1: consume unused HW area with useful flips when possible.
    if enable_area_fill:
        for _ in range(max(1, max_iters // 2)):
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
                c = _cost(TG, cand, eval_mode)
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
            cur_cost = _cost(TG, part, eval_mode)
            improved = True

    # Stage 2: local search by 1-flip and swap.
    for _ in range(max_iters):
        best_part = None
        best_cost = cur_cost

        nodes = list(TG.graph.nodes())
        for n in nodes:
            cand = _flip(part, n, 1 - part[n])
            if TG.violates(cand):
                continue
            c = _cost(TG, cand, eval_mode)
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
                    c = _cost(TG, cand, eval_mode)
                    if c + 1e-9 < best_cost:
                        best_cost = c
                        best_part = cand

        if best_part is None:
            break
        part = best_part
        cur_cost = best_cost
        improved = True

    return part, {
        "cost": float(cur_cost),
        "hw_area": float(_hardware_area(TG, part)),
        "budget": float(budget),
        "improved": bool(improved),
        "eval_mode": str(eval_mode),
    }

