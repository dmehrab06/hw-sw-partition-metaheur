from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping

import networkx as nx


def _normalize_partition(
    graph_nodes: Iterable[Hashable],
    partition_assignment: Mapping[Hashable, int | float | str],
) -> Dict[Hashable, int]:
    normalized: Dict[Hashable, int] = {}
    graph_node_set = set(graph_nodes)

    for node in graph_node_set:
        if node not in partition_assignment:
            raise ValueError(f"Partition not specified for node {node}")

        value = partition_assignment[node]
        if isinstance(value, str):
            vv = value.strip().lower()
            if vv in {"hardware", "hw", "1"}:
                normalized[node] = 1
                continue
            if vv in {"software", "sw", "0"}:
                normalized[node] = 0
                continue
            raise ValueError(f"Unsupported partition value for node {node}: {value!r}")

        ivalue = int(value)
        if ivalue not in (0, 1):
            raise ValueError(f"Unsupported partition value for node {node}: {value!r}")
        normalized[node] = ivalue

    return normalized


def _node_exec_time(task_graph, node: Hashable, partition: Dict[Hashable, int]) -> float:
    if partition[node] == 1:
        return float(task_graph.hardware_costs[node])
    return float(task_graph.software_costs[node])


def _edge_comm_time(task_graph, src: Hashable, dst: Hashable, partition: Dict[Hashable, int]) -> float:
    if partition[src] == partition[dst]:
        return 0.0
    return float(task_graph.communication_costs.get((src, dst), 0.0))


def _compute_static_priorities_core(
    task_graph,
    partition: Dict[Hashable, int],
    topo_order: list[Hashable],
) -> Dict[Hashable, float]:
    priorities: Dict[Hashable, float] = {}
    graph = task_graph.graph

    for node in reversed(topo_order):
        exec_t = _node_exec_time(task_graph, node, partition)
        successors = list(graph.successors(node))
        if not successors:
            priorities[node] = exec_t
            continue

        best_tail = -1.0
        for succ in successors:
            comm = _edge_comm_time(task_graph, node, succ, partition)
            candidate = comm + priorities[succ]
            if candidate > best_tail:
                best_tail = candidate
        priorities[node] = exec_t + max(0.0, best_tail)

    return priorities


def compute_static_priorities(
    task_graph,
    partition_assignment: Mapping[Hashable, int | float | str],
) -> Dict[Hashable, float]:
    """
    LSSP-style static priorities:
    Pri(node) = longest path from node to sink, including node execution time and
    boundary communication costs.
    """
    graph = task_graph.graph
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("LSSP scheduling requires a DAG task graph.")

    partition = _normalize_partition(graph.nodes(), partition_assignment)
    topo = list(nx.topological_sort(graph))
    return _compute_static_priorities_core(task_graph, partition, topo)


def evaluate_makespan_lssp(
    task_graph,
    partition_assignment: Mapping[Hashable, int | float | str],
) -> Dict[str, object]:
    """
    Priority-list scheduling with static priorities (LSSP-style):
    - hardware tasks execute in parallel once dependencies are met
    - software tasks execute on one serialized processor
    - communication delay applies only when predecessor/successor are on different partitions
    """
    graph = task_graph.graph
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("LSSP scheduling requires a DAG task graph.")

    partition = _normalize_partition(graph.nodes(), partition_assignment)
    topo = list(nx.topological_sort(graph))
    topo_idx = {node: i for i, node in enumerate(topo)}
    priorities = _compute_static_priorities_core(task_graph, partition, topo)

    start_times: Dict[Hashable, float] = {}
    finish_times: Dict[Hashable, float] = {}
    scheduled: set[Hashable] = set()
    sw_available = 0.0

    def earliest_start(node: Hashable) -> float:
        est = 0.0
        for pred in graph.predecessors(node):
            pred_finish = finish_times[pred] + _edge_comm_time(task_graph, pred, node, partition)
            if pred_finish > est:
                est = pred_finish
        return est

    while len(scheduled) < len(topo):
        ready = [
            node
            for node in topo
            if node not in scheduled and all(pred in scheduled for pred in graph.predecessors(node))
        ]
        if not ready:
            raise RuntimeError("LSSP scheduler stalled: no ready nodes in remaining subgraph.")

        ready.sort(key=lambda node: (-priorities[node], topo_idx[node]))
        for node in ready:
            est = earliest_start(node)
            if partition[node] == 0:
                st = max(est, sw_available)
                ft = st + float(task_graph.software_costs[node])
                sw_available = ft
            else:
                st = est
                ft = st + float(task_graph.hardware_costs[node])
            start_times[node] = st
            finish_times[node] = ft
            scheduled.add(node)

    makespan = max(finish_times.values()) if finish_times else 0.0
    hw_nodes = [node for node, side in partition.items() if side == 1]
    sw_nodes = [node for node, side in partition.items() if side == 0]
    hw_finish_times = [finish_times[node] for node in hw_nodes if node in finish_times]
    sw_finish_times = [finish_times[node] for node in sw_nodes if node in finish_times]
    hw_makespan = max(hw_finish_times) if hw_finish_times else 0.0
    sw_makespan = max(sw_finish_times) if sw_finish_times else 0.0

    total_comm_delay = 0.0
    active_comm_edges = []
    for src, dst in graph.edges():
        if partition[src] != partition[dst]:
            comm = float(task_graph.communication_costs.get((src, dst), 0.0))
            total_comm_delay += comm
            if comm > 0.0:
                active_comm_edges.append((src, dst, comm))

    area_used = float(sum(task_graph.hardware_area.get(node, 0.0) for node in hw_nodes))
    is_violation = bool(task_graph.violates(partition))
    if is_violation:
        violation_cost = float(getattr(task_graph, "violation_cost", 1e9))
        makespan = violation_cost
        hw_makespan = violation_cost
        sw_makespan = violation_cost
        total_comm_delay = violation_cost

    result = {
        "makespan": float(makespan),
        "start_times": start_times,
        "finish_times": finish_times,
        "end_times": finish_times,
        "hardware_nodes": hw_nodes,
        "software_nodes": sw_nodes,
        "hardware_makespan": float(hw_makespan),
        "software_makespan": float(sw_makespan),
        "total_communication_delay": float(total_comm_delay),
        "active_communication_edges": active_comm_edges,
        "static_priorities": priorities,
        "execution_summary": {
            "total_nodes": len(graph.nodes()),
            "completed_nodes": len(scheduled),
            "hardware_execution_time": float(hw_makespan),
            "software_execution_time": float(sw_makespan),
            "communication_overhead": float(total_comm_delay),
            "total_makespan": float(makespan),
            "area_used": area_used,
            "area_constraint": float(getattr(task_graph, "area_constraint", 0.0)),
            "violation": is_violation,
        },
    }
    return result


__all__ = ["compute_static_priorities", "evaluate_makespan_lssp"]
