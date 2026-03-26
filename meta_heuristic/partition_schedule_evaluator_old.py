from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any, Hashable
import math
import pickle
import random
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PartitionScheduleProblem:
    graph: nx.DiGraph
    hardware_costs: dict[Hashable, float]
    software_costs: dict[Hashable, float]
    hardware_area: dict[Hashable, float]
    communication_costs: dict[tuple[Hashable, Hashable], float]
    area_constraint: float
    total_area: float
    violation_cost: float


def _resolve_runtime_overrides(
    config_or_area_constraint: Mapping[str, Any] | float | int | None,
    violation_cost: float | None = None,
) -> tuple[float | None, float | None]:
    area_override = None
    violation_override = violation_cost

    if isinstance(config_or_area_constraint, Mapping):
        if "area-constraint" in config_or_area_constraint:
            area_override = float(config_or_area_constraint["area-constraint"])
        elif "area_constraint" in config_or_area_constraint:
            area_override = float(config_or_area_constraint["area_constraint"])

        if violation_override is None:
            if "violation-cost" in config_or_area_constraint:
                violation_override = float(config_or_area_constraint["violation-cost"])
            elif "violation_cost" in config_or_area_constraint:
                violation_override = float(config_or_area_constraint["violation_cost"])
    elif config_or_area_constraint is not None:
        area_override = float(config_or_area_constraint)

    return area_override, violation_override


def synchronize_problem_with_config(
    problem_or_task_graph: Any,
    config_or_area_constraint: Mapping[str, Any] | float | int | None,
    violation_cost: float | None = None,
):
    """
    Align runtime scheduling constraints with the active config.

    This is primarily used when a TaskGraph is loaded from pickle and may carry
    stale metadata such as an older area constraint.
    """
    area_override, violation_override = _resolve_runtime_overrides(
        config_or_area_constraint=config_or_area_constraint,
        violation_cost=violation_cost,
    )
    if area_override is None and violation_override is None:
        return problem_or_task_graph

    if isinstance(problem_or_task_graph, PartitionScheduleProblem):
        updates = {}
        if area_override is not None:
            updates["area_constraint"] = float(area_override)
        if violation_override is not None:
            updates["violation_cost"] = float(violation_override)
        if not updates:
            return problem_or_task_graph
        return replace(problem_or_task_graph, **updates)

    if area_override is not None and hasattr(problem_or_task_graph, "area_constraint"):
        problem_or_task_graph.area_constraint = float(area_override)
    if violation_override is not None and hasattr(problem_or_task_graph, "violation_cost"):
        problem_or_task_graph.violation_cost = float(violation_override)

    if getattr(problem_or_task_graph, "total_area", None) in (None, 0, 0.0):
        hardware_area = getattr(problem_or_task_graph, "hardware_area", None)
        if hardware_area:
            problem_or_task_graph.total_area = float(sum(float(v) for v in hardware_area.values()))

    return problem_or_task_graph


def build_problem(task_graph_or_problem: Any) -> PartitionScheduleProblem:
    if isinstance(task_graph_or_problem, PartitionScheduleProblem):
        return task_graph_or_problem

    graph = getattr(task_graph_or_problem, "graph", task_graph_or_problem)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("build_problem expects a TaskGraph-like object or a networkx.DiGraph.")

    hardware_costs_attr = getattr(task_graph_or_problem, "hardware_costs", {})
    software_costs_attr = getattr(task_graph_or_problem, "software_costs", {})
    hardware_area_attr = getattr(task_graph_or_problem, "hardware_area", {})
    communication_costs_attr = getattr(task_graph_or_problem, "communication_costs", {})

    hardware_costs = {
        node: float(
            hardware_costs_attr.get(node, graph.nodes[node].get("hardware_time", 0.0))
        )
        for node in graph.nodes()
    }
    software_costs = {
        node: float(
            software_costs_attr.get(node, graph.nodes[node].get("software_time", 0.0))
        )
        for node in graph.nodes()
    }
    hardware_area = {
        node: float(
            hardware_area_attr.get(node, graph.nodes[node].get("area_cost", 0.0))
        )
        for node in graph.nodes()
    }
    communication_costs = {
        (u, v): float(
            communication_costs_attr.get((u, v), graph.edges[u, v].get("communication_cost", 0.0))
        )
        for u, v in graph.edges()
    }

    total_area_attr = getattr(task_graph_or_problem, "total_area", None)
    total_area = float(total_area_attr) if total_area_attr is not None else float(sum(hardware_area.values()))
    if total_area <= 0.0:
        total_area = float(sum(hardware_area.values()))

    area_constraint = float(getattr(task_graph_or_problem, "area_constraint", 1.0))
    violation_cost = float(getattr(task_graph_or_problem, "violation_cost", 1e9))

    return PartitionScheduleProblem(
        graph=graph,
        hardware_costs=hardware_costs,
        software_costs=software_costs,
        hardware_area=hardware_area,
        communication_costs=communication_costs,
        area_constraint=area_constraint,
        total_area=total_area,
        violation_cost=violation_cost,
    )


def _require_dag(problem: PartitionScheduleProblem) -> list[Hashable]:
    if not nx.is_directed_acyclic_graph(problem.graph):
        raise ValueError("Partition scheduling requires a DAG task graph.")
    return list(nx.topological_sort(problem.graph))


def _node_label(node: Hashable) -> str:
    return str(node)


def _normalize_partition_value(value: Any) -> int:
    if isinstance(value, str):
        vv = value.strip().lower()
        if vv in {"hardware", "hw", "1"}:
            return 1
        if vv in {"software", "sw", "0"}:
            return 0
        raise ValueError(f"Unsupported partition value: {value!r}")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            fv = float(value)
            if fv <= 0.0:
                return 0
            if fv >= 1.0:
                return 1
            return 1 if fv > 0.5 else 0

    raise ValueError(f"Unsupported partition value: {value!r}")


def _normalize_partition(
    problem: PartitionScheduleProblem,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
) -> dict[Hashable, int]:
    nodes = list(problem.graph.nodes())

    if isinstance(partition_assignment, Mapping):
        partition = {}
        for node in nodes:
            if node not in partition_assignment:
                raise ValueError(f"Partition not specified for node {node}")
            partition[node] = _normalize_partition_value(partition_assignment[node])
        return partition

    if isinstance(partition_assignment, Sequence) and not isinstance(partition_assignment, (str, bytes)):
        if len(partition_assignment) != len(nodes):
            raise ValueError(
                f"Partition length ({len(partition_assignment)}) does not match number of nodes ({len(nodes)})."
            )
        return {
            node: _normalize_partition_value(partition_assignment[idx])
            for idx, node in enumerate(nodes)
        }

    raise TypeError("Partition assignment must be a mapping or a sequence aligned with graph.nodes().")


def _area_budget(problem: PartitionScheduleProblem) -> float:
    if problem.total_area <= 0.0:
        return float("inf")
    return max(0.0, float(problem.area_constraint)) * float(problem.total_area)


def _hardware_area_used(problem: PartitionScheduleProblem, partition: Mapping[Hashable, int]) -> float:
    return float(
        sum(problem.hardware_area.get(node, 0.0) for node, side in partition.items() if int(side) == 1)
    )


def _partition_is_valid(problem: PartitionScheduleProblem, partition: Mapping[Hashable, int]) -> bool:
    if problem.total_area <= 0.0:
        return True
    return _hardware_area_used(problem, partition) <= _area_budget(problem) + 1e-9


def _node_exec_time(problem: PartitionScheduleProblem, node: Hashable, partition: Mapping[Hashable, int]) -> float:
    if int(partition[node]) == 1:
        return float(problem.hardware_costs.get(node, 0.0))
    return float(problem.software_costs.get(node, 0.0))


def _edge_comm_time(
    problem: PartitionScheduleProblem,
    src: Hashable,
    dst: Hashable,
    partition: Mapping[Hashable, int],
) -> float:
    if int(partition[src]) == int(partition[dst]):
        return 0.0
    return float(problem.communication_costs.get((src, dst), 0.0))


def compute_static_priorities(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
) -> dict[Hashable, float]:
    problem = build_problem(problem_or_task_graph)
    topo = _require_dag(problem)
    partition = _normalize_partition(problem, partition_assignment)

    priorities: dict[Hashable, float] = {}
    for node in reversed(topo):
        exec_t = _node_exec_time(problem, node, partition)
        successors = list(problem.graph.successors(node))
        if not successors:
            priorities[node] = exec_t
            continue

        best_tail = max(
            (
                _edge_comm_time(problem, node, succ, partition) + priorities[succ]
                for succ in successors
            ),
            default=0.0,
        )
        priorities[node] = exec_t + max(0.0, best_tail)
    return priorities


def make_partition_valid(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    problem = build_problem(problem_or_task_graph)
    topo = _require_dag(problem)
    topo_idx = {node: idx for idx, node in enumerate(topo)}
    partition = _normalize_partition(problem, partition_assignment)
    original_partition = dict(partition)

    strategy_key = str(strategy or "benefit_per_area").strip().lower()
    if strategy_key != "benefit_per_area":
        raise ValueError(f"Unsupported repair strategy: {strategy}")

    if _partition_is_valid(problem, partition):
        area_used = _hardware_area_used(problem, partition)
        return {
            "partition": partition,
            "original_partition": original_partition,
            "was_repaired": False,
            "repaired_nodes": [],
            "is_valid": True,
            "repair_strategy": strategy_key,
            "area_used": area_used,
            "area_budget": _area_budget(problem),
        }

    repaired_nodes: list[Hashable] = []
    area_budget = _area_budget(problem)

    while not _partition_is_valid(problem, partition):
        removable_nodes = [node for node in topo if int(partition[node]) == 1]
        if not removable_nodes:
            break

        def _repair_key(node: Hashable) -> tuple[float, float, float, int, str]:
            area = float(problem.hardware_area.get(node, 0.0))
            benefit = float(problem.software_costs.get(node, 0.0) - problem.hardware_costs.get(node, 0.0))
            ratio = benefit / area if area > 1e-12 else math.inf
            zero_area_rank = 1.0 if area <= 1e-12 else 0.0
            return (
                zero_area_rank,
                ratio,
                abs(benefit),
                topo_idx[node],
                _node_label(node),
            )

        chosen = min(removable_nodes, key=_repair_key)
        if float(problem.hardware_area.get(chosen, 0.0)) <= 1e-12 and not math.isinf(area_budget):
            break
        partition[chosen] = 0
        repaired_nodes.append(chosen)

    area_used = _hardware_area_used(problem, partition)
    return {
        "partition": partition,
        "original_partition": original_partition,
        "was_repaired": bool(repaired_nodes),
        "repaired_nodes": repaired_nodes,
        "is_valid": _partition_is_valid(problem, partition),
        "repair_strategy": strategy_key,
        "area_used": area_used,
        "area_budget": area_budget,
    }


def _prepare_partition(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    auto_repair: bool,
    repair_strategy: str,
) -> tuple[PartitionScheduleProblem, dict[Hashable, int], dict[Hashable, int], dict[str, Any]]:
    problem = build_problem(problem_or_task_graph)
    original_partition = _normalize_partition(problem, partition_assignment)

    if auto_repair:
        repair_info = make_partition_valid(problem, original_partition, strategy=repair_strategy)
        evaluated_partition = dict(repair_info["partition"])
    else:
        repair_info = {
            "partition": dict(original_partition),
            "original_partition": dict(original_partition),
            "was_repaired": False,
            "repaired_nodes": [],
            "is_valid": _partition_is_valid(problem, original_partition),
            "repair_strategy": str(repair_strategy or "benefit_per_area").strip().lower(),
            "area_used": _hardware_area_used(problem, original_partition),
            "area_budget": _area_budget(problem),
        }
        evaluated_partition = dict(original_partition)

    return problem, original_partition, evaluated_partition, repair_info


def _build_common_result(
    problem: PartitionScheduleProblem,
    mode: str,
    original_partition: dict[Hashable, int],
    partition: dict[Hashable, int],
    repair_info: dict[str, Any],
    start_times: dict[Hashable, float],
    finish_times: dict[Hashable, float],
    total_comm_delay: float,
    active_comm_edges: list[tuple[Hashable, Hashable, float]],
    makespan: float,
) -> dict[str, Any]:
    hw_nodes = [node for node, side in partition.items() if int(side) == 1]
    sw_nodes = [node for node, side in partition.items() if int(side) == 0]
    hw_finish_times = [finish_times[node] for node in hw_nodes if node in finish_times]
    sw_finish_times = [finish_times[node] for node in sw_nodes if node in finish_times]
    raw_hw_makespan = max(hw_finish_times) if hw_finish_times else 0.0
    raw_sw_makespan = max(sw_finish_times) if sw_finish_times else 0.0

    is_valid = bool(repair_info.get("is_valid", True))
    if is_valid:
        hardware_makespan = float(raw_hw_makespan)
        software_makespan = float(raw_sw_makespan)
        reported_comm_delay = float(total_comm_delay)
    else:
        hardware_makespan = float(problem.violation_cost)
        software_makespan = float(problem.violation_cost)
        reported_comm_delay = float(problem.violation_cost)

    return {
        "mode": mode,
        "partition": dict(partition),
        "original_partition": dict(original_partition),
        "was_repaired": bool(repair_info.get("was_repaired", False)),
        "repaired_nodes": list(repair_info.get("repaired_nodes", [])),
        "is_valid": is_valid,
        "makespan": float(makespan),
        "start_times": {node: float(time) for node, time in start_times.items()},
        "finish_times": {node: float(time) for node, time in finish_times.items()},
        "end_times": {node: float(time) for node, time in finish_times.items()},
        "hardware_nodes": hw_nodes,
        "software_nodes": sw_nodes,
        "hardware_makespan": hardware_makespan,
        "software_makespan": software_makespan,
        "total_communication_delay": reported_comm_delay,
        "active_communication_edges": list(active_comm_edges),
        "execution_summary": {
            "total_nodes": len(problem.graph.nodes()),
            "completed_nodes": len(finish_times),
            "hardware_execution_time": hardware_makespan,
            "software_execution_time": software_makespan,
            "communication_overhead": reported_comm_delay,
            "total_makespan": float(makespan),
            "area_used": float(repair_info.get("area_used", _hardware_area_used(problem, partition))),
            "area_budget": float(repair_info.get("area_budget", _area_budget(problem))),
            "area_constraint": float(problem.area_constraint),
            "violation": not is_valid,
            "was_repaired": bool(repair_info.get("was_repaired", False)),
        },
    }


def evaluate_partition_lssp(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    auto_repair: bool = True,
    repair_strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    problem, original_partition, partition, repair_info = _prepare_partition(
        problem_or_task_graph=problem_or_task_graph,
        partition_assignment=partition_assignment,
        auto_repair=auto_repair,
        repair_strategy=repair_strategy,
    )
    topo = _require_dag(problem)
    topo_idx = {node: idx for idx, node in enumerate(topo)}
    priorities = compute_static_priorities(problem, partition)

    start_times: dict[Hashable, float] = {}
    finish_times: dict[Hashable, float] = {}
    edge_start_times: dict[tuple[Hashable, Hashable], float] = {}
    edge_finish_times: dict[tuple[Hashable, Hashable], float] = {}
    bus_schedule: list[dict[str, Any]] = []
    scheduled: set[Hashable] = set()
    unscheduled: set[Hashable] = set(topo)
    sw_available = 0.0
    bus_available = 0.0

    while unscheduled:
        ready = [
            node
            for node in topo
            if node in unscheduled and all(pred in scheduled for pred in problem.graph.predecessors(node))
        ]
        if not ready:
            raise RuntimeError("LSSP scheduler stalled: no ready node found in the remaining subgraph.")

        ready.sort(key=lambda node: (-priorities[node], topo_idx[node], _node_label(node)))

        for node in ready:
            dep_ready = 0.0
            preds = sorted(
                problem.graph.predecessors(node),
                key=lambda pred: (
                    float(finish_times[pred]),
                    topo_idx[pred],
                    _node_label(pred),
                ),
            )

            for pred in preds:
                comm = _edge_comm_time(problem, pred, node, partition)
                edge_ready = max(float(finish_times[pred]), dep_ready)
                if comm > 0.0:
                    edge_start = max(edge_ready, bus_available)
                    edge_finish = edge_start + comm
                    bus_available = edge_finish
                    bus_schedule.append(
                        {
                            "edge": (pred, node),
                            "source": pred,
                            "target": node,
                            "start_time": float(edge_start),
                            "finish_time": float(edge_finish),
                            "duration": float(comm),
                        }
                    )
                else:
                    edge_start = edge_ready
                    edge_finish = edge_ready

                edge_start_times[(pred, node)] = float(edge_start)
                edge_finish_times[(pred, node)] = float(edge_finish)
                dep_ready = max(dep_ready, edge_finish)

            exec_time = _node_exec_time(problem, node, partition)
            if int(partition[node]) == 0:
                start = max(dep_ready, sw_available)
                finish = start + exec_time
                sw_available = finish
            else:
                start = dep_ready
                finish = start + exec_time

            start_times[node] = float(start)
            finish_times[node] = float(finish)
            scheduled.add(node)
            unscheduled.remove(node)

    active_comm_edges = [
        (u, v, float(problem.communication_costs.get((u, v), 0.0)))
        for u, v in problem.graph.edges()
        if int(partition[u]) != int(partition[v]) and float(problem.communication_costs.get((u, v), 0.0)) > 0.0
    ]
    total_comm_delay = float(sum(item[2] for item in active_comm_edges))
    raw_makespan = max(finish_times.values()) if finish_times else 0.0
    reported_makespan = raw_makespan if repair_info.get("is_valid", True) else float(problem.violation_cost)

    result = _build_common_result(
        problem=problem,
        mode="lssp",
        original_partition=original_partition,
        partition=partition,
        repair_info=repair_info,
        start_times=start_times,
        finish_times=finish_times,
        total_comm_delay=total_comm_delay,
        active_comm_edges=active_comm_edges,
        makespan=reported_makespan,
    )
    result.update(
        {
            "static_priorities": {node: float(priority) for node, priority in priorities.items()},
            "edge_start_times": {edge: float(time) for edge, time in edge_start_times.items()},
            "edge_finish_times": {edge: float(time) for edge, time in edge_finish_times.items()},
            "bus_schedule": bus_schedule,
        }
    )
    return result


def evaluate_partition_dag(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    auto_repair: bool = True,
    repair_strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    problem, original_partition, partition, repair_info = _prepare_partition(
        problem_or_task_graph=problem_or_task_graph,
        partition_assignment=partition_assignment,
        auto_repair=auto_repair,
        repair_strategy=repair_strategy,
    )
    topo = _require_dag(problem)
    node_order = list(problem.graph.nodes())

    from utils.scheduler_utils import compute_dag_makespan

    dag_partition = [1 - int(partition[node]) for node in node_order]
    raw_makespan, raw_start_times = compute_dag_makespan(problem.graph, dag_partition)
    start_times = {node: float(raw_start_times[node]) for node in node_order}
    finish_times = {
        node: float(start_times[node] + _node_exec_time(problem, node, partition))
        for node in node_order
    }
    active_comm_edges = [
        (u, v, float(problem.communication_costs.get((u, v), 0.0)))
        for u, v in problem.graph.edges()
        if int(partition[u]) != int(partition[v]) and float(problem.communication_costs.get((u, v), 0.0)) > 0.0
    ]
    total_comm_delay = float(sum(item[2] for item in active_comm_edges))
    reported_makespan = float(raw_makespan) if repair_info.get("is_valid", True) else float(problem.violation_cost)

    result = _build_common_result(
        problem=problem,
        mode="dag",
        original_partition=original_partition,
        partition=partition,
        repair_info=repair_info,
        start_times=start_times,
        finish_times=finish_times,
        total_comm_delay=total_comm_delay,
        active_comm_edges=active_comm_edges,
        makespan=reported_makespan,
    )
    result.update(
        {
            "software_order": [node for node in topo if int(partition[node]) == 0],
            "solver_status": None,
        }
    )
    return result


def evaluate_partition(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    mode: str = "lssp",
    auto_repair: bool = True,
    repair_strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    mode_key = str(mode or "lssp").strip().lower()
    if mode_key == "lssp":
        return evaluate_partition_lssp(
            problem_or_task_graph=problem_or_task_graph,
            partition_assignment=partition_assignment,
            auto_repair=auto_repair,
            repair_strategy=repair_strategy,
        )
    if mode_key == "dag":
        return evaluate_partition_dag(
            problem_or_task_graph=problem_or_task_graph,
            partition_assignment=partition_assignment,
            auto_repair=auto_repair,
            repair_strategy=repair_strategy,
        )
    raise ValueError(f"Unsupported schedule evaluation mode: {mode}")


def evaluate_makespan_lssp(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    auto_repair: bool = True,
    repair_strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    return evaluate_partition_lssp(
        problem_or_task_graph=problem_or_task_graph,
        partition_assignment=partition_assignment,
        auto_repair=auto_repair,
        repair_strategy=repair_strategy,
    )


def evaluate_makespan_dag(
    problem_or_task_graph: Any,
    partition_assignment: Mapping[Hashable, Any] | Sequence[Any],
    auto_repair: bool = True,
    repair_strategy: str = "benefit_per_area",
) -> dict[str, Any]:
    return evaluate_partition_dag(
        problem_or_task_graph=problem_or_task_graph,
        partition_assignment=partition_assignment,
        auto_repair=auto_repair,
        repair_strategy=repair_strategy,
    )


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _load_demo_config(config_path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config_path = Path(config_path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    config["config"] = str(config_path)

    for key in ("graph-file", "taskgraph-pickle", "solution-dir", "output-dir", "taskgraph-dir"):
        resolved = _resolve_repo_path(config.get(key))
        if resolved is not None:
            config[key] = str(resolved)

    return config


def _load_task_graph_from_config(config: Mapping[str, Any]):
    from meta_heuristic import TaskGraph

    tg_pickle = _resolve_repo_path(config.get("taskgraph-pickle"))
    if tg_pickle is not None and tg_pickle.exists():
        with open(tg_pickle, "rb") as f:
            task_graph = pickle.load(f)
        return synchronize_problem_with_config(task_graph, config)

    graph_file = _resolve_repo_path(config.get("graph-file"))
    if graph_file is None or not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {config.get('graph-file')}")

    task_graph = TaskGraph(area_constraint=float(config.get("area-constraint", 1.0)))
    task_graph.load_graph_from_pydot(
        str(graph_file),
        k=float(config.get("hw-scale-factor", 1.0)),
        l=float(config.get("hw-scale-variance", 0.0)),
        mu=float(config.get("comm-scale-factor", 1.0)),
        A_max=100,
        seed=int(config.get("seed", 42)),
    )
    return synchronize_problem_with_config(task_graph, config)


def _sample_random_partition(
    problem: PartitionScheduleProblem,
    rng: random.Random,
    hw_probability: float = 0.5,
) -> dict[Hashable, int]:
    p_hw = min(max(float(hw_probability), 0.0), 1.0)
    return {
        node: (1 if rng.random() < p_hw else 0)
        for node in problem.graph.nodes()
    }


def _format_partition(partition: Mapping[Hashable, int]) -> str:
    items = ", ".join(f"{node}:{int(partition[node])}" for node in partition)
    return "{" + items + "}"


def _format_times(problem: PartitionScheduleProblem, times: Mapping[Hashable, float]) -> str:
    ordered = ", ".join(
        f"{node}:{float(times[node]):.2f}"
        for node in problem.graph.nodes()
        if node in times
    )
    return "{" + ordered + "}"


def _demo_output_paths(config_path: Path, out_dir: Path) -> dict[str, Path]:
    stem = config_path.stem
    return {
        "input": out_dir / f"{stem}_random_demo_input.png",
        "dag": out_dir / f"{stem}_random_demo_dag.png",
        "lssp": out_dir / f"{stem}_random_demo_lssp.png",
    }


def run_demo(
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int | None = None,
    hw_probability: float = 0.5,
) -> dict[str, Any]:
    from tools.visualize_schedule_from_partitions import (
        save_input_graph_visualization,
        save_schedule_visualization,
    )

    config_path = Path(config_path or (ROOT / "configs" / "config_fig3_taskgraph_gnn_fast_simple.yaml")).resolve()
    out_root = Path(out_dir or (ROOT / "Figs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = _load_demo_config(config_path)
    task_graph = _load_task_graph_from_config(config)
    problem = build_problem(task_graph)
    demo_seed = int(config.get("seed", 42) if seed is None else seed)
    rng = random.Random(demo_seed)

    random_partition = _sample_random_partition(problem, rng, hw_probability=hw_probability)
    repair_info = make_partition_valid(problem, random_partition)
    evaluated_partition = dict(repair_info["partition"])

    dag_result = evaluate_partition_dag(problem, random_partition, auto_repair=True)
    lssp_result = evaluate_partition_lssp(problem, random_partition, auto_repair=True)

    figure_paths = _demo_output_paths(config_path, out_root)
    context = {
        "run_name": f"{config_path.stem}_random_demo",
        "seed": demo_seed,
        "partition_file": "in-memory random assignment",
    }

    save_input_graph_visualization(
        task_graph,
        str(figure_paths["input"]),
        context={"run_name": f"{config_path.stem}_random_demo", "seed": demo_seed},
    )
    save_schedule_visualization(
        task_graph,
        random_partition,
        method="random_demo_dag",
        out_path=str(figure_paths["dag"]),
        context=context,
        config=config,
        mode="dag",
        schedule_result=dag_result,
    )
    save_schedule_visualization(
        task_graph,
        random_partition,
        method="random_demo_lssp",
        out_path=str(figure_paths["lssp"]),
        context=context,
        config=config,
        mode="lssp",
        schedule_result=lssp_result,
    )

    print(f"Demo config: {config_path}")
    print(
        "Task graph:"
        f" nodes={len(problem.graph.nodes())}"
        f" edges={len(problem.graph.edges())}"
        f" area_limit={_area_budget(problem):.2f}/{problem.total_area:.2f}"
    )
    print(f"Random seed: {demo_seed}")
    print(f"Random assignment (raw): {_format_partition(random_partition)}")
    if repair_info["was_repaired"]:
        print(f"Random assignment repaired for area feasibility: {_format_partition(evaluated_partition)}")
        print(f"Repaired nodes: {repair_info['repaired_nodes']}")
    else:
        print("Random assignment was already area-feasible.")
    print(f"DAG makespan: {float(dag_result['makespan']):.2f}")
    print(f"DAG start times: {_format_times(problem, dag_result['start_times'])}")
    print(f"DAG finish times: {_format_times(problem, dag_result['finish_times'])}")
    print(f"LSSP makespan: {float(lssp_result['makespan']):.2f}")
    print(f"LSSP start times: {_format_times(problem, lssp_result['start_times'])}")
    print(f"LSSP finish times: {_format_times(problem, lssp_result['finish_times'])}")
    print("Saved figures:")
    for key in ("input", "dag", "lssp"):
        print(f"  {key}: {figure_paths[key]}")

    return {
        "config": config,
        "task_graph": task_graph,
        "random_partition": random_partition,
        "repaired_partition": evaluated_partition,
        "repair_info": repair_info,
        "dag_result": dag_result,
        "lssp_result": lssp_result,
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo the unified DAG and LSSP partition evaluators on a small task graph.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "config_fig3_taskgraph_gnn_fast_simple.yaml"),
        help="YAML config to load for the demo task graph.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "Figs"),
        help="Directory where demo figures will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the sampled demo partition. Defaults to the config seed.",
    )
    parser.add_argument(
        "--hw-probability",
        type=float,
        default=0.3,
        help="Probability of placing each task in hardware before feasibility repair.",
    )
    args = parser.parse_args()

    run_demo(
        config_path=args.config,
        out_dir=args.out_dir,
        seed=args.seed,
        hw_probability=args.hw_probability,
    )


__all__ = [
    "PartitionScheduleProblem",
    "build_problem",
    "synchronize_problem_with_config",
    "make_partition_valid",
    "compute_static_priorities",
    "evaluate_partition_lssp",
    "evaluate_partition_dag",
    "evaluate_partition",
    "evaluate_makespan_lssp",
    "evaluate_makespan_dag",
    "run_demo",
]


if __name__ == "__main__":
    main()
