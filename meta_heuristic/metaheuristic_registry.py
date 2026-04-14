from __future__ import annotations

from typing import Dict, Callable, Any, List, Mapping
import pandas as pd
from dataclasses import dataclass
from utils.logging_utils import LogManager
from meta_heuristic.partition_schedule_evaluator import evaluate_partition_dag, evaluate_partition_lssp
import time
import math

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/method_registry.log")

logger = LogManager.get_logger(__name__)

_LARGE_GRAPH_DIFF_METHODS = {"diff_gnn", "diff_gnn_order"}
_LARGE_GRAPH_DIFF_THRESHOLD = 1000

def _normalize_partition(partition: dict):
    """
    Accepts:
      - 0/1
      - "hardware"/"software" (and "hw"/"sw")
    Returns a dict node -> 0/1 in the format expected by TaskGraph.evaluate_makespan().
    Convention used by TaskGraph: 1 = hardware, 0 = software. :contentReference[oaicite:1]{index=1}
    """
    out = {}
    for node, a in partition.items():
        if a in (0, 1):
            out[node] = int(a)
            continue
        if isinstance(a, str):
            aa = a.strip().lower()
            if aa in ("hardware", "hw"):
                out[node] = 1
                continue
            if aa in ("software", "sw"):
                out[node] = 0
                continue
        raise ValueError(f"Invalid partition value for node={node}: {a!r}")
    return out

def _compute_queue_makespan(task_graph, partition: dict) -> float:
    """Compute queue-simulation makespan using TaskGraph.evaluate_makespan()."""
    if task_graph.violates(partition):
        return task_graph.violation_cost
    result = task_graph.evaluate_makespan(partition)
    return float(result["makespan"])


def _compute_lp_makespan(task_graph, partition: dict) -> float:
    """Compute LP/legacy makespan using compute_dag_makespan()."""
    if task_graph.violates(partition):
        return task_graph.violation_cost
    return float(
        evaluate_partition_dag(
            task_graph,
            partition,
            auto_repair=False,
        )["makespan"]
    )


def _resolve_objective_mode(opt_cost_type: str) -> str:
    opt_key = str(opt_cost_type or "queue").strip().lower()
    if opt_key == "partition":
        return "partition"
    if opt_key == "mip":
        return "lp"
    if opt_key == "lssp":
        return "lssp"
    return "queue"


def _compute_objective_value(task_graph, partition: dict, mode: str) -> float:
    if mode == "lp":
        return _compute_lp_makespan(task_graph, partition)
    if mode == "lssp":
        return _compute_lssp_makespan(task_graph, partition)
    if mode == "queue":
        return _compute_queue_makespan(task_graph, partition)
    raise ValueError(f"Unsupported objective mode: {mode}")


def _compute_lssp_makespan(task_graph, partition: dict) -> float:
    if task_graph.violates(partition):
        return task_graph.violation_cost
    return float(
        evaluate_partition_lssp(
            task_graph,
            partition,
            auto_repair=False,
        )["makespan"]
    )


def _extract_method_runtime_meta(func: Callable) -> dict:
    meta = getattr(func, "last_run_meta", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _coerce_runtime_seconds(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return max(0.0, out)


def _use_large_graph_diff_lssp_only_final(
    method_name: str,
    task_graph,
    config: Mapping[str, Any] | None = None,
) -> bool:
    method_key = str(method_name).lower()
    if method_key not in _LARGE_GRAPH_DIFF_METHODS:
        return False
    if task_graph is None or not hasattr(task_graph, "graph") or getattr(task_graph, "graph", None) is None:
        return False

    method_cfg = {}
    if isinstance(config, Mapping):
        cfg_key = "diffgnn_order" if method_key == "diff_gnn_order" else "diffgnn"
        method_cfg_raw = config.get(cfg_key, {})
        if isinstance(method_cfg_raw, Mapping):
            method_cfg = dict(method_cfg_raw)

    enabled = bool(method_cfg.get("large_graph_fast_policy", True))
    threshold = int(method_cfg.get("large_graph_fast_threshold", _LARGE_GRAPH_DIFF_THRESHOLD))
    return enabled and int(len(task_graph.graph.nodes())) > threshold


def _compute_schedule_metrics(
    task_graph,
    partition: dict,
    method_name: str,
    learned_sw_scores: Mapping | None = None,
    final_lssp_only: bool = False,
) -> dict:
    """
    Compute schedule metrics on a single (possibly repaired) partition.
    - LSSP uses static priorities.
    - DAG uses fixed topological processing.
    - diff_gnn_order optionally adds learned SW-priority LSSP.
    """
    lssp_result = evaluate_partition_lssp(task_graph, partition)
    repaired_partition = dict(lssp_result["partition"])
    lssp_makespan = float(lssp_result["makespan"])

    if final_lssp_only:
        dag_makespan = float("nan")
    else:
        dag_makespan = float(
            evaluate_partition_dag(
                task_graph,
                repaired_partition,
                auto_repair=False,
            )["makespan"]
        )

    lssp_swprio_makespan = None
    if str(method_name).lower() == "diff_gnn_order" and isinstance(learned_sw_scores, Mapping):
        lssp_swprio_makespan = float(
            evaluate_partition_lssp(
                task_graph,
                repaired_partition,
                auto_repair=False,
                software_priority_scores=learned_sw_scores,
            )["makespan"]
        )

    method_key = str(method_name).lower()
    if final_lssp_only and method_key == "diff_gnn_order":
        if lssp_swprio_makespan is None:
            best_makespan = float(lssp_makespan)
        else:
            best_makespan = float(min(lssp_makespan, lssp_swprio_makespan))
    elif final_lssp_only:
        best_makespan = float(lssp_makespan)
    elif method_key == "diff_gnn_order":
        if lssp_swprio_makespan is None:
            best_makespan = float(lssp_makespan)
        else:
            best_makespan = float(min(lssp_makespan, lssp_swprio_makespan))
    elif lssp_swprio_makespan is None:
        best_makespan = float(min(dag_makespan, lssp_makespan))
    else:
        best_makespan = float(min(dag_makespan, lssp_makespan, lssp_swprio_makespan))

    return {
        "partition": repaired_partition,
        "lssp_result": lssp_result,
        "dag_makespan": dag_makespan,
        "lssp_makespan": lssp_makespan,
        "lssp_swprio_makespan": lssp_swprio_makespan,
        "best_makespan": best_makespan,
    }


def _build_validity_note(schedule_result: Mapping[str, Any]) -> str:
    is_valid = bool(schedule_result.get("is_valid", True))
    was_repaired = bool(schedule_result.get("was_repaired", False))
    repair_strategy = str(schedule_result.get("repair_strategy", "benefit_per_area"))

    if is_valid and not was_repaired:
        return "Valid solution; no area repair needed."
    if is_valid and was_repaired:
        return (
            "Invalid before post-processing; repaired with "
            f"{repair_strategy} greedy fixing to satisfy the area constraint."
        )
    return (
        "Invalid solution after post-processing; area constraint still violated "
        "and reported makespan reflects the violation penalty."
    )


def _get_naive_baseline(
    task_graph,
    opt_cost_type: str,
    config: dict | None,
    objective_mode_override: str | None = None,
) -> tuple[float, dict]:
    partition = {node: 0 for node in task_graph.graph.nodes()}
    mode = str(objective_mode_override or _resolve_objective_mode(opt_cost_type)).strip().lower()

    if mode == "partition":
        return task_graph.evaluate_partition_cost(partition), partition
    return _compute_objective_value(task_graph, partition, mode), partition


@dataclass
class MethodResult:
    """Container for method optimization results"""
    method_name: str
    best_optimization_cost: float
    func_as_black_box: str
    makespan: float
    partition_cost: float
    partition_assignment: Dict[str, Any]
    optimization_time: float
    postprocess_time: float = 0.0
    total_runtime: float = 0.0
    additional_metrics: Dict[str, Any] = None

class MethodRegistry:
    """Registry for optimization methods with automatic result collection"""
    
    def __init__(self):
        self.methods: Dict[str, Callable] = {}
        self.results: Dict[str, MethodResult] = {}
    
    def register_method(self, name: str, func: Callable, **kwargs):
        """Register an optimization method"""
        self.methods[name] = {'func': func, 'kwargs': kwargs}
    
    def run_method(self, name: str, dim: int, func_to_optimize: Callable, 
                   config: dict, task_graph=None, naive_opt_func_name='partition') -> MethodResult:
        """Run a registered method and store results"""
        if name not in self.methods:
            raise ValueError(f"Method {name} not registered")

        if task_graph is None:
            raise ValueError(f"Cannot run without a task graph")
        
        method_info = self.methods[name]
        func = method_info['func']
        kwargs = method_info['kwargs']

        large_graph_lssp_only = _use_large_graph_diff_lssp_only_final(name, task_graph, config)
        naive_mode_override = "lp" if large_graph_lssp_only else None
        if large_graph_lssp_only:
            logger.info(
                "%s large-graph policy enabled: using cheap DAG baseline/selection during optimization; "
                "final reporting will evaluate only the selected partition with LSSP.",
                str(name).upper(),
            )

        # get a naive solution first
        best_cost, partition = _get_naive_baseline(
            task_graph,
            naive_opt_func_name,
            config,
            objective_mode_override=naive_mode_override,
        )
            
        logger.info(f"naive assignment has a opt_cost of {best_cost}")

        start = time.perf_counter()
        # Run the optimization method
        opt_cost, opt_solution = func(dim, func_to_optimize, config, **kwargs)
        method_call_time = time.perf_counter() - start

        if opt_cost<best_cost:
            # Create partition from solution in the form of numpy array
            logger.info(f"{name.upper()} was able to find better partition than all software partition")
            best_cost = opt_cost
            partition = task_graph.get_partitioning(opt_solution, method=name)
        
        print(partition)
        partition = _normalize_partition(partition)
        print(partition)
        method_key = str(name).lower()
        method_runtime_meta = _extract_method_runtime_meta(func)
        learned_sw_scores = None
        if method_key == "diff_gnn_order":
            learned_sw_scores = method_runtime_meta.get("sw_priority_scores")
        schedule_start = time.perf_counter()
        schedule_metrics = _compute_schedule_metrics(
            task_graph,
            partition,
            method_name=name,
            learned_sw_scores=learned_sw_scores,
            final_lssp_only=large_graph_lssp_only,
        )
        schedule_time = time.perf_counter() - schedule_start
        schedule_result = schedule_metrics["lssp_result"]
        partition = dict(schedule_metrics["partition"])
        makespan = float(schedule_metrics["lssp_makespan"])
        dag_makespan = float(schedule_metrics["dag_makespan"])
        lssp_swprio_makespan = schedule_metrics["lssp_swprio_makespan"]
        best_makespan = float(schedule_metrics["best_makespan"])
        partition_cost = task_graph.evaluate_partition_cost(partition)
        optimization_time = _coerce_runtime_seconds(
            method_runtime_meta.get("optimization_time_sec", None)
        )
        if optimization_time is None:
            optimization_time = max(0.0, method_call_time)
        optimization_time = min(optimization_time, max(0.0, method_call_time))
        wrapper_postprocess_time = max(0.0, method_call_time - optimization_time)
        postprocess_time = wrapper_postprocess_time + schedule_time
        total_runtime = optimization_time + postprocess_time
        reported_opt_cost = float(best_cost)
        if large_graph_lssp_only and method_key == "diff_gnn_order":
            reported_opt_cost = float(best_makespan)
            logger.info(
                "DIFF_GNN_ORDER large-graph final reporting: lssp=%.6f lssp_swprio=%s -> reported_opt_cost=%.6f",
                makespan,
                f"{float(lssp_swprio_makespan):.6f}" if lssp_swprio_makespan is not None else "nan",
                reported_opt_cost,
            )
        elif large_graph_lssp_only and method_key == "diff_gnn":
            reported_opt_cost = float(makespan)
            logger.info(
                "DIFF_GNN large-graph final reporting: lssp=%.6f -> reported_opt_cost=%.6f",
                makespan,
                reported_opt_cost,
            )
        elif method_key == "diff_gnn_order":
            # diff_gnn_order is trained with a surrogate objective, but final reporting
            # for this project is based only on executable LSSP schedules.
            reported_opt_cost = float(best_makespan)
            logger.info(
                "DIFF_GNN_ORDER final reporting: raw_opt_cost=%.6f dag=%.6f lssp=%.6f lssp_swprio=%s best_makespan=%.6f -> reported_opt_cost=%.6f",
                float(best_cost),
                dag_makespan,
                makespan,
                f"{float(lssp_swprio_makespan):.6f}" if lssp_swprio_makespan is not None else "nan",
                best_makespan,
                reported_opt_cost,
            )
        
        # Store result
        result = MethodResult(
            method_name=name,
            best_optimization_cost = reported_opt_cost,
            func_as_black_box = getattr(func_to_optimize, '__name__', 'Unknown'),
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition,
            optimization_time = optimization_time,
            postprocess_time = postprocess_time,
            total_runtime = total_runtime,
            additional_metrics = {
                "solution_valid": bool(schedule_result.get("is_valid", True)),
                "initial_solution_valid": not bool(schedule_result.get("was_repaired", False)),
                "was_repaired": bool(schedule_result.get("was_repaired", False)),
                "num_repaired_nodes": len(schedule_result.get("repaired_nodes", [])),
                "repair_strategy": schedule_result.get("repair_strategy"),
                "area_used": float(schedule_result.get("execution_summary", {}).get("area_used", 0.0)),
                "area_budget": float(schedule_result.get("execution_summary", {}).get("area_budget", 0.0)),
                "validity_note": _build_validity_note(schedule_result),
                "dag_makespan": dag_makespan,
                "lssp_makespan": makespan,
                "lssp_swprio_makespan": lssp_swprio_makespan,
                "best_makespan": best_makespan,
                "runtime_method_call_time_sec": float(method_call_time),
                "runtime_schedule_eval_time_sec": float(schedule_time),
            },
        )
        
        self.results[name] = result
        return result
    
    def add_manual_result(self, name: str, best_cost: float, best_solution: Any, 
                         task_graph=None, timing_info = 0.0, naive_opt_func_name='partition',
                         config: dict | None = None) -> MethodResult:
        """Add a result from a method that doesn't follow the standard interface (like greedy)"""
        partition = task_graph.get_partitioning(best_solution, method=name)
        print(partition)
        partition = _normalize_partition(partition)
        print(partition)
        schedule_start = time.perf_counter()
        schedule_metrics = _compute_schedule_metrics(task_graph, partition, method_name=name)
        schedule_time = time.perf_counter() - schedule_start
        schedule_result = schedule_metrics["lssp_result"]
        partition = dict(schedule_metrics["partition"])
        makespan = float(schedule_metrics["lssp_makespan"])
        dag_makespan = float(schedule_metrics["dag_makespan"])
        lssp_swprio_makespan = schedule_metrics["lssp_swprio_makespan"]
        best_makespan = float(schedule_metrics["best_makespan"])
        partition_cost = task_graph.evaluate_partition_cost(partition)
        optimization_time = _coerce_runtime_seconds(timing_info)
        if optimization_time is None:
            optimization_time = 0.0
        postprocess_time = max(0.0, schedule_time)
        total_runtime = optimization_time + postprocess_time

        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = 'None',
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition,
            optimization_time = optimization_time,
            postprocess_time = postprocess_time,
            total_runtime = total_runtime,
            additional_metrics = {
                "solution_valid": bool(schedule_result.get("is_valid", True)),
                "initial_solution_valid": not bool(schedule_result.get("was_repaired", False)),
                "was_repaired": bool(schedule_result.get("was_repaired", False)),
                "num_repaired_nodes": len(schedule_result.get("repaired_nodes", [])),
                "repair_strategy": schedule_result.get("repair_strategy"),
                "area_used": float(schedule_result.get("execution_summary", {}).get("area_used", 0.0)),
                "area_budget": float(schedule_result.get("execution_summary", {}).get("area_budget", 0.0)),
                "validity_note": _build_validity_note(schedule_result),
                "dag_makespan": dag_makespan,
                "lssp_makespan": makespan,
                "lssp_swprio_makespan": lssp_swprio_makespan,
                "best_makespan": best_makespan,
                "runtime_schedule_eval_time_sec": float(schedule_time),
            },
        )
        
        self.results[name] = result
        return result
    
    def get_results_dict(self, naive_lb: float) -> Dict[str, Any]:
        """Generate results dictionary for CSV export"""
        results_dict = {}
        
        for name, result in self.results.items():
            
            # Add to results dictionary
            results_dict[f'{name}_opt_cost'] = result.best_optimization_cost
            results_dict[f'{name}_opt_ratio'] = ((result.best_optimization_cost / naive_lb) if naive_lb > 0 else 0)
            results_dict[f'{name}_partition_cost'] = result.partition_cost
            results_dict[f'{name}_bb'] = result.func_as_black_box
            results_dict[f'{name}_makespan'] = result.makespan
            results_dict[f'{name}_time'] = result.optimization_time
            results_dict[f'{name}_optimization_time_sec'] = result.optimization_time
            results_dict[f'{name}_postprocess_time_sec'] = result.postprocess_time
            results_dict[f'{name}_total_runtime_sec'] = result.total_runtime

            if result.additional_metrics:
                for metric in result.additional_metrics:
                    results_dict[f'{name}_{metric}'] = result.additional_metrics[metric]
        
        return results_dict
    
    def get_all_method_names(self) -> List[str]:
        """Get list of all method names (registered + manual)"""
        return list(self.results.keys())
    
    def get_registered_method_names(self) -> List[str]:
        """Get list of registered method names only"""
        return list(self.methods.keys())
