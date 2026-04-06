from __future__ import annotations

from typing import Dict, Callable, Any, List, Mapping
import pandas as pd
from dataclasses import dataclass
from utils.logging_utils import LogManager
from meta_heuristic.partition_schedule_evaluator import evaluate_partition_dag, evaluate_partition_lssp
import time

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/method_registry.log")

logger = LogManager.get_logger(__name__)

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
    return "queue"


def _compute_objective_value(task_graph, partition: dict, mode: str) -> float:
    if mode == "lp":
        return _compute_lp_makespan(task_graph, partition)
    if mode == "queue":
        return _compute_queue_makespan(task_graph, partition)
    raise ValueError(f"Unsupported objective mode: {mode}")


def _compute_lssp_makespan(task_graph, partition: dict) -> float:
    return float(evaluate_partition_lssp(task_graph, partition)["makespan"])


def _extract_diff_gnn_order_meta(func: Callable) -> dict:
    meta = getattr(func, "last_run_meta", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _compute_schedule_metrics(
    task_graph,
    partition: dict,
    method_name: str,
    learned_sw_scores: Mapping | None = None,
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

    if lssp_swprio_makespan is None:
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


def _get_naive_baseline(task_graph, opt_cost_type: str, config: dict | None) -> tuple[float, dict]:
    partition = {node: 0 for node in task_graph.graph.nodes()}
    mode = _resolve_objective_mode(opt_cost_type)

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

        # get a naive solution first
        best_cost, partition = _get_naive_baseline(task_graph, naive_opt_func_name, config)
            
        logger.info(f"naive assignment has a opt_cost of {best_cost}")

        start = time.time()
        # Run the optimization method
        opt_cost, opt_solution = func(dim, func_to_optimize, config, **kwargs)
        opt_time = time.time()-start

        if opt_cost<best_cost:
            # Create partition from solution in the form of numpy array
            logger.info(f"{name.upper()} was able to find better partition than all software partition")
            best_cost = opt_cost
            partition = task_graph.get_partitioning(opt_solution, method=name)
        
        print(partition)
        partition = _normalize_partition(partition)
        print(partition)
        diff_meta = _extract_diff_gnn_order_meta(func) if str(name).lower() == "diff_gnn_order" else {}
        schedule_metrics = _compute_schedule_metrics(
            task_graph,
            partition,
            method_name=name,
            learned_sw_scores=diff_meta.get("sw_priority_scores"),
        )
        schedule_result = schedule_metrics["lssp_result"]
        partition = dict(schedule_metrics["partition"])
        makespan = float(schedule_metrics["lssp_makespan"])
        dag_makespan = float(schedule_metrics["dag_makespan"])
        lssp_swprio_makespan = schedule_metrics["lssp_swprio_makespan"]
        best_makespan = float(schedule_metrics["best_makespan"])
        partition_cost = task_graph.evaluate_partition_cost(partition)
        reported_opt_cost = float(best_cost)
        if str(name).lower() == "diff_gnn_order":
            # diff_gnn_order is trained/evaluated with queue-style objective in-model,
            # but final reporting for this project is LSSP-based. Keep the better of both.
            reported_opt_cost = float(min(reported_opt_cost, best_makespan))
            logger.info(
                "DIFF_GNN_ORDER combined objective: raw_opt_cost=%.6f dag=%.6f lssp=%.6f lssp_swprio=%s best_makespan=%.6f -> reported_opt_cost=%.6f",
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
            optimization_time = opt_time,
            additional_metrics = {
                "was_repaired": bool(schedule_result.get("was_repaired", False)),
                "num_repaired_nodes": len(schedule_result.get("repaired_nodes", [])),
                "dag_makespan": dag_makespan,
                "lssp_makespan": makespan,
                "lssp_swprio_makespan": lssp_swprio_makespan,
                "best_makespan": best_makespan,
            },
            ## later add time here
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
        schedule_metrics = _compute_schedule_metrics(task_graph, partition, method_name=name)
        schedule_result = schedule_metrics["lssp_result"]
        partition = dict(schedule_metrics["partition"])
        makespan = float(schedule_metrics["lssp_makespan"])
        dag_makespan = float(schedule_metrics["dag_makespan"])
        lssp_swprio_makespan = schedule_metrics["lssp_swprio_makespan"]
        best_makespan = float(schedule_metrics["best_makespan"])
        partition_cost = task_graph.evaluate_partition_cost(partition)
        
        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = 'None',
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition,
            optimization_time = timing_info,
            additional_metrics = {
                "was_repaired": bool(schedule_result.get("was_repaired", False)),
                "num_repaired_nodes": len(schedule_result.get("repaired_nodes", [])),
                "dag_makespan": dag_makespan,
                "lssp_makespan": makespan,
                "lssp_swprio_makespan": lssp_swprio_makespan,
                "best_makespan": best_makespan,
            },
            ## add timing info later maybe
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
