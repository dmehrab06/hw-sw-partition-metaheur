from typing import Dict, Callable, Any, List
import pandas as pd
from dataclasses import dataclass
from utils.logging_utils import LogManager
from utils.scheduler_utils import compute_dag_makespan
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

def _compute_lp_makespan(task_graph, partition: dict) -> float:
    """Compute LP makespan using compute_dag_makespan with area-constraint penalty."""
    if task_graph.violates(partition):
        return task_graph.violation_cost
    assignment = [1 - partition[n] for n in task_graph.rounak_graph]
    makespan, _ = compute_dag_makespan(task_graph.rounak_graph, assignment)
    return makespan


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
        if naive_opt_func_name=='partition':
            best_cost, partition = task_graph.get_naive_solution()
        else:
            best_cost, partition = task_graph.get_naive_solution_makespan()
            
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
        if naive_opt_func_name == 'mip':
            makespan = _compute_lp_makespan(task_graph, partition)
        else:
            makespan = task_graph.evaluate_makespan(partition)['makespan']
        partition_cost = task_graph.evaluate_partition_cost(partition)
        
        # Store result
        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = getattr(func_to_optimize, '__name__', 'Unknown'),
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition,
            optimization_time = opt_time
            ## later add time here
        )
        
        self.results[name] = result
        return result
    
    def add_manual_result(self, name: str, best_cost: float, best_solution: Any, 
                         task_graph=None, timing_info = 0.0, naive_opt_func_name='partition') -> MethodResult:
        """Add a result from a method that doesn't follow the standard interface (like greedy)"""
        partition = task_graph.get_partitioning(best_solution, method=name)
        print(partition)
        partition = _normalize_partition(partition)
        print(partition)
        if naive_opt_func_name == 'mip':
            makespan = _compute_lp_makespan(task_graph, partition)
        else:
            makespan = task_graph.evaluate_makespan(partition)['makespan']
        partition_cost = task_graph.evaluate_partition_cost(partition)
        
        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = 'None',
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition,
            optimization_time = timing_info
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
                for metric in results.additional_metrics:
                    results_dict[f'{name}_{metric}'] = result.additional_metrics[metric]
        
        return results_dict
    
    def get_all_method_names(self) -> List[str]:
        """Get list of all method names (registered + manual)"""
        return list(self.results.keys())
    
    def get_registered_method_names(self) -> List[str]:
        """Get list of registered method names only"""
        return list(self.methods.keys())
