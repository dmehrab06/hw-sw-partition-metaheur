from typing import Dict, Callable, Any, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class MethodResult:
    """Container for method optimization results"""
    method_name: str
    best_optimization_cost: float
    func_as_black_box: str
    best_solution: Any
    makespan: float
    partition_cost: float
    partition_assignment: Dict[str, Any]
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
                   config: dict, task_graph=None) -> MethodResult:
        """Run a registered method and store results"""
        if name not in self.methods:
            raise ValueError(f"Method {name} not registered")
        
        method_info = self.methods[name]
        func = method_info['func']
        kwargs = method_info['kwargs']
        
        # Run the optimization method
        best_cost, best_solution = func(dim, func_to_optimize, config, **kwargs)
        
        # Create partition from solution if task_graph is provided
        partition = task_graph.get_partitioning(best_solution, method=name)
        makespan = task_graph.evaluate_makespan(partition)['makespan']
        partition_cost = task_graph.evaluate_partition_cost(partition)
        
        # Store result
        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = getattr(func_to_optimize, '__name__', 'Unknown'),
            best_solution=best_solution,
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition
            ## later add time here
        )
        
        self.results[name] = result
        return result
    
    def add_manual_result(self, name: str, best_cost: float, best_solution: Any, 
                         task_graph=None) -> MethodResult:
        """Add a result from a method that doesn't follow the standard interface (like greedy)"""
        partition = task_graph.get_partitioning(best_solution, method=name)
        makespan = task_graph.evaluate_makespan(partition)['makespan']
        partition_cost = task_graph.evaluate_partition_cost(partition)
        
        result = MethodResult(
            method_name=name,
            best_optimization_cost = best_cost,
            func_as_black_box = 'None',
            best_solution=best_solution,
            makespan = makespan,
            partition_cost = partition_cost,
            partition_assignment = partition
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

