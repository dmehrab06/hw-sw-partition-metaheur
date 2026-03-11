"""
Initial Solution Generators for HW/SW Partitioning with Scheduling.

This module provides various strategies for generating initial solutions:
- Random partitioning (with/without repair)
- Greedy hardware-first (maximize HW usage within constraint)
- Greedy software-first (minimize HW usage)
- Area-aware (consider area efficiency)
- Performance-aware (prioritize critical path tasks)
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import networkx as nx

# Import from project files
import sys
import os
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.local_search import HWSWPartitionScheduler
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/initial_solution_test.log")
logger = LogManager.get_logger(__name__)


class InitialSolutionGenerator(ABC):
    """
    Abstract base class for initial solution generators.
    
    All concrete generators must implement the generate() method.
    """
    
    def __init__(self, repair_infeasible: bool = True):
        """
        Initialize the generator.
        
        Args:
            repair_infeasible: Whether to repair infeasible solutions
        """
        self.repair_infeasible = repair_infeasible
    
    @abstractmethod
    def generate(self, scheduler: HWSWPartitionScheduler, **kwargs) -> np.ndarray:
        """
        Generate an initial partition solution.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            partition: Binary array where partition[i]=1 means task i on HW
        """
        pass
    
    def _repair_partition(self, partition: np.ndarray, 
                         scheduler: HWSWPartitionScheduler) -> np.ndarray:
        """
        Repair an infeasible partition by moving tasks from HW to SW.
        
        Args:
            partition: Current partition
            scheduler: HWSWPartitionScheduler instance
            
        Returns:
            Repaired partition
        """
        partition = partition.copy()
        
        # Calculate current HW area usage
        hw_area = sum(scheduler.hardware_areas[i] for i in range(len(partition))
                     if partition[i] == 1)
        
        if hw_area <= scheduler.max_hardware_area:
            return partition  # Already feasible
        
        # Move tasks from HW to SW until feasible
        hw_tasks = [i for i in range(len(partition)) if partition[i] == 1]
        
        # Sort by area (largest first) - greedy repair
        hw_tasks.sort(key=lambda i: scheduler.hardware_areas[i], reverse=True)
        
        for task_id in hw_tasks:
            if hw_area <= scheduler.max_hardware_area:
                break
            
            # Move to SW
            partition[task_id] = 0
            hw_area -= scheduler.hardware_areas[task_id]
        
        logger.debug(f"Repaired partition: HW area {hw_area:.2f} / {scheduler.max_hardware_area:.2f}")
        
        return partition
    
    def _is_feasible(self, partition: np.ndarray, 
                    scheduler: HWSWPartitionScheduler) -> bool:
        """
        Check if partition is feasible (satisfies HW area constraint).
        
        Args:
            partition: Partition to check
            scheduler: HWSWPartitionScheduler instance
            
        Returns:
            True if feasible, False otherwise
        """
        hw_area = sum(scheduler.hardware_areas[i] for i in range(len(partition))
                     if partition[i] == 1)
        return hw_area <= scheduler.max_hardware_area


class RandomInitialSolution(InitialSolutionGenerator):
    """
    Generate a random initial partition.
    
    Each task is randomly assigned to HW or SW with given probability.
    """
    
    def generate(self, scheduler: HWSWPartitionScheduler, 
                hw_probability: float = 0.5, **kwargs) -> np.ndarray:
        """
        Generate random partition.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            hw_probability: Probability of assigning a task to HW
            
        Returns:
            Random partition
        """
        n_tasks = len(scheduler.graph.nodes())
        partition = np.array([1 if random.random() < hw_probability else 0 
                             for _ in range(n_tasks)])
        
        if self.repair_infeasible and not self._is_feasible(partition, scheduler):
            partition = self._repair_partition(partition, scheduler)
        
        logger.debug(f"Generated random partition: {np.sum(partition)} HW tasks")
        return partition


class GreedyHardwareFirstSolution(InitialSolutionGenerator):
    """
    Generate partition by greedily assigning tasks to HW until area limit.
    
    Prioritizes tasks by a scoring function (e.g., speedup, area efficiency).
    """
    
    def generate(self, scheduler: HWSWPartitionScheduler,
                scoring: str = 'speedup', **kwargs) -> np.ndarray:
        """
        Generate greedy HW-first partition.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            scoring: Scoring method - 'speedup', 'area_efficiency', 'random'
            
        Returns:
            Greedy partition
        """
        n_tasks = len(scheduler.graph.nodes())
        partition = np.zeros(n_tasks, dtype=int)
        
        # Calculate scores for each task
        scores = []
        for i in range(n_tasks):
            if scoring == 'speedup':
                # Speedup = SW_time / HW_time
                sw_time = scheduler.software_times[i]
                hw_time = scheduler.hardware_times[i]
                score = sw_time / hw_time if hw_time > 0 else 1.0
            elif scoring == 'area_efficiency':
                # Time saved per unit area
                time_saved = scheduler.software_times[i] - scheduler.hardware_times[i]
                area = scheduler.hardware_areas[i]
                score = time_saved / area if area > 0 else 0.0
            elif scoring == 'random':
                score = random.random()
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")
            
            scores.append((i, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Greedily assign to HW
        current_hw_area = 0.0
        for task_id, score in scores:
            task_area = scheduler.hardware_areas[task_id]
            
            if current_hw_area + task_area <= scheduler.max_hardware_area:
                partition[task_id] = 1
                current_hw_area += task_area
        
        logger.debug(f"Generated greedy HW-first partition ({scoring}): "
                    f"{np.sum(partition)} HW tasks, area {current_hw_area:.2f}")
        return partition


class GreedySoftwareFirstSolution(InitialSolutionGenerator):
    """
    Generate partition by starting with all SW and selectively moving to HW.
    
    Moves only the most beneficial tasks to HW.
    """
    
    def generate(self, scheduler: HWSWPartitionScheduler,
                n_hw_tasks: Optional[int] = None,
                scoring: str = 'speedup', **kwargs) -> np.ndarray:
        """
        Generate greedy SW-first partition.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            n_hw_tasks: Number of tasks to assign to HW (if None, use heuristic)
            scoring: Scoring method - 'speedup', 'area_efficiency'
            
        Returns:
            Greedy partition
        """
        n_tasks = len(scheduler.graph.nodes())
        partition = np.zeros(n_tasks, dtype=int)
        
        # Determine target number of HW tasks
        if n_hw_tasks is None:
            # Use ~30% of tasks as heuristic
            n_hw_tasks = max(1, int(0.3 * n_tasks))
        
        # Calculate scores
        scores = []
        for i in range(n_tasks):
            if scoring == 'speedup':
                sw_time = scheduler.software_times[i]
                hw_time = scheduler.hardware_times[i]
                score = sw_time / hw_time if hw_time > 0 else 1.0
            elif scoring == 'area_efficiency':
                time_saved = scheduler.software_times[i] - scheduler.hardware_times[i]
                area = scheduler.hardware_areas[i]
                score = time_saved / area if area > 0 else 0.0
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")
            
            scores.append((i, score))
        
        # Sort by score and select top tasks
        scores.sort(key=lambda x: x[1], reverse=True)
        
        current_hw_area = 0.0
        hw_assigned = 0
        
        for task_id, score in scores:
            if hw_assigned >= n_hw_tasks:
                break
            
            task_area = scheduler.hardware_areas[task_id]
            if current_hw_area + task_area <= scheduler.max_hardware_area:
                partition[task_id] = 1
                current_hw_area += task_area
                hw_assigned += 1
        
        logger.debug(f"Generated greedy SW-first partition ({scoring}): "
                    f"{hw_assigned} HW tasks")
        return partition


class CriticalPathSolution(InitialSolutionGenerator):
    """
    Generate partition by prioritizing tasks on the critical path.
    
    Identifies critical path and assigns those tasks to HW first.
    """
    
    def generate(self, scheduler: HWSWPartitionScheduler,
                critical_path_ratio: float = 0.8, **kwargs) -> np.ndarray:
        """
        Generate critical-path-aware partition.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            critical_path_ratio: Fraction of critical path tasks to assign to HW
            
        Returns:
            Critical-path-aware partition
        """
        n_tasks = len(scheduler.graph.nodes())
        partition = np.zeros(n_tasks, dtype=int)
        
        # Find critical path using longest path in DAG
        try:
            # Get topological sort
            topo_order = list(nx.topological_sort(scheduler.graph))
            
            # Calculate longest path to each node (using SW times as baseline)
            longest_path = {}
            for node in topo_order:
                # Path to this node
                incoming_paths = [longest_path[pred] for pred in scheduler.graph.predecessors(node)]
                if incoming_paths:
                    longest_path[node] = max(incoming_paths) + scheduler.software_times[node]
                else:
                    longest_path[node] = scheduler.software_times[node]
            
            # Identify critical path nodes (top percentile)
            path_threshold = max(longest_path.values()) * critical_path_ratio
            critical_nodes = [node for node, path_len in longest_path.items() 
                            if path_len >= path_threshold]
            
            # Calculate speedup for critical nodes
            scores = []
            for node in critical_nodes:
                sw_time = scheduler.software_times[node]
                hw_time = scheduler.hardware_times[node]
                speedup = sw_time / hw_time if hw_time > 0 else 1.0
                scores.append((node, speedup))
            
            # Sort by speedup
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign to HW greedily
            current_hw_area = 0.0
            for task_id, speedup in scores:
                task_area = scheduler.hardware_areas[task_id]
                if current_hw_area + task_area <= scheduler.max_hardware_area:
                    partition[task_id] = 1
                    current_hw_area += task_area
            
            logger.debug(f"Generated critical-path partition: "
                        f"{np.sum(partition)} HW tasks from {len(critical_nodes)} critical tasks")
            
        except Exception as e:
            logger.warning(f"Critical path calculation failed: {e}. Using random partition.")
            # Fallback to random
            random_gen = RandomInitialSolution(repair_infeasible=self.repair_infeasible)
            partition = random_gen.generate(scheduler)
        
        return partition


class BalancedSolution(InitialSolutionGenerator):
    """
    Generate partition that balances HW and SW workload.
    
    Tries to distribute work evenly between HW and SW processors.
    """
    
    def generate(self, scheduler: HWSWPartitionScheduler,
                target_hw_ratio: float = 0.5, **kwargs) -> np.ndarray:
        """
        Generate balanced partition.
        
        Args:
            scheduler: HWSWPartitionScheduler instance
            target_hw_ratio: Target ratio of total work on HW (0.0-1.0)
            
        Returns:
            Balanced partition
        """
        n_tasks = len(scheduler.graph.nodes())
        partition = np.zeros(n_tasks, dtype=int)
        
        # Calculate total work (using SW times as baseline)
        total_work = sum(scheduler.software_times[i] for i in range(n_tasks))
        target_hw_work = total_work * target_hw_ratio
        
        # Calculate benefit scores (time saved per area)
        scores = []
        for i in range(n_tasks):
            time_saved = scheduler.software_times[i] - scheduler.hardware_times[i]
            area = scheduler.hardware_areas[i]
            score = time_saved / area if area > 0 else 0.0
            scores.append((i, score, scheduler.hardware_times[i]))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign to HW until target work reached
        current_hw_area = 0.0
        current_hw_work = 0.0
        
        for task_id, score, hw_time in scores:
            task_area = scheduler.hardware_areas[task_id]
            
            # Check both area constraint and work balance
            if (current_hw_area + task_area <= scheduler.max_hardware_area and
                current_hw_work < target_hw_work):
                partition[task_id] = 1
                current_hw_area += task_area
                current_hw_work += hw_time
        
        logger.debug(f"Generated balanced partition: {np.sum(partition)} HW tasks, "
                    f"HW work ratio {current_hw_work/total_work:.2%}")
        return partition


# ==================== Convenience Functions ====================

def generate_random_initial_solution(scheduler: HWSWPartitionScheduler,
                                     repair: bool = True,
                                     hw_probability: float = 0.5) -> np.ndarray:
    """
    Convenience function to generate a random initial solution.
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        repair: Whether to repair infeasible solutions
        hw_probability: Probability of assigning to HW
        
    Returns:
        Random partition
    """
    generator = RandomInitialSolution(repair_infeasible=repair)
    return generator.generate(scheduler, hw_probability=hw_probability)


def generate_greedy_initial_solution(scheduler: HWSWPartitionScheduler,
                                     repair: bool = True,
                                     scoring: str = 'speedup',
                                     hw_first: bool = True) -> np.ndarray:
    """
    Convenience function to generate a greedy initial solution.
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        repair: Whether to repair infeasible solutions
        scoring: Scoring method ('speedup' or 'area_efficiency')
        hw_first: If True, use HW-first greedy; else SW-first
        
    Returns:
        Greedy partition
    """
    if hw_first:
        generator = GreedyHardwareFirstSolution(repair_infeasible=repair)
    else:
        generator = GreedySoftwareFirstSolution(repair_infeasible=repair)
    
    return generator.generate(scheduler, scoring=scoring)


def generate_critical_path_solution(scheduler: HWSWPartitionScheduler,
                                    repair: bool = True,
                                    critical_path_ratio: float = 0.8) -> np.ndarray:
    """
    Convenience function to generate a critical-path-aware solution.
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        repair: Whether to repair infeasible solutions
        critical_path_ratio: Fraction of critical path to prioritize
        
    Returns:
        Critical-path-aware partition
    """
    generator = CriticalPathSolution(repair_infeasible=repair)
    return generator.generate(scheduler, critical_path_ratio=critical_path_ratio)


def generate_balanced_solution(scheduler: HWSWPartitionScheduler,
                               repair: bool = True,
                               target_hw_ratio: float = 0.5) -> np.ndarray:
    """
    Convenience function to generate a balanced solution.
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        repair: Whether to repair infeasible solutions
        target_hw_ratio: Target ratio of work on HW
        
    Returns:
        Balanced partition
    """
    generator = BalancedSolution(repair_infeasible=repair)
    return generator.generate(scheduler, target_hw_ratio=target_hw_ratio)


# ==================== Solution Generator Factory ====================

class InitialSolutionFactory:
    """
    Factory class to create initial solution generators by name.
    """
    
    _generators = {
        'random': RandomInitialSolution,
        'greedy_hw': GreedyHardwareFirstSolution,
        'greedy_sw': GreedySoftwareFirstSolution,
        'critical_path': CriticalPathSolution,
        'balanced': BalancedSolution
    }
    
    @classmethod
    def create(cls, strategy_name: str, repair_infeasible: bool = True) -> InitialSolutionGenerator:
        """
        Create an initial solution generator by name.
        
        Args:
            strategy_name: Name of the strategy
            repair_infeasible: Whether to repair infeasible solutions
            
        Returns:
            InitialSolutionGenerator instance
        """
        if strategy_name not in cls._generators:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Available: {list(cls._generators.keys())}")
        
        return cls._generators[strategy_name](repair_infeasible=repair_infeasible)
    
    @classmethod
    def list_strategies(cls):
        """List all available strategies."""
        return list(cls._generators.keys())


# ==================== Test/Demo ====================

if __name__ == "__main__":
    """Test initial solution generators."""
    from utils.task_graph_generation import create_data_lists, TaskGraphDataset
    import torch
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("="*70)
    logger.info("TESTING INITIAL SOLUTION GENERATORS")
    logger.info("="*70)
    
    # Generate test instance
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=1,
        min_nodes=15,
        max_nodes=15,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, 
                               edge_features_list, hw_area_limits)
    
    graph, adj_matrix, node_features, edge_features, hw_area_limit = dataset[0]
    
    # Create scheduler
    software_costs = {i: node_features[0, i].item() for i in range(len(graph.nodes()))}
    hardware_costs = {i: node_features[2, i].item() for i in range(len(graph.nodes()))}
    hardware_areas = {i: node_features[1, i].item() for i in range(len(graph.nodes()))}
    communication_costs = {e: edge_features[0, i].item() 
                          for i, e in enumerate(graph.edges())} if edge_features is not None else {}
    
    scheduler = HWSWPartitionScheduler(
        graph=graph,
        software_times=software_costs,
        hardware_times=hardware_costs,
        hardware_areas=hardware_areas,
        communication_costs=communication_costs,
        max_hardware_area=hw_area_limit.item()
    )
    
    logger.info(f"\nTest instance: {len(graph.nodes())} tasks, HW area limit: {hw_area_limit.item():.2f}")
    
    # Test all strategies
    strategies = InitialSolutionFactory.list_strategies()
    
    for strategy in strategies:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {strategy}")
        logger.info('='*70)
        
        generator = InitialSolutionFactory.create(strategy, repair_infeasible=True)
        partition = generator.generate(scheduler)
        solution = scheduler.evaluate_solution(partition)
        
        logger.info(f"Partition: {np.sum(partition)} HW tasks, {len(partition) - np.sum(partition)} SW tasks")
        logger.info(f"Total cost: {solution.total_cost:.4f}")
        logger.info(f"Makespan: {solution.makespan:.4f}")
        logger.info(f"Feasible: {generator._is_feasible(partition, scheduler)}")
    
    logger.info(f"\n{'='*70}")
    logger.info("All strategies tested successfully!")
    logger.info('='*70)