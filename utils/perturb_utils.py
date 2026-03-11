"""
ILS Perturbation Functions for Hardware-Software Partitioning with Scheduling

This module implements various perturbation strategies for the Iterated Local Search
algorithm applied to the HW/SW partitioning problem using a standardized interface.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.local_search import (
    HWSWPartitionScheduler,
    PartitionScheduleSolution
)
from utils.search_history import SearchHistory

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_perturbation.log")

logger = LogManager.get_logger(__name__)


# ============================================================================
# Base Class for Perturbations
# ============================================================================

class PerturbationStrategy(ABC):
    """
    Abstract base class for perturbation strategies.
    
    All perturbation strategies must implement the perturb() method with
    a standardized interface.
    """
    
    def __init__(self, name: str):
        """
        Initialize perturbation strategy.
        
        Args:
            name: Name of the perturbation strategy
        """
        self.name = name
        logger.info(f"Initialized perturbation strategy: {name}")
    
    @abstractmethod
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        """
        Apply perturbation to partition.
        
        Args:
            partition: Current partition to perturb
            scheduler: Scheduler instance for feasibility checking
            strength: Perturbation strength (interpretation depends on strategy)
            solution: Optional current solution (for extracting additional info)
            history: Optional search history (for adaptive strategies)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Perturbed partition
        """
        pass
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Phase 1: Core Perturbation Strategies
# ============================================================================

class RandomFlipsPerturbation(PerturbationStrategy):
    """
    Randomly flip `strength` tasks between HW/SW.
    
    Most basic perturbation strategy.
    """
    
    def __init__(self):
        super().__init__("random_flips")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        perturbed = partition.copy()
        n_tasks = len(partition)
        
        # Randomly select tasks to flip
        tasks_to_flip = np.random.choice(
            n_tasks, 
            size=min(strength, n_tasks), 
            replace=False
        )
        
        logger.info(f"Random flips: flipping {len(tasks_to_flip)} tasks: {tasks_to_flip}")
        
        # Flip selected tasks
        for task in tasks_to_flip:
            perturbed[task] = 1 - perturbed[task]
        
        return perturbed


class RandomSwapsPerturbation(PerturbationStrategy):
    """
    Perform `strength` random swap moves.
    
    Better feasibility preservation than random flips.
    Maintains HW/SW balance.
    """
    
    def __init__(self):
        super().__init__("random_swaps")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        perturbed = partition.copy()
        swaps_performed = 0
        
        for _ in range(strength):
            # Find HW and SW tasks
            hw_tasks = np.where(perturbed == 1)[0]
            sw_tasks = np.where(perturbed == 0)[0]
            
            # If we have tasks in both categories, swap one from each
            if len(hw_tasks) > 0 and len(sw_tasks) > 0:
                hw_task = np.random.choice(hw_tasks)
                sw_task = np.random.choice(sw_tasks)
                
                # Swap assignments
                perturbed[hw_task] = 0
                perturbed[sw_task] = 1
                swaps_performed += 1
        
        logger.info(f"Random swaps: performed {swaps_performed} swaps")
        
        return perturbed


class DoubleBridgePerturbation(PerturbationStrategy):
    """
    Perform a double-bridge move (inspired by TSP).
    
    Exchange two pairs of tasks (HW↔SW, SW↔HW).
    Fixed strength (4 tasks affected).
    Good for escaping local optima.
    """
    
    def __init__(self):
        super().__init__("double_bridge")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        perturbed = partition.copy()
        
        # Find HW and SW tasks
        hw_tasks = np.where(perturbed == 1)[0]
        sw_tasks = np.where(perturbed == 0)[0]
        
        # Need at least 2 tasks in each category
        if len(hw_tasks) >= 2 and len(sw_tasks) >= 2:
            # Select 2 HW tasks and 2 SW tasks
            hw_selected = np.random.choice(hw_tasks, size=2, replace=False)
            sw_selected = np.random.choice(sw_tasks, size=2, replace=False)
            
            logger.info(f"Double bridge: swapping HW tasks {hw_selected} with SW tasks {sw_selected}")
            
            # Swap them
            perturbed[hw_selected] = 0
            perturbed[sw_selected] = 1
        else:
            # Fallback to random flips if we don't have enough tasks
            logger.info(f"Double bridge: insufficient tasks (HW={len(hw_tasks)}, SW={len(sw_tasks)}), falling back to random flips")
            fallback = RandomFlipsPerturbation()
            return fallback.perturb(partition, scheduler, 4)
        
        return perturbed


# ============================================================================
# Phase 2: Advanced Perturbation Strategies
# ============================================================================

class GuidedCriticalPathPerturbation(PerturbationStrategy):
    """
    Perturb tasks on the critical path.
    
    Extract critical path and preferentially perturb those tasks.
    More likely to improve makespan.
    """
    
    def __init__(self):
        super().__init__("guided_critical_path")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        # Keyword argument
        solution = kwargs.get('solution', None)
        
        # Initialize perturbed partition
        perturbed = partition.copy()
        
        # Extract critical path
        if solution is not None:
            critical_path = extract_critical_path(solution, scheduler)
        else:
            logger.info("No solution provided, evaluating current partition to extract critical path")
            temp_solution = scheduler.evaluate_solution(partition)
            critical_path = extract_critical_path(temp_solution, scheduler)
        
        logger.info(f"Critical path contains {len(critical_path)} tasks: {critical_path}")
        
        if len(critical_path) == 0:
            logger.warning("Critical path is empty, falling back to random flips")
            fallback = RandomFlipsPerturbation()
            return fallback.perturb(partition, scheduler, strength)
        
        # Select tasks from critical path with higher probability
        n_from_critical = min(strength, len(critical_path))
        n_from_other = strength - n_from_critical
        
        # Select from critical path
        tasks_to_flip = list(np.random.choice(
            critical_path, 
            size=n_from_critical, 
            replace=False
        ))
        
        # Add some random tasks if needed
        if n_from_other > 0:
            other_tasks = [t for t in range(len(partition)) if t not in critical_path]
            if len(other_tasks) > 0:
                tasks_to_flip.extend(np.random.choice(
                    other_tasks,
                    size=min(n_from_other, len(other_tasks)),
                    replace=False
                ))
        
        logger.info(f"Guided critical path: flipping {len(tasks_to_flip)} tasks ({n_from_critical} from critical path)")
        
        # Flip selected tasks
        for task in tasks_to_flip:
            perturbed[task] = 1 - perturbed[task]
        
        return perturbed


class GuidedAreaIntensivePerturbation(PerturbationStrategy):
    """
    Target tasks with high HW area usage.
    
    Move from HW to SW to free up resources.
    Useful when near area constraint.
    """
    
    def __init__(self):
        super().__init__("guided_area_intensive")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        perturbed = partition.copy()
        
        # Find HW tasks sorted by area (descending)
        hw_tasks = np.where(perturbed == 1)[0]
        
        if len(hw_tasks) == 0:
            logger.warning("No HW tasks to move, falling back to random flips")
            fallback = RandomFlipsPerturbation()
            return fallback.perturb(partition, scheduler, strength)
        
        # Sort by hardware area (largest first)
        hw_areas = [(task, scheduler.hardware_areas[task]) for task in hw_tasks]
        hw_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Select top area-consuming tasks
        tasks_to_move = [task for task, _ in hw_areas[:min(strength, len(hw_areas))]]
        
        total_area_freed = sum(scheduler.hardware_areas[task] for task in tasks_to_move)
        logger.info(f"Guided area-intensive: moving {len(tasks_to_move)} high-area tasks from HW to SW (freeing {total_area_freed:.2f} area)")
        
        # Move from HW to SW
        for task in tasks_to_move:
            perturbed[task] = 0
        
        return perturbed


class ChainExchangePerturbation(PerturbationStrategy):
    """
    Exchange entire chains of dependent tasks.
    
    Maintains dependency locality.
    Inspired by job-shop perturbations.
    """
    
    def __init__(self, chain_length: int = 3):
        super().__init__("chain_exchange")
        self.default_chain_length = chain_length
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        # Keyword argument
        chain_length = kwargs.get('chain_length', None)
        
        # Initialize perturbed partition
        perturbed = partition.copy()
        graph = scheduler.graph
        
        # Use provided chain_length or default
        target_length = chain_length if chain_length is not None else self.default_chain_length
        
        # Start from a random task
        start_task = np.random.randint(0, len(partition))
        
        # Build a chain forward through successors
        chain = [start_task]
        current = start_task
        
        while len(chain) < target_length:
            successors = list(graph.successors(current))
            if len(successors) == 0:
                break
            # Randomly select next in chain
            current = random.choice(successors)
            chain.append(current)
        
        logger.info(f"Chain exchange: built chain of length {len(chain)} starting from task {start_task}: {chain}")
        
        # Flip all tasks in the chain
        for task in chain:
            perturbed[task] = 1 - perturbed[task]
        
        return perturbed


class AdaptiveStrengthPerturbation(PerturbationStrategy):
    """
    Adjust perturbation strength based on search progress.
    
    Increase if stagnating, decrease if improving rapidly.
    Wraps another perturbation strategy.
    """
    
    def __init__(
        self, 
        base_strategy: Optional[PerturbationStrategy] = None,
        base_strength: int = 3
    ):
        super().__init__("adaptive_strength")
        self.base_strategy = base_strategy if base_strategy else RandomFlipsPerturbation()
        self.base_strength = base_strength
        logger.info(f"Adaptive strength using base strategy: {self.base_strategy.name}")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        # Keyword argument
        history = kwargs.get('history', None)
        solution = kwargs.get('solution', None)

        # Calculate adaptive strength
        adapted_strength = strength if strength > 0 else self.base_strength
        original_strength = adapted_strength
        
        if history is not None:
            stagnation_level = history.get_stagnation_level()
            
            # Increase strength if stagnating
            if stagnation_level > 0.8:
                adapted_strength = int(adapted_strength * 2)
                logger.info(f"High stagnation ({stagnation_level:.2f}), doubling strength: {original_strength} -> {adapted_strength}")
            elif stagnation_level > 0.5:
                adapted_strength = int(adapted_strength * 1.5)
                logger.info(f"Moderate stagnation ({stagnation_level:.2f}), increasing strength: {original_strength} -> {adapted_strength}")
            # Decrease if improving rapidly
            elif stagnation_level < 0.2:
                adapted_strength = max(1, int(adapted_strength * 0.5))
                logger.info(f"Rapid improvement ({stagnation_level:.2f}), decreasing strength: {original_strength} -> {adapted_strength}")
        
        # Apply base strategy with adapted strength
        return self.base_strategy.perturb(
            partition, 
            scheduler, 
            adapted_strength,
            solution=solution,
            history=history,
            **kwargs
        )


class VariableNeighborhoodPerturbation(PerturbationStrategy):
    """
    VNS-style systematic neighborhood change.
    
    Uses k parameter to select different perturbation types.
    Increase k when no improvement, reset when improvement found.
    """
    
    def __init__(self, max_k: int = 5):
        super().__init__("variable_neighborhood")
        self.max_k = max_k
        
        # Define neighborhood strategies
        self.strategies = [
            RandomFlipsPerturbation(),
            RandomSwapsPerturbation(),
            DoubleBridgePerturbation(),
        ]
        logger.info(f"Variable neighborhood with max_k={max_k}, strategies: {[s.name for s in self.strategies]}")
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strength: int,
        k: int = 1,
        **kwargs
    ) -> np.ndarray:
        # Keyword arguments
        solution = kwargs.get('solution', None)
        history = kwargs.get('history', None)

        # Use k as the perturbation strength/strategy selector
        effective_k = min(k, self.max_k)
        
        # Select strategy based on k
        if effective_k <= 2:
            strategy = self.strategies[0]  # Random flips
            logger.info(f"VNS: k={effective_k}, using {strategy.name} with strength {effective_k}")
            return strategy.perturb(partition, scheduler, effective_k, solution, history)
        elif effective_k <= 4:
            strategy = self.strategies[1]  # Random swaps
            swap_strength = effective_k // 2
            logger.info(f"VNS: k={effective_k}, using {strategy.name} with strength {swap_strength}")
            return strategy.perturb(partition, scheduler, swap_strength, solution, history)
        else:
            strategy = self.strategies[2]  # Double bridge
            logger.info(f"VNS: k={effective_k}, using {strategy.name}")
            return strategy.perturb(partition, scheduler, effective_k, solution, history)


# ============================================================================
# Perturbation Manager
# ============================================================================

class PerturbationManager:
    """
    Manages multiple perturbation strategies and provides a unified interface.
    
    This class simplifies the use of different perturbation strategies by:
    - Providing a registry of available strategies
    - Standardizing the interface
    - Handling feasibility repair
    - Supporting easy strategy switching
    """
    
    def __init__(self, repair_infeasible: bool = True):
        """
        Initialize the perturbation manager.
        
        Args:
            repair_infeasible: Whether to automatically repair infeasible solutions
        """
        self.repair_infeasible = repair_infeasible
        self.strategies: Dict[str, PerturbationStrategy] = {}
        
        logger.info(f"Initializing PerturbationManager (repair_infeasible={repair_infeasible})")
        
        # Register default strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all default perturbation strategies."""
        self.register_strategy(RandomFlipsPerturbation())
        self.register_strategy(RandomSwapsPerturbation())
        self.register_strategy(DoubleBridgePerturbation())
        self.register_strategy(GuidedCriticalPathPerturbation())
        self.register_strategy(GuidedAreaIntensivePerturbation())
        self.register_strategy(ChainExchangePerturbation())
        self.register_strategy(AdaptiveStrengthPerturbation())
        self.register_strategy(VariableNeighborhoodPerturbation())
        
        logger.info(f"Registered {len(self.strategies)} default perturbation strategies")
    
    def register_strategy(self, strategy: PerturbationStrategy):
        """
        Register a new perturbation strategy.
        
        Args:
            strategy: Perturbation strategy to register
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
    
    def get_strategy(self, name: str) -> PerturbationStrategy:
        """
        Get a registered strategy by name.
        
        Args:
            name: Name of the strategy
            
        Returns:
            The requested perturbation strategy
        """
        if name not in self.strategies:
            logger.error(f"Unknown perturbation strategy: {name}")
            raise ValueError(f"Unknown perturbation strategy: {name}")
        return self.strategies[name]
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self.strategies.keys())
    
    def perturb(
        self,
        partition: np.ndarray,
        scheduler: HWSWPartitionScheduler,
        strategy_name: str,
        strength: int,
        **kwargs
    ) -> np.ndarray:
        """
        Apply perturbation using specified strategy.
        
        Args:
            partition: Current partition
            scheduler: Scheduler instance
            strategy_name: Name of strategy to use
            strength: Perturbation strength
            **kwargs: Strategy-specific parameters
            
        Returns:
            Perturbed partition
        """

        logger.info(f"Applying perturbation: {strategy_name} with strength {strength}")
        
        strategy = self.get_strategy(strategy_name)
        
        # Apply perturbation
        perturbed = strategy.perturb(
            partition,
            scheduler,
            strength,
            **kwargs
        )
        
        # Check feasibility
        is_feasible = scheduler.is_feasible(perturbed)
        
        # Repair if needed
        if self.repair_infeasible and not is_feasible:
            logger.info(f"Perturbed solution is infeasible, repairing...")
            perturbed = scheduler.repair_solution(perturbed)
            logger.info("Solution repaired")
        elif not is_feasible:
            logger.warning(f"Perturbed solution is infeasible and repair is disabled")
        
        return perturbed


# ============================================================================
# Utility Functions
# ============================================================================

def extract_critical_path(
    solution: PartitionScheduleSolution,
    scheduler: HWSWPartitionScheduler
) -> List[int]:
    """
    Extract the critical path from a solution.
    
    The critical path is the longest path from any source node to any
    sink node in terms of execution time.
    
    Args:
        solution: Current solution
        scheduler: Scheduler instance
        
    Returns:
        List of task IDs on the critical path
    """
    graph = scheduler.graph
    schedule = solution.schedule
    
    # Find the task that finishes last (end of critical path)
    finish_times = {}
    for task in graph.nodes():
        start_time = schedule[task]
        if solution.partition[task] == 1:  # HW
            exec_time = scheduler.hardware_times[task]
        else:  # SW
            exec_time = scheduler.software_times[task]
        finish_times[task] = start_time + exec_time
    
    # Task with maximum finish time
    end_task = max(finish_times, key=finish_times.get)
    
    logger.info(f"Critical path ends at task {end_task} (finish time: {finish_times[end_task]:.2f})")
    
    # Trace back through predecessors to find critical path
    critical_path = [end_task]
    current = end_task
    
    while True:
        predecessors = list(graph.predecessors(current))
        if len(predecessors) == 0:
            break
        
        # Find predecessor that determines start time
        best_pred = None
        best_time = -1
        
        for pred in predecessors:
            pred_finish = finish_times[pred]
            
            # Add communication cost if on different resource
            if solution.partition[pred] != solution.partition[current]:
                comm_cost = scheduler.communication_costs.get((pred, current), 0.0)
                pred_finish += comm_cost
            
            if pred_finish > best_time:
                best_time = pred_finish
                best_pred = pred
        
        if best_pred is None:
            break
        
        critical_path.append(best_pred)
        current = best_pred
    
    # Reverse to get path from start to end
    critical_path.reverse()
    
    logger.info(f"Extracted critical path with {len(critical_path)} tasks")
    
    return critical_path


def calculate_perturbation_strength(
    base_strength: int,
    n_tasks: int,
    instance_size_scaling: bool = True,
    adaptive_factor: float = 1.0,
    min_strength: int = 1,
    max_strength: Optional[int] = None
) -> int:
    """
    Calculate perturbation strength with optional scaling and adaptation.
    
    Args:
        base_strength: Base perturbation strength
        n_tasks: Number of tasks in the instance
        instance_size_scaling: Whether to scale with instance size
        adaptive_factor: Adaptive factor (from search progress)
        min_strength: Minimum strength
        max_strength: Maximum strength (if None, use n_tasks)
        
    Returns:
        Calculated perturbation strength
    """
    strength = base_strength
    
    # Scale with instance size if requested
    if instance_size_scaling:
        # Scale logarithmically with instance size
        scale_factor = np.log(n_tasks + 1) / np.log(10)
        strength = int(base_strength * scale_factor)
    
    # Apply adaptive factor
    strength = int(strength * adaptive_factor)
    
    # Enforce bounds
    strength = max(min_strength, strength)
    if max_strength is None:
        max_strength = n_tasks
    strength = min(strength, max_strength)
    
    logger.info(f"Calculated perturbation strength: {strength} (base={base_strength}, n_tasks={n_tasks}, scale={instance_size_scaling}, factor={adaptive_factor:.2f})")
    
    return strength


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import torch
    from task_graph_generation import create_data_lists, TaskGraphDataset
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("="*70)
    logger.info("PERTURBATION STRATEGIES DEMO")
    logger.info("="*70)
    
    # Generate a test instance
    logger.info("Generating test instance...")
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=1,
        min_nodes=10,
        max_nodes=10,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, 
                               edge_features_list, hw_area_limits)
    
    # Get instance
    graph, adj_matrix, node_features, edge_features, hw_area_limit = dataset[0]
    
    logger.info(f"Generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
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
    
    # Generate initial partition
    n_tasks = len(graph.nodes())
    initial_partition = np.random.randint(0, 2, size=n_tasks)
    
    # Evaluate initial solution
    initial_solution = scheduler.evaluate_solution(initial_partition)
    logger.info(f"Initial solution cost: {initial_solution.total_cost:.4f}")
    logger.info(f"Initial partition: {initial_partition}")
    logger.info(f"Feasible: {scheduler.is_feasible(initial_partition)}")
    
    # Create perturbation manager
    manager = PerturbationManager(repair_infeasible=True)
    
    logger.info(f"Available strategies: {manager.list_strategies()}")
    
    # Test each strategy
    logger.info("="*70)
    logger.info("TESTING STRATEGIES")
    logger.info("="*70)
    
    test_configs = [
        ('random_flips', 3, {}),
        ('random_swaps', 2, {}),
        ('double_bridge', 0, {}),
        ('guided_critical_path', 3, {'solution': initial_solution}),
        ('guided_area_intensive', 2, {}),
        ('chain_exchange', 0, {'chain_length': 3}),
    ]
    
    for strategy_name, strength, kwargs in test_configs:
        logger.info(f"\nTesting strategy: {strategy_name}")
        
        perturbed = manager.perturb(
            initial_partition,
            scheduler,
            strategy_name,
            strength,
            **kwargs
        )
        
        # Evaluate perturbed solution
        perturbed_solution = scheduler.evaluate_solution(perturbed)
        
        # Calculate Hamming distance
        hamming_dist = np.sum(perturbed != initial_partition)
        
        logger.info(f"  Hamming distance: {hamming_dist}")
        logger.info(f"  Cost: {perturbed_solution.total_cost:.4f}")
        logger.info(f"  Delta: {perturbed_solution.total_cost - initial_solution.total_cost:+.4f}")
        logger.info(f"  Feasible: {scheduler.is_feasible(perturbed)}")