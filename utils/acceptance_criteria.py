"""
Acceptance Criteria for Iterated Local Search (ILS).

This module provides various acceptance criteria strategies for deciding
whether to accept a new solution in the ILS algorithm:
- Better (only accept improvements)
- Simulated Annealing-like (accept worse with probability)
- Threshold Accepting (accept if within threshold)
- Late Acceptance (compare with older solutions)
- Record-to-Record Travel (accept if within % of best)
"""

import numpy as np
import random
import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.local_search import PartitionScheduleSolution, HWSWPartitionScheduler
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_acceptance_criteria.log")
logger = LogManager.get_logger(__name__)


class AcceptanceCriterion(ABC):
    """
    Abstract base class for acceptance criteria.
    
    All concrete criteria must implement the accept() method.
    """
    
    def __init__(self):
        """Initialize the acceptance criterion."""
        self.iteration = 0
        self.accepts = 0
        self.rejects = 0
    
    @abstractmethod
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Decide whether to accept the new solution.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Best solution found so far (optional)
            **kwargs: Additional criterion-specific parameters
            
        Returns:
            True if new solution should be accepted, False otherwise
        """
        pass
    
    def reset(self):
        """Reset acceptance statistics."""
        self.iteration = 0
        self.accepts = 0
        self.rejects = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get acceptance statistics."""
        total = self.accepts + self.rejects
        acceptance_rate = self.accepts / total if total > 0 else 0.0
        
        return {
            'iterations': self.iteration,
            'accepts': self.accepts,
            'rejects': self.rejects,
            'acceptance_rate': acceptance_rate
        }


class BetterAcceptance(AcceptanceCriterion):
    """
    Accept only if new solution is better (improvement).
    
    This is the most conservative criterion - only improvements are accepted.
    """
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept only if new solution is strictly better.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Not used
            
        Returns:
            True if new_solution.total_cost < current_solution.total_cost
        """
        self.iteration += 1
        
        # Accept if better (lower cost)
        accept = new_solution.total_cost < current_solution.total_cost
        
        if accept:
            self.accepts += 1
            logger.debug(f"Accepted: new cost {new_solution.total_cost:.4f} < "
                        f"current {current_solution.total_cost:.4f}")
        else:
            self.rejects += 1
        
        return accept


class AlwaysAcceptance(AcceptanceCriterion):
    """
    Always accept the new solution.
    
    This is useful for testing and for pure random walk strategies.
    """
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """Always accept."""
        self.iteration += 1
        self.accepts += 1
        return True


class SimulatedAnnealingAcceptance(AcceptanceCriterion):
    """
    Simulated Annealing acceptance criterion.
    
    Accept worse solutions with probability exp(-delta / temperature).
    Temperature decreases over iterations.
    """
    
    def __init__(self, initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 0.01):
        """
        Initialize SA acceptance criterion.
        
        Args:
            initial_temperature: Starting temperature
            cooling_rate: Temperature multiplier per iteration (< 1.0)
            min_temperature: Minimum temperature
        """
        super().__init__()
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept based on SA criterion.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Not used
            
        Returns:
            True if accepted
        """
        self.iteration += 1
        
        # Calculate cost difference
        delta = new_solution.total_cost - current_solution.total_cost
        
        # Always accept improvements
        if delta < 0:
            self.accepts += 1
            logger.debug(f"SA: Accepted improvement, delta={delta:.4f}")
            return True
        
        # Accept worse with probability
        if self.temperature > 0:
            acceptance_prob = math.exp(-delta / self.temperature)
        else:
            acceptance_prob = 0.0
        
        accept = random.random() < acceptance_prob
        
        if accept:
            self.accepts += 1
            logger.debug(f"SA: Accepted worse, delta={delta:.4f}, "
                        f"temp={self.temperature:.4f}, prob={acceptance_prob:.4f}")
        else:
            self.rejects += 1
        
        # Cool down
        self.temperature = max(self.min_temperature, 
                              self.temperature * self.cooling_rate)
        
        return accept
    
    def reset(self):
        """Reset and restore initial temperature."""
        super().reset()
        self.temperature = self.initial_temperature


class ThresholdAcceptance(AcceptanceCriterion):
    """
    Threshold Accepting criterion.
    
    Accept if new cost is within a threshold of current cost.
    Threshold decreases over iterations.
    """
    
    def __init__(self, initial_threshold: float = 10.0,
                 threshold_decay: float = 0.95,
                 min_threshold: float = 0.0):
        """
        Initialize threshold acceptance.
        
        Args:
            initial_threshold: Starting threshold
            threshold_decay: Threshold multiplier per iteration
            min_threshold: Minimum threshold
        """
        super().__init__()
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.threshold_decay = threshold_decay
        self.min_threshold = min_threshold
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept if within threshold.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Not used
            
        Returns:
            True if accepted
        """
        self.iteration += 1
        
        # Accept if within threshold
        delta = new_solution.total_cost - current_solution.total_cost
        accept = delta <= self.threshold
        
        if accept:
            self.accepts += 1
            logger.debug(f"TA: Accepted, delta={delta:.4f} <= threshold={self.threshold:.4f}")
        else:
            self.rejects += 1
        
        # Decay threshold
        self.threshold = max(self.min_threshold,
                            self.threshold * self.threshold_decay)
        
        return accept
    
    def reset(self):
        """Reset and restore initial threshold."""
        super().reset()
        self.threshold = self.initial_threshold


class LateAcceptance(AcceptanceCriterion):
    """
    Late Acceptance Hill Climbing (LAHC).
    
    Compare new solution with solution from L iterations ago.
    Maintains a list of costs from previous iterations.
    """
    
    def __init__(self, list_length: int = 50):
        """
        Initialize LAHC.
        
        Args:
            list_length: Number of previous costs to maintain
        """
        super().__init__()
        self.list_length = list_length
        self.cost_list = []
        self.current_index = 0
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept based on late acceptance criterion.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Not used
            
        Returns:
            True if accepted
        """
        self.iteration += 1
        
        # Initialize cost list if needed
        if len(self.cost_list) < self.list_length:
            self.cost_list.append(current_solution.total_cost)
            accept = new_solution.total_cost <= current_solution.total_cost
        else:
            # Compare with old cost
            old_cost = self.cost_list[self.current_index]
            accept = new_solution.total_cost <= old_cost
            
            # Update list
            self.cost_list[self.current_index] = current_solution.total_cost
            self.current_index = (self.current_index + 1) % self.list_length
        
        if accept:
            self.accepts += 1
            logger.debug(f"LAHC: Accepted, new={new_solution.total_cost:.4f}")
        else:
            self.rejects += 1
        
        return accept
    
    def reset(self):
        """Reset cost list."""
        super().reset()
        self.cost_list = []
        self.current_index = 0


class RecordToRecordTravel(AcceptanceCriterion):
    """
    Record-to-Record Travel (RRT).
    
    Accept if new solution is within a percentage (deviation) of the best solution.
    """
    
    def __init__(self, deviation_percentage: float = 0.05):
        """
        Initialize RRT.
        
        Args:
            deviation_percentage: Allowed deviation from best (0.05 = 5%)
        """
        super().__init__()
        self.deviation_percentage = deviation_percentage
        self.record_cost = float('inf')
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept based on RRT criterion.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Best solution found so far (required for RRT)
            
        Returns:
            True if accepted
        """
        self.iteration += 1
        
        # Update record if best solution provided
        if best_solution is not None:
            self.record_cost = min(self.record_cost, best_solution.total_cost)
        else:
            # Use current best
            self.record_cost = min(self.record_cost, current_solution.total_cost)
        
        # Calculate threshold
        threshold = self.record_cost * (1.0 + self.deviation_percentage)
        
        # Accept if within threshold
        accept = new_solution.total_cost <= threshold
        
        if accept:
            self.accepts += 1
            logger.debug(f"RRT: Accepted, cost={new_solution.total_cost:.4f} <= "
                        f"threshold={threshold:.4f} (record={self.record_cost:.4f})")
        else:
            self.rejects += 1
        
        return accept
    
    def reset(self):
        """Reset record."""
        super().reset()
        self.record_cost = float('inf')


class GreatDelugeAcceptance(AcceptanceCriterion):
    """
    Great Deluge Algorithm acceptance.
    
    Accept if new solution is below a dynamically decreasing water level.
    """
    
    def __init__(self, initial_water_level: Optional[float] = None,
                 rain_speed: float = 0.01):
        """
        Initialize Great Deluge.
        
        Args:
            initial_water_level: Starting water level (if None, use first solution)
            rain_speed: Rate at which water level decreases
        """
        super().__init__()
        self.initial_water_level = initial_water_level
        self.water_level = initial_water_level
        self.rain_speed = rain_speed
        self.initialized = False
    
    def accept(self, current_solution: PartitionScheduleSolution,
               new_solution: PartitionScheduleSolution,
               best_solution: Optional[PartitionScheduleSolution] = None,
               **kwargs) -> bool:
        """
        Accept based on Great Deluge criterion.
        
        Args:
            current_solution: Current solution
            new_solution: New candidate solution
            best_solution: Not used
            
        Returns:
            True if accepted
        """
        self.iteration += 1
        
        # Initialize water level if needed
        if not self.initialized:
            if self.water_level is None:
                self.water_level = current_solution.total_cost * 1.1
            self.initialized = True
        
        # Accept if below water level
        accept = new_solution.total_cost <= self.water_level
        
        if accept:
            self.accepts += 1
            logger.debug(f"GD: Accepted, cost={new_solution.total_cost:.4f} <= "
                        f"water_level={self.water_level:.4f}")
        else:
            self.rejects += 1
        
        # Lower water level
        self.water_level -= self.rain_speed
        
        return accept
    
    def reset(self):
        """Reset water level."""
        super().reset()
        self.water_level = self.initial_water_level
        self.initialized = False


# ==================== Convenience Functions ====================

def accept_better(current_solution: PartitionScheduleSolution,
                 new_solution: PartitionScheduleSolution) -> bool:
    """
    Convenience function: accept only if better.
    
    Args:
        current_solution: Current solution
        new_solution: New candidate solution
        
    Returns:
        True if new solution is better
    """
    criterion = BetterAcceptance()
    return criterion.accept(current_solution, new_solution)


def accept_simulated_annealing(current_solution: PartitionScheduleSolution,
                              new_solution: PartitionScheduleSolution,
                              temperature: float = 100.0) -> bool:
    """
    Convenience function: accept with SA criterion.
    
    Args:
        current_solution: Current solution
        new_solution: New candidate solution
        temperature: Current temperature
        
    Returns:
        True if accepted
    """
    criterion = SimulatedAnnealingAcceptance(initial_temperature=temperature)
    return criterion.accept(current_solution, new_solution)


def accept_threshold(current_solution: PartitionScheduleSolution,
                    new_solution: PartitionScheduleSolution,
                    threshold: float = 10.0) -> bool:
    """
    Convenience function: accept within threshold.
    
    Args:
        current_solution: Current solution
        new_solution: New candidate solution
        threshold: Acceptance threshold
        
    Returns:
        True if accepted
    """
    criterion = ThresholdAcceptance(initial_threshold=threshold)
    return criterion.accept(current_solution, new_solution)


# ==================== Acceptance Criterion Factory ====================

class AcceptanceCriterionFactory:
    """
    Factory class to create acceptance criteria by name.
    """
    
    _criteria = {
        'better': BetterAcceptance,
        'always': AlwaysAcceptance,
        'simulated_annealing': SimulatedAnnealingAcceptance,
        'threshold': ThresholdAcceptance,
        'late_acceptance': LateAcceptance,
        'rrt': RecordToRecordTravel,
        'great_deluge': GreatDelugeAcceptance
    }
    
    @classmethod
    def create(cls, criterion_name: str, **kwargs) -> AcceptanceCriterion:
        """
        Create an acceptance criterion by name.
        
        Args:
            criterion_name: Name of the criterion
            **kwargs: Criterion-specific parameters
            
        Returns:
            AcceptanceCriterion instance
        """
        if criterion_name not in cls._criteria:
            raise ValueError(f"Unknown criterion: {criterion_name}. "
                           f"Available: {list(cls._criteria.keys())}")
        
        return cls._criteria[criterion_name](**kwargs)
    
    @classmethod
    def list_criteria(cls):
        """List all available criteria."""
        return list(cls._criteria.keys())


# ==================== Test/Demo ====================

if __name__ == "__main__":
    """Test acceptance criteria."""
    from utils.task_graph_generation import create_data_lists, TaskGraphDataset
    import torch
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("="*70)
    logger.info("TESTING ACCEPTANCE CRITERIA")
    logger.info("="*70)
    
    # Generate test instance
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=1,
        min_nodes=10,
        max_nodes=10,
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
    
    # Create test solutions
    partition1 = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    partition2 = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    partition3 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    
    # Repair if needed
    hw_area1 = sum(scheduler.hardware_areas[i] for i in range(len(partition1)) if partition1[i] == 1)
    if hw_area1 > scheduler.max_hardware_area:
        partition1[2] = 0
    
    solution1 = scheduler.evaluate_solution(partition1)
    solution2 = scheduler.evaluate_solution(partition2)
    solution3 = scheduler.evaluate_solution(partition3)
    
    logger.info(f"\nSolution 1 cost: {solution1.total_cost:.4f}")
    logger.info(f"Solution 2 cost: {solution2.total_cost:.4f}")
    logger.info(f"Solution 3 cost: {solution3.total_cost:.4f}")
    
    # Test each criterion
    criteria_configs = [
        ('better', {}),
        ('simulated_annealing', {'initial_temperature': 50.0}),
        ('threshold', {'initial_threshold': 5.0}),
        ('late_acceptance', {'list_length': 10}),
        ('rrt', {'deviation_percentage': 0.1}),
        ('great_deluge', {'rain_speed': 0.1})
    ]
    
    for criterion_name, config in criteria_configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {criterion_name}")
        logger.info('='*70)
        
        criterion = AcceptanceCriterionFactory.create(criterion_name, **config)
        
        # Test multiple iterations
        current = solution1
        for i, new in enumerate([solution2, solution3, solution1, solution2]):
            accepted = criterion.accept(current, new, best_solution=solution1)
            logger.info(f"Iteration {i+1}: Current={current.total_cost:.4f}, "
                  f"New={new.total_cost:.4f}, Accepted={accepted}")
            if accepted:
                current = new
        
        stats = criterion.get_statistics()
        logger.info(f"\nStatistics: {stats}")
    
    logger.info(f"\n{'='*70}")
    logger.info("All criteria tested successfully!")
    logger.info('='*70)