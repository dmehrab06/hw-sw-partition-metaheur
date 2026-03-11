"""
Search History Tracking for Iterated Local Search

This module implements the SearchHistory class for tracking and analyzing
the search trajectory during ILS execution.
"""

import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/search_history.log")

logger = LogManager.get_logger(__name__)

@dataclass
class SearchHistory:
    """
    Track search trajectory for adaptive strategies and analysis.
    
    This class records the complete history of an ILS run, including:
    - Solution costs over time
    - Perturbations applied
    - Acceptance decisions
    - Timing information
    
    This information enables:
    - Adaptive perturbation strength
    - Adaptive acceptance criteria
    - Post-hoc analysis and visualization
    - Detection of stagnation
    """
    
    # Core trajectory
    iterations: List[int] = field(default_factory=list)
    current_costs: List[float] = field(default_factory=list)
    best_costs: List[float] = field(default_factory=list)
    
    # Perturbation tracking
    perturbations_applied: List[str] = field(default_factory=list)
    perturbation_strengths: List[int] = field(default_factory=list)
    
    # Acceptance tracking
    acceptance_decisions: List[bool] = field(default_factory=list)
    acceptance_reasons: List[str] = field(default_factory=list)
    
    # Feasibility tracking
    infeasible_attempts: int = 0
    repairs_performed: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    iteration_times: List[float] = field(default_factory=list)
    
    # Additional metrics
    local_search_times: List[float] = field(default_factory=list)
    perturbation_times: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize timing."""
        self.start_time = time.time()
    
    def update(
        self,
        iteration: int,
        current_cost: float,
        best_cost: float,
        perturbation_type: str,
        perturbation_strength: int,
        accepted: bool,
        acceptance_reason: str,
        iteration_time: Optional[float] = None,
        local_search_time: Optional[float] = None,
        perturbation_time: Optional[float] = None
    ):
        """
        Record iteration data.
        
        Args:
            iteration: Iteration number
            current_cost: Cost of current solution
            best_cost: Best cost found so far
            perturbation_type: Type of perturbation applied
            perturbation_strength: Strength of perturbation
            accepted: Whether new solution was accepted
            acceptance_reason: Reason for acceptance/rejection
            iteration_time: Time taken for this iteration
            local_search_time: Time spent in local search
            perturbation_time: Time spent in perturbation
        """
        self.iterations.append(iteration)
        self.current_costs.append(current_cost)
        self.best_costs.append(best_cost)
        self.perturbations_applied.append(perturbation_type)
        self.perturbation_strengths.append(perturbation_strength)
        self.acceptance_decisions.append(accepted)
        self.acceptance_reasons.append(acceptance_reason)
        
        if iteration_time is not None:
            self.iteration_times.append(iteration_time)
        if local_search_time is not None:
            self.local_search_times.append(local_search_time)
        if perturbation_time is not None:
            self.perturbation_times.append(perturbation_time)
    
    def record_infeasible(self):
        """Record an infeasible solution attempt."""
        self.infeasible_attempts += 1
    
    def record_repair(self):
        """Record a repair operation."""
        self.repairs_performed += 1
    
    def get_stagnation_level(self, window: int = 50) -> float:
        """
        Return measure of how stuck the search is.
        
        Computes the ratio of iterations without improvement in recent history.
        
        Args:
            window: Number of recent iterations to consider
            
        Returns:
            Stagnation level: 0.0 = improving, 1.0 = completely stuck
        """
        if len(self.best_costs) < 2:
            return 0.0
        
        # Look at recent history
        recent_best = self.best_costs[-window:] if len(self.best_costs) >= window else self.best_costs
        
        if len(recent_best) < 2:
            return 0.0
        
        # Count iterations without improvement
        improvements = 0
        for i in range(1, len(recent_best)):
            if recent_best[i] < recent_best[i-1]:
                improvements += 1
        
        # Stagnation level is inversely related to improvement frequency
        stagnation = 1.0 - (improvements / (len(recent_best) - 1))
        
        return stagnation
    
    def get_improvement_rate(self, window: int = 50) -> float:
        """
        Average improvement per iteration in recent history.
        
        Args:
            window: Number of recent iterations to consider
            
        Returns:
            Average improvement per iteration (negative means worsening)
        """
        if len(self.best_costs) < 2:
            return 0.0
        
        # Look at recent history
        recent_best = self.best_costs[-window:] if len(self.best_costs) >= window else self.best_costs
        
        if len(recent_best) < 2:
            return 0.0
        
        # Total improvement over window
        total_improvement = recent_best[0] - recent_best[-1]
        
        # Average per iteration
        avg_improvement = total_improvement / (len(recent_best) - 1)
        
        return avg_improvement
    
    def should_increase_perturbation(
        self, 
        stagnation_threshold: float = 0.8,
        window: int = 50
    ) -> bool:
        """
        Heuristic: increase perturbation if stagnating.
        
        Args:
            stagnation_threshold: Threshold above which to increase
            window: Window for stagnation calculation
            
        Returns:
            True if perturbation should be increased
        """
        return self.get_stagnation_level(window) > stagnation_threshold
    
    def should_restart(
        self, 
        criterion: str = 'iterations',
        threshold: int = 100
    ) -> bool:
        """
        Decide if restart is needed.
        
        Args:
            criterion: Restart criterion ('iterations', 'stagnation', 'time')
            threshold: Threshold value for the criterion
            
        Returns:
            True if restart is recommended
        """
        if criterion == 'iterations':
            # Restart if no improvement for threshold iterations
            if len(self.best_costs) < threshold + 1:
                return False
            
            recent_best = self.best_costs[-threshold:]
            return len(set(recent_best)) == 1  # All same = stagnant
        
        elif criterion == 'stagnation':
            # Restart if stagnation level exceeds threshold
            return self.get_stagnation_level() > (threshold / 100.0)
        
        elif criterion == 'time':
            # Restart if time exceeds threshold
            elapsed = time.time() - self.start_time
            return elapsed > threshold
        
        return False
    
    def get_acceptance_rate(self, window: Optional[int] = None) -> float:
        """
        Calculate acceptance rate over recent history.
        
        Args:
            window: Number of recent iterations (None = all)
            
        Returns:
            Acceptance rate (0.0 to 1.0)
        """
        if len(self.acceptance_decisions) == 0:
            return 0.0
        
        decisions = self.acceptance_decisions[-window:] if window else self.acceptance_decisions
        
        if len(decisions) == 0:
            return 0.0
        
        return sum(decisions) / len(decisions)
    
    def get_summary_statistics(self) -> Dict:
        """
        Return comprehensive statistics.
        
        Returns:
            Dictionary containing summary statistics
        """
        if len(self.iterations) == 0:
            return {
                'total_iterations': 0,
                'total_time': 0.0,
                'best_cost': None,
                'initial_cost': None,
                'final_cost': None,
                'total_improvement': None,
                'acceptance_rate': 0.0,
                'infeasible_rate': 0.0,
                'stagnation_level': 0.0
            }
        
        total_time = time.time() - self.start_time
        
        stats = {
            'total_iterations': len(self.iterations),
            'total_time': total_time,
            'avg_iteration_time': np.mean(self.iteration_times) if self.iteration_times else None,
            'best_cost': min(self.best_costs),
            'initial_cost': self.current_costs[0],
            'final_cost': self.current_costs[-1],
            'total_improvement': self.current_costs[0] - min(self.best_costs),
            'improvement_percentage': ((self.current_costs[0] - min(self.best_costs)) / 
                                      self.current_costs[0] * 100) if self.current_costs[0] > 0 else 0,
            'acceptance_rate': self.get_acceptance_rate(),
            'infeasible_attempts': self.infeasible_attempts,
            'repairs_performed': self.repairs_performed,
            'infeasible_rate': self.infeasible_attempts / len(self.iterations) if len(self.iterations) > 0 else 0,
            'stagnation_level': self.get_stagnation_level(),
            'improvement_rate': self.get_improvement_rate(),
        }
        
        # Perturbation statistics
        if self.perturbations_applied:
            perturbation_counts = {}
            for p in self.perturbations_applied:
                perturbation_counts[p] = perturbation_counts.get(p, 0) + 1
            stats['perturbation_counts'] = perturbation_counts
            stats['avg_perturbation_strength'] = np.mean(self.perturbation_strengths)
        
        # Timing statistics
        if self.local_search_times:
            stats['total_local_search_time'] = sum(self.local_search_times)
            stats['avg_local_search_time'] = np.mean(self.local_search_times)
            stats['local_search_time_percentage'] = (sum(self.local_search_times) / 
                                                     total_time * 100) if total_time > 0 else 0
        
        if self.perturbation_times:
            stats['total_perturbation_time'] = sum(self.perturbation_times)
            stats['avg_perturbation_time'] = np.mean(self.perturbation_times)
            stats['perturbation_time_percentage'] = (sum(self.perturbation_times) / 
                                                     total_time * 100) if total_time > 0 else 0
        
        return stats
    
    def get_improvement_iterations(self) -> List[int]:
        """
        Get list of iterations where improvement occurred.
        
        Returns:
            List of iteration numbers with improvement
        """
        improvements = []
        
        for i in range(1, len(self.best_costs)):
            if self.best_costs[i] < self.best_costs[i-1]:
                improvements.append(self.iterations[i])
        
        return improvements
    
    def get_cost_trajectory(self) -> Dict[str, List]:
        """
        Get cost trajectory data for plotting.
        
        Returns:
            Dictionary with iterations, current costs, and best costs
        """
        return {
            'iterations': self.iterations.copy(),
            'current_costs': self.current_costs.copy(),
            'best_costs': self.best_costs.copy()
        }
    
    def print_summary(self):
        """Print a human-readable summary of the search."""
        stats = self.get_summary_statistics()
        
        logger.info("=" * 70)
        logger.info("SEARCH HISTORY SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Iterations: {stats['total_iterations']}")
        logger.info(f"Total Time: {stats['total_time']:.2f} seconds")
        if stats['avg_iteration_time']:
            logger.info(f"Avg Iteration Time: {stats['avg_iteration_time']:.4f} seconds")
        
        
        logger.info("COST TRAJECTORY:")
        logger.info(f"  Initial Cost: {stats['initial_cost']:.4f}")
        logger.info(f"  Final Cost: {stats['final_cost']:.4f}")
        logger.info(f"  Best Cost: {stats['best_cost']:.4f}")
        logger.info(f"  Total Improvement: {stats['total_improvement']:.4f} ({stats['improvement_percentage']:.2f}%)")
        
        
        logger.info("SEARCH BEHAVIOR:")
        logger.info(f"  Acceptance Rate: {stats['acceptance_rate']:.2%}")
        logger.info(f"  Stagnation Level: {stats['stagnation_level']:.2%}")
        logger.info(f"  Improvement Rate: {stats['improvement_rate']:.4f} per iteration")
        
        
        logger.info("FEASIBILITY:")
        logger.info(f"  Infeasible Attempts: {stats['infeasible_attempts']}")
        logger.info(f"  Repairs Performed: {stats['repairs_performed']}")
        logger.info(f"  Infeasible Rate: {stats['infeasible_rate']:.2%}")
        
        
        if 'perturbation_counts' in stats:
            logger.info("PERTURBATIONS:")
            for pert_type, count in stats['perturbation_counts'].items():
                percentage = count / stats['total_iterations'] * 100
                logger.info(f"  {pert_type}: {count} ({percentage:.1f}%)")
            logger.info(f"  Avg Strength: {stats['avg_perturbation_strength']:.1f}")
            
        
        if 'total_local_search_time' in stats:
            logger.info("TIMING BREAKDOWN:")
            logger.info(f"  Local Search: {stats['total_local_search_time']:.2f}s ({stats['local_search_time_percentage']:.1f}%)")
            if 'total_perturbation_time' in stats:
                logger.info(f"  Perturbation: {stats['total_perturbation_time']:.2f}s ({stats['perturbation_time_percentage']:.1f}%)")
        
        logger.info("=" * 70)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import random
    
    # Simulate an ILS run
    logger.info("Simulating ILS run...")
    
    history = SearchHistory()
    
    current_cost = 100.0
    best_cost = 100.0
    
    for i in range(100):
        # Simulate perturbation and local search
        perturbation_type = random.choice(['random_flips', 'random_swaps', 'double_bridge'])
        perturbation_strength = random.randint(2, 5)
        
        # Simulate cost change
        delta = random.gauss(-0.5, 2.0)  # Slight bias towards improvement
        new_cost = max(1.0, current_cost + delta)
        
        # Acceptance decision
        accepted = new_cost <= current_cost or random.random() < 0.3
        
        if accepted:
            current_cost = new_cost
            acceptance_reason = "better" if new_cost <= current_cost else "diversification"
        else:
            acceptance_reason = "rejected"
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
        
        # Record iteration
        history.update(
            iteration=i,
            current_cost=current_cost,
            best_cost=best_cost,
            perturbation_type=perturbation_type,
            perturbation_strength=perturbation_strength,
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            iteration_time=random.uniform(0.1, 0.5),
            local_search_time=random.uniform(0.05, 0.3),
            perturbation_time=random.uniform(0.01, 0.1)
        )
        
        # Simulate some infeasible attempts
        if random.random() < 0.1:
            history.record_infeasible()
            history.record_repair()
    
    # Print summary
    history.print_summary()
    
    # Test adaptive methods
    logger.info("\nADAPTIVE QUERIES:")
    logger.info(f"Stagnation level (last 50): {history.get_stagnation_level(50):.2%}")
    logger.info(f"Should increase perturbation? {history.should_increase_perturbation()}")
    logger.info(f"Should restart? {history.should_restart('iterations', 50)}")
    logger.info(f"Improvement iterations: {len(history.get_improvement_iterations())}")
    
    # Test trajectory extraction
    trajectory = history.get_cost_trajectory()
    logger.info(f"\nTrajectory data points: {len(trajectory['iterations'])}")