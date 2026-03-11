"""
Local search for constrained hardware-software partitioning with scheduling.

This module implements local search for the problem of:
1. Partitioning tasks between hardware and software
2. Scheduling software tasks to minimize makespan
3. Respecting hardware budget constraints
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy
import torch
import random
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.csched_env import CSchedEnv, EnvState, NOT_READY, READY, IN_PROGRESS, COMPLETE
from utils.task_graph_generation import TaskGraphDataset, create_data_lists
from utils.scheduler_utils import compute_dag_execution_time
from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_local_search.log")

logger = LogManager.get_logger(__name__)


@dataclass
class PartitionScheduleSolution:
    """
    Represents a complete solution with partitioning and scheduling.
    
    Attributes:
        partition: Binary array (0=software, 1=hardware) for each task
        schedule: Dictionary mapping task_id -> start_time
        makespan: Maximum finish time across all tasks (objective value)
        total_cost: Same as makespan (kept for compatibility)
    """
    partition: np.ndarray
    schedule: Dict[int, float]
    makespan: float
    total_cost: float  # Equal to makespan
    
    def copy(self):
        """Create a deep copy of the solution."""
        return PartitionScheduleSolution(
            partition=self.partition.copy(),
            schedule=self.schedule.copy(),
            makespan=self.makespan,
            total_cost=self.total_cost
        )


class HWSWPartitionScheduler:
    """
    Handler for hardware-software partitioning with software scheduling.
    """
    
    def __init__(self, 
                 graph: nx.DiGraph,
                 software_times: Dict[int, float],
                 hardware_times: Dict[int, float],
                 hardware_areas: Dict[int, float],
                 communication_costs: Dict[Tuple[int, int], float],
                 max_hardware_area: float):
        """
        Initialize the partitioner-scheduler.
        
        Args:
            graph: Directed graph representing task dependencies
            software_times: Execution time for each task on software
            hardware_times: Execution time for each task on hardware
            hardware_areas: Area cost for each task on hardware
            communication_costs: Communication cost between tasks
            max_hardware_area: Maximum allowed hardware budget
            logger: Logger instance
        """
        self.graph = graph
        self.software_times = software_times
        self.hardware_times = hardware_times
        self.hardware_areas = hardware_areas
        self.communication_costs = communication_costs
        self.max_hardware_area = max_hardware_area
        
        self.n_tasks = len(graph.nodes())
        
        # Compute topological order for scheduling
        if nx.is_directed_acyclic_graph(graph):
            self.topo_order = list(nx.topological_sort(graph))
        else:
            logger.warning("Graph has cycles! Using arbitrary ordering.")
            self.topo_order = list(graph.nodes())
    
    def compute_schedule(self, partition: np.ndarray) -> Tuple[Dict[int, float], float]:
        """
        Compute schedule for all tasks given a partition.
        Uses list scheduling with earliest start time for each task.
        
        Args:
            partition: Binary array indicating task assignments
            
        Returns:
            Tuple of (schedule_dict, makespan)
            - schedule_dict contains start times for ALL tasks
            - makespan is the maximum finish time across ALL tasks
        """
        schedule = {}
        finish_times = {}
        
        # Process tasks in topological order
        for task in self.topo_order:
            # Compute earliest start time based on predecessors
            earliest_start = 0.0
            
            for pred in self.graph.predecessors(task):
                # Predecessor must finish before this task starts
                pred_finish = finish_times.get(pred, 0.0)
                
                # Add communication cost if predecessor is on different resource
                if partition[pred] != partition[task]:
                    comm_cost = self.communication_costs.get((pred, task), 0.0)
                    pred_finish += comm_cost
                
                earliest_start = max(earliest_start, pred_finish)
            
            start_time = earliest_start
            schedule[task] = start_time
            
            # Calculate finish time based on task assignment
            if partition[task] == 1:  # Hardware task
                finish_time = start_time + self.hardware_times[task]
            else:  # Software task
                finish_time = start_time + self.software_times[task]
            
            finish_times[task] = finish_time
        
        # Makespan is the maximum finish time across ALL tasks
        makespan = max(finish_times.values()) if finish_times else 0.0
        
        return schedule, makespan
    
    def evaluate_solution(self, partition: np.ndarray) -> PartitionScheduleSolution:
        """
        Evaluate a partition by computing schedule and makespan.
        
        Args:
            partition: Binary array of task assignments
            
        Returns:
            Complete solution with schedule and costs
        """
        # Compute schedule for all tasks
        schedule, makespan = self.compute_schedule(partition)
        
        # Objective function is ONLY the makespan
        total_cost = makespan
        
        return PartitionScheduleSolution(
            partition=partition.copy(),
            schedule=schedule,
            makespan=makespan,
            total_cost=total_cost
        )
    
    def is_feasible(self, partition: np.ndarray) -> bool:
        """Check if partition satisfies hardware budget constraint."""
        total_hw_area = sum(partition[i] * self.hardware_areas[i] 
                           for i in range(self.n_tasks))
        return total_hw_area <= self.max_hardware_area
    
    def repair_solution(self, partition: np.ndarray) -> np.ndarray:
        """
        Repair infeasible solution by moving tasks from hardware to software.
        Prioritizes moving tasks with:
        - High hardware cost
        - Low impact on makespan
        """
        repaired = partition.copy()
        
        while not self.is_feasible(repaired):
            # Find hardware tasks
            hw_tasks = [i for i in range(self.n_tasks) if repaired[i] == 1]
            
            if not hw_tasks:
                raise ValueError("Cannot repair: no hardware tasks remaining")
            
            # Evaluate cost increase for moving each task to software
            costs = []
            for task in hw_tasks:
                temp = repaired.copy()
                temp[task] = 0
                
                sol_before = self.evaluate_solution(repaired)
                sol_after = self.evaluate_solution(temp)
                cost_increase = sol_after.total_cost - sol_before.total_cost
                
                costs.append((task, cost_increase))
            
            # Move task with minimum cost increase
            task_to_move = min(costs, key=lambda x: x[1])[0]
            repaired[task_to_move] = 0
            
            logger.info(f"Moved task {task_to_move} from HW to SW")
        
        return repaired


def local_search_partition_schedule(
    scheduler: HWSWPartitionScheduler,
    initial_partition: np.ndarray,
    max_iterations: int = 1000,
    use_first_improvement: bool = True,
    neighborhood_type: str = 'flip'
) -> PartitionScheduleSolution:
    """
    Local search for hardware-software partitioning with scheduling.
    
    The algorithm:
    1. Start with initial partition
    2. Compute schedule for software tasks
    3. Explore neighborhood by changing partition
    4. Re-schedule after each partition change
    5. Accept improving moves until local optimum
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        initial_partition: Initial binary partition
        max_iterations: Max iterations without improvement
        use_first_improvement: First vs best improvement strategy
        neighborhood_type: 'flip' (single task) or 'swap' (exchange two tasks)
        
    Returns:
        Best solution found
    """
    # Initialize with repaired solution if needed
    current_partition = initial_partition.copy()
    if not scheduler.is_feasible(current_partition):
        logger.info("Initial partition infeasible. Repairing...")
        current_partition = scheduler.repair_solution(current_partition)
    
    current_solution = scheduler.evaluate_solution(current_partition)
    best_solution = current_solution.copy()
    
    iterations_without_improvement = 0
    total_iterations = 0
    
    logger.info(f"Starting local search. Initial cost: {current_solution.total_cost:.4f}, "
               f"Makespan: {current_solution.makespan:.4f}")
    
    while iterations_without_improvement < max_iterations:
        improved = False
        total_iterations += 1
        
        if use_first_improvement:
            # First improvement
            neighbor_solution = find_first_improving_neighbor(
                scheduler, current_solution, neighborhood_type
            )
        else:
            # Best improvement
            neighbor_solution = find_best_improving_neighbor(
                scheduler, current_solution, neighborhood_type
            )
        
        # Check if improvement found
        if neighbor_solution is not None and neighbor_solution.total_cost < current_solution.total_cost:
            current_solution = neighbor_solution
            improved = True
            
            if current_solution.total_cost < best_solution.total_cost:
                best_solution = current_solution.copy()
                logger.info(f"Iter {total_iterations}: New best cost = {best_solution.total_cost:.4f}, "
                           f"Makespan = {best_solution.makespan:.4f}")
        
        if improved:
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        
        # Periodic logging
        if total_iterations % 100 == 0:
            logger.info(f"Iter {total_iterations}: Current = {current_solution.total_cost:.4f}, "
                       f"Best = {best_solution.total_cost:.4f}")
    
    logger.info(f"Local search completed. Total iterations: {total_iterations}")
    logger.info(f"Best cost: {best_solution.total_cost:.4f}, Makespan: {best_solution.makespan:.4f}")
    
    # Log partition details
    n_hw = np.sum(best_solution.partition)
    n_sw = len(best_solution.partition) - n_hw
    logger.info(f"Final partition: {n_hw} HW tasks, {n_sw} SW tasks")
    
    return best_solution


def find_first_improving_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution,
    neighborhood_type: str
) -> Optional[PartitionScheduleSolution]:
    """
    Find first improving neighbor.
    
    Args:
        scheduler: Scheduler instance
        current_solution: Current solution
        neighborhood_type: 'flip' or 'swap'
        
    Returns:
        First improving neighbor or None
    """
    if neighborhood_type == 'flip':
        return find_first_flip_neighbor(scheduler, current_solution)
    elif neighborhood_type == 'swap':
        return find_first_swap_neighbor(scheduler, current_solution)
    else:
        raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")


def find_best_improving_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution,
    neighborhood_type: str
) -> Optional[PartitionScheduleSolution]:
    """
    Find best improving neighbor.
    
    Args:
        scheduler: Scheduler instance
        current_solution: Current solution
        neighborhood_type: 'flip' or 'swap'
        
    Returns:
        Best improving neighbor or None
    """
    if neighborhood_type == 'flip':
        return find_best_flip_neighbor(scheduler, current_solution)
    elif neighborhood_type == 'swap':
        return find_best_swap_neighbor(scheduler, current_solution)
    else:
        raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")


def find_first_flip_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution
) -> Optional[PartitionScheduleSolution]:
    """Flip neighborhood: change assignment of one task."""
    n = len(current_solution.partition)
    
    for i in range(n):
        neighbor_partition = current_solution.partition.copy()
        neighbor_partition[i] = 1 - neighbor_partition[i]  # Flip
        
        # Check feasibility
        if not scheduler.is_feasible(neighbor_partition):
            continue
        
        # Evaluate neighbor (includes re-scheduling)
        neighbor_solution = scheduler.evaluate_solution(neighbor_partition)
        
        # Return first improvement
        if neighbor_solution.total_cost < current_solution.total_cost:
            return neighbor_solution
    
    return None


def find_best_flip_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution
) -> Optional[PartitionScheduleSolution]:
    """Find best neighbor in flip neighborhood."""
    n = len(current_solution.partition)
    best_neighbor = None
    
    for i in range(n):
        neighbor_partition = current_solution.partition.copy()
        neighbor_partition[i] = 1 - neighbor_partition[i]
        
        if not scheduler.is_feasible(neighbor_partition):
            continue
        
        neighbor_solution = scheduler.evaluate_solution(neighbor_partition)
        
        if neighbor_solution.total_cost < current_solution.total_cost:
            if best_neighbor is None or neighbor_solution.total_cost < best_neighbor.total_cost:
                best_neighbor = neighbor_solution
    
    return best_neighbor


def find_first_swap_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution
) -> Optional[PartitionScheduleSolution]:
    """Swap neighborhood: exchange assignments of two tasks."""
    n = len(current_solution.partition)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only swap if tasks have different assignments
            if current_solution.partition[i] == current_solution.partition[j]:
                continue
            
            neighbor_partition = current_solution.partition.copy()
            neighbor_partition[i], neighbor_partition[j] = \
                neighbor_partition[j], neighbor_partition[i]
            
            if not scheduler.is_feasible(neighbor_partition):
                continue
            
            neighbor_solution = scheduler.evaluate_solution(neighbor_partition)
            
            if neighbor_solution.total_cost < current_solution.total_cost:
                return neighbor_solution
    
    return None


def find_best_swap_neighbor(
    scheduler: HWSWPartitionScheduler,
    current_solution: PartitionScheduleSolution
) -> Optional[PartitionScheduleSolution]:
    """Find best neighbor in swap neighborhood."""
    n = len(current_solution.partition)
    best_neighbor = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if current_solution.partition[i] == current_solution.partition[j]:
                continue
            
            neighbor_partition = current_solution.partition.copy()
            neighbor_partition[i], neighbor_partition[j] = \
                neighbor_partition[j], neighbor_partition[i]
            
            if not scheduler.is_feasible(neighbor_partition):
                continue
            
            neighbor_solution = scheduler.evaluate_solution(neighbor_partition)
            
            if neighbor_solution.total_cost < current_solution.total_cost:
                if best_neighbor is None or neighbor_solution.total_cost < best_neighbor.total_cost:
                    best_neighbor = neighbor_solution
    
    return best_neighbor

def update_env_presolve(env, opt_sols_batch):

    partition_batch   = opt_sols_batch["partitions"]
    start_times_batch = opt_sols_batch["start_times"]
    end_times_batch   = opt_sols_batch["end_times"]
    makespan_batch    = opt_sols_batch["makespans"]

    batch_size = len(partition_batch)
    n = partition_batch.shape[1]   # TODO: this maybe should vary per the bsatch sample

    env.op_status_batch   = COMPLETE*torch.ones((batch_size, n),dtype=torch.int32) # COMPLETE
    env.op_resource_batch = partition_batch.type(torch.int32)
    
    env.current_time_batch = makespan_batch
    env.makespan_batch     = makespan_batch
    
    env.op_start_time_batch = start_times_batch
    env.op_end_time_batch   = end_times_batch



# Returns an opt_sols_batch of size 1
def solve_dataset_instance(dataset: TaskGraphDataset, idx:int=0):
    k = idx

    print("Presolving testset instance {}".format(k+1))
    graph, adj_matrices, node_features, edge_features, hw_area_limit = dataset[k]

    node_list = list(graph.nodes())

    software_costs = node_features[0,:]
    hardware_areas = node_features[1,:]                  #  node_features = torch.cat([sw_computation_cost, hw_area_cost], dim=0)
    hardware_costs = software_costs*0.5
    communication_costs = edge_features[0,:]             # only one edge feature
    
    hardware_areas_dict = {i: hwa for i, hwa in enumerate(hardware_areas.tolist())}
    hardware_costs_dict = {i: hwc for i, hwc in enumerate(hardware_costs.tolist())}
    software_costs_dict = {i: swc for i, swc in enumerate(software_costs.tolist())}
    communication_costs_dict = {e: cc for (e, cc) in zip(list(graph.edges),communication_costs.tolist())}

    # Set node attributes
    for n in graph.nodes():
        graph.nodes[n]['software_time'] = software_costs_dict[n]
        graph.nodes[n]['hardware_time'] = hardware_costs_dict[n]
        graph.nodes[n]['hardware_area'] = hardware_areas_dict[n]
    # Set edge attributes
    for u,v in graph.edges():
        graph.edges[u,v]['communication_cost'] = communication_costs_dict[(u,v)]

    # Create scheduler
    scheduler = HWSWPartitionScheduler(
        graph=graph,
        software_times=software_costs_dict,
        hardware_times=hardware_costs_dict,
        hardware_areas=hardware_areas_dict,
        communication_costs=communication_costs_dict,
        max_hardware_area=hw_area_limit
    )
    
    # Generate initial random partition
    n = len(graph.nodes())
    initial_partition = np.random.randint(0, 2, size=n)
    
    logger.info("="*60)
    logger.info("Testing HW-SW Partitioning with Scheduling")
    logger.info("="*60)
    logger.info(f"Number of tasks: {n}")
    logger.info(f"Number of dependencies: {len(graph.edges)}")
    logger.info(f"Max hardware area: {hw_area_limit}")
    logger.info(f"Initial partition: {initial_partition}")
    
    # Evaluate initial solution
    initial_solution = scheduler.evaluate_solution(initial_partition)
    logger.info(f"Initial total cost: {initial_solution.total_cost:.4f}")
    logger.info(f"Initial makespan: {initial_solution.makespan:.4f}")
    
    # Run local search with flip neighborhood
    logger.info("\n" + "="*60)
    logger.info("Running Local Search (Flip Neighborhood)")
    logger.info("="*60)
    
    best_solution = local_search_partition_schedule(
        scheduler=scheduler,
        initial_partition=initial_partition,
        max_iterations=100,
        use_first_improvement=True,
        neighborhood_type='flip'
    )
    
    logger.info("\n" + "="*60)
    logger.info("Final Results")
    logger.info("="*60)
    logger.info(f"Best partition: {best_solution.partition}")
    logger.info(f"Best total cost: {best_solution.total_cost:.4f}")
    logger.info(f"Best makespan: {best_solution.makespan:.4f}")
    logger.info(f"Schedule: {best_solution.schedule}")
    
    # Calculate improvement
    improvement = ((initial_solution.total_cost - best_solution.total_cost) / 
                   initial_solution.total_cost * 100)
    logger.info(f"Improvement: {improvement:.2f}%")
    
    # Verify feasibility
    hw_cost = sum(best_solution.partition[i] * hardware_areas[i] for i in range(n))
    logger.info(f"Hardware area used: {hw_cost:.4f} / {hw_area_limit:.4f}")
    logger.info(f"Feasible: {scheduler.is_feasible(best_solution.partition)}")

    # Partition dictionary
    partition = {}
    for i,n in enumerate(node_list):
        partition[n] = int(best_solution.partition[i])

    # Compute execution time
    result = compute_dag_execution_time(graph, partition, verbose=False, full_dict = True)
    start_times  = result['start_times']
    end_times = result['finish_times']
    hw_nodes = result['hardware_nodes']
    sw_nodes = result['software_nodes']
    makespan = result['makespan']


    start_times_list = [start_times[node] for node in node_list]
    end_times_list   = [end_times[node] for node in node_list]
    partition_list   = [partition[node] for node in node_list]

    partition_t   = torch.Tensor(partition_list)
    start_times_t = torch.Tensor(start_times_list)
    end_times_t   = torch.Tensor(end_times_list)
    makespan_t   = torch.Tensor([makespan])

    proc_times_t = partition_t*hardware_costs + (1-partition_t)*software_costs


    partition_t_list = [partition_t]
    start_times_t_list = [start_times_t]
    end_times_t_list = [end_times_t]
    makespan_t_list = [makespan_t]

    partition_batch = torch.stack(partition_t_list)
    start_times_batch = torch.stack(start_times_t_list)
    end_times_batch = torch.stack(end_times_t_list)
    makespan_batch = torch.stack(makespan_t_list)


    opt_sols_batch = {"partitions":  partition_batch,
                      "start_times": start_times_batch,
                      "end_times":   end_times_batch,
                      "makespans":   makespan_batch
    }

    return opt_sols_batch


# Example usage and testing
if __name__ == "__main__":
    
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate a dataset of instances
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=1,
        min_nodes=10,
        max_nodes=10,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits)

    # Setup gym environment
    env_paras = {
        "batch_size": 1,
        "device": "cpu",
        "timestep_mode": "next_complete",
        "timestep_trigger": "every",
        "prevent_all_HW": False
    }
    env = CSchedEnv(dataset, env_paras)

    # Get MIP solutions for each instance
    opt_sols_batch = solve_dataset_instance(dataset, idx=0)

    # Update solutions in the environment's batch of instances
    update_env_presolve(env, opt_sols_batch)

    # Argument to render is the instance's index
    fig = env.render(0)
    fig.savefig("outputs/local_search_example_gantt_chart.png", dpi=300, bbox_inches='tight')