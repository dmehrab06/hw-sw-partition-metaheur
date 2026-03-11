"""
Main Iterated Local Search (ILS) implementation for HW/SW Partitioning with Scheduling.

This script implements the core ILS algorithm that iteratively:
1. Applies local search to find a local optimum
2. Perturbs the solution to escape the local optimum
3. Applies local search again
4. Decides whether to accept the new solution
"""

import numpy as np
import torch
import random
import os
import sys
from typing import Tuple, Dict, Optional
import time

# Add parent directory to path
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.task_graph_generation import create_data_lists, TaskGraphDataset
from utils.local_search import (
    HWSWPartitionScheduler,
    PartitionScheduleSolution,
    local_search_partition_schedule
)
from utils.perturb_utils import PerturbationManager
from utils.acceptance_criteria import accept_better
from utils.csched_env import CSchedEnv, COMPLETE

# Set up logging
LogManager.initialize("logs/main_ils.log")
logger = LogManager.get_logger(__name__)


def iterated_local_search(
    scheduler: HWSWPartitionScheduler,
    initial_partition: np.ndarray,
    max_iterations: int = 1000,
    perturbation_type: str = 'random_flips',
    perturbation_strength: int = 3,
    local_search_max_iter: int = 100,
    local_search_neighborhood: str = 'flip',
    local_search_improvement: str = 'first',
    verbose: bool = True
) -> Tuple[PartitionScheduleSolution, Dict]:
    """
    Main Iterated Local Search algorithm for HW/SW partitioning with scheduling.
    
    Algorithm:
        s0 = GenerateInitialSolution()
        s* = LocalSearch(s0)
        best = s*
        
        repeat until stopping criterion:
            s' = Perturbation(s*, strength)
            s*' = LocalSearch(s')
            s* = AcceptanceCriterion(s*, s*')
            
            if cost(s*) < cost(best):
                best = s*
        
        return best
    
    Args:
        scheduler: HWSWPartitionScheduler instance
        initial_partition: Initial partition to start from
        max_iterations: Maximum number of ILS iterations
        perturbation_type: Type of perturbation to apply
        perturbation_strength: Strength of perturbation
        local_search_max_iter: Max iterations for local search
        local_search_neighborhood: Neighborhood type for local search
        local_search_improvement: Improvement strategy for local search
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_solution, history_dict)
    """
    
    logger.info("="*70)
    logger.info("STARTING ITERATED LOCAL SEARCH")
    logger.info("="*70)
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Perturbation: {perturbation_type} with strength {perturbation_strength}")
    logger.info(f"Local search: {local_search_neighborhood} neighborhood, "
                f"{local_search_improvement} improvement")
    
    # Initialize perturbation manager
    perturb_manager = PerturbationManager(repair_infeasible=True)
    
    # Step 1: Apply local search to initial solution
    logger.info("\nApplying initial local search...")
    current_solution = local_search_partition_schedule(
        scheduler=scheduler,
        initial_partition=initial_partition,
        max_iterations=local_search_max_iter,
        use_first_improvement=(local_search_improvement == 'first'),
        neighborhood_type=local_search_neighborhood
    )
    
    # Initialize best solution
    best_solution = current_solution.copy()
    
    logger.info(f"Initial solution cost: {current_solution.total_cost:.4f}")
    logger.info(f"Initial makespan: {current_solution.makespan:.4f}")
    
    # Track history
    history = {
        'iterations': [],
        'current_costs': [],
        'best_costs': [],
        'perturbations_applied': [],
        'acceptance_decisions': [],
        'iteration_times': []
    }
    
    iterations_without_improvement = 0
    start_time = time.time()
    
    # Main ILS loop
    logger.info("\n" + "="*70)
    logger.info("STARTING ILS ITERATIONS")
    logger.info("="*70)
    
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        
        # Step 2: Perturb current solution
        perturbed_partition = perturb_manager.perturb(
            partition=current_solution.partition,
            scheduler=scheduler,
            strategy_name=perturbation_type,
            strength=perturbation_strength,
            solution=current_solution
        )
        
        # Step 3: Apply local search to perturbed solution
        new_solution = local_search_partition_schedule(
            scheduler=scheduler,
            initial_partition=perturbed_partition,
            max_iterations=local_search_max_iter,
            use_first_improvement=(local_search_improvement == 'first'),
            neighborhood_type=local_search_neighborhood
        )
        
        # Step 4: Acceptance criterion
        accepted = accept_better(current_solution, new_solution)
        
        if accepted:
            current_solution = new_solution
            iterations_without_improvement = 0
            
            # Update best if improved
            if current_solution.total_cost < best_solution.total_cost:
                improvement = best_solution.total_cost - current_solution.total_cost
                best_solution = current_solution.copy()
                
                if verbose:
                    logger.info(f"\nIteration {iteration+1}: NEW BEST SOLUTION!")
                    logger.info(f"  Cost: {best_solution.total_cost:.4f}")
                    logger.info(f"  Improvement: {improvement:.4f}")
                    logger.info(f"  Makespan: {best_solution.makespan:.4f}")
        else:
            iterations_without_improvement += 1
        
        # Record history
        iter_time = time.time() - iter_start_time
        history['iterations'].append(iteration)
        history['current_costs'].append(current_solution.total_cost)
        history['best_costs'].append(best_solution.total_cost)
        history['perturbations_applied'].append(perturbation_type)
        history['acceptance_decisions'].append(accepted)
        history['iteration_times'].append(iter_time)
        
        # Periodic logging
        if verbose and (iteration + 1) % 50 == 0:
            logger.info(f"\nIteration {iteration+1}/{max_iterations}")
            logger.info(f"  Current cost: {current_solution.total_cost:.4f}")
            logger.info(f"  Best cost: {best_solution.total_cost:.4f}")
            logger.info(f"  Iterations without improvement: {iterations_without_improvement}")
        
        # Early stopping if stuck
        if iterations_without_improvement >= max_iterations // 2:
            logger.info(f"\nStopping early at iteration {iteration+1}: "
                       f"{iterations_without_improvement} iterations without improvement")
            break
    
    # Final statistics
    total_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("ILS COMPLETED")
    logger.info("="*70)
    logger.info(f"Total iterations: {len(history['iterations'])}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Best cost found: {best_solution.total_cost:.4f}")
    logger.info(f"Best makespan: {best_solution.makespan:.4f}")
    
    # Calculate acceptance rate
    acceptance_rate = sum(history['acceptance_decisions']) / len(history['acceptance_decisions'])
    logger.info(f"Acceptance rate: {acceptance_rate:.2%}")
    
    # Partition statistics
    n_hw = np.sum(best_solution.partition)
    n_sw = len(best_solution.partition) - n_hw
    logger.info(f"Final partition: {n_hw} HW tasks, {n_sw} SW tasks")
    
    hw_usage = sum(scheduler.hardware_areas[i] for i in range(len(best_solution.partition))
                   if best_solution.partition[i] == 1)
    logger.info(f"HW area usage: {hw_usage:.2f} / {scheduler.max_hardware_area:.2f} "
                f"({hw_usage/scheduler.max_hardware_area*100:.1f}%)")
    
    return best_solution, history

def solve_dataset_instance_ils(dataset: TaskGraphDataset, idx: int = 0,
                              max_iterations: int = 500,
                              perturbation_type: str = 'random_flips',
                              perturbation_strength: int = 3,
                              local_search_max_iter: int = 100,
                              initial_solution_type: str = 'random',
                              acceptance_criterion: str = 'better',
                              verbose: bool = True):
    """
    Solve a single instance from the dataset using ILS.
    
    Args:
        dataset: TaskGraphDataset instance
        idx: Index of instance to solve
        max_iterations: Maximum ILS iterations
        perturbation_type: Perturbation strategy to use
        perturbation_strength: Strength of perturbation
        local_search_max_iter: Max iterations for local search
        initial_solution_type: Type of initial solution generator
        acceptance_criterion: Acceptance criterion to use
        verbose: Whether to print progress
        
    Returns:
        opt_sols_batch: Dictionary with solution tensors (compatible with env)
    """
    from utils.scheduler_utils import compute_dag_execution_time
    from utils.initial_solution import InitialSolutionFactory
    
    k = idx
    logger.info(f"Solving instance {k+1} with ILS")
    
    # Load instance
    graph, adj_matrix, node_features, edge_features, hw_area_limit = dataset[k]
    node_list = list(graph.nodes())
    n = len(node_list)
    
    # Extract costs
    software_costs = node_features[0,:]
    hardware_areas = node_features[1,:]                  #  node_features = torch.cat([sw_computation_cost, hw_area_cost], dim=0)
    hardware_costs = software_costs*0.5
    communication_costs = edge_features[0,:]             # only one edge feature
    
    # Create dictionaries
    hardware_areas_dict = {i: hwa for i, hwa in enumerate(hardware_areas.tolist())}
    hardware_costs_dict = {i: hwc for i, hwc in enumerate(hardware_costs.tolist())}
    software_costs_dict = {i: swc for i, swc in enumerate(software_costs.tolist())}
    communication_costs_dict = {e: cc for (e, cc) in zip(list(graph.edges()), communication_costs.tolist())}
    
    # Set graph attributes
    for n_id in graph.nodes():
        graph.nodes[n_id]['software_time'] = software_costs_dict[n_id]
        graph.nodes[n_id]['hardware_time'] = hardware_costs_dict[n_id]
        graph.nodes[n_id]['hardware_area'] = hardware_areas_dict[n_id]
    
    for u, v in graph.edges():
        graph.edges[u, v]['communication_cost'] = communication_costs_dict[(u, v)]
    
    # Create scheduler
    scheduler = HWSWPartitionScheduler(
        graph=graph,
        software_times=software_costs_dict,
        hardware_times=hardware_costs_dict,
        hardware_areas=hardware_areas_dict,
        communication_costs=communication_costs_dict,
        max_hardware_area=hw_area_limit.item()
    )
    
    logger.info("="*70)
    logger.info("ILS FOR HW/SW PARTITIONING WITH SCHEDULING")
    logger.info("="*70)
    logger.info(f"Instance {k+1}")
    logger.info(f"Number of tasks: {n}")
    logger.info(f"Number of dependencies: {len(graph.edges())}")
    logger.info(f"Max hardware area: {hw_area_limit.item():.4f}")
    
    # Generate initial solution
    logger.info(f"\nGenerating initial solution (type: {initial_solution_type})...")
    initial_gen = InitialSolutionFactory.create(initial_solution_type, repair_infeasible=True)
    initial_partition = initial_gen.generate(scheduler)
    
    initial_solution = scheduler.evaluate_solution(initial_partition)
    logger.info(f"Initial solution cost: {initial_solution.total_cost:.4f}")
    logger.info(f"Initial makespan: {initial_solution.makespan:.4f}")
    
    # Run ILS
    logger.info("\n" + "="*70)
    logger.info("RUNNING ITERATED LOCAL SEARCH")
    logger.info("="*70)
    
    best_solution, history = iterated_local_search(
        scheduler=scheduler,
        initial_partition=initial_partition,
        max_iterations=max_iterations,
        perturbation_type=perturbation_type,
        perturbation_strength=perturbation_strength,
        local_search_max_iter=local_search_max_iter,
        local_search_neighborhood='flip',
        local_search_improvement='first',
        verbose=verbose
    )
    
    # Calculate improvement
    improvement = ((initial_solution.total_cost - best_solution.total_cost) / 
                   initial_solution.total_cost * 100)
    
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"Best partition: {best_solution.partition}")
    logger.info(f"Initial cost: {initial_solution.total_cost:.4f}")
    logger.info(f"Best cost: {best_solution.total_cost:.4f}")
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info(f"Best makespan: {best_solution.makespan:.4f}")
    
    # Verify feasibility
    hw_cost = sum(best_solution.partition[i] * hardware_areas[i].item() for i in range(n))
    logger.info(f"Hardware area used: {hw_cost:.4f} / {hw_area_limit.item():.4f}")
    logger.info(f"Feasible: {scheduler.is_feasible(best_solution.partition)}")
    
    # Create partition dictionary
    partition = {}
    for i, node in enumerate(node_list):
        partition[node] = int(best_solution.partition[i])
    
    # Compute execution time using scheduler utils
    result = compute_dag_execution_time(graph, partition, verbose=False, full_dict=True)
    start_times = result['start_times']
    end_times = result['finish_times']
    hw_nodes = result['hardware_nodes']
    sw_nodes = result['software_nodes']
    makespan = result['makespan']
    
    logger.info(f"Verified makespan: {makespan:.4f}")
    logger.info(f"HW nodes: {len(hw_nodes)}, SW nodes: {len(sw_nodes)}")
    
    # Convert to tensors (matching the format from solve_dataset_instance)
    start_times_list = [start_times[node] for node in node_list]
    end_times_list = [end_times[node] for node in node_list]
    partition_list = [partition[node] for node in node_list]
    
    partition_t = torch.Tensor(partition_list)
    start_times_t = torch.Tensor(start_times_list)
    end_times_t = torch.Tensor(end_times_list)
    makespan_t = torch.Tensor([makespan])
    
    # Create batch (single instance)
    partition_batch = partition_t.unsqueeze(0)
    start_times_batch = start_times_t.unsqueeze(0)
    end_times_batch = end_times_t.unsqueeze(0)
    makespan_batch = makespan_t.unsqueeze(0)
    
    # Return in expected format
    opt_sols_batch = {
        "partitions": partition_batch,
        "start_times": start_times_batch,
        "end_times": end_times_batch,
        "makespans": makespan_batch
    }
    
    return opt_sols_batch

def solve_dataset_instance_milp(dataset: TaskGraphDataset, idx: int = 0):
    """
    Solve a single instance from the dataset using MILP solver.
    
    Args:
        dataset: TaskGraphDataset instance
        idx: Index of instance to solve

    Returns:
        opt_sols_batch: Dictionary with solution tensors (compatible with env)
    """
    from utils.partition_utils import ScheduleConstPartitionSolver
    from utils.scheduler_utils import compute_dag_execution_time

    graph, adj_matrices, node_features, edge_features, hw_area_limit = dataset[idx]

    software_costs = node_features[0,:]
    hardware_areas = node_features[1,:]                  #  node_features = torch.cat([sw_computation_cost, hw_area_cost], dim=0)
    hardware_costs = software_costs*0.5
    communication_costs = edge_features[0,:]             # only one edge feature

    node_list = list(graph.nodes())

    solver = ScheduleConstPartitionSolver()
    solver.load_networkx_graph_with_torch_feats(graph, hardware_areas, hardware_costs, software_costs, communication_costs)
    solution = solver.solve_optimization(A_max=hw_area_limit)
    partition = {}
    for n in solution['hardware_nodes']:
        partition[n] = 1
    for n in solution['software_nodes']:
        partition[n] = 0

    # Compute execution time
    result = compute_dag_execution_time(graph, partition, verbose=False, full_dict = True)
    start_times  = result['start_times']
    end_times = result['finish_times']
    hw_nodes = result['hardware_nodes']
    sw_nodes = result['software_nodes']
    makespan = result['makespan']

    logger.info(f"Verified makespan: {makespan:.4f}")
    logger.info(f"HW nodes: {len(hw_nodes)}, SW nodes: {len(sw_nodes)}")

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



def main():
    """
    Main function to demonstrate ILS on a test instance.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate test instance
    logger.info("Generating test instance...")
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=1,
        min_nodes=10,
        max_nodes=10,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, 
                               edge_features_list, hw_area_limits)
    
    # logger.info("="*70)
    # logger.info("ILS FOR HW/SW PARTITIONING WITH SCHEDULING")
    # logger.info("="*70)

    # # Use the new solve_dataset_instance_ils method
    # opt_sols_batch = solve_dataset_instance_ils(
    #     dataset=dataset,
    #     idx=0,
    #     max_iterations=500,
    #     perturbation_type='random_flips',
    #     perturbation_strength=3,
    #     local_search_max_iter=100,
    #     initial_solution_type='random',
    #     acceptance_criterion='better',
    #     verbose=True
    # )

    # logger.info("ILS process completed successfully.")

    # logger.info("="*70)
    # logger.info("MILP FOR HW/SW PARTITIONING WITH SCHEDULING")
    # logger.info("="*70)

    # Now solve using MILP for comparison
    opt_sols_batch_milp = solve_dataset_instance_milp(dataset=dataset, idx=0)
    logger.info("MILP process completed successfully.")

    return opt_sols_batch_milp, dataset

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

if __name__ == "__main__":
    
    opt_sols_batch, dataset = main()
    # Setup gym environment
    env_paras = {
        "batch_size": 1,
        "device": "cpu",
        "timestep_mode": "next_complete",
        "timestep_trigger": "every",
        "prevent_all_HW": False
    }
    env = CSchedEnv(dataset, env_paras)

    # Update solutions in the environment's batch of instances
    update_env_presolve(env, opt_sols_batch)

    # Argument to render is the instance's index
    fig = env.render(0)
    fig.savefig("outputs/local_search_example_gantt_chart.png", dpi=300, bbox_inches='tight')