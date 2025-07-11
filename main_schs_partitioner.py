"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time
LogManager.initialize("logs/test_sdp_optimizer.log")
logger = LogManager.get_logger(__name__)

def main():
    # Create solver instance
    solver = ScheduleConstPartitionSolver()
    
    # Create a random DAG
    logger.info("Creating random DAG...")
    np.random.seed(123)
    graph = solver.create_random_dag(n_nodes=8, edge_probability=0.4)
    
    # Save the graph
    solver.save_graph("data/example_task_graph.pkl")

    # Load graph and create matrices
    solver.load_graph("data/example_task_graph.pkl")
    
    # Display the original graph
    solver.display_graph("Original Task Graph")
    
    # Solve optimization with area constraint
    A_max = np.sum(solver.a) * 0.6  # Allow 60% of total area for hardware
    solution = solver.solve_optimization(A_max=A_max)
    
    # Display solution
    if solution:
        solver.display_solution(solution)
    
    partition_assignment = {}
    for n in solution['hardware_nodes']:
        partition_assignment[n] = 'hardware'
    for n in solution['software_nodes']:
        partition_assignment[n] = 'software'
    
    # Compute execution time
    result = compute_dag_execution_time(graph, partition_assignment, verbose=True)

    print(f"Execution time: {result['makespan']}")


if __name__ == "__main__":
    main()