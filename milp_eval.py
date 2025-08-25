"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time
from utils.parser_utils import parse_arguments
LogManager.initialize("logs/test_milp_optimizer.log")
logger = LogManager.get_logger(__name__)

def main():
    config = parse_arguments()

    # Create solver instance
    solver = ScheduleConstPartitionSolver()
    
    # Set random seeds for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")

    try:
        # Initialize Task Graph
        logger.info(f"Loading graph from {config['graph-file']}")
        
        # graph = solver.load_pydot_graph(
        #     pydot_file=config['graph-file'], 
        #     k=config['hw-scale-factor'],
        #     l=config['hw-scale-variance'],
        #     mu=config['comm-scale-factor'],
        #     A_max=100
        #     )
        graph = solver.load_pickle_graph(config['graph-file'])
    except Exception as e:
        logger.error(f"An error occurred during loading graph from input file: {str(e)}", exc_info=True)
        raise
    
    # Display the original graph
    # solver.display_graph("Original Task Graph")
    
    # Solve optimization with area constraint
    A_max = np.sum(solver.a) * config['area-constraint']
    solution = solver.solve_optimization(A_max=A_max)
    
    # # Display solution
    # if solution:
    #     solver.display_solution(solution)
    
    partition_assignment = {}
    for n in solution['hardware_nodes']:
        partition_assignment[n] = 'hardware'
    for n in solution['software_nodes']:
        partition_assignment[n] = 'software'
    
    # Compute execution time
    result = compute_dag_execution_time(graph, partition_assignment, verbose=False)

    print(f"Execution time: {result['makespan']}")


if __name__ == "__main__":
    main()