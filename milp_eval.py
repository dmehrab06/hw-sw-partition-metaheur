"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
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
        
        if os.path.exists(config.get('taskgraph-pickle', "")):
            logger.info(f"Loading graph from {config['taskgraph-pickle']}")
            graph = solver.load_pickle_graph(config['taskgraph-pickle'])
        else:
            logger.info(f"Loading graph from {config['graph-file']}")
            graph = solver.load_pydot_graph(
                pydot_file=config['graph-file'], 
                k=config['hw-scale-factor'],
                l=config['hw-scale-variance'],
                mu=config['comm-scale-factor'],
                A_max=100
                )
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

    partition_assignment_dump = {}
    for n in solution['hardware_nodes']:
        partition_assignment_dump[n] = 1
    for n in solution['software_nodes']:
        partition_assignment_dump[n] = 0
    import pickle
    with open("makespan-opt-partitions/taskgraph-squeeze_net_tosa_area-0.50_hwscale-0.1_hwvar-0.50_comm-1.00_seed-42_assignment-mip.pkl",'wb') as f:
        pickle.dump(partition_assignment_dump,f)


if __name__ == "__main__":
    main()