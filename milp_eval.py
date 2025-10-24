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
from utils.cuopt_utils import CuOptScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time
from utils.parser_utils import parse_arguments

def main():
    config = parse_arguments()

    LogManager.initialize(f"logs/run_milp_optimizer_area-{config['area-constraint']:.2f}_hw-{config['hw-scale-factor']:.1f}_seed-{config['seed']}.log")
    logger = LogManager.get_logger(__name__)

    # Create solver instance
    if config['solver-tool'] == 'cvxpy':
        solver = ScheduleConstPartitionSolver()
    elif config['solver-tool'] == 'cuopt':
        solver = CuOptScheduleConstPartitionSolver()
    else:
        logger.error(f"Unsupported solver tool: {config['solver-tool']}")
        raise NotImplementedError(f"Unsupported solver tool: {config['solver-tool']}")
    
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
    
    # Solve optimization with area constraint
    A_max = np.sum(solver.a) * config['area-constraint']
    solution = solver.solve_optimization(A_max=A_max)
    
    

    partition_assignment = {}
    for n in solution['hardware_nodes']:
        partition_assignment[n] = 1
    for n in solution['software_nodes']:
        partition_assignment[n] = 0
    
    # Compute execution time
    result = compute_dag_execution_time(graph, partition_assignment, verbose=False)
    logger.info(f"Execution time: {result['makespan']}")
    
    import pickle
    area_constraint_str = f"{config['area-constraint']:.2f}"
    hwscale_str = f"{config['hw-scale-factor']:.1f}"
    hwvar_str = f"{config['hw-scale-variance']:.2f}"
    comm_str = f"{config['comm-scale-factor']:.2f}"
    seed_str = f"{config['seed']}"
    with open(f"makespan-opt-partitions/taskgraph-squeeze_net_tosa_area-{area_constraint_str}_hwscale-{hwscale_str}_hwvar-{hwvar_str}_comm-{comm_str}_seed-{seed_str}_assignment-mip.pkl",'wb') as f:
        pickle.dump(partition_assignment,f)


if __name__ == "__main__":
    main()