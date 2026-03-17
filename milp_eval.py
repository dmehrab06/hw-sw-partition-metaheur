"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
import random
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.cuopt_utils import CuOptScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time
from utils.parser_utils import parse_arguments

def main():
    config = parse_arguments()

    log_file = f"logs/mip_runs/run_milp_optimizer_area-{config['area-constraint']:.2f}_hw-{config['hw-scale-factor']:.1f}_seed-{config['seed']}.log"
    LogManager.initialize(log_file)
    logger = LogManager.get_logger(__name__)

    # Create solver instance
    if config['solver-tool'] == 'cvxpy-xpress':
        solver = ScheduleConstPartitionSolver(solver="xpress")
    elif config['solver-tool'] == 'cvxpy-scip':
        solver = ScheduleConstPartitionSolver(solver="scip")
    elif config['solver-tool'] == 'cuopt':
        solver = CuOptScheduleConstPartitionSolver()
    else:
        logger.error(f"Unsupported solver tool: {config['solver-tool']}. Solving with SCIP")
        solver = ScheduleConstPartitionSolver(solver="scip")
    
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
    time_limit = 3600

    wall_start = time.perf_counter()
    solution = solver.solve_optimization(A_max=A_max, time_limit_sec=time_limit)
    wall_time = time.perf_counter() - wall_start
    

    partition_assignment = {}
    for n in solution['hardware_nodes']:
        partition_assignment[n] = 1
    for n in solution['software_nodes']:
        partition_assignment[n] = 0
    
    solution["partition_assignment"] = partition_assignment,
    solution["wall_time"] = wall_time
    
    from pathlib import Path
    import json
    area_constraint_str = f"{config['area-constraint']:.2f}"
    hwscale_str = f"{config['hw-scale-factor']:.1f}"
    hwvar_str = f"{config['hw-scale-variance']:.2f}"
    seed_str = f"{config['seed']}"
    output_dir = f"{config['solution-dir']}"
    
    dir = Path(output_dir)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dir, 0o777)

    logger.info(f"Saving partitions as pickle file in {output_dir}")
    
    with open(f"{output_dir}/taskgraph-squeeze_net_tosa_area-{area_constraint_str}_hwscale-{hwscale_str}_hwvar-{hwvar_str}_seed-{seed_str}_assignment-mip.json",'w') as f:
        json.dump(solution,f,indent=2)


if __name__ == "__main__":
    main()