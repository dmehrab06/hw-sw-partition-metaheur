"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
import json
import shutil
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.cuopt_utils import CuOptScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time, compute_dag_makespan
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
            taskgraph_pickle_used = os.path.abspath(config['taskgraph-pickle'])
            logger.info(f"Loading graph from {taskgraph_pickle_used}")
            graph = solver.load_pickle_graph(taskgraph_pickle_used)
        else:
            taskgraph_pickle_used = None
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

    # Print final assignment to stdout so run_all_mip_configs.sh logs capture it
    hw_nodes_sorted = sorted(solution['hardware_nodes'])
    sw_nodes_sorted = sorted(solution['software_nodes'])
    print(f"[mip] hardware nodes ({len(hw_nodes_sorted)}): {', '.join(hw_nodes_sorted)}")
    print(f"[mip] software nodes ({len(sw_nodes_sorted)}): {', '.join(sw_nodes_sorted)}")
    
    # Compute execution time
    lp_assignment = [1 - partition_assignment[n] for n in graph.nodes()]
    makespan,_ = compute_dag_makespan(graph, lp_assignment)
    logger.info(f"LP makespan: {makespan}")
    
    from pathlib import Path
    import pickle
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
    partition_base = f"taskgraph-squeeze_net_tosa_area-{area_constraint_str}_hwscale-{hwscale_str}_hwvar-{hwvar_str}_seed-{seed_str}"
    partition_path = Path(output_dir) / f"{partition_base}_assignment-mip.pkl"
    with open(partition_path, 'wb') as f:
        pickle.dump(partition_assignment,f)

    # Persist the exact TaskGraph pickle used for this solve (prevents later overwrite mismatches)
    taskgraph_copy_path = None
    if taskgraph_pickle_used:
        taskgraph_copy_path = Path(output_dir) / f"{partition_base}_taskgraph.pkl"
        shutil.copy2(taskgraph_pickle_used, taskgraph_copy_path)
        logger.info(f"Copied TaskGraph pickle to {taskgraph_copy_path}")

    # Write solve metadata alongside the partition
    meta = {
        "taskgraph_pickle": taskgraph_pickle_used,
        "taskgraph_pickle_copy": str(taskgraph_copy_path) if taskgraph_copy_path else None,
        "graph_file": config.get("graph-file"),
        "area_constraint": config.get("area-constraint"),
        "hw_scale_factor": config.get("hw-scale-factor"),
        "hw_scale_variance": config.get("hw-scale-variance"),
        "comm_scale_factor": config.get("comm-scale-factor"),
        "seed": config.get("seed"),
        "solver_tool": config.get("solver-tool"),
    }
    meta_path = Path(output_dir) / f"{partition_base}_assignment-mip.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Wrote solve metadata to {meta_path}")


if __name__ == "__main__":
    main()
