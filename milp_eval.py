"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
import json
import shutil
import random
import pickle
from pathlib import Path

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.cuopt_utils import CuOptScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time, compute_dag_makespan
from utils.parser_utils import parse_arguments

def _to_plain_dict(cfg):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        try:
            return dict(cfg)
        except Exception:
            return cfg


def _resolve_taskgraph_for_visualization(config, taskgraph_pickle_used, taskgraph_copy_path):
    for candidate in (
        str(taskgraph_copy_path) if taskgraph_copy_path else None,
        taskgraph_pickle_used,
        config.get("taskgraph-pickle", None),
    ):
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, "rb") as f:
                    return pickle.load(f), candidate
            except Exception:
                continue
    return None, None


def main():
    t0 = time.perf_counter()
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
    t_solve0 = time.perf_counter()
    if config['solver-tool'] == 'cvxpy':
        solution = solver.solve_optimization(A_max=A_max, solver_cfg=config.get('mip', None))
    else:
        solution = solver.solve_optimization(A_max=A_max)
    solve_sec = time.perf_counter() - t_solve0

    if solution is None:
        logger.error("Solver did not return a valid solution")
        raise RuntimeError("MILP/approx solver failed to produce a solution")
    
    

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

    # Print summary metrics to stdout for quick terminal inspection
    print("[mip] summary:")
    print(f"  status: {solution.get('status')}")
    print(f"  model_makespan: {float(solution.get('makespan', float('nan'))):.6f}")
    print(f"  lp_makespan: {float(makespan):.6f}")
    print(f"  total_hw_area: {float(solution.get('total_hardware_area', float('nan'))):.6f}")
    print(f"  area_limit: {float(A_max):.6f}")
    print(f"  solve_time_sec: {solve_sec:.3f}")
    print(f"  total_time_sec: {time.perf_counter() - t0:.3f}")
    
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

    # Save visualization images using the same plotting pipeline as diff_gnn
    viz_cfg = config.get("visualization", {})
    viz_enabled = bool(viz_cfg.get("enabled", True))
    if viz_enabled:
        try:
            from tools import visualize_schedule_from_partitions as viz_tools
        except Exception as e:
            logger.warning(f"Visualization module import failed: {e}")
            print(f"[mip] warning: visualization module unavailable ({e})")
            viz_tools = None

        if viz_tools is not None:
            try:
                task_graph_viz, task_graph_viz_src = _resolve_taskgraph_for_visualization(
                    config=config,
                    taskgraph_pickle_used=taskgraph_pickle_used,
                    taskgraph_copy_path=taskgraph_copy_path,
                )
                if task_graph_viz is None:
                    raise FileNotFoundError(
                        "TaskGraph pickle not available; cannot use shared diff_gnn visualization pipeline."
                    )

                cfg_for_viz = _to_plain_dict(config)
                if not isinstance(cfg_for_viz, dict):
                    cfg_for_viz = dict(config)
                cfg_for_viz.setdefault("config", "config_mip")
                if cfg_for_viz.get("config") == "config_mip":
                    tg_pickle_name = os.path.basename(str(config.get("taskgraph-pickle", "")))
                    marker = "instance-config-"
                    if marker in tg_pickle_name and tg_pickle_name.endswith(".pkl"):
                        cfg_name = tg_pickle_name.split(marker, 1)[-1][:-4]
                        cfg_for_viz["config"] = f"{cfg_name}.yaml"
                run_tag = viz_tools._run_tag_from_config(cfg_for_viz)

                vis_cfg_plain = _to_plain_dict(viz_cfg)
                if not isinstance(vis_cfg_plain, dict):
                    vis_cfg_plain = {}
                out_root = vis_cfg_plain.get(
                    "out_dir",
                    os.path.join(config.get("output-dir", "outputs"), "final_visualizations", "mip"),
                )
                input_out_dir = vis_cfg_plain.get("input_dir", os.path.join(out_root, "input"))
                sched_out_dir = vis_cfg_plain.get("schedule_dir", os.path.join(out_root, "schedule"))
                os.makedirs(input_out_dir, exist_ok=True)
                os.makedirs(sched_out_dir, exist_ok=True)

                input_png = os.path.join(input_out_dir, f"{run_tag}_input_taskgraph.png")
                sched_png = os.path.join(sched_out_dir, f"{run_tag}_mip_schedule.png")

                viz_tools._plot_input_task_graph(
                    task_graph_viz,
                    input_png,
                    context={"run_name": run_tag, "seed": config.get("seed", "-")},
                )
                viz_tools._plot_schedule(
                    task_graph_viz,
                    partition_assignment,
                    "mip",
                    sched_png,
                    context={
                        "run_name": run_tag,
                        "seed": config.get("seed", "-"),
                        "partition_file": os.path.basename(str(partition_path)),
                    },
                )
                logger.info(f"Visualization task graph source: {task_graph_viz_src}")
                print(f"[mip] saved input graph image: {input_png}")
                print(f"[mip] saved partition image: {sched_png}")
            except Exception as e:
                logger.warning(f"Failed to generate shared visualizations: {e}", exc_info=True)
                print(f"[mip] warning: shared visualization failed ({e})")


if __name__ == "__main__":
    main()
