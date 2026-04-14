"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
import json
import random
import pickle
from pathlib import Path

import numpy as np
import time
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from meta_heuristic.partition_schedule_evaluator import (
    PartitionScheduleProblem,
    evaluate_partition_lssp,
    synchronize_problem_with_config,
)
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
                    task_graph = pickle.load(f)
                return synchronize_problem_with_config(task_graph, config), candidate
            except Exception:
                continue
    return None, None


def _build_problem_from_graph(graph, area_constraint: float) -> PartitionScheduleProblem:
    hardware_area = {n: float(graph.nodes[n].get("area_cost", 0.0)) for n in graph.nodes()}
    return PartitionScheduleProblem(
        graph=graph,
        hardware_costs={n: float(graph.nodes[n].get("hardware_time", 0.0)) for n in graph.nodes()},
        software_costs={n: float(graph.nodes[n].get("software_time", 0.0)) for n in graph.nodes()},
        hardware_area=hardware_area,
        communication_costs={(u, v): float(graph.edges[u, v].get("communication_cost", 0.0)) for u, v in graph.edges()},
        area_constraint=float(area_constraint),
        total_area=float(sum(hardware_area.values())),
        violation_cost=1e9,
    )


def _graph_base_name(config) -> str:
    graph_file = str(config.get("graph-file", "")).strip()
    if graph_file:
        return Path(graph_file).stem
    return "taskgraph"


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _normalize_solver_tool(config, logger) -> str:
    tool = str(config.get("solver-tool", "cvxpy")).strip().lower()
    if tool in {"cvxpy", "cuopt"}:
        return tool

    requested = None
    if tool == "cvxpy-scip":
        requested = "SCIP"
    elif tool == "cvxpy-highs":
        requested = "HIGHS"
    elif tool == "cvxpy-gurobi":
        requested = "GUROBI"
    elif tool == "cvxpy-xpress":
        requested = "XPRESS"
    else:
        logger.error(f"Unsupported solver tool: {tool}")
        raise NotImplementedError(f"Unsupported solver tool: {tool}")

    mip_cfg = dict(_to_plain_dict(config.get("mip", {})) or {})
    preferred = [str(s).upper() for s in mip_cfg.get("preferred-solvers", [])]
    preferred = [requested] + [solver for solver in preferred if solver != requested]
    mip_cfg["preferred-solvers"] = preferred
    config["mip"] = mip_cfg
    config["solver-tool"] = "cvxpy"

    installed = {str(s).upper() for s in cp.installed_solvers()}
    if requested not in installed:
        logger.warning(
            "Requested solver %s via solver-tool=%s is not installed in this environment. "
            "Will fall back using preferred-solvers=%s.",
            requested,
            tool,
            preferred,
        )
    else:
        logger.info("Using requested solver backend %s via solver-tool=%s", requested, tool)
    return "cvxpy"

def main():
    t0 = time.perf_counter()
    config = parse_arguments()

    LogManager.initialize(f"logs/run_milp_optimizer_area-{config['area-constraint']:.2f}_hw-{config['hw-scale-factor']:.1f}_seed-{config['seed']}.log")
    logger = LogManager.get_logger(__name__)
    solver_tool = _normalize_solver_tool(config, logger)

    # Create solver instance
    if solver_tool == 'cvxpy':
        solver = ScheduleConstPartitionSolver()
    elif solver_tool == 'cuopt':
        solver = CuOptScheduleConstPartitionSolver()
    else:
        logger.error(f"Unsupported solver tool: {solver_tool}")
        raise NotImplementedError(f"Unsupported solver tool: {solver_tool}")
    
    # Set random seeds for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")

    try:
        # Initialize Task Graph
        task_graph_eval = None
        if os.path.exists(config.get('taskgraph-pickle', "")):
            taskgraph_pickle_used = os.path.abspath(config['taskgraph-pickle'])
            logger.info(f"Loading graph from {taskgraph_pickle_used}")
            graph = solver.load_pickle_graph(taskgraph_pickle_used)
            with open(taskgraph_pickle_used, "rb") as f:
                task_graph_eval = pickle.load(f)
            loaded_area = getattr(task_graph_eval, "area_constraint", None)
            task_graph_eval = synchronize_problem_with_config(task_graph_eval, config)
            if loaded_area is not None and abs(float(loaded_area) - float(config['area-constraint'])) > 1e-9:
                logger.warning(
                    "Loaded TaskGraph area constraint %.8f differs from config %.8f. "
                    "Using config value at runtime.",
                    float(loaded_area),
                    float(config['area-constraint']),
                )
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
            task_graph_eval = _build_problem_from_graph(graph, config['area-constraint'])
    except Exception as e:
        logger.error(f"An error occurred during loading graph from input file: {str(e)}", exc_info=True)
        raise
    
    # Solve optimization with area constraint
    A_max = np.sum(solver.a) * config['area-constraint']
    t_solve0 = time.perf_counter()
    if solver_tool == 'cvxpy':
        solution = solver.solve_optimization(A_max=A_max, solver_cfg=config.get('mip', None))
    else:
        solution = solver.solve_optimization(A_max=A_max)
    solve_sec = time.perf_counter() - t_solve0
    timed_out = bool(getattr(solver, "last_solve_timed_out", False))
    if solution is not None and not timed_out:
        solver_stats = solution.get("solver_stats", {}) if isinstance(solution, dict) else {}
        solver_stats_status = str(solver_stats.get("status", "")).strip().lower()
        if "time limit reached" in solver_stats_status:
            timed_out = True

    if solution is None:
        solver_status = getattr(solver, "last_solve_status", None)
        logger.error("Solver did not return a valid solution (status=%s)", solver_status)
        print("[mip] summary:")
        print(f"  status: {solver_status}")
        print(f"  area_limit: {float(A_max):.6f}")
        print(f"  solve_time_sec: {solve_sec:.3f}")
        print("  postprocess_time_sec: 0.000")
        print(f"  algorithm_total_time_sec: {solve_sec:.3f}")
        print(f"  total_time_sec: {time.perf_counter() - t0:.3f}")
        if timed_out:
            raise SystemExit(124)
        raise RuntimeError("MILP solver failed to produce an accepted exact solution")
    
    

    postprocess_t0 = time.perf_counter()

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
    
    # Compute final makespan from the solved partition using the shared LSSP evaluator
    lp_assignment = [1 - partition_assignment[n] for n in graph.nodes()]
    lp_makespan, _ = compute_dag_makespan(graph, lp_assignment)
    lssp_result = evaluate_partition_lssp(task_graph_eval, partition_assignment)
    lssp_makespan = float(lssp_result["makespan"])
    postprocess_sec = time.perf_counter() - postprocess_t0
    algorithm_total_sec = solve_sec + postprocess_sec
    logger.info(f"LP makespan: {lp_makespan}")
    logger.info(f"LSSP makespan: {lssp_makespan}")

    # Print summary metrics to stdout for quick terminal inspection
    print("[mip] summary:")
    print(f"  status: {solution.get('status')}")
    print(f"  model_makespan: {float(solution.get('makespan', float('nan'))):.6f}")
    print(f"  lp_makespan: {float(lp_makespan):.6f}")
    print(f"  final_lssp_makespan: {float(lssp_makespan):.6f}")
    print(f"  total_hw_area: {float(solution.get('total_hardware_area', float('nan'))):.6f}")
    print(f"  area_limit: {float(A_max):.6f}")
    print(f"  solve_time_sec: {solve_sec:.3f}")
    print(f"  postprocess_time_sec: {postprocess_sec:.3f}")
    print(f"  algorithm_total_time_sec: {algorithm_total_sec:.3f}")
    print(f"  total_time_sec: {time.perf_counter() - t0:.3f}")
    
    area_constraint_str = f"{config['area-constraint']:.2f}"
    hwscale_str = f"{config['hw-scale-factor']:.1f}"
    hwvar_str = f"{config['hw-scale-variance']:.2f}"
    seed_str = f"{config['seed']}"
    output_dir = f"{config['solution-dir']}"
    graph_base = _graph_base_name(config)
    
    dir = Path(output_dir)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dir, 0o777)

    logger.info(f"Saving partitions as pickle file in {output_dir}")
    partition_base = f"taskgraph-{graph_base}_area-{area_constraint_str}_hwscale-{hwscale_str}_hwvar-{hwvar_str}_seed-{seed_str}"
    partition_path = Path(output_dir) / f"{partition_base}_assignment-mip.pkl"
    with open(partition_path, 'wb') as f:
        pickle.dump(partition_assignment,f)

    json_solution = dict(solution)
    json_solution["partition_assignment"] = [partition_assignment]
    json_solution["wall_time"] = float(time.perf_counter() - t0)
    json_solution["lp_makespan"] = float(lp_makespan)
    json_solution["final_lssp_makespan"] = float(lssp_makespan)
    json_solution["solve_time_sec"] = float(solve_sec)
    json_solution["postprocess_time_sec"] = float(postprocess_sec)
    json_solution["algorithm_total_time_sec"] = float(algorithm_total_sec)
    json_solution["total_time_sec"] = float(time.perf_counter() - t0)
    json_solution["solver_tool"] = config.get("solver-tool")
    json_solution["taskgraph_pickle"] = taskgraph_pickle_used
    json_solution["graph_file"] = config.get("graph-file")
    json_solution["time_limit_exceeded"] = bool(timed_out)
    json_path = Path(output_dir) / f"{partition_base}_assignment-mip.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(json_solution), f, indent=2)
    logger.info(f"Wrote main-compatible JSON solution to {json_path}")

    # Persist the exact TaskGraph pickle used for this solve (prevents later overwrite mismatches)
    taskgraph_copy_path = None
    if taskgraph_pickle_used:
        taskgraph_copy_path = Path(output_dir) / f"{partition_base}_taskgraph.pkl"
        with open(taskgraph_copy_path, "wb") as f:
            pickle.dump(task_graph_eval, f)
        logger.info(f"Saved synchronized TaskGraph pickle to {taskgraph_copy_path}")

    # Write solve metadata alongside the partition
    meta = {
        "taskgraph_pickle": taskgraph_pickle_used,
        "taskgraph_pickle_copy": str(taskgraph_copy_path) if taskgraph_copy_path else None,
        "assignment_json": str(json_path),
        "graph_file": config.get("graph-file"),
        "area_constraint": config.get("area-constraint"),
        "hw_scale_factor": config.get("hw-scale-factor"),
        "hw_scale_variance": config.get("hw-scale-variance"),
        "comm_scale_factor": config.get("comm-scale-factor"),
        "seed": config.get("seed"),
        "solver_tool": config.get("solver-tool"),
        "solver_backend": solution.get("solver_backend"),
        "solver_status": solution.get("status"),
        "solver_stats": solution.get("solver_stats"),
        "time_limit_exceeded": bool(timed_out),
        "model_makespan": float(solution.get("makespan", float("nan"))),
        "lp_makespan": float(lp_makespan),
        "final_lssp_makespan": float(lssp_makespan),
        "solve_time_sec": float(solve_sec),
        "postprocess_time_sec": float(postprocess_sec),
        "algorithm_total_time_sec": float(algorithm_total_sec),
        "total_time_sec": float(time.perf_counter() - t0),
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
                    config=cfg_for_viz,
                )
                logger.info(f"Visualization task graph source: {task_graph_viz_src}")
                print(f"[mip] saved input graph image: {input_png}")
                print(f"[mip] saved partition image: {sched_png}")
            except Exception as e:
                logger.warning(f"Failed to generate shared visualizations: {e}", exc_info=True)
                print(f"[mip] warning: shared visualization failed ({e})")

    if timed_out:
        logger.warning("MILP hit the time limit; incumbent artifacts were written for downstream reporting.")
        raise SystemExit(124)


if __name__ == "__main__":
    main()
