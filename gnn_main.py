import numpy as np
import pandas as pd
import random
import warnings
import csv
from datetime import datetime
import os
from pathlib import Path
import pickle
from pprint import pprint
from collections.abc import Mapping

from utils.logging_utils import LogManager
warnings.filterwarnings('ignore')

logger = LogManager.get_logger(__name__)

from meta_heuristic import ( 
    TaskGraph, parse_arguments, 
    simulate_PSO, random_assignment, simulate_GL25,
    simulate_DBPSO, simulate_CLPSO, simulate_CCPSO,
    simulate_SHADE, simulate_JADE, simulate_ESA,
    simulate_nondiff_GNN, simulate_diff_GNN, simulate_diff_GNN_order,
    simulate_gcps
)

from meta_heuristic.metaheuristic_registry import MethodRegistry
from meta_heuristic.partition_schedule_evaluator import synchronize_problem_with_config
from meta_heuristic.Configuration import (
    get_classical_search_objective_default,
    is_classical_search_method,
)
try:
    from tools.visualize_schedule_from_partitions import generate_visualizations_for_run
except Exception:
    generate_visualizations_for_run = None

def _greedy_adapter(dim, func_to_optimize, config, task_graph=None, **kwargs):
    if task_graph is None:
        raise ValueError("Greedy adapter requires task_graph")
    _, solution = task_graph.greedy_heur()
    assignment = np.array(
        [float(solution[node]) for node in task_graph.graph.nodes()],
        dtype=float,
    )
    best_cost = float(func_to_optimize(assignment))
    return best_cost, solution


def _resolve_effective_search_objective(config, method_name):
    """Resolve the search-time objective used by each method."""
    requested = str(config.get("opt-cost-type", "partition")).strip().lower()
    if requested not in {"partition", "makespan", "mip", "lssp"}:
        return requested
    if requested == "partition":
        return requested
    method_key = str(method_name).lower()
    if not is_classical_search_method(method_key):
        return requested

    override = (
        os.getenv("HWSW_CLASSICAL_SEARCH_OBJECTIVE")
        or config.get("classical-search-objective")
        or get_classical_search_objective_default()
    )
    normalized = str(override).strip().lower()
    if normalized in {"requested", "same"}:
        return requested
    if normalized in {"lssp"}:
        return "lssp"
    if normalized in {"mip", "dag", "fast-dag", "fast_dag"}:
        return "mip"
    if normalized in {"makespan", "queue", "taskgraph"}:
        return "makespan"
    return requested


def _describe_objective_mode(mode):
    if mode == "partition":
        return "partition-cost"
    if mode == "mip":
        return "fast-dag"
    if mode == "lssp":
        return "lssp-makespan"
    if mode == "makespan":
        return "queue-makespan"
    return str(mode)


def _select_optimization_callable(task_graph, method_name, objective_mode):
    if method_name == "non_diffgnn":
        return (
            task_graph.optimize_gcomopt_makespan
            if objective_mode == "makespan"
            else task_graph.optimize_gcomopt_makespan_lssp
            if objective_mode == "lssp"
            else task_graph.optimize_gcomopt_makespan_mip
            if objective_mode == "mip"
            else task_graph.optimize_gcomopt
        )
    if method_name == "pso":
        return (
            task_graph.optimize_swarm_makespan
            if objective_mode == "makespan"
            else task_graph.optimize_swarm_makespan_lssp
            if objective_mode == "lssp"
            else task_graph.optimize_swarm_makespan_mip
            if objective_mode == "mip"
            else task_graph.optimize_swarm
        )
    if method_name in ["random", "dbpso"]:
        return (
            task_graph.optimize_random_makespan
            if objective_mode == "makespan"
            else task_graph.optimize_random_makespan_lssp
            if objective_mode == "lssp"
            else task_graph.optimize_random_makespan_mip
            if objective_mode == "mip"
            else task_graph.optimize_random
        )
    return (
        task_graph.optimize_single_point_makespan
        if objective_mode == "makespan"
        else task_graph.optimize_single_point_makespan_lssp
        if objective_mode == "lssp"
        else task_graph.optimize_single_point_makespan_mip
        if objective_mode == "mip"
        else task_graph.optimize_single_point
    )

AVAILABLE_METHODS = {
    'random': random_assignment,
    'greedy': _greedy_adapter,
    # 'non_diffgnn': simulate_nondiff_GNN,
    'diff_gnn': simulate_diff_GNN,
    'diff_gnn_order': simulate_diff_GNN_order,
    'gcps': simulate_gcps,
    'pso': simulate_PSO,
    'dbpso': simulate_DBPSO,
    'clpso': simulate_CLPSO,
    'ccpso': simulate_CCPSO,
    'esa': simulate_ESA,
    'shade': simulate_SHADE,
    'jade': simulate_JADE,
    'gl25': simulate_GL25,
}

#DEFAULT_METHODS = list(AVAILABLE_METHODS.keys())
DEFAULT_METHODS = ['random', 'greedy', 'diff_gnn', 'diff_gnn_order', 'gl25']

def _parse_methods(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.lower() in ("all", "auto"):
            return ["all"]
        return [v.strip() for v in raw.split(",") if v.strip()]
    return None

def _get_selected_methods(config):
    env_methods = os.getenv("HWSW_METHODS") or os.getenv("METHODS")
    selected = _parse_methods(env_methods)
    if selected is None:
        selected = _parse_methods(config.get("methods", None) or config.get("method-list", None))
    if not selected:
        return DEFAULT_METHODS
    if len(selected) == 1 and selected[0].lower() == "all":
        return list(AVAILABLE_METHODS.keys())
    return selected


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
        if s in ("0", "false", "no", "off", "disable", "disabled"):
            return False
    return default


def _run_auto_visualizations(config, task_graph, methods):
    vis_cfg_raw = config.get("visualization", {})
    vis_cfg = dict(vis_cfg_raw) if isinstance(vis_cfg_raw, Mapping) else {}
    enabled = _as_bool(vis_cfg.get("enabled", False), False)
    if not enabled:
        return

    if generate_visualizations_for_run is None:
        logger.warning("Visualization enabled, but visualization module is unavailable.")
        return

    include_input = _as_bool(
        vis_cfg.get("include_input_graph", vis_cfg.get("input_graph", True)),
        True,
    )
    include_output = _as_bool(
        vis_cfg.get("include_output_schedule", vis_cfg.get("output_schedule", True)),
        True,
    )
    strict_partitions = _as_bool(vis_cfg.get("strict_partitions", False), False)
    vis_methods = _parse_methods(vis_cfg.get("methods", None))
    if not vis_methods:
        vis_methods = list(methods)
    out_dir = vis_cfg.get("out_dir", None) or vis_cfg.get("output_dir", None)

    try:
        saved_paths = generate_visualizations_for_run(
            config=config,
            methods=vis_methods,
            out_dir=out_dir,
            task_graph=task_graph,
            include_input=include_input,
            include_output=include_output,
            strict_partitions=strict_partitions,
        )
        if saved_paths:
            logger.info("Auto visualization generated %d file(s).", len(saved_paths))
            for p in saved_paths:
                logger.info("Visualization: %s", p)
        else:
            logger.warning("Visualization enabled, but no files were generated.")
    except Exception as e:
        logger.warning("Visualization failed: %s", str(e), exc_info=True)

def _print_partition_assignment(method_name, partition):
    """Print compact HW/SW node lists for a partition assignment."""
    if not isinstance(partition, dict):
        print(f"[{method_name}] partition is not a dict; got {type(partition).__name__}")
        return
    hw_nodes = sorted([n for n, v in partition.items() if v == 1])
    sw_nodes = sorted([n for n, v in partition.items() if v == 0])
    print(f"[{method_name}] hardware nodes ({len(hw_nodes)}): {', '.join(hw_nodes)}")
    print(f"[{method_name}] software nodes ({len(sw_nodes)}): {', '.join(sw_nodes)}")

def save_taskgraph(config, task_graph):
    """Save TaskGraph instance to pickle file"""
    # If config specifies an explicit pickle path, honor it for consistency across runs
    explicit_path = config.get('taskgraph-pickle', None)
    if explicit_path:
        filepath = explicit_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        graph_name = Path(config['graph-file']).stem
        config_name = Path(config['config']).stem
        filename = f"taskgraph-{graph_name}-instance-config-{config_name}.pkl"
        taskgraph_dir = config.get('taskgraph-dir', 'taskgraph_instances')
        os.makedirs(taskgraph_dir, exist_ok=True)
        filepath = os.path.join(taskgraph_dir, filename)
    
    try:
        with open(filepath, "wb") as file:
            pickle.dump(task_graph, file)
        logger.info(f"TaskGraph instance saved to: {filepath}")
        # Keep config aligned for downstream consumers
        config['taskgraph-pickle'] = filepath
        return filepath
    except Exception as e:
        logger.error(f"Failed to save TaskGraph instance: {e}")
        return None

def load_taskgraph_if_available(config):
    """Load a TaskGraph pickle if it exists and regeneration is not forced."""
    tg_pickle = os.getenv("HWSW_TASKGRAPH_PICKLE") or config.get('taskgraph-pickle')
    force_regen = os.getenv("HWSW_FORCE_TG_REGEN", "0").lower() in ("1", "true", "yes")

    def _ensure_taskgraph_runtime_fields(task_graph):
        if not hasattr(task_graph, "violation_cost"):
            task_graph.violation_cost = 1e9
        if not hasattr(task_graph, "node_to_num") or not getattr(task_graph, "node_to_num", None):
            task_graph.node_to_num = {node: i for i, node in enumerate(task_graph.graph.nodes())}
        if not hasattr(task_graph, "num_to_node") or not getattr(task_graph, "num_to_node", None):
            task_graph.num_to_node = {idx: node for node, idx in task_graph.node_to_num.items()}
        if not hasattr(task_graph, "total_area") or getattr(task_graph, "total_area", 0.0) == 0.0:
            task_graph.total_area = float(sum(getattr(task_graph, "hardware_area", {}).values()))

    if tg_pickle and os.path.exists(tg_pickle) and not force_regen:
        try:
            with open(tg_pickle, "rb") as f:
                TG = pickle.load(f)
            _ensure_taskgraph_runtime_fields(TG)
            loaded_area = getattr(TG, "area_constraint", None)
            synchronize_problem_with_config(TG, config)
            logger.info(f"Loaded TaskGraph instance from: {tg_pickle}")
            if loaded_area is not None and abs(float(loaded_area) - float(config['area-constraint'])) > 1e-9:
                logger.warning(
                    "Loaded TaskGraph area constraint %.8f differs from config %.8f. "
                    "Using config value at runtime.",
                    float(loaded_area),
                    float(config['area-constraint']),
                )
            # Keep config aligned for downstream consumers
            config['taskgraph-pickle'] = tg_pickle
            return TG
        except Exception as e:
            logger.warning(f"Failed to load TaskGraph pickle {tg_pickle}: {e}. Will regenerate.")
    return None

def save_partition(args, solution, method='random'):
    """Save partition to pickle file"""
    assert isinstance(solution, dict), "The object is not of type 'dict'"

    graph_name = Path(args['graph-file']).stem
    run_tag = os.getenv("HWSW_RUN_TAG", "").strip()
    
    filename = (f"taskgraph-{graph_name}_"
               f"area-{args['area-constraint']:.2f}_"
               f"hwscale-{args['hw-scale-factor']:.1f}_"
               f"hwvar-{args['hw-scale-variance']:.2f}_"
               f"comm-{args['comm-scale-factor']:.2f}_"
               f"seed-{args['seed']}_"
               f"assignment-{method}.pkl")
    if run_tag:
        filename = filename.replace(".pkl", f"__run-{run_tag}.pkl")

    solution_dir = os.getenv("HWSW_SOLUTION_DIR") or args['solution-dir']
    os.makedirs(solution_dir, exist_ok=True)
    with open(os.path.join(solution_dir, filename), "wb") as file:
        pickle.dump(solution, file)

def save_results_to_csv(config, results_dict, N, very_naive_lower_bound):
    """Save results to CSV file."""
    formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    config_base = Path(config['config']).stem
    
    # Base result data
    base_data = {
        'SimTime': formatted_time,
        'RunTag': os.getenv("HWSW_RUN_TAG", ""),
        'Config': config_base,
        'GraphName': config['graph-file'],
        'N': N,
        'HW_Scale_Factor': config['hw-scale-factor'],
        'HW_Scale_Var': config['hw-scale-variance'],
        'Comm_Scale_Var': config['comm-scale-factor'],
        'Area_Percentage': config['area-constraint'],
        'Seed': config['seed'],
        'LB_Naive': very_naive_lower_bound,
    }
    base_cols = list(base_data.keys())

    # Build a consistent column list for all known methods
    method_names = list(AVAILABLE_METHODS.keys())
    method_cols = []
    for method in method_names:
        for suffix in [
            'opt_cost',
            'opt_ratio',
            'partition_cost',
            'bb',
            'makespan',
            'time',
            'optimization_time_sec',
            'postprocess_time_sec',
            'total_runtime_sec',
            'dag_makespan',
            'lssp_makespan',
            'lssp_swprio_makespan',
            'best_makespan',
        ]:
            method_cols.append(f"{method}_{suffix}")

    # Default empty values for all method columns so they exist in the CSV
    full_row = {**base_data, **{c: None for c in method_cols}, **results_dict}

    # Include any extra metrics not in the standard columns
    extra_cols = [c for c in full_row.keys() if c not in base_cols + method_cols]
    full_cols = base_cols + method_cols + extra_cols

    result_df = pd.DataFrame.from_dict([full_row]).reindex(columns=full_cols)
    
    out_dir = os.getenv("HWSW_CSV_DIR") or config.get('output-dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    csv_name_override = (
        os.getenv("HWSW_RESULT_CSV")
        or config.get("result-csv")
        or config.get("result-csv-name")
    )
    result_prefix = os.getenv("HWSW_RESULT_PREFIX") or config.get('result-file-prefix', 'results')

    if csv_name_override:
        file_path = (
            csv_name_override
            if os.path.isabs(csv_name_override)
            else os.path.join(out_dir, csv_name_override)
        )
    else:
        file_path = os.path.join(out_dir, f"{result_prefix}-result-summary-soda-graphs-config.csv")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If a CSV already exists, align this row to the existing header columns so values
    # land in the appropriate column positions. Any new columns are appended.
    if os.path.exists(file_path):
        with open(file_path, newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            result_df.to_csv(file_path, mode='w', index=False, header=True)
        else:
            existing_cols = list(rows[0])
            ordered_cols = existing_cols + [c for c in result_df.columns if c not in existing_cols]

            row_dicts = []
            for raw in rows[1:]:
                padded = list(raw) + [""] * max(0, len(ordered_cols) - len(raw))
                row_dicts.append(dict(zip(ordered_cols, padded[:len(ordered_cols)])))

            if len(ordered_cols) != len(existing_cols):
                existing_df = pd.DataFrame(row_dicts, columns=ordered_cols)
                result_df_reordered = result_df.reindex(columns=ordered_cols)
                combined_df = pd.concat([existing_df, result_df_reordered], ignore_index=True)
                combined_df.to_csv(file_path, mode='w', index=False, header=True)
            else:
                result_df_reordered = result_df.reindex(columns=ordered_cols)
                result_df_reordered.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # New file: write header
        result_df.to_csv(file_path, mode='a', index=False, header=True)
    
    logger.info(f"Results saved to {file_path}")

def main():
    LogManager.initialize("logs/run_meta_heuristic.log")

    # Parse arguments and load config
    config = parse_arguments()
    
    # Set random seeds for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")
    
    try:
        # Initialize Task Graph (prefer existing pickle for consistent comparisons)
        TG = load_taskgraph_if_available(config)
        if TG is None:
            logger.info(f"Loading graph from {config['graph-file']}")
            TG = TaskGraph(area_constraint=config['area-constraint'])
            TG.load_graph_from_pydot(
                config['graph-file'],
                k=config['hw-scale-factor'],
                l=config['hw-scale-variance'],
                mu=config['comm-scale-factor'],
                A_max=100,
                seed=config['seed']
            )
            synchronize_problem_with_config(TG, config)

            save_taskgraph(config, TG)
        
        N = len(TG.graph.nodes())
        logger.info(f"Graph loaded successfully with {N} nodes")
        
        # Initialize method registry
        registry = MethodRegistry()

        selected_methods = _get_selected_methods(config)
        unknown_methods = [m for m in selected_methods if m not in AVAILABLE_METHODS]
        if unknown_methods:
            logger.warning("Unknown methods requested (skipping): %s", ", ".join(unknown_methods))
        selected_methods = [m for m in selected_methods if m in AVAILABLE_METHODS]
        if not selected_methods:
            raise ValueError("No valid methods selected. Check config 'methods' or env HWSW_METHODS.")

        logger.info("Available methods: %s", ", ".join(AVAILABLE_METHODS.keys()))
        logger.info("Selected methods: %s", ", ".join(selected_methods))

        for method_name in selected_methods:
            if method_name == 'greedy':
                registry.register_method(method_name, AVAILABLE_METHODS[method_name], task_graph=TG)
            else:
                registry.register_method(method_name, AVAILABLE_METHODS[method_name])
        
        # Calculate baseline
        very_naive_lower_bound = sum(min(TG.software_costs[node], TG.hardware_costs[node]) 
                                   for node in TG.graph.nodes())
        
        # Run all registered optimization methods
        for method_name in registry.get_registered_method_names():
            logger.info('='*50)
            logger.info(f'STARTING {method_name.upper()} OPTIMIZATION')
            logger.info('='*50)
            requested_objective = str(config.get('opt-cost-type', 'partition')).strip().lower()
            effective_objective = _resolve_effective_search_objective(config, method_name)
            func_to_optimize = _select_optimization_callable(TG, method_name, effective_objective)

            logger.info(
                "%s search objective requested=%s (%s), effective=%s (%s), black-box=%s",
                method_name.upper(),
                requested_objective,
                _describe_objective_mode(requested_objective),
                effective_objective,
                _describe_objective_mode(effective_objective),
                getattr(func_to_optimize, '__name__', 'didntgetaname'),
            )
            
            result = registry.run_method(
                method_name,
                N,
                func_to_optimize,
                config,
                TG,
                naive_opt_func_name=effective_objective,
            )
            
            logger.info(f"{method_name.upper()} Result: {result.best_optimization_cost:.4f}")
            _print_partition_assignment(method_name, result.partition_assignment)
        
        # Save all partitions
        logger.info('='*50)
        logger.info('SAVING RESULTANT PARTITIONS')
        logger.info('='*50)
        
        for method_name in registry.get_all_method_names():
            result = registry.results[method_name]
            save_partition(config, result.partition_assignment, method_name)
        
        # Generate results dictionary
        results_dict = registry.get_results_dict(very_naive_lower_bound)
        
        # Log comprehensive results summary
        logger.info('='*50)
        logger.info('RESULTS SUMMARY')
        logger.info('='*50)
        
        # for method_name in registry.get_all_method_names():
        #     result = registry.results[method_name]
        #     ratio = result.best_cost / very_naive_lower_bound if very_naive_lower_bound > 0 else 0
        #     makespan = result.get('makespan', -1e9)
        #     logger.info(f"{method_name.upper()}: Cost={result.best_cost:.4f}, Ratio={ratio:.4f}, Makespan={makespan}")
        
        # Save results to CSV
        save_results_to_csv(config, results_dict, N, very_naive_lower_bound)

        # Optional post-run visualization (input DAG + output schedules)
        _run_auto_visualizations(config, TG, registry.get_all_method_names())

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        logger.info("="*80)
        logger.info("TASK GRAPH PARTITIONING EVALUATION - LOG END")
        logger.info("="*80)

if __name__ == "__main__":
    main()
