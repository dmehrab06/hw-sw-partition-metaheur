import numpy as np
import pandas as pd
import random
import warnings
from datetime import datetime
import os
from pathlib import Path
import pickle
from pprint import pprint

from utils.logging_utils import LogManager
print(f"After LogManager - report.log exists: {os.path.exists('report.log')}")
warnings.filterwarnings('ignore')

LogManager.initialize("logs/run_meta_heuristic.log")
print(f"After LogManager.initialize - report.log exists: {os.path.exists('report.log')}")
logger = LogManager.get_logger(__name__)

from meta_heuristic import ( 
    TaskGraph, parse_arguments, 
    simulate_PSO, random_assignment, simulate_GL25,
    simulate_DBPSO, simulate_CLPSO, simulate_CCPSO,
    simulate_SHADE, simulate_JADE, simulate_ESA,
    simulate_nondiff_GNN, simulate_diff_GNN
)

from meta_heuristic.metaheuristic_registry import MethodRegistry

def save_taskgraph(config, task_graph):
    """Save TaskGraph instance to pickle file"""
    graph_name = Path(config['graph-file']).stem
    config_name = Path(config['config']).stem
    
    filename = f"taskgraph-{graph_name}-instance-config-{config_name}.pkl"

    # Create directory for TaskGraph instances
    taskgraph_dir = config.get('taskgraph-dir', 'taskgraph_instances')
    os.makedirs(taskgraph_dir, exist_ok=True)
    
    filepath = os.path.join(taskgraph_dir, filename)
    
    try:
        with open(filepath, "wb") as file:
            pickle.dump(task_graph, file)
        logger.info(f"TaskGraph instance saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save TaskGraph instance: {e}")
        return None

def save_partition(args, solution, method='random'):
    """Save partition to pickle file"""
    assert isinstance(solution, dict), "The object is not of type 'dict'"

    graph_name = Path(args['graph-file']).stem
    
    filename = (f"taskgraph-{graph_name}_"
               f"area-{args['area-constraint']:.2f}_"
               f"hwscale-{args['hw-scale-factor']:.1f}_"
               f"hwvar-{args['hw-scale-variance']:.2f}_"
               f"comm-{args['comm-scale-factor']:.2f}_"
               f"seed-{args['seed']}_"
               f"assignment-{method}.pkl")

    os.makedirs(args['solution-dir'], exist_ok=True)
    with open(f"{args['solution-dir']}/{filename}", "wb") as file:
        pickle.dump(solution, file)

def save_results_to_csv(config, results_dict, N, very_naive_lower_bound):
    """Save results to CSV file."""
    formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    config_base = Path(config['config']).stem
    
    # Base result data
    base_data = {
        'SimTime': formatted_time,
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
    
    # Merge with method results
    result_summary_data = [{**base_data, **results_dict}]
    
    result_df = pd.DataFrame.from_dict(result_summary_data)

    print(result_df)
    pprint(result_summary_data)
    # print(result_summary_data)
    
    # Create outputs directory if it doesn't exist
    os.makedirs(config['output-dir'], exist_ok=True)
    file_path = f"outputs/{config['result-file-prefix']}-result-summary-soda-graphs-config.csv"
    
    # Ensure outputs dir exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If a CSV already exists, align this row to the existing header columns so values
    # land in the appropriate column positions. Any new columns are appended.
    if os.path.exists(file_path):
        try:
            existing_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
        except Exception:
            # Fallback: if header can't be read, just append with current header
            result_df.to_csv(file_path, mode='a', index=False, header=False)
        else:
            ordered_cols = existing_cols + [c for c in result_df.columns if c not in existing_cols]
            result_df_reordered = result_df.reindex(columns=ordered_cols)
            result_df_reordered.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # New file: write header
        result_df.to_csv(file_path, mode='a', index=False, header=True)
    
    logger.info(f"Results saved to {file_path}")

def main():
    # Parse arguments and load config
    config = parse_arguments()
    print(config)
    
    # Set random seeds for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")
    
    try:
        # Initialize Task Graph
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

        print("visualize the task graph")

        # from figures.utils import visualize_task_graph
        # visualize_task_graph(config=config, task_graph=TG, out_dir='Figs', fig_name='initial_task_graph')

        save_taskgraph(config,TG)
        
        N = len(TG.graph.nodes())
        logger.info(f"Graph loaded successfully with {N} nodes")
        
        # Initialize method registry
        registry = MethodRegistry()
        
        # Register optimization methods
        
        #registry.register_method('random', random_assignment) 
        #registry.register_method('non_diffgnn', simulate_nondiff_GNN)
        registry.register_method('diff_gnn', simulate_diff_GNN)
        # registry.register_method('gcon', simulate_gcon)
        
        # registry.register_method('pso', simulate_PSO)
        # registry.register_method('dbpso',simulate_DBPSO)
        # registry.register_method('clpso',simulate_CLPSO)
        # registry.register_method('ccpso',simulate_CCPSO)
        
        # registry.register_method('esa',simulate_ESA)
        
        # registry.register_method('shade',simulate_SHADE)
        # registry.register_method('jade',simulate_JADE)
        
        # registry.register_method('gl25', simulate_GL25)
        
        # Calculate baseline
        very_naive_lower_bound = sum(min(TG.software_costs[node], TG.hardware_costs[node]) 
                                   for node in TG.graph.nodes())
        
        # Run greedy heuristic first (since it's internal to TaskGraph)
        logger.info('='*50)
        logger.info('STARTING GREEDY HEURISTIC')
        logger.info('='*50)
        
        greedy_best_cost, greedy_solution = TG.greedy_heur()
        greedy_result = registry.add_manual_result(
            'greedy', greedy_best_cost, greedy_solution, TG
        )
        logger.info(f"Greedy Result: {greedy_best_cost:.4f}")
        
        # Run all registered optimization methods
        for method_name in registry.get_registered_method_names():
            logger.info('='*50)
            logger.info(f'STARTING {method_name.upper()} OPTIMIZATION')
            logger.info('='*50)
            
            if method_name == 'non_diffgnn':
                func_to_optimize = (
                    TG.optimize_gcomopt_makespan
                    if config['opt-cost-type'] == 'makespan'
                    else TG.optimize_gcomopt_makespan_mip
                    if config['opt-cost-type'] == 'mip'
                    else TG.optimize_gcomopt
                )

                print(f"func_to_optimize is {getattr(func_to_optimize,'__name__','didntgetaname')}")

            elif method_name == 'pso':
                func_to_optimize = (TG.optimize_swarm_makespan if config['opt-cost-type']=='makespan' 
                                    else (TG.optimize_swarm_makespan_mip if config['opt-cost-type']=='mip' else TG.optimize_swarm))
            elif method_name in ['random','dbpso']:
                func_to_optimize = (TG.optimize_random_makespan if config['opt-cost-type']=='makespan' 
                                   else (TG.optimize_random_makespan_mip if config['opt-cost-type']=='mip' else TG.optimize_random))
            else:
                func_to_optimize = (TG.optimize_single_point_makespan if config['opt-cost-type']=='makespan' 
                                    else (TG.optimize_single_point_makespan_mip if config['opt-cost-type']=='mip' else TG.optimize_single_point))

            logger.info(f"{method_name.upper()} will optimize function {getattr(func_to_optimize,'__name__','didntgetaname')} as black box")
            
            result = registry.run_method(
                method_name, N, func_to_optimize, config, TG, naive_opt_func_name = config['opt-cost-type']
            )
            
            logger.info(f"{method_name.upper()} Result: {result.best_optimization_cost:.4f}")
        
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

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        logger.info("="*80)
        logger.info("TASK GRAPH PARTITIONING EVALUATION - LOG END")
        logger.info("="*80)

if __name__ == "__main__":
    main()