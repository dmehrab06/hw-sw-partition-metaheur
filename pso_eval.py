
import numpy as np
import pandas as pd
import random
import warnings
from datetime import datetime

import os

from utils.logging_utils import LogManager
warnings.filterwarnings('ignore')

LogManager.initialize("logs/run_meta_heuristic.log")
logger = LogManager.get_logger(__name__)

from meta_heuristic import ( 
    TaskGraph, parse_arguments, 
    simulate_PSO, random_assignment
)


def log_results_summary(config, results):
    """
    Log a comprehensive summary of all results.
    
    Args:
        config: Input arguments
        results: Dictionary containing all results
        logger: Logger instance
    """
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    logger.info(f"Graph: {config['graph-file']} (N={results['N']} nodes)")
    logger.info(f"Environment parameters:")
    logger.info(f"  - Hardware scale factor (k): {config['hw-scale-factor']}")
    logger.info(f"  - Hardware scale variance (l): {config['hw-scale-variance']}")
    logger.info(f"  - Communication scale factor (\\mu): {config['comm-scale-factor']}")
    logger.info(f"  - Area constraint: {config['area-constraint']}")
    
    logger.info(f"\nAlgorithm Performance:")
    logger.info(f"  - Naive Lower Bound: {results['lower_bound']:.4f}")
    logger.info(f"  - Random Assignment: {results['random_cost']:.4f} (ratio: {results['random_ratio']:.4f})")
    logger.info(f"  - Greedy Heuristic: {results['greedy_cost']:.4f} (ratio: {results['greedy_ratio']:.4f})")
    logger.info(f"  - PSO Optimization: {results['pso_cost']:.4f} (ratio: {results['pso_ratio']:.4f})")
    logger.info(f"  - Best PSO params: c1={results['pso_params'][0]}, c2={results['pso_params'][1]}, w={results['pso_params'][2]}")
    
    logger.info(f"\nPerformance Ranking:")
    methods = [
        ('PSO', results['pso_cost']),
        ('Greedy', results['greedy_cost']),
        ('Random', results['random_cost'])
    ]
    methods.sort(key=lambda x: x[1])
    
    for i, (method, cost) in enumerate(methods, 1):
        logger.info(f"  {i}. {method}: {cost:.4f}")
    
    logger.info("="*80)

def main():
    # Parse arguments first
    config = parse_arguments()
    
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
            A_max=100
        )
        N = len(TG.graph.nodes())
        logger.info(f"Graph loaded successfully with {N} nodes")
        
        # Method 1: Particle Swarm Optimization
        logger.info('='*50)
        logger.info('STARTING PSO OPTIMIZATION')
        logger.info('='*50)
        
        PSO_best_cost = 1e9
        PSO_best_soln = None
        PSO_params = None
        
        param_combinations = []
        for c1 in [0.575, 1.05, 1.525]:
            for c2 in [0.1]:
                for w in [0.575, 1.05, 1.525]:
                    param_combinations.append((c1, c2, w))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, (c1, c2, w) in enumerate(param_combinations, 1):
            logger.info(f"PSO run {i}/{len(param_combinations)}: c1={c1}, c2={c2}, w={w}")
            
            best_cost, best_sol = simulate_PSO(
                N, c1, c2, w, TG.evaluation_from_swarm,
                n_particles=100, verbose=True
            )
            
            logger.info(f"  Result: {best_cost:.4f}")
            
            if best_cost < PSO_best_cost:
                PSO_best_cost = best_cost
                PSO_best_soln = best_sol
                PSO_params = (c1, c2, w)
                logger.info(f"  *** NEW BEST PSO SOLUTION ***")
        
        logger.info(f'Best PSO result: {PSO_best_cost:.4f} with params: {PSO_params}')
        
        # Method 2: Random Assignment
        logger.info('='*50)
        logger.info('STARTING RANDOM ASSIGNMENT')
        logger.info('='*50)
        
        Random_best_cost, Random_best_soln = random_assignment(N, TG.evaluation_from_swarm)
        logger.info(f'Random assignment result: {Random_best_cost:.4f}')
        
        # Method 3: Greedy heuristic
        logger.info('='*50)
        logger.info('STARTING GREEDY HEURISTIC')
        logger.info('='*50)
        
        greedy_best_cost, greedy_best_soln = TG.greedy_heur()
        logger.info(f'Greedy heuristic result: {greedy_best_cost:.4f}')
        
        # Lower bound calculation
        very_naive_lower_bound = TG.naive_lower_bound()
        logger.info(f'Naive lower bound: {very_naive_lower_bound:.4f}')
        
        # Calculate ratios
        pso_ratio = PSO_best_cost / very_naive_lower_bound
        random_ratio = Random_best_cost / very_naive_lower_bound
        greedy_ratio = greedy_best_cost / very_naive_lower_bound
        
        # Prepare results dictionary
        results = {
            'N': N,
            'lower_bound': very_naive_lower_bound,
            'random_cost': Random_best_cost,
            'greedy_cost': greedy_best_cost,
            'pso_cost': PSO_best_cost,
            'pso_params': PSO_params,
            'random_ratio': random_ratio,
            'greedy_ratio': greedy_ratio,
            'pso_ratio': pso_ratio
        }
        
        # Log comprehensive results summary
        log_results_summary(config, results)
        
        # Save results to CSV
        formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_summary_data = [{
            'SimTime': formatted_time,
            'GraphName': config['graph-file'],
            'N': N,
            'HW_Scale_Factor': config['hw-scale-factor'],
            'HW_Scale_Var': config['hw-scale-variance'],
            'Comm_Scale_Var': config['comm-scale-factor'],
            'Area_Percentage': config['area-constraint'],
            'Seed': config['seed'],
            'PSO_best_c1': PSO_params[0],
            'PSO_best_c2': PSO_params[1],
            'PSO_best_w': PSO_params[2],
            'Random_best': Random_best_cost,
            'Greedy_best': greedy_best_cost,
            'PSO_best': PSO_best_cost,
            'LB_Naive': very_naive_lower_bound,
            'Random_ratio': random_ratio,
            'Greedy_ratio': greedy_ratio,
            'PSO_ratio': pso_ratio
        }]
        
        result_df = pd.DataFrame.from_dict(result_summary_data)
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        file_path = 'outputs/result_summary_soda_graphs.csv'
        
        # Write header if file doesn't exist
        write_header = not os.path.exists(file_path)
        result_df.to_csv(file_path, mode='a', index=False, header=write_header)
        
        logger.info(f"Results saved to {file_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        logger.info("="*80)
        logger.info("TASK GRAPH PARTITIONING EVALUATION - LOG END")
        logger.info("="*80)

if __name__ == "__main__":
    main()