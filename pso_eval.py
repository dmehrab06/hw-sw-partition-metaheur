import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
import os
from meta_heuristic.TaskGraph import TaskGraph
from meta_heuristic.pso_utils import simulate_PSO, random_assignment

warnings.filterwarnings('ignore')

def create_log_filename(args):
    """
    Create a descriptive log filename based on input arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        str: Formatted filename that includes key parameters
    """
    # Extract graph name without extension
    graph_name = Path(args.graph_file).stem
    
    # Format timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive filename
    filename = (f"taskgraph-{graph_name}_"
               f"area-{args.area_constraint:.2f}_"
               f"hwscale-{args.hw_scale_factor:.1f}_"
               f"hwvar-{args.hw_scale_variance:.2f}_"
               f"comm-{args.comm_scale_factor:.2f}_"
               f"seed-{args.seed}_"
               f"{timestamp}.log")
    
    return filename

def setup_logging(verbose_level, log_filename, log_dir="logs"):
    """
    Setup logging with different verbosity levels to both file and console.
    
    Args:
        verbose_level (int): Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        log_filename (str): Name of the log file
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Determine log level
    if verbose_level == 0:
        level = logging.WARNING
    elif verbose_level == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - always log everything to file
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - respect verbosity level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log the configuration and input parameters
    logger.info("="*80)
    logger.info("TASK GRAPH PARTITIONING EVALUATION - LOG START")
    logger.info("="*80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Console verbosity level: {logging.getLevelName(level)}")
    
    return logger

def log_input_parameters(args, logger):
    """
    Log all input parameters for reference.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
    """
    logger.info("INPUT PARAMETERS:")
    logger.info("-" * 40)
    logger.info(f"Graph file: {args.graph_file}")
    logger.info(f"Area constraint: {args.area_constraint}")
    logger.info(f"Hardware scale factor: {args.hw_scale_factor}")
    logger.info(f"Hardware scale variance: {args.hw_scale_variance}")
    logger.info(f"Communication scale factor: {args.comm_scale_factor}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Verbosity level: {args.verbose}")
    logger.info("-" * 40)

def parse_arguments():
    """Parse command line arguments with proper validation."""
    parser = argparse.ArgumentParser(description='Task Graph Partitioning Evaluation')
    
    parser.add_argument('--graph-file', type=str, required=True,
                       help='Graph file name to load')
    parser.add_argument('--area-constraint', type=float, required=True,
                       help='Area constraint (should be between 0 and 1)')
    parser.add_argument('--hw-scale-factor', type=float, required=True,
                       help='Hardware scale factor (should be positive)')
    parser.add_argument('--hw-scale-variance', type=float, required=True,
                       help='Hardware scale variance (should be positive)')
    parser.add_argument('--comm-scale-factor', type=float, required=True,
                       help='Communication scale factor (should be positive)')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase verbosity level (use -v, -vv, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to store log files (default: logs)')
    
    return parser.parse_args()

def validate_arguments(args, logger):
    """Validate input arguments with assertions."""
    try:
        assert 0 < args.area_constraint <= 1, f"Area constraint must be between 0 and 1, got {args.area_constraint}"
        assert args.hw_scale_factor > 0, f"Hardware scale factor must be positive, got {args.hw_scale_factor}"
        assert args.hw_scale_variance > 0, f"Hardware scale variance must be positive, got {args.hw_scale_variance}"
        assert args.comm_scale_factor > 0, f"Communication scale factor must be positive, got {args.comm_scale_factor}"
        
        if args.hw_scale_factor > 1:
            logger.warning(f"Hardware scale factor is greater than 1 ({args.hw_scale_factor}). This might lead to unexpected behavior.")
            
        logger.info("All input arguments validated successfully")
        
    except AssertionError as e:
        logger.error(f"Argument validation failed: {e}")
        sys.exit(1)

def log_results_summary(args, results, logger):
    """
    Log a comprehensive summary of all results.
    
    Args:
        args: Input arguments
        results: Dictionary containing all results
        logger: Logger instance
    """
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    logger.info(f"Graph: {args.graph_file} (N={results['N']} nodes)")
    logger.info(f"Environment parameters:")
    logger.info(f"  - Hardware scale factor (k): {args.hw_scale_factor}")
    logger.info(f"  - Hardware scale variance (l): {args.hw_scale_variance}")
    logger.info(f"  - Communication scale factor (μ): {args.comm_scale_factor}")
    logger.info(f"  - Area constraint: {args.area_constraint}")
    
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
    args = parse_arguments()
    
    # Create log filename based on parameters
    log_filename = create_log_filename(args)
    
    # Setup logging with the generated filename
    logger = setup_logging(args.verbose, log_filename, args.log_dir)
    
    # Log input parameters
    log_input_parameters(args, logger)
    
    # Validate arguments
    validate_arguments(args, logger)
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    try:
        # Initialize Task Graph
        logger.info(f"Loading graph from {args.graph_file}")
        TG = TaskGraph(area_constraint=args.area_constraint)
        TG.load_graph_from_pydot(
            args.graph_file,
            k=args.hw_scale_factor,
            l=args.hw_scale_variance,
            mu=args.comm_scale_factor,
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
                n_particles=100, verbose=(args.verbose >= 2)
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
        log_results_summary(args, results, logger)
        
        # Save results to CSV
        formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_summary_data = [{
            'SimTime': formatted_time,
            'LogFile': log_filename,  # Include log filename in CSV
            'GraphName': args.graph_file,
            'N': N,
            'HW_Scale_Factor': args.hw_scale_factor,
            'HW_Scale_Var': args.hw_scale_variance,
            'Comm_Scale_Var': args.comm_scale_factor,
            'Area_Percentage': args.area_constraint,
            'Seed': args.seed,
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
        logger.info(f"Detailed log saved to {os.path.join(args.log_dir, log_filename)}")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        logger.info("="*80)
        logger.info("TASK GRAPH PARTITIONING EVALUATION - LOG END")
        logger.info("="*80)

if __name__ == "__main__":
    main()