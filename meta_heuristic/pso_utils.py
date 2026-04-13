import numpy as np
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.topology import Star
from pypop7.optimizers.pso.clpso import CLPSO
from pypop7.optimizers.pso.ccpso2 import CCPSO2
import os, sys
import logging

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
try:
    from .Configuration import resolve_classical_method_config
except ImportError:
    from Configuration import resolve_classical_method_config

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/pso_utils.log")

logger = LogManager.get_logger(__name__)

def simulate_PSO(dim, func_to_optimize, config):
    """
    Simulate Particle Swarm Optimization using configuration.
    
    Args:
        dim: Problem dimension
        func_to_optimize: Objective function
        config: Configuration object containing PSO parameters
        verbose: Whether to log verbose output
    
    Returns:
        tuple: (best_cost, best_position)
    """
    logger = logging.getLogger('__main__')
    pso_cfg = resolve_classical_method_config(config, "pso")
    
    # Set up PSO components
    my_options = {
        'c1': pso_cfg['c1'], 
        'c2': pso_cfg['c2'], 
        'w': pso_cfg['w']
    }
    
    my_swarm = ps.single.GlobalBestPSO(
        n_particles=int(pso_cfg['n_particles']), 
        dimensions=dim, 
        options=my_options
    )

    if pso_cfg["verbose"]:
        logger.info(f'Starting PSO with {pso_cfg["n_particles"]} particles for {pso_cfg["iterations"]} iterations')
        logger.debug(f'PSO parameters: c1={pso_cfg["c1"]}, c2={pso_cfg["c2"]}, w={pso_cfg["w"]}')
    
    best_cost, best_pos = my_swarm.optimize(
        func_to_optimize, 
        iters=int(pso_cfg['iterations']), 
        verbose=bool(pso_cfg['verbose'])
    )

    if pso_cfg["verbose"]:
        logger.info(f'PSO completed. Best cost: {best_cost:.4f}')
    
    return best_cost, best_pos

def simulate_DBPSO(dim, func_to_optimize, config):
    """
    Simulate Discrete Binary PSO using configuration.
    
    Args:
        dim: Problem dimension
        func_to_optimize: Objective function
        config: Configuration object containing DBPSO parameters
        verbose: Whether to log verbose output
    
    Returns:
        tuple: (best_cost, best_position)
    """
    logger = logging.getLogger('__main__')
    dbpso_cfg = resolve_classical_method_config(config, "dbpso")
    
    # Set up PSO components
    my_options = {
        'c1': dbpso_cfg['c1'], 
        'c2': dbpso_cfg['c2'], 
        'w': dbpso_cfg['w'], 
        'k': dbpso_cfg['k'], 
        'p': dbpso_cfg['p']
    }
    
    my_swarm = ps.discrete.BinaryPSO(
        n_particles=int(dbpso_cfg['n_particles']), 
        dimensions=dim, 
        options=my_options
    )

    if dbpso_cfg['verbose']:
        logger.info(f'Starting DBPSO with {dbpso_cfg["n_particles"]} particles for {dbpso_cfg["iterations"]} iterations')
        logger.debug(f'DBPSO parameters: c1={my_options["c1"]}, c2={my_options["c2"]}, w={my_options["w"]}, k = {my_options["k"]}, p = {my_options["p"]}')
    
    best_cost, best_pos = my_swarm.optimize(
        func_to_optimize, 
        iters=int(dbpso_cfg['iterations']), 
        verbose=bool(dbpso_cfg['verbose'])
    )

    if dbpso_cfg['verbose']:
        logger.info(f'DBPSO completed. Best cost: {best_cost:.4f}')
    
    return best_cost, best_pos

def simulate_CLPSO(dim, func_to_optimize, config):
    """
    Simulate Comprehensive Learning PSO using configuration.
    """
    logger = logging.getLogger('__main__')
    clpso_cfg = resolve_classical_method_config(config, "clpso")
    
    problem = {
        'fitness_function': func_to_optimize, 
        'ndim_problem': dim, 
        'lower_boundary': 0.0 * np.ones((dim,)), 
        'upper_boundary': 1.0 * np.ones((dim,))
    }
    
    options = {
        'max_function_evaluations': int(clpso_cfg['iterations']), 
        'seed_rng': (config.get('seed', 2022) if clpso_cfg.get('seed_rng') is None else clpso_cfg.get('seed_rng')),
        'n_individuals': int(clpso_cfg['n_individuals']),
        'c': clpso_cfg['c']
    }
    
    model = CLPSO(problem, options)
    results = model.optimize()

    return results['best_so_far_y'], results['best_so_far_x']

def simulate_CCPSO(dim, func_to_optimize, config):
    """
    Simulate Cooperative Coevolutionary PSO using configuration.
    """
    logger = logging.getLogger('__main__')
    ccpso_cfg = resolve_classical_method_config(config, "ccpso")
    raw_group_sizes = list(ccpso_cfg.get("group_sizes", [5, 10, 20]))
    valid_group_sizes = [int(size) for size in raw_group_sizes if int(size) <= int(dim)]
    if not valid_group_sizes:
        valid_group_sizes = [max(1, min(int(dim), 5))]
    
    problem = {
        'fitness_function': func_to_optimize, 
        'ndim_problem': dim, 
        'lower_boundary': 0.0 * np.ones((dim,)), 
        'upper_boundary': 1.0 * np.ones((dim,))
    }
    
    options = {
        'max_function_evaluations': int(ccpso_cfg['iterations']), 
        'seed_rng': (config.get('seed', 2022) if ccpso_cfg.get('seed_rng') is None else ccpso_cfg.get('seed_rng')),
        'n_individuals': max(500, int(ccpso_cfg['n_individuals'])),
        'c': ccpso_cfg['c'],
        'group_sizes': valid_group_sizes,
    }
    
    model = CCPSO2(problem, options)
    results = model.optimize()

    return results['best_so_far_y'], results['best_so_far_x']
