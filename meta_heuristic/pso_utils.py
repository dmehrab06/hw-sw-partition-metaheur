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
    
    # Set up PSO components
    my_options = {
        'c1': config['pso']['c1'], 
        'c2': config['pso']['c2'], 
        'w': config['pso']['w']
    }
    
    my_swarm = ps.single.GlobalBestPSO(
        n_particles=config['pso']['n_particles'], 
        dimensions=dim, 
        options=my_options
    )

    if config["pso"]["verbose"]:
        logger.info(f'Starting PSO with {config["pso"]["n_particles"]} particles for {config["pso"]["iterations"]} iterations')
        logger.debug(f'PSO parameters: c1={config["pso"]["c1"]}, c2={config["pso"]["c2"]}, w={config["pso"]["w"]}')
    
    best_cost, best_pos = my_swarm.optimize(
        func_to_optimize, 
        iters=config['pso']['iterations'], 
        verbose=config['pso']['verbose']
    )

    if config["pso"]["verbose"]:
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
    
    # Set up PSO components
    my_options = {
        'c1': config['dbpso']['c1'], 
        'c2': config['dbpso']['c2'], 
        'w': config['dbpso']['w'], 
        'k': config['dbpso']['k'], 
        'p': config['dbpso']['p']
    }
    
    my_swarm = ps.discrete.BinaryPSO(
        n_particles=config['dbpso']['n_particles'], 
        dimensions=dim, 
        options=my_options
    )

    if config['dbpso']['verbose']:
        logger.info(f'Starting DBPSO with {config["dbpso"]["n_particles"]} particles for {config["dbpso"]["iterations"]} iterations')
        logger.debug(f'DBPSO parameters: c1={my_options["c1"]}, c2={my_options["c2"]}, w={my_options["w"]}, k = {my_options["k"]}, p = {my_options["p"]}')
    
    best_cost, best_pos = my_swarm.optimize(
        func_to_optimize, 
        iters=config['dbpso']['iterations'], 
        verbose=config['dbpso']['verbose']
    )

    if config['dbpso']['verbose']:
        logger.info(f'DBPSO completed. Best cost: {best_cost:.4f}')
    
    return best_cost, best_pos

def simulate_CLPSO(dim, func_to_optimize, config):
    """
    Simulate Comprehensive Learning PSO using configuration.
    """
    logger = logging.getLogger('__main__')
    
    problem = {
        'fitness_function': func_to_optimize, 
        'ndim_problem': dim, 
        'lower_boundary': 0.0 * np.ones((dim,)), 
        'upper_boundary': 1.0 * np.ones((dim,))
    }
    
    options = {
        'max_function_evaluations': config['clpso']['iterations'], 
        'seed_rng': config.get('seed', 2022),
        'n_individuals': config['clpso']['n_individuals'],
        'c': config['clpso']['c']
    }
    
    model = CLPSO(problem, options)
    results = model.optimize()

    return results['best_so_far_y'], results['best_so_far_x']

def simulate_CCPSO(dim, func_to_optimize, config):
    """
    Simulate Cooperative Coevolutionary PSO using configuration.
    """
    logger = logging.getLogger('__main__')
    
    problem = {
        'fitness_function': func_to_optimize, 
        'ndim_problem': dim, 
        'lower_boundary': 0.0 * np.ones((dim,)), 
        'upper_boundary': 1.0 * np.ones((dim,))
    }
    
    options = {
        'max_function_evaluations': config['ccpso']['iterations'], 
        'seed_rng': config.get('seed', 2022),
        'n_individuals': max(500, config['ccpso']['n_individuals']),
        'c': config['ccpso']['c'],
        'group_sizes': config['ccpso']['group_sizes']
    }
    
    model = CCPSO2(problem, options)
    results = model.optimize()

    return results['best_so_far_y'], results['best_so_far_x']