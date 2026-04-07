import numpy as np
from pypop7.optimizers.de.shade import SHADE
from pypop7.optimizers.de.jade import JADE
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
    LogManager.initialize("logs/de_utils.log")

logger = LogManager.get_logger(__name__)

def simulate_JADE(dim, func_to_optimize, config):
    
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    jade_cfg = resolve_classical_method_config(config, "jade")
    
    problem = {'fitness_function': func_to_optimize, 'ndim_problem': dim, 
               'lower_boundary': 0.0 * np.ones((dim,)), 'upper_boundary': 1.0 * np.ones((dim,))}
    
    options = {
        'max_function_evaluations': int(jade_cfg['iter']),
        'seed_rng': jade_cfg.get('seed_rng', 2022),
        'n_individuals': int(jade_cfg['n_individuals']),
    }
    
    model = JADE(problem, options)  # initialize the optimizer class
    results = model.optimize()  # run the optimization process

    return results['best_so_far_y'],results['best_so_far_x']

def simulate_SHADE(dim, func_to_optimize, config):
    
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    shade_cfg = resolve_classical_method_config(config, "shade")
    
    problem = {'fitness_function': func_to_optimize, 'ndim_problem': dim, 
               'lower_boundary': 0.0 * np.ones((dim,)), 'upper_boundary': 1.0 * np.ones((dim,))}
    
    options = {
        'max_function_evaluations': int(shade_cfg['iter']),
        'seed_rng': shade_cfg.get('seed_rng', 2022),
        'n_individuals': int(shade_cfg['n_individuals']),
    }
    
    model = SHADE(problem, options)  # initialize the optimizer class
    results = model.optimize()  # run the optimization process

    return results['best_so_far_y'],results['best_so_far_x']
