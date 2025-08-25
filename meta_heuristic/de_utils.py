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

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/de_utils.log")

logger = LogManager.get_logger(__name__)

def simulate_JADE(dim, func_to_optimize, config):
    
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    
    problem = {'fitness_function': func_to_optimize, 'ndim_problem': dim, 
               'lower_boundary': 0.0 * np.ones((dim,)), 'upper_boundary': 1.0 * np.ones((dim,))}
    
    options = {'max_function_evaluations': config['jade']['iter'], 'seed_rng': 2022,'n_individuals':config['jade']['n_individuals']}
    
    model = JADE(problem, options)  # initialize the optimizer class
    results = model.optimize()  # run the optimization process

    return results['best_so_far_y'],results['best_so_far_x']

def simulate_SHADE(dim, func_to_optimize, config):
    
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    
    problem = {'fitness_function': func_to_optimize, 'ndim_problem': dim, 
               'lower_boundary': 0.0 * np.ones((dim,)), 'upper_boundary': 1.0 * np.ones((dim,))}
    
    options = {'max_function_evaluations': config['shade']['iter'], 'seed_rng': 2022,'n_individuals':config['shade']['n_individuals']}
    
    model = SHADE(problem, options)  # initialize the optimizer class
    results = model.optimize()  # run the optimization process

    return results['best_so_far_y'],results['best_so_far_x']
