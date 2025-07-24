import numpy as np
import pyswarms as ps
from pyswarms.backend.topology import Star

from .task_graph import TaskGraph
from .parser_utils import parse_arguments
from .pso_utils import simulate_PSO, simulate_DBPSO, simulate_CLPSO, simulate_CCPSO
from .ga_utils import simulate_GL25
from .sa_utils import simulate_ESA
from .de_utils import simulate_SHADE, simulate_JADE

def random_assignment(dim, func_to_optimize, config):
    
    all_samples = []
    for i in range(config['random']['num_samples']):
        bernoulli_samples = np.random.binomial(n=1, p=config['random']['p'], size=dim)
        all_samples.append(bernoulli_samples)
    
    # Convert to numpy array and evaluate all solutions
    sample_array = np.array(all_samples)
    all_costs = func_to_optimize(sample_array)
    
    # Find best solution
    best_cost = np.min(all_costs)
    min_index = np.argmin(all_costs)
    best_solution = all_samples[min_index]
    
    return best_cost, best_solution


__all__ = [
    'TaskGraph', 
    'parse_arguments', 
    'simulate_PSO', 
    'simulate_DBPSO',
    'simulate_CLPSO',
    'simulate_CCPSO',
    'random_assignment',
    'simulate_GL25',
    'simulate_ESA',
    'simulate_SHADE',
    'simulate_JADE'
]