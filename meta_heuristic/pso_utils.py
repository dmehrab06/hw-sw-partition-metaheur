import numpy as np
import pyswarms.backend as P
import pyswarms as ps
from pyswarms.backend.topology import Star
import logging

def simulate_vanilla_PSO(dim, c1, c2, w, func_to_optimize, iterations=100, n_particles=500, verbose=True, reproduce=True):
    """
    Simulate Particle Swarm Optimization to minimize a given objective function.
    
    This function implements the standard PSO algorithm with personal best and global best
    updates using a star topology for information sharing between particles.
    
    Args:
        dim (int): Dimension of each particle (number of decision variables)
        c1 (float): Cognitive component coefficient (attraction to personal best)
        c2 (float): Social component coefficient (attraction to global best)
        w (float): Inertia weight (momentum factor for velocity updates)
        func_to_optimize (callable): Objective function to minimize
            Should accept array of shape (n_particles, dim) and return array of shape (n_particles,)
        iterations (int, optional): Number of PSO iterations to run (default: 100)
        n_particles (int, optional): Number of particles in the swarm (default: 500)
        verbose (bool, optional): If True, print progress information (default: True)
        reproduce (bool, optional): If True, set random seed for reproducibility (default: True)
    
    Returns:
        tuple: (best_cost, best_position) where:
            - best_cost (float): Best objective function value found
            - best_position (np.ndarray): Best particle position of shape (dim,)
    
    Algorithm Details:
        1. Initialize swarm with random positions and velocities
        2. For each iteration:
           a. Evaluate objective function for all particles
           b. Update personal bests for each particle
           c. Update global best across all particles
           d. Update velocities using PSO velocity equation
           e. Update positions by adding velocities
        3. Return the best solution found
    
    Note:
        The PSO velocity update equation is:
        v_new = w * v_old + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
        where r1 and r2 are random numbers between 0 and 1.
    """
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    
    # Set up PSO components
    my_topology = Star()  # Star topology for global best computation
    my_options = {'c1': c1, 'c2': c2, 'w': w}
    
    my_swarm = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=my_options)

    if verbose:
        logger.info(f'Starting PSO with {n_particles} particles for {iterations} iterations')
        logger.debug(f'PSO parameters: c1={c1}, c2={c2}, w={w}')
    
    best_cost, best_pos = my_swarm.optimize(func_to_optimize, iters=iterations,verbose=True)

    if verbose:
        logger.info(f'PSO completed. Best cost: {best_cost:.4f}')
    
    return best_cost, best_pos

def simulate_BPSO(dim, c1, c2, w, func_to_optimize, iterations=100, n_particles=500, verbose=True, reproduce=True):
    """
    Simulate Binary Particle Swarm Optimization to minimize a given objective function.
    
    This function implements the Discrete PSO algorithm with personal best and global best
    updates using a star topology for information sharing between particles.
    
    Args:
        dim (int): Dimension of each particle (number of decision variables)
        c1 (float): Cognitive component coefficient (attraction to personal best)
        c2 (float): Social component coefficient (attraction to global best)
        w (float): Inertia weight (momentum factor for velocity updates)
        func_to_optimize (callable): Objective function to minimize
            Should accept array of shape (n_particles, dim) and return array of shape (n_particles,)
        iterations (int, optional): Number of PSO iterations to run (default: 100)
        n_particles (int, optional): Number of particles in the swarm (default: 500)
        verbose (bool, optional): If True, print progress information (default: True)
        reproduce (bool, optional): If True, set random seed for reproducibility (default: True)
    
    Returns:
        tuple: (best_cost, best_position) where:
            - best_cost (float): Best objective function value found
            - best_position (np.ndarray): Best particle position of shape (dim,)
    """
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('__main__')
    
    # Set up PSO components
    #my_topology = Star()  # Star topology for global best computation
    my_options = {'c1': c1, 'c2': c2, 'w': w, 'k':4, 'p':2}
    
    my_swarm = ps.discrete.binary.BinaryPSO(n_particles=n_particles, dimensions=dim, options=my_options)

    if verbose:
        logger.info(f'Starting DBPSO with {n_particles} particles for {iterations} iterations')
        logger.debug(f'DBPSO parameters: c1={c1}, c2={c2}, w={w}, k = 4, p = 2')
    
    best_cost, best_pos = my_swarm.optimize(func_to_optimize, iters=iterations,verbose=True)

    if verbose:
        logger.info(f'DBPSO completed. Best cost: {best_cost:.4f}')
    
    return best_cost, best_pos


def random_assignment(dim, func_to_optimize, num_samples=5000, random_prob=0.5, reproduce=True):
    """
    Generate random binary assignments and find the best solution.
    
    This function serves as a baseline comparison method that generates random
    binary solutions using Bernoulli distribution and evaluates them using
    the provided objective function.
    
    Args:
        dim (int): Dimension of each solution vector (number of decision variables)
        func_to_optimize (callable): Objective function to minimize
            Should accept array of shape (num_samples, dim) and return array of shape (num_samples,)
        num_samples (int, optional): Number of random solutions to generate (default: 5000)
        random_prob (float, optional): Probability parameter for Bernoulli distribution (default: 0.5)
            Controls the bias towards 0 or 1 in random assignments
        reproduce (bool, optional): If True, set random seed for reproducibility (default: True)
    
    Returns:
        tuple: (best_cost, best_solution) where:
            - best_cost (float): Best objective function value found
            - best_solution (np.ndarray): Best solution vector of shape (dim,)
    
    Algorithm:
        1. Generate num_samples random binary vectors using Bernoulli(random_prob)
        2. Evaluate all solutions using the objective function
        3. Return the solution with minimum cost
    
    Note:
        This method provides a simple baseline for comparison with more sophisticated
        optimization algorithms. The random_prob parameter can be tuned based on
        problem characteristics (e.g., expected ratio of hardware vs software assignments).
    """
    logger = logging.getLogger(__name__)
    
    # Generate random binary solutions
    all_samples = []
    for i in range(num_samples):
        bernoulli_samples = np.random.binomial(n=1, p=random_prob, size=dim)
        all_samples.append(bernoulli_samples)
    
    # Convert to numpy array and evaluate all solutions
    sample_array = np.array(all_samples)
    all_costs = func_to_optimize(sample_array)
    
    # Find best solution
    best_cost = np.min(all_costs)
    min_index = np.argmin(all_costs)
    best_solution = all_samples[min_index]
    
    logger.info(f'Random assignment completed. Best cost: {best_cost:.4f} from {num_samples} samples')
    
    return best_cost, best_solution


# Benchmark functions for testing optimization algorithms
def banana(swarms):
    """
    Rosenbrock's banana function (also known as Rosenbrock function).
    
    This is a classic optimization benchmark function with a global minimum
    at (1, 1) with function value 0. The function has a narrow curved valley
    that makes it challenging for optimization algorithms.
    
    Args:
        swarms (np.ndarray): Array of shape (n_particles, 2) containing 2D points
    
    Returns:
        np.ndarray: Function values of shape (n_particles,)
    
    Mathematical form:
        f(x, y) = (1-x)² + 100(y-x²)²
    
    Properties:
        - Global minimum: f(1, 1) = 0
        - Search domain: typically [-5, 5] × [-5, 5]
        - Characteristics: Non-convex, narrow curved valley
    """
    assert swarms.shape[1] == 2, f"Banana function requires 2D input, got {swarms.shape[1]}D"
    
    x = swarms[:, 0]
    y = swarms[:, 1]
    
    return (1 - x)**2 + 100 * (y - x*x)**2


def beale(swarms):
    """
    Beale's function - a multimodal optimization benchmark.
    
    This function has a global minimum and is commonly used to test
    optimization algorithms' ability to find global optima.
    
    Args:
        swarms (np.ndarray): Array of shape (n_particles, 2) containing 2D points
    
    Returns:
        np.ndarray: Function values of shape (n_particles,)
    
    Mathematical form:
        f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    
    Properties:
        - Global minimum: f(3, 0.5) = 0
        - Search domain: typically [-4.5, 4.5] × [-4.5, 4.5]
        - Characteristics: Multimodal with several local minima
    """
    assert swarms.shape[1] == 2, f"Beale function requires 2D input, got {swarms.shape[1]}D"
    
    x = swarms[:, 0]
    y = swarms[:, 1]
    
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y*y)**2
    term3 = (2.625 - x + x*y*y*y)**2
    
    return term1 + term2 + term3
