import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import combinations
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.graph_generator import GraphGenerator

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/sdp_relax.log")

logger = LogManager.get_logger(__name__)

def create_random_graph(N, edge_probability=0.3, seed=42):
    """Create a random undirected graph with node and edge attributes."""
    np.random.seed(seed)
    
    # Create random graph
    G = nx.gnp_random_graph(N, edge_probability, seed=seed)
    
    # Assign hardware costs
    for node in G.nodes():
        G.nodes[node]['hardware_cost'] = np.random.uniform(1, 10)
        G.nodes[node]['software_cost'] = np.random.uniform(1, 10)
    
    # Assign communication costs to edges
    for u, v in G.edges():
        G[u][v]['communication_cost'] = np.random.uniform(0.5, 5)
    
    return G

def extract_costs(G):
    """Extract costs from graph into vectors and matrices."""
    N = G.number_of_nodes()
    
    # Extract hardware and software costs
    h = np.zeros(N)
    s = np.zeros(N)
    for node in G.nodes():
        h[node] = G.nodes[node]['hardware_cost']
        s[node] = G.nodes[node]['software_cost']
    
    # Create edge weighted adjacency matrix C
    C = np.zeros((N, N))
    for u, v in G.edges():
        comm_cost = G[u][v]['communication_cost']
        C[u, v] = comm_cost
        C[v, u] = comm_cost  # Since the graph is undirected
    
    return h, s, C

def solve_sdp_relaxation(h, s, C):
    """Solve SDP relaxation of the problem (continuous relaxation)."""
    N = len(h)
    ones = np.ones(N)
    d = s - C @ ones  # Calculate d = s - C1
    
    # Define variables
    X = cp.Variable((N+1, N+1), symmetric=True)
    
    # PSD constraint on X
    constraints = [X >> 0]
    
    # X[N,N] = 1 (homogenization)
    constraints.append(X[N, N] == 1)
    
    # Express d^Tx - Trace(CX) <= 0 in terms of X
    C_padded = np.zeros((N+1, N+1))
    C_padded[:N, :N] = C
    
    d_padded = np.zeros(N+1)
    d_padded[:N] = d
    d_padded[N] = 0
    
    # Linear objective in terms of X
    h_padded = np.zeros(N+1)
    h_padded[:N] = h
    h_padded[N] = 0  # constant term
    
    # Objective: minimize h^T(1-x)
    # This translates to minimizing sum(h) - h^Tx
    # In homogeneous form, this is constant - trace(H*X)
    H = np.zeros((N+1, N+1))
    for i in range(N):
        H[i, N] = h[i] / 2
        H[N, i] = h[i] / 2
    
    objective = cp.Minimize(np.sum(h) - cp.trace(H @ X))
    
    # The constraint d^Tx - Trace(CX) <= 0
    D = np.zeros((N+1, N+1))
    for i in range(N):
        D[i, N] = d[i] / 2
        D[N, i] = d[i] / 2
    
    constraints.append(cp.trace(D @ X) - cp.trace(C_padded @ X) <= 0)
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS)
        X_val = X.value
        
        # Extract x from the solution
        x_relaxed = np.zeros(N)
        for i in range(N):
            x_relaxed[i] = X_val[i, N] / X_val[N, N]
        
        return {
            'status': problem.status,
            'objective_value': problem.value,
            'X': X_val,
            'x_relaxed': x_relaxed
        }
    except Exception as e:
        logger.error(f"SDP solve failed: {e}")
        return None

def random_hyperplane_rounding(X_val, N, num_samples=100):
    """
    Apply random hyperplane rounding to get a binary solution from SDP relaxation.
    This is a standard technique for SDP relaxation of binary problems.
    """
    best_x = None
    best_score = float('inf')
    
    # Extract the relevant part of X
    V = X_val[:N, :N]
    
    # Try to find a Cholesky factorization to get vectors
    try:
        # If X is positive definite
        L = np.linalg.cholesky(V + 1e-10 * np.eye(N))
    except np.linalg.LinAlgError:
        # If X is only positive semidefinite, use eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(V)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
    
    # Perform random hyperplane rounding
    for _ in range(num_samples):
        # Generate random hyperplane
        r = np.random.normal(0, 1, L.shape[1])
        
        # Project vectors onto the random direction
        projections = L @ r
        
        # Round based on sign
        x_rounded = (projections > 0).astype(int)
        
        # We'll evaluate these later to pick the best one
        if best_x is None:
            best_x = x_rounded
        
    return best_x

def evaluate_solution(h, s, C, x):
    """Evaluate the objective value and constraints for a given solution x."""
    N = len(h)
    ones = np.ones(N)
    d = s - C @ ones
    
    # Calculate hardware/software cost
    hw_sw_cost = h @ (ones - x) + s @ x
    
    # Calculate communication cost
    comm_cost = 0
    for i in range(N):
        for j in range(i+1, N):
            if C[i, j] > 0:  # If there's an edge
                if abs(x[i] - x[j]) > 0.1:  # Different partitions
                    comm_cost += C[i, j]
    
    # Check constraint satisfaction
    constraint_val = d @ x
    for i in range(N):
        for j in range(N):
            constraint_val -= C[i, j] * x[i] * x[j]
    
    return {
        'hw_sw_cost': hw_sw_cost,
        'comm_cost': comm_cost,
        'total_cost': hw_sw_cost + comm_cost,
        'constraint_value': constraint_val,
        'constraint_satisfied': constraint_val <= 1e-6
    }

def local_search(h, s, C, x_init):
    """Perform local search to improve the solution."""
    x = x_init.copy()
    N = len(x)
    
    improved = True
    while improved:
        improved = False
        current_eval = evaluate_solution(h, s, C, x)
        
        # Try flipping each bit
        for i in range(N):
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]
            
            # Evaluate new solution
            new_eval = evaluate_solution(h, s, C, x_new)
            
            # Accept if better and satisfies constraint
            if (new_eval['total_cost'] < current_eval['total_cost'] and 
                new_eval['constraint_satisfied']):
                x = x_new
                improved = True
                break
    
    return x

def visualize_solution(G, x):
    """Visualize the graph with node partitioning."""
    pos = nx.spring_layout(G, seed=42)
    
    # Node colors based on partition (hardware=red, software=blue)
    node_colors = ['red' if xi > 0.5 else 'blue' for xi in x]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=700, font_weight='bold', font_color='white', ax=ax)
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['communication_cost']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Add node labels with costs
    node_labels = {}
    for node in G.nodes():
        h_cost = G.nodes[node]['hardware_cost']
        s_cost = G.nodes[node]['software_cost']
        node_labels[node] = f"H:{h_cost:.1f}, S:{s_cost:.1f}"
    
    # Position the node cost labels slightly below the nodes
    pos_labels = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos_labels, labels=node_labels, font_size=8, ax=ax)
    
    ax.set_title("Node Partitioning Solution\nRed: Hardware Nodes, Blue: Software Nodes")
    plt.axis('off')
    fig.savefig("results/node_partitioning_solution.png", bbox_inches='tight')

def main():
    
    graph_generator = GraphGenerator()
    G = graph_generator.load_graph("data/example_graph.pkl")

    # Extract costs
    hw_costs, sw_costs, comm_matrix = extract_costs(G)
    
    logger.info(f"Hardware costs: {hw_costs}")
    logger.info(f"Software costs: {sw_costs}")
    logger.info("Communication cost matrix:")
    logger.info(comm_matrix)
    
    # Step 1: Solve SDP relaxation
    logger.info("\nSolving SDP relaxation...")
    sdp_result = solve_sdp_relaxation(hw_costs, sw_costs, comm_matrix)
    
    if sdp_result is None:
        logger.info("Failed to solve SDP relaxation")
        return
        
    logger.info(f"SDP relaxation status: {sdp_result['status']}")
    logger.info(f"SDP relaxation objective: {sdp_result['objective_value']}")
    
    # Check rank of X to see if relaxation is tight
    N = G.number_of_nodes()
    X_val = sdp_result['X'][:N, :N]
    eigvals = np.linalg.eigvalsh(X_val)
    rank_approx = sum(e > 1e-6 for e in eigvals)
    logger.info(f"Approximate rank of X: {rank_approx}")
    
    # Print relaxed solution
    x_relaxed = sdp_result['x_relaxed']
    logger.info(f"Relaxed solution (continuous): {x_relaxed}")
    
    # Step 2: Random hyperplane rounding
    logger.info("\nApplying random hyperplane rounding...")
    x_rounded = random_hyperplane_rounding(sdp_result['X'], N)
    logger.info(f"Rounded solution (binary): {x_rounded}")
    
    # Step 3: Local search to improve the solution
    logger.info("\nApplying local search...")
    x_final = local_search(hw_costs, sw_costs, comm_matrix, x_rounded)
    logger.info(f"Final solution: {x_final}")
    
    # Evaluate final solution
    final_eval = evaluate_solution(hw_costs, sw_costs, comm_matrix, x_final)
    logger.info("\nFinal solution evaluation:")
    logger.info(f"Hardware/Software cost: {final_eval['hw_sw_cost']:.2f}")
    logger.info(f"Communication cost: {final_eval['comm_cost']:.2f}")
    logger.info(f"Total cost: {final_eval['total_cost']:.2f}")
    logger.info(f"Constraint value: {final_eval['constraint_value']:.2e}")
    logger.info(f"Constraint satisfied: {final_eval['constraint_satisfied']}")
    
    logger.info("\nNode assignments:")
    for i in range(N):
        node_type = "Software" if x_final[i] > 0.5 else "Hardware"
        hw_cost = hw_costs[i]
        sw_cost = sw_costs[i]
        logger.info(f"Node {i}: {node_type} (HW cost: {hw_cost:.2f}, SW cost: {sw_cost:.2f})")
    
    # Visualize the solution
    visualize_solution(G, x_final)

if __name__ == "__main__":
    main()