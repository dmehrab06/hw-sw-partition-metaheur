"""
Module to solve mixed integer semidefinite programming (MISDP) problems
for optimal node partitioning in graphs.
"""

import os
import sys
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import time

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/optimizer.log")

logger = LogManager.get_logger(__name__)


class MISDPSolver:
    """
    Class to solve the mixed integer semidefinite programming problem for node partitioning.
    """
    
    def __init__(self):
        """Initialize the solver."""
        self.graph = None
        self.h = None  # Hardware costs
        self.s = None  # Software costs
        self.C = None  # Communication costs matrix
        self.d = None  # d = s - C1
        self.x = None  # Solution vector
        self.X = None  # Solution matrix
        logger.info("MISDPSolver initialized")
    
    def setup_from_graph(self, graph):
        """
        Set up the optimization problem from an input graph.
        
        Args:
            graph (nx.Graph): NetworkX graph with required node and edge attributes
                - Each node should have 'hardware_cost' and 'software_cost' attributes
                - Each edge should have 'communication_cost' attribute
                
        Returns:
            tuple: (h, s, C, d) vectors and matrices for the optimization
        """
        logger.info(f"Setting up optimization from graph with {graph.number_of_nodes()} nodes")
        
        self.graph = graph
        num_nodes = graph.number_of_nodes()
        
        # Validate required attributes
        for node in graph.nodes():
            if 'hardware_cost' not in graph.nodes[node] or 'software_cost' not in graph.nodes[node]:
                logger.error(f"Node {node} missing required cost attributes")
                raise ValueError(f"Node {node} missing required cost attributes")
        
        for u, v in graph.edges():
            if 'communication_cost' not in graph.edges[u, v]:
                logger.error(f"Edge ({u}, {v}) missing communication_cost attribute")
                raise ValueError(f"Edge ({u}, {v}) missing communication_cost attribute")
        
        # Extract hardware and software costs
        self.h = np.zeros(num_nodes)
        self.s = np.zeros(num_nodes)
        
        for node in graph.nodes():
            self.h[node] = graph.nodes[node]['hardware_cost']
            self.s[node] = graph.nodes[node]['software_cost']
        
        # Extract communication costs
        self.C = np.zeros((num_nodes, num_nodes))
        for u, v, data in graph.edges(data=True):
            self.C[u, v] = data['communication_cost']
            self.C[v, u] = data['communication_cost']  # Ensure symmetry
        
        # Calculate d = s - C1
        ones = np.ones(num_nodes)
        self.d = self.s + self.C @ ones
        logger.info(self.d)
        
        logger.info("Problem setup complete")
        logger.info(f"Hardware costs range: [{np.min(self.h):.2f}, {np.max(self.h):.2f}]")
        logger.info(f"Software costs range: [{np.min(self.s):.2f}, {np.max(self.s):.2f}]")
        logger.info(f"Communication costs range: [{np.min(self.C[self.C > 0]):.2f}, {np.max(self.C):.2f}]")
        
        return self.h, self.s, self.C, self.d
    
    def solve_sdp_relaxation_v1(self, R=0):
        """
        Solve the SDP relaxation of the problem.
        
        Args:
            R (float): Constant for the constraint d^Tx - Trace(CX) <= R
            
        Returns:
            dict: Results of the SDP relaxation
        """
        if self.graph is None:
            logger.error("Cannot solve: No problem loaded")
            raise ValueError("No problem loaded")
            
        N = len(self.h)
        logger.info(f"Solving SDP relaxation for {N} nodes with constraint bound R={R}")
        
        start_time = time.time()
        
        # Define variables for the SDP relaxation with homogenization
        X = cp.Variable((N+1, N+1), symmetric=True)
        
        # Constraints
        constraints = [X >> 0]  # PSD constraint
        constraints.append(X[N, N] == 1)  # Homogenization constraint
        
        # Pad matrices for objective and constraints
        C_padded = np.zeros((N+1, N+1))
        C_padded[:N, :N] = self.C
        
        H = np.zeros((N+1, N+1))
        for i in range(N):
            H[i, N] = self.h[i] / 2
            H[N, i] = self.h[i] / 2
        
        D = np.zeros((N+1, N+1))
        for i in range(N):
            D[i, N] = self.d[i] / 2
            D[N, i] = self.d[i] / 2
        
        # Objective: minimize h^T(1-x) = sum(h) - h^Tx
        objective = cp.Minimize(np.sum(self.h) - cp.trace(H @ X))
        
        # Constraint: d^Tx - Trace(CX) <= R
        constraints.append(cp.trace(D @ X) - cp.trace(C_padded @ X) <= R)
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        
        try:
            logger.info("Attempting to solve SDP with SCS solver")
            problem.solve(solver=cp.SCS, verbose=True, eps=1e-6)
            solve_time = time.time() - start_time
            logger.info(f"SDP solved in {solve_time:.2f} seconds with status: {problem.status}")
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f"SDP solution not optimal: {problem.status}")
                return None
            
            # Extract the solution
            X_val = X.value
            
            # Extract relaxed x values
            x_relaxed = np.zeros(N)
            for i in range(N):
                x_relaxed[i] = X_val[i, N] / X_val[N, N]
            
            # Check rank of X to evaluate tightness of relaxation
            X_sub = X_val[:N, :N]
            eigvals = np.linalg.eigvalsh(X_sub)
            rank_approx = sum(e > 1e-6 for e in eigvals)
            
            logger.info(f"SDP relaxation complete. Approximate rank: {rank_approx}")
            logger.info(f"Relaxed solution: min={min(x_relaxed):.4f}, max={max(x_relaxed):.4f}")
            
            return {
                'status': problem.status,
                'objective_value': problem.value,
                'X': X_val,
                'x_relaxed': x_relaxed,
                'rank': rank_approx,
                'solve_time': solve_time
            }
        except Exception as e:
            logger.error(f"SDP solver failed: {e}")
            return None
        
    def solve_sdp_relaxation(self, R=0):
        """
        Solve the SDP relaxation of the problem.
        
        Args:
            R (float): Constant for the constraint d^Tx - Trace(CX) <= R
            
        Returns:
            dict: Results of the SDP relaxation
        """
        if self.graph is None:
            logger.error("Cannot solve: No problem loaded")
            raise ValueError("No problem loaded")
            
        N = len(self.h)
        logger.info(f"Solving SDP relaxation for {N} nodes with constraint bound R={R}")
        
        start_time = time.time()
        
        # Define variables for the SDP relaxation with homogenization
        X = cp.Variable((N+1, N+1), symmetric=True)
        
        # Constraints
        constraints = [X >> 0]  # PSD constraint
        constraints.append(X[N, N] == 1)  # Homogenization constraint
        
        # 1. Main constraint using S-procedure: d^T x - x^T C x <= R
        # Equivalent to: Tr([C -d/2; -d^T/2 R] X) >= 0
        A_1 = np.zeros((N+1, N+1))
        A_1[0:N, 0:N] = self.C
        A_1[0:N, N] = -0.5 * self.d
        A_1[N, 0:N] = -0.5 * self.d
        A_1[N, N] = R
        constraints.append(cp.trace(A_1 @ X) >= 0)

        # 2. Binary constraints relaxation
    
        # 2.1 Bounds: 0 <= x_i <= 1
        for i in range(N):
            # x_i >= 0
            A_2i = np.zeros((N+1, N+1))
            A_2i[i, N] = A_2i[N, i] = -0.5
            constraints.append(cp.trace(A_2i @ X) <= 0)
            
            # x_i <= 1
            A_3i = np.zeros((N+1, N+1))
            A_3i[i, N] = A_3i[N, i] = 0.5
            A_3i[N, N] = -1
            constraints.append(cp.trace(A_3i @ X) <= 0)
        
        # 2.2 Diagonal constraints: (X_11)_ii = x_i
        for i in range(N):
            A_4i = np.zeros((N+1, N+1))
            A_4i[i, i] = 1
            A_4i[i, N] = A_4i[N, i] = -0.5
            constraints.append(cp.trace(A_4i @ X) == 0)
        
        # 2.3 Correlation constraints: (X_11)_ij <= min(x_i, x_j)
        for i in range(N):
            for j in range(i+1, N):  # Only iterate over upper triangle
                # (X_11)_ij <= x_i
                A_5ij = np.zeros((N+1, N+1))
                A_5ij[i, j] = A_5ij[j, i] = 1  # Symmetric matrix
                A_5ij[i, N] = A_5ij[N, i] = -0.5
                constraints.append(cp.trace(A_5ij @ X) <= 0)
                
                # (X_11)_ij <= x_j
                A_6ij = np.zeros((N+1, N+1))
                A_6ij[i, j] = A_6ij[j, i] = 1  # Symmetric matrix
                A_6ij[j, N] = A_6ij[N, j] = -0.5
                constraints.append(cp.trace(A_6ij @ X) <= 0)
                
                # (X_11)_ij >= 0 (non-negativity of correlations)
                A_7ij = np.zeros((N+1, N+1))
                A_7ij[i, j] = A_7ij[j, i] = -1  # Symmetric matrix
                constraints.append(cp.trace(A_7ij @ X) <= 0)
        
        
        
        # Objective: minimize h^T(1-x) = sum(h) - h^Tx
        H = np.zeros((N+1, N+1))
        for i in range(N):
            H[i, N] = self.h[i] / 2
            H[N, i] = self.h[i] / 2
        objective = cp.Minimize(np.sum(self.h) - cp.trace(H @ X))
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        
        try:
            logger.info("Attempting to solve SDP with SCS solver")
            problem.solve(solver=cp.SCS, verbose=True, eps=1e-6)
            solve_time = time.time() - start_time
            logger.info(f"SDP solved in {solve_time:.2f} seconds with status: {problem.status}")
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f"SDP solution not optimal: {problem.status}")
                return None
            
            # Extract the solution
            X_val = X.value
            
            # Extract relaxed x values
            x_relaxed = np.zeros(N)
            for i in range(N):
                x_relaxed[i] = X_val[i, N] / X_val[N, N]
            
            # Check rank of X to evaluate tightness of relaxation
            X_sub = X_val[:N, :N]
            eigvals = np.linalg.eigvalsh(X_sub)
            rank_approx = sum(e > 1e-6 for e in eigvals)
            
            logger.info(f"SDP relaxation complete. Approximate rank: {rank_approx}")
            logger.info(f"Relaxed x: {x_relaxed}")
            logger.info(f"Relaxed solution: min={min(x_relaxed):.4f}, max={max(x_relaxed):.4f}")
            
            return {
                'status': problem.status,
                'objective_value': problem.value,
                'X': X_val,
                'x_relaxed': x_relaxed,
                'rank': rank_approx,
                'solve_time': solve_time
            }
        except Exception as e:
            logger.error(f"SDP solver failed: {e}")
            return None
    
    def random_hyperplane_rounding(self, X_val, num_samples=100, R=0):
        """
        Apply random hyperplane rounding to get a binary solution.
        
        Args:
            X_val: Solution matrix from SDP
            num_samples (int): Number of samples for rounding
            
        Returns:
            np.array: Binary solution vector
        """
        logger.info(f"Applying random hyperplane rounding with {num_samples} samples")
        N = len(self.h)
        
        best_x = None
        best_score = float('inf')
        best_satisfied = False
        
        # Extract the relevant part of X
        V = X_val[:N, :N]
        
        # Try to find a Cholesky factorization
        try:
            # If X is positive definite
            L = np.linalg.cholesky(V + 1e-10 * np.eye(N))
            logger.info("Applied Cholesky factorization")
        except np.linalg.LinAlgError:
            # If X is only positive semidefinite, use eigendecomposition
            logger.info("Using eigen decomposition")
            eigvals, eigvecs = np.linalg.eigh(V)
            L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        
        # Perform random hyperplane rounding
        for i in range(num_samples):
            # Generate random hyperplane
            r = np.random.normal(0, 1, L.shape[1])
            
            # Project vectors onto the random direction
            projections = L @ r
            
            # Round based on sign
            x_rounded = (projections > 0).astype(int)
            
            # Evaluate the solution
            eval_result = self.evaluate_solution(x_rounded, R=R)

            # Print aspect of each sample
            # logger.info(f"Sample {i+1}: {x_rounded} SW/Comm Cost: {eval_result['sw_comm_cost']}, HW Cost: {eval_result['hw_cost']}, Constraint value: {eval_result['constraint_value']}, Status: {eval_result['constraint_satisfied']}")
            
            # Track the best solution that satisfies constraints
            if eval_result['constraint_satisfied']:
                if not best_satisfied or eval_result['hw_cost'] < best_score:
                    best_x = x_rounded
                    best_score = eval_result['hw_cost']
                    best_satisfied = True
                    logger.info(f"Accepting sample {i+1}: {x_rounded}")
                    logger.info(f"Best score: {best_score}")
            # elif not best_satisfied and (best_x is None or eval_result['hw_cost'] < best_score):
            #     logger.info("Constraint not satisfied, but objective is less than best")
            #     best_x = x_rounded
            #     best_score = eval_result['hw_cost']
        
        if best_x is None:
            logger.warning("No feasible solution found in rounding")
            # Fallback to naive rounding
            x_relaxed = X_val[:N, N]
            best_x = (x_relaxed > 0.5).astype(int)
        else:
            logger.info(f"Best rounded solution found with score: {best_score:.4f}")
            logger.info(f"Constraint satisfied: {best_satisfied}")
        
        return best_x
    
    def local_search(self, x_init, R=0, max_iterations=100):
        """
        Perform local search to improve the solution.
        
        Args:
            x_init: Initial binary solution
            R (float): Constraint bound
            max_iterations (int): Maximum number of iterations
            
        Returns:
            np.array: Improved binary solution
        """
        logger.info("Performing local search optimization")
        x = x_init.copy()
        N = len(x)
        
        # Initial evaluation
        best_eval = self.evaluate_solution(x, R)
        logger.info(f"Initial solution cost: {best_eval['hw_cost']:.4f}")
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try flipping each bit
            for i in range(N):
                x_new = x.copy()
                x_new[i] = 1 - x_new[i]
                
                # Evaluate new solution
                new_eval = self.evaluate_solution(x_new, R)
                
                # Accept if better and satisfies constraint (or if both violate but new is better)
                if (new_eval['constraint_satisfied'] and 
                    (not best_eval['constraint_satisfied'] or 
                     new_eval['hw_cost'] < best_eval['hw_cost'])):
                    x = x_new
                    best_eval = new_eval
                    improved = True
                    logger.info(f"Iteration {iteration}: Improved solution with cost: {best_eval['hw_cost']:.4f}")
                    break
                # elif (not new_eval['constraint_satisfied'] and 
                #       not best_eval['constraint_satisfied'] and 
                #       new_eval['constraint_violation'] < best_eval['constraint_violation']):
                #     x = x_new
                #     best_eval = new_eval
                #     improved = True
                #     logger.info(f"Iteration {iteration}: Improved constraint violation: {best_eval['constraint_violation']:.4e}")
                #     break
        
        logger.info(f"Local search completed after {iteration} iterations")
        logger.info(f"Final solution cost: {best_eval['hw_cost']:.4f}")
        logger.info(f"Constraint satisfied: {best_eval['constraint_satisfied']}")
        
        return x
    
    def evaluate_solution(self, x, R=0):
        """
        Evaluate a solution.
        
        Args:
            x: Binary solution vector (0=hardware, 1=software)
            R (float): Constraint bound
            
        Returns:
            dict: Evaluation metrics
        """
        # Calculate hardware cost (for hardware nodes, x=0)
        hw_cost = sum(self.h[i] * (1 - x[i]) for i in range(len(x)))
        
        # Calculate software cost (for software nodes, x=1)
        sw_cost = sum(self.s[i] * x[i] for i in range(len(x)))
        
        # Total hardware/software cost
        hw_sw_cost = hw_cost + sw_cost
        
        # Calculate communication cost
        comm_cost = 0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if self.C[i, j] > 0:  # If there's an edge
                    if abs(x[i] - x[j]) > 0.1:  # Different partitions
                        comm_cost += self.C[i, j]
        
        # Check constraint satisfaction
        X_implicit = np.outer(x, x)
        constraint_val = self.d @ x - np.trace(self.C @ X_implicit) - R
        
        return {
            'hw_cost': hw_cost,
            'sw_cost': sw_cost,
            'hw_sw_cost': hw_sw_cost,
            'sw_comm_cost':comm_cost+sw_cost,
            'comm_cost': comm_cost,
            'total_cost': hw_sw_cost + comm_cost,
            'constraint_value': constraint_val,
            'constraint_satisfied': constraint_val <= 1e-6,
            'constraint_violation': max(0, constraint_val)
        }
    
    def solve(self, graph, R=0, num_samples=100):
        """
        Solve the complete MISDP problem.
        
        Args:
            graph (nx.Graph): NetworkX graph with required node and edge attributes
            R (float): Constraint bound
            num_samples (int): Number of samples for rounding
            
        Returns:
            dict: Solution results
        """
        # Set up problem from graph
        self.setup_from_graph(graph)
        
        start_time = time.time()
        
        # Step 1: Solve SDP relaxation
        logger.info("Step 1: Solving SDP relaxation")
        sdp_result = self.solve_sdp_relaxation(R)
        
        if sdp_result is None:
            logger.error("Failed to solve SDP relaxation")
            return None
        
        # Step 2: Random hyperplane rounding
        logger.info("Step 2: Applying random hyperplane rounding")
        x_rounded = self.random_hyperplane_rounding(sdp_result['X'], num_samples, R=R)
        logger.info(f"Hyperplane rounding result: {x_rounded}")
        
        # Step 3: Local search to improve the solution
        logger.info("Step 3: Applying local search")
        self.x = self.local_search(x_rounded, R)
        
        # Evaluate final solution
        final_eval = self.evaluate_solution(self.x, R)
        
        # Calculate implicit X from x
        self.X = np.outer(self.x, self.x)
        
        total_time = time.time() - start_time
        
        result = {
            'x': self.x,
            'X': self.X,
            'sdp_objective': sdp_result['objective_value'],
            'hw_cost': final_eval['hw_cost'],
            'sw_cost': final_eval['sw_cost'],
            'hw_sw_cost': final_eval['hw_sw_cost'],
            'sw_comm_cost': final_eval['sw_comm_cost'],
            'comm_cost': final_eval['comm_cost'],
            'total_cost': final_eval['total_cost'],
            'constraint_value': final_eval['constraint_value'],
            'constraint_satisfied': final_eval['constraint_satisfied'],
            'constraint_violation': final_eval['constraint_violation'],
            'solve_time': total_time
        }
        
        logger.info(f"Solution complete in {total_time:.2f} seconds")
        logger.info(f"Final objective value: {final_eval['hw_cost']:.4f}")
        logger.info(f"Constraint satisfied: {final_eval['constraint_satisfied']}")
        
        return result
    
    def visualize_solution(self, R=0, output_file=None):
        """
        Visualize the solution.
        
        Args:
            output_file (str, optional): Path to save the visualization
        """
        if self.graph is None or self.x is None:
            logger.error("Cannot visualize: No solution available")
            raise ValueError("No solution available")
        
        logger.info("Generating solution visualization")
        
        # Create a spring layout for the graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Node colors based on partition (hardware=blue, software=red)
        logger.info(self.x)
        node_colors = ['blue' if xi < 0.5 else 'red' for xi in self.x]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Draw nodes and edges
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, 
                node_size=700, font_weight='bold', font_color='white', ax=ax)
        
        # Draw edge labels (communication costs)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['communication_cost']:.1f}"
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=13, ax=ax)
        
        # Draw node labels (hardware/software costs)
        node_labels = {}
        for node in self.graph.nodes():
            hw_cost = self.graph.nodes[node]['hardware_cost']
            sw_cost = self.graph.nodes[node]['software_cost']
            node_labels[node] = f"H:{hw_cost:.1f}, S:{sw_cost:.1f}"
        
        # Position the node cost labels slightly below the nodes
        pos_labels = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(self.graph, pos_labels, labels=node_labels, font_size=13, ax=ax)
        
        # Create a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Hardware (x=0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Software (x=1)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        # Add title with solution information
        solution_eval = self.evaluate_solution(self.x, R=R)
        hw_count = sum(x < 0.5 for x in self.x)
        sw_count = sum(x >= 0.5 for x in self.x)
        
        ax.set_title(f"Node Partitioning Solution with maximum SW+Comm cost = {R}\n"
                 f"Hardware Nodes: {hw_count}, Software Nodes: {sw_count}\n"
                 f"HW Cost: {solution_eval['hw_cost']:.2f}, "
                 f"SW Cost: {solution_eval['sw_cost']:.2f}, "
                 f"Com Cost: {solution_eval['comm_cost']:.2f}, "
                 f"SW+Com Cost: {solution_eval['sw_comm_cost']:.2f}", fontsize=15)
        
        plt.axis('off')
        # plt.tight_layout()
        
        # Save or show the figure
        if output_file:
            directory = os.path.dirname(output_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()



# if __name__ == "__main__":
#     # Ensure necessary directories exist
#     os.makedirs("logs", exist_ok=True)
#     os.makedirs("results", exist_ok=True)
    
#     # Run main function
#     main()