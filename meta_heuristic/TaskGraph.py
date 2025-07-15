import networkx as nx
import random
import numpy as np
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/task_graph.log")

logger = LogManager.get_logger(__name__)

class TaskGraph:
    """
    A class representing a task graph for hardware-software partitioning optimization.
    
    This class handles the creation and evaluation of task graphs where nodes can be
    assigned to either hardware or software, with associated costs and area constraints.
    
    Attributes:
        graph (nx.Graph): NetworkX graph representing the task dependencies
        software_costs (dict): Software execution costs for each node
        hardware_costs (dict): Hardware execution costs for each node
        hardware_area (dict): Hardware area requirements for each node
        communication_costs (dict): Communication costs between nodes
        area_constraint (float): Maximum allowed hardware area utilization (0-1)
        violation_cost (float): Penalty cost for constraint violations
        node_to_num (dict): Mapping from node names to numerical indices
        num_to_node (dict): Mapping from numerical indices to node names
        total_area (float): Total available hardware area
    """
    
    def __init__(self, area_constraint=0.8):
        """
        Initialize the TaskGraph Class.
        
        Args:
            area_constraint (float): Maximum hardware area utilization ratio (default: 0.8)
                Must be between 0 and 1.
        """
        self.graph = None
        self.software_costs = {}
        self.hardware_costs = {}
        self.hardware_area = {}
        self.communication_costs = {}
        self.area_constraint = area_constraint
        self.violation_cost = 1e9
        self.node_to_num = {}
        self.num_to_node = {}
        self.total_area = 0.0
        
        # Use the same logger as the main module
        logger.info("TaskGraph initialized with area constraint: %f", area_constraint)

    def load_graph_from_pydot(self, pydot_file, k=1.5, l=0.2, mu=0.5, A_max=100):
        """
        Load a graph from a PyDot file and assign random cost attributes.
        
        This method loads a graph structure and assigns random costs following
        specific distributions to simulate realistic hardware-software partitioning scenarios.
        
        Args:
            pydot_file (str): Path to the PyDot file to load
            k (float): Hardware cost scale factor relative to software costs (default: 1.5)
            l (float): Hardware cost variance factor (default: 0.2)
            mu (float): Communication cost scale factor (default: 0.5)
            A_max (float): Maximum hardware area for any single node (default: 100)
            
        Raises:
            AssertionError: If the loaded graph is not connected
            
        Note:
            - Software costs are uniformly distributed between 1 and 100
            - Hardware costs follow a normal distribution: N(k*s_i, (l*k*s_i)²)
            - Hardware areas are uniformly distributed between 1 and A_max
            - Communication costs are uniformly distributed between 0 and 2*mu*s_max
        """
        # Load graph structure
        self.graph = nx.Graph(nx.nx_pydot.read_dot(pydot_file))
        logger.info(f"Loaded graph from {pydot_file} with {len(self.graph.nodes())} nodes")
        
        # Assign software costs and hardware areas
        self.software_costs = {node: random.uniform(1, 100) for node in self.graph.nodes()}
        self.hardware_area = {node: random.uniform(1, A_max) for node in self.graph.nodes()}
        
        # Create node mappings for numerical operations
        self.node_to_num = {node: i for i, node in enumerate(self.graph.nodes())}
        self.num_to_node = {self.node_to_num[key]: key for key in self.node_to_num}
        
        # Calculate total area and maximum software cost
        s_max = max(self.software_costs.values())
        self.total_area = sum(self.hardware_area.values())
        
        # Ensure graph connectivity
        assert nx.is_connected(self.graph), "Graph must be connected"
        
        # Assign hardware costs based on software costs
        for node, s_i in self.software_costs.items():
            mean = k * s_i
            std_dev = l * k * s_i
            # Ensure hardware costs are positive
            hw_cost = max(0.1, np.random.normal(mean, std_dev))
            self.hardware_costs[node] = hw_cost
        
        # Assign communication costs
        for u, v in self.graph.edges():
            comm_cost = random.uniform(0, 2 * mu * s_max)
            self.communication_costs[(u, v)] = comm_cost
            
        logger.info(f"Graph initialized with total area: {self.total_area}")

    def evaluate_partition_cost(self, solution):
        """
        Evaluate the total cost of a given hardware-software partition.
        
        This method calculates the total cost including execution costs and
        communication costs, with penalties for area constraint violations.
        
        Args:
            solution (dict): Partition assignment where keys are node names and
                           values are assignment probabilities (<=0.5: software, >0.5: hardware)
        
        Returns:
            float: Total cost of the partition including:
                  - Execution costs (hardware or software based on assignment)
                  - Communication costs between differently assigned nodes
                  - Penalty cost if area constraint is violated
        """
        cost = 0
        area = 0
        
        # Calculate execution costs and area usage
        for node, placement in solution.items():
            if placement <= 0.5:  # Software assignment
                cost += self.software_costs[node]
            else:  # Hardware assignment
                cost += self.hardware_costs[node]
                area += self.hardware_area[node]
        
        # Calculate communication costs
        for edge in self.communication_costs:
            # Add communication cost if nodes are on different platforms
            if (solution[edge[0]] <= 0.5) != (solution[edge[1]] <= 0.5):
                cost += self.communication_costs[edge]
        
        # Apply area constraint penalty
        if area / self.total_area > self.area_constraint:
            return self.violation_cost
        else:
            return cost

    def evaluation_from_swarm(self, swarms):
        """
        Evaluate costs for a batch of particle swarm solutions.
        
        This method is designed to work with particle swarm optimization algorithms
        that provide solutions as matrices of continuous values.
        
        Args:
            swarms (np.ndarray): Array of shape (n_particles, n_nodes) containing
                               particle positions that will be converted to probabilities
                               using sigmoid function
        
        Returns:
            np.ndarray: Array of costs for each particle solution
            
        Note:
            Uses sigmoid transformation: exp_swarms = 1.0/(1+exp(-swarms))
        """
        exp_swarms = 1.0 / (1 + np.exp(-swarms))
        
        assert exp_swarms.shape[1] == len(self.software_costs), \
            f"Swarm dimension {exp_swarms.shape[1]} doesn't match number of nodes {len(self.software_costs)}"
        
        all_costs = []
        for swarm in exp_swarms:
            solution = {node: swarm[self.node_to_num[node]] for node in self.graph.nodes()}
            all_costs.append(self.evaluate_partition_cost(solution))
        
        return np.array(all_costs)

    def find_best_cost(self, swarms):
        """
        Find the best (minimum) cost from a batch of solutions.
        
        Args:
            swarms (np.ndarray): Array of particle positions
            
        Returns:
            float: Best (minimum) cost found among all solutions
        """
        exp_swarms = np.exp(swarms)
        assert exp_swarms.shape[1] == len(self.software_costs)
        
        best_cost = 1e9
        for swarm in exp_swarms:
            solution = {node: swarm[self.node_to_num[node]] for node in self.graph.nodes()}
            cur_cost = self.evaluate_partition_cost(solution)
            if cur_cost < best_cost:
                best_cost = cur_cost
        
        return best_cost

    def naive_lower_bound(self):
        """
        Calculate a naive lower bound for the partitioning problem.
        
        This bound assumes no communication costs and selects the minimum
        execution cost (hardware or software) for each node, ignoring area constraints.
        
        Returns:
            float: Naive lower bound on the optimal solution cost
            
        Note:
            This is an optimistic bound that may not be achievable due to
            area constraints and communication costs.
        """
        total_time = 0.0
        for node in self.graph.nodes():
            total_time += min(self.hardware_costs[node], self.software_costs[node])
        
        logger.info(f"Calculated naive lower bound: {total_time}")
        return total_time

    def greedy_heur(self):
        """
        Implement a greedy heuristic based on gain per area ratio.
        
        This heuristic selects nodes for hardware implementation based on the
        gain per unit area, similar to a fractional knapsack approach.
        
        Returns:
            tuple: (best_cost, assignment) where:
                  - best_cost (float): Cost of the greedy solution
                  - assignment (dict): Node assignments (0: software, 1: hardware)
                  
        Algorithm:
            1. Calculate gain per area for each node: (software_cost - hardware_cost) / area
            2. Sort nodes by gain per area in descending order
            3. Assign nodes to hardware until area constraint is reached
            4. Assign remaining nodes to software
        """
        gain_list = []
        
        # Calculate gain per area for each node
        for node in self.hardware_costs:
            gain_per_area = (self.software_costs[node] - self.hardware_costs[node]) / self.hardware_area[node]
            gain_list.append((gain_per_area, node))
        
        # Sort by gain per area (descending)
        gain_list.sort(reverse=True)
        
        # Initialize all nodes to software
        assignment = {node: 0 for node in self.hardware_costs}
        area_used = 0
        
        # Greedily assign nodes to hardware
        for gain_per_area, node in gain_list:
            if (area_used + self.hardware_area[node]) / self.total_area > self.area_constraint:
                break
            assignment[node] = 1
            area_used += self.hardware_area[node]
        
        best_cost_heur = self.evaluate_partition_cost(assignment)
        logger.info(f"Greedy heuristic found solution with cost: {best_cost_heur}")
        return best_cost_heur, assignment
