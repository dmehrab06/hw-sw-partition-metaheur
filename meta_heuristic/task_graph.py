import networkx as nx
import random
import numpy as np
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.scheduler_utils import event_driven_heuristic,shortest_processing_time_heuristic,communication_aware_heuristic
from utils.scheduler_utils import critical_path_list_scheduling,compute_dag_execution_time

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
        self.rounak_graph = None
        self.current_heuristic = None
        self.software_costs = {}
        self.hardware_costs = {}
        self.hardware_area = {}
        self.communication_costs = {}
        self.area_constraint = area_constraint
        self.violation_cost = 1e9
        self.node_to_num = {}
        self.num_to_node = {}
        self.total_area = 0.0

        self.from_name_to_heuristics = {"makespan": compute_dag_execution_time,
                                        "ed": event_driven_heuristic,
                                       "cpl": critical_path_list_scheduling,
                                       "spt": shortest_processing_time_heuristic,
                                       "comm": communication_aware_heuristic}
        
        # Use the same logger as the main module
        logger.info("TaskGraph initialized with area constraint: %f", area_constraint)

    def load_graph_from_pydot(self, pydot_file, k=1.5, l=0.2, mu=0.5, A_max=100, seed=42, reproduce=True):
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
        try:
            # Try the original approach
            self.graph = nx.DiGraph(nx.nx_pydot.read_dot(pydot_file))
        except TypeError:
            # Fallback to pygraphviz
            self.graph = nx.DiGraph(nx.nx_agraph.read_dot(pydot_file))
            
        #self.graph = nx.DiGraph(nx.nx_pydot.read_dot(pydot_file))
        logger.info(f"Loaded graph from {pydot_file} with {len(self.graph.nodes())} nodes")

        if reproduce:
            np.random.seed(seed)
            random.seed(seed)
        
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
        # assert nx.is_connected(self.graph), "Graph must be connected"
        
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

        ## Initializing nx Graph for Rounak's function
        self.rounak_graph = self.graph
        nx.set_node_attributes(self.rounak_graph, self.hardware_area, 'area_cost')
        nx.set_node_attributes(self.rounak_graph, self.hardware_costs, 'hardware_time')
        nx.set_node_attributes(self.rounak_graph, self.software_costs, 'software_time')
        nx.set_edge_attributes(self.rounak_graph, self.communication_costs, 'communication_cost')
        logger.info(f"Graph initialized with total area: {self.total_area} with seed {seed}")

    def violates(self, solution):
        # Calculate execution costs and area usage
        area_consumption = [(self.hardware_area[node] if placement>0.5 else 0) for node,placement in solution.items()]
        return (1 if (sum(area_consumption)/self.total_area) > self.area_constraint else 0)
    
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

    def set_opt_heuristic(self,heuristic_name):
        if heuristic_name in self.from_name_to_heuristics:
            self.current_heuristic = heuristic_name
        else:
            self.current_heuristic = None
    
    def optimize_swarm(self, swarms):
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

    def optimize_swarm_heur(self, swarms):
        """
        Evaluate costs for a batch of particle swarm solutions based on mip makespan
        
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
        
        selected_heuristic = event_driven_heuristic
        if self.current_heuristic is not None and self.current_heuristic in self.from_name_to_heuristics:
            selected_heuristic = self.from_name_to_heuristics[self.current_heuristic]
            #print('selected heuristic based on',self.current_heuristic,'func name',selected_heuristic)
        
        exp_swarms = 1.0 / (1 + np.exp(-swarms))
        
        assert exp_swarms.shape[1] == len(self.software_costs), \
            f"Swarm dimension {exp_swarms.shape[1]} doesn't match number of nodes {len(self.software_costs)}"
        
        all_costs = []
        
        for swarm in exp_swarms:
            solution = {node: (0 if swarm[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            violation = self.violates(solution)
            if violation:
                #print('faced violation')
                all_costs.append(self.violation_cost)
            else:
                mip_assignment = solution
                makespan,_ = selected_heuristic(self.rounak_graph,mip_assignment)
                all_costs.append(makespan)
        
        return np.array(all_costs)
    
    def optimize_single_point(self, x, type='random'):
        """
        Evaluate costs for a single candidate solution.
        
        This method is designed to work with most pypop algorithms
        
        Args:
            x (np.ndarray): Array of shape (n_nodes)
        
        Returns:
            Scaler: Cost for the particular solution
            
        Note:
            Uses sigmoid transformation: exp_swarms = 1.0/(1+exp(-swarms))
        """
        if type=='vanilla' or type=='pso':
            swarms = 1.0 / (1 + np.exp(-x))
        
        assert x.shape[0] == len(self.software_costs), \
            f"Swarm dimension {x.shape[0]} doesn't match number of nodes {len(self.software_costs)}"

        solution = {node: x[self.node_to_num[node]] for node in self.graph.nodes()}
        
        return self.evaluate_partition_cost(solution)

    def optimize_single_point_heur(self, x, type='random'):
        """
        Evaluate costs for a single solution based on mip makespan calculation
        
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
        if type=='vanilla' or type=='pso':
            x = 1.0 / (1 + np.exp(-x))
        
        assert x.shape[0] == len(self.software_costs), \
            f"Swarm dimension {x.shape[0]} doesn't match number of nodes {len(self.software_costs)}"

        selected_heuristic = compute_dag_execution_time
        if self.current_heuristic is not None and self.current_heuristic in self.from_name_to_heuristics:
            selected_heuristic = self.from_name_to_heuristics[self.current_heuristic]
            #print('selected heuristic based on',self.current_heuristic,'func name',selected_heuristic)

        solution = {node: (0 if x[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
        violation = self.violates(solution)
        if violation:
            #print('faced violation')
            return self.violation_cost
        mip_assignment = solution
        makespan,_ = selected_heuristic(self.rounak_graph,mip_assignment)
        return makespan
    
    def optimize_random(self,assignment_candidates):
        """
        Evaluate costs for a batch of assignment probabilities.
        
        This method is designed to work with assignment probabilities directly, should NOT be DIRECTLY called with PSO.
        
        Args:
            assignment_candidates (np.ndarray): Array of shape (n_candidate, n_nodes) containing assignment probabilities
        
        Returns:
            np.ndarray: Array of costs for each particle solution
        """
        assert assignment_candidates.shape[1] == len(self.software_costs), \
            f"Swarm dimension {assignment_candidates.shape[1]} doesn't match number of nodes {len(self.software_costs)}"
        
        all_costs = []
        for assignment in assignment_candidates:
            solution = {node: assignment[self.node_to_num[node]] for node in self.graph.nodes()}
            all_costs.append(self.evaluate_partition_cost(solution))
        
        return np.array(all_costs)


    def optimize_random_heur(self,assignment_candidates):
        """
        Evaluate costs for a batch of assignment probabilities based on mip makespan.
        
        This method is designed to work with assignment probabilities directly; should NOT be DIRECTLY called with PSO.
        
        Args:
            assignment_candidates (np.ndarray): Array of shape (n_candidate, n_nodes) containing assignment probabilities
        
        Returns:
            np.ndarray: Array of costs for each particle solution
        """
        assert assignment_candidates.shape[1] == len(self.software_costs), \
            f"Swarm dimension {assignment_candidates.shape[1]} doesn't match number of nodes {len(self.software_costs)}"

        selected_heuristic = compute_dag_execution_time
        if self.current_heuristic is not None and self.current_heuristic in self.from_name_to_heuristics:
            selected_heuristic = self.from_name_to_heuristics[self.current_heuristic]
            #print('selected heuristic based on',self.current_heuristic,'func name',selected_heuristic)
        
        all_costs = []
        for assignment in assignment_candidates:
            solution = {node: (0 if assignment[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            violation = self.violates(solution)
            if violation:
                #print('faced violation')
                all_costs.append(self.violation_cost)
            else:
                mip_assignment = solution
                makespan,_ = selected_heuristic(self.rounak_graph,mip_assignment)
                all_costs.append(makespan)
        
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

    def get_partitioning(self,solution,method='random'):
        if method=='greedy':
            return solution #already in proper format
        
        #assert method in ['random','pso','dpso','gl25'], f"Method type {method} is not supported"
        assert solution.shape[0] == len(self.software_costs), \
            f"Solution dimension {solution.shape[0]} doesn't match number of nodes {len(self.software_costs)}"

        if method=='pso':
            solution = 1.0 / (1 + np.exp(-solution))

        return {node: (1 if solution[self.node_to_num[node]]>0.5 else 0) for node in self.graph.nodes()}
    
    def get_naive_solution(self):
         partition =  {node: 0 for node in self.graph.nodes()}
         return self.evaluate_partition_cost(partition),partition

    def evaluate_makespan(self, partition_assignment, verbose = False):
        makespan,_ = compute_dag_execution_time(self.rounak_graph,partition_assignment)
        return makespan
