import networkx as nx
import random
import numpy as np
import os, sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.scheduler_utils import compute_dag_makespan

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

    def optimize_swarm_makespan(self, swarms):
        """
        Evaluate costs for a batch of particle swarm solutions based on makespan
        
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
            solution = {node: (0 if swarm[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            all_costs.append(self.evaluate_makespan(solution)['makespan'])
        
        return np.array(all_costs)

    def optimize_swarm_makespan_mip(self, swarms):
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
        exp_swarms = 1.0 / (1 + np.exp(-swarms))
        
        assert exp_swarms.shape[1] == len(self.software_costs), \
            f"Swarm dimension {exp_swarms.shape[1]} doesn't match number of nodes {len(self.software_costs)}"
        
        all_costs = []
        
        for swarm in exp_swarms:
            solution = {node: (0 if swarm[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            violation = self.violates(solution)
            if violation:
                all_costs.append(self.violation_cost)
            else:
                mip_assignment = [1-solution[k] for k in self.rounak_graph]
                makespan,_ = compute_dag_makespan(self.rounak_graph,mip_assignment)
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

    def optimize_single_point_makespan(self, x, type='random'):
        """
        Evaluate costs for a single solution based on makespan calculation
        
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
            swarms = 1.0 / (1 + np.exp(-x))
        
        assert x.shape[0] == len(self.software_costs), \
            f"Swarm dimension {x.shape[0]} doesn't match number of nodes {len(self.software_costs)}"

        solution = {node: (0 if x[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
        
        return self.evaluate_makespan(solution)['makespan']

    def optimize_single_point_makespan_mip(self, x, type='random'):
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

        solution = {node: (0 if x[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
        violation = self.violates(solution)
        if violation:
            return self.violation_cost
        mip_assignment = [1-solution[k] for k in self.rounak_graph]
        makespan,_ = compute_dag_makespan(self.rounak_graph,mip_assignment)
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

    def optimize_random_makespan(self,assignment_candidates):
        """
        Evaluate costs for a batch of assignment probabilities based on makespan calculation.
        
        This method is designed to work with assignment probabilities directly; should NOT be DIRECTLY called with PSO.
        
        Args:
            assignment_candidates (np.ndarray): Array of shape (n_candidate, n_nodes) containing assignment probabilities
        
        Returns:
            np.ndarray: Array of costs for each particle solution
        """
        assert assignment_candidates.shape[1] == len(self.software_costs), \
            f"Swarm dimension {assignment_candidates.shape[1]} doesn't match number of nodes {len(self.software_costs)}"
        
        all_costs = []
        for assignment in assignment_candidates:
            solution = {node: (0 if assignment[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            all_costs.append(self.evaluate_makespan(solution)['makespan'])
        
        return np.array(all_costs)

    def optimize_random_makespan_mip(self,assignment_candidates):
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
        
        all_costs = []
        for assignment in assignment_candidates:
            solution = {node: (0 if assignment[self.node_to_num[node]]<0.5 else 1) for node in self.graph.nodes()}
            violation = self.violates(solution)
            if violation:
                all_costs.append(self.violation_cost)
            else:
                mip_assignment = [1-solution[k] for k in self.rounak_graph]
                makespan,_ = compute_dag_makespan(self.rounak_graph,mip_assignment)
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

    def get_naive_solution_makespan(self):
         partition =  {node: 0 for node in self.graph.nodes()}
         return self.evaluate_makespan(partition)['makespan'],partition

    def evaluate_makespan(self, partition_assignment, verbose = False):
        """
        Compute overall execution time (estimated scheduling) for a DAG with given hardware/software partitions
        Adopted from Rounak's function utils/scheduler_utils.py
        
        Args:
            partition_assignment: Dict mapping node_id -> 'hardware'(1) or 'software'(0)
            verbose: If True, print detailed execution schedule
            
        Returns:
            Dict containing execution details including makespan, start times, finish times
        """
        
        # Validate partition assignment
        for node in self.graph.nodes():
            if node not in partition_assignment:
                raise ValueError(f"Partition not specified for node {node}")
            if partition_assignment[node] not in [1, 0]:
                raise ValueError(f"Invalid partition '{partition_assignment[node]}' for node {node}")
        
        # # Validate required node attributes
        # required_node_attrs = ['hardware_time', 'software_time', 'area_cost']
        # for node in graph.nodes():
        #     for attr in required_node_attrs:
        #         if attr not in graph.nodes[node]:
        #             raise ValueError(f"Node {node} missing required attribute '{attr}'")
        
        # # Validate required edge attributes
        # for u, v in graph.edges():
        #     if 'communication_cost' not in graph.edges[u, v]:
        #         raise ValueError(f"Edge ({u}, {v}) missing required attribute 'communication_cost'")
        
        # Initialize data structures
        start_times = {}
        finish_times = {}
        hw_nodes = [node for node, partition in partition_assignment.items() if partition == 1]
        sw_nodes = [node for node, partition in partition_assignment.items() if partition == 0]
        
        # Track which nodes have completed execution
        completed_nodes = set()
        
        # Get node and edge attributes
        node_hw_times = self.hardware_costs
        node_sw_times = self.software_costs
        edge_comm_costs = self.communication_costs
        
        def get_execution_time(node):
            """Get execution time for a node based on its partition"""
            if partition_assignment[node] == 1:
                return node_hw_times[node]
            else:
                return node_sw_times[node]
        
        def get_earliest_start_time(node):
            """Calculate earliest start time considering dependencies and communication"""
            earliest_start = 0
            
            for predecessor in self.graph.predecessors(node):
                if predecessor not in finish_times:
                    return None  # Predecessor not yet completed
                
                pred_finish = finish_times[predecessor]
                
                # Add communication delay if nodes are in different partitions
                if partition_assignment[predecessor] != partition_assignment[node]:
                    comm_delay = edge_comm_costs.get((predecessor, node), 0)
                    pred_finish += comm_delay
                
                earliest_start = max(earliest_start, pred_finish)
            
            return earliest_start
        
        # Get topological order for processing
        topo_order = list(nx.topological_sort(self.graph))
        
        # Initialize with source nodes (nodes with no predecessors)
        ready_nodes = [node for node in topo_order if self.graph.in_degree(node) == 0]
        
        # Track software execution (sequential)
        sw_execution_queue = []
        sw_current_time = 0
        
        # Track hardware nodes ready to execute
        hw_ready_nodes = []
        
        current_time = 0
        iteration = 0
        max_iterations = len(self.graph.nodes()) * 10  # Prevent infinite loops
        
        if verbose:
            logger.info(f"Starting execution simulation...")
            logger.info(f"Hardware nodes: {hw_nodes}")
            logger.info(f"Software nodes: {sw_nodes}")
            logger.info(f"Initial ready nodes: {ready_nodes}")
        
        while (ready_nodes or hw_ready_nodes or sw_execution_queue) and iteration < max_iterations:
            iteration += 1
            
            if verbose:
                logger.info(f"\\n--- Iteration {iteration}, Time {current_time:.2f} ---")
                logger.info(f"Ready nodes: {ready_nodes}")
                logger.info(f"HW ready queue: {[n for n, _ in hw_ready_nodes]}")
                logger.info(f"SW execution queue: {[n for n, _ in sw_execution_queue]}")
                logger.info(f"Completed nodes: {sorted(completed_nodes)}")
            
            # Process ready nodes and categorize them
            new_hw_ready = []
            new_sw_ready = []
            
            for node in ready_nodes:
                earliest_start = get_earliest_start_time(node)
                if earliest_start is not None:
                    if partition_assignment[node] == 'hardware':
                        new_hw_ready.append((node, earliest_start))
                    else:
                        new_sw_ready.append((node, earliest_start))
            
            ready_nodes = []
            
            # Add new hardware nodes to ready queue
            hw_ready_nodes.extend(new_hw_ready)
            
            # Start hardware nodes that can execute now
            hw_to_start = [(node, ready_time) for node, ready_time in hw_ready_nodes 
                           if ready_time <= current_time]
            
            for node, ready_time in hw_to_start:
                actual_start = max(current_time, ready_time)
                start_times[node] = actual_start
                finish_times[node] = actual_start + get_execution_time(node)
                completed_nodes.add(node)
                hw_ready_nodes.remove((node, ready_time))
                
                if verbose:
                    logger.info(f"  Started HW node {node}: start={start_times[node]:.2f}, "
                         f"finish={finish_times[node]:.2f}, duration={get_execution_time(node):.2f}")
            
            # Add software nodes to execution queue
            for node, earliest_start in new_sw_ready:
                sw_execution_queue.append((node, earliest_start))
            
            # Execute next software node if possible
            if sw_execution_queue:
                node, earliest_start = sw_execution_queue[0]
                if earliest_start <= max(sw_current_time, current_time):
                    sw_execution_queue.pop(0)
                    actual_start = max(sw_current_time, earliest_start, current_time)
                    start_times[node] = actual_start
                    finish_times[node] = actual_start + get_execution_time(node)
                    sw_current_time = finish_times[node]
                    completed_nodes.add(node)
                    
                    if verbose:
                        logger.info(f"  Started SW node {node}: start={start_times[node]:.2f}, "
                             f"finish={finish_times[node]:.2f}, duration={get_execution_time(node):.2f}")
            
            # Find newly ready nodes
            for node in topo_order:
                if (node not in completed_nodes and 
                    node not in ready_nodes and
                    node not in [n for n, _ in hw_ready_nodes] and
                    node not in [n for n, _ in sw_execution_queue]):
                    
                    # Check if all predecessors are completed
                    predecessors = list(self.graph.predecessors(node))
                    if all(pred in completed_nodes for pred in predecessors):
                        ready_nodes.append(node)
                        if verbose:
                            logger.info(f"  Node {node} became ready (all predecessors completed)")
            
            # Advance time to next event
            next_events = []
            
            # Hardware nodes that will complete
            hw_completions = [finish_times[node] for node in completed_nodes 
                             if node in finish_times and finish_times[node] > current_time]
            next_events.extend(hw_completions)
            
            # Hardware nodes that will become ready
            hw_ready_times = [ready_time for node, ready_time in hw_ready_nodes 
                             if ready_time > current_time]
            next_events.extend(hw_ready_times)
            
            # Next software node ready time
            if sw_execution_queue:
                node, ready_time = sw_execution_queue[0]
                next_sw_time = max(ready_time, sw_current_time)
                if next_sw_time > current_time:
                    next_events.append(next_sw_time)
            
            # Advance to next event or make small increment
            if next_events:
                current_time = min(next_events)
            elif ready_nodes or hw_ready_nodes or sw_execution_queue:
                current_time += 0.1  # Small increment to avoid infinite loop
            else:
                break  # No more events
        
        if iteration >= max_iterations:
            logger.warning(f"Maximum iterations ({max_iterations}) reached. Possible infinite loop.")
        
        # Calculate results
        if not finish_times:
            return {'makespan': 0, 'start_times': {}, 'finish_times': {}, 'error': 'No nodes completed'}
        
        makespan = max(finish_times.values())
        
        # Calculate partition-specific metrics
        hw_finish_times = [finish_times[node] for node in hw_nodes if node in finish_times]
        sw_finish_times = [finish_times[node] for node in sw_nodes if node in finish_times]
        
        hw_makespan = max(hw_finish_times) if hw_finish_times else 0
        sw_makespan = max(sw_finish_times) if sw_finish_times else 0
        
        # Calculate communication overhead
        total_comm_delay = 0
        active_comm_edges = []
        
        for u, v in self.graph.edges():
            if partition_assignment[u] != partition_assignment[v]:
                comm_delay = edge_comm_costs[(u, v)]
                total_comm_delay += comm_delay
                active_comm_edges.append((u, v, comm_delay))

        area_used = 0

        for node in hw_nodes:
            area_used = area_used + self.hardware_area[node]
        
        if area_used / self.total_area > self.area_constraint:
            makespan = self.violation_cost
            hw_makespan = self.violation_cost
            sw_makespan = self.violation_cost
            total_comm_delay = self.violation_cost
        
        result = {
            'makespan': makespan,
            'start_times': start_times,
            'finish_times': finish_times,
            'hardware_nodes': hw_nodes,
            'software_nodes': sw_nodes,
            'hardware_makespan': hw_makespan,
            'software_makespan': sw_makespan,
            'total_communication_delay': total_comm_delay,
            'active_communication_edges': active_comm_edges,
            'execution_summary': {
                'total_nodes': len(self.graph.nodes()),
                'completed_nodes': len(completed_nodes),
                'hardware_execution_time': hw_makespan,
                'software_execution_time': sw_makespan,
                'communication_overhead': total_comm_delay,
                'total_makespan': makespan
            }
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"EXECUTION SUMMARY")
            print(f"{'='*50}")
            print(f"Total Makespan: {makespan:.2f}")
            print(f"Hardware Makespan: {hw_makespan:.2f} (nodes: {hw_nodes})")
            print(f"Software Makespan: {sw_makespan:.2f} (nodes: {sw_nodes})")
            print(f"Communication Overhead: {total_comm_delay:.2f}")
            print(f"Active Communication Edges: {len(active_comm_edges)}")
            print(f"Nodes Completed: {len(completed_nodes)}/{len(self.graph.nodes())}")
            
            if active_comm_edges:
                print(f"\nInter-partition communications:")
                for u, v, delay in active_comm_edges:
                    print(f"  ({u} -> {v}): {delay:.2f}")
        
        return result
