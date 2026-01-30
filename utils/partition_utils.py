"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import sys, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cvxpy as cp
import pickle
import random
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/schedule_partitioner.log")

logger = LogManager.get_logger(__name__)

class ScheduleConstPartitionSolver:
    """
    Schedule Constrained Hardware-Software Partitioning Solver using MILP formulation
    """
    
    def __init__(self, graph:nx.DiGraph=None):
        if not graph:
            self.graph = None
            self.n_nodes = 0
            self.n_edges = 0
            self.edge_list = []
            
            # Problem matrices and vectors
            self.h = None  # hardware execution times
            self.s = None  # software execution times  
            self.a = None  # hardware area costs
            self.c = None  # communication costs
            self.S_source = None  # source selection matrix
            self.S_target = None  # target selection matrix
        
        else:
            self.graph = graph
            self.n_nodes = len(self.graph.nodes())
            self.n_edges = self.graph.number_of_edges()
            self.edge_list = list(self.graph.edges())
            self._create_problem_matrices()
            logger.info("Loaded graph and created problem matrices successfully")
        
        # Solution
        self.x_sol = None  # partition assignment
        self.t_sol = None  # start times
        self.f_sol = None  # finish times
        self.z_sol = None  # communication variables
        self.T_sol = None  # makespan
        self.Y_sol = None  # software ordering

    def load_pickle_graph(self, graph_file):
        try:
            with open(graph_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded successfully from {graph_file}")
            # You can now work with the loaded graph 'G'
        except FileNotFoundError:
            logger.error(f"Error: The pickle file '{graph_file}' was not found.")
            raise FileNotFoundError(f"Error: The pickle file '{graph_file}' was not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading the graph data: {e}")
            raise
        
        self.populate_graph_with_attributes(data)
        self.n_nodes = len(self.graph.nodes())
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())

        # Create problem vectors
        self._create_problem_matrices()
        logger.info(f"Created DAG with {self.n_nodes} nodes and {self.n_edges} edges")
        return self.graph
    
    # def populate_graph_with_attributes(self, data):
    #     self.graph = data.graph
    #     # node/edge attributes
    #     nx.set_node_attributes(self.graph, data.hardware_area, 'area_cost')
    #     nx.set_node_attributes(self.graph, data.hardware_costs, 'hardware_time')
    #     nx.set_node_attributes(self.graph, data.software_costs, 'software_time')
    #     nx.set_edge_attributes(self.graph, data.communication_costs, 'communication_cost')
    #     return

    def populate_graph_with_attributes(self, graph, hardware_area, hardware_costs, software_costs, communication_costs):

        self.graph = graph
        # node/edge attributes
        nx.set_node_attributes(self.graph, hardware_area, 'area_cost')
        nx.set_node_attributes(self.graph, hardware_costs, 'hardware_time')
        nx.set_node_attributes(self.graph, software_costs, 'software_time')
        nx.set_edge_attributes(self.graph, communication_costs, 'communication_cost')
        return
    
    def load_networkx_graph_with_torch_feats(self, graph, hardware_areas, hardware_costs, software_costs, communication_costs):

        hardware_areas_dict = {i: hwa for i, hwa in enumerate(hardware_areas.tolist())}
        hardware_costs_dict = {i: hwc for i, hwc in enumerate(hardware_costs.tolist())}
        software_costs_dict = {i: swc for i, swc in enumerate(software_costs.tolist())}
        communication_costs_dict = {e: cc for (e, cc) in zip(list(graph.edges),communication_costs.tolist())}

        self.populate_graph_with_attributes(graph, hardware_areas_dict, hardware_costs_dict, software_costs_dict, communication_costs_dict)

        # The above function assigns global class var self.graph
        self.n_nodes = len(self.graph.nodes())
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())

        # Create problem vectors
        self._create_problem_matrices()
        logger.info(f"Created DAG with {self.n_nodes} nodes and {self.n_edges} edges")
        return self.graph
        
    def load_pydot_graph(self, pydot_file, k=1.5, l=0.2, mu=0.5, A_max=100) -> nx.DiGraph:
        """
        Create a random directed acyclic graph with required attributes
        
        Args:
            
            
        Returns:
            NetworkX directed graph
        """
        # Create random DAG
        self.graph = nx.DiGraph(nx.nx_pydot.read_dot(pydot_file))
        logger.info(f"Loaded graph from {pydot_file} with {len(self.graph.nodes())} nodes")

        # Assign node attributes
        for node in self.graph.nodes():
            
            # Assign software execution time and hardware areas
            software_time = random.uniform(1, 100)
            hardware_area = random.uniform(1, A_max)
            self.graph.nodes[node]['software_time'] = software_time
            self.graph.nodes[node]['area_cost'] = hardware_area

            # Assign random hardware execution time
            mean = k * software_time
            std_dev = l * k * software_time
            # Ensure hardware costs are positive
            hardware_time = max(0.1, np.random.normal(mean, std_dev))
            self.graph.nodes[node]['hardware_time'] = hardware_time

        # Calculate total area and maximum software cost
        s_max = max([self.graph.nodes[n]['software_time'] for n in self.graph.nodes])
        total_area = sum([self.graph.nodes[n]['area_cost'] for n in self.graph.nodes])

        # # Ensure graph connectivity
        # assert nx.is_connected(self.graph), "Graph must be connected"
        
        # Assign communication costs
        for u, v in self.graph.edges():
            comm_cost = random.uniform(0, 2 * mu * s_max)
            self.graph[u][v]['communication_cost'] = comm_cost
            
        logger.info(f"Graph initialized with total hardware area requirement: {total_area}")
        
        self.n_nodes = len(self.graph.nodes())
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())

        # Create problem vectors
        self._create_problem_matrices()
        logger.info(f"Created DAG with {self.n_nodes} nodes and {self.n_edges} edges")
        return self.graph
    
    
    def _create_problem_matrices(self):
        """Create matrices and vectors for optimization problem"""
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        # Get sorted list of node IDs for consistent ordering
        self.node_list = sorted(list(self.graph.nodes()))
        self.node_to_index = {node: i for i, node in enumerate(self.node_list)}
        self.index_to_node = {i: node for i, node in enumerate(self.node_list)}
        
        # Create node attribute vectors (indexed by position in node_list)
        self.h = np.array([self.graph.nodes[node]['hardware_time'] for node in self.node_list])
        self.s = np.array([self.graph.nodes[node]['software_time'] for node in self.node_list])
        self.a = np.array([self.graph.nodes[node]['area_cost'] for node in self.node_list])
        
        # Create communication cost vector
        self.c = np.array([self.graph.edges[edge]['communication_cost'] for edge in self.edge_list])
        
        # Create selection matrices
        self.S_source = np.zeros((self.n_edges, self.n_nodes))
        self.S_target = np.zeros((self.n_edges, self.n_nodes))
        
        for k, (source_node, target_node) in enumerate(self.edge_list):
            source_idx = self.node_to_index[source_node]
            target_idx = self.node_to_index[target_node]
            self.S_source[k, source_idx] = 1  # edge k starts from source_node
            self.S_target[k, target_idx] = 1  # edge k ends at target_node
        
        logger.info("Problem matrices and vectors created successfully")
    
    def solve_optimization(self, A_max: float, big_M: float = 1000, use_reduced_sw_constraints=True, partition_assignment:dict=None) -> Dict:
        """
        Solve the hardware-software partitioning optimization problem
        
        Args:
            A_max: Maximum hardware area constraint
            big_M: Big-M constant for logical constraints
            use_reduced_sw_constraints: flag to reduce SW sequence constraints through topological ordering
            partition_assignment (dict): dictionary of partition assignments, if only solving scheduling problem. Default is None 
            
            !!!IMPORTANT: Note that the convention used in partition assignment dictionary is opposite. HW is 1, SW is 0
            
        Returns:
            Dictionary containing solution details
        """
        if self.S_source is None:
            raise ValueError("Problem matrices not created. Load a graph first.")
        
        logger.info(f"Solving optimization with A_max = {A_max}")
        
        # Decision variables
        x = cp.Variable(self.n_nodes, boolean=True)  # partition assignment
        z = cp.Variable(self.n_edges, boolean=True)  # communication variables
        t = cp.Variable(self.n_nodes, nonneg=True)   # start times
        f = cp.Variable(self.n_nodes, nonneg=True)   # finish times
        T = cp.Variable(nonneg=True)                 # makespan
        
        # Objective: minimize makespan
        objective = cp.Minimize(T)
        
        # Constraints
        constraints = []
        
        # 1. Hardware area constraint
        constraints.append(self.a.T @ (1 - x) <= A_max)
        
        # 2. Execution time definition
        constraints.append(f == t + cp.multiply(self.h, 1 - x) + cp.multiply(self.s, x))
        
        # 3. Precedence constraints with communication
        constraints.append(self.S_source @ f + cp.multiply(self.c, z) <= self.S_target @ t)
        
        # 4. Communication cost variables (different partition detection)
        constraints.append(z >= self.S_source @ x - self.S_target @ x)
        constraints.append(z >= self.S_target @ x - self.S_source @ x)
        constraints.append(z <= self.S_source @ x + self.S_target @ x)
        constraints.append(z <= 2 - self.S_source @ x - self.S_target @ x)
        
        # 5. Sequential execution of software nodes
        if use_reduced_sw_constraints:
            # Smart reduction: only constrain node pairs that can potentially execute in parallel
            sw_constraint_pairs = self._get_software_constraint_pairs_with_levels()
            
            print(f"Smart reduction: {len(sw_constraint_pairs)} software constraint pairs (vs {self.n_nodes*(self.n_nodes-1)//2} full pairwise)")
            # sys.exit(0)
            
            for i, j in sw_constraint_pairs:
                # Use binary variable to enforce ordering between potentially parallel software nodes
                y_ij = cp.Variable(boolean=True)
                
                # Either i finishes before j starts, or j finishes before i starts
                constraints.append(f[i] <= t[j] + big_M * y_ij + big_M * (2 - x[i] - x[j]))
                constraints.append(f[j] <= t[i] + big_M * (1 - y_ij) + big_M * (2 - x[i] - x[j]))
        else:
            # Original O(n²) approach with binary ordering variables
            Y = cp.Variable((self.n_nodes, self.n_nodes), boolean=True)  # software ordering
            
            logger.info(f"Adding {self.n_nodes*(self.n_nodes-1)} software sequencing constraints (full pairwise)")
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        constraints.append(f[i] <= t[j] + big_M * (1 - Y[i, j]) + big_M * (2 - x[i] - x[j]))
                        constraints.append(f[j] <= t[i] + big_M * Y[i, j] + big_M * (2 - x[i] - x[j]))
                        constraints.append(Y[i, j] <= x[i])
                        constraints.append(Y[i, j] <= x[j])
        
        # 6. Makespan definition
        for i in range(self.n_nodes):
            constraints.append(T >= f[i])

        # Special constraints for known partitions
        if partition_assignment:
            logger.info("Adding additional constraints to set binary variables.")
            for i,n in enumerate(self.node_list):
                constraints.append(x[i] == 1 - partition_assignment[n])
            for k,e in enumerate(self.edge_list):
                src_node, dst_node = e
                if partition_assignment[src_node] == partition_assignment[dst_node]:
                    constraints.append(z[k] == 0)
                else:
                    constraints.append(z[k] == 1)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIP, verbose=True)
        
        if problem.status != cp.OPTIMAL:
            logger.error(f"Optimization failed with status: {problem.status}")
            return None
        
        # Store solution
        self.x_sol = x.value.round().astype(int)
        self.t_sol = t.value
        self.f_sol = f.value
        self.z_sol = z.value.round().astype(int)
        self.T_sol = T.value
        if not use_reduced_sw_constraints:
            self.Y_sol = Y.value.round().astype(int)
        else:
            self.Y_sol = None  # Not used in reduced formulation
        
        # Convert solution back to node IDs
        hw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 0]
        sw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 1]
        
        # Sort by start times
        hw_sequence = sorted(hw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
        sw_sequence = sorted(sw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
        
        # Create start/finish time dictionaries with node IDs
        start_times_dict = {self.node_list[i]: self.t_sol[i] for i in range(self.n_nodes)}
        finish_times_dict = {self.node_list[i]: self.f_sol[i] for i in range(self.n_nodes)}
        
        solution = {
            'status': problem.status,
            'makespan': self.T_sol,
            'hardware_nodes': hw_nodes,
            'software_nodes': sw_nodes,
            'hardware_sequence': hw_sequence,
            'software_sequence': sw_sequence,
            'start_times': start_times_dict,
            'finish_times': finish_times_dict,
            'total_hardware_area': np.sum(self.a * (1 - self.x_sol)),
            'area_constraint': A_max
        }
        
        logger.info(f"Optimization successful! Makespan = {self.T_sol:.2f}")
        logger.info(f"Hardware nodes: {hw_nodes}")
        logger.info(f"Software nodes: {sw_nodes}")
        logger.info(f"Total hardware area used: {solution['total_hardware_area']:.2f} / {A_max}")
        
        return solution
    
    def _get_software_constraint_pairs(self):
        """
        Get the minimal set of node pairs that need software sequencing constraints.
        
        Key insight: We ONLY need constraints for nodes that are "incomparable" in the DAG,
        meaning they have NO dependency relationship between them. These are the nodes that
        could potentially be ready to execute at the same time.
        
        Why this is minimal:
        - If A->B (direct or transitive), precedence constraints handle ordering
        - If neither A->B nor B->A, they could be ready simultaneously
        - For software execution, we must impose sequential ordering on such pairs
        
        Returns:
            List of tuples (i, j) where i < j (incomparable node pairs)
        """
        # Compute transitive closure to find all reachability relationships
        transitive_closure = nx.transitive_closure_dag(self.graph)
        
        constraint_pairs = []
        
        # Check all pairs of nodes
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                node_i = self.node_list[i]
                node_j = self.node_list[j]
                
                # Check if there's a dependency path in either direction
                i_reaches_j = transitive_closure.has_edge(node_i, node_j)
                j_reaches_i = transitive_closure.has_edge(node_j, node_i)
                
                # If NO dependency exists in either direction, they are incomparable
                # These nodes could potentially execute simultaneously
                if not i_reaches_j and not j_reaches_i:
                    constraint_pairs.append((i, j))
        
        if constraint_pairs:
            print(f"\nSoftware sequencing constraint analysis:")
            print(f"  Incomparable node pairs (need sequencing): {len(constraint_pairs)}")
            print(f"  Total possible pairs: {self.n_nodes*(self.n_nodes-1)//2}")
            print(f"  Reduction: {100*(1-len(constraint_pairs)/(self.n_nodes*(self.n_nodes-1)//2)):.1f}%")
            
            # Show some examples if verbose
            if len(constraint_pairs) <= 10:
                print(f"  Incomparable pairs: {[(self.node_list[i], self.node_list[j]) for i, j in constraint_pairs]}")
            
            # Calculate how many pairs have dependencies
            dependent_pairs = self.n_nodes*(self.n_nodes-1)//2 - len(constraint_pairs)
            print(f"  Pairs with dependencies (handled by precedence): {dependent_pairs}")
        else:
            print(f"\nNo software sequencing constraints needed (all nodes have dependencies)")
        
        return constraint_pairs
    
    def _get_software_constraint_pairs_with_levels(self):
        """
        Find constraint pairs by analyzing nodes at the same DAG level.
        
        Nodes at the same level (same distance from sources) have no dependencies between
        them and could execute simultaneously. We need constraints between all such pairs.
        
        This is equivalent to finding incomparable pairs but may be more intuitive.
        
        Returns:
            List of tuples (i, j) where i < j
        """
        # Compute levels (longest path from any source)
        levels = {}
        
        # Find all source nodes
        sources = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        
        # Initialize sources at level 0
        for source in sources:
            levels[source] = 0
        
        # Use topological order to compute levels
        for node in nx.topological_sort(self.graph):
            if node not in levels:
                # Level is max level of predecessors + 1
                pred_levels = [levels[pred] for pred in self.graph.predecessors(node)]
                levels[node] = max(pred_levels) + 1 if pred_levels else 0
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Generate constraints for all pairs within each level
        constraint_pairs = []
        
        for level, nodes in level_groups.items():
            if len(nodes) > 1:
                # All pairs within this level need sequencing constraints
                node_indices = [self.node_to_index[node] for node in nodes]
                for k in range(len(node_indices)):
                    for m in range(k + 1, len(node_indices)):
                        i, j = node_indices[k], node_indices[m]
                        if i > j:
                            i, j = j, i  # Ensure i < j
                        constraint_pairs.append((i, j))
        
        print(f"\nLevel-based software sequencing analysis:")
        print(f"  Number of levels: {len(level_groups)}")
        print(f"  Nodes per level: {[len(nodes) for nodes in level_groups.values()]}")
        print(f"  Constraint pairs needed: {len(constraint_pairs)}")
        
        for level, nodes in sorted(level_groups.items()):
            if len(nodes) > 1:
                print(f"    Level {level}: {len(nodes)} nodes → {len(nodes)*(len(nodes)-1)//2} constraints")
        
        return constraint_pairs
    
    def _compute_hierarchical_layout(self):
        """Compute hierarchical layout for DAG visualization"""
        if self.graph is None:
            raise ValueError("No graph available")
        
        # Find source nodes (no incoming edges)
        source_nodes = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        
        # If no source nodes, handle as a special case
        if not source_nodes:
            source_nodes = [min(self.graph.nodes())]  # fallback to first node
        
        # Compute levels using longest path from any source
        levels = {}
        
        # Initialize source nodes at level 0
        for node in source_nodes:
            levels[node] = 0
        
        # Use topological sort to process nodes in dependency order
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Fallback if graph has cycles
            topo_order = list(self.graph.nodes())
        
        # Assign levels based on maximum level of predecessors + 1
        for node in topo_order:
            if node not in levels:  # Not a source node
                predecessors = list(self.graph.predecessors(node))
                if predecessors:
                    # Level is max level of all predecessors + 1
                    pred_levels = [levels.get(pred, 0) for pred in predecessors]
                    levels[node] = max(pred_levels) + 1
                else:
                    levels[node] = 0  # Isolated node
        
        # Group nodes by level (vertical lines)
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Create positions
        pos = {}
        max_nodes_in_level = max(len(nodes) for nodes in level_groups.values()) if level_groups else 1
        
        for level, nodes in level_groups.items():
            x_coord = level * 3  # Horizontal spacing between levels
            n_nodes_in_level = len(nodes)
            
            # Center nodes vertically in each level
            if n_nodes_in_level == 1:
                y_coords = [0]
            else:
                # Spread nodes vertically, centered around y=0
                y_range = max(2, n_nodes_in_level - 1)
                y_coords = np.linspace(-y_range, y_range, n_nodes_in_level)
            
            # Sort nodes by ID for consistent positioning
            sorted_nodes = sorted(nodes)
            for i, node in enumerate(sorted_nodes):
                pos[node] = (x_coord, y_coords[i])
        
        return pos
    
    def display_graph(self, title: str = "Task Graph with Attributes"):
        """Display the directed graph with all attributes and edge weights"""
        if self.graph is None:
            raise ValueError("No graph to display")
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create hierarchical layout
        if not hasattr(self, '_fixed_pos') or self._fixed_pos is None:
            self._fixed_pos = self._compute_hierarchical_layout()
        pos = self._fixed_pos
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=25, arrowstyle='->', 
                              width=2, alpha=0.7, connectionstyle="arc3,rad=0.1")
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=20, font_weight='bold')
        
        # Add node attribute labels
        node_labels = {}
        for node in self.graph.nodes():
            hw_time = self.graph.nodes[node]['hardware_time']
            sw_time = self.graph.nodes[node]['software_time'] 
            area = self.graph.nodes[node]['area_cost']
            node_labels[node] = f"HW:{hw_time:.1f}\nSW:{sw_time:.1f}\nArea:{area:.1f}"
        
        # Position labels below nodes
        label_pos = {node: (pos[node][0], pos[node][1] - 0.4) for node in pos}
        
        for node, label in node_labels.items():
            plt.text(label_pos[node][0], label_pos[node][1], label, 
                    horizontalalignment='center', fontsize=20,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add edge weight labels
        edge_labels = {}
        for edge in self.graph.edges():
            comm_cost = self.graph.edges[edge]['communication_cost']
            edge_labels[edge] = f"{comm_cost:.1f}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=16)
        
        plt.title(title, fontsize=25, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig('data/test_example.png', bbox_inches='tight')
    
    def display_solution(self, solution: Dict = None):
        """
        Display the result partitions with execution sequence
        Hardware nodes in red, software nodes in blue
        """
        if self.graph is None:
            raise ValueError("No graph to display")
        
        if self.x_sol is None:
            raise ValueError("No solution available. Solve optimization first.")
        
        if solution is None:
            # Create basic solution info
            hw_nodes = [i for i in range(self.n_nodes) if self.x_sol[i] == 0]
            sw_nodes = [i for i in range(self.n_nodes) if self.x_sol[i] == 1]
            solution = {
                'hardware_nodes': hw_nodes,
                'software_nodes': sw_nodes,
                'makespan': self.T_sol
            }
        
        fig = plt.figure(figsize=(20, 15))
        
        # Use the same fixed layout as the original graph
        if not hasattr(self, '_fixed_pos') or self._fixed_pos is None:
            self._fixed_pos = self._compute_hierarchical_layout()
        pos = self._fixed_pos
        
        # Draw edges with communication costs
        edge_colors = []
        edge_widths = []
        for edge in self.graph.edges():
            edge_idx = self.edge_list.index(edge)
            if self.z_sol[edge_idx] == 1:  # Inter-partition communication
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('gray')
                edge_widths.append(2)
        
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                              arrows=True, arrowsize=25, arrowstyle='->', 
                              width=edge_widths, alpha=0.8, connectionstyle="arc3,rad=0.1")
        
        # Draw nodes with partition colors
        node_colors = []
        for node in self.graph.nodes():
            node_idx = self.node_to_index[node]
            if self.x_sol[node_idx] == 0:  # Hardware
                node_colors.append('red')
            else:  # Software
                node_colors.append('blue')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=2500, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=20, 
                               font_weight='bold', font_color='white')
        
        # Add execution time information
        for node in self.graph.nodes():
            node_idx = self.node_to_index[node]
            x_pos, y_pos = pos[node]
            start_time = self.t_sol[node_idx]
            finish_time = self.f_sol[node_idx]
            partition = "HW" if self.x_sol[node_idx] == 0 else "SW"
            
            label = f"{partition}\nStart: {start_time:.1f}\nEnd: {finish_time:.1f}"
            
            plt.text(x_pos, y_pos - 0.4, label,
                    horizontalalignment='center', fontsize=20,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Add edge communication cost labels
        edge_labels = {}
        for edge in self.graph.edges():
            edge_idx = self.edge_list.index(edge)
            comm_cost = self.c[edge_idx]
            active = "✓" if self.z_sol[edge_idx] == 1 else ""
            edge_labels[edge] = f"{comm_cost:.1f}{active}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=20)
        
        # Create legend
        hw_patch = mpatches.Patch(color='red', label='Hardware Nodes')
        sw_patch = mpatches.Patch(color='blue', label='Software Nodes')
        comm_patch = mpatches.Patch(color='red', label='Inter-partition Communication')
        no_comm_patch = mpatches.Patch(color='gray', label='Intra-partition Communication')
        
        plt.legend(handles=[hw_patch, sw_patch, comm_patch, no_comm_patch], 
                  loc='upper right', bbox_to_anchor=(1.15, 1), markerscale=3, fontsize=20)
        
        # Add execution sequence information
        hw_seq = [node for node in self.graph.nodes() if self.x_sol[self.node_to_index[node]] == 0]
        sw_seq = [node for node in self.graph.nodes() if self.x_sol[self.node_to_index[node]] == 1]
        hw_seq.sort(key=lambda node: self.t_sol[self.node_to_index[node]])
        sw_seq.sort(key=lambda node: self.t_sol[self.node_to_index[node]])
        
        info_text = f"Makespan: {solution['makespan']:.2f}\n"
        info_text += f"Hardware Execution (Parallel): {hw_seq}\n"
        info_text += f"Software Execution (Sequential): {sw_seq}\n"
        info_text += f"Total Hardware Area: {np.sum(self.a * (1 - self.x_sol)):.2f}"
        
        plt.figtext(0.7, 0.2, info_text, fontsize=20, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.title("Hardware-Software Partitioning Solution", fontsize=25, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig('data/test_result.png', bbox_inches='tight')

