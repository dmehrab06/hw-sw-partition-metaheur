"""
Module to generate random graphs with associated software, hardware, and communication costs.
"""

import os
import sys
import random
import numpy as np
import networkx as nx
import pickle
from typing import Tuple, Dict, List

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/graph_generator.log")

logger = LogManager.get_logger(__name__)


class GraphGenerator:
    """Class to generate random graphs with cost parameters."""
    
    def __init__(self):
        """Initialize the GraphGenerator."""
        np.random.seed(42)
        self.graph = None
        self.software_costs = {}
        self.hardware_costs = {}
        self.communication_costs = {}
        logger.info("GraphGenerator initialized")
    
    def generate_graph(self, num_nodes: int, k: float, l: float, mu: float) -> nx.Graph:
        """
        Generate a random graph with the specified parameters.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            k (float): Hardware cost scale
            l (float): Hardware cost variance
            mu (float): Communication cost factor
            
        Returns:
            nx.Graph: Generated graph with cost attributes
        """
        logger.info(f"Generating random graph with {num_nodes} nodes")
        
        # Generate a random connected graph
        # Using Erdős-Rényi model with probability p=0.5 for edge creation
        self.graph = nx.erdos_renyi_graph(num_nodes, 0.5)
        
        # Ensure the graph is connected, if not, add edges to make it connected
        if not nx.is_connected(self.graph):
            logger.info("Original graph is not connected. Adding edges to ensure connectivity.")
            components = list(nx.connected_components(self.graph))
            
            # Connect all components
            for i in range(len(components) - 1):
                comp1 = list(components[i])
                comp2 = list(components[i + 1])
                # Add an edge between a random node from each component
                u = random.choice(comp1)
                v = random.choice(comp2)
                self.graph.add_edge(u, v)
                logger.debug(f"Added edge ({u}, {v}) to connect components")
        
        # Generate software costs for each node
        self.software_costs = {node: random.uniform(1, 100) for node in self.graph.nodes()}
        s_max = max(self.software_costs.values())
        logger.info(f"Generated software costs with maximum value {s_max}")
        
        # Generate hardware costs based on software costs
        self.hardware_costs = {}
        for node, s_i in self.software_costs.items():
            mean = k * s_i
            std_dev = l * k * s_i
            # Ensure hardware costs are positive
            hw_cost = max(0.1, np.random.normal(mean, std_dev))
            self.hardware_costs[node] = hw_cost
        
        logger.info("Generated hardware costs")
        
        # Generate communication costs for each edge
        self.communication_costs = {}
        for u, v in self.graph.edges():
            comm_cost = random.uniform(0, 2 * mu * s_max)
            self.communication_costs[(u, v)] = comm_cost
            self.communication_costs[(v, u)] = comm_cost  # Ensure symmetry
        
        logger.info("Generated communication costs")
        
        # Add costs as attributes to the graph
        for node in self.graph.nodes():
            self.graph.nodes[node]['software_cost'] = self.software_costs[node]
            self.graph.nodes[node]['hardware_cost'] = self.hardware_costs[node]
        
        for u, v in self.graph.edges():
            self.graph[u][v]['communication_cost'] = self.communication_costs[(u, v)]
        
        logger.info("Graph generation complete")
        return self.graph
    
    def extract_costs(self) -> Tuple[Dict[int, float], Dict[int, float], Dict[Tuple[int, int], float]]:
        """
        Extract the costs from the graph.
        
        Returns:
            Tuple containing:
                - Dict of software costs by node
                - Dict of hardware costs by node
                - Dict of communication costs by edge
        """
        if self.graph is None:
            logger.error("Cannot extract costs: Graph has not been generated yet")
            raise ValueError("Graph has not been generated yet")
        
        logger.info("Extracting costs from graph")
        return self.software_costs, self.hardware_costs, self.communication_costs
    
    @classmethod
    def extract_cost_matrices(cls, graph: nx.Graph) -> Tuple[List[float], List[float], List[List[float]]]:
        """
        Extract the costs as vectors and matrices for optimization algorithms from a graph.
        
        Args:
            graph (nx.Graph): Graph with cost attributes
            
        Returns:
            Tuple containing:
                - List of software costs
                - List of hardware costs
                - 2D matrix of communication costs
        """
        if graph is None:
            logger.error("Cannot extract cost matrices: Graph is None")
            raise ValueError("Graph is None")
        
        n = len(graph.nodes())
        
        # Create vectors for software and hardware costs
        s_costs = [0] * n
        h_costs = [0] * n
        
        # Create a matrix for communication costs
        c_costs = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fill in the vectors and matrix
        for node in graph.nodes():
            s_costs[node] = graph.nodes[node].get('software_cost', 0)
            h_costs[node] = graph.nodes[node].get('hardware_cost', 0)
        
        for u, v, data in graph.edges(data=True):
            c_costs[u][v] = data.get('communication_cost', 0)
            c_costs[v][u] = data.get('communication_cost', 0)  # Ensure symmetry
        
        logger.info("Cost matrices extraction complete")
        return s_costs, h_costs, c_costs
    
    def save_graph(self, filename: str) -> None:
        """
        Save the graph to a file.
        
        Args:
            filename (str): Path to save the graph
        """
        if self.graph is None:
            logger.error("Cannot save graph: Graph has not been generated yet")
            raise ValueError("Graph has not been generated yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save graph using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"Graph saved to {filename}")
    
    @staticmethod
    def load_graph(filename: str) -> nx.Graph:
        """
        Load a graph from a file.
        
        Args:
            filename (str): Path to load the graph from
            
        Returns:
            nx.Graph: Loaded graph
        """
        if not os.path.exists(filename):
            logger.error(f"Cannot load graph: File {filename} does not exist")
            raise FileNotFoundError(f"File {filename} does not exist")
        
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"Graph loaded from {filename}")
        return graph


# Example usage
if __name__ == "__main__":
    # Example parameters
    num_nodes = 4
    k = 1.5  # Hardware cost scale
    l = 0.2  # Hardware cost variance
    mu = 0.5  # Communication cost factor
    
    generator = GraphGenerator()
    graph = generator.generate_graph(num_nodes, k, l, mu)
    
    # Extract costs
    s_costs, h_costs, c_costs = generator.extract_costs()
    
    # Extract cost matrices
    s_matrix, h_matrix, c_matrix = generator.extract_cost_matrices(graph)
    
    # Log some information about the generated graph
    logger.info(f"Generated graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    logger.info(f"Software costs range: [{min(s_costs.values()):.2f}, {max(s_costs.values()):.2f}]")
    logger.info(f"Hardware costs range: [{min(h_costs.values()):.2f}, {max(h_costs.values()):.2f}]")
    
    comm_values = list(c_costs.values())
    logger.info(f"Communication costs range: [{min(comm_values):.2f}, {max(comm_values):.2f}]")
    
    # Save the graph
    generator.save_graph("data/example_graph_4node.pkl")