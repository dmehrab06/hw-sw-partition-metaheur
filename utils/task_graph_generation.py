import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union


def generate_random_dag(num_nodes: int, edge_probability: float = 0.3,
                        min_edges: Optional[int] = None, max_edges: Optional[int] = None) -> nx.DiGraph:
    """
    Generate a random Directed Acyclic Graph (DAG).

    Args:
        num_nodes: Number of nodes in the graph
        edge_probability: Probability of edge creation between valid node pairs
        min_edges: Minimum number of edges (if specified)
        max_edges: Maximum number of edges (if specified)

    Returns:
        A NetworkX DiGraph that is guaranteed to be acyclic
    """
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Create a random ordering of nodes to ensure acyclicity
    # (edges will only go from lower to higher indices in this ordering)
    node_ordering = list(range(num_nodes))
    random.shuffle(node_ordering)

    # Create a mapping from original node IDs to their positions in the ordering
    node_positions = {node: pos for pos, node in enumerate(node_ordering)}

    # Create edges based on probability, ensuring acyclicity
    possible_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Only add edges from lower to higher position in the ordering
            if node_positions[i] < node_positions[j]:
                possible_edges.append((i, j))

    # Shuffle possible edges to randomize selection
    random.shuffle(possible_edges)

    # Determine number of edges to create
    if min_edges is not None and max_edges is not None:
        num_edges = random.randint(min_edges, max_edges)
        edges_to_add = possible_edges[:num_edges]
    else:
        # Use probability-based approach
        edges_to_add = [edge for edge in possible_edges if random.random() < edge_probability]

    # Add the selected edges
    G.add_edges_from(edges_to_add)

    # Ensure the graph is connected by adding a minimum spanning tree if necessary
    if not nx.is_weakly_connected(G) and num_nodes > 1:
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components) - 1):
            # Connect each component to the next one
            source = random.choice(list(components[i]))
            target = random.choice(list(components[i+1]))

            # Ensure the edge direction maintains acyclicity
            if node_positions[source] > node_positions[target]:
                source, target = target, source

            G.add_edge(source, target)

    return G


def generate_node_features(G: nx.DiGraph, num_features: int = 2, sw_scale_factor: float = 10.0) -> torch.Tensor:
    """
    Generate random features for each node in the graph.

    Args:
        G: NetworkX DiGraph
        num_features: Number of features per node

    Returns:
        Tensor of shape (num_features, num_nodes) with random features
    """
    num_nodes = G.number_of_nodes()

    # Generate random features
    # Feature 1: Softwarec computation time/cost (positive values)
    sw_computation_cost = torch.rand(1, num_nodes) * sw_scale_factor  # Values between 0 and 10

    hw_computation_cost = sw_computation_cost*0.5  # Values between 0 and 10


    # Feature 2: Hardward area/cost (positive values)
    hw_area_cost = torch.rand(1, num_nodes) * 10  # Values between 0 and 5


    # Combine features
    node_features = torch.cat([sw_computation_cost, hw_area_cost, hw_computation_cost], dim=0)

    return node_features


def generate_edge_features(G: nx.DiGraph, num_features: int = 1, comm_scale_factor: float = 5.0) -> torch.Tensor:
    """
    Generate random features for each edge in the graph.

    Args:
        G: NetworkX DiGraph
        num_features: Number of features per edge

    Returns:
        Tensor of shape (num_features, num_edges) with random features
    """
    num_edges = G.number_of_edges()

    if num_edges == 0:
        return None

    # Generate random communication cost/time for each edge
    edge_features = torch.rand(num_features, num_edges) * comm_scale_factor  # Values between 0 and 5

    return edge_features


def networkx_to_pytorch(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a NetworkX DiGraph to PyTorch tensors.

    Args:
        G: NetworkX DiGraph

    Returns:
        Tuple containing:
        - Adjacency matrix as a tensor
        - Node features tensor
        - Edge features tensor
        - HW area limit as a scalar tensor
    """
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

    # Generate node features
    node_features = generate_node_features(G)

    # Generate edge features
    edge_list = list(G.edges())
    edge_features = generate_edge_features(G)

    # Generate HW area limit
    # Calculate based on the sum of node HW costs (which is the second feature)
    # with some variability to create interesting constraints
    total_hw_cost = node_features[1, :].sum().item()
    # Set the HW area limit to be between 10% and 50% of the total HW cost
    hw_area_limit = torch.tensor(total_hw_cost * (0.1 + 0.4 * random.random()), dtype=torch.float32)

    return adj_tensor, node_features, edge_features, hw_area_limit


def visualize_dag(G: nx.DiGraph, node_features: Optional[torch.Tensor] = None,
                 edge_features: Optional[torch.Tensor] = None) -> None:
    """
    Visualize a DAG with optional node and edge features.

    Args:
        G: NetworkX DiGraph
        node_features: Optional tensor of node features
        edge_features: Optional tensor of edge features
    """
    plt.figure(figsize=(10, 8))

    # Position nodes using hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)

    # Draw labels
    labels = {}
    if node_features is not None:
        for node in G.nodes():
            features = node_features[:, node].tolist()
            features_str = ", ".join([f"{f:.2f}" for f in features])
            labels[node] = f"{node}\n({features_str})"
    else:
        labels = {node: str(node) for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Draw edge labels if edge features are provided
    if edge_features is not None:
        edge_labels = {}
        for i, (u, v) in enumerate(G.edges()):
            edge_labels[(u, v)] = f"{edge_features[0, i]:.2f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Task Graph (DAG)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def create_data_lists(num_samples: int, min_nodes: int = 5, max_nodes: int = 20,
             edge_probability: float = 0.3, min_edges: Optional[int] = None,
             max_edges: Optional[int] = None):

    """
    Initialize the dataset.

    Args:
        num_samples: Number of graph samples to generate
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        edge_probability: Probability of edge creation
        min_edges: Minimum number of edges (if specified)
        max_edges: Maximum number of edges (if specified)
    """
    #self.num_samples = num_samples
    #self.min_nodes = min_nodes
    #self.max_nodes = max_nodes
    #self.edge_probability = edge_probability
    #self.min_edges = min_edges
    #self.max_edges = max_edges

    # Generate all graphs at initialization
    graphs = []
    adj_matrices = []
    node_features_list = []
    edge_features_list = []
    hw_area_limits = []

    for _ in range(num_samples):
        # Randomly determine number of nodes for this graph
        num_nodes = random.randint(min_nodes, max_nodes)

        # Generate the graph
        G = generate_random_dag(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            min_edges=min_edges,
            max_edges=max_edges
        )

        # Convert to PyTorch tensors
        adj_matrix, node_features, edge_features, hw_area_limit = networkx_to_pytorch(G)


        # Store the data
        graphs.append(G)
        adj_matrices.append(adj_matrix)
        node_features_list.append(node_features)
        edge_features_list.append(edge_features)
        hw_area_limits.append(hw_area_limit)

    return graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits


class TaskGraphDataset(Dataset):
    """
    PyTorch Dataset for task graphs.
    """
    #def __init__(self, num_samples: int, min_nodes: int = 5, max_nodes: int = 20,
    #             edge_probability: float = 0.3, min_edges: Optional[int] = None,
    #             max_edges: Optional[int] = None):
    def __init__(self, graphs,
                       adj_matrices,
                       node_features_list,
                       edge_features_list,
                       hw_area_limits):

        """
        Initialize the dataset.

        Args:
            num_samples: Number of graph samples to generate
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            edge_probability: Probability of edge creation
            min_edges: Minimum number of edges (if specified)
            max_edges: Maximum number of edges (if specified)
        """


        """
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.edge_probability = edge_probability
        self.min_edges = min_edges
        self.max_edges = max_edges

        # Generate all graphs at initialization
        self.graphs = []
        self.adj_matrices = []
        self.node_features_list = []
        self.edge_features_list = []
        self.hw_area_limits = []

        for _ in range(num_samples):
            # Randomly determine number of nodes for this graph
            num_nodes = random.randint(min_nodes, max_nodes)

            # Generate the graph
            G = generate_random_dag(
                num_nodes=num_nodes,
                edge_probability=edge_probability,
                min_edges=min_edges,
                max_edges=max_edges
            )

            # Convert to PyTorch tensors
            adj_matrix, node_features, edge_features, hw_area_limit = networkx_to_pytorch(G)


            # Store the data
            self.graphs.append(G)
            self.adj_matrices.append(adj_matrix)
            self.node_features_list.append(node_features)
            self.edge_features_list.append(edge_features)
            self.hw_area_limits.append(hw_area_limit)

        """

        #self.graphs, self.adj_matrices, self.node_features_list, self.edge_features_list, self.hw_area_limits = create_data_lists(num_samples, min_nodes, max_nodes, edge_probability, min_edges, max_edges)

        self.graphs = graphs
        self.adj_matrices = adj_matrices
        self.node_features_list = node_features_list
        self.edge_features_list = edge_features_list
        self.hw_area_limits = hw_area_limits


        self.num_samples = len(self.graphs)
        #print("self.num_samples")
        #print( self.num_samples )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
            - Adjacency matrix
            - Node features
            - Edge features
            - HW area limit as a scalar tensor
        """
        return self.graphs[idx], self.adj_matrices[idx], self.node_features_list[idx], self.edge_features_list[idx], self.hw_area_limits[idx]


def create_task_graph_dataloader(num_samples: int = 100, batch_size: int = 16,
                                 min_nodes: int = 5, max_nodes: int = 20,
                                 edge_probability: float = 0.3) -> DataLoader:
    """
    Create a DataLoader for task graphs.

    Args:
        num_samples: Number of graph samples to generate
        batch_size: Batch size for the DataLoader
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        edge_probability: Probability of edge creation

    Returns:
        PyTorch DataLoader for task graphs
    """
    dataset = TaskGraphDataset(
        num_samples=num_samples,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        edge_probability=edge_probability
    )

    # Note: We can't use standard DataLoader with batching since graphs may have different sizes
    # Instead, we'll use batch_size=1 and handle custom batching if needed
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    return dataloader


if __name__ == "__main__":
    # Example usage
    # Generate a single random DAG
    G = generate_random_dag(num_nodes=10, edge_probability=0.3)
    adj_matrix, node_features, edge_features, hw_area_limit = networkx_to_pytorch(G)

    print(f"Generated DAG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge features shape: {edge_features.shape}")
    print(f"HW area limit: {hw_area_limit.item():.2f}")

    # Visualize the graph
    visualize_dag(G, node_features, edge_features)

    # Create a dataset and dataloader
    dataset = TaskGraphDataset(num_samples=5, min_nodes=5, max_nodes=15)

    # Display information about the first sample
    adj, node_feat, edge_feat, hw_limit = dataset[0]
    print(f"\nSample from dataset:")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Node features shape: {node_feat.shape}")
    print(f"Edge features shape: {edge_feat.shape}")
    print(f"HW area limit: {hw_limit.item():.2f}")
