"""
Graph loading utilities for distributed GNN training.

Supports loading from various formats and datasets.
"""

from typing import Dict, Tuple, Optional
import networkx as nx
import numpy as np
from pathlib import Path


def load_edge_list(path: str, weighted: bool = False, directed: bool = False) -> nx.Graph:
    """
    Load graph from edge list file.

    Format: Each line is "source target [weight]"

    Args:
        path: Path to edge list file
        weighted: Whether edges have weights
        directed: Whether graph is directed

    Returns:
        NetworkX graph
    """
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source, target = int(parts[0]), int(parts[1])

            if weighted and len(parts) >= 3:
                weight = float(parts[2])
                graph.add_edge(source, target, weight=weight)
            else:
                graph.add_edge(source, target)

    return graph


def load_cora_dataset(data_dir: str = "data/cora") -> Tuple[nx.Graph, Dict, Dict, Dict, Dict, Dict]:
    """
    Load Cora citation network dataset.

    Returns:
        graph: NetworkX graph
        features: Dict mapping node_id -> feature vector
        labels: Dict mapping node_id -> class label
        train_mask: Dict mapping node_id -> bool (True if in training set)
        val_mask: Dict mapping node_id -> bool (True if in validation set)
        test_mask: Dict mapping node_id -> bool (True if in test set)
    """
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError:
        raise ImportError("torch_geometric is required to load Cora dataset. "
                         "Install with: pip install torch-geometric")

    # Download/load Cora dataset
    data_path = Path(data_dir).parent
    dataset = Planetoid(root=str(data_path), name='Cora')
    data = dataset[0]

    # Convert to NetworkX graph
    edge_index = data.edge_index.numpy()
    graph = nx.Graph()

    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        graph.add_edge(int(source), int(target))

    # Extract features
    features_array = data.x.numpy()
    features = {i: features_array[i] for i in range(len(features_array))}

    # Extract labels
    labels_array = data.y.numpy()
    labels = {i: int(labels_array[i]) for i in range(len(labels_array))}

    # Extract masks
    train_mask = {i: bool(data.train_mask[i]) for i in range(len(data.train_mask))}
    val_mask = {i: bool(data.val_mask[i]) for i in range(len(data.val_mask))}
    test_mask = {i: bool(data.test_mask[i]) for i in range(len(data.test_mask))}

    return graph, features, labels, train_mask, val_mask, test_mask


def generate_synthetic_graph(num_nodes: int = 1000,
                            num_edges: int = 5000,
                            feature_dim: int = 64,
                            num_classes: int = 5,
                            graph_type: str = 'erdos_renyi') -> Tuple[nx.Graph, Dict, Dict]:
    """
    Generate synthetic graph for testing.

    Args:
        num_nodes: Number of nodes
        num_edges: Approximate number of edges
        feature_dim: Dimension of node features
        num_classes: Number of classes for node classification
        graph_type: Type of graph ('erdos_renyi', 'barabasi_albert', 'watts_strogatz')

    Returns:
        graph: NetworkX graph
        features: Dict mapping node_id -> feature vector
        labels: Dict mapping node_id -> class label
    """
    # Generate graph structure
    if graph_type == 'erdos_renyi':
        p = num_edges / (num_nodes * (num_nodes - 1) / 2)
        graph = nx.erdos_renyi_graph(num_nodes, p)
    elif graph_type == 'barabasi_albert':
        m = max(1, num_edges // num_nodes)
        graph = nx.barabasi_albert_graph(num_nodes, m)
    elif graph_type == 'watts_strogatz':
        k = max(2, num_edges // num_nodes)
        p = 0.3
        graph = nx.watts_strogatz_graph(num_nodes, k, p)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Generate random features
    features = {
        i: np.random.randn(feature_dim).astype(np.float32)
        for i in range(num_nodes)
    }

    # Generate random labels
    labels = {
        i: np.random.randint(0, num_classes)
        for i in range(num_nodes)
    }

    return graph, features, labels


def save_graph(graph: nx.Graph, features: Dict, labels: Dict,
              output_dir: str, name: str = "graph") -> None:
    """
    Save graph, features, and labels to disk.

    Args:
        graph: NetworkX graph
        features: Node features
        labels: Node labels
        output_dir: Directory to save to
        name: Name prefix for saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save edge list
    edge_list_path = output_path / f"{name}_edges.txt"
    nx.write_edgelist(graph, edge_list_path, data=False)

    # Save features
    features_path = output_path / f"{name}_features.npy"
    feature_array = np.array([features[i] for i in sorted(features.keys())])
    np.save(features_path, feature_array)

    # Save labels
    labels_path = output_path / f"{name}_labels.npy"
    label_array = np.array([labels[i] for i in sorted(labels.keys())])
    np.save(labels_path, label_array)

    print(f"Saved graph to {output_dir}:")
    print(f"  - Edges: {edge_list_path}")
    print(f"  - Features: {features_path}")
    print(f"  - Labels: {labels_path}")


def get_graph_statistics(graph: nx.Graph) -> Dict:
    """
    Compute statistics about a graph.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary of graph statistics
    """
    degrees = [d for _, d in graph.degree()]

    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'avg_degree': np.mean(degrees),
        'max_degree': np.max(degrees),
        'min_degree': np.min(degrees),
        'density': nx.density(graph),
    }

    # Add connected components info
    if isinstance(graph, nx.Graph):
        stats['num_components'] = nx.number_connected_components(graph)
        stats['largest_component_size'] = len(max(nx.connected_components(graph), key=len))

    return stats
