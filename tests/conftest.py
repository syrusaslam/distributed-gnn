"""
Pytest configuration and fixtures for distributed GNN tests.
"""

import pytest
import numpy as np
import networkx as nx
from typing import Dict, Tuple

from src.graph.shard import GraphShard
from src.graph.loader import generate_synthetic_graph


@pytest.fixture
def small_graph() -> nx.Graph:
    """Create a small test graph (10 nodes)"""
    graph = nx.Graph()
    edges = [
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 4), (3, 4), (3, 5), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9)
    ]
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def medium_graph() -> nx.Graph:
    """Create a medium test graph (100 nodes)"""
    graph, _, _ = generate_synthetic_graph(
        num_nodes=100,
        num_edges=500,
        graph_type='barabasi_albert'
    )
    return graph


@pytest.fixture
def test_features() -> Dict[int, np.ndarray]:
    """Create test node features (10 nodes, 8-dim features)"""
    return {
        i: np.random.randn(8).astype(np.float32)
        for i in range(10)
    }


@pytest.fixture
def test_labels() -> Dict[int, int]:
    """Create test node labels (10 nodes, 3 classes)"""
    return {i: i % 3 for i in range(10)}


@pytest.fixture
def sample_graph_shard() -> GraphShard:
    """Create a sample GraphShard for testing"""
    return GraphShard(
        worker_id=0,
        local_nodes=np.array([0, 1, 2, 3]),
        neighbor_nodes=np.array([4, 5]),
        edges=np.array([
            [0, 1], [0, 2], [1, 2],
            [1, 4], [2, 5], [3, 4]
        ]),
        node_features={
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([2.0, 3.0, 4.0]),
            2: np.array([3.0, 4.0, 5.0]),
            3: np.array([4.0, 5.0, 6.0]),
            4: np.array([5.0, 6.0, 7.0]),
            5: np.array([6.0, 7.0, 8.0]),
        }
    )


@pytest.fixture
def synthetic_dataset() -> Tuple[nx.Graph, Dict, Dict]:
    """Create a synthetic dataset for testing"""
    return generate_synthetic_graph(
        num_nodes=50,
        num_edges=200,
        feature_dim=16,
        num_classes=4,
        graph_type='erdos_renyi'
    )


# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")
