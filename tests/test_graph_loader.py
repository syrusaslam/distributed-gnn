"""
Tests for graph loading utilities.
"""

import pytest
import numpy as np
import networkx as nx
import tempfile
from pathlib import Path

from src.graph.loader import (
    load_edge_list,
    generate_synthetic_graph,
    save_graph,
    get_graph_statistics
)


def test_generate_synthetic_graph():
    """Test synthetic graph generation"""
    graph, features, labels = generate_synthetic_graph(
        num_nodes=50,
        num_edges=100,
        feature_dim=16,
        num_classes=5,
        graph_type='erdos_renyi'
    )

    # Check graph size
    assert graph.number_of_nodes() == 50
    assert graph.number_of_edges() > 0  # Approximately 100, but random

    # Check features
    assert len(features) == 50
    assert all(f.shape == (16,) for f in features.values())

    # Check labels
    assert len(labels) == 50
    assert all(0 <= l < 5 for l in labels.values())


def test_generate_barabasi_albert():
    """Test Barabasi-Albert graph generation"""
    graph, features, labels = generate_synthetic_graph(
        num_nodes=100,
        graph_type='barabasi_albert'
    )

    assert graph.number_of_nodes() == 100
    assert graph.number_of_edges() > 0


def test_load_edge_list():
    """Test loading graph from edge list file"""
    # Create temporary edge list file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("# Test edge list\n")
        f.write("0 1\n")
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("0 3\n")
        temp_path = f.name

    try:
        graph = load_edge_list(temp_path)

        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 4
        assert graph.has_edge(0, 1)
        assert graph.has_edge(2, 3)
    finally:
        Path(temp_path).unlink()


def test_load_weighted_edge_list():
    """Test loading weighted graph"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("0 1 0.5\n")
        f.write("1 2 1.0\n")
        f.write("2 3 0.75\n")
        temp_path = f.name

    try:
        graph = load_edge_list(temp_path, weighted=True)

        assert graph.number_of_edges() == 3
        assert graph[0][1]['weight'] == 0.5
        assert graph[1][2]['weight'] == 1.0
    finally:
        Path(temp_path).unlink()


def test_save_and_load_graph():
    """Test saving and loading graph"""
    # Generate a small graph
    graph, features, labels = generate_synthetic_graph(
        num_nodes=20,
        num_edges=40,
        feature_dim=8,
        num_classes=3
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save graph
        save_graph(graph, features, labels, tmpdir, name="test")

        # Verify files exist
        assert (Path(tmpdir) / "test_edges.txt").exists()
        assert (Path(tmpdir) / "test_features.npy").exists()
        assert (Path(tmpdir) / "test_labels.npy").exists()

        # Load features and labels back
        loaded_features = np.load(Path(tmpdir) / "test_features.npy")
        loaded_labels = np.load(Path(tmpdir) / "test_labels.npy")

        assert loaded_features.shape == (20, 8)
        assert loaded_labels.shape == (20,)


def test_get_graph_statistics(small_graph):
    """Test graph statistics computation"""
    stats = get_graph_statistics(small_graph)

    assert stats['num_nodes'] == 10
    assert stats['num_edges'] == 12
    assert 'avg_degree' in stats
    assert 'max_degree' in stats
    assert 'density' in stats
    assert stats['num_components'] >= 1


def test_statistics_on_disconnected_graph():
    """Test statistics on disconnected graph"""
    graph = nx.Graph()
    # Two disconnected components
    graph.add_edges_from([(0, 1), (1, 2)])
    graph.add_edges_from([(3, 4), (4, 5)])

    stats = get_graph_statistics(graph)

    assert stats['num_nodes'] == 6
    assert stats['num_edges'] == 4
    assert stats['num_components'] == 2
    assert stats['largest_component_size'] == 3


@pytest.mark.slow
def test_large_synthetic_graph():
    """Test generating larger synthetic graph"""
    graph, features, labels = generate_synthetic_graph(
        num_nodes=1000,
        num_edges=5000,
        feature_dim=128,
        num_classes=10
    )

    assert graph.number_of_nodes() == 1000
    assert len(features) == 1000
    assert all(f.shape == (128,) for f in features.values())


@pytest.mark.skipif(True, reason="Requires torch_geometric installation")
def test_load_cora():
    """Test loading Cora dataset (skipped by default)"""
    from src.graph.loader import load_cora_dataset

    graph, features, labels, train_mask, val_mask, test_mask = load_cora_dataset()

    # Cora has 2708 nodes
    assert graph.number_of_nodes() == 2708
    assert len(features) == 2708
    assert len(labels) == 2708

    # Check masks
    assert any(train_mask.values())
    assert any(val_mask.values())
    assert any(test_mask.values())
