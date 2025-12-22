"""
Tests for GraphShard data structure.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.graph.shard import GraphShard


def test_graph_shard_creation(sample_graph_shard):
    """Test basic GraphShard creation"""
    shard = sample_graph_shard

    assert shard.worker_id == 0
    assert len(shard.local_nodes) == 4
    assert len(shard.neighbor_nodes) == 2
    assert len(shard.edges) == 6


def test_get_neighbors(sample_graph_shard):
    """Test neighbor lookup"""
    shard = sample_graph_shard

    # Node 0 should have neighbors 1 and 2
    neighbors_0 = shard.get_neighbors(0)
    assert set(neighbors_0) == {1, 2}

    # Node 1 should have neighbors 2 and 4
    neighbors_1 = shard.get_neighbors(1)
    assert set(neighbors_1) == {2, 4}

    # Node without edges should return empty list
    neighbors_9 = shard.get_neighbors(9)
    assert neighbors_9 == []


def test_node_membership(sample_graph_shard):
    """Test node membership checks"""
    shard = sample_graph_shard

    # Local nodes
    assert shard.is_local_node(0)
    assert shard.is_local_node(1)
    assert shard.has_node(0)

    # Neighbor nodes (not local)
    assert not shard.is_local_node(4)
    assert shard.has_node(4)

    # Non-existent nodes
    assert not shard.has_node(99)
    assert not shard.is_local_node(99)


def test_get_node_feature(sample_graph_shard):
    """Test feature retrieval"""
    shard = sample_graph_shard

    # Get existing feature
    feat_0 = shard.get_node_feature(0)
    assert feat_0 is not None
    assert np.array_equal(feat_0, np.array([1.0, 2.0, 3.0]))

    # Get non-existent feature
    feat_99 = shard.get_node_feature(99)
    assert feat_99 is None


def test_statistics(sample_graph_shard):
    """Test statistics computation"""
    shard = sample_graph_shard
    stats = shard.get_statistics()

    assert stats['worker_id'] == 0
    assert stats['num_local_nodes'] == 4
    assert stats['num_neighbor_nodes'] == 2
    assert stats['num_edges'] == 6
    assert stats['avg_degree'] == 6 / 4  # 6 edges / 4 local nodes


def test_serialization():
    """Test serialization and deserialization"""
    # Create a shard
    original = GraphShard(
        worker_id=1,
        local_nodes=np.array([10, 11, 12]),
        neighbor_nodes=np.array([13, 14]),
        edges=np.array([[10, 11], [11, 12], [12, 13]]),
        node_features={
            10: np.array([1.0, 2.0]),
            11: np.array([3.0, 4.0]),
            12: np.array([5.0, 6.0]),
            13: np.array([7.0, 8.0]),
            14: np.array([9.0, 10.0]),
        }
    )

    # Serialize to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_shard.pkl"
        original.serialize(str(path))

        # Deserialize
        loaded = GraphShard.deserialize(str(path))

        # Verify equality
        assert loaded.worker_id == original.worker_id
        assert np.array_equal(loaded.local_nodes, original.local_nodes)
        assert np.array_equal(loaded.neighbor_nodes, original.neighbor_nodes)
        assert np.array_equal(loaded.edges, original.edges)

        # Verify features
        for node_id in original.node_features:
            assert np.array_equal(loaded.node_features[node_id],
                                original.node_features[node_id])

        # Verify adjacency list was rebuilt
        assert loaded.get_neighbors(10) == original.get_neighbors(10)


def test_empty_shard():
    """Test shard with no edges"""
    shard = GraphShard(
        worker_id=0,
        local_nodes=np.array([0, 1, 2]),
        neighbor_nodes=np.array([]),
        edges=np.array([]).reshape(0, 2),  # Empty edge array
        node_features={
            0: np.array([1.0]),
            1: np.array([2.0]),
            2: np.array([3.0]),
        }
    )

    assert shard.num_edges() == 0
    assert shard.num_local_nodes() == 3
    assert shard.get_neighbors(0) == []


def test_shard_repr(sample_graph_shard):
    """Test string representation"""
    shard = sample_graph_shard
    repr_str = repr(shard)

    assert "GraphShard" in repr_str
    assert "worker_id=0" in repr_str
    assert "local_nodes=4" in repr_str
    assert "edges=6" in repr_str
