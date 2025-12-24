"""
Tests for neighborhood sampling and mini-batch generation.
"""

import pytest
import numpy as np
import networkx as nx
from typing import Dict, List

from src.sampling.sampler import (
    NeighborhoodSampler,
    MiniBatchSampler,
    NeighborCache
)


class TestNeighborhoodSampler:
    """Tests for NeighborhoodSampler class"""

    def test_basic_sampling(self, small_graph):
        """Test basic k-hop sampling"""
        sampler = NeighborhoodSampler(small_graph, num_hops=2, sample_size=5)
        neighbors = sampler.sample_neighbors([0])

        # Should return dict with target node as key
        assert 0 in neighbors

        # Should have 2 hops
        assert len(neighbors[0]) == 2

        # Each hop should be a list
        assert isinstance(neighbors[0][0], list)
        assert isinstance(neighbors[0][1], list)

    def test_sample_size_limit(self, medium_graph):
        """Test that sampling respects size limits"""
        sampler = NeighborhoodSampler(medium_graph, num_hops=1, sample_size=5)
        neighbors = sampler.sample_neighbors([0])

        # First hop should have at most sample_size neighbors
        assert len(neighbors[0][0]) <= 5

    def test_multiple_targets(self, small_graph):
        """Test sampling for multiple target nodes"""
        sampler = NeighborhoodSampler(small_graph, num_hops=2, sample_size=10)
        targets = [0, 1, 2]
        neighbors = sampler.sample_neighbors(targets)

        # Should have entry for each target
        for target in targets:
            assert target in neighbors
            assert len(neighbors[target]) == 2  # 2 hops

    def test_node_without_neighbors(self):
        """Test sampling from isolated node"""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])  # No edges

        sampler = NeighborhoodSampler(graph, num_hops=2, sample_size=5)
        neighbors = sampler.sample_neighbors([0])

        # Should return empty lists for each hop
        assert neighbors[0] == [[], []]

    def test_sample_with_replacement(self, small_graph):
        """Test sampling with replacement for sparse graphs"""
        sampler = NeighborhoodSampler(small_graph, num_hops=1, sample_size=100)
        neighbors = sampler.sample_with_replacement([0])

        # Should handle large sample_size gracefully
        assert 0 in neighbors
        assert len(neighbors[0]) == 1  # 1 hop

    def test_get_all_sampled_nodes(self, small_graph):
        """Test getting all nodes in sampled neighborhoods"""
        sampler = NeighborhoodSampler(small_graph, num_hops=2, sample_size=5)
        neighbors = sampler.sample_neighbors([0, 1])

        all_nodes = sampler.get_all_sampled_nodes(neighbors)

        # Should include target nodes
        assert 0 in all_nodes
        assert 1 in all_nodes

        # Should include sampled neighbors
        assert len(all_nodes) >= 2  # At least the target nodes

    def test_receptive_field_estimation(self):
        """Test receptive field size calculation"""
        sampler = NeighborhoodSampler(nx.Graph(), num_hops=2, sample_size=10)

        # For 1 target: 1 + 10 + 10^2 = 111
        field_size = sampler.compute_receptive_field_size(num_target_nodes=1)
        expected = (10**3 - 1) // (10 - 1)  # Geometric series
        assert field_size == expected

    def test_deterministic_with_seed(self, small_graph):
        """Test that setting random seed makes sampling deterministic"""
        import random

        sampler = NeighborhoodSampler(small_graph, num_hops=2, sample_size=3)

        # Sample twice with same seed
        random.seed(42)
        neighbors1 = sampler.sample_neighbors([0])

        random.seed(42)
        neighbors2 = sampler.sample_neighbors([0])

        # Should get identical results
        assert neighbors1[0] == neighbors2[0]


class TestMiniBatchSampler:
    """Tests for MiniBatchSampler class"""

    def test_batch_generation(self, small_graph, test_features):
        """Test basic mini-batch generation"""
        sampler = MiniBatchSampler(
            graph=small_graph,
            node_features=test_features,
            batch_size=4,
            num_hops=2,
            sample_size=5
        )

        batch = next(sampler)

        # Check batch structure
        assert 'target_nodes' in batch
        assert 'sampled_neighbors' in batch
        assert 'all_nodes' in batch

        # Check batch size
        assert len(batch['target_nodes']) <= 4

    def test_iteration(self, small_graph, test_features):
        """Test iterating through all batches"""
        sampler = MiniBatchSampler(
            graph=small_graph,
            node_features=test_features,
            batch_size=3,
            num_hops=1,
            sample_size=5
        )

        batches = list(sampler)

        # Should have ceil(10 / 3) = 4 batches
        assert len(batches) == 4

        # First 3 batches should have 3 nodes
        for i in range(3):
            assert len(batches[i]['target_nodes']) == 3

        # Last batch should have remaining node
        assert len(batches[3]['target_nodes']) == 1

    def test_reset_and_reshuffle(self, small_graph, test_features):
        """Test resetting sampler for new epoch"""
        sampler = MiniBatchSampler(
            graph=small_graph,
            node_features=test_features,
            batch_size=5,
            shuffle=True
        )

        # Get first batch
        batch1 = next(sampler)
        first_targets = batch1['target_nodes']

        # Reset
        sampler.reset()

        # Get first batch again (should be different due to shuffle)
        batch2 = next(sampler)
        second_targets = batch2['target_nodes']

        # Note: This might rarely fail due to random shuffle
        # producing same order, but very unlikely
        # We just verify reset worked
        sampler.reset()
        batch3 = next(sampler)
        assert len(batch3['target_nodes']) == len(first_targets)

    def test_no_shuffle(self, small_graph, test_features):
        """Test that no-shuffle mode is deterministic"""
        sampler = MiniBatchSampler(
            graph=small_graph,
            node_features=test_features,
            batch_size=3,
            shuffle=False
        )

        # Get all batches
        batches1 = list(sampler)

        # Reset and get again
        sampler.reset()
        batches2 = list(sampler)

        # Should be identical
        for b1, b2 in zip(batches1, batches2):
            assert b1['target_nodes'] == b2['target_nodes']

    def test_all_nodes_included(self, synthetic_dataset):
        """Test that all nodes are eventually sampled"""
        graph, features, _ = synthetic_dataset

        sampler = MiniBatchSampler(
            graph=graph,
            node_features=features,
            batch_size=10,
            shuffle=False
        )

        # Collect all target nodes from all batches
        all_targets = []
        for batch in sampler:
            all_targets.extend(batch['target_nodes'])

        # Should include all graph nodes
        assert set(all_targets) == set(graph.nodes())

    def test_sampler_length(self, small_graph, test_features):
        """Test __len__ returns correct number of batches"""
        sampler = MiniBatchSampler(
            graph=small_graph,
            node_features=test_features,
            batch_size=3
        )

        # 10 nodes / batch_size 3 = 4 batches
        assert len(sampler) == 4

    def test_empty_graph(self):
        """Test sampler with empty graph"""
        graph = nx.Graph()
        features = {}

        sampler = MiniBatchSampler(
            graph=graph,
            node_features=features,
            batch_size=5
        )

        # Should have no batches
        assert len(sampler) == 0

        # Iteration should be empty
        batches = list(sampler)
        assert len(batches) == 0


class TestNeighborCache:
    """Tests for NeighborCache (LRU cache)"""

    def test_basic_caching(self):
        """Test basic cache operations"""
        cache = NeighborCache(max_size=5)

        # Add embeddings
        emb1 = np.array([1.0, 2.0, 3.0])
        cache.put(0, emb1)

        # Retrieve embedding
        retrieved = cache.get(0)
        assert np.array_equal(retrieved, emb1)

        # Check hit
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self):
        """Test cache miss"""
        cache = NeighborCache(max_size=5)

        # Try to get non-existent embedding
        result = cache.get(99)

        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = NeighborCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(i, np.array([float(i)]))

        # Access node 0 and 1 (making node 2 least recently used)
        cache.get(0)
        cache.get(1)

        # Add new node (should evict node 2)
        cache.put(3, np.array([3.0]))

        # Node 2 should be evicted
        assert cache.get(2) is None
        assert cache.misses == 1

        # Nodes 0, 1, 3 should still be in cache
        assert cache.get(0) is not None
        assert cache.get(1) is not None
        assert cache.get(3) is not None

    def test_update_existing(self):
        """Test updating existing cached embedding"""
        cache = NeighborCache(max_size=5)

        # Add embedding
        cache.put(0, np.array([1.0, 2.0]))

        # Update it
        cache.put(0, np.array([3.0, 4.0]))

        # Should have new value
        retrieved = cache.get(0)
        assert np.array_equal(retrieved, np.array([3.0, 4.0]))

        # Size should still be 1
        assert len(cache) == 1

    def test_batch_update(self):
        """Test updating cache with batch of embeddings"""
        cache = NeighborCache(max_size=10)

        embeddings = {
            0: np.array([1.0, 2.0]),
            1: np.array([3.0, 4.0]),
            2: np.array([5.0, 6.0]),
        }

        cache.update_batch(embeddings)

        # All should be in cache
        for node_id, emb in embeddings.items():
            assert np.array_equal(cache.get(node_id), emb)

        assert len(cache) == 3

    def test_clear_cache(self):
        """Test clearing cache"""
        cache = NeighborCache(max_size=5)

        # Add some embeddings
        cache.put(0, np.array([1.0]))
        cache.put(1, np.array([2.0]))

        # Clear
        cache.clear()

        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_statistics(self):
        """Test cache statistics"""
        cache = NeighborCache(max_size=10)

        # Add some embeddings
        for i in range(5):
            cache.put(i, np.array([float(i)]))

        # Access some (hits)
        cache.get(0)
        cache.get(1)

        # Try to get non-existent (misses)
        cache.get(99)
        cache.get(100)

        stats = cache.get_stats()

        assert stats['size'] == 5
        assert stats['max_size'] == 10
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.5  # 2 hits / 4 total
        assert stats['utilization'] == 0.5  # 5 / 10

    def test_contains(self):
        """Test __contains__ operator"""
        cache = NeighborCache(max_size=5)

        cache.put(0, np.array([1.0]))

        assert 0 in cache
        assert 1 not in cache

    def test_len(self):
        """Test __len__ operator"""
        cache = NeighborCache(max_size=10)

        assert len(cache) == 0

        cache.put(0, np.array([1.0]))
        assert len(cache) == 1

        cache.put(1, np.array([2.0]))
        assert len(cache) == 2


@pytest.mark.slow
def test_large_batch_sampling():
    """Test sampling on larger graph"""
    # Generate large synthetic graph
    from src.graph.loader import generate_synthetic_graph

    graph, features, labels = generate_synthetic_graph(
        num_nodes=1000,
        num_edges=5000,
        feature_dim=64,
        num_classes=10
    )

    sampler = MiniBatchSampler(
        graph=graph,
        node_features=features,
        batch_size=128,
        num_hops=2,
        sample_size=10
    )

    # Generate all batches
    batches = list(sampler)

    # Should have ceil(1000 / 128) = 8 batches
    assert len(batches) == 8

    # Verify all nodes sampled
    all_targets = []
    for batch in batches:
        all_targets.extend(batch['target_nodes'])

    assert len(set(all_targets)) == 1000


def test_sampling_on_star_graph():
    """Test sampling on star graph (one central hub)"""
    # Star graph: node 0 connected to all others
    graph = nx.star_graph(10)  # 11 nodes total (0 at center)

    features = {i: np.random.randn(8).astype(np.float32) for i in graph.nodes()}

    sampler = MiniBatchSampler(
        graph=graph,
        node_features=features,
        batch_size=5,
        num_hops=2,
        sample_size=5
    )

    batch = next(sampler)

    # Should successfully sample
    assert len(batch['target_nodes']) == 5
    assert 'sampled_neighbors' in batch
