"""
Neighborhood sampling for GNN mini-batch training.

Implements k-hop neighborhood sampling with size limits.
"""

import random
from typing import Dict, List, Set, Tuple
import numpy as np
import networkx as nx
from collections import defaultdict


class NeighborhoodSampler:
    """
    Sample k-hop neighborhoods for GNN training.

    In GNN training, we need to aggregate information from multi-hop neighbors.
    This class efficiently samples neighbors at each hop.
    """

    def __init__(self, graph: nx.Graph, num_hops: int = 2, sample_size: int = 10):
        """
        Initialize neighborhood sampler.

        Args:
            graph: NetworkX graph to sample from
            num_hops: Number of hops to sample (typically 2 for 2-layer GNN)
            sample_size: Maximum neighbors to sample per hop per node
        """
        self.graph = graph
        self.num_hops = num_hops
        self.sample_size = sample_size

        # Pre-compute adjacency list for faster sampling
        self._adj_list = {node: list(graph.neighbors(node)) for node in graph.nodes()}

    def sample_neighbors(self, target_nodes: List[int]) -> Dict[int, List[List[int]]]:
        """
        Sample k-hop neighborhoods for target nodes.

        Returns nested structure:
        {
            node_id: [
                [hop_0_neighbors],  # 1-hop neighbors
                [hop_1_neighbors],  # 2-hop neighbors
                ...
            ]
        }

        Example:
            >>> sampler = NeighborhoodSampler(graph, num_hops=2, sample_size=5)
            >>> neighbors = sampler.sample_neighbors([0, 1])
            >>> print(neighbors[0])
            [[2, 5, 7], [3, 8, 9, 12]]  # [1-hop neighbors, 2-hop neighbors]

        Args:
            target_nodes: List of node IDs to sample neighborhoods for

        Returns:
            Dictionary mapping node_id -> list of neighbor lists (one per hop)
        """
        result = {node: [] for node in target_nodes}

        for target in target_nodes:
            frontier = {target}  # Current nodes to expand from

            for hop in range(self.num_hops):
                hop_neighbors = []

                # Expand frontier by sampling neighbors
                for node in frontier:
                    if node not in self._adj_list:
                        continue

                    neighbors = self._adj_list[node]

                    # Sample subset if too many neighbors
                    if len(neighbors) > self.sample_size:
                        sampled = random.sample(neighbors, self.sample_size)
                    else:
                        sampled = neighbors.copy()

                    hop_neighbors.extend(sampled)

                # Remove duplicates while preserving order
                hop_neighbors = list(dict.fromkeys(hop_neighbors))

                result[target].append(hop_neighbors)

                # Next frontier is the sampled neighbors
                frontier = set(hop_neighbors)

        return result

    def sample_with_replacement(self, target_nodes: List[int]) -> Dict[int, List[List[int]]]:
        """
        Sample neighborhoods with replacement (useful for very sparse graphs).

        This allows sampling the same neighbor multiple times if a node has
        fewer neighbors than sample_size.

        Args:
            target_nodes: List of node IDs to sample neighborhoods for

        Returns:
            Dictionary mapping node_id -> list of neighbor lists (one per hop)
        """
        result = {node: [] for node in target_nodes}

        for target in target_nodes:
            frontier = {target}

            for hop in range(self.num_hops):
                hop_neighbors = []

                for node in frontier:
                    if node not in self._adj_list or not self._adj_list[node]:
                        continue

                    neighbors = self._adj_list[node]

                    # Sample with replacement
                    sampled = random.choices(neighbors, k=self.sample_size)
                    hop_neighbors.extend(sampled)

                hop_neighbors = list(dict.fromkeys(hop_neighbors))
                result[target].append(hop_neighbors)
                frontier = set(hop_neighbors)

        return result

    def get_all_sampled_nodes(self, sampled_neighbors: Dict[int, List[List[int]]]) -> Set[int]:
        """
        Get all unique nodes that were sampled (useful for feature fetching).

        Args:
            sampled_neighbors: Output from sample_neighbors()

        Returns:
            Set of all node IDs that appear in the sampled neighborhoods
        """
        all_nodes = set()

        for target, hops in sampled_neighbors.items():
            all_nodes.add(target)
            for hop_neighbors in hops:
                all_nodes.update(hop_neighbors)

        return all_nodes

    def compute_receptive_field_size(self, num_target_nodes: int = 1) -> int:
        """
        Estimate receptive field size (total nodes needed for computation).

        For a node with avg degree d, sample_size s, and num_hops h:
        Receptive field size ≈ 1 + s + s^2 + ... + s^h

        Args:
            num_target_nodes: Number of target nodes in a batch

        Returns:
            Estimated number of nodes in receptive field
        """
        # Geometric series: sum(s^i for i in 0..h)
        if self.sample_size == 1:
            receptive_field = self.num_hops + 1
        else:
            receptive_field = (self.sample_size ** (self.num_hops + 1) - 1) // (self.sample_size - 1)

        return receptive_field * num_target_nodes


class MiniBatchSampler:
    """
    Generate mini-batches with sampled neighborhoods for GNN training.

    Combines node batching with neighborhood sampling.
    """

    def __init__(self,
                 graph: nx.Graph,
                 node_features: Dict[int, np.ndarray],
                 batch_size: int = 32,
                 num_hops: int = 2,
                 sample_size: int = 10,
                 shuffle: bool = True):
        """
        Initialize mini-batch sampler.

        Args:
            graph: NetworkX graph
            node_features: Dictionary mapping node_id -> feature vector
            batch_size: Number of target nodes per batch
            num_hops: Number of hops for neighborhood sampling
            sample_size: Max neighbors to sample per hop
            shuffle: Whether to shuffle nodes at each epoch
        """
        self.graph = graph
        self.node_features = node_features
        self.batch_size = batch_size
        self.shuffle_flag = shuffle

        # Initialize neighborhood sampler
        self.neighborhood_sampler = NeighborhoodSampler(
            graph=graph,
            num_hops=num_hops,
            sample_size=sample_size
        )

        # All nodes available for sampling
        self.all_nodes = list(graph.nodes())
        self.current_idx = 0

        if self.shuffle_flag:
            random.shuffle(self.all_nodes)

    def __iter__(self):
        """Make sampler iterable"""
        return self

    def __next__(self) -> Dict:
        """
        Generate next mini-batch.

        Returns:
            Dictionary with:
            - 'target_nodes': List of node IDs in this batch
            - 'sampled_neighbors': Dict from target -> [hop0_neighbors, hop1_neighbors, ...]
            - 'all_nodes': Set of all nodes needed (for feature fetching)
        """
        if self.current_idx >= len(self.all_nodes):
            raise StopIteration

        # Get batch of target nodes
        end_idx = min(self.current_idx + self.batch_size, len(self.all_nodes))
        target_nodes = self.all_nodes[self.current_idx:end_idx]
        self.current_idx = end_idx

        # Sample neighborhoods
        sampled_neighbors = self.neighborhood_sampler.sample_neighbors(target_nodes)

        # Get all nodes that need features
        all_nodes = self.neighborhood_sampler.get_all_sampled_nodes(sampled_neighbors)

        return {
            'target_nodes': target_nodes,
            'sampled_neighbors': sampled_neighbors,
            'all_nodes': all_nodes,
        }

    def reset(self):
        """Reset iterator to beginning (optionally re-shuffle)"""
        self.current_idx = 0
        if self.shuffle_flag:
            random.shuffle(self.all_nodes)

    def __len__(self) -> int:
        """Number of batches per epoch"""
        return (len(self.all_nodes) + self.batch_size - 1) // self.batch_size


class NeighborCache:
    """
    LRU cache for neighbor embeddings (used in distributed setting).

    In distributed GNN training, fetching remote embeddings is expensive.
    This cache stores frequently accessed embeddings.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize neighbor cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache = {}  # node_id -> embedding
        self.access_order = []  # LRU tracking
        self.hits = 0
        self.misses = 0

    def get(self, node_id: int) -> np.ndarray:
        """
        Get embedding from cache.

        Args:
            node_id: Node to get embedding for

        Returns:
            Embedding if in cache, None otherwise
        """
        if node_id in self.cache:
            self.hits += 1
            # Update access order (LRU)
            self.access_order.remove(node_id)
            self.access_order.append(node_id)
            return self.cache[node_id]
        else:
            self.misses += 1
            return None

    def put(self, node_id: int, embedding: np.ndarray) -> None:
        """
        Add embedding to cache (with LRU eviction if full).

        Args:
            node_id: Node ID
            embedding: Embedding vector
        """
        # If already in cache, update it
        if node_id in self.cache:
            self.cache[node_id] = embedding
            self.access_order.remove(node_id)
            self.access_order.append(node_id)
            return

        # If cache is full, evict least recently used
        if len(self.cache) >= self.max_size:
            lru_node = self.access_order.pop(0)
            del self.cache[lru_node]

        # Add new embedding
        self.cache[node_id] = embedding
        self.access_order.append(node_id)

    def update_batch(self, embeddings: Dict[int, np.ndarray]) -> None:
        """
        Update cache with batch of embeddings.

        Args:
            embeddings: Dictionary mapping node_id -> embedding
        """
        for node_id, embedding in embeddings.items():
            self.put(node_id, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit rate, size, etc.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }

    def __len__(self) -> int:
        """Current number of cached embeddings"""
        return len(self.cache)

    def __contains__(self, node_id: int) -> bool:
        """Check if node is in cache"""
        return node_id in self.cache
