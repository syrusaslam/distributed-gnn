"""
Graph shard data structure for distributed GNN training.

A GraphShard represents a partition of the graph assigned to a worker.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np
import pickle
from pathlib import Path


@dataclass
class GraphShard:
    """
    Represents a partition of the graph owned by a worker.

    In edge-cut partitioning:
    - Each edge is assigned to exactly one worker
    - Nodes are replicated across workers where they appear as neighbors
    - local_nodes: nodes whose edges are owned by this worker
    - neighbor_nodes: nodes referenced by edges but owned by other workers
    """

    worker_id: int
    local_nodes: np.ndarray  # Node IDs owned by this worker, shape (num_local_nodes,)
    neighbor_nodes: np.ndarray  # Node IDs referenced but not owned, shape (num_neighbor_nodes,)
    edges: np.ndarray  # Edge tuples (source, target), shape (num_edges, 2)
    node_features: Dict[int, np.ndarray]  # Mapping node_id -> feature vector

    def __post_init__(self):
        """Validate shard consistency"""
        # Ensure arrays are numpy arrays
        if not isinstance(self.local_nodes, np.ndarray):
            self.local_nodes = np.array(self.local_nodes)
        if not isinstance(self.neighbor_nodes, np.ndarray):
            self.neighbor_nodes = np.array(self.neighbor_nodes)
        if not isinstance(self.edges, np.ndarray):
            self.edges = np.array(self.edges)

        # Build adjacency list for fast neighbor lookup
        self._build_adjacency_list()

    def _build_adjacency_list(self) -> None:
        """Build adjacency list from edges for O(1) neighbor lookup"""
        self._adj_list = {}
        for source, target in self.edges:
            if source not in self._adj_list:
                self._adj_list[source] = []
            self._adj_list[source].append(int(target))

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Return list of neighbor node IDs for given node.

        Args:
            node_id: The node to get neighbors for

        Returns:
            List of neighbor node IDs (empty list if node has no neighbors)
        """
        return self._adj_list.get(node_id, [])

    def get_node_feature(self, node_id: int) -> Optional[np.ndarray]:
        """
        Get feature vector for a node.

        Args:
            node_id: The node to get features for

        Returns:
            Feature vector or None if node not in shard
        """
        return self.node_features.get(node_id)

    def has_node(self, node_id: int) -> bool:
        """Check if node exists in this shard (local or neighbor)"""
        return (node_id in self.local_nodes) or (node_id in self.neighbor_nodes)

    def is_local_node(self, node_id: int) -> bool:
        """Check if node is owned by this shard"""
        return node_id in self.local_nodes

    def num_local_nodes(self) -> int:
        """Number of nodes owned by this shard"""
        return len(self.local_nodes)

    def num_edges(self) -> int:
        """Number of edges owned by this shard"""
        return len(self.edges)

    def num_neighbor_nodes(self) -> int:
        """Number of neighbor nodes (replicated from other shards)"""
        return len(self.neighbor_nodes)

    def get_statistics(self) -> Dict:
        """
        Get statistics about this shard.

        Returns:
            Dictionary with shard statistics
        """
        return {
            'worker_id': self.worker_id,
            'num_local_nodes': self.num_local_nodes(),
            'num_neighbor_nodes': self.num_neighbor_nodes(),
            'num_edges': self.num_edges(),
            'replication_factor': self.num_neighbor_nodes() / max(self.num_local_nodes(), 1),
            'avg_degree': self.num_edges() / max(self.num_local_nodes(), 1),
        }

    def serialize(self, path: str) -> None:
        """
        Save shard to disk.

        Args:
            path: File path to save to (will create parent directories if needed)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize(cls, path: str) -> 'GraphShard':
        """
        Load shard from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded GraphShard instance
        """
        with open(path, 'rb') as f:
            shard = pickle.load(f)

        # Rebuild adjacency list (not pickled)
        if not hasattr(shard, '_adj_list'):
            shard._build_adjacency_list()

        return shard

    def __repr__(self) -> str:
        """String representation of shard"""
        return (f"GraphShard(worker_id={self.worker_id}, "
                f"local_nodes={self.num_local_nodes()}, "
                f"neighbor_nodes={self.num_neighbor_nodes()}, "
                f"edges={self.num_edges()})")
