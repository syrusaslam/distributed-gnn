# Distributed GNN Implementation Plan
## Detailed Task Breakdown for Solo Developer

---

## Quick Start Checklist (Week 1)

- [ ] Project setup and environment configuration
- [ ] Install core dependencies
- [ ] Create directory structure
- [ ] Download Cora dataset
- [ ] Implement basic graph loading
- [ ] Run "Hello World" GNN (single forward pass)

---

## Phase 1: Single-Machine Baseline (Weeks 1-2)

### Goal
Build a working GNN training pipeline on a single machine to establish a baseline before adding distributed complexity.

### Tasks

#### 1.1 Project Setup (Day 1)

**Tasks:**
- [ ] Initialize Git repository
- [ ] Create virtual environment (`python3 -m venv venv`)
- [ ] Create directory structure:
  ```
  distributed_gnn/
  ├── src/
  │   ├── graph/
  │   ├── model/
  │   ├── sampling/
  │   ├── worker/
  │   ├── coordinator/
  │   └── utils/
  ├── tests/
  ├── examples/
  ├── scripts/
  ├── data/
  ├── checkpoints/
  └── logs/
  ```
- [ ] Create `requirements.txt`
- [ ] Create `.gitignore` (include venv/, data/, checkpoints/, logs/)
- [ ] Create `README.md` with project overview

**Dependencies (`requirements.txt`):**
```txt
torch>=2.0.0
torch-geometric>=2.3.0
networkx>=3.0
numpy>=1.24.0
scikit-learn>=1.3.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
grpcio>=1.60.0
grpcio-tools>=1.60.0
protobuf>=4.25.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=2.0.0
```

**Time estimate:** 1-2 hours

**Validation:**
- `pytest` runs without errors (even with no tests)
- All directories created
- Git repository initialized

---

#### 1.2 Graph Loading and Data Structures (Days 1-2)

**Tasks:**
- [ ] Implement `GraphShard` dataclass (`src/graph/shard.py`)
  - Fields: worker_id, local_nodes, neighbor_nodes, edges, node_features
  - Methods: serialize(), deserialize(), get_neighbors()
- [ ] Implement graph loading from edge list (`src/graph/loader.py`)
  - Load from CSV/TXT format: `source,target,weight`
  - Convert to NetworkX graph
  - Extract node features (if available)
- [ ] Create synthetic graph generator (`scripts/generate_synthetic.py`)
  - Use NetworkX: `nx.erdos_renyi_graph()`, `nx.barabasi_albert_graph()`
  - Add random node features
- [ ] Download Cora dataset (`scripts/download_datasets.py`)
  - Use PyTorch Geometric: `from torch_geometric.datasets import Planetoid`
  - Convert to NetworkX format
  - Save as edge list

**Code skeleton (`src/graph/shard.py`):**
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pickle

@dataclass
class GraphShard:
    worker_id: int
    local_nodes: np.ndarray  # Node IDs owned by this worker
    neighbor_nodes: np.ndarray  # Referenced but not owned
    edges: np.ndarray  # shape (num_edges, 2)
    node_features: Dict[int, np.ndarray]  # node_id -> feature vector

    def get_neighbors(self, node_id: int) -> List[int]:
        """Return list of neighbor node IDs for given node"""
        # Find all edges where source == node_id
        mask = self.edges[:, 0] == node_id
        neighbors = self.edges[mask, 1].tolist()
        return neighbors

    def serialize(self, path: str) -> None:
        """Save shard to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(cls, path: str) -> 'GraphShard':
        """Load shard from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)
```

**Time estimate:** 4-6 hours

**Validation:**
- Load Cora dataset successfully
- Print graph statistics: # nodes, # edges, feature dimension
- Verify GraphShard can serialize/deserialize without data loss

**Test file (`tests/test_graph_loading.py`):**
```python
def test_load_cora():
    graph = load_cora_dataset()
    assert graph.number_of_nodes() == 2708
    assert graph.number_of_edges() == 10556  # Undirected, so 5278*2

def test_graph_shard_serialization():
    shard = create_test_shard()
    shard.serialize('/tmp/test_shard.pkl')
    loaded = GraphShard.deserialize('/tmp/test_shard.pkl')
    assert np.array_equal(shard.local_nodes, loaded.local_nodes)
    assert np.array_equal(shard.edges, loaded.edges)
```

---

#### 1.3 GraphSAGE Model Implementation (Days 2-3)

**Tasks:**
- [ ] Implement `SAGELayer` (`src/model/layers.py`)
  - MLP with mean aggregation
  - Handle variable number of neighbors
- [ ] Implement `GraphSAGE` model (`src/model/graphsage.py`)
  - Stack 2-3 SAGELayers
  - Support multi-layer forward pass
  - Implement get_embeddings() for inference
- [ ] Add model initialization helpers
  - Xavier/Kaiming initialization
  - Layer normalization (optional)

**Code skeleton (`src/model/graphsage.py`):**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, aggregator: str = 'mean'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator

        # Transformation for concatenated (self + neighbor) features
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_features: torch.Tensor,
                neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        node_features: (batch_size, in_dim)
        neighbor_features: (batch_size, num_neighbors, in_dim)

        Returns: (batch_size, out_dim)
        """
        # Aggregate neighbors (mean pooling)
        if self.aggregator == 'mean':
            neighbor_agg = neighbor_features.mean(dim=1)  # (batch_size, in_dim)
        elif self.aggregator == 'max':
            neighbor_agg = neighbor_features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Concatenate self features with aggregated neighbor features
        combined = torch.cat([node_features, neighbor_agg], dim=1)  # (batch_size, in_dim*2)

        # Apply linear transformation and activation
        out = self.linear(combined)
        out = F.relu(out)

        return out

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(SAGELayer(in_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGELayer(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(SAGELayer(hidden_dim, out_dim))

    def forward(self, batch_data: Dict) -> torch.Tensor:
        """
        batch_data contains:
        - 'target_features': (batch_size, in_dim) - features of target nodes
        - 'neighbor_features': List of (batch_size, num_neighbors, dim) for each layer

        Returns: (batch_size, out_dim) - embeddings for target nodes
        """
        x = batch_data['target_features']

        for i, layer in enumerate(self.layers):
            neighbor_feats = batch_data['neighbor_features'][i]
            x = layer(x, neighbor_feats)

            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
```

**Time estimate:** 6-8 hours

**Validation:**
- Forward pass produces correct output shape
- Gradients flow correctly (run dummy backward pass)
- No NaN/Inf values in outputs

**Test file (`tests/test_model.py`):**
```python
def test_sage_layer_forward():
    layer = SAGELayer(in_dim=64, out_dim=32)
    node_feat = torch.randn(16, 64)  # batch_size=16
    neighbor_feat = torch.randn(16, 10, 64)  # 10 neighbors per node

    output = layer(node_feat, neighbor_feat)
    assert output.shape == (16, 32)

def test_graphsage_forward():
    model = GraphSAGE(in_dim=1433, hidden_dim=128, out_dim=7, num_layers=2)
    batch_data = {
        'target_features': torch.randn(32, 1433),
        'neighbor_features': [
            torch.randn(32, 10, 1433),  # Layer 0 neighbors
            torch.randn(32, 10, 128),   # Layer 1 neighbors
        ]
    }

    output = model(batch_data)
    assert output.shape == (32, 7)
```

---

#### 1.4 Mini-Batch Sampling (Days 3-4)

**Tasks:**
- [ ] Implement k-hop neighborhood sampling (`src/sampling/sampler.py`)
  - BFS/DFS to find k-hop neighbors
  - Random sampling if too many neighbors
  - Handle edge cases (nodes with no neighbors)
- [ ] Implement `MiniBatchSampler` class
  - Generate batches of target nodes
  - Sample neighbors for each target
  - Prepare data format for model
- [ ] Add data collation helpers
  - Convert sampled neighbors to tensors
  - Handle variable-length neighbor lists (padding)

**Code skeleton (`src/sampling/sampler.py`):**
```python
import random
from typing import Dict, List, Set
import numpy as np
import networkx as nx

class NeighborhoodSampler:
    def __init__(self, graph: nx.Graph, num_hops: int = 2, sample_size: int = 10):
        self.graph = graph
        self.num_hops = num_hops
        self.sample_size = sample_size

    def sample_neighbors(self, target_nodes: List[int]) -> Dict[int, List[List[int]]]:
        """
        Sample k-hop neighborhoods for target nodes

        Returns: Dict mapping node_id -> [hop0_neighbors, hop1_neighbors, ...]
        """
        result = {node: [] for node in target_nodes}

        for target in target_nodes:
            frontier = {target}

            for hop in range(self.num_hops):
                next_frontier = set()
                hop_neighbors = []

                for node in frontier:
                    neighbors = list(self.graph.neighbors(node))

                    # Sample subset if too many neighbors
                    if len(neighbors) > self.sample_size:
                        neighbors = random.sample(neighbors, self.sample_size)

                    hop_neighbors.extend(neighbors)
                    next_frontier.update(neighbors)

                result[target].append(hop_neighbors)
                frontier = next_frontier

        return result

class MiniBatchSampler:
    def __init__(self, graph: nx.Graph, node_features: Dict[int, np.ndarray],
                 batch_size: int = 32, num_hops: int = 2, sample_size: int = 10):
        self.graph = graph
        self.node_features = node_features
        self.batch_size = batch_size
        self.sampler = NeighborhoodSampler(graph, num_hops, sample_size)

        # All nodes available for sampling
        self.all_nodes = list(graph.nodes())
        self.current_idx = 0
        self.shuffle()

    def shuffle(self):
        """Shuffle nodes at epoch start"""
        random.shuffle(self.all_nodes)
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        """Generate next mini-batch"""
        if self.current_idx >= len(self.all_nodes):
            self.shuffle()
            raise StopIteration

        # Get batch of target nodes
        end_idx = min(self.current_idx + self.batch_size, len(self.all_nodes))
        target_nodes = self.all_nodes[self.current_idx:end_idx]
        self.current_idx = end_idx

        # Sample neighbors
        neighbors_dict = self.sampler.sample_neighbors(target_nodes)

        # Prepare batch data
        batch = self._collate_batch(target_nodes, neighbors_dict)

        return batch

    def _collate_batch(self, target_nodes: List[int],
                      neighbors_dict: Dict[int, List[List[int]]]) -> Dict:
        """Convert sampled data to tensors"""
        import torch

        # Target node features
        target_features = torch.tensor(
            [self.node_features[node] for node in target_nodes],
            dtype=torch.float32
        )

        # Neighbor features for each hop (need to handle variable lengths)
        neighbor_features = []
        for hop in range(self.sampler.num_hops):
            hop_features = []
            for node in target_nodes:
                hop_neighbors = neighbors_dict[node][hop]

                # Get features for neighbors
                if hop_neighbors:
                    feats = [self.node_features[n] for n in hop_neighbors[:self.sampler.sample_size]]
                else:
                    # If no neighbors, use zero features
                    feats = [np.zeros_like(list(self.node_features.values())[0])]

                # Pad to sample_size
                while len(feats) < self.sampler.sample_size:
                    feats.append(np.zeros_like(feats[0]))

                hop_features.append(np.stack(feats))

            neighbor_features.append(torch.tensor(np.stack(hop_features), dtype=torch.float32))

        return {
            'target_nodes': target_nodes,
            'target_features': target_features,
            'neighbor_features': neighbor_features
        }
```

**Time estimate:** 6-8 hours

**Validation:**
- Sample k-hop neighbors correctly
- Batch shapes match model expectations
- No out-of-bounds node IDs

**Test file (`tests/test_sampling.py`):**
```python
def test_neighborhood_sampling():
    graph = create_test_graph(num_nodes=100)
    sampler = NeighborhoodSampler(graph, num_hops=2, sample_size=5)

    neighbors = sampler.sample_neighbors([0, 1])
    assert 0 in neighbors
    assert len(neighbors[0]) == 2  # 2 hops
    assert len(neighbors[0][0]) <= 5  # At most 5 neighbors per hop

def test_minibatch_sampler():
    graph = create_test_graph(num_nodes=100)
    features = {i: np.random.randn(64) for i in graph.nodes()}
    sampler = MiniBatchSampler(graph, features, batch_size=16)

    batch = next(sampler)
    assert batch['target_features'].shape == (16, 64)
    assert len(batch['neighbor_features']) == 2  # 2 hops
```

---

#### 1.5 Training Loop (Days 4-5)

**Tasks:**
- [ ] Implement training loop (`examples/single_machine_train.py`)
  - Forward pass with loss computation
  - Backward pass with optimizer
  - Epoch loop with progress tracking
- [ ] Add loss functions (cross-entropy for node classification)
- [ ] Add optimizer configuration (Adam with lr scheduling)
- [ ] Implement evaluation metrics (accuracy, F1)
- [ ] Add logging and checkpointing

**Code skeleton (`examples/single_machine_train.py`):**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.graph.loader import load_cora_dataset
from src.model.graphsage import GraphSAGE
from src.sampling.sampler import MiniBatchSampler

def train_single_machine(num_epochs: int = 10, batch_size: int = 32):
    """Single-machine GNN training baseline"""

    # Load dataset
    print("Loading Cora dataset...")
    graph, features, labels, train_mask, val_mask, test_mask = load_cora_dataset()

    num_nodes = graph.number_of_nodes()
    in_dim = list(features.values())[0].shape[0]
    num_classes = len(set(labels.values()))

    print(f"Graph: {num_nodes} nodes, {graph.number_of_edges()} edges")
    print(f"Features: {in_dim} dimensions, {num_classes} classes")

    # Initialize model
    model = GraphSAGE(
        in_dim=in_dim,
        hidden_dim=128,
        out_dim=num_classes,
        num_layers=2,
        dropout=0.5
    )

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Mini-batch sampler
    sampler = MiniBatchSampler(
        graph=graph,
        node_features=features,
        batch_size=batch_size,
        num_hops=2,
        sample_size=10
    )

    # Training loop
    best_val_acc = 0.0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        sampler.shuffle()

        with tqdm(total=len(sampler.all_nodes), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in sampler:
                target_nodes = batch['target_nodes']

                # Filter to training nodes only
                train_indices = [i for i, node in enumerate(target_nodes)
                                if train_mask.get(node, False)]

                if not train_indices:
                    continue

                # Forward pass
                logits = model(batch)  # (batch_size, num_classes)

                # Get labels for target nodes
                batch_labels = torch.tensor([labels[node] for node in target_nodes])

                # Compute loss only on training nodes
                loss = criterion(logits[train_indices], batch_labels[train_indices])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.update(len(target_nodes))

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)

        # Validation
        val_acc = evaluate(model, graph, features, labels, val_mask, batch_size)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')

    # Final test evaluation
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    test_acc = evaluate(model, graph, features, labels, test_mask, batch_size)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    return model, losses

def evaluate(model, graph, features, labels, mask, batch_size):
    """Evaluate model on validation/test set"""
    model.eval()

    sampler = MiniBatchSampler(graph, features, batch_size, num_hops=2, sample_size=10)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in sampler:
            target_nodes = batch['target_nodes']

            # Filter to eval nodes only
            eval_indices = [i for i, node in enumerate(target_nodes)
                           if mask.get(node, False)]

            if not eval_indices:
                continue

            logits = model(batch)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds[eval_indices].cpu().numpy())
            all_labels.extend([labels[target_nodes[i]] for i in eval_indices])

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return accuracy

if __name__ == '__main__':
    train_single_machine(num_epochs=50, batch_size=256)
```

**Time estimate:** 8-10 hours

**Validation:**
- Loss decreases over epochs
- Validation accuracy > 60% on Cora (baseline)
- Training completes without errors

---

#### 1.6 Testing and Validation (Day 5)

**Tasks:**
- [ ] Write comprehensive unit tests
  - Graph loading tests
  - Model forward/backward tests
  - Sampling tests
- [ ] Write integration test (end-to-end training)
- [ ] Run test suite with coverage (`pytest --cov`)
- [ ] Fix any bugs found during testing

**Time estimate:** 4-6 hours

**Validation:**
- All tests pass
- Code coverage > 70%
- Training on Cora achieves reasonable accuracy (>65%)

---

### Phase 1 Milestones

**Definition of Done:**
- ✅ Single-machine GNN training works end-to-end
- ✅ Loss decreases monotonically on Cora dataset
- ✅ Validation accuracy > 65% (reasonable baseline)
- ✅ All unit tests pass
- ✅ Code is modular and well-documented

**Estimated Total Time:** 10-14 days (depending on experience level)

---

## Phase 2: Graph Partitioning (Weeks 3-4)

### Goal
Partition graphs across multiple workers and validate partition quality.

### Tasks

#### 2.1 Hash-Based Partitioning (Days 1-2)

**Tasks:**
- [ ] Implement `GraphPartitioner` class (`src/graph/partitioner.py`)
- [ ] Hash-based edge partitioning algorithm
- [ ] Identify neighbor nodes for replication
- [ ] Create partition metadata (edge counts, replication factor)
- [ ] Implement partition statistics (balance, communication volume)

**Code skeleton:**
```python
class GraphPartitioner:
    def __init__(self, num_workers: int, strategy: str = 'edge_cut_hash'):
        self.num_workers = num_workers
        self.strategy = strategy

    def partition(self, graph: nx.Graph,
                  node_features: Dict[int, np.ndarray]) -> Dict[int, GraphShard]:
        """Partition graph into shards"""

        if self.strategy == 'edge_cut_hash':
            return self._edge_cut_hash(graph, node_features)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _edge_cut_hash(self, graph, node_features):
        """Simple hash-based edge partitioning"""

        # Assign edges to workers
        worker_edges = {i: [] for i in range(self.num_workers)}

        for u, v in graph.edges():
            worker_id = hash((u, v)) % self.num_workers
            worker_edges[worker_id].append((u, v))

        # Create shards
        shards = {}
        for worker_id in range(self.num_workers):
            edges = np.array(worker_edges[worker_id])

            # Local nodes: source nodes of assigned edges
            local_nodes = np.unique(edges[:, 0])

            # Neighbor nodes: target nodes that aren't local
            all_referenced = np.unique(edges.flatten())
            neighbor_nodes = np.setdiff1d(all_referenced, local_nodes)

            # Collect features for both local and neighbor nodes
            shard_features = {}
            for node in all_referenced:
                if node in node_features:
                    shard_features[node] = node_features[node]

            shards[worker_id] = GraphShard(
                worker_id=worker_id,
                local_nodes=local_nodes,
                neighbor_nodes=neighbor_nodes,
                edges=edges,
                node_features=shard_features
            )

        return shards

    def get_partition_stats(self, shards: Dict[int, GraphShard]) -> Dict:
        """Compute partition quality metrics"""
        edge_counts = [len(shard.edges) for shard in shards.values()]
        replication_factors = [len(shard.neighbor_nodes) for shard in shards.values()]

        return {
            'num_workers': self.num_workers,
            'edge_counts': edge_counts,
            'edge_balance': max(edge_counts) / min(edge_counts) if min(edge_counts) > 0 else float('inf'),
            'total_replicated_nodes': sum(replication_factors),
            'avg_replication': np.mean(replication_factors)
        }
```

**Validation:**
- Edges distributed across workers
- Edge balance < 1.5 (within 50% of ideal)
- All nodes accounted for

**Time estimate:** 6-8 hours

---

#### 2.2 METIS-Based Partitioning (Days 3-4) [Optional]

**Tasks:**
- [ ] Install METIS library (`pip install metis` or use `pymetis`)
- [ ] Implement METIS wrapper
- [ ] Compare partition quality: hash vs METIS
- [ ] Benchmark partitioning time

**Note:** This is optional for Phase 2. Can be added later if hash-based partitioning has poor balance.

**Time estimate:** 6-8 hours

---

#### 2.3 Partition Validation and Testing (Day 5)

**Tasks:**
- [ ] Write partition correctness tests
  - All edges accounted for
  - No duplicate edges
  - Neighbor nodes correctly identified
- [ ] Visualize partition (for small graphs)
- [ ] Benchmark partitioning on larger graphs (100K+ nodes)

**Time estimate:** 4-6 hours

---

### Phase 2 Milestones

**Definition of Done:**
- ✅ Graph can be partitioned into N shards
- ✅ Edge balance < 1.5 for hash-based partitioning
- ✅ Partition metadata is correct
- ✅ Shards can be serialized/deserialized
- ✅ All partition tests pass

**Estimated Total Time:** 8-12 days

---

## Phase 3: Multi-Machine Communication (Weeks 5-7)

### Goal
Enable workers to communicate and exchange neighbor embeddings via gRPC.

### Tasks

#### 3.1 gRPC Service Definition (Days 1-2)

**Tasks:**
- [ ] Define protobuf messages (`protos/worker_service.proto`)
- [ ] Define service methods (FetchEmbeddings, ProcessMiniBatch, etc.)
- [ ] Generate Python code (`python -m grpc_tools.protoc ...`)
- [ ] Create gRPC server boilerplate
- [ ] Create gRPC client boilerplate

**Time estimate:** 6-8 hours

---

#### 3.2 Worker Service Implementation (Days 3-5)

**Tasks:**
- [ ] Implement `WorkerService` class (`src/worker/service.py`)
- [ ] RPC endpoint: FetchEmbeddings (return embeddings for requested nodes)
- [ ] RPC endpoint: ReceiveModelUpdate (update local model parameters)
- [ ] Implement embedding cache with LRU eviction
- [ ] Add request batching logic

**Time estimate:** 10-12 hours

---

#### 3.3 Worker-to-Worker Communication (Days 6-8)

**Tasks:**
- [ ] Implement peer discovery (workers know each other's addresses)
- [ ] Test worker-to-worker embedding fetch
- [ ] Measure RPC latency and throughput
- [ ] Add retry logic and timeout handling
- [ ] Implement batched RPC requests

**Time estimate:** 10-12 hours

---

#### 3.4 Integration Testing (Days 9-10)

**Tasks:**
- [ ] Test 2 workers exchanging embeddings
- [ ] Test with docker-compose (multi-container setup)
- [ ] Measure communication overhead
- [ ] Profile and optimize hot paths

**Time estimate:** 8-10 hours

---

### Phase 3 Milestones

**Definition of Done:**
- ✅ Workers can fetch embeddings from each other via gRPC
- ✅ RPC latency < 50ms for reasonable batch sizes
- ✅ Request batching reduces overhead by >5x
- ✅ All communication tests pass
- ✅ Docker-compose setup works

**Estimated Total Time:** 14-20 days

---

## Phase 4: Distributed Training (Weeks 8-10)

### Goal
Run full distributed GNN training with gradient aggregation.

### Tasks

#### 4.1 Ring AllReduce Implementation (Days 1-4)

**Tasks:**
- [ ] Implement `RingAllReduce` class (`src/coordinator/allreduce.py`)
- [ ] Phase 1: Scatter-Reduce
- [ ] Phase 2: AllGather
- [ ] Test with 2 workers, then 4
- [ ] Compare with parameter server baseline

**Time estimate:** 12-16 hours

---

#### 4.2 Distributed Batch Generation (Days 5-6)

**Tasks:**
- [ ] Implement `DistributedBatchGenerator` in workers
- [ ] Each worker samples from local partition
- [ ] Coordinate epoch boundaries across workers

**Time estimate:** 6-8 hours

---

#### 4.3 End-to-End Distributed Training (Days 7-10)

**Tasks:**
- [ ] Implement lightweight coordinator
- [ ] Orchestrate training rounds
- [ ] Workers process local batches
- [ ] Ring AllReduce for gradients
- [ ] Model parameter synchronization
- [ ] Verify convergence matches single-machine

**Time estimate:** 16-20 hours

---

#### 4.4 Performance Benchmarking (Days 11-12)

**Tasks:**
- [ ] Benchmark speedup: 1, 2, 4 workers
- [ ] Measure communication vs computation time
- [ ] Profile bottlenecks
- [ ] Optimize based on profiling results

**Time estimate:** 8-10 hours

---

### Phase 4 Milestones

**Definition of Done:**
- ✅ Distributed training converges to same accuracy as single-machine
- ✅ 2-worker speedup > 1.5x
- ✅ 4-worker speedup > 2.5x
- ✅ Communication overhead < 20%
- ✅ All integration tests pass

**Estimated Total Time:** 18-24 days

---

## Phase 5: Fault Tolerance (Weeks 11-12)

### Goal
Handle worker failures gracefully with checkpointing and recovery.

### Tasks

#### 5.1 Checkpointing (Days 1-2)

**Tasks:**
- [ ] Implement periodic model checkpointing
- [ ] Save to shared storage (or local disk for testing)
- [ ] Include training state (step number, optimizer state)

**Time estimate:** 6-8 hours

---

#### 5.2 Failure Detection (Days 3-4)

**Tasks:**
- [ ] Implement heartbeat mechanism
- [ ] Detect worker failures (timeout-based)
- [ ] Log failure events

**Time estimate:** 6-8 hours

---

#### 5.3 Recovery Protocol (Days 5-8)

**Tasks:**
- [ ] Implement checkpoint recovery
- [ ] Restore model and optimizer state
- [ ] Resume training from checkpoint
- [ ] Test with simulated failures

**Time estimate:** 12-16 hours

---

#### 5.4 Hot Standby (Optional, Days 9-10)

**Tasks:**
- [ ] Implement hot standby workers
- [ ] Fast failover mechanism
- [ ] Test failover latency

**Time estimate:** 8-10 hours

---

### Phase 5 Milestones

**Definition of Done:**
- ✅ System recovers from worker failure
- ✅ Model quality maintained after recovery
- ✅ Recovery time < 2 minutes (checkpoint-based)
- ✅ All fault tolerance tests pass

**Estimated Total Time:** 12-16 days

---

## Phase 6: Optimization and Evaluation (Week 13-14)

### Goal
Optimize performance and document results.

### Tasks

#### 6.1 Feature Streaming (Days 1-3)

**Tasks:**
- [ ] Implement disk-backed feature store
- [ ] LRU caching for features
- [ ] Adaptive prefetching
- [ ] Test with graph too large for RAM

**Time estimate:** 10-12 hours

---

#### 6.2 Dynamic Load Balancing (Days 4-5)

**Tasks:**
- [ ] Implement load monitoring
- [ ] Dynamic batch size adjustment
- [ ] Test with imbalanced partitions

**Time estimate:** 8-10 hours

---

#### 6.3 Final Evaluation (Days 6-8)

**Tasks:**
- [ ] Comprehensive benchmarks (scaling, efficiency)
- [ ] Communication analysis
- [ ] Cache hit rate optimization
- [ ] Generate performance plots
- [ ] Write final report

**Time estimate:** 10-12 hours

---

### Phase 6 Milestones

**Definition of Done:**
- ✅ Feature streaming works for 100M+ node graphs
- ✅ Load balancing reduces stragglers
- ✅ Complete performance evaluation documented
- ✅ Final report written

**Estimated Total Time:** 12-16 days

---

## Summary Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Single-Machine Baseline | 2 weeks | 2 weeks |
| Phase 2: Graph Partitioning | 2 weeks | 4 weeks |
| Phase 3: Communication Layer | 3 weeks | 7 weeks |
| Phase 4: Distributed Training | 3 weeks | 10 weeks |
| Phase 5: Fault Tolerance | 2 weeks | 12 weeks |
| Phase 6: Optimization | 2 weeks | 14 weeks |

**Total: ~14 weeks (3.5 months)**

---

## Daily Development Routine

**Morning (2-3 hours):**
- Pick 1-2 tasks from current phase
- Implement core functionality
- Write basic tests

**Afternoon (2-3 hours):**
- Continue implementation
- Write comprehensive tests
- Debug and refine

**Evening (1 hour):**
- Document what you built
- Update task list
- Plan next day's work

**Weekly Review (Friday):**
- Review progress against milestones
- Adjust timeline if needed
- Write weekly summary

---

## Risk Mitigation

**Common Risks:**

1. **Scope creep** → Stick to phase goals, defer optimizations
2. **Debugging distributed issues** → Extensive logging, reproduce locally
3. **Performance bottlenecks** → Profile early and often
4. **Time underestimation** → Build in 20% buffer for each phase

**Mitigation Strategies:**

- Test incrementally (don't write lots of code before testing)
- Start simple, optimize later
- Use existing libraries when possible
- Document as you go (don't defer to end)

---

## Success Metrics

At the end of the project, you should be able to demonstrate:

1. ✅ Train GNN on graph distributed across 4+ workers
2. ✅ 3x+ speedup vs single-machine baseline
3. ✅ Communication overhead < 20%
4. ✅ Recovery from worker failure in < 2 minutes
5. ✅ Support graphs with 100M+ nodes (via feature streaming)
6. ✅ Well-documented codebase with >70% test coverage

---

## Next Steps

Ready to start implementing Phase 1? I can help you with:

1. **Set up the project structure** - Create directories and files
2. **Download Cora dataset** - Get data ready for testing
3. **Implement GraphShard** - Start with core data structures
4. **Implement GraphSAGE** - Build the GNN model
5. **Write first tests** - Set up pytest and write initial tests

Which would you like to tackle first?
