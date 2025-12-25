# Distributed Graph Neural Network Training Framework

A distributed system for training Graph Neural Networks (GNNs) on large-scale graphs that don't fit on a single machine.

## Features

- **Distributed Training**: Partition graphs across multiple workers
- **Efficient Communication**: Mini-batch sampling with cross-partition neighbor fetching
- **Gradient Aggregation**: Support for synchronous and asynchronous updates
- **Fault Tolerance**: Checkpointing and recovery from worker failures
- **Scalable**: Handle graphs with 10M-100M+ nodes

## Architecture

The system uses an edge-cut partitioning strategy with:
- **Coordinator**: Orchestrates training, manages synchronization
- **Workers**: Own graph shards, compute local gradients, exchange embeddings
- **Communication**: gRPC-based peer-to-peer and coordinator communication
- **Model**: GraphSAGE (sample-and-aggregate pattern)

## Project Structure

```
distributed_gnn/
├── src/
│   ├── graph/          # Graph partitioning and data structures
│   ├── model/          # GNN model implementation (GraphSAGE)
│   ├── sampling/       # Mini-batch sampling and neighbor fetching
│   ├── worker/         # Worker service implementation
│   ├── coordinator/    # Coordinator and fault tolerance
│   └── utils/          # Utilities, metrics, serialization
├── tests/              # Unit and integration tests
├── examples/           # Example training scripts
├── scripts/            # Dataset download, benchmarking
├── protos/             # gRPC protocol buffer definitions
└── docs/               # Documentation

```

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Single-Machine Training (Phase 1)

```bash
# Download Cora dataset
python scripts/download_datasets.py

# Train on single machine (baseline)
python examples/single_machine_train.py --epochs 50 --batch-size 256
```

### 3. Distributed Training (Phase 4)

```bash
# Terminal 1: Start coordinator
python -m src.coordinator.coordinator --num-workers 2 --port 50050

# Terminal 2: Start worker 0
WORKER_ID=0 python -m src.worker.service --coordinator localhost:50050 --port 50051

# Terminal 3: Start worker 1
WORKER_ID=1 python -m src.worker.service --coordinator localhost:50050 --port 50052

# Terminal 4: Run distributed training
python examples/distributed_train.py --dataset cora --epochs 50
```

### 4. Docker-based Deployment

```bash
# Build and run with docker-compose
docker-compose up --build
```

## Development Phases

### ✅ Phase 1: Single-Machine Baseline (Weeks 1-2)
- [x] Graph loading and storage
- [x] GraphSAGE model implementation
- [x] Mini-batch sampling
- [x] Training loop
- [x] Evaluation on Cora dataset

### 🚧 Phase 2: Graph Partitioning (Weeks 3-4)
- [ ] Hash-based edge partitioning
- [ ] Partition quality metrics
- [ ] Serialization and deserialization

### 📋 Phase 3: Communication Layer (Weeks 5-7)
- [ ] gRPC service definitions
- [ ] Worker-to-worker communication
- [ ] Embedding cache and batching

### 📋 Phase 4: Distributed Training (Weeks 8-10)
- [ ] Ring AllReduce gradient aggregation
- [ ] Distributed batch generation
- [ ] End-to-end training orchestration

### 📋 Phase 5: Fault Tolerance (Weeks 11-12)
- [ ] Checkpointing and recovery
- [ ] Worker failure detection
- [ ] Hot standby workers (optional)

### 📋 Phase 6: Optimization (Weeks 13-14)
- [ ] Feature streaming for large graphs
- [ ] Dynamic load balancing
- [ ] Performance tuning

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Performance Benchmarks

| Configuration | Throughput | Speedup | Communication Overhead |
|--------------|------------|---------|------------------------|
| 1 worker     | baseline   | 1.0x    | 0%                     |
| 2 workers    | TBD        | TBD     | TBD                    |
| 4 workers    | TBD        | TBD     | TBD                    |

## Documentation

- [Implementation Specification](distributed_gnn_implementation_spec.md)
- [Refined Architecture](refined_architecture.md)
- [Implementation Plan](implementation_plan.md)

## Contributing

This is a solo research project, but feedback and suggestions are welcome!

## License

MIT License

## References

- Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Gonzalez et al. (2012). "PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs"
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

## Author

Built as a distributed systems + ML learning project.

---

**Current Status**: Phase 1 in progress - Building single-machine baseline
