"""
End-to-end training with NumPy-based GNN.

This demonstrates the complete training pipeline using the simplified
NumPy implementation. For production, use train_pytorch_gnn.py with PyTorch.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.loader import generate_synthetic_graph, get_graph_statistics
from src.sampling.sampler import MiniBatchSampler, NeighborhoodSampler
from src.model.numpy_gnn import SimpleGNN, SGDOptimizer, cross_entropy_loss
from src.utils.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_f1_score,
    Timer,
    EarlyStopping
)


def prepare_batch_for_gnn(batch: Dict, graph, node_features: Dict,
                         sampler: NeighborhoodSampler, hidden_dim: int) -> Dict:
    """
    Convert mini-batch to format expected by GNN.

    Args:
        batch: Batch from MiniBatchSampler
        graph: NetworkX graph
        node_features: Dictionary of node features
        sampler: NeighborhoodSampler instance
        hidden_dim: Hidden dimension (for layer 2 neighbors)

    Returns:
        Dictionary with properly formatted features
    """
    target_nodes = batch['target_nodes']
    batch_size = len(target_nodes)

    # Get target node features
    node_feat = np.array([node_features[node] for node in target_nodes])

    # Sample 2-hop neighborhoods
    neighborhoods = sampler.sample_neighbors(target_nodes)

    # Layer 0: Original features for 1-hop neighbors
    neighbor_feat_l0_list = []
    for node in target_nodes:
        hop_0_neighbors = neighborhoods[node][0] if len(neighborhoods[node]) > 0 else []
        if hop_0_neighbors:
            feats = np.array([node_features[n] for n in hop_0_neighbors])
        else:
            feats = np.zeros((0, node_feat.shape[1]))
        neighbor_feat_l0_list.append(feats)

    # Pad to same number of neighbors (for batching)
    max_neighbors_l0 = max(len(f) for f in neighbor_feat_l0_list)
    if max_neighbors_l0 == 0:
        max_neighbors_l0 = 1  # At least 1 for shape consistency

    neighbor_feat_l0_padded = []
    for feats in neighbor_feat_l0_list:
        if len(feats) == 0:
            feats = np.zeros((1, node_feat.shape[1]))
        # Pad or truncate
        if len(feats) < max_neighbors_l0:
            padding = np.zeros((max_neighbors_l0 - len(feats), node_feat.shape[1]))
            feats = np.vstack([feats, padding])
        elif len(feats) > max_neighbors_l0:
            feats = feats[:max_neighbors_l0]
        neighbor_feat_l0_padded.append(feats)

    neighbor_feat_l0 = np.array(neighbor_feat_l0_padded)

    # Layer 1: Random features for 2-hop neighbors (simplified)
    # In real implementation, would compute layer 1 embeddings
    max_neighbors_l1 = 5
    neighbor_feat_l1 = np.random.randn(batch_size, max_neighbors_l1, hidden_dim) * 0.1

    return {
        'node_features': node_feat,
        'neighbor_features_l0': neighbor_feat_l0,
        'neighbor_features_l1': neighbor_feat_l1,
    }


def evaluate_model(model: SimpleGNN, graph, node_features: Dict, labels: Dict,
                  mask: Dict, sampler: NeighborhoodSampler,
                  batch_size: int = 32) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: GNN model
        graph: NetworkX graph
        node_features: Node features
        labels: Node labels
        mask: Boolean mask for nodes to evaluate
        sampler: NeighborhoodSampler
        batch_size: Batch size

    Returns:
        (accuracy, loss)
    """
    eval_nodes = [node for node, include in mask.items() if include]

    if not eval_nodes:
        return 0.0, float('inf')

    all_predictions = []
    all_labels = []
    all_losses = []

    # Process in batches
    for i in range(0, len(eval_nodes), batch_size):
        batch_nodes = eval_nodes[i:i+batch_size]

        # Prepare batch
        batch_data = prepare_batch_for_gnn(
            {'target_nodes': batch_nodes},
            graph,
            node_features,
            sampler,
            model.hidden_dim
        )

        # Forward pass
        logits = model.forward(batch_data)
        predictions = model.predict(batch_data)

        # Get labels
        batch_labels = np.array([labels[node] for node in batch_nodes])

        # Compute loss
        loss, _ = cross_entropy_loss(logits, batch_labels)

        all_predictions.extend(predictions)
        all_labels.extend(batch_labels)
        all_losses.append(loss)

    # Compute metrics
    accuracy = compute_accuracy(np.array(all_predictions), np.array(all_labels))
    avg_loss = np.mean(all_losses)

    return accuracy, avg_loss


def create_train_val_test_split(num_nodes: int,
                                train_ratio: float = 0.6,
                                val_ratio: float = 0.2) -> Tuple[Dict, Dict, Dict]:
    """
    Create train/val/test split.

    Args:
        num_nodes: Number of nodes
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        (train_mask, val_mask, test_mask)
    """
    nodes = list(range(num_nodes))
    np.random.shuffle(nodes)

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_nodes = set(nodes[:num_train])
    val_nodes = set(nodes[num_train:num_train+num_val])
    test_nodes = set(nodes[num_train+num_val:])

    train_mask = {i: i in train_nodes for i in range(num_nodes)}
    val_mask = {i: i in val_nodes for i in range(num_nodes)}
    test_mask = {i: i in test_nodes for i in range(num_nodes)}

    return train_mask, val_mask, test_mask


def train_gnn(num_nodes: int = 500,
             num_edges: int = 2000,
             feature_dim: int = 32,
             num_classes: int = 5,
             hidden_dim: int = 64,
             num_epochs: int = 50,
             batch_size: int = 32,
             learning_rate: float = 0.01,
             patience: int = 10):
    """
    Train GNN on synthetic graph.

    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph
        feature_dim: Feature dimension
        num_classes: Number of classes
        hidden_dim: Hidden layer dimension
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patience: Early stopping patience
    """
    print("=" * 60)
    print("Training NumPy-based GNN")
    print("=" * 60)

    # Generate synthetic graph
    print("\n1. Generating synthetic graph...")
    graph, features, labels = generate_synthetic_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feature_dim,
        num_classes=num_classes,
        graph_type='barabasi_albert'
    )

    # Print graph statistics
    stats = get_graph_statistics(graph)
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Edges: {stats['num_edges']}")
    print(f"   Avg Degree: {stats['avg_degree']:.2f}")
    print(f"   Feature Dim: {feature_dim}")
    print(f"   Num Classes: {num_classes}")

    # Create train/val/test split
    print("\n2. Creating data splits...")
    train_mask, val_mask, test_mask = create_train_val_test_split(num_nodes)

    num_train = sum(train_mask.values())
    num_val = sum(val_mask.values())
    num_test = sum(test_mask.values())

    print(f"   Train: {num_train} nodes ({num_train/num_nodes*100:.1f}%)")
    print(f"   Val:   {num_val} nodes ({num_val/num_nodes*100:.1f}%)")
    print(f"   Test:  {num_test} nodes ({num_test/num_nodes*100:.1f}%)")

    # Initialize model
    print(f"\n3. Initializing model...")
    print(f"   Architecture: {feature_dim} -> {hidden_dim} -> {num_classes}")

    model = SimpleGNN(in_dim=feature_dim, hidden_dim=hidden_dim, out_dim=num_classes)
    optimizer = SGDOptimizer(learning_rate=learning_rate, momentum=0.9)

    # Initialize samplers
    neighborhood_sampler = NeighborhoodSampler(graph, num_hops=2, sample_size=10)

    # Initialize metrics tracking
    tracker = MetricsTracker()
    early_stop = EarlyStopping(patience=patience, mode='max')

    print(f"\n4. Training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {patience}")
    print("")

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        with Timer() as timer:
            # Get training nodes
            train_nodes = [node for node, include in train_mask.items() if include]
            np.random.shuffle(train_nodes)

            epoch_losses = []

            # Mini-batch training
            for i in range(0, len(train_nodes), batch_size):
                batch_nodes = train_nodes[i:i+batch_size]

                # Prepare batch
                batch_data = prepare_batch_for_gnn(
                    {'target_nodes': batch_nodes},
                    graph,
                    features,
                    neighborhood_sampler,
                    hidden_dim
                )

                # Get labels
                batch_labels = np.array([labels[node] for node in batch_nodes])

                # Forward pass
                logits = model.forward(batch_data)

                # Compute loss
                loss, grad_loss = cross_entropy_loss(logits, batch_labels)

                # Backward pass
                gradients = model.backward(grad_loss)

                # Update parameters
                optimizer.step(model, gradients)

                epoch_losses.append(loss)

            # Compute training metrics
            train_loss = np.mean(epoch_losses)

            # Evaluate on validation set
            val_acc, val_loss = evaluate_model(
                model, graph, features, labels, val_mask,
                neighborhood_sampler, batch_size
            )

        # Log metrics
        tracker.log_epoch(epoch, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, epoch_time=timer.elapsed)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {timer.elapsed:.2f}s")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   → New best validation accuracy!")

        # Early stopping
        if early_stop(val_acc):
            print(f"\n   Early stopping triggered at epoch {epoch+1}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("5. Final Evaluation")
    print("=" * 60)

    test_acc, test_loss = evaluate_model(
        model, graph, features, labels, test_mask,
        neighborhood_sampler, batch_size
    )

    print(f"\nTest Results:")
    print(f"   Loss:     {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")

    # Print training summary
    print("\n" + "=" * 60)
    print("6. Training Summary")
    print("=" * 60)

    summary = tracker.summary()

    print(f"\nTraining Loss:")
    print(f"   Best:  {summary['train_loss']['best']:.4f}")
    print(f"   Final: {summary['train_loss']['latest']:.4f}")

    print(f"\nValidation Accuracy:")
    print(f"   Best:  {summary['val_acc']['best']:.4f}")
    print(f"   Final: {summary['val_acc']['latest']:.4f}")

    print(f"\nEpoch Time:")
    print(f"   Mean:  {summary['epoch_time']['mean']:.2f}s")
    print(f"   Total: {summary['epoch_time']['total']:.2f}s")

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)

    return model, tracker


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NumPy-based GNN')
    parser.add_argument('--num-nodes', type=int, default=500, help='Number of nodes')
    parser.add_argument('--num-edges', type=int, default=2000, help='Number of edges')
    parser.add_argument('--feature-dim', type=int, default=32, help='Feature dimension')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Train model
    model, tracker = train_gnn(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )
