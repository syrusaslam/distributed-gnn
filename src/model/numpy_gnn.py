"""
Simplified NumPy-based GNN implementation.

This is a lightweight implementation for testing and demonstration purposes.
For production, use the PyTorch-based GraphSAGE model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU"""
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation (numerically stable)"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class SimpleGNNLayer:
    """
    Single GNN layer using mean aggregation (GraphSAGE-like).

    Process:
    1. Aggregate neighbor features (mean pooling)
    2. Concatenate node features with aggregated neighbor features
    3. Apply linear transformation
    4. Apply activation (ReLU)
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = 'relu'):
        """
        Initialize GNN layer.

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            activation: Activation function ('relu' or 'none')
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        # Weight matrix for concatenated features [self + neighbor]
        # Xavier initialization
        limit = np.sqrt(6 / (in_dim * 2 + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim * 2, out_dim))
        self.b = np.zeros(out_dim)

        # Cache for backprop
        self.cache = {}

    def forward(self, node_features: np.ndarray,
                neighbor_features: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            node_features: (batch_size, in_dim) - target node features
            neighbor_features: (batch_size, num_neighbors, in_dim) - neighbor features

        Returns:
            (batch_size, out_dim) - output features
        """
        # Aggregate neighbors (mean pooling)
        if neighbor_features.size > 0:
            neighbor_agg = np.mean(neighbor_features, axis=1)  # (batch_size, in_dim)
        else:
            neighbor_agg = np.zeros_like(node_features)

        # Concatenate self + neighbor
        combined = np.concatenate([node_features, neighbor_agg], axis=1)  # (batch_size, in_dim*2)

        # Linear transformation
        z = combined @ self.W + self.b  # (batch_size, out_dim)

        # Activation
        if self.activation == 'relu':
            output = relu(z)
        else:
            output = z

        # Cache for backprop
        self.cache = {
            'node_features': node_features,
            'neighbor_features': neighbor_features,
            'neighbor_agg': neighbor_agg,
            'combined': combined,
            'z': z,
            'output': output
        }

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Backward pass.

        Args:
            grad_output: Gradient from next layer (batch_size, out_dim)

        Returns:
            grad_input: Gradient w.r.t. input
            grads: Dictionary of parameter gradients
        """
        # Gradient through activation
        if self.activation == 'relu':
            grad_z = grad_output * relu_derivative(self.cache['z'])
        else:
            grad_z = grad_output

        # Gradient w.r.t. parameters
        grad_W = self.cache['combined'].T @ grad_z
        grad_b = np.sum(grad_z, axis=0)

        # Gradient w.r.t. input (combined)
        grad_combined = grad_z @ self.W.T

        # Split gradient into node and neighbor parts
        grad_node = grad_combined[:, :self.in_dim]
        grad_neighbor_agg = grad_combined[:, self.in_dim:]

        return grad_node, {
            'W': grad_W,
            'b': grad_b
        }


class SimpleGNN:
    """
    Simple 2-layer GNN for node classification.

    Architecture:
    - Layer 1: in_dim -> hidden_dim (ReLU)
    - Layer 2: hidden_dim -> out_dim (Linear)
    - Output: Softmax for classification
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        """
        Initialize GNN.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (number of classes)
        """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Layers
        self.layer1 = SimpleGNNLayer(in_dim, hidden_dim, activation='relu')
        self.layer2 = SimpleGNNLayer(hidden_dim, out_dim, activation='none')

    def forward(self, batch_data: Dict) -> np.ndarray:
        """
        Forward pass through network.

        Args:
            batch_data: Dictionary with:
                - 'node_features': (batch_size, num_nodes, in_dim)
                - 'neighbor_features_l0': (batch_size, num_neighbors_l0, in_dim)
                - 'neighbor_features_l1': (batch_size, num_neighbors_l1, hidden_dim)

        Returns:
            logits: (batch_size, out_dim) - class logits
        """
        # Layer 1
        node_feat_l0 = batch_data['node_features']
        neighbor_feat_l0 = batch_data.get('neighbor_features_l0',
                                         np.zeros((len(node_feat_l0), 0, self.in_dim)))

        h1 = self.layer1.forward(node_feat_l0, neighbor_feat_l0)

        # Layer 2
        # For layer 2, we need layer 1's output as node features
        # and layer 1's neighbor outputs
        neighbor_feat_l1 = batch_data.get('neighbor_features_l1',
                                         np.zeros((len(h1), 0, self.hidden_dim)))

        logits = self.layer2.forward(h1, neighbor_feat_l1)

        return logits

    def backward(self, grad_loss: np.ndarray) -> Dict:
        """
        Backward pass through network.

        Args:
            grad_loss: Gradient of loss w.r.t. output (batch_size, out_dim)

        Returns:
            Dictionary of all parameter gradients
        """
        # Backward through layer 2
        grad_h1, grads_l2 = self.layer2.backward(grad_loss)

        # Backward through layer 1
        grad_input, grads_l1 = self.layer1.backward(grad_h1)

        return {
            'layer1_W': grads_l1['W'],
            'layer1_b': grads_l1['b'],
            'layer2_W': grads_l2['W'],
            'layer2_b': grads_l2['b'],
        }

    def predict(self, batch_data: Dict) -> np.ndarray:
        """
        Predict class labels.

        Args:
            batch_data: Batch data dictionary

        Returns:
            (batch_size,) - predicted class labels
        """
        logits = self.forward(batch_data)
        return np.argmax(logits, axis=1)

    def predict_proba(self, batch_data: Dict) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            batch_data: Batch data dictionary

        Returns:
            (batch_size, out_dim) - class probabilities
        """
        logits = self.forward(batch_data)
        return softmax(logits)


class SGDOptimizer:
    """Simple SGD optimizer with momentum"""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize optimizer.

        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, model: SimpleGNN, gradients: Dict) -> None:
        """
        Update model parameters.

        Args:
            model: GNN model
            gradients: Dictionary of gradients
        """
        # Update layer 1
        if 'layer1_W' not in self.velocity:
            self.velocity['layer1_W'] = np.zeros_like(model.layer1.W)
            self.velocity['layer1_b'] = np.zeros_like(model.layer1.b)
            self.velocity['layer2_W'] = np.zeros_like(model.layer2.W)
            self.velocity['layer2_b'] = np.zeros_like(model.layer2.b)

        # Momentum update
        self.velocity['layer1_W'] = (self.momentum * self.velocity['layer1_W'] -
                                     self.lr * gradients['layer1_W'])
        self.velocity['layer1_b'] = (self.momentum * self.velocity['layer1_b'] -
                                     self.lr * gradients['layer1_b'])
        self.velocity['layer2_W'] = (self.momentum * self.velocity['layer2_W'] -
                                     self.lr * gradients['layer2_W'])
        self.velocity['layer2_b'] = (self.momentum * self.velocity['layer2_b'] -
                                     self.lr * gradients['layer2_b'])

        # Apply updates
        model.layer1.W += self.velocity['layer1_W']
        model.layer1.b += self.velocity['layer1_b']
        model.layer2.W += self.velocity['layer2_W']
        model.layer2.b += self.velocity['layer2_b']


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and gradient.

    Args:
        logits: (batch_size, num_classes) - model outputs
        labels: (batch_size,) - true class labels

    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t. logits
    """
    batch_size = logits.shape[0]

    # Softmax probabilities
    probs = softmax(logits)

    # Cross-entropy loss
    # Prevent log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    loss = -np.mean(np.log(probs_clipped[np.arange(batch_size), labels]))

    # Gradient: softmax - one_hot
    grad = probs.copy()
    grad[np.arange(batch_size), labels] -= 1
    grad /= batch_size

    return loss, grad
