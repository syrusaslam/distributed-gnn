"""
Tests for NumPy-based GNN implementation.
"""

import pytest
import numpy as np

from src.model.numpy_gnn import (
    SimpleGNNLayer,
    SimpleGNN,
    SGDOptimizer,
    cross_entropy_loss,
    relu,
    softmax
)


class TestActivations:
    """Tests for activation functions"""

    def test_relu(self):
        """Test ReLU activation"""
        x = np.array([[-1, 0, 1, 2]])
        result = relu(x)
        expected = np.array([[0, 0, 1, 2]])
        assert np.array_equal(result, expected)

    def test_softmax(self):
        """Test softmax activation"""
        x = np.array([[1, 2, 3], [1, 1, 1]])
        result = softmax(x)

        # Check probabilities sum to 1
        assert np.allclose(np.sum(result, axis=1), [1, 1])

        # Check all probabilities positive
        assert np.all(result > 0)

        # Check softmax properties
        assert result[0, 2] > result[0, 1] > result[0, 0]
        assert np.allclose(result[1], [1/3, 1/3, 1/3])


class TestSimpleGNNLayer:
    """Tests for single GNN layer"""

    def test_layer_initialization(self):
        """Test layer initialization"""
        layer = SimpleGNNLayer(in_dim=64, out_dim=32)

        assert layer.W.shape == (128, 32)  # in_dim*2 x out_dim
        assert layer.b.shape == (32,)

    def test_forward_pass(self):
        """Test forward pass with proper shapes"""
        layer = SimpleGNNLayer(in_dim=64, out_dim=32)

        node_features = np.random.randn(16, 64)  # batch_size=16
        neighbor_features = np.random.randn(16, 10, 64)  # 10 neighbors

        output = layer.forward(node_features, neighbor_features)

        assert output.shape == (16, 32)

    def test_forward_no_neighbors(self):
        """Test forward pass with no neighbors"""
        layer = SimpleGNNLayer(in_dim=64, out_dim=32)

        node_features = np.random.randn(16, 64)
        neighbor_features = np.zeros((16, 0, 64))  # No neighbors

        output = layer.forward(node_features, neighbor_features)

        assert output.shape == (16, 32)

    def test_backward_pass(self):
        """Test backward pass returns correct gradient shapes"""
        layer = SimpleGNNLayer(in_dim=64, out_dim=32)

        # Forward pass
        node_features = np.random.randn(16, 64)
        neighbor_features = np.random.randn(16, 10, 64)
        output = layer.forward(node_features, neighbor_features)

        # Backward pass
        grad_output = np.random.randn(16, 32)
        grad_input, grads = layer.backward(grad_output)

        # Check shapes
        assert grad_input.shape == (16, 64)
        assert grads['W'].shape == (128, 32)
        assert grads['b'].shape == (32,)

    def test_gradient_flow(self):
        """Test that gradients flow properly (non-zero)"""
        layer = SimpleGNNLayer(in_dim=8, out_dim=4)

        node_features = np.random.randn(4, 8)
        neighbor_features = np.random.randn(4, 3, 8)

        output = layer.forward(node_features, neighbor_features)
        grad_output = np.ones_like(output)

        grad_input, grads = layer.backward(grad_output)

        # Gradients should not be all zeros
        assert not np.allclose(grad_input, 0)
        assert not np.allclose(grads['W'], 0)


class TestSimpleGNN:
    """Tests for full GNN model"""

    def test_model_initialization(self):
        """Test model initialization"""
        model = SimpleGNN(in_dim=64, hidden_dim=128, out_dim=7)

        assert model.layer1.W.shape == (128, 128)  # 64*2 -> 128
        assert model.layer2.W.shape == (256, 7)    # 128*2 -> 7

    def test_forward_pass(self):
        """Test forward pass through full model"""
        model = SimpleGNN(in_dim=64, hidden_dim=128, out_dim=7)

        batch_data = {
            'node_features': np.random.randn(16, 64),
            'neighbor_features_l0': np.random.randn(16, 10, 64),
            'neighbor_features_l1': np.random.randn(16, 10, 128),
        }

        logits = model.forward(batch_data)

        assert logits.shape == (16, 7)

    def test_predict(self):
        """Test prediction returns class labels"""
        model = SimpleGNN(in_dim=64, hidden_dim=128, out_dim=7)

        batch_data = {
            'node_features': np.random.randn(16, 64),
            'neighbor_features_l0': np.random.randn(16, 10, 64),
            'neighbor_features_l1': np.random.randn(16, 10, 128),
        }

        predictions = model.predict(batch_data)

        assert predictions.shape == (16,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 7)

    def test_predict_proba(self):
        """Test probability predictions"""
        model = SimpleGNN(in_dim=64, hidden_dim=128, out_dim=7)

        batch_data = {
            'node_features': np.random.randn(16, 64),
            'neighbor_features_l0': np.random.randn(16, 10, 64),
            'neighbor_features_l1': np.random.randn(16, 10, 128),
        }

        probs = model.predict_proba(batch_data)

        assert probs.shape == (16, 7)
        # Probabilities sum to 1
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        # All probabilities positive
        assert np.all(probs >= 0)

    def test_backward_pass(self):
        """Test backward pass through full model"""
        model = SimpleGNN(in_dim=8, hidden_dim=16, out_dim=4)

        batch_data = {
            'node_features': np.random.randn(4, 8),
            'neighbor_features_l0': np.random.randn(4, 3, 8),
            'neighbor_features_l1': np.random.randn(4, 3, 16),
        }

        # Forward
        logits = model.forward(batch_data)

        # Backward
        grad_loss = np.ones_like(logits)
        grads = model.backward(grad_loss)

        # Check all gradients computed
        assert 'layer1_W' in grads
        assert 'layer1_b' in grads
        assert 'layer2_W' in grads
        assert 'layer2_b' in grads

        # Check shapes
        assert grads['layer1_W'].shape == model.layer1.W.shape
        assert grads['layer1_b'].shape == model.layer1.b.shape


class TestSGDOptimizer:
    """Tests for SGD optimizer"""

    def test_optimizer_step(self):
        """Test optimizer updates parameters"""
        model = SimpleGNN(in_dim=8, hidden_dim=16, out_dim=4)
        optimizer = SGDOptimizer(learning_rate=0.01)

        # Save initial weights
        W1_before = model.layer1.W.copy()

        # Dummy gradients
        gradients = {
            'layer1_W': np.ones_like(model.layer1.W),
            'layer1_b': np.ones_like(model.layer1.b),
            'layer2_W': np.ones_like(model.layer2.W),
            'layer2_b': np.ones_like(model.layer2.b),
        }

        # Update
        optimizer.step(model, gradients)

        # Weights should have changed
        assert not np.array_equal(model.layer1.W, W1_before)

    def test_optimizer_momentum(self):
        """Test momentum works"""
        model = SimpleGNN(in_dim=8, hidden_dim=16, out_dim=4)
        optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)

        gradients = {
            'layer1_W': np.ones_like(model.layer1.W),
            'layer1_b': np.ones_like(model.layer1.b),
            'layer2_W': np.ones_like(model.layer2.W),
            'layer2_b': np.ones_like(model.layer2.b),
        }

        # First step
        W1_before = model.layer1.W.copy()
        optimizer.step(model, gradients)
        delta1 = model.layer1.W - W1_before

        # Second step (with same gradients, momentum should amplify)
        W1_before = model.layer1.W.copy()
        optimizer.step(model, gradients)
        delta2 = model.layer1.W - W1_before

        # Second step should be larger due to momentum
        assert np.linalg.norm(delta2) > np.linalg.norm(delta1)


class TestLoss:
    """Tests for loss function"""

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss computation"""
        logits = np.array([[1.0, 2.0, 3.0],
                          [3.0, 2.0, 1.0]])
        labels = np.array([2, 0])  # Correct classes

        loss, grad = cross_entropy_loss(logits, labels)

        # Loss should be positive
        assert loss > 0

        # Gradient should have same shape as logits
        assert grad.shape == logits.shape

    def test_cross_entropy_perfect_prediction(self):
        """Test loss when predictions are perfect"""
        # Very confident correct predictions
        logits = np.array([[0.0, 0.0, 100.0],
                          [100.0, 0.0, 0.0]])
        labels = np.array([2, 0])

        loss, grad = cross_entropy_loss(logits, labels)

        # Loss should be very small
        assert loss < 0.01

    def test_cross_entropy_wrong_prediction(self):
        """Test loss when predictions are wrong"""
        # Confident but wrong predictions
        logits = np.array([[100.0, 0.0, 0.0],
                          [0.0, 100.0, 0.0]])
        labels = np.array([2, 0])  # Actual classes

        loss, grad = cross_entropy_loss(logits, labels)

        # Loss should be large
        assert loss > 1.0


def test_training_iteration():
    """Integration test: One training iteration"""
    # Small model
    model = SimpleGNN(in_dim=8, hidden_dim=16, out_dim=3)
    optimizer = SGDOptimizer(learning_rate=0.01)

    # Dummy batch
    batch_data = {
        'node_features': np.random.randn(4, 8),
        'neighbor_features_l0': np.random.randn(4, 3, 8),
        'neighbor_features_l1': np.random.randn(4, 3, 16),
    }
    labels = np.array([0, 1, 2, 0])

    # Forward pass
    logits = model.forward(batch_data)

    # Compute loss
    loss, grad_loss = cross_entropy_loss(logits, labels)

    # Backward pass
    gradients = model.backward(grad_loss)

    # Update parameters
    optimizer.step(model, gradients)

    # Should complete without errors
    assert loss > 0
    assert logits.shape == (4, 3)


def test_overfitting_small_dataset():
    """Test that model can overfit to small dataset (sanity check)"""
    np.random.seed(42)

    # Very small dataset (should be easy to overfit)
    model = SimpleGNN(in_dim=8, hidden_dim=32, out_dim=2)
    optimizer = SGDOptimizer(learning_rate=0.1, momentum=0.9)

    # Single batch
    batch_data = {
        'node_features': np.random.randn(4, 8),
        'neighbor_features_l0': np.random.randn(4, 2, 8),
        'neighbor_features_l1': np.random.randn(4, 2, 32),
    }
    labels = np.array([0, 1, 0, 1])

    losses = []
    for _ in range(100):
        logits = model.forward(batch_data)
        loss, grad_loss = cross_entropy_loss(logits, labels)
        gradients = model.backward(grad_loss)
        optimizer.step(model, gradients)
        losses.append(loss)

    # Loss should decrease significantly
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.3  # Should achieve low loss on tiny dataset
