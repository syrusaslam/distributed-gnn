"""
Tests for metrics and evaluation utilities.
"""

import pytest
import numpy as np
import time

from src.utils.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_f1_score,
    compute_confusion_matrix,
    Timer,
    EarlyStopping
)


class TestMetricsTracker:
    """Tests for MetricsTracker"""

    def test_basic_tracking(self):
        """Test basic metric tracking"""
        tracker = MetricsTracker()

        tracker.update({'loss': 1.5, 'acc': 0.6})
        tracker.update({'loss': 1.2, 'acc': 0.7})

        assert tracker.get_latest('loss') == 1.2
        assert tracker.get_latest('acc') == 0.7

    def test_log_epoch(self):
        """Test logging full epoch"""
        tracker = MetricsTracker()

        tracker.log_epoch(0, {'train_loss': 1.5, 'val_acc': 0.65}, epoch_time=2.5)
        tracker.log_epoch(1, {'train_loss': 1.2, 'val_acc': 0.70}, epoch_time=2.3)

        assert tracker.current_epoch == 1
        assert len(tracker.epoch_times) == 2

    def test_get_best(self):
        """Test getting best metric value"""
        tracker = MetricsTracker()

        tracker.update({'loss': 1.5})
        tracker.update({'loss': 1.2})
        tracker.update({'loss': 1.8})

        best_loss, best_epoch = tracker.get_best('loss', mode='min')
        assert best_loss == 1.2
        assert best_epoch == 1

    def test_get_best_accuracy(self):
        """Test getting best accuracy (max mode)"""
        tracker = MetricsTracker()

        tracker.update({'acc': 0.6})
        tracker.update({'acc': 0.8})
        tracker.update({'acc': 0.7})

        best_acc, best_epoch = tracker.get_best('acc', mode='max')
        assert best_acc == 0.8
        assert best_epoch == 1

    def test_get_history(self):
        """Test getting full metric history"""
        tracker = MetricsTracker()

        for loss in [1.5, 1.3, 1.1, 0.9]:
            tracker.update({'loss': loss})

        history = tracker.get_history('loss')
        assert history == [1.5, 1.3, 1.1, 0.9]

    def test_avg_epoch_time(self):
        """Test average epoch time calculation"""
        tracker = MetricsTracker()

        tracker.log_epoch(0, {}, epoch_time=2.0)
        tracker.log_epoch(1, {}, epoch_time=3.0)
        tracker.log_epoch(2, {}, epoch_time=2.5)

        avg_time = tracker.get_avg_epoch_time()
        assert avg_time == 2.5

    def test_summary(self):
        """Test summary statistics"""
        tracker = MetricsTracker()

        tracker.update({'loss': 1.5})
        tracker.update({'loss': 1.2})
        tracker.update({'loss': 1.0})

        summary = tracker.summary()

        assert 'loss' in summary
        assert summary['loss']['latest'] == 1.0
        assert summary['loss']['best'] == 1.0  # Min for loss

    def test_empty_tracker(self):
        """Test tracker with no data"""
        tracker = MetricsTracker()

        assert tracker.get_latest('loss') is None
        best_val, best_epoch = tracker.get_best('acc', mode='max')
        assert best_val is None
        assert best_epoch is None


class TestAccuracy:
    """Tests for accuracy computation"""

    def test_perfect_accuracy(self):
        """Test with perfect predictions"""
        predictions = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])

        acc = compute_accuracy(predictions, labels)
        assert acc == 1.0

    def test_zero_accuracy(self):
        """Test with all wrong predictions"""
        predictions = np.array([0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1])

        acc = compute_accuracy(predictions, labels)
        assert acc == 0.0

    def test_partial_accuracy(self):
        """Test with partially correct predictions"""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 1, 3, 3])

        acc = compute_accuracy(predictions, labels)
        assert acc == 0.6  # 3 out of 5 correct

    def test_empty_predictions(self):
        """Test with empty arrays"""
        predictions = np.array([])
        labels = np.array([])

        acc = compute_accuracy(predictions, labels)
        assert acc == 0.0


class TestF1Score:
    """Tests for F1 score computation"""

    def test_perfect_f1(self):
        """Test with perfect predictions"""
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])

        f1 = compute_f1_score(predictions, labels, num_classes=3)
        assert f1 == 1.0

    def test_macro_f1(self):
        """Test macro-averaged F1 score"""
        # Simple case: 2 classes, balanced
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 1, 0])

        f1 = compute_f1_score(predictions, labels, num_classes=2, average='macro')

        # Both classes: TP=1, FP=1, FN=1
        # Precision = Recall = 0.5 for each
        # F1 = 0.5 for each class
        assert f1 == 0.5

    def test_micro_f1(self):
        """Test micro-averaged F1 score"""
        predictions = np.array([0, 1, 2, 0])
        labels = np.array([0, 1, 1, 2])

        f1 = compute_f1_score(predictions, labels, num_classes=3, average='micro')

        # Should be same as accuracy for balanced data
        acc = compute_accuracy(predictions, labels)
        assert abs(f1 - acc) < 0.01

    def test_multi_class_f1(self):
        """Test F1 on multi-class problem"""
        predictions = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3, 1, 2, 3, 0])

        f1 = compute_f1_score(predictions, labels, num_classes=4)

        # F1 should be between 0 and 1
        assert 0.0 <= f1 <= 1.0


class TestConfusionMatrix:
    """Tests for confusion matrix"""

    def test_binary_confusion(self):
        """Test confusion matrix for binary classification"""
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 1, 0])

        cm = compute_confusion_matrix(predictions, labels, num_classes=2)

        # Expected:
        # [[1, 1],   # True 0: 1 predicted as 0, 1 predicted as 1
        #  [1, 1]]   # True 1: 1 predicted as 0, 1 predicted as 1
        assert cm[0, 0] == 1  # True 0, predicted 0
        assert cm[0, 1] == 1  # True 0, predicted 1
        assert cm[1, 0] == 1  # True 1, predicted 0
        assert cm[1, 1] == 1  # True 1, predicted 1

    def test_multiclass_confusion(self):
        """Test confusion matrix for multi-class"""
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])

        cm = compute_confusion_matrix(predictions, labels, num_classes=3)

        # Perfect predictions: diagonal should be non-zero
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2

        # Off-diagonal should be zero
        assert np.sum(cm) - np.trace(cm) == 0

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape"""
        predictions = np.array([0, 1, 2])
        labels = np.array([1, 2, 0])

        cm = compute_confusion_matrix(predictions, labels, num_classes=3)

        assert cm.shape == (3, 3)


class TestTimer:
    """Tests for Timer context manager"""

    def test_basic_timing(self):
        """Test basic timing"""
        with Timer("test") as timer:
            time.sleep(0.1)

        assert timer.elapsed is not None
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should be close to 0.1

    def test_timer_without_name(self):
        """Test timer without name"""
        with Timer() as timer:
            pass

        assert timer.elapsed is not None

    def test_timer_repr(self):
        """Test timer string representation"""
        with Timer("my_timer") as timer:
            time.sleep(0.01)

        repr_str = repr(timer)
        assert "my_timer" in repr_str
        assert "s" in repr_str  # Should show seconds


class TestEarlyStopping:
    """Tests for EarlyStopping"""

    def test_no_early_stop_improving(self):
        """Test that early stopping doesn't trigger when improving"""
        early_stop = EarlyStopping(patience=3, mode='min')

        # Continuously improving
        assert not early_stop(1.5)
        assert not early_stop(1.2)
        assert not early_stop(1.0)
        assert not early_stop(0.8)

        assert not early_stop.should_stop

    def test_early_stop_triggered(self):
        """Test that early stopping triggers after patience exhausted"""
        early_stop = EarlyStopping(patience=2, mode='min')

        early_stop(1.0)  # Best
        early_stop(1.1)  # Worse, counter = 1
        early_stop(1.2)  # Worse, counter = 2

        # Should trigger on third consecutive non-improvement
        assert early_stop.should_stop

    def test_max_mode(self):
        """Test early stopping in max mode (for accuracy)"""
        early_stop = EarlyStopping(patience=2, mode='max')

        early_stop(0.6)  # Best
        early_stop(0.7)  # Better, reset counter
        early_stop(0.65)  # Worse, counter = 1
        early_stop(0.64)  # Worse, counter = 2

        assert early_stop.should_stop

    def test_min_delta(self):
        """Test minimum delta for improvement"""
        early_stop = EarlyStopping(patience=2, mode='min', min_delta=0.1)

        early_stop(1.0)  # Best
        early_stop(0.95)  # Improved by 0.05 < 0.1, doesn't count
        early_stop(0.94)  # Still not enough improvement

        # Should have incremented counter twice
        assert early_stop.counter == 2

    def test_reset(self):
        """Test resetting early stopping"""
        early_stop = EarlyStopping(patience=2, mode='min')

        early_stop(1.0)
        early_stop(1.1)
        early_stop(1.2)

        # Should be triggered
        assert early_stop.should_stop

        # Reset
        early_stop.reset()

        assert early_stop.counter == 0
        assert early_stop.best_value is None
        assert not early_stop.should_stop

    def test_call_return_value(self):
        """Test return value of __call__"""
        early_stop = EarlyStopping(patience=2, mode='min')

        assert early_stop(1.0) is False  # First value
        assert early_stop(1.1) is False  # First non-improvement (counter=1)
        assert early_stop(1.2) is True   # Second non-improvement (counter=2, patience exhausted)


def test_metrics_integration():
    """Integration test: Track metrics through a simulated training loop"""
    tracker = MetricsTracker()
    early_stop = EarlyStopping(patience=3, mode='max')

    # Simulate training
    val_accs = [0.5, 0.6, 0.7, 0.75, 0.74, 0.73, 0.72, 0.71]

    for epoch, val_acc in enumerate(val_accs):
        with Timer(f"epoch_{epoch}") as timer:
            time.sleep(0.01)  # Simulate training

        tracker.log_epoch(
            epoch,
            {'val_acc': val_acc},
            epoch_time=timer.elapsed
        )

        if early_stop(val_acc):
            break

    # Should have stopped early (after epoch 6, patience=3)
    assert tracker.current_epoch < len(val_accs) - 1

    # Best accuracy should be 0.75 (epoch 3)
    best_acc, best_epoch = tracker.get_best('val_acc', mode='max')
    assert best_acc == 0.75
    assert best_epoch == 3
