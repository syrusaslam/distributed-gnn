"""
Metrics and evaluation utilities for GNN training.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import time


class MetricsTracker:
    """
    Track training metrics across epochs.

    Maintains history of loss, accuracy, and custom metrics.
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.history = defaultdict(list)
        self.epoch_times = []
        self.current_epoch = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics for current step/epoch.

        Args:
            metrics: Dictionary of metric_name -> value
        """
        for name, value in metrics.items():
            self.history[name].append(value)

    def log_epoch(self, epoch: int, metrics: Dict[str, float],
                  epoch_time: Optional[float] = None) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            epoch_time: Time taken for epoch (seconds)
        """
        self.current_epoch = epoch
        self.update(metrics)

        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric_name in self.history and self.history[metric_name]:
            return self.history[metric_name][-1]
        return None

    def get_best(self, metric_name: str, mode: str = 'max') -> tuple:
        """
        Get best value and epoch for a metric.

        Args:
            metric_name: Metric to find best value for
            mode: 'max' for accuracy, 'min' for loss

        Returns:
            (best_value, best_epoch)
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return None, None

        values = self.history[metric_name]

        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:  # mode == 'min'
            best_value = min(values)
            best_epoch = values.index(best_value)

        return best_value, best_epoch

    def get_history(self, metric_name: str) -> List[float]:
        """Get full history for a metric"""
        return self.history.get(metric_name, [])

    def get_avg_epoch_time(self) -> float:
        """Get average epoch time"""
        if not self.epoch_times:
            return 0.0
        return np.mean(self.epoch_times)

    def summary(self) -> Dict:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with summary stats
        """
        summary = {}

        for metric_name, values in self.history.items():
            if not values:
                continue

            summary[metric_name] = {
                'latest': values[-1],
                'best': max(values) if 'acc' in metric_name.lower() else min(values),
                'mean': np.mean(values),
                'std': np.std(values),
            }

        if self.epoch_times:
            summary['epoch_time'] = {
                'mean': np.mean(self.epoch_times),
                'total': sum(self.epoch_times),
            }

        return summary

    def __repr__(self) -> str:
        """String representation"""
        summary = self.summary()
        lines = [f"MetricsTracker(epochs={self.current_epoch})"]

        for metric, stats in summary.items():
            if metric != 'epoch_time':
                lines.append(f"  {metric}: latest={stats['latest']:.4f}, best={stats['best']:.4f}")

        return "\n".join(lines)


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class labels, shape (n,)
        labels: True class labels, shape (n,)

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    correct = np.sum(predictions == labels)
    return correct / len(predictions)


def compute_f1_score(predictions: np.ndarray, labels: np.ndarray,
                    num_classes: int, average: str = 'macro') -> float:
    """
    Compute F1 score for multi-class classification.

    Args:
        predictions: Predicted class labels
        labels: True class labels
        num_classes: Number of classes
        average: 'macro' (average across classes) or 'micro' (global)

    Returns:
        F1 score
    """
    if len(predictions) == 0:
        return 0.0

    # Compute per-class precision and recall
    precisions = []
    recalls = []

    for cls in range(num_classes):
        # True positives
        tp = np.sum((predictions == cls) & (labels == cls))

        # False positives
        fp = np.sum((predictions == cls) & (labels != cls))

        # False negatives
        fn = np.sum((predictions != cls) & (labels == cls))

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # F1 score per class
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if (p + r) > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0.0)

    if average == 'macro':
        return np.mean(f1_scores)
    else:  # micro
        # Global TP, FP, FN
        tp_total = np.sum(predictions == labels)
        fp_total = np.sum(predictions != labels)
        fn_total = fp_total  # For balanced datasets

        precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

        if (precision_micro + recall_micro) > 0:
            return 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        else:
            return 0.0


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray,
                             num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class labels
        labels: True class labels
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        Where element [i, j] is count of samples with true label i predicted as j
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(labels, predictions):
        confusion[true_label, pred_label] += 1

    return confusion


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = ""):
        """
        Initialize timer.

        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer"""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Stop timer and compute elapsed time"""
        self.elapsed = time.time() - self.start_time

    def __repr__(self) -> str:
        """String representation"""
        if self.elapsed is not None:
            return f"Timer({self.name}): {self.elapsed:.4f}s"
        return f"Timer({self.name}): running"


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training when validation metric stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current validation metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        # Check if improved
        if self.mode == 'max':
            improved = (current_value - self.best_value) > self.min_delta
        else:  # mode == 'min'
            improved = (self.best_value - current_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
