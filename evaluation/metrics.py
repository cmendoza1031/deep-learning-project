"""
Evaluation metrics for MedHetLoRA.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def compute_balanced_accuracy(y_true, y_pred, num_classes=7):
    """Computes balanced accuracy (average recall per class)."""
    return balanced_accuracy_score(y_true, y_pred)


def compute_per_class_recall(y_true, y_pred, num_classes=7):
    """Returns dict of {class_id: recall}."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    recalls = {}
    for i in range(num_classes):
        total = cm[i, :].sum()
        correct = cm[i, i]
        recalls[i] = correct / total if total > 0 else 0.0
    return recalls
