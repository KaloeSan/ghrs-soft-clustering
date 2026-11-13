"""Evaluation metrics from the paper (Sec. 4.3)."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_recall(y_true, y_pred, relevant_min: float = 4.0, selected_min: float = 3.5):
    """Relevant: actual rating in [4, 5] (paper). Selected: predicted rating rounds to
    >= 4, i.e. >= 3.5 (the paper does not state the prediction threshold)."""
    rel = np.asarray(y_true) >= relevant_min
    sel = np.asarray(y_pred) >= selected_min
    tp = np.sum(rel & sel)
    precision = tp / sel.sum() if sel.sum() else np.nan
    recall = tp / rel.sum() if rel.sum() else np.nan
    return float(precision), float(recall)
