"""Evaluation metrics from Section 4.3 of the paper.

Besides RMSE, precision and recall, the module contains a few alternative definitions
(MAE, Eq. 6 exactly as printed, per-user RMSE). They are used to check whether the
paper's number could have been computed differently.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def eq6_literal(y_true, y_pred) -> float:
    """Eq. 6 as printed in the paper, i.e. without the square (signed square root)."""
    m = float(np.mean(np.asarray(y_true, float) - np.asarray(y_pred, float)))
    return float(np.sign(m) * np.sqrt(abs(m)))


def per_user_rmse(uids, y_true, y_pred) -> float:
    """RMSE computed per user and then averaged over users."""
    uids = np.asarray(uids)
    se = (np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2
    _, inv = np.unique(uids, return_inverse=True)
    return float(np.mean(np.sqrt(np.bincount(inv, se) / np.bincount(inv))))


def precision_recall(y_true, y_pred, relevant_min: float = 4.0, selected_min: float = 3.5):
    """Precision and recall of the "relevant" ratings.

    A rating is relevant if the true value is 4 or 5, as in the paper. The paper does
    not say when a prediction counts as selected; by default we use 3.5, i.e. a
    prediction that rounds to at least 4.
    """
    rel = np.asarray(y_true) >= relevant_min
    sel = np.asarray(y_pred) >= selected_min
    tp = np.sum(rel & sel)
    precision = tp / sel.sum() if sel.sum() else np.nan
    recall = tp / rel.sum() if rel.sum() else np.nan
    return float(precision), float(recall)


def max_precision_at_recall(y_true, y_pred, recall_min: float = 0.799, relevant_min: float = 4.0) -> float:
    """Highest precision over all selection thresholds that keep recall >= recall_min.

    This removes the unknown threshold from the comparison: if the value is below the
    paper's 0.771, no threshold can reproduce its precision at its recall.
    """
    order = np.argsort(-np.asarray(y_pred, float), kind="stable")
    ps = np.asarray(y_pred, float)[order]
    rel = (np.asarray(y_true) >= relevant_min)[order]
    tp = np.cumsum(rel)
    k = np.arange(1, len(rel) + 1)
    # A threshold can only separate different predicted values, not split a group of ties.
    boundary = np.r_[ps[:-1] != ps[1:], True]
    ok = (tp / max(rel.sum(), 1) >= recall_min) & boundary
    return float((tp / k)[ok].max()) if ok.any() else float("nan")


def all_metrics(uids, y_true, y_pred) -> dict:
    p35, r35 = precision_recall(y_true, y_pred, selected_min=3.5)
    p40, r40 = precision_recall(y_true, y_pred, selected_min=4.0)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "eq6_literal": eq6_literal(y_true, y_pred),
        "per_user_rmse": per_user_rmse(uids, y_true, y_pred),
        "rmse_rounded": rmse(y_true, np.clip(np.rint(y_pred), 1, 5)),
        "precision@3.5": p35, "recall@3.5": r35,
        "precision@4.0": p40, "recall@4.0": r40,
        "maxP@R>=0.799": max_precision_at_recall(y_true, y_pred),
    }
