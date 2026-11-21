"""Significance tests for comparing models evaluated on the same cross-validation folds."""

import numpy as np
from scipy import stats


def corrected_ttest(d: np.ndarray, test_train_ratio: float) -> float:
    """Two-sided corrected resampled t-test (Nadeau & Bengio 2003) on per-fold differences.

    The usual paired t-test treats folds as independent, but cross-validation folds
    share most of their training data; the correction widens the variance by the
    test/train size ratio.
    """
    k = len(d)
    var = float(np.var(d, ddof=1)) if k > 1 else 0.0
    if var == 0.0:
        return float("nan")
    t = float(np.mean(d)) / np.sqrt((1.0 / k + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), k - 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment; NaN p-values are left out of the family."""
    valid = sorted((p, n) for n, p in pvalues.items() if not np.isnan(p))
    adjusted, running = {n: float("nan") for n in pvalues}, 0.0
    for i, (p, n) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - i) * p))
        adjusted[n] = running
    return adjusted
