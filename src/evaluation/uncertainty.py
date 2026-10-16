"""Bootstrap uncertainty for metric differences between two models on the same rows."""
from __future__ import annotations
import numpy as np


def bootstrap_metric_diff(y: np.ndarray, mu_a: np.ndarray, mu_b: np.ndarray, metric, n_boot: int = 1000,
                          seed: int = 42, ci: float = 0.95) -> dict:
    """metric(y, mu_a) - metric(y, mu_b) with a percentile bootstrap interval (resampling rows).

    Negative differences mean model A has the lower (better) loss. If the interval contains 0, the
    data do not show a reliable difference.
    """
    y, a, b = (np.asarray(v, float) for v in (y, mu_a, mu_b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError("Inputs must have equal length")
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"diff": float(metric(y, a) - metric(y, b)), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}
