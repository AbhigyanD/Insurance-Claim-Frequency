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


def bootstrap_ratio_ci(actual: np.ndarray, expected: np.ndarray, n_boot: int = 500, seed: int = 42,
                       ci: float = 0.95, chunk: int = 50) -> dict:
    """Percentile bootstrap interval for sum(actual) / sum(expected), resampling policies.

    This is the actual-to-expected (A/E) ratio used for calibration. The interval says how much the
    ratio could move from sampling noise alone; if it contains 1.0 the data do not show a miscalibration.
    Heavy-tailed outcomes (claim costs) give wide intervals, which is the point.
    """
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    if len(a) != len(e) or len(a) == 0:
        raise ValueError("actual and expected must be non-empty and the same length")
    if not (np.isfinite(a).all() and np.isfinite(e).all()):
        raise ValueError("Non-finite values in actual or expected")
    if e.sum() <= 0:
        raise ValueError("Expected total must be positive to form a ratio")
    rng = np.random.default_rng(seed)
    n = len(a)
    ratios = np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        den = e[idx].sum(axis=1)
        ratios[done:done + k] = a[idx].sum(axis=1) / den
        done += k
    lo, hi = np.quantile(ratios, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    point = float(a.sum() / e.sum())
    return {"ratio": point, "ci_low": float(lo), "ci_high": float(hi), "contains_one": bool(lo <= 1.0 <= hi)}
