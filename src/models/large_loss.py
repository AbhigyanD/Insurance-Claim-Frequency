"""Large-loss treatment: model capped severity, then load for the excess above the cap."""
from __future__ import annotations

import numpy as np


def cap_threshold(amounts: np.ndarray, quantile: float) -> float:
    """Cap level taken from TRAIN claim amounts only."""
    a = np.asarray(amounts, float)
    if len(a) == 0 or (a <= 0).any() or not np.isfinite(a).all():
        raise ValueError("Claim amounts must be non-empty, finite and strictly positive")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(a, quantile))


def excess_load(amounts: np.ndarray, cap: float) -> float:
    """Portfolio-wide load = total excess over the cap / total capped cost.

    Capped severity x (1 + load) restores the full-claim mean on the training data by construction.
    """
    a = np.asarray(amounts, float)
    if cap <= 0:
        raise ValueError("cap must be positive")
    capped = np.minimum(a, cap)
    return float(np.maximum(a - cap, 0.0).sum() / capped.sum())


def load_sensitivity(amounts: np.ndarray, cap: float, drop_top: int = 1) -> dict:
    """How much of the load comes from the very largest claims? Recomputes the load without them."""
    a = np.sort(np.asarray(amounts, float))
    full = excess_load(a, cap)
    trimmed = excess_load(a[:-drop_top], cap) if drop_top and len(a) > drop_top else full
    return {"load": full, "load_without_top": trimmed, "top_claims_share_of_excess": _top_share(a, cap, drop_top)}


def _top_share(sorted_amounts: np.ndarray, cap: float, k: int) -> float:
    excess = np.maximum(sorted_amounts - cap, 0.0)
    total = excess.sum()
    return float(excess[-k:].sum() / total) if total > 0 else 0.0
