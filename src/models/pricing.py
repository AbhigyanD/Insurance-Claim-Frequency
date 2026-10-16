"""Combine frequency and severity into expected claims cost (pure premium, not a final price)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data.validate import validate_exposure


def expected_cost_table(df: pd.DataFrame, freq_counts: np.ndarray, severity: np.ndarray) -> pd.DataFrame:
    """Policy-level predictions.

    - predicted_frequency: claims per exposure-year  = predicted counts / exposure
    - predicted_severity: expected cost per claim
    - predicted_annual_cost: frequency * severity (cost of a FULL year of cover, comparable across policies)
    - predicted_cost_exposure: counts * severity (cost over the exposure actually observed; compare with actuals)

    Invalid outputs raise instead of being clipped, so they get investigated.
    """
    validate_exposure(df)
    freq_counts, severity = np.asarray(freq_counts, float), np.asarray(severity, float)
    if not (len(df) == len(freq_counts) == len(severity)):
        raise ValueError("Prediction length does not match the number of policies")
    if not np.isfinite(freq_counts).all() or (freq_counts < 0).any():
        raise ValueError("Frequency predictions must be finite and non-negative")
    if not np.isfinite(severity).all() or (severity <= 0).any():
        raise ValueError("Severity predictions must be finite and strictly positive")
    out = pd.DataFrame({"IDpol": df["IDpol"].to_numpy()})
    out["predicted_frequency"] = freq_counts / df["Exposure"].to_numpy(float)
    out["predicted_severity"] = severity
    out["predicted_annual_cost"] = out["predicted_frequency"] * out["predicted_severity"]
    out["predicted_cost_exposure"] = freq_counts * severity
    if not np.isfinite(out[["predicted_annual_cost", "predicted_cost_exposure"]].to_numpy()).all():
        raise ValueError("Non-finite expected cost produced")
    return out


def segment_edges(train_annual_cost: np.ndarray, n_segments: int) -> np.ndarray:
    """Cut points from TRAIN predictions only (no information from validation/test)."""
    return np.quantile(train_annual_cost, np.linspace(0, 1, n_segments + 1)[1:-1])


def assign_segments(annual_cost: np.ndarray, edges: np.ndarray, labels: list[str]) -> np.ndarray:
    if len(labels) != len(edges) + 1:
        raise ValueError("labels must have len(edges)+1 entries")
    return np.asarray(labels, dtype=object)[np.searchsorted(edges, annual_cost, side="right")]
