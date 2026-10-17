"""Population Stability Index (PSI) for monitoring feature and prediction drift."""
from __future__ import annotations
import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10, eps: float = 1e-4) -> float:
    """PSI between a reference ('expected') and a new ('actual') sample.

    Numeric: bins are quantiles of the reference. Categorical: one bin per level, plus unseen levels.
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift (a convention, not a test).
    """
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("Empty sample")
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:                                    # (near-)constant reference
            return 0.0 if np.allclose(np.mean(actual), np.mean(expected)) else float("inf")
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(sort=False)
        a = pd.cut(actual, edges).value_counts(sort=False)
    else:
        levels = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts().reindex(levels, fill_value=0)
        a = actual.astype(str).value_counts().reindex(levels, fill_value=0)
    pe = np.clip(e.to_numpy() / e.sum(), eps, None)
    pa = np.clip(a.to_numpy() / a.sum(), eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(reference: pd.DataFrame, new: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(reference[c], new[c])} for c in columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "major"])
    return out
