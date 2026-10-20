"""Frequency / severity evaluation metrics. All frequency metrics are exposure-aware."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_counts(y: np.ndarray, mu: np.ndarray) -> None:
    if not np.isfinite(mu).all():
        raise ValueError("Non-finite predictions")
    if (mu < 0).any():
        raise ValueError("Negative predictions")


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance per policy for predicted *counts* mu (= rate * exposure).

    2 * [ y*log(y/mu) - (y - mu) ], with y*log(y/mu) = 0 when y = 0.
    """
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    _check_counts(y, mu)
    if (mu == 0).any():
        raise ValueError("Zero predicted mean makes Poisson deviance undefined")
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def frequency_summary(y: np.ndarray, mu: np.ndarray, exposure: np.ndarray) -> dict:
    """Metrics on counts: Poisson deviance, MAE, and overall actual/predicted ratio."""
    y, mu, exposure = (np.asarray(a, float) for a in (y, mu, exposure))
    return {
        "poisson_deviance": poisson_deviance(y, mu),
        "mae_counts": float(np.mean(np.abs(y - mu))),
        "actual_freq": float(y.sum() / exposure.sum()),
        "predicted_freq": float(mu.sum() / exposure.sum()),
        "actual_to_predicted": float(y.sum() / mu.sum()),
    }


def lift_table(y: np.ndarray, mu: np.ndarray, exposure: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rank policies by predicted frequency and group into exposure-equal bins.

    Returns actual vs predicted frequency per bin (calibration) and lift vs the portfolio mean.
    """
    d = pd.DataFrame({"y": y, "mu": mu, "e": exposure})
    d["rate"] = d["mu"] / d["e"]
    d = d.sort_values("rate").reset_index(drop=True)
    d["bin"] = np.minimum((d["e"].cumsum() / d["e"].sum() * n_bins).astype(int), n_bins - 1) + 1
    g = d.groupby("bin").agg(policies=("y", "size"), exposure=("e", "sum"), claims=("y", "sum"), expected=("mu", "sum"))
    g["actual_freq"] = g["claims"] / g["exposure"]
    g["predicted_freq"] = g["expected"] / g["exposure"]
    g["actual_to_predicted"] = g["claims"] / g["expected"]
    g["lift_vs_mean"] = g["actual_freq"] / (g["claims"].sum() / g["exposure"].sum())
    return g.reset_index()


def gini(y: np.ndarray, score: np.ndarray, exposure: np.ndarray) -> float:
    """Exposure-weighted Gini of ranking by predicted rate (0 = random, higher = better ranking)."""
    d = pd.DataFrame({"y": y, "s": np.asarray(score) / np.asarray(exposure), "e": exposure}).sort_values("s")
    cum_e = d["e"].cumsum() / d["e"].sum()
    cum_y = d["y"].cumsum() / d["y"].sum()
    area = np.trapezoid(cum_y.to_numpy(), cum_e.to_numpy())
    return float(1 - 2 * area)


def gamma_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Gamma deviance: 2 * [ (y-mu)/mu - log(y/mu) ]. Requires y > 0 and mu > 0."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError("Severity predictions must be finite and strictly positive")
    if (y <= 0).any():
        raise ValueError("Gamma deviance needs strictly positive actual severities")
    return float(np.mean(2.0 * ((y - mu) / mu - np.log(y / mu))))


def severity_summary(y: np.ndarray, mu: np.ndarray) -> dict:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return {
        "gamma_deviance": gamma_deviance(y, mu),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "actual_mean": float(y.mean()),
        "predicted_mean": float(mu.mean()),
        "actual_to_predicted": float(y.sum() / mu.sum()),
    }


def severity_calibration(y: np.ndarray, mu: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Group claims by predicted severity quantile; compare actual vs predicted mean severity."""
    d = pd.DataFrame({"y": y, "mu": mu})
    d["bin"] = pd.qcut(d["mu"].rank(method="first"), n_bins, labels=False) + 1
    g = d.groupby("bin").agg(claims=("y", "size"), actual_mean=("y", "mean"), predicted_mean=("mu", "mean"))
    g["actual_to_predicted"] = g["actual_mean"] / g["predicted_mean"]
    return g.reset_index()


def large_claim_analysis(y: np.ndarray, mu: np.ndarray, quantile: float = 0.99) -> dict:
    """How much of total cost sits in the top claims, and how far predictions fall short there."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    thr = np.quantile(y, quantile)
    m = y >= thr
    return {"threshold": float(thr), "n_large": int(m.sum()), "share_of_total_cost": float(y[m].sum() / y.sum()),
            "actual_mean_large": float(y[m].mean()), "predicted_mean_large": float(mu[m].mean()),
            "actual_to_predicted_large": float(y[m].sum() / mu[m].sum())}
