"""Frequency models. Targets are claim COUNTS; exposure enters as a log offset (or sample weight)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.validate import validate_exposure


class ConstantRateBaseline:
    """Predicts the portfolio-wide claim rate per exposure-year (training data only)."""

    def fit(self, df: pd.DataFrame) -> "ConstantRateBaseline":
        self.rate_ = float(df["ClaimNb"].sum() / df["Exposure"].sum())
        return self

    def predict_counts(self, df: pd.DataFrame) -> np.ndarray:
        validate_exposure(df)
        return self.rate_ * df["Exposure"].to_numpy(float)


def glm_design(df: pd.DataFrame, cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """Actuarial-style GLM design matrix: banded continuous rating factors + categorical dummies.

    Bands/caps are fixed a priori (not learned from the target), so no leakage. Pass the
    training `columns` at inference time to guarantee an identical schema.
    """
    x = pd.DataFrame(index=df.index)
    x["DrivAge_band"] = pd.cut(df["DrivAge"], [17, 21, 25, 30, 40, 50, 60, 70, 120]).astype(str)
    x["VehAge_band"] = pd.cut(df["VehAge"], [-1, 0, 1, 2, 5, 10, 15, 100]).astype(str)
    x["BonusMalus_band"] = pd.cut(df["BonusMalus"].clip(upper=150), [49, 50, 55, 60, 70, 80, 100, 125, 150]).astype(str)
    x["VehPower"] = df["VehPower"].clip(upper=9).astype(str)
    x["LogDensity"] = np.log(df["Density"].clip(lower=1))
    for c in ("VehBrand", "VehGas", "Area", "Region"):
        x[c] = df[c].astype(str)
    X = pd.get_dummies(x, drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class PoissonGLM:
    """Poisson GLM, log link, offset = log(Exposure). Equivalent to modelling rates with exposure weights."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, df: pd.DataFrame) -> "PoissonGLM":
        validate_exposure(df)
        X = glm_design(df, self.cfg)
        self.columns_ = list(X.columns)
        self.result_ = sm.GLM(df["ClaimNb"].to_numpy(float), X, family=sm.families.Poisson(),
                              offset=np.log(df["Exposure"].to_numpy(float))).fit()
        return self

    def predict_counts(self, df: pd.DataFrame) -> np.ndarray:
        validate_exposure(df)
        X = glm_design(df, self.cfg, self.columns_)
        mu = self.result_.predict(X, offset=np.log(df["Exposure"].to_numpy(float)))
        return np.asarray(mu, float)

    def dispersion(self) -> float:
        """Pearson chi2 / residual df. Values clearly above 1 indicate overdispersion."""
        return float(self.result_.pearson_chi2 / self.result_.df_resid)
