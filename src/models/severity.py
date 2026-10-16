"""Severity models. Trained ONLY on claims with a positive amount (one row per claim)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.frequency import glm_design


def claims_table(policies: pd.DataFrame, claims: pd.DataFrame, id_col: str = "IDpol") -> pd.DataFrame:
    """One row per claim with the policy's risk features. Restricted to the given split's policies."""
    out = claims.merge(policies.drop(columns=["ClaimNb", "TotalClaimAmount"]), on=id_col, how="inner")
    if (out["ClaimAmount"] <= 0).any():
        raise ValueError("Non-positive claim amounts found in severity data")
    return out


class ConstantSeverityBaseline:
    def fit(self, df: pd.DataFrame) -> "ConstantSeverityBaseline":
        self.mean_ = float(df["ClaimAmount"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_)


class GammaGLM:
    """Gamma GLM, log link. If cap_quantile is set, training claims are capped at that TRAIN quantile
    (a documented choice to limit the pull of extreme claims; the cap is learned from train only)."""

    def __init__(self, cfg: dict, cap_quantile: float | None = None):
        self.cfg, self.cap_quantile = cfg, cap_quantile

    def fit(self, df: pd.DataFrame) -> "GammaGLM":
        y = df["ClaimAmount"].to_numpy(float)
        if self.cap_quantile is not None:
            self.cap_ = float(np.quantile(y, self.cap_quantile))
            y = np.minimum(y, self.cap_)
        X = glm_design(df, self.cfg)
        self.columns_ = list(X.columns)
        self.result_ = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.result_.predict(glm_design(df, self.cfg, self.columns_)), float)


class LightGBMSeverity:
    """Gradient-boosted Gamma (log-link) model. Early stopping on the validation claims."""

    def __init__(self, cfg: dict, params: dict | None = None):
        self.cfg = cfg
        self.params = {"objective": "gamma", "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 100,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 10.0,
                       "n_estimators": 2000, "random_state": cfg["seed"], "n_jobs": 4, "verbose": -1}
        self.params.update(params or {})

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self.cfg["features"]
        x = df[f["numeric"] + f["categorical"]].copy()
        for c in f["categorical"]:
            x[c] = pd.Categorical(x[c].astype(str), categories=self.levels_[c])
        return x

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "LightGBMSeverity":
        import lightgbm as lgb
        self.levels_ = {c: sorted(train[c].astype(str).unique()) for c in self.cfg["features"]["categorical"]}
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(self._x(train), train["ClaimAmount"].to_numpy(float),
                        eval_set=[(self._x(val), val["ClaimAmount"].to_numpy(float))],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        self.best_iteration_ = int(self.model_.best_iteration_)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._x(df)), float)
