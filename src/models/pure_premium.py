"""Direct Tweedie pure-premium model: one model for loss cost per exposure-year."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import validate_exposure


class TweedieLightGBM:
    """LightGBM Tweedie on rate = TotalClaimAmount / Exposure with exposure as the sample weight.

    Predicts the expected loss cost per exposure-year directly (compound Poisson-Gamma assumption,
    variance power between 1 and 2), as an alternative to frequency x severity.
    """

    def __init__(self, cfg: dict, variance_power: float = 1.5, params: dict | None = None):
        if not 1.0 < variance_power < 2.0:
            raise ValueError("variance_power must be in (1, 2) for a compound Poisson-Gamma model")
        self.cfg, self.variance_power = cfg, variance_power
        self.params = {"objective": "tweedie", "tweedie_variance_power": variance_power, "learning_rate": 0.03,
                       "num_leaves": 8, "min_child_samples": 500, "subsample": 0.8, "subsample_freq": 1,
                       "colsample_bytree": 0.8, "reg_lambda": 20.0, "n_estimators": 2000,
                       "random_state": cfg["seed"], "n_jobs": 4, "verbose": -1, **(params or {})}

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self.cfg["features"]
        x = df[f["numeric"] + f["categorical"]].copy()
        for c in f["categorical"]:
            x[c] = pd.Categorical(x[c].astype(str), categories=self.levels_[c])
        return x

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> TweedieLightGBM:
        import lightgbm as lgb
        validate_exposure(train)
        validate_exposure(val)
        self.levels_ = {c: sorted(train[c].astype(str).unique()) for c in self.cfg["features"]["categorical"]}
        def rate(d: pd.DataFrame) -> np.ndarray:
            return d["TotalClaimAmount"].to_numpy(float) / d["Exposure"].to_numpy(float)

        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(self._x(train), rate(train), sample_weight=train["Exposure"].to_numpy(float),
                        eval_set=[(self._x(val), rate(val))], eval_sample_weight=[val["Exposure"].to_numpy(float)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        self.best_iteration_ = int(self.model_.best_iteration_)
        return self

    def predict_annual_cost(self, df: pd.DataFrame) -> np.ndarray:
        """Expected loss cost per exposure-year."""
        validate_exposure(df)
        p = np.asarray(self.model_.predict(self._x(df)), float)
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid Tweedie predictions (non-finite or negative)")
        return p
