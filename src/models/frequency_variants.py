"""Modern GBM variants for frequency: monotone-constrained and GLM-boosted models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import validate_exposure
from src.models.frequency import LightGBMFrequency, PoissonGLM


class MonotoneLightGBMFrequency(LightGBMFrequency):
    """LightGBM Poisson with monotone constraints, e.g. {"BonusMalus": 1} = claim rate never decreases
    as bonus-malus rises. Encodes domain knowledge, usually costing little accuracy for much better
    explainability and sanity of relativities."""

    def __init__(self, cfg: dict, monotone: dict[str, int], params: dict | None = None):
        super().__init__(cfg, params)
        f = cfg["features"]
        cols = f["numeric"] + f["categorical"]
        unknown = set(monotone) - set(f["numeric"])
        if unknown:
            raise ValueError(f"Monotone constraints only apply to numeric features; got {sorted(unknown)}")
        self.monotone = monotone
        self.params["monotone_constraints"] = [monotone.get(c, 0) for c in cols]


class GLMBoostedFrequency:
    """LightGBM learns a multiplicative correction on top of a Poisson GLM.

    init_score = log(GLM expected counts) (which already contains the log-exposure offset), so the
    final prediction is GLM_counts * exp(boosted correction). With early stopping at zero trees this
    reduces to the GLM, so it can only help if the data support a correction.
    """

    def __init__(self, cfg: dict, params: dict | None = None):
        self.cfg = cfg
        self.glm_ = PoissonGLM(cfg)
        self.gbm_ = LightGBMFrequency(cfg, {"learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 500,
                                            "reg_lambda": 20.0, **(params or {})})

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> GLMBoostedFrequency:
        import lightgbm as lgb
        validate_exposure(train)
        validate_exposure(val)
        self.glm_.fit(train)
        g = self.gbm_
        g.levels_ = {c: sorted(train[c].astype(str).unique()) for c in self.cfg["features"]["categorical"]}
        g.model_ = lgb.LGBMRegressor(**g.params)
        g.model_.fit(g._x(train), train["ClaimNb"].to_numpy(float), init_score=np.log(self.glm_.predict_counts(train)),
                     eval_set=[(g._x(val), val["ClaimNb"].to_numpy(float))],
                     eval_init_score=[np.log(self.glm_.predict_counts(val))],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        self.best_iteration_ = int(g.model_.best_iteration_)
        return self

    def predict_counts(self, df: pd.DataFrame) -> np.ndarray:
        validate_exposure(df)
        raw = self.gbm_.model_.predict(self.gbm_._x(df), raw_score=True)
        return np.exp(raw) * self.glm_.predict_counts(df)
