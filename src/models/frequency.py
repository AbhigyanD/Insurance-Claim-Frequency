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


def estimate_nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """NB2 dispersion alpha (Var = mu + alpha*mu^2) via the Cameron-Trivedi auxiliary regression.

    Regress ((y-mu)^2 - y)/mu on mu with no intercept; the slope estimates alpha. Clipped at a small
    positive value so the NB family stays defined if the data are not overdispersed.
    """
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    z = ((y - mu) ** 2 - y) / mu
    alpha = float((mu * z).sum() / (mu * mu).sum())
    return max(alpha, 1e-6)


class NegBinGLM(PoissonGLM):
    """NB2 GLM, log link, log-exposure offset. Alpha is estimated from a Poisson fit on training data."""

    def fit(self, df: pd.DataFrame) -> "NegBinGLM":
        super().fit(df)
        poisson_mu = np.asarray(self.result_.fittedvalues, float)
        self.alpha_ = estimate_nb_alpha(df["ClaimNb"].to_numpy(), poisson_mu)
        X = glm_design(df, self.cfg, self.columns_)
        self.result_ = sm.GLM(df["ClaimNb"].to_numpy(float), X, family=sm.families.NegativeBinomial(alpha=self.alpha_),
                              offset=np.log(df["Exposure"].to_numpy(float))).fit()
        return self

    def dispersion(self) -> float:
        return float(self.result_.pearson_chi2 / self.result_.df_resid)

    def log_likelihood(self, df: pd.DataFrame) -> float:
        """Mean NB2 log-likelihood per policy under the fitted alpha."""
        from scipy.special import gammaln
        mu, y, a = self.predict_counts(df), df["ClaimNb"].to_numpy(float), self.alpha_
        r = 1.0 / a
        ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))
        return float(ll.mean())


def poisson_log_likelihood(y: np.ndarray, mu: np.ndarray) -> float:
    from scipy.special import gammaln
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float((y * np.log(mu) - mu - gammaln(y + 1)).mean())


class LightGBMFrequency:
    """Gradient-boosted Poisson model; exposure enters as a log offset via init_score.

    The booster models log(rate); predicted counts = exp(raw score) * exposure, so predictions scale
    exactly with exposure like the GLM. Categorical levels are fixed from training data.
    """

    def __init__(self, cfg: dict, params: dict | None = None):
        self.cfg = cfg
        self.params = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": 16, "min_child_samples": 200,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0,
                       "n_estimators": 2000, "random_state": cfg["seed"], "n_jobs": 4, "verbose": -1}
        self.params.update(params or {})

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self.cfg["features"]
        x = df[f["numeric"] + f["categorical"]].copy()
        for c in f["categorical"]:
            x[c] = pd.Categorical(x[c].astype(str), categories=self.levels_[c])
        return x

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "LightGBMFrequency":
        import lightgbm as lgb
        validate_exposure(train); validate_exposure(val)
        self.levels_ = {c: sorted(train[c].astype(str).unique()) for c in self.cfg["features"]["categorical"]}
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(
            self._x(train), train["ClaimNb"].to_numpy(float), init_score=np.log(train["Exposure"].to_numpy(float)),
            eval_set=[(self._x(val), val["ClaimNb"].to_numpy(float))],
            eval_init_score=[np.log(val["Exposure"].to_numpy(float))],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        self.best_iteration_ = int(self.model_.best_iteration_)
        return self

    def predict_counts(self, df: pd.DataFrame) -> np.ndarray:
        validate_exposure(df)
        raw = self.model_.predict(self._x(df), raw_score=True)
        return np.exp(raw) * df["Exposure"].to_numpy(float)
