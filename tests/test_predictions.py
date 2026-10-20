import numpy as np
import pandas as pd
import pytest

from src.data.validate import DataValidationError
from src.models.frequency import ConstantRateBaseline, PoissonGLM, glm_design

CFG = {}


def _toy(n=3000, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "DrivAge": r.integers(18, 90, n), "VehAge": r.integers(0, 25, n), "BonusMalus": r.integers(50, 200, n),
        "VehPower": r.integers(4, 15, n), "Density": r.integers(1, 20000, n),
        "VehBrand": r.choice(["B1", "B2", "B3"], n), "VehGas": r.choice(["Diesel", "Regular"], n),
        "Area": r.choice(list("ABC"), n), "Region": r.choice(["R1", "R2"], n),
        "Exposure": r.uniform(0.05, 1.0, n)})
    d["ClaimNb"] = r.poisson(0.07 * d["Exposure"] * (1 + (d["BonusMalus"] > 100)))
    return d


def test_glm_predictions_finite_nonnegative_and_scale_with_exposure():
    tr, te = _toy(seed=0), _toy(seed=1)
    m = PoissonGLM(CFG).fit(tr)
    mu = m.predict_counts(te)
    assert np.isfinite(mu).all() and (mu >= 0).all() and len(mu) == len(te)
    half = te.assign(Exposure=te["Exposure"] / 2)
    assert np.allclose(m.predict_counts(half), mu / 2)          # log-exposure offset


def test_inference_schema_matches_training_even_with_unseen_category():
    tr, te = _toy(seed=0), _toy(seed=1)
    te.loc[te.index[:5], "VehBrand"] = "NEW"
    cols = list(glm_design(tr, CFG).columns)
    assert list(glm_design(te, CFG, cols).columns) == cols


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_invalid_exposure_rejected_at_fit_and_predict(bad):
    tr = _toy()
    broken = tr.copy()
    broken.loc[broken.index[0], "Exposure"] = bad
    with pytest.raises(DataValidationError):
        PoissonGLM(CFG).fit(broken)
    with pytest.raises(DataValidationError):
        ConstantRateBaseline().fit(tr).predict_counts(broken)


def test_negbin_and_lightgbm_predictions_valid_and_scale_with_exposure():
    from src.models.frequency import LightGBMFrequency, NegBinGLM
    cfg = {"seed": 1, "features": {"numeric": ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
                                   "categorical": ["VehBrand", "VehGas", "Area", "Region"]}}
    tr, va, te = _toy(seed=0), _toy(seed=2), _toy(seed=1)
    for m in (NegBinGLM(cfg).fit(tr), LightGBMFrequency(cfg, {"n_estimators": 30}).fit(tr, va)):
        mu = m.predict_counts(te)
        assert np.isfinite(mu).all() and (mu >= 0).all()
        assert np.allclose(m.predict_counts(te.assign(Exposure=te["Exposure"] / 2)), mu / 2)


def test_severity_models_positive_finite_and_exclude_zero_claims():
    import pytest as _pt

    from src.models.severity import GammaGLM, LightGBMSeverity, claims_table
    cfg = {"seed": 1, "features": {"numeric": ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
                                   "categorical": ["VehBrand", "VehGas", "Area", "Region"]}}
    r = np.random.default_rng(3)
    pol = _toy(2000, seed=0).assign(IDpol=np.arange(2000), TotalClaimAmount=0.0)
    claims = pd.DataFrame({"IDpol": r.integers(0, 2000, 800), "ClaimAmount": r.gamma(2.0, 1000, 800)})
    tr = claims_table(pol, claims)
    assert len(tr) == len(claims) and (tr["ClaimAmount"] > 0).all()
    with _pt.raises(ValueError):
        claims_table(pol, claims.assign(ClaimAmount=0.0))
    for m in (GammaGLM(cfg).fit(tr), GammaGLM(cfg, 0.9).fit(tr), LightGBMSeverity(cfg, {"n_estimators": 20}).fit(tr, tr)):
        p = m.predict(tr)
        assert np.isfinite(p).all() and (p > 0).all()
