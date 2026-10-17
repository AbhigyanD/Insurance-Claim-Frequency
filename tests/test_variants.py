import numpy as np
import pandas as pd
import pytest

from src.evaluation.drift import psi, psi_table
from src.models.frequency_variants import GLMBoostedFrequency, MonotoneLightGBMFrequency
from src.models.pure_premium import TweedieLightGBM
from tests.test_predictions import _toy

CFG = {"seed": 1, "features": {"numeric": ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
                               "categorical": ["VehBrand", "VehGas", "Area", "Region"]}}


def test_psi_zero_for_identical_and_large_for_shift_and_categorical():
    r = np.random.default_rng(0)
    a = pd.Series(r.normal(0, 1, 5000))
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, pd.Series(r.normal(0, 1, 5000))) < 0.05
    assert psi(a, pd.Series(r.normal(1.5, 1, 5000))) > 0.25
    c1, c2 = pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 50 + ["b"] * 40 + ["new"] * 10)
    assert psi(c1, c2) > 0.25
    t = psi_table(pd.DataFrame({"x": a}), pd.DataFrame({"x": a + 2}), ["x"])
    assert t["status"].iloc[0] == "major"
    with pytest.raises(ValueError):
        psi(a, pd.Series([], dtype=float))


def test_monotone_constraint_is_respected_and_validated():
    tr, va = _toy(4000, 0), _toy(2000, 2)
    m = MonotoneLightGBMFrequency(CFG, {"BonusMalus": 1}, {"n_estimators": 60}).fit(tr, va)
    base = _toy(200, 5)
    grid = []
    for bm in range(50, 200, 10):
        grid.append(m.predict_counts(base.assign(BonusMalus=bm)))
    grid = np.array(grid)
    assert (np.diff(grid, axis=0) >= -1e-12).all()          # never decreases with BonusMalus
    with pytest.raises(ValueError):
        MonotoneLightGBMFrequency(CFG, {"VehBrand": 1})


def test_glm_boosted_and_tweedie_valid_outputs():
    tr, va, te = _toy(4000, 0), _toy(2000, 2), _toy(1000, 1)
    gb = GLMBoostedFrequency(CFG, {"n_estimators": 30}).fit(tr, va)
    mu = gb.predict_counts(te)
    assert np.isfinite(mu).all() and (mu >= 0).all()
    assert np.allclose(gb.predict_counts(te.assign(Exposure=te["Exposure"] / 2)), mu / 2)
    for d in (tr, va):
        d["TotalClaimAmount"] = d["ClaimNb"] * 1500.0
    tw = TweedieLightGBM(CFG, 1.5, {"n_estimators": 30}).fit(tr, va)
    p = tw.predict_annual_cost(te)
    assert np.isfinite(p).all() and (p >= 0).all()
    with pytest.raises(ValueError):
        TweedieLightGBM(CFG, 2.5)
