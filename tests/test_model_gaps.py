"""Targeted tests for model and utility branches that other tests did not reach."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data.validate import DataValidationError, quality_report
from src.evaluation.drift import psi
from src.evaluation.metrics import severity_summary
from src.models.frequency import (
    ConstantRateBaseline,
    NegBinGLM,
    estimate_nb_alpha,
    poisson_log_likelihood,
)
from src.models.pricing import expected_cost_table
from src.models.pure_premium import TweedieLightGBM
from src.models.severity import ConstantSeverityBaseline, RidgeGammaGLM
from tests.test_predictions import _toy

CFG = {"seed": 1, "features": {"numeric": ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
                               "categorical": ["VehBrand", "VehGas", "Area", "Region"]}}


def test_estimate_nb_alpha_recovers_known_overdispersion_and_floors_at_poisson():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.05, 0.4, 200_000)
    alpha = 0.8
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))           # NB2: Var = mu + alpha * mu^2
    assert estimate_nb_alpha(y, mu) == pytest.approx(alpha, rel=0.2)
    assert 1e-6 <= estimate_nb_alpha(rng.poisson(mu), mu) < 0.2         # Poisson data: alpha near 0 but never below the floor


def test_log_likelihoods_match_scipy():
    tr, te = _toy(3000, 0), _toy(1500, 1)
    nb = NegBinGLM(CFG).fit(tr)
    mu, y, r = nb.predict_counts(te), te["ClaimNb"].to_numpy(), 1 / nb.alpha_
    expected = stats.nbinom.logpmf(y, n=r, p=r / (r + mu)).mean()
    assert nb.log_likelihood(te) == pytest.approx(expected, rel=1e-9)
    assert poisson_log_likelihood(y, mu) == pytest.approx(stats.poisson.logpmf(y, mu).mean(), rel=1e-9)
    assert np.isfinite(nb.dispersion())


def test_constant_rate_baseline_uses_exposure_weighted_rate_and_validates():
    df = pd.DataFrame({"ClaimNb": [1, 0, 3], "Exposure": [0.5, 0.5, 1.0]})
    m = ConstantRateBaseline().fit(df)
    assert m.rate_ == pytest.approx(4 / 2.0)                              # total claims / total exposure, not mean of rates
    assert np.allclose(m.predict_counts(df), [1.0, 1.0, 2.0])
    with pytest.raises(DataValidationError):
        m.predict_counts(df.assign(Exposure=[0.5, 0.0, 1.0]))


def _claims(n=1500, seed=3):
    r = np.random.default_rng(seed)
    d = _toy(n, seed)
    return d.assign(ClaimAmount=r.gamma(2.0, 1000, n) * (1 + 0.5 * (d["VehGas"] == "Diesel")))


def test_constant_severity_baseline_is_the_training_mean():
    tr = _claims()
    m = ConstantSeverityBaseline().fit(tr)
    assert m.mean_ == pytest.approx(tr["ClaimAmount"].mean())
    assert np.all(m.predict(tr.head(5)) == m.mean_)


def test_ridge_gamma_glm_picks_the_best_alpha_on_validation():
    tr, va = _claims(2000, 3), _claims(1000, 4)
    alphas = (1e-3, 1e-1, 10.0)
    m = RidgeGammaGLM(CFG, alphas).fit(tr, va)
    assert set(m.search_) == set(alphas) and m.alpha_ in alphas
    assert m.search_[m.alpha_] == min(m.search_.values())                 # chosen alpha has the lowest validation deviance
    p = m.predict(va)
    assert np.isfinite(p).all() and (p > 0).all() and len(p) == len(va)


def test_expected_cost_overflow_is_an_error_not_a_silent_inf():
    df = pd.DataFrame({"IDpol": [1], "Exposure": [1.0]})
    with pytest.raises(ValueError, match="Non-finite expected cost"):
        expected_cost_table(df, np.array([1e308]), np.array([1e308]))   # each finite, product overflows


def test_tweedie_rejects_invalid_model_output(monkeypatch):
    tr, va = _toy(1500, 0).assign(TotalClaimAmount=0.0), _toy(800, 2).assign(TotalClaimAmount=0.0)
    tr["TotalClaimAmount"] = tr["ClaimNb"] * 1000.0
    va["TotalClaimAmount"] = va["ClaimNb"] * 1000.0
    m = TweedieLightGBM(CFG, 1.5, {"n_estimators": 10}).fit(tr, va)
    monkeypatch.setattr(m.model_, "predict", lambda X: np.full(len(X), np.nan))
    with pytest.raises(ValueError, match="Invalid Tweedie"):
        m.predict_annual_cost(va)


def test_psi_constant_reference():
    const = pd.Series([5.0] * 100)
    assert psi(const, pd.Series([5.0] * 50)) == 0.0
    assert psi(const, pd.Series([9.0] * 50)) == float("inf")


def test_severity_summary_known_values():
    s = severity_summary(np.array([100.0, 300.0]), np.array([200.0, 200.0]))
    assert s["mae"] == 100.0 and s["rmse"] == 100.0
    assert s["actual_mean"] == 200.0 and s["predicted_mean"] == 200.0 and s["actual_to_predicted"] == 1.0
    assert s["gamma_deviance"] > 0


def test_quality_report_counts_integrity_problems():
    freq = pd.DataFrame({"IDpol": [1, 2, 2, 3], "ClaimNb": [1, 0, 0, 2], "Exposure": [1.0, 1.5, 0.5, 1.0]})
    sev = pd.DataFrame({"IDpol": [1, 3, 99], "ClaimAmount": [10.0, 20.0, 30.0]})
    # duplicate ids would make the join ambiguous, so use the realistic case of unique policies
    freq = freq.drop_duplicates("IDpol").reset_index(drop=True)
    q = quality_report(freq, sev, "IDpol")
    assert q["n_policies"] == 3 and q["n_severity_rows"] == 3
    assert q["exposure_gt_1"] == 1 and q["sev_ids_not_in_freq"] == 1
    assert q["claimnb_ne_sev_rows"] == 1                                  # policy 3 has ClaimNb=2 but one severity row
    assert q["claimnb_gt0_without_sev"] == 0 and q["sev_rows_for_zero_claimnb"] == 0


def test_load_config_reads_yaml_and_tracking_creates_sqlite_dir(tmp_path, monkeypatch):
    from src.utils import tracking
    from src.utils.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\nmlflow:\n  tracking_uri: sqlite:///sub/dir/m.db\n")
    cfg = load_config(p)
    assert cfg["seed"] == 7
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: None)
    tracking._setup(cfg, "exp")
    assert (tmp_path / "sub" / "dir").is_dir()                            # relative sqlite path is created under the project root


def test_poisson_glm_dispersion_is_finite_and_positive():
    from src.models.frequency import PoissonGLM
    d = PoissonGLM(CFG).fit(_toy(3000, 0)).dispersion()
    assert np.isfinite(d) and d > 0


def test_insights_pick_requires_exactly_one_row():
    from src.reporting.insights import _pick
    df = pd.DataFrame({"model": ["a", "a", "b"], "x": [1, 2, 3]})
    assert _pick(df, model="b")["x"] == 3
    with pytest.raises(ValueError, match="exactly one row"):
        _pick(df, model="a")
    with pytest.raises(ValueError, match="exactly one row"):
        _pick(df, model="missing")


def test_log_run_uploads_only_existing_artifacts(tmp_path):
    import mlflow

    from src.utils.tracking import log_run
    cfg = {"seed": 1, "mlflow": {"tracking_uri": f"sqlite:////{tmp_path}/m.db"}}
    real = tmp_path / "metrics.csv"
    real.write_text("a,b\n1,2\n")
    rid = log_run(cfg, "artifacts", "run", {"p": 1}, {"m": 1.0}, artifacts=[real, tmp_path / "does_not_exist.csv"])
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    names = [a.path for a in mlflow.MlflowClient().list_artifacts(rid)]
    assert names == ["metrics.csv"]                                      # the missing file is skipped, not an error
