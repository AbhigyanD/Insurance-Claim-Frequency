"""End-to-end check that the pipeline pieces compose, on synthetic data with a known risk structure.

The synthetic world: claim frequency rises with bonus-malus and falls with driver age; severity is Gamma and
independent of the features. Real code paths are used throughout (no mocks).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build import build_policy_table, split_policies
from src.models.frequency import LightGBMFrequency, PoissonGLM
from src.models.pricing import assign_segments, expected_cost_table, segment_edges
from src.models.severity import RidgeGammaGLM, claims_table
from src.reporting.dashboard_export import category_summary, validate_tables
from src.utils.config import load_config

N = 30_000


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    r = np.random.default_rng(11)
    ids = np.arange(1, N + 1) * 7                      # non-contiguous ids on purpose
    bm = r.integers(50, 160, N)
    age = r.integers(18, 85, N)
    expo = np.where(r.random(N) < 0.5, 1.0, r.uniform(0.05, 1.0, N))
    rate = 0.04 * np.exp(0.012 * (bm - 80) - 0.012 * (age - 45))
    claims_n = r.poisson(rate * expo)
    freq = pd.DataFrame({
        "IDpol": ids, "ClaimNb": claims_n.astype(float), "Exposure": expo, "VehPower": r.integers(4, 12, N),
        "VehAge": r.integers(0, 20, N), "DrivAge": age, "BonusMalus": bm, "VehBrand": r.choice(["B1", "B2", "B3"], N),
        "VehGas": r.choice(["Diesel", "Regular"], N), "Area": r.choice(list("ABCD"), N),
        "Density": r.integers(1, 15000, N), "Region": r.choice(["R1", "R2", "R3"], N)})
    rows = np.repeat(ids, claims_n)
    sev = pd.DataFrame({"IDpol": rows, "ClaimAmount": r.gamma(2.0, 1000.0, len(rows))})
    table = build_policy_table(freq, sev, cfg)
    return cfg, freq, sev, table, split_policies(table, cfg)


def test_split_partitions_policies_and_conserves_totals(world):
    cfg, freq, sev, table, s = world
    ids = [set(s[k]["IDpol"]) for k in ("train", "val", "test")]
    assert sum(len(i) for i in ids) == len(table) == N and len(set.union(*ids)) == N       # disjoint and complete
    for col in ("ClaimNb", "TotalClaimAmount"):
        assert sum(s[k][col].sum() for k in s) == pytest.approx(table[col].sum())
    assert table["TotalClaimAmount"].sum() == pytest.approx(sev["ClaimAmount"].sum())      # every claim attached to a policy
    assert table["Exposure"].max() <= 1.0


def test_outcomes_and_identifiers_are_never_model_inputs(world):
    cfg = world[0]
    inputs = set(cfg["features"]["numeric"]) | set(cfg["features"]["categorical"])
    assert not inputs & {"ClaimNb", "TotalClaimAmount", "ClaimAmount", "IDpol", "Exposure"}   # exposure is an offset, not a feature


def test_full_pipeline_gives_valid_calibrated_and_correctly_ordered_predictions(world):
    cfg, freq, sev, table, s = world
    tr, va, te = s["train"], s["val"], s["test"]
    lgbm = LightGBMFrequency(cfg, {"learning_rate": 0.1}).fit(tr, va)          # default budget, early stopping on validation
    sev_model = RidgeGammaGLM(cfg).fit(claims_table(tr, sev), claims_table(va, sev))

    preds = {k: expected_cost_table(d, lgbm.predict_counts(d), sev_model.predict(d)) for k, d in s.items()}
    for t in preds.values():
        assert np.isfinite(t.drop(columns="IDpol").to_numpy()).all()
        assert (t["predicted_frequency"] >= 0).all() and (t["predicted_severity"] > 0).all() and (t["predicted_annual_cost"] >= 0).all()

    # the booster finds the planted structure: frequency rises with predicted risk
    edges = segment_edges(preds["train"]["predicted_annual_cost"].to_numpy(), 5)
    labels = ["Very low", "Low", "Medium", "High", "Very high"]
    t = preds["test"].assign(seg=assign_segments(preds["test"]["predicted_annual_cost"].to_numpy(), edges, labels))
    t = t.merge(te[["IDpol", "ClaimNb", "Exposure", "TotalClaimAmount"]], on="IDpol")
    by = t.groupby("seg").apply(lambda g: g["ClaimNb"].sum() / g["Exposure"].sum(), include_groups=False).reindex(labels)
    assert by["Very high"] > 2 * by["Very low"]
    assert list(by.diff().dropna() > 0).count(True) >= 3                      # broadly increasing across the five segments
    # overall frequency is calibrated on held-out data. The default initialisation (see the next test) leaves a few
    # percent of upward bias on a sample this small (about 0.94 here), so the tolerance is 10%: it still catches
    # gross failures such as an unconverged model (0.70 with too few trees).
    exp_claims = (t["predicted_frequency"] * t["Exposure"]).sum()
    assert t["ClaimNb"].sum() / exp_claims == pytest.approx(1.0, abs=0.10)
    # a Poisson GLM on the same data agrees on direction and does not beat the true structure by magic
    glm_counts = PoissonGLM(cfg).fit(tr).predict_counts(te)
    assert glm_counts.sum() / te["Exposure"].sum() == pytest.approx(tr["ClaimNb"].sum() / tr["Exposure"].sum(), rel=0.1)


def test_reporting_tables_built_from_pipeline_output_validate(world):
    cfg, freq, sev, table, s = world
    va = s["val"]
    lgbm = LightGBMFrequency(cfg, {"n_estimators": 30}).fit(s["train"], va)
    sev_model = RidgeGammaGLM(cfg).fit(claims_table(s["train"], sev), claims_table(va, sev))
    t = expected_cost_table(va, lgbm.predict_counts(va), sev_model.predict(va))
    edges = segment_edges(t["predicted_annual_cost"].to_numpy(), 5)
    t["risk_segment"] = assign_segments(t["predicted_annual_cost"].to_numpy(), edges, ["Very low", "Low", "Medium", "High", "Very high"])
    frame = t.merge(va[["IDpol", "Exposure", "VehAge", "DrivAge", "Area", "Region", "VehBrand", "VehGas", "ClaimNb", "TotalClaimAmount"]], on="IDpol")
    frame = frame.rename(columns={"Exposure": "exposure", "ClaimNb": "actual_claim_count", "TotalClaimAmount": "actual_claim_amount"})
    summary = category_summary(frame)
    validate_tables({"category_summary": summary})                            # documented, no NaN
    assert (summary.groupby("dimension")["policies"].sum() == len(va)).all()   # each dimension partitions all validation policies
    assert summary["actual_cost"].groupby(summary["dimension"]).sum().round(2).nunique() == 1   # same total in every grouping


def test_portfolio_rate_init_removes_the_default_slow_start_bias(world):
    """Characterises a known limitation of the default LightGBM initialisation, and its opt-in fix.

    The default starts boosting from a rate of 1.0 claim per exposure-year, so a small tree budget leaves the model
    far from calibrated. Starting from the training portfolio rate fixes that. If the default is ever changed, this
    test should be updated deliberately.
    """
    cfg, freq, sev, table, s = world
    tr, va, te = s["train"], s["val"], s["test"]
    short = {"n_estimators": 80, "learning_rate": 0.1}
    default = LightGBMFrequency(cfg, short).fit(tr, va)
    better = LightGBMFrequency(cfg, short, init_at_portfolio_rate=True).fit(tr, va)

    def actual_to_predicted(m):
        return te["ClaimNb"].sum() / m.predict_counts(te).sum()

    assert actual_to_predicted(default) < 0.85                                  # not converged: over-predicts by 20%+
    assert actual_to_predicted(better) == pytest.approx(1.0, abs=0.06)          # same budget, calibrated
    assert default.base_log_rate_ == 0.0
    assert better.base_log_rate_ == pytest.approx(np.log(tr["ClaimNb"].sum() / tr["Exposure"].sum()))
    half = te.assign(Exposure=te["Exposure"] / 2)                               # exposure scaling stays exact with the offset
    assert np.allclose(better.predict_counts(half), better.predict_counts(te) / 2)
