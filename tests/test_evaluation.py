import numpy as np
import pytest

from src.evaluation.metrics import frequency_summary, gini, lift_table, poisson_deviance


def test_poisson_deviance_zero_for_perfect_and_known_value():
    y = np.array([1.0, 2.0, 3.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0)
    # y=0, mu=0.5 -> deviance = 2*0.5 = 1.0
    assert poisson_deviance(np.array([0.0]), np.array([0.5])) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([-1.0]), np.array([0.0])])
def test_invalid_predictions_raise(bad):
    with pytest.raises(ValueError):
        poisson_deviance(np.array([1.0]), bad)


def test_summary_and_lift_are_consistent():
    rng = np.random.default_rng(1)
    e = rng.uniform(0.2, 1.0, 5000)
    rate = rng.uniform(0.02, 0.2, 5000)
    y = rng.poisson(rate * e)
    mu = rate * e
    s = frequency_summary(y, mu, e)
    assert 0.9 < s["actual_to_predicted"] < 1.1
    lt = lift_table(y, mu, e, n_bins=5)
    assert lt["policies"].sum() == 5000 and len(lt) == 5
    assert lt["actual_freq"].iloc[-1] > lt["actual_freq"].iloc[0]   # ranking is informative
    assert gini(y, mu, e) > 0


def test_gamma_deviance_known_and_validation():
    from src.evaluation.metrics import gamma_deviance, severity_calibration, large_claim_analysis
    y = np.array([100.0, 200.0, 400.0])
    assert gamma_deviance(y, y) == pytest.approx(0.0)
    assert gamma_deviance(np.array([2.0]), np.array([1.0])) == pytest.approx(2 * (1 - np.log(2)))
    for bad_mu in (np.array([0.0, 1, 1]), np.array([np.nan, 1, 1])):
        with pytest.raises(ValueError):
            gamma_deviance(y, bad_mu)
    with pytest.raises(ValueError):
        gamma_deviance(np.array([0.0, 1.0, 1.0]), np.ones(3))
    rng = np.random.default_rng(0)
    yy = rng.gamma(2.0, 500, 1000)
    cal = severity_calibration(yy, np.full(1000, yy.mean()) * rng.uniform(0.5, 1.5, 1000))
    assert len(cal) == 5 and cal["claims"].sum() == 1000
    assert large_claim_analysis(yy, np.full(1000, yy.mean()))["share_of_total_cost"] > 0.01


def test_bootstrap_diff_detects_real_gap_and_not_noise():
    from src.evaluation.metrics import gamma_deviance
    from src.evaluation.uncertainty import bootstrap_metric_diff
    rng = np.random.default_rng(0)
    mu = rng.uniform(500, 3000, 2000)
    y = rng.gamma(2.0, mu / 2.0)
    good = bootstrap_metric_diff(y, mu, np.full(2000, y.mean()), gamma_deviance, n_boot=200)
    assert good["diff"] < 0 and good["significant"]
    same = bootstrap_metric_diff(y, mu, mu * 1.0001, gamma_deviance, n_boot=200)
    assert not same["significant"] or abs(same["diff"]) < 1e-3
    with pytest.raises(ValueError):
        bootstrap_metric_diff(y, mu[:10], mu, gamma_deviance)


def test_bootstrap_ratio_ci_covers_truth_is_reproducible_and_shrinks_with_n():
    from src.evaluation.uncertainty import bootstrap_ratio_ci
    rng = np.random.default_rng(3)

    def sample(n):
        e = rng.uniform(0.5, 2.0, n)
        return rng.poisson(e * 0.1).astype(float), e * 0.1          # well calibrated: true A/E = 1

    a, e = sample(4000)
    r = bootstrap_ratio_ci(a, e, n_boot=300, seed=1)
    assert r["ci_low"] <= r["ratio"] <= r["ci_high"] and r["contains_one"]
    assert r == bootstrap_ratio_ci(a, e, n_boot=300, seed=1)                      # reproducible
    small = bootstrap_ratio_ci(*sample(300), n_boot=300, seed=1)
    big = bootstrap_ratio_ci(*sample(20000), n_boot=300, seed=1)
    assert (big["ci_high"] - big["ci_low"]) < (small["ci_high"] - small["ci_low"])  # more data, narrower
    off = bootstrap_ratio_ci(a, e * 1.5, n_boot=300, seed=1)                      # clearly miscalibrated
    assert not off["contains_one"] and off["ci_high"] < 1.0


@pytest.mark.parametrize("a,e", [([], []), ([1.0], [1.0, 2.0]), ([1.0], [0.0]), ([np.nan], [1.0])])
def test_bootstrap_ratio_ci_rejects_invalid_input(a, e):
    from src.evaluation.uncertainty import bootstrap_ratio_ci
    with pytest.raises(ValueError):
        bootstrap_ratio_ci(np.array(a), np.array(e))
