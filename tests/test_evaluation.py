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
