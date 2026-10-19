import numpy as np
import pytest

from src.models.large_loss import cap_threshold, excess_load, load_sensitivity


def test_excess_load_known_value():
    a = np.array([100.0, 100.0, 100.0, 1000.0])
    # cap 200: capped = 100+100+100+200 = 500, excess = 800  ->  load 1.6
    assert excess_load(a, 200.0) == pytest.approx(1.6)


def test_load_is_zero_above_max_and_decreases_with_cap():
    a = np.random.default_rng(0).gamma(1.5, 1500, 5000)
    assert excess_load(a, a.max() * 2) == 0.0
    loads = [excess_load(a, cap_threshold(a, q)) for q in (0.9, 0.99, 0.999)]
    assert loads[0] > loads[1] > loads[2] >= 0


def test_capped_times_load_restores_total_cost():
    a = np.random.default_rng(1).lognormal(7, 1.2, 4000)
    cap = cap_threshold(a, 0.995)
    restored = np.minimum(a, cap).sum() * (1 + excess_load(a, cap))
    assert restored == pytest.approx(a.sum())          # exact identity on the data used to set the load


def test_sensitivity_exposes_dependence_on_one_extreme_claim():
    a = np.concatenate([np.full(999, 1000.0), [4_000_000.0]])
    s = load_sensitivity(a, cap=2000.0, drop_top=1)
    assert s["top_claims_share_of_excess"] == pytest.approx(1.0)      # all the excess is the single claim
    assert s["load_without_top"] == 0.0 and s["load"] > 1.0


@pytest.mark.parametrize("amounts,q", [([], 0.9), ([1.0, -1.0], 0.9), ([1.0, np.nan], 0.9), ([1.0, 2.0], 1.5)])
def test_invalid_inputs_raise(amounts, q):
    with pytest.raises(ValueError):
        cap_threshold(np.array(amounts), q)


def test_non_positive_cap_rejected():
    with pytest.raises(ValueError):
        excess_load(np.array([1.0, 2.0]), 0.0)
