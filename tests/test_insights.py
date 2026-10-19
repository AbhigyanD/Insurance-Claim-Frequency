import pandas as pd
import pytest

from src.reporting.insights import REQUIRED, render_insights


def _tables(cost_ae=0.8, cost_hi=0.9):
    return {
        "test_results": pd.DataFrame([
            {"target": "frequency", "model": "constant_rate", "poisson_deviance": 0.2529, "gini": 0.002, "actual_to_predicted": 1.003, "gamma_deviance": None},
            {"target": "frequency", "model": "poisson_glm", "poisson_deviance": 0.2412, "gini": 0.288, "actual_to_predicted": 1.003, "gamma_deviance": None,
             "selected_dev_diff_vs_this": -0.004, "ci_low": -0.005, "ci_high": -0.003},
            {"target": "frequency", "model": "lgbm (selected)", "poisson_deviance": 0.2374, "gini": 0.321, "actual_to_predicted": 1.004, "gamma_deviance": None},
            {"target": "severity", "model": "ridge (selected)", "poisson_deviance": None, "gini": None, "actual_to_predicted": 0.81, "gamma_deviance": 1.435},
            {"target": "severity", "model": "constant_mean", "poisson_deviance": None, "gini": None, "actual_to_predicted": 0.77, "gamma_deviance": 1.468}]),
        "test_segments": pd.DataFrame([
            {"segment": "ALL", "freq_ae": 1.004, "freq_ci_low": 0.977, "freq_ci_high": 1.033, "cost_ae": cost_ae, "cost_ci_low": 0.73, "cost_ci_high": cost_hi},
            {"segment": "Low", "freq_ae": 1.0, "freq_ci_low": 0.9, "freq_ci_high": 1.1, "cost_ae": 0.8, "cost_ci_low": 0.6, "cost_ci_high": 0.97},
            {"segment": "High", "freq_ae": 1.0, "freq_ci_low": 0.9, "freq_ci_high": 1.1, "cost_ae": 0.9, "cost_ci_low": 0.7, "cost_ci_high": 1.2}]),
        "val_segment_ae": pd.DataFrame({"segment": ["ALL"]}),
        "lift": pd.DataFrame({"bin": [1, 2, 10], "lift_vs_mean": [0.39, 0.54, 2.6]}),
        "shap": pd.DataFrame({"feature": ["BonusMalus", "DrivAge", "Density", "Area"], "mean_abs_shap": [0.36, 0.12, 0.10, 0.01]}),
        "relativities": pd.DataFrame({"term": ["DrivAge_band_(25, 30]", "DrivAge_band_(40, 50]", "BonusMalus_band_(49, 50]", "BonusMalus_band_(125, 150]"],
                                      "relativity": [0.5, 0.85, 0.125, 1.3]}),
        "severity_bootstrap": pd.DataFrame({"model": ["a", "b"], "significant": [False, False]}),
        "large_loss": pd.DataFrame({"cap_quantile": [0.995], "variant": ["capped+load"], "train_load": [0.44],
                                    "top_claim_share_of_excess": [0.34], "load_without_top_claim": [0.29]}),
        "val_freq": pd.DataFrame({"model": ["x"]}),
    }


def test_numbers_in_report_come_from_inputs():
    md = render_insights(_tables())
    # model metrics, formatted exactly as in the input tables
    for expected in ["0.2374", "0.2412", "0.2529", "1.435", "1.468"]:
        assert expected in md, expected
    assert "Gini) is 0.32 for the model, 0.29 for the GLM and 0.00 for the flat rate" in md
    assert "2.6x the portfolio claim frequency and the lowest has 0.4x" in md        # lift of top / bottom bin
    assert "BonusMalus, DrivAge, Density" in md                                      # top-3 SHAP features, in order
    assert "about 8 times lower than the base band" in md                            # 1 / 0.125 for the best bonus band
    assert "about 2.0x the claim rate of the lowest-risk band ((25, 30])" in md      # base level (1.0) / 0.5
    assert "about 25% higher than actual cost" in md                                 # 1 / 0.8 - 1
    assert "actual/expected 0.80, 95% interval 0.73 to 0.90" in md


def test_caveats_are_always_present_and_wording_is_not_causal():
    md = render_insights(_tables())
    for phrase in ["pure-premium estimate of losses", "associations, not causes", "What the evidence does not support",
                   "no dates", "None of this shows that changing a rating factor", "Use the model for **ranking"]:
        assert phrase in md, phrase


def test_cost_segment_counts_reflect_intervals():
    md = render_insights(_tables())
    assert "in 1 of 2 segments" in md                            # only Low has an interval excluding 1.0


def test_conclusions_follow_the_intervals_not_a_fixed_script():
    # frequency interval for one segment excludes 1.0, GLM improvement not reliable, overall cost consistent with 1.0
    t = _tables(cost_ae=0.95, cost_hi=1.05)
    t["test_segments"].loc[t["test_segments"]["segment"] == "Low", ["freq_ci_low", "freq_ci_high"]] = [1.02, 1.2]
    t["test_results"].loc[t["test_results"]["model"] == "poisson_glm", ["selected_dev_diff_vs_this", "ci_low", "ci_high"]] = [-0.001, -0.003, 0.002]
    md = render_insights(t)
    assert "exclude 1.0 for: Low" in md and "Every risk segment's frequency" not in md
    assert "segments with a frequency interval excluding 1.0: Low" in md
    assert "not statistically reliable" in md
    assert "sits in severity, not frequency" not in md                 # cost ratio interval includes 1.0 here
    assert "overall ratio also excludes" not in md


def test_default_case_states_the_reliable_findings():
    t = _tables()
    t["test_results"].loc[t["test_results"]["model"] == "poisson_glm", ["selected_dev_diff_vs_this", "ci_low", "ci_high"]] = [-0.004, -0.005, -0.003]
    md = render_insights(t)
    assert "the improvement over the GLM is statistically reliable" in md
    assert "Every risk segment's frequency actual/expected interval on test includes 1.0" in md
    assert "sits in severity, not frequency" in md and "overall ratio also excludes 1.0" in md


def test_missing_input_fails_loudly():
    t = _tables()
    del t["large_loss"]
    with pytest.raises(ValueError, match="Missing inputs"):
        render_insights(t)
    assert "large_loss" in REQUIRED
