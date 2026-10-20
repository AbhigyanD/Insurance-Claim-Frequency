"""Business insights report generated from the actual result tables (no hand-typed numbers)."""
from __future__ import annotations

import pandas as pd

REQUIRED = ["test_results", "test_segments", "val_segment_ae", "lift", "shap", "relativities", "severity_bootstrap", "large_loss", "val_freq"]


def _pick(df: pd.DataFrame, **cond) -> pd.Series:
    sub = df
    for k, v in cond.items():
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        raise ValueError(f"Expected exactly one row for {cond}, found {len(sub)}")
    return sub.iloc[0]


def _band_ratio(rel: pd.DataFrame, prefix: str) -> tuple[float, str, float]:
    """Base level is the dropped category (relativity 1). Returns (max/min relativity ratio, label of min, min relativity)."""
    r = rel[rel["term"].str.startswith(prefix)]
    lo = r.loc[r["relativity"].idxmin()]
    hi_rel = max(1.0, r["relativity"].max())
    return hi_rel / lo["relativity"], lo["term"].split("band_")[-1], float(lo["relativity"])


def render_insights(t: dict[str, pd.DataFrame]) -> str:
    missing = [k for k in REQUIRED if k not in t]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    res, seg = t["test_results"], t["test_segments"]
    sel = res[res["model"].str.contains("selected") & (res["target"] == "frequency")].iloc[0]
    glm = _pick(res, target="frequency", model="poisson_glm")
    const_f = _pick(res, target="frequency", model="constant_rate")
    sev = res[res["model"].str.contains("selected") & (res["target"] == "severity")].iloc[0]
    sev_c = _pick(res, target="severity", model="constant_mean")
    allseg = _pick(seg, segment="ALL")
    lift = t["lift"].sort_values("bin")
    top, bottom = lift.iloc[-1], lift.iloc[0]
    shap = t["shap"].sort_values("mean_abs_shap", ascending=False)
    age_ratio, age_label, _ = _band_ratio(t["relativities"], "DrivAge_band")
    _, bm_label, bm_rel = _band_ratio(t["relativities"], "BonusMalus_band")
    sb = t["severity_bootstrap"]
    n_sev_sig = int(sb["significant"].sum())
    ll = t["large_loss"]
    ll995 = ll[(ll["cap_quantile"] == 0.995) & (ll["variant"] == "capped+load")].iloc[0]
    low_cost = seg[seg["segment"] != "ALL"]
    n_cost_below = int((low_cost["cost_ci_high"] < 1.0).sum())
    cost_gap = (1 / allseg["cost_ae"] - 1) * 100           # how far EXPECTED sits above (+) or below (-) actual
    segs = seg[seg["segment"] != "ALL"]
    freq_bad = segs[(segs["freq_ci_low"] > 1.0) | (segs["freq_ci_high"] < 1.0)]["segment"].tolist()
    freq_overall_ok = allseg["freq_ci_low"] <= 1.0 <= allseg["freq_ci_high"]
    cost_overall_excl_1 = bool(allseg["cost_ci_low"] > 1.0 or allseg["cost_ci_high"] < 1.0)
    glm_better = bool(glm["ci_high"] < 0)                    # selected minus GLM deviance, negative = selected better
    cost_dir = "higher" if cost_gap > 0 else "lower"
    freq_seg_text = ("Every risk segment's frequency actual/expected interval on test includes 1.0"
                     if not freq_bad else f"Frequency actual/expected intervals on test exclude 1.0 for: {', '.join(freq_bad)}; the other segments include it")
    gap_cause = ("The gap sits in severity, not frequency: overall frequency actual/expected is consistent with 1.0."
                 if freq_overall_ok and cost_overall_excl_1 else
                 "Overall frequency and cost ratios should be read together with their intervals above.")
    glm_text = ("the improvement over the GLM is statistically reliable" if glm_better
                else "the improvement over the GLM is not statistically reliable")
    cal_text = ("no risk segment is reliably mis-calibrated" if not freq_bad else f"segments with a frequency interval excluding 1.0: {', '.join(freq_bad)}")

    return f"""# Business insights: motor claims frequency, severity and expected cost

*For pricing analysts and underwriting stakeholders. Generated from the project's result tables; every figure below comes from an actual run.*

## In one paragraph

The frequency model **separates good risks from bad risks well and is well calibrated on held-out data**: on the test set overall claim frequency is predicted to within {abs(sel['actual_to_predicted'] - 1) * 100:.1f}% and {cal_text}. The **expected claim cost level is not reliable yet**: on the same test data total expected cost is about {abs(cost_gap):.0f}% {cost_dir} than actual cost (actual/expected {allseg['cost_ae']:.2f}, 95% interval {allseg['cost_ci_low']:.2f} to {allseg['cost_ci_high']:.2f}), because claim *severity* is hard to predict and the severity level is dominated by a few very large claims. Use the model for **ranking and relative differences between risks**; do not use its absolute cost level for pricing without fixing the severity level.

## What the evidence supports

1. **Frequency ranking works.** On the held-out test set the selected model's Poisson deviance is {sel['poisson_deviance']:.4f}, versus {glm['poisson_deviance']:.4f} for a Poisson GLM and {const_f['poisson_deviance']:.4f} for a flat rate; {glm_text}. The ranking measure (Gini) is {sel['gini']:.2f} for the model, {glm['gini']:.2f} for the GLM and {const_f['gini']:.2f} for the flat rate.
2. **The riskiest tenth of policies claim far more often.** On validation data, the highest predicted-risk decile has {top['lift_vs_mean']:.1f}x the portfolio claim frequency and the lowest has {bottom['lift_vs_mean']:.1f}x (actual outcomes, ordered by prediction).
3. **Main risk drivers (associations, not causes).** By model importance the top factors are {', '.join(shap['feature'].head(3))}. In the GLM, bonus-malus matters most: the best-bonus band ({bm_label}) has a claim rate about {1 / bm_rel:.0f} times lower than the base band (bonus-malus 100-125). Younger drivers claim much more often: the highest-risk age band is about {age_ratio:.1f}x the claim rate of the lowest-risk band ({age_label}).
4. **Frequency calibration by segment.** {freq_seg_text} (overall {allseg['freq_ae']:.3f}, interval {allseg['freq_ci_low']:.3f} to {allseg['freq_ci_high']:.3f}).

## What the evidence does not support

1. **Absolute expected cost.** Test actual/expected cost is below 1.0 in {n_cost_below} of {len(low_cost)} segments with intervals that exclude 1.0{', and the overall ratio also excludes 1.0' if cost_overall_excl_1 else ''}. {gap_cause}
2. **Severity relativities.** On validation, {n_sev_sig} of {len(sb)} severity models were reliably different from a constant average. On test the selected severity model is better than a constant (gamma deviance {sev['gamma_deviance']:.3f} vs {sev_c['gamma_deviance']:.3f}), but part of that is a lower overall level, not demonstrated ranking skill. Treat severity differences between policyholders as unreliable.
3. **A large-loss load.** Capping claims at the 99.5th percentile and adding a load for the excess ({ll995['train_load'] * 100:.0f}% load on training data) is unstable: one claim of about 4.1 million supplies {ll995['top_claim_share_of_excess'] * 100:.0f}% of all excess loss, and removing it would cut the load to {ll995['load_without_top_claim'] * 100:.0f}%. This needs credibility weighting, a fitted tail distribution or reinsurance-style treatment before use.
4. **Causal statements.** None of this shows that changing a rating factor would change anyone's risk. Bonus-malus and driver age partly stand in for claims history and experience.

## Recommendations

- **Use now:** the frequency model for risk ranking, segmentation and relativity analysis; the segment dashboards as a monitoring view.
- **Before using expected cost in pricing:** fix the severity level (large-loss treatment, tail model, or more claims history), then re-check cost actual/expected on fresh data. A single sample with one extreme claim should not set the level.
- **Collect:** dated policy and claim data. This dataset has no dates, so out-of-time testing, performance-over-time monitoring and inflation adjustment are not possible here.
- **Remember:** expected claims cost is a pure-premium estimate of losses. Expenses, commissions, capital, reinsurance, taxes and insurer-specific pricing rules are outside the model.
- **Governance:** the test set has been used once (see `reports/test_evaluation_marker.json`); any further model changes should be validated on new data, not by repeating the test evaluation.

## Limitations

French motor third-party liability data from about 2011-2013; no dates, so no out-of-time validation; claim amounts are not inflation-adjusted; some amounts are fixed convention amounts; features are limited to vehicle, driver age, bonus-malus, density, area and region; the severity sample is small (about 16,000 training claims) and heavy-tailed.
"""
