# Business insights: motor claims frequency, severity and expected cost

*For pricing analysts and underwriting stakeholders. Generated from the project's result tables; every figure below comes from an actual run.*

## In one paragraph

The frequency model **separates good risks from bad risks well and is well calibrated on held-out data**: on the test set overall claim frequency is predicted to within 0.4% and no risk segment is reliably mis-calibrated. The **expected claim cost level is not reliable yet**: on the same test data total expected cost is about 23% higher than actual cost (actual/expected 0.81, 95% interval 0.73 to 0.90), because claim *severity* is hard to predict and the severity level is dominated by a few very large claims. Use the model for **ranking and relative differences between risks**; do not use its absolute cost level for pricing without fixing the severity level.

## What the evidence supports

1. **Frequency ranking works.** On the held-out test set the selected model's Poisson deviance is 0.2374, versus 0.2412 for a Poisson GLM and 0.2529 for a flat rate; the improvement over the GLM is statistically reliable. The ranking measure (Gini) is 0.32 for the model, 0.29 for the GLM and 0.00 for the flat rate.
2. **The riskiest tenth of policies claim far more often.** On validation data, the highest predicted-risk decile has 3.0x the portfolio claim frequency and the lowest has 0.4x (actual outcomes, ordered by prediction).
3. **Main risk drivers (associations, not causes).** By model importance the top factors are BonusMalus, DrivAge, Density. In the GLM, bonus-malus matters most: the best-bonus band ((49, 50]) has a claim rate about 7 times lower than the base band (bonus-malus 100-125). Younger drivers claim much more often: the highest-risk age band is about 2.1x the claim rate of the lowest-risk band ((25, 30]).
4. **Frequency calibration by segment.** Every risk segment's frequency actual/expected interval on test includes 1.0 (overall 1.004, interval 0.977 to 1.033).

## What the evidence does not support

1. **Absolute expected cost.** Test actual/expected cost is below 1.0 in 3 of 5 segments with intervals that exclude 1.0, and the overall ratio also excludes 1.0. The gap sits in severity, not frequency: overall frequency actual/expected is consistent with 1.0.
2. **Severity relativities.** On validation, 0 of 4 severity models were reliably different from a constant average. On test the selected severity model is better than a constant (gamma deviance 1.435 vs 1.468), but part of that is a lower overall level, not demonstrated ranking skill. Treat severity differences between policyholders as unreliable.
3. **A large-loss load.** Capping claims at the 99.5th percentile and adding a load for the excess (44% load on training data) is unstable: one claim of about 4.1 million supplies 34% of all excess loss, and removing it would cut the load to 29%. This needs credibility weighting, a fitted tail distribution or reinsurance-style treatment before use.
4. **Causal statements.** None of this shows that changing a rating factor would change anyone's risk. Bonus-malus and driver age partly stand in for claims history and experience.

## Recommendations

- **Use now:** the frequency model for risk ranking, segmentation and relativity analysis; the segment dashboards as a monitoring view.
- **Before using expected cost in pricing:** fix the severity level (large-loss treatment, tail model, or more claims history), then re-check cost actual/expected on fresh data. A single sample with one extreme claim should not set the level.
- **Collect:** dated policy and claim data. This dataset has no dates, so out-of-time testing, performance-over-time monitoring and inflation adjustment are not possible here.
- **Remember:** expected claims cost is a pure-premium estimate of losses. Expenses, commissions, capital, reinsurance, taxes and insurer-specific pricing rules are outside the model.
- **Governance:** the test set has been used once (see `reports/test_evaluation_marker.json`); any further model changes should be validated on new data, not by repeating the test evaluation.

## Limitations

French motor third-party liability data from about 2011-2013; no dates, so no out-of-time validation; claim amounts are not inflation-adjusted; some amounts are fixed convention amounts; features are limited to vehicle, driver age, bonus-malus, density, area and region; the severity sample is small (about 16,000 training claims) and heavy-tailed.
