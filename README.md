# Insurance-Claim-Frequency
Build a model that predicts claim frequency and claim severity, then combines them to estimate expected annual claims cost.  Models: GLM with Poisson or Tweedie distributions, LightGBM/CatBoost, XGBoost.  Evaluation: MAE, deviance, calibration, lift by risk segment, and temporal holdout performance.  

## Status (work in progress)

Implemented so far: config, data download/loading from CASdatasets (`freMTPL2freq`/`freMTPL2sev`),
exposure validation, policy-level table, leakage-safe policy-level train/val/test split,
preprocessing pipeline, EDA script, and unit tests (`pytest`).

Frequency models (actual run, `python -m scripts.train_frequency`; fit on train, scored on the validation split only, test untouched):

| Model | Split | Poisson deviance | MAE (counts) | Actual/Predicted | Gini |
|---|---|---|---|---|---|
| Constant rate | validation | 0.2524 | 0.0741 | 0.994 | -0.003 |
| Poisson GLM (log-exposure offset) | validation | 0.2395 | 0.0725 | 0.995 | 0.303 |
| Negative Binomial GLM (NB2, offset) | validation | 0.2396 | 0.0727 | 0.990 | 0.303 |
| LightGBM Poisson (log-exposure init score) | validation | 0.2338 | 0.0717 | 0.997 | 0.339 |

- LightGBM is best on validation (lower deviance, Gini 0.339 vs 0.303). Its train Gini is 0.378, so there is a train/validation gap (some overfitting); early stopping on validation chose 534 trees.
- Overdispersion: Poisson GLM Pearson dispersion is 1.89 on train and the estimated NB2 alpha is 1.05. Mean predictions are almost identical to the Poisson GLM, so NB gives no ranking or deviance gain here. Its validation log-likelihood is slightly better (-0.15697 vs -0.15717), which supports overdispersion mainly affecting uncertainty/standard errors rather than point predictions.
- The Poisson GLM lift table is in `reports/frequency_lift_validation_poisson_glm.csv`; LightGBM's is alongside it. These are validation results used for model selection, not final test results.

Severity models (actual run, `python -m scripts.train_severity`; trained on claims only: 15,872 train / 5,266 validation claims; test untouched):

| Model | Split | Gamma deviance | MAE | Actual mean | Predicted mean |
|---|---|---|---|---|---|
| Constant mean | validation | 1.669 | 2,281 | 2,143 | 2,438 |
| Gamma GLM (log link) | validation | 1.726 | 2,240 | 2,143 | 2,314 |
| Gamma GLM, claims capped at train p99.5 | validation | 1.698 | 1,746 | 2,143 | 1,691 |
| Ridge Gamma GLM (penalty chosen on validation) | validation | 1.664 | 2,187 | 2,143 | 2,310 |
| LightGBM (gamma objective) | validation | 1.656 | 2,131 | 2,143 | 2,241 |

Honest reading: severity is only weakly predictable from the available features.
- The Gamma GLM is *worse* than the constant mean on validation deviance (train 1.54 vs validation 1.73), i.e. it overfits with 16k claims.
  Its predicted-severity ranking is also not monotone (top predicted quintile has actual/predicted 0.57), so its relativities are not trustworthy as they stand.
- LightGBM beats the constant mean by only ~0.8% in deviance; I do not treat that as a real improvement without an uncertainty estimate.
- Capping lowers MAE because it predicts a lower mean, but it under-predicts average severity (actual/predicted 1.27), so it would need a large-loss load.
- RMSE is dominated by a few extreme claims (train RMSE ~37k vs validation ~11.6k because of a single claim of about 4.1M in train).
- Top-1% validation claims (about a third of total cost) are under-predicted by roughly 24-41x by every model, so a large-loss treatment is needed before pricing use.
Uncertainty (`reports/severity_bootstrap_vs_constant.csv`, 1,000 bootstrap resamples of validation claims; deviance minus the constant-mean deviance, negative = better):

| Model | Difference | 95% interval | Reliably different? |
|---|---|---|---|
| Gamma GLM | +0.057 | [-0.073, +0.182] | No |
| Gamma GLM, capped p99.5 | +0.030 | [-0.082, +0.148] | No |
| Ridge Gamma GLM | -0.005 | [-0.031, +0.026] | No |
| LightGBM | -0.012 | [-0.038, +0.017] | No |

No severity model is distinguishable from a constant mean on validation. The ridge search also chose the largest penalty on its grid (alpha=1.0, a boundary), i.e. the data pull the model toward a constant.
Practical implication: with these features, differences in expected claims cost across policies will be driven almost entirely by the frequency model; any severity relativities should be treated as unreliable.
Next step for severity: a large-loss/capped component and possibly a coarser, domain-driven set of severity factors.

## Expected claims cost (validation)

Selected models (fixed in `configs/config.yaml` from validation results, before any test-set evaluation): LightGBM Poisson for frequency; ridge Gamma GLM for severity (no severity model reliably beat a constant mean, so the most regularised interpretable one is used).
`python -m scripts.generate_predictions` writes a policy-level table (`data/processed/predictions.parquet`, not committed) with predicted frequency (per exposure-year), predicted severity, expected annual cost (= frequency x severity for a full year), expected cost over the observed exposure, a risk segment (quintiles of predicted annual cost, cut points from train only), and the actual outcomes in separate columns. Outputs that are non-finite or negative raise an error rather than being clipped. Expected claims cost is a pure-premium estimate of losses, not a customer price: expenses, commissions, capital, reinsurance and taxes are not modelled.

`python -m scripts.evaluate_models` (validation only), `reports/segment_performance_validation.csv`:

| Segment | Policies | Frequency actual/expected | Cost actual/expected | Actual pure premium | Expected pure premium |
|---|---|---|---|---|---|
| Very low | 27,087 | 1.012 | 1.210 | 101 | 84 |
| Low | 27,200 | 0.936 | 0.640 | 71 | 111 |
| Medium | 27,165 | 0.969 | 0.966 | 130 | 135 |
| High | 27,095 | 0.967 | 0.929 | 167 | 180 |
| Very high | 27,051 | 1.044 | 0.921 | 367 | 398 |

- Overall validation actual/expected cost is 0.925: expected cost is about 8% too high, because the train mean claim (2,438, inflated by a single ~4.1M claim) is above the validation mean (2,143). This is a severity effect, not a frequency one.
- Frequency is well calibrated (0.94-1.04) in every segment and in the driver-age, fuel and area groups checked (`reports/stability_by_feature_validation.csv`). Cost actual/expected varies much more by group (about 0.6-1.3) because it rests on few, heavy-tailed claims.
- The "Very high" segment has about 3.7x the actual pure premium of the "Very low"/"Low" segments, so the model separates risk, but ordering at the low end is not monotone (Low 71 < Very low 101) and should not be over-interpreted.
- Test-set results have not been produced yet.

## SQL analytics (DuckDB)

`python -m scripts.build_database` builds `data/processed/claims.duckdb` (git-ignored) from the processed parquet files and report CSVs using `sql/schema.sql`: tables `policies`, `predictions`, `model_eval_frequency`, `model_eval_severity` and views `v_policy_features_targets`, `v_segment_performance` (actual vs expected loss by predicted risk segment) and `v_model_evaluation` (long-format metrics).
`python -m scripts.build_database --query sql/analytics.sql` runs the example queries in `sql/analytics.sql` (actual vs expected cost, segment performance, expected annual cost by area and region with sample sizes, driver-age frequency, model metrics). The views are covered by `tests/test_sql.py`.

Not yet implemented: explainability (SHAP), MLflow tracking, dashboard exports, notebooks.
All model results above are validation results used for model selection; no final test-set results have been produced yet.

Data checks and EDA findings (from actual runs, train split only):
- 677,991 policies, no missing values or duplicate ids; 1,224 have exposure > 1 (capped at 1).
- Claim counts in the frequency file match the severity rows exactly, so the two files join cleanly.
- 96.3% of train policies have zero claims; overall frequency 0.0738 per exposure-year.
- Raw var/mean of claim counts is 1.11 (ignores exposure and covariates); overdispersion is still to be tested on fitted models.
- Severity is heavy-tailed: median 1,172, p99.9 156,057, max 4,075,401; the top 0.1% of claims make up 25.9% of total claim amount.
- Train/val/test frequencies are close (0.0738 / 0.0734 / 0.0741). See `reports/eda_summary.txt` and `reports/figures/`.

Run: `pip install -e .[dev]`, `python -m scripts.prepare_data`, `python -m scripts.run_eda`, `python -m scripts.train_frequency`, `python -m scripts.train_severity`, `python -m scripts.generate_predictions`, `python -m scripts.evaluate_models`, `pytest`.
See `data/README.md` for dataset source, license and limitations.
