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
| LightGBM (gamma objective) | validation | 1.656 | 2,131 | 2,143 | 2,241 |

Honest reading: severity is only weakly predictable from the available features.
- The Gamma GLM is *worse* than the constant mean on validation deviance (train 1.54 vs validation 1.73), i.e. it overfits with 16k claims.
  Its predicted-severity ranking is also not monotone (top predicted quintile has actual/predicted 0.57), so its relativities are not trustworthy as they stand.
- LightGBM beats the constant mean by only ~0.8% in deviance; I do not treat that as a real improvement without an uncertainty estimate.
- Capping lowers MAE because it predicts a lower mean, but it under-predicts average severity (actual/predicted 1.27), so it would need a large-loss load.
- RMSE is dominated by a few extreme claims (train RMSE ~37k vs validation ~11.6k because of a single claim of about 4.1M in train).
- Top-1% validation claims (about a third of total cost) are under-predicted by roughly 24-41x by every model, so a large-loss treatment is needed before pricing use.
Next steps for severity: a smaller / regularised GLM, bootstrap intervals on deviance differences, and a large-loss/capped component.

Not yet implemented: expected-cost (pure premium) predictions, explainability (SHAP), SQL analytics, MLflow tracking, dashboard exports, notebooks.
All model results above are validation results used for model selection; no final test-set results have been produced yet.

Data checks and EDA findings (from actual runs, train split only):
- 677,991 policies, no missing values or duplicate ids; 1,224 have exposure > 1 (capped at 1).
- Claim counts in the frequency file match the severity rows exactly, so the two files join cleanly.
- 96.3% of train policies have zero claims; overall frequency 0.0738 per exposure-year.
- Raw var/mean of claim counts is 1.11 (ignores exposure and covariates); overdispersion is still to be tested on fitted models.
- Severity is heavy-tailed: median 1,172, p99.9 156,057, max 4,075,401; the top 0.1% of claims make up 25.9% of total claim amount.
- Train/val/test frequencies are close (0.0738 / 0.0734 / 0.0741). See `reports/eda_summary.txt` and `reports/figures/`.

Run: `pip install -e .[dev]`, `python -m scripts.prepare_data`, `python -m scripts.run_eda`, `python -m scripts.train_frequency`, `python -m scripts.train_severity`, `pytest`.
See `data/README.md` for dataset source, license and limitations.
