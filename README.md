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

Not yet implemented: severity models, pure-premium predictions, explainability,
SQL analytics, MLflow tracking, dashboard exports. No model results exist yet.

Data checks and EDA findings (from actual runs, train split only):
- 677,991 policies, no missing values or duplicate ids; 1,224 have exposure > 1 (capped at 1).
- Claim counts in the frequency file match the severity rows exactly, so the two files join cleanly.
- 96.3% of train policies have zero claims; overall frequency 0.0738 per exposure-year.
- Raw var/mean of claim counts is 1.11 (ignores exposure and covariates); overdispersion is still to be tested on fitted models.
- Severity is heavy-tailed: median 1,172, p99.9 156,057, max 4,075,401; the top 0.1% of claims make up 25.9% of total claim amount.
- Train/val/test frequencies are close (0.0738 / 0.0734 / 0.0741). See `reports/eda_summary.txt` and `reports/figures/`.

Run: `pip install -e .[dev]`, `python -m scripts.prepare_data`, `python -m scripts.run_eda`, `python -m scripts.train_frequency`, `pytest`.
See `data/README.md` for dataset source, license and limitations.
