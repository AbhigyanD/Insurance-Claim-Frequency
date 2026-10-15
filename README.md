# Insurance-Claim-Frequency
Build a model that predicts claim frequency and claim severity, then combines them to estimate expected annual claims cost.  Models: GLM with Poisson or Tweedie distributions, LightGBM/CatBoost, XGBoost.  Evaluation: MAE, deviance, calibration, lift by risk segment, and temporal holdout performance.  

## Status (work in progress)

Implemented so far: config, data download/loading from CASdatasets (`freMTPL2freq`/`freMTPL2sev`),
exposure validation, policy-level table, leakage-safe policy-level train/val/test split,
preprocessing pipeline, EDA script, and unit tests (`pytest`).

Not yet implemented: frequency/severity models, pure-premium predictions, explainability,
SQL analytics, MLflow tracking, dashboard exports. No model results exist yet.

Data checks and EDA findings (from actual runs, train split only):
- 677,991 policies, no missing values or duplicate ids; 1,224 have exposure > 1 (capped at 1).
- Claim counts in the frequency file match the severity rows exactly, so the two files join cleanly.
- 96.3% of train policies have zero claims; overall frequency 0.0738 per exposure-year.
- Raw var/mean of claim counts is 1.11 (ignores exposure and covariates); overdispersion is still to be tested on fitted models.
- Severity is heavy-tailed: median 1,172, p99.9 156,057, max 4,075,401; the top 0.1% of claims make up 25.9% of total claim amount.
- Train/val/test frequencies are close (0.0738 / 0.0734 / 0.0741). See `reports/eda_summary.txt` and `reports/figures/`.

Run: `pip install -e .[dev]`, `python -m scripts.prepare_data`, `python -m scripts.run_eda`, `pytest`.
See `data/README.md` for dataset source, license and limitations.
