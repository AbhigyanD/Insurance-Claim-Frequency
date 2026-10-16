-- DuckDB analytical schema. ${PROCESSED} and ${REPORTS} are substituted by scripts/build_database.py.

CREATE OR REPLACE TABLE policies AS
SELECT 'train' AS split, * FROM read_parquet('${PROCESSED}/policies_train.parquet')
UNION ALL BY NAME
SELECT 'val'   AS split, * FROM read_parquet('${PROCESSED}/policies_val.parquet')
UNION ALL BY NAME
SELECT 'test'  AS split, * FROM read_parquet('${PROCESSED}/policies_test.parquet');

CREATE OR REPLACE TABLE predictions AS
SELECT * FROM read_parquet('${PROCESSED}/predictions.parquet');

CREATE OR REPLACE TABLE model_eval_frequency AS
SELECT * FROM read_csv_auto('${REPORTS}/frequency_model_comparison.csv');

CREATE OR REPLACE TABLE model_eval_severity AS
SELECT * FROM read_csv_auto('${REPORTS}/severity_model_comparison.csv');

-- Policy-level features and targets (outcomes in clearly named columns).
CREATE OR REPLACE VIEW v_policy_features_targets AS
SELECT IDpol, split, Exposure,
       VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Region, Density,
       ClaimNb          AS target_claim_count,
       TotalClaimAmount AS target_claim_amount
FROM policies;

-- Long-format model evaluation (one row per model / split / metric).
CREATE OR REPLACE VIEW v_model_evaluation AS
SELECT 'frequency' AS target, model, split, metric, value
FROM (UNPIVOT (SELECT model, split, poisson_deviance, mae_counts, actual_to_predicted, gini FROM model_eval_frequency)
      ON poisson_deviance, mae_counts, actual_to_predicted, gini INTO NAME metric VALUE value)
UNION ALL
SELECT 'severity' AS target, model, split, metric, value
FROM (UNPIVOT (SELECT model, split, gamma_deviance, mae, rmse, actual_to_predicted FROM model_eval_severity)
      ON gamma_deviance, mae, rmse, actual_to_predicted INTO NAME metric VALUE value);

-- Actual vs expected loss by predicted risk segment.
CREATE OR REPLACE VIEW v_segment_performance AS
SELECT split, risk_segment,
       COUNT(*)                                         AS policies,
       SUM(exposure)                                    AS exposure,
       SUM(actual_claim_count)                          AS actual_claims,
       SUM(predicted_frequency * exposure)              AS expected_claims,
       SUM(actual_claim_amount)                         AS actual_cost,
       SUM(predicted_cost_exposure)                     AS expected_cost,
       SUM(actual_claim_count) / SUM(predicted_frequency * exposure) AS frequency_actual_to_expected,
       SUM(actual_claim_amount) / SUM(predicted_cost_exposure)       AS cost_actual_to_expected,
       SUM(actual_claim_amount) / SUM(exposure)         AS actual_pure_premium,
       SUM(predicted_cost_exposure) / SUM(exposure)     AS expected_pure_premium
FROM predictions
GROUP BY split, risk_segment;
