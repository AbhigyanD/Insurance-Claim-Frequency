-- Example analytical queries. Run with: python -m scripts.build_database --query sql/analytics.sql
-- Validation split only; test-set figures are reported separately after model selection.

-- 1. Aggregate actual vs expected claims cost by split (train and validation only; the test split is
--    deliberately excluded here and is reported once, in the final evaluation).
SELECT split, COUNT(*) AS policies, ROUND(SUM(actual_claim_amount)) AS actual_cost,
       ROUND(SUM(predicted_cost_exposure)) AS expected_cost,
       ROUND(SUM(actual_claim_amount) / SUM(predicted_cost_exposure), 3) AS actual_to_expected
FROM predictions
WHERE split IN ('train', 'val')
GROUP BY split
ORDER BY split;

-- 2. Risk-segment performance (validation).
SELECT risk_segment, policies, ROUND(frequency_actual_to_expected, 3) AS freq_a_to_e,
       ROUND(cost_actual_to_expected, 3) AS cost_a_to_e,
       ROUND(actual_pure_premium, 1) AS actual_pp, ROUND(expected_pure_premium, 1) AS expected_pp
FROM v_segment_performance
WHERE split = 'val'
ORDER BY expected_pure_premium;

-- 3. Expected annual cost by area, with sample sizes (validation).
SELECT Area, COUNT(*) AS policies, ROUND(AVG(predicted_annual_cost), 1) AS avg_expected_annual_cost,
       ROUND(SUM(actual_claim_amount) / SUM(exposure), 1) AS actual_pure_premium
FROM predictions
WHERE split = 'val'
GROUP BY Area
ORDER BY avg_expected_annual_cost DESC;

-- 4. Highest-cost regions, restricted to regions with enough exposure to be meaningful.
SELECT Region, COUNT(*) AS policies, ROUND(SUM(exposure)) AS exposure,
       ROUND(AVG(predicted_annual_cost), 1) AS avg_expected_annual_cost
FROM predictions
WHERE split = 'val'
GROUP BY Region
HAVING SUM(exposure) >= 1000
ORDER BY avg_expected_annual_cost DESC
LIMIT 5;

-- 5. Driver-age bands: predicted vs actual frequency.
SELECT CASE WHEN DrivAge < 25 THEN '18-24' WHEN DrivAge < 35 THEN '25-34' WHEN DrivAge < 50 THEN '35-49'
            WHEN DrivAge < 65 THEN '50-64' ELSE '65+' END AS driver_age_band,
       COUNT(*) AS policies,
       ROUND(SUM(actual_claim_count) / SUM(exposure), 4) AS actual_frequency,
       ROUND(SUM(predicted_frequency * exposure) / SUM(exposure), 4) AS predicted_frequency
FROM predictions
WHERE split = 'val'
GROUP BY driver_age_band
ORDER BY driver_age_band;

-- 6. Model evaluation results, validation split.
SELECT target, model, metric, ROUND(value, 4) AS value
FROM v_model_evaluation
WHERE split = 'validation' AND metric IN ('poisson_deviance', 'gini', 'gamma_deviance')
ORDER BY target, metric, value;
