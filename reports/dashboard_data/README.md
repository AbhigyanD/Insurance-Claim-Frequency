# Power BI data and build instructions

Power BI Desktop cannot be automated from this repository, so this folder holds clean, documented CSV tables and the steps to build the report.
All tables use the **validation split** only (no test-split rows) and carry a `data_kind` column that separates historical outcomes,
model predictions and side-by-side comparisons. No simulated data is included.

## Tables

| File | Grain | Use |
|---|---|---|
| `category_summary.csv` | one row per dimension level (risk segment, driver age, vehicle age, area, region, brand, fuel) | executive and risk-analysis pages |
| `model_metrics.csv` | one row per target / model / metric | model performance page |
| `predicted_cost_distribution.csv` | one row per histogram bin | distribution of predicted risk |
| `drift_psi.csv` | one row per feature or prediction | monitoring page |
| `policy_predictions_validation.csv` | one row per policy (135,598) | drill-through; **not committed** (21 MB), regenerate with the command below |
| `data_dictionary.csv` | one row per table column | field definitions and which columns are outcomes vs predictions |

Regenerate everything: `python -m scripts.prepare_data && python -m scripts.generate_predictions && python -m scripts.export_dashboard_data`.

## Build the report (Power BI Desktop)

1. **Get data > Text/CSV**, load the tables above (the policy table is optional but enables drill-through). Set `small_sample` to True/False and numeric columns to Decimal.
2. Create a relationship from `policy_predictions_validation[risk_segment]` to a small `Segment` table if you want one slicer driving all pages (the segment order is Very low, Low, Medium, High, Very high; set *Sort by column* accordingly).
3. Add these measures (validation split, so totals are over the loaded rows):

```DAX
Policies = COUNTROWS(policy_predictions_validation)
Actual Cost = SUM(policy_predictions_validation[actual_claim_amount])
Expected Cost = SUM(policy_predictions_validation[predicted_cost_exposure])
Cost Actual / Expected = DIVIDE([Actual Cost], [Expected Cost])
Actual Frequency = DIVIDE(SUM(policy_predictions_validation[actual_claim_count]), SUM(policy_predictions_validation[exposure]))
Predicted Frequency = DIVIDE(SUMX(policy_predictions_validation, policy_predictions_validation[predicted_frequency] * policy_predictions_validation[exposure]), SUM(policy_predictions_validation[exposure]))
```

4. Pages:
   - **Executive overview**: cards for `Policies`, `Actual Cost`, `Expected Cost`, `Cost Actual / Expected`; clustered column of `actual_pure_premium` vs `expected_pure_premium` from `category_summary` filtered to dimension = Risk segment; a table of the headline rows of `model_metrics`.
   - **Risk analysis**: a slicer on `category_summary[dimension]`; bar chart of `expected_pure_premium` by `level`; table with `policies`, `exposure`, `small_sample` so small groups are visible; histogram from `predicted_cost_distribution`.
   - **Model performance**: matrix from `model_metrics` (rows = model, columns = metric, filtered by target); actual vs predicted frequency by `level` from `category_summary`.
   - **Monitoring**: bar chart of `drift_psi[psi]` by `feature` with reference lines at 0.1 and 0.25; show the note below.
5. Label every visual as **Historical outcome**, **Model prediction** or **Actual vs predicted** using `data_kind`.

## Interpretation notes to put on the report

- Figures are validation-split results used for model selection, not final test results.
- Expected cost is a pure-premium estimate of claims losses, not a customer price (no expenses, commissions, capital, reinsurance or taxes).
- Cost actual/expected ratios are noisy: they rest on few, heavy-tailed claims. See `reports/segment_ae_uncertainty_validation.csv` for bootstrap intervals.
- PSI compares train with validation, both random draws from one population, so values near zero are expected. The data has no dates, so performance over time cannot be shown.
