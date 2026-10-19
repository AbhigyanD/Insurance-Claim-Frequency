"""Power BI-ready tables built from model outputs (VALIDATION split only).

Every table carries a `data_kind` column so historical outcomes, model predictions and (if ever added)
simulated examples can be told apart in a report.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL, PREDICTION, COMPARISON = "historical_outcome", "model_prediction", "actual_vs_predicted"

BANDS = {
    "driver_age_band": ("DrivAge", [17, 24, 34, 49, 64, 120], ["18-24", "25-34", "35-49", "50-64", "65+"]),
    "vehicle_age_band": ("VehAge", [-1, 2, 5, 10, 200], ["0-2", "3-5", "6-10", "11+"]),
}
DIMENSIONS = {"Risk segment": "risk_segment", "Driver age": "driver_age_band", "Vehicle age": "vehicle_age_band",
              "Area": "Area", "Region": "Region", "Brand": "VehBrand", "Fuel": "VehGas"}


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for new, (col, edges, labels) in BANDS.items():
        out[new] = pd.cut(out[col], edges, labels=labels).astype(str)
    return out


def policy_table(pred_val: pd.DataFrame) -> pd.DataFrame:
    """Policy-level rows: identifiers, risk features, predictions and actuals in clearly separate columns."""
    df = add_bands(pred_val)
    cols = ["IDpol", "risk_segment", "exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area",
            "Region", "Density", "driver_age_band", "vehicle_age_band", "predicted_frequency", "predicted_severity",
            "predicted_annual_cost", "predicted_cost_exposure", "actual_claim_count", "actual_claim_amount"]
    return df[cols].rename(columns={"IDpol": "policy_id"})


def category_summary(pred_val: pd.DataFrame, dimensions: dict[str, str] = DIMENSIONS) -> pd.DataFrame:
    """Long table: one row per dimension level with actual and expected claims/cost and sample size."""
    df = add_bands(pred_val)
    df["expected_claims"] = df["predicted_frequency"] * df["exposure"]
    rows = []
    for name, col in dimensions.items():
        g = df.groupby(col, observed=True).agg(
            policies=("policy_id" if "policy_id" in df else "IDpol", "size"), exposure=("exposure", "sum"),
            actual_claims=("actual_claim_count", "sum"), expected_claims=("expected_claims", "sum"),
            actual_cost=("actual_claim_amount", "sum"), expected_cost=("predicted_cost_exposure", "sum")).reset_index()
        g = g.rename(columns={col: "level"}).assign(dimension=name)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["actual_frequency"] = out["actual_claims"] / out["exposure"]
    out["predicted_frequency"] = out["expected_claims"] / out["exposure"]
    out["actual_pure_premium"] = out["actual_cost"] / out["exposure"]
    out["expected_pure_premium"] = out["expected_cost"] / out["exposure"]
    out["cost_actual_to_expected"] = out["actual_cost"] / out["expected_cost"]
    out["small_sample"] = out["exposure"] < 1000
    out["data_kind"] = COMPARISON
    out["level"] = out["level"].astype(str)
    return out[["dimension", "level", "policies", "exposure", "actual_claims", "expected_claims", "actual_cost", "expected_cost",
                "actual_frequency", "predicted_frequency", "actual_pure_premium", "expected_pure_premium",
                "cost_actual_to_expected", "small_sample", "data_kind"]]


def model_metrics_long(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Validation metrics in long format: target, model, metric, value."""
    f = freq[freq["split"] == "validation"].melt(id_vars=["model"], value_vars=["poisson_deviance", "mae_counts", "actual_to_predicted", "gini"],
                                                  var_name="metric", value_name="value").assign(target="frequency")
    s = sev[sev["split"] == "validation"].melt(id_vars=["model"], value_vars=["gamma_deviance", "mae", "rmse", "actual_to_predicted"],
                                               var_name="metric", value_name="value").assign(target="severity")
    out = pd.concat([f, s], ignore_index=True)[["target", "model", "metric", "value"]]
    out["data_kind"] = PREDICTION
    return out


def distribution_table(pred_val: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    cost = pred_val["predicted_annual_cost"].to_numpy()
    edges = np.linspace(cost.min(), np.quantile(cost, 0.995), bins + 1)
    counts, _ = np.histogram(np.clip(cost, edges[0], edges[-1]), bins=edges)
    return pd.DataFrame({"bin_from": edges[:-1], "bin_to": edges[1:], "policies": counts, "data_kind": PREDICTION})


# (table, column) -> description. Exports are validated against this so nothing ships undocumented.
_COMMON = {
    "data_kind": "historical_outcome (observed), model_prediction (model output) or actual_vs_predicted (both side by side). Simulated data is never mixed in.",
}
DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "policy_predictions_validation": {
        "policy_id": "Policy identifier from the source data (IDpol). Stable, not personal data.",
        "risk_segment": "Quintile of predicted annual cost; cut points set from training data only.",
        "exposure": "Policy exposure in years (capped at 1).",
        "VehPower": "Vehicle power (ordered integer). Model input.", "VehAge": "Vehicle age in years. Model input.",
        "DrivAge": "Driver age in years. Model input.", "BonusMalus": "Bonus-malus score (50 best, 100 neutral). Model input.",
        "VehBrand": "Anonymised vehicle brand group. Model input.", "VehGas": "Diesel or Regular. Model input.",
        "Area": "Anonymised density-based area code A-F. Model input.", "Region": "French region. Model input.",
        "Density": "Inhabitants per km2 in the driver's city. Model input.",
        "driver_age_band": "DrivAge grouped for reporting.", "vehicle_age_band": "VehAge grouped for reporting.",
        "predicted_frequency": "MODEL PREDICTION: expected claims per exposure-year.",
        "predicted_severity": "MODEL PREDICTION: expected cost per claim (EUR).",
        "predicted_annual_cost": "MODEL PREDICTION: frequency x severity for a full year (EUR). A pure-premium estimate, not a price.",
        "predicted_cost_exposure": "MODEL PREDICTION: expected cost over the exposure actually observed (EUR).",
        "actual_claim_count": "HISTORICAL OUTCOME: observed number of claims. Never a model input.",
        "actual_claim_amount": "HISTORICAL OUTCOME: observed total claim amount (EUR). Never a model input.",
    },
    "category_summary": {
        "dimension": "Grouping used (Risk segment, Driver age, Vehicle age, Area, Region, Brand, Fuel).", "level": "Value of the dimension.",
        "policies": "Number of policies.", "exposure": "Exposure-years.",
        "actual_claims": "HISTORICAL OUTCOME: observed claims.", "expected_claims": "MODEL PREDICTION: expected claims over observed exposure.",
        "actual_cost": "HISTORICAL OUTCOME: observed claim amount (EUR).", "expected_cost": "MODEL PREDICTION: expected cost (EUR).",
        "actual_frequency": "HISTORICAL OUTCOME: claims per exposure-year.", "predicted_frequency": "MODEL PREDICTION: claims per exposure-year.",
        "actual_pure_premium": "HISTORICAL OUTCOME: cost per exposure-year (EUR).", "expected_pure_premium": "MODEL PREDICTION: cost per exposure-year (EUR).",
        "cost_actual_to_expected": "Actual / expected cost. Noisy for small or heavy-tailed groups; see small_sample.",
        "small_sample": "True if the group has under 1,000 exposure-years; read ratios with care.",
        **_COMMON,
    },
    "model_metrics": {
        "target": "frequency or severity.", "model": "Model name.", "metric": "Metric name (deviance, MAE, Gini, ...).",
        "value": "Validation-split value. Lower is better for deviance/MAE/RMSE; higher is better for Gini.", **_COMMON,
    },
    "predicted_cost_distribution": {
        "bin_from": "Lower bound of the predicted annual cost bin (EUR).", "bin_to": "Upper bound (EUR); last bin also holds values above the 99.5th percentile.",
        "policies": "Policies in the bin.", **_COMMON,
    },
    "drift_psi": {
        "feature": "Feature or prediction column.", "psi": "Population stability index, training vs validation.",
        "status": "stable (<0.1), moderate (0.1-0.25) or major (>0.25).", "kind": "feature or prediction.", **_COMMON,
    },
}


def dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame([{"table": t, "column": c, "description": d} for t, cols in DATA_DICTIONARY.items() for c, d in cols.items()])


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Fail loudly instead of shipping undocumented columns, test-split rows or NaN values."""
    for name, df in tables.items():
        if name not in DATA_DICTIONARY:
            raise ValueError(f"Table '{name}' has no data dictionary entry")
        missing = set(df.columns) - set(DATA_DICTIONARY[name])
        if missing:
            raise ValueError(f"Table '{name}' has undocumented columns: {sorted(missing)}")
        if "split" in df.columns and set(df["split"]) != {"val"}:
            raise ValueError(f"Table '{name}' contains non-validation rows")
        if df.isna().any().any():
            raise ValueError(f"Table '{name}' contains missing values")


def export_all(pred: pd.DataFrame, reports_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Write all tables as CSV. Only validation rows are ever used."""
    val = pred[pred["split"] == "val"]
    if val.empty:
        raise ValueError("No validation rows in predictions")
    psi = pd.read_csv(reports_dir / "drift_psi_train_vs_val.csv").assign(data_kind=PREDICTION)
    tables = {
        "policy_predictions_validation": policy_table(val),
        "category_summary": category_summary(val),
        "model_metrics": model_metrics_long(pd.read_csv(reports_dir / "frequency_model_comparison.csv"),
                                            pd.read_csv(reports_dir / "severity_model_comparison.csv")),
        "predicted_cost_distribution": distribution_table(val),
        "drift_psi": psi[["feature", "psi", "status", "kind", "data_kind"]],
    }
    validate_tables(tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    dictionary_frame().to_csv(out_dir / "data_dictionary.csv", index=False)
    return tables
