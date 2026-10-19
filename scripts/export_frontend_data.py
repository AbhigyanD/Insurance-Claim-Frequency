"""Export small aggregate JSON files (VALIDATION split only) for the React dashboard.

No policy-level rows and no test-split figures are exported.
"""
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_frontend_data")

OUT = ROOT / "frontend" / "public" / "data"


def clean(obj):
    """Convert pandas/numpy values to JSON-safe Python (NaN -> None)."""
    if isinstance(obj, pd.DataFrame):
        return [clean(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if np.isscalar(obj) else False:
        return None
    return obj


def dump(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(clean(data), indent=1, allow_nan=False))


def group_table(v: pd.DataFrame, by: str) -> pd.DataFrame:
    g = v.groupby(by, observed=True).agg(
        policies=("IDpol", "size"), exposure=("exposure", "sum"), claims=("actual_claim_count", "sum"),
        actual_cost=("actual_claim_amount", "sum"), expected_cost=("predicted_cost_exposure", "sum"),
        avg_expected_annual_cost=("predicted_annual_cost", "mean")).reset_index()
    exp_claims = v.assign(ec=v["predicted_frequency"] * v["exposure"]).groupby(by, observed=True)["ec"].sum().to_numpy()
    g["actual_frequency"] = g["claims"] / g["exposure"]
    g["predicted_frequency"] = exp_claims / g["exposure"]
    g["actual_pure_premium"] = g["actual_cost"] / g["exposure"]
    g["expected_pure_premium"] = g["expected_cost"] / g["exposure"]
    g["cost_actual_to_expected"] = g["actual_cost"] / g["expected_cost"]
    return g.rename(columns={by: "level"})


def main() -> None:
    cfg = load_config()
    rep, proc = ROOT / "reports", ROOT / cfg["paths"]["processed_dir"]
    pred = pd.read_parquet(proc / "predictions.parquet")
    v = pred[pred["split"] == "val"].copy()

    fm = pd.read_csv(rep / "frequency_model_comparison.csv")
    sm = pd.read_csv(rep / "severity_model_comparison.csv")
    sel = cfg["selection"]
    fsel = fm[(fm["model"] == sel["frequency"]) & (fm["split"] == "validation")].iloc[0]
    ssel = sm[(sm["model"] == sel["severity"]) & (sm["split"] == "validation")].iloc[0]
    exp_claims = float((v["predicted_frequency"] * v["exposure"]).sum())
    ae = pd.read_csv(rep / "segment_ae_uncertainty_validation.csv").query("segment == 'ALL'").iloc[0]
    dump("overview", {
        "cost_ae_ci_low": ae["cost_ci_low"], "cost_ae_ci_high": ae["cost_ci_high"],
        "freq_ae": ae["freq_ae"], "freq_ae_ci_low": ae["freq_ci_low"], "freq_ae_ci_high": ae["freq_ci_high"],
        "policies": len(v), "exposure": v["exposure"].sum(), "actual_claims": v["actual_claim_count"].sum(),
        "expected_claims": exp_claims, "actual_cost": v["actual_claim_amount"].sum(),
        "expected_cost": v["predicted_cost_exposure"].sum(),
        "actual_to_expected": v["actual_claim_amount"].sum() / v["predicted_cost_exposure"].sum(),
        "selected_frequency_model": sel["frequency"], "selected_severity_model": sel["severity"],
        "frequency_metrics": fsel[["poisson_deviance", "mae_counts", "actual_to_predicted", "gini"]].to_dict(),
        "severity_metrics": ssel[["gamma_deviance", "mae", "rmse", "actual_to_predicted"]].to_dict()})

    labels = cfg["segments"]["labels"]
    seg = pd.read_csv(rep / "segment_performance_validation.csv")
    seg["risk_segment"] = pd.Categorical(seg["risk_segment"], labels, ordered=True)
    dump("segments", seg.sort_values("risk_segment").assign(risk_segment=lambda d: d["risk_segment"].astype(str)))

    v["driver_age_band"] = pd.cut(v["DrivAge"], [17, 24, 34, 49, 64, 120], labels=["18-24", "25-34", "35-49", "50-64", "65+"]).astype(str)
    v["vehicle_age_band"] = pd.cut(v["VehAge"], [-1, 2, 5, 10, 100], labels=["0-2", "3-5", "6-10", "11+"]).astype(str)
    risk = {k: group_table(v, c) for k, c in (("area", "Area"), ("region", "Region"), ("driver_age", "driver_age_band"),
                                              ("vehicle_age", "vehicle_age_band"), ("fuel", "VehGas"), ("brand", "VehBrand"))}
    risk["region"] = risk["region"].sort_values("avg_expected_annual_cost", ascending=False)
    dump("risk", risk)

    cost = v["predicted_annual_cost"].to_numpy()
    edges = np.linspace(cost.min(), np.quantile(cost, 0.995), 41)
    counts, _ = np.histogram(np.clip(cost, edges[0], edges[-1]), bins=edges)
    dump("distribution", {"bins": [{"from": a, "to": b, "policies": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)],
                          "note": "Predicted annual cost per policy; values above the 99.5th percentile are grouped in the last bin.",
                          "p50": np.quantile(cost, .5), "p90": np.quantile(cost, .9), "p99": np.quantile(cost, .99)})

    fv, pv = pd.read_csv(rep / "frequency_variants_validation.csv"), pd.read_csv(rep / "pure_premium_variants_validation.csv")
    dump("models", {
        "frequency": fm[fm["split"] == "validation"], "severity": sm[sm["split"] == "validation"],
        "severity_bootstrap": pd.read_csv(rep / "severity_bootstrap_vs_constant.csv"),
        "frequency_variants": fv, "pure_premium_variants": pv,
        "dispersion": json.loads((rep / "frequency_dispersion.json").read_text())})
    dump("calibration", {
        "frequency_lightgbm": pd.read_csv(rep / "frequency_lift_validation_lightgbm_poisson.csv"),
        "frequency_poisson_glm": pd.read_csv(rep / "frequency_lift_validation_poisson_glm.csv"),
        "severity_lightgbm": pd.read_csv(rep / "severity_calibration_validation_lightgbm_gamma.csv"),
        "severity_gamma_glm": pd.read_csv(rep / "severity_calibration_validation_gamma_glm.csv"),
        "stability": pd.read_csv(rep / "stability_by_feature_validation.csv")})
    rel = pd.read_csv(rep / "frequency_glm_relativities.csv")
    dump("explain", {"shap": pd.read_csv(rep / "frequency_shap_importance.csv"),
                     "relativities": rel[rel["term"].str.startswith(("DrivAge_band", "BonusMalus_band", "VehGas", "LogDensity"))]})
    dump("monitoring", {"psi": pd.read_csv(rep / "drift_psi_train_vs_val.csv"),
                        "note": "PSI compares the training split with the validation split. Both are random draws from one population and the dataset has no date column, so values near zero are expected. This is a baseline, not evidence about production drift."})
    dump("meta", {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "validation",
                  "source": "freMTPL2freq / freMTPL2sev (CASdatasets, French motor third-party liability, ~2011-2013)",
                  "note": "All figures come from the validation split used for model selection. The test split is not shown. Expected cost is a pure-premium estimate of claims losses, not a customer price."})
    log.info("Wrote %d JSON files to %s", len(list(OUT.glob("*.json"))), OUT)


if __name__ == "__main__":
    main()
