"""Segment-level actual vs expected cost. Uses the VALIDATION split only (test is reported separately)."""
import logging

import pandas as pd

from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_models")


def segment_table(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    g = pred.groupby(by, observed=True).agg(
        policies=("IDpol", "size"), exposure=("exposure", "sum"), actual_claims=("actual_claim_count", "sum"),
        expected_claims=("predicted_frequency", lambda s: float((s * pred.loc[s.index, "exposure"]).sum())),
        actual_cost=("actual_claim_amount", "sum"), expected_cost=("predicted_cost_exposure", "sum"))
    g["frequency_actual_to_expected"] = g["actual_claims"] / g["expected_claims"]
    g["cost_actual_to_expected"] = g["actual_cost"] / g["expected_cost"]
    g["actual_pure_premium"] = g["actual_cost"] / g["exposure"]
    g["expected_pure_premium"] = g["expected_cost"] / g["exposure"]
    return g.reset_index()


def main() -> None:
    cfg = load_config()
    pred = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / "predictions.parquet")
    val = pred[pred["split"] == "val"].copy()
    val["risk_segment"] = pd.Categorical(val["risk_segment"], cfg["segments"]["labels"], ordered=True)
    seg = segment_table(val, "risk_segment")
    log.info("Validation by predicted risk segment:\n%s", seg.round(3).to_string(index=False))
    seg.to_csv(ROOT / "reports" / "segment_performance_validation.csv", index=False)
    val["DrivAgeBand"] = pd.cut(val["DrivAge"], [17, 25, 35, 45, 55, 65, 100]).astype(str)
    stab = pd.concat([segment_table(val, c).rename(columns={c: "level"}).assign(feature=c) for c in ("DrivAgeBand", "VehGas", "Area")])
    stab.to_csv(ROOT / "reports" / "stability_by_feature_validation.csv", index=False)
    log.info("Stability by feature (cost actual/expected):\n%s", stab[["feature", "level", "policies", "frequency_actual_to_expected", "cost_actual_to_expected"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
