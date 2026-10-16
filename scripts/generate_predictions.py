"""Fit the SELECTED models on train and write policy-level expected-cost predictions for all splits."""
import logging

import joblib
import pandas as pd

from src.models.frequency import LightGBMFrequency
from src.models.pricing import assign_segments, expected_cost_table, segment_edges
from src.models.severity import RidgeGammaGLM, claims_table
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_predictions")

RISK_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Region", "Density"]


def main() -> None:
    cfg = load_config()
    sel = cfg["selection"]
    assert sel["frequency"] == "lightgbm_poisson" and sel["severity"] == "ridge_gamma_glm", "update script for new selection"
    proc = ROOT / cfg["paths"]["processed_dir"]
    splits = {n: pd.read_parquet(proc / f"policies_{n}.parquet") for n in ("train", "val", "test")}
    claims = pd.read_parquet(proc / "severity_claims.parquet")

    freq = LightGBMFrequency(cfg).fit(splits["train"], splits["val"])
    sev = RidgeGammaGLM(cfg).fit(claims_table(splits["train"], claims), claims_table(splits["val"], claims))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"frequency": freq, "severity": sev, "config": cfg}, ROOT / "models" / "selected_models.joblib")

    tables = []
    for name, df in splits.items():
        t = expected_cost_table(df, freq.predict_counts(df), sev.predict(df))
        t.insert(1, "split", name)
        tables.append(t)
    edges = segment_edges(tables[0]["predicted_annual_cost"].to_numpy(), cfg["segments"]["n_segments"])
    out = []
    for (name, df), t in zip(splits.items(), tables):
        t = t.copy()
        t["risk_segment"] = assign_segments(t["predicted_annual_cost"].to_numpy(), edges, cfg["segments"]["labels"])
        # actual outcomes kept in clearly named columns, separate from model inputs
        t["actual_claim_count"] = df["ClaimNb"].to_numpy()
        t["actual_claim_amount"] = df["TotalClaimAmount"].to_numpy()
        t["exposure"] = df["Exposure"].to_numpy()
        for c in RISK_COLS:
            t[c] = df[c].to_numpy()
        out.append(t)
    pred = pd.concat(out, ignore_index=True)
    dest = ROOT / cfg["paths"]["processed_dir"] / "predictions.parquet"
    pred.to_parquet(dest, index=False)
    log.info("Wrote %d predictions to %s; segment cut points (train): %s", len(pred), dest, edges.round(1))
    v = pred[pred["split"] == "val"]
    log.info("Validation: actual cost=%.0f, expected cost over exposure=%.0f (ratio %.3f)",
             v["actual_claim_amount"].sum(), v["predicted_cost_exposure"].sum(),
             v["actual_claim_amount"].sum() / v["predicted_cost_exposure"].sum())


if __name__ == "__main__":
    main()
