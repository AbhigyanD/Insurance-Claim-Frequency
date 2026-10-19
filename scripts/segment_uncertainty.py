"""Bootstrap intervals for actual/expected ratios by risk segment. VALIDATION split only."""
import logging

import pandas as pd

from src.evaluation.uncertainty import bootstrap_ratio_ci
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("segment_uncertainty")


def main() -> None:
    cfg = load_config()
    pred = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / "predictions.parquet")
    val = pred[pred["split"] == "val"].copy()
    val["expected_claims"] = val["predicted_frequency"] * val["exposure"]
    rows = []
    groups = [("ALL", val)] + [(s, val[val["risk_segment"] == s]) for s in cfg["segments"]["labels"]]
    for name, g in groups:
        f = bootstrap_ratio_ci(g["actual_claim_count"], g["expected_claims"], seed=cfg["seed"])
        c = bootstrap_ratio_ci(g["actual_claim_amount"], g["predicted_cost_exposure"], seed=cfg["seed"])
        rows.append({"segment": name, "policies": len(g),
                     "freq_ae": f["ratio"], "freq_ci_low": f["ci_low"], "freq_ci_high": f["ci_high"], "freq_contains_one": f["contains_one"],
                     "cost_ae": c["ratio"], "cost_ci_low": c["ci_low"], "cost_ci_high": c["ci_high"], "cost_contains_one": c["contains_one"]})
    out = pd.DataFrame(rows)
    log.info("A/E with 95%% bootstrap intervals (validation):\n%s", out.round(3).to_string(index=False))
    out.to_csv(ROOT / "reports" / "segment_ae_uncertainty_validation.csv", index=False)


if __name__ == "__main__":
    main()
