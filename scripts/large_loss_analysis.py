"""Cap-and-load analysis. Fit on TRAIN, evaluate on VALIDATION only; the test split is not read."""
import logging

import numpy as np
import pandas as pd

from src.evaluation.metrics import severity_summary
from src.evaluation.uncertainty import bootstrap_ratio_ci
from src.models.large_loss import cap_threshold, excess_load, load_sensitivity
from src.models.severity import GammaGLM, claims_table
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("large_loss")


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train_p, val_p = (pd.read_parquet(proc / f"policies_{n}.parquet") for n in ("train", "val"))
    claims = pd.read_parquet(proc / "severity_claims.parquet")
    tr, va = claims_table(train_p, claims), claims_table(val_p, claims)
    pred = pd.read_parquet(proc / "predictions.parquet").query("split == 'val'").set_index("IDpol").loc[val_p["IDpol"]]
    exp_claims = (pred["predicted_frequency"] * pred["exposure"]).to_numpy()
    actual_cost = pred["actual_claim_amount"].to_numpy()

    rows = []
    for q in (0.99, 0.995, 0.999):
        cap = cap_threshold(tr["ClaimAmount"], q)
        sens = load_sensitivity(tr["ClaimAmount"].to_numpy(), cap)
        m = GammaGLM(cfg, q).fit(tr)
        for variant, factor in (("capped", 1.0), ("capped+load", 1.0 + sens["load"]), ("capped+load(no top claim)", 1.0 + sens["load_without_top"])):
            sev_claims = m.predict(va) * factor
            s = severity_summary(va["ClaimAmount"], sev_claims)
            cost = exp_claims * m.predict(val_p) * factor
            ci = bootstrap_ratio_ci(actual_cost, cost, seed=cfg["seed"])
            rows.append({"cap_quantile": q, "cap": cap, "variant": variant, "train_load": sens["load"],
                         "load_without_top_claim": sens["load_without_top"], "top_claim_share_of_excess": sens["top_claims_share_of_excess"],
                         "sev_pred_mean": s["predicted_mean"], "sev_actual_mean": s["actual_mean"], "sev_mae": s["mae"],
                         "cost_ae": ci["ratio"], "cost_ae_low": ci["ci_low"], "cost_ae_high": ci["ci_high"]})
    out = pd.DataFrame(rows)
    log.info("Train claims: n=%d mean=%.0f max=%.0f\n%s", len(tr), tr["ClaimAmount"].mean(), tr["ClaimAmount"].max(),
             out.round(3).to_string(index=False))
    out.to_csv(ROOT / "reports" / "large_loss_analysis_validation.csv", index=False)


if __name__ == "__main__":
    main()
