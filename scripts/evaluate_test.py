"""FINAL evaluation on the held-out test split. Run once, after model selection is fixed in the config.

Models were fitted on train only (validation was used for early stopping and for choosing models).
Baselines are refit on train. Nothing here changes any model.
"""
import argparse
import logging

import pandas as pd

from src.evaluation.holdout_guard import check_unused, record_use
from src.evaluation.metrics import (
    frequency_summary,
    gamma_deviance,
    gini,
    poisson_deviance,
    severity_summary,
)
from src.evaluation.uncertainty import bootstrap_metric_diff, bootstrap_ratio_ci
from src.models.frequency import ConstantRateBaseline, PoissonGLM
from src.models.severity import ConstantSeverityBaseline, claims_table
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run even though the test set was already used")
    args = ap.parse_args()
    cfg = load_config()
    marker = ROOT / "reports" / "test_evaluation_marker.json"
    check_unused(marker, args.force)

    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "policies_train.parquet")
    claims = pd.read_parquet(proc / "severity_claims.parquet")
    pred = pd.read_parquet(proc / "predictions.parquet")
    test_pred = pred[pred["split"] == "test"].set_index("IDpol")
    test = pd.read_parquet(proc / "policies_test.parquet")
    assert set(test["IDpol"]) == set(test_pred.index), "test predictions do not match the test policies"
    test_pred = test_pred.loc[test["IDpol"]]
    y, e = test["ClaimNb"].to_numpy(), test["Exposure"].to_numpy()

    # ---- frequency: selected model vs baselines (counts = rate x exposure) ----
    mu = {"constant_rate": ConstantRateBaseline().fit(train).predict_counts(test),
          "poisson_glm": PoissonGLM(cfg).fit(train).predict_counts(test),
          f"{cfg['selection']['frequency']} (selected)": (test_pred["predicted_frequency"] * test_pred["exposure"]).to_numpy()}
    sel_name = f"{cfg['selection']['frequency']} (selected)"
    rows = []
    for name, m in mu.items():
        s = frequency_summary(y, m, e)
        s["gini"] = gini(y, m, e)
        row = {"target": "frequency", "model": name, **s}
        if name != sel_name:
            b = bootstrap_metric_diff(y, mu[sel_name], m, poisson_deviance, n_boot=500, seed=cfg["seed"])
            row.update({"selected_dev_diff_vs_this": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
        rows.append(row)

    # ---- severity: selected vs constant mean, on test claims ----
    tclaims = claims_table(test, claims)
    sev_sel = test_pred.loc[tclaims["IDpol"], "predicted_severity"].to_numpy()
    const = ConstantSeverityBaseline().fit(claims_table(train, claims)).predict(tclaims)
    ya = tclaims["ClaimAmount"].to_numpy()
    sev_name = f"{cfg['selection']['severity']} (selected)"
    for name, m in ((sev_name, sev_sel), ("constant_mean", const)):
        row = {"target": "severity", "model": name, **severity_summary(ya, m)}
        if name == sev_name:
            b = bootstrap_metric_diff(ya, sev_sel, const, gamma_deviance, n_boot=500, seed=cfg["seed"])
            row.update({"selected_dev_diff_vs_this": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
        rows.append(row)
    res = pd.DataFrame(rows)
    log.info("Test-set results:\n%s", res.round(4).to_string(index=False))
    res.to_csv(ROOT / "reports" / "test_results.csv", index=False)

    # ---- expected cost: overall and by segment, with bootstrap intervals ----
    test_pred = test_pred.assign(expected_claims=test_pred["predicted_frequency"] * test_pred["exposure"])
    seg_rows = []
    for name, g in [("ALL", test_pred)] + [(s, test_pred[test_pred["risk_segment"] == s]) for s in cfg["segments"]["labels"]]:
        f = bootstrap_ratio_ci(g["actual_claim_count"], g["expected_claims"], seed=cfg["seed"])
        c = bootstrap_ratio_ci(g["actual_claim_amount"], g["predicted_cost_exposure"], seed=cfg["seed"])
        seg_rows.append({"segment": name, "policies": len(g), "freq_ae": f["ratio"], "freq_ci_low": f["ci_low"], "freq_ci_high": f["ci_high"],
                         "cost_ae": c["ratio"], "cost_ci_low": c["ci_low"], "cost_ci_high": c["ci_high"],
                         "actual_pure_premium": g["actual_claim_amount"].sum() / g["exposure"].sum(),
                         "expected_pure_premium": g["predicted_cost_exposure"].sum() / g["exposure"].sum()})
    seg = pd.DataFrame(seg_rows)
    log.info("Test-set segments:\n%s", seg.round(3).to_string(index=False))
    seg.to_csv(ROOT / "reports" / "test_segment_performance.csv", index=False)

    record_use(marker, {"selected_frequency": cfg["selection"]["frequency"], "selected_severity": cfg["selection"]["severity"],
                        "test_policies": len(test), "test_claims": len(tclaims)})


if __name__ == "__main__":
    main()
