"""Fit severity models on TRAIN claims, evaluate on VALIDATION claims. Test split is not read here."""
import logging

import pandas as pd

from src.evaluation.metrics import large_claim_analysis, severity_calibration, severity_summary
from src.evaluation.metrics import gamma_deviance
from src.evaluation.uncertainty import bootstrap_metric_diff
from src.models.severity import ConstantSeverityBaseline, GammaGLM, LightGBMSeverity, RidgeGammaGLM, claims_table
from src.utils.config import ROOT, load_config
from src.utils.tracking import dataset_fingerprint, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_severity")


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    claims = pd.read_parquet(proc / "severity_claims.parquet")
    tr = claims_table(pd.read_parquet(proc / "policies_train.parquet"), claims)
    va = claims_table(pd.read_parquet(proc / "policies_val.parquet"), claims)
    log.info("Severity rows: train=%d validation=%d", len(tr), len(va))

    cap_q = cfg["severity"]["cap_quantile"]
    models = {"constant_mean": ConstantSeverityBaseline().fit(tr), "gamma_glm": GammaGLM(cfg).fit(tr),
              f"gamma_glm_capped_p{cap_q * 100:g}": GammaGLM(cfg, cap_q).fit(tr), "ridge_gamma_glm": RidgeGammaGLM(cfg).fit(tr, va),
              "lightgbm_gamma": LightGBMSeverity(cfg).fit(tr, va)}
    log.info("Ridge alpha search (validation gamma deviance): %s; chosen alpha=%s",
             models["ridge_gamma_glm"].search_, models["ridge_gamma_glm"].alpha_)
    rows = []
    for name, m in models.items():
        for split, d in (("train", tr), ("validation", va)):
            rows.append({"model": name, "split": split, **severity_summary(d["ClaimAmount"], m.predict(d))})
        la = large_claim_analysis(va["ClaimAmount"].to_numpy(), m.predict(va))
        rows[-1].update({f"top1pct_{k}": v for k, v in la.items() if k in ("actual_to_predicted_large",)})
    res = pd.DataFrame(rows)
    log.info("\n%s", res.round(3).to_string(index=False))
    for name in ("gamma_glm", "lightgbm_gamma"):
        cal = severity_calibration(va["ClaimAmount"].to_numpy(), models[name].predict(va))
        log.info("Validation calibration (%s):\n%s", name, cal.round(3).to_string(index=False))
        cal.to_csv(ROOT / "reports" / f"severity_calibration_validation_{name}.csv", index=False)
    log.info("Large-claim analysis (validation, gamma_glm): %s",
             large_claim_analysis(va["ClaimAmount"].to_numpy(), models["gamma_glm"].predict(va)))
    res.to_csv(ROOT / "reports" / "severity_model_comparison.csv", index=False)
    data_hash = dataset_fingerprint(proc / "severity_claims.parquet")
    for name, m in models.items():
        v = res[(res["model"] == name) & (res["split"] == "validation")].iloc[0].to_dict()
        params = {"model": name, "features_numeric": cfg["features"]["numeric"], "features_categorical": cfg["features"]["categorical"],
                  "severity_data_hash": data_hash}
        if hasattr(m, "params"):
            params.update(m.params)
        if hasattr(m, "alpha_"):
            params["ridge_alpha"] = m.alpha_
        log_run(cfg, cfg["mlflow"]["severity_experiment"], name, params,
                {f"val_{k}": v[k] for k in ("gamma_deviance", "mae", "rmse", "actual_to_predicted")},
                tags={"selected": str(cfg["selection"]["severity"] == name)})

    # Is any model reliably better than the constant mean? Bootstrap validation claims.
    y, base = va["ClaimAmount"].to_numpy(), models["constant_mean"].predict(va)
    boot = []
    for name, m in models.items():
        if name == "constant_mean":
            continue
        r = bootstrap_metric_diff(y, m.predict(va), base, gamma_deviance, n_boot=1000, seed=cfg["seed"])
        boot.append({"model": name, "metric": "gamma_deviance_minus_constant", **r})
    boot = pd.DataFrame(boot)
    log.info("Bootstrap (validation): negative diff = better than constant mean\n%s", boot.round(4).to_string(index=False))
    boot.to_csv(ROOT / "reports" / "severity_bootstrap_vs_constant.csv", index=False)


if __name__ == "__main__":
    main()
