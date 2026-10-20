"""Experiments with modern variants. VALIDATION split only; the test split is never read here.

1. Frequency: monotone-constrained LightGBM, GLM-boosted LightGBM vs the selected LightGBM.
2. Pure premium: direct Tweedie LightGBM vs the frequency x severity pipeline.
3. PSI drift: train vs validation (features and predictions).
"""
import logging

import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from src.evaluation.drift import psi_table
from src.evaluation.metrics import frequency_summary, gini, poisson_deviance
from src.evaluation.uncertainty import bootstrap_metric_diff
from src.models.frequency import LightGBMFrequency
from src.models.frequency_variants import GLMBoostedFrequency, MonotoneLightGBMFrequency
from src.models.pure_premium import TweedieLightGBM
from src.utils.config import ROOT, load_config
from src.utils.tracking import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experiments")


def main() -> None:
    cfg = load_config()
    proc, rep = ROOT / cfg["paths"]["processed_dir"], ROOT / "reports"
    train, val = (pd.read_parquet(proc / f"policies_{n}.parquet") for n in ("train", "val"))

    # ---- 1. frequency variants ----
    models = {"lightgbm_poisson": LightGBMFrequency(cfg).fit(train, val),
              "lightgbm_monotone_bonusmalus": MonotoneLightGBMFrequency(cfg, {"BonusMalus": 1}).fit(train, val),
              "glm_boosted_lightgbm": GLMBoostedFrequency(cfg).fit(train, val)}
    y, e = val["ClaimNb"].to_numpy(), val["Exposure"].to_numpy()
    base_mu = models["lightgbm_poisson"].predict_counts(val)
    rows = []
    for name, m in models.items():
        mu = m.predict_counts(val)
        s = frequency_summary(y, mu, e)
        s["gini"] = gini(y, mu, e)
        row = {"model": name, **s}
        if name != "lightgbm_poisson":
            b = bootstrap_metric_diff(y, mu, base_mu, poisson_deviance, n_boot=500, seed=cfg["seed"])
            row.update({"dev_diff_vs_lightgbm": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"], "significant": b["significant"]})
        rows.append(row)
        log_run(cfg, "claim-frequency-experiments", name, {"model": name},
                {f"val_{k}": v for k, v in {**s, "gini": gini(y, mu, e)}.items()})
    fv = pd.DataFrame(rows)
    log.info("Frequency variants (validation):\n%s", fv.round(4).to_string(index=False))
    fv.to_csv(rep / "frequency_variants_validation.csv", index=False)

    # ---- 2. direct Tweedie pure premium vs frequency x severity ----
    pred = pd.read_parquet(proc / "predictions.parquet").query("split == 'val'").set_index("IDpol")
    v = val.set_index("IDpol")
    pipe_rate = pred.loc[v.index, "predicted_annual_cost"].to_numpy()
    ycost, w = v["TotalClaimAmount"].to_numpy(), v["Exposure"].to_numpy()
    yrate = ycost / w

    def pp_row(name, rate, extra=None):
        return {"model": name, "tweedie_dev_p1.5": float(mean_tweedie_deviance(yrate, rate, sample_weight=w, power=1.5)),
                "cost_gini": gini(ycost, rate * w, w), "actual_to_expected": float(ycost.sum() / (rate * w).sum()), **(extra or {})}

    pp = [pp_row("frequency_x_severity (selected)", pipe_rate)]
    for p in (1.3, 1.5, 1.7):
        t = TweedieLightGBM(cfg, p).fit(train, val)
        pp.append(pp_row(f"tweedie_lightgbm_p{p}", t.predict_annual_cost(val), {"trees": t.best_iteration_}))
    pp = pd.DataFrame(pp)
    log.info("Pure premium (validation):\n%s", pp.round(4).to_string(index=False))
    pp.to_csv(rep / "pure_premium_variants_validation.csv", index=False)

    # ---- 3. PSI drift ----
    cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    ptrain = pd.read_parquet(proc / "predictions.parquet").query("split == 'train'")
    pval = pd.read_parquet(proc / "predictions.parquet").query("split == 'val'")
    drift = psi_table(train, val, cols)
    pdrift = psi_table(ptrain, pval, ["predicted_frequency", "predicted_severity", "predicted_annual_cost"])
    out = pd.concat([drift.assign(kind="feature"), pdrift.assign(kind="prediction")], ignore_index=True)
    out.to_csv(rep / "drift_psi_train_vs_val.csv", index=False)
    log.info("PSI train vs validation:\n%s", out.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
