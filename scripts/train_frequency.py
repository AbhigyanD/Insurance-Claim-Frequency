"""Fit frequency baselines on TRAIN, evaluate on VALIDATION. The test split is not read here."""
import json
import logging

import pandas as pd

from src.evaluation.metrics import frequency_summary, gini, lift_table
from src.models.frequency import ConstantRateBaseline, LightGBMFrequency, NegBinGLM, PoissonGLM, poisson_log_likelihood
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_frequency")


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "policies_train.parquet")
    val = pd.read_parquet(proc / "policies_val.parquet")

    glm, nb = PoissonGLM(cfg).fit(train), NegBinGLM(cfg).fit(train)
    # Early stopping uses the VALIDATION split (model selection); the test split is never read here.
    lgbm = LightGBMFrequency(cfg).fit(train, val)
    log.info("NB alpha=%.4f; LightGBM best iteration=%d", nb.alpha_, lgbm.best_iteration_)
    models = {"constant_rate": ConstantRateBaseline().fit(train), "poisson_glm": glm, "negbin_glm": nb,
              "lightgbm_poisson": lgbm}
    rows, lifts = [], {}
    for name, m in models.items():
        for split, df in (("train", train), ("validation", val)):
            mu = m.predict_counts(df)
            s = frequency_summary(df["ClaimNb"], mu, df["Exposure"])
            s["gini"] = gini(df["ClaimNb"].to_numpy(), mu, df["Exposure"].to_numpy())
            rows.append({"model": name, "split": split, **s})
        lifts[name] = lift_table(val["ClaimNb"].to_numpy(), m.predict_counts(val), val["Exposure"].to_numpy())
    res = pd.DataFrame(rows)
    disp = glm.dispersion()
    ll = {"poisson_glm_val_loglik": poisson_log_likelihood(val["ClaimNb"], glm.predict_counts(val)),
          "negbin_glm_val_loglik": nb.log_likelihood(val), "nb_alpha": nb.alpha_,
          "negbin_pearson_dispersion_train": nb.dispersion()}
    log.info("Validation mean log-likelihood / NB alpha: %s", ll)
    log.info("Poisson GLM Pearson dispersion (train): %.3f", disp)
    log.info("\n%s", res.round(4).to_string(index=False))
    log.info("Validation lift (poisson_glm):\n%s", lifts["poisson_glm"].round(4).to_string(index=False))

    out = ROOT / "reports"
    res.to_csv(out / "frequency_model_comparison.csv", index=False)
    for n in ("poisson_glm", "lightgbm_poisson"):
        lifts[n].to_csv(out / f"frequency_lift_validation_{n}.csv", index=False)
    (out / "frequency_dispersion.json").write_text(json.dumps({"poisson_glm_pearson_dispersion_train": disp, **ll}, indent=2))


if __name__ == "__main__":
    main()
