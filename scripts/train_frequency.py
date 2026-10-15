"""Fit frequency baselines on TRAIN, evaluate on VALIDATION. The test split is not read here."""
import json
import logging

import pandas as pd

from src.evaluation.metrics import frequency_summary, gini, lift_table
from src.models.frequency import ConstantRateBaseline, PoissonGLM
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_frequency")


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "policies_train.parquet")
    val = pd.read_parquet(proc / "policies_val.parquet")

    models = {"constant_rate": ConstantRateBaseline().fit(train), "poisson_glm": PoissonGLM(cfg).fit(train)}
    rows, lifts = [], {}
    for name, m in models.items():
        for split, df in (("train", train), ("validation", val)):
            mu = m.predict_counts(df)
            s = frequency_summary(df["ClaimNb"], mu, df["Exposure"])
            s["gini"] = gini(df["ClaimNb"].to_numpy(), mu, df["Exposure"].to_numpy())
            rows.append({"model": name, "split": split, **s})
        lifts[name] = lift_table(val["ClaimNb"].to_numpy(), m.predict_counts(val), val["Exposure"].to_numpy())
    res = pd.DataFrame(rows)
    disp = models["poisson_glm"].dispersion()
    log.info("Poisson GLM Pearson dispersion (train): %.3f", disp)
    log.info("\n%s", res.round(4).to_string(index=False))
    log.info("Validation lift (poisson_glm):\n%s", lifts["poisson_glm"].round(4).to_string(index=False))

    out = ROOT / "reports"
    res.to_csv(out / "frequency_model_comparison.csv", index=False)
    lifts["poisson_glm"].to_csv(out / "frequency_lift_validation.csv", index=False)
    (out / "frequency_dispersion.json").write_text(json.dumps({"poisson_glm_pearson_dispersion_train": disp}, indent=2))


if __name__ == "__main__":
    main()
