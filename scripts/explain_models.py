"""Explainability for the selected frequency model (SHAP) and the Poisson GLM (relativities).
Uses TRAIN/VALIDATION data only. Everything here describes association, not causation."""
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.frequency import PoissonGLM
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("explain")


def main() -> None:
    cfg = load_config()
    proc, figs = ROOT / cfg["paths"]["processed_dir"], ROOT / cfg["paths"]["figures_dir"]
    train = pd.read_parquet(proc / "policies_train.parquet")
    val = pd.read_parquet(proc / "policies_val.parquet")
    freq = joblib.load(ROOT / "models" / "selected_models.joblib")["frequency"]

    # --- SHAP on the LightGBM frequency model (values are in log-rate space) ---
    sample = val.sample(5000, random_state=cfg["seed"])
    X = freq._x(sample)
    sv = shap.TreeExplainer(freq.model_).shap_values(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp["share_of_total"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.to_csv(ROOT / "reports" / "frequency_shap_importance.csv", index=False)
    log.info("SHAP importance (log-rate scale):\n%s", imp.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.iloc[::-1].plot.barh(x="feature", y="mean_abs_shap", ax=ax, legend=False)
    ax.set_xlabel("mean |SHAP| (log claim rate)"); ax.set_title("Frequency model: feature importance (association only)")
    fig.tight_layout(); fig.savefig(figs / "frequency_shap_importance.png", dpi=110); plt.close(fig)
    for f in ("BonusMalus", "DrivAge"):
        i = list(X.columns).index(f)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(X[f], sv[:, i], s=3, alpha=0.4); ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f); ax.set_ylabel("SHAP (log rate)"); ax.set_title(f"SHAP dependence: {f}")
        fig.tight_layout(); fig.savefig(figs / f"frequency_shap_dependence_{f}.png", dpi=110); plt.close(fig)

    # --- Poisson GLM relativities with 95% intervals ---
    glm = PoissonGLM(cfg).fit(train)
    r = glm.result_
    ci = r.conf_int()
    rel = pd.DataFrame({"term": r.params.index, "relativity": np.exp(r.params.values),
                        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values), "p_value": r.pvalues.values})
    rel = rel[rel["term"] != "const"]
    rel.to_csv(ROOT / "reports" / "frequency_glm_relativities.csv", index=False)
    key = rel[rel["term"].str.startswith(("DrivAge_band", "BonusMalus_band", "VehGas", "LogDensity"))]
    log.info("GLM relativities (exp(coef), vs base level; base levels are the dropped categories):\n%s", key.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
