"""EDA on the TRAIN split only (so no test information informs modelling choices)."""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eda")


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"),
                                           claims=("ClaimNb", "sum"), amount=("TotalClaimAmount", "sum"))
    g["frequency"] = g["claims"] / g["exposure"]
    g["pure_premium"] = g["amount"] / g["exposure"]
    return g


def main() -> None:
    cfg = load_config()
    proc, figs = ROOT / cfg["paths"]["processed_dir"], ROOT / cfg["paths"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(proc / "policies_train.parquet")
    val = pd.read_parquet(proc / "policies_val.parquet")
    test = pd.read_parquet(proc / "policies_test.parquet")
    sev = pd.read_parquet(proc / "severity_claims.parquet")
    sev = sev[sev["IDpol"].isin(train["IDpol"])]

    lines = []
    mean, var = train["ClaimNb"].mean(), train["ClaimNb"].var()
    lines.append(f"train policies: {len(train)}; zero-claim share: {(train.ClaimNb == 0).mean():.4f}")
    lines.append(f"claim count mean={mean:.4f} var={var:.4f} var/mean={var / mean:.3f} (>1 suggests overdispersion; "
                 "note this ignores exposure and covariates, so is only indicative)")
    lines.append(f"overall frequency per exposure-year: {train.ClaimNb.sum() / train.Exposure.sum():.4f}")
    q = sev["ClaimAmount"].quantile([.5, .9, .99, .999, 1.0])
    lines.append(f"severity rows={len(sev)} mean={sev.ClaimAmount.mean():.0f} median={q[.5]:.0f} "
                 f"p99={q[.99]:.0f} p99.9={q[.999]:.0f} max={q[1.0]:.0f} skew={sev.ClaimAmount.skew():.1f}")
    top = sev["ClaimAmount"].nlargest(max(1, int(0.001 * len(sev)))).sum() / sev["ClaimAmount"].sum()
    lines.append(f"top 0.1% of claims = {top:.1%} of total claim amount")
    # Distribution shift check, train vs val vs test
    for n, d in [("val", val), ("test", test)]:
        lines.append(f"{n}: freq={d.ClaimNb.sum() / d.Exposure.sum():.4f} mean exposure={d.Exposure.mean():.3f} "
                     f"mean DrivAge={d.DrivAge.mean():.2f} mean BonusMalus={d.BonusMalus.mean():.2f}")
    (ROOT / "reports" / "eda_summary.txt").write_text("\n".join(lines) + "\n")
    for ln in lines:
        log.info(ln)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    train["Exposure"].hist(bins=50, ax=ax[0])
    ax[0].set_title("Exposure (years)")
    train["ClaimNb"].value_counts().sort_index().plot.bar(ax=ax[1], logy=True)
    ax[1].set_title("Claim count (log scale)")
    np.log10(sev["ClaimAmount"]).hist(bins=60, ax=ax[2])
    ax[2].set_title("log10 claim amount")
    fig.tight_layout()
    fig.savefig(figs / "eda_distributions.png", dpi=110)
    plt.close(fig)

    tmp = train.assign(DrivAgeBand=pd.cut(train.DrivAge, [17, 25, 35, 45, 55, 65, 100]),
                       VehAgeBand=pd.cut(train.VehAge, [-1, 2, 5, 10, 15, 100]),
                       BonusMalusBand=pd.cut(train.BonusMalus, [49, 50, 60, 80, 100, 150, 400]))
    cols = ["DrivAgeBand", "VehAgeBand", "BonusMalusBand", "VehBrand", "VehGas", "Area", "Region"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.ravel()
    for a, c in zip(axes, cols):
        r = rate_by(tmp, c)
        r["frequency"].plot.bar(ax=a)
        a.set_title(f"Frequency by {c}")
        a.set_xlabel("")
        a.axhline(train.ClaimNb.sum() / train.Exposure.sum(), color="r", ls="--", lw=1)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(figs / "eda_frequency_by_feature.png", dpi=110)
    plt.close(fig)
    log.info("Figures written to %s", figs)


if __name__ == "__main__":
    main()
