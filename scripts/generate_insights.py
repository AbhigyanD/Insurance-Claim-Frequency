"""Render reports/business_insights.md from the result tables."""
import logging

import pandas as pd

from src.reporting.insights import render_insights
from src.utils.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_insights")


def main() -> None:
    r = ROOT / "reports"
    tables = {
        "test_results": pd.read_csv(r / "test_results.csv"),
        "test_segments": pd.read_csv(r / "test_segment_performance.csv"),
        "val_segment_ae": pd.read_csv(r / "segment_ae_uncertainty_validation.csv"),
        "lift": pd.read_csv(r / "frequency_lift_validation_lightgbm_poisson.csv"),
        "shap": pd.read_csv(r / "frequency_shap_importance.csv"),
        "relativities": pd.read_csv(r / "frequency_glm_relativities.csv"),
        "severity_bootstrap": pd.read_csv(r / "severity_bootstrap_vs_constant.csv"),
        "large_loss": pd.read_csv(r / "large_loss_analysis_validation.csv"),
        "val_freq": pd.read_csv(r / "frequency_model_comparison.csv"),
    }
    out = r / "business_insights.md"
    out.write_text(render_insights(tables))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
