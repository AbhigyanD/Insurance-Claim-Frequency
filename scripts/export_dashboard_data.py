"""Write Power BI-ready CSV tables and a data dictionary to reports/dashboard_data (validation split only)."""
import logging

import pandas as pd

from src.reporting.dashboard_export import export_all
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_dashboard_data")


def main() -> None:
    cfg = load_config()
    pred = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / "predictions.parquet")
    out = ROOT / "reports" / "dashboard_data"
    tables = export_all(pred, ROOT / "reports", out)
    for name, df in tables.items():
        log.info("%-32s %8d rows x %2d cols", name, len(df), df.shape[1])
    log.info("Wrote %d tables and data_dictionary.csv to %s", len(tables), out)


if __name__ == "__main__":
    main()
