"""Download freMTPL2 data, validate, build the policy table, split, and cache to data/processed."""
import json
import logging

from src.data.load import load_raw
from src.data.validate import quality_report
from src.features.build import build_policy_table, split_policies
from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_data")


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    freq, sev = load_raw(cfg)
    report = quality_report(freq, sev, cfg["data"]["id_col"])
    log.info("Quality report: %s", report)
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    sev.to_parquet(out / "severity_claims.parquet", index=False)
    table = build_policy_table(freq, sev, cfg)
    for name, part in split_policies(table, cfg).items():
        part.to_parquet(out / f"policies_{name}.parquet", index=False)
        log.info("%s: %d policies, claim rate %.4f", name, len(part), part["ClaimNb"].sum() / part["Exposure"].sum())


if __name__ == "__main__":
    main()
