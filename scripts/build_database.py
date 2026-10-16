"""Build the DuckDB analytical database from processed data and report CSVs; optionally run a SQL file."""
import argparse
import logging

import duckdb

from src.utils.config import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_database")


def build(con: duckdb.DuckDBPyConnection, processed: str, reports: str) -> None:
    sql = (ROOT / "sql" / "schema.sql").read_text().replace("${PROCESSED}", processed).replace("${REPORTS}", reports)
    con.execute(sql)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="SQL file to run after building (prints each result)")
    a = ap.parse_args()
    cfg = load_config()
    db = ROOT / cfg["paths"]["processed_dir"] / "claims.duckdb"
    con = duckdb.connect(str(db))
    build(con, str(ROOT / cfg["paths"]["processed_dir"]), str(ROOT / "reports"))
    log.info("Built %s with tables: %s", db, [r[0] for r in con.execute("SHOW TABLES").fetchall()])
    if a.query:
        text = "\n".join(l for l in (ROOT / a.query).read_text().splitlines() if not l.strip().startswith("--"))
        for body in (st.strip() for st in text.split(";")):
            if body:
                print(con.execute(body).df().to_string(index=False), "\n")


if __name__ == "__main__":
    main()
