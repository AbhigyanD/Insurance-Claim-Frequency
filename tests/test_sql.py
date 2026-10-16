import duckdb
import pandas as pd
import pytest

from src.utils.config import ROOT


def _con_with_toy(tmp_path):
    pred = pd.DataFrame({
        "IDpol": [1, 2, 3, 4], "split": ["val"] * 4, "risk_segment": ["Low", "Low", "High", "High"],
        "exposure": [1.0, 1.0, 0.5, 0.5], "actual_claim_count": [0, 1, 1, 0], "actual_claim_amount": [0.0, 1000.0, 3000.0, 0.0],
        "predicted_frequency": [0.1, 0.1, 0.4, 0.4], "predicted_cost_exposure": [100.0, 100.0, 300.0, 300.0]})
    pred.to_parquet(tmp_path / "predictions.parquet")
    pd.DataFrame({"model": ["m"], "split": ["validation"], "poisson_deviance": [0.2], "mae_counts": [0.07],
                  "actual_to_predicted": [1.0], "gini": [0.3]}).to_csv(tmp_path / "freq.csv", index=False)
    pd.DataFrame({"model": ["m"], "split": ["validation"], "gamma_deviance": [1.6], "mae": [2000.0], "rmse": [9000.0],
                  "actual_to_predicted": [0.95]}).to_csv(tmp_path / "sev.csv", index=False)
    con = duckdb.connect()
    con.execute(f"CREATE TABLE predictions AS SELECT * FROM read_parquet('{tmp_path}/predictions.parquet')")
    con.execute(f"CREATE TABLE model_eval_frequency AS SELECT * FROM read_csv_auto('{tmp_path}/freq.csv')")
    con.execute(f"CREATE TABLE model_eval_severity AS SELECT * FROM read_csv_auto('{tmp_path}/sev.csv')")
    con.execute("""CREATE TABLE policies (split VARCHAR, IDpol BIGINT, Exposure DOUBLE, VehPower INT, VehAge INT, DrivAge INT,
        BonusMalus INT, VehBrand VARCHAR, VehGas VARCHAR, Area VARCHAR, Region VARCHAR, Density INT,
        ClaimNb DOUBLE, TotalClaimAmount DOUBLE)""")
    con.execute("INSERT INTO policies VALUES ('val', 1, 1.0, 5, 3, 40, 50, 'B1', 'Diesel', 'A', 'R', 100, 0, 0)")
    # run the views exactly as written in schema.sql (everything from the first view onward)
    schema = (ROOT / "sql" / "schema.sql").read_text()
    con.execute(schema[schema.index("-- Policy-level features and targets"):])
    return con


def test_segment_view_actual_vs_expected(tmp_path):
    con = _con_with_toy(tmp_path)
    r = con.execute("SELECT * FROM v_segment_performance WHERE risk_segment='High'").df().iloc[0]
    assert r["policies"] == 2 and r["actual_cost"] == 3000 and r["expected_cost"] == 600
    assert r["cost_actual_to_expected"] == pytest.approx(5.0)
    assert r["frequency_actual_to_expected"] == pytest.approx(1 / 0.4)   # 1 claim / (0.4*0.5*2)


def test_model_evaluation_view_is_long_format(tmp_path):
    con = _con_with_toy(tmp_path)
    df = con.execute("SELECT * FROM v_model_evaluation").df()
    assert set(df["target"]) == {"frequency", "severity"}
    assert df.query("target=='frequency' and metric=='gini'")["value"].iloc[0] == pytest.approx(0.3)
    assert len(df) == 8
