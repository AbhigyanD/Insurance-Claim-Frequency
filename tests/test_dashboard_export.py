import numpy as np
import pandas as pd
import pytest

from src.reporting.dashboard_export import (DATA_DICTIONARY, category_summary, dictionary_frame, distribution_table,
                                            export_all, model_metrics_long, policy_table, validate_tables)


def _pred(n=400, seed=0, split="val"):
    r = np.random.default_rng(seed)
    e = r.uniform(0.2, 1.0, n)
    f = r.uniform(0.03, 0.2, n)
    sev = r.uniform(800, 3000, n)
    return pd.DataFrame({
        "IDpol": np.arange(n), "split": split, "risk_segment": r.choice(["Low", "High"], n), "exposure": e,
        "VehPower": r.integers(4, 12, n), "VehAge": r.integers(0, 20, n), "DrivAge": r.integers(18, 90, n),
        "BonusMalus": r.integers(50, 150, n), "VehBrand": r.choice(["B1", "B2"], n), "VehGas": r.choice(["Diesel", "Regular"], n),
        "Area": r.choice(list("ABC"), n), "Region": r.choice(["R1", "R2"], n), "Density": r.integers(1, 9000, n),
        "predicted_frequency": f, "predicted_severity": sev, "predicted_annual_cost": f * sev,
        "predicted_cost_exposure": f * sev * e, "actual_claim_count": r.poisson(f * e), "actual_claim_amount": r.poisson(f * e) * 1500.0})


def test_category_summary_matches_manual_aggregation_and_flags_small_samples():
    p = _pred()
    t = category_summary(p)
    row = t[(t["dimension"] == "Area") & (t["level"] == "A")].iloc[0]
    a = p[p["Area"] == "A"]
    assert row["policies"] == len(a) and row["exposure"] == pytest.approx(a["exposure"].sum())
    assert row["actual_cost"] == pytest.approx(a["actual_claim_amount"].sum())
    assert row["expected_claims"] == pytest.approx((a["predicted_frequency"] * a["exposure"]).sum())
    assert row["cost_actual_to_expected"] == pytest.approx(a["actual_claim_amount"].sum() / a["predicted_cost_exposure"].sum())
    assert t["small_sample"].all()                     # 400 toy policies are far below 1,000 exposure-years
    assert set(t["dimension"]) >= {"Risk segment", "Driver age", "Vehicle age", "Area", "Region", "Brand", "Fuel"}
    # every dimension partitions all policies
    assert (t.groupby("dimension")["policies"].sum() == len(p)).all()


def test_policy_table_keeps_actuals_and_predictions_in_separate_named_columns():
    t = policy_table(_pred())
    actual = [c for c in t.columns if c.startswith("actual_")]
    pred = [c for c in t.columns if c.startswith("predicted_")]
    assert actual == ["actual_claim_count", "actual_claim_amount"] and len(pred) == 4
    assert "policy_id" in t.columns and "IDpol" not in t.columns


def test_model_metrics_long_format_and_validation_only():
    f = pd.DataFrame({"model": ["m", "m"], "split": ["train", "validation"], "poisson_deviance": [0.24, 0.25], "mae_counts": [.07, .08],
                      "actual_to_predicted": [1.0, .99], "gini": [.3, .29]})
    s = pd.DataFrame({"model": ["c"], "split": ["validation"], "gamma_deviance": [1.6], "mae": [2000.0], "rmse": [9000.0], "actual_to_predicted": [.95]})
    t = model_metrics_long(f, s)
    assert len(t) == 8 and set(t["target"]) == {"frequency", "severity"}
    assert t.query("model == 'm' and metric == 'gini'")["value"].iloc[0] == pytest.approx(0.29)   # validation row, not train


def test_distribution_table_counts_all_policies():
    p = _pred()
    d = distribution_table(p, bins=10)
    assert d["policies"].sum() == len(p) and (d["bin_to"] > d["bin_from"]).all()


def test_dictionary_documents_every_exported_column_and_flags_undocumented_ones():
    p = _pred()
    tables = {"policy_predictions_validation": policy_table(p), "category_summary": category_summary(p),
              "predicted_cost_distribution": distribution_table(p)}
    validate_tables(tables)                                        # passes
    bad = {"category_summary": category_summary(p).assign(surprise_column=1)}
    with pytest.raises(ValueError, match="undocumented"):
        validate_tables(bad)
    with pytest.raises(ValueError, match="no data dictionary"):
        validate_tables({"unknown_table": pd.DataFrame({"a": [1]})})
    d = dictionary_frame()
    assert d["description"].str.len().min() > 10 and set(d["table"]) == set(DATA_DICTIONARY)


def test_validation_rejects_test_rows_and_missing_values():
    p = _pred()
    with_split = {"category_summary": category_summary(p).assign(split="test")}
    DATA_DICTIONARY["category_summary"]["split"] = "test-only column used by this unit test"
    try:
        with pytest.raises(ValueError, match="non-validation"):
            validate_tables(with_split)
    finally:
        del DATA_DICTIONARY["category_summary"]["split"]
    tbl = category_summary(p)
    tbl.loc[0, "actual_cost"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_tables({"category_summary": tbl})


def test_export_all_uses_only_validation_rows_and_writes_files(tmp_path):
    p = pd.concat([_pred(300, 0, "val"), _pred(300, 1, "test").assign(IDpol=lambda d: d["IDpol"] + 10_000),
                   _pred(300, 2, "train").assign(IDpol=lambda d: d["IDpol"] + 20_000)], ignore_index=True)
    rep = tmp_path / "reports"
    rep.mkdir()
    pd.DataFrame({"model": ["m"] * 2, "split": ["train", "validation"], "poisson_deviance": [.2, .21], "mae_counts": [.07, .07],
                  "actual_to_predicted": [1, 1], "gini": [.3, .3]}).to_csv(rep / "frequency_model_comparison.csv", index=False)
    pd.DataFrame({"model": ["c"], "split": ["validation"], "gamma_deviance": [1.6], "mae": [1.0], "rmse": [2.0],
                  "actual_to_predicted": [.9]}).to_csv(rep / "severity_model_comparison.csv", index=False)
    pd.DataFrame({"feature": ["x"], "psi": [0.0], "status": ["stable"], "kind": ["feature"]}).to_csv(rep / "drift_psi_train_vs_val.csv", index=False)
    tables = export_all(p, rep, tmp_path / "out")
    assert len(tables["policy_predictions_validation"]) == 300                          # only the validation rows
    assert set(tables["policy_predictions_validation"]["policy_id"]) == set(range(300))  # no test/train ids
    assert (tmp_path / "out" / "data_dictionary.csv").exists()
    assert {f.name for f in (tmp_path / "out").glob("*.csv")} >= {f"{n}.csv" for n in tables}
    with pytest.raises(ValueError, match="No validation"):
        export_all(p[p["split"] != "val"], rep, tmp_path / "out2")
