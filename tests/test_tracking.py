import mlflow

from src.utils.tracking import dataset_fingerprint, log_run


def test_log_run_records_params_metrics_and_tags(tmp_path):
    cfg = {"seed": 7, "mlflow": {"tracking_uri": f"sqlite:////{tmp_path}/m.db"}}
    rid = log_run(cfg, "unit-test", "run1", {"model": "x", "lr": 0.1}, {"val_gini": 0.3, "bad": float("nan")},
                  tags={"selected": "True"})
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    run = mlflow.get_run(rid)
    assert run.data.params["model"] == "x" and run.data.metrics["val_gini"] == 0.3
    assert "bad" not in run.data.metrics                      # NaN metrics are skipped, not logged
    assert run.data.tags["seed"] == "7" and run.data.tags["selected"] == "True"


def test_dataset_fingerprint_changes_with_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    assert dataset_fingerprint(a) != dataset_fingerprint(b) and dataset_fingerprint(a) == dataset_fingerprint(a)
