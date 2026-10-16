"""Thin MLflow helper. Local SQLite tracking store; no credentials or remote servers involved."""
from __future__ import annotations
import hashlib
from pathlib import Path

import mlflow

from src.utils.config import ROOT


def _setup(cfg: dict, experiment: str) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        path = ROOT / uri.removeprefix("sqlite:///")
        path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def dataset_fingerprint(path: Path) -> str:
    """Short content hash of a file, to tie a run to the exact data it used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def log_run(cfg: dict, experiment: str, run_name: str, params: dict, metrics: dict,
            artifacts: list[Path] | None = None, tags: dict | None = None) -> str:
    """Log one model run and return its run id."""
    _setup(cfg, experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        mlflow.set_tags({"seed": str(cfg["seed"]), **(tags or {})})
        for a in artifacts or []:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
        return run.info.run_id
