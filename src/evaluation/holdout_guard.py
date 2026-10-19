"""Protect the held-out test set: it should be evaluated once, after model selection is final."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


class HoldoutAlreadyUsed(RuntimeError):
    pass


def check_unused(marker: Path, force: bool = False) -> None:
    """Raise if the test set was already evaluated. Re-evaluating after seeing results invalidates the holdout."""
    if Path(marker).exists() and not force:
        info = json.loads(Path(marker).read_text())
        raise HoldoutAlreadyUsed(
            f"The test set was already evaluated on {info.get('evaluated_at')}. Repeating the evaluation and then adjusting "
            "models would make test results optimistic. Pass --force only if you accept that and will say so.")


def record_use(marker: Path, info: dict) -> None:
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}, indent=2))
