import json

import pytest

from src.evaluation.holdout_guard import HoldoutAlreadyUsed, check_unused, record_use


def test_first_use_allowed_then_blocked_unless_forced(tmp_path):
    marker = tmp_path / "reports" / "marker.json"
    check_unused(marker)                                            # nothing recorded yet: fine
    record_use(marker, {"selected_frequency": "m", "test_policies": 10})
    info = json.loads(marker.read_text())
    assert info["selected_frequency"] == "m" and "evaluated_at" in info
    with pytest.raises(HoldoutAlreadyUsed, match="already evaluated"):
        check_unused(marker)                                        # second run is refused
    check_unused(marker, force=True)                                # explicit override is possible


def test_error_message_explains_why(tmp_path):
    marker = tmp_path / "m.json"
    record_use(marker, {})
    with pytest.raises(HoldoutAlreadyUsed, match="optimistic"):
        check_unused(marker)
