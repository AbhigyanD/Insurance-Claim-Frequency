import numpy as np
import pandas as pd
import pytest

from src.data.validate import DataValidationError
from src.models.pricing import assign_segments, expected_cost_table, segment_edges

DF = pd.DataFrame({"IDpol": [1, 2, 3], "Exposure": [0.5, 1.0, 0.25]})


def test_expected_cost_math_and_units():
    t = expected_cost_table(DF, np.array([0.05, 0.10, 0.025]), np.array([2000.0, 2000.0, 4000.0]))
    assert np.allclose(t["predicted_frequency"], [0.1, 0.1, 0.1])            # per exposure-year
    assert np.allclose(t["predicted_annual_cost"], [200.0, 200.0, 400.0])    # frequency * severity
    assert np.allclose(t["predicted_cost_exposure"], [100.0, 200.0, 100.0])  # counts * severity
    assert (t["predicted_annual_cost"] >= 0).all()


@pytest.mark.parametrize("freq,sev", [
    (np.array([np.nan, .1, .1]), np.ones(3)), (np.array([-.1, .1, .1]), np.ones(3)),
    (np.array([.1, .1, .1]), np.array([0.0, 1, 1])), (np.array([.1, .1, .1]), np.array([np.inf, 1, 1])),
    (np.array([.1, .1]), np.ones(3))])
def test_invalid_model_outputs_raise_not_clip(freq, sev):
    with pytest.raises(ValueError):
        expected_cost_table(DF, freq, sev)


def test_invalid_exposure_raises():
    with pytest.raises(DataValidationError):
        expected_cost_table(DF.assign(Exposure=[0.5, 0.0, 1.0]), np.ones(3), np.ones(3))


def test_segments_use_train_edges_and_labels():
    train = np.arange(100.0)
    edges = segment_edges(train, 5)
    labels = ["a", "b", "c", "d", "e"]
    seg = assign_segments(np.array([-5.0, 10.0, 50.0, 99.0, 1e9]), edges, labels)
    assert list(seg) == ["a", "a", "c", "e", "e"]
    with pytest.raises(ValueError):
        assign_segments(np.array([1.0]), edges, ["a", "b"])
