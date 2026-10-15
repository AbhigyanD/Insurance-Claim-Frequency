import numpy as np
import pandas as pd
import pytest

from src.data.validate import DataValidationError, clean_exposure, validate_exposure


def _df(exposure):
    return pd.DataFrame({"Exposure": exposure})


@pytest.mark.parametrize("bad", [[0.5, np.nan], [0.5, 0.0], [0.5, -0.1], [0.5, np.inf], [0.5, 1.5]])
def test_invalid_exposure_rejected(bad):
    with pytest.raises(DataValidationError):
        validate_exposure(_df(bad))


def test_missing_exposure_column():
    with pytest.raises(DataValidationError):
        validate_exposure(pd.DataFrame({"x": [1]}))


def test_valid_exposure_passes_and_cap_works():
    validate_exposure(_df([0.1, 1.0]))
    capped = clean_exposure(_df([0.5, 2.0]))
    assert capped["Exposure"].max() == 1.0
    validate_exposure(capped)
