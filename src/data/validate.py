"""Data-quality checks. Report issues; never silently alter data."""
from __future__ import annotations
import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    pass


def validate_exposure(df: pd.DataFrame, col: str = "Exposure", max_exposure: float = 1.0) -> None:
    """Exposure must be present, finite and strictly positive (it is the Poisson offset)."""
    if col not in df.columns:
        raise DataValidationError(f"Missing exposure column '{col}'")
    e = df[col]
    if e.isna().any():
        raise DataValidationError(f"{int(e.isna().sum())} missing exposure values")
    if not np.isfinite(e.to_numpy(dtype=float)).all():
        raise DataValidationError("Non-finite exposure values")
    if (e <= 0).any():
        raise DataValidationError(f"{int((e <= 0).sum())} policies with exposure <= 0")
    if (e > max_exposure).any():
        raise DataValidationError(f"{int((e > max_exposure).sum())} policies with exposure > {max_exposure}")


def clean_exposure(df: pd.DataFrame, col: str = "Exposure", max_exposure: float = 1.0) -> pd.DataFrame:
    """Cap exposure at max_exposure (values > 1 year are data errors in freMTPL2)."""
    out = df.copy()
    out[col] = out[col].clip(upper=max_exposure)
    return out


def quality_report(freq: pd.DataFrame, sev: pd.DataFrame, id_col: str = "IDpol") -> dict:
    """Summary of integrity issues in the raw data."""
    sev_ids = set(sev[id_col])
    freq_ids = set(freq[id_col])
    claims_per_pol = sev.groupby(id_col).size()
    merged = freq.set_index(id_col).join(claims_per_pol.rename("n_sev"), how="left").fillna({"n_sev": 0})
    return {
        "n_policies": len(freq),
        "n_severity_rows": len(sev),
        "duplicate_policy_ids": int(freq[id_col].duplicated().sum()),
        "missing_values": int(freq.isna().sum().sum()),
        "exposure_gt_1": int((freq["Exposure"] > 1).sum()),
        "sev_ids_not_in_freq": len(sev_ids - freq_ids),
        "policies_with_claims": int((freq["ClaimNb"] > 0).sum()),
        "policies_with_sev_rows": len(sev_ids & freq_ids),
        "claimnb_ne_sev_rows": int((merged["ClaimNb"] != merged["n_sev"]).sum()),
        "claimnb_gt0_without_sev": int(((merged["ClaimNb"] > 0) & (merged["n_sev"] == 0)).sum()),
        "sev_rows_for_zero_claimnb": int(((merged["ClaimNb"] == 0) & (merged["n_sev"] > 0)).sum()),
    }
