"""Policy-level modelling table, leakage-safe split and preprocessing pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import clean_exposure, validate_exposure


def build_policy_table(freq: pd.DataFrame, sev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per policy: risk features, exposure, and *outcomes* (ClaimNb, ClaimAmount total).

    Outcomes are kept in clearly named columns so they can be dropped from model inputs.
    """
    d = cfg["data"]
    df = clean_exposure(freq, d["exposure_col"], d["max_exposure"])
    validate_exposure(df, d["exposure_col"], d["max_exposure"])
    total = sev.groupby(d["id_col"])[d["claim_amount_col"]].sum().rename("TotalClaimAmount")
    df = df.merge(total, on=d["id_col"], how="left")
    df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0.0)
    return df


def split_policies(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Reproducible train/val/test split by policy, stratified on 'has a claim'.

    Each policy appears in exactly one split, so there is no policy-level leakage.
    The test set is only for final reporting.
    """
    s, seed, id_col = cfg["split"], cfg["seed"], cfg["data"]["id_col"]
    strat = (df[cfg["data"]["claim_count_col"]] > 0).astype(int)
    trainval, test = train_test_split(df, test_size=s["test_size"], random_state=seed, stratify=strat)
    strat_tv = (trainval[cfg["data"]["claim_count_col"]] > 0).astype(int)
    rel_val = s["val_size"] / (1 - s["test_size"])
    train, val = train_test_split(trainval, test_size=rel_val, random_state=seed, stratify=strat_tv)
    out = {"train": train, "val": val, "test": test}
    ids = [set(v[id_col]) for v in out.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "policy appears in >1 split"
    return out


def make_preprocessor(cfg: dict) -> ColumnTransformer:
    """Fit on train only. Numeric: standardise. Categorical: one-hot (unseen -> all zeros)."""
    f = cfg["features"]
    return ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler())]), f["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), f["categorical"]),
    ])
