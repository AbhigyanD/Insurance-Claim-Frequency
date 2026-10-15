import numpy as np
import pandas as pd

from src.features.build import build_policy_table, make_preprocessor, split_policies

CFG = {
    "seed": 42,
    "data": {"id_col": "IDpol", "exposure_col": "Exposure", "claim_count_col": "ClaimNb",
             "claim_amount_col": "ClaimAmount", "max_exposure": 1.0},
    "features": {"categorical": ["VehBrand"], "numeric": ["VehAge"]},
    "split": {"test_size": 0.2, "val_size": 0.2},
}


def _toy(n=500):
    rng = np.random.default_rng(0)
    freq = pd.DataFrame({"IDpol": np.arange(n), "ClaimNb": rng.poisson(0.1, n),
                         "Exposure": rng.uniform(0.1, 1.2, n),
                         "VehAge": rng.integers(0, 20, n), "VehBrand": rng.choice(["B1", "B2"], n)})
    sev = pd.DataFrame({"IDpol": freq.loc[freq.ClaimNb > 0, "IDpol"], "ClaimAmount": 1000.0})
    return freq, sev


def test_policy_table_caps_exposure_and_zero_fills_amount():
    t = build_policy_table(*_toy(), CFG)
    assert t["Exposure"].max() <= 1.0
    assert (t.loc[t.ClaimNb == 0, "TotalClaimAmount"] == 0).all()


def test_split_is_disjoint_complete_and_reproducible():
    t = build_policy_table(*_toy(), CFG)
    a, b = split_policies(t, CFG), split_policies(t, CFG)
    assert sum(len(v) for v in a.values()) == len(t)
    ids = [set(v.IDpol) for v in a.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for k in a:
        assert a[k].IDpol.tolist() == b[k].IDpol.tolist()


def test_preprocessor_fit_on_train_only_handles_unseen_category():
    t = build_policy_table(*_toy(), CFG)
    s = split_policies(t, CFG)
    pre = make_preprocessor(CFG).fit(s["train"])
    unseen = s["test"].head(3).copy()
    unseen["VehBrand"] = "NEW"
    X = pre.transform(unseen)
    assert X.shape[0] == 3 and np.isfinite(X).all()
    # scaler statistics come from train, not test
    assert np.isclose(pre.named_transformers_["num"]["scale"].mean_[0], s["train"]["VehAge"].mean())
