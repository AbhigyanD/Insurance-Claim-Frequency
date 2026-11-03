"""Loader tests. These cover src/data/load.py without needing the real dataset or network."""
import numpy as np
import pandas as pd
import pytest

from src.data import load as L


def test_coerce_columns_maps_factors_ints_and_floats():
    df = pd.DataFrame({
        "Region": pd.Categorical(["Alsace", "Corse", "Alsace"]),
        "VehAge": pd.array([1, 2, 3], dtype="Int32"),          # R integers arrive as nullable Int32
        "Exposure": [0.5, 1.0, 0.25],
        "IDpol": pd.Categorical(["1", "3", "5"]),               # a factor of ids stays a string here
    })
    out = L.coerce_columns(df)
    assert out["Region"].tolist() == ["Alsace", "Corse", "Alsace"]
    assert out["VehAge"].dtype == "int64" and out["Exposure"].dtype == "float64"
    assert out["IDpol"].tolist() == ["1", "3", "5"]
    assert not any(isinstance(t, pd.CategoricalDtype) for t in out.dtypes)
    assert list(out.index) == [0, 1, 2]


def test_normalize_ids_handles_r_scientific_notation_regression():
    # In the real severity file R stored policy id 100000 as the factor label "1e+05"; int("1e+05") raises.
    df = pd.DataFrame({"IDpol": pd.Categorical(["1e+05", "12", "2e+05", "6114330"]), "ClaimAmount": [1.0, 2.0, 3.0, 4.0]})
    out = L.normalize_ids(L.coerce_columns(df), "IDpol")
    assert out["IDpol"].tolist() == [100000, 12, 200000, 6114330] and out["IDpol"].dtype == "int64"
    assert df["IDpol"].tolist()[0] == "1e+05"                   # input left untouched


@pytest.mark.parametrize("bad", [["1.5", "2"], ["abc", "2"]])
def test_normalize_ids_rejects_fractional_or_non_numeric_ids(bad):
    with pytest.raises(ValueError):
        L.normalize_ids(pd.DataFrame({"IDpol": bad}), "IDpol")


def _cfg(tmp_path):
    return {"paths": {"raw_dir": str(tmp_path / "raw")}, "data": {"base_url": "https://example.invalid/data", "id_col": "IDpol",
                                                                  "files": {"freq": "f.rda", "sev": "s.rda"}}}


def test_download_fetches_missing_files_skips_existing_and_forces(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(L.urllib.request, "urlretrieve", lambda url, dest: (calls.append(url), open(dest, "wb").write(b"x")))
    cfg = _cfg(tmp_path)
    paths = L.download(cfg)
    assert sorted(calls) == ["https://example.invalid/data/f.rda", "https://example.invalid/data/s.rda"]
    assert all(p.exists() for p in paths.values()) and set(paths) == {"freq", "sev"}
    calls.clear()
    L.download(cfg)
    assert calls == []                                          # existing files are not downloaded again
    L.download(cfg, force=True)
    assert len(calls) == 2


def test_download_failure_gives_actionable_manual_instructions(tmp_path, monkeypatch):
    def boom(url, dest):
        raise OSError("blocked by proxy")
    monkeypatch.setattr(L.urllib.request, "urlretrieve", boom)
    with pytest.raises(RuntimeError) as e:
        L.download(_cfg(tmp_path))
    msg = str(e.value)
    assert "blocked by proxy" in msg and "f.rda" in msg and "github.com/dutangc/CASdatasets" in msg and str(tmp_path / "raw") in msg


def test_read_rda_takes_first_frame_and_coerces(monkeypatch, tmp_path):
    import rdata
    frame = pd.DataFrame({"IDpol": pd.Categorical(["1e+05"]), "ClaimAmount": [10.0]})
    monkeypatch.setattr(rdata.parser, "parse_file", lambda p: ("parsed", p))
    monkeypatch.setattr(rdata.conversion, "convert", lambda parsed: {"freMTPL2sev": frame, "ignored": pd.DataFrame()})
    out = L._read_rda(tmp_path / "x.rda")
    assert out["IDpol"].tolist() == ["1e+05"] and out["ClaimAmount"].dtype == "float64"


def test_load_raw_returns_int_ids_that_join_across_files(monkeypatch, tmp_path):
    freq = pd.DataFrame({"IDpol": [100000.0, 5.0], "ClaimNb": [1.0, 0.0], "Exposure": [1.0, 0.5]})
    sev = pd.DataFrame({"IDpol": ["1e+05"], "ClaimAmount": [900.0]})
    monkeypatch.setattr(L, "download", lambda cfg: {"freq": "F", "sev": "S"})
    monkeypatch.setattr(L, "_read_rda", lambda path: {"F": freq, "S": sev}[path].copy())
    f, s = L.load_raw(_cfg(tmp_path))
    assert f["IDpol"].dtype == s["IDpol"].dtype == np.dtype("int64")
    assert set(s["IDpol"]) <= set(f["IDpol"])                   # the severity claim now joins to its policy
