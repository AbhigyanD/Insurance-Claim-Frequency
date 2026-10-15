"""Download and load the freMTPL2 frequency / severity datasets (CASdatasets, GPL >= 2)."""
from __future__ import annotations
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT, load_config

log = logging.getLogger(__name__)


def download(cfg: dict | None = None, force: bool = False) -> dict[str, Path]:
    """Download the .rda files into data/raw. Raises with manual instructions on failure."""
    cfg = cfg or load_config()
    raw = ROOT / cfg["paths"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fname in cfg["data"]["files"].items():
        dest = raw / fname
        if force or not dest.exists():
            url = f"{cfg['data']['base_url']}/{fname}"
            log.info("Downloading %s", url)
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download {url} ({e}). Manually obtain {fname} from "
                    f"https://github.com/dutangc/CASdatasets/tree/master/data and place it in {raw}."
                ) from e
        out[key] = dest
    return out


def _read_rda(path: Path) -> pd.DataFrame:
    import pyreadr
    return next(iter(pyreadr.read_r(str(path)).values()))


def load_raw(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    paths = download(cfg)
    freq, sev = _read_rda(paths["freq"]), _read_rda(paths["sev"])
    id_col = cfg["data"]["id_col"]
    freq[id_col] = freq[id_col].astype("int64")
    sev[id_col] = sev[id_col].astype("int64")
    return freq, sev
