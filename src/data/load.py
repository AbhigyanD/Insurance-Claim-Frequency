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
    """Read the single data frame in an .rda file.

    Uses the pure-Python `rdata` package: pyreadr was found to be impractically slow
    (>15 min) on the 678k-row freMTPL2freq file.
    """
    import rdata
    df = next(iter(rdata.conversion.convert(rdata.parser.parse_file(str(path))).values()))
    df.columns = [str(c) for c in df.columns]
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = s.astype(str)
        elif pd.api.types.is_integer_dtype(s):
            out[c] = s.astype("int64")
        else:
            out[c] = s.astype("float64")
    return out.reset_index(drop=True)


def load_raw(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    paths = download(cfg)
    freq, sev = _read_rda(paths["freq"]), _read_rda(paths["sev"])
    id_col = cfg["data"]["id_col"]
    # R stores IDpol as a factor in the sev file with labels such as "1e+05" (= 100000),
    # so parse via numeric rather than int().
    for d in (freq, sev):
        ids = pd.to_numeric(d[id_col])
        if not (ids == ids.round()).all():
            raise ValueError("Non-integer policy ids after parsing")
        d[id_col] = ids.astype("int64")
    return freq, sev
