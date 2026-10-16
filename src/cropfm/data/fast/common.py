"""Shared helpers for the fast (non-GEE) download pipeline.

Every modality extractor produces an *observation table* for a set of points:
    idx  (int32, n_obs)        index into the point list
    day  (int32, n_obs)        days since 1970-01-01
    vals (float32, n_obs, C)   band values (NaN = missing)
saved as a compressed npz. `assemble.py` turns these tables into the same raw
zarr layout that `utils/save2zarr.py` produced, so `clean_data.py` keeps working.
"""
import datetime as dt
import json
import os

import numpy as np

REF_ZARR = "/projects/prjs1788/CropFM/global_data_1200_merged_cleaned_rechunked.zarr"

AGERA5_VARS = [
    "Precipitation_Flux",
    "Vapour_Pressure_Mean_24h",
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Min_24h",
    "Temperature_Air_2m_Mean_24h",
    "Solar_Radiation_Flux",
    "Wind_Speed_10m_Mean_24h",
    "Derived_Relative_Humidity_2m_Max_24h",
]
FAPAR_VARS = ["FAPAR", "LAI"]
S2_VARS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
           "SCL", "MSK_CLDPRB", "QA60"]
S1_VARS = ["VV", "VH", "angle"]

_EPOCH = dt.date(1970, 1, 1)


def to_day(date) -> int:
    if isinstance(date, str):
        date = dt.date.fromisoformat(date[:10])
    return (date - _EPOCH).days


def from_day(day: int) -> str:
    return (_EPOCH + dt.timedelta(days=int(day))).isoformat()


def load_points(subset_file: str | None = None, ref_zarr: str = REF_ZARR):
    """Return (global_idx, lon, lat) for the points to process.

    Coordinates come from the reference zarr's metadata/coordinates (lon, lat).
    `subset_file` is a json {"indices": [...]}; None = all points.
    """
    import zarr

    root = zarr.open_group(ref_zarr, mode="r")
    coords = root["metadata/coordinates"][:]
    if subset_file:
        with open(subset_file) as f:
            idx = np.asarray(json.load(f)["indices"], dtype=np.int64)
    else:
        idx = np.arange(len(coords), dtype=np.int64)
    return idx, coords[idx, 0], coords[idx, 1]


PROD_ROOT = "/projects/prjs1788/CropFM/fast_dl/raw"


def mod_dir(out, year, mod):
    """<out>/<year>/<modality>/ - one directory per (year, modality)."""
    d = os.path.join(out, str(year), mod)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def log_failure(path, msg):
    """Append one line to a shared failure log (small O_APPEND writes are atomic enough)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(msg.replace("\n", " ")[:2000] + "\n")


def nearest_index(grid, values):
    """Index of the nearest grid coordinate (grid monotonic, ascending or descending)."""
    g = np.asarray(grid, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if g[0] > g[-1]:
        g, v = -g, -v
    i = np.clip(np.searchsorted(g, v), 1, len(g) - 1)
    return np.where(np.abs(v - g[i - 1]) <= np.abs(g[i] - v), i - 1, i)


def save_obs(path, idx, day, vals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, idx=np.asarray(idx, np.int32), day=np.asarray(day, np.int32),
                        vals=np.asarray(vals, np.float32))
    os.replace(tmp, path)


def load_obs(path):
    z = np.load(path)
    return z["idx"], z["day"], z["vals"]
