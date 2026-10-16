"""Sentinel-1 IW VV+VH point time series from Planetary Computer's RTC collection.

Replaces GEE `COPERNICUS/S1_GRD` (dB, sigma0). PC RTC is radiometrically terrain
corrected gamma0 in linear power; we convert to dB. Values are therefore NOT identical
to GEE (typically a ~1-3 dB offset depending on incidence angle) - compare.py reports
the offset. The `angle` band is not provided by RTC and is NaN (model uses only VV/VH).

usage: python s1_pc.py --year 2021 --points SUBSET.json --out DIR [--task-id I --n-tasks N]
"""
import argparse

import numpy as np

from cropfm.data.fast.cogsample import sample_item
from cropfm.data.fast.common import S1_VARS
from cropfm.data.fast.stac_cells import add_args, run, sign

NODATA = -32768.0


def is_iw_dv(item):
    p = item.properties
    pol = [x.upper() for x in p.get("sar:polarizations", [])]
    return p.get("sar:instrument_mode", "IW") == "IW" and "VV" in pol and "VH" in pol


def scene(item, lon, lat):
    got = sample_item({"VV": item.assets["vv"].href, "VH": item.assets["vh"].href},
                      lon, lat, sign=sign)
    out = np.full((len(lon), len(S1_VARS)), np.nan, np.float32)
    for k in ("VV", "VH"):
        v = got[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(v)
        db[(v == NODATA) | (v <= 0)] = np.nan
        out[:, S1_VARS.index(k)] = db
    out[np.isnan(out[:, :2]).any(axis=1)] = np.nan  # need both polarisations
    return out


if __name__ == "__main__":
    ap = add_args(argparse.ArgumentParser())
    a = ap.parse_args()
    run(a, "sentinel1", "sentinel-1-rtc", None, lambda items: [i for i in items if is_iw_dv(i)],
        scene, S1_VARS)
