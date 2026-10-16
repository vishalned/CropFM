"""Sentinel-2 L2A point time series from Microsoft Planetary Computer (COGs, West Europe).

Replaces GEE `COPERNICUS/S2_SR_HARMONIZED` (scene cloud filter < cloud_cover_threshold).
Bands match GEE integer DN scaling (reflectance*10000, processing-baseline >= 04.00
offset of 1000 removed like the *_HARMONIZED collection). MSK_CLDPRB and QA60 do not
exist on PC and are filled with NaN (the model does not use them).

usage: python s2_pc.py --year 2021 --points SUBSET.json --out DIR [--task-id I --n-tasks N]
"""
import argparse

import numpy as np

from cropfm.data.fast.cogsample import sample_item
from cropfm.data.fast.common import S2_VARS
from cropfm.data.fast.stac_cells import add_args, run, sign

ASSET = {"B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05", "B6": "B06",
         "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09", "B11": "B11", "B12": "B12",
         "SCL": "SCL"}
REFL = [v for v in S2_VARS if v not in ("SCL", "MSK_CLDPRB", "QA60")]


def dedupe(items):
    """PC can hold the same acquisition twice (original + reprocessed baseline): keep the newest."""
    best = {}
    for it in items:
        key = (it.properties.get("s2:mgrs_tile"), it.datetime.strftime("%Y-%m-%dT%H:%M"))
        if key not in best or it.properties.get("s2:processing_baseline", "") > \
                best[key].properties.get("s2:processing_baseline", ""):
            best[key] = it
    return list(best.values())


def scene(item, lon, lat):
    hrefs = {k: item.assets[a].href for k, a in ASSET.items()}
    got = sample_item(hrefs, lon, lat, sign=sign)
    out = np.full((len(lon), len(S2_VARS)), np.nan, np.float32)
    old = float(item.properties.get("s2:processing_baseline", "0")) < 4.0
    for k, v in got.items():
        if k in REFL and not old:
            # harmonise: remove the +1000 offset, keep 0 = nodata, floor at 1
            v = np.where(v > 0, np.maximum(v - 1000, 1), v)
        out[:, S2_VARS.index(k)] = v
    nodata = np.nan_to_num(out[:, [S2_VARS.index(b) for b in REFL]], nan=0).max(1) == 0
    out[nodata] = np.nan  # outside the swath: no row (GEE masks these)
    return out


if __name__ == "__main__":
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--cloud", type=float, default=50.0)
    a = ap.parse_args()
    run(a, "sentinel2", "sentinel-2-l2a", {"eo:cloud_cover": {"lt": a.cloud}},
        dedupe, scene, S2_VARS)
