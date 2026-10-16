"""FAPAR / LAI from the NOAA CDR VIIRS archive on AWS open data (anonymous S3 over HTTPS).

Replaces GEE `NOAA/CDR/VIIRS/LAI_FAPAR/V1`. One ~6-9 MB netCDF per day on a 0.05 deg
grid. Each file is downloaded to node-local temp space, sampled at all points by index
lookup, and deleted immediately - nothing but the sampled values is kept.

Output (resumable, one file per month): <out>/<year>/fapar/fapar_<YYYYMM>.npz (+ .json QC)
  idx = position in the points list, day = days since 1970, vals = (n, 2) raw integer
  values [FAPAR, LAI] (scale 0.001, as in the GEE sample() output).

No-data handling: _FillValue (-100) and values outside the product's valid_range are set
to NaN; rows where *all* bands are NaN are dropped (= pixels GEE masked, no row). The QC
json counts fill / out-of-range values and lists days missing from the archive.

usage: python fapar_noaa.py --year 2021 --points SUBSET.json --out DIR [--months 1 2] [--workers 16]
"""
import argparse
import os
import re
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import netCDF4
import numpy as np
import requests

from cropfm.data.fast.common import (FAPAR_VARS, load_points, log_failure, mod_dir, save_obs,
                                     to_day, write_json)

BUCKET = "https://noaa-cdr-leaf-area-index-fapar-pds.s3.amazonaws.com"
FILL = -100
VALID_MAX = {"FAPAR": 1000, "LAI": 7000}  # valid_range upper bounds from the netCDF attrs
_HDF5_LOCK = threading.Lock()  # netCDF4/HDF5 is not thread-safe; downloads stay parallel


def list_year(year: int) -> dict[str, str]:
    """{YYYYMMDD: key} for one year (one file per day; if reprocessed, newest wins)."""
    out, token = {}, None
    while True:
        params = {"list-type": "2", "prefix": f"data/{year}/"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        root = ET.fromstring(r.content)
        for c in root.findall("s:Contents", ns):
            key = c.find("s:Key", ns).text
            m = re.search(r"_(\d{8})_c\d+\.nc$", key)
            if m:
                out[m.group(1)] = max(key, out.get(m.group(1), ""))
        if root.find("s:IsTruncated", ns).text == "true":
            token = root.find("s:NextContinuationToken", ns).text
        else:
            return out


def grid_index(lon, lat):
    """Pixel indices on the 0.05 deg grid (lat 89.975 -> -89.975, lon -179.975 -> 179.975)."""
    row = np.clip(np.floor((90.0 - lat) / 0.05).astype(int), 0, 3599)
    col = np.clip(np.floor((lon + 180.0) / 0.05).astype(int), 0, 7199)
    return row, col


def process_day(key, row, col, tmpdir):
    """-> (vals (n, 2) float32 with NaN no-data, n_fill, n_out_of_range)"""
    url = f"{BUCKET}/{key}"
    path = os.path.join(tmpdir, os.path.basename(key))
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with _HDF5_LOCK:
                ds = netCDF4.Dataset(path)
                cols = []
                for v in FAPAR_VARS:
                    var = ds[v]
                    var.set_auto_maskandscale(False)
                    cols.append(var[0][row, col].astype(np.float32))
                ds.close()
            os.remove(path)
            vals = np.stack(cols, axis=1)
            fill = vals == FILL
            oor = np.zeros_like(fill)
            for j, v in enumerate(FAPAR_VARS):
                oor[:, j] = (vals[:, j] > VALID_MAX[v]) | ((vals[:, j] < 0) & ~fill[:, j])
            vals[fill | oor] = np.nan
            return vals, fill.sum(0), oor.sum(0)
        except Exception:  # noqa: BLE001
            if os.path.exists(path):
                os.remove(path)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    a = ap.parse_args()

    t0 = time.time()
    idx, lon, lat = load_points(a.points)
    row, col = grid_index(lon, lat)
    files = list_year(a.year)
    od = mod_dir(a.out, a.year, "fapar")
    flog = os.path.join(a.out, "failures.log")
    print(f"{len(idx)} points, {len(files)} daily files for {a.year}", flush=True)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp, \
            ThreadPoolExecutor(a.workers) as ex:
        for m in a.months:
            path = os.path.join(od, f"fapar_{a.year}{m:02d}.npz")
            if os.path.exists(path) and os.path.exists(path + ".json"):
                continue
            ymds = sorted(k for k in files if k.startswith(f"{a.year}{m:02d}"))
            futs = {k: ex.submit(process_day, files[k], row, col, tmp) for k in ymds}
            ps, ds_, vs = [], [], []
            qc = {"year": a.year, "month": m, "n_points": len(idx), "days_in_archive": len(ymds),
                  "failed_days": [], "fill_pixels": [0, 0], "out_of_range_pixels": [0, 0], "n_obs": 0}
            for k in ymds:
                try:
                    vals, nfill, noor = futs[k].result()
                except Exception as e:  # noqa: BLE001
                    qc["failed_days"].append(k)
                    log_failure(flog, f"fapar\t{a.year}\t{k}\t{e}")
                    continue
                qc["fill_pixels"] = [x + int(y) for x, y in zip(qc["fill_pixels"], nfill)]
                qc["out_of_range_pixels"] = [x + int(y) for x, y in zip(qc["out_of_range_pixels"], noor)]
                keep = ~np.isnan(vals).all(axis=1)  # all bands masked -> GEE returns no row
                ps.append(np.nonzero(keep)[0].astype(np.int32))
                ds_.append(np.full(keep.sum(), to_day(f"{k[:4]}-{k[4:6]}-{k[6:]}"), np.int32))
                vs.append(vals[keep])
            qc["expected_days"] = __import__("calendar").monthrange(a.year, m)[1]
            qc["missing_days_in_archive"] = qc["expected_days"] - len(ymds)
            if ps:
                p, d, v = np.concatenate(ps), np.concatenate(ds_), np.concatenate(vs)
            else:
                p, d, v = np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 2), np.float32)
            qc["n_obs"] = int(len(p))
            qc["points_with_any_obs"] = int(len(np.unique(p)))
            qc["complete"] = not qc["failed_days"]
            save_obs(path, p, d, v)
            if qc["complete"]:
                write_json(path + ".json", qc)  # marker: month done (failed months are retried)
            else:
                write_json(path + ".incomplete.json", qc)
            print(f"  {a.year}-{m:02d}: {qc['n_obs']} obs, {len(qc['failed_days'])} failed days, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
