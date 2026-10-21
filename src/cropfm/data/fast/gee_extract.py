"""Batched Google Earth Engine extraction of Sentinel-2 / Sentinel-1 point time series.

Replaces the one-point-per-getInfo() GEE pipeline. Earth Engine reads the pixels on Google's
side; we only download a small CSV table, so (unlike COG streaming) it moves almost no data
over the cluster network. Same collections / filters / bands as the original pipeline:
  S2: COPERNICUS/S2_SR_HARMONIZED, CLOUDY_PIXEL_PERCENTAGE < 50, 15 bands, scale 10 m
  S1: COPERNICUS/S1_GRD, IW, VV+VH, bands VV VH angle, scale 10 m (dB, as in the old data)

One request = (chunk of <= --pts-per-req points of one 1-degree cell) x (<= --months-per-req
months): reduceRegions(first) over every image, flatten, getDownloadURL(CSV). Requests run in
a thread pool across a batch of cells. Output (same layout as the other extractors):
    <out>/<year>/<sensor>/cell_<lon>_<lat>.npz   idx (position in points list), day, vals
    <out>/<year>/<sensor>/cell_<lon>_<lat>.json  QC + completion marker
Cells are skipped if their .json exists, so re-running resumes.

No-data: masked pixels come back as null from Earth Engine and are dropped (no row), as in
the original pipeline. S2 rows whose 12 reflectance bands are all 0 (outside the swath) are
dropped too (the old cleaning turned them into NaN anyway). Remaining NaNs are per band.

usage: python gee_extract.py --sensor sentinel2 --year 2021 --points SUBSET.json --out DIR
"""
import argparse
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
import requests

from cropfm.data.fast.common import (S1_VARS, S2_VARS, load_points, log_failure, mod_dir,
                                     save_obs, to_day, write_json)

SENSORS = {
    "sentinel2": dict(bands=S2_VARS, first="B2"),
    "sentinel1": dict(bands=S1_VARS, first="VV"),
}
HIGHVOLUME = "https://earthengine-highvolume.googleapis.com"
_init_lock = threading.Lock()
_inited = False


DEFAULT_PROJECT = "cropfm"


def default_project():
    return DEFAULT_PROJECT


def init(project=None):
    global _inited
    with _init_lock:
        if not _inited:
            proj = project or os.environ.get("EE_PROJECT") or default_project()
            key = os.environ.get("EE_KEY_FILE")
            if key:  # service account: no browser login, no expiring refresh token
                creds = ee.ServiceAccountCredentials(json.load(open(key))["client_email"], key)
                ee.Initialize(creds, project=proj, opt_url=HIGHVOLUME)
            else:
                ee.Initialize(project=proj, opt_url=HIGHVOLUME)
            _inited = True


def build_collection(sensor, start, end, rect, cloud):
    if sensor == "sentinel2":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end)
               .filterBounds(rect).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
    else:
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, end).filterBounds(rect)
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
               .filter(ee.Filter.eq("instrumentMode", "IW")))
    return col.select(SENSORS[sensor]["bands"])


def request_rows(sensor, pids, lons, lats, start, end, cloud, tile_scale):
    """One EE request -> DataFrame [pid, date, bands...] (possibly empty)."""
    bands = SENSORS[sensor]["bands"]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"pid": int(p)})
                               for p, x, y in zip(pids, lons, lats)])
    rect = ee.Geometry.Rectangle([float(lons.min()) - 1e-4, float(lats.min()) - 1e-4,
                                  float(lons.max()) + 1e-4, float(lats.max()) + 1e-4])
    col = build_collection(sensor, start, end, rect, cloud)

    def per_image(img):
        d = img.date().format("YYYY-MM-dd")
        return img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10,
                                 tileScale=tile_scale).map(lambda f: f.set("date", d))

    table = (col.map(per_image).flatten()
             .filter(ee.Filter.notNull([SENSORS[sensor]["first"]])))
    url = table.getDownloadURL(filetype="csv", selectors=["pid", "date"] + bands)
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame(columns=["pid", "date"] + bands)
    return pd.read_csv(io.StringIO(r.text))


def with_retry(fn, tries=7):
    err = None
    for attempt in range(tries):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:400]
            time.sleep(min(120, 5 * 2 ** attempt))
    return None, err


def month_windows(year, months_per_req):
    wins = []
    for m in range(1, 13, months_per_req):
        m2 = min(m + months_per_req, 13)
        wins.append((f"{year}-{m:02d}-01", f"{year + 1}-01-01" if m2 == 13 else f"{year}-{m2:02d}-01"))
    return wins


def frame_to_obs(sensor, df):
    bands = SENSORS[sensor]["bands"]
    if len(df) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, len(bands)), np.float32)
    for b in bands:
        if b not in df.columns:
            df[b] = np.nan
    vals = df[bands].to_numpy(np.float32)
    day = (pd.to_datetime(df["date"]).values.astype("datetime64[D]").astype(np.int64)).astype(np.int32)
    pid = df["pid"].to_numpy(np.int64).astype(np.int32)
    keep = ~np.isnan(vals).all(axis=1)
    if sensor == "sentinel2":  # all 12 reflectance bands == 0 -> outside swath
        refl = [bands.index(b) for b in bands if b not in ("SCL", "MSK_CLDPRB", "QA60")]
        keep &= ~(np.nan_to_num(vals[:, refl], nan=0.0).max(axis=1) == 0)
    return pid[keep], day[keep], vals[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=list(SENSORS), required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16, help="concurrent EE requests")
    ap.add_argument("--pts-per-req", type=int, default=250)
    ap.add_argument("--months-per-req", type=int, default=3)
    ap.add_argument("--cell-batch", type=int, default=16, help="cells processed together")
    ap.add_argument("--cloud", type=float, default=50.0)
    ap.add_argument("--tile-scale", type=int, default=1)
    ap.add_argument("--project", default=None)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=None, help="benchmark: stop after N cells")
    a = ap.parse_args()

    init(a.project)
    t0 = time.time()
    idx, lon, lat = load_points(a.points)
    cid = (np.floor(lon).astype(int) + 180) * 1000 + (np.floor(lat).astype(int) + 90)
    order = np.argsort(cid, kind="stable")
    cells, starts = np.unique(cid[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    pick = np.arange(a.task_id, len(cells), a.n_tasks)
    od = mod_dir(a.out, a.year, a.sensor)
    flog = os.path.join(a.out, "failures.log")
    wins = month_windows(a.year, a.months_per_req)
    todo = [k for k in pick if not os.path.exists(os.path.join(
        od, f"cell_{int(cells[k] // 1000 - 180)}_{int(cells[k] % 1000 - 90)}.json"))]
    if a.max_cells:
        todo = todo[:a.max_cells]
    print(f"[gee {a.sensor} {a.year}] task {a.task_id}/{a.n_tasks}: {len(todo)} cells to do of "
          f"{len(pick)}; {len(wins)} windows; {a.pts_per_req} pts/req; {a.workers} workers", flush=True)
    stats = {"requests": 0, "rows": 0, "seconds_in_requests": 0.0}
    stat_lock = threading.Lock()

    def do_request(job):
        k, sel, w = job
        t = time.time()
        # a spatially compact chunk keeps the bounding rectangle (and the image list) small
        df, err = with_retry(lambda: request_rows(a.sensor, sel, lon[sel], lat[sel], w[0], w[1],
                                                  a.cloud, a.tile_scale))
        with stat_lock:
            stats["requests"] += 1
            stats["rows"] += 0 if df is None else len(df)
            stats["seconds_in_requests"] += time.time() - t
        return k, w, df, err

    n_done = n_inc = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for b0 in range(0, len(todo), a.cell_batch):
            batch = todo[b0:b0 + a.cell_batch]
            tb = time.time()
            jobs = []
            for k in batch:
                sel = order[starts[k]:ends[k]]
                sel = sel[np.lexsort((lon[sel], np.floor(lat[sel] * 4)))]  # compact chunks
                for c0 in range(0, len(sel), a.pts_per_req):
                    for w in wins:
                        jobs.append((k, sel[c0:c0 + a.pts_per_req], w))
            results = {k: [] for k in batch}
            errors = {k: [] for k in batch}
            for k, w, df, err in ex.map(do_request, jobs):
                (errors if err else results)[k].append((w, err) if err else df)
            for k in batch:
                c = cells[k]
                lo, la = int(c // 1000 - 180), int(c % 1000 - 90)
                base = os.path.join(od, f"cell_{lo}_{la}")
                npts = int(ends[k] - starts[k])
                frames = [d for d in results[k] if len(d)]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                p, d, v = frame_to_obs(a.sensor, df)
                # EE can return an image twice only if the collection has duplicates; keep as is
                save_obs(base + ".npz", p, d, v)
                nanfrac = np.isnan(v).mean(axis=0) if len(v) else np.zeros(len(SENSORS[a.sensor]["bands"]))
                write_json(base + (".json" if not errors[k] else ".incomplete.json"), {
                    "year": a.year, "cell": [lo, la], "source": "gee", "n_points": npts,
                    "n_requests": len([1 for j in jobs if j[0] == k]),
                    "failed_requests": [{"window": list(w), "error": e} for w, e in errors[k]],
                    "rows_returned": int(len(df)), "n_obs": int(len(p)),
                    "points_with_any_obs": int(len(np.unique(p))),
                    "nan_fraction_per_band_among_kept": dict(zip(SENSORS[a.sensor]["bands"],
                                                                 [float(x) for x in nanfrac])),
                    "complete": not errors[k],
                    "note": "masked (null) pixels are dropped by Earth Engine and not counted"})
                for w, e in errors[k]:
                    log_failure(flog, f"{a.sensor}\t{a.year}\tcell_{lo}_{la}\t{w}\t{e}")
                n_done += 1
                n_inc += bool(errors[k])
            el = time.time() - tb
            print(f"  batch of {len(batch)} cells ({len(jobs)} requests): {el:.0f}s | total "
                  f"{n_done} cells, {stats['requests']} requests, {stats['rows']} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)
    el = time.time() - t0
    print(f"[gee {a.sensor} {a.year}] finished: {n_done} cells ({n_inc} incomplete) in {el:.0f}s; "
          f"{stats['requests']} requests, avg {stats['seconds_in_requests']/max(stats['requests'],1):.1f}s each, "
          f"{stats['requests']/max(el,1e-9):.2f} req/s, {stats['rows']/max(el,1e-9):.0f} rows/s", flush=True)


if __name__ == "__main__":
    main()
