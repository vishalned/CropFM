"""Shared driver for STAC-based S1/S2 extraction, organised by 1-degree cells.

For each cell: STAC search -> per scene, sample all points of the cell (threads over
scenes; COG blocks are streamed over HTTP and never written to disk) -> one observation
file per cell:
    <out>/<year>/<prefix>/cell_<lon>_<lat>.npz   (idx, day, vals)
    <out>/<year>/<prefix>/cell_<lon>_<lat>.json  (QC / completion marker)
A cell counts as done only when the .json says complete=true (no scene failed). Interrupted
or array jobs can simply be re-run: done cells are skipped, incomplete ones are retried.
Cells that raise are appended to <out>/failures.log.
"""
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
import pystac_client

from cropfm.data.fast.common import load_points, log_failure, mod_dir, save_obs, to_day, write_json

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def cell_ids(lon, lat):
    return (np.floor(lon).astype(int) + 180) * 1000 + (np.floor(lat).astype(int) + 90)


def sign(href):
    return pc.sign(href)


def search(catalog, collection, bbox, year, query=None):
    for attempt in range(6):
        try:
            return list(catalog.search(collections=[collection], bbox=bbox,
                                       datetime=f"{year}-01-01/{year}-12-31",
                                       query=query).items())
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def sample_with_retry(fn, tries=5):
    """-> (result, error_string_or_None)"""
    err = None
    for attempt in range(tries):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
            time.sleep(3 * (attempt + 1))
    return None, err


def process_cell(args, catalog, c, sel, clon, clat, prefix, collection, query, scene_filter,
                 scene_fn, n_vars, var_names):
    items = scene_filter(search(catalog, collection, [int(c // 1000 - 180), int(c % 1000 - 90),
                                                      int(c // 1000 - 180) + 1,
                                                      int(c % 1000 - 90) + 1], args.year, query))
    stats = {"candidate_pairs": 0, "kept_pairs": 0}

    def work(item):
        b = item.bbox
        cand = np.nonzero((clon >= b[0]) & (clon <= b[2]) & (clat >= b[1]) & (clat <= b[3]))[0]
        if len(cand) == 0:
            return item.id, None, None, 0
        vals, err = sample_with_retry(lambda: scene_fn(item, clon[cand], clat[cand]))
        if err:
            return item.id, None, err, len(cand)
        keep = ~np.isnan(vals).all(axis=1)  # no valid band at all = outside swath / nodata
        if not keep.any():
            return item.id, None, None, len(cand)
        return item.id, (sel[cand[keep]], to_day(item.datetime.date()), vals[keep]), None, len(cand)

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(work, items))

    failed = [(i, e) for i, _, e, _ in res if e]
    parts = [r for _, r, _, _ in res if r is not None]
    stats["candidate_pairs"] = int(sum(n for *_, n in res))
    if parts:
        p = np.concatenate([r[0] for r in parts])
        d = np.concatenate([np.full(len(r[0]), r[1]) for r in parts])
        v = np.concatenate([r[2] for r in parts])
    else:
        p, d, v = np.zeros(0, int), np.zeros(0, int), np.zeros((0, n_vars), np.float32)
    stats["kept_pairs"] = int(len(p))
    return items, failed, (p, d, v), stats


def run(args, prefix, collection, query, scene_filter, scene_fn, var_names):
    """scene_fn(item, lon, lat) -> (n_pts, n_vars) float32 with NaN = no data."""
    t0 = time.time()
    n_vars = len(var_names)
    idx, lon, lat = load_points(args.points)
    cid = cell_ids(lon, lat)
    order = np.argsort(cid, kind="stable")
    cells, starts = np.unique(cid[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    pick = np.arange(args.task_id, len(cells), args.n_tasks)  # stride split for slurm arrays
    od = mod_dir(args.out, args.year, prefix)
    flog = os.path.join(args.out, "failures.log")
    catalog = pystac_client.Client.open(STAC_URL)
    print(f"[{prefix} {args.year}] task {args.task_id}/{args.n_tasks}: {len(pick)} of "
          f"{len(cells)} cells, {len(idx)} points total", flush=True)

    n_done = n_skip = n_inc = n_err = 0
    for k in pick:
        c = cells[k]
        lo, la = int(c // 1000 - 180), int(c % 1000 - 90)
        base = os.path.join(od, f"cell_{lo}_{la}")
        if os.path.exists(base + ".json"):
            n_skip += 1
            continue
        tc = time.time()
        sel = order[starts[k]:ends[k]]
        try:
            items, failed, (p, d, v), stats = process_cell(
                args, catalog, c, sel, lon[sel], lat[sel], prefix, collection, query,
                scene_filter, scene_fn, n_vars, var_names)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            msg = f"{prefix}\t{args.year}\tcell_{lo}_{la}\t{type(e).__name__}: {e}"
            print("  CELL FAILED", msg, flush=True)
            log_failure(flog, msg)
            continue
        save_obs(base + ".npz", p, d, v)  # idx = position in the points list
        nan_per_band = np.isnan(v).mean(axis=0) if len(v) else np.zeros(n_vars)
        write_json(base + ".json" if not failed else base + ".incomplete.json", {
            "year": args.year, "cell": [lo, la], "n_points": int(len(sel)),
            "n_scenes_found": len(items), "n_scenes_failed": len(failed),
            "failed_scenes": [{"id": i, "error": e} for i, e in failed],
            "n_obs": int(len(p)), "points_with_any_obs": int(len(np.unique(p))),
            "point_scene_pairs_in_bbox": stats["candidate_pairs"],
            "pairs_dropped_nodata": stats["candidate_pairs"] - stats["kept_pairs"],
            "nan_fraction_per_band_among_kept": dict(zip(var_names, [float(x) for x in nan_per_band])),
            "complete": not failed, "seconds": round(time.time() - tc, 1),
        })
        for i, e in failed:
            log_failure(flog, f"{prefix}\t{args.year}\tcell_{lo}_{la}\tscene {i}\t{e}")
        n_done += 1
        n_inc += bool(failed)
        print(f"  cell ({lo},{la}): {len(sel)} pts, {len(items)} scenes, {len(p)} obs, "
              f"{len(failed)} failed scenes, {time.time()-tc:.0f}s", flush=True)
    print(f"[{prefix} {args.year}] finished: {n_done} new ({n_inc} incomplete), {n_skip} already "
          f"done, {n_err} errored, {time.time()-t0:.0f}s", flush=True)


def add_args(ap):
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32, help="threads over scenes")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    return ap
