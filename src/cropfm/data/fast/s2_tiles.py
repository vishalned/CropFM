"""Sentinel-2 extraction grouped by MGRS tile (each scene is opened once for ALL its points).

The cell-based driver (`stac_cells.py`) opens every scene once per 1-degree cell that the
scene overlaps (~4x). Here the work unit is an S2 tile: all points assigned to the tile are
sampled from each of the tile's scenes in a single pass.

Stage 1 (tile assignment, same for every year, cached in <out>/tiles/): for each 1-degree
cell with points, one STAC search over a short summer window lists the tiles touching the
cell and their centres; every point is assigned to the tile whose centre is nearest (the
nearest-centre tile always contains the point, tiles overlap by ~5 km).
Stage 2: per tile and year, search that tile's scenes (cloud < threshold), sample, save
    <out>/<year>/sentinel2/tile_<ID>.npz + .json (same QC fields as the cell driver).

usage: python s2_tiles.py --year 2021 --points SUBSET.json --out DIR [--task-id I --n-tasks N]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac_client
from shapely.geometry import shape

from cropfm.data.fast.common import (S2_VARS, load_points, log_failure, mod_dir, save_obs, to_day,
                                     write_json)
from cropfm.data.fast.s2_pc import dedupe, scene
from cropfm.data.fast.stac_cells import STAC_URL, cell_ids, sample_with_retry, search


def build_assignment(lon, lat, year, workers, path):
    """-> (tiles list[str], assign int32 array: tile index per point or -1). Cached at `path`."""
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return list(z["tiles"]), z["assign"]
    catalog = pystac_client.Client.open(STAC_URL)
    cid = cell_ids(lon, lat)
    cells = np.unique(cid)

    def one(c):
        lo, la = int(c // 1000 - 180), int(c % 1000 - 90)
        for attempt in range(6):
            try:
                items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=[lo, la, lo + 1, la + 1],
                                            datetime=f"{year}-06-01/{year}-06-20").items())
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))
        best = {}
        for it in items:  # most complete (least nodata) footprint per tile -> centre
            t = it.properties["s2:mgrs_tile"]
            nd = it.properties.get("s2:nodata_pixel_percentage", 100.0)
            if t not in best or nd < best[t][0]:
                c_ = shape(it.geometry).centroid
                best[t] = (nd, c_.x, c_.y)
        return c, {t: (v[1], v[2]) for t, v in best.items()}

    with ThreadPoolExecutor(workers) as ex:
        per_cell = dict(ex.map(one, cells))
    tiles = sorted({t for d in per_cell.values() for t in d})
    tindex = {t: i for i, t in enumerate(tiles)}
    assign = np.full(len(lon), -1, np.int32)
    for c in cells:
        d = per_cell[c]
        if not d:
            continue
        sel = np.nonzero(cid == c)[0]
        names = list(d)
        cx = np.array([d[t][0] for t in names])
        cy = np.array([d[t][1] for t in names])
        dx = (lon[sel][:, None] - cx[None, :]) * np.cos(np.radians(lat[sel]))[:, None]
        dy = lat[sel][:, None] - cy[None, :]
        assign[sel] = [tindex[names[j]] for j in np.argmin(dx * dx + dy * dy, axis=1)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", tiles=np.array(tiles), assign=assign)
    os.replace(path + ".tmp.npz", path)
    return tiles, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32, help="threads over scenes")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--cloud", type=float, default=50.0)
    ap.add_argument("--assignment", default=None, help="cache file for the tile assignment")
    a = ap.parse_args()

    t0 = time.time()
    idx, lon, lat = load_points(a.points)
    path = a.assignment or os.path.join(a.out, "tiles", "s2_tile_assignment.npz")
    tiles, assign = build_assignment(lon, lat, a.year, 8, path)
    print(f"[tiles] {len(tiles)} tiles, {int((assign < 0).sum())} unassigned points "
          f"(assignment {time.time()-t0:.0f}s)", flush=True)

    od = mod_dir(a.out, a.year, "sentinel2")
    flog = os.path.join(a.out, "failures.log")
    catalog = pystac_client.Client.open(STAC_URL)
    order = np.argsort(assign, kind="stable")
    a_sorted = assign[order]
    pick = list(range(a.task_id, len(tiles), a.n_tasks))
    n_done = n_skip = n_err = 0
    for ti in pick:
        base = os.path.join(od, f"tile_{tiles[ti]}")
        if os.path.exists(base + ".json"):
            n_skip += 1
            continue
        lo_i, hi_i = np.searchsorted(a_sorted, ti), np.searchsorted(a_sorted, ti, side="right")
        sel = order[lo_i:hi_i]
        if len(sel) == 0:
            continue
        tc = time.time()
        plon, plat = lon[sel], lat[sel]
        try:
            bbox = [float(plon.min()) - 1e-3, float(plat.min()) - 1e-3,
                    float(plon.max()) + 1e-3, float(plat.max()) + 1e-3]
            items = dedupe(search(catalog, "sentinel-2-l2a", bbox, a.year,
                                  {"eo:cloud_cover": {"lt": a.cloud},
                                   "s2:mgrs_tile": {"eq": tiles[ti]}}))

            def work(item):
                b = item.bbox
                cand = np.nonzero((plon >= b[0]) & (plon <= b[2]) & (plat >= b[1]) & (plat <= b[3]))[0]
                if len(cand) == 0:
                    return item.id, None, None, 0
                vals, err = sample_with_retry(lambda: scene(item, plon[cand], plat[cand]))
                if err:
                    return item.id, None, err, len(cand)
                keep = ~np.isnan(vals).all(axis=1)
                if not keep.any():
                    return item.id, None, None, len(cand)
                return item.id, (sel[cand[keep]], to_day(item.datetime.date()), vals[keep]), None, len(cand)

            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(work, items))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            log_failure(flog, f"sentinel2\t{a.year}\ttile_{tiles[ti]}\t{type(e).__name__}: {e}")
            print(f"  TILE FAILED {tiles[ti]}: {e}", flush=True)
            continue
        failed = [(i, e) for i, _, e, _ in res if e]
        parts = [r for _, r, _, _ in res if r is not None]
        pairs = int(sum(n for *_, n in res))
        if parts:
            p = np.concatenate([r[0] for r in parts])
            d = np.concatenate([np.full(len(r[0]), r[1]) for r in parts])
            v = np.concatenate([r[2] for r in parts])
        else:
            p, d, v = np.zeros(0, int), np.zeros(0, int), np.zeros((0, len(S2_VARS)), np.float32)
        save_obs(base + ".npz", p, d, v)
        write_json(base + (".json" if not failed else ".incomplete.json"), {
            "year": a.year, "tile": tiles[ti], "n_points": int(len(sel)), "n_scenes_found": len(items),
            "n_scenes_failed": len(failed), "failed_scenes": [{"id": i, "error": e} for i, e in failed],
            "n_obs": int(len(p)), "points_with_any_obs": int(len(np.unique(p))),
            "point_scene_pairs_in_bbox": pairs, "pairs_dropped_nodata": pairs - int(len(p)),
            "nan_fraction_per_band_among_kept": dict(zip(S2_VARS, [float(x) for x in
                                                      (np.isnan(v).mean(axis=0) if len(v) else np.zeros(len(S2_VARS)))])),
            "complete": not failed, "seconds": round(time.time() - tc, 1)})
        for i, e in failed:
            log_failure(flog, f"sentinel2\t{a.year}\ttile_{tiles[ti]}\tscene {i}\t{e}")
        n_done += 1
        print(f"  tile {tiles[ti]}: {len(sel)} pts, {len(items)} scenes, {len(p)} obs, "
              f"{len(failed)} failed, {time.time()-tc:.0f}s", flush=True)
    print(f"[sentinel2 tiles {a.year}] finished: {n_done} new, {n_skip} done, {n_err} errored, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
