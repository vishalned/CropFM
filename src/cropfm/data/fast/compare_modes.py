"""Check that tile-grouped and cell-grouped S2 extraction give the same observations.

Compares the sets of (point, day) observations and the band values of matched pairs.
usage: python compare_modes.py CELL_DIR TILE_DIR     (dirs containing cell_*.npz / tile_*.npz)
"""
import glob
import os
import sys

import numpy as np

from cropfm.data.fast.common import load_obs


def load(d, pat):
    parts = [load_obs(f) for f in sorted(glob.glob(os.path.join(d, pat))) if ".tmp" not in f]
    idx = np.concatenate([p[0] for p in parts]).astype(np.int64)
    day = np.concatenate([p[1] for p in parts]).astype(np.int64)
    val = np.concatenate([p[2] for p in parts])
    key = idx * 100000 + day
    _, first = np.unique(key, return_index=True)  # duplicates (overlapping tiles): keep first
    return key[first], val[first], len(key)


kc, vc, nc = load(sys.argv[1], "cell_*.npz")
kt, vt, nt = load(sys.argv[2], "tile_*.npz")
print(f"cell mode: {nc} rows ({len(kc)} unique point-days); tile mode: {nt} rows ({len(kt)} unique)")
common = np.intersect1d(kc, kt)
print(f"unique point-days in both: {len(common)}  only cell: {len(kc)-len(common)}  only tile: {len(kt)-len(common)}")
ic = np.searchsorted(kc, common); it = np.searchsorted(kt, common)
a, b = vc[ic], vt[it]
ok = ~(np.isnan(a) | np.isnan(b))
d = np.abs(a - b)
print("matched values identical (all bands, ignoring NaN): %.4f%%" % (100 * (d[ok] == 0).mean()))
print("max abs diff over matched values:", float(np.nanmax(np.where(ok, d, 0))))
pts_c = len(np.unique(kc // 100000)); pts_t = len(np.unique(kt // 100000))
print(f"points with any obs: cell {pts_c}, tile {pts_t}")
