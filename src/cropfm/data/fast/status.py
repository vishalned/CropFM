"""Summarise download progress and QC from the per-cell / per-month json markers.

usage: python -m cropfm.data.fast.status [--out /projects/prjs1788/CropFM/fast_dl/raw]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from cropfm.data.fast.common import PROD_ROOT

N_CELLS_EXPECTED = None  # filled from the points list on demand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PROD_ROOT)
    ap.add_argument("--expected-cells", type=int, default=5914)
    a = ap.parse_args()

    print(f"{'year':>5s} {'modality':10s} {'done':>7s} {'incomplete':>10s} {'expected':>8s}  notes")
    for ydir in sorted(glob.glob(os.path.join(a.out, "20??"))):
        year = os.path.basename(ydir)
        for mod in ("sentinel2", "sentinel1"):
            d = os.path.join(ydir, mod)
            if not os.path.isdir(d):
                continue
            done = len(glob.glob(os.path.join(d, "cell_*[0-9].json")))
            inc = glob.glob(os.path.join(d, "cell_*.incomplete.json"))
            obs = pts = 0
            for f in glob.glob(os.path.join(d, "cell_*[0-9].json")):
                j = json.load(open(f))
                obs += j["n_obs"]; pts += j["points_with_any_obs"]
            print(f"{year:>5s} {mod:10s} {done:7d} {len(inc):10d} {a.expected_cells:8d}  "
                  f"{obs:,} obs, {pts:,} point-cells with data")
        for mod, pat, n in (("fapar", "fapar_*.npz.json", 12), ("agera5", "agera5_*.zarr.json", 12)):
            d = os.path.join(ydir, mod)
            if os.path.isdir(d):
                done = len(glob.glob(os.path.join(d, pat)))
                inc = len(glob.glob(os.path.join(d, "*.incomplete.json")))
                print(f"{year:>5s} {mod:10s} {done:7d} {inc:10d} {n:8d}  (months)")
    fl = os.path.join(a.out, "failures.log")
    if os.path.exists(fl):
        kinds = defaultdict(int)
        for line in open(fl):
            kinds[tuple(line.split("\t")[:2])] += 1
        print("\nfailures.log entries by (modality, year):", dict(kinds))


if __name__ == "__main__":
    main()
