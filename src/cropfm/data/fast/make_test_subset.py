"""Pick a small, spatially clustered test subset: the N densest 1-degree cells, one or
two per continent. Clustering mimics real density, so S1/S2 throughput numbers are
representative (random scattered points would be unrealistically wasteful).

usage: python make_test_subset.py OUT.json
"""
import json
import sys

import numpy as np
import zarr

from cropfm.data.fast.common import REF_ZARR

# (continent, extra filter on (lon, lat) cell corner) -> pick densest cell
REGIONS = {
    "Europe": lambda lo, la: True,
    "North America": lambda lo, la: True,
    "South America": lambda lo, la: True,
    "Africa": lambda lo, la: True,
    "Oceania": lambda lo, la: True,
    "Asia": lambda lo, la: lo < 100 and la < 30,  # tropical / monsoon Asia
}

r = zarr.open_group(REF_ZARR, mode="r")
c = r["metadata/coordinates"][:]
cont = r["metadata/sample_info"][:]["continent"]
cell_lon = np.floor(c[:, 0]).astype(int)
cell_lat = np.floor(c[:, 1]).astype(int)
cell = (cell_lon + 180) * 1000 + (cell_lat + 90)

indices, info = [], {}
for name, ok in REGIONS.items():
    m = cont == name
    u, cnt = np.unique(cell[m], return_counts=True)
    for j in np.argsort(-cnt):
        lo, la = int(u[j] // 1000 - 180), int(u[j] % 1000 - 90)
        if ok(lo, la):
            sel = np.where(m & (cell == u[j]))[0]
            indices.extend(sel.tolist())
            info[name] = {"cell_lon": lo, "cell_lat": la, "n": len(sel)}
            break

out = sys.argv[1]
json.dump({"indices": sorted(indices), "cells": info}, open(out, "w"))
print(info, "total", len(indices))
