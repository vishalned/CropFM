"""Compare newly downloaded observations with the existing (GEE-derived, cleaned) zarr.

The new observations are put through the *same* weekly aggregation as the reference
(`clean_data.aggregate_to_weekly`) and compared week-by-week (matched on ISO-week key).

usage: python compare.py --obs-dir DIR --points SUBSET.json --year 2021 \
           [--modalities agera5 fapar sentinel1 sentinel2] [--report out.json]
"""
import argparse
import glob
import json
import os

import numpy as np
import zarr

from cropfm.data.clean_data import aggregate_to_weekly
from cropfm.data.fast.common import (AGERA5_VARS, FAPAR_VARS, REF_ZARR, S1_VARS, S2_VARS,
                                     from_day, load_obs, load_points)

VARS = {"agera5": AGERA5_VARS, "fapar": FAPAR_VARS, "sentinel1": S1_VARS, "sentinel2": S2_VARS}


def agera5_months_to_obs(paths):
    import zarr

    parts = []
    for p in paths:
        arr = zarr.open_array(p, mode="r")
        days = np.asarray(arr.attrs["days"])
        cube = arr[:]
        keep = ~np.isnan(cube).all(axis=2)
        pi, di = np.nonzero(keep)
        parts.append((pi.astype(np.int32), days[di].astype(np.int32), cube[pi, di]))
    return parts


def load_modality(obs_dir, mod, year):
    """Supports the production layout <dir>/<year>/<modality>/ and the older test layout."""
    sub = {"agera5": "agera5", "fapar": "fapar", "sentinel1": "sentinel1", "sentinel2": "sentinel2"}[mod]
    d = os.path.join(obs_dir, str(year), sub)
    if os.path.isdir(d):
        if mod == "agera5":
            return agera5_months_to_obs(sorted(glob.glob(os.path.join(d, "agera5_*.zarr"))))
        pat = "fapar_*.npz" if mod == "fapar" else "cell_*.npz"
        return [load_obs(f) for f in sorted(glob.glob(os.path.join(d, pat)))
                if not f.endswith(".tmp.npz")]
    if mod in ("agera5", "fapar"):
        f = os.path.join(obs_dir, f"{mod}_{year}.npz")
        return [load_obs(f)] if os.path.exists(f) else []
    files = sorted(glob.glob(os.path.join(obs_dir, f"{mod}_{year}", "cell_*.npz")))
    return [load_obs(f) for f in files]


def group_by_point(parts, n_points):
    idx = np.concatenate([p[0] for p in parts])
    day = np.concatenate([p[1] for p in parts])
    val = np.concatenate([p[2] for p in parts])
    order = np.lexsort((day, idx))
    idx, day, val = idx[order], day[order], val[order]
    bounds = np.searchsorted(idx, np.arange(n_points + 1))
    return bounds, day, val


def compare_modality(mod, parts, glob_idx, ref):
    n = len(glob_idx)
    bounds, day, val = group_by_point(parts, n)
    names = list(VARS[mod])
    rdata = ref[f"temporal_modalities/{mod}/data"]
    rkeys = ref[f"temporal_modalities/{mod}/timestamps"]
    rvalid = ref[f"temporal_modalities/{mod}/valid_mask"]
    ref_data, ref_keys, ref_valid = (rdata.oindex[glob_idx], rkeys.oindex[glob_idx],
                                     rvalid.oindex[glob_idx])

    diffs = [[] for _ in names]
    refs = [[] for _ in names]
    news = [[] for _ in names]
    both_valid = only_new = only_ref = 0
    from collections import Counter
    wk_new, wk_ref = Counter(), Counter()
    for p in range(n):
        s, e = bounds[p], bounds[p + 1]
        if e > s:
            ts = np.array([from_day(d) for d in day[s:e]])
            nd, nk, nv = aggregate_to_weekly(val[s:e], ts, mod, "mean")
        else:
            nd, nk, nv = np.full((52, len(names)), np.nan), np.array(["NA"] * 52), np.zeros(52, bool)
        new = {k: (nd[i], nv[i]) for i, k in enumerate(nk) if k != "NA"}
        old = {k: (ref_data[p, i], ref_valid[p, i]) for i, k in enumerate(ref_keys[p]) if k != "NA"}
        for k in set(new) | set(old):
            nvld = k in new and new[k][1]
            ovld = k in old and old[k][1]
            if nvld and ovld:
                both_valid += 1
                for c in range(len(names)):
                    a, b = new[k][0][c], old[k][0][c]
                    if np.isfinite(a) and np.isfinite(b):
                        news[c].append(a), refs[c].append(b), diffs[c].append(a - b)
            elif nvld:
                only_new += 1
                wk_new[k] += 1
            elif ovld:
                only_ref += 1
                wk_ref[k] += 1

    rep = {"weeks_both_valid": both_valid, "weeks_only_new": only_new, "weeks_only_ref": only_ref,
           "agreement_of_validity": both_valid / max(both_valid + only_new + only_ref, 1), "vars": {},
           "only_new_by_week": dict(sorted(wk_new.items())), "only_ref_by_week": dict(sorted(wk_ref.items()))}
    for c, name in enumerate(names):
        if not diffs[c]:
            rep["vars"][name] = {"n": 0}
            continue
        d, r, a = np.array(diffs[c]), np.array(refs[c]), np.array(news[c])
        scale = np.maximum(np.abs(r), 1e-9)
        rep["vars"][name] = {
            "n": int(len(d)),
            "bias(new-ref)": float(d.mean()),
            "median_abs_diff": float(np.median(np.abs(d))),
            "median_rel_diff": float(np.median(np.abs(d) / scale)),
            "frac_rel_diff_lt_1pct": float((np.abs(d) / scale < 0.01).mean()),
            "frac_rel_diff_lt_5pct": float((np.abs(d) / scale < 0.05).mean()),
            "corr": float(np.corrcoef(a, r)[0, 1]) if len(d) > 2 and a.std() > 0 and r.std() > 0 else None,
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--modalities", nargs="+", default=list(VARS))
    ap.add_argument("--report", default=None)
    ap.add_argument("--ref", default=REF_ZARR)
    a = ap.parse_args()

    glob_idx, _, _ = load_points(a.points)
    ref = zarr.open_group(a.ref, mode="r")
    report = {}
    for mod in a.modalities:
        parts = load_modality(a.obs_dir, mod, a.year)
        if not parts:
            print(f"[{mod}] no observation files found, skipping")
            continue
        report[mod] = compare_modality(mod, parts, glob_idx, ref)
        r = report[mod]
        print(f"\n=== {mod} === weeks valid in both: {r['weeks_both_valid']}, only new: "
              f"{r['weeks_only_new']}, only ref: {r['weeks_only_ref']} "
              f"(validity agreement {r['agreement_of_validity']:.3f})")
        print("  only-new weeks:", dict(list(r["only_new_by_week"].items())[:60]))
        print("  only-ref weeks:", dict(list(r["only_ref_by_week"].items())[:60]))
        for k, v in r["vars"].items():
            if v["n"]:
                print(f"  {k:40s} n={v['n']:7d} bias={v['bias(new-ref)']:+.4g} medAbs={v['median_abs_diff']:.4g} "
                      f"medRel={v['median_rel_diff']:.4g} <1%={v['frac_rel_diff_lt_1pct']:.3f} "
                      f"<5%={v['frac_rel_diff_lt_5pct']:.3f} corr={v['corr']}")
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
