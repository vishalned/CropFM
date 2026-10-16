"""AgERA5 daily meteorology from the Copernicus Climate Data Store (CDS).

Replaces the Climate Engine GEE mirror `ce-ag-era5-v2`. Requires a CDS account, an
accepted licence for `sis-agrometeorological-indicators` and ~/.cdsapirc.

Per month: one global CDS request per variable (zip of daily netCDFs, kept only in
node-local temp space and deleted after sampling) -> nearest-cell lookup at all points ->
one zarr array per month:  <out>/<year>/agera5/agera5_<YYYYMM>.zarr
    shape (n_points, n_days, 8) float32, NaN = no data (e.g. a point whose 0.1 deg cell is
    sea); attrs: days (days since 1970), variables. Plus a .json QC file.

`Derived_Relative_Humidity_2m_Max_24h` is a Climate Engine derived product; we
reconstruct it as e / es(Tmin) (max RH occurs at the daily minimum temperature).
compare.py checks this against the existing data.

usage: python agera5_cds.py --year 2021 --points SUBSET.json --out DIR [--months 1] [--workers 4]
"""
import argparse
import calendar
import glob
import os
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import netCDF4
import numpy as np

from cropfm.data.fast.common import (AGERA5_VARS, load_points, log_failure, mod_dir,
                                     nearest_index, to_day, write_json)

# our name -> (CDS variable, CDS statistic)
CDS_REQ = {
    "Precipitation_Flux": ("precipitation_flux", None),
    "Vapour_Pressure_Mean_24h": ("vapour_pressure", "24_hour_mean"),
    "Temperature_Air_2m_Max_24h": ("2m_temperature", "24_hour_maximum"),
    "Temperature_Air_2m_Min_24h": ("2m_temperature", "24_hour_minimum"),
    "Temperature_Air_2m_Mean_24h": ("2m_temperature", "24_hour_mean"),
    "Solar_Radiation_Flux": ("solar_radiation_flux", None),
    "Wind_Speed_10m_Mean_24h": ("10m_wind_speed", "24_hour_mean"),
}
_HDF5_LOCK = threading.Lock()  # netCDF4/HDF5 is not thread-safe


def retrieve(name, year, month, tmp):
    import cdsapi

    var, stat = CDS_REQ[name]
    ndays = calendar.monthrange(year, month)[1]
    req = {
        "variable": var,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "version": "2_0",
    }
    if stat:
        req["statistic"] = [stat]
    target = os.path.join(tmp, f"{name}_{year}{month:02d}.zip")
    for attempt in range(6):
        try:
            cdsapi.Client(quiet=True).retrieve("sis-agrometeorological-indicators", req, target)
            break
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(60 * (attempt + 1))
    d = os.path.join(tmp, f"{name}_{year}{month:02d}")
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            z.extractall(d)
        os.remove(target)
    else:  # single-file response
        os.makedirs(d, exist_ok=True)
        os.replace(target, os.path.join(d, "x.nc"))
    return d, sorted(glob.glob(os.path.join(d, "**", "*.nc"), recursive=True))


def sample_files(files, lon, lat):
    with _HDF5_LOCK:
        return _sample_files(files, lon, lat)


def _sample_files(files, lon, lat):
    """-> (days (n_days,), vals (n_points, n_days)) sampled with nearest-cell lookup."""
    days, cols, ij = [], [], None
    for f in files:
        ds = netCDF4.Dataset(f)
        vname = [v for v in ds.variables if ds[v].ndim == 3][0]
        if ij is None:
            la, lo = ds["lat"][:], ds["lon"][:]
            lo_q = np.where(lon < lo.min(), lon + 360.0, lon) if lo.max() > 180 else lon
            ij = (nearest_index(la, lat), nearest_index(lo, lo_q))
        t = ds["time"]
        date = netCDF4.num2date(t[0], t.units, only_use_cftime_datetimes=False)
        days.append(to_day(f"{date.year:04d}-{date.month:02d}-{date.day:02d}"))
        var = ds[vname]
        var.set_auto_maskandscale(True)
        arr = np.ma.filled(var[0].astype(np.float32), np.nan)
        cols.append(arr[ij])
        ds.close()
    order = np.argsort(days)
    return np.asarray(days)[order], np.stack(cols, axis=1)[:, order]


def es_hpa(t_kelvin):
    t = t_kelvin - 273.15
    return 6.108 * np.exp(17.27 * t / (t + 237.3))


def do_month(a, year, month, lon, lat, od, flog, t0):
    path = os.path.join(od, f"agera5_{year}{month:02d}.zarr")
    if os.path.exists(path + ".json"):
        return
    import zarr

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        def run(name):
            d, files = retrieve(name, year, month, tmp)
            days, vals = sample_files(files, lon, lat)
            shutil.rmtree(d, ignore_errors=True)  # drop the global fields right away
            return name, days, vals

        with ThreadPoolExecutor(a.workers) as ex:
            res = {n: (d, v) for n, d, v in ex.map(run, list(CDS_REQ))}

    days = res["Precipitation_Flux"][0]
    for n, (d, _) in res.items():
        assert np.array_equal(d, days), f"day mismatch in {n}"
    cube = np.full((len(lon), len(days), len(AGERA5_VARS)), np.nan, np.float32)
    for k, name in enumerate(AGERA5_VARS):
        if name in res:
            cube[:, :, k] = res[name][1]
    e = cube[:, :, AGERA5_VARS.index("Vapour_Pressure_Mean_24h")]
    tmin = cube[:, :, AGERA5_VARS.index("Temperature_Air_2m_Min_24h")]
    cube[:, :, AGERA5_VARS.index("Derived_Relative_Humidity_2m_Max_24h")] = \
        np.clip(100.0 * e / es_hpa(tmin), 0, 100)

    tmp_path = path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    arr = zarr.open_array(tmp_path, mode="w", shape=cube.shape, dtype="float32",
                          chunks=(min(65536, len(lon)), len(days), len(AGERA5_VARS)),
                          fill_value=float("nan"))
    arr[:] = cube
    arr.attrs.update({"days": [int(x) for x in days], "variables": AGERA5_VARS})
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp_path, path)
    nan_frac = np.isnan(cube).mean(axis=(0, 1))
    write_json(path + ".json", {
        "year": year, "month": month, "n_points": int(len(lon)), "n_days": int(len(days)),
        "variables": AGERA5_VARS, "nan_fraction_per_variable": [float(x) for x in nan_frac],
        "points_all_nan": int(np.isnan(cube).all(axis=(1, 2)).sum()), "complete": True,
        "nodata_note": "NaN = masked cell in AgERA5 (typically sea) or missing day",
    })
    print(f"  {year}-{month:02d} done {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    a = ap.parse_args()

    t0 = time.time()
    idx, lon, lat = load_points(a.points)
    od = mod_dir(a.out, a.year, "agera5")
    flog = os.path.join(a.out, "failures.log")
    print(f"{len(idx)} points", flush=True)
    for m in a.months:
        try:
            do_month(a, a.year, m, lon, lat, od, flog, t0)
        except Exception as e:  # noqa: BLE001
            print(f"  {a.year}-{m:02d} FAILED: {e}", flush=True)
            log_failure(flog, f"agera5\t{a.year}\t{m:02d}\t{e}")
    print(f"finished in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
