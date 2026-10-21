# Fast download (branch `fast-download`): status summary

Last updated 2026-10-21. Scope: Google Earth Engine (GEE) only for Sentinel-2 and Sentinel-1.
The Planetary Computer (COG) route for S1/S2 was dropped; its code is still in `src/cropfm/data/fast/`
but is not used.

## What we want to download

- **Locations:** the 1,792,422 crop points already in the reference zarr
  (`/projects/prjs1788/CropFM/global_data_1200_merged_cleaned_rechunked.zarr`, `metadata/coordinates`).
  They were sampled earlier: 5 km grid per country, kept where WorldCereal 2021 temporary-crop area >= 5 km2,
  then up to 10 random temporary-crop pixels per grid cell (`src/cropfm/data/grids/`).
- **Years:** 2018-2025 (all satellites have proper data from 2018).
- **Modalities:** AgERA5 (CDS), FAPAR/LAI (NOAA S3), Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`,
  cloudy pixel % < 50) and Sentinel-1 GRD (IW, VV+VH, angle) from GEE.
  Static modalities (soil, elevation, WorldCereal) are not re-downloaded. AlphaEarth and landcover are not wanted.
- **GEE method:** points are grouped into 5,914 one-degree cells. Per cell and per 250-point / 3-month request,
  one `reduceRegions` + `getDownloadURL(CSV)` returns a small table (`src/cropfm/data/fast/gee_extract.py`).
  One cell = one `cell_<lon>_<lat>.npz` plus a `.json` marker (or `.incomplete.json` if requests failed).
  Reruns skip finished cells and retry incomplete ones.
- **Output:** `/projects/prjs1788/CropFM/fast_dl/raw/<year>/<modality>/`; failures in `raw/failures.log`.
- **Auth:** Earth Engine service-account key at `/projects/prjs1788/CropFM/fast_dl/ee_key.json`
  (set via `EE_KEY_FILE` in `slurm_scripts/surf/fast_env.sh`; never committed). GEE project: `cropfm`.
- **SURF rule:** only ONE download job at a time. All remaining work is the single job
  `slurm_scripts/surf/slurm_fast_prod_all.sh` (AgERA5 + S2 + S1 streams, resumable, resubmits itself).

## What has finished

| Modality | Status |
|---|---|
| FAPAR | complete, 2018-2025 |
| AgERA5 | complete, 12 months for each of 2018-2025 |
| Sentinel-2, 2018 | 2,181 of 5,914 cells done (571,934 points); 1,051 cells incomplete |
| Sentinel-1, 2018 | 1,809 of 5,914 cells done (476,643 points); 1,327 cells incomplete |
| Sentinel-2/1, 2019-2025 | not started |

The GEE output was validated on one dense cell (India, 1,226 points, 2021) against the reference zarr:
98.1% weekly validity agreement, median relative difference 0.1-1.5% per band, correlation 0.98-0.998.
The extra weeks in the new data are late in the year, where the old download capped observations per point.
GEE S1 is sigma0 in dB, as in the reference zarr, so it was kept as is.

## What is left

Totals are 5,914 cells per year per sensor, so 47,312 cell-extractions per sensor over 8 years.

| Sensor | Done | Left | Of which 2018 (partly done) | Of which 2019-2025 |
|---|---|---|---|---|
| Sentinel-2 | 2,181 | 45,131 | 3,733 (1,051 incomplete + 2,682 untouched) | 41,398 |
| Sentinel-1 | 1,809 | 45,503 | 4,105 (1,327 incomplete + 2,778 untouched) | 41,398 |

Cancelled-job leftovers: about 8,100 failed requests in `raw/failures.log`, all from the quota problem
below. They are retried automatically on rerun.

## Why it stopped (2026-10-21)

The single production job (28369609) ran 28 hours, then was cancelled by us. About 2 hours after the start
(around 2026-10-20 15:00) the GEE project `cropfm` ran out of its monthly compute quota (Community tier,
150 EECU-hours). It went into Restricted Mode: requests fail with "Exceeded Earth Engine concurrency limit",
and throughput fell from about 1,300 cells/hour to about 64 cells/hour. The quota resets on the 1st of each
month at midnight Pacific time.

Rough, unmeasured estimate: 150 EECU-hours covered about half a year of S2+S1 together, so all 8 years would need
roughly 2,000-2,500 EECU-hours. The 2018 benchmark speed (S2 about 2.7 s/cell, S1 about 3.5 s/cell with
32/48 concurrent requests) suggests about 1-2 days of wall time once quota is available.

## Options before restarting

1. Switch the GEE project to the Contributor tier (1,000 EECU-hours/month, needs a billing account for
   identity, noncommercial use is not charged): https://console.cloud.google.com/earth-engine/configuration/manage-tier
2. Apply for the Partner tier (100,000 EECU-hours, several weeks, climate/nature eligibility).
3. Run Sentinel-2 first for all years and do Sentinel-1 last or for fewer years (S1 costs more per cell and is not
   used downstream).
4. Wait for the monthly reset on Nov 1.

Before restarting, measure real EECU use per cell in Cloud Monitoring so the plan uses a number, not an estimate:
https://developers.google.com/earth-engine/guides/monitoring_usage. To restart, submit
`slurm_scripts/surf/slurm_fast_prod_all.sh` (to run S2 only, remove the `gee sentinel1` line).

## Later

Post-processing: weekly aggregation into the cleaned zarr layout (`aggregate_to_weekly` in `clean_data.py`),
year-aware weekly cleaning, and a check of S1/S2 completeness per point.
