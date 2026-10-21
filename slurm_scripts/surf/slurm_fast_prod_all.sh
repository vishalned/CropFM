#!/bin/bash
#SBATCH --job-name=fast_prod
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/prod_all_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=5-00:00:00

# PRODUCTION, the ONLY download job (SURF allows one). Three streams run inside it:
#   agera5 : remaining months via CDS, 2 workers (retries failed months)
#   s2     : Sentinel-2 via batched Earth Engine, years 2018..2025
#   s1     : Sentinel-1 via batched Earth Engine, years 2018..2025
# FAPAR is already complete. Everything is resumable (finished cells/months are skipped), and the job
# resubmits itself if it hits the 5-day limit with work left. Earth Engine only returns small CSV tables.
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
SELF=/gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/slurm_fast_prod_all.sh
YEARS="2018 2019 2020 2021 2022 2023 2024 2025"
END=$(( $(date +%s) + 5*24*3600 - 1800 ))     # stop 30 min before the limit
FLAG=$FAST_DL/logs/.timed_out_$SLURM_JOB_ID
echo "node: $(hostname) start: $(date) job: $SLURM_JOB_ID"

run() { # name, then command...; killed at the deadline, which flags a resubmit
  local name=$1; shift
  local rem=$(( END - $(date +%s) ))
  if [ $rem -lt 300 ]; then touch $FLAG; echo "[$name] no time left, skipped"; return; fi
  timeout ${rem}s "$@"; local rc=$?
  if [ $rc -eq 124 ]; then touch $FLAG; echo "[$name] hit the deadline"; fi
  echo "[$name] rc=$rc $(date)"
}
gee() { # sensor workers
  for y in $YEARS; do
    run gee-$1-$y python -u -W ignore -m cropfm.data.fast.gee_extract --sensor $1 --year $y \
        --out $FAST_DL/raw --workers $2 --cell-batch 32 --pts-per-req 250 --months-per-req 3
  done
}
agera5() {
  for y in $YEARS; do
    run agera5-$y python -u -W ignore -m cropfm.data.fast.agera5_cds --year $y --out $FAST_DL/raw --workers 2
  done
}
agera5 > $FAST_DL/logs/prod_agera5_$SLURM_JOB_ID.log 2>&1 &
gee sentinel2 32 > $FAST_DL/logs/prod_s2_$SLURM_JOB_ID.log 2>&1 &
gee sentinel1 48 > $FAST_DL/logs/prod_s1_$SLURM_JOB_ID.log 2>&1 &
wait
echo "streams finished: $(date)"
python -m cropfm.data.fast.status 2>&1 | tail -40
if [ -e $FLAG ]; then echo "deadline reached with work left -> resubmitting"; sbatch $SELF; fi
