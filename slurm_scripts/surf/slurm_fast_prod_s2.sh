#!/bin/bash
#SBATCH --job-name=prod_s2
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/prod_s2_%A_%a.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-191%20

# PRODUCTION: Sentinel-2 point extraction, all years, all 1.79M points.
# 192 array tasks = 8 years x 24 tasks. Each task runs 8 independent worker processes
# (the work is network-bound; SURF bills a minimum of 16 cores per job anyway).
# Everything is resumable: re-submitting skips finished 1-degree cells.
# Output: /projects/prjs1788/CropFM/fast_dl/raw/<year>/<sensor>/cell_*.npz|json
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
YEARS=(2018 2019 2020 2021 2022 2023 2024 2025)
NT=24      # array tasks per year
NP=8       # worker processes per array task
OUT=$FAST_DL/raw
YEAR=${YEARS[$((SLURM_ARRAY_TASK_ID / NT))]}
T=$((SLURM_ARRAY_TASK_ID % NT))
echo "node: $(hostname) year: $YEAR task: $T start: $(date)"
for p in $(seq 0 $((NP-1))); do
  python -u -W ignore -m cropfm.data.fast.s2_pc --year $YEAR --out $OUT --workers 16 \
      --task-id $((T*NP+p)) --n-tasks $((NT*NP)) \
      > /projects/prjs1788/CropFM/fast_dl/logs/s2_${YEAR}_w$((T*NP+p)).out 2>&1 &
done
wait
echo "end: $(date)"
