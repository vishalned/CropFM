#!/bin/bash
#SBATCH --job-name=prod_fapar
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/prod_fapar_%A_%a.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7

# PRODUCTION: FAPAR/LAI (NOAA VIIRS) for one year per array task. Resumable per month.
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
YEARS=(2018 2019 2020 2021 2022 2023 2024 2025)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
echo "node: $(hostname) year: $YEAR start: $(date)"
python -u -W ignore -m cropfm.data.fast.fapar_noaa --year $YEAR --out $FAST_DL/raw --workers 16
echo "end: $(date)"
