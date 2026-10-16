#!/bin/bash
#SBATCH --job-name=fast_compare
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/compare_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# Compare new downloads (test subset) with the existing GEE-derived zarr. Override with e.g.
#   sbatch --export=ALL,MODS="fapar sentinel2",OBS=/projects/prjs1788/CropFM/fast_dl/test_v2 slurm_fast_compare.sh
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
OBS=${OBS:-$FAST_DL/test2021}
python -u -W ignore -m cropfm.data.fast.compare --obs-dir $OBS --points $SUBSET --year ${YEAR:-2021} \
    --modalities ${MODS:-agera5 fapar sentinel1 sentinel2} --report $OBS/compare_report_${SLURM_JOB_ID}.json
