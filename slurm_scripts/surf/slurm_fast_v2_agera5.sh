#!/bin/bash
#SBATCH --job-name=v2_agera5
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/v2_agera5_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
OUT=$FAST_DL/test_v2
echo "node: $(hostname) start: $(date)"
python -u -W ignore -m cropfm.data.fast.agera5_cds --year 2021 --points $SUBSET --out $OUT --months 7 --workers 4
echo "end: $(date)"
