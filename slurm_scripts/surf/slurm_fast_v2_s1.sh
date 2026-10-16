#!/bin/bash
#SBATCH --job-name=v2_s1
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/v2_s1_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
OUT=$FAST_DL/test_v2
echo "node: $(hostname) start: $(date)"
python -u -W ignore -m cropfm.data.fast.s1_pc --year 2021 --points $SUBSET --out $OUT --workers 32 --task-id 4 --n-tasks 6
echo "end: $(date)"
