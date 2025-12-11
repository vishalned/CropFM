#!/bin/bash

#SBATCH --job-name=run_norm_stats
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/run_norm_stats_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Run norm stats
srun python -u src/cropfm/data/compute_norm_stats.py

