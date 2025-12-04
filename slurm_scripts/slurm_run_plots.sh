#!/bin/bash

#SBATCH --job-name=run_plots
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/run_plots_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Run plots
srun python -u analysis/plot_distribution.py

