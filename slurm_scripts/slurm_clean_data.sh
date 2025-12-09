#!/bin/bash

#SBATCH --job-name=clean_data
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/clean_data_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Run clean data
srun python -u src/cropfm/data/clean_data.py

