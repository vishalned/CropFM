#!/bin/bash
#SBATCH --job-name=download_modalities
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/download_modalities_%A_%a.out
#SBATCH --time=8-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-39

# This script runs 40 parallel jobs, each processing one batch of points
# Each job creates its own zarr file: <zarr_path>_batch_XXX.zarr

source ~/.bashrc
conda activate cropfm

# Change to project directory
cd /lustre/scratch/WUR/AIN/nedun001/CropFM

# Run the download script
# SLURM_ARRAY_TASK_ID will be automatically set (0-39)
python -u src/cropfm/data/download_modalities_slurm.py

