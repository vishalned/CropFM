#!/bin/bash
#SBATCH --job-name=download_modalities
#SBATCH --output=/home/vnedungadi/CropFM/slurm_logs/download_modalities_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G


source ~/.bashrc
conda activate cropfm

srun python -u src/cropfm/data/download_modalities.py