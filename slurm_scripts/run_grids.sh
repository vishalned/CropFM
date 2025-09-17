#!/bin/bash

#SBATCH --job-name=grid_majorTOM
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/grid_majorTOM_%j.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/grid_majorTOM_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-0:00:00




source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm


srun python -u src/cropfm/grids/grid_majorTOM.py

