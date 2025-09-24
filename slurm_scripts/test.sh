#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/grid_majorTOM_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00




source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm


srun python -u tmp.py

