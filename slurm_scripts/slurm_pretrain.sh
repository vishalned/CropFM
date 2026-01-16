#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/pretrain_%j.out
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/pretrain_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia&A100'
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

module load GPU

# Run clean data
srun python -u src/cropfm/models/main_pretrain.py

