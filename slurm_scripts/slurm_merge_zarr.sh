#!/bin/bash

#SBATCH --job-name=merge_zarr_batches
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/merge_zarr_batches_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Merge zarr batches
srun python -u src/cropfm/data/merge_zarr_batches.py \
    --base-zarr-path /lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200.zarr \
    --total-batches 40 \
    --output-zarr-path /lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/europe_data_1200_merged.zarr

