#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --output=/home/vnedungadi/CropFM/slurm_logs/pretrain_contrastive_no_cls_%j.out
#SBATCH --error=/home/vnedungadi/CropFM/slurm_logs/pretrain_contrastive_no_cls_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --constraint=scratch-node

source ~/.bashrc
conda activate cropfm

module load 2024 CUDA/12.6.0

# Debug: Check GPU visibility
echo "=== GPU Debug Info ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi || echo "nvidia-smi not available"
echo "Checking PyTorch CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "PyTorch check failed"
echo "======================"

## COPY DATA TO SCRATCH NODE
SCRATCH_DIR="/scratch-node/${USER}.${SLURM_JOB_ID}"
ZARR_FILE="europe_data_1200_merged_cleaned2_rechunked.zarr"

# Source zarr path (can be set via environment variable or use default)
SOURCE_ZARR_PATH=${CROPFM_DATA_DIR}/europe_data_1200_merged_cleaned2_rechunked.zarr # global is at /lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/global_data_1200_merged_cleaned_rechunked.zarr
SOURCE_NORM_PATH=${CROPFM_DATA_DIR}/normalization_stats_europe.json # global is at /lustre/scratch/WUR/AIN/nedun001/CropFM/data/zarr_files/normalization_stats_global.json

echo "Copying zarr data from ${SOURCE_ZARR_PATH} to ${SCRATCH_DIR}..."
cp -r ${SOURCE_ZARR_PATH} ${SCRATCH_DIR}
echo "Copying normalization stats from ${SOURCE_NORM_PATH} to ${SCRATCH_DIR}..."
cp -r ${SOURCE_NORM_PATH} ${SCRATCH_DIR}


## change CROPFM_DATA_DIR to SCRATCH_DIR
export CROPFM_DATA_DIR=${SCRATCH_DIR}

# Run pretraining
python -u src/cropfm/models/main_pretrain.py \
    --config-name pretrain.contrastive \
    hydra.run.dir=experiments/pretrain/contrastive_run_no_cls \
    callbacks.callbacks_list=[checkpoint,real_time_memory,model_summary,wandb]

