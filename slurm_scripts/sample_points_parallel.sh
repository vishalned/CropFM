#!/bin/bash

#SBATCH --job-name=sample-points
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/sampling/sample_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --array=1-15%15  # 15 jobs max, respecting GEE limit

# Parallel point sampling from Europe grids
# Total grids: 106,826
# Batch size per job: ~7,122 grids (106,826 / 15)

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Configuration
TOTAL_GRIDS=106826
NUM_JOBS=15
BATCH_SIZE=$((TOTAL_GRIDS / NUM_JOBS))
POINTS_PER_GRID=10

# Calculate start and end indices for this job
START_IDX=$(((SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE))
END_IDX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))

# Adjust the last job to include any remaining grids
if [ $SLURM_ARRAY_TASK_ID -eq $NUM_JOBS ]; then
    END_IDX=$TOTAL_GRIDS
fi

echo "=================================================="
echo "POINT SAMPLING JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Processing grids: $START_IDX to $END_IDX"
echo "Batch size: $((END_IDX - START_IDX))"
echo "Points per grid: $POINTS_PER_GRID"
echo "Expected points: ~$((((END_IDX - START_IDX)) * POINTS_PER_GRID))"
echo "Starting at: $(date)"
echo "=================================================="

cd /lustre/scratch/WUR/AIN/nedun001/CropFM

# Run the point sampling for this batch
srun python -u src/cropfm/grids/sample_points_from_grids.py \
    --input data/europe_grids_5km2.geojson \
    --points-per-grid $POINTS_PER_GRID \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --output data/sampling_batches/europe_points_batch_${SLURM_ARRAY_TASK_ID}.geojson

echo "=================================================="
echo "Completed batch $SLURM_ARRAY_TASK_ID at: $(date)"
echo "=================================================="
