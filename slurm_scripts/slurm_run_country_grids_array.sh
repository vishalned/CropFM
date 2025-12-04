#!/bin/bash

#SBATCH --job-name=create-grids
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/grids/create-grids_%A_%a.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/CropFM/slurm_logs/grids/create-grids_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --array=1-286%15  # Process all 286 countries, max 15 concurrent jobs # 286 is the actual number of countries

# Respect GEE's ~15 concurrent user limit
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cropfm

# Get country list
COUNTRY_FILE="/lustre/scratch/WUR/AIN/nedun001/CropFM/data/country_list.txt"

# Get the country for this array task
COUNTRY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COUNTRY_FILE)

if [ -z "$COUNTRY" ]; then
    echo "No country found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing country: $COUNTRY"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting at: $(date)"

# Change to project directory
cd /lustre/scratch/WUR/AIN/nedun001/CropFM

# Run the grid creation for this country
srun python -u src/cropfm/grids/create_grids.py --country "$COUNTRY"

echo "Completed processing for $COUNTRY at: $(date)"