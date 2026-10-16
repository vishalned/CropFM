# sourced by the fast-download slurm scripts
source ~/.bashrc
conda activate cropfm-dl
export PYTHONPATH=/gpfs/home2/vnedungadi/CropFM/src:$PYTHONPATH
export FAST_DL=/projects/prjs1788/CropFM/fast_dl        # all data + logs live in the project dir
export SUBSET=$FAST_DL/test_subset.json
export TMPDIR=${TMPDIR:-/scratch-local/$USER.$SLURM_JOB_ID}   # node-local scratch for transient files
mkdir -p $FAST_DL/logs "$TMPDIR" 2>/dev/null
