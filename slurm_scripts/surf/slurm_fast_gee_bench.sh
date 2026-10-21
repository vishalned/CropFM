#!/bin/bash
#SBATCH --job-name=gee_bench
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/gee_bench_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=03:00:00

# ONE job. Benchmarks batched Earth Engine extraction on the densest test cell (India, 1,226 pts, 2021)
# with different request shapes, to pick pts-per-request / months-per-request / concurrency.
# Needs the EE service-account key at $FAST_DL/ee_key.json (set in fast_env.sh).
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
SUB=$FAST_DL/test_subset_india.json
B=$FAST_DL/bench_gee
rm -rf $B; mkdir -p $B
echo "node: $(hostname) start: $(date)"
run() { # tag sensor pts months workers
  echo "######## $1: $2 pts/req=$3 months/req=$4 workers=$5"
  python -u -W ignore -m cropfm.data.fast.gee_extract --sensor $2 --year 2021 --points $SUB \
      --out $B/$1 --pts-per-req $3 --months-per-req $4 --workers $5 --cell-batch 1
}
run s2_A  sentinel2 250 3 16
run s2_B  sentinel2 500 1 16
run s2_C  sentinel2 125 6 16
run s2_D  sentinel2 250 3 32
run s1_A  sentinel1 250 3 16
echo "end: $(date)"
