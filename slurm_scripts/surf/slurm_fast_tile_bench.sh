#!/bin/bash
#SBATCH --job-name=tile_bench
#SBATCH --output=/projects/prjs1788/CropFM/fast_dl/logs/tile_bench_%j.out
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=03:00:00

# ONE job (SURF allows one download job at a time). Benchmarks S2 extraction on the same
# 3,875 points (3 one-degree cells: India, Africa, Europe), year 2021, sequentially:
#   1. cell-grouped (current)   2. tile-grouped (new)
# and reports wall time and bytes received on the node's external interface for each.
# NOTE: rx bytes come from /proc/net/dev of a shared node, so they include other jobs' traffic;
# an idle 20 s reading is subtracted to estimate that background.
source /gpfs/home2/vnedungadi/CropFM/slurm_scripts/surf/fast_env.sh
SUB=$FAST_DL/test_subset_3cells.json
B=$FAST_DL/bench_tiles
rm -rf $B; mkdir -p $B

IFACE=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
rx() { awk -v i="$IFACE:" '$1==i {print $2}' /proc/net/dev; }
echo "node: $(hostname)  iface: $IFACE  start: $(date)"

r0=$(rx); sleep 20; r1=$(rx); BG=$(( (r1-r0)/20 )); echo "background rx on node: $((BG/1000)) kB/s"

run_phase() { # name, command...
  local name=$1; shift
  local a=$(rx); local t=$(date +%s)
  "$@"
  local b=$(rx); local e=$(date +%s); local dt=$((e-t))
  local net=$(( (b-a) - BG*dt ))
  echo "### PHASE $name: ${dt}s  rx=$(( net/1000000 )) MB  avg=$(( net/1000/(dt>0?dt:1) )) kB/s"
}

run_phase cell_mode python -u -W ignore -m cropfm.data.fast.s2_pc --year 2021 --points $SUB --out $B/cellmode --workers 32
run_phase tile_mode python -u -W ignore -m cropfm.data.fast.s2_tiles --year 2021 --points $SUB --out $B/tilemode --workers 32

echo "=== equivalence ==="
python -W ignore -m cropfm.data.fast.compare_modes $B/cellmode/2021/sentinel2 $B/tilemode/2021/sentinel2
echo "end: $(date)"
