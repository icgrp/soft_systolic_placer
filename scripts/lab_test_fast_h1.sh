#!/bin/bash
# Run the >=100x speedup config (fast_h1) on the lab machine, single-threaded,
# vs a single-threaded VTR baseline. Mirrors cluster/run_design.sh but runs
# sequentially in the foreground (no bsub) so wall-clock timing isn't
# contended by other jobs sharing the box.
#
# usage: scripts/lab_test_fast_h1.sh [runs] [tag] [configs]
#   runs    seeds per (benchmark, config)              default 3
#   tag     output prefix -> cluster/<tag>_<bench>.csv  default lab_h1
#   configs comma-separated harness configs             default vtr,fast_h1
set -e
cd "$(dirname "$0")/.."   # soft_systolic_placer/

RUNS="${1:-3}"
TAG="${2:-lab_h1}"
CFGS="${3:-vtr,fast_h1}"

# Same knobs validated on the cluster: force single-threaded on BOTH sides.
# (SYSTOLIC_THREADS unset => omp_set_num_threads never called => anneal grabs
# all cores by default -- must pin explicitly.)
export SYSTOLIC_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p cluster
DESIGNS=$(grep -vE '^W=|^SWAPS|^[[:space:]]*$' koios2_params.txt | cut -d, -f1)
N=$(echo "$DESIGNS" | wc -l)
echo "=== lab_test_fast_h1: $N designs, runs=$RUNS, configs=$CFGS, tag=$TAG ==="

i=0
for d in $DESIGNS; do
  i=$((i+1))
  echo "[$i/$N] $d ..."
  python3 run_paper_bench.py --params koios2_params.txt --synth koios2_synth \
      --benchmarks "$d" --configs "$CFGS" --runs "$RUNS" --workers 1 --no-cache \
      --workdir "cluster/wk_${TAG}_$d" --out "cluster/${TAG}_$d.csv"
done

echo
echo "=== done. aggregating ==="
python3 scripts/plot_speedup_vs_cells.py --tag "$TAG" --configs "$(echo "$CFGS" | tr ',' '\n' | grep -v '^vtr$' | paste -sd,)"
