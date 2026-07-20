#!/bin/bash
# Run the >=100x speedup config (fast_h1) on the lab machine, single-threaded anneal,
# vs a VTR baseline. Mirrors cluster/run_design.sh but runs sequentially in the
# foreground (no bsub) so wall-clock timing isn't contended by other jobs sharing the box.
#
# usage: scripts/lab_test_fast_h1.sh [runs] [tag] [configs] [sta_workers]
#   runs        seeds per (benchmark, config)              default 3
#   tag         output prefix -> cluster/<tag>_<bench>.csv  default lab_h1
#   configs     comma-separated harness configs             default vtr,fast_h1
#   sta_workers VPR -j / --workers (STA + packing + routing) default 1
#
# NOTE on sta_workers>1: this only multithreads STA (Tatum) if the vpr binary you're
# running is actually linked against TBB (TATUM_EXECUTION_ENGINE=auto silently falls
# back to SERIAL if TBB wasn't found at build time -- the cluster build has no TBB, so
# -j there only parallelizes packing/routing, not STA). Check before trusting the
# threaded-STA numbers:
#   grep -i 'TBB_DIR' <build_dir>/CMakeCache.txt   # TBB_DIR-NOTFOUND => no TBB, rebuild needed
#   ldd <path_to_vpr> | grep -i tbb                # empty => not linked
# The anneal itself (SYSTOLIC_THREADS) stays pinned at 1 regardless -- that's the
# speedup story we're validating, independent of STA threading.
set -e
cd "$(dirname "$0")/.."   # soft_systolic_placer/

RUNS="${1:-3}"
TAG="${2:-lab_h1}"
CFGS="${3:-vtr,fast_h1}"
STA_WORKERS="${4:-1}"

# Anneal stays single-threaded on BOTH sides for the clean speedup comparison.
# (SYSTOLIC_THREADS unset => omp_set_num_threads never called => anneal grabs
# all cores by default -- must pin explicitly.)
export SYSTOLIC_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p cluster
DESIGNS=$(grep -vE '^W=|^SWAPS|^[[:space:]]*$' koios2_params.txt | cut -d, -f1)
N=$(echo "$DESIGNS" | wc -l)
echo "=== lab_test_fast_h1: $N designs, runs=$RUNS, configs=$CFGS, tag=$TAG, sta_workers=$STA_WORKERS ==="

i=0
for d in $DESIGNS; do
  i=$((i+1))
  echo "[$i/$N] $d ..."
  python3 run_paper_bench.py --params koios2_params.txt --synth koios2_synth \
      --benchmarks "$d" --configs "$CFGS" --runs "$RUNS" --workers "$STA_WORKERS" --no-cache \
      --workdir "cluster/wk_${TAG}_$d" --out "cluster/${TAG}_$d.csv"
done

echo
echo "=== done. aggregating ==="
python3 scripts/plot_speedup_vs_cells.py --tag "$TAG" --configs "$(echo "$CFGS" | tr ',' '\n' | grep -v '^vtr$' | paste -sd,)"
