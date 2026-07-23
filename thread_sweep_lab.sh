#!/usr/bin/env bash
# Thread-scaling sweep for the lab machine (up to 12 threads, real TBB/OMP hardware -- cluster
# nodes here are shared/virtualized and not representative). Uses the "default" config (hold=4 +
# adaptive_swps=1 + SWPS=10, all baked into its own dict in run_paper_bench.py -- see TODO_IDEAS.md's
# top-of-file callout) throughout, not swept, since it's already the confirmed production default;
# this sweep is purely about how well the placer's own parallelism (swap phases, lock-free
# red-black) scales with cores.
# Run this ON THE LAB MACHINE, not via bsub -- it's meant to reflect real hardware, not the cluster.
# VTR's own worker threads (--workers, packing/STA/routing's -j) are held FIXED at 12 (the full
# machine) throughout, deliberately decoupled from SYSTOLIC_THREADS -- this isolates the systolic
# placer's own thread scaling from VTR's, instead of conflating both axes at once.
#
# RUNS trials per (design, thread count) -- run_paper_bench.py already varies the seed per trial
# (VPR --seed = trial index + 1 for the vtr comparator; SYSTOLIC_SEED likewise for our placer), so a
# real spread falls out of --runs without any extra plumbing here.
#
# Also runs a one-time plain-VTR ("vtr" config, VPR's own timing-driven default annealer -- the
# project's standard baseline comparator) reference at the end, same RUNS trials, same fixed
# VTR_WORKERS. It's pulled OUT of the thread loop deliberately: VTR's own anneal doesn't read
# SYSTOLIC_THREADS at all, so running it inside the loop would just repeat identical work at every
# thread value for no new signal. Its rows are tagged with the sentinel "vtr" instead of a thread
# count so they're easy to pick out/exclude when plotting the systolic scaling curve.
#
# Usage: ./thread_sweep_lab.sh [threads...]
#   e.g. ./thread_sweep_lab.sh 1 4 12
set -u
cd "$(dirname "$0")" || exit 2

VTR_WORKERS=12
RUNS=3

THREADS=("$@")
[ ${#THREADS[@]} -eq 0 ] && THREADS=(1 4 12)

# A handful of designs spanning the size range seen this session (not the full 18 -- this is a
# scaling characterization, not a quality validation run; use run_paper_bench.py directly for that).
# --benchmarks takes nargs="*" (space-separated argv items, NOT a comma-joined string) -- array, not a string.
DESIGNS=(spmv bnn proxy.6 dnnweaver)

OUT=cluster/thread_sweep_lab.csv
rm -f "$OUT"

for T in "${THREADS[@]}"; do
    echo "=== threads=$T (VTR workers=$VTR_WORKERS, runs=$RUNS) ===" >&2
    export SYSTOLIC_THREADS="$T"
    export OMP_NUM_THREADS="$T"
    TAG="threadsweep_t${T}"
    python3 run_paper_bench.py --params koios2_params.txt --synth koios2_synth \
        --benchmarks "${DESIGNS[@]}" --configs default --runs "$RUNS" --workers "$VTR_WORKERS" --no-cache \
        --workdir "cluster/wk_${TAG}" --out "cluster/${TAG}.csv"
    # tag each row with the thread count and append to the combined CSV
    if [ ! -f "$OUT" ]; then head -1 "cluster/${TAG}.csv" | sed 's/^/threads,/' > "$OUT"; fi
    tail -n +2 "cluster/${TAG}.csv" | sed "s/^/${T},/" >> "$OUT"
done

echo "=== vtr reference (workers=$VTR_WORKERS, runs=$RUNS) ===" >&2
python3 run_paper_bench.py --params koios2_params.txt --synth koios2_synth \
    --benchmarks "${DESIGNS[@]}" --configs vtr --runs "$RUNS" --workers "$VTR_WORKERS" --no-cache \
    --workdir "cluster/wk_threadsweep_vtr" --out "cluster/threadsweep_vtr.csv"
tail -n +2 "cluster/threadsweep_vtr.csv" | sed "s/^/vtr,/" >> "$OUT"

echo "Wrote $OUT" >&2
column -s, -t "$OUT"
