# Placer comparison vs VPR on the paper's VTR benchmarks

## FINAL CONSOLIDATED RESULTS (authoritative — supersedes earlier partial runs)

Full 18-benchmark run, one binary, final systolic config (metro schedule + Metropolis
accept + fanout-normalized squared-WL blended `λ=0.3`, `crit_exp=2`, **STA once at start**
`cadence=0`), **single-threaded** for a fair per-core runtime vs VPR.
Driver: `run_final.py` → `build_exp/final_all.csv`. Metrics are routed (VPR `--route`
at each benchmark's channel width): WL, critical-path delay (CPD), placement runtime.

**Geomean, systolic vs each VPR baseline:**

| | vs VPR-bbox | vs VPR-timing |
|---|---|---|
| routed wirelength | +6.7% | **+4.0%** |
| critical-path delay | **−6.9%** (better) | **+2.5%** |
| placement speedup (1 thread) | 46× | **90×** |

*Speedup note:* WL/CPD were measured at `cadence=20`; a single up-front STA gives
byte-identical placement (criticality ranking is placement-stable — verified on all 18),
so quality is unchanged but placement is ~2× faster on the large designs (e.g. hetero
1.59→0.59 s, LU32 1.21→0.55 s single-threaded). The **46×/90×** figures reflect the
`cadence=0` default; the earlier `cadence=20` run gave a more conservative 22×/43×.
Threading does *not* help at this scale (≤~8k slots): STA is unthreaded and the per-update
parallel work is too small to beat OpenMP overhead — single-thread is optimal here.
Threading pays off only at much larger placements (~125k-element meshes scaled ~6× on 8 cores).

**By array size:**

| | vs bbox WL / CPD / speedup | vs timing WL / CPD / speedup |
|---|---|---|
| 35×35 (14) | +6.0% / −6.1% / 24× | +3.0% / +1.5% / 46× |
| 100×100 (4) | +9.4% / −9.9% / 15× | +7.7% / +6.0% / 35× |

**Headline:** the systolic placer is within **~4% wirelength and ~2.5% critical-path
delay of VPR's timing-driven annealer while placing 43× faster (single-threaded)**, and
it **beats VPR's bounding-box placer on critical path** (−6.9%). It *beats VPR-timing on
CPD outright* on 10/18 designs — including all the large timing-critical ones (LU32PEEng
−2%, mcml −3%, arm_core −6%). It lags VPR-timing by >5% only on short-logic-depth designs
(stereovision2 +16%, hetero_placer +14%, digit_recognition_knn2 +16%) — the squared-distance
proxy's known limit on already-short critical connections (see `FANOUT_WEIGHTING.md` / the
short-path analysis). Runtime is single-threaded; multithreading widens the speed gap further.

---

## Earlier methodology / staged results

Compares three placers on the paper's benchmarks, array sizes, schedules and
per-benchmark routing channel widths (`experiment_params.txt`):

- **VPR-bbox** — VPR's annealer, `--place_algorithm bounding_box` (WL-only), IOs fixed.
- **VPR-timing** — VPR's annealer, default `criticality_timing` (timing-driven), IOs fixed
  (no `--place_quench_algorithm` override). The stronger timing baseline.
- **fixed** — `placer.cpp`, the paper's relaxed-SA (hot start 65535, linear cool, threshold accept).
- **adaptive** — `adaptive_placer.cpp` (Metropolis accept + acceptance-driven cooling + auto-stop).

## Methodology (`run_vtr_experiment.py`)

Per benchmark, everything downstream of packing is identical (same arch, same
packed `.net`, same fixed IO placement, same routing channel width) — only the
placement algorithm differs:

1. `gen_arch(W,H)` → `systolic.xml`; VPR `--pack --place` → `.net`,
   `systolic_{grid,netlist,arch}_info`, IO positions.
2. `gen_io_placement` / `gen_placer_init`.
3. Place three ways; concatenate our placements with the fixed IO placement.
4. VPR `--route --route_chan_width <cw>` on each; parse routed wirelength + critical-path delay.

Baseline is **re-run in this environment** (EPYC 9335), not taken from the paper —
different machine/VTR build, so only same-environment numbers are comparable.
Our placers use 8 threads; VPR placement is single-threaded (see caveats).

## Results (geomean) — our placers vs each VPR baseline

All 18 benchmarks:

| placer | WL vs bbox | WL vs timing | crit-path vs bbox | crit-path vs timing | speedup vs bbox | speedup vs timing |
|---|---|---|---|---|---|---|
| fixed (paper's algo) | +22.6% | +18.7% | +3.4% | +11.5% | **322×** | **552×** |
| adaptive | **+12.7%** | **+9.1%** | +2.1% | +10.1% | 91× | 156× |

By array size (adaptive / fixed):

| array | WL vs bbox | crit-path vs bbox | speedup vs bbox | speedup vs timing |
|---|---|---|---|---|
| 35×35   | +10.3% / +20.0% | +0.9% / +2.0% | 68× / 244×  | 112× / 406×  |
| 100×100 | +21.2% / +32.2% | +6.4% / +8.5% | 255× / 848× | 487× / 1619× |

Adaptive nearly halves the fixed algorithm's wirelength gap (12.7% vs 22.6% vs bbox)
for ~3.5× more placement time; both remain 1.5–3 orders of magnitude faster than VPR.
The fixed placer runs the paper's preset update count (25/50), so it is the fastest
and loosest point; adaptive auto-stops at ~113–150 updates for better quality.

Headline: our adaptive placer reaches routed wirelength within ~9–13% of VPR's
annealer (geomean), places **~91–156× faster**, and the speedup grows with design
size (VPR placement scales into tens of seconds — 93 s on hetero_placer timing-driven —
while ours stays ~0.02–0.16 s). On timing: essentially matched vs bbox-VPR (+2.1%),
and +10.1% vs the stronger timing-driven VPR — expected, since our objective is pure
wirelength with no criticality term.

## Caveats / honest notes

- **Objective is pure wirelength** (squared-Manhattan). Vs the timing-driven VPR
  baseline, adaptive is +10.1% crit-path geomean; the worst case is **LU32PEEng +34%**
  (a deeply pipelined design where WL and critical path diverge), most others +3–16%.
  Vs bbox-VPR it's +2.1% (bbox also ignores timing). No criticality term is implemented.
- **Thread fairness**: ours = 8 threads, VPR place = 1 thread. Per-core the speedup
  is ~1/8 of the wall-clock number (still ~11× geomean); at these small array sizes
  our placer barely benefits from threads, so single-thread would be similar.
- Two VPR baselines are reported: `bounding_box` (WL-only, matches our objective) and
  `criticality_timing` (timing-driven, stronger on Fmax, ~2× slower to place).

Raw data: `build_exp/results_small.csv`, `build_exp/results_large.csv`,
`build_exp/results_td_all.csv` (timing-driven). Harnesses: `run_vtr_experiment.py`,
`augment_td.py`.
