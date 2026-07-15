#!/usr/bin/env python3
"""
Paper-benchmark placement study for the systolic placer: runtime/quality tradeoff
+ ablation, structured for migration to another machine.

For each benchmark (at its page size + per-benchmark channel width), run serially:

  0. gen_arch(W,H)      -> systolic.xml            (the "proper page size")
  1. VPR --pack --place -> {name}.net, io.place    (VPR timing run -> FIXED IO, shared by
                                                    every config + seed of this benchmark)
  2. for each CONFIG in the ablation ladder, RUNS seeds each:
       a. place (fixed IO, SYSTOLIC_SEED=<seed>)
       b. route at the benchmark channel width
       c. record placement time + routed WL + Fmax/CPD, then delete the run's products

Configs (systolic knobs via env; vtr* are VPR comparators). Ablation ladder = baseline..full:

  baseline : MODE=fixed  CRIT=0 FANORM=0  (+paper SWPS/UPDTS)  fixed linear-cool schedule (the PAPER config)
  metro    : MODE=metro  CRIT=0 FANORM=0                        + adaptive schedule & Metropolis accept
  timing   : MODE=metro  CRIT=1 FANORM=0                        + timing-driven (STA)
  full     : MODE=metro  CRIT=1 FANORM=1                        + inverse-fanout WL weight [default]
  tuned    : MODE=fixed 50/100  CRIT=1 FANORM=0 CRIT_EXP=1 LAMBDA=0.1   (Koios-motivated; see KOIOS_FMAX_FINDINGS.md)
  soph     : MODE=metro CRIT=1 FANORM=1 CRIT_EXP=1 LAMBDA=0.5 CADENCE=10 + VPREXP ramp 1->8 (full timing stack;
             needs the rebuilt VPR with the crit_exponent fix; beat the paper baseline ~+18% Fmax)
  vtr      : VPR criticality_timing, same fixed IO             (timing-driven comparator)
  vtr_bb   : VPR bounding_box + bounding_box quench, same IO   (wirelength-only comparator)
Note: 'soph' requires the rebuilt vpr (crit_exponent default 1.0 + cadence delay-refresh + exponent ramp
in placer.cpp/systolic_placer.cpp); against an unmodified vpr its timing knobs have no effect.

place_time_s is the reported placement time: for systolic it is sys_total_s (time inside
run_systolic: STA + weighting + annealing + setup/writeback -- "how long the accelerator
runs", excluding VPR overhead); for vtr it is VPR's "# Placement took". Same measurement
convention as the paper. All seed values are written to CSV; medians are printed.

Runs execute one at a time (no contention). --workers = VPR -j (packing/STA/routing).
Env overrides: SELF_HOSTED_ROOT, VPR, VPR_NUM_WORKERS.
"""
import os, re, time, csv, argparse, shutil, subprocess, statistics, filecmp

ROOT     = os.environ.get("SELF_HOSTED_ROOT", "/home/ethomas/research/internship/self_hosted_placement")
VPR      = os.environ.get("VPR", f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr")
PP       = f"{ROOT}/systolic_page_placer"
SCR      = f"{PP}/scripts"
TPL      = f"{PP}/vtr_integration/arch/heterogeneous_k10.xml"
SYNTH    = f"{ROOT}/soft_systolic_placer/synth"
PARAMS   = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

CSV_FIELDS = ["bench", "config", "seed", "W", "H", "cw", "threads",
              "place_time_s",     # systolic: sys_total_s | vtr: VPR "# Placement took"
              "sys_place_core_s", # systolic annealing only (blank for vtr)
              "sys_sta_s",        # systolic initial STA (blank for vtr)
              "route_wall_s",
              "routed_ok", "wl", "cpd_ns", "fmax_mhz", "error"]

LADDER = ["baseline", "metro", "timing", "full"]

# systolic env per config (SEED/THREADS added per run; fixed-mode adds paper SWPS/UPDTS).
# Cumulative ladder: adaptive schedule (metro), then timing (crit), then inverse-fanout --
# a timing-coupled refinement, cleanest applied last (timing -> full). vtr handled separately.
def sys_env(cfg):
    return {
        "baseline": {"SYSTOLIC_MODE": "fixed", "SYSTOLIC_CRIT": 0, "SYSTOLIC_FANORM": 0},
        "metro":    {"SYSTOLIC_MODE": "metro", "SYSTOLIC_CRIT": 0, "SYSTOLIC_FANORM": 0},
        "timing":   {"SYSTOLIC_MODE": "metro", "SYSTOLIC_CRIT": 1, "SYSTOLIC_FANORM": 0},
        "full":     {"SYSTOLIC_MODE": "metro", "SYSTOLIC_CRIT": 1, "SYSTOLIC_FANORM": 1},
        # recommended config from the Koios Fmax study (see KOIOS_FMAX_FINDINGS.md): fixed 50x100
        # schedule, timing on but crit_exp=1 (linear, not squared) + lambda=0.1 (heavy timing weight),
        # fanorm OFF. Strictly better than "full" on both Koios and classic arm_core.
        "tuned":    {"SYSTOLIC_MODE": "fixed", "SYSTOLIC_CRIT": 1, "SYSTOLIC_FANORM": 0,
                     "SYSTOLIC_CRIT_EXP": 1, "SYSTOLIC_LAMBDA": 0.1,
                     "SYSTOLIC_SWPS": 50, "SYSTOLIC_UPDTS": 100},
        # "soph": full timing stack developed after the crit_exponent interface-bug fix. Metro schedule,
        # timing on, fanorm on, criticality re-STA'd every 10 updates (CADENCE), with the VPR criticality
        # exponent RAMPED 1->8 over the anneal (VTR-style) and our reweight exp held at 1, lambda=0.5.
        # Requires the rebuilt VPR (VPREXP default 1.0 + cadence delay-refresh + exponent ramp). Beat the
        # paper WL-only baseline by ~+18% Fmax (same-setup, 8 designs). See crit-exponent-interface-bug memo.
        "soph":     {"SYSTOLIC_MODE": "metro", "SYSTOLIC_CRIT": 1, "SYSTOLIC_FANORM": 1,
                     "SYSTOLIC_CRIT_EXP": 1, "SYSTOLIC_LAMBDA": 0.5, "SYSTOLIC_CADENCE": 10,
                     "SYSTOLIC_VPREXP_RAMP": 1, "SYSTOLIC_VPREXP_FIRST": 1, "SYSTOLIC_VPREXP_LAST": 8},
    }[cfg]


def read_params(path):
    groups, cur = [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("W="):
            m = re.match(r"W=(\d+),H=(\d+)", line)
            cur = {"W": int(m.group(1)), "H": int(m.group(2)), "SW": 10, "UP": 25, "bench": []}
            groups.append(cur)
        elif line.startswith("SWAPS_PER_UPDATE"):
            m = re.match(r"SWAPS_PER_UPDATE=(\d+),UPDATES=(\d+)", line)
            cur["SW"], cur["UP"] = int(m.group(1)), int(m.group(2))  # paper fixed schedule (baseline rung)
        else:
            b, cw = line.split(",")
            b = b.strip()
            if not b.endswith(".blif"):   # koios params list names without the suffix
                b += ".blif"
            cur["bench"].append((b, int(cw)))
    return groups


def sh(cmd, log, cwd, env=None, timeout=10800):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e, timeout=timeout).returncode
    return rc, time.time() - t0


def _grep(pat, text, cast=float, flags=0):
    m = re.search(pat, text, flags)
    return cast(m.group(1)) if m else None


def parse_route(text):
    ok = "Circuit successfully routed" in text
    wl = _grep(r"Total wirelength:\s*(\d+)", text, int)
    cpd = (_grep(r"Final critical path delay.*?:\s*([\d.]+)\s*ns", text)
           or _grep(r"critical path delay.*?:\s*([\d.]+)\s*ns", text, float, re.I))
    fmax = _grep(r"Fmax:\s*([\d.]+)\s*MHz", text)
    return ok, wl, cpd, fmax


def parse_systolic(text):
    total = _grep(r"\[systolic\].*?cost=[-0-9]+\s+([\d.]+)\s*s", text)
    core  = _grep(r"\[systolic\] time: place ([\d.]+)s", text)
    sta   = _grep(r"\[systolic\] time:.*?STA ([\d.]+)s", text)
    return total, core, sta


def clean_run(workdir, keepset):
    for fn in os.listdir(workdir):
        if fn not in keepset:
            p = os.path.join(workdir, fn)
            try:
                os.remove(p) if os.path.isfile(p) else shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def main():
    global VPR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("VPR_NUM_WORKERS", "8")),
                    help="VPR -j: internal parallelism for packing, STA, routing")
    ap.add_argument("--runs", type=int, default=10, help="seeds per (benchmark, config)")
    ap.add_argument("--which", choices=["all", "small", "large"], default="all")
    ap.add_argument("--benchmarks", nargs="*", default=None, help="subset by name (no .blif)")
    ap.add_argument("--configs", default=None,
                    help="comma list from {baseline,metro,timing,full,tuned,soph,vtr,vtr_bb} (default: full). "
                         "'soph' = full timing stack (needs rebuilt vpr); 'baseline' = the paper config. "
                         "e.g. --configs baseline,soph,vtr")
    ap.add_argument("--ablation", action="store_true", help="run the full systolic ladder")
    ap.add_argument("--vtr", action="store_true", help="also run the VTR timing-driven comparator")
    ap.add_argument("--vtr-bb", dest="vtr_bb", action="store_true",
                    help="also run the VTR bounding-box comparator (bounding_box place + quench)")
    ap.add_argument("--workdir", default=f"{ROOT}/soft_systolic_placer/bench_work")
    ap.add_argument("--out", default=None, help="CSV path (default <workdir>/results.csv)")
    ap.add_argument("--keep", action="store_true", help="keep VTR products + rr-graph caches (debug)")
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--vpr", default=VPR, help="path to the vpr binary with the systolic placer")
    ap.add_argument("--systolic-threads", type=int, default=1, help="SYSTOLIC_THREADS for the systolic placer")
    ap.add_argument("--systolic-thread-sweep", dest="thread_sweep", default=None,
                    help="comma list of SYSTOLIC_THREADS to sweep for systolic configs, e.g. 1,2,4,8. "
                         "Placement is thread-invariant, so each seed is placed once per T (anneal scaling, "
                         "-j fixed at --workers) but ROUTED ONCE; placements are verified byte-identical across T.")
    ap.add_argument("--no-cache", dest="cache", action="store_false", help="disable rr-graph/delay-lookup caching")
    ap.add_argument("--params", default=PARAMS, help="benchmark params file (page size + per-benchmark cw)")
    ap.add_argument("--synth", default=SYNTH, help="directory containing the .blif files")
    args = ap.parse_args()
    VPR = args.vpr
    args.workdir = os.path.abspath(args.workdir)  # VPR runs with cwd=benchmark dir -> paths must be absolute
    args.params, args.synth = os.path.abspath(args.params), os.path.abspath(args.synth)

    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = (LADDER if args.ablation else ["full"]) + (["vtr"] if args.vtr else []) + (["vtr_bb"] if args.vtr_bb else [])
    SWEEP = [int(x) for x in args.thread_sweep.split(",") if x.strip()] if args.thread_sweep else None
    print(f"vpr = {VPR}\nconfigs = {configs}" + (f"\nthread sweep = {SWEEP}" if SWEEP else ""), flush=True)

    os.makedirs(args.workdir, exist_ok=True)
    cache = f"{args.workdir}/_cache"
    if args.cache:
        os.makedirs(cache, exist_ok=True)
    def place_cache(W, H): return f"{cache}/place_{W}x{H}.rr.bin", f"{cache}/place_{W}x{H}.dl"
    def route_cache(name): return f"{cache}/route_{name}.rr.bin"

    out = args.out or f"{args.workdir}/results.csv"
    new = not os.path.exists(out)
    csvf = open(out, "a", newline="")
    wr = csv.DictWriter(csvf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if new:
        wr.writeheader(); csvf.flush()

    for gi, g in enumerate(read_params(args.params)):
        if args.which == "small" and gi != 0: continue
        if args.which == "large" and gi != 1: continue
        W, H, SW, UP = g["W"], g["H"], g["SW"], g["UP"]
        for bench, cw in g["bench"]:
            name = bench[:-5] if bench.endswith(".blif") else bench
            if args.benchmarks and name not in args.benchmarks: continue
            blif = f"{args.synth}/{bench}"
            if not os.path.exists(blif):
                print(f"[skip] {name}: no blif at {blif}", flush=True); continue
            w = f"{args.workdir}/{name}"
            shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
            shutil.copy(blif, f"{w}/{bench}")
            if args.cache and not args.keep:   # bound disk to one benchmark's graphs
                for fn in os.listdir(cache):
                    try: os.remove(os.path.join(cache, fn))
                    except OSError: pass
            J = ["-j", str(args.workers)]
            print(f"[{time.strftime('%H:%M:%S')}] {name}  page={W}x{H} cw={cw}", flush=True)

            # ---- 0. arch ----
            rc, _ = sh(["python3", f"{SCR}/gen_arch.py", str(W), str(H), TPL, f"{w}/systolic.xml"],
                       f"{w}/gen_arch.log", w, timeout=300)
            if rc != 0:
                wr.writerow({"bench": name, "config": "-", "W": W, "H": H, "cw": cw, "error": "gen_arch"}); csvf.flush(); continue

            # ---- 1. timing VPR run -> fixed IO (also writes the per-page-size place cache) ----
            prr, pdl = place_cache(W, H)
            pc = (["--read_rr_graph", prr, "--read_placement_delay_lookup", pdl, "--route_chan_width", str(cw)]
                  if args.cache and os.path.exists(prr) and os.path.exists(pdl)
                  else (["--write_rr_graph", prr, "--write_placement_delay_lookup", pdl] if args.cache else []))
            rc, _ = sh([VPR, "systolic.xml", bench, "--pack", "--place", *pc, *J], f"{w}/io_timing.log", w, timeout=args.timeout)
            if rc != 0:
                wr.writerow({"bench": name, "config": "-", "W": W, "H": H, "cw": cw, "error": "pack_place"}); csvf.flush()
                if not args.keep: shutil.rmtree(w, ignore_errors=True)
                continue
            rc, _ = sh(["python3", f"{SCR}/gen_io_placement.py", f"{w}/{name}.place",
                        f"{w}/systolic_netlist_info", f"{w}/io.place"], f"{w}/gen_io.log", w, timeout=300)
            if rc != 0 or not os.path.exists(f"{w}/io.place"):
                wr.writerow({"bench": name, "config": "-", "W": W, "H": H, "cw": cw, "error": "gen_io"}); csvf.flush()
                if not args.keep: shutil.rmtree(w, ignore_errors=True)
                continue

            keepset = {bench, "systolic.xml", f"{name}.net", "io.place",
                       "systolic_netlist_info", "systolic_grid_info", "systolic_arch_info"}
            pc_read = (["--read_rr_graph", prr, "--read_placement_delay_lookup", pdl, "--route_chan_width", str(cw)]
                       if args.cache and os.path.exists(prr) and os.path.exists(pdl) else [])
            rrt = route_cache(name)

            # ---- 2. configs x seeds ----
            for cfg in configs:
                is_vtr = cfg.startswith("vtr")
                vtr_algo = (["--place_algorithm", "bounding_box", "--place_quench_algorithm", "bounding_box"]
                            if cfg == "vtr_bb" else ["--place_algorithm", "criticality_timing"])
                # systolic thread counts to sweep (placement is thread-invariant -> route once per seed).
                # vtr never sweeps: its parallelism is -j workers, not SYSTOLIC_THREADS.
                tlist = (SWEEP if (SWEEP and not is_vtr) else [args.systolic_threads])
                fmaxes, wls = [], []
                ptimes = {t: [] for t in tlist}   # place_time per thread count
                for seed in range(args.runs):
                    # per-T placement results; route ONCE (placement identical across T)
                    ptime, core, sta = {}, {}, {}
                    route_pf = None
                    try:
                        if is_vtr:
                            pf, plog = f"{w}/{cfg}_s{seed}.place", f"{w}/pl_{cfg}_s{seed}.log"
                            sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                                *vtr_algo, "--fix_clusters", "io.place",
                                "--seed", str(seed + 1), "--place_file", pf, *pc_read, *J], plog, w, timeout=args.timeout)
                            T0 = tlist[0]
                            ptime[T0] = _grep(r"# Placement took ([\d.]+) seconds", open(plog).read())
                            route_pf = pf
                        else:
                            for T in tlist:   # anneal threads scale; -j (STA) held at --workers
                                pf, plog = f"{w}/{cfg}_s{seed}_t{T}.place", f"{w}/pl_{cfg}_s{seed}_t{T}.log"
                                env = {"SYSTOLIC_SEED": seed, "SYSTOLIC_THREADS": T}
                                env.update(sys_env(cfg))
                                if env["SYSTOLIC_MODE"] == "fixed" and "SYSTOLIC_SWPS" not in env:
                                    env["SYSTOLIC_SWPS"], env["SYSTOLIC_UPDTS"] = SW, UP   # paper fixed schedule (baseline)
                                sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                                    "--place_algorithm", "systolic", "--fix_clusters", "io.place",
                                    "--place_file", pf, *pc_read, *J], plog, w, env=env, timeout=args.timeout)
                                ptime[T], core[T], sta[T] = parse_systolic(open(plog).read())
                                if route_pf is None:
                                    route_pf = pf
                                elif not filecmp.cmp(pf, route_pf, shallow=False):
                                    print(f"    WARN: {cfg} s{seed}: placement at t{T} differs from t{tlist[0]} "
                                          f"(NOT thread-invariant!)", flush=True)
                        # ---- route ONCE; reuse Fmax/WL across all thread counts ----
                        rlog = f"{w}/rt_{cfg}_s{seed}.log"
                        rc_ = (["--read_rr_graph", rrt] if args.cache and os.path.exists(rrt)
                               else (["--write_rr_graph", rrt] if args.cache else []))
                        _, route_wall = sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net",
                                            "--place_file", route_pf, "--route", "--route_chan_width", str(cw), *rc_, *J],
                                           rlog, w, timeout=args.timeout)
                        routed_ok, wl, cpd, fmax = parse_route(open(rlog).read())
                        if fmax: fmaxes.append(fmax)
                        if wl: wls.append(wl)
                        for T in tlist:
                            wr.writerow({"bench": name, "config": cfg, "seed": seed, "W": W, "H": H, "cw": cw,
                                         "threads": (args.workers if is_vtr else T),
                                         "place_time_s": ptime.get(T),
                                         "sys_place_core_s": (None if is_vtr else core.get(T)),
                                         "sys_sta_s": (None if is_vtr else sta.get(T)),
                                         "route_wall_s": route_wall, "routed_ok": routed_ok,
                                         "wl": wl, "cpd_ns": cpd, "fmax_mhz": fmax})
                            if ptime.get(T): ptimes[T].append(ptime[T])
                        csvf.flush()
                    except Exception as ex:
                        wr.writerow({"bench": name, "config": cfg, "seed": seed, "W": W, "H": H,
                                     "cw": cw, "error": type(ex).__name__}); csvf.flush()
                    if not args.keep: clean_run(w, keepset)
                if len(tlist) > 1:   # thread-sweep: one line per T (Fmax/WL constant across T)
                    print(f"    {cfg:9s}: median Fmax {_median(fmaxes):7.2f} MHz | median WL {_median(wls):9.0f} "
                          f"(n={len(fmaxes)})", flush=True)
                    for T in tlist:
                        print(f"      t{T:<3d}: median place {_median(ptimes[T]):.4f}s", flush=True)
                else:
                    T = tlist[0]
                    print(f"    {cfg:9s}: median Fmax {_median(fmaxes):7.2f} MHz | median WL {_median(wls):9.0f} | "
                          f"median place {_median(ptimes[T]):.4f}s  (n={len(fmaxes)})", flush=True)

            if not args.keep:
                shutil.rmtree(w, ignore_errors=True)

    csvf.close()
    if args.cache and not args.keep:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
