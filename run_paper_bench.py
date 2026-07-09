#!/usr/bin/env python3
"""
Paper-benchmark placement study for the systolic placer, structured for migration
to another machine.

For each paper benchmark (at its page size + per-benchmark channel width), run
strictly serially:

  0. gen_arch(W,H)                 -> systolic.xml            (the "proper page size")
  1. VPR --pack --place            -> {name}.net, {name}.place, systolic_netlist_info
       (VPR's default timing-driven placement; this is the "timing VTR run" that
        determines the fixed IO locations)
  2. gen_io_placement              -> io.place                (IO fixed from step 1,
        shared by all RUNS of this benchmark)
  3. RUNS times, a different seed each:
       a. systolic place  (adaptive-metro DEFAULTS, 1 thread, SYSTOLIC_SEED=<run>)
       b. route at the benchmark channel width
       c. record placement time + routed Fmax, then delete this run's VTR products

Only per-run runtime + Fmax are kept (streamed to CSV). No VTR output products are
retained. Runs and benchmarks execute one at a time so machine resources are never
in contention; --workers controls VTR-INTERNAL parallelism only (packing, STA,
routing) via VPR's -j, which matters most on the TBB-enabled machine.

Env overrides: SELF_HOSTED_ROOT, VTR_ROOT, VPR_NUM_WORKERS.
"""
import os, re, sys, time, csv, argparse, shutil, subprocess, statistics

ROOT     = os.environ.get("SELF_HOSTED_ROOT", "/home/ethomas/research/internship/self_hosted_placement")
# The systolic placer lives in ROOT/vtr-verilog-to-routing; default there (NOT $VTR_ROOT,
# which may point at a different tree). Override with --vpr or $VPR.
VPR      = os.environ.get("VPR", f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr")
PP       = f"{ROOT}/systolic_page_placer"
SCR      = f"{PP}/scripts"
TPL      = f"{PP}/vtr_integration/arch/heterogeneous_k10.xml"
SYNTH    = f"{ROOT}/soft_systolic_placer/synth"
PARAMS   = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

# The reported systolic placement time is sys_total_s: the time INSIDE run_systolic
# (initial STA + weighting + annealing + a small adjacency-build/writeback). It excludes
# all VPR overhead (delay-lookup build, post-place timing reports, netlist load) -- i.e.
# it is "how long it takes to run the accelerator", the systolic analogue of what the
# paper reports for VTR. sys_place_core_s / sys_sta_s / sys_setup_wb_s decompose it, so
# you can report annealing-only, annealing+STA, or total-minus-setup as you prefer.
CSV_FIELDS = ["bench", "seed", "W", "H", "cw",
              "sys_total_s",      # REPORTED placement time: STA + weighting + annealing (+ setup/writeback)
              "sys_place_core_s", # annealing only (gather+swaps)
              "sys_sta_s",        # the single initial Tatum STA
              "sys_setup_wb_s",   # adjacency build + final writeback (subtract if you want anneal+STA+weight only)
              "route_wall_s",
              "routed_ok", "wl", "cpd_ns", "fmax_mhz", "error"]


def read_params(path):
    groups, cur = [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("W="):
            m = re.match(r"W=(\d+),H=(\d+)", line)
            cur = {"W": int(m.group(1)), "H": int(m.group(2)), "bench": []}
            groups.append(cur)
        elif line.startswith("SWAPS_PER_UPDATE"):
            pass  # paper hardware schedule; the software placer uses adaptive defaults
        else:
            b, cw = line.split(",")
            cur["bench"].append((b.strip(), int(cw)))
    return groups


def sh(cmd, log, cwd, env=None, timeout=10800):
    """Run cmd, tee to log, return (returncode, wall_seconds)."""
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=cwd, env=e, timeout=timeout).returncode
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
    total   = _grep(r"\[systolic\].*?cost=[-0-9]+\s+([\d.]+)\s*s", text)
    place   = _grep(r"\[systolic\] time: place ([\d.]+)s", text)
    sta     = _grep(r"\[systolic\] time:.*?STA ([\d.]+)s", text)
    setupwb = _grep(r"setup\+writeback ([\d.]+)s", text)
    return total, place, sta, setupwb


KEEP_EXT = None  # set per benchmark to the shared-input filenames


def clean_run(workdir, keepset):
    for fn in os.listdir(workdir):
        if fn not in keepset:
            p = os.path.join(workdir, fn)
            try:
                os.remove(p) if os.path.isfile(p) else shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass


def main():
    global VPR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("VPR_NUM_WORKERS", "8")),
                    help="VPR -j: internal parallelism for packing, STA, routing")
    ap.add_argument("--runs", type=int, default=10, help="seeds per benchmark")
    ap.add_argument("--which", choices=["all", "small", "large"], default="all")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="subset by name (no .blif), e.g. sha bgm  (for quick testing)")
    ap.add_argument("--workdir", default=f"{ROOT}/soft_systolic_placer/bench_work")
    ap.add_argument("--out", default=None, help="CSV path (default <workdir>/results.csv)")
    ap.add_argument("--keep", action="store_true", help="keep VTR products + rr-graph caches (debug)")
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--vpr", default=VPR, help="path to the vpr binary with the systolic placer")
    ap.add_argument("--systolic-threads", type=int, default=1,
                    help="SYSTOLIC_THREADS for the systolic placer (1 = fair per-core; >1 uses the persistent OpenMP team)")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="disable rr-graph/delay-lookup caching (rebuild every run, default VPR behavior)")
    args = ap.parse_args()
    VPR = args.vpr
    # VPR runs with cwd set to each benchmark subdir, so all paths we hand it must be
    # absolute (place files, rr-graph caches, delay lookups).
    args.workdir = os.path.abspath(args.workdir)
    print(f"vpr = {VPR}", flush=True)

    os.makedirs(args.workdir, exist_ok=True)
    # rr-graph / delay-lookup cache. Default VPR behavior is preserved (native 4*num_pins
    # width for the place delay model, cw for routing) -- we only avoid recomputing the
    # SAME graph/lookup repeatedly. The place delay model is arch-only (identical for every
    # same-page-size benchmark) -> cache per page size; the routing graph is at the
    # per-benchmark channel width -> cache per benchmark (reused across its seeds).
    cache = f"{args.workdir}/_cache"
    if args.cache:
        os.makedirs(cache, exist_ok=True)
    def place_cache(W, H):  # native-width delay-model graph + delta lookup (per page size)
        return f"{cache}/place_{W}x{H}.rr.bin", f"{cache}/place_{W}x{H}.dl"
    def route_cache(name):  # routing graph at this benchmark's channel width (per benchmark)
        return f"{cache}/route_{name}.rr.bin"
    out = args.out or f"{args.workdir}/results.csv"
    new = not os.path.exists(out)
    csvf = open(out, "a", newline="")
    wr = csv.DictWriter(csvf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if new:
        wr.writeheader(); csvf.flush()

    groups = read_params(PARAMS)
    for gi, g in enumerate(groups):
        if args.which == "small" and gi != 0: continue
        if args.which == "large" and gi != 1: continue
        W, H = g["W"], g["H"]
        for bench, cw in g["bench"]:
            name = bench[:-5] if bench.endswith(".blif") else bench
            if args.benchmarks and name not in args.benchmarks:
                continue
            blif = f"{SYNTH}/{bench}"
            if not os.path.exists(blif):
                print(f"[skip] {name}: no blif at {blif}", flush=True); continue
            w = f"{args.workdir}/{name}"
            shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
            shutil.copy(blif, f"{w}/{bench}")
            # Start each benchmark with an empty cache so disk stays bounded to ONE
            # benchmark's graphs (~2 GB peak for 100x100), not the whole run. This trades
            # cross-benchmark reuse of the place graph for disk safety; within a benchmark
            # the graph is still built once and reused across all its seeds.
            if args.cache and not args.keep:
                for fn in os.listdir(cache):
                    try: os.remove(os.path.join(cache, fn))
                    except OSError: pass
            J = ["-j", str(args.workers)]
            print(f"[{time.strftime('%H:%M:%S')}] {name}  page={W}x{H} cw={cw} workers={args.workers}", flush=True)

            # ---- 0. arch (proper page size) ----
            rc, _ = sh(["python3", f"{SCR}/gen_arch.py", str(W), str(H), TPL, f"{w}/systolic.xml"],
                       f"{w}/gen_arch.log", w, timeout=300)
            if rc != 0:
                wr.writerow({"bench": name, "W": W, "H": H, "cw": cw, "error": "gen_arch"}); csvf.flush(); continue

            # ---- 1. timing VPR run: pack + timing-driven place ----
            # Also seeds/uses the per-page-size place delay-model cache: the FIRST
            # same-size benchmark writes it (native width), later ones read it.
            prr, pdl = place_cache(W, H)
            pc = []
            if args.cache:
                if os.path.exists(prr) and os.path.exists(pdl):
                    pc = ["--read_rr_graph", prr, "--read_placement_delay_lookup", pdl,
                          "--route_chan_width", str(cw)]  # width is a formality; loaded graph is native
                else:
                    pc = ["--write_rr_graph", prr, "--write_placement_delay_lookup", pdl]
            rc, _ = sh([VPR, "systolic.xml", bench, "--pack", "--place", *pc, *J],
                       f"{w}/io_timing.log", w, timeout=args.timeout)
            if rc != 0:
                wr.writerow({"bench": name, "W": W, "H": H, "cw": cw, "error": "pack_place"}); csvf.flush()
                if not args.keep: shutil.rmtree(w, ignore_errors=True)
                continue

            # ---- 2. fix IO from that timing placement ----
            rc, _ = sh(["python3", f"{SCR}/gen_io_placement.py",
                        f"{w}/{name}.place", f"{w}/systolic_netlist_info", f"{w}/io.place"],
                       f"{w}/gen_io.log", w, timeout=300)
            if rc != 0 or not os.path.exists(f"{w}/io.place"):
                wr.writerow({"bench": name, "W": W, "H": H, "cw": cw, "error": "gen_io"}); csvf.flush()
                if not args.keep: shutil.rmtree(w, ignore_errors=True)
                continue

            keepset = {bench, "systolic.xml", f"{name}.net", "io.place",
                       "systolic_netlist_info", "systolic_grid_info", "systolic_arch_info"}

            # ---- 3. RUNS seeds: place (fixed IO) -> route -> record -> clean ----
            rrt = route_cache(name)
            fmaxes, ptimes = [], []
            for seed in range(args.runs):
                pf, plog, rlog = f"{w}/s{seed}.place", f"{w}/pl_{seed}.log", f"{w}/rt_{seed}.log"
                row = {"bench": name, "seed": seed, "W": W, "H": H, "cw": cw}
                try:
                    # place: read the per-page-size native delay-model cache (byte-identical)
                    pc = (["--read_rr_graph", prr, "--read_placement_delay_lookup", pdl,
                           "--route_chan_width", str(cw)]
                          if args.cache and os.path.exists(prr) and os.path.exists(pdl) else [])
                    sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                        "--place_algorithm", "systolic", "--fix_clusters", "io.place",
                        "--place_file", pf, *pc, *J],
                       plog, w, env={"SYSTOLIC_THREADS": args.systolic_threads, "SYSTOLIC_SEED": seed},
                       timeout=args.timeout)
                    (row["sys_total_s"], row["sys_place_core_s"],
                     row["sys_sta_s"], row["sys_setup_wb_s"]) = parse_systolic(open(plog).read())

                    # route: build the per-benchmark cw graph on first seed, reuse after
                    rc_ = (["--read_rr_graph", rrt] if args.cache and os.path.exists(rrt)
                           else (["--write_rr_graph", rrt] if args.cache else []))
                    _, row["route_wall_s"] = sh(
                        [VPR, "systolic.xml", bench, "--net_file", f"{name}.net",
                         "--place_file", pf, "--route", "--route_chan_width", str(cw), *rc_, *J],
                        rlog, w, timeout=args.timeout)
                    row["routed_ok"], row["wl"], row["cpd_ns"], row["fmax_mhz"] = parse_route(open(rlog).read())
                except Exception as ex:
                    row["error"] = type(ex).__name__
                wr.writerow(row); csvf.flush()
                if row.get("fmax_mhz"): fmaxes.append(row["fmax_mhz"])
                if row.get("sys_total_s"): ptimes.append(row["sys_total_s"])
                print(f"    seed {seed}: systolic_place={row.get('sys_total_s')}s "
                      f"(anneal={row.get('sys_place_core_s')} sta={row.get('sys_sta_s')}) "
                      f"Fmax={row.get('fmax_mhz')} ok={row.get('routed_ok')}", flush=True)
                if not args.keep:
                    clean_run(w, keepset)

            if fmaxes:
                fm = statistics.mean(fmaxes)
                sd = statistics.pstdev(fmaxes) if len(fmaxes) > 1 else 0.0
                pm = statistics.median(ptimes) if ptimes else float("nan")
                print(f"    -> Fmax {fm:.2f} +/- {sd:.2f} MHz (n={len(fmaxes)}), median systolic_place {pm:.4f}s", flush=True)
            if not args.keep:
                shutil.rmtree(w, ignore_errors=True)  # caches are cleared at the next benchmark's start

    csvf.close()
    if args.cache and not args.keep:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
