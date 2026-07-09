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

Ablation ladder (each rung adds one improvement over the paper's baseline accelerator;
all WL-only until 'full'), driven purely by env knobs:

  baseline : MODE=fixed  CRIT=0 FANORM=0  (+paper SWPS/UPDTS)  fixed linear-cool schedule
  metro    : MODE=metro  CRIT=0 FANORM=0                        + adaptive schedule & Metropolis accept
  timing   : MODE=metro  CRIT=1 FANORM=0                        + timing-driven (STA)
  full     : MODE=metro  CRIT=1 FANORM=1                        + inverse-fanout WL weight (free) [default]
  vtr      : VPR criticality_timing, same fixed IO             (baseline comparator)
Note: inverse-fanout is a timing-coupled refinement (frees the WL term from high-fanout
non-critical nets so criticality can tighten critical paths) -- cleanest as the last rung.

place_time_s is the reported placement time: for systolic it is sys_total_s (time inside
run_systolic: STA + weighting + annealing + setup/writeback -- "how long the accelerator
runs", excluding VPR overhead); for vtr it is VPR's "# Placement took". Same measurement
convention as the paper. All seed values are written to CSV; medians are printed.

Runs execute one at a time (no contention). --workers = VPR -j (packing/STA/routing).
Env overrides: SELF_HOSTED_ROOT, VPR, VPR_NUM_WORKERS.
"""
import os, re, time, csv, argparse, shutil, subprocess, statistics

ROOT     = os.environ.get("SELF_HOSTED_ROOT", "/home/ethomas/research/internship/self_hosted_placement")
VPR      = os.environ.get("VPR", f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr")
PP       = f"{ROOT}/systolic_page_placer"
SCR      = f"{PP}/scripts"
TPL      = f"{PP}/vtr_integration/arch/heterogeneous_k10.xml"
SYNTH    = f"{ROOT}/soft_systolic_placer/synth"
PARAMS   = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

CSV_FIELDS = ["bench", "config", "seed", "W", "H", "cw",
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
            cur["bench"].append((b.strip(), int(cw)))
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
                    help="comma list from {baseline,metro,fanout,full,vtr} (default: full)")
    ap.add_argument("--ablation", action="store_true", help="run the full systolic ladder")
    ap.add_argument("--vtr", action="store_true", help="also run the VTR-timing comparator")
    ap.add_argument("--workdir", default=f"{ROOT}/soft_systolic_placer/bench_work")
    ap.add_argument("--out", default=None, help="CSV path (default <workdir>/results.csv)")
    ap.add_argument("--keep", action="store_true", help="keep VTR products + rr-graph caches (debug)")
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--vpr", default=VPR, help="path to the vpr binary with the systolic placer")
    ap.add_argument("--systolic-threads", type=int, default=1, help="SYSTOLIC_THREADS for the systolic placer")
    ap.add_argument("--no-cache", dest="cache", action="store_false", help="disable rr-graph/delay-lookup caching")
    args = ap.parse_args()
    VPR = args.vpr
    args.workdir = os.path.abspath(args.workdir)  # VPR runs with cwd=benchmark dir -> paths must be absolute

    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        configs = (LADDER if args.ablation else ["full"]) + (["vtr"] if args.vtr else [])
    print(f"vpr = {VPR}\nconfigs = {configs}", flush=True)

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

    for gi, g in enumerate(read_params(PARAMS)):
        if args.which == "small" and gi != 0: continue
        if args.which == "large" and gi != 1: continue
        W, H, SW, UP = g["W"], g["H"], g["SW"], g["UP"]
        for bench, cw in g["bench"]:
            name = bench[:-5] if bench.endswith(".blif") else bench
            if args.benchmarks and name not in args.benchmarks: continue
            blif = f"{SYNTH}/{bench}"
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
                is_vtr = (cfg == "vtr")
                fmaxes, wls, ptimes = [], [], []
                for seed in range(args.runs):
                    tag = f"{cfg}_s{seed}"
                    pf, plog, rlog = f"{w}/{tag}.place", f"{w}/pl_{tag}.log", f"{w}/rt_{tag}.log"
                    row = {"bench": name, "config": cfg, "seed": seed, "W": W, "H": H, "cw": cw}
                    try:
                        if is_vtr:
                            sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                                "--place_algorithm", "criticality_timing", "--fix_clusters", "io.place",
                                "--seed", str(seed + 1), "--place_file", pf, *pc_read, *J], plog, w, timeout=args.timeout)
                            row["place_time_s"] = _grep(r"# Placement took ([\d.]+) seconds", open(plog).read())
                        else:
                            env = {"SYSTOLIC_SEED": seed, "SYSTOLIC_THREADS": args.systolic_threads}
                            env.update(sys_env(cfg))
                            if env["SYSTOLIC_MODE"] == "fixed":   # paper fixed schedule (baseline + fanout rungs)
                                env["SYSTOLIC_SWPS"], env["SYSTOLIC_UPDTS"] = SW, UP
                            sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                                "--place_algorithm", "systolic", "--fix_clusters", "io.place",
                                "--place_file", pf, *pc_read, *J], plog, w, env=env, timeout=args.timeout)
                            row["place_time_s"], row["sys_place_core_s"], row["sys_sta_s"] = parse_systolic(open(plog).read())
                        rc_ = (["--read_rr_graph", rrt] if args.cache and os.path.exists(rrt)
                               else (["--write_rr_graph", rrt] if args.cache else []))
                        _, row["route_wall_s"] = sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net",
                                                     "--place_file", pf, "--route", "--route_chan_width", str(cw), *rc_, *J],
                                                    rlog, w, timeout=args.timeout)
                        row["routed_ok"], row["wl"], row["cpd_ns"], row["fmax_mhz"] = parse_route(open(rlog).read())
                    except Exception as ex:
                        row["error"] = type(ex).__name__
                    wr.writerow(row); csvf.flush()
                    if row.get("fmax_mhz"): fmaxes.append(row["fmax_mhz"])
                    if row.get("wl"): wls.append(row["wl"])
                    if row.get("place_time_s"): ptimes.append(row["place_time_s"])
                    if not args.keep: clean_run(w, keepset)
                print(f"    {cfg:9s}: median Fmax {_median(fmaxes):7.2f} MHz | median WL {_median(wls):9.0f} | "
                      f"median place {_median(ptimes):.4f}s  (n={len(fmaxes)})", flush=True)

            if not args.keep:
                shutil.rmtree(w, ignore_errors=True)

    csvf.close()
    if args.cache and not args.keep:
        shutil.rmtree(cache, ignore_errors=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
