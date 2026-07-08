#!/usr/bin/env python3
"""
Stage-2 study: does adding timing criticality to the objective let the adaptive
(metro) systolic placer beat the fixed one on Fmax by MORE than it does on WL?

Per benchmark (reusing the existing build_exp/<name> pack outputs), place with:
  vpr_bbox, vpr_timing (VPR baselines), and systolic in 4 configs
  {fixed,metro} x {WL-only, timing-weighted}; route each; record routed WL + CPD.
"""
import os, re, sys, time, subprocess, csv

ROOT = "/home/ethomas/research/internship/self_hosted_placement"
OUT  = f"{ROOT}/soft_systolic_placer/build_exp"
VPR  = f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr"
PARAMS = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

def read_params(path):
    groups = []; cur = None
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith("W="):
            cur = {"bench": []}; groups.append(cur)
        elif line.startswith("SWAPS"):
            m = re.match(r"SWAPS_PER_UPDATE=(\d+),UPDATES=(\d+)", line)
            cur["SW"] = int(m.group(1)); cur["UP"] = int(m.group(2))
        else:
            b, cw = line.split(","); cur["bench"].append((b.strip(), int(cw)))
    return groups

def sh(cmd, log, cwd, env=None, timeout=7200):
    e = dict(os.environ);
    if env: e.update({k: str(v) for k, v in env.items()})
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e, timeout=timeout).returncode

def parse_route(log):
    t = open(log).read()
    ok = "Circuit successfully routed" in t
    wl = re.search(r"Total wirelength:\s*(\d+)", t)
    cp = re.search(r"[Cc]ritical path delay.*?:\s*([\d.]+)\s*ns", t)
    return ok, (int(wl.group(1)) if wl else None), (float(cp.group(1)) if cp else None)

def place_time(log):
    m = re.search(r"# Placement took ([\d.]+) seconds", open(log).read())
    return float(m.group(1)) if m else None

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    groups = read_params(PARAMS)
    rows = []
    for gi, g in enumerate(groups):
        if which == "small" and gi != 0: continue
        if which == "large" and gi != 1: continue
        SW, UP = g["SW"], g["UP"]
        for bench, cw in g["bench"]:
            name = bench[:-5]; w = f"{OUT}/{name}"
            if not os.path.exists(f"{w}/{name}.net"):
                print(f"[skip] {name}", flush=True); continue
            print(f"[{time.strftime('%H:%M:%S')}] {name} cw={cw}", flush=True)
            r = {"bench": name, "cw": cw}
            configs = [
                ("vpr_bbox",    ["--place_algorithm", "bounding_box"], {}),
                ("vpr_timing",  ["--place_algorithm", "criticality_timing"], {}),
                ("fix_wl",      ["--place_algorithm", "systolic"], {"SYSTOLIC_MODE":"fixed","SYSTOLIC_CRIT":0,"SYSTOLIC_SWPS":SW,"SYSTOLIC_UPDTS":UP}),
                ("fix_time",    ["--place_algorithm", "systolic"], {"SYSTOLIC_MODE":"fixed","SYSTOLIC_CRIT":1,"SYSTOLIC_SWPS":SW,"SYSTOLIC_UPDTS":UP}),
                ("metro_wl",    ["--place_algorithm", "systolic"], {"SYSTOLIC_MODE":"metro","SYSTOLIC_CRIT":0,"SYSTOLIC_SWPS":SW}),
                ("metro_time",  ["--place_algorithm", "systolic"], {"SYSTOLIC_MODE":"metro","SYSTOLIC_CRIT":1,"SYSTOLIC_SWPS":SW}),
            ]
            for lbl, algo, env in configs:
                try:
                    pf = f"{w}/s2_{lbl}.place"
                    sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place",
                        *algo, "--fix_clusters", "io.place", "--place_file", pf],
                       f"{w}/s2_pl_{lbl}.log", w, env)
                    sh([VPR, "systolic.xml", bench, "--net_file", f"{name}.net", "--place_file", pf,
                        "--route", "--route_chan_width", str(cw)], f"{w}/s2_rt_{lbl}.log", w)
                    ok, wl, cp = parse_route(f"{w}/s2_rt_{lbl}.log")
                    r[f"{lbl}_wl"] = wl; r[f"{lbl}_cp"] = cp
                    r[f"{lbl}_pt"] = place_time(f"{w}/s2_pl_{lbl}.log")
                except Exception as e:
                    r[f"{lbl}_wl"] = None; r[f"{lbl}_err"] = f"{type(e).__name__}"
            print(f"    WL bbox={r.get('vpr_bbox_wl')} fixT={r.get('fix_time_wl')} metroT={r.get('metro_time_wl')} | "
                  f"CPD vprT={r.get('vpr_timing_cp')} fixT={r.get('fix_time_cp')} metroT={r.get('metro_time_cp')}", flush=True)
            rows.append(r)
    keys = ["bench","cw"]
    for lbl in ["vpr_bbox","vpr_timing","fix_wl","fix_time","metro_wl","metro_time"]:
        keys += [f"{lbl}_wl", f"{lbl}_cp", f"{lbl}_pt"]
    with open(f"{OUT}/stage2_{which}.csv","w",newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore"); wr.writeheader()
        for r in rows: wr.writerow(r)
    print(f"\nwrote {OUT}/stage2_{which}.csv", flush=True)

if __name__ == "__main__":
    main()
