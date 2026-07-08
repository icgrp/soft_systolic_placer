#!/usr/bin/env python3
"""
Compare placement quality/runtime of VPR's own placer vs our fixed (placer.cpp)
and adaptive (adaptive_placer.cpp, Metropolis) placers, on the paper's benchmarks,
array sizes, schedules and per-benchmark routing channel widths.

Per benchmark:
  1. gen_arch(W,H) -> systolic.xml
  2. VPR --pack --place  (produces .net, systolic_{grid,netlist,arch}_info, VPR .place)
  3. gen_io_placement / gen_placer_init  -> io.place, placer_init
  4. our placer.cpp (fixed, SW/UP, hot) and adaptive_placer.cpp (metro, auto-stop)
  5. VPR baseline place: --place bounding_box --fix_clusters io.place
  6. route all three at the benchmark's channel width; parse routed WL + crit path
Everything uses the identical arch / .net / fixed IOs / channel width, so only the
placement algorithm differs.
"""
import os, re, sys, time, subprocess, shutil, csv

ROOT   = "/home/ethomas/research/internship/self_hosted_placement"
PP     = f"{ROOT}/systolic_page_placer"
SCR    = f"{PP}/scripts"
TPL    = f"{PP}/vtr_integration/arch/heterogeneous_k10.xml"
SYNTH  = f"{ROOT}/soft_systolic_placer/synth"
VPR    = f"{os.environ['VTR_ROOT']}/build/vpr/vpr"
PL     = "/tmp/pl"      # fixed placer binary
ADP    = "/tmp/adp"     # adaptive placer binary
OUT    = f"{ROOT}/soft_systolic_placer/build_exp"
THREADS= "8"
PARAMS = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

def sh(cmd, log, timeout, cwd=None):
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
    return p.returncode

def cap(cmd, timeout, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return p.stdout + p.stderr

def parse_route(log):
    t = open(log).read()
    ok = "Circuit successfully routed" in t
    wl = re.search(r"Total wirelength:\s*(\d+)", t)
    cp = re.search(r"[Cc]ritical path delay.*?:\s*([\d.]+)\s*ns", t)
    return ok, (int(wl.group(1)) if wl else None), (float(cp.group(1)) if cp else None)

def parse_place_time(cpp_stderr):
    m = re.search(r"\[place ([\d.]+) s", cpp_stderr)
    return float(m.group(1)) if m else None

def parse_cost(cpp_stderr):
    m = re.search(r"cost (\d+)", cpp_stderr)
    return int(m.group(1)) if m else None

def parse_vpr_place_time(log):
    m = re.search(r"# Placement took ([\d.]+) seconds", open(log).read())
    return float(m.group(1)) if m else None

def read_params(path):
    groups=[]; cur=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith("W="):
            m=re.match(r"W=(\d+),H=(\d+)", line); cur={"W":int(m.group(1)),"H":int(m.group(2)),"bench":[]}; groups.append(cur)
        elif line.startswith("SWAPS_PER_UPDATE"):
            m=re.match(r"SWAPS_PER_UPDATE=(\d+),UPDATES=(\d+)", line); cur["SW"]=int(m.group(1)); cur["UP"]=int(m.group(2))
        else:
            b,cw=line.split(","); cur["bench"].append((b.strip(), int(cw)))
    return groups

def run_bench(bench, cw, W, H, SW, UP, results):
    name=bench[:-5] if bench.endswith(".blif") else bench
    w=f"{OUT}/{name}"; shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
    blif=f"{SYNTH}/{bench}"; shutil.copy(blif, f"{w}/{bench}")
    row={"bench":name,"W":W,"H":H,"cw":cw}
    try:
        sh(["python3",f"{SCR}/gen_arch.py",str(W),str(H),TPL,f"{w}/systolic.xml"], f"{w}/gen_arch.log", 60)
        # 1. pack + initial place (for .net, info files, IO positions)
        t0=time.time()
        rc=sh([VPR,"systolic.xml",bench,"--pack","--place","--inner_num","1.0"], f"{w}/pack.log", 3600, cwd=w)
        row["pack_s"]=round(time.time()-t0,1)
        if rc!=0: row["error"]="pack_fail"; results.append(row); return row
        G,N,IO,I=f"{w}/systolic_grid_info",f"{w}/systolic_netlist_info",f"{w}/io.place",f"{w}/placer_init"
        sh(["python3",f"{SCR}/gen_io_placement.py",f"{w}/{name}.place",N,IO], f"{w}/io.log", 300)
        sh(["python3",f"{SCR}/gen_placer_init.py",G,I], f"{w}/init.log", 300)
        # 2. our placers
        se=cap([PL,G,N,IO,I,f"{w}/out_fix","--no-trace","--threads",THREADS,"--swps",str(SW),"--updts",str(UP),"--temp","65535"], 3600)
        row["fix_place_s"]=parse_place_time(se); row["fix_cost"]=parse_cost(se)
        se=cap([ADP,G,N,IO,I,f"{w}/out_adp","--no-trace","--threads",THREADS,"--swps",str(SW),"--updts","100000","--temp","65535"], 3600)
        row["adp_place_s"]=parse_place_time(se); row["adp_cost"]=parse_cost(se)
        for m in ("fix","adp"):
            with open(f"{w}/complete_{m}.place","w") as o:
                o.write(open(f"{w}/out_{m}/final_systolic.place").read()); o.write(open(IO).read())
        # 3. VPR baseline place (bounding_box, fixed IO)
        t0=time.time()
        sh([VPR,"systolic.xml",bench,"--net_file",f"{w}/{name}.net","--place",
            "--place_algorithm","bounding_box","--place_quench_algorithm","bounding_box",
            "--fix_clusters",IO,"--inner_num","1.0"], f"{w}/vplace.log", 3600, cwd=w)
        row["vpr_place_s"]=parse_vpr_place_time(f"{w}/vplace.log")
        # 4. route all three at cw
        places={"vpr":f"{w}/{name}.place","fix":f"{w}/complete_fix.place","adp":f"{w}/complete_adp.place"}
        for m,pf in places.items():
            sh([VPR,"systolic.xml",bench,"--net_file",f"{w}/{name}.net","--place_file",pf,
                "--route","--route_chan_width",str(cw)], f"{w}/route_{m}.log", 7200, cwd=w)
            ok,wl,cp=parse_route(f"{w}/route_{m}.log")
            row[f"{m}_routed"]=ok; row[f"{m}_wl"]=wl; row[f"{m}_cp"]=cp
    except subprocess.TimeoutExpired as e:
        row["error"]=f"timeout:{e.cmd[0].split('/')[-1]}"
    except Exception as e:
        row["error"]=f"{type(e).__name__}:{e}"
    results.append(row); return row

def main():
    which = sys.argv[1] if len(sys.argv)>1 else "all"   # "small"/"large"/"all"/benchname
    os.makedirs(OUT, exist_ok=True)
    groups=read_params(PARAMS)
    results=[]
    for gi,g in enumerate(groups):
        if which=="small" and gi!=0: continue
        if which=="large" and gi!=1: continue
        for bench,cw in g["bench"]:
            if which not in ("all","small","large") and which not in bench: continue
            print(f"[{time.strftime('%H:%M:%S')}] running {bench} ({g['W']}x{g['H']}, cw={cw}) ...", flush=True)
            r=run_bench(bench,cw,g["W"],g["H"],g["SW"],g["UP"],results)
            def f(k): return r.get(k)
            if r.get("error"):
                print(f"    ERROR: {r['error']}", flush=True)
            else:
                print(f"    WL  vpr={f('vpr_wl')} fix={f('fix_wl')} adp={f('adp_wl')} | "
                      f"cp(ns) vpr={f('vpr_cp')} fix={f('fix_cp')} adp={f('adp_cp')} | "
                      f"place_s vpr={f('vpr_place_s')} fix={f('fix_place_s')} adp={f('adp_place_s')}", flush=True)
    # write CSV
    keys=["bench","W","H","cw","pack_s","vpr_place_s","fix_place_s","adp_place_s",
          "vpr_wl","fix_wl","adp_wl","vpr_cp","fix_cp","adp_cp",
          "vpr_routed","fix_routed","adp_routed","fix_cost","adp_cost","error"]
    csvp=f"{OUT}/results_{which}.csv"
    with open(csvp,"w",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=keys); wr.writeheader()
        for r in results: wr.writerow({k:r.get(k) for k in keys})
    print(f"\nwrote {csvp}", flush=True)

if __name__=="__main__":
    main()
