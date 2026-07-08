#!/usr/bin/env python3
"""
Consolidated final study: systolic placer (final config) vs VPR bounding_box and
VPR criticality_timing, on all paper benchmarks, at the per-benchmark channel width.
Reports routed WL, critical-path delay, and placement runtime.

Systolic config = current defaults (metro schedule, Metropolis accept, fanout-norm,
lambda=0.3, crit_exp=2, cadence=0 i.e. a single up-front STA); only SYSTOLIC_SWPS set
per array size, and single-threaded for a fair per-core runtime comparison to VPR.
"""
import os, re, sys, time, subprocess, csv, math

ROOT = "/home/ethomas/research/internship/self_hosted_placement"
OUT  = f"{ROOT}/soft_systolic_placer/build_exp"
VPR  = f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr"
PARAMS = f"{ROOT}/soft_systolic_placer/experiment_params.txt"

def read_params(path):
    g=[]; cur=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith("W="): cur={"bench":[]}; g.append(cur)
        elif line.startswith("SWAPS"):
            m=re.match(r"SWAPS_PER_UPDATE=(\d+),UPDATES=(\d+)",line); cur["SW"]=int(m.group(1)); cur["UP"]=int(m.group(2))
        else:
            b,cw=line.split(","); cur["bench"].append((b.strip(),int(cw)))
    return g

def sh(cmd, log, cwd, env=None, timeout=10800):
    e=dict(os.environ)
    if env: e.update({k:str(v) for k,v in env.items()})
    with open(log,"w") as f:
        return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=e,timeout=timeout).returncode

def parse_route(log):
    t=open(log).read()
    ok="Circuit successfully routed" in t
    wl=re.search(r"Total wirelength:\s*(\d+)",t)
    cp=re.search(r"[Cc]ritical path delay.*?:\s*([\d.]+)\s*ns",t)
    return ok,(int(wl.group(1)) if wl else None),(float(cp.group(1)) if cp else None)

def vpr_ptime(log):
    m=re.search(r"# Placement took ([\d.]+) seconds",open(log).read()); return float(m.group(1)) if m else None
def sys_ptime(log):
    m=re.search(r"\[systolic\].*?cost=[0-9-]+\s+([\d.]+) s",open(log).read()); return float(m.group(1)) if m else None

def main():
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    groups=read_params(PARAMS); rows=[]
    for gi,g in enumerate(groups):
        if which=="small" and gi!=0: continue
        if which=="large" and gi!=1: continue
        SW=g["SW"]
        for bench,cw in g["bench"]:
            name=bench[:-5]; w=f"{OUT}/{name}"
            if not os.path.exists(f"{w}/{name}.net"): print(f"[skip] {name}",flush=True); continue
            print(f"[{time.strftime('%H:%M:%S')}] {name} cw={cw}",flush=True)
            r={"bench":name,"cw":cw}
            cfgs=[("bbox",["--place_algorithm","bounding_box"],None),
                  ("timing",["--place_algorithm","criticality_timing"],None),
                  ("sys",["--place_algorithm","systolic"],{"SYSTOLIC_SWPS":SW,"SYSTOLIC_THREADS":1})]
            for lbl,algo,env in cfgs:
                try:
                    pf=f"{w}/fin_{lbl}.place"
                    sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place",*algo,
                        "--fix_clusters","io.place","--place_file",pf], f"{w}/fin_pl_{lbl}.log", w, env)
                    sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place_file",pf,
                        "--route","--route_chan_width",str(cw)], f"{w}/fin_rt_{lbl}.log", w)
                    ok,wl,cp=parse_route(f"{w}/fin_rt_{lbl}.log")
                    r[f"{lbl}_wl"]=wl; r[f"{lbl}_cp"]=cp
                    r[f"{lbl}_pt"]=sys_ptime(f"{w}/fin_pl_{lbl}.log") if lbl=="sys" else vpr_ptime(f"{w}/fin_pl_{lbl}.log")
                    r[f"{lbl}_ok"]=ok
                except Exception as ex:
                    r[f"{lbl}_wl"]=None; r[f"{lbl}_err"]=type(ex).__name__
            print(f"    WL bbox={r.get('bbox_wl')} tmg={r.get('timing_wl')} sys={r.get('sys_wl')} | "
                  f"CPD bbox={r.get('bbox_cp')} tmg={r.get('timing_cp')} sys={r.get('sys_cp')} | "
                  f"pt bbox={r.get('bbox_pt')} tmg={r.get('timing_pt')} sys={r.get('sys_pt')}",flush=True)
            rows.append(r)
    keys=["bench","cw"]
    for l in ["bbox","timing","sys"]: keys+=[f"{l}_wl",f"{l}_cp",f"{l}_pt",f"{l}_ok"]
    with open(f"{OUT}/final_{which}.csv","w",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=keys,extrasaction="ignore"); wr.writeheader()
        for r in rows: wr.writerow(r)
    # geomean summary
    def g(r,k): v=r.get(k); return float(v) if v not in (None,"","None") else None
    def gm(xs): xs=[x for x in xs if x and x>0]; return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')
    for base in ["bbox","timing"]:
        wl=[g(r,'sys_wl')/g(r,f'{base}_wl') for r in rows if g(r,'sys_wl') and g(r,f'{base}_wl')]
        cp=[g(r,'sys_cp')/g(r,f'{base}_cp') for r in rows if g(r,'sys_cp') and g(r,f'{base}_cp')]
        sp=[g(r,f'{base}_pt')/g(r,'sys_pt') for r in rows if g(r,f'{base}_pt') and g(r,'sys_pt')]
        print(f"\nsystolic vs VPR-{base} (geomean, n={len(wl)}): WL {(gm(wl)-1)*100:+.1f}% | CPD {(gm(cp)-1)*100:+.1f}% | place speedup {gm(sp):.0f}x",flush=True)
    print(f"\nwrote {OUT}/final_{which}.csv",flush=True)

if __name__=="__main__": main()
