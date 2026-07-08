#!/usr/bin/env python3
"""
Add a VPR *timing-driven* baseline (criticality_timing, no bounding_box quench) to
each existing build_exp/<bench> workdir, reusing the already-packed .net / arch /
fixed IO. Writes to <name>_td.place so the earlier bounding_box results survive.
"""
import os, re, sys, time, subprocess, csv

ROOT="/home/ethomas/research/internship/self_hosted_placement"
OUT=f"{ROOT}/soft_systolic_placer/build_exp"
VPR=f"{os.environ['VTR_ROOT']}/build/vpr/vpr"
PARAMS=f"{ROOT}/soft_systolic_placer/experiment_params.txt"

def sh(cmd,log,timeout,cwd):
    with open(log,"w") as f:
        return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,cwd=cwd).returncode

def read_params(path):
    groups=[];cur=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith("W="):
            m=re.match(r"W=(\d+),H=(\d+)",line);cur={"bench":[]};groups.append(cur)
        elif line.startswith("SWAPS"): pass
        else:
            b,cw=line.split(",");cur["bench"].append((b.strip(),int(cw)))
    return groups

def parse_route(log):
    t=open(log).read()
    ok="Circuit successfully routed" in t
    wl=re.search(r"Total wirelength:\s*(\d+)",t)
    cp=re.search(r"[Cc]ritical path delay.*?:\s*([\d.]+)\s*ns",t)
    return ok,(int(wl.group(1)) if wl else None),(float(cp.group(1)) if cp else None)

def parse_ptime(log):
    m=re.search(r"# Placement took ([\d.]+) seconds",open(log).read()); return float(m.group(1)) if m else None

def main():
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    groups=read_params(PARAMS); rows=[]
    for gi,g in enumerate(groups):
        if which=="small" and gi!=0: continue
        if which=="large" and gi!=1: continue
        for bench,cw in g["bench"]:
            if which not in ("all","small","large") and which not in bench: continue
            name=bench[:-5]; w=f"{OUT}/{name}"
            if not os.path.exists(f"{w}/{name}.net"):
                print(f"[skip] {name}: no workdir",flush=True); continue
            print(f"[{time.strftime('%H:%M:%S')}] timing-driven {name} (cw={cw}) ...",flush=True)
            r={"bench":name,"cw":cw}
            try:
                sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place",
                    "--place_algorithm","criticality_timing","--place_file",f"{name}_td.place",
                    "--fix_clusters","io.place","--inner_num","1.0"], f"{w}/vplace_td.log",7200,w)
                r["vprtd_place_s"]=parse_ptime(f"{w}/vplace_td.log")
                sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place_file",f"{name}_td.place",
                    "--route","--route_chan_width",str(cw)], f"{w}/route_vprtd.log",7200,w)
                ok,wl,cp=parse_route(f"{w}/route_vprtd.log")
                r["vprtd_routed"]=ok; r["vprtd_wl"]=wl; r["vprtd_cp"]=cp
                print(f"    WL={wl} cp={cp}ns place_s={r['vprtd_place_s']}",flush=True)
            except Exception as e:
                r["error"]=f"{type(e).__name__}:{e}"; print(f"    ERROR {r['error']}",flush=True)
            rows.append(r)
    with open(f"{OUT}/results_td_{which}.csv","w",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=["bench","cw","vprtd_place_s","vprtd_wl","vprtd_cp","vprtd_routed","error"])
        wr.writeheader()
        for r in rows: wr.writerow({k:r.get(k) for k in wr.fieldnames})
    print(f"\nwrote {OUT}/results_td_{which}.csv",flush=True)

if __name__=="__main__": main()
