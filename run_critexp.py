#!/usr/bin/env python3
"""
Sweep the criticality exponent (SYSTOLIC_CRIT_EXP) for the timing-weighted systolic
placer. Higher exponent sharpens criticality (only near-critical connections get
weight), which should make the objective a tighter proxy for the max path (Fmax).
Runs both fixed and metro schedules on the 35x35 suite; records routed WL + CPD.
"""
import os, re, sys, time, subprocess, csv

ROOT = "/home/ethomas/research/internship/self_hosted_placement"
OUT  = f"{ROOT}/soft_systolic_placer/build_exp"
VPR  = f"{ROOT}/vtr-verilog-to-routing/build/vpr/vpr"
PARAMS = f"{ROOT}/soft_systolic_placer/experiment_params.txt"
EXPS = [1, 2, 4, 8]

def read_params(path):
    groups=[]; cur=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith("W="): cur={"bench":[]}; groups.append(cur)
        elif line.startswith("SWAPS"):
            m=re.match(r"SWAPS_PER_UPDATE=(\d+),UPDATES=(\d+)",line); cur["SW"]=int(m.group(1)); cur["UP"]=int(m.group(2))
        else:
            b,cw=line.split(","); cur["bench"].append((b.strip(),int(cw)))
    return groups

def sh(cmd, log, cwd, env=None, timeout=7200):
    e=dict(os.environ)
    if env: e.update({k:str(v) for k,v in env.items()})
    with open(log,"w") as f:
        return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=e,timeout=timeout).returncode

def parse_route(log):
    t=open(log).read()
    wl=re.search(r"Total wirelength:\s*(\d+)",t); cp=re.search(r"[Cc]ritical path delay.*?:\s*([\d.]+)\s*ns",t)
    return (int(wl.group(1)) if wl else None),(float(cp.group(1)) if cp else None)

def main():
    which = sys.argv[1] if len(sys.argv)>1 else "small"
    groups=read_params(PARAMS); rows=[]
    for gi,g in enumerate(groups):
        if which=="small" and gi!=0: continue
        if which=="large" and gi!=1: continue
        SW,UP=g["SW"],g["UP"]
        for bench,cw in g["bench"]:
            name=bench[:-5]; w=f"{OUT}/{name}"
            if not os.path.exists(f"{w}/{name}.net"): continue
            r={"bench":name,"cw":cw}
            for mode in ("fix","metro"):
                for exp in EXPS:
                    env={"SYSTOLIC_MODE":"fixed" if mode=="fix" else "metro","SYSTOLIC_CRIT":1,
                         "SYSTOLIC_SWPS":SW,"SYSTOLIC_CRIT_EXP":exp}
                    if mode=="fix": env["SYSTOLIC_UPDTS"]=UP
                    lbl=f"{mode}_e{exp}"
                    try:
                        pf=f"{w}/ce_{lbl}.place"
                        sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place",
                            "--place_algorithm","systolic","--fix_clusters","io.place","--place_file",pf],
                           f"{w}/ce_pl_{lbl}.log",w,env)
                        sh([VPR,"systolic.xml",bench,"--net_file",f"{name}.net","--place_file",pf,
                            "--route","--route_chan_width",str(cw)],f"{w}/ce_rt_{lbl}.log",w)
                        wl,cp=parse_route(f"{w}/ce_rt_{lbl}.log")
                        r[f"{lbl}_wl"]=wl; r[f"{lbl}_cp"]=cp
                    except Exception as ex:
                        r[f"{lbl}_wl"]=None
            print(f"[{time.strftime('%H:%M:%S')}] {name}: "
                  + " ".join(f"m_e{e}={r.get(f'metro_e{e}_cp')}" for e in EXPS), flush=True)
            rows.append(r)
    keys=["bench","cw"]
    for mode in ("fix","metro"):
        for e in EXPS: keys+=[f"{mode}_e{e}_wl",f"{mode}_e{e}_cp"]
    with open(f"{OUT}/critexp_{which}.csv","w",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=keys,extrasaction="ignore"); wr.writeheader()
        for r in rows: wr.writerow(r)
    print(f"\nwrote {OUT}/critexp_{which}.csv",flush=True)

if __name__=="__main__": main()
