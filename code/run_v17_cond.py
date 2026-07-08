import sys, time, numpy as np, pandas as pd, importlib.util
spec=importlib.util.spec_from_file_location("v17_fast","v17_fast.py")
vf=importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)
cond=sys.argv[1]
betas=vf.V17_CONDITIONS[cond]
rows=[]
t0=time.time()
for seed in vf.V17_CONFIG["seeds"]:
    cfg=dict(vf.V17_CONFIG); cfg["beta_drift"]=betas["beta_drift"]; cfg["beta_var"]=betas["beta_var"]
    rr=vf.run_condition_seed("full_bqml", seed, cfg)   # machinery=full_bqml, betas set per condition
    for r in rr:
        r["condition"]=cond; r["beta_drift"]=betas["beta_drift"]; r["beta_var"]=betas["beta_var"]
    rows+=rr
    print(f"  {cond} seed={seed} done ({time.time()-t0:.0f}s)")
pd.DataFrame(rows).to_csv(f"v17_{cond}.csv", index=False)
print(f"wrote v17_{cond}.csv  rows={len(rows)}  elapsed={time.time()-t0:.0f}s")
