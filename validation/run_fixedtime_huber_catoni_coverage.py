#!/usr/bin/env python3
"""Reproduce fixed-time coverage/interval-width panel for robust CS methods."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import decision_risk_huber_sensitivity as huber
from huber_catoni_reference import catoni_huber_cs

SEED=20261031; REPEATS=300; N=1024; CLEAN_MEAN=.37; ACTUAL_RHO=.12

def run(repeats=REPEATS,n=N):
    qcont=(1-ACTUAL_RHO)*CLEAN_MEAN
    conditions={"clean_actual0":{"actual_rho":0.0,"observed_mean":CLEAN_MEAN,"target_mean":CLEAN_MEAN,"methods":{}},
      "contaminated_actual0_12":{"actual_rho":ACTUAL_RHO,"observed_mean":qcont,"target_mean":CLEAN_MEAN,"methods":{}}}
    collect={k:{} for k in conditions}; rng=np.random.default_rng(SEED)
    for _ in range(repeats):
        xclean=rng.binomial(1,CLEAN_MEAN,size=n).astype(np.int8)
        xcont=rng.binomial(1,qcont,size=n).astype(np.int8)
        for name,x in (("clean_actual0",xclean),("contaminated_actual0_12",xcont)):
            succ=int(x.sum()); beta=huber.base.beta_mixture_cs(succ,n,huber.DELTA/4)
            collect[name].setdefault("beta_nominal",[]).append(beta)
            eps_grid=(0.0,ACTUAL_RHO) if name=="contaminated_actual0_12" else (0.0,)
            for eps in eps_grid:
                collect[name].setdefault(f"beta_sensitivity_eps_{eps:.2f}",[]).append(huber.clean_mean_interval(*beta,eps))
                collect[name].setdefault(f"catoni_huber_eps_{eps:.2f}",[]).append(catoni_huber_cs(x,alpha=huber.DELTA/4,eps=eps,moment_bound=.25))
    for name,methods in collect.items():
        target=conditions[name]["target_mean"]
        for method,intervals in methods.items():
            widths=[hi-lo for lo,hi in intervals]; cover=sum(lo<=target<=hi for lo,hi in intervals)
            conditions[name]["methods"][method]={"coverage_count":cover,"coverage_rate":cover/repeats,"mean_width":float(np.mean(widths)),"median_width":float(np.median(widths)),"p95_width":float(np.quantile(widths,.95)),"n":n,"repeats":repeats,"alpha_cell":huber.DELTA/4}
    return {"schema":"symthaea-huber-catoni-fixed-time-coverage-panel-v1","status":"FIXED_TIME_MONTE_CARLO_CALIBRATION_NOT_PROOF","protocol":{"seed":SEED,"repeats_per_condition":repeats,"n":n,"true_clean_mean":CLEAN_MEAN,"actual_contamination_rates":[0.0,ACTUAL_RHO],"contaminant":"R=0","familywise_delta":huber.DELTA,"per_cell_alpha":huber.DELTA/4,"moment_bound":.25},"conditions":conditions,"limitations":["Empirical coverage in a finite Monte Carlo panel is not a formal validity proof.","Bernoulli bounded case only.","Paper-equation Catoni method is independently reimplemented, not authors reference code.","Sensitivity adjustment requires declared contamination bound to cover actual contamination."]}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repeats",type=int,default=REPEATS);p.add_argument("--n",type=int,default=N);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=run(a.repeats,a.n);data=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(data)
    print(json.dumps({"path":str(a.output),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
