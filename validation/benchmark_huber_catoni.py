#!/usr/bin/env python3
"""Matched comparison: nominal beta-mixture CS, sensitivity inversion, Huber-Catoni CS."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import decision_risk_huber_sensitivity as huber
from huber_catoni_reference import catoni_huber_cs

SEED = 20261031
LOOKS = (512, 1024, 2048, 4096)
CLAIMED_GRID = (0.0, 0.025, 0.05, 0.12, 0.18)
ACTUAL_GRID = (0.12, 0.18)
METHODS = ("nominal_beta_cs", "sensitivity_beta_cs", "paper_equation_huber_catoni_cs")


def assess(model, streams, alpha_cell, method, claimed_eps):
    for n in LOOKS:
        lows, highs = np.zeros(len(model)), np.ones(len(model))
        for i, xs in enumerate(streams):
            prefix = xs[:n]
            if method == "paper_equation_huber_catoni_cs":
                lows[i], highs[i] = catoni_huber_cs(prefix, alpha=alpha_cell, eps=claimed_eps, moment_bound=.25)
            else:
                lo, hi = huber.base.beta_mixture_cs(int(np.sum(prefix)), n, alpha_cell)
                if method == "sensitivity_beta_cs":
                    lo, hi = huber.clean_mean_interval(lo, hi, claimed_eps)
                lows[i], highs[i] = lo, hi
        status, err_lo, err_hi = huber.decision(model, lows, highs)
        if status is not None:
            return {"status": status, "total_observations": int(n*len(model)), "per_cell_observations": [n]*len(model), "final_error_lower": err_lo, "final_error_upper": err_hi, "looks": LOOKS.index(n)+1}
    return {"status": "refuse_unresolved", "total_observations": int(LOOKS[-1]*len(model)), "per_cell_observations": [LOOKS[-1]]*len(model), "final_error_lower": err_lo, "final_error_upper": err_hi, "looks": len(LOOKS)}


def wilson(k, n, z=1.959963984540054):
    p = k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(repeats=100, actual_grid=ACTUAL_GRID, claimed_grid=CLAIMED_GRID):
    if repeats < 30: raise ValueError("use at least 30 repeats")
    model = np.array([.15,.30,.60,.80])
    clean = np.array([.15,.37,.60,.80])
    rng = np.random.default_rng(SEED)
    output = {}
    for actual in actual_grid:
        q=clean.copy(); q[1]=(1-actual)*clean[1]
        rows={f"claimed_{eps:.3f}":{m:[] for m in METHODS} for eps in claimed_grid}
        for rep in range(repeats):
            streams=np.vstack([rng.binomial(1,p,size=LOOKS[-1]) for p in q]).astype(np.int8)
            nominal=assess(model, streams, huber.DELTA/len(model), "nominal_beta_cs", 0.0)
            for eps in claimed_grid:
                rows[f"claimed_{eps:.3f}"]["nominal_beta_cs"].append(nominal)
                for method in ("sensitivity_beta_cs","paper_equation_huber_catoni_cs"):
                    rows[f"claimed_{eps:.3f}"][method].append(assess(model, streams, huber.DELTA/len(model), method, eps))
        table={}
        for label, methods in rows.items():
            table[label]={}; claimed=float(label.split("_")[1])
            for method, runs in methods.items():
                admits=sum(x["status"]=="admit" for x in runs); rejects=sum(x["status"]=="reject" for x in runs); unresolved=sum(x["status"]=="refuse_unresolved" for x in runs)
                obs=np.array([x["total_observations"] for x in runs],float)
                table[label][method]={"claimed_eps":claimed,"bound_covers_actual":claimed+1e-12>=actual,"admit":admits,"reject":rejects,"unresolved":unresolved,"unsafe_admissions":admits,"unsafe_admission_rate":admits/repeats,"unsafe_admission_wilson95":wilson(admits,repeats),"mean_observations":float(obs.mean()),"median_observations":float(np.median(obs)),"p95_observations":float(np.quantile(obs,.95))}
        output[f"actual_rho_{actual:.3f}"]={"actual_rho":actual,"clean_max_model_error":.07,"observed_attacked_cell_mean":float(q[1]),"claimed_bound_results":table}
    return {"schema":"symthaea-huber-catoni-matched-baseline-v1","status":"INDEPENDENT_PAPER_EQUATION_REIMPLEMENTATION_RESEARCH_ONLY","protocol":{"seed":SEED,"repeats":repeats,"look_sample_counts_per_cell":list(LOOKS),"actual_rho_grid":list(actual_grid),"claimed_epsilon_grid":list(claimed_grid),"cells":4,"familywise_delta":huber.DELTA,"per_cell_alpha":huber.DELTA/4,"error_tolerance":huber.EPS,"clean_model_risk":model.tolist(),"clean_target_risk":clean.tolist(),"moment_bound":.25,"paired_streams":True},"results":output,"methods":{"nominal_beta_cs":"beta-mixture CS for observed Q, falsely interprets Q as clean P under contamination","sensitivity_beta_cs":"beta-mixture CS for Q mapped through Huber clean-mean sensitivity set","paper_equation_huber_catoni_cs":"independent implementation of robust Catoni-CS equations specified in Wang & Ramdas (AISTATS 2023); not their official implementation"},"limitations":["IID Bernoulli Huber mixture, one cell contaminated with R=0","paper equations independently implemented and need further independent validation; no official author code executed","all confidence claims require stated contamination and moment bounds to be correct","50 repeats; 0 unsafe admissions still has nonzero upper bound","no production Rust integration or direct SOTA claim"],"literature":["https://proceedings.mlr.press/v206/wang23p.html","https://github.com/HJ-Wg/catoni-confseq"],"environment":{"python":platform.python_version(),"numpy":np.__version__},"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repeats",type=int,default=100); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); start=time.perf_counter(); result=run(a.repeats)
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode(); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(raw)
    print(json.dumps({"path":str(a.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"runtime_seconds":round(time.perf_counter()-start,3)},indent=2))
