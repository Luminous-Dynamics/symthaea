#!/usr/bin/env python3
"""Matched audit allocation policies under per-cell anytime-valid confidence sequences.

Compares uniform, maximum-error-upper-bound (current baseline), widest-active-CI,
and marginal-CI-reduction scheduling. Fixed support and iid Bernoulli assumptions
are required; this is a synthetic decision-certificate benchmark, not deployment proof.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import decision_risk_adaptive_audit_allocation as base

SCHEMA = "symthaea-decision-risk-constraint-aware-allocation-v1"
SEED = 20261029
DELTA = base.DELTA
EPS = base.EPSILON_ALLOWANCE
PILOT = base.PILOT
BATCH = base.BATCH
MAX_PER_CELL = base.MAX_PER_CELL
TOL = 1e-12


def run_strategy(streams: np.ndarray, model: np.ndarray, strategy: str) -> dict:
    k = len(model)
    if strategy not in ("uniform", "max_error_upper", "widest_active_ci", "marginal_ci_reduction"):
        raise ValueError("unknown allocation strategy")
    ns = np.full(k, PILOT, dtype=np.int64)
    successes = np.array([int(np.sum(streams[i, :PILOT])) for i in range(k)], dtype=np.int64)
    alpha_cell = DELTA / k
    def bounds():
        lows = np.zeros(k); highs = np.ones(k)
        err_lo = np.zeros(k); err_hi = np.full(k, np.inf)
        widths = np.ones(k)
        for i, n in enumerate(ns):
            lo, hi = base.beta_mixture_cs(int(successes[i]), int(n), alpha_cell)
            lows[i], highs[i] = lo, hi
            err_hi[i] = max(abs(float(model[i])-lo), abs(float(model[i])-hi))
            err_lo[i] = max(0.0, lo-float(model[i]), float(model[i])-hi)
            widths[i] = hi-lo
        return lows, highs, err_lo, err_hi, widths
    lows, highs, err_lo, err_hi, widths = bounds()
    status = base.classify(err_lo, err_hi)
    total = int(ns.sum()); looks = 1
    if status:
        return {"status":status,"total_observations":total,"per_cell_observations":ns.tolist(),"looks":looks}
    while True:
        eligible = np.where(ns + BATCH <= MAX_PER_CELL)[0]
        if not len(eligible):
            break
        # If a capped cell still blocks admission but is not conclusively over
        # threshold, further samples elsewhere cannot certify the whole matrix.
        capped_blocker = (ns >= MAX_PER_CELL) & (err_hi > EPS + TOL) & (err_lo <= EPS + TOL)
        if np.any(capped_blocker):
            break
        if strategy == "uniform":
            # Uniform means all cells receive the same count; if any reaches cap,
            # the common allocation is exhausted and the run refuses unresolved.
            if np.any(ns[eligible] != ns.max()) or len(eligible) != k:
                break
            chosen = np.arange(k)
        else:
            blockers = eligible[err_hi[eligible] > EPS + TOL]
            if not len(blockers):
                break
            if strategy == "max_error_upper":
                score = err_hi[blockers]
            elif strategy == "widest_active_ci":
                score = widths[blockers]
            else:
                # Approximate marginal confidence-width reduction per observation:
                # width / n; select only constraints that presently block admission.
                score = widths[blockers] / ns[blockers]
            chosen = np.array([int(blockers[np.argmax(score)])])
        for i in chosen:
            start, stop = int(ns[i]), int(ns[i] + BATCH)
            successes[i] += int(np.sum(streams[i, start:stop]))
            ns[i] += BATCH
        total += int(len(chosen)*BATCH); looks += 1
        lows, highs, err_lo, err_hi, widths = bounds()
        status = base.classify(err_lo, err_hi)
        if status:
            break
    if status is None:
        status = "refuse_unresolved"
    return {"status":status,"total_observations":total,"per_cell_observations":ns.tolist(),"looks":looks}


def wilson(successes: int, trials: int, z: float=1.959963984540054) -> list[float]:
    p=successes/trials; den=1+z*z/trials; center=(p+z*z/(2*trials))/den
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return [max(0.0,center-half),min(1.0,center+half)]


def run(repeats: int=200) -> dict:
    if repeats < 50: raise ValueError("use at least 50 repeats")
    truth=np.array([0.015,0.035,0.075,0.14,0.27,0.50,0.73,0.94],dtype=float)
    models={"correct_model":truth.copy(),"safe_offset_high_variance_cell":truth.copy(),
            "near_over_budget_high_variance_cell":truth.copy(),"clearly_over_budget_high_variance_cell":truth.copy()}
    models["safe_offset_high_variance_cell"][5]+=0.02
    models["near_over_budget_high_variance_cell"][5]+=EPS+0.001
    models["clearly_over_budget_high_variance_cell"][5]+=EPS+0.02
    true_err={name:float(np.max(np.abs(m-truth))) for name,m in models.items()}
    methods=("uniform","max_error_upper","widest_active_ci","marginal_ci_reduction")
    all_rows={name:{meth:[] for meth in methods} for name in models}
    rng=np.random.default_rng(SEED)
    for _ in range(repeats):
        streams=np.vstack([rng.binomial(1,p,size=MAX_PER_CELL) for p in truth]).astype(np.int8)
        for name,model in models.items():
            for meth in methods:
                all_rows[name][meth].append(run_strategy(streams,model,meth))
    result={}
    for name in models:
        result[name]={}
        for meth,rows in all_rows[name].items():
            unsafe=sum(r["status"]=="admit" and true_err[name]>EPS+TOL for r in rows)
            result[name][meth]={
                "true_max_error_for_scoring_only":true_err[name],
                "status_counts":{s:sum(r["status"]==s for r in rows) for s in ("admit","reject","refuse_unresolved")},
                "admission_rate":sum(r["status"]=="admit" for r in rows)/repeats,
                "false_admissions":unsafe,"false_admission_rate":unsafe/repeats,"false_admission_wilson95":wilson(unsafe,repeats),
                "mean_total_observations":float(np.mean([r["total_observations"] for r in rows])),
                "median_total_observations":float(np.median([r["total_observations"] for r in rows])),
                "p95_total_observations":float(np.quantile([r["total_observations"] for r in rows],.95)),
                "mean_per_cell_observations":[float(x) for x in np.mean([r["per_cell_observations"] for r in rows],axis=0)],
            }
    clean_uniform=result["correct_model"]["uniform"]["mean_total_observations"]
    for meth in methods:
        result["correct_model"][meth]["observations_saved_vs_uniform_percent"] = 100*(1-result["correct_model"][meth]["mean_total_observations"]/clean_uniform)
    payload={"schema":SCHEMA,"status":"SYNTHETIC_CONSTRAINT_AWARE_ALLOCATION_NOT_PRODUCTION_CERTIFICATION",
      "protocol":{"seed":SEED,"repeats":repeats,"cells":len(truth),"truth_means_for_generating_streams":[float(x) for x in truth],
        "loss_range":[0,1],"familywise_delta":DELTA,"per_cell_delta":DELTA/len(truth),"epsilon_allowance":EPS,
        "pilot_per_cell":PILOT,"batch_size":BATCH,"max_per_cell":MAX_PER_CELL,
        "stream_reuse":"all strategies see identical per-cell streams within each repeat and scenario",
        "heuristics":{"uniform":"sample every cell at each look","max_error_upper":"sample active blocking cell with largest error upper bound; current adaptive baseline",
          "widest_active_ci":"sample active blocking cell with widest confidence interval",
          "marginal_ci_reduction":"sample active blocking cell maximizing confidence-interval width divided by sample count"}},
      "results":result,
      "limitations":["8-cell iid Bernoulli synthetic game; no real-world loss calibration","all methods use same anytime-valid beta-mixture CS; familywise validity assumes fixed support and independent per-cell streams","allocation strategies are heuristics, not proven optimal","no published APGAI/Almost Tracking implementation run; results are not a direct SOTA comparison","does not solve systematic bias, missing support, changing policy identity, or invalid Bernoulli assumptions","Python-only research lane; no Rust planner/dispatch integration"],
      "environment":{"python":platform.python_version(),"numpy":np.__version__},
      "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return payload

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=200);a=p.parse_args()
 start=time.perf_counter();out=run(a.repeats);raw=(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(raw)
 print(json.dumps({'path':str(a.output),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'runtime_seconds':round(time.perf_counter()-start,3)},indent=2))
