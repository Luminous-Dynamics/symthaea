#!/usr/bin/env python3
"""Paired experiment: pooled vs group-robust conformal ambiguity, fixed budget.

Synthetic finite-game research fixture, not a real Symthaea SCM integration or
external SOTA benchmark. Equal-sized calibration/test samples per known synthetic
regime isolate subgroup undercoverage from mixture-frequency effects. The
regime-conditional radius is an oracle diagnostic requiring regime metadata at
prediction time; the group-max radius is the deployable label-free common radius
for the declared regime taxonomy.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from prior_set_coverage_double_oracle_benchmark import N_HYP, N_GRAPH, N_KERNEL, OBJECTIVES, build_risk_tensor

REGIMES = ("diffuse", "sparse", "graph-skewed", "kernel-skewed", "diagonal-correlated", "anti-correlated")
MARGIN = 1e-9


def feature_matrix():
    rows=[]
    for g in range(N_GRAPH): rows.append([1.0 if i // N_KERNEL == g else 0.0 for i in range(N_HYP)])
    for k in range(N_KERNEL): rows.append([1.0 if i % N_KERNEL == k else 0.0 for i in range(N_HYP)])
    rows.append([1.0 if i // N_KERNEL == i % N_KERNEL else 0.0 for i in range(N_HYP)])
    rows.append([1.0 if i % N_KERNEL == N_KERNEL-1-i//N_KERNEL else 0.0 for i in range(N_HYP)])
    return np.asarray(rows,dtype=float)


def meta_alphas():
    alpha=[np.full(N_HYP,2.0),np.full(N_HYP,0.15)]
    gw=np.asarray([0.55,0.25,0.13,0.07]); alpha.append(np.asarray([0.12+2.5*gw[g] for g in range(4) for _k in range(4)]))
    kw=np.asarray([0.52,0.26,0.14,0.08]); alpha.append(np.asarray([0.12+2.5*kw[k] for _g in range(4) for k in range(4)]))
    alpha.append(np.asarray([2.8 if g==k else 0.13 for g in range(4) for k in range(4)]))
    alpha.append(np.asarray([2.8 if k==3-g else 0.13 for g in range(4) for k in range(4)]))
    return tuple(alpha)


def conformal_quantile(scores,target):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or len(values)==0 or np.any(~np.isfinite(values)): raise ValueError("invalid calibration scores")
    if not 0<target<1: raise ValueError("target coverage outside (0,1)")
    rank=math.ceil((len(values)+1)*target)
    return 1.0 if rank>len(values) else float(np.sort(values)[rank-1])


def make_polytope(qhat,radius,C,floor=0.0):
    center=C@qhat; lo=np.maximum(0.0,center-radius); hi=np.minimum(1.0,center+radius)
    Aub=[]; bub=[]
    for j,row in enumerate(C):
        Aub.append(row); bub.append(hi[j]); Aub.append(-row); bub.append(-lo[j])
    return {"floor":floor,"A_ub":np.asarray(Aub),"b_ub":np.asarray(bub),"radius":radius}


def solve_robust_game(A,Q):
    A=np.asarray(A,dtype=float); n_policy,n_world=A.shape
    risk_ub=np.column_stack([-A,np.ones(n_policy)])
    poly_ub=np.column_stack([Q["A_ub"],np.zeros(len(Q["b_ub"]))])
    Aub=np.vstack([risk_ub,poly_ub]); bub=np.r_[np.zeros(n_policy),Q["b_ub"]]
    Aeq=np.r_[np.ones(n_world),0.0][None,:]
    res=linprog(np.r_[np.zeros(n_world),-1.0],A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=[1.0],
                bounds=[(Q["floor"],1.0)]*n_world+[(None,None)],method="highs")
    if not res.success: raise RuntimeError("robust value LP failed: "+res.message)
    qstar=res.x[:n_world]; value=float(res.x[-1])
    dual=-np.asarray(res.ineqlin.marginals[:n_policy],dtype=float); dual=np.maximum(dual,0.0)
    if dual.sum()<=0: raise RuntimeError("dual policy mixture has no positive mass")
    mu=dual/dual.sum()
    upper=linprog(-(A.T@mu),A_ub=Q["A_ub"],b_ub=Q["b_ub"],A_eq=np.ones((1,n_world)),b_eq=[1.0],
                  bounds=[(Q["floor"],1.0)]*n_world,method="highs")
    if not upper.success: raise RuntimeError("independent world oracle failed: "+upper.message)
    lower_raw=float(np.min(A@qstar)); upper_raw=float(-upper.fun)
    if lower_raw>upper_raw+2e-7 or abs(value-lower_raw)>2e-7 or abs(value-upper_raw)>2e-7:
        raise AssertionError(f"certificate mismatch lower={lower_raw}, lp={value}, upper={upper_raw}")
    gap=(upper_raw+MARGIN)-(lower_raw-MARGIN)
    return {"mu":mu,"value":value,"gap":gap,"lower":lower_raw-MARGIN,"upper":upper_raw+MARGIN}


def wilson(success,n,z=1.959963984540054):
    p=success/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0,ctr-half),min(1.0,ctr+half)]


SCHEMA = "symthaea-groupwise-robust-coverage-benchmark-v1"
TARGETS = (0.90, 0.95)
BUDGETS = (60, 120, 240)
N_CAL_PER_REGIME = 100
N_TEST_PER_REGIME = 30
SEED = 20261031
METHODS = ("pooled_global", "group_max_common_radius", "regime_conditional_oracle", "unrestricted_full_simplex")


def nested_estimates(rng, q, budgets=BUDGETS):
    obs = rng.choice(N_HYP, size=max(budgets), p=q)
    result = {}
    for n in budgets:
        counts = np.bincount(obs[:n], minlength=N_HYP)
        result[n] = (counts + 0.5) / (n + 0.5 * N_HYP)
    return result


def balanced_samples(rng, alphas, per_regime, budgets):
    out=[]
    for regime, alpha in enumerate(alphas):
        for i in range(per_regime):
            q=rng.dirichlet(alpha)
            out.append({"q":q,"regime":regime,"est":nested_estimates(rng,q,budgets)})
    return out


def max_group_radius(group_radii):
    values=[float(v) for v in group_radii.values()]
    if not values or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("group radii must be a non-empty set of finite nonnegative values")
    return max(values)


def score_radius(samples, regime, budget, C, target):
    scores=[float(np.max(np.abs(C@x["q"]-C@x["est"][budget]))) for x in samples if regime is None or x["regime"]==regime]
    if not scores: raise ValueError("no calibration samples for requested regime")
    return conformal_quantile(scores,target)


def paired_bootstrap_mean_ci(deltas, seed=91, reps=4000):
    x=np.asarray(deltas,dtype=float)
    rng=np.random.default_rng(seed)
    means=np.empty(reps,dtype=float)
    # Chunking prevents large transient allocations for future larger panels.
    for start in range(0,reps,200):
        m=min(200,reps-start)
        idx=rng.integers(0,len(x),size=(m,len(x)))
        means[start:start+m]=x[idx].mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def coverage_entry(q, center, radius, C):
    Q=make_polytope(center,radius,C)
    return bool(np.max(Q["A_ub"]@q-Q["b_ub"])<=1e-9 and abs(float(q.sum())-1.0)<=1e-9)


def one_prediction(q, qhat, radius, C, risk):
    Q=make_polytope(qhat,radius,C)
    sol=solve_robust_game(risk,Q)
    risks=risk@q
    oracle=float(np.min(risks))
    realized=float(sol["mu"]@risks)
    covered=bool(np.max(Q["A_ub"]@q-Q["b_ub"])<=1e-9 and abs(float(q.sum())-1.0)<=1e-9)
    return {"covered":covered,"regret_bits":max(0.0,realized-oracle),"realized_risk_bits":realized,
            "oracle_known_prior_risk_bits":oracle,"robust_value_bits":float(sol["value"]),"certificate_gap_bits":float(sol["gap"])}


def run(n_cal_per_regime=N_CAL_PER_REGIME,n_test_per_regime=N_TEST_PER_REGIME,seed=SEED):
    start=time.perf_counter(); rng=np.random.default_rng(seed); alphas=meta_alphas(); C=feature_matrix()
    calibration=balanced_samples(rng,alphas,n_cal_per_regime,BUDGETS)
    test=balanced_samples(rng,alphas,n_test_per_regime,BUDGETS)
    tensor,_,_=build_risk_tensor(); risk=np.asarray(tensor["joint"]["risk_by_policy_and_world"],dtype=float)
    settings=[]; rows_by_setting={}
    for budget in BUDGETS:
        for target in TARGETS:
            pooled=score_radius(calibration,None,budget,C,target)
            group={i:score_radius(calibration,i,budget,C,target) for i in range(len(REGIMES))}
            radii={"pooled_global":{i:pooled for i in range(len(REGIMES))},
                   "group_max_common_radius":{i:max_group_radius(group) for i in range(len(REGIMES))},
                   "regime_conditional_oracle":group,
                   "unrestricted_full_simplex":{i:1.0 for i in range(len(REGIMES))}}
            rows=[]
            for ti,x in enumerate(test):
                q=x["q"]; reg=x["regime"]; qhat=x["est"][budget]
                row={"test_id":ti,"regime":REGIMES[reg],"methods":{}}
                for method in METHODS:
                    radius=radii[method][reg]
                    row["methods"][method]={"radius":float(radius),**one_prediction(q,qhat,radius,C,risk)}
                rows.append(row)
            pooled_reg=np.asarray([r["methods"]["pooled_global"]["regret_bits"] for r in rows])
            summaries={}
            for method in METHODS:
                vals=[r["methods"][method] for r in rows]
                covered=sum(v["covered"] for v in vals)
                per_regime={}
                for ri,name in enumerate(REGIMES):
                    sub=[r["methods"][method]["covered"] for r in rows if r["regime"]==name]
                    per_regime[name]={"covered":int(sum(sub)),"n":len(sub),"rate":float(np.mean(sub)),"wilson95":wilson(sum(sub),len(sub))}
                reg=np.asarray([v["regret_bits"] for v in vals])
                summary={"coverage":{"covered":int(covered),"n":len(vals),"rate":float(covered/len(vals)),"wilson95":wilson(covered,len(vals))},
                    "coverage_by_regime":per_regime,"worst_regime_coverage":float(min(z["rate"] for z in per_regime.values())),
                    "mean_regret_bits":float(reg.mean()),"p95_regret_bits":float(np.quantile(reg,.95)),
                    "mean_radius":float(np.mean([v["radius"] for v in vals])),
                    "width_proxy":float(np.mean([2*C.shape[0]*v["radius"] for v in vals])),
                    "max_certificate_gap_bits":float(max(v["certificate_gap_bits"] for v in vals))}
                if method!="pooled_global":
                    delta=reg-pooled_reg
                    summary["paired_regret_delta_vs_pooled_bits"]={"mean":float(delta.mean()),"bootstrap95_ci":paired_bootstrap_mean_ci(delta,seed=seed+budget+int(target*100)+(sum(ord(c) for c in method)),reps=2000)}
                summaries[method]=summary
            key=f"target={target:.2f};budget={budget}"
            settings.append({"target":target,"budget":budget,"radii":{"pooled_global":pooled,
                "group_radii_by_regime":{REGIMES[i]:group[i] for i in group},"group_max_common_radius":max_group_radius(group)},"methods":summaries})
            rows_by_setting[key]=rows
    return {"schema":SCHEMA,"status":"SYNTHETIC_FINITE_GAME_NOT_REAL_SCM_OR_EXTERNAL_BENCHMARK",
      "protocol":{"seed":seed,"calibration_per_regime":n_cal_per_regime,"test_per_regime":n_test_per_regime,
        "calibration_total":len(calibration),"test_total":len(test),"targets":list(TARGETS),"budgets":list(BUDGETS),
        "regimes":list(REGIMES),"balanced_regime_sampling":True,"same_samples_and_observation_prefixes_across_methods":True,
        "methods":{"pooled_global":"one conformal radius over pooled calibration scores",
          "group_max_common_radius":"max of per-regime conformal radii, deployed as a single radius without test-time regime labels",
          "regime_conditional_oracle":"per-regime radius; diagnostic only, requires true regime label at prediction time",
          "unrestricted_full_simplex":"radius 1 removes all feature constraints and approaches unrestricted-prior minimax policy"},
        "calibration_quantile":"ceil((n+1)*target) order statistic; radius=1 if order exceeds n",
        "features":["graph marginals x4","kernel marginals x4","diagonal mass","anti-diagonal mass"],
        "estimator":"multinomial samples with symmetric Dirichlet(0.5) smoothing; each budget is a nested prefix",
        "test_population":"equal sample count per known synthetic regime, so worst-regime coverage is directly observable"},
      "model":{"latent_hypotheses":N_HYP,"policy_trees":8,"horizon":2,"objective":"joint cross-entropy regret in bits","feature_count":int(C.shape[0])},
      "environment":{"python":platform.python_version(),"numpy":np.__version__},"runtime_seconds_local":time.perf_counter()-start,
      "limitations":["synthetic prior meta-population and finite decision game","regime taxonomy assumed known in calibration for group-max; unseen regimes are not protected",
        "regime-conditional result is an oracle unless regime metadata is available at prediction time","10-dimensional feature radius does not constrain every direction of the full joint prior",
        "finite empirical coverage has sampling uncertainty; intervals are Wilson intervals, not proof of future conditional coverage",
        "no Symthaea SCM integration, external benchmark run, or SOTA claim"],
      "settings":settings,"rows_by_setting":rows_by_setting}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--cal-per-regime",type=int,default=N_CAL_PER_REGIME);p.add_argument("--test-per-regime",type=int,default=N_TEST_PER_REGIME);a=p.parse_args()
    result=run(a.cal_per_regime,a.test_per_regime)
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(raw)
    print(json.dumps({"output":str(a.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"settings":len(result["settings"]),
      "max_certificate_gap_bits":max(m["max_certificate_gap_bits"] for s in result["settings"] for m in s["methods"].values())},indent=2))
