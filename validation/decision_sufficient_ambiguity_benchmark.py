#!/usr/bin/env python3
"""Compare prior-feature and decision-risk ambiguity geometries on the same game.

Synthetic, finite-horizon benchmark only. A risk-space region covers the vector of
expected losses of all candidate policies, not the latent prior itself. It is a
candidate decision-sufficient representation, not a claim of general sufficiency.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import groupwise_robust_coverage_benchmark as g
from prior_set_coverage_double_oracle_benchmark import N_HYP, build_risk_tensor

SCHEMA = "symthaea-decision-sufficient-ambiguity-v1"
METHODS = ("feature_pooled", "feature_group_max", "risk_pooled", "risk_group_max", "full_prior")


def score(sample, budget, geometry):
    return float(np.max(np.abs(geometry @ sample["q"] - geometry @ sample["est"][budget])))


def quantile_for(samples, regime, budget, geometry, target):
    vals = [score(s, budget, geometry) for s in samples if regime is None or s["regime"] == regime]
    if not vals:
        raise ValueError("empty calibration subset")
    return g.conformal_quantile(vals, target)


def make_region(qhat, radius, geometry, unbounded=False):
    n = qhat.size
    if unbounded:
        return {"floor": 0.0, "A_ub": np.empty((0, n)), "b_ub": np.empty((0,)), "radius": None}
    center = geometry @ qhat
    lo = np.maximum(np.min(geometry, axis=1), center - radius)
    hi = np.minimum(np.max(geometry, axis=1), center + radius)
    aub=[]; bub=[]
    for j,row in enumerate(geometry):
        aub.append(row); bub.append(float(hi[j]))
        aub.append(-row); bub.append(float(-lo[j]))
    return {"floor":0.0,"A_ub":np.asarray(aub,dtype=float),"b_ub":np.asarray(bub,dtype=float),"radius":float(radius)}


def included(q, geometry, region, tol=1e-9):
    if region["A_ub"].shape[0] == 0:
        return True
    return bool(np.max(region["A_ub"] @ q - region["b_ub"]) <= tol and abs(float(q.sum())-1.0)<=tol)


def predict(q, qhat, radius, geometry, risk, unbounded=False):
    region=make_region(qhat, radius, geometry, unbounded=unbounded)
    sol=g.solve_robust_game(risk,region)
    actual_policy_risks=risk @ q
    oracle=float(np.min(actual_policy_risks))
    realized=float(sol["mu"] @ actual_policy_risks)
    return {"covered":included(q,geometry,region),"regret_bits":max(0.0,realized-oracle),
        "realized_risk_bits":realized,"oracle_known_prior_risk_bits":oracle,
        "robust_value_bits":float(sol["value"]),"certificate_gap_bits":float(sol["gap"])}


def summarize(rows, method, geometry_kind, radius_for_regime):
    vals=[r["methods"][method] for r in rows]
    covered=sum(v["covered"] for v in vals); n=len(vals)
    groups={}
    for name in g.REGIMES:
        sub=[r["methods"][method]["covered"] for r in rows if r["regime"]==name]
        groups[name]={"covered":sum(sub),"n":len(sub),"rate":float(np.mean(sub)),"wilson95":g.wilson(sum(sub),len(sub))}
    regrets=np.asarray([v["regret_bits"] for v in vals])
    return {"geometry":"unrestricted_prior" if method=="full_prior" else geometry_kind,
       "coverage_semantics":("all priors admitted by definition" if method=="full_prior" else
          "coverage of 10 prior features" if geometry_kind=="prior_feature_space" else "coverage of the vector of 8 candidate-policy expected risks"),
       "coverage":{"covered":int(covered),"n":n,"rate":float(covered/n),"wilson95":g.wilson(covered,n)},
       "coverage_by_regime":groups,"worst_regime_coverage":float(min(x["rate"] for x in groups.values())),
       "mean_regret_bits":float(regrets.mean()),"p95_regret_bits":float(np.quantile(regrets,.95)),
       "mean_radius":None if method=="full_prior" else float(np.mean([r["methods"][method]["radius"] for r in rows])),
       "max_certificate_gap_bits":float(max(v["certificate_gap_bits"] for v in vals))}


def paired_bootstrap_ci(delta, seed, reps=2000):
    return g.paired_bootstrap_mean_ci(np.asarray(delta,dtype=float),seed=seed,reps=reps)


def run(n_cal_per_regime=100,n_test_per_regime=30,seed=20261031,budgets=(60,120,240),targets=(0.90,0.95)):
    t0=time.perf_counter(); rng=np.random.default_rng(seed); alphas=g.meta_alphas(); features=g.feature_matrix()
    cal=g.balanced_samples(rng,alphas,n_cal_per_regime,budgets)
    test=g.balanced_samples(rng,alphas,n_test_per_regime,budgets)
    tensor,_,_=build_risk_tensor(); risk=np.asarray(tensor["joint"]["risk_by_policy_and_world"],dtype=float)
    settings=[]
    for budget in budgets:
      for target in targets:
        fglob=quantile_for(cal,None,budget,features,target)
        fgroup={i:quantile_for(cal,i,budget,features,target) for i in range(len(g.REGIMES))}
        rglob=quantile_for(cal,None,budget,risk,target)
        rgroup={i:quantile_for(cal,i,budget,risk,target) for i in range(len(g.REGIMES))}
        radii={"feature_pooled":{i:fglob for i in range(len(g.REGIMES))},
          "feature_group_max":{i:max(fgroup.values()) for i in range(len(g.REGIMES))},
          "risk_pooled":{i:rglob for i in range(len(g.REGIMES))},
          "risk_group_max":{i:max(rgroup.values()) for i in range(len(g.REGIMES))},
          "full_prior":{i:None for i in range(len(g.REGIMES))}}
        rows=[]
        for ti,s in enumerate(test):
          row={"test_id":ti,"regime":g.REGIMES[s["regime"]],"methods":{}}
          for method in METHODS:
            geom=features if method.startswith("feature_") else risk
            geo_kind="prior_feature_space" if method.startswith("feature_") else "decision_risk_space"
            pred=predict(s["q"],s["est"][budget],radii[method][s["regime"]],geom,risk,unbounded=(method=="full_prior"))
            row["methods"][method]={"radius":radii[method][s["regime"]],**pred}
          rows.append(row)
        summaries={}
        kinds={"feature_pooled":"prior_feature_space","feature_group_max":"prior_feature_space",
               "risk_pooled":"decision_risk_space","risk_group_max":"decision_risk_space","full_prior":"unrestricted_prior"}
        for method in METHODS:
          s=summarize(rows,method,kinds[method],radii[method]); regs=np.asarray([r["methods"][method]["regret_bits"] for r in rows])
          if method != "feature_pooled":
            delta=regs-np.asarray([r["methods"]["feature_pooled"]["regret_bits"] for r in rows])
            s["paired_regret_delta_vs_feature_pooled_bits"]={"mean":float(delta.mean()),"bootstrap95_ci":paired_bootstrap_ci(delta,seed+budget+round(target*100)+sum(map(ord,method)))}
          summaries[method]=s
        settings.append({"target":target,"budget":budget,"radii":{"feature_pooled":fglob,
          "feature_group_by_regime":{g.REGIMES[i]:fgroup[i] for i in fgroup},"feature_group_max":max(fgroup.values()),
          "risk_pooled":rglob,"risk_group_by_regime":{g.REGIMES[i]:rgroup[i] for i in rgroup},"risk_group_max":max(rgroup.values())},
          "methods":summaries,"rows":rows})
    return {"schema":SCHEMA,"status":"SYNTHETIC_FINITE_GAME_NOT_REAL_SCM_OR_EXTERNAL_SOTA_BENCHMARK",
      "protocol":{"seed":seed,"calibration_per_regime":n_cal_per_regime,"test_per_regime":n_test_per_regime,
        "calibration_total":len(cal),"test_total":len(test),"budgets":list(budgets),"targets":list(targets),"regimes":list(g.REGIMES),
        "same_calibration_and_test_episodes_across_geometries":True,"nested_observation_prefixes":True,
        "feature_geometry":"10 graph/kernel/diagonal prior features; coverage means these features lie in calibrated box",
        "decision_risk_geometry":"8 expected losses for candidate policies; coverage means the entire policy-risk vector lies in calibrated box",
        "full_prior":"simplex-only ambiguity region; coverage is trivially 100 percent and regret is a conservative baseline",
        "score":"maximum absolute coordinate error in the selected geometry, with split-conformal finite-sample quantile",
        "regime_max":"max of per-regime conformal radii, applied as a single common radius without a test-time regime label"},
      "model":{"latent_hypotheses":N_HYP,"candidate_policies":int(risk.shape[0]),"horizon":2,"objective":"joint cross-entropy regret bits"},
      "environment":{"python":platform.python_version(),"numpy":np.__version__},"runtime_seconds_local":time.perf_counter()-t0,
      "limitations":["synthetic prior population and finite game","risk space is decision-sufficient only for the currently enumerated policy loss vectors; not for unseen actions or alternate objectives",
       "no real SCM integration or external benchmark run","coverage guarantees depend on exchangeability within calibration/test regimes and known regime family","LP certificate gap is a floating-point numerical check, not formal interval arithmetic","no SOTA claim"],"settings":settings}

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--cal-per-regime",type=int,default=100);p.add_argument("--test-per-regime",type=int,default=30);a=p.parse_args()
 out=run(a.cal_per_regime,a.test_per_regime);raw=(json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(raw)
 print(json.dumps({"output":str(a.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"settings":len(out["settings"]),"max_certificate_gap_bits":max(m["max_certificate_gap_bits"] for s in out["settings"] for m in s["methods"].values())},indent=2))
