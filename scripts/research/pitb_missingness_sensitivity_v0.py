#!/usr/bin/env python3
"""Independent, research-only synthetic PITB missingness/estimand screening.
Not the production Rust scorer; omitted-row results are invalid complete-case
comparators, not permitted confirmatory benchmark outcomes.
"""
import argparse, csv, json, platform
from pathlib import Path
import numpy as np

SEED = 20261028
P_MEAN, ICC, PAIR_ASSOC, EFFECT_SD = 0.35, 0.10, 0.50, 0.08

def sizes(profile, c=40):
    if profile == "balanced": return np.full(c, 4, dtype=int)
    x = np.ones(c, dtype=int); x[int(.70*c):int(.90*c)] = 4; x[int(.90*c):] = 15
    return x

def run_one(sz, name, rng, boot):
    c, total = len(sz), int(sz.sum())
    conc = 1/ICC - 1
    p0 = rng.beta(P_MEAN*conc, (1-P_MEAN)*conc, c)
    raw = -np.log(sz.astype(float)) if name == "size_correlated_complete" else rng.normal(size=c)
    raw = raw - np.average(raw, weights=sz)
    sd = np.sqrt(np.average(raw**2, weights=sz))
    eff = np.zeros(c) if sd < 1e-12 else raw * EFFECT_SD/sd
    scale = 1.0
    for e, p in zip(eff, p0):
        if e > 0: scale = min(scale, (1-p)/e)
        elif e < 0: scale = min(scale, p/(-e))
    eff *= max(0.0, min(1.0, scale))
    p1 = p0 + eff
    base, treat, fam = [], [], []
    for j, n in enumerate(sz):
        a, b = float(p0[j]), float(p1[j])
        joint = a*b + PAIR_ASSOC*(min(a,b)-a*b)
        u = rng.random(int(n)); both = u < joint
        t_only = (u >= joint) & (u < b)
        b_only = (u >= b) & (u < b + a - joint)
        base.append((both | b_only).astype(np.int8))
        treat.append((both | t_only).astype(np.int8))
        fam.append(np.full(int(n), j, dtype=np.int32))
    base, treat, fam = np.concatenate(base), np.concatenate(treat), np.concatenate(fam)
    capability_true = float(np.average(eff, weights=sz))
    true = capability_true
    observed = np.ones(total, dtype=bool)
    if name == "mcar_row_omission_15":
        observed = rng.random(total) >= .15
    elif name == "mnar_treatment_loss_60_vs_05":
        loss = (base == 1) & (treat == 0)
        observed = rng.random(total) >= np.where(loss, .60, .05)
    elif name == "size_informative_component_omission":
        keep = rng.random(c) >= np.where(sz >= np.median(sz), .30, .05)
        observed = keep[fam]
    elif name == "size_informative_runtime_outage":
        out = np.where(sz >= np.median(sz), .35, .05)
        treat = treat * (rng.random(total) >= out[fam]).astype(np.int8)
        true = float(np.average(p1*(1-out)-p0, weights=sz))
    elif name not in ("complete_balanced", "complete_heavy_tailed", "size_correlated_complete"):
        raise ValueError(name)
    delta = (treat-base)[observed]; fobs = fam[observed]
    fdelta = np.bincount(fobs, weights=delta, minlength=c).astype(float)
    fn = np.bincount(fobs, minlength=c).astype(float)
    estimate = fdelta.sum()/fn.sum() if fn.sum() else float("nan")
    ix = rng.integers(0, c, size=(boot,c))
    den = fn[ix].sum(axis=1); nums = fdelta[ix].sum(axis=1)
    bvals = nums[den > 0]/den[den > 0]
    lo, hi = np.quantile(bvals, [.025,.975]) if len(bvals) >= max(100,int(.9*boot)) else (np.nan,np.nan)
    informative = bool(np.isfinite(lo) and np.isfinite(hi) and hi-lo > np.finfo(float).eps)
    return dict(est=estimate, lo=lo, hi=hi, true=true, observed=float(observed.mean()),
                informative=informative, cover=bool(informative and lo <= true <= hi),
                reject=bool(informative and lo > 0), gate=bool(informative and estimate >= .10 and lo > 0 and c >= 20),
                fam_effect=float(np.mean(eff)), capability=capability_true)

def scenario(name, sz, studies, boot, seed):
    rng=np.random.default_rng(seed); rows=[run_one(sz,name,rng,boot) for _ in range(studies)]
    est=np.array([r["est"] for r in rows]); truth=np.array([r["true"] for r in rows]); bias=est-truth
    avail=np.array([r["informative"] for r in rows]); cover=np.array([r["cover"] for r in rows])
    return {
      "scenario_id":name,"component_count":len(sz),"target_count":int(sz.sum()),
      "size_min":int(sz.min()),"size_median":float(np.median(sz)),"size_max":int(sz.max()),
      "studies":studies,"bootstrap_replicates_per_study":boot,
      "mean_observed_fraction":float(np.mean([r["observed"] for r in rows])),
      "mean_true_effect_percentage_points":float(100*truth.mean()),
      "mean_estimated_effect_percentage_points":float(100*np.nanmean(est)),
      "bias_percentage_points":float(100*np.nanmean(bias)),
      "rmse_percentage_points":float(100*np.sqrt(np.nanmean(bias*bias))),
      "interval_availability_probability":float(avail.mean()),
      "coverage_conditional_on_informative_interval":float(cover.sum()/avail.sum()) if avail.sum() else None,
      "lower_bound_rejection_rate_unconditional":float(np.mean([r["reject"] for r in rows])),
      "primary_utility_gate_pass_rate_unconditional":float(np.mean([r["gate"] for r in rows])),
      "mean_interval_width_percentage_points":float(100*np.nanmean([r["hi"]-r["lo"] for r in rows])),
      "mean_family_average_effect_percentage_points":float(100*np.mean([r["fam_effect"] for r in rows])),
      "family_minus_capability_target_estimand_pp":float(100*np.mean([r["fam_effect"]-r["capability"] for r in rows])),
      "interpretation":("explicit outage is an observed non-solve; estimand is operational-system success" if name=="size_informative_runtime_outage" else "adversarial omitted-row complete-case comparator; forbidden as confirmatory PITB analysis" if "omission" in name or "mnar" in name else "synthetic reference only")}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--studies',type=int,default=1000); p.add_argument('--bootstrap-reps',type=int,default=299); p.add_argument('--seed',type=int,default=SEED); p.add_argument('--out',type=Path,default=Path('pitb-missingness-sensitivity-results.json')); a=p.parse_args()
    if a.studies < 100 or a.bootstrap_reps < 199: raise SystemExit('screening requires >=100 studies and >=199 bootstrap replicates')
    work=[('complete_balanced',sizes('balanced')),('complete_heavy_tailed',sizes('heavy_tailed')),('size_correlated_complete',sizes('heavy_tailed')),('mcar_row_omission_15',sizes('heavy_tailed')),('mnar_treatment_loss_60_vs_05',sizes('heavy_tailed')),('size_informative_component_omission',sizes('heavy_tailed')),('size_informative_runtime_outage',sizes('heavy_tailed'))]
    results=[scenario(n,s,a.studies,a.bootstrap_reps,a.seed+i*0x9E3779B1) for i,(n,s) in enumerate(work)]
    report={"schema_version":"pitb.independent-missingness-sensitivity.v0","status":"exploratory_synthetic_screen_not_qualification","seed":a.seed,"numpy_version":np.__version__,"python_version":platform.python_version(),"baseline_mean_assumption":P_MEAN,"family_icc_assumption":ICC,"pair_association_fraction_assumption":PAIR_ASSOC,"requested_family_effect_sd":EFFECT_SD,"scenarios":results,"limitations":["Synthetic profiles only; no actual PITB test/validation outcomes or audited component sizes were used.","Screening repetitions are not confirmatory Monte Carlo precision.","Omitted-row complete-case scenarios are deliberately invalid comparators; confirmatory runs must fail closed.","Outcome-dependent missingness is not identifiable from observed rows without additional assumptions.","This NumPy harness is independent research code, not the production Rust scorer or independent statistical review."]}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with a.out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(json.dumps({"json":str(a.out),"csv":str(a.out.with_suffix('.csv')),"scenario_count":len(results)},indent=2))
if __name__=='__main__': main()
