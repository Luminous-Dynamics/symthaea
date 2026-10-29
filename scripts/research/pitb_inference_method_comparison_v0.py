#!/usr/bin/env python3
"""Research-only comparison of PITB paired cluster uncertainty methods.

Compares the current whole-component percentile bootstrap with a weighted
cluster-sandwich t interval under synthetic zero target-weighted-effect data.
This is calibration research, not the production Rust scorer or a qualified
recommendation. No PITB outcome data are read.
"""
from __future__ import annotations
import argparse, json, math, platform
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

SEED = 20261029
P_MEAN, ICC, PAIR_ASSOC, EFFECT_SD = 0.35, 0.10, 0.50, 0.08
BOOT = 299
STUDIES = 1000
PROFILES = {
    "balanced_40": [4] * 40,
    "moderate_tail_40": [1] * 24 + [2] * 10 + [5] * 4 + [10] * 2,
    "heavy_tail_40": [1] * 28 + [4] * 8 + [15] * 4,
    "single_giant_40": [1] * 35 + [5] * 4 + [45],
    "few_clusters_20": [1] * 12 + [3] * 6 + [15] * 2,
    "many_clusters_60": [1] * 42 + [4] * 12 + [15] * 6,
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1-p) / n + z*z/(4*n*n)) / den
    return [center-half, center+half]


def effect_vector(sizes: np.ndarray, rng: np.random.Generator, structure: str) -> np.ndarray:
    if structure == "independent":
        raw = rng.normal(size=len(sizes))
    elif structure == "size_correlated":
        raw = np.log(sizes.astype(float))
    else:
        raise ValueError(structure)
    raw = raw - np.average(raw, weights=sizes)
    sd = np.sqrt(np.average(raw*raw, weights=sizes))
    effect = np.zeros(len(sizes)) if sd < 1e-12 else raw * (EFFECT_SD / sd)
    return effect


def generate_study(sizes: np.ndarray, rng: np.random.Generator, structure: str):
    c = len(sizes)
    concentration = 1 / ICC - 1
    p0 = rng.beta(P_MEAN*concentration, (1-P_MEAN)*concentration, c)
    eff = effect_vector(sizes, rng, structure)
    scale = 1.0
    for delta, p in zip(eff, p0):
        if delta > 0:
            scale = min(scale, (1-p)/delta)
        elif delta < 0:
            scale = min(scale, p/(-delta))
    eff *= max(0.0, min(1.0, scale))
    p1 = p0 + eff
    component_delta_counts = np.zeros(c, dtype=float)
    for j, n in enumerate(sizes):
        a, b = float(p0[j]), float(p1[j])
        joint = a*b + PAIR_ASSOC*(min(a,b)-a*b)
        u = rng.random(int(n))
        both = u < joint
        t_only = (u >= joint) & (u < b)
        c_only = (u >= b) & (u < b+a-joint)
        control = both | c_only
        treatment = both | t_only
        component_delta_counts[j] = float(np.sum(treatment.astype(np.int8)-control.astype(np.int8)))
    truth = float(np.average(eff, weights=sizes))
    return component_delta_counts, truth


def interval_percentile(delta_counts: np.ndarray, sizes: np.ndarray, rng: np.random.Generator, reps: int):
    c = len(sizes)
    index = rng.integers(0, c, size=(reps, c))
    den = sizes[index].sum(axis=1)
    nums = delta_counts[index].sum(axis=1)
    values = nums / den
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def interval_weighted_cluster_t(delta_counts: np.ndarray, sizes: np.ndarray):
    """CR1-style sandwich t interval for target-weighted ratio estimator.

    This is a diagnostic comparator only; it has no CR2/leverage correction.
    Its superpopulation assumptions may not fit PITB's finite frozen cohort.
    """
    c = len(sizes)
    total_n = float(sizes.sum())
    estimate = float(delta_counts.sum()/total_n)
    cluster_scores = delta_counts - estimate*sizes
    variance = (c/(c-1)) * float(np.sum(cluster_scores*cluster_scores)) / (total_n*total_n)
    se = math.sqrt(max(0.0, variance))
    critical = float(student_t.ppf(0.975, df=c-1))
    return estimate-critical*se, estimate+critical*se


def run_cell(profile: str, sizes_list: list[int], structure: str, studies: int, boot: int, seed: int):
    sizes = np.asarray(sizes_list, dtype=int)
    rng = np.random.default_rng(seed)
    by_method = {"component_percentile_bootstrap": [], "weighted_cluster_sandwich_t": []}
    for _ in range(studies):
        counts, truth = generate_study(sizes, rng, structure)
        estimate = float(counts.sum()/sizes.sum())
        bounds = {
            "component_percentile_bootstrap": interval_percentile(counts, sizes, rng, boot),
            "weighted_cluster_sandwich_t": interval_weighted_cluster_t(counts, sizes),
        }
        for method, (lo, hi) in bounds.items():
            informative = bool(math.isfinite(lo) and math.isfinite(hi) and hi-lo > np.finfo(float).eps)
            by_method[method].append({
                "estimate": estimate, "truth": truth, "lower": lo, "upper": hi,
                "informative": informative,
                "covers": bool(informative and lo <= truth <= hi),
                "rejects_zero": bool(informative and lo > 0),
                "passes_primary_utility_gate": bool(informative and estimate >= 0.10 and lo > 0),
            })
    output = []
    for method, rows in by_method.items():
        n = len(rows)
        available = sum(r["informative"] for r in rows)
        rejects = sum(r["rejects_zero"] for r in rows)
        passes = sum(r["passes_primary_utility_gate"] for r in rows)
        covered = sum(r["covers"] for r in rows)
        widths = [r["upper"]-r["lower"] for r in rows if r["informative"]]
        bias_pp = 100*float(np.mean([r["estimate"]-r["truth"] for r in rows]))
        output.append({
            "profile": profile, "effect_structure": structure, "method": method,
            "component_count": len(sizes), "target_count": int(sizes.sum()),
            "size_cv": float(np.std(sizes)/np.mean(sizes)),
            "largest_component_target_share": float(sizes.max()/sizes.sum()),
            "studies": n, "bootstrap_replicates": boot if method == "component_percentile_bootstrap" else None,
            "interval_availability_rate": available/n,
            "conditional_coverage": covered/available if available else None,
            "coverage_mc_wilson95": wilson(covered, available) if available else [None,None],
            "unconditional_null_rejection_rate": rejects/n,
            "null_rejection_mc_wilson95": wilson(rejects, n),
            "false_primary_gate_rate": passes/n,
            "false_primary_gate_mc_wilson95": wilson(passes, n),
            "mean_interval_width_pp": 100*float(np.mean(widths)) if widths else None,
            "mean_estimator_bias_pp": bias_pp,
        })
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=int,default=STUDIES)
    p.add_argument('--bootstrap-reps',type=int,default=BOOT)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--out',type=Path,default=Path('pitb-inference-method-comparison.json'))
    a=p.parse_args()
    if a.studies < 100 or a.bootstrap_reps < 199:
        raise SystemExit('requires >=100 simulated studies and >=199 bootstrap replicates')
    ss=np.random.SeedSequence(a.seed).spawn(len(PROFILES)*2)
    results=[]; i=0
    for profile,sizes in PROFILES.items():
        for structure in ('independent','size_correlated'):
            results.extend(run_cell(profile,sizes,structure,a.studies,a.bootstrap_reps,int(ss[i].generate_state(1)[0])))
            i+=1
    report={
      'schema_version':'pitb.inference-method-comparison.v0',
      'status':'exploratory_synthetic_method_comparison_not_qualification',
      'seed':a.seed,'python_version':platform.python_version(),'numpy_version':np.__version__,
      'scipy_version':__import__('scipy').__version__,
      'assumptions':{'baseline_mean':P_MEAN,'component_icc':ICC,'paired_outcome_association_fraction':PAIR_ASSOC,
        'requested_component_effect_sd':EFFECT_SD,'studies_per_cell':a.studies,'bootstrap_replicates':a.bootstrap_reps,
        'interval_methods':['whole-component target-weighted percentile bootstrap','weighted cluster-sandwich CR1-style t interval with df=components-1, no CR2 correction'],
        'null':'true target-weighted treatment effect is centered to zero; effect heterogeneity may correlate with component size'},
      'results':results,
      'limitations':['Synthetic profiles only; no actual PITB connected-component manifest or outcomes were used.',
        'Monte Carlo Wilson intervals quantify simulation variability only.',
        'The weighted cluster-sandwich t comparator omits CR2/leverage correction and is not endorsed.',
        'Neither method is qualified by this screen; the finite frozen target estimand and component sampling mechanism require independent statistical review.',
        'This Python implementation is not the production Rust scorer and does not report proof-transfer results.']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.out),'scenario_count':len(PROFILES)*2,'method_rows':len(results),'status':report['status']},indent=2))
    for r in results:
        print(f"{r['profile']:18} {r['effect_structure']:15} {r['method']:30} C={r['component_count']:2} N={r['target_count']:3} cov={r['conditional_coverage']:.3f} reject={r['unconditional_null_rejection_rate']:.3f} false_gate={r['false_primary_gate_rate']:.3f} width={r['mean_interval_width_pp']:.2f}pp")

if __name__ == '__main__': main()
