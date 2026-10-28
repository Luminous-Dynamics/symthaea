#!/usr/bin/env python3
"""Stress-test global, per-regime, and worst-regime calibrated ambiguity radii.

Synthetic diagnostic only. The regime label is used for the groupwise-oracle
upper bound; the max-radius method does not need a test-time regime label.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from calibrated_prior_ambiguity_benchmark import (
    TARGET_COVERAGE, draw_true_prior, estimate_prior, feature_matrix,
    feature_radius, conformal_radius, calibrated_polytope, inside_polytope,
    wilson_interval, meta_regime_alphas,
)
from prior_set_coverage_double_oracle_benchmark import (
    OBJECTIVES, build_risk_tensor,
)
from constrained_prior_polytope_benchmark import solve_constrained_minimax

SCHEMA = "symthaea-global-vs-groupwise-calibration-v1"


def run_benchmark(n_calibration=400, n_test=120, seed=20261029,
                  target_coverage=TARGET_COVERAGE, n_observations=120):
    if n_calibration < 20 or n_test < 1:
        raise ValueError("need at least 20 calibration points and one test point")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    alphas = meta_regime_alphas()
    C = feature_matrix()
    calibration_scores = []
    by_regime = [[] for _ in alphas]
    for _ in range(n_calibration):
        q, regime = draw_true_prior(rng, alphas)
        q_hat = estimate_prior(rng, q, n_observations)
        score = feature_radius(q, q_hat, C)
        calibration_scores.append(score)
        by_regime[regime].append(score)
    global_radius = conformal_radius(calibration_scores, target_coverage)
    group_radii = []
    group_sample_sizes = []
    for scores in by_regime:
        group_sample_sizes.append(len(scores))
        if len(scores) < 10:
            # Fail conservatively rather than claim small-sample conditional calibration.
            group_radii.append(1.0)
        else:
            group_radii.append(conformal_radius(scores, target_coverage))
    worst_group_radius = max(group_radii)

    tensor, _, _ = build_risk_tensor()
    rows = []
    for test_id in range(n_test):
        q, regime = draw_true_prior(rng, alphas)
        q_hat = estimate_prior(rng, q, n_observations)
        score = feature_radius(q, q_hat, C)
        radii = {
            "global": global_radius,
            "groupwise_oracle": group_radii[regime],
            "worst_group": worst_group_radius,
        }
        row = {"test_id": test_id, "meta_regime": regime, "feature_score": score,
               "radii": radii,
               "coverage": {k: score <= v + 1e-12 for k, v in radii.items()},
               "feature_width_proxy": {}, "regret_bits": {}}
        for method, radius in radii.items():
            Q = calibrated_polytope(q_hat, radius, floor=0.0, C=C)
            row["coverage"][method] = inside_polytope(q, Q)
            row["feature_width_proxy"][method] = 2.0 * C.shape[0] * radius
            for objective in OBJECTIVES:
                A = tensor[objective]["risk_by_policy_and_world"]
                robust = solve_constrained_minimax(A, Q)
                if robust["status"] != "CERTIFIED_CONSTRAINED_POLYTOPE_GAME":
                    raise RuntimeError(f"uncertified robust policy {method}/{objective}")
                risks = A @ q
                oracle = float(np.min(risks))
                mu = np.asarray(robust["policy_weights"], dtype=float)
                row["regret_bits"].setdefault(method, {})[objective] = max(0.0, float(mu @ risks) - oracle)
        rows.append(row)

    summary = {"target_coverage": target_coverage, "global_radius": global_radius,
               "group_radii": group_radii, "group_calibration_sample_sizes": group_sample_sizes,
               "worst_group_radius": worst_group_radius, "methods": {}}
    methods = ("global", "groupwise_oracle", "worst_group")
    regimes = ["diffuse", "sparse", "graph-skewed", "kernel-skewed", "diagonal-correlated", "anti-correlated"]
    for method in methods:
        covered = sum(r["coverage"][method] for r in rows)
        summary["methods"][method] = {
            "coverage_successes": covered, "n_test": n_test,
            "empirical_coverage": covered / n_test,
            "wilson_95": wilson_interval(covered, n_test),
            "mean_radius": float(np.mean([r["radii"][method] for r in rows])),
            "mean_feature_width_proxy": float(np.mean([r["feature_width_proxy"][method] for r in rows])),
            "coverage_by_regime": {}, "regret_mean_bits": {}, "regret_p95_bits": {},
        }
        for idx, name in enumerate(regimes):
            local = [r for r in rows if r["meta_regime"] == idx]
            count = sum(r["coverage"][method] for r in local)
            summary["methods"][method]["coverage_by_regime"][name] = {
                "covered": count, "n": len(local), "rate": count / len(local) if local else None
            }
        for objective in OBJECTIVES:
            vals = np.asarray([r["regret_bits"][method][objective] for r in rows])
            summary["methods"][method]["regret_mean_bits"][objective] = float(vals.mean())
            summary["methods"][method]["regret_p95_bits"][objective] = float(np.quantile(vals, .95))
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_REGIMEWISE_CALIBRATION_DIAGNOSTIC_NOT_REAL_WORLD_COVERAGE",
        "protocol": {"n_calibration":n_calibration,"n_test":n_test,"seed":seed,
                     "target_coverage":target_coverage,"n_observations":n_observations,
                     "regimes":regimes,
                     "groupwise_oracle_requires_known_test_regime":True,
                     "worst_group_radius_uses_max_of_calibrated_group_radii_and_requires_no_test_time_regime_label":True},
        "model": {"latent_hypotheses":16,"policy_trees":8,"horizon":2,"objectives":list(OBJECTIVES),"feature_count":int(C.shape[0])},
        "summary":summary,"rows":rows,
        "timing_seconds_local":time.perf_counter()-start,
        "limitations":["synthetic meta-population only","groupwise oracle uses known true regime, an unattainable diagnostic unless regime is observed/classified",
                       "max group radius can be conservative","coverage is marginal within each declared group only under within-group exchangeability",
                       "no Symthaea SCM integration or external benchmark"],
    }


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--n-calibration",type=int,default=400)
    ap.add_argument("--n-test",type=int,default=120)
    ap.add_argument("--seed",type=int,default=20261029)
    ap.add_argument("--observations",type=int,default=120)
    args=ap.parse_args()
    r=run_benchmark(args.n_calibration,args.n_test,args.seed,n_observations=args.observations)
    raw=(json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"output":str(args.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
      "global_radius":r["summary"]["global_radius"],"worst_group_radius":r["summary"]["worst_group_radius"],
      "coverage":{k:v["empirical_coverage"] for k,v in r["summary"]["methods"].items()}},indent=2))

if __name__=="__main__": main()


