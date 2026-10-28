#!/usr/bin/env python3
"""Falsification suite for decision-sufficient ambiguity: risk misspecification and OOD regime.

Synthetic finite-game diagnostic only; no production, real-SCM, or SOTA claim.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import groupwise_robust_coverage_benchmark as g
import decision_sufficient_ambiguity_benchmark as d
from prior_set_coverage_double_oracle_benchmark import N_HYP, build_risk_tensor

SEED = 20261032
BUDGET = 120
TARGET = 0.90
N_CAL_PER_REGIME = 100
N_TEST_PER_REGIME = 20
OOD_TEST_N = 800


def novel_regime_alpha():
    # Unseen conflict regime concentrated on latent hypotheses 9 and 15,
    # whose true eight-policy risk vectors have the largest Euclidean separation.
    # Neither this regime nor these alpha values appear in calibration.
    a = np.full(N_HYP, 0.002, dtype=float)
    a[9] = 0.5
    a[15] = 0.5
    return a


def perturbed_risk(risk, level, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=risk.shape)
    # Level is a fraction of the global per-cell risk standard deviation.
    return np.maximum(0.01, risk + level * float(np.std(risk)) * noise)


def summarize(values, include_bounds=False):
    covered = [x for x in values if x["covered"]]
    result = {
        "n": len(values),
        "coverage": float(np.mean([x["covered"] for x in values])) if values else None,
        "coverage_wilson95": g.wilson(sum(x["covered"] for x in values), len(values)) if values else None,
        "mean_regret_bits": float(np.mean([x["regret_bits"] for x in values])) if values else None,
        "p95_regret_bits": float(np.quantile([x["regret_bits"] for x in values], .95)) if values else None,
        "covered_n": len(covered),
        "covered_case_mean_regret_bits": float(np.mean([x["regret_bits"] for x in covered])) if covered else None,
        "bound_2r_violations": sum(x["regret_bits"] > 2 * x["radius"] + 1e-7 for x in covered if x["radius"] is not None) if include_bounds else None,
        "max_covered_regret_over_2r": float(max((x["regret_bits"] / (2*x["radius"]) for x in covered if x["radius"] is not None and x["radius"] > 0), default=0.0)) if include_bounds else None,
        "max_certificate_gap_bits": max((x["certificate_gap_bits"] for x in values), default=0.0),
    }
    return result


def prediction(q, qhat, radius, planning_geometry, planning_risk, true_risk):
    region = d.make_region(qhat, radius, planning_geometry)
    solution = g.solve_robust_game(planning_risk, region)
    actual = true_risk @ q
    actual_chosen = float(solution["mu"] @ actual)
    oracle = float(np.min(actual))
    covered = d.included(q, planning_geometry, region)
    return {"covered": covered, "regret_bits": max(0.0, actual_chosen - oracle),
            "radius": float(radius), "realized_risk_bits": actual_chosen,
            "oracle_known_prior_risk_bits": oracle, "certificate_gap_bits": float(solution["gap"])}


def run():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    alphas = g.meta_alphas()
    calibration = g.balanced_samples(rng, alphas, N_CAL_PER_REGIME, (60, BUDGET, 240))
    test = g.balanced_samples(rng, alphas, N_TEST_PER_REGIME, (60, BUDGET, 240))
    tensor, _, _ = build_risk_tensor()
    true_risk = np.asarray(tensor["joint"]["risk_by_policy_and_world"], dtype=float)

    conditions = {
        "correct_model": true_risk,
        "risk_misspec_5pct": perturbed_risk(true_risk, .05, SEED + 5),
        "risk_misspec_20pct": perturbed_risk(true_risk, .20, SEED + 20),
        "policy_row_identity_mismatch": true_risk[[3, 1, 7, 0, 6, 2, 5, 4], :],
    }
    known_results = {}
    clean_pooled_regrets = None
    clean_group_max_regrets = None
    for name, plan_risk in conditions.items():
        pooled_radius = d.quantile_for(calibration, None, BUDGET, plan_risk, TARGET)
        group_radii = {i: d.quantile_for(calibration, i, BUDGET, plan_risk, TARGET) for i in range(len(g.REGIMES))}
        group_max = max(group_radii.values())
        pooled_rows = []
        max_rows = []
        for sample in test:
            q, qhat = sample["q"], sample["est"][BUDGET]
            pooled_rows.append(prediction(q, qhat, pooled_radius, plan_risk, plan_risk, true_risk))
            max_rows.append(prediction(q, qhat, group_max, plan_risk, plan_risk, true_risk))
        pooled_summary = summarize(pooled_rows, include_bounds=True)
        max_summary = summarize(max_rows, include_bounds=True)
        pooled_regrets = np.asarray([x["regret_bits"] for x in pooled_rows], dtype=float)
        max_regrets = np.asarray([x["regret_bits"] for x in max_rows], dtype=float)
        paired = None
        if name == "correct_model":
            clean_pooled_regrets = pooled_regrets.copy()
            clean_group_max_regrets = max_regrets.copy()
        else:
            d_pooled = pooled_regrets - clean_pooled_regrets
            d_max = max_regrets - clean_group_max_regrets
            paired = {
                "pooled_mean_delta_vs_correct_bits": float(d_pooled.mean()),
                "pooled_paired_bootstrap95_ci_bits": g.paired_bootstrap_mean_ci(d_pooled, seed=SEED + len(name), reps=4000),
                "group_max_mean_delta_vs_correct_bits": float(d_max.mean()),
                "group_max_paired_bootstrap95_ci_bits": g.paired_bootstrap_mean_ci(d_max, seed=SEED + len(name) + 77, reps=4000),
            }
        known_results[name] = {
            "risk_matrix_global_rmse_vs_true": float(np.sqrt(np.mean((plan_risk - true_risk) ** 2))),
            "pooled_radius": pooled_radius,
            "group_max_radius": group_max,
            "pooled": pooled_summary,
            "group_max": max_summary,
            "paired_regret_delta_vs_correct_model": paired,
        }

    # A seventh unseen regime is excluded from both pooled and group-maximum calibration.
    ood_alpha = novel_regime_alpha()
    ood = []
    for _ in range(OOD_TEST_N):
        q = rng.dirichlet(ood_alpha)
        ests = g.nested_estimates(rng, q, (60, BUDGET, 240))
        ood.append({"q": q, "est": ests})
    ood_results = {}
    for plan_name, plan_risk in (("correct_model", true_risk), ("risk_misspec_20pct", conditions["risk_misspec_20pct"])):
        pooled_radius = d.quantile_for(calibration, None, BUDGET, plan_risk, TARGET)
        group_radii = {i: d.quantile_for(calibration, i, BUDGET, plan_risk, TARGET) for i in range(len(g.REGIMES))}
        group_max = max(group_radii.values())
        result = {"pooled": [], "group_max": []}
        for sample in ood:
            q, qhat = sample["q"], sample["est"][BUDGET]
            result["pooled"].append(prediction(q, qhat, pooled_radius, plan_risk, plan_risk, true_risk))
            result["group_max"].append(prediction(q, qhat, group_max, plan_risk, plan_risk, true_risk))
        ood_results[plan_name] = {
            "unseen_regime_definition": "Dirichlet prior with alpha 0.5 at latent hypotheses 9 and 15, alpha 0.002 elsewhere; these two hypotheses have maximum Euclidean separation in the true eight-policy risk-vector matrix; excluded from calibration",
            "pooled_radius": pooled_radius, "group_max_radius": group_max,
            "pooled": summarize(result["pooled"]), "group_max": summarize(result["group_max"]),
        }

    result = {
        "schema": "symthaea-decision-sufficient-negative-controls-v1",
        "status": "SYNTHETIC_FALSIFICATION_PILOT_NOT_REAL_SCM_OR_EXTERNAL_SOTA",
        "protocol": {"seed": SEED, "calibration_per_regime": N_CAL_PER_REGIME,
            "test_per_regime": N_TEST_PER_REGIME, "calibration_total": len(calibration),
            "known_test_total": len(test), "unseen_regime_test_n": OOD_TEST_N,
            "budget": BUDGET, "target": TARGET, "regimes_in_calibration": list(g.REGIMES),
            "negative_controls": list(conditions.keys()) + ["unseen_seventh_regime"],
            "all_conditions_use_same_calibration_and_known_test_cases": True},
        "model": {"latent_hypotheses": N_HYP, "policies": int(true_risk.shape[0]),
            "true_risk_min": float(true_risk.min()), "true_risk_max": float(true_risk.max()),
            "true_risk_std": float(true_risk.std()), "objective": "joint cross-entropy regret bits"},
        "known_regime_results": known_results,
        "unseen_regime_results": ood_results,
        "interpretation": [
            "Correct-model risk-space calibration is valid only for the enumerated policy family, exact risk matrix and exchangeable target distribution.",
            "Risk-matrix perturbation and policy-row identity mismatch intentionally violate assumptions; the 2r bound is not expected to hold against true risks there.",
            "The unseen seventh regime is out-of-calibration support. Failure of pooled or max-known-regime coverage demonstrates that known-regime conformal calibration does not guarantee coverage for arbitrary novel regimes.",
            "All outputs are a synthetic pilot, not a formal guarantee, production result or SOTA comparison."],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "runtime_seconds_local": time.perf_counter() - t0,
    }
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    payload = run()
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(raw)
    print(json.dumps({"output": str(a.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_seconds": payload["runtime_seconds_local"],
        "max_certificate_gap_bits": max(v[m]["max_certificate_gap_bits"] for v in payload["known_regime_results"].values() for m in ("pooled", "group_max"))}, indent=2))
