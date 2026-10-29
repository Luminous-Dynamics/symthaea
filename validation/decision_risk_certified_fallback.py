#!/usr/bin/env python3
"""Decision-regret admission gate using calibrated uncertainty + bounded model error.

Synthetic finite-game diagnostic. The error envelope is oracle-known in this
experiment; a deployed system would require a separately justified upper bound.
"""
from __future__ import annotations
import hashlib, json, time, platform
from pathlib import Path
import numpy as np
import groupwise_robust_coverage_benchmark as g
import decision_sufficient_ambiguity_benchmark as d
import decision_sufficient_negative_controls as nc
from prior_set_coverage_double_oracle_benchmark import build_risk_tensor

SCHEMA = "symthaea-decision-risk-certified-fallback-v1"
SEED = 20261032
BUDGET = 120
TARGET = 0.90
N_CAL = 100
N_TEST = 20
REGRET_BUDGET_BITS = 0.20
TOL = 1e-7


def certified_regret_bound(radius: float, matrix_error_upper_bound: float) -> float:
    if not np.isfinite(radius) or not np.isfinite(matrix_error_upper_bound) or radius < 0 or matrix_error_upper_bound < 0:
        raise ValueError("radius and matrix error upper bound must be finite and nonnegative")
    return 2.0 * (radius + matrix_error_upper_bound)


def summarize_rows(rows, bound, reference_weights):
    covered = [row for row in rows if row["covered_by_planning_geometry"]]
    violations = [row for row in covered if row["planner_regret_bits"] > bound + TOL]
    gated_regret = np.asarray([row["gated_regret_bits"] for row in rows], dtype=float)
    planner_regret = np.asarray([row["planner_regret_bits"] for row in rows], dtype=float)
    fallback_rate = float(np.mean([row["used_fallback"] for row in rows]))
    return {
        "n": len(rows),
        "coverage_by_planning_geometry": float(np.mean([row["covered_by_planning_geometry"] for row in rows])),
        "covered_n": len(covered),
        "conditional_regret_bound_violations": len(violations),
        "max_covered_planner_regret_over_bound": float(max((row["planner_regret_bits"] / bound for row in covered), default=0.0)) if bound > 0 else 0.0,
        "planner_mean_regret_bits": float(planner_regret.mean()),
        "planner_p95_regret_bits": float(np.quantile(planner_regret, .95)),
        "gated_mean_regret_bits": float(gated_regret.mean()),
        "gated_p95_regret_bits": float(np.quantile(gated_regret, .95)),
        "fallback_rate": fallback_rate,
        "fallback_reference_policy_weights_oracle_computed_for_benchmark_only": [float(x) for x in reference_weights],
        "mean_gated_regret_delta_vs_always_planner_bits": float(gated_regret.mean() - planner_regret.mean()),
    }


def run():
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    alphas = g.meta_alphas()
    calibration = g.balanced_samples(rng, alphas, N_CAL, (60, BUDGET, 240))
    tests = g.balanced_samples(rng, alphas, N_TEST, (60, BUDGET, 240))
    tensor, _, _ = build_risk_tensor()
    true_risk = np.asarray(tensor["joint"]["risk_by_policy_and_world"], dtype=float)
    planning_conditions = {
        "correct_model": true_risk,
        "risk_misspec_5pct": nc.perturbed_risk(true_risk, 0.05, SEED + 5),
        "risk_misspec_20pct": nc.perturbed_risk(true_risk, 0.20, SEED + 20),
        "policy_row_identity_mismatch": true_risk[[3, 1, 7, 0, 6, 2, 5, 4], :],
    }
    # Full-simplex minimax mixture computed with simulator truth. This is an
    # oracle reference for the experiment, not an implementable baseline; a real
    # system needs an independently qualified reference controller.
    fallback_region = d.make_region(np.ones(true_risk.shape[1]) / true_risk.shape[1], 1.0, true_risk, unbounded=True)
    reference_solution = g.solve_robust_game(true_risk, fallback_region)
    reference_weights = reference_solution["mu"]
    results = {}
    for condition, planning_risk in planning_conditions.items():
        model_error = float(np.max(np.abs(planning_risk - true_risk)))
        pooled_radius = d.quantile_for(calibration, None, BUDGET, planning_risk, TARGET)
        regime_radii = {regime: d.quantile_for(calibration, regime, BUDGET, planning_risk, TARGET) for regime in range(len(g.REGIMES))}
        radii = {"pooled": pooled_radius, "group_max": max(regime_radii.values())}
        per_method = {}
        for method, radius in radii.items():
            bound = certified_regret_bound(radius, model_error)
            admitted = bound <= REGRET_BUDGET_BITS
            rows = []
            for sample in tests:
                q = sample["q"]
                qhat = sample["est"][BUDGET]
                region = d.make_region(qhat, radius, planning_risk)
                solution = g.solve_robust_game(planning_risk, region)
                true_policy_losses = true_risk @ q
                oracle = float(np.min(true_policy_losses))
                planner_loss = float(solution["mu"] @ true_policy_losses)
                planner_regret = max(0.0, planner_loss - oracle)
                # The calibrated event is in the estimated-model risk space.
                # The epsilon envelope upgrades the conditional regret bound to
                # the true risk matrix, assuming action identities remain bound.
                model_risk_error = float(np.max(np.abs(planning_risk @ q - planning_risk @ qhat)))
                covered = model_risk_error <= radius + TOL
                fallback_loss = float(reference_weights @ true_policy_losses)
                fallback_regret = max(0.0, fallback_loss - oracle)
                used_fallback = not admitted
                gated_regret = fallback_regret if used_fallback else planner_regret
                rows.append({
                    "regime": g.REGIMES[sample["regime"]],
                    "covered_by_planning_geometry": covered,
                    "planner_regret_bits": planner_regret,
                    "fallback_regret_bits": fallback_regret,
                    "gated_regret_bits": gated_regret,
                    "used_fallback": used_fallback,
                    "certificate_gap_bits": float(solution["gap"]),
                })
            sm = summarize_rows(rows, bound, reference_weights)
            sm.update({
                "radius": float(radius),
                "matrix_error_upper_bound": model_error,
                "certified_regret_bound_bits": bound,
                "declared_regret_budget_bits": REGRET_BUDGET_BITS,
                "admitted": admitted,
                "max_lp_certificate_gap_bits": float(max(row["certificate_gap_bits"] for row in rows)),
            })
            per_method[method] = sm
        results[condition] = per_method
    out = {
        "schema": SCHEMA,
        "status": "SYNTHETIC_FINITE_GAME_ORACLE_ERROR_ENVELOPE_NOT_REAL_DEPLOYMENT",
        "protocol": {
            "seed": SEED, "calibration_per_regime": N_CAL, "test_per_regime": N_TEST,
            "calibration_total": len(calibration), "test_total": len(tests), "budget": BUDGET,
            "nominal_coverage_target": TARGET, "regret_budget_bits": REGRET_BUDGET_BITS,
            "regimes": list(g.REGIMES), "conditions_use_same_episodes": True,
            "risk_error_envelope": "exact maximum absolute cell error computed against simulator truth; oracle diagnostic only",
            "fallback_reference": "oracle-computed full-simplex minimax policy mixture using true simulator risk matrix; proxy only, not implementable in real deployment without separate qualification",
        },
        "model": {"latent_hypotheses": int(true_risk.shape[1]), "policies": int(true_risk.shape[0]), "objective": "joint cross-entropy regret bits"},
        "theorem_used": "If model-risk coverage holds within radius r and max absolute per-cell loss-matrix error is bounded by epsilon, then true regret is at most 2(r+epsilon) for the fixed finite policy family; this is conditional on the coverage event and valid epsilon envelope.",
        "limitations": [
            "epsilon is oracle-known from simulator truth, not estimated by an implemented production uncertainty procedure",
            "reference fallback mixture is computed from simulator truth and is not an executable real-world baseline",
            "bound requires a fixed policy identity mapping; a trusted manifest should reject changed mapping rather than merely inflate epsilon",
            "coverage applies to the estimated-model risk coordinates under exchangeability, not arbitrary unseen regimes",
            "synthetic finite game; no external benchmark or SOTA claim",
        ],
        "results": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "runtime_seconds_local": time.perf_counter() - started,
    }
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run()
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "runtime_seconds": payload["runtime_seconds_local"]}, indent=2))
