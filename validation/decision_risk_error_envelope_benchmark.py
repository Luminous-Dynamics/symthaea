#!/usr/bin/env python3
"""Finite-sample audit of the risk-matrix error envelope used by a regret gate.

Synthetic finite-game research benchmark. It replaces oracle-known epsilon with
independent bounded audit samples and simultaneous confidence envelopes. It does
not establish that real latent-world risk cells are observable or exchangeable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import decision_sufficient_negative_controls as nc
from prior_set_coverage_double_oracle_benchmark import build_risk_tensor

SCHEMA = "symthaea-decision-risk-error-envelope-v1"
SEED = 20261028
DELTA = 0.05
LOSS_MIN = 0.0
LOSS_MAX = 4.0
LOSS_RANGE = LOSS_MAX - LOSS_MIN
AUDIT_NOISE_HALF_WIDTH = 0.05
N_REPEATS = 200
SAMPLE_SIZES = (32, 128, 512, 2048, 4096, 8192)
NOMINAL_RADIUS_BITS = 0.06664248830771191  # Exact pooled radius from certified-fallback pilot.
REGRET_BUDGET_BITS = 0.20
TOL = 1e-12


def _validate_samples(samples: np.ndarray, *, loss_min: float, loss_max: float) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("audit samples must have shape (n>=2, policies>=1, worlds>=1)")
    if np.any(~np.isfinite(x)):
        raise ValueError("audit samples must be complete and finite; missing cells fail closed")
    if np.any(x < loss_min) or np.any(x > loss_max):
        raise ValueError("audit samples fall outside the declared loss range")
    return x


def empirical_bernstein_cell_radii(samples: np.ndarray, *, delta: float = DELTA,
                                   loss_min: float = LOSS_MIN,
                                   loss_max: float = LOSS_MAX) -> np.ndarray:
    """Two-sided, familywise cellwise empirical-Bernstein radii.

    Uses the Maurer-Pontil bounded empirical-Bernstein form. The per-tail error
    allocation is delta/(2*K), with K=policy_count*world_count; the log term is
    therefore log(4*K/delta). Requires independent bounded audit observations
    within each cell and the fixed finite cell family specified before sampling.
    """
    x = _validate_samples(samples, loss_min=loss_min, loss_max=loss_max)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie strictly between 0 and 1")
    n, n_policy, n_world = x.shape
    k = n_policy * n_world
    span = loss_max - loss_min
    log_term = math.log(4.0 * k / delta)
    variance = np.var(x, axis=0, ddof=1)
    radii = np.sqrt(2.0 * variance * log_term / n) + (7.0 * span * log_term) / (3.0 * (n - 1))
    if np.any(~np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("computed non-finite empirical-Bernstein radius")
    return radii


def hoeffding_cell_radius(n: int, n_cells: int, *, delta: float = DELTA,
                          loss_min: float = LOSS_MIN, loss_max: float = LOSS_MAX) -> float:
    if n < 1 or n_cells < 1 or not (0.0 < delta < 1.0) or loss_max <= loss_min:
        raise ValueError("invalid Hoeffding-bound parameters")
    # Two-sided Hoeffding + a union bound over the finite cell family.
    return (loss_max - loss_min) * math.sqrt(math.log(2.0 * n_cells / delta) / (2.0 * n))


def error_envelopes(planning_matrix: np.ndarray, audit_samples: np.ndarray,
                    *, delta: float = DELTA) -> dict:
    x = _validate_samples(audit_samples, loss_min=LOSS_MIN, loss_max=LOSS_MAX)
    model = np.asarray(planning_matrix, dtype=float)
    if model.shape != x.shape[1:] or np.any(~np.isfinite(model)):
        raise ValueError("planning matrix must be finite and exactly match audit cell shape")
    n_cells = model.size
    means = np.mean(x, axis=0)
    plugin = float(np.max(np.abs(model - means)))
    eb_radii = empirical_bernstein_cell_radii(x, delta=delta)
    empirical_bernstein = float(np.max(np.abs(model - means) + eb_radii))
    hoeffding = plugin + hoeffding_cell_radius(x.shape[0], n_cells, delta=delta)
    return {
        "n_per_policy_world_cell": int(x.shape[0]),
        "plugin_residual_max_bits_not_a_confidence_bound": plugin,
        "empirical_bernstein_error_upper_bound_bits": empirical_bernstein,
        "hoeffding_error_upper_bound_bits": hoeffding,
        "mean_empirical_bernstein_cell_radius_bits": float(np.mean(eb_radii)),
        "max_empirical_bernstein_cell_radius_bits": float(np.max(eb_radii)),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial count")
    p = successes / trials
    den = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(*, repeats: int = N_REPEATS, sample_sizes: tuple[int, ...] = SAMPLE_SIZES) -> dict:
    if repeats < 20:
        raise ValueError("use at least 20 Monte Carlo repeats for a useful diagnostic")
    sizes = tuple(sorted(set(int(n) for n in sample_sizes)))
    if not sizes or sizes[0] < 2:
        raise ValueError("sample sizes must be at least 2")
    n_max = sizes[-1]
    tensor, _, _ = build_risk_tensor()
    truth = np.asarray(tensor["joint"]["risk_by_policy_and_world"], dtype=float)
    if truth.min() < LOSS_MIN or truth.max() > LOSS_MAX:
        raise ValueError("simulator risk tensor escaped the predeclared [0,4] loss range")
    epsilon_allowance = REGRET_BUDGET_BITS / 2.0 - NOMINAL_RADIUS_BITS
    near_raw = nc.perturbed_risk(truth, 0.05, SEED + 35)
    near_delta = near_raw - truth
    near_target_epsilon = epsilon_allowance + 0.0015
    near_budget_model = truth + near_delta * (near_target_epsilon / float(np.max(np.abs(near_delta))))
    models = {
        "correct_model": truth,
        "risk_misspec_1pct": nc.perturbed_risk(truth, 0.01, SEED + 1),
        "risk_misspec_near_budget": near_budget_model,
        "risk_misspec_5pct": nc.perturbed_risk(truth, 0.05, SEED + 5),
        "risk_misspec_20pct": nc.perturbed_risk(truth, 0.20, SEED + 20),
    }
    true_eps = {name: float(np.max(np.abs(model - truth))) for name, model in models.items()}
    n_cells = truth.size
    if epsilon_allowance <= 0:
        raise ValueError("regret budget leaves no nonnegative model-error allowance")

    # Keep counts/sums only. Every method sees the same held-out audit sample
    # within a repeat, giving paired comparisons across error-envelope methods.
    book = {
        model_name: {
            str(n): {
                method: {"covered": 0, "plugin_admitted": 0, "upper_bound_admitted": 0,
                         "plugin_unsafe_admitted": 0, "upper_unsafe_admitted": 0,
                         "epsilon_sum": 0.0, "bound_sum": 0.0}
                for method in ("plugin", "empirical_bernstein", "hoeffding")
            }
            for n in sizes
        }
        for model_name in models
    }
    rng = np.random.default_rng(SEED)
    for _rep in range(repeats):
        # Independent, bounded, unbiased audit residuals. The uniform-noise
        # construction is deliberately low-variance; the range supplied to the
        # proof remains the predeclared [0,4], not the observed noise range.
        audit = truth[None, :, :] + rng.uniform(
            -AUDIT_NOISE_HALF_WIDTH, AUDIT_NOISE_HALF_WIDTH,
            size=(n_max, truth.shape[0], truth.shape[1]),
        )
        if audit.min() < LOSS_MIN or audit.max() > LOSS_MAX:
            raise AssertionError("audit sample escaped declared range")
        csum = np.cumsum(audit, axis=0)
        csq = np.cumsum(audit * audit, axis=0)
        for n in sizes:
            sum_x = csum[n-1]
            sum_x2 = csq[n-1]
            means = sum_x / n
            variance = np.maximum((sum_x2 - (sum_x * sum_x) / n) / (n-1), 0.0)
            log_h = math.log(2.0 * n_cells / DELTA)
            hoeffding_radius = LOSS_RANGE * math.sqrt(log_h / (2.0*n))
            log_eb = math.log(4.0 * n_cells / DELTA)
            eb_radii = np.sqrt(2.0 * variance * log_eb / n) + 7.0 * LOSS_RANGE * log_eb / (3.0*(n-1))
            for model_name, model in models.items():
                residual = np.abs(model - means)
                plugin_epsilon = float(np.max(residual))
                envelopes = {
                    "plugin": plugin_epsilon,
                    "empirical_bernstein": float(np.max(residual + eb_radii)),
                    "hoeffding": plugin_epsilon + hoeffding_radius,
                }
                eps_true = true_eps[model_name]
                for method, eps_upper in envelopes.items():
                    rec = book[model_name][str(n)][method]
                    is_covered = eps_true <= eps_upper + TOL
                    cert_bound = 2.0 * (NOMINAL_RADIUS_BITS + eps_upper)
                    plugin_admitted = 2.0 * (NOMINAL_RADIUS_BITS + plugin_epsilon) <= REGRET_BUDGET_BITS
                    upper_admitted = cert_bound <= REGRET_BUDGET_BITS
                    rec["covered"] += int(is_covered)
                    rec["plugin_admitted"] += int(plugin_admitted)
                    rec["upper_bound_admitted"] += int(upper_admitted)
                    rec["plugin_unsafe_admitted"] += int(plugin_admitted and eps_true > epsilon_allowance + TOL)
                    rec["upper_unsafe_admitted"] += int(upper_admitted and eps_true > epsilon_allowance + TOL)
                    rec["epsilon_sum"] += eps_upper
                    rec["bound_sum"] += cert_bound

    results = {}
    for model_name in models:
        results[model_name] = {}
        for n in sizes:
            results[model_name][str(n)] = {}
            for method in ("plugin", "empirical_bernstein", "hoeffding"):
                raw = book[model_name][str(n)][method]
                count = raw["covered"]
                results[model_name][str(n)][method] = {
                    "epsilon_true_bits_oracle_for_scoring_only": true_eps[model_name],
                    "envelope_coverage_rate": count / repeats,
                    "envelope_coverage_wilson95": wilson_interval(count, repeats),
                    "mean_reported_epsilon_upper_bits": raw["epsilon_sum"] / repeats,
                    "mean_regret_certificate_bound_bits": raw["bound_sum"] / repeats,
                    "admission_rate_using_plugin_point_estimate": raw["plugin_admitted"] / repeats,
                    "unsafe_admission_rate_using_plugin_point_estimate": raw["plugin_unsafe_admitted"] / repeats,
                    "admission_rate_using_this_method": raw["upper_bound_admitted"] / repeats,
                    "unsafe_admission_rate_using_this_method": raw["upper_unsafe_admitted"] / repeats,
                    "regret_budget_bits": REGRET_BUDGET_BITS,
                    "nominal_radius_bits": NOMINAL_RADIUS_BITS,
                    "epsilon_allowance_for_admission_bits": epsilon_allowance,
                    "upper_bound_coverage_count": count,
                    "monte_carlo_repeats": repeats,
                }
    # Worst-case sample count predicted by plain Hoeffding for the entire
    # remaining epsilon budget; rounded up and contrasted with the EB pilot.
    log_h = math.log(2.0 * n_cells / DELTA)
    hoeffding_n_required = math.ceil((LOSS_RANGE**2 * log_h) / (2.0 * epsilon_allowance**2))
    payload = {
        "schema": SCHEMA,
        "status": "SYNTHETIC_BOUNDED_AUDIT_STUDY_NOT_DEPLOYMENT_CERTIFICATION",
        "protocol": {
            "seed": SEED, "monte_carlo_repeats": repeats, "sample_sizes_per_policy_world_cell": list(sizes),
            "policies": int(truth.shape[0]), "latent_worlds": int(truth.shape[1]), "cells": int(n_cells),
            "familywise_delta": DELTA, "target_confidence": 1.0-DELTA,
            "declared_audit_loss_range_bits": [LOSS_MIN, LOSS_MAX],
            "audit_observation_model": f"true cell risk + iid Uniform(-{AUDIT_NOISE_HALF_WIDTH},{AUDIT_NOISE_HALF_WIDTH}); samples are unbiased and within [0,4]",
            "models": "correct risk matrix, 1%, 5%, 20%-scale perturbations, and an explicit near-budget case rescaled to 0.0015 bits beyond the maximum admissible epsilon; row-identity mismatch is excluded and must be rejected by policy-risk binding before error-envelope estimation",
            "nominal_radius_bits": NOMINAL_RADIUS_BITS, "regret_budget_bits": REGRET_BUDGET_BITS,
            "max_epsilon_for_admission_bits": epsilon_allowance,
            "simultaneous_family": "fixed finite policy × latent-world cells; two-sided cell confidence intervals and union bound",
        },
        "theory": {
            "decision_bound": "true regret <= 2(r + epsilon), conditional on decision-risk coverage and valid epsilon",
            "empirical_bernstein": "epsilon_UCB = max_cell(|planning_cell - audit_mean_cell| + empirical_Bernstein_radius_cell); bounded iid cell audits and predeclared finite family are required",
            "hoeffding_sample_count_for_epsilon_budget": hoeffding_n_required,
            "sample_count_warning": "These per-cell counts are not total episodes. Real latent worlds may not be directly observable or independently sampleable; if support is not audited, epsilon is unverified and the gate must not admit.",
        },
        "oracle_truth_used_only_for_scoring": {"risk_matrix_sha256_f64be": hashlib.sha256(np.asarray(truth, dtype=">f8").tobytes()).hexdigest(), "true_epsilon_by_model_bits": true_eps},
        "results": results,
        "limitations": [
            "synthetic finite-game 8-policy × 16-world only",
            "audit data are simulated iid low-noise bounded measurements; no evidence that real SCM latent-world cells are observable or exchangeable",
            "empirical Bernstein confidence is conditional on independent bounded audit measurements, a predeclared finite cell family, and the declared [0,4] range",
            "the nominal conformal radius is imported from the earlier synthetic calibration pilot, not jointly recalibrated here",
            "the simulator truth is used only to score whether each envelope covered; it is not fed into the estimators",
            "policy-row identity mismatch is an integrity failure and must be rejected by the manifest gate rather than managed as ordinary statistical noise",
            "no published baseline implementations or real-world datasets were run; no SOTA claim",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(repeats=args.repeats)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "runtime_seconds": round(time.perf_counter()-started, 3)}, indent=2))


if __name__ == "__main__":
    main()
