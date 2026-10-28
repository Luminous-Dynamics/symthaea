#!/usr/bin/env python3
"""Compare fixed-look risk certificates with time-uniform sequential audit bounds.

Synthetic finite-cell benchmark. A fixed-time empirical-Bernstein bound is valid
at a prespecified sample size; repeatedly checking it and stopping opportunistically
does not preserve that single-look guarantee. The time-uniform method here uses a
summable alpha-spending schedule plus a fixed-time empirical-Bernstein inequality.
This is intentionally simple and conservative, not claimed as SOTA.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np

SCHEMA = "symthaea-decision-risk-sequential-audit-v1"
SEED = 20261029
DELTA = 0.10
LOSS_MIN = 0.0
LOSS_MAX = 1.0
LOSS_RANGE = LOSS_MAX - LOSS_MIN
RADIUS_BITS = 0.06664248830771191
REGRET_BUDGET_BITS = 0.20
EPSILON_ALLOWANCE_BITS = REGRET_BUDGET_BITS / 2.0 - RADIUS_BITS
SAMPLE_SIZES = (4096, 8192, 16384, 32768, 65536)
N_REPEATS = 1000
TOL = 1e-12


def empirical_bernstein_radius(variance: np.ndarray, n: int, log_term: float,
                                loss_range: float = LOSS_RANGE) -> np.ndarray:
    if n < 2 or np.any(~np.isfinite(variance)) or np.any(variance < 0) or log_term <= 0 or loss_range <= 0:
        raise ValueError("invalid empirical-Bernstein inputs")
    return np.sqrt(2.0 * variance * log_term / n) + 7.0 * loss_range * log_term / (3.0 * (n - 1))


def fixed_time_radius(variance: np.ndarray, n: int, n_cells: int, delta: float = DELTA) -> np.ndarray:
    if not (0 < delta < 1) or n_cells < 1:
        raise ValueError("invalid familywise delta or cell count")
    return empirical_bernstein_radius(variance, n, math.log(4.0 * n_cells / delta))


def finite_look_radius(variance: np.ndarray, n: int, n_cells: int, n_looks: int, delta: float = DELTA) -> np.ndarray:
    """Familywise empirical-Bernstein radius for a fixed, predeclared look schedule."""
    if n_looks < 1:
        raise ValueError("number of predeclared looks must be positive")
    return fixed_time_radius(variance, n, n_cells, delta / n_looks)


def anytime_radius(variance: np.ndarray, n: int, n_cells: int, delta: float = DELTA) -> np.ndarray:
    """Time-uniform cellwise radius by union bound over all n >= 1.

    Allocate delta_n = delta * 6/(pi^2*n^2), whose sum across n is delta.
    At each n, allocate delta_n familywise across the n_cells two-sided bounds.
    This is an anytime guarantee only under the stated independent bounded audit
    sampling assumptions and fixed predeclared cell family.
    """
    if not (0 < delta < 1) or n_cells < 1:
        raise ValueError("invalid familywise delta or cell count")
    delta_n = delta * 6.0 / (math.pi ** 2 * n ** 2)
    log_term = math.log(4.0 * n_cells / delta_n)
    return empirical_bernstein_radius(variance, n, log_term)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    den = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(*, repeats: int = N_REPEATS, sample_sizes: tuple[int, ...] = SAMPLE_SIZES) -> dict:
    if repeats < 100:
        raise ValueError("use at least 100 repeats for a useful sequential audit diagnostic")
    sizes = tuple(sorted(set(int(n) for n in sample_sizes)))
    if not sizes or sizes[0] < 2:
        raise ValueError("sample sizes must be >= 2")
    if EPSILON_ALLOWANCE_BITS <= 0:
        raise ValueError("regret budget leaves no error allowance")

    # Fixed 4-cell family; Bernoulli audit losses attain high variance while
    # remaining in the predeclared [0,1] range. The planning matrix differs from
    # the true mean matrix only in one candidate cell.
    truth = np.array([[0.50, 0.52], [0.48, 0.55]], dtype=float)
    eps_near = EPSILON_ALLOWANCE_BITS + 1e-5
    models = {
        "correct_model": truth.copy(),
        "just_over_regret_budget": truth.copy(),
        "clearly_over_regret_budget": truth.copy(),
    }
    models["just_over_regret_budget"][0, 0] += eps_near
    models["clearly_over_regret_budget"][0, 0] += EPSILON_ALLOWANCE_BITS + 0.01
    true_errors = {name: float(np.max(np.abs(model-truth))) for name, model in models.items()}
    k = truth.size
    rng = np.random.default_rng(SEED)
    # Counts for each method/model/endpoint; audit stream is nested within repeat.
    stats = {name: {method: {"false_admissions": 0, "any_admission": 0,
                              "admitted_by_look": {str(n): 0 for n in sizes},
                              "sum_stopping_n": 0, "stopped": 0,
                              "final_upper_bound_sum": 0.0}
                    for method in ("fixed_time_repeated_peek", "finite_look_alpha_spending", "anytime_alpha_spending")}
             for name in models}

    for _ in range(repeats):
        # Generate nested Bernoulli audit streams (all outcomes in [0,1]).
        draws = rng.binomial(1, truth[None, :, :], size=(sizes[-1], *truth.shape)).astype(float)
        csum = np.cumsum(draws, axis=0)
        csq = np.cumsum(draws * draws, axis=0)
        for name, model in models.items():
            for method in ("fixed_time_repeated_peek", "finite_look_alpha_spending", "anytime_alpha_spending"):
                rec = stats[name][method]
                stopped = False
                last_upper = float("inf")
                for n in sizes:
                    total = csum[n-1]
                    sumsq = csq[n-1]
                    means = total / n
                    variances = np.maximum((sumsq - total*total/n) / (n-1), 0.0)
                    if method == "fixed_time_repeated_peek":
                        radii = fixed_time_radius(variances, n, k, DELTA)
                    elif method == "finite_look_alpha_spending":
                        radii = finite_look_radius(variances, n, k, len(sizes), DELTA)
                    else:
                        radii = anytime_radius(variances, n, k, DELTA)
                    upper = float(np.max(np.abs(model - means) + radii))
                    last_upper = upper
                    admitted = upper <= EPSILON_ALLOWANCE_BITS + TOL
                    if admitted:
                        rec["admitted_by_look"][str(n)] += 1
                        if true_errors[name] > EPSILON_ALLOWANCE_BITS + TOL:
                            rec["false_admissions"] += 1
                        rec["any_admission"] += 1
                        rec["sum_stopping_n"] += n
                        rec["stopped"] += 1
                        stopped = True
                        break
                if not stopped:
                    rec["sum_stopping_n"] += sizes[-1]
                    rec["final_upper_bound_sum"] += last_upper

    results = {}
    for name in models:
        results[name] = {}
        for method, rec in stats[name].items():
            denom = repeats
            results[name][method] = {
                "true_max_cell_error_bits_scoring_only": true_errors[name],
                "false_admissions": rec["false_admissions"],
                "false_admission_rate": rec["false_admissions"] / denom,
                "false_admission_rate_wilson95": wilson_interval(rec["false_admissions"], denom),
                "any_admission_rate": rec["any_admission"] / denom,
                "admitted_by_look_rate": {n: count / denom for n, count in rec["admitted_by_look"].items()},
                "mean_audit_observations_per_cell_until_stop_or_horizon": rec["sum_stopping_n"] / denom,
                "mean_final_upper_bound_when_not_admitted_bits": (rec["final_upper_bound_sum"] / (denom - rec["stopped"])) if denom > rec["stopped"] else None,
                "stopped_before_horizon_count": rec["stopped"],
            }
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_SEQUENTIAL_AUDIT_BENCHMARK_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {
            "seed": SEED, "repeats": repeats, "looks": list(sizes), "cells": k,
            "loss_range": [LOSS_MIN, LOSS_MAX], "cell_outcomes": "independent Bernoulli draws with per-cell means given by fixed true risk matrix",
            "familywise_delta_per_fixed_look": DELTA,
            "time_uniform_alpha_spending": "delta_n = delta * 6/(pi^2*n^2), sum_n delta_n = delta; cellwise two-sided empirical-Bernstein bound",
            "true_max_cell_error_bits": true_errors,
            "regret_budget_bits": REGRET_BUDGET_BITS,
            "decision_radius_bits": RADIUS_BITS,
            "maximum_admissible_epsilon_bits": EPSILON_ALLOWANCE_BITS,
        },
        "methods": {
            "fixed_time_repeated_peek": "Uses the single-look empirical-Bernstein bound with delta=0.10 at each checkpoint; it is not claimed valid uniformly over opportunistic stopping.",
            "finite_look_alpha_spending": "Uses delta/L per each of L predeclared looks and familywise correction across cells; valid for the stated fixed look schedule under bounded independent sampling.",
            "anytime_alpha_spending": "Uses summable alpha spending across every integer sample size and the predeclared finite family of cells; valid under bounded independent sampling and fixed model/support assumptions.",
        },
        "results": results,
        "limitations": [
            "fixed finite four-cell Bernoulli toy; not a real-world distribution or industry benchmark",
            "true risk means used only to score the false-admission event",
            "both alpha-spending methods use union bounds, not an optimized betting confidence sequence",
            "the finite-look method is valid only for the exact predeclared look schedule; adding unscheduled peeks requires a new error allocation",
            "does not protect against shared audit bias, hidden dependence or incomplete support; those require the separate preflight/receipt gate",
            "no published baseline implementation run; no SOTA claim",
            "not integrated into native Rust decision or dispatch path",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = run(repeats=args.repeats)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "runtime_seconds": round(time.perf_counter()-started, 3)}, indent=2))

if __name__ == "__main__":
    main()
