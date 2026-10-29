#!/usr/bin/env python3
"""Exploratory PITB component-size / estimand calibration, independent NumPy harness.

Research-only: this does not read PITB outcomes, replace the production Rust
analyzer, or qualify the benchmark. It stress-tests the proposed inference
contract over synthetic connected-component size profiles.
"""
import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

SEED = 20261029
P_MEAN = 0.35
ICC = 0.10
PAIR_ASSOC = 0.50
REQUESTED_EFFECT_SD = 0.08
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


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def generate_effect(sizes, rng, kind):
    if kind == "independent":
        raw = rng.normal(size=len(sizes))
    elif kind == "size_correlated":
        raw = np.log(np.asarray(sizes, dtype=float))
    else:
        raise ValueError(kind)
    raw = raw - np.average(raw, weights=sizes)
    sd = np.sqrt(np.average(raw * raw, weights=sizes))
    effect = np.zeros(len(sizes)) if sd < 1e-12 else raw * (REQUESTED_EFFECT_SD / sd)
    return effect


def one_study(sizes, rng, boot=BOOT, effect_kind="independent"):
    sizes = np.asarray(sizes, dtype=int)
    component_count = len(sizes)
    target_count = int(sizes.sum())
    concentration = 1 / ICC - 1
    p0 = rng.beta(P_MEAN * concentration, (1 - P_MEAN) * concentration, component_count)
    effect = generate_effect(sizes, rng, effect_kind)

    # Keep potential success rates valid without changing the centered estimand.
    scale = 1.0
    for delta, probability in zip(effect, p0):
        if delta > 0:
            scale = min(scale, (1 - probability) / delta)
        elif delta < 0:
            scale = min(scale, probability / (-delta))
    effect = effect * max(0.0, min(1.0, scale))
    p1 = p0 + effect

    component_deltas = np.zeros(component_count, dtype=float)
    for index, size in enumerate(sizes):
        control_probability = float(p0[index])
        treatment_probability = float(p1[index])
        joint_success = control_probability * treatment_probability + PAIR_ASSOC * (
            min(control_probability, treatment_probability)
            - control_probability * treatment_probability
        )
        u = rng.random(int(size))
        both = u < joint_success
        treatment_only = (u >= joint_success) & (u < treatment_probability)
        control_only = (u >= treatment_probability) & (
            u < treatment_probability + control_probability - joint_success
        )
        control = both | control_only
        treatment = both | treatment_only
        component_deltas[index] = np.sum(treatment.astype(np.int8) - control.astype(np.int8))

    estimate = float(component_deltas.sum() / target_count)
    # Percentile bootstrap over whole components, preserving the target-weighted estimand.
    indices = rng.integers(0, component_count, size=(boot, component_count))
    numerator = component_deltas[indices].sum(axis=1)
    denominator = sizes[indices].sum(axis=1)
    bootstrap_values = numerator / denominator
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    informative = bool(upper - lower > np.finfo(float).eps)
    passed_utility_gate = bool(informative and estimate >= 0.10 and lower > 0)
    rejects_zero = bool(informative and lower > 0)

    # Diagnostic only: this is a different, equal-component estimand.
    equal_component_estimate = float(np.mean(component_deltas / sizes))
    true_target_effect = float(np.average(effect, weights=sizes))
    true_equal_component_effect = float(np.mean(effect))
    realized_effect_sd = float(np.sqrt(np.average(effect * effect, weights=sizes)))
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "informative": informative,
        "passed_utility_gate": passed_utility_gate,
        "rejects_zero": rejects_zero,
        "true_target_effect": true_target_effect,
        "true_equal_component_effect": true_equal_component_effect,
        "equal_component_estimate": equal_component_estimate,
        "realized_effect_sd": realized_effect_sd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=int, default=STUDIES)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("pitb-component-size-stress-screen.json"))
    args = parser.parse_args()
    if args.studies < 100 or args.bootstrap_reps < 199:
        raise SystemExit("screen requires at least 100 studies and 199 bootstrap replicates per study")

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(PROFILES) * 2)
    results = []
    seed_index = 0
    for name, raw_sizes in PROFILES.items():
        sizes = np.asarray(raw_sizes, dtype=int)
        for effect_kind in ("independent", "size_correlated"):
            rng = np.random.default_rng(child_seeds[seed_index])
            seed_index += 1
            rows = [
                one_study(sizes, rng, boot=args.bootstrap_reps, effect_kind=effect_kind)
                for _ in range(args.studies)
            ]
            available = np.array([row["informative"] for row in rows])
            rejected = np.array([row["rejects_zero"] for row in rows])
            gated = np.array([row["passed_utility_gate"] for row in rows])
            covered = np.array([
                row["informative"] and row["lower"] <= row["true_target_effect"] <= row["upper"]
                for row in rows
            ])
            available_count = int(available.sum())
            rejected_count = int(rejected.sum())
            gate_count = int(gated.sum())
            results.append({
                "profile": name,
                "effect_structure": effect_kind,
                "component_count": len(sizes),
                "target_count": int(sizes.sum()),
                "size_min": int(sizes.min()),
                "size_median": float(np.median(sizes)),
                "size_max": int(sizes.max()),
                "size_cv": float(np.std(sizes) / np.mean(sizes)),
                "largest_component_target_share": float(sizes.max() / sizes.sum()),
                "study_replicates": args.studies,
                "bootstrap_replicates_per_study": args.bootstrap_reps,
                "mean_realized_weighted_effect_sd_pp": 100 * float(np.mean([r["realized_effect_sd"] for r in rows])),
                "mean_true_target_weighted_effect_pp": 100 * float(np.mean([r["true_target_effect"] for r in rows])),
                "mean_true_equal_component_effect_pp": 100 * float(np.mean([r["true_equal_component_effect"] for r in rows])),
                "mean_target_weighted_estimate_pp": 100 * float(np.mean([r["estimate"] for r in rows])),
                "target_weighted_bias_pp": 100 * float(np.mean([
                    r["estimate"] - r["true_target_effect"] for r in rows
                ])),
                "equal_component_estimator_bias_vs_own_estimand_pp": 100 * float(np.mean([
                    r["equal_component_estimate"] - r["true_equal_component_effect"] for r in rows
                ])),
                "interval_availability_rate": float(available.mean()),
                "conditional_coverage_given_informative": float(covered.sum() / available_count) if available_count else None,
                "unconditional_lower_bound_rejection_rate": float(rejected.mean()),
                "lower_bound_rejection_mc_wilson95": wilson(rejected_count, args.studies),
                "utility_gate_pass_rate_under_weighted_weak_null": float(gated.mean()),
                "utility_gate_mc_wilson95": wilson(gate_count, args.studies),
                "mean_lower_ci_endpoint_pp": 100 * float(np.mean([r["lower"] for r in rows])),
                "mean_upper_ci_endpoint_pp": 100 * float(np.mean([r["upper"] for r in rows])),
            })

    report = {
        "schema_version": "pitb.component-size-stress-screen.v0",
        "status": "exploratory_synthetic_calibration_not_qualification",
        "seed": args.seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "assumptions": {
            "baseline_mean": P_MEAN,
            "component_icc": ICC,
            "paired_outcome_association_fraction": PAIR_ASSOC,
            "requested_component_effect_sd": REQUESTED_EFFECT_SD,
            "studies_per_cell": args.studies,
            "bootstrap_replicates_per_study": args.bootstrap_reps,
            "interval": "two-sided 95% percentile bootstrap; whole components resampled; target-weighted estimate",
        },
        "scenarios": results,
        "limitations": [
            "Synthetic size profiles are stress cases, not the actual reviewed PITB component graph.",
            "Monte Carlo Wilson intervals describe simulation noise only, not uncertainty about model realism.",
            "The percentile cluster bootstrap is not thereby endorsed; alternative methods require independent statistical review.",
            "Target-weighted fixed-cohort and equal-component effects are different estimands; select one before viewing results.",
            "This research-only NumPy implementation is not the production Rust scorer and does not qualify PITB.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.out),
        "scenario_count": len(results),
        "schema_version": report["schema_version"],
        "status": report["status"],
    }, indent=2))
    for row in results:
        print(
            f'{row["profile"]:18} {row["effect_structure"]:15} '
            f'C={row["component_count"]:2} N={row["target_count"]:3} '
            f'CV={row["size_cv"]:.2f} maxshare={row["largest_component_target_share"]:.3f} '
            f'coverage={row["conditional_coverage_given_informative"]:.3f} '
            f'reject={row["unconditional_lower_bound_rejection_rate"]:.3f} '
            f'utility_pass={row["utility_gate_pass_rate_under_weighted_weak_null"]:.3f} '
            f'equal_weighted_true={row["mean_true_equal_component_effect_pp"]:.2f}pp'
        )


if __name__ == "__main__":
    main()
