#!/usr/bin/env python3
"""Design-only power screen for PITB simultaneous Wilson support-landscape gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


def wilson_lower(successes: int, trials: int, z: float) -> float:
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius)


def binomial_upper_tail(trials: int, threshold: int, probability: float) -> float:
    """P[X >= threshold] for X~Binomial(trials, probability), via log-sum-exp."""
    if threshold <= 0:
        return 1.0
    if threshold > trials:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    logs = []
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_fact_n = math.lgamma(trials + 1)
    for successes in range(threshold, trials + 1):
        log_choose = log_fact_n - math.lgamma(successes + 1) - math.lgamma(trials - successes + 1)
        logs.append(log_choose + successes * log_p + (trials - successes) * log_q)
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs))


def critical_successes(trials: int, reliability_floor: float, z: float) -> int | None:
    for successes in range(trials + 1):
        if wilson_lower(successes, trials, z) >= reliability_floor:
            return successes
    return None


def plan_for_context_count(
    context_count: int,
    reliability_floor: float = 0.80,
    true_success_probability: float = 0.90,
    familywise_alpha: float = 0.05,
    target_power: float = 0.80,
    max_replicates: int = 2000,
) -> dict:
    if context_count < 1:
        raise ValueError("context_count must be positive")
    z = NormalDist().inv_cdf(1.0 - familywise_alpha / context_count)
    for trials in range(2, max_replicates + 1):
        required = critical_successes(trials, reliability_floor, z)
        if required is None:
            continue
        achieved_power = binomial_upper_tail(trials, required, true_success_probability)
        if achieved_power >= target_power:
            return {
                "well_formed_context_count": context_count,
                "familywise_alpha": familywise_alpha,
                "simultaneous_confidence_level": 1.0 - familywise_alpha,
                "reliability_floor": reliability_floor,
                "design_true_success_probability": true_success_probability,
                "target_power": target_power,
                "minimum_replicates_per_context": trials,
                "minimum_successes_to_qualify": required,
                "observed_rate_at_threshold": required / trials,
                "one_sided_bonferroni_z": z,
                "achieved_binomial_power_at_design_alternative": achieved_power,
                "null_boundary_probability_at_rejection_threshold": binomial_upper_tail(trials, required, reliability_floor),
                "method": "one-sided Bonferroni-adjusted Wilson lower bound; binomial alternative power screen",
                "interpretation": "Design screen only; verify with independent Monte Carlo and exact confidence procedures before confirmatory use.",
            }
    return {
        "well_formed_context_count": context_count,
        "status": "not_found_within_max_replicates",
        "max_replicates": max_replicates,
        "interpretation": "No design meeting the requested power was found within the configured search cap.",
    }


def self_test() -> None:
    expected = {
        4: (126, 111),
        8: (152, 134),
        16: (172, 152),
        64: (225, 199),
        256: (272, 241),
    }
    for contexts, pair in expected.items():
        result = plan_for_context_count(contexts)
        actual = (result["minimum_replicates_per_context"], result["minimum_successes_to_qualify"])
        assert actual == pair, (contexts, actual, pair)
    assert binomial_upper_tail(10, 0, 0.5) == 1.0
    assert binomial_upper_tail(10, 11, 0.5) == 0.0
    print("PITB_SUPPORT_LANDSCAPE_POWER_SELF_TEST_PASS: five simultaneous-context design anchors and binomial tail boundaries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-counts", default="4,8,16,64,256")
    parser.add_argument("--reliability-floor", type=float, default=0.80)
    parser.add_argument("--true-success-probability", type=float, default=0.90)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--max-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 0.5 < args.reliability_floor < 1.0:
        parser.error("--reliability-floor must lie in (0.5,1)")
    if not 0.5 < args.true_success_probability < 1.0:
        parser.error("--true-success-probability must lie in (0.5,1)")
    if not 0.0 < args.familywise_alpha < 0.5:
        parser.error("--familywise-alpha must lie in (0,0.5)")
    if not 0.0 < args.target_power < 1.0:
        parser.error("--target-power must lie in (0,1)")
    try:
        counts = [int(value) for value in args.context_counts.split(",")]
    except ValueError:
        parser.error("--context-counts must be a comma-separated list of positive integers")
    if not counts or any(value < 1 for value in counts):
        parser.error("--context-counts must contain positive integers")
    results = [
        plan_for_context_count(count, args.reliability_floor, args.true_success_probability,
                               args.familywise_alpha, args.target_power, args.max_replicates)
        for count in counts
    ]
    report = {
        "schema_version": "pitb.support-landscape-power-screen.v0",
        "parameters": {
            "reliability_floor": args.reliability_floor,
            "true_success_probability": args.true_success_probability,
            "familywise_alpha": args.familywise_alpha,
            "target_power": args.target_power,
        },
        "scenarios": results,
        "interpretation_boundary": "Design-only planning calculation. It does not validate the analyzer, estimate real prover reliability, or replace independent Monte Carlo/exact inference review.",
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
