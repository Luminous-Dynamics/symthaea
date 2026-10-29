#!/usr/bin/env python3
"""Synthetic, design-only screen for PITB component-first split allocation.

This is not a theorem-prover evaluation and does not load the private/unreviewed
PITB relationship graph. It compares the v0-style family-deficit allocator to a
candidate that fixes component-count quotas first, then performs deterministic
best-improving component swaps to reduce relative family-count imbalance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

SHARES = (0.70, 0.15, 0.15)
SPLITS = ("train", "validation", "test")
CANDIDATE_SEED = "pitb-component-first-candidate-2026-10-29"
SCREEN_SEED = 20261029
FAMILY_TOTAL = 372
LOSS_IMPROVEMENT_EPSILON = 1e-12


def digest_rank(component_id: str) -> str:
    material = f"{CANDIDATE_SEED}\0{component_id}".encode()
    return hashlib.sha256(material).hexdigest()


def quotas(component_count: int) -> tuple[int, int, int]:
    train_end = component_count * 70 // 100
    validation_end = component_count * 85 // 100
    return (
        train_end,
        validation_end - train_end,
        component_count - validation_end,
    )


def loss(family_counts: list[int], total_families: int) -> float:
    return sum(
        ((family_counts[i] - SHARES[i] * total_families) / max(SHARES[i] * total_families, 1.0)) ** 2
        for i in range(3)
    )


def metrics(sizes: list[int], component_assignment: list[int]) -> dict[str, Any]:
    family_counts = [0, 0, 0]
    component_counts = [0, 0, 0]
    for size, split in zip(sizes, component_assignment, strict=True):
        family_counts[split] += size
        component_counts[split] += 1
    total_families = sum(sizes)
    family_shares = [count / total_families for count in family_counts]
    component_shares = [count / len(sizes) for count in component_counts]
    family_error = max(abs(family_shares[i] - SHARES[i]) for i in range(3))
    component_error = max(abs(component_shares[i] - SHARES[i]) for i in range(3))
    return {
        "family_count_by_split": dict(zip(SPLITS, family_counts, strict=True)),
        "component_count_by_split": dict(zip(SPLITS, component_counts, strict=True)),
        "family_share_by_split": dict(zip(SPLITS, family_shares, strict=True)),
        "component_share_by_split": dict(zip(SPLITS, component_shares, strict=True)),
        "max_absolute_family_share_error": family_error,
        "max_absolute_component_share_error": component_error,
        "max_of_family_and_component_share_errors": max(family_error, component_error),
    }


def legacy_family_deficit(sizes: list[int]) -> list[int]:
    components = [(f"component-{i:04d}", size) for i, size in enumerate(sizes)]
    components.sort(key=lambda row: (digest_rank(row[0]), row[0]))
    targets = [share * sum(sizes) for share in SHARES]
    family_counts = [0, 0, 0]
    assignment_by_id: dict[str, int] = {}
    for component_id, size in components:
        deficits = [
            (targets[i] - family_counts[i]) / max(targets[i], 1.0)
            for i in range(3)
        ]
        split = max(range(3), key=lambda i: (deficits[i], -i))
        family_counts[split] += size
        assignment_by_id[component_id] = split
    return [assignment_by_id[f"component-{i:04d}"] for i in range(len(sizes))]


def component_first_with_family_swaps(sizes: list[int]) -> tuple[list[int], int]:
    components = [(f"component-{i:04d}", size) for i, size in enumerate(sizes)]
    components.sort(key=lambda row: (digest_rank(row[0]), row[0]))
    target_quotas = quotas(len(components))
    assignment = [
        0 if index < target_quotas[0]
        else 1 if index < target_quotas[0] + target_quotas[1]
        else 2
        for index in range(len(components))
    ]
    family_counts = [0, 0, 0]
    for (_, size), split in zip(components, assignment, strict=True):
        family_counts[split] += size

    iterations = 0
    while True:
        current_loss = loss(family_counts, sum(sizes))
        best_gain = LOSS_IMPROVEMENT_EPSILON
        best: tuple[int, int, list[int]] | None = None
        for left in range(len(components)):
            left_size = components[left][1]
            left_split = assignment[left]
            for right in range(left + 1, len(components)):
                right_size = components[right][1]
                right_split = assignment[right]
                if left_split == right_split or left_size == right_size:
                    continue
                candidate = family_counts.copy()
                candidate[left_split] = candidate[left_split] - left_size + right_size
                candidate[right_split] = candidate[right_split] - right_size + left_size
                gain = current_loss - loss(candidate, sum(sizes))
                if gain > best_gain:
                    best_gain = gain
                    best = (left, right, candidate)
        if best is None:
            break
        left, right, family_counts = best
        assignment[left], assignment[right] = assignment[right], assignment[left]
        iterations += 1

    expected = quotas(len(components))
    actual = tuple(assignment.count(i) for i in range(3))
    if actual != expected:
        raise AssertionError(f"component quota changed: expected {expected}, got {actual}")
    return assignment, iterations


def synthetic_random_partition(rng: random.Random) -> list[int]:
    sizes: list[int] = []
    total = 0
    while total < FAMILY_TOTAL:
        size = min(40, max(1, int(rng.expovariate(1 / 2.5)) + 1))
        size = min(size, FAMILY_TOTAL - total)
        sizes.append(size)
        total += size
    rng.shuffle(sizes)
    return sizes


def scenario_sizes() -> dict[str, list[int]]:
    return {
        "all_singletons": [1] * 372,
        "pairs_plus_singletons": [2] * 61 + [1] * 250,
        "one_giant_component_120": [120] + [1] * 252,
        "two_giant_components_80_70": [80, 70] + [1] * 222,
        "moderate_long_tail": [
            20, 16, 12, 10, 8, 8, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2
        ] + [1] * 258,
    }


def summarize_pair(sizes: list[int]) -> dict[str, Any]:
    legacy = legacy_family_deficit(sizes)
    candidate, swap_count = component_first_with_family_swaps(sizes)
    return {
        "family_total": sum(sizes),
        "component_total": len(sizes),
        "legacy_family_deficit": metrics(sizes, legacy),
        "component_first_with_family_swaps": {
            **metrics(sizes, candidate),
            "quota_preserving_swap_count": swap_count,
            "target_component_count_by_split": dict(zip(SPLITS, quotas(len(sizes)), strict=True)),
            "final_family_balance_loss": loss(
                [
                    sum(size for size, split in zip(sizes, candidate, strict=True) if split == i)
                    for i in range(3)
                ],
                sum(sizes),
            ),
        },
    }


def run_screen(random_replicates: int) -> dict[str, Any]:
    scenarios = {
        name: summarize_pair(sizes)
        for name, sizes in scenario_sizes().items()
    }
    rng = random.Random(SCREEN_SEED)
    improved_family = worse_family = equal_family = 0
    improved_joint = worse_joint = equal_joint = 0
    family_deltas: list[float] = []
    component_deltas: list[float] = []
    joint_deltas: list[float] = []
    swap_counts: list[int] = []
    for _ in range(random_replicates):
        sizes = synthetic_random_partition(rng)
        pair = summarize_pair(sizes)
        old = pair["legacy_family_deficit"]
        new = pair["component_first_with_family_swaps"]
        family_delta = new["max_absolute_family_share_error"] - old["max_absolute_family_share_error"]
        component_delta = new["max_absolute_component_share_error"] - old["max_absolute_component_share_error"]
        joint_delta = new["max_of_family_and_component_share_errors"] - old["max_of_family_and_component_share_errors"]
        family_deltas.append(family_delta)
        component_deltas.append(component_delta)
        joint_deltas.append(joint_delta)
        swap_counts.append(new["quota_preserving_swap_count"])
        if family_delta < -1e-12:
            improved_family += 1
        elif family_delta > 1e-12:
            worse_family += 1
        else:
            equal_family += 1
        if joint_delta < -1e-12:
            improved_joint += 1
        elif joint_delta > 1e-12:
            worse_joint += 1
        else:
            equal_joint += 1

    return {
        "schema_version": "pitb.component-allocation-sensitivity-screen.v0",
        "status": "synthetic_screen_only",
        "interpretation": "Design diagnostic only. It contains no PITB relationship graph, theorem outcomes, or model evaluation.",
        "screen_seed": SCREEN_SEED,
        "candidate_seed": CANDIDATE_SEED,
        "random_replicates": random_replicates,
        "synthetic_generator": {
            "family_total_per_replicate": FAMILY_TOTAL,
            "component_sizes": "independently sample positive sizes from a capped exponential-like distribution (mean scale 2.5), truncate the final component to total 372, then shuffle size-to-ID assignment",
            "comparison": "hash-ordered family-deficit allocation versus exact floor-based 70/15/15 component quotas followed by deterministic best-improving cross-split swaps minimizing squared relative family-count error",
            "reported_metrics": "maximum absolute split-share error for family counts, component counts, and the maximum of those errors",
        },
        "scenario_results": scenarios,
        "randomized_summary": {
            "family_share_error_better_equal_worse": {
                "better": improved_family,
                "equal": equal_family,
                "worse": worse_family,
            },
            "joint_max_share_error_better_equal_worse": {
                "better": improved_joint,
                "equal": equal_joint,
                "worse": worse_joint,
            },
            "mean_candidate_minus_legacy_family_error": statistics.mean(family_deltas),
            "median_candidate_minus_legacy_family_error": statistics.median(family_deltas),
            "mean_candidate_minus_legacy_component_error": statistics.mean(component_deltas),
            "median_candidate_minus_legacy_component_error": statistics.median(component_deltas),
            "mean_candidate_minus_legacy_joint_error": statistics.mean(joint_deltas),
            "median_candidate_minus_legacy_joint_error": statistics.median(joint_deltas),
            "mean_quota_preserving_swap_count": statistics.mean(swap_counts),
            "maximum_quota_preserving_swap_count": max(swap_counts, default=0),
        },
        "limitations": [
            "Synthetic component sizes do not model the actual PITB relationship graph.",
            "A deterministic swap procedure reaches a local optimum, not a global optimum.",
            "The 70/15/15 component quotas are a proposed successor design and require power analysis before confirmatory use.",
            "The results do not establish a lower false-claim rate, better theorem proving, transfer utility, or state of the art.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.random_replicates < 1:
        parser.error("--random-replicates must be >= 1")
    report = run_screen(args.random_replicates)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
