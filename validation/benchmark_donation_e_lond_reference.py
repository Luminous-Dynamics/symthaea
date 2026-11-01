"""Exact microbenchmark for online donation e-LOND.

Measures the indexed single-pass path, a final O(t) reference threshold scan,
and an independent O(t) state verifier. It is a Python validation benchmark,
not a production Rust benchmark or a stochastic-validity proof.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from fractions import Fraction
from typing import Any

from donation_e_lond_reference import (
    DonationELONDIndex,
    donation_threshold_reference,
)


def _stream(n: int, seed: int) -> tuple[list[Fraction], list[Fraction]]:
    rng = random.Random(seed)
    gamma = [Fraction(1, i * (i + 1)) for i in range(1, n + 1)]
    evidence = [
        Fraction(rng.randrange(0, 100_001), 1000) for _ in range(n)
    ]
    return evidence, gamma


def _trial(n: int, seed: int, delta: Fraction = Fraction(1, 10)) -> dict[str, Any]:
    evidence, gamma = _stream(n, seed)
    index = DonationELONDIndex(delta)

    start = time.perf_counter_ns()
    rows = []
    for e_value, weight in zip(evidence, gamma):
        rows.append(index.step(e_value, weight))
    indexed_run_ms = (time.perf_counter_ns() - start) / 1e6

    # Measure one independent O(t) threshold scan at the final arrival.
    next_weight = Fraction(1, n + 1) / Fraction(1, n + 2)
    rejected_ids = [
        i + 1 for i, rejected in enumerate(index.rejected) if rejected
    ]
    start = time.perf_counter_ns()
    reference = donation_threshold_reference(
        evidence, gamma, rejected_ids, next_weight, delta
    )
    reference_query_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    verified = index.verify()
    index.assert_valid_avl()
    verify_ms = (time.perf_counter_ns() - start) / 1e6

    indexed = index.threshold(next_weight)
    if not verified:
        raise AssertionError("independent final-state verifier failed")
    for key in (
        "wealth",
        "rejection_count",
        "capped_wealth",
        "denominator",
        "test_level",
        "e_value_threshold",
    ):
        if indexed[key] != reference[key]:
            raise AssertionError(f"indexed and reference {key} mismatch")
    if len(rows) != n:
        raise AssertionError("one decision per arriving e-value is required")

    return {
        "n": n,
        "rejected": index.rejection_count,
        "indexed_online_run_ms": indexed_run_ms,
        "final_scan_query_ms": reference_query_ms,
        "final_verifier_ms": verify_ms,
        "final_threshold_exact_match": True,
        "final_verifier_passed": True,
        "avl_height": index.root.height if index.root else 0,
    }


def run(
    sizes: tuple[int, ...] = (1_000, 10_000, 50_000),
    repeats: int = 3,
    seed: int = 9813,
) -> dict[str, Any]:
    if repeats < 1 or any(n < 1 for n in sizes):
        raise ValueError("repeats and sizes must be positive")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        trials = [_trial(n, seed + n + rep) for rep in range(repeats)]
        rows.append(
            {
                "n": n,
                "repeats": repeats,
                "median_indexed_online_run_ms": statistics.median(
                    item["indexed_online_run_ms"] for item in trials
                ),
                "median_final_scan_query_ms": statistics.median(
                    item["final_scan_query_ms"] for item in trials
                ),
                "median_final_verifier_ms": statistics.median(
                    item["final_verifier_ms"] for item in trials
                ),
                "median_rejected": statistics.median(
                    item["rejected"] for item in trials
                ),
                "median_avl_height": statistics.median(
                    item["avl_height"] for item in trials
                ),
                "all_final_thresholds_match": all(
                    item["final_threshold_exact_match"] for item in trials
                ),
                "all_verifiers_pass": all(
                    item["final_verifier_passed"] for item in trials
                ),
            }
        )
    return {
        "benchmark": "donation_e_lond_avl_vs_scan",
        "version": 1,
        "seed": seed,
        "delta": "1/10",
        "gamma": "1/(i*(i+1))",
        "e_value_generator": "uniform integer 0..100000 divided by 1000",
        "arithmetic": "fractions.Fraction exact",
        "timing": "time.perf_counter_ns",
        "results": rows,
        "limitations": [
            "Python reference only, not production Rust latency.",
            "Online run includes AVL updates and O(log t) decision queries.",
            "Final scan and final independent verifier are O(t) checks, measured separately.",
            "Single synthetic e-value generator; statistical validity comes from the e-value inputs/theorem assumptions, not this performance benchmark.",
            "One e-value is supplied per hypothesis; repeated evidence updates are outside the published donation e-LOND theorem.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 50_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9813)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    result = run(tuple(args.sizes), args.repeats, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
