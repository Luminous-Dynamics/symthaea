"""Sparse-update event benchmark for the exact projective closure index.

Uses Fraction arithmetic and the module's independent O(n) reference scan.
This is a Python research microbenchmark, not a production Rust benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from fractions import Fraction
from typing import Any

from projective_closure_index import ProjectiveClosureIndex, raw_separator_score


def _fraction(rng: random.Random) -> Fraction:
    return Fraction(rng.randint(-1_000_000, 1_000_000), 1_000_000)


def _trial(n: int, k: int, seed: int) -> dict[str, Any]:
    if n < 1 or not 0 <= k <= n:
        raise ValueError("require n >= 1 and 0 <= k <= n")
    rng = random.Random(seed)
    values = [_fraction(rng) for _ in range(n)]
    selected = [rng.random() < 0.5 for _ in range(n)]
    indices = rng.sample(range(n), k)
    updates = [(i, _fraction(rng)) for i in indices]

    build_start = time.perf_counter_ns()
    index = ProjectiveClosureIndex(values, selected)
    build_ns = time.perf_counter_ns() - build_start

    fast_values = values.copy()
    fast_start = time.perf_counter_ns()
    for i, value in updates:
        index.update_evidence(i, value)
        fast_values[i] = value
    fast_score = index.score()
    fast_ns = time.perf_counter_ns() - fast_start

    scan_values = values.copy()
    scan_start = time.perf_counter_ns()
    for i, value in updates:
        scan_values[i] = value
    scan_score = raw_separator_score(scan_values, selected)
    scan_ns = time.perf_counter_ns() - scan_start

    if fast_score != scan_score:
        raise AssertionError("indexed score differs from independent O(n) scan")
    if not index.verify():
        raise AssertionError("independent index verifier failed")

    return {
        "n": n,
        "k_changed": k,
        "build_ms": build_ns / 1e6,
        "indexed_update_and_query_ms": fast_ns / 1e6,
        "scan_update_and_query_ms": scan_ns / 1e6,
        "speedup": scan_ns / fast_ns if fast_ns else None,
        "exact_scores_equal": True,
    }


def run(
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000),
    update_fractions: tuple[float, ...] = (0.00001, 0.00064, 0.01, 0.1),
    repeats: int = 3,
    seed: int = 9746,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        update_counts = sorted(
            {
                max(1, min(n, int(round(n * fraction))))
                for fraction in update_fractions
            }
        )
        for k in update_counts:
            trials = [_trial(n, k, seed + n + 101 * k + rep) for rep in range(repeats)]
            rows.append(
                {
                    "n": n,
                    "k_changed": k,
                    "repeats": repeats,
                    "median_build_ms": statistics.median(x["build_ms"] for x in trials),
                    "median_indexed_update_and_query_ms": statistics.median(
                        x["indexed_update_and_query_ms"] for x in trials
                    ),
                    "median_scan_update_and_query_ms": statistics.median(
                        x["scan_update_and_query_ms"] for x in trials
                    ),
                    "median_speedup": statistics.median(x["speedup"] for x in trials),
                    "all_exact_scores_equal": all(x["exact_scores_equal"] for x in trials),
                }
            )
    return {
        "benchmark": "projective_closure_index_sparse_update",
        "version": 1,
        "seed": seed,
        "arithmetic": "fractions.Fraction exact rational",
        "timer": "time.perf_counter_ns",
        "repeats": repeats,
        "build_cost_reported_separately": True,
        "scan_is_independent_reference": True,
        "results": rows,
        "limitations": [
            "Python prototype only; not a Rust production benchmark.",
            "Index build cost is reported separately and excluded from event latency.",
            "Dense updates may be slower than a full scan.",
            "One hardware/runtime environment does not establish general performance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9746)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    payload = run(repeats=args.repeats, seed=args.seed)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
