"""Benchmark additive supremum-anchored e-BH selection.

The benchmark measures the exact O(n log n) e-BH selector, an independent
O(n) closure-feasibility verifier, and the exhaustive all-subsets closure
oracle only on small finite families. This is a computational benchmark, not
a simulation of the theorem's stochastic assumptions.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import tracemalloc
from fractions import Fraction
from itertools import combinations
from typing import Any

from supremum_anchored_eclosure import (
    select_additive_anchored_e_bh,
    select_largest_certified_superset,
)


def _make_values(n: int, seed: int, alpha: Fraction) -> tuple[dict[str, Fraction], frozenset[str]]:
    rng = random.Random(seed)
    ids = [f"h-{i:08d}" for i in range(n)]
    anchor = frozenset(i for i in ids if rng.random() < 0.12)
    values: dict[str, Fraction] = {}
    for hypothesis_id in ids:
        numerator = rng.randint(0, 500_000)
        # Ever-rejected baseline members have e_i >= 1/alpha.
        if hypothesis_id in anchor:
            numerator += 1_000_000
        values[hypothesis_id] = Fraction(numerator, 50_000)
    return values, anchor


def _linear_closure_verify(
    values: dict[str, Fraction],
    rejection: frozenset[str],
    alpha: Fraction,
) -> bool:
    """Independent O(n) check of all additive closure constraints.

    Because local E_S=sum_{i in S} e_i, all subset constraints hold iff
    every selected coordinate clears 1/(alpha*|R|).
    """
    if not rejection:
        return True
    cutoff = Fraction(1, 1) / (alpha * len(rejection))
    return all(values[i] >= cutoff for i in rejection)


def _exhaustive_closure(
    values: dict[str, Fraction],
    anchor: frozenset[str],
    alpha: Fraction,
) -> frozenset[str]:
    ids = tuple(sorted(values))
    subset_values = {
        frozenset(row): sum((values[i] for i in row), Fraction(0))
        for size in range(1, len(ids) + 1)
        for row in combinations(ids, size)
    }
    return select_largest_certified_superset(
        universe=ids,
        anchor=anchor,
        local_e_values=subset_values,
        alpha=alpha,
    )


def _time_call(callable_, repeats: int) -> tuple[Any, list[float]]:
    timings: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = callable_()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return result, timings


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[min(len(values) - 1, int(0.95 * len(values)))],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _trial(n: int, seed: int, repeats: int, alpha: Fraction) -> dict[str, Any]:
    values, anchor = _make_values(n, seed, alpha)
    selected, selector_times = _time_call(
        lambda: select_additive_anchored_e_bh(
            coordinate_e_values=values,
            alpha=alpha,
            anchor=anchor,
        ),
        repeats,
    )
    feasible, verifier_times = _time_call(
        lambda: _linear_closure_verify(values, selected, alpha),
        repeats,
    )
    if not feasible:
        raise AssertionError("e-BH selector returned a closure-infeasible set")
    if not anchor <= selected:
        raise AssertionError("e-BH selector did not retain the baseline anchor")

    # Measure peak allocator overhead for one selector call independently.
    tracemalloc.start()
    select_additive_anchored_e_bh(
        coordinate_e_values=values, alpha=alpha, anchor=anchor
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exact_match = None
    exhaustive_ms = None
    if n <= 16:
        exact, exhaustive_times = _time_call(
            lambda: _exhaustive_closure(values, anchor, alpha), 1
        )
        exhaustive_ms = exhaustive_times[0]
        exact_match = exact == selected
        if not exact_match:
            raise AssertionError("e-BH selector differs from exhaustive closure oracle")

    return {
        "n": n,
        "baseline_anchor_size": len(anchor),
        "selected_size": len(selected),
        "additional_over_baseline": len(selected - anchor),
        "selector": _summary(selector_times),
        "independent_linear_verifier": _summary(verifier_times),
        "exhaustive_all_subsets_ms": exhaustive_ms,
        "matches_exhaustive": exact_match,
        "peak_traced_bytes_selector": peak_bytes,
        "anchor_retained": anchor <= selected,
        "closure_feasible": feasible,
    }


def run(
    sizes: tuple[int, ...] = (8, 12, 16, 100, 1_000, 10_000, 100_000),
    repeats: int = 7,
    seed: int = 97801,
    alpha: Fraction = Fraction(1, 20),
) -> dict[str, Any]:
    if repeats < 1 or not sizes or any(n < 1 for n in sizes):
        raise ValueError("sizes and repeats must be positive")
    rows = [
        _trial(n, seed + 17 * n, repeats, alpha)
        for n in sizes
    ]
    return {
        "benchmark": "supremum_anchored_additive_ebh",
        "version": 1,
        "seed": seed,
        "repeats": repeats,
        "alpha": f"{alpha.numerator}/{alpha.denominator}",
        "arithmetic": "fractions.Fraction exact",
        "selector_complexity": "O(n log n)",
        "independent_closure_verifier_complexity": "O(n)",
        "exhaustive_oracle_max_n": 16,
        "results": rows,
        "limitations": [
            "Computational selector benchmark only; no stochastic FDR calibration claim.",
            "The theorem requires ex-ante coordinate risk budgets q_i and auxiliary running-supremum budgets v_i.",
            "The exhaustive oracle is used only for small n; larger instances are checked by the independent additive-closure criterion.",
            "Python timings are not Rust production p50/p95 measurements.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[8, 12, 16, 100, 1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=97801)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    payload = run(tuple(args.sizes), args.repeats, args.seed)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
