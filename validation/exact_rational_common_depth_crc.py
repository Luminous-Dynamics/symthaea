#!/usr/bin/env python3
"""Exact-rational verifier for the common-depth CRC recertification table.

Independent reference implementation of Bai & Ramdas (2026), Appendix B.7.
It enumerates every calibration and augmented count vector and uses Fraction
arithmetic throughout. No tolerance, Monte Carlo result, or float comparison
can turn an invalid table into a passing certificate.

This is a research oracle, not a production Rust verifier.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable

SCHEMA = "symthaea-exact-rational-common-depth-crc-v1"
SNAPSHOT_ROUNDS = (0, 1, 10, 20, 50, 100, 200, 500)


def as_fraction(value: Fraction | int | str | float) -> Fraction:
    """Convert decimal text exactly; do not import a float's binary expansion."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("alpha must be finite")
        return Fraction(str(value))
    return Fraction(value)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def count_vectors(total: int, types: int = 3) -> list[tuple[int, ...]]:
    if total < 0 or types < 1:
        raise ValueError("total must be nonnegative and types positive")
    if types == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for tail in count_vectors(total - first, types - 1):
            result.append((first,) + tail)
    return result


def error_loss(error_type: int, depth: Fraction) -> Fraction:
    if not Fraction(0) <= depth <= Fraction(2):
        raise ValueError("depth must lie in [0,2]")
    if error_type == 0:
        return Fraction(0)
    if error_type == 1:
        return max(depth - 1, Fraction(0)) / 2
    if error_type == 2:
        return depth / 2
    raise ValueError("error type must be 0, 1 or 2")


def largest_piecewise_feasible_depth(
    h1: int, h2: int, budget: Fraction, *, envelope: bool
) -> Fraction:
    """Solve the CRC/envelope baseline exactly on [0,1] and [1,2]."""
    if h1 < 0 or h2 < 0:
        raise ValueError("calibration type counts must be nonnegative")

    if envelope:
        # For a <= 1: (h2 + 1) * a / 2 <= budget.
        slope0 = Fraction(h2 + 1, 2)
        a0 = min(Fraction(1), budget / slope0)
        if a0 < 1:
            return max(Fraction(0), a0)

        # For a >= 1: h1*(a-1) + (h2+1)*a/2 <= budget.
        slope1 = Fraction(h1) + Fraction(h2 + 1, 2)
        if slope1 == 0:
            return Fraction(2)
        candidate = (budget + h1) / slope1
        return min(Fraction(2), max(Fraction(1), candidate))

    # Standard CRC uses the constant unseen-test-loss envelope U(a)=1.
    if budget < 1:
        return Fraction(0)
    slope0 = Fraction(h2, 2)
    a0 = min(Fraction(1), (budget - 1) / slope0) if slope0 else Fraction(1)
    if a0 < 1:
        return max(Fraction(0), a0)

    slope1 = Fraction(h1) + Fraction(h2, 2)
    if slope1 == 0:
        return Fraction(2)
    candidate = (budget - 1 + h1) / slope1
    return min(Fraction(2), max(Fraction(1), candidate))


class ExactCommonDepthCRC:
    """Exact count-vector controller and verifier for the CIFAR common-depth family."""

    def __init__(self, n_calibration: int, alpha: Fraction | str | float):
        if n_calibration < 1:
            raise ValueError("n_calibration must be positive")
        alpha_fraction = as_fraction(alpha)
        if not 0 < alpha_fraction < 1:
            raise ValueError("alpha must lie in (0,1)")
        self.n = int(n_calibration)
        self.alpha = alpha_fraction
        self.budget = (self.n + 1) * self.alpha
        self.states = count_vectors(self.n, 3)
        self.augmented_states = count_vectors(self.n + 1, 3)
        self._state_set = set(self.states)

    def baseline(self, kind: str = "envelope") -> dict[tuple[int, ...], Fraction]:
        if kind not in {"crc", "envelope"}:
            raise ValueError("kind must be 'crc' or 'envelope'")
        envelope = kind == "envelope"
        table = {
            h: largest_piecewise_feasible_depth(
                h[1], h[2], self.budget, envelope=envelope
            )
            for h in self.states
        }
        report = self.verify(table)
        if not report["valid"]:
            raise ArithmeticError(f"{kind} baseline is invalid: {report}")
        return table

    def augmented_totals(
        self, table: dict[tuple[int, ...], Fraction]
    ) -> dict[tuple[int, ...], Fraction]:
        if set(table) != self._state_set:
            raise ValueError("table must cover every calibration count vector exactly once")
        for depth in table.values():
            if not isinstance(depth, Fraction) or not 0 <= depth <= 2:
                raise ValueError("all table depths must be Fractions in [0,2]")

        totals: dict[tuple[int, ...], Fraction] = {}
        for q in self.augmented_states:
            total = Fraction(0)
            for error_type, multiplicity in enumerate(q):
                if multiplicity == 0:
                    continue
                heldout = list(q)
                heldout[error_type] -= 1
                h = tuple(heldout)
                total += multiplicity * error_loss(error_type, table[h])
            totals[q] = total
        return totals

    def verify(
        self, table: dict[tuple[int, ...], Fraction],
        totals: dict[tuple[int, ...], Fraction] | None = None,
    ) -> dict:
        computed = self.augmented_totals(table) if totals is None else totals
        if set(computed) != set(self.augmented_states):
            raise ValueError("verification totals omit one or more augmented states")
        worst_q = self.augmented_states[0]
        worst_total = computed[worst_q]
        for q in self.augmented_states[1:]:
            if computed[q] > worst_total:
                worst_q, worst_total = q, computed[q]
        excess = worst_total - self.budget
        return {
            "valid": excess <= 0,
            "worst_augmented_count": list(worst_q),
            "worst_total": fraction_text(worst_total),
            "budget": fraction_text(self.budget),
            "exact_excess": fraction_text(excess),
            "positive_excess": excess > 0,
            "checked_augmented_states": len(computed),
        }

    def recertify_round(
        self,
        table: dict[tuple[int, ...], Fraction],
        totals: dict[tuple[int, ...], Fraction] | None = None,
    ) -> tuple[dict[tuple[int, ...], Fraction], dict[tuple[int, ...], Fraction]]:
        current_totals = self.augmented_totals(table) if totals is None else totals
        initial_report = self.verify(table, current_totals)
        if not initial_report["valid"]:
            raise ValueError(f"starting table is invalid: {initial_report}")

        updated: dict[tuple[int, ...], Fraction] = {}
        for h in self.states:
            a = table[h]
            q1 = (h[0], h[1] + 1, h[2])
            q2 = (h[0], h[1], h[2] + 1)
            slack1 = self.alpha - current_totals[q1] / (self.n + 1)
            slack2 = self.alpha - current_totals[q2] / (self.n + 1)
            if slack1 < 0 or slack2 < 0:
                raise ArithmeticError(
                    f"negative exact slack at {h}: type1={slack1}, type2={slack2}"
                )
            candidate_type2 = a + 2 * slack2
            candidate_type1 = 1 + max(a - 1, Fraction(0)) + 2 * slack1
            updated[h] = min(Fraction(2), candidate_type1, candidate_type2)

        if any(updated[h] < table[h] for h in self.states):
            raise ArithmeticError("recertification decreased a depth")
        updated_totals = self.augmented_totals(updated)
        updated_report = self.verify(updated, updated_totals)
        if not updated_report["valid"]:
            raise ArithmeticError(f"recertification broke exact validity: {updated_report}")
        return updated, updated_totals

    def run(
        self,
        kind: str = "envelope",
        rounds: int = 500,
        snapshot_rounds: Iterable[int] = SNAPSHOT_ROUNDS,
    ) -> dict:
        if rounds < 0:
            raise ValueError("rounds must be nonnegative")
        requested = sorted(set(int(r) for r in snapshot_rounds))
        if any(r < 0 or r > rounds for r in requested):
            raise ValueError("snapshot rounds must lie in [0,rounds]")
        table = self.baseline(kind)
        totals = self.augmented_totals(table)
        snapshots = []
        validation_table_count = 1
        if 0 in requested:
            snapshots.append(self._snapshot(0, table, totals))
        for round_number in range(1, rounds + 1):
            table, totals = self.recertify_round(table, totals)
            validation_table_count += 1
            if round_number in requested:
                snapshots.append(self._snapshot(round_number, table, totals))
        return {
            "n_calibration": self.n,
            "alpha": fraction_text(self.alpha),
            "kind": kind,
            "rounds": rounds,
            "calibration_states": len(self.states),
            "augmented_states": len(self.augmented_states),
            "verified_table_count": validation_table_count,
            "exact_augmented_inequality_checks": validation_table_count * len(self.augmented_states),
            "all_rounds_valid": True,
            "snapshots": snapshots,
            "final_depth_max_denominator_digits": max(
                len(str(value.denominator)) for value in table.values()
            ),
        }

    def _snapshot(self, round_number, table, totals) -> dict:
        report = self.verify(table, totals)
        return {
            "round": round_number,
            "valid": report["valid"],
            "worst_augmented_count": report["worst_augmented_count"],
            "worst_total": report["worst_total"],
            "budget": report["budget"],
            "exact_excess": report["exact_excess"],
            "min_depth": fraction_text(min(table.values())),
            "max_depth": fraction_text(max(table.values())),
            "max_denominator_digits": max(
                len(str(value.denominator)) for value in table.values()
            ),
        }


def run_reference_targets(
    n_calibration: int = 50,
    alphas: Iterable[str] = ("0.02", "0.05", "0.10", "0.15"),
    rounds: int = 500,
) -> dict:
    started = time.perf_counter()
    results = []
    for alpha in alphas:
        engine = ExactCommonDepthCRC(n_calibration, alpha)
        results.append(engine.run("envelope", rounds))
    return {
        "schema": SCHEMA,
        "status": "EXACT_RATIONAL_COMMON_DEPTH_CRC_VERIFICATION",
        "assumption": "finite exchangeability; deterministic augmented multiset characterization",
        "input": {
            "n_calibration": n_calibration,
            "alphas": [fraction_text(as_fraction(x)) for x in alphas],
            "rounds": rounds,
        },
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version()},
        "limitations": [
            "common-depth CIFAR family only; not the confidence-adaptive per-image controller",
            "exactness applies to rational alpha and the exact piecewise-linear loss equations",
            "does not prove validity under arbitrary distribution shift",
            "Python Fraction is a reference oracle, not the production runtime target",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--alpha", nargs="+", default=["0.02", "0.05", "0.10", "0.15"])
    parser.add_argument("--rounds", type=int, default=500)
    args = parser.parse_args()
    result = run_reference_targets(args.n, args.alpha, args.rounds)
    Path(args.output).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": args.output,
        "runtime_seconds": result["runtime_seconds"],
        "all_rounds_valid": all(x["all_rounds_valid"] for x in result["results"]),
        "total_exact_inequality_checks": sum(
            x["exact_augmented_inequality_checks"] for x in result["results"]
        ),
    }, indent=2))
