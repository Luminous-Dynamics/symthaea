#!/usr/bin/env python3
"""Independent PIE-001 interval mass-balance oracle.

Research/reference implementation only. This script does not import Symthaea
code, does not model chemistry, and does not authorize any industrial process.

Sign convention for residual:
    residual_kg = output_mass_kg - input_mass_kg

For uncertain intervals, interval overlap means only that conservation is
*possible* within the declared bounds. It never upgrades an uncertain case to
ExactBalanced.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


class MassBalanceStatus(str, Enum):
    """Conservative classification of one process-basis mass balance."""

    EXACT_BALANCED = "ExactBalanced"
    EXACT_UNBALANCED = "ExactUnbalanced"
    POSSIBLE_WITH_UNCERTAINTY = "PossibleWithUncertainty"
    IMPOSSIBLE_WITHIN_BOUNDS = "ImpossibleWithinBounds"


@dataclass(frozen=True)
class MassInterval:
    """Inclusive mass interval in kilograms."""

    min_kg: float
    max_kg: float

    def validate(self) -> None:
        if not (math.isfinite(self.min_kg) and math.isfinite(self.max_kg)):
            raise ValueError("mass interval endpoints must be finite")
        if self.min_kg < 0.0 or self.max_kg < 0.0:
            raise ValueError("mass interval endpoints must be nonnegative")
        if self.min_kg > self.max_kg:
            raise ValueError("mass interval min_kg must be <= max_kg")

    @property
    def is_point(self) -> bool:
        return self.min_kg == self.max_kg


@dataclass(frozen=True)
class MassBalanceTolerance:
    """Absolute + relative tolerance policy."""

    absolute_kg: float = 0.0
    relative_fraction: float = 0.0

    def validate(self) -> None:
        if not (math.isfinite(self.absolute_kg) and math.isfinite(self.relative_fraction)):
            raise ValueError("tolerances must be finite")
        if self.absolute_kg < 0.0 or self.relative_fraction < 0.0:
            raise ValueError("tolerances must be nonnegative")


@dataclass(frozen=True)
class MassBalanceReport:
    """Mass-balance result for one declared process basis."""

    status: str
    input_min_kg: float
    input_max_kg: float
    output_min_kg: float
    output_max_kg: float
    residual_min_kg: float
    residual_max_kg: float
    tolerance_kg: float
    exact_residual_kg: float | None
    all_streams_exact: bool


def _checked_sum(values: Iterable[float], label: str) -> float:
    """Sum non-negative finite values and reject aggregate floating overflow."""
    total = 0.0
    for value in values:
        total += value
        if not math.isfinite(total):
            raise ValueError(f"{label} must remain finite after aggregation")
    return total


def _sum_intervals(streams: list[MassInterval]) -> MassInterval:
    if not streams:
        raise ValueError("at least one mass stream is required")
    for stream in streams:
        stream.validate()
    return MassInterval(
        min_kg=_checked_sum(
            (stream.min_kg for stream in streams), "aggregate minimum mass"
        ),
        max_kg=_checked_sum(
            (stream.max_kg for stream in streams), "aggregate maximum mass"
        ),
    )


def evaluate_mass_balance(
    inputs: list[MassInterval],
    outputs: list[MassInterval],
    tolerance: MassBalanceTolerance,
) -> MassBalanceReport:
    """Classify a process-basis mass balance conservatively."""
    tolerance.validate()
    total_in = _sum_intervals(inputs)
    total_out = _sum_intervals(outputs)

    reference_mass = max(total_in.max_kg, total_out.max_kg)
    relative_tolerance_kg = tolerance.relative_fraction * reference_mass
    if not math.isfinite(relative_tolerance_kg):
        raise ValueError("effective relative mass-balance tolerance must be finite")
    tolerance_kg = max(tolerance.absolute_kg, relative_tolerance_kg)
    if not math.isfinite(tolerance_kg):
        raise ValueError("effective mass-balance tolerance must be finite")

    # output - input
    residual_min = total_out.min_kg - total_in.max_kg
    residual_max = total_out.max_kg - total_in.min_kg
    if not (math.isfinite(residual_min) and math.isfinite(residual_max)):
        raise ValueError("mass-balance residual bounds must remain finite")

    all_exact = all(stream.is_point for stream in inputs + outputs)
    if all_exact:
        exact_residual = total_out.min_kg - total_in.min_kg
        if not math.isfinite(exact_residual):
            raise ValueError("exact mass-balance residual must remain finite")
        status = (
            MassBalanceStatus.EXACT_BALANCED
            if abs(exact_residual) <= tolerance_kg
            else MassBalanceStatus.EXACT_UNBALANCED
        )
        return MassBalanceReport(
            status=status.value,
            input_min_kg=total_in.min_kg,
            input_max_kg=total_in.max_kg,
            output_min_kg=total_out.min_kg,
            output_max_kg=total_out.max_kg,
            residual_min_kg=residual_min,
            residual_max_kg=residual_max,
            tolerance_kg=tolerance_kg,
            exact_residual_kg=exact_residual,
            all_streams_exact=True,
        )

    # The residual interval is output-input. It is disjoint from zero beyond the
    # declared tolerance exactly when the entire interval lies above +tolerance
    # or below -tolerance. Comparing residual bounds avoids overflow-prone
    # additions such as input_max + tolerance.
    disjoint_beyond_tolerance = (
        residual_min > tolerance_kg or residual_max < -tolerance_kg
    )
    status = (
        MassBalanceStatus.IMPOSSIBLE_WITHIN_BOUNDS
        if disjoint_beyond_tolerance
        else MassBalanceStatus.POSSIBLE_WITH_UNCERTAINTY
    )
    return MassBalanceReport(
        status=status.value,
        input_min_kg=total_in.min_kg,
        input_max_kg=total_in.max_kg,
        output_min_kg=total_out.min_kg,
        output_max_kg=total_out.max_kg,
        residual_min_kg=residual_min,
        residual_max_kg=residual_max,
        tolerance_kg=tolerance_kg,
        exact_residual_kg=None,
        all_streams_exact=False,
    )


def _interval_from_json(item: dict[str, Any]) -> MassInterval:
    return MassInterval(float(item["min_kg"]), float(item["max_kg"]))


def evaluate_json(payload: dict[str, Any]) -> MassBalanceReport:
    inputs = [_interval_from_json(item) for item in payload["inputs"]]
    outputs = [_interval_from_json(item) for item in payload["outputs"]]
    tolerance_payload = payload.get("tolerance", {})
    tolerance = MassBalanceTolerance(
        absolute_kg=float(tolerance_payload.get("absolute_kg", 0.0)),
        relative_fraction=float(tolerance_payload.get("relative_fraction", 0.0)),
    )
    return evaluate_mass_balance(inputs, outputs, tolerance)


def self_test() -> None:
    zero = MassBalanceTolerance()

    # Exact closure.
    r = evaluate_mass_balance(
        [MassInterval(10.0, 10.0)],
        [MassInterval(10.0, 10.0)],
        zero,
    )
    assert r.status == MassBalanceStatus.EXACT_BALANCED.value
    assert r.exact_residual_kg == 0.0

    # Exact mismatch is controlled only by explicit tolerance.
    loose = evaluate_mass_balance(
        [MassInterval(10.0, 10.0)],
        [MassInterval(9.9, 9.9)],
        MassBalanceTolerance(absolute_kg=0.2),
    )
    tight = evaluate_mass_balance(
        [MassInterval(10.0, 10.0)],
        [MassInterval(9.9, 9.9)],
        MassBalanceTolerance(absolute_kg=0.01),
    )
    assert loose.status == MassBalanceStatus.EXACT_BALANCED.value
    assert tight.status == MassBalanceStatus.EXACT_UNBALANCED.value

    # Overlapping uncertain intervals are only possible, never exact evidence.
    possible = evaluate_mass_balance(
        [MassInterval(9.0, 11.0)],
        [MassInterval(10.0, 12.0)],
        zero,
    )
    assert possible.status == MassBalanceStatus.POSSIBLE_WITH_UNCERTAINTY.value
    assert possible.exact_residual_kg is None

    # Disjoint intervals cannot close.
    impossible = evaluate_mass_balance(
        [MassInterval(9.0, 10.0)],
        [MassInterval(11.0, 12.0)],
        zero,
    )
    assert impossible.status == MassBalanceStatus.IMPOSSIBLE_WITHIN_BOUNDS.value

    # Product + by-product + waste all remain in the output total.
    split = evaluate_mass_balance(
        [MassInterval(10.0, 10.0)],
        [
            MassInterval(6.0, 6.0),  # product
            MassInterval(2.0, 2.0),  # by-product
            MassInterval(2.0, 2.0),  # waste
        ],
        zero,
    )
    assert split.status == MassBalanceStatus.EXACT_BALANCED.value
    assert split.output_min_kg == 10.0

    # Multiple inputs (e.g. feedstock + reagent/working fluid) all count.
    multi_input = evaluate_mass_balance(
        [MassInterval(8.0, 8.0), MassInterval(2.0, 2.0)],
        [MassInterval(10.0, 10.0)],
        zero,
    )
    assert multi_input.status == MassBalanceStatus.EXACT_BALANCED.value

    # More uncertainty may weaken an exact failure to "possible", but can never
    # turn it into ExactBalanced.
    exact_bad = evaluate_mass_balance(
        [MassInterval(10.0, 10.0)],
        [MassInterval(9.0, 9.0)],
        zero,
    )
    widened = evaluate_mass_balance(
        [MassInterval(9.0, 11.0)],
        [MassInterval(8.0, 10.0)],
        zero,
    )
    assert exact_bad.status == MassBalanceStatus.EXACT_UNBALANCED.value
    assert widened.status == MassBalanceStatus.POSSIBLE_WITH_UNCERTAINTY.value
    assert widened.status != MassBalanceStatus.EXACT_BALANCED.value

    # Relative tolerance uses the larger conservative total bound.
    rel = evaluate_mass_balance(
        [MassInterval(100.0, 100.0)],
        [MassInterval(99.0, 99.0)],
        MassBalanceTolerance(relative_fraction=0.02),
    )
    assert rel.status == MassBalanceStatus.EXACT_BALANCED.value
    assert abs(rel.tolerance_kg - 2.0) < 1e-12

    # Aggregate floating-point overflow must fail closed rather than creating an
    # infinite total that could accidentally satisfy a comparison.
    overflow_cases = [
        lambda: evaluate_mass_balance(
            [
                MassInterval(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")),
                MassInterval(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")),
            ],
            [MassInterval(1.0, 1.0)],
            zero,
        ),
        lambda: evaluate_mass_balance(
            [MassInterval(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023"))],
            [MassInterval(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023"))],
            MassBalanceTolerance(
                relative_fraction=float.fromhex("0x1.fffffffffffffp+1023")
            ),
        ),
    ]
    for case in overflow_cases:
        try:
            case()
        except ValueError:
            pass
        else:
            raise AssertionError("floating-point overflow must fail closed")

    # Malformed inputs fail closed.
    malformed = [
        lambda: evaluate_mass_balance(
            [MassInterval(-1.0, 1.0)], [MassInterval(0.0, 0.0)], zero
        ),
        lambda: evaluate_mass_balance(
            [MassInterval(2.0, 1.0)], [MassInterval(0.0, 0.0)], zero
        ),
        lambda: evaluate_mass_balance(
            [MassInterval(float("nan"), 1.0)], [MassInterval(0.0, 0.0)], zero
        ),
        lambda: evaluate_mass_balance([], [MassInterval(0.0, 0.0)], zero),
        lambda: evaluate_mass_balance(
            [MassInterval(0.0, 0.0)],
            [],
            MassBalanceTolerance(),
        ),
        lambda: evaluate_mass_balance(
            [MassInterval(0.0, 0.0)],
            [MassInterval(0.0, 0.0)],
            MassBalanceTolerance(absolute_kg=-1.0),
        ),
    ]
    for case in malformed:
        try:
            case()
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise AssertionError("malformed mass-balance input must fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", help="inline JSON payload")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return

    if not args.json:
        parser.error("--self-test or --json is required")

    payload = json.loads(args.json)
    report = evaluate_json(payload)
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
