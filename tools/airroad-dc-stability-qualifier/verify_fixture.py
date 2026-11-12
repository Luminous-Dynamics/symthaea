#!/usr/bin/env python3
"""Independent exact-rational checker for the AIRROAD DC stability fixture.

This is an offline qualification helper only. Production engineering code remains
Rust. The script intentionally imports no Symthaea or production-model module.
Run from the repository root:

    python3 tools/airroad-dc-stability-qualifier/verify_fixture.py
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any


def rational(obj: dict[str, Any]) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def exact_sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("cannot take an exact real square root of a negative value")
    numerator_root = isqrt(value.numerator)
    denominator_root = isqrt(value.denominator)
    if numerator_root * numerator_root != value.numerator:
        raise ValueError(f"non-square rational numerator: {value}")
    if denominator_root * denominator_root != value.denominator:
        raise ValueError(f"non-square rational denominator: {value}")
    return Fraction(numerator_root, denominator_root)


def require_equal(actual: Fraction, expected: Fraction, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def verify_fixture(path: Path) -> int:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    model = fixture["model"]["parameters"]
    source_voltage = rational(model["Vs"]["exact_rational"])
    source_resistance = rational(model["Rs"]["exact_rational"])
    capacitance = rational(model["C"]["exact_rational"])

    if min(source_voltage, source_resistance, capacitance) <= 0:
        raise AssertionError("source fixture parameters must be positive")

    critical_power = source_voltage**2 / (4 * source_resistance)
    cases = fixture["cases"]
    if not cases:
        raise AssertionError("fixture contains no cases")

    for case in cases:
        case_id = case["id"]
        load_power = rational(case["P"]["exact_rational"])
        expected_delta = source_voltage**2 - 4 * source_resistance * load_power
        observed_delta = rational(case["delta"]["exact_rational"])
        require_equal(observed_delta, expected_delta, f"{case_id}: discriminant")

        expected_margin = expected_delta / source_voltage**2
        observed_margin = rational(case["normalized_static_fold_margin"])
        require_equal(observed_margin, expected_margin, f"{case_id}: normalized fold margin")

        expected_loading = load_power / critical_power
        observed_loading = rational(case["power_loading_fraction"])
        require_equal(observed_loading, expected_loading, f"{case_id}: power loading fraction")

        roots = case["roots"]
        if expected_delta > 0:
            sqrt_delta = exact_sqrt(expected_delta)
            lower_voltage = (source_voltage - sqrt_delta) / 2
            upper_voltage = (source_voltage + sqrt_delta) / 2
            expected_roots = [
                (upper_voltage, "locally_asymptotically_stable"),
                (lower_voltage, "unstable"),
            ]
            if case["disposition"] != "two_equilibria":
                raise AssertionError(f"{case_id}: expected two-equilibria disposition")
            if len(roots) != 2:
                raise AssertionError(f"{case_id}: expected exactly two roots")

            for root, (voltage, classification) in zip(roots, expected_roots, strict=True):
                observed_voltage = Fraction(root["voltage"])
                observed_current = Fraction(root["current"])
                observed_eigenvalue = rational(root["eigenvalue"])
                expected_current = load_power / voltage
                expected_eigenvalue = (1 / capacitance) * (
                    -(1 / source_resistance) + load_power / voltage**2
                )
                require_equal(observed_voltage, voltage, f"{case_id}: root voltage")
                require_equal(observed_current, expected_current, f"{case_id}: root current")
                require_equal(
                    observed_eigenvalue, expected_eigenvalue, f"{case_id}: root eigenvalue"
                )
                if root["classification"] != classification:
                    raise AssertionError(
                        f"{case_id}: expected {classification}, "
                        f"observed {root['classification']}"
                    )
        elif expected_delta == 0:
            root_voltage = source_voltage / 2
            if case["disposition"] != "critical_fold_non_pass":
                raise AssertionError(f"{case_id}: fold must remain a non-PASS")
            if len(roots) != 1:
                raise AssertionError(f"{case_id}: expected one repeated root")
            root = roots[0]
            require_equal(Fraction(root["voltage"]), root_voltage, f"{case_id}: fold voltage")
            require_equal(Fraction(root["current"]), load_power / root_voltage, f"{case_id}: fold current")
            require_equal(rational(root["eigenvalue"]), Fraction(0), f"{case_id}: fold eigenvalue")
            expected_class = (
                "non_hyperbolic_semistable_from_above_repelling_from_below"
            )
            if root["classification"] != expected_class:
                raise AssertionError(f"{case_id}: repeated-root classification mismatch")
        else:
            if case["disposition"] != "no_equilibrium_non_pass":
                raise AssertionError(f"{case_id}: negative discriminant must be non-PASS")
            if roots:
                raise AssertionError(f"{case_id}: no-equilibrium case must have no roots")

        print(f"PASS {case_id}: exact rational reference agrees")

    print(f"PASS {len(cases)} exact-rational cases; fixture={path}")
    return len(cases)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_fixture = (
        repo_root
        / "docs"
        / "research"
        / "fixtures"
        / "airroad-dc-dynamic-stability-v0.1.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=default_fixture,
        help="fixture JSON path (default: repository frozen V0.1 fixture)",
    )
    args = parser.parse_args()
    verify_fixture(args.fixture)


if __name__ == "__main__":
    main()
