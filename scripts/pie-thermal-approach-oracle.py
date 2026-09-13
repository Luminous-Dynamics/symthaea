#!/usr/bin/env python3
"""Independent PIE-002C thermal approach-temperature screening oracle.

This standard-library-only reference freezes conservative temperature-headroom
semantics. It does not import Symthaea and grants no thermodynamic, equipment,
process, or control authority.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class ApproachStatus(str, Enum):
    GUARANTEED = "Guaranteed"
    POSSIBLE = "Possible"
    IMPOSSIBLE = "Impossible"


class ApproachBasis(str, Enum):
    ALGEBRAIC_ENVELOPE = "AlgebraicEnvelope"
    FINITE_DRIVING_FORCE = "FiniteDrivingForce"


@dataclass(frozen=True)
class Range:
    min: float
    max: float

    def validate(
        self,
        label: str,
        *,
        strictly_positive: bool = False,
        nonnegative: bool = True,
    ) -> None:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError(f"{label}: non-finite range")
        if self.min > self.max:
            raise ValueError(f"{label}: reversed range")
        if nonnegative and (self.min < 0.0 or self.max < 0.0):
            raise ValueError(f"{label}: negative range")
        if strictly_positive and self.min <= 0.0:
            raise ValueError(f"{label}: lower bound must be strictly positive")


@dataclass(frozen=True)
class ThermalApproachCase:
    source_temperature_k: Range
    required_temperature_k: Range
    minimum_approach_k: Range
    basis: ApproachBasis


@dataclass(frozen=True)
class ThermalApproachReport:
    status: ApproachStatus
    headroom_k: Range
    required_source_temperature_k: Range
    basis: ApproachBasis
    supports_finite_driving_force_claim: bool


def _finite_sum(a: float, b: float, label: str) -> float:
    value = a + b
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite derived value")
    return value


def _finite_difference3(a: float, b: float, c: float, label: str) -> float:
    value = a - b - c
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite derived value")
    return value


def screen_thermal_approach(case: ThermalApproachCase) -> ThermalApproachReport:
    if not isinstance(case.basis, ApproachBasis):
        raise ValueError("basis: unknown")

    case.source_temperature_k.validate(
        "source_temperature_k", strictly_positive=True
    )
    case.required_temperature_k.validate(
        "required_temperature_k", strictly_positive=True
    )
    case.minimum_approach_k.validate("minimum_approach_k")

    if (
        case.basis is ApproachBasis.FINITE_DRIVING_FORCE
        and case.minimum_approach_k.min <= 0.0
    ):
        raise ValueError(
            "minimum_approach_k: finite-driving-force basis requires positive lower bound"
        )

    required_source = Range(
        _finite_sum(
            case.required_temperature_k.min,
            case.minimum_approach_k.min,
            "required_source_temperature_k",
        ),
        _finite_sum(
            case.required_temperature_k.max,
            case.minimum_approach_k.max,
            "required_source_temperature_k",
        ),
    )
    required_source.validate(
        "required_source_temperature_k", strictly_positive=True
    )

    # Conservative interval arithmetic for
    # source - required_process_temperature - required_approach.
    headroom = Range(
        _finite_difference3(
            case.source_temperature_k.min,
            case.required_temperature_k.max,
            case.minimum_approach_k.max,
            "thermal_headroom_k",
        ),
        _finite_difference3(
            case.source_temperature_k.max,
            case.required_temperature_k.min,
            case.minimum_approach_k.min,
            "thermal_headroom_k",
        ),
    )
    headroom.validate("thermal_headroom_k", nonnegative=False)

    if headroom.min >= 0.0:
        status = ApproachStatus.GUARANTEED
    elif headroom.max < 0.0:
        status = ApproachStatus.IMPOSSIBLE
    else:
        status = ApproachStatus.POSSIBLE

    return ThermalApproachReport(
        status=status,
        headroom_k=headroom,
        required_source_temperature_k=required_source,
        basis=case.basis,
        supports_finite_driving_force_claim=(
            case.basis is ApproachBasis.FINITE_DRIVING_FORCE
            and case.minimum_approach_k.min > 0.0
        ),
    )


def _r(lo: float, hi: float | None = None) -> Range:
    return Range(lo, lo if hi is None else hi)


def self_test() -> None:
    guaranteed = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(520.0),
            required_temperature_k=_r(500.0),
            minimum_approach_k=_r(10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert guaranteed.status is ApproachStatus.GUARANTEED
    assert guaranteed.headroom_k == _r(10.0)
    assert guaranteed.required_source_temperature_k == _r(510.0)
    assert guaranteed.supports_finite_driving_force_claim

    equality_is_not_transfer_with_positive_approach = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(500.0),
            required_temperature_k=_r(500.0),
            minimum_approach_k=_r(10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert equality_is_not_transfer_with_positive_approach.status is ApproachStatus.IMPOSSIBLE
    assert equality_is_not_transfer_with_positive_approach.headroom_k == _r(-10.0)

    algebraic_equality = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(500.0),
            required_temperature_k=_r(500.0),
            minimum_approach_k=_r(0.0),
            basis=ApproachBasis.ALGEBRAIC_ENVELOPE,
        )
    )
    assert algebraic_equality.status is ApproachStatus.GUARANTEED
    assert not algebraic_equality.supports_finite_driving_force_claim

    overlap = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(505.0, 520.0),
            required_temperature_k=_r(500.0, 510.0),
            minimum_approach_k=_r(5.0, 10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert overlap.status is ApproachStatus.POSSIBLE
    assert overlap.headroom_k == _r(-15.0, 15.0)

    narrow = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(520.0),
            required_temperature_k=_r(500.0),
            minimum_approach_k=_r(10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    widened = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(505.0, 525.0),
            required_temperature_k=_r(495.0, 510.0),
            minimum_approach_k=_r(5.0, 15.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert narrow.status is ApproachStatus.GUARANTEED
    assert widened.status is ApproachStatus.POSSIBLE

    impossible = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(450.0, 480.0),
            required_temperature_k=_r(500.0, 510.0),
            minimum_approach_k=_r(5.0, 10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert impossible.status is ApproachStatus.IMPOSSIBLE
    assert impossible.headroom_k.max < 0.0

    boundary_possible = screen_thermal_approach(
        ThermalApproachCase(
            source_temperature_k=_r(509.0, 510.0),
            required_temperature_k=_r(500.0),
            minimum_approach_k=_r(10.0),
            basis=ApproachBasis.FINITE_DRIVING_FORCE,
        )
    )
    assert boundary_possible.status is ApproachStatus.POSSIBLE
    assert boundary_possible.headroom_k.max == 0.0

    malformed = [
        ThermalApproachCase(
            _r(-1.0), _r(500.0), _r(10.0), ApproachBasis.FINITE_DRIVING_FORCE
        ),
        ThermalApproachCase(
            _r(500.0), Range(510.0, 500.0), _r(10.0), ApproachBasis.FINITE_DRIVING_FORCE
        ),
        ThermalApproachCase(
            _r(500.0), _r(500.0), _r(-1.0), ApproachBasis.FINITE_DRIVING_FORCE
        ),
        ThermalApproachCase(
            _r(500.0), _r(500.0), _r(0.0), ApproachBasis.FINITE_DRIVING_FORCE
        ),
        ThermalApproachCase(
            _r(float("nan")), _r(500.0), _r(10.0), ApproachBasis.FINITE_DRIVING_FORCE
        ),
    ]
    for case in malformed:
        try:
            screen_thermal_approach(case)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed case did not fail closed: {case!r}")

    try:
        screen_thermal_approach(
            ThermalApproachCase(
                source_temperature_k=_r(1.0),
                required_temperature_k=_r(float.fromhex("0x1.fffffffffffffp+1023")),
                minimum_approach_k=_r(float.fromhex("0x1.fffffffffffffp+1023")),
                basis=ApproachBasis.FINITE_DRIVING_FORCE,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("derived overflow did not fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    parser.error("only --self-test is supported")


if __name__ == "__main__":
    main()
