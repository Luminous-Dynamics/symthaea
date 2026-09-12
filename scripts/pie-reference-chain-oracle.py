#!/usr/bin/env python3
"""Independent PIE Phase-0 synthetic reference-chain audit harness.

This is deliberately synthetic. It composes conservative gate semantics into
one Moon-shaped reference campaign. It imports no Symthaea code and claims no
real Moon/Mars process performance.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    GUARANTEED = "Guaranteed"
    POSSIBLE = "Possible"
    IMPOSSIBLE = "Impossible"


class Closure(str, Enum):
    LOCAL = "LocallyClosed"
    IMPORT_DEPENDENT = "ImportDependent"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def validate(self, upper: float | None = None) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("interval endpoints must be finite")
        if self.lo < 0 or self.lo > self.hi:
            raise ValueError("invalid nonnegative interval")
        if upper is not None and self.hi > upper:
            raise ValueError("interval exceeds upper bound")


@dataclass(frozen=True)
class Scenario:
    occurrence_kg: Interval
    recovery_fraction: Interval
    transport_available: bool
    transport_capacity_kg: Interval
    delivery_fraction: Interval
    process_capacity_kg: Interval
    feed_required_kg: Interval
    output_masses_kg: tuple[float, ...]
    tracked_input_kg: float
    tracked_output_kg: float
    unknown_composition_fraction: float
    grade_min_fraction: float
    known_grade_fraction: float
    energy_required_j: Interval
    energy_available_j: Interval
    peak_power_required_w: Interval
    peak_power_available_w: Interval
    waste_kg: float
    recycle_fraction: float
    recycle_allocations_kg: tuple[float, ...]
    required_service_spares: int
    available_service_spares: int
    imported_blocker_available: bool
    local_alternative_for_blocker: bool


@dataclass(frozen=True)
class Audit:
    resource: str
    throughput: str
    mass: str
    constituent: str
    grade: str
    utilities: str
    circularity: str
    lifecycle: str
    closure: str
    overall: str


def interval_mul(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo * b.lo, a.hi * b.hi)


def conservative_fit(capacity: Interval, demand: Interval) -> Status:
    capacity.validate()
    demand.validate()
    if capacity.lo >= demand.hi:
        return Status.GUARANTEED
    if capacity.hi < demand.lo:
        return Status.IMPOSSIBLE
    return Status.POSSIBLE


def worst_status(statuses: list[Status]) -> Status:
    if Status.IMPOSSIBLE in statuses:
        return Status.IMPOSSIBLE
    if Status.POSSIBLE in statuses:
        return Status.POSSIBLE
    return Status.GUARANTEED


def audit(s: Scenario) -> Audit:
    s.occurrence_kg.validate()
    s.recovery_fraction.validate(1.0)
    recovered = interval_mul(s.occurrence_kg, s.recovery_fraction)

    if s.transport_available:
        s.transport_capacity_kg.validate()
        s.delivery_fraction.validate(1.0)
        shipped = Interval(
            min(recovered.lo, s.transport_capacity_kg.lo),
            min(recovered.hi, s.transport_capacity_kg.hi),
        )
        delivered = interval_mul(shipped, s.delivery_fraction)
    else:
        delivered = Interval(0.0, 0.0)
    resource = conservative_fit(delivered, s.feed_required_kg)

    throughput = conservative_fit(s.process_capacity_kg, s.feed_required_kg)

    total_out = sum(s.output_masses_kg)
    exact_feed = s.feed_required_kg.lo == s.feed_required_kg.hi
    if not exact_feed:
        mass = Status.POSSIBLE
    elif abs(total_out - s.feed_required_kg.lo) <= 1e-9:
        mass = Status.GUARANTEED
    else:
        mass = Status.IMPOSSIBLE

    constituent = (
        Status.GUARANTEED
        if abs(s.tracked_input_kg - s.tracked_output_kg) <= 1e-9
        else Status.IMPOSSIBLE
    )

    for value in (
        s.unknown_composition_fraction,
        s.grade_min_fraction,
        s.known_grade_fraction,
        s.recycle_fraction,
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("invalid fraction")

    if s.known_grade_fraction >= s.grade_min_fraction and s.unknown_composition_fraction == 0.0:
        grade = Status.GUARANTEED
    elif s.known_grade_fraction + s.unknown_composition_fraction < s.grade_min_fraction:
        grade = Status.IMPOSSIBLE
    else:
        grade = Status.POSSIBLE

    energy = conservative_fit(s.energy_available_j, s.energy_required_j)
    power = conservative_fit(s.peak_power_available_w, s.peak_power_required_w)
    utilities = worst_status([energy, power])

    if s.waste_kg < 0 or any(value < 0 for value in s.recycle_allocations_kg):
        raise ValueError("invalid recycle ledger")
    recovered_waste = s.waste_kg * s.recycle_fraction
    circularity = (
        Status.GUARANTEED
        if sum(s.recycle_allocations_kg) <= recovered_waste + 1e-9
        else Status.IMPOSSIBLE
    )

    if s.required_service_spares < 0 or s.available_service_spares < 0:
        raise ValueError("invalid spare count")
    lifecycle = (
        Status.GUARANTEED
        if s.available_service_spares >= s.required_service_spares
        else Status.IMPOSSIBLE
    )

    if s.local_alternative_for_blocker:
        closure = Closure.LOCAL
    elif s.imported_blocker_available:
        closure = Closure.IMPORT_DEPENDENT
    else:
        closure = Closure.UNAVAILABLE

    physical = worst_status(
        [resource, throughput, mass, constituent, grade, utilities, circularity, lifecycle]
    )
    overall = Status.IMPOSSIBLE if closure == Closure.UNAVAILABLE else physical

    return Audit(
        resource.value,
        throughput.value,
        mass.value,
        constituent.value,
        grade.value,
        utilities.value,
        circularity.value,
        lifecycle.value,
        closure.value,
        overall.value,
    )


def base_scenario() -> Scenario:
    return Scenario(
        occurrence_kg=Interval(120.0, 120.0),
        recovery_fraction=Interval(0.8, 0.8),
        transport_available=True,
        transport_capacity_kg=Interval(100.0, 100.0),
        delivery_fraction=Interval(0.95, 0.95),
        process_capacity_kg=Interval(100.0, 100.0),
        feed_required_kg=Interval(90.0, 90.0),
        output_masses_kg=(20.0, 60.0, 10.0),
        tracked_input_kg=30.0,
        tracked_output_kg=30.0,
        unknown_composition_fraction=0.0,
        grade_min_fraction=0.95,
        known_grade_fraction=0.98,
        energy_required_j=Interval(1000.0, 1000.0),
        energy_available_j=Interval(1200.0, 1200.0),
        peak_power_required_w=Interval(100.0, 100.0),
        peak_power_available_w=Interval(150.0, 150.0),
        waste_kg=10.0,
        recycle_fraction=0.5,
        recycle_allocations_kg=(5.0,),
        required_service_spares=1,
        available_service_spares=1,
        imported_blocker_available=True,
        local_alternative_for_blocker=False,
    )


def self_test() -> None:
    base = base_scenario()
    result = audit(base)
    assert result.overall == Status.GUARANTEED.value
    assert result.closure == Closure.IMPORT_DEPENDENT.value

    changed = {**base.__dict__, "transport_available": False}
    assert audit(Scenario(**changed)).overall == Status.IMPOSSIBLE.value

    changed = {**base.__dict__, "tracked_output_kg": 29.0}
    assert audit(Scenario(**changed)).constituent == Status.IMPOSSIBLE.value

    changed = {
        **base.__dict__,
        "unknown_composition_fraction": 0.05,
        "known_grade_fraction": 0.93,
    }
    uncertain_grade = audit(Scenario(**changed))
    assert uncertain_grade.grade == Status.POSSIBLE.value
    assert uncertain_grade.overall == Status.POSSIBLE.value

    changed = {**base.__dict__, "recycle_allocations_kg": (5.0, 1.0)}
    assert audit(Scenario(**changed)).circularity == Status.IMPOSSIBLE.value

    changed = {**base.__dict__, "available_service_spares": 0}
    assert audit(Scenario(**changed)).lifecycle == Status.IMPOSSIBLE.value

    changed = {**base.__dict__, "imported_blocker_available": False}
    unavailable = audit(Scenario(**changed))
    assert unavailable.closure == Closure.UNAVAILABLE.value
    assert unavailable.overall == Status.IMPOSSIBLE.value

    changed["local_alternative_for_blocker"] = True
    local = audit(Scenario(**changed))
    assert local.closure == Closure.LOCAL.value
    assert local.overall == Status.GUARANTEED.value

    changed = {**base.__dict__, "peak_power_available_w": Interval(80.0, 80.0)}
    assert audit(Scenario(**changed)).utilities == Status.IMPOSSIBLE.value

    changed = {**base.__dict__, "recovery_fraction": Interval(0.5, 0.8)}
    widened = audit(Scenario(**changed))
    assert widened.resource == Status.POSSIBLE.value
    assert widened.overall == Status.POSSIBLE.value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    parser.error("--self-test is required in the reference oracle")


if __name__ == "__main__":
    main()
