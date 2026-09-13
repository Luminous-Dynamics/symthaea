#!/usr/bin/env python3
"""Independent PIE-002D projection-to-accounting binding oracle.

This standard-library-only reference proves only that a complete electrical-demand
projection plus an explicit supply/recovery context can be bound into one accounting
case without inventing defaults. It does not evaluate utility feasibility.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class ProjectionStatus(str, Enum):
    COMPLETE = "Complete"
    INCOMPLETE = "Incomplete"
    AMBIGUOUS = "Ambiguous"


class ProjectionReason(str, Enum):
    MISSING_ELECTRICAL_ENERGY = "MissingElectricalEnergy"
    MISSING_PEAK_POWER = "MissingPeakPower"
    MULTIPLE_PEAK_POWER = "MultiplePeakPower"
    MISSING_PROCESS_TIME = "MissingProcessTime"
    MULTIPLE_PROCESS_TIME = "MultipleProcessTime"
    THERMAL_TEMPERATURE_UNBOUND = "ThermalTemperatureUnbound"
    COOLING_REJECTION_UNBOUND = "CoolingRejectionUnbound"


ELECTRICAL_BLOCKING_REASONS = frozenset(
    {
        ProjectionReason.MISSING_ELECTRICAL_ENERGY,
        ProjectionReason.MISSING_PEAK_POWER,
        ProjectionReason.MULTIPLE_PEAK_POWER,
        ProjectionReason.MISSING_PROCESS_TIME,
        ProjectionReason.MULTIPLE_PROCESS_TIME,
    }
)


@dataclass(frozen=True)
class Range:
    min: float
    max: float

    def validate(self, label: str, *, strictly_positive: bool = False) -> None:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError(f"{label}: non-finite range")
        if self.min < 0.0 or self.max < 0.0 or self.min > self.max:
            raise ValueError(f"{label}: invalid range")
        if strictly_positive and self.min <= 0.0:
            raise ValueError(f"{label}: lower bound must be strictly positive")


@dataclass(frozen=True)
class FractionRange:
    min: float
    max: float

    def validate(self, label: str) -> None:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError(f"{label}: non-finite range")
        if self.min < 0.0 or self.max > 1.0 or self.min > self.max:
            raise ValueError(f"{label}: expected 0 <= min <= max <= 1")


@dataclass(frozen=True)
class ElectricalProjection:
    process_id: str
    status: ProjectionStatus
    electrical_energy_j: Range | None
    peak_power_w: Range | None
    process_time_s: Range | None
    unresolved: tuple[ProjectionReason, ...] = ()


@dataclass(frozen=True)
class SupplyRecoveryContext:
    recoverable_energy_j: Range
    recovery_duration_s: Range
    storage_acceptance_j: Range
    storage_charge_power_w: Range
    storage_discharge_power_w: Range
    recovery_delivery_fraction: FractionRange
    available_energy_capacity_j: Range
    available_sustained_power_w: Range
    available_peak_power_w: Range

    def validate(self) -> None:
        self.recoverable_energy_j.validate("recoverable_energy_j")
        self.recovery_duration_s.validate(
            "recovery_duration_s", strictly_positive=True
        )
        self.storage_acceptance_j.validate("storage_acceptance_j")
        self.storage_charge_power_w.validate("storage_charge_power_w")
        self.storage_discharge_power_w.validate("storage_discharge_power_w")
        self.recovery_delivery_fraction.validate("recovery_delivery_fraction")
        self.available_energy_capacity_j.validate("available_energy_capacity_j")
        self.available_sustained_power_w.validate("available_sustained_power_w")
        self.available_peak_power_w.validate("available_peak_power_w")


@dataclass(frozen=True)
class BoundElectricalAccountingCase:
    process_id: str
    gross_energy_j: Range
    batch_duration_s: Range
    peak_power_w: Range
    recoverable_energy_j: Range
    recovery_duration_s: Range
    storage_acceptance_j: Range
    storage_charge_power_w: Range
    storage_discharge_power_w: Range
    recovery_delivery_fraction: FractionRange
    available_energy_capacity_j: Range
    available_sustained_power_w: Range
    available_peak_power_w: Range
    unresolved_non_electrical: tuple[ProjectionReason, ...]


def bind_accounting_case(
    projection: ElectricalProjection,
    context: SupplyRecoveryContext,
) -> BoundElectricalAccountingCase:
    if not projection.process_id or not projection.process_id.strip():
        raise ValueError("process_id: required")
    if not isinstance(projection.status, ProjectionStatus):
        raise ValueError("projection status: unknown")
    if projection.status is not ProjectionStatus.COMPLETE:
        raise ValueError("projection: electrical basis is not Complete")

    if projection.electrical_energy_j is None:
        raise ValueError("projection: Complete basis is missing electrical energy")
    if projection.peak_power_w is None:
        raise ValueError("projection: Complete basis is missing peak power")
    if projection.process_time_s is None:
        raise ValueError("projection: Complete basis is missing process time")

    projection.electrical_energy_j.validate("electrical_energy_j")
    projection.peak_power_w.validate("peak_power_w")
    projection.process_time_s.validate("process_time_s", strictly_positive=True)

    if len(set(projection.unresolved)) != len(projection.unresolved):
        raise ValueError("projection: duplicate unresolved reason")
    for reason in projection.unresolved:
        if not isinstance(reason, ProjectionReason):
            raise ValueError("projection: unknown unresolved reason")
        if reason in ELECTRICAL_BLOCKING_REASONS:
            raise ValueError("projection: Complete basis carries electrical blocking reason")

    context.validate()

    return BoundElectricalAccountingCase(
        process_id=projection.process_id,
        gross_energy_j=projection.electrical_energy_j,
        batch_duration_s=projection.process_time_s,
        peak_power_w=projection.peak_power_w,
        recoverable_energy_j=context.recoverable_energy_j,
        recovery_duration_s=context.recovery_duration_s,
        storage_acceptance_j=context.storage_acceptance_j,
        storage_charge_power_w=context.storage_charge_power_w,
        storage_discharge_power_w=context.storage_discharge_power_w,
        recovery_delivery_fraction=context.recovery_delivery_fraction,
        available_energy_capacity_j=context.available_energy_capacity_j,
        available_sustained_power_w=context.available_sustained_power_w,
        available_peak_power_w=context.available_peak_power_w,
        unresolved_non_electrical=projection.unresolved,
    )


def _r(lo: float, hi: float | None = None) -> Range:
    return Range(lo, lo if hi is None else hi)


def baseline_projection() -> ElectricalProjection:
    return ElectricalProjection(
        process_id="p1",
        status=ProjectionStatus.COMPLETE,
        electrical_energy_j=_r(90.0, 110.0),
        peak_power_w=_r(20.0, 25.0),
        process_time_s=_r(9.0, 11.0),
    )


def baseline_context() -> SupplyRecoveryContext:
    return SupplyRecoveryContext(
        recoverable_energy_j=_r(30.0, 40.0),
        recovery_duration_s=_r(4.0, 5.0),
        storage_acceptance_j=_r(50.0, 60.0),
        storage_charge_power_w=_r(10.0, 12.0),
        storage_discharge_power_w=_r(8.0, 9.0),
        recovery_delivery_fraction=FractionRange(0.8, 0.9),
        available_energy_capacity_j=_r(120.0, 140.0),
        available_sustained_power_w=_r(15.0, 20.0),
        available_peak_power_w=_r(30.0, 35.0),
    )


def self_test() -> None:
    projection = baseline_projection()
    context = baseline_context()
    bound = bind_accounting_case(projection, context)
    assert bound.process_id == "p1"
    assert bound.gross_energy_j == projection.electrical_energy_j
    assert bound.batch_duration_s == projection.process_time_s
    assert bound.peak_power_w == projection.peak_power_w
    assert bound.recoverable_energy_j == context.recoverable_energy_j
    assert bound.recovery_delivery_fraction == context.recovery_delivery_fraction
    assert bound.available_sustained_power_w == context.available_sustained_power_w

    for status in (ProjectionStatus.INCOMPLETE, ProjectionStatus.AMBIGUOUS):
        bad = ElectricalProjection(
            process_id="p1",
            status=status,
            electrical_energy_j=_r(100.0),
            peak_power_w=_r(20.0),
            process_time_s=_r(10.0),
        )
        try:
            bind_accounting_case(bad, context)
        except ValueError:
            pass
        else:
            raise AssertionError("non-complete projection must not bind")

    forged_missing_peak = ElectricalProjection(
        process_id="p1",
        status=ProjectionStatus.COMPLETE,
        electrical_energy_j=_r(100.0),
        peak_power_w=None,
        process_time_s=_r(10.0),
    )
    try:
        bind_accounting_case(forged_missing_peak, context)
    except ValueError:
        pass
    else:
        raise AssertionError("forged Complete projection missing peak must fail")

    forged_blocking_reason = ElectricalProjection(
        process_id="p1",
        status=ProjectionStatus.COMPLETE,
        electrical_energy_j=_r(100.0),
        peak_power_w=_r(20.0),
        process_time_s=_r(10.0),
        unresolved=(ProjectionReason.MULTIPLE_PEAK_POWER,),
    )
    try:
        bind_accounting_case(forged_blocking_reason, context)
    except ValueError:
        pass
    else:
        raise AssertionError("Complete projection with blocking reason must fail")

    unresolved_heat = ElectricalProjection(
        process_id="p1",
        status=ProjectionStatus.COMPLETE,
        electrical_energy_j=_r(100.0),
        peak_power_w=_r(20.0),
        process_time_s=_r(10.0),
        unresolved=(
            ProjectionReason.THERMAL_TEMPERATURE_UNBOUND,
            ProjectionReason.COOLING_REJECTION_UNBOUND,
        ),
    )
    heat_bound = bind_accounting_case(unresolved_heat, context)
    assert heat_bound.unresolved_non_electrical == unresolved_heat.unresolved

    zero_recovery_window = SupplyRecoveryContext(
        recoverable_energy_j=context.recoverable_energy_j,
        recovery_duration_s=_r(0.0, 1.0),
        storage_acceptance_j=context.storage_acceptance_j,
        storage_charge_power_w=context.storage_charge_power_w,
        storage_discharge_power_w=context.storage_discharge_power_w,
        recovery_delivery_fraction=context.recovery_delivery_fraction,
        available_energy_capacity_j=context.available_energy_capacity_j,
        available_sustained_power_w=context.available_sustained_power_w,
        available_peak_power_w=context.available_peak_power_w,
    )
    try:
        bind_accounting_case(projection, zero_recovery_window)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-inclusive recovery duration must fail")

    bad_fraction = SupplyRecoveryContext(
        recoverable_energy_j=context.recoverable_energy_j,
        recovery_duration_s=context.recovery_duration_s,
        storage_acceptance_j=context.storage_acceptance_j,
        storage_charge_power_w=context.storage_charge_power_w,
        storage_discharge_power_w=context.storage_discharge_power_w,
        recovery_delivery_fraction=FractionRange(0.9, 1.1),
        available_energy_capacity_j=context.available_energy_capacity_j,
        available_sustained_power_w=context.available_sustained_power_w,
        available_peak_power_w=context.available_peak_power_w,
    )
    try:
        bind_accounting_case(projection, bad_fraction)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid delivery fraction must fail")

    duplicate_reason = ElectricalProjection(
        process_id="p1",
        status=ProjectionStatus.COMPLETE,
        electrical_energy_j=_r(100.0),
        peak_power_w=_r(20.0),
        process_time_s=_r(10.0),
        unresolved=(
            ProjectionReason.THERMAL_TEMPERATURE_UNBOUND,
            ProjectionReason.THERMAL_TEMPERATURE_UNBOUND,
        ),
    )
    try:
        bind_accounting_case(duplicate_reason, context)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate unresolved reasons must fail")

    print("ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("only --self-test is supported")


if __name__ == "__main__":
    main()
