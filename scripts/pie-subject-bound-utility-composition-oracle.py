#!/usr/bin/env python3
"""Independent PIE-002E subject-bound process-to-accounting composition oracle.

This standard-library-only reference proves one narrow property: the preferred
production composition path takes the current process record plus explicit
supply/recovery context, validates the process, recomputes its utility projection
inside the call, and immediately binds that derived projection. A detached or
cached projection is never accepted as an input to the subject-bound entry point.

Composition is not feasibility evaluation.
"""

from __future__ import annotations

import argparse
import inspect
import math
from dataclasses import dataclass, replace
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


class UtilityKind(str, Enum):
    ELECTRICAL_ENERGY = "ElectricalEnergy"
    PEAK_POWER = "PeakElectricalPower"
    PROCESS_TIME = "ProcessTime"
    THERMAL_ENERGY = "ThermalEnergy"
    COOLING_ENERGY = "CoolingEnergy"


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
class UtilityDeclaration:
    kind: UtilityKind
    value: Range


@dataclass(frozen=True)
class ProcessDefinition:
    process_id: str
    has_inputs: bool
    has_outputs: bool
    utilities: tuple[UtilityDeclaration, ...]

    def validate(self) -> None:
        if not self.process_id or not self.process_id.strip():
            raise ValueError("process_id: required")
        if not self.has_inputs:
            raise ValueError("process: missing inputs")
        if not self.has_outputs:
            raise ValueError("process: missing outputs")
        for utility in self.utilities:
            if not isinstance(utility.kind, UtilityKind):
                raise ValueError("utility: unknown kind")
            utility.value.validate("utility")


@dataclass(frozen=True)
class ElectricalProjection:
    process_id: str
    status: ProjectionStatus
    electrical_energy_j: Range | None
    peak_power_w: Range | None
    process_time_s: Range | None
    unresolved: tuple[ProjectionReason, ...]


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
        self.recovery_duration_s.validate("recovery_duration_s", strictly_positive=True)
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


def _sum_ranges(values: tuple[Range, ...], label: str) -> Range | None:
    if not values:
        return None
    lo = 0.0
    hi = 0.0
    for value in values:
        value.validate(label)
        lo += value.min
        hi += value.max
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"{label}: aggregate overflow")
    return Range(lo, hi)


def project_process_utilities(process: ProcessDefinition) -> ElectricalProjection:
    electrical = tuple(
        item.value for item in process.utilities if item.kind is UtilityKind.ELECTRICAL_ENERGY
    )
    peaks = tuple(
        item.value for item in process.utilities if item.kind is UtilityKind.PEAK_POWER
    )
    times = tuple(
        item.value for item in process.utilities if item.kind is UtilityKind.PROCESS_TIME
    )
    thermal = tuple(
        item.value for item in process.utilities if item.kind is UtilityKind.THERMAL_ENERGY
    )
    cooling = tuple(
        item.value for item in process.utilities if item.kind is UtilityKind.COOLING_ENERGY
    )

    electrical_energy = _sum_ranges(electrical, "electrical_energy_j")
    for value in peaks:
        value.validate("peak_power_w")
    for value in times:
        value.validate("process_time_s", strictly_positive=True)
    for value in thermal:
        value.validate("thermal_energy_j")
    for value in cooling:
        value.validate("cooling_energy_j")

    unresolved: list[ProjectionReason] = []
    if electrical_energy is None:
        unresolved.append(ProjectionReason.MISSING_ELECTRICAL_ENERGY)
    if len(peaks) == 0:
        unresolved.append(ProjectionReason.MISSING_PEAK_POWER)
    elif len(peaks) > 1:
        unresolved.append(ProjectionReason.MULTIPLE_PEAK_POWER)
    if len(times) == 0:
        unresolved.append(ProjectionReason.MISSING_PROCESS_TIME)
    elif len(times) > 1:
        unresolved.append(ProjectionReason.MULTIPLE_PROCESS_TIME)
    if thermal:
        unresolved.append(ProjectionReason.THERMAL_TEMPERATURE_UNBOUND)
    if cooling:
        unresolved.append(ProjectionReason.COOLING_REJECTION_UNBOUND)

    if len(peaks) > 1 or len(times) > 1:
        status = ProjectionStatus.AMBIGUOUS
    elif electrical_energy is None or len(peaks) != 1 or len(times) != 1:
        status = ProjectionStatus.INCOMPLETE
    else:
        status = ProjectionStatus.COMPLETE

    return ElectricalProjection(
        process_id=process.process_id,
        status=status,
        electrical_energy_j=electrical_energy,
        peak_power_w=peaks[0] if len(peaks) == 1 else None,
        process_time_s=times[0] if len(times) == 1 else None,
        unresolved=tuple(unresolved),
    )


def bind_accounting_case(
    projection: ElectricalProjection,
    context: SupplyRecoveryContext,
) -> BoundElectricalAccountingCase:
    if projection.status is not ProjectionStatus.COMPLETE:
        raise ValueError("projection: electrical basis is not Complete")
    if projection.electrical_energy_j is None:
        raise ValueError("projection: missing electrical energy")
    if projection.peak_power_w is None:
        raise ValueError("projection: missing peak power")
    if projection.process_time_s is None:
        raise ValueError("projection: missing process time")

    projection.electrical_energy_j.validate("electrical_energy_j")
    projection.peak_power_w.validate("peak_power_w")
    projection.process_time_s.validate("process_time_s", strictly_positive=True)

    blocking = {
        ProjectionReason.MISSING_ELECTRICAL_ENERGY,
        ProjectionReason.MISSING_PEAK_POWER,
        ProjectionReason.MULTIPLE_PEAK_POWER,
        ProjectionReason.MISSING_PROCESS_TIME,
        ProjectionReason.MULTIPLE_PROCESS_TIME,
    }
    if any(reason in blocking for reason in projection.unresolved):
        raise ValueError("projection: Complete basis carries electrical blocking reason")
    if len(set(projection.unresolved)) != len(projection.unresolved):
        raise ValueError("projection: duplicate unresolved reason")

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


def compose_current_process_accounting_case(
    process: ProcessDefinition,
    context: SupplyRecoveryContext,
) -> BoundElectricalAccountingCase:
    """Validate the exact current process, recompute its projection, then bind it."""

    process.validate()
    projection = project_process_utilities(process)
    return bind_accounting_case(projection, context)


def _r(lo: float, hi: float | None = None) -> Range:
    return Range(lo, lo if hi is None else hi)


def _u(kind: UtilityKind, lo: float, hi: float | None = None) -> UtilityDeclaration:
    return UtilityDeclaration(kind, _r(lo, hi))


def baseline_process() -> ProcessDefinition:
    return ProcessDefinition(
        process_id="p1",
        has_inputs=True,
        has_outputs=True,
        utilities=(
            _u(UtilityKind.ELECTRICAL_ENERGY, 90.0, 110.0),
            _u(UtilityKind.PEAK_POWER, 20.0, 25.0),
            _u(UtilityKind.PROCESS_TIME, 9.0, 11.0),
        ),
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


def _must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(label)


def self_test() -> None:
    process = baseline_process()
    context = baseline_context()
    bound = compose_current_process_accounting_case(process, context)
    assert bound.process_id == process.process_id
    assert bound.gross_energy_j == _r(90.0, 110.0)
    assert bound.peak_power_w == _r(20.0, 25.0)
    assert bound.batch_duration_s == _r(9.0, 11.0)
    assert bound.recoverable_energy_j == context.recoverable_energy_j
    assert bound.available_sustained_power_w == context.available_sustained_power_w

    # The preferred subject-bound function exposes no detached projection input.
    assert tuple(inspect.signature(compose_current_process_accounting_case).parameters) == (
        "process",
        "context",
    )

    invalid_structure = replace(process, has_inputs=False)
    _must_fail(
        lambda: compose_current_process_accounting_case(invalid_structure, context),
        "structurally invalid process must fail before binding",
    )

    # Capture a valid old projection, then mutate the current process demand.
    # The direct path must derive 140..160 J from the changed record, not reuse
    # the stale 90..110 J projection.
    stale_projection = project_process_utilities(process)
    changed = replace(
        process,
        utilities=(
            _u(UtilityKind.ELECTRICAL_ENERGY, 140.0, 160.0),
            _u(UtilityKind.PEAK_POWER, 30.0, 35.0),
            _u(UtilityKind.PROCESS_TIME, 12.0, 14.0),
        ),
    )
    rebound = compose_current_process_accounting_case(changed, context)
    assert stale_projection.electrical_energy_j == _r(90.0, 110.0)
    assert rebound.gross_energy_j == _r(140.0, 160.0)
    assert rebound.peak_power_w == _r(30.0, 35.0)
    assert rebound.batch_duration_s == _r(12.0, 14.0)

    incomplete = replace(
        process,
        utilities=(
            _u(UtilityKind.ELECTRICAL_ENERGY, 90.0, 110.0),
            _u(UtilityKind.PEAK_POWER, 20.0, 25.0),
        ),
    )
    _must_fail(
        lambda: compose_current_process_accounting_case(incomplete, context),
        "incomplete process utility basis must fail closed",
    )

    ambiguous_peak = replace(
        process,
        utilities=process.utilities + (_u(UtilityKind.PEAK_POWER, 1.0),),
    )
    _must_fail(
        lambda: compose_current_process_accounting_case(ambiguous_peak, context),
        "duplicate peak power must remain ambiguous",
    )

    ambiguous_time = replace(
        process,
        utilities=process.utilities + (_u(UtilityKind.PROCESS_TIME, 1.0),),
    )
    _must_fail(
        lambda: compose_current_process_accounting_case(ambiguous_time, context),
        "duplicate process time must remain ambiguous",
    )

    heat = replace(
        process,
        utilities=process.utilities
        + (
            _u(UtilityKind.THERMAL_ENERGY, 50.0, 60.0),
            _u(UtilityKind.COOLING_ENERGY, 10.0, 20.0),
        ),
    )
    heat_bound = compose_current_process_accounting_case(heat, context)
    assert heat_bound.unresolved_non_electrical == (
        ProjectionReason.THERMAL_TEMPERATURE_UNBOUND,
        ProjectionReason.COOLING_REJECTION_UNBOUND,
    )

    # Identity comes only from the exact current process argument.
    renamed = replace(process, process_id="p2")
    renamed_bound = compose_current_process_accounting_case(renamed, context)
    assert renamed_bound.process_id == "p2"

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
