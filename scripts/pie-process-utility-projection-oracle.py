#!/usr/bin/env python3
"""Independent PIE-002B process-basis utility projection oracle.

This standard-library-only reference freezes fail-closed projection semantics from
neutral process utility declarations into an electrical accounting basis. It does
not import Symthaea and grants no process, equipment, thermal, or control authority.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class ProjectionStatus(str, Enum):
    COMPLETE = "Complete"
    INCOMPLETE = "Incomplete"
    AMBIGUOUS = "Ambiguous"


class UnresolvedReason(str, Enum):
    MISSING_ELECTRICAL_ENERGY = "MissingElectricalEnergy"
    MISSING_PEAK_POWER = "MissingPeakPower"
    MULTIPLE_PEAK_POWER = "MultiplePeakPower"
    MISSING_PROCESS_TIME = "MissingProcessTime"
    MULTIPLE_PROCESS_TIME = "MultipleProcessTime"
    THERMAL_TEMPERATURE_UNBOUND = "ThermalTemperatureUnbound"
    COOLING_REJECTION_UNBOUND = "CoolingRejectionUnbound"


class UtilityKind(str, Enum):
    ELECTRICAL_ENERGY = "ElectricalEnergy"
    THERMAL_ENERGY = "ThermalEnergy"
    COOLING_ENERGY = "CoolingEnergy"
    PEAK_ELECTRICAL_POWER = "PeakElectricalPower"
    PROCESS_TIME = "ProcessTime"


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
            raise ValueError(f"{label}: range must be strictly positive")


@dataclass(frozen=True)
class Utility:
    kind: UtilityKind
    value: Range


@dataclass(frozen=True)
class ProcessUtilityProjection:
    process_id: str
    electrical_status: ProjectionStatus
    electrical_energy_j: Range | None
    peak_electrical_power_w: Range | None
    process_time_s: Range | None
    thermal_energy_j: Range | None
    cooling_energy_j: Range | None
    unresolved: tuple[UnresolvedReason, ...]


def _checked_sum(ranges: Iterable[Range], label: str) -> Range | None:
    items = list(ranges)
    if not items:
        return None
    lo = 0.0
    hi = 0.0
    for item in items:
        item.validate(label)
        lo += item.min
        hi += item.max
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError(f"{label}: aggregate overflow")
    return Range(lo, hi)


def _single(entries: Sequence[Utility], kind: UtilityKind) -> Range | None:
    matches = [entry.value for entry in entries if entry.kind is kind]
    if len(matches) == 1:
        return matches[0]
    return None


def project_process_utilities(process_id: str, utilities: Sequence[Utility]) -> ProcessUtilityProjection:
    if not process_id or not process_id.strip():
        raise ValueError("process_id: required")

    for entry in utilities:
        if not isinstance(entry.kind, UtilityKind):
            raise ValueError("utility kind: unknown")
        entry.value.validate(
            entry.kind.value,
            strictly_positive=entry.kind is UtilityKind.PROCESS_TIME,
        )

    electrical_entries = [u for u in utilities if u.kind is UtilityKind.ELECTRICAL_ENERGY]
    thermal_entries = [u for u in utilities if u.kind is UtilityKind.THERMAL_ENERGY]
    cooling_entries = [u for u in utilities if u.kind is UtilityKind.COOLING_ENERGY]
    peak_entries = [u for u in utilities if u.kind is UtilityKind.PEAK_ELECTRICAL_POWER]
    time_entries = [u for u in utilities if u.kind is UtilityKind.PROCESS_TIME]

    electrical_energy = _checked_sum((u.value for u in electrical_entries), "electrical_energy_j")
    thermal_energy = _checked_sum((u.value for u in thermal_entries), "thermal_energy_j")
    cooling_energy = _checked_sum((u.value for u in cooling_entries), "cooling_energy_j")

    reasons: list[UnresolvedReason] = []
    ambiguous = False

    if not electrical_entries:
        reasons.append(UnresolvedReason.MISSING_ELECTRICAL_ENERGY)

    if len(peak_entries) == 0:
        reasons.append(UnresolvedReason.MISSING_PEAK_POWER)
    elif len(peak_entries) > 1:
        reasons.append(UnresolvedReason.MULTIPLE_PEAK_POWER)
        ambiguous = True

    if len(time_entries) == 0:
        reasons.append(UnresolvedReason.MISSING_PROCESS_TIME)
    elif len(time_entries) > 1:
        reasons.append(UnresolvedReason.MULTIPLE_PROCESS_TIME)
        ambiguous = True

    if thermal_entries:
        reasons.append(UnresolvedReason.THERMAL_TEMPERATURE_UNBOUND)
    if cooling_entries:
        reasons.append(UnresolvedReason.COOLING_REJECTION_UNBOUND)

    if ambiguous:
        electrical_status = ProjectionStatus.AMBIGUOUS
    elif electrical_energy is None or len(peak_entries) != 1 or len(time_entries) != 1:
        electrical_status = ProjectionStatus.INCOMPLETE
    else:
        electrical_status = ProjectionStatus.COMPLETE

    return ProcessUtilityProjection(
        process_id=process_id,
        electrical_status=electrical_status,
        electrical_energy_j=electrical_energy,
        peak_electrical_power_w=_single(peak_entries, UtilityKind.PEAK_ELECTRICAL_POWER),
        process_time_s=_single(time_entries, UtilityKind.PROCESS_TIME),
        thermal_energy_j=thermal_energy,
        cooling_energy_j=cooling_energy,
        unresolved=tuple(reasons),
    )


def _u(kind: UtilityKind, lo: float, hi: float | None = None) -> Utility:
    return Utility(kind, Range(lo, lo if hi is None else hi))


def self_test() -> None:
    baseline = [
        _u(UtilityKind.ELECTRICAL_ENERGY, 40.0, 50.0),
        _u(UtilityKind.ELECTRICAL_ENERGY, 10.0, 20.0),
        _u(UtilityKind.PEAK_ELECTRICAL_POWER, 15.0, 20.0),
        _u(UtilityKind.PROCESS_TIME, 5.0, 10.0),
    ]
    projected = project_process_utilities("p1", baseline)
    assert projected.electrical_status is ProjectionStatus.COMPLETE
    assert projected.electrical_energy_j == Range(50.0, 70.0)
    assert projected.peak_electrical_power_w == Range(15.0, 20.0)
    assert projected.process_time_s == Range(5.0, 10.0)
    assert projected.unresolved == ()

    missing_time = project_process_utilities("p1", baseline[:-1])
    assert missing_time.electrical_status is ProjectionStatus.INCOMPLETE
    assert UnresolvedReason.MISSING_PROCESS_TIME in missing_time.unresolved

    duplicate_peak = project_process_utilities(
        "p1", baseline + [_u(UtilityKind.PEAK_ELECTRICAL_POWER, 1.0)]
    )
    assert duplicate_peak.electrical_status is ProjectionStatus.AMBIGUOUS
    assert duplicate_peak.peak_electrical_power_w is None
    assert UnresolvedReason.MULTIPLE_PEAK_POWER in duplicate_peak.unresolved

    duplicate_time = project_process_utilities(
        "p1", baseline + [_u(UtilityKind.PROCESS_TIME, 1.0)]
    )
    assert duplicate_time.electrical_status is ProjectionStatus.AMBIGUOUS
    assert duplicate_time.process_time_s is None
    assert UnresolvedReason.MULTIPLE_PROCESS_TIME in duplicate_time.unresolved

    unresolved_heat = project_process_utilities(
        "p1",
        baseline
        + [
            _u(UtilityKind.THERMAL_ENERGY, 25.0, 30.0),
            _u(UtilityKind.COOLING_ENERGY, 5.0, 8.0),
        ],
    )
    assert unresolved_heat.electrical_status is ProjectionStatus.COMPLETE
    assert unresolved_heat.thermal_energy_j == Range(25.0, 30.0)
    assert unresolved_heat.cooling_energy_j == Range(5.0, 8.0)
    assert UnresolvedReason.THERMAL_TEMPERATURE_UNBOUND in unresolved_heat.unresolved
    assert UnresolvedReason.COOLING_REJECTION_UNBOUND in unresolved_heat.unresolved

    no_energy = project_process_utilities(
        "p1",
        [
            _u(UtilityKind.PEAK_ELECTRICAL_POWER, 15.0),
            _u(UtilityKind.PROCESS_TIME, 5.0),
        ],
    )
    assert no_energy.electrical_status is ProjectionStatus.INCOMPLETE
    assert UnresolvedReason.MISSING_ELECTRICAL_ENERGY in no_energy.unresolved

    try:
        project_process_utilities(
            "p1",
            [
                _u(UtilityKind.ELECTRICAL_ENERGY, float("1.7976931348623157e308")),
                _u(UtilityKind.ELECTRICAL_ENERGY, float("1.7976931348623157e308")),
                _u(UtilityKind.PEAK_ELECTRICAL_POWER, 1.0),
                _u(UtilityKind.PROCESS_TIME, 1.0),
            ],
        )
    except ValueError as exc:
        assert "aggregate overflow" in str(exc)
    else:
        raise AssertionError("aggregate overflow must fail closed")

    for bad in (
        [Utility(UtilityKind.ELECTRICAL_ENERGY, Range(-1.0, 1.0))],
        [Utility(UtilityKind.ELECTRICAL_ENERGY, Range(2.0, 1.0))],
        [_u(UtilityKind.PROCESS_TIME, 0.0, 1.0)],
    ):
        try:
            project_process_utilities("p1", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed utility must fail closed")

    try:
        project_process_utilities("", baseline)
    except ValueError:
        pass
    else:
        raise AssertionError("blank process id must fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
    else:
        parser.error("only --self-test is supported")


if __name__ == "__main__":
    main()
