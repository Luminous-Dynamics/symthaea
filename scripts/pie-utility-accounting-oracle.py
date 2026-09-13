#!/usr/bin/env python3
"""Independent PIE-002 utility and thermal-accounting oracle.

Reference/research implementation only. This script does not import Symthaea,
does not prove thermodynamic feasibility, and grants no plant/control authority.

Recovery accounting deliberately separates current-cycle gross supply screening
from steady-cycle recovery credit. Captured/recoverable energy does not prove it
is available early enough to reduce the current cycle's source requirement.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Feasibility(str, Enum):
    """Conservative demand-vs-capacity classification."""

    GUARANTEED = "Guaranteed"
    POSSIBLE = "Possible"
    IMPOSSIBLE = "Impossible"


@dataclass(frozen=True)
class Interval:
    """Inclusive non-negative interval."""

    min: float
    max: float

    def validate(self, label: str, *, strictly_positive: bool = False) -> None:
        if not (math.isfinite(self.min) and math.isfinite(self.max)):
            raise ValueError(f"{label} bounds must be finite")
        if strictly_positive:
            if self.min <= 0.0 or self.max <= 0.0:
                raise ValueError(f"{label} bounds must be positive")
        elif self.min < 0.0 or self.max < 0.0:
            raise ValueError(f"{label} bounds must be nonnegative")
        if self.min > self.max:
            raise ValueError(f"{label} min must be <= max")


@dataclass(frozen=True)
class FractionInterval:
    """Inclusive dimensionless fraction in [0, 1]."""

    min: float
    max: float

    def validate(self, label: str) -> None:
        if not (math.isfinite(self.min) and math.isfinite(self.max)):
            raise ValueError(f"{label} bounds must be finite")
        if self.min < 0.0 or self.max > 1.0 or self.min > self.max:
            raise ValueError(f"{label} must satisfy 0 <= min <= max <= 1")


@dataclass(frozen=True)
class ElectricalCase:
    gross_energy_j: Interval
    batch_duration_s: Interval
    peak_power_w: Interval
    recoverable_energy_j: Interval
    recovery_duration_s: Interval
    storage_acceptance_j: Interval
    storage_charge_power_w: Interval
    storage_discharge_power_w: Interval
    recovery_delivery_fraction: FractionInterval
    available_energy_capacity_j: Interval
    available_continuous_power_w: Interval
    available_peak_power_w: Interval


@dataclass(frozen=True)
class ElectricalReport:
    gross_energy_j: Interval
    accepted_recovery_j: Interval
    usable_steady_cycle_recovery_j: Interval
    net_steady_cycle_energy_j: Interval
    gross_average_power_w: Interval
    net_steady_cycle_average_power_w: Interval
    peak_power_w: Interval
    gross_energy_capacity: str
    steady_cycle_energy_capacity: str
    gross_average_power_screen: str
    steady_cycle_average_power_screen: str
    peak_power: str


@dataclass(frozen=True)
class ThermalSupply:
    energy_j: Interval
    source_temperature_k: Interval


@dataclass(frozen=True)
class ThermalDemand:
    energy_j: Interval
    required_temperature_k: Interval


@dataclass(frozen=True)
class ThermalReport:
    energy_feasibility: str
    temperature_compatibility: str
    combined_screening: str


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must remain finite")
    return value


def feasibility(demand: Interval, capacity: Interval) -> Feasibility:
    """Classify uncertain demand against uncertain capacity conservatively."""
    demand.validate("demand")
    capacity.validate("capacity")
    if demand.max <= capacity.min:
        return Feasibility.GUARANTEED
    if demand.min > capacity.max:
        return Feasibility.IMPOSSIBLE
    return Feasibility.POSSIBLE


def _min_intervals(*items: Interval) -> Interval:
    if not items:
        raise ValueError("at least one interval is required")
    for index, item in enumerate(items):
        item.validate(f"min_interval_{index}")
    return Interval(
        min=min(item.min for item in items),
        max=min(item.max for item in items),
    )


def _multiply_nonnegative(a: Interval, b: Interval, label: str) -> Interval:
    a.validate(f"{label}_lhs")
    b.validate(f"{label}_rhs")
    return Interval(
        min=_finite(a.min * b.min, f"{label}_min"),
        max=_finite(a.max * b.max, f"{label}_max"),
    )


def _multiply_fraction(a: Interval, fraction: FractionInterval, label: str) -> Interval:
    a.validate(f"{label}_value")
    fraction.validate(f"{label}_fraction")
    return Interval(
        min=_finite(a.min * fraction.min, f"{label}_min"),
        max=_finite(a.max * fraction.max, f"{label}_max"),
    )


def _subtract_nonnegative(a: Interval, b: Interval, label: str) -> Interval:
    a.validate(f"{label}_lhs")
    b.validate(f"{label}_rhs")
    return Interval(
        min=max(0.0, _finite(a.min - b.max, f"{label}_min")),
        max=max(0.0, _finite(a.max - b.min, f"{label}_max")),
    )


def _energy_over_duration(energy: Interval, duration: Interval, label: str) -> Interval:
    energy.validate(f"{label}_energy")
    duration.validate(f"{label}_duration", strictly_positive=True)
    return Interval(
        min=_finite(energy.min / duration.max, f"{label}_min"),
        max=_finite(energy.max / duration.min, f"{label}_max"),
    )


def evaluate_electrical(case: ElectricalCase) -> ElectricalReport:
    """Evaluate gross supply and explicitly separate steady-cycle recovery credit."""
    case.gross_energy_j.validate("gross_energy_j")
    case.batch_duration_s.validate("batch_duration_s", strictly_positive=True)
    case.peak_power_w.validate("peak_power_w")
    case.recoverable_energy_j.validate("recoverable_energy_j")
    case.recovery_duration_s.validate("recovery_duration_s", strictly_positive=True)
    case.storage_acceptance_j.validate("storage_acceptance_j")
    case.storage_charge_power_w.validate("storage_charge_power_w")
    case.storage_discharge_power_w.validate("storage_discharge_power_w")
    case.recovery_delivery_fraction.validate("recovery_delivery_fraction")
    case.available_energy_capacity_j.validate("available_energy_capacity_j")
    case.available_continuous_power_w.validate("available_continuous_power_w")
    case.available_peak_power_w.validate("available_peak_power_w")

    # Storage charge power limits how much of the recoverable stream can actually
    # be accepted during the declared recovery window.
    charge_limited_recovery = _multiply_nonnegative(
        case.storage_charge_power_w,
        case.recovery_duration_s,
        "charge_limited_recovery",
    )
    accepted = _min_intervals(
        case.recoverable_energy_j,
        case.storage_acceptance_j,
        charge_limited_recovery,
        case.gross_energy_j,
    )

    # Captured energy is not automatically reusable energy. Apply the explicit
    # delivery/round-trip fraction, then cap credit by storage discharge power
    # over the next batch. This is still steady-cycle accounting, not proof that
    # the current cycle can consume energy recovered later in that same cycle.
    delivered = _multiply_fraction(
        accepted,
        case.recovery_delivery_fraction,
        "delivered_recovery",
    )
    discharge_limited_recovery = _multiply_nonnegative(
        case.storage_discharge_power_w,
        case.batch_duration_s,
        "discharge_limited_recovery",
    )
    usable = _min_intervals(
        delivered,
        discharge_limited_recovery,
        case.gross_energy_j,
    )
    net_steady = _subtract_nonnegative(
        case.gross_energy_j,
        usable,
        "net_steady_cycle_energy",
    )

    gross_average = _energy_over_duration(
        case.gross_energy_j,
        case.batch_duration_s,
        "gross_average_power",
    )
    net_average = _energy_over_duration(
        net_steady,
        case.batch_duration_s,
        "net_steady_cycle_average_power",
    )

    return ElectricalReport(
        gross_energy_j=case.gross_energy_j,
        accepted_recovery_j=accepted,
        usable_steady_cycle_recovery_j=usable,
        net_steady_cycle_energy_j=net_steady,
        gross_average_power_w=gross_average,
        net_steady_cycle_average_power_w=net_average,
        peak_power_w=case.peak_power_w,
        # Gross screens make no temporal recovery-credit assumption. The power result is
        # a cycle-average demand screen against sustained capacity, not a time-resolved
        # dispatch or load-profile proof.
        gross_energy_capacity=feasibility(
            case.gross_energy_j, case.available_energy_capacity_j
        ).value,
        gross_average_power_screen=feasibility(
            gross_average, case.available_continuous_power_w
        ).value,
        # Steady-cycle screens are separately named because they assume energy
        # captured in a previous cycle can be delivered in the next cycle under
        # the declared charge/discharge/efficiency limits.
        steady_cycle_energy_capacity=feasibility(
            net_steady, case.available_energy_capacity_j
        ).value,
        steady_cycle_average_power_screen=feasibility(
            net_average, case.available_continuous_power_w
        ).value,
        peak_power=feasibility(
            case.peak_power_w, case.available_peak_power_w
        ).value,
    )


def evaluate_thermal(supply: ThermalSupply, demand: ThermalDemand) -> ThermalReport:
    """Screen waste/process heat by quantity and temperature envelope."""
    supply.energy_j.validate("thermal_supply_energy")
    supply.source_temperature_k.validate("source_temperature_k", strictly_positive=True)
    demand.energy_j.validate("thermal_demand_energy")
    demand.required_temperature_k.validate("required_temperature_k", strictly_positive=True)

    energy_state = feasibility(demand.energy_j, supply.energy_j)

    # Temperature is only a compatibility screen. It does not calculate entropy,
    # exergy, heat-exchanger approach temperature, phase change, or losses.
    if supply.source_temperature_k.min >= demand.required_temperature_k.max:
        temperature_state = Feasibility.GUARANTEED
    elif supply.source_temperature_k.max < demand.required_temperature_k.min:
        temperature_state = Feasibility.IMPOSSIBLE
    else:
        temperature_state = Feasibility.POSSIBLE

    if Feasibility.IMPOSSIBLE in (energy_state, temperature_state):
        combined = Feasibility.IMPOSSIBLE
    elif energy_state is Feasibility.GUARANTEED and temperature_state is Feasibility.GUARANTEED:
        combined = Feasibility.GUARANTEED
    else:
        combined = Feasibility.POSSIBLE

    return ThermalReport(
        energy_feasibility=energy_state.value,
        temperature_compatibility=temperature_state.value,
        combined_screening=combined.value,
    )


def _interval(payload: dict[str, Any], key: str) -> Interval:
    item = payload[key]
    return Interval(float(item["min"]), float(item["max"]))


def _fraction(payload: dict[str, Any], key: str) -> FractionInterval:
    item = payload[key]
    return FractionInterval(float(item["min"]), float(item["max"]))


def electrical_from_json(payload: dict[str, Any]) -> ElectricalReport:
    return evaluate_electrical(ElectricalCase(
        gross_energy_j=_interval(payload, "gross_energy_j"),
        batch_duration_s=_interval(payload, "batch_duration_s"),
        peak_power_w=_interval(payload, "peak_power_w"),
        recoverable_energy_j=_interval(payload, "recoverable_energy_j"),
        recovery_duration_s=_interval(payload, "recovery_duration_s"),
        storage_acceptance_j=_interval(payload, "storage_acceptance_j"),
        storage_charge_power_w=_interval(payload, "storage_charge_power_w"),
        storage_discharge_power_w=_interval(payload, "storage_discharge_power_w"),
        recovery_delivery_fraction=_fraction(payload, "recovery_delivery_fraction"),
        available_energy_capacity_j=_interval(payload, "available_energy_capacity_j"),
        available_continuous_power_w=_interval(payload, "available_continuous_power_w"),
        available_peak_power_w=_interval(payload, "available_peak_power_w"),
    ))


def thermal_from_json(payload: dict[str, Any]) -> ThermalReport:
    supply_payload = payload["supply"]
    demand_payload = payload["demand"]
    return evaluate_thermal(
        ThermalSupply(
            energy_j=_interval(supply_payload, "energy_j"),
            source_temperature_k=_interval(supply_payload, "source_temperature_k"),
        ),
        ThermalDemand(
            energy_j=_interval(demand_payload, "energy_j"),
            required_temperature_k=_interval(demand_payload, "required_temperature_k"),
        ),
    )


def _base_case(**overrides: object) -> ElectricalCase:
    values: dict[str, object] = dict(
        gross_energy_j=Interval(100.0, 100.0),
        batch_duration_s=Interval(10.0, 10.0),
        peak_power_w=Interval(50.0, 50.0),
        recoverable_energy_j=Interval(0.0, 0.0),
        recovery_duration_s=Interval(10.0, 10.0),
        storage_acceptance_j=Interval(0.0, 0.0),
        storage_charge_power_w=Interval(0.0, 0.0),
        storage_discharge_power_w=Interval(0.0, 0.0),
        recovery_delivery_fraction=FractionInterval(1.0, 1.0),
        available_energy_capacity_j=Interval(200.0, 200.0),
        available_continuous_power_w=Interval(100.0, 100.0),
        available_peak_power_w=Interval(100.0, 100.0),
    )
    values.update(overrides)
    return ElectricalCase(**values)  # type: ignore[arg-type]


def self_test() -> None:
    fast = evaluate_electrical(_base_case(batch_duration_s=Interval(1.0, 1.0)))
    slow = evaluate_electrical(_base_case(batch_duration_s=Interval(10.0, 10.0)))
    assert fast.gross_average_power_w == Interval(100.0, 100.0)
    assert slow.gross_average_power_w == Interval(10.0, 10.0)

    peak_fail = evaluate_electrical(_base_case(
        peak_power_w=Interval(150.0, 150.0),
        available_continuous_power_w=Interval(20.0, 20.0),
        available_peak_power_w=Interval(100.0, 100.0),
    ))
    assert peak_fail.gross_energy_capacity == Feasibility.GUARANTEED.value
    assert peak_fail.gross_average_power_screen == Feasibility.GUARANTEED.value
    assert peak_fail.peak_power == Feasibility.IMPOSSIBLE.value

    recovery = evaluate_electrical(_base_case(
        recoverable_energy_j=Interval(30.0, 30.0),
        storage_acceptance_j=Interval(50.0, 50.0),
        storage_charge_power_w=Interval(10.0, 10.0),
        storage_discharge_power_w=Interval(10.0, 10.0),
    ))
    assert recovery.gross_energy_j == Interval(100.0, 100.0)
    assert recovery.accepted_recovery_j == Interval(30.0, 30.0)
    assert recovery.usable_steady_cycle_recovery_j == Interval(30.0, 30.0)
    assert recovery.net_steady_cycle_energy_j == Interval(70.0, 70.0)

    charge_capped = evaluate_electrical(_base_case(
        recoverable_energy_j=Interval(80.0, 80.0),
        recovery_duration_s=Interval(2.0, 2.0),
        storage_acceptance_j=Interval(100.0, 100.0),
        storage_charge_power_w=Interval(10.0, 10.0),
        storage_discharge_power_w=Interval(100.0, 100.0),
    ))
    assert charge_capped.accepted_recovery_j == Interval(20.0, 20.0)

    efficiency_capped = evaluate_electrical(_base_case(
        recoverable_energy_j=Interval(80.0, 80.0),
        storage_acceptance_j=Interval(80.0, 80.0),
        storage_charge_power_w=Interval(100.0, 100.0),
        storage_discharge_power_w=Interval(100.0, 100.0),
        recovery_delivery_fraction=FractionInterval(0.5, 0.5),
    ))
    assert efficiency_capped.accepted_recovery_j == Interval(80.0, 80.0)
    assert efficiency_capped.usable_steady_cycle_recovery_j == Interval(40.0, 40.0)
    assert efficiency_capped.net_steady_cycle_energy_j == Interval(60.0, 60.0)

    discharge_capped = evaluate_electrical(_base_case(
        recoverable_energy_j=Interval(80.0, 80.0),
        storage_acceptance_j=Interval(80.0, 80.0),
        storage_charge_power_w=Interval(100.0, 100.0),
        storage_discharge_power_w=Interval(2.0, 2.0),
    ))
    assert discharge_capped.accepted_recovery_j == Interval(80.0, 80.0)
    assert discharge_capped.usable_steady_cycle_recovery_j == Interval(20.0, 20.0)
    assert discharge_capped.net_steady_cycle_energy_j == Interval(80.0, 80.0)

    # Recovery can improve steady-cycle accounting but must not silently strengthen
    # the current/gross supply screen.
    separated = evaluate_electrical(_base_case(
        recoverable_energy_j=Interval(60.0, 60.0),
        storage_acceptance_j=Interval(60.0, 60.0),
        storage_charge_power_w=Interval(100.0, 100.0),
        storage_discharge_power_w=Interval(100.0, 100.0),
        available_energy_capacity_j=Interval(50.0, 50.0),
        available_continuous_power_w=Interval(5.0, 5.0),
    ))
    assert separated.gross_energy_capacity == Feasibility.IMPOSSIBLE.value
    assert separated.steady_cycle_energy_capacity == Feasibility.GUARANTEED.value
    assert separated.gross_average_power_screen == Feasibility.IMPOSSIBLE.value
    assert separated.steady_cycle_average_power_screen == Feasibility.GUARANTEED.value

    low_heat = evaluate_thermal(
        ThermalSupply(Interval(100.0, 100.0), Interval(350.0, 400.0)),
        ThermalDemand(Interval(50.0, 50.0), Interval(500.0, 500.0)),
    )
    assert low_heat.energy_feasibility == Feasibility.GUARANTEED.value
    assert low_heat.temperature_compatibility == Feasibility.IMPOSSIBLE.value
    assert low_heat.combined_screening == Feasibility.IMPOSSIBLE.value

    good_heat = evaluate_thermal(
        ThermalSupply(Interval(80.0, 100.0), Interval(800.0, 900.0)),
        ThermalDemand(Interval(50.0, 70.0), Interval(600.0, 700.0)),
    )
    assert good_heat.combined_screening == Feasibility.GUARANTEED.value

    uncertain_heat = evaluate_thermal(
        ThermalSupply(Interval(40.0, 80.0), Interval(650.0, 750.0)),
        ThermalDemand(Interval(50.0, 70.0), Interval(700.0, 800.0)),
    )
    assert uncertain_heat.combined_screening == Feasibility.POSSIBLE.value

    narrow = feasibility(Interval(90.0, 90.0), Interval(100.0, 100.0))
    wide = feasibility(Interval(80.0, 120.0), Interval(100.0, 100.0))
    assert narrow is Feasibility.GUARANTEED
    assert wide is Feasibility.POSSIBLE

    bad = [
        lambda: feasibility(Interval(-1.0, 1.0), Interval(2.0, 2.0)),
        lambda: feasibility(Interval(2.0, 1.0), Interval(2.0, 2.0)),
        lambda: evaluate_electrical(_base_case(batch_duration_s=Interval(0.0, 1.0))),
        lambda: evaluate_electrical(_base_case(recovery_duration_s=Interval(0.0, 1.0))),
        lambda: evaluate_electrical(_base_case(
            recovery_delivery_fraction=FractionInterval(-0.1, 1.0)
        )),
        lambda: evaluate_electrical(_base_case(
            gross_energy_j=Interval(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")),
            batch_duration_s=Interval(float.fromhex("0x0.0000000000001p-1022"), float.fromhex("0x0.0000000000001p-1022")),
        )),
        lambda: evaluate_thermal(
            ThermalSupply(Interval(1.0, 1.0), Interval(float("nan"), 300.0)),
            ThermalDemand(Interval(1.0, 1.0), Interval(200.0, 200.0)),
        ),
    ]
    for case in bad:
        try:
            case()
        except (ValueError, TypeError, KeyError):
            pass
        else:
            raise AssertionError("malformed utility input must fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("electrical", "thermal"))
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    if not args.mode or not args.json:
        parser.error("--self-test or both --mode and --json are required")
    payload = json.loads(args.json)
    report = electrical_from_json(payload) if args.mode == "electrical" else thermal_from_json(payload)
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
