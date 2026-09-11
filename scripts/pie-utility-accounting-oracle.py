#!/usr/bin/env python3
"""Independent PIE-002 utility and thermal-accounting oracle.

Reference/research implementation only. This script does not import Symthaea,
does not prove thermodynamic feasibility, and grants no plant/control authority.
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
class ElectricalCase:
    gross_energy_j: Interval
    batch_duration_s: Interval
    peak_power_w: Interval
    recoverable_energy_j: Interval
    storage_acceptance_j: Interval
    available_energy_capacity_j: Interval
    available_continuous_power_w: Interval
    available_peak_power_w: Interval


@dataclass(frozen=True)
class ElectricalReport:
    gross_energy_j: Interval
    accepted_recovery_j: Interval
    net_energy_j: Interval
    average_power_w: Interval
    peak_power_w: Interval
    energy_capacity: str
    continuous_power: str
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
    temperature_feasibility: str
    combined_feasibility: str


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
    for index, item in enumerate(items):
        item.validate(f"min_interval_{index}")
    return Interval(
        min=min(item.min for item in items),
        max=min(item.max for item in items),
    )


def evaluate_electrical(case: ElectricalCase) -> ElectricalReport:
    """Evaluate energy, average power, peak power, and bounded recovery."""
    case.gross_energy_j.validate("gross_energy_j")
    case.batch_duration_s.validate("batch_duration_s", strictly_positive=True)
    case.peak_power_w.validate("peak_power_w")
    case.recoverable_energy_j.validate("recoverable_energy_j")
    case.storage_acceptance_j.validate("storage_acceptance_j")
    case.available_energy_capacity_j.validate("available_energy_capacity_j")
    case.available_continuous_power_w.validate("available_continuous_power_w")
    case.available_peak_power_w.validate("available_peak_power_w")

    # Actual recovery is bounded by physically recoverable energy, storage
    # acceptance, and the gross source energy. Gross demand stays explicit.
    accepted = _min_intervals(
        case.recoverable_energy_j,
        case.storage_acceptance_j,
        case.gross_energy_j,
    )

    # Conservative interval subtraction: gross - accepted.
    net = Interval(
        min=max(0.0, case.gross_energy_j.min - accepted.max),
        max=max(0.0, case.gross_energy_j.max - accepted.min),
    )

    average = Interval(
        min=net.min / case.batch_duration_s.max,
        max=net.max / case.batch_duration_s.min,
    )

    return ElectricalReport(
        gross_energy_j=case.gross_energy_j,
        accepted_recovery_j=accepted,
        net_energy_j=net,
        average_power_w=average,
        peak_power_w=case.peak_power_w,
        energy_capacity=feasibility(net, case.available_energy_capacity_j).value,
        continuous_power=feasibility(average, case.available_continuous_power_w).value,
        peak_power=feasibility(case.peak_power_w, case.available_peak_power_w).value,
    )


def evaluate_thermal(supply: ThermalSupply, demand: ThermalDemand) -> ThermalReport:
    """Screen waste/process heat by quantity and temperature envelope."""
    supply.energy_j.validate("thermal_supply_energy")
    supply.source_temperature_k.validate("source_temperature_k", strictly_positive=True)
    demand.energy_j.validate("thermal_demand_energy")
    demand.required_temperature_k.validate("required_temperature_k", strictly_positive=True)

    energy_state = feasibility(demand.energy_j, supply.energy_j)

    # Temperature is a screening condition only. This does not calculate
    # entropy, exergy, heat-exchanger approach temperature, phase change, or loss.
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
        temperature_feasibility=temperature_state.value,
        combined_feasibility=combined.value,
    )


def _interval(payload: dict[str, Any], key: str) -> Interval:
    item = payload[key]
    return Interval(float(item["min"]), float(item["max"]))


def electrical_from_json(payload: dict[str, Any]) -> ElectricalReport:
    return evaluate_electrical(ElectricalCase(
        gross_energy_j=_interval(payload, "gross_energy_j"),
        batch_duration_s=_interval(payload, "batch_duration_s"),
        peak_power_w=_interval(payload, "peak_power_w"),
        recoverable_energy_j=_interval(payload, "recoverable_energy_j"),
        storage_acceptance_j=_interval(payload, "storage_acceptance_j"),
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


def self_test() -> None:
    base = dict(
        gross_energy_j=Interval(100.0, 100.0),
        peak_power_w=Interval(50.0, 50.0),
        recoverable_energy_j=Interval(0.0, 0.0),
        storage_acceptance_j=Interval(0.0, 0.0),
        available_energy_capacity_j=Interval(200.0, 200.0),
        available_continuous_power_w=Interval(100.0, 100.0),
        available_peak_power_w=Interval(100.0, 100.0),
    )
    fast = evaluate_electrical(ElectricalCase(batch_duration_s=Interval(1.0, 1.0), **base))
    slow = evaluate_electrical(ElectricalCase(batch_duration_s=Interval(10.0, 10.0), **base))
    assert fast.average_power_w == Interval(100.0, 100.0)
    assert slow.average_power_w == Interval(10.0, 10.0)

    peak_fail = evaluate_electrical(ElectricalCase(
        gross_energy_j=Interval(100.0, 100.0), batch_duration_s=Interval(10.0, 10.0),
        peak_power_w=Interval(150.0, 150.0), recoverable_energy_j=Interval(0.0, 0.0),
        storage_acceptance_j=Interval(0.0, 0.0), available_energy_capacity_j=Interval(200.0, 200.0),
        available_continuous_power_w=Interval(20.0, 20.0), available_peak_power_w=Interval(100.0, 100.0),
    ))
    assert peak_fail.energy_capacity == Feasibility.GUARANTEED.value
    assert peak_fail.continuous_power == Feasibility.GUARANTEED.value
    assert peak_fail.peak_power == Feasibility.IMPOSSIBLE.value

    recovery = evaluate_electrical(ElectricalCase(
        gross_energy_j=Interval(100.0, 100.0), batch_duration_s=Interval(10.0, 10.0),
        peak_power_w=Interval(20.0, 20.0), recoverable_energy_j=Interval(30.0, 30.0),
        storage_acceptance_j=Interval(50.0, 50.0), available_energy_capacity_j=Interval(100.0, 100.0),
        available_continuous_power_w=Interval(20.0, 20.0), available_peak_power_w=Interval(30.0, 30.0),
    ))
    assert recovery.gross_energy_j == Interval(100.0, 100.0)
    assert recovery.accepted_recovery_j == Interval(30.0, 30.0)
    assert recovery.net_energy_j == Interval(70.0, 70.0)

    capped = evaluate_electrical(ElectricalCase(
        gross_energy_j=Interval(100.0, 100.0), batch_duration_s=Interval(10.0, 10.0),
        peak_power_w=Interval(20.0, 20.0), recoverable_energy_j=Interval(80.0, 80.0),
        storage_acceptance_j=Interval(25.0, 25.0), available_energy_capacity_j=Interval(100.0, 100.0),
        available_continuous_power_w=Interval(20.0, 20.0), available_peak_power_w=Interval(30.0, 30.0),
    ))
    assert capped.accepted_recovery_j == Interval(25.0, 25.0)
    assert capped.net_energy_j == Interval(75.0, 75.0)

    overclaim = evaluate_electrical(ElectricalCase(
        gross_energy_j=Interval(40.0, 40.0), batch_duration_s=Interval(10.0, 10.0),
        peak_power_w=Interval(20.0, 20.0), recoverable_energy_j=Interval(80.0, 80.0),
        storage_acceptance_j=Interval(100.0, 100.0), available_energy_capacity_j=Interval(100.0, 100.0),
        available_continuous_power_w=Interval(20.0, 20.0), available_peak_power_w=Interval(30.0, 30.0),
    ))
    assert overclaim.accepted_recovery_j == Interval(40.0, 40.0)
    assert overclaim.net_energy_j == Interval(0.0, 0.0)

    low_heat = evaluate_thermal(
        ThermalSupply(Interval(100.0, 100.0), Interval(350.0, 400.0)),
        ThermalDemand(Interval(50.0, 50.0), Interval(500.0, 500.0)),
    )
    assert low_heat.energy_feasibility == Feasibility.GUARANTEED.value
    assert low_heat.temperature_feasibility == Feasibility.IMPOSSIBLE.value
    assert low_heat.combined_feasibility == Feasibility.IMPOSSIBLE.value

    good_heat = evaluate_thermal(
        ThermalSupply(Interval(80.0, 100.0), Interval(800.0, 900.0)),
        ThermalDemand(Interval(50.0, 70.0), Interval(600.0, 700.0)),
    )
    assert good_heat.combined_feasibility == Feasibility.GUARANTEED.value

    uncertain_heat = evaluate_thermal(
        ThermalSupply(Interval(40.0, 80.0), Interval(650.0, 750.0)),
        ThermalDemand(Interval(50.0, 70.0), Interval(700.0, 800.0)),
    )
    assert uncertain_heat.combined_feasibility == Feasibility.POSSIBLE.value

    narrow = feasibility(Interval(90.0, 90.0), Interval(100.0, 100.0))
    wide = feasibility(Interval(80.0, 120.0), Interval(100.0, 100.0))
    assert narrow is Feasibility.GUARANTEED
    assert wide is Feasibility.POSSIBLE

    bad = [
        lambda: feasibility(Interval(-1.0, 1.0), Interval(2.0, 2.0)),
        lambda: feasibility(Interval(2.0, 1.0), Interval(2.0, 2.0)),
        lambda: evaluate_electrical(ElectricalCase(
            gross_energy_j=Interval(1.0, 1.0), batch_duration_s=Interval(0.0, 1.0),
            peak_power_w=Interval(1.0, 1.0), recoverable_energy_j=Interval(0.0, 0.0),
            storage_acceptance_j=Interval(0.0, 0.0), available_energy_capacity_j=Interval(1.0, 1.0),
            available_continuous_power_w=Interval(1.0, 1.0), available_peak_power_w=Interval(1.0, 1.0),
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
