#!/usr/bin/env python3
"""Independent PIE-006 circularity / inventory / heat-allocation oracle.

Research-only reference. No Symthaea imports, process qualification, economics,
or hardware authority.

The oracle enforces three accounting rules:
1. material inventory is consumable and cannot be allocated twice;
2. recycle loops require real starting inventory and explicit recovery losses;
3. waste heat is a finite temperature-qualified resource and cannot be credited
   to multiple sinks beyond the source energy.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field


def finite_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass
class Inventory:
    mass_by_material: dict[str, float] = field(default_factory=dict)

    def add(self, material: str, mass_kg: float) -> None:
        if not material.strip():
            raise ValueError("material id must be nonempty")
        mass_kg = finite_nonnegative(mass_kg, "mass")
        self.mass_by_material[material] = self.mass_by_material.get(material, 0.0) + mass_kg

    def available(self, material: str) -> float:
        return self.mass_by_material.get(material, 0.0)

    def consume(self, material: str, mass_kg: float) -> None:
        mass_kg = finite_nonnegative(mass_kg, "mass")
        available = self.available(material)
        if mass_kg > available + 1e-12:
            raise ValueError(f"insufficient {material}: need {mass_kg}, have {available}")
        remaining = available - mass_kg
        self.mass_by_material[material] = 0.0 if abs(remaining) < 1e-12 else remaining


@dataclass(frozen=True)
class RecoveryReport:
    input_kg: float
    recovered_kg: float
    reject_kg: float
    loss_kg: float
    recovery_fraction: float
    residual_kg: float


def evaluate_recovery(
    input_kg: float,
    recovered_kg: float,
    reject_kg: float,
    loss_kg: float,
    tolerance_kg: float = 0.0,
) -> RecoveryReport:
    values = [input_kg, recovered_kg, reject_kg, loss_kg, tolerance_kg]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("recovery quantities must be finite and nonnegative")
    if input_kg <= 0:
        raise ValueError("recovery input must be positive")
    residual = recovered_kg + reject_kg + loss_kg - input_kg
    if abs(residual) > tolerance_kg:
        raise ValueError("recovery mass does not close")
    return RecoveryReport(
        input_kg,
        recovered_kg,
        reject_kg,
        loss_kg,
        recovered_kg / input_kg,
        residual,
    )


@dataclass(frozen=True)
class ProcessRecipe:
    process_id: str
    input_material: str
    input_kg: float
    output_material: str
    output_kg: float
    reject_material: str | None = None
    reject_kg: float = 0.0
    loss_kg: float = 0.0

    def validate(self) -> None:
        for value, name in (
            (self.input_kg, "input_kg"),
            (self.output_kg, "output_kg"),
            (self.reject_kg, "reject_kg"),
            (self.loss_kg, "loss_kg"),
        ):
            finite_nonnegative(value, name)
        if not self.process_id.strip() or not self.input_material.strip() or not self.output_material.strip():
            raise ValueError("process/material ids must be nonempty")
        if self.input_kg <= 0:
            raise ValueError("process input must be positive")
        if abs((self.output_kg + self.reject_kg + self.loss_kg) - self.input_kg) > 1e-12:
            raise ValueError("process mass does not close")
        if self.reject_kg > 0 and (self.reject_material is None or not self.reject_material.strip()):
            raise ValueError("reject material id required when reject mass is nonzero")


def run_process(inventory: Inventory, recipe: ProcessRecipe) -> None:
    recipe.validate()
    inventory.consume(recipe.input_material, recipe.input_kg)
    inventory.add(recipe.output_material, recipe.output_kg)
    if recipe.reject_kg:
        inventory.add(recipe.reject_material or "", recipe.reject_kg)
    # loss_kg intentionally leaves tracked inventory but remains explicit in recipe.


@dataclass
class HeatSource:
    source_id: str
    energy_j: float
    temperature_k: float
    allocated_j: float = 0.0

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source id required")
        finite_nonnegative(self.energy_j, "heat energy")
        finite_nonnegative(self.allocated_j, "allocated heat")
        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.allocated_j > self.energy_j + 1e-12:
            raise ValueError("heat source over-allocated")

    @property
    def remaining_j(self) -> float:
        return self.energy_j - self.allocated_j


@dataclass(frozen=True)
class HeatSink:
    sink_id: str
    demand_j: float
    minimum_temperature_k: float

    def validate(self) -> None:
        if not self.sink_id.strip():
            raise ValueError("sink id required")
        finite_nonnegative(self.demand_j, "heat demand")
        if not math.isfinite(self.minimum_temperature_k) or self.minimum_temperature_k <= 0:
            raise ValueError("minimum temperature must be finite and positive")


def allocate_heat(source: HeatSource, sink: HeatSink, energy_j: float | None = None) -> float:
    source.validate()
    sink.validate()
    requested = sink.demand_j if energy_j is None else finite_nonnegative(energy_j, "allocated heat")
    if requested > sink.demand_j + 1e-12:
        raise ValueError("allocation exceeds sink demand")
    if source.temperature_k + 1e-12 < sink.minimum_temperature_k:
        raise ValueError("source temperature below sink requirement")
    if requested > source.remaining_j + 1e-12:
        raise ValueError("insufficient unallocated heat")
    source.allocated_j += requested
    return requested


def self_test() -> None:
    inv = Inventory({"scrap": 10.0})
    inv.consume("scrap", 6.0)
    assert abs(inv.available("scrap") - 4.0) < 1e-12
    try:
        inv.consume("scrap", 5.0)
    except ValueError:
        pass
    else:
        raise AssertionError("material double-allocation must fail")

    report = evaluate_recovery(10.0, 8.0, 1.5, 0.5)
    assert abs(report.recovery_fraction - 0.8) < 1e-12

    mass = 10.0
    for _ in range(3):
        mass *= 0.8
    assert abs(mass - 5.12) < 1e-12
    assert mass < 10.0

    a_to_b = ProcessRecipe("a-to-b", "A", 1.0, "B", 1.0)
    b_to_a = ProcessRecipe("b-to-a", "B", 1.0, "A", 1.0)
    empty = Inventory()
    for recipe in (a_to_b, b_to_a):
        try:
            run_process(empty, recipe)
        except ValueError:
            pass
        else:
            raise AssertionError("zero-inventory recycle cycle must not self-start")

    seeded = Inventory({"A": 1.0})
    run_process(seeded, a_to_b)
    assert seeded.available("A") == 0.0 and seeded.available("B") == 1.0
    run_process(seeded, b_to_a)
    assert seeded.available("A") == 1.0 and seeded.available("B") == 0.0

    lossy = ProcessRecipe(
        "lossy-recycle",
        "scrap",
        1.0,
        "scrap",
        0.8,
        reject_material="reject",
        reject_kg=0.1,
        loss_kg=0.1,
    )
    recycle_inv = Inventory({"scrap": 5.0})
    run_process(recycle_inv, lossy)
    assert abs(recycle_inv.available("scrap") - 4.8) < 1e-12
    assert abs(recycle_inv.available("reject") - 0.1) < 1e-12

    source = HeatSource("furnace", 100.0, 800.0)
    sink_a = HeatSink("dryer", 60.0, 500.0)
    sink_b = HeatSink("habitat", 50.0, 300.0)
    assert allocate_heat(source, sink_a) == 60.0
    try:
        allocate_heat(source, sink_b)
    except ValueError:
        pass
    else:
        raise AssertionError("heat double-allocation must fail")
    assert abs(source.remaining_j - 40.0) < 1e-12

    cold = HeatSource("low-grade", 1000.0, 350.0)
    hot_sink = HeatSink("smelter-preheat", 100.0, 500.0)
    try:
        allocate_heat(cold, hot_sink)
    except ValueError:
        pass
    else:
        raise AssertionError("temperature-incompatible heat reuse must fail")

    split = HeatSource("reactor", 100.0, 700.0)
    assert allocate_heat(split, HeatSink("s1", 40.0, 400.0)) == 40.0
    assert allocate_heat(split, HeatSink("s2", 60.0, 300.0)) == 60.0
    assert abs(split.remaining_j) < 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    parser.error("--self-test is required in PIE-006A")


if __name__ == "__main__":
    main()
