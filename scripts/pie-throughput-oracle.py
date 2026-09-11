#!/usr/bin/env python3
"""Independent PIE Phase-0 process scale/throughput oracle."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class Feasibility(str, Enum):
    GUARANTEED = "Guaranteed"
    POSSIBLE = "Possible"
    IMPOSSIBLE = "Impossible"


class Basis(str, Enum):
    CONTINUOUS = "Continuous"
    BATCH = "Batch"


class ScaleMode(str, Enum):
    SINGLE_ONLY = "SingleOnly"
    INDEPENDENT_PARALLEL = "IndependentParallel"


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def validate(self, upper: float | None = None, strictly_positive: bool = False) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("interval endpoints must be finite")
        if self.lo < 0 or self.lo > self.hi:
            raise ValueError("invalid nonnegative interval")
        if strictly_positive and self.lo <= 0:
            raise ValueError("interval must be strictly positive")
        if upper is not None and self.hi > upper:
            raise ValueError("interval exceeds upper bound")


@dataclass(frozen=True)
class UnitModel:
    basis: Basis
    rate_kg_per_h: Interval | None
    batch_mass_kg: Interval | None
    cycle_time_h: Interval | None
    duty_cycle: Interval
    availability: Interval


@dataclass(frozen=True)
class ParallelRule:
    mode: ScaleMode
    efficiency: Interval


@dataclass(frozen=True)
class CapacityResult:
    nominal_unit_rate_kg_h: tuple[float, float]
    effective_output_kg: tuple[float, float]
    feasibility: str


def mul(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo * b.lo, a.hi * b.hi)


def div_pos(a: Interval, b: Interval) -> Interval:
    b.validate(strictly_positive=True)
    return Interval(a.lo / b.hi, a.hi / b.lo)


def unit_rate(model: UnitModel) -> Interval:
    model.duty_cycle.validate(1.0)
    model.availability.validate(1.0)
    if model.basis == Basis.CONTINUOUS:
        if model.rate_kg_per_h is None or model.batch_mass_kg is not None or model.cycle_time_h is not None:
            raise ValueError("continuous basis requires only rate")
        model.rate_kg_per_h.validate()
        return model.rate_kg_per_h
    if model.basis == Basis.BATCH:
        if model.rate_kg_per_h is not None or model.batch_mass_kg is None or model.cycle_time_h is None:
            raise ValueError("batch basis requires batch mass and cycle time")
        model.batch_mass_kg.validate()
        return div_pos(model.batch_mass_kg, model.cycle_time_h)
    raise ValueError("unknown process basis")


def assess_capacity(
    model: UnitModel,
    unit_count: int,
    parallel_rule: ParallelRule,
    horizon_h: Interval,
    demand_kg: Interval,
) -> CapacityResult:
    if not isinstance(unit_count, int) or unit_count < 1:
        raise ValueError("unit count must be a positive integer")
    horizon_h.validate(strictly_positive=True)
    demand_kg.validate()
    rate = unit_rate(model)
    effective_rate = mul(mul(rate, model.duty_cycle), model.availability)

    if unit_count > 1:
        parallel_rule.efficiency.validate(1.0)
        if parallel_rule.mode != ScaleMode.INDEPENDENT_PARALLEL:
            raise ValueError("no declared multi-unit scaling rule")
        scale = Interval(
            unit_count * parallel_rule.efficiency.lo,
            unit_count * parallel_rule.efficiency.hi,
        )
    else:
        scale = Interval(1.0, 1.0)

    output = mul(mul(effective_rate, scale), horizon_h)
    if output.lo >= demand_kg.hi:
        status = Feasibility.GUARANTEED
    elif output.hi < demand_kg.lo:
        status = Feasibility.IMPOSSIBLE
    else:
        status = Feasibility.POSSIBLE

    return CapacityResult((rate.lo, rate.hi), (output.lo, output.hi), status.value)


def self_test() -> None:
    continuous = UnitModel(
        Basis.CONTINUOUS,
        Interval(10.0, 10.0),
        None,
        None,
        Interval(1.0, 1.0),
        Interval(1.0, 1.0),
    )
    single = ParallelRule(ScaleMode.SINGLE_ONLY, Interval(1.0, 1.0))
    parallel = ParallelRule(ScaleMode.INDEPENDENT_PARALLEL, Interval(1.0, 1.0))

    one = assess_capacity(continuous, 1, single, Interval(10.0, 10.0), Interval(100.0, 100.0))
    assert one.effective_output_kg == (100.0, 100.0)
    assert one.feasibility == Feasibility.GUARANTEED.value

    try:
        assess_capacity(continuous, 2, single, Interval(10.0, 10.0), Interval(150.0, 150.0))
    except ValueError:
        pass
    else:
        raise AssertionError("multi-unit scale-up without an explicit rule must fail")

    doubled = assess_capacity(continuous, 2, parallel, Interval(10.0, 10.0), Interval(200.0, 200.0))
    assert doubled.effective_output_kg == (200.0, 200.0)
    assert doubled.feasibility == Feasibility.GUARANTEED.value

    partial_parallel = ParallelRule(ScaleMode.INDEPENDENT_PARALLEL, Interval(0.8, 1.0))
    uncertain_parallel = assess_capacity(
        continuous, 2, partial_parallel, Interval(10.0, 10.0), Interval(180.0, 180.0)
    )
    assert uncertain_parallel.effective_output_kg == (160.0, 200.0)
    assert uncertain_parallel.feasibility == Feasibility.POSSIBLE.value

    batch = UnitModel(
        Basis.BATCH,
        None,
        Interval(20.0, 20.0),
        Interval(2.0, 2.0),
        Interval(1.0, 1.0),
        Interval(1.0, 1.0),
    )
    batch_result = assess_capacity(batch, 1, single, Interval(10.0, 10.0), Interval(100.0, 100.0))
    assert batch_result.nominal_unit_rate_kg_h == (10.0, 10.0)
    assert batch_result.effective_output_kg == (100.0, 100.0)

    uncertain = UnitModel(
        Basis.CONTINUOUS,
        Interval(8.0, 10.0),
        None,
        None,
        Interval(0.8, 1.0),
        Interval(0.9, 1.0),
    )
    widened = assess_capacity(uncertain, 1, single, Interval(10.0, 10.0), Interval(80.0, 80.0))
    assert widened.feasibility == Feasibility.POSSIBLE.value
    assert assess_capacity(uncertain, 1, single, Interval(10.0, 10.0), Interval(101.0, 101.0)).feasibility == Feasibility.IMPOSSIBLE.value

    try:
        bad_batch = UnitModel(
            Basis.BATCH, None, Interval(20.0, 20.0), Interval(0.0, 2.0),
            Interval(1.0, 1.0), Interval(1.0, 1.0),
        )
        assess_capacity(bad_batch, 1, single, Interval(10.0, 10.0), Interval(1.0, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero cycle time must fail")


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
