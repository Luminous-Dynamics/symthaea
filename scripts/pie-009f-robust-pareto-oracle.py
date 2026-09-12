#!/usr/bin/env python3
"""PIE-009F independent robust architecture-selection oracle.

Synthetic, deterministic reference semantics only. No Moon/Mars reliability,
probability, economics, or operational-control claim is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import math

@dataclass(frozen=True)
class Architecture:
    architecture_id: str
    seed_mass: float
    reserve_mass: float
    blocker_import_mass: float
    productive_closure: float

@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    mandatory: bool
    minimum_essential_floor: float
    max_recovery_steps: Optional[int] = None

@dataclass(frozen=True)
class ScenarioResult:
    architecture_id: str
    scenario_id: str
    essential_floor: float
    cumulative_deficit: float
    recovery_steps: Optional[int]

@dataclass(frozen=True)
class RobustVector:
    architecture_id: str
    robust: bool
    worst_essential_floor: float
    worst_cumulative_deficit: float
    worst_recovery_steps: float
    seed_mass: float
    reserve_mass: float
    blocker_import_mass: float
    productive_closure: float
    classification: str = ""

def _nn(x: float, name: str):
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and nonnegative")

def validate(architectures: List[Architecture], scenarios: List[ScenarioSpec], results: List[ScenarioResult]):
    if not architectures or not scenarios:
        raise ValueError("architectures and scenarios required")
    aids, sids = set(), set()
    for a in architectures:
        if not a.architecture_id or a.architecture_id in aids:
            raise ValueError("bad architecture id")
        aids.add(a.architecture_id)
        for value, name in [
            (a.seed_mass, "seed_mass"),
            (a.reserve_mass, "reserve_mass"),
            (a.blocker_import_mass, "blocker_import_mass"),
        ]:
            _nn(value, name)
        if not math.isfinite(a.productive_closure) or not (0 <= a.productive_closure <= 1):
            raise ValueError("productive_closure must be [0,1]")
    for s in scenarios:
        if not s.scenario_id or s.scenario_id in sids:
            raise ValueError("bad scenario id")
        sids.add(s.scenario_id)
        if not math.isfinite(s.minimum_essential_floor) or s.minimum_essential_floor < 0:
            raise ValueError("bad floor requirement")
        if s.max_recovery_steps is not None and s.max_recovery_steps < 0:
            raise ValueError("bad recovery requirement")
    seen = set()
    for r in results:
        key = (r.architecture_id, r.scenario_id)
        if r.architecture_id not in aids or r.scenario_id not in sids or key in seen:
            raise ValueError("bad or duplicate result")
        seen.add(key)
        if not math.isfinite(r.essential_floor) or r.essential_floor < 0:
            raise ValueError("bad essential floor")
        _nn(r.cumulative_deficit, "cumulative_deficit")
        if r.recovery_steps is not None and r.recovery_steps < 0:
            raise ValueError("bad recovery steps")
    expected = {(aid, sid) for aid in aids for sid in sids}
    if seen != expected:
        raise ValueError("every architecture requires every scenario result")

def _dominates(a: RobustVector, b: RobustVector) -> bool:
    no_worse = (
        a.worst_essential_floor >= b.worst_essential_floor
        and a.worst_cumulative_deficit <= b.worst_cumulative_deficit
        and a.worst_recovery_steps <= b.worst_recovery_steps
        and a.seed_mass <= b.seed_mass
        and a.reserve_mass <= b.reserve_mass
        and a.blocker_import_mass <= b.blocker_import_mass
        and a.productive_closure >= b.productive_closure
    )
    strictly_better = (
        a.worst_essential_floor > b.worst_essential_floor
        or a.worst_cumulative_deficit < b.worst_cumulative_deficit
        or a.worst_recovery_steps < b.worst_recovery_steps
        or a.seed_mass < b.seed_mass
        or a.reserve_mass < b.reserve_mass
        or a.blocker_import_mass < b.blocker_import_mass
        or a.productive_closure > b.productive_closure
    )
    return no_worse and strictly_better

def evaluate(architectures, scenarios, results) -> List[RobustVector]:
    validate(architectures, scenarios, results)
    by = {(r.architecture_id, r.scenario_id): r for r in results}
    vectors = []
    for a in architectures:
        robust = True
        floors, deficits, recovery = [], [], []
        for s in scenarios:
            r = by[(a.architecture_id, s.scenario_id)]
            floors.append(r.essential_floor)
            deficits.append(r.cumulative_deficit)
            recovery.append(float("inf") if r.recovery_steps is None else float(r.recovery_steps))
            if s.mandatory:
                if r.essential_floor + 1e-12 < s.minimum_essential_floor:
                    robust = False
                if s.max_recovery_steps is not None:
                    if r.recovery_steps is None or r.recovery_steps > s.max_recovery_steps:
                        robust = False
        vectors.append(RobustVector(
            a.architecture_id,
            robust,
            min(floors),
            max(deficits),
            max(recovery),
            a.seed_mass,
            a.reserve_mass,
            a.blocker_import_mass,
            a.productive_closure,
        ))

    robust_vectors = [v for v in vectors if v.robust]
    output = []
    for v in vectors:
        if not v.robust:
            classification = "NotRobust"
        elif any(
            _dominates(other, v)
            for other in robust_vectors
            if other.architecture_id != v.architecture_id
        ):
            classification = "RobustDominated"
        else:
            classification = "RobustPareto"
        output.append(RobustVector(**{**v.__dict__, "classification": classification}))
    return output

def self_test():
    architectures = [
        Architecture("bulk", 40, 2, 4, 0.35),
        Architecture("buffered", 48, 10, 4, 0.45),
        Architecture("diverse", 50, 8, 2, 0.70),
        Architecture("dominated", 55, 12, 5, 0.60),
    ]
    scenarios = [
        ScenarioSpec("nominal", True, 1.0, 0),
        ScenarioSpec("power_loss", True, 1.0, 2),
        ScenarioSpec("common_mode", True, 1.0, 3),
    ]
    results = []
    add = lambda a, s, f, d, r: results.append(ScenarioResult(a, s, f, d, r))

    add("bulk", "nominal", 1, 0, 0)
    add("bulk", "power_loss", 0.8, 40, 3)
    add("bulk", "common_mode", 0.0, 70, None)

    add("buffered", "nominal", 1, 0, 0)
    add("buffered", "power_loss", 1, 18, 1)
    add("buffered", "common_mode", 1, 25, 2)

    add("diverse", "nominal", 1, 0, 0)
    add("diverse", "power_loss", 1, 20, 1)
    add("diverse", "common_mode", 1, 10, 1)

    add("dominated", "nominal", 1, 0, 0)
    add("dominated", "power_loss", 1, 25, 2)
    add("dominated", "common_mode", 1, 30, 3)

    output = {v.architecture_id: v for v in evaluate(architectures, scenarios, results)}
    assert output["bulk"].classification == "NotRobust"
    assert output["buffered"].classification == "RobustPareto"
    assert output["diverse"].classification == "RobustPareto"
    assert output["dominated"].classification == "RobustDominated"

    nominal_only = [ScenarioSpec("nominal", True, 1.0, 0)]
    nominal_results = [r for r in results if r.scenario_id == "nominal"]
    before = {v.architecture_id: v.robust for v in evaluate(architectures, nominal_only, nominal_results)}
    after = {v.architecture_id: v.robust for v in evaluate(architectures, scenarios, results)}
    for aid in after:
        if not before[aid]:
            assert not after[aid]

    relaxed = [
        ScenarioSpec("nominal", True, 0.9, 0),
        ScenarioSpec("power_loss", True, 0.8, 3),
        ScenarioSpec("common_mode", True, 0.0, None),
    ]
    relaxed_set = {v.architecture_id for v in evaluate(architectures, relaxed, results) if v.robust}
    strict_set = {v.architecture_id for v in evaluate(architectures, scenarios, results) if v.robust}
    assert strict_set <= relaxed_set

    try:
        evaluate(architectures, scenarios, results[:-1])
    except ValueError as exc:
        assert "every architecture" in str(exc)
    else:
        raise AssertionError("missing scenario evidence must fail closed")

if __name__ == "__main__":
    self_test()
    print("ok")
