#!/usr/bin/env python3
"""PIE-009Q independent calibration-chain / metrology-closure oracle."""
from dataclasses import dataclass
from typing import Tuple, FrozenSet

@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    product: str
    prerequisites: Tuple[str, ...]

@dataclass(frozen=True)
class CalibrationTarget:
    sensor_id: str
    calibration_capability: str

def validate(recipes, targets, all_nodes):
    recipe_ids=set()
    for r in recipes:
        if not r.recipe_id or r.recipe_id in recipe_ids:
            raise ValueError("duplicate or empty recipe id")
        recipe_ids.add(r.recipe_id)
        if r.product not in all_nodes or any(p not in all_nodes for p in r.prerequisites):
            raise ValueError("unknown node")
        if r.product in r.prerequisites:
            raise ValueError("direct self dependency")
    sensor_ids=set()
    for t in targets:
        if not t.sensor_id or t.sensor_id in sensor_ids:
            raise ValueError("duplicate or empty sensor id")
        sensor_ids.add(t.sensor_id)
        if t.calibration_capability not in all_nodes:
            raise ValueError("unknown calibration target")

def closure(seeds: FrozenSet[str], recipes, all_nodes):
    if not seeds <= all_nodes:
        raise ValueError("unknown seed capability")
    have=set(seeds)
    changed=True
    while changed:
        changed=False
        for r in recipes:
            if r.product not in have and set(r.prerequisites) <= have:
                have.add(r.product)
                changed=True
    return frozenset(have)

def classify(targets, operational, local):
    out={}
    for t in targets:
        if t.calibration_capability in local:
            out[t.sensor_id]="LocallyRenewable"
        elif t.calibration_capability in operational:
            out[t.sensor_id]="ImportDependent"
        else:
            out[t.sensor_id]="Unavailable"
    return out

def self_test():
    nodes=frozenset({
        "bench", "machine_shop", "local_electrical_standard",
        "import_pressure_reference", "optical_fixture",
        "temp_cal", "pressure_cal", "optical_cal",
        "cycle_a", "cycle_b",
    })
    recipes=[
        Recipe("temp-route", "temp_cal", ("bench", "local_electrical_standard")),
        Recipe("pressure-route", "pressure_cal", ("bench", "import_pressure_reference")),
        Recipe("optical-route", "optical_cal", ("bench", "optical_fixture")),
        Recipe("cycle-a", "cycle_a", ("cycle_b",)),
        Recipe("cycle-b", "cycle_b", ("cycle_a",)),
    ]
    targets=[
        CalibrationTarget("temp_sensor", "temp_cal"),
        CalibrationTarget("pressure_sensor", "pressure_cal"),
        CalibrationTarget("optical_sensor", "optical_cal"),
    ]
    validate(recipes, targets, nodes)

    operational=closure(
        frozenset({"bench", "machine_shop", "local_electrical_standard", "import_pressure_reference"}),
        recipes, nodes,
    )
    local=closure(
        frozenset({"bench", "machine_shop", "local_electrical_standard"}),
        recipes, nodes,
    )
    result=classify(targets, operational, local)
    assert result == {
        "temp_sensor": "LocallyRenewable",
        "pressure_sensor": "ImportDependent",
        "optical_sensor": "Unavailable",
    }
    assert "cycle_a" not in operational and "cycle_b" not in operational

    # Adding an explicit local route for the pressure reference promotes closure.
    recipes2=recipes + [
        Recipe("local-pressure-ref", "import_pressure_reference", ("machine_shop", "local_electrical_standard"))
    ]
    validate(recipes2, targets, nodes)
    local2=closure(frozenset({"bench", "machine_shop", "local_electrical_standard"}), recipes2, nodes)
    operational2=closure(
        frozenset({"bench", "machine_shop", "local_electrical_standard", "import_pressure_reference"}),
        recipes2, nodes,
    )
    assert classify(targets, operational2, local2)["pressure_sensor"] == "LocallyRenewable"

    # Removing a local standard cannot improve local metrology closure.
    weaker=closure(frozenset({"bench", "machine_shop"}), recipes, nodes)
    assert "temp_cal" not in weaker

    # Direct self-calibration is malformed and fails closed.
    try:
        validate(recipes + [Recipe("bad", "optical_fixture", ("optical_fixture",))], targets, nodes)
        raise AssertionError("direct self-dependency accepted")
    except ValueError:
        pass

    # Unknown prerequisites fail closed.
    try:
        validate(recipes + [Recipe("bad2", "temp_cal", ("missing_ref",))], targets, nodes)
        raise AssertionError("unknown prerequisite accepted")
    except ValueError:
        pass

    print("ok")

if __name__ == "__main__":
    self_test()
