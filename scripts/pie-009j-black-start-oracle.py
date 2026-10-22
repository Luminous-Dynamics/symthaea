#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
import math

EPS = 1e-12

@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    essential_minimum: float

@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    island_id: str
    startup_energy: float
    generation_per_step: float
    load_per_step: float
    requires_live_bus: bool
    requires_operational: Tuple[str, ...] = ()
    startup_consumables: Tuple[Tuple[str, int], ...] = ()
    services: Tuple[Tuple[str, float], ...] = ()
    productive_core: bool = False

@dataclass(frozen=True)
class StartStep:
    start_targets: Tuple[str, ...] = ()

@dataclass(frozen=True)
class StepResult:
    step: int
    opening_operational: Tuple[str, ...]
    started: Tuple[str, ...]
    island_energy_end: Tuple[Tuple[str, float], ...]
    service_output: Tuple[Tuple[str, float], ...]
    essential_restored: bool
    productive_core_restored: bool

@dataclass(frozen=True)
class CampaignSummary:
    steps: Tuple[StepResult, ...]
    essential_restore_step: Optional[int]
    productive_core_restore_step: Optional[int]
    final_operational: Tuple[str, ...]
    final_energy: Tuple[Tuple[str, float], ...]
    final_consumables: Tuple[Tuple[str, int], ...]


def _finite_nonnegative(x: float, name: str) -> None:
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate_model(services: Iterable[ServiceSpec], components: Iterable[ComponentSpec]) -> None:
    services = tuple(services)
    components = tuple(components)
    if not services:
        raise ValueError("at least one service required")
    sids = set()
    for s in services:
        if not s.service_id or s.service_id in sids:
            raise ValueError("service ids must be unique and nonempty")
        sids.add(s.service_id)
        _finite_nonnegative(s.essential_minimum, "essential_minimum")

    cids = set()
    for c in components:
        if not c.component_id or c.component_id in cids:
            raise ValueError("component ids must be unique and nonempty")
        cids.add(c.component_id)
        if not c.island_id:
            raise ValueError("island id must be nonempty")
        for value, name in [
            (c.startup_energy, "startup_energy"),
            (c.generation_per_step, "generation_per_step"),
            (c.load_per_step, "load_per_step"),
        ]:
            _finite_nonnegative(value, name)
        if c.generation_per_step > 0 and c.load_per_step > 0:
            raise ValueError("component must not simultaneously be source and load in this reference")
        if len(set(c.requires_operational)) != len(c.requires_operational):
            raise ValueError("duplicate operational prerequisite")
        seen_cons = set()
        for item, count in c.startup_consumables:
            if not item or item in seen_cons or count < 0:
                raise ValueError("invalid startup consumable")
            seen_cons.add(item)
        seen_services = set()
        for sid, qty in c.services:
            if sid not in sids or sid in seen_services:
                raise ValueError("invalid service reference")
            _finite_nonnegative(qty, "service output")
            seen_services.add(sid)

    for c in components:
        for dep in c.requires_operational:
            if dep not in cids:
                raise ValueError("unknown operational prerequisite")


def run_black_start(
    services,
    components,
    initial_operational,
    initial_energy_by_island: Dict[str, float],
    initial_consumables: Dict[str, int],
    plan,
) -> CampaignSummary:
    services = tuple(services)
    components = tuple(components)
    plan = tuple(plan)
    validate_model(services, components)
    comp = {c.component_id: c for c in components}
    operational = set(initial_operational)
    if operational - set(comp):
        raise ValueError("unknown initial operational component")
    islands = {c.island_id for c in components}
    energy = {i: 0.0 for i in islands}
    for island, value in initial_energy_by_island.items():
        if island not in islands:
            raise ValueError("energy provided for unknown island")
        _finite_nonnegative(value, "initial island energy")
        energy[island] = value
    consumables = dict(initial_consumables)
    for item, count in consumables.items():
        if not item or count < 0:
            raise ValueError("invalid initial consumable inventory")

    productive_ids = {c.component_id for c in components if c.productive_core}
    essential_step = None
    productive_step = None
    results = []

    def snapshot_services(opening):
        out = {s.service_id: 0.0 for s in services}
        for cid in opening:
            for sid, qty in comp[cid].services:
                out[sid] += qty
        essential = all(out[s.service_id] + EPS >= s.essential_minimum for s in services)
        productive = productive_ids.issubset(opening)
        return out, essential, productive

    for idx, step in enumerate(plan):
        if len(set(step.start_targets)) != len(step.start_targets):
            raise ValueError("duplicate start target")
        opening = set(operational)
        service_out, essential, productive = snapshot_services(opening)
        if essential and essential_step is None:
            essential_step = idx
        if productive and productive_step is None:
            productive_step = idx

        for island in islands:
            generation = sum(comp[cid].generation_per_step for cid in opening if comp[cid].island_id == island)
            load = sum(comp[cid].load_per_step for cid in opening if comp[cid].island_id == island)
            energy[island] += generation
            if load > energy[island] + EPS:
                raise ValueError(f"island {island} cannot sustain opening operational load")
            energy[island] -= load

        staged_energy = dict(energy)
        staged_consumables = dict(consumables)
        started = []
        for cid in step.start_targets:
            if cid not in comp:
                raise ValueError("unknown start target")
            if cid in opening:
                raise ValueError("cannot start already operational component")
            c = comp[cid]
            if any(dep not in opening for dep in c.requires_operational):
                raise ValueError("startup prerequisite is not opening-operational")

            opening_island = [comp[x] for x in opening if comp[x].island_id == c.island_id]
            live_bus = staged_energy[c.island_id] > EPS or any(x.generation_per_step > 0 for x in opening_island)
            if c.requires_live_bus and not live_bus:
                raise ValueError("component requires an already live island bus")
            if c.startup_energy > staged_energy[c.island_id] + EPS:
                raise ValueError("insufficient island startup energy")
            for item, count in c.startup_consumables:
                if staged_consumables.get(item, 0) < count:
                    raise ValueError("insufficient startup consumable")

            staged_energy[c.island_id] -= c.startup_energy
            for item, count in c.startup_consumables:
                staged_consumables[item] = staged_consumables.get(item, 0) - count
            started.append(cid)

        energy = staged_energy
        consumables = staged_consumables
        operational.update(started)
        results.append(StepResult(
            idx,
            tuple(sorted(opening)),
            tuple(started),
            tuple(sorted(energy.items())),
            tuple(sorted(service_out.items())),
            essential,
            productive,
        ))

    _, final_essential, final_productive = snapshot_services(operational)
    boundary_step = len(plan)
    if final_essential and essential_step is None:
        essential_step = boundary_step
    if final_productive and productive_step is None:
        productive_step = boundary_step

    return CampaignSummary(
        tuple(results),
        essential_step,
        productive_step,
        tuple(sorted(operational)),
        tuple(sorted(energy.items())),
        tuple(sorted(consumables.items())),
    )


def self_test() -> None:
    services = [
        ServiceSpec("life_support", 1.0),
        ServiceSpec("water", 1.0),
    ]
    components = [
        ComponentSpec("control", "hab", 1, 0, 0, False),
        ComponentSpec("gen", "hab", 2, 10, 0, True, ("control",), (("igniter", 1),)),
        ComponentSpec("life", "hab", 1, 0, 3, True, ("control",), services=(("life_support", 1.0),)),
        ComponentSpec("water", "hab", 1, 0, 2, True, ("control",), services=(("water", 1.0),)),
        ComponentSpec("shop", "ind", 2, 0, 2, True, productive_core=True),
        ComponentSpec("metrology", "ind", 1, 0, 1, True, ("shop",), productive_core=True),
        ComponentSpec("ind_gen", "ind", 2, 8, 0, False),
    ]

    nominal = run_black_start(
        services, components, [], {"hab": 5, "ind": 2}, {"igniter": 1},
        [
            StartStep(("control", "ind_gen")),
            StartStep(("gen",)),
            StartStep(("life", "water", "shop")),
            StartStep(("metrology",)),
        ],
    )
    assert nominal.essential_restore_step == 3
    assert nominal.productive_core_restore_step == 4
    assert set(nominal.final_operational) == {c.component_id for c in components}

    try:
        run_black_start(
            services,
            [
                ComponentSpec("g", "x", 1, 5, 0, False),
                ComponentSpec("load", "x", 1, 0, 1, True),
            ],
            [], {"x": 1}, {}, [StartStep(("g", "load"))],
        )
        raise AssertionError("same-step bootstrap should fail")
    except ValueError as e:
        assert "live island bus" in str(e) or "startup energy" in str(e)

    try:
        run_black_start(
            services,
            [ComponentSpec("g", "x", 1, 5, 0, False)],
            [], {"x": 0}, {}, [StartStep(("g",))],
        )
        raise AssertionError("zero-energy startup should fail")
    except ValueError as e:
        assert "startup energy" in str(e)

    cyc = [
        ComponentSpec("a", "x", 0, 0, 0, False, ("b",)),
        ComponentSpec("b", "x", 0, 0, 0, False, ("a",)),
    ]
    try:
        run_black_start(services, cyc, [], {"x": 10}, {}, [StartStep(("a",))])
        raise AssertionError("dependency cycle should not self-start")
    except ValueError as e:
        assert "prerequisite" in str(e)

    cons = [
        ComponentSpec("g1", "x", 1, 2, 0, False, startup_consumables=(("ign", 1),)),
        ComponentSpec("g2", "x", 1, 2, 0, False, startup_consumables=(("ign", 1),)),
    ]
    try:
        run_black_start(services, cons, [], {"x": 5}, {"ign": 1}, [StartStep(("g1", "g2"))])
        raise AssertionError("consumable double-spend should fail")
    except ValueError as e:
        assert "consumable" in str(e)

    try:
        run_black_start(
            services,
            [ComponentSpec("bload", "B", 1, 0, 1, True)],
            [], {"B": 0}, {}, [StartStep(("bload",))],
        )
        raise AssertionError("cross-island magic energy should fail")
    except ValueError as e:
        assert "live island bus" in str(e) or "startup energy" in str(e)

    partial = run_black_start(
        services,
        [
            ComponentSpec("g", "x", 1, 10, 0, False),
            ComponentSpec("life", "x", 1, 0, 2, True, services=(("life_support", 1),)),
            ComponentSpec("water", "x", 1, 0, 2, True, services=(("water", 1),)),
            ComponentSpec("shop", "x", 1, 0, 2, True, productive_core=True),
        ],
        [], {"x": 3}, {}, [StartStep(("g",)), StartStep(("life", "water"))],
    )
    assert partial.essential_restore_step == 2
    assert partial.productive_core_restore_step is None

    try:
        run_black_start(
            services,
            [ComponentSpec("bad", "x", 0, 0, 0, False, ("missing",))],
            [], {"x": 0}, {}, [],
        )
        raise AssertionError("unknown dependency should fail")
    except ValueError as e:
        assert "unknown operational prerequisite" in str(e)

    print("ok")

if __name__ == "__main__":
    self_test()
