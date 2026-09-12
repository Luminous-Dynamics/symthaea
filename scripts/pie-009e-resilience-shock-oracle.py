#!/usr/bin/env python3
"""PIE-009E independent resilience / shock-campaign oracle.

Synthetic, deterministic reference semantics only. No Moon/Mars reliability,
probability, economics, or operational-control claim is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    nominal_output: float
    essential_minimum: float
    power_per_output: float
    priority: int

@dataclass(frozen=True)
class MachineSpec:
    machine_id: str
    service_id: str
    capacity: float
    common_mode_group: Optional[str]
    repair_spare_id: Optional[str]
    repair_steps: int

@dataclass(frozen=True)
class ShockStep:
    external_power: float
    reserve_dispatch: float = 0.0
    failed_machine_ids: Tuple[str, ...] = ()
    failed_groups: Tuple[str, ...] = ()
    site_factors: Tuple[Tuple[str, float], ...] = ()
    transport_open: bool = True
    repair_targets: Tuple[str, ...] = ()
    scheduled_spare_imports: Tuple[Tuple[str, int], ...] = ()

@dataclass
class MachineState:
    failed: bool = False
    repair_remaining: int = 0

@dataclass(frozen=True)
class StepResult:
    step: int
    delivered: Dict[str, float]
    essential_met: Dict[str, bool]
    reserve_used: float
    missed_imports: Dict[str, int]
    started_repairs: Tuple[str, ...]
    completed_repairs: Tuple[str, ...]
    failed_machines: Tuple[str, ...]

@dataclass(frozen=True)
class CampaignSummary:
    steps: int
    essential_floor_ratio: float
    first_critical_shortfall_step: Optional[int]
    cumulative_nominal_deficit: float
    cumulative_nonessential_deficit: float
    reserve_energy_used: float
    missed_import_units: int
    final_failed_machines: Tuple[str, ...]
    recovery_step: Optional[int]
    step_results: Tuple[StepResult, ...]

@dataclass(frozen=True)
class ScenarioSetSummary:
    scenarios: int
    worst_essential_floor_ratio: float
    earliest_critical_shortfall_step: Optional[int]
    worst_cumulative_nominal_deficit: float
    max_missed_import_units: int
    all_scenarios_preserve_essential: bool

def _finite_nonnegative(x: float, name: str) -> None:
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and nonnegative")

def validate_model(services: List[ServiceSpec], machines: List[MachineSpec]) -> None:
    if not services:
        raise ValueError("at least one service required")
    ids, priorities = set(), set()
    for s in services:
        if not s.service_id or s.service_id in ids:
            raise ValueError("service ids must be unique and nonempty")
        ids.add(s.service_id)
        if s.priority in priorities:
            raise ValueError("service priorities must be unique and explicit")
        priorities.add(s.priority)
        for x, n in [
            (s.nominal_output, "nominal_output"),
            (s.essential_minimum, "essential_minimum"),
            (s.power_per_output, "power_per_output"),
        ]:
            _finite_nonnegative(x, n)
        if s.essential_minimum > s.nominal_output:
            raise ValueError("essential minimum cannot exceed nominal output")
        if s.nominal_output > 0 and s.power_per_output <= 0:
            raise ValueError("positive-output service needs positive power_per_output")
    mids = set()
    for m in machines:
        if not m.machine_id or m.machine_id in mids:
            raise ValueError("machine ids must be unique and nonempty")
        mids.add(m.machine_id)
        if m.service_id not in ids:
            raise ValueError("machine references unknown service")
        _finite_nonnegative(m.capacity, "machine capacity")
        if m.repair_steps < 0:
            raise ValueError("repair_steps cannot be negative")
        if m.repair_spare_id is None and m.repair_steps > 0:
            raise ValueError("repairable machine must declare spare id")

def run_campaign(services, machines, initial_reserve_energy, initial_spares, steps):
    validate_model(services, machines)
    _finite_nonnegative(initial_reserve_energy, "initial reserve energy")
    service_ids = {s.service_id for s in services}
    machine_by_id = {m.machine_id: m for m in machines}
    state = {m.machine_id: MachineState() for m in machines}
    spares = dict(initial_spares)
    reserve = initial_reserve_energy
    results = []
    first = None
    min_floor = math.inf
    total_def = 0.0
    nonessential_def = 0.0
    reserve_used = 0.0
    missed_total = 0
    ordered = sorted(services, key=lambda s: s.priority)

    for idx, st in enumerate(steps):
        _finite_nonnegative(st.external_power, "external_power")
        _finite_nonnegative(st.reserve_dispatch, "reserve_dispatch")
        if st.reserve_dispatch > reserve + 1e-12:
            raise ValueError("reserve dispatch exceeds opening reserve inventory")

        site_factor = {sid: 1.0 for sid in service_ids}
        seen = set()
        for sid, factor in st.site_factors:
            if sid not in service_ids or sid in seen or not math.isfinite(factor) or not (0 <= factor <= 1):
                raise ValueError("invalid site factor")
            seen.add(sid)
            site_factor[sid] = factor

        missed = {}
        for spare, count in st.scheduled_spare_imports:
            if count < 0:
                raise ValueError("invalid scheduled spare import")
            if st.transport_open:
                spares[spare] = spares.get(spare, 0) + count
            else:
                missed[spare] = missed.get(spare, 0) + count
                missed_total += count

        failed_ids = set(st.failed_machine_ids)
        if failed_ids - set(machine_by_id):
            raise ValueError("failure references unknown machine")
        failed_groups = set(st.failed_groups)
        for m in machines:
            if m.machine_id in failed_ids or (
                m.common_mode_group is not None and m.common_mode_group in failed_groups
            ):
                state[m.machine_id].failed = True
                state[m.machine_id].repair_remaining = 0

        if len(set(st.repair_targets)) != len(st.repair_targets):
            raise ValueError("repair target duplicated")
        started = []
        for mid in st.repair_targets:
            if mid not in machine_by_id:
                raise ValueError("repair references unknown machine")
            ms = state[mid]
            m = machine_by_id[mid]
            if not ms.failed:
                raise ValueError("cannot repair operational machine")
            if ms.repair_remaining > 0:
                raise ValueError("machine already under repair")
            if m.repair_steps == 0:
                ms.failed = False
                started.append(mid)
                continue
            spare = m.repair_spare_id
            if spares.get(spare, 0) <= 0:
                raise ValueError("repair plan lacks required spare")
            spares[spare] -= 1
            ms.repair_remaining = m.repair_steps
            started.append(mid)

        available_power = st.external_power + st.reserve_dispatch
        reserve -= st.reserve_dispatch
        reserve_used += st.reserve_dispatch

        capacities = {s.service_id: 0.0 for s in services}
        for m in machines:
            if not state[m.machine_id].failed:
                capacities[m.service_id] += m.capacity
        for sid in capacities:
            capacities[sid] *= site_factor[sid]

        delivered = {}
        essential = {}
        remaining_power = available_power
        for s in ordered:
            max_by_power = remaining_power / s.power_per_output if s.power_per_output > 0 else 0.0
            out = max(0.0, min(s.nominal_output, capacities[s.service_id], max_by_power))
            delivered[s.service_id] = out
            remaining_power -= out * s.power_per_output
            essential[s.service_id] = out + 1e-12 >= s.essential_minimum
            if s.essential_minimum > 0:
                min_floor = min(min_floor, out / s.essential_minimum)
                if not essential[s.service_id] and first is None:
                    first = idx
            deficit = max(0.0, s.nominal_output - out)
            total_def += deficit
            if s.essential_minimum == 0:
                nonessential_def += deficit

        completed = []
        for mid, ms in state.items():
            if ms.failed and ms.repair_remaining > 0:
                ms.repair_remaining -= 1
                if ms.repair_remaining == 0:
                    ms.failed = False
                    completed.append(mid)

        results.append(StepResult(
            idx,
            delivered,
            essential,
            st.reserve_dispatch,
            missed,
            tuple(started),
            tuple(completed),
            tuple(sorted(mid for mid, ms in state.items() if ms.failed)),
        ))

    if min_floor is math.inf:
        min_floor = 1.0

    recovery = None
    if first is not None:
        for result in results[first + 1:]:
            if all(result.delivered[s.service_id] + 1e-12 >= s.nominal_output for s in services):
                recovery = result.step
                break

    return CampaignSummary(
        len(steps),
        min_floor,
        first,
        total_def,
        nonessential_def,
        reserve_used,
        missed_total,
        tuple(sorted(mid for mid, ms in state.items() if ms.failed)),
        recovery,
        tuple(results),
    )

def summarize_scenarios(summaries):
    if not summaries:
        raise ValueError("at least one scenario summary required")
    shortfalls = [
        s.first_critical_shortfall_step
        for s in summaries
        if s.first_critical_shortfall_step is not None
    ]
    return ScenarioSetSummary(
        len(summaries),
        min(s.essential_floor_ratio for s in summaries),
        min(shortfalls) if shortfalls else None,
        max(s.cumulative_nominal_deficit for s in summaries),
        max(s.missed_import_units for s in summaries),
        all(s.first_critical_shortfall_step is None for s in summaries),
    )

def self_test():
    services = [
        ServiceSpec("life_support", 30, 25, 1, 0),
        ServiceSpec("water", 25, 15, 1, 1),
        ServiceSpec("industry", 35, 0, 1, 2),
    ]
    machines = [
        MachineSpec("ls-a", "life_support", 30, "ls_common", "ctrl", 1),
        MachineSpec("water-a", "water", 15, "water_common", "pump", 1),
        MachineSpec("water-b", "water", 15, "water_common", "pump", 1),
        MachineSpec("ind-a", "industry", 20, "ind_a", "bearing", 1),
        MachineSpec("ind-b", "industry", 20, "ind_b", "bearing", 1),
    ]

    base = run_campaign(
        services, machines, 0, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(100), ShockStep(100)]
    )
    assert base.first_critical_shortfall_step is None
    assert base.cumulative_nominal_deficit == 0

    shock = run_campaign(
        services, machines, 20, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(100), ShockStep(35, 10), ShockStep(100)]
    )
    assert shock.first_critical_shortfall_step is None
    assert shock.step_results[1].delivered == {
        "life_support": 30,
        "water": 15,
        "industry": 0,
    }

    weaker = run_campaign(
        services, machines, 5, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(100), ShockStep(35, 5), ShockStep(100)]
    )
    assert weaker.essential_floor_ratio <= shock.essential_floor_ratio + 1e-12

    common_mode = run_campaign(
        services, machines, 0, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(100), ShockStep(100, failed_groups=("water_common",))]
    )
    assert common_mode.first_critical_shortfall_step == 1
    assert common_mode.step_results[1].delivered["water"] == 0

    repaired = run_campaign(
        services, machines, 0, {"ctrl": 1, "pump": 1, "bearing": 2},
        [
            ShockStep(100, failed_machine_ids=("water-a",), repair_targets=("water-a",)),
            ShockStep(100),
        ]
    )
    assert repaired.step_results[0].delivered["water"] == 15
    assert "water-a" in repaired.step_results[0].completed_repairs
    assert repaired.step_results[1].delivered["water"] == 25

    try:
        run_campaign(
            services, machines, 0, {"ctrl": 1, "pump": 0, "bearing": 2},
            [ShockStep(
                100,
                failed_machine_ids=("water-a",),
                repair_targets=("water-a",),
                transport_open=False,
                scheduled_spare_imports=(("pump", 1),),
            )]
        )
    except ValueError as exc:
        assert "lacks required spare" in str(exc)
    else:
        raise AssertionError("future/missed spare import must not be borrowable")

    site = run_campaign(
        services, machines, 0, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(100), ShockStep(100, site_factors=(("industry", 0.0),))]
    )
    assert site.first_critical_shortfall_step is None
    assert site.step_results[1].delivered["industry"] == 0

    altered_services = [
        ServiceSpec("industry", 35, 0, 1, 0),
        ServiceSpec("life_support", 30, 25, 1, 1),
        ServiceSpec("water", 25, 15, 1, 2),
    ]
    altered = run_campaign(
        altered_services, machines, 0, {"ctrl": 1, "pump": 2, "bearing": 2},
        [ShockStep(45)]
    )
    assert altered.first_critical_shortfall_step == 0
    assert altered.step_results[0].delivered["industry"] == 35

    scenario_set = summarize_scenarios([base, shock, common_mode, site])
    assert scenario_set.scenarios == 4
    assert not scenario_set.all_scenarios_preserve_essential
    assert scenario_set.earliest_critical_shortfall_step == 1
    assert scenario_set.worst_essential_floor_ratio == 0

    try:
        validate_model(
            [ServiceSpec("a", 1, 0, 1, 0), ServiceSpec("b", 1, 0, 1, 0)],
            []
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate service priorities must fail closed")

if __name__ == "__main__":
    self_test()
    print("ok")
