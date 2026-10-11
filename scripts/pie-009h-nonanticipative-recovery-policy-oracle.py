#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
import copy, math

EPS = 1e-12

@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    demand: float
    essential_minimum: float

@dataclass(frozen=True)
class EquipmentSpec:
    equipment_id: str
    kind: str
    capacity: float
    service_id: Optional[str] = None
    spare_kind: str = "generic_spare"

@dataclass(frozen=True)
class FailureEvent:
    step: int
    equipment_ids: Tuple[str, ...]
    detection_delay_steps: int = 0

@dataclass(frozen=True)
class ShockScenario:
    scenario_id: str
    horizon_steps: int
    failures: Tuple[FailureEvent, ...] = ()

@dataclass(frozen=True)
class Observation:
    step: int
    service_ids: Tuple[str, ...]
    essential_service_ids: Tuple[str, ...]
    observed_failed_equipment: Tuple[str, ...]
    reserve_energy_remaining: float
    spares: Tuple[Tuple[str, int], ...]
    previous_delivery: Tuple[Tuple[str, float], ...]
    pending_repairs: Tuple[str, ...]

@dataclass(frozen=True)
class PolicyAction:
    service_priority: Tuple[str, ...]
    reserve_for_essential_only: bool
    repair_targets: Tuple[str, ...]

@dataclass(frozen=True)
class StepResult:
    step: int
    observation: Observation
    action: PolicyAction
    actual_failed_equipment: Tuple[str, ...]
    delivered: Tuple[Tuple[str, float], ...]
    essential_minimum_met: bool
    reserve_used: float
    repairs_started: Tuple[str, ...]

@dataclass(frozen=True)
class CampaignResult:
    scenario_id: str
    policy_id: str
    steps: Tuple[StepResult, ...]
    essential_floor_ratio: float
    cumulative_essential_deficit: float
    cumulative_total_deficit: float
    reserve_used: float
    repairs_started: int
    first_critical_shortfall_step: Optional[int]

class RecoveryPolicy:
    policy_id = "base"
    def decide(self, obs: Observation) -> PolicyAction:
        raise NotImplementedError

class EssentialFirstPolicy(RecoveryPolicy):
    policy_id = "essential-first"
    def decide(self, obs: Observation) -> PolicyAction:
        nonessential = tuple(s for s in obs.service_ids if s not in obs.essential_service_ids)
        priority = tuple(obs.essential_service_ids) + nonessential
        repairs = tuple(sorted(obs.observed_failed_equipment)[:1])
        return PolicyAction(priority, True, repairs)

class ProductiveCapacityPolicy(RecoveryPolicy):
    policy_id = "productive-capacity-first"
    def decide(self, obs: Observation) -> PolicyAction:
        ess = list(obs.essential_service_ids)
        rest = [s for s in obs.service_ids if s not in ess]
        ranked = sorted(rest, key=lambda s: (0 if s in ("machine_shop", "industry") else 1, s))
        repairs = tuple(sorted(obs.observed_failed_equipment, reverse=True)[:1])
        return PolicyAction(tuple(ess + ranked), False, repairs)

@dataclass
class _Runtime:
    failed: set[str] = field(default_factory=set)
    pending_repairs: set[str] = field(default_factory=set)
    reserve_energy: float = 0.0
    spares: Dict[str, int] = field(default_factory=dict)
    previous_delivery: Dict[str, float] = field(default_factory=dict)

class Model:
    def __init__(self, services: Sequence[ServiceSpec], equipment: Sequence[EquipmentSpec], initial_reserve_energy: float, initial_spares: Dict[str, int]):
        self.services = tuple(services)
        self.equipment = tuple(equipment)
        self.service_by_id = {s.service_id: s for s in self.services}
        self.equipment_by_id = {e.equipment_id: e for e in self.equipment}
        if not self.services or len(self.service_by_id) != len(self.services):
            raise ValueError("invalid services")
        if not self.equipment or len(self.equipment_by_id) != len(self.equipment):
            raise ValueError("invalid equipment")
        for s in self.services:
            if not s.service_id or not math.isfinite(s.demand) or s.demand < 0:
                raise ValueError("invalid service")
            if not math.isfinite(s.essential_minimum) or s.essential_minimum < 0 or s.essential_minimum > s.demand + EPS:
                raise ValueError("invalid essential minimum")
        for e in self.equipment:
            if not e.equipment_id or not math.isfinite(e.capacity) or e.capacity < 0:
                raise ValueError("invalid equipment")
            if e.kind not in ("generator", "service"):
                raise ValueError("invalid equipment kind")
            if e.kind == "service" and e.service_id not in self.service_by_id:
                raise ValueError("service equipment references unknown service")
            if e.kind == "generator" and e.service_id is not None:
                raise ValueError("generator cannot reference service")
            if not e.spare_kind:
                raise ValueError("spare kind required")
        if not math.isfinite(initial_reserve_energy) or initial_reserve_energy < 0:
            raise ValueError("invalid reserve")
        if any((not k) or (not isinstance(v, int)) or v < 0 for k, v in initial_spares.items()):
            raise ValueError("invalid spares")
        self.initial_reserve_energy = initial_reserve_energy
        self.initial_spares = dict(initial_spares)

    @property
    def service_ids(self):
        return tuple(s.service_id for s in self.services)

    @property
    def essential_ids(self):
        return tuple(s.service_id for s in self.services if s.essential_minimum > 0)

    def _validate_scenario(self, sc: ShockScenario) -> None:
        if not sc.scenario_id or sc.horizon_steps <= 0:
            raise ValueError("invalid scenario")
        for ev in sc.failures:
            if ev.step < 0 or ev.step >= sc.horizon_steps or ev.detection_delay_steps < 0:
                raise ValueError("invalid failure event")
            if not ev.equipment_ids:
                raise ValueError("empty failure event")
            for eq in ev.equipment_ids:
                if eq not in self.equipment_by_id:
                    raise ValueError("failure references unknown equipment")

    def _visible_failures(self, sc: ShockScenario, step: int, actual_failed: set[str]) -> Tuple[str, ...]:
        visible = set()
        for ev in sc.failures:
            if ev.step + ev.detection_delay_steps <= step:
                visible.update(ev.equipment_ids)
        visible.intersection_update(actual_failed)
        return tuple(sorted(visible))

    def _validate_action(self, action: PolicyAction) -> None:
        if len(action.service_priority) != len(self.service_ids) or set(action.service_priority) != set(self.service_ids):
            raise ValueError("policy priority must contain every service exactly once")
        if len(action.repair_targets) != len(set(action.repair_targets)):
            raise ValueError("duplicate repair target")
        for eq in action.repair_targets:
            if eq not in self.equipment_by_id:
                raise ValueError("unknown repair target")

    def run(self, sc: ShockScenario, policy: RecoveryPolicy) -> CampaignResult:
        self._validate_scenario(sc)
        rt = _Runtime(
            reserve_energy=self.initial_reserve_energy,
            spares=copy.deepcopy(self.initial_spares),
            previous_delivery={sid: self.service_by_id[sid].demand for sid in self.service_ids},
        )
        steps = []
        total_reserve = 0.0
        total_repairs = 0
        cumulative_essential_deficit = 0.0
        cumulative_total_deficit = 0.0
        floor = 1.0
        first_short = None

        for step in range(sc.horizon_steps):
            if rt.pending_repairs:
                rt.failed.difference_update(rt.pending_repairs)
                rt.pending_repairs.clear()

            for ev in sc.failures:
                if ev.step == step:
                    rt.failed.update(ev.equipment_ids)

            obs = Observation(
                step,
                self.service_ids,
                self.essential_ids,
                self._visible_failures(sc, step, rt.failed),
                rt.reserve_energy,
                tuple(sorted(rt.spares.items())),
                tuple(sorted(rt.previous_delivery.items())),
                tuple(sorted(rt.pending_repairs)),
            )

            # Non-anticipation boundary: the policy receives only obs, never sc or future events.
            action = policy.decide(obs)
            self._validate_action(action)

            repairs_started = []
            for eq_id in action.repair_targets:
                if eq_id not in rt.failed:
                    continue
                spec = self.equipment_by_id[eq_id]
                have = rt.spares.get(spec.spare_kind, 0)
                if have <= 0:
                    continue
                rt.spares[spec.spare_kind] = have - 1
                rt.pending_repairs.add(eq_id)
                repairs_started.append(eq_id)
            total_repairs += len(repairs_started)

            generation = sum(e.capacity for e in self.equipment if e.kind == "generator" and e.equipment_id not in rt.failed)
            service_capacity = {
                sid: sum(e.capacity for e in self.equipment if e.kind == "service" and e.service_id == sid and e.equipment_id not in rt.failed)
                for sid in self.service_ids
            }

            delivered = {sid: 0.0 for sid in self.service_ids}
            remaining_generation = generation
            reserve_used = 0.0
            for sid in action.service_priority:
                spec = self.service_by_id[sid]
                max_possible = min(spec.demand, service_capacity[sid])
                give = min(max_possible, remaining_generation)
                delivered[sid] = give
                remaining_generation -= give
                shortage = max_possible - give
                if shortage > EPS and rt.reserve_energy > EPS:
                    if (not action.reserve_for_essential_only) or spec.essential_minimum > 0:
                        r = min(shortage, rt.reserve_energy)
                        delivered[sid] += r
                        rt.reserve_energy -= r
                        reserve_used += r
            total_reserve += reserve_used

            essential_ok = True
            for sid in self.service_ids:
                spec = self.service_by_id[sid]
                d = delivered[sid]
                cumulative_total_deficit += max(0.0, spec.demand - d)
                if spec.essential_minimum > 0:
                    cumulative_essential_deficit += max(0.0, spec.essential_minimum - d)
                    floor = min(floor, d / spec.essential_minimum)
                    if d + EPS < spec.essential_minimum:
                        essential_ok = False
            if not essential_ok and first_short is None:
                first_short = step

            rt.previous_delivery = delivered
            steps.append(StepResult(
                step,
                obs,
                action,
                tuple(sorted(rt.failed)),
                tuple(sorted(delivered.items())),
                essential_ok,
                reserve_used,
                tuple(sorted(repairs_started)),
            ))

        return CampaignResult(
            sc.scenario_id,
            policy.policy_id,
            tuple(steps),
            floor,
            cumulative_essential_deficit,
            cumulative_total_deficit,
            total_reserve,
            total_repairs,
            first_short,
        )

def action_prefix(result: CampaignResult, through_step: int):
    return tuple(s.action for s in result.steps if s.step <= through_step)

def observation_prefix(result: CampaignResult, through_step: int):
    return tuple(s.observation for s in result.steps if s.step <= through_step)

def assert_nonanticipative_prefix(a: CampaignResult, b: CampaignResult, through_step: int) -> None:
    if observation_prefix(a, through_step) != observation_prefix(b, through_step):
        raise AssertionError("scenarios are not observationally indistinguishable through requested step")
    if action_prefix(a, through_step) != action_prefix(b, through_step):
        raise AssertionError("anticipatory action detected")

def policy_vector(results):
    if not results:
        raise ValueError("at least one result required")
    shortfalls = [r.first_critical_shortfall_step for r in results if r.first_critical_shortfall_step is not None]
    return {
        "worst_essential_floor_ratio": min(r.essential_floor_ratio for r in results),
        "worst_cumulative_essential_deficit": max(r.cumulative_essential_deficit for r in results),
        "worst_cumulative_total_deficit": max(r.cumulative_total_deficit for r in results),
        "worst_reserve_used": max(r.reserve_used for r in results),
        "worst_repairs_started": max(r.repairs_started for r in results),
        "earliest_critical_shortfall_step": min(shortfalls) if shortfalls else None,
    }

def fixture_model(reserve=6.0, spares=2):
    return Model(
        [
            ServiceSpec("life_support", 4, 4),
            ServiceSpec("water", 3, 2),
            ServiceSpec("machine_shop", 3, 0),
            ServiceSpec("industry", 5, 0),
        ],
        [
            EquipmentSpec("gen_a", "generator", 8, spare_kind="power_spare"),
            EquipmentSpec("gen_b", "generator", 8, spare_kind="power_spare"),
            EquipmentSpec("life_unit", "service", 4, "life_support", "service_spare"),
            EquipmentSpec("water_a", "service", 2, "water", "service_spare"),
            EquipmentSpec("water_b", "service", 2, "water", "service_spare"),
            EquipmentSpec("shop_unit", "service", 3, "machine_shop", "service_spare"),
            EquipmentSpec("industry_unit", "service", 5, "industry", "service_spare"),
        ],
        reserve,
        {"power_spare": spares, "service_spare": spares},
    )

def self_test():
    m = fixture_model()
    p = EssentialFirstPolicy()
    early = (FailureEvent(1, ("gen_a",), 0),)
    s1 = ShockScenario("future-water", 5, early + (FailureEvent(3, ("water_a",), 0),))
    s2 = ShockScenario("future-industry", 5, early + (FailureEvent(3, ("industry_unit",), 0),))
    r1, r2 = m.run(s1, p), m.run(s2, p)
    assert_nonanticipative_prefix(r1, r2, 2)

    s1_clone = ShockScenario("renamed-scenario", 5, s1.failures)
    r1c = m.run(s1_clone, p)
    assert tuple(x.observation for x in r1.steps) == tuple(x.observation for x in r1c.steps)
    assert tuple(x.action for x in r1.steps) == tuple(x.action for x in r1c.steps)

    delayed = ShockScenario("delayed-detection", 4, (FailureEvent(1, ("gen_a",), 1),))
    rd = m.run(delayed, p)
    assert "gen_a" in rd.steps[1].actual_failed_equipment
    assert "gen_a" not in rd.steps[1].observation.observed_failed_equipment
    assert "gen_a" not in rd.steps[1].repairs_started
    assert "gen_a" in rd.steps[2].observation.observed_failed_equipment
    assert "gen_a" in rd.steps[2].repairs_started
    assert "gen_a" in rd.steps[2].actual_failed_equipment
    assert "gen_a" not in rd.steps[3].actual_failed_equipment

    prod = ProductiveCapacityPolicy()
    rp = m.run(delayed, prod)
    assert rp.steps[0].action != rd.steps[0].action
    assert prod.decide(rd.steps[0].observation) == prod.decide(rd.steps[0].observation)

    low = fixture_model(reserve=0).run(delayed, p)
    high = fixture_model(reserve=10).run(delayed, p)
    assert high.essential_floor_ratio + EPS >= low.essential_floor_ratio

    no_future = ShockScenario("no-future", 5, early)
    with_future = ShockScenario("with-future", 5, early + (FailureEvent(4, ("life_unit",), 0),))
    assert_nonanticipative_prefix(m.run(no_future, p), m.run(with_future, p), 3)

    class BadPolicy(RecoveryPolicy):
        policy_id = "bad"
        def decide(self, obs):
            return PolicyAction(("life_support",), True, ())
    try:
        m.run(ShockScenario("bad", 1), BadPolicy())
    except ValueError:
        pass
    else:
        raise AssertionError("malformed policy priority must fail")

    try:
        m.run(ShockScenario("bad-ref", 1, (FailureEvent(0, ("missing",), 0),)), p)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown equipment reference must fail")

    vec = policy_vector([m.run(s, p) for s in (s1, s2, delayed)])
    assert set(vec) == {
        "worst_essential_floor_ratio",
        "worst_cumulative_essential_deficit",
        "worst_cumulative_total_deficit",
        "worst_reserve_used",
        "worst_repairs_started",
        "earliest_critical_shortfall_step",
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
    else:
        parser.error("--self-test required")
