#!/usr/bin/env python3
"""PIE-009L independent partially observed black-start planning oracle.

Synthetic structural reference only. No Moon/Mars mission, reliability, or
hardware-control claim is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable


class Tri(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class ActionKind(str, Enum):
    PROBE_GENERATOR = "probe_generator"
    PROBE_TIE = "probe_tie"
    START_GENERATOR = "start_generator"
    CLOSE_TIE = "close_tie"
    START_WATER = "start_water"
    WAIT = "wait"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class HiddenWorld:
    generator_healthy: bool
    tie_closed: bool
    generator_sensor_healthy: bool
    tie_sensor_healthy: bool


@dataclass(frozen=True)
class Observation:
    generator_healthy: Tri
    tie_closed: Tri
    topology_version: int


@dataclass(frozen=True)
class BeliefState:
    worlds: FrozenSet[HiddenWorld]
    topology_version: int


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    expected_topology_version: int
    reason: str


def _validate_worlds(worlds: Iterable[HiddenWorld]) -> FrozenSet[HiddenWorld]:
    frozen = frozenset(worlds)
    if not frozen:
        raise ValueError("belief must contain at least one hidden world")
    return frozen


def initial_belief(worlds: Iterable[HiddenWorld], topology_version: int = 0) -> BeliefState:
    if topology_version < 0:
        raise ValueError("topology_version must be nonnegative")
    return BeliefState(_validate_worlds(worlds), topology_version)


def observe(
    world: HiddenWorld,
    topology_version: int,
    *,
    probe_generator: bool = False,
    probe_tie: bool = False,
) -> Observation:
    if topology_version < 0:
        raise ValueError("topology_version must be nonnegative")
    generator = Tri.UNKNOWN
    tie = Tri.UNKNOWN
    if probe_generator and world.generator_sensor_healthy:
        generator = Tri.YES if world.generator_healthy else Tri.NO
    if probe_tie and world.tie_sensor_healthy:
        tie = Tri.YES if world.tie_closed else Tri.NO
    return Observation(generator, tie, topology_version)


def update_belief(belief: BeliefState, observation: Observation) -> BeliefState:
    if observation.topology_version < belief.topology_version:
        raise ValueError("stale observation")
    compatible = []
    for world in belief.worlds:
        if observation.generator_healthy is not Tri.UNKNOWN:
            expected = Tri.YES if world.generator_healthy else Tri.NO
            if observation.generator_healthy is not expected:
                continue
        if observation.tie_closed is not Tri.UNKNOWN:
            expected = Tri.YES if world.tie_closed else Tri.NO
            if observation.tie_closed is not expected:
                continue
        compatible.append(world)
    if not compatible:
        raise ValueError("observation contradicts all admissible hidden worlds")
    return BeliefState(frozenset(compatible), observation.topology_version)


def _safe_in_world(action: ActionKind, world: HiddenWorld, operational: FrozenSet[str]) -> bool:
    if action is ActionKind.START_GENERATOR:
        return "controls" in operational and world.generator_healthy
    if action is ActionKind.CLOSE_TIE:
        return "controls" in operational and not world.tie_closed
    if action is ActionKind.START_WATER:
        return "generator" in operational and world.tie_closed
    if action in (
        ActionKind.PROBE_GENERATOR,
        ActionKind.PROBE_TIE,
        ActionKind.WAIT,
        ActionKind.BLOCKED,
    ):
        return True
    raise ValueError("unknown action")


def universally_safe(action: Action, belief: BeliefState, operational: FrozenSet[str]) -> bool:
    if action.expected_topology_version != belief.topology_version:
        return False
    return all(_safe_in_world(action.kind, world, operational) for world in belief.worlds)


def _values(belief: BeliefState, attr: str) -> FrozenSet[bool]:
    return frozenset(getattr(world, attr) for world in belief.worlds)


def plan_one_step(belief: BeliefState, operational: FrozenSet[str]) -> Action:
    """Choose one conservative local action, then require re-observation/replanning."""
    version = belief.topology_version

    if "water" in operational:
        return Action(ActionKind.WAIT, version, "essential water already restored")

    if "controls" not in operational:
        return Action(ActionKind.BLOCKED, version, "controls unavailable")

    if "generator" not in operational:
        candidate = Action(ActionKind.START_GENERATOR, version, "generator known safe to start")
        if universally_safe(candidate, belief, operational):
            return candidate
        health = _values(belief, "generator_healthy")
        if len(health) > 1:
            return Action(ActionKind.PROBE_GENERATOR, version, "generator health unresolved")
        return Action(ActionKind.BLOCKED, version, "generator known unavailable")

    candidate = Action(ActionKind.START_WATER, version, "energized route known")
    if universally_safe(candidate, belief, operational):
        return candidate

    tie = _values(belief, "tie_closed")
    if len(tie) > 1:
        return Action(ActionKind.PROBE_TIE, version, "tie state unresolved")
    if tie == frozenset({False}):
        close = Action(ActionKind.CLOSE_TIE, version, "tie known open")
        if universally_safe(close, belief, operational):
            return close
    return Action(ActionKind.BLOCKED, version, "no universally safe water-restoration action")


def apply_known_action(
    action: Action,
    belief: BeliefState,
    operational: FrozenSet[str],
) -> FrozenSet[str]:
    """Apply only actions already proven universally safe in the current belief."""
    if not universally_safe(action, belief, operational):
        raise ValueError("action is not safe in every admissible hidden world")
    mutable = set(operational)
    if action.kind is ActionKind.START_GENERATOR:
        mutable.add("generator")
    elif action.kind is ActionKind.START_WATER:
        mutable.add("water")
    return frozenset(mutable)


def self_test() -> None:
    healthy_open = HiddenWorld(True, False, True, True)
    failed_open = HiddenWorld(False, False, True, True)

    # 1) Uncertain generator health requires diagnosis, not a guessed start.
    belief = initial_belief([healthy_open, failed_open])
    operational = frozenset({"controls"})
    first = plan_one_step(belief, operational)
    assert first.kind is ActionKind.PROBE_GENERATOR
    unsafe = Action(ActionKind.START_GENERATOR, belief.topology_version, "test")
    assert not universally_safe(unsafe, belief, operational)

    # 2) Healthy measurement collapses belief and permits a start.
    obs = observe(healthy_open, 0, probe_generator=True)
    healthy_belief = update_belief(belief, obs)
    assert len(healthy_belief.worlds) == 1
    start = plan_one_step(healthy_belief, operational)
    assert start.kind is ActionKind.START_GENERATOR
    operational = apply_known_action(start, healthy_belief, operational)
    assert "generator" in operational

    # 3) Failed sensor returns UNKNOWN and cannot create false certainty.
    blind_worlds = [
        HiddenWorld(True, False, False, True),
        HiddenWorld(False, False, False, True),
    ]
    blind = initial_belief(blind_worlds)
    blind_obs = observe(blind_worlds[0], 0, probe_generator=True)
    assert blind_obs.generator_healthy is Tri.UNKNOWN
    blind_after = update_belief(blind, blind_obs)
    assert blind_after.worlds == blind.worlds
    assert plan_one_step(blind_after, frozenset({"controls"})).kind is ActionKind.PROBE_GENERATOR

    # 4) Tie uncertainty requires diagnosis before starting water.
    tie_uncertain = initial_belief([
        HiddenWorld(True, False, True, True),
        HiddenWorld(True, True, True, True),
    ])
    op_with_gen = frozenset({"controls", "generator"})
    assert plan_one_step(tie_uncertain, op_with_gen).kind is ActionKind.PROBE_TIE

    # 5) Known-open tie produces close-tie, not water-start.
    open_obs = observe(HiddenWorld(True, False, True, True), 0, probe_tie=True)
    known_open = update_belief(tie_uncertain, open_obs)
    assert plan_one_step(known_open, op_with_gen).kind is ActionKind.CLOSE_TIE

    # 6) Known-closed tie allows water start.
    closed_obs = observe(HiddenWorld(True, True, True, True), 0, probe_tie=True)
    known_closed = update_belief(tie_uncertain, closed_obs)
    water = plan_one_step(known_closed, op_with_gen)
    assert water.kind is ActionKind.START_WATER
    assert "water" in apply_known_action(water, known_closed, op_with_gen)

    # 7) Topology-version drift makes an otherwise valid action stale.
    stale_water = Action(ActionKind.START_WATER, 0, "stale plan")
    advanced = BeliefState(known_closed.worlds, 1)
    assert not universally_safe(stale_water, advanced, op_with_gen)

    # 8) Older observations fail closed.
    try:
        update_belief(advanced, Observation(Tri.UNKNOWN, Tri.UNKNOWN, 0))
        raise AssertionError("stale observation should fail")
    except ValueError:
        pass

    # 9) Contradictory measurements fail closed instead of fabricating a world.
    only_healthy = initial_belief([healthy_open])
    try:
        update_belief(only_healthy, Observation(Tri.NO, Tri.UNKNOWN, 0))
        raise AssertionError("contradictory observation should fail")
    except ValueError:
        pass

    # 10) Identical belief states imply identical deterministic actions.
    same_a = initial_belief([healthy_open, failed_open])
    same_b = initial_belief([failed_open, healthy_open])
    assert plan_one_step(same_a, frozenset({"controls"})) == plan_one_step(
        same_b, frozenset({"controls"})
    )

    print("ok")


if __name__ == "__main__":
    self_test()
