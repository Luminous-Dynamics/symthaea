#!/usr/bin/env python3
"""PIE-009N independent diagnosability / sensor-cut oracle.

Synthetic structural reference only. No reliability probabilities, hardware
control, electrical transient claims, or mission-design claims are made.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class World:
    world_id: str
    decision: str


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    common_mode_group: str
    outputs: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class DiagnosabilityResult:
    diagnosable: bool
    unresolved_cross_decision_pairs: Tuple[Tuple[str, str], ...]


def _powerset(items: Sequence[str]):
    for r in range(len(items) + 1):
        yield from combinations(items, r)


def validate_model(worlds: Sequence[World], sensors: Sequence[Sensor]) -> None:
    if len(worlds) < 2:
        raise ValueError("at least two worlds required")
    world_ids = [w.world_id for w in worlds]
    if any(not x for x in world_ids) or len(set(world_ids)) != len(world_ids):
        raise ValueError("world ids must be unique and nonempty")
    if any(not w.decision for w in worlds):
        raise ValueError("decision labels must be nonempty")
    if len({w.decision for w in worlds}) < 2:
        raise ValueError("at least two decision classes required")

    sensor_ids = [s.sensor_id for s in sensors]
    if any(not x for x in sensor_ids) or len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("sensor ids must be unique and nonempty")
    if any(not s.common_mode_group for s in sensors):
        raise ValueError("every sensor requires an explicit common-mode group")

    world_set = set(world_ids)
    for sensor in sensors:
        seen = set()
        for world_id, observation in sensor.outputs:
            if world_id not in world_set or world_id in seen:
                raise ValueError("sensor output references unknown/duplicate world")
            if observation == "":
                raise ValueError("use explicit UNKNOWN rather than an empty observation")
            seen.add(world_id)
        if seen != world_set:
            raise ValueError("each sensor must declare one observation for every world")


def _sensor_map(sensors: Sequence[Sensor]) -> Dict[str, Sensor]:
    return {s.sensor_id: s for s in sensors}


def diagnose(
    worlds: Sequence[World],
    sensors: Sequence[Sensor],
    active_sensor_ids: Iterable[str],
) -> DiagnosabilityResult:
    validate_model(worlds, sensors)
    sensors_by_id = _sensor_map(sensors)
    active = tuple(sorted(set(active_sensor_ids)))
    unknown = set(active) - set(sensors_by_id)
    if unknown:
        raise ValueError(f"unknown active sensors: {sorted(unknown)}")

    output_maps = {
        sensor_id: dict(sensors_by_id[sensor_id].outputs)
        for sensor_id in active
    }
    unresolved: List[Tuple[str, str]] = []
    for i, left in enumerate(worlds):
        for right in worlds[i + 1 :]:
            if left.decision == right.decision:
                continue
            indistinguishable = all(
                output_maps[sensor_id][left.world_id]
                == output_maps[sensor_id][right.world_id]
                for sensor_id in active
            )
            if indistinguishable:
                unresolved.append((left.world_id, right.world_id))
    return DiagnosabilityResult(not unresolved, tuple(unresolved))


def minimal_sufficient_sensor_sets(
    worlds: Sequence[World], sensors: Sequence[Sensor]
) -> Tuple[Tuple[str, ...], ...]:
    sensor_ids = tuple(sorted(s.sensor_id for s in sensors))
    sufficient: List[Tuple[str, ...]] = []
    for subset in _powerset(sensor_ids):
        if not diagnose(worlds, sensors, subset).diagnosable:
            continue
        subset_set = set(subset)
        if any(set(previous).issubset(subset_set) for previous in sufficient):
            continue
        sufficient.append(subset)
    return tuple(sufficient)


def minimal_sensor_cut_sets(
    worlds: Sequence[World],
    sensors: Sequence[Sensor],
    baseline_active_ids: Iterable[str],
) -> Tuple[Tuple[str, ...], ...]:
    baseline = tuple(sorted(set(baseline_active_ids)))
    if not diagnose(worlds, sensors, baseline).diagnosable:
        raise ValueError("baseline must be diagnosable")

    cuts: List[Tuple[str, ...]] = []
    for removed in _powerset(baseline):
        if not removed:
            continue
        removed_set = set(removed)
        active = tuple(x for x in baseline if x not in removed_set)
        if diagnose(worlds, sensors, active).diagnosable:
            continue
        if any(set(previous).issubset(removed_set) for previous in cuts):
            continue
        cuts.append(removed)
    return tuple(cuts)


def minimal_common_mode_cut_sets(
    worlds: Sequence[World],
    sensors: Sequence[Sensor],
    baseline_active_ids: Iterable[str],
) -> Tuple[Tuple[str, ...], ...]:
    baseline = tuple(sorted(set(baseline_active_ids)))
    if not diagnose(worlds, sensors, baseline).diagnosable:
        raise ValueError("baseline must be diagnosable")

    sensors_by_id = _sensor_map(sensors)
    unknown = set(baseline) - set(sensors_by_id)
    if unknown:
        raise ValueError("baseline references unknown sensor")
    groups = tuple(
        sorted({sensors_by_id[sensor_id].common_mode_group for sensor_id in baseline})
    )

    cuts: List[Tuple[str, ...]] = []
    for lost_groups in _powerset(groups):
        if not lost_groups:
            continue
        lost = set(lost_groups)
        active = tuple(
            sensor_id
            for sensor_id in baseline
            if sensors_by_id[sensor_id].common_mode_group not in lost
        )
        if diagnose(worlds, sensors, active).diagnosable:
            continue
        if any(set(previous).issubset(lost) for previous in cuts):
            continue
        cuts.append(lost_groups)
    return tuple(cuts)


def self_test() -> None:
    worlds = [
        World("healthy_closed", "PROCEED"),
        World("failed_closed", "BLOCK"),
        World("healthy_open", "BLOCK"),
        World("failed_open", "BLOCK"),
    ]

    # Two independent generator-health paths and two independent topology paths.
    sensors = [
        Sensor(
            "gen_current",
            "gen_electrical",
            (
                ("healthy_closed", "H"),
                ("failed_closed", "F"),
                ("healthy_open", "H"),
                ("failed_open", "F"),
            ),
        ),
        Sensor(
            "gen_vibration",
            "gen_mechanical",
            (
                ("healthy_closed", "H"),
                ("failed_closed", "F"),
                ("healthy_open", "H"),
                ("failed_open", "F"),
            ),
        ),
        Sensor(
            "tie_aux",
            "switch_aux",
            (
                ("healthy_closed", "C"),
                ("failed_closed", "C"),
                ("healthy_open", "O"),
                ("failed_open", "O"),
            ),
        ),
        Sensor(
            "tie_optical",
            "optical",
            (
                ("healthy_closed", "C"),
                ("failed_closed", "C"),
                ("healthy_open", "O"),
                ("failed_open", "O"),
            ),
        ),
        Sensor(
            "failed_diag",
            "diag_bus",
            tuple((world.world_id, "UNKNOWN") for world in worlds),
        ),
    ]
    validate_model(worlds, sensors)
    all_ids = tuple(sensor.sensor_id for sensor in sensors)

    assert diagnose(worlds, sensors, all_ids).diagnosable

    minimum_sets = set(minimal_sufficient_sensor_sets(worlds, sensors))
    assert minimum_sets == {
        ("gen_current", "tie_aux"),
        ("gen_current", "tie_optical"),
        ("gen_vibration", "tie_aux"),
        ("gen_vibration", "tie_optical"),
    }

    # UNKNOWN cannot manufacture certainty.
    unknown_result = diagnose(worlds, sensors, ["failed_diag", "tie_aux"])
    assert not unknown_result.diagnosable
    assert unknown_result.unresolved_cross_decision_pairs

    # Each individual path has redundancy in the baseline.
    for sensor_id in ("gen_current", "gen_vibration", "tie_aux", "tie_optical"):
        active = [x for x in all_ids if x != sensor_id]
        assert diagnose(worlds, sensors, active).diagnosable

    cuts = set(minimal_sensor_cut_sets(worlds, sensors, all_ids))
    assert cuts == {
        ("gen_current", "gen_vibration"),
        ("tie_aux", "tie_optical"),
    }

    group_cuts = set(minimal_common_mode_cut_sets(worlds, sensors, all_ids))
    assert ("gen_electrical", "gen_mechanical") in group_cuts
    assert ("optical", "switch_aux") in group_cuts
    assert all(len(cut) == 2 for cut in group_cuts)

    # Removing information cannot improve this decision's diagnosability.
    assert not diagnose(worlds, sensors, []).diagnosable
    assert not diagnose(worlds, sensors, ["gen_current"]).diagnosable
    assert diagnose(worlds, sensors, ["gen_current", "tie_aux"]).diagnosable

    # Malformed inputs fail closed.
    try:
        duplicate = sensors + [
            Sensor(
                "gen_current",
                "duplicate_group",
                tuple((world.world_id, "X") for world in worlds),
            )
        ]
        validate_model(worlds, duplicate)
        raise AssertionError("duplicate sensor id accepted")
    except ValueError:
        pass

    try:
        incomplete = [Sensor("only", "group", (("healthy_closed", "H"),))]
        validate_model(worlds, incomplete)
        raise AssertionError("incomplete sensor map accepted")
    except ValueError:
        pass

    try:
        minimal_sensor_cut_sets(worlds, sensors, ["failed_diag"])
        raise AssertionError("undiagnosable baseline accepted")
    except ValueError:
        pass

    print("ok")


if __name__ == "__main__":
    self_test()
