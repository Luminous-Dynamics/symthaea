#!/usr/bin/env python3
"""PIE-009O independent safe active-diagnosis oracle.

Synthetic symbolic reference only. No probabilities, physical test procedures,
optimal-control claims, reliability claims, or hardware authority are implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class World:
    world_id: str
    decision: str


@dataclass(frozen=True)
class Probe:
    probe_id: str
    safe_worlds: Tuple[str, ...]
    outputs: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PlanNode:
    worst_case_depth: int
    probe_id: Optional[str]
    branches: Tuple[Tuple[str, "PlanNode"], ...]


def validate_model(worlds: Sequence[World], probes: Sequence[Probe]) -> None:
    if len(worlds) < 2:
        raise ValueError("at least two worlds required")
    world_ids = [world.world_id for world in worlds]
    if any(not world_id for world_id in world_ids):
        raise ValueError("world ids must be nonempty")
    if len(set(world_ids)) != len(world_ids):
        raise ValueError("duplicate world id")
    if any(not world.decision for world in worlds):
        raise ValueError("decision labels must be nonempty")

    probe_ids = [probe.probe_id for probe in probes]
    if any(not probe_id for probe_id in probe_ids):
        raise ValueError("probe ids must be nonempty")
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("duplicate probe id")

    world_set = set(world_ids)
    for probe in probes:
        safe = set(probe.safe_worlds)
        if not safe <= world_set:
            raise ValueError("probe safe set references unknown world")
        outputs = dict(probe.outputs)
        if set(outputs) != world_set or len(outputs) != len(probe.outputs):
            raise ValueError("probe must declare one output per world")
        if any(observation == "" for observation in outputs.values()):
            raise ValueError("probe observations must be explicit")


def solve_plan(
    worlds: Sequence[World],
    probes: Sequence[Probe],
    belief_world_ids: Iterable[str],
    available_probe_ids: Iterable[str],
) -> Optional[PlanNode]:
    validate_model(worlds, probes)
    world_by_id = {world.world_id: world for world in worlds}
    probe_by_id = {probe.probe_id: probe for probe in probes}

    initial_belief = tuple(sorted(set(belief_world_ids)))
    available = tuple(sorted(set(available_probe_ids)))
    if not initial_belief:
        raise ValueError("belief cannot be empty")
    if set(initial_belief) - set(world_by_id):
        raise ValueError("belief references unknown world")
    if set(available) - set(probe_by_id):
        raise ValueError("available probes reference unknown probe")

    def homogeneous(belief: Tuple[str, ...]) -> bool:
        return len({world_by_id[world_id].decision for world_id in belief}) == 1

    @lru_cache(maxsize=None)
    def solve(belief: Tuple[str, ...], remaining: Tuple[str, ...]) -> Optional[PlanNode]:
        if homogeneous(belief):
            return PlanNode(0, None, ())

        best: Optional[PlanNode] = None
        for probe_id in remaining:
            probe = probe_by_id[probe_id]
            if not set(belief) <= set(probe.safe_worlds):
                continue

            output_map = dict(probe.outputs)
            partitions: Dict[str, list[str]] = {}
            for world_id in belief:
                partitions.setdefault(output_map[world_id], []).append(world_id)
            if len(partitions) <= 1:
                continue

            next_remaining = tuple(x for x in remaining if x != probe_id)
            branches = []
            child_depths = []
            feasible = True
            for observation in sorted(partitions):
                child_belief = tuple(sorted(partitions[observation]))
                child = solve(child_belief, next_remaining)
                if child is None:
                    feasible = False
                    break
                branches.append((observation, child))
                child_depths.append(child.worst_case_depth)
            if not feasible:
                continue

            candidate = PlanNode(
                1 + max(child_depths, default=0),
                probe_id,
                tuple(branches),
            )
            if (
                best is None
                or candidate.worst_case_depth < best.worst_case_depth
                or (
                    candidate.worst_case_depth == best.worst_case_depth
                    and candidate.probe_id < best.probe_id
                )
            ):
                best = candidate
        return best

    return solve(initial_belief, available)


def self_test() -> None:
    worlds = (
        World("healthy_closed", "PROCEED"),
        World("failed_closed", "BLOCK"),
        World("healthy_open", "BLOCK"),
        World("failed_open", "BLOCK"),
    )

    probes = (
        Probe(
            "probe_gen",
            tuple(world.world_id for world in worlds),
            (
                ("healthy_closed", "H"),
                ("failed_closed", "F"),
                ("healthy_open", "H"),
                ("failed_open", "F"),
            ),
        ),
        Probe(
            "probe_tie",
            tuple(world.world_id for world in worlds),
            (
                ("healthy_closed", "C"),
                ("failed_closed", "C"),
                ("healthy_open", "O"),
                ("failed_open", "O"),
            ),
        ),
        Probe(
            "probe_gen_redundant",
            tuple(world.world_id for world in worlds),
            (
                ("healthy_closed", "H"),
                ("failed_closed", "F"),
                ("healthy_open", "H"),
                ("failed_open", "F"),
            ),
        ),
        Probe(
            "live_spin_test",
            ("healthy_closed", "healthy_open"),
            (
                ("healthy_closed", "OK"),
                ("failed_closed", "FAULT"),
                ("healthy_open", "OK"),
                ("failed_open", "FAULT"),
            ),
        ),
    )
    validate_model(worlds, probes)
    all_worlds = tuple(world.world_id for world in worlds)

    plan = solve_plan(
        worlds,
        probes,
        all_worlds,
        ("probe_gen", "probe_tie", "live_spin_test"),
    )
    assert plan is not None
    assert plan.worst_case_depth == 2
    assert plan.probe_id == "probe_gen"
    branches = dict(plan.branches)
    assert branches["F"].worst_case_depth == 0
    assert branches["H"].probe_id == "probe_tie"
    assert branches["H"].worst_case_depth == 1

    assert plan.probe_id != "live_spin_test"

    blocked = solve_plan(
        worlds,
        probes,
        all_worlds,
        ("probe_gen", "live_spin_test"),
    )
    assert blocked is None

    done = solve_plan(
        worlds,
        probes,
        ("failed_closed", "failed_open"),
        ("probe_gen", "probe_tie"),
    )
    assert done is not None
    assert done.worst_case_depth == 0
    assert done.probe_id is None

    with_redundancy = solve_plan(
        worlds,
        probes,
        all_worlds,
        ("probe_gen", "probe_gen_redundant", "probe_tie", "live_spin_test"),
    )
    assert with_redundancy is not None
    assert with_redundancy.worst_case_depth <= plan.worst_case_depth

    uninformative = Probe(
        "uninformative",
        all_worlds,
        tuple((world_id, "SAME") for world_id in all_worlds),
    )
    extended = probes + (uninformative,)
    assert solve_plan(worlds, extended, all_worlds, ("uninformative",)) is None

    try:
        bad = probes + (
            Probe("probe_gen", all_worlds, tuple((w, "X") for w in all_worlds)),
        )
        validate_model(worlds, bad)
        raise AssertionError("duplicate probe id accepted")
    except ValueError:
        pass

    try:
        incomplete = (Probe("bad", all_worlds, (("healthy_closed", "X"),)),)
        validate_model(worlds, incomplete)
        raise AssertionError("incomplete probe output map accepted")
    except ValueError:
        pass

    try:
        solve_plan(worlds, probes, ("unknown_world",), ("probe_gen",))
        raise AssertionError("unknown belief world accepted")
    except ValueError:
        pass

    print("ok")


if __name__ == "__main__":
    self_test()
