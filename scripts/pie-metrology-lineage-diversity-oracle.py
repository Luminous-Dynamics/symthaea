#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

class ModelError(ValueError):
    pass

@dataclass(frozen=True)
class TracePath:
    id: str
    sensor_id: str
    anchor_id: str
    failure_groups: tuple[str, ...]
    qualified: bool = True

@dataclass(frozen=True)
class DiversityPolicy:
    min_paths: int
    min_distinct_anchors: int

def validate(paths: list[TracePath], policy: DiversityPolicy) -> None:
    if policy.min_paths < 1 or policy.min_distinct_anchors < 1:
        raise ModelError("invalid policy")
    if policy.min_distinct_anchors > policy.min_paths:
        raise ModelError("anchor diversity cannot exceed path count")
    seen = set()
    for p in paths:
        if not p.id or p.id in seen:
            raise ModelError("duplicate/empty path id")
        seen.add(p.id)
        if not p.sensor_id or not p.anchor_id:
            raise ModelError("missing sensor/anchor id")
        if len(set(p.failure_groups)) != len(p.failure_groups):
            raise ModelError("duplicate failure group within path")
        if any(not g for g in p.failure_groups):
            raise ModelError("empty failure group")
        if p.anchor_id not in p.failure_groups:
            raise ModelError("anchor must appear as explicit failure group")

def _independent(subset: tuple[TracePath, ...], policy: DiversityPolicy) -> bool:
    if len(subset) < policy.min_paths:
        return False
    if len({p.sensor_id for p in subset}) < policy.min_paths:
        return False
    if len({p.anchor_id for p in subset}) < policy.min_distinct_anchors:
        return False
    group_sets = [set(p.failure_groups) for p in subset]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            if group_sets[i] & group_sets[j]:
                return False
    return True

def minimal_sufficient_sets(paths: list[TracePath], policy: DiversityPolicy) -> list[tuple[str, ...]]:
    validate(paths, policy)
    qualified = [p for p in paths if p.qualified]
    out: list[tuple[str, ...]] = []
    for n in range(policy.min_paths, len(qualified) + 1):
        for subset in combinations(qualified, n):
            ids = tuple(sorted(p.id for p in subset))
            if not _independent(subset, policy):
                continue
            if any(set(prev).issubset(ids) for prev in out):
                continue
            out.append(ids)
    return sorted(out)

def _survivors_after_path_loss(paths: list[TracePath], lost: set[str]) -> list[TracePath]:
    return [p for p in paths if p.id not in lost]

def _survivors_after_group_loss(paths: list[TracePath], lost_groups: set[str]) -> list[TracePath]:
    return [p for p in paths if not (set(p.failure_groups) & lost_groups)]

def minimal_path_cut_sets(paths: list[TracePath], policy: DiversityPolicy) -> list[tuple[str, ...]]:
    validate(paths, policy)
    if not minimal_sufficient_sets(paths, policy):
        raise ModelError("baseline not sufficient")
    ids = sorted(p.id for p in paths if p.qualified)
    cuts: list[tuple[str, ...]] = []
    for n in range(1, len(ids) + 1):
        for combo in combinations(ids, n):
            lost = set(combo)
            if minimal_sufficient_sets(_survivors_after_path_loss(paths, lost), policy):
                continue
            if any(set(prev).issubset(lost) for prev in cuts):
                continue
            cuts.append(combo)
    return cuts

def minimal_group_cut_sets(paths: list[TracePath], policy: DiversityPolicy) -> list[tuple[str, ...]]:
    validate(paths, policy)
    if not minimal_sufficient_sets(paths, policy):
        raise ModelError("baseline not sufficient")
    groups = sorted({g for p in paths if p.qualified for g in p.failure_groups})
    cuts: list[tuple[str, ...]] = []
    for n in range(1, len(groups) + 1):
        for combo in combinations(groups, n):
            lost = set(combo)
            if minimal_sufficient_sets(_survivors_after_group_loss(paths, lost), policy):
                continue
            if any(set(prev).issubset(lost) for prev in cuts):
                continue
            cuts.append(combo)
    return cuts

def self_test() -> None:
    paths = [
        TracePath("current", "sensor_current", "anchor_imported",
                  ("anchor_imported", "transfer_current", "lab_current")),
        TracePath("vibration", "sensor_vibration", "anchor_imported",
                  ("anchor_imported", "transfer_vibration", "lab_vibration")),
        TracePath("optical", "sensor_optical", "anchor_local_a",
                  ("anchor_local_a", "transfer_optical", "lab_optical")),
        TracePath("thermal", "sensor_thermal", "anchor_local_b",
                  ("anchor_local_b", "transfer_thermal", "lab_thermal")),
        TracePath("stale_spare", "sensor_spare", "anchor_local_b",
                  ("anchor_local_b", "transfer_spare"), qualified=False),
    ]
    policy = DiversityPolicy(min_paths=2, min_distinct_anchors=2)

    suff = minimal_sufficient_sets(paths, policy)
    suff_sets = {frozenset(x) for x in suff}
    assert frozenset(("current", "vibration")) not in suff_sets
    assert frozenset(("current", "optical")) in suff_sets
    assert frozenset(("vibration", "thermal")) in suff_sets
    assert frozenset(("optical", "thermal")) in suff_sets
    assert all("stale_spare" not in s for s in suff)

    path_cuts = minimal_path_cut_sets(paths, policy)
    assert all(len(c) >= 2 for c in path_cuts)

    group_cuts = minimal_group_cut_sets(paths, policy)
    anchors = {"anchor_imported", "anchor_local_a", "anchor_local_b"}
    anchor_only_cuts = [set(c) for c in group_cuts if set(c) <= anchors]
    assert {"anchor_imported", "anchor_local_a"} in anchor_only_cuts
    assert {"anchor_imported", "anchor_local_b"} in anchor_only_cuts
    assert {"anchor_local_a", "anchor_local_b"} in anchor_only_cuts
    assert not any(len(c) == 1 and c[0] in anchors for c in group_cuts)

    same_anchor = paths[:2]
    assert minimal_sufficient_sets(same_anchor, policy) == []

    bad_shared = [
        TracePath("a", "sa", "pa", ("pa", "shared_fixture")),
        TracePath("b", "sb", "pb", ("pb", "shared_fixture")),
    ]
    assert minimal_sufficient_sets(bad_shared, policy) == []

    before = {frozenset(x) for x in suff}
    extended = paths + [
        TracePath("acoustic", "sensor_acoustic", "anchor_local_c",
                  ("anchor_local_c", "transfer_acoustic", "lab_acoustic"))
    ]
    after = {frozenset(x) for x in minimal_sufficient_sets(extended, policy)}
    assert before.issubset(after)

    try:
        minimal_sufficient_sets([
            TracePath("x", "sx", "px", ("not_px",))
        ], policy)
        raise AssertionError("missing explicit anchor failure group should fail")
    except ModelError:
        pass

    print("ok")

if __name__ == "__main__":
    self_test()
