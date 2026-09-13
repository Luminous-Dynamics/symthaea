#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import FrozenSet, Mapping, Sequence


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class Reference:
    id: str
    critical: bool = True
    active: bool = True


@dataclass(frozen=True)
class Comparison:
    id: str
    a: str
    b: str
    qualified: bool
    failure_groups: FrozenSet[str]


@dataclass(frozen=True)
class CampaignPolicy:
    max_evidence_age: int
    horizon: int


def _validate(
    references: Sequence[Reference],
    comparisons: Sequence[Comparison],
    policy: CampaignPolicy,
    initial_available_step: Mapping[str, int],
) -> tuple[dict[str, Reference], dict[str, Comparison]]:
    if policy.max_evidence_age < 0 or policy.horizon < 0:
        raise ModelError("negative policy field")

    ref_ids = [r.id for r in references]
    if not ref_ids or any(not rid for rid in ref_ids) or len(set(ref_ids)) != len(ref_ids):
        raise ModelError("invalid or duplicate reference ID")
    ref_map = {r.id: r for r in references}

    cmp_ids = [c.id for c in comparisons]
    if any(not cid for cid in cmp_ids) or len(set(cmp_ids)) != len(cmp_ids):
        raise ModelError("invalid or duplicate comparison ID")

    pair_keys: set[tuple[str, str]] = set()
    cmp_map: dict[str, Comparison] = {}
    for c in comparisons:
        if c.a == c.b or c.a not in ref_map or c.b not in ref_map:
            raise ModelError("bad comparison endpoint")
        pair = tuple(sorted((c.a, c.b)))
        if pair in pair_keys:
            raise ModelError("duplicate or reversed-duplicate comparison pair")
        pair_keys.add(pair)
        if not c.failure_groups or any(not g for g in c.failure_groups):
            raise ModelError("comparison requires non-empty failure groups")
        cmp_map[c.id] = c

    for cid, available_step in initial_available_step.items():
        if cid not in cmp_map:
            raise ModelError("unknown initial comparison")
        if not isinstance(available_step, int) or available_step > 0:
            raise ModelError("initial evidence must be available by opening step 0")

    return ref_map, cmp_map


def _edge_between(
    comparisons: Mapping[str, Comparison], a: str, b: str
) -> Comparison | None:
    target = {a, b}
    for c in comparisons.values():
        if {c.a, c.b} == target:
            return c
    return None


def _covered(
    reference_id: str,
    references: Mapping[str, Reference],
    comparisons: Mapping[str, Comparison],
    available_step: Mapping[str, int],
    opening_step: int,
    max_evidence_age: int,
) -> bool:
    peers = sorted(
        r.id for r in references.values()
        if r.active and r.id != reference_id
    )

    for p, q in combinations(peers, 2):
        edges = (
            _edge_between(comparisons, reference_id, p),
            _edge_between(comparisons, reference_id, q),
            _edge_between(comparisons, p, q),
        )
        if any(e is None or not e.qualified for e in edges):
            continue
        assert all(e is not None for e in edges)
        edges = tuple(e for e in edges if e is not None)

        if any(e.id not in available_step for e in edges):
            continue

        ages = tuple(opening_step - available_step[e.id] for e in edges)
        if any(age < 0 or age > max_evidence_age for age in ages):
            continue

        if any(
            a.failure_groups & b.failure_groups
            for a, b in combinations(edges, 2)
        ):
            continue

        return True

    return False


def _opening_critical_coverage(
    references: Mapping[str, Reference],
    comparisons: Mapping[str, Comparison],
    available_step: Mapping[str, int],
    opening_step: int,
    policy: CampaignPolicy,
) -> bool:
    return all(
        _covered(
            ref.id,
            references,
            comparisons,
            available_step,
            opening_step,
            policy.max_evidence_age,
        )
        for ref in references.values()
        if ref.active and ref.critical
    )


def _subset_choices(ids: tuple[str, ...], capacity: int):
    for size in range(capacity + 1):
        yield from combinations(ids, size)


def find_schedule_for_capacity_profile(
    references: Sequence[Reference],
    comparisons: Sequence[Comparison],
    policy: CampaignPolicy,
    initial_available_step: Mapping[str, int],
    slots_by_step: Sequence[int],
) -> tuple[tuple[str, ...], ...] | None:
    ref_map, cmp_map = _validate(
        references, comparisons, policy, initial_available_step
    )

    if len(slots_by_step) != policy.horizon:
        raise ModelError("capacity profile length must equal campaign horizon")
    if any((not isinstance(x, int)) or x < 0 for x in slots_by_step):
        raise ModelError("capacity profile contains invalid slot count")

    qualified_ids = tuple(sorted(c.id for c in comparisons if c.qualified))

    @lru_cache(maxsize=None)
    def search(
        step: int,
        available_items: tuple[tuple[str, int], ...],
    ) -> tuple[tuple[str, ...], ...] | None:
        available = dict(available_items)

        if step >= policy.horizon:
            return ()

        if not _opening_critical_coverage(
            ref_map, cmp_map, available, step, policy
        ):
            return None

        capacity = slots_by_step[step]
        for chosen in _subset_choices(qualified_ids, capacity):
            next_available = dict(available)
            # Comparison work in N is visible only at opening N+1.
            for cid in chosen:
                next_available[cid] = step + 1

            suffix = search(
                step + 1,
                tuple(sorted(next_available.items())),
            )
            if suffix is not None:
                return (tuple(chosen),) + suffix

        return None

    return search(0, tuple(sorted(initial_available_step.items())))


def minimum_constant_capacity(
    references: Sequence[Reference],
    comparisons: Sequence[Comparison],
    policy: CampaignPolicy,
    initial_available_step: Mapping[str, int],
) -> dict:
    # No step can use more pairwise comparison channels than exist.
    for capacity in range(len(comparisons) + 1):
        schedule = find_schedule_for_capacity_profile(
            references,
            comparisons,
            policy,
            initial_available_step,
            [capacity] * policy.horizon,
        )
        if schedule is not None:
            return {
                "status": "Feasible",
                "minimum_slots_per_step": capacity,
                "schedule": schedule,
            }

    return {
        "status": "Unschedulable",
        "minimum_slots_per_step": None,
        "schedule": None,
    }


def _fixture():
    refs = [
        Reference("A"),
        Reference("B"),
        Reference("C"),
    ]
    comparisons = [
        Comparison("AB", "A", "B", True, frozenset({"ab-domain"})),
        Comparison("AC", "A", "C", True, frozenset({"ac-domain"})),
        Comparison("BC", "B", "C", True, frozenset({"bc-domain"})),
    ]
    initial = {"AB": 0, "AC": 0, "BC": 0}
    return refs, comparisons, initial


def self_test() -> None:
    refs, comparisons, initial = _fixture()

    policy = CampaignPolicy(max_evidence_age=2, horizon=7)
    result = minimum_constant_capacity(refs, comparisons, policy, initial)
    assert result["status"] == "Feasible"
    assert result["minimum_slots_per_step"] == 1
    assert result["schedule"] is not None

    zero = find_schedule_for_capacity_profile(
        refs, comparisons, policy, initial, [0] * policy.horizon
    )
    assert zero is None

    # If evidence may never age beyond zero, all three pairwise comparisons
    # must be refreshed every step for the next opening.
    strict = minimum_constant_capacity(
        refs,
        comparisons,
        CampaignPolicy(max_evidence_age=0, horizon=4),
        initial,
    )
    assert strict["minimum_slots_per_step"] == 3

    # Planned comparison-bench outages are feasible only with enough
    # compensating capacity around them.
    outage_profile = [2, 1, 0, 2, 1, 0, 2]
    outage_schedule = find_schedule_for_capacity_profile(
        refs, comparisons, policy, initial, outage_profile
    )
    assert outage_schedule is not None
    assert outage_schedule[2] == ()
    assert outage_schedule[5] == ()

    uncompensated = find_schedule_for_capacity_profile(
        refs,
        comparisons,
        policy,
        initial,
        [1, 1, 0, 1, 1, 1, 1],
    )
    assert uncompensated is None

    # Shared comparison common mode makes the surveillance topology
    # structurally unusable, irrespective of raw slot capacity.
    common_mode = [
        Comparison("AB", "A", "B", True, frozenset({"shared"})),
        Comparison("AC", "A", "C", True, frozenset({"shared"})),
        Comparison("BC", "B", "C", True, frozenset({"bc"})),
    ]
    impossible = minimum_constant_capacity(
        refs, common_mode, policy, initial
    )
    assert impossible["status"] == "Unschedulable"

    # A noncritical unmonitored reference does not raise the critical
    # minimum simply because it exists.
    refs_plus = refs + [Reference("D", critical=False)]
    with_noncritical = minimum_constant_capacity(
        refs_plus, comparisons, policy, initial
    )
    assert with_noncritical["minimum_slots_per_step"] == 1

    # Malformed profiles fail closed.
    try:
        find_schedule_for_capacity_profile(
            refs, comparisons, policy, initial, [1, 1]
        )
        raise AssertionError("short capacity profile accepted")
    except ModelError:
        pass

    try:
        find_schedule_for_capacity_profile(
            refs, comparisons, policy, initial, [1, 1, -1, 1, 1, 1, 1]
        )
        raise AssertionError("negative capacity accepted")
    except ModelError:
        pass

    print("ok")


if __name__ == "__main__":
    self_test()
