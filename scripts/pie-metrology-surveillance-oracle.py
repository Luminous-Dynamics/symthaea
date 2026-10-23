#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, Mapping, Sequence


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
class SurveillancePolicy:
    max_evidence_age: int
    slots_per_step: int
    horizon: int


def _validate(
    references: Sequence[Reference],
    comparisons: Sequence[Comparison],
    policy: SurveillancePolicy,
    initial_available_step: Mapping[str, int],
    schedule: Mapping[int, Sequence[str]],
) -> tuple[dict[str, Reference], dict[str, Comparison]]:
    if policy.max_evidence_age < 0 or policy.slots_per_step < 0 or policy.horizon < 0:
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
        key = tuple(sorted((c.a, c.b)))
        if key in pair_keys:
            raise ModelError("duplicate or reversed-duplicate comparison pair")
        pair_keys.add(key)
        if not c.failure_groups or any(not g for g in c.failure_groups):
            raise ModelError("comparison requires non-empty failure groups")
        cmp_map[c.id] = c

    for cid, available_step in initial_available_step.items():
        if cid not in cmp_map:
            raise ModelError("unknown initial comparison")
        if not isinstance(available_step, int) or available_step > 0:
            raise ModelError("initial evidence must be available no later than opening step 0")

    for step, ids in schedule.items():
        if not isinstance(step, int) or step < 0 or step >= policy.horizon:
            raise ModelError("schedule step out of range")
        if len(ids) > policy.slots_per_step:
            raise ModelError("comparison capacity exceeded")
        if len(ids) != len(set(ids)):
            raise ModelError("duplicate comparison scheduled in one step")
        for cid in ids:
            if cid not in cmp_map:
                raise ModelError("unknown scheduled comparison")
            if not cmp_map[cid].qualified:
                raise ModelError("unqualified comparison cannot be scheduled")

    return ref_map, cmp_map


def _edge_between(
    comparisons: Mapping[str, Comparison], a: str, b: str
) -> Comparison | None:
    target = {a, b}
    for c in comparisons.values():
        if {c.a, c.b} == target:
            return c
    return None


def _fresh_independent_witness(
    reference_id: str,
    references: Mapping[str, Reference],
    comparisons: Mapping[str, Comparison],
    available_step: Mapping[str, int],
    opening_step: int,
    max_evidence_age: int,
) -> dict | None:
    peers = [
        r.id
        for r in references.values()
        if r.active and r.id != reference_id
    ]
    candidates: list[tuple] = []

    for p, q in combinations(sorted(peers), 2):
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

        candidates.append(
            (
                max(ages),
                tuple(sorted(e.id for e in edges)),
                (p, q),
                tuple(sorted(ages)),
            )
        )

    if not candidates:
        return None

    oldest_age, edge_ids, peers, ages = min(candidates)
    return {
        "reference_id": reference_id,
        "peer_ids": peers,
        "comparison_ids": edge_ids,
        "oldest_evidence_age": oldest_age,
        "comparison_ages": ages,
    }


def evaluate_campaign(
    references: Sequence[Reference],
    comparisons: Sequence[Comparison],
    policy: SurveillancePolicy,
    initial_available_step: Mapping[str, int],
    schedule: Mapping[int, Sequence[str]],
) -> dict:
    ref_map, cmp_map = _validate(
        references, comparisons, policy, initial_available_step, schedule
    )

    available_step: dict[str, int] = dict(initial_available_step)
    pending: dict[int, list[str]] = {}
    critical_lapses: list[dict] = []
    all_opening_steps: list[dict] = []
    worst_critical_evidence_age = 0

    for step in range(policy.horizon):
        # Work completed during N becomes usable only at opening N+1.
        for cid in pending.pop(step, []):
            available_step[cid] = step

        opening = {"step": step, "references": {}}
        for ref in sorted(ref_map.values(), key=lambda r: r.id):
            if not ref.active:
                continue
            witness = _fresh_independent_witness(
                ref.id,
                ref_map,
                cmp_map,
                available_step,
                step,
                policy.max_evidence_age,
            )
            covered = witness is not None
            opening["references"][ref.id] = {
                "critical": ref.critical,
                "covered": covered,
                "witness": witness,
            }
            if ref.critical:
                if covered:
                    worst_critical_evidence_age = max(
                        worst_critical_evidence_age,
                        witness["oldest_evidence_age"],
                    )
                else:
                    critical_lapses.append(
                        {"step": step, "reference_id": ref.id}
                    )

        for cid in schedule.get(step, ()):
            pending.setdefault(step + 1, []).append(cid)

        all_opening_steps.append(opening)

    return {
        "critical_continuity": not critical_lapses,
        "critical_lapses": critical_lapses,
        "worst_critical_evidence_age": worst_critical_evidence_age,
        "declared_max_evidence_age": policy.max_evidence_age,
        "opening_steps": all_opening_steps,
    }


def _base_fixture():
    refs = [Reference("A"), Reference("B"), Reference("C")]
    comparisons = [
        Comparison("AB", "A", "B", True, frozenset({"fixture-ab"})),
        Comparison("AC", "A", "C", True, frozenset({"fixture-ac"})),
        Comparison("BC", "B", "C", True, frozenset({"fixture-bc"})),
    ]
    initial = {"AB": 0, "AC": 0, "BC": 0}
    schedule = {
        0: ["AB"],
        1: ["AC"],
        2: ["BC"],
        3: ["AB"],
        4: ["AC"],
        5: ["BC"],
        6: ["AB"],
    }
    return refs, comparisons, initial, schedule


def self_test() -> None:
    refs, comparisons, initial, schedule = _base_fixture()
    policy = SurveillancePolicy(max_evidence_age=2, slots_per_step=1, horizon=7)

    baseline = evaluate_campaign(refs, comparisons, policy, initial, schedule)
    assert baseline["critical_continuity"] is True
    assert baseline["critical_lapses"] == []
    assert baseline["worst_critical_evidence_age"] == 2

    # Delay AC: opening step 3 has stale AC evidence; same-step work cannot repair it.
    delayed = dict(schedule)
    delayed.pop(1)
    delayed[3] = ["AC"]
    delayed.pop(6)
    lapse = evaluate_campaign(refs, comparisons, policy, initial, delayed)
    assert lapse["critical_continuity"] is False
    assert any(x["step"] == 3 for x in lapse["critical_lapses"])
    assert all(
        not rec["covered"]
        for rec in lapse["opening_steps"][3]["references"].values()
        if rec["critical"]
    )

    # Bench capacity is a hard constraint.
    try:
        evaluate_campaign(refs, comparisons, policy, initial, {0: ["AB", "AC"]})
        raise AssertionError("over-capacity schedule was accepted")
    except ModelError:
        pass

    # Fresh numbers with a common comparison failure domain do not make
    # an independent surveillance triangle.
    common_mode = [
        Comparison("AB", "A", "B", True, frozenset({"shared"})),
        Comparison("AC", "A", "C", True, frozenset({"shared"})),
        Comparison("BC", "B", "C", True, frozenset({"bc"})),
    ]
    cm = evaluate_campaign(refs, common_mode, SurveillancePolicy(2, 0, 1), initial, {})
    assert cm["critical_continuity"] is False

    # An unqualified edge cannot support the witness.
    unqualified = [
        Comparison("AB", "A", "B", True, frozenset({"ab"})),
        Comparison("AC", "A", "C", False, frozenset({"ac"})),
        Comparison("BC", "B", "C", True, frozenset({"bc"})),
    ]
    uq = evaluate_campaign(refs, unqualified, SurveillancePolicy(2, 0, 1), initial, {})
    assert uq["critical_continuity"] is False

    # A noncritical unmonitored reference may lapse without invalidating
    # otherwise continuous critical coverage.
    refs_plus = refs + [Reference("D", critical=False)]
    noncritical = evaluate_campaign(refs_plus, comparisons, policy, initial, schedule)
    assert noncritical["critical_continuity"] is True
    assert noncritical["opening_steps"][0]["references"]["D"]["covered"] is False

    # Duplicate/reversed duplicate pairs fail closed.
    try:
        evaluate_campaign(
            refs,
            comparisons + [Comparison("BA2", "B", "A", True, frozenset({"ba2"}))],
            policy,
            initial,
            schedule,
        )
        raise AssertionError("duplicate pair accepted")
    except ModelError:
        pass

    # Unknown schedule IDs fail closed.
    try:
        evaluate_campaign(refs, comparisons, policy, initial, {0: ["UNKNOWN"]})
        raise AssertionError("unknown scheduled comparison accepted")
    except ModelError:
        pass

    print("ok")


if __name__ == "__main__":
    self_test()
