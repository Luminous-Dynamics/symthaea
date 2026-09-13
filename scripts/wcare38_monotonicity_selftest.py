#!/usr/bin/env python3
"""Dependency-free WCARE-38 monotonicity self-test.

This is an algorithm-level test for the governing theorem:
removing authentication can never increase authenticated evidentiary weight.
It grants no runtime authority and makes no claim about any real panel.
"""
from __future__ import annotations

from itertools import combinations
import json
import sys

IDENTITIES = ("a", "b", "c", "d")
LINEAGE = {"a": "L1", "b": "L2", "c": "L3", "d": "L4"}
BASELINE_ACCEPTED_PAIRS = tuple(combinations(IDENTITIES, 2))


def components(adjacency: dict[str, set[str]]) -> int:
    remaining = set(IDENTITIES)
    count = 0
    while remaining:
        count += 1
        start = min(remaining)
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node] - seen)
        remaining -= seen
    return count


def metrics(
    authenticated_identities: frozenset[str],
    authenticated_relations: frozenset[tuple[str, str]],
) -> tuple[int, int, int, int]:
    authenticated_lineages = {LINEAGE[i] for i in authenticated_identities}
    adjacency = {identity: set() for identity in IDENTITIES}
    accepted_pairs = 0
    for left, right in BASELINE_ACCEPTED_PAIRS:
        can_remain_separate = (
            left in authenticated_identities
            and right in authenticated_identities
            and (left, right) in authenticated_relations
        )
        if can_remain_separate:
            accepted_pairs += 1
        else:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return (
        len(authenticated_identities),
        len(authenticated_lineages),
        accepted_pairs,
        components(adjacency),
    )


def subsets(values: tuple) -> list[frozenset]:
    out: list[frozenset] = []
    for size in range(len(values) + 1):
        out.extend(frozenset(choice) for choice in combinations(values, size))
    return out


def main() -> int:
    identity_sets = subsets(IDENTITIES)
    relation_sets = subsets(BASELINE_ACCEPTED_PAIRS)
    states_checked = 0
    removal_edges_checked = 0

    baseline = metrics(frozenset(IDENTITIES), frozenset(BASELINE_ACCEPTED_PAIRS))
    if baseline != (4, 4, 6, 4):
        raise AssertionError(f"unexpected_baseline:{baseline}")

    for ids in identity_sets:
        for rels in relation_sets:
            current = metrics(ids, rels)
            states_checked += 1
            if any(observed > upper for observed, upper in zip(current, baseline, strict=True)):
                raise AssertionError(f"baseline_monotonicity_violation:{ids}:{rels}:{current}:{baseline}")

            for identity in ids:
                weaker = metrics(ids - {identity}, rels)
                removal_edges_checked += 1
                if any(after > before for after, before in zip(weaker, current, strict=True)):
                    raise AssertionError(
                        f"identity_removal_increased_weight:{identity}:{current}:{weaker}"
                    )

            for relation in rels:
                weaker = metrics(ids, rels - {relation})
                removal_edges_checked += 1
                if any(after > before for after, before in zip(weaker, current, strict=True)):
                    raise AssertionError(
                        f"relation_removal_increased_weight:{relation}:{current}:{weaker}"
                    )

    payload = {
        "authority": "MeasurementOnly",
        "classification": "PASS_MONOTONICITY_SELFTEST",
        "states_checked": states_checked,
        "single_removal_transitions_checked": removal_edges_checked,
        "authenticated_independence_can_exceed_baseline": False,
        "runtime_authority_granted": False,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
