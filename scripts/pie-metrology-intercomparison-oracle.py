#!/usr/bin/env python3
"""PIE-009V independent metrology intercomparison oracle."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from math import isfinite

EPSILON = 1e-12

@dataclass(frozen=True)
class Reference:
    reference_id: str
    qualified: bool = True

@dataclass(frozen=True)
class Comparison:
    left_id: str
    right_id: str
    observed_delta: float
    allowed_delta: float
    qualified: bool
    failure_groups: tuple[str, ...]

def pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))

def validate(references: list[Reference], comparisons: list[Comparison]) -> None:
    ids = [ref.reference_id for ref in references]
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("invalid reference registry")
    known = set(ids)
    seen: set[tuple[str, str]] = set()
    for comparison in comparisons:
        if comparison.left_id == comparison.right_id or comparison.left_id not in known or comparison.right_id not in known:
            raise ValueError("invalid comparison endpoints")
        key = pair(comparison.left_id, comparison.right_id)
        if key in seen:
            raise ValueError("duplicate comparison pair")
        seen.add(key)
        if not isfinite(comparison.observed_delta) or not isfinite(comparison.allowed_delta):
            raise ValueError("non-finite comparison value")
        if comparison.allowed_delta < 0.0:
            raise ValueError("negative comparison tolerance")
        if not comparison.failure_groups or len(comparison.failure_groups) != len(set(comparison.failure_groups)) or any(not group for group in comparison.failure_groups):
            raise ValueError("invalid comparison failure groups")

def relation(comparison: Comparison) -> str:
    if not comparison.qualified:
        return "UNKNOWN"
    return "AGREE" if abs(comparison.observed_delta) <= comparison.allowed_delta + EPSILON else "DISAGREE"

def evidence_digest(references: list[Reference], comparisons: list[Comparison]) -> str:
    payload = {
        "references": [asdict(ref) for ref in sorted(references, key=lambda item: item.reference_id)],
        "comparisons": [
            {**asdict(comparison), "failure_groups": list(comparison.failure_groups)}
            for comparison in sorted(comparisons, key=lambda item: pair(item.left_id, item.right_id))
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def analyze(references: list[Reference], comparisons: list[Comparison]) -> dict[str, object]:
    validate(references, comparisons)
    qualified_refs = {ref.reference_id for ref in references if ref.qualified}
    comparison_map = {pair(c.left_id, c.right_id): c for c in comparisons}
    relations: dict[tuple[str, str], str] = {}
    for key, comparison in comparison_map.items():
        if comparison.left_id in qualified_refs and comparison.right_id in qualified_refs:
            relations[key] = relation(comparison)
    disagreements = [key for key, value in relations.items() if value == "DISAGREE"]
    digest = evidence_digest(references, comparisons)
    if not disagreements:
        status = "NO_DETECTED_DISAGREEMENT" if any(value == "AGREE" for value in relations.values()) else "INSUFFICIENT_EVIDENCE"
        return {"status": status, "isolated_relative_outlier": None, "ambiguous_references": [], "quarantine_candidates": [], "evidence_digest": digest}
    isolated: list[tuple[str, dict[str, object]]] = []
    for suspect in sorted(qualified_refs):
        peers = sorted(ref for ref in qualified_refs if ref != suspect)
        witness = None
        for first, second in combinations(peers, 2):
            cohort_pair = pair(first, second)
            suspect_first = pair(suspect, first)
            suspect_second = pair(suspect, second)
            if relations.get(cohort_pair) != "AGREE" or relations.get(suspect_first) != "DISAGREE" or relations.get(suspect_second) != "DISAGREE":
                continue
            channels = [comparison_map[cohort_pair], comparison_map[suspect_first], comparison_map[suspect_second]]
            groups = [set(channel.failure_groups) for channel in channels]
            if any(groups[left] & groups[right] for left, right in combinations(range(3), 2)):
                continue
            witness = {"agreeing_cohort": [first, second], "comparison_pairs": [list(cohort_pair), list(suspect_first), list(suspect_second)]}
            break
        if witness is not None:
            isolated.append((suspect, witness))
    if len(isolated) == 1:
        suspect, witness = isolated[0]
        return {"status": "RELATIVE_OUTLIER_ISOLATED", "isolated_relative_outlier": suspect, "witness": witness, "ambiguous_references": [], "quarantine_candidates": [suspect], "evidence_digest": digest}
    involved = sorted({ref for key in disagreements for ref in key})
    return {"status": "AMBIGUOUS_CONFLICT", "isolated_relative_outlier": None, "ambiguous_references": involved, "quarantine_candidates": involved, "evidence_digest": digest}

def expect_value_error(callable_) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

def self_test() -> None:
    references = [Reference("A"), Reference("B"), Reference("C")]
    all_agree = [Comparison("A", "B", 0.10, 0.20, True, ("cmp_ab",)), Comparison("A", "C", 0.10, 0.20, True, ("cmp_ac",)), Comparison("B", "C", 0.05, 0.20, True, ("cmp_bc",))]
    assert analyze(references, all_agree)["status"] == "NO_DETECTED_DISAGREEMENT"
    single_conflict = [Comparison("A", "B", 0.50, 0.20, True, ("cmp_ab",))]
    result = analyze(references, single_conflict)
    assert result["status"] == "AMBIGUOUS_CONFLICT" and result["quarantine_candidates"] == ["A", "B"]
    isolated = [Comparison("A", "B", 0.10, 0.20, True, ("cmp_ab",)), Comparison("A", "C", 0.70, 0.20, True, ("cmp_ac",)), Comparison("B", "C", 0.60, 0.20, True, ("cmp_bc",))]
    result = analyze(references, isolated)
    assert result["status"] == "RELATIVE_OUTLIER_ISOLATED" and result["isolated_relative_outlier"] == "C"
    shared_common_mode = [Comparison("A", "B", 0.10, 0.20, True, ("shared_fixture",)), Comparison("A", "C", 0.70, 0.20, True, ("shared_fixture",)), Comparison("B", "C", 0.60, 0.20, True, ("cmp_bc",))]
    assert analyze(references, shared_common_mode)["status"] == "AMBIGUOUS_CONFLICT"
    incomplete_witness = [Comparison("A", "B", 0.10, 0.20, True, ("cmp_ab",)), Comparison("A", "C", 0.70, 0.20, True, ("cmp_ac",)), Comparison("B", "C", 0.60, 0.20, False, ("cmp_bc",))]
    assert analyze(references, incomplete_witness)["status"] == "AMBIGUOUS_CONFLICT"
    all_disagree = [Comparison("A", "B", 0.50, 0.20, True, ("cmp_ab",)), Comparison("A", "C", 0.70, 0.20, True, ("cmp_ac",)), Comparison("B", "C", 0.60, 0.20, True, ("cmp_bc",))]
    assert analyze(references, all_disagree)["status"] == "AMBIGUOUS_CONFLICT"
    assert relation(Comparison("A", "B", 0.20, 0.20, True, ("cmp_ab",))) == "AGREE"
    digest = evidence_digest(references, isolated)
    assert digest == evidence_digest(references, isolated)
    changed = isolated[:-1] + [Comparison("B", "C", 0.61, 0.20, True, ("cmp_bc",))]
    assert digest != evidence_digest(references, changed)
    expect_value_error(lambda: validate(references, [Comparison("A", "B", 0.10, 0.20, True, ("ab",)), Comparison("B", "A", 0.10, 0.20, True, ("ba",))]))
    print("ok")

if __name__ == "__main__":
    self_test()
