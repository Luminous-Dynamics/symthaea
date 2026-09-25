#!/usr/bin/env python3
"""Independent SENSE-DESIGN-001A1 comparison oracle.

Reference-only verifier for the frozen synthetic corpus from #5827 / PR #5863.
It imports no Symthaea production code and makes no physical sensor claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_SCHEMA = "sense-comparison-fixtures-v1"
ALLOWED_PLANES = {"P1", "P2", "P3"}
ALLOWED_DIRECTIONS = {"LOWER", "HIGHER", "TARGET"}
ALLOWED_DISPOSITIONS = {
    "PARETO",
    "TRADEOFF",
    "EQUIVALENT",
    "PROTECTED_REGRESSION",
    "DOMINATED",
    "INSUFFICIENT",
    "INVALID",
    "OUT_OF_PROFILE",
}
EXPECTED_AXIS_TUPLE = [
    "id", "plane", "direction", "baseline", "candidate",
    "protected", "tolerance", "required",
]
EXPECTED_FIXTURE_TUPLE = [
    "id", "title", "validity", "expected", "axes", "notes",
]


def canonical_digest(document: dict) -> str:
    body = dict(document)
    expected = body.pop("known_answer_digest_sha256", None)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("missing/invalid known-answer digest")
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"digest mismatch: expected {expected}, got {actual}")
    return actual


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _state(value):
    if isinstance(value, dict):
        state = value.get("state")
        if state not in {"MISSING", "INVALID", "OUT_OF_PROFILE"}:
            raise ValueError(f"unsupported observation state: {state!r}")
        return state
    return "OBSERVED"


def compare_axis(axis):
    if len(axis) not in (8, 9):
        raise ValueError("axis tuple must have 8 fields plus optional metadata")
    axis_id, plane, direction, baseline, candidate, protected, tolerance, required = axis[:8]
    metadata = axis[8] if len(axis) == 9 else {}

    if not isinstance(axis_id, str) or not axis_id:
        raise ValueError("axis id must be nonempty")
    if plane not in ALLOWED_PLANES:
        raise ValueError(f"unknown plane: {plane!r}")
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"unknown direction: {direction!r}")
    if not isinstance(protected, bool) or not isinstance(required, bool):
        raise ValueError("protected/required must be booleans")
    tolerance = _finite_number(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if metadata and not isinstance(metadata, dict):
        raise ValueError("axis metadata must be an object")

    evidence_class = metadata.get("evidence_class")
    if evidence_class == "SIMULATION_ONLY":
        return "OUT_OF_PROFILE"

    b_state = _state(baseline)
    c_state = _state(candidate)
    if b_state != "OBSERVED" or c_state != "OBSERVED":
        states = {b_state, c_state}
        if "INVALID" in states:
            return "INVALID"
        if "OUT_OF_PROFILE" in states:
            return "OUT_OF_PROFILE"
        if "MISSING" in states:
            return "MISSING_REQUIRED" if required else "MISSING_OPTIONAL"

    baseline = _finite_number(baseline, "baseline")
    candidate = _finite_number(candidate, "candidate")

    if direction == "TARGET":
        # No target-interval fixture exists in 001A0. Fail closed until the
        # fixture schema carries an explicit interval rather than guessing.
        raise ValueError("TARGET direction requires a future explicit interval schema")

    delta = candidate - baseline
    if abs(delta) <= tolerance:
        return "EQUIVALENT"
    if direction == "LOWER":
        return "BETTER" if candidate < baseline else "WORSE"
    return "BETTER" if candidate > baseline else "WORSE"


def evaluate_fixture(fixture):
    if len(fixture) != 6:
        raise ValueError("fixture tuple must have exactly 6 fields")
    fixture_id, title, validity, expected, axes, notes = fixture
    if not isinstance(fixture_id, str) or not fixture_id:
        raise ValueError("fixture id must be nonempty")
    if not isinstance(title, str) or not title:
        raise ValueError(f"{fixture_id}: title must be nonempty")
    if expected not in ALLOWED_DISPOSITIONS:
        raise ValueError(f"{fixture_id}: unknown expected disposition {expected!r}")
    if not isinstance(axes, list) or not isinstance(notes, list):
        raise ValueError(f"{fixture_id}: axes/notes must be arrays")

    if validity != "VALID":
        return "INVALID"

    axis_results = [compare_axis(axis) for axis in axes]

    if "INVALID" in axis_results:
        return "INVALID"
    if "OUT_OF_PROFILE" in axis_results:
        return "OUT_OF_PROFILE"
    if "MISSING_REQUIRED" in axis_results:
        return "INSUFFICIENT"

    comparable = [r for r in axis_results if not r.startswith("MISSING_")]
    if not comparable:
        return "INSUFFICIENT"

    protected_worse = any(
        result == "WORSE" and bool(axis[5])
        for axis, result in zip(axes, axis_results)
    )
    if protected_worse:
        return "PROTECTED_REGRESSION"

    better = sum(r == "BETTER" for r in comparable)
    worse = sum(r == "WORSE" for r in comparable)
    equal = sum(r == "EQUIVALENT" for r in comparable)

    if worse == 0 and better > 0:
        return "PARETO"
    if better == 0 and worse > 0:
        return "DOMINATED"
    if better == 0 and worse == 0 and equal == len(comparable):
        return "EQUIVALENT"
    if better > 0 and worse > 0:
        return "TRADEOFF"

    raise ValueError(f"{fixture_id}: unresolved relation set {comparable!r}")


def validate_document(document: dict) -> tuple[str, list[tuple[str, str]]]:
    if document.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected schema")
    if document.get("axis_tuple") != EXPECTED_AXIS_TUPLE:
        raise ValueError("axis tuple schema drift")
    if document.get("fixture_tuple") != EXPECTED_FIXTURE_TUPLE:
        raise ValueError("fixture tuple schema drift")
    if set(document.get("planes", [])) != ALLOWED_PLANES:
        raise ValueError("plane vocabulary drift")
    if not ALLOWED_DIRECTIONS.issuperset(document.get("directions", [])):
        raise ValueError("direction vocabulary drift")
    if set(document.get("dispositions", [])) != ALLOWED_DISPOSITIONS:
        raise ValueError("disposition vocabulary drift")

    digest = canonical_digest(document)
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a nonempty list")

    seen = set()
    mismatches = []
    for fixture in fixtures:
        fixture_id = fixture[0] if fixture else "<malformed>"
        if fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture_id}")
        seen.add(fixture_id)
        actual = evaluate_fixture(fixture)
        expected = fixture[3]
        if actual != expected:
            mismatches.append((fixture_id, f"expected {expected}, got {actual}"))

    return digest, mismatches


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: sense-design-001a1-comparison-oracle.py <fixture.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    document = json.loads(path.read_text(encoding="utf-8"))
    digest, mismatches = validate_document(document)
    if mismatches:
        for fixture_id, message in mismatches:
            print(f"{fixture_id}: {message}", file=sys.stderr)
        return 1
    print(f"ok fixtures={len(document['fixtures'])} digest={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
