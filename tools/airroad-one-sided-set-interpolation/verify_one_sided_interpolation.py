#!/usr/bin/env python3
"""Exact-rational verifier for AIRROAD's one-sided convex-set interpolation."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-one-sided-set-interpolation-v0.1.json"

Point = tuple[Fraction, Fraction]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def point(value: list[dict[str, str]]) -> Point:
    if len(value) != 2:
        raise ValueError("the V0.1 fixture requires 2D points")
    return q(value[0]), q(value[1])


def encoded_point(value: Point) -> list[dict[str, str]]:
    return [
        {"numerator": str(coordinate.numerator),
         "denominator": str(coordinate.denominator)}
        for coordinate in value
    ]


def cross(origin: Point, a: Point, b: Point) -> Fraction:
    return (
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def convex_hull(points: list[Point]) -> list[Point]:
    """Return the exact CCW monotone-chain hull, starting at lexicographic min."""
    unique = sorted(set(points))
    if not unique:
        raise ValueError("a convex endpoint set must contain at least one point")
    if len(unique) <= 2:
        return unique

    lower: list[Point] = []
    for item in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], item) <= 0:
            lower.pop()
        lower.append(item)

    upper: list[Point] = []
    for item in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], item) <= 0:
            upper.pop()
        upper.append(item)

    return lower[:-1] + upper[:-1]


def resolve_endpoint(knot: dict[str, Any], semantics: str) -> list[Point]:
    if semantics not in {"incoming_minus", "outgoing_plus"}:
        raise ValueError("unknown one-sided endpoint semantics")
    records = knot.get(semantics)
    if not records:
        raise ValueError(f"{knot['knot_id']} has no {semantics} endpoint")
    ids = [record["generator_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate generator identity")
    return [point(record["point"]) for record in records]


def affine_product(a: list[Point], b: list[Point], lam: Fraction) -> list[Point]:
    if not Fraction(0) <= lam <= Fraction(1):
        raise ValueError("lambda must lie in [0,1]")
    if not a or not b:
        raise ValueError("both endpoint generator sets must be nonempty")
    return [
        (
            (1 - lam) * left[0] + lam * right[0],
            (1 - lam) * left[1] + lam * right[1],
        )
        for left in a
        for right in b
    ]


def decode_hull(value: list[list[dict[str, str]]]) -> list[Point]:
    return [point(item) for item in value]


def assert_metadata(data: dict[str, Any]) -> None:
    assert data["schema"] == "luminous.airroad.one_sided_set_interpolation.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == (
        "synthetic_exact_rational_hard_polytope_interpolation"
    )
    assert data["uncertainty_semantics"] == "deterministic_hard_set"
    assert data["frequency_unit"] == "normalized_frequency"
    assert data["dimension"] == 2
    assert data["port_order"] == ["p0", "p1"]
    assert data["model_identity"] == "sha256:one-sided-set-model-v0.1"


def verify_interval(
    interval: dict[str, Any],
    knots: dict[str, dict[str, Any]],
) -> list[list[Point]]:
    if interval["left_endpoint_semantics"] != "outgoing_plus":
        raise ValueError("left interval endpoint must use the outgoing plus set")
    if interval["right_endpoint_semantics"] != "incoming_minus":
        raise ValueError("right interval endpoint must use the incoming minus set")

    left_knot = knots[interval["left_knot_id"]]
    right_knot = knots[interval["right_knot_id"]]
    left_frequency = q(left_knot["frequency"])
    right_frequency = q(right_knot["frequency"])
    if not left_frequency < right_frequency:
        raise ValueError("interval frequency endpoints are not strictly ordered")

    left = resolve_endpoint(left_knot, interval["left_endpoint_semantics"])
    right = resolve_endpoint(right_knot, interval["right_endpoint_semantics"])

    results: list[list[Point]] = []
    lambdas = [q(value) for value in interval["lambda_values"]]
    expected = interval["expected_hulls"]
    assert len(lambdas) == len(expected)

    for lam, expected_hull in zip(lambdas, expected):
        candidates = affine_product(left, right, lam)
        actual = convex_hull(candidates)
        declared = decode_hull(expected_hull)
        assert actual == declared, (
            f"{interval['interval_id']} lambda={lam}: "
            f"expected {declared}, computed {actual}"
        )

        # Every pairwise affine generator combination is represented by the
        # computed convex hull, by construction; check deterministic replay.
        permuted = convex_hull(affine_product(list(reversed(left)),
                                               list(reversed(right)), lam))
        assert permuted == actual
        results.append(actual)

    return results


def verify_one_sided_knot(data: dict[str, Any], knots: dict[str, dict[str, Any]]) -> None:
    expected = data["expected"]
    middle = knots[expected["middle_knot_id"]]
    incoming = resolve_endpoint(middle, "incoming_minus")
    outgoing = resolve_endpoint(middle, "outgoing_plus")
    assert incoming == decode_hull(expected["incoming_left_limit"])
    assert outgoing == decode_hull(expected["outgoing_right_limit"])
    assert incoming != outgoing
    assert expected["one_sided_sets_are_distinct"] is True


def verify_negative_controls(data: dict[str, Any], knots: dict[str, dict[str, Any]]) -> None:
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls == {
        "swap-endpoint-signs": "reject",
        "reuse-middle-outgoing-as-interval-0-right": "reject",
        "collapse-middle-incoming-and-outgoing": "reject-without-evidence",
        "missing-generator": "reject-incomplete-hull",
        "lambda-outside-unit-interval": "reject",
        "metadata-mismatch": "not-evaluable",
        "statistical-as-hard-set": "reject-claim-class",
        "sampled-polytope-as-full-containment": "reject",
    }

    first = data["intervals"][0]
    try:
        invalid = dict(first)
        invalid["right_endpoint_semantics"] = "outgoing_plus"
        verify_interval(invalid, knots)
    except ValueError:
        pass
    else:
        raise AssertionError("endpoint sign swap was not rejected")

    try:
        affine_product([(Fraction(0), Fraction(0))],
                       [(Fraction(1), Fraction(1))], Fraction(3, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("lambda outside [0,1] was not rejected")

    try:
        convex_hull([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty endpoint generator set was not rejected")

    middle = knots[data["expected"]["middle_knot_id"]]
    incoming = resolve_endpoint(middle, "incoming_minus")
    outgoing = resolve_endpoint(middle, "outgoing_plus")
    assert incoming != outgoing, "one-sided sets were incorrectly collapsed"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert_metadata(data)

    knots_list = data["knots"]
    knot_ids = [knot["knot_id"] for knot in knots_list]
    assert len(knot_ids) == len(set(knot_ids))
    knots = {knot["knot_id"]: knot for knot in knots_list}
    frequencies = [q(knot["frequency"]) for knot in knots_list]
    assert frequencies == sorted(frequencies)
    assert len(knots_list) == 3

    intervals = data["intervals"]
    assert len(intervals) == 2
    assert intervals[0]["right_knot_id"] == intervals[1]["left_knot_id"]
    assert intervals[0]["right_knot_id"] == data["expected"]["middle_knot_id"]

    results = [
        verify_interval(interval, knots)
        for interval in intervals
    ]
    assert results[0][1] == decode_hull(data["expected"]["lambda_half_hull_i0"])
    assert results[1][1] == decode_hull(data["expected"]["lambda_half_hull_i1"])
    verify_one_sided_knot(data, knots)
    verify_negative_controls(data, knots)

    print("AIRROAD one-sided set interpolation exact-rational reference: PASS")
    print("interval i0 lambda=1/2 hull:", [encoded_point(p) for p in results[0][1]])
    print("interval i1 lambda=1/2 hull:", [encoded_point(p) for p in results[1][1]])
    print("middle knot incoming-minus and outgoing-plus sets remain distinct")


if __name__ == "__main__":
    main()
