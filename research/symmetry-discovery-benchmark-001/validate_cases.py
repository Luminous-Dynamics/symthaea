#!/usr/bin/env python3
"""Exact, dependency-free validation of the frozen D4 benchmark fixtures.

This is a benchmark-fixture checker only. It is not part of Symthaea runtime.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "fixtures" / "v0.1" / "cases.json"
SCHEMA = ROOT / "schemas" / "case_manifest.schema.json"


def fail(message: str) -> None:
    raise SystemExit(f"benchmark fixture validation failed: {message}")


def as_fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        fail("boolean is not a numeric fixture value")
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    fail(f"unsupported numeric value: {value!r}")


def matrix_key(matrix: list[list[int]]) -> tuple[int, ...]:
    return tuple(value for row in matrix for value in row)


def matrix_product(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(2)) for column in range(2)]
        for row in range(2)
    ]


def matrix_vector(matrix: list[list[int]], vector: list[Fraction]) -> list[Fraction]:
    return [
        sum(Fraction(matrix[row][column]) * vector[column] for column in range(2))
        for row in range(2)
    ]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"could not parse {path}: {error}")


def validate_manifest_shape(manifest: dict, schema: dict) -> None:
    required = set(schema["required"])
    missing = sorted(required - manifest.keys())
    if missing:
        fail(f"manifest is missing top-level fields: {missing}")
    if manifest["schema_version"] != "0.1.0":
        fail("unexpected schema_version")
    if manifest["benchmark_id"] != "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001":
        fail("unexpected benchmark_id")
    if manifest["freeze_status"] != "protocol_fixtures_not_benchmark_results":
        fail("fixture manifest must not claim benchmark execution")
    split = manifest["split_contract"]
    split_ids = [
        split["discovery_split_id"],
        split["calibration_split_id"],
        split["final_holdout_split_id"],
    ]
    if len(set(split_ids)) != 3:
        fail("discovery, calibration and final holdout split IDs must be distinct")
    if any(not value.startswith("pending-") for value in split_ids):
        fail("fixture split IDs must remain unassigned until split digests are frozen")
    case_ids = [case["case_id"] for case in manifest["cases"]]
    if len(set(case_ids)) != len(case_ids):
        fail("duplicate case_id")
    if not case_ids:
        fail("no benchmark cases were declared")


def validate_group(group: dict) -> tuple[list[dict], dict[tuple[int, ...], str]]:
    elements = group["elements"]
    ids = [element["id"] for element in elements]
    if len(set(ids)) != len(ids):
        fail("duplicate group element id")
    by_matrix = {}
    for element in elements:
        matrix = element["matrix"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            fail(f"invalid 2x2 matrix for {element['id']}")
        if any(not isinstance(value, int) for row in matrix for value in row):
            fail(f"non-integer matrix entry for {element['id']}")
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if abs(determinant) != 1:
            fail(f"group element {element['id']} is not unimodular")
        key = matrix_key(matrix)
        if key in by_matrix:
            fail("duplicate group matrix")
        by_matrix[key] = element["id"]

    identity = ((1, 0), (0, 1))
    if matrix_key([list(row) for row in identity]) not in by_matrix:
        fail("ambient group lacks the identity")
    identity_id = by_matrix[matrix_key([list(row) for row in identity])]

    for left in elements:
        inverse_found = False
        for right in elements:
            product = matrix_product(left["matrix"], right["matrix"])
            if by_matrix.get(matrix_key(product)) == identity_id:
                inverse_found = True
                break
        if not inverse_found:
            fail(f"no inverse found for group element {left['id']}")
        for right in elements:
            product = matrix_product(left["matrix"], right["matrix"])
            if matrix_key(product) not in by_matrix:
                fail(f"group is not closed: {left['id']} * {right['id']}")
    return elements, by_matrix


def exact_point_set_stabilizer(points: list[list[object]], elements: list[dict]) -> list[str]:
    rational_points = [tuple(as_fraction(value) for value in point) for point in points]
    if any(len(point) != 2 for point in rational_points):
        fail("point-cloud fixture is not two-dimensional")
    if len(set(rational_points)) != len(rational_points):
        fail("point cloud contains duplicate points")
    original = set(rational_points)
    stabilizer = []
    for element in elements:
        transformed = {
            tuple(matrix_vector(element["matrix"], list(point)))
            for point in rational_points
        }
        if transformed == original:
            stabilizer.append(element["id"])
    return sorted(stabilizer)


def polynomial_map(terms: list[dict]) -> dict[tuple[int, int], Fraction]:
    result: dict[tuple[int, int], Fraction] = {}
    for term in terms:
        exponents = term["exponents"]
        if len(exponents) != 2 or any(not isinstance(value, int) or value < 0 for value in exponents):
            fail("invalid polynomial exponent tuple")
        key = (exponents[0], exponents[1])
        if key in result:
            fail(f"duplicate polynomial term {key}")
        coefficient = as_fraction(term["coefficient"])
        if coefficient == 0:
            fail(f"zero polynomial coefficient should be omitted: {key}")
        result[key] = coefficient
    return result


def transform_polynomial(
    terms: list[dict],
    matrix: list[list[int]],
) -> dict[tuple[int, int], Fraction]:
    """Expand f(M[x,y]) exactly using integer binomial coefficients."""
    result: dict[tuple[int, int], Fraction] = {}
    a, b = matrix[0]
    c, d = matrix[1]
    for term in terms:
        p, q = term["exponents"]
        coefficient = as_fraction(term["coefficient"])
        for i in range(p + 1):
            x_factor = Fraction(math.comb(p, i) * (a**i) * (b ** (p - i)))
            for j in range(q + 1):
                y_factor = Fraction(math.comb(q, j) * (c**j) * (d ** (q - j)))
                exponent = (i + j, (p - i) + (q - j))
                result[exponent] = result.get(exponent, Fraction(0)) + coefficient * x_factor * y_factor
    return {exponent: value for exponent, value in result.items() if value != 0}


def exact_polynomial_stabilizer(terms: list[dict], elements: list[dict]) -> list[str]:
    original = polynomial_map(terms)
    return sorted(
        element["id"]
        for element in elements
        if transform_polynomial(terms, element["matrix"]) == original
    )


def validate_cases(manifest: dict, elements: list[dict]) -> list[str]:
    known_ids = {element["id"] for element in elements}
    validated = []
    for case in manifest["cases"]:
        target = case["target_subgroup"]
        if not target or len(set(target)) != len(target) or not set(target) <= known_ids:
            fail(f"invalid target subgroup declaration in {case['case_id']}")
        if case["track"] == "exact_finite_structural_stabilizer":
            actual = exact_point_set_stabilizer(case["geometry"]["points"], elements)
        elif case["track"] == "finite_reference_group_regression_discovery":
            if case["ambient_group_id"] != manifest["ambient_group"]["group_id"]:
                fail(f"wrong ambient group in {case['case_id']}")
            actual = exact_polynomial_stabilizer(case["polynomial"]["terms"], elements)
        else:
            fail(f"unrecognized benchmark track in {case['case_id']}")
        expected = sorted(target)
        if actual != expected:
            fail(f"{case['case_id']} target mismatch: expected {expected}, exact oracle returned {actual}")
        validated.append(case["case_id"])
    return validated


def main() -> None:
    manifest = load_json(MANIFEST)
    schema = load_json(SCHEMA)
    validate_manifest_shape(manifest, schema)
    elements, _ = validate_group(manifest["ambient_group"])
    cases = validate_cases(manifest, elements)
    print(f"manifest/schema JSON parse: PASS")
    print(f"ambient group closure/inverses: PASS ({len(elements)} exact elements)")
    for case_id in cases:
        print(f"exact ground-truth oracle: PASS {case_id}")
    print(f"validated cases: {len(cases)}; benchmark runs: 0")


if __name__ == "__main__":
    main()
