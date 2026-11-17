#!/usr/bin/env python3
"""Exact, dependency-light fixture checks for SYMM-Q-001.

This is benchmark/test tooling only. It does not implement a production
symmetry learner. Decimal JSON numbers are interpreted as exact base-10
rationals via Fraction(str(value)).
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "v0.1" / "cases.json"
SCHEMA = ROOT / "schemas" / "case_manifest.schema.json"
GENERATOR = ROOT / "tools" / "generate_d4_pointsets.py"


def fail(message: str) -> None:
    raise SystemExit(f"SYMM_Q_FIXTURE_FAIL: {message}")


def fraction(value: int | float | str) -> Fraction:
    if isinstance(value, bool):
        fail("boolean is not a numeric fixture value")
    return Fraction(str(value))


def matrix_key(matrix: list[list[int]]) -> tuple[int, int, int, int]:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        fail(f"expected 2x2 matrix, got {matrix!r}")
    flattened = tuple(matrix[0] + matrix[1])
    if any(not isinstance(value, int) or isinstance(value, bool) for value in flattened):
        fail(f"group matrix entries must be exact integers: {matrix!r}")
    return flattened  # type: ignore[return-value]


def matrix_multiply(left: tuple[int, int, int, int], right: tuple[int, int, int, int]):
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def matrix_apply(matrix: tuple[int, int, int, int], point: tuple[Fraction, Fraction]):
    a, b, c, d = matrix
    x, y = point
    return (a * x + b * y, c * x + d * y)


def parse_group(ambient: dict):
    elements = ambient["elements"]
    element_ids = [element["id"] for element in elements]
    if len(element_ids) != len(set(element_ids)):
        fail("duplicate group element id")
    matrices = {element["id"]: matrix_key(element["matrix"]) for element in elements}
    if len(set(matrices.values())) != len(matrices):
        fail("duplicate group matrix")

    declared_order = ambient["group_order"]
    if declared_order != len(elements):
        fail(f"group_order={declared_order}, but found {len(elements)} elements")

    identity_id = ambient["identity_element_id"]
    if identity_id not in matrices or matrices[identity_id] != (1, 0, 0, 1):
        fail("declared identity is not the 2x2 identity")

    id_by_matrix = {matrix: element_id for element_id, matrix in matrices.items()}
    declared_products = {}
    for record in ambient["multiplication_table"]:
        key = (record["left_element_id"], record["right_element_id"])
        if key in declared_products:
            fail(f"duplicate multiplication entry: {key}")
        declared_products[key] = record["product_element_id"]

    expected_pairs = {(left, right) for left in element_ids for right in element_ids}
    if set(declared_products) != expected_pairs:
        fail("multiplication table does not contain exactly one entry for every ordered pair")

    product_table = {}
    for left in element_ids:
        for right in element_ids:
            product_matrix = matrix_multiply(matrices[left], matrices[right])
            product_id = id_by_matrix.get(product_matrix)
            if product_id is None:
                fail(f"ambient matrices are not closed: {left} * {right}")
            declared = declared_products[(left, right)]
            if declared != product_id:
                fail(f"wrong product witness for {left} * {right}: {declared} != {product_id}")
            product_table[(left, right)] = product_id

    inverses = {}
    for record in ambient["inverse_table"]:
        element_id = record["element_id"]
        inverse_id = record["inverse_element_id"]
        if element_id in inverses:
            fail(f"duplicate inverse entry for {element_id}")
        if element_id not in matrices or inverse_id not in matrices:
            fail(f"inverse table references unknown element: {record}")
        if product_table[(element_id, inverse_id)] != identity_id:
            fail(f"right inverse witness is invalid for {element_id}")
        if product_table[(inverse_id, element_id)] != identity_id:
            fail(f"left inverse witness is invalid for {element_id}")
        inverses[element_id] = inverse_id
    if set(inverses) != set(element_ids):
        fail("inverse table does not cover every group element")

    # Matrix multiplication is associative; still check the frozen table itself.
    for a in element_ids:
        for b in element_ids:
            for c in element_ids:
                left = product_table[(product_table[(a, b)], c)]
                right = product_table[(a, product_table[(b, c)])]
                if left != right:
                    fail(f"multiplication table is not associative at ({a}, {b}, {c})")

    for element_id in element_ids:
        if product_table[(identity_id, element_id)] != element_id:
            fail(f"left identity failed for {element_id}")
        if product_table[(element_id, identity_id)] != element_id:
            fail(f"right identity failed for {element_id}")

    return element_ids, matrices, product_table, inverses


def exact_point_set_stabilizer(points, matrices):
    parsed_points = [
        (fraction(point[0]), fraction(point[1]))
        for point in points
    ]
    point_set = set(parsed_points)
    if len(point_set) != len(parsed_points):
        fail("point-set fixture contains duplicate points")
    return sorted(
        element_id
        for element_id, matrix in matrices.items()
        if {matrix_apply(matrix, point) for point in parsed_points} == point_set
    )


def polynomial_map(terms):
    result = defaultdict(Fraction)
    for term in terms:
        exponents = term["exponents"]
        if len(exponents) != 2 or any(
            not isinstance(power, int) or isinstance(power, bool) or power < 0
            for power in exponents
        ):
            fail(f"invalid polynomial exponents: {exponents!r}")
        result[tuple(exponents)] += fraction(term["coefficient"])
    return {monomial: coefficient for monomial, coefficient in result.items() if coefficient}


def binomial(n: int, k: int) -> int:
    return math.comb(n, k)


def transform_polynomial(terms, matrix):
    a, b, c, d = matrix
    transformed = defaultdict(Fraction)
    for term in terms:
        power_x, power_y = term["exponents"]
        coefficient = fraction(term["coefficient"])
        for choose_x in range(power_x + 1):
            x_factor = (
                binomial(power_x, choose_x)
                * (a ** choose_x)
                * (b ** (power_x - choose_x))
            )
            for choose_y in range(power_y + 1):
                y_factor = (
                    binomial(power_y, choose_y)
                    * (c ** choose_y)
                    * (d ** (power_y - choose_y))
                )
                monomial = (
                    choose_x + choose_y,
                    (power_x - choose_x) + (power_y - choose_y),
                )
                transformed[monomial] += coefficient * x_factor * y_factor
    return {monomial: coefficient for monomial, coefficient in transformed.items() if coefficient}


def polynomial_stabilizer(terms, matrices):
    original = polynomial_map(terms)
    return sorted(
        element_id
        for element_id, matrix in matrices.items()
        if transform_polynomial(terms, matrix) == original
    )


def validate_case(case, matrices):
    case_id = case["case_id"]
    expected = sorted(case["target_subgroup"])
    if len(expected) != len(set(expected)):
        fail(f"{case_id}: target subgroup contains duplicates")
    unknown = set(expected) - set(matrices)
    if unknown:
        fail(f"{case_id}: unknown target elements {sorted(unknown)}")

    if case["track"] == "exact_finite_structural_stabilizer":
        actual = exact_point_set_stabilizer(case["geometry"]["points"], matrices)
    elif case["track"] == "finite_reference_group_regression_discovery":
        actual = polynomial_stabilizer(case["polynomial"]["terms"], matrices)
    else:
        fail(f"{case_id}: unexpected exact fixture track {case['track']!r}")

    if actual != expected:
        fail(f"{case_id}: expected stabilizer {expected}, got {actual}")
    return case_id, actual


def main():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        fail(f"jsonschema is required by the pinned benchmark workflow: {error}")

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(fixture)

    spec = importlib.util.spec_from_file_location("d4_pointset_generator", GENERATOR)
    if spec is None or spec.loader is None:
        fail("could not load deterministic D4 generator")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated = generator.generate_cases(fixture)
    stored_generated = [
        case for case in fixture["cases"]
        if case["case_id"].startswith("GENERATED-")
    ]
    if stored_generated != generated:
        fail("generated point-set corpus differs from deterministic generator output")

    ambient = fixture["ambient_group"]
    element_ids, matrices, product_table, inverses = parse_group(ambient)
    case_ids = [case["case_id"] for case in fixture["cases"]]
    if len(case_ids) != len(set(case_ids)):
        fail("duplicate case_id")

    results = [validate_case(case, matrices) for case in fixture["cases"]]
    split = fixture["split_contract"]
    split_ids = [
        split["discovery_split_id"],
        split["calibration_split_id"],
        split["final_holdout_split_id"],
    ]
    if len(split_ids) != len(set(split_ids)):
        fail("discovery/calibration/final holdout split IDs must be distinct")

    print("SYMM_Q_FIXTURE_PASS")
    print(f"group={ambient['group_id']} order={len(element_ids)} products={len(product_table)} inverse_witnesses={len(inverses)}")
    for case_id, stabilizer in results:
        print(f"{case_id}: {','.join(stabilizer)}")
    print(f"generated_exact_cases={len(generated)}; families=5; cases_per_family=30")
    print("benchmark_status=fixture_validation_only; external_baseline_runs=not_started")


if __name__ == "__main__":
    main()
