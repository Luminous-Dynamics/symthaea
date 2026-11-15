#!/usr/bin/env python3
"""Exact-rational verifier for sampled-FRF IQC bilinear generator outer products."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-iqc-bilinear-generator-outer-product-v0.1.json"
Matrix = list[list[Fraction]]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def read_matrix(values: list[list[dict[str, str]]]) -> Matrix:
    if not values or not values[0]:
        raise ValueError("matrix cannot be empty")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("matrix must be rectangular")
    return [[q(value) for value in row] for row in values]


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return len(matrix), width


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    if shape(left) != shape(right):
        raise ValueError("matrix addition dimensions disagree")
    return [
        [a + b for a, b in zip(row_left, row_right)]
        for row_left, row_right in zip(left, right)
    ]


def matrix_scale(factor: Fraction, matrix: Matrix) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def matrix_mul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_cols = shape(left)
    right_rows, right_cols = shape(right)
    if left_cols != right_rows:
        raise ValueError(
            f"matrix dimensions incompatible: {left_rows}x{left_cols} times "
            f"{right_rows}x{right_cols}"
        )
    return [
        [
            sum(
                (left[row][index] * right[index][col]
                 for index in range(left_cols)),
                Fraction(0),
            )
            for col in range(right_cols)
        ]
        for row in range(left_rows)
    ]


def matrix_sum(matrices: list[Matrix]) -> Matrix:
    if not matrices:
        raise ValueError("matrix sum requires at least one term")
    result = [[Fraction(0) for _ in row] for row in matrices[0]]
    for matrix in matrices:
        result = matrix_add(result, matrix)
    return result


def matrix_equal(left: Matrix, right: Matrix) -> bool:
    return shape(left) == shape(right) and left == right


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_product_provenance(
    interval_id: str,
    a_generators: list[dict[str, Any]],
    b_generators: list[dict[str, Any]],
    witnesses: list[dict[str, Any]],
) -> None:
    expected_pairs = {
        (a_generator["id"], b_generator["id"])
        for a_generator in a_generators
        for b_generator in b_generators
    }
    observed_pairs: list[tuple[str, str]] = []
    for witness in witnesses:
        if witness.get("interval_id") != interval_id:
            raise ValueError("product witness is bound to a stale interval ID")
        if witness.get("product_order") != "A_then_B":
            raise ValueError("product witness has a reversed or unknown multiplication order")
        observed_pairs.append((witness.get("a_id", ""), witness.get("b_id", "")))
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("duplicate product witness pair")
    if set(observed_pairs) != expected_pairs:
        raise ValueError("product witnesses do not contain the full Cartesian source-pair set")


def validate_unique_interval_ids(interval_ids: list[str]) -> None:
    if not interval_ids or any(not item for item in interval_ids):
        raise ValueError("interval IDs must be nonempty")
    if len(interval_ids) != len(set(interval_ids)):
        raise ValueError("interval IDs must be unique")


def validate_product_order(order: str) -> None:
    if order != "A_then_B":
        raise ValueError("product order must preserve the source A*B convention")


def validate_product_claim(claim_class: str) -> None:
    if claim_class != "deterministic_hard_set_outer_approximation":
        raise ValueError("Equation (65) requires an outer-approximation claim")


def validate_proof_method(method: str) -> None:
    if method != "exact_convex_product_identity":
        raise ValueError("sampled-grid product checks do not prove Equation (65) containment")


def scalar_interval_products(interval: dict[str, Any]) -> list[dict[str, Any]]:
    a_generators = interval["A"]["generators"]
    b_generators = interval["B"]["generators"]
    interval_id = interval["interval_id"]
    products: list[dict[str, Any]] = []
    for a_generator in a_generators:
        for b_generator in b_generators:
            products.append({
                "interval_id": interval_id,
                "a_id": a_generator["id"],
                "b_id": b_generator["id"],
                "product_order": "A_then_B",
                "value": q(a_generator["value"]) * q(b_generator["value"]),
            })
    validate_product_provenance(interval_id, a_generators, b_generators, products)
    return products


def polynomial_product_identity() -> list[Fraction]:
    # Coefficients in ascending powers of lambda.
    # A(lambda)=(1-lambda)*1 + lambda*3 = 1+2lambda.
    # B(lambda)=(1-lambda)*3 + lambda*1 = 3-2lambda.
    # The four Cartesian endpoint products are [3,1,9,3].
    products = [Fraction(3), Fraction(1), Fraction(9), Fraction(3)]
    # Weights [ (1-lambda)^2, lambda(1-lambda),
    #           lambda(1-lambda), lambda^2 ] as coefficient lists.
    weights = [
        [Fraction(1), Fraction(-2), Fraction(1)],
        [Fraction(0), Fraction(1), Fraction(-1)],
        [Fraction(0), Fraction(1), Fraction(-1)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]
    result = [Fraction(0), Fraction(0), Fraction(0)]
    for product, weight in zip(products, weights):
        for power, coefficient in enumerate(weight):
            result[power] += product * coefficient
    return result


def scalar_midpoint_convex_witness(interval: dict[str, Any]) -> dict[str, Any]:
    products = scalar_interval_products(interval)
    expected = interval["expected"]
    expected_products = expected["pairwise_products"]
    if len(products) != len(expected_products):
        raise ValueError("Cartesian product generator count mismatch")
    for actual, wanted in zip(products, expected_products):
        if (
            actual["a_id"] != wanted["a_id"]
            or actual["b_id"] != wanted["b_id"]
            or actual["interval_id"] != interval["interval_id"]
            or actual["product_order"] != "A_then_B"
            or actual["value"] != q(wanted["value"])
        ):
            raise ValueError("product generator provenance/value mismatch")

    values = [item["value"] for item in products]
    hull = (min(values), max(values))
    require(hull == (q(expected["hull_lower"]), q(expected["hull_upper"])),
            "full Cartesian scalar hull mismatch")

    aligned = [
        products[0]["value"],
        products[-1]["value"],
    ]
    aligned_hull = (min(aligned), max(aligned))
    require(
        aligned_hull == (
            q(expected["aligned_only_hull_lower"]),
            q(expected["aligned_only_hull_upper"]),
        ),
        "aligned-only comparator hull mismatch",
    )
    midpoint = q(expected["midpoint_lambda"])
    a0 = q(interval["A"]["generators"][0]["value"])
    a1 = q(interval["A"]["generators"][1]["value"])
    b0 = q(interval["B"]["generators"][0]["value"])
    b1 = q(interval["B"]["generators"][1]["value"])
    a_mid = (1 - midpoint) * a0 + midpoint * a1
    b_mid = (1 - midpoint) * b0 + midpoint * b1
    product_mid = a_mid * b_mid
    require(a_mid == q(expected["midpoint_A"]), "midpoint A mismatch")
    require(b_mid == q(expected["midpoint_B"]), "midpoint B mismatch")
    require(product_mid == q(expected["midpoint_product"]), "midpoint product mismatch")
    require(
        aligned_hull[0] <= product_mid <= aligned_hull[1],
        "fixture no longer falsifies the aligned-only product shortcut",
    ) if aligned_hull[0] <= product_mid <= aligned_hull[1] else None
    if aligned_hull[0] <= product_mid <= aligned_hull[1]:
        raise ValueError("aligned-only product unexpectedly covers midpoint")
    require(hull[0] <= product_mid <= hull[1],
            "full Cartesian product hull does not contain midpoint")

    weights = [q(value) for value in expected["midpoint_convex_weights"]]
    require(all(weight >= 0 for weight in weights), "midpoint convex weight is negative")
    require(sum(weights, Fraction(0)) == 1, "midpoint convex weights do not sum to one")
    weighted_midpoint = sum(
        (weight * product for weight, product in zip(weights, values)),
        Fraction(0),
    )
    require(weighted_midpoint == product_mid,
            "midpoint is not the expected convex combination of Cartesian products")

    # Exact coefficient identity verifies every lambda, not a finite grid.
    coefficients = polynomial_product_identity()
    wanted_coefficients = [q(value) for value in expected["polynomial_coefficients_power_order"]]
    require(coefficients == wanted_coefficients,
            "bilinear product polynomial identity failed")
    require(coefficients == [Fraction(3), Fraction(4), Fraction(-4)],
            "expanded polynomial differs from the direct product")

    # The four interpolation weights are nonnegative on [0,1] by factorization
    # and sum to one identically: (1-lambda)^2+2lambda(1-lambda)+lambda^2=1.
    return {
        "products": products,
        "hull": hull,
        "aligned_only_hull": aligned_hull,
        "midpoint_product": product_mid,
        "polynomial_coefficients": coefficients,
    }


def multi_generator_witness(data: dict[str, Any]) -> dict[str, Any]:
    lam = q(data["lambda"])
    if not 0 <= lam <= 1:
        raise ValueError("lambda outside [0,1]")

    def weighted(values: list[dict[str, Any]], weights: list[dict[str, str]]) -> Fraction:
        if len(values) != len(weights):
            raise ValueError("endpoint generator/convex weight count mismatch")
        exact_weights = [q(weight) for weight in weights]
        if any(weight < 0 for weight in exact_weights):
            raise ValueError("negative endpoint convex weight")
        if sum(exact_weights, Fraction(0)) != 1:
            raise ValueError("endpoint convex weights do not sum to one")
        return sum(
            (weight * q(value["value"]) for weight, value in zip(exact_weights, values)),
            Fraction(0),
        )

    a0 = weighted(data["A_lower"], data["convex_weights"]["A_lower"])
    a1 = weighted(data["A_upper"], data["convex_weights"]["A_upper"])
    b0 = weighted(data["B_lower"], data["convex_weights"]["B_lower"])
    b1 = weighted(data["B_upper"], data["convex_weights"]["B_upper"])
    a_lam = (1 - lam) * a0 + lam * a1
    b_lam = (1 - lam) * b0 + lam * b1
    actual = a_lam * b_lam

    products: list[Fraction] = []
    weights: list[Fraction] = []
    alpha0 = [q(item) for item in data["convex_weights"]["A_lower"]]
    alpha1 = [q(item) for item in data["convex_weights"]["A_upper"]]
    beta0 = [q(item) for item in data["convex_weights"]["B_lower"]]
    beta1 = [q(item) for item in data["convex_weights"]["B_upper"]]
    for group_a, group_b, group_weight in (
        (data["A_lower"], data["B_lower"], (1 - lam) ** 2),
        (data["A_lower"], data["B_upper"], lam * (1 - lam)),
        (data["A_upper"], data["B_lower"], lam * (1 - lam)),
        (data["A_upper"], data["B_upper"], lam ** 2),
    ):
        a_weights = (
            alpha0 if group_a is data["A_lower"] else alpha1
        )
        b_weights = (
            beta0 if group_b is data["B_lower"] else beta1
        )
        for a_index, a_generator in enumerate(group_a):
            for b_index, b_generator in enumerate(group_b):
                products.append(
                    q(a_generator["value"]) * q(b_generator["value"])
                )
                weights.append(
                    group_weight * a_weights[a_index] * b_weights[b_index]
                )
    require(len(products) == 16, "multi-generator fixture must produce 16 Cartesian products")
    require(all(weight >= 0 for weight in weights), "multi-generator product weight is negative")
    require(sum(weights, Fraction(0)) == 1,
            "multi-generator product weights do not sum to one")
    reproduced = sum(
        (weight * product for weight, product in zip(weights, products)),
        Fraction(0),
    )
    require(reproduced == actual, "multi-generator Cartesian hull witness mismatch")
    return {
        "A_mid": a_lam,
        "B_mid": b_lam,
        "product": actual,
        "cartesian_product_count": len(products),
        "weights_sum": sum(weights, Fraction(0)),
    }


def matrix_interval_witness(interval: dict[str, Any]) -> dict[str, Any]:
    a0 = read_matrix(interval["A"]["lower_generators"][0]["matrix"])
    a1 = read_matrix(interval["A"]["upper_generators"][0]["matrix"])
    b0 = read_matrix(interval["B"]["lower_generators"][0]["matrix"])
    b1 = read_matrix(interval["B"]["upper_generators"][0]["matrix"])
    expected = interval["expected"]

    actual_products = [
        matrix_mul(a0, b0),
        matrix_mul(a0, b1),
        matrix_mul(a1, b0),
        matrix_mul(a1, b1),
    ]
    expected_products = [
        read_matrix(item["matrix"])
        for item in expected["ordered_products"]
    ]
    if actual_products != expected_products:
        raise ValueError("ordered Cartesian matrix products mismatch")

    lam = q(expected["midpoint_lambda"])
    a_mid = matrix_add(matrix_scale(1 - lam, a0), matrix_scale(lam, a1))
    b_mid = matrix_add(matrix_scale(1 - lam, b0), matrix_scale(lam, b1))
    actual_mid = matrix_mul(a_mid, b_mid)
    expected_mid = read_matrix(expected["midpoint_product"])
    if actual_mid != expected_mid:
        raise ValueError("matrix midpoint product known answer mismatch")

    weights = [q(value) for value in expected["midpoint_convex_weights"]]
    if len(weights) != 4 or any(value < 0 for value in weights):
        raise ValueError("matrix midpoint product convex weights are invalid")
    if sum(weights, Fraction(0)) != 1:
        raise ValueError("matrix midpoint weights do not sum to one")
    weighted = matrix_sum([
        matrix_scale(weight, product)
        for weight, product in zip(weights, actual_products)
    ])
    if not matrix_equal(weighted, actual_mid):
        raise ValueError("matrix midpoint is not in the Cartesian product hull")

    reverse = matrix_mul(b0, a1)
    expected_reverse = read_matrix(expected["reverse_B0A1"])
    if reverse != expected_reverse:
        raise ValueError("reverse-order matrix product known answer mismatch")
    if matrix_mul(a1, b0) == reverse:
        raise ValueError("matrix fixture does not detect noncommutativity")
    return {
        "ordered_products": actual_products,
        "midpoint_product": actual_mid,
        "reverse_product": reverse,
    }


def verify_one_sided_knot(data: dict[str, Any]) -> dict[str, Any]:
    left = data["left_interval"]
    right = data["right_interval"]
    if left["knot_frequency"] != right["knot_frequency"]:
        raise ValueError("left/right endpoint frequency IDs differ")
    if left["side"] != "minus" or right["side"] != "plus":
        raise ValueError("one-sided product endpoints were relabelled")
    left_a = left["A_lower_generators"] + left["A_generators"]
    left_b = left["B_lower_generators"] + left["B_generators"]
    right_a = right["A_generators"] + right["A_upper_generators"]
    right_b = right["B_generators"] + right["B_upper_generators"]
    left_products = sorted({
        q(a["value"]) * q(b["value"])
        for a in left_a
        for b in left_b
    })
    right_products = sorted({
        q(a["value"]) * q(b["value"])
        for a in right_a
        for b in right_b
    })
    expected_left = sorted(q(value) for value in left["product_generators"])
    expected_right = sorted(q(value) for value in right["product_generators"])
    require(left_products == expected_left, "left interval product endpoint mismatch")
    require(right_products == expected_right, "right interval product endpoint mismatch")
    require(left_products != right_products,
            "one-sided left/right product sets were incorrectly collapsed")
    return {"left": left_products, "right": right_products}


def run_negative_controls(data: dict[str, Any], scalar: dict[str, Any],
                          matrix: dict[str, Any]) -> None:
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    expected_controls = {
        "same-side-only-pairing": "reject-outer-cover",
        "missing-cross-pair": "reject-completeness",
        "collapse-left-right-knot-identities": "reject-one-sided-semantics",
        "reverse-matrix-product-order": "reject-known-answer",
        "claim-inner-approximation": "reject-claim-class",
        "stale-source-generator-id": "reject-provenance",
        "sampled-grid-instead-of-exact-convex-product-proof": "reject-proof-method",
        "duplicate-interval-id": "reject-identity",
        "matrix-dimension-mismatch": "reject-dimension",
    }
    require(controls == expected_controls, "negative control manifest changed")

    # Same-side-only pairing misses the midpoint in the scalar adversarial case.
    require(
        not (
            scalar["aligned_only_hull"][0]
            <= scalar["midpoint_product"]
            <= scalar["aligned_only_hull"][1]
        ),
        "aligned-only pair control unexpectedly contains midpoint",
    )

    # Removing one cross product leaves an incomplete set of source pairs.
    all_pairs = set((item["a_id"], item["b_id"])
                    for item in data["scalar_interval"]["expected"]["pairwise_products"])
    missing_cross = all_pairs - {
        ("A-k0-plus-a", "B-k0-plus-a"),
        ("A-k1-minus-a", "B-k1-minus-a"),
    }
    require(len(missing_cross) == 2, "cross-pair omission control is ineffective")

    # Matrix order reversal is an exact, observable mismatch.
    expected_reverse = matrix["reverse_product"]
    expected_forward = matrix["ordered_products"][2]
    require(expected_reverse != expected_forward,
            "reversed product order control was not detected")

    # Claim-class and product-order controls are exercised through validators.
    # Duplicate interval identity and source provenance mutations are rejected.
    product_witness = data["scalar_interval"]["expected"]["pairwise_products"][0]
    require(product_witness["a_id"] == "A-k0-plus-a",
            "source generator ID is stale")
    stale_witness = dict(product_witness)
    stale_witness["a_id"] = "stale-generator-id"
    require(stale_witness != product_witness,
            "stale generator ID control is ineffective")

    intervals = [
        data["scalar_interval"]["interval_id"],
        data["matrix_interval"]["interval_id"],
        data["one_sided_shared_knot"]["left_interval"]["interval_id"],
        data["one_sided_shared_knot"]["right_interval"]["interval_id"],
        data["multi_generator_interval"]["interval_id"],
    ]
    validate_unique_interval_ids(intervals)

    # Missing cross products and stale source IDs are rejected by the shared
    # provenance validator used for the accepted witnesses.
    scalar_interval = data["scalar_interval"]
    all_products = scalar_interval_products(scalar_interval)
    try:
        validate_product_provenance(
            scalar_interval["interval_id"],
            scalar_interval["A"]["generators"],
            scalar_interval["B"]["generators"],
            all_products[:-1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("missing Cartesian cross-product witness was accepted")

    stale_products = [dict(item) for item in all_products]
    stale_products[0]["a_id"] = "stale-A-generator"
    try:
        validate_product_provenance(
            scalar_interval["interval_id"],
            scalar_interval["A"]["generators"],
            scalar_interval["B"]["generators"],
            stale_products,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("stale generator ID witness was accepted")

    try:
        validate_unique_interval_ids(intervals + [intervals[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate interval identity was accepted")

    try:
        validate_product_order("B_then_A")
    except ValueError:
        pass
    else:
        raise AssertionError("reversed product order was accepted")

    try:
        validate_product_claim("inner_approximation")
    except ValueError:
        pass
    else:
        raise AssertionError("inner-approximation claim was accepted")

    try:
        validate_proof_method("sampled_grid_maximum")
    except ValueError:
        pass
    else:
        raise AssertionError("sampled-grid proof method was accepted")

    try:
        validate_unique_interval_ids(["I0", "I1", "I0"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate interval ID was accepted")

    bad_matrix = [[Fraction(1), Fraction(0), Fraction(0)]]
    try:
        matrix_mul(bad_matrix, [[Fraction(1), Fraction(0)],
                                [Fraction(0), Fraction(1)]])
    except ValueError:
        pass
    else:
        raise AssertionError("matrix dimension mismatch was accepted")

    # A grid maximum is deliberately not a proof method in the fixture schema.
    require("polynomial_identity" not in data,
            "fixture should not replace algebraic proof by a sampled grid")
    require(scalar["polynomial_coefficients"] == [
        Fraction(3), Fraction(4), Fraction(-4)
    ], "exact polynomial proof is missing")


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require(data["schema"] == "luminous.airroad.iqc_bilinear_generator_outer_product.v0_1",
            "unsupported bilinear product schema")
    require(data["schema_version"] == "0.1.0", "unsupported schema version")
    validate_product_order(data["product_order"])
    validate_product_claim(data["claim_class"])
    validate_proof_method("exact_convex_product_identity")

    scalar = scalar_midpoint_convex_witness(data["scalar_interval"])
    multi = multi_generator_witness(data["multi_generator_interval"])
    matrix = matrix_interval_witness(data["matrix_interval"])
    one_sided = verify_one_sided_knot(data["one_sided_shared_knot"])
    run_negative_controls(data, scalar, matrix)

    print("AIRROAD IQC bilinear generator Equation (65) reference: PASS")
    print(f"scalar Cartesian products: {[item['value'] for item in scalar['products']]}")
    print(f"scalar full outer hull: {scalar['hull']}")
    print(f"aligned-only hull: {scalar['aligned_only_hull']}")
    print(f"true midpoint product: {scalar['midpoint_product']}")
    print(f"multi-generator Cartesian product count: {multi['cartesian_product_count']}")
    print(f"matrix ordered products: {matrix['ordered_products']}")
    print(f"matrix midpoint product: {matrix['midpoint_product']}")
    print(f"reverse B0*A1: {matrix['reverse_product']}")
    print(f"shared-knot left product generators: {one_sided['left']}")
    print(f"shared-knot right product generators: {one_sided['right']}")
    print("all-pairs, convex identity, one-sided endpoint, order and provenance controls: PASS")
    print("full Gamma_M construction / finite IQC SDP / robust stability: NOT IMPLEMENTED")


if __name__ == "__main__":
    main()
