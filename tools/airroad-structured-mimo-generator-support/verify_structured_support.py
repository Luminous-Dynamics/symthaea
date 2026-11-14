#!/usr/bin/env python3
"""Exact-rational proof checks for structured MIMO support and simplex outer cover."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-structured-mimo-generator-support-v0.1.json"
Vector = list[Fraction]
Matrix = list[list[Fraction]]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def decode_vector(values: list[dict[str, str]]) -> Vector:
    return [q(value) for value in values]


def decode_matrix(values: list[list[dict[str, str]]]) -> Matrix:
    if not values or not values[0]:
        raise ValueError("matrix cannot be empty")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("matrix must be rectangular")
    return [[q(value) for value in row] for row in values]


def dot(left: Vector, right: Vector) -> Fraction:
    if len(left) != len(right):
        raise ValueError("vector dimensions disagree")
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        raise ValueError("matrix cannot be empty")
    return [list(row) for row in zip(*matrix)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        raise ValueError("matrix product cannot have empty inputs")
    left_rows, left_cols = len(left), len(left[0])
    right_rows, right_cols = len(right), len(right[0])
    if any(len(row) != left_cols for row in left):
        raise ValueError("left matrix is ragged")
    if any(len(row) != right_cols for row in right):
        raise ValueError("right matrix is ragged")
    if left_cols != right_rows:
        raise ValueError("matrix product dimensions disagree")
    return [
        [
            sum((left[row][k] * right[k][col] for k in range(left_cols)),
                Fraction(0))
            for col in range(right_cols)
        ]
        for row in range(left_rows)
    ]


def rank(matrix: Matrix) -> int:
    if not matrix or not matrix[0]:
        return 0
    work = [row[:] for row in matrix]
    row_count, col_count = len(work), len(work[0])
    if any(len(row) != col_count for row in work):
        raise ValueError("rank matrix is ragged")
    pivot_row = 0
    for col in range(col_count):
        pivot = next(
            (row for row in range(pivot_row, row_count)
             if work[row][col] != 0),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][col]
        for index in range(col, col_count):
            work[pivot_row][index] /= pivot_value
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = work[row][col]
            if factor != 0:
                for index in range(col, col_count):
                    work[row][index] -= factor * work[pivot_row][index]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def line_intersection(
    left_normal: Vector,
    right_normal: Vector,
) -> Vector:
    if len(left_normal) != 2 or len(right_normal) != 2:
        raise ValueError("simplex normals must be two-dimensional")
    a, b = left_normal
    c, d = right_normal
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("parallel simplex boundaries have no unique vertex")
    # Solve [[a,b],[c,d]] q = [1,1] exactly.
    return [
        (d - b) / determinant,
        (a - c) / determinant,
    ]


def validate_structural_contract(
    data: dict[str, Any],
    basis: Matrix,
) -> dict[str, Any]:
    ambient = data["ambient"]
    contract = data["structural_contract"]
    support = data["support_basis"]

    expected_dimension = (
        2 * int(ambient["matrix_rows"]) * int(ambient["matrix_cols"])
    )
    if ambient["scalar_field"] != "complex":
        raise ValueError("ambient scalar field is not complex")
    if expected_dimension != int(ambient["real_dimension"]):
        raise ValueError("ambient real dimension does not match complex matrix shape")
    if ambient["real_dimension"] != 8:
        raise ValueError("V0.1 exact fixture expects ambient dimension 8")

    expected_order = [
        "Re00", "Re01", "Re10", "Re11",
        "Im00", "Im01", "Im10", "Im11",
    ]
    if ambient["vectorization_order"] != expected_order:
        raise ValueError("ambient vectorization order changed")

    if contract["authority_class"] != "declared_hard_structural_constraint":
        raise ValueError("support authority is not a hard structural contract")
    if contract["scope"] != "all_admissible_matrices_not_just_sampled_observations":
        raise ValueError("support declaration does not cover the entire admitted set")
    if contract["source_identity"] == "":
        raise ValueError("structural source identity is missing")
    if contract["exact_form"] != "Delta(alpha,beta) = diag(alpha,beta), alpha^2 + beta^2 <= 1":
        raise ValueError("the declared source set is not the frozen diagonal-real support family")
    if contract["model_identity"] != data["model_identity"]:
        raise ValueError("model identity mismatch in structural contract")
    if contract["topology_identity"] != data["topology_identity"]:
        raise ValueError("topology identity mismatch in structural contract")

    coordinates = contract["coordinate_frequency_bindings"]
    if not coordinates:
        raise ValueError("shared calibration coordinate bindings are missing")
    expected_frequency_ids = [
        item["frequency_id"] for item in data["coordinates"]
    ]
    bound_frequency_ids = [item["frequency_id"] for item in coordinates]
    if bound_frequency_ids != expected_frequency_ids:
        raise ValueError("shared calibration frequency bindings are incomplete or reordered")
    nuisance_ids = {item["calibration_nuisance_id"] for item in coordinates}
    if nuisance_ids != {contract["calibration_nuisance_id"]}:
        raise ValueError("calibration coordinates split across frequencies")

    if len(basis) != expected_dimension or not basis:
        raise ValueError("support basis ambient dimension mismatch")
    if any(len(vector) != int(support["support_dimension"]) for vector in basis):
        raise ValueError("support basis coordinate dimension mismatch")
    if rank(basis) != int(support["expected_rank"]):
        raise ValueError("support basis is not full column rank")
    if rank(basis) != int(support["support_dimension"]):
        raise ValueError("support rank disagrees with declared support dimension")

    gram = matmul(transpose(basis), basis)
    expected_gram = decode_matrix(support["expected_gram"])
    if gram != expected_gram:
        raise ValueError("support basis Frobenius Gram matrix is not identity")

    constraints = contract["hard_constraints"]
    if len(constraints) != 6:
        raise ValueError("V0.1 requires six annihilator constraints")
    constraint_indices = [int(item["coordinate_index"]) for item in constraints]
    if len(set(constraint_indices)) != len(constraint_indices):
        raise ValueError("duplicate hard support constraint")
    if set(constraint_indices) != {1, 2, 4, 5, 6, 7}:
        raise ValueError("hard annihilator constraints do not match diagonal-real support")
    for item in constraints:
        index = int(item["coordinate_index"])
        required = q(item["value"])
        if required != 0:
            raise ValueError("support annihilator must be an exact zero constraint")
        if any(vector[index] != 0 for vector in basis):
            raise ValueError("support basis violates a hard annihilator constraint")

    # Basis is an exact complete parameterization of the declared support class.
    # The source contract, not observed samples, states that every member is in
    # span(B); basis rank plus the annihilator constraints proves this support
    # has dimension exactly r for this frozen structural model.
    return {
        "ambient_dimension": expected_dimension,
        "support_dimension": rank(basis),
        "basis_gram": gram,
        "calibration_nuisance_ids": sorted(nuisance_ids),
        "proof_status": "STRUCTURAL_SUPPORT_PROVED_FOR_DECLARED_SYNTHETIC_CLASS",
    }


def validate_simplex(
    simplex: dict[str, Any],
    support_dimension: int,
) -> dict[str, Any]:
    if simplex["coordinate_dimension"] != support_dimension:
        raise ValueError("simplex dimension does not match proved support")
    normals = [decode_vector(row) for row in simplex["normals"]]
    vertices = [decode_vector(row) for row in simplex["vertices"]]
    if len(normals) != support_dimension + 1:
        raise ValueError("simplex requires r+1 facets in this V0.1 construction")
    if len(vertices) != support_dimension + 1:
        raise ValueError("simplex requires r+1 vertices")
    if simplex.get("inequalities") != [
        "n1 dot q <= 1", "n2 dot q <= 1", "n3 dot q <= 1"
    ]:
        raise ValueError("simplex half-space orientation is missing or reversed")

    for normal in normals:
        if dot(normal, normal) != 1:
            raise ValueError("simplex normal does not have exact unit norm")

    # Every vertex is the unique intersection of a pair of facet boundaries.
    derived = [
        line_intersection(normals[0], normals[1]),
        line_intersection(normals[0], normals[2]),
        line_intersection(normals[1], normals[2]),
    ]
    if vertices != derived:
        raise ValueError("rational simplex vertices do not match exact intersections")

    for vertex in vertices:
        for normal in normals:
            if dot(normal, vertex) > 1:
                raise ValueError("simplex vertex lies outside a half-space")

    # Positive combination of the unit normals is zero, proving their
    # directions positively span R^2 and the half-space intersection is bounded.
    positive_weights = [Fraction(6, 5), Fraction(1), Fraction(1)]
    normal_sum = [
        sum((positive_weights[index] * normals[index][coordinate]
             for index in range(3)), Fraction(0))
        for coordinate in range(2)
    ]
    if normal_sum != [Fraction(0), Fraction(0)]:
        raise ValueError("simplex normals do not prove boundedness")

    norms_squared = [dot(vertex, vertex) for vertex in vertices]
    max_norm_squared = max(norms_squared)
    if max_norm_squared != q(simplex["expected_maximum_vertex_norm_squared"]):
        raise ValueError("simplex maximum vertex norm mismatch")

    # Every support-unit-ball point obeys n_i^T q <= ||n_i|| ||q|| <= 1.
    # The verifier records this exact argument; no finite sample/grid is used.
    if any(dot(normal, normal) != 1 for normal in normals):
        raise AssertionError("Cauchy-Schwarz containment requires unit normals")

    return {
        "normal_squared_norms": [dot(normal, normal) for normal in normals],
        "vertices": vertices,
        "vertex_norms_squared": norms_squared,
        "maximum_vertex_norm_squared": max_norm_squared,
        "vertex_count": len(vertices),
        "boundedness_positive_combination": positive_weights,
        "unit_disk_containment": "PROVED_BY_EXACT_UNIT_NORMAL_CAUCHY_SCHWARZ",
    }


def validate_full_ambient_claim(data: dict[str, Any]) -> None:
    ambient_vertices = int(data["ambient"]["real_dimension"]) + 1
    declared_baseline = int(data["ambient"]["full_dimensional_simplex_vertex_count"])
    if ambient_vertices != declared_baseline:
        raise ValueError("full-dimensional d+1 simplex baseline mismatch")
    if data["simplex"]["expected_full_ambient_vertex_baseline"] != ambient_vertices:
        raise ValueError("structured simplex has invalid ambient baseline")
    if data["simplex"]["expected_vertex_count"] >= ambient_vertices:
        raise ValueError("V0.1 no longer demonstrates a structured vertex reduction")
    # A 3-vertex polygon cannot enclose an 8D ball. The only supported claim is
    # containment of the 2D disk in the proved support plane.
    if data["claim_ceiling"][0].find("only in the declared two-dimensional") < 0:
        raise ValueError("claim ceiling does not restrict the cover to support")


def run_negative_controls(data: dict[str, Any], basis: Matrix) -> None:
    contract = data["structural_contract"]
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    expected_controls = {
        "allowed-offdiagonal-real-generator": "reject-support-proof",
        "allowed-imaginary-generator": "reject-support-proof",
        "linearly-dependent-support-basis": "reject-rank",
        "basis-gram-not-identity": "reject-orthonormality",
        "nonunit-simplex-normal": "reject-containment-proof",
        "missing-simplex-vertex": "reject-containment",
        "reversed-halfspace": "reject-containment",
        "covariance-rank-used-as-hard-support-proof": "reject-evidence-class",
        "split-calibration-coordinate": "reject-dependency",
        "three-vertices-claimed-for-full-eight-dimensional-ball": "reject-claim-class",
    }
    if controls != expected_controls:
        raise AssertionError("negative-control manifest changed")

    # Explicit off-diagonal and imaginary generators must break the support
    # annihilator proof, even when their amplitudes are small.
    off_diagonal = [Fraction(0) for _ in range(8)]
    off_diagonal[1] = Fraction(1, 10)
    imaginary = [Fraction(0) for _ in range(8)]
    imaginary[4] = Fraction(1, 10)
    constraints = contract["hard_constraints"]
    for perturbation in (off_diagonal, imaginary):
        violations = [
            item for item in constraints
            if perturbation[int(item["coordinate_index"])] != q(item["value"])
        ]
        if not violations:
            raise AssertionError("out-of-support perturbation did not violate the hard constraints")

    # Try to enlarge the admitted basis with a real off-diagonal generator.
    off_diagonal_basis = [row[:] + [Fraction(0)] for row in basis]
    off_diagonal_basis[1][2] = Fraction(1)
    try:
        validate_structural_contract(data, off_diagonal_basis)
    except ValueError:
        pass
    else:
        raise AssertionError("an allowed off-diagonal basis generator was accepted")

    # Try to enlarge the admitted basis with an imaginary generator.
    imaginary_basis = [row[:] + [Fraction(0)] for row in basis]
    imaginary_basis[4][2] = Fraction(1)
    try:
        validate_structural_contract(data, imaginary_basis)
    except ValueError:
        pass
    else:
        raise AssertionError("an allowed imaginary basis generator was accepted")

    dependent_basis = [vector[:] for vector in basis]
    for row in dependent_basis:
        row[1] = row[0]
    try:
        validate_structural_contract(data, dependent_basis)
    except ValueError:
        pass
    else:
        raise AssertionError("dependent support basis was accepted")

    bad_gram_basis = [vector[:] for vector in basis]
    bad_gram_basis[0][0] = Fraction(2)
    try:
        validate_structural_contract(data, bad_gram_basis)
    except ValueError:
        pass
    else:
        raise AssertionError("non-orthonormal support basis was accepted")

    bad_normal_simplex = json.loads(json.dumps(data["simplex"]))
    bad_normal_simplex["normals"][1][0] = {"numerator": "-1", "denominator": "2"}
    try:
        validate_simplex(bad_normal_simplex, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("nonunit simplex normal was accepted")

    missing_vertex_simplex = json.loads(json.dumps(data["simplex"]))
    missing_vertex_simplex["vertices"] = missing_vertex_simplex["vertices"][:-1]
    try:
        validate_simplex(missing_vertex_simplex, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("missing simplex vertex was accepted")

    reversed_halfspace_simplex = json.loads(json.dumps(data["simplex"]))
    reversed_halfspace_simplex["inequalities"][1] = "n2 dot q >= 1"
    try:
        validate_simplex(reversed_halfspace_simplex, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("reversed half-space was accepted")

    covariance_only = json.loads(json.dumps(data))
    covariance_only["structural_contract"]["authority_class"] = (
        "empirical_covariance_rank"
    )
    try:
        validate_structural_contract(covariance_only, basis)
    except ValueError:
        pass
    else:
        raise AssertionError("covariance rank was accepted as a hard support proof")

    split_coordinates = json.loads(json.dumps(data))
    split_coordinates["structural_contract"]["coordinate_frequency_bindings"][1][
        "calibration_nuisance_id"
    ] = "other-frequency-local-coordinate"
    try:
        validate_structural_contract(
            split_coordinates,
            basis,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("split calibration coordinate was accepted")

    # A claim that the structured polygon covers the 8D ambient ball is rejected
    # by the explicit scope/vertex-count guard.
    wrong_claim = json.loads(json.dumps(data))
    wrong_claim["ambient"]["full_dimensional_simplex_vertex_count"] = 3
    try:
        validate_full_ambient_claim(wrong_claim)
    except ValueError:
        pass
    else:
        raise AssertionError("three vertices were accepted for the full ambient ball")


def identity(size: int) -> Matrix:
    return [
        [Fraction(1) if row == col else Fraction(0) for col in range(size)]
        for row in range(size)
    ]


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if data["schema"] != "luminous.airroad.structured_mimo_generator_support.v0_1":
        raise ValueError("unsupported structured support schema")
    if data["schema_version"] != "0.1.0":
        raise ValueError("unsupported schema version")
    if data["claim_class"] != "deterministic_hard_set_outer_approximation":
        raise ValueError("statistical covariance is not a hard support proof")

    basis = decode_matrix(data["support_basis"]["basis_vectors"])
    support = validate_structural_contract(data, basis)
    simplex = validate_simplex(data["simplex"], support["support_dimension"])
    validate_full_ambient_claim(data)
    run_negative_controls(data, basis)

    print("AIRROAD structured MIMO generator support reference: PASS")
    print(f"ambient real dimension: {support['ambient_dimension']}")
    print(f"proved hard support dimension: {support['support_dimension']}")
    print(f"basis Gram: {support['basis_gram']}")
    print(f"structured outer simplex vertices: {simplex['vertex_count']}")
    print(f"full-dimensional ambient baseline vertices: "
          f"{data['ambient']['full_dimensional_simplex_vertex_count']}")
    print(f"normal squared norms: {simplex['normal_squared_norms']}")
    print(f"vertices: {simplex['vertices']}")
    print(f"max vertex norm squared: {simplex['maximum_vertex_norm_squared']}")
    print(f"support disk containment: {simplex['unit_disk_containment']}")
    print("off-support, rank, Gram, normal, coverage, calibration and claim controls: PASS")
    print("physical structural qualification: NOT ESTABLISHED")


if __name__ == "__main__":
    main()
