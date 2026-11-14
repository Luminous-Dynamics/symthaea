#!/usr/bin/env python3
"""Exact-rational 2x2 MIMO data-consistency and LMI oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-data-consistency-v0.1.json"
Vec = list[Fraction]
Matrix = list[list[Fraction]]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def det(matrix: Matrix) -> Fraction:
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if n == 3:
        return (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
    raise ValueError("fixture verifier supports principal minors up to 3x3")


def principal_minors_nonnegative(matrix: Matrix) -> bool:
    n = len(matrix)
    return all(
        det([[matrix[i][j] for j in subset] for i in subset]) >= 0
        for size in range(1, n + 1)
        for subset in combinations(range(n), size)
    )


def mat_vec(matrix: Matrix, vector: Vec) -> Vec:
    return [sum((matrix[i][j] * vector[j] for j in range(len(vector))), Fraction(0))
            for i in range(len(matrix))]


def norm_squared(vector: Vec) -> Fraction:
    return sum((value * value for value in vector), Fraction(0))


def lmi_matrix(gamma: Fraction, residual: Vec, input_squared_norm: Fraction) -> Matrix:
    p = len(residual)
    return (
        [[gamma if i == j else Fraction(0) for j in range(p)] + [residual[i]]
         for i in range(p)]
        + [[*residual, input_squared_norm]]
    )


def full_rank_2x2(columns: list[Vec]) -> bool:
    if len(columns) != 2 or any(len(column) != 2 for column in columns):
        return False
    return columns[0][0] * columns[1][1] - columns[1][0] * columns[0][1] != 0


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_data_consistent_additive_uncertainty.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_single_frequency_data_consistency"
    assert data["model_identity"] == "sha256:identity-nominal-two-port-v0.1"
    assert data["topology_identity"] == "sha256:two-port-bus-a-b-reference-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-operating-point-v0.1"
    assert q(data["frequency"]) == 3
    assert data["frequency_unit"] == "rad_per_second"
    assert data["port_order"] == ["bus-a", "bus-b"]
    assert data["uncertainty_semantics"] == "deterministic_hard_euclidean_ball"

    g0: Matrix = [[q(item) for item in row] for row in data["nominal_transfer_matrix"]]
    assert g0 == [[1, 0], [0, 1]]

    experiment_results = []
    input_columns: list[Vec] = []
    squared_column_bounds = []
    for item in data["experiments"]:
        u = [q(value) for value in item["input_vector"]]
        y = [q(value) for value in item["observed_output_vector"]]
        epsilon = q(item["output_error_radius"])
        assert epsilon == Fraction(1, 100)

        predicted = mat_vec(g0, u)
        residual = [y_i - pred_i for y_i, pred_i in zip(y, predicted)]
        worst = [q(value) for value in item["worst_case_residual_vector"]]
        input_sq = norm_squared(u)
        residual_sq = norm_squared(residual)
        worst_sq = norm_squared(worst)
        gamma = q(item["gamma"])

        assert residual_sq == Fraction(1, 100)
        assert worst_sq == (Fraction(1, 10) + epsilon) ** 2
        assert worst_sq == Fraction(121, 10000)
        assert input_sq == q(item["squared_input_norm"]) == 1
        assert gamma == worst_sq / input_sq
        assert residual_sq <= worst_sq

        # The worst-case residual is chosen in the measured residual direction.
        assert residual == [Fraction(0), Fraction(1, 10)] if item["experiment_id"] == "excite-bus-a" else True
        if item["experiment_id"] == "excite-bus-b":
            assert residual == [Fraction(1, 10), Fraction(0)]
        lmi = lmi_matrix(gamma, worst, input_sq)
        frozen_lmi = [[q(value) for value in row] for row in item["lmi"]]
        assert lmi == frozen_lmi
        assert principal_minors_nonnegative(lmi)

        # Reducing gamma below the exact required value makes the relevant
        # 2x2 principal minor negative, so the LMI is no longer PSD.
        too_small = q(data["counterfactuals"]["gamma_too_small"])
        small_lmi = lmi_matrix(too_small, worst, input_sq)
        assert not principal_minors_nonnegative(small_lmi)

        input_columns.append(u)
        squared_column_bounds.append(gamma * input_sq)
        experiment_results.append((residual, worst, gamma, lmi))

    assert len(experiment_results) == 2
    assert full_rank_2x2(input_columns)
    input_matrix = data["input_ensemble"]["matrix_columns"]
    columns = [[q(value) for value in column] for column in input_matrix]
    determinant = columns[0][0] * columns[1][1] - columns[1][0] * columns[0][1]
    assert determinant == q(data["input_ensemble"]["determinant"]) == 1
    assert data["input_ensemble"]["rank"] == data["input_ensemble"]["input_dimension"] == 2

    frobenius_sq = sum(squared_column_bounds, Fraction(0))
    expected_frob = q(data["expected"]["frobenius_norm_squared_upper_bound"])
    expected_spectral = q(data["expected"]["spectral_norm_squared_upper_bound"])
    assert frobenius_sq == expected_frob == expected_spectral == Fraction(121, 5000)

    nominal_residual_matrix = [
        [q(value) for value in row]
        for row in data["expected"]["nominal_residual_matrix"]
    ]
    assert nominal_residual_matrix == [[0, Fraction(1, 10)], [Fraction(1, 10), 0]]
    nominal_spectral_sq = q(data["expected"]["nominal_residual_spectral_norm_squared"])
    assert nominal_spectral_sq == Fraction(1, 100)
    assert nominal_spectral_sq < expected_spectral

    # Negative controls: input zero/nonzero residual is not an admissible gain
    # sample; rank one cannot support a full 2x2 operator claim.
    zero_case = data["counterfactuals"]["zero_input_with_nonzero_residual"]
    zero_input = [q(value) for value in zero_case["input_vector"]]
    zero_residual = [q(value) for value in zero_case["residual_vector"]]
    assert norm_squared(zero_input) == 0 and norm_squared(zero_residual) > 0
    assert not full_rank_2x2([
        [q(value) for value in column]
        for column in data["counterfactuals"]["rank_one_input_matrix"]
    ])

    assert data["counterfactuals"]["statistical_semantics"] != "deterministic_hard_euclidean_ball"
    assert data["expected"]["result_scope"] == "single_frequency_knot_only"

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["gamma-below-required-bound"] == "lmi-not-psd"
    assert controls["zero-input-nonzero-residual"] == "reject"
    assert controls["rank-deficient-inputs"] == "no-full-matrix-bound"
    assert controls["noise-bound-omitted"] == "not-evaluable"
    assert controls["statistical-only-noise"] == "reject-claim-class"
    assert controls["wrong-model-topology-or-operating-point"] == "not-evaluable"
    assert controls["pointwise-result-promoted-to-continuous"] == "reject"
    assert controls["frobenius-bound-labelled-exact-spectral-bound"] == "reject-claim-class"

    print("PASS exact nominal residuals and hard-noise expansion")
    print("PASS exact 3x3 LMI positive-semidefinite checks")
    print(f"PASS full-rank input ensemble and Frobenius/spectral squared bound {frobenius_sq}")
    print("PASS gamma, zero-input, rank, uncertainty-class and scope negative controls")
    print("PASS exact-rational MIMO data-consistency reference")


if __name__ == "__main__":
    main()
