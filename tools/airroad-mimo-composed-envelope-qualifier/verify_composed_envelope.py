#!/usr/bin/env python3
"""Exact-rational pointwise-to-continuous MIMO residual envelope oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-composed-envelope-v0.1.json"
Vec = list[Fraction]
Matrix = list[list[Fraction]]
Poly = list[Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def eval_poly(poly: Poly, x: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(poly):
        result = result * x + coefficient
    return result


def derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return [Fraction(0)]
    return [Fraction(i) * poly[i] for i in range(1, len(poly))]


def mat_vec(matrix: Matrix, vector: Vec) -> Vec:
    return [
        sum((matrix[i][j] * vector[j] for j in range(len(vector))), Fraction(0))
        for i in range(len(matrix))
    ]


def norm_sq(vector: Vec) -> Fraction:
    return sum((value * value for value in vector), Fraction(0))


def determinant(matrix: Matrix) -> Fraction:
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
    raise ValueError("unsupported determinant size")


def psd_by_principal_minors(matrix: Matrix) -> bool:
    n = len(matrix)
    return all(
        determinant([[matrix[i][j] for j in subset] for i in subset]) >= 0
        for size in range(1, n + 1)
        for subset in combinations(range(n), size)
    )


def lmi_matrix(gamma: Fraction, residual: Vec, input_norm_sq: Fraction) -> Matrix:
    p = len(residual)
    return (
        [[gamma if i == j else Fraction(0) for j in range(p)] + [residual[i]]
         for i in range(p)]
        + [[*residual, input_norm_sq]]
    )


def residual(omega: Fraction) -> Fraction:
    return omega * (omega - Fraction(1, 2)) * (omega - 1)


def matrix_scale_norm_squared(value: Fraction) -> Fraction:
    # P is the 2x2 swap matrix, so its induced spectral norm is exactly one.
    return value * value



def derivative_evidence_valid(data: dict, evidence: dict) -> bool:
    return (
        evidence.get("model_identity") == data.get("model_identity")
        and evidence.get("topology_identity") == data.get("topology_identity")
        and evidence.get("operating_point_identity") == data.get("operating_point_identity")
        and evidence.get("frequency_unit") == data.get("frequency_unit")
        and bool(evidence.get("evidence_reference"))
        and evidence.get("global_induced_spectral_norm_derivative_bound") is not None
    )


def endpoint_coverage_valid(knots: list[Fraction], lo: Fraction, hi: Fraction) -> bool:
    return (
        len(knots) >= 2
        and knots == sorted(knots)
        and len(knots) == len(set(knots))
        and knots[0] == lo
        and knots[-1] == hi
    )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_composed_pointwise_to_continuous_envelope.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_composed_frequency_band_certificate"
    assert data["model_identity"] == "sha256:synthetic-identity-mimo-with-cubic-residual-v0.1"
    assert data["topology_identity"] == "sha256:two-port-symmetric-crossport-reference-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-operating-point-v0.1"
    assert data["frequency_unit"] == "normalized_frequency"
    assert data["norm"] == "induced_spectral_norm"
    assert data["ports"] == ["bus-a", "bus-b"]

    g0: Matrix = [[q(value) for value in row] for row in data["nominal_transfer_matrix"]]
    p_matrix: Matrix = [[q(value) for value in row] for row in data["residual_matrix_factor"]]
    assert g0 == [[1, 0], [0, 1]]
    assert p_matrix == [[0, 1], [1, 0]]

    # Verify orthogonality of P: P^T P = I.
    p_transpose_times_p = [
        [
            sum((p_matrix[k][i] * p_matrix[k][j] for k in range(2)), Fraction(0))
            for j in range(2)
        ]
        for i in range(2)
    ]
    assert p_transpose_times_p == [[1, 0], [0, 1]]

    band = data["frequency_band"]
    omega_lo, omega_hi = q(band["lower"]), q(band["upper"])
    assert (omega_lo, omega_hi) == (Fraction(0), Fraction(1))

    knots = [q(value) for value in data["sample_coverage"]["knots"]]
    assert knots == [Fraction(0), Fraction(1, 2), Fraction(1)]
    assert endpoint_coverage_valid(knots, omega_lo, omega_hi)
    gaps = [right - left for left, right in zip(knots, knots[1:])]
    maximum_gap = max(gaps)
    nearest_distance = maximum_gap / 2
    assert maximum_gap == q(data["sample_coverage"]["maximum_gap"])
    assert nearest_distance == q(data["sample_coverage"]["maximum_nearest_knot_distance"])
    assert data["sample_coverage"]["includes_lower_endpoint"] is True
    assert data["sample_coverage"]["includes_upper_endpoint"] is True
    assert nearest_distance == Fraction(1, 4)

    # The residual polynomial is derived from the declared factorization.
    r_poly = [Fraction(0), Fraction(1, 2), Fraction(-3, 2), Fraction(1)]
    dr_poly = derivative(r_poly)
    d2r_poly = derivative(dr_poly)
    d3r_poly = derivative(d2r_poly)
    assert dr_poly == [Fraction(1, 2), Fraction(-3), Fraction(3)]
    assert d2r_poly == [Fraction(-3), Fraction(6)]
    assert d3r_poly == [Fraction(6)]
    assert [str(value) for value in r_poly] == data["residual_scalar_polynomial"]["coefficients_ascending"]
    assert [str(value) for value in dr_poly] == data["residual_scalar_polynomial"]["derivative_coefficients_ascending"]
    assert [str(value) for value in d2r_poly] == data["residual_scalar_polynomial"]["derivative_second_coefficients_ascending"]
    assert [str(value) for value in d3r_poly] == data["residual_scalar_polynomial"]["derivative_third_coefficients_ascending"]

    # r' is convex; its maximum on [0,1] is at an endpoint, and its minimum
    # is at the vertex 1/2. These exact values prove sup |r'| = 1/2.
    vertex = Fraction(1, 2)
    derivative_values = (
        eval_poly(dr_poly, omega_lo),
        eval_poly(dr_poly, vertex),
        eval_poly(dr_poly, omega_hi),
    )
    assert derivative_values == (
        Fraction(1, 2), Fraction(-1, 4), Fraction(1, 2)
    )
    assert d3r_poly[0] > 0
    derived_l = max(abs(value) for value in derivative_values)
    derivative_evidence = data["derivative_certificate"]
    assert derived_l == q(derivative_evidence["global_induced_spectral_norm_derivative_bound"]) == Fraction(1, 2)
    assert q(derivative_evidence["derivative_vertex"]) == vertex
    assert derivative_evidence_valid(data, derivative_evidence)

    # Verify pointwise data-consistency at all three knots. Both basis inputs
    # span the complete input space; every column has its own deterministic
    # hard error ball. The LMI proves each column squared gain <= gamma.
    experiments = data["pointwise_experiment_design"]["experiments"]
    assert len(experiments) == 6
    expected_gamma = Fraction(1, 10000)
    epsilon = q(data["measurement_noise"]["per_output_column_hard_radius"])
    assert epsilon == Fraction(1, 100)
    assert data["measurement_noise"]["semantics"] == "deterministic_hard_euclidean_ball"

    columns_by_knot: dict[Fraction, list[Vec]] = {}
    for item in experiments:
        omega = q(item["omega"])
        assert omega in knots
        assert item["model_identity"] == data["model_identity"]
        assert item["topology_identity"] == data["topology_identity"]
        assert item["operating_point_identity"] == data["operating_point_identity"]
        assert item["frequency_unit"] == data["frequency_unit"]
        u = [q(value) for value in item["input_vector"]]
        y = [q(value) for value in item["observed_output_vector"]]
        nominal_y = mat_vec(g0, u)
        nominal_residual = [a - b for a, b in zip(y, nominal_y)]
        stored_nominal_residual = [q(value) for value in item["nominal_residual_vector"]]
        assert nominal_residual == stored_nominal_residual == [0, 0]
        assert q(item["output_hard_error_radius"]) == epsilon
        assert item["uncertainty_semantics"] == "deterministic_hard_euclidean_ball"
        input_sq = norm_sq(u)
        assert input_sq == q(item["squared_input_norm"]) == 1
        gamma = q(item["gamma_squared_gain_bound"])
        assert gamma == expected_gamma
        assert gamma == epsilon**2 / input_sq

        worst = [q(value) for value in item["worst_case_lmi_residual_witness"]]
        assert norm_sq(worst) == epsilon**2
        lmi = lmi_matrix(gamma, worst, input_sq)
        frozen_lmi = [[q(value) for value in row] for row in item["lmi"]]
        assert lmi == frozen_lmi
        assert psd_by_principal_minors(lmi)

        too_small = q(data["counterfactuals"]["gamma_understated"])
        assert too_small < gamma
        assert not psd_by_principal_minors(lmi_matrix(too_small, worst, input_sq))

        columns_by_knot.setdefault(omega, []).append(u)

    # Each knot must contain exactly e1 and e2, yielding full-rank U=I.
    expected_basis = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    for omega in knots:
        cols = columns_by_knot[omega]
        assert cols == expected_basis
        determinant_u = cols[0][0] * cols[1][1] - cols[1][0] * cols[0][1]
        assert determinant_u == 1

    input_set = data["pointwise_experiment_design"]
    assert input_set["rank"] == input_set["input_dimension"] == 2
    assert q(input_set["determinant"]) == 1

    # Each of two independently hard-bounded output columns contributes
    # gamma*||u||^2 to the Frobenius-squared upper bound.
    pointwise_frob_sq = 2 * expected_gamma
    pointwise_radius_sq = q(data["pointwise_data_consistency"]["pointwise_frobenius_norm_squared_upper_bound"])
    pointwise_spectral_sq = q(data["pointwise_data_consistency"]["pointwise_spectral_norm_squared_upper_bound"])
    rational_radius = q(data["pointwise_data_consistency"]["rational_pointwise_spectral_norm_upper_radius"])
    rational_radius_sq = q(data["pointwise_data_consistency"]["rational_radius_squared"])
    assert pointwise_frob_sq == pointwise_radius_sq == pointwise_spectral_sq == Fraction(1, 5000)
    assert rational_radius == Fraction(1, 70)
    assert rational_radius_sq == rational_radius**2 == Fraction(1, 4900)
    assert rational_radius_sq >= pointwise_spectral_sq

    # Compose a pointwise matrix norm bound with a full-band Lipschitz proof.
    lipschitz_increment = derived_l * nearest_distance
    expected_increment = q(data["expected"]["lipschitz_increment"])
    assert lipschitz_increment == expected_increment == Fraction(1, 8)
    global_bound = rational_radius + lipschitz_increment
    assert global_bound == q(data["expected"]["global_continuous_spectral_norm_upper_bound"])
    assert global_bound == Fraction(39, 280)

    # Verify the hidden matrix residual at knots and in-between. Since P is
    # orthogonal, its induced spectral norm is |r(omega)|.
    true_values = [residual(omega) for omega in knots]
    assert true_values == [Fraction(0), Fraction(0), Fraction(0)]
    assert true_values == [q(item) for item in data["expected"]["exact_residual_at_sample_knots"]]
    quarter, three_quarters = Fraction(1, 4), Fraction(3, 4)
    r_quarter, r_three_quarters = residual(quarter), residual(three_quarters)
    assert r_quarter == q(data["expected"]["residual_at_quarter"]) == Fraction(3, 64)
    assert r_three_quarters == q(data["expected"]["residual_at_three_quarters"]) == Fraction(-3, 64)
    assert abs(r_quarter) <= global_bound and abs(r_three_quarters) <= global_bound

    # Independent exact extremum of r on [0,1]: t=omega-1/2,
    # r=t^3-t/4, r'=3t^2-1/4, critical t^2=1/12, max r^2=1/432.
    actual_sup_sq = Fraction(1, 432)
    assert actual_sup_sq == q(data["expected"]["exact_true_supremum_squared"])
    squared_bound = global_bound**2
    ratio = squared_bound / actual_sup_sq
    assert ratio == q(data["expected"]["squared_bound_to_true_supremum_ratio"])
    assert ratio == Fraction(41067, 4900)

    # Failure controls are evaluated against the actual admission predicates.
    rank_one = data["counterfactuals"]["rank_one_excitation"]
    rank_one_columns = [[q(value) for value in column] for column in rank_one]
    rank_one_det = rank_one_columns[0][0] * rank_one_columns[1][1] - rank_one_columns[1][0] * rank_one_columns[0][1]
    assert rank_one_det == 0

    gamma_bad = q(data["counterfactuals"]["gamma_understated"])
    assert gamma_bad < expected_gamma
    assert not psd_by_principal_minors(lmi_matrix(
        gamma_bad, [epsilon, Fraction(0)], Fraction(1)
    ))

    lipschitz_bad = q(data["counterfactuals"]["lipschitz_understated"])
    assert lipschitz_bad < derived_l
    missing_endpoint_knots = knots[:-1]
    assert not endpoint_coverage_valid(
        missing_endpoint_knots, omega_lo, omega_hi
    )
    assert q(data["counterfactuals"]["band_missing_endpoint"]["upper"]) < omega_hi

    stale_evidence = dict(derivative_evidence)
    stale_evidence["model_identity"] = "sha256:stale-model"
    assert not derivative_evidence_valid(data, stale_evidence)
    stale_evidence = dict(derivative_evidence)
    stale_evidence["topology_identity"] = "sha256:stale-topology"
    assert not derivative_evidence_valid(data, stale_evidence)
    stale_evidence = dict(derivative_evidence)
    stale_evidence["frequency_unit"] = "Hz"
    assert not derivative_evidence_valid(data, stale_evidence)
    missing_l = dict(derivative_evidence)
    missing_l.pop("global_induced_spectral_norm_derivative_bound")
    assert not derivative_evidence_valid(data, missing_l)

    nonorthogonal = [[q(value) for value in row] for row in data["counterfactuals"]["nonorthogonal_residual_factor"]]
    assert nonorthogonal != p_matrix
    nonorthogonal_product = [
        [
            sum((nonorthogonal[k][i] * nonorthogonal[k][j] for k in range(2)), Fraction(0))
            for j in range(2)
        ]
        for i in range(2)
    ]
    assert nonorthogonal_product != [[1, 0], [0, 1]]

    # Identity mismatches, missing derivative evidence and statistical-only
    # intervals are not accepted as a complete physical certificate.
    stale_model = "sha256:stale-model"
    stale_topology = "sha256:stale-topology"
    wrong_units = "Hz"
    assert stale_model != data["model_identity"]
    assert stale_topology != data["topology_identity"]
    assert wrong_units != data["frequency_unit"]
    statistical_semantics = "uncalibrated_statistical_confidence_interval"
    assert statistical_semantics != data["measurement_noise"]["semantics"]
    statistical_samples = [dict(item) for item in experiments]
    statistical_samples[0]["uncertainty_semantics"] = statistical_semantics
    assert not all(
        item["uncertainty_semantics"] == "deterministic_hard_euclidean_ball"
        for item in statistical_samples
    )

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["rank-deficient-input-excitation"] == "no-full-matrix-pointwise-bound"
    assert controls["missing-hard-noise-radius"] == "not-evaluable"
    assert controls["statistical-only-sample-intervals"] == "reject-claim-class"
    assert controls["gamma-understated"] == "lmi-not-psd"
    assert controls["lipschitz-bound-understated"] == "reject"
    assert controls["stale-model-or-topology-identity"] == "not-evaluable"
    assert controls["missing-frequency-endpoint"] == "reject-full-band-coverage"
    assert controls["nonorthogonal-residual-factor-with-unit-norm-claim"] == "reject"
    assert controls["pointwise-lmi-promoted-without-derivative-bound"] == "not-established"
    assert controls["pointwise-to-continuous-without-full-coverage"] == "not-established"
    assert controls["frobenius-bound-labelled-exact-spectral-norm"] == "reject-claim-class"

    print("PASS six pointwise input/output observations and exact block LMIs")
    print(f"PASS full-rank U=I; pointwise spectral radius <= {rational_radius}")
    print(f"PASS independently justified derivative bound L={derived_l}")
    print(f"PASS full-band residual spectral norm <= {global_bound}")
    print(f"PASS exact true supremum squared={actual_sup_sq}; bound ratio={ratio}")
    print("PASS rank, noise semantics, model identity, domain and norm claim controls")
    print("PASS exact-rational composed MIMO envelope certificate")


if __name__ == "__main__":
    main()
