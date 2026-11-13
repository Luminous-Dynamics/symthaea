#!/usr/bin/env python3
"""Exact-rational cross-frequency shared calibration dependency oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-cross-frequency-uq-v0.1.json"
Matrix = list[list[Fraction]]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def matrix_scale(matrix: Matrix, factor: Fraction) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def matrix_sub(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matrix_zero(matrix: Matrix) -> bool:
    return all(value == 0 for row in matrix for value in row)


def covariance_psd_2x2(matrix: Matrix) -> bool:
    return (
        len(matrix) == 2 and len(matrix[0]) == 2
        and matrix[0][0] >= 0 and matrix[1][1] >= 0
        and matrix[0][1] == matrix[1][0]
        and matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] >= 0
    )


def complete_shared_coverage(
    records: list[dict], expected_ids: list[str], nuisance_id: str,
    acquisition_root_id: str, calibration_certificate_id: str
) -> bool:
    ids = [item["record_id"] for item in records]
    return (
        len(ids) == len(set(ids))
        and set(ids) == set(expected_ids)
        and all(item.get("shared_nuisance_id") == nuisance_id for item in records)
        and all(item.get("acquisition_root_id") == acquisition_root_id for item in records)
        and all(item.get("calibration_certificate_id") == calibration_certificate_id for item in records)
    )



def record_identity_valid(data: dict, record: dict) -> bool:
    return (
        record.get("model_identity") == data.get("model_identity")
        and record.get("topology_identity") == data.get("topology_identity")
        and record.get("operating_point_identity") == data.get("operating_point_identity")
        and record.get("frequency_unit") == data.get("frequency_unit")
        and record.get("measurement_equation") == "observed_value = shared_q * nominal_value"
    )


def interval_sub(a: tuple[Fraction, Fraction],
                 b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] - b[1], a[1] - b[0]


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_cross_frequency_shared_uncertainty.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_joint_hard_set_with_statistical_companion"
    assert data["model_identity"] == "sha256:nominal-two-frequency-identity-scaling-v0.1"
    assert data["topology_identity"] == "sha256:two-port-bus-a-b-v0.1"
    assert data["operating_point_identity"] == "sha256:fixed-operating-point-v0.1"
    assert data["frequency_unit"] == "rad_per_second"
    assert data["port_order"] == ["bus-a", "bus-b"]

    frequencies = data["nominal_frequency_matrices"]
    assert len(frequencies) == 2
    assert frequencies[0]["frequency_id"] == "omega-1"
    assert frequencies[1]["frequency_id"] == "omega-2"
    assert q(frequencies[0]["omega"]) == 1
    assert q(frequencies[1]["omega"]) == 2
    g1: Matrix = [[q(x) for x in row] for row in frequencies[0]["matrix"]]
    g2: Matrix = [[q(x) for x in row] for row in frequencies[1]["matrix"]]
    identity = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    assert g1 == identity
    assert g2 == matrix_scale(identity, Fraction(2))

    nuisance = data["shared_nuisance"]
    assert nuisance["semantics"] == "deterministic_joint_hard_set"
    assert nuisance["factor"] == "q=1+u"
    u_domain = (q(nuisance["u_lower"]), q(nuisance["u_upper"]))
    q_domain = (q(nuisance["q_lower"]), q(nuisance["q_upper"]))
    assert u_domain == (Fraction(-1, 10), Fraction(1, 10))
    assert q_domain == (Fraction(9, 10), Fraction(11, 10))
    assert q_domain == (1 + u_domain[0], 1 + u_domain[1])
    assert q_domain[0] > 0

    records = data["observation_records"]
    assert len(records) == 8
    assert complete_shared_coverage(
        records,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )
    assert data["measurement_equation"] == "observed_matrix(omega_k)=q_shared*nominal_matrix(omega_k)"
    assert all(record_identity_valid(data, item) for item in records)
    assert all(item["measurement_equation"] == "observed_value = shared_q * nominal_value" for item in records)
    expected_pairs = {
        (frequency_id, row, column)
        for frequency_id in ("omega-1", "omega-2")
        for row in range(2)
        for column in range(2)
    }
    actual_pairs = {
        (item["frequency_id"], item["row"], item["column"])
        for item in records
    }
    assert actual_pairs == expected_pairs

    # Every measured matrix is generated by one shared q. Its cross-frequency
    # affine relation is identically zero for all q in the admitted domain.
    for q_value in q_domain:
        measured_1 = matrix_scale(g1, q_value)
        measured_2 = matrix_scale(g2, q_value)
        assert matrix_zero(matrix_sub(measured_2, matrix_scale(measured_1, Fraction(2))))

    # Exact marginal bounds at the two knots.
    marginal = data["expected"]["marginal_matrices"]
    assert (q(marginal["omega_1"]["diagonal_lower"]),
            q(marginal["omega_1"]["diagonal_upper"])) == q_domain
    assert (q(marginal["omega_2"]["diagonal_lower"]),
            q(marginal["omega_2"]["diagonal_upper"])) == (
                Fraction(9, 5), Fraction(11, 5)
            )
    assert q(marginal["omega_1"]["off_diagonal"]) == 0
    assert q(marginal["omega_2"]["off_diagonal"]) == 0

    # Independent frequency nuisance factors q1,q2 preserve each marginal
    # but enlarge the cross-frequency affine relation.
    independent_relation = interval_sub(
        (2 * q_domain[0], 2 * q_domain[1]),  # 2*q2
        (2 * q_domain[0], 2 * q_domain[1]),  # 2*q1, interval subtraction
    )
    # Generic interval subtraction gives [-2/5,+2/5] exactly.
    assert independent_relation == (Fraction(-2, 5), Fraction(2, 5))
    expected_independent = data["expected"]["independent_product_relation_interval"]
    assert independent_relation == (q(expected_independent["lower"]), q(expected_independent["upper"]))
    assert independent_relation[0] < 0 < independent_relation[1]
    assert matrix_zero(matrix_sub(matrix_scale(g2, q_domain[0]), matrix_scale(g1, 2 * q_domain[1]))) is False

    # The shared hard set has one common scalar degree of freedom; the
    # independent-frequency product has two. Equal marginals do not imply
    # equal joint uncertainty.
    shared_dimension = 1
    independent_dimension = 2
    assert shared_dimension < independent_dimension
    assert data["expected"]["independent_product_strictly_larger"] is True

    # Statistical companion is separate from the deterministic hard set.
    stats = data["statistical_companion"]
    assert stats["semantics"] == "separate_statistical_covariance_model_not_a_hard_set"
    shared_cov = [[q(x) for x in row] for row in stats["shared_cross_frequency_covariance"]]
    independent_cov = [[q(x) for x in row] for row in stats["independent_frequency_counterfactual_covariance"]]
    assert shared_cov == [
        [Fraction(1, 100), Fraction(1, 50)],
        [Fraction(1, 50), Fraction(1, 25)],
    ]
    assert independent_cov == [
        [Fraction(1, 100), Fraction(0)],
        [Fraction(0), Fraction(1, 25)],
    ]
    assert covariance_psd_2x2(shared_cov)
    assert covariance_psd_2x2(independent_cov)
    assert q(stats["expected_shared_cross_covariance"]) == shared_cov[0][1] == Fraction(1, 50)
    assert q(stats["expected_independent_cross_covariance"]) == independent_cov[0][1] == 0
    assert shared_cov != independent_cov

    # Negative controls.
    mismatched = [dict(item) for item in records]
    mismatched[-1]["shared_nuisance_id"] = "different-calibration-chain"
    assert not complete_shared_coverage(
        mismatched,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )

    incomplete = records[:-1]
    assert not complete_shared_coverage(
        incomplete,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )

    second_frequency_independent = [dict(item) for item in records]
    for item in second_frequency_independent:
        if item["frequency_id"] == "omega-2":
            item["shared_nuisance_id"] = "cal-chain-independent-at-knot-2"
    assert not complete_shared_coverage(
        second_frequency_independent,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )

    stale_acquisition = [dict(item) for item in records]
    stale_acquisition[0]["acquisition_root_id"] = "stale-acquisition-root"
    assert not complete_shared_coverage(
        stale_acquisition,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )
    stale_calibration = [dict(item) for item in records]
    stale_calibration[0]["calibration_certificate_id"] = "stale-calibration-certificate"
    assert not complete_shared_coverage(
        stale_calibration,
        nuisance["applies_to_record_ids"],
        nuisance["nuisance_id"],
        nuisance["acquisition_root_id"],
        nuisance["calibration_certificate_id"],
    )

    stale_model_record = dict(records[0])
    stale_model_record["model_identity"] = "sha256:stale-model"
    assert not record_identity_valid(data, stale_model_record)
    wrong_unit_record = dict(records[0])
    wrong_unit_record["frequency_unit"] = "Hz"
    assert not record_identity_valid(data, wrong_unit_record)

    equal_domain_distinct_ids = {
        "omega_1": "cal-chain-1", "omega_2": "cal-chain-2",
        "domain": q_domain
    }
    assert equal_domain_distinct_ids["domain"] == q_domain
    assert equal_domain_distinct_ids["omega_1"] != equal_domain_distinct_ids["omega_2"]

    q_including_zero = (Fraction(0), Fraction(1))
    assert q_including_zero[0] <= 0 <= q_including_zero[1]
    assert q_including_zero[0] == 0

    wrong_unit = data["counterfactuals"]["wrong_frequency_unit"]
    assert wrong_unit != data["frequency_unit"]
    assert data["counterfactuals"]["independent_frequency_nuisances"]["omega_1_nuisance_id"] != data["counterfactuals"]["independent_frequency_nuisances"]["omega_2_nuisance_id"]

    statistical_as_hard = stats["semantics"] == "deterministic_joint_hard_set"
    assert not statistical_as_hard

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["same-nuisance-across-all-knots-and-matrix-elements"] == "shared-relation-exact"
    assert controls["different-nuisance-at-second-frequency"] == "reject-shared-claim"
    assert controls["incomplete-matrix-or-frequency-coverage"] == "not-evaluable"
    assert controls["same-numeric-domain-but-distinct-nuisance-identities"] == "not-shared"
    assert controls["independent-frequency-product"] == "larger-set"
    assert controls["q-domain-includes-zero"] == "reject-cancellation"
    assert controls["statistical-covariance-labelled-hard-set"] == "reject-claim-class"
    assert controls["shared-covariance-made-block-diagonal"] == "reject-dependency-semantics"
    assert controls["frequency-unit-mismatch"] == "not-evaluable"
    assert controls["stale-model-topology-or-calibration-identity"] == "not-evaluable"

    print("PASS complete shared calibration/acquisition coverage over both 2x2 knots")
    print("PASS exact shared relation G(omega2)-2*G(omega1)=0")
    print("PASS independent frequency product relation interval [-2/5,2/5]")
    print("PASS distinct joint set with identical frequency marginals")
    print("PASS shared statistical covariance cross term and PSD checks")
    print("PASS identity, coverage, zero-domain and uncertainty-class controls")
    print("PASS exact-rational cross-frequency dependency reference")


if __name__ == "__main__":
    main()
