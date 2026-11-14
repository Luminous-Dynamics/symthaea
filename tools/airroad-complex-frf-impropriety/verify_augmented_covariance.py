#!/usr/bin/env python3
"""Exact-rational reference for complex covariance and complementary covariance."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-complex-frf-impropriety-v0.1.json"

ComplexQ = tuple[Fraction, Fraction]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def read_complex(value: dict[str, Any]) -> ComplexQ:
    return q(value["re"]), q(value["im"])


def add(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] + right[0], left[1] + right[1]


def sub(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] - right[0], left[1] - right[1]


def mul(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def conj(value: ComplexQ) -> ComplexQ:
    return value[0], -value[1]


def scale(value: Fraction, z: ComplexQ) -> ComplexQ:
    return value * z[0], value * z[1]


def is_zero(z: ComplexQ) -> bool:
    return z == (Fraction(0), Fraction(0))


def real_covariance(values: list[list[dict[str, str]]]) -> list[list[Fraction]]:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise ValueError("V0.1 scalar complex reference requires a 2x2 real covariance")
    matrix = [[q(item) for item in row] for row in values]
    if matrix[0][1] != matrix[1][0]:
        raise ValueError("real covariance must be symmetric")
    return matrix


def derive_complex_statistics(
    sigma_xy: list[list[Fraction]],
) -> tuple[ComplexQ, ComplexQ]:
    var_x = sigma_xy[0][0]
    covariance_xy = sigma_xy[0][1]
    var_y = sigma_xy[1][1]
    covariance = (var_x + var_y, Fraction(0))
    complementary = (
        var_x - var_y,
        2 * covariance_xy,
    )
    return covariance, complementary


def real_covariance_from_complex(
    covariance: ComplexQ,
    complementary: ComplexQ,
) -> list[list[Fraction]]:
    if covariance[1] != 0:
        raise ValueError("scalar covariance C must be real")
    var_x = (covariance[0] + complementary[0]) / 2
    var_y = (covariance[0] - complementary[0]) / 2
    covariance_xy = complementary[1] / 2
    return [
        [var_x, covariance_xy],
        [covariance_xy, var_y],
    ]


def augmented_covariance(
    covariance: ComplexQ,
    complementary: ComplexQ,
) -> list[list[ComplexQ]]:
    return [
        [covariance, complementary],
        [conj(complementary), conj(covariance)],
    ]


def determinant_real_2x2(matrix: list[list[Fraction]]) -> Fraction:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("expected a 2x2 real matrix")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def determinant_complex_2x2(matrix: list[list[ComplexQ]]) -> ComplexQ:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("expected a 2x2 complex matrix")
    return sub(mul(matrix[0][0], matrix[1][1]),
               mul(matrix[0][1], matrix[1][0]))


def assert_real_covariance_psd(matrix: list[list[Fraction]]) -> None:
    assert matrix[0][0] >= 0
    assert matrix[1][1] >= 0
    assert matrix[0][1] == matrix[1][0]
    assert determinant_real_2x2(matrix) >= 0


def assert_augmented_hermitian_psd(matrix: list[list[ComplexQ]]) -> Fraction:
    assert matrix[0][0][1] == 0
    assert matrix[1][1][1] == 0
    assert matrix[0][0][0] >= 0
    assert matrix[1][1][0] >= 0
    assert matrix[1][0] == conj(matrix[0][1])
    determinant = determinant_complex_2x2(matrix)
    assert determinant[1] == 0
    assert determinant[0] >= 0
    return determinant[0]


def assert_proper_claim(
    complementary: ComplexQ,
    evidence: str,
) -> None:
    if evidence != "explicit_test_pass" or not is_zero(complementary):
        raise ValueError("properness requires explicit evidence and zero P")


def validate_claim_class(source_semantics: str, requested_claim: str) -> None:
    if (
        source_semantics == "statistical_covariance_and_complementary_covariance"
        and requested_claim == "deterministic_hard_set"
    ):
        raise ValueError("statistical covariance cannot be promoted to a hard set")


def validate_identity_bindings(data: dict[str, Any], bindings: dict[str, str]) -> None:
    for key in (
        "campaign_id",
        "model_identity",
        "acquisition_id",
        "sample_clock_identity",
    ):
        if bindings.get(key) != data.get(key):
            raise ValueError(f"stale complex covariance identity: {key}")


def verify_case(data: dict[str, Any], name: str) -> dict[str, Any]:
    case = data[name]
    validate_identity_bindings(data, case["identity_bindings"])
    sigma = real_covariance(case["real_covariance"])
    assert_real_covariance_psd(sigma)
    covariance, complementary = derive_complex_statistics(sigma)

    expected = case["expected"]
    assert covariance == read_complex(expected["covariance"])
    assert complementary == read_complex(expected["complementary_covariance"])

    reconstructed = real_covariance_from_complex(covariance, complementary)
    assert reconstructed == sigma

    augmented = augmented_covariance(covariance, complementary)
    expected_augmented = [
        [read_complex(item) for item in row]
        for row in expected["augmented_covariance"]
    ]
    assert augmented == expected_augmented
    augmented_det = assert_augmented_hermitian_psd(augmented)
    assert determinant_real_2x2(sigma) == q(expected["real_covariance_determinant"])
    assert augmented_det == q(expected["augmented_covariance_determinant"])
    assert augmented_det == 4 * determinant_real_2x2(sigma)

    proper = is_zero(complementary)
    assert proper is expected["is_proper"]
    evidence = case["properness_evidence"]
    if proper:
        assert_proper_claim(complementary, evidence)
    elif evidence == "explicit_test_pass":
        raise AssertionError("a nonzero pseudo-covariance was called proper")

    return {
        "covariance": covariance,
        "complementary_covariance": complementary,
        "real_covariance": sigma,
        "augmented_covariance": augmented,
        "real_determinant": determinant_real_2x2(sigma),
        "augmented_determinant": augmented_det,
        "is_proper": proper,
    }


def run_negative_controls(data: dict[str, Any], improper: dict[str, Any]) -> None:
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls == {
        "discard-nonzero-complementary-covariance": "reject",
        "claim-proper-without-evidence": "reject",
        "covariance-only-same-c-but-different-p": "not-equivalent",
        "invalid-augmented-covariance": "reject-not-psd",
        "statistical-augmented-covariance-as-hard-set": "reject-claim-class",
        "stale-campaign-model-acquisition-clock-identity": "not-evaluable",
    }

    # Ordinary covariance equality does not identify the joint complex law.
    try:
        assert_proper_claim(improper["complementary_covariance"], "explicit_test_pass")
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero P was silently discarded under a properness claim")

    try:
        assert_proper_claim((Fraction(0), Fraction(0)), "assumed_proper")
    except ValueError:
        pass
    else:
        raise AssertionError("properness was accepted without explicit test evidence")

    invalid_c = (Fraction(1), Fraction(0))
    invalid_p = (Fraction(2), Fraction(0))
    invalid_augmented = augmented_covariance(invalid_c, invalid_p)
    determinant = determinant_complex_2x2(invalid_augmented)
    assert determinant == (Fraction(-3), Fraction(0))
    try:
        assert_augmented_hermitian_psd(invalid_augmented)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-PSD augmented covariance was accepted")

    try:
        validate_claim_class(
            "statistical_covariance_and_complementary_covariance",
            "deterministic_hard_set",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("statistical covariance was promoted to a hard set")

    stale = dict(data["improper_case"]["identity_bindings"])
    stale["sample_clock_identity"] = "stale-clock"
    try:
        validate_identity_bindings(data, stale)
    except ValueError:
        pass
    else:
        raise AssertionError("stale sample clock identity was accepted")


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.complex_frf_impropriety.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["uncertainty_semantics"] == (
        "statistical_covariance_and_complementary_covariance"
    )

    improper = verify_case(data, "improper_case")
    proper = verify_case(data, "proper_control_case")
    assert improper["covariance"] == proper["covariance"] == (
        Fraction(5), Fraction(0)
    )
    assert improper["complementary_covariance"] == (
        Fraction(3), Fraction(1)
    )
    assert proper["complementary_covariance"] == (
        Fraction(0), Fraction(0)
    )
    assert improper["augmented_covariance"] != proper["augmented_covariance"]
    assert improper["real_covariance"] != proper["real_covariance"]
    assert data["same_ordinary_covariance_different_joint_law"] is True

    run_negative_controls(data, improper)

    print("AIRROAD complex FRF impropriety exact-rational reference: PASS")
    print(f"same ordinary covariance C: {improper['covariance']}")
    print(f"improper complementary covariance P: {improper['complementary_covariance']}")
    print(f"improper det(real covariance): {improper['real_determinant']}")
    print(f"improper det(augmented covariance): {improper['augmented_determinant']}")
    print(f"proper control P: {proper['complementary_covariance']}")
    print("same C, different P and different augmented joint law: PASS")
    print("PSD, properness-evidence, identity and hard-set negative controls: PASS")


if __name__ == "__main__":
    main()
