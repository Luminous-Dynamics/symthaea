#!/usr/bin/env python3
"""Exact-rational verifier for hierarchical frequency-response uncertainty."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-hierarchical-frf-uq-v0.1.json"


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def vector(values: list[dict[str, str]]) -> list[Fraction]:
    return [q(value) for value in values]


def matrix(values: list[list[dict[str, str]]]) -> list[list[Fraction]]:
    return [vector(row) for row in values]


def mean_vectors(rows: list[list[Fraction]]) -> list[Fraction]:
    if not rows:
        raise ValueError("at least one period is required")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("period vectors must have one common nonzero dimension")
    count = Fraction(len(rows))
    return [
        sum((row[col] for row in rows), Fraction(0)) / count
        for col in range(width)
    ]


def centered(rows: list[list[Fraction]]) -> tuple[list[Fraction], list[list[Fraction]]]:
    average = mean_vectors(rows)
    return average, [
        [value - average[col] for col, value in enumerate(row)]
        for row in rows
    ]


def sample_covariance_of_mean(
    rows: list[list[Fraction]],
) -> list[list[Fraction]]:
    """Compute 1/[P(P-1)] sum_s (h_s-h_bar)(h_s-h_bar)^T."""
    count = len(rows)
    if count < 2:
        raise ValueError("at least two independent periods are required")
    _, deviations = centered(rows)
    width = len(deviations[0])
    denominator = Fraction(count * (count - 1))
    result = [[Fraction(0) for _ in range(width)] for _ in range(width)]
    for deviation in deviations:
        for row in range(width):
            for col in range(width):
                result[row][col] += deviation[row] * deviation[col]
    return [
        [value / denominator for value in row]
        for row in result
    ]


def outer(left: list[Fraction], right: list[Fraction]) -> list[list[Fraction]]:
    if not left or len(left) != len(right):
        raise ValueError("outer-product vectors must have equal nonzero length")
    return [[a * b for b in right] for a in left]


def scale(
    scalar: Fraction,
    values: list[list[Fraction]],
) -> list[list[Fraction]]:
    return [[scalar * value for value in row] for row in values]


def add(
    left: list[list[Fraction]],
    right: list[list[Fraction]],
) -> list[list[Fraction]]:
    if len(left) != len(right) or any(
        len(a) != len(b) for a, b in zip(left, right)
    ):
        raise ValueError("covariance dimensions do not match")
    return [
        [a + b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(left, right)
    ]


def determinant_2x2(values: list[list[Fraction]]) -> Fraction:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise ValueError("V0.1 determinant oracle supports 2x2 matrices only")
    return values[0][0] * values[1][1] - values[0][1] * values[1][0]


def assert_symmetric_psd_2x2(values: list[list[Fraction]]) -> None:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise AssertionError("expected a 2x2 covariance matrix")
    assert values[0][1] == values[1][0], "covariance is not symmetric"
    assert values[0][0] >= 0
    assert values[1][1] >= 0
    assert determinant_2x2(values) >= 0


def decode_expected(values: list[list[dict[str, str]]]) -> list[list[Fraction]]:
    return matrix(values)


def period_rows(data: dict[str, Any]) -> list[list[Fraction]]:
    frequencies = data["frequencies"]
    expected_ids = [item["frequency_id"] for item in frequencies]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("frequency identities must be unique and nonempty")

    rows: list[list[Fraction]] = []
    for period in data["periods"]:
        actual_ids = [item["frequency_id"] for item in period["frequency_values"]]
        if actual_ids != expected_ids:
            raise ValueError("period frequency coverage/order mismatch")
        if period["campaign_id"] != data["campaign_id"]:
            raise ValueError("stale campaign identity")
        if period["acquisition_id"] != data["acquisition_id"]:
            raise ValueError("stale acquisition identity")
        if period["model_identity"] != data["model_identity"]:
            raise ValueError("stale model identity")
        if period["topology_identity"] != data["topology_identity"]:
            raise ValueError("stale topology identity")
        rows.append([q(item["value"]) for item in period["frequency_values"]])
    return rows


def calibration_covariance(data: dict[str, Any]) -> list[list[Fraction]]:
    calibration = data["calibration"]
    if calibration["scope"] != (
        "fixed_across_periods_and_frequencies_within_campaign"
    ):
        raise ValueError("calibration scope is not campaign-fixed")
    sensitivity = vector(calibration["sensitivity"])
    variance = q(calibration["variance"])
    if variance < 0:
        raise ValueError("calibration variance cannot be negative")
    return scale(variance, outer(sensitivity, sensitivity))


def compose_statistical_covariance(
    period_cov: list[list[Fraction]],
    calibration_cov: list[list[Fraction]],
    composition: dict[str, Any],
) -> list[list[Fraction]]:
    if not composition.get("source_independence_assumption", False):
        raise ValueError("cross-source covariance is not declared zero")
    if composition.get("period_noise_calibration_cross_covariance") != "declared_zero":
        raise ValueError("unknown/nonzero source cross-covariance requires cross terms")
    if composition.get("hard_model_discrepancy", {}).get("covariance_conversion") != "forbidden":
        raise ValueError("hard model-discrepancy semantics are not protected")
    return add(period_cov, calibration_cov)


def assert_metadata(data: dict[str, Any]) -> None:
    assert data["schema"] == "luminous.airroad.hierarchical_frf_uq.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == (
        "synthetic_exact_rational_statistical_covariance_reference"
    )
    assert data["uncertainty_semantics"] == "statistical_covariance"
    assert data["campaign_id"] == "campaign-hfrf-001"
    assert data["acquisition_id"] == "acquisition-multisine-001"
    assert data["frequency_unit"] == "normalized_frequency"
    assert data["model_identity"] == "sha256:hierarchical-frf-model-v0.1"
    assert data["topology_identity"] == "sha256:two-port-topology-v0.1"


def validate_calibration_bindings(data: dict[str, Any]) -> None:
    expected_frequency_ids = [item["frequency_id"] for item in data["frequencies"]]
    bindings = data["calibration"].get("frequency_bindings", [])
    actual_frequency_ids = [item["frequency_id"] for item in bindings]
    if actual_frequency_ids != expected_frequency_ids:
        raise ValueError("calibration frequency coverage/order mismatch")
    nuisance_ids = {item["nuisance_id"] for item in bindings}
    if nuisance_ids != {data["calibration"]["nuisance_id"]}:
        raise ValueError("shared calibration nuisance identity mismatch")


def assert_claim_controls(data: dict[str, Any]) -> None:
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls == {
        "fewer-than-two-periods": "reject",
        "wrong-sample-covariance-denominator": "reject",
        "uncentered-covariance": "reject",
        "calibration-id-mismatch-between-frequencies": "reject-dependency",
        "stale-acquisition-identity": "not-evaluable",
        "lost-cross-frequency-covariance": "reject-joint-law",
        "unknown-period-calibration-cross-covariance": "do-not-add",
        "covariance-promoted-to-hard-set": "reject-claim-class",
        "repeated-periods-claim-to-identify-fixed-calibration": "reject",
        "hard-model-error-converted-to-covariance": "reject-claim-class",
    }


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert_metadata(data)
    assert_claim_controls(data)

    rows = period_rows(data)
    validate_calibration_bindings(data)
    assert len(rows) >= 2
    average, deviations = centered(rows)
    expected = data["expected"]
    assert average == vector(expected["mean"])
    assert deviations == [vector(item) for item in expected["deviations"]]

    period_cov = sample_covariance_of_mean(rows)
    assert period_cov == decode_expected(expected["period_covariance"])
    assert_symmetric_psd_2x2(period_cov)
    period_det = determinant_2x2(period_cov)
    assert period_det == q(expected["period_covariance_determinant"])
    assert period_det == Fraction(1, 12)
    assert period_cov[0][1] == Fraction(1, 6)
    assert period_cov[0][1] != 0

    # Repeat-period covariance is invariant to an additive campaign-fixed
    # calibration offset because the estimator centers each coordinate.
    common_offset = [Fraction(7), Fraction(-3)]
    shifted_rows = [
        [value + common_offset[col] for col, value in enumerate(row)]
        for row in rows
    ]
    shifted_cov = sample_covariance_of_mean(shifted_rows)
    assert shifted_cov == period_cov
    assert data["expected"]["common_offset_does_not_change_centered_covariance"]

    # Two common mistakes must not match the frozen estimator.
    uncentered = [[Fraction(0), Fraction(0)] for _ in range(2)]
    for row in rows:
        contribution = outer(row, row)
        uncentered = add(uncentered, contribution)
    uncentered = scale(Fraction(1, len(rows) * (len(rows) - 1)), uncentered)
    assert uncentered != period_cov

    # This is the covariance of one period's observations, not the covariance
    # of their sample mean; it is larger by a factor P.
    individual_period_cov = scale(Fraction(len(rows)), period_cov)
    assert individual_period_cov != period_cov

    cal_cov = calibration_covariance(data)
    assert cal_cov == decode_expected(expected["calibration_covariance"])
    assert_symmetric_psd_2x2(cal_cov)
    assert determinant_2x2(cal_cov) == 0  # one shared scalar nuisance coordinate

    total_cov = compose_statistical_covariance(
        period_cov,
        cal_cov,
        data["composition"],
    )
    assert total_cov == decode_expected(expected["total_covariance"])
    assert_symmetric_psd_2x2(total_cov)
    total_det = determinant_2x2(total_cov)
    assert total_det == q(expected["total_covariance_determinant"])
    assert total_det == Fraction(7, 75)

    # Diagonalization preserves both marginal variances, but not the joint law.
    diagonal = [
        [total_cov[0][0], Fraction(0)],
        [Fraction(0), total_cov[1][1]],
    ]
    assert diagonal == decode_expected(expected["diagonal_only_counterfactual"])
    assert diagonal[0][0] == total_cov[0][0]
    assert diagonal[1][1] == total_cov[1][1]
    assert diagonal != total_cov
    assert diagonal[0][1] == 0 and total_cov[0][1] == Fraction(14, 75)

    # Unknown cross-source dependence must prevent additive covariance claims.
    unknown_composition = dict(data["composition"])
    unknown_composition["source_independence_assumption"] = False
    unknown_composition["period_noise_calibration_cross_covariance"] = "unknown"
    try:
        compose_statistical_covariance(period_cov, cal_cov, unknown_composition)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown cross-source dependence was silently set to zero")

    # A deterministic hard discrepancy is not part of a statistical covariance.
    hard_as_covariance = dict(data["composition"])
    hard_as_covariance["hard_model_discrepancy"] = {
        "semantics": "deterministic_hard_set",
        "covariance_conversion": "promoted",
        "value": {"radius": "1/10"},
    }
    try:
        compose_statistical_covariance(period_cov, cal_cov, hard_as_covariance)
    except ValueError:
        pass
    else:
        raise AssertionError("hard model discrepancy was promoted into covariance")

    # P=1 has no sample-covariance estimator.
    try:
        sample_covariance_of_mean(rows[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("single-period sample covariance was accepted")

    stale_periods = json.loads(json.dumps(data))
    stale_periods["periods"][1]["acquisition_id"] = "stale-acquisition"
    try:
        period_rows(stale_periods)
    except ValueError:
        pass
    else:
        raise AssertionError("stale acquisition identity was accepted")

    split_calibration = json.loads(json.dumps(data))
    split_calibration["calibration"]["frequency_bindings"][1]["nuisance_id"] = (
        "different-calibration-root"
    )
    try:
        validate_calibration_bindings(split_calibration)
    except ValueError:
        pass
    else:
        raise AssertionError("frequency-wise calibration ID change was accepted")

    print("AIRROAD hierarchical FRF UQ exact-rational reference: PASS")
    print(f"period covariance: {period_cov}")
    print(f"period covariance determinant: {period_det}")
    print(f"shared calibration covariance: {cal_cov}")
    print(f"independent-source total covariance: {total_cov}")
    print(f"total covariance determinant: {total_det}")
    print(f"same-marginal diagonal ablation differs: {diagonal != total_cov}")
    print("fixed shared offset is not identified by centered repeat-period covariance")
    print("hard-set and unknown-dependence promotion controls: PASS")


if __name__ == "__main__":
    main()
