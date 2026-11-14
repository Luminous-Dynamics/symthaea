#!/usr/bin/env python3
"""Exact-rational admissibility gate for finite-sample MIMO multisine FRF designs."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-finite-sample-frf-v0.1.json"


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def decode_matrix(values: list[list[dict[str, str]]]) -> list[list[Fraction]]:
    if not values or not values[0]:
        raise ValueError("matrix must be nonempty")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("matrix rows must have equal lengths")
    return [[q(item) for item in row] for row in values]


def determinant_2x2(values: list[list[Fraction]]) -> Fraction:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise ValueError("V0.1 exact rank oracle expects 2x2 matrices")
    return values[0][0] * values[1][1] - values[0][1] * values[1][0]


def full_column_rank_2x2(values: list[list[Fraction]]) -> bool:
    return determinant_2x2(values) != 0


def modulo_one(value: Fraction) -> Fraction:
    """Exact residue in [0, 1)."""
    return value % 1


def signed_alias_residues(
    normalized_frequencies: list[Fraction],
) -> tuple[list[Fraction], dict[Fraction, list[str]]]:
    residues: list[Fraction] = [Fraction(0)]
    labels: dict[Fraction, list[str]] = {Fraction(0): ["dc"]}
    for index, frequency in enumerate(normalized_frequencies, start=1):
        if frequency <= 0:
            raise ValueError("positive multisine lines must have positive frequency")
        positive = modulo_one(frequency)
        negative = modulo_one(-frequency)
        for residue, label in (
            (positive, f"f{index}:+"),
            (negative, f"f{index}:-"),
        ):
            residues.append(residue)
            labels.setdefault(residue, []).append(label)
    return residues, labels


def check_aliases(normalized_frequencies: list[Fraction]) -> dict[str, Any]:
    residues, labels = signed_alias_residues(normalized_frequencies)
    collisions = {
        str(residue): names
        for residue, names in labels.items()
        if len(names) > 1
    }
    return {
        "distinct": not collisions,
        "residues": sorted(set(residues)),
        "collisions": collisions,
        "count_with_multiplicity": len(residues),
    }


def decode_design(
    design: dict[str, Any],
    expected_frequency_ids: list[str],
    n_inputs: int,
) -> tuple[list[list[Fraction]], list[list[list[Fraction]]], list[list[Fraction]]]:
    a0 = decode_matrix(design["dc_offset_matrix_a0"])
    if len(a0) != n_inputs or len(a0[0]) != n_inputs:
        raise ValueError("A0 dimensions do not match n_inputs")

    line_records = design["frequency_matrices"]
    if [item["frequency_id"] for item in line_records] != expected_frequency_ids:
        raise ValueError("input design frequency identity/order mismatch")

    line_matrices: list[list[list[Fraction]]] = []
    all_amplitudes: list[list[Fraction]] = []
    for record in line_records:
        amplitudes = decode_matrix(record["amplitudes"])
        a_l = decode_matrix(record["a_l"])
        signs = record["phase_signs"]
        if len(amplitudes) != n_inputs or len(amplitudes[0]) != n_inputs:
            raise ValueError("amplitude matrix dimensions do not match n_inputs")
        if len(a_l) != n_inputs or len(a_l[0]) != n_inputs:
            raise ValueError("A_l dimensions do not match n_inputs")
        if len(signs) != n_inputs or any(len(row) != n_inputs for row in signs):
            raise ValueError("phase sign matrix dimensions do not match n_inputs")
        if any(sign not in (-1, 1) for row in signs for sign in row):
            raise ValueError("V0.1 phase signs must be +1 or -1")
        if any(value <= 0 for row in amplitudes for value in row):
            raise ValueError("every excited input amplitude must be nonzero and positive")
        for row in range(n_inputs):
            for col in range(n_inputs):
                if a_l[row][col] != amplitudes[row][col] * signs[row][col] / 2:
                    raise ValueError("A_l does not match amplitude/phase encoding")
        line_matrices.append(a_l)
        all_amplitudes.extend(amplitudes)
    return a0, line_matrices, all_amplitudes


def exact_design_gate(data: dict[str, Any]) -> dict[str, Any]:
    dimensions = data["dimensions"]
    n_inputs = int(dimensions["n_inputs"])
    experiments = int(dimensions["experiments"])
    lines = int(dimensions["positive_frequency_lines"])
    samples = int(dimensions["samples_per_experiment"])

    frequencies = data["frequency_lines"]
    frequency_ids = [item["frequency_id"] for item in frequencies]
    if len(frequency_ids) != lines or len(set(frequency_ids)) != lines:
        raise ValueError("frequency line count or identities do not match dimensions")
    q_values = [q(item["cycles_per_sample"]) for item in frequencies]
    if q_values != sorted(q_values) or any(value <= 0 for value in q_values):
        raise ValueError("positive frequency lines must be strictly increasing")
    if len(set(q_values)) != len(q_values):
        raise ValueError("positive frequency lines must be unique")

    alias = check_aliases(q_values)
    a0, line_matrices, amplitudes = decode_design(
        data["input_design"], frequency_ids, n_inputs
    )
    determinants = {
        "a0": determinant_2x2(a0),
        **{
            f"a{index}": determinant_2x2(matrix)
            for index, matrix in enumerate(line_matrices, start=1)
        },
    }
    rank_pass = all(value != 0 for value in determinants.values())
    experiment_count_pass = experiments >= n_inputs
    sample_count_pass = samples > 2 * lines
    stationary = data["stationarity"]["status"] == "declared_synthetic_true"
    noise = data["noise"]
    noise_mean_zero = noise["mean"] == "zero"
    noise_iid = (
        noise["temporal_model"] == "iid_white"
        and noise["experiment_model"] == "independent"
    )
    noise_covariance = decode_matrix(noise["output_covariance"])
    noise_covariance_positive_definite = (
        len(noise_covariance) == 2
        and determinant_2x2(noise_covariance) > 0
        and all(noise_covariance[i][i] > 0 for i in range(2))
    )

    whole_period_counts = [Fraction(samples) * value for value in q_values]
    no_spectral_leakage = all(value.denominator == 1 for value in whole_period_counts)

    base_admissible = (
        experiment_count_pass
        and rank_pass
        and alias["distinct"]
        and sample_count_pass
        and stationary
        and noise_mean_zero
        and noise_iid
    )
    covariance_eligible = base_admissible and noise_covariance_positive_definite
    decorrelation_eligible = covariance_eligible and no_spectral_leakage

    return {
        "experiment_count_pass": experiment_count_pass,
        "input_design_rank_pass": rank_pass,
        "a0_determinant": determinants["a0"],
        "frequency_determinants": {
            key: value for key, value in determinants.items() if key != "a0"
        },
        "alias_nonoverlap_pass": alias["distinct"],
        "alias_residues": alias["residues"],
        "alias_collisions": alias["collisions"],
        "sample_count_pass": sample_count_pass,
        "stationarity_declared": stationary,
        "zero_mean_noise_declared": noise_mean_zero,
        "iid_white_noise_declared": noise_iid,
        "noise_covariance_positive_definite": noise_covariance_positive_definite,
        "whole_period_counts": whole_period_counts,
        "no_spectral_leakage_pass": no_spectral_leakage,
        "ls_unbiasedness_eligible": base_admissible,
        "theorem_2_covariance_eligible": covariance_eligible,
        "frequency_decorrelation_eligible": decorrelation_eligible,
        "continuous_frequency_adequacy": "not_established",
    }


def validate_metadata(data: dict[str, Any]) -> None:
    assert data["schema"] == "luminous.airroad.mimo_finite_sample_frf_admissibility.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_experiment_design"
    assert data["claim_class"] == "statistical_estimator_admissibility"
    assert data["frequency_unit"] == "cycles_per_sample"
    assert data["sample_clock"]["clock_identity"] == "sample-clock-001"
    assert data["sample_clock"]["sample_period_identity"] == "sample-period-001"
    assert data["sample_clock"]["relation_status"] == "exact_rational"


def run_negative_controls(data: dict[str, Any]) -> None:
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls == {
        "fewer-experiments-than-inputs": "reject",
        "rank-deficient-dc-offset-matrix": "reject",
        "rank-deficient-frequency-amplitude-matrix": "reject",
        "zero-frequency-amplitude": "reject",
        "positive-negative-alias-collision": "reject",
        "nyquist-positive-negative-collision": "reject",
        "integer-shift-alias-collision": "reject",
        "sample-count-not-greater-than-two-m": "reject",
        "no-spectral-leakage-not-satisfied": "unbiased-eligible-decorrelation-ineligible",
        "stationarity-not-established": "not-evaluable",
        "unknown-sample-clock-relation": "not-evaluable",
        "clock-identity-mismatch": "not-evaluable",
        "statistical-covariance-promoted-to-hard-set": "reject-claim-class",
        "knotwise-frf-promoted-to-continuous-band": "reject-claim-class",
    }

    assert not check_aliases([Fraction(1, 5), Fraction(4, 5)])["distinct"]
    assert not check_aliases([Fraction(1, 2)])["distinct"]
    assert not check_aliases([Fraction(6, 5), Fraction(1, 5)])["distinct"]
    assert check_aliases([Fraction(1, 5), Fraction(1, 3)])["distinct"]

    # When the signed alias lines are distinct but the record does not contain
    # whole periods, the unbiased/covariance conditions can still hold while
    # the stronger uncorrelated-frequency consequence is unavailable.
    data_without_leakage = json.loads(json.dumps(data))
    data_without_leakage["dimensions"]["samples_per_experiment"] = 5
    leakage_gate = exact_design_gate(data_without_leakage)
    assert leakage_gate["ls_unbiasedness_eligible"] is True
    assert leakage_gate["theorem_2_covariance_eligible"] is True
    assert leakage_gate["frequency_decorrelation_eligible"] is False
    assert leakage_gate["alias_nonoverlap_pass"] is True

    # Rank-deficient A0 must be rejected even with enough experiments.
    rank_deficient_a0 = json.loads(json.dumps(data))
    rank_deficient_a0["input_design"]["dc_offset_matrix_a0"] = [
        [{"numerator": "1", "denominator": "1"},
         {"numerator": "2", "denominator": "1"}],
        [{"numerator": "2", "denominator": "1"},
         {"numerator": "4", "denominator": "1"}],
    ]
    assert exact_design_gate(rank_deficient_a0)["ls_unbiasedness_eligible"] is False

    # A_l rank failure is independent of A0 rank.
    rank_deficient_line = json.loads(json.dumps(data))
    rank_deficient_line["input_design"]["frequency_matrices"][0]["a_l"] = [
        [{"numerator": "1", "denominator": "2"},
         {"numerator": "1", "denominator": "2"}],
        [{"numerator": "1", "denominator": "2"},
         {"numerator": "1", "denominator": "2"}],
    ]
    assert exact_design_gate(rank_deficient_line)["ls_unbiasedness_eligible"] is False

    zero_amplitude = json.loads(json.dumps(data))
    zero_amplitude["input_design"]["frequency_matrices"][0]["amplitudes"][0][0] = {
        "numerator": "0", "denominator": "1"
    }
    try:
        exact_design_gate(zero_amplitude)
    except ValueError:
        pass
    else:
        raise AssertionError("zero input amplitude was accepted")

    insufficient_experiments = json.loads(json.dumps(data))
    insufficient_experiments["dimensions"]["experiments"] = 1
    assert exact_design_gate(insufficient_experiments)["ls_unbiasedness_eligible"] is False

    insufficient_samples = json.loads(json.dumps(data))
    insufficient_samples["dimensions"]["samples_per_experiment"] = 4
    assert exact_design_gate(insufficient_samples)["ls_unbiasedness_eligible"] is False

    nonstationary = json.loads(json.dumps(data))
    nonstationary["stationarity"]["status"] = "not_established"
    assert exact_design_gate(nonstationary)["ls_unbiasedness_eligible"] is False

    bad_clock = json.loads(json.dumps(data))
    bad_clock["sample_clock"]["relation_status"] = "unknown_interval"
    assert bad_clock["sample_clock"]["relation_status"] != "exact_rational"

    stale_clock = json.loads(json.dumps(data))
    stale_clock["sample_clock"]["clock_identity"] = "different-clock"
    assert stale_clock["sample_clock"]["clock_identity"] != "sample-clock-001"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    validate_metadata(data)
    gate = exact_design_gate(data)

    expected = data["expected"]
    assert gate["alias_residues"] == [
        q(value) for value in expected["signed_alias_residues"]
    ]
    assert gate["alias_nonoverlap_pass"] is expected["alias_residues_distinct"]
    assert gate["a0_determinant"] == q(expected["determinants"]["a0"])
    for key, encoded in expected["determinants"].items():
        actual = (
            gate["a0_determinant"]
            if key == "a0"
            else gate["frequency_determinants"][key]
        )
        assert actual == q(encoded)
    assert gate["sample_count_pass"] is expected["sample_count_condition"]
    assert gate["whole_period_counts"] == [
        q(value) for value in expected["whole_period_counts"]
    ]
    assert gate["ls_unbiasedness_eligible"] is expected["ls_unbiasedness_eligible"]
    assert gate["theorem_2_covariance_eligible"] is expected["theorem_2_covariance_eligible"]
    assert gate["frequency_decorrelation_eligible"] is expected["frequency_decorrelation_eligible"]
    assert gate["continuous_frequency_adequacy"] == expected["continuous_frequency_adequacy"]

    run_negative_controls(data)

    print("AIRROAD finite-sample MIMO FRF admissibility reference: PASS")
    print(f"exact signed alias residues: {gate['alias_residues']}")
    print(f"input determinants: A0={gate['a0_determinant']}, "
          f"frequency={gate['frequency_determinants']}")
    print(f"whole-period counts: {gate['whole_period_counts']}")
    print(f"LS unbiasedness eligible: {gate['ls_unbiasedness_eligible']}")
    print(f"Theorem 2 covariance eligible: {gate['theorem_2_covariance_eligible']}")
    print(f"frequency decorrelation eligible: {gate['frequency_decorrelation_eligible']}")
    print(f"continuous-frequency adequacy: {gate['continuous_frequency_adequacy']}")
    print("alias, rank, stationarity, sample-count and leakage negative controls: PASS")


if __name__ == "__main__":
    main()
