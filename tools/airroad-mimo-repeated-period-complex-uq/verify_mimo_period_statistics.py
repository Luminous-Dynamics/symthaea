#!/usr/bin/env python3
"""Exact-rational MIMO repeated-period covariance and complementary covariance."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-repeated-period-complex-uq-v0.1.json"

ComplexQ = tuple[Fraction, Fraction]
ComplexMatrix = list[list[ComplexQ]]
RealMatrix = list[list[Fraction]]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def read_complex(value: dict[str, Any]) -> ComplexQ:
    return q(value["re"]), q(value["im"])


def cadd(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] + right[0], left[1] + right[1]


def csub(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] - right[0], left[1] - right[1]


def cmul(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def cconj(value: ComplexQ) -> ComplexQ:
    return value[0], -value[1]


def cdiv(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    denominator = right[0] * right[0] + right[1] * right[1]
    if denominator == 0:
        raise ZeroDivisionError("complex rational division by zero")
    return (
        (left[0] * right[0] + left[1] * right[1]) / denominator,
        (left[1] * right[0] - left[0] * right[1]) / denominator,
    )


def czero() -> ComplexQ:
    return Fraction(0), Fraction(0)


def cone() -> ComplexQ:
    return Fraction(1), Fraction(0)


def is_czero(value: ComplexQ) -> bool:
    return value == czero()


def decode_real_matrix(values: list[list[dict[str, str]]]) -> RealMatrix:
    if not values or not values[0]:
        raise ValueError("real matrix cannot be empty")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("real matrix rows must have equal length")
    return [[q(item) for item in row] for row in values]


def decode_complex_matrix(values: list[list[dict[str, Any]]]) -> ComplexMatrix:
    if not values or not values[0]:
        raise ValueError("complex matrix cannot be empty")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("complex matrix rows must have equal length")
    return [[read_complex(item) for item in row] for row in values]


def real_determinant(matrix: RealMatrix) -> Fraction:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("real determinant requires a nonempty square matrix")
    work = [row[:] for row in matrix]
    determinant = Fraction(1)
    for pivot_index in range(size):
        pivot = work[pivot_index][pivot_index]
        if pivot == 0:
            swap = next(
                (row for row in range(pivot_index + 1, size)
                 if work[row][pivot_index] != 0),
                None,
            )
            if swap is None:
                return Fraction(0)
            work[pivot_index], work[swap] = work[swap], work[pivot_index]
            determinant = -determinant
            pivot = work[pivot_index][pivot_index]
        determinant *= pivot
        for row in range(pivot_index + 1, size):
            factor = work[row][pivot_index] / pivot
            for col in range(pivot_index + 1, size):
                work[row][col] -= factor * work[pivot_index][col]
    return determinant


def complex_determinant(matrix: ComplexMatrix) -> ComplexQ:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("complex determinant requires a nonempty square matrix")
    work = [row[:] for row in matrix]
    determinant = cone()
    for pivot_index in range(size):
        pivot = work[pivot_index][pivot_index]
        if is_czero(pivot):
            swap = next(
                (row for row in range(pivot_index + 1, size)
                 if not is_czero(work[row][pivot_index])),
                None,
            )
            if swap is None:
                return czero()
            work[pivot_index], work[swap] = work[swap], work[pivot_index]
            determinant = (-determinant[0], -determinant[1])
            pivot = work[pivot_index][pivot_index]
        determinant = cmul(determinant, pivot)
        for row in range(pivot_index + 1, size):
            factor = cdiv(work[row][pivot_index], pivot)
            for col in range(pivot_index + 1, size):
                work[row][col] = csub(
                    work[row][col],
                    cmul(factor, work[pivot_index][col]),
                )
    return determinant


def matrix_rank(matrix: RealMatrix) -> int:
    if not matrix or not matrix[0]:
        return 0
    work = [row[:] for row in matrix]
    row_count, col_count = len(work), len(work[0])
    if any(len(row) != col_count for row in work):
        raise ValueError("rank matrix rows must have equal length")
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
        divisor = work[pivot_row][col]
        for j in range(col, col_count):
            work[pivot_row][j] /= divisor
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = work[row][col]
            if factor != 0:
                for j in range(col, col_count):
                    work[row][j] -= factor * work[pivot_row][j]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def mean_complex_rows(
    rows: list[list[ComplexQ]],
) -> list[ComplexQ]:
    if not rows or not rows[0]:
        raise ValueError("at least one nonempty period vector is required")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("period vectors must share one coordinate count")
    count = Fraction(len(rows))
    result: list[ComplexQ] = []
    for col in range(width):
        total = czero()
        for row in rows:
            total = cadd(total, row[col])
        result.append((total[0] / count, total[1] / count))
    return result


def center_complex_rows(
    rows: list[list[ComplexQ]],
) -> tuple[list[ComplexQ], list[list[ComplexQ]]]:
    average = mean_complex_rows(rows)
    deviations = [
        [csub(value, average[col]) for col, value in enumerate(row)]
        for row in rows
    ]
    return average, deviations


def complex_mean_covariances(
    rows: list[list[ComplexQ]],
) -> tuple[list[ComplexQ], list[list[ComplexQ]], ComplexMatrix, ComplexMatrix]:
    period_count = len(rows)
    if period_count < 2:
        raise ValueError("at least two independent periods are required")
    average, deviations = center_complex_rows(rows)
    width = len(average)
    denominator = Fraction(period_count * (period_count - 1))
    covariance = [[czero() for _ in range(width)] for _ in range(width)]
    complementary = [[czero() for _ in range(width)] for _ in range(width)]

    for deviation in deviations:
        for row in range(width):
            for col in range(width):
                covariance[row][col] = cadd(
                    covariance[row][col],
                    cmul(deviation[row], cconj(deviation[col])),
                )
                complementary[row][col] = cadd(
                    complementary[row][col],
                    cmul(deviation[row], deviation[col]),
                )

    covariance = [
        [(value[0] / denominator, value[1] / denominator) for value in row]
        for row in covariance
    ]
    complementary = [
        [(value[0] / denominator, value[1] / denominator) for value in row]
        for row in complementary
    ]
    return average, deviations, covariance, complementary


def real_vectors_from_complex(rows: list[list[ComplexQ]]) -> list[list[Fraction]]:
    width = len(rows[0])
    return [
        [value[0] for value in row] + [value[1] for value in row]
        for row in rows
    ]


def real_covariance_of_mean(
    rows: list[list[ComplexQ]],
) -> tuple[list[Fraction], list[list[Fraction]], RealMatrix]:
    period_count = len(rows)
    if period_count < 2:
        raise ValueError("at least two independent periods are required")
    average, deviations = center_complex_rows(rows)
    real_deviations = real_vectors_from_complex(deviations)
    dimension = len(real_deviations[0])
    denominator = Fraction(period_count * (period_count - 1))
    covariance = [
        [Fraction(0) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for deviation in real_deviations:
        for row in range(dimension):
            for col in range(dimension):
                covariance[row][col] += deviation[row] * deviation[col]
    covariance = [
        [value / denominator for value in row]
        for row in covariance
    ]
    real_mean = [value for z_value in average for value in ()]
    real_mean = [value[0] for value in average] + [
        value[1] for value in average
    ]
    return real_mean, real_deviations, covariance


def complex_moments_from_real(
    sigma: RealMatrix,
) -> tuple[ComplexMatrix, ComplexMatrix]:
    size = len(sigma)
    if size == 0 or size % 2 != 0 or any(len(row) != size for row in sigma):
        raise ValueError("real covariance must be square with even dimension")
    n = size // 2
    for row in range(size):
        for col in range(size):
            if sigma[row][col] != sigma[col][row]:
                raise ValueError("real covariance is not symmetric")

    a = [row[:n] for row in sigma[:n]]
    b = [row[n:] for row in sigma[:n]]
    bt = [row[:n] for row in sigma[n:]]
    d = [row[n:] for row in sigma[n:]]

    covariance: ComplexMatrix = []
    complementary: ComplexMatrix = []
    for row in range(n):
        c_row: list[ComplexQ] = []
        p_row: list[ComplexQ] = []
        for col in range(n):
            c_row.append((
                a[row][col] + d[row][col],
                bt[row][col] - b[row][col],
            ))
            p_row.append((
                a[row][col] - d[row][col],
                b[row][col] + bt[row][col],
            ))
        covariance.append(c_row)
        complementary.append(p_row)
    return covariance, complementary


def real_covariance_from_complex(
    covariance: ComplexMatrix,
    complementary: ComplexMatrix,
) -> RealMatrix:
    n = len(covariance)
    if n == 0 or len(complementary) != n:
        raise ValueError("C and P dimensions do not match")
    if any(len(row) != n for row in covariance + complementary):
        raise ValueError("C and P must be square")
    xx = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    yy = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    xy = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    yx = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(n):
            c_value = covariance[row][col]
            p_value = complementary[row][col]
            xx[row][col] = (c_value[0] + p_value[0]) / 2
            yy[row][col] = (c_value[0] - p_value[0]) / 2
            xy[row][col] = (p_value[1] - c_value[1]) / 2
            yx[row][col] = (p_value[1] + c_value[1]) / 2
    for row in range(n):
        for col in range(n):
            if xy[row][col] != yx[col][row]:
                raise ValueError("C/P map does not produce symmetric real covariance")
    return [
        xx[row] + xy[row]
        for row in range(n)
    ] + [
        yx[row] + yy[row]
        for row in range(n)
    ]


def augmented_covariance(
    covariance: ComplexMatrix,
    complementary: ComplexMatrix,
) -> ComplexMatrix:
    n = len(covariance)
    if n == 0 or len(complementary) != n:
        raise ValueError("augmented covariance inputs do not match")
    top = [
        covariance[row][:] + complementary[row][:]
        for row in range(n)
    ]
    bottom = [
        [cconj(value) for value in complementary[row]]
        + [cconj(value) for value in covariance[row]]
        for row in range(n)
    ]
    return top + bottom


def leading_real_minors(matrix: RealMatrix) -> list[Fraction]:
    return [
        real_determinant([row[:size] for row in matrix[:size]])
        for size in range(1, len(matrix) + 1)
    ]


def leading_complex_minors(matrix: ComplexMatrix) -> list[Fraction]:
    result: list[Fraction] = []
    for size in range(1, len(matrix) + 1):
        determinant = complex_determinant(
            [row[:size] for row in matrix[:size]]
        )
        if determinant[1] != 0:
            raise AssertionError("Hermitian leading principal minor is not real")
        result.append(determinant[0])
    return result


def assert_hermitian_and_p_symmetric(
    covariance: ComplexMatrix,
    complementary: ComplexMatrix,
) -> None:
    n = len(covariance)
    for row in range(n):
        for col in range(n):
            if covariance[row][col] != cconj(covariance[col][row]):
                raise AssertionError("C is not Hermitian")
            if complementary[row][col] != complementary[col][row]:
                raise AssertionError("P is not complex symmetric")


def validate_identity(data: dict[str, Any]) -> None:
    period_ids = [period["period_id"] for period in data["periods"]]
    if len(period_ids) != len(set(period_ids)):
        raise ValueError("period identities must be unique")
    required = {
        "campaign_id": data["campaign_id"],
        "acquisition_id": data["acquisition_id"],
        "model_identity": data["model_identity"],
        "topology_identity": data["topology_identity"],
        "sample_clock_identity": data["sample_clock_identity"],
        "sample_period_identity": data["sample_period_identity"],
        "calibration_nuisance_id": data["calibration"]["nuisance_id"],
    }
    for period in data["periods"]:
        for key, expected in required.items():
            if period.get(key) != expected:
                raise ValueError(f"period identity mismatch: {key}")
    coordinate_ids = [item["coordinate_id"] for item in data["coordinates"]]
    if len(coordinate_ids) != len(set(coordinate_ids)):
        raise ValueError("coordinate IDs must be unique")
    if len(coordinate_ids) != len(data["vectorization_order"]):
        raise ValueError("coordinate count and vectorization order disagree")
    if data["vectorization_order"] != [
        "f0:Re(z0)", "f1:Re(z1)", "f0:Im(z0)", "f1:Im(z1)"
    ]:
        raise ValueError("unsupported or changed vectorization order")


def validate_properness_claim(
    complementary: ComplexMatrix,
    requested_proper: bool,
) -> None:
    actually_proper = all(
        is_czero(value)
        for row in complementary
        for value in row
    )
    if requested_proper and not actually_proper:
        raise ValueError("nonzero P cannot be silently discarded under properness")


def validate_claim_class(source_semantics: str, requested_claim: str) -> None:
    if (
        source_semantics == "statistical_covariance_and_complementary_covariance"
        and requested_claim in {"deterministic_hard_set", "continuous_model_error_bound"}
    ):
        raise ValueError("statistical C/P cannot be promoted to a deterministic hard set")


def validate_covariance_composition(
    independence_declared: bool,
    cross_covariance_known: bool,
) -> None:
    if not independence_declared and not cross_covariance_known:
        raise ValueError("unknown source dependence is not zero; covariance is NOT_EVALUABLE")


def common_offset(rows: list[list[ComplexQ]], offset: list[ComplexQ]) -> list[list[ComplexQ]]:
    return [
        [cadd(value, offset[col]) for col, value in enumerate(row)]
        for row in rows
    ]


def run_negative_controls(data: dict[str, Any], raw_rows: list[list[ComplexQ]]) -> None:
    try:
        complex_mean_covariances(raw_rows[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("P=1 repeated-period covariance was accepted")

    duplicate = json.loads(json.dumps(data))
    duplicate["periods"][1]["period_id"] = duplicate["periods"][0]["period_id"]
    try:
        validate_identity(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate period ID was accepted")

    stale = json.loads(json.dumps(data))
    stale["periods"][2]["acquisition_id"] = "stale-acquisition"
    try:
        validate_identity(stale)
    except ValueError:
        pass
    else:
        raise AssertionError("stale acquisition identity was accepted")

    split_cal = json.loads(json.dumps(data))
    split_cal["periods"][3]["calibration_nuisance_id"] = "different-calibration"
    try:
        validate_identity(split_cal)
    except ValueError:
        pass
    else:
        raise AssertionError("split calibration nuisance was accepted")

    reordered = json.loads(json.dumps(data))
    reordered["vectorization_order"][0] = "f1:Re(z1)"
    try:
        validate_identity(reordered)
    except ValueError:
        pass
    else:
        raise AssertionError("changed vectorization order was accepted")

    duplicate_cov = data["expected"]["complex_covariance"]
    duplicate_p = data["expected"]["complementary_covariance"]
    try:
        validate_properness_claim(
            decode_complex_matrix(duplicate_p),
            requested_proper=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero P was silently dropped")

    try:
        validate_claim_class(
            "statistical_covariance_and_complementary_covariance",
            "deterministic_hard_set",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("statistical covariance promoted to hard set")

    try:
        validate_covariance_composition(
            independence_declared=False,
            cross_covariance_known=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown covariance dependence treated as zero")

    truncated = raw_rows[:4]
    _, _, real4 = real_covariance_of_mean(truncated)
    rank4 = matrix_rank(real4)
    assert rank4 <= 3
    assert rank4 == 3
    if rank4 == 4:
        raise AssertionError("four-period rank bound was violated")

    try:
        validate_claim_class(
            "statistical_covariance_and_complementary_covariance",
            "continuous_model_error_bound",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("statistical C/P promoted to continuous hard bound")


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_repeated_period_complex_uq.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["uncertainty_semantics"] == (
        "statistical_covariance_and_complementary_covariance"
    )
    validate_identity(data)

    raw_rows = [
        [read_complex(value) for value in period["values"]]
        for period in data["periods"]
    ]
    mean, deviations, covariance, complementary = complex_mean_covariances(raw_rows)
    real_mean, real_deviations, real_covariance = real_covariance_of_mean(raw_rows)

    expected = data["expected"]
    expected_mean = [read_complex(value) for value in expected["sample_mean"]]
    assert mean == expected_mean
    assert real_mean == [value[0] for value in mean] + [value[1] for value in mean]
    assert deviations == [
        [csub(value, mean[col]) for col, value in enumerate(row)]
        for row in raw_rows
    ]

    expected_c = decode_complex_matrix(expected["complex_covariance"])
    expected_p = decode_complex_matrix(expected["complementary_covariance"])
    expected_real = decode_real_matrix(expected["real_covariance_of_mean"])
    expected_augmented = decode_complex_matrix(expected["augmented_covariance"])
    assert covariance == expected_c
    assert complementary == expected_p
    assert real_covariance == expected_real
    assert complex_moments_from_real(real_covariance) == (covariance, complementary)
    assert real_covariance_from_complex(covariance, complementary) == real_covariance

    augmented = augmented_covariance(covariance, complementary)
    assert augmented == expected_augmented
    assert_hermitian_and_p_symmetric(covariance, complementary)
    assert_hermitian_and_p_symmetric(
        [[augmented[row][col] for col in range(2)] for row in range(2)],
        [[augmented[row][col + 2] for col in range(2)] for row in range(2)],
    )

    real_minors = leading_real_minors(real_covariance)
    augmented_minors = leading_complex_minors(augmented)
    assert real_minors == [
        q(value) for value in expected["real_covariance_leading_principal_minors"]
    ]
    assert augmented_minors == [
        q(value) for value in expected["augmented_leading_principal_minors"]
    ]
    assert all(value > 0 for value in real_minors)
    assert all(value > 0 for value in augmented_minors)

    real_det = real_determinant(real_covariance)
    augmented_det = complex_determinant(augmented)
    assert augmented_det[1] == 0
    assert real_det == q(expected["real_covariance_determinant"])
    assert augmented_det[0] == q(expected["augmented_covariance_determinant"])
    assert augmented_det[0] == (
        expected["augmented_determinant_factor_over_real"] * real_det
    )

    # A shared campaign-fixed additive complex offset vanishes under centering.
    offset = [
        (Fraction(7, 3), Fraction(-2, 5)),
        (Fraction(-4, 7), Fraction(3, 8)),
    ]
    shifted = common_offset(raw_rows, offset)
    _, _, shifted_c, shifted_p = complex_mean_covariances(shifted)
    _, _, shifted_real = real_covariance_of_mean(shifted)
    assert shifted_c == covariance
    assert shifted_p == complementary
    assert shifted_real == real_covariance
    assert expected["common_offset_invariance"] is True

    # The four-period truncation is mathematically rank-deficient in R^4.
    truncated_real = real_covariance_of_mean(raw_rows[:4])[2]
    truncated_rank = matrix_rank(truncated_real)
    assert truncated_rank == expected["four_period_rank_upper_bound"] == 3
    assert real_determinant(truncated_real) == 0
    assert expected["four_period_real_covariance_singular"] is True
    assert expected["four_period_augmented_covariance_singular"] is True
    assert expected["proper"] is False
    assert any(not is_czero(value) for row in complementary for value in row)

    run_negative_controls(data, raw_rows)

    print("AIRROAD MIMO repeated-period complex UQ reference: PASS")
    print(f"period count: {len(raw_rows)}; complex dimension: {len(mean)}")
    print(f"mean: {mean}")
    print(f"C: {covariance}")
    print(f"P: {complementary}")
    print(f"det(real covariance): {real_det}")
    print(f"det(augmented covariance): {augmented_det[0]}")
    print(f"real leading principal minors: {real_minors}")
    print(f"augmented leading principal minors: {augmented_minors}")
    print("real covariance <-> (C,P) reconstruction: PASS")
    print("campaign-fixed offset invariance and shared-calibration identity: PASS")
    print(f"four-period truncated real covariance rank: {truncated_rank}/4 (singular as required)")
    print("P=1, stale IDs, split calibration, ordering, properness and claim-class controls: PASS")


if __name__ == "__main__":
    main()
