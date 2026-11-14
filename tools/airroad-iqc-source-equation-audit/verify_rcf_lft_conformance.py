#!/usr/bin/env python3
"""Exact-rational conformance oracle for AIRROAD sampled-FRF IQC RCF/LFT identities."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-iqc-source-equation-rcf-lft-v0.1.json"
Matrix = list[list[Fraction]]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def decode_matrix(encoded: list[list[dict[str, str]]]) -> Matrix:
    if not encoded or not encoded[0]:
        raise ValueError("matrix cannot be empty")
    width = len(encoded[0])
    if any(len(row) != width for row in encoded):
        raise ValueError("matrix rows must be rectangular")
    return [[q(value) for value in row] for row in encoded]


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must be rectangular")
    return len(matrix), width


def zeros(rows: int, cols: int) -> Matrix:
    return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    result = zeros(size, size)
    for index in range(size):
        result[index][index] = Fraction(1)
    return result


def add(left: Matrix, right: Matrix) -> Matrix:
    if shape(left) != shape(right):
        raise ValueError("matrix addition dimensions disagree")
    return [
        [a + b for a, b in zip(row_left, row_right)]
        for row_left, row_right in zip(left, right)
    ]


def sub(left: Matrix, right: Matrix) -> Matrix:
    if shape(left) != shape(right):
        raise ValueError("matrix subtraction dimensions disagree")
    return [
        [a - b for a, b in zip(row_left, row_right)]
        for row_left, row_right in zip(left, right)
    ]


def scale(factor: Fraction, matrix: Matrix) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_cols = shape(left)
    right_rows, right_cols = shape(right)
    if left_cols != right_rows:
        raise ValueError(
            f"matrix product dimensions disagree: {(left_rows, left_cols)} x "
            f"{(right_rows, right_cols)}"
        )
    return [
        [
            sum(
                (left[row][inner] * right[inner][col]
                 for inner in range(left_cols)),
                Fraction(0),
            )
            for col in range(right_cols)
        ]
        for row in range(left_rows)
    ]


def inverse_2x2(matrix: Matrix) -> Matrix:
    if shape(matrix) != (2, 2):
        raise ValueError("V0.1 inverse supports 2x2 matrices")
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("singular 2x2 matrix")
    return [
        [d / determinant, -b / determinant],
        [-c / determinant, a / determinant],
    ]


def determinant(matrix: Matrix) -> Fraction:
    rows, cols = shape(matrix)
    if rows != cols:
        raise ValueError("determinant requires a square matrix")
    work = [row[:] for row in matrix]
    result = Fraction(1)
    for pivot_index in range(rows):
        pivot_row = next(
            (row for row in range(pivot_index, rows)
             if work[row][pivot_index] != 0),
            None,
        )
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = (
                work[pivot_row],
                work[pivot_index],
            )
            result = -result
        pivot = work[pivot_index][pivot_index]
        result *= pivot
        for row in range(pivot_index + 1, rows):
            factor = work[row][pivot_index] / pivot
            for col in range(pivot_index + 1, rows):
                work[row][col] -= factor * work[pivot_index][col]
    return result


def leading_principal_minors(matrix: Matrix) -> list[Fraction]:
    rows, cols = shape(matrix)
    if rows != cols:
        raise ValueError("principal minors require a square matrix")
    return [
        determinant([row[:size] for row in matrix[:size]])
        for size in range(1, rows + 1)
    ]


def is_positive_definite(matrix: Matrix) -> bool:
    return all(
        minor > 0
        for minor in leading_principal_minors(matrix)
    )


def is_negative_definite(matrix: Matrix) -> bool:
    return all(
        ((-1) ** index) * minor > 0
        for index, minor in enumerate(leading_principal_minors(matrix), start=1)
    )


def transpose(matrix: Matrix) -> Matrix:
    return [list(values) for values in zip(*matrix)]


def block_matrix(blocks: list[list[Matrix]]) -> Matrix:
    if not blocks or not blocks[0]:
        raise ValueError("block matrix cannot be empty")
    row_heights = [shape(blocks[row][0])[0] for row in range(len(blocks))]
    col_widths = [shape(blocks[0][col])[1] for col in range(len(blocks[0]))]
    for row in range(len(blocks)):
        if any(shape(blocks[row][col])[0] != row_heights[row]
               for col in range(len(blocks[row]))):
            raise ValueError("inconsistent block row heights")
    for col in range(len(blocks[0])):
        if any(shape(blocks[row][col])[1] != col_widths[col]
               for row in range(len(blocks))):
            raise ValueError("inconsistent block column widths")

    result: Matrix = []
    for block_row in range(len(blocks)):
        for local_row in range(row_heights[block_row]):
            row_values: list[Fraction] = []
            for block_col in range(len(blocks[block_row])):
                row_values.extend(blocks[block_row][block_col][local_row])
            result.append(row_values)
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decode_fixture(data: dict[str, Any]) -> dict[str, Any]:
    plant = data["generalized_plant"]
    controller = data["controller_rcf"]
    factors = data["plant_lcf"]
    return {
        "Pzw": decode_matrix(plant["P_zw"]),
        "Pzu": decode_matrix(plant["P_zu"]),
        "Pyw": decode_matrix(plant["P_yw"]),
        "Pyu": decode_matrix(plant["P_yu"]),
        "U": decode_matrix(controller["U"]),
        "V": decode_matrix(controller["V"]),
        "X": decode_matrix(controller["X"]),
        "Y": decode_matrix(controller["Y"]),
        "D": decode_matrix(factors["D"]),
        "N": decode_matrix(factors["N"]),
    }


def validate_dimensions(objects: dict[str, Matrix]) -> None:
    pzw = shape(objects["Pzw"])
    pzu = shape(objects["Pzu"])
    pyw = shape(objects["Pyw"])
    pyu = shape(objects["Pyu"])
    u = shape(objects["U"])
    v = shape(objects["V"])
    x = shape(objects["X"])
    y = shape(objects["Y"])
    d = shape(objects["D"])
    n = shape(objects["N"])

    require(pzw == (1, 1), "fixture Pzw must be 1x1")
    require(pzu == (1, 2), "fixture Pzu must be 1x2")
    require(pyw == (2, 1), "fixture Pyw must be 2x1")
    require(pyu == (2, 2), "fixture Pyu must be 2x2")
    require(u == (2, 2), "controller U must be nu x ny")
    require(v == (2, 2), "controller V must be ny x ny")
    require(x == (2, 2), "Bézout X must be ny x ny")
    require(y == (2, 2), "Bézout Y must be ny x nu")
    require(d == (2, 2), "plant left factor D must be ny x ny")
    require(n == (2, 2), "plant left factor N must be ny x nu")


def evaluate_source_identities(objects: dict[str, Matrix]) -> dict[str, Matrix]:
    pzw, pzu, pyw, pyu = (
        objects["Pzw"], objects["Pzu"], objects["Pyw"], objects["Pyu"]
    )
    u, v, x, y, d, n = (
        objects["U"], objects["V"], objects["X"], objects["Y"],
        objects["D"], objects["N"]
    )

    d_inverse = inverse_2x2(d)
    pyu_from_lcf = matmul(d_inverse, n)
    v_inverse = inverse_2x2(v)
    controller_k = matmul(u, v_inverse)
    bezout = add(matmul(x, v), matmul(y, u))
    lam = sub(matmul(d, v), matmul(n, u))
    lam_inverse = inverse_2x2(lam)

    # Source Equation (4): T = Pzw + Pzu K (I - Pyu K)^-1 Pyw.
    loop_inverse = inverse_2x2(sub(identity(2), matmul(pyu, controller_k)))
    direct_t = add(
        pzw,
        matmul(
            matmul(matmul(pzu, controller_k), loop_inverse),
            pyw,
        ),
    )

    # Source Equation (15): T = Pzw + Pzu U Lambda^-1 D Pyw.
    rcf_t = add(
        pzw,
        matmul(
            matmul(matmul(pzu, u), lam_inverse),
            matmul(d, pyw),
        ),
    )

    # Source Equation (16): M = [[I-Lambda, D Pyw], [Pzu U, Pzw]].
    m = block_matrix([
        [sub(identity(2), lam), matmul(d, pyw)],
        [matmul(pzu, u), pzw],
    ])
    m11 = [row[:2] for row in m[:2]]
    m12 = [row[2:] for row in m[:2]]
    m21 = [m[2][:2]]
    m22 = [[m[2][2]]]
    lft_t = add(
        m22,
        matmul(
            matmul(m21, inverse_2x2(sub(identity(2), m11))),
            m12,
        ),
    )

    # Negative controls: common transcription mistakes.
    plus_loop_inverse = inverse_2x2(add(identity(2), matmul(pyu, controller_k)))
    wrong_feedback_sign = add(
        pzw,
        matmul(
            matmul(matmul(pzu, controller_k), plus_loop_inverse),
            pyw,
        ),
    )
    wrong_factor_order = add(
        pzw,
        matmul(
            matmul(matmul(pzu, u), matmul(d, lam_inverse)),
            pyw,
        ),
    )

    return {
        "Pyu_from_lcf": pyu_from_lcf,
        "K": controller_k,
        "Bezout": bezout,
        "Lambda": lam,
        "Lambda_inverse": lam_inverse,
        "T_direct": direct_t,
        "T_rcf": rcf_t,
        "M": m,
        "T_lft": lft_t,
        "wrong_feedback_sign": wrong_feedback_sign,
        "wrong_D_Lambda_order": wrong_factor_order,
    }


def validate_against_expected(
    result: dict[str, Matrix],
    expected: dict[str, Any],
) -> None:
    mapping = {
        "Pyu_from_lcf": "P_yu_from_D_inverse_N",
        "K": "K_U_V_inverse",
        "Bezout": "bezout_XV_plus_YU",
        "Lambda": "Lambda_DV_minus_NU",
        "Lambda_inverse": "Lambda_inverse",
        "T_direct": "direct_feedback_Tzw",
        "T_rcf": "rcf_Tzw",
        "M": "M",
        "T_lft": "lft_identity_Tzw",
        "wrong_feedback_sign": "wrong_feedback_sign_Tzw",
        "wrong_D_Lambda_order": "wrong_D_Lambda_order_Tzw",
    }
    for actual_key, expected_key in mapping.items():
        wanted = decode_matrix(expected[expected_key])
        require(result[actual_key] == wanted, f"known-answer mismatch: {actual_key}")

    require(result["T_direct"] == result["T_rcf"], "direct and RCF closed-loop maps differ")
    require(result["T_direct"] == result["T_lft"], "direct and LFT closed-loop maps differ")
    require(result["wrong_feedback_sign"] != result["T_direct"],
            "wrong feedback sign was not detected")
    require(result["wrong_D_Lambda_order"] != result["T_direct"],
            "wrong D/Lambda inverse order was not detected")


def run_negative_controls(data: dict[str, Any], baseline: dict[str, Matrix]) -> None:
    # Swapping left-factor multiplication order changes the known Pyu block.
    swapped_pyu = matmul(
        decode_matrix(data["plant_lcf"]["N"]),
        inverse_2x2(decode_matrix(data["plant_lcf"]["D"])),
    )
    require(
        swapped_pyu != baseline["Pyu_from_lcf"],
        "left-factor order control did not distinguish D^-1 N from N D^-1",
    )

    # A singular V cannot be an admissible RCF denominator.
    singular_v = [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]]
    try:
        inverse_2x2(singular_v)
    except ValueError:
        pass
    else:
        raise AssertionError("singular RCF V was accepted")

    # Break the Bézout identity and ensure it no longer equals I.
    objects = decode_fixture(data)
    broken_y = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
    broken_bezout = add(matmul(objects["X"], objects["V"]),
                        matmul(broken_y, objects["U"]))
    require(broken_bezout != identity(2), "broken Bézout witness was accepted")

    # Explicit dimension-error controls.
    bad_u = [[Fraction(1)], [Fraction(0)]]
    try:
        add(matmul(objects["Pzu"], bad_u), objects["Pzu"])
    except ValueError:
        pass
    else:
        raise AssertionError("controller U dimension mismatch was accepted")

    # The source identity Tzw = M22 + M21 (I-M11)^-1 M12 is order-sensitive.
    m = baseline["M"]
    m11 = [row[:2] for row in m[:2]]
    m12 = [row[2:] for row in m[:2]]
    m21 = [m[2][:2]]
    m22 = [[m[2][2]]]
    wrong_lft = add(
        m22,
        matmul(
            matmul(
                [[m12[0][0], m12[1][0]]],
                inverse_2x2(sub(identity(2), m11)),
            ),
            [[m21[0][0]], [m21[0][1]]],
        ),
    )
    require(wrong_lft != baseline["T_lft"], "wrong LFT port orientation was accepted")


def verify_schur_complement_case(
    case: dict[str, Any],
    expected_feasible: bool,
) -> dict[str, Any]:
    pi11 = decode_matrix(case["Pi11"])
    pi12 = decode_matrix(case["Pi12"])
    pi22 = decode_matrix(case["Pi22"])
    m_matrix = decode_matrix(case["M"])
    require(shape(pi11) == (2, 2), "Pi11 must be 2x2 in V0.1")
    require(shape(pi12) == (2, 2), "Pi12 must be 2x2 in V0.1")
    require(shape(pi22) == (2, 2), "Pi22 must be 2x2 in V0.1")
    require(shape(m_matrix) == (2, 2), "M must be 2x2 in V0.1")
    require(pi11[0][1] == pi11[1][0], "Pi11 must be symmetric")
    require(pi22[0][1] == pi22[1][0], "Pi22 must be symmetric")
    require(pi12[0][1] == pi12[1][0], "Pi12 must be symmetric in this real fixture")
    require(is_positive_definite(pi11), "Pi11 must be strictly positive definite")

    pi11_inverse = inverse_2x2(pi11)
    m_transpose = transpose(m_matrix)
    s_matrix = add(
        add(pi22, matmul(m_transpose, pi12)),
        matmul(transpose(pi12), m_matrix),
    )
    reduced = add(
        s_matrix,
        matmul(matmul(m_transpose, pi11), m_matrix),
    )
    q_matrix = block_matrix([
        [s_matrix, m_transpose],
        [m_matrix, scale(Fraction(-1), pi11_inverse)],
    ])

    expected = case["expected"]
    require(
        leading_principal_minors(pi11)
        == [q(value) for value in expected["Pi11_leading_principal_minors"]]
        if "Pi11_leading_principal_minors" in expected
        else is_positive_definite(pi11),
        "Pi11 principal-minor check failed",
    )
    require(s_matrix == decode_matrix(expected["S"]), "Schur block S mismatch")
    require(
        reduced == decode_matrix(expected["reduced_schur"]),
        "Schur-complement reduced matrix mismatch",
    )
    require(q_matrix == decode_matrix(expected["Q"]), "full block Q mismatch")

    q_minors = leading_principal_minors(q_matrix)
    require(
        q_minors == [q(value) for value in expected["Q_leading_principal_minors"]],
        "full block Q principal-minor witness mismatch",
    )
    q_is_negative = is_negative_definite(q_matrix)
    reduced_is_negative = is_negative_definite(reduced)
    require(q_is_negative is expected["Q_negative_definite"],
            "full block Q definiteness status mismatch")
    require(
        reduced_is_negative is expected["reduced_schur_negative_definite"],
        "reduced Schur definiteness status mismatch",
    )
    require(q_is_negative is expected_feasible,
            "source LMI feasibility classification mismatch")
    require(reduced_is_negative is expected_feasible,
            "Schur-reduced feasibility classification mismatch")

    q_det = determinant(q_matrix)
    if "Q_determinant" in expected:
        require(q_det == q(expected["Q_determinant"]), "full Q determinant mismatch")

    wrong_sign_reduced = sub(
        s_matrix,
        matmul(matmul(m_transpose, pi11), m_matrix),
    )
    if "wrong_sign_reduced_schur" in expected:
        require(
            wrong_sign_reduced == decode_matrix(expected["wrong_sign_reduced_schur"]),
            "wrong-sign Schur control mismatch",
        )
        require(
            is_negative_definite(wrong_sign_reduced)
            is expected["wrong_sign_would_misclassify_as_feasible"],
            "wrong-sign control did not expose false feasibility",
        )

    return {
        "S": s_matrix,
        "reduced": reduced,
        "Q": q_matrix,
        "Q_minors": q_minors,
        "Q_determinant": q_det,
        "Q_negative_definite": q_is_negative,
        "reduced_negative_definite": reduced_is_negative,
        "wrong_sign_reduced": wrong_sign_reduced,
    }


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require(
        data["schema"] == "luminous.airroad.iqc_source_equation_audit.v0_1",
        "unsupported source-equation fixture schema",
    )
    require(data["schema_version"] == "0.1.0", "unsupported version")
    objects = decode_fixture(data)
    validate_dimensions(objects)
    result = evaluate_source_identities(objects)
    validate_against_expected(result, data["expected"])
    run_negative_controls(data, result)

    schur = data["schur_complement_conformance"]
    feasible_schur = verify_schur_complement_case(
        schur["feasible_case"],
        expected_feasible=True,
    )
    infeasible_schur = verify_schur_complement_case(
        schur["infeasible_case"],
        expected_feasible=False,
    )
    require(
        infeasible_schur["wrong_sign_reduced"] != infeasible_schur["reduced"],
        "Schur sign negative control did not change the result",
    )

    print("AIRROAD IQC source equation RCF/LFT conformance: PASS")
    print(f"P_yu = D^-1 N: {result['Pyu_from_lcf']}")
    print(f"K = U V^-1: {result['K']}")
    print(f"X V + Y U: {result['Bezout']}")
    print(f"Lambda = D V - N U: {result['Lambda']}")
    print(f"direct Tzw: {result['T_direct']}")
    print(f"RCF Tzw: {result['T_rcf']}")
    print(f"LFT Tzw: {result['T_lft']}")
    print(f"M: {result['M']}")
    print("factor-order, feedback-sign, singular-RCF, Bézout and LFT-order controls: PASS")
    print(f"feasible Schur-reduced matrix: {feasible_schur['reduced']}")
    print(f"feasible full-Q leading minors: {feasible_schur['Q_minors']}")
    print(f"infeasible Schur-reduced matrix: {infeasible_schur['reduced']}")
    print(f"wrong-sign control would appear negative definite: "
          f"{is_negative_definite(infeasible_schur['wrong_sign_reduced'])}")
    print("Schur complement sign, inverse and strict-negative controls: PASS")
    print("full paper generator construction / SDP / robust stability: NOT IMPLEMENTED")


if __name__ == "__main__":
    main()
