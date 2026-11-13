#!/usr/bin/env python3
"""Independent SCS 3.3.1 cross-check of AIRROAD's analytic PSD smoke cases."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scs


def realify_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return [[Re(H), -Im(H)], [Im(H), Re(H)]] after Hermitian checks."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("input must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("input contains non-finite entries")
    if not np.array_equal(matrix, matrix.conj().T):
        raise ValueError("input is not exactly Hermitian")
    a = matrix.real
    b = matrix.imag
    return np.block([[a, -b], [b, a]])


def svec_lower(matrix: np.ndarray) -> np.ndarray:
    """SCS real-PSD convention: sqrt(2)-scaled lower triangle, columnwise."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("input must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or not np.array_equal(matrix, matrix.T):
        raise ValueError("input must be finite and symmetric")
    values: list[float] = []
    for col in range(matrix.shape[0]):
        for row in range(col, matrix.shape[0]):
            value = float(matrix[row, col])
            if row != col:
                value *= np.sqrt(2.0)
            if not np.isfinite(value):
                raise ValueError("svec produced a non-finite value")
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def mat_from_svec_lower(vector: np.ndarray, order: int) -> np.ndarray:
    """Decode the SCS lower-triangle columnwise representation."""
    expected = order * (order + 1) // 2
    if order <= 0 or vector.ndim != 1 or vector.size != expected:
        raise ValueError("invalid PSD vector dimension")
    if not np.all(np.isfinite(vector)):
        raise ValueError("PSD vector contains non-finite values")
    matrix = np.zeros((order, order), dtype=np.float64)
    index = 0
    for col in range(order):
        for row in range(col, order):
            value = float(vector[index])
            if row != col:
                value /= np.sqrt(2.0)
            matrix[row, col] = value
            matrix[col, row] = value
            index += 1
    return matrix


def trace_cost_svec_lower(order: int) -> np.ndarray:
    """Return c such that c @ svec_lower(X) == trace(X)."""
    if order <= 0:
        raise ValueError("matrix order must be positive")
    length = order * (order + 1) // 2
    cost = np.zeros(length, dtype=np.float64)
    index = 0
    for col in range(order):
        for row in range(col, order):
            if row == col:
                cost[index] = 1.0
            index += 1
    return cost


def solver_options() -> dict:
    return {
        "verbose": False,
        "normalize": True,
        "max_iters": 100_000,
        "eps_abs": 1.0e-7,
        "eps_rel": 1.0e-7,
        "eps_infeas": 1.0e-8,
        "linear_solver": scs.LinearSolver.QDLDL,
    }


def test_feasible_complex_hermitian_realification() -> None:
    h_complex = np.asarray(
        [[2.0 + 0.0j, 1.0 + 2.0j], [1.0 - 2.0j, 5.0 + 0.0j]],
        dtype=np.complex128,
    )
    h_real = realify_hermitian(h_complex)
    order = h_real.shape[0]
    expected_x = svec_lower(h_real)
    variable_count = expected_x.size

    # min trace(X), s.t. X - R(H) >= 0, expressed as A x + s = b.
    a = -sp.eye(variable_count, format="csc", dtype=np.float64)
    b = -expected_x
    c = trace_cost_svec_lower(order)
    assert np.flatnonzero(c).tolist() == [0, 4, 7, 9]
    assert float(c @ svec_lower(np.diag([1.0, 2.0, 3.0, 4.0]))) == 10.0
    p = sp.csc_matrix((variable_count, variable_count), dtype=np.float64)
    cone = {"s": [order]}

    solver = scs.SCS(
        {"P": p, "A": a, "b": b, "c": c},
        cone,
        **solver_options(),
    )
    solution = solver.solve()
    info = solution["info"]
    assert info["status"] == "solved", info
    assert abs(float(info["pobj"]) - 14.0) < 2.0e-5
    assert abs(float(info["dobj"]) - 14.0) < 2.0e-5
    assert float(info["res_pri"]) < 2.0e-6
    assert float(info["res_dual"]) < 2.0e-6

    decoded = mat_from_svec_lower(np.asarray(solution["x"]), order)
    assert np.max(np.abs(decoded - h_real)) < 2.0e-5

    # Confirm upper- and lower-triangle encodings differ in order for n>2,
    # but both represent the same symmetric matrix.
    upper = np.asarray(
        [
            h_real[row, col] * (np.sqrt(2.0) if row != col else 1.0)
            for col in range(order)
            for row in range(col + 1)
        ],
        dtype=np.float64,
    )
    assert not np.array_equal(upper, expected_x)
    print(
        "SCS feasible SDP: PASS; "
        f"pobj={info['pobj']:.9g}, dobj={info['dobj']:.9g}, "
        f"res_pri={info['res_pri']:.3e}, res_dual={info['res_dual']:.3e}"
    )


def test_primal_infeasibility_farkas_witness() -> None:
    # Cone row order in SCS is zero cone first, then PSD cone.
    # x_11 = 0 conflicts with X - I >= 0.
    a = sp.csc_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
    )
    b = np.asarray([0.0, -1.0, 0.0, -1.0], dtype=np.float64)
    c = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
    p = sp.csc_matrix((3, 3), dtype=np.float64)
    cone = {"z": 1, "s": [2]}

    solver = scs.SCS(
        {"P": p, "A": a, "b": b, "c": c},
        cone,
        **solver_options(),
    )
    solution = solver.solve()
    info = solution["info"]
    assert info["status"] == "infeasible", info

    y = np.asarray(solution["y"], dtype=np.float64)
    assert y.size == 4
    psd_dual = mat_from_svec_lower(y[1:], 2)
    eigenvalues = np.linalg.eigvalsh(psd_dual)
    assert float(np.min(eigenvalues)) >= -2.0e-6

    at_y = np.asarray(a.T @ y).reshape(-1)
    b_dot_y = float(b @ y)
    assert np.max(np.abs(at_y)) < 2.0e-6
    assert b_dot_y < -1.0e-6
    print(
        "SCS infeasibility certificate: PASS; "
        f"||A.T y||_inf={np.max(np.abs(at_y)):.3e}, "
        f"b.T y={b_dot_y:.9g}, min_eig(Z)={np.min(eigenvalues):.3e}"
    )


def test_scs_trace_objective_vectorization() -> None:
    for order, expected_indices in (
        (2, [0, 2]),
        (3, [0, 3, 5]),
        (4, [0, 4, 7, 9]),
    ):
        cost = trace_cost_svec_lower(order)
        assert np.flatnonzero(cost).tolist() == expected_indices
        diagonal = np.arange(1.0, order + 1.0)
        packed = svec_lower(np.diag(diagonal))
        assert float(cost @ packed) == float(np.sum(diagonal))


if __name__ == "__main__":
    print(f"Independent cross-solver version: SCS {scs.__version__}")
    test_scs_trace_objective_vectorization()
    test_feasible_complex_hermitian_realification()
    test_primal_infeasibility_farkas_witness()
