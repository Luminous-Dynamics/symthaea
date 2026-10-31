#!/usr/bin/env python3
"""Independent primal/dual verifier for closed-eBH utility LP receipts.

The verifier does not trust solver status, reported objective values, or the
solver's cut-generation trace. It reconstructs each supplied row from its
semantic descriptor, independently separates over every subset cardinality,
and checks a dual-feasible upper bound for the full LP. Omitted intersection
rows have zero dual multiplier, which is valid because the dual of the full
finite LP permits zero weight on any constraint.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


SCHEMA = "symthaea-closed-ebh-primal-dual-v1"


def _digest(e: np.ndarray, utilities: np.ndarray, alpha: float) -> str:
    payload = {
        "evalues": e.tolist(),
        "utilities": utilities.tolist(),
        "alpha": float(alpha),
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _objective_coefficients(u: np.ndarray) -> np.ndarray:
    k = len(u)
    coeff = np.empty(k, dtype=np.float64)
    coeff[0] = u[0]
    for j in range(1, k):
        r = j + 1
        coeff[j] = r * u[j] - (r - 2) * u[j - 1]
    return coeff


def _row_from_spec(
    spec: dict[str, Any], e_sorted: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    k = len(e_sorted)
    row = np.zeros(k, dtype=np.float64)
    kind = spec.get("kind")

    if kind == "monotone":
        j = spec.get("index")
        if not isinstance(j, int) or not 0 <= j < k - 1:
            raise ValueError("invalid monotone row index")
        row[j] = -1.0
        row[j + 1] = 1.0
        return row, 0.0

    if kind == "convex":
        j = spec.get("index")
        if not isinstance(j, int) or not 0 <= j < k:
            raise ValueError("invalid convex row index")
        row[j] = -1.0
        if j + 1 < k:
            row[j + 1] = 2.0
        if j + 2 < k:
            row[j + 2] = -1.0
        return row, 0.0

    if kind == "suffix":
        j = spec.get("index")
        if not isinstance(j, int) or not 0 <= j < k:
            raise ValueError("invalid suffix row index")
        row[j] = 1.0
        suffix_mean = float(np.mean(e_sorted[j:]))
        rhs = alpha * suffix_mean
        if j == 0:
            rhs = min(1.0, rhs)
        return row, rhs

    if kind == "intersection":
        subset = spec.get("subset")
        if not isinstance(subset, list) or not subset:
            raise ValueError("intersection row needs a nonempty subset")
        if any(not isinstance(i, int) or not 0 <= i < k for i in subset):
            raise ValueError("intersection subset index out of range")
        if subset != sorted(set(subset)):
            raise ValueError("intersection subset must be sorted and unique")
        for j in subset:
            row[j] += 1.0
            if j + 1 < k:
                row[j + 1] -= 1.0
        rhs = alpha * float(np.mean(e_sorted[subset]))
        return row, rhs

    raise ValueError(f"unsupported constraint row kind: {kind!r}")


def verify_primal_dual_certificate(
    evalues: Any,
    utilities: Any,
    alpha: float,
    receipt: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Verify validity and LP optimality for one closed-eBH decision receipt.

    A successful result means:
      * the supplied rejection distribution is a valid member of the complete
        closed-eBH menu (the separator checks all subsets implicitly);
      * the supplied dual multipliers define a feasible upper bound for the
        full LP; and
      * the primal/dual gap is within tolerance.

    It does not prove the statistical theorem's assumptions about the input
    e-values; those belong to the separate risk-certificate contract.
    """
    failures: list[str] = []
    e = np.asarray(evalues, dtype=np.float64)
    u0 = np.asarray(utilities, dtype=np.float64)
    if e.ndim != 1 or e.size < 1 or np.any(~np.isfinite(e)) or np.any(e < 0):
        raise ValueError("e-values must be a nonempty finite nonnegative vector")
    if u0.shape != e.shape or np.any(~np.isfinite(u0)) or np.any(u0 < 0):
        raise ValueError("utilities must match e-values and be finite nonnegative")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(receipt, dict) or receipt.get("certificate_schema") != SCHEMA:
        raise ValueError("unsupported or missing certificate schema")

    expected_digest = _digest(e, u0, alpha)
    if receipt.get("input_sha256") != expected_digest:
        failures.append("input digest mismatch")

    k = len(e)
    order = np.argsort(-e, kind="stable")
    e_sorted = e[order]
    u_sorted = u0[order]
    coeff = _objective_coefficients(u_sorted)

    s = np.asarray(receipt.get("primal_vector"), dtype=np.float64)
    if s.shape != (k,) or np.any(~np.isfinite(s)):
        raise ValueError("primal_vector must be a finite length-K vector")
    if np.any(s < -tolerance):
        failures.append("negative primal variable")

    s_ext = np.r_[s, 0.0]
    c = s_ext[:-1] - s_ext[1:]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    q = ranks * s_ext[:-1] - (ranks - 1.0) * s_ext[1:]
    p = np.r_[1.0 - q[0], q[:-1] - q[1:], q[-1]]

    if np.any(np.diff(s_ext, n=2) < -tolerance):
        failures.append("cumulative charges are not convex")
    if np.any(np.diff(s_ext) > tolerance):
        failures.append("cumulative charges are not nonincreasing")
    if np.any(q < -tolerance) or np.any(q > 1.0 + tolerance):
        failures.append("rejection tails outside [0,1]")
    if np.any(np.diff(q) > tolerance):
        failures.append("rejection tails are not nonincreasing")
    if np.any(p < -tolerance) or abs(float(p.sum()) - 1.0) > tolerance:
        failures.append("rejection-count probabilities are invalid")

    reported_q = np.asarray(receipt.get("tails"), dtype=np.float64)
    if reported_q.shape != (k,) or not np.allclose(
        reported_q, q, atol=tolerance, rtol=0.0
    ):
        failures.append("reported tails do not match primal vector")

    # Reconstruct the conditional FDP charges from the rejection-count law.
    # p[0] is P(kappa=0); p[n] is P(kappa=n).
    c_from_p = np.asarray(
        [
            sum(p[n] / n for n in range(r, k + 1))
            for r in range(1, k + 1)
        ],
        dtype=np.float64,
    )
    if not np.allclose(c_from_p, c, atol=tolerance, rtol=0.0):
        failures.append("primal charge vector disagrees with rejection law")

    # Independent all-subset separation: for each size t, the most violated
    # subset consists of the t largest c_r - alpha * e_r / t values.
    max_subset_violation = -math.inf
    worst_subset: list[int] = []
    for t in range(1, k + 1):
        scores = c - alpha * e_sorted / float(t)
        idx = np.argsort(-scores, kind="stable")[:t]
        violation = float(np.sum(scores[idx]))
        if violation > max_subset_violation:
            max_subset_violation = violation
            worst_subset = sorted(int(i) for i in idx)
    if max_subset_violation > tolerance:
        failures.append("closed-eBH intersection constraint violated")

    rows = receipt.get("dual_rows")
    multipliers = np.asarray(receipt.get("dual_multipliers"), dtype=np.float64)
    if not isinstance(rows, list) or multipliers.shape != (len(rows),):
        raise ValueError("dual row descriptors and multipliers have incompatible shapes")
    if np.any(~np.isfinite(multipliers)):
        failures.append("non-finite dual multiplier")
    if np.any(multipliers < -tolerance):
        failures.append("negative dual multiplier")

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    for spec in rows:
        if not isinstance(spec, dict):
            raise ValueError("dual row descriptor must be an object")
        row, rhs = _row_from_spec(spec, e_sorted, alpha)
        A_rows.append(row)
        b_rows.append(rhs)
    A = np.asarray(A_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)

    primal_row_slack = A @ s - b
    max_primal_row_violation = (
        float(np.max(primal_row_slack)) if primal_row_slack.size else -math.inf
    )
    if max_primal_row_violation > tolerance:
        failures.append("primal violates a supplied constraint row")

    # For max c^T s subject to A s <= b and s >= 0, y >= 0 with
    # A^T y >= c is dual-feasible and gives the global upper bound b^T y.
    dual_slack = A.T @ multipliers - coeff
    min_dual_slack = float(np.min(dual_slack))
    if min_dual_slack < -tolerance:
        failures.append("dual is not feasible for the objective")

    primal_objective = float(np.dot(u_sorted, q))
    primal_from_s = float(np.dot(coeff, s))
    dual_objective = float(np.dot(b, multipliers))
    dual_gap = dual_objective - primal_objective
    if abs(primal_objective - primal_from_s) > tolerance:
        failures.append("utility objective disagrees with primal representation")
    if abs(float(receipt.get("primal_objective", math.nan)) - primal_objective) > tolerance:
        failures.append("reported primal objective mismatch")
    if abs(float(receipt.get("dual_objective", math.nan)) - dual_objective) > tolerance:
        failures.append("reported dual objective mismatch")
    if dual_gap < -tolerance:
        failures.append("dual objective is below primal objective")
    if dual_gap > tolerance:
        failures.append("primal/dual optimality gap exceeds tolerance")

    return {
        "valid": not failures,
        "failures": failures,
        "input_sha256": expected_digest,
        "max_subset_violation": max_subset_violation,
        "worst_subset": worst_subset,
        "max_primal_row_violation": max_primal_row_violation,
        "minimum_dual_slack": min_dual_slack,
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": dual_gap,
        "hypotheses": k,
        "dual_rows": len(rows),
    }
