#!/usr/bin/env python3
"""Exact standard-library verifier for the synthetic BH-EGB covariance fixture.

Research regression only: not production physics, a calibrated probability model,
or an experiment-design policy. All arithmetic uses fractions.Fraction.
"""
import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path


def v(xs):
    if not isinstance(xs, list) or not xs:
        raise ValueError("expected non-empty vector")
    return [F(x) for x in xs]


def m(xs):
    rows = [v(row) for row in xs]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("invalid/ragged matrix")
    return rows


def tr(a):
    return [list(col) for col in zip(*a)]


def mm(a, b):
    if len(a[0]) != len(b):
        raise ValueError("matrix shape mismatch")
    bt = tr(b)
    return [[sum((x*y for x, y in zip(r, c)), F(0)) for c in bt] for r in a]


def mv(a, x):
    if len(a[0]) != len(x):
        raise ValueError("matrix/vector shape mismatch")
    return [sum((x_i*y_i for x_i, y_i in zip(r, x)), F(0)) for r in a]


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), F(0))


def det(a):
    n = len(a)
    if not n or any(len(r) != n for r in a):
        raise ValueError("determinant requires square matrix")
    a = [r[:] for r in a]
    sign, out = F(1), F(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c]), None)
        if p is None:
            return F(0)
        if p != c:
            a[c], a[p], sign = a[p], a[c], -sign
        z = a[c][c]
        out *= z
        for r in range(c+1, n):
            f = a[r][c] / z
            for j in range(c+1, n):
                a[r][j] -= f*a[c][j]
            a[r][c] = F(0)
    return sign*out


def inv(a):
    n = len(a)
    if not n or any(len(r) != n for r in a):
        raise ValueError("inverse requires square matrix")
    z = [r[:] + [F(i == j) for j in range(n)] for i, r in enumerate(a)]
    for c in range(n):
        p = next((r for r in range(c, n) if z[r][c]), None)
        if p is None:
            raise ValueError("singular matrix")
        z[c], z[p] = z[p], z[c]
        q = z[c][c]
        z[c] = [x/q for x in z[c]]
        for r in range(n):
            if r != c:
                q = z[r][c]
                z[r] = [x-q*y for x, y in zip(z[r], z[c])]
    return [r[n:] for r in z]


def rank(a):
    z, nr, nc, rnk = [r[:] for r in a], len(a), len(a[0]), 0
    for c in range(nc):
        p = next((r for r in range(rnk, nr) if z[r][c]), None)
        if p is None:
            continue
        z[rnk], z[p] = z[p], z[rnk]
        q = z[rnk][c]
        z[rnk] = [x/q for x in z[rnk]]
        for r in range(nr):
            if r != rnk:
                q = z[r][c]
                z[r] = [x-q*y for x, y in zip(z[r], z[rnk])]
        rnk += 1
        if rnk == nr:
            break
    return rnk


def same(a, b, label):
    if a != b:
        raise AssertionError(f"{label}: expected {b}, got {a}")


def verify_invertible_transport(base_delta, base_covariance, payload, label):
    transform = m(payload["transformation_matrix"])
    transformed_delta = v(payload["transformed_mean_difference"])
    transformed_covariance = m(payload["transformed_difference_covariance"])
    n = len(base_delta)
    if len(transform) != n or any(len(row) != n for row in transform):
        raise ValueError(f"{label} transform must be square")
    same(mv(transform, base_delta), transformed_delta, f"{label} transformed mean")
    same(mm(mm(transform, base_covariance), tr(transform)), transformed_covariance,
         f"{label} covariance transport")
    inverse_transform = inv(transform)
    same(det(transform), F(payload["determinant"]), f"{label} transform determinant")
    transformed_precision = mm(mm(tr(inverse_transform), inv(base_covariance)), inverse_transform)
    transformed_score = dot(transformed_delta, mv(transformed_precision, transformed_delta))
    base_score = dot(base_delta, mv(inv(base_covariance), base_delta))
    same(transformed_score, F(payload["expected_mahalanobis_squared"]),
         f"{label} expected Mahalanobis squared")
    same(transformed_score, base_score, f"{label} Mahalanobis invariance")
    return transformed_score


def verify(p):
    h0 = v(p["competing_hypotheses"]["H0"]["predicted_mean"])
    h1 = v(p["competing_hypotheses"]["H1"]["predicted_mean"])
    if len(h0) != len(h1):
        raise ValueError("hypothesis dimensions differ")
    d = [x-y for x, y in zip(h1, h0)]
    s, l = m(p["predicted_difference_covariance"]), m(p["redundant_observable_transform"]["transformation_matrix"])
    da = v(p["redundant_observable_transform"]["augmented_predicted_difference"])
    sa = m(p["redundant_observable_transform"]["augmented_predicted_difference_covariance"])
    n, k = len(d), len(da)
    if len(s) != n or any(len(r) != n for r in s) or len(l) != k or any(len(r) != n for r in l):
        raise ValueError("base/transform dimensions inconsistent")
    if len(sa) != k or any(len(r) != k for r in sa) or tr(sa) != sa or tr(s) != s:
        raise ValueError("covariance shape or symmetry invalid")
    if any(det([r[:i] for r in s[:i]]) <= 0 for i in range(1, n+1)):
        raise ValueError("base covariance is not positive definite")
    same(mv(l, d), da, "transformed mean")
    same(mm(mm(l, s), tr(l)), sa, "covariance transport")
    r = rank(l)
    same(r, n, "full-column-rank transform")
    same(p["redundant_observable_transform"]["rank"], r, "declared covariance rank")
    if r >= k:
        raise ValueError("fixture lacks redundant output")
    same(rank(sa), r, "augmented covariance rank")
    same(det(sa), F(0), "augmented covariance singularity")

    # For A=L S L^T, with L full-column-rank and S positive definite:
    # A+ = L^{+T} S^-1 L+, where L+=(L^T L)^-1 L^T.
    lt = tr(l)
    lp = mm(inv(mm(lt, l)), lt)
    d_base = mv(lp, da)
    same(d_base, d, "recovered base coordinates")
    same(mv(l, d_base), da, "augmented mean lies in covariance support")
    precision = inv(s)
    ap = mm(mm(tr(lp), precision), lp)
    same(mm(mm(sa, ap), sa), sa, "Moore-Penrose A A+ A")
    same(mm(mm(ap, sa), ap), ap, "Moore-Penrose A+ A A+")
    same(mm(sa, ap), tr(mm(sa, ap)), "Moore-Penrose A A+ symmetry")
    same(mm(ap, sa), tr(mm(ap, sa)), "Moore-Penrose A+ A symmetry")
    q0, q1 = dot(d, mv(precision, d)), dot(da, mv(ap, da))
    expected = p["expected"]
    same(q0, F(expected["covariance_aware_mahalanobis_squared"]), "base Mahalanobis squared")
    same(q1, F(p["redundant_observable_transform"]["expected"]["covariance_aware_pseudoinverse_mahalanobis_squared"]), "augmented Mahalanobis squared")
    same(q1, q0, "redundancy invariance")
    basis_q = verify_invertible_transport(d, s, p["invertible_basis_transport"], "basis")
    units_q = verify_invertible_transport(d, s, p["unit_rescaling"], "unit rescaling")
    same(basis_q, q0, "basis transport preserves base score")
    same(units_q, q0, "unit rescaling preserves base score")
    same(q0, F(4), "exact known answer")
    same(F(expected["covariance_aware_mahalanobis_distance"])**2, q0, "declared distance")
    same(m(p["exact_derivation"]["inverse_base_covariance"]), precision, "declared inverse")
    same(v(p["exact_derivation"]["inverse_times_delta"]), mv(precision, d), "inverse times delta")
    same(F(p["exact_derivation"]["quadratic_form"]), q0, "declared quadratic form")

    def diagonal_score(delta, cov):
        if any(cov[i][i] <= 0 for i in range(len(delta))):
            raise ValueError("non-positive diagonal variance")
        return sum((delta[i]**2/cov[i][i] for i in range(len(delta))), F(0))

    before, after = diagonal_score(d, s), diagonal_score(da, sa)
    same(before, F(expected["diagonal_only_independence_score_squared_without_redundancy"]), "naive score before")
    same(after, F(p["redundant_observable_transform"]["expected"]["naive_diagonal_independence_score_squared_if_duplicate_counted_as_new"]), "naive score after")
    same((before, after), (F(5), F(8)), "naive independence false gain")

    rejected = []
    def must_reject(candidate, label):
        try:
            verify(candidate)
        except (ValueError, AssertionError, KeyError, TypeError):
            rejected.append(label)
            return
        raise AssertionError(f"negative control accepted: {label}")

    bad = copy.deepcopy(p)
    bad["predicted_difference_covariance"] = [["1", "2"], ["2", "1"]]
    must_reject(bad, "indefinite_base_covariance")
    bad = copy.deepcopy(p)
    bad["redundant_observable_transform"]["augmented_predicted_difference"][2] = "4"
    must_reject(bad, "mean_outside_declared_support")
    bad = copy.deepcopy(p)
    bad["redundant_observable_transform"]["augmented_predicted_difference_covariance"][0][2] = "2"
    must_reject(bad, "covariance_transport_mismatch")
    bad = copy.deepcopy(p)
    bad["redundant_observable_transform"]["rank"] = 1
    must_reject(bad, "false_rank_declaration")

    return {
        "status": "PASS_EXACT_SYNTHETIC_FIXTURE_ONLY",
        "base_mahalanobis_squared": str(q0),
        "augmented_singular_covariance_pseudoinverse_squared": str(q1),
        "augmented_covariance_rank": r,
        "augmented_output_count": k,
        "redundancy_invariance": True,
        "basis_metamorphic_invariance": True,
        "unit_rescaling_invariance": True,
        "naive_diagonal_score_before": str(before),
        "naive_diagonal_score_after": str(after),
        "negative_controls_passed": len(rejected),
        "negative_controls": rejected,
        "fixture_scope": p["scope"],
        "claim_ceiling": p["claim_ceiling"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", nargs="?", type=Path, default=Path(__file__).with_name("BH_EGB_COVARIANCE_REDUNDANCY_CONTROL_V0_1.json"))
    args = ap.parse_args()
    try:
        p = json.loads(args.fixture.read_text(encoding="utf-8"))
        if not isinstance(p, dict):
            raise ValueError("fixture root must be a JSON object")
        report = verify(p)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
