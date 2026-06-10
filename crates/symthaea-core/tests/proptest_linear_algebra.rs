// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for the hyperdimensional linear algebra engine.
//!
//! Tests mathematical invariants that must hold for ALL valid matrices,
//! not just hand-picked examples. Catches numerical edge cases.

use proptest::prelude::*;
use symthaea_core::hdc::linear_algebra::{HdcMatrix, HdcVector};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn arb_small_matrix(n: usize) -> impl Strategy<Value = HdcMatrix> {
    proptest::collection::vec(-10.0..=10.0f64, n * n)
        .prop_map(move |data| HdcMatrix::new(data, n, n))
}

fn arb_vector(n: usize) -> impl Strategy<Value = HdcVector> {
    proptest::collection::vec(-10.0..=10.0f64, n).prop_map(|data| HdcVector::new(data))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matrix arithmetic properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// A + B = B + A (addition commutativity)
    #[test]
    fn matrix_addition_commutative(
        a in arb_small_matrix(3),
        b in arb_small_matrix(3),
    ) {
        let (ab, _) = a.add(&b);
        let (ba, _) = b.add(&a);
        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    (ab.get(i, j) - ba.get(i, j)).abs() < 1e-10,
                    "A+B should equal B+A at ({i},{j})"
                );
            }
        }
    }

    /// A + 0 = A (additive identity)
    #[test]
    fn matrix_additive_identity(a in arb_small_matrix(3)) {
        let zero = HdcMatrix::zeros(3, 3);
        let (result, _) = a.add(&zero);
        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    (result.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "A+0 should equal A"
                );
            }
        }
    }

    /// (A^T)^T = A (transpose involution)
    #[test]
    fn transpose_involution(a in arb_small_matrix(3)) {
        let att = a.transpose().transpose();
        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    (att.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "Double transpose should be identity"
                );
            }
        }
    }

    /// tr(A + B) = tr(A) + tr(B) (trace linearity)
    #[test]
    fn trace_linearity(
        a in arb_small_matrix(3),
        b in arb_small_matrix(3),
    ) {
        let (sum, _) = a.add(&b);
        let trace_sum = sum.trace();
        let sum_traces = a.trace() + b.trace();
        prop_assert!(
            (trace_sum - sum_traces).abs() < 1e-8,
            "tr(A+B) should equal tr(A) + tr(B): {} vs {}",
            trace_sum, sum_traces
        );
    }

    /// I * A = A (identity multiplication)
    #[test]
    fn identity_multiplication(a in arb_small_matrix(3)) {
        let eye = HdcMatrix::identity(3);
        let (result, _) = eye.mul(&a);
        for i in 0..3 {
            for j in 0..3 {
                prop_assert!(
                    (result.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "I*A should equal A at ({i},{j}): {} vs {}",
                    result.get(i, j), a.get(i, j)
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Determinant properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// det(I) = 1
    #[test]
    fn determinant_of_identity(n in 2..=5usize) {
        let eye = HdcMatrix::identity(n);
        let (det, _) = eye.determinant();
        prop_assert!(
            (det - 1.0).abs() < 1e-10,
            "det(I) should be 1.0, got {det}"
        );
    }

    /// det(A^T) = det(A)
    #[test]
    fn determinant_transpose_invariant(a in arb_small_matrix(3)) {
        let (det_a, _) = a.determinant();
        let (det_at, _) = a.transpose().determinant();
        if det_a.is_finite() && det_at.is_finite() {
            prop_assert!(
                (det_a - det_at).abs() < 1e-6,
                "det(A) should equal det(A^T): {} vs {}",
                det_a, det_at
            );
        }
    }

    /// det(cA) = c^n * det(A) for n×n matrix
    #[test]
    fn determinant_scalar_multiple(
        a in arb_small_matrix(3),
        c in -5.0..=5.0f64,
    ) {
        let scaled = a.scale(c);
        let (det_a, _) = a.determinant();
        let (det_scaled, _) = scaled.determinant();
        let expected = c.powi(3) * det_a;
        if det_a.is_finite() && det_scaled.is_finite() && expected.is_finite() {
            let tol = expected.abs().max(1.0) * 1e-6;
            prop_assert!(
                (det_scaled - expected).abs() < tol,
                "det(cA) = c^3 * det(A): det(cA)={det_scaled:.6}, c^3*det(A)={expected:.6}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vector properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Dot product commutativity: a·b = b·a
    #[test]
    fn dot_product_commutative(
        a in arb_vector(5),
        b in arb_vector(5),
    ) {
        let ab = a.dot(&b);
        let ba = b.dot(&a);
        prop_assert!(
            (ab - ba).abs() < 1e-10,
            "a·b should equal b·a"
        );
    }

    /// Norm is non-negative
    #[test]
    fn norm_non_negative(a in arb_vector(5)) {
        prop_assert!(a.norm() >= 0.0, "Norm should be non-negative");
    }

    /// Normalized vector has unit norm (unless zero)
    #[test]
    fn normalized_has_unit_norm(a in arb_vector(5)) {
        if a.norm() > 1e-10 {
            let n = a.normalized();
            prop_assert!(
                (n.norm() - 1.0).abs() < 1e-8,
                "Normalized vector should have unit norm: {:.6}",
                n.norm()
            );
        }
    }

    /// Cauchy-Schwarz: |a·b| <= ||a|| * ||b||
    #[test]
    fn cauchy_schwarz(
        a in arb_vector(5),
        b in arb_vector(5),
    ) {
        let dot = a.dot(&b).abs();
        let product = a.norm() * b.norm();
        prop_assert!(
            dot <= product + 1e-8,
            "Cauchy-Schwarz violated: |a·b|={dot:.6} > ||a||·||b||={product:.6}"
        );
    }
}
