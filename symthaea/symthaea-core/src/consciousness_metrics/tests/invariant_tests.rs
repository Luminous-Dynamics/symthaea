// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mathematical invariant tests.
//!
//! Property-based tests for fundamental IIT/entropy properties.

use super::*;

/// Invariant: Entropy is always non-negative
#[test]
fn test_invariant_entropy_non_negative() {
    let est = ContinuousEntropyEstimator::default();

    for seed in 0..10 {
        let hv = ContinuousHV::random(HDC_DIMENSION, seed);
        let h = est.entropy(&hv);
        assert!(
            h >= 0.0,
            "Entropy must be non-negative: H = {:.6} for seed {}",
            h,
            seed
        );
    }
}

/// Invariant: Mutual information is symmetric I(X;Y) = I(Y;X)
#[test]
fn test_invariant_mi_symmetric() {
    let est = ContinuousEntropyEstimator::default();

    for seed in 0..5 {
        let a = ContinuousHV::random(HDC_DIMENSION, seed * 2);
        let b = ContinuousHV::random(HDC_DIMENSION, seed * 2 + 1);

        let mi_ab = est.mutual_information_fast(&a, &b);
        let mi_ba = est.mutual_information_fast(&b, &a);

        let diff = (mi_ab - mi_ba).abs();
        assert!(
            diff < 1e-10,
            "MI should be symmetric: I(A;B)={:.6}, I(B;A)={:.6}",
            mi_ab,
            mi_ba
        );
    }
}

/// Invariant: Φ is always non-negative
#[test]
fn test_invariant_phi_non_negative() {
    let calc = TruePhiCalculator::new();

    for seed in 0..5 {
        let components: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, seed * 100 + i))
            .collect();

        let result = calc.compute_true_phi(&components);
        assert!(
            result.phi >= 0.0,
            "\u{03a6} must be non-negative: {:.6} for seed {}",
            result.phi,
            seed
        );
    }
}

/// Invariant: MIP partitions all elements
#[test]
fn test_invariant_partition_complete() {
    let calc = TruePhiCalculator::new();

    for n in 2..=6 {
        let components: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        let result = calc.compute_true_phi(&components);
        let total = result.mip.part_a.len() + result.mip.part_b.len();

        assert_eq!(
            total, n,
            "MIP should partition all {} elements, got {}",
            n, total
        );
        assert!(
            !result.mip.part_a.is_empty(),
            "MIP part A should not be empty for n={}",
            n
        );
        assert!(
            !result.mip.part_b.is_empty(),
            "MIP part B should not be empty for n={}",
            n
        );
    }
}

/// Invariant: System EI >= MIP EI (definition of integration)
#[test]
fn test_invariant_system_ei_geq_mip_ei() {
    let calc = TruePhiCalculator::new();

    for seed in 0..5 {
        let base = ContinuousHV::random(HDC_DIMENSION, seed * 1000);
        let components: Vec<ContinuousHV> = (0..4)
            .map(|i| {
                ContinuousHV::weighted_bundle(
                    &[
                        &base,
                        &ContinuousHV::random(HDC_DIMENSION, seed * 1000 + 100 + i),
                    ],
                    &[0.8, 0.2],
                )
            })
            .collect();

        let result = calc.compute_true_phi(&components);

        // Φ = system_ei - mip_ei, so system_ei should be >= mip_ei
        // (allowing small numerical tolerance)
        assert!(
            result.system_ei >= result.mip_ei - 1e-10,
            "System EI ({:.6}) should be >= MIP EI ({:.6})",
            result.system_ei,
            result.mip_ei
        );
    }
}

/// Invariant: Binding is approximately reversible
/// a * b * b ~ a (with some noise due to continuous values)
#[test]
fn test_invariant_binding_reversibility() {
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);

    let bound = a.bind(&b);
    let recovered = bound.bind(&b); // Unbind by binding again with same key

    let sim = a.similarity(&recovered);

    // Should have high similarity (binding is self-inverse)
    assert!(
        sim > 0.5,
        "Binding should be approximately reversible: similarity = {:.4}",
        sim
    );
}
