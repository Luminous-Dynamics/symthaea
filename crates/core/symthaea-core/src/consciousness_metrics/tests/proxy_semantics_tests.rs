// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC proxy semantics tests.
//!
//! These tests exercise the behavior of `TruePhiCalculator` on synthetic
//! hypervector representations. They do not invoke PyPhi, consume an external
//! oracle, construct an explicit state-indexed transition probability model,
//! or validate IIT 3.0 / IIT 4.0. The legacy `phi` field is treated here only
//! as the output of Symthaea's pairwise-information partition proxy.

use super::*;

/// Unrelated random representations should have a small pairwise-information proxy.
///
/// This tests representation-level behavior only; random seeds do not define
/// a causal-independence model and the assertion is not an IIT oracle check.
#[test]
fn unrelated_random_representations_have_near_zero_pairwise_mi_proxy() {
    let calc = TruePhiCalculator::new();

    // Create unrelated random representations. This is not a causal model.
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 1000000); // Very different seed

    let result = calc.compute_true_phi(&[a, b]);

    // For unrelated random representations, the binned pairwise-MI proxy should be small.
    assert!(
        result.phi < 0.05,
        "Unrelated random representations should have near-zero \u{03a6}: {:.6}",
        result.phi
    );

    println!(
        "HDC proxy - unrelated 2-vector case: \u{03a6} = {:.6} (expected \u{2248} 0)",
        result.phi
    );
}

/// Shared-base representations should have positive pairwise-information proxy.
///
/// The assertion is about the estimator's output on correlated representations,
/// not about integrated information in an explicit causal system.
#[test]
fn shared_base_representations_have_positive_pairwise_mi_proxy() {
    let calc = TruePhiCalculator::new();

    // Create strongly correlated vector representations from a shared base.
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let a = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 100)],
        &[0.95, 0.05],
    );
    let b = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.95, 0.05],
    );

    let result = calc.compute_true_phi(&[a, b]);

    // Shared-base representations should have positive Φ
    assert!(
        result.phi > 0.0,
        "Correlated elements should have positive \u{03a6}: {:.6}",
        result.phi
    );

    // System EI should be positive (there's mutual information)
    assert!(
        result.system_ei > 0.0,
        "System EI should be positive: {:.6}",
        result.system_ei
    );

    println!(
        "HDC proxy - shared-base 2-vector case: \u{03a6} = {:.6}, EI = {:.6}",
        result.phi, result.system_ei
    );
}

/// HDC binding should remain a valid finite input to the proxy.
///
/// `bind` is an HDC representation operation here, not an XOR causal mechanism
/// or a substitute for a binary transition probability matrix.
#[test]
fn hdc_binding_operation_produces_valid_proxy_outputs() {
    let calc = TruePhiCalculator::new();

    // Create an HDC binding representation; no causal mechanism is defined.
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let c = a.bind(&b); // XOR-like: c encodes relationship between a and b

    // System with just a and b
    let phi_ab = calc.compute_true_phi(&[a.clone(), b.clone()]);

    // System with a, b, and their bound output
    let phi_abc = calc.compute_true_phi(&[a, b, c]);

    println!("HDC proxy smoke test - binding representation:");
    println!("  \u{03a6}(a,b) = {:.6}", phi_ab.phi);
    println!("  \u{03a6}(a,b,c) = {:.6}", phi_abc.phi);

    // This is a smoke test only: no IIT interpretation or expected ranking.
    assert!(phi_ab.phi >= 0.0 && phi_abc.phi >= 0.0);
}

/// Copy-like and binding representations produce finite nonnegative proxy values.
///
/// This does not assert a causal ordering between copy and XOR mechanisms.
#[test]
fn binding_and_copy_like_representations_have_finite_nonnegative_proxy() {
    let calc = TruePhiCalculator::new();

    // HDC binding representation (not a modeled XOR gate).
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let xor = a.bind(&b);

    // Copy-like noisy representation (not an explicit identity transition).
    let copy = ContinuousHV::weighted_bundle(
        &[&a, &ContinuousHV::random(HDC_DIMENSION, 3)],
        &[0.99, 0.01],
    );

    let phi_xor = calc.compute_true_phi(&[a.clone(), b, xor]);
    let phi_copy = calc.compute_true_phi(&[a, copy]);

    println!("HDC proxy smoke test - binding vs copy-like representations:");
    println!("  \u{03a6}(XOR) = {:.6}", phi_xor.phi);
    println!("  \u{03a6}(Copy) = {:.6}", phi_copy.phi);

    // Both should be valid computations
    assert!(phi_xor.phi >= 0.0 && phi_copy.phi >= 0.0);
}

/// The summed pairwise-MI estimator grows on a shared-base ensemble.
///
/// This is an estimator/size diagnostic, not a universal claim about IIT Phi.
#[test]
fn shared_base_pairwise_information_grows_with_ensemble_size() {
    let calc = TruePhiCalculator::new();
    let base = ContinuousHV::random(HDC_DIMENSION, 42);

    let create_correlated = |n: usize| -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| {
                ContinuousHV::weighted_bundle(
                    &[&base, &ContinuousHV::random(HDC_DIMENSION, 100 + i as u64)],
                    &[0.8, 0.2],
                )
            })
            .collect()
    };

    let phi_2 = calc.compute_true_phi(&create_correlated(2));
    let phi_3 = calc.compute_true_phi(&create_correlated(3));
    let phi_4 = calc.compute_true_phi(&create_correlated(4));

    println!("HDC pairwise-information proxy - ensemble-size diagnostic:");
    println!(
        "  \u{03a6}(n=2) = {:.6}, EI = {:.6}",
        phi_2.phi, phi_2.system_ei
    );
    println!(
        "  \u{03a6}(n=3) = {:.6}, EI = {:.6}",
        phi_3.phi, phi_3.system_ei
    );
    println!(
        "  \u{03a6}(n=4) = {:.6}, EI = {:.6}",
        phi_4.phi, phi_4.system_ei
    );

    // This estimator sums pairwise MI, so the shared-base ensemble should grow here.
    assert!(
        phi_4.system_ei > phi_2.system_ei,
        "Larger systems should have more total information"
    );
}
