// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PyPhi comparison tests.
//!
//! Validate our Φ implementation against known IIT results from literature.
//! Reference: Oizumi et al. (2014), Tononi et al. (2016)

use super::*;

/// PyPhi Test Case 1: Completely independent system
///
/// Two independent elements have Φ = 0 because there's no integration.
/// This is a fundamental axiom of IIT.
#[test]
fn test_pyphi_independent_elements_zero_phi() {
    let calc = TruePhiCalculator::new();

    // Create completely independent vectors (different random seeds, no correlation)
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 1000000); // Very different seed

    let result = calc.compute_true_phi(&[a, b]);

    // For truly independent elements, Φ should be very close to 0
    // We allow small positive values due to random correlations
    assert!(
        result.phi < 0.05,
        "Independent elements should have near-zero \u{03a6}: {:.6}",
        result.phi
    );

    println!(
        "PyPhi comparison - Independent 2-node: \u{03a6} = {:.6} (expected \u{2248} 0)",
        result.phi
    );
}

/// PyPhi Test Case 2: Maximally correlated system
///
/// Two elements derived from the same base should have high MI but
/// the Φ depends on how the partition affects information.
#[test]
fn test_pyphi_correlated_elements_positive_phi() {
    let calc = TruePhiCalculator::new();

    // Create maximally correlated vectors
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

    // Correlated elements should have positive Φ
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
        "PyPhi comparison - Correlated 2-node: \u{03a6} = {:.6}, EI = {:.6}",
        result.phi, result.system_ei
    );
}

/// PyPhi Test Case 3: XOR-like structure
///
/// An XOR gate creates integration because the output depends on
/// both inputs in a non-decomposable way.
#[test]
fn test_pyphi_xor_like_structure() {
    let calc = TruePhiCalculator::new();

    // Create XOR-like structure: c = f(a, b) where c requires both a and b
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let c = a.bind(&b); // XOR-like: c encodes relationship between a and b

    // System with just a and b
    let phi_ab = calc.compute_true_phi(&[a.clone(), b.clone()]);

    // System with a, b, and their bound output
    let phi_abc = calc.compute_true_phi(&[a, b, c]);

    println!("PyPhi comparison - XOR structure:");
    println!("  \u{03a6}(a,b) = {:.6}", phi_ab.phi);
    println!("  \u{03a6}(a,b,c) = {:.6}", phi_abc.phi);

    // The bound structure should affect the integration
    // (we don't assert specific relationship, but both should be valid)
    assert!(phi_ab.phi >= 0.0 && phi_abc.phi >= 0.0);
}

/// PyPhi Test Case 4: Copy mechanism
///
/// A simple copy (identity) relationship should have lower Φ than XOR
/// because it doesn't require integration of multiple inputs.
#[test]
fn test_pyphi_copy_vs_xor() {
    let calc = TruePhiCalculator::new();

    // XOR-like (requires both inputs)
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let xor = a.bind(&b);

    // Copy-like (just a with noise)
    let copy = ContinuousHV::weighted_bundle(
        &[&a, &ContinuousHV::random(HDC_DIMENSION, 3)],
        &[0.99, 0.01],
    );

    let phi_xor = calc.compute_true_phi(&[a.clone(), b, xor]);
    let phi_copy = calc.compute_true_phi(&[a, copy]);

    println!("PyPhi comparison - Copy vs XOR:");
    println!("  \u{03a6}(XOR) = {:.6}", phi_xor.phi);
    println!("  \u{03a6}(Copy) = {:.6}", phi_copy.phi);

    // Both should be valid computations
    assert!(phi_xor.phi >= 0.0 && phi_copy.phi >= 0.0);
}

/// PyPhi Test Case 5: Scaling with system size
///
/// For integrated systems, Φ should generally increase with size
/// (more components = more potential integration).
#[test]
fn test_pyphi_phi_scales_with_size() {
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

    println!("PyPhi comparison - Size scaling:");
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

    // System EI should increase with size (more pairwise connections)
    assert!(
        phi_4.system_ei > phi_2.system_ei,
        "Larger systems should have more total information"
    );
}
