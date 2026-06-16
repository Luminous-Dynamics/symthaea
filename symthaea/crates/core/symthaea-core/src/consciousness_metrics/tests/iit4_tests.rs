// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IIT 4.0 tests: intrinsic difference, small phi, and big Phi.

use super::*;

#[test]
fn test_iit4_intrinsic_difference() {
    let calc = IIT4Calculator::new();

    // Same vector should have zero intrinsic difference
    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let id_same = calc.intrinsic_difference(&a, &a);
    assert!(
        id_same < 0.01,
        "Same vector should have near-zero id: {:.6}",
        id_same
    );

    // Different vectors should have positive id
    let b = ContinuousHV::random(HDC_DIMENSION, 2);
    let id_diff = calc.intrinsic_difference(&a, &b);
    assert!(
        id_diff >= 0.0,
        "Different vectors should have non-negative id: {:.6}",
        id_diff
    );
}

#[test]
fn test_iit4_small_phi() {
    let calc = IIT4Calculator::new();

    // Create a mechanism with context
    let mechanism = ContinuousHV::random(HDC_DIMENSION, 1);
    let context = vec![
        ContinuousHV::random(HDC_DIMENSION, 2),
        ContinuousHV::random(HDC_DIMENSION, 3),
    ];

    let phi = calc.small_phi(&mechanism, &context);
    assert!(phi >= 0.0, "Small phi should be non-negative: {:.6}", phi);
}

#[test]
fn test_iit4_analyze() {
    let calc = IIT4Calculator::new();

    let components: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    let result = calc.analyze(&components);

    assert!(result.intrinsic_difference >= 0.0);
    assert!(result.small_phi >= 0.0);
    assert!(result.big_phi >= 0.0);
    assert!(result.intrinsic_information >= 0.0);

    println!("IIT 4.0 Analysis:");
    println!("  Intrinsic Difference: {:.6}", result.intrinsic_difference);
    println!("  Small \u{03c6} (avg): {:.6}", result.small_phi);
    println!("  Big \u{03a6}: {:.6}", result.big_phi);
    println!(
        "  Intrinsic Information: {:.6}",
        result.intrinsic_information
    );
    println!("  Concept Count: {}", result.concept_count);
}

#[test]
fn test_iit4_correlated_higher_phi() {
    let calc = IIT4Calculator::new();

    // Independent system
    let independent: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i * 1000))
        .collect();

    // Correlated system
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let correlated: Vec<ContinuousHV> = (0..4)
        .map(|i| {
            ContinuousHV::weighted_bundle(
                &[&base, &ContinuousHV::random(HDC_DIMENSION, 100 + i)],
                &[0.8, 0.2],
            )
        })
        .collect();

    let ind_result = calc.analyze(&independent);
    let cor_result = calc.analyze(&correlated);

    // Both results should have valid, finite values
    assert!(
        ind_result.big_phi.is_finite(),
        "Independent Φ should be finite"
    );
    assert!(
        cor_result.big_phi.is_finite(),
        "Correlated Φ should be finite"
    );
    assert!(
        ind_result.big_phi >= 0.0,
        "Independent Φ should be non-negative"
    );
    assert!(
        cor_result.big_phi >= 0.0,
        "Correlated Φ should be non-negative"
    );

    // Correlated should have higher Φ than independent
    assert!(
        cor_result.big_phi > ind_result.big_phi,
        "Correlated Φ ({:.6}) should exceed independent Φ ({:.6})",
        cor_result.big_phi,
        ind_result.big_phi
    );

    println!(
        "IIT 4.0 - Independent \u{03a6}: {:.6}, Correlated \u{03a6}: {:.6}",
        ind_result.big_phi, cor_result.big_phi
    );
}
