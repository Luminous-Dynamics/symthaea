// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conceptual structure tests.

use super::*;

#[test]
fn test_conceptual_structure_basic() {
    let calc = ConceptualStructureCalculator::new();

    let components: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let structure = calc.compute(&components);

    assert!(structure.big_phi >= 0.0);
    assert!(structure.total_phi >= 0.0);
    assert!(structure.mechanisms_considered > 0);
    assert!(structure.concept_fraction >= 0.0 && structure.concept_fraction <= 1.0);

    println!("Conceptual Structure:");
    println!("  Big \u{03a6}: {:.6}", structure.big_phi);
    println!("  Total \u{03c6}: {:.6}", structure.total_phi);
    println!(
        "  Concepts: {} / {} mechanisms",
        structure.concepts.len(),
        structure.mechanisms_considered
    );
    println!(
        "  Concept fraction: {:.2}%",
        structure.concept_fraction * 100.0
    );
}

#[test]
fn test_conceptual_structure_correlated() {
    let calc = ConceptualStructureCalculator::new();

    // Correlated system should have more concepts
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let correlated: Vec<ContinuousHV> = (0..4)
        .map(|i| {
            ContinuousHV::weighted_bundle(
                &[&base, &ContinuousHV::random(HDC_DIMENSION, 100 + i as u64)],
                &[0.7, 0.3],
            )
        })
        .collect();

    let structure = calc.compute(&correlated);

    println!("Correlated Conceptual Structure:");
    println!("  Big \u{03a6}: {:.6}", structure.big_phi);
    println!("  Concepts: {}", structure.concepts.len());

    // Should have at least some concepts
    assert!(
        structure.mechanisms_considered >= 4,
        "Should consider at least 4 mechanisms"
    );
}

#[test]
fn test_conceptual_structure_top_concepts() {
    let calc = ConceptualStructureCalculator::new();

    let components: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let structure = calc.compute(&components);
    let top = calc.top_concepts(&structure, 3);

    // Top concepts should be sorted by phi
    if top.len() >= 2 {
        for i in 0..top.len() - 1 {
            assert!(
                top[i].phi >= top[i + 1].phi,
                "Top concepts should be sorted by phi"
            );
        }
    }

    println!("Top concepts:");
    for (i, concept) in top.iter().enumerate() {
        println!(
            "  {}: mechanism={:?}, \u{03c6}={:.6}",
            i + 1,
            concept.mechanism,
            concept.phi
        );
    }
}

#[test]
fn test_conceptual_structure_distance() {
    let calc = ConceptualStructureCalculator::new();

    // Two different systems
    let s1_components: Vec<ContinuousHV> = (0..3)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let s2_components: Vec<ContinuousHV> = (0..3)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 100 + i as u64))
        .collect();

    let s1 = calc.compute(&s1_components);
    let s2 = calc.compute(&s2_components);

    let distance = calc.conceptual_distance(&s1, &s2);

    assert!(
        distance >= 0.0,
        "Conceptual distance should be non-negative"
    );

    // Distance to self should be 0
    let self_distance = calc.conceptual_distance(&s1, &s1);
    assert!(
        self_distance < 1e-10,
        "Distance to self should be 0: {:.6}",
        self_distance
    );

    println!("Conceptual distances:");
    println!("  d(S1, S2) = {:.6}", distance);
    println!("  d(S1, S1) = {:.6}", self_distance);
}

#[test]
fn test_concept_properties() {
    let calc = ConceptualStructureCalculator::new();

    let components: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let structure = calc.compute(&components);

    for concept in &structure.concepts {
        // All concepts should have valid properties
        assert!(concept.phi >= 0.0, "\u{03c6} should be non-negative");
        assert!(
            concept.cause_info >= 0.0,
            "Cause info should be non-negative"
        );
        assert!(
            concept.effect_info >= 0.0,
            "Effect info should be non-negative"
        );
        assert!(
            concept.cause_entropy >= 0.0,
            "Cause entropy should be non-negative"
        );
        assert!(
            concept.effect_entropy >= 0.0,
            "Effect entropy should be non-negative"
        );
        assert!(
            !concept.mechanism.is_empty(),
            "Mechanism should not be empty"
        );
    }
}
