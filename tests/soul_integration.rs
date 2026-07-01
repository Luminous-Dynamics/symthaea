// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the soul/ module.
//!
//! Tests cover:
//! - Soul creation with default and custom configs
//! - Value alignment evaluation
//! - Experience integration and statistics tracking
//! - Value updating and custom value addition
//! - Self-model inspection
//! - WeaverActor discovery recording

use symthaea::soul::{ConceptDiscovery, CoreValue, Experience, Soul, SoulConfig, WeaverActor};
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// Soul Creation & Config Tests
// ============================================================================

#[test]
fn test_soul_default_has_seven_harmonies() {
    let soul = Soul::default();
    let values: Vec<_> = soul.core_values().collect();
    assert_eq!(
        values.len(),
        8,
        "Default soul should have 8 core values (Eight Harmonies)"
    );
    assert!(
        soul.get_value("resonance").is_some(),
        "Default soul should include 'resonance' value"
    );
}

#[test]
fn test_soul_custom_config() {
    let config = SoulConfig {
        dimension: 256,
        ..Default::default()
    };
    let soul = Soul::new(config);
    // Essence should exist and be a valid HV
    let essence = soul.essence();
    assert!(
        essence.dim() > 0,
        "Essence should have positive dimensionality"
    );
}

// ============================================================================
// Alignment Tests
// ============================================================================

#[test]
fn test_alignment_produces_valid_scores() {
    let soul = Soul::default();
    let random_hv = ContinuousHV::random(512, 42);
    let result = soul.evaluate_alignment(&random_hv);

    assert!(
        result.overall_alignment >= -1.0 && result.overall_alignment <= 1.0,
        "Overall alignment should be in [-1, 1], got {}",
        result.overall_alignment
    );
    assert!(
        result.most_aligned.is_some(),
        "most_aligned should be Some for a soul with values"
    );
}

#[test]
fn test_alignment_with_own_essence() {
    let soul = Soul::default();
    let essence = soul.essence().clone();
    let result = soul.evaluate_alignment(&essence);

    assert!(
        result.overall_alignment > 0.0,
        "Alignment with own essence should be positive, got {}",
        result.overall_alignment
    );
}

// ============================================================================
// Experience Integration Tests
// ============================================================================

#[test]
fn test_experience_integration_updates_stats() {
    let mut soul = Soul::default();
    for i in 0..3 {
        soul.integrate_experience(Experience {
            embedding: ContinuousHV::random(512, 100 + i),
            value_alignment: 0.5,
            emotional_valence: 0.3,
            lessons: vec![format!("lesson_{i}")],
            timestamp: i,
        });
    }
    assert_eq!(
        soul.stats().experiences_integrated,
        3,
        "Should have integrated exactly 3 experiences"
    );
    assert!(
        soul.stats().avg_value_alignment.is_finite(),
        "Average alignment should be finite"
    );
}

#[test]
fn test_experience_integration_many() {
    let mut soul = Soul::default();
    for i in 0..50 {
        soul.integrate_experience(Experience {
            embedding: ContinuousHV::random(512, 200 + i),
            value_alignment: (i as f32) / 50.0,
            emotional_valence: 0.0,
            lessons: vec![],
            timestamp: i,
        });
    }
    let stats = soul.stats();
    assert_eq!(stats.experiences_integrated, 50);
    assert!(stats.avg_value_alignment.is_finite());
    assert!(stats.soul_coherence.is_finite());
}

// ============================================================================
// Value Management Tests
// ============================================================================

#[test]
fn test_update_value_blends_embedding() {
    let mut soul = Soul::default();
    let original = soul.get_value("resonance").unwrap().embedding.clone();

    let new_hv = ContinuousHV::random(512, 999);
    soul.update_value("resonance", new_hv);

    assert_eq!(
        soul.stats().value_updates,
        1,
        "Should record 1 value update"
    );
    let updated = &soul.get_value("resonance").unwrap().embedding;
    // The embedding should have changed (blended, not replaced entirely)
    assert_ne!(
        original.as_slice(),
        updated.as_slice(),
        "Embedding should change after update"
    );
}

#[test]
fn test_add_custom_value() {
    let mut soul = Soul::default();
    let custom = CoreValue::new("custom", "Custom Value", "A test value", 512);
    soul.add_value(custom);

    assert!(
        soul.get_value("custom").is_some(),
        "Custom value should be retrievable"
    );
    let values: Vec<_> = soul.core_values().collect();
    assert_eq!(
        values.len(),
        9,
        "Should now have 9 values (8 default + 1 custom)"
    );
}

// ============================================================================
// Self-Model Tests
// ============================================================================

#[test]
fn test_self_model_has_defaults() {
    let soul = Soul::default();
    let model = soul.self_model();

    assert!(
        !model.purpose.is_empty(),
        "Self-model purpose should be non-empty"
    );
    assert!(
        model.current_assessment.coherence.is_finite(),
        "Self-model coherence should be finite"
    );
}

// ============================================================================
// WeaverActor Tests
// ============================================================================

#[test]
fn test_weaver_actor_records_discoveries() {
    let mut weaver = WeaverActor::new();
    assert_eq!(weaver.discovery_count(), 0);

    for i in 0..3 {
        weaver.record_concept_discovery(&ConceptDiscovery {
            uid: format!("concept_{i}"),
            name: format!("Concept {i}"),
            attractor_signature: vec![0.1, 0.2, 0.3],
            consciousness_at_discovery: 0.5 + i as f32 * 0.1,
        });
    }

    assert_eq!(weaver.discovery_count(), 3, "Should have 3 discoveries");
    assert_eq!(weaver.discoveries().len(), 3);
    assert_eq!(weaver.discoveries()[0].uid, "concept_0");
}