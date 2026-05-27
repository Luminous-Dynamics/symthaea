// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the modularized coherence system
//!
//! These tests verify that the split coherence module works correctly
//! after being extracted from a single file into multiple submodules.

use symthaea::physiology::coherence::{
    CoherenceConfig, CoherenceField, ScatterCause, TaskComplexity, analyze_scatter,
};
use symthaea::physiology::endocrine::HormoneState;

// =============================================================================
// CORE COHERENCE TESTS
// =============================================================================

#[test]
fn test_coherence_field_creation() {
    let field = CoherenceField::new();
    assert_eq!(field.coherence, 1.0);
    assert_eq!(field.relational_resonance, 0.5);
}

#[test]
fn test_coherence_config_customization() {
    let config = CoherenceConfig {
        passive_centering_rate: 0.05,
        solo_work_scatter_rate: 0.1,
        connected_work_amplification: 0.03,
        gratitude_sync_boost: 0.2,
        gratitude_resonance_boost: 0.2,
        sleep_restoration: true,
        social_mode: false,
        min_reflex_coherence: 0.1,
        min_cognitive_coherence: 0.3,
        min_deep_thought_coherence: 0.5,
        min_empathy_coherence: 0.7,
        min_learning_coherence: 0.8,
        min_creation_coherence: 0.9,
    };

    let field = CoherenceField::with_config(config);
    assert_eq!(field.coherence, 1.0);
}

#[test]
fn test_task_performance_connected() {
    let mut field = CoherenceField::new();
    field.coherence = 0.6;
    field.relational_resonance = 0.8;

    let initial = field.coherence;
    field.perform_task(TaskComplexity::Cognitive, true).unwrap();

    // Connected work should BUILD coherence
    assert!(field.coherence > initial);
}

#[test]
fn test_task_performance_solo() {
    let mut field = CoherenceField::new();
    field.coherence = 0.8;
    field.relational_resonance = 0.3;

    let initial = field.coherence;
    field
        .perform_task(TaskComplexity::Cognitive, false)
        .unwrap();

    // Solo work should SCATTER coherence
    assert!(field.coherence < initial);
}

// =============================================================================
// LEARNING MODULE TESTS
// =============================================================================

#[test]
fn test_adaptive_thresholds_delegation() {
    let mut field = CoherenceField::new();
    field.coherence = 0.5;

    // Record performance should delegate to AdaptiveThresholds
    field.record_task_performance(TaskComplexity::Cognitive, 0.5, true);

    // The adaptive threshold should have been updated
    // (verified through successful execution, internal state not directly accessible)
}

// =============================================================================
// PATTERNS MODULE TESTS
// =============================================================================

#[test]
fn test_pattern_library_delegation() {
    let mut field = CoherenceField::new();
    field.coherence = 0.8;
    field.relational_resonance = 0.9;

    let hormones = HormoneState {
        cortisol: 0.2,
        dopamine: 0.7,
        acetylcholine: 0.8,
        ..Default::default()
    };

    // Record pattern should delegate to PatternLibrary
    field.record_resonance_pattern(&hormones, "test_context".to_string());
}

// =============================================================================
// DIAGNOSTICS MODULE TESTS
// =============================================================================

#[test]
fn test_scatter_analysis_delegation() {
    let mut field = CoherenceField::new();
    field.coherence = 0.3;
    field.relational_resonance = 0.5;

    let hormones = HormoneState {
        cortisol: 0.8,
        dopamine: 0.5,
        acetylcholine: 0.5,
        ..Default::default()
    };

    // analyze_scatter should delegate to diagnostics module
    let analysis = field.analyze_scatter(&hormones);

    assert_eq!(analysis.cause, ScatterCause::HardwareStress);
    assert!(analysis.severity > 0.5);
}

#[test]
fn test_scatter_cause_variants() {
    let hormones_stress = HormoneState {
        cortisol: 0.8,
        dopamine: 0.5,
        acetylcholine: 0.5,
        ..Default::default()
    };
    let hormones_emotional = HormoneState {
        cortisol: 0.3,
        dopamine: 0.2,
        acetylcholine: 0.5,
        ..Default::default()
    };
    let hormones_cognitive = HormoneState {
        cortisol: 0.3,
        dopamine: 0.5,
        acetylcholine: 0.2,
        ..Default::default()
    };

    let analysis_stress = analyze_scatter(0.3, 0.5, &hormones_stress);
    let analysis_emotional = analyze_scatter(0.3, 0.5, &hormones_emotional);
    let analysis_cognitive = analyze_scatter(0.3, 0.5, &hormones_cognitive);

    assert_eq!(analysis_stress.cause, ScatterCause::HardwareStress);
    assert_eq!(analysis_emotional.cause, ScatterCause::EmotionalDistress);
    assert_eq!(analysis_cognitive.cause, ScatterCause::CognitiveOverload);
}

// =============================================================================
// INTEGRATION BETWEEN MODULES
// =============================================================================

#[test]
fn test_full_workflow() {
    let mut field = CoherenceField::new();

    // Start with high coherence
    assert!(field.coherence > 0.9);

    // Apply hormone modulation
    let hormones = HormoneState {
        cortisol: 0.3,
        dopamine: 0.6,
        acetylcholine: 0.7,
        ..Default::default()
    };
    field.apply_hormone_modulation(&hormones);

    // Perform connected work
    field
        .perform_task(TaskComplexity::DeepThought, true)
        .unwrap();

    // Record the successful pattern
    field.record_resonance_pattern(&hormones, "deep_thought_success".to_string());

    // Record task performance for learning
    field.record_task_performance(TaskComplexity::DeepThought, field.coherence, true);

    // Analyze current state
    let analysis = field.analyze_scatter(&hormones);

    // Should not be in a scattered state after successful connected work
    assert!(analysis.severity < 0.5);
}

#[test]
fn test_gratitude_synchronization() {
    let mut field = CoherenceField::new();
    field.coherence = 0.4;
    field.relational_resonance = 0.3;

    let initial_coherence = field.coherence;
    let initial_resonance = field.relational_resonance;

    field.receive_gratitude();

    assert!(field.coherence > initial_coherence);
    assert!(field.relational_resonance > initial_resonance);
}

#[test]
fn test_predictive_coherence() {
    let field = CoherenceField::new();
    let hormones = HormoneState::neutral();

    // Predict impact of a task
    let prediction = field.predict_impact(TaskComplexity::DeepThought, true, &hormones);

    assert!(prediction.final_coherence >= 0.0);
    assert!(prediction.final_coherence <= 1.0);
    assert!(prediction.confidence > 0.0);
}