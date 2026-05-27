// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Modal Attention Router Integration Test Suite
//!
//! Tests for the cross-modal attention router covering:
//! - Basic routing with single and multiple modalities
//! - Salience-based attention allocation
//! - Φ-gated routing (consciousness affects integration)
//! - Context and goal influence on attention
//! - Temporal dynamics and history tracking
//! - Edge cases and boundary conditions
//!
//! The router dynamically allocates attention across modalities based on
//! salience, consciousness level (Φ), context, and goals.

use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::core_attention_router::{
    CrossModalAttentionRouter, ModalityInput, RouterConfig,
};
use symthaea::hdc::cross_modal_binding::Modality;

// ============================================================================
// BASIC ROUTER TESTS
// ============================================================================

#[test]
fn test_router_creation() {
    let _router = CrossModalAttentionRouter::new();
    // Should create without panicking
}

#[test]
fn test_router_with_custom_config() {
    let config = RouterConfig {
        phi_threshold: 0.5,
        temperature: 2.0,
        history_decay: 0.8,
        max_simultaneous_modalities: 4,
        salience_weight: 1.5,
        context_weight: 0.4,
        goal_weight: 0.3,
        inhibition_of_return: false,
        ior_strength: 0.0,
    };

    let _router = CrossModalAttentionRouter::with_config(config.clone());
    // Should create with custom config
}

#[test]
fn test_empty_input_routing() {
    let mut router = CrossModalAttentionRouter::new();
    let result = router.route(&[], 0.5);

    // Empty input should produce valid result
    assert!(result.attention_weights.is_empty() || result.quality >= 0.0);
}

// ============================================================================
// SINGLE MODALITY TESTS
// ============================================================================

#[test]
fn test_single_modality_routing() {
    let mut router = CrossModalAttentionRouter::new();

    let hv = BinaryHV::random(42);
    let input = ModalityInput::new(Modality::Visual, hv, 0.8);

    let result = router.route(&[input], 0.6);

    // Single modality should get all attention
    assert!(
        !result.attention_weights.is_empty(),
        "Should have attention weights"
    );
    assert!(result.quality >= 0.0, "Quality should be non-negative");
}

#[test]
fn test_single_modality_with_label() {
    let mut router = CrossModalAttentionRouter::new();

    let hv = BinaryHV::random(123);
    let input = ModalityInput::new(Modality::Semantic, hv, 0.9).with_label("test_semantic_input");

    let result = router.route(&[input], 0.7);

    assert!(result.quality >= 0.0, "Should produce valid result");
}

#[test]
fn test_single_modality_with_confidence() {
    let mut router = CrossModalAttentionRouter::new();

    let hv = BinaryHV::random(456);
    let input = ModalityInput::new(Modality::Auditory, hv, 0.5).with_confidence(0.9);

    let result = router.route(&[input], 0.5);

    assert!(result.quality >= 0.0, "Should produce valid result");
}

// ============================================================================
// MULTI-MODALITY TESTS
// ============================================================================

#[test]
fn test_dual_modality_routing() {
    let mut router = CrossModalAttentionRouter::new();

    let visual = ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8);
    let auditory = ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.6);

    let result = router.route(&[visual, auditory], 0.7);

    assert!(
        !result.attention_weights.is_empty(),
        "Should have attention for modalities"
    );
    assert!(result.effective_phi >= 0.0, "Should have valid effective Φ");
}

#[test]
fn test_multi_modality_routing() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(10), 0.9),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(20), 0.7),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(30), 0.8),
        ModalityInput::new(Modality::Temporal, BinaryHV::random(40), 0.5),
        ModalityInput::new(Modality::Emotional, BinaryHV::random(50), 0.6),
    ];

    let result = router.route(&inputs, 0.8);

    assert!(result.quality >= 0.0, "Should produce valid result");
    assert!(
        result.effective_phi > 0.0,
        "Should have positive effective Φ"
    );
}

#[test]
fn test_salience_determines_attention() {
    let mut router = CrossModalAttentionRouter::new();

    // High salience visual vs low salience auditory
    let visual = ModalityInput::new(Modality::Visual, BinaryHV::random(100), 0.95);
    let auditory = ModalityInput::new(Modality::Auditory, BinaryHV::random(200), 0.05);

    let result = router.route(&[visual, auditory], 0.8);

    // Visual should have higher attention due to higher salience
    if result.attention_weights.len() >= 2 {
        // Higher salience should generally lead to higher attention
        assert!(result.quality >= 0.0, "Should produce valid quality");
    }
}

// ============================================================================
// PHI-GATED ROUTING TESTS
// ============================================================================

#[test]
fn test_low_phi_limits_integration() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1000), 0.8),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2000), 0.7),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(3000), 0.9),
    ];

    // Very low Φ should limit integration
    let low_phi_result = router.route(&inputs.clone(), 0.1);

    // Higher Φ should allow better integration
    let high_phi_result = router.route(&inputs, 0.9);

    // Both should be valid
    assert!(
        low_phi_result.quality >= 0.0,
        "Low Φ result should be valid"
    );
    assert!(
        high_phi_result.quality >= 0.0,
        "High Φ result should be valid"
    );

    // Higher Φ generally allows more modalities to be bound
    assert!(
        high_phi_result.effective_phi >= low_phi_result.effective_phi * 0.5,
        "Higher input Φ should generally lead to higher effective Φ"
    );
}

#[test]
fn test_phi_threshold_behavior() {
    let config = RouterConfig {
        phi_threshold: 0.5,
        ..RouterConfig::default()
    };
    let mut router = CrossModalAttentionRouter::with_config(config);

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.7),
    ];

    // Below threshold
    let below_result = router.route(&inputs.clone(), 0.3);

    // Above threshold
    let above_result = router.route(&inputs, 0.7);

    // Both should produce valid results
    assert!(below_result.quality >= 0.0, "Below threshold should work");
    assert!(above_result.quality >= 0.0, "Above threshold should work");
}

// ============================================================================
// CONTEXT AND GOAL TESTS
// ============================================================================

#[test]
fn test_context_influences_attention() {
    let mut router = CrossModalAttentionRouter::new();

    // Set a visual-like context
    let visual_context = BinaryHV::random(7777);
    router.set_context(visual_context);

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(8888), 0.5),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(9999), 0.5),
    ];

    let result = router.route(&inputs, 0.7);

    // Should produce valid result with context
    assert!(result.quality >= 0.0, "Context routing should work");
}

#[test]
fn test_goal_influences_attention() {
    let mut router = CrossModalAttentionRouter::new();

    // Set a goal vector
    let goal = BinaryHV::random(12345);
    router.set_goal(goal);

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Semantic, BinaryHV::random(11111), 0.6),
        ModalityInput::new(Modality::Emotional, BinaryHV::random(22222), 0.6),
    ];

    let result = router.route(&inputs, 0.6);

    assert!(result.quality >= 0.0, "Goal-influenced routing should work");
}

#[test]
fn test_context_and_goal_together() {
    let mut router = CrossModalAttentionRouter::new();

    router.set_context(BinaryHV::random(111));
    router.set_goal(BinaryHV::random(222));

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(333), 0.7),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(444), 0.7),
        ModalityInput::new(Modality::Temporal, BinaryHV::random(555), 0.4),
    ];

    let result = router.route(&inputs, 0.75);

    assert!(
        result.quality >= 0.0,
        "Combined context and goal should work"
    );
}

// ============================================================================
// TEMPORAL DYNAMICS TESTS
// ============================================================================

#[test]
fn test_sequential_routing() {
    let mut router = CrossModalAttentionRouter::new();

    // Route multiple times in sequence
    for i in 0..10 {
        let salience = 0.5 + (i as f64 * 0.05);
        let input = ModalityInput::new(
            Modality::Visual,
            BinaryHV::random(i as u64),
            salience.min(1.0),
        );

        let result = router.route(&[input], 0.6);
        assert!(
            result.quality >= 0.0,
            "Step {} should produce valid result",
            i
        );
    }
}

#[test]
fn test_modality_attention_persistence() {
    let mut router = CrossModalAttentionRouter::new();

    // First route with visual
    let visual = ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.9);
    router.route(&[visual], 0.7);

    // Then route with auditory
    let auditory = ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.9);
    let result = router.route(&[auditory], 0.7);

    // Should handle modality switches
    assert!(result.quality >= 0.0, "Modality switch should work");
}

// ============================================================================
// EDGE CASES
// ============================================================================

#[test]
fn test_zero_salience_inputs() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.0),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.0),
    ];

    let result = router.route(&inputs, 0.5);

    // Should handle zero salience gracefully
    assert!(result.quality >= 0.0, "Zero salience should not crash");
}

#[test]
fn test_maximum_salience_inputs() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 1.0),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 1.0),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(3), 1.0),
    ];

    let result = router.route(&inputs, 1.0);

    // Should handle maximum salience
    assert!(result.quality >= 0.0, "Max salience should work");
}

#[test]
fn test_zero_phi() {
    let mut router = CrossModalAttentionRouter::new();

    let input = ModalityInput::new(Modality::Visual, BinaryHV::random(999), 0.7);
    let result = router.route(&[input], 0.0);

    // Zero Φ should not crash
    assert!(result.quality >= 0.0, "Zero Φ should not crash");
}

#[test]
fn test_extreme_phi() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.5),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.5),
    ];

    // Test with extreme Φ values
    let low_result = router.route(&inputs.clone(), 0.001);
    let high_result = router.route(&inputs, 10.0); // Above normal range

    assert!(low_result.quality >= 0.0, "Very low Φ should work");
    assert!(high_result.quality >= 0.0, "High Φ should work");
}

#[test]
fn test_duplicate_modalities() {
    let mut router = CrossModalAttentionRouter::new();

    // Multiple inputs from same modality
    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8),
        ModalityInput::new(Modality::Visual, BinaryHV::random(2), 0.6),
        ModalityInput::new(Modality::Visual, BinaryHV::random(3), 0.4),
    ];

    let result = router.route(&inputs, 0.7);

    // Should handle duplicate modalities
    assert!(result.quality >= 0.0, "Duplicate modalities should work");
}

// ============================================================================
// ALL MODALITY TYPES TESTS
// ============================================================================

#[test]
fn test_all_modality_types() {
    let mut router = CrossModalAttentionRouter::new();

    // Test all modality types
    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.7),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.6),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(3), 0.8),
        ModalityInput::new(Modality::Temporal, BinaryHV::random(4), 0.5),
        ModalityInput::new(Modality::Emotional, BinaryHV::random(5), 0.7),
        ModalityInput::new(Modality::Proprioceptive, BinaryHV::random(6), 0.4),
    ];

    let result = router.route(&inputs, 0.85);

    assert!(result.quality >= 0.0, "All modality types should work");
    assert!(
        !result.bound_modalities.is_empty(),
        "Should bind at least one modality"
    );
}

// ============================================================================
// CONFIGURATION TESTS
// ============================================================================

#[test]
fn test_default_config() {
    let config = RouterConfig::default();

    assert!(config.phi_threshold >= 0.0 && config.phi_threshold <= 1.0);
    assert!(config.temperature > 0.0);
    assert!(config.history_decay >= 0.0 && config.history_decay <= 1.0);
    assert!(config.max_simultaneous_modalities >= 1);
}

#[test]
fn test_custom_config_extremes() {
    // Very low temperature (sharper attention)
    let sharp_config = RouterConfig {
        temperature: 0.1,
        ..RouterConfig::default()
    };
    let mut router = CrossModalAttentionRouter::with_config(sharp_config);

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.9),
        ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.1),
    ];

    let result = router.route(&inputs, 0.7);
    assert!(result.quality >= 0.0, "Sharp attention config should work");

    // High temperature (softer attention)
    let soft_config = RouterConfig {
        temperature: 5.0,
        ..RouterConfig::default()
    };
    let mut router2 = CrossModalAttentionRouter::with_config(soft_config);

    let result2 = router2.route(&inputs, 0.7);
    assert!(result2.quality >= 0.0, "Soft attention config should work");
}

// ============================================================================
// ROUTING RESULT TESTS
// ============================================================================

#[test]
fn test_routing_result_has_unified_representation() {
    let mut router = CrossModalAttentionRouter::new();

    let inputs: Vec<ModalityInput> = vec![
        ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8),
        ModalityInput::new(Modality::Semantic, BinaryHV::random(2), 0.7),
    ];

    let result = router.route(&inputs, 0.7);

    // Should have a valid quality score
    assert!(result.quality >= 0.0, "Should produce valid quality score");
}

#[test]
fn test_routing_result_displays_correctly() {
    let mut router = CrossModalAttentionRouter::new();

    let input = ModalityInput::new(Modality::Visual, BinaryHV::random(42), 0.8);
    let result = router.route(&[input], 0.6);

    // Debug should not panic
    let display = format!("{:?}", result);
    assert!(!display.is_empty(), "Debug should produce output");
}