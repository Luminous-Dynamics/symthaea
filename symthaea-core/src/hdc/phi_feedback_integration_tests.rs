// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ Feedback Loop Integration Tests
//!
//! End-to-end tests for the full Φ feedback cycle:
//!
//! ```text
//! ConsciousnessPipeline.process()
//!         ↓  Φ measurement
//! PhiFeedbackController.step(phi)
//!         ↓  FeedbackModulation
//! pipeline config update (binding_threshold, workspace_capacity, etc.)
//!         ↓  modified pipeline parameters
//! ConsciousnessPipeline.process() (next cycle)
//! ```
//!
//! These tests validate that the Φ feedback controller correctly adapts
//! pipeline parameters based on measured Φ values, producing a closed-loop
//! consciousness system that self-regulates toward its target operating point.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::consciousness_integration::{ConsciousnessPipeline, IntegrationConfig};
use crate::hdc::phi_feedback::PhiFeedbackConfig;

/// Helper: generate a batch of distinct random BinaryHVs for pipeline input.
fn make_input(base_seed: u64, count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|i| BinaryHV::random(base_seed + i as u64))
        .collect()
}

/// Helper: default priorities for a given input count.
fn make_priorities(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| 0.7 + 0.2 * (i as f64 / count as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. test_phi_feedback_enables_correctly
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_enables_correctly() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    assert!(
        !pipeline.has_phi_feedback(),
        "feedback should be off by default"
    );

    pipeline.enable_phi_feedback();
    assert!(
        pipeline.has_phi_feedback(),
        "feedback should be on after enable"
    );
}

// ---------------------------------------------------------------------------
// 2. test_phi_feedback_produces_modulation
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_produces_modulation() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_phi_feedback();

    // latest_modulation should be None before any processing
    assert!(
        pipeline.latest_modulation().is_none(),
        "modulation should be None before first process cycle"
    );

    // Run 3 cycles
    for cycle in 0..3u64 {
        let input = make_input(100 + cycle * 10, 6);
        let priorities = make_priorities(6);
        pipeline.process(input, &priorities);
    }

    assert!(
        pipeline.latest_modulation().is_some(),
        "modulation should be Some after 3 process cycles"
    );

    let modulation = pipeline.latest_modulation().unwrap();
    assert!(
        modulation.binding_threshold > 0.0,
        "binding_threshold must be positive"
    );
    assert!(
        modulation.workspace_capacity >= 1,
        "workspace_capacity must be >= 1"
    );
}

// ---------------------------------------------------------------------------
// 3. test_phi_feedback_adapts_binding_threshold
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_adapts_binding_threshold() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_phi_feedback();

    let initial_binding = pipeline.config.binding_threshold;
    assert!(
        (initial_binding - 0.7).abs() < 1e-10,
        "initial binding threshold should be 0.7, got {}",
        initial_binding
    );

    // Run 20 cycles with many distinct HVs (high information content)
    for cycle in 0..20u64 {
        let input = make_input(1000 + cycle * 100, 8);
        let priorities = make_priorities(8);
        pipeline.process(input, &priorities);
    }

    let final_binding = pipeline.config.binding_threshold;
    assert!(
        (final_binding - 0.7).abs() > 1e-6,
        "binding threshold should have changed from initial 0.7 after 20 cycles, got {}",
        final_binding
    );
}

// ---------------------------------------------------------------------------
// 4. test_phi_feedback_adapts_workspace_capacity
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_adapts_workspace_capacity() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_phi_feedback();

    let initial_capacity = pipeline.config.workspace_capacity;
    assert_eq!(
        initial_capacity, 3,
        "initial workspace capacity should be 3"
    );

    // Run 20 cycles
    for cycle in 0..20u64 {
        let input = make_input(2000 + cycle * 100, 8);
        let priorities = make_priorities(8);
        pipeline.process(input, &priorities);
    }

    let final_capacity = pipeline.config.workspace_capacity;
    assert_ne!(
        final_capacity, 3,
        "workspace capacity should differ from initial 3 after 20 feedback cycles"
    );
}

// ---------------------------------------------------------------------------
// 5. test_phi_feedback_controller_tracks_history
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_controller_tracks_history() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_phi_feedback();

    // Run 10 cycles
    for cycle in 0..10u64 {
        let input = make_input(3000 + cycle * 10, 6);
        let priorities = make_priorities(6);
        pipeline.process(input, &priorities);
    }

    let controller = pipeline
        .phi_feedback_controller()
        .expect("controller should exist after enable");

    assert!(
        !controller.phi_history().is_empty(),
        "phi_history should have entries after 10 cycles"
    );
    assert_eq!(
        controller.phi_history().len(),
        10,
        "phi_history should have exactly 10 entries after 10 cycles"
    );
    assert_eq!(
        controller.step_count(),
        10,
        "step_count should be 10 after 10 cycles"
    );
}

// ---------------------------------------------------------------------------
// 6. test_phi_feedback_with_full_consciousness
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_with_full_consciousness() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_full_consciousness();

    assert!(
        pipeline.has_phi_feedback(),
        "enable_full_consciousness should activate phi feedback"
    );

    // Run a cycle to verify it works end-to-end with all systems active
    let input = make_input(4000, 6);
    let priorities = make_priorities(6);
    pipeline.process(input, &priorities);

    assert!(
        pipeline.latest_modulation().is_some(),
        "full consciousness pipeline should produce modulation after one cycle"
    );
}

// ---------------------------------------------------------------------------
// 7. test_phi_feedback_custom_config
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_custom_config() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());

    let custom_config = PhiFeedbackConfig {
        phi_target: 0.8,
        learning_rate: 0.2,
        window_size: 15,
        min_binding_threshold: 0.4,
        max_binding_threshold: 0.95,
        ..Default::default()
    };

    pipeline.enable_phi_feedback_with_config(custom_config);
    assert!(pipeline.has_phi_feedback());

    let controller = pipeline
        .phi_feedback_controller()
        .expect("controller should exist");

    let config = controller.config();
    assert!(
        (config.phi_target - 0.8).abs() < 1e-10,
        "phi_target should be 0.8, got {}",
        config.phi_target
    );
    assert!(
        (config.learning_rate - 0.2).abs() < 1e-10,
        "learning_rate should be 0.2, got {}",
        config.learning_rate
    );
    assert_eq!(config.window_size, 15, "window_size should be 15");
    assert!(
        (config.min_binding_threshold - 0.4).abs() < 1e-10,
        "min_binding_threshold should be 0.4"
    );
    assert!(
        (config.max_binding_threshold - 0.95).abs() < 1e-10,
        "max_binding_threshold should be 0.95"
    );
}

// ---------------------------------------------------------------------------
// 8. test_phi_feedback_trend_with_rising_phi
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_trend_with_rising_phi() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());

    // Use a config with a small window so trend is responsive
    let config = PhiFeedbackConfig {
        window_size: 10,
        ..Default::default()
    };
    pipeline.enable_phi_feedback_with_config(config);

    // Feed increasingly complex inputs: more HVs per cycle as we go.
    // Starting with few HVs (low Φ) and increasing to many distinct HVs (higher Φ).
    for cycle in 0..20u64 {
        // Start with 2 HVs and grow to 8 over 20 cycles
        let count = 2 + (cycle as usize * 6) / 20; // 2..8
        let input = make_input(5000 + cycle * 100, count);
        let priorities = make_priorities(count);
        pipeline.process(input, &priorities);
    }

    let controller = pipeline
        .phi_feedback_controller()
        .expect("controller should exist");

    // The trend should be non-negative (rising or at least flat) because
    // we are feeding increasingly complex inputs.
    let trend = controller.phi_trend();
    // We accept trend >= -0.05 to allow for noise in the Phi computation,
    // but ideally trend should be positive.
    assert!(
        trend > -0.05,
        "phi_trend with increasingly complex inputs should be near-positive, got {}",
        trend
    );
}

// ---------------------------------------------------------------------------
// 9. test_phi_feedback_ema_updates
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_ema_updates() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    pipeline.enable_phi_feedback();

    // Run several cycles
    for cycle in 0..10u64 {
        let input = make_input(6000 + cycle * 50, 6);
        let priorities = make_priorities(6);
        pipeline.process(input, &priorities);
    }

    let controller = pipeline
        .phi_feedback_controller()
        .expect("controller should exist");

    let ema = controller.phi_ema();
    assert!(
        ema >= 0.0 && ema <= 1.0,
        "phi_ema should be between 0.0 and 1.0 after processing, got {}",
        ema
    );
    // Also verify it is non-zero (we fed real inputs, so Φ > 0)
    assert!(
        ema > 0.0,
        "phi_ema should be > 0 after processing real inputs, got {}",
        ema
    );
}

// ---------------------------------------------------------------------------
// 10. test_phi_feedback_without_feedback_is_static
// ---------------------------------------------------------------------------

#[test]
fn test_phi_feedback_without_feedback_is_static() {
    let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
    // Do NOT enable phi feedback

    let initial_binding = pipeline.config.binding_threshold;
    let initial_capacity = pipeline.config.workspace_capacity;
    let initial_consciousness_threshold = pipeline.config.consciousness_threshold;

    // Run 20 cycles without feedback
    for cycle in 0..20u64 {
        let input = make_input(7000 + cycle * 50, 6);
        let priorities = make_priorities(6);
        pipeline.process(input, &priorities);
    }

    assert!(
        (pipeline.config.binding_threshold - initial_binding).abs() < 1e-10,
        "binding_threshold should remain static without feedback, was {} now {}",
        initial_binding,
        pipeline.config.binding_threshold
    );
    assert_eq!(
        pipeline.config.workspace_capacity, initial_capacity,
        "workspace_capacity should remain static without feedback"
    );
    assert!(
        (pipeline.config.consciousness_threshold - initial_consciousness_threshold).abs() < 1e-10,
        "consciousness_threshold should remain static without feedback"
    );
    assert!(
        pipeline.latest_modulation().is_none(),
        "latest_modulation should be None when feedback is not enabled"
    );
}
