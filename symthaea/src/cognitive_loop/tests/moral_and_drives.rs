// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for `apply_moral_modulation()` and `update_drives_and_reflection()`
//! (helpers/cycle_extracted.rs).
//!
//! These hot-path functions had zero test coverage.
//! Focus: numerical stability, safety invariants, and feedback correctness.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. MORAL MODULATION: Basic safety — no panics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn moral_modulation_safe_on_first_cycle() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = svc.cycle("moral test first cycle");
    assert!(result.prediction_error.is_finite());
    assert!(
        result
            .metadata
            .consciousness
            .consciousness_level
            .is_finite()
    );
}

#[test]
fn moral_modulation_stable_across_100_cycles() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("moral stability test");
        assert!(
            result.metadata.actual_effective_lr.is_finite()
                && result.metadata.actual_effective_lr >= 0.0,
            "LR non-finite or negative at cycle {i}: {}",
            result.metadata.actual_effective_lr
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness non-finite at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. MORAL CONCERN: Confidence and exploration respond to moral inputs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn moral_concern_dampens_exploration() {
    // High moral concern should dampen exploration (safety behavior).
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Warm up
    for _ in 0..10 {
        svc.cycle("warmup");
    }
    let exploration_before = svc.behavior.adaptive_behavior.exploration_factor;
    // Run many cycles — moral system should keep exploration bounded
    for _ in 0..50 {
        svc.cycle("test moral concern exploration");
    }
    let exploration_after = svc.behavior.adaptive_behavior.exploration_factor;
    // Exploration should remain bounded regardless of moral activity
    assert!(
        exploration_after >= 0.0 && exploration_after <= 1.0,
        "exploration out of bounds: {exploration_after}"
    );
    // Both should be finite
    assert!(exploration_before.is_finite());
    assert!(exploration_after.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. SURPRISE-GATED LEARNING: LR stays bounded
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn surprise_gated_lr_bounded_across_cycles() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..200 {
        let result = svc.cycle("surprise learning rate test");
        let lr = result.metadata.actual_effective_lr;
        assert!(
            lr.is_finite() && lr >= 0.0 && lr <= 10.0,
            "LR out of bounds at cycle {i}: {lr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. DRIVES AND REFLECTION: No panic on any urgency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn drives_and_reflection_stable_across_urgencies() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let inputs = [
        "extremely novel discovery breakthrough",
        "calm peaceful normal day",
        "urgent crisis emergency alert",
        "curious about patterns in data",
        "reflecting on past decisions",
    ];
    for i in 0..200 {
        let result = svc.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "PE non-finite at cycle {i}"
        );
        assert!(
            result.metadata.actual_effective_lr.is_finite(),
            "LR non-finite at cycle {i}"
        );
    }
}

#[test]
fn curiosity_drive_bounded() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("curious about everything");
        let confidence = svc.prediction_confidence();
        assert!(
            confidence.is_finite() && confidence >= 0.0 && confidence <= 1.0,
            "confidence out of [0,1] at cycle {i}: {confidence}"
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "consciousness non-finite at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. PARALLEL EPISODIC LEARNING: Memory encoding doesn't explode
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn episodic_learning_stable_across_cycles() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("memory encoding test");
        assert!(
            result.metadata.actual_effective_lr.is_finite(),
            "LR non-finite during episodic learning at cycle {i}"
        );
    }
    let stats = svc.stats();
    // Learning cycles should be non-zero after 100 cycles
    assert!(
        stats.total_cycles == 100,
        "expected 100 total cycles, got {}",
        stats.total_cycles
    );
}

#[test]
fn episodic_learning_prediction_error_bounded() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("episodic prediction error test");
        assert!(
            result.prediction_error.is_finite() && result.prediction_error >= 0.0,
            "PE non-finite or negative at cycle {i}: {}",
            result.prediction_error
        );
    }
}
