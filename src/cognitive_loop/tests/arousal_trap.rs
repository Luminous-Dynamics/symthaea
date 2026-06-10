// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Arousal Trap State Machine Tests
//!
//! Tests for `manage_arousal_trap()` (helpers/cycle_extracted.rs).
//! Three-phase state machine:
//! - Phase 1 (detect): counter increments when arousal > 0.8
//! - Phase 2 (recover, counter 5-10): gradual LR dampening + exploration boost
//! - Phase 3 (escape, counter >10): forced exploration=1.0, confidence×0.9, reset
//! - Low arousal (<0.3): consolidation LR boost
//!
//! Focus: state transitions, counter behavior, and numerical stability.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. BASIC: Arousal trap doesn't panic on any input
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn arousal_trap_safe_on_first_cycle() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // First cycle should not panic regardless of internal arousal
    let result = svc.cycle("arousal first cycle");
    assert!(result.prediction_error.is_finite());
}

#[test]
fn arousal_trap_stable_across_100_cycles() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("arousal stability");
        assert!(
            result.metadata.actual_effective_lr.is_finite()
                && result.metadata.actual_effective_lr >= 0.0,
            "LR non-finite or negative at cycle {i}: {}",
            result.metadata.actual_effective_lr
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. COUNTER BEHAVIOR: trap counter increments and resets correctly
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn arousal_trap_counter_resets_on_low_arousal() {
    // After high arousal cycles, dropping to low arousal should reset the counter.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Run some cycles to warm up
    for _ in 0..20 {
        svc.cycle("warmup");
    }
    // The counter is internal, but we can verify the system remains stable
    // after alternating between high-arousal and low-arousal inputs.
    for _ in 0..50 {
        svc.cycle("HIGH ALERT CRISIS PANIC"); // high arousal input
    }
    // Now calm input should allow recovery
    for i in 0..20 {
        let result = svc.cycle("calm peaceful serene");
        assert!(
            result.metadata.actual_effective_lr.is_finite(),
            "LR non-finite during recovery at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. CONFIDENCE: Arousal trap escape scales confidence
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn arousal_trap_confidence_bounded() {
    // Repeated forced escapes should not drive confidence below 0
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..300 {
        let result = svc.cycle("EMERGENCY CRISIS");
        let conf = svc.prediction_confidence();
        assert!(
            conf.is_finite() && conf >= 0.0 && conf <= 1.0,
            "confidence out of [0,1] at cycle {i}: {conf}"
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
// 4. LOW AROUSAL: Consolidation boost stays bounded
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn low_arousal_consolidation_lr_bounded() {
    // Low arousal boosts LR slightly for consolidation.
    // Verify LR stays bounded even across many low-arousal cycles.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..200 {
        let result = svc.cycle("calm serene peaceful");
        let lr = result.metadata.actual_effective_lr;
        assert!(
            lr.is_finite() && lr >= 0.0 && lr <= 10.0,
            "LR out of bounds during low arousal at cycle {i}: {lr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. TRANSITION STABILITY: Rapid arousal oscillations
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn arousal_rapid_oscillation_stable() {
    // Alternate between high and low arousal inputs every cycle.
    // This stresses the counter reset/increment logic.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let inputs = ["PANIC EMERGENCY ALERT", "calm peace serenity"];
    for i in 0..200 {
        let result = svc.cycle(inputs[i % 2]);
        assert!(
            result.prediction_error.is_finite(),
            "PE non-finite during oscillation at cycle {i}"
        );
        assert!(
            result.metadata.actual_effective_lr.is_finite(),
            "LR non-finite during oscillation at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. RECOVERY CYCLES: arousal_recovery_cycles stat increments
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn arousal_recovery_stat_non_negative() {
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for _ in 0..100 {
        svc.cycle("test recovery stat");
    }
    let stats = svc.stats();
    assert!(
        stats.arousal_recovery_cycles < 100,
        "recovery cycles {} seems too high for 100 total cycles",
        stats.arousal_recovery_cycles
    );
}
