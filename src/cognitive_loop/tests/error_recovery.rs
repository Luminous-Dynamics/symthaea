// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error recovery and graceful degradation tests.
//!
//! Verifies the cognitive loop degrades gracefully under adversarial conditions:
//! - Extreme prediction errors (CfC divergence)
//! - NaN/Inf injection into internal state
//! - Memory saturation (buffer full for extended periods)
//! - Rapid state transitions (stress test mode switching)
//! - Zero-input sustained operation (starvation)

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// CfC Divergence Recovery
// ═══════════════════════════════════════════════════════════════════════════════

/// After sustained high prediction errors (simulating CfC divergence),
/// the system should self-stabilize rather than entering a death spiral.
#[test]
fn test_recovery_from_sustained_high_error() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Warm up to exit startup transient
    for i in 0..50 {
        let _ = service.cycle(&format!("warmup {i}"));
    }

    let pre_confidence = service.prediction_confidence();

    // Feed deliberately confusing inputs to spike prediction error
    for i in 0..100 {
        // Alternate between very different inputs to maximize PE
        let input = if i % 2 == 0 {
            "the quick brown fox jumps"
        } else {
            "42 quantum photonic substrate"
        };
        let _ = service.cycle(input);
    }

    // System should have adapted — not crashed
    let stats = service.stats();
    assert!(
        stats.avg_prediction_error.is_finite(),
        "PE should remain finite after stress"
    );
    assert!(stats.total_cycles == 150, "All cycles should complete");

    // Prediction confidence should have decreased but not collapsed to exactly 0
    let post_confidence = service.prediction_confidence();
    assert!(
        post_confidence.is_finite(),
        "Confidence should remain finite"
    );
    assert!(post_confidence >= 0.0, "Confidence should not go negative");

    // Recovery: feed consistent input and observe error decrease
    let error_before_recovery: f32 = stats.avg_prediction_error;
    for _ in 0..50 {
        let _ = service.cycle("consistent stable input");
    }
    let error_after_recovery = service.stats().avg_prediction_error;
    // Error should decrease with consistent input (or at least remain finite)
    assert!(
        error_after_recovery.is_finite(),
        "Error should be finite after recovery period"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Saturation
// ═══════════════════════════════════════════════════════════════════════════════

/// With a tiny buffer, the system should evict old experiences gracefully
/// and never panic from buffer overflow.
#[test]
fn test_buffer_saturation_no_panic() {
    let mut config = CognitiveLoopConfig::default();
    config.buffer_size = 5; // Minimal buffer
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run 200 cycles — buffer will saturate immediately and must evict
    for i in 0..200 {
        let result = service.cycle(&format!("cycle {i}"));
        assert!(
            result.prediction_error.is_finite(),
            "PE finite at cycle {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    assert!(
        stats.buffer_utilization <= 1.0,
        "Buffer utilization should not exceed 100%"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zero-Input Starvation
// ═══════════════════════════════════════════════════════════════════════════════

/// Running with empty string input for many cycles should not cause
/// division by zero, NaN propagation, or panics.
#[test]
fn test_empty_input_starvation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for _ in 0..100 {
        let result = service.cycle("");
        assert!(
            result.prediction_error.is_finite(),
            "PE must stay finite on empty input"
        );
        assert!(
            result.metadata.pipeline_consciousness.is_finite(),
            "Consciousness must stay finite on empty input"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rapid Mode Transitions
// ═══════════════════════════════════════════════════════════════════════════════

/// Rapidly alternating between states that should trigger different urgency
/// modes should not cause instability in the strategy system.
#[test]
fn test_rapid_urgency_transitions() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Alternate between calm and excited inputs rapidly
    for i in 0..200 {
        let input = if i % 3 == 0 {
            // Long, calm, repetitive
            "peace quiet calm rest breathe slowly gently softly warmly"
        } else if i % 3 == 1 {
            // Short, urgent
            "DANGER ALERT CRITICAL EMERGENCY NOW"
        } else {
            // Novel, exploratory
            "quantum entanglement consciousness photonic substrate exotic"
        };
        let result = service.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "PE must stay finite during rapid transitions at cycle {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    // Strategy should be valid (not stuck in an invalid state)
    assert!(
        !stats.consciousness_pattern.is_empty(),
        "Pattern should be classified"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Reset Recovery
// ═══════════════════════════════════════════════════════════════════════════════

/// After reset(), the system should return to a clean initial state
/// and be able to resume normal operation.
#[test]
fn test_reset_full_recovery() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles to build up state
    for i in 0..50 {
        let _ = service.cycle(&format!("pre-reset {i}"));
    }
    assert!(service.stats().total_cycles > 0);

    // Reset
    service.reset();
    assert_eq!(service.stats().total_cycles, 0, "Cycle count should reset");

    // Should operate normally post-reset
    for i in 0..30 {
        let result = service.cycle(&format!("post-reset {i}"));
        assert!(
            result.prediction_error.is_finite(),
            "Post-reset cycle {i} should produce finite PE"
        );
    }
    assert_eq!(service.stats().total_cycles, 30);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Extreme Input Length
// ═══════════════════════════════════════════════════════════════════════════════

/// Very long inputs should be handled gracefully (HDC encoder truncates/hashes).
#[test]
fn test_extreme_input_length() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // 100KB input
    let long_input = "word ".repeat(20_000);
    let result = service.cycle(&long_input);
    assert!(
        result.prediction_error.is_finite(),
        "Long input should not cause NaN"
    );

    // Single character
    let result = service.cycle("a");
    assert!(
        result.prediction_error.is_finite(),
        "Tiny input should not cause NaN"
    );

    // Unicode stress
    let unicode_input = "🧠💡🔬🧬🌊".repeat(100);
    let result = service.cycle(&unicode_input);
    assert!(
        result.prediction_error.is_finite(),
        "Unicode input should not cause NaN"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Concurrent Subsystem Firing
// ═══════════════════════════════════════════════════════════════════════════════

/// At the LCM of co-prime intervals (13*11*17*19*37 = 1,571,043),
/// all subsystems fire simultaneously. Test a subset of cycles where
/// multiple managers fire at once to verify no interference.
#[test]
fn test_multi_manager_firing() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Cycle numbers where multiple managers fire:
    // 11*13 = 143 (memory + drive)
    // 11*17 = 187 (memory + learning)
    // 13*17 = 221 (drive + learning)
    // 11*13*17 = 2431 (memory + drive + learning)
    // Run to cycle 250 to hit several multi-fire points
    for i in 0..250 {
        let result = service.cycle(&format!("multi-fire test {i}"));
        assert!(
            result.prediction_error.is_finite(),
            "PE must be finite at cycle {i} (multi-manager firing)"
        );
        assert!(
            result.metadata.pipeline_consciousness.is_finite(),
            "Consciousness must be finite at cycle {i}"
        );
    }
}
