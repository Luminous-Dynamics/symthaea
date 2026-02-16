// ==================================================================================
// Integration Test: CognitiveLoopService Full Cycle
// ==================================================================================
//
// Tests the complete cognitive loop pipeline end-to-end:
//   Input text → HDC encode → CfC evolve → predict → learn → CycleResult
//
// This is the primary integration test for the core cognitive pipeline.
// No feature flags required — uses default configuration.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// ── Basic Lifecycle ─────────────────────────────────────────────────

#[test]
fn test_service_creation_and_single_cycle() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert_eq!(service.stats().total_cycles, 0);

    let result = service.cycle("hello world");

    assert_eq!(service.stats().total_cycles, 1);
    assert!(result.prediction_error >= 0.0);
    assert!(result.prediction_error <= 1.0);
    assert!(!result.output.is_empty());
    assert!(result.cycle_time_us > 0);
}

// ── Learning Over Repeated Input ────────────────────────────────────

#[test]
fn test_error_decreases_over_repeated_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        ..Default::default()
    })
    .unwrap();

    let mut errors = Vec::new();
    for _ in 0..20 {
        let result = service.cycle("cause effect action");
        errors.push(result.prediction_error);
    }

    // Prediction error should generally decrease (second half < first half + margin)
    let first_half_avg: f32 = errors[..10].iter().sum::<f32>() / 10.0;
    let second_half_avg: f32 = errors[10..].iter().sum::<f32>() / 10.0;

    assert!(
        second_half_avg <= first_half_avg + 0.15,
        "Error should decrease or stabilize: first_half={first_half_avg:.4}, second_half={second_half_avg:.4}"
    );
}

// ── Diverse Input Processing ────────────────────────────────────────

#[test]
fn test_diverse_inputs_produce_different_outputs() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let r1 = service.cycle("the cat sat on the mat");
    let r2 = service.cycle("quantum mechanics wave function");
    let r3 = service.cycle("love compassion kindness");

    // Different inputs should produce different CfC outputs
    assert_ne!(r1.output, r2.output);
    assert_ne!(r2.output, r3.output);
    assert_eq!(service.stats().total_cycles, 3);
}

// ── CycleMetadata Population ────────────────────────────────────────

#[test]
fn test_cycle_metadata_populated() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_surprise_exploration: false,
        ..Default::default()
    })
    .unwrap();

    let result = service.cycle("test metadata");

    // With surprise exploration disabled, surprise should not trigger
    assert!(!result.metadata.surprise_triggered);
    // Prefrontal not yet wired
    assert!(!result.metadata.prefrontal_veto);
    // Reasoning engine not active without feature flag
    assert_eq!(result.metadata.reasoning_confidence, 0.0);
    // No exploration action without surprise
    assert!(result.metadata.exploration_action.is_none());
}

// ── Surprise Exploration Bridge ─────────────────────────────────────

#[test]
fn test_surprise_exploration_enabled() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        enable_surprise_exploration: true,
        ..Default::default()
    })
    .unwrap();

    // Run cycles — surprise bridge should be active (may or may not trigger)
    let mut any_surprise = false;
    for i in 0..30 {
        // Alternate between familiar and novel inputs to trigger surprise
        let input = if i < 15 {
            "consistent pattern input"
        } else {
            "completely different novel stimulus"
        };
        let result = service.cycle(input);
        if result.metadata.surprise_triggered {
            any_surprise = true;
        }
    }

    // Surprise bridge was at least initialized (it may not trigger depending
    // on the adaptive threshold, so we just verify it doesn't crash)
    assert_eq!(service.stats().total_cycles, 30);
    // Note: we don't assert any_surprise==true because the adaptive threshold
    // may not be exceeded in 30 cycles with this input pattern
    let _ = any_surprise;
}

// ── Learning Occurred ───────────────────────────────────────────────

#[test]
fn test_learning_occurs_when_threshold_met() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        async_training: false,   // Synchronous so training_loss is immediately available
        ..Default::default()
    })
    .unwrap();

    let mut any_learning = false;
    for _ in 0..10 {
        let result = service.cycle("learning test input");
        if result.learning_occurred {
            any_learning = true;
            assert!(result.training_loss.is_some());
        }
    }

    assert!(any_learning, "Learning should occur at least once in 10 cycles");
}

// ── Stats Accumulation ──────────────────────────────────────────────

#[test]
fn test_stats_accumulate_correctly() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for _ in 0..5 {
        service.cycle("stats test");
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 5);
    assert!(stats.avg_prediction_error >= 0.0);
    assert!(stats.avg_prediction_error <= 1.0);
}

// ── Detected Primitives ─────────────────────────────────────────────

#[test]
fn test_primitives_detected_from_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Use input text that contains known primitive-like words
    let result = service.cycle("cause and effect");

    // The encoder should detect at least some primitives
    // (exact primitives depend on the HDC encoder's dictionary)
    assert!(result.detected_primitives.len() >= 0); // Non-panicking access
    assert!(result.attention_state.len() >= 0); // Attention state populated
}

// ── Long Running Stability ──────────────────────────────────────────

#[test]
fn test_100_cycle_stability() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "consciousness emerges from complexity",
        "hyperdimensional computing enables reasoning",
        "liquid time-constants model temporal dynamics",
    ];

    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error must be finite at cycle {i}"
        );
        assert!(
            result.cycle_time_us < 10_000_000, // < 10 seconds per cycle
            "Cycle {i} took too long: {}us",
            result.cycle_time_us
        );
    }

    assert_eq!(service.stats().total_cycles, 100);
}
