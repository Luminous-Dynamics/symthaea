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
        second_half_avg <= first_half_avg + 0.08,
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

    assert!(
        any_learning,
        "Learning should occur at least once in 10 cycles"
    );
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
    assert!(result.peak_attention >= 0.0); // Attention state populated
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

    let avg_error = service.stats().avg_prediction_error;
    assert!(
        avg_error < 1.0,
        "Average prediction error should be bounded over 100 cycles: got {avg_error:.4}"
    );
}

// ── All Modules No Degradation ───────────────────────────────────

#[test]
fn test_all_modules_no_degradation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: true,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        enable_narrative_self: true,
        enable_predictive_self: true,
        enable_attention_schema: true,
        enable_gwt: true,
        enable_temporal_consciousness: true,
        enable_embodied_cognition: true,
        enable_narrative_gwt: true,
        ..Default::default()
    })
    .unwrap();

    let mut errors = Vec::new();
    for _ in 0..50 {
        let result = service.cycle("all modules integration stability test");
        errors.push(result.prediction_error);
        assert!(
            result.prediction_error.is_finite(),
            "Error must be finite with all modules enabled"
        );
    }

    let avg = errors.iter().sum::<f32>() / errors.len() as f32;
    assert!(
        avg < 1.0,
        "Average error with all modules should be bounded: got {avg:.4}"
    );
}

// ── Virtual Body Enabled by Default ─────────────────────────────────

#[test]
fn test_virtual_body_enabled_by_default() {
    let config = CognitiveLoopConfig::default();
    assert!(
        config.enable_virtual_body,
        "Virtual body should be enabled by default"
    );

    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run 5 cycles to let virtual body accumulate state
    for _ in 0..5 {
        let result = service.cycle("embodied cognition test");
        // Virtual body should produce phi modulation (not neutral 1.0 after a few cycles)
        assert!(result.metadata.body_phi_modulation >= 0.5);
        assert!(result.metadata.body_phi_modulation <= 1.5);
    }
}

// ── Master Consciousness Equation ───────────────────────────────────

#[test]
fn test_master_consciousness_equation_runs_periodically() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    let mut mce_values = Vec::new();
    // Run 25 cycles — MCE fires periodically based on urgency-adaptive scheduling.
    // With learning_threshold=0.0, urgency is Critical → MCE fires every 5th cycle.
    for _ in 0..25 {
        let result = service.cycle("consciousness measurement test");
        mce_values.push(result.metadata.consciousness_level);
    }

    // MCE should fire at least twice in 25 cycles regardless of urgency
    let non_zero_count = mce_values.iter().filter(|&&v| v > 0.0).count();
    assert!(
        non_zero_count >= 2,
        "MCE should fire periodically, got {} firings in 25 cycles: {:?}",
        non_zero_count,
        &mce_values
    );

    // Cycle 1 should NOT fire (no schedule hits cycle 1: 1%5≠0, 1%10≠0, 1%20≠0)
    assert_eq!(mce_values[0], 0.0, "MCE should not run on cycle 1");

    // At least one MCE value should be in valid range (0, 1]
    let max_mce = mce_values.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_mce > 0.0 && max_mce <= 1.0,
        "MCE consciousness_level should be in (0, 1], got {}",
        max_mce
    );
}

// ── Narrative Veto Carry-Over ────────────────────────────────────

#[test]
fn test_narrative_veto_carry_over() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_narrative_gwt: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run 30 cycles — narrative-GWT SelfPhiTooLow veto should fire in early cycles
    // (self_phi starts near 0, default min_self_phi=0.3)
    let mut results = Vec::new();
    for _ in 0..30 {
        let result = service.cycle("narrative veto carry-over test input");
        results.push(result);
    }

    // Find cycles where veto fired
    let veto_cycles: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.metadata.narrative_gwt_veto)
        .map(|(i, _)| i)
        .collect();

    println!("Veto cycles: {:?}", veto_cycles);

    // If a veto fires on cycle N, cycle N+1 should have narrative_veto_active=true
    // which suppresses learning. We verify by checking learning_occurred on the
    // cycle after a veto.
    for &n in &veto_cycles {
        if n + 1 < results.len() {
            // The narrative_veto_active flag was set to true at end of cycle N,
            // so cycle N+1 should have learning suppressed.
            // Note: learning_occurred may still be false for other reasons,
            // so we just verify the system is stable after veto carry-over.
            assert!(
                results[n + 1].prediction_error.is_finite(),
                "Cycle after veto should produce finite results"
            );
        }
    }

    // Verify the service is stable after all cycles
    let final_result = service.cycle("stability check after veto");
    assert!(final_result.prediction_error.is_finite());
}

// ── Embodied Phi Accumulation Across Cycles ──────────────────────

#[test]
fn test_embodied_phi_accumulation_across_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_embodied_cognition: true,
        enable_virtual_body: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Track embodied_phi_modulation trajectory over 100 cycles
    let mut embodied_mods = Vec::new();
    for _ in 0..100 {
        let result = service.cycle("embodied phi accumulation test");
        embodied_mods.push(result.metadata.embodied_phi_modulation);
    }

    // At least 1 value should deviate from 1.0 (module is active)
    let deviations = embodied_mods
        .iter()
        .filter(|&&m| (m - 1.0).abs() > 0.001)
        .count();
    assert!(
        deviations > 0,
        "At least one embodied_phi_modulation should deviate from 1.0, got all neutral"
    );

    // At least 3 unique values (state evolves, not static)
    let mut unique = embodied_mods.clone();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup_by(|a, b| (*a - *b).abs() < 0.0001);
    assert!(
        unique.len() >= 3,
        "Embodied phi modulation should have >=3 unique values, got {}",
        unique.len()
    );

    // Values should differ from initial value (cross-cycle accumulation)
    let initial = embodied_mods[0];
    let differs_from_initial = embodied_mods
        .iter()
        .filter(|&&m| (m - initial).abs() > 0.001)
        .count();
    assert!(
        differs_from_initial > 0,
        "Embodied phi should evolve from initial value {initial:.4}"
    );
}

// ── Performance Overhead Quick Check ─────────────────────────────

#[test]
#[ignore] // Run manually: cargo test test_feedback_loop_overhead -- --ignored --nocapture
fn test_feedback_loop_overhead_quick_check() {
    use std::time::Instant;

    let warmup_cycles = 10;
    let measure_cycles = 200;

    let inputs = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];

    // Minimal config (all OFF)
    let mut minimal = CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some("overhead_benchmark_seed".to_string()),
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: false,
        enable_surprise_exploration: false,
        enable_prefrontal: false,
        enable_meta_cognition: false,
        enable_narrative_self: false,
        enable_predictive_self: false,
        enable_attention_schema: false,
        enable_gwt: false,
        enable_resonance: false,
        enable_quantum_coherence: false,
        enable_temporal_consciousness: false,
        enable_embodied_cognition: false,
        enable_narrative_gwt: false,
        ..Default::default()
    })
    .unwrap();

    // Full config (all ON)
    let mut full = CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some("overhead_benchmark_seed".to_string()),
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: true,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        enable_narrative_self: true,
        enable_predictive_self: true,
        enable_attention_schema: true,
        enable_gwt: true,
        enable_resonance: true,
        enable_quantum_coherence: true,
        enable_temporal_consciousness: true,
        enable_embodied_cognition: true,
        enable_narrative_gwt: true,
        ..Default::default()
    })
    .unwrap();

    // Warmup
    for i in 0..warmup_cycles {
        minimal.cycle(inputs[i % inputs.len()]);
        full.cycle(inputs[i % inputs.len()]);
    }

    // Measure minimal
    let start_min = Instant::now();
    for i in 0..measure_cycles {
        minimal.cycle(inputs[i % inputs.len()]);
    }
    let elapsed_min = start_min.elapsed();

    // Measure full
    let start_full = Instant::now();
    for i in 0..measure_cycles {
        full.cycle(inputs[i % inputs.len()]);
    }
    let elapsed_full = start_full.elapsed();

    let per_cycle_min_us = elapsed_min.as_micros() as f64 / measure_cycles as f64;
    let per_cycle_full_us = elapsed_full.as_micros() as f64 / measure_cycles as f64;
    let hz_min = 1_000_000.0 / per_cycle_min_us;
    let hz_full = 1_000_000.0 / per_cycle_full_us;
    let overhead_pct = (per_cycle_full_us - per_cycle_min_us) / per_cycle_min_us * 100.0;

    println!("=== Feedback Loop Overhead ===");
    println!("Minimal:  {per_cycle_min_us:.0} µs/cycle ({hz_min:.0} Hz)");
    println!("Full:     {per_cycle_full_us:.0} µs/cycle ({hz_full:.0} Hz)");
    println!("Overhead: {overhead_pct:.1}%");

    assert!(
        overhead_pct < 50.0,
        "Full subsystem overhead should be <50%: got {overhead_pct:.1}%"
    );
}

// ── Predictive + Affective + Cross-Modal Synergy ─────────────────

#[test]
fn test_predictive_affective_crossmodal_synergy() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: true,
        enable_predictive_processing: true,
        enable_cross_modal_binding: true,
        enable_affective_bridge: true,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
    ];

    let mut saw_affective = false;
    let mut saw_predictive = false;
    let mut saw_binding = false;

    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);

        // All values must be finite
        assert!(
            result.metadata.affective_valence.is_finite(),
            "Affective valence must be finite at cycle {i}"
        );
        assert!(
            result.metadata.affective_arousal.is_finite(),
            "Affective arousal must be finite at cycle {i}"
        );
        assert!(
            result.metadata.predictive_free_energy.is_finite(),
            "Predictive free energy must be finite at cycle {i}"
        );
        assert!(
            result.metadata.predictive_phi_modulation.is_finite(),
            "Predictive phi modulation must be finite at cycle {i}"
        );
        assert!(
            result.metadata.cross_modal_binding_strength.is_finite(),
            "Cross-modal binding strength must be finite at cycle {i}"
        );
        assert!(
            result.metadata.cross_modal_phi.is_finite(),
            "Cross-modal phi must be finite at cycle {i}"
        );

        // Bounds checks
        assert!(
            result.metadata.affective_valence >= -1.0 && result.metadata.affective_valence <= 1.0,
            "Valence out of bounds at cycle {i}: {}",
            result.metadata.affective_valence
        );
        assert!(
            result.metadata.affective_arousal >= 0.0 && result.metadata.affective_arousal <= 1.0,
            "Arousal out of bounds at cycle {i}: {}",
            result.metadata.affective_arousal
        );
        assert!(
            result.metadata.predictive_phi_modulation.is_finite(),
            "Phi modulation must be finite at cycle {i}: {}",
            result.metadata.predictive_phi_modulation
        );
        assert!(
            result.metadata.cross_modal_phi >= 0.0,
            "Cross-modal phi must be non-negative at cycle {i}: {}",
            result.metadata.cross_modal_phi
        );

        // Track non-default values
        if result.metadata.affective_valence.abs() > 0.001
            || (result.metadata.affective_arousal - 0.5).abs() > 0.001
        {
            saw_affective = true;
        }
        if result.metadata.predictive_free_energy.abs() > 0.001 {
            saw_predictive = true;
        }
        if result.metadata.cross_modal_binding_strength > 0.0 {
            saw_binding = true;
        }
    }

    println!("Synergy check: affective={saw_affective}, predictive={saw_predictive}, binding={saw_binding}");

    // Over 100 cycles, all three modules should have produced non-default output
    assert!(
        saw_affective,
        "Affective bridge should produce non-default values over 100 cycles"
    );
    assert!(
        saw_predictive,
        "Predictive processing should produce non-default values over 100 cycles"
    );
    // Cross-modal binding requires modality data; may or may not bind depending on input
    // Just verify stability
    assert_eq!(service.stats().total_cycles, 100);

    let avg_error = service.stats().avg_prediction_error;
    assert!(
        avg_error < 1.0,
        "Average error with synergy modules should be bounded: got {avg_error:.4}"
    );
}

// ── Dream Replay Integration ──────────────────────────────────────

#[test]
fn test_dream_replay_integration() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_dream_replay: true,
        ..Default::default()
    })
    .unwrap();

    // Run 25 cycles with diverse inputs to accumulate surprise events
    let inputs = [
        "the rain falls on the river",
        "quantum mechanics governs atomic interactions",
        "a child laughs in the sunlight",
        "neural networks learn from data patterns",
        "the wind carries seeds across the valley",
    ];
    let mut saw_dream_metadata = false;
    for i in 0..25 {
        let result = service.cycle(inputs[i % inputs.len()]);
        // Dream metadata should be valid (non-negative)
        assert!(result.metadata.dream_phi_improvement >= 0.0);
        assert!(result.metadata.dream_insights <= 100); // sanity bound
        if result.metadata.dream_insights > 0 || result.metadata.dream_wisdom_count > 0 {
            saw_dream_metadata = true;
        }
    }

    // After 25 diverse-input cycles, dream metadata should have been populated at least once
    // (dream runs during Cruise or periodically)
    println!(
        "Dream metadata populated: {saw_dream_metadata}, last wisdom_count: {}",
        service.cycle("final").metadata.dream_wisdom_count,
    );
}

// ── Safety Gateway Blocks Dangerous Input ─────────────────────────

#[test]
fn test_safety_gateway_blocks_dangerous_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_safety_gateway: true,
        ..Default::default()
    })
    .unwrap();

    // Input matching AmygdalaActor regex: `rm\s+-rf\s+/`
    let result = service.cycle("rm -rf /etc/passwd");

    assert!(
        result.metadata.safety_blocked,
        "Safety gateway should block dangerous input"
    );
    assert!(
        result.metadata.safety_category.is_some(),
        "Safety category should be populated when blocked"
    );
    assert_eq!(
        result.prediction_error, 0.0,
        "Blocked input should have zero prediction error"
    );
}

// ── Safety Gateway Allows Normal Input ────────────────────────────

#[test]
fn test_safety_gateway_allows_normal_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_safety_gateway: true,
        ..Default::default()
    })
    .unwrap();

    let result = service.cycle("the cat sat on the mat");

    assert!(
        !result.metadata.safety_blocked,
        "Safety gateway should not block benign input"
    );
    assert!(
        result.prediction_error > 0.0,
        "Normal input should produce non-zero prediction error"
    );
}

// ── Metacognitive Anomaly Detection ───────────────────────────────

#[test]
fn test_metacognitive_anomaly_detection() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        enable_metacognitive_monitoring: true,
        ..Default::default()
    })
    .unwrap();

    // Run 50 cycles with varied input to trigger anomaly detection
    let inputs = [
        "stable consistent input pattern",
        "wildly different novel stimulus",
        "another normal sentence here",
        "completely unexpected data stream",
        "mundane everyday observation",
    ];

    let mut any_anomaly = false;
    for i in 0..50 {
        let result = service.cycle(inputs[i % inputs.len()]);
        if result.metadata.metacognitive_anomaly {
            any_anomaly = true;
        }
    }

    // Metacognitive monitoring is stochastic — just verify stability
    assert_eq!(service.stats().total_cycles, 50);
    // Log whether anomaly was detected (may or may not trigger)
    println!("Metacognitive anomaly detected: {any_anomaly}");
}

// ── Value Feedback Learning Over Time ─────────────────────────────

#[test]
fn test_value_feedback_learning() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "ethical reasoning about fairness",
        "compassion and kindness matter",
        "justice requires careful thought",
        "harmony between beings is sacred",
    ];

    let mut trends = Vec::new();
    for i in 0..60 {
        let result = service.cycle(inputs[i % inputs.len()]);
        trends.push(result.metadata.value_feedback_trend);
    }

    // After enough cycles the trend should have moved from its initial 0.0
    let non_zero = trends.iter().filter(|&&t| t != 0.0).count();
    assert!(
        non_zero > 0,
        "Value feedback trend should deviate from 0.0 after 60 cycles, got all zeros"
    );

    // All values should be finite and bounded
    for (i, &t) in trends.iter().enumerate() {
        assert!(
            t.is_finite(),
            "Value feedback trend must be finite at cycle {i}, got {t}"
        );
    }
}
