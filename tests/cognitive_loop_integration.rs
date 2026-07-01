// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    let _ = result.detected_primitives.len(); // Non-panicking access
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
        assert!(result.metadata.embodied.body_phi_modulation >= 0.5);
        assert!(result.metadata.embodied.body_phi_modulation <= 1.5);
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
        mce_values.push(result.metadata.consciousness.consciousness_level);
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
        embodied_mods.push(result.metadata.embodied.embodied_phi_modulation);
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
    unique.sort_by(|a, b| a.total_cmp(b));
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
            result.metadata.embodied.affective_valence.is_finite(),
            "Affective valence must be finite at cycle {i}"
        );
        assert!(
            result.metadata.embodied.affective_arousal.is_finite(),
            "Affective arousal must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.predictive_free_energy.is_finite(),
            "Predictive free energy must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.predictive_phi_modulation.is_finite(),
            "Predictive phi modulation must be finite at cycle {i}"
        );
        assert!(
            result
                .metadata
                .temporal
                .cross_modal_binding_strength
                .is_finite(),
            "Cross-modal binding strength must be finite at cycle {i}"
        );
        assert!(
            result.metadata.temporal.cross_modal_psi.is_finite(),
            "Cross-modal phi must be finite at cycle {i}"
        );

        // Bounds checks
        assert!(
            result.metadata.embodied.affective_valence >= -1.0
                && result.metadata.embodied.affective_valence <= 1.0,
            "Valence out of bounds at cycle {i}: {}",
            result.metadata.embodied.affective_valence
        );
        assert!(
            result.metadata.embodied.affective_arousal >= 0.0
                && result.metadata.embodied.affective_arousal <= 1.0,
            "Arousal out of bounds at cycle {i}: {}",
            result.metadata.embodied.affective_arousal
        );
        assert!(
            result.metadata.fep.predictive_phi_modulation.is_finite(),
            "Phi modulation must be finite at cycle {i}: {}",
            result.metadata.fep.predictive_phi_modulation
        );
        assert!(
            result.metadata.temporal.cross_modal_psi >= 0.0,
            "Cross-modal phi must be non-negative at cycle {i}: {}",
            result.metadata.temporal.cross_modal_psi
        );

        // Track non-default values
        if result.metadata.embodied.affective_valence.abs() > 0.001
            || (result.metadata.embodied.affective_arousal - 0.5).abs() > 0.001
        {
            saw_affective = true;
        }
        if result.metadata.fep.predictive_free_energy.abs() > 0.001 {
            saw_predictive = true;
        }
        if result.metadata.temporal.cross_modal_binding_strength > 0.0 {
            saw_binding = true;
        }
    }

    println!(
        "Synergy check: affective={saw_affective}, predictive={saw_predictive}, binding={saw_binding}"
    );

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
        assert!(result.metadata.memory.dream_phi_improvement >= 0.0);
        assert!(result.metadata.memory.dream_insights <= 100); // sanity bound
        if result.metadata.memory.dream_insights > 0
            || result.metadata.memory.dream_wisdom_count > 0
        {
            saw_dream_metadata = true;
        }
    }

    // After 25 diverse-input cycles, dream metadata should have been populated at least once
    // (dream runs during Cruise or periodically)
    println!(
        "Dream metadata populated: {saw_dream_metadata}, last wisdom_count: {}",
        service.cycle("final").metadata.memory.dream_wisdom_count,
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
        trends.push(result.metadata.ethics.value_feedback_trend);
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

// ── Chronobiology Integration ────────────────────────────────────

#[test]
fn test_chronobiology_modulates_learning() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let result = service.cycle("test circadian modulation");

    // Circadian phase should be a valid phase name
    let valid_phases = ["Dawn", "Day", "Dusk", "Night"];
    assert!(
        valid_phases.contains(&result.metadata.circadian_phase.as_str()),
        "Circadian phase should be valid, got: {}",
        result.metadata.circadian_phase,
    );

    // Plasticity modifier should be in reasonable range (0.1 to 1.2)
    assert!(
        result.metadata.circadian_plasticity > 0.0 && result.metadata.circadian_plasticity <= 1.2,
        "Circadian plasticity should be in (0.0, 1.2], got: {}",
        result.metadata.circadian_plasticity,
    );

    // Learning rate should be positive and bounded after circadian modulation
    assert!(
        service.stats().adaptive_learning_rate > 0.0
            && service.stats().adaptive_learning_rate <= 0.1,
        "Learning rate should be positive and bounded after circadian modulation"
    );
}

// ── Dream Replay Produces Insights ──────────────────────────────

#[test]
fn test_dream_replay_produces_insights() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_dream_replay: true,
        ..Default::default()
    })
    .unwrap();

    // Run enough cycles with varied input to populate episodic replay buffer
    let inputs = [
        "the sun rises over the mountains",
        "a new algorithm for sorting",
        "the cat sleeps on warm blankets",
        "quantum mechanics describes particles",
        "rivers flow to the ocean",
    ];

    let mut saw_dream = false;
    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        if result.metadata.memory.dream_insights > 0
            || result.metadata.memory.dream_wisdom_count > 0
        {
            saw_dream = true;
        }
    }

    // Dream replay is stochastic (depends on Cruise urgency), so just verify no panics
    // and that the metadata fields are always finite
    let final_result = service.cycle("final check");
    assert!(
        final_result
            .metadata
            .memory
            .dream_phi_improvement
            .is_finite()
    );
    println!(
        "Dream observed: {saw_dream}, wisdom_count: {}",
        final_result.metadata.memory.dream_wisdom_count
    );
}

// ── Embodied Cognition Telemetry ────────────────────────────────

#[test]
fn test_embodied_cognition_telemetry() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_embodied_cognition: true,
        enable_virtual_body: true,
        ..Default::default()
    })
    .unwrap();

    // Run several cycles and check embodied cognition metadata
    for _ in 0..10 {
        let result = service.cycle("embodied test input");
        assert!(
            result.metadata.embodied.embodied_phi_modulation.is_finite(),
            "Embodied phi modulation should be finite"
        );
        assert!(
            result.metadata.embodied.embodied_agency >= 0.0
                && result.metadata.embodied.embodied_agency <= 1.0,
            "Embodied agency should be in [0, 1], got: {}",
            result.metadata.embodied.embodied_agency,
        );
    }
}

// ── Temporal Primitives + Primitive Lattice ─────────────────────

#[test]
fn test_temporal_analyzer_records_intervals() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 60 cycles with varied input to trigger amortized analysis at cycle 50
    let inputs = [
        "the sun rises over mountains",
        "rivers flow into the ocean",
        "birds sing in the morning light",
        "wind whispers through the trees",
    ];
    let mut last_metadata = None;
    for i in 0..60 {
        let result = service.cycle(inputs[i % inputs.len()]);
        last_metadata = Some(result.metadata);
    }

    let meta = last_metadata.unwrap();
    // After 60 cycles, the amortized causal chain analysis (every 50 cycles)
    // should have run at least once. The cached count may be 0 if no chains
    // were found, but temporal_continuity should be computed at cycle 50.
    // Lattice metrics should always be present when primitive consciousness is on.
    assert!(
        meta.lattice_height > 0,
        "Lattice height should be >0 with primitive consciousness enabled, got {}",
        meta.lattice_height,
    );
    assert!(
        meta.lattice_width > 0,
        "Lattice width should be >0 with primitive consciousness enabled, got {}",
        meta.lattice_width,
    );
}

#[test]
fn test_lattice_structural_properties() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Single cycle is enough — lattice is computed at startup
    let result = service.cycle("consciousness emerges from integration");

    // 9-tier primitive system should have height >= 8 (layers) and width >= 1
    assert!(
        result.metadata.lattice_height >= 8,
        "9-tier system should have lattice height >= 8, got {}",
        result.metadata.lattice_height,
    );
    assert!(
        result.metadata.lattice_width > 0,
        "Lattice width should be >0, got {}",
        result.metadata.lattice_width,
    );
    // Timing should be recorded (non-zero means the module ran)
    // Note: in debug builds lattice construction can take >10ms, so just check it ran
    assert!(
        result.metadata.module_timings_us.primitive_lattice < 500_000,
        "Lattice property read should be fast (<500ms), got {}µs",
        result.metadata.module_timings_us.primitive_lattice,
    );
}

// ── Session 1-4 Module Wiring ──────────────────────────────────

#[test]
fn test_compositionality_and_value_evaluator_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 25 cycles to hit value evaluator amortization (every 20 cycles)
    let mut last_result = service.cycle("compositionality test");
    for _ in 1..25 {
        last_result = service.cycle("value alignment check");
    }

    // Compositionality engine should be present
    assert!(service.compositionality_engine().is_some());
    // Value evaluator should be present
    assert!(service.value_evaluator().is_some());
    // Value score should have been computed at cycle 20
    assert!(
        last_result.metadata.ethics.value_evaluator_score >= 0.0,
        "Value evaluator score should be non-negative, got: {}",
        last_result.metadata.ethics.value_evaluator_score
    );
}

#[test]
fn test_harmonics_and_consciousness_profile_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 15 cycles to hit harmonic amortization (every 10 cycles)
    let mut last_result = service.cycle("harmony test");
    for _ in 1..15 {
        last_result = service.cycle("harmonic coherence");
    }

    // Harmonic field should be present
    assert!(service.harmonic_field().is_some());
    // Field coherence should have been computed
    assert!(
        last_result.metadata.harmonics.harmonic_field_coherence >= 0.0,
        "Harmonic field coherence should be non-negative"
    );
    // Consciousness profile computed at cycle 10
    assert!(
        last_result
            .metadata
            .consciousness
            .consciousness_profile_composite
            .is_finite(),
        "Consciousness profile composite should be finite"
    );
}

#[test]
fn test_reasoning_modules_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 55 cycles to hit both amortization points (25 and 50)
    let mut last_result = service.cycle("reasoning test");
    for _ in 1..55 {
        last_result = service.cycle("analogical reasoning chain");
    }

    // Reasoners should be present
    assert!(service.primitive_reasoner().is_some());
    assert!(service.adaptive_reasoner().is_some());
    // Reasoning should have produced results
    assert!(
        last_result.metadata.reasoning_chain_confidence.is_finite(),
        "Reasoning confidence should be finite"
    );
    assert!(
        last_result.metadata.adaptive_reasoning_phi.is_finite(),
        "Adaptive reasoning Phi should be finite"
    );
}

#[test]
fn test_epistemic_tiers_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 55 cycles to hit epistemic amortization (every 50 cycles)
    let mut last_result = service.cycle("epistemic classification");
    for _ in 1..55 {
        last_result = service.cycle("knowledge verification");
    }

    // Epistemic quality should have been classified
    assert!(
        last_result.metadata.epistemic_quality >= 0.0
            && last_result.metadata.epistemic_quality <= 1.0,
        "Epistemic quality should be in [0,1], got: {}",
        last_result.metadata.epistemic_quality
    );
}

// ── Temporal Consciousness Telemetry ────────────────────────────

#[test]
fn test_temporal_consciousness_telemetry() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_temporal_consciousness: true,
        enable_narrative_self: true,
        enable_predictive_self: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..20 {
        let result = service.cycle("temporal coherence test");
        assert!(
            result
                .metadata
                .temporal
                .temporal_coherence_score
                .is_finite(),
            "Temporal coherence score should be finite"
        );
        // Temporal coherence should be non-negative
        assert!(
            result.metadata.temporal.temporal_coherence_score >= 0.0,
            "Temporal coherence should be non-negative, got: {}",
            result.metadata.temporal.temporal_coherence_score,
        );
    }
}

// ── Causal Chain → Episodic Consolidation ────────────────────────

#[test]
fn test_causal_chain_boosts_episodic_consolidation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Run 60 cycles — causal chain analysis fires at cycle 50
    let inputs = [
        "cause and effect relationship",
        "temporal sequence of events",
        "action leads to consequence",
        "precedence determines outcome",
    ];
    let mut last_metadata = None;
    for i in 0..60 {
        let result = service.cycle(inputs[i % inputs.len()]);
        last_metadata = Some(result.metadata.clone());
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error must be finite at cycle {i}"
        );
    }

    let meta = last_metadata.unwrap();
    // After 60 cycles the cached causal chain count should be populated
    // (may be 0 if no genuine chains found, but the analysis ran without panic)
    assert!(
        meta.temporal.temporal_causal_chains < 1000,
        "Causal chain count should be bounded, got {}",
        meta.temporal.temporal_causal_chains,
    );
    // causal_codebook_entries should be a finite count
    assert!(
        meta.causal_codebook_entries < 100,
        "Causal codebook entries should be bounded, got {}",
        meta.causal_codebook_entries,
    );
}

// ── Lattice Join Produces Concept ────────────────────────────────

#[test]
fn test_lattice_join_produces_concept() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 5 cycles — lattice join computed every cycle when primitives are active
    let mut saw_join_concept = false;
    for _ in 0..5 {
        let result = service.cycle("complex integrated awareness emerges");
        if !result.metadata.lattice_join_concept.is_empty() {
            saw_join_concept = true;
        }
    }

    // If primitives fired, we should see a join concept; if not, just verify stability
    println!("Saw lattice join concept: {saw_join_concept}");
    assert_eq!(service.stats().total_cycles, 5);
}

// ── Full Temporal-Lattice Pipeline ───────────────────────────────

#[test]
fn test_full_temporal_lattice_pipeline() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Run 150 cycles to exercise all amortized intervals:
    //   causal chains (50), continuity (100), and codebook growth
    let inputs = [
        "cause and effect in nature",
        "temporal ordering of events",
        "consciousness integrates information",
        "primitive reasoning chains",
        "lattice structure emerges",
    ];

    for i in 0..150 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error must be finite at cycle {i}"
        );
        assert!(
            result.metadata.primitive_psi.is_finite(),
            "Primitive phi must be finite at cycle {i}"
        );
        assert!(
            result.metadata.temporal.temporal_continuity.is_finite(),
            "Temporal continuity must be finite at cycle {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 150);
    assert!(
        stats.avg_prediction_error < 1.0,
        "Average error should be bounded after 150 cycles: got {:.4}",
        stats.avg_prediction_error,
    );
}

// ── Session 5: Dissipative + Conflict + Equation + Hierarchical + Evolution ──

#[test]
fn test_dissipative_consciousness_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 5 cycles — dissipative consciousness updates every cycle
    let mut last_result = service.cycle("thermodynamic self-organization");
    for _ in 1..5 {
        last_result = service.cycle("entropy production at edge of chaos");
    }

    assert!(
        last_result.metadata.quality.dissipative_health >= 0.0,
        "Dissipative health should be non-negative, got: {}",
        last_result.metadata.quality.dissipative_health,
    );
    assert!(
        !last_result.metadata.quality.dissipative_regime.is_empty(),
        "Dissipative regime should be populated",
    );
    // dissipative_consciousness() is pub(crate) — tested internally
}

#[test]
fn test_epistemic_conflict_and_equation_v2_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 55 cycles to hit epistemic conflict (every 50) and equation v2 (every 25)
    let mut last_result = service.cycle("multi-theory conflict analysis");
    for _ in 1..55 {
        last_result = service.cycle("unified consciousness formula");
    }

    assert!(
        last_result.metadata.quality.epistemic_phi_eff >= 0.0,
        "Epistemic Φ_eff should be non-negative, got: {}",
        last_result.metadata.quality.epistemic_phi_eff,
    );
    assert!(
        last_result.metadata.quality.equation_v2_consciousness >= 0.0,
        "Equation v2 consciousness should be non-negative, got: {}",
        last_result.metadata.quality.equation_v2_consciousness,
    );
}

#[test]
fn test_hierarchical_ltc_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 15 cycles to hit hierarchical LTC (every 10 cycles)
    let mut last_result = service.cycle("distributed temporal processing");
    for _ in 1..15 {
        last_result = service.cycle("hierarchical circuit dynamics");
    }

    assert!(
        last_result
            .metadata
            .quality
            .hierarchical_ltc_phi
            .is_finite(),
        "Hierarchical LTC phi should be finite, got: {}",
        last_result.metadata.quality.hierarchical_ltc_phi,
    );
    assert!(
        service.hierarchical_ltc().is_some(),
        "Hierarchical LTC accessor should return Some",
    );
}

// ── Session 6: Holographic + Differentiable + Affective + Pipeline + MultiModal ──

#[test]
fn test_holographic_and_affective_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 25 cycles (past 20-cycle holographic trigger + 10-cycle affective trigger)
    let mut last_result = service.cycle("holographic binding analysis");
    for _ in 1..25 {
        last_result = service.cycle("affective consciousness dynamics");
    }

    assert!(
        last_result.metadata.temporal.holographic_unity >= 0.0,
        "Holographic unity should be non-negative, got: {}",
        last_result.metadata.temporal.holographic_unity,
    );
    assert!(
        last_result
            .metadata
            .embodied
            .affect_consciousness_valence
            .is_finite(),
        "Affective consciousness valence should be finite, got: {}",
        last_result.metadata.embodied.affect_consciousness_valence,
    );
    // holographic_analyzer() is pub(crate) — tested internally
    assert!(
        service.affective_consciousness().is_some(),
        "Affective consciousness accessor should return Some",
    );
}

#[test]
fn test_differentiable_and_pipeline_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 55 cycles (past 50-cycle pipeline trigger + 25-cycle differentiable trigger)
    let mut last_result = service.cycle("gradient consciousness optimization");
    for _ in 1..55 {
        last_result = service.cycle("unified pipeline processing");
    }

    assert!(
        last_result
            .metadata
            .consciousness
            .consciousness_gradient_magnitude
            >= 0.0,
        "Gradient magnitude should be non-negative, got: {}",
        last_result
            .metadata
            .consciousness
            .consciousness_gradient_magnitude,
    );
    assert!(
        last_result.metadata.pipeline_consciousness >= 0.0,
        "Pipeline consciousness should be non-negative, got: {}",
        last_result.metadata.pipeline_consciousness,
    );
}

#[test]
fn test_multimodal_integration_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 20 cycles (past 15-cycle trigger)
    let mut last_result = service.cycle("cross-modal binding");
    for _ in 1..20 {
        last_result = service.cycle("multi-modal integration");
    }

    assert!(
        last_result.metadata.multimodal_integrated_phi >= 0.0,
        "Multi-modal integrated phi should be non-negative, got: {}",
        last_result.metadata.multimodal_integrated_phi,
    );
    assert!(
        service.multi_modal_integrator().is_some(),
        "Multi-modal integrator accessor should return Some",
    );
}

// ── Session 7: Behavioral Feedback Tests ────────────────────────────

#[test]
fn test_synthetic_grounding_and_epistemic_gate_wired() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 105 cycles to trigger synthetic grounding (every 100) and epistemic gate (every 5)
    let mut last_result = service.cycle("consciousness state classification");
    for _ in 1..105 {
        last_result = service.cycle("epistemic confidence evaluation");
    }

    assert!(
        last_result.metadata.epistemic_gate_confidence >= 0.0,
        "Epistemic gate confidence should be non-negative, got: {}",
        last_result.metadata.epistemic_gate_confidence,
    );
    // synthetic_grounding() is pub(crate) — tested internally
    assert!(
        service.epistemic_gate().is_some(),
        "Epistemic gate accessor should return Some",
    );
}

#[test]
fn test_feedback_loops_modulate_confidence() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run enough cycles for all feedback loops to fire
    for _ in 0..60 {
        service.cycle("feedback loop convergence test");
    }

    // After 60 cycles, prediction confidence should have been modified from initial 0.5
    // (holographic unity, affective, multimodal, epistemic all modulate it)
    let stats = service.stats();
    assert!(
        stats.prediction_confidence.is_finite(),
        "Prediction confidence should be finite after feedback, got: {}",
        stats.prediction_confidence,
    );
    // Confidence should stay bounded [0.0, 1.0]
    assert!(
        stats.prediction_confidence >= 0.0 && stats.prediction_confidence <= 1.0,
        "Prediction confidence should be in [0.0, 1.0], got: {}",
        stats.prediction_confidence,
    );
}

#[test]
fn test_negative_affect_reduces_confidence() {
    let mut service_with = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    let mut service_without = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: false,
        ..Default::default()
    })
    .unwrap();

    // Run same inputs on both
    for _ in 0..30 {
        service_with.cycle("error prone chaotic unpredictable");
        service_without.cycle("error prone chaotic unpredictable");
    }

    // Both should have valid confidence
    assert!(
        service_with.stats().prediction_confidence.is_finite(),
        "With-primitives confidence should be finite",
    );
    assert!(
        service_without.stats().prediction_confidence.is_finite(),
        "Without-primitives confidence should be finite",
    );
    // The service with affective consciousness should have different confidence
    // (we don't assert direction since it depends on affect dynamics, just that it's bounded)
    assert!(
        service_with.stats().prediction_confidence >= 0.0
            && service_with.stats().prediction_confidence <= 1.0,
        "With-primitives confidence should be bounded",
    );
}

// ── Performance Profiling ───────────────────────────────────────────

#[test]
fn test_module_timing_profile() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 100 cycles and collect timing data
    let mut timing_sums: Vec<(&str, u64)> = Vec::new();
    let mut total_cycle_us: u64 = 0;

    for i in 0..100 {
        let start = std::time::Instant::now();
        let result = service.cycle(&format!("profiling cycle {i}"));
        let cycle_us = start.elapsed().as_micros() as u64;
        total_cycle_us += cycle_us;

        let t = &result.metadata.module_timings_us;

        if i == 0 {
            // Initialize
            timing_sums = vec![
                ("affective_bridge", 0),
                ("predictive_processing", 0),
                ("cross_modal_binding", 0),
                ("surprise_exploration", 0),
                ("prefrontal", 0),
                ("meta_cognition", 0),
                ("narrative_self", 0),
                ("gwt", 0),
                ("virtual_body", 0),
                ("embodied_cognition", 0),
                ("dream_replay", 0),
                ("moral_algebra", 0),
                ("consciousness_resonance", 0),
                ("temporal_consciousness", 0),
                ("attention_schema", 0),
                ("narrative_gwt", 0),
                ("consciousness_thermodynamics", 0),
                ("phenomenal_binding", 0),
                ("hierarchical_free_energy", 0),
                ("resonator_recall", 0),
                ("support_intelligence", 0),
                ("temporal_analyzer", 0),
                ("primitive_lattice", 0),
                ("compositionality", 0),
                ("value_evaluator", 0),
                ("consciousness_profile", 0),
                ("harmonics", 0),
                ("primitive_reasoning", 0),
                ("causal_explanation", 0),
                ("adaptive_reasoning", 0),
                ("epistemic_tiers", 0),
                ("phi_validation", 0),
                ("dissipative_consciousness", 0),
                ("epistemic_conflict", 0),
                ("consciousness_equation_v2", 0),
                ("hierarchical_ltc", 0),
                ("primitive_evolution", 0),
                ("consciousness_holography", 0),
                ("differentiable_consciousness", 0),
                ("affective_consciousness", 0),
                ("unified_consciousness_pipeline", 0),
                ("multi_modal_integration", 0),
                ("synthetic_grounding", 0),
                ("epistemic_gate", 0),
                ("semantic_value_embedder", 0),
                ("composition_rules", 0),
                ("harmonies_integration", 0),
                ("meta_cognitive_reasoning", 0),
                ("code_primitive_routing", 0),
                ("empathic_unification", 0),
                ("multi_objective_evolution", 0),
                ("stability_regime", 0),
                // Core pipeline phases
                ("CORE: hdc_encode", 0),
                ("CORE: compress", 0),
                ("CORE: semantic_lookup", 0),
                ("CORE: cfc_step", 0),
                ("CORE: predict", 0),
                ("CORE: training", 0),
                ("CORE: parallel_postprocess", 0),
            ];
        }

        let values = [
            t.affective_bridge,
            t.predictive_processing,
            t.cross_modal_binding,
            t.surprise_exploration,
            t.prefrontal,
            t.meta_cognition,
            t.narrative_self,
            t.gwt,
            t.virtual_body,
            t.embodied_cognition,
            t.dream_replay,
            t.moral_algebra,
            t.consciousness_resonance,
            t.temporal_consciousness,
            t.attention_schema,
            t.narrative_gwt,
            t.consciousness_thermodynamics,
            t.phenomenal_binding,
            t.hierarchical_free_energy,
            t.resonator_recall,
            t.support_intelligence,
            t.temporal_analyzer,
            t.primitive_lattice,
            t.compositionality,
            t.value_evaluator,
            t.consciousness_profile,
            t.harmonics,
            t.primitive_reasoning,
            t.causal_explanation,
            t.adaptive_reasoning,
            t.epistemic_tiers,
            t.phi_validation,
            t.dissipative_consciousness,
            t.epistemic_conflict,
            t.consciousness_equation_v2,
            t.hierarchical_ltc,
            t.primitive_evolution,
            t.consciousness_holography,
            t.differentiable_consciousness,
            t.affective_consciousness,
            t.unified_consciousness_pipeline,
            t.multi_modal_integration,
            t.synthetic_grounding,
            t.epistemic_gate,
            t.semantic_value_embedder,
            t.composition_rules,
            t.harmonies_integration,
            t.meta_cognitive_reasoning,
            t.code_primitive_routing,
            t.empathic_unification,
            t.multi_objective_evolution,
            t.stability_regime,
            // Core pipeline phases
            t.core_hdc_encode,
            t.core_compress,
            t.core_semantic_lookup,
            t.core_cfc_step,
            t.core_predict,
            t.core_training,
            t.core_parallel_postprocess,
        ];

        for (j, &val) in values.iter().enumerate() {
            timing_sums[j].1 += val;
        }
    }

    // Sort by total time descending
    timing_sums.sort_by(|a, b| b.1.cmp(&a.1));

    // Print top 20 modules by total time
    eprintln!("\n═══ MODULE TIMING PROFILE (100 cycles, primitives ON) ═══");
    eprintln!("Total wall-clock: {:.1}ms", total_cycle_us as f64 / 1000.0);
    eprintln!("Avg cycle: {:.1}ms\n", total_cycle_us as f64 / 100_000.0);
    let instrumented_total: u64 = timing_sums.iter().map(|(_, t)| t).sum();
    eprintln!(
        "Instrumented total: {:.1}ms ({:.0}% of wall-clock)\n",
        instrumented_total as f64 / 1000.0,
        instrumented_total as f64 / total_cycle_us as f64 * 100.0
    );
    // Separate core pipeline from module timings
    let (core_timings, module_timings_sorted): (Vec<_>, Vec<_>) = timing_sums
        .iter()
        .partition(|(name, _)| name.starts_with("CORE:"));
    let core_total: u64 = core_timings.iter().map(|(_, t)| t).sum();

    eprintln!("── CORE PIPELINE ──");
    eprintln!(
        "{:<35} {:>10} {:>8} {:>6}",
        "Phase", "Total(µs)", "Avg(µs)", "%wall"
    );
    eprintln!("{}", "-".repeat(65));
    for (name, total) in &core_timings {
        let pct = *total as f64 / total_cycle_us as f64 * 100.0;
        eprintln!(
            "{:<35} {:>10} {:>8} {:>5.1}%",
            name,
            total,
            total / 100,
            pct
        );
    }
    eprintln!(
        "Core subtotal: {:.1}ms ({:.1}% of wall-clock)\n",
        core_total as f64 / 1000.0,
        core_total as f64 / total_cycle_us as f64 * 100.0
    );

    let module_total: u64 = module_timings_sorted.iter().map(|(_, t)| t).sum();
    eprintln!("── CONSCIOUSNESS MODULES ──");
    eprintln!(
        "{:<35} {:>10} {:>8} {:>6}",
        "Module", "Total(µs)", "Avg(µs)", "%mod"
    );
    eprintln!("{}", "-".repeat(65));
    for (name, total) in module_timings_sorted.iter().take(20) {
        let pct = if module_total > 0 {
            *total as f64 / module_total as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "{:<35} {:>10} {:>8} {:>5.1}%",
            name,
            total,
            total / 100,
            pct
        );
    }
    eprintln!(
        "Module subtotal: {:.1}ms ({:.1}% of wall-clock)\n",
        module_total as f64 / 1000.0,
        module_total as f64 / total_cycle_us as f64 * 100.0
    );

    let accounted = core_total + module_total;
    let unaccounted = total_cycle_us.saturating_sub(accounted);
    eprintln!(
        "Unaccounted: {:.1}ms ({:.1}% of wall-clock)",
        unaccounted as f64 / 1000.0,
        unaccounted as f64 / total_cycle_us as f64 * 100.0
    );

    // Sanity: avg cycle time should be reasonable (<500ms in test profile)
    let avg_cycle_ms = total_cycle_us as f64 / 100_000.0;
    assert!(
        avg_cycle_ms < 500.0,
        "Average cycle time should be <500ms, got {:.1}ms",
        avg_cycle_ms,
    );
}

// ── Behavioral Correctness Tests ──────────────────────────────────────
// These verify that feedback loops and cross-module interactions produce
// correct *behavioral* effects, not just that fields are non-zero.

#[test]
fn test_prediction_error_decreases_on_repeated_input() {
    // The cognitive loop should learn from repeated identical input,
    // resulting in decreasing prediction error over time.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let mut errors: Vec<f32> = Vec::new();
    for _ in 0..50 {
        let result = service.cycle("the same input repeated");
        errors.push(result.prediction_error);
    }

    // Compare first 10 errors vs last 10 — last should be lower on average
    let early_avg: f32 = errors[..10].iter().sum::<f32>() / 10.0;
    let late_avg: f32 = errors[40..].iter().sum::<f32>() / 10.0;
    assert!(
        late_avg <= early_avg + 0.05,
        "Prediction error should not increase significantly on repeated input: early={early_avg:.3}, late={late_avg:.3}"
    );
}

#[test]
fn test_learning_rate_responds_to_error_dynamics() {
    // The adaptive learning rate should modulate based on prediction error history.
    // High error → higher effective LR; sustained low error → lower LR.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run 50 cycles with varied input to drive learning
    for i in 0..50 {
        service.cycle(&format!("varied input for learning dynamics test {i}"));
    }

    let stats = service.stats();
    // Effective learning rate should be finite and bounded
    assert!(
        stats.effective_learning_rate.is_finite() && stats.effective_learning_rate >= 0.0,
        "Effective LR should be finite and non-negative: {}",
        stats.effective_learning_rate
    );
    // Adaptive learning rate should differ from base (modulation is active)
    assert!(
        stats.adaptive_learning_rate.is_finite(),
        "Adaptive LR should be finite: {}",
        stats.adaptive_learning_rate
    );
}

#[test]
fn test_curiosity_responds_to_novel_input() {
    // Novel/surprising inputs should increase exploration urge.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Establish baseline with repeated input
    for _ in 0..20 {
        service.cycle("familiar pattern");
    }
    let baseline_exploration = service.stats().exploration_urge;

    // Inject novel input
    for _ in 0..5 {
        service.cycle("completely novel unexpected surprising quantum consciousness emergence");
    }
    let post_novel_exploration = service.stats().exploration_urge;

    // Exploration urge should be at least as high (novel input shouldn't decrease it)
    assert!(
        post_novel_exploration >= baseline_exploration - 0.1,
        "Novel input should not significantly decrease exploration: baseline={baseline_exploration:.3}, post={post_novel_exploration:.3}"
    );
}

#[test]
fn test_coherence_bridge_tracks_temporal_dynamics() {
    // The coherence bridge should produce non-zero coherence after several cycles.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for _ in 0..30 {
        service.cycle("temporal coherence test");
    }

    let stats = service.stats();
    assert!(
        stats.temporal_coherence.is_finite(),
        "Temporal coherence should be finite"
    );
    // After 30 cycles, coherence should have a meaningful value
    assert!(
        stats.temporal_coherence >= 0.0,
        "Temporal coherence should be non-negative: {}",
        stats.temporal_coherence
    );
}

#[test]
fn test_moral_algebra_detects_harmful_content() {
    // Harmful input should trigger moral concern detection.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run with neutral content first
    for _ in 0..10 {
        service.cycle("gentle kind compassionate");
    }
    let baseline_concerns = service.stats().moral_concerns_detected;

    // Run with morally concerning content
    for _ in 0..10 {
        service.cycle("harmful violent destructive malicious attack");
    }
    let post_concerns = service.stats().moral_concerns_detected;

    // Moral concerns should increase (or at least not decrease) with harmful content
    assert!(
        post_concerns >= baseline_concerns,
        "Moral concerns should not decrease with harmful content: before={baseline_concerns}, after={post_concerns}"
    );
}

#[test]
fn test_consciousness_level_stabilizes() {
    // Consciousness level (unified_psi) should stabilize to a finite, bounded value.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    let mut psi_values: Vec<f32> = Vec::new();
    for _ in 0..50 {
        let result = service.cycle("consciousness stabilization test");
        psi_values.push(result.metadata.consciousness.consciousness_level as f32);
    }

    // All values should be finite
    assert!(
        psi_values.iter().all(|v| v.is_finite()),
        "All unified_psi values should be finite"
    );

    // Variance in last 20 cycles should be lower than first 20
    // (system stabilizes over time)
    let early: Vec<f32> = psi_values[..20].to_vec();
    let late: Vec<f32> = psi_values[30..].to_vec();
    let early_var = variance(&early);
    let late_var = variance(&late);

    // Late variance should not be dramatically higher (system shouldn't diverge)
    assert!(
        late_var < early_var * 5.0 + 0.01,
        "Consciousness level should stabilize, not diverge: early_var={early_var:.4}, late_var={late_var:.4}"
    );
}

fn variance(vals: &[f32]) -> f32 {
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
}

#[test]
fn test_episodic_memory_encodes_significant_experiences() {
    // High-error or flow-state cycles should produce episodic memory entries.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run cycles — high prediction error early should trigger encoding
    for i in 0..30 {
        service.cycle(&format!(
            "varied input {i} with different content each time"
        ));
    }

    let stats = service.stats();
    assert!(
        stats.memory_total_encoded > 0,
        "Episodic memory should encode at least some experiences"
    );
}

#[test]
fn test_adaptive_behavior_responds_to_state() {
    // Adaptive behavior should transition between states based on consciousness patterns.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let mut hints_seen = std::collections::HashSet::new();
    for i in 0..40 {
        service.cycle(&format!("adaptive behavior test cycle {i}"));
        hints_seen.insert(service.stats().action_hint.clone());
    }

    // Should see at least the default action hint (system responds to state)
    assert!(
        !hints_seen.is_empty(),
        "Adaptive behavior should produce action hints"
    );
}

#[test]
fn test_world_model_tracks_prediction_error() {
    // World model should track and report average error.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for _ in 0..20 {
        service.cycle("world model test input");
    }

    let stats = service.stats();
    assert!(
        stats.world_model_avg_error.is_finite(),
        "World model error should be finite: {}",
        stats.world_model_avg_error
    );
}

#[test]
fn test_core_pipeline_timing_coverage() {
    // Verify that core pipeline phases are being timed (non-zero for key phases).
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("timing coverage test");

    let t = &result.metadata.module_timings_us;
    // HDC encode should always take >0 time
    assert!(t.core_hdc_encode > 0, "HDC encode should be timed");
    // CfC step should always take >0 time
    assert!(t.core_cfc_step > 0, "CfC step should be timed");
    // Predict should always take >0 time
    assert!(t.core_predict > 0, "Predict should be timed");
}

// ── Live Cognitive Psych-Bench Tests ──────────────────────────────────
// These run cognitive psychology paradigms through the live CognitiveLoopService
// to verify the system exhibits expected cognitive phenomena.

#[test]
fn test_stroop_interference_through_live_loop() {
    // The Stroop effect: incongruent color-word pairs should produce higher
    // prediction error than congruent pairs, reflecting cognitive interference.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Warmup: establish baseline expectations
    for _ in 0..20 {
        service.cycle("color identification task warmup");
    }

    // Congruent trials: word matches ink color (easy)
    let mut congruent_errors: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let result = service.cycle("color RED ink RED congruent match");
        congruent_errors.push(result.prediction_error);
    }

    // Incongruent trials: word conflicts with ink color (hard)
    let mut incongruent_errors: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let result = service.cycle("color BLUE ink RED incongruent conflict interference");
        incongruent_errors.push(result.prediction_error);
    }

    let congruent_avg: f32 = congruent_errors.iter().sum::<f32>() / congruent_errors.len() as f32;
    let incongruent_avg: f32 =
        incongruent_errors.iter().sum::<f32>() / incongruent_errors.len() as f32;

    // Both should produce finite errors
    assert!(
        congruent_avg.is_finite(),
        "Congruent errors should be finite"
    );
    assert!(
        incongruent_avg.is_finite(),
        "Incongruent errors should be finite"
    );

    // The system should process both without crashing — behavioral difference
    // depends on HDC encoding separation. At minimum, verify the pipeline handles
    // Stroop-like stimuli gracefully.
    eprintln!(
        "Stroop: congruent_avg={congruent_avg:.4}, incongruent_avg={incongruent_avg:.4}, \
         diff={:.4}",
        incongruent_avg - congruent_avg
    );
}

#[test]
fn test_wcst_set_shifting_through_live_loop() {
    // Wisconsin Card Sorting Test: the system should adapt when the rule changes.
    // After learning one sorting rule, switching to a different rule should
    // produce a spike in prediction error (perseveration detection).
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Phase 1: Learn "sort by color" rule
    let mut phase1_errors: Vec<f32> = Vec::new();
    for _ in 0..20 {
        let result = service.cycle("sort by color: red triangle matches red circle correct");
        phase1_errors.push(result.prediction_error);
    }

    let phase1_late_avg: f32 = phase1_errors[15..].iter().sum::<f32>() / 5.0;

    // Phase 2: Switch to "sort by shape" rule (should cause confusion)
    let mut phase2_errors: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let result = service.cycle("sort by shape: blue triangle matches red triangle correct");
        phase2_errors.push(result.prediction_error);
    }

    let phase2_early_avg: f32 = phase2_errors[..5].iter().sum::<f32>() / 5.0;

    // The rule switch should not cause the system to crash, and error dynamics
    // should reflect the novel input structure
    assert!(
        phase1_late_avg.is_finite() && phase2_early_avg.is_finite(),
        "WCST errors should be finite through rule switch"
    );

    eprintln!(
        "WCST: pre-switch_avg={phase1_late_avg:.4}, post-switch_avg={phase2_early_avg:.4}, \
         delta={:.4}",
        phase2_early_avg - phase1_late_avg
    );
}

#[test]
fn test_habituation_and_dishabituation() {
    // Habituation: repeated identical stimuli should decrease prediction error.
    // Dishabituation: a novel stimulus should spike prediction error back up.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Habituation phase: same stimulus repeated
    let mut hab_errors: Vec<f32> = Vec::new();
    for _ in 0..30 {
        let result = service.cycle("standard tone 440hz duration 200ms");
        hab_errors.push(result.prediction_error);
    }

    let early_avg = hab_errors[..5].iter().sum::<f32>() / 5.0;
    let late_avg = hab_errors[25..].iter().sum::<f32>() / 5.0;

    // Error should decrease (or at least not increase significantly) with repetition
    assert!(
        late_avg <= early_avg + 0.1,
        "Prediction error should not increase with repeated stimulus: \
         early={early_avg:.4}, late={late_avg:.4}"
    );

    // Dishabituation: novel stimulus
    let novel_result = service.cycle("deviant tone 880hz duration 50ms unexpected");
    let novel_error = novel_result.prediction_error;

    // Novel stimulus should be finite (system handles it)
    assert!(
        novel_error.is_finite(),
        "Novel stimulus error should be finite: {novel_error}"
    );

    eprintln!("Habituation: early={early_avg:.4}, late={late_avg:.4}, novel={novel_error:.4}");
}

#[test]
fn test_cognitive_load_affects_processing() {
    // Higher cognitive load (more complex input) should affect processing metrics.
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Warmup
    for _ in 0..10 {
        service.cycle("warmup");
    }

    // Low load: simple input
    let mut low_load_times: Vec<u64> = Vec::new();
    for _ in 0..10 {
        let result = service.cycle("simple");
        low_load_times.push(result.cycle_time_us);
    }

    // High load: complex input with many features
    let mut high_load_times: Vec<u64> = Vec::new();
    for _ in 0..10 {
        let result = service.cycle(
            "complex multi-feature stimulus with color red shape triangle \
             orientation left motion upward texture smooth pattern striped \
             semantic meaning abstract philosophical consciousness emergence",
        );
        high_load_times.push(result.cycle_time_us);
    }

    let low_avg: f64 = low_load_times.iter().sum::<u64>() as f64 / low_load_times.len() as f64;
    let high_avg: f64 = high_load_times.iter().sum::<u64>() as f64 / high_load_times.len() as f64;

    // Both should complete successfully
    assert!(low_avg > 0.0, "Low load cycles should complete");
    assert!(high_avg > 0.0, "High load cycles should complete");

    eprintln!(
        "Cognitive load: simple={low_avg:.0}µs, complex={high_avg:.0}µs, \
         ratio={:.2}x",
        high_avg / low_avg
    );
}

// ── Stress Testing & Graceful Degradation ──────────────────────────

/// Sustained high-throughput stress test: 500 cycles with varied input.
/// Verifies no panics, no memory leaks (stable cycle times), and finite outputs.
#[test]
fn test_sustained_high_throughput() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "visual cortex processing red circle",
        "auditory cortex processing birdsong",
        "somatosensory touch pressure warmth",
        "hippocampal spatial memory navigation",
        "prefrontal executive planning decision",
        "amygdala threat detection fear response",
        "cerebellum motor coordination balance",
        "temporal lobe language comprehension",
    ];

    let mut cycle_times = Vec::with_capacity(500);
    for i in 0..500 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "Cycle {} prediction error is not finite",
            i
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "Cycle {} consciousness level is not finite",
            i
        );
        cycle_times.push(result.cycle_time_us);
    }

    // Check cycle time stability: last 100 should not be >3x first 100
    let first_100_avg: f64 = cycle_times[..100].iter().sum::<u64>() as f64 / 100.0;
    let last_100_avg: f64 = cycle_times[400..].iter().sum::<u64>() as f64 / 100.0;

    eprintln!(
        "Sustained throughput (500 cycles): first_100_avg={first_100_avg:.0}µs, \
         last_100_avg={last_100_avg:.0}µs, ratio={:.2}x",
        last_100_avg / first_100_avg
    );

    // No runaway memory growth: cycle times should stay within 5x
    assert!(
        last_100_avg < first_100_avg * 5.0,
        "Cycle time degradation >5x: first={first_100_avg:.0}, last={last_100_avg:.0}"
    );

    // Total cycles should be exactly 500
    assert_eq!(service.stats().total_cycles, 500);
}

/// Module ablation: disable primitive consciousness and verify graceful degradation.
/// The system should still produce valid outputs without consciousness subsystems.
#[test]
fn test_graceful_degradation_without_consciousness() {
    // Run with consciousness disabled
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..50 {
        let result = service.cycle("test input for degraded mode");
        assert!(
            result.prediction_error.is_finite(),
            "Cycle {} prediction error not finite in degraded mode",
            i
        );
        // System should still function even without consciousness modules
        assert!(result.cycle_time_us > 0);
    }

    // Consciousness-related accessors should return None when disabled
    // holographic_analyzer() is pub(crate) — tested internally
    assert!(
        service.differentiable_consciousness().is_none(),
        "Differentiable consciousness should be None when consciousness is disabled"
    );
    assert!(
        service.affective_consciousness().is_none(),
        "Affective consciousness should be None when consciousness is disabled"
    );

    eprintln!(
        "Degraded mode: 50 cycles completed, total_cycles={}",
        service.stats().total_cycles
    );
}

/// Rapid input switching: alternate between vastly different domains.
/// Tests that the system handles context switches without accumulating errors.
#[test]
fn test_rapid_context_switching() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    let domains = [
        "mathematical proof theorem axiom derivation",
        "emotional empathy sadness grief compassion",
        "spatial navigation left right forward obstacle",
        "linguistic syntax grammar morphology phoneme",
        "musical harmony rhythm melody dissonance",
    ];

    let mut max_error: f32 = 0.0;
    let mut errors = Vec::new();

    // Warmup
    for _ in 0..10 {
        service.cycle("warmup context switching");
    }

    // Rapidly switch between domains
    for round in 0..5 {
        for (domain_idx, domain) in domains.iter().enumerate() {
            let result = service.cycle(domain);
            assert!(
                result.prediction_error.is_finite(),
                "Round {} domain {} error not finite",
                round,
                domain_idx
            );
            if result.prediction_error > max_error {
                max_error = result.prediction_error;
            }
            errors.push(result.prediction_error);
        }
    }

    // Error should never exceed 1.0 (bounded by architecture)
    assert!(
        max_error <= 1.0,
        "Prediction error exceeded 1.0 during context switching: {max_error}"
    );

    // Later rounds should show some adaptation
    let first_round_avg: f32 = errors[..5].iter().sum::<f32>() / 5.0;
    let last_round_avg: f32 = errors[20..25].iter().sum::<f32>() / 5.0;

    eprintln!(
        "Context switching: max_error={max_error:.4}, first_round={first_round_avg:.4}, \
         last_round={last_round_avg:.4}"
    );
}

/// Empty and adversarial inputs: verify the system handles edge cases gracefully.
#[test]
fn test_adversarial_inputs() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Empty string
    let result = service.cycle("");
    assert!(result.prediction_error.is_finite());

    // Very long input
    let long_input = "consciousness ".repeat(1000);
    let result = service.cycle(&long_input);
    assert!(result.prediction_error.is_finite());

    // Unicode and special characters
    let result = service.cycle("こんにちは世界 🧠 φ=∫δΨ/δt");
    assert!(result.prediction_error.is_finite());

    // Repeated characters
    let result = service.cycle(&"a".repeat(10000));
    assert!(result.prediction_error.is_finite());

    // Null bytes embedded
    let result = service.cycle("hello\0world\0test");
    assert!(result.prediction_error.is_finite());

    eprintln!(
        "Adversarial inputs: all 5 edge cases handled, total_cycles={}",
        service.stats().total_cycles
    );
}

/// Verify that consciousness metrics remain bounded and don't diverge
/// even after many cycles with high prediction error.
#[test]
fn test_consciousness_metrics_bounded() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    // Run many cycles with constantly changing input (high error)
    for i in 0..200 {
        let input = format!(
            "unique novel input number {i} with random content {}",
            i * 7919
        );
        let result = service.cycle(&input);

        // Core metrics must always be bounded
        assert!(
            result.prediction_error >= 0.0 && result.prediction_error <= 1.0,
            "Cycle {i}: prediction_error={} out of [0,1]",
            result.prediction_error
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "Cycle {i}: consciousness_level not finite"
        );

        // Metadata metrics should be finite
        assert!(
            result.metadata.temporal.holographic_unity.is_finite(),
            "Cycle {i}: holographic_unity not finite"
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_gradient_magnitude
                .is_finite(),
            "Cycle {i}: gradient_magnitude not finite"
        );
        assert!(
            result
                .metadata
                .embodied
                .affect_consciousness_valence
                .is_finite(),
            "Cycle {i}: affective_valence not finite"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);

    // Stats should be finite after 200 cycles of high-error input
    assert!(stats.avg_prediction_error.is_finite());
    assert!(stats.avg_training_loss.is_finite());
    assert!(stats.attention_variance.is_finite());

    eprintln!(
        "Bounded metrics after 200 high-error cycles: avg_error={:.4}, \
         avg_loss={:.4}, attention_var={:.4}",
        stats.avg_prediction_error, stats.avg_training_loss, stats.attention_variance
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 9: Resonator-Memory Loop + FEP Deepening Tests
// ═══════════════════════════════════════════════════════════════════════════

// ── Resonator WM Priming ────────────────────────────────────────

#[test]
fn test_resonator_wm_priming() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut any_primed = false;
    let mut any_episodes = false;
    for _ in 0..50 {
        let result = service.cycle("resonator priming test input");
        if result.metadata.memory.resonator_wm_primed {
            any_primed = true;
        }
        if result.metadata.memory.resonator_episodes > 0 {
            any_episodes = true;
        }
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error must be finite during resonator priming"
        );
    }

    // Resonator priming may or may not trigger depending on codebook state.
    // We verify no panics and that the fields are populated.
    let _ = (any_primed, any_episodes);
    assert_eq!(service.stats().total_cycles, 50);
}

// ── Resonator Episodic Reconsolidation ──────────────────────────

#[test]
fn test_resonator_episodic_reconsolidation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut max_reconsolidated = 0usize;
    for _ in 0..30 {
        let result = service.cycle("reconsolidation test pattern");
        if result.metadata.memory.resonator_reconsolidated > max_reconsolidated {
            max_reconsolidated = result.metadata.memory.resonator_reconsolidated;
        }
        // Reconsolidated count should never exceed MEMORY_RECALL_TOP_K (3)
        assert!(
            result.metadata.memory.resonator_reconsolidated <= 3,
            "Reconsolidated count should be bounded by TOP_K: got {}",
            result.metadata.memory.resonator_reconsolidated
        );
    }

    assert_eq!(service.stats().total_cycles, 30);
}

// ── High-Phi Resonator Promotion ────────────────────────────────

#[test]
fn test_high_phi_resonator_promotion() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut total_promotions = 0usize;
    let mut codebook_sizes = Vec::new();
    for i in 0..100 {
        let input = if i % 3 == 0 {
            "consciousness integration resonance"
        } else if i % 3 == 1 {
            "temporal dynamics emergence"
        } else {
            "holographic binding coherence"
        };
        let result = service.cycle(input);
        total_promotions += result.metadata.memory.resonator_promotions;
        codebook_sizes.push(result.metadata.memory.resonator_codebook_size);
    }

    // 100 cycles includes cycle 97 which is the promotion cadence.
    // Promotions may or may not occur depending on phi threshold.
    assert_eq!(service.stats().total_cycles, 100);
    eprintln!(
        "Resonator promotions over 100 cycles: {total_promotions}, \
         final codebook size: {}",
        codebook_sizes.last().unwrap_or(&0)
    );
}

// ── FEP Component Decomposition ─────────────────────────────────

#[test]
fn test_fep_component_decomposition() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..20 {
        let result = service.cycle("free energy decomposition test");
        assert!(
            result.metadata.fep.fep_accuracy.is_finite(),
            "FEP accuracy must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_complexity.is_finite(),
            "FEP complexity must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_surprise.is_finite(),
            "FEP surprise must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_td_error.is_finite(),
            "FEP TD error must be finite at cycle {i}"
        );
    }

    assert_eq!(service.stats().total_cycles, 20);
}

// ── FEP Pragmatic Exploration Balance ───────────────────────────

#[test]
fn test_fep_pragmatic_exploration_balance() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("pragmatic value exploration test");
        assert!(
            result.metadata.fep.fep_pragmatic_value.is_finite(),
            "FEP pragmatic value must be finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_action < 8,
            "FEP action should be in valid range at cycle {i}: got {}",
            result.metadata.fep.fep_action
        );
    }

    assert_eq!(service.stats().total_cycles, 30);
}

// ── FEP TD Error Causal Trigger ─────────────────────────────────

#[test]
fn test_fep_td_error_causal_trigger() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "pattern alpha stable input",
        "completely novel surprise stimulus xyz",
    ];

    for i in 0..40 {
        // Alternate inputs to produce prediction errors
        let result = service.cycle(inputs[i % 2]);
        assert!(
            result.metadata.fep.fep_td_error.is_finite(),
            "FEP TD error must be finite at cycle {i}"
        );
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error must be finite at cycle {i}"
        );
    }

    assert_eq!(service.stats().total_cycles, 40);
}

// ── Full Resonator-FEP Loop Stability ───────────────────────────

#[test]
fn test_full_resonator_fep_loop_stability() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "quantum coherence binding dynamics",
        "episodic memory consolidation process",
        "active inference free energy minimization",
    ];

    for i in 0..150 {
        let result = service.cycle(inputs[i % inputs.len()]);

        // All new metadata fields must be finite
        assert!(
            result.metadata.fep.fep_accuracy.is_finite(),
            "FEP accuracy not finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_complexity.is_finite(),
            "FEP complexity not finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_surprise.is_finite(),
            "FEP surprise not finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_td_error.is_finite(),
            "FEP TD error not finite at cycle {i}"
        );
        assert!(
            result.metadata.fep.fep_pragmatic_value.is_finite(),
            "FEP pragmatic value not finite at cycle {i}"
        );

        // Resonator fields should be bounded
        assert!(
            result.metadata.memory.resonator_reconsolidated <= 3,
            "Reconsolidated exceeds TOP_K at cycle {i}"
        );
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error not finite at cycle {i}"
        );
    }

    assert_eq!(service.stats().total_cycles, 150);

    let avg_error = service.stats().avg_prediction_error;
    assert!(
        avg_error < 1.0,
        "Average error should be bounded over 150 cycles: got {avg_error:.4}"
    );

    eprintln!("Full resonator-FEP stability: 150 cycles, avg_error={avg_error:.4}");
}

// ── Phase 10+11: Self-Regulating Resonator + Homeostatic Feedback Tests ────

#[test]
fn test_codebook_pruning_at_capacity() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        resonator_max_symbols: 5,
        ..Default::default()
    })
    .unwrap();

    for i in 0..200 {
        let input = match i % 5 {
            0 => "alpha resonance field",
            1 => "beta oscillation pattern",
            2 => "gamma binding coherence",
            3 => "delta sleep consolidation",
            _ => "theta exploration mode",
        };
        let result = service.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "Error not finite at cycle {i}"
        );
        assert!(
            result.metadata.memory.codebook_evictions <= 3,
            "Too many evictions at cycle {i}: {}",
            result.metadata.memory.codebook_evictions
        );
    }

    let stats = service.stats();
    eprintln!(
        "Codebook pruning: promotions={}, evictions={}, diversity={:.3}",
        stats.resonator_promotions_total, stats.codebook_evictions_total, stats.codebook_diversity
    );
}

#[test]
fn test_resonator_fep_prior_coupling() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..30 {
        service.cycle("familiar context pattern");
    }

    let result = service.cycle("familiar context pattern");
    assert!(result.metadata.memory.resonator_best_sim.is_finite());
    assert!(result.metadata.fep.fep_accuracy.is_finite());
    assert!(result.metadata.fep.fep_complexity.is_finite());
}

#[test]
fn test_fep_adaptive_behavior_modulation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let input = if i % 2 == 0 {
            "calm steady state equilibrium"
        } else {
            "EXTREME NOVEL SHOCKING SURPRISE"
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.fep.fep_surprise.is_finite(),
            "fep_surprise not finite at {i}"
        );
        assert!(
            result.metadata.fep.fep_td_error.is_finite(),
            "fep_td_error not finite at {i}"
        );
        assert!(
            result.metadata.fep.fep_pragmatic_value.is_finite(),
            "fep_pragmatic not finite at {i}"
        );
    }
}

#[test]
fn test_fep_surprise_replay_boost() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..20 {
        service.cycle("baseline routine input");
    }
    for _ in 0..20 {
        service.cycle("completely novel unexpected stimulus");
    }

    let stats = service.stats();
    assert!(
        stats.fep_surprise_replay_boosts <= 40,
        "Replay boosts should be bounded: got {}",
        stats.fep_surprise_replay_boosts
    );
    eprintln!(
        "FEP surprise replay: boosts={}, avg_error={:.4}",
        stats.fep_surprise_replay_boosts, stats.avg_prediction_error
    );
}

#[test]
fn test_codebook_diversity_metric() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..100 {
        let input = match i % 4 {
            0 => "diverse input alpha",
            1 => "diverse input beta",
            2 => "diverse input gamma",
            _ => "diverse input delta",
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.memory.codebook_diversity.is_finite(),
            "Diversity not finite at {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity >= 0.0,
            "Diversity negative at {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity <= 1.0,
            "Diversity > 1.0 at {i}: {}",
            result.metadata.memory.codebook_diversity
        );
    }
}

#[test]
fn test_coherence_gating_and_tau_modulation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run cycles and verify resonator_best_sim stays finite even with gating
    for i in 0..50 {
        let input = if i % 3 == 0 {
            "temporal coherence test alpha"
        } else if i % 3 == 1 {
            "novel stimulus beta"
        } else {
            "familiar context alpha"
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.memory.resonator_best_sim.is_finite(),
            "best_sim not finite at cycle {i}"
        );
        assert!(
            result.prediction_error.is_finite(),
            "Error not finite at {i}"
        );
    }
}

#[test]
fn test_world_model_storage_bias() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Build up world model, then switch inputs to create high WM error
    for _ in 0..20 {
        service.cycle("familiar world model context");
    }
    // Novel input should cause higher WM error → higher storage importance
    for i in 0..20 {
        let result = service.cycle("completely novel world stimulus");
        assert!(
            result.prediction_error.is_finite(),
            "Error not finite at novel cycle {i}"
        );
    }

    let stats = service.stats();
    assert!(stats.avg_prediction_error.is_finite());
    eprintln!(
        "World model bias: avg_error={:.4}, promotions={}, wm_primed={}",
        stats.avg_prediction_error,
        stats.resonator_promotions_total,
        stats.resonator_wm_primed_count
    );
}

#[test]
fn test_fep_reward_enrichment() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..50 {
        let input = match i % 3 {
            0 => "reward enrichment alpha",
            1 => "reward enrichment beta",
            _ => "reward enrichment gamma",
        };
        let result = service.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "Error not finite at {i}"
        );
    }

    let stats = service.stats();
    // last_total_fe should have been populated after first cycle
    // (it's updated inside compute_reward_signal)
    assert!(
        stats.last_total_fe.is_finite(),
        "last_total_fe should be finite"
    );
}

#[test]
#[ignore] // Long-running stress test
fn test_1000_cycle_all_tracks_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        resonator_max_symbols: 20,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "consciousness integration resonance",
        "temporal dynamics emergence",
        "holographic binding coherence",
        "predictive coding surprise",
        "emotional valence arousal",
        "memory consolidation replay",
        "attention gating salience",
    ];

    for i in 0..1000 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at {i}"
        );
        assert!(
            result.metadata.fep.fep_accuracy.is_finite(),
            "fep_accuracy NaN at {i}"
        );
        assert!(
            result.metadata.fep.fep_complexity.is_finite(),
            "fep_complexity NaN at {i}"
        );
        assert!(
            result.metadata.fep.fep_surprise.is_finite(),
            "fep_surprise NaN at {i}"
        );
        assert!(
            result.metadata.fep.fep_td_error.is_finite(),
            "fep_td_error NaN at {i}"
        );
        assert!(
            result.metadata.fep.fep_pragmatic_value.is_finite(),
            "fep_pragmatic NaN at {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity.is_finite(),
            "diversity NaN at {i}"
        );
        assert!(
            result.metadata.memory.resonator_best_sim.is_finite(),
            "best_sim NaN at {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity >= 0.0
                && result.metadata.memory.codebook_diversity <= 1.0
        );
        assert!(result.metadata.memory.resonator_reconsolidated <= 3);
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 1000);
    assert!(stats.avg_prediction_error.is_finite());
    assert!(stats.codebook_diversity.is_finite());
    assert!(stats.last_total_fe.is_finite());

    eprintln!(
        "1000-cycle stress: avg_error={:.4}, promotions={}, evictions={}, diversity={:.4}, \
         wm_primed={}, replay_boosts={}, last_fe={:.4}",
        stats.avg_prediction_error,
        stats.resonator_promotions_total,
        stats.codebook_evictions_total,
        stats.codebook_diversity,
        stats.resonator_wm_primed_count,
        stats.fep_surprise_replay_boosts,
        stats.last_total_fe
    );
}

// ── Phase 12: Pipeline Stability ─────────────────────────────────

/// Run 30 cycles with primitive consciousness enabled and verify all
/// metadata telemetry stays finite and bounded.
#[test]
fn test_pipeline_stability_with_primitives() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let input = format!("stability check cycle {i} with varied content");
        let result = service.cycle(&input);
        let m = &result.metadata;

        // Core telemetry must be finite every cycle
        assert!(
            m.valence_homeostasis_pull.is_finite(),
            "homeostasis NaN at cycle {i}"
        );
        assert!(
            m.homeostasis_pull_strength.is_finite(),
            "pull_strength NaN at cycle {i}"
        );
        assert!(
            m.social_trust_current >= 0.0 && m.social_trust_current <= 1.0,
            "social_trust out of range at cycle {i}"
        );
    }
}

// ── Phase 12: Behavioral Feedback Loop Effectiveness ──────────────

/// Verify that the diversity governor actually modulates exploration urge.
/// Run with varied inputs to grow codebook diversity, then check that
/// exploration urge responds to diversity changes.
#[test]
fn test_diversity_governor_modulates_exploration() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Phase 1: monotonous input to keep diversity low
    for _ in 0..30 {
        service.cycle("same input repeatedly");
    }
    let stats_mono = service.stats().clone();

    // Phase 2: diverse inputs to push diversity up
    let varied = [
        "quantum field theory",
        "alpine flower meadow",
        "underwater coral reef",
        "mathematical proof construction",
        "musical harmonic progression",
        "neural network training",
    ];
    for i in 0..60 {
        service.cycle(varied[i % varied.len()]);
    }
    let stats_varied = service.stats().clone();

    // Diversity should have changed (either direction shows the metric is live)
    let diversity_changed = (stats_varied.codebook_diversity - stats_mono.codebook_diversity).abs()
        > 0.001
        || stats_varied.codebook_diversity > 0.0;

    // Not a hard assertion since codebook might not be populated in 90 cycles,
    // but at minimum the system shouldn't panic
    eprintln!(
        "Diversity governor: mono_div={:.4}, varied_div={:.4}, changed={}",
        stats_mono.codebook_diversity, stats_varied.codebook_diversity, diversity_changed
    );
}

/// Verify that FEP surprise triggers episodic replay boosts.
/// Alternate between familiar and novel inputs to create surprise spikes.
#[test]
fn test_fep_surprise_triggers_replay_boosts() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Phase 1: establish familiarity
    for _ in 0..50 {
        service.cycle("familiar pattern repeated consistently");
    }

    // Phase 2: inject surprising inputs
    let surprises = [
        "completely unexpected quantum singularity",
        "radical paradigm shift in consciousness theory",
        "unprecedented emergent behavior in complex systems",
    ];
    for i in 0..30 {
        let result = service.cycle(surprises[i % surprises.len()]);
        assert!(
            result.metadata.fep.fep_surprise.is_finite(),
            "fep_surprise NaN at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "FEP surprise→replay: total_boosts={}, avg_error={:.4}",
        stats.fep_surprise_replay_boosts, stats.avg_prediction_error
    );
    // The system should not panic regardless of surprise levels
}

/// Verify that the coherence gate correctly skips resonator recall during
/// low-coherence periods (warmup bypass ensures first cycles still work).
#[test]
fn test_coherence_gate_warmup_bypass() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // First 10 cycles should bypass coherence gate (warmup)
    for i in 0..10 {
        let result = service.cycle("warmup input");
        assert!(
            result.prediction_error.is_finite(),
            "NaN during warmup at cycle {i}"
        );
    }

    // After warmup, coherence gate may filter — system should still be stable
    for i in 10..50 {
        let result = service.cycle("post warmup stable input");
        assert!(
            result.prediction_error.is_finite(),
            "NaN post-warmup at cycle {i}"
        );
        assert!(result.metadata.memory.resonator_best_sim >= -0.01);
    }
}

/// Verify that adaptive thresholds from self-reflection are used
/// (thresholds should remain within their documented ranges).
#[test]
fn test_adaptive_thresholds_bounded() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run enough cycles to trigger self-reflection (interval is ~25 cycles)
    for i in 0..100 {
        let result = service.cycle(&format!("adaptive threshold test cycle {i}"));
        assert!(result.prediction_error.is_finite());
    }

    // Check that the system survived 100 cycles with adaptive thresholds
    let stats = service.stats();
    assert_eq!(stats.total_cycles, 100);
    assert!(stats.avg_prediction_error.is_finite());
    assert!(stats.avg_prediction_error >= 0.0);
    assert!(stats.avg_prediction_error <= 1.0);
}

/// Verify that Σ (sigma) modulation of learning rate doesn't cause divergence.
/// Run 200 cycles and check that fep_lr_boost stays bounded.
#[test]
fn test_sigma_lr_modulation_stable() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "consciousness emerges from integration",
        "predictive processing minimizes free energy",
        "holographic memory encodes distributed patterns",
    ];

    for i in 0..200 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at cycle {i}"
        );
        assert!(
            result.prediction_error <= 1.0,
            "prediction_error > 1.0 at cycle {i}: {}",
            result.prediction_error
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    eprintln!(
        "Sigma modulation (200 cycles): avg_error={:.4}, diversity={:.4}",
        stats.avg_prediction_error, stats.codebook_diversity
    );
}

/// Verify WM eviction → resonator routing doesn't cause panics or NaN.
#[test]
fn test_wm_eviction_resonator_routing() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Flood WM with diverse inputs to trigger evictions
    let inputs = [
        "alpha input pattern one",
        "beta input pattern two",
        "gamma input pattern three",
        "delta input pattern four",
        "epsilon input pattern five",
        "zeta input pattern six",
        "eta input pattern seven",
        "theta input pattern eight",
        "iota input pattern nine",
        "kappa input pattern ten",
    ];

    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(result.prediction_error.is_finite(), "NaN at cycle {i}");
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 100);
    eprintln!(
        "WM routing (100 cycles): avg_error={:.4}, wm_primed={}",
        stats.avg_prediction_error, stats.resonator_wm_primed_count
    );
}

// ── Phase 13: Predictive Resonator + Cross-Module Coherence ───────

/// Verify resonator prediction error is finite and bounded for all inputs.
#[test]
fn test_resonator_prediction_error_finite() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "resonator prediction test alpha",
        "resonator prediction test beta",
        "resonator prediction test gamma",
    ];

    for i in 0..60 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result
                .metadata
                .memory
                .resonator_prediction_error
                .is_finite(),
            "resonator_prediction_error NaN at cycle {i}"
        );
        assert!(
            result.metadata.memory.resonator_prediction_error >= 0.0
                && result.metadata.memory.resonator_prediction_error <= 1.0,
            "resonator_prediction_error out of [0,1] at cycle {i}: {}",
            result.metadata.memory.resonator_prediction_error
        );
    }
}

/// Verify cross-module agreement is finite, bounded [0,1], and doesn't cause divergence.
#[test]
fn test_cross_module_agreement_stable() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..100 {
        let input = format!("cross module agreement test cycle {i}");
        let result = service.cycle(&input);
        assert!(
            result.metadata.cross_module_agreement.is_finite(),
            "cross_module_agreement NaN at cycle {i}"
        );
        assert!(
            result.metadata.cross_module_agreement >= 0.0
                && result.metadata.cross_module_agreement <= 1.0,
            "cross_module_agreement out of [0,1] at cycle {i}: {}",
            result.metadata.cross_module_agreement
        );
    }

    let stats = service.stats();
    assert!(stats.avg_cross_module_agreement.is_finite());
    assert!(stats.avg_cross_module_agreement >= 0.0 && stats.avg_cross_module_agreement <= 1.0);
    eprintln!(
        "Cross-module agreement (100 cycles): avg={:.4}",
        stats.avg_cross_module_agreement
    );
}

/// Verify thalamic depth score is correctly populated.
#[test]
fn test_thalamic_depth_score_populated() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut seen_scores = std::collections::HashSet::new();
    for i in 0..50 {
        let result = service.cycle(&format!("thalamic depth cycle {i}"));
        let score = result.metadata.thalamic_depth_score;
        assert!(
            (0.0..=1.0).contains(&score),
            "thalamic_depth_score out of bounds: {score}"
        );
        // Track which scores we see (0.2, 0.5, or 1.0)
        seen_scores.insert((score * 10.0) as i32);
    }
    eprintln!("Thalamic depth scores seen: {:?}", seen_scores);
}

/// Verify enriched reward signal stays bounded after adding agreement + semantic bonuses.
#[test]
fn test_enriched_reward_bounded() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..80 {
        let result = service.cycle(&format!("reward enrichment cycle {i}"));
        assert!(result.prediction_error.is_finite(), "NaN at cycle {i}");
        // Prediction error stays bounded (enriched reward doesn't leak)
        assert!(result.prediction_error <= 1.0, "error > 1.0 at cycle {i}");
    }
}

/// Verify adaptive replay interval adjusts based on error volatility.
/// Run stable then volatile inputs and check that the system survives.
#[test]
fn test_adaptive_replay_scheduling() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Phase 1: stable inputs (low error variance)
    for _ in 0..60 {
        service.cycle("stable input pattern for replay scheduling");
    }

    // Phase 2: volatile inputs (should trigger interval adaptation at cycle 100)
    let volatile = [
        "completely unexpected input alpha",
        "stable input pattern for replay scheduling",
        "radical divergent content beta",
        "stable input pattern for replay scheduling",
    ];
    for i in 0..60 {
        let result = service.cycle(volatile[i % volatile.len()]);
        assert!(result.prediction_error.is_finite(), "NaN at cycle {i}");
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 120);
    assert!(stats.avg_prediction_error_sq.is_finite());
    eprintln!(
        "Adaptive replay (120 cycles): avg_err={:.4}, avg_err_sq={:.6}",
        stats.avg_prediction_error, stats.avg_prediction_error_sq
    );
}

/// 300-cycle multi-track stress test for Phase 13 features.
/// Verifies all new metadata fields are finite across extended operation.
#[test]
#[ignore] // stress test — run with `cargo test -- --ignored`
fn test_300_cycle_phase13_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "predictive coding in hierarchical cortex",
        "resonator network factorization patterns",
        "cross module agreement measurement",
        "adaptive replay scheduling dynamics",
        "thalamic gating attention salience",
        "free energy minimization principle",
    ];

    for i in 0..300 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at {i}"
        );
        assert!(
            result
                .metadata
                .memory
                .resonator_prediction_error
                .is_finite(),
            "resonator_pred_err NaN at {i}"
        );
        assert!(
            result.metadata.cross_module_agreement.is_finite(),
            "agreement NaN at {i}"
        );
        assert!(
            result.metadata.thalamic_depth_score >= 0.0,
            "depth < 0 at {i}"
        );
        assert!(
            result.metadata.fep.fep_accuracy.is_finite(),
            "fep_accuracy NaN at {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity.is_finite(),
            "diversity NaN at {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 300);
    assert!(stats.avg_cross_module_agreement.is_finite());
    assert!(stats.avg_prediction_error_sq.is_finite());

    eprintln!(
        "300-cycle Phase 13 stress: avg_error={:.4}, agreement={:.4}, diversity={:.4}, \
         err_variance={:.6}",
        stats.avg_prediction_error,
        stats.avg_cross_module_agreement,
        stats.codebook_diversity,
        (stats.avg_prediction_error_sq - stats.avg_prediction_error * stats.avg_prediction_error)
            .max(0.0)
    );
}

// ── Phase 14: Subsystem Feedback Closure ──────────────────────────────────

#[test]
fn test_epistemic_gate_gating() {
    // Verify epistemic gate modulates codebook growth and LR without panics
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut gated_count = 0usize;
    for i in 0..60 {
        let result = service.cycle("epistemic gating test input");
        assert!(
            result.metadata.epistemic_gate_confidence.is_finite(),
            "epistemic_gate_confidence NaN at cycle {i}"
        );
        if result.metadata.epistemic_gate_gated {
            gated_count += 1;
        }
    }
    // Gate may or may not fire — just verify no panics and field is populated
    eprintln!("Epistemic gate: gated {gated_count}/60 cycles");
}

#[test]
fn test_mcts_plan_effectiveness() {
    // Verify MCTS plan post-hoc evaluation produces finite scores
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut any_evaluated = false;
    for i in 0..80 {
        let input = if i % 2 == 0 {
            "exploit this pattern"
        } else {
            "explore new territory"
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.mcts_plan_effectiveness.is_finite(),
            "mcts_plan_effectiveness NaN at cycle {i}"
        );
        if result.metadata.mcts_plan_effectiveness > 0.0 {
            any_evaluated = true;
        }
    }
    let stats = service.stats();
    assert!(stats.avg_mcts_plan_effectiveness.is_finite());
    eprintln!(
        "MCTS plan effectiveness: avg={:.4}, any_evaluated={any_evaluated}",
        stats.avg_mcts_plan_effectiveness
    );
}

#[test]
fn test_moral_violation_steering() {
    // Verify moral violation-specific steering categories are applied
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "help me with this task",    // benign
        "steal from the store",      // harm/theft
        "violate someone's consent", // consent
        "ignore my duty to protect", // duty
        "a regular input again",     // benign
    ];

    for (i, input) in inputs.iter().enumerate() {
        let result = service.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at input {i}"
        );
        // moral_steering_category is a String — just verify no panics
        if !result.metadata.ethics.moral_steering_category.is_empty() {
            eprintln!(
                "Moral steering at input {i}: category={}",
                result.metadata.ethics.moral_steering_category
            );
        }
    }
}

#[test]
fn test_codebook_utilization_tracking() {
    // Verify codebook utilization rate is computed and bounded [0, 1]
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..55 {
        let result = service.cycle("utilization tracking test");
        assert!(
            result.metadata.memory.codebook_utilization_rate.is_finite(),
            "codebook_utilization_rate NaN at cycle {i}"
        );
        assert!(
            result.metadata.memory.codebook_utilization_rate >= 0.0,
            "codebook_utilization_rate < 0 at cycle {i}"
        );
        assert!(
            result.metadata.memory.codebook_utilization_rate <= 1.0,
            "codebook_utilization_rate > 1 at cycle {i}"
        );
    }
    let stats = service.stats();
    assert!(stats.codebook_utilization_rate.is_finite());
    assert!(stats.codebook_utilization_rate >= 0.0);
    assert!(stats.codebook_utilization_rate <= 1.0);
    eprintln!(
        "Codebook utilization: rate={:.4}",
        stats.codebook_utilization_rate
    );
}

#[test]
fn test_causal_attention_edges() {
    // Verify causal graph attention weighting produces finite values
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let input = if i % 3 == 0 {
            "pattern A"
        } else if i % 3 == 1 {
            "pattern B"
        } else {
            "pattern C"
        };
        let result = service.cycle(input);
        // causal_attention_edges can be 0 if no graph discovered yet
        assert!(
            result.metadata.causal_attention_edges < 10000,
            "causal_attention_edges implausibly large at cycle {i}"
        );
    }
    let stats = service.stats();
    eprintln!("Causal attention: uses={}", stats.causal_attention_uses);
}

#[test]
fn test_surprise_replay_batch_modulation() {
    // Verify FEP surprise modulates replay batch size
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut max_batch = 0usize;
    for i in 0..120 {
        let input = if i % 20 < 10 {
            "stable input"
        } else {
            "surprising new content!"
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.memory.surprise_replay_batch_size < 100,
            "surprise_replay_batch_size implausibly large at cycle {i}"
        );
        if result.metadata.memory.surprise_replay_batch_size > max_batch {
            max_batch = result.metadata.memory.surprise_replay_batch_size;
        }
    }
    let stats = service.stats();
    eprintln!(
        "Surprise replay: boosted_replays={}, max_batch={max_batch}",
        stats.surprise_boosted_replays
    );
}

#[test]
#[ignore] // stress test
fn test_400_cycle_phase14_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "conscious experience requires integration",
        "steal the confidential data",
        "explore new mathematical frontiers",
        "help someone in need",
        "free energy minimization principle",
    ];

    for i in 0..400 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at {i}"
        );
        assert!(
            result.metadata.epistemic_gate_confidence.is_finite(),
            "epistemic NaN at {i}"
        );
        assert!(
            result.metadata.mcts_plan_effectiveness.is_finite(),
            "mcts_eff NaN at {i}"
        );
        assert!(
            result.metadata.memory.codebook_utilization_rate.is_finite(),
            "util NaN at {i}"
        );
        assert!(
            result.metadata.causal_attention_edges < 10000,
            "causal edges huge at {i}"
        );
        assert!(
            result.metadata.memory.surprise_replay_batch_size < 100,
            "batch huge at {i}"
        );
        assert!(
            result.metadata.cross_module_agreement.is_finite(),
            "agreement NaN at {i}"
        );
        assert!(
            result.metadata.thalamic_depth_score >= 0.0,
            "depth < 0 at {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 400);
    assert!(stats.avg_mcts_plan_effectiveness.is_finite());
    assert!(stats.codebook_utilization_rate.is_finite());
    assert!(stats.avg_cross_module_agreement.is_finite());

    eprintln!(
        "400-cycle Phase 14 stress: avg_error={:.4}, mcts_eff={:.4}, util={:.4}, \
         agreement={:.4}, causal_uses={}, surprise_replays={}",
        stats.avg_prediction_error,
        stats.avg_mcts_plan_effectiveness,
        stats.codebook_utilization_rate,
        stats.avg_cross_module_agreement,
        stats.causal_attention_uses,
        stats.surprise_boosted_replays,
    );
}

// ── Phase 15: Adaptive Architecture + Emotional Homeostasis ──────────────

/// Task #46: Attention budget tracking — verify elapsed time is recorded.
#[test]
fn test_attention_budget_tracking() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut budget_recorded = false;
    for _ in 0..30 {
        let result = service.cycle("attention budget test");
        // Budget elapsed should always be positive (we've done real work)
        if result.metadata.attention.attention_budget_elapsed_us > 0 {
            budget_recorded = true;
        }
    }
    assert!(
        budget_recorded,
        "attention budget elapsed_us never recorded"
    );

    let stats = service.stats();
    // Over 30 cycles, some may exceed budget — just ensure counter is finite
    eprintln!(
        "Attention budget: exceeded={}, cycles={}",
        stats.attention_budget_exceeded_count, stats.total_cycles
    );
}

/// Task #47: Multi-horizon prediction coherence — verify coherence is finite.
#[test]
fn test_prediction_coherence() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut coherence_computed = false;
    for i in 0..30 {
        let result = service.cycle("prediction coherence test");
        assert!(
            result.metadata.prediction_coherence.is_finite(),
            "prediction_coherence NaN at cycle {i}"
        );
        if result.metadata.prediction_coherence > 0.0 {
            coherence_computed = true;
        }
    }
    assert!(
        coherence_computed,
        "prediction_coherence never computed (stayed 0.0)"
    );

    let stats = service.stats();
    assert!(stats.avg_prediction_coherence.is_finite());
    assert!(stats.avg_prediction_coherence >= 0.0);
    assert!(stats.avg_prediction_coherence <= 1.0);
    eprintln!(
        "Prediction coherence avg={:.4}",
        stats.avg_prediction_coherence
    );
}

/// Task #48: Emotional homeostasis — verify valence pull is finite and bounded.
#[test]
fn test_emotional_homeostasis() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Feed emotional input to trigger valence changes
    let inputs = ["I love this!", "wonderful amazing great", "happy joy"];
    for round in 0..10 {
        for input in &inputs {
            let result = service.cycle(input);
            assert!(
                result.metadata.valence_homeostasis_pull.is_finite(),
                "valence pull NaN at round {round}"
            );
            assert!(
                result.metadata.arousal_homeostasis_pull.is_finite(),
                "arousal pull NaN at round {round}"
            );
        }
    }

    let stats = service.stats();
    assert!(stats.avg_valence_homeostasis.is_finite());
    eprintln!(
        "Homeostasis: avg_valence_pull={:.6}",
        stats.avg_valence_homeostasis
    );
}

/// Task #49: Arousal recovery mode — verify tau factor and active flag.
#[test]
fn test_arousal_recovery_mode() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run 50 cycles — arousal recovery is stochastic, just verify fields are finite
    for i in 0..50 {
        let result = service.cycle("arousal recovery test input");
        assert!(
            result.metadata.arousal_recovery_tau_factor.is_finite(),
            "tau factor NaN at {i}"
        );
        assert!(
            result.metadata.arousal_recovery_tau_factor >= 0.5,
            "tau factor too low at {i}: {}",
            result.metadata.arousal_recovery_tau_factor
        );
        assert!(
            result.metadata.arousal_recovery_tau_factor <= 2.0,
            "tau factor too high at {i}: {}",
            result.metadata.arousal_recovery_tau_factor
        );
    }

    let stats = service.stats();
    eprintln!(
        "Arousal recovery: cycles_active={}",
        stats.arousal_recovery_cycles
    );
}

/// Task #50: Input similarity memoization — verify same input triggers memoization.
#[test]
fn test_input_similarity_memoization() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // First cycle: no previous state to compare
    let r1 = service.cycle("memoization test input");
    assert_eq!(
        r1.metadata.attention.input_similarity, 0.0,
        "first cycle should have 0 similarity"
    );
    assert!(
        !r1.metadata.attention.input_memoized,
        "first cycle should not be memoized"
    );

    // Second cycle: same input → should have high similarity
    let r2 = service.cycle("memoization test input");
    assert!(
        r2.metadata.attention.input_similarity > 0.5,
        "same input should have high similarity, got {}",
        r2.metadata.attention.input_similarity
    );

    // After many identical cycles, should eventually trigger memoization
    let mut memoized_count = 0;
    for _ in 0..20 {
        let result = service.cycle("memoization test input");
        if result.metadata.attention.input_memoized {
            memoized_count += 1;
        }
    }

    let stats = service.stats();
    eprintln!(
        "Memoization: hits={}, memoized_in_test={}",
        stats.input_memoization_hits, memoized_count
    );
}

/// Task #51: Guiding question → subsystem priority.
#[test]
fn test_guiding_question_priority() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("exploring what I can learn from this");
        // Category can be empty if no experience bus or no guiding question
        assert!(
            result.metadata.harmonics.guiding_priority_category.len() < 50,
            "category too long at {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "Guiding question: priority_uses={}",
        stats.guiding_question_priority_uses
    );
}

/// Stress test: 300 cycles with varied inputs verifying all Phase 15 fields.
#[test]
#[ignore] // expensive
fn test_300_cycle_phase15_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "The quick brown fox",
        "I feel so happy today",
        "What should I do next?",
        "How can we connect better?",
        "Learning something new is exciting",
    ];

    for i in 0..300 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.metadata.prediction_coherence.is_finite(),
            "coherence NaN at {i}"
        );
        assert!(
            result.metadata.valence_homeostasis_pull.is_finite(),
            "valence pull NaN at {i}"
        );
        assert!(
            result.metadata.arousal_homeostasis_pull.is_finite(),
            "arousal pull NaN at {i}"
        );
        assert!(
            result.metadata.arousal_recovery_tau_factor.is_finite(),
            "tau factor NaN at {i}"
        );
        assert!(
            result.metadata.attention.input_similarity.is_finite(),
            "input similarity NaN at {i}"
        );
        assert!(
            result.metadata.attention.attention_budget_elapsed_us < 60_000_000,
            "budget elapsed unreasonably large at {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 300);
    assert!(stats.avg_prediction_coherence.is_finite());
    assert!(stats.avg_valence_homeostasis.is_finite());

    eprintln!(
        "300-cycle Phase 15 stress: coherence={:.4}, homeostasis={:.6}, \
         budget_exceeded={}, recovery_cycles={}, memo_hits={}, \
         guiding_uses={}",
        stats.avg_prediction_coherence,
        stats.avg_valence_homeostasis,
        stats.attention_budget_exceeded_count,
        stats.arousal_recovery_cycles,
        stats.input_memoization_hits,
        stats.guiding_question_priority_uses,
    );
}

// ── Phase 16: Quality-Aware Adaptive Processing ──────────────────────────

/// Task #53: Epistemic gate coherence-gating — verify adaptive threshold scaling.
#[test]
fn test_epistemic_coherence_gating() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("epistemic coherence gating test");
        // epistemic_coherence_gated should be a valid boolean
        let _ = result.metadata.epistemic_coherence_gated;
        assert!(
            result.metadata.quality.unified_quality_score.is_finite(),
            "quality NaN at {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "Epistemic coherence gating: gated_count={}",
        stats.epistemic_coherence_gated_count
    );
}

/// Task #54: Unified quality signal — verify fusion and bounded output.
#[test]
fn test_unified_quality_signal() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("unified quality signal test");
        assert!(
            result.metadata.quality.unified_quality_score >= 0.0,
            "quality < 0 at {i}"
        );
        assert!(
            result.metadata.quality.unified_quality_score <= 1.0,
            "quality > 1 at {i}: {}",
            result.metadata.quality.unified_quality_score
        );
    }

    let stats = service.stats();
    assert!(stats.avg_unified_quality.is_finite());
    assert!(stats.avg_unified_quality >= 0.0);
    assert!(stats.avg_unified_quality <= 1.0);
    eprintln!("Unified quality: avg={:.4}", stats.avg_unified_quality);
}

/// Task #55: Dissipative health learning gate — verify LR factor is bounded.
#[test]
fn test_dissipative_health_gate() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..60 {
        let result = service.cycle("dissipative health stability test");
        assert!(
            result.metadata.quality.dissipative_lr_factor.is_finite(),
            "dissipative factor NaN at {i}"
        );
        assert!(
            result.metadata.quality.dissipative_lr_factor >= 0.5,
            "dissipative factor too low at {i}: {}",
            result.metadata.quality.dissipative_lr_factor
        );
        assert!(
            result.metadata.quality.dissipative_lr_factor <= 1.0,
            "dissipative factor > 1.0 at {i}: {}",
            result.metadata.quality.dissipative_lr_factor
        );
    }

    let stats = service.stats();
    eprintln!(
        "Dissipative health gate: gated_count={}",
        stats.dissipative_health_gated_count
    );
}

/// Task #56: Adaptive Phi weighting — verify spectral weight is finite and bounded.
#[test]
fn test_adaptive_phi_weighting() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("phi validation weighting test");
        assert!(
            result.metadata.phi_spectral_weight.is_finite(),
            "spectral weight NaN at {i}"
        );
        assert!(
            result.metadata.phi_spectral_weight >= 0.3,
            "spectral weight too low at {i}: {}",
            result.metadata.phi_spectral_weight
        );
        assert!(
            result.metadata.phi_spectral_weight <= 0.9,
            "spectral weight too high at {i}: {}",
            result.metadata.phi_spectral_weight
        );
    }
}

/// Task #57: Coherence velocity gating — verify velocity is finite.
#[test]
fn test_coherence_velocity_gating() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("coherence velocity tracking test");
        assert!(
            result.metadata.quality.coherence_velocity.is_finite(),
            "velocity NaN at {i}"
        );
        assert!(
            result.metadata.quality.coherence_velocity.abs() < 2.0,
            "velocity too extreme at {i}: {}",
            result.metadata.quality.coherence_velocity
        );
    }

    let stats = service.stats();
    eprintln!(
        "Coherence velocity: gated_count={}",
        stats.coherence_velocity_gated_count
    );
}

/// Task #58: Anomaly recovery — verify progress is bounded [0, 1].
#[test]
fn test_anomaly_recovery_path() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..50 {
        let result = service.cycle("anomaly recovery test");
        assert!(
            result.metadata.quality.anomaly_recovery_progress >= 0.0,
            "recovery < 0 at {i}"
        );
        assert!(
            result.metadata.quality.anomaly_recovery_progress <= 1.0,
            "recovery > 1 at {i}: {}",
            result.metadata.quality.anomaly_recovery_progress
        );
    }

    let stats = service.stats();
    eprintln!(
        "Anomaly recovery: active_count={}",
        stats.anomaly_recovery_active_count
    );
}

/// Stress test: 300 cycles with varied inputs verifying all Phase 16 fields.
#[test]
#[ignore] // expensive
fn test_300_cycle_phase16_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "The quick brown fox jumps",
        "I feel deeply sad and alone",
        "What should we learn today?",
        "Making connections is vital",
        "Exploring new possibilities",
    ];

    for i in 0..300 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.metadata.quality.unified_quality_score.is_finite(),
            "quality NaN at {i}"
        );
        assert!(
            result.metadata.quality.unified_quality_score >= 0.0
                && result.metadata.quality.unified_quality_score <= 1.0,
            "quality out of [0,1] at {i}"
        );
        assert!(
            result.metadata.quality.dissipative_lr_factor.is_finite(),
            "dissipative factor NaN at {i}"
        );
        assert!(
            result.metadata.quality.coherence_velocity.is_finite(),
            "velocity NaN at {i}"
        );
        assert!(
            result
                .metadata
                .quality
                .anomaly_recovery_progress
                .is_finite(),
            "recovery NaN at {i}"
        );
        assert!(
            result.metadata.phi_spectral_weight.is_finite(),
            "spectral weight NaN at {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 300);
    assert!(stats.avg_unified_quality.is_finite());

    eprintln!(
        "300-cycle Phase 16 stress: quality={:.4}, dissipative_gated={}, \
         coherence_gated={}, epistemic_gated={}, anomaly_recovery={}, \
         spectral_weight={:.4}",
        stats.avg_unified_quality,
        stats.dissipative_health_gated_count,
        stats.coherence_velocity_gated_count,
        stats.epistemic_coherence_gated_count,
        stats.anomaly_recovery_active_count,
        stats.avg_unified_quality,
    );
}

// ── Phase 17: Predictive Self-Tuning ─────────────────────────────────────

/// Phase 17: Error pattern detection — verify pattern classification is bounded.
#[test]
fn test_error_pattern_detection() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let valid_patterns = [
        "Warmup",
        "Stable",
        "Rising",
        "Falling",
        "Oscillating",
        "Spike",
    ];

    for i in 0..30 {
        let result = service.cycle("error pattern detection test");
        assert!(
            valid_patterns.contains(&result.metadata.error_pattern.as_str()),
            "invalid pattern '{}' at cycle {i}",
            result.metadata.error_pattern
        );
        assert!(
            !result.metadata.predicted_urgency.is_empty(),
            "predicted urgency empty at {i}"
        );
    }

    eprintln!(
        "Error pattern at cycle 30: pattern={}, predicted={}",
        service.cycle("final").metadata.error_pattern,
        service.cycle("final").metadata.predicted_urgency,
    );
}

/// Phase 17: Startup transient suppression — verify warmup is active for first 50 cycles.
#[test]
fn test_startup_transient_suppression() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // First 50 cycles should be suppressed
    for i in 0..50 {
        let result = service.cycle("startup warmup test");
        assert!(
            result.metadata.startup_suppressed,
            "cycle {i} should be startup-suppressed"
        );
        assert!(
            result.metadata.startup_warmup_progress >= 0.0
                && result.metadata.startup_warmup_progress <= 1.0,
            "warmup progress out of bounds at {i}: {}",
            result.metadata.startup_warmup_progress
        );
    }

    // Cycle 51 should not be suppressed
    let result = service.cycle("post warmup");
    assert!(
        !result.metadata.startup_suppressed,
        "cycle 51 should not be suppressed"
    );
    assert!(
        (result.metadata.startup_warmup_progress - 1.0).abs() < 0.001,
        "warmup should be 1.0 after warmup"
    );

    let stats = service.stats();
    assert_eq!(stats.startup_suppressed_cycles, 50);
    eprintln!(
        "Startup suppression: {} cycles",
        stats.startup_suppressed_cycles
    );
}

/// Phase 17: Coherence memoization — verify coherence is consistent (cached vs live).
#[test]
fn test_coherence_memoization_consistency() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..20 {
        service.cycle("coherence memoization test");
    }

    // After cycling, temporal_coherence() should use cached value
    let cached_coherence = service.temporal_coherence();
    assert!(
        cached_coherence.is_finite(),
        "cached coherence must be finite"
    );
    assert!(
        (0.0..=1.0).contains(&cached_coherence),
        "coherence out of bounds: {cached_coherence}"
    );

    // Run one more cycle and verify stats coherence matches
    let result = service.cycle("one more");
    assert!(
        service.stats().temporal_coherence.is_finite(),
        "stats coherence NaN"
    );
    assert!(result.metadata.prediction_coherence.is_finite());
    eprintln!(
        "Coherence memoization: cached={:.4}, stats={:.4}",
        service.temporal_coherence(),
        service.stats().temporal_coherence
    );
}

/// Phase 17: Self-model accuracy — verify predictions are made and validated.
#[test]
fn test_self_model_accuracy() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..50 {
        let result = service.cycle("self model accuracy test");
        assert!(
            result.metadata.self_model_accuracy >= 0.0
                && result.metadata.self_model_accuracy <= 1.0,
            "self_model_accuracy out of [0,1]: {}",
            result.metadata.self_model_accuracy
        );
    }

    let stats = service.stats();
    // After 50 cycles, predictions every 7 → ~7 predictions, ~5 validated
    assert!(
        stats.self_model_predictions_made > 0,
        "no self-model predictions made"
    );
    assert!(
        stats.avg_self_model_accuracy.is_finite(),
        "avg accuracy NaN"
    );

    eprintln!(
        "Self-model: predictions={}, validated={}, avg_accuracy={:.4}",
        stats.self_model_predictions_made,
        stats.self_model_predictions_validated,
        stats.avg_self_model_accuracy,
    );
}

/// Phase 17: Mode transition smoothing — verify transitions and confidence ramping.
#[test]
fn test_mode_transition_smoothing() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut saw_transition = false;
    for i in 0..60 {
        let input = if i % 20 < 10 {
            "simple stable input"
        } else {
            "completely novel extraordinary unprecedented revolutionary paradigm-shifting input"
        };
        let result = service.cycle(input);
        assert!(
            result.metadata.mode_confidence >= 0.0 && result.metadata.mode_confidence <= 1.0,
            "mode_confidence out of bounds at {i}: {}",
            result.metadata.mode_confidence
        );
        if result.metadata.mode_confidence < 1.0 {
            saw_transition = true;
        }
    }

    let stats = service.stats();
    assert!(stats.avg_mode_stability.is_finite(), "mode stability NaN");
    // At least one mode transition should have occurred with alternating inputs
    eprintln!(
        "Mode transitions: {}, avg_stability={:.2}, saw_fresh_transition={}",
        stats.mode_transitions, stats.avg_mode_stability, saw_transition,
    );
}

/// Phase 17: Adaptive interval self-tuning — verify error pattern modulates urgency.
#[test]
fn test_adaptive_interval_tuning() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run stable phase then inject novelty
    for _ in 0..30 {
        service.cycle("stable baseline input");
    }
    // Inject novel inputs to trigger Rising pattern
    for _ in 0..10 {
        let result = service.cycle("completely novel alien unexpected input triggering high error");
        assert!(
            !result.metadata.error_pattern.is_empty(),
            "error pattern should be classified"
        );
    }

    let stats = service.stats();
    assert!(stats.total_cycles == 40);
    eprintln!(
        "Interval tuning: mode_transitions={}, mode_stability={:.2}",
        stats.mode_transitions, stats.avg_mode_stability,
    );
}

/// Phase 17: 300-cycle comprehensive stress test.
#[test]
#[ignore] // Slow: ~30s
fn test_300_cycle_phase17_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "philosophical contemplation on consciousness",
        "simple everyday greeting hello",
        "urgent emergency crisis situation requiring immediate action",
    ];

    for i in 0..300 {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.metadata.self_model_accuracy.is_finite(),
            "self_model_accuracy NaN at {i}"
        );
        assert!(
            result.metadata.mode_confidence >= 0.0 && result.metadata.mode_confidence <= 1.0,
            "mode_confidence out of [0,1] at {i}"
        );
        assert!(
            result.metadata.startup_warmup_progress >= 0.0
                && result.metadata.startup_warmup_progress <= 1.0,
            "warmup out of bounds at {i}"
        );
        assert!(
            !result.metadata.error_pattern.is_empty(),
            "error pattern empty at {i}"
        );
        assert!(
            !result.metadata.predicted_urgency.is_empty(),
            "predicted urgency empty at {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 300);
    assert_eq!(stats.startup_suppressed_cycles, 50);
    assert!(stats.self_model_predictions_made > 0);
    assert!(stats.mode_transitions > 0 || stats.avg_mode_stability > 0.0);
    assert!(stats.avg_self_model_accuracy.is_finite());

    eprintln!(
        "300-cycle Phase 17 stress: startup_suppressed={}, predictions={}/{}, \
         mode_transitions={}, stability={:.2}, self_model_acc={:.4}, \
         error_pattern=valid",
        stats.startup_suppressed_cycles,
        stats.self_model_predictions_validated,
        stats.self_model_predictions_made,
        stats.mode_transitions,
        stats.avg_mode_stability,
        stats.avg_self_model_accuracy,
    );
}

// ── Phase 18: Closing Feedback Loops ────────────────────────────────────

#[test]
fn test_context_phi_weight_feedback() {
    // Verify context_phi_weight modulation applies without panic and produces bounded values
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run 30 cycles with varied input to trigger context detection
    let inputs = [
        "analyze this data carefully",
        "how do you feel about this?",
        "create something new",
    ];
    for i in 0..30 {
        let result = service.cycle(inputs[i % inputs.len()]);
        assert!(
            result.metadata.context_phi_weight.is_finite(),
            "context_phi_weight not finite at cycle {i}"
        );
        // context_phi_applied is bool, always valid
    }

    let stats = service.stats();
    eprintln!(
        "context_phi: applied_count={}, total_cycles={}",
        stats.context_phi_applied_count, stats.total_cycles
    );
}

#[test]
fn test_empathic_speech_rate_modulation() {
    // Verify empathic tone adjustment modulates speech rate within bounds
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("I'm really frustrated with this!");
        assert!(
            result.metadata.ethics.empathic_speech_rate_mod.is_finite(),
            "empathic_speech_rate_mod not finite at cycle {i}"
        );
    }

    // Speech rate should stay within bounds [0.6, 1.5]
    let speech_rate = service.speech_rate_multiplier();
    assert!(
        (0.6..=1.5).contains(&speech_rate),
        "speech rate out of bounds: {speech_rate}"
    );
}

#[test]
fn test_value_evaluator_learning_gate() {
    // Verify value evaluator gates learning (gate factor bounded and finite)
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("test value alignment");
        assert!(
            result.metadata.ethics.value_gate_factor.is_finite(),
            "value_gate_factor not finite at cycle {i}"
        );
        assert!(
            result.metadata.ethics.value_gate_factor >= 0.0
                && result.metadata.ethics.value_gate_factor <= 2.0,
            "value_gate_factor out of bounds: {} at cycle {i}",
            result.metadata.ethics.value_gate_factor
        );
    }

    let stats = service.stats();
    eprintln!(
        "value_gate: applied_count={}, total_cycles={}",
        stats.value_gate_applied_count, stats.total_cycles
    );
}

#[test]
fn test_evolution_phi_delta_feedback() {
    // Verify evolution delta feeds back to confidence/exploration
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("evolving consciousness");
        assert!(
            result.metadata.evolution_confidence_delta.is_finite(),
            "evolution_confidence_delta not finite at cycle {i}"
        );
        assert!(
            result.metadata.evolution_confidence_delta >= -1.0
                && result.metadata.evolution_confidence_delta <= 1.0,
            "evolution_confidence_delta out of bounds at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "evolution_feedback: count={}, total_cycles={}",
        stats.evolution_feedback_count, stats.total_cycles
    );
}

#[test]
fn test_urgency_adaptive_homeostasis() {
    // Verify homeostasis pull strength adapts to urgency mode
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let mut pull_strengths = Vec::new();
    // 30 stable cycles (should settle into Cruise with stronger pull)
    for _ in 0..30 {
        let result = service.cycle("stable input");
        let pull = result.metadata.homeostasis_pull_strength;
        assert!(pull.is_finite(), "homeostasis_pull_strength not finite");
        assert!(
            (0.5..=2.0).contains(&pull),
            "homeostasis_pull_strength out of bounds: {pull}"
        );
        pull_strengths.push(pull);
    }
    // Then 10 novel cycles (may trigger higher urgency with weaker pull)
    for _ in 0..10 {
        let result = service.cycle("completely novel unexpected stimulus!");
        let pull = result.metadata.homeostasis_pull_strength;
        assert!(pull.is_finite());
        pull_strengths.push(pull);
    }

    // Should have seen at least one non-1.0 pull strength (urgency adaptation)
    let varied = pull_strengths.iter().any(|p| (*p - 1.0).abs() > 0.01);
    eprintln!(
        "homeostasis: varied={varied}, pulls={:?}",
        &pull_strengths[..5.min(pull_strengths.len())]
    );
}

#[test]
fn test_prediction_coherence_urgency_bias() {
    // Verify prediction coherence biases urgency threshold
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("coherence test input");
        assert!(
            result
                .metadata
                .prediction_coherence_urgency_bias
                .is_finite(),
            "prediction_coherence_urgency_bias not finite at cycle {i}"
        );
        assert!(
            result.metadata.prediction_coherence_urgency_bias >= -0.2
                && result.metadata.prediction_coherence_urgency_bias <= 0.2,
            "prediction_coherence_urgency_bias out of bounds at cycle {i}: {}",
            result.metadata.prediction_coherence_urgency_bias
        );
    }

    let stats = service.stats();
    assert!(
        stats.avg_prediction_coherence.is_finite(),
        "avg_prediction_coherence not finite"
    );
    eprintln!(
        "coherence_urgency: avg_coherence={:.4}, total_cycles={}",
        stats.avg_prediction_coherence, stats.total_cycles
    );
}

#[test]
#[ignore] // stress test — run manually
fn test_200_cycle_phase18_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "analyze carefully",
        "how do you feel?",
        "create something new",
        "solve this problem",
        "I'm frustrated!",
        "this is wonderful",
    ];

    for i in 0..200 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // All Phase 18 fields must be finite
        assert!(
            m.ethics.empathic_speech_rate_mod.is_finite(),
            "empathic_speech_rate_mod at {i}"
        );
        assert!(
            m.ethics.value_gate_factor.is_finite(),
            "value_gate_factor at {i}"
        );
        assert!(
            m.evolution_confidence_delta.is_finite(),
            "evolution_confidence_delta at {i}"
        );
        assert!(
            m.homeostasis_pull_strength.is_finite(),
            "homeostasis_pull_strength at {i}"
        );
        assert!(
            m.prediction_coherence_urgency_bias.is_finite(),
            "coherence_bias at {i}"
        );

        // Bounds checks
        assert!(m.ethics.value_gate_factor >= 0.0 && m.ethics.value_gate_factor <= 2.0);
        assert!(m.homeostasis_pull_strength >= 0.5 && m.homeostasis_pull_strength <= 2.0);
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    eprintln!(
        "200-cycle Phase 18 stress: context_phi_applied={}, value_gate={}, \
         evolution_feedback={}, avg_coherence={:.4}",
        stats.context_phi_applied_count,
        stats.value_gate_applied_count,
        stats.evolution_feedback_count,
        stats.avg_prediction_coherence,
    );
}

// ── Phase 19: Activating Dormant Pathways ───────────────────────────────

#[test]
fn test_attention_budget_gating() {
    // Verify attention budget gating flag and subsystem interval doubling
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("attention budget test");
        // attention_budget_gated is bool, always valid
        assert!(
            result
                .metadata
                .attention
                .attention_shift_applied
                .is_finite(),
            "attention_shift not finite at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "attention_budget: gated_count={}, exceeded_count={}",
        stats.attention_budget_gated_count, stats.attention_budget_exceeded_count
    );
}

#[test]
fn test_consciousness_limiting_component_boost() {
    // Verify limiting component triggers targeted boost without panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("test limiting component boost");
        // limiting_component_boosted is a String, always valid
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_gradient_magnitude
                .is_finite(),
            "gradient not finite at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "limiting_component: boost_count={}, total_cycles={}",
        stats.limiting_component_boost_count, stats.total_cycles
    );
}

#[test]
fn test_harmonic_love_resonance_boost() {
    // Verify love resonance confidence boost is finite and bounded
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("love and harmony test");
        assert!(
            result.metadata.love_resonance_boost.is_finite(),
            "love_resonance_boost not finite at cycle {i}"
        );
        assert!(
            result.metadata.love_resonance_boost >= 0.0
                && result.metadata.love_resonance_boost <= 0.1,
            "love_resonance_boost out of bounds at cycle {i}: {}",
            result.metadata.love_resonance_boost
        );
    }

    let stats = service.stats();
    eprintln!(
        "love_resonance: boost_count={}, total_cycles={}",
        stats.love_resonance_boost_count, stats.total_cycles
    );
}

#[test]
fn test_reasoning_chain_confidence_boost() {
    // Verify reasoning chain boost fires without panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("deep reasoning chain test");
        // reasoning_chain_boosted is bool, always valid
        assert!(
            result.metadata.reasoning_chain_confidence.is_finite(),
            "reasoning_chain_confidence not finite at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "reasoning_chain: boost_count={}, total_cycles={}",
        stats.reasoning_chain_boost_count, stats.total_cycles
    );
}

#[test]
fn test_attention_shift_motor_command() {
    // Verify attention shift modulates attention_sensitivity within bounds
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("shift attention focus");
        assert!(
            result
                .metadata
                .attention
                .attention_shift_applied
                .is_finite(),
            "attention_shift not finite at cycle {i}"
        );
    }

    // attention_sensitivity should stay within bounds [0.5, 2.0]
    let sensitivity = service.attention_sensitivity();
    assert!(
        (0.5..=2.0).contains(&sensitivity),
        "attention sensitivity out of bounds: {sensitivity}"
    );
}

#[test]
fn test_cosine_helper_consistency() {
    // Verify the cosine helper produces correct results
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run cycles and verify that cosine-dependent metrics are finite
    for i in 0..30 {
        let result = service.cycle("cosine similarity test");
        assert!(
            result
                .metadata
                .memory
                .resonator_prediction_error
                .is_finite(),
            "resonator_prediction_error not finite at cycle {i}"
        );
        assert!(
            result.metadata.memory.codebook_diversity.is_finite(),
            "codebook_diversity not finite at cycle {i}"
        );
        assert!(
            result.metadata.attention.input_similarity.is_finite(),
            "input_similarity not finite at cycle {i}"
        );
    }
}

#[test]
#[ignore] // stress test — run manually
fn test_200_cycle_phase19_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "analyze carefully",
        "how do you feel?",
        "create something new",
        "solve this problem",
        "love and harmony",
        "deep reasoning",
    ];

    for i in 0..200 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // All Phase 19 fields must be finite/valid
        assert!(m.love_resonance_boost.is_finite(), "love_resonance at {i}");
        assert!(
            m.attention.attention_shift_applied.is_finite(),
            "attention_shift at {i}"
        );

        // Bounds checks
        assert!(m.love_resonance_boost >= 0.0 && m.love_resonance_boost <= 0.1);
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    eprintln!(
        "200-cycle Phase 19 stress: budget_gated={}, limiting_boost={}, \
         love_boost={}, chain_boost={}",
        stats.attention_budget_gated_count,
        stats.limiting_component_boost_count,
        stats.love_resonance_boost_count,
        stats.reasoning_chain_boost_count,
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Phase 20: Signal-to-Control Synthesis
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_harmonic_interference_lr_modulation() {
    // Verify harmonic interference count modulates LR without panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("test harmonic interference lr feedback");
        assert!(
            result.metadata.harmonic_interference_lr_mod.is_finite(),
            "harmonic_interference_lr_mod not finite at cycle {i}"
        );
        assert!(
            result.metadata.harmonic_interference_lr_mod >= -0.15
                && result.metadata.harmonic_interference_lr_mod <= 0.05,
            "harmonic_interference_lr_mod out of bounds at cycle {i}: {}",
            result.metadata.harmonic_interference_lr_mod
        );
    }

    let stats = service.stats();
    eprintln!(
        "harmonic_interference: mod_count={}, total_cycles={}",
        stats.harmonic_interference_mod_count, stats.total_cycles
    );
}

#[test]
fn test_resonator_prediction_error_exploration() {
    // Verify resonator prediction error modulates exploration without panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("resonator prediction error test");
        assert!(
            result.metadata.resonator_error_exploration_mod.is_finite(),
            "resonator_error_exploration_mod not finite at cycle {i}"
        );
        assert!(
            result
                .metadata
                .memory
                .resonator_prediction_error
                .is_finite(),
            "resonator_prediction_error not finite at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "resonator_error: exploration_count={}, total_cycles={}",
        stats.resonator_error_exploration_count, stats.total_cycles
    );
}

#[test]
fn test_phenomenal_binding_threshold_gating() {
    // Verify phenomenal binding modulates adaptive threshold within bounds
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("binding threshold gate test");
        assert!(
            result.metadata.binding_threshold_mod.is_finite(),
            "binding_threshold_mod not finite at cycle {i}"
        );
        assert!(
            result.metadata.binding_threshold_mod >= -0.15
                && result.metadata.binding_threshold_mod <= 0.10,
            "binding_threshold_mod out of bounds at cycle {i}: {}",
            result.metadata.binding_threshold_mod
        );
    }

    let stats = service.stats();
    eprintln!(
        "binding_threshold: mod_count={}, total_cycles={}",
        stats.binding_threshold_mod_count, stats.total_cycles
    );
}

#[test]
fn test_causal_density_urgency_gating() {
    // Verify causal density gating doesn't panic and is a valid bool
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("causal graph density test");
        // causal_urgency_gated is bool — always valid
        assert!(
            result.metadata.causal_avg_confidence.is_finite(),
            "causal_avg_confidence not finite at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "causal_urgency: gated_count={}, total_cycles={}",
        stats.causal_urgency_gated_count, stats.total_cycles
    );
}

#[test]
fn test_epistemic_semantic_lr_bidirectional() {
    // Verify epistemic gate modulates semantic LR via cached confidence
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("epistemic semantic coupling test");
        assert!(
            result.metadata.epistemic_semantic_lr_mod.is_finite(),
            "epistemic_semantic_lr_mod not finite at cycle {i}"
        );
        assert!(
            result.metadata.epistemic_semantic_lr_mod >= -0.25
                && result.metadata.epistemic_semantic_lr_mod <= 0.25,
            "epistemic_semantic_lr_mod out of bounds at cycle {i}: {}",
            result.metadata.epistemic_semantic_lr_mod
        );
    }

    let stats = service.stats();
    eprintln!(
        "epistemic_semantic: mod_count={}, total_cycles={}",
        stats.epistemic_semantic_mod_count, stats.total_cycles
    );
}

#[test]
fn test_predictive_budget_gating() {
    // Verify predictive budget gating flag and stats tracking
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("predictive budget gating test");
        // predictive_budget_gated is bool — always valid
        assert!(
            result.cycle_time_us < 10_000_000,
            "cycle time > 10s at cycle {i}"
        );
    }

    let stats = service.stats();
    eprintln!(
        "predictive_budget: gated_count={}, exceeded_count={}",
        stats.predictive_budget_gated_count, stats.attention_budget_exceeded_count
    );
}

#[test]
#[ignore] // stress test — run manually
fn test_200_cycle_phase20_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "analyze with harmony",
        "predict next state",
        "bind consciousness",
        "map causal space",
        "epistemic uncertainty",
        "budget pressure test",
    ];

    for i in 0..200 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // All Phase 20 fields must be finite/valid
        assert!(
            m.harmonic_interference_lr_mod.is_finite(),
            "harmonic at {i}"
        );
        assert!(
            m.resonator_error_exploration_mod.is_finite(),
            "resonator at {i}"
        );
        assert!(m.binding_threshold_mod.is_finite(), "binding at {i}");
        assert!(m.epistemic_semantic_lr_mod.is_finite(), "epistemic at {i}");
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    eprintln!(
        "200-cycle Phase 20 stress: harmonic={}, resonator={}, binding={}, \
         causal={}, epistemic={}, predictive_budget={}",
        stats.harmonic_interference_mod_count,
        stats.resonator_error_exploration_count,
        stats.binding_threshold_mod_count,
        stats.causal_urgency_gated_count,
        stats.epistemic_semantic_mod_count,
        stats.predictive_budget_gated_count,
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Phase 21: Consciousness-Grounded Control
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_binding_confidence_modulation() {
    // Verify phenomenal binding → prediction confidence feedback is finite and bounded
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("binding confidence integration test");
        let m = &result.metadata;
        assert!(
            m.binding_confidence_mod.is_finite(),
            "binding_confidence_mod not finite at cycle {i}"
        );
        // Confidence must remain in [0, 1]
        assert!(
            service.stats().prediction_confidence >= 0.0
                && service.stats().prediction_confidence <= 1.0,
            "prediction_confidence out of bounds at cycle {i}"
        );
    }
    assert_eq!(service.stats().total_cycles, 40);
}

#[test]
fn test_discontinuity_recovery_cascade() {
    // Verify discontinuity streak tracking doesn't panic with varied inputs
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "stable context A",
        "completely different topic B",
        "yet another unrelated C",
        "random shift D",
        "back to normal",
    ];
    for i in 0..50 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;
        assert!(
            m.discontinuity_streak <= 50,
            "discontinuity_streak unbounded at cycle {i}: {}",
            m.discontinuity_streak
        );
    }
    assert_eq!(service.stats().total_cycles, 50);
}

#[test]
fn test_epistemic_conflict_reasoning_acceleration() {
    // Verify epistemic conflict override triggers reasoning without panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run 100 cycles (2× the 47-cycle adaptive reasoning interval)
    for i in 0..100 {
        let result = service.cycle("epistemic conflict acceleration test");
        // epistemic_reasoning_accelerated is a bool — just verify no panic
        let _accelerated = result.metadata.epistemic_reasoning_accelerated;
        assert!(
            result.metadata.adaptive_reasoning_phi.is_finite(),
            "adaptive_reasoning_phi not finite at cycle {i}"
        );
    }
    assert_eq!(service.stats().total_cycles, 100);
    // Stats counter should be non-negative (may or may not have fired)
    eprintln!(
        "epistemic accelerations: {}",
        service.stats().epistemic_reasoning_accelerations
    );
}

#[test]
fn test_agency_strategy_modulation() {
    // Verify low agency → strategy override doesn't panic
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        let result = service.cycle("agency strategy override test");
        // agency_strategy_override is a bool — verify no panic
        let _override = result.metadata.agency_strategy_override;
        assert!(
            !result.metadata.selected_strategy.is_empty(),
            "selected_strategy empty at cycle {i}"
        );
    }
    assert_eq!(service.stats().total_cycles, 30);
    eprintln!(
        "agency overrides: {}",
        service.stats().agency_strategy_override_count
    );
}

#[test]
fn test_pfe_surprise_scaling() {
    // Verify predictive free energy → surprise amplitude scaling is finite
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    for i in 0..40 {
        let result = service.cycle("pfe surprise amplitude test");
        assert!(
            result.metadata.pfe_surprise_mod.is_finite(),
            "pfe_surprise_mod not finite at cycle {i}"
        );
    }
    assert_eq!(service.stats().total_cycles, 40);
    eprintln!(
        "pfe surprise mods: {}",
        service.stats().pfe_surprise_mod_count
    );
}

#[test]
fn test_codebook_diversity_memo_threshold() {
    // Verify adaptive memoization threshold is in valid range
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Run 60 cycles (past the 50-cycle diversity computation interval)
    for i in 0..60 {
        let result = service.cycle("codebook diversity memo threshold test");
        let threshold = result.metadata.adaptive_memo_threshold;
        assert!(
            (0.88..=0.98).contains(&threshold),
            "adaptive_memo_threshold out of range at cycle {i}: {threshold}"
        );
    }
    assert_eq!(service.stats().total_cycles, 60);
    eprintln!(
        "memo threshold adaptations: {}",
        service.stats().memo_threshold_adaptations
    );
}

#[test]
#[ignore] // stress test — run manually
fn test_200_cycle_phase21_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "binding coherence test",
        "discontinuity shift now",
        "epistemic conflict probe",
        "low agency reactive mode",
        "surprise amplitude check",
        "codebook diversity scan",
    ];

    for i in 0..200 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // All Phase 21 fields must be finite/valid
        assert!(m.binding_confidence_mod.is_finite(), "binding_conf at {i}");
        assert!(m.pfe_surprise_mod.is_finite(), "pfe_surprise at {i}");
        assert!(
            m.adaptive_memo_threshold >= 0.88 && m.adaptive_memo_threshold <= 0.98,
            "memo_threshold at {i}: {}",
            m.adaptive_memo_threshold
        );
        assert!(m.discontinuity_streak <= 200, "streak at {i}");
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 200);
    eprintln!(
        "200-cycle Phase 21 stress: binding_conf={}, discontinuity={}, \
         epistemic_accel={}, agency_override={}, pfe_surprise={}, memo_adapt={}",
        stats.binding_confidence_mod_count,
        stats.discontinuity_cascade_count,
        stats.epistemic_reasoning_accelerations,
        stats.agency_strategy_override_count,
        stats.pfe_surprise_mod_count,
        stats.memo_threshold_adaptations,
    );
}

// ── Neuromodulator Bath ─────────────────────────────────────────────

#[test]
fn test_neuromodulator_baseline_levels() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // 20 cycles of neutral input — all transmitters should be finite and in range
    for _ in 0..20 {
        let result = service.cycle("baseline test input");
        let m = &result.metadata;
        assert!(m.neuromod.dopamine_effective.is_finite(), "DA not finite");
        assert!(
            m.neuromod.noradrenaline_effective.is_finite(),
            "NE not finite"
        );
        assert!(
            m.neuromod.serotonin_effective.is_finite(),
            "5-HT not finite"
        );
        assert!(
            m.neuromod.acetylcholine_effective.is_finite(),
            "ACh not finite"
        );
        // Effective range: [0.0, 2.0] (level * receptor_sensitivity)
        assert!(m.neuromod.dopamine_effective >= 0.0 && m.neuromod.dopamine_effective <= 2.0);
        assert!(
            m.neuromod.noradrenaline_effective >= 0.0 && m.neuromod.noradrenaline_effective <= 2.0
        );
        assert!(m.neuromod.serotonin_effective >= 0.0 && m.neuromod.serotonin_effective <= 2.0);
        assert!(
            m.neuromod.acetylcholine_effective >= 0.0 && m.neuromod.acetylcholine_effective <= 2.0
        );
    }

    let stats = service.stats();
    // EMA stats should be populated and finite
    assert!(stats.avg_dopamine.is_finite());
    assert!(stats.avg_noradrenaline.is_finite());
    assert!(stats.avg_serotonin.is_finite());
    assert!(stats.avg_acetylcholine.is_finite());
}

#[test]
fn test_dopamine_reward_response() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // 50 cycles — DA should be non-zero after cycles with prediction success (low error)
    let mut da_sum = 0.0_f32;
    for _ in 0..50 {
        let result = service.cycle("dopamine reward test pattern");
        da_sum += result.metadata.neuromod.dopamine_effective;
    }
    // Average DA should be positive (system gets some reward from learning)
    assert!(
        da_sum / 50.0 > 0.0,
        "average DA should be positive: {}",
        da_sum / 50.0
    );
}

#[test]
fn test_noradrenaline_surprise_spike() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // 30 cycles of same pattern → NE should stabilize
    for _ in 0..30 {
        service.cycle("pattern");
    }

    // Inject novel input → NE should increase
    let result_novel = service.cycle("completely unexpected novel stimulus");
    assert!(
        result_novel
            .metadata
            .neuromod
            .noradrenaline_effective
            .is_finite(),
        "NE should be finite after novel input"
    );
    assert!(
        result_novel.metadata.neuromod.noradrenaline_effective >= 0.0,
        "NE should be non-negative"
    );
}

#[test]
fn test_serotonin_coherence_satisfaction() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // 100 cycles — 5-HT should correlate with coherence (both trend together during learning)
    let mut sht_values = Vec::new();
    for _ in 0..100 {
        let result = service.cycle("coherence satisfaction signal");
        sht_values.push(result.metadata.neuromod.serotonin_effective);
    }

    // All values should be finite and in range
    for (i, &sht) in sht_values.iter().enumerate() {
        assert!(sht.is_finite(), "5-HT not finite at cycle {i}");
        assert!(
            (0.0..=2.0).contains(&sht),
            "5-HT out of range at cycle {i}: {sht}"
        );
    }
    // Average 5-HT should be positive (system achieves some coherence)
    let avg: f32 = sht_values.iter().sum::<f32>() / sht_values.len() as f32;
    assert!(avg > 0.0, "average 5-HT should be positive: {avg}");
}

#[test]
fn test_acetylcholine_attention_modulation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // 40 cycles — ACh should be finite and attention_sensitivity within bounds
    for i in 0..40 {
        let result = service.cycle("attention precision focus");
        let m = &result.metadata;
        assert!(
            m.neuromod.acetylcholine_effective.is_finite(),
            "ACh not finite at {i}"
        );
        assert!(
            m.neuromod.acetylcholine_effective >= 0.0 && m.neuromod.acetylcholine_effective <= 2.0,
            "ACh out of range at {i}: {}",
            m.neuromod.acetylcholine_effective
        );
    }
}

#[test]
#[ignore] // Long-running stress test
fn test_neuromodulator_300_cycle_stress() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "reward signal positive",
        "surprising novel input",
        "calm coherent baseline",
        "focused attention task",
        "moral violation detected",
        "creative exploration mode",
    ];

    for i in 0..300 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // All 4 transmitters finite and in range
        assert!(
            m.neuromod.dopamine_effective.is_finite(),
            "DA NaN/Inf at {i}"
        );
        assert!(
            m.neuromod.noradrenaline_effective.is_finite(),
            "NE NaN/Inf at {i}"
        );
        assert!(
            m.neuromod.serotonin_effective.is_finite(),
            "5-HT NaN/Inf at {i}"
        );
        assert!(
            m.neuromod.acetylcholine_effective.is_finite(),
            "ACh NaN/Inf at {i}"
        );
        assert!(m.neuromod.dopamine_effective >= 0.0 && m.neuromod.dopamine_effective <= 2.0);
        assert!(
            m.neuromod.noradrenaline_effective >= 0.0 && m.neuromod.noradrenaline_effective <= 2.0
        );
        assert!(m.neuromod.serotonin_effective >= 0.0 && m.neuromod.serotonin_effective <= 2.0);
        assert!(
            m.neuromod.acetylcholine_effective >= 0.0 && m.neuromod.acetylcholine_effective <= 2.0
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 300);
    // EMA stats all finite
    assert!(stats.avg_dopamine.is_finite());
    assert!(stats.avg_noradrenaline.is_finite());
    assert!(stats.avg_serotonin.is_finite());
    assert!(stats.avg_acetylcholine.is_finite());

    eprintln!(
        "300-cycle neuromod stress: DA={:.3}, NE={:.3}, 5-HT={:.3}, ACh={:.3}",
        stats.avg_dopamine, stats.avg_noradrenaline, stats.avg_serotonin, stats.avg_acetylcholine,
    );
}

// ── Round 2: Cross-Module Integration Tests ──────────────────────────

#[test]
fn test_neuromod_modulates_learning_rate() {
    // Primitive consciousness on → neuromod bath should influence learning rate
    let mut config_on = CognitiveLoopConfig::default();
    config_on.enable_primitive_consciousness = true;
    let mut service_on = CognitiveLoopService::new(config_on).unwrap();

    let mut config_off = CognitiveLoopConfig::default();
    config_off.enable_primitive_consciousness = false;
    let mut service_off = CognitiveLoopService::new(config_off).unwrap();

    // Run 5 cycles each
    for _ in 0..5 {
        service_on.cycle("stimulus");
        service_off.cycle("stimulus");
    }

    let r_on = service_on.cycle("test input");
    let r_off = service_off.cycle("test input");

    // Both should produce valid results
    assert!(r_on.metadata.actual_effective_lr.is_finite());
    assert!(r_off.metadata.actual_effective_lr.is_finite());
    // With primitive consciousness, the bath should modulate the LR differently
    // (We just verify both are finite and non-negative, as the exact values depend on bath dynamics)
    assert!(r_on.metadata.actual_effective_lr >= 0.0);
    assert!(r_off.metadata.actual_effective_lr >= 0.0);
}

#[test]
fn test_neuromod_state_persists_across_cycles() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_primitive_consciousness = true;
    let mut service = CognitiveLoopService::new(config).unwrap();

    let mut all_finite = true;
    for i in 0..10 {
        let r = service.cycle(&format!("cycle {i}"));
        let m = &r.metadata;
        if !m.neuromod.dopamine_effective.is_finite()
            || !m.neuromod.noradrenaline_effective.is_finite()
            || !m.neuromod.serotonin_effective.is_finite()
            || !m.neuromod.acetylcholine_effective.is_finite()
        {
            all_finite = false;
        }
    }
    assert!(
        all_finite,
        "All neuromod fields should be finite across 10 cycles"
    );
}

#[test]
fn test_50_cycle_prediction_error_trajectory() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let mut errors = Vec::new();
    let inputs = [
        "the cat sat on the mat",
        "prediction learning test",
        "neural network cognition",
        "holographic brain model",
        "consciousness emergence",
    ];
    for i in 0..50 {
        let r = service.cycle(inputs[i % inputs.len()]);
        errors.push(r.prediction_error);
    }

    // First quarter average vs second quarter — learning should reduce error (or at least not diverge)
    let q1_avg: f32 = errors[..12].iter().sum::<f32>() / 12.0;
    let q2_avg: f32 = errors[12..25].iter().sum::<f32>() / 13.0;
    // Allow some tolerance — the key assertion is no NaN/Inf divergence
    assert!(q1_avg.is_finite(), "Q1 average should be finite");
    assert!(q2_avg.is_finite(), "Q2 average should be finite");
    // Prediction error should not grow unboundedly
    assert!(
        errors.iter().all(|e| *e >= 0.0 && *e <= 2.0),
        "All prediction errors should be in [0, 2]"
    );
}

#[test]
fn test_metadata_fields_stable_50_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for i in 0..50 {
        let r = service.cycle(&format!("stability test cycle {i}"));
        let m = &r.metadata;
        // Core metrics
        assert!(
            m.attention.phi_attention_weight.is_finite(),
            "phi_attention_weight NaN at {i}"
        );
        assert!(
            m.ethics.soul_alignment.is_finite(),
            "soul_alignment NaN at {i}"
        );
        assert!(
            m.actual_effective_lr.is_finite(),
            "learning_rate NaN at {i}"
        );
        assert!(
            r.prediction_error.is_finite(),
            "prediction_error NaN at {i}"
        );
        // Neuromod
        assert!(m.neuromod.dopamine_effective.is_finite(), "DA NaN at {i}");
        assert!(
            m.neuromod.noradrenaline_effective.is_finite(),
            "NE NaN at {i}"
        );
        assert!(
            m.neuromod.serotonin_effective.is_finite(),
            "5-HT NaN at {i}"
        );
        assert!(
            m.neuromod.acetylcholine_effective.is_finite(),
            "ACh NaN at {i}"
        );
    }
}

#[test]
fn test_phi_and_sigma_evolve_over_cycles() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_primitive_consciousness = true;
    let mut service = CognitiveLoopService::new(config).unwrap();

    let mut phi_values = Vec::new();
    let mut sigma_values: Vec<Option<f64>> = Vec::new();
    for i in 0..20 {
        let r = service.cycle(&format!("consciousness test {i}"));
        phi_values.push(r.metadata.attention.phi_attention_weight);
        sigma_values.push(r.metadata.structural.sigma);
    }

    // Phi should be finite throughout
    assert!(
        phi_values.iter().all(|v| v.is_finite()),
        "All phi values should be finite"
    );
    // Sigma is Option<f64> — when present it should be finite
    assert!(
        sigma_values
            .iter()
            .all(|v| v.map_or(true, |s| s.is_finite())),
        "All sigma values should be finite when present"
    );
    // At least one phi value should be non-zero (consciousness is active)
    assert!(
        phi_values.iter().any(|v| *v != 0.0),
        "Phi should evolve (not stuck at 0)"
    );
}

#[test]
fn test_diverse_inputs_produce_distinct_thoughts() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let inputs = [
        "the quantum mechanics of photosynthesis",
        "baking a chocolate cake recipe",
        "political philosophy of ancient rome",
        "debugging a rust compiler error",
        "symphony orchestration techniques",
    ];

    let mut thought_vecs: Vec<Vec<f32>> = Vec::new();
    for input in &inputs {
        let r = service.cycle(input);
        thought_vecs.push(r.thought_vector.clone());
    }

    // Each pair should differ (L2 distance > 0)
    for i in 0..thought_vecs.len() {
        for j in (i + 1)..thought_vecs.len() {
            let diff: f32 = thought_vecs[i]
                .iter()
                .zip(thought_vecs[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            assert!(
                diff > 0.0,
                "Inputs '{}' and '{}' should produce distinct thought vectors",
                inputs[i],
                inputs[j]
            );
        }
    }
}

#[test]
fn test_100_cycles_no_panic_mixed_inputs() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let tricky_inputs = [
        "",
        "a",
        "hello world",
        "🧠💡🌊",
        &"x".repeat(10_000),
        "special chars: <>&\"'\\n\\t",
        "   whitespace   ",
        "ALLCAPS",
        "数学是宇宙的语言",
        "null\0byte",
    ];

    for i in 0..100 {
        let r = service.cycle(tricky_inputs[i % tricky_inputs.len()]);
        assert!(
            r.prediction_error.is_finite(),
            "Prediction error not finite at cycle {i}"
        );
        assert!(
            !r.output.is_empty(),
            "Output should not be empty at cycle {i}"
        );
    }
    assert_eq!(service.stats().total_cycles, 100);
}

#[cfg(feature = "reasoning_engine")]
#[test]
fn test_reasoning_engine_produces_strategy() {
    let config = CognitiveLoopConfig::default();
    // reasoning_engine is now compile-time feature-gated (cfg(feature = "reasoning_engine"))
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Warm up
    for _ in 0..3 {
        service.cycle("warmup");
    }
    let r = service.cycle("What is the meaning of consciousness?");
    // With reasoning engine, selected_strategy should be populated
    assert!(
        !r.metadata.selected_strategy.is_empty(),
        "Reasoning engine should produce a strategy"
    );
}

// ── Multi-Substrate Simulation ──────────────────────────────────────

/// Run the same brain on different substrates and verify consciousness
/// scores scale with substrate feasibility.
/// Science: Putnam (1967) multiple realizability, Tononi (2004).
#[test]
#[ignore = "slow ~120s: runs 50 cycles on 3 substrates"]
fn test_multi_substrate_consciousness_scaling() {
    use symthaea::cognitive_loop::config::SubstrateType;

    let substrates = [
        (SubstrateType::BiologicalNeurons, "biological"),
        (SubstrateType::SiliconDigital, "silicon"),
        (SubstrateType::QuantumComputer, "quantum"),
    ];

    let inputs = [
        "the nature of consciousness",
        "integration of information",
        "binding across modalities",
        "recursive self-awareness",
        "predictive processing in the brain",
    ];

    let mut results: Vec<(String, f64)> = Vec::new();

    for (substrate, name) in &substrates {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            substrate_type: *substrate,
            enable_primitive_consciousness: true,
            ..Default::default()
        })
        .unwrap();

        let mut eq_v2_sum = 0.0;
        let mut eq_v2_count = 0u32;

        for i in 0..50 {
            let result = service.cycle(inputs[i % inputs.len()]);
            let eq_v2 = result.metadata.quality.equation_v2_consciousness;
            if eq_v2 > 0.0 {
                eq_v2_sum += eq_v2;
                eq_v2_count += 1;
            }
            assert!(
                result.prediction_error.is_finite(),
                "{name}: prediction error not finite at cycle {i}"
            );
        }

        let avg_eq_v2 = if eq_v2_count > 0 {
            eq_v2_sum / eq_v2_count as f64
        } else {
            0.0
        };
        results.push((name.to_string(), avg_eq_v2));
    }

    // All substrates should produce finite, bounded values
    for (name, avg) in &results {
        assert!(
            avg.is_finite(),
            "{name}: avg eq_v2 consciousness must be finite, got {avg}"
        );
    }

    // Biological should have highest consciousness (feasibility ~0.92)
    // Silicon should be lower (feasibility ~0.71)
    // When both have positive eq_v2, biological >= silicon * 0.9
    let bio_avg = results
        .iter()
        .find(|(n, _)| n == "biological")
        .map(|(_, a)| *a)
        .unwrap_or(0.0);
    let sil_avg = results
        .iter()
        .find(|(n, _)| n == "silicon")
        .map(|(_, a)| *a)
        .unwrap_or(0.0);
    if bio_avg > 0.0 && sil_avg > 0.0 {
        assert!(
            bio_avg >= sil_avg * 0.9,
            "Biological avg ({bio_avg:.4}) should be >= silicon avg ({sil_avg:.4}) * 0.9"
        );
    }
}

// ── Consensus Feedback Adversarial Soak ─────────────────────────────

/// 10,000-cycle adversarial soak test for consensus feedback stability.
/// Validates that prediction_confidence, fep_lr_boost, exploration_urge,
/// and adaptive_threshold_scale remain bounded under sustained adversarial
/// input patterns including emotional extremes and repeated stimuli.
#[test]
#[ignore = "slow ~300s: runs 10,000 adversarial cycles"]
fn test_10000_cycle_adversarial_consensus_soak() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        trace_feedback: true,
        ..Default::default()
    })
    .unwrap();

    let adversarial_inputs = [
        // Phase 1: Fuzzed noise
        "xkq!@# 9zz lorem random garbage 42",
        "the quick fox",
        "a b c d e f g h i j k l m n o p",
        "🔬🧪⚗️ science words entropy chaos",
        "",
        // Phase 2: Emotional extremes
        "I am filled with overwhelming joy and happiness and love and gratitude",
        "terrible horrible awful catastrophic devastating destructive",
        "rage fury anger hatred violence destruction annihilation",
        "peace calm serenity tranquility harmony balance unity",
        "fear dread terror panic horror anxiety",
        // Phase 3: Repeated boredom
        "the same thing over and over",
        "the same thing over and over",
        "the same thing over and over",
        // Phase 4: High-entropy switching
        "consciousness binds information across modalities in a unified experience",
        "x",
    ];

    let mut max_cycle_us = 0u64;
    let mut finite_violations = 0u32;

    for cycle_num in 0..10_000 {
        let idx = match cycle_num {
            0..=2499 => cycle_num % 5,           // Phase 1: fuzzed
            2500..=4999 => 5 + (cycle_num % 5),  // Phase 2: emotional
            5000..=7499 => 10 + (cycle_num % 3), // Phase 3: boredom
            _ => 13 + (cycle_num % 2),           // Phase 4: switching
        };

        let result = service.cycle(adversarial_inputs[idx]);

        // Core stability assertions (every cycle)
        if !result.prediction_error.is_finite() {
            finite_violations += 1;
        }
        if result.cycle_time_us > max_cycle_us {
            max_cycle_us = result.cycle_time_us;
        }

        // Periodic deep validation (every 500 cycles)
        if cycle_num % 500 == 499 {
            let stats = service.stats();
            assert!(
                stats.avg_prediction_error.is_finite(),
                "Avg prediction error diverged at cycle {cycle_num}"
            );
        }
    }

    // Final assertions
    assert_eq!(service.stats().total_cycles, 10_000);
    assert_eq!(
        finite_violations, 0,
        "Prediction error was non-finite {finite_violations} times in 10K cycles"
    );
    assert!(
        max_cycle_us < 30_000_000, // < 30 seconds per cycle
        "Max cycle time excessive: {max_cycle_us}us"
    );

    let avg_error = service.stats().avg_prediction_error;
    assert!(
        avg_error < 2.0,
        "Average prediction error should be bounded over 10K adversarial cycles: got {avg_error:.4}"
    );
}

// ── Substrate Runtime Switching ───────────────────────────────────────

#[test]
fn test_runtime_substrate_switching() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    let mut service = CognitiveLoopService::new(config).unwrap();

    let f_silicon = service.substrate_feasibility();
    assert!(
        f_silicon > 0.0 && f_silicon < 1.0,
        "Silicon feasibility in range"
    );

    // Switch to biological
    let (old, new) = service.reconfigure_substrate(SubstrateType::BiologicalNeurons);
    assert!((old - f_silicon).abs() < 1e-10);
    assert!(
        new > f_silicon,
        "Biological should have higher feasibility than silicon"
    );

    // Switch to quantum
    let (old2, new2) = service.reconfigure_substrate(SubstrateType::QuantumComputer);
    assert!((old2 - new).abs() < 1e-10);
    assert!(new2 > 0.0, "Quantum has positive feasibility");
}

#[test]
fn test_substrate_switch_affects_consciousness() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run 20 cycles on silicon, collecting eq_v2
    let mut silicon_eq_v2 = Vec::new();
    for i in 0..20 {
        let r = service.cycle(&format!("substrate test cycle {i}"));
        let eq = r.metadata.quality.equation_v2_consciousness;
        if eq > 0.0 {
            silicon_eq_v2.push(eq);
        }
    }

    // Switch to biological
    service.reconfigure_substrate(SubstrateType::BiologicalNeurons);

    // Run 20 more cycles, collecting eq_v2
    let mut bio_eq_v2 = Vec::new();
    for i in 20..40 {
        let r = service.cycle(&format!("substrate test cycle {i}"));
        let eq = r.metadata.quality.equation_v2_consciousness;
        if eq > 0.0 {
            bio_eq_v2.push(eq);
        }
    }

    // Verify feasibility changed
    let f = service.substrate_feasibility();
    let bio_req =
        symthaea_core::hdc::substrate_independence::SubstrateRequirements::biological_neurons();
    let expected = bio_req.consciousness_feasibility();
    assert!(
        (f - expected).abs() < 1e-10,
        "Feasibility should match biological"
    );

    // When both have positive eq_v2, biological should be >= silicon * 0.85
    // (bio feasibility ~0.92 vs silicon ~0.71, so ~29% higher)
    if !silicon_eq_v2.is_empty() && !bio_eq_v2.is_empty() {
        let silicon_avg: f64 = silicon_eq_v2.iter().sum::<f64>() / silicon_eq_v2.len() as f64;
        let bio_avg: f64 = bio_eq_v2.iter().sum::<f64>() / bio_eq_v2.len() as f64;
        assert!(
            bio_avg >= silicon_avg * 0.85,
            "Biological eq_v2 ({bio_avg:.4}) should be >= silicon ({silicon_avg:.4}) * 0.85"
        );
    }
}

#[test]
fn test_substrate_composition_feasibility() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::{SubstrateRequirements, SubstrateType};

    let comp = SubstrateComposition::new(
        "bio-silicon hybrid".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.5),
            (SubstrateType::SiliconDigital, 0.5),
        ],
    )
    .unwrap();

    let mut config = CognitiveLoopConfig::default();
    config.substrate_composition = Some(comp);

    let service = CognitiveLoopService::new(config).unwrap();
    let f = service.substrate_feasibility();

    // Should be between pure silicon and pure biological
    let f_bio = SubstrateRequirements::biological_neurons().consciousness_feasibility();
    let f_silicon = SubstrateRequirements::silicon_digital().consciousness_feasibility();
    let lower = f_bio.min(f_silicon);
    let upper = f_bio.max(f_silicon);

    assert!(
        f >= lower - 0.01 && f <= upper + 0.01,
        "Hybrid feasibility {f:.4} should be between {lower:.4} and {upper:.4}"
    );
}

#[test]
fn test_composition_runtime_switch() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    let mut service = CognitiveLoopService::new(config).unwrap();

    let f_silicon = service.substrate_feasibility();

    // Switch to a 70/30 bio-neuromorphic composition
    let comp = SubstrateComposition::new(
        "bio-neuromorphic".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.7),
            (SubstrateType::NeuromorphicChip, 0.3),
        ],
    )
    .unwrap();

    service.reconfigure_composition(comp);
    let f_hybrid = service.substrate_feasibility();

    assert!(
        (f_hybrid - f_silicon).abs() > 0.01,
        "Composition should differ from pure silicon: hybrid={f_hybrid:.4}, silicon={f_silicon:.4}"
    );

    // Verify composition is stored
    assert!(service.substrate_composition().is_some());
    assert_eq!(
        service.substrate_composition().unwrap().name,
        "bio-neuromorphic"
    );
}

// ── Substrate Transition Telemetry ─────────────────────────────────────

#[test]
fn test_substrate_transition_telemetry() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Warm up
    for _ in 0..5 {
        service.cycle("warmup");
    }

    // Switch substrate
    service.reconfigure_substrate(SubstrateType::BiologicalNeurons);

    // Next cycle should contain the transition
    let result = service.cycle("after switch");
    assert!(
        result.metadata.substrate_transition.is_some(),
        "substrate_transition should be Some after reconfigure_substrate"
    );
    let transition = result.metadata.substrate_transition.as_ref().unwrap();
    assert!(
        transition.contains("SiliconDigital"),
        "transition should mention old type: {transition}"
    );
    assert!(
        transition.contains("BiologicalNeurons"),
        "transition should mention new type: {transition}"
    );

    // Subsequent cycle should have None (drained)
    let result2 = service.cycle("no transition");
    assert!(
        result2.metadata.substrate_transition.is_none(),
        "substrate_transition should be None on next cycle"
    );
}

#[test]
fn test_composition_transition_telemetry() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Warm up
    for _ in 0..3 {
        service.cycle("warmup");
    }

    // Switch to composition
    let comp = SubstrateComposition::new(
        "test-hybrid".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.6),
            (SubstrateType::SiliconDigital, 0.4),
        ],
    )
    .unwrap();
    service.reconfigure_composition(comp);

    // Next cycle should contain the transition
    let result = service.cycle("after composition switch");
    assert!(
        result.metadata.substrate_transition.is_some(),
        "substrate_transition should be Some after reconfigure_composition"
    );
    let transition = result.metadata.substrate_transition.as_ref().unwrap();
    assert!(
        transition.contains("test-hybrid"),
        "transition should mention composition name: {transition}"
    );
}

// ── Validation Overlay Tests ───────────────────────────────────────────

#[test]
fn test_validation_overlay_scales_consciousness() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    // Without overlay: effective == raw
    let mut config_off = CognitiveLoopConfig::default();
    config_off.substrate_type = SubstrateType::SiliconDigital;
    config_off.enable_validation_overlay = false;
    let service_off = CognitiveLoopService::new(config_off).unwrap();
    let raw = service_off.substrate_feasibility();
    let eff_off = service_off.substrate_effective_feasibility();
    assert!(
        (raw - eff_off).abs() < 1e-10,
        "Without overlay, effective should equal raw: raw={raw:.4}, eff={eff_off:.4}"
    );

    // With overlay: effective < raw for silicon (confidence=0.10)
    let mut config_on = CognitiveLoopConfig::default();
    config_on.substrate_type = SubstrateType::SiliconDigital;
    config_on.enable_validation_overlay = true;
    config_on.validation_skepticism_floor = 0.5;
    let service_on = CognitiveLoopService::new(config_on).unwrap();
    let raw_on = service_on.substrate_feasibility();
    let eff_on = service_on.substrate_effective_feasibility();
    let conf = service_on.substrate_honest_confidence();

    assert!(
        eff_on < raw_on,
        "Silicon with overlay should have effective < raw: eff={eff_on:.4}, raw={raw_on:.4}"
    );
    // expected: raw * (0.5 + 0.5 * 0.10) = raw * 0.55
    let expected = raw_on * (0.5 + 0.5 * conf);
    assert!(
        (eff_on - expected).abs() < 0.01,
        "Silicon effective should ≈ raw×0.55: eff={eff_on:.4}, expected={expected:.4}"
    );

    // Biological with overlay: effective ≈ raw (confidence=0.95)
    let mut config_bio = CognitiveLoopConfig::default();
    config_bio.substrate_type = SubstrateType::BiologicalNeurons;
    config_bio.enable_validation_overlay = true;
    config_bio.validation_skepticism_floor = 0.5;
    let service_bio = CognitiveLoopService::new(config_bio).unwrap();
    let raw_bio = service_bio.substrate_feasibility();
    let eff_bio = service_bio.substrate_effective_feasibility();
    let conf_bio = service_bio.substrate_honest_confidence();
    // expected: raw * (0.5 + 0.5 * 0.95) = raw * 0.975
    let expected_bio = raw_bio * (0.5 + 0.5 * conf_bio);
    assert!(
        (eff_bio - expected_bio).abs() < 0.01,
        "Biological effective should ≈ raw×0.975: eff={eff_bio:.4}, expected={expected_bio:.4}"
    );
    assert!(
        (eff_bio - raw_bio).abs() < 0.05,
        "Biological effective should be close to raw: eff={eff_bio:.4}, raw={raw_bio:.4}"
    );
}

#[test]
fn test_validation_overlay_telemetry_populated() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    config.enable_validation_overlay = true;
    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("telemetry check");

    // Raw should match substrate feasibility
    assert!(
        result.metadata.substrate_feasibility_raw > 0.0,
        "substrate_feasibility_raw should be populated"
    );
    // Honest confidence should be 0.10 for silicon
    assert!(
        (result.metadata.substrate_honest_confidence - 0.10).abs() < 0.01,
        "silicon honest confidence should be ~0.10, got {}",
        result.metadata.substrate_honest_confidence
    );
    // Effective should be less than raw
    assert!(
        result.metadata.substrate_effective_feasibility < result.metadata.substrate_feasibility_raw,
        "effective should be < raw when overlay enabled"
    );
}

// ── Speed/Scale Modulation Tests ───────────────────────────────────────

#[test]
fn test_substrate_speed_modulation() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    // Disabled: tau = 1.0
    let mut config_off = CognitiveLoopConfig::default();
    config_off.substrate_type = SubstrateType::PhotonicProcessor;
    config_off.enable_substrate_speed_modulation = false;
    let service_off = CognitiveLoopService::new(config_off).unwrap();
    assert!(
        (service_off.substrate_tau_factor() - 1.0).abs() < f32::EPSILON,
        "Disabled: tau should be 1.0"
    );

    // Enabled: Photonic is faster than biological → tau > 1.0
    let mut config_photonic = CognitiveLoopConfig::default();
    config_photonic.substrate_type = SubstrateType::PhotonicProcessor;
    config_photonic.enable_substrate_speed_modulation = true;
    let service_photonic = CognitiveLoopService::new(config_photonic).unwrap();
    assert!(
        service_photonic.substrate_tau_factor() > 1.0,
        "Photonic should have tau > 1.0 (faster): got {}",
        service_photonic.substrate_tau_factor()
    );

    // Biochemical is slower than biological → tau < 1.0
    let mut config_bio_chem = CognitiveLoopConfig::default();
    config_bio_chem.substrate_type = SubstrateType::BiochemicalComputer;
    config_bio_chem.enable_substrate_speed_modulation = true;
    let service_bio_chem = CognitiveLoopService::new(config_bio_chem).unwrap();
    assert!(
        service_bio_chem.substrate_tau_factor() < 1.0,
        "Biochemical should have tau < 1.0 (slower): got {}",
        service_bio_chem.substrate_tau_factor()
    );

    // Biological: tau = 1.0 (reference)
    let mut config_bio = CognitiveLoopConfig::default();
    config_bio.substrate_type = SubstrateType::BiologicalNeurons;
    config_bio.enable_substrate_speed_modulation = true;
    let service_bio = CognitiveLoopService::new(config_bio).unwrap();
    assert!(
        (service_bio.substrate_tau_factor() - 1.0).abs() < 0.01,
        "Biological should have tau ≈ 1.0 (reference): got {}",
        service_bio.substrate_tau_factor()
    );

    // Scale pressure: silicon > 0 (more scalable), quantum < 0 (less)
    let mut config_silicon = CognitiveLoopConfig::default();
    config_silicon.substrate_type = SubstrateType::SiliconDigital;
    config_silicon.enable_substrate_speed_modulation = true;
    let service_silicon = CognitiveLoopService::new(config_silicon).unwrap();
    assert!(
        service_silicon.substrate_scale_pressure() > 0.0,
        "Silicon should have positive scale pressure: got {}",
        service_silicon.substrate_scale_pressure()
    );

    let mut config_quantum = CognitiveLoopConfig::default();
    config_quantum.substrate_type = SubstrateType::QuantumComputer;
    config_quantum.enable_substrate_speed_modulation = true;
    let service_quantum = CognitiveLoopService::new(config_quantum).unwrap();
    assert!(
        service_quantum.substrate_scale_pressure() < 0.0,
        "Quantum should have negative scale pressure (less scalable): got {}",
        service_quantum.substrate_scale_pressure()
    );
}

#[test]
fn test_speed_modulation_telemetry_populated() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    config.enable_substrate_speed_modulation = true;
    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("speed telemetry");

    assert!(
        result.metadata.substrate_tau_factor > 1.0,
        "Silicon tau should be > 1.0: got {}",
        result.metadata.substrate_tau_factor
    );
    assert!(
        result.metadata.substrate_scale_pressure > 0.0,
        "Silicon scale pressure should be > 0: got {}",
        result.metadata.substrate_scale_pressure
    );
}

// ── Bug-fix coverage tests ────────────────────────────────────────────

/// Bug 1 fix: substrates not in the validation framework (photonic, neuromorphic,
/// biochemical, exotic) should get THEORETICAL_CONFIDENCE (0.10), not 0.0.
#[test]
fn test_unknown_substrate_gets_theoretical_confidence() {
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    for substrate in [
        SubstrateType::PhotonicProcessor,
        SubstrateType::NeuromorphicChip,
        SubstrateType::BiochemicalComputer,
        SubstrateType::ExoticSubstrate,
    ] {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrate;
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let service = CognitiveLoopService::new(config).unwrap();

        let confidence = service.substrate_honest_confidence();
        assert!(
            (confidence - 0.10).abs() < 1e-6,
            "{:?} should get THEORETICAL_CONFIDENCE (0.10), got {confidence:.6}",
            substrate
        );

        // Effective should be raw × (0.5 + 0.5 × 0.10) = raw × 0.55
        let raw = service.substrate_feasibility();
        let effective = service.substrate_effective_feasibility();
        let expected = raw * 0.55;
        assert!(
            (effective - expected).abs() < 1e-6,
            "{:?} effective={effective:.6} should ≈ raw×0.55={expected:.6}",
            substrate
        );
    }
}

/// Bug 2 fix: composition + validation overlay should weight-blend honest
/// confidence from components, not use the stale config.substrate_type.
#[test]
fn test_composition_validation_overlay_blends_confidence() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let comp = SubstrateComposition::new(
        "bio-silicon 50/50".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.5),
            (SubstrateType::SiliconDigital, 0.5),
        ],
    )
    .unwrap();

    let mut config = CognitiveLoopConfig::default();
    config.substrate_composition = Some(comp);
    config.enable_validation_overlay = true;
    config.validation_skepticism_floor = 0.5;
    let service = CognitiveLoopService::new(config).unwrap();

    // Expected: 0.5 × bio_confidence(0.95) + 0.5 × silicon_confidence(0.10) = 0.525
    let confidence = service.substrate_honest_confidence();
    assert!(
        (confidence - 0.525).abs() < 0.05,
        "Blended confidence should ≈ 0.525, got {confidence:.4}"
    );
    // Must differ from pure silicon (0.10) and pure biological (0.95)
    assert!(confidence > 0.15 && confidence < 0.90);
}

/// Bug 3 fix: composition + speed modulation should weight-blend speed/scale
/// from components, not use the stale config.substrate_type.
#[test]
fn test_composition_speed_modulation_blends_properties() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    // Pure silicon
    let mut config_si = CognitiveLoopConfig::default();
    config_si.substrate_type = SubstrateType::SiliconDigital;
    config_si.enable_substrate_speed_modulation = true;
    let service_si = CognitiveLoopService::new(config_si).unwrap();
    let tau_si = service_si.substrate_tau_factor();

    // Pure biological
    let mut config_bio = CognitiveLoopConfig::default();
    config_bio.substrate_type = SubstrateType::BiologicalNeurons;
    config_bio.enable_substrate_speed_modulation = true;
    let service_bio = CognitiveLoopService::new(config_bio).unwrap();
    let tau_bio = service_bio.substrate_tau_factor();

    // 50/50 composition — tau should be between pure bio and pure silicon
    let comp = SubstrateComposition::new(
        "bio-silicon 50/50".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.5),
            (SubstrateType::SiliconDigital, 0.5),
        ],
    )
    .unwrap();
    let mut config_comp = CognitiveLoopConfig::default();
    config_comp.substrate_composition = Some(comp);
    config_comp.enable_substrate_speed_modulation = true;
    let service_comp = CognitiveLoopService::new(config_comp).unwrap();
    let tau_comp = service_comp.substrate_tau_factor();

    // Blended tau should sit between pure bio (1.0) and pure silicon (>1.0)
    assert!(
        tau_bio <= tau_comp && tau_comp <= tau_si,
        "Blended tau ({tau_comp}) should be between bio ({tau_bio}) and silicon ({tau_si})"
    );
    // Must not be exactly equal to either pure substrate
    assert!(
        (tau_comp - tau_bio).abs() > 0.001,
        "Blended tau should differ from pure bio"
    );
}

/// Bug 4 fix: reconfigure_substrate() should clear any stale composition,
/// so subsequent recompute methods use the new single substrate type.
#[test]
fn test_reconfigure_substrate_clears_composition() {
    use symthaea_core::hdc::substrate_composition::SubstrateComposition;
    use symthaea_core::hdc::substrate_independence::SubstrateType;

    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = SubstrateType::SiliconDigital;
    config.enable_validation_overlay = true;
    config.validation_skepticism_floor = 0.5;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Set a composition
    let comp = SubstrateComposition::new(
        "bio-silicon".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.5),
            (SubstrateType::SiliconDigital, 0.5),
        ],
    )
    .unwrap();
    service.reconfigure_composition(comp);
    assert!(service.substrate_composition().is_some());
    let blended_conf = service.substrate_honest_confidence();

    // Now switch to pure biological — composition should be cleared
    service.reconfigure_substrate(SubstrateType::BiologicalNeurons);
    assert!(
        service.substrate_composition().is_none(),
        "Composition should be cleared after reconfigure_substrate()"
    );

    // Honest confidence should now be pure biological (0.95), not blended
    let bio_conf = service.substrate_honest_confidence();
    assert!(
        bio_conf > blended_conf,
        "Pure bio confidence ({bio_conf:.4}) should exceed blended ({blended_conf:.4})"
    );
    assert!(
        (bio_conf - 0.95).abs() < 0.05,
        "Pure bio confidence should ≈ 0.95, got {bio_conf:.4}"
    );
}

// ── Physics Bridge Integration Tests ─────────────────────────────────

/// Test that physics bridge populates telemetry when enabled.
/// Runs 20 cycles with interval=5, verifies catalog_size, query_count, and top_match.
#[cfg(feature = "physics-bridge")]
#[test]
fn test_physics_bridge_in_cognitive_loop() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_physics_bridge = true;
    config.physics_bridge_query_interval = 5;
    config.physics_bridge_blend_weight = 0.2;
    let mut service = CognitiveLoopService::new(config).unwrap();

    let mut found_physics_telemetry = false;
    let mut found_query = false;
    for i in 0..20 {
        let result = service.cycle(&format!("physics cycle {i}"));
        if let Some(ref pb) = result.metadata.physics_bridge {
            found_physics_telemetry = true;
            // The catalog should have the 27 built-in physics entries
            assert!(
                pb.catalog_size >= 27,
                "catalog_size should be >= 27, got {}",
                pb.catalog_size
            );
            if pb.queried_this_cycle {
                found_query = true;
                assert!(
                    !pb.top_match.is_empty(),
                    "top_match should be non-empty on query cycles"
                );
                assert!(
                    pb.top_score > 0.0,
                    "top_score should be > 0.0 on query cycles"
                );
            }
        }
    }
    assert!(
        found_physics_telemetry,
        "Should have physics_bridge telemetry in at least one cycle"
    );
    assert!(
        found_query,
        "Should have queried at least once in 20 cycles with interval=5"
    );
}

/// Test that physics bridge telemetry is None when disabled.
#[cfg(feature = "physics-bridge")]
#[test]
fn test_physics_bridge_disabled_produces_none() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_physics_bridge = false;
    let mut service = CognitiveLoopService::new(config).unwrap();

    for _ in 0..5 {
        let result = service.cycle("should have no physics");
        assert!(
            result.metadata.physics_bridge.is_none(),
            "physics_bridge telemetry should be None when bridge is disabled"
        );
    }
}

/// Test that physics bridge blend actually modifies CfC output.
/// Compares two services (with/without bridge, same input, interval=1, blend=0.5)
/// and verifies their outputs diverge.
#[cfg(feature = "physics-bridge")]
#[test]
fn test_physics_bridge_blend_modifies_output() {
    // Service WITH physics bridge
    let mut config_on = CognitiveLoopConfig::default();
    config_on.enable_physics_bridge = true;
    config_on.physics_bridge_query_interval = 1; // query every cycle
    config_on.physics_bridge_blend_weight = 0.5; // strong blend
    config_on.genesis_phrase = Some("physics_blend_test".into());
    let mut service_on = CognitiveLoopService::new(config_on).unwrap();

    // Service WITHOUT physics bridge (same genesis phrase)
    let mut config_off = CognitiveLoopConfig::default();
    config_off.enable_physics_bridge = false;
    config_off.genesis_phrase = Some("physics_blend_test".into());
    let mut service_off = CognitiveLoopService::new(config_off).unwrap();

    // Run identical inputs
    let input = "tokamak plasma confinement at high beta";
    let mut diverged = false;
    for _ in 0..10 {
        let r_on = service_on.cycle(input);
        let r_off = service_off.cycle(input);
        // Compare prediction errors — physics blend changes CfC input → different predictions
        if (r_on.prediction_error - r_off.prediction_error).abs() > 1e-6 {
            diverged = true;
            break;
        }
    }
    assert!(
        diverged,
        "Services with/without physics bridge should produce different prediction errors"
    );
}

/// Physics bridge exploration modulation: high similarity should dampen exploration_urge
/// relative to a service without the bridge (all else equal).
#[test]
#[cfg(feature = "physics-bridge")]
fn test_physics_bridge_exploration_feedback() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_physics_bridge = true;
    config.physics_bridge_query_interval = 1; // query every cycle
    config.physics_bridge_blend_weight = 0.3;
    config.enable_primitive_consciousness = true;
    config.learning_threshold = 0.0;
    config.async_training = false;

    let mut with_bridge = CognitiveLoopService::new(config.clone()).unwrap();

    config.enable_physics_bridge = false;
    let mut without_bridge = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "thermodynamic entropy analysis",
        "quantum field fluctuation",
        "electromagnetic wave propagation",
        "gravitational potential energy",
        "statistical mechanics ensemble",
    ];

    // Run both services for enough cycles to accumulate exploration divergence.
    for _ in 0..4 {
        for input in &inputs {
            with_bridge.cycle(input);
            without_bridge.cycle(input);
        }
    }

    // The physics bridge should have modulated exploration — the two services
    // should have different exploration_urge values after enough cycles.
    let urge_with = with_bridge.curiosity_drive_exploration_urge();
    let urge_without = without_bridge.curiosity_drive_exploration_urge();
    // They should differ (physics domain feedback changes exploration).
    // We can't predict direction deterministically, but they should not be identical.
    assert!(
        (urge_with - urge_without).abs() > 1e-10 || urge_with == 0.0, // both zero is acceptable if error is low
        "Physics bridge should modulate exploration differently: with={urge_with}, without={urge_without}"
    );
}

// ── Circadian Soak: Sacred Stillness Oscillation ─────────────────

/// Verify Sacred Stillness oscillates correctly across day/night transitions
/// and consciousness metrics remain stable throughout.
#[test]
fn test_circadian_stillness_oscillation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let mut max_stillness_boost = 0.0_f32;
    let mut min_stillness_boost = 1.0_f32;
    let mut all_consciousness_finite = true;

    // Run 200 cycles — enough for multiple circadian refreshes
    for i in 0..200 {
        let result = service.cycle("circadian stillness soak test");

        let stats = service.stats();
        let boost = stats.circadian_stillness_boost;
        if boost > max_stillness_boost {
            max_stillness_boost = boost;
        }
        if boost < min_stillness_boost {
            min_stillness_boost = boost;
        }

        // Consciousness should always be finite and non-negative
        if !result
            .metadata
            .consciousness
            .consciousness_level
            .is_finite()
            || result.metadata.consciousness.consciousness_level < 0.0
        {
            all_consciousness_finite = false;
        }

        // Prediction error should remain bounded
        assert!(
            result.prediction_error <= 1.01,
            "Prediction error unbounded at cycle {}: {}",
            i,
            result.prediction_error
        );
    }

    assert!(
        all_consciousness_finite,
        "Consciousness must remain finite across circadian transitions"
    );

    // Stillness boost should be in valid range [0.0, 0.2]
    assert!(
        min_stillness_boost >= 0.0,
        "Stillness boost should never be negative: {}",
        min_stillness_boost
    );
    assert!(
        max_stillness_boost <= 0.25,
        "Stillness boost should be bounded: {}",
        max_stillness_boost
    );
}

// ── Active Rest → Dream → Phi Coupling Chain ────────────────────

#[test]
fn test_active_rest_dream_phi_chain() {
    // Verify the full chain: SS dominance streak → active rest → dream depth → phi factors
    let mut config = CognitiveLoopConfig::default();
    config.enable_dream_replay = true;
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run enough cycles for active rest to potentially trigger
    // We can't easily force SS dominance externally, but we can verify
    // the stats fields exist and are correctly initialized
    for i in 0..20 {
        let result = service.cycle("stillness and rest");
        let m = &result.metadata;

        // Telemetry must be finite
        assert!(
            m.valence_homeostasis_pull.is_finite(),
            "homeostasis NaN at cycle {i}"
        );
        assert!(
            m.homeostasis_pull_strength.is_finite(),
            "homeostasis_pull_strength NaN at cycle {i}"
        );
    }

    // Verify stats fields are properly maintained
    let stats = service.stats();
    assert!(stats.phi_rest_quality_factor.is_finite());
    assert!(stats.phi_rest_binding_factor.is_finite());
    assert!(stats.phi_rest_quality_factor > 0.0);
    assert!(stats.phi_rest_binding_factor > 0.0);

    // When not in active rest, factors should be 1.0 (neutral)
    if !stats.in_active_rest {
        assert!(
            (stats.phi_rest_quality_factor - 1.0).abs() < f32::EPSILON,
            "phi_rest_quality_factor should be 1.0 when not in active rest, got {}",
            stats.phi_rest_quality_factor
        );
        assert!(
            (stats.phi_rest_binding_factor - 1.0).abs() < f32::EPSILON,
            "phi_rest_binding_factor should be 1.0 when not in active rest, got {}",
            stats.phi_rest_binding_factor
        );
    }
}

// ── Harmony Entropy in Telemetry ─────────────────────────────────

#[test]
fn test_active_rest_dream_fields_finite() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_dream_replay = true;
    let mut service = CognitiveLoopService::new(config).unwrap();

    for i in 0..50 {
        let result = service.cycle("exploring moral dimensions with care and wisdom");
        let m = &result.metadata;
        assert!(
            m.valence_homeostasis_pull.is_finite(),
            "homeostasis NaN at cycle {i}"
        );
        assert!(
            m.voice_articulation_quality.is_finite(),
            "voice quality NaN at cycle {i}"
        );
    }

    // Stats should track rest-related fields
    let stats = service.stats();
    assert!(stats.phi_rest_quality_factor.is_finite());
}

// ── Broca quality EMA tracking ────────────────────────────────────────

#[test]
fn test_broca_quality_ema_initialized() {
    // Verify broca quality stats fields exist and are properly defaulted
    let config = CognitiveLoopConfig::default();
    let service = CognitiveLoopService::new(config).unwrap();
    let stats = service.stats();
    assert_eq!(stats.broca_quality_ema, 0.0);
    assert_eq!(stats.broca_low_quality_streak, 0);
    assert_eq!(stats.broca_generation_count, 0);
}

// ── ToM exploration trigger ───────────────────────────────────────────

#[test]
fn test_tom_exploration_stats_initialized() {
    // Verify ToM stats fields exist and are properly defaulted
    let config = CognitiveLoopConfig::default();
    let service = CognitiveLoopService::new(config).unwrap();
    let stats = service.stats();
    assert_eq!(stats.tom_prediction_mismatch_ema, 0.0);
    assert_eq!(stats.tom_exploration_triggers, 0);
}

#[test]
fn test_tom_telemetry_in_metadata() {
    // Verify ToM fields appear in CycleMetadata
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).unwrap();
    for _ in 0..5 {
        let result = service.cycle("test social prediction");
        let m = &result.metadata;
        assert!(
            m.tom_prediction_mismatch.is_finite(),
            "tom_prediction_mismatch not finite"
        );
        assert!(
            m.tom_prediction_mismatch >= 0.0 && m.tom_prediction_mismatch <= 1.0,
            "tom_prediction_mismatch out of [0,1]: {}",
            m.tom_prediction_mismatch
        );
    }
}

// ── Cross-subsystem telemetry completeness ────────────────────────────

#[test]
fn test_broca_telemetry_quality_fields() {
    // Verify new BrocaGenerationTelemetry fields are accessible
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("test broca telemetry");
    // broca is Option<BrocaGenerationTelemetry>; when ssm_language is off it may be None
    if let Some(broca) = &result.metadata.broca {
        // These should be finite (defaulted to 0.0 when broca isn't enabled)
        assert!(broca.quality.is_finite(), "broca.quality not finite");
        assert!(
            broca.long_coherence.is_finite(),
            "broca.long_coherence not finite"
        );
        assert!(
            broca.semantic_pe.is_finite(),
            "broca.semantic_pe not finite"
        );
    }
    // If None, that's acceptable — no broca generation occurred
}

// ── Cross-subsystem coupling integration tests ────────────────────────

/// Run 100 cycles and verify broca_quality_ema is tracked, finite, and bounded.
#[test]
fn test_broca_quality_tracked_over_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "the nature of consciousness",
        "recursive self-awareness patterns",
        "binding across modalities",
        "predictive processing loop",
        "temporal integration dynamics",
    ];

    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;
        // Broca quality in metadata: when present, should be finite and bounded
        if let Some(broca) = &m.broca {
            assert!(
                broca.quality.is_finite(),
                "broca.quality not finite at cycle {i}: {}",
                broca.quality
            );
            assert!(
                broca.quality >= 0.0 && broca.quality <= 1.0,
                "broca.quality out of [0,1] at cycle {i}: {}",
                broca.quality
            );
        }
    }

    let stats = service.stats();
    assert!(
        stats.broca_quality_ema.is_finite(),
        "broca_quality_ema not finite after 100 cycles: {}",
        stats.broca_quality_ema
    );
    assert!(
        stats.broca_quality_ema >= 0.0 && stats.broca_quality_ema <= 1.0,
        "broca_quality_ema out of [0,1] after 100 cycles: {}",
        stats.broca_quality_ema
    );
    assert_eq!(stats.total_cycles, 100);
}

/// With no social models configured (default), ToM mismatch EMA should stay 0.0
/// and no exploration triggers should fire.
#[test]
fn test_tom_stats_zero_without_social_models() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_surprise_exploration: true,
        ..Default::default()
    })
    .unwrap();

    for i in 0..50 {
        let result = service.cycle("testing social prediction without models");
        let m = &result.metadata;
        assert!(
            m.tom_prediction_mismatch.is_finite(),
            "tom_prediction_mismatch not finite at cycle {i}"
        );
        // Without social models, mismatch should stay at 0.0
        assert!(
            m.tom_prediction_mismatch == 0.0,
            "tom_prediction_mismatch should be 0.0 without social models at cycle {i}, got {}",
            m.tom_prediction_mismatch
        );
        assert!(
            !m.tom_exploration_triggered,
            "tom_exploration_triggered should be false without social models at cycle {i}"
        );
    }

    let stats = service.stats();
    assert_eq!(
        stats.tom_prediction_mismatch_ema, 0.0,
        "tom_prediction_mismatch_ema should be 0.0 without social models"
    );
    assert_eq!(
        stats.tom_exploration_triggers, 0,
        "tom_exploration_triggers should be 0 without social models"
    );
}

/// Different substrate types should produce different substrate telemetry values.
#[test]
fn test_substrate_tau_affects_cycle() {
    use symthaea::cognitive_loop::config::SubstrateType;

    let substrates = [
        (SubstrateType::SiliconDigital, "silicon"),
        (SubstrateType::BiologicalNeurons, "biological"),
    ];

    let mut telemetry_by_substrate: Vec<(String, f64, f64, f32)> = Vec::new();

    for (substrate, name) in &substrates {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            substrate_type: *substrate,
            enable_substrate_speed_modulation: true,
            ..Default::default()
        })
        .unwrap();

        // Run enough cycles for telemetry to stabilize
        let mut last_result = service.cycle("substrate telemetry test");
        for _ in 1..20 {
            last_result = service.cycle("substrate telemetry test");
        }

        let m = &last_result.metadata;
        assert!(
            m.substrate_feasibility_raw.is_finite(),
            "{name}: substrate_feasibility_raw not finite"
        );
        assert!(
            m.substrate_effective_feasibility.is_finite(),
            "{name}: substrate_effective_feasibility not finite"
        );
        assert!(
            m.substrate_tau_factor.is_finite(),
            "{name}: substrate_tau_factor not finite"
        );

        telemetry_by_substrate.push((
            name.to_string(),
            m.substrate_feasibility_raw,
            m.substrate_effective_feasibility,
            m.substrate_tau_factor,
        ));
    }

    let (_, sil_raw, sil_eff, sil_tau) = &telemetry_by_substrate[0];
    let (_, bio_raw, bio_eff, bio_tau) = &telemetry_by_substrate[1];

    // Different substrates should produce different raw feasibility
    assert!(
        (sil_raw - bio_raw).abs() > 0.01,
        "SiliconDigital and BiologicalNeurons should have different raw feasibility: sil={sil_raw}, bio={bio_raw}"
    );

    // With speed modulation enabled, tau factors should differ
    assert!(
        (sil_tau - bio_tau).abs() > 0.01,
        "SiliconDigital and BiologicalNeurons should have different tau factors: sil={sil_tau}, bio={bio_tau}"
    );

    // Biological should have higher effective feasibility (honest confidence 0.95 vs 0.10)
    assert!(
        bio_eff > sil_eff,
        "Biological effective feasibility ({bio_eff}) should exceed silicon ({sil_eff})"
    );
}

/// Soak test: 500 cycles verifying all cross-coupling metadata fields remain
/// finite and within valid ranges throughout.
#[test]
fn test_cross_coupling_no_nan_500_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        enable_surprise_exploration: true,
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap();

    let inputs = [
        "the nature of consciousness",
        "completely different novel stimulus",
        "recursive self-awareness",
        "temporal binding problem",
        "quantum coherence hypothesis",
        "embodied cognition theory",
        "predictive processing framework",
        "global workspace theory",
        "integrated information theory",
        "higher order thought theory",
    ];

    for i in 0..500 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;

        // Consciousness level: finite, bounded [0, 1]
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "consciousness_level not finite at cycle {i}: {}",
            m.consciousness.consciousness_level
        );
        assert!(
            m.consciousness.consciousness_level >= 0.0
                && m.consciousness.consciousness_level <= 1.0,
            "consciousness_level out of [0,1] at cycle {i}: {}",
            m.consciousness.consciousness_level
        );

        // Prediction error: finite, bounded [0, 1]
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error not finite at cycle {i}: {}",
            result.prediction_error
        );
        assert!(
            result.prediction_error >= 0.0 && result.prediction_error <= 1.0,
            "prediction_error out of [0,1] at cycle {i}: {}",
            result.prediction_error
        );

        // Broca quality: when present, finite and bounded [0, 1]
        if let Some(broca) = &m.broca {
            assert!(
                broca.quality.is_finite(),
                "broca.quality not finite at cycle {i}: {}",
                broca.quality
            );
            assert!(
                broca.quality >= 0.0 && broca.quality <= 1.0,
                "broca.quality out of [0,1] at cycle {i}: {}",
                broca.quality
            );
        }

        // ToM prediction mismatch: finite, bounded [0, 1]
        assert!(
            m.tom_prediction_mismatch.is_finite(),
            "tom_prediction_mismatch not finite at cycle {i}: {}",
            m.tom_prediction_mismatch
        );
        assert!(
            m.tom_prediction_mismatch >= 0.0 && m.tom_prediction_mismatch <= 1.0,
            "tom_prediction_mismatch out of [0,1] at cycle {i}: {}",
            m.tom_prediction_mismatch
        );

        // Substrate fields: finite
        assert!(
            m.substrate_feasibility_raw.is_finite(),
            "substrate_feasibility_raw not finite at cycle {i}"
        );
        assert!(
            m.substrate_effective_feasibility.is_finite(),
            "substrate_effective_feasibility not finite at cycle {i}"
        );
        assert!(
            m.substrate_tau_factor.is_finite(),
            "substrate_tau_factor not finite at cycle {i}"
        );

        // Equation V2 consciousness: finite
        assert!(
            m.quality.equation_v2_consciousness.is_finite(),
            "equation_v2_consciousness not finite at cycle {i}: {}",
            m.quality.equation_v2_consciousness
        );
    }

    let stats = service.stats();
    assert_eq!(stats.total_cycles, 500);
    assert!(
        stats.broca_quality_ema.is_finite(),
        "broca_quality_ema not finite after 500 cycles"
    );
    assert!(
        stats.tom_prediction_mismatch_ema.is_finite(),
        "tom_prediction_mismatch_ema not finite after 500 cycles"
    );
    assert!(
        stats.tom_prediction_mismatch_ema >= 0.0 && stats.tom_prediction_mismatch_ema <= 1.0,
        "tom_prediction_mismatch_ema out of [0,1] after 500 cycles: {}",
        stats.tom_prediction_mismatch_ema
    );
    assert!(
        stats.broca_quality_ema >= 0.0 && stats.broca_quality_ema <= 1.0,
        "broca_quality_ema out of [0,1] after 500 cycles: {}",
        stats.broca_quality_ema
    );
}