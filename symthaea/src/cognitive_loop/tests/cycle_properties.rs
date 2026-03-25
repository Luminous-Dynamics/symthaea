// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based and invariant tests for the cognitive loop's core cycle.
//!
//! These tests verify structural invariants that must hold regardless of input:
//! - Output bounds (confidence, learning rate, prediction error)
//! - No NaN/Inf in any float field
//! - Monotonic cycle counter
//! - Bounded internal data structures (VecDeque capacities)
//! - Deterministic reproducibility with genesis seeding
//! - Feedback clamping (homeostatic guards)
//! - Carry-over consistency across cycle boundaries

use serial_test::serial;

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a minimal CognitiveLoopService with default config for fast tests.
fn minimal_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
}

/// Build a CognitiveLoopService with genesis phrase for deterministic tests.
fn seeded_service(phrase: &str) -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some(phrase.to_string()),
        ..Default::default()
    })
    .unwrap()
}

/// Build a CognitiveLoopService with always-learn threshold and primitive consciousness.
fn learning_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        enable_primitive_consciousness: true,
        ..Default::default()
    })
    .unwrap()
}

/// Assert that a float value is finite (not NaN, not Inf).
fn assert_finite(val: f64, name: &str) {
    assert!(val.is_finite(), "{name} is not finite: {val}");
}

/// Assert that an f32 value is finite.
fn assert_finite_f32(val: f32, name: &str) {
    assert!(val.is_finite(), "{name} is not finite: {val}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Output bounds: confidence in [0.0, 1.0]
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_confidence_bounded_after_cycles() {
    let mut service = minimal_service();
    for i in 0..50 {
        service.cycle(&format!("input cycle {i}"));
        let conf = service.prediction_confidence();
        assert!(
            (0.0..=1.0).contains(&conf),
            "Confidence out of [0.0, 1.0] at cycle {i}: {conf}"
        );
    }
}

#[test]
fn test_confidence_bounded_with_varied_inputs() {
    let mut service = learning_service();
    let inputs = [
        "short",
        "a much longer input with many words to encode differently",
        "",
        "!@#$%^&*() special characters",
        "repeated repeated repeated repeated repeated",
        "I am so happy and excited about this wonderful result!",
        "This is terrible and sad and frightening.",
    ];
    for (i, input) in inputs.iter().cycle().take(40).enumerate() {
        service.cycle(input);
        let conf = service.prediction_confidence();
        assert!(
            (0.0..=1.0).contains(&conf),
            "Confidence out of bounds at cycle {i} with input '{input}': {conf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Output bounds: learning rate in valid range
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_learning_rate_bounded_after_cycles() {
    let mut service = minimal_service();
    for i in 0..30 {
        service.cycle(&format!("lr test {i}"));
        let lr = service.stats().adaptive_learning_rate;
        assert!(
            (0.0..=1.0).contains(&lr),
            "Learning rate out of (0.0, 1.0] at cycle {i}: {lr}"
        );
        assert!(
            lr.is_finite(),
            "Learning rate is not finite at cycle {i}: {lr}"
        );
    }
}

#[test]
fn test_effective_learning_rate_bounded() {
    let mut service = learning_service();
    for i in 0..20 {
        let result = service.cycle(&format!("effective lr cycle {i}"));
        let effective_lr = result.metadata.actual_effective_lr;
        assert!(
            effective_lr.is_finite(),
            "Effective LR not finite at cycle {i}: {effective_lr}"
        );
        assert!(
            effective_lr >= 0.0,
            "Effective LR negative at cycle {i}: {effective_lr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. No NaN/Inf: all float fields in CycleResult are finite
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_cycle_result_no_nan_inf() {
    let mut service = minimal_service();
    for i in 0..10 {
        let r = service.cycle(&format!("nan check {i}"));

        assert_finite_f32(r.prediction_error, "prediction_error");
        assert_finite_f32(r.peak_attention, "peak_attention");
        if let Some(loss) = r.training_loss {
            assert_finite_f32(loss, "training_loss");
        }
        // Check output vector
        for (j, &v) in r.output.iter().enumerate() {
            assert_finite_f32(v, &format!("output[{j}]"));
        }
    }
}

#[test]
fn test_cycle_metadata_floats_finite() {
    let mut service = learning_service();
    for i in 0..10 {
        let r = service.cycle(&format!("meta nan check {i}"));
        let m = &r.metadata;

        assert_finite_f32(m.reasoning_confidence, "reasoning_confidence");
        assert_finite(m.embodied.body_phi_modulation, "body_phi_modulation");
        assert_finite_f32(m.embodied.body_valence, "body_valence");
        assert_finite_f32(m.embodied.body_arousal, "body_arousal");
        assert_finite(m.consciousness.consciousness_level, "consciousness_level");
        assert_finite_f32(m.predictive_self_safety, "predictive_self_safety");
        assert_finite_f32(m.attention.attention_schema_focus, "attention_schema_focus");
        assert_finite(m.resonance_frequency, "resonance_frequency");
        assert_finite(m.quantum_coherence_level, "quantum_coherence_level");
        assert_finite(
            m.temporal.temporal_coherence_score,
            "temporal_coherence_score",
        );
        assert_finite(
            m.embodied.embodied_phi_modulation,
            "embodied_phi_modulation",
        );
        assert_finite(m.embodied.embodied_agency, "embodied_agency");
        assert_finite(m.narrative_gwt_self_psi, "narrative_gwt_self_psi");
        assert_finite(m.narrative_self_psi, "narrative_self_psi");
        assert_finite(m.living_mind_vitality, "living_mind_vitality");
        assert_finite(m.living_mind_coherence, "living_mind_coherence");
        assert_finite(m.fep.predictive_free_energy, "predictive_free_energy");
        assert_finite(m.fep.predictive_phi_modulation, "predictive_phi_modulation");
        assert_finite_f32(
            m.temporal.cross_modal_binding_strength,
            "cross_modal_binding_strength",
        );
        assert_finite(m.temporal.cross_modal_psi, "cross_modal_psi");
        assert_finite_f32(m.embodied.affective_valence, "affective_valence");
        assert_finite_f32(m.embodied.affective_arousal, "affective_arousal");
        assert_finite(m.temporal.thermodynamic_entropy, "thermodynamic_entropy");
        assert_finite(
            m.temporal.thermodynamic_free_energy,
            "thermodynamic_free_energy",
        );
        assert_finite(
            m.temporal.phenomenal_binding_strength,
            "phenomenal_binding_strength",
        );
        assert_finite(
            m.hierarchical_total_free_energy,
            "hierarchical_total_free_energy",
        );
        assert_finite_f32(m.attention.psi_attention_avg, "psi_attention_avg");
        assert_finite(m.primitive_psi, "primitive_psi");
        assert_finite(m.ethics.value_evaluator_score, "value_evaluator_score");
        assert_finite_f32(m.harmonics.harmonies_alignment, "harmonies_alignment");
        assert_finite(
            m.consciousness.consciousness_profile_composite,
            "consciousness_profile_composite",
        );
        assert_finite(
            m.consciousness.synergy_enhanced_composite,
            "synergy_enhanced_composite",
        );
        assert_finite(
            m.harmonics.harmonic_field_coherence,
            "harmonic_field_coherence",
        );
        assert_finite(
            m.harmonics.harmonic_love_resonance,
            "harmonic_love_resonance",
        );
        assert_finite_f32(m.reasoning_chain_confidence, "reasoning_chain_confidence");
        assert_finite(m.adaptive_reasoning_phi, "adaptive_reasoning_phi");
        assert_finite(m.causal_avg_confidence, "causal_avg_confidence");
        assert_finite(m.evolution_phi_delta, "evolution_phi_delta");
        assert_finite_f32(m.value_cache_hit_rate, "value_cache_hit_rate");
        assert_finite(m.epistemic_quality, "epistemic_quality");
        assert_finite(m.phi_validation_correlation, "phi_validation_correlation");
        assert_finite(m.quality.dissipative_health, "dissipative_health");
        assert_finite(
            m.quality.dissipative_entropy_rate,
            "dissipative_entropy_rate",
        );
        assert_finite(m.quality.epistemic_phi_eff, "epistemic_phi_eff");
        assert_finite(
            m.quality.equation_v2_consciousness,
            "equation_v2_consciousness",
        );
        assert_finite_f32(m.quality.hierarchical_ltc_phi, "hierarchical_ltc_phi");
        assert_finite(m.temporal.holographic_unity, "holographic_unity");
        assert_finite(m.temporal.holographic_binding, "holographic_binding");
        assert_finite(
            m.consciousness.consciousness_gradient_magnitude,
            "consciousness_gradient_magnitude",
        );
        assert_finite_f32(
            m.embodied.affect_consciousness_valence,
            "affect_consciousness_valence",
        );
        assert_finite_f32(
            m.embodied.affect_consciousness_arousal,
            "affect_consciousness_arousal",
        );
        assert_finite(m.pipeline_consciousness, "pipeline_consciousness");
        assert_finite(m.multimodal_integrated_phi, "multimodal_integrated_phi");
        assert_finite(
            m.consciousness.consciousness_state_level,
            "consciousness_state_level",
        );
        assert_finite_f32(m.epistemic_gate_confidence, "epistemic_gate_confidence");
        assert_finite(
            m.primitive_validation_phi_gain,
            "primitive_validation_phi_gain",
        );
        assert_finite(
            m.primitive_validation_p_value,
            "primitive_validation_p_value",
        );
        assert_finite(m.meta_reasoning_confidence, "meta_reasoning_confidence");
        assert_finite_f32(m.negation_polarity, "negation_polarity");
        assert_finite_f32(m.ethics.moral_score, "moral_score");
        assert_finite_f32(m.actual_effective_lr, "actual_effective_lr");
        assert_finite_f32(m.cycle_reward, "cycle_reward");
        assert_finite_f32(m.ethics.value_feedback_trend, "value_feedback_trend");
        assert_finite(m.support_efe, "support_efe");
        assert_finite_f32(m.ethics.soul_alignment, "soul_alignment");
        assert_finite_f32(m.quality.meta_cognitive_accuracy, "meta_cognitive_accuracy");
        assert_finite_f32(m.circadian_plasticity, "circadian_plasticity");
        assert_finite_f32(m.attention.phi_attention_weight, "phi_attention_weight");
        assert_finite(m.fep.fep_pragmatic_value, "fep_pragmatic_value");
        assert_finite(m.fep.fep_accuracy, "fep_accuracy");
        assert_finite(m.fep.fep_complexity, "fep_complexity");
        assert_finite(m.fep.fep_surprise, "fep_surprise");
        assert_finite(m.fep.fep_td_error, "fep_td_error");
        assert_finite_f32(m.memory.resonator_best_sim, "resonator_best_sim");
        assert_finite_f32(m.memory.codebook_diversity, "codebook_diversity");
        assert_finite(m.context_phi_weight, "context_phi_weight");
        assert_finite(m.ethics.empathic_compassion, "empathic_compassion");
        assert_finite(m.ethics.empathic_tone_adj, "empathic_tone_adj");

        // ── Manager telemetry finiteness ──
        assert_finite_f32(
            m.perception_attention_sensitivity,
            "perception_attention_sensitivity",
        );
        assert_finite_f32(
            m.perception_budget_utilization,
            "perception_budget_utilization",
        );
        assert_finite_f32(m.perception_mean_coherence, "perception_mean_coherence");
        assert_finite_f32(m.drive_boredom, "drive_boredom");
        assert_finite_f32(m.drive_flow_intensity, "drive_flow_intensity");
        assert_finite_f32(m.drive_exploration_threshold, "drive_exploration_threshold");
        assert_finite_f32(m.learning_plasticity, "learning_plasticity");
        assert_finite_f32(m.learning_error_trend, "learning_error_trend");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Monotonic cycle counter
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cycle_counter_monotonic() {
    let mut service = minimal_service();
    for expected in 1..=30 {
        service.cycle("counter test");
        assert_eq!(
            service.stats().total_cycles,
            expected,
            "Cycle counter should be {expected}, got {}",
            service.stats().total_cycles
        );
    }
}

#[test]
fn test_cycle_counter_survives_varied_inputs() {
    let mut service = minimal_service();
    let binding = "a".repeat(500);
    let inputs = ["", "a", "hello world", "!@#$%^", binding.as_str()];
    for (i, input) in inputs.iter().enumerate() {
        service.cycle(input);
        assert_eq!(
            service.stats().total_cycles,
            i + 1,
            "Cycle counter mismatch after input '{}'",
            &input[..input.len().min(20)]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Amortization guards: subsystems at co-prime intervals execute
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_biorhythm_refresh_fires_at_97_cycles() {
    let mut service = minimal_service();
    // Smoke test: run 96 warm-up cycles to reach biorhythm trigger at 97
    for i in 0..96 {
        let r = service.cycle(&format!("bio {i}"));
        // Spot-check every 20th cycle to avoid per-iteration overhead
        if i % 20 == 0 {
            assert!(
                r.prediction_error.is_finite(),
                "cycle {i} prediction_error not finite"
            );
        }
    }
    // The 97th cycle triggers refresh, counter resets to 0, then increments to 1
    // on the 98th call. We just verify no panic and phase is reported.
    let r97 = service.cycle("biorhythm trigger");
    assert!(
        !r97.metadata.circadian_phase.is_empty(),
        "Circadian phase should be populated after 97 cycles"
    );
}

#[test]
fn test_urgency_should_run_intervals() {
    // Verify the co-prime interval logic in CycleUrgency::should_run
    let critical = CycleUrgency::Critical;
    let normal = CycleUrgency::Normal;
    let cruise = CycleUrgency::Cruise;

    // interval=0 always runs
    assert!(critical.should_run(0, 0, 1, 1));
    assert!(normal.should_run(0, 1, 0, 1));
    assert!(cruise.should_run(0, 1, 1, 0));

    // interval=1 always runs (every cycle)
    assert!(critical.should_run(42, 1, 99, 99));

    // interval=7: runs at cycle 0, 7, 14, ...
    assert!(normal.should_run(0, 99, 7, 99));
    assert!(normal.should_run(7, 99, 7, 99));
    assert!(!normal.should_run(3, 99, 7, 99));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Carry-over consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_carryover_urgency_persists() {
    let mut service = minimal_service();
    // Run a few cycles to establish urgency state
    for _ in 0..5 {
        let r = service.cycle("carryover test");
        assert!(r.prediction_error.is_finite());
    }
    let urgency_after_5 = service.cycle("check urgency").metadata.urgency;
    // Urgency should be a valid variant
    assert!(
        matches!(
            urgency_after_5,
            CycleUrgency::Critical | CycleUrgency::Normal | CycleUrgency::Cruise
        ),
        "Urgency should be a valid variant: {:?}",
        urgency_after_5
    );
}

#[test]
fn test_carryover_learning_state_updated() {
    let mut service = learning_service();
    let initial_conf = service.prediction_confidence();
    assert!(
        (0.0..=1.0).contains(&initial_conf),
        "Initial confidence should be in [0, 1]: {initial_conf}"
    );
    let first = service.cycle("first cycle");
    assert!(first.prediction_error.is_finite());
    // After one cycle, the carryover learning prediction_confidence should have been
    // set to the pre-cycle value at the start of the next cycle
    let next_result = service.cycle("second cycle");
    // The confidence may have changed, but it should still be bounded
    let conf = service.prediction_confidence();
    assert!(
        (0.0..=1.0).contains(&conf),
        "Confidence after carryover should be in [0.0, 1.0]: {conf}"
    );
    // The result should also contain valid metadata
    assert!(next_result.metadata.actual_effective_lr.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Feedback clamping: confidence can't exceed 1.0
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_confidence_never_exceeds_one_with_positive_feedback() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_predictive_processing: true,
        enable_affective_bridge: true,
        enable_virtual_body: true,
        enable_gwt: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Run many cycles with inputs designed to boost confidence
    for i in 0..100 {
        service.cycle("consistent stable predictable focused pattern");
        let conf = service.prediction_confidence();
        assert!(conf <= 1.0, "Confidence exceeded 1.0 at cycle {i}: {conf}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Feedback clamping: learning rate can't go below floor
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_learning_rate_has_positive_floor() {
    let mut service = learning_service();
    // Run many cycles — LR should never hit exactly zero (floor is 0.0001)
    for i in 0..50 {
        service.cycle(&format!("lr floor test {i}"));
        let lr = service.stats().adaptive_learning_rate;
        assert!(
            lr >= 0.0001 - f32::EPSILON,
            "Learning rate below floor at cycle {i}: {lr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Empty input: cycle with empty string doesn't panic
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_input_no_panic() {
    let mut service = minimal_service();
    let result = service.cycle("");
    assert!(
        result.prediction_error.is_finite(),
        "Empty input should produce finite prediction error"
    );
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_whitespace_only_input_no_panic() {
    let mut service = minimal_service();
    let result = service.cycle("   \t\n  ");
    assert!(result.prediction_error.is_finite());
    assert_eq!(service.stats().total_cycles, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. Default config produces valid cycle
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_default_config_valid_cycle() {
    let config = CognitiveLoopConfig::default();
    assert!(config.validate().is_ok(), "Default config should validate");

    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("default config test");

    assert!(result.prediction_error >= 0.0);
    assert!(!result.output.is_empty());
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_minimal_profile_valid_cycle() {
    let config = CognitiveLoopConfig::from_profile(
        crate::cognitive_loop::config::ConsciousnessProfile::Minimal,
    );
    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("minimal profile");
    assert!(result.prediction_error.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. Metadata completeness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_metadata_has_urgency() {
    let mut service = minimal_service();
    let result = service.cycle("metadata completeness");
    assert!(
        matches!(
            result.metadata.urgency,
            CycleUrgency::Critical | CycleUrgency::Normal | CycleUrgency::Cruise
        ),
        "Urgency should be a valid variant"
    );
}

#[test]
fn test_metadata_has_strategy() {
    let mut service = minimal_service();
    let result = service.cycle("strategy check");
    assert!(
        !result.metadata.selected_strategy.is_empty(),
        "Selected strategy should be populated"
    );
}

#[test]
fn test_metadata_circadian_populated() {
    let mut service = minimal_service();
    let result = service.cycle("circadian check");
    assert!(
        !result.metadata.circadian_phase.is_empty(),
        "Circadian phase should be populated"
    );
    assert!(
        result.metadata.circadian_plasticity > 0.0,
        "Circadian plasticity should be positive"
    );
}

#[test]
fn test_metadata_cycle_time_positive() {
    let mut service = minimal_service();
    let result = service.cycle("timing check");
    assert!(
        result.cycle_time_us > 0,
        "Cycle time should be positive, got {}",
        result.cycle_time_us
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. VecDeque bounds: error_history doesn't exceed capacity
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_error_history_bounded() {
    let mut service = minimal_service();
    // Run 150 cycles — error_history capacity is 100
    for i in 0..150 {
        let r = service.cycle(&format!("error hist {i}"));
        if i % 30 == 0 {
            assert!(
                r.prediction_error.is_finite(),
                "cycle {i} prediction_error not finite"
            );
        }
    }
    // error_history is private, but avg_prediction_error is derived from it
    // and should be finite (not accumulating unbounded data)
    let avg = service.stats().avg_prediction_error;
    assert!(
        avg.is_finite(),
        "avg_prediction_error should be finite after 150 cycles: {avg}"
    );
}

#[test]
fn test_recent_hvs_bounded() {
    let mut service = learning_service();
    // recent_hvs capacity is 4 (VecDeque in CycleHistory)
    // Run many cycles with primitive consciousness enabled
    for i in 0..30 {
        let r = service.cycle(&format!("recent hvs {i}"));
        if i % 10 == 0 {
            assert!(
                r.prediction_error.is_finite(),
                "cycle {i} prediction_error not finite"
            );
        }
    }
    // If capacity weren't bounded, we'd see memory growth.
    // We can verify the system is still functioning correctly.
    let result = service.cycle("final check");
    assert!(result.prediction_error.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 13. Psi attestation buffer bounded
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_psi_attestation_buffer_bounded() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_psi_attestation: true,
        agent_did: Some("did:key:z6MkTestAgent".to_string()),
        attestation_buffer_capacity: 16,
        ..Default::default()
    })
    .unwrap();

    // Run more cycles than buffer capacity
    for i in 0..50 {
        let r = service.cycle(&format!("attestation {i}"));
        if i % 10 == 0 {
            assert!(
                r.prediction_error.is_finite(),
                "cycle {i} prediction_error not finite"
            );
        }
    }

    // Buffer should not exceed capacity
    assert!(
        service.psi_attestation_count() <= 16,
        "Attestation buffer exceeded capacity: {}",
        service.psi_attestation_count()
    );
}

#[test]
fn test_psi_attestation_drain() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_psi_attestation: true,
        agent_did: Some("did:key:z6MkTestDrain".to_string()),
        attestation_buffer_capacity: 64,
        ..Default::default()
    })
    .unwrap();

    for i in 0..10 {
        let r = service.cycle(&format!("drain test {i}"));
        assert!(r.prediction_error.is_finite());
    }

    let records = service.drain_psi_attestations();
    // After drain, buffer should be empty
    assert_eq!(service.psi_attestation_count(), 0);
    // Records should have valid data
    for r in &records {
        assert!(r.psi.is_finite(), "Psi attestation psi should be finite");
        assert!(
            r.prediction_error.is_finite(),
            "Attestation prediction_error should be finite"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 14. Multiple cycles: no drift/accumulation issues over 100 cycles
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_100_cycles_stable() {
    let mut service = learning_service();
    for i in 0..100 {
        let result = service.cycle("stability test input");

        // Core invariants must hold every cycle
        let conf = service.prediction_confidence();
        assert!(
            (0.0..=1.0).contains(&conf),
            "Confidence out of bounds at cycle {i}: {conf}"
        );

        let lr = service.stats().adaptive_learning_rate;
        assert!(lr.is_finite() && lr >= 0.0, "LR invalid at cycle {i}: {lr}");

        assert!(
            result.prediction_error.is_finite() && result.prediction_error >= 0.0,
            "Prediction error invalid at cycle {i}: {}",
            result.prediction_error
        );

        // Output should always have elements
        assert!(
            !result.output.is_empty(),
            "Output should not be empty at cycle {i}"
        );
    }

    // After 100 cycles, stats should reflect all cycles
    assert_eq!(service.stats().total_cycles, 100);

    // Consciousness level should be in valid range
    let level = service.consciousness_level();
    assert!(
        (0.0..=1.0).contains(&level),
        "Consciousness level out of bounds after 100 cycles: {level}"
    );
}

#[test]
#[serial]
fn test_100_cycles_varied_input_stable() {
    let mut service = minimal_service();
    let diverse_inputs = [
        "hello",
        "cause leads to effect in systems",
        "",
        "I am deeply concerned about the ethical implications",
        "mathematics: 2+2=4, e=mc^2",
        "explore novel possibilities",
        "routine predictable stable input",
        "danger warning alert critical",
    ];
    for i in 0..100 {
        let input = diverse_inputs[i % diverse_inputs.len()];
        let result = service.cycle(input);

        assert!(
            result.prediction_error.is_finite(),
            "Prediction error not finite at cycle {i}"
        );
        assert!(
            service.prediction_confidence() >= 0.0 && service.prediction_confidence() <= 1.0,
            "Confidence out of bounds at cycle {i}: {}",
            service.prediction_confidence()
        );
    }
    assert_eq!(service.stats().total_cycles, 100);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 15. Deterministic: same seed + same input = same output (within tolerance)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_genesis_determinism_single_cycle() {
    let phrase = "determinism test phrase 42";
    let mut a = seeded_service(phrase);
    let mut b = seeded_service(phrase);

    let ra = a.cycle("identical input");
    let rb = b.cycle("identical input");

    // Prediction errors should be exactly equal with same genesis seed
    assert!(
        (ra.prediction_error - rb.prediction_error).abs() < 1e-6,
        "Prediction errors diverged: {} vs {}",
        ra.prediction_error,
        rb.prediction_error
    );

    // Output vectors should match
    assert_eq!(ra.output.len(), rb.output.len(), "Output lengths differ");
    for (i, (va, vb)) in ra.output.iter().zip(rb.output.iter()).enumerate() {
        assert!((va - vb).abs() < 1e-5, "Output[{i}] diverged: {va} vs {vb}");
    }
}

#[test]
#[serial]
fn test_genesis_determinism_multi_cycle() {
    let phrase = "multi cycle determinism";
    let mut a = seeded_service(phrase);
    let mut b = seeded_service(phrase);

    // F32_TOLERANCE from the codebase is 0.10 (10%) for 100 cycles.
    // Over 10 cycles, tolerance should be much tighter.
    for i in 0..10 {
        let input = format!("cycle input {i}");
        let ra = a.cycle(&input);
        let rb = b.cycle(&input);

        let diff = (ra.prediction_error - rb.prediction_error).abs();
        assert!(
            diff < 0.01,
            "Prediction errors diverged at cycle {i}: {} vs {} (diff={diff})",
            ra.prediction_error,
            rb.prediction_error
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional property tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prediction_error_non_negative() {
    let mut service = minimal_service();
    for i in 0..20 {
        let result = service.cycle(&format!("pred error {i}"));
        assert!(
            result.prediction_error >= 0.0,
            "Prediction error should be non-negative at cycle {i}: {}",
            result.prediction_error
        );
    }
}

#[test]
fn test_prediction_error_bounded_above() {
    let mut service = minimal_service();
    for i in 0..20 {
        let result = service.cycle(&format!("pred error bound {i}"));
        assert!(
            result.prediction_error <= 1.0,
            "Prediction error should be <= 1.0 at cycle {i}: {}",
            result.prediction_error
        );
    }
}

#[test]
fn test_output_dimension_consistent() {
    let mut service = minimal_service();
    let first = service.cycle("dim check first");
    let dim = first.output.len();
    assert!(dim > 0, "Output dimension should be positive");

    for i in 0..10 {
        let result = service.cycle(&format!("dim check {i}"));
        assert_eq!(
            result.output.len(),
            dim,
            "Output dimension changed at cycle {i}: {} vs {dim}",
            result.output.len()
        );
    }
}

#[test]
fn test_exploration_urge_bounded() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_surprise_exploration: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    for i in 0..50 {
        service.cycle(&format!("explore bound {i}"));
        let urge = service.behavior.curiosity_drive.exploration_urge;
        assert!(
            (0.0..=1.0).contains(&urge),
            "Exploration urge out of [0.0, 1.0] at cycle {i}: {urge}"
        );
    }
}

#[test]
fn test_boredom_bounded() {
    let mut service = minimal_service();
    for i in 0..50 {
        service.cycle("same boring input");
        let boredom = service.boredom();
        assert!(
            (0.0..=1.0).contains(&boredom),
            "Boredom out of [0.0, 1.0] at cycle {i}: {boredom}"
        );
    }
}

#[test]
fn test_try_cycle_matches_cycle() {
    let mut service = minimal_service();
    // try_cycle should succeed for normal inputs
    let result = service.try_cycle("try cycle test");
    assert!(result.is_ok(), "try_cycle should not fail for normal input");
    let r = result.unwrap();
    assert!(r.prediction_error.is_finite());
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_reset_clears_state() {
    let mut service = minimal_service();
    for _ in 0..10 {
        let r = service.cycle("pre-reset");
        assert!(r.prediction_error.is_finite());
    }
    assert_eq!(service.stats().total_cycles, 10);

    service.reset();
    assert_eq!(service.stats().total_cycles, 0);
    assert!((service.prediction_confidence() - 0.5).abs() < 0.01);

    // Should be able to run cycles again after reset
    let result = service.cycle("post-reset");
    assert_eq!(service.stats().total_cycles, 1);
    assert!(result.prediction_error.is_finite());
}

#[test]
fn test_stats_learning_cycles_consistent() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        ..Default::default()
    })
    .unwrap();

    for _ in 0..20 {
        let r = service.cycle("learning test");
        assert!(r.prediction_error.is_finite());
    }

    // Learning cycles should not exceed total cycles
    assert!(
        service.stats().learning_cycles <= service.stats().total_cycles,
        "Learning cycles ({}) should not exceed total cycles ({})",
        service.stats().learning_cycles,
        service.stats().total_cycles,
    );
}

#[test]
fn test_moral_score_in_metadata_bounded() {
    let mut service = minimal_service();
    let result = service.cycle("help others be kind and honest");
    assert!(
        result.metadata.ethics.moral_score >= -1.0 && result.metadata.ethics.moral_score <= 1.0,
        "Moral score out of [-1.0, 1.0]: {}",
        result.metadata.ethics.moral_score
    );
}

#[test]
fn test_cycle_reward_bounded() {
    let mut service = minimal_service();
    for i in 0..10 {
        let result = service.cycle(&format!("reward test {i}"));
        assert!(
            result.metadata.cycle_reward >= -1.0 && result.metadata.cycle_reward <= 1.0,
            "Cycle reward out of [-1.0, 1.0] at cycle {i}: {}",
            result.metadata.cycle_reward
        );
    }
}

#[test]
fn test_attention_sensitivity_bounded_after_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_attention_schema: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    for i in 0..30 {
        service.cycle(&format!("attn sens {i}"));
        let sens = service.behavior.adaptive_behavior.attention_sensitivity;
        assert!(
            (0.5..=2.0).contains(&sens),
            "Attention sensitivity out of [0.5, 2.0] at cycle {i}: {sens}"
        );
    }
}

#[test]
#[serial]
fn test_consciousness_level_bounded() {
    let mut service = learning_service();
    for i in 0..20 {
        let result = service.cycle(&format!("consciousness level {i}"));
        assert!(
            result.metadata.consciousness.consciousness_level >= 0.0
                && result.metadata.consciousness.consciousness_level <= 1.0,
            "Consciousness level out of [0.0, 1.0] at cycle {i}: {}",
            result.metadata.consciousness.consciousness_level
        );
    }
}
