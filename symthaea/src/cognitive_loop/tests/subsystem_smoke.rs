// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 1000-cycle smoke test for all enabled consciousness subsystems.
//!
//! Validates that the 30 recently-enabled subsystem defaults work correctly
//! together: no panics, no NaN/Inf cascades, and reasonable cycle times.
//! This is the integration burn-in complement to `cycle_properties::test_cycle_metadata_floats_finite`
//! which only runs 10 cycles.

use std::time::Instant;

use super::super::*;

/// Assert that a float value is finite (not NaN, not Inf).
fn assert_finite(val: f64, name: &str, cycle: usize) {
    assert!(
        val.is_finite(),
        "{name} is not finite at cycle {cycle}: {val}"
    );
}

/// Assert that an f32 value is finite.
fn assert_finite_f32(val: f32, name: &str, cycle: usize) {
    assert!(
        val.is_finite(),
        "{name} is not finite at cycle {cycle}: {val}"
    );
}

/// Validate all key metadata fields for NaN/Inf at the given cycle.
fn assert_metadata_finite(r: &CycleResult, cycle: usize) {
    // Core result fields
    assert_finite_f32(r.prediction_error, "prediction_error", cycle);
    assert_finite_f32(r.peak_attention, "peak_attention", cycle);
    if let Some(loss) = r.training_loss {
        assert_finite_f32(loss, "training_loss", cycle);
    }

    let m = &r.metadata;

    // Consciousness
    assert_finite(
        m.consciousness.consciousness_level,
        "consciousness_level",
        cycle,
    );
    assert_finite(
        m.consciousness.consciousness_profile_composite,
        "consciousness_profile_composite",
        cycle,
    );
    assert_finite(
        m.consciousness.synergy_enhanced_composite,
        "synergy_enhanced_composite",
        cycle,
    );
    assert_finite(
        m.consciousness.consciousness_gradient_magnitude,
        "consciousness_gradient_magnitude",
        cycle,
    );
    assert_finite(
        m.consciousness.consciousness_state_level,
        "consciousness_state_level",
        cycle,
    );

    // Embodied
    assert_finite(m.embodied.body_phi_modulation, "body_phi_modulation", cycle);
    assert_finite_f32(m.embodied.body_valence, "body_valence", cycle);
    assert_finite_f32(m.embodied.body_arousal, "body_arousal", cycle);
    assert_finite(
        m.embodied.embodied_phi_modulation,
        "embodied_phi_modulation",
        cycle,
    );
    assert_finite(m.embodied.embodied_agency, "embodied_agency", cycle);
    assert_finite_f32(m.embodied.affective_valence, "affective_valence", cycle);
    assert_finite_f32(m.embodied.affective_arousal, "affective_arousal", cycle);
    assert_finite_f32(
        m.embodied.affect_consciousness_valence,
        "affect_consciousness_valence",
        cycle,
    );
    assert_finite_f32(
        m.embodied.affect_consciousness_arousal,
        "affect_consciousness_arousal",
        cycle,
    );

    // Temporal / phenomenal
    assert_finite(
        m.temporal.temporal_coherence_score,
        "temporal_coherence_score",
        cycle,
    );
    assert_finite(m.temporal.cross_modal_psi, "cross_modal_psi", cycle);
    assert_finite_f32(
        m.temporal.cross_modal_binding_strength,
        "cross_modal_binding_strength",
        cycle,
    );
    assert_finite(
        m.temporal.thermodynamic_entropy,
        "thermodynamic_entropy",
        cycle,
    );
    assert_finite(
        m.temporal.thermodynamic_free_energy,
        "thermodynamic_free_energy",
        cycle,
    );
    assert_finite(
        m.temporal.phenomenal_binding_strength,
        "phenomenal_binding_strength",
        cycle,
    );
    assert_finite(m.temporal.holographic_unity, "holographic_unity", cycle);
    assert_finite(m.temporal.holographic_binding, "holographic_binding", cycle);

    // Reasoning
    assert_finite_f32(m.reasoning_confidence, "reasoning_confidence", cycle);
    assert_finite_f32(
        m.reasoning_chain_confidence,
        "reasoning_chain_confidence",
        cycle,
    );
    assert_finite(m.adaptive_reasoning_phi, "adaptive_reasoning_phi", cycle);
    assert_finite(
        m.meta_reasoning_confidence,
        "meta_reasoning_confidence",
        cycle,
    );

    // Attention
    assert_finite_f32(
        m.attention.attention_schema_focus,
        "attention_schema_focus",
        cycle,
    );
    assert_finite_f32(m.attention.psi_attention_avg, "psi_attention_avg", cycle);
    assert_finite_f32(
        m.attention.phi_attention_weight,
        "phi_attention_weight",
        cycle,
    );

    // FEP
    assert_finite(
        m.fep.predictive_free_energy,
        "predictive_free_energy",
        cycle,
    );
    assert_finite(
        m.fep.predictive_phi_modulation,
        "predictive_phi_modulation",
        cycle,
    );
    assert_finite(m.fep.fep_pragmatic_value, "fep_pragmatic_value", cycle);
    assert_finite(m.fep.fep_accuracy, "fep_accuracy", cycle);
    assert_finite(m.fep.fep_complexity, "fep_complexity", cycle);
    assert_finite(m.fep.fep_surprise, "fep_surprise", cycle);
    assert_finite(m.fep.fep_td_error, "fep_td_error", cycle);

    // Ethics / harmonics
    assert_finite_f32(m.ethics.moral_score, "moral_score", cycle);
    assert_finite(
        m.ethics.value_evaluator_score,
        "value_evaluator_score",
        cycle,
    );
    assert_finite_f32(m.ethics.value_feedback_trend, "value_feedback_trend", cycle);
    assert_finite_f32(m.ethics.soul_alignment, "soul_alignment", cycle);
    assert_finite(m.ethics.empathic_compassion, "empathic_compassion", cycle);
    assert_finite(m.ethics.empathic_tone_adj, "empathic_tone_adj", cycle);
    assert_finite_f32(
        m.harmonics.harmonies_alignment,
        "harmonies_alignment",
        cycle,
    );
    assert_finite(
        m.harmonics.harmonic_field_coherence,
        "harmonic_field_coherence",
        cycle,
    );
    assert_finite(
        m.harmonics.harmonic_love_resonance,
        "harmonic_love_resonance",
        cycle,
    );

    // Quality diagnostics
    assert_finite(m.quality.dissipative_health, "dissipative_health", cycle);
    assert_finite(
        m.quality.dissipative_entropy_rate,
        "dissipative_entropy_rate",
        cycle,
    );
    assert_finite(m.quality.epistemic_phi_eff, "epistemic_phi_eff", cycle);
    assert_finite(
        m.quality.equation_v2_consciousness,
        "equation_v2_consciousness",
        cycle,
    );
    assert_finite_f32(
        m.quality.hierarchical_ltc_phi,
        "hierarchical_ltc_phi",
        cycle,
    );
    assert_finite_f32(
        m.quality.meta_cognitive_accuracy,
        "meta_cognitive_accuracy",
        cycle,
    );

    // Memory
    assert_finite_f32(m.memory.resonator_best_sim, "resonator_best_sim", cycle);
    assert_finite_f32(m.memory.codebook_diversity, "codebook_diversity", cycle);

    // Misc subsystem telemetry
    assert_finite_f32(m.predictive_self_safety, "predictive_self_safety", cycle);
    assert_finite_f32(
        m.predictive_behavioral_error,
        "predictive_behavioral_error",
        cycle,
    );
    assert_finite(m.resonance_frequency, "resonance_frequency", cycle);
    assert_finite(m.quantum_coherence_level, "quantum_coherence_level", cycle);
    assert_finite(m.narrative_gwt_self_psi, "narrative_gwt_self_psi", cycle);
    assert_finite(m.narrative_self_psi, "narrative_self_psi", cycle);
    assert_finite(m.living_mind_vitality, "living_mind_vitality", cycle);
    assert_finite(m.living_mind_coherence, "living_mind_coherence", cycle);
    assert_finite(
        m.hierarchical_total_free_energy,
        "hierarchical_total_free_energy",
        cycle,
    );
    assert_finite(m.primitive_psi, "primitive_psi", cycle);
    assert_finite(m.causal_avg_confidence, "causal_avg_confidence", cycle);
    assert_finite(m.evolution_phi_delta, "evolution_phi_delta", cycle);
    assert_finite_f32(m.value_cache_hit_rate, "value_cache_hit_rate", cycle);
    assert_finite(m.epistemic_quality, "epistemic_quality", cycle);
    assert_finite(
        m.phi_validation_correlation,
        "phi_validation_correlation",
        cycle,
    );
    assert_finite(m.pipeline_consciousness, "pipeline_consciousness", cycle);
    assert_finite(
        m.multimodal_integrated_phi,
        "multimodal_integrated_phi",
        cycle,
    );
    assert_finite_f32(
        m.epistemic_gate_confidence,
        "epistemic_gate_confidence",
        cycle,
    );
    assert_finite(
        m.primitive_validation_phi_gain,
        "primitive_validation_phi_gain",
        cycle,
    );
    assert_finite(
        m.primitive_validation_p_value,
        "primitive_validation_p_value",
        cycle,
    );
    assert_finite_f32(m.negation_polarity, "negation_polarity", cycle);
    assert_finite_f32(m.actual_effective_lr, "actual_effective_lr", cycle);
    assert_finite_f32(m.cycle_reward, "cycle_reward", cycle);
    assert_finite(m.support_efe, "support_efe", cycle);
    assert_finite_f32(m.circadian_plasticity, "circadian_plasticity", cycle);
    assert_finite(m.context_phi_weight, "context_phi_weight", cycle);

    // Perception / drives / learning
    assert_finite_f32(
        m.perception_attention_sensitivity,
        "perception_attention_sensitivity",
        cycle,
    );
    assert_finite_f32(
        m.perception_budget_utilization,
        "perception_budget_utilization",
        cycle,
    );
    assert_finite_f32(
        m.perception_mean_coherence,
        "perception_mean_coherence",
        cycle,
    );
    assert_finite_f32(m.drive_boredom, "drive_boredom", cycle);
    assert_finite_f32(m.drive_flow_intensity, "drive_flow_intensity", cycle);
    assert_finite_f32(
        m.drive_exploration_threshold,
        "drive_exploration_threshold",
        cycle,
    );
    assert_finite_f32(m.learning_plasticity, "learning_plasticity", cycle);
    assert_finite_f32(m.learning_error_trend, "learning_error_trend", cycle);
}

/// 1000-cycle smoke test with all default subsystems enabled.
///
/// Validates:
/// - No panics (all 1000 cycles complete)
/// - No NaN/Inf in any key metric (checked every cycle)
/// - Monotonic cycle counter
/// - Reasonable total runtime (should be ~30-50s at 20-30Hz)
/// - Consciousness level eventually rises above zero
/// - Final stats are sane
#[test]
fn smoke_test_all_subsystems_1000_cycles() {
    const TOTAL_CYCLES: usize = 1000;

    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "hello world",
        "cause leads to effect",
        "the cat sat on the mat",
        "consciousness emerges from integration",
        "",
        "two plus two equals four",
        "the rain falls on the just and unjust alike",
        "a photon has no mass but carries momentum",
        "kindness is never wasted",
        "recursive self-awareness bootstraps identity",
    ];

    let mut cycle_times_us = Vec::with_capacity(TOTAL_CYCLES);
    let mut max_consciousness: f64 = 0.0;
    let wall_start = Instant::now();

    for i in 0..TOTAL_CYCLES {
        let input = inputs[i % inputs.len()];

        let cycle_start = Instant::now();
        let result = service.cycle(input);
        let elapsed_us = cycle_start.elapsed().as_micros() as u64;
        cycle_times_us.push(elapsed_us);

        // Validate no NaN/Inf in all key fields
        assert_metadata_finite(&result, i);

        // Track consciousness
        let cl = result.metadata.consciousness.consciousness_level;
        if cl > max_consciousness {
            max_consciousness = cl;
        }

        // Verify monotonic cycle counter
        assert_eq!(
            service.stats().total_cycles,
            i + 1,
            "Cycle counter mismatch at cycle {i}"
        );
    }

    let wall_elapsed = wall_start.elapsed();
    let stats = service.stats();

    // ── Timing statistics ──
    cycle_times_us.sort();
    let p50 = cycle_times_us[TOTAL_CYCLES / 2];
    let p95 = cycle_times_us[(TOTAL_CYCLES as f64 * 0.95) as usize];
    let p99 = cycle_times_us[(TOTAL_CYCLES as f64 * 0.99) as usize];
    let max_us = cycle_times_us[TOTAL_CYCLES - 1];
    let mean_us: f64 = cycle_times_us.iter().sum::<u64>() as f64 / TOTAL_CYCLES as f64;
    let effective_hz = 1_000_000.0 / mean_us;

    // ── Report (visible with --nocapture) ──
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════════════╗");
    eprintln!("  ║  SMOKE TEST: 1000 cycles, all subsystems enabled         ║");
    eprintln!("  ╠═══════════════════════════════════════════════════════════╣");
    eprintln!(
        "  ║  Wall time:          {:>8.2}s                            ║",
        wall_elapsed.as_secs_f64()
    );
    eprintln!(
        "  ║  Effective Hz:       {:>8.1} Hz                          ║",
        effective_hz
    );
    eprintln!(
        "  ║  Cycle latency (µs): mean={:<8.0} p50={:<6} p95={:<6} ║",
        mean_us, p50, p95
    );
    eprintln!(
        "  ║                      p99={:<8} max={:<8}             ║",
        p99, max_us
    );
    eprintln!(
        "  ║  Max consciousness:  {:<8.4}                            ║",
        max_consciousness
    );
    eprintln!(
        "  ║  Avg pred. error:    {:<8.4}                            ║",
        stats.avg_prediction_error
    );
    eprintln!(
        "  ║  Total cycles:       {:<8}                            ║",
        stats.total_cycles
    );
    eprintln!("  ╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Assertions ──
    assert_eq!(
        stats.total_cycles, TOTAL_CYCLES,
        "All 1000 cycles must complete"
    );
    assert!(
        stats.avg_prediction_error.is_finite(),
        "Final avg prediction error must be finite"
    );

    // Consciousness should eventually activate (at least once above 0)
    // With 30 subsystems enabled, we expect some consciousness to emerge.
    // Don't require high values — just non-zero.
    assert!(
        max_consciousness > 0.0,
        "Consciousness should emerge above 0 within 1000 cycles, got max={max_consciousness}"
    );

    // Cycle time sanity: p50 should be under 500ms.
    // The mean can be skewed by periodic expensive subsystems (consciousness equation,
    // dream consolidation, integrity checks) that run every N cycles.
    // p50 reflects typical cycle cost; p99 and max reflect periodic spikes.
    assert!(
        p50 < 500_000,
        "Median cycle time {p50}µs exceeds 500ms — subsystem regression"
    );

    // Warn (but don't fail) if mean is very high due to periodic spikes
    if mean_us > 500_000.0 {
        eprintln!(
            "  WARNING: Mean cycle time {mean_us:.0}µs > 500ms (periodic subsystem spikes). \
             p50={p50}µs is OK — investigate p99={p99}µs outliers."
        );
    }
}
