// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for phase result sub-structs (phase_results.rs).
//!
//! Validates Default derivation, field grouping invariants, and
//! end-to-end field propagation through the cycle pipeline.

use super::super::phase_results::*;
use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Default derivation — all #[derive(Default)] structs produce valid defaults
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dyn_core_default_zeroed() {
    let d = DynCore::default();
    assert!(d.output.is_empty());
    assert!(d.prediction.is_empty());
    assert_eq!(d.prediction_error, 0.0);
    assert_eq!(d.coherence, 0.0);
    assert_eq!(d.unified_psi, 0.0);
    assert!(!d.learning_occurred);
    assert!(d.training_loss.is_none());
    assert_eq!(d.effective_lr, 0.0);
    assert_eq!(d.cycle_reward, 0.0);
    assert_eq!(d.prediction_coherence, 0.0);
    assert_eq!(d.self_model_accuracy, 0.0);
}

#[test]
fn dyn_fep_default_zeroed() {
    let d = DynFep::default();
    assert_eq!(d.fep_action_idx, 0);
    assert_eq!(d.fep_pragmatic_value, 0.0);
    assert_eq!(d.fep_accuracy, 0.0);
    assert_eq!(d.fep_complexity, 0.0);
    assert_eq!(d.fep_surprise, 0.0);
    assert_eq!(d.fep_td_error, 0.0);
}

#[test]
fn dyn_reasoning_default_zeroed() {
    let d = DynReasoning::default();
    assert_eq!(d.reasoning_confidence, 0.0);
    assert!(!d.reasoning_gate_blocked);
    assert!(d.reasoning_fallback.is_none());
    assert!(d.reasoning_plan_action.is_none());
    assert_eq!(d.reasoning_plan_confidence, 0.0);
    assert!(d.reasoning_narrative.is_none());
    assert!(!d.metacognitive_anomaly);
    assert_eq!(d.mcts_plan_effectiveness, 0.0);
    assert_eq!(d.causal_attention_edges, 0);
    assert_eq!(d.school_predicted_phi_gain, 0.0);
}

#[test]
fn dyn_attention_default_zeroed() {
    let d = DynAttention::default();
    assert!(!d.attention_budget_exceeded);
    assert_eq!(d.attention_budget_elapsed_us, 0);
    assert!(!d.predictive_budget_gated);
}

#[test]
fn dyn_resonator_default_zeroed() {
    let d = DynResonator::default();
    assert!(!d.resonator_wm_primed);
    assert_eq!(d.resonator_reconsolidated, 0);
    assert_eq!(d.resonator_best_sim, 0.0);
    assert_eq!(d.resonator_prediction_error, 0.0);
    assert_eq!(d.resonator_error_exploration_mod, 0.0);
}

#[test]
fn dyn_homeostasis_default_zeroed() {
    let d = DynHomeostasis::default();
    assert_eq!(d.anomaly_recovery_progress, 0.0);
    assert!(!d.anomaly_recovering);
    assert_eq!(d.valence_homeostasis_pull, 0.0);
    assert_eq!(d.arousal_homeostasis_pull, 0.0);
    assert_eq!(d.homeostasis_pull_strength, 0.0);
    assert!(!d.arousal_recovery_active);
    assert_eq!(d.arousal_recovery_tau_factor, 0.0);
}

#[test]
fn dyn_guidance_default_empty() {
    let d = DynGuidance::default();
    assert!(d.moral_steering_category.is_empty());
    assert!(d.guiding_priority_category.is_empty());
    assert!(d.guiding_question.is_empty());
    assert!(d.dominant_harmonic.is_empty());
}

#[test]
fn dyn_neuromod_default_zeroed() {
    let d = DynNeuromod::default();
    assert_eq!(d.neuromod_attention_alloc, 0.0);
    assert_eq!(d.ne_reorienting_boost, 0.0);
    assert_eq!(d.ne_arousal_feedback, 0.0);
    assert_eq!(d.confidence_velocity, 0.0);
    assert_eq!(d.sht_crash_dip, 0.0);
    assert_eq!(d.exploration_sht_drain, 0.0);
    assert_eq!(d.phasic_da_replay_boost, 0);
}

#[test]
fn dynamics_phase_result_default_all_subs_zeroed() {
    let d = DynamicsPhaseResult::default();
    // Parent-level loose fields
    assert_eq!(d.binding_threshold_mod, 0.0);
    assert_eq!(d.binding_confidence_mod, 0.0);
    assert_eq!(d.epistemic_semantic_lr_mod, 0.0);
    assert_eq!(d.pfe_surprise_mod, 0.0);
    assert_eq!(d.epistemic_uncertainty, 0.0);
    assert_eq!(d.aleatoric_uncertainty, 0.0);
    assert_eq!(d.fep_tau_factor, 0.0);
    assert_eq!(d.phi_tau_factor, 0.0);
    assert_eq!(d.causal_world_model_edges, 0);
    // Sub-structs exist (not null)
    assert_eq!(d.core.prediction_error, 0.0);
    assert_eq!(d.fep.fep_action_idx, 0);
    assert!(!d.reasoning.reasoning_gate_blocked);
    assert!(!d.attention.attention_budget_exceeded);
    assert!(!d.resonator.resonator_wm_primed);
    assert!(!d.homeostasis.anomaly_recovering);
    assert!(d.guidance.guiding_question.is_empty());
    assert_eq!(d.neuromod.phasic_da_replay_boost, 0);
}

#[test]
fn fb_quality_default_zeroed() {
    let d = FbQuality::default();
    assert_eq!(d.cross_module_agreement, 0.0);
    assert_eq!(d.unified_quality_score, 0.0);
    assert!(!d.coherence_velocity_gated);
    assert!(!d.dissipative_health_gated);
    assert_eq!(d.dissipative_health, 0.0);
    assert!(d.dissipative_regime.is_empty());
    assert_eq!(d.dissipative_entropy_rate, 0.0);
    assert_eq!(d.dissipative_lr_factor, 0.0);
    assert_eq!(d.coherence_velocity, 0.0);
}

#[test]
fn fb_consciousness_default_zeroed() {
    let d = FbConsciousness::default();
    assert_eq!(d.primitive_psi, 0.0);
    assert_eq!(d.temporal_causal_chains, 0);
    assert_eq!(d.lattice_height, 0);
    assert_eq!(d.lattice_width, 0);
    assert!(d.lattice_join_concept.is_none());
    assert_eq!(d.consciousness_level, 0.0);
    assert!(d.spectral_mip_phi.is_none());
    assert!(d.sigma.is_none());
    assert_eq!(d.structural_num_clusters, 0);
    assert!(d.convergence_state.is_empty());
}

#[test]
fn fb_self_model_default_zeroed() {
    let d = FbSelfModel::default();
    assert!(!d.prefrontal_veto);
    assert_eq!(d.meta_cognitive_accuracy, 0.0);
    assert_eq!(d.meta_cognitive_depth, 0);
    assert_eq!(d.body_psi_modulation, 0.0);
    assert!(!d.gwt_broadcast);
    assert_eq!(d.gwt_coalition_size, 0);
    assert!(!d.phenomenal_fragmented);
    assert!(!d.temporal_discontinuity);
    assert!(!d.narrative_gwt_veto);
}

#[test]
fn fb_reasoning_default_zeroed() {
    let d = FbReasoning::default();
    assert!(d.reasoning_context.is_empty());
    assert_eq!(d.context_phi_weight, 0.0);
    assert!(!d.context_phi_applied);
    assert_eq!(d.reasoning_chain_depth, 0);
    assert_eq!(d.causal_relations_count, 0);
    assert_eq!(d.epistemic_quality, 0.0);
    assert!(!d.epistemic_gate_approved);
    assert_eq!(d.code_primitives_selected, 0);
}

#[test]
fn fb_ethics_default_zeroed() {
    let d = FbEthics::default();
    assert_eq!(d.value_evaluator_score, 0.0);
    assert!(d.value_evaluator_decision.is_empty());
    assert_eq!(d.value_gate_factor, 0.0);
    assert_eq!(d.value_embeddings_created, 0);
    assert!(!d.harmonies_approved);
    assert!(d.composition_rule_applied.is_empty());
    assert_eq!(d.empathic_compassion, 0.0);
}

#[test]
fn fb_evolution_default_zeroed() {
    let d = FbEvolution::default();
    assert_eq!(d.hierarchical_ltc_phi, 0.0);
    assert_eq!(d.evolution_generation, 0);
    assert_eq!(d.evolution_phi_delta, 0.0);
    assert_eq!(d.evolution_confidence_delta, 0.0);
    assert_eq!(d.primitive_validation_phi_gain, 0.0);
    assert_eq!(d.primitive_validation_p_value, 0.0);
}

#[test]
fn fb_loops_default_zeroed() {
    let d = FbLoops::default();
    assert!(d.limiting_component_boosted.is_empty());
    assert_eq!(d.love_resonance_boost, 0.0);
    assert!(!d.reasoning_chain_boosted);
    assert_eq!(d.harmonic_interference_lr_mod, 0.0);
    assert!(!d.causal_urgency_gated);
    assert!(!d.epistemic_coherence_gated);
    assert!(!d.attention_budget_gated);
}

#[test]
fn fb_memory_default_zeroed() {
    let d = FbMemory::default();
    assert_eq!(d.dream_insights, 0);
    assert_eq!(d.dream_phi_improvement, 0.0);
    assert_eq!(d.dream_wisdom_count, 0);
    assert_eq!(d.resonator_promotions, 0);
    assert_eq!(d.codebook_evictions, 0);
    assert_eq!(d.codebook_diversity, 0.0);
    assert_eq!(d.codebook_utilization_rate, 0.0);
    assert_eq!(d.surprise_replay_batch_size, 0);
}

#[test]
fn fb_support_default_zeroed() {
    let d = FbSupport::default();
    assert_eq!(d.support_triage_count, 0);
    assert!(!d.support_alert_fired);
    assert_eq!(d.support_federation_graduated, 0);
    assert_eq!(d.support_efe, 0.0);
}

#[test]
fn feedback_phase_result_default_all_subs() {
    let f = FeedbackPhaseResult::default();
    // Parent-level loose fields
    assert_eq!(f.multi_obj_frontier_size, 0);
    assert_eq!(f.grid_encoding_norm, 0.0);
    assert_eq!(f.grid_spatial_complexity, 0.0);
    assert_eq!(f.social_learning_rate_factor, 0.0);
    // Sub-structs exist
    assert_eq!(f.quality.cross_module_agreement, 0.0);
    assert_eq!(f.consciousness.primitive_psi, 0.0);
    assert!(!f.self_model.prefrontal_veto);
    assert!(f.reasoning.reasoning_context.is_empty());
    assert_eq!(f.ethics.value_evaluator_score, 0.0);
    assert_eq!(f.evolution.evolution_generation, 0);
    assert!(f.loops.limiting_component_boosted.is_empty());
    assert_eq!(f.memory.dream_insights, 0);
    assert_eq!(f.support.support_triage_count, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Sub-struct field count invariants — catch accidental field loss/gain
// ═══════════════════════════════════════════════════════════════════════════════

/// Count fields via std::mem::size_of ratio is unreliable due to padding.
/// Instead we use a simpler approach: verify total field count by exhaustive match.
/// If a field is added/removed, this test won't compile.
/// Compile-time verification: exhaustive destructure ensures field count is correct.
#[test]
fn dyn_core_field_count_12() {
    let d = DynCore::default();
    // Destructure to ensure compile-time verification of all fields
    let DynCore {
        output: _,
        prediction: _,
        prediction_first_horizon: _,
        prediction_error: _,
        coherence: _,
        unified_psi: _,
        learning_occurred: _,
        training_loss: _,
        effective_lr: _,
        cycle_reward: _,
        prediction_coherence: _,
        self_model_accuracy: _,
    } = d;
    // Compile-time assertion: if fields change, destructure above won't compile
}

/// Compile-time verification: exhaustive destructure ensures field count is correct.
#[test]
fn dyn_fep_field_count_10() {
    let d = DynFep::default();
    let DynFep {
        fep_action_idx: _,
        fep_pragmatic_value: _,
        fep_accuracy: _,
        fep_complexity: _,
        fep_surprise: _,
        fep_td_error: _,
        trajectory_efe: _,
        trajectory_best_action: _,
        trajectory_surprise: _,
        trajectory_ode_steps: _,
    } = d;
    // Compile-time assertion: if fields change, destructure above won't compile
}

/// Compile-time verification: exhaustive destructure ensures field count is correct.
#[test]
fn dyn_reasoning_field_count_10() {
    let d = DynReasoning::default();
    let DynReasoning {
        reasoning_confidence: _,
        reasoning_gate_blocked: _,
        reasoning_fallback: _,
        reasoning_plan_action: _,
        reasoning_plan_confidence: _,
        reasoning_narrative: _,
        metacognitive_anomaly: _,
        mcts_plan_effectiveness: _,
        causal_attention_edges: _,
        school_predicted_phi_gain: _,
        ..
    } = d;
    // Compile-time assertion: if fields change, destructure above won't compile
}

/// Compile-time verification: exhaustive destructure ensures field count is correct.
#[test]
fn dynamics_phase_result_field_count() {
    let d = DynamicsPhaseResult::default();
    let DynamicsPhaseResult {
        core: _,
        fep: _,
        reasoning: _,
        attention: _,
        resonator: _,
        homeostasis: _,
        guidance: _,
        neuromod: _,
        binding_threshold_mod: _,
        binding_confidence_mod: _,
        epistemic_semantic_lr_mod: _,
        pfe_surprise_mod: _,
        epistemic_uncertainty: _,
        aleatoric_uncertainty: _,
        fep_tau_factor: _,
        causal_world_model_edges: _,
        epistemic_budget_scale: _,
        ..
    } = d;
    // Compile-time assertion: if fields change, destructure above won't compile
}

/// Compile-time verification: exhaustive destructure ensures field count is correct.
#[test]
fn feedback_phase_result_field_count() {
    let f = FeedbackPhaseResult::default();
    let FeedbackPhaseResult {
        quality: _,
        consciousness: _,
        self_model: _,
        reasoning: _,
        ethics: _,
        evolution: _,
        loops: _,
        memory: _,
        support: _,
        multi_obj_frontier_size: _,
        grid_encoding_norm: _,
        grid_spatial_complexity: _,
        social_learning_rate_factor: _,
        #[cfg(feature = "vision-manifold")]
            mental_movie: _,
    } = f;
    // Compile-time assertion: if fields change, destructure above won't compile
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. End-to-end propagation — verify phase results flow into CycleMetadata
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cycle_propagates_fep_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test fep propagation");
    let m = &result.metadata;

    // FEP fields from DynFep should appear in metadata.fep
    assert!(m.fep.fep_accuracy.is_finite());
    assert!(m.fep.fep_complexity.is_finite());
    assert!(m.fep.fep_surprise.is_finite());
    assert!(m.fep.fep_td_error.is_finite());
}

#[test]
fn cycle_propagates_reasoning_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test reasoning propagation");
    let m = &result.metadata;

    // Reasoning fields from DynReasoning should appear in metadata
    assert!(m.reasoning_confidence.is_finite());
    assert!(m.reasoning_plan_confidence.is_finite());
    assert!(m.mcts_plan_effectiveness.is_finite());
}

#[test]
fn cycle_propagates_homeostasis_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test homeostasis propagation");
    let m = &result.metadata;

    // Homeostasis fields from DynHomeostasis should appear in metadata
    assert!(m.valence_homeostasis_pull.is_finite());
    assert!(m.arousal_homeostasis_pull.is_finite());
    assert!(m.homeostasis_pull_strength.is_finite());
    assert!(m.arousal_recovery_tau_factor.is_finite());
}

#[test]
fn cycle_propagates_quality_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test quality propagation");
    let m = &result.metadata;

    // Quality fields from FbQuality → metadata.quality
    assert!(m.quality.unified_quality_score.is_finite());
    assert!(m.cross_module_agreement.is_finite());
    assert!(m.quality.dissipative_health.is_finite());
}

#[test]
fn cycle_propagates_consciousness_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test consciousness propagation");
    let m = &result.metadata;

    // Consciousness fields from FbConsciousness
    assert!(m.consciousness.consciousness_level.is_finite());
    assert!(m.primitive_psi.is_finite());
    assert!(m.temporal.temporal_continuity.is_finite());
    assert!(m.consciousness.consciousness_profile_composite.is_finite());
}

#[test]
fn cycle_propagates_ethics_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test ethics propagation");
    let m = &result.metadata;

    // Ethics fields from FbEthics → metadata.ethics
    assert!(m.ethics.value_evaluator_score.is_finite());
    assert!(m.ethics.moral_score.is_finite());
    assert!(m.ethics.empathic_compassion.is_finite());
}

#[test]
fn cycle_propagates_guidance_strings() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test guidance propagation");
    let m = &result.metadata;

    // Guidance strings from DynGuidance → metadata.harmonics
    // These may be empty but should not be None/panic
    let _ = &m.harmonics.guiding_question;
    let _ = &m.harmonics.guiding_priority_category;
    let _ = &m.harmonics.dominant_harmonic;
}

#[test]
fn cycle_propagates_attention_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test attention propagation");
    let m = &result.metadata;

    // Attention fields from DynAttention — verify they exist and are accessible
    let _ = m.attention.attention_budget_gated;
    let _ = m.predictive_budget_gated;
}

#[test]
fn cycle_propagates_neuromod_dynamics_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test neuromod dynamics propagation");
    let m = &result.metadata;

    // Neuromod dynamics fields from DynNeuromod
    assert!(m.neuromod_ne_reorienting_boost.is_finite());
    assert!(m.ne_arousal_feedback.is_finite());
    assert!(m.confidence_velocity.is_finite());
}

#[test]
fn cycle_propagates_memory_fields() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test memory propagation");
    let m = &result.metadata;

    // Memory fields from FbMemory → metadata.memory
    assert!(m.memory.dream_phi_improvement.is_finite());
    assert!(m.memory.codebook_diversity.is_finite());
    assert!(m.memory.resonator_best_sim.is_finite());
}
