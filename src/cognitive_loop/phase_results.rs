// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase result structs — carry local variables between cognitive cycle phases.
//!
//! All fields are `pub(super)` so that the phase modules can construct/read them,
//! but they are invisible outside the cognitive_loop module.
//!
//! Organized by phase:
//! - `Perc*` + `PerceptionPhaseResult` — perception phase output
//! - `Dyn*` + `DynamicsPhaseResult` — dynamics phase output
//! - `Fb*` + `FeedbackPhaseResult` — feedback phase output

#[cfg(feature = "vision-manifold")]
use super::types::MentalMovie;
use super::types::MoralJudgmentSummary;
use super::{CycleUrgency, ResponseStrategy};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::predictive_encoder::EncodingResult;

// ═══════════════════════════════════════════════════════════════════════════════
// PERCEPTION PHASE
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC encoding and attention results.
pub(super) struct PercEncoding {
    pub(super) encoding_result: EncodingResult,
    pub(super) hv16_cached: BinaryHV,
    pub(super) compressed_state: Vec<f32>,
    pub(super) phi_attention_weight: f32,
    pub(super) input_memoized: bool,
    pub(super) input_similarity: f32,
    pub(super) memo_threshold: f32,
    pub(super) effective_threshold: f32,
    pub(super) temporal_binding_strength: f32,
}

/// Moral evaluation results.
pub(super) struct PercMoral {
    pub(super) moral_concern_detected: bool,
    pub(super) moral_score: f32,
    pub(super) moral_judgment: MoralJudgmentSummary,
    pub(super) soul_alignment: f32,
    /// 18D Spinozist affect fingerprint.
    pub(super) moral_affect_coords: [f32; 18],
    /// FluctuatioAnimi max tension.
    pub(super) moral_fluctuatio_tension: f32,
    /// Moral ambiguity flag.
    pub(super) moral_is_ambiguous: bool,
    /// Epistemic confidence from affect adequacy.
    pub(super) moral_epistemic_confidence: f32,
}

/// Response strategy selection.
pub(super) struct PercStrategy {
    pub(super) selected_strategy: ResponseStrategy,
    pub(super) agency_strategy_override: bool,
    pub(super) social_strategy_bias: bool,
}

/// Exploration and surprise signals.
pub(super) struct PercExploration {
    pub(super) exploration_urge_start: f32,
    pub(super) surprise_triggered: bool,
    pub(super) exploration_action: Option<String>,
}

/// Math intent detection results.
#[derive(Default)]
pub(super) struct PercMath {
    /// Whether math intent was detected in the input
    pub(super) math_detected: bool,
    /// Classified problem type (if detected)
    #[cfg(feature = "mathematics")]
    pub(super) problem_type: Option<crate::cognitive_loop::math_service::MathProblemType>,
    /// Phi from the math computation (0.0 if no math)
    pub(super) phi: f64,
    /// Confidence in the math result (0.0 if no math)
    pub(super) confidence: f64,
}

/// Urgency classification and prediction.
pub(super) struct PercUrgency {
    pub(super) urgency: CycleUrgency,
    pub(super) error_pattern: &'static str,
    pub(super) predicted_urgency: &'static str,
    pub(super) prediction_coherence_urgency_bias: f32,
    pub(super) prediction_error: f32,
    pub(super) error_slope: f32,
    pub(super) oscillation_ratio: f32,
}

/// Result of the perception phase (Phases 0–1.2).
pub(super) struct PerceptionPhaseResult {
    pub(super) encoding: PercEncoding,
    pub(super) moral: PercMoral,
    pub(super) strategy: PercStrategy,
    pub(super) exploration: PercExploration,
    pub(super) urgency: PercUrgency,
    pub(super) math: PercMath,
    pub(super) startup_suppressed: bool,
    pub(super) startup_warmup_progress: f32,
    pub(super) negation_detected: f32,
    // Vision manifold fields (only present when feature is enabled)
    #[cfg(feature = "vision-manifold")]
    pub(super) vision_mean_surprise: f32,
    #[cfg(feature = "vision-manifold")]
    pub(super) cross_manifold_prediction_error: f32,
    #[cfg(feature = "vision-manifold")]
    pub(super) vision_horizon_errors: Vec<f32>,
    #[cfg(feature = "vision-manifold")]
    pub(super) scene_recognized: bool,
    #[cfg(feature = "vision-manifold")]
    pub(super) vision_telemetry: Option<symthaea_vision_manifold::VisionTelemetry>,
    // Foveation fields (only present when feature is enabled)
    #[cfg(feature = "foveation")]
    pub(super) foveation_recognition_count: usize,
    #[cfg(feature = "foveation")]
    pub(super) foveation_top_confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMICS PHASE
// ═══════════════════════════════════════════════════════════════════════════════

/// Core CfC dynamics: output, prediction, learning.
#[derive(Default)]
pub(super) struct DynCore {
    pub(super) output: Vec<f32>,
    pub(super) prediction: Vec<f32>,
    /// Shortest-horizon raw prediction (bits-saved diagnostics only; None on
    /// fallback/degenerate prediction paths).
    pub(super) prediction_first_horizon: Option<Vec<f32>>,
    /// Predictive Compression C3: whether episodic recall blended into the
    /// prediction this cycle.
    pub(super) recall_fired: bool,
    /// C3: top-1 recall similarity this cycle, when a recall was attempted.
    pub(super) recall_similarity: Option<f32>,
    /// C3c: write-cycle number of the matched episode, when a recall was attempted.
    pub(super) recall_matched_timestamp: Option<u64>,
    pub(super) prediction_error: f32,
    pub(super) coherence: f32,
    pub(super) unified_psi: f64,
    pub(super) learning_occurred: bool,
    pub(super) training_loss: Option<f32>,
    pub(super) effective_lr: f32,
    pub(super) cycle_reward: f32,
    pub(super) prediction_coherence: f32,
    pub(super) self_model_accuracy: f32,
}

/// FEP active inference metrics.
#[derive(Default)]
pub(super) struct DynFep {
    pub(super) fep_action_idx: usize,
    pub(super) fep_pragmatic_value: f64,
    pub(super) fep_accuracy: f64,
    pub(super) fep_complexity: f64,
    pub(super) fep_surprise: f64,
    pub(super) fep_td_error: f64,
    pub(super) trajectory_efe: f64,
    pub(super) trajectory_best_action: usize,
    pub(super) trajectory_surprise: f64,
    pub(super) trajectory_ode_steps: usize,
}

/// Reasoning engine and planning metrics.
#[derive(Default)]
pub(super) struct DynReasoning {
    pub(super) reasoning_confidence: f32,
    pub(super) reasoning_gate_blocked: bool,
    pub(super) reasoning_fallback: Option<String>,
    pub(super) reasoning_plan_action: Option<usize>,
    pub(super) reasoning_plan_confidence: f32,
    pub(super) reasoning_narrative: Option<String>,
    pub(super) metacognitive_anomaly: bool,
    pub(super) mcts_plan_effectiveness: f32,
    pub(super) causal_attention_edges: usize,
    pub(super) school_predicted_phi_gain: f32,
    // ── Internal reasoning engine diagnostics ───────────────────────────
    pub(super) re_phi_eff_raw: f32,
    pub(super) re_phi_eff: f32,
    pub(super) re_epistemic_mod: f32,
    pub(super) re_gamma: f32,
    pub(super) re_reliability: f32,
    pub(super) re_budget_consumed: f32,
    pub(super) re_wall_time_us: u64,
    pub(super) re_steps_taken: u32,
    pub(super) re_tier_reached: u32,
    pub(super) re_gate_checks: u32,
    pub(super) re_budget_exceeded: bool,
    pub(super) re_evs: f32,
    pub(super) re_mcts_iterations: u32,
    pub(super) re_did_simulate: bool,
}

/// Attention budget metrics.
#[derive(Default)]
pub(super) struct DynAttention {
    pub(super) attention_budget_exceeded: bool,
    pub(super) attention_budget_elapsed_us: u64,
    pub(super) predictive_budget_gated: bool,
}

/// Resonator memory metrics.
#[derive(Default)]
pub(super) struct DynResonator {
    pub(super) resonator_wm_primed: bool,
    pub(super) resonator_reconsolidated: usize,
    pub(super) resonator_best_sim: f32,
    pub(super) resonator_prediction_error: f32,
    pub(super) resonator_error_exploration_mod: f32,
}

/// Homeostasis and anomaly recovery.
#[derive(Default)]
pub(super) struct DynHomeostasis {
    pub(super) anomaly_recovery_progress: f32,
    pub(super) anomaly_recovering: bool,
    pub(super) valence_homeostasis_pull: f32,
    pub(super) arousal_homeostasis_pull: f32,
    pub(super) homeostasis_pull_strength: f32,
    pub(super) arousal_recovery_active: bool,
    pub(super) arousal_recovery_tau_factor: f32,
}

/// Moral steering and harmonic guidance.
#[derive(Default)]
pub(super) struct DynGuidance {
    pub(super) moral_steering_category: String,
    pub(super) guiding_priority_category: String,
    pub(super) guiding_question: String,
    pub(super) dominant_harmonic: String,
}

/// Neuromodulator feedback signals.
#[derive(Default)]
pub(super) struct DynNeuromod {
    pub(super) neuromod_attention_alloc: f32,
    pub(super) ne_reorienting_boost: f32,
    pub(super) ne_arousal_feedback: f32,
    pub(super) confidence_velocity: f32,
    pub(super) sht_crash_dip: f32,
    pub(super) exploration_sht_drain: f32,
    pub(super) phasic_da_replay_boost: usize,
}

/// Math solver dispatch results from dynamics phase.
#[derive(Default)]
pub(super) struct DynMath {
    /// Whether a math solver was dispatched this cycle
    pub(super) solved: bool,
    /// Phi from the solver (0.0 if not solved)
    pub(super) phi: f64,
    /// Confidence from the solver (0.0 if not solved)
    pub(super) confidence: f64,
    /// Whether multi-path verification succeeded
    pub(super) multipath_verified: bool,
    /// Human-readable answer (empty if not solved)
    pub(super) answer: String,
    /// Epistemic caveat from the solver (None if N/A)
    pub(super) epistemic_caveat: Option<String>,
    /// Error bound on numerical result (None if N/A)
    pub(super) error_bound: Option<f64>,
}

/// Result of the dynamics phase (Phases A–12).
#[derive(Default)]
pub(super) struct DynamicsPhaseResult {
    pub(super) core: DynCore,
    pub(super) fep: DynFep,
    pub(super) reasoning: DynReasoning,
    pub(super) attention: DynAttention,
    pub(super) resonator: DynResonator,
    pub(super) homeostasis: DynHomeostasis,
    pub(super) guidance: DynGuidance,
    pub(super) neuromod: DynNeuromod,
    pub(super) math: DynMath,
    pub(super) binding_threshold_mod: f32,
    pub(super) binding_confidence_mod: f32,
    pub(super) epistemic_semantic_lr_mod: f32,
    pub(super) pfe_surprise_mod: f32,
    // Adaptive dynamics telemetry
    pub(super) epistemic_uncertainty: f32,
    pub(super) aleatoric_uncertainty: f32,
    pub(super) fep_tau_factor: f32,
    pub(super) phi_tau_factor: f32,
    pub(super) prediction_horizon_tau: f32,
    pub(super) causal_world_model_edges: usize,
    pub(super) epistemic_budget_scale: f32,
    // Session 11: crash detector telemetry
    pub(super) confidence_crash_detected: bool,
    pub(super) lr_frozen: bool,
    pub(super) semantic_evictions: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK PHASE
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality/dissipative metrics from feedback phase.
#[derive(Default)]
pub(super) struct FbQuality {
    pub(super) cross_module_agreement: f32,
    pub(super) unified_quality_score: f32,
    pub(super) coherence_velocity_gated: bool,
    pub(super) dissipative_health_gated: bool,
    pub(super) dissipative_health: f64,
    pub(super) dissipative_regime: String,
    pub(super) dissipative_entropy_rate: f64,
    pub(super) dissipative_lr_factor: f32,
    pub(super) coherence_velocity: f32,
    pub(super) agreement_confidence_coupling: bool,
}

/// Consciousness engine and pipeline metrics.
#[derive(Default)]
pub(super) struct FbConsciousness {
    pub(super) primitive_psi: f64,
    pub(super) temporal_causal_chains: usize,
    pub(super) temporal_continuity: f64,
    pub(super) temporal_max_chain_length: usize,
    pub(super) causal_codebook_entries_len: usize,
    pub(super) continuity_replay_needed: bool,
    pub(super) lattice_height: usize,
    pub(super) lattice_width: usize,
    pub(super) lattice_join_concept: Option<String>,
    pub(super) compositionality_total: usize,
    pub(super) consciousness_profile_composite: f64,
    pub(super) synergy_enhanced_composite: f64,
    pub(super) emergent_properties_count: usize,
    pub(super) equation_v2_consciousness: f64,
    pub(super) eq_v2_limiting_component: String,
    pub(super) pipeline_consciousness: f64,
    pub(super) multimodal_integrated_phi: f64,
    pub(super) consciousness_state_label: String,
    pub(super) consciousness_state_level: f64,
    pub(super) consciousness_gradient_magnitude: f64,
    pub(super) consciousness_limiting_component: String,
    pub(super) holographic_unity: f64,
    pub(super) holographic_binding: f64,
    pub(super) affect_cons_valence: f32,
    pub(super) affect_cons_arousal: f32,
    pub(super) consciousness_level: f64,
    pub(super) spectral_mip_phi: Option<f64>,
    pub(super) sigma: Option<f64>,
    pub(super) phi_spectral_weight: f32,
    pub(super) structural_micro_phi: f64,
    pub(super) structural_meso_phi: f64,
    pub(super) structural_macro_phi: f64,
    pub(super) structural_bottleneck: f64,
    pub(super) structural_emergence_ratio: f64,
    pub(super) structural_num_clusters: usize,
    pub(super) consciousness_weights: [f64; 4],
    pub(super) consciousness_weight_variance: f64,
    pub(super) convergence_state: String,
}

/// Self-model tier: embodiment, attention, phenomenal binding.
#[derive(Default)]
pub(super) struct FbSelfModel {
    pub(super) prefrontal_veto: bool,
    pub(super) meta_cognitive_accuracy: f32,
    pub(super) meta_cognitive_depth: u8,
    pub(super) body_psi_modulation: f64,
    pub(super) body_valence: f32,
    pub(super) body_arousal: f32,
    pub(super) affective_valence: f32,
    pub(super) affective_arousal: f32,
    pub(super) narrative_self_psi: f64,
    pub(super) predictive_free_energy: f64,
    pub(super) predictive_psi_modulation: f64,
    pub(super) hierarchical_total_free_energy: f64,
    pub(super) predictive_self_safety: f32,
    pub(super) predictive_behavioral_error: f32,
    pub(super) attention_schema_focus: f32,
    pub(super) attention_fatigue: f32,
    pub(super) attention_prediction_accuracy: f32,
    pub(super) psi_attention_avg: f32,
    pub(super) gwt_broadcast: bool,
    pub(super) gwt_coalition_size: u32,
    pub(super) cross_modal_binding_strength: f32,
    pub(super) cross_modal_psi: f64,
    pub(super) resonance_frequency: f64,
    pub(super) quantum_coherence_level: f64,
    pub(super) phenomenal_binding_strength: f64,
    pub(super) phenomenal_fragmented: bool,
    pub(super) temporal_coherence_score: f64,
    pub(super) temporal_discontinuity: bool,
    pub(super) thermodynamic_entropy: f64,
    pub(super) thermodynamic_free_energy: f64,
    pub(super) embodied_psi_modulation: f64,
    pub(super) embodied_agency: f64,
    pub(super) narrative_gwt_veto: bool,
    pub(super) narrative_gwt_self_psi: f64,
    pub(super) living_mind_vitality: f64,
    pub(super) living_mind_coherence: f64,
    pub(super) hierarchical_free_energy_lr_boost: f32,
    pub(super) predictive_phi_lr_delta: f32,
    pub(super) body_valence_confidence_delta: f32,
    pub(super) narrative_self_confidence_factor: f32,
}

/// Reasoning, epistemic, and causal metrics.
#[derive(Default)]
pub(super) struct FbReasoning {
    pub(super) reasoning_context: String,
    pub(super) context_phi_weight: f64,
    pub(super) context_phi_applied: bool,
    pub(super) reasoning_chain_confidence: f32,
    pub(super) reasoning_chain_depth: usize,
    pub(super) causal_relations_count: usize,
    pub(super) causal_avg_confidence: f64,
    pub(super) adaptive_reasoning_phi: f64,
    pub(super) epistemic_quality: f64,
    pub(super) phi_validation_correlation: f64,
    pub(super) epistemic_phi_eff: f64,
    pub(super) epistemic_conflict_count: usize,
    pub(super) epistemic_gate_confidence: f32,
    pub(super) epistemic_gate_approved: bool,
    pub(super) meta_reasoning_confidence: f64,
    pub(super) meta_reasoning_insights: usize,
    pub(super) code_primitives_selected: usize,
}

/// Ethics, values, harmonics, and empathy metrics.
#[derive(Default)]
pub(super) struct FbEthics {
    pub(super) value_evaluator_score: f64,
    pub(super) value_evaluator_decision: String,
    pub(super) value_gate_factor: f32,
    pub(super) value_embeddings_created: u64,
    pub(super) value_cache_hit_rate: f32,
    pub(super) harmonies_alignment: f32,
    pub(super) harmonies_approved: bool,
    pub(super) composition_rule_applied: String,
    pub(super) harmonic_field_coherence: f64,
    pub(super) harmonic_love_resonance: f64,
    pub(super) harmonic_interferences: usize,
    pub(super) empathic_compassion: f64,
    pub(super) empathic_tone_adj: f64,
    pub(super) empathic_speech_rate_mod: f32,
    pub(super) kosmic_coherence: f32,
}

/// Evolution and validation metrics.
#[derive(Default)]
pub(super) struct FbEvolution {
    pub(super) hierarchical_ltc_phi: f32,
    pub(super) evolution_generation: usize,
    pub(super) evolution_phi_delta: f64,
    pub(super) evolution_confidence_delta: f32,
    pub(super) primitive_validation_phi_gain: f64,
    pub(super) primitive_validation_p_value: f64,
}

/// Feedback loop gating signals.
#[derive(Default)]
pub(super) struct FbLoops {
    pub(super) limiting_component_boosted: String,
    pub(super) love_resonance_boost: f32,
    pub(super) reasoning_chain_boosted: bool,
    pub(super) harmonic_interference_lr_mod: f32,
    pub(super) causal_urgency_gated: bool,
    pub(super) epistemic_coherence_gated: bool,
    pub(super) attention_budget_gated: bool,
}

/// Dream and memory resonator metrics.
#[derive(Default)]
pub(super) struct FbMemory {
    pub(super) dream_insights: usize,
    pub(super) dream_phi_improvement: f32,
    pub(super) dream_wisdom_count: usize,
    pub(super) resonator_promotions: usize,
    pub(super) codebook_evictions: usize,
    pub(super) codebook_diversity: f32,
    pub(super) codebook_utilization_rate: f32,
    pub(super) surprise_replay_batch_size: usize,
    pub(super) memory_db_flushed: bool,
}

/// Support intelligence metrics.
#[derive(Default)]
pub(super) struct FbSupport {
    pub(super) support_triage_count: u32,
    pub(super) support_alert_fired: bool,
    pub(super) support_federation_graduated: usize,
    pub(super) support_efe: f64,
}

/// Result of the feedback phase (Phases 14–21+).
#[derive(Default)]
pub(super) struct FeedbackPhaseResult {
    pub(super) quality: FbQuality,
    pub(super) consciousness: FbConsciousness,
    pub(super) self_model: FbSelfModel,
    pub(super) reasoning: FbReasoning,
    pub(super) ethics: FbEthics,
    pub(super) evolution: FbEvolution,
    pub(super) loops: FbLoops,
    pub(super) memory: FbMemory,
    pub(super) support: FbSupport,
    pub(super) multi_obj_frontier_size: usize,
    pub(super) grid_encoding_norm: f32,
    pub(super) grid_spatial_complexity: f32,
    pub(super) social_learning_rate_factor: f32,
    /// Mental movie generated from mental simulation.
    #[cfg(feature = "vision-manifold")]
    pub(super) mental_movie: Option<MentalMovie>,
}
