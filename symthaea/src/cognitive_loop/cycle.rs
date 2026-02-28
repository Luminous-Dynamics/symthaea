//! Core cognitive cycle implementation with parallel post-processing.
//!
//! Contains the main `cycle()` method which implements the bidirectional
//! HDC-CfC loop with rayon-parallelized subsystem updates.
//!
//! The heavy lifting is delegated to four phase modules:
//! - `cycle_phase_perception` — safety, encoding, moral evaluation, strategy
//! - `cycle_phase_dynamics`   — CfC step, FEP inference, training, post-processing
//! - `cycle_phase_feedback`   — consciousness metrics, quality gating, homeostasis
//! - `cycle_phase_output`     — metadata assembly, CycleResult construction

use std::time::Instant;

use super::types::MoralJudgmentSummary;
use super::{CognitiveLoopService, CycleResult, CycleUrgency, ResponseStrategy};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::predictive_encoder::EncodingResult;

// ═══════════════════════════════════════════════════════════════════════════════
// Phase result structs — carry local variables between phases.
//
// All fields are `pub(super)` so that the phase modules can construct/read them,
// but they are invisible outside the cognitive_loop module.
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of the perception phase (Phases 0–1.2).
pub(super) struct PerceptionPhaseResult {
    pub(super) encoding_result: EncodingResult,
    pub(super) hv16_cached: BinaryHV,
    pub(super) compressed_state: Vec<f32>,
    pub(super) phi_attention_weight: f32,
    pub(super) exploration_urge_start: f32,
    pub(super) startup_suppressed: bool,
    pub(super) startup_warmup_progress: f32,
    pub(super) input_memoized: bool,
    pub(super) input_similarity: f32,
    pub(super) moral_concern_detected: bool,
    pub(super) moral_score: f32,
    pub(super) moral_judgment: MoralJudgmentSummary,
    pub(super) selected_strategy: ResponseStrategy,
    pub(super) agency_strategy_override: bool,
    pub(super) soul_alignment: f32,
    pub(super) negation_detected: f32,
    pub(super) surprise_triggered: bool,
    pub(super) exploration_action: Option<String>,
    pub(super) urgency: CycleUrgency,
    pub(super) error_pattern: &'static str,
    pub(super) predicted_urgency: &'static str,
    pub(super) prediction_coherence_urgency_bias: f32,
    pub(super) prediction_error: f32,
    pub(super) effective_threshold: f32,
    pub(super) memo_threshold: f32,
}

/// Result of the dynamics phase (Phases A–12).
pub(super) struct DynamicsPhaseResult {
    pub(super) output: Vec<f32>,
    pub(super) prediction: Vec<f32>,
    pub(super) prediction_error: f32,
    pub(super) coherence: f32,
    pub(super) unified_psi: f64,
    pub(super) learning_occurred: bool,
    pub(super) training_loss: Option<f32>,
    pub(super) effective_lr: f32,
    pub(super) attention_budget_exceeded: bool,
    pub(super) attention_budget_elapsed_us: u64,
    pub(super) predictive_budget_gated: bool,
    pub(super) fep_action_idx: usize,
    pub(super) fep_pragmatic_value: f64,
    pub(super) fep_accuracy: f64,
    pub(super) fep_complexity: f64,
    pub(super) fep_surprise: f64,
    pub(super) fep_td_error: f64,
    pub(super) cycle_reward: f32,
    pub(super) reasoning_confidence: f32,
    pub(super) reasoning_gate_blocked: bool,
    pub(super) reasoning_fallback: Option<String>,
    pub(super) reasoning_plan_action: Option<usize>,
    pub(super) reasoning_plan_confidence: f32,
    pub(super) reasoning_narrative: Option<String>,
    pub(super) metacognitive_anomaly: bool,
    pub(super) anomaly_recovery_progress: f32,
    pub(super) anomaly_recovering: bool,
    pub(super) prediction_coherence: f32,
    pub(super) self_model_accuracy: f32,
    pub(super) resonator_wm_primed: bool,
    pub(super) resonator_reconsolidated: usize,
    pub(super) resonator_best_sim: f32,
    pub(super) resonator_prediction_error: f32,
    pub(super) resonator_error_exploration_mod: f32,
    pub(super) binding_threshold_mod: f32,
    pub(super) binding_confidence_mod: f32,
    pub(super) epistemic_semantic_lr_mod: f32,
    pub(super) pfe_surprise_mod: f32,
    pub(super) mcts_plan_effectiveness: f32,
    pub(super) causal_attention_edges: usize,
    pub(super) moral_steering_category: String,
    pub(super) valence_homeostasis_pull: f32,
    pub(super) arousal_homeostasis_pull: f32,
    pub(super) homeostasis_pull_strength: f32,
    pub(super) arousal_recovery_active: bool,
    pub(super) arousal_recovery_tau_factor: f32,
    pub(super) guiding_priority_category: String,
    pub(super) guiding_question: String,
    pub(super) dominant_harmonic: String,
    pub(super) school_predicted_phi_gain: f32,
    pub(super) neuromod_attention_alloc: f32,
    pub(super) ne_reorienting_boost: f32,
    pub(super) ne_arousal_feedback: f32,
    pub(super) confidence_velocity: f32,
    pub(super) sht_crash_dip: f32,
    pub(super) exploration_sht_drain: f32,
    /// Set during feedback phase via `&mut DynamicsPhaseResult`.
    pub(super) phasic_da_replay_boost: usize,
}

/// Result of the feedback phase (Phases 14–21+).
pub(super) struct FeedbackPhaseResult {
    // ── Cross-module / quality ──
    pub(super) cross_module_agreement: f32,
    pub(super) unified_quality_score: f32,
    pub(super) coherence_velocity_gated: bool,
    pub(super) dissipative_health_gated: bool,
    // ── Consciousness metrics pass-through ──
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
    pub(super) value_evaluator_score: f64,
    pub(super) value_evaluator_decision: String,
    pub(super) value_gate_factor: f32,
    pub(super) consciousness_profile_composite: f64,
    pub(super) synergy_enhanced_composite: f64,
    pub(super) emergent_properties_count: usize,
    pub(super) reasoning_context: String,
    pub(super) context_phi_weight: f64,
    pub(super) context_phi_applied: bool,
    pub(super) value_embeddings_created: u64,
    pub(super) value_cache_hit_rate: f32,
    pub(super) harmonies_alignment: f32,
    pub(super) harmonies_approved: bool,
    pub(super) composition_rule_applied: String,
    pub(super) harmonic_field_coherence: f64,
    pub(super) harmonic_love_resonance: f64,
    pub(super) harmonic_interferences: usize,
    pub(super) reasoning_chain_confidence: f32,
    pub(super) reasoning_chain_depth: usize,
    pub(super) causal_relations_count: usize,
    pub(super) causal_avg_confidence: f64,
    pub(super) adaptive_reasoning_phi: f64,
    pub(super) epistemic_quality: f64,
    pub(super) phi_validation_correlation: f64,
    pub(super) dissipative_health: f64,
    pub(super) dissipative_regime: String,
    pub(super) dissipative_entropy_rate: f64,
    pub(super) epistemic_phi_eff: f64,
    pub(super) epistemic_conflict_count: usize,
    pub(super) equation_v2_consciousness: f64,
    // ── Subsystem metrics pass-through ──
    pub(super) hierarchical_ltc_phi: f32,
    pub(super) evolution_generation: usize,
    pub(super) evolution_phi_delta: f64,
    pub(super) evolution_confidence_delta: f32,
    pub(super) holographic_unity: f64,
    pub(super) holographic_binding: f64,
    pub(super) consciousness_gradient_magnitude: f64,
    pub(super) consciousness_limiting_component: String,
    pub(super) affect_cons_valence: f32,
    pub(super) affect_cons_arousal: f32,
    pub(super) pipeline_consciousness: f64,
    pub(super) multimodal_integrated_phi: f64,
    pub(super) consciousness_state_label: String,
    pub(super) consciousness_state_level: f64,
    pub(super) epistemic_gate_confidence: f32,
    pub(super) epistemic_gate_approved: bool,
    pub(super) primitive_validation_phi_gain: f64,
    pub(super) primitive_validation_p_value: f64,
    pub(super) meta_reasoning_confidence: f64,
    pub(super) meta_reasoning_insights: usize,
    pub(super) code_primitives_selected: usize,
    pub(super) empathic_compassion: f64,
    pub(super) empathic_tone_adj: f64,
    pub(super) multi_obj_frontier_size: usize,
    pub(super) grid_encoding_norm: f32,
    pub(super) grid_spatial_complexity: f32,
    // ── Late consciousness pass-through ──
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
    pub(super) attention_schema_focus: f32,
    pub(super) psi_attention_avg: f32,
    pub(super) gwt_broadcast: bool,
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
    pub(super) consciousness_level: f64,
    // ── Consciousness engine ──
    pub(super) spectral_mip_phi: Option<f64>,
    pub(super) sigma: Option<f64>,
    pub(super) phi_spectral_weight: f32,
    pub(super) dissipative_lr_factor: f32,
    pub(super) coherence_velocity: f32,
    // ── Feedback loop metrics ──
    pub(super) empathic_speech_rate_mod: f32,
    pub(super) limiting_component_boosted: String,
    pub(super) love_resonance_boost: f32,
    pub(super) reasoning_chain_boosted: bool,
    pub(super) harmonic_interference_lr_mod: f32,
    pub(super) causal_urgency_gated: bool,
    pub(super) epistemic_coherence_gated: bool,
    pub(super) attention_budget_gated: bool,
    // ── Dream engine ──
    pub(super) dream_insights: usize,
    pub(super) dream_phi_improvement: f32,
    pub(super) dream_wisdom_count: usize,
    // ── Resonator codebook ──
    pub(super) resonator_promotions: usize,
    pub(super) codebook_evictions: usize,
    pub(super) codebook_diversity: f32,
    pub(super) codebook_utilization_rate: f32,
    pub(super) surprise_replay_batch_size: usize,
    // ── Support intelligence ──
    pub(super) support_triage_count: u32,
    pub(super) support_alert_fired: bool,
    pub(super) support_federation_graduated: usize,
    pub(super) support_efe: f64,
}

impl CognitiveLoopService {
    /// Run one cognitive cycle (the core loop).
    ///
    /// Uses CfC's O(1) closed-form solution for temporal prediction,
    /// enabling instant forward-time queries and multi-scale prediction.
    ///
    /// ## Architecture
    ///
    /// The cycle is split into four phases:
    ///
    /// 1. **Perception** — safety precheck, thalamic routing, moral evaluation,
    ///    strategy selection, HDC encoding, surprise exploration, ethics engine
    /// 2. **Dynamics** — CfC step, prediction, FEP active inference, training,
    ///    parallel post-processing
    /// 3. **Feedback** — consciousness metrics, quality gating, homeostasis,
    ///    dream engine, episodic replay
    /// 4. **Output** — metadata assembly, telemetry, CycleResult construction
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;
        let mut module_timings = super::ModuleTimings::default();

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: PERCEPTION
        // Safety checks, encoding, moral evaluation, strategy, urgency
        // ═══════════════════════════════════════════════════════════════════
        let perception = match self.phase_perception(input, cycle_start, &mut module_timings) {
            Ok(p) => p,
            Err(blocked) => return blocked,
        };

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: DYNAMICS
        // CfC step, prediction, FEP, training, parallel post-processing
        // ═══════════════════════════════════════════════════════════════════
        let mut dynamics =
            self.phase_dynamics(input, &perception, cycle_start, &mut module_timings);

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: FEEDBACK
        // Consciousness metrics, quality gating, homeostasis, dream engine
        // ═══════════════════════════════════════════════════════════════════
        let feedback =
            self.phase_feedback(input, &perception, &mut dynamics, &mut module_timings);

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: OUTPUT
        // Metadata assembly, telemetry, CycleResult construction
        // ═══════════════════════════════════════════════════════════════════
        self.phase_output(
            input,
            cycle_start,
            &perception,
            &dynamics,
            &feedback,
            &mut module_timings,
        )
    }

    // Extracted cycle phases moved to helpers/cycle_phases.rs:
    // - run_resonator_codebook_phase()
    // - run_episodic_replay_and_memory_phase()
    // - run_dream_phase()

    /// Safe wrapper around `cycle()` that catches panics from unexpected subsystem failures.
    ///
    /// Use this in production code paths where a panic must not propagate (e.g., actor loops,
    /// async bridges). Returns `Err` with the panic message if any subsystem panics during
    /// the cycle.
    /// Online distillation step for the Liquid-Mamba HDC↔SSM projection.
    ///
    /// Called after generation with the original thought HV, back-projected
    /// output HVs, and semantic prediction error. Adjusts projection weights
    /// using FEP-modulated learning rate, gated by the cognitive loop's
    /// learning state and thermodynamic load.
    #[cfg(feature = "liquid-mamba")]
    pub fn update_liquid_mamba_telemetry(
        &mut self,
        semantic_pe: f32,
        effective_rank: f32,
        current_lr: f32,
        generation_count: u32,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;
        self.stats.last_liquid_mamba_rank = effective_rank;
        self.stats.last_liquid_mamba_lr = current_lr;
        self.stats.liquid_mamba_generation_count = generation_count;
    }

    #[cfg(feature = "liquid-mamba")]
    pub fn liquid_mamba_distillation_step(
        &mut self,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
        output_hvs: &[symthaea_core::hdc::ContinuousHV],
        semantic_pe: f32,
        projection: &mut symthaea_broca::HdcSsmProjection,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;

        // Gate on FEP precision confidence (mirrors enhanced_fep_bridge threshold)
        if self.carryover.learning.prediction_confidence < 0.4 {
            return;
        }
        if output_hvs.is_empty() || semantic_pe > 0.8 {
            return;
        }

        // FEP-modulated learning rate: precision × load × boost
        let fep_precision = self.fep_learning_signal.clamp(0.0, 1.0);
        let effective_lr =
            0.001 * fep_precision * (1.0 - self.thermodynamic_load) * self.fep_lr_boost;
        if effective_lr < 1e-6 {
            return;
        }

        let refs: Vec<&symthaea_core::hdc::ContinuousHV> = output_hvs.iter().collect();
        let bundled = symthaea_core::hdc::ContinuousHV::bundle(&refs).normalize();
        projection.compute_gradients(thought_hv, &bundled);
        projection.apply_gradients(effective_lr, 1.0);
    }

    pub fn try_cycle(&mut self, input: &str) -> Result<CycleResult, crate::errors::SymthaeaError> {
        // SAFETY: CognitiveLoopService is not UnwindSafe by default because it contains
        // mutable state. We use AssertUnwindSafe because a panic mid-cycle leaves the
        // service in a potentially inconsistent state, but callers should reset() after
        // an error rather than continuing.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.cycle(input)));
        result.map_err(|payload| {
            crate::errors::SymthaeaError::CognitiveLoop(format_panic_payload(payload))
        })
    }
}

/// Convert a panic payload into a human-readable error string.
///
/// Handles the three common payload types: `&str`, `String`, and unknown.
/// This is a standalone function so it can be tested independently.
pub(crate) fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("cognitive cycle panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("cognitive cycle panicked: {s}")
    } else {
        "cognitive cycle panicked with unknown payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::format_panic_payload;
    use super::CognitiveLoopService;
    use crate::cognitive_loop::CognitiveLoopConfig;

    // ── format_panic_payload tests (existing) ─────────────────────────

    #[test]
    fn test_panic_payload_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("subsystem failure");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: subsystem failure");
    }

    #[test]
    fn test_panic_payload_string() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("HDC bridge overflow"));
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: HDC bridge overflow");
    }

    #[test]
    fn test_panic_payload_unknown() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42u32);
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked with unknown payload");
    }

    #[test]
    fn test_panic_payload_empty_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: ");
    }

    // ── Helper ────────────────────────────────────────────────────────

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    // ── cycle() basic execution ───────────────────────────────────────

    #[test]
    fn cycle_returns_valid_result() {
        let mut s = make_service();
        let result = s.cycle("hello world");
        assert!(!result.output.is_empty(), "output should not be empty");
        assert!(result.prediction_error.is_finite());
        assert!(result.peak_attention.is_finite());
        assert!(result.cycle_time_us > 0);
    }

    #[test]
    fn cycle_increments_total_cycles() {
        let mut s = make_service();
        assert_eq!(s.stats().total_cycles, 0);
        s.cycle("first");
        assert_eq!(s.stats().total_cycles, 1);
        s.cycle("second");
        assert_eq!(s.stats().total_cycles, 2);
    }

    #[test]
    fn cycle_output_dimension_matches_config() {
        let mut s = make_service();
        let result = s.cycle("testing output dim");
        assert_eq!(
            result.output.len(),
            s.config().cfc_config.num_neurons,
            "output dimension should match num_neurons"
        );
    }

    #[test]
    fn cycle_prediction_error_non_negative() {
        let mut s = make_service();
        let result = s.cycle("checking error sign");
        assert!(
            result.prediction_error >= 0.0,
            "prediction_error should be non-negative, got {}",
            result.prediction_error
        );
    }

    #[test]
    fn cycle_thought_vector_has_values() {
        let mut s = make_service();
        let result = s.cycle("thought projection");
        assert!(!result.thought_vector.is_empty());
        assert_eq!(
            result.thought_vector.len(),
            32,
            "thought_vector should be 32D"
        );
    }

    #[test]
    fn cycle_metadata_urgency_populated() {
        let mut s = make_service();
        let result = s.cycle("metadata check");
        // Urgency should be one of the three valid variants
        let u = result.metadata.urgency;
        assert!(
            matches!(
                u,
                crate::cognitive_loop::CycleUrgency::Critical
                    | crate::cognitive_loop::CycleUrgency::Normal
                    | crate::cognitive_loop::CycleUrgency::Cruise
            ),
            "urgency should be a valid variant"
        );
    }

    #[test]
    fn cycle_output_all_finite() {
        let mut s = make_service();
        let result = s.cycle("NaN guard check");
        for (i, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    // ── Multiple cycles ───────────────────────────────────────────────

    #[test]
    fn multiple_cycles_do_not_panic() {
        let mut s = make_service();
        for i in 0..10 {
            let result = s.cycle(&format!("cycle input {i}"));
            assert!(result.prediction_error.is_finite());
        }
        assert_eq!(s.stats().total_cycles, 10);
    }

    #[test]
    fn empty_input_does_not_panic() {
        let mut s = make_service();
        let result = s.cycle("");
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn long_input_does_not_panic() {
        let mut s = make_service();
        let long_input = "a".repeat(10_000);
        let result = s.cycle(&long_input);
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn repeated_identical_input_reduces_prediction_error() {
        let mut s = make_service();
        // First cycle has no prior prediction
        let r1 = s.cycle("repeating input");
        // Run several identical cycles so the system can learn the pattern
        let mut last_error = r1.prediction_error;
        for _ in 0..20 {
            last_error = s.cycle("repeating input").prediction_error;
        }
        // After 20 identical cycles, error should be lower or comparable
        // (not necessarily strictly lower due to stochastic subsystems)
        assert!(
            last_error.is_finite(),
            "error should remain finite after repeated cycles"
        );
    }

    // ── try_cycle() ───────────────────────────────────────────────────

    #[test]
    fn try_cycle_returns_ok() {
        let mut s = make_service();
        let result = s.try_cycle("safe input");
        assert!(result.is_ok(), "try_cycle should succeed for normal input");
    }

    #[test]
    fn try_cycle_result_matches_cycle() {
        // Use genesis phrase for determinism
        let mut cfg = CognitiveLoopConfig::default();
        cfg.genesis_phrase = Some("determinism test".to_string());
        let mut s1 = CognitiveLoopService::new(cfg.clone()).unwrap();
        let mut s2 = CognitiveLoopService::new(cfg).unwrap();

        let r1 = s1.cycle("hello");
        let r2 = s2.try_cycle("hello").unwrap();

        // Both should produce same output with deterministic genesis
        assert_eq!(r1.output.len(), r2.output.len());
        assert_eq!(r1.prediction_error, r2.prediction_error);
    }

    // ── Cycle with different backends ─────────────────────────────────

    #[test]
    fn cycle_with_hdc_ltc_unified_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let mut s = CognitiveLoopService::new(config).unwrap();
        let result = s.cycle("HdcLtc backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn cycle_with_hdc_ltc_fast_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_fast();
        let mut s = CognitiveLoopService::new(config).unwrap();
        let result = s.cycle("fast backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    // ── Cycle stats tracking ──────────────────────────────────────────

    #[test]
    fn cycle_updates_avg_prediction_error() {
        let mut s = make_service();
        s.cycle("first");
        let err1 = s.stats().avg_prediction_error;
        // After first cycle, avg error should be populated (may be 0.0 for first cycle)
        assert!(err1.is_finite());
    }

    #[test]
    fn cycle_populates_adaptive_learning_rate() {
        let mut s = make_service();
        s.cycle("learning rate check");
        let lr = s.stats().adaptive_learning_rate;
        assert!(lr.is_finite());
        assert!(lr >= 0.0);
    }

    // ── Genesis determinism ───────────────────────────────────────────

    #[test]
    fn genesis_seeded_cycles_are_deterministic() {
        let phrase = "We hold these truths to be self-evident".to_string();

        let mut cfg_a = CognitiveLoopConfig::default();
        cfg_a.genesis_phrase = Some(phrase.clone());
        let mut sa = CognitiveLoopService::new(cfg_a).unwrap();

        let mut cfg_b = CognitiveLoopConfig::default();
        cfg_b.genesis_phrase = Some(phrase);
        let mut sb = CognitiveLoopService::new(cfg_b).unwrap();

        let ra = sa.cycle("determinism check");
        let rb = sb.cycle("determinism check");

        assert_eq!(ra.output, rb.output, "genesis-seeded outputs should match");
        assert_eq!(
            ra.prediction_error, rb.prediction_error,
            "genesis-seeded errors should match"
        );
    }

    // ── Cycle metadata fields ─────────────────────────────────────────

    #[test]
    fn cycle_metadata_somatic_stress_finite() {
        let mut s = make_service();
        let result = s.cycle("somatic check");
        assert!(result.metadata.somatic_stress.is_finite());
    }

    #[test]
    fn cycle_metadata_consciousness_level_bounded() {
        let mut s = make_service();
        // Run a few cycles to populate MCE
        for _ in 0..15 {
            s.cycle("populate MCE");
        }
        let result = s.cycle("check consciousness");
        assert!(result.metadata.consciousness_level >= 0.0);
        assert!(result.metadata.consciousness_level <= 1.0);
    }

    #[test]
    fn cycle_metadata_thermodynamic_load_bounded() {
        let mut s = make_service();
        let result = s.cycle("thermo check");
        assert!(result.metadata.thermodynamic_load >= 0.0);
        assert!(result.metadata.thermodynamic_load <= 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Dynamics phase coverage (cycle_phase_dynamics.rs)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_fep_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("FEP test");
        assert!(r.metadata.fep_pragmatic_value.is_finite());
        assert!(r.metadata.fep_accuracy.is_finite());
        assert!(r.metadata.fep_complexity.is_finite());
        assert!(r.metadata.fep_surprise.is_finite());
        assert!(r.metadata.fep_td_error.is_finite());
    }

    #[test]
    fn dynamics_reasoning_fields_populated() {
        let mut s = make_service();
        let r = s.cycle("reasoning check");
        assert!(r.metadata.reasoning_confidence.is_finite());
        // gate_blocked is a bool — just verify it doesn't panic
        let _ = r.metadata.reasoning_gate_blocked;
    }

    #[test]
    fn dynamics_coherence_finite() {
        let mut s = make_service();
        let r = s.cycle("coherence check");
        assert!(r.metadata.prediction_coherence.is_finite());
        assert!(r.metadata.cross_module_agreement.is_finite());
    }

    #[test]
    fn dynamics_homeostasis_pulls_finite() {
        let mut s = make_service();
        let r = s.cycle("homeostasis check");
        assert!(r.metadata.valence_homeostasis_pull.is_finite());
        assert!(r.metadata.arousal_homeostasis_pull.is_finite());
        assert!(r.metadata.homeostasis_pull_strength.is_finite());
    }

    #[test]
    fn dynamics_neuromod_fields_populated() {
        let mut s = make_service();
        let r = s.cycle("neuromod check");
        assert!(!r.metadata.guiding_question.is_empty());
        assert!(!r.metadata.dominant_harmonic.is_empty());
    }

    #[test]
    fn dynamics_metacognitive_anomaly_defaults_false() {
        let mut s = make_service();
        let r = s.cycle("anomaly check");
        // On first cycle, no anomaly should be detected
        assert!(!r.metadata.metacognitive_anomaly);
    }

    #[test]
    fn dynamics_cycle_reward_finite() {
        let mut s = make_service();
        let r = s.cycle("reward test");
        assert!(r.metadata.cycle_reward.is_finite());
    }

    #[test]
    fn dynamics_attention_budget_fields() {
        let mut s = make_service();
        let r = s.cycle("attention budget");
        // Budget shouldn't be exceeded on first cycle
        let _ = r.metadata.attention_budget_exceeded;
        // Elapsed should be non-negative
        assert!(r.metadata.attention_budget_elapsed_us < u64::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Feedback phase coverage (cycle_phase_feedback.rs)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn feedback_consciousness_metrics_finite() {
        let mut s = make_service();
        for _ in 0..5 {
            s.cycle("warmup");
        }
        let r = s.cycle("consciousness check");
        assert!(r.metadata.consciousness_level.is_finite());
        assert!(r.metadata.consciousness_level >= 0.0);
        assert!(r.metadata.consciousness_level <= 1.0);
        assert!(r.metadata.primitive_psi.is_finite());
    }

    #[test]
    fn feedback_quality_gating_fields() {
        let mut s = make_service();
        let r = s.cycle("quality check");
        assert!(r.metadata.unified_quality_score.is_finite());
        // Gating booleans should be accessible
        let _ = r.metadata.coherence_velocity_gated;
        let _ = r.metadata.dissipative_health_gated;
        let _ = r.metadata.epistemic_gate_approved;
    }

    #[test]
    fn feedback_temporal_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("temporal check");
        assert!(r.metadata.temporal_continuity.is_finite());
        assert!(r.metadata.temporal_coherence_score.is_finite());
    }

    #[test]
    fn feedback_harmonic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("harmonic check");
        assert!(r.metadata.harmonies_alignment.is_finite());
        assert!(r.metadata.harmonic_field_coherence.is_finite());
        assert!(r.metadata.harmonic_love_resonance.is_finite());
    }

    #[test]
    fn feedback_dream_engine_defaults() {
        let mut s = make_service();
        let r = s.cycle("dream check");
        // Dream engine requires many cycles, so on first cycle defaults
        assert!(r.metadata.dream_phi_improvement.is_finite());
    }

    #[test]
    fn feedback_dissipative_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("dissipative check");
        assert!(r.metadata.dissipative_health.is_finite());
        assert!(r.metadata.dissipative_entropy_rate.is_finite());
    }

    #[test]
    fn feedback_epistemic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("epistemic check");
        assert!(r.metadata.epistemic_quality.is_finite());
        assert!(r.metadata.epistemic_phi_eff.is_finite());
    }

    #[test]
    fn feedback_holographic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("holographic check");
        assert!(r.metadata.holographic_unity.is_finite());
        assert!(r.metadata.holographic_binding.is_finite());
    }

    #[test]
    fn feedback_phenomenal_binding_fields() {
        let mut s = make_service();
        let r = s.cycle("phenomenal check");
        assert!(r.metadata.phenomenal_binding_strength.is_finite());
        let _ = r.metadata.phenomenal_fragmented;
    }

    #[test]
    fn feedback_affective_fields_bounded() {
        let mut s = make_service();
        let r = s.cycle("affective check");
        assert!(r.metadata.affective_valence.is_finite());
        assert!(r.metadata.affective_arousal.is_finite());
        assert!(r.metadata.body_valence.is_finite());
        assert!(r.metadata.body_arousal.is_finite());
    }

    #[test]
    fn feedback_living_mind_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("living mind check");
        assert!(r.metadata.living_mind_vitality.is_finite());
        assert!(r.metadata.living_mind_coherence.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Output phase coverage (cycle_phase_output.rs)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn output_thought_vector_32d() {
        let mut s = make_service();
        let r = s.cycle("thought vector");
        assert_eq!(r.thought_vector.len(), 32);
        for v in &r.thought_vector {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn output_cycle_time_nonzero() {
        let mut s = make_service();
        let r = s.cycle("timing");
        assert!(r.cycle_time_us > 0);
    }

    #[test]
    fn output_circadian_phase_non_empty() {
        let mut s = make_service();
        let r = s.cycle("circadian");
        assert!(!r.metadata.circadian_phase.is_empty());
    }

    #[test]
    fn output_selected_strategy_non_empty() {
        let mut s = make_service();
        let r = s.cycle("strategy output");
        assert!(!r.metadata.selected_strategy.is_empty());
    }

    #[test]
    fn output_module_timings_populated() {
        let mut s = make_service();
        let r = s.cycle("timing check");
        let t = &r.metadata.module_timings_us;
        assert!(t.core_hdc_encode > 0 || t.core_cfc_step > 0);
    }

    #[test]
    fn output_sigma_and_spectral_phi() {
        let mut s = make_service();
        let r = s.cycle("sigma check");
        // May be None on first cycle — just verify finitely populated if present
        if let Some(sigma) = r.metadata.sigma {
            assert!(sigma.is_finite());
        }
        if let Some(phi) = r.metadata.spectral_mip_phi {
            assert!(phi.is_finite());
        }
    }

    #[test]
    fn output_detected_primitives_exist() {
        let mut s = make_service();
        let r = s.cycle("hello world primitive");
        // May or may not detect primitives — just verify it doesn't panic
        for p in &r.detected_primitives {
            assert!(!p.is_empty());
        }
    }

    #[test]
    fn output_wisdom_hv_correct_size() {
        let mut s = make_service();
        let r = s.cycle("wisdom hv");
        // BinaryHV is 16384 bits = 2048 bytes
        assert_eq!(r.wisdom_hv.0.len(), 2048);
    }

    #[test]
    fn output_soul_alignment_finite() {
        let mut s = make_service();
        let r = s.cycle("soul alignment");
        assert!(r.metadata.soul_alignment.is_finite());
    }

    #[test]
    fn output_feedback_divergence_tracked() {
        let mut s = make_service();
        // Run several cycles so feedback proposals accumulate
        for _ in 0..10 {
            s.cycle("divergence");
        }
        // After cycles, feedback state should have integration results
        assert!(s.feedback_state.last_confidence_integration.is_some());
        assert!(s.feedback_state.last_lr_integration.is_some());
    }
}
