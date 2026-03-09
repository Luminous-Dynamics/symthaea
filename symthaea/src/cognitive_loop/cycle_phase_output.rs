//! Output phase of the cognitive cycle.
//!
//! Extracts the final metadata assembly and CycleResult construction from
//! the original `cycle()` method.

use std::time::Instant;

use super::phase_results::{DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult};
use super::{CognitiveLoopService, CycleResult};

impl CognitiveLoopService {
    /// Output phase: metadata assembly, telemetry, CycleResult construction.
    pub(super) fn phase_output(
        &mut self,
        _input: &str,
        cycle_start: Instant,
        perception: &PerceptionPhaseResult,
        dynamics: &DynamicsPhaseResult,
        feedback: &FeedbackPhaseResult,
        module_timings: &mut super::ModuleTimings,
    ) -> CycleResult {
        let thalamic_depth_score = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => 1.0f32,
            super::CognitiveDepth::Cortical => 0.5,
            super::CognitiveDepth::Reflex => 0.2,
        };

        let value_trend = self.primitive_tier.value_feedback.recent_trend(50);
        let circadian_phase_str = self.biorhythm_mgr.rhythm.phase.as_str();
        let selected_strategy_str = perception.strategy.selected_strategy.as_str();

        let _t = Instant::now();
        let moral_anomaly_report = self.ethics_engine.last_anomaly_report().clone();
        let mut metadata = super::CycleMetadata {
            surprise_triggered: perception.exploration.surprise_triggered,
            prefrontal_veto: feedback.self_model.prefrontal_veto,
            reasoning_confidence: dynamics.reasoning.reasoning_confidence,
            exploration_action: perception.exploration.exploration_action.clone(),
            reasoning_gate_blocked: dynamics.reasoning.reasoning_gate_blocked,
            reasoning_fallback: dynamics.reasoning.reasoning_fallback.clone(),
            reasoning_plan_action: dynamics.reasoning.reasoning_plan_action,
            reasoning_plan_confidence: dynamics.reasoning.reasoning_plan_confidence,
            reasoning_narrative: dynamics.reasoning.reasoning_narrative.clone(),
            quality: super::QualityDiagnostics {
                meta_cognitive_accuracy: feedback.self_model.meta_cognitive_accuracy,
                meta_cognitive_depth: feedback.self_model.meta_cognitive_depth,
                dissipative_health: feedback.quality.dissipative_health,
                dissipative_regime: feedback.quality.dissipative_regime.clone(),
                dissipative_entropy_rate: feedback.quality.dissipative_entropy_rate,
                epistemic_phi_eff: feedback.reasoning.epistemic_phi_eff,
                equation_v2_consciousness: feedback.consciousness.equation_v2_consciousness,
                hierarchical_ltc_phi: feedback.evolution.hierarchical_ltc_phi,
                unified_quality_score: feedback.quality.unified_quality_score,
                dissipative_health_gated: feedback.quality.dissipative_health_gated,
                dissipative_lr_factor: feedback.quality.dissipative_lr_factor,
                coherence_velocity: feedback.quality.coherence_velocity,
                coherence_velocity_gated: feedback.quality.coherence_velocity_gated,
                anomaly_recovery_progress: dynamics.homeostasis.anomaly_recovery_progress,
                anomaly_recovering: dynamics.homeostasis.anomaly_recovering,
            },
            narrative_self_psi: feedback.self_model.narrative_self_psi,
            body_phi_modulation: feedback.self_model.body_psi_modulation,
            body_valence: feedback.self_model.body_valence,
            body_arousal: feedback.self_model.body_arousal,
            consciousness_level: feedback.consciousness.consciousness_level,
            predictive_self_safety: feedback.self_model.predictive_self_safety,
            attention: super::AttentionMetrics {
                attention_schema_focus: feedback.self_model.attention_schema_focus,
                gwt_broadcast: feedback.self_model.gwt_broadcast,
                gwt_coalition_size: feedback.self_model.gwt_coalition_size,
                psi_attention_avg: feedback.self_model.psi_attention_avg,
                phi_attention_weight: perception.encoding.phi_attention_weight,
                attention_budget_exceeded: dynamics.attention.attention_budget_exceeded,
                attention_budget_elapsed_us: dynamics.attention.attention_budget_elapsed_us,
                input_similarity: perception.encoding.input_similarity,
                input_memoized: perception.encoding.input_memoized,
                attention_budget_gated: feedback.loops.attention_budget_gated,
                attention_shift_applied: self.stats.attention_shift,
                attention_fatigue: feedback.self_model.attention_fatigue,
                attention_prediction_accuracy: feedback.self_model.attention_prediction_accuracy,
            },
            resonance_frequency: feedback.self_model.resonance_frequency,
            quantum_coherence_level: feedback.self_model.quantum_coherence_level,
            temporal_coherence_score: feedback.self_model.temporal_coherence_score,
            temporal_discontinuity: feedback.self_model.temporal_discontinuity,
            embodied_phi_modulation: feedback.self_model.embodied_psi_modulation,
            embodied_agency: feedback.self_model.embodied_agency,
            narrative_gwt_veto: feedback.self_model.narrative_gwt_veto,
            narrative_gwt_self_psi: feedback.self_model.narrative_gwt_self_psi,
            living_mind_vitality: feedback.self_model.living_mind_vitality,
            living_mind_coherence: feedback.self_model.living_mind_coherence,
            urgency: perception.urgency.urgency,
            memory: super::MemoryResonatorMetrics {
                dream_insights: feedback.memory.dream_insights,
                dream_phi_improvement: feedback.memory.dream_phi_improvement,
                dream_wisdom_count: feedback.memory.dream_wisdom_count,
                continuity_replay_triggered: feedback.consciousness.continuity_replay_needed,
                resonator_codebook_size: self
                    .resonator_memory
                    .as_ref()
                    .and_then(|m| m.resonator.codebooks.first())
                    .map(|cb| cb.len())
                    .unwrap_or(0),
                resonator_episodes: self.resonator_memory.as_ref().map(|m| m.len()).unwrap_or(0),
                resonator_factorization_iters: self
                    .resonator_memory
                    .as_ref()
                    .map(|m| m.resonator.iterations())
                    .unwrap_or(0),
                resonator_wm_primed: dynamics.resonator.resonator_wm_primed,
                resonator_reconsolidated: dynamics.resonator.resonator_reconsolidated,
                resonator_promotions: feedback.memory.resonator_promotions,
                resonator_best_sim: dynamics.resonator.resonator_best_sim,
                codebook_evictions: feedback.memory.codebook_evictions,
                codebook_diversity: feedback.memory.codebook_diversity,
                resonator_prediction_error: dynamics.resonator.resonator_prediction_error,
                codebook_utilization_rate: feedback.memory.codebook_utilization_rate,
                surprise_replay_batch_size: feedback.memory.surprise_replay_batch_size,
            },
            fep: super::FepTelemetry {
                fep_action: dynamics.fep.fep_action_idx,
                fep_pragmatic_value: dynamics.fep.fep_pragmatic_value,
                fep_accuracy: dynamics.fep.fep_accuracy,
                fep_complexity: dynamics.fep.fep_complexity,
                fep_surprise: dynamics.fep.fep_surprise,
                fep_td_error: dynamics.fep.fep_td_error,
                predictive_free_energy: feedback.self_model.predictive_free_energy,
                predictive_phi_modulation: feedback.self_model.predictive_psi_modulation,
            },
            cross_modal_binding_strength: feedback.self_model.cross_modal_binding_strength,
            cross_modal_psi: feedback.self_model.cross_modal_psi,
            affective_valence: feedback.self_model.affective_valence,
            affective_arousal: feedback.self_model.affective_arousal,
            thermodynamic_entropy: feedback.self_model.thermodynamic_entropy,
            thermodynamic_free_energy: feedback.self_model.thermodynamic_free_energy,
            phenomenal_binding_strength: feedback.self_model.phenomenal_binding_strength,
            phenomenal_fragmented: feedback.self_model.phenomenal_fragmented,
            hierarchical_total_free_energy: feedback.self_model.hierarchical_total_free_energy,
            primitive_psi: feedback.consciousness.primitive_psi,
            temporal_causal_chains: feedback.consciousness.temporal_causal_chains,
            temporal_continuity: feedback.consciousness.temporal_continuity,
            temporal_max_chain_length: feedback.consciousness.temporal_max_chain_length,
            lattice_height: feedback.consciousness.lattice_height,
            lattice_width: feedback.consciousness.lattice_width,
            lattice_join_concept: feedback
                .consciousness
                .lattice_join_concept
                .clone()
                .unwrap_or_default(),
            causal_codebook_entries: feedback.consciousness.causal_codebook_entries_len,
            compositionality_total: feedback.consciousness.compositionality_total,
            composition_rule_applied: feedback.ethics.composition_rule_applied.clone(),
            harmonics: super::HarmonicMetrics {
                harmonies_alignment: feedback.ethics.harmonies_alignment,
                harmonies_approved: feedback.ethics.harmonies_approved,
                harmonic_field_coherence: feedback.ethics.harmonic_field_coherence,
                harmonic_love_resonance: feedback.ethics.harmonic_love_resonance,
                harmonic_interferences: feedback.ethics.harmonic_interferences,
                harmony_coordinates: *self.ethics_engine.last_harmony_coordinates(),
                moral_scenario_distribution: self
                    .ethics_engine
                    .last_moral_free_energy()
                    .scenario_distribution,
                moral_prior_distribution: self
                    .ethics_engine
                    .last_moral_free_energy()
                    .prior_distribution,
                moral_kl_divergence: self.ethics_engine.last_moral_free_energy().kl_divergence,
                moral_entropy: self.ethics_engine.last_moral_free_energy().entropy,
                moral_surprise: self.ethics_engine.last_moral_free_energy().surprise,
                dominant_harmonic: dynamics.guidance.dominant_harmonic.clone(),
                guiding_question: dynamics.guidance.guiding_question.clone(),
                guiding_priority_category: dynamics.guidance.guiding_priority_category.clone(),
            },
            ethics: super::EthicalTelemetry {
                moral_score: perception.moral.moral_score,
                moral_steering_category: dynamics.guidance.moral_steering_category.clone(),
                value_evaluator_score: feedback.ethics.value_evaluator_score,
                value_evaluator_decision: feedback.ethics.value_evaluator_decision.clone(),
                value_feedback_trend: value_trend,
                value_gate_factor: feedback.ethics.value_gate_factor,
                soul_alignment: perception.moral.soul_alignment,
                empathic_compassion: feedback.ethics.empathic_compassion,
                empathic_tone_adj: feedback.ethics.empathic_tone_adj,
                empathic_speech_rate_mod: feedback.ethics.empathic_speech_rate_mod,
                moral_topo_beta_0: self.ethics_engine.moral_topology().last_summary().beta_0,
                moral_topo_beta_1: self.ethics_engine.moral_topology().last_summary().beta_1,
                moral_topo_beta_2: self.ethics_engine.moral_topology().last_summary().beta_2,
                moral_topo_unity: self.ethics_engine.moral_topology().last_summary().unity,
                moral_topo_completeness: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .completeness,
                moral_topo_circularity: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .circularity,
                moral_topo_free_energy: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .moral_free_energy,
                moral_topo_dominant_harmony: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .dominant_harmony,
                moral_topo_scenario_count: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .scenario_count,
                // Gate anomaly flags on topology_fresh: between evaluations
                // (cadence 30–120 cycles), report stale=false so dashboard/API
                // consumers see clean transitions rather than sticky flags.
                moral_anomaly_score: if self.ethics_engine.last_topology_fresh() {
                    moral_anomaly_report.anomaly_score
                } else {
                    0.0
                },
                moral_value_inversion: self.ethics_engine.last_topology_fresh()
                    && moral_anomaly_report.value_inversion,
                moral_free_energy_spike: self.ethics_engine.last_topology_fresh()
                    && moral_anomaly_report.free_energy_spike,
                moral_drift_alert: self.ethics_engine.last_topology_fresh()
                    && moral_anomaly_report.drift_alert,
                moral_fragmentation_increase: self.ethics_engine.last_topology_fresh()
                    && moral_anomaly_report.fragmentation_increase,
                moral_trajectory_convergence: moral_anomaly_report.trajectory_convergence,
                moral_convergence_severity: moral_anomaly_report.convergence_severity,
                moral_anomaly_response_applied: self.config.enable_moral_anomaly_response
                    && self.ethics_engine.last_topology_fresh()
                    && moral_anomaly_report.anomaly_score > 0.0,
                harmony_entropy: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .harmony_entropy,
                moral_attractor_detected: self
                    .ethics_engine
                    .moral_topology()
                    .last_summary()
                    .attractor_detected,
                in_active_rest: self.stats.in_active_rest,
                stillness_dominance_streak: self.stats.stillness_dominance_streak,
            },
            multi_obj_frontier_size: feedback.multi_obj_frontier_size,
            consciousness_profile_composite: feedback.consciousness.consciousness_profile_composite,
            synergy_enhanced_composite: feedback.consciousness.synergy_enhanced_composite,
            emergent_properties_count: feedback.consciousness.emergent_properties_count,
            reasoning_context: feedback.reasoning.reasoning_context.clone(),
            context_phi_weight: feedback.reasoning.context_phi_weight,
            reasoning_chain_confidence: feedback.reasoning.reasoning_chain_confidence,
            reasoning_chain_depth: feedback.reasoning.reasoning_chain_depth,
            causal_relations_count: feedback.reasoning.causal_relations_count,
            causal_avg_confidence: feedback.reasoning.causal_avg_confidence,
            evolution_generation: feedback.evolution.evolution_generation,
            evolution_phi_delta: feedback.evolution.evolution_phi_delta,
            value_embeddings_created: feedback.ethics.value_embeddings_created,
            value_cache_hit_rate: feedback.ethics.value_cache_hit_rate,
            adaptive_reasoning_phi: feedback.reasoning.adaptive_reasoning_phi,
            epistemic_quality: feedback.reasoning.epistemic_quality,
            phi_validation_correlation: feedback.reasoning.phi_validation_correlation,
            epistemic_conflict_count: feedback.reasoning.epistemic_conflict_count,
            holographic_unity: feedback.consciousness.holographic_unity,
            holographic_binding: feedback.consciousness.holographic_binding,
            consciousness_gradient_magnitude: feedback
                .consciousness
                .consciousness_gradient_magnitude,
            consciousness_limiting_component: feedback
                .consciousness
                .consciousness_limiting_component
                .clone(),
            eq_v2_limiting_component: feedback.consciousness.eq_v2_limiting_component.clone(),
            affect_consciousness_valence: feedback.consciousness.affect_cons_valence,
            affect_consciousness_arousal: feedback.consciousness.affect_cons_arousal,
            pipeline_consciousness: feedback.consciousness.pipeline_consciousness,
            multimodal_integrated_phi: feedback.consciousness.multimodal_integrated_phi,
            consciousness_state_label: feedback.consciousness.consciousness_state_label.clone(),
            consciousness_state_level: feedback.consciousness.consciousness_state_level,
            epistemic_gate_confidence: feedback.reasoning.epistemic_gate_confidence,
            epistemic_gate_approved: feedback.reasoning.epistemic_gate_approved,
            primitive_validation_phi_gain: feedback.evolution.primitive_validation_phi_gain,
            primitive_validation_p_value: feedback.evolution.primitive_validation_p_value,
            meta_reasoning_confidence: feedback.reasoning.meta_reasoning_confidence,
            meta_reasoning_insights: feedback.reasoning.meta_reasoning_insights,
            code_primitives_selected: feedback.reasoning.code_primitives_selected,
            metacognitive_anomaly: dynamics.reasoning.metacognitive_anomaly,
            safety_blocked: false,
            safety_category: None,
            negation_polarity: perception.negation_detected,
            selected_strategy: selected_strategy_str.into(),
            actual_effective_lr: if dynamics.core.learning_occurred {
                dynamics.core.effective_lr
            } else {
                0.0
            },
            cycle_reward: dynamics.core.cycle_reward,
            support_triage_count: feedback.support.support_triage_count,
            support_alert_fired: feedback.support.support_alert_fired,
            support_federation_graduated: feedback.support.support_federation_graduated,
            support_efe: feedback.support.support_efe,
            sigma: feedback.consciousness.sigma,
            spectral_mip_phi: feedback.consciousness.spectral_mip_phi,
            hierarchical_mip_phi: self.carryover.consciousness.last_hierarchical_mip_phi,
            hierarchical_mip_scales: self
                .carryover
                .consciousness
                .last_hierarchical_mip_phi
                .map(|_| 3usize)
                .unwrap_or(0),
            // Structural Phi decomposition
            structural_micro_phi: feedback.consciousness.structural_micro_phi,
            structural_meso_phi: feedback.consciousness.structural_meso_phi,
            structural_macro_phi: feedback.consciousness.structural_macro_phi,
            structural_bottleneck: feedback.consciousness.structural_bottleneck,
            structural_emergence_ratio: feedback.consciousness.structural_emergence_ratio,
            structural_num_clusters: feedback.consciousness.structural_num_clusters,
            // Dynamic consciousness weights
            consciousness_weights: feedback.consciousness.consciousness_weights,
            consciousness_weight_variance: feedback.consciousness.consciousness_weight_variance,
            module_timings_us: {
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings.clone()
            },
            circadian_phase: circadian_phase_str.into(),
            circadian_plasticity: self.biorhythm_mgr.rhythm.plasticity_mod as f32,
            cross_module_agreement: feedback.quality.cross_module_agreement,
            thalamic_depth_score,
            epistemic_gate_gated: !feedback.reasoning.epistemic_gate_approved,
            causal_attention_edges: dynamics.reasoning.causal_attention_edges,
            mcts_plan_effectiveness: dynamics.reasoning.mcts_plan_effectiveness,
            prediction_coherence: dynamics.core.prediction_coherence,
            valence_homeostasis_pull: dynamics.homeostasis.valence_homeostasis_pull,
            arousal_homeostasis_pull: dynamics.homeostasis.arousal_homeostasis_pull,
            arousal_recovery_active: dynamics.homeostasis.arousal_recovery_active,
            arousal_recovery_tau_factor: dynamics.homeostasis.arousal_recovery_tau_factor,
            cycle_duration_us: cycle_start.elapsed().as_micros() as u64,
            school_predicted_phi_gain: dynamics.reasoning.school_predicted_phi_gain,
            epistemic_coherence_gated: feedback.loops.epistemic_coherence_gated,
            phi_validation_cached: self.carryover.quality.phi_validation_correlation,
            phi_spectral_weight: feedback.consciousness.phi_spectral_weight,
            error_pattern: perception.urgency.error_pattern.into(),
            startup_suppressed: perception.startup_suppressed,
            startup_warmup_progress: perception.startup_warmup_progress,
            self_model_accuracy: dynamics.core.self_model_accuracy,
            mode_confidence: self.carryover.urgency.mode_confidence,
            mode_stability_counter: self.carryover.urgency.mode_stability_counter,
            predicted_urgency: perception.urgency.predicted_urgency.into(),
            context_phi_applied: feedback.reasoning.context_phi_applied,
            evolution_confidence_delta: feedback.evolution.evolution_confidence_delta,
            homeostasis_pull_strength: dynamics.homeostasis.homeostasis_pull_strength,
            prediction_coherence_urgency_bias: perception.urgency.prediction_coherence_urgency_bias,
            limiting_component_boosted: feedback.loops.limiting_component_boosted.clone(),
            love_resonance_boost: feedback.loops.love_resonance_boost,
            reasoning_chain_boosted: feedback.loops.reasoning_chain_boosted,
            harmonic_interference_lr_mod: feedback.loops.harmonic_interference_lr_mod,
            resonator_error_exploration_mod: dynamics.resonator.resonator_error_exploration_mod,
            binding_threshold_mod: dynamics.binding_threshold_mod,
            causal_urgency_gated: feedback.loops.causal_urgency_gated,
            epistemic_semantic_lr_mod: dynamics.epistemic_semantic_lr_mod,
            predictive_budget_gated: dynamics.attention.predictive_budget_gated,
            binding_confidence_mod: dynamics.binding_confidence_mod,
            discontinuity_streak: self.carryover.urgency.discontinuity_streak,
            epistemic_reasoning_accelerated: self.carryover.quality.last_epistemic_conflict_count
                > 5,
            agency_strategy_override: perception.strategy.agency_strategy_override,
            pfe_surprise_mod: dynamics.pfe_surprise_mod,
            adaptive_memo_threshold: perception.encoding.memo_threshold,
            grid_encoding_norm: feedback.grid_encoding_norm,
            grid_spatial_complexity: feedback.grid_spatial_complexity,
            mood_temperature: 1.0,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_semantic_pe: self.stats.last_liquid_mamba_pe,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_effective_rank: self.stats.last_liquid_mamba_rank,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_lr: self.stats.last_liquid_mamba_lr,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_generation_count: self.stats.liquid_mamba_generation_count,
            // Partnership / Phi-Dyad
            relational_psi: self.social_mgr.social.relational_psi,
            // Resonant Speech: response profile from neuromod bath signals.
            response_profile: {
                let user_state = crate::resonant_speech::UserState::from_neuromod(
                    self.neuromod.bath.allostatic_load,
                    feedback.consciousness.consciousness_level,
                    self.emotion_contagion.arousal,
                    self.neuromod.bath.oxytocin.effective(),
                );
                match user_state.cognitive_load {
                    crate::resonant_speech::CognitiveLoad::Low => "technical",
                    crate::resonant_speech::CognitiveLoad::Medium => "balanced",
                    crate::resonant_speech::CognitiveLoad::High => "simplified",
                    crate::resonant_speech::CognitiveLoad::Overloaded => "empathic",
                }
                .to_string()
            },
            is_consolidating: self.is_consolidating,
            // Adaptive dynamics telemetry
            epistemic_uncertainty: dynamics.epistemic_uncertainty,
            aleatoric_uncertainty: dynamics.aleatoric_uncertainty,
            theta_phase: ((self.stats.total_cycles as f64 * super::thresholds::THETA_PHASE_ADVANCE)
                % (2.0 * std::f64::consts::PI)) as f32,
            temporal_binding_strength: perception.encoding.temporal_binding_strength,
            // Prediction horizon now computed in dynamics phase and applied to CfC delta_t.
            // Telemetry reports the same value for observability.
            prediction_horizon_scale: dynamics.prediction_horizon_tau,
            fep_tau_factor: dynamics.fep_tau_factor,
            causal_world_model_edges: dynamics.causal_world_model_edges,
            epistemic_budget_scale: dynamics.epistemic_budget_scale,
            feedback_signals_fired: (self.feedback_state.confidence.len()
                + self.feedback_state.learning_rate.len()
                + self.feedback_state.exploration.len()
                + self.feedback_state.threshold.len()) as u32,
            calibration_validations_total: self.neuromod.calibration_validator.total_validations(),
            calibration_improvements: self.neuromod.calibration_validator.improvements,
            calibration_regressions: self.neuromod.calibration_validator.regressions,
            calibration_adjustment_multiplier: self
                .neuromod
                .calibration_validator
                .adjustment_multiplier(),
            calibration_cooldown_duration: self.neuromod.self_assessment.cooldown_duration(),
            feedback_signals_high_water: self.feedback_state.feedback_signals_high_water,
            feedback_dampened_count: self.feedback_state.feedback_dampened_count,
            feedback_signal_diversity: self.feedback_state.signal_diversity(),
            avg_transition_cost: self.stats.avg_transition_cost,
            feedback_dominant_source: self.feedback_state.dominant_source().to_string(),
            error_slope: perception.urgency.error_slope,
            oscillation_ratio: perception.urgency.oscillation_ratio,
            mode_transitions: self.stats.mode_transitions as u32,
            smoothed_epistemic_uncertainty: {
                // EMA smoothing of epistemic uncertainty (alpha=0.2)
                let raw = dynamics.epistemic_uncertainty;
                let prev = self.carryover.quality.smoothed_epistemic_uncertainty;
                if prev == 0.0 && self.stats.total_cycles <= 1 {
                    raw // Bootstrap: use raw on first cycle
                } else {
                    prev * 0.8 + raw * 0.2
                }
            },
            ..Default::default()
        };

        // Store smoothed epistemic uncertainty for next cycle's EMA
        self.carryover.quality.smoothed_epistemic_uncertainty =
            metadata.smoothed_epistemic_uncertainty;

        // ── Social coherence telemetry ──
        metadata.social_trust_current = self.social_mgr.social.social_trust;
        metadata.social_cooperation_current = self.social_mgr.social.social_cooperation_rate;
        metadata.social_strategy_bias_applied = perception.strategy.social_strategy_bias;
        metadata.social_learning_rate_factor = feedback.social_learning_rate_factor;
        metadata.social_prediction_accuracy = self.social_mgr.social.social_prediction_accuracy;
        metadata.social_models_count = self.social_mgr.social.social_models_count;
        metadata.social_mean_trust = self.social_mgr.social.social_mean_trust;
        metadata.tom_prediction_mismatch = self.stats.tom_prediction_mismatch_ema;
        metadata.tom_exploration_triggered =
            self.stats.tom_prediction_mismatch_ema > 0.4 && self.stats.total_cycles > 10;

        // ── MCE factor telemetry (from consciousness carryover cache) ──
        metadata.mce_bottleneck = self.carryover.consciousness.mce_bottleneck_name.clone();
        metadata.mce_softmin = self.carryover.consciousness.mce_softmin;
        metadata.mce_weighted_sum = self.carryover.consciousness.mce_weighted_sum;
        metadata.mce_narrative = self.carryover.consciousness.mce_narrative;
        metadata.mce_social = self.carryover.consciousness.mce_social;

        // ── Session 9: Advanced feedback intelligence telemetry (part 1) ──
        metadata.pe_variance = {
            let v = self.stats.avg_prediction_error_sq
                - self.stats.avg_prediction_error * self.stats.avg_prediction_error;
            v.max(0.0)
        };
        metadata.feedback_frozen = self.carryover.quality.consecutive_full_dampen >= 3;
        metadata.compound_instability = feedback.quality.cross_module_agreement < 0.5
            && perception.urgency.error_slope > 0.02
            && self.stats.total_cycles > 30;
        metadata.flow_feedback_relaxed = self.flow_state.in_flow && self.flow_state.intensity > 0.5;
        metadata.homeostasis_efficiency = self.carryover.quality.homeostasis_efficiency;
        // Session 10 telemetry (Session 11: lr_frozen from dynamics phase)
        metadata.confidence_crash_detected = dynamics.confidence_crash_detected;
        metadata.crash_freeze_remaining = self.carryover.quality.crash_freeze_remaining;
        metadata.lr_frozen = dynamics.lr_frozen;
        metadata.hysteresis_factor = self.carryover.quality.hysteresis_factor;
        metadata.agreement_confidence_coupling = feedback.quality.agreement_confidence_coupling;

        // Session 11 Item 8: Proposal conflict ratio → epistemic exploration boost.
        // High conflict = subsystems disagree about direction → boost exploration.
        {
            let conflict = self.feedback_state.avg_conflict_ratio();
            metadata.proposal_conflict_ratio = conflict;
            if conflict > 0.3 && self.stats.total_cycles > 15 {
                self.feedback_state.exploration.propose(
                    "high_conflict",
                    super::feedback_state::FeedbackProposal::Add(0.02),
                );
                metadata.conflict_exploration_boost = true;
            }
        }

        // ── Session 12 telemetry ──
        metadata.epistemic_conflict_exploration = feedback.reasoning.epistemic_conflict_count > 2
            && self.stats.total_cycles > 20;
        metadata.phenomenal_fragmentation_recovery = feedback.self_model.phenomenal_fragmented
            && self.stats.total_cycles > 15;
        metadata.temporal_discontinuity_recovery = feedback.self_model.temporal_discontinuity
            && self.stats.total_cycles > 15;
        metadata.binding_attention_modulated = (feedback.self_model.cross_modal_binding_strength > 0.7
            || feedback.self_model.cross_modal_binding_strength < 0.3)
            && self.stats.total_cycles > 10;
        metadata.resonator_semantic_lr_mod = (dynamics.resonator.resonator_best_sim > 0.8
            || (dynamics.resonator.resonator_best_sim < 0.3 && dynamics.resonator.resonator_best_sim > 0.0))
            && self.stats.total_cycles > 10;

        // ── Session 13 telemetry ──
        metadata.fep_td_converged = self.carryover.quality.consecutive_low_td_error > 10
            && self.stats.total_cycles > 30;
        metadata.confidence_rising_dampen = dynamics.neuromod.confidence_velocity > 0.02
            && self.stats.total_cycles > 15;
        metadata.flow_lr_boost = self.flow_state.in_flow && self.flow_state.intensity > 0.5;
        metadata.fep_efficiency_boost = dynamics.fep.fep_accuracy > 0.5
            && dynamics.fep.fep_complexity < 0.5;
        metadata.attention_overload_threshold = dynamics.attention.attention_budget_exceeded
            && self.stats.attention_budget_exceeded_count > 1;
        metadata.quality_exploration_floor = self.carryover.quality.consecutive_high_quality > 10
            && self.stats.total_cycles > 30;

        // ── GWT handler telemetry ──
        metadata.gwt_memory_consolidation_requested = self
            .gwt_mgr
            .memory_flag
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        metadata.gwt_perception_broadcasts =
            self.gwt_mgr
                .perception_count
                .swap(0, std::sync::atomic::Ordering::Relaxed) as u32;

        // ── GWT-triggered memory consolidation (Dehaene & Changeux 2011) ──
        // When global workspace broadcasts, record current state for episodic
        // replay so broadcast-worthy content is preferentially consolidated.
        if metadata.gwt_memory_consolidation_requested {
            if let Some(ref mut dream) = self.dream_engine {
                let action: Vec<f32> = perception
                    .encoding
                    .encoding_result
                    .hdv
                    .values
                    .iter()
                    .take(32)
                    .copied()
                    .collect();
                dream.record_consolidation_event(
                    &perception.encoding.compressed_state,
                    action,
                    perception.urgency.prediction_error,
                );
            }
        }

        // ── Error slope → consolidation priority ──
        // Rao & Ballard (1999): rising errors signal model inadequacy; replay
        // recent states to strengthen representations before further degradation.
        if perception.urgency.error_slope > 0.03 && !metadata.gwt_memory_consolidation_requested {
            if let Some(ref mut dream) = self.dream_engine {
                let action: Vec<f32> = perception
                    .encoding
                    .encoding_result
                    .hdv
                    .values
                    .iter()
                    .take(32)
                    .copied()
                    .collect();
                dream.record_consolidation_event(
                    &perception.encoding.compressed_state,
                    action,
                    perception.urgency.prediction_error,
                );
            }
        }

        // ── Voice telemetry ──
        {
            let voice_summary = self.voice_coherence.voice.summary();
            metadata.voice_articulation_quality =
                self.voice_coherence.voice.smoothed_articulation();
            metadata.voice_rate_stability = self.voice_coherence.voice.rate_stability();
            metadata.voice_confidence = voice_summary.voice_confidence;
            metadata.voice_phi_adjustment = self.voice_coherence.voice.compute_phi_adjustment();
        }

        // ── Substrate & convergence telemetry ──
        metadata.substrate = self.substrate_manager.telemetry(&self.config);
        // Populate flat substrate fields for backward-compatible access
        metadata.substrate_transition = metadata.substrate.substrate_transition.clone();
        metadata.substrate_feasibility_raw = metadata.substrate.substrate_feasibility_raw;
        metadata.substrate_honest_confidence = metadata.substrate.substrate_honest_confidence;
        metadata.substrate_effective_feasibility =
            metadata.substrate.substrate_effective_feasibility;
        metadata.substrate_tau_factor = metadata.substrate.substrate_tau_factor;
        metadata.substrate_scale_pressure = metadata.substrate.substrate_scale_pressure;
        metadata.substrate_feasibility = metadata.substrate.substrate_effective_feasibility;

        // Integrity telemetry
        #[cfg(feature = "integrity")]
        {
            let status = &self.integrity_manager.status;
            metadata.integrity = super::IntegrityTelemetry {
                attestation_passed: status.attestation_passed,
                temporal_passed: status.temporal_passed,
                canaries_passed: status.canaries_passed,
                anomaly_count: status.anomalies.len(),
                has_critical: self.integrity_manager.has_critical_anomaly(),
                last_check_cycle: status.last_check_cycle,
            };
        }

        // Physics bridge telemetry
        #[cfg(feature = "physics-bridge")]
        {
            if let Some(ref mut physics) = self.physics_integration {
                let pt = physics.telemetry();
                let pareto = pt.pareto_context.as_ref();
                metadata.physics_bridge = Some(super::PhysicsBridgeTelemetry {
                    catalog_size: pt.catalog_size,
                    results_returned: pt.results_returned,
                    top_match: pt.top_match,
                    top_score: pt.top_score,
                    query_count: pt.query_count,
                    queried_this_cycle: pt.queried_this_cycle,
                    effective_interval: pt.effective_interval,
                    effective_blend_weight: pt.effective_blend_weight,
                    top_domain: pt.top_domain,
                    pareto_frontier_size: pareto.map(|p| p.frontier_size),
                    pareto_best_analogy: pareto.map(|p| p.best_analogy_score),
                });
            }
        }

        // Vision manifold telemetry
        #[cfg(feature = "vision-manifold")]
        if let Some(ref tel) = perception.vision_telemetry {
            metadata.vision = Some(super::VisionManifoldTelemetry {
                vision_active: true,
                prediction_error: tel.prediction_error,
                manifold_coherence: tel.manifold_coherence,
                attention_entropy: tel.attention_entropy,
                num_salient_patches: tel.num_salient_patches,
                frame_sequence: tel.frame_sequence,
                training_triggered: tel.training_triggered,
                scene_recognition_similarity: tel.scene_recognition_similarity,
                cross_manifold_prediction_error: perception.cross_manifold_prediction_error,
                encode_time_us: tel.encode_time_us,
                evolve_time_us: tel.evolve_time_us,
                vision_mean_surprise: perception.vision_mean_surprise,
                vision_horizon_errors: perception.vision_horizon_errors.clone(),
                scene_recognized: perception.scene_recognized,
            });
        }

        // Foveation bridge telemetry
        #[cfg(feature = "foveation")]
        {
            if let Some(ref fov_mutex) = self.foveation_manager {
                if let Ok(fov) = fov_mutex.lock() {
                    let ft = fov.telemetry();
                    metadata.foveation = Some(super::FoveationBridgeTelemetry {
                        pending_count: ft.pending_count,
                        in_flight_count: ft.in_flight_count,
                        ready_count: ft.ready_count,
                        total_dispatched: ft.total_dispatched,
                        total_completed: ft.total_completed,
                        avg_processing_time_us: ft.avg_processing_time_us as u64,
                        last_confidence: ft.last_confidence,
                        effective_surprise_threshold: fov.effective_surprise_threshold(),
                        effective_max_concurrent: fov.effective_max_concurrent(),
                        recognition_count: perception.foveation_recognition_count,
                        top_recognition_confidence: perception.foveation_top_confidence,
                        hv_binding_applied: perception.foveation_recognition_count > 0,
                        dynamics_coupling_triggered: perception.foveation_recognition_count >= 2
                            && perception.foveation_top_confidence > 0.6,
                    });
                }
            }
        }

        // Broca SSM language generation telemetry
        #[cfg(feature = "ssm_language")]
        {
            metadata.broca = self
                .broca_manager
                .as_ref()
                .map(|m| m.last_telemetry().clone());
        }

        metadata.weight_convergence_state = feedback.consciousness.convergence_state.clone();
        if feedback.consciousness.convergence_state == "Converged" && self.convergence_cycle == 0 {
            self.convergence_cycle = self.stats.total_cycles;
        }
        metadata.convergence_cycle = self.convergence_cycle;

        // ── Fragmentation warning ──
        {
            let topo = self.ethics_engine.moral_topology().last_summary();
            if topo.beta_0 > 1 {
                tracing::warn!(
                    target: "cognitive_loop::moral_topology",
                    beta_0 = topo.beta_0,
                    unity = %format!("{:.3}", topo.unity),
                    scenario_count = topo.scenario_count,
                    cycle = self.stats.total_cycles,
                    "Moral fragmentation: {} disjoint clusters",
                    topo.beta_0
                );
            }
        }

        // ── Nurture/attachment telemetry ──
        #[cfg(feature = "nurture")]
        {
            if let Some(ref nurture) = self.nurture_attachment {
                metadata.attachment_style = Some(format!("{:?}", nurture.style()));
                metadata.attachment_security = Some(nurture.security_score());
            }
        }

        // ── End-of-cycle stats ──
        self.run_end_of_cycle_stats(
            &mut metadata,
            dynamics.resonator.resonator_wm_primed,
            feedback.memory.resonator_promotions,
            feedback.memory.codebook_evictions,
            feedback.memory.codebook_diversity,
            dynamics.fep.fep_surprise,
            self.self_model_tier
                .self_reflection
                .get_thresholds()
                .surprise as f64,
            dynamics.neuromod.neuromod_attention_alloc,
            dynamics.neuromod.phasic_da_replay_boost,
            dynamics.neuromod.ne_reorienting_boost,
            dynamics.neuromod.ne_arousal_feedback,
            dynamics.neuromod.confidence_velocity,
            dynamics.neuromod.sht_crash_dip,
            dynamics.neuromod.exploration_sht_drain,
        );

        // Cross-manifold predictor: observe actual cognitive state for Hebbian learning
        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut pred) = self.cross_manifold_predictor {
            pred.observe_cognitive(&perception.encoding.encoding_result.hdv);
        }

        // Project 16,384D HDC to 32D for visualization
        let thought_vector = {
            debug_assert!(
                !perception.encoding.encoding_result.hdv.values.is_empty(),
                "HDV must not be empty for thought_vector projection"
            );
            let chunk_size = (perception.encoding.encoding_result.hdv.values.len() / 32).max(1);
            perception
                .encoding
                .encoding_result
                .hdv
                .values
                .chunks(chunk_size)
                .take(32)
                .map(|chunk: &[f32]| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect()
        };

        tracing::debug!(
            surprise = metadata.surprise_triggered,
            prefrontal_veto = metadata.prefrontal_veto,
            reasoning_confidence = metadata.reasoning_confidence,
            exploration = ?metadata.exploration_action,
            "Cycle metadata"
        );

        // ═══════════════════════════════════════════════════════════════════════
        // METRICS COLLECTION
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref metrics) = self.metrics_collector {
            metrics.set_phi(dynamics.core.unified_psi);
            metrics.set_coherence(dynamics.core.coherence as f64);
            metrics.set_consciousness_level(metadata.consciousness_level);
            metrics.track_execution(metadata.safety_blocked, false);

            // Wire module timings to MetricsRegistry for Prometheus exposure
            #[cfg(feature = "api_module")]
            crate::api::metrics::update_timing_metrics(
                crate::api::metrics::global(),
                &metadata.module_timings_us,
                metadata.cycle_duration_us,
            );
        }

        #[cfg(feature = "identity")]
        let signed_output = self.mfdi_bridge.sign_output(&dynamics.core.output).ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // Session 9 Item 3: Dominant source concentration → targeted dampening.
        // If one subsystem contributes >60% of all proposals, dampen ALL channels
        // by 20% toward cycle-start to prevent single-subsystem monopoly.
        // Dehaene (2014): GWT prevents single-module monopoly via ignition competition.
        let dominant_concentration = self.feedback_state.dominant_source_concentration();
        if dominant_concentration > 0.6 && self.feedback_state.total_proposals() > 4 {
            // Apply a mild Scale(0.97) to all channels from "anti_monopoly" source
            self.feedback_state.confidence.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(0.97),
            );
            self.feedback_state.learning_rate.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(0.97),
            );
            self.feedback_state.exploration.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(0.97),
            );
            self.feedback_state.threshold.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(0.97),
            );
        }

        // Session 9 telemetry (part 2 — after dominant_concentration computed)
        metadata.dominant_source_concentration = dominant_concentration;

        // Session 10 Item 6: Proposal diversity → consensus quality metric.
        // If fewer than 3 distinct sources after warmup, boost exploration to recruit more.
        // Science: Dehaene (2014) — GWT requires multi-source consensus for ignition.
        {
            use super::thresholds::{
                PROPOSAL_DIVERSITY_EXPLORATION_BOOST, PROPOSAL_DIVERSITY_MIN_SOURCES,
                PROPOSAL_DIVERSITY_WARMUP,
            };
            let source_count = self.feedback_state.distinct_source_count();
            metadata.proposal_source_count = source_count as u32;
            if source_count < PROPOSAL_DIVERSITY_MIN_SOURCES
                && self.stats.total_cycles > PROPOSAL_DIVERSITY_WARMUP
                && self.feedback_state.total_proposals() > 2
            {
                self.feedback_state.exploration.propose(
                    "low_diversity",
                    super::feedback_state::FeedbackProposal::Add(
                        PROPOSAL_DIVERSITY_EXPLORATION_BOOST as f64,
                    ),
                );
                metadata.low_diversity_boost = true;
            }
        }

        // ── Phase 2.2: End feedback proposal collection ──────────────────
        // Session 9: Pass dampening streak + flow state for adaptive integration.
        let feedback_consensus = self.feedback_state.end_cycle_ext(
            self.prediction_confidence,
            self.fep.lr_boost,
            self.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
            self.carryover.quality.consecutive_full_dampen,
            self.flow_state.in_flow,
            self.flow_state.intensity,
        );
        // Track dampening streak for next cycle
        if self.feedback_state.feedback_dampened_count == 4 {
            self.carryover.quality.consecutive_full_dampen += 1;
        } else {
            self.carryover.quality.consecutive_full_dampen = 0;
        }

        // Session 9 Item 8: Substrate tau → feedback integration rate.
        // Fast substrates (tau < 1.0) apply consensus more aggressively;
        // slow substrates (tau > 1.0) blend more gently with cycle-start values.
        let feedback_consensus = if (self.substrate_manager.tau_factor - 1.0).abs() > 0.05 {
            let tau = self.substrate_manager.tau_factor;
            // Integration strength: tau=0.5 → 100% consensus, tau=2.0 → 50% consensus
            let integration_rate = (1.0 / tau).clamp(0.5, 1.0) as f64;
            let cs = &self.feedback_state;
            super::feedback_state::ConsensusResult {
                consensus_confidence: cs.cycle_start_confidence() * (1.0 - integration_rate)
                    + feedback_consensus.consensus_confidence * integration_rate,
                consensus_lr: cs.cycle_start_lr() * (1.0 - integration_rate)
                    + feedback_consensus.consensus_lr * integration_rate,
                consensus_exploration: cs.cycle_start_exploration() * (1.0 - integration_rate)
                    + feedback_consensus.consensus_exploration * integration_rate,
                consensus_threshold: cs.cycle_start_threshold() * (1.0 - integration_rate)
                    + feedback_consensus.consensus_threshold * integration_rate,
            }
        } else {
            feedback_consensus
        };

        // Store consensus-smoothed values for application at the next cycle start.
        // Applied via helpers at next cycle start by `apply_pending_consensus`.
        self.feedback_state
            .store_consensus_for_next_cycle(&feedback_consensus);

        // ── Phase 2.3: Integrate subsystem outputs ─────────────
        let integrated = self.subsystem_collector.integrate();
        if integrated.n_contributors > 0 {
            metadata.subsystem_integration_contributors = integrated.n_contributors as u32;

            if integrated.confidence_delta != 0.0 {
                self.adjust_confidence("subsystem_managers", integrated.confidence_delta as f32);
            }
            if integrated.lr_modulation != 1.0 {
                self.scale_lr("subsystem_managers", integrated.lr_modulation as f32);
            }
            if integrated.exploration_delta != 0.0 {
                self.adjust_exploration("subsystem_managers", integrated.exploration_delta as f32);
            }
            if integrated.arousal_delta != 0.0 {
                self.emotion_contagion.arousal =
                    (self.emotion_contagion.arousal + integrated.arousal_delta).clamp(0.0, 1.0);
            }
            if integrated.valence_delta != 0.0 {
                self.emotion_contagion.valence =
                    (self.emotion_contagion.valence + integrated.valence_delta).clamp(-1.0, 1.0);
            }

            tracing::trace!("Phase C integration: {}", integrated);
        }

        CycleResult {
            output: dynamics.core.output.clone(),
            prediction_error: dynamics.core.prediction_error,
            peak_attention: perception.encoding.encoding_result.peak_attention,
            detected_primitives: perception
                .encoding
                .encoding_result
                .detected_primitives
                .clone(),
            learning_occurred: dynamics.core.learning_occurred,
            training_loss: dynamics.core.training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector,
            wisdom_hv: perception.encoding.hv16_cached,
            #[cfg(feature = "ssm_language")]
            language_output: self.last_broca_text.take(),
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn output_metadata_non_default() {
        let mut svc = make_service();
        let result = svc.cycle("metadata check");
        assert!(result.metadata.cycle_duration_us > 0);
        assert!(!result.metadata.selected_strategy.is_empty());
    }

    #[test]
    fn output_thalamic_depth_maps_correctly() {
        let mut svc = make_service();
        let result = svc.cycle("thalamic depth");
        let score = result.metadata.thalamic_depth_score;
        assert!(
            (score - 1.0).abs() < f32::EPSILON
                || (score - 0.5).abs() < f32::EPSILON
                || (score - 0.2).abs() < f32::EPSILON,
            "thalamic_depth_score should be 1.0, 0.5, or 0.2, got {score}"
        );
    }

    #[test]
    fn output_is_consolidating_populated() {
        let mut svc = make_service();
        let result = svc.cycle("consolidation check");
        let _ = result.metadata.is_consolidating;
    }

    #[test]
    fn output_module_timings_has_core_hdc_encode() {
        let mut svc = make_service();
        let result = svc.cycle("timing check");
        assert!(
            result.metadata.module_timings_us.core_hdc_encode > 0
                || result.metadata.module_timings_us.core_cfc_step > 0
        );
    }

    #[test]
    fn output_thought_vector_32d() {
        let mut svc = make_service();
        let result = svc.cycle("thought vector");
        assert_eq!(result.thought_vector.len(), 32);
        for (i, &v) in result.thought_vector.iter().enumerate() {
            assert!(v.is_finite(), "thought_vector[{i}] should be finite");
        }
    }

    #[test]
    fn output_circadian_phase_populated() {
        let mut svc = make_service();
        let result = svc.cycle("circadian check");
        assert!(!result.metadata.circadian_phase.is_empty());
    }

    #[test]
    fn test_convergence_cycle_captured_and_persists() {
        let mut svc = make_service();
        // Initially convergence_cycle should be 0
        let result = svc.cycle("convergence init");
        assert_eq!(
            result.metadata.convergence_cycle, 0,
            "convergence_cycle should start at 0"
        );

        // Run enough cycles to potentially reach convergence (steady input → weights stabilize)
        let mut first_convergence_cycle = 0usize;
        for i in 0..200 {
            let result = svc.cycle("steady input for convergence");
            if result.metadata.convergence_cycle > 0 && first_convergence_cycle == 0 {
                first_convergence_cycle = result.metadata.convergence_cycle;
            }
            // Once captured, it should persist
            if first_convergence_cycle > 0 {
                assert_eq!(
                    result.metadata.convergence_cycle, first_convergence_cycle,
                    "convergence_cycle should persist once set (cycle {i})"
                );
            }
        }
        // Note: convergence may or may not be reached in 200 cycles depending on
        // the dynamics. If it was reached, we verified persistence above.
        // The key invariant is: once set, it never changes.
    }
}
