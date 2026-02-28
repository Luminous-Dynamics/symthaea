//! Output phase of the cognitive cycle.
//!
//! Extracts the final metadata assembly and CycleResult construction from
//! the original `cycle()` method.

use std::time::Instant;

use super::cycle::{DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult};
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
        let circadian_phase_str = self.biorhythm.phase.as_str();
        let selected_strategy_str = perception.selected_strategy.as_str();

        let _t = Instant::now();
        let mut metadata = super::CycleMetadata {
            surprise_triggered: perception.surprise_triggered,
            prefrontal_veto: feedback.prefrontal_veto,
            reasoning_confidence: dynamics.reasoning_confidence,
            exploration_action: perception.exploration_action.clone(),
            reasoning_gate_blocked: dynamics.reasoning_gate_blocked,
            reasoning_fallback: dynamics.reasoning_fallback.clone(),
            reasoning_plan_action: dynamics.reasoning_plan_action,
            reasoning_plan_confidence: dynamics.reasoning_plan_confidence,
            reasoning_narrative: dynamics.reasoning_narrative.clone(),
            meta_cognitive_accuracy: feedback.meta_cognitive_accuracy,
            meta_cognitive_depth: feedback.meta_cognitive_depth,
            narrative_self_psi: feedback.narrative_self_psi,
            body_phi_modulation: feedback.body_psi_modulation,
            body_valence: feedback.body_valence,
            body_arousal: feedback.body_arousal,
            consciousness_level: feedback.consciousness_level,
            predictive_self_safety: feedback.predictive_self_safety,
            attention_schema_focus: feedback.attention_schema_focus,
            gwt_broadcast: feedback.gwt_broadcast,
            resonance_frequency: feedback.resonance_frequency,
            quantum_coherence_level: feedback.quantum_coherence_level,
            temporal_coherence_score: feedback.temporal_coherence_score,
            temporal_discontinuity: feedback.temporal_discontinuity,
            embodied_phi_modulation: feedback.embodied_psi_modulation,
            embodied_agency: feedback.embodied_agency,
            narrative_gwt_veto: feedback.narrative_gwt_veto,
            narrative_gwt_self_psi: feedback.narrative_gwt_self_psi,
            living_mind_vitality: feedback.living_mind_vitality,
            living_mind_coherence: feedback.living_mind_coherence,
            urgency: perception.urgency,
            dream_insights: feedback.dream_insights,
            dream_phi_improvement: feedback.dream_phi_improvement,
            dream_wisdom_count: feedback.dream_wisdom_count,
            predictive_free_energy: feedback.predictive_free_energy,
            predictive_phi_modulation: feedback.predictive_psi_modulation,
            cross_modal_binding_strength: feedback.cross_modal_binding_strength,
            cross_modal_psi: feedback.cross_modal_psi,
            affective_valence: feedback.affective_valence,
            affective_arousal: feedback.affective_arousal,
            thermodynamic_entropy: feedback.thermodynamic_entropy,
            thermodynamic_free_energy: feedback.thermodynamic_free_energy,
            phenomenal_binding_strength: feedback.phenomenal_binding_strength,
            phenomenal_fragmented: feedback.phenomenal_fragmented,
            hierarchical_total_free_energy: feedback.hierarchical_total_free_energy,
            psi_attention_avg: feedback.psi_attention_avg,
            primitive_psi: feedback.primitive_psi,
            temporal_causal_chains: feedback.temporal_causal_chains,
            temporal_continuity: feedback.temporal_continuity,
            temporal_max_chain_length: feedback.temporal_max_chain_length,
            lattice_height: feedback.lattice_height,
            lattice_width: feedback.lattice_width,
            lattice_join_concept: feedback.lattice_join_concept.clone().unwrap_or_default(),
            causal_codebook_entries: feedback.causal_codebook_entries_len,
            continuity_replay_triggered: feedback.continuity_replay_needed,
            compositionality_total: feedback.compositionality_total,
            composition_rule_applied: feedback.composition_rule_applied.clone(),
            harmonies_alignment: feedback.harmonies_alignment,
            harmonies_approved: feedback.harmonies_approved,
            empathic_compassion: feedback.empathic_compassion,
            empathic_tone_adj: feedback.empathic_tone_adj,
            multi_obj_frontier_size: feedback.multi_obj_frontier_size,
            value_evaluator_score: feedback.value_evaluator_score,
            value_evaluator_decision: feedback.value_evaluator_decision.clone(),
            consciousness_profile_composite: feedback.consciousness_profile_composite,
            synergy_enhanced_composite: feedback.synergy_enhanced_composite,
            emergent_properties_count: feedback.emergent_properties_count,
            reasoning_context: feedback.reasoning_context.clone(),
            context_phi_weight: feedback.context_phi_weight,
            harmonic_field_coherence: feedback.harmonic_field_coherence,
            harmonic_love_resonance: feedback.harmonic_love_resonance,
            harmonic_interferences: feedback.harmonic_interferences,
            reasoning_chain_confidence: feedback.reasoning_chain_confidence,
            reasoning_chain_depth: feedback.reasoning_chain_depth,
            causal_relations_count: feedback.causal_relations_count,
            causal_avg_confidence: feedback.causal_avg_confidence,
            evolution_generation: feedback.evolution_generation,
            evolution_phi_delta: feedback.evolution_phi_delta,
            value_embeddings_created: feedback.value_embeddings_created,
            value_cache_hit_rate: feedback.value_cache_hit_rate,
            adaptive_reasoning_phi: feedback.adaptive_reasoning_phi,
            epistemic_quality: feedback.epistemic_quality,
            phi_validation_correlation: feedback.phi_validation_correlation,
            dissipative_health: feedback.dissipative_health,
            dissipative_regime: feedback.dissipative_regime.clone(),
            dissipative_entropy_rate: feedback.dissipative_entropy_rate,
            epistemic_phi_eff: feedback.epistemic_phi_eff,
            epistemic_conflict_count: feedback.epistemic_conflict_count,
            equation_v2_consciousness: feedback.equation_v2_consciousness,
            hierarchical_ltc_phi: feedback.hierarchical_ltc_phi,
            holographic_unity: feedback.holographic_unity,
            holographic_binding: feedback.holographic_binding,
            consciousness_gradient_magnitude: feedback.consciousness_gradient_magnitude,
            consciousness_limiting_component: feedback.consciousness_limiting_component.clone(),
            affect_consciousness_valence: feedback.affect_cons_valence,
            affect_consciousness_arousal: feedback.affect_cons_arousal,
            pipeline_consciousness: feedback.pipeline_consciousness,
            multimodal_integrated_phi: feedback.multimodal_integrated_phi,
            consciousness_state_label: feedback.consciousness_state_label.clone(),
            consciousness_state_level: feedback.consciousness_state_level,
            epistemic_gate_confidence: feedback.epistemic_gate_confidence,
            epistemic_gate_approved: feedback.epistemic_gate_approved,
            primitive_validation_phi_gain: feedback.primitive_validation_phi_gain,
            primitive_validation_p_value: feedback.primitive_validation_p_value,
            meta_reasoning_confidence: feedback.meta_reasoning_confidence,
            meta_reasoning_insights: feedback.meta_reasoning_insights,
            code_primitives_selected: feedback.code_primitives_selected.clone(),
            metacognitive_anomaly: dynamics.metacognitive_anomaly,
            safety_blocked: false,
            safety_category: None,
            negation_polarity: perception.negation_detected,
            moral_score: perception.moral_score,
            selected_strategy: selected_strategy_str.into(),
            actual_effective_lr: if dynamics.learning_occurred {
                dynamics.effective_lr
            } else {
                0.0
            },
            cycle_reward: dynamics.cycle_reward,
            fep_action: dynamics.fep_action_idx,
            value_feedback_trend: value_trend,
            support_triage_count: feedback.support_triage_count,
            support_alert_fired: feedback.support_alert_fired,
            support_federation_graduated: feedback.support_federation_graduated,
            support_efe: feedback.support_efe,
            soul_alignment: perception.soul_alignment,
            sigma: feedback.sigma,
            spectral_mip_phi: feedback.spectral_mip_phi,
            hierarchical_mip_phi: self.carryover.consciousness.last_hierarchical_mip_phi,
            hierarchical_mip_scales: self
                .carryover
                .consciousness
                .last_hierarchical_mip_phi
                .map(|_| 3usize)
                .unwrap_or(0),
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
            module_timings_us: {
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings.clone()
            },
            circadian_phase: circadian_phase_str.into(),
            circadian_plasticity: self.biorhythm.plasticity_mod as f32,
            phi_attention_weight: perception.phi_attention_weight,
            guiding_question: dynamics.guiding_question.clone(),
            dominant_harmonic: dynamics.dominant_harmonic.clone(),
            resonator_wm_primed: dynamics.resonator_wm_primed,
            resonator_reconsolidated: dynamics.resonator_reconsolidated,
            resonator_promotions: feedback.resonator_promotions,
            fep_pragmatic_value: dynamics.fep_pragmatic_value,
            fep_accuracy: dynamics.fep_accuracy,
            fep_complexity: dynamics.fep_complexity,
            fep_surprise: dynamics.fep_surprise,
            fep_td_error: dynamics.fep_td_error,
            resonator_best_sim: dynamics.resonator_best_sim,
            codebook_evictions: feedback.codebook_evictions,
            codebook_diversity: feedback.codebook_diversity,
            resonator_prediction_error: dynamics.resonator_prediction_error,
            cross_module_agreement: feedback.cross_module_agreement,
            thalamic_depth_score,
            epistemic_gate_gated: !feedback.epistemic_gate_approved,
            causal_attention_edges: dynamics.causal_attention_edges,
            mcts_plan_effectiveness: dynamics.mcts_plan_effectiveness,
            moral_steering_category: dynamics.moral_steering_category.clone(),
            codebook_utilization_rate: feedback.codebook_utilization_rate,
            surprise_replay_batch_size: feedback.surprise_replay_batch_size,
            attention_budget_exceeded: dynamics.attention_budget_exceeded,
            attention_budget_elapsed_us: dynamics.attention_budget_elapsed_us,
            prediction_coherence: dynamics.prediction_coherence,
            valence_homeostasis_pull: dynamics.valence_homeostasis_pull,
            arousal_homeostasis_pull: dynamics.arousal_homeostasis_pull,
            arousal_recovery_active: dynamics.arousal_recovery_active,
            arousal_recovery_tau_factor: dynamics.arousal_recovery_tau_factor,
            input_similarity: perception.input_similarity,
            input_memoized: perception.input_memoized,
            guiding_priority_category: dynamics.guiding_priority_category.clone(),
            cycle_duration_us: cycle_start.elapsed().as_micros() as u64,
            school_predicted_phi_gain: dynamics.school_predicted_phi_gain,
            epistemic_coherence_gated: feedback.epistemic_coherence_gated,
            unified_quality_score: feedback.unified_quality_score,
            dissipative_health_gated: feedback.dissipative_health_gated,
            dissipative_lr_factor: feedback.dissipative_lr_factor,
            phi_validation_cached: self.carryover.quality.phi_validation_correlation,
            phi_spectral_weight: feedback.phi_spectral_weight,
            coherence_velocity: feedback.coherence_velocity,
            coherence_velocity_gated: feedback.coherence_velocity_gated,
            anomaly_recovery_progress: dynamics.anomaly_recovery_progress,
            anomaly_recovering: dynamics.anomaly_recovering,
            error_pattern: perception.error_pattern.into(),
            startup_suppressed: perception.startup_suppressed,
            startup_warmup_progress: perception.startup_warmup_progress,
            self_model_accuracy: dynamics.self_model_accuracy,
            mode_confidence: self.carryover.urgency.mode_confidence,
            mode_stability_counter: self.carryover.urgency.mode_stability_counter,
            predicted_urgency: perception.predicted_urgency.into(),
            context_phi_applied: feedback.context_phi_applied,
            empathic_speech_rate_mod: feedback.empathic_speech_rate_mod,
            value_gate_factor: feedback.value_gate_factor,
            evolution_confidence_delta: feedback.evolution_confidence_delta,
            homeostasis_pull_strength: dynamics.homeostasis_pull_strength,
            prediction_coherence_urgency_bias: perception.prediction_coherence_urgency_bias,
            attention_budget_gated: feedback.attention_budget_gated,
            limiting_component_boosted: feedback.limiting_component_boosted.clone(),
            love_resonance_boost: feedback.love_resonance_boost,
            reasoning_chain_boosted: feedback.reasoning_chain_boosted,
            attention_shift_applied: self.stats.attention_shift,
            harmonic_interference_lr_mod: feedback.harmonic_interference_lr_mod,
            resonator_error_exploration_mod: dynamics.resonator_error_exploration_mod,
            binding_threshold_mod: dynamics.binding_threshold_mod,
            causal_urgency_gated: feedback.causal_urgency_gated,
            epistemic_semantic_lr_mod: dynamics.epistemic_semantic_lr_mod,
            predictive_budget_gated: dynamics.predictive_budget_gated,
            binding_confidence_mod: dynamics.binding_confidence_mod,
            discontinuity_streak: self.carryover.urgency.discontinuity_streak,
            epistemic_reasoning_accelerated: self.carryover.quality.last_epistemic_conflict_count
                > 5,
            agency_strategy_override: perception.agency_strategy_override,
            pfe_surprise_mod: dynamics.pfe_surprise_mod,
            adaptive_memo_threshold: perception.memo_threshold,
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
            ..Default::default()
        };

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
            dynamics.resonator_wm_primed,
            feedback.resonator_promotions,
            feedback.codebook_evictions,
            feedback.codebook_diversity,
            dynamics.fep_surprise,
            self.self_reflection.get_thresholds().surprise as f64,
            dynamics.neuromod_attention_alloc,
            dynamics.phasic_da_replay_boost,
            dynamics.ne_reorienting_boost,
            dynamics.ne_arousal_feedback,
            dynamics.confidence_velocity,
            dynamics.sht_crash_dip,
            dynamics.exploration_sht_drain,
        );

        // Project 16,384D HDC to 32D for visualization
        let thought_vector = {
            let chunk_size = perception.encoding_result.hdv.values.len() / 32;
            perception
                .encoding_result
                .hdv
                .values
                .chunks(chunk_size)
                .take(32)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
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
            metrics.set_phi(dynamics.unified_psi);
            metrics.set_coherence(dynamics.coherence as f64);
            metrics.set_consciousness_level(metadata.consciousness_level);
            metrics.track_execution(metadata.safety_blocked, false);
        }

        #[cfg(feature = "identity")]
        let signed_output = self.mfdi_bridge.sign_output(&dynamics.output).ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // ── Phase 2.2: End feedback proposal collection ──────────────────
        let feedback_divergence = self.feedback_state.end_cycle(
            self.prediction_confidence as f64,
            self.fep_lr_boost as f64,
            self.curiosity_drive.exploration_urge as f64,
            self.carryover.learning.adaptive_threshold_scale as f64,
        );

        // Store consensus-smoothed values for application at the next cycle start.
        // Routing through `store_consensus_for_next_cycle` + `apply_pending_consensus`
        // ensures the divergence tracker sees the writeback as a Set proposal rather
        // than an out-of-band mutation.
        if self.config.consensus_feedback {
            self.feedback_state
                .store_consensus_for_next_cycle(&feedback_divergence);
        }

        if feedback_divergence.confidence > 0.01
            || feedback_divergence.learning_rate > 0.01
            || feedback_divergence.exploration > 0.01
            || feedback_divergence.threshold > 0.01
        {
            tracing::trace!(
                conf_div = feedback_divergence.confidence,
                lr_div = feedback_divergence.learning_rate,
                explore_div = feedback_divergence.exploration,
                thresh_div = feedback_divergence.threshold,
                "Feedback divergence >1%"
            );
        }

        // ── Phase 2.3: Integrate subsystem outputs ─────────────
        let integrated = self.subsystem_collector.integrate();
        if integrated.n_contributors > 0 {
            metadata.subsystem_integration_contributors = integrated.n_contributors as u32;

            if integrated.confidence_delta != 0.0 {
                self.adjust_confidence(
                    "subsystem_managers",
                    integrated.confidence_delta as f32,
                );
            }
            if integrated.lr_modulation != 1.0 {
                self.scale_lr("subsystem_managers", integrated.lr_modulation as f32);
            }
            if integrated.exploration_delta != 0.0 {
                self.adjust_exploration(
                    "subsystem_managers",
                    integrated.exploration_delta as f32,
                );
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
            output: dynamics.output.clone(),
            prediction_error: dynamics.prediction_error,
            peak_attention: perception.encoding_result.peak_attention,
            detected_primitives: perception.encoding_result.detected_primitives.clone(),
            learning_occurred: dynamics.learning_occurred,
            training_loss: dynamics.training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector,
            wisdom_hv: perception.hv16_cached.clone(),
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
        }
    }
}
