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
        let moral_anomaly_report = self.ethics_engine.last_anomaly_report().clone();
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
            quality: super::QualityDiagnostics {
                meta_cognitive_accuracy: feedback.meta_cognitive_accuracy,
                meta_cognitive_depth: feedback.meta_cognitive_depth,
                dissipative_health: feedback.dissipative_health,
                dissipative_regime: feedback.dissipative_regime.clone(),
                dissipative_entropy_rate: feedback.dissipative_entropy_rate,
                epistemic_phi_eff: feedback.epistemic_phi_eff,
                equation_v2_consciousness: feedback.equation_v2_consciousness,
                hierarchical_ltc_phi: feedback.hierarchical_ltc_phi,
                unified_quality_score: feedback.unified_quality_score,
                dissipative_health_gated: feedback.dissipative_health_gated,
                dissipative_lr_factor: feedback.dissipative_lr_factor,
                coherence_velocity: feedback.coherence_velocity,
                coherence_velocity_gated: feedback.coherence_velocity_gated,
                anomaly_recovery_progress: dynamics.anomaly_recovery_progress,
                anomaly_recovering: dynamics.anomaly_recovering,
            },
            narrative_self_psi: feedback.narrative_self_psi,
            body_phi_modulation: feedback.body_psi_modulation,
            body_valence: feedback.body_valence,
            body_arousal: feedback.body_arousal,
            consciousness_level: feedback.consciousness_level,
            predictive_self_safety: feedback.predictive_self_safety,
            attention: super::AttentionMetrics {
                attention_schema_focus: feedback.attention_schema_focus,
                gwt_broadcast: feedback.gwt_broadcast,
                gwt_coalition_size: feedback.gwt_coalition_size,
                psi_attention_avg: feedback.psi_attention_avg,
                phi_attention_weight: perception.phi_attention_weight,
                attention_budget_exceeded: dynamics.attention_budget_exceeded,
                attention_budget_elapsed_us: dynamics.attention_budget_elapsed_us,
                input_similarity: perception.input_similarity,
                input_memoized: perception.input_memoized,
                attention_budget_gated: feedback.attention_budget_gated,
                attention_shift_applied: self.stats.attention_shift,
            },
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
            memory: super::MemoryResonatorMetrics {
                dream_insights: feedback.dream_insights,
                dream_phi_improvement: feedback.dream_phi_improvement,
                dream_wisdom_count: feedback.dream_wisdom_count,
                continuity_replay_triggered: feedback.continuity_replay_needed,
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
                resonator_wm_primed: dynamics.resonator_wm_primed,
                resonator_reconsolidated: dynamics.resonator_reconsolidated,
                resonator_promotions: feedback.resonator_promotions,
                resonator_best_sim: dynamics.resonator_best_sim,
                codebook_evictions: feedback.codebook_evictions,
                codebook_diversity: feedback.codebook_diversity,
                resonator_prediction_error: dynamics.resonator_prediction_error,
                codebook_utilization_rate: feedback.codebook_utilization_rate,
                surprise_replay_batch_size: feedback.surprise_replay_batch_size,
            },
            fep: super::FepTelemetry {
                fep_action: dynamics.fep_action_idx,
                fep_pragmatic_value: dynamics.fep_pragmatic_value,
                fep_accuracy: dynamics.fep_accuracy,
                fep_complexity: dynamics.fep_complexity,
                fep_surprise: dynamics.fep_surprise,
                fep_td_error: dynamics.fep_td_error,
                predictive_free_energy: feedback.predictive_free_energy,
                predictive_phi_modulation: feedback.predictive_psi_modulation,
            },
            cross_modal_binding_strength: feedback.cross_modal_binding_strength,
            cross_modal_psi: feedback.cross_modal_psi,
            affective_valence: feedback.affective_valence,
            affective_arousal: feedback.affective_arousal,
            thermodynamic_entropy: feedback.thermodynamic_entropy,
            thermodynamic_free_energy: feedback.thermodynamic_free_energy,
            phenomenal_binding_strength: feedback.phenomenal_binding_strength,
            phenomenal_fragmented: feedback.phenomenal_fragmented,
            hierarchical_total_free_energy: feedback.hierarchical_total_free_energy,
            primitive_psi: feedback.primitive_psi,
            temporal_causal_chains: feedback.temporal_causal_chains,
            temporal_continuity: feedback.temporal_continuity,
            temporal_max_chain_length: feedback.temporal_max_chain_length,
            lattice_height: feedback.lattice_height,
            lattice_width: feedback.lattice_width,
            lattice_join_concept: feedback.lattice_join_concept.clone().unwrap_or_default(),
            causal_codebook_entries: feedback.causal_codebook_entries_len,
            compositionality_total: feedback.compositionality_total,
            composition_rule_applied: feedback.composition_rule_applied.clone(),
            harmonics: super::HarmonicMetrics {
                harmonies_alignment: feedback.harmonies_alignment,
                harmonies_approved: feedback.harmonies_approved,
                harmonic_field_coherence: feedback.harmonic_field_coherence,
                harmonic_love_resonance: feedback.harmonic_love_resonance,
                harmonic_interferences: feedback.harmonic_interferences,
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
                dominant_harmonic: dynamics.dominant_harmonic.clone(),
                guiding_question: dynamics.guiding_question.clone(),
                guiding_priority_category: dynamics.guiding_priority_category.clone(),
            },
            ethics: super::EthicalTelemetry {
                moral_score: perception.moral_score,
                moral_steering_category: dynamics.moral_steering_category.clone(),
                value_evaluator_score: feedback.value_evaluator_score,
                value_evaluator_decision: feedback.value_evaluator_decision.clone(),
                value_feedback_trend: value_trend,
                value_gate_factor: feedback.value_gate_factor,
                soul_alignment: perception.soul_alignment,
                empathic_compassion: feedback.empathic_compassion,
                empathic_tone_adj: feedback.empathic_tone_adj,
                empathic_speech_rate_mod: feedback.empathic_speech_rate_mod,
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
                moral_anomaly_score: moral_anomaly_report.anomaly_score,
                moral_value_inversion: moral_anomaly_report.value_inversion,
                moral_free_energy_spike: moral_anomaly_report.free_energy_spike,
                moral_drift_alert: moral_anomaly_report.drift_alert,
                moral_fragmentation_increase: moral_anomaly_report.fragmentation_increase,
                moral_anomaly_response_applied: self.config.enable_moral_anomaly_response
                    && moral_anomaly_report.anomaly_score > 0.0,
            },
            multi_obj_frontier_size: feedback.multi_obj_frontier_size,
            consciousness_profile_composite: feedback.consciousness_profile_composite,
            synergy_enhanced_composite: feedback.synergy_enhanced_composite,
            emergent_properties_count: feedback.emergent_properties_count,
            reasoning_context: feedback.reasoning_context.clone(),
            context_phi_weight: feedback.context_phi_weight,
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
            epistemic_conflict_count: feedback.epistemic_conflict_count,
            holographic_unity: feedback.holographic_unity,
            holographic_binding: feedback.holographic_binding,
            consciousness_gradient_magnitude: feedback.consciousness_gradient_magnitude,
            consciousness_limiting_component: feedback.consciousness_limiting_component.clone(),
            eq_v2_limiting_component: feedback.eq_v2_limiting_component.clone(),
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
            selected_strategy: selected_strategy_str.into(),
            actual_effective_lr: if dynamics.learning_occurred {
                dynamics.effective_lr
            } else {
                0.0
            },
            cycle_reward: dynamics.cycle_reward,
            support_triage_count: feedback.support_triage_count,
            support_alert_fired: feedback.support_alert_fired,
            support_federation_graduated: feedback.support_federation_graduated,
            support_efe: feedback.support_efe,
            sigma: feedback.sigma,
            spectral_mip_phi: feedback.spectral_mip_phi,
            hierarchical_mip_phi: self.carryover.consciousness.last_hierarchical_mip_phi,
            hierarchical_mip_scales: self
                .carryover
                .consciousness
                .last_hierarchical_mip_phi
                .map(|_| 3usize)
                .unwrap_or(0),
            // Structural Phi decomposition
            structural_micro_phi: feedback.structural_micro_phi,
            structural_meso_phi: feedback.structural_meso_phi,
            structural_macro_phi: feedback.structural_macro_phi,
            structural_bottleneck: feedback.structural_bottleneck,
            structural_emergence_ratio: feedback.structural_emergence_ratio,
            structural_num_clusters: feedback.structural_num_clusters,
            // Dynamic consciousness weights
            consciousness_weights: feedback.consciousness_weights,
            consciousness_weight_variance: feedback.consciousness_weight_variance,
            module_timings_us: {
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings.clone()
            },
            circadian_phase: circadian_phase_str.into(),
            circadian_plasticity: self.biorhythm.plasticity_mod as f32,
            cross_module_agreement: feedback.cross_module_agreement,
            thalamic_depth_score,
            epistemic_gate_gated: !feedback.epistemic_gate_approved,
            causal_attention_edges: dynamics.causal_attention_edges,
            mcts_plan_effectiveness: dynamics.mcts_plan_effectiveness,
            prediction_coherence: dynamics.prediction_coherence,
            valence_homeostasis_pull: dynamics.valence_homeostasis_pull,
            arousal_homeostasis_pull: dynamics.arousal_homeostasis_pull,
            arousal_recovery_active: dynamics.arousal_recovery_active,
            arousal_recovery_tau_factor: dynamics.arousal_recovery_tau_factor,
            cycle_duration_us: cycle_start.elapsed().as_micros() as u64,
            school_predicted_phi_gain: dynamics.school_predicted_phi_gain,
            epistemic_coherence_gated: feedback.epistemic_coherence_gated,
            phi_validation_cached: self.carryover.quality.phi_validation_correlation,
            phi_spectral_weight: feedback.phi_spectral_weight,
            error_pattern: perception.error_pattern.into(),
            startup_suppressed: perception.startup_suppressed,
            startup_warmup_progress: perception.startup_warmup_progress,
            self_model_accuracy: dynamics.self_model_accuracy,
            mode_confidence: self.carryover.urgency.mode_confidence,
            mode_stability_counter: self.carryover.urgency.mode_stability_counter,
            predicted_urgency: perception.predicted_urgency.into(),
            context_phi_applied: feedback.context_phi_applied,
            evolution_confidence_delta: feedback.evolution_confidence_delta,
            homeostasis_pull_strength: dynamics.homeostasis_pull_strength,
            prediction_coherence_urgency_bias: perception.prediction_coherence_urgency_bias,
            limiting_component_boosted: feedback.limiting_component_boosted.clone(),
            love_resonance_boost: feedback.love_resonance_boost,
            reasoning_chain_boosted: feedback.reasoning_chain_boosted,
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
            // Partnership / Phi-Dyad
            relational_psi: self.social_coherence.social.relational_psi,
            // Resonant Speech: response profile from neuromod bath signals.
            response_profile: {
                let user_state = crate::resonant_speech::UserState::from_neuromod(
                    self.neuromod.bath.allostatic_load,
                    feedback.consciousness_level,
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
            ..Default::default()
        };

        // ── Substrate & convergence telemetry ──
        metadata.substrate = self.substrate_manager.telemetry();

        // Physics bridge telemetry
        #[cfg(feature = "physics-bridge")]
        {
            if let Some(ref mut physics) = self.physics_integration {
                let pt = physics.telemetry();
                metadata.physics_bridge = Some(super::types::telemetry::PhysicsBridgeTelemetry {
                    catalog_size: pt.catalog_size,
                    results_returned: pt.results_returned,
                    top_match: pt.top_match,
                    top_score: pt.top_score,
                    query_count: pt.query_count,
                    queried_this_cycle: pt.queried_this_cycle,
                });
            }
        }

        // Foveation bridge telemetry
        #[cfg(feature = "foveation")]
        {
            if let Some(ref fov) = self.foveation_manager {
                let ft = fov.telemetry();
                metadata.foveation =
                    Some(super::types::telemetry::FoveationBridgeTelemetry {
                        pending_count: ft.pending_count,
                        in_flight_count: ft.in_flight_count,
                        ready_count: ft.ready_count,
                        total_dispatched: ft.total_dispatched,
                        total_completed: ft.total_completed,
                        avg_processing_time_us: ft.avg_processing_time_us,
                        last_confidence: ft.last_confidence,
                        effective_surprise_threshold: fov.effective_surprise_threshold(),
                        effective_max_concurrent: fov.effective_max_concurrent(),
                    });
            }
        }

        metadata.weight_convergence_state = feedback.convergence_state.clone();
        if feedback.convergence_state == "Converged" && self.convergence_cycle == 0 {
            self.convergence_cycle = self.stats.total_cycles as usize;
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
            dynamics.resonator_wm_primed,
            feedback.resonator_promotions,
            feedback.codebook_evictions,
            feedback.codebook_diversity,
            dynamics.fep_surprise,
            self.self_model_tier
                .self_reflection
                .get_thresholds()
                .surprise as f64,
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
            debug_assert!(
                !perception.encoding_result.hdv.values.is_empty(),
                "HDV must not be empty for thought_vector projection"
            );
            let chunk_size = (perception.encoding_result.hdv.values.len() / 32).max(1);
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

            // Wire module timings to MetricsRegistry for Prometheus exposure
            #[cfg(feature = "api_module")]
            crate::api::metrics::update_timing_metrics(
                crate::api::metrics::global(),
                &metadata.module_timings_us,
                metadata.cycle_duration_us,
            );
        }

        #[cfg(feature = "identity")]
        let signed_output = self.mfdi_bridge.sign_output(&dynamics.output).ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // ── Phase 2.2: End feedback proposal collection ──────────────────
        let feedback_consensus = self.feedback_state.end_cycle(
            self.prediction_confidence,
            self.fep_lr_boost,
            self.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
        );

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
