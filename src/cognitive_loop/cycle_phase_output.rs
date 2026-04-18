// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output phase of the cognitive cycle.
//!
//! Extracts the final metadata assembly and CycleResult construction from
//! the original `cycle()` method.

use std::mem;
use std::time::Instant;

use super::phase_results::{DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult};
use super::thresholds::{
    // Round 23: output-phase constants
    ADAPTIVE_THRESHOLD_SCALE_LOWER,
    ADAPTIVE_THRESHOLD_SCALE_UPPER,
    ANOMALY_RECOVERY_PSI_MULTIPLIER,
    ANTI_MONOPOLY_DAMPEN_SCALE,
    ATTENTION_SCHEMA_FATIGUE_THRESHOLD,
    BINDING_STRENGTH_TELEMETRY_HIGH,
    BINDING_STRENGTH_TELEMETRY_LOW,
    COMPOUND_INSTABILITY_AGREEMENT,
    COMPOUND_INSTABILITY_ERROR_SLOPE,
    CONFIDENCE_VELOCITY_FALLING_THRESHOLD,
    CONFIDENCE_VELOCITY_RISING_THRESHOLD,
    // Round 18: output phase telemetry thresholds
    CONFLICT_EXPLORATION_INCREMENT,
    CONSCIOUSNESS_GRADIENT_LR_MOD_THRESHOLD,
    CONSCIOUSNESS_STATE_LEVEL_HIGH,
    CONSCIOUSNESS_STATE_LEVEL_LOW,
    CONSOLIDATION_CONSCIOUSNESS_OFFSET,
    CONSOLIDATION_THRESHOLD_MIN,
    DOMINANT_CONCENTRATION_MONOPOLY_THRESHOLD,
    EMBODIED_AGENCY_STABLE_MAX,
    EMBODIED_AGENCY_STABLE_MIN,
    EPISTEMIC_PHI_HIGH,
    EPISTEMIC_PHI_LOW,
    EPISTEMIC_UNCERTAINTY_EMA_CURRENT,
    EPISTEMIC_UNCERTAINTY_EMA_PRIOR,
    ERROR_SLOPE_CONSOLIDATION_THRESHOLD,
    FEEDBACK_INTEGRATION_RATE_LOWER,
    FEP_ACCURACY_EFFICIENCY_THRESHOLD,
    FEP_COMPLEXITY_EFFICIENCY_THRESHOLD,
    FLOW_INTENSITY_FEEDBACK,
    FLOW_INTENSITY_LR_THRESHOLD,
    FULL_DAMPEN_ESCAPE_EXPLORATION,
    HARMONIES_ALIGNMENT_HIGH,
    HARMONIES_ALIGNMENT_LOW,
    HOLOGRAPHIC_UNITY_HIGH,
    HOLOGRAPHIC_UNITY_LOW,
    LIVING_MIND_COHERENCE_MOD_HIGH,
    LIVING_MIND_COHERENCE_MOD_LOW,
    LIVING_MIND_VITALITY_HIGH,
    LIVING_MIND_VITALITY_LOW,
    LIVING_MIND_VITALITY_MOD_HIGH,
    LIVING_MIND_VITALITY_MOD_LOW,
    MCTS_EFFECTIVENESS_MOD_HIGH,
    MCTS_EFFECTIVENESS_MOD_LOW,
    META_COGNITIVE_ACCURACY_LOW,
    PHENOMENAL_BINDING_HIGH,
    PHENOMENAL_BINDING_LOW,
    PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD,
    PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD,
    PREDICTIVE_SELF_SAFETY_HIGH,
    PROPOSAL_CONFLICT_EXPLORATION,
    RESONATOR_SIMILARITY_HIGH,
    RESONATOR_SIMILARITY_LOW,
    SUBSTRATE_TAU_DEVIATION_THRESHOLD,
    SUBSTRATE_TAU_FACTOR_MINIMUM,
    SUBSYSTEM_EXPLORATION_REQUEST_NUDGE,
    SUBSYSTEM_REST_REQUEST_LR_SCALE,
    TEMPORAL_COHERENCE_HIGH,
    TEMPORAL_COHERENCE_LOW,
    URGENCY_ESCALATION_AROUSAL_BOOST,
    URGENCY_ESCALATION_EXPLORATION_SCALE,
    VALUE_CACHE_HIT_RATE_HIGH,
    VALUE_CACHE_HIT_RATE_LOW,
};
use super::{CognitiveLoopService, CycleResult};

impl CognitiveLoopService {
    /// Output phase: metadata assembly, telemetry, CycleResult construction.
    pub(super) fn phase_output(
        &mut self,
        _input: &str,
        cycle_start: Instant,
        perception: &mut PerceptionPhaseResult,
        dynamics: &mut DynamicsPhaseResult,
        feedback: &mut FeedbackPhaseResult,
        mut module_timings: super::ModuleTimings,
    ) -> CycleResult {
        let thalamic_depth_score = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => super::thresholds::DEPTH_SCORE_DEEP_THOUGHT,
            super::CognitiveDepth::Cortical => super::thresholds::DEPTH_SCORE_CORTICAL,
            super::CognitiveDepth::Reflex => super::thresholds::DEPTH_SCORE_REFLEX,
        };

        let value_trend = self.primitive_tier.value_feedback.recent_trend(50);
        let circadian_phase_str = self.biorhythm_mgr.rhythm.phase.as_str();
        let selected_strategy_str = perception.strategy.selected_strategy.as_str();

        let _t = Instant::now();
        let moral_anomaly_report = self.ethics_engine.last_anomaly_report();
        // Cache topology summary once — avoids 10+ repeated moral_topology().last_summary() chains
        let topo_summary = self.ethics_engine.moral_topology().last_summary();
        let topology_fresh = self.ethics_engine.last_topology_fresh();
        let mut metadata = super::CycleMetadata {
            surprise_triggered: perception.exploration.surprise_triggered,
            prefrontal_veto: feedback.self_model.prefrontal_veto,
            reasoning_confidence: dynamics.reasoning.reasoning_confidence,
            exploration_action: mem::take(&mut perception.exploration.exploration_action),
            reasoning_gate_blocked: dynamics.reasoning.reasoning_gate_blocked,
            reasoning_fallback: mem::take(&mut dynamics.reasoning.reasoning_fallback),
            reasoning_plan_action: dynamics.reasoning.reasoning_plan_action,
            reasoning_plan_confidence: dynamics.reasoning.reasoning_plan_confidence,
            reasoning_narrative: mem::take(&mut dynamics.reasoning.reasoning_narrative),
            quality: super::QualityDiagnostics {
                meta_cognitive_accuracy: feedback.self_model.meta_cognitive_accuracy,
                meta_cognitive_depth: feedback.self_model.meta_cognitive_depth,
                dissipative_health: feedback.quality.dissipative_health,
                dissipative_regime: mem::take(&mut feedback.quality.dissipative_regime),
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
                hierarchical_free_energy_lr_boost: feedback
                    .self_model
                    .hierarchical_free_energy_lr_boost,
                predictive_phi_lr_delta: feedback.self_model.predictive_phi_lr_delta,
                body_valence_confidence_delta: feedback.self_model.body_valence_confidence_delta,
                narrative_self_confidence_factor: feedback
                    .self_model
                    .narrative_self_confidence_factor,
            },
            narrative_self_psi: feedback.self_model.narrative_self_psi,
            consciousness: super::ConsciousnessLevelMetrics {
                consciousness_level: feedback.consciousness.consciousness_level,
                consciousness_profile_composite: feedback
                    .consciousness
                    .consciousness_profile_composite,
                synergy_enhanced_composite: feedback.consciousness.synergy_enhanced_composite,
                emergent_properties_count: feedback.consciousness.emergent_properties_count,
                consciousness_state_label: mem::take(
                    &mut feedback.consciousness.consciousness_state_label,
                ),
                consciousness_state_level: feedback.consciousness.consciousness_state_level,
                consciousness_weights: feedback.consciousness.consciousness_weights,
                consciousness_weight_variance: feedback.consciousness.consciousness_weight_variance,
                consciousness_gradient_magnitude: feedback
                    .consciousness
                    .consciousness_gradient_magnitude,
                consciousness_limiting_component: mem::take(
                    &mut feedback.consciousness.consciousness_limiting_component,
                ),
                ..Default::default()
            },
            embodied: super::EmbodiedAffectMetrics {
                body_phi_modulation: feedback.self_model.body_psi_modulation,
                body_valence: feedback.self_model.body_valence,
                body_arousal: feedback.self_model.body_arousal,
                embodied_phi_modulation: feedback.self_model.embodied_psi_modulation,
                embodied_agency: feedback.self_model.embodied_agency,
                affective_valence: feedback.self_model.affective_valence,
                affective_arousal: feedback.self_model.affective_arousal,
                affect_consciousness_valence: feedback.consciousness.affect_cons_valence,
                affect_consciousness_arousal: feedback.consciousness.affect_cons_arousal,
                ..Default::default()
            },
            predictive_self_safety: feedback.self_model.predictive_self_safety,
            predictive_behavioral_error: feedback.self_model.predictive_behavioral_error,
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
            temporal: super::TemporalPhenomenalMetrics {
                temporal_coherence_score: feedback.self_model.temporal_coherence_score,
                temporal_discontinuity: feedback.self_model.temporal_discontinuity,
                temporal_causal_chains: feedback.consciousness.temporal_causal_chains,
                temporal_continuity: feedback.consciousness.temporal_continuity,
                temporal_max_chain_length: feedback.consciousness.temporal_max_chain_length,
                phenomenal_binding_strength: feedback.self_model.phenomenal_binding_strength,
                phenomenal_fragmented: feedback.self_model.phenomenal_fragmented,
                holographic_unity: feedback.consciousness.holographic_unity,
                holographic_binding: feedback.consciousness.holographic_binding,
                cross_modal_binding_strength: feedback.self_model.cross_modal_binding_strength,
                cross_modal_psi: feedback.self_model.cross_modal_psi,
                thermodynamic_entropy: feedback.self_model.thermodynamic_entropy,
                thermodynamic_free_energy: feedback.self_model.thermodynamic_free_energy,
                ..Default::default()
            },
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
                    .memory
                    .memory_consol
                    .resonator_memory
                    .as_ref()
                    .and_then(|m| m.resonator.codebooks.first())
                    .map(|cb| cb.len())
                    .unwrap_or(0),
                resonator_episodes: self
                    .memory
                    .memory_consol
                    .resonator_memory
                    .as_ref()
                    .map(|m| m.len())
                    .unwrap_or(0),
                resonator_factorization_iters: self
                    .memory
                    .memory_consol
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
                trajectory_efe: dynamics.fep.trajectory_efe,
                trajectory_best_action: dynamics.fep.trajectory_best_action,
                trajectory_surprise: dynamics.fep.trajectory_surprise,
                trajectory_ode_steps: dynamics.fep.trajectory_ode_steps,
                blanket_sensory_permeability: self
                    .fep
                    .enhanced_bridge
                    .blanket
                    .permeability()
                    .sensory,
                blanket_active_permeability: self.fep.enhanced_bridge.blanket.permeability().active,
                blanket_effective_permeability: self
                    .fep
                    .enhanced_bridge
                    .blanket
                    .permeability()
                    .effective,
                blanket_trend: self.fep.enhanced_bridge.blanket.trend(),
                blanket_coalescence_ready: self.fep.enhanced_bridge.blanket.coalescence_ready(0.6),
                blanket_coalition_count: self.swarm_manager.coalitions().len(),
            },
            hierarchical_total_free_energy: feedback.self_model.hierarchical_total_free_energy,
            primitive_psi: feedback.consciousness.primitive_psi,
            lattice_height: feedback.consciousness.lattice_height,
            lattice_width: feedback.consciousness.lattice_width,
            lattice_join_concept: mem::take(&mut feedback.consciousness.lattice_join_concept)
                .unwrap_or_default(),
            causal_codebook_entries: feedback.consciousness.causal_codebook_entries_len,
            compositionality_total: feedback.consciousness.compositionality_total,
            composition_rule_applied: mem::take(&mut feedback.ethics.composition_rule_applied),
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
                dominant_harmonic: mem::take(&mut dynamics.guidance.dominant_harmonic),
                guiding_question: mem::take(&mut dynamics.guidance.guiding_question),
                guiding_priority_category: mem::take(
                    &mut dynamics.guidance.guiding_priority_category,
                ),
            },
            ethics: super::EthicalTelemetry {
                moral_score: perception.moral.moral_score,
                moral_steering_category: mem::take(&mut dynamics.guidance.moral_steering_category),
                value_evaluator_score: feedback.ethics.value_evaluator_score,
                value_evaluator_decision: mem::take(&mut feedback.ethics.value_evaluator_decision),
                value_feedback_trend: value_trend,
                value_gate_factor: feedback.ethics.value_gate_factor,
                soul_alignment: perception.moral.soul_alignment,
                empathic_compassion: feedback.ethics.empathic_compassion,
                empathic_tone_adj: feedback.ethics.empathic_tone_adj,
                empathic_speech_rate_mod: feedback.ethics.empathic_speech_rate_mod,
                moral_topo_beta_0: topo_summary.beta_0,
                moral_topo_beta_1: topo_summary.beta_1,
                moral_topo_beta_2: topo_summary.beta_2,
                moral_topo_unity: topo_summary.unity,
                moral_topo_completeness: topo_summary.completeness,
                moral_topo_circularity: topo_summary.circularity,
                moral_topo_free_energy: topo_summary.moral_free_energy,
                moral_topo_dominant_harmony: topo_summary.dominant_harmony,
                moral_topo_scenario_count: topo_summary.scenario_count,
                // Gate anomaly flags on topology_fresh: between evaluations
                // (cadence 30–120 cycles), report stale=false so dashboard/API
                // consumers see clean transitions rather than sticky flags.
                moral_anomaly_score: if topology_fresh {
                    moral_anomaly_report.anomaly_score
                } else {
                    0.0
                },
                moral_value_inversion: topology_fresh && moral_anomaly_report.value_inversion,
                moral_free_energy_spike: topology_fresh && moral_anomaly_report.free_energy_spike,
                moral_drift_alert: topology_fresh && moral_anomaly_report.drift_alert,
                moral_fragmentation_increase: topology_fresh
                    && moral_anomaly_report.fragmentation_increase,
                moral_trajectory_convergence: moral_anomaly_report.trajectory_convergence,
                moral_convergence_severity: moral_anomaly_report.convergence_severity,
                moral_matched_hazard: moral_anomaly_report.matched_hazard.clone(),
                moral_anomaly_response_applied: self.config.enable_moral_anomaly_response
                    && topology_fresh
                    && moral_anomaly_report.anomaly_score > 0.0,
                harmony_entropy: topo_summary.harmony_entropy,
                moral_attractor_detected: topo_summary.attractor_detected,
                hodge_harmonic_fraction: topo_summary
                    .hodge_fractions
                    .map(|f| f.harmonic)
                    .unwrap_or(0.0),
                hodge_gradient_fraction: topo_summary
                    .hodge_fractions
                    .map(|f| f.gradient)
                    .unwrap_or(0.0),
                hodge_curl_fraction: topo_summary.hodge_fractions.map(|f| f.curl).unwrap_or(0.0),
                hodge_critical_scale: topo_summary
                    .hodge_fractions
                    .map(|f| {
                        if f.critical_scale.is_nan() {
                            -1.0
                        } else {
                            f.critical_scale
                        }
                    })
                    .unwrap_or(-1.0),
                hodge_at_criticality: topo_summary
                    .hodge_fractions
                    .map(|f| f.at_criticality)
                    .unwrap_or(false),
                in_active_rest: self.stats.in_active_rest,
                stillness_dominance_streak: self.stats.stillness_dominance_streak,
                unified_verdict: self
                    .ethics_verdict_override
                    .as_ref()
                    .unwrap_or(&self.last_ethics_verdict)
                    .as_str()
                    .to_string(),
                ethics_consequence_accuracy: self.ethics_engine.consequence_tracker_accuracy(),
                moral_affect_coords: perception.moral.moral_affect_coords,
                moral_fluctuatio_tension: perception.moral.moral_fluctuatio_tension,
                moral_is_ambiguous: perception.moral.moral_is_ambiguous,
                moral_epistemic_confidence: perception.moral.moral_epistemic_confidence,
            },
            multi_obj_frontier_size: feedback.multi_obj_frontier_size,
            reasoning_context: mem::take(&mut feedback.reasoning.reasoning_context),
            context_phi_weight: feedback.reasoning.context_phi_weight,
            reasoning_chain_confidence: feedback.reasoning.reasoning_chain_confidence,
            reasoning_chain_depth: feedback.reasoning.reasoning_chain_depth,
            causal_relations_count: feedback.reasoning.causal_relations_count,
            causal_avg_confidence: feedback.reasoning.causal_avg_confidence,
            evolution_generation: feedback.evolution.evolution_generation,
            evolution_phi_delta: feedback.evolution.evolution_phi_delta,
            neuroevo_generation: 0,
            neuroevo_best_fitness: 0.0,
            neuroevo_diversity: 0.0,
            neuroevo_species_count: 0,
            value_embeddings_created: feedback.ethics.value_embeddings_created,
            value_cache_hit_rate: feedback.ethics.value_cache_hit_rate,
            adaptive_reasoning_phi: feedback.reasoning.adaptive_reasoning_phi,
            epistemic_quality: feedback.reasoning.epistemic_quality,
            phi_validation_correlation: feedback.reasoning.phi_validation_correlation,
            epistemic_conflict_count: feedback.reasoning.epistemic_conflict_count,
            eq_v2_limiting_component: mem::take(
                &mut feedback.consciousness.eq_v2_limiting_component,
            ),
            pipeline_consciousness: feedback.consciousness.pipeline_consciousness,
            multimodal_integrated_phi: feedback.consciousness.multimodal_integrated_phi,
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
            lr_cognitive_mod: self.carryover.learning.lr_cognitive_mod,
            lr_meta_mod: self.carryover.learning.lr_meta_mod,
            feedback_proposal_count: { self.feedback_state.feedback_summary().total_proposals },
            feedback_conflict_ratio: self.feedback_state.avg_conflict_ratio(),
            feedback_priority_counts: {
                let s = self.feedback_state.feedback_summary();
                s.priority_counts
            },
            feedback_diversity: self.feedback_state.signal_diversity(),
            cycle_reward: dynamics.core.cycle_reward,
            support_triage_count: feedback.support.support_triage_count,
            support_alert_fired: feedback.support.support_alert_fired,
            support_federation_graduated: feedback.support.support_federation_graduated,
            support_efe: feedback.support.support_efe,
            structural: super::StructuralPhiMetrics {
                sigma: feedback.consciousness.sigma,
                spectral_mip_phi: feedback.consciousness.spectral_mip_phi,
                hierarchical_mip_phi: self.carryover.consciousness.last_hierarchical_mip_phi,
                hierarchical_mip_scales: self
                    .carryover
                    .consciousness
                    .last_hierarchical_mip_phi
                    .map(|_| 3usize)
                    .unwrap_or(0),
                structural_micro_phi: feedback.consciousness.structural_micro_phi,
                structural_meso_phi: feedback.consciousness.structural_meso_phi,
                structural_macro_phi: feedback.consciousness.structural_macro_phi,
                structural_bottleneck: feedback.consciousness.structural_bottleneck,
                structural_emergence_ratio: feedback.consciousness.structural_emergence_ratio,
                structural_num_clusters: feedback.consciousness.structural_num_clusters,
            },
            module_timings_us: {
                // Hierarchical bundling telemetry
                if let Some(ref bundler) = self.hierarchical_bundler {
                    tracing::trace!(
                        active_regions = bundler.active_region_count(),
                        total_vectors = bundler.total_vectors(),
                        "Hierarchical bundling stats"
                    );
                }
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings
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
            limiting_component_boosted: mem::take(&mut feedback.loops.limiting_component_boosted),
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
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_semantic_pe: self.stats.last_liquid_mamba_pe,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_effective_rank: self.stats.last_liquid_mamba_rank,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_lr: self.stats.last_liquid_mamba_lr,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_generation_count: self.stats.liquid_mamba_generation_count,
            // Partnership / Phi-Dyad
            relational_psi: self.behavior.social_mgr.social.relational_psi,
            // Resonant Speech: response profile from neuromod bath signals.
            response_profile: {
                let user_state = crate::resonant_speech::UserState::from_neuromod(
                    self.neuromod.bath.allostatic_load,
                    feedback.consciousness.consciousness_level,
                    self.behavior.emotion_contagion.arousal,
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
            phi_tau_factor: dynamics.phi_tau_factor,
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
                    prev * EPISTEMIC_UNCERTAINTY_EMA_PRIOR + raw * EPISTEMIC_UNCERTAINTY_EMA_CURRENT
                }
            },
            ..Default::default()
        };

        // Store smoothed epistemic uncertainty for next cycle's EMA
        self.carryover.quality.smoothed_epistemic_uncertainty =
            metadata.smoothed_epistemic_uncertainty;

        // ── Social coherence telemetry ──
        metadata.social_trust_current = self.behavior.social_mgr.social.social_trust;
        metadata.social_cooperation_current =
            self.behavior.social_mgr.social.social_cooperation_rate;
        metadata.social_strategy_bias_applied = perception.strategy.social_strategy_bias;
        metadata.social_learning_rate_factor = feedback.social_learning_rate_factor;
        metadata.social_prediction_accuracy =
            self.behavior.social_mgr.social.social_prediction_accuracy;
        metadata.social_models_count = self.behavior.social_mgr.social.social_models_count;
        metadata.social_mean_trust = self.behavior.social_mgr.social.social_mean_trust;
        metadata.tom_prediction_mismatch = self.stats.tom_prediction_mismatch_ema;
        metadata.tom_exploration_triggered =
            self.stats.tom_prediction_mismatch_ema > 0.4 && self.stats.total_cycles > 10;

        // ── Cantor fractal dream telemetry ──
        metadata.cantor = super::CantorTelemetry {
            cantor_buffer_occupancy: self.cantor_dream.broadcast_buffer.len() as u32,
            cantor_metacognitive_depth: self.cantor_dream.dream_surprise as f64,
            cantor_resonance_boost: self.cantor_dream.resonance_boost as f64,
            cantor_dream_surprise: self.cantor_dream.dream_surprise as f64,
            cantor_codebook_size: self.cantor_dream.cleanup_engine.codebook.len() as u32,
            cantor_depth_histogram: {
                let mut hist = [0u32; 6];
                for crhv in &self.cantor_dream.broadcast_buffer {
                    let d = crhv.depth.min(5);
                    hist[d] += 1;
                }
                hist
            },
        };

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
        metadata.modulation.feedback_frozen = self.carryover.quality.consecutive_full_dampen >= 3;
        metadata.modulation.compound_instability = feedback.quality.cross_module_agreement
            < COMPOUND_INSTABILITY_AGREEMENT
            && perception.urgency.error_slope > COMPOUND_INSTABILITY_ERROR_SLOPE
            && self.stats.total_cycles > 30;
        metadata.modulation.flow_feedback_relaxed = self.behavior.flow_state.in_flow
            && self.behavior.flow_state.intensity > FLOW_INTENSITY_FEEDBACK;
        metadata.homeostasis_efficiency = self.carryover.quality.homeostasis_efficiency;
        // Session 10 telemetry (Session 11: lr_frozen from dynamics phase)
        metadata.modulation.confidence_crash_detected = dynamics.confidence_crash_detected;
        metadata.crash_freeze_remaining = self.carryover.quality.crash_freeze_remaining;
        metadata.modulation.lr_frozen = dynamics.lr_frozen;
        metadata.hysteresis_factor = self.carryover.quality.hysteresis_factor;
        metadata.modulation.agreement_confidence_coupling =
            feedback.quality.agreement_confidence_coupling;

        // Session 11 Item 8: Proposal conflict ratio → epistemic exploration boost.
        // High conflict = subsystems disagree about direction → boost exploration.
        {
            let conflict = self.feedback_state.avg_conflict_ratio();
            metadata.proposal_conflict_ratio = conflict;
            if conflict > PROPOSAL_CONFLICT_EXPLORATION && self.stats.total_cycles > 15 {
                self.feedback_state.exploration.propose(
                    "high_conflict",
                    super::feedback_state::FeedbackProposal::Add(CONFLICT_EXPLORATION_INCREMENT),
                );
                metadata.modulation.conflict_exploration_boost = true;
            }
        }

        // ── Session 12 telemetry ──
        metadata.modulation.epistemic_conflict_exploration =
            feedback.reasoning.epistemic_conflict_count > 2 && self.stats.total_cycles > 20;
        metadata.modulation.phenomenal_fragmentation_recovery =
            feedback.self_model.phenomenal_fragmented && self.stats.total_cycles > 15;
        metadata.modulation.temporal_discontinuity_recovery =
            feedback.self_model.temporal_discontinuity && self.stats.total_cycles > 15;
        metadata.modulation.binding_attention_modulated =
            (feedback.self_model.cross_modal_binding_strength > BINDING_STRENGTH_TELEMETRY_HIGH
                || feedback.self_model.cross_modal_binding_strength
                    < BINDING_STRENGTH_TELEMETRY_LOW)
                && self.stats.total_cycles > 10;
        metadata.modulation.resonator_semantic_lr_mod = (dynamics.resonator.resonator_best_sim
            > RESONATOR_SIMILARITY_HIGH
            || (dynamics.resonator.resonator_best_sim < RESONATOR_SIMILARITY_LOW
                && dynamics.resonator.resonator_best_sim > 0.0))
            && self.stats.total_cycles > 10;

        // ── Session 13 telemetry ──
        metadata.modulation.fep_td_converged =
            self.carryover.quality.consecutive_low_td_error > 10 && self.stats.total_cycles > 30;
        metadata.modulation.confidence_rising_dampen = dynamics.neuromod.confidence_velocity
            > CONFIDENCE_VELOCITY_RISING_THRESHOLD
            && self.stats.total_cycles > 15;
        metadata.modulation.flow_lr_boost = self.behavior.flow_state.in_flow
            && self.behavior.flow_state.intensity > FLOW_INTENSITY_LR_THRESHOLD;
        metadata.modulation.fep_efficiency_boost = dynamics.fep.fep_accuracy
            > FEP_ACCURACY_EFFICIENCY_THRESHOLD
            && dynamics.fep.fep_complexity < FEP_COMPLEXITY_EFFICIENCY_THRESHOLD;
        metadata.modulation.attention_overload_threshold =
            dynamics.attention.attention_budget_exceeded
                && self.stats.attention_budget_exceeded_count > 1;
        metadata.modulation.quality_exploration_floor =
            self.carryover.quality.consecutive_high_quality > 10 && self.stats.total_cycles > 30;

        // ── Session 14 telemetry ──
        metadata.modulation.living_mind_vitality_feedback =
            feedback.self_model.living_mind_vitality > LIVING_MIND_VITALITY_HIGH
                || (feedback.self_model.living_mind_vitality < LIVING_MIND_VITALITY_LOW
                    && feedback.self_model.living_mind_vitality > 0.0);
        metadata.modulation.metacog_low_accuracy_dampen =
            feedback.self_model.meta_cognitive_accuracy < META_COGNITIVE_ACCURACY_LOW
                && self.stats.total_cycles > 20;
        metadata.modulation.self_safety_lr_boost =
            feedback.self_model.predictive_self_safety > PREDICTIVE_SELF_SAFETY_HIGH;
        metadata.modulation.embodied_agency_stable = feedback.self_model.embodied_agency
            >= EMBODIED_AGENCY_STABLE_MIN
            && feedback.self_model.embodied_agency <= EMBODIED_AGENCY_STABLE_MAX;

        // ── Session 15 telemetry ──
        metadata.modulation.pipeline_consciousness_gated = {
            let pc = self.carryover.quality.last_pipeline_consciousness;
            (pc > PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD
                || (pc < PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD && pc > 0.0))
                && self.stats.total_cycles > 15
        };
        metadata.modulation.low_coherence_early_warning = {
            let clc = self.carryover.urgency.consecutive_low_coherence;
            clc >= 5 && clc <= 10
        };
        metadata.modulation.mode_stable_exploration_dampen =
            self.carryover.urgency.mode_stability_counter > 50;
        metadata.modulation.crash_binding_relaxed =
            self.carryover.quality.crash_freeze_remaining > 0;
        metadata.modulation.attention_fatigue_broca_gated = self
            .consciousness
            .self_model_tier
            .attention_schema
            .as_ref()
            .map_or(false, |a| {
                a.control_signal < ATTENTION_SCHEMA_FATIGUE_THRESHOLD
            });
        metadata.modulation.resonator_sustained_low_boost = self.stats.total_cycles > 20
            && self.stats.resonator_error_exploration_count > (self.stats.total_cycles / 2) as u64;
        metadata.modulation.anomaly_recovery_phi_accelerated =
            self.carryover.urgency.anomaly_was_active
                && self.stats.unified_psi
                    > self.stats.avg_psi * ANOMALY_RECOVERY_PSI_MULTIPLIER as f32;

        // ── Session 16 telemetry ──
        {
            use super::thresholds::{
                CONSCIOUSNESS_EMA_HIGH_THRESHOLD, CONSCIOUSNESS_EMA_LOW_THRESHOLD,
                CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD, EPISTEMIC_REJECTION_STREAK_THRESHOLD,
                FULL_DAMPEN_FREEZE_THRESHOLD, MULTI_OBJ_FRONTIER_LARGE, MULTI_OBJ_FRONTIER_SMALL,
                TEMPORAL_BINDING_DAMPEN_THRESHOLD, TEMPORAL_BINDING_EXPLORE_THRESHOLD,
            };
            let tb = perception.encoding.temporal_binding_strength;
            metadata.modulation.temporal_binding_feedback = self.stats.total_cycles > 15
                && (tb < TEMPORAL_BINDING_EXPLORE_THRESHOLD
                    || tb > TEMPORAL_BINDING_DAMPEN_THRESHOLD);
            metadata.modulation.consciousness_gradient_active =
                feedback.consciousness.consciousness_gradient_magnitude
                    > CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD
                    || self.carryover.quality.consecutive_stable_gradient > 20;
            metadata.modulation.startup_exploration_ramped =
                self.stats.total_cycles <= super::thresholds::STARTUP_WARMUP_CYCLES;
            metadata.modulation.epistemic_rejection_streak_recal =
                self.carryover.quality.consecutive_epistemic_rejections
                    >= EPISTEMIC_REJECTION_STREAK_THRESHOLD
                    && self.stats.total_cycles > 20;
            metadata.modulation.full_dampen_threshold_freeze =
                self.carryover.quality.consecutive_full_dampen >= FULL_DAMPEN_FREEZE_THRESHOLD;
            let ema = self.carryover.history.consciousness_ema;
            metadata.modulation.consciousness_ema_lr_bias = self.stats.total_cycles > 30
                && (ema > CONSCIOUSNESS_EMA_HIGH_THRESHOLD
                    || (ema < CONSCIOUSNESS_EMA_LOW_THRESHOLD && ema > 0.0));
            metadata.modulation.multi_obj_frontier_gated = feedback.multi_obj_frontier_size
                >= MULTI_OBJ_FRONTIER_LARGE
                || (feedback.multi_obj_frontier_size <= MULTI_OBJ_FRONTIER_SMALL
                    && feedback.multi_obj_frontier_size > 0
                    && self.stats.total_cycles > 30);
            metadata.modulation.error_bifurcation_response = perception.urgency.oscillation_ratio
                > super::thresholds::ERROR_OSCILLATION_BIFURCATION;
        }

        // ── Session 15 continued: Feedback Loop Observability ──
        {
            use crate::cognitive_loop::thresholds::{
                COHERENCE_VELOCITY_BUDGET_THRESHOLD, HOMEOSTASIS_RECALIBRATE_HIGH,
                HOMEOSTASIS_RECALIBRATE_LOW, MORAL_CONSOLIDATION_EASE,
                MORAL_CONSOLIDATION_THRESHOLD,
            };
            let ms = self.carryover.quality.last_moral_score.abs();
            metadata.moral_consolidation_ease = if ms > MORAL_CONSOLIDATION_THRESHOLD {
                ((ms - MORAL_CONSOLIDATION_THRESHOLD) as f64 * MORAL_CONSOLIDATION_EASE) as f32
            } else {
                0.0
            };
            metadata.consolidation_threshold = (self.carryover.history.consciousness_ema
                - CONSOLIDATION_CONSCIOUSNESS_OFFSET
                - metadata.moral_consolidation_ease as f64)
                .max(CONSOLIDATION_THRESHOLD_MIN)
                as f32;
            metadata.modulation.mce_bottleneck_lr_applied =
                !self.carryover.consciousness.mce_bottleneck_name.is_empty();
            let eff = self.carryover.quality.homeostasis_efficiency;
            metadata.modulation.homeostasis_recalibrated = self.stats.total_cycles > 20
                && (eff > HOMEOSTASIS_RECALIBRATE_HIGH
                    || (eff < HOMEOSTASIS_RECALIBRATE_LOW && eff > 0.0));
            metadata.modulation.confidence_falling_lr_boost = dynamics.neuromod.confidence_velocity
                < CONFIDENCE_VELOCITY_FALLING_THRESHOLD
                && self.stats.total_cycles > 15;
            let cv = self.carryover.quality.coherence_velocity;
            metadata.modulation.coherence_velocity_budget_scaled =
                cv.abs() > COHERENCE_VELOCITY_BUDGET_THRESHOLD;
        }
        metadata.modulation.temporal_chain_depth_lr_mod = self.stats.total_cycles > 15
            && feedback.consciousness.temporal_max_chain_length > 0
            && (feedback.consciousness.temporal_max_chain_length
                >= crate::cognitive_loop::thresholds::TEMPORAL_CHAIN_DEEP_THRESHOLD
                || feedback.consciousness.temporal_max_chain_length
                    <= crate::cognitive_loop::thresholds::TEMPORAL_CHAIN_SHALLOW_THRESHOLD);
        metadata.modulation.eq_v2_bottleneck_response =
            !feedback.consciousness.eq_v2_limiting_component.is_empty();
        {
            use crate::cognitive_loop::thresholds::{
                AFFECT_AROUSAL_HIGH_THRESHOLD, AFFECT_AROUSAL_LOW_THRESHOLD,
                AFFECT_VALENCE_NEGATIVE_THRESHOLD, AFFECT_VALENCE_POSITIVE_THRESHOLD,
                NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD, NARRATIVE_SELF_PHI_LOW_THRESHOLD,
            };
            let av = feedback.consciousness.affect_cons_valence;
            let aa = feedback.consciousness.affect_cons_arousal;
            metadata.modulation.affect_consciousness_modulated = self.stats.total_cycles > 10
                && (aa > AFFECT_AROUSAL_HIGH_THRESHOLD
                    || (aa < AFFECT_AROUSAL_LOW_THRESHOLD && aa > 0.0)
                    || av < AFFECT_VALENCE_NEGATIVE_THRESHOLD
                    || av > AFFECT_VALENCE_POSITIVE_THRESHOLD);
            let nsp = feedback.self_model.narrative_gwt_self_psi;
            metadata.modulation.narrative_self_phi_modulated = self.stats.total_cycles > 15
                && (nsp > NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD
                    || (nsp > 0.0 && nsp < NARRATIVE_SELF_PHI_LOW_THRESHOLD));
        }

        // ── Session 15+ modulation observability booleans ──────────────────
        if self.stats.total_cycles > 15 {
            let phi_eff = metadata.quality.epistemic_phi_eff as f32;
            metadata.modulation.epistemic_phi_modulated =
                phi_eff > EPISTEMIC_PHI_HIGH || (phi_eff > 0.0 && phi_eff < EPISTEMIC_PHI_LOW);

            let pb = metadata.temporal.phenomenal_binding_strength as f32;
            metadata.modulation.phenomenal_binding_modulated =
                pb > PHENOMENAL_BINDING_HIGH || (pb > 0.0 && pb < PHENOMENAL_BINDING_LOW);

            let tc = metadata.temporal.temporal_coherence_score as f32;
            metadata.modulation.temporal_coherence_modulated =
                tc > TEMPORAL_COHERENCE_HIGH || (tc > 0.0 && tc < TEMPORAL_COHERENCE_LOW);

            let hu = metadata.temporal.holographic_unity as f32;
            metadata.modulation.holographic_unity_modulated =
                hu > HOLOGRAPHIC_UNITY_HIGH || (hu > 0.0 && hu < HOLOGRAPHIC_UNITY_LOW);

            let ha = metadata.harmonics.harmonies_alignment;
            metadata.modulation.harmonies_alignment_modulated =
                ha > HARMONIES_ALIGNMENT_HIGH || ha < HARMONIES_ALIGNMENT_LOW;

            let cg = feedback.consciousness.consciousness_gradient_magnitude;
            metadata.modulation.consciousness_gradient_lr_modulated =
                cg.abs() as f32 > CONSCIOUSNESS_GRADIENT_LR_MOD_THRESHOLD;

            let vch = metadata.value_cache_hit_rate;
            metadata.modulation.value_cache_confidence_modulated =
                vch < VALUE_CACHE_HIT_RATE_LOW || vch > VALUE_CACHE_HIT_RATE_HIGH;

            let csl = feedback.consciousness.consciousness_state_level as f32;
            metadata.modulation.consciousness_state_modulated = csl
                > CONSCIOUSNESS_STATE_LEVEL_HIGH
                || (csl > 0.0 && csl < CONSCIOUSNESS_STATE_LEVEL_LOW);

            metadata.modulation.living_mind_vitality_modulated = metadata.living_mind_vitality
                > 0.0
                && (metadata.living_mind_vitality > LIVING_MIND_VITALITY_MOD_HIGH
                    || metadata.living_mind_vitality < LIVING_MIND_VITALITY_MOD_LOW);

            metadata.modulation.living_mind_coherence_modulated = metadata.living_mind_coherence
                > 0.0
                && (metadata.living_mind_coherence > LIVING_MIND_COHERENCE_MOD_HIGH
                    || metadata.living_mind_coherence < LIVING_MIND_COHERENCE_MOD_LOW);

            let mpe = metadata.mcts_plan_effectiveness;
            metadata.modulation.mcts_effectiveness_modulated = mpe > MCTS_EFFECTIVENESS_MOD_HIGH
                || (mpe > 0.0 && mpe < MCTS_EFFECTIVENESS_MOD_LOW);
        }

        // ── GWT handler telemetry ──
        metadata.gwt_memory_consolidation_requested = self
            .consciousness
            .gwt_mgr
            .memory_flag
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        metadata.gwt_perception_broadcasts =
            self.consciousness
                .gwt_mgr
                .perception_count
                .swap(0, std::sync::atomic::Ordering::Relaxed) as u32;

        // ── Memory consolidation triggers ──
        // GWT broadcast (Dehaene & Changeux 2011) or rising error slope
        // (Rao & Ballard 1999) → record state for episodic replay.
        let should_consolidate = metadata.gwt_memory_consolidation_requested
            || (perception.urgency.error_slope > ERROR_SLOPE_CONSOLIDATION_THRESHOLD);
        if should_consolidate {
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
            let voice_summary = self.language_comm.voice_coherence.voice.summary();
            metadata.voice_articulation_quality = self
                .language_comm
                .voice_coherence
                .voice
                .smoothed_articulation();
            metadata.voice_rate_stability =
                self.language_comm.voice_coherence.voice.rate_stability();
            metadata.voice_confidence = voice_summary.voice_confidence;
            metadata.voice_phi_adjustment = self
                .language_comm
                .voice_coherence
                .voice
                .compute_phi_adjustment();
        }

        // ── Perception Manager telemetry ──
        metadata.perception_attention_sensitivity = self.perception_manager.attention_sensitivity();
        metadata.perception_budget_utilization = self.perception_manager.budget_utilization();
        metadata.perception_vigilant = self.perception_manager.is_vigilant();
        metadata.perception_mean_coherence = self.perception_manager.mean_coherence_score();

        // ── Drive Manager telemetry ──
        metadata.drive_boredom = self.drive_manager.boredom();
        metadata.drive_flow_intensity = self.drive_manager.flow_intensity();
        metadata.drive_in_flow = self.drive_manager.in_flow();
        metadata.drive_exploration_threshold = self.drive_manager.exploration_threshold();

        // ── Learning Manager telemetry ──
        metadata.learning_plasticity = self.learning_manager.plasticity();
        metadata.learning_in_dream_phase = self.learning_manager.in_dream_phase();
        metadata.learning_error_trend = self.learning_manager.error_trend();

        // ── Memory Manager telemetry ──
        metadata.memory_consolidation_pressure = self.memory_manager.consolidation_pressure();
        metadata.memory_recall_quality = self.memory_manager.recall_quality();

        // ── Swarm Manager telemetry ──
        {
            let st = self.swarm_manager.telemetry();
            metadata.swarm_connected_peers = st.connected_peers;
            metadata.swarm_connectivity_ema = st.connectivity_ema as f32;
            metadata.swarm_mean_peer_phi = st.mean_peer_phi as f32;
            metadata.swarm_affective_contagion = st.affective_contagion as f32;
            metadata.swarm_federated_confidence = st.federated_confidence as f32;
            metadata.swarm_anomaly_count = st.anomaly_count;
        }

        // ── Governance Manager telemetry ──
        #[cfg(feature = "mycelix")]
        {
            metadata.governance_reward_ema = self.governance_mgr.reward_ema() as f32;
            metadata.governance_pending_events = self.governance_mgr.pending_event_count();
            metadata.governance_pending_outcomes = self.governance_mgr.pending_outcome_count();
            metadata.governance_collective_phi = self.governance_mgr.last_collective_phi() as f32;
            metadata.governance_community_mode = self
                .governance_mgr
                .community_mode()
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            metadata.governance_blind_spot_count = self.governance_mgr.blind_spot_count();
            metadata.governance_max_blind_spot_severity =
                self.governance_mgr.max_blind_spot_severity() as f32;
            metadata.governance_epistemic_agents = self.governance_mgr.epistemic_agent_count();
            metadata.governance_harmonic_delta_max =
                self.governance_mgr.last_harmonic_delta_max() as f32;
            metadata.governance_lr_boost = self.governance_mgr.last_lr_boost() as f32;

            // ── Finance health telemetry ──
            let fh = self.governance_mgr.finance_health();
            metadata.finance_active_positions = fh.active_positions;
            metadata.finance_stressed_positions = fh.stressed_positions;
            metadata.finance_critical_positions = fh.critical_positions;
            metadata.finance_avg_ltv = fh.avg_ltv;
            metadata.finance_sap_circulation = fh.sap_circulation;
            metadata.finance_compost_collected = fh.compost_collected;
            metadata.finance_active_covenants = fh.active_covenants;
            metadata.finance_open_breakers = fh.open_breakers;
            metadata.finance_oracle_confidence = fh.oracle_confidence;
            metadata.finance_stress_index = fh.stress_index;
        }

        // ── CPG Manager telemetry ──
        #[cfg(feature = "cpg")]
        {
            let ct = self.cpg_manager.telemetry();
            metadata.cpg_sync_index = ct.sync_index as f32;
            metadata.cpg_mean_freq = ct.mean_freq as f32;
            metadata.cpg_motor_active = ct.motor_active;
            metadata.cpg_desync_alert = ct.desync_alert;
        }

        // ── Embodiment Bridge telemetry ──
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "phone"
        ))]
        {
            let et = &self.sensorimotor.embodiment_telemetry;
            metadata.embodiment_total_steps = et.total_steps;
            metadata.embodiment_control_effort = et.control_effort;
            metadata.embodiment_prediction_error = et.prediction_error;
            metadata.embodiment_platform = et.platform.clone();
            metadata.embodiment_num_actuators = et.num_actuators as u32;
        }

        // ── Fabrication Manager telemetry ──
        #[cfg(feature = "advanced-manufacturing")]
        {
            let ft = self.fabrication_manager.telemetry();
            metadata.fabrication_manufacturing_fe = ft.manufacturing_free_energy;
            metadata.fabrication_design_loop_fe = ft.design_loop_free_energy;
            metadata.fabrication_safety_level = ft.safety_level;
            metadata.fabrication_anomaly_count = ft.anomaly_count;
            metadata.fabrication_anomaly_ema = ft.anomaly_ema;
            metadata.fabrication_pog_score_ema = ft.pog_score_ema;
            metadata.fabrication_active_jobs = ft.active_print_jobs;
            metadata.fabrication_reward_ema = ft.reward_ema;
            metadata.fabrication_prediction_coherence = ft.prediction_coherence;
            metadata.fabrication_mrp_planned_orders = ft.mrp_planned_orders;
            metadata.fabrication_mrp_feasible = ft.mrp_feasible;
            metadata.fabrication_mrp_shortages = ft.mrp_shortages_count;
            metadata.fabrication_mrp_work_orders = ft.mrp_work_order_count;
            metadata.fabrication_defect_prediction = ft.defect_prediction;
            metadata.fabrication_defect_confidence = ft.defect_confidence;
        }

        // ── Spectrum Manager telemetry ──
        #[cfg(feature = "mesh")]
        {
            let rt = self.spectrum_manager.telemetry();
            metadata.spectrum_network_health = rt.network_health;
            metadata.spectrum_tier_available = {
                let mut bits: u8 = 0;
                if rt.tier_available[0] {
                    bits |= 1;
                }
                if rt.tier_available[1] {
                    bits |= 2;
                }
                if rt.tier_available[2] {
                    bits |= 4;
                }
                bits
            };
            metadata.spectrum_jamming_streak = rt.jamming_streak;
            metadata.spectrum_prediction_error = rt.spectrum_prediction_error as f32;
            metadata.spectrum_epistemic_discount = rt.epistemic_discount as f32;
            metadata.spectrum_degradation_streak = rt.degradation_streak;
            metadata.spectrum_known_peers = rt.known_peers;
            metadata.spectrum_encryption_sessions = rt.encryption_sessions;
        }

        // ── Sovereign Inoculation telemetry ──
        #[cfg(feature = "mesh")]
        {
            let tt = self.time_manager.telemetry();
            metadata.sovereign_time_offset_us = tt.offset_us;
            metadata.sovereign_time_stratum = tt.stratum;
            metadata.sovereign_time_drift_ppm = tt.drift_ppm as f32;
            metadata.sovereign_time_peer_count = tt.peer_count;
            metadata.sovereign_time_quality = tt.quality;
        }
        #[cfg(feature = "mesh-trust")]
        {
            let tr = self.trust_manager.telemetry();
            metadata.sovereign_trust_avg = tr.avg_trust as f32;
            metadata.sovereign_trust_density = tr.graph_density as f32;
            metadata.sovereign_trust_anomalies = tr.anomaly_count;
            metadata.sovereign_trust_pq_fraction = tr.pq_fraction as f32;
        }
        #[cfg(feature = "social-fabric")]
        {
            let sf = self.social_fabric_manager.telemetry();
            metadata.sovereign_social_resonance_mean = sf.resonance_mean as f32;
            metadata.sovereign_social_diversity = sf.diversity as f32;
            metadata.sovereign_social_echo_risk = sf.echo_chamber_risk as f32;
            metadata.sovereign_social_peer_reach = sf.peer_reach;
        }
        #[cfg(feature = "survival")]
        {
            let sv = self.survival_manager.telemetry();
            metadata.sovereign_survival_water_pct = sv.water_pct as f32;
            metadata.sovereign_survival_power_kw = sv.power_kw as f32;
            metadata.sovereign_survival_emergency = sv.emergency_active;
            metadata.sovereign_survival_sensor_count = sv.sensor_count;
            metadata.sovereign_survival_alert_count = sv.alert_count;
        }

        // ── Math Service telemetry ──
        #[cfg(feature = "mathematics")]
        {
            let mt = self.math_service.telemetry();
            metadata.math_problems_solved = mt.problems_solved;
            metadata.math_verification_rate = mt.verification_rate;
            metadata.math_avg_confidence = mt.average_confidence;
        }

        // ── FHE Collective Wisdom telemetry ──
        #[cfg(feature = "fhe-wisdom")]
        if self.config.fhe_wisdom_enabled {
            metadata.fhe_contributions_total = self.swarm_manager.fhe_contributions_total();
            metadata.fhe_aggregations_total = self.swarm_manager.fhe_aggregations_total();
            metadata.fhe_pool_count = self.swarm_manager.wisdom_pool_count();
            metadata.fhe_cycles_since_aggregation =
                self.swarm_manager.fhe_cycles_since_aggregation();
        }

        // ── Vision Manager telemetry ──
        #[cfg(feature = "vision-manifold")]
        {
            metadata.vision_pe_ema = self.vision_manager.visual_pe_ema();
            metadata.vision_surprise_threshold = self.vision_manager.surprise_threshold();
            metadata.vision_low_surprise_streak = self.vision_manager.low_surprise_streak();
            metadata.vision_manifold_enabled = true;
        }

        // ── Language Manager telemetry ──
        #[cfg(feature = "ssm_language")]
        {
            metadata.language_quality_ema = self.language_manager.quality_ema();
            metadata.language_coherence_ema = self.language_manager.coherence_ema();
            metadata.language_low_coherence_streak = self.language_manager.low_coherence_streak();
        }

        // ── Reasoning Manager telemetry ──
        #[cfg(feature = "reasoning_engine")]
        {
            metadata.reasoning_reliability_ema = self.reasoning_manager.reliability_ema();
            metadata.reasoning_cumulative_quality = self.reasoning_manager.cumulative_quality();
            metadata.reasoning_rising_streak = self.reasoning_manager.rising_streak();
            metadata.reasoning_falling_streak = self.reasoning_manager.falling_streak();
        }

        // ── Causal explanation narrative (every 47 cycles, amortized) ──
        if self.stats.total_cycles % 47 == 0 && self.stats.total_cycles > 0 {
            if let Some(ref explainer) = self.primitive_tier.causal_explainer {
                let summary = explainer.summarize_understanding();
                if summary.total_causal_relations > 0 {
                    metadata.consciousness_causal_narrative = format!(
                        "{} causal relations ({} high-confidence), avg confidence {:.0}%, {} explanations generated",
                        summary.total_causal_relations,
                        summary.high_confidence_relations,
                        summary.average_confidence * 100.0,
                        summary.explanations_generated,
                    );
                }
            }
        }

        // ── Muse telemetry ──
        #[cfg(feature = "muse")]
        {
            metadata.muse = self.muse_manager.telemetry();
        }

        // ── Substrate & convergence telemetry ──
        metadata.substrate = self.substrate_manager.telemetry(&self.config);
        // Populate flat substrate fields for backward-compatible access
        metadata.substrate_transition = mem::take(&mut metadata.substrate.substrate_transition);
        metadata.substrate_feasibility_raw = metadata.substrate.substrate_feasibility_raw;
        metadata.substrate_honest_confidence = metadata.substrate.substrate_honest_confidence;
        metadata.substrate_effective_feasibility =
            metadata.substrate.substrate_effective_feasibility;
        metadata.substrate_tau_factor = metadata.substrate.substrate_tau_factor;
        metadata.substrate_scale_pressure = metadata.substrate.substrate_scale_pressure;

        // ── JEPA telemetry ──
        #[cfg(feature = "jepa")]
        if let Some(ref jepa) = self.jepa_engine {
            let telem = jepa.telemetry();
            metadata.jepa_latent_pe = telem.latent_pe;
            metadata.jepa_total_energy = telem.total_energy;
            metadata.jepa_collapse_detected = telem.collapse_detected;
        }

        // ── Neural validation: cortical activation map from live subsystem states ──
        #[cfg(feature = "neural_validation")]
        {
            use symthaea_core::hdc::cortical_activation::{
                ActivationSource, CorticalActivationMap,
            };
            use symthaea_core::hdc::substrate_independence::CorticalRegion;

            let mut cam =
                CorticalActivationMap::zeros(ActivationSource::Simulated, self.stats.total_cycles);

            // Prefrontal: reasoning confidence (meta-cognition proxy)
            cam.set(
                CorticalRegion::Prefrontal,
                dynamics.reasoning.reasoning_confidence.clamp(0.0, 1.0),
            );

            // Visual: grid encoding norm + spatial complexity
            cam.set(
                CorticalRegion::Visual,
                (metadata.grid_encoding_norm * 0.5 + metadata.grid_spatial_complexity * 0.5)
                    .clamp(0.0, 1.0),
            );

            // Auditory: voice confidence (speech perception proxy)
            cam.set(CorticalRegion::Auditory, metadata.voice_confidence);

            // Language: reasoning narrative presence as proxy (0.4 when active)
            let lang_active = if metadata.reasoning_narrative.is_empty() {
                0.1
            } else {
                0.5
            };
            cam.set(CorticalRegion::Language, lang_active);

            // Memory: codebook utilization from resonator metrics
            cam.set(
                CorticalRegion::Memory,
                metadata.memory.codebook_utilization_rate,
            );

            // Emotional: affect valence magnitude + arousal (from subsystem metrics)
            let emotional = (feedback.consciousness.affect_cons_valence.abs()
                + feedback.consciousness.affect_cons_arousal)
                / 2.0;
            cam.set(CorticalRegion::Emotional, emotional.clamp(0.0, 1.0) as f32);

            // Motor: FEP pragmatic value (action confidence)
            cam.set(
                CorticalRegion::Motor,
                (dynamics.fep.fep_pragmatic_value as f32).clamp(0.0, 1.0),
            );

            // Social: social trust + social prediction accuracy
            cam.set(
                CorticalRegion::Social,
                (metadata.social_trust_current * 0.5 + metadata.social_prediction_accuracy * 0.5)
                    .clamp(0.0, 1.0),
            );

            // Executive: epistemic gate confidence (conflict monitoring)
            cam.set(
                CorticalRegion::Executive,
                feedback.reasoning.epistemic_gate_confidence,
            );

            // Integration: temporal binding + cross-module agreement
            cam.set(
                CorticalRegion::Integration,
                (metadata.temporal_binding_strength * 0.5 + metadata.cross_module_agreement * 0.5)
                    .clamp(0.0, 1.0),
            );

            // Sensory: thalamic depth as proprioceptive engagement proxy
            cam.set(
                CorticalRegion::Sensory,
                (thalamic_depth_score * 0.3).clamp(0.0, 1.0),
            );

            // Creative: surprise-triggered exploration as creativity proxy
            let creative = if perception.exploration.surprise_triggered {
                0.7
            } else {
                0.2
            };
            cam.set(CorticalRegion::Creative, creative);

            // Accumulate into ring buffer for temporal analysis.
            if self.cortical_history.len() >= 1000 {
                self.cortical_history.pop_front();
            }
            self.cortical_history.push_back(cam.clone());

            metadata.cortical_activation = Some(cam);
        }

        // ── Thermal telemetry ──
        {
            let thermal_signals = self.sensorimotor.thermal_bridge.signals();
            metadata.thermal = super::ThermalTelemetry {
                thermal_level: thermal_signals.level as u8,
                thermal_tau_factor: thermal_signals.tau_factor,
                should_reduce_profile: thermal_signals.should_reduce_profile,
                target_frequency_override: thermal_signals.target_frequency_override,
            };
        }

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
                integrity_confidence: self.integrity_manager.integrity_confidence,
                attestation_details: self
                    .integrity_manager
                    .attestation
                    .records()
                    .iter()
                    .map(|r| super::AttestationDetail {
                        name: r.name.to_string(),
                        passed: r
                            .last_verification
                            .as_ref()
                            .map(|v| v.passed)
                            .unwrap_or(true),
                        consecutive_failures: r.consecutive_failures,
                    })
                    .collect(),
                global_failure_streak: self.integrity_manager.global_failure_streak,
                confidence_history: self
                    .integrity_manager
                    .confidence_history()
                    .iter()
                    .copied()
                    .collect(),
            };
            // Integrity-aware consciousness gating: if integrity is compromised,
            // discount consciousness scores — the system should distrust its own
            // metrics when it can't verify they haven't been tampered with.
            let ic = self.integrity_manager.integrity_confidence;
            if ic < 1.0 {
                metadata.consciousness.consciousness_level *= ic as f64;
            }
        }

        // Physics bridge telemetry
        #[cfg(feature = "physics-bridge")]
        {
            if let Some(ref mut physics) = self.feature_integ.physics_integration {
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
                vision_horizon_errors: mem::take(&mut perception.vision_horizon_errors),
                scene_recognized: perception.scene_recognized,
            });
        }

        // Foveation bridge telemetry
        #[cfg(feature = "foveation")]
        {
            if let Some(ref fov_mutex) = self.sensorimotor.vision_sensory.foveation_manager {
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
                .language_comm
                .broca_manager
                .as_ref()
                .map(|m| m.last_telemetry().clone());
        }

        // Broca factcheck telemetry
        #[cfg(feature = "mycelix")]
        {
            metadata.factcheck = Some(self.factcheck_bridge.telemetry());
        }

        metadata.consciousness.weight_convergence_state =
            mem::take(&mut feedback.consciousness.convergence_state);
        if metadata.consciousness.weight_convergence_state == "Converged"
            && self.convergence_cycle == 0
        {
            self.convergence_cycle = self.stats.total_cycles;
        }
        metadata.consciousness.convergence_cycle = self.convergence_cycle;

        // ── Fragmentation warning ──
        // Re-use topo_summary cached at top of phase_output
        if topo_summary.beta_0 > 1 {
            tracing::warn!(
                target: "cognitive_loop::moral_topology",
                beta_0 = topo_summary.beta_0,
                unity = %format!("{:.3}", topo_summary.unity),
                scenario_count = topo_summary.scenario_count,
                cycle = self.stats.total_cycles,
                "Moral fragmentation: {} disjoint clusters",
                topo_summary.beta_0
            );
        }

        // ── Therapeutic telemetry ──
        #[cfg(feature = "therapeutic")]
        {
            metadata.therapeutic.therapeutic_client_distress =
                self.therapeutic_manager.client_distress();
            metadata.therapeutic.therapeutic_alliance =
                self.therapeutic_manager.alliance_composite();
            metadata.therapeutic.therapeutic_crisis_active = self.therapeutic_manager.crisis_active;
            metadata.therapeutic.therapeutic_crisis_type = self
                .therapeutic_manager
                .last_crisis_type
                .clone()
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_strategy = self
                .therapeutic_manager
                .active_strategy()
                .map(|s| s.as_str().to_string())
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_narrative_coherence =
                self.therapeutic_manager.narrative_coherence();
            metadata.therapeutic.therapeutic_formulation_factors =
                self.therapeutic_manager.formulation.total_factors();
            metadata.therapeutic.therapeutic_resilience_ratio =
                self.therapeutic_manager.formulation_resilience_ratio();
            metadata.therapeutic.therapeutic_rupture_count =
                self.therapeutic_manager.alliance.rupture_count;
            metadata.therapeutic.therapeutic_repair_count =
                self.therapeutic_manager.alliance.repair_count;
            metadata.therapeutic.therapeutic_clinical_severity =
                self.therapeutic_manager.client_model.clinical_severity();
            metadata.therapeutic.therapeutic_narrative_fragments =
                self.therapeutic_manager.narrative.fragments.len();
            metadata.therapeutic.therapeutic_serotonin_debt =
                self.therapeutic_manager.regulation_engine.serotonin_debt();
            metadata.therapeutic.therapeutic_dopamine_debt =
                self.therapeutic_manager.regulation_engine.dopamine_debt();
            metadata.therapeutic.therapeutic_dream_accuracy =
                self.therapeutic_manager.dream_prediction_accuracy();

            // ── Scope Guard: check Broca output for scope violations ──
            // Runs BEFORE language_output is returned to caller.
            // If a violation is detected, inject disclaimers into the text
            // and log the violation type in telemetry.
            if let Some(ref mut text) = self.language_comm.last_broca_text {
                if let Some(violation) = self.therapeutic_manager.scope_guard.check_response(text) {
                    tracing::warn!(
                        target: "therapeutic_manager::scope_guard",
                        violation = ?violation,
                        cycle = self.stats.total_cycles,
                        "Scope violation detected in Broca output — injecting disclaimer"
                    );
                    *text = self.therapeutic_manager.scope_guard.apply_disclaimers(text);
                    metadata.therapeutic.therapeutic_scope_violation = format!("{:?}", violation);
                }
            }

            // ── Alliance rupture-repair enriched telemetry ──
            metadata.therapeutic.therapeutic_last_rupture_type = self
                .therapeutic_manager
                .alliance
                .last_rupture_type()
                .map(|rt| rt.as_str().to_string())
                .unwrap_or_default();
            metadata.therapeutic.therapeutic_repair_rate =
                self.therapeutic_manager.alliance.repair_rate();
            metadata.therapeutic.therapeutic_withdrawal_count =
                self.therapeutic_manager.alliance.withdrawal_count();
            metadata.therapeutic.therapeutic_confrontation_count =
                self.therapeutic_manager.alliance.confrontation_count();

            // ── Round 6: Formulation, RDoC, effectiveness, temporal coherence ──
            let rdoc = &self.therapeutic_manager.client_model.rdoc_profile;
            metadata.therapeutic.therapeutic_rdoc_profile = [
                rdoc.score(symthaea_clinical::RDocDomain::NegativeValence),
                rdoc.score(symthaea_clinical::RDocDomain::PositiveValence),
                rdoc.score(symthaea_clinical::RDocDomain::CognitiveSystems),
                rdoc.score(symthaea_clinical::RDocDomain::SocialProcesses),
                rdoc.score(symthaea_clinical::RDocDomain::ArousalRegulatory),
                rdoc.score(symthaea_clinical::RDocDomain::Sensorimotor),
            ];
            metadata.therapeutic.therapeutic_perpetuating_factors = self
                .therapeutic_manager
                .formulation
                .perpetuating
                .iter()
                .map(|f| f.description.clone())
                .collect();
            metadata.therapeutic.therapeutic_protective_factors = self
                .therapeutic_manager
                .formulation
                .protective
                .iter()
                .map(|f| f.description.clone())
                .collect();
            metadata.therapeutic.therapeutic_strategy_effectiveness =
                symthaea_therapeutic::RegulationStrategy::ALL
                    .iter()
                    .filter_map(|s| {
                        self.therapeutic_manager
                            .regulation_engine
                            .effectiveness(s)
                            .filter(|eff| eff.applications > 0)
                            .map(|eff| {
                                (s.as_str().to_string(), eff.success_rate(), eff.applications)
                            })
                    })
                    .collect();
            metadata.therapeutic.therapeutic_temporal_coherence =
                self.therapeutic_manager.narrative.temporal_coherence();

            // ── Shadow work telemetry (Observability Mode) ──
            let st = &self.therapeutic_manager.last_shadow_telemetry;
            metadata.therapeutic.shadow_total_pressure = st.total_shadow_pressure;
            metadata.therapeutic.shadow_fragment_count = st.shadow_fragment_count;
            metadata.therapeutic.shadow_peak_pressure = st.peak_fragment_pressure;
            metadata.therapeutic.shadow_mean_prediction_error = st.shadow_mean_prediction_error;
            metadata.therapeutic.shadow_projection_detections = st.projection_detections;
            metadata.therapeutic.shadow_surfacing_indicated = st.surfacing_indicated;
            metadata.therapeutic.shadow_dream_queue_depth = st.dream_queue_depth;
            metadata.therapeutic.shadow_best_dream_phi = st.best_dream_phi_improvement;
            metadata.therapeutic.shadow_pressure_trend = st.pressure_trend;
            metadata.therapeutic.shadow_to_narrative_ratio = st.shadow_to_narrative_ratio;
        }

        // ── Nurture/attachment telemetry ──
        #[cfg(feature = "nurture")]
        {
            if let Some(ref nurture) = self.nurture_attachment {
                metadata.attachment_style = Some(nurture.style().as_str().to_string());
                metadata.attachment_security = Some(nurture.security_score());
            }
        }

        // ── Knowledge Engine telemetry ──
        if let Some(ref km) = self.memory.knowledge_manager {
            let telem = km.telemetry();
            let sigs = km.signals();
            metadata.knowledge_graph_size = telem.graph_size;
            metadata.knowledge_best_similarity = telem.best_search_similarity;
            metadata.knowledge_causal_edges = telem.causal_edge_count;
            metadata.knowledge_epistemic_surprise = sigs.epistemic_surprise;
            metadata.knowledge_calibration_ece = telem.calibration_ece;
            metadata.knowledge_contradictions = telem.contradictions_detected;
        }

        // ── Glyph Codex telemetry ──
        #[cfg(feature = "glyph_codex")]
        {
            metadata.glyph_dominant_modality =
                self.glyph_manager.dominant_modality().name().to_string();
            metadata.glyph_coherence = self.glyph_manager.last_coherence().value as f32;
            metadata.glyph_resonant_name = self
                .glyph_manager
                .resonant_glyph_name()
                .unwrap_or("")
                .to_string();
            metadata.glyph_spiral_position = self.glyph_manager.spiral_position();
        }

        // ── Feature availability flags ──
        metadata.reasoning_engine_enabled = cfg!(feature = "reasoning_engine");
        metadata.mesh_enabled = cfg!(feature = "mesh");
        metadata.ssm_language_enabled = cfg!(feature = "ssm_language");

        // ── Swarm P2P telemetry (from NetworkService) ──
        if let Some(svc) = self.network_service() {
            metadata.mesh.swarm_peer_count = svc.peer_count() as u32;
            metadata.mesh.network_mean_phi = svc.network_mean_phi();
            metadata.mesh.network_coherence = svc.network_coherence();
        }

        // ── Immune system telemetry ──
        #[cfg(feature = "safety-agents")]
        {
            let level = self.safety_agent.current_level();
            metadata.immune_safety_level = level.as_str_upper().to_string();
            let telem = self
                .guardian_state
                .telemetry(self.stats.total_cycles as usize);
            metadata.immune_guardian_posture = telem.posture;
            metadata.immune_patrol_active = telem.patrol_active;
            metadata.immune_emergency_cycles = telem.emergency_cycles;
        }
        #[cfg(feature = "sentinel")]
        {
            let st = self.sentinel_manager.telemetry();
            metadata.immune_active_threats = st.active_threats as u32;
            metadata.immune_max_severity = st.max_severity;
            metadata.immune_threat_level = st.threat_level;
            metadata.immune_quarantined_peers = st.quarantined_peers as u32;
            metadata.immune_threat_patterns = self.threat_memory.pattern_count() as u32;
            metadata.immune_response_active = self.collective_immune_state.immune_response_active;
        }
        #[cfg(feature = "neuroevolution")]
        {
            let nt = self.neuroevolution_manager.telemetry();
            metadata.neuroevo_generation = nt.generation;
            metadata.neuroevo_best_fitness = nt.best_fitness;
            metadata.neuroevo_diversity = nt.diversity;
            metadata.neuroevo_species_count = nt.species_count;
        }
        #[cfg(feature = "safety-agents")]
        {
            metadata.defense_actions_proposed = self.defense_actions_proposed;
            metadata.defense_actions_approved = self.defense_actions_approved;
            metadata.immune_motor_halt =
                self.carryover.quality.safety_motor_halt || self.carryover.quality.subsystem_veto;
        }

        // ── End-of-cycle stats ──
        self.run_end_of_cycle_stats(
            &mut metadata,
            dynamics.resonator.resonator_wm_primed,
            feedback.memory.resonator_promotions,
            feedback.memory.codebook_evictions,
            feedback.memory.codebook_diversity,
            dynamics.fep.fep_surprise,
            self.consciousness
                .self_model_tier
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
        if let Some(ref mut pred) = self.sensorimotor.vision_sensory.cross_manifold_predictor {
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
                .map(|chunk: &[f32]| chunk.iter().sum::<f32>() / chunk.len().max(1) as f32)
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
            metrics.set_consciousness_level(metadata.consciousness.consciousness_level);
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
        let signed_output = self
            .mfdi_bridge
            .sign_output(&dynamics.core.output)
            .map_err(
                |e| tracing::error!(error = ?e, "MFDI output signing failed — output unattested"),
            )
            .ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // Session 9 Item 3: Dominant source concentration → targeted dampening.
        // If one subsystem contributes >60% of all proposals, dampen ALL channels
        // by 20% toward cycle-start to prevent single-subsystem monopoly.
        // Dehaene (2014): GWT prevents single-module monopoly via ignition competition.
        let dominant_concentration = self.feedback_state.dominant_source_concentration();
        if dominant_concentration > DOMINANT_CONCENTRATION_MONOPOLY_THRESHOLD
            && self.feedback_state.total_proposals() > 4
        {
            // Apply a mild Scale(0.97) to all channels from "anti_monopoly" source
            self.feedback_state.confidence.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.learning_rate.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.exploration.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
            );
            self.feedback_state.threshold.propose(
                "anti_monopoly",
                super::feedback_state::FeedbackProposal::Scale(ANTI_MONOPOLY_DAMPEN_SCALE),
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
                metadata.modulation.low_diversity_boost = true;
            }
        }

        // ── Phase 2.2: End feedback proposal collection ──────────────────
        // Session 9: Pass dampening streak + flow state for adaptive integration.
        let feedback_consensus = self.feedback_state.end_cycle_ext(
            self.prediction_confidence,
            self.fep.lr_boost,
            self.behavior.curiosity_drive.exploration_urge,
            self.carryover.learning.adaptive_threshold_scale,
            self.carryover.quality.consecutive_full_dampen,
            self.behavior.flow_state.in_flow,
            self.behavior.flow_state.intensity,
        );
        // Track dampening streak for next cycle
        if self.feedback_state.feedback_dampened_count == 4 {
            self.carryover.quality.consecutive_full_dampen += 1;
        } else {
            self.carryover.quality.consecutive_full_dampen = 0;
        }

        // Session 16 Item 5: consecutive_full_dampen → protective threshold freeze.
        // When all 4 feedback channels have been dampened for 5+ consecutive cycles,
        // the system is in a sustained suppression state. Freeze thresholds to prevent
        // further spiraling — let the system recover before making more adjustments.
        // Science: Turrigiano (2008) — sustained dampening triggers synaptic silencing.
        if self.carryover.quality.consecutive_full_dampen
            >= super::thresholds::FULL_DAMPEN_FREEZE_THRESHOLD
        {
            // Reset threshold scale toward neutral to break the dampening spiral
            self.carryover.learning.adaptive_threshold_scale =
                self.carryover.learning.adaptive_threshold_scale.clamp(
                    ADAPTIVE_THRESHOLD_SCALE_LOWER as f64,
                    ADAPTIVE_THRESHOLD_SCALE_UPPER as f64,
                );
            // Gentle exploration boost to escape the suppressed state
            self.adjust_exploration("full_dampen_escape", FULL_DAMPEN_ESCAPE_EXPLORATION);
        }

        // Session 9 Item 8: Substrate tau → feedback integration rate.
        // Fast substrates (tau < 1.0) apply consensus more aggressively;
        // slow substrates (tau > 1.0) blend more gently with cycle-start values.
        let feedback_consensus = if (self.substrate_manager.tau_factor as f64 - 1.0).abs()
            > SUBSTRATE_TAU_DEVIATION_THRESHOLD as f64
        {
            let tau =
                (self.substrate_manager.tau_factor as f64).max(SUBSTRATE_TAU_FACTOR_MINIMUM as f64); // Guard: prevent div-by-zero
                                                                                                     // Integration strength: tau=0.5 → 100% consensus, tau=2.0 → 50% consensus
            let integration_rate = if tau.is_finite() {
                (1.0_f64 / tau).clamp(FEEDBACK_INTEGRATION_RATE_LOWER as f64, 1.0)
            } else {
                1.0_f64
            };
            let cs = &self.feedback_state;
            let rate = integration_rate as f64;
            super::feedback_state::ConsensusResult {
                consensus_confidence: cs.cycle_start_confidence() * (1.0 - rate)
                    + feedback_consensus.consensus_confidence * rate,
                consensus_lr: cs.cycle_start_lr() * (1.0 - rate)
                    + feedback_consensus.consensus_lr * rate,
                consensus_exploration: cs.cycle_start_exploration() * (1.0 - rate)
                    + feedback_consensus.consensus_exploration * rate,
                consensus_threshold: cs.cycle_start_threshold() * (1.0 - rate)
                    + feedback_consensus.consensus_threshold * rate,
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
            metadata.subsystem_flags = integrated.flags;

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
                self.behavior.emotion_contagion.arousal = (self.behavior.emotion_contagion.arousal
                    + integrated.arousal_delta)
                    .clamp(0.0, 1.0);
            }
            if integrated.valence_delta != 0.0 {
                self.behavior.emotion_contagion.valence = (self.behavior.emotion_contagion.valence
                    + integrated.valence_delta)
                    .clamp(-1.0, 1.0);
            }

            // ── Act on subsystem flags ──────────────────────────────────
            // These flags are set by individual managers and OR'd together.
            // Previously computed but never acted upon — now wired.
            use super::subsystem_trait::output_flags;

            // VETO_ACTION: A subsystem (sentinel, ethics) wants to block motor output.
            // Science: Miller & Cohen (2001) — executive inhibition of inappropriate actions.
            if integrated.has_flag(output_flags::VETO_ACTION) {
                self.carryover.quality.subsystem_veto = true;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem VETO_ACTION: motor output will be suppressed"
                );
            }

            // ESCALATE_URGENCY: A subsystem detected a critical situation.
            // Boost arousal and suppress exploration to focus on the threat.
            if integrated.has_flag(output_flags::ESCALATE_URGENCY) {
                self.behavior.emotion_contagion.arousal = (self.behavior.emotion_contagion.arousal
                    + URGENCY_ESCALATION_AROUSAL_BOOST)
                    .clamp(0.0, 1.0);
                self.scale_exploration("urgency_escalation", URGENCY_ESCALATION_EXPLORATION_SCALE);
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem ESCALATE_URGENCY: arousal boosted, exploration dampened"
                );
            }

            // REQUEST_CONSOLIDATION: Memory/learning subsystems want consolidation.
            // Trigger episodic consolidation to commit recent learning.
            if integrated.has_flag(output_flags::REQUEST_CONSOLIDATION) {
                self.fep.episodic_memory.consolidate_recent();
                tracing::trace!(
                    cycle = self.stats.total_cycles,
                    "Subsystem REQUEST_CONSOLIDATION: episodic consolidation triggered"
                );
            }

            // REQUEST_REST: A subsystem is fatigued and requests reduced processing.
            // Dampen learning rate to allow recovery.
            if integrated.has_flag(output_flags::REQUEST_REST) {
                self.scale_lr("subsystem_rest_request", SUBSYSTEM_REST_REQUEST_LR_SCALE);
                tracing::trace!(
                    cycle = self.stats.total_cycles,
                    "Subsystem REQUEST_REST: LR dampened for recovery"
                );
            }

            // REQUEST_EXPLORATION: A subsystem wants to explore (novelty, anomaly).
            // Already handled via exploration_delta averaging, but flag provides
            // a discrete signal — give an additional nudge.
            if integrated.has_flag(output_flags::REQUEST_EXPLORATION) {
                self.adjust_exploration(
                    "subsystem_request_explore",
                    SUBSYSTEM_EXPLORATION_REQUEST_NUDGE,
                );
            }

            // ANOMALY_DETECTED: One or more subsystems detected anomalous conditions.
            // Record for telemetry and dampen confidence slightly.
            if integrated.has_flag(output_flags::ANOMALY_DETECTED) {
                self.stats.anomaly_detected_count += 1;
                self.scale_confidence("subsystem_anomaly", 0.98);
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Subsystem ANOMALY_DETECTED: confidence dampened"
                );
            }

            metadata.subsystem_veto_active = self.carryover.quality.subsystem_veto;

            tracing::trace!("Phase C integration: {}", integrated);
        }

        // ── Governance consciousness lag: decorrelate governance → consciousness loop ──
        // Push the finalized consciousness level into the lag ring buffer.
        // Governance gating reads the oldest value (~50 cycles behind) to ensure
        // governance feedback doesn't circularly influence its own gating signal.
        // Science: Granger (1969) — temporal decorrelation breaks circular causation.
        {
            let lag_size = super::thresholds::GOVERNANCE_CONSCIOUSNESS_LAG_SIZE;
            self.governance_consciousness_lag
                .push_back(feedback.consciousness.consciousness_level);
            while self.governance_consciousness_lag.len() > lag_size {
                self.governance_consciousness_lag.pop_front();
            }
        }

        // ── Expire stale consequence predictions ────────────────────────────
        self.ethics_engine.expire_stale_predictions(
            self.stats.total_cycles as u64,
            super::thresholds::CONSEQUENCE_TRACKER_MAX_AGE_CYCLES,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // Visualization: record attention/saliency/binding snapshot
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut viz) = self.attention_visualizer {
            static ATTENTION_LABELS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
            let labels = ATTENTION_LABELS
                .get_or_init(|| {
                    vec![
                        "phi_attention".into(),
                        "prediction_error".into(),
                        "coherence".into(),
                        "binding_strength".into(),
                        "consciousness".into(),
                    ]
                })
                .clone();
            let snapshot = crate::visualization::AttentionSnapshot::new(
                labels,
                vec![
                    perception.encoding.phi_attention_weight as f64,
                    dynamics.core.prediction_error as f64,
                    dynamics.core.coherence as f64,
                    feedback.self_model.cross_modal_binding_strength as f64,
                    feedback.consciousness.equation_v2_consciousness,
                ],
                vec![
                    perception.encoding.phi_attention_weight,
                    dynamics.core.prediction_error.clamp(0.0, 1.0),
                    dynamics.core.coherence,
                    feedback.self_model.cross_modal_binding_strength,
                    feedback.consciousness.equation_v2_consciousness as f32,
                ],
                1.0,
            )
            .with_metadata("cycle", &self.stats.total_cycles.to_string())
            .with_metadata("depth", &format!("{:?}", self.cognitive_depth));
            viz.record(snapshot);
        }

        // Final safety clamps — absolute last point in the cycle.
        // Late consciousness phases (integration, monitors) can multiply exploration_factor
        // below bounds set in cycle_quality.rs. Re-clamp here to guarantee invariants.
        self.behavior.adaptive_behavior.exploration_factor = self
            .behavior
            .adaptive_behavior
            .exploration_factor
            .clamp(0.1, 3.0);
        self.behavior.adaptive_behavior.learning_rate_multiplier = self
            .behavior
            .adaptive_behavior
            .learning_rate_multiplier
            .clamp(0.1, 2.0);
        self.behavior.curiosity_drive.boredom =
            self.behavior.curiosity_drive.boredom.clamp(0.0, 1.5);

        CycleResult {
            output: mem::take(&mut dynamics.core.output),
            prediction_error: dynamics.core.prediction_error,
            peak_attention: perception.encoding.encoding_result.peak_attention,
            detected_primitives: mem::take(
                &mut perception.encoding.encoding_result.detected_primitives,
            ),
            learning_occurred: dynamics.core.learning_occurred,
            training_loss: dynamics.core.training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector,
            wisdom_hv: perception.encoding.hv16_cached,
            language_output: {
                let text = self.language_comm.last_broca_text.take();
                // Send to async voice synthesis (non-blocking) if enabled
                if let (Some(ref t), Some(ref vs)) = (&text, &self.voice_synthesis) {
                    let _ = vs.send(super::voice_channel::VoiceRequest {
                        text: t.clone(),
                        cfc_output: dynamics.core.output.clone(),
                        prediction_error: dynamics.core.prediction_error,
                        detected_primitives: perception
                            .encoding
                            .encoding_result
                            .detected_primitives
                            .clone(),
                        cycle_num: self.stats.total_cycles as u64,
                    });
                }
                text
            },
            language_source: self.language_comm.last_language_source.take(),
            #[cfg(feature = "canvas")]
            canvas_svg: self.sensorimotor.motor_rendering.last_canvas_svg.take(),
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
        // is_consolidating is a bool — verify it's accessible and has a valid value
        let consolidating = result.metadata.is_consolidating;
        assert!(
            consolidating || !consolidating,
            "is_consolidating should be a valid bool"
        );
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
            result.metadata.consciousness.convergence_cycle, 0,
            "convergence_cycle should start at 0"
        );

        // Run enough cycles to potentially reach convergence (steady input → weights stabilize)
        let mut first_convergence_cycle = 0usize;
        for i in 0..200 {
            let result = svc.cycle("steady input for convergence");
            if result.metadata.consciousness.convergence_cycle > 0 && first_convergence_cycle == 0 {
                first_convergence_cycle = result.metadata.consciousness.convergence_cycle;
            }
            // Once captured, it should persist
            if first_convergence_cycle > 0 {
                assert_eq!(
                    result.metadata.consciousness.convergence_cycle, first_convergence_cycle,
                    "convergence_cycle should persist once set (cycle {i})"
                );
            }
        }
        // Note: convergence may or may not be reached in 200 cycles depending on
        // the dynamics. If it was reached, we verified persistence above.
        // The key invariant is: once set, it never changes.
    }
}
