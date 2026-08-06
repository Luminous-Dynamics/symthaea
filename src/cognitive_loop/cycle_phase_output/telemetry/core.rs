// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::prelude::*;

impl CognitiveLoopService {
    pub(in crate::cognitive_loop::cycle_phase_output) fn populate_core_telemetry(
        &self,
        metadata: &mut CycleMetadata,
        perception: &mut PerceptionPhaseResult,
        dynamics: &mut DynamicsPhaseResult,
        feedback: &mut FeedbackPhaseResult,
        circadian_phase_str: &str,
        selected_strategy_str: &str,
        value_trend: f32,
        topo_summary: &crate::hdc::moral_topology::MoralTopologySummary,
        topology_fresh: bool,
        moral_anomaly_report: &crate::hdc::moral_topology::MoralAnomalyReport,
        thalamic_depth_score: f32,
    ) {
        metadata.surprise_triggered = perception.exploration.surprise_triggered;
        metadata.prefrontal_veto = feedback.self_model.prefrontal_veto;
        metadata.reasoning_confidence = dynamics.reasoning.reasoning_confidence;
        metadata.exploration_action = mem::take(&mut perception.exploration.exploration_action);
        metadata.reasoning_gate_blocked = dynamics.reasoning.reasoning_gate_blocked;
        metadata.reasoning_gate_evaluated = dynamics.reasoning.re_gate_checks > 0;
        metadata.reasoning_fallback = mem::take(&mut dynamics.reasoning.reasoning_fallback);
        metadata.reasoning_plan_action = dynamics.reasoning.reasoning_plan_action;
        metadata.reasoning_plan_confidence = dynamics.reasoning.reasoning_plan_confidence;
        metadata.reasoning_narrative = mem::take(&mut dynamics.reasoning.reasoning_narrative);

        metadata.quality = super::super::super::QualityDiagnostics {
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
            narrative_self_confidence_factor: feedback.self_model.narrative_self_confidence_factor,
        };

        metadata.narrative_self_psi = feedback.self_model.narrative_self_psi;
        metadata.consciousness = super::super::super::ConsciousnessLevelMetrics {
            consciousness_level: feedback.consciousness.consciousness_level,
            consciousness_profile_composite: feedback.consciousness.consciousness_profile_composite,
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
        };

        metadata.embodied = super::super::super::EmbodiedAffectMetrics {
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
        };

        let allostatic_load = self.neuromod.bath.allostatic_load;
        let psi = feedback.consciousness.consciousness_level;
        let arousal = self.neuromod.bath.noradrenaline.effective();
        let oxytocin = self.neuromod.bath.oxytocin.effective();
        let user_state = crate::resonant_speech::UserState::from_neuromod(
            allostatic_load,
            psi,
            arousal,
            oxytocin,
        );

        metadata.response_profile = if user_state.needs_empathy() {
            "empathic".to_string()
        } else {
            match user_state.cognitive_load {
                crate::resonant_speech::CognitiveLoad::Low => "technical".to_string(),
                crate::resonant_speech::CognitiveLoad::Medium => "balanced".to_string(),
                crate::resonant_speech::CognitiveLoad::High
                | crate::resonant_speech::CognitiveLoad::Overloaded => "simplified".to_string(),
            }
        };

        metadata.predictive_self_safety = feedback.self_model.predictive_self_safety;
        metadata.predictive_behavioral_error = feedback.self_model.predictive_behavioral_error;
        metadata.attention = super::super::super::AttentionMetrics {
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
        };

        metadata.resonance_frequency = feedback.self_model.resonance_frequency;
        metadata.quantum_coherence_level = feedback.self_model.quantum_coherence_level;
        metadata.temporal = super::super::super::TemporalPhenomenalMetrics {
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
        };

        metadata.narrative_gwt_veto = feedback.self_model.narrative_gwt_veto;
        metadata.narrative_gwt_self_psi = feedback.self_model.narrative_gwt_self_psi;
        metadata.living_mind_vitality = feedback.self_model.living_mind_vitality;
        metadata.living_mind_coherence = feedback.self_model.living_mind_coherence;
        metadata.urgency = perception.urgency.urgency;

        metadata.memory = super::super::super::MemoryResonatorMetrics {
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
        };

        metadata.fep = super::super::super::FepTelemetry {
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
            blanket_sensory_permeability: self.fep.enhanced_bridge.blanket.permeability().sensory,
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
        };

        metadata.hierarchical_total_free_energy =
            feedback.self_model.hierarchical_total_free_energy;
        metadata.primitive_psi = feedback.consciousness.primitive_psi;
        metadata.lattice_height = feedback.consciousness.lattice_height;
        metadata.lattice_width = feedback.consciousness.lattice_width;
        metadata.lattice_join_concept =
            mem::take(&mut feedback.consciousness.lattice_join_concept).unwrap_or_default();
        metadata.causal_codebook_entries = feedback.consciousness.causal_codebook_entries_len;
        metadata.compositionality_total = feedback.consciousness.compositionality_total;
        metadata.composition_rule_applied =
            mem::take(&mut feedback.ethics.composition_rule_applied);

        metadata.harmonics = super::super::super::HarmonicMetrics {
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
            guiding_priority_category: mem::take(&mut dynamics.guidance.guiding_priority_category),
        };

        metadata.ethics = super::super::super::EthicalTelemetry {
            moral_score: perception.moral.moral_score,
            moral_concern_detected: perception.moral.moral_concern_detected,
            consent_violation: perception.moral.moral_judgment.consent_violation,
            moral_violations: perception.moral.moral_judgment.violations.clone(),
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
        };

        metadata.multi_obj_frontier_size = feedback.multi_obj_frontier_size;
        metadata.reasoning_context = mem::take(&mut feedback.reasoning.reasoning_context);
        metadata.context_phi_weight = feedback.reasoning.context_phi_weight;
        metadata.reasoning_chain_confidence = feedback.reasoning.reasoning_chain_confidence;
        metadata.reasoning_chain_depth = feedback.reasoning.reasoning_chain_depth;
        metadata.causal_relations_count = feedback.reasoning.causal_relations_count;
        metadata.causal_avg_confidence = feedback.reasoning.causal_avg_confidence;
        metadata.evolution_generation = feedback.evolution.evolution_generation;
        metadata.evolution_phi_delta = feedback.evolution.evolution_phi_delta;
        metadata.value_embeddings_created = feedback.ethics.value_embeddings_created;
        metadata.value_cache_hit_rate = feedback.ethics.value_cache_hit_rate;
        metadata.adaptive_reasoning_phi = feedback.reasoning.adaptive_reasoning_phi;
        metadata.epistemic_quality = feedback.reasoning.epistemic_quality;
        metadata.phi_validation_correlation = feedback.reasoning.phi_validation_correlation;
        metadata.epistemic_conflict_count = feedback.reasoning.epistemic_conflict_count;
        metadata.eq_v2_limiting_component =
            mem::take(&mut feedback.consciousness.eq_v2_limiting_component);
        metadata.pipeline_consciousness = feedback.consciousness.pipeline_consciousness;
        metadata.multimodal_integrated_phi = feedback.consciousness.multimodal_integrated_phi;
        metadata.epistemic_gate_confidence = feedback.reasoning.epistemic_gate_confidence;
        metadata.epistemic_gate_approved = feedback.reasoning.epistemic_gate_approved;
        metadata.primitive_validation_phi_gain = feedback.evolution.primitive_validation_phi_gain;
        metadata.primitive_validation_p_value = feedback.evolution.primitive_validation_p_value;
        metadata.meta_reasoning_confidence = feedback.reasoning.meta_reasoning_confidence;
        metadata.meta_reasoning_insights = feedback.reasoning.meta_reasoning_insights;
        metadata.code_primitives_selected = feedback.reasoning.code_primitives_selected;
        metadata.metacognitive_anomaly = dynamics.reasoning.metacognitive_anomaly;
        metadata.negation_polarity = perception.negation_detected;
        metadata.selected_strategy = selected_strategy_str.into();
        metadata.actual_effective_lr = if dynamics.core.learning_occurred {
            dynamics.core.effective_lr
        } else {
            0.0
        };
        metadata.lr_cognitive_mod = self.carryover.learning.lr_cognitive_mod;
        metadata.lr_meta_mod = self.carryover.learning.lr_meta_mod;
        metadata.feedback_proposal_count = self.feedback_state.feedback_summary().total_proposals;
        metadata.feedback_conflict_ratio = self.feedback_state.avg_conflict_ratio();
        metadata.feedback_priority_counts = self.feedback_state.feedback_summary().priority_counts;
        metadata.feedback_diversity = self.feedback_state.signal_diversity();
        metadata.cycle_reward = dynamics.core.cycle_reward;
        metadata.support_triage_count = feedback.support.support_triage_count;
        metadata.support_alert_fired = feedback.support.support_alert_fired;
        metadata.support_federation_graduated = feedback.support.support_federation_graduated;
        metadata.support_efe = feedback.support.support_efe;

        metadata.structural = super::super::super::StructuralPhiMetrics {
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
        };

        metadata.circadian_phase = circadian_phase_str.into();
        metadata.circadian_plasticity = self.biorhythm_mgr.rhythm.plasticity_mod as f32;
        metadata.cross_module_agreement = feedback.quality.cross_module_agreement;
        metadata.thalamic_depth_score = thalamic_depth_score;
        metadata.epistemic_gate_gated = !feedback.reasoning.epistemic_gate_approved;
        metadata.causal_attention_edges = dynamics.reasoning.causal_attention_edges;
        metadata.mcts_plan_effectiveness = dynamics.reasoning.mcts_plan_effectiveness;
        metadata.prediction_coherence = dynamics.core.prediction_coherence;
        metadata.valence_homeostasis_pull = dynamics.homeostasis.valence_homeostasis_pull;
        metadata.arousal_homeostasis_pull = dynamics.homeostasis.arousal_homeostasis_pull;
        metadata.arousal_recovery_active = dynamics.homeostasis.arousal_recovery_active;
        metadata.arousal_recovery_tau_factor = dynamics.homeostasis.arousal_recovery_tau_factor;
        metadata.school_predicted_phi_gain = dynamics.reasoning.school_predicted_phi_gain;
        metadata.epistemic_coherence_gated = feedback.loops.epistemic_coherence_gated;
        metadata.phi_validation_cached = self.carryover.quality.phi_validation_correlation;
        metadata.phi_spectral_weight = feedback.consciousness.phi_spectral_weight;
        metadata.error_pattern = perception.urgency.error_pattern.into();
        metadata.startup_suppressed = perception.startup_suppressed;
        metadata.startup_warmup_progress = perception.startup_warmup_progress;
        metadata.self_model_accuracy = dynamics.core.self_model_accuracy;
        metadata.mode_confidence = self.carryover.urgency.mode_confidence;
        metadata.mode_stability_counter = self.carryover.urgency.mode_stability_counter;
        metadata.predicted_urgency = perception.urgency.predicted_urgency.into();
        metadata.context_phi_applied = feedback.reasoning.context_phi_applied;
        metadata.evolution_confidence_delta = feedback.evolution.evolution_confidence_delta;
        metadata.homeostasis_pull_strength = dynamics.homeostasis.homeostasis_pull_strength;
        metadata.prediction_coherence_urgency_bias =
            perception.urgency.prediction_coherence_urgency_bias;
        metadata.limiting_component_boosted =
            mem::take(&mut feedback.loops.limiting_component_boosted);
        metadata.love_resonance_boost = feedback.loops.love_resonance_boost;
        metadata.reasoning_chain_boosted = feedback.loops.reasoning_chain_boosted;
        metadata.harmonic_interference_lr_mod = feedback.loops.harmonic_interference_lr_mod;
        metadata.resonator_error_exploration_mod =
            dynamics.resonator.resonator_error_exploration_mod;
        metadata.binding_threshold_mod = dynamics.binding_threshold_mod;
        metadata.causal_urgency_gated = feedback.loops.causal_urgency_gated;
        metadata.epistemic_semantic_lr_mod = dynamics.epistemic_semantic_lr_mod;
        metadata.predictive_budget_gated = dynamics.attention.predictive_budget_gated;
        metadata.binding_confidence_mod = dynamics.binding_confidence_mod;
        metadata.discontinuity_streak = self.carryover.urgency.discontinuity_streak;
        metadata.epistemic_reasoning_accelerated =
            self.carryover.quality.last_epistemic_conflict_count > 5;
        metadata.agency_strategy_override = perception.strategy.agency_strategy_override;
        metadata.pfe_surprise_mod = dynamics.pfe_surprise_mod;
        metadata.adaptive_memo_threshold = perception.encoding.memo_threshold;

        metadata.grid_encoding_norm = feedback.grid_encoding_norm;
        metadata.grid_spatial_complexity = feedback.grid_spatial_complexity;

        metadata.relational_psi = self.behavior.social_mgr.social.relational_psi;
        metadata.is_consolidating = self.is_consolidating;
        metadata.epistemic_uncertainty = dynamics.epistemic_uncertainty;
        metadata.aleatoric_uncertainty = dynamics.aleatoric_uncertainty;
        metadata.theta_phase = ((self.stats.total_cycles as f64
            * super::super::super::thresholds::THETA_PHASE_ADVANCE)
            % (2.0 * std::f64::consts::PI)) as f32;
        metadata.temporal_binding_strength = perception.encoding.temporal_binding_strength;
        metadata.prediction_horizon_scale = dynamics.prediction_horizon_tau;
        metadata.fep_tau_factor = dynamics.fep_tau_factor;
        metadata.phi_tau_factor = dynamics.phi_tau_factor;
        metadata.causal_world_model_edges = dynamics.causal_world_model_edges;
        metadata.epistemic_budget_scale = dynamics.epistemic_budget_scale;

        metadata.feedback_signals_fired = (self.feedback_state.confidence.len()
            + self.feedback_state.learning_rate.len()
            + self.feedback_state.exploration.len()
            + self.feedback_state.threshold.len()) as u32;
        metadata.calibration_validations_total =
            self.neuromod.calibration_validator.total_validations();
        metadata.calibration_improvements = self.neuromod.calibration_validator.improvements;
        metadata.calibration_regressions = self.neuromod.calibration_validator.regressions;
        metadata.calibration_adjustment_multiplier =
            self.neuromod.calibration_validator.adjustment_multiplier();
        metadata.calibration_cooldown_duration = self.neuromod.self_assessment.cooldown_duration();
        metadata.feedback_signals_high_water = self.feedback_state.feedback_signals_high_water;
        metadata.feedback_dampened_count = self.feedback_state.feedback_dampened_count;
        metadata.feedback_signal_diversity = self.feedback_state.signal_diversity();
        metadata.avg_transition_cost = self.stats.avg_transition_cost;
        metadata.feedback_dominant_source = self.feedback_state.dominant_source().to_string();
        metadata.error_slope = perception.urgency.error_slope;
        metadata.oscillation_ratio = perception.urgency.oscillation_ratio;
        metadata.mode_transitions = self.stats.mode_transitions as u32;

        metadata.smoothed_epistemic_uncertainty = {
            let raw = dynamics.epistemic_uncertainty;
            let prev = self.carryover.quality.smoothed_epistemic_uncertainty;
            if prev == 0.0 && self.stats.total_cycles <= 1 {
                raw
            } else {
                prev * EPISTEMIC_UNCERTAINTY_EMA_PRIOR + raw * EPISTEMIC_UNCERTAINTY_EMA_CURRENT
            }
        };
    }
}
