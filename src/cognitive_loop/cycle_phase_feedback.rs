//! Feedback integration phase of the cognitive cycle.
//!
//! Extracts the post-processing feedback loops from the original `cycle()` method:
//! consciousness metrics, advanced subsystems, late consciousness monitors,
//! quality-aware adaptive processing, homeostasis, consciousness engine,
//! soul experience integration, cross-module agreement, quality fusion.

use std::time::Instant;

use super::phase_results::{
    DynamicsPhaseResult, FbConsciousness, FbEthics, FbEvolution, FbLoops, FbMemory, FbQuality,
    FbReasoning, FbSelfModel, FbSupport, FeedbackPhaseResult, PerceptionPhaseResult,
};
use super::helpers::{DreamPhaseResult, EpisodicReplayResult, ResonatorCodebookResult};
use super::thresholds::{
    EPISTEMIC_APPROVAL_LR_SCALE, EPISTEMIC_APPROVAL_THRESHOLD, EPISTEMIC_CAUTION_SCALE,
    EPISTEMIC_CAUTION_THRESHOLD, EPISTEMIC_REJECTION_CONFIDENCE_SCALE,
    EPISTEMIC_REJECTION_LR_SCALE, EPISTEMIC_TRUST_SCALE, EPISTEMIC_TRUST_THRESHOLD,
    EVOLUTION_NEGATIVE_EXPLORATION_MAX, EVOLUTION_NEGATIVE_EXPLORATION_SCALE,
    EVOLUTION_PHI_THRESHOLD, EVOLUTION_POSITIVE_CONFIDENCE_MAX,
    EVOLUTION_POSITIVE_CONFIDENCE_SCALE, HARMONIC_ALL_CLEAR_BOOST, HARMONIC_INTERFERENCE_DAMPEN,
    HARMONIC_INTERFERENCE_MAX_COUNT, HARMONIC_INTERFERENCE_MAX_DAMPEN,
    SUBSYSTEM_LR_FACTOR_MAX, SUBSYSTEM_LR_FACTOR_MIN, CROSS_MODULE_AGREEMENT_HIGH,
    CROSS_MODULE_AGREEMENT_LOW, UNIFIED_QUALITY_PREDICTION_WEIGHT,
    UNIFIED_QUALITY_AGREEMENT_WEIGHT, UNIFIED_QUALITY_ANOMALY_WEIGHT,
};
use super::{CognitiveLoopService, CycleState};

impl CognitiveLoopService {
    /// Feedback integration phase: consciousness metrics, advanced subsystems,
    /// late consciousness monitors, quality gating, homeostasis, consciousness
    /// engine, soul experience integration, cross-module agreement.
    pub(super) fn phase_feedback(
        &mut self,
        input: &str,
        perception: &PerceptionPhaseResult,
        dynamics: &mut DynamicsPhaseResult,
        module_timings: &mut super::ModuleTimings,
    ) -> FeedbackPhaseResult {
        let prediction_error = dynamics.core.prediction_error;
        let coherence = dynamics.core.coherence;
        let unified_psi = dynamics.core.unified_psi;

        // ═══════════════════════════════════════════════════════════════════════
        // CYCLE STATE: Shared read-only snapshot for extracted phase functions
        // ═══════════════════════════════════════════════════════════════════════
        let cycle_state = CycleState {
            compressed_state: &perception.encoding.compressed_state,
            output: &dynamics.core.output,
            prediction_error,
            coherence,
            unified_psi,
            phi_attention_weight: perception.encoding.phi_attention_weight,
            hv16_cached: &perception.encoding.hv16_cached,
            input,
            urgency: perception.urgency.urgency,
            attention_budget_exceeded: dynamics.attention.attention_budget_exceeded,
            predictive_budget_gated: dynamics.attention.predictive_budget_gated,
            #[cfg(feature = "vision-manifold")]
            scene_recognized: perception.scene_recognized,
        };

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS METRICS (extracted to cycle_consciousness.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let consciousness_metrics =
            self.compute_consciousness_metrics(&cycle_state, module_timings);

        // Destructure consciousness metrics
        let primitive_psi = consciousness_metrics.primitive_psi;
        let active_primitive_names = consciousness_metrics.active_primitive_names;
        let temporal_causal_chains = consciousness_metrics.temporal_causal_chains;
        let temporal_continuity = consciousness_metrics.temporal_continuity;
        let temporal_max_chain_length = consciousness_metrics.temporal_max_chain_length;
        let _chain_cycle_numbers = consciousness_metrics.chain_cycle_numbers;
        let causal_codebook_entries = consciousness_metrics.causal_codebook_entries;
        let continuity_replay_needed = consciousness_metrics.continuity_replay_needed;
        let lattice_height = consciousness_metrics.lattice_height;
        let lattice_width = consciousness_metrics.lattice_width;
        let lattice_join_concept = consciousness_metrics.lattice_join_concept;
        let compositionality_total = consciousness_metrics.compositionality_total;
        let value_evaluator_score = consciousness_metrics.value_evaluator_score;
        let value_evaluator_decision = consciousness_metrics.value_evaluator_decision;
        let value_gate_factor = consciousness_metrics.value_gate_factor;
        let consciousness_profile_composite = consciousness_metrics.consciousness_profile_composite;
        let synergy_enhanced_composite = consciousness_metrics.synergy_enhanced_composite;
        let emergent_properties_count = consciousness_metrics.emergent_properties_count;
        let reasoning_context = consciousness_metrics.reasoning_context;
        let context_phi_weight = consciousness_metrics.context_phi_weight;

        // ── Phase 18: Context Phi Weight → Unified Psi modulation ────────────
        let context_phi_applied = context_phi_weight > 0.0 && context_phi_weight != 1.0;
        if context_phi_applied {
            let scale = 0.8 + context_phi_weight * 0.4;
            let adjusted_psi = (unified_psi * scale).clamp(0.0, 1.0);
            self.unification_engine.update_psi(adjusted_psi);
            self.stats.context_phi_applied_count += 1;
        }

        let value_embeddings_created = consciousness_metrics.value_embeddings_created;
        let value_cache_hit_rate = consciousness_metrics.value_cache_hit_rate;
        let harmonies_alignment = consciousness_metrics.harmonies_alignment;
        let harmonies_approved = consciousness_metrics.harmonies_approved;
        let composition_rule_applied = consciousness_metrics.composition_rule_applied;
        let harmonic_field_coherence = consciousness_metrics.harmonic_field_coherence;
        let harmonic_love_resonance = consciousness_metrics.harmonic_love_resonance;
        let harmonic_interferences = consciousness_metrics.harmonic_interferences;
        let reasoning_chain_confidence = consciousness_metrics.reasoning_chain_confidence;
        let reasoning_chain_depth = consciousness_metrics.reasoning_chain_depth;
        let causal_relations_count = consciousness_metrics.causal_relations_count;
        let causal_avg_confidence = consciousness_metrics.causal_avg_confidence;
        let adaptive_reasoning_phi = consciousness_metrics.adaptive_reasoning_phi;
        let epistemic_quality = consciousness_metrics.epistemic_quality;
        let phi_validation_correlation = consciousness_metrics.phi_validation_correlation;
        let dissipative_health = consciousness_metrics.dissipative_health
            * (1.0
                - self
                    .somatic_bridge
                    .to_interoceptive_signals()
                    .dissipative_health_penalty);
        let dissipative_regime = consciousness_metrics.dissipative_regime;
        let dissipative_entropy_rate = consciousness_metrics.dissipative_entropy_rate;
        let epistemic_phi_eff = consciousness_metrics.epistemic_phi_eff;
        let epistemic_conflict_count = consciousness_metrics.epistemic_conflict_count;
        let mut equation_v2_consciousness = consciousness_metrics.equation_v2_consciousness;

        // ═══════════════════════════════════════════════════════════════════════
        // ADVANCED SUBSYSTEMS (extracted to cycle_subsystems.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let subsystem_metrics =
            self.run_advanced_subsystems(&cycle_state, &active_primitive_names, module_timings);

        let hierarchical_ltc_phi = subsystem_metrics.hierarchical_ltc_phi;
        let evolution_generation = subsystem_metrics.evolution_generation;
        let evolution_phi_delta = subsystem_metrics.evolution_phi_delta;
        let evolution_confidence_delta = if evolution_phi_delta > EVOLUTION_PHI_THRESHOLD {
            (evolution_phi_delta * EVOLUTION_POSITIVE_CONFIDENCE_SCALE)
                .min(EVOLUTION_POSITIVE_CONFIDENCE_MAX) as f32
        } else if evolution_phi_delta < -EVOLUTION_PHI_THRESHOLD {
            -((-evolution_phi_delta) * EVOLUTION_NEGATIVE_EXPLORATION_SCALE)
                .min(EVOLUTION_NEGATIVE_EXPLORATION_MAX) as f32
        } else {
            0.0
        };
        let holographic_unity = subsystem_metrics.holographic_unity;
        let holographic_binding = subsystem_metrics.holographic_binding;
        let consciousness_gradient_magnitude = subsystem_metrics.consciousness_gradient_magnitude;
        let consciousness_limiting_component = subsystem_metrics.consciousness_limiting_component;
        let affect_cons_valence = subsystem_metrics.affect_cons_valence;
        let affect_cons_arousal = subsystem_metrics.affect_cons_arousal;
        let pipeline_consciousness = subsystem_metrics.pipeline_consciousness;
        let multimodal_integrated_phi = subsystem_metrics.multimodal_integrated_phi;
        let consciousness_state_label = subsystem_metrics.consciousness_state_label;
        let consciousness_state_level = subsystem_metrics.consciousness_state_level;
        let epistemic_gate_confidence = subsystem_metrics.epistemic_gate_confidence;
        let epistemic_gate_approved = subsystem_metrics.epistemic_gate_approved;
        let primitive_validation_phi_gain = subsystem_metrics.primitive_validation_phi_gain;
        let primitive_validation_p_value = subsystem_metrics.primitive_validation_p_value;
        let meta_reasoning_confidence = subsystem_metrics.meta_reasoning_confidence;
        let meta_reasoning_insights = subsystem_metrics.meta_reasoning_insights;
        let code_primitives_selected = subsystem_metrics.code_primitives_selected;
        let empathic_compassion = subsystem_metrics.empathic_compassion;
        let empathic_tone_adj = subsystem_metrics.empathic_tone_adj;
        let multi_obj_frontier_size = subsystem_metrics.multi_obj_frontier_size;
        let grid_encoding_norm = subsystem_metrics.grid_encoding_norm;
        let grid_spatial_complexity = subsystem_metrics.grid_spatial_complexity;

        // ── Phase 18: Empathic tone → speech rate modulation ─────────────────
        let empathic_speech_rate_mod = if empathic_tone_adj.abs() > 0.1 {
            let rate_mod = 1.0 - empathic_tone_adj as f32 * 0.1;
            self.adaptive_behavior.speech_rate_multiplier *= rate_mod;
            self.adaptive_behavior.speech_rate_multiplier = self
                .adaptive_behavior
                .speech_rate_multiplier
                .clamp(0.6, 1.5);
            empathic_tone_adj as f32
        } else {
            0.0
        };

        // ── Phase 19: Consciousness limiting component → targeted boost ─────
        let limiting_component_boosted = if !consciousness_limiting_component.is_empty()
            && consciousness_gradient_magnitude > 0.01
        {
            match consciousness_limiting_component.as_str() {
                "Attention" => {
                    self.adaptive_behavior.attention_sensitivity =
                        (self.adaptive_behavior.attention_sensitivity * 1.05).min(2.0);
                    self.stats.limiting_component_boost_count += 1;
                    "Attention"
                }
                "Binding" => {
                    self.adjust_confidence("limit_binding", 0.01);
                    self.stats.limiting_component_boost_count += 1;
                    "Binding"
                }
                "Efficacy" => {
                    self.scale_lr("limit_efficacy", 1.05);
                    self.stats.limiting_component_boost_count += 1;
                    "Efficacy"
                }
                _ => "",
            }
        } else {
            ""
        };

        // ── Phase 19: Harmonic love resonance → confidence/soul amplifier ────
        let love_resonance_boost = if harmonic_love_resonance > 0.6 {
            let boost = ((harmonic_love_resonance - 0.6) * 0.04) as f32;
            self.adjust_confidence("love_resonance", boost);
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost * 0.5;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
            self.stats.love_resonance_boost_count += 1;
            boost
        } else {
            0.0
        };

        // ── Phase 19: Reasoning chain confidence + depth → confidence ────────
        let reasoning_chain_boosted =
            reasoning_chain_confidence > 0.7 && reasoning_chain_depth >= 3;
        if reasoning_chain_boosted {
            let chain_boost = (reasoning_chain_confidence - 0.7) * 0.05;
            self.adjust_confidence("reasoning_chain", chain_boost);
            self.stats.reasoning_chain_boost_count += 1;
        }

        // ── Phase 20: Harmonic interferences → LR feedback ───────────────────
        let harmonic_interference_lr_mod: f32 =
            if harmonic_interferences > HARMONIC_INTERFERENCE_MAX_COUNT {
                let dampen = ((harmonic_interferences - HARMONIC_INTERFERENCE_MAX_COUNT) as f32
                    * HARMONIC_INTERFERENCE_DAMPEN)
                    .min(HARMONIC_INTERFERENCE_MAX_DAMPEN);
                self.carryover.learning.subsystem_lr_factor *= 1.0 - dampen;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
                self.stats.harmonic_interference_mod_count += 1;
                -dampen
            } else if harmonic_interferences == 0 {
                self.carryover.learning.subsystem_lr_factor *= 1.0 + HARMONIC_ALL_CLEAR_BOOST;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
                self.stats.harmonic_interference_mod_count += 1;
                HARMONIC_ALL_CLEAR_BOOST
            } else {
                0.0
            };

        // ── Social trust → learning rate modulation (Decety & Chaminade 2003) ──
        let social_learning_rate_factor = 0.8 + 0.4 * self.social_mgr.social.social_trust; // [0.8, 1.2]
        if (social_learning_rate_factor - 1.0).abs() > 0.01 {
            self.scale_lr("social_trust", social_learning_rate_factor);
        }

        // ── Phase 20: Causal relations density → urgency gating ──────────────
        let causal_urgency_gated = causal_relations_count > 10
            && causal_avg_confidence > 0.6
            && self.stats.total_cycles > 20;
        if causal_urgency_gated {
            self.carryover.urgency.consecutive_low_error = self
                .carryover
                .urgency
                .consecutive_low_error
                .saturating_add(2);
            self.stats.causal_urgency_gated_count += 1;
        }

        let attention_budget_gated =
            dynamics.attention.attention_budget_exceeded && self.stats.attention_budget_exceeded_count > 3;

        // ── Track 5a: Epistemic gate → actual information gating ─────────────
        let mut epistemic_coherence_gated = false;
        if !epistemic_gate_approved {
            let rejection_strength = (1.0 - epistemic_gate_confidence).clamp(0.0, 0.5);
            self.carryover.learning.subsystem_lr_factor *=
                1.0 - rejection_strength * EPISTEMIC_REJECTION_LR_SCALE;
            self.scale_confidence(
                "epistemic_reject",
                1.0 - rejection_strength * EPISTEMIC_REJECTION_CONFIDENCE_SCALE,
            );
        } else if epistemic_gate_confidence > EPISTEMIC_APPROVAL_THRESHOLD {
            let approval_boost = (epistemic_gate_confidence - EPISTEMIC_APPROVAL_THRESHOLD)
                * EPISTEMIC_APPROVAL_LR_SCALE;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + approval_boost;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
        }

        if epistemic_gate_confidence < EPISTEMIC_CAUTION_THRESHOLD
            && epistemic_gate_confidence > 0.0
        {
            let caution_factor =
                (EPISTEMIC_CAUTION_THRESHOLD - epistemic_gate_confidence) * EPISTEMIC_CAUTION_SCALE;
            self.scale_threshold("epistemic_gate_caution", 1.0 + caution_factor);
            epistemic_coherence_gated = true;
            self.stats.epistemic_coherence_gated_count += 1;
        } else if epistemic_gate_confidence > EPISTEMIC_TRUST_THRESHOLD {
            let trust_factor =
                (epistemic_gate_confidence - EPISTEMIC_TRUST_THRESHOLD) * EPISTEMIC_TRUST_SCALE;
            self.scale_threshold("epistemic_gate_trust", 1.0 - trust_factor);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // RESONATOR CODEBOOK GROWTH
        // ═══════════════════════════════════════════════════════════════════════
        let reflection_thresholds = self.self_model_tier.self_reflection.get_thresholds();
        let ResonatorCodebookResult {
            resonator_promotions,
            codebook_evictions,
            codebook_diversity,
            codebook_utilization_rate,
        } = if perception.encoding.input_memoized {
            ResonatorCodebookResult {
                resonator_promotions: 0,
                codebook_evictions: 0,
                codebook_diversity: self.stats.codebook_diversity,
                codebook_utilization_rate: self.stats.codebook_utilization_rate,
            }
        } else {
            self.run_resonator_codebook_phase(
                epistemic_gate_approved,
                &perception.encoding.compressed_state,
                &active_primitive_names,
                &causal_codebook_entries,
                &reflection_thresholds,
                module_timings,
            )
        };

        // ═══════════════════════════════════════════════════════════════════════
        // EPISODIC REPLAY + MEMORY COORDINATOR
        // ═══════════════════════════════════════════════════════════════════════
        let surprise_thresh = reflection_thresholds.surprise as f64;
        let EpisodicReplayResult {
            surprise_replay_batch_size,
            phasic_da_replay_boost,
        } = self.run_episodic_replay_and_memory_phase(
            &cycle_state,
            dynamics.fep.fep_surprise as f32, // memory_context_boost already handled
            dynamics.fep.fep_surprise,
            surprise_thresh,
            module_timings,
        );
        dynamics.neuromod.phasic_da_replay_boost = phasic_da_replay_boost;

        // ═══════════════════════════════════════════════════════════════════════
        // SUPPORT INTELLIGENCE
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        #[cfg(feature = "support")]
        let (support_triage_count, support_alert_fired, support_federation_graduated, support_efe) = {
            self.support_cycle_counter += 1;

            let mut triage_count: u32 = 0;
            if let Some(ref engine) = self.support_triage_engine {
                let result = engine.triage(input, "");
                triage_count = 1;
                if let Some(ref manager) = self.support_knowledge_manager {
                    let category_str = format!("{:?}", result.suggested_category);
                    let articles = manager.search(&category_str, 3);
                    if !articles.is_empty() {
                        tracing::trace!(
                            category = %category_str,
                            articles = articles.len(),
                            "Support: matched knowledge articles for input"
                        );
                    }
                }
            }

            let mut alert_fired = false;
            let mut efe = 0.0_f64;
            if self.support_cycle_counter % 47 == 0 {
                if let Some(ref engine) = self.support_predictive_engine {
                    let telemetry = symthaea_support::telemetry::collect_telemetry();
                    let prediction = engine.assess_system_state(&telemetry);
                    efe = prediction.expected_free_energy;
                    if engine.should_alert(&prediction) {
                        alert_fired = true;
                        tracing::warn!(
                            efe = prediction.expected_free_energy,
                            failure = ?prediction.predicted_failure,
                            "Support predictive alert: elevated free energy"
                        );
                    }
                }
            }

            let mut graduated: usize = 0;
            if self.support_cycle_counter % 97 == 0 {
                let can_share = self
                    .support_privacy_manager
                    .as_ref()
                    .map(|pm| pm.can_share_cognitive())
                    .unwrap_or(true);

                if can_share {
                    if let Some(ref manager) = self.support_knowledge_manager {
                        let pending = Vec::new();
                        let result =
                            symthaea_support::federation::check_graduations(manager, &pending);
                        graduated = result.graduated;
                        if result.graduated > 0 {
                            tracing::debug!(
                                graduated = result.graduated,
                                deferred = result.deferred,
                                "Support federation: knowledge graduated"
                            );
                        }
                    }
                } else {
                    tracing::trace!("Support federation skipped: privacy tier blocks sharing");
                }
            }

            (triage_count, alert_fired, graduated, efe)
        };
        #[cfg(not(feature = "support"))]
        let (support_triage_count, support_alert_fired, support_federation_graduated, support_efe) =
            (0u32, false, 0usize, 0.0f64);
        module_timings.support_intelligence = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 10.5: Hyper-Parameter Optimization
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let opt_result = self.run_parameter_optimization_phase();
        module_timings.parameter_optimization = _t.elapsed().as_micros() as u64;

        if opt_result.swap_occurred {
            self.stats.brain_swaps_count = self.stats.brain_swaps_count.saturating_add(1);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 11: DREAM ENGINE (substrate-gated: skip when degraded)
        // ═══════════════════════════════════════════════════════════════════════
        let DreamPhaseResult {
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
        } = if self.substrate_manager.should_degrade_consciousness() {
            DreamPhaseResult {
                dream_insights: 0,
                dream_phi_improvement: 0.0,
                dream_wisdom_count: 0,
            }
        } else {
            self.run_dream_phase(&cycle_state, &dynamics.core.prediction, module_timings)
        };

        // 7 (deferred). Send prediction to encoder for next cycle
        // SAFETY: We take the prediction out of dynamics. The output phase must
        // not read dynamics.core.prediction after this point. We use std::mem::take
        // to move the Vec without allocation.
        self.encoder
            .set_prediction(std::mem::take(&mut dynamics.core.prediction));

        // ═══════════════════════════════════════════════════════════════════════
        // LATE CONSCIOUSNESS MONITORS
        // ═══════════════════════════════════════════════════════════════════════
        use super::cycle_late_consciousness::LateConsciousnessContext;

        let late_ctx = LateConsciousnessContext {
            prediction_error,
            coherence,
            unified_psi,
            hv16_cached: perception.encoding.hv16_cached,
            compressed_state: &perception.encoding.compressed_state,
            input,
            urgency: perception.urgency.urgency,
            moral_concern_detected: perception.moral.moral_concern_detected,
            surprise_triggered: perception.exploration.surprise_triggered,
            reasoning_gate_blocked: dynamics.reasoning.reasoning_gate_blocked,
            pp_phi: self.unification_engine.psi as f32,
            peak_attention: perception.encoding.encoding_result.peak_attention,
        };

        let late_result = self.run_late_consciousness_monitors(&late_ctx, module_timings);
        let integration_result =
            self.run_consciousness_integration(&late_ctx, &late_result, module_timings);

        let prefrontal_veto = late_result.prefrontal_veto;
        let meta_cognitive_accuracy = late_result.meta_cognitive_accuracy;
        let meta_cognitive_depth = late_result.meta_cognitive_depth;
        let body_psi_modulation = late_result.body_psi_modulation;
        let body_valence = late_result.body_valence;
        let body_arousal = late_result.body_arousal;
        let affective_valence = late_result.affective_valence;
        let affective_arousal = late_result.affective_arousal;
        let narrative_self_psi = late_result.narrative_self_psi;
        let predictive_free_energy = late_result.predictive_free_energy;
        self.carryover.consciousness.last_predictive_free_energy = predictive_free_energy;
        let predictive_psi_modulation = late_result.predictive_psi_modulation;
        let hierarchical_total_free_energy = late_result.hierarchical_total_free_energy;
        let predictive_self_safety = late_result.predictive_self_safety;
        let attention_schema_focus = late_result.attention_schema_focus;
        let attention_fatigue = late_result.attention_fatigue;
        let attention_prediction_accuracy = late_result.attention_prediction_accuracy;
        let psi_attention_avg = late_result.psi_attention_avg;

        let gwt_broadcast = integration_result.gwt_broadcast;
        let gwt_coalition_size = integration_result.gwt_coalition_size;
        let cross_modal_binding_strength = integration_result.cross_modal_binding_strength;
        let cross_modal_psi = integration_result.cross_modal_psi;
        let resonance_frequency = integration_result.resonance_frequency;
        let quantum_coherence_level = integration_result.quantum_coherence_level;
        let phenomenal_binding_strength = integration_result.phenomenal_binding_strength;
        let phenomenal_fragmented = integration_result.phenomenal_fragmented;
        let temporal_coherence_score = integration_result.temporal_coherence_score;
        let temporal_discontinuity = integration_result.temporal_discontinuity;
        let thermodynamic_entropy = integration_result.thermodynamic_entropy;
        let thermodynamic_free_energy = integration_result.thermodynamic_free_energy;
        let embodied_psi_modulation = integration_result.embodied_psi_modulation;
        let embodied_agency = integration_result.embodied_agency;
        self.carryover.consciousness.last_embodied_agency = embodied_agency;
        let narrative_gwt_veto = integration_result.narrative_gwt_veto;
        let narrative_gwt_self_psi = integration_result.narrative_gwt_self_psi;
        let living_mind_vitality = integration_result.living_mind_vitality;
        let living_mind_coherence = integration_result.living_mind_coherence;
        let consciousness_level = integration_result.consciousness_level;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 16 + HOMEOSTASIS (extracted to cycle_quality.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let quality_result = self.run_quality_and_homeostasis(
            coherence,
            temporal_discontinuity,
            perception.exploration.exploration_urge_start,
            dynamics.homeostasis.homeostasis_pull_strength,
            dissipative_health,
            phenomenal_binding_strength,
            module_timings,
        );
        let dissipative_lr_factor = quality_result.dissipative_lr_factor;
        let dissipative_health_gated = quality_result.dissipative_health_gated;
        let coherence_velocity = quality_result.coherence_velocity;
        let coherence_velocity_gated = quality_result.coherence_velocity_gated;
        self.carryover.urgency.urgency = perception.urgency.urgency;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED CONSCIOUSNESS ENGINE
        // ═══════════════════════════════════════════════════════════════════════
        let encoding_hdv = &perception.encoding.encoding_result.hdv;
        let phi_spectral_weight = self.carryover.quality.phi_spectral_weight;
        let consciousness_output = self.consciousness_engine.measure(
            &super::consciousness_engine::ConsciousnessEngineInput {
                hdv: encoding_hdv,
                hv16: &perception.encoding.hv16_cached,
                cycle: self.stats.total_cycles as u64,
                unified_psi,
                coherence,
                prediction_error,
                phi_attention_weight: perception.encoding.phi_attention_weight,
                epistemic_quality: self.carryover.quality.last_epistemic_quality,
                phi_validation_correlation: self.carryover.quality.phi_validation_correlation,
                // Phase 6: bath → consciousness coupling
                bath_entropy: self.neuromod.phase_tracker.entropy(),
                attractor_detected: self.neuromod.phase_tracker.detect_attractor().is_some(),
                sht_2a_signal: self.neuromod.bath.sht_2a_signal(),
                gaba_a_signal: self.neuromod.bath.gaba_a_signal(),
                substrate_feasibility: self.substrate_manager.effective_feasibility,
                // Substrate requirement dimensions → consciousness coupling
                binding_capability: self.substrate_manager.binding_capability(&self.config),
                workspace_capability: self.substrate_manager.workspace_capability(&self.config),
                attention_capability: self.substrate_manager.attention_capability(&self.config),
                // Moral topology → consciousness coupling
                moral_drift: self.ethics_engine.moral_topology().moral_drift(20),
                moral_anomaly_score: self.ethics_engine.last_anomaly_report().anomaly_score,
            },
        );
        self.consciousness_engine
            .update_cache(&mut self.carryover.consciousness);
        if consciousness_output.confidence_delta != 0.0 {
            self.adjust_confidence(
                "consciousness_engine",
                consciousness_output.confidence_delta,
            );
        }
        if consciousness_output.lr_factor != 1.0 {
            self.scale_lr("consciousness_engine", consciousness_output.lr_factor);
        }
        if consciousness_output.exploration_delta != 0.0 {
            self.adjust_exploration(
                "consciousness_engine",
                consciousness_output.exploration_delta,
            );
        }
        if consciousness_output.subsystem_lr_factor != 1.0 {
            self.carryover.learning.subsystem_lr_factor *= consciousness_output.subsystem_lr_factor;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.2);
        }
        if let Some(consolidation_boost) = consciousness_output.episodic_consolidation_boost {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                replay.boost_recent_consolidation(consolidation_boost);
            }
        }
        // Theta phase → Phi modulation (Buzsáki 2006).
        // 6Hz theta oscillation at 50Hz loop rate. Peaks enhance integration; troughs suppress.
        // EMA-smoothed to prevent 6Hz artifacts in downstream consciousness metrics.
        let theta_phase = (self.stats.total_cycles as f64
            * super::thresholds::THETA_PHASE_ADVANCE)
            % (2.0 * std::f64::consts::PI);
        let theta_phi_mod = theta_phase.sin()
            * super::thresholds::THETA_PHI_MODULATION_AMPLITUDE;
        let spectral_mip_phi = consciousness_output
            .spectral_mip_phi
            .map(|phi| {
                let raw = (phi * (1.0 + theta_phi_mod)).max(0.0);
                // EMA smooth to prevent rhythmic artifacts in consciousness telemetry.
                let alpha = super::thresholds::THETA_PHI_SMOOTH_ALPHA;
                let prev = self.carryover.consciousness.last_spectral_mip_phi
                    .unwrap_or(raw);
                prev * (1.0 - alpha) + raw * alpha
            });
        let sigma = consciousness_output.sigma;
        let eq_v2_limiting_component = consciousness_output
            .limiting_component
            .map(|c| format!("{c:?}"))
            .unwrap_or_default();
        self.carryover.quality.last_pipeline_consciousness =
            consciousness_output.pipeline_consciousness;
        module_timings.spectral_mip = consciousness_output.spectral_mip_us;
        module_timings.consciousness_engine = consciousness_output.total_us;
        module_timings.consciousness_engine_equation_v2 = consciousness_output.equation_v2_us;
        module_timings.consciousness_engine_pipeline = consciousness_output.pipeline_us;
        module_timings.consciousness_engine_multimodal = consciousness_output.multimodal_us;

        // ── Structural Phi telemetry ────────────────────────────────────
        let (struct_micro, struct_meso, struct_macro, struct_bn, struct_er, struct_nc) =
            if let Some(ref sp) = consciousness_output.structural_phi {
                // Hierarchical scale diversity boost: when the temporal network
                // operates at multiple scales (HierarchicalCfC), cross-scale
                // integration contributes to emergence. Mediano et al. (2022):
                // multi-scale integrated information exceeds single-scale Phi.
                let scale_boost = if let Some(taus) =
                    self.temporal_network.hierarchical_effective_taus()
                {
                    let mean_tau =
                        taus.iter().sum::<f32>() / taus.len().max(1) as f32;
                    if mean_tau > 0.0 {
                        let var = taus
                            .iter()
                            .map(|t| (t - mean_tau).powi(2))
                            .sum::<f32>()
                            / taus.len().max(1) as f32;
                        let cv = var.sqrt() / mean_tau;
                        // CV for default taus [0.01,0.1,1.0,10.0] ~ 1.7
                        // Map to 0-15% boost via sigmoid
                        (0.15 * (1.0 / (1.0 + (-2.0 * (cv - 1.0)).exp()))) as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                (
                    sp.micro_phi * (1.0 + scale_boost * 0.5),
                    sp.meso_phi * (1.0 + scale_boost),
                    sp.macro_phi * (1.0 + scale_boost),
                    sp.bottleneck_score,
                    sp.emergence_ratio * (1.0 + scale_boost),
                    sp.num_clusters,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0)
            };
        let consciousness_weights = consciousness_output.current_weights;
        let consciousness_weight_variance = consciousness_output.weight_variance;
        let convergence_state = format!("{:?}", consciousness_output.convergence_state);

        // Phi-Harmony coupling: during Sacred Stillness, weight Phi by integration
        // quality rather than raw intensity. Rest-state Phi should reflect how
        // well the system maintains coherent integration while reducing activity.
        // Science: Tononi (2004) — Phi measures integrated information regardless
        // of activity level; rest-state Phi is not diminished but differently structured.
        if self.stats.in_active_rest {
            // During active rest, boost the coherence component of Phi
            // and dampen the binding intensity component
            let coherence_weight = 1.2; // 20% more weight on coherence
            let binding_dampen = 0.8; // 20% less weight on binding intensity
            self.stats.phi_rest_quality_factor = coherence_weight;
            self.stats.phi_rest_binding_factor = binding_dampen;
            // Apply: modulate the EqV2 consciousness score by rest factors
            // Quality factor boosts integration coherence, binding factor dampens raw intensity
            let rest_modulation = (coherence_weight + binding_dampen) / 2.0;
            equation_v2_consciousness *= rest_modulation as f64;
        } else {
            self.stats.phi_rest_quality_factor = 1.0;
            self.stats.phi_rest_binding_factor = 1.0;
        }

        // ── EqV2 limiting component → targeted boost ─────────────────────
        // Complements Phase 19 gradient-based boost with equation-level
        // bottleneck detection. Covers all 7 CoreComponents including
        // Workspace, Recursion, Integration, Knowledge (not in gradient path).
        if let Some(ref component) = consciousness_output.limiting_component {
            use crate::consciousness::consciousness_equation_v2::CoreComponent;
            match component {
                CoreComponent::Workspace => {
                    // Low workspace → boost attention budget and GWT broadcast
                    self.adjust_confidence("eq_v2_workspace", 0.005);
                }
                CoreComponent::Recursion => {
                    // Low recursion (HOT depth) → boost meta-cognitive sensitivity
                    self.scale_lr("eq_v2_recursion", 1.02);
                }
                CoreComponent::Integration => {
                    // Low integration → increase coherence sensitivity
                    self.adjust_confidence("eq_v2_integration", 0.008);
                }
                CoreComponent::Knowledge => {
                    // Low epistemic quality → boost exploration for information gain
                    self.adjust_exploration("eq_v2_knowledge", 0.02);
                }
                // Attention, Binding, Efficacy already handled by Phase 19 gradient path
                _ => {}
            }
        }

        // Soul experience integration
        let _t = Instant::now();
        if let Some(ref mut soul) = self.soul {
            let moral_score = self
                .last_moral_judgment
                .as_ref()
                .map(|j| j.moral_score)
                .unwrap_or(0.0);
            let experience = crate::soul::Experience {
                embedding: encoding_hdv.clone(),
                value_alignment: moral_score,
                emotional_valence: self.emotion_contagion.valence,
                lessons: Vec::new(),
                timestamp: self.stats.total_cycles as u64,
            };
            soul.integrate_experience(experience);
        }
        module_timings.soul_experience = _t.elapsed().as_micros() as u64;

        // ── Phi-Dyad: Relational Consciousness ─────────────────────────────
        // Compute Φ_dyad from recent AI + input HVs (Phase 6 wiring).
        if self.social_mgr.recent_ai_hvs.len() >= 2 {
            if let (Some(ref dyad), Some(ref model)) = (&self.social_mgr.phi_dyad, &self.social_mgr.partner_model) {
                use symthaea_core::hdc::relational_consciousness::{
                    RelationMode, RelationalAssessment,
                };
                let relational = RelationalAssessment {
                    agent_a: "symthaea".to_string(),
                    agent_b: model.partner_id.clone(),
                    phi_relation: model.phi_relational,
                    stage: model.stage,
                    synchrony: model.trust as f64,
                    turn_taking_quality: 0.7,
                    mutual_information: model.reciprocity as f64,
                    mode: if model.trust > 0.3 {
                        RelationMode::IThou
                    } else {
                        RelationMode::IIt
                    },
                    num_interactions: model.interactions_count as usize,
                    relationship_age: 0.0,
                    explanation: String::new(),
                };
                let input = crate::partnership::DyadInput {
                    ai_states: &self.social_mgr.recent_ai_hvs,
                    human_states: &self.social_mgr.recent_input_hvs,
                    relational: &relational,
                    human_model: model,
                    weights: crate::partnership::DyadWeights::default(),
                };
                let result = dyad.compute(&input);
                self.social_mgr.social.relational_psi = result.phi_dyad;

                // Phi divergence → exploration (novel relational territory)
                // Science: Friston (2010) — high divergence = high epistemic value
                let phi_divergence = (result.phi_ai - result.phi_human).abs();
                if phi_divergence > 0.1 {
                    let boost = (phi_divergence - 0.1).min(0.2) * 0.15;
                    self.adjust_exploration("phi_divergence", boost as f32);
                }

                // Phi relational → oxytocin (prosocial bonding)
                // Science: Feldman (2012) — relational coherence drives oxytocin release
                if result.phi_relational > 0.3 {
                    let oxy = (result.phi_relational - 0.3) * 0.05;
                    self.neuromod.bath.oxytocin.produce(oxy as f32);
                }
            }
        }

        // Trust evolution from cycle coherence (Bowlby 1969)
        // Coherence > 0.5 builds trust, < 0.5 erodes it; slow decay prevents runaway
        if let Some(ref mut model) = self.social_mgr.partner_model {
            let signal = (dynamics.core.coherence as f64 - 0.5) * 0.01;
            model.trust = ((model.trust as f64 + signal).clamp(0.0, 1.0) * 0.999) as f32;
        }

        // ── Track 4b: Cross-module agreement metric ─────────────────────────
        let fep_confidence = (1.0 - dynamics.fep.fep_surprise.min(1.0)).max(0.0) as f32;
        let resonator_confidence = dynamics.resonator.resonator_best_sim;
        let moral_confidence = self
            .last_moral_judgment
            .as_ref()
            .map(|j| (j.moral_score + 1.0) / 2.0)
            .unwrap_or(0.5);
        let mcts_confidence = self
            .carryover
            .history
            .mcts_plan
            .as_ref()
            .map(|&(_, c)| c)
            .unwrap_or(0.5);
        let signals = [
            fep_confidence,
            resonator_confidence,
            moral_confidence,
            mcts_confidence,
        ];
        let mean_signal: f32 = signals.iter().sum::<f32>() / signals.len() as f32;
        let variance: f32 = signals
            .iter()
            .map(|s| (s - mean_signal).powi(2))
            .sum::<f32>()
            / signals.len() as f32;
        let cross_module_agreement = (1.0 - (variance * 4.0).sqrt()).clamp(0.0, 1.0);
        if cross_module_agreement > CROSS_MODULE_AGREEMENT_HIGH {
            self.adjust_confidence("cross_mod_agree", (cross_module_agreement - CROSS_MODULE_AGREEMENT_HIGH) * 0.05);
        } else if cross_module_agreement < CROSS_MODULE_AGREEMENT_LOW {
            self.scale_confidence(
                "cross_mod_disagree",
                1.0 - (CROSS_MODULE_AGREEMENT_LOW - cross_module_agreement) * 0.1,
            );
            self.adjust_exploration(
                "cross_module_disagree",
                (CROSS_MODULE_AGREEMENT_LOW - cross_module_agreement) * 0.15,
            );
        }
        self.stats.avg_cross_module_agreement =
            self.stats.avg_cross_module_agreement * 0.95 + cross_module_agreement * 0.05;

        // Cross-module agreement velocity: rapid drops signal subsystem desynchronization.
        // Analogous to coherence_velocity but for inter-module rather than intra-module signals.
        let agreement_velocity = cross_module_agreement
            - self.carryover.quality.prev_cross_module_agreement;
        self.carryover.quality.prev_cross_module_agreement = cross_module_agreement;
        if agreement_velocity < -0.15 && self.stats.total_cycles > 30 {
            // Rapid agreement drop → dampen LR, boost exploration preemptively.
            // Science: desynchronization across subsystems means conflicting learning signals.
            self.scale_lr("agreement_vel_drop", 0.97);
            self.adjust_exploration("agreement_vel_drop", 0.015);
        }

        // ── Unified quality signal fusion ───────────────────────────
        let unified_quality_score;
        {
            let anomaly_factor = if dynamics.reasoning.metacognitive_anomaly {
                0.0
            } else {
                1.0
            };
            unified_quality_score = UNIFIED_QUALITY_PREDICTION_WEIGHT * dynamics.core.prediction_coherence
                + UNIFIED_QUALITY_AGREEMENT_WEIGHT * cross_module_agreement
                + UNIFIED_QUALITY_ANOMALY_WEIGHT * anomaly_factor;
            self.stats.avg_unified_quality =
                self.stats.avg_unified_quality * 0.9 + unified_quality_score * 0.1;

            if unified_quality_score > CROSS_MODULE_AGREEMENT_HIGH {
                let quality_boost = (unified_quality_score - CROSS_MODULE_AGREEMENT_HIGH) * 0.25;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + quality_boost;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.5);
            }
            if unified_quality_score < CROSS_MODULE_AGREEMENT_LOW && self.stats.total_cycles > 30 {
                self.scale_exploration("low_quality_dampen", 0.9);
            }
        }

        FeedbackPhaseResult {
            quality: FbQuality {
                cross_module_agreement,
                unified_quality_score,
                coherence_velocity_gated,
                dissipative_health_gated,
                dissipative_health,
                dissipative_regime,
                dissipative_entropy_rate,
                dissipative_lr_factor,
                coherence_velocity,
            },
            consciousness: FbConsciousness {
                primitive_psi,
                temporal_causal_chains,
                temporal_continuity,
                temporal_max_chain_length,
                causal_codebook_entries_len: causal_codebook_entries.len(),
                continuity_replay_needed,
                lattice_height,
                lattice_width,
                lattice_join_concept,
                compositionality_total,
                consciousness_profile_composite,
                synergy_enhanced_composite,
                emergent_properties_count,
                equation_v2_consciousness,
                eq_v2_limiting_component,
                pipeline_consciousness,
                multimodal_integrated_phi,
                consciousness_state_label,
                consciousness_state_level,
                consciousness_gradient_magnitude,
                consciousness_limiting_component,
                holographic_unity,
                holographic_binding,
                affect_cons_valence,
                affect_cons_arousal,
                consciousness_level,
                spectral_mip_phi,
                sigma,
                phi_spectral_weight,
                structural_micro_phi: struct_micro,
                structural_meso_phi: struct_meso,
                structural_macro_phi: struct_macro,
                structural_bottleneck: struct_bn,
                structural_emergence_ratio: struct_er,
                structural_num_clusters: struct_nc,
                consciousness_weights,
                consciousness_weight_variance,
                convergence_state,
            },
            self_model: FbSelfModel {
                prefrontal_veto,
                meta_cognitive_accuracy,
                meta_cognitive_depth,
                body_psi_modulation,
                body_valence,
                body_arousal,
                affective_valence,
                affective_arousal,
                narrative_self_psi,
                predictive_free_energy,
                predictive_psi_modulation,
                hierarchical_total_free_energy,
                predictive_self_safety,
                attention_schema_focus,
                attention_fatigue,
                attention_prediction_accuracy,
                psi_attention_avg,
                gwt_broadcast,
                gwt_coalition_size,
                cross_modal_binding_strength,
                cross_modal_psi,
                resonance_frequency,
                quantum_coherence_level,
                phenomenal_binding_strength,
                phenomenal_fragmented,
                temporal_coherence_score,
                temporal_discontinuity,
                thermodynamic_entropy,
                thermodynamic_free_energy,
                embodied_psi_modulation,
                embodied_agency,
                narrative_gwt_veto,
                narrative_gwt_self_psi,
                living_mind_vitality,
                living_mind_coherence,
            },
            reasoning: FbReasoning {
                reasoning_context,
                context_phi_weight,
                context_phi_applied,
                reasoning_chain_confidence,
                reasoning_chain_depth,
                causal_relations_count,
                causal_avg_confidence,
                adaptive_reasoning_phi,
                epistemic_quality,
                phi_validation_correlation,
                epistemic_phi_eff,
                epistemic_conflict_count,
                epistemic_gate_confidence,
                epistemic_gate_approved,
                meta_reasoning_confidence,
                meta_reasoning_insights,
                code_primitives_selected,
            },
            ethics: FbEthics {
                value_evaluator_score,
                value_evaluator_decision,
                value_gate_factor,
                value_embeddings_created,
                value_cache_hit_rate,
                harmonies_alignment,
                harmonies_approved,
                composition_rule_applied,
                harmonic_field_coherence,
                harmonic_love_resonance,
                harmonic_interferences,
                empathic_compassion,
                empathic_tone_adj,
                empathic_speech_rate_mod,
            },
            evolution: FbEvolution {
                hierarchical_ltc_phi,
                evolution_generation,
                evolution_phi_delta,
                evolution_confidence_delta,
                primitive_validation_phi_gain,
                primitive_validation_p_value,
            },
            loops: FbLoops {
                limiting_component_boosted: limiting_component_boosted.into(),
                love_resonance_boost,
                reasoning_chain_boosted,
                harmonic_interference_lr_mod,
                causal_urgency_gated,
                epistemic_coherence_gated,
                attention_budget_gated,
            },
            memory: FbMemory {
                dream_insights,
                dream_phi_improvement,
                dream_wisdom_count,
                resonator_promotions,
                codebook_evictions,
                codebook_diversity,
                codebook_utilization_rate,
                surprise_replay_batch_size,
            },
            support: FbSupport {
                support_triage_count,
                support_alert_fired,
                support_federation_graduated,
                support_efe,
            },
            multi_obj_frontier_size,
            grid_encoding_norm,
            grid_spatial_complexity,
            social_learning_rate_factor,
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
    fn feedback_consciousness_level_in_range() {
        let mut svc = make_service();
        let result = svc.cycle("consciousness level");
        assert!(
            result.metadata.consciousness_level >= 0.0
                && result.metadata.consciousness_level <= 1.0
        );
    }

    #[test]
    fn feedback_quality_diagnostics_populated() {
        let mut svc = make_service();
        let result = svc.cycle("quality diag");
        assert!(result.metadata.quality.dissipative_health.is_finite());
        assert!(result.metadata.quality.meta_cognitive_accuracy.is_finite());
        assert!(result.metadata.quality.unified_quality_score.is_finite());
    }

    #[test]
    fn feedback_epistemic_gate_fields_populated() {
        let mut svc = make_service();
        let result = svc.cycle("epistemic gate");
        assert!(result.metadata.epistemic_gate_confidence.is_finite());
        let _ = result.metadata.epistemic_gate_approved;
    }

    #[test]
    fn feedback_sigma_finite() {
        let mut svc = make_service();
        let result = svc.cycle("sigma test");
        if let Some(sigma) = result.metadata.sigma {
            assert!(sigma.is_finite());
        }
    }

    #[test]
    fn feedback_pipeline_consciousness_finite() {
        let mut svc = make_service();
        let result = svc.cycle("pipeline test");
        assert!(result.metadata.pipeline_consciousness.is_finite());
    }

    #[test]
    fn feedback_cross_module_agreement_finite() {
        let mut svc = make_service();
        let result = svc.cycle("agreement test");
        assert!(result.metadata.cross_module_agreement.is_finite());
    }
}
