//! Feedback integration phase of the cognitive cycle.
//!
//! Extracts the post-processing feedback loops from the original `cycle()` method:
//! consciousness metrics, advanced subsystems, late consciousness monitors,
//! quality-aware adaptive processing, homeostasis, consciousness engine,
//! soul experience integration, cross-module agreement, quality fusion.

use std::time::Instant;

use super::cycle::{DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult};
use super::helpers::{DreamPhaseResult, EpisodicReplayResult, ResonatorCodebookResult};
use super::thresholds::{
    EPISTEMIC_APPROVAL_LR_SCALE, EPISTEMIC_APPROVAL_THRESHOLD, EPISTEMIC_CAUTION_SCALE,
    EPISTEMIC_CAUTION_THRESHOLD, EPISTEMIC_REJECTION_CONFIDENCE_SCALE, EPISTEMIC_REJECTION_LR_SCALE,
    EPISTEMIC_TRUST_SCALE, EPISTEMIC_TRUST_THRESHOLD, EVOLUTION_NEGATIVE_EXPLORATION_MAX,
    EVOLUTION_NEGATIVE_EXPLORATION_SCALE, EVOLUTION_PHI_THRESHOLD,
    EVOLUTION_POSITIVE_CONFIDENCE_MAX, EVOLUTION_POSITIVE_CONFIDENCE_SCALE,
    HARMONIC_ALL_CLEAR_BOOST, HARMONIC_INTERFERENCE_DAMPEN, HARMONIC_INTERFERENCE_MAX_COUNT,
    HARMONIC_INTERFERENCE_MAX_DAMPEN,
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
        let prediction_error = dynamics.prediction_error;
        let coherence = dynamics.coherence;
        let unified_psi = dynamics.unified_psi;

        // ═══════════════════════════════════════════════════════════════════════
        // CYCLE STATE: Shared read-only snapshot for extracted phase functions
        // ═══════════════════════════════════════════════════════════════════════
        let cycle_state = CycleState {
            compressed_state: &perception.compressed_state,
            output: &dynamics.output,
            prediction_error,
            coherence,
            unified_psi,
            phi_attention_weight: perception.phi_attention_weight,
            hv16_cached: &perception.hv16_cached,
            input,
            urgency: perception.urgency,
            attention_budget_exceeded: dynamics.attention_budget_exceeded,
            predictive_budget_gated: dynamics.predictive_budget_gated,
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
            * (1.0 - self.somatic_bridge.to_interoceptive_signals().dissipative_health_penalty);
        let dissipative_regime = consciousness_metrics.dissipative_regime;
        let dissipative_entropy_rate = consciousness_metrics.dissipative_entropy_rate;
        let epistemic_phi_eff = consciousness_metrics.epistemic_phi_eff;
        let epistemic_conflict_count = consciousness_metrics.epistemic_conflict_count;
        let equation_v2_consciousness = consciousness_metrics.equation_v2_consciousness;

        // ═══════════════════════════════════════════════════════════════════════
        // ADVANCED SUBSYSTEMS (extracted to cycle_subsystems.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let subsystem_metrics = self.run_advanced_subsystems(
            &cycle_state,
            &active_primitive_names,
            module_timings,
        );

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
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
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
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
                self.stats.harmonic_interference_mod_count += 1;
                -dampen
            } else if harmonic_interferences == 0 {
                self.carryover.learning.subsystem_lr_factor *= 1.0 + HARMONIC_ALL_CLEAR_BOOST;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
                self.stats.harmonic_interference_mod_count += 1;
                HARMONIC_ALL_CLEAR_BOOST
            } else {
                0.0
            };

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
            dynamics.attention_budget_exceeded && self.stats.attention_budget_exceeded_count > 3;

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
            let approval_boost =
                (epistemic_gate_confidence - EPISTEMIC_APPROVAL_THRESHOLD) * EPISTEMIC_APPROVAL_LR_SCALE;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + approval_boost;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
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
        } = if perception.input_memoized {
            ResonatorCodebookResult {
                resonator_promotions: 0,
                codebook_evictions: 0,
                codebook_diversity: self.stats.codebook_diversity,
                codebook_utilization_rate: self.stats.codebook_utilization_rate,
            }
        } else {
            self.run_resonator_codebook_phase(
                epistemic_gate_approved,
                &perception.compressed_state,
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
            dynamics.fep_surprise as f32, // memory_context_boost already handled
            dynamics.fep_surprise,
            surprise_thresh,
            module_timings,
        );
        dynamics.phasic_da_replay_boost = phasic_da_replay_boost;

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
        // Phase 11: DREAM ENGINE
        // ═══════════════════════════════════════════════════════════════════════
        let DreamPhaseResult {
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
        } = self.run_dream_phase(&cycle_state, &dynamics.prediction, module_timings);

        // 7 (deferred). Send prediction to encoder for next cycle
        // SAFETY: We take the prediction out of dynamics. The output phase must
        // not read dynamics.prediction after this point. We use std::mem::take
        // to move the Vec without allocation.
        self.encoder
            .set_prediction(std::mem::take(&mut dynamics.prediction));

        // ═══════════════════════════════════════════════════════════════════════
        // LATE CONSCIOUSNESS MONITORS
        // ═══════════════════════════════════════════════════════════════════════
        use super::cycle_late_consciousness::LateConsciousnessContext;

        let late_ctx = LateConsciousnessContext {
            prediction_error,
            coherence,
            unified_psi,
            hv16_cached: perception.hv16_cached.clone(),
            compressed_state: &perception.compressed_state,
            input,
            urgency: perception.urgency,
            moral_concern_detected: perception.moral_concern_detected,
            surprise_triggered: perception.surprise_triggered,
            reasoning_gate_blocked: dynamics.reasoning_gate_blocked,
            pp_phi: self.unification_engine.psi as f32,
            peak_attention: perception.encoding_result.peak_attention,
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
        let psi_attention_avg = late_result.psi_attention_avg;

        let gwt_broadcast = integration_result.gwt_broadcast;
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
            perception.exploration_urge_start,
            dynamics.homeostasis_pull_strength,
            dissipative_health,
            phenomenal_binding_strength,
            module_timings,
        );
        let dissipative_lr_factor = quality_result.dissipative_lr_factor;
        let dissipative_health_gated = quality_result.dissipative_health_gated;
        let coherence_velocity = quality_result.coherence_velocity;
        let coherence_velocity_gated = quality_result.coherence_velocity_gated;
        self.carryover.urgency.urgency = perception.urgency;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED CONSCIOUSNESS ENGINE
        // ═══════════════════════════════════════════════════════════════════════
        let encoding_hdv = &perception.encoding_result.hdv;
        let phi_spectral_weight = self.carryover.quality.phi_spectral_weight;
        let consciousness_output = self.consciousness_engine.measure(
            &super::consciousness_engine::ConsciousnessEngineInput {
                hdv: encoding_hdv,
                hv16: &perception.hv16_cached,
                cycle: self.stats.total_cycles as u64,
                unified_psi,
                coherence,
                prediction_error,
                phi_attention_weight: perception.phi_attention_weight,
                epistemic_quality: self.carryover.quality.last_epistemic_quality,
                phi_validation_correlation: self.carryover.quality.phi_validation_correlation,
                // Phase 6: bath → consciousness coupling
                bath_entropy: self.neuromod.phase_tracker.entropy(),
                attractor_detected: self.neuromod.phase_tracker.detect_attractor().is_some(),
                sht_2a_signal: self.neuromod.bath.sht_2a_signal(),
                gaba_a_signal: self.neuromod.bath.gaba_a_signal(),
                substrate_feasibility: self.substrate_feasibility,
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
        let spectral_mip_phi = consciousness_output.spectral_mip_phi;
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
                (
                    sp.micro_phi,
                    sp.meso_phi,
                    sp.macro_phi,
                    sp.bottleneck_score,
                    sp.emergence_ratio,
                    sp.num_clusters,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0)
            };
        let consciousness_weights = consciousness_output.current_weights;
        let consciousness_weight_variance = consciousness_output.weight_variance;

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
        if self.recent_ai_hvs.len() >= 2 {
            if let (Some(ref dyad), Some(ref model)) =
                (&self.phi_dyad, &self.partner_model)
            {
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
                    ai_states: &self.recent_ai_hvs,
                    human_states: &self.recent_input_hvs,
                    relational: &relational,
                    human_model: model,
                    weights: crate::partnership::DyadWeights::default(),
                };
                let result = dyad.compute(&input);
                self.social.relational_psi = result.phi_dyad;

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
        if let Some(ref mut model) = self.partner_model {
            let signal = (dynamics.coherence as f64 - 0.5) * 0.01;
            model.trust = ((model.trust as f64 + signal).clamp(0.0, 1.0) * 0.999) as f32;
        }

        // ── Track 4b: Cross-module agreement metric ─────────────────────────
        let fep_confidence = (1.0 - dynamics.fep_surprise.min(1.0)).max(0.0) as f32;
        let resonator_confidence = dynamics.resonator_best_sim;
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
        if cross_module_agreement > 0.8 {
            self.adjust_confidence("cross_mod_agree", (cross_module_agreement - 0.8) * 0.05);
        } else if cross_module_agreement < 0.3 {
            self.scale_confidence(
                "cross_mod_disagree",
                1.0 - (0.3 - cross_module_agreement) * 0.1,
            );
            self.adjust_exploration(
                "cross_module_disagree",
                (0.3 - cross_module_agreement) * 0.15,
            );
        }
        self.stats.avg_cross_module_agreement =
            self.stats.avg_cross_module_agreement * 0.95 + cross_module_agreement * 0.05;

        // ── Unified quality signal fusion ───────────────────────────
        let unified_quality_score;
        {
            let anomaly_factor = if dynamics.metacognitive_anomaly {
                0.0
            } else {
                1.0
            };
            unified_quality_score = 0.5 * dynamics.prediction_coherence
                + 0.3 * cross_module_agreement
                + 0.2 * anomaly_factor;
            self.stats.avg_unified_quality =
                self.stats.avg_unified_quality * 0.9 + unified_quality_score * 0.1;

            if unified_quality_score > 0.8 {
                let quality_boost = (unified_quality_score - 0.8) * 0.25;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + quality_boost;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.5);
            }
            if unified_quality_score < 0.3 && self.stats.total_cycles > 30 {
                self.scale_exploration("low_quality_dampen", 0.9);
            }
        }

        FeedbackPhaseResult {
            cross_module_agreement,
            unified_quality_score,
            coherence_velocity_gated,
            dissipative_health_gated,
            // Consciousness metrics pass-through
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
            value_evaluator_score,
            value_evaluator_decision,
            value_gate_factor,
            consciousness_profile_composite,
            synergy_enhanced_composite,
            emergent_properties_count,
            reasoning_context,
            context_phi_weight,
            context_phi_applied,
            value_embeddings_created,
            value_cache_hit_rate,
            harmonies_alignment,
            harmonies_approved,
            composition_rule_applied,
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
            reasoning_chain_confidence,
            reasoning_chain_depth,
            causal_relations_count,
            causal_avg_confidence,
            adaptive_reasoning_phi,
            epistemic_quality,
            phi_validation_correlation,
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
            epistemic_phi_eff,
            epistemic_conflict_count,
            equation_v2_consciousness,
            eq_v2_limiting_component,
            // Subsystem metrics pass-through
            hierarchical_ltc_phi,
            evolution_generation,
            evolution_phi_delta,
            evolution_confidence_delta,
            holographic_unity,
            holographic_binding,
            consciousness_gradient_magnitude,
            consciousness_limiting_component,
            affect_cons_valence,
            affect_cons_arousal,
            pipeline_consciousness,
            multimodal_integrated_phi,
            consciousness_state_label,
            consciousness_state_level,
            epistemic_gate_confidence,
            epistemic_gate_approved,
            primitive_validation_phi_gain,
            primitive_validation_p_value,
            meta_reasoning_confidence,
            meta_reasoning_insights,
            code_primitives_selected,
            empathic_compassion,
            empathic_tone_adj,
            multi_obj_frontier_size,
            grid_encoding_norm,
            grid_spatial_complexity,
            // Late consciousness pass-through
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
            psi_attention_avg,
            gwt_broadcast,
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
            consciousness_level,
            // Additional pass-through
            spectral_mip_phi,
            sigma,
            phi_spectral_weight,
            dissipative_lr_factor,
            coherence_velocity,
            empathic_speech_rate_mod,
            limiting_component_boosted: limiting_component_boosted.into(),
            love_resonance_boost,
            reasoning_chain_boosted,
            harmonic_interference_lr_mod,
            causal_urgency_gated,
            epistemic_coherence_gated,
            attention_budget_gated,
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
            resonator_promotions,
            codebook_evictions,
            codebook_diversity,
            codebook_utilization_rate,
            surprise_replay_batch_size,
            support_triage_count,
            support_alert_fired,
            support_federation_graduated,
            support_efe,
            // Structural Phi decomposition
            structural_micro_phi: struct_micro,
            structural_meso_phi: struct_meso,
            structural_macro_phi: struct_macro,
            structural_bottleneck: struct_bn,
            structural_emergence_ratio: struct_er,
            structural_num_clusters: struct_nc,
            // Dynamic consciousness weights
            consciousness_weights,
            consciousness_weight_variance,
        }
    }
}
