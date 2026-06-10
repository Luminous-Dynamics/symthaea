// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feedback integration phase of the cognitive cycle.
//!
//! Extracts the post-processing feedback loops from the original `cycle()` method:
//! consciousness metrics, advanced subsystems, late consciousness monitors,
//! quality-aware adaptive processing, homeostasis, consciousness engine,
//! soul experience integration, cross-module agreement, quality fusion.

use std::time::Instant;

use super::feedback_state::Priority;
use super::helpers::{DreamPhaseResult, EpisodicReplayResult, ResonatorCodebookResult};
use super::phase_results::{
    DynamicsPhaseResult, FbConsciousness, FbEthics, FbEvolution, FbLoops, FbMemory, FbQuality,
    FbReasoning, FbSelfModel, FbSupport, FeedbackPhaseResult, PerceptionPhaseResult,
};
use super::thresholds::{
    AGREEMENT_COHERENCE_VELOCITY_THRESHOLD,
    AGREEMENT_CONFIDENCE_COUPLING_SCALE,
    AGREEMENT_CONFIDENCE_COUPLING_THRESHOLD,
    AGREEMENT_CRITICAL_CAUTION_SCALE,
    AGREEMENT_CRITICAL_THRESHOLD,
    AGREEMENT_EMA_DECAY,
    AGREEMENT_HIGH_CONFIDENCE_SCALE,
    AGREEMENT_LOW_CONFIDENCE_SCALE,
    AGREEMENT_LOW_EXPLORATION_SCALE,
    AGREEMENT_VELOCITY_DROP_EXPLORATION,
    AGREEMENT_VELOCITY_DROP_LR,
    AGREEMENT_VELOCITY_DROP_THRESHOLD,
    ATTENTION_BUDGET_GATED_LR_SCALE,
    CAUSAL_URGENCY_CONFIDENCE,
    COHERENCE_TRUST_CENTER,
    COMPOUND_INSTABILITY_ERROR_SLOPE,
    COMPOUND_INSTABILITY_EXPLORATION,
    COMPOUND_INSTABILITY_LR_SCALE,
    COMPOUND_INSTABILITY_VELOCITY,
    // Round 17: feedback magic number extraction
    CONSCIOUSNESS_GRADIENT_STABILITY_THRESHOLD,
    CONTEXT_PHI_SCALE_BASE,
    CONTEXT_PHI_SCALE_RANGE,
    // Round 22: psi→neuromod + epistemic gate constants
    CPG_SYNC_DEFAULT,
    CROSS_MODAL_BINDING_ALPHA,
    CROSS_MODAL_BINDING_HIGH_SCALE,
    CROSS_MODAL_BINDING_HIGH_THRESHOLD,
    CROSS_MODAL_BINDING_LOW_FLOOR,
    CROSS_MODAL_BINDING_LOW_SCALE,
    CROSS_MODAL_BINDING_LOW_THRESHOLD,
    CROSS_MODAL_BINDING_MOMENTUM,
    CROSS_MODULE_AGREEMENT_ADJUSTMENT_CENTER,
    CROSS_MODULE_AGREEMENT_HIGH,
    CROSS_MODULE_AGREEMENT_LOW,
    CROSS_MODULE_AGREEMENT_NEUTRAL,
    CROSS_MODULE_VARIANCE_AMPLIFICATION,
    EFFICACY_ATTENTION_BOOST,
    EFFICACY_LR_BOOST,
    EMPATHIC_TONE_RATE_SCALE,
    EMPATHIC_TONE_THRESHOLD,
    ENTROPY_LR_MIN,
    ENTROPY_LR_RANGE,
    EPISTEMIC_APPROVAL_LR_SCALE,
    EPISTEMIC_APPROVAL_THRESHOLD,
    EPISTEMIC_CAUTION_SCALE,
    EPISTEMIC_CAUTION_THRESHOLD,
    EPISTEMIC_CONFLICT_EXPLORATION_SCALE,
    EPISTEMIC_REJECTION_CLAMP_MAX,
    EPISTEMIC_REJECTION_CONFIDENCE_SCALE,
    EPISTEMIC_REJECTION_LR_SCALE,
    EPISTEMIC_TRUST_SCALE,
    EPISTEMIC_TRUST_THRESHOLD,
    EVOLUTION_NEGATIVE_EXPLORATION_MAX,
    EVOLUTION_NEGATIVE_EXPLORATION_SCALE,
    EVOLUTION_PHI_THRESHOLD,
    EVOLUTION_POSITIVE_CONFIDENCE_MAX,
    EVOLUTION_POSITIVE_CONFIDENCE_SCALE,
    FLOW_STATE_LR_BOOST_MULTIPLIER,
    HARMONIC_ALL_CLEAR_BOOST,
    HARMONIC_INTERFERENCE_DAMPEN,
    HARMONIC_INTERFERENCE_FREE_CYCLES,
    HARMONIC_INTERFERENCE_MAX_COUNT,
    HARMONIC_INTERFERENCE_MAX_DAMPEN,
    HOT_DEPTH_DEFAULT,
    HOT_HUBRIS_CONFIDENCE_DAMPEN,
    KNOWLEDGE_CONTRADICTION_FACTOR_DENOM,
    KNOWLEDGE_REASONING_LOG_SCALE,
    LIMITING_BINDING_CONFIDENCE_DELTA,
    LIMITING_COMPONENT_GRADIENT_THRESHOLD,
    LOVE_RESONANCE_CONFIDENCE_SCALE,
    LOVE_RESONANCE_LR_FRACTION,
    LOVE_RESONANCE_THRESHOLD,
    LOW_QUALITY_EXPLORATION_DAMPEN,
    MICRO_PHI_SCALE_BOOST_FACTOR,
    MORAL_SCORE_NORMALIZE_OFFSET,
    MORAL_SCORE_NORMALIZE_SCALE,
    NEUROMOD_ATTENTION_SENSITIVITY_MAX,
    PHENOMENAL_FRAGMENTED_CONFIDENCE_DAMPEN,
    PHENOMENAL_FRAGMENTED_EXPLORATION_BOOST,
    PHI_DIVERGENCE_MAX,
    PHI_DIVERGENCE_SCALE,
    PHI_DIVERGENCE_THRESHOLD,
    PHI_RELATIONAL_OXY_SCALE,
    PHI_RELATIONAL_OXY_THRESHOLD,
    PIPELINE_CONSCIOUSNESS_CAUTION,
    PIPELINE_CONSCIOUSNESS_CAUTION_SCALE,
    PIPELINE_CONSCIOUSNESS_RELAX,
    PIPELINE_CONSCIOUSNESS_RELAX_SCALE,
    PSI_5HT_CAP,
    PSI_5HT_SCALE,
    PSI_5HT_THRESHOLD,
    PSI_DA_CAP,
    PSI_DA_SCALE,
    PSI_DA_THRESHOLD,
    PSI_NE_CAP,
    PSI_NE_SCALE,
    PSI_NE_THRESHOLD,
    QUALITY_EMA_DECAY,
    QUALITY_FLOOR_EXPLORATION_ADJUSTMENT,
    QUALITY_HIGH_LR_SCALE,
    QUALITY_LR_CLAMP_MAX,
    QUALITY_LR_CLAMP_MIN,
    REASONING_CHAIN_BOOST_SCALE,
    REASONING_CHAIN_CONFIDENCE_THRESHOLD,
    REASONING_GATE_SUCCESS_LR_SCALE,
    REASONING_RELIABILITY_CENTER,
    REST_BINDING_DAMPEN,
    REST_COHERENCE_WEIGHT,
    REST_MODULATION_BINDING_FRAC,
    REST_MODULATION_COHERENCE_FRAC,
    SCALE_BOOST_SIGMOID_AMPLITUDE,
    SCALE_BOOST_SIGMOID_EXPONENT,
    SOCIAL_LR_BASE,
    SOCIAL_LR_CHANGE_THRESHOLD,
    SOCIAL_LR_RANGE,
    SOCIAL_MODEL_TRUST_MODE_THRESHOLD,
    SOCIAL_MODEL_TURN_TAKING_DEFAULT,
    SPEECH_RATE_CLAMP_MAX,
    SPEECH_RATE_CLAMP_MIN,
    STRUCTURAL_BOTTLENECK_LR_SCALE,
    STRUCTURAL_BOTTLENECK_THRESHOLD,
    STRUCTURAL_EMERGENCE_CONFIDENCE_BOOST,
    STRUCTURAL_EMERGENCE_CONFIDENCE_THRESHOLD,
    SUBSYSTEM_LR_FACTOR_MAX,
    SUBSYSTEM_LR_FACTOR_MIN,
    SUPPORT_GRADUATION_INTERVAL,
    SUPPORT_TELEMETRY_INTERVAL,
    TEMPORAL_BINDING_HIGH_EXPLORATION_SCALE,
    TEMPORAL_CHAIN_DEEP_LR_SCALE,
    TEMPORAL_CHAIN_DEEP_THRESHOLD,
    TEMPORAL_CHAIN_SHALLOW_LR_SCALE,
    TEMPORAL_CHAIN_SHALLOW_THRESHOLD,
    TEMPORAL_DISCONTINUITY_EXPLORATION_BOOST,
    TEMPORAL_DISCONTINUITY_LR_DAMPEN,
    TOM_ACCURACY_HIGH,
    TOM_ACCURACY_LOW,
    TOM_ACCURACY_SCALE,
    TRUST_DECAY_FACTOR,
    TRUST_SIGNAL_MIDPOINT,
    TRUST_SIGNAL_RATE,
    UNIFIED_QUALITY_AGREEMENT_WEIGHT,
    UNIFIED_QUALITY_ANOMALY_WEIGHT,
    UNIFIED_QUALITY_HIGH_THRESHOLD,
    UNIFIED_QUALITY_PREDICTION_WEIGHT,
};
#[cfg(feature = "vision-manifold")]
use super::types::MentalMovie;
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
            #[cfg(feature = "semantic-encoder")]
            semantic_embedding: self.feature_integ.last_semantic_continuous.clone(),
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
        // chain_cycle_numbers: computed upstream but not consumed in feedback phase
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
        let context_phi_applied =
            context_phi_weight > 0.0 && (context_phi_weight - 1.0).abs() > f64::EPSILON;
        if context_phi_applied {
            let scale =
                CONTEXT_PHI_SCALE_BASE as f64 + context_phi_weight * CONTEXT_PHI_SCALE_RANGE as f64;
            let adjusted_psi = (unified_psi * scale).clamp(0.0, 1.0);
            self.unification_engine.update_psi(adjusted_psi);
            self.stats.context_phi_applied_count += 1;
        }

        // ── Temporal chain depth → LR gating ──────────────────────────────
        // Deep causal chains need stable consolidation (dampen LR);
        // shallow chains need rapid hypothesis testing (boost LR).
        // Science: Zelazo (2004) — cognitive complexity demands stable representations;
        // Gopnik (2012) — shallow causal models benefit from rapid exploration.
        if temporal_max_chain_length >= TEMPORAL_CHAIN_DEEP_THRESHOLD
            && self.stats.total_cycles > 15
        {
            self.scale_lr("deep_causal_chain", TEMPORAL_CHAIN_DEEP_LR_SCALE);
        } else if temporal_max_chain_length > 0
            && temporal_max_chain_length <= TEMPORAL_CHAIN_SHALLOW_THRESHOLD
            && self.stats.total_cycles > 15
        {
            self.scale_lr("shallow_causal_chain", TEMPORAL_CHAIN_SHALLOW_LR_SCALE);
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
                    .sensorimotor
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

        // Session 16 Item 7: Multi-objective frontier size → exploration gating.
        // Large Pareto frontier = many competing objectives → boost exploration.
        // Single-point frontier = converged → dampen exploration.
        // Science: Deb (2002) — Pareto frontier diversity signals solution space richness.
        {
            use super::thresholds::{
                MULTI_OBJ_FRONTIER_DAMPEN, MULTI_OBJ_FRONTIER_EXPLORE_SCALE,
                MULTI_OBJ_FRONTIER_LARGE, MULTI_OBJ_FRONTIER_SMALL,
            };
            if multi_obj_frontier_size >= MULTI_OBJ_FRONTIER_LARGE && self.stats.total_cycles > 15 {
                let boost = (multi_obj_frontier_size - MULTI_OBJ_FRONTIER_LARGE + 1) as f32
                    * MULTI_OBJ_FRONTIER_EXPLORE_SCALE;
                self.adjust_exploration("frontier_large", boost.min(0.1));
            } else if multi_obj_frontier_size <= MULTI_OBJ_FRONTIER_SMALL
                && multi_obj_frontier_size > 0
                && self.stats.total_cycles > 30
            {
                self.scale_exploration("frontier_converged", MULTI_OBJ_FRONTIER_DAMPEN);
            }
        }

        // ── Phase 18: Empathic tone → speech rate modulation ─────────────────
        let empathic_speech_rate_mod = if empathic_tone_adj.abs() > EMPATHIC_TONE_THRESHOLD as f64 {
            let rate_mod = 1.0 - empathic_tone_adj as f32 * EMPATHIC_TONE_RATE_SCALE;
            self.behavior.adaptive_behavior.speech_rate_multiplier *= rate_mod;
            self.behavior.adaptive_behavior.speech_rate_multiplier = self
                .behavior
                .adaptive_behavior
                .speech_rate_multiplier
                .clamp(SPEECH_RATE_CLAMP_MIN, SPEECH_RATE_CLAMP_MAX);
            empathic_tone_adj as f32
        } else {
            0.0
        };

        // ── Phase 19: Consciousness limiting component → targeted boost ─────
        let limiting_component_boosted = if !consciousness_limiting_component.is_empty()
            && consciousness_gradient_magnitude > LIMITING_COMPONENT_GRADIENT_THRESHOLD
        {
            match consciousness_limiting_component.as_str() {
                "Attention" => {
                    self.behavior.adaptive_behavior.attention_sensitivity =
                        (self.behavior.adaptive_behavior.attention_sensitivity
                            * EFFICACY_ATTENTION_BOOST)
                            .min(NEUROMOD_ATTENTION_SENSITIVITY_MAX);
                    self.stats.limiting_component_boost_count += 1;
                    "Attention"
                }
                "Binding" => {
                    self.adjust_confidence("limit_binding", LIMITING_BINDING_CONFIDENCE_DELTA);
                    self.stats.limiting_component_boost_count += 1;
                    "Binding"
                }
                "Efficacy" => {
                    self.scale_lr("limit_efficacy", EFFICACY_LR_BOOST);
                    self.stats.limiting_component_boost_count += 1;
                    "Efficacy"
                }
                _ => "",
            }
        } else {
            ""
        };

        // Session 16 Item 2: Consciousness gradient magnitude → stability recovery.
        // Large gradient = rapid consciousness change → cautious LR dampening.
        // Near-zero gradient sustained → stable integration → confidence recovery.
        // Science: Oizumi et al. (2014) — consciousness gradient tracks integration dynamics.
        {
            use super::thresholds::{
                CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD, CONSCIOUSNESS_GRADIENT_LR_SCALE,
                CONSCIOUSNESS_GRADIENT_RECOVERY_BOOST, CONSCIOUSNESS_GRADIENT_STABLE_CYCLES,
            };
            if consciousness_gradient_magnitude > CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD
                && self.stats.total_cycles > 15
            {
                self.scale_lr("gradient_caution", CONSCIOUSNESS_GRADIENT_LR_SCALE);
                self.carryover.quality.consecutive_stable_gradient = 0;
            } else if consciousness_gradient_magnitude
                < CONSCIOUSNESS_GRADIENT_STABILITY_THRESHOLD as f64
            {
                self.carryover.quality.consecutive_stable_gradient = self
                    .carryover
                    .quality
                    .consecutive_stable_gradient
                    .saturating_add(1);
                if self.carryover.quality.consecutive_stable_gradient
                    > CONSCIOUSNESS_GRADIENT_STABLE_CYCLES
                    && self.stats.total_cycles > 20
                {
                    self.adjust_confidence(
                        "gradient_stable_recovery",
                        CONSCIOUSNESS_GRADIENT_RECOVERY_BOOST,
                    );
                }
            } else {
                self.carryover.quality.consecutive_stable_gradient = 0;
            }
        }

        // ── Phase 19: Harmonic love resonance → confidence/soul amplifier ────
        let love_resonance_boost = if harmonic_love_resonance > LOVE_RESONANCE_THRESHOLD as f64 {
            let boost = ((harmonic_love_resonance - LOVE_RESONANCE_THRESHOLD as f64)
                * LOVE_RESONANCE_CONFIDENCE_SCALE as f64) as f32;
            self.adjust_confidence_pri("love_resonance", boost, Priority::Aesthetic);
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost * LOVE_RESONANCE_LR_FRACTION;
            self.carryover.learning.subsystem_lr_factor = self
                .carryover
                .learning
                .subsystem_lr_factor
                .clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
            self.stats.love_resonance_boost_count += 1;
            boost
        } else {
            0.0
        };

        // ── Phase 19: Reasoning chain confidence + depth → confidence ────────
        let reasoning_chain_boosted = reasoning_chain_confidence
            > REASONING_CHAIN_CONFIDENCE_THRESHOLD
            && reasoning_chain_depth >= 3;
        if reasoning_chain_boosted {
            let chain_boost = (reasoning_chain_confidence - REASONING_CHAIN_CONFIDENCE_THRESHOLD)
                * REASONING_CHAIN_BOOST_SCALE;
            self.adjust_confidence("reasoning_chain", chain_boost);
            self.stats.reasoning_chain_boost_count += 1;
        }

        // ── Phase 20: Harmonic interferences → LR feedback ───────────────────
        let harmonic_interference_lr_mod: f32 = if harmonic_interferences
            > HARMONIC_INTERFERENCE_MAX_COUNT
        {
            let dampen = ((harmonic_interferences - HARMONIC_INTERFERENCE_MAX_COUNT) as f32
                * HARMONIC_INTERFERENCE_DAMPEN)
                .min(HARMONIC_INTERFERENCE_MAX_DAMPEN);
            self.carryover.learning.subsystem_lr_factor *= 1.0 - dampen;
            self.carryover.learning.subsystem_lr_factor = self
                .carryover
                .learning
                .subsystem_lr_factor
                .clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
            self.stats.harmonic_interference_mod_count += 1;
            self.carryover.quality.interference_free_cycles = 0;
            -dampen
        } else if harmonic_interferences == 0 {
            self.carryover.quality.interference_free_cycles = self
                .carryover
                .quality
                .interference_free_cycles
                .saturating_add(1);
            // Session 13 Item 2: Grace period before harmonic all-clear boost.
            // Require 3 consecutive interference-free cycles to prevent LR whiplash.
            // Science: Kelso (1995) — stability requires sustained absence of perturbation.
            if self.carryover.quality.interference_free_cycles >= HARMONIC_INTERFERENCE_FREE_CYCLES
            {
                self.carryover.learning.subsystem_lr_factor *= 1.0 + HARMONIC_ALL_CLEAR_BOOST;
                self.carryover.learning.subsystem_lr_factor = self
                    .carryover
                    .learning
                    .subsystem_lr_factor
                    .clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
                self.stats.harmonic_interference_mod_count += 1;
                HARMONIC_ALL_CLEAR_BOOST
            } else {
                0.0
            }
        } else {
            self.carryover.quality.interference_free_cycles = 0;
            0.0
        };

        // ── Social trust → learning rate modulation (Decety & Chaminade 2003) ──
        let social_learning_rate_factor =
            SOCIAL_LR_BASE + SOCIAL_LR_RANGE * self.behavior.social_mgr.social.social_trust; // [0.8, 1.2]
        if (social_learning_rate_factor - 1.0).abs() > SOCIAL_LR_CHANGE_THRESHOLD {
            self.scale_lr("social_trust", social_learning_rate_factor);
        }

        // ── ToM accuracy → prediction confidence modulation (Frith & Frith 2006) ──
        // High social prediction accuracy → boost prediction confidence (we understand the user).
        // Low accuracy → dampen confidence (our model is unreliable).
        // Guard: only active when social models exist (avoid constant dampening
        // when no social context has been injected — default accuracy is 0.0).
        if self.behavior.social_mgr.social.social_models_count > 0 {
            let tom_accuracy = self.behavior.social_mgr.social.social_prediction_accuracy;
            if tom_accuracy > TOM_ACCURACY_HIGH {
                let boost = (tom_accuracy - TOM_ACCURACY_HIGH) * TOM_ACCURACY_SCALE; // [0, 0.015]
                self.adjust_confidence("tom_accurate", boost);
            } else if tom_accuracy < TOM_ACCURACY_LOW && self.stats.total_cycles > 10 {
                let dampen = 1.0 - (TOM_ACCURACY_LOW - tom_accuracy) * TOM_ACCURACY_SCALE; // [0.985, 1.0]
                self.scale_confidence("tom_inaccurate", dampen);
            }
        }

        // ── Phase 20: Causal relations density → urgency gating ──────────────
        let causal_urgency_gated = causal_relations_count > 10
            && causal_avg_confidence > CAUSAL_URGENCY_CONFIDENCE as f64
            && self.stats.total_cycles > 20;
        if causal_urgency_gated {
            self.carryover.urgency.consecutive_low_error = self
                .carryover
                .urgency
                .consecutive_low_error
                .saturating_add(2);
            self.stats.causal_urgency_gated_count += 1;
        }

        let attention_budget_gated = dynamics.attention.attention_budget_exceeded
            && self.stats.attention_budget_exceeded_count > 3;
        // Attention budget persistently exceeded → slow learning (cognitive overload).
        // Science: Lavie (2005) — perceptual load theory: high load degrades encoding.
        if attention_budget_gated {
            self.scale_lr("attention_budget_gated", ATTENTION_BUDGET_GATED_LR_SCALE);
        }

        // ── Track 5a: Epistemic gate → actual information gating ─────────────
        let mut epistemic_coherence_gated = false;
        if !epistemic_gate_approved {
            let rejection_strength =
                (1.0 - epistemic_gate_confidence).clamp(0.0, EPISTEMIC_REJECTION_CLAMP_MAX);
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
            self.carryover.learning.subsystem_lr_factor = self
                .carryover
                .learning
                .subsystem_lr_factor
                .clamp(SUBSYSTEM_LR_FACTOR_MIN, SUBSYSTEM_LR_FACTOR_MAX);
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

        // Session 16 Item 4: Epistemic gate rejection streak → recalibration.
        // Consecutive rejections signal systematic model failure. After threshold,
        // boost exploration and relax thresholds to let new information through.
        // Science: Berlyne (1960) — sustained rejection = need for epistemic recalibration.
        if !epistemic_gate_approved {
            self.carryover.quality.consecutive_epistemic_rejections = self
                .carryover
                .quality
                .consecutive_epistemic_rejections
                .saturating_add(1);
        } else {
            self.carryover.quality.consecutive_epistemic_rejections = 0;
        }
        if self.carryover.quality.consecutive_epistemic_rejections
            >= super::thresholds::EPISTEMIC_REJECTION_STREAK_THRESHOLD
            && self.stats.total_cycles > 20
        {
            self.adjust_exploration(
                "epistemic_streak_recal",
                super::thresholds::EPISTEMIC_REJECTION_STREAK_EXPLORE,
            );
            self.scale_threshold(
                "epistemic_streak_recal",
                super::thresholds::EPISTEMIC_REJECTION_STREAK_THRESHOLD_RELAX,
            );
        }

        // Session 15 Item 1: Pipeline consciousness → epistemic caution modulation.
        // High pipeline consciousness → relax epistemic caution (system is integrated).
        // Low pipeline consciousness → tighten caution (subsystems aren't coherent).
        // Science: Dehaene (2014) — global workspace ignition requires integrated processing.
        self.carryover.quality.last_pipeline_consciousness = pipeline_consciousness;
        if pipeline_consciousness > PIPELINE_CONSCIOUSNESS_RELAX as f64
            && self.stats.total_cycles > 15
        {
            self.scale_threshold(
                "pipeline_conscious_relax",
                PIPELINE_CONSCIOUSNESS_RELAX_SCALE,
            );
        } else if pipeline_consciousness < PIPELINE_CONSCIOUSNESS_CAUTION as f64
            && pipeline_consciousness > 0.0
            && self.stats.total_cycles > 15
        {
            self.scale_threshold(
                "pipeline_conscious_caution",
                PIPELINE_CONSCIOUSNESS_CAUTION_SCALE,
            );
        }

        // ═══════════════════════════════════════════════════════════════════════
        // RESONATOR CODEBOOK GROWTH
        // ═══════════════════════════════════════════════════════════════════════
        let reflection_thresholds = self
            .consciousness
            .self_model_tier
            .self_reflection
            .get_thresholds();
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
            memory_db_flushed,
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
            self.support.cycle_counter += 1;

            let mut triage_count: u32 = 0;
            if let Some(ref engine) = self.support.triage_engine {
                let result = engine.triage(input, "");
                triage_count = 1;
                if let Some(ref manager) = self.support.memory.knowledge_manager {
                    let category_str = result.suggested_category.as_str();
                    let articles = manager.search(category_str, 3);
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
            if self.support.cycle_counter % SUPPORT_TELEMETRY_INTERVAL == 0 {
                if let Some(ref engine) = self.support.predictive_engine {
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
            if self.support.cycle_counter % SUPPORT_GRADUATION_INTERVAL == 0 {
                let can_share = self
                    .support
                    .privacy_manager
                    .as_ref()
                    .map(|pm| pm.can_share_cognitive())
                    .unwrap_or(true);

                if can_share {
                    if let Some(ref manager) = self.support.memory.knowledge_manager {
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
        self.carryover.consciousness.last_moral_fluctuatio_tension =
            perception.moral.moral_fluctuatio_tension;
        let predictive_psi_modulation = late_result.predictive_psi_modulation;
        let hierarchical_total_free_energy = late_result.hierarchical_total_free_energy;
        let predictive_self_safety = late_result.predictive_self_safety;
        let predictive_behavioral_error = late_result.predictive_behavioral_error;
        let attention_schema_focus = late_result.attention_schema_focus;
        let attention_fatigue = late_result.attention_fatigue;
        let attention_prediction_accuracy = late_result.attention_prediction_accuracy;
        let psi_attention_avg = late_result.psi_attention_avg;
        let hierarchical_free_energy_lr_boost = late_result.hierarchical_free_energy_lr_boost;
        let predictive_phi_lr_delta = late_result.predictive_phi_lr_delta;
        let body_valence_confidence_delta = late_result.body_valence_confidence_delta;
        let narrative_self_confidence_factor = late_result.narrative_self_confidence_factor;

        let gwt_broadcast = integration_result.gwt_broadcast;
        let gwt_coalition_size = integration_result.gwt_coalition_size;
        let mut cross_modal_binding_strength = integration_result.cross_modal_binding_strength;
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
        let consciousness_output = self.consciousness.consciousness_engine.measure(
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
                // GWT broadcast state → Workspace component
                gwt_broadcast_occurred: self.carryover.gwt_broadcast_occurred,
                gwt_coalition_size: self.carryover.gwt_coalition_size,
                // Prediction precision (inverse variance of recent PE)
                prediction_precision: {
                    let pe_history = &self.carryover.history.error_history;
                    if pe_history.len() >= 4 {
                        let mean: f32 = pe_history.iter().sum::<f32>() / pe_history.len() as f32;
                        let variance: f32 =
                            pe_history.iter().map(|e| (e - mean).powi(2)).sum::<f32>()
                                / pe_history.len() as f32;
                        (1.0 / (variance + 1e-6)).clamp(0.1, 10.0)
                    } else {
                        1.0 // neutral precision when insufficient history
                    }
                },
                // Moral topology → consciousness coupling
                moral_drift: self.ethics_engine.moral_topology().moral_drift(20),
                moral_anomaly_score: self.ethics_engine.last_anomaly_report().anomaly_score,
                // HOT recursion depth: meta_cognition depth × substrate HOT capability
                // Attenuated by moral hubris: overconfidence undermines self-knowledge.
                // Basis: Kruger & Dunning (1999) — miscalibrated self-assessment.
                hot_depth: {
                    let raw_hot = self
                        .consciousness
                        .self_model_tier
                        .meta_cognition
                        .as_ref()
                        .map(|mc| {
                            // ── Continuous HOT depth (Brown, Lau & LeDoux 2019) ──
                            // When `continuous_hot` is enabled, blend discrete recursion
                            // depth with meta-cognitive accuracy for sub-level resolution,
                            // and enrich with narrative self-Φ (self-integration deepens
                            // recursive awareness) and attention schema focus (Graziano 2019:
                            // consciousness IS the model of attention).
                            #[cfg(feature = "continuous_hot")]
                            {
                                let discrete = mc.depth() as f64 / 3.0;
                                let accuracy = mc.accuracy() as f64;

                                // Blend: 60% discrete depth + 20% meta-cognitive accuracy
                                let mut continuous = discrete * 0.6 + accuracy * 0.2;

                                // Self-Phi enrichment: narrative integration deepens recursion
                                let self_phi = narrative_self_psi; // from late_result
                                continuous += 0.1 * self_phi.clamp(0.0, 1.0);

                                // Attention schema: focus intensity as HOT modulator
                                continuous += 0.1 * (attention_schema_focus as f64).clamp(0.0, 1.0);

                                // Scale by substrate HOT capability
                                continuous.clamp(0.0, 1.0)
                                    * self.substrate_manager.hot_capability(&self.config)
                            }

                            #[cfg(not(feature = "continuous_hot"))]
                            {
                                let normalized_depth = mc.depth() as f64 / 3.0;
                                normalized_depth
                                    * self.substrate_manager.hot_capability(&self.config)
                            }
                        })
                        .unwrap_or(HOT_DEPTH_DEFAULT); // preserve backward compat when disabled
                    // Hubris attenuates HOT: can't claim deep self-knowledge while
                    // morally overconfident. 0.7× during hubris, 1.0× otherwise.
                    if self.ethics_engine.last_anomaly_report().moral_hubris {
                        raw_hot * HOT_HUBRIS_CONFIDENCE_DAMPEN as f64
                    } else {
                        raw_hot
                    }
                },
                // CPG sync → consciousness coupling (Varela et al. 2001)
                cpg_sync_index: {
                    #[cfg(feature = "cpg")]
                    {
                        self.cpg_manager.sync_index()
                    }
                    #[cfg(not(feature = "cpg"))]
                    {
                        0.5
                    } // Neutral: no modulation when CPG disabled
                },
                // Cantor metacognitive depth: derived from dream surprise EMA
                // Higher surprise = richer fractal structure = deeper self-reference
                cantor_metacognitive_depth: (self.cantor_dream.dream_surprise as f64)
                    .clamp(0.0, 1.0),
                // Governance collective Phi: inter-agent consciousness integration
                governance_collective_phi: {
                    #[cfg(feature = "mycelix")]
                    {
                        self.governance_mgr.last_collective_phi()
                    }
                    #[cfg(not(feature = "mycelix"))]
                    {
                        0.0
                    }
                },
                // Knowledge grounding: dynamic from KnowledgeManager signals
                // Science: Barsalou (2008), Clark (2013) — grounded cognition modulates consciousness
                knowledge_grounding: if let Some(ref km) = self.memory.knowledge_manager {
                    let s = km.signals();
                    let grounding = (s.relevance
                        * super::thresholds::KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT
                        + (1.0 - s.uncertainty)
                            * super::thresholds::KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT)
                        .clamp(0.0, 1.0);
                    if grounding.is_finite() {
                        grounding
                    } else {
                        0.5
                    }
                } else {
                    0.5
                },
                // Knowledge coherence: composite quality from graph size, calibration, contradictions.
                // Formula: (log2(graph_size+1)/10) × (1-ece) × (1/(1 + contradictions×0.1))
                // Science: Stanovich (2009) — epistemic rationality; Guo et al. (2017) — calibration.
                knowledge_coherence: if let Some(ref km) = self.memory.knowledge_manager {
                    let t = km.telemetry();
                    let log_scale = super::thresholds::KNOWLEDGE_COHERENCE_LOG_SCALE;
                    let size_factor =
                        ((t.graph_size as f64 + 1.0).log2() / log_scale).clamp(0.0, 1.0);
                    let ece_factor = (1.0 - t.calibration_ece).clamp(0.0, 1.0);
                    let contradiction_factor = 1.0
                        / (1.0
                            + t.contradictions_detected as f64
                                * KNOWLEDGE_CONTRADICTION_FACTOR_DENOM as f64);
                    let coherence = size_factor * ece_factor * contradiction_factor;
                    if coherence.is_finite() {
                        coherence
                    } else {
                        0.0
                    }
                } else {
                    0.0
                },
                // Glyph coherence: symbolic consciousness field integration
                glyph_coherence: {
                    #[cfg(feature = "glyph_codex")]
                    {
                        self.glyph_manager.last_coherence().value
                    }
                    #[cfg(not(feature = "glyph_codex"))]
                    {
                        0.0
                    }
                },
                // CfC temporal coherence → consciousness (Clark 2013)
                temporal_coherence_phi: self
                    .language_comm
                    .voice_coherence
                    .bridge
                    .phi_contribution(),
            },
        );
        self.consciousness
            .consciousness_engine
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
            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                replay.boost_recent_consolidation(consolidation_boost);
            }
        }
        // ── Knowledge Engine: PE → ontology learning rate ────────────────
        // High prediction error → faster ontology adaptation (Rescorla-Wagner 1972).
        if let Some(ref mut km) = self.memory.knowledge_manager {
            km.set_ontology_lr_from_pe(prediction_error as f32);
            km.modulate_lr_from_consciousness(self.stats.unified_psi as f64);
        }

        // Theta phase → Phi modulation (Buzsáki 2006).
        // 6Hz theta oscillation at 50Hz loop rate. Peaks enhance integration; troughs suppress.
        // EMA-smoothed to prevent 6Hz artifacts in downstream consciousness metrics.
        let theta_phase = (self.stats.total_cycles as f64 * super::thresholds::THETA_PHASE_ADVANCE)
            % (2.0 * std::f64::consts::PI);
        let theta_phi_mod = theta_phase.sin() * super::thresholds::THETA_PHI_MODULATION_AMPLITUDE;
        let spectral_mip_phi = consciousness_output.spectral_mip_phi.map(|phi| {
            let raw = (phi * (1.0 + theta_phi_mod)).max(0.0);
            // EMA smooth to prevent rhythmic artifacts in consciousness telemetry.
            let alpha = super::thresholds::THETA_PHI_SMOOTH_ALPHA;
            let prev = self
                .carryover
                .consciousness
                .last_spectral_mip_phi
                .unwrap_or(raw);
            prev * (1.0 - alpha) + raw * alpha
        });
        let sigma = consciousness_output.sigma;
        let eq_v2_limiting_component = consciousness_output
            .limiting_component
            .map(|c| c.as_str().to_string())
            .unwrap_or_default();
        self.carryover.quality.last_pipeline_consciousness =
            consciousness_output.pipeline_consciousness;
        // Override stale subsystems value with fresh consciousness engine output
        // (subsystems reads carryover BEFORE the engine updates it, causing 1-cycle lag)
        let pipeline_consciousness = consciousness_output.pipeline_consciousness;
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
                    let mean_tau = taus.iter().sum::<f32>() / taus.len().max(1) as f32;
                    if mean_tau > 0.0 && mean_tau.is_finite() {
                        let var = taus.iter().map(|t| (t - mean_tau).powi(2)).sum::<f32>()
                            / taus.len().max(1) as f32;
                        let cv = if var.is_finite() {
                            var.sqrt() / mean_tau
                        } else {
                            0.0
                        };
                        // CV for default taus [0.01,0.1,1.0,10.0] ~ 1.7
                        // Map to 0-15% boost via sigmoid
                        (SCALE_BOOST_SIGMOID_AMPLITUDE
                            * (1.0 / (1.0 + (-SCALE_BOOST_SIGMOID_EXPONENT * (cv - 1.0)).exp())))
                            as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                (
                    sp.micro_phi * (1.0 + scale_boost * MICRO_PHI_SCALE_BOOST_FACTOR),
                    sp.meso_phi * (1.0 + scale_boost),
                    sp.macro_phi * (1.0 + scale_boost),
                    sp.bottleneck_score,
                    sp.emergence_ratio * (1.0 + scale_boost),
                    sp.num_clusters,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0)
            };
        // Structural Phi → behavioral coupling:
        // High bottleneck = poor integration → slow LR (protect against fragmentation).
        // High emergence = synergistic self-organization → boost confidence.
        // Science: Oizumi et al. (2014) — bottleneck constrains information flow;
        // Mediano et al. (2022) — high emergence = synergistic information.
        if struct_bn > STRUCTURAL_BOTTLENECK_THRESHOLD as f64 && self.stats.total_cycles > 15 {
            self.scale_lr("structural_bottleneck", STRUCTURAL_BOTTLENECK_LR_SCALE);
        }
        if struct_er > STRUCTURAL_EMERGENCE_CONFIDENCE_THRESHOLD as f64
            && self.stats.total_cycles > 15
        {
            self.adjust_confidence(
                "structural_emergence",
                STRUCTURAL_EMERGENCE_CONFIDENCE_BOOST,
            );
        }

        // ── Affective consciousness → behavior coupling ──────────────────
        // Valence and arousal from the consciousness equation's affective bridge
        // should drive learning and exploration, not just sit in telemetry.
        // Science: Barrett (2017) — affect is the primary driver of cognition;
        // Damasio (1999) — somatic markers guide all decisions.
        {
            use crate::cognitive_loop::thresholds::{
                AFFECT_AROUSAL_HIGH_LR_SCALE, AFFECT_AROUSAL_HIGH_THRESHOLD,
                AFFECT_AROUSAL_LOW_EXPLORE_DAMPEN, AFFECT_AROUSAL_LOW_THRESHOLD,
                AFFECT_VALENCE_NEGATIVE_EXPLORE_BOOST, AFFECT_VALENCE_NEGATIVE_THRESHOLD,
                AFFECT_VALENCE_POSITIVE_CONFIDENCE_BOOST, AFFECT_VALENCE_POSITIVE_THRESHOLD,
            };
            if affect_cons_arousal > AFFECT_AROUSAL_HIGH_THRESHOLD && self.stats.total_cycles > 10 {
                self.scale_lr("affect_high_arousal", AFFECT_AROUSAL_HIGH_LR_SCALE);
            } else if affect_cons_arousal < AFFECT_AROUSAL_LOW_THRESHOLD
                && affect_cons_arousal > 0.0
                && self.stats.total_cycles > 10
            {
                self.scale_exploration("affect_low_arousal", AFFECT_AROUSAL_LOW_EXPLORE_DAMPEN);
            }
            if affect_cons_valence < AFFECT_VALENCE_NEGATIVE_THRESHOLD
                && self.stats.total_cycles > 10
            {
                self.adjust_exploration(
                    "affect_negative_valence",
                    AFFECT_VALENCE_NEGATIVE_EXPLORE_BOOST,
                );
            } else if affect_cons_valence > AFFECT_VALENCE_POSITIVE_THRESHOLD
                && self.stats.total_cycles > 10
            {
                self.adjust_confidence(
                    "affect_positive_valence",
                    AFFECT_VALENCE_POSITIVE_CONFIDENCE_BOOST,
                );
            }
        }

        // ── Narrative self-phi → confidence coupling ─────────────────────
        // Self-coherence (narrative identity integration) grounds confidence
        // in decisions; fragmented identity → explore to find coherence.
        // Science: Gallagher (2000) — narrative self underpins identity;
        // Conway & Pleydell-Pearce (2000) — self-coherence → decision confidence.
        {
            use crate::cognitive_loop::thresholds::{
                NARRATIVE_SELF_PHI_CONFIDENCE_SCALE, NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD,
                NARRATIVE_SELF_PHI_LOW_EXPLORE_BOOST, NARRATIVE_SELF_PHI_LOW_THRESHOLD,
            };
            if narrative_gwt_self_psi > NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD
                && self.stats.total_cycles > 15
            {
                let boost = ((narrative_gwt_self_psi - NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD)
                    * NARRATIVE_SELF_PHI_CONFIDENCE_SCALE as f64)
                    as f32;
                self.adjust_confidence("narrative_self_coherent", boost);
            } else if narrative_gwt_self_psi > 0.0
                && narrative_gwt_self_psi < NARRATIVE_SELF_PHI_LOW_THRESHOLD
                && self.stats.total_cycles > 15
            {
                self.adjust_exploration(
                    "narrative_self_fragmented",
                    NARRATIVE_SELF_PHI_LOW_EXPLORE_BOOST,
                );
            }
        }

        let consciousness_weights = consciousness_output.current_weights;
        let consciousness_weight_variance = consciousness_output.weight_variance;
        let convergence_state = consciousness_output.convergence_state.as_str().to_string();

        // Phi-Harmony coupling: during Sacred Stillness, weight Phi by integration
        // quality rather than raw intensity. Rest-state Phi should reflect how
        // well the system maintains coherent integration while reducing activity.
        // Science: Tononi (2004) — Phi measures integrated information regardless
        // of activity level; rest-state Phi is not diminished but differently structured.
        if self.stats.in_active_rest {
            // During active rest, boost the coherence component of Phi
            // and dampen the binding intensity component.
            // Net effect: rest-state consciousness emphasizes integration quality
            // over raw binding intensity (~4% net boost).
            let coherence_weight: f32 = REST_COHERENCE_WEIGHT; // 20% more weight on coherence
            let binding_dampen: f32 = REST_BINDING_DAMPEN; // 20% less weight on binding intensity
            self.stats.phi_rest_quality_factor = coherence_weight;
            self.stats.phi_rest_binding_factor = binding_dampen;
            // Weighted combination: coherence contributes 60%, binding 40%
            // (coherence matters more during rest than binding)
            let rest_modulation = coherence_weight * REST_MODULATION_COHERENCE_FRAC
                + binding_dampen * REST_MODULATION_BINDING_FRAC;
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
            use crate::cognitive_loop::thresholds::{
                EQ_V2_INTEGRATION_CONFIDENCE_BOOST, EQ_V2_KNOWLEDGE_EXPLORATION_BOOST,
                EQ_V2_RECURSION_LR_SCALE, EQ_V2_WORKSPACE_CONFIDENCE_BOOST,
            };
            use crate::consciousness::consciousness_equation_v2::CoreComponent;
            match component {
                CoreComponent::Workspace => {
                    // Low workspace → boost attention budget and GWT broadcast
                    self.adjust_confidence("eq_v2_workspace", EQ_V2_WORKSPACE_CONFIDENCE_BOOST);
                }
                CoreComponent::Recursion => {
                    // Low recursion (HOT depth) → boost meta-cognitive sensitivity
                    self.scale_lr("eq_v2_recursion", EQ_V2_RECURSION_LR_SCALE);
                }
                CoreComponent::Integration => {
                    // Low integration → increase coherence sensitivity
                    self.adjust_confidence("eq_v2_integration", EQ_V2_INTEGRATION_CONFIDENCE_BOOST);
                }
                CoreComponent::Knowledge => {
                    // Low epistemic quality → boost exploration for information gain
                    self.adjust_exploration("eq_v2_knowledge", EQ_V2_KNOWLEDGE_EXPLORATION_BOOST);
                }
                // Attention, Binding, Efficacy already handled by Phase 19 gradient path
                _ => {}
            }
        }

        // ── Reasoning engine reliability → confidence + LR feedback ───────
        // When the reasoning engine produces high-reliability assessments,
        // nudge prediction confidence upward. If the gate was passed AND
        // the outcome was good, boost learning rate to reinforce the pathway.
        // Science: Stanovich & West (2000) — dual-process theory: System 2
        // reliability calibrates metacognitive confidence.
        #[cfg(feature = "reasoning_engine")]
        {
            // Extract data from reasoning engine (immutable borrow) before mutable calls
            let reasoning_feedback = self.reasoning_engine.as_ref().and_then(|re| {
                re.last_event().map(|evt| {
                    let gate_success = evt
                        .posthoc_outcome
                        .as_ref()
                        .is_some_and(|p| p.gate_passed && p.outcome_good);
                    (evt.reliability, gate_success)
                })
            });
            if let Some((r, gate_success)) = reasoning_feedback {
                use crate::cognitive_loop::thresholds::{
                    REASONING_RELIABILITY_CONFIDENCE_SCALE, REASONING_RELIABILITY_THRESHOLD,
                };
                // High reliability → nudge confidence up
                if r.is_finite() && r > REASONING_RELIABILITY_THRESHOLD {
                    let boost = ((r - 0.5) * REASONING_RELIABILITY_CONFIDENCE_SCALE) as f32;
                    self.adjust_confidence("reasoning_reliability", boost);
                }
                // Gate passed + good outcome → boost LR to reinforce
                if gate_success {
                    self.scale_lr("reasoning_gate_success", REASONING_GATE_SUCCESS_LR_SCALE);
                }
            }
        }

        // Soul experience integration
        let _t = Instant::now();
        {
            let moral_score = self
                .last_moral_judgment()
                .map(|j| j.moral_score)
                .unwrap_or(0.0);
            let valence = self.behavior.emotion_contagion.valence;
            let cycles = self.stats.total_cycles as u64;
            if let Some(ref mut soul) = self.ethics_values.soul {
                let experience = crate::soul::Experience {
                    embedding: encoding_hdv.clone(),
                    value_alignment: moral_score,
                    emotional_valence: valence,
                    lessons: Vec::new(),
                    timestamp: cycles,
                };
                soul.integrate_experience(experience);
            }
        }
        module_timings.soul_experience = _t.elapsed().as_micros() as u64;

        // ── Phi-Dyad: Relational Consciousness ─────────────────────────────
        // Compute Φ_dyad from recent AI + input HVs (Phase 6 wiring).
        if self.behavior.social_mgr.recent_ai_hvs.len() >= 2 {
            if let (Some(dyad), Some(model)) = (
                &self.behavior.social_mgr.phi_dyad,
                &self.behavior.social_mgr.partner_model,
            ) {
                use symthaea_core::hdc::relational_consciousness::{
                    RelationMode, RelationalAssessment,
                };
                let relational = RelationalAssessment {
                    agent_a: "symthaea".to_string(),
                    agent_b: model.partner_id.clone(),
                    phi_relation: model.phi_relational,
                    stage: model.stage,
                    synchrony: model.trust as f64,
                    turn_taking_quality: SOCIAL_MODEL_TURN_TAKING_DEFAULT as f64,
                    mutual_information: model.reciprocity as f64,
                    mode: if model.trust > SOCIAL_MODEL_TRUST_MODE_THRESHOLD {
                        RelationMode::IThou
                    } else {
                        RelationMode::IIt
                    },
                    num_interactions: model.interactions_count as usize,
                    relationship_age: 0.0,
                    explanation: String::new(),
                };
                let input = crate::partnership::DyadInput {
                    ai_states: &self.behavior.social_mgr.recent_ai_hvs,
                    human_states: &self.behavior.social_mgr.recent_input_hvs,
                    relational: &relational,
                    human_model: model,
                    weights: crate::partnership::DyadWeights::default(),
                };
                let result = dyad.compute(&input);
                self.behavior.social_mgr.social.relational_psi = result.phi_dyad;

                // Phi divergence → exploration (novel relational territory)
                // Science: Friston (2010) — high divergence = high epistemic value
                let phi_divergence = (result.phi_ai - result.phi_human).abs();
                if phi_divergence > PHI_DIVERGENCE_THRESHOLD {
                    let boost = (phi_divergence - PHI_DIVERGENCE_THRESHOLD).min(PHI_DIVERGENCE_MAX)
                        * PHI_DIVERGENCE_SCALE;
                    self.adjust_exploration_pri(
                        "phi_divergence",
                        boost as f32,
                        Priority::Aesthetic,
                    );
                }

                // Phi relational → oxytocin (prosocial bonding)
                // Science: Feldman (2012) — relational coherence drives oxytocin release
                if result.phi_relational > PHI_RELATIONAL_OXY_THRESHOLD {
                    let oxy = (result.phi_relational - PHI_RELATIONAL_OXY_THRESHOLD)
                        * PHI_RELATIONAL_OXY_SCALE;
                    self.neuromod.bath.oxytocin.produce(oxy as f32);
                }
            }
        }

        // ── Consciousness → Neuromod reverse coupling ────────────────────
        // Science: Dehaene & Changeux (2011) — conscious access modulates
        // catecholamine release.
        {
            let psi = self.stats.unified_psi;
            if psi > PSI_DA_THRESHOLD as f32 {
                let da_signal =
                    ((psi as f64 - PSI_DA_THRESHOLD) * PSI_DA_SCALE).min(PSI_DA_CAP as f64) as f32;
                self.neuromod.bath.dopamine.produce(da_signal);
            }
            if psi > PSI_5HT_THRESHOLD as f32 {
                let sht_signal = ((psi as f64 - PSI_5HT_THRESHOLD) * PSI_5HT_SCALE)
                    .min(PSI_5HT_CAP as f64) as f32;
                self.neuromod.bath.serotonin.produce(sht_signal);
            }
            if psi < PSI_NE_THRESHOLD as f32 && psi > 0.0 {
                let ne_signal =
                    ((PSI_NE_THRESHOLD - psi as f64) * PSI_NE_SCALE).min(PSI_NE_CAP as f64) as f32;
                self.neuromod.bath.noradrenaline.produce(ne_signal);
            }
        }

        // Trust evolution from cycle coherence (Bowlby 1969)
        // Coherence > 0.5 builds trust, < 0.5 erodes it; slow decay prevents runaway
        if let Some(ref mut model) = self.behavior.social_mgr.partner_model {
            let signal =
                (dynamics.core.coherence as f64 - TRUST_SIGNAL_MIDPOINT) * TRUST_SIGNAL_RATE;
            model.trust =
                ((model.trust as f64 + signal).clamp(0.0, 1.0) * TRUST_DECAY_FACTOR) as f32;
        }

        // ── Track 4b: Cross-module agreement metric ─────────────────────────
        let fep_confidence = (1.0 - dynamics.fep.fep_surprise.min(1.0)).max(0.0) as f32;
        let resonator_confidence = dynamics.resonator.resonator_best_sim;
        let moral_confidence = self
            .last_moral_judgment()
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
        // Guard: NaN in any signal would propagate through mean/variance/sqrt
        let all_finite = signals.iter().all(|s| s.is_finite());
        let mean_signal: f32 = signals.iter().sum::<f32>() / signals.len().max(1) as f32;
        let variance: f32 = signals
            .iter()
            .map(|s| (s - mean_signal).powi(2))
            .sum::<f32>()
            / signals.len().max(1) as f32;
        let cross_module_agreement: f32 = if all_finite {
            (1.0_f32 - (variance * CROSS_MODULE_VARIANCE_AMPLIFICATION as f32).sqrt())
                .clamp(0.0, 1.0)
        } else {
            0.5 // neutral agreement on non-finite input
        };
        if cross_module_agreement > CROSS_MODULE_AGREEMENT_HIGH {
            self.adjust_confidence(
                "cross_mod_agree",
                (cross_module_agreement - CROSS_MODULE_AGREEMENT_HIGH)
                    * AGREEMENT_HIGH_CONFIDENCE_SCALE,
            );
        } else if cross_module_agreement < CROSS_MODULE_AGREEMENT_LOW {
            self.scale_confidence(
                "cross_mod_disagree",
                1.0 - (CROSS_MODULE_AGREEMENT_LOW - cross_module_agreement)
                    * AGREEMENT_LOW_CONFIDENCE_SCALE,
            );
            self.adjust_exploration(
                "cross_module_disagree",
                (CROSS_MODULE_AGREEMENT_LOW - cross_module_agreement)
                    * AGREEMENT_LOW_EXPLORATION_SCALE,
            );
            // Session 11 Item 7: Very low agreement → raise threshold for urgency escalation.
            // Subsystems disagree about error magnitude → require stronger signal before reacting.
            // Science: Tononi (2004) — incoherent integration requires cautious interpretation.
            if cross_module_agreement < AGREEMENT_CRITICAL_THRESHOLD && self.stats.total_cycles > 20
            {
                self.scale_threshold("low_agreement_caution", AGREEMENT_CRITICAL_CAUTION_SCALE);
            }
        }
        self.stats.avg_cross_module_agreement = self.stats.avg_cross_module_agreement
            * AGREEMENT_EMA_DECAY
            + cross_module_agreement * (1.0 - AGREEMENT_EMA_DECAY);

        // Cross-module agreement velocity: rapid drops signal subsystem desynchronization.
        // Analogous to coherence_velocity but for inter-module rather than intra-module signals.
        let agreement_velocity =
            cross_module_agreement - self.carryover.quality.prev_cross_module_agreement;
        self.carryover.quality.prev_cross_module_agreement = cross_module_agreement;
        // Session 9 Item 4: Compound instability detector.
        // When agreement drops AND errors are rising simultaneously → cascading failure.
        // Friston (2010): cascading precision failures require active recovery.
        let error_slope = perception.urgency.error_slope;
        let compound_instability = agreement_velocity < COMPOUND_INSTABILITY_VELOCITY
            && error_slope > COMPOUND_INSTABILITY_ERROR_SLOPE
            && self.stats.total_cycles > 30;
        if compound_instability {
            // Stronger protective response than either alone
            self.scale_lr("compound_instability", COMPOUND_INSTABILITY_LR_SCALE);
            self.adjust_exploration("compound_instability", COMPOUND_INSTABILITY_EXPLORATION);
        } else if agreement_velocity < AGREEMENT_VELOCITY_DROP_THRESHOLD
            && self.stats.total_cycles > 30
        {
            // Rapid agreement drop alone → dampen LR, boost exploration preemptively.
            // Science: desynchronization across subsystems means conflicting learning signals.
            self.scale_lr("agreement_vel_drop", AGREEMENT_VELOCITY_DROP_LR);
            self.adjust_exploration("agreement_vel_drop", AGREEMENT_VELOCITY_DROP_EXPLORATION);
        }

        // Session 10 Item 8: Agreement rising + confidence falling → gentle correction.
        // When subsystems converge but output confidence drops, the issue is in the
        // final integration, not the subsystems. Apply gentle confidence scale to re-align.
        // Science: Tononi (2004) — agreement rise with confidence fall = integration bottleneck.
        let agreement_confidence_coupling = agreement_velocity
            > AGREEMENT_CONFIDENCE_COUPLING_THRESHOLD
            && self.carryover.quality.coherence_velocity < AGREEMENT_COHERENCE_VELOCITY_THRESHOLD
            && self.stats.total_cycles > 20;
        if agreement_confidence_coupling {
            self.scale_confidence("agree_conf_coupling", AGREEMENT_CONFIDENCE_COUPLING_SCALE);
        }

        // ── Unified quality signal fusion ───────────────────────────
        let unified_quality_score;
        {
            let anomaly_factor = if dynamics.reasoning.metacognitive_anomaly {
                0.0
            } else {
                1.0
            };
            unified_quality_score = UNIFIED_QUALITY_PREDICTION_WEIGHT
                * dynamics.core.prediction_coherence
                + UNIFIED_QUALITY_AGREEMENT_WEIGHT * cross_module_agreement
                + UNIFIED_QUALITY_ANOMALY_WEIGHT * anomaly_factor;
            self.stats.avg_unified_quality = self.stats.avg_unified_quality * QUALITY_EMA_DECAY
                + unified_quality_score * (1.0 - QUALITY_EMA_DECAY);

            if unified_quality_score > CROSS_MODULE_AGREEMENT_HIGH {
                let quality_boost =
                    (unified_quality_score - CROSS_MODULE_AGREEMENT_HIGH) * QUALITY_HIGH_LR_SCALE;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + quality_boost;
                self.carryover.learning.subsystem_lr_factor = self
                    .carryover
                    .learning
                    .subsystem_lr_factor
                    .clamp(QUALITY_LR_CLAMP_MIN, QUALITY_LR_CLAMP_MAX);
            }
            if unified_quality_score < CROSS_MODULE_AGREEMENT_LOW && self.stats.total_cycles > 30 {
                self.scale_exploration("low_quality_dampen", LOW_QUALITY_EXPLORATION_DAMPEN);
            }
        }

        // Harmony entropy → learning rate modulation: broad moral engagement
        // (high entropy) slightly boosts learning rate. Specialization (low entropy)
        // slightly dampens. Range: entropy ∈ [0, ln(8)≈2.08], mapped to [0.95, 1.05].
        // Science: broader exploration of value space → richer training signal.
        {
            let entropy = self
                .ethics_engine
                .moral_topology()
                .last_summary()
                .harmony_entropy;
            let max_entropy = (symthaea_types::N_HARMONIES as f64).ln();
            if max_entropy > 0.0 && entropy.is_finite() {
                let normalized = (entropy / max_entropy).clamp(0.0, 1.0); // 0..1
                let lr_mod = ENTROPY_LR_MIN + normalized * ENTROPY_LR_RANGE; // 0.95..1.05
                self.carryover.learning.subsystem_lr_factor *= lr_mod as f32;
                self.carryover.learning.subsystem_lr_factor = self
                    .carryover
                    .learning
                    .subsystem_lr_factor
                    .clamp(QUALITY_LR_CLAMP_MIN, QUALITY_LR_CLAMP_MAX);
            }
        }

        // Session 13 Item 5: Flow state → subsystem LR modulation.
        // Flow = optimal learning zone → gently boost subsystem learning.
        // Science: Csikszentmihalyi (1990) — flow maximizes skill acquisition.
        if self.behavior.flow_state.in_flow && self.behavior.flow_state.intensity > 0.5 {
            self.carryover.learning.subsystem_lr_factor *= 1.05;
            self.carryover.learning.subsystem_lr_factor = self
                .carryover
                .learning
                .subsystem_lr_factor
                .clamp(QUALITY_LR_CLAMP_MIN, QUALITY_LR_CLAMP_MAX);
        }

        // Session 13 Item 8: Sustained high quality → exploration floor.
        // Prevent total convergence when system is performing well.
        // Science: Dayan & Sejnowski (1996) — minimum exploration prevents local optima.
        if unified_quality_score > 0.7 {
            self.carryover.quality.consecutive_high_quality = self
                .carryover
                .quality
                .consecutive_high_quality
                .saturating_add(1);
        } else {
            self.carryover.quality.consecutive_high_quality = 0;
        }
        if self.carryover.quality.consecutive_high_quality > 10 && self.stats.total_cycles > 30 {
            self.adjust_exploration("quality_floor", 0.01);
        }

        // Session 12 Item 4: Epistemic conflict → exploration boost.
        // Multiple conflicting epistemic signals indicate unresolved uncertainty.
        // Science: Berlyne (1960) — epistemic curiosity arises from conflicting beliefs.
        if epistemic_conflict_count > 2 && self.stats.total_cycles > 20 {
            self.adjust_exploration(
                "epistemic_conflict",
                epistemic_conflict_count as f32 * EPISTEMIC_CONFLICT_EXPLORATION_SCALE,
            );
        }

        // Session 12 Item 5: Phenomenal fragmentation → binding recovery.
        // Fragmented binding = unreliable integration → dampen confidence, boost exploration.
        // Science: Tononi (2004) — low integration → low consciousness quality.
        if phenomenal_fragmented && self.stats.total_cycles > 15 {
            self.scale_confidence(
                "phenomenal_fragmented",
                PHENOMENAL_FRAGMENTED_CONFIDENCE_DAMPEN,
            );
            self.adjust_exploration(
                "phenomenal_fragmented",
                PHENOMENAL_FRAGMENTED_EXPLORATION_BOOST,
            );
        }

        // Session 12 Item 6: Temporal discontinuity → LR dampen + exploration.
        // Temporal gaps make learning unreliable (missing causal chain).
        // Science: Howard & Kahana (2002) — temporal context model: gaps disrupt encoding.
        if temporal_discontinuity && self.stats.total_cycles > 15 {
            self.scale_lr("temporal_discontinuity", TEMPORAL_DISCONTINUITY_LR_DAMPEN);
            self.adjust_exploration(
                "temporal_discontinuity",
                TEMPORAL_DISCONTINUITY_EXPLORATION_BOOST,
            );
        }

        // Session 16 Item 1: Temporal binding strength → exploration/LR feedback.
        // Strong theta-gated binding → stable temporal model → exploit (dampen exploration).
        // Weak binding → poor temporal integration → explore + dampen LR.
        // Science: Buzsáki (2002) — theta oscillation gates temporal context binding.
        {
            use super::thresholds::{
                TEMPORAL_BINDING_DAMPEN_THRESHOLD, TEMPORAL_BINDING_EXPLORE_SCALE,
                TEMPORAL_BINDING_EXPLORE_THRESHOLD, TEMPORAL_BINDING_LOW_LR_SCALE,
            };
            let tb = perception.encoding.temporal_binding_strength;
            if tb < TEMPORAL_BINDING_EXPLORE_THRESHOLD && self.stats.total_cycles > 15 {
                let boost =
                    (TEMPORAL_BINDING_EXPLORE_THRESHOLD - tb) * TEMPORAL_BINDING_EXPLORE_SCALE;
                self.adjust_exploration("temporal_binding_low", boost);
                self.scale_lr("temporal_binding_low", TEMPORAL_BINDING_LOW_LR_SCALE);
            } else if tb > TEMPORAL_BINDING_DAMPEN_THRESHOLD && self.stats.total_cycles > 15 {
                self.scale_exploration(
                    "temporal_binding_high",
                    TEMPORAL_BINDING_HIGH_EXPLORATION_SCALE,
                );
            }
        }

        // Session 12 Item 7: Cross-modal binding → attention sensitivity.
        // High binding (>0.7) → more modalities integrated → boost attention sensitivity.
        // Low binding (<0.3) → weak integration → dampen (trust only primary modality).
        // Science: Engel et al. (2001) — synchrony-based binding gates cross-modal attention.
        if cross_modal_binding_strength > CROSS_MODAL_BINDING_HIGH_THRESHOLD
            && self.stats.total_cycles > 10
        {
            let binding_boost = (cross_modal_binding_strength - CROSS_MODAL_BINDING_HIGH_THRESHOLD)
                * CROSS_MODAL_BINDING_HIGH_SCALE;
            self.adjust_confidence("binding_attention_hi", binding_boost);
        } else if cross_modal_binding_strength < CROSS_MODAL_BINDING_LOW_THRESHOLD
            && self.stats.total_cycles > 10
        {
            let binding_dampen = 1.0
                - (CROSS_MODAL_BINDING_LOW_THRESHOLD - cross_modal_binding_strength)
                    * CROSS_MODAL_BINDING_LOW_SCALE;
            self.scale_confidence(
                "binding_attention_lo",
                binding_dampen.max(CROSS_MODAL_BINDING_LOW_FLOOR),
            );
        }

        // Hierarchical bundling: accumulate current cycle's BinaryHV per region
        // and compute cross-region binding strength from structured aggregates.
        if let Some(ref mut bundler) = self.hierarchical_bundler {
            use symthaea_core::hdc::substrate_independence::CorticalRegion;

            // Feed perception HV into Sensory region (not just Integration)
            bundler.add(CorticalRegion::Sensory, perception.encoding.hv16_cached);

            // CfC output → Integration region (threshold continuous → binary).
            // CfC hidden state is typically 128-256 floats. We tile it across the
            // full 16384-bit BinaryHV (2048 bytes) so all bits carry signal, rather
            // than leaving most of the HV zeroed.
            {
                use symthaea_core::hdc::BinaryHV;
                let cfc_output = &dynamics.core.output;
                if !cfc_output.is_empty() {
                    let mut binary_bytes = [0u8; 2048];
                    for (byte_idx, byte) in binary_bytes.iter_mut().enumerate() {
                        let mut val = 0u8;
                        for bit in 0..8 {
                            let flat_idx = byte_idx * 8 + bit;
                            // Tile: wrap around CfC output via modulo
                            let src_idx = flat_idx % cfc_output.len();
                            if cfc_output[src_idx] > 0.0 {
                                val |= 1 << bit;
                            }
                        }
                        *byte = val;
                    }
                    let cfc_hv = BinaryHV(binary_bytes);
                    bundler.add(CorticalRegion::Integration, cfc_hv);
                }
            }

            // Once we have enough accumulated vectors, compute aggregate metrics
            if bundler.total_vectors() >= 12 {
                let bundles = bundler.all_bundles();

                // Compute cross-region binding strength from bundle similarities
                if bundles.len() >= 2 {
                    if let Some(aggregate) = bundler.aggregate() {
                        let mut binding_sum = 0.0f32;
                        let mut count = 0usize;
                        for bundle in &bundles {
                            if let Some(recovered) =
                                bundler.unbind_region(&aggregate, &bundle.region)
                            {
                                binding_sum += recovered.similarity(&bundle.local_bundle);
                                count += 1;
                            }
                        }
                        if count > 0 {
                            let avg_binding = binding_sum / count as f32;
                            // EMA-blend into cross-modal binding for downstream consumers
                            cross_modal_binding_strength = cross_modal_binding_strength
                                * CROSS_MODAL_BINDING_MOMENTUM
                                + avg_binding * CROSS_MODAL_BINDING_ALPHA;
                            tracing::debug!(
                                avg_binding,
                                active_regions = bundles.len(),
                                total_vectors = bundler.total_vectors(),
                                "Hierarchical bundling: cross-region binding computed"
                            );
                        }
                    }
                }

                // Reset bundler periodically to keep accumulation fresh
                if bundler.total_vectors() > 100 {
                    bundler.clear();
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // GEO-SYNTH: Decode geodesic path into mental movie (if requested)
        // ═══════════════════════════════════════════════════════════════════════
        #[cfg(feature = "vision-manifold")]
        let mental_movie = if self.carryover.quality.last_request_geodesic {
            if let Some(ref bridge) = self.sensorimotor.vision_sensory.vision_bridge {
                let manifold = bridge.manifold();
                let path = manifold.last_geodesic();

                if !path.is_empty() {
                    let frames = manifold.decode_geodesic_to_frames_improved(path);
                    if !frames.is_empty() {
                        Some(crate::cognitive_loop::types::MentalMovie {
                            frames,
                            width: self.config.vision_frame_width,
                            height: self.config.vision_frame_height,
                            channels: bridge.manifold().last_frame_channels(),
                            path_length: path.len(),
                            semantic_coherence: 0.0, // can be enhanced later
                            trajectory: path.to_vec(),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // ── Social Coherence: Sync metrics from ToM engine ───────────
        self.behavior.social_mgr.sync_coherence_metrics();

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
                agreement_confidence_coupling,
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
                predictive_behavioral_error,
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
                hierarchical_free_energy_lr_boost,
                predictive_phi_lr_delta,
                body_valence_confidence_delta,
                narrative_self_confidence_factor,
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
                kosmic_coherence: self.carryover.quality.last_kosmic_coherence,
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
                memory_db_flushed,
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
            #[cfg(feature = "vision-manifold")]
            mental_movie,
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
            result.metadata.consciousness.consciousness_level >= 0.0
                && result.metadata.consciousness.consciousness_level <= 1.0
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
        if let Some(sigma) = result.metadata.structural.sigma {
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

    // ── Modulation pathway tests ──────────────────────────────────────

    /// Social trust → learning rate modulation (Decety & Chaminade 2003).
    /// High trust should produce higher social LR factor than low trust.
    #[test]
    fn feedback_social_trust_modulates_lr() {
        let mut svc_hi = make_service();
        svc_hi.behavior.social_mgr.social.social_trust = 0.9;
        let hi = svc_hi.cycle("trust high");

        let mut svc_lo = make_service();
        svc_lo.behavior.social_mgr.social.social_trust = 0.1;
        let lo = svc_lo.cycle("trust low");

        // Social LR factor = 0.8 + 0.4 * trust → [0.84, 1.16]
        assert!(
            hi.metadata.social_learning_rate_factor > lo.metadata.social_learning_rate_factor,
            "High trust ({}) should yield higher social LR factor than low trust ({})",
            hi.metadata.social_learning_rate_factor,
            lo.metadata.social_learning_rate_factor,
        );
    }

    /// ToM accuracy gating: only active when social models exist.
    #[test]
    fn feedback_tom_accuracy_requires_social_models() {
        let mut svc = make_service();
        svc.behavior.social_mgr.social.social_prediction_accuracy = 0.0;
        svc.behavior.social_mgr.social.social_models_count = 0;
        for _ in 0..15 {
            svc.cycle("warmup");
        }
        let result = svc.cycle("test tom guard");
        assert!(
            result.metadata.prediction_coherence.is_finite(),
            "ToM guard: prediction coherence should be finite without social models"
        );
    }

    /// Cross-module agreement is bounded [0, 1] over multiple cycles.
    #[test]
    fn feedback_cross_module_agreement_bounded() {
        let mut svc = make_service();
        for i in 0..20 {
            let r = svc.cycle(&format!("cycle {}", i));
            let a = r.metadata.cross_module_agreement;
            assert!(
                a >= 0.0 && a <= 1.0,
                "Agreement {} out of [0,1] at cycle {}",
                a,
                i
            );
        }
    }

    /// Unified quality score composition: weighted blend remains finite and bounded.
    #[test]
    fn feedback_unified_quality_score_bounded() {
        let mut svc = make_service();
        for _ in 0..15 {
            svc.cycle("warmup");
        }
        let r = svc.cycle("quality composition");
        let q = r.metadata.quality.unified_quality_score;
        assert!(q.is_finite(), "Quality score must be finite");
        assert!(
            q >= 0.0 && q <= 1.5,
            "Quality score {} out of expected range",
            q
        );
    }

    /// Temporal continuity fields remain finite over many cycles.
    #[test]
    fn feedback_temporal_continuity_fields_finite() {
        let mut svc = make_service();
        for _ in 0..20 {
            let r = svc.cycle("temporal test");
            assert!(r.metadata.temporal.temporal_continuity.is_finite());
            assert!(r.metadata.temporal.temporal_max_chain_length < 10000);
        }
    }

    /// Epistemic gate confidence stays bounded [0, 1] and finite.
    #[test]
    fn feedback_epistemic_gate_stable_over_cycles() {
        let mut svc = make_service();
        for _ in 0..25 {
            let r = svc.cycle("epistemic stability");
            let c = r.metadata.epistemic_gate_confidence;
            assert!(
                c.is_finite() && c >= 0.0 && c <= 1.0,
                "Epistemic gate confidence {} out of bounds",
                c
            );
        }
    }

    /// Consciousness gradient recovery doesn't produce NaN.
    #[test]
    fn feedback_consciousness_gradient_no_nan() {
        let mut svc = make_service();
        for _ in 0..30 {
            svc.cycle("gradient warmup");
        }
        let r = svc.cycle("gradient recovery");
        assert!(
            r.metadata
                .consciousness
                .consciousness_gradient_magnitude
                .is_finite()
        );
        assert!(r.metadata.prediction_coherence.is_finite());
    }

    /// Agreement velocity and coherence velocity remain finite over cycles.
    #[test]
    fn feedback_velocity_fields_no_nan() {
        let mut svc = make_service();
        for i in 0..30 {
            let r = svc.cycle(&format!("vel_{}", i));
            assert!(
                r.metadata.cross_module_agreement.is_finite(),
                "Agreement NaN at {}",
                i
            );
            assert!(
                r.metadata.quality.coherence_velocity.is_finite(),
                "Coherence vel NaN at {}",
                i
            );
        }
    }
}
