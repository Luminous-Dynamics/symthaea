// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::prelude::*;

impl CognitiveLoopService {
    pub(in crate::cognitive_loop::cycle_phase_output) fn populate_modulation_telemetry(
        &mut self,
        metadata: &mut CycleMetadata,
        perception: &PerceptionPhaseResult,
        dynamics: &DynamicsPhaseResult,
        feedback: &FeedbackPhaseResult,
    ) {
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

        metadata.cantor = super::super::super::CantorTelemetry {
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

        metadata.mce_bottleneck = self.carryover.consciousness.mce_bottleneck_name.clone();
        metadata.mce_softmin = self.carryover.consciousness.mce_softmin;
        metadata.mce_weighted_sum = self.carryover.consciousness.mce_weighted_sum;
        metadata.mce_narrative = self.carryover.consciousness.mce_narrative;
        metadata.mce_social = self.carryover.consciousness.mce_social;

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
        metadata.modulation.confidence_crash_detected = dynamics.confidence_crash_detected;
        metadata.crash_freeze_remaining = self.carryover.quality.crash_freeze_remaining;
        metadata.modulation.lr_frozen = dynamics.lr_frozen;
        metadata.hysteresis_factor = self.carryover.quality.hysteresis_factor;
        metadata.modulation.agreement_confidence_coupling =
            feedback.quality.agreement_confidence_coupling;

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

        {
            use super::super::super::thresholds::{
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
                self.stats.total_cycles <= super::super::super::thresholds::STARTUP_WARMUP_CYCLES;
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
                > super::super::super::thresholds::ERROR_OSCILLATION_BIFURCATION;
        }

        {
            use super::super::super::thresholds::{
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
                >= super::super::super::thresholds::TEMPORAL_CHAIN_DEEP_THRESHOLD
                || feedback.consciousness.temporal_max_chain_length
                    <= super::super::super::thresholds::TEMPORAL_CHAIN_SHALLOW_THRESHOLD);
        metadata.modulation.eq_v2_bottleneck_response =
            !feedback.consciousness.eq_v2_limiting_component.is_empty();

        {
            use super::super::super::thresholds::{
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
    }
}
