// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::time::Instant;

use super::super::CognitiveLoopService;
use super::super::feedback_state::Priority;
use super::super::helpers;
use super::super::phase_results::PerceptionPhaseResult;
use super::super::thresholds::*;

impl CognitiveLoopService {
    /// Memory recall, resonator matching, phenomenal binding, and goal attention.
    ///
    /// Performs episodic recall, resonator-enhanced factorization, binding→threshold/confidence
    /// gating, resonator similarity→consolidation, and goal system attention bias.
    pub(super) fn phase_dynamics_memory_binding(
        &mut self,
        perception: &PerceptionPhaseResult,
        urgency: super::super::CycleUrgency,
        prediction_error: f32,
        module_timings: &mut super::super::ModuleTimings,
    ) -> super::MemoryBindingResult {
        // ═══════════════════════════════════════════════════════════════════════
        // 1a. Memory System Integration: Recall relevant episodic memories
        // ═══════════════════════════════════════════════════════════════════════
        let memory_context_boost =
            self.recall_episodic_context(&perception.encoding.compressed_state);

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.1 Resonator-enhanced recall: factorize bundled memories
        // ═══════════════════════════════════════════════════════════════════════
        let mut resonator_wm_primed = false;
        let mut resonator_reconsolidated: usize = 0;
        let mut resonator_best_sim: f32 = 0.0;

        let resonator_prediction_error: f32 =
            if let Some(ref prev_pred) = self.stats.last_resonator_prediction {
                let sim = helpers::cosine_f32(prev_pred, &perception.encoding.compressed_state);
                (1.0 - sim).clamp(0.0, 1.0)
            } else {
                0.0
            };

        // ── Phase 20: Resonator prediction error → exploration/confidence ────
        let resonator_error_exploration_mod = if resonator_prediction_error
            > RESONATOR_ERROR_EXPLORATION_THRESHOLD
            && self.stats.total_cycles > RESONATOR_STARTUP_CYCLES
        {
            let boost = (resonator_prediction_error - RESONATOR_ERROR_EXPLORATION_THRESHOLD)
                * RESONATOR_ERROR_EXPLORATION_SCALE;
            self.adjust_exploration("resonator_error_high", boost);
            self.adjust_confidence(
                "resonator_error_high",
                -boost * RESONATOR_ERROR_CONFIDENCE_DAMPEN,
            );
            self.stats.resonator_error_exploration_count += 1;
            boost
        } else if resonator_prediction_error < RESONATOR_LOW_ERROR_THRESHOLD
            && resonator_prediction_error > 0.0
        {
            let confidence_boost = (RESONATOR_LOW_ERROR_THRESHOLD - resonator_prediction_error)
                * RESONATOR_LOW_ERROR_CONFIDENCE_SCALE;
            self.adjust_confidence("resonator_error_low", confidence_boost);
            self.stats.resonator_error_exploration_count += 1;
            // Session 15 Item 7: Sustained low resonator error → confidence recovery.
            // If >80% of recent cycles had low error, give an additional confidence nudge.
            // Science: Bar (2009) — consistent prediction accuracy signals reliable model.
            if self.stats.total_cycles > DYNAMICS_POST_BOOT_CYCLES
                && self.stats.resonator_error_exploration_count
                    > (self.stats.total_cycles / 2) as u64
            {
                self.adjust_confidence(
                    "resonator_sustained_low",
                    super::super::thresholds::RESONATOR_SUSTAINED_LOW_CONFIDENCE,
                );
            }
            -confidence_boost
        } else {
            0.0
        };

        // ── Phase 17: Coherence memoization — cache pre-update value ─────
        let pre_update_coherence = self
            .language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence();

        // ── Phase 20: Phenomenal binding → threshold gating ──────────────────
        let cached_binding = self.carryover.quality.last_phenomenal_binding as f32;
        let binding_threshold_mod = if cached_binding > BINDING_CONFIDENCE_THRESHOLD {
            let relief =
                (cached_binding - BINDING_CONFIDENCE_THRESHOLD) * BINDING_STRONG_RELIEF_SCALE;
            self.scale_threshold("binding_strong_relief", 1.0 - relief);
            self.stats.binding_threshold_mod_count += 1;
            -relief
        } else if cached_binding < BINDING_LOW_THRESHOLD && cached_binding > 0.0 {
            let caution = (BINDING_LOW_THRESHOLD - cached_binding) * BINDING_WEAK_CAUTION_SCALE;
            self.scale_threshold("binding_weak_caution", 1.0 + caution);
            self.stats.binding_threshold_mod_count += 1;
            caution
        } else {
            0.0
        };

        // ── Phase 21: Phenomenal binding → prediction confidence ─────────
        // Confidence = cached_binding: strong binding carries full weight,
        // weak binding is discounted in the consensus.
        // Science: Treisman (1998) — binding confidence tracks integration strength.
        let binding_confidence_mod = if cached_binding > BINDING_CONFIDENCE_THRESHOLD {
            let conf_boost =
                (cached_binding - BINDING_CONFIDENCE_THRESHOLD) * BINDING_STRONG_CONFIDENCE_SCALE;
            self.adjust_confidence_weighted(
                "binding_strong",
                conf_boost,
                Priority::Cognitive,
                cached_binding.clamp(0.0, 1.0),
            );
            self.stats.binding_confidence_mod_count += 1;
            conf_boost
        } else if cached_binding < BINDING_LOW_THRESHOLD && cached_binding > 0.0 {
            let conf_dampen =
                (BINDING_LOW_THRESHOLD - cached_binding) * BINDING_WEAK_CONFIDENCE_SCALE;
            self.adjust_confidence("binding_weak", -conf_dampen);
            self.stats.binding_confidence_mod_count += 1;
            -conf_dampen
        } else {
            0.0
        };

        // Coherence gate: skip resonator recall during unstable CfC dynamics
        let reflection_thresholds = self
            .consciousness
            .self_model_tier
            .self_reflection
            .get_thresholds();
        let resonator_coherence_gate = pre_update_coherence > reflection_thresholds.coherence_gate
            || self.stats.total_cycles < DYNAMICS_STARTUP_WARMUP_CYCLES;
        if resonator_coherence_gate && urgency.should_run(self.stats.total_cycles, 1, 1, 4) {
            if let Some(ref mut res_mem) = self.memory.memory_consol.resonator_memory {
                let res_start = Instant::now();

                let res_dim_ok =
                    perception.encoding.compressed_state.len() == res_mem.resonator.config.dim;
                if res_dim_ok && !res_mem.is_empty() {
                    if let Ok(matches) =
                        res_mem.retrieve(&[("content", &perception.encoding.compressed_state)])
                    {
                        // Thalamic depth → recall depth gating
                        // Science: Cowan (2001) — WM capacity scales with attentional focus
                        let recall_k = match self.cognitive_depth {
                            super::super::CognitiveDepth::DeepThought => MEMORY_RECALL_TOP_K * 2,
                            super::super::CognitiveDepth::Cortical => MEMORY_RECALL_TOP_K,
                            super::super::CognitiveDepth::Reflex => 1,
                        };
                        let top_matches: Vec<_> = matches.into_iter().take(recall_k).collect();

                        // Compute similarities once; reuse for both max-sim and argmax.
                        let sims: Vec<f32> = top_matches
                            .iter()
                            .map(|m| {
                                helpers::cosine_f32(&perception.encoding.compressed_state, &m.hv)
                            })
                            .collect();
                        let best_match_sim = sims.iter().copied().fold(0.0f32, f32::max);
                        let match_timestamps: Vec<u64> =
                            top_matches.iter().map(|m| m.timestamp).collect();
                        resonator_best_sim = best_match_sim;

                        if best_match_sim > RESONATOR_SIMILARITY_PRIME_THRESHOLD {
                            let best_idx = sims
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i);
                            if let Some(idx) = best_idx {
                                self.stats.last_resonator_prediction =
                                    Some(top_matches[idx].hv.clone());
                            }
                        }

                        let bundled = if top_matches.len() >= 2 {
                            let dim = perception.encoding.compressed_state.len();
                            let mut b = vec![0.0f32; dim];
                            let n = top_matches.len() as f32;
                            for ep in &top_matches {
                                for (j, &v) in ep.hv.iter().take(dim).enumerate() {
                                    b[j] += v;
                                }
                            }
                            for v in &mut b {
                                *v /= n;
                            }
                            Some(b)
                        } else {
                            None
                        };

                        drop(top_matches);

                        if let Some(bundled) = bundled {
                            if let Ok(factors) = res_mem.query_factorize(
                                &bundled,
                                &[("content", &perception.encoding.compressed_state)],
                            ) {
                                for (label, _hv) in &factors {
                                    match label.as_str() {
                                        "positive" => {
                                            self.behavior.emotion_contagion.valence =
                                                (self.behavior.emotion_contagion.valence + 0.1)
                                                    .clamp(-1.0, 1.0);
                                        }
                                        "negative" => {
                                            self.behavior.emotion_contagion.valence =
                                                (self.behavior.emotion_contagion.valence - 0.1)
                                                    .clamp(-1.0, 1.0);
                                        }
                                        "high" => {
                                            self.adjust_confidence(
                                                "resonator_factor_high",
                                                super::super::thresholds::RESONATOR_FACTOR_HIGH_CONFIDENCE,
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        if best_match_sim > RESONATOR_SIMILARITY_PRIME_THRESHOLD {
                            self.adjust_confidence(
                                "resonator_recall_prime",
                                best_match_sim
                                    * super::super::thresholds::RESONATOR_RECALL_PRIME_SCALE,
                            );
                            resonator_wm_primed = true;
                        }

                        if !match_timestamps.is_empty() {
                            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                                replay.boost_causal_consolidation(
                                    &match_timestamps,
                                    super::super::thresholds::RESONATOR_CAUSAL_CONSOLIDATION_BOOST
                                        as f64,
                                );
                                resonator_reconsolidated = match_timestamps.len();
                            }
                        }
                    }
                }

                module_timings.resonator_recall = res_start.elapsed().as_micros() as u64;
            }
        }

        // Resonator similarity → unified consolidation response.
        // High similarity = familiar pattern → lock precision + slow LR (consolidate).
        // Low similarity = novel pattern → fast LR (rapid encoding).
        // Unified threshold prevents contradictory signals (precision lock without LR slow).
        // Science: McClelland et al. (1995) — complementary learning systems.
        if resonator_best_sim > RESONATOR_CONSOLIDATION_THRESHOLD {
            self.fep.agent.precision.prior_precision = (self.fep.agent.precision.prior_precision
                + (resonator_best_sim - RESONATOR_CONSOLIDATION_THRESHOLD) as f64
                    * super::super::thresholds::RESONATOR_CONSOLIDATION_PRECISION_SCALE)
                .min(super::super::thresholds::RESONATOR_CONSOLIDATION_PRECISION_MAX);
            if self.stats.total_cycles > DYNAMICS_STARTUP_WARMUP_CYCLES {
                self.scale_lr_pri(
                    "resonator_familiar",
                    RESONATOR_FAMILIAR_LR_SCALE,
                    Priority::Aesthetic,
                );
            }
        } else if resonator_best_sim < RESONATOR_NOVEL_THRESHOLD
            && resonator_best_sim > 0.0
            && self.stats.total_cycles > DYNAMICS_STARTUP_WARMUP_CYCLES
        {
            self.scale_lr_pri(
                "resonator_novel",
                RESONATOR_NOVEL_LR_SCALE,
                Priority::Aesthetic,
            );
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.2. Goal System: Apply attention bias from active goals
        // ═══════════════════════════════════════════════════════════════════════
        let goal_attention_bias = self.fep.goal_system.attention_bias();

        if let Some(top) = self.fep.goal_system.top_goal() {
            let goal_priority = top.priority;
            // Session 12 Item 2: Skip goal LR boost during Critical urgency.
            // Critical = recovery mode; goal-chasing works against stability.
            // Science: Yerkes-Dodson (1908) — high arousal impairs goal-directed learning.
            if goal_priority > GOAL_PRIORITY_LR_THRESHOLD
                && !matches!(urgency, super::super::CycleUrgency::Critical)
            {
                let goal_lr_boost = (goal_priority - GOAL_PRIORITY_LR_THRESHOLD)
                    * super::super::thresholds::GOAL_PRIORITY_LR_SCALE;
                self.scale_lr("goal_priority", 1.0 + goal_lr_boost);
            }
            if prediction_error < self.config.learning_threshold
                && goal_priority > GOAL_PRIORITY_EXPLORATION_THRESHOLD
            {
                self.adjust_exploration(
                    "goal_pursuit",
                    goal_priority * super::super::thresholds::GOAL_PURSUIT_EXPLORATION_SCALE,
                );
            }
        }

        super::MemoryBindingResult {
            memory_context_boost,
            resonator_wm_primed,
            resonator_reconsolidated,
            resonator_best_sim,
            resonator_prediction_error,
            resonator_error_exploration_mod,
            binding_threshold_mod,
            binding_confidence_mod,
            pre_update_coherence,
            goal_attention_bias,
        }
    }
}
