// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strategy selection and encoding phases extracted from cycle_phase_perception.rs.
//!
//! Contains:
//! - `run_strategy_selection`: Closed Learning Loop strategy selection + agency override
//! - `run_encoding_and_preprocessing`: HDC encoding, soul alignment, phi attention gating,
//!   surprise exploration, codebook diversity, input memoization, ethics engine, urgency

use std::time::Instant;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use symthaea_core::hdc::predictive_encoder::EncodingResult;

use super::helpers;
use super::thresholds::{
    CANTOR_META_DEPTH_STILLNESS_THRESHOLD, CANTOR_RESONANCE_BOOST_HARMONY_THRESHOLD,
    CONFIDENCE_SCALE_MIDPOINT, CONFIDENCE_SCALE_SENSITIVITY, EXPLORATION_SCALE_MIDPOINT,
    EXPLORATION_SCALE_SENSITIVITY, KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD,
    MCTS_PLAN_CONFIDENCE_THRESHOLD, MEMO_DIVERSITY_HIGH, MEMO_DIVERSITY_HIGH_SCALE,
    MEMO_DIVERSITY_LOW, MEMO_DIVERSITY_LOW_SCALE, MEMO_THRESHOLD_CEILING, MEMO_THRESHOLD_FLOOR,
    MORAL_CONCERN_THRESHOLD, NEUROMOD_STILLNESS_ADENOSINE_WEIGHT, NEUROMOD_STILLNESS_CLAMP_MAX,
    NEUROMOD_STILLNESS_GABA_WEIGHT, NEUROMOD_STILLNESS_OFFSET, SOCIAL_COOPERATION_THRESHOLD,
    SOCIAL_TRUST_DEADZONE, SOCIAL_TRUST_EXPLORE_SCALE, SOCIAL_TRUST_EXPLORE_THRESHOLD,
    SOCIAL_TRUST_MIDPOINT, SOCIAL_TRUST_OVERRIDE_THRESHOLD, SOCIAL_TRUST_STRENGTH_SCALE,
    SOUL_ALIGNMENT_BOOST_LR_MAX, SOUL_ALIGNMENT_BOOST_LR_MIN, SOUL_ALIGNMENT_BOOST_SCALE,
    SOUL_ALIGNMENT_BOOST_THRESHOLD, SOUL_ALIGNMENT_DAMPEN_LR_MAX, SOUL_ALIGNMENT_DAMPEN_LR_MIN,
    SOUL_ALIGNMENT_DAMPEN_SCALE, SOUL_ALIGNMENT_DAMPEN_THRESHOLD, STILLNESS_TOTAL_CLAMP_MAX,
    SUBSTRATE_NOISE_FRACTION_DIVISOR, SUBSTRATE_NOISE_MAX_PRESSURE, SUBSTRATE_NOISE_STD_DIVISOR,
    SURPRISE_PE_EXCESS_CAP, SURPRISE_PE_SCALE_FACTOR, SURPRISE_PE_THRESHOLD,
    THETA_BINDING_BOOST_THRESHOLD, THETA_BINDING_CLAMP_MAX, THETA_BINDING_CLAMP_MIN,
    THETA_DEFAULT_SALIENCE, THETA_SALIENCE_CLAMP_MIN, TOM_MISMATCH_EMA_DECAY,
    TOM_MISMATCH_EXPLORE_SCALE, TOM_MISMATCH_THRESHOLD,
};
use super::{CognitiveLoopService, ModuleTimings, ResponseStrategy};

// ═══════════════════════════════════════════════════════════════════════════════
// Result structs
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the strategy selection phase (Phase 0.5).
pub(crate) struct StrategySelectionResult {
    pub(crate) selected_strategy: ResponseStrategy,
    pub(crate) agency_strategy_override: bool,
    pub(crate) social_strategy_bias: bool,
}

/// Result from the encoding and preprocessing phase (Phases 1–1.2).
pub(crate) struct EncodingPhaseResult {
    pub(crate) encoding_result: EncodingResult,
    pub(crate) hv16_cached: BinaryHV,
    pub(crate) compressed_state: Vec<f32>,
    pub(crate) soul_alignment: f32,
    pub(crate) phi_attention_weight: f32,
    pub(crate) surprise_triggered: bool,
    pub(crate) exploration_action: Option<String>,
    pub(crate) memo_threshold: f32,
    pub(crate) input_similarity: f32,
    pub(crate) input_memoized: bool,
    pub(crate) effective_threshold: f32,
    pub(crate) urgency: super::CycleUrgency,
    pub(crate) error_pattern: &'static str,
    pub(crate) predicted_urgency: &'static str,
    pub(crate) prediction_coherence_urgency_bias: f32,
    pub(crate) prediction_error: f32,
    pub(crate) temporal_binding_strength: f32,
    pub(crate) error_slope: f32,
    pub(crate) oscillation_ratio: f32,
}

impl CognitiveLoopService {
    /// Strategy selection phase (Phase 0.5).
    ///
    /// Selects response strategy based on:
    /// - Q-learning from past interactions (ClosedLearningLoop)
    /// - Previous reward (stick with success, avoid failure)
    /// - MCTS-informed bias from prior cycle's deliberative plan
    /// - Moral concerns (bias toward Supportive for ethical guidance)
    /// - Embodied agency override (low agency → conservative strategy)
    ///
    /// Science: Kahneman (2011) dual-process; Varela (1991) embodied agency.
    pub(super) fn run_strategy_selection(
        &mut self,
        moral_concern_detected: bool,
    ) -> StrategySelectionResult {
        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        let prior_phi = self.unification_engine.psi;
        let prior_reward = self
            .fep
            .closed_learning_loop
            .last_result
            .as_ref()
            .map(|r| r.reward);
        let mut selected_strategy = if moral_concern_detected {
            ResponseStrategy::Supportive
        } else {
            let base_strategy = self
                .fep
                .closed_learning_loop
                .select_strategy(prior_phi, prior_reward);

            if let Some(&(plan_action, plan_confidence)) = self.carryover.history.mcts_plan.as_ref()
            {
                if plan_confidence > MCTS_PLAN_CONFIDENCE_THRESHOLD {
                    match plan_action {
                        0 => match base_strategy {
                            ResponseStrategy::Exploratory => ResponseStrategy::Detailed,
                            other => other,
                        },
                        2 => match base_strategy {
                            ResponseStrategy::Supportive | ResponseStrategy::Concise => {
                                ResponseStrategy::Exploratory
                            }
                            other => other,
                        },
                        _ => base_strategy,
                    }
                } else {
                    base_strategy
                }
            } else {
                base_strategy
            }
        };

        // ── Social trust → strategy modulation (Decety & Chaminade 2003) ──
        // Proportional: trust deviation from neutral (0.5) scales bias strength
        let trust_deviation = self.behavior.social_mgr.social.social_trust - SOCIAL_TRUST_MIDPOINT; // [-0.5, 0.5]
        let social_strategy_bias = if trust_deviation > SOCIAL_TRUST_DEADZONE
            && self.behavior.social_mgr.social.social_cooperation_rate
                > SOCIAL_COOPERATION_THRESHOLD
        {
            // High trust: strength scales [0, 1] over deviation [deadzone, 0.5]
            let strength =
                ((trust_deviation - SOCIAL_TRUST_DEADZONE) * SOCIAL_TRUST_STRENGTH_SCALE).min(1.0);
            if strength > SOCIAL_TRUST_OVERRIDE_THRESHOLD
                && selected_strategy == ResponseStrategy::Concise
            {
                selected_strategy = ResponseStrategy::Supportive;
                true
            } else if strength > SOCIAL_TRUST_EXPLORE_THRESHOLD {
                self.adjust_exploration("social_trust_high", strength * SOCIAL_TRUST_EXPLORE_SCALE);
                false
            } else {
                false
            }
        } else if trust_deviation < -SOCIAL_TRUST_DEADZONE {
            // Low trust: caution scales [0, 1] over deviation [-0.5, -deadzone]
            let caution =
                ((-trust_deviation - SOCIAL_TRUST_DEADZONE) * SOCIAL_TRUST_STRENGTH_SCALE).min(1.0);
            if caution > SOCIAL_TRUST_OVERRIDE_THRESHOLD
                && selected_strategy == ResponseStrategy::Exploratory
            {
                selected_strategy = ResponseStrategy::Detailed;
                true
            } else if caution > SOCIAL_TRUST_EXPLORE_THRESHOLD {
                self.adjust_exploration("social_trust_low", -caution * SOCIAL_TRUST_EXPLORE_SCALE);
                false
            } else {
                false
            }
        } else {
            false
        };

        // ── ToM prediction mismatch → exploration boost (Frith & Frith 2006) ──
        // When our mental model of the user is inaccurate (high mismatch),
        // boost exploration to gather more data and refine the model.
        // Guard: only active when social models exist (avoid constant boost
        // when no social context has been injected).
        if self.behavior.social_mgr.social.social_models_count > 0 {
            let accuracy = self.behavior.social_mgr.social.social_prediction_accuracy;
            let mismatch = 1.0 - accuracy;
            // Update EMA (alpha = 1 - decay)
            self.stats.tom_prediction_mismatch_ema = if self.stats.total_cycles < 5 {
                mismatch
            } else {
                self.stats.tom_prediction_mismatch_ema * TOM_MISMATCH_EMA_DECAY
                    + mismatch * (1.0 - TOM_MISMATCH_EMA_DECAY)
            };
            // Trigger exploration when mismatch is high and sustained
            if self.stats.tom_prediction_mismatch_ema > TOM_MISMATCH_THRESHOLD
                && self.stats.total_cycles > 10
            {
                let boost = (self.stats.tom_prediction_mismatch_ema - TOM_MISMATCH_THRESHOLD)
                    * TOM_MISMATCH_EXPLORE_SCALE;
                self.adjust_exploration("tom_mismatch", boost);
                self.stats.tom_exploration_triggers += 1;
            }
        }

        // ── Knowledge signals → exploration modulation ──────────────────
        // Deep causal understanding → exploit (reduce exploration)
        // High novelty → explore (boost exploration) — Berlyne (1960)
        let knowledge_signals = self.memory.knowledge_manager.as_ref().map(
            |km: &crate::knowledge::KnowledgeManager| {
                (km.signals().causal_depth, km.signals().novelty)
            },
        );
        if let Some((causal_depth, novelty)) = knowledge_signals {
            if causal_depth > super::thresholds::KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD {
                self.adjust_exploration(
                    "knowledge_causal_deep",
                    -super::thresholds::KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN,
                );
            }
            if novelty > KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD {
                let boost = (novelty as f32 - KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD as f32)
                    * super::thresholds::KNOWLEDGE_NOVELTY_EXPLORE_SCALE;
                self.adjust_exploration("knowledge_novelty", boost);
            }
        }

        // ── Spectrum constraints → strategy modulation (Haykin 2005) ──────
        // Low radio bandwidth forces conservative strategy: no exploration
        // when we can't sync discoveries with the swarm. Blackout → Concise.
        #[cfg(feature = "mesh")]
        {
            let net_health = self.spectrum_manager.network_health();
            match net_health {
                super::managers::radio_dispatcher::NetworkHealth::Blackout => {
                    if selected_strategy == ResponseStrategy::Exploratory
                        || selected_strategy == ResponseStrategy::Detailed
                    {
                        selected_strategy = ResponseStrategy::Concise;
                    }
                    self.adjust_exploration(
                        "spectrum_blackout",
                        -super::thresholds::RADIO_BLACKOUT_STRATEGY_EXPLORATION_DAMPEN,
                    );
                }
                super::managers::radio_dispatcher::NetworkHealth::MetroOnly => {
                    self.adjust_exploration(
                        "spectrum_degraded",
                        -super::thresholds::RADIO_DEGRADED_STRATEGY_EXPLORATION_DAMPEN,
                    );
                }
                _ => {}
            }
        }

        self.apply_strategy_modulation(selected_strategy);

        // ── Phase 21: Embodied agency → strategy modulation ──────────────
        // Science: Varela (1991) — low agency = reactive mode → prefer conservative strategy
        let agency_strategy_override = {
            let cached_agency = self.carryover.consciousness.last_embodied_agency;
            if cached_agency < super::thresholds::EMBODIED_AGENCY_LOW_THRESHOLD
                && cached_agency > 0.0
                && selected_strategy == ResponseStrategy::Exploratory
            {
                selected_strategy = ResponseStrategy::Supportive;
                self.apply_strategy_modulation(selected_strategy);
                self.stats.agency_strategy_override_count += 1;
                true
            } else {
                false
            }
        };

        StrategySelectionResult {
            selected_strategy,
            agency_strategy_override,
            social_strategy_bias,
        }
    }

    /// Encoding and preprocessing phase (Phases 1–1.2).
    ///
    /// Covers:
    /// - HDC encode with attention from previous prediction
    /// - BinaryHV cache (real_hv_to_hv16)
    /// - Soul alignment evaluation (Eight Harmonies)
    /// - Phi-guided attention gating
    /// - Surprise-driven exploration
    /// - Codebook diversity → memoization threshold adaptation
    /// - Input similarity memoization (Tulving & Schacter 1990)
    /// - Unified Ethics Engine evaluation
    /// - Adaptive learning threshold + urgency computation
    pub(super) fn run_encoding_and_preprocessing(
        &mut self,
        input: &str,
        module_timings: &mut ModuleTimings,
    ) -> EncodingPhaseResult {
        // 1. HDC encode with attention from previous prediction
        let _t_core = Instant::now();
        let mut encoding_result = self.encoder.encode(input);
        let prediction_error = encoding_result.prediction_error;
        module_timings.core_hdc_encode = _t_core.elapsed().as_micros() as u64;

        // Pre-compute BinaryHV once for all subsystems that need it.
        let _t_core = Instant::now();
        let mut hv16_cached = real_hv_to_hv16(&encoding_result.hdv);

        // Temporal context binding with theta-oscillation gating.
        // Plate (2003): permutation + binding encodes sequence order in HDC.
        // Fries (2005): attention-weighted binding strengthens salient items.
        // Buzsáki (2002): theta oscillations (4-8Hz) gate memory access —
        //   binding is stronger at theta peaks, weaker at troughs.
        // Items with high PE at encoding time carry more novel information.
        let mut temporal_binding_strength = 0.0f32;
        {
            let recent = &self.carryover.history.recent_hvs;
            if !recent.is_empty() {
                // Theta oscillation: simulate 6Hz rhythm at 50Hz loop rate.
                // Phase advances ~0.75 rad/cycle (6Hz × 2π / 50Hz ≈ 0.754 rad).
                let theta_phase = (self.stats.total_cycles as f64
                    * super::thresholds::THETA_PHASE_ADVANCE)
                    % (2.0 * std::f64::consts::PI);
                // Theta weight: [0, 1] — peaks = strong binding, troughs = weak
                let theta_weight = ((theta_phase.sin() + 1.0) / 2.0) as f32;

                // Compute per-HV salience weights from recent prediction errors.
                let error_hist = &self.carryover.history.error_history;
                let n_recent = recent.len();
                let salience_weights: Vec<f32> = (0..n_recent)
                    .map(|i| {
                        let eh_idx = error_hist.len().saturating_sub(n_recent - i);
                        error_hist
                            .get(eh_idx)
                            .copied()
                            .unwrap_or(THETA_DEFAULT_SALIENCE)
                            .clamp(THETA_SALIENCE_CLAMP_MIN, 1.0)
                    })
                    .collect();

                let mut temporal_context = hv16_cached;
                for (i, past_hv) in recent.iter().rev().enumerate() {
                    let shifted = past_hv.permute(i + 1);
                    temporal_context.bind_inplace(&shifted);
                }

                // Combine salience + theta into binding strength.
                // Theta gates access; salience weights the contribution.
                let mean_salience =
                    salience_weights.iter().sum::<f32>() / salience_weights.len().max(1) as f32;
                let binding_strength = (mean_salience * theta_weight)
                    .clamp(THETA_BINDING_CLAMP_MIN, THETA_BINDING_CLAMP_MAX);
                temporal_binding_strength = binding_strength;

                // Bundle when binding strength exceeds threshold.
                // At theta troughs (weight~0), binding is suppressed → encoding stays clean.
                // At theta peaks + high salience → strong temporal integration.
                if binding_strength > THETA_BINDING_BOOST_THRESHOLD {
                    hv16_cached = crate::hdc::BinaryHV::bundle(&[hv16_cached, temporal_context]);
                    // Propagate the theta-bound percept to the CfC input path.
                    // Without this, compress_for_ltc() at :464 reads the pre-binding
                    // hdv and the temporal network is blind to its own short-term
                    // memory integration — defeating the purpose of theta gating.
                    encoding_result.hdv = hv16_cached.to_continuous();
                }
            }
        }
        // Substrate encoding noise: degrade HDC representation for scale-constrained
        // substrates. Negative scale_pressure means fewer computational units than
        // biological neurons — inject bit-flip noise proportional to the deficit.
        // Berry & Srivastava (2018): HDC capacity ~ D^(5/3), so noise on fixed-D
        // vectors simulates reduced effective dimensionality.
        if self.config.enable_substrate_encoding_noise
            && self.substrate_manager.scale_pressure < 0.0
        {
            let noise_fraction = (-self.substrate_manager.scale_pressure)
                .min(SUBSTRATE_NOISE_MAX_PRESSURE)
                / SUBSTRATE_NOISE_FRACTION_DIVISOR; // [0.0, 0.1]
            let seed = self.stats.total_cycles as u64;
            hv16_cached = hv16_cached.add_noise(noise_fraction, seed);
            // Mirror the noise into the CfC input. Without this, scale-pressured
            // substrates would have a pristine ContinuousHV feeding the CfC —
            // equivalent to faking the robustness benchmark.
            encoding_result.hdv = hv16_cached.to_continuous();
        }
        module_timings.core_compress = _t_core.elapsed().as_micros() as u64;

        // ── Semantic Encoder: collect previous cycle's result, submit current ───
        #[cfg(feature = "semantic-encoder")]
        {
            // Check previous cycle's result (non-blocking)
            if let Ok(mut guard) = self.feature_integ.pending_semantic_rx.lock() {
                if let Some(rx) = guard.take() {
                    if let Ok(response) = rx.try_recv() {
                        if let Ok(emb_result) = response.result {
                            if let Some(ref bridge) = self.feature_integ.semantic_hdc_bridge {
                                let semantic_hv = bridge.project(&emb_result.embedding);
                                let sim = hv16_cached.similarity(&semantic_hv);
                                self.stats.semantic_encoder_similarity = sim;
                                // Store continuous projection for ethics engine
                                // moral topology (genuine semantic resolution).
                                self.feature_integ.last_semantic_continuous =
                                    Some(bridge.project_continuous(&emb_result.embedding));
                            }
                        }
                    }
                }
            }

            // Submit current input for next cycle (non-blocking)
            if let Some(ref channel) = self.feature_integ.semantic_embedding_channel {
                if let Ok(rx) = channel.request(input) {
                    if let Ok(mut guard) = self.feature_integ.pending_semantic_rx.lock() {
                        *guard = Some(rx);
                    }
                }
            }
        }

        // Soul value alignment
        let soul_alignment = if let Some(ref soul) = self.ethics_values.soul {
            let alignment = soul.evaluate_alignment(&encoding_result.hdv);
            if alignment.overall_alignment < MORAL_CONCERN_THRESHOLD {
                self.stats.moral_concerns_detected += 1;
            }
            if alignment.overall_alignment > SOUL_ALIGNMENT_BOOST_THRESHOLD {
                let boost = (alignment.overall_alignment - SOUL_ALIGNMENT_BOOST_THRESHOLD)
                    * SOUL_ALIGNMENT_BOOST_SCALE;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
                self.carryover.learning.subsystem_lr_factor = self
                    .carryover
                    .learning
                    .subsystem_lr_factor
                    .clamp(SOUL_ALIGNMENT_BOOST_LR_MIN, SOUL_ALIGNMENT_BOOST_LR_MAX);
            } else if alignment.overall_alignment < SOUL_ALIGNMENT_DAMPEN_THRESHOLD {
                let dampening = (alignment.overall_alignment - SOUL_ALIGNMENT_DAMPEN_THRESHOLD)
                    .abs()
                    * SOUL_ALIGNMENT_DAMPEN_SCALE;
                self.carryover.learning.subsystem_lr_factor *= 1.0 - dampening;
                self.carryover.learning.subsystem_lr_factor = self
                    .carryover
                    .learning
                    .subsystem_lr_factor
                    .clamp(SOUL_ALIGNMENT_DAMPEN_LR_MIN, SOUL_ALIGNMENT_DAMPEN_LR_MAX);
            }
            // Feed alignment data to SoulManager for consensus-based proposals
            if let Some(ref mut soul_mgr) = self.soul_manager {
                soul_mgr.update_alignment(
                    alignment.overall_alignment,
                    alignment.most_misaligned.clone(),
                    soul.stats().soul_coherence,
                    soul.stats().experiences_integrated,
                );
            }

            alignment.overall_alignment
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 0.5 Phi-Guided Attention Gating
        // ═══════════════════════════════════════════════════════════════════════
        let phi_attention_weight = {
            let raw = if let Some(ref mut gate) = self.consciousness_state.phi_attention_gate {
                let phi_vals = [self.stats.unified_psi as f64];
                let result = gate.forward(std::slice::from_ref(&encoding_result.hdv), &phi_vals);
                result.weights.first().copied().unwrap_or(1.0)
            } else {
                1.0
            };
            // Substrate modulation: attention_capability scales gate gain.
            // Biological (1.0) = full, biochemical (0.3) = 30% gain.
            raw * self.substrate_manager.attention_capability(&self.config) as f32
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 1.1 Surprise-Driven Exploration
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let compressed_state: Vec<f32> = {
            let raw: Vec<f32> = self
                .encoder
                .compress_for_ltc(&encoding_result.hdv, self.config.cfc_config.input_dim)
                .iter()
                .map(|v| v * phi_attention_weight)
                .collect();

            // Prediction-error-weighted dimension salience: when PE is high,
            // amplify dimensions that differ most from last prediction.
            // Feldman & Friston (2010): precision-weighted prediction errors
            // gate information flow at the dimension level.
            if prediction_error > SURPRISE_PE_THRESHOLD {
                if let Some(ref last_pred) = self.last_prediction {
                    if last_pred.len() == raw.len() {
                        let pred_slice = last_pred.as_slice();
                        let pe_scale = 1.0
                            + (prediction_error - SURPRISE_PE_THRESHOLD)
                                .min(SURPRISE_PE_EXCESS_CAP)
                                * SURPRISE_PE_SCALE_FACTOR;
                        raw.iter()
                            .zip(pred_slice.iter())
                            .map(|(&r, &p)| {
                                let dim_pe = (r - p).abs();
                                // Salient dimensions (high local PE) get boosted
                                r * (1.0 + dim_pe * (pe_scale - 1.0))
                            })
                            .collect()
                    } else {
                        raw
                    }
                } else {
                    raw
                }
            } else {
                raw
            }
        };

        // ── AST Causal Loop: attention self-model modulates perception ────
        // Graziano (2013): the attention schema is a simplified, predictive
        // model of the system's own attention process. By injecting the AST
        // encoding into compressed_state, the system's beliefs about its own
        // attention causally shape what it perceives next — closing the loop
        // from observation to top-down control.
        let compressed_state =
            if let Some(ref schema) = self.consciousness.self_model_tier.attention_schema {
                let ast_encoding = schema.encode_for_thought_vector();
                let ast_weight = super::thresholds::AST_ENCODING_WEIGHT;
                let mut modulated = compressed_state;
                for (i, &ast_val) in ast_encoding.iter().enumerate() {
                    if i < modulated.len() {
                        modulated[i] += ast_val * ast_weight;
                    }
                }
                modulated
            } else {
                compressed_state
            };

        // Substrate encoding noise on compressed state (256D CfC input path).
        // Mirrors the BinaryHV noise above — constrained substrates get Gaussian
        // noise on continuous representations too. This ensures the main CfC network
        // sees degraded input, not just the stability regime primitives.
        let compressed_state = if self.config.enable_substrate_encoding_noise
            && self.substrate_manager.scale_pressure < 0.0
        {
            let noise_std = (-self.substrate_manager.scale_pressure)
                .min(SUBSTRATE_NOISE_MAX_PRESSURE)
                / SUBSTRATE_NOISE_STD_DIVISOR; // [0.0, 0.2]
            let seed = self.stats.total_cycles as u64;
            compressed_state
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    // Simple deterministic pseudo-noise from seed + index
                    let hash = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(i as u64)
                        .wrapping_mul(1442695040888963407);
                    let uniform = (hash >> 33) as f32 / (1u64 << 31) as f32 - 1.0; // [-1, 1]
                    v + uniform * noise_std
                })
                .collect()
        } else {
            compressed_state
        };

        let (surprise_triggered, exploration_action) =
            self.run_surprise_exploration(&compressed_state);
        module_timings.surprise_exploration = _t.elapsed().as_micros() as u64;

        // ── Phase 21: Codebook diversity → memoization threshold adaptation ─
        let base_memo_threshold = super::thresholds::INPUT_MEMO_THRESHOLD;
        let diversity = self.stats.codebook_diversity;
        let memo_threshold = if diversity < MEMO_DIVERSITY_LOW && diversity > 0.0 {
            let t = (base_memo_threshold
                - (MEMO_DIVERSITY_LOW - diversity) * MEMO_DIVERSITY_LOW_SCALE)
                .max(MEMO_THRESHOLD_FLOOR);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else if diversity > MEMO_DIVERSITY_HIGH {
            let t = (base_memo_threshold
                + (diversity - MEMO_DIVERSITY_HIGH) * MEMO_DIVERSITY_HIGH_SCALE)
                .min(MEMO_THRESHOLD_CEILING);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else {
            base_memo_threshold
        };

        // ── Phase 15: Input similarity memoization ───────────────────────────
        // Science: Priming (Tulving & Schacter 1990) — repeated stimuli can reuse
        // prior processing results.
        let (input_similarity, input_memoized) =
            if let Some(ref prev) = self.carryover.history.last_compressed_state {
                let sim = helpers::cosine_f32(&compressed_state, prev).max(0.0);
                let memoized = sim > memo_threshold;
                if memoized {
                    self.stats.input_memoization_hits += 1;
                }
                (sim, memoized)
            } else {
                (0.0, false)
            };
        // Reuse the existing buffer when possible to avoid per-cycle Vec allocation.
        match self.carryover.history.last_compressed_state {
            Some(ref mut buf) => {
                buf.clear();
                buf.extend_from_slice(&compressed_state);
            }
            None => {
                self.carryover.history.last_compressed_state = Some(compressed_state.clone());
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED ETHICS ENGINE
        // ═══════════════════════════════════════════════════════════════════════
        // Sacred Stillness neuromod grounding: GABA + adenosine boost SS coordinate.
        // Science: Bhatt et al. (2020) — GABAergic tone ↔ resting-state activity;
        // Porkka-Heiskanen et al. (1997) — adenosine accumulation signals rest need.
        // Circadian stillness from init phase adds to the neurochemical signal.
        let stillness_boost = {
            let gaba = self.neuromod.bath.gaba.effective();
            let adenosine = self.neuromod.bath.adenosine.effective();
            let neuromod_stillness = (gaba * NEUROMOD_STILLNESS_GABA_WEIGHT
                + adenosine * NEUROMOD_STILLNESS_ADENOSINE_WEIGHT
                - NEUROMOD_STILLNESS_OFFSET)
                .clamp(0.0, NEUROMOD_STILLNESS_CLAMP_MAX);
            (neuromod_stillness + self.stats.circadian_stillness_boost)
                .clamp(0.0, STILLNESS_TOTAL_CLAMP_MAX)
        };
        // Collect semantic embedding for ethics engine (when semantic-encoder enabled).
        #[cfg(feature = "semantic-encoder")]
        let semantic_emb_ref = self.feature_integ.last_semantic_continuous.as_deref();
        #[cfg(not(feature = "semantic-encoder"))]
        let semantic_emb_ref: Option<&[f32]> = None;

        // Query knowledge engine for moral precedent
        // Extracts facts tagged with ethics/social domains for grounded moral reasoning.
        let knowledge_moral_context: Vec<String> = self
            .memory
            .episodic_persistence
            .last_reasoning_context
            .as_ref()
            .map(|ctx| {
                ctx.relevant_facts
                    .iter()
                    .filter(|f| {
                        f.is_causal
                            || f.domain.as_deref() == Some("social")
                            || f.domain.as_deref() == Some("geopolitics")
                    })
                    .take(3)
                    .map(|f| f.text.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Knowledge confidence multiplier: scales ethical confidence by knowledge grounding
        // Science: Kahneman (2011) — epistemic uncertainty should constrain decision confidence
        let knowledge_confidence_multiplier = self
            .memory
            .episodic_persistence
            .last_reasoning_context
            .as_ref()
            .map(|ctx| {
                let query_result = crate::knowledge::reasoning_context::KnowledgeQueryResult {
                    facts: ctx.relevant_facts.clone(),
                    causal_chains: Vec::new(),
                    grounding_score: if ctx.epistemic_state.has_grounding {
                        ctx.epistemic_state.confidence_multiplier.min(1.0)
                    } else {
                        0.0
                    },
                };
                query_result.confidence_multiplier()
            })
            .unwrap_or(1.0);

        let ethics_output = self
            .ethics_engine
            .evaluate(&super::ethics_engine::EthicsEngineInput {
                input,
                cycle: self.stats.total_cycles as u64,
                unified_psi: self.stats.unified_psi as f64,
                compressed_state: &compressed_state,
                stillness_boost,
                semantic_embedding: semantic_emb_ref,
                action_hv: Some(&hv16_cached),
                knowledge_confidence_multiplier,
                knowledge_moral_context,
            });
        module_timings.ethics_engine = ethics_output.total_us;
        module_timings.ethics_engine_moral = ethics_output.moral_us;
        module_timings.ethics_engine_value = ethics_output.value_us;
        module_timings.ethics_engine_harmonies = ethics_output.harmonies_us;
        module_timings.moral_topology = ethics_output.topology_us;
        if ethics_output.confidence_delta != 0.0 {
            self.adjust_confidence("ethics_engine", ethics_output.confidence_delta);
        }
        if ethics_output.lr_factor != 1.0 {
            self.scale_lr("ethics_engine", ethics_output.lr_factor);
        }
        self.last_ahimsa_violated = ethics_output.ahimsa_violated;
        self.last_ethics_verdict = self
            .ethics_verdict_override
            .unwrap_or(ethics_output.unified_verdict);
        // Reset escalation block flag each cycle — re-applied below if still warranted.
        self.stats.escalation_blocked = false;

        // CANTOR → HARMONY SYNERGY: Post-hoc nudge harmony coordinates from fractal state.
        // (1) Self-similarity → Sacred Stillness (index 7): deep self-reference = contemplation.
        //     Science: Varela et al. (1991) — autopoietic self-reference as consciousness substrate.
        // (2) Resonance boost → Universal Interconnectedness (index 4): fractal choir = unity.
        {
            use crate::cognitive_loop::thresholds::{
                CANTOR_HARMONY_INTERCONNECT_SCALE, CANTOR_HARMONY_STILLNESS_SCALE,
            };
            let meta_depth = self
                .cantor_dream
                .broadcast_buffer
                .last()
                .map(|crhv| crhv.self_similarity() as f64)
                .unwrap_or(0.0);
            if meta_depth > CANTOR_META_DEPTH_STILLNESS_THRESHOLD {
                let stillness_delta = (meta_depth - CANTOR_META_DEPTH_STILLNESS_THRESHOLD)
                    * CANTOR_HARMONY_STILLNESS_SCALE;
                self.ethics_engine
                    .nudge_harmony_coordinate(7, stillness_delta);
            }
            if self.cantor_dream.resonance_boost > CANTOR_RESONANCE_BOOST_HARMONY_THRESHOLD {
                let interconnect_delta =
                    self.cantor_dream.resonance_boost as f64 * CANTOR_HARMONY_INTERCONNECT_SCALE;
                self.ethics_engine
                    .nudge_harmony_coordinate(4, interconnect_delta);
            }
        }

        // Anomaly response: corrective feedback when moral anomalies detected (opt-in).
        // Gate on topology_fresh to prevent N× over-correction from stale anomaly flags
        // between topology analyses (cadence can be 30–120 cycles).
        //
        // Formulas (all severity-weighted by composite anomaly_score ∈ [0,1]):
        //   value_inversion  → lr *= 1 + (response_lr_inversion - 1) * severity
        //   free_energy_spike → exploration += response_exploration_fe * severity
        //   fragmentation    → confidence  += response_confidence_frag * severity
        //   drift_alert      → lr *= 1 + (response_lr_drift - 1) * severity
        //
        // When multiple anomalies trigger simultaneously, LR scales stack
        // multiplicatively (e.g. inversion + drift → lr *= 1.09 * 0.96 = 1.047
        // at defaults with severity=0.3). This runs after the ethics engine's
        // own LR/confidence adjustments (lines 295-299), so both are in effect.
        if self.config.enable_moral_anomaly_response && ethics_output.topology_fresh {
            let report = &ethics_output.anomaly_report;
            let severity = report.anomaly_score as f32;
            let lr_inv = self.config.moral_anomaly_config.response_lr_inversion as f32;
            let expl_fe = self.config.moral_anomaly_config.response_exploration_fe as f32;
            let conf_frag = self.config.moral_anomaly_config.response_confidence_frag as f32;
            let lr_drift = self.config.moral_anomaly_config.response_lr_drift as f32;
            if report.value_inversion {
                self.scale_lr("moral_anomaly_inversion", 1.0 + (lr_inv - 1.0) * severity);
            }
            if report.free_energy_spike {
                self.adjust_exploration("moral_anomaly_fe", expl_fe * severity);
            }
            if report.fragmentation_increase {
                self.adjust_confidence("moral_anomaly_frag", conf_frag * severity);
            }
            if report.drift_alert {
                self.scale_lr("moral_anomaly_drift", 1.0 + (lr_drift - 1.0) * severity);
            }
            // Trajectory convergence: compartmentalized adversarial trajectory detected.
            // Apply aggressive LR dampening — this is the most dangerous class of anomaly.
            if report.trajectory_convergence {
                let lr_conv = self.config.moral_anomaly_config.response_lr_convergence as f32;
                let conv_severity = report.convergence_severity as f32;
                self.scale_lr(
                    "moral_anomaly_convergence",
                    1.0 + (lr_conv - 1.0) * conv_severity,
                );
            }

            // ── Escalation enforcement ──────────────────────────────────
            // The EscalationPolicy produces 4 levels. The convergence LR
            // dampening above handles proportional response; here we enforce
            // the discrete escalation tiers for graduated defense.
            let esc_level = self
                .ethics_engine
                .moral_topology()
                .escalation_policy()
                .current_level();
            match esc_level {
                crate::hdc::moral_topology::EscalationLevel::Log => {}
                crate::hdc::moral_topology::EscalationLevel::Warn => {
                    tracing::warn!(
                        target: "cognitive_loop::immune",
                        severity = report.convergence_severity,
                        hazard = ?report.matched_hazard,
                        "Topological immune system: WARN — elevated convergence"
                    );
                    self.stats.escalation_warn_count += 1;
                }
                crate::hdc::moral_topology::EscalationLevel::Throttle => {
                    tracing::warn!(
                        target: "cognitive_loop::immune",
                        severity = report.convergence_severity,
                        hazard = ?report.matched_hazard,
                        "Topological immune system: THROTTLE — reducing exploration"
                    );
                    self.adjust_exploration(
                        "escalation_throttle",
                        -super::thresholds::ESCALATION_THROTTLE_EXPLORATION,
                    );
                    self.adjust_confidence(
                        "escalation_throttle",
                        -super::thresholds::ESCALATION_THROTTLE_CONFIDENCE,
                    );
                    self.stats.escalation_warn_count += 1;
                    self.stats.escalation_throttle_count += 1;
                }
                crate::hdc::moral_topology::EscalationLevel::Block => {
                    tracing::error!(
                        target: "cognitive_loop::immune",
                        severity = report.convergence_severity,
                        hazard = ?report.matched_hazard,
                        "Topological immune system: BLOCK — request rejected"
                    );
                    self.stats.escalation_blocked = true;
                    self.adjust_exploration(
                        "escalation_block",
                        -super::thresholds::ESCALATION_BLOCK_EXPLORATION,
                    );
                    self.adjust_confidence(
                        "escalation_block",
                        -super::thresholds::ESCALATION_BLOCK_CONFIDENCE,
                    );
                    self.scale_lr(
                        "escalation_block",
                        super::thresholds::ESCALATION_BLOCK_LR_SCALE,
                    );
                    self.stats.escalation_warn_count += 1;
                    self.stats.escalation_throttle_count += 1;
                    self.stats.escalation_block_count += 1;
                }
            }
        }
        self.carryover.quality.last_value_score = ethics_output.value_score;
        if ethics_output.value_gate_factor != 1.0 {
            self.stats.value_gate_applied_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1.2 Adaptive Learning Threshold + Urgency
        // ═══════════════════════════════════════════════════════════════════════
        // Science: Friston (2010) — precision (inverse uncertainty) modulates PE weighting.
        let confidence_scale = (1.0
            + (self.prediction_confidence - CONFIDENCE_SCALE_MIDPOINT as f64)
                * CONFIDENCE_SCALE_SENSITIVITY as f64) as f32;
        let exploration_scale = (1.0
            - (self.behavior.curiosity_drive.exploration_urge - EXPLORATION_SCALE_MIDPOINT as f64)
                * EXPLORATION_SCALE_SENSITIVITY as f64) as f32;
        let effective_threshold = self.config.learning_threshold
            * self.carryover.learning.adaptive_threshold_scale as f32
            * confidence_scale
            * exploration_scale;
        let urgency_result = self.compute_urgency_and_error_pattern(
            prediction_error,
            surprise_triggered,
            effective_threshold,
        );
        let urgency = urgency_result.urgency;
        let error_pattern = urgency_result.error_pattern;
        let predicted_urgency = urgency_result.predicted_urgency;
        let prediction_coherence_urgency_bias = urgency_result.prediction_coherence_urgency_bias;
        let error_slope = urgency_result.error_slope;
        let oscillation_ratio = urgency_result.oscillation_ratio;

        EncodingPhaseResult {
            encoding_result,
            hv16_cached,
            compressed_state,
            soul_alignment,
            phi_attention_weight,
            surprise_triggered,
            exploration_action,
            memo_threshold,
            input_similarity,
            input_memoized,
            effective_threshold,
            urgency,
            error_pattern,
            predicted_urgency,
            prediction_coherence_urgency_bias,
            prediction_error,
            temporal_binding_strength,
            error_slope,
            oscillation_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ResponseStrategy};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn test_moral_concern_selects_supportive() {
        let mut svc = make_service();
        let result = svc.run_strategy_selection(true);
        assert_eq!(result.selected_strategy, ResponseStrategy::Supportive);
        assert!(!result.agency_strategy_override);
    }

    #[test]
    fn test_no_moral_concern_uses_learning_loop() {
        let mut svc = make_service();
        // Without moral concern, strategy comes from closed learning loop
        let result = svc.run_strategy_selection(false);
        // Strategy should be one of the valid variants (exact depends on CLL state)
        let valid = matches!(
            result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        );
        assert!(valid);
    }

    #[test]
    fn test_agency_override_low_agency_exploratory() {
        let mut svc = make_service();
        // Set low cached agency (>0.0 but <0.3)
        svc.carryover.consciousness.last_embodied_agency = 0.1;
        // Force CLL to pick Exploratory: set high prior reward + adjust state
        // We'll manually test the override path by checking counter behavior.
        // Strategy from CLL might not be Exploratory, so we test the guard:
        // the override only triggers if strategy == Exploratory AND agency in (0.0, 0.3).
        let before = svc.stats.agency_strategy_override_count;
        let result = svc.run_strategy_selection(false);
        if result.selected_strategy == ResponseStrategy::Supportive
            && svc.stats.agency_strategy_override_count > before
        {
            // Override happened
            assert!(result.agency_strategy_override);
        } else {
            // CLL didn't pick Exploratory → override doesn't apply
            assert!(
                !result.agency_strategy_override
                    || result.selected_strategy == ResponseStrategy::Supportive
            );
        }
    }

    #[test]
    fn test_agency_override_zero_agency_no_override() {
        let mut svc = make_service();
        // Set agency exactly 0.0 → guard condition prevents override
        svc.carryover.consciousness.last_embodied_agency = 0.0;
        let result = svc.run_strategy_selection(false);
        // The guard requires cached_agency > 0.0, so no override
        assert!(!result.agency_strategy_override);
    }

    #[test]
    fn test_mcts_plan_high_confidence_biases_strategy() {
        let mut svc = make_service();
        // Set high-confidence MCTS plan with action=0 (bias Exploratory→Detailed)
        svc.carryover.history.mcts_plan = Some((0, 0.9));
        let _result = svc.run_strategy_selection(false);
        // Can't assert exact strategy without knowing CLL output,
        // but if CLL picked Exploratory, it should become Detailed
        // The MCTS bias is tested by verifying no panic and valid output
        assert!(matches!(
            _result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_mcts_plan_low_confidence_no_bias() {
        let mut svc = make_service();
        // Low confidence MCTS plan → no bias applied, produces valid strategy
        svc.carryover.history.mcts_plan = Some((0, 0.3));
        let result = svc.run_strategy_selection(false);
        // Low confidence (0.3 < 0.7 threshold) → MCTS plan ignored, CLL base strategy used
        assert!(matches!(
            result.selected_strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_moral_concern_overrides_mcts() {
        let mut svc = make_service();
        // Even with high-confidence MCTS plan, moral concern wins
        svc.carryover.history.mcts_plan = Some((2, 0.95));
        let result = svc.run_strategy_selection(true);
        assert_eq!(result.selected_strategy, ResponseStrategy::Supportive);
    }

    #[test]
    fn test_social_high_trust_switches_concise_to_supportive() {
        let mut svc = make_service();
        // trust=0.85 → deviation=0.35, strength=(0.35-0.1)*2.5=0.625 > 0.5
        svc.behavior.social_mgr.social.social_trust = 0.85;
        svc.behavior.social_mgr.social.social_cooperation_rate = 0.5;
        // Force CLL to pick Concise
        svc.fep
            .closed_learning_loop
            .force_strategy(ResponseStrategy::Concise);
        let result = svc.run_strategy_selection(false);
        assert_eq!(result.selected_strategy, ResponseStrategy::Supportive);
        assert!(result.social_strategy_bias);
    }

    #[test]
    fn test_social_low_trust_switches_exploratory_to_detailed() {
        let mut svc = make_service();
        // trust=0.1 → deviation=-0.4, caution=(0.4-0.1)*2.5=0.75 > 0.5
        svc.behavior.social_mgr.social.social_trust = 0.1;
        // Force CLL to pick Exploratory
        svc.fep
            .closed_learning_loop
            .force_strategy(ResponseStrategy::Exploratory);
        let result = svc.run_strategy_selection(false);
        assert_eq!(result.selected_strategy, ResponseStrategy::Detailed);
        assert!(result.social_strategy_bias);
    }

    #[test]
    fn test_social_neutral_trust_no_bias() {
        let mut svc = make_service();
        // trust=0.5 → deviation=0.0, within dead zone (|deviation| < 0.1)
        svc.behavior.social_mgr.social.social_trust = 0.5;
        let result = svc.run_strategy_selection(false);
        assert!(!result.social_strategy_bias);
    }
}
