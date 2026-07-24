// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Internal helper methods for CognitiveLoopService.
//!
//! Contains experience creation, statistics updates, error trend computation,
//! prediction confidence tracking, reset, and neural bridge processing.
//!
//! Sub-modules hold extracted cycle() helpers:
//! - `cycle_extracted`: Phases 1–4 (methods on `&mut self`)
//! - `parallel`: Phase 5 free functions for `rayon::join` branches
#![allow(unexpected_cfgs)]

mod cycle_extracted;
mod cycle_phases;
mod cycle_phases_dream;
mod cycle_phases_init_stats;
mod cycle_phases_memory;
mod cycle_phases_urgency;
mod feedback_helpers;
mod parallel;

// Re-export Phase 5 items so cycle.rs can use `helpers::run_stability_regime` etc.
pub(super) use parallel::{
    EpisodicLearningContext, parallel_episodic_learning, parallel_semantic_causal,
    run_stability_regime,
};

// Re-export consciousness metrics parallel branch items (Phase "consciousness metrics").
pub(in crate::cognitive_loop) use parallel::{
    ConsciousnessMetricsBranchA, ConsciousnessMetricsBranchB, DeferredFeedback,
    parallel_consciousness_branch_a, parallel_consciousness_branch_b,
};

// Re-export Phase 7 result structs so cycle.rs can destructure them
pub(super) use cycle_phases::{DreamPhaseResult, EpisodicReplayResult, ResonatorCodebookResult};

// Constants formerly re-exported here now live in `thresholds.rs`.

use super::CycleCarryover;
use crate::consciousness::fep_active_inference::ActiveInferenceAgent;
use crate::dynamics::temporal_signatures::ConsciousnessPattern;
#[cfg(feature = "neural-bridge")]
use anyhow::Result;
use ndarray::Array1;
use std::time::{Duration, Instant};

#[cfg(feature = "neural-bridge")]
use super::CycleResult;
use super::thresholds::{
    ERROR_EMA_ALPHA, EXPERIENCE_BASE_IMPORTANCE, PRED_COHERENCE_HIGH_BOOST_SCALE,
    PRED_COHERENCE_HIGH_THRESHOLD, PRED_COHERENCE_LOW_DAMPEN_SCALE, PRED_COHERENCE_LOW_THRESHOLD,
    PREFRONTAL_FLOOR, PREFRONTAL_GRADUATION_WEIGHT, PREFRONTAL_UTILIZATION_WEIGHT,
    TIMING_EMA_ALPHA,
};
use super::{ActionHint, AdaptiveBehavior, CognitiveLoopService, Experience, LoopStats};

/// Cosine similarity between two f32 slices.
///
/// Returns 0.0 for mismatched lengths or empty/zero-norm vectors.
/// Delegates to `symthaea_core::hdc::simd_continuous::similarity_simd` which
/// uses AVX2+FMA SIMD when available. This eliminates a local duplicate
/// implementation.
#[inline]
pub(super) fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    if a.is_empty() {
        return 0.0;
    }
    let sim = symthaea_core::hdc::simd_continuous::similarity_simd(a, b);
    // Guard: SIMD path may overflow to NaN/Inf for very large magnitudes
    if sim.is_finite() {
        sim.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Safe arithmetic mean for f32 slices. Returns `default` if slice is empty.
#[inline]
pub(crate) fn safe_mean_f32(values: &[f32], default: f32) -> f32 {
    if values.is_empty() {
        default
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Safe arithmetic mean for f64 slices. Returns `default` if slice is empty.
#[inline]
pub(crate) fn safe_mean_f64(values: &[f64], default: f64) -> f64 {
    if values.is_empty() {
        default
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

impl CognitiveLoopService {
    /// Process a pre-computed text embedding through the neural bridge and
    /// cognitive loop.
    ///
    /// Pipeline: embedding (e.g. BGE-M3 768-d) -> NeuralBridge linear probe
    /// -> 16384-d HDC vector -> compress -> CfC temporal processing -> CycleResult.
    ///
    /// This bypasses the text-based HDC encoder and instead uses a trained
    /// probe to project dense embeddings directly into HDC space, giving
    /// the cognitive loop access to rich semantic representations.
    #[cfg(feature = "neural-bridge")]
    pub fn process_text_input(&mut self, embedding: &[f32]) -> Result<CycleResult> {
        use symthaea_core::hdc::ContinuousHV;

        let bridge =
            self.feature_integ.neural_bridge.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Neural bridge not loaded (no probe weights found)")
            })?;

        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // 1. Project embedding -> continuous HDC vector (16384-d)
        let hdc_continuous = bridge.project(embedding)?;

        // 2. Wrap as ContinuousHV so we can reuse compress_for_ltc
        let hdv = ContinuousHV::from_vec(hdc_continuous);

        // 3. Compress HDC -> CfC input dimension via random projection
        let compressed_state = self
            .encoder
            .compress_for_ltc(&hdv, self.config.cfc_config.input_dim);

        // 4. Convert to ndarray and step the temporal network
        let input_array = Array1::from_iter(compressed_state.iter().copied());
        let delta_t = self.config.cfc_config.delta_t;
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(err = %e, "temporal_network.step failed in cycle_with_embedding");
        }

        // 5. Multi-scale prediction
        let (prediction, raw_predictions) = self.get_multi_scale_prediction(&input_array);

        // 5b. Use prediction coherence as confidence signal
        if raw_predictions.len() >= 2 {
            let coh = Self::compute_prediction_coherence_from_cache(&raw_predictions);
            // High agreement → boost confidence, low → reduce
            if coh > PRED_COHERENCE_HIGH_THRESHOLD {
                self.adjust_confidence(
                    "embed_pred_coherence_high",
                    (coh - PRED_COHERENCE_HIGH_THRESHOLD) * PRED_COHERENCE_HIGH_BOOST_SCALE,
                );
            } else if coh < PRED_COHERENCE_LOW_THRESHOLD {
                self.scale_confidence(
                    "embed_pred_coherence_low",
                    1.0 - (PRED_COHERENCE_LOW_THRESHOLD - coh) * PRED_COHERENCE_LOW_DAMPEN_SCALE,
                );
            }
        }

        // 6. Read CfC output state
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|e| {
                tracing::warn!(err = %e, "temporal_network.read_state failed — using zero state");
                vec![0.0; self.config.cfc_config.num_neurons]
            });

        // 7. Feed prediction back to encoder: deferred to after last &prediction use

        // 8. Compute prediction error against previous prediction
        let prediction_error = if let Some(ref prev) = self.last_prediction {
            let n = compressed_state.len().min(prev.len());
            if n == 0 {
                0.0
            } else {
                compressed_state[..n]
                    .iter()
                    .zip(prev[..n].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / n as f32
            }
        } else {
            0.0
        };

        // 9. Store experience
        self.create_experience(&compressed_state, &prediction, prediction_error);

        // 7 (deferred). Move prediction to encoder (no clone needed)
        self.encoder.set_prediction(prediction);

        // 10. Learning step: consolidate periodically
        let mut learning_occurred = false;
        let mut training_loss = None;
        if self.config.enable_consolidation && self.stats.total_cycles % 50 == 0 {
            match self.consolidate() {
                Ok(loss) => {
                    if loss > 0.0 {
                        learning_occurred = true;
                        training_loss = Some(loss);
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, cycle = self.stats.total_cycles, "Consolidation failed");
                }
            }
        }

        // 11. Update error history — capacity bound: 100 elements, evict before push
        if self.error_history.len() >= super::thresholds::ERROR_HISTORY_CAPACITY {
            self.error_history.pop_front();
        }
        self.error_history.push_back(prediction_error);
        self.stats.avg_prediction_error =
            self.error_history.iter().sum::<f32>() / self.error_history.len().max(1) as f32;

        // Pre-compute identity fields before moving output
        #[cfg(feature = "identity")]
        let signed_output = self
            .mfdi_bridge
            .sign_output(&output)
            .map_err(|e| {
                tracing::debug!(error = %e, "MFDI output signing failed");
                e
            })
            .ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        Ok(CycleResult {
            output,
            prediction_error,
            peak_attention: 0.0, // No text-based attention for embedding input
            detected_primitives: Vec::new(), // No text primitives for embedding input
            learning_occurred,
            training_loss,
            // Embedding path bypasses the text encoder's bits-saved diagnostics.
            bits_saved_persist: None,
            bits_saved_zero: None,
            bits_kappa: None,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata: super::CycleMetadata::default(),
            thought_vector: vec![0.0; 32],
            wisdom_hv: symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(&hdv),
            language_output: None,
            language_source: None,
            #[cfg(feature = "canvas")]
            canvas_svg: None,
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
        })
    }

    /// Process a pre-encoded hypervector through the cognitive loop.
    ///
    /// Pipeline: ContinuousHV -> compress -> CfC temporal processing -> predict -> learn -> CycleResult.
    ///
    /// This bypasses the text-based HDC encoder and instead accepts a pre-encoded
    /// hypervector directly. Use this for non-text modalities like:
    /// - Image classification (MNIST, ISOLET): encode pixels/features into HDC space,
    ///   then feed through CfC for temporal/consciousness processing.
    /// - Sensor data: encode sensor readings via HDC, feed to consciousness pipeline.
    /// - Pre-computed embeddings: any data already in HDC hypervector form.
    ///
    /// The CfC output state can be used as a consciousness-enriched representation
    /// for downstream classification or decision-making.
    pub fn cycle_with_hv(&mut self, hdv: &symthaea_core::hdc::ContinuousHV) -> super::CycleResult {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // 1. Compress HDC -> CfC input dimension via random projection
        let compressed_state = self
            .encoder
            .compress_for_ltc(hdv, self.config.cfc_config.input_dim);

        // 2. Convert to ndarray and step the temporal network
        let input_array = Array1::from_iter(compressed_state.iter().copied());
        let delta_t = self.config.cfc_config.delta_t;
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(err = %e, "temporal_network.step failed in cycle_no_embedding");
        }

        // 3. Multi-scale prediction
        let (prediction, raw_predictions) = self.get_multi_scale_prediction(&input_array);

        // 3b. Use prediction coherence as confidence signal
        if raw_predictions.len() >= 2 {
            let coh = Self::compute_prediction_coherence_from_cache(&raw_predictions);
            if coh > PRED_COHERENCE_HIGH_THRESHOLD {
                self.adjust_confidence(
                    "hv_pred_coherence_high",
                    (coh - PRED_COHERENCE_HIGH_THRESHOLD) * PRED_COHERENCE_HIGH_BOOST_SCALE,
                );
            } else if coh < PRED_COHERENCE_LOW_THRESHOLD {
                self.scale_confidence(
                    "hv_pred_coherence_low",
                    1.0 - (PRED_COHERENCE_LOW_THRESHOLD - coh) * PRED_COHERENCE_LOW_DAMPEN_SCALE,
                );
            }
        }

        // 4. Read CfC output state
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|e| {
                tracing::warn!(err = %e, "temporal_network.read_state failed — using zero state");
                vec![0.0; self.config.cfc_config.num_neurons]
            });

        // 5. Feed prediction back to encoder: deferred to after last &prediction use

        // 6. Compute prediction error against previous prediction
        let prediction_error = if let Some(ref prev) = self.last_prediction {
            let n = compressed_state.len().min(prev.len());
            if n == 0 {
                0.0
            } else {
                compressed_state[..n]
                    .iter()
                    .zip(prev[..n].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / n as f32
            }
        } else {
            0.0
        };

        // 7. Store experience
        self.create_experience(&compressed_state, &prediction, prediction_error);

        // 5 (deferred). Move prediction to encoder (no clone needed)
        self.encoder.set_prediction(prediction);

        // 8. Update coherence bridge with current tau values (zero-clone on CfC hot path)
        self.temporal_network.with_tau_refs(|refs| {
            self.language_comm.voice_coherence.bridge.update(refs);
        });
        let coherence = self
            .language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence();

        // Effective threshold matches cycle() behavior (adaptive scaling)
        let effective_threshold = self.config.learning_threshold
            * self.carryover.learning.adaptive_threshold_scale as f32;

        // 9. Learn if error is significant
        let (learning_occurred, training_loss) = if prediction_error > effective_threshold {
            self.stats.learning_cycles += 1;
            if let Some(ref prev_state) = self.last_state {
                // Build arrays from iterators — avoids 3 unnecessary Vec clones
                let train_input: Array1<f32> = prev_state.iter().copied().collect();
                let train_target: Array1<f32> = compressed_state.iter().copied().collect();
                let lr = self.config.cfc_config.learning_rate;
                match self.temporal_network.train_step_bptt(
                    &train_input,
                    &train_target,
                    delta_t,
                    lr,
                ) {
                    Ok(loss) => {
                        self.update_loss_stats(loss);
                        (true, Some(loss))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "CfC embedding training step failed");
                        (false, None)
                    }
                }
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        // 10. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());
        self.stats.temporal_coherence = coherence;

        // 10b. Modality-agnostic consciousness computation.
        // The consciousness equation blends whatever inputs are available:
        // - temporal_coherence: FRESH (just computed from CfC tau convergence)
        // - prediction_confidence: from prior cycles (delayed but valid)
        // - flow_intensity: from flow state tracker (updated if conditions met)
        // - pattern_confidence: from temporal classifier (available)
        // This makes consciousness measurement modality-agnostic: the same
        // equation works for text, raw sensors, proprioception, or any mix.
        // Science: IIT predicts consciousness = integrated information across
        // modules, regardless of the modality producing that information.
        let pattern_confidence = self
            .language_comm
            .voice_coherence
            .temporal
            .classify_state()
            .1;
        let consciousness_level =
            super::snapshot::ConsciousnessSnapshot::compute_consciousness_level(
                self.prediction_confidence as f32,
                coherence,
                self.behavior.flow_state.intensity,
                pattern_confidence,
            ) as f64;
        self.carryover.history.consciousness_level = consciousness_level;

        // 11. Buffer PsiAttestation record if enabled (mirrors cycle.rs step 10h.0)
        // For the HV path, unified_psi is derived from temporal coherence since
        // we don't run the full consciousness subsystems.
        let urgency = self.carryover.urgency.urgency;
        if self.config.enable_psi_attestation
            && self.config.agent_did.is_some()
            && (self.stats.total_cycles % 10 == 0)
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            let record = super::PsiAttestationRecord {
                psi: coherence.clamp(0.0, 1.0) as f64,
                cycle_id: self.stats.total_cycles as u64,
                captured_at_us: now,
                prediction_error,
                urgency,
            };
            self.push_psi_attestation(record);
        }

        super::CycleResult {
            output,
            prediction_error,
            peak_attention: 0.0,
            detected_primitives: Vec::new(),
            learning_occurred,
            training_loss,
            bits_saved_persist: None,
            bits_saved_zero: None,
            bits_kappa: None,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata: super::CycleMetadata {
                urgency,
                actual_effective_lr: if learning_occurred {
                    self.config.cfc_config.learning_rate
                } else {
                    0.0
                },
                consciousness: super::types::telemetry::ConsciousnessLevelMetrics {
                    consciousness_level,
                    ..Default::default()
                },
                ..super::CycleMetadata::default()
            },
            thought_vector: vec![0.0; 32],
            wisdom_hv: symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(hdv),
            language_output: None,
            language_source: None,
            #[cfg(feature = "canvas")]
            canvas_svg: None,
            #[cfg(feature = "identity")]
            signed_output: None,
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
        }
    }

    /// Push a PsiAttestationRecord into the bounded buffer, evicting oldest if at capacity.
    pub(super) fn push_psi_attestation(&mut self, record: super::PsiAttestationRecord) {
        while self.psi_attestation_buffer.len() >= self.config.attestation_buffer_capacity {
            let _ = self.psi_attestation_buffer.pop_front();
        }
        self.psi_attestation_buffer.push_back(record);
    }

    /// Update prediction confidence based on consciousness state and prediction accuracy
    ///
    /// Confidence decays during uncertain/transitioning states and grows when
    /// predictions are accurate in stable states.
    pub(super) fn update_prediction_confidence(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        pattern_confidence: f32,
    ) {
        use ConsciousnessPattern::*;

        // Base decay/growth parameters
        const DECAY_RATE_UNCERTAIN: f32 = 0.05; // Fast decay when uncertain
        const DECAY_RATE_TRANSITION: f32 = 0.03; // Moderate decay during transitions
        const GROWTH_RATE_ACCURATE: f32 = 0.02; // Slow growth for stability
        const ERROR_THRESHOLD: f32 = 0.3; // Below this = accurate prediction

        // Decay rate depends on consciousness state
        let decay_rate = match pattern {
            Uncertain => DECAY_RATE_UNCERTAIN,
            Transitioning => DECAY_RATE_TRANSITION,
            Resting => DECAY_RATE_TRANSITION * 0.5, // Slight decay in resting
            _ => 0.0,                               // No decay in stable states
        };

        // Growth when predictions are accurate in stable states
        let growth_rate = if prediction_error < ERROR_THRESHOLD {
            match pattern {
                Focused | Contemplative => GROWTH_RATE_ACCURATE * 1.5,
                Excited | Exploratory => GROWTH_RATE_ACCURATE,
                _ => GROWTH_RATE_ACCURATE * 0.5,
            }
        } else {
            0.0
        };

        // Apply decay and growth
        let confidence_delta = growth_rate - decay_rate;

        // Scale by pattern confidence (more confident = stronger effect)
        let scaled_delta = confidence_delta * pattern_confidence;

        // Update with bounds
        self.adjust_confidence("consciousness_pattern", scaled_delta);

        // Additional penalty for very high prediction errors — but only after warmup.
        // During the first STARTUP_WARMUP_CYCLES cycles PE=1.0 is a bootstrap artifact
        // (no prediction exists yet), not a genuine prediction failure.
        // Penalizing during warmup creates a death spiral: prediction_confidence→0 →
        // MCE knowledge input→0 → consciousness crushed → pattern stays Uncertain.
        if prediction_error > 0.7
            && self.stats.total_cycles > crate::cognitive_loop::thresholds::STARTUP_WARMUP_CYCLES
        {
            self.scale_confidence("high_pred_error", 0.95);
        }
    }

    pub(super) fn create_experience(&mut self, state: &[f32], prediction: &[f32], error: f32) {
        // Update last experience with next_state
        // Own the taken value (not `ref`) so we can move it into Experience without cloning
        if let Some(last_state) = self.last_state.take() {
            if let Some(last_pred) = self.last_prediction.take() {
                // Calculate importance based on error
                let importance = error + EXPERIENCE_BASE_IMPORTANCE;

                let exp = Experience {
                    state: last_state, // move, not clone
                    prediction: last_pred,
                    next_state: Some(state.to_vec()),
                    error,
                    importance,
                };

                if self.buffer.len() >= self.config.buffer_size {
                    self.buffer.pop_front();
                }
                self.buffer.push_back(exp);
            }
        }

        // Store current state for next cycle
        self.last_state = Some(state.to_vec());
        self.last_prediction = Some(prediction.to_vec());
    }

    /// Copy `last_state` into the reusable `training_state_buf` without
    /// allocating a new Vec.  On the first cycle the buffer is allocated once;
    /// subsequent cycles reuse it via `copy_from_slice`.  This must be called
    /// *before* `create_experience`, which moves `last_state` into the replay
    /// buffer.
    pub(super) fn copy_last_state_to_training_buf(&mut self) {
        match (&self.last_state, &mut self.training_state_buf) {
            (Some(src), Some(buf)) if buf.len() == src.len() => {
                buf.copy_from_slice(src);
            }
            (Some(src), _) => {
                // First cycle or dimension changed — allocate once
                self.training_state_buf = Some(src.clone());
            }
            (None, _) => {
                self.training_state_buf = None;
            }
        }
    }

    pub(super) fn update_stats(&mut self, error: f32, cycle_time: Duration) {
        // EMA for error
        let alpha = ERROR_EMA_ALPHA;
        self.stats.avg_prediction_error =
            self.stats.avg_prediction_error * (1.0 - alpha) + error * alpha;
        self.stats.avg_prediction_error_sq =
            self.stats.avg_prediction_error_sq * (1.0 - alpha) + (error * error) * alpha;

        // Error trend — capacity bound: 100 elements, evict before push
        if self.error_history.len() >= super::thresholds::ERROR_HISTORY_CAPACITY {
            self.error_history.pop_front();
        }
        self.error_history.push_back(error);
        self.stats.error_trend = self.compute_error_trend();

        // Attention stats from encoder
        let encoder_stats = self.encoder.stats();
        self.stats.attention_variance = encoder_stats.attention_variance;
        self.stats.diverged_primitives = encoder_stats.diverged_primitives;

        // Buffer utilization
        self.stats.buffer_utilization = self.buffer.len() as f32 / self.config.buffer_size as f32;

        // Timing stats
        let cycle_us = cycle_time.as_micros() as f32;
        self.stats.avg_cycle_time_us =
            self.stats.avg_cycle_time_us * (1.0 - TIMING_EMA_ALPHA) + cycle_us * TIMING_EMA_ALPHA;

        // Cycles per second
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.stats.cycles_per_second = self.stats.total_cycles as f32 / elapsed;
        }

        // CfC state diversity (already updated in cycle(), but ensure consistency)
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Voice feedback stats
        let voice_summary = self.language_comm.voice_coherence.voice.summary();
        self.stats.voice_articulation_quality = voice_summary.articulation_quality;
        self.stats.voice_rate_stability = voice_summary.rate_stability;
        self.stats.voice_phi_adjustment = voice_summary.phi_adjustment;

        // Combined phi = coherence contribution + voice adjustment
        self.stats.combined_phi_contribution =
            self.stats.coherence_phi_contribution + self.stats.voice_phi_adjustment;

        // Consciousness pattern from temporal signatures
        let temporal_summary = self.language_comm.voice_coherence.temporal.summary();
        let pat_str = temporal_summary.pattern.as_str();
        if self.stats.consciousness_pattern != pat_str {
            self.stats.consciousness_pattern = pat_str.to_string();
        }
        self.stats.pattern_confidence = temporal_summary.confidence;
        self.stats.tau_mean = temporal_summary.features.mean;
        self.stats.tau_trend = temporal_summary.features.trend;

        // Adaptive behavior stats
        self.stats.adaptive_confidence = self.behavior.adaptive_behavior.confidence;
        let hint_str = self.behavior.adaptive_behavior.action_hint.as_str();
        if self.stats.action_hint != hint_str {
            self.stats.action_hint = hint_str.to_string();
        }
        self.stats.learning_paused = self.behavior.adaptive_behavior.pause_learning;
        self.stats.adaptive_learning_rate = self
            .behavior
            .adaptive_behavior
            .effective_learning_rate(self.combined_learning_rate());
        self.stats.adaptive_speech_rate = self.behavior.adaptive_behavior.speech_rate_multiplier;

        // Prediction confidence stats
        self.stats.prediction_confidence = self.prediction_confidence as f32;
        // Decay rate: higher when in uncertain states
        self.stats.confidence_decay_rate = match self.behavior.adaptive_behavior.action_hint {
            ActionHint::Stabilize | ActionHint::SeekInput => 0.05,
            ActionHint::SlowDown => 0.03,
            _ => 0.0,
        };

        // Flow state stats
        self.stats.in_flow = self.behavior.flow_state.in_flow;
        self.stats.flow_intensity = self.behavior.flow_state.intensity;
        self.stats.flow_streak = self.behavior.flow_state.streak;
        self.stats.flow_learning_boost = self.behavior.flow_state.learning_boost;

        // Emotion contagion stats — read from unified emotional state (canonical source)
        // EmotionContagion is now a stateless preprocessor; UnifiedEmotionalState
        // is the single source of truth for affect (Phase 2 consolidation).
        let unified_emo = self.unification_engine.emotional.state();
        self.stats.emotional_valence = unified_emo.valence as f32;
        self.stats.emotional_arousal = unified_emo.arousal as f32;
        let (nudge_pattern, nudge_strength) = self.behavior.emotion_contagion.pattern_nudge();
        let nudge_str = nudge_pattern.map(|p| p.as_str()).unwrap_or("None");
        if self.stats.emotion_nudge_pattern != nudge_str {
            self.stats.emotion_nudge_pattern = nudge_str.to_string();
        }
        self.stats.emotion_nudge_strength = nudge_strength;

        // Curiosity drive stats
        self.stats.boredom = self.behavior.curiosity_drive.boredom;
        self.stats.curiosity = self.behavior.curiosity_drive.curiosity;
        self.stats.exploration_urge = self.behavior.curiosity_drive.exploration_urge as f32;
        self.stats.curiosity_exploring = self.behavior.curiosity_drive.should_explore();
        self.stats.novelty_bonus = self.behavior.curiosity_drive.novelty_bonus;

        // Self-reflection stats
        let assess_str = self
            .consciousness
            .self_model_tier
            .self_reflection
            .self_assessment
            .as_str();
        if self.stats.self_assessment != assess_str {
            self.stats.self_assessment = assess_str.to_string();
        }
        self.stats.reflection_count = self
            .consciousness
            .self_model_tier
            .self_reflection
            .reflection_count;
        self.stats.adjustments_made = self
            .consciousness
            .self_model_tier
            .self_reflection
            .adjustments_made;
        self.stats.learning_effectiveness = self
            .consciousness
            .self_model_tier
            .self_reflection
            .learning_effectiveness();
        let summary = self.consciousness.self_model_tier.self_reflection.summary();
        self.stats.next_reflection_in = summary.next_reflection_in;
        self.stats.adapted_flow_threshold = self
            .consciousness
            .self_model_tier
            .self_reflection
            .flow_error_threshold;
        self.stats.adapted_boredom_threshold = self
            .consciousness
            .self_model_tier
            .self_reflection
            .boredom_threshold;

        // ═══════════════════════════════════════════════════════════════════════
        // MEGA-UNIFIED ARCHITECTURE STATS
        // ═══════════════════════════════════════════════════════════════════════

        // Cognitive depth from thalamic routing
        let depth_str = self.cognitive_depth.as_str();
        if self.stats.cognitive_depth != depth_str {
            self.stats.cognitive_depth = depth_str.to_string();
        }

        // Unified Phi from the unification engine
        self.stats.unified_psi = self.unification_engine.psi as f32;

        // Unified emotional state (VAD)
        let unified_state = self.unification_engine.emotional.state();
        self.stats.unified_emotional_valence = unified_state.valence as f32;
        self.stats.unified_emotional_arousal = unified_state.arousal as f32;
        self.stats.unified_emotional_dominance = unified_state.dominance as f32;
        let emo_str = unified_state
            .discrete_emotion
            .map(|e| e.as_str())
            .unwrap_or("Neutral");
        if self.stats.unified_emotion != emo_str {
            self.stats.unified_emotion = emo_str.to_string();
        }

        // Emotional pattern from the bridge
        let emo_pat_str = self.unification_engine.emotional.detect_pattern().as_str();
        if self.stats.emotional_pattern != emo_pat_str {
            self.stats.emotional_pattern = emo_pat_str.to_string();
        }

        // Thalamic routing statistics
        let (reflex_rate, cortical_rate, deep_rate) = self.behavior.thalamic_router.routing_stats();
        self.stats.thalamic_reflex_rate = reflex_rate;
        self.stats.thalamic_cortical_rate = cortical_rate;
        self.stats.thalamic_deep_rate = deep_rate;

        // Active Inference Bridge statistics
        let ai_stats = self.fep.active_inference_bridge.statistics();
        self.stats.active_inference_modulation_index =
            ai_stats.modulation_index.map(|mi| mi as f32).unwrap_or(0.0);
        let coupling_str = ai_stats.coupling_quality.as_str();
        if self.stats.active_inference_coupling_quality != coupling_str {
            self.stats.active_inference_coupling_quality = coupling_str.to_string();
        }
        self.stats.active_inference_avg_error = ai_stats
            .average_prediction_error
            .map(|e| e as f32)
            .unwrap_or(0.5);

        // Enhanced FEP Bridge statistics
        self.stats.fep_learning_signal = self.fep.learning_signal;
        // attention_shift is updated during cycle processing
        // fep_action_outcome_coupling is updated during cycle processing by enhanced FEP bridge

        // Closed Learning Loop statistics
        let cur_strat_str = self.fep.closed_learning_loop.current_strategy.as_str();
        if self.stats.current_strategy != cur_strat_str {
            self.stats.current_strategy = cur_strat_str.to_string();
        }
        let best_strat_str = self.fep.closed_learning_loop.best_strategy().as_str();
        if self.stats.best_strategy != best_strat_str {
            self.stats.best_strategy = best_strat_str.to_string();
        }
        self.stats.average_reward = self.fep.closed_learning_loop.average_reward();
        self.stats.exploration_rate = self.fep.closed_learning_loop.exploration_rate();
        self.stats.learning_loop_interactions = self.fep.closed_learning_loop.total_interactions();

        // Memory system statistics
        let (short_term, long_term) = self.fep.episodic_memory.memory_count();
        self.stats.memory_short_term_count = short_term;
        self.stats.memory_long_term_count = long_term;
        self.stats.memory_total_encoded = self.fep.episodic_memory.stats.total_encoded;
        self.stats.world_model_avg_error = self.fep.world_model.avg_error;
        self.stats.active_goals_count = self.fep.goal_system.active_goals().len();
    }

    pub(super) fn update_loss_stats(&mut self, loss: f32) {
        let alpha = ERROR_EMA_ALPHA;
        self.stats.avg_training_loss = self.stats.avg_training_loss * (1.0 - alpha) + loss * alpha;
    }

    pub(super) fn compute_error_trend(&self) -> f32 {
        if self.error_history.len() < 10 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.error_history.len() as f32;
        let errors: Vec<f32> = self.error_history.iter().cloned().collect();

        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f32 = errors.iter().sum::<f32>() / n;

        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;

        for (i, &y) in errors.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() > 0.0001 {
            let slope = numerator / denominator;
            if slope.is_finite() { slope } else { 0.0 }
        } else {
            0.0
        }
    }

    /// Prefrontal working memory capability (0.0-1.0). Returns 0.5 (neutral)
    /// when the prefrontal cortex is not enabled.
    ///
    /// Raw `avg_memory_utilization` (len/capacity) is a poor consciousness
    /// proxy: an empty WM has low utilization but HIGH available capacity.
    /// Blend utilization (active use) with graduation quality (items that
    /// integrated well enough to persist to episodic memory) for a signal
    /// that reflects how well WM *serves* consciousness.
    /// Science: Baddeley (2003) — WM capacity correlates with consciousness
    /// through both active maintenance and successful episodic transfer.
    pub(super) fn prefrontal_utilization(&self) -> f64 {
        self.prefrontal
            .as_ref()
            .map(|p| {
                let stats = p.stats();
                let utilization = stats.avg_memory_utilization as f64;
                // Graduation rate: fraction of processed items that graduated
                // to episodic memory (vs being dropped). High graduation = WM
                // is doing its job of filtering and consolidating.
                let graduation_quality = if stats.items_processed > 0 {
                    let total_exits = stats.graduations + stats.evictions_dropped;
                    if total_exits > 0 {
                        stats.graduations as f64 / total_exits as f64
                    } else {
                        0.5 // No exits yet — neutral
                    }
                } else {
                    0.5 // No items yet — neutral
                };
                // Blend: 30% utilization (active use) + 30% graduation quality
                // + 40% floor (WM capacity exists and is available even when
                // idle — an unused but available workspace is not a deficit).
                (utilization * PREFRONTAL_UTILIZATION_WEIGHT
                    + graduation_quality * PREFRONTAL_GRADUATION_WEIGHT
                    + PREFRONTAL_FLOOR)
                    .clamp(0.0, 1.0)
            })
            .unwrap_or(0.5)
    }

    /// Reset all learning state
    pub fn reset(&mut self) {
        self.encoder.reset_attention();
        // Reset CfC state by injecting zeros
        let zeros = Array1::from_vec(vec![0.0f32; self.config.cfc_config.input_dim]);
        let _ = self.temporal_network.inject(&zeros);
        self.buffer.clear();
        self.error_history.clear();
        self.last_state = None;
        self.training_state_buf = None;
        self.last_prediction = None;
        self.stats = LoopStats::default();
        self.start_time = Instant::now();
        self.language_comm.reset();
        self.behavior.adaptive_behavior = AdaptiveBehavior::default();
        self.prediction_confidence = 0.5;
        self.feedback_state = super::feedback_state::FeedbackState::new();
        self.feedback_state.begin_cycle();
        self.feedback_state.snapshot_cycle_start(0.5, 1.0, 0.0, 1.0);
        self.behavior.flow_state.reset();
        self.behavior.emotion_contagion.reset();
        self.behavior.curiosity_drive.reset();
        // Reset unified emotional state so emotional_valence() returns 0.0
        self.unification_engine.emotional =
            crate::consciousness::dynamics::consciousness_unification::EmotionalBridge::new();
        self.consciousness.self_model_tier.self_reflection.reset(); // Preserves learned thresholds
        self.fep.agent = ActiveInferenceAgent::new(self.fep.agent.config.clone());
        self.coherence_tracker.reset();
        self.behavior.social_mgr = super::SocialManager::default();
        // user_state reset handled by self.language_comm.reset() above
        self.policy_agreement_window.clear();
        self.carryover = CycleCarryover::default();
        self.consciousness_state.reset();
        if let Some(ref mut thermo) = self.consciousness.consciousness_monitors.thermodynamics {
            *thermo = crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer::new(
                crate::consciousness::consciousness_thermodynamics::ThermodynamicsConfig::default(),
            );
        }
        if let Some(ref mut binding) = self.consciousness.consciousness_monitors.phenomenal_binding
        {
            *binding =
                crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer::new(
                    crate::consciousness::phenomenal_binding::PhenomenalBindingConfig::default(),
                );
        }
        if let Some(ref mut hfe) = self
            .consciousness
            .consciousness_monitors
            .hierarchical_free_energy
        {
            *hfe = crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy::new(
                crate::consciousness::hierarchical_free_energy::HierarchicalFEConfig::default(),
            );
        }
        self.ethics_values.reset();
        self.primitive_tier.reset();
        self.memory.memory_consol.reset();
        self.feature_integ.reset();
        self.sensorimotor.vision_sensory.reset();
        // Note: predictive_phi_modulation and cross_modal_psi already reset
        // via self.carryover = CycleCarryover::default() above.
        self.subsystem_collector.clear();
        self.last_snapshot = None;
        self.drive_manager = super::managers::DriveManager::default();
        self.memory_manager = super::managers::MemoryManager::default();
        self.learning_manager = super::managers::LearningManager::default();
        self.perception_manager = super::managers::PerceptionManager::default();
        self.swarm_manager = super::managers::SwarmManager::default();
        #[cfg(feature = "mesh")]
        {
            let domain = self.spectrum_manager.domain_profile().clone();
            self.spectrum_manager = super::managers::SpectrumManager::with_domain_profile(domain);
        }
        #[cfg(feature = "therapeutic")]
        {
            self.therapeutic_manager = super::managers::TherapeuticManager::default();
        }
        #[cfg(feature = "support")]
        {
            self.support.reset();
        }
    }
}
