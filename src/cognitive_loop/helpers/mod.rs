//! Internal helper methods for CognitiveLoopService.
//!
//! Contains experience creation, statistics updates, error trend computation,
//! prediction confidence tracking, reset, and neural bridge processing.
//!
//! Sub-modules hold extracted cycle() helpers:
//! - `cycle_extracted`: Phases 1–4 (methods on `&mut self`)
//! - `parallel`: Phase 5 free functions for `rayon::join` branches

mod cycle_extracted;
mod parallel;

// Re-export Phase 5 items so cycle.rs can use `helpers::run_stability_regime` etc.
pub(super) use parallel::{
    parallel_episodic_learning, parallel_semantic_causal, run_stability_regime,
    EpisodicLearningContext,
};

// Re-export pub(super) constants used by cycle.rs
pub(super) use cycle_extracted::MEMORY_RECALL_TOP_K;

use super::CycleCarryover;
use crate::consciousness::fep_active_inference::ActiveInferenceAgent;
use crate::dynamics::temporal_signatures::ConsciousnessPattern;
#[cfg(feature = "neural-bridge")]
use anyhow::Result;
use ndarray::Array1;
use std::time::{Duration, Instant};

#[cfg(feature = "neural-bridge")]
use super::CycleResult;
use super::{ActionHint, AdaptiveBehavior, CognitiveLoopService, Experience, LoopStats};

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

        let bridge = self
            .neural_bridge
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Neural bridge not loaded (no probe weights found)"))?;

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
        let input_array = Array1::from_vec(compressed_state.clone());
        let delta_t = self.config.cfc_config.delta_t;
        let _ = self.temporal_network.step(&input_array, delta_t);

        // 5. Multi-scale prediction
        let prediction = self.get_multi_scale_prediction(&input_array);

        // 6. Read CfC output state
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);

        // 7. Feed prediction back to encoder for next cycle
        self.encoder.set_prediction(prediction.clone());

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

        // 10. Learning step: consolidate periodically
        let mut learning_occurred = false;
        let mut training_loss = None;
        if self.config.enable_consolidation && self.stats.total_cycles % 50 == 0 {
            if let Ok(loss) = self.consolidate() {
                if loss > 0.0 {
                    learning_occurred = true;
                    training_loss = Some(loss);
                }
            }
        }

        // 11. Update error history — capacity bound: 100 elements, evict before push
        if self.error_history.len() >= 100 {
            self.error_history.pop_front();
        }
        self.error_history.push_back(prediction_error);
        self.stats.avg_prediction_error =
            self.error_history.iter().sum::<f32>() / self.error_history.len().max(1) as f32;

        Ok(CycleResult {
            output: output.clone(),
            prediction_error,
            peak_attention: 0.0, // No text-based attention for embedding input
            detected_primitives: Vec::new(), // No text primitives for embedding input
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata: super::CycleMetadata::default(),
            #[cfg(feature = "identity")]
            signed_output: self.mfdi_bridge.sign_output(output.clone()).ok(),
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
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
        let input_array = Array1::from_vec(compressed_state.clone());
        let delta_t = self.config.cfc_config.delta_t;
        let _ = self.temporal_network.step(&input_array, delta_t);

        // 3. Multi-scale prediction
        let prediction = self.get_multi_scale_prediction(&input_array);

        // 4. Read CfC output state
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);

        // 5. Feed prediction back to encoder for next cycle
        self.encoder.set_prediction(prediction.clone());

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

        // 8. Update coherence bridge with current tau values
        let tau_owned: Vec<ndarray::Array1<f32>> = self.temporal_network.all_tau_owned();
        let tau_refs: Vec<&ndarray::Array1<f32>> = tau_owned.iter().collect();
        self.coherence_bridge.update(&tau_refs);
        let coherence = self.coherence_bridge.smoothed_coherence();

        // Effective threshold matches cycle() behavior (adaptive scaling)
        let effective_threshold =
            self.config.learning_threshold * self.carryover.learning.adaptive_threshold_scale;

        // 9. Learn if error is significant
        let (learning_occurred, training_loss) = if prediction_error > effective_threshold {
            self.stats.learning_cycles += 1;
            if let Some(ref prev_state) = self.last_state.clone() {
                let train_input = Array1::from_vec(prev_state.clone());
                let train_target = Array1::from_vec(compressed_state.clone());
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
                    Err(_) => (false, None),
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

        // 11. Buffer PsiAttestation record if enabled (mirrors cycle.rs step 10h.0)
        // For the HV path, unified_psi is derived from temporal coherence since
        // we don't run the full consciousness subsystems.
        let urgency = self.carryover.urgency.urgency;
        if self.config.enable_psi_attestation && self.config.agent_did.is_some() {
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
            // Capacity bound: attestation_buffer_capacity (max 256) — evict before push
            while self.psi_attestation_buffer.len() >= self.config.attestation_buffer_capacity {
                let _ = self.psi_attestation_buffer.pop_front();
            }
            self.psi_attestation_buffer.push_back(record);
        }

        super::CycleResult {
            output,
            prediction_error,
            peak_attention: 0.0,
            detected_primitives: Vec::new(),
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata: super::CycleMetadata {
                urgency,
                actual_effective_lr: if learning_occurred {
                    self.config.cfc_config.learning_rate
                } else {
                    0.0
                },
                ..super::CycleMetadata::default()
            },
            #[cfg(feature = "identity")]
            signed_output: None,
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
        }
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
        self.prediction_confidence = (self.prediction_confidence + scaled_delta).clamp(0.0, 1.0);

        // Additional penalty for very high prediction errors
        if prediction_error > 0.7 {
            self.prediction_confidence *= 0.95; // 5% penalty for bad predictions
        }
    }

    pub(super) fn create_experience(&mut self, state: &[f32], prediction: &[f32], error: f32) {
        // Update last experience with next_state
        if let Some(ref last_state) = self.last_state.take() {
            if let Some(last_pred) = self.last_prediction.take() {
                // Calculate importance based on error
                let importance = error + 0.1; // Base importance

                let exp = Experience {
                    state: last_state.clone(),
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

    pub(super) fn update_stats(&mut self, error: f32, cycle_time: Duration) {
        // EMA for error
        let alpha = 0.1;
        self.stats.avg_prediction_error =
            self.stats.avg_prediction_error * (1.0 - alpha) + error * alpha;
        self.stats.avg_prediction_error_sq =
            self.stats.avg_prediction_error_sq * (1.0 - alpha) + (error * error) * alpha;

        // Error trend — capacity bound: 100 elements, evict before push
        if self.error_history.len() >= 100 {
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
        self.stats.avg_cycle_time_us = self.stats.avg_cycle_time_us * 0.99 + cycle_us * 0.01;

        // Cycles per second
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.stats.cycles_per_second = self.stats.total_cycles as f32 / elapsed;
        }

        // CfC state diversity (already updated in cycle(), but ensure consistency)
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Voice feedback stats
        let voice_summary = self.voice_feedback_bridge.summary();
        self.stats.voice_articulation_quality = voice_summary.articulation_quality;
        self.stats.voice_rate_stability = voice_summary.rate_stability;
        self.stats.voice_phi_adjustment = voice_summary.phi_adjustment;

        // Combined phi = coherence contribution + voice adjustment
        self.stats.combined_phi_contribution =
            self.stats.coherence_phi_contribution + self.stats.voice_phi_adjustment;

        // Consciousness pattern from temporal signatures
        let temporal_summary = self.temporal_signature_encoder.summary();
        self.stats.consciousness_pattern = format!("{:?}", temporal_summary.pattern);
        self.stats.pattern_confidence = temporal_summary.confidence;
        self.stats.tau_mean = temporal_summary.features.mean;
        self.stats.tau_trend = temporal_summary.features.trend;

        // Adaptive behavior stats
        self.stats.adaptive_confidence = self.adaptive_behavior.confidence;
        self.stats.action_hint = format!("{:?}", self.adaptive_behavior.action_hint);
        self.stats.learning_paused = self.adaptive_behavior.pause_learning;
        self.stats.adaptive_learning_rate = self
            .adaptive_behavior
            .effective_learning_rate(self.combined_learning_rate());
        self.stats.adaptive_speech_rate = self.adaptive_behavior.speech_rate_multiplier;

        // Prediction confidence stats
        self.stats.prediction_confidence = self.prediction_confidence;
        // Decay rate: higher when in uncertain states
        self.stats.confidence_decay_rate = match self.adaptive_behavior.action_hint {
            ActionHint::Stabilize | ActionHint::SeekInput => 0.05,
            ActionHint::SlowDown => 0.03,
            _ => 0.0,
        };

        // Flow state stats
        self.stats.in_flow = self.flow_state.in_flow;
        self.stats.flow_intensity = self.flow_state.intensity;
        self.stats.flow_streak = self.flow_state.streak;
        self.stats.flow_learning_boost = self.flow_state.learning_boost;

        // Emotion contagion stats
        self.stats.emotional_valence = self.emotion_contagion.smoothed_valence();
        self.stats.emotional_arousal = self.emotion_contagion.smoothed_arousal();
        let (nudge_pattern, nudge_strength) = self.emotion_contagion.pattern_nudge();
        self.stats.emotion_nudge_pattern = nudge_pattern
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "None".to_string());
        self.stats.emotion_nudge_strength = nudge_strength;

        // Curiosity drive stats
        self.stats.boredom = self.curiosity_drive.boredom;
        self.stats.curiosity = self.curiosity_drive.curiosity;
        self.stats.exploration_urge = self.curiosity_drive.exploration_urge;
        self.stats.curiosity_exploring = self.curiosity_drive.should_explore();
        self.stats.novelty_bonus = self.curiosity_drive.novelty_bonus;

        // Self-reflection stats
        self.stats.self_assessment = format!("{:?}", self.self_reflection.self_assessment);
        self.stats.reflection_count = self.self_reflection.reflection_count;
        self.stats.adjustments_made = self.self_reflection.adjustments_made;
        self.stats.learning_effectiveness = self.self_reflection.learning_effectiveness();
        let summary = self.self_reflection.summary();
        self.stats.next_reflection_in = summary.next_reflection_in;
        self.stats.adapted_flow_threshold = self.self_reflection.flow_error_threshold;
        self.stats.adapted_boredom_threshold = self.self_reflection.boredom_threshold;

        // ═══════════════════════════════════════════════════════════════════════
        // MEGA-UNIFIED ARCHITECTURE STATS
        // ═══════════════════════════════════════════════════════════════════════

        // Cognitive depth from thalamic routing
        self.stats.cognitive_depth = format!("{:?}", self.cognitive_depth);

        // Unified Phi from the unification engine
        self.stats.unified_psi = self.unification_engine.psi as f32;

        // Unified emotional state (VAD)
        let unified_state = self.unification_engine.emotional.state();
        self.stats.unified_emotional_valence = unified_state.valence as f32;
        self.stats.unified_emotional_arousal = unified_state.arousal as f32;
        self.stats.unified_emotional_dominance = unified_state.dominance as f32;
        self.stats.unified_emotion = unified_state
            .discrete_emotion
            .map(|e| format!("{e:?}"))
            .unwrap_or_else(|| "Neutral".to_string());

        // Emotional pattern from the bridge
        self.stats.emotional_pattern =
            format!("{:?}", self.unification_engine.emotional.detect_pattern());

        // Thalamic routing statistics
        let (reflex_rate, cortical_rate, deep_rate) = self.thalamic_router.routing_stats();
        self.stats.thalamic_reflex_rate = reflex_rate;
        self.stats.thalamic_cortical_rate = cortical_rate;
        self.stats.thalamic_deep_rate = deep_rate;

        // Active Inference Bridge statistics
        let ai_stats = self.active_inference_bridge.statistics();
        self.stats.active_inference_modulation_index =
            ai_stats.modulation_index.map(|mi| mi as f32).unwrap_or(0.0);
        self.stats.active_inference_coupling_quality = format!("{:?}", ai_stats.coupling_quality);
        self.stats.active_inference_avg_error = ai_stats
            .average_prediction_error
            .map(|e| e as f32)
            .unwrap_or(0.5);

        // Enhanced FEP Bridge statistics
        self.stats.fep_learning_signal = self.fep_learning_signal;
        // attention_shift is updated during cycle processing
        // fep_action_outcome_coupling is updated during cycle processing by enhanced FEP bridge

        // Closed Learning Loop statistics
        self.stats.current_strategy = format!("{:?}", self.closed_learning_loop.current_strategy);
        self.stats.best_strategy = format!("{:?}", self.closed_learning_loop.best_strategy());
        self.stats.average_reward = self.closed_learning_loop.average_reward();
        self.stats.exploration_rate = self.closed_learning_loop.exploration_rate();
        self.stats.learning_loop_interactions = self.closed_learning_loop.total_interactions();

        // Memory system statistics
        let (short_term, long_term) = self.episodic_memory.memory_count();
        self.stats.memory_short_term_count = short_term;
        self.stats.memory_long_term_count = long_term;
        self.stats.memory_total_encoded = self.episodic_memory.stats.total_encoded;
        self.stats.world_model_avg_error = self.world_model.avg_error;
        self.stats.active_goals_count = self.goal_system.active_goals().len();
    }

    pub(super) fn update_loss_stats(&mut self, loss: f32) {
        let alpha = 0.1;
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
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Prefrontal working memory utilization (0.0-1.0). Returns 0.5 (neutral)
    /// when the prefrontal cortex is not enabled.
    pub(super) fn prefrontal_utilization(&self) -> f64 {
        self.prefrontal
            .as_ref()
            .map(|p| p.stats().avg_memory_utilization as f64)
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
        self.last_prediction = None;
        self.stats = LoopStats::default();
        self.start_time = Instant::now();
        self.coherence_bridge.reset();
        self.voice_feedback_bridge.reset();
        self.temporal_signature_encoder.reset();
        self.adaptive_behavior = AdaptiveBehavior::default();
        self.prediction_confidence = 0.5; // Reset to neutral confidence
        self.flow_state.reset();
        self.emotion_contagion.reset();
        self.curiosity_drive.reset();
        self.self_reflection.reset(); // Preserves learned thresholds
        self.fep_agent = ActiveInferenceAgent::new(self.fep_agent.config.clone());
        self.coherence_tracker.reset();
        self.external_reward = 0.0;
        self.social_trust = 0.5;
        self.social_cooperation_rate = 0.0;
        if let Some(ref mut usi) = self.user_state {
            usi.reset();
        }
        self.policy_agreement_window.clear();
        self.carryover = CycleCarryover::default();
        if let Some(ref mut mind) = self.predictive_mind {
            *mind = crate::consciousness::predictive_processing::PredictiveMind::new(
                crate::consciousness::predictive_processing::PredictiveConfig::default(),
            );
        }
        if let Some(ref mut binder) = self.cross_modal_binder {
            binder.clear();
        }
        if let Some(ref mut bridge) = self.affective_bridge {
            *bridge = crate::brain::affective_bridge::AffectiveBridge::default();
        }
        if let Some(ref mut thermo) = self.consciousness_thermodynamics {
            *thermo = crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer::new(
                crate::consciousness::consciousness_thermodynamics::ThermodynamicsConfig::default(),
            );
        }
        if let Some(ref mut binding) = self.phenomenal_binding {
            *binding =
                crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer::new(
                    crate::consciousness::phenomenal_binding::BindingConfig::default(),
                );
        }
        if let Some(ref mut hfe) = self.hierarchical_free_energy {
            *hfe = crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy::new(
                crate::consciousness::hierarchical_free_energy::HierarchicalFEConfig::default(),
            );
        }
        if let Some(ref mut cw) = self.contextual_weights {
            *cw = crate::consciousness::contextual_weights::ContextualWeights::new();
        }
        if let Some(ref mut pa) = self.phi_attention {
            *pa = crate::consciousness::phi_attention::AdaptiveThresholds::new(100);
        }
        if let Some(ref mut nd) = self.negation_detector {
            *nd = crate::consciousness::negation_detector::NegationDetector::new();
        }
        if let Some(ref mut pp) = self.primitive_processor {
            *pp =
                crate::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor::new(
                );
        }
        // Note: predictive_phi_modulation and cross_modal_psi already reset
        // via self.carryover = CycleCarryover::default() above.
    }
}
