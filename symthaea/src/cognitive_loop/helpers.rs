//! Internal helper methods for CognitiveLoopService.
//!
//! Contains experience creation, statistics updates, error trend computation,
//! prediction confidence tracking, reset, and neural bridge processing.

use super::CycleCarryover;
use crate::consciousness::cross_modal_binding::{ModalRepresentation, Modality};
use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, Observation};
use crate::dynamics::temporal_signatures::ConsciousnessPattern;
#[cfg(feature = "neural-bridge")]
use anyhow::Result;
use ndarray::Array1;
use std::time::{Duration, Instant};

use super::{
    ActionHint, AdaptiveBehavior, CognitiveLoopService, CycleResult, Experience, LoopStats,
    MoralJudgmentSummary, ResponseStrategy,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Tuning constants for extracted helpers (moved from cycle.rs)
// ═══════════════════════════════════════════════════════════════════════════════

// -- Moral evaluation --
const MORAL_EVAL_INTERVAL: usize = 7;
const MORAL_CONCERN_THRESHOLD: f32 = -0.3;
const NEGATION_POLARITY_THRESHOLD: f32 = 0.5;
const NEGATION_DAMPENING: f32 = 0.3;

// -- Memory recall --
pub(super) const MEMORY_RECALL_TOP_K: usize = 3;
pub(super) const MEMORY_RECALL_SIM_THRESHOLD: f32 = 0.3;
const MEMORY_CONTEXT_BOOST_SCALE: f32 = 0.1;

// -- Surprise & exploration --
const SURPRISE_BOREDOM_DAMPEN: f32 = 0.7;

// -- Psi synthesis --
const FLOW_PSI_WEIGHT: f32 = 0.2;
const RELATIONAL_PSI_WEIGHT: f32 = 0.15;
const BODY_PSI_WEIGHT: f64 = 0.1;
const EMBODIED_PSI_WEIGHT: f64 = 0.05;

// -- Strategy modulation --
const STRATEGY_EXPLORATORY_FACTOR: f32 = 0.8;
const STRATEGY_DETAILED_SENSITIVITY: f32 = 1.2;
const STRATEGY_CONCISE_SPEECH_RATE: f32 = 1.2;
const STRATEGY_CLARIFYING_FACTOR: f32 = 0.5;
const STRATEGY_SUPPORTIVE_PAUSE: f32 = 1.3;

// -- Reward computation (RL) --
const REWARD_GOOD_BASE: f32 = 0.5;
const REWARD_GOOD_CONFIDENCE_SCALE: f32 = 0.5;
const REWARD_BAD_BASE: f32 = -0.3;
const REWARD_BAD_SCALE: f32 = -0.2;
const REWARD_MID_BASE: f32 = 0.2;
const REWARD_MID_SCALE: f32 = -0.5;
const REWARD_EXTERNAL_BLEND: f32 = 0.5;

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
    /// Pipeline: ContinuousHV → compress → CfC temporal processing → predict → learn → CycleResult.
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

        // 1. Compress HDC → CfC input dimension via random projection
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
        let (learning_occurred, training_loss) =
            if prediction_error > effective_threshold {
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

    /// Prefrontal working memory utilization (0.0–1.0). Returns 0.5 (neutral)
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
            *pp = crate::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor::new();
        }
        // Note: predictive_phi_modulation and cross_modal_psi already reset
        // via self.carryover = CycleCarryover::default() above.
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Extracted helpers from cycle() — Phase 1 (LOW risk, &self / &mut self)
    // ═══════════════════════════════════════════════════════════════════════

    /// Safety pre-check: fast amygdala veto before expensive encoding.
    ///
    /// Returns `Some(CycleResult)` with a safe default response if the safety
    /// gateway blocks the input, or `None` if the input is allowed.
    pub(super) fn safety_precheck(&mut self, input: &str, cycle_start: Instant) -> Option<CycleResult> {
        let gateway = self.safety_gateway.as_mut()?;
        let decision = gateway.check(crate::safety::SafetyCheck::Query(input));
        if decision.allowed {
            return None;
        }
        let mut metadata = super::CycleMetadata::default();
        metadata.safety_blocked = true;
        metadata.safety_category = decision.category.map(|c| format!("{c:?}"));
        metadata.urgency = self.carryover.urgency.urgency;
        tracing::warn!(
            target: "cognitive_loop::safety",
            category = ?decision.category,
            message = ?decision.message,
            "Safety gateway blocked input — returning safe default"
        );
        Some(CycleResult {
            output: vec![0.0; self.config.cfc_config.num_neurons],
            prediction_error: 0.0,
            peak_attention: 0.0,
            detected_primitives: vec![],
            learning_occurred: false,
            training_loss: None,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            #[cfg(feature = "identity")]
            signed_output: None,
            #[cfg(feature = "identity")]
            assurance_level: crate::identity::AssuranceLevel::E0Anonymous,
        })
    }

    /// Thalamic routing: determine cognitive depth from prior state.
    ///
    /// Uses temporal signature pattern, prosody valence, and average prediction
    /// error to select Reflex / Cortical / DeepThought processing depth.
    pub(super) fn update_cognitive_depth(&mut self) {
        let prior_pattern = self.temporal_signature_encoder.classify_state().0;
        let prior_valence = self.emotion_contagion.prosody_valence();
        let prior_error = self.stats.avg_prediction_error;
        self.cognitive_depth =
            self.thalamic_router
                .route_from_cycle(prior_error, prior_pattern, prior_valence);
    }

    /// Detect negation polarity across safety-critical terms.
    ///
    /// Returns the max polarity score across "harmful", "dangerous", "unethical".
    /// Returns 0.0 if no negation detector is configured.
    /// Science: Wason (1959) — negation processing in human reasoning.
    pub(super) fn detect_negation_polarity(&self, input: &str) -> f32 {
        if let Some(ref detector) = self.negation_detector {
            detector.get_polarity(input, "harmful")
                .max(detector.get_polarity(input, "dangerous"))
                .max(detector.get_polarity(input, "unethical"))
        } else {
            0.0
        }
    }

    /// Compose the effective learning rate from all modulation sources.
    ///
    /// Combines: base coherence LR → adaptive behavior → flow state → semantic
    /// context → reasoning reliability → curiosity novelty → FEP boost →
    /// MCE consciousness boost → subsystem LR factor (from previous cycle).
    ///
    /// Resets `carryover.learning.subsystem_lr_factor` for the next cycle's
    /// accumulation. Hard-capped to [0.0, 0.01].
    pub(super) fn compose_effective_lr(
        &mut self,
        semantic_lr_factor: f32,
        reasoning_lr_factor: f32,
    ) -> f32 {
        let base_lr = self.combined_learning_rate();
        let adaptive_lr = self.adaptive_behavior.effective_learning_rate(base_lr);
        let flow_lr = self.flow_state.effective_learning_multiplier(adaptive_lr);
        let semantic_modulated_lr = flow_lr * semantic_lr_factor * reasoning_lr_factor;
        // Apply subsystem LR factor from PREVIOUS cycle (meta-cognition, predictive processing,
        // predictive self, phenomenal binding, consciousness thermodynamics). Reset for next cycle.
        let subsystem_lr = self.carryover.learning.subsystem_lr_factor.clamp(0.5, 2.0);
        self.carryover.learning.subsystem_lr_factor = 1.0;
        (self
            .curiosity_drive
            .effective_learning_rate(semantic_modulated_lr)
            * self.fep_lr_boost
            * (1.0 + self.carryover.learning.mce_lr_boost)
            * subsystem_lr)
            .clamp(0.0, 0.01)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Extracted helpers from cycle() — Phase 2 (MEDIUM-LOW risk)
    // ═══════════════════════════════════════════════════════════════════════

    /// Run the full moral evaluation phase: throttle, evaluate, apply negation,
    /// contextual weights, and value feedback.
    ///
    /// Returns `(moral_score, moral_concern_detected, moral_judgment)`.
    /// The judgment is cached in `self.last_moral_judgment` for throttled reuse.
    pub(super) fn run_moral_phase(
        &mut self,
        input: &str,
        input_negation_polarity: f32,
    ) -> (f32, bool, MoralJudgmentSummary) {
        // Throttled evaluation: re-evaluate on interval or new input
        let moral_judgment = if self.stats.total_cycles % MORAL_EVAL_INTERVAL == 1
            || self
                .last_moral_judgment
                .as_ref()
                .map_or(true, |j| j.input != input)
        {
            let j = self.evaluate_moral_alignment(input);
            self.stats.moral_evaluations += 1;
            j
        } else {
            self.last_moral_judgment
                .clone()
                .expect("last_moral_judgment guaranteed Some by map_or guard")
        };

        let moral_concern_detected = moral_judgment.moral_score < MORAL_CONCERN_THRESHOLD
            || moral_judgment.consent_violation
            || !moral_judgment.violations.is_empty();

        if moral_concern_detected {
            self.stats.moral_concerns_detected += 1;
        }

        // Write moral stats
        self.stats.moral_score = moral_judgment.moral_score;
        self.stats.consent_violation = moral_judgment.consent_violation;
        self.stats.deontological_violations = moral_judgment
            .violations
            .iter()
            .filter(|v| {
                v.contains("perfect") || v.contains("impermissible") || v.contains("deontological")
            })
            .count();

        // Apply negation polarity: "not harmful" dampens negative moral score toward 0
        let moral_score = if input_negation_polarity > NEGATION_POLARITY_THRESHOLD {
            moral_judgment.moral_score * NEGATION_DAMPENING
        } else {
            moral_judgment.moral_score
        };

        // Contextual harmony weighting: domain-aware moral modulation
        // Science: Haidt (2001) — moral foundations vary across contexts
        let contextual_weight_factor = if let Some(ref mut cw) = self.contextual_weights {
            use crate::consciousness::contextual_weights::{ActionType, DomainClassifier};
            let domain = DomainClassifier::new().classify(input);
            let action_type = if moral_concern_detected {
                ActionType::Governance
            } else {
                ActionType::Basic
            };
            let weights = cw.get_all_weights(action_type, domain);
            let weight_avg =
                weights.iter().map(|(_, w)| w).sum::<f32>() / weights.len().max(1) as f32;
            if weight_avg.abs() < f32::EPSILON {
                1.0
            } else {
                weight_avg
            }
        } else {
            1.0
        };
        let moral_score = moral_score * contextual_weight_factor;

        // Value feedback: self-correcting moral alignment via TD-learning trend
        let value_trend = self.value_feedback.recent_trend(50);
        let moral_feedback = 1.0 + (value_trend * 0.1).clamp(-0.1, 0.1);
        let moral_score = moral_score * moral_feedback;
        {
            let signal = self.value_feedback.create_signal(
                input,
                crate::consciousness::value_feedback_loop::FeedbackType::SelfAssessment,
                moral_score,
            );
            self.value_feedback.process_feedback(signal);
        }

        (moral_score, moral_concern_detected, moral_judgment)
    }

    /// Recall episodic memories and apply emotional/consciousness priming.
    ///
    /// Returns the memory context boost (confidence contribution from recalled
    /// memories). Side effects: biases emotional valence and prediction confidence
    /// from recalled episode metadata (Damasio 1999).
    pub(super) fn recall_episodic_context(&mut self, compressed_state: &[f32]) -> f32 {
        let hdv_sample: Vec<f32> =
            compressed_state[..64.min(compressed_state.len())].to_vec();
        let recalled_memories = self.episodic_memory.recall(
            &hdv_sample,
            MEMORY_RECALL_TOP_K,
            MEMORY_RECALL_SIM_THRESHOLD,
        );

        let memory_context_boost = if !recalled_memories.is_empty() {
            recalled_memories
                .iter()
                .map(|(_, sim)| sim)
                .sum::<f32>()
                / recalled_memories.len().max(1) as f32
                * MEMORY_CONTEXT_BOOST_SCALE
        } else {
            0.0
        };

        // Extract rich context from recalled memories (valence + Phi at encoding time)
        // Science: Damasio (1999) — emotional re-experiencing from recalled episodes
        if !recalled_memories.is_empty() {
            let n = recalled_memories.len() as f32;
            let memory_valence_avg: f32 =
                recalled_memories.iter().map(|(m, _)| m.valence).sum::<f32>() / n;
            let memory_phi_avg: f32 =
                recalled_memories.iter().map(|(m, _)| m.phi_at_encoding).sum::<f32>() / n;

            // Memory valence biases current emotional state (emotional re-experiencing)
            if memory_valence_avg.abs() > 0.1 {
                let valence_nudge = memory_valence_avg * 0.15;
                self.emotion_contagion.valence =
                    (self.emotion_contagion.valence + valence_nudge).clamp(-1.0, 1.0);
            }
            // Memory Phi primes consciousness expectation
            if memory_phi_avg > 0.4 {
                self.prediction_confidence =
                    (self.prediction_confidence + (memory_phi_avg - 0.4) * 0.05).clamp(0.0, 1.0);
            }
        }

        memory_context_boost
    }

    /// Run the surprise exploration bridge cycle.
    ///
    /// Returns `(surprise_triggered, exploration_action)`. Side effects: adjusts
    /// boredom threshold and exploration urge when surprise is detected.
    pub(super) fn run_surprise_exploration(
        &mut self,
        compressed_state: &[f32],
    ) -> (bool, Option<String>) {
        let mut surprise_triggered = false;
        let mut exploration_action = None;

        if let Some(ref mut bridge) = self.surprise_bridge {
            let predicted = self.last_prediction.as_deref().unwrap_or(&[]);
            let actual_len = predicted.len().max(1).min(compressed_state.len());
            let actual = &compressed_state[..actual_len];
            let current_state = self.last_state.as_deref().unwrap_or(compressed_state);
            let (surprise, should_explore, action) =
                bridge.cycle(predicted, actual, current_state);

            if should_explore {
                surprise_triggered = true;
                let current_threshold = self.curiosity_drive.get_boredom_threshold();
                self.curiosity_drive
                    .set_boredom_threshold(current_threshold * SURPRISE_BOREDOM_DAMPEN);
                self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                    + bridge.exploration_factor * 0.3)
                    .clamp(0.0, 1.0);
                exploration_action = action.map(|a| {
                    format!(
                        "perturbation[{}d,scale={:.3}]",
                        a.len(),
                        bridge.exploration_factor
                    )
                });
                tracing::debug!(
                    surprise = surprise,
                    threshold = bridge.tracker().threshold(),
                    exploration_factor = bridge.exploration_factor,
                    cycle = self.stats.total_cycles,
                    "Surprise exploration triggered"
                );
            }
        }

        (surprise_triggered, exploration_action)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Extracted helpers from cycle() — Phase 3 (MEDIUM risk)
    // ═══════════════════════════════════════════════════════════════════════

    /// Compute unified Psi (Layer 1 consciousness estimate) from subsystem contributions.
    ///
    /// Combines: temporal coherence + voice quality + flow state + relational
    /// dyad + interoceptive body + embodied cognition. Clamps to [0.0, 1.0].
    /// Updates the unification engine with the result.
    pub(super) fn compute_unified_psi(&mut self) -> f64 {
        let coherence_psi = self.coherence_bridge.phi_contribution();
        let voice_psi = self.voice_feedback_bridge.summary().phi_adjustment;
        let flow_psi = if self.flow_state.in_flow {
            self.flow_state.intensity * FLOW_PSI_WEIGHT
        } else {
            0.0
        };
        let relational_psi_contrib = if self.relational_psi > 0.0 {
            self.relational_psi as f32 * RELATIONAL_PSI_WEIGHT
        } else {
            0.0
        };
        let body_psi_contrib =
            (self.carryover.consciousness.body_phi_modulation - 1.0) * BODY_PSI_WEIGHT;
        let embodied_psi_contrib =
            (self.carryover.consciousness.embodied_phi_modulation - 1.0) * EMBODIED_PSI_WEIGHT;
        let unified_psi = (coherence_psi
            + voice_psi
            + flow_psi
            + relational_psi_contrib
            + body_psi_contrib as f32
            + embodied_psi_contrib as f32)
            .clamp(0.0, 1.0) as f64;
        self.unification_engine.update_psi(unified_psi);
        unified_psi
    }

    /// Compute the cycle reward signal for reinforcement learning.
    ///
    /// Blends internal reward (based on prediction error vs learning threshold)
    /// with any pending external reward. Consumes the external reward.
    /// Returns the clamped reward in [-1.0, 1.0].
    pub(super) fn compute_reward_signal(
        &mut self,
        prediction_error: f32,
        learning_threshold: f32,
    ) -> f32 {
        let internal_reward = if prediction_error < learning_threshold {
            REWARD_GOOD_BASE + REWARD_GOOD_CONFIDENCE_SCALE * self.prediction_confidence
        } else if prediction_error > 0.5 {
            REWARD_BAD_BASE + REWARD_BAD_SCALE * (prediction_error - 0.5)
        } else {
            REWARD_MID_BASE + REWARD_MID_SCALE * prediction_error
        };

        // FEP free energy reduction enrichment
        // Science: Friston (2010) — free energy minimization as the objective
        // Reward FE reduction (prev > current = improving model)
        let fep_bonus = if let Some(ref fe) = self.fep_agent.last_fe_components {
            let current_fe = fe.total;
            let prev_fe = self.stats.last_total_fe;
            // Update cached value for next cycle
            self.stats.last_total_fe = current_fe;
            if prev_fe > 0.0 {
                // FE reduction: positive when improving (prev > current)
                let fe_reduction = (prev_fe - current_fe).clamp(-1.0, 1.0);
                (fe_reduction * 0.15) as f32 // 15% weight for FEP signal
            } else {
                0.0 // first cycle — no previous FE
            }
        } else {
            0.0
        };

        let enriched_reward = internal_reward + fep_bonus;

        let cycle_reward = if self.external_reward.abs() > f32::EPSILON {
            let blended = enriched_reward * REWARD_EXTERNAL_BLEND
                + self.external_reward * REWARD_EXTERNAL_BLEND;
            self.external_reward = 0.0; // consume
            blended
        } else {
            enriched_reward
        };
        cycle_reward.clamp(-1.0, 1.0)
    }

    /// Apply initial strategy modulation to adaptive behavior.
    ///
    /// Sets exploration factor, attention sensitivity, speech rate, or pause
    /// multiplier based on the selected response strategy.
    pub(super) fn apply_strategy_modulation(&mut self, strategy: ResponseStrategy) {
        match strategy {
            ResponseStrategy::Exploratory => {
                self.adaptive_behavior.exploration_factor = STRATEGY_EXPLORATORY_FACTOR;
            }
            ResponseStrategy::Detailed => {
                self.adaptive_behavior.attention_sensitivity = STRATEGY_DETAILED_SENSITIVITY;
            }
            ResponseStrategy::Concise => {
                self.adaptive_behavior.speech_rate_multiplier = STRATEGY_CONCISE_SPEECH_RATE;
            }
            ResponseStrategy::Clarifying => {
                self.adaptive_behavior.exploration_factor = STRATEGY_CLARIFYING_FACTOR;
            }
            ResponseStrategy::Supportive => {
                self.adaptive_behavior.pause_multiplier = STRATEGY_SUPPORTIVE_PAUSE;
            }
        }
    }

    /// Re-apply strategy modulations ON TOP of consciousness-derived base.
    ///
    /// Uses max/min/multiply to merge strategy with consciousness state,
    /// preserving the stronger signal. Called after `from_consciousness_state()`
    /// resets adaptive behavior.
    pub(super) fn reapply_strategy_modulation(&mut self, strategy: ResponseStrategy) {
        match strategy {
            ResponseStrategy::Exploratory => {
                self.adaptive_behavior.exploration_factor =
                    self.adaptive_behavior.exploration_factor.max(STRATEGY_EXPLORATORY_FACTOR);
            }
            ResponseStrategy::Detailed => {
                self.adaptive_behavior.attention_sensitivity *= STRATEGY_DETAILED_SENSITIVITY;
            }
            ResponseStrategy::Concise => {
                self.adaptive_behavior.speech_rate_multiplier = self
                    .adaptive_behavior
                    .speech_rate_multiplier
                    .max(STRATEGY_CONCISE_SPEECH_RATE);
            }
            ResponseStrategy::Clarifying => {
                self.adaptive_behavior.exploration_factor =
                    self.adaptive_behavior.exploration_factor.min(STRATEGY_CLARIFYING_FACTOR);
            }
            ResponseStrategy::Supportive => {
                self.adaptive_behavior.pause_multiplier =
                    self.adaptive_behavior.pause_multiplier.max(STRATEGY_SUPPORTIVE_PAUSE);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 4: Medium-high risk extractions
    // ═══════════════════════════════════════════════════════════════════════════

    /// FEP active inference perception-action loop.
    ///
    /// Constructs an observation from current cognitive state, runs the FEP agent's
    /// perceive→select_action→act pipeline, then applies action-specific modulations
    /// (learning rate boost, sensory precision reset, exploration nudge, trust tightening).
    ///
    /// Returns (action_index, action_probabilities, is_surprised).
    pub(super) fn step_fep_active_inference(
        &mut self,
        prediction_error: f32,
        coherence: f32,
    ) -> (usize, Vec<f64>, bool, f64) {
        let effective_lr = self.stats.adaptive_learning_rate;
        let fep_obs = Observation::from_consciousness_state(
            prediction_error as f64,
            coherence as f64,
            self.prediction_confidence as f64,
            effective_lr as f64,
        );
        let _perception = self.fep_agent.perceive(&fep_obs);
        let action_result = self.fep_agent.select_action();
        let _outcome = self.fep_agent.act(action_result.action);

        let fep_action_idx = action_result.action;
        let fep_action_probs = action_result.action_probabilities.clone();

        let is_surprised = self.fep_agent.is_surprised();
        match action_result.action {
            0 => {
                // Boost learning rate when free energy is high
                if let Some(ref fe) = self.fep_agent.last_fe_components {
                    let fe_boost = (fe.total.abs() as f32 / 2.0).clamp(0.0, 1.5);
                    self.fep_lr_boost =
                        (self.fep_lr_boost * (1.0 + fe_boost * 0.5)).clamp(1.0, 2.0);
                }
            }
            1 => {
                // Reset sensory precision toward 1.0 to trust new observations after shift
                let current = self.fep_agent.precision.sensory_precision;
                self.fep_agent.precision.sensory_precision = current * 0.7 + 1.0 * 0.3;
            }
            2 => {
                // Boost exploration -- stronger nudge when surprised
                let nudge = if is_surprised { 0.15 } else { 0.05 };
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + nudge).clamp(0.0, 1.0);
            }
            3 => {
                // Tighten trust via precision
                if let Some(ref fe) = self.fep_agent.last_fe_components {
                    let precision_mod = (1.0 - fe.prediction_error).clamp(0.0, 1.0) as f32;
                    self.self_reflection.trust_threshold =
                        (self.self_reflection.trust_threshold * 0.9 + precision_mod * 0.1)
                            .clamp(0.1, 0.9);
                }
            }
            _ => {}
        }

        (fep_action_idx, fep_action_probs, is_surprised, action_result.pragmatic_value)
    }

    /// Cross-modal binding: bind HDC encodings across linguistic and affective modalities.
    ///
    /// Clears stale representations, adds linguistic (from BinaryHV) and affective
    /// (if bridge enabled) modalities, computes binding strength and cross-modal Psi.
    /// Also applies feedback loops: high Psi boosts prediction confidence, high binding
    /// boosts predictive precision, high free energy dampens binding attention.
    ///
    /// Returns (binding_strength, cross_modal_psi).
    pub(super) fn update_cross_modal_binding(
        &mut self,
        hv16: &symthaea_core::hdc::binary_hv::BinaryHV,
        affective_valence: f32,
        predictive_free_energy: f64,
    ) -> (f32, f64) {
        let (cross_modal_binding_strength, cross_modal_psi) =
            if let Some(ref mut binder) = self.cross_modal_binder {
                use symthaea_core::hdc::unified_hv::ContinuousHV;
                binder.clear();
                let linguistic_repr = ModalRepresentation::new(
                    Modality::Linguistic,
                    ContinuousHV::from_vec(hv16.to_bipolar()),
                    0.8,
                    "encoder",
                );
                binder.add_representation(linguistic_repr);
                if self.affective_bridge.is_some() {
                    let affect_seed = (affective_valence * 1000.0) as u64;
                    let affective_hv =
                        symthaea_core::hdc::binary_hv::BinaryHV::random(affect_seed);
                    binder.update_modality(Modality::Affective, affective_hv);
                }
                let strength = binder.bind().map(|r| r.strength).unwrap_or(0.0);
                let phi = binder.cross_modal_psi();
                self.carryover.consciousness.cross_modal_psi = phi;
                (strength, phi)
            } else {
                (0.0, 0.0)
            };

        // FEEDBACK: High cross-modal Phi boosts confidence (binding integration)
        // Science: Treisman (1996) — coherent binding → confident perception
        if cross_modal_psi > 0.3 {
            let boost = ((cross_modal_psi - 0.3) * 0.05) as f32;
            self.prediction_confidence = (self.prediction_confidence + boost).clamp(0.0, 1.0);
        }

        // FEEDBACK: Predictive ↔ Cross-Modal bidirectional coupling (Talsma 2015)
        if let Some(ref mut mind) = self.predictive_mind {
            if cross_modal_binding_strength > 0.5 {
                let precision_boost = (cross_modal_binding_strength - 0.5) as f64 * 0.1;
                mind.precision.boost_precision(precision_boost);
            }
        }
        if let Some(ref mut binder) = self.cross_modal_binder {
            if predictive_free_energy > 0.6 {
                let dampen = (1.0 - (predictive_free_energy - 0.6) * 0.3).max(0.5) as f32;
                binder.set_attention_weight(dampen);
            }
        }

        (cross_modal_binding_strength, cross_modal_psi)
    }
}
