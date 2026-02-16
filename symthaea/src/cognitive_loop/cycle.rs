//! Core cognitive cycle implementation with parallel post-processing.
//!
//! Contains the main `cycle()` method which implements the bidirectional
//! HDC-CfC loop with rayon-parallelized subsystem updates.

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use crate::consciousness::stability_regime::RegimeTransition;
use ndarray::Array1;
use rayon::join as rayon_join;
use std::time::Instant;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;

use super::temporal_network::TemporalNetwork;
use super::training::TrainingSample;
use super::{
    AdaptiveBehavior, CognitiveLoopService, CycleLearningResult, CycleResult, ResponseStrategy,
    TrainingMethod,
};

impl CognitiveLoopService {
    /// Run one cognitive cycle (the core loop)
    ///
    /// Uses CfC's O(1) closed-form solution for temporal prediction,
    /// enabling instant forward-time queries and multi-scale prediction.
    ///
    /// ## Mega-Unified Architecture Integration
    ///
    /// This cycle now integrates:
    /// - **Thalamic Routing**: Determines cognitive depth (Reflex/Cortical/DeepThought)
    /// - **ConsciousnessUnificationEngine**: Unified emotional bridge with VAD emotions
    /// - **Phi Updates**: Feeds consciousness level to the unification engine
    /// - **Moral Algebra**: Evaluates ethical alignment of inputs
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE -1: Ingest background-trained weights (non-blocking)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut trainer) = self.async_trainer {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                trainer.apply_latest_weights(cfc);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0: Thalamic Routing (Cognitive Depth Selection)
        // ═══════════════════════════════════════════════════════════════════════
        // Determine how deep to process BEFORE encoding, based on prior state

        let prior_pattern = self.temporal_signature_encoder.classify_state().0;
        let prior_valence = self.emotion_contagion.prosody_valence();
        let prior_error = self.stats.avg_prediction_error;

        self.cognitive_depth =
            self.thalamic_router
                .route_from_cycle(prior_error, prior_pattern, prior_valence);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.4: Moral Evaluation
        // ═══════════════════════════════════════════════════════════════════════
        // Evaluate input for moral alignment using HDC-based moral algebra.
        // This informs downstream processing and can trigger ethical safeguards.

        let moral_judgment = self.evaluate_moral_alignment(input);
        let moral_concern_detected = moral_judgment.moral_score < -0.3
            || moral_judgment.consent_violation
            || !moral_judgment.violations.is_empty();

        // Update stats with moral evaluation
        self.stats.moral_evaluations += 1;
        if moral_concern_detected {
            self.stats.moral_concerns_detected += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        // Select response strategy BEFORE processing, based on:
        // - Q-learning from past interactions
        // - Previous reward (stick with success, avoid failure)
        // - Phi-gating (high Phi -> Exploratory, low Phi -> Supportive)
        // - Moral concerns (bias toward Supportive for ethical guidance)

        let prior_phi = self.unification_engine.phi;
        let prior_reward = self
            .closed_learning_loop
            .last_result
            .as_ref()
            .map(|r| r.reward);
        let selected_strategy = if moral_concern_detected {
            // Bias toward supportive strategy when moral concerns detected
            ResponseStrategy::Supportive
        } else {
            self.closed_learning_loop
                .select_strategy(prior_phi, prior_reward)
        };

        // Strategy influences adaptive behavior
        match selected_strategy {
            ResponseStrategy::Exploratory => {
                self.adaptive_behavior.exploration_factor = 0.8;
            }
            ResponseStrategy::Detailed => {
                self.adaptive_behavior.attention_sensitivity = 1.2;
            }
            ResponseStrategy::Concise => {
                self.adaptive_behavior.speech_rate_multiplier = 1.2;
            }
            ResponseStrategy::Clarifying => {
                self.adaptive_behavior.exploration_factor = 0.5;
            }
            ResponseStrategy::Supportive => {
                self.adaptive_behavior.pause_multiplier = 1.3;
            }
        }

        // 1. HDC encode with attention from previous prediction
        let encoding_result = self.encoder.encode(input);
        let prediction_error = encoding_result.prediction_error;

        // ═══════════════════════════════════════════════════════════════════════
        // 1.1 Surprise-Driven Exploration: Track surprise, modulate curiosity
        // ═══════════════════════════════════════════════════════════════════════
        // When enabled, feed prediction error to surprise bridge. If surprise
        // exceeds the adaptive threshold, lower the boredom threshold to
        // encourage exploration of novel states.
        let mut surprise_triggered = false;
        let mut exploration_action: Option<String> = None;
        if let Some(ref mut bridge) = self.surprise_bridge {
            let predicted = self.last_prediction.as_deref().unwrap_or(&[]);
            let actual = encoding_result
                .hdv
                .as_slice()
                .get(..predicted.len().max(1).min(64))
                .unwrap_or(&[]);
            let current_state = self
                .last_state
                .as_deref()
                .unwrap_or(&[0.0; 8]);
            let (surprise, should_explore, action) =
                bridge.cycle(predicted, actual, current_state);

            if should_explore {
                surprise_triggered = true;
                // Lower boredom threshold to encourage exploration
                let current_threshold = self.curiosity_drive.get_boredom_threshold();
                self.curiosity_drive
                    .set_boredom_threshold(current_threshold * 0.7);
                // Boost exploration urge proportional to surprise intensity
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + bridge.exploration_factor * 0.3)
                        .clamp(0.0, 1.0);
                exploration_action =
                    action.map(|a| format!("perturbation[{}d,scale={:.3}]", a.len(), bridge.exploration_factor));
                tracing::debug!(
                    surprise = surprise,
                    threshold = bridge.tracker().threshold(),
                    exploration_factor = bridge.exploration_factor,
                    cycle = self.stats.total_cycles,
                    "Surprise exploration triggered"
                );
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a. Memory System Integration: Recall relevant episodic memories
        // ═══════════════════════════════════════════════════════════════════════
        // Use HDC embedding to query episodic memory for context

        let hdv_sample: Vec<f32> =
            encoding_result.hdv.as_slice()[..64.min(encoding_result.hdv.dim())].to_vec();
        let recalled_memories = self.episodic_memory.recall(&hdv_sample, 3, 0.3);
        let memory_context_boost = if !recalled_memories.is_empty() {
            // Recalled memories boost prediction confidence slightly (safe division with max(1))
            recalled_memories.iter().map(|(_, sim)| sim).sum::<f32>()
                / recalled_memories.len().max(1) as f32
                * 0.1
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.2. Goal System: Apply attention bias from active goals
        // ═══════════════════════════════════════════════════════════════════════

        let goal_attention_bias = self.goal_system.attention_bias();
        self.adaptive_behavior.attention_sensitivity *= goal_attention_bias;

        // 1b. Analyze emotional content for simple contagion (keyword-based)
        self.emotion_contagion.analyze(input);

        // ═══════════════════════════════════════════════════════════════════════
        // 1c. Update Unified Emotional Bridge (VAD-based, richer than simple contagion)
        // ═══════════════════════════════════════════════════════════════════════
        // Bridge the simple EmotionContagion to the unified EmotionalBridge
        // Convert valence/arousal to the full VAD emotional system

        let simple_valence = self.emotion_contagion.prosody_valence() as f64;
        let simple_arousal = self.emotion_contagion.prosody_arousal() as f64;
        // Dominance estimated from confidence and flow state
        let dominance = if self.flow_state.in_flow {
            0.6 + 0.2 * self.flow_state.intensity as f64
        } else if self.prediction_confidence > 0.6 {
            0.4
        } else {
            0.2
        };

        self.unification_engine.emotional.update_from_core_affect(
            simple_valence,
            simple_arousal,
            dominance,
        );

        // 2. Compress HDC state for CfC (using Random Projection)
        let compressed_state = self
            .encoder
            .compress_for_ltc(&encoding_result.hdv, self.config.cfc_config.input_dim);

        // ═══════════════════════════════════════════════════════════════════════
        // 2a. SEMANTIC MEMORY: HDC-based similarity lookup for CfC context
        // ═══════════════════════════════════════════════════════════════════════
        // Project to HDC space and find similar past inputs.
        // Use their prediction errors to modulate learning rate:
        // - High error on similar inputs -> boost learning (we struggled before)
        // - Low error on similar inputs -> reduce learning (familiar territory)
        //
        // For HdcLtc backend: use the native HDC projection
        // For CfC backend: use the compressed state as the semantic vector

        let semantic_hdc = self
            .temporal_network
            .project_to_hdc_vec(&compressed_state)
            .unwrap_or_else(|| compressed_state.clone());
        // Phi-weighted learning rate: consciousness level modulates how aggressively
        // we adjust to prediction errors on similar past inputs.
        let current_phi_for_lr = self.coherence_bridge.smoothed_coherence() as f64;
        let semantic_lr_factor = self.semantic_memory.compute_lr_factor_phi_weighted(
            &semantic_hdc,
            3,
            current_phi_for_lr,
            self.stats.total_cycles as u64,
        );

        // 3. Convert to ndarray for CfC
        let input_array = Array1::from_vec(compressed_state.clone());

        // 4. Step CfC forward with current input
        let delta_t = self.config.cfc_config.delta_t;
        let _ = self.temporal_network.step(&input_array, delta_t);

        // 5. Get multi-scale predictions using CfC's O(1) predict_forward
        // This is the key advantage: instant prediction at any future time
        let prediction = self.get_multi_scale_prediction(&input_array);

        // 6. Get current CfC state as output
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);

        // ═══════════════════════════════════════════════════════════════════════
        // 6b. World Model: Update hierarchical world model with sensory input
        // ═══════════════════════════════════════════════════════════════════════

        self.world_model.update_sensory(&compressed_state);

        // 7. Send prediction to encoder for next cycle
        self.encoder.set_prediction(prediction.clone());

        // 8. Capture previous state BEFORE create_experience updates it
        let previous_state = self.last_state.clone();

        // 9. Create experience and add to buffer (this updates last_state)
        self.create_experience(&compressed_state, &prediction, prediction_error);

        // 10. Update coherence bridge with current tau values
        // Note: We use all_tau_owned() for backend compatibility (HdcLtc returns owned values)
        let tau_owned: Vec<ndarray::Array1<f32>> = self.temporal_network.all_tau_owned();
        let tau_refs: Vec<&ndarray::Array1<f32>> = tau_owned.iter().collect();
        self.coherence_bridge.update(&tau_refs);

        // 10b. Update temporal signature encoder with tau values
        // Record mean tau for consciousness pattern detection
        let flattened_tau = self.temporal_network.flattened_tau();
        self.temporal_signature_encoder.record_batch(&flattened_tau);

        // 10c. Update adaptive behavior based on consciousness state
        let (pattern, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let coherence = self.coherence_bridge.smoothed_coherence();
        let voice_confidence = self.voice_feedback_bridge.summary().voice_confidence;
        self.adaptive_behavior = AdaptiveBehavior::from_consciousness_state(
            pattern,
            pattern_confidence,
            coherence,
            voice_confidence,
        );

        // 10d. Update prediction confidence with decay during uncertain states
        self.update_prediction_confidence(pattern, prediction_error, pattern_confidence);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.5 Active Inference Bridge: Observe prediction resolution for PAC tracking
        // ═══════════════════════════════════════════════════════════════════════
        // Track prediction-outcome coupling via Phase-Amplitude Coupling (PAC)
        // This enables precision-weighted prediction errors

        // Consider prediction "successful" if error is below learning threshold
        let prediction_success = prediction_error < self.config.learning_threshold;
        self.active_inference_bridge
            .observe_resolution(self.prediction_confidence as f64, prediction_success);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6 FEP Active Inference: Full perception-action loop
        // ═══════════════════════════════════════════════════════════════════════
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

        // Apply FEP-selected action to modulate cognitive parameters
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

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Moral Modulation of Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        // Apply moral constraints to FEP-selected actions:
        // - Negative moral score -> reduce exploration, increase caution
        // - Consent violation -> strong ethical override
        // - Deontological violations -> trigger reflective pause

        if moral_concern_detected {
            // Reduce exploration when facing moral concerns
            self.curiosity_drive.exploration_urge *= 0.5;

            // Increase trust threshold (be more cautious)
            self.self_reflection.trust_threshold =
                (self.self_reflection.trust_threshold * 1.2).clamp(0.1, 0.95);

            // Boost reflective processing (take time to consider ethics)
            self.adaptive_behavior.pause_multiplier *= 1.5;

            // If severe moral violation (perfect duty or consent), flag for review
            if moral_judgment.consent_violation
                || moral_judgment
                    .violations
                    .iter()
                    .any(|v| v.contains("perfect") || v.contains("harm"))
            {
                self.stats.moral_review_needed = true;
            }
        } else if moral_judgment.moral_score > 0.5 {
            // Positive moral alignment boosts confidence slightly
            self.prediction_confidence = (self.prediction_confidence * 1.05).clamp(0.0, 1.0);
        }

        // Surprise-gated learning rate boost: when FEP detects surprise, accelerate adaptation
        if is_surprised {
            let surprise_boost =
                (self.fep_agent.current_free_energy() as f32 / 3.0).clamp(0.1, 0.5);
            self.fep_lr_boost = (self.fep_lr_boost + surprise_boost).clamp(1.0, 2.0);
        } else {
            // Decay boost back toward 1.0 when not surprised
            self.fep_lr_boost = (self.fep_lr_boost * 0.95).max(1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6b Enhanced FEP Bridge: Motor commands and learning signals
        // ═══════════════════════════════════════════════════════════════════════
        // Run enhanced FEP cycle for motor system integration and learning signals
        let enhanced_result = self.enhanced_fep_bridge.cycle(
            prediction_error as f64,
            coherence as f64,
            self.prediction_confidence as f64,
            effective_lr as f64,
        );

        // Update learning signal for downstream systems
        self.fep_learning_signal = enhanced_result.learning_signal as f32;

        // Apply motor command-based modulations
        match enhanced_result.motor_command.command_type {
            MotorCommandType::AttentionShift => {
                // Shift attention based on motor command intensity
                let shift_amount = enhanced_result.motor_command.intensity as f32 * 0.1;
                // Could modulate HDC attention weights here
                self.stats.attention_shift = shift_amount;
            }
            MotorCommandType::LearningRateAdjust => {
                // Precision-weighted learning rate adjustment
                if enhanced_result.should_learn {
                    let lr_mod = enhanced_result.fep_result.learning_rate_modulation as f32;
                    self.stats.adaptive_learning_rate =
                        (self.stats.adaptive_learning_rate * 0.9 + lr_mod * 0.1).clamp(0.01, 1.0);
                }
            }
            MotorCommandType::ExplorationTrigger => {
                // Boost exploration based on epistemic value
                if enhanced_result.fep_result.epistemic_value > 0.5 {
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge + 0.1).clamp(0.0, 1.0);
                }
            }
            MotorCommandType::ReflectionInitiate => {
                // Force reflection if motor command intensity is high
                if enhanced_result.motor_command.intensity > 0.7 {
                    self.self_reflection.force_reflection();
                }
            }
            MotorCommandType::MemoryConsolidate => {
                // Signal episodic memory for consolidation
                if enhanced_result.motor_command.intensity > 0.5 {
                    self.episodic_memory.consolidate_recent();
                }
            }
            MotorCommandType::ExpectationReset => {
                // Clear prediction cache if action-outcome coupling is poor
                if enhanced_result.action_outcome_coupling < 0.3 {
                    self.last_prediction = None;
                    self.prediction_confidence = 0.5;
                }
            }
            MotorCommandType::MotorOutput | MotorCommandType::NoOp => {
                // No cognitive modulation
            }
        }

        // Use learning signal to modulate other systems
        if self.fep_learning_signal > 0.5 && enhanced_result.should_learn {
            // High learning signal: increase plasticity in world model
            self.world_model
                .increase_plasticity(self.fep_learning_signal);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Coherence tracking with degradation detection
        // ═══════════════════════════════════════════════════════════════════════
        let degraded = self.coherence_tracker.record_turn(coherence);
        if degraded {
            // Coherence degradation -> boost learning rate to accelerate recovery
            self.fep_lr_boost = (self.fep_lr_boost * 1.3).clamp(1.0, 2.0);
            let urgency = self.coherence_tracker.correction_urgency();
            // Feed urgency as a high-error observation to drive FEP learning
            let urgent_obs = Observation::from_consciousness_state(
                urgency as f64,
                0.1,
                0.1,
                effective_lr as f64,
            );
            self.fep_agent.perceive(&urgent_obs);
            // Also signal enhanced bridge about degradation
            self.enhanced_fep_bridge
                .cycle(urgency as f64, 0.1, 0.1, effective_lr as f64);
        }

        // 10e. Update flow state with adaptive thresholds from self-reflection
        let adapted_thresholds = self.self_reflection.get_thresholds();
        self.flow_state.update_with_thresholds(
            pattern,
            prediction_error,
            coherence,
            self.prediction_confidence,
            adapted_thresholds.flow_error,
            adapted_thresholds.flow_coherence,
        );

        // 10f. Update curiosity drive with adaptive boredom threshold
        self.curiosity_drive
            .set_boredom_threshold(adapted_thresholds.boredom);
        self.curiosity_drive.update(prediction_error);

        // 10g. Self-reflection for meta-learning
        self.self_reflection.record_cycle(
            prediction_error,
            self.flow_state.in_flow,
            self.curiosity_drive.should_explore(),
            self.prediction_confidence,
        );
        // Perform reflection if it's time (adjusts thresholds automatically)
        if self.self_reflection.should_reflect() {
            let _recommendations = self.self_reflection.reflect();
            // Recommendations are stored in self_reflection.recommendations
            // and can be queried by external systems
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h. Update Consciousness Unification Engine with current Phi
        // ═══════════════════════════════════════════════════════════════════════
        // Compute unified Phi from coherence, confidence, and flow state
        // This feeds the dialogue pipeline for consciousness-aware responses

        let coherence_phi = self.coherence_bridge.phi_contribution();
        let voice_phi = self.voice_feedback_bridge.summary().phi_adjustment;
        let flow_phi = if self.flow_state.in_flow {
            self.flow_state.intensity * 0.2
        } else {
            0.0
        };
        // Combine contributions: temporal coherence + voice quality + flow state
        let unified_phi = (coherence_phi + voice_phi + flow_phi).clamp(0.0, 1.0) as f64;
        self.unification_engine.update_phi(unified_phi);

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1 Conscious Reasoning Engine: unified 7-step reasoning cycle
        // ═══════════════════════════════════════════════════════════════════════
        // When the reasoning_engine feature is enabled, run the full conscious
        // reasoning cycle (conflict detection -> Phi_eff -> planning -> gating ->
        // counterfactual -> telemetry) with tiered degradation.
        #[allow(unused_mut)]
        let mut reasoning_confidence: f32 = 0.0;
        #[allow(unused_mut)]
        let mut reasoning_lr_factor: f32 = 1.0;
        #[cfg(feature = "reasoning_engine")]
        if let Some(ref mut reasoning_engine) = self.reasoning_engine {
            use crate::consciousness::epistemic_conflict::MultiTheoryMetrics as ECMetrics;
            use crate::consciousness::reasoning_engine::ReasoningContext;

            // Build theory metrics from available consciousness signals
            let ec_metrics = ECMetrics {
                phi: unified_phi,
                gwt: coherence as f64,
                ast: self.prediction_confidence as f64,
                pp: (1.0 - prediction_error as f64).clamp(0.0, 1.0),
                rpt: pattern_confidence as f64,
                embodiment: self.fep_learning_signal as f64,
                unified: unified_phi,
            };

            // Compute available budget: 20ms target cycle minus time already spent
            let elapsed_us = cycle_start.elapsed().as_micros() as u64;
            let available_us = 20_000u64.saturating_sub(elapsed_us);

            let reasoning_ctx = ReasoningContext {
                theory_metrics: ec_metrics,
                phi: unified_phi,
                available_budget_us: available_us,
                available_actions: Vec::new(), // populated by external action providers
                tool: None,                    // populated by shell integration
                recent_utility: 0.5,
                cycle_id: self.stats.total_cycles as u64,
            };

            let reasoning_result = reasoning_engine.reason(&reasoning_ctx);

            // Capture reasoning outputs for downstream use:
            // 1. Phi_eff modulates confidence (higher = more reliable reasoning)
            reasoning_confidence = reasoning_result.phi_eff as f32;

            // 2. Reliability modulates learning rate — low reliability = cautious learning
            reasoning_lr_factor = reasoning_result.reliability as f32;

            // 3. Log reasoning tier and timing for observability
            tracing::debug!(
                tier = ?reasoning_result.tier,
                phi_eff = reasoning_result.phi_eff,
                reliability = reasoning_result.reliability,
                wall_time_us = reasoning_result.wall_time_us,
                budget_exceeded = reasoning_result.budget_exceeded,
                "Reasoning engine cycle"
            );
        }

        // Get adaptive learning rate (respects pause_learning and all modulations)
        // Include flow state boost, curiosity novelty bonus, and semantic context
        let base_lr = self.combined_learning_rate();
        let adaptive_lr = self.adaptive_behavior.effective_learning_rate(base_lr);
        let flow_lr = self.flow_state.effective_learning_multiplier(adaptive_lr);
        // Apply semantic memory modulation: boost learning when similar inputs had high error
        // Also apply reasoning engine reliability factor (low reliability = cautious learning)
        let semantic_modulated_lr = flow_lr * semantic_lr_factor * reasoning_lr_factor;
        let effective_lr = (self
            .curiosity_drive
            .effective_learning_rate(semantic_modulated_lr)
            * self.fep_lr_boost)
            .clamp(0.0, 0.01); // Hard cap: reduced from 0.05 to 0.01 to prevent oscillation with cyclic patterns

        // 11. Learn if error is significant AND we have a previous state AND not paused
        let (learning_occurred, training_loss) = if prediction_error
            > self.config.learning_threshold
            && !self.adaptive_behavior.pause_learning
        {
            self.stats.learning_cycles += 1;

            // Build training sample
            let (train_input, train_target, lr) = if let Some(ref prev) = previous_state {
                (
                    Array1::from_vec(prev.clone()),
                    Array1::from_vec(compressed_state.clone()),
                    effective_lr,
                )
            } else {
                // First cycle: bootstrap with self-prediction
                let current_array = Array1::from_vec(compressed_state.clone());
                (current_array.clone(), current_array, effective_lr * 0.1)
            };

            // --- Async path: send sample to background thread (never blocks) ---
            if let Some(ref trainer) = self.async_trainer {
                trainer.send(TrainingSample {
                    input: train_input,
                    target: train_target,
                    dt: delta_t,
                    learning_rate: lr,
                    method: self.config.training_method,
                    avg_loss: self.stats.avg_training_loss,
                });
                // Loss arrives later via weight updates; mark learning in-flight.
                (true, None)
            } else {
                // --- Sync path: train inline (original behaviour) ---
                let result = match self.config.training_method {
                    TrainingMethod::Spsa => {
                        self.stats.spsa_fallback_steps += 1;
                        self.temporal_network.train_step_spsa(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        )
                    }
                    TrainingMethod::Bptt => {
                        self.stats.bptt_steps += 1;
                        self.temporal_network.train_step_bptt(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        )
                    }
                    TrainingMethod::BpttWithSpsaFallback => {
                        let old_loss = self.stats.avg_training_loss;
                        let bptt_result = self.temporal_network.train_step_bptt(
                            &train_input,
                            &train_target,
                            delta_t,
                            lr,
                        );
                        match bptt_result {
                            Ok(loss)
                                if loss.is_finite()
                                    && (old_loss <= 0.0 || loss < old_loss * 2.0) =>
                            {
                                self.stats.bptt_steps += 1;
                                Ok(loss)
                            }
                            _ => {
                                self.stats.spsa_fallback_steps += 1;
                                self.temporal_network.train_step_spsa(
                                    &train_input,
                                    &train_target,
                                    delta_t,
                                    lr,
                                )
                            }
                        }
                    }
                };

                match result {
                    Ok(loss) => {
                        self.update_loss_stats(loss);
                        (true, Some(loss))
                    }
                    Err(_) => (false, None),
                }
            }
        } else {
            (false, None)
        };

        // 12. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());

        // Update state diversity from CfC
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Adaptive HDC dimension: resize if error demands it
        self.temporal_network.maybe_resize(prediction_error);

        // Update coherence metrics in stats
        self.stats.temporal_coherence = self.coherence_bridge.smoothed_coherence();
        self.stats.effective_learning_rate = effective_lr;
        self.stats.coherence_phi_contribution = self.coherence_bridge.phi_contribution();

        // ═══════════════════════════════════════════════════════════════════════
        // PARALLEL POST-PROCESSING: Independent subsystem updates via rayon
        // ═══════════════════════════════════════════════════════════════════════
        // These subsystem updates use disjoint fields and run in parallel:
        //   Branch A: Stability regime + Semantic memory + Causal enhancement
        //   Branch B: Episodic memory + Primitive-belief bridge + Closed learning loop
        // Sequential after join: Episodic replay + Memory coordinator (cross-dependencies)

        // Pre-compute read-only values needed by parallel branches
        let pp_total_cycles = self.stats.total_cycles;
        let pp_in_flow = self.flow_state.in_flow;
        let pp_emotional_valence = self.emotion_contagion.prosody_valence();
        let pp_phi = self.unification_engine.phi as f32;
        let pp_smoothed_coh = self.coherence_bridge.smoothed_coherence() as f64;
        let pp_learning_threshold = self.config.learning_threshold;

        // Compute cycle reward before parallel section (reads prediction_confidence, flow_state)
        let cycle_reward = if prediction_error < pp_learning_threshold {
            0.5 + 0.5 * self.prediction_confidence
        } else if prediction_error > 0.5 {
            -0.3 - 0.2 * (prediction_error - 0.5)
        } else {
            0.2 - 0.5 * prediction_error
        };

        let cycle_learning_result = CycleLearningResult {
            reward: cycle_reward.clamp(-1.0, 1.0),
            strategy_used: selected_strategy,
            successful: prediction_error < pp_learning_threshold && pp_in_flow,
            prediction_error,
            coherence,
        };

        // Take disjoint mutable borrows for parallel processing.
        // The block scope ensures all borrows are dropped before the sequential section.
        {
            let stability_regime = &mut self.stability_regime;
            let discovery_service = &mut self.discovery_service;
            let semantic_memory = &mut self.semantic_memory;
            let causal_enhancer = &mut self.causal_enhancer;
            let episodic_memory = &mut self.episodic_memory;
            let primitive_belief_bridge = &mut self.primitive_belief_bridge;
            let closed_learning_loop = &mut self.closed_learning_loop;
            let fep_learning_signal = &mut self.fep_learning_signal;
            let prev_primitive_state = &mut self.prev_primitive_state;
            let prediction_confidence_ref = &mut self.prediction_confidence;

            rayon_join(
                // -- Branch A: Stability Regime + Semantic Memory + Causal Enhancement --
                || {
                    // Stability regime: CfC dynamics for primitives
                    // Frequently-used primitives crystallize, rarely-used stay fluid
                    let hv16_input = real_hv_to_hv16(&encoding_result.hdv);
                    let timestamp = pp_total_cycles as f64 * delta_t as f64;
                    let (_regime_state, transitions) =
                        stability_regime.process_input(&hv16_input, delta_t, timestamp);

                    for transition in &transitions {
                        if let RegimeTransition::Crystallized {
                            primitive_name,
                            encoding,
                        } = transition
                        {
                            discovery_service.seed_neighbor_exploration(primitive_name, encoding);
                        }
                    }

                    // Semantic memory: store HDC vector + prediction error for future similarity lookup
                    semantic_memory.store_with_timestamp(
                        semantic_hdc,
                        prediction_error,
                        None,
                        pp_total_cycles as u64,
                    );

                    // Causal enhancement: track (input, output) pairs and discover structure
                    if let Some(ref mut enhancer) = causal_enhancer {
                        enhancer.record_cycle_from_f32(&compressed_state, &output);

                        if enhancer.should_discover() {
                            let causal_graph = enhancer.run_discovery();

                            if !causal_graph.is_empty() {
                                tracing::info!(
                                    edges = causal_graph.edges.len(),
                                    cycle = pp_total_cycles,
                                    "Causal structure discovered in cognitive loop"
                                );
                                enhancer.log_discoveries();
                            }
                        }
                    }
                },
                // -- Branch B: Episodic Memory + Primitive-Belief + Closed Learning --
                || {
                    // Episodic memory: encode significant experiences
                    if prediction_error > 0.1 || pp_in_flow {
                        episodic_memory.encode(
                            input,
                            hdv_sample.clone(),
                            pp_emotional_valence,
                            pp_phi,
                            pp_total_cycles,
                        );
                    }

                    // Apply memory context boost to confidence
                    *prediction_confidence_ref =
                        (*prediction_confidence_ref + memory_context_boost).clamp(0.0, 1.0);

                    // Primitive-Belief Bridge: map primitives to beliefs, compute TD signals
                    let prim_state = CognitiveLoopService::build_primitive_state(
                        &encoding_result.detected_primitives,
                        pp_smoothed_coh,
                        pp_total_cycles as f64,
                    );

                    if let Some(ref prev_state) = prev_primitive_state {
                        let pred_error = primitive_belief_bridge
                            .compute_prediction_error(prev_state, &prim_state);
                        let td_signal = primitive_belief_bridge.td_error_signal(&pred_error);
                        *fep_learning_signal += td_signal as f32 * 0.2;
                        *fep_learning_signal = fep_learning_signal.clamp(-1.0, 1.0);
                    }

                    *prev_primitive_state = Some(prim_state);

                    // Closed learning loop: update Q-values from cycle results
                    closed_learning_loop.update(cycle_learning_result);
                },
            );
        } // end parallel scope -- disjoint borrows released

        // Update semantic memory stats after parallel join completes
        self.stats.semantic_hits = self.semantic_memory.stats().semantic_hits;
        self.stats.semantic_misses = self.semantic_memory.stats().semantic_misses;
        self.stats.semantic_lr_factor = semantic_lr_factor;
        self.stats.semantic_avg_retrieved_error = self.semantic_memory.stats().avg_retrieved_error;
        self.stats.semantic_entries_stored = self.semantic_memory.stats().total_stored;

        // ═══════════════════════════════════════════════════════════════════════
        // DEMAND-DRIVEN CONSOLIDATION TRIGGERS
        // ═══════════════════════════════════════════════════════════════════════
        // Trigger early episodic replay when:
        //   (a) prediction error spikes >2x the moving average, or
        //   (b) semantic memory returned zero hits (retrieval miss)
        // The periodic 100-cycle floor is still enforced by should_replay().
        if let Some(ref mut replay) = self.phi_episodic_replay {
            let avg_err = self.stats.avg_prediction_error;
            let error_spike = avg_err > 0.01 && prediction_error > avg_err * 2.0;
            let semantic_miss = self.semantic_memory.stats().semantic_misses > 0
                && recalled_memories.is_empty()
                && self.stats.total_cycles > 10;

            if error_spike || semantic_miss {
                replay.trigger_demand_replay();
                tracing::trace!(
                    error_spike,
                    semantic_miss,
                    cycle = self.stats.total_cycles,
                    "Demand-driven consolidation triggered"
                );
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // SEQUENTIAL: Episodic replay + Memory coordinator
        // ═══════════════════════════════════════════════════════════════════════
        // These remain sequential because:
        // - Episodic replay needs &mut temporal_network for CfC retraining
        // - Memory coordinator needs &mut phi_episodic_replay after replay completes
        if let Some(ref mut replay) = self.phi_episodic_replay {
            let coherence_summary = self.coherence_bridge.summary();
            let current_phi = coherence_summary.smoothed_coherence as f64;

            let input_hv =
                symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(compressed_state.clone());
            let output_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(output.clone());

            let episode = crate::memory::episodic_replay::Episode::with_metadata(
                input_hv,
                output_hv,
                current_phi,
                self.stats.total_cycles as u64,
                prediction_error,
                self.emotion_contagion.smoothed_valence(),
                coherence_summary.coherence,
            );

            let stored = replay.store_if_significant(episode);
            if stored {
                tracing::trace!(
                    phi = current_phi,
                    cycle = self.stats.total_cycles,
                    "High-Phi episode stored for replay"
                );
            }

            if replay.should_replay() {
                if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                    let learning_rate = self.config.cfc_config.learning_rate;
                    let result = replay.replay_session(cfc, learning_rate);

                    if !result.skipped {
                        tracing::debug!(
                            episodes = result.episodes_replayed,
                            avg_loss = result.average_loss,
                            avg_phi = result.average_phi,
                            "Episodic replay session completed"
                        );
                    }
                }
            }
        }

        // Memory coordinator: broadcast signals and process graduations
        {
            let coord_phi = self.coherence_bridge.smoothed_coherence() as f64;
            let coord_coherence = coherence as f64;
            self.memory_coordinator
                .update_signals(coord_phi, coord_coherence);

            if let Some(ref mut replay) = self.phi_episodic_replay {
                let graduated = self.memory_coordinator.process_graduations(replay);
                if graduated > 0 {
                    tracing::debug!(
                        graduated,
                        "Memory coordinator graduated items to episodic storage"
                    );
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PREFRONTAL CORTEX: Executive control and working memory gating
        // ═══════════════════════════════════════════════════════════════════════
        let prefrontal_veto = if let Some(ref mut pfc) = self.prefrontal {
            // Add current input as a working memory item
            let wm_item = crate::brain::prefrontal::WorkingMemoryItem::new(
                format!("cycle_{}", self.stats.total_cycles),
                symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(
                    compressed_state.clone(),
                ),
            );
            pfc.add_to_memory(wm_item);

            // Advance time (decay activations, evict expired items)
            pfc.tick();

            // Check memory utilization — high utilization triggers inhibition
            let utilization =
                pfc.memory_contents().len() as f32 / 7.0; // default capacity
            let veto = utilization > self.config.learning_threshold.max(0.8);

            if veto {
                tracing::debug!(
                    utilization,
                    cycle = self.stats.total_cycles,
                    "Prefrontal veto: working memory overloaded"
                );
            }

            // Graduate evicted items to memory coordinator
            let graduates = pfc.drain_graduates();
            if !graduates.is_empty() {
                tracing::trace!(
                    count = graduates.len(),
                    "Prefrontal graduated items to episodic memory"
                );
            }

            veto
        } else {
            false
        };

        // Build cycle metadata for observability
        let metadata = super::CycleMetadata {
            surprise_triggered,
            prefrontal_veto,
            reasoning_confidence,
            exploration_action,
        };

        tracing::debug!(
            surprise = metadata.surprise_triggered,
            prefrontal_veto = metadata.prefrontal_veto,
            reasoning_confidence = metadata.reasoning_confidence,
            exploration = ?metadata.exploration_action,
            "Cycle metadata"
        );

        CycleResult {
            output: output.clone(),
            prediction_error,
            attention_state: encoding_result.attention_snapshot,
            detected_primitives: encoding_result.detected_primitives,
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            #[cfg(feature = "identity")]
            signed_output: self.mfdi_bridge.sign_output(output.clone()).ok(),
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
        }
    }
}
