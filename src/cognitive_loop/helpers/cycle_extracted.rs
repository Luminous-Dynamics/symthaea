//! Extracted helpers from cycle() — Phases 1–4.
//!
//! Phase 1 (LOW risk): safety precheck, cognitive depth, negation detection, LR composition
//! Phase 2 (MEDIUM-LOW risk): moral phase, episodic recall, surprise exploration
//! Phase 3 (MEDIUM risk): Psi synthesis, reward signal, strategy modulation
//! Phase 4 (MEDIUM-HIGH risk): FEP active inference, cross-modal binding

use std::time::Instant;

use crate::consciousness::cross_modal_binding::{ModalRepresentation, Modality};
use crate::consciousness::fep_active_inference::Observation;

use super::super::{CognitiveLoopService, CycleResult, MoralJudgmentSummary, ResponseStrategy};

// ═══════════════════════════════════════════════════════════════════════════════
// Tuning constants — see `thresholds.rs` for the centralized registry.
// ═══════════════════════════════════════════════════════════════════════════════
use crate::cognitive_loop::thresholds::{
    BODY_PSI_WEIGHT,
    EMBODIED_PSI_WEIGHT,
    // Psi synthesis
    FLOW_PSI_WEIGHT,
    // FEP / Surprise
    FEP_LR_DECAY,
    FEP_SURPRISE_SCALE,
    MEMORY_CONTEXT_BOOST_SCALE,
    MEMORY_RECALL_SIM_THRESHOLD,
    // Memory recall
    MEMORY_RECALL_TOP_K,
    // Moral evaluation
    MORAL_BENEFIT_CONFIDENCE_BOOST,
    MORAL_BENEFIT_THRESHOLD,
    MORAL_CONCERN_EXPLORATION_DAMPEN,
    MORAL_CONCERN_PAUSE_BOOST,
    MORAL_CONCERN_THRESHOLD,
    MORAL_EVAL_INTERVAL,
    NEGATION_DAMPENING,
    NEGATION_POLARITY_THRESHOLD,
    RELATIONAL_PSI_WEIGHT,
    REWARD_BAD_BASE,
    REWARD_BAD_SCALE,
    REWARD_EXTERNAL_BLEND,
    // Reward computation
    REWARD_GOOD_BASE,
    REWARD_GOOD_CONFIDENCE_SCALE,
    REWARD_MID_BASE,
    REWARD_MID_SCALE,
    STRATEGY_CLARIFYING_FACTOR,
    STRATEGY_CONCISE_SPEECH_RATE,
    STRATEGY_DETAILED_SENSITIVITY,
    // Strategy modulation
    STRATEGY_EXPLORATORY_FACTOR,
    STRATEGY_SUPPORTIVE_PAUSE,
    // Surprise & exploration
    SURPRISE_BOREDOM_DAMPEN,
};

impl CognitiveLoopService {
    // ═══════════════════════════════════════════════════════════════════════
    // Phase 1 (LOW risk, &self / &mut self)
    // ═══════════════════════════════════════════════════════════════════════

    /// Safety pre-check: fast amygdala veto before expensive encoding.
    ///
    /// Returns `Some(CycleResult)` with a safe default response if the safety
    /// gateway blocks the input, or `None` if the input is allowed.
    pub(in crate::cognitive_loop) fn safety_precheck(
        &mut self,
        input: &str,
        cycle_start: Instant,
    ) -> Option<CycleResult> {
        let gateway = self.safety_gateway.as_mut()?;
        let decision = gateway.check(crate::safety::SafetyCheck::Query(input));
        if decision.allowed {
            return None;
        }
        let metadata = super::super::CycleMetadata {
            safety_blocked: true,
            safety_category: decision.category.map(|c| format!("{c:?}")),
            urgency: self.carryover.urgency.urgency,
            ..Default::default()
        };
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
            thought_vector: vec![0.0; 32],
            wisdom_hv: symthaea_core::hdc::BinaryHV([0u8; 2048]),
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
    pub(in crate::cognitive_loop) fn update_cognitive_depth(&mut self) {
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
    pub(in crate::cognitive_loop) fn detect_negation_polarity(&self, input: &str) -> f32 {
        if let Some(ref detector) = self.negation_detector {
            detector
                .get_polarity(input, "harmful")
                .max(detector.get_polarity(input, "dangerous"))
                .max(detector.get_polarity(input, "unethical"))
        } else {
            0.0
        }
    }

    /// Compose the effective learning rate from all modulation sources.
    ///
    /// Combines: base coherence LR -> adaptive behavior -> flow state -> semantic
    /// context -> reasoning reliability -> curiosity novelty -> FEP boost ->
    /// MCE consciousness boost -> subsystem LR factor (from previous cycle).
    ///
    /// Resets `carryover.learning.subsystem_lr_factor` for the next cycle's
    /// accumulation. Hard-capped to [0.0, 0.01].
    pub(in crate::cognitive_loop) fn compose_effective_lr(
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
    // Phase 2 (MEDIUM-LOW risk)
    // ═══════════════════════════════════════════════════════════════════════

    /// Run the full moral evaluation phase: throttle, evaluate, apply negation,
    /// contextual weights, and value feedback.
    ///
    /// Returns `(moral_score, moral_concern_detected, moral_judgment)`.
    /// The judgment is cached in `self.last_moral_judgment` for throttled reuse.
    pub(in crate::cognitive_loop) fn run_moral_phase(
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
        let value_trend = self.primitive_tier.value_feedback.recent_trend(50);
        let moral_feedback = 1.0 + (value_trend * 0.1).clamp(-0.1, 0.1);
        let moral_score = moral_score * moral_feedback;
        {
            let signal = self.primitive_tier.value_feedback.create_signal(
                input,
                crate::consciousness::value_feedback_loop::FeedbackType::SelfAssessment,
                moral_score,
            );
            self.primitive_tier.value_feedback.process_feedback(signal);
        }

        (moral_score, moral_concern_detected, moral_judgment)
    }

    /// Recall episodic memories and apply emotional/consciousness priming.
    ///
    /// Returns the memory context boost (confidence contribution from recalled
    /// memories). Side effects: biases emotional valence and prediction confidence
    /// from recalled episode metadata (Damasio 1999).
    pub(in crate::cognitive_loop) fn recall_episodic_context(
        &mut self,
        compressed_state: &[f32],
    ) -> f32 {
        let hdv_sample: Vec<f32> = compressed_state[..64.min(compressed_state.len())].to_vec();
        let recalled_memories = self.episodic_memory.recall(
            &hdv_sample,
            MEMORY_RECALL_TOP_K,
            MEMORY_RECALL_SIM_THRESHOLD,
        );

        let memory_context_boost = if !recalled_memories.is_empty() {
            recalled_memories.iter().map(|(_, sim)| sim).sum::<f32>()
                / recalled_memories.len().max(1) as f32
                * MEMORY_CONTEXT_BOOST_SCALE
        } else {
            0.0
        };

        // Extract rich context from recalled memories (valence + Phi at encoding time)
        // Science: Damasio (1999) — emotional re-experiencing from recalled episodes
        if !recalled_memories.is_empty() {
            let n = recalled_memories.len() as f32;
            let memory_valence_avg: f32 = recalled_memories
                .iter()
                .map(|(m, _)| m.valence)
                .sum::<f32>()
                / n;
            let memory_phi_avg: f32 = recalled_memories
                .iter()
                .map(|(m, _)| m.phi_at_encoding)
                .sum::<f32>()
                / n;

            // Memory valence biases current emotional state (emotional re-experiencing)
            if memory_valence_avg.abs() > 0.1 {
                let valence_nudge = memory_valence_avg * 0.15;
                self.emotion_contagion.valence =
                    (self.emotion_contagion.valence + valence_nudge).clamp(-1.0, 1.0);
            }
            // Memory Phi primes consciousness expectation
            if memory_phi_avg > 0.4 {
                self.adjust_confidence("memory_phi_prime", (memory_phi_avg - 0.4) * 0.05);
            }
        }

        memory_context_boost
    }

    /// Run the surprise exploration bridge cycle.
    ///
    /// Returns `(surprise_triggered, exploration_action)`. Side effects: adjusts
    /// boredom threshold and exploration urge when surprise is detected.
    pub(in crate::cognitive_loop) fn run_surprise_exploration(
        &mut self,
        compressed_state: &[f32],
    ) -> (bool, Option<String>) {
        let mut surprise_triggered = false;
        let mut exploration_action = None;

        let mut deferred_exploration_delta: Option<f32> = None;
        if let Some(ref mut bridge) = self.surprise_bridge {
            let predicted = self.last_prediction.as_deref().unwrap_or(&[]);
            let actual_len = predicted.len().max(1).min(compressed_state.len());
            let actual = &compressed_state[..actual_len];
            let current_state = self.last_state.as_deref().unwrap_or(compressed_state);
            let (surprise, should_explore, action) = bridge.cycle(predicted, actual, current_state);

            if should_explore {
                surprise_triggered = true;
                let current_threshold = self.curiosity_drive.get_boredom_threshold();
                self.curiosity_drive
                    .set_boredom_threshold(current_threshold * SURPRISE_BOREDOM_DAMPEN);
                let expl_factor = bridge.exploration_factor;
                deferred_exploration_delta = Some(expl_factor * 0.3);
                exploration_action = action.map(|a| {
                    format!(
                        "perturbation[{}d,scale={:.3}]",
                        a.len(),
                        expl_factor
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
        // Apply exploration adjustment after releasing the surprise_bridge borrow
        if let Some(delta) = deferred_exploration_delta {
            self.adjust_exploration("surprise_bridge", delta);
        }

        (surprise_triggered, exploration_action)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 3 (MEDIUM risk)
    // ═══════════════════════════════════════════════════════════════════════

    /// Compute unified Psi (Layer 1 consciousness estimate) from subsystem contributions.
    ///
    /// Combines: temporal coherence + voice quality + flow state + relational
    /// dyad + interoceptive body + embodied cognition. Clamps to [0.0, 1.0].
    /// Updates the unification engine with the result.
    pub(in crate::cognitive_loop) fn compute_unified_psi(&mut self) -> f64 {
        let coherence_psi = self.coherence_bridge.phi_contribution();
        let voice_psi = self.voice_feedback_bridge.summary().phi_adjustment;
        let flow_psi = if self.flow_state.in_flow {
            self.flow_state.intensity * FLOW_PSI_WEIGHT
        } else {
            0.0
        };
        let relational_psi_contrib = if self.social.relational_psi > 0.0 {
            self.social.relational_psi as f32 * RELATIONAL_PSI_WEIGHT
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
    pub(in crate::cognitive_loop) fn compute_reward_signal(
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

        let cycle_reward = if self.social.external_reward.abs() > f32::EPSILON {
            let blended = enriched_reward * REWARD_EXTERNAL_BLEND
                + self.social.external_reward * REWARD_EXTERNAL_BLEND;
            self.social.external_reward = 0.0; // consume
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
    pub(in crate::cognitive_loop) fn apply_strategy_modulation(
        &mut self,
        strategy: ResponseStrategy,
    ) {
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
    pub(in crate::cognitive_loop) fn reapply_strategy_modulation(
        &mut self,
        strategy: ResponseStrategy,
    ) {
        match strategy {
            ResponseStrategy::Exploratory => {
                self.adaptive_behavior.exploration_factor = self
                    .adaptive_behavior
                    .exploration_factor
                    .max(STRATEGY_EXPLORATORY_FACTOR);
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
                self.adaptive_behavior.exploration_factor = self
                    .adaptive_behavior
                    .exploration_factor
                    .min(STRATEGY_CLARIFYING_FACTOR);
            }
            ResponseStrategy::Supportive => {
                self.adaptive_behavior.pause_multiplier = self
                    .adaptive_behavior
                    .pause_multiplier
                    .max(STRATEGY_SUPPORTIVE_PAUSE);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 4 (MEDIUM-HIGH risk)
    // ═══════════════════════════════════════════════════════════════════════════

    /// FEP active inference perception-action loop.
    ///
    /// Constructs an observation from current cognitive state, runs the FEP agent's
    /// perceive->select_action->act pipeline, then applies action-specific modulations
    /// (learning rate boost, sensory precision reset, exploration nudge, trust tightening).
    ///
    /// Returns (action_index, action_probabilities, is_surprised, pragmatic_value).
    pub(in crate::cognitive_loop) fn step_fep_active_inference(
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
                    self.scale_lr("fep_free_energy", 1.0 + fe_boost * 0.5);
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
                self.adjust_exploration("perturbation", nudge);
            }
            3 => {
                // Tighten trust via precision
                if let Some(ref fe) = self.fep_agent.last_fe_components {
                    let precision_mod = (1.0 - fe.prediction_error).clamp(0.0, 1.0) as f32;
                    self.self_model_tier.self_reflection.trust_threshold =
                        (self.self_model_tier.self_reflection.trust_threshold * 0.9 + precision_mod * 0.1)
                            .clamp(0.1, 0.9);
                }
            }
            _ => {}
        }

        (
            fep_action_idx,
            fep_action_probs,
            is_surprised,
            action_result.pragmatic_value,
        )
    }

    /// Cross-modal binding: bind HDC encodings across linguistic and affective modalities.
    ///
    /// Clears stale representations, adds linguistic (from BinaryHV) and affective
    /// (if bridge enabled) modalities, computes binding strength and cross-modal Psi.
    /// Also applies feedback loops: high Psi boosts prediction confidence, high binding
    /// boosts predictive precision, high free energy dampens binding attention.
    ///
    /// Returns (binding_strength, cross_modal_psi).
    pub(in crate::cognitive_loop) fn update_cross_modal_binding(
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
                    let affective_hv = symthaea_core::hdc::binary_hv::BinaryHV::random(affect_seed);
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
        // Science: Treisman (1996) — coherent binding -> confident perception
        if cross_modal_psi > 0.3 {
            let boost = ((cross_modal_psi - 0.3) * 0.05) as f32;
            self.adjust_confidence("cross_modal_psi", boost);
        }

        // FEEDBACK: Predictive <-> Cross-Modal bidirectional coupling (Talsma 2015)
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

    /// Meta-Forge: Simulate historical episodes with candidate parameters.
    ///
    /// Replays a batch of episodes using the current network state but with
    /// candidate hyper-parameters (e.g. tau scaling). Returns the average Phi
    /// achieved during the simulation.
    pub fn simulate_episodes(
        &mut self,
        episodes: &[crate::memory::episodic_replay::Episode],
        tau_scale: f32,
    ) -> f64 {
        // 1. Snapshot current tau
        let original_tau = self.temporal_network.all_tau_owned();

        // 2. Apply candidate scaling
        self.temporal_network.scale_tau_all(tau_scale);

        // 3. Replay and measure
        let mut total_phi = 0.0;
        for ep in episodes {
            let input_array = ndarray::Array1::from_vec(ep.input.values.clone());
            let _ = self.temporal_network.step(&input_array, 0.02); // Standard dt

            // Measure integration (Layer 1 proxy)
            let phi = match &self.temporal_network {
                super::super::temporal_network::TemporalNetwork::CfC(cfc) => {
                    cfc.consciousness_level() as f64
                }
                _ => 0.5,
            };
            total_phi += phi;
        }

        // 4. Restore original tau
        self.temporal_network.set_tau_all(original_tau);

        total_phi / episodes.len().max(1) as f64
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Arousal/Seizure management
    // ═══════════════════════════════════════════════════════════════════════

    /// Arousal trap state machine: detect, recover, and escape high-arousal traps.
    ///
    /// Three-phase approach based on Yerkes-Dodson (1908) and Porges (2011):
    /// - Phase 1 (detect): increment counter when arousal > 0.8
    /// - Phase 2 (recover, cycles 5–10): gradual LR dampening + exploration boost
    /// - Phase 3 (escape, cycles >10): forced exploration=1.0, confidence×0.9, reset
    /// - Low arousal (<0.3): consolidation LR boost (Steriade 1996)
    ///
    /// Note: 4 emergency exploration bypasses elsewhere are intentional (direct mutations).
    pub(in crate::cognitive_loop) fn manage_arousal_trap(&mut self, affective_arousal: f32) {
        if affective_arousal > 0.7 {
            // Attenuated 50%: DA learning_rate_factor() already scales LR via the bath
            let arousal_suppress = ((affective_arousal - 0.7) * 0.25).min(0.08);
            self.scale_lr("affective_arousal_suppress", 1.0 - arousal_suppress);

            // Arousal trap detection (Yerkes-Dodson 1908 — inverted-U performance curve)
            if affective_arousal > 0.8 {
                self.carryover.urgency.arousal_trap_counter = self
                    .carryover
                    .urgency
                    .arousal_trap_counter
                    .saturating_add(1);
            }
            // Phase 2: Active arousal recovery mode (Porges 2011 polyvagal theory)
            if self.carryover.urgency.arousal_trap_counter > 5
                && self.carryover.urgency.arousal_trap_counter <= 10
            {
                let recovery_intensity =
                    (self.carryover.urgency.arousal_trap_counter - 5) as f32 / 5.0;
                // Gradual LR dampening: attenuated 50% (NE exploration_delta handles arousal)
                self.scale_lr("arousal_trap_recovery", 1.0 - recovery_intensity * 0.1);
                // Slight exploration boost: attenuated 50% (NE exploration_delta covers this)
                self.adjust_exploration("arousal_trap_recovery", recovery_intensity * 0.025);
                self.stats.arousal_recovery_cycles += 1;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    counter = self.carryover.urgency.arousal_trap_counter,
                    recovery_intensity,
                    "Arousal recovery mode: dampening LR, boosting exploration"
                );
            }
            // Phase 3: Forced escape
            if self.carryover.urgency.arousal_trap_counter > 10 {
                self.set_exploration("arousal_trap_escape", 1.0); // forced escape attempt
                self.scale_confidence("arousal_trap_escape", 0.9);
                self.carryover.urgency.arousal_trap_counter = 0;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Arousal trap escape: forced exploration after 10 high-arousal cycles"
                );
            }
        } else {
            // Reset trap counter when arousal drops below threshold
            self.carryover.urgency.arousal_trap_counter = 0;

            if affective_arousal < 0.3 {
                // Low arousal enhances consolidation (Steriade 1996)
                // Attenuated 50%: DA handles low-error consolidation boost via the bath
                let consolidation_boost = ((0.3 - affective_arousal) * 0.3).min(0.05);
                self.scale_lr("low_arousal_consolidate", 1.0 + consolidation_boost);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 5: Dynamics decomposition helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// Apply moral modulation to active inference and surprise-gated learning.
    ///
    /// Handles consent/harm/duty steering, surprise LR boost/decay, and
    /// predictive free energy surprise amplitude scaling.
    ///
    /// Returns `(moral_steering_category, pfe_surprise_mod)`.
    pub(in crate::cognitive_loop) fn apply_moral_modulation(
        &mut self,
        moral_concern_detected: bool,
        moral_judgment: &MoralJudgmentSummary,
        moral_score: f32,
        is_surprised: bool,
    ) -> (&'static str, f32) {
        let mut moral_steering_category: &str = "";
        if moral_concern_detected {
            self.scale_exploration("moral_concern", MORAL_CONCERN_EXPLORATION_DAMPEN);

            self.self_model_tier.self_reflection.trust_threshold =
                (self.self_model_tier.self_reflection.trust_threshold * 1.2).clamp(0.1, 0.95);

            self.adaptive_behavior.pause_multiplier *= MORAL_CONCERN_PAUSE_BOOST;

            if moral_judgment.consent_violation
                || moral_judgment
                    .violations
                    .iter()
                    .any(|v| v.contains("perfect") || v.contains("harm"))
            {
                self.stats.moral_review_needed = true;
            }

            if moral_judgment.consent_violation {
                self.scale_confidence("moral_consent_viol", 0.7);
                self.carryover.learning.subsystem_lr_factor *= 0.5;
                moral_steering_category = "consent";
            } else if moral_judgment
                .violations
                .iter()
                .any(|v| v.contains("harm"))
            {
                self.scale_exploration("harm_detected", 0.4);
                self.scale_confidence("moral_harm_detect", 0.85);
                moral_steering_category = "harm";
            } else if moral_judgment
                .violations
                .iter()
                .any(|v| v.contains("perfect") || v.contains("duty"))
            {
                self.self_model_tier.self_reflection.force_reflection();
                self.carryover.learning.subsystem_lr_factor *= 0.8;
                moral_steering_category = "duty";
            } else if !moral_judgment.violations.is_empty() {
                self.carryover.learning.subsystem_lr_factor *= 0.9;
                moral_steering_category = "other";
            }
        } else if moral_score > MORAL_BENEFIT_THRESHOLD {
            self.scale_confidence("moral_benefit", MORAL_BENEFIT_CONFIDENCE_BOOST);
        }

        // Surprise-gated learning rate boost
        if is_surprised {
            let surprise_boost =
                (self.fep_agent.current_free_energy() as f32 / FEP_SURPRISE_SCALE).clamp(0.1, 0.5);
            self.adjust_lr("fep_surprise", surprise_boost);
        } else {
            self.scale_lr("fep_decay", FEP_LR_DECAY);
        }

        // Phase 21: Predictive free energy → surprise amplitude scaling
        let cached_pfe = self.carryover.consciousness.last_predictive_free_energy;
        let pfe_surprise_mod = if is_surprised && cached_pfe > 0.5 {
            let amplification = ((cached_pfe - 0.5) * 0.2).min(0.1) as f32;
            self.adjust_exploration("pfe_surprise_amplify", amplification);
            self.stats.pfe_surprise_mod_count += 1;
            amplification
        } else if is_surprised && cached_pfe < 0.2 && cached_pfe > 0.0 {
            let dampening = ((0.2 - cached_pfe) * 0.15).min(0.05) as f32;
            self.scale_exploration("pfe_surprise_dampen", 1.0 - dampening);
            self.stats.pfe_surprise_mod_count += 1;
            -dampening
        } else {
            0.0
        };

        (moral_steering_category, pfe_surprise_mod)
    }

    /// Update flow state, curiosity drive, and self-reflection.
    ///
    /// Applies meta-cognitive recommendations (LR ±10%, exploration ±) when
    /// self-reflection confidence > 0.5.
    pub(in crate::cognitive_loop) fn update_drives_and_reflection(
        &mut self,
        pattern: crate::dynamics::temporal_signatures::ConsciousnessPattern,
        prediction_error: f32,
        coherence: f32,
    ) {
        let adapted_thresholds = self.self_model_tier.self_reflection.get_thresholds();
        self.flow_state.update_with_thresholds(
            pattern,
            prediction_error,
            coherence,
            self.prediction_confidence,
            adapted_thresholds.flow_error,
            adapted_thresholds.flow_coherence,
        );

        // Curiosity drive — route exploration mutation through feedback system
        self.curiosity_drive
            .set_boredom_threshold(adapted_thresholds.boredom);
        match self.curiosity_drive.update(prediction_error) {
            super::super::drives::ExplorationUpdate::Set(val) => {
                self.set_exploration("curiosity_drive_boredom", val);
            }
            super::super::drives::ExplorationUpdate::Scale(factor) => {
                self.scale_exploration("curiosity_drive_decay", factor);
            }
        }

        // Self-reflection
        self.self_model_tier.self_reflection.record_cycle(
            prediction_error,
            self.flow_state.in_flow,
            self.curiosity_drive.should_explore(),
            self.prediction_confidence,
        );
        if self.self_model_tier.self_reflection.should_reflect() {
            let recommendations = self.self_model_tier.self_reflection.reflect();
            for rec in &recommendations {
                if rec.confidence < 0.5 {
                    continue;
                }
                match rec.target {
                    super::super::RecommendationTarget::LearningRate => match rec.direction {
                        super::super::AdjustmentDirection::Decrease => {
                            self.scale_lr("reflection_decrease", 0.9);
                        }
                        super::super::AdjustmentDirection::Increase => {
                            self.scale_lr("reflection_increase", 1.1);
                        }
                        _ => {}
                    },
                    super::super::RecommendationTarget::ExplorationFactor => match rec.direction {
                        super::super::AdjustmentDirection::Increase => {
                            self.adjust_exploration("metacog_explore_increase", 0.12);
                        }
                        super::super::AdjustmentDirection::Decrease => {
                            self.scale_exploration("metacog_explore_decrease", 0.75);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
}
