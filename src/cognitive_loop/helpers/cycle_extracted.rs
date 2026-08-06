// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use crate::cognitive_loop::feedback_state::Priority;

/// Geometric mean of positive f32 values. Returns 1.0 for empty/invalid input.
fn geometric_mean(factors: &[f32]) -> f32 {
    let mut log_sum = 0.0f32;
    let mut count = 0u32;
    for &f in factors {
        if f > 0.0 && f.is_finite() {
            log_sum += f.ln();
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    let mean = (log_sum / count as f32).exp();
    if mean.is_finite() { mean } else { 1.0 }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tuning constants — see `thresholds.rs` for the centralized registry.
// ═══════════════════════════════════════════════════════════════════════════════
use crate::cognitive_loop::thresholds::{
    // Arousal management
    AROUSAL_SUPPRESS_MAX,
    AROUSAL_SUPPRESS_SCALE,
    AROUSAL_SUPPRESS_THRESHOLD,
    AROUSAL_TRAP_THRESHOLD,
    BASELINE_INTEGRATION_SCALE,
    BODY_PSI_WEIGHT,
    // Cross-modal binding
    CROSS_MODAL_FE_DAMPEN_MIN,
    CROSS_MODAL_FE_DAMPEN_SCALE,
    CROSS_MODAL_FE_DAMPEN_THRESHOLD,
    CROSS_MODAL_LINGUISTIC_CONFIDENCE,
    CROSS_MODAL_PRECISION_BOOST_SCALE,
    CROSS_MODAL_PRECISION_BOOST_THRESHOLD,
    EMBODIED_PSI_WEIGHT,
    // FEP actions
    FEP_ACTION_FE_BOOST_SCALE,
    FEP_ACTION_FE_DIVISOR,
    FEP_EXPLORATION_NUDGE_CALM,
    FEP_EXPLORATION_NUDGE_SURPRISED,
    // FEP / Surprise
    FEP_LR_DECAY,
    FEP_REWARD_WEIGHT,
    FEP_SENSORY_PRECISION_BLEND,
    FEP_SURPRISE_SCALE,
    // Psi synthesis
    FLOW_PSI_WEIGHT,
    // Low arousal consolidation
    LOW_AROUSAL_CONSOLIDATION_MAX,
    LOW_AROUSAL_CONSOLIDATION_SCALE,
    LOW_AROUSAL_CONSOLIDATION_THRESHOLD,
    MEMORY_CONTEXT_BOOST_SCALE,
    // Memory / episodic
    MEMORY_PHI_PRIME_SCALE,
    MEMORY_PHI_PRIME_THRESHOLD,
    MEMORY_RECALL_SIM_THRESHOLD,
    // Memory recall
    MEMORY_RECALL_TOP_K,
    MEMORY_VALENCE_NUDGE_SCALE,
    MEMORY_VALENCE_THRESHOLD,
    // Moral evaluation
    MORAL_BENEFIT_CONFIDENCE_BOOST,
    MORAL_BENEFIT_THRESHOLD,
    MORAL_CONCERN_EXPLORATION_DAMPEN,
    MORAL_CONCERN_PAUSE_BOOST,
    MORAL_CONCERN_THRESHOLD,
    MORAL_CONSENT_CONFIDENCE_SCALE,
    MORAL_CONSENT_LR_FACTOR,
    MORAL_DUTY_LR_FACTOR,
    MORAL_EVAL_INTERVAL,
    MORAL_HARM_CONFIDENCE_SCALE,
    MORAL_HARM_EXPLORATION_SCALE,
    MORAL_OTHER_LR_FACTOR,
    MORAL_VALUE_FEEDBACK_SCALE,
    NEGATION_DAMPENING,
    NEGATION_POLARITY_THRESHOLD,
    // PFE surprise
    PFE_SURPRISE_AMPLIFY_MAX,
    PFE_SURPRISE_AMPLIFY_SCALE,
    PFE_SURPRISE_DAMPEN_MAX,
    PFE_SURPRISE_DAMPEN_SCALE,
    PFE_SURPRISE_HIGH_THRESHOLD,
    PFE_SURPRISE_LOW_THRESHOLD,
    PHYSICS_EXPLOIT_SCALE,
    // Physics exploration
    PHYSICS_EXPLOIT_THRESHOLD,
    PHYSICS_EXPLORE_SCALE,
    PHYSICS_EXPLORE_THRESHOLD,
    PSI_PHI_COUPLING_WEIGHT,
    // Self-reflection
    REFLECTION_CONFIDENCE_THRESHOLD,
    REFLECTION_EXPLORATION_DECREASE,
    REFLECTION_EXPLORATION_INCREASE,
    REFLECTION_LR_DECREASE,
    REFLECTION_LR_INCREASE,
    RELATIONAL_PSI_WEIGHT,
    REWARD_BAD_BASE,
    REWARD_BAD_SCALE,
    REWARD_EXTERNAL_BLEND,
    // Reward computation
    REWARD_GOOD_BASE,
    REWARD_GOOD_CONFIDENCE_SCALE,
    // Reward boundary
    REWARD_HIGH_ERROR_THRESHOLD,
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
    SURPRISE_EXPLORATION_FACTOR_SCALE,
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
            bits_saved_persist: None,
            bits_saved_zero: None,
            bits_kappa: None,
            recall_fired: false,
            recall_similarity: None,
            recall_matched_timestamp: None,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector: vec![0.0; 32],
            wisdom_hv: symthaea_core::hdc::BinaryHV([0u8; 2048]),
            language_output: None,
            language_source: None,
            #[cfg(feature = "canvas")]
            canvas_svg: None,
            #[cfg(feature = "identity")]
            signed_output: None,
            #[cfg(feature = "identity")]
            assurance_level: crate::identity::AssuranceLevel::E0Anonymous,
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
        })
    }
    /// Thalamic routing: determine cognitive depth from prior state.
    ///
    /// Uses temporal signature pattern, prosody valence, and average prediction
    /// error to select Reflex / Cortical / DeepThought processing depth.
    pub(in crate::cognitive_loop) fn update_cognitive_depth(&mut self) {
        let prior_pattern = self
            .language_comm
            .voice_coherence
            .temporal
            .classify_state()
            .0;
        let prior_valence = self.unification_engine.emotional.state().valence as f32;
        let prior_error = self.stats.avg_prediction_error;
        self.cognitive_depth = self.behavior.thalamic_router.route_from_cycle(
            prior_error,
            prior_pattern,
            prior_valence,
        );
    }

    /// Detect negation polarity across safety-critical terms.
    ///
    /// Returns the max polarity score across "harmful", "dangerous", "unethical".
    /// Returns 0.0 if no negation detector is configured.
    /// Science: Wason (1959) — negation processing in human reasoning.
    pub(in crate::cognitive_loop) fn detect_negation_polarity(&self, input: &str) -> f32 {
        if let Some(ref detector) = self.ethics_values.negation_detector {
            detector
                .get_polarity(input, "harmful")
                .max(detector.get_polarity(input, "dangerous"))
                .max(detector.get_polarity(input, "unethical"))
        } else {
            0.0
        }
    }

    /// Compose the effective learning rate using 3-group geometric mean structure.
    ///
    /// **Base LR**: combined_learning_rate() -> adaptive behavior.
    /// **Cognitive mod** (geomean): flow, semantic, reasoning, curiosity.
    /// **Meta-learning mod** (geomean): FEP boost, MCE boost, subsystem LR factor.
    /// Final: `base x cognitive_mod x meta_mod`, clamped [0, 0.01].
    pub fn compose_effective_lr(
        &mut self,
        semantic_lr_factor: f32,
        reasoning_lr_factor: f32,
    ) -> f32 {
        let base_lr = self.combined_learning_rate();
        let base = self
            .behavior
            .adaptive_behavior
            .effective_learning_rate(base_lr);

        let flow_factor = self.behavior.flow_state.learning_boost;
        let curiosity_factor = self.behavior.curiosity_drive.novelty_bonus;
        let cognitive_mod = geometric_mean(&[
            flow_factor.max(0.01),
            semantic_lr_factor.max(0.01),
            reasoning_lr_factor.max(0.01),
            curiosity_factor.max(0.01),
        ]);

        let fep_factor = self.fep.lr_boost as f32;
        let mce_factor = 1.0 + self.carryover.learning.mce_lr_boost;
        let subsystem_factor = self.carryover.learning.subsystem_lr_factor.clamp(0.5, 2.0);
        self.carryover.learning.subsystem_lr_factor = 1.0;
        let meta_mod = geometric_mean(&[
            fep_factor.max(0.01),
            mce_factor.max(0.01),
            subsystem_factor.max(0.01),
        ]);

        self.carryover.learning.lr_cognitive_mod = cognitive_mod;
        self.carryover.learning.lr_meta_mod = meta_mod;

        (base * cognitive_mod * meta_mod).clamp(0.0, 0.01)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 2 (MEDIUM-LOW risk)
    // ═══════════════════════════════════════════════════════════════════════

    /// Run the full moral evaluation phase: throttle, evaluate, apply negation,
    /// contextual weights, and value feedback.
    ///
    /// Returns `(moral_score, moral_concern_detected, moral_judgment)`.
    /// The judgment is cached in `self.ethics_values.last_moral_judgment` for throttled reuse.
    /// Returns (moral_score, concern_detected, judgment, affect_coords, fluctuatio_tension, is_ambiguous, epistemic_confidence).
    #[allow(clippy::type_complexity)]
    pub(in crate::cognitive_loop) fn run_moral_phase(
        &mut self,
        input: &str,
        input_negation_polarity: f32,
    ) -> (f32, bool, MoralJudgmentSummary, [f32; 18], f32, bool, f32) {
        // Throttled evaluation: re-evaluate on interval or new input
        let moral_judgment = if self.stats.total_cycles % MORAL_EVAL_INTERVAL == 1
            || self
                .ethics_values
                .last_moral_judgment
                .as_ref()
                .map_or(true, |j| j.input != input)
        {
            let j = self.evaluate_moral_alignment(input);
            self.stats.moral_evaluations += 1;
            j
        } else {
            // The map_or guard above ensures last_moral_judgment is Some here,
            // but we use unwrap_or_else for panic-freedom in the 50Hz hot path.
            match self.ethics_values.last_moral_judgment.clone() {
                Some(j) => j,
                None => {
                    // Should be unreachable — but if it is, re-evaluate rather than panic.
                    let j = self.evaluate_moral_alignment(input);
                    self.stats.moral_evaluations += 1;
                    j
                }
            }
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
        let contextual_weight_factor =
            if let Some(ref mut cw) = self.ethics_values.contextual_weights {
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
        let moral_feedback = 1.0
            + (value_trend * MORAL_VALUE_FEEDBACK_SCALE)
                .clamp(-MORAL_VALUE_FEEDBACK_SCALE, MORAL_VALUE_FEEDBACK_SCALE);
        let moral_score = moral_score * moral_feedback;
        {
            let signal = self.primitive_tier.value_feedback.create_signal(
                input,
                crate::consciousness::value_feedback_loop::FeedbackType::SelfAssessment,
                moral_score,
            );
            self.primitive_tier.value_feedback.process_feedback(signal);
        }

        // Compute Spinozist affect fingerprint for telemetry.
        // Uses deliberate() for trajectory-aware fingerprinting.
        use crate::hdc::spinozist_geometry::{NUM_AFFECTS, SpinozistClassifier};
        use std::sync::OnceLock;
        static SPINOZIST: OnceLock<SpinozistClassifier> = OnceLock::new();
        let classifier = SPINOZIST.get_or_init(SpinozistClassifier::new);
        let (fingerprint, fluctuatio) = classifier.deliberate(input);
        let mut affect_coords = [0.0f32; 18];
        let n = fingerprint.affect_coords.len().min(18);
        affect_coords[..n].copy_from_slice(&fingerprint.affect_coords[..n]);
        let fluctuatio_tension = fluctuatio.max_tension;
        let is_ambiguous = fluctuatio.is_ambiguous;
        let epistemic_confidence = fingerprint.epistemic_confidence;

        (
            moral_score,
            moral_concern_detected,
            moral_judgment,
            affect_coords,
            fluctuatio_tension,
            is_ambiguous,
            epistemic_confidence,
        )
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
        let recalled_memories = self.fep.episodic_memory.recall(
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
            if memory_valence_avg.abs() > MEMORY_VALENCE_THRESHOLD {
                let valence_nudge = memory_valence_avg * MEMORY_VALENCE_NUDGE_SCALE;
                self.behavior.emotion_contagion.valence =
                    (self.behavior.emotion_contagion.valence + valence_nudge).clamp(-1.0, 1.0);
            }
            // Memory Phi primes consciousness expectation
            if memory_phi_avg > MEMORY_PHI_PRIME_THRESHOLD {
                self.adjust_confidence(
                    "memory_phi_prime",
                    (memory_phi_avg - MEMORY_PHI_PRIME_THRESHOLD) * MEMORY_PHI_PRIME_SCALE,
                );
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
        if let Some(ref mut bridge) = self.fep.surprise_bridge {
            let predicted = self.last_prediction.as_deref().unwrap_or(&[]);
            let actual_len = predicted.len().max(1).min(compressed_state.len());
            let actual = &compressed_state[..actual_len];
            let current_state = self.last_state.as_deref().unwrap_or(compressed_state);
            let (surprise, should_explore, action) = bridge.cycle(predicted, actual, current_state);

            if should_explore {
                surprise_triggered = true;
                let current_threshold = self.behavior.curiosity_drive.get_boredom_threshold();
                self.behavior
                    .curiosity_drive
                    .set_boredom_threshold(current_threshold * SURPRISE_BOREDOM_DAMPEN);
                let expl_factor = bridge.exploration_factor;
                deferred_exploration_delta = Some(expl_factor * SURPRISE_EXPLORATION_FACTOR_SCALE);
                exploration_action =
                    action.map(|a| format!("perturbation[{}d,scale={:.3}]", a.len(), expl_factor));
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
    ///
    /// Ψ vs Φ (decided 2026-07-04): this is a *different* quantity from
    /// `consciousness_level`/Φ (`consciousness_engine/measure.rs`, driven by
    /// `SpectralMIPFinder`'s Fiedler-ordering MIP search). Φ answers "is this system
    /// doing real structural integration right now" and gates motor safety; Ψ answers
    /// "is this system in a good state for social/communicative output" and feeds
    /// `ethics_engine.rs`'s input plus Broca's generation-trigger gate (`broca_psi >
    /// 0.4`). Deliberate split, not merged/cross-validated: a topology-integration
    /// measure isn't obviously the right gate for "should I speak now," and Ψ's
    /// engagement/flow/relational inputs are a closer fit for that question than Φ's
    /// graph-structural one. Counter-argument this doesn't fully resolve: two
    /// uncross-validated "consciousness" numbers risk ethics and motor safety disagreeing
    /// about the same underlying state. `ConsciousnessEquationV2` has an unused
    /// `sigmoid(Φ)*0.7 + Ψ*0.3` blend (`consciousness_engine/measure.rs`) that could
    /// reconcile the two if its cold-start near-zero-output issue is ever fixed.
    pub(in crate::cognitive_loop) fn compute_unified_psi(&mut self) -> f64 {
        let coherence_psi = self.language_comm.voice_coherence.bridge.phi_contribution();
        let voice_psi = self
            .language_comm
            .voice_coherence
            .voice
            .summary()
            .phi_adjustment;
        let flow_psi = if self.behavior.flow_state.in_flow {
            self.behavior.flow_state.intensity * FLOW_PSI_WEIGHT
        } else {
            0.0
        };
        let relational_psi_contrib = if self.behavior.social_mgr.social.relational_psi > 0.0 {
            self.behavior.social_mgr.social.relational_psi as f32 * RELATIONAL_PSI_WEIGHT
        } else {
            0.0
        };
        let body_psi_contrib =
            (self.carryover.consciousness.body_phi_modulation - 1.0) * BODY_PSI_WEIGHT;
        let embodied_psi_contrib =
            (self.carryover.consciousness.embodied_phi_modulation - 1.0) * EMBODIED_PSI_WEIGHT;

        // Baseline: the CfC network IS integrating information from cycle 1.
        // Temporal coherence (smoothed) serves as a proxy for ongoing integration
        // activity — this is the raw coherence, not the doubly-weighted phi_contribution.
        // Science: Tononi (2004) — Φ reflects the degree of information integration
        // in a system. An active recurrent network has non-trivial Φ by design.
        let baseline_integration = self
            .language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence()
            * BASELINE_INTEGRATION_SCALE;

        let component_psi = (baseline_integration
            + coherence_psi
            + voice_psi
            + flow_psi
            + relational_psi_contrib
            + body_psi_contrib as f32
            + embodied_psi_contrib as f32)
            .clamp(0.0, 1.0) as f64;

        // Φ→Ψ coupling (2026-07-16, user-approved design — see
        // PSI_PHI_COUPLING_WEIGHT's doc): in a text-only loop the social
        // components above are structurally inert, so component_psi is a
        // near-constant coherence proxy (~0.51) and Ψ's gates (ethics input,
        // Broca's speak-trigger at 0.4) never fired. Blending in the
        // regime-discriminating consciousness_level gives Ψ real dynamic
        // range. Coupling applies only once Φ has actually been measured:
        // carryover.history.consciousness_level cold-starts at the 0.05
        // consciousness floor, which is a prior, not a measurement — treating
        // it as one would suppress Ψ (and Broca's speech) for the first ~47
        // cycles until spectral Φ first computes (the absent-vs-zero rule).
        let phi_level = self.carryover.history.consciousness_level;
        let unified_psi = if phi_level > 0.05 {
            ((1.0 - PSI_PHI_COUPLING_WEIGHT) * component_psi + PSI_PHI_COUPLING_WEIGHT * phi_level)
                .clamp(0.0, 1.0)
        } else {
            component_psi
        };
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
            REWARD_GOOD_BASE + REWARD_GOOD_CONFIDENCE_SCALE * self.prediction_confidence as f32
        } else if prediction_error > REWARD_HIGH_ERROR_THRESHOLD {
            REWARD_BAD_BASE + REWARD_BAD_SCALE * (prediction_error - REWARD_HIGH_ERROR_THRESHOLD)
        } else {
            REWARD_MID_BASE + REWARD_MID_SCALE * prediction_error
        };

        // FEP free energy reduction enrichment
        // Science: Friston (2010) — free energy minimization as the objective
        // Reward FE reduction (prev > current = improving model)
        let fep_bonus = if let Some(ref fe) = self.fep.agent.last_fe_components {
            let current_fe = fe.total;
            let prev_fe = self.stats.last_total_fe;
            // Update cached value for next cycle
            self.stats.last_total_fe = current_fe;
            if prev_fe > 0.0 {
                // FE reduction: positive when improving (prev > current)
                let fe_reduction = (prev_fe - current_fe).clamp(-1.0, 1.0);
                (fe_reduction * FEP_REWARD_WEIGHT as f64) as f32
            } else {
                0.0 // first cycle — no previous FE
            }
        } else {
            0.0
        };

        let enriched_reward = internal_reward + fep_bonus;

        let cycle_reward = if self.behavior.social_mgr.social.external_reward.abs() > f32::EPSILON {
            let blended = enriched_reward * REWARD_EXTERNAL_BLEND
                + self.behavior.social_mgr.social.external_reward * REWARD_EXTERNAL_BLEND;
            self.behavior.social_mgr.social.external_reward = 0.0; // consume
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
                self.behavior.adaptive_behavior.exploration_factor = STRATEGY_EXPLORATORY_FACTOR;
            }
            ResponseStrategy::Detailed => {
                self.behavior.adaptive_behavior.attention_sensitivity =
                    STRATEGY_DETAILED_SENSITIVITY;
            }
            ResponseStrategy::Concise => {
                self.behavior.adaptive_behavior.speech_rate_multiplier =
                    STRATEGY_CONCISE_SPEECH_RATE;
            }
            ResponseStrategy::Clarifying => {
                self.behavior.adaptive_behavior.exploration_factor = STRATEGY_CLARIFYING_FACTOR;
            }
            ResponseStrategy::Supportive => {
                self.behavior.adaptive_behavior.pause_multiplier = STRATEGY_SUPPORTIVE_PAUSE;
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
                self.behavior.adaptive_behavior.exploration_factor = self
                    .behavior
                    .adaptive_behavior
                    .exploration_factor
                    .max(STRATEGY_EXPLORATORY_FACTOR);
            }
            ResponseStrategy::Detailed => {
                self.behavior.adaptive_behavior.attention_sensitivity *=
                    STRATEGY_DETAILED_SENSITIVITY;
            }
            ResponseStrategy::Concise => {
                self.behavior.adaptive_behavior.speech_rate_multiplier = self
                    .behavior
                    .adaptive_behavior
                    .speech_rate_multiplier
                    .max(STRATEGY_CONCISE_SPEECH_RATE);
            }
            ResponseStrategy::Clarifying => {
                self.behavior.adaptive_behavior.exploration_factor = self
                    .behavior
                    .adaptive_behavior
                    .exploration_factor
                    .min(STRATEGY_CLARIFYING_FACTOR);
            }
            ResponseStrategy::Supportive => {
                self.behavior.adaptive_behavior.pause_multiplier = self
                    .behavior
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
            self.prediction_confidence,
            effective_lr as f64,
        );
        let _perception = self.fep.agent.perceive(&fep_obs);
        let action_result = self.fep.agent.select_action();
        let _outcome = self.fep.agent.act(action_result.action);

        let fep_action_idx = action_result.action;
        let fep_action_probs = action_result.action_probabilities.clone();

        let is_surprised = self.fep.agent.is_surprised();
        match action_result.action {
            0 => {
                // Boost learning rate when free energy is high
                if let Some(ref fe) = self.fep.agent.last_fe_components {
                    let fe_boost = (fe.total.abs() as f32 / FEP_ACTION_FE_DIVISOR).clamp(0.0, 1.5);
                    self.scale_lr_pri(
                        "fep_free_energy",
                        1.0 + fe_boost * FEP_ACTION_FE_BOOST_SCALE,
                        Priority::Homeostatic,
                    );
                }
            }
            1 => {
                // Reset sensory precision toward 1.0 to trust new observations after shift
                let current = self.fep.agent.precision.sensory_precision;
                self.fep.agent.precision.sensory_precision = current
                    * (1.0 - FEP_SENSORY_PRECISION_BLEND as f64)
                    + FEP_SENSORY_PRECISION_BLEND as f64;
            }
            2 => {
                // Boost exploration -- stronger nudge when surprised
                let nudge = if is_surprised {
                    FEP_EXPLORATION_NUDGE_SURPRISED
                } else {
                    FEP_EXPLORATION_NUDGE_CALM
                };
                self.adjust_exploration_pri("perturbation", nudge, Priority::Homeostatic);
            }
            3 => {
                // Tighten trust via precision
                if let Some(ref fe) = self.fep.agent.last_fe_components {
                    let precision_mod = (1.0 - fe.prediction_error).clamp(0.0, 1.0) as f32;
                    self.consciousness
                        .self_model_tier
                        .self_reflection
                        .trust_threshold = (self
                        .consciousness
                        .self_model_tier
                        .self_reflection
                        .trust_threshold
                        * 0.9
                        + precision_mod * 0.1)
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
            if let Some(ref mut binder) = self.consciousness_state.cross_modal_binder {
                use symthaea_core::hdc::unified_hv::ContinuousHV;
                binder.clear();
                let linguistic_repr = ModalRepresentation::new(
                    Modality::Linguistic,
                    ContinuousHV::from_vec(hv16.to_bipolar()),
                    CROSS_MODAL_LINGUISTIC_CONFIDENCE,
                    "encoder",
                );
                binder.add_representation(linguistic_repr);
                if self.consciousness_state.affective_bridge.is_some() {
                    // Map [-1.0, 1.0] → [0, 2000] so negative valences get distinct seeds.
                    // Without this, negative f32→u64 saturates to 0, collapsing all negative
                    // emotions into the same hypervector (Rust 2021+ float-to-uint semantics).
                    let affect_seed = ((affective_valence + 1.0) * 1000.0) as u64;
                    let affective_hv = symthaea_core::hdc::binary_hv::BinaryHV::random(affect_seed);
                    binder.update_modality(Modality::Affective, affective_hv);
                }
                let strength = binder.bind().map(|r| r.strength).unwrap_or(0.0);
                let raw_phi = binder.cross_modal_psi();
                // Temporal smoothing: blend with previous-cycle psi to prevent
                // rapid binding/unbinding oscillations.
                // Engel et al. (2001): cross-modal binding requires sustained synchronization.
                let prev_phi = self.carryover.consciousness.cross_modal_psi;
                let alpha = super::super::thresholds::CROSS_MODAL_PSI_TEMPORAL_SMOOTHING;
                let phi = raw_phi * (1.0 - alpha) + prev_phi * alpha;
                self.carryover.consciousness.cross_modal_psi = phi;
                (strength, phi)
            } else {
                (0.0, 0.0)
            };

        // FEEDBACK: High cross-modal Phi boosts confidence (binding integration)
        // Science: Treisman (1996) — coherent binding -> confident perception
        {
            use crate::cognitive_loop::thresholds::{
                CROSS_MODAL_PSI_CONFIDENCE_SCALE, CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD,
            };
            if cross_modal_psi > CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD {
                let boost = ((cross_modal_psi - CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD)
                    * CROSS_MODAL_PSI_CONFIDENCE_SCALE) as f32;
                self.adjust_confidence("cross_modal_psi", boost);
            }
        }

        // FEEDBACK: Predictive <-> Cross-Modal bidirectional coupling (Talsma 2015)
        if let Some(ref mut mind) = self.consciousness_state.predictive_mind {
            if cross_modal_binding_strength > CROSS_MODAL_PRECISION_BOOST_THRESHOLD {
                let precision_boost =
                    (cross_modal_binding_strength - CROSS_MODAL_PRECISION_BOOST_THRESHOLD) as f64
                        * CROSS_MODAL_PRECISION_BOOST_SCALE;
                mind.precision.boost_precision(precision_boost);
            }
        }
        if let Some(ref mut binder) = self.consciousness_state.cross_modal_binder {
            if predictive_free_energy > CROSS_MODAL_FE_DAMPEN_THRESHOLD {
                let dampen = (1.0
                    - (predictive_free_energy - CROSS_MODAL_FE_DAMPEN_THRESHOLD)
                        * CROSS_MODAL_FE_DAMPEN_SCALE)
                    .max(CROSS_MODAL_FE_DAMPEN_MIN as f64) as f32;
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
            if let Err(e) = self.temporal_network.step(&input_array, 0.02) {
                tracing::warn!(err = %e, "temporal_network.step failed in episode replay");
            }

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
    pub(in crate::cognitive_loop) fn manage_arousal_trap(&mut self, affective_arousal: f32) {
        if affective_arousal > AROUSAL_SUPPRESS_THRESHOLD {
            // Attenuated 50%: DA learning_rate_factor() already scales LR via the bath
            let arousal_suppress = ((affective_arousal - AROUSAL_SUPPRESS_THRESHOLD)
                * AROUSAL_SUPPRESS_SCALE)
                .min(AROUSAL_SUPPRESS_MAX);
            self.scale_lr_pri(
                "affective_arousal_suppress",
                1.0 - arousal_suppress,
                Priority::Homeostatic,
            );

            // Arousal trap detection (Yerkes-Dodson 1908 — inverted-U performance curve)
            if affective_arousal > AROUSAL_TRAP_THRESHOLD {
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
                self.scale_lr_pri(
                    "arousal_trap_recovery",
                    1.0 - recovery_intensity * 0.1,
                    Priority::Homeostatic,
                );
                // Slight exploration boost: attenuated 50% (NE exploration_delta covers this)
                self.adjust_exploration_pri(
                    "arousal_trap_recovery",
                    recovery_intensity * 0.025,
                    Priority::Homeostatic,
                );
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
                self.set_exploration_pri("arousal_trap_escape", 1.0, Priority::Safety); // forced escape attempt
                self.scale_confidence_pri("arousal_trap_escape", 0.9, Priority::Safety);
                self.carryover.urgency.arousal_trap_counter = 0;
                tracing::debug!(
                    cycle = self.stats.total_cycles,
                    "Arousal trap escape: forced exploration after 10 high-arousal cycles"
                );
            }
        } else {
            // Reset trap counter when arousal drops below threshold
            self.carryover.urgency.arousal_trap_counter = 0;

            if affective_arousal < LOW_AROUSAL_CONSOLIDATION_THRESHOLD {
                // Low arousal enhances consolidation (Steriade 1996)
                // Attenuated 50%: DA handles low-error consolidation boost via the bath
                let consolidation_boost = ((LOW_AROUSAL_CONSOLIDATION_THRESHOLD
                    - affective_arousal)
                    * LOW_AROUSAL_CONSOLIDATION_SCALE)
                    .min(LOW_AROUSAL_CONSOLIDATION_MAX);
                self.scale_lr_pri(
                    "low_arousal_consolidate",
                    1.0 + consolidation_boost,
                    Priority::Homeostatic,
                );
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
            self.scale_exploration_pri(
                "moral_concern",
                MORAL_CONCERN_EXPLORATION_DAMPEN,
                Priority::Safety,
            );

            self.consciousness
                .self_model_tier
                .self_reflection
                .trust_threshold = (self
                .consciousness
                .self_model_tier
                .self_reflection
                .trust_threshold
                * 1.2)
                .clamp(0.1, 0.95);

            self.behavior.adaptive_behavior.pause_multiplier *= MORAL_CONCERN_PAUSE_BOOST;

            if moral_judgment.consent_violation
                || moral_judgment
                    .violations
                    .iter()
                    .any(|v| v.contains("perfect") || v.contains("harm"))
            {
                self.stats.moral_review_needed = true;
            }

            if moral_judgment.consent_violation {
                // Confidence = 1.0 - moral_score: consent violations with
                // low moral scores get maximum weight.
                let violation_conf = (1.0 - moral_score).clamp(0.3, 1.0);
                self.scale_confidence_weighted(
                    "moral_consent_viol",
                    MORAL_CONSENT_CONFIDENCE_SCALE,
                    Priority::Safety,
                    violation_conf,
                );
                self.carryover.learning.subsystem_lr_factor *= MORAL_CONSENT_LR_FACTOR;
                moral_steering_category = "consent";
            } else if moral_judgment.violations.iter().any(|v| v.contains("harm")) {
                let violation_conf = (1.0 - moral_score).clamp(0.3, 1.0);
                self.scale_exploration_pri(
                    "harm_detected",
                    MORAL_HARM_EXPLORATION_SCALE,
                    Priority::Safety,
                );
                self.scale_confidence_weighted(
                    "moral_harm_detect",
                    MORAL_HARM_CONFIDENCE_SCALE,
                    Priority::Safety,
                    violation_conf,
                );
                moral_steering_category = "harm";
            } else if moral_judgment
                .violations
                .iter()
                .any(|v| v.contains("perfect") || v.contains("duty"))
            {
                self.consciousness
                    .self_model_tier
                    .self_reflection
                    .force_reflection();
                self.carryover.learning.subsystem_lr_factor *= MORAL_DUTY_LR_FACTOR;
                moral_steering_category = "duty";
            } else if !moral_judgment.violations.is_empty() {
                self.carryover.learning.subsystem_lr_factor *= MORAL_OTHER_LR_FACTOR;
                moral_steering_category = "other";
            }
        } else if moral_score > MORAL_BENEFIT_THRESHOLD {
            // Confidence = moral_score: high-scoring moral assessments carry
            // more weight than borderline ones.
            // Science: Greene (2013) — moral intuition reliability scales with clarity.
            self.scale_confidence_weighted(
                "moral_benefit",
                MORAL_BENEFIT_CONFIDENCE_BOOST,
                Priority::Aesthetic,
                moral_score.clamp(0.0, 1.0),
            );
        }

        // FluctuatioAnimi → FEP exploration: moral ambiguity drives information-seeking.
        // High tension between opposing affects (e.g., Harm+Care in "mercy killing")
        // increases epistemic uncertainty, making the system explore rather than commit.
        // Science: Cushman (2013) — dual-process moral cognition under uncertainty.
        {
            let tension = self.carryover.consciousness.last_moral_fluctuatio_tension;
            if tension > 0.5 {
                let exploration_boost = (tension - 0.5) * 0.6; // 0.0–0.3 range
                self.adjust_exploration("fluctuatio_animi", exploration_boost);
                tracing::trace!(
                    tension = tension,
                    boost = exploration_boost,
                    "Moral ambiguity detected — boosting FEP exploration"
                );
            }
        }

        // Surprise-gated learning rate boost
        if is_surprised {
            let surprise_boost =
                (self.fep.agent.current_free_energy() as f32 / FEP_SURPRISE_SCALE).clamp(0.1, 0.5);
            self.adjust_lr_pri("fep_surprise", surprise_boost, Priority::Homeostatic);
        } else {
            self.scale_lr_pri("fep_decay", FEP_LR_DECAY, Priority::Homeostatic);
        }

        // Phase 21: Predictive free energy → surprise amplitude scaling
        let cached_pfe = self.carryover.consciousness.last_predictive_free_energy;
        let pfe_surprise_mod = if is_surprised && cached_pfe > PFE_SURPRISE_HIGH_THRESHOLD {
            let amplification = ((cached_pfe - PFE_SURPRISE_HIGH_THRESHOLD)
                * PFE_SURPRISE_AMPLIFY_SCALE)
                .min(PFE_SURPRISE_AMPLIFY_MAX) as f32;
            self.adjust_exploration("pfe_surprise_amplify", amplification);
            self.stats.pfe_surprise_mod_count += 1;
            amplification
        } else if is_surprised && cached_pfe < PFE_SURPRISE_LOW_THRESHOLD && cached_pfe > 0.0 {
            let dampening = ((PFE_SURPRISE_LOW_THRESHOLD - cached_pfe) * PFE_SURPRISE_DAMPEN_SCALE)
                .min(PFE_SURPRISE_DAMPEN_MAX) as f32;
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
        let adapted_thresholds = self
            .consciousness
            .self_model_tier
            .self_reflection
            .get_thresholds();
        self.behavior.flow_state.update_with_thresholds(
            pattern,
            prediction_error,
            coherence,
            self.prediction_confidence as f32,
            adapted_thresholds.flow_error,
            adapted_thresholds.flow_coherence,
        );

        // Curiosity drive — route exploration mutation through feedback system
        self.behavior
            .curiosity_drive
            .set_boredom_threshold(adapted_thresholds.boredom);
        match self.behavior.curiosity_drive.update(prediction_error) {
            super::super::drives::ExplorationUpdate::Set(val) => {
                self.set_exploration("curiosity_drive_boredom", val);
            }
            super::super::drives::ExplorationUpdate::Scale(factor) => {
                self.scale_exploration("curiosity_drive_decay", factor);
            }
        }

        // Physics-domain exploration modulation:
        // High physics similarity → exploit (known territory), low → explore (uncharted).
        #[cfg(feature = "physics-bridge")]
        if let Some(ref physics) = self.feature_integ.physics_integration {
            let score = physics.last_top_score;
            if score > PHYSICS_EXPLOIT_THRESHOLD {
                // Known physics territory — dampen exploration (scale 0.85–0.95)
                let factor = 1.0 - (score - PHYSICS_EXPLOIT_THRESHOLD) * PHYSICS_EXPLOIT_SCALE; // PHYSICS_EXPLOIT_THRESHOLD→1.0, 1.0→0.85
                self.scale_exploration("physics_domain_exploit", factor);
            } else if score > 0.0 && score < PHYSICS_EXPLORE_THRESHOLD {
                // Uncharted territory — boost exploration (scale 1.05–1.10)
                let factor = 1.0 + (PHYSICS_EXPLORE_THRESHOLD - score) * PHYSICS_EXPLORE_SCALE; // 0.0→1.10, PHYSICS_EXPLORE_THRESHOLD→1.00
                self.scale_exploration("physics_domain_explore", factor);
            }
        }

        // Self-reflection
        self.consciousness
            .self_model_tier
            .self_reflection
            .record_cycle(
                prediction_error,
                self.behavior.flow_state.in_flow,
                self.behavior.curiosity_drive.should_explore(),
                self.prediction_confidence as f32,
            );
        if self
            .consciousness
            .self_model_tier
            .self_reflection
            .should_reflect()
        {
            let recommendations = self.consciousness.self_model_tier.self_reflection.reflect();
            for rec in &recommendations {
                if rec.confidence < REFLECTION_CONFIDENCE_THRESHOLD {
                    continue;
                }
                match rec.target {
                    super::super::RecommendationTarget::LearningRate => match rec.direction {
                        super::super::AdjustmentDirection::Decrease => {
                            self.scale_lr("reflection_decrease", REFLECTION_LR_DECREASE);
                        }
                        super::super::AdjustmentDirection::Increase => {
                            self.scale_lr("reflection_increase", REFLECTION_LR_INCREASE);
                        }
                        _ => {}
                    },
                    super::super::RecommendationTarget::ExplorationFactor => match rec.direction {
                        super::super::AdjustmentDirection::Increase => {
                            self.adjust_exploration(
                                "metacog_explore_increase",
                                REFLECTION_EXPLORATION_INCREASE,
                            );
                        }
                        super::super::AdjustmentDirection::Decrease => {
                            self.scale_exploration(
                                "metacog_explore_decrease",
                                REFLECTION_EXPLORATION_DECREASE,
                            );
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "cycle_extracted_tests.rs"]
mod tests;
