//! Core dynamics phase of the cognitive cycle.
//!
//! Single method `phase_dynamics()` (~1900 LOC) that runs Phases A–11.
//! Ordering is load-bearing — do not reorder sections.
//!
//! ## Section index
//!
//! | Line  | Section | Description |
//! |-------|---------|-------------|
//! |  ~86  | Phase A: OBSERVE | Build immutable CycleSnapshot |
//! | ~123  | Phase B: COMPUTE | Run subsystem managers via trait |
//! | ~156  | Self-model | Accuracy tracking (EMA) |
//! | ~200  | Foveation | Vision-manifold coupling (cfg) |
//! | ~240  | 1a: Memory | Episodic recall + resonator + goals |
//! | ~290  | Binding | Phenomenal binding → threshold/confidence |
//! | ~468  | 1b+15+18: Emotion | Contagion + homeostasis (→ `apply_emotional_homeostasis`) |
//! | ~479  | 1c: Emotion | Unified emotional bridge (VAD) |
//! | ~498  | 2a: Semantic | Semantic memory lookup + LR modulation |
//! | ~660  | CfC step | Temporal network forward + prediction |
//! | ~780  | 6b: World model | World model stiffness → LR scaling |
//! | ~853  | MCTS | Plan evaluation + application |
//! | ~967  | FEP decomp | Free energy → accuracy/complexity/pragmatic |
//! | ~1006 | 10d.7: Moral | Moral modulation of inference |
//! | ~1030 | 10d.6b: FEP | Enhanced FEP bridge |
//! | ~1143 | Attention | Budget check + substrate tau scaling |
//! | ~1240 | Training | CfC weight update + async dispatch |
//! | ~1830 | Parallel | rayon::join post-processing (stability, episodic) |
//! | ~1869 | Result | Assemble DynamicsPhaseResult |

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use ndarray::Array1;
#[cfg(feature = "parallel")]
use rayon::join as rayon_join;
use std::borrow::Cow;
use std::time::Instant;

use super::helpers;
use super::phase_results::{
    DynAttention, DynCore, DynFep, DynGuidance, DynHomeostasis, DynNeuromod, DynReasoning,
    DynResonator, DynamicsPhaseResult, PerceptionPhaseResult,
};
use super::thresholds::{
    ALEATORIC_UNCERTAINTY_DEFAULT, AROUSAL_RECOVERY_TAU_SCALE, AROUSAL_TAU_DEADZONE,
    AROUSAL_TAU_SENSITIVITY, ATTENTION_BUDGET_US, BINDING_CONFIDENCE_THRESHOLD,
    BINDING_LOW_THRESHOLD, BINDING_STRONG_CONFIDENCE_SCALE, BINDING_STRONG_RELIEF_SCALE,
    BINDING_WEAK_CAUTION_SCALE, BINDING_WEAK_CONFIDENCE_SCALE, CAUSAL_ATTENTION_CONFIDENCE_SCALE,
    CAUSAL_ATTENTION_STRENGTH_THRESHOLD, CODEBOOK_FAMILIAR_TAU_SCALE, CODEBOOK_FAMILIAR_THRESHOLD,
    CODEBOOK_NOVEL_TAU_SCALE, CODEBOOK_NOVEL_THRESHOLD, COHERENCE_CONFIDENCE_BOOST,
    COHERENCE_HIGH_THRESHOLD, COHERENCE_LOW_DAMPEN_SCALE, COHERENCE_LOW_THRESHOLD,
    COHERENCE_PREDICTION_EMA, COHERENCE_VELOCITY_TAU_BOOST, COHERENCE_VELOCITY_TAU_DAMPEN,
    COHERENCE_VELOCITY_TAU_THRESHOLD, CONFIDENCE_CRASH_EXPLORATION_BOOST,
    CONFIDENCE_CRASH_FREEZE_CYCLES, CONFIDENCE_CRASH_THRESHOLD, DOMINANCE_CONFIDENCE_THRESHOLD,
    DOMINANCE_CONFIDENT, DOMINANCE_DEFAULT, DOMINANCE_FLOW_BASE, DOMINANCE_FLOW_SCALE,
    EPISTEMIC_EXPLORE_SCALE, EPISTEMIC_EXPLORE_THRESHOLD, EPISTEMIC_LOW_DAMPEN,
    EPISTEMIC_LOW_THRESHOLD, EPISTEMIC_OSCILLATION_MULTIPLIER, EPISTEMIC_OSCILLATION_THRESHOLD,
    EPISTEMIC_SEMANTIC_BOOST_SCALE, EPISTEMIC_SEMANTIC_BOOST_THRESHOLD,
    EPISTEMIC_SEMANTIC_CAUTION_BASE, EPISTEMIC_SEMANTIC_CAUTION_SCALE,
    EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD, EPISTEMIC_UNCERTAINTY_DEFAULT,
    FEP_ACCURACY_CONFIDENCE_THRESHOLD, FEP_COMPLEXITY_THRESHOLD, FEP_LEARNING_PLASTICITY_THRESHOLD,
    FEP_PRAGMATIC_EXPLOIT_THRESHOLD, FEP_PRAGMATIC_EXPLORE_THRESHOLD, FEP_SURPRISE_TAU_SCALE,
    FEP_TD_ERROR_DISCOVERY_THRESHOLD, HOMEOSTASIS_AROUSAL_TARGET, HOMEOSTASIS_EFFICIENCY_EMA,
    HOMEOSTASIS_EFFICIENCY_HIGH, HOMEOSTASIS_EFFICIENCY_LOW, HOMEOSTASIS_PULL_CRITICAL,
    HOMEOSTASIS_PULL_CRUISE, HOMEOSTASIS_PULL_INCREASE, HOMEOSTASIS_PULL_NORMAL,
    HOMEOSTASIS_PULL_REDUCTION, HORIZON_PE_CONTRACT_RATE, HORIZON_PE_CONTRACT_THRESHOLD,
    HORIZON_PE_EXPAND_RATE, HORIZON_PE_EXPAND_THRESHOLD, HORIZON_SLOPE_CONTRACT_CAP,
    HORIZON_SLOPE_CONTRACT_RATE, HORIZON_SLOPE_EXPAND_CAP, HORIZON_SLOPE_EXPAND_RATE,
    HORIZON_SLOPE_THRESHOLD, MCTS_CONSOLIDATE_CONFIDENCE_SCALE,
    MCTS_EFFECTIVENESS_CONFIDENCE_SCALE, MCTS_EFFECTIVENESS_EMA, MCTS_EFFECTIVENESS_EXPLORE_SCALE,
    MCTS_EFFECTIVENESS_HIGH, MCTS_EFFECTIVENESS_LOW, MCTS_EXPLOIT_LR_SCALE, MCTS_EXPLORE_SCALE,
    MCTS_PLAN_CONFIDENCE_THRESHOLD, MCTS_PLAN_WEIGHT_SCALE, MEMORY_RECALL_TOP_K,
    PE_VARIANCE_DAMPEN_SCALE, PE_VARIANCE_MAX_EFFECT, PE_VARIANCE_THRESHOLD,
    POLICY_FULL_AGREEMENT_BOOST, POLICY_MIN_WINDOW, POLICY_SOFT_THRESHOLD, POLICY_TEMP_BASE,
    POLICY_TEMP_RANGE, POLICY_WINDOW_SIZE, PREDICTION_HORIZON_MAX_SCALE,
    PREDICTION_HORIZON_MIN_SCALE, PREDICTIVE_BUDGET_GATING_RATIO, QUANTUM_COHERENCE_BOOST_SCALE,
    QUANTUM_COHERENCE_THRESHOLD, RESONANCE_TAU_CENTER, RESONANCE_TAU_SCALE,
    RESONATOR_CONSOLIDATION_THRESHOLD, RESONATOR_ERROR_CONFIDENCE_DAMPEN,
    RESONATOR_ERROR_EXPLORATION_SCALE, RESONATOR_ERROR_EXPLORATION_THRESHOLD,
    RESONATOR_FAMILIAR_LR_SCALE, RESONATOR_LOW_ERROR_CONFIDENCE_SCALE,
    RESONATOR_LOW_ERROR_THRESHOLD, RESONATOR_NOVEL_LR_SCALE, RESONATOR_NOVEL_THRESHOLD,
    SELF_MODEL_ACCURACY_EMA, SELF_MODEL_CONFIDENCE_WEIGHT, SELF_MODEL_HIGH_THRESHOLD,
    SELF_MODEL_HIGH_TRUST_BOOST, SELF_MODEL_LOW_CONFIDENCE_SCALE, SELF_MODEL_LOW_THRESHOLD,
    SELF_MODEL_URGENCY_WEIGHT, SELF_MODEL_WEIGHT_BONUS, SELF_MODEL_WEIGHT_HIGH_THRESHOLD,
    SELF_MODEL_WEIGHT_LOW_THRESHOLD, SELF_MODEL_WEIGHT_PENALTY, THALAMIC_DEEP_BUDGET_SCALE,
    THALAMIC_DEEP_LR_FACTOR, THALAMIC_DEEP_SALIENCE, THALAMIC_REFLEX_BUDGET_SCALE,
    THALAMIC_REFLEX_LR_FACTOR, THALAMIC_REFLEX_SALIENCE, TRAINING_BASE_IMPORTANCE,
    TRANSITION_COST_MAX_EFFECT, TRANSITION_COST_STRENGTH_SCALE, TRANSITION_COST_THRESHOLD,
    WM_MISMATCH_CONFIDENCE_SCALE, WM_MISMATCH_LR_SCALE, WORLD_MODEL_SPONGINESS_THRESHOLD,
    WORLD_MODEL_SPONGY_LR_SCALE, WORLD_MODEL_STIFFNESS_LR_SCALE, WORLD_MODEL_STIFFNESS_THRESHOLD,
    COHERENCE_VELOCITY_BUDGET_CONTRACT, COHERENCE_VELOCITY_BUDGET_EXPAND,
    COHERENCE_VELOCITY_BUDGET_THRESHOLD, HOMEOSTASIS_NEUROMOD_STEP,
    HOMEOSTASIS_RECALIBRATE_HIGH, HOMEOSTASIS_RECALIBRATE_LOW,
};
#[cfg(feature = "vision-manifold")]
use super::thresholds::{
    TRAINING_MAX_IMPORTANCE, VISION_SURPRISE_TRAINING_IMPORTANCE_SCALE,
    VISION_TRAINING_IMPORTANCE_SCALE,
};
use super::training::TrainingSample;
use super::{
    ActionHint, AdaptiveBehavior, CognitiveLoopService, CycleLearningResult, TrainingMethod,
};

impl CognitiveLoopService {
    /// Core dynamics phase: CycleSnapshot, subsystem managers, self-model tracking,
    /// resonator recall, semantic memory, CfC step, prediction, world model,
    /// emotion, FEP active inference, moral modulation, neuromod bath downstream,
    /// training, parallel post-processing.
    pub(super) fn phase_dynamics(
        &mut self,
        input: &str,
        perception: &PerceptionPhaseResult,
        cycle_start: Instant,
        module_timings: &mut super::ModuleTimings,
    ) -> DynamicsPhaseResult {
        let prediction_error = perception.urgency.prediction_error;
        let urgency = perception.urgency.urgency;
        let phi_attention_weight = perception.encoding.phi_attention_weight;
        let surprise_triggered = perception.exploration.surprise_triggered;
        let moral_concern_detected = perception.moral.moral_concern_detected;
        let selected_strategy = perception.strategy.selected_strategy;

        // Cache moral_score for neuromod feedback (consumed in helpers/cycle_phases.rs)
        self.carryover.quality.last_moral_score = perception.moral.moral_score;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE A: OBSERVE — Build immutable CycleSnapshot (Phase 2.3)
        // ═══════════════════════════════════════════════════════════════════════
        let cycle_snapshot = super::subsystem_trait::CycleSnapshot::build(
            self.stats.total_cycles as u64,
            self.prediction_confidence,
            self.fep.lr_boost,
            prediction_error,
            self.voice_coherence.bridge.smoothed_coherence(),
            self.stats.unified_psi as f64,
            phi_attention_weight,
            self.emotion_contagion.arousal,
            self.emotion_contagion.valence,
            self.thermodynamic_load,
            self.carryover.quality.last_dissipative_health,
            self.somatic_bridge.systemic_stress(),
            urgency,
            false, // attention_budget_exceeded not yet known at this point
            &perception.encoding.compressed_state,
            &perception.encoding.hv16_cached,
            &self.carryover.consciousness,
            &self.carryover.quality,
        );
        self.last_snapshot = Some(cycle_snapshot);

        // ── Phase B: COMPUTE — Run managers via CognitiveSubsystem trait ──
        {
            use super::subsystem_trait::CognitiveSubsystem;
            if let Some(ref snapshot) = self.last_snapshot {
                let urgency_u8 = snapshot.urgency;
                let cycle_num = snapshot.cycle_number;

                if self.drive_manager.should_run(cycle_num, urgency_u8) {
                    let drive_output = self.drive_manager.process(snapshot);
                    self.subsystem_collector
                        .record("drive_manager", drive_output);
                }

                if self.memory_manager.should_run(cycle_num, urgency_u8) {
                    let memory_output = self.memory_manager.process(snapshot);
                    self.subsystem_collector
                        .record("memory_manager", memory_output);
                }

                if self.learning_manager.should_run(cycle_num, urgency_u8) {
                    let learning_output = self.learning_manager.process(snapshot);
                    self.subsystem_collector
                        .record("learning_manager", learning_output);
                }

                if self.perception_manager.should_run(cycle_num, urgency_u8) {
                    let perception_output = self.perception_manager.process(snapshot);
                    self.subsystem_collector
                        .record("perception_manager", perception_output);
                }
            }
        }

        // ── Phase 17: Self-model accuracy tracking ───────────────────────
        let self_model_accuracy = self.carryover.learning.self_model_accuracy;
        if let Some((made_at, pred_confidence, pred_urgency)) =
            self.carryover.history.self_model_prediction.take()
        {
            if self.stats.total_cycles >= made_at + 5 {
                let confidence_error = (self.prediction_confidence - pred_confidence).abs() as f32;
                let urgency_match = if urgency == pred_urgency { 1.0f32 } else { 0.0 };
                let accuracy = (1.0 - confidence_error) * SELF_MODEL_CONFIDENCE_WEIGHT
                    + urgency_match * SELF_MODEL_URGENCY_WEIGHT;
                self.carryover.learning.self_model_accuracy =
                    self.carryover.learning.self_model_accuracy * SELF_MODEL_ACCURACY_EMA
                        + accuracy * (1.0 - SELF_MODEL_ACCURACY_EMA);
                self.stats.self_model_predictions_validated += 1;
                self.stats.avg_self_model_accuracy = self.stats.avg_self_model_accuracy
                    * SELF_MODEL_ACCURACY_EMA
                    + accuracy * (1.0 - SELF_MODEL_ACCURACY_EMA);

                if self.carryover.learning.self_model_accuracy > SELF_MODEL_HIGH_THRESHOLD {
                    let trust_boost = (self.carryover.learning.self_model_accuracy
                        - SELF_MODEL_HIGH_THRESHOLD)
                        * SELF_MODEL_HIGH_TRUST_BOOST;
                    self.adjust_confidence("self_model_trust", trust_boost);
                }
                if self.carryover.learning.self_model_accuracy < SELF_MODEL_LOW_THRESHOLD {
                    self.scale_confidence("self_model_low_acc", SELF_MODEL_LOW_CONFIDENCE_SCALE);
                }
            } else {
                self.carryover.history.self_model_prediction =
                    Some((made_at, pred_confidence, pred_urgency));
            }
        }
        if self.stats.total_cycles % 7 == 0
            && self.carryover.history.self_model_prediction.is_none()
        {
            self.carryover.history.self_model_prediction =
                Some((self.stats.total_cycles, self.prediction_confidence, urgency));
            self.stats.self_model_predictions_made += 1;
        }

        // Session 10 Item 2: Self-model accuracy → proposal weighting.
        // Accurate self-model → boost self-model confidence proposals; inaccurate → dampen.
        // Science: Friston (2010) — interoceptive accuracy modulates self-model trust.
        if self_model_accuracy > SELF_MODEL_WEIGHT_HIGH_THRESHOLD {
            self.scale_confidence(
                "self_model_weight_hi",
                (1.0 + SELF_MODEL_WEIGHT_BONUS) as f32,
            );
        } else if self_model_accuracy < SELF_MODEL_WEIGHT_LOW_THRESHOLD {
            self.scale_confidence("self_model_weight_lo", SELF_MODEL_WEIGHT_PENALTY as f32);
        }

        // Session 10 Item 1: Confidence crash detector → emergency stabilization.
        // >30% confidence drop in 1 cycle → freeze LR for 3 cycles, boost exploration.
        // Science: Cools et al. (2008) — rapid confidence collapse triggers serotonergic dip.
        // Session 11 Fix: Use Set proposal to pin LR at cycle-start value during freeze.
        let confidence_crash_detected;
        let lr_frozen;
        {
            let prev_conf = self.carryover.quality.prev_confidence_for_crash;
            let current_conf = self.prediction_confidence;
            let drop = prev_conf - current_conf;
            // Session 11 Item 4: Flow state raises crash threshold (×1.5).
            // Flow = committed mode, transient dips are normal.
            // Science: Csikszentmihalyi (1990) — flow tolerates transient perturbation.
            let effective_crash_threshold = if self.flow_state.in_flow {
                CONFIDENCE_CRASH_THRESHOLD * 1.5
            } else {
                CONFIDENCE_CRASH_THRESHOLD
            };
            confidence_crash_detected = drop > prev_conf * effective_crash_threshold
                && prev_conf > 0.15
                && self.stats.total_cycles > 10;
            if confidence_crash_detected {
                // Session 11 Item 5: Grace period — lighter freeze after recent mode transition.
                // Post-transition confidence drops are expected, not emergencies.
                let freeze_duration = if self.carryover.urgency.mode_stability_counter < 3 {
                    1 // Light freeze: mode just changed, drop is expected
                } else {
                    CONFIDENCE_CRASH_FREEZE_CYCLES // Full freeze
                };
                self.carryover.quality.crash_freeze_remaining = freeze_duration;
                self.adjust_exploration("confidence_crash", CONFIDENCE_CRASH_EXPLORATION_BOOST);
                tracing::debug!(
                    "Confidence crash detected: {prev_conf:.3} → {current_conf:.3} (drop={drop:.3}), \
                     freezing LR for {freeze_duration} cycles"
                );
            }
            // Session 11 Fix: Pin LR to cycle-start value with Set (overrides all other proposals).
            lr_frozen = self.carryover.quality.crash_freeze_remaining > 0;
            if lr_frozen {
                let frozen_lr = self.feedback_state.cycle_start_lr() as f32;
                self.set_lr("crash_freeze", frozen_lr);
                self.carryover.quality.crash_freeze_remaining -= 1;
            }
        }

        // Session 9 Item 1: PE variance → confidence modulation.
        // High error variance (unstable PE) should dampen confidence more than steady errors.
        // Yu & Dayan (2005): expected vs unexpected uncertainty differentially modulate ACh/NE.
        let pe_variance = self.stats.avg_prediction_error_sq
            - self.stats.avg_prediction_error * self.stats.avg_prediction_error;
        let pe_variance = pe_variance.max(0.0); // Clamp numerical noise
        if pe_variance > PE_VARIANCE_THRESHOLD && self.stats.total_cycles > 20 {
            // High variance = unstable errors → dampen confidence proportionally
            let variance_dampen = 1.0
                - (pe_variance - PE_VARIANCE_THRESHOLD).min(PE_VARIANCE_MAX_EFFECT)
                    * PE_VARIANCE_DAMPEN_SCALE; // 0.90–1.0
            self.scale_confidence("pe_variance", variance_dampen);
        }

        // FEEDBACK: Quantum coherence boosts exploration (prev cycle)
        if self.carryover.consciousness.quantum_coherence > QUANTUM_COHERENCE_THRESHOLD {
            let coherence_boost = (self.carryover.consciousness.quantum_coherence
                - QUANTUM_COHERENCE_THRESHOLD) as f32
                * QUANTUM_COHERENCE_BOOST_SCALE;
            self.adjust_exploration("quantum_coherence", coherence_boost);
        }

        // ── Foveation → dynamics coupling ────────────────────────────────
        // Corbetta & Shulman (2002): recognized objects reduce attentional vigilance;
        // novel objects boost learning.
        #[cfg(feature = "foveation")]
        {
            use super::thresholds::{
                FOVEATION_CONFIDENCE_BOOST, FOVEATION_FAMILIAR_EXPLORATION_DAMPEN,
                FOVEATION_FAMILIAR_RECOGNITION_COUNT, FOVEATION_HIGH_CONFIDENCE_THRESHOLD,
                FOVEATION_NOVEL_LR_BOOST,
            };
            let fov_count = perception.foveation_recognition_count;
            let fov_conf = perception.foveation_top_confidence;

            if fov_count >= FOVEATION_FAMILIAR_RECOGNITION_COUNT
                && fov_conf > FOVEATION_HIGH_CONFIDENCE_THRESHOLD
            {
                // Familiar scene: dampen exploration, boost confidence
                self.scale_exploration("foveation_familiar", FOVEATION_FAMILIAR_EXPLORATION_DAMPEN);
                self.scale_confidence("foveation_familiar", FOVEATION_CONFIDENCE_BOOST);
            } else if fov_count > 0 && fov_conf < FOVEATION_HIGH_CONFIDENCE_THRESHOLD {
                // Novel objects: boost learning rate
                self.scale_lr("foveation_novel", FOVEATION_NOVEL_LR_BOOST);
            }
        }

        // ── Vision surprise → exploration urgency ────────────────────────
        // Friston (2010): free energy (surprise) is the fundamental drive for active inference.
        #[cfg(feature = "vision-manifold")]
        {
            use super::thresholds::{
                VISION_SURPRISE_EXPLORATION_SCALE, VISION_SURPRISE_EXPLORATION_THRESHOLD,
            };
            let mean_surp = perception.vision_mean_surprise;
            if mean_surp > VISION_SURPRISE_EXPLORATION_THRESHOLD {
                let boost = (mean_surp - VISION_SURPRISE_EXPLORATION_THRESHOLD)
                    * VISION_SURPRISE_EXPLORATION_SCALE;
                self.adjust_exploration("vision_surprise", boost);
            }
        }

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
            && self.stats.total_cycles > 5
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
            -confidence_boost
        } else {
            0.0
        };

        // ── Phase 17: Coherence memoization — cache pre-update value ─────
        let pre_update_coherence = self.voice_coherence.bridge.smoothed_coherence();

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
        let binding_confidence_mod = if cached_binding > BINDING_CONFIDENCE_THRESHOLD {
            let conf_boost =
                (cached_binding - BINDING_CONFIDENCE_THRESHOLD) * BINDING_STRONG_CONFIDENCE_SCALE;
            self.adjust_confidence("binding_strong", conf_boost);
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
        let reflection_thresholds = self.self_model_tier.self_reflection.get_thresholds();
        let resonator_coherence_gate = pre_update_coherence > reflection_thresholds.coherence_gate
            || self.stats.total_cycles < 10;
        if resonator_coherence_gate && urgency.should_run(self.stats.total_cycles, 1, 1, 4) {
            if let Some(ref mut res_mem) = self.resonator_memory {
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
                            super::CognitiveDepth::DeepThought => MEMORY_RECALL_TOP_K * 2,
                            super::CognitiveDepth::Cortical => MEMORY_RECALL_TOP_K,
                            super::CognitiveDepth::Reflex => 1,
                        };
                        let top_matches: Vec<_> = matches.into_iter().take(recall_k).collect();

                        let best_match_sim = top_matches
                            .iter()
                            .map(|m| {
                                helpers::cosine_f32(&perception.encoding.compressed_state, &m.hv)
                            })
                            .fold(0.0f32, f32::max);
                        let match_timestamps: Vec<u64> =
                            top_matches.iter().map(|m| m.timestamp).collect();
                        resonator_best_sim = best_match_sim;

                        if best_match_sim > 0.3 {
                            let best_ep = top_matches.iter().max_by(|a, b| {
                                let sa: f32 = perception
                                    .encoding
                                    .compressed_state
                                    .iter()
                                    .zip(a.hv.iter())
                                    .map(|(x, y)| x * y)
                                    .sum();
                                let sb: f32 = perception
                                    .encoding
                                    .compressed_state
                                    .iter()
                                    .zip(b.hv.iter())
                                    .map(|(x, y)| x * y)
                                    .sum();
                                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            if let Some(ep) = best_ep {
                                self.stats.last_resonator_prediction = Some(ep.hv.clone());
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
                                            self.emotion_contagion.valence =
                                                (self.emotion_contagion.valence + 0.1)
                                                    .clamp(-1.0, 1.0);
                                        }
                                        "negative" => {
                                            self.emotion_contagion.valence =
                                                (self.emotion_contagion.valence - 0.1)
                                                    .clamp(-1.0, 1.0);
                                        }
                                        "high" => {
                                            self.adjust_confidence("resonator_factor_high", 0.03);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        if best_match_sim > 0.3 {
                            self.adjust_confidence("resonator_recall_prime", best_match_sim * 0.02);
                            resonator_wm_primed = true;
                        }

                        if !match_timestamps.is_empty() {
                            if let Some(ref mut replay) = self.phi_episodic_replay {
                                replay.boost_causal_consolidation(&match_timestamps, 0.05);
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
                + (resonator_best_sim - RESONATOR_CONSOLIDATION_THRESHOLD) as f64 * 0.1)
                .min(2.0);
            if self.stats.total_cycles > 10 {
                self.scale_lr("resonator_familiar", RESONATOR_FAMILIAR_LR_SCALE);
            }
        } else if resonator_best_sim < RESONATOR_NOVEL_THRESHOLD
            && resonator_best_sim > 0.0
            && self.stats.total_cycles > 10
        {
            self.scale_lr("resonator_novel", RESONATOR_NOVEL_LR_SCALE);
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
            if goal_priority > 0.5 && !matches!(urgency, super::CycleUrgency::Critical) {
                let goal_lr_boost = (goal_priority - 0.5) * 0.1;
                self.scale_lr("goal_priority", 1.0 + goal_lr_boost);
            }
            if prediction_error < self.config.learning_threshold && goal_priority > 0.3 {
                self.adjust_exploration("goal_pursuit", goal_priority * 0.03);
            }
        }

        // 1b. Analyze emotional content for simple contagion (keyword-based)
        self.emotion_contagion.analyze(input);

        // ── Phase 15+18: Emotional homeostasis ──
        // Session 9 Item 7: Track pre-pull valence distance for efficiency computation.
        let pre_pull_valence = self.emotion_contagion.valence;
        let (valence_homeostasis_pull, arousal_homeostasis_pull, mut homeostasis_pull_strength) =
            self.apply_emotional_homeostasis();

        // Compute homeostasis efficiency: ratio of post/pre distance to target (0.0).
        // <1.0 = pulls working, >1.0 = overcorrecting.
        // Cannon (1929)/Ashby (1960): homeostatic regulation must be monitored for overshoot.
        let post_pull_valence = self.emotion_contagion.valence;
        let pre_dist = pre_pull_valence.abs().max(0.01);
        let post_dist = post_pull_valence.abs();
        let cycle_efficiency = post_dist / pre_dist;
        // EMA smooth (alpha=0.2)
        self.carryover.quality.homeostasis_efficiency =
            self.carryover.quality.homeostasis_efficiency * (1.0 - HOMEOSTASIS_EFFICIENCY_EMA)
                + cycle_efficiency * HOMEOSTASIS_EFFICIENCY_EMA;

        // High transition cost → strengthen homeostasis to resist unnecessary mode changes.
        // Kelso (1995): costly transitions increase the system's tendency to stay in current attractor.
        if self.stats.avg_transition_cost > TRANSITION_COST_THRESHOLD {
            homeostasis_pull_strength *= 1.0
                + (self.stats.avg_transition_cost - TRANSITION_COST_THRESHOLD)
                    .min(TRANSITION_COST_MAX_EFFECT)
                    * TRANSITION_COST_STRENGTH_SCALE;
        }

        // Session 10 Item 4: Homeostasis efficiency → pull strength adaptation.
        // Efficient regulation → reduce pull (self-attenuate); inefficient → increase.
        // Science: Ashby (1960) — good regulation requires less intervention over time.
        let eff = self.carryover.quality.homeostasis_efficiency;
        if eff > HOMEOSTASIS_EFFICIENCY_HIGH {
            homeostasis_pull_strength *= HOMEOSTASIS_PULL_REDUCTION;
        } else if eff < HOMEOSTASIS_EFFICIENCY_LOW && eff > 0.0 {
            homeostasis_pull_strength *= HOMEOSTASIS_PULL_INCREASE;
        }

        // Homeostasis efficiency → learning recalibration.
        // Sustained overshoot (>1.15) → system is overcorrecting → dampen LR.
        // Sustained undershoot (<0.85) → system is sluggish → boost LR.
        // Science: Turrigiano (2008) — homeostatic failure triggers synaptic recalibration.
        if eff > HOMEOSTASIS_RECALIBRATE_HIGH && self.stats.total_cycles > 20 {
            self.scale_lr("homeostasis_overcorrect", 1.0 - HOMEOSTASIS_NEUROMOD_STEP);
            self.scale_exploration("homeostasis_overcorrect", 1.0 + HOMEOSTASIS_NEUROMOD_STEP);
        } else if eff < HOMEOSTASIS_RECALIBRATE_LOW && eff > 0.0 && self.stats.total_cycles > 20 {
            self.scale_lr("homeostasis_sluggish", 1.0 + HOMEOSTASIS_NEUROMOD_STEP);
        }

        // Session 10 Item 3: Coherence velocity → CfC tau modulation.
        // Rising coherence → slow down (stabilize); falling → speed up (explore corrections).
        // Science: Buzsáki (2006) — coherent oscillations modulate integration timescale.
        // (Applied below in delta_t product chain as coherence_velocity_tau_factor.)

        // ═══════════════════════════════════════════════════════════════════════
        // 1c. Update Unified Emotional Bridge (VAD-based)
        // ═══════════════════════════════════════════════════════════════════════
        let simple_valence = self.emotion_contagion.prosody_valence() as f64;
        let simple_arousal = self.emotion_contagion.prosody_arousal() as f64;
        let dominance = if self.flow_state.in_flow {
            DOMINANCE_FLOW_BASE + DOMINANCE_FLOW_SCALE * self.flow_state.intensity as f64
        } else if self.prediction_confidence > DOMINANCE_CONFIDENCE_THRESHOLD {
            DOMINANCE_CONFIDENT
        } else {
            DOMINANCE_DEFAULT
        };

        self.unification_engine.emotional.update_from_core_affect(
            simple_valence,
            simple_arousal,
            dominance,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // 2a. SEMANTIC MEMORY
        // ═══════════════════════════════════════════════════════════════════════
        let _t_core = Instant::now();
        let semantic_hdc: Cow<'_, [f32]> = self
            .temporal_network
            .project_to_hdc_vec(&perception.encoding.compressed_state)
            .map(Cow::Owned)
            .unwrap_or(Cow::Borrowed(&perception.encoding.compressed_state));
        let current_phi_for_lr = pre_update_coherence as f64;
        let mut semantic_lr_factor = self.semantic_memory.compute_lr_factor_phi_weighted(
            &semantic_hdc,
            3,
            current_phi_for_lr,
            self.stats.total_cycles as u64,
        );
        module_timings.core_semantic_lookup = _t_core.elapsed().as_micros() as u64;

        // ── Phase 20: Epistemic gate → semantic memory LR bidirectionality ───
        let prev_epistemic = self.carryover.quality.last_epistemic_confidence;
        let epistemic_semantic_lr_mod: f32 =
            if prev_epistemic < EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD && prev_epistemic > 0.0 {
                let caution = EPISTEMIC_SEMANTIC_CAUTION_BASE
                    + prev_epistemic * EPISTEMIC_SEMANTIC_CAUTION_SCALE;
                semantic_lr_factor *= caution;
                self.stats.epistemic_semantic_mod_count += 1;
                caution - 1.0
            } else if prev_epistemic > EPISTEMIC_SEMANTIC_BOOST_THRESHOLD {
                let boost = 1.0_f32
                    + (prev_epistemic - EPISTEMIC_SEMANTIC_BOOST_THRESHOLD)
                        * EPISTEMIC_SEMANTIC_BOOST_SCALE;
                semantic_lr_factor *= boost;
                self.stats.epistemic_semantic_mod_count += 1;
                boost - 1.0
            } else {
                0.0
            };

        // 2b. Physics bridge: blend physics-informed HDC into compressed state
        #[allow(unused_mut)]
        let mut compressed_for_cfc = perception.encoding.compressed_state.clone();
        #[cfg(feature = "physics-bridge")]
        if let Some(ref mut physics) = self.physics_integration {
            physics.query_cycle(
                self.stats.total_cycles,
                self.config.physics_bridge_query_interval,
                self.config.physics_bridge_blend_weight,
                self.substrate_manager.tau_factor,
                self.substrate_manager.scale_pressure,
                &perception.encoding.hv16_cached,
                &mut compressed_for_cfc,
            );
        }

        // 3. Convert to ndarray for CfC
        let input_array: Array1<f32> = compressed_for_cfc.iter().copied().collect();

        // 4. Step CfC forward with current input
        let resonance_tau_factor = if self.carryover.history.resonance_frequency > 0.0 {
            let deviation = (self.carryover.history.resonance_frequency as f32
                - RESONANCE_TAU_CENTER as f32)
                .clamp(-0.5, 0.5);
            1.0 - (deviation * RESONANCE_TAU_SCALE)
        } else {
            1.0
        };
        let arousal_tau_factor =
            if (self.carryover.history.body_arousal - 0.5).abs() > AROUSAL_TAU_DEADZONE {
                1.0 + (self.carryover.history.body_arousal - 0.5) * AROUSAL_TAU_SENSITIVITY
            } else {
                1.0
            };
        let codebook_tau_factor = if resonator_best_sim > CODEBOOK_FAMILIAR_THRESHOLD {
            1.0 - (resonator_best_sim - CODEBOOK_FAMILIAR_THRESHOLD) * CODEBOOK_FAMILIAR_TAU_SCALE
        } else if resonator_best_sim > 0.0 && resonator_best_sim < CODEBOOK_NOVEL_THRESHOLD {
            1.0 + (CODEBOOK_NOVEL_THRESHOLD - resonator_best_sim) * CODEBOOK_NOVEL_TAU_SCALE
        } else {
            1.0
        };
        let arousal_recovery_tau_factor;
        let arousal_recovery_active;
        if self.carryover.urgency.arousal_trap_counter > 5
            && self.carryover.urgency.arousal_trap_counter <= 10
        {
            let recovery_intensity = (self.carryover.urgency.arousal_trap_counter - 5) as f32 / 5.0;
            arousal_recovery_tau_factor = 1.0 + recovery_intensity * AROUSAL_RECOVERY_TAU_SCALE;
            arousal_recovery_active = true;
        } else {
            arousal_recovery_tau_factor = 1.0;
            arousal_recovery_active = false;
        }

        // FEP surprise → CfC time constant modulation.
        // Friston (2010): high surprise (free energy) accelerates inference dynamics;
        // low surprise allows consolidation via slower dynamics.
        // Factor: [0.8, 1.2] — moderate modulation to prevent instability.
        let fep_tau_factor = if let Some(ref fe) = self.fep.agent.last_fe_components {
            let surprise_norm = (fe.surprise as f32).clamp(0.0, 2.0) / 2.0; // [0, 1]
            1.0 - surprise_norm * FEP_SURPRISE_TAU_SCALE // high surprise → 0.8 (faster), low → 1.0
        } else {
            1.0
        };

        // Session 10 Item 3: Coherence velocity tau factor.
        // Session 11 Item 3: Gate behind cycle > 5 to avoid spurious velocity from default init.
        let coherence_velocity_tau_factor = {
            let cv = self.carryover.quality.coherence_velocity;
            if self.stats.total_cycles > 5 && cv > COHERENCE_VELOCITY_TAU_THRESHOLD {
                COHERENCE_VELOCITY_TAU_BOOST
            } else if self.stats.total_cycles > 5 && cv < -COHERENCE_VELOCITY_TAU_THRESHOLD {
                COHERENCE_VELOCITY_TAU_DAMPEN
            } else {
                1.0
            }
        };

        // Prediction horizon → CfC temporal integration depth.
        // Clark (2013): high PE → contract horizons (focus near-term);
        // low PE → expand horizons (exploit stability for planning).
        // This complements FEP surprise tau (Friston 2010) — they work in synergy:
        // FEP surprise drives fast dynamics, horizon scale drives planning depth.
        let prediction_horizon_tau = {
            let pe = self.stats.avg_prediction_error.clamp(0.0, 1.0);
            let pe_scale = if pe > HORIZON_PE_CONTRACT_THRESHOLD {
                1.0 - (pe - HORIZON_PE_CONTRACT_THRESHOLD) * HORIZON_PE_CONTRACT_RATE
            } else if pe < HORIZON_PE_EXPAND_THRESHOLD {
                1.0 + (HORIZON_PE_EXPAND_THRESHOLD - pe) * HORIZON_PE_EXPAND_RATE
            } else {
                1.0
            };
            let slope = perception.urgency.error_slope;
            let slope_scale = if slope > HORIZON_SLOPE_THRESHOLD {
                1.0 - (slope - HORIZON_SLOPE_THRESHOLD).min(HORIZON_SLOPE_CONTRACT_CAP)
                    * HORIZON_SLOPE_CONTRACT_RATE
            } else if slope < -HORIZON_SLOPE_THRESHOLD {
                1.0 + (-slope - HORIZON_SLOPE_THRESHOLD).min(HORIZON_SLOPE_EXPAND_CAP)
                    * HORIZON_SLOPE_EXPAND_RATE
            } else {
                1.0
            };
            (pe_scale * slope_scale)
                .clamp(PREDICTION_HORIZON_MIN_SCALE, PREDICTION_HORIZON_MAX_SCALE)
        };

        let delta_t = self.config.cfc_config.delta_t
            * resonance_tau_factor
            * arousal_tau_factor
            * codebook_tau_factor
            * arousal_recovery_tau_factor
            * fep_tau_factor
            * coherence_velocity_tau_factor
            * prediction_horizon_tau
            * self
                .somatic_bridge
                .to_interoceptive_signals()
                .tau_slowdown_factor as f32
            * self.substrate_manager.tau_factor;
        let _t_core = Instant::now();
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(error = %e, "CfC temporal step failed — continuing with stale state");
        }
        module_timings.core_cfc_step = _t_core.elapsed().as_micros() as u64;

        // 5. Get multi-scale predictions
        let _t_core = Instant::now();
        let (prediction, raw_predictions) = self.get_multi_scale_prediction(&input_array);

        let prediction_coherence = if self.stats.total_cycles % 11 == 0 {
            let coh = Self::compute_prediction_coherence_from_cache(&raw_predictions);
            self.stats.avg_prediction_coherence = self.stats.avg_prediction_coherence
                * COHERENCE_PREDICTION_EMA
                + coh * (1.0 - COHERENCE_PREDICTION_EMA);
            if coh < COHERENCE_LOW_THRESHOLD {
                let coh_dampen = (COHERENCE_LOW_THRESHOLD - coh) * COHERENCE_LOW_DAMPEN_SCALE;
                self.scale_confidence("pred_coherence_low", 1.0 - coh_dampen);
            }
            if coh > COHERENCE_HIGH_THRESHOLD {
                let coh_boost = (coh - COHERENCE_HIGH_THRESHOLD) * COHERENCE_CONFIDENCE_BOOST;
                self.adjust_confidence("pred_coherence_high", coh_boost);
            }
            coh
        } else {
            self.stats.avg_prediction_coherence
        };

        // 5b. Epistemic vs aleatoric uncertainty decomposition.
        // Epistemic (model uncertainty): prediction disagreement across horizons — reducible
        // by exploration. Aleatoric (data noise): mean per-horizon prediction variance — not
        // reducible. Only epistemic uncertainty should drive exploration.
        // Depeweg et al. (2018): decomposing uncertainty for active learning.
        let (epistemic_uncertainty, aleatoric_uncertainty) = if raw_predictions.len() >= 2 {
            // Epistemic ≈ 1 - cross-horizon coherence (disagreement = model uncertainty)
            let epistemic = (1.0 - prediction_coherence).clamp(0.0, 1.0);

            // Aleatoric ≈ mean within-dimension variance across predictions
            let dim = raw_predictions[0].len().max(1);
            let n = raw_predictions.len() as f32;
            let mut mean_var = 0.0f32;
            for d in 0..dim {
                let mean: f32 = raw_predictions.iter().map(|p| p[d]).sum::<f32>() / n;
                let var: f32 = raw_predictions
                    .iter()
                    .map(|p| (p[d] - mean).powi(2))
                    .sum::<f32>()
                    / n;
                mean_var += var;
            }
            let aleatoric = (mean_var / dim as f32).sqrt().clamp(0.0, 1.0);
            (epistemic, aleatoric)
        } else {
            (EPISTEMIC_UNCERTAINTY_DEFAULT, ALEATORIC_UNCERTAINTY_DEFAULT) // defaults when insufficient data
        };

        // Only epistemic uncertainty drives exploration (aleatoric is irreducible noise).
        // Use smoothed epistemic for stability; raw for responsiveness on first cycle.
        // Depeweg et al. (2018): decomposing uncertainty for active learning.
        let smoothed_eu = self.carryover.quality.smoothed_epistemic_uncertainty;
        let eu_for_exploration = if smoothed_eu > 0.0 {
            smoothed_eu
        } else {
            epistemic_uncertainty
        };
        if eu_for_exploration > EPISTEMIC_EXPLORE_THRESHOLD && self.stats.total_cycles % 7 == 0 {
            let mut epistemic_explore =
                (eu_for_exploration - EPISTEMIC_EXPLORE_THRESHOLD) * EPISTEMIC_EXPLORE_SCALE;
            // Oscillation + high uncertainty = confused AND unstable → stronger exploration.
            // Doya (2002) + Schmidhuber (2010): compound uncertainty warrants aggressive search.
            if perception.urgency.oscillation_ratio > EPISTEMIC_OSCILLATION_THRESHOLD {
                epistemic_explore *= EPISTEMIC_OSCILLATION_MULTIPLIER;
            }
            self.adjust_exploration("epistemic_uncertainty", epistemic_explore);
        } else if eu_for_exploration < EPISTEMIC_LOW_THRESHOLD && self.stats.total_cycles % 7 == 0 {
            // Low epistemic uncertainty → dampen exploration (model is confident).
            self.adjust_exploration("epistemic_low", -EPISTEMIC_LOW_DAMPEN);
        }

        // 6. Get current CfC state as output
        let output = self
            .temporal_network
            .read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);
        module_timings.core_predict = _t_core.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // 6b. World Model
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        self.fep
            .world_model
            .update_sensory(&perception.encoding.compressed_state);

        // Incorporate causal structure into world model (every 41 cycles, co-prime).
        // Pearl (2009): causal knowledge provides structural priors beyond correlation.
        if self.stats.total_cycles % 41 == 0 {
            if let Some(ref enhancer) = self.causal_enhancer {
                let graph = enhancer.current_graph();
                if !graph.is_empty() {
                    let edges: Vec<(usize, usize, f32)> = graph
                        .edges
                        .iter()
                        .map(|e| (e.from, e.to, e.strength as f32))
                        .collect();
                    self.fep.world_model.incorporate_causal_structure(&edges);
                }
            }
        }

        let wm_stiffness = self.fep.world_model.avg_error.clamp(0.0, 1.0);
        if self.stats.total_cycles > 20 {
            if wm_stiffness > WORLD_MODEL_STIFFNESS_THRESHOLD {
                let stiffness_nudge = (wm_stiffness - WORLD_MODEL_STIFFNESS_THRESHOLD)
                    * WORLD_MODEL_STIFFNESS_LR_SCALE;
                self.adjust_lr("wm_stiff", stiffness_nudge);
            } else if wm_stiffness < WORLD_MODEL_SPONGINESS_THRESHOLD {
                let spongy_dampen =
                    (WORLD_MODEL_SPONGINESS_THRESHOLD - wm_stiffness) * WORLD_MODEL_SPONGY_LR_SCALE;
                self.scale_lr("wm_spongy", 1.0 - spongy_dampen);
            }
        }

        let level_errors = self.fep.world_model.level_errors();
        let mut wm_sensory_mismatch = false;
        if level_errors.len() >= 2 && self.stats.total_cycles > 10 {
            let sensory_error = level_errors[0];
            let abstract_error = level_errors[level_errors.len() - 1];
            if abstract_error > sensory_error * 1.5 && abstract_error > 0.1 {
                self.adjust_exploration("conceptual_confusion", 0.08);
            }
            wm_sensory_mismatch = sensory_error > abstract_error * 2.0 && sensory_error > 0.1;
        }
        module_timings.world_model = _t.elapsed().as_micros() as u64;

        // 8. Capture previous state BEFORE create_experience updates it
        let previous_state = self.last_state.clone();

        // 9. Create experience
        self.create_experience(
            &perception.encoding.compressed_state,
            &prediction,
            prediction_error,
        );

        // 10. Update coherence bridge
        let tau_owned: Vec<ndarray::Array1<f32>> = self.temporal_network.all_tau_owned();
        let tau_refs: Vec<&ndarray::Array1<f32>> = tau_owned.iter().collect();
        self.voice_coherence.bridge.update(&tau_refs);

        // 10b. Update temporal signature encoder
        let flattened_tau: Vec<f32> = tau_owned.iter().flat_map(|a| a.iter().copied()).collect();
        self.voice_coherence.temporal.record_batch(&flattened_tau);

        // 10c. Update adaptive behavior
        let (pattern, pattern_confidence) = self.voice_coherence.temporal.classify_state();
        let coherence = self.voice_coherence.bridge.smoothed_coherence();
        self.carryover.history.cached_coherence = Some(coherence);

        // Voice feedback heartbeat: synthesize metrics from cognitive state
        // to keep the voice→cognition loop alive (Liberman & Mattingly 1985).
        // NOTE: This is a synthetic proxy — real voice output metrics will replace
        // these when the vocal tract pipeline is wired into the cycle (Phase 28+).
        // Until then, this provides a non-trivial self-referential signal that
        // keeps AdaptiveBehavior.confidence responsive to cognitive coherence.
        let voice_heartbeat = crate::voice::VoiceOutputMetrics {
            articulation_score: coherence.clamp(0.0, 1.0),
            formant_accuracy: (1.0 - prediction_error).clamp(0.0, 1.0),
            speech_rate: 4.0 * self.adaptive_behavior.speech_rate_multiplier,
            pitch_stability: pattern_confidence,
            coarticulation_smoothness: coherence.clamp(0.0, 1.0) * 0.8,
            listener_prediction: if prediction_error < self.config.learning_threshold {
                0.8
            } else {
                0.3
            },
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
        };
        self.voice_coherence.voice.update(voice_heartbeat);

        let voice_confidence = self.voice_coherence.voice.summary().voice_confidence;
        self.adaptive_behavior = AdaptiveBehavior::from_consciousness_state(
            pattern,
            pattern_confidence,
            coherence,
            voice_confidence,
        );

        self.reapply_strategy_modulation(selected_strategy);

        self.adaptive_behavior.attention_sensitivity *= goal_attention_bias;
        if wm_sensory_mismatch {
            self.adaptive_behavior.attention_sensitivity *= 1.08;
            // Sensory-abstract mismatch → slow consolidation + dampen confidence.
            // Hierarchical decomposition is breaking → protect abstract representations.
            // Science: Friston (2010) — hierarchical level misalignment = high free energy.
            self.scale_lr("wm_sensory_mismatch", WM_MISMATCH_LR_SCALE);
            self.scale_confidence("wm_sensory_mismatch", WM_MISMATCH_CONFIDENCE_SCALE);
        }

        // 10d. Update prediction confidence
        self.update_prediction_confidence(pattern, prediction_error, pattern_confidence);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.5 Active Inference Bridge
        // ═══════════════════════════════════════════════════════════════════════
        let prediction_success = prediction_error < self.config.learning_threshold;
        self.fep
            .active_inference_bridge
            .observe_resolution(self.prediction_confidence, prediction_success);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6 FEP Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        let effective_lr = self.stats.adaptive_learning_rate;
        let (fep_action_idx, fep_action_probs, is_surprised, fep_pragmatic_value_raw) =
            self.step_fep_active_inference(prediction_error, coherence);

        // ── Cross-manifold prediction error → attention reallocation ──
        // Rao & Ballard (1999): large prediction errors between visual and cognitive
        // streams indicate world-model mismatch → re-engage attention, update model.
        #[cfg(feature = "vision-manifold")]
        {
            use super::thresholds::{
                CROSS_MANIFOLD_CONFIDENCE_DAMPEN, CROSS_MANIFOLD_ERROR_THRESHOLD,
                CROSS_MANIFOLD_EXPLORATION_SCALE, CROSS_MANIFOLD_LR_BOOST,
            };
            let cm_error = perception.cross_manifold_prediction_error;
            if cm_error > CROSS_MANIFOLD_ERROR_THRESHOLD {
                let excess = cm_error - CROSS_MANIFOLD_ERROR_THRESHOLD;
                self.adjust_exploration(
                    "cross_manifold_error",
                    excess * CROSS_MANIFOLD_EXPLORATION_SCALE,
                );
                self.scale_confidence("cross_manifold_error", CROSS_MANIFOLD_CONFIDENCE_DAMPEN);
                self.scale_lr("cross_manifold_error", CROSS_MANIFOLD_LR_BOOST);
            }
        }

        // ── Vision temporal horizons → FEP modulation ────────────────────
        // Adams et al. (2013): prediction errors at multiple timescales drive
        // hierarchical active inference. Short-horizon errors = immediate surprise;
        // long-horizon errors = planning uncertainty.
        #[cfg(feature = "vision-manifold")]
        if !perception.vision_horizon_errors.is_empty() {
            use super::thresholds::{
                VISION_HORIZON_CONFIDENCE_DAMPEN, VISION_HORIZON_EXPLORATION_SCALE,
                VISION_LONG_HORIZON_CONFIDENCE_THRESHOLD, VISION_SHORT_HORIZON_ERROR_THRESHOLD,
            };
            // Short-term error (next frame, ~33ms) → immediate surprise
            let short_err = perception.vision_horizon_errors[0];
            if short_err > VISION_SHORT_HORIZON_ERROR_THRESHOLD {
                let boost = (short_err - VISION_SHORT_HORIZON_ERROR_THRESHOLD)
                    * VISION_HORIZON_EXPLORATION_SCALE;
                self.adjust_exploration("vision_horizon_short", boost);
            }

            // Long-term error (500ms+, index 2) → planning uncertainty
            if let Some(&long_err) = perception.vision_horizon_errors.get(2) {
                if long_err > VISION_LONG_HORIZON_CONFIDENCE_THRESHOLD {
                    self.scale_confidence("vision_horizon_long", VISION_HORIZON_CONFIDENCE_DAMPEN);
                }
            }
        }

        // ── Track 5b: MCTS plan post-hoc evaluation ─────────────────────────
        let mcts_plan_effectiveness: f32 =
            if let Some((_prev_action, _prev_confidence, prev_error)) =
                self.carryover.history.mcts_plan_applied.take()
            {
                let error_reduction = prev_error - prediction_error;
                let raw_effectiveness = if prev_error > 0.0 {
                    (error_reduction / prev_error).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let effectiveness = (raw_effectiveness * 0.5 + 0.5).clamp(0.0, 1.0);
                if effectiveness > MCTS_EFFECTIVENESS_HIGH {
                    self.adjust_confidence(
                        "mcts_effective",
                        (effectiveness - MCTS_EFFECTIVENESS_HIGH)
                            * MCTS_EFFECTIVENESS_CONFIDENCE_SCALE,
                    );
                } else if effectiveness < MCTS_EFFECTIVENESS_LOW {
                    self.adjust_exploration(
                        "mcts_poor_plan",
                        (MCTS_EFFECTIVENESS_LOW - effectiveness) * MCTS_EFFECTIVENESS_EXPLORE_SCALE,
                    );
                }
                self.stats.avg_mcts_plan_effectiveness = self.stats.avg_mcts_plan_effectiveness
                    * MCTS_EFFECTIVENESS_EMA
                    + effectiveness * (1.0 - MCTS_EFFECTIVENESS_EMA);
                effectiveness
            } else {
                0.0
            };

        // ── Apply previous cycle's MCTS plan ──────────────
        if let Some((plan_action, plan_confidence)) = self.carryover.history.mcts_plan.take() {
            if plan_confidence > MCTS_PLAN_CONFIDENCE_THRESHOLD && plan_action != fep_action_idx {
                self.carryover.history.mcts_plan_applied =
                    Some((plan_action, plan_confidence, prediction_error));
                let plan_weight = plan_confidence * MCTS_PLAN_WEIGHT_SCALE;
                match plan_action {
                    0 => {
                        self.scale_lr("mcts_exploit", 1.0 - plan_weight * MCTS_EXPLOIT_LR_SCALE);
                    }
                    1 => {
                        self.adjust_confidence(
                            "mcts_consolidate",
                            plan_weight * MCTS_CONSOLIDATE_CONFIDENCE_SCALE,
                        );
                    }
                    2 => {
                        self.adjust_exploration(
                            "plan_explore_directive",
                            plan_weight * MCTS_EXPLORE_SCALE,
                        );
                    }
                    _ => {}
                }
            }
        }

        // ── FEP Free Energy Decomposition ──────────────
        let fep_vals = self
            .fep
            .agent
            .last_fe_components
            .as_ref()
            .map(|fe| (fe.accuracy, fe.complexity, fe.surprise, fe.prediction_error));
        let (fep_accuracy, fep_complexity, fep_surprise, fep_td_error) =
            if let Some((acc, comp, surp, pe)) = fep_vals {
                if acc > FEP_ACCURACY_CONFIDENCE_THRESHOLD {
                    self.adjust_confidence("fep_accuracy_high", 0.01);
                }
                if comp > FEP_COMPLEXITY_THRESHOLD {
                    self.scale_lr(
                        "fep_complexity",
                        1.0 - ((comp - FEP_COMPLEXITY_THRESHOLD).min(0.5) * 0.1) as f32,
                    );
                }
                if surp > reflection_thresholds.surprise as f64 {
                    let s_explore =
                        ((surp - reflection_thresholds.surprise as f64) * 0.1).min(0.05) as f32;
                    self.adjust_exploration("reflection_surprise", s_explore);
                }
                (acc, comp, surp, pe)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        let fep_pragmatic_value = fep_pragmatic_value_raw;
        if fep_pragmatic_value > FEP_PRAGMATIC_EXPLOIT_THRESHOLD {
            self.scale_exploration(
                "fep_pragmatic_exploit",
                (1.0 - (fep_pragmatic_value - FEP_PRAGMATIC_EXPLOIT_THRESHOLD) * 0.3) as f32,
            );
        } else if fep_pragmatic_value < FEP_PRAGMATIC_EXPLORE_THRESHOLD && fep_pragmatic_value > 0.0
        {
            let p_explore =
                ((FEP_PRAGMATIC_EXPLORE_THRESHOLD - fep_pragmatic_value) * 0.15).min(0.05) as f32;
            self.adjust_exploration("fep_pragmatic_low", p_explore);
        }

        if fep_td_error.abs() > FEP_TD_ERROR_DISCOVERY_THRESHOLD {
            if let Some(ref mut enhancer) = self.causal_enhancer {
                if enhancer.should_discover() {
                    let _ = enhancer.run_discovery();
                }
            }
            self.carryover.quality.consecutive_low_td_error = 0;
        } else if fep_td_error.abs() < 0.01 {
            self.carryover.quality.consecutive_low_td_error = self
                .carryover
                .quality
                .consecutive_low_td_error
                .saturating_add(1);
        }
        // Session 13 Item 3: Sustained low TD error → model converged → dampen exploration.
        // Science: Sutton & Barto (2018) — convergent TD signals indicate policy stability.
        if self.carryover.quality.consecutive_low_td_error > 10 && self.stats.total_cycles > 30 {
            self.scale_exploration("fep_td_converged", 0.97);
        }

        // ── Track 5e: Causal graph → attention weighting ─────────────────
        let causal_attention_edges: usize = if let Some(ref enhancer) = self.causal_enhancer {
            let graph = enhancer.current_graph();
            let edge_count = graph.edges.len();
            if edge_count > 0 {
                let avg_confidence = if edge_count > 0 {
                    graph.edges.iter().map(|e| e.confidence).sum::<f64>() / edge_count as f64
                } else {
                    0.0
                };
                if edge_count > 5 && avg_confidence > 0.5 {
                    self.adjust_confidence(
                        "causal_graph_dense",
                        (avg_confidence as f32 - 0.5) * 0.03,
                    );
                } else if edge_count >= 3 && avg_confidence > 0.4 {
                    // Session 13 Item 1: Fill dead zone for moderate causal density.
                    // 3-5 edges with decent confidence = emerging structure → small boost.
                    // Science: Pearl (2000) — partial causal knowledge still informative.
                    self.adjust_confidence(
                        "causal_graph_emerging",
                        (avg_confidence as f32 - 0.4) * 0.01,
                    );
                }
                if edge_count < 2 && self.stats.total_cycles > 200 {
                    self.adjust_exploration("sparse_causal_graph", 0.02);
                }
                self.stats.causal_attention_uses += 1;
            }
            edge_count
        } else {
            0
        };

        // ── FEP decomposition → adaptive behavior modulation ─────────────
        if fep_accuracy > 0.5 && fep_complexity < 0.5 {
            self.adaptive_behavior.learning_rate_multiplier =
                (self.adaptive_behavior.learning_rate_multiplier * 1.1).min(2.0);
            self.adaptive_behavior.exploration_factor *= 0.8;
            // Session 13 Item 6: Wire FEP efficiency into proposal system.
            // High accuracy + low complexity = efficient model → boost confidence.
            // Science: Friston (2010) — low complexity = good model evidence.
            self.adjust_confidence("fep_efficient", 0.01);
        }
        let surprise_thresh = reflection_thresholds.surprise as f64;
        if fep_surprise > surprise_thresh {
            self.adaptive_behavior.exploration_factor =
                (self.adaptive_behavior.exploration_factor + 0.15).min(1.0);
            self.adaptive_behavior.action_hint = ActionHint::Explore;
        }
        if fep_complexity > 1.0 {
            self.adaptive_behavior.learning_rate_multiplier =
                (self.adaptive_behavior.learning_rate_multiplier * 0.85).max(0.1);
            self.adaptive_behavior.pause_multiplier =
                (self.adaptive_behavior.pause_multiplier * 1.2).min(2.0);
            self.adaptive_behavior.action_hint = ActionHint::SlowDown;
        }

        if fep_surprise > surprise_thresh {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                let surprise_boost = (fep_surprise - surprise_thresh).min(0.5) * 0.2;
                replay.boost_recent_consolidation(surprise_boost);
            }
        }

        if self.social_mgr.social.external_reward.abs() > f32::EPSILON {
            let outcome_obs = Observation::from_consciousness_state(
                self.social_mgr.social.external_reward as f64,
                coherence as f64,
                self.prediction_confidence,
                effective_lr as f64,
            );
            self.fep
                .agent
                .learn_from_outcome(fep_action_idx, &outcome_obs);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Moral Modulation of Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        let (moral_steering_category, pfe_surprise_mod) = self.apply_moral_modulation(
            moral_concern_detected,
            &perception.moral.moral_judgment,
            perception.moral.moral_score,
            is_surprised,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH + PSI SYNTHESIS (extracted to cycle_neuromod_phase.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let neuromod_result = self.run_neuromodulator_and_psi_phase(prediction_error, coherence);
        let ne_reorienting_boost = neuromod_result.ne_reorienting_boost;
        let ne_arousal_feedback = neuromod_result.ne_arousal_feedback;
        let sht_crash_dip = neuromod_result.sht_crash_dip;
        let exploration_sht_drain = neuromod_result.exploration_sht_drain;
        let confidence_velocity = neuromod_result.confidence_velocity;
        // Session 13 Item 4: Rising confidence → dampen exploration.
        // Positive velocity = model converging → exploit learned knowledge.
        // Science: Daw et al. (2006) — confidence trajectory gates explore/exploit trade-off.
        if confidence_velocity > 0.02 && self.stats.total_cycles > 15 {
            let dampen = (1.0 - confidence_velocity * 0.1).max(0.95);
            self.scale_exploration("confidence_rising", dampen);
        }
        // Falling confidence → speed up learning (model needs correction).
        // Confidence collapse signals prediction degradation → recalibrate faster.
        // Science: Cools et al. (2008) — rapid confidence decline triggers
        // serotonergic recalibration and increased learning rate.
        if confidence_velocity < -0.05 && self.stats.total_cycles > 15 {
            let boost = (1.0 + (-confidence_velocity - 0.05) * 0.3).min(1.15);
            self.scale_lr("confidence_falling", boost);
        }
        let unified_psi = neuromod_result.unified_psi;
        let guiding_question = neuromod_result.guiding_question;
        let dominant_harmonic = neuromod_result.dominant_harmonic;
        let guiding_priority_category = neuromod_result.guiding_priority_category;

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6b Enhanced FEP Bridge
        // ═══════════════════════════════════════════════════════════════════════
        let run_enhanced = surprise_triggered
            || is_surprised
            || prediction_error > self.config.learning_threshold
            || urgency.should_run(self.stats.total_cycles, 1, 4, 8);
        let enhanced_result = if run_enhanced {
            let r = self.fep.enhanced_bridge.cycle(
                prediction_error as f64,
                coherence as f64,
                self.prediction_confidence,
                effective_lr as f64,
            );
            self.fep.learning_signal = r.learning_signal as f32;
            Some(r)
        } else {
            None
        };

        if let Some(ref er) = enhanced_result {
            self.stats.fep_action_outcome_coupling = er.action_outcome_coupling as f32;
        }

        if let Some(ref enhanced_result) = enhanced_result {
            match enhanced_result.motor_command.command_type {
                MotorCommandType::AttentionShift => {
                    let shift_amount = enhanced_result.motor_command.intensity as f32 * 0.1;
                    self.adaptive_behavior.attention_sensitivity =
                        (self.adaptive_behavior.attention_sensitivity * (1.0 + shift_amount * 0.1))
                            .clamp(0.5, 2.0);
                    self.stats.attention_shift = shift_amount;
                }
                MotorCommandType::LearningRateAdjust => {
                    if enhanced_result.should_learn {
                        let lr_mod = enhanced_result.fep_result.learning_rate_modulation as f32;
                        self.stats.adaptive_learning_rate =
                            (self.stats.adaptive_learning_rate * 0.9 + lr_mod * 0.1)
                                .clamp(0.01, 1.0);
                    }
                }
                MotorCommandType::ExplorationTrigger => {
                    let intensity = enhanced_result.motor_command.intensity as f32;
                    if enhanced_result.fep_result.epistemic_value > 0.5 {
                        // Scale exploration boost by epistemic value
                        let boost = (intensity * 0.15).min(0.2);
                        self.adjust_exploration("motor_exploration_trigger", boost);
                    }
                    // High-intensity exploration → boost learning to absorb novelty
                    if intensity > 0.8 {
                        self.scale_lr("motor_explore_intense", 1.1);
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    let intensity = enhanced_result.motor_command.intensity as f32;
                    if intensity > 0.5 {
                        self.self_model_tier.self_reflection.force_reflection();
                        // Boost meta-awareness proportional to intensity
                        self.adjust_confidence("motor_reflection", (intensity - 0.5) * 0.05);
                    }
                }
                MotorCommandType::MemoryConsolidate => {
                    if enhanced_result.motor_command.intensity > 0.5 {
                        self.fep.episodic_memory.consolidate_recent();
                        // Also increase world model plasticity to lock in patterns
                        self.fep
                            .world_model
                            .increase_plasticity(enhanced_result.motor_command.intensity as f32);
                    }
                }
                MotorCommandType::ExpectationReset => {
                    if enhanced_result.action_outcome_coupling < 0.3 {
                        self.last_prediction = None;
                        self.set_confidence("inference_mode_init", 0.5);
                        // Reset world model levels to accept new patterns
                        self.fep.world_model.reset();
                    }
                }
                MotorCommandType::MotorOutput | MotorCommandType::NoOp => {}
            }

            if self.fep.learning_signal > FEP_LEARNING_PLASTICITY_THRESHOLD
                && enhanced_result.should_learn
            {
                self.fep
                    .world_model
                    .increase_plasticity(self.fep.learning_signal);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Coherence tracking with degradation detection
        // ═══════════════════════════════════════════════════════════════════════
        let degraded = self.coherence_tracker.record_turn(coherence);
        if degraded {
            self.scale_lr("coherence_degraded", 1.3);
            let coh_urgency = self.coherence_tracker.correction_urgency();
            let urgent_obs = Observation::from_consciousness_state(
                coh_urgency as f64,
                0.1,
                0.1,
                effective_lr as f64,
            );
            self.fep.agent.perceive(&urgent_obs);
            self.fep
                .enhanced_bridge
                .cycle(coh_urgency as f64, 0.1, 0.1, effective_lr as f64);
        }

        // 10e–g. Update flow state, curiosity drive, self-reflection
        self.update_drives_and_reflection(pattern, prediction_error, coherence);

        // Unified Psi + Experience Bus + Guiding Question computed by
        // run_neuromodulator_and_psi_phase() above — values already in neuromod_result.

        // ── Phase 15: Attention budget check ─────────────────────────────────
        let neuromod_attention_alloc = self.neuromod.bath.attention_budget_allocation();
        // Thalamic depth → attention budget scaling
        // Science: Kahneman (1973) — deliberation allocates more attentional resources
        let depth_budget_scale = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => THALAMIC_DEEP_BUDGET_SCALE,
            super::CognitiveDepth::Cortical => 1.0,
            super::CognitiveDepth::Reflex => THALAMIC_REFLEX_BUDGET_SCALE,
        };
        // Epistemic uncertainty → attention budget expansion.
        // Science: Gottlieb et al. (2013) — uncertain environments demand more attentional resources.
        // High epistemic (>0.4) scales budget up to 1.3×; low (<0.2) contracts to 0.9×.
        let epistemic_budget_scale = if epistemic_uncertainty > 0.4 {
            1.0 + (epistemic_uncertainty - 0.4).min(0.3)
        } else if epistemic_uncertainty < 0.2 {
            0.9 + epistemic_uncertainty * 0.5 // 0.9 at 0.0, 1.0 at 0.2
        } else {
            1.0
        };
        // Sacred Stillness → attention budget contraction: when the dominant
        // harmony is rest/stillness, reduce computation budget (genuine rest).
        // Science: Raichle (2010) — default mode network reduces task-positive
        // resource allocation during rest states.
        let stillness_budget_scale = {
            let ss_coord = self.ethics_engine.last_harmony_coordinates()[7]; // SacredStillness
            if ss_coord > 0.5 {
                // High stillness activation → contract budget by up to 30%
                1.0 - (ss_coord - 0.5).min(0.3)
            } else {
                1.0
            }
        };
        // Coherence velocity → attention budget allocation.
        // Dropping coherence → preserve budget (system losing grip);
        // rising coherence → expand budget (model confidence growing).
        // Science: Bar (2009) — coherence collapse demands attention reallocation.
        let coherence_velocity_budget_scale = {
            let cv = self.carryover.quality.coherence_velocity;
            if cv < -COHERENCE_VELOCITY_BUDGET_THRESHOLD {
                COHERENCE_VELOCITY_BUDGET_CONTRACT
            } else if cv > COHERENCE_VELOCITY_BUDGET_THRESHOLD {
                COHERENCE_VELOCITY_BUDGET_EXPAND
            } else {
                1.0
            }
        };
        let attention_budget_us = (ATTENTION_BUDGET_US as f64
            * neuromod_attention_alloc as f64
            * depth_budget_scale
            * epistemic_budget_scale as f64
            * stillness_budget_scale
            * coherence_velocity_budget_scale) as u64;

        // Active Rest Mode: track Sacred Stillness dominance streak
        {
            let ss_coord = self.ethics_engine.last_harmony_coordinates()[7];
            let dominant_idx = self
                .ethics_engine
                .last_harmony_coordinates()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if dominant_idx == 7 && ss_coord > 0.3 {
                self.stats.stillness_dominance_streak += 1;
            } else {
                self.stats.stillness_dominance_streak = 0;
                self.stats.in_active_rest = false;
            }
            if self.stats.stillness_dominance_streak >= super::thresholds::ACTIVE_REST_THRESHOLD {
                if !self.stats.in_active_rest {
                    tracing::info!(
                        streak = self.stats.stillness_dominance_streak,
                        "Entering active rest mode — redirecting computation to consolidation"
                    );
                    self.stats.in_active_rest = true;
                }
                // Active rest: trigger dream consolidation and memory defragmentation
                if let Some(ref mut dream) = self.dream_engine {
                    if let Ok(result) = dream.dream() {
                        if result.insights > 0 {
                            tracing::debug!(
                                insights = result.insights,
                                "Active rest dream consolidation"
                            );
                        }
                    }
                }
                // Active rest: boost episodic memory consolidation priority.
                // Rest states are when the brain consolidates recent experiences
                // (Diekelmann & Born 2010 — memory consolidation during rest).
                if let Some(ref mut replay) = self.phi_episodic_replay {
                    replay.boost_recent_consolidation(0.15);
                }
            }
        }

        // Moral attractor dampening: on the rising edge of attractor detection,
        // reduce exploration rate by 20% — the system has settled on an ethical stance.
        // Only fires once per attractor entry to prevent exponential decay to floor.
        {
            let attractor_now = self
                .ethics_engine
                .moral_topology()
                .last_summary()
                .attractor_detected;
            if attractor_now && !self.stats.prev_moral_attractor {
                let rate = self.fep.closed_learning_loop.exploration_rate();
                self.fep
                    .closed_learning_loop
                    .set_exploration_rate(rate * 0.8);
            }
            self.stats.prev_moral_attractor = attractor_now;
        }

        let attention_budget_elapsed_us = cycle_start.elapsed().as_micros() as u64;
        let attention_budget_exceeded = attention_budget_elapsed_us > attention_budget_us;
        if attention_budget_exceeded {
            self.stats.attention_budget_exceeded_count += 1;
            // Session 13 Item 7: Budget exceeded → raise threshold (be more selective).
            // Overloaded system should require stronger signals before full processing.
            // Science: Lavie (2005) — perceptual load theory: high load raises selection threshold.
            if self.stats.attention_budget_exceeded_count > 1 {
                self.scale_threshold("attention_overload", 1.1);
            }
            if self.stats.attention_budget_exceeded_count > 3 {
                tracing::warn!(
                    elapsed_us = attention_budget_elapsed_us,
                    consecutive = self.stats.attention_budget_exceeded_count,
                    "Cycle budget exceeded for {} consecutive cycles",
                    self.stats.attention_budget_exceeded_count,
                );
            }
        } else {
            self.stats.attention_budget_exceeded_count = 0;
        }

        let predictive_budget_gated = attention_budget_elapsed_us
            > (attention_budget_us as f64 * PREDICTIVE_BUDGET_GATING_RATIO) as u64
            && !attention_budget_exceeded;
        if predictive_budget_gated {
            self.stats.predictive_budget_gated_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.0 PsiAttestation record
        // ═══════════════════════════════════════════════════════════════════════
        if self.config.enable_psi_attestation && self.config.agent_did.is_some() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            let record = super::PsiAttestationRecord {
                psi: unified_psi,
                cycle_id: self.stats.total_cycles as u64,
                captured_at_us: now,
                prediction_error,
                urgency,
            };
            while self.psi_attestation_buffer.len() >= self.config.attestation_buffer_capacity {
                let _ = self.psi_attestation_buffer.pop_front();
            }
            self.psi_attestation_buffer.push_back(record);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1 Conscious Reasoning Engine
        // ═══════════════════════════════════════════════════════════════════════
        #[allow(unused_mut)]
        let mut reasoning_confidence: f32 = 0.0;
        #[allow(unused_mut)]
        let mut reasoning_lr_factor: f32 = 1.0;
        #[allow(unused_mut)]
        let mut reasoning_gate_blocked: bool = false;
        #[allow(unused_mut)]
        let mut reasoning_fallback: Option<String> = None;
        #[allow(unused_mut)]
        let mut reasoning_plan_action: Option<usize> = None;
        #[allow(unused_mut)]
        let mut reasoning_plan_confidence: f32 = 0.0;
        #[allow(unused_mut)]
        let mut reasoning_narrative: Option<String> = None;
        // Used by reasoning_engine gate below; suppress warning when feature is off.
        #[allow(unused_variables)]
        let substrate_degraded = self.substrate_manager.should_degrade_consciousness();
        #[cfg(feature = "reasoning_engine")]
        if !substrate_degraded {
            if let Some(ref mut reasoning_engine) = self.reasoning_engine {
                use crate::consciousness::epistemic_conflict::MultiTheoryMetrics as ECMetrics;
                use crate::consciousness::reasoning_engine::ReasoningContext;

                let ec_metrics = ECMetrics {
                    phi: unified_psi,
                    gwt: coherence as f64,
                    ast: self.prediction_confidence,
                    pp: (1.0 - prediction_error as f64).clamp(0.0, 1.0),
                    rpt: pattern_confidence as f64,
                    embodiment: self.fep.learning_signal as f64,
                    unified: unified_psi,
                };

                let elapsed_us = cycle_start.elapsed().as_micros() as u64;
                let available_us = 20_000u64.saturating_sub(elapsed_us);

                // Populate available actions for MCTS planning.
                // When the code_generation feature is enabled, include code-specific
                // actions so the reasoning engine can plan code tasks.
                #[allow(unused_mut)]
                let mut actions: Vec<
                    crate::consciousness::temporal_planning::types::PlannedAction,
                > = Vec::new();

                #[cfg(feature = "code_generation")]
                {
                    use crate::consciousness::temporal_planning::types::PlannedAction;
                    actions.extend([
                        PlannedAction {
                            id: "code_generate".to_string(),
                            description: "Generate code from specification".to_string(),
                            embedding: vec![0.8, 0.2, 0.1, 0.9],
                            prior: 0.3,
                            is_epistemic: false,
                        },
                        PlannedAction {
                            id: "code_verify".to_string(),
                            description: "Verify generated code via compilation".to_string(),
                            embedding: vec![0.6, 0.4, 0.3, 0.7],
                            prior: 0.2,
                            is_epistemic: true,
                        },
                        PlannedAction {
                            id: "code_refactor".to_string(),
                            description: "Refactor code for clarity or performance".to_string(),
                            embedding: vec![0.5, 0.5, 0.6, 0.4],
                            prior: 0.15,
                            is_epistemic: false,
                        },
                        PlannedAction {
                            id: "code_explain".to_string(),
                            description: "Explain code structure and intent".to_string(),
                            embedding: vec![0.3, 0.7, 0.8, 0.2],
                            prior: 0.15,
                            is_epistemic: true,
                        },
                        PlannedAction {
                            id: "code_debug".to_string(),
                            description: "Debug and diagnose code issues".to_string(),
                            embedding: vec![0.4, 0.6, 0.5, 0.5],
                            prior: 0.2,
                            is_epistemic: true,
                        },
                    ]);
                }

                let reasoning_ctx = ReasoningContext {
                    theory_metrics: ec_metrics,
                    phi: unified_psi,
                    available_budget_us: available_us,
                    available_actions: actions,
                    tool: None,
                    recent_utility: 0.5,
                    cycle_id: self.stats.total_cycles as u64,
                    neuromod_exploration_mod: self.neuromod.bath.mcts_exploration_modulation(),
                };

                let reasoning_result = reasoning_engine.reason(&reasoning_ctx);

                reasoning_confidence = reasoning_result.phi_eff as f32;
                reasoning_lr_factor = reasoning_result.reliability as f32;

                if let Some(ref gate) = reasoning_result.gate {
                    if !gate.is_allowed() {
                        reasoning_gate_blocked = true;
                        reasoning_fallback = gate.fallback.as_ref().map(|f| format!("{:?}", f));
                        reasoning_lr_factor = 0.0;
                        tracing::info!(
                            risk = ?gate.risk_level,
                            required_phi = gate.required_phi,
                            actual_phi = gate.actual_phi_eff,
                            "Reasoning gate blocked action"
                        );
                    }
                }

                if let Some(ref plan) = reasoning_result.plan {
                    if plan.did_plan {
                        reasoning_plan_action = plan.best_action_idx;
                        reasoning_plan_confidence = plan.confidence as f32;
                    }
                }

                reasoning_narrative = reasoning_result.narrative.clone();

                tracing::debug!(
                    tier = ?reasoning_result.tier,
                    phi_eff = reasoning_result.phi_eff,
                    reliability = reasoning_result.reliability,
                    gate_blocked = reasoning_gate_blocked,
                    plan_confidence = reasoning_plan_confidence,
                    wall_time_us = reasoning_result.wall_time_us,
                    budget_exceeded = reasoning_result.budget_exceeded,
                    "Reasoning engine cycle"
                );
            }
        } // if !substrate_degraded (reasoning_engine)

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.2 KL-divergence policy gate + adaptive temperature
        // ═══════════════════════════════════════════════════════════════════════
        #[allow(unused_assignments)]
        let mut policy_agreement = false;
        if let Some(mcts_idx) = reasoning_plan_action {
            let fep_prob_for_mcts = fep_action_probs.get(mcts_idx).copied().unwrap_or(0.0);
            if mcts_idx == fep_action_idx {
                reasoning_plan_confidence =
                    (reasoning_plan_confidence * POLICY_FULL_AGREEMENT_BOOST).min(1.0);
                policy_agreement = true;
            } else if fep_prob_for_mcts > POLICY_SOFT_THRESHOLD {
                policy_agreement = true;
                reasoning_plan_confidence =
                    (reasoning_plan_confidence * (1.0 + fep_prob_for_mcts as f32 * 0.3)).min(1.0);
            } else {
                let dampen = (0.3 + fep_prob_for_mcts * 0.7) as f32;
                self.fep.learning_signal *= dampen;
                reasoning_plan_confidence *= dampen;
            }

            if self.policy_agreement_window.len() >= POLICY_WINDOW_SIZE {
                self.policy_agreement_window.pop_front();
            }
            self.policy_agreement_window.push_back(policy_agreement);
            if self.policy_agreement_window.len() >= POLICY_MIN_WINDOW {
                let agree_rate = self.policy_agreement_window.iter().filter(|&&a| a).count() as f64
                    / self.policy_agreement_window.len() as f64;
                let adaptive_temp = POLICY_TEMP_BASE + (1.0 - agree_rate) * POLICY_TEMP_RANGE;
                self.fep.agent.config.action_temperature = adaptive_temp;
            }
        }

        self.carryover.history.mcts_plan =
            reasoning_plan_action.map(|a| (a, reasoning_plan_confidence));

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1.5 Metacognitive Monitoring
        // ═══════════════════════════════════════════════════════════════════════
        let mut metacognitive_anomaly = false;
        let mut anomaly_recovery_progress: f32 = 0.0;
        let anomaly_recovering;
        if let Some(ref mut monitor) = self.metacognitive_monitor {
            if monitor.observe_phi(unified_psi) {
                metacognitive_anomaly = true;
                reasoning_lr_factor *= 0.5;
                self.carryover.urgency.anomaly_recovery_counter = 0;
                self.carryover.urgency.anomaly_was_active = true;
                tracing::debug!(
                    target: "cognitive_loop::metacognition",
                    unified_psi,
                    "Metacognitive anomaly detected — dampening learning rate"
                );
            }
        }

        if !metacognitive_anomaly && self.carryover.urgency.anomaly_was_active {
            self.carryover.urgency.anomaly_recovery_counter = self
                .carryover
                .urgency
                .anomaly_recovery_counter
                .saturating_add(1);
            let counter = self.carryover.urgency.anomaly_recovery_counter;
            if counter <= 20 {
                let recovery = counter as f32 / 20.0;
                reasoning_lr_factor *= 0.5 + recovery * 0.5;
                anomaly_recovery_progress = recovery;
                self.stats.anomaly_recovery_active_count += 1;
            } else {
                self.carryover.urgency.anomaly_was_active = false;
                anomaly_recovery_progress = 1.0;
            }
            anomaly_recovering = counter <= 20;
        } else {
            anomaly_recovering = false;
        }

        // Compose effective LR
        let effective_lr = self.compose_effective_lr(semantic_lr_factor, reasoning_lr_factor);
        let effective_lr = effective_lr * self.neuromod.bath.gradient_scale_factor();
        let effective_lr = effective_lr * self.neuromod.bath.plasticity_gate();
        let effective_lr = if self.neuromod.bath.sleep_pressure() > 0.7 {
            let pressure_factor = 1.0 - (self.neuromod.bath.sleep_pressure() - 0.7) * 0.5;
            effective_lr * pressure_factor.clamp(0.5, 1.0)
        } else {
            effective_lr
        };

        // Thalamic depth → learning rate gating
        // Science: Aston-Jones & Cohen (2005) — phasic NE (exploitation/learning) vs tonic (exploration)
        // DeepThought enhances learning; Reflex minimizes it for cached patterns
        let effective_lr = effective_lr
            * match self.cognitive_depth {
                super::CognitiveDepth::DeepThought => THALAMIC_DEEP_LR_FACTOR,
                super::CognitiveDepth::Cortical => 1.0,
                super::CognitiveDepth::Reflex => THALAMIC_REFLEX_LR_FACTOR,
            };

        let neuromod_threshold =
            perception.encoding.effective_threshold * self.neuromod.bath.threshold_gate();

        // 11. Learn if error is significant
        let _t_core = Instant::now();
        let consciousness_awake =
            self.carryover.history.consciousness_level > 0.0 || self.stats.total_cycles < 20;
        let (learning_occurred, training_loss) = if prediction_error > neuromod_threshold
            && !self.adaptive_behavior.pause_learning
            && !self.carryover.quality.narrative_veto_active
            && consciousness_awake
        {
            self.stats.learning_cycles += 1;

            let (train_input, train_target, lr) = if let Some(prev) = previous_state {
                (
                    Array1::from_vec(prev),
                    perception
                        .encoding
                        .compressed_state
                        .iter()
                        .copied()
                        .collect(),
                    effective_lr,
                )
            } else {
                let train_input: Array1<f32> = perception
                    .encoding
                    .compressed_state
                    .iter()
                    .copied()
                    .collect();
                let train_target: Array1<f32> = perception
                    .encoding
                    .compressed_state
                    .iter()
                    .copied()
                    .collect();
                (train_input, train_target, effective_lr * 0.1)
            };

            // Compute vision-surprise importance weight for training
            #[cfg(feature = "vision-manifold")]
            let importance = (TRAINING_BASE_IMPORTANCE
                + perception.cross_manifold_prediction_error * VISION_TRAINING_IMPORTANCE_SCALE
                + perception.vision_mean_surprise * VISION_SURPRISE_TRAINING_IMPORTANCE_SCALE)
                .min(TRAINING_MAX_IMPORTANCE);
            #[cfg(not(feature = "vision-manifold"))]
            let importance = TRAINING_BASE_IMPORTANCE;

            if let Some(ref trainer) = self.async_trainer {
                trainer.send(TrainingSample {
                    input: train_input,
                    target: train_target,
                    dt: delta_t,
                    learning_rate: lr,
                    method: self.config.training_method,
                    avg_loss: self.stats.avg_training_loss,
                    importance,
                });
                (true, None)
            } else {
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
        module_timings.core_training = _t_core.elapsed().as_micros() as u64;

        // #13: Report learning activity to glutamate channel
        {
            let is_night =
                self.biorhythm_mgr.rhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.neuromod
                .bath
                .report_learning(effective_lr, prediction_error, is_night);
            let fatigue = self.neuromod.bath.learning_fatigue_factor();
            if fatigue < 1.0 {
                self.scale_lr("glutamate_fatigue", fatigue);
            }
        }

        // Goal←Cognition feedback
        if !learning_occurred && self.carryover.urgency.consecutive_low_error > 5 {
            if let Some(top) = self.fep.goal_system.top_goal() {
                let top_id = top.id.clone();
                let delta = (0.01 * (1.0 + self.prediction_confidence * 0.5)) as f32;
                self.fep.goal_system.update_progress(&top_id, delta);
            }
        }

        // 12. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());

        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        let consciousness_resize_factor =
            1.0 + (self.carryover.history.consciousness_level as f32 - 0.5) * 0.3;
        self.temporal_network
            .maybe_resize(prediction_error * consciousness_resize_factor);

        self.stats.temporal_coherence = coherence;
        self.stats.effective_learning_rate = effective_lr;
        self.stats.coherence_phi_contribution = self.voice_coherence.bridge.phi_contribution();

        #[cfg(feature = "school_learning")]
        let school_predicted_phi_gain = if self.stats.total_cycles % 53 == 0 {
            if let Some(ref school) = self.school_bridge {
                school
                    .recommend_next()
                    .ok()
                    .filter(|r| r.predicted_phi_gain > 0.001)
                    .map(|r| r.predicted_phi_gain)
                    .unwrap_or(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        #[cfg(not(feature = "school_learning"))]
        let school_predicted_phi_gain = 0.0f32;

        let causal_attention_boost = if self.stats.total_cycles % 41 == 0 {
            if let Some(ref mut cc) = self.causal_consciousness {
                let vars: Vec<Vec<f64>> = perception
                    .encoding
                    .compressed_state
                    .chunks(8)
                    .map(|chunk: &[f32]| chunk.iter().map(|&v| v as f64).collect())
                    .collect();
                if vars.len() >= 2 {
                    let attention = cc.attention.compute_attention(&vars);
                    let top_strength = attention
                        .iter()
                        .enumerate()
                        .flat_map(|(i, row)| {
                            row.iter()
                                .enumerate()
                                .filter(move |&(j, _)| i != j)
                                .map(|(_, &v)| v)
                        })
                        .fold(0.0f64, f64::max);
                    if top_strength > CAUSAL_ATTENTION_STRENGTH_THRESHOLD as f64 {
                        top_strength as f32
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        if causal_attention_boost > 0.0 {
            // Amplified from ×0.05 to be behaviorally meaningful.
            // Science: Pearl (2000) — strong causal structure justifies confidence.
            self.adjust_confidence(
                "causal_attention",
                causal_attention_boost * CAUSAL_ATTENTION_CONFIDENCE_SCALE,
            );
        }

        // ── Broca SSM language generation + feedback ─────────────────────────
        // Demand-driven: generate only when consciousness is sufficient AND there's
        // novel content worth articulating. Minimum cadence 7 to prevent spam.
        // Biologically: Broca's area activates for speech production when there's
        // something meaningful to express (Hickok & Poeppel 2007).
        #[cfg(feature = "ssm_language")]
        {
            let broca_psi = self.unification_engine.psi as f32;
            let broca_novelty =
                prediction_error > self.config.learning_threshold || surprise_triggered;
            // Adaptive cadence: poor user model → generate more (probe to refine)
            let broca_min_spacing = if self.stats.tom_prediction_mismatch_ema > 0.5 {
                5 // more frequent when user model is inaccurate
            } else {
                7 // default spacing
            };
            let broca_should_generate = broca_psi > 0.4
                && broca_novelty
                && self.stats.total_cycles % broca_min_spacing != 0;
            if broca_should_generate {
                // Generate in a scoped borrow, then apply feedback outside
                let broca_feedback = if let Some(ref mut broca) = self.broca_manager {
                    let signals = super::broca_bridge::BrocaConsciousnessSignals {
                        epistemic_confidence: self.carryover.quality.last_epistemic_confidence,
                        emotional_valence: self.emotion_contagion.prosody_valence(),
                        emotional_arousal: self.emotion_contagion.prosody_arousal(),
                        emotional_warmth: 0.5,
                        consciousness_level: broca_psi,
                        meta_awareness: self.carryover.learning.self_model_accuracy as f32,
                        coherence,
                    };
                    if let Some(result) = broca.generate(signals) {
                        // Surface the generated text for consumers
                        if !result.text.is_empty() {
                            self.last_broca_text = Some(result.text.clone());
                        }

                        // ── Composite quality metric ──
                        #[cfg(feature = "liquid-mamba")]
                        let semantic_pe = result.semantic_pe;
                        #[cfg(not(feature = "liquid-mamba"))]
                        let semantic_pe = 0.0_f32;
                        let broca_quality = result.final_coherence * 0.4
                            + (1.0 - semantic_pe.min(1.0)) * 0.4
                            + result.long_coherence * 0.2;
                        let broca_quality = broca_quality.clamp(0.0, 1.0);

                        // Update quality EMA (alpha = 0.15)
                        self.stats.broca_quality_ema = if self.stats.broca_generation_count == 0 {
                            broca_quality
                        } else {
                            self.stats.broca_quality_ema * 0.85 + broca_quality * 0.15
                        };
                        self.stats.broca_generation_count += 1;

                        // Track low-quality streak for adaptive gating
                        if broca_quality < 0.3 {
                            self.stats.broca_low_quality_streak =
                                self.stats.broca_low_quality_streak.saturating_add(1);
                        } else {
                            self.stats.broca_low_quality_streak = 0;
                        }

                        // Adaptive consciousness gating: raise threshold after
                        // sustained low quality (3+ consecutive poor generations).
                        // Science: Hickok & Poeppel (2007) — speech production
                        // requires sufficient consciousness; poor output → raise bar.
                        if self.stats.broca_low_quality_streak >= 3 {
                            broca.consciousness_threshold =
                                (broca.consciousness_threshold + 0.05).min(0.5);
                        } else if self.stats.broca_quality_ema > 0.7
                            && broca.consciousness_threshold > 0.1
                        {
                            broca.consciousness_threshold =
                                (broca.consciousness_threshold - 0.02).max(0.1);
                        }

                        // Populate telemetry
                        broca.last_telemetry.quality = broca_quality;
                        broca.last_telemetry.long_coherence = result.long_coherence;
                        broca.last_telemetry.semantic_pe = semantic_pe;

                        // Return feedback values to apply after borrow ends
                        #[cfg(feature = "liquid-mamba")]
                        let deferred_semantic_pe = result.semantic_pe;
                        #[cfg(not(feature = "liquid-mamba"))]
                        let deferred_semantic_pe = 0.0_f32;

                        Some((
                            result.final_coherence,
                            broca_quality,
                            result.veto_triggered,
                            deferred_semantic_pe,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Apply deferred feedback outside the broca_manager borrow
                if let Some((final_coherence, broca_quality, veto_triggered, deferred_sem_pe)) =
                    broca_feedback
                {
                    // Coherence feedback: high Broca coherence → confidence boost
                    if final_coherence > 0.7 {
                        self.adjust_confidence("broca_coherent", (final_coherence - 0.7) * 0.03);
                    } else if final_coherence < 0.3 {
                        self.scale_confidence(
                            "broca_incoherent",
                            1.0 - (0.3 - final_coherence) * 0.05,
                        );
                    }

                    // Quality-driven LR modulation: high quality → slight LR boost
                    // Science: successful articulation reinforces associated representations
                    if broca_quality > 0.6 {
                        let lr_boost = 1.0 + (broca_quality - 0.6) * 0.1; // [1.0, 1.04]
                        self.scale_lr("broca_quality", lr_boost);
                    }

                    // Veto feedback: triggered veto → dampen exploration
                    if veto_triggered {
                        self.scale_exploration("broca_veto", 0.95);
                    }

                    // Semantic PE → FEP: language reconstruction error as
                    // additional surprise signal (closes language-perception loop).
                    let _ = deferred_sem_pe; // suppress unused warning when liquid-mamba disabled
                    #[cfg(feature = "liquid-mamba")]
                    if deferred_sem_pe > 0.1 {
                        let sem_obs = Observation::from_consciousness_state(
                            deferred_sem_pe as f64,
                            0.1,
                            coherence as f64,
                            effective_lr as f64,
                        );
                        self.fep.agent.perceive(&sem_obs);
                    }
                }

                // Broca quality → attention budget: successful articulation
                // reduces sensory search need (Levelt 1989 — monitoring loop).
                if self.stats.broca_quality_ema > 0.7 {
                    let contraction = 1.0 - (self.stats.broca_quality_ema - 0.7) * 0.15; // [0.955, 1.0]
                    self.scale_confidence("broca_attention_contract", contraction);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PARALLEL POST-PROCESSING
        // ═══════════════════════════════════════════════════════════════════════
        let _t_core = Instant::now();

        let pp_total_cycles = self.stats.total_cycles;
        let pp_in_flow = self.flow_state.in_flow;
        let pp_emotional_valence = self.emotion_contagion.prosody_valence();
        let pp_phi = self.unification_engine.psi as f32;
        let pp_smoothed_coh = coherence as f64;
        let pp_wm_importance_boost = self.fep.world_model.avg_error.clamp(0.0, 1.0) * 0.3;
        let pp_thalamic_salience = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => THALAMIC_DEEP_SALIENCE,
            super::CognitiveDepth::Cortical => 0.0,
            super::CognitiveDepth::Reflex => THALAMIC_REFLEX_SALIENCE,
        };
        let pp_learning_threshold = self.config.learning_threshold;

        let cycle_reward = self.compute_reward_signal(prediction_error, pp_learning_threshold);

        let cycle_learning_result = CycleLearningResult {
            reward: cycle_reward,
            strategy_used: selected_strategy,
            successful: prediction_error < pp_learning_threshold && pp_in_flow,
            prediction_error,
            coherence,
        };

        let (_, memory_confidence_boost) = {
            let stability_regime = &mut self.stability_regime;
            let discovery_service = &mut self.discovery_service;
            let semantic_memory = &mut self.semantic_memory;
            let causal_enhancer = &mut self.causal_enhancer;
            let episodic_memory = &mut self.fep.episodic_memory;
            let primitive_belief_bridge = &mut self.primitive_belief_bridge;
            let closed_learning_loop = &mut self.fep.closed_learning_loop;
            let fep_learning_signal = &mut self.fep.learning_signal;
            let prev_primitive_state = &mut self.prev_primitive_state;
            let resonator_memory = &mut self.resonator_memory;

            module_timings.stability_regime = helpers::run_stability_regime(
                stability_regime,
                discovery_service,
                &perception.encoding.hv16_cached,
                delta_t,
                pp_total_cycles,
                urgency,
            );

            let episodic_ctx = helpers::EpisodicLearningContext {
                prediction_error,
                in_flow: pp_in_flow,
                input,
                compressed_state: &perception.encoding.compressed_state,
                emotional_valence: pp_emotional_valence,
                phi: pp_phi,
                total_cycles: pp_total_cycles,
                smoothed_coh: pp_smoothed_coh,
                detected_primitives: &perception.encoding.encoding_result.detected_primitives,
                memory_context_boost,
                wm_importance_boost: pp_wm_importance_boost + pp_thalamic_salience,
            };

            {
                let semantic_fn = || {
                    helpers::parallel_semantic_causal(
                        semantic_memory,
                        causal_enhancer,
                        semantic_hdc.into_owned(),
                        &perception.encoding.compressed_state,
                        &output,
                        prediction_error,
                        pp_total_cycles,
                    )
                };
                let episodic_fn = || {
                    helpers::parallel_episodic_learning(
                        episodic_memory,
                        resonator_memory,
                        primitive_belief_bridge,
                        prev_primitive_state,
                        fep_learning_signal,
                        closed_learning_loop,
                        &episodic_ctx,
                        cycle_learning_result,
                    )
                };
                #[cfg(feature = "parallel")]
                {
                    rayon_join(semantic_fn, episodic_fn)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    (semantic_fn(), episodic_fn())
                }
            }
        };
        // Apply memory context boost to confidence after rayon::join (deferred from parallel branch)
        if memory_confidence_boost.abs() > f32::EPSILON {
            self.adjust_confidence("memory_context_boost", memory_confidence_boost);
        }

        module_timings.core_parallel_postprocess = _t_core.elapsed().as_micros() as u64;

        self.stats.semantic_hits = self.semantic_memory.stats().semantic_hits;
        self.stats.semantic_misses = self.semantic_memory.stats().semantic_misses;
        self.stats.semantic_lr_factor = semantic_lr_factor;
        self.stats.semantic_avg_retrieved_error = self.semantic_memory.stats().avg_retrieved_error;
        self.stats.semantic_entries_stored = self.semantic_memory.stats().total_stored;

        DynamicsPhaseResult {
            core: DynCore {
                output,
                prediction,
                prediction_error,
                coherence,
                unified_psi,
                learning_occurred,
                training_loss,
                effective_lr,
                cycle_reward,
                prediction_coherence,
                self_model_accuracy,
            },
            fep: DynFep {
                fep_action_idx,
                fep_pragmatic_value,
                fep_accuracy,
                fep_complexity,
                fep_surprise,
                fep_td_error,
            },
            reasoning: DynReasoning {
                reasoning_confidence,
                reasoning_gate_blocked,
                reasoning_fallback,
                reasoning_plan_action,
                reasoning_plan_confidence,
                reasoning_narrative,
                metacognitive_anomaly,
                mcts_plan_effectiveness,
                causal_attention_edges,
                school_predicted_phi_gain,
            },
            attention: DynAttention {
                attention_budget_exceeded,
                attention_budget_elapsed_us,
                predictive_budget_gated,
            },
            resonator: DynResonator {
                resonator_wm_primed,
                resonator_reconsolidated,
                resonator_best_sim,
                resonator_prediction_error,
                resonator_error_exploration_mod,
            },
            homeostasis: DynHomeostasis {
                anomaly_recovery_progress,
                anomaly_recovering,
                valence_homeostasis_pull,
                arousal_homeostasis_pull,
                homeostasis_pull_strength,
                arousal_recovery_active,
                arousal_recovery_tau_factor,
            },
            guidance: DynGuidance {
                moral_steering_category: moral_steering_category.into(),
                guiding_priority_category,
                guiding_question,
                dominant_harmonic,
            },
            neuromod: DynNeuromod {
                neuromod_attention_alloc,
                ne_reorienting_boost,
                ne_arousal_feedback,
                confidence_velocity,
                sht_crash_dip,
                exploration_sht_drain,
                phasic_da_replay_boost: 0, // set during feedback phase
            },
            binding_threshold_mod,
            binding_confidence_mod,
            epistemic_semantic_lr_mod,
            pfe_surprise_mod,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            fep_tau_factor,
            prediction_horizon_tau,
            causal_world_model_edges: if self
                .causal_enhancer
                .as_ref()
                .map_or(false, |e| e.has_causal_structure())
            {
                self.causal_enhancer
                    .as_ref()
                    .map_or(0, |e| e.current_graph().edges.len())
            } else {
                0
            },
            epistemic_budget_scale,
            confidence_crash_detected,
            lr_frozen,
        }
    }

    /// Apply emotional homeostasis: pull valence toward neutral, arousal toward target.
    /// Returns (valence_pull, arousal_pull, pull_strength).
    fn apply_emotional_homeostasis(&mut self) -> (f32, f32, f32) {
        let curr_v = self.emotion_contagion.valence;
        let curr_a = self.emotion_contagion.prosody_arousal();

        let pull_mult = match self.carryover.urgency.urgency {
            super::CycleUrgency::Cruise => HOMEOSTASIS_PULL_CRUISE,
            super::CycleUrgency::Normal => HOMEOSTASIS_PULL_NORMAL,
            super::CycleUrgency::Critical => HOMEOSTASIS_PULL_CRITICAL,
        };

        let v_pull = -curr_v * 0.05 * pull_mult;
        let a_pull = (HOMEOSTASIS_AROUSAL_TARGET - curr_a) * 0.05 * pull_mult;
        self.emotion_contagion.valence = (curr_v + v_pull).clamp(-1.0, 1.0);

        self.stats.avg_valence_homeostasis =
            self.stats.avg_valence_homeostasis * 0.95 + v_pull.abs() * 0.05;

        self.carryover.history.last_emotion_valence = self.emotion_contagion.valence;
        self.carryover.history.last_emotion_arousal = curr_a;

        (v_pull, a_pull, pull_mult)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn dynamics_produces_finite_cfc_output() {
        let mut svc = make_service();
        let result = svc.cycle("dynamics finite check");
        for (i, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "CfC output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn dynamics_prediction_error_finite_non_nan() {
        let mut svc = make_service();
        let result = svc.cycle("prediction error check");
        assert!(result.prediction_error.is_finite());
        assert!(!result.prediction_error.is_nan());
    }

    #[test]
    fn dynamics_reasoning_gate_not_blocked_when_engine_absent() {
        let mut svc = make_service();
        let result = svc.cycle("reasoning gate test");
        // reasoning_engine is disabled by default, so the gate never fires
        assert!(!result.metadata.reasoning_gate_blocked);
    }

    #[test]
    fn dynamics_fep_fields_populated() {
        let mut svc = make_service();
        let result = svc.cycle("fep check");
        assert!(result.metadata.fep.fep_accuracy.is_finite());
        assert!(result.metadata.fep.fep_complexity.is_finite());
        assert!(result.metadata.fep.fep_surprise.is_finite());
    }

    #[test]
    fn dynamics_learning_occurred_flag_consistent() {
        let mut svc = make_service();
        let result = svc.cycle("learning flag");
        if result.learning_occurred {
            // training_loss is None when async_training=true (default)
            if let Some(loss) = result.training_loss {
                assert!(loss.is_finite());
            }
        }
    }

    #[test]
    fn dynamics_actual_effective_lr_zero_when_no_learning() {
        let mut cfg = CognitiveLoopConfig::default();
        // Max valid threshold (1.0) — PE rarely exceeds this on first cycle
        cfg.learning_threshold = 1.0;
        let mut svc = CognitiveLoopService::new(cfg).unwrap();
        let result = svc.cycle("no learning");
        if !result.learning_occurred {
            assert_eq!(result.metadata.actual_effective_lr, 0.0);
        }
    }
}
