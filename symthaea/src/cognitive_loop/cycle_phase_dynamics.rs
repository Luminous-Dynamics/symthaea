//! Core dynamics phase of the cognitive cycle.
//!
//! Extracts Phases A–11 from the original `cycle()` method:
//! CycleSnapshot build, subsystem manager computation, self-model tracking,
//! resonator recall, semantic memory, CfC step, prediction, world model,
//! emotion, FEP active inference, moral modulation, training, parallel
//! post-processing.

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use ndarray::Array1;
use rayon::join as rayon_join;
use std::borrow::Cow;
use std::time::Instant;

use super::cycle::{DynamicsPhaseResult, PerceptionPhaseResult};
use super::helpers;
use super::training::TrainingSample;
use super::thresholds::{
    ATTENTION_BUDGET_US, BINDING_CONFIDENCE_THRESHOLD, BINDING_LOW_THRESHOLD,
    BINDING_STRONG_CONFIDENCE_SCALE, BINDING_STRONG_RELIEF_SCALE, BINDING_WEAK_CAUTION_SCALE,
    BINDING_WEAK_CONFIDENCE_SCALE, COHERENCE_CONFIDENCE_BOOST, COHERENCE_HIGH_THRESHOLD,
    COHERENCE_LOW_DAMPEN_SCALE, COHERENCE_LOW_THRESHOLD, COHERENCE_PREDICTION_EMA,
    DOMINANCE_CONFIDENT, DOMINANCE_DEFAULT, DOMINANCE_FLOW_BASE, DOMINANCE_FLOW_SCALE,
    MEMORY_RECALL_TOP_K, POLICY_FULL_AGREEMENT_BOOST, POLICY_MIN_WINDOW, POLICY_SOFT_THRESHOLD,
    POLICY_TEMP_BASE, POLICY_TEMP_RANGE, POLICY_WINDOW_SIZE, QUANTUM_COHERENCE_BOOST_SCALE,
    QUANTUM_COHERENCE_THRESHOLD, RESONANCE_TAU_CENTER, RESONANCE_TAU_SCALE,
    RESONATOR_ERROR_CONFIDENCE_DAMPEN, RESONATOR_ERROR_EXPLORATION_SCALE,
    RESONATOR_ERROR_EXPLORATION_THRESHOLD, RESONATOR_LOW_ERROR_CONFIDENCE_SCALE,
    RESONATOR_LOW_ERROR_THRESHOLD, SELF_MODEL_ACCURACY_EMA, SELF_MODEL_HIGH_THRESHOLD,
    SELF_MODEL_HIGH_TRUST_BOOST, SELF_MODEL_LOW_CONFIDENCE_SCALE, SELF_MODEL_LOW_THRESHOLD,
    THALAMIC_DEEP_SALIENCE, THALAMIC_REFLEX_SALIENCE, WORLD_MODEL_SPONGINESS_THRESHOLD,
    WORLD_MODEL_SPONGY_LR_SCALE, WORLD_MODEL_STIFFNESS_LR_SCALE, WORLD_MODEL_STIFFNESS_THRESHOLD,
};
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
        let prediction_error = perception.prediction_error;
        let urgency = perception.urgency;
        let phi_attention_weight = perception.phi_attention_weight;
        let surprise_triggered = perception.surprise_triggered;
        let moral_concern_detected = perception.moral_concern_detected;
        let _input_memoized = perception.input_memoized;
        let selected_strategy = perception.selected_strategy;

        // Cache moral_score for neuromod feedback (consumed in helpers/cycle_phases.rs)
        self.carryover.quality.last_moral_score = perception.moral_score;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE A: OBSERVE — Build immutable CycleSnapshot (Phase 2.3)
        // ═══════════════════════════════════════════════════════════════════════
        let cycle_snapshot = super::subsystem_trait::CycleSnapshot::build(
            self.stats.total_cycles as u64,
            self.prediction_confidence as f32,
            self.fep_lr_boost,
            prediction_error,
            self.coherence_bridge.smoothed_coherence(),
            self.stats.unified_psi as f64,
            phi_attention_weight,
            self.emotion_contagion.arousal,
            self.emotion_contagion.valence,
            self.thermodynamic_load,
            self.carryover.quality.last_dissipative_health,
            self.somatic_bridge.systemic_stress(),
            urgency,
            false, // attention_budget_exceeded not yet known at this point
            &perception.compressed_state,
            &perception.hv16_cached,
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
                let confidence_error = (self.prediction_confidence - pred_confidence).abs();
                let urgency_match = if urgency == pred_urgency { 1.0f32 } else { 0.0 };
                let accuracy = (1.0 - confidence_error) * 0.7 + urgency_match * 0.3;
                self.carryover.learning.self_model_accuracy = self
                    .carryover
                    .learning
                    .self_model_accuracy
                    * SELF_MODEL_ACCURACY_EMA
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

        // FEEDBACK: Quantum coherence boosts exploration (prev cycle)
        if self.carryover.consciousness.quantum_coherence > QUANTUM_COHERENCE_THRESHOLD {
            let coherence_boost = (self.carryover.consciousness.quantum_coherence
                - QUANTUM_COHERENCE_THRESHOLD) as f32
                * QUANTUM_COHERENCE_BOOST_SCALE;
            self.adjust_exploration("quantum_coherence", coherence_boost);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a. Memory System Integration: Recall relevant episodic memories
        // ═══════════════════════════════════════════════════════════════════════
        let memory_context_boost = self.recall_episodic_context(&perception.compressed_state);

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.1 Resonator-enhanced recall: factorize bundled memories
        // ═══════════════════════════════════════════════════════════════════════
        let mut resonator_wm_primed = false;
        let mut resonator_reconsolidated: usize = 0;
        let mut resonator_best_sim: f32 = 0.0;

        let resonator_prediction_error: f32 =
            if let Some(ref prev_pred) = self.stats.last_resonator_prediction {
                let sim = helpers::cosine_f32(prev_pred, &perception.compressed_state);
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
        let pre_update_coherence = self.coherence_bridge.smoothed_coherence();

        // ── Phase 20: Phenomenal binding → threshold gating ──────────────────
        let cached_binding = self.carryover.quality.last_phenomenal_binding as f32;
        let binding_threshold_mod = if cached_binding > BINDING_CONFIDENCE_THRESHOLD {
            let relief =
                (cached_binding - BINDING_CONFIDENCE_THRESHOLD) * BINDING_STRONG_RELIEF_SCALE;
            self.scale_threshold("binding_strong_relief", 1.0 - relief);
            self.stats.binding_threshold_mod_count += 1;
            -relief
        } else if cached_binding < BINDING_LOW_THRESHOLD && cached_binding > 0.0 {
            let caution =
                (BINDING_LOW_THRESHOLD - cached_binding) * BINDING_WEAK_CAUTION_SCALE;
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
                    perception.compressed_state.len() == res_mem.resonator.config.dim;
                if res_dim_ok && !res_mem.is_empty() {
                    if let Ok(matches) =
                        res_mem.retrieve(&[("content", &perception.compressed_state)])
                    {
                        let top_matches: Vec<_> =
                            matches.into_iter().take(MEMORY_RECALL_TOP_K).collect();

                        let best_match_sim = top_matches
                            .iter()
                            .map(|m| {
                                helpers::cosine_f32(&perception.compressed_state, &m.hv)
                            })
                            .fold(0.0f32, f32::max);
                        let match_timestamps: Vec<u64> =
                            top_matches.iter().map(|m| m.timestamp).collect();
                        resonator_best_sim = best_match_sim;

                        if best_match_sim > 0.3 {
                            let best_ep = top_matches.iter().max_by(|a, b| {
                                let sa: f32 = perception
                                    .compressed_state
                                    .iter()
                                    .zip(a.hv.iter())
                                    .map(|(x, y)| x * y)
                                    .sum();
                                let sb: f32 = perception
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
                            let dim = perception.compressed_state.len();
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
                                &[("content", &perception.compressed_state)],
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
                                            self.adjust_confidence(
                                                "resonator_factor_high",
                                                0.03,
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        if best_match_sim > 0.3 {
                            self.adjust_confidence(
                                "resonator_recall_prime",
                                best_match_sim * 0.02,
                            );
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

        if resonator_best_sim > 0.5 {
            self.fep_agent.precision.prior_precision = (self.fep_agent.precision.prior_precision
                + (resonator_best_sim - 0.5) as f64 * 0.1)
                .min(2.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.2. Goal System: Apply attention bias from active goals
        // ═══════════════════════════════════════════════════════════════════════
        let goal_attention_bias = self.goal_system.attention_bias();

        if let Some(top) = self.goal_system.top_goal() {
            let goal_priority = top.priority;
            if goal_priority > 0.5 {
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
        let valence_homeostasis_pull;
        let arousal_homeostasis_pull;
        let homeostasis_pull_strength;
        {
            let _prev_v = self.carryover.history.last_emotion_valence;
            let _prev_a = self.carryover.history.last_emotion_arousal;
            let curr_v = self.emotion_contagion.valence;
            let curr_a = self.emotion_contagion.prosody_arousal();

            let pull_mult = match self.carryover.urgency.urgency {
                super::CycleUrgency::Cruise => 1.5f32,
                super::CycleUrgency::Normal => 1.0,
                super::CycleUrgency::Critical => 0.6,
            };
            homeostasis_pull_strength = pull_mult;

            let v_pull = -curr_v * 0.05 * pull_mult;
            let a_pull = (0.3 - curr_a) * 0.05 * pull_mult;
            self.emotion_contagion.valence = (curr_v + v_pull).clamp(-1.0, 1.0);

            valence_homeostasis_pull = v_pull;
            arousal_homeostasis_pull = a_pull;

            self.stats.avg_valence_homeostasis =
                self.stats.avg_valence_homeostasis * 0.95 + v_pull.abs() * 0.05;

            self.carryover.history.last_emotion_valence = self.emotion_contagion.valence;
            self.carryover.history.last_emotion_arousal = curr_a;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1c. Update Unified Emotional Bridge (VAD-based)
        // ═══════════════════════════════════════════════════════════════════════
        let simple_valence = self.emotion_contagion.prosody_valence() as f64;
        let simple_arousal = self.emotion_contagion.prosody_arousal() as f64;
        let dominance = if self.flow_state.in_flow {
            DOMINANCE_FLOW_BASE + DOMINANCE_FLOW_SCALE * self.flow_state.intensity as f64
        } else if self.prediction_confidence > 0.6 {
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
            .project_to_hdc_vec(&perception.compressed_state)
            .map(Cow::Owned)
            .unwrap_or(Cow::Borrowed(&perception.compressed_state));
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
        let epistemic_semantic_lr_mod: f32 = if prev_epistemic < 0.4 && prev_epistemic > 0.0 {
            let caution = 0.8_f32 + prev_epistemic * 0.5;
            semantic_lr_factor *= caution;
            self.stats.epistemic_semantic_mod_count += 1;
            caution - 1.0
        } else if prev_epistemic > 0.8 {
            let boost = 1.0_f32 + (prev_epistemic - 0.8) * 1.0;
            semantic_lr_factor *= boost;
            self.stats.epistemic_semantic_mod_count += 1;
            boost - 1.0
        } else {
            0.0
        };

        // 3. Convert to ndarray for CfC
        let input_array: Array1<f32> = perception.compressed_state.iter().copied().collect();

        // 4. Step CfC forward with current input
        let resonance_tau_factor = if self.carryover.history.resonance_frequency > 0.0 {
            let deviation = (self.carryover.history.resonance_frequency as f32
                - RESONANCE_TAU_CENTER as f32)
                .clamp(-0.5, 0.5);
            1.0 - (deviation * RESONANCE_TAU_SCALE)
        } else {
            1.0
        };
        let arousal_tau_factor = if (self.carryover.history.body_arousal - 0.5).abs() > 0.1 {
            1.0 + (self.carryover.history.body_arousal - 0.5) * 0.1
        } else {
            1.0
        };
        let codebook_tau_factor = if resonator_best_sim > 0.5 {
            1.0 - (resonator_best_sim - 0.5) * 0.1
        } else if resonator_best_sim > 0.0 && resonator_best_sim < 0.2 {
            1.0 + (0.2 - resonator_best_sim) * 0.15
        } else {
            1.0
        };
        let arousal_recovery_tau_factor;
        let arousal_recovery_active;
        if self.carryover.urgency.arousal_trap_counter > 5
            && self.carryover.urgency.arousal_trap_counter <= 10
        {
            let recovery_intensity = (self.carryover.urgency.arousal_trap_counter - 5) as f32 / 5.0;
            arousal_recovery_tau_factor = 1.0 + recovery_intensity * 0.2;
            arousal_recovery_active = true;
        } else {
            arousal_recovery_tau_factor = 1.0;
            arousal_recovery_active = false;
        }

        let delta_t = self.config.cfc_config.delta_t
            * resonance_tau_factor
            * arousal_tau_factor
            * codebook_tau_factor
            * arousal_recovery_tau_factor
            * self.somatic_bridge.to_interoceptive_signals().tau_slowdown_factor as f32;
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
        self.world_model.update_sensory(&perception.compressed_state);

        let wm_stiffness = self.world_model.avg_error.clamp(0.0, 1.0);
        if self.stats.total_cycles > 20 {
            if wm_stiffness > WORLD_MODEL_STIFFNESS_THRESHOLD {
                let stiffness_nudge =
                    (wm_stiffness - WORLD_MODEL_STIFFNESS_THRESHOLD) * WORLD_MODEL_STIFFNESS_LR_SCALE;
                self.adjust_lr("wm_stiff", stiffness_nudge);
            } else if wm_stiffness < WORLD_MODEL_SPONGINESS_THRESHOLD {
                let spongy_dampen = (WORLD_MODEL_SPONGINESS_THRESHOLD - wm_stiffness)
                    * WORLD_MODEL_SPONGY_LR_SCALE;
                self.scale_lr("wm_spongy", 1.0 - spongy_dampen);
            }
        }

        let level_errors = self.world_model.level_errors();
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
        self.create_experience(&perception.compressed_state, &prediction, prediction_error);

        // 10. Update coherence bridge
        let tau_owned: Vec<ndarray::Array1<f32>> = self.temporal_network.all_tau_owned();
        let tau_refs: Vec<&ndarray::Array1<f32>> = tau_owned.iter().collect();
        self.coherence_bridge.update(&tau_refs);

        // 10b. Update temporal signature encoder
        let flattened_tau: Vec<f32> = tau_owned.iter().flat_map(|a| a.iter().copied()).collect();
        self.temporal_signature_encoder.record_batch(&flattened_tau);

        // 10c. Update adaptive behavior
        let (pattern, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let coherence = self.coherence_bridge.smoothed_coherence();
        self.carryover.history.cached_coherence = Some(coherence);
        let voice_confidence = self.voice_feedback_bridge.summary().voice_confidence;
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
        }

        // 10d. Update prediction confidence
        self.update_prediction_confidence(pattern, prediction_error, pattern_confidence);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.5 Active Inference Bridge
        // ═══════════════════════════════════════════════════════════════════════
        let prediction_success = prediction_error < self.config.learning_threshold;
        self.active_inference_bridge
            .observe_resolution(self.prediction_confidence as f64, prediction_success);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6 FEP Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        let effective_lr = self.stats.adaptive_learning_rate;
        let (fep_action_idx, fep_action_probs, is_surprised, fep_pragmatic_value_raw) =
            self.step_fep_active_inference(prediction_error, coherence);

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
                if effectiveness > 0.6 {
                    self.adjust_confidence("mcts_effective", (effectiveness - 0.6) * 0.03);
                } else if effectiveness < 0.3 {
                    self.adjust_exploration("mcts_poor_plan", (0.3 - effectiveness) * 0.02);
                }
                self.stats.avg_mcts_plan_effectiveness =
                    self.stats.avg_mcts_plan_effectiveness * 0.9 + effectiveness * 0.1;
                effectiveness
            } else {
                0.0
            };

        // ── Apply previous cycle's MCTS plan ──────────────
        if let Some((plan_action, plan_confidence)) = self.carryover.history.mcts_plan.take() {
            if plan_confidence > 0.7 && plan_action != fep_action_idx {
                self.carryover.history.mcts_plan_applied =
                    Some((plan_action, plan_confidence, prediction_error));
                let plan_weight = plan_confidence * 0.4;
                match plan_action {
                    0 => {
                        self.scale_lr("mcts_exploit", 1.0 - plan_weight * 0.1);
                    }
                    1 => {
                        self.adjust_confidence("mcts_consolidate", plan_weight * 0.05);
                    }
                    2 => {
                        self.adjust_exploration("plan_explore_directive", plan_weight * 0.08);
                    }
                    _ => {}
                }
            }
        }

        // ── FEP Free Energy Decomposition ──────────────
        let fep_vals = self
            .fep_agent
            .last_fe_components
            .as_ref()
            .map(|fe| (fe.accuracy, fe.complexity, fe.surprise, fe.prediction_error));
        let (fep_accuracy, fep_complexity, fep_surprise, fep_td_error) =
            if let Some((acc, comp, surp, pe)) = fep_vals {
                if acc > 0.5 {
                    self.adjust_confidence("fep_accuracy_high", 0.01);
                }
                if comp > 1.0 {
                    self.scale_lr("fep_complexity", 1.0 - ((comp - 1.0).min(0.5) * 0.1) as f32);
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
        if fep_pragmatic_value > 0.7 {
            self.scale_exploration(
                "fep_pragmatic_exploit",
                (1.0 - (fep_pragmatic_value - 0.7) * 0.3) as f32,
            );
        } else if fep_pragmatic_value < 0.3 && fep_pragmatic_value > 0.0 {
            let p_explore = ((0.3 - fep_pragmatic_value) * 0.15).min(0.05) as f32;
            self.adjust_exploration("fep_pragmatic_low", p_explore);
        }

        if fep_td_error.abs() > 0.5 {
            if let Some(ref mut enhancer) = self.causal_enhancer {
                if enhancer.should_discover() {
                    let _ = enhancer.run_discovery();
                }
            }
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

        if self.social.external_reward.abs() > f32::EPSILON {
            let outcome_obs = Observation::from_consciousness_state(
                self.social.external_reward as f64,
                coherence as f64,
                self.prediction_confidence as f64,
                effective_lr as f64,
            );
            self.fep_agent
                .learn_from_outcome(fep_action_idx, &outcome_obs);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Moral Modulation of Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        let (moral_steering_category, pfe_surprise_mod) = self.apply_moral_modulation(
            moral_concern_detected,
            &perception.moral_judgment,
            perception.moral_score,
            is_surprised,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH + PSI SYNTHESIS (extracted to cycle_neuromod_phase.rs)
        // ═══════════════════════════════════════════════════════════════════════
        let neuromod_result =
            self.run_neuromodulator_and_psi_phase(prediction_error, coherence);
        let ne_reorienting_boost = neuromod_result.ne_reorienting_boost;
        let ne_arousal_feedback = neuromod_result.ne_arousal_feedback;
        let sht_crash_dip = neuromod_result.sht_crash_dip;
        let exploration_sht_drain = neuromod_result.exploration_sht_drain;
        let confidence_velocity = neuromod_result.confidence_velocity;
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
            let r = self.enhanced_fep_bridge.cycle(
                prediction_error as f64,
                coherence as f64,
                self.prediction_confidence as f64,
                effective_lr as f64,
            );
            self.fep_learning_signal = r.learning_signal as f32;
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
                    if enhanced_result.fep_result.epistemic_value > 0.5 {
                        self.adjust_exploration("motor_exploration_trigger", 0.1);
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    if enhanced_result.motor_command.intensity > 0.7 {
                        self.self_model_tier.self_reflection.force_reflection();
                    }
                }
                MotorCommandType::MemoryConsolidate => {
                    if enhanced_result.motor_command.intensity > 0.5 {
                        self.episodic_memory.consolidate_recent();
                    }
                }
                MotorCommandType::ExpectationReset => {
                    if enhanced_result.action_outcome_coupling < 0.3 {
                        self.last_prediction = None;
                        self.set_confidence("inference_mode_init", 0.5);
                    }
                }
                MotorCommandType::MotorOutput | MotorCommandType::NoOp => {}
            }

            if self.fep_learning_signal > 0.5 && enhanced_result.should_learn {
                self.world_model
                    .increase_plasticity(self.fep_learning_signal);
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
            self.fep_agent.perceive(&urgent_obs);
            self.enhanced_fep_bridge
                .cycle(coh_urgency as f64, 0.1, 0.1, effective_lr as f64);
        }

        // 10e–g. Update flow state, curiosity drive, self-reflection
        self.update_drives_and_reflection(pattern, prediction_error, coherence);

        // Unified Psi + Experience Bus + Guiding Question computed by
        // run_neuromodulator_and_psi_phase() above — values already in neuromod_result.

        // ── Phase 15: Attention budget check ─────────────────────────────────
        let neuromod_attention_alloc = self.neuromod.bath.attention_budget_allocation();
        let attention_budget_us = (ATTENTION_BUDGET_US as f32 * neuromod_attention_alloc) as u64;
        let attention_budget_elapsed_us = cycle_start.elapsed().as_micros() as u64;
        let attention_budget_exceeded = attention_budget_elapsed_us > attention_budget_us;
        if attention_budget_exceeded {
            self.stats.attention_budget_exceeded_count += 1;
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

        let predictive_budget_gated = attention_budget_elapsed_us > (ATTENTION_BUDGET_US * 4 / 5)
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
        #[cfg(feature = "reasoning_engine")]
        if let Some(ref mut reasoning_engine) = self.reasoning_engine {
            use crate::consciousness::epistemic_conflict::MultiTheoryMetrics as ECMetrics;
            use crate::consciousness::reasoning_engine::ReasoningContext;

            let ec_metrics = ECMetrics {
                phi: unified_psi,
                gwt: coherence as f64,
                ast: self.prediction_confidence as f64,
                pp: (1.0 - prediction_error as f64).clamp(0.0, 1.0),
                rpt: pattern_confidence as f64,
                embodiment: self.fep_learning_signal as f64,
                unified: unified_psi,
            };

            let elapsed_us = cycle_start.elapsed().as_micros() as u64;
            let available_us = 20_000u64.saturating_sub(elapsed_us);

            let reasoning_ctx = ReasoningContext {
                theory_metrics: ec_metrics,
                phi: unified_psi,
                available_budget_us: available_us,
                available_actions: Vec::new(),
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
                self.fep_learning_signal *= dampen;
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
                self.fep_agent.config.action_temperature = adaptive_temp;
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

        let neuromod_threshold =
            perception.effective_threshold * self.neuromod.bath.threshold_gate();

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
                    perception.compressed_state.iter().copied().collect(),
                    effective_lr,
                )
            } else {
                let train_input: Array1<f32> =
                    perception.compressed_state.iter().copied().collect();
                let train_target: Array1<f32> =
                    perception.compressed_state.iter().copied().collect();
                (train_input, train_target, effective_lr * 0.1)
            };

            if let Some(ref trainer) = self.async_trainer {
                trainer.send(TrainingSample {
                    input: train_input,
                    target: train_target,
                    dt: delta_t,
                    learning_rate: lr,
                    method: self.config.training_method,
                    avg_loss: self.stats.avg_training_loss,
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
            let is_night = self.biorhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.neuromod.bath
                .report_learning(effective_lr, prediction_error, is_night);
            let fatigue = self.neuromod.bath.learning_fatigue_factor();
            if fatigue < 1.0 {
                self.scale_lr("glutamate_fatigue", fatigue);
            }
        }

        // Goal←Cognition feedback
        if !learning_occurred && self.carryover.urgency.consecutive_low_error > 5 {
            if let Some(top) = self.goal_system.top_goal() {
                let top_id = top.id.clone();
                let delta = 0.01 * (1.0 + self.prediction_confidence * 0.5);
                self.goal_system.update_progress(&top_id, delta);
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
        self.stats.coherence_phi_contribution = self.coherence_bridge.phi_contribution();

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
                    .compressed_state
                    .chunks(8)
                    .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
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
                    if top_strength > 0.3 {
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
            self.adjust_confidence("causal_attention", causal_attention_boost * 0.05);
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
        let pp_wm_importance_boost = self.world_model.avg_error.clamp(0.0, 1.0) * 0.3;
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
            let episodic_memory = &mut self.episodic_memory;
            let primitive_belief_bridge = &mut self.primitive_belief_bridge;
            let closed_learning_loop = &mut self.closed_learning_loop;
            let fep_learning_signal = &mut self.fep_learning_signal;
            let prev_primitive_state = &mut self.prev_primitive_state;
            let resonator_memory = &mut self.resonator_memory;

            module_timings.stability_regime = helpers::run_stability_regime(
                stability_regime,
                discovery_service,
                &perception.hv16_cached,
                delta_t,
                pp_total_cycles,
                urgency,
            );

            let episodic_ctx = helpers::EpisodicLearningContext {
                prediction_error,
                in_flow: pp_in_flow,
                input,
                compressed_state: &perception.compressed_state,
                emotional_valence: pp_emotional_valence,
                phi: pp_phi,
                total_cycles: pp_total_cycles,
                smoothed_coh: pp_smoothed_coh,
                detected_primitives: &perception.encoding_result.detected_primitives,
                memory_context_boost,
                wm_importance_boost: pp_wm_importance_boost + pp_thalamic_salience,
            };

            rayon_join(
                || {
                    helpers::parallel_semantic_causal(
                        semantic_memory,
                        causal_enhancer,
                        semantic_hdc.into_owned(),
                        &perception.compressed_state,
                        &output,
                        prediction_error,
                        pp_total_cycles,
                    )
                },
                || {
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
                },
            )
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
            output,
            prediction,
            prediction_error,
            coherence,
            unified_psi,
            learning_occurred,
            training_loss,
            effective_lr,
            attention_budget_exceeded,
            attention_budget_elapsed_us,
            predictive_budget_gated,
            fep_action_idx,
            fep_pragmatic_value,
            fep_accuracy,
            fep_complexity,
            fep_surprise,
            fep_td_error,
            cycle_reward,
            reasoning_confidence,
            reasoning_gate_blocked,
            reasoning_fallback,
            reasoning_plan_action,
            reasoning_plan_confidence,
            reasoning_narrative,
            metacognitive_anomaly,
            anomaly_recovery_progress,
            anomaly_recovering,
            prediction_coherence,
            self_model_accuracy,
            resonator_wm_primed,
            resonator_reconsolidated,
            resonator_best_sim,
            resonator_prediction_error,
            resonator_error_exploration_mod,
            binding_threshold_mod,
            binding_confidence_mod,
            epistemic_semantic_lr_mod,
            pfe_surprise_mod,
            mcts_plan_effectiveness,
            causal_attention_edges,
            moral_steering_category: moral_steering_category.into(),
            valence_homeostasis_pull,
            arousal_homeostasis_pull,
            homeostasis_pull_strength,
            arousal_recovery_active,
            arousal_recovery_tau_factor,
            guiding_priority_category,
            guiding_question,
            dominant_harmonic,
            school_predicted_phi_gain,
            neuromod_attention_alloc,
            ne_reorienting_boost,
            ne_arousal_feedback,
            confidence_velocity,
            sht_crash_dip,
            exploration_sht_drain,
            phasic_da_replay_boost: 0, // set during feedback phase
        }
    }
}
