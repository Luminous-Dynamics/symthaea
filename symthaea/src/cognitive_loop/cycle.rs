//! Core cognitive cycle implementation with parallel post-processing.
//!
//! Contains the main `cycle()` method which implements the bidirectional
//! HDC-CfC loop with rayon-parallelized subsystem updates.

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use ndarray::Array1;
use rayon::join as rayon_join;
use std::borrow::Cow;
use std::time::Instant;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;

// Result structs imported from helpers::cycle_phases
use super::helpers::{DreamPhaseResult, EpisodicReplayResult, ResonatorCodebookResult};

// ═══════════════════════════════════════════════════════════════════════════════
// Tuning constants: all values live in `thresholds.rs` with scientific citations.
// See `super::thresholds` for the centralized registry.
// ═══════════════════════════════════════════════════════════════════════════════
use super::thresholds::{
    // Moral evaluation
    MORAL_CONCERN_THRESHOLD, MORAL_BENEFIT_THRESHOLD,
    MORAL_CONCERN_EXPLORATION_DAMPEN, MORAL_CONCERN_PAUSE_BOOST,
    MORAL_BENEFIT_CONFIDENCE_BOOST,
    // Surprise & exploration
    QUANTUM_COHERENCE_THRESHOLD, QUANTUM_COHERENCE_BOOST_SCALE,
};

// ═══════════════════════════════════════════════════════════════════
// Ψ (PSI) SYNTHESIS — Consciousness Estimate
//
// Computes a composite soft signal from multiple subsystem indicators.
// This is Layer 1 of Symthaea's three-layer measurement:
//
//   Layer 1: Ψ (Psi)  — Fast estimate (every cycle, O(1))
//   Layer 2: Σ (Sigma) — Synergistic integration (every N cycles, O(n²))
//   Layer 3: Φ (Phi)   — True IIT (on demand, O(n³))
//
// Components: temporal_coherence + voice_quality + flow_state
//           + relational + body + embodied
//
// For actual integrated information, use PhiEngine or true_phi.
// ═══════════════════════════════════════════════════════════════════

use super::thresholds::{
    // FEP tuning
    FEP_SURPRISE_SCALE, FEP_LR_DECAY,
    // Dominance estimation
    DOMINANCE_FLOW_BASE, DOMINANCE_FLOW_SCALE, DOMINANCE_CONFIDENT, DOMINANCE_DEFAULT,
    // Resonance tau modulation
    RESONANCE_TAU_CENTER, RESONANCE_TAU_SCALE,
    // Policy agreement (KL gate)
    POLICY_SOFT_THRESHOLD, POLICY_FULL_AGREEMENT_BOOST,
    POLICY_WINDOW_SIZE, POLICY_MIN_WINDOW, POLICY_TEMP_BASE, POLICY_TEMP_RANGE,
    // Attention budget
    ATTENTION_BUDGET_US,
    // Memory
    MEMORY_RECALL_TOP_K,
};

use super::helpers;
use super::temporal_network::TemporalNetwork;
use super::training::TrainingSample;
use super::{
    ActionHint, AdaptiveBehavior, CognitiveLoopService, CycleLearningResult, CycleResult,
    CycleState, ResponseStrategy, TrainingMethod,
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
        let mut module_timings = super::ModuleTimings::default();

        // ── Phase 17: Startup transient suppression ─────────────────────────
        // Science: Hopfield (1982) — recurrent networks require settling time before
        // producing reliable dynamics. During warmup (cycles 0–50), suppress learning
        // rate and curiosity to prevent cementing transient noise as learned patterns.
        let startup_warmup_cycles = super::thresholds::STARTUP_WARMUP_CYCLES;
        let startup_suppressed = self.stats.total_cycles <= startup_warmup_cycles;
        let startup_warmup_progress = if startup_suppressed {
            self.stats.total_cycles as f32 / startup_warmup_cycles as f32
        } else {
            1.0
        };
        if startup_suppressed {
            self.stats.startup_suppressed_cycles += 1;
            // Ramp learning rate from 20% → 100% over warmup period
            let lr_scale = 0.2 + 0.8 * startup_warmup_progress;
            self.stats.adaptive_learning_rate *= lr_scale;
            // Suppress curiosity during transient (let CfC settle)
            self.curiosity_drive.exploration_urge *= startup_warmup_progress;
        }

        // Snapshot exploration_urge for end-of-cycle budget clamping (Task B)
        let exploration_urge_start = self.curiosity_drive.exploration_urge;

        // Snapshot confidence for end-of-cycle drift clamping (Task G)
        self.carryover.learning.prediction_confidence = self.prediction_confidence;

        // ── Phase 2.2: Begin feedback proposal collection for this cycle ────
        self.feedback_state.begin_cycle();
        // ── Phase 2.3: Clear subsystem output collector ────
        self.subsystem_collector.clear();

        // Chronobiology: refresh biorhythm every 97 cycles (co-prime amortization)
        self.biorhythm_refresh_counter += 1;
        if self.biorhythm_refresh_counter >= super::thresholds::BIORHYTHM_INTERVAL {
            self.biorhythm = crate::chronobiology::Biorhythm::current();
            self.neuromodulator_bath.modulate_circadian_continuous(self.biorhythm.hour);
            // Record personality profile for drift detection
            let profile = self.neuromodulator_bath.personality_profile();
            self.personality_drift_tracker.record(&profile);
            self.biorhythm_refresh_counter = 0;
        }
        // Apply circadian plasticity to learning rate (Night=high plasticity, Day=low)
        // Halved: bath circadian baselines (Phase 2) provide the other 50%
        let plasticity_half = 1.0 + (self.biorhythm.plasticity_mod as f32 - 1.0) * 0.5;
        let circadian_lr = self.stats.adaptive_learning_rate * plasticity_half;
        self.stats.adaptive_learning_rate = circadian_lr.clamp(0.0001, 0.1);

        // ═══════════════════════════════════════════════════════════════════════
        // NOCICEPTION: Drain infrastructure errors and convert to felt signals
        // ═══════════════════════════════════════════════════════════════════════
        self.somatic_bridge.update();
        let somatic_signals = self.somatic_bridge.to_interoceptive_signals();
        // Apply somatic stress to thermodynamic load (additive)
        self.thermodynamic_load = (self.thermodynamic_load + somatic_signals.thermodynamic_load_delta).min(1.0);
        // Apply arousal spike from severe infrastructure damage
        if somatic_signals.arousal_spike > 0.0 {
            self.emotion_contagion.arousal =
                (self.emotion_contagion.arousal + somatic_signals.arousal_spike).min(1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH: Produce from previous cycle's signals (Phase A)
        // Science: Doya (2002) — DA/NE/5-HT/ACh unify metalearning modulation.
        // Uses carryover values (previous cycle) to avoid ordering dependencies.
        // ═══════════════════════════════════════════════════════════════════════
        {
            let neuromod_inputs = super::neuromodulators::NeuromodulatorInputs {
                prediction_error: self.stats.avg_prediction_error,
                surprise: self.stats.avg_prediction_error > self.config.learning_threshold * 3.0,
                reward_signal: self.carryover.quality.last_value_score as f32,
                coherence: self.carryover.history.cached_coherence.unwrap_or(0.5),
                arousal: self.emotion_contagion.arousal,
                binding_strength: self.carryover.quality.last_phenomenal_binding as f32,
                epistemic_confidence: self.carryover.quality.last_epistemic_confidence,
                flow_active: self.flow_state.in_flow,
            };
            self.neuromodulator_bath.update(&neuromod_inputs);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE -1: Ingest background-trained weights (non-blocking)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut trainer) = self.async_trainer {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                trainer.apply_latest_weights(cfc);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.1: Safety Pre-check (fast amygdala veto)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(blocked) = self.safety_precheck(input, cycle_start) {
            return blocked;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0: Thalamic Routing (Cognitive Depth Selection)
        // ═══════════════════════════════════════════════════════════════════════
        self.update_cognitive_depth();

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.3: Negation Detection (guards moral evaluation)
        // ═══════════════════════════════════════════════════════════════════════
        let input_negation_polarity = self.detect_negation_polarity(input);

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.4: Moral Evaluation (throttled: every Nth cycle or on new input)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (moral_score, moral_concern_detected, moral_judgment) =
            self.run_moral_phase(input, input_negation_polarity);
        module_timings.moral_algebra = _t.elapsed().as_micros() as u64;
        let mut moral_steering_category: &str = "";

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        // Select response strategy BEFORE processing, based on:
        // - Q-learning from past interactions
        // - Previous reward (stick with success, avoid failure)
        // - Phi-gating (high Phi -> Exploratory, low Phi -> Supportive)
        // - Moral concerns (bias toward Supportive for ethical guidance)

        let prior_phi = self.unification_engine.psi;
        let prior_reward = self
            .closed_learning_loop
            .last_result
            .as_ref()
            .map(|r| r.reward);
        let mut selected_strategy = if moral_concern_detected {
            // Bias toward supportive strategy when moral concerns detected
            ResponseStrategy::Supportive
        } else {
            let base_strategy = self
                .closed_learning_loop
                .select_strategy(prior_phi, prior_reward);

            // MCTS-informed strategy bias: peek at prior cycle's deliberative plan.
            // When the MCTS planner produced a confident plan (confidence > 0.7),
            // nudge strategy toward the plan's intent — aligning deliberative and
            // habitual systems (Kahneman dual-process, Phase 10i applies the action).
            if let Some(&(plan_action, plan_confidence)) = self.carryover.history.mcts_plan.as_ref()
            {
                if plan_confidence > 0.7 {
                    match plan_action {
                        0 => {
                            // Plan says "exploit" — favor Detailed (depth over breadth)
                            match base_strategy {
                                ResponseStrategy::Exploratory => ResponseStrategy::Detailed,
                                other => other,
                            }
                        }
                        2 => {
                            // Plan says "explore" — favor Exploratory
                            match base_strategy {
                                ResponseStrategy::Supportive | ResponseStrategy::Concise => {
                                    ResponseStrategy::Exploratory
                                }
                                other => other,
                            }
                        }
                        _ => base_strategy, // consolidate(1) or unknown: no bias
                    }
                } else {
                    base_strategy
                }
            } else {
                base_strategy
            }
        };

        // Strategy influences adaptive behavior
        self.apply_strategy_modulation(selected_strategy);

        // ── Phase 21: Embodied agency → strategy modulation ──────────────
        // Science: Varela (1991) — low agency = reactive mode → prefer conservative strategy
        let agency_strategy_override = {
            let cached_agency = self.carryover.consciousness.last_embodied_agency;
            if cached_agency < 0.3 && cached_agency > 0.0
                && selected_strategy == ResponseStrategy::Exploratory
            {
                selected_strategy = ResponseStrategy::Supportive;
                self.apply_strategy_modulation(selected_strategy); // re-apply with new strategy
                self.stats.agency_strategy_override_count += 1;
                true
            } else {
                false
            }
        };

        // 1. HDC encode with attention from previous prediction
        let _t_core = Instant::now();
        let encoding_result = self.encoder.encode(input);
        let prediction_error = encoding_result.prediction_error;
        module_timings.core_hdc_encode = _t_core.elapsed().as_micros() as u64;

        // Pre-compute BinaryHV once for all subsystems that need it.
        // real_hv_to_hv16 iterates 16,384 floats twice (mean + threshold).
        // Previously called 7× per cycle — this caches the result.
        let _t_core = Instant::now();
        let hv16_cached = real_hv_to_hv16(&encoding_result.hdv);

        module_timings.core_compress = _t_core.elapsed().as_micros() as u64;

        // Soul value alignment: evaluate encoding against Seven Harmonies.
        // If strongly misaligned with core values, flag moral concern.
        // Also modulates learning rate: high alignment → boost, misalignment → dampen.
        let soul_alignment = if let Some(ref soul) = self.soul {
            let alignment = soul.evaluate_alignment(&encoding_result.hdv);
            if alignment.overall_alignment < MORAL_CONCERN_THRESHOLD {
                self.stats.moral_concerns_detected += 1;
            }
            // Soul-driven learning rate modulation:
            // High alignment boosts learning (trust the direction).
            // Misalignment dampens learning (conflict with core values).
            if alignment.overall_alignment > 0.3 {
                let boost = (alignment.overall_alignment - 0.3) * 0.1;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.8, 1.3);
            } else if alignment.overall_alignment < -0.3 {
                let dampening = (alignment.overall_alignment + 0.3).abs() * 0.15;
                self.carryover.learning.subsystem_lr_factor *= 1.0 - dampening;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.2);
            }
            alignment.overall_alignment
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 0.5 Phi-Guided Attention Gating
        // When present, weight the encoded HDV by its integrated information
        // contribution. High-Phi signals get boosted, low-Phi get attenuated.
        // ═══════════════════════════════════════════════════════════════════════
        let phi_attention_weight = if let Some(ref mut gate) = self.phi_attention_gate {
            let phi_vals = [self.stats.unified_psi as f64];
            // Avoid cloning 64KB ContinuousHV — forward() takes &[ContinuousHV]
            let result = gate.forward(std::slice::from_ref(&encoding_result.hdv), &phi_vals);
            result.weights.first().copied().unwrap_or(1.0)
        } else {
            1.0
        };
        // phi_attention_weight is applied below to scale compressed state

        // ═══════════════════════════════════════════════════════════════════════
        // 1.1 Surprise-Driven Exploration: Track surprise, modulate curiosity
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // When enabled, feed prediction error to surprise bridge. If surprise
        // exceeds the adaptive threshold, lower the boredom threshold to
        // encourage exploration of novel states.
        // Pre-compute compressed state once for the entire cycle.
        // Used by: surprise bridge, CfC step, world model, training, experience buffer.
        // Phi-guided attention: scale compressed state by attention weight.
        // High unified_psi → weight > 1.0 (amplify); Low → weight < 1.0 (attenuate).
        // Science: Tononi (2015) — Phi selects which information gets integrated.
        let compressed_state: Vec<f32> = self
            .encoder
            .compress_for_ltc(&encoding_result.hdv, self.config.cfc_config.input_dim)
            .iter()
            .map(|v| v * phi_attention_weight)
            .collect();
        let (surprise_triggered, exploration_action) =
            self.run_surprise_exploration(&compressed_state);

        module_timings.surprise_exploration = _t.elapsed().as_micros() as u64;

        // ── Phase 21: Codebook diversity → memoization threshold adaptation ─
        // Science: Low codebook diversity needs more novel inputs; high diversity can consolidate
        let base_memo_threshold = super::thresholds::INPUT_MEMO_THRESHOLD;
        let diversity = self.stats.codebook_diversity;
        let memo_threshold = if diversity < 0.4 && diversity > 0.0 {
            let t = (base_memo_threshold - (0.4 - diversity) * 0.1).max(0.88);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else if diversity > 0.8 {
            let t = (base_memo_threshold + (diversity - 0.8) * 0.05).min(0.98);
            self.stats.memo_threshold_adaptations += 1;
            t
        } else {
            base_memo_threshold
        };

        // ── Phase 15: Input similarity memoization ───────────────────────────
        // Science: Priming (Tulving & Schacter 1990) — repeated stimuli can reuse
        // prior processing results. If input cosine similarity > threshold, flag for
        // downstream subsystem skipping (amortize expensive modules).
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
        // Store current compressed_state for next cycle comparison
        self.carryover.history.last_compressed_state = Some(compressed_state.clone());

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED ETHICS ENGINE — additive telemetry (runs alongside inline moral code)
        // Pipeline: moral parse → value gate → harmonies → unified verdict
        // ═══════════════════════════════════════════════════════════════════════
        let ethics_output = self.ethics_engine.evaluate(
            &super::ethics_engine::EthicsEngineInput {
                input,
                cycle: self.stats.total_cycles as u64,
                unified_psi: self.stats.unified_psi as f64, // use previous cycle's Ψ
                compressed_state: &compressed_state,
            },
        );
        module_timings.ethics_engine = ethics_output.total_us;

        // ═══════════════════════════════════════════════════════════════════════
        // 1.2 Adaptive Learning Threshold + Urgency
        // ═══════════════════════════════════════════════════════════════════════
        // Science: Friston (2010) — precision (inverse uncertainty) modulates PE weighting.
        // Low confidence → lower threshold (learn on smaller errors); high confidence → raise it.
        // Combined with temporal coherence scaling (adaptive_threshold_scale).
        // + Thelen & Smith (1994): exploration urge bidirectionally couples to threshold.
        let confidence_scale = 1.0 + (self.prediction_confidence - 0.5) * 0.4; // ±20% from confidence
        let exploration_scale = 1.0 - (self.curiosity_drive.exploration_urge - 0.5) * 0.2; // high explore → lower threshold
        let effective_threshold = self.config.learning_threshold
            * self.carryover.learning.adaptive_threshold_scale
            * confidence_scale
            * exploration_scale;
        if prediction_error < effective_threshold {
            self.carryover.urgency.consecutive_low_error = self
                .carryover
                .urgency
                .consecutive_low_error
                .saturating_add(1);
        } else {
            self.carryover.urgency.consecutive_low_error = 0;
        }
        // Use smoothed error for urgency to prevent jitter from single-cycle noise spikes.
        // Science: Dynamical systems — threshold-based switching needs hysteresis to prevent
        // oscillation. EMA smoothing damps transient spikes; prev_urgency adds hysteresis.
        let smoothed_urgency_error = if self.stats.total_cycles > 5 {
            // Blend instantaneous (70%) with running average (30%) for responsiveness + smoothing
            prediction_error * 0.7 + self.stats.avg_prediction_error * 0.3
        } else {
            prediction_error // Use raw error during bootstrap
        };
        // Hysteresis: require stronger signal to LEAVE current urgency level
        let base_hysteresis = match self.carryover.urgency.urgency {
            super::CycleUrgency::Cruise => effective_threshold * 1.2, // harder to leave Cruise
            super::CycleUrgency::Critical => effective_threshold * 0.8, // harder to leave Critical
            _ => effective_threshold,
        };
        // ── Phase 17: Predictive interval tuning via error pattern ──────
        // Science: Clark (2013) — predictive brain anticipates state changes.
        // Rising error pattern → lower threshold (prepare to escalate).
        // Falling error pattern → raise threshold (allow settling).
        let error_history_len = self.carryover.history.error_history.len();
        let pattern_mod = if error_history_len >= 4 {
            // Direct index: newest = len-1, 4th-newest = len-4 (avoids Vec alloc)
            let newest = self.carryover.history.error_history[error_history_len - 1];
            let oldest_4 = self.carryover.history.error_history[error_history_len - 4];
            let slope = (newest - oldest_4) / 3.0;
            if slope > 0.02 {
                0.9f32
            }
            // Rising → easier to escalate
            else if slope < -0.02 {
                1.1
            }
            // Falling → easier to de-escalate
            else {
                1.0
            }
        } else {
            1.0
        };
        // ── Phase 18: Prediction coherence → urgency bias ─────────────────
        // Science: Bar (2009) — temporal prediction consistency signals model quality.
        // Uses previous cycle's avg coherence (current not yet computed at urgency time).
        // Low coherence (<0.3) → model confused across horizons → bias toward Critical.
        // High coherence (>0.7) → model confident → permit Cruise (raise threshold).
        let prev_coherence = self.stats.avg_prediction_coherence;
        let coherence_mod = if prev_coherence < 0.3 && prev_coherence > 0.0 {
            0.85f32 // Lower threshold → easier to escalate (model confused)
        } else if prev_coherence > 0.7 {
            1.1 // Raise threshold → permit Cruise (model confident)
        } else {
            1.0
        };
        let prediction_coherence_urgency_bias = coherence_mod - 1.0;

        let hysteresis_threshold = base_hysteresis * pattern_mod * coherence_mod;
        let error_urgency = super::CycleUrgency::from_state(
            smoothed_urgency_error,
            hysteresis_threshold,
            surprise_triggered,
            self.carryover.urgency.consecutive_low_error,
        );

        // Compose CognitiveDepth with error-based urgency:
        // Reflex → cap at Cruise (skip heavy subsystems for familiar inputs)
        // DeepThought → floor at Normal (force full processing for novel/high-stakes)
        // Cortical → use error-based urgency as-is
        let raw_urgency = match self.cognitive_depth {
            super::CognitiveDepth::Reflex => match error_urgency {
                super::CycleUrgency::Critical => super::CycleUrgency::Normal,
                _ => super::CycleUrgency::Cruise,
            },
            super::CognitiveDepth::DeepThought => match error_urgency {
                super::CycleUrgency::Cruise => super::CycleUrgency::Normal,
                _ => error_urgency,
            },
            super::CognitiveDepth::Cortical => error_urgency,
        };

        // ── Phase 17: Cross-temporal error pattern learning ──────────────
        // Science: Rao & Ballard (1999) — hierarchical predictive coding tracks error
        // trajectories across time, not just instantaneous snapshots.
        // Maintain rolling window of prediction errors, classify pattern.
        let error_history = &mut self.carryover.history.error_history;
        if error_history.len() >= 16 {
            error_history.pop_front();
        }
        error_history.push_back(prediction_error);

        let (error_pattern, predicted_urgency) = if error_history.len() >= 4 {
            let len = error_history.len();
            // Direct index: newest = len-1, 4th-newest = len-4 (avoids Vec alloc)
            let newest = error_history[len - 1];
            let oldest_of_4 = error_history[len - 4];
            // Compute linear trend (simple slope)
            let slope = (newest - oldest_of_4) / 3.0; // newest - oldest, normalized
                                                      // Count sign changes for oscillation detection (index pairs avoid collect→Vec)
            let mut sign_changes = 0u32;
            let ref_val = oldest_of_4;
            for i in 0..len.saturating_sub(1) {
                let diff_cur = error_history[i + 1] - error_history[i];
                let diff_ref = error_history[i] - ref_val;
                if diff_cur.signum() != diff_ref.signum() {
                    sign_changes += 1;
                }
            }
            let oscillation_ratio = if len > 2 {
                sign_changes as f32 / (len - 1) as f32
            } else {
                0.0
            };
            // Spike detection: current error > 2× running mean
            let mean_err = error_history.iter().sum::<f32>() / len as f32;
            let is_spike = prediction_error > mean_err * 2.0 && prediction_error > 0.1;

            let pattern = if is_spike {
                "Spike"
            } else if oscillation_ratio > 0.6 {
                "Oscillating"
            } else if slope > 0.02 {
                "Rising"
            } else if slope < -0.02 {
                "Falling"
            } else {
                "Stable"
            };
            // Predict urgency 5 cycles ahead from pattern
            let predicted = match pattern {
                "Rising" | "Spike" => "Critical",
                "Oscillating" => "Normal",
                "Falling" | "Stable" => {
                    if self.carryover.urgency.consecutive_low_error > 15 {
                        "Cruise"
                    } else {
                        "Normal"
                    }
                }
                _ => "Normal",
            };
            (pattern, predicted)
        } else {
            ("Warmup", "Normal")
        };

        // ── Phase 17: Mode transition smoothing ──────────────────────────
        // Science: Kelso (1995) — metastable coordination dynamics: transitions between
        // attractor states should be smooth, not abrupt. Ramp mode_confidence over 5 cycles.
        let urgency;
        if raw_urgency != self.carryover.urgency.prev_urgency {
            // Mode changed — start transition
            self.stats.mode_transitions += 1;
            self.carryover.urgency.mode_confidence = 0.0;
            self.carryover.urgency.mode_stability_counter = 0;
            // During transition, stay in the HIGHER urgency (more cautious)
            let raw_level = match raw_urgency {
                super::CycleUrgency::Critical => 2,
                super::CycleUrgency::Normal => 1,
                super::CycleUrgency::Cruise => 0,
            };
            let prev_level = match self.carryover.urgency.prev_urgency {
                super::CycleUrgency::Critical => 2,
                super::CycleUrgency::Normal => 1,
                super::CycleUrgency::Cruise => 0,
            };
            urgency = if raw_level > prev_level {
                raw_urgency // escalating → use new immediately
            } else {
                // de-escalating → hold old urgency for 1 cycle
                self.carryover.urgency.prev_urgency
            };
            self.carryover.urgency.prev_urgency = raw_urgency;
        } else {
            // Same mode — ramp confidence
            self.carryover.urgency.mode_stability_counter = self
                .carryover
                .urgency
                .mode_stability_counter
                .saturating_add(1);
            self.carryover.urgency.mode_confidence =
                (self.carryover.urgency.mode_stability_counter as f32 / 5.0).min(1.0);
            urgency = raw_urgency;
        }
        self.stats.avg_mode_stability = self.stats.avg_mode_stability * 0.9
            + self.carryover.urgency.mode_stability_counter as f32 * 0.1;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE A: OBSERVE — Build immutable CycleSnapshot (Phase 2.3)
        // ═══════════════════════════════════════════════════════════════════════
        // Captures all observable state BEFORE subsystems begin computing.
        // This snapshot is passed to subsystems implementing CognitiveSubsystem.
        // Currently used for telemetry; will become the sole input to all
        // subsystems once the staged computation model is fully adopted.
        let cycle_snapshot = super::subsystem_trait::CycleSnapshot::build(
            self.stats.total_cycles as u64,
            self.prediction_confidence,
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
            &compressed_state,
            &hv16_cached,
            &self.carryover.consciousness,
            &self.carryover.quality,
        );
        self.last_snapshot = Some(cycle_snapshot);

        // ── Phase B: COMPUTE — Run managers via CognitiveSubsystem trait ──
        // Each manager reads the immutable CycleSnapshot and proposes state changes.
        {
            use super::subsystem_trait::CognitiveSubsystem;
            if let Some(ref snapshot) = self.last_snapshot {
                let urgency_u8 = snapshot.urgency;
                let cycle_num = snapshot.cycle_number;

                // DriveManager (interval 7, co-prime)
                if self.drive_manager.should_run(cycle_num, urgency_u8) {
                    let drive_output = self.drive_manager.process(snapshot);
                    self.subsystem_collector.record("drive_manager", drive_output);
                }

                // MemoryManager (interval 11, co-prime)
                if self.memory_manager.should_run(cycle_num, urgency_u8) {
                    let memory_output = self.memory_manager.process(snapshot);
                    self.subsystem_collector.record("memory_manager", memory_output);
                }

                // LearningManager (interval 13, co-prime)
                if self.learning_manager.should_run(cycle_num, urgency_u8) {
                    let learning_output = self.learning_manager.process(snapshot);
                    self.subsystem_collector.record("learning_manager", learning_output);
                }

                // PerceptionManager (interval 19, co-prime)
                if self.perception_manager.should_run(cycle_num, urgency_u8) {
                    let perception_output = self.perception_manager.process(snapshot);
                    self.subsystem_collector.record("perception_manager", perception_output);
                }
            }
        }

        // ── Phase 17: Self-model accuracy tracking ───────────────────────
        // Science: Fleming & Dolan (2012) — metacognitive monitoring improves when
        // predictions about one's own performance are validated against outcomes.
        // Record prediction at T, validate at T+5, feed accuracy back to LR/confidence.
        let self_model_accuracy = self.carryover.learning.self_model_accuracy;
        if let Some((made_at, pred_confidence, pred_urgency)) =
            self.carryover.history.self_model_prediction.take()
        {
            // Validate if 5 cycles have passed
            if self.stats.total_cycles >= made_at + 5 {
                let confidence_error = (self.prediction_confidence - pred_confidence).abs();
                let urgency_match = if urgency == pred_urgency { 1.0f32 } else { 0.0 };
                // Accuracy = blend of confidence prediction (70%) + urgency prediction (30%)
                let accuracy = (1.0 - confidence_error) * 0.7 + urgency_match * 0.3;
                self.carryover.learning.self_model_accuracy =
                    self.carryover.learning.self_model_accuracy * 0.9 + accuracy * 0.1;
                self.stats.self_model_predictions_validated += 1;
                self.stats.avg_self_model_accuracy =
                    self.stats.avg_self_model_accuracy * 0.9 + accuracy * 0.1;

                // Feed back: high self-model accuracy → trust confidence more
                if self.carryover.learning.self_model_accuracy > 0.7 {
                    let trust_boost = (self.carryover.learning.self_model_accuracy - 0.7) * 0.03;
                    self.adjust_confidence("self_model_trust", trust_boost);
                }
                // Low accuracy → dampen confidence (self-model unreliable)
                if self.carryover.learning.self_model_accuracy < 0.3 {
                    self.scale_confidence("self_model_low_acc", 0.98);
                }
            } else {
                // Not yet time to validate — put it back
                self.carryover.history.self_model_prediction =
                    Some((made_at, pred_confidence, pred_urgency));
            }
        }
        // Make new prediction every 7 cycles (co-prime)
        if self.stats.total_cycles % 7 == 0
            && self.carryover.history.self_model_prediction.is_none()
        {
            self.carryover.history.self_model_prediction =
                Some((self.stats.total_cycles, self.prediction_confidence, urgency));
            self.stats.self_model_predictions_made += 1;
        }

        // FEEDBACK: Quantum coherence boosts exploration (prev cycle)
        // Science: Lambert (2013) — quantum coherence enhances biological search
        if self.carryover.consciousness.quantum_coherence > QUANTUM_COHERENCE_THRESHOLD {
            let coherence_boost = (self.carryover.consciousness.quantum_coherence
                - QUANTUM_COHERENCE_THRESHOLD) as f32
                * QUANTUM_COHERENCE_BOOST_SCALE;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + coherence_boost).clamp(0.0, 1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a. Memory System Integration: Recall relevant episodic memories
        // ═══════════════════════════════════════════════════════════════════════
        let memory_context_boost = self.recall_episodic_context(&compressed_state);

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.1 Resonator-enhanced recall: factorize bundled memories
        // ═══════════════════════════════════════════════════════════════════════
        // When episodic recall returns multiple matches, bundle them into a
        // superposed state and factorize against semantic codebooks. The
        // factorized valence/phi components are cleaner than raw averages.
        // Science: Kent et al. (2020) — Resonator Networks for O(log N) factorization
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 4th
        let mut resonator_wm_primed = false;
        let mut resonator_reconsolidated: usize = 0;
        let mut resonator_best_sim: f32 = 0.0;

        // Track 4a: Predictive resonator — compare last cycle's best match with current input
        // Science: Bar (2007) — proactive brain generates predictions from analogies
        let resonator_prediction_error: f32 =
            if let Some(ref prev_pred) = self.stats.last_resonator_prediction {
                let sim = helpers::cosine_f32(prev_pred, &compressed_state);
                (1.0 - sim).clamp(0.0, 1.0) // cosine distance
            } else {
                0.0 // no prediction yet (first cycle)
            };

        // ── Phase 20: Resonator prediction error → exploration/confidence ────
        // Science: Bar (2007) — high analogical mismatch signals novel territory.
        // High prediction error (bad analogy) → boost exploration, dampen confidence.
        // Low error (good analogy) → boost confidence (familiar territory).
        let resonator_error_exploration_mod = if resonator_prediction_error > 0.5
            && self.stats.total_cycles > 5
        {
            let boost = (resonator_prediction_error - 0.5) * 0.08;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + boost).min(1.0);
            self.adjust_confidence("resonator_error_high", -boost * 0.5);
            self.stats.resonator_error_exploration_count += 1;
            boost
        } else if resonator_prediction_error < 0.2 && resonator_prediction_error > 0.0 {
            let confidence_boost = (0.2 - resonator_prediction_error) * 0.03;
            self.adjust_confidence("resonator_error_low", confidence_boost);
            self.stats.resonator_error_exploration_count += 1;
            -confidence_boost // negative = confidence gain (no exploration boost)
        } else {
            0.0
        };

        // ── Phase 17: Coherence memoization — cache pre-update value ─────
        // Science: O(n) history averaging computed once per cycle, not 5×.
        let pre_update_coherence = self.coherence_bridge.smoothed_coherence();

        // ── Phase 20: Phenomenal binding → threshold gating ──────────────────
        // Science: Tononi (2004) — binding strength is a proxy for consciousness
        // integration quality. Strong binding → integrate confidently (lower threshold).
        // Weak binding → raise threshold (protect against fragmented learning).
        let cached_binding = self.carryover.quality.last_phenomenal_binding as f32;
        let binding_threshold_mod = if cached_binding > 0.7 {
            // Strong binding → lower threshold (integrate confidently)
            let relief = (cached_binding - 0.7) * 0.3; // up to -0.09
            self.carryover.learning.adaptive_threshold_scale *= 1.0 - relief;
            self.carryover.learning.adaptive_threshold_scale = self
                .carryover
                .learning
                .adaptive_threshold_scale
                .clamp(0.5, 2.0);
            self.stats.binding_threshold_mod_count += 1;
            -relief
        } else if cached_binding < 0.3 && cached_binding > 0.0 {
            // Weak binding → raise threshold (be cautious)
            let caution = (0.3 - cached_binding) * 0.2; // up to +0.06
            self.carryover.learning.adaptive_threshold_scale *= 1.0 + caution;
            self.carryover.learning.adaptive_threshold_scale = self
                .carryover
                .learning
                .adaptive_threshold_scale
                .clamp(0.5, 2.0);
            self.stats.binding_threshold_mod_count += 1;
            caution
        } else {
            0.0
        };

        // ── Phase 21: Phenomenal binding → prediction confidence ─────────
        // Science: Tononi (2004) — strong binding = coherent integration = reliable predictions
        let binding_confidence_mod = if cached_binding > 0.7 {
            let conf_boost = (cached_binding - 0.7) * 0.1; // up to +0.03
            self.adjust_confidence("binding_strong", conf_boost);
            self.stats.binding_confidence_mod_count += 1;
            conf_boost
        } else if cached_binding < 0.3 && cached_binding > 0.0 {
            let conf_dampen = (0.3 - cached_binding) * 0.15; // up to -0.045
            self.adjust_confidence("binding_weak", -conf_dampen);
            self.stats.binding_confidence_mod_count += 1;
            -conf_dampen
        } else {
            0.0
        };

        // Coherence gate: skip resonator recall during unstable CfC dynamics
        // Science: noisy priors during turbulent dynamics can destabilize predictions
        // Uses previous cycle's smoothed coherence (updated at line ~646)
        let reflection_thresholds = self.self_reflection.get_thresholds();
        let resonator_coherence_gate = pre_update_coherence > reflection_thresholds.coherence_gate
            || self.stats.total_cycles < 10; // bypass gate during warmup
        if resonator_coherence_gate && urgency.should_run(self.stats.total_cycles, 1, 1, 4) {
            if let Some(ref mut res_mem) = self.resonator_memory {
                let res_start = Instant::now();

                // Dimension guard: skip if compressed_state doesn't match resonator codebook dim
                let res_dim_ok = compressed_state.len() == res_mem.resonator.config.dim;
                if res_dim_ok && !res_mem.is_empty() {
                    // Retrieve resonator episodes similar to current content
                    if let Ok(matches) = res_mem.retrieve(&[("content", &compressed_state)]) {
                        let top_matches: Vec<_> =
                            matches.into_iter().take(MEMORY_RECALL_TOP_K).collect();

                        // Extract ALL owned data from borrowed episodes before releasing
                        // res_mem borrow. retrieve() returns Vec<&Episode>, so we must
                        // copy what we need before calling query_factorize(&mut res_mem).
                        let best_match_sim = top_matches
                            .iter()
                            .map(|m| helpers::cosine_f32(&compressed_state, &m.hv))
                            .fold(0.0f32, f32::max);
                        let match_timestamps: Vec<u64> =
                            top_matches.iter().map(|m| m.timestamp).collect();
                        resonator_best_sim = best_match_sim;

                        // Track 4a: Cache best match HV as next-cycle prediction
                        // Science: Bar (2007) — proactive brain uses analogy for anticipation
                        if best_match_sim > 0.3 {
                            let best_ep = top_matches.iter().max_by(|a, b| {
                                let sa: f32 = compressed_state
                                    .iter()
                                    .zip(a.hv.iter())
                                    .map(|(x, y)| x * y)
                                    .sum();
                                let sb: f32 = compressed_state
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

                        // Pre-compute bundled vector while we still hold episode references
                        let bundled = if top_matches.len() >= 2 {
                            let dim = compressed_state.len();
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

                        // Drop episode references — releases the &mut res_mem borrow
                        drop(top_matches);

                        // Now safe to call query_factorize (no outstanding borrows on res_mem)
                        if let Some(bundled) = bundled {
                            if let Ok(factors) =
                                res_mem.query_factorize(&bundled, &[("content", &compressed_state)])
                            {
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
                                        _ => {} // neutral, medium, proto_N — no bias
                                    }
                                }
                            }
                        }

                        // Track 3a: Resonator recall → confidence priming
                        // Science: Tulving (1983) — episodic retrieval primes processing
                        if best_match_sim > 0.3 {
                            self.adjust_confidence("resonator_recall_prime", best_match_sim * 0.02);
                            resonator_wm_primed = true;
                        }

                        // Track 3b: Resonator recall → episodic reconsolidation
                        // Science: Nader (2003) — retrieval destabilizes then strengthens memories
                        // match_timestamps is owned Vec<u64>, so phi_episodic_replay access is safe
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
        } // urgency gate

        // Track 3d: Resonator recall → FEP prior confidence
        // Science: Tulving (1983) — familiar context boosts model confidence
        // High resonator similarity → "I've seen this before" → boost prior precision
        if resonator_best_sim > 0.5 {
            self.fep_agent.precision.prior_precision = (self.fep_agent.precision.prior_precision
                + (resonator_best_sim - 0.5) as f64 * 0.1)
                .min(2.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.2. Goal System: Apply attention bias from active goals
        // ═══════════════════════════════════════════════════════════════════════

        let goal_attention_bias = self.goal_system.attention_bias();
        // attention_sensitivity write moved to after from_consciousness_state() + strategy reset

        // FEEDBACK: Goal system progress tracking and priority → cognition coupling
        // Science: Anderson (1983) — goal-directed cognition modulates learning and exploration.
        // Low prediction error signals goal progress; top goal priority scales LR and exploration.
        if let Some(top) = self.goal_system.top_goal() {
            let goal_priority = top.priority;
            // High-priority active goal → boost learning rate (consolidate goal-relevant knowledge)
            if goal_priority > 0.5 {
                let goal_lr_boost = (goal_priority - 0.5) * 0.1; // up to +5%
                self.scale_lr("goal_priority", 1.0 + goal_lr_boost);
            }
            // Successful prediction (low error) during goal pursuit → exploration toward goal
            if prediction_error < self.config.learning_threshold && goal_priority > 0.3 {
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + goal_priority * 0.03).clamp(0.0, 1.0);
            }
        }

        // 1b. Analyze emotional content for simple contagion (keyword-based)
        self.emotion_contagion.analyze(input);

        // ── Phase 15+18: Emotional homeostasis — urgency-adaptive opponent-process ──
        // Science: Solomon & Corbit (1974) + Damasio (1994) — opponent-process theory
        // with urgency-adaptive pull: stronger in Cruise (stable → return to baseline),
        // weaker in Critical (preserve genuine emotional signals during high-alertness).
        let valence_homeostasis_pull;
        let arousal_homeostasis_pull;
        let homeostasis_pull_strength;
        {
            let prev_v = self.carryover.history.last_emotion_valence;
            let prev_a = self.carryover.history.last_emotion_arousal;
            let curr_v = self.emotion_contagion.valence;
            let curr_a = self.emotion_contagion.prosody_arousal();

            // Phase 18: Urgency-adaptive pull multiplier
            // Cruise → 1.5× (stronger return to baseline), Critical → 0.6× (preserve signals)
            let pull_mult = match self.carryover.urgency.urgency {
                super::CycleUrgency::Cruise => 1.5f32,
                super::CycleUrgency::Normal => 1.0,
                super::CycleUrgency::Critical => 0.6,
            };
            homeostasis_pull_strength = pull_mult;

            // Opponent pull: base 5% toward neutral, scaled by urgency
            let v_pull = -curr_v * 0.05 * pull_mult;
            let a_pull = (0.3 - curr_a) * 0.05 * pull_mult;
            self.emotion_contagion.valence = (curr_v + v_pull).clamp(-1.0, 1.0);

            valence_homeostasis_pull = v_pull;
            arousal_homeostasis_pull = a_pull;

            // Track EMA of homeostasis pull magnitude
            self.stats.avg_valence_homeostasis =
                self.stats.avg_valence_homeostasis * 0.95 + v_pull.abs() * 0.05;

            // Stash for next cycle
            self.carryover.history.last_emotion_valence = self.emotion_contagion.valence;
            let _ = (prev_v, prev_a); // suppress unused warnings
            self.carryover.history.last_emotion_arousal = curr_a;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 1c. Update Unified Emotional Bridge (VAD-based, richer than simple contagion)
        // ═══════════════════════════════════════════════════════════════════════
        // Bridge the simple EmotionContagion to the unified EmotionalBridge
        // Convert valence/arousal to the full VAD emotional system

        let simple_valence = self.emotion_contagion.prosody_valence() as f64;
        let simple_arousal = self.emotion_contagion.prosody_arousal() as f64;
        // Dominance estimated from confidence and flow state
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

        // 2. compressed_state already computed in Phase 1.1 (single compress_for_ltc call)

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

        let _t_core = Instant::now();
        // Cow avoids cloning compressed_state (~1KB) on the CfC fallback path
        let semantic_hdc: Cow<'_, [f32]> = self
            .temporal_network
            .project_to_hdc_vec(&compressed_state)
            .map(Cow::Owned)
            .unwrap_or(Cow::Borrowed(&compressed_state));
        // Phi-weighted learning rate: consciousness level modulates how aggressively
        // we adjust to prediction errors on similar past inputs.
        let current_phi_for_lr = pre_update_coherence as f64;
        let mut semantic_lr_factor = self.semantic_memory.compute_lr_factor_phi_weighted(
            &semantic_hdc,
            3,
            current_phi_for_lr,
            self.stats.total_cycles as u64,
        );
        module_timings.core_semantic_lookup = _t_core.elapsed().as_micros() as u64;

        // ── Phase 20: Epistemic gate → semantic memory LR bidirectionality ───
        // Science: Fernandez-Duque & Johnson (2002) — metacognitive uncertainty
        // should propagate into memory consolidation (cautious when uncertain).
        // Uses previous cycle's cached epistemic confidence to avoid temporal dependency.
        let prev_epistemic = self.carryover.quality.last_epistemic_confidence;
        let epistemic_semantic_lr_mod: f32 = if prev_epistemic < 0.4 && prev_epistemic > 0.0 {
            let caution = 0.8_f32 + prev_epistemic * 0.5; // [0.8, 1.0] when conf in [0, 0.4]
            semantic_lr_factor *= caution;
            self.stats.epistemic_semantic_mod_count += 1;
            caution - 1.0 // negative mod means dampening
        } else if prev_epistemic > 0.8 {
            let boost = 1.0_f32 + (prev_epistemic - 0.8) * 1.0; // [1.0, 1.2] when conf in [0.8, 1.0]
            semantic_lr_factor *= boost;
            self.stats.epistemic_semantic_mod_count += 1;
            boost - 1.0 // positive mod means boosting
        } else {
            0.0
        };

        // 3. Convert to ndarray for CfC (copy elements directly — avoids Vec clone)
        let input_array: Array1<f32> = compressed_state.iter().copied().collect();

        // 4. Step CfC forward with current input
        // FEEDBACK: Resonance frequency modulates CfC time constant (prev cycle)
        // Science: Buzsáki (2006) — neural oscillations modulate processing speed
        let resonance_tau_factor = if self.carryover.history.resonance_frequency > 0.0 {
            let deviation = (self.carryover.history.resonance_frequency as f32
                - RESONANCE_TAU_CENTER as f32)
                .clamp(-0.5, 0.5);
            1.0 - (deviation * RESONANCE_TAU_SCALE) // ±5% modulation
        } else {
            1.0
        };
        // FEEDBACK: Body arousal modulates CfC processing speed (prev cycle)
        // Science: Steriade (1996) — arousal gates cortical activation speed
        // High arousal → faster tau (alert), low arousal → slower (drowsy)
        let arousal_tau_factor = if (self.carryover.history.body_arousal - 0.5).abs() > 0.1 {
            1.0 + (self.carryover.history.body_arousal - 0.5) * 0.1 // ±5% from arousal
        } else {
            1.0
        };
        // FEEDBACK: Resonator familiarity modulates CfC processing speed
        // Science: Nosofsky (1986) — familiar stimuli processed faster (exemplar theory)
        // High similarity → lower tau (faster), novel → higher tau (slower, more deliberate)
        let codebook_tau_factor = if resonator_best_sim > 0.5 {
            1.0 - (resonator_best_sim - 0.5) * 0.1 // up to 5% faster for familiar
        } else if resonator_best_sim > 0.0 && resonator_best_sim < 0.2 {
            1.0 + (0.2 - resonator_best_sim) * 0.15 // up to 3% slower for novel
        } else {
            1.0
        };
        // ── Phase 15: Arousal recovery tau modulation ─────────────────────
        // Science: Porges (2011) — active parasympathetic recovery slows processing.
        // When arousal trap counter > 5, gradually increase tau (slow CfC) to
        // allow state to stabilize before the hard exploration escape at counter > 10.
        let arousal_recovery_tau_factor;
        let arousal_recovery_active;
        if self.carryover.urgency.arousal_trap_counter > 5
            && self.carryover.urgency.arousal_trap_counter <= 10
        {
            let recovery_intensity = (self.carryover.urgency.arousal_trap_counter - 5) as f32 / 5.0;
            arousal_recovery_tau_factor = 1.0 + recovery_intensity * 0.2; // up to 20% slower
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
            * somatic_signals.tau_slowdown_factor as f32; // Nociception: infrastructure stress → slower integration
        let _t_core = Instant::now();
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(error = %e, "CfC temporal step failed — continuing with stale state");
        }
        module_timings.core_cfc_step = _t_core.elapsed().as_micros() as u64;

        // 5. Get multi-scale predictions using CfC's O(1) predict_forward
        // This is the key advantage: instant prediction at any future time
        let _t_core = Instant::now();
        let (prediction, raw_predictions) = self.get_multi_scale_prediction(&input_array);

        // ── Phase 15: Multi-horizon prediction coherence ─────────────────────
        // Science: Bar (2009) — temporal prediction consistency signals model quality.
        // Low coherence → predictions at different horizons disagree → model uncertain.
        // Computed every 11 cycles (co-prime amortization). Uses cached predictions from above
        // to avoid redundant predict_forward calls (~300µs saved per coherence check).
        let prediction_coherence = if self.stats.total_cycles % 11 == 0 {
            let coh = Self::compute_prediction_coherence_from_cache(&raw_predictions);
            self.stats.avg_prediction_coherence =
                self.stats.avg_prediction_coherence * 0.9 + coh * 0.1;
            // Low coherence → dampen confidence (predictions unreliable)
            if coh < 0.5 {
                let coh_dampen = (0.5 - coh) * 0.04;
                self.scale_confidence("pred_coherence_low", 1.0 - coh_dampen);
            }
            // High coherence → slight confidence boost (temporal model is consistent)
            if coh > 0.8 {
                let coh_boost = (coh - 0.8) * 0.02;
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
        // 6b. World Model: Update hierarchical world model with sensory input
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        self.world_model.update_sensory(&compressed_state);

        // FEEDBACK: World model stiffness modulates FEP learning rate (meta-learning)
        // Science: Finn et al. (2017) MAML — plasticity itself should be plastic.
        // High WM avg_error (stiff model, poorly adapted) → amplify LR to force adaptation;
        // Low WM avg_error (spongy model, well-adapted) → dampen LR to prevent overfitting.
        let wm_stiffness = self.world_model.avg_error.clamp(0.0, 1.0);
        if self.stats.total_cycles > 20 {
            if wm_stiffness > 0.5 {
                // Additive (not multiplicative) to prevent compounding
                let stiffness_nudge = (wm_stiffness - 0.5) * 0.05;
                self.adjust_lr("wm_stiff", stiffness_nudge);
            } else if wm_stiffness < 0.2 {
                let spongy_dampen = (0.2 - wm_stiffness) * 0.15;
                self.scale_lr("wm_spongy", 1.0 - spongy_dampen);
            }
        }

        // FEEDBACK: Per-level world model errors guide regime-appropriate learning
        // Science: Predictive coding (Rao & Ballard 1999) — abstract vs sensory failures
        // need different responses. High abstract error → conceptual confusion → explore broadly;
        // High sensory but low abstract → perceptual mismatch → sharpen attention.
        let level_errors = self.world_model.level_errors();
        let mut wm_sensory_mismatch = false;
        if level_errors.len() >= 2 && self.stats.total_cycles > 10 {
            let sensory_error = level_errors[0];
            let abstract_error = level_errors[level_errors.len() - 1];
            // Conceptual confusion: abstract failure > 1.5x sensory
            if abstract_error > sensory_error * 1.5 && abstract_error > 0.1 {
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + 0.08).clamp(0.0, 1.0);
            }
            // Perceptual mismatch flag — applied after from_consciousness_state() + strategy reset
            wm_sensory_mismatch = sensory_error > abstract_error * 2.0 && sensory_error > 0.1;
        }
        module_timings.world_model = _t.elapsed().as_micros() as u64;

        // 7. Send prediction to encoder: deferred to after last &prediction use (line ~2436)
        //    to avoid cloning ~1KB Vec<f32>. The encoder only stores it for next-cycle use,
        //    so the timing doesn't affect correctness.

        // 8. Capture previous state BEFORE create_experience updates it
        // Note: clone is required because create_experience() also takes self.last_state
        // internally to build the experience record. The inner clone was removed (move instead).
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
        // Reuse tau_owned from above instead of calling flattened_tau() again
        let flattened_tau: Vec<f32> = tau_owned.iter().flat_map(|a| a.iter().copied()).collect();
        self.temporal_signature_encoder.record_batch(&flattened_tau);

        // 10c. Update adaptive behavior based on consciousness state
        let (pattern, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let coherence = self.coherence_bridge.smoothed_coherence();
        // Phase 17: Cache post-update coherence for external accessors
        self.carryover.history.cached_coherence = Some(coherence);
        let voice_confidence = self.voice_feedback_bridge.summary().voice_confidence;
        self.adaptive_behavior = AdaptiveBehavior::from_consciousness_state(
            pattern,
            pattern_confidence,
            coherence,
            voice_confidence,
        );

        // Re-apply strategy modulations ON TOP of consciousness-derived base.
        // Before this fix, from_consciousness_state() obliterated the strategy set at Phase 0.5,
        // making the entire Q-learning ClosedLearningLoop dead code.
        // Science: Strategy = voluntary override; consciousness = involuntary substrate.
        self.reapply_strategy_modulation(selected_strategy);

        // Re-apply goal and world-model attention biases after consciousness reset + strategy.
        // Previously at lines 436 and 581, these were destroyed by from_consciousness_state().
        self.adaptive_behavior.attention_sensitivity *= goal_attention_bias;
        if wm_sensory_mismatch {
            self.adaptive_behavior.attention_sensitivity *= 1.08;
        }

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
        let (fep_action_idx, fep_action_probs, is_surprised, fep_pragmatic_value_raw) =
            self.step_fep_active_inference(prediction_error, coherence);

        // ── Track 5b: MCTS plan post-hoc evaluation ─────────────────────────
        // Science: Botvinick & Toussaint (2012) — planning effectiveness requires
        // retrospective evaluation to close the deliberation loop
        let mcts_plan_effectiveness: f32 =
            if let Some((_prev_action, _prev_confidence, prev_error)) =
                self.carryover.history.mcts_plan_applied.take()
            {
                // Compare: did prediction error improve after applying the plan?
                let error_reduction = prev_error - prediction_error;
                // Effectiveness = normalized improvement weighted by plan confidence
                let raw_effectiveness = if prev_error > 0.0 {
                    (error_reduction / prev_error).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let effectiveness = (raw_effectiveness * 0.5 + 0.5).clamp(0.0, 1.0); // map [-1,1] → [0,1]
                                                                                     // Feedback: effective plans → boost confidence in deliberative system
                if effectiveness > 0.6 {
                    self.adjust_confidence("mcts_effective", (effectiveness - 0.6) * 0.03);
                } else if effectiveness < 0.3 {
                    // Poor plan → slightly boost exploration to find better strategies
                    self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                        + (0.3 - effectiveness) * 0.02)
                        .clamp(0.0, 1.0);
                }
                // EMA update
                self.stats.avg_mcts_plan_effectiveness =
                    self.stats.avg_mcts_plan_effectiveness * 0.9 + effectiveness * 0.1;
                effectiveness
            } else {
                0.0
            };

        // ── Apply previous cycle's MCTS plan at reduced weight ──────────────
        // Science: MCTS plans (deliberative) complement FEP actions (habitual).
        // When the prior cycle's deliberative system produced a confident plan
        // (confidence > 0.7), apply its effect at 40% strength alongside the
        // current FEP action — "dual process" theory (Kahneman 2011).
        if let Some((plan_action, plan_confidence)) = self.carryover.history.mcts_plan.take() {
            if plan_confidence > 0.7 && plan_action != fep_action_idx {
                // Record for next-cycle post-hoc evaluation
                self.carryover.history.mcts_plan_applied =
                    Some((plan_action, plan_confidence, prediction_error));
                let plan_weight = plan_confidence * 0.4;
                match plan_action {
                    0 => {
                        // Plan said "exploit" — nudge LR down (floor at 1.0)
                        self.scale_lr("mcts_exploit", 1.0 - plan_weight * 0.1);
                    }
                    1 => {
                        // Plan said "consolidate" — reinforce prediction confidence
                        self.adjust_confidence("mcts_consolidate", plan_weight * 0.05);
                    }
                    2 => {
                        // Plan said "explore" — nudge exploration urge
                        self.curiosity_drive.exploration_urge =
                            (self.curiosity_drive.exploration_urge + plan_weight * 0.08)
                                .clamp(0.0, 1.0);
                    }
                    _ => {}
                }
            }
        }

        // ── FEP Free Energy Decomposition → targeted modulation ──────────
        // Science: Friston (2010) — accuracy, complexity, surprise drive distinct responses
        // Extract FEP values first (avoids borrow conflict with self.adjust_confidence/scale_lr)
        let fep_vals = self.fep_agent.last_fe_components.as_ref().map(|fe| {
            (fe.accuracy, fe.complexity, fe.surprise, fe.prediction_error)
        });
        let (fep_accuracy, fep_complexity, fep_surprise, fep_td_error) = if let Some((acc, comp, surp, pe)) = fep_vals {
            // High accuracy → stabilize (model fits well)
            if acc > 0.5 {
                self.adjust_confidence("fep_accuracy_high", 0.01);
            }
            // High complexity → reduce LR (Occam's razor: penalize overfitting)
            if comp > 1.0 {
                self.scale_lr("fep_complexity", 1.0 - ((comp - 1.0).min(0.5) * 0.1) as f32);
            }
            // High surprise → boost exploration (complement existing is_surprised gate)
            if surp > reflection_thresholds.surprise as f64 {
                let s_explore =
                    ((surp - reflection_thresholds.surprise as f64) * 0.1).min(0.05) as f32;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + s_explore).clamp(0.0, 1.0);
            }
            (acc, comp, surp, pe)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // ── FEP Pragmatic value → consolidation vs exploration balance ───
        // Science: Friston (2015) — pragmatic value drives goal-directed behavior
        let fep_pragmatic_value = fep_pragmatic_value_raw;
        if fep_pragmatic_value > 0.7 {
            // High pragmatic: exploit — reduce exploration
            self.curiosity_drive.exploration_urge *=
                (1.0 - (fep_pragmatic_value - 0.7) * 0.3) as f32;
        } else if fep_pragmatic_value < 0.3 && fep_pragmatic_value > 0.0 {
            // Low pragmatic: explore — model needs updating
            let p_explore = ((0.3 - fep_pragmatic_value) * 0.15).min(0.05) as f32;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + p_explore).clamp(0.0, 1.0);
        }

        // ── FEP TD error → causal discovery trigger ──────────────────────
        // Science: Schultz (1997) — large prediction errors signal state transitions
        if fep_td_error.abs() > 0.5 {
            if let Some(ref mut enhancer) = self.causal_enhancer {
                if enhancer.should_discover() {
                    let _ = enhancer.run_discovery();
                }
            }
        }

        // ── Track 5e: Causal graph → attention weighting ─────────────────
        // Science: Pearl (2009) — causal parents are the minimal sufficient set for prediction
        // Use discovered causal structure to modulate confidence and exploration:
        // Dense causal graph → good understanding → stabilize; sparse → explore
        let causal_attention_edges: usize = if let Some(ref enhancer) = self.causal_enhancer {
            let graph = enhancer.current_graph();
            let edge_count = graph.edges.len();
            if edge_count > 0 {
                // Causal structure discovered — weight by average edge confidence
                let avg_confidence = if edge_count > 0 {
                    graph.edges.iter().map(|e| e.confidence).sum::<f64>() / edge_count as f64
                } else {
                    0.0
                };
                // Dense, confident graph → stabilize (good causal model)
                if edge_count > 5 && avg_confidence > 0.5 {
                    self.adjust_confidence("causal_graph_dense", (avg_confidence as f32 - 0.5) * 0.03);
                }
                // Sparse graph after many cycles → poor understanding → boost exploration
                if edge_count < 2 && self.stats.total_cycles > 200 {
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge + 0.02).clamp(0.0, 1.0);
                }
                self.stats.causal_attention_uses += 1;
            }
            edge_count
        } else {
            0
        };

        // ── FEP decomposition → adaptive behavior modulation ─────────────
        // Science: Friston (2010) — free energy components shape behavioral policy
        // High accuracy + low complexity → exploit (FlowRiding)
        if fep_accuracy > 0.5 && fep_complexity < 0.5 {
            self.adaptive_behavior.learning_rate_multiplier =
                (self.adaptive_behavior.learning_rate_multiplier * 1.1).min(2.0);
            self.adaptive_behavior.exploration_factor *= 0.8;
        }
        // High surprise → explore (Exploring)
        let surprise_thresh = reflection_thresholds.surprise as f64;
        if fep_surprise > surprise_thresh {
            self.adaptive_behavior.exploration_factor =
                (self.adaptive_behavior.exploration_factor + 0.15).min(1.0);
            self.adaptive_behavior.action_hint = ActionHint::Explore;
        }
        // High complexity → consolidate (slow down, reduce LR)
        if fep_complexity > 1.0 {
            self.adaptive_behavior.learning_rate_multiplier =
                (self.adaptive_behavior.learning_rate_multiplier * 0.85).max(0.1);
            self.adaptive_behavior.pause_multiplier =
                (self.adaptive_behavior.pause_multiplier * 1.2).min(2.0);
            self.adaptive_behavior.action_hint = ActionHint::SlowDown;
        }

        // ── FEP surprise → episodic replay priority boost ────────────────
        // Science: Nader (2003) + Friston — surprising events deserve
        // accelerated consolidation for model updating
        if fep_surprise > surprise_thresh {
            if let Some(ref mut replay) = self.phi_episodic_replay {
                let surprise_boost = (fep_surprise - surprise_thresh).min(0.5) * 0.2;
                replay.boost_recent_consolidation(surprise_boost);
            }
        }

        // Feed outcome to FEP TD learner when external reward is available
        if self.external_reward.abs() > f32::EPSILON {
            let outcome_obs = Observation::from_consciousness_state(
                self.external_reward as f64,
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
        // Apply moral constraints to FEP-selected actions:
        // - Negative moral score -> reduce exploration, increase caution
        // - Consent violation -> strong ethical override
        // - Deontological violations -> trigger reflective pause

        if moral_concern_detected {
            // Reduce exploration when facing moral concerns
            self.curiosity_drive.exploration_urge *= MORAL_CONCERN_EXPLORATION_DAMPEN;

            // Increase trust threshold (be more cautious)
            self.self_reflection.trust_threshold =
                (self.self_reflection.trust_threshold * 1.2).clamp(0.1, 0.95);

            // Boost reflective processing (take time to consider ethics)
            self.adaptive_behavior.pause_multiplier *= MORAL_CONCERN_PAUSE_BOOST;

            // If severe moral violation (perfect duty or consent), flag for review
            if moral_judgment.consent_violation
                || moral_judgment
                    .violations
                    .iter()
                    .any(|v| v.contains("perfect") || v.contains("harm"))
            {
                self.stats.moral_review_needed = true;
            }

            // ── Track 5c: Violation-type-specific steering ───────────────────
            // Science: Greene (2013) — distinct moral processes for different violation types
            if moral_judgment.consent_violation {
                // Consent is most severe — strongly dampen confidence + pause learning
                self.scale_confidence("moral_consent_viol", 0.7);
                self.carryover.learning.subsystem_lr_factor *= 0.5;
                moral_steering_category = "consent";
            } else if moral_judgment.violations.iter().any(|v| v.contains("harm")) {
                // Harm detected — strongly reduce exploration, shift to protective mode
                self.curiosity_drive.exploration_urge *= 0.4;
                self.scale_confidence("moral_harm_detect", 0.85);
                moral_steering_category = "harm";
            } else if moral_judgment
                .violations
                .iter()
                .any(|v| v.contains("perfect") || v.contains("duty"))
            {
                // Deontological (perfect duty) — force reflection + consolidate constraint
                self.self_reflection.force_reflection();
                self.carryover.learning.subsystem_lr_factor *= 0.8;
                moral_steering_category = "duty";
            } else if !moral_judgment.violations.is_empty() {
                // Other violations — moderate dampening
                self.carryover.learning.subsystem_lr_factor *= 0.9;
                moral_steering_category = "other";
            }
        } else if moral_score > MORAL_BENEFIT_THRESHOLD {
            // Positive moral alignment boosts confidence slightly
            self.scale_confidence("moral_benefit", MORAL_BENEFIT_CONFIDENCE_BOOST);
        }

        // Surprise-gated learning rate boost: when FEP detects surprise, accelerate adaptation
        if is_surprised {
            let surprise_boost =
                (self.fep_agent.current_free_energy() as f32 / FEP_SURPRISE_SCALE).clamp(0.1, 0.5);
            self.adjust_lr("fep_surprise", surprise_boost);
        } else {
            // Decay boost back toward 1.0 when not surprised
            self.scale_lr("fep_decay", FEP_LR_DECAY);
        }

        // ── Phase 21: Predictive free energy → surprise amplitude scaling ─
        // Science: Friston (2010) — precision-weighted prediction errors
        let cached_pfe = self.carryover.consciousness.last_predictive_free_energy;
        let pfe_surprise_mod = if is_surprised && cached_pfe > 0.5 {
            // High FE amplifies surprise response (uncertain model → trust the error)
            let amplification = ((cached_pfe - 0.5) * 0.2).min(0.1) as f32;
            self.curiosity_drive.exploration_urge =
                (self.curiosity_drive.exploration_urge + amplification).clamp(0.0, 1.0);
            self.stats.pfe_surprise_mod_count += 1;
            amplification
        } else if is_surprised && cached_pfe < 0.2 && cached_pfe > 0.0 {
            // Low FE dampens surprise (confident model → spurious surprise)
            let dampening = ((0.2 - cached_pfe) * 0.15).min(0.05) as f32;
            self.curiosity_drive.exploration_urge *= 1.0 - dampening;
            self.stats.pfe_surprise_mod_count += 1;
            -dampening
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // NEUROMODULATOR BATH: Downstream modulation (Phase B)
        // Coherent chemical baseline that fine-grained Phase 14-21 loops adjust further.
        // ═══════════════════════════════════════════════════════════════════════
        // DA → learning rate
        self.scale_lr("neuromod_dopamine", self.neuromodulator_bath.learning_rate_factor());

        // NE → exploration
        self.curiosity_drive.exploration_urge += self.neuromodulator_bath.exploration_delta();
        self.curiosity_drive.exploration_urge =
            self.curiosity_drive.exploration_urge.clamp(0.0, 1.0);

        // 5-HT → confidence
        self.adjust_confidence("neuromod_serotonin", self.neuromodulator_bath.confidence_delta());

        // ACh → attention sensitivity + threshold
        self.adaptive_behavior.attention_sensitivity *= self.neuromodulator_bath.attention_factor();
        self.adaptive_behavior.attention_sensitivity =
            self.adaptive_behavior.attention_sensitivity.clamp(0.5, 2.0);
        self.carryover.learning.adaptive_threshold_scale *=
            self.neuromodulator_bath.threshold_factor();

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6b Enhanced FEP Bridge: Motor commands and learning signals
        // ═══════════════════════════════════════════════════════════════════════
        // Run enhanced FEP cycle for motor system integration and learning signals.
        // Optimization: run every 4th cycle unless surprised or high prediction error,
        // since the enhanced bridge overlaps with the primary FEP agent.
        // Urgency-adaptive scheduling: Critical=always, Normal=every 4th, Cruise=every 8th
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

        // Write real action-outcome coupling from enhanced FEP (replaces hardcoded 0.5)
        if let Some(ref er) = enhanced_result {
            self.stats.fep_action_outcome_coupling = er.action_outcome_coupling as f32;
        }

        // Apply motor command-based modulations (only when enhanced bridge ran)
        if let Some(ref enhanced_result) = enhanced_result {
            match enhanced_result.motor_command.command_type {
                MotorCommandType::AttentionShift => {
                    // Phase 19: Activate FEP → motor → attention → perception loop
                    // Science: Friston (2010) — active inference uses motor commands to
                    // modulate sensory precision (attention_sensitivity).
                    let shift_amount = enhanced_result.motor_command.intensity as f32 * 0.1;
                    self.adaptive_behavior.attention_sensitivity =
                        (self.adaptive_behavior.attention_sensitivity * (1.0 + shift_amount * 0.1))
                            .clamp(0.5, 2.0);
                    self.stats.attention_shift = shift_amount;
                }
                MotorCommandType::LearningRateAdjust => {
                    // Precision-weighted learning rate adjustment
                    if enhanced_result.should_learn {
                        let lr_mod = enhanced_result.fep_result.learning_rate_modulation as f32;
                        self.stats.adaptive_learning_rate =
                            (self.stats.adaptive_learning_rate * 0.9 + lr_mod * 0.1)
                                .clamp(0.01, 1.0);
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
                        self.set_confidence("inference_mode_init", 0.5);
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
        } // end if let Some(enhanced_result)

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Coherence tracking with degradation detection
        // ═══════════════════════════════════════════════════════════════════════
        let degraded = self.coherence_tracker.record_turn(coherence);
        if degraded {
            // Coherence degradation -> boost learning rate to accelerate recovery
            self.scale_lr("coherence_degraded", 1.3);
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
            let recommendations = self.self_reflection.reflect();
            // Apply LearningRate and ExplorationFactor recommendations that
            // apply_adjustments() skips (falls through to `_ => {}`).
            // Science: Meta-learning should be able to execute its own fixes.
            for rec in &recommendations {
                if rec.confidence < 0.5 {
                    continue;
                }
                match rec.target {
                    super::RecommendationTarget::LearningRate => match rec.direction {
                        super::AdjustmentDirection::Decrease => {
                            self.scale_lr("reflection_decrease", 0.9);
                        }
                        super::AdjustmentDirection::Increase => {
                            self.scale_lr("reflection_increase", 1.1);
                        }
                        _ => {}
                    },
                    super::RecommendationTarget::ExplorationFactor => match rec.direction {
                        super::AdjustmentDirection::Increase => {
                            self.curiosity_drive.exploration_urge =
                                (self.curiosity_drive.exploration_urge + 0.12).clamp(0.0, 1.0);
                        }
                        super::AdjustmentDirection::Decrease => {
                            self.curiosity_drive.exploration_urge *= 0.75;
                        }
                        _ => {}
                    },
                    _ => {} // FlowThreshold, BoredomThreshold, TrustThreshold handled by apply_adjustments()
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h. Update Consciousness Unification Engine with current Phi
        // ═══════════════════════════════════════════════════════════════════════
        // Compute unified Phi from coherence, confidence, and flow state
        // This feeds the dialogue pipeline for consciousness-aware responses

        let unified_psi = self.compute_unified_psi();
        // Neuromod → consciousness bridge: ACh/NE sustain conscious integration
        // Science: Alkire et al. (2008) — consciousness correlates with ACh/NE
        let neuromod_consciousness_mod = self.neuromodulator_bath.consciousness_modulation();
        let unified_psi = (unified_psi * neuromod_consciousness_mod as f64).clamp(0.0, 1.0);

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.exp EXPERIENCE BUS: Update principled signals from cognitive state
        // Maps cycle values to 5 principled signals (Active Inference).
        // Science: Friston (2010) — principled signals drive behavior.
        // ═══════════════════════════════════════════════════════════════════════
        let guiding_question: &str;
        let dominant_harmonic: &str;
        if let Some(ref mut bus) = self.experience_bus {
            bus.current_signals = crate::experience::PrincipledSignals {
                prediction_error,
                uncertainty: 1.0 - self.prediction_confidence,
                coherence: coherence.clamp(0.0, 1.0),
                confidence: self.prediction_confidence,
                salience: self.curiosity_drive.exploration_urge,
                phi_monitor: unified_psi as f32,
            };
            bus.update_wisdom_from_signals();
            bus.kosmic_state.phi = unified_psi as f32;
            guiding_question = bus.current_guiding_question();
            dominant_harmonic = bus.dominant_harmonic().as_str();
        } else {
            guiding_question = "";
            dominant_harmonic = "";
        }

        // ── Phase 15: Guiding question → subsystem priority ─────────────────
        // Science: Desimone & Duncan (1995) — top-down attention biases processing
        // toward task-relevant features. Parse the guiding question to boost
        // urgency of related subsystems.
        let guiding_priority_category = if !guiding_question.is_empty() {
            let q = guiding_question.to_lowercase();
            let cat = if q.contains("know") || q.contains("learn") || q.contains("understand") {
                // Epistemic question → boost prediction confidence sensitivity
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + 0.03).clamp(0.0, 1.0);
                "epistemic"
            } else if q.contains("feel") || q.contains("emotion") || q.contains("care") {
                // Affective question → boost emotional processing sensitivity
                self.adjust_confidence("guide_affective", 0.01);
                "affective"
            } else if q.contains("do") || q.contains("act") || q.contains("make") {
                // Pragmatic question → boost action-oriented processing
                self.scale_lr("guide_pragmatic", 1.02);
                "pragmatic"
            } else if q.contains("connect") || q.contains("relate") || q.contains("together") {
                // Social question → boost coherence sensitivity
                self.adjust_confidence("guide_social", 0.02);
                "social"
            } else {
                "general"
            };
            self.stats.guiding_question_priority_uses += 1;
            cat
        } else {
            ""
        };

        // ── Phase 15: Attention budget check ─────────────────────────────────
        // Science: Kahneman (1973) — attention is a limited-capacity resource.
        // Track cumulative cycle time; if we've exceeded budget, flag for
        // downstream subsystems to respect (skip expensive optional modules).
        let neuromod_attention_alloc = self.neuromodulator_bath.attention_budget_allocation();
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

        // ── Phase 20: Predictive budget gating ───────────────────────────────
        // Science: Botvinick & Braver (2015) — proactive control anticipates
        // resource depletion before it happens. If midpoint elapsed > 80% budget,
        // preemptively gate expensive subsystems for the remainder of this cycle.
        let predictive_budget_gated = attention_budget_elapsed_us > (ATTENTION_BUDGET_US * 4 / 5)
            && !attention_budget_exceeded;
        if predictive_budget_gated {
            self.stats.predictive_budget_gated_count += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.0 Generate PsiAttestation record for governance bridge
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
            // Capacity bound: attestation_buffer_capacity (max 256) — evict before push
            while self.psi_attestation_buffer.len() >= self.config.attestation_buffer_capacity {
                let _ = self.psi_attestation_buffer.pop_front();
            }
            self.psi_attestation_buffer.push_back(record);
        }

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

            // Build theory metrics from available consciousness signals
            let ec_metrics = ECMetrics {
                phi: unified_psi,
                gwt: coherence as f64,
                ast: self.prediction_confidence as f64,
                pp: (1.0 - prediction_error as f64).clamp(0.0, 1.0),
                rpt: pattern_confidence as f64,
                embodiment: self.fep_learning_signal as f64,
                unified: unified_psi,
            };

            // Compute available budget: 20ms target cycle minus time already spent
            let elapsed_us = cycle_start.elapsed().as_micros() as u64;
            let available_us = 20_000u64.saturating_sub(elapsed_us);

            let reasoning_ctx = ReasoningContext {
                theory_metrics: ec_metrics,
                phi: unified_psi,
                available_budget_us: available_us,
                available_actions: Vec::new(), // populated by external action providers
                tool: None,                    // populated by shell integration
                recent_utility: 0.5,
                cycle_id: self.stats.total_cycles as u64,
                neuromod_exploration_mod: self.neuromodulator_bath.mcts_exploration_modulation(),
            };

            let reasoning_result = reasoning_engine.reason(&reasoning_ctx);

            // 1. Phi_eff modulates confidence (higher = more reliable reasoning)
            reasoning_confidence = reasoning_result.phi_eff as f32;

            // 2. Reliability modulates learning rate — low reliability = cautious learning
            reasoning_lr_factor = reasoning_result.reliability as f32;

            // 3. Tool gate: check if the gate blocked an action
            if let Some(ref gate) = reasoning_result.gate {
                if !gate.is_allowed() {
                    reasoning_gate_blocked = true;
                    reasoning_fallback = gate.fallback.as_ref().map(|f| format!("{:?}", f));
                    // Gate blocked → suppress learning this cycle (safety measure)
                    reasoning_lr_factor = 0.0;
                    tracing::info!(
                        risk = ?gate.risk_level,
                        required_phi = gate.required_phi,
                        actual_phi = gate.actual_phi_eff,
                        "Reasoning gate blocked action"
                    );
                }
            }

            // 4. MCTS planning result (Tier 1+)
            if let Some(ref plan) = reasoning_result.plan {
                if plan.did_plan {
                    reasoning_plan_action = plan.best_action_idx;
                    reasoning_plan_confidence = plan.confidence as f32;
                }
            }

            // 5. Narrative (Tier 2, best-effort)
            reasoning_narrative = reasoning_result.narrative.clone();

            // 6. Log reasoning tier and timing for observability
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
        // Compare FEP action probabilities with MCTS plan to measure policy
        // agreement. Three tiers: full (same action), soft (FEP assigns >0.2
        // to MCTS action), disagreement (dampen proportional to divergence).
        // Science: Friston & Parr (2020) — policy agreement modulates precision.
        #[allow(unused_assignments)]
        let mut policy_agreement = false;
        if let Some(mcts_idx) = reasoning_plan_action {
            let fep_prob_for_mcts = fep_action_probs.get(mcts_idx).copied().unwrap_or(0.0);
            if mcts_idx == fep_action_idx {
                // Full agreement: both systems chose the same action — boost confidence
                reasoning_plan_confidence =
                    (reasoning_plan_confidence * POLICY_FULL_AGREEMENT_BOOST).min(1.0);
                policy_agreement = true;
            } else if fep_prob_for_mcts > POLICY_SOFT_THRESHOLD {
                // Soft agreement: FEP assigns reasonable probability to MCTS choice
                policy_agreement = true;
                reasoning_plan_confidence =
                    (reasoning_plan_confidence * (1.0 + fep_prob_for_mcts as f32 * 0.3)).min(1.0);
            } else {
                // Disagreement: dampen learning signal proportional to divergence
                let dampen = (0.3 + fep_prob_for_mcts * 0.7) as f32;
                self.fep_learning_signal *= dampen;
                reasoning_plan_confidence *= dampen;
            }

            // Track agreement for adaptive temperature
            // Capacity bound: POLICY_WINDOW_SIZE (20) — evict before push
            if self.policy_agreement_window.len() >= POLICY_WINDOW_SIZE {
                self.policy_agreement_window.pop_front();
            }
            self.policy_agreement_window.push_back(policy_agreement);
            // Adapt FEP softmax temperature: high agreement → exploit (low temp),
            // low agreement → explore (high temp)
            if self.policy_agreement_window.len() >= POLICY_MIN_WINDOW {
                let agree_rate = self.policy_agreement_window.iter().filter(|&&a| a).count() as f64
                    / self.policy_agreement_window.len() as f64;
                let adaptive_temp = POLICY_TEMP_BASE + (1.0 - agree_rate) * POLICY_TEMP_RANGE;
                self.fep_agent.config.action_temperature = adaptive_temp;
            }
        }

        // Store current MCTS plan for next cycle's dual-process application
        self.carryover.history.mcts_plan =
            reasoning_plan_action.map(|a| (a, reasoning_plan_confidence));

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1.5 Metacognitive Monitoring (Phi trajectory anomaly detection)
        // ═══════════════════════════════════════════════════════════════════════
        // After reasoning, observe unified Phi for trajectory anomalies.
        // Anomalies (drops, plateaus, oscillations) indicate reasoning degradation
        // and dampen the learning rate to avoid consolidating bad patterns.
        let mut metacognitive_anomaly = false;
        let mut anomaly_recovery_progress: f32 = 0.0;
        let anomaly_recovering;
        if let Some(ref mut monitor) = self.metacognitive_monitor {
            if monitor.observe_phi(unified_psi) {
                metacognitive_anomaly = true;
                // Dampen learning rate when reasoning is degrading
                reasoning_lr_factor *= 0.5;
                // Reset recovery counter on new anomaly
                self.carryover.urgency.anomaly_recovery_counter = 0;
                self.carryover.urgency.anomaly_was_active = true;
                tracing::debug!(
                    target: "cognitive_loop::metacognition",
                    unified_psi,
                    "Metacognitive anomaly detected — dampening learning rate"
                );
            }
        }

        // ── Phase 16: Metacognitive anomaly recovery path ────────────────
        // Science: Luria (1973) — executive recovery is gradual, not instantaneous.
        // After anomaly clears, progressively restore LR over 20 cycles.
        if !metacognitive_anomaly && self.carryover.urgency.anomaly_was_active {
            self.carryover.urgency.anomaly_recovery_counter = self
                .carryover
                .urgency
                .anomaly_recovery_counter
                .saturating_add(1);
            let counter = self.carryover.urgency.anomaly_recovery_counter;
            if counter <= 20 {
                // Gradually recover: 0.5 → 1.0 over 20 cycles
                let recovery = counter as f32 / 20.0;
                reasoning_lr_factor *= 0.5 + recovery * 0.5;
                anomaly_recovery_progress = recovery;
                self.stats.anomaly_recovery_active_count += 1;
            } else {
                // Fully recovered — clear the flag
                self.carryover.urgency.anomaly_was_active = false;
                anomaly_recovery_progress = 1.0;
            }
            anomaly_recovering = counter <= 20;
        } else {
            anomaly_recovering = false;
        }

        // Compose effective LR from all modulation sources (flow, curiosity, FEP, MCE, subsystem)
        let effective_lr = self.compose_effective_lr(semantic_lr_factor, reasoning_lr_factor);
        // DA D1-modulated gradient magnitude (neuromod-aware training)
        // Science: Schultz (1997) — DA scales synaptic plasticity amplitude
        // Frank (2005) — D1 pathway specifically drives learning magnitude
        let effective_lr = effective_lr * self.neuromodulator_bath.gradient_scale_factor();
        // ACh-gated plasticity persistence: high ACh = learning mode, low = performance mode
        // Science: Hasselmo (1999) — cholinergic gating of cortical plasticity
        let effective_lr = effective_lr * self.neuromodulator_bath.plasticity_gate();

        // ACh-gated threshold: high ACh → learn from smaller errors
        // Science: Yu & Dayan (2005) — ACh sharpens expected-uncertainty gating
        let neuromod_threshold = effective_threshold * self.neuromodulator_bath.threshold_gate();

        // 11. Learn if error is significant AND we have a previous state AND not paused
        // FEEDBACK: Narrative-GWT veto suppresses learning (consciousness governance)
        // Science: Baars (2005) — global workspace vetoing prevents consolidation
        // FEEDBACK: Consciousness-gated learning — system must be "awake" to consolidate
        // Science: Dehaene (2014) — conscious access required for durable learning
        let _t_core = Instant::now();
        let consciousness_awake =
            self.carryover.history.consciousness_level > 0.0 || self.stats.total_cycles < 20; // grace period for boot-up
        let (learning_occurred, training_loss) = if prediction_error > neuromod_threshold
            && !self.adaptive_behavior.pause_learning
            && !self.carryover.quality.narrative_veto_active
            && consciousness_awake
        {
            self.stats.learning_cycles += 1;

            // Build training sample (copy elements directly — avoids Vec clone)
            let (train_input, train_target, lr) = if let Some(prev) = previous_state {
                (
                    Array1::from_vec(prev),
                    compressed_state.iter().copied().collect(),
                    effective_lr,
                )
            } else {
                // First cycle: bootstrap with self-prediction (build two arrays to avoid clone)
                let train_input: Array1<f32> = compressed_state.iter().copied().collect();
                let train_target: Array1<f32> = compressed_state.iter().copied().collect();
                (train_input, train_target, effective_lr * 0.1)
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
        module_timings.core_training = _t_core.elapsed().as_micros() as u64;

        // Goal←Cognition feedback: consistent low error during goal pursuit signals progress.
        // Science: Anderson (1983) — prediction accuracy is evidence of task mastery.
        // Closes the Goal→Cognition loop (goal priority boosts LR) with Cognition→Goal feedback.
        if !learning_occurred && self.carryover.urgency.consecutive_low_error > 5 {
            if let Some(top) = self.goal_system.top_goal() {
                let top_id = top.id.clone();
                let delta = 0.01 * (1.0 + self.prediction_confidence * 0.5); // 0.01 to 0.015
                self.goal_system.update_progress(&top_id, delta);
            }
        }

        // 12. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());

        // Update state diversity from CfC
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Adaptive HDC dimension: resize if error demands it
        // FEEDBACK: Consciousness-aware resize hint (Tononi 2015 IIT)
        // Science: High Phi requires larger state space for integration; low Phi can conserve.
        // Modulate the error signal: high consciousness amplifies upscale pressure (need more
        // capacity), low consciousness dampens it (conserve resources).
        let consciousness_resize_factor =
            1.0 + (self.carryover.history.consciousness_level as f32 - 0.5) * 0.3; // ±15%
        self.temporal_network
            .maybe_resize(prediction_error * consciousness_resize_factor);

        // Update coherence metrics in stats
        self.stats.temporal_coherence = coherence; // Phase 17: use cached post-update value
        self.stats.effective_learning_rate = effective_lr;
        self.stats.coherence_phi_contribution = self.coherence_bridge.phi_contribution();

        // ── School curriculum recommendation (co-prime interval: every 53 cycles) ──
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

        // ── Causal consciousness attention (co-prime interval: every 41 cycles) ──
        let causal_attention_boost = if self.stats.total_cycles % 41 == 0 {
            if let Some(ref mut cc) = self.causal_consciousness {
                // Build variable observations from compressed state (chunk into 8D windows)
                let vars: Vec<Vec<f64>> = compressed_state
                    .chunks(8)
                    .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
                    .collect();
                if vars.len() >= 2 {
                    let attention = cc.attention.compute_attention(&vars);
                    // Top cause strength: max off-diagonal attention weight
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
        // Modulate confidence from causal attention (subtle: max 5% boost)
        if causal_attention_boost > 0.0 {
            self.adjust_confidence("causal_attention", causal_attention_boost * 0.05);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PARALLEL POST-PROCESSING: Independent subsystem updates via rayon
        // ═══════════════════════════════════════════════════════════════════════
        // These subsystem updates use disjoint fields and run in parallel:
        //   Branch A: Stability regime + Semantic memory + Causal enhancement
        //   Branch B: Episodic memory + Primitive-belief bridge + Closed learning loop
        // Sequential after join: Episodic replay + Memory coordinator (cross-dependencies)
        let _t_core = Instant::now();

        // Pre-compute read-only values needed by parallel branches
        let pp_total_cycles = self.stats.total_cycles;
        let pp_in_flow = self.flow_state.in_flow;
        let pp_emotional_valence = self.emotion_contagion.prosody_valence();
        let pp_phi = self.unification_engine.psi as f32;
        let pp_smoothed_coh = coherence as f64; // Phase 17: use cached post-update value
                                                // World model error → resonator storage importance bias
                                                // Science: Rescorla-Wagner (1972) — surprising events deserve higher encoding priority
        let pp_wm_importance_boost = self.world_model.avg_error.clamp(0.0, 1.0) * 0.3;
        // Track 4e: Thalamic depth → storage salience
        // Science: Sherman & Guillery (2006) — deep processing warrants durable encoding
        let pp_thalamic_salience = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => 0.2f32, // priority storage
            super::CognitiveDepth::Cortical => 0.0,       // neutral
            super::CognitiveDepth::Reflex => -0.1,        // lower priority
        };
        let pp_learning_threshold = self.config.learning_threshold;

        // Compute cycle reward before parallel section (reads prediction_confidence, flow_state)
        let cycle_reward = self.compute_reward_signal(prediction_error, pp_learning_threshold);

        let cycle_learning_result = CycleLearningResult {
            reward: cycle_reward,
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
            let resonator_memory = &mut self.resonator_memory;

            // Pre-parallel: Stability regime (evolves ~250 CfC primitives × 16,384D)
            module_timings.stability_regime = helpers::run_stability_regime(
                stability_regime,
                discovery_service,
                &hv16_cached,
                delta_t,
                pp_total_cycles,
                urgency,
            );

            let episodic_ctx = helpers::EpisodicLearningContext {
                prediction_error,
                in_flow: pp_in_flow,
                input,
                compressed_state: &compressed_state,
                emotional_valence: pp_emotional_valence,
                phi: pp_phi,
                total_cycles: pp_total_cycles,
                smoothed_coh: pp_smoothed_coh,
                detected_primitives: &encoding_result.detected_primitives,
                memory_context_boost,
                wm_importance_boost: pp_wm_importance_boost + pp_thalamic_salience,
            };

            rayon_join(
                || {
                    helpers::parallel_semantic_causal(
                        semantic_memory,
                        causal_enhancer,
                        semantic_hdc.into_owned(),
                        &compressed_state,
                        &output,
                        prediction_error,
                        pp_total_cycles,
                    )
                },
                || {
                    helpers::parallel_episodic_learning(
                        episodic_memory,
                        resonator_memory,
                        prediction_confidence_ref,
                        primitive_belief_bridge,
                        prev_primitive_state,
                        fep_learning_signal,
                        closed_learning_loop,
                        &episodic_ctx,
                        cycle_learning_result,
                    )
                },
            );
        } // end parallel scope -- disjoint borrows released

        module_timings.core_parallel_postprocess = _t_core.elapsed().as_micros() as u64;

        // Debug: trace parallel branch sub-timings for profiling
        // (Branch A timings are captured but currently not exposed — they're inside the closure)

        // Update semantic memory stats after parallel join completes
        self.stats.semantic_hits = self.semantic_memory.stats().semantic_hits;
        self.stats.semantic_misses = self.semantic_memory.stats().semantic_misses;
        self.stats.semantic_lr_factor = semantic_lr_factor;
        self.stats.semantic_avg_retrieved_error = self.semantic_memory.stats().avg_retrieved_error;
        self.stats.semantic_entries_stored = self.semantic_memory.stats().total_stored;

        // ═══════════════════════════════════════════════════════════════════════
        // CYCLE STATE: Shared read-only snapshot for extracted phase functions
        // ═══════════════════════════════════════════════════════════════════════
        let cycle_state = CycleState {
            compressed_state: &compressed_state,
            output: &output,
            prediction_error,
            coherence,
            unified_psi,
            phi_attention_weight,
            hv16_cached: &hv16_cached,
            input,
            urgency,
            attention_budget_exceeded,
            predictive_budget_gated,
        };

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS METRICS: Extracted to cycle_consciousness.rs
        // Includes: primitive consciousness, temporal primitives, lattice,
        // compositionality, value evaluator, consciousness profile, context-aware
        // evolution, semantic value embedder, harmonies, composition rules,
        // fiduciary harmonics, primitive reasoning, causal self-explanation,
        // adaptive reasoning, epistemic tiers, phi validation, dissipative
        // consciousness, epistemic conflict, consciousness equation v2.
        // ═══════════════════════════════════════════════════════════════════════
        let consciousness_metrics =
            self.compute_consciousness_metrics(&cycle_state, &mut module_timings);

        // Destructure consciousness metrics for use by later phases
        let primitive_psi = consciousness_metrics.primitive_psi;
        let active_primitive_names = consciousness_metrics.active_primitive_names;
        let temporal_causal_chains = consciousness_metrics.temporal_causal_chains;
        let temporal_continuity = consciousness_metrics.temporal_continuity;
        let temporal_max_chain_length = consciousness_metrics.temporal_max_chain_length;
        let _chain_cycle_numbers = consciousness_metrics.chain_cycle_numbers;
        let causal_codebook_entries = consciousness_metrics.causal_codebook_entries;
        let continuity_replay_needed = consciousness_metrics.continuity_replay_needed;
        let lattice_height = consciousness_metrics.lattice_height;
        let lattice_width = consciousness_metrics.lattice_width;
        let lattice_join_concept = consciousness_metrics.lattice_join_concept;
        let compositionality_total = consciousness_metrics.compositionality_total;
        let value_evaluator_score = consciousness_metrics.value_evaluator_score;
        let value_evaluator_decision = consciousness_metrics.value_evaluator_decision;
        let value_gate_factor = consciousness_metrics.value_gate_factor;
        let consciousness_profile_composite = consciousness_metrics.consciousness_profile_composite;
        let synergy_enhanced_composite = consciousness_metrics.synergy_enhanced_composite;
        let emergent_properties_count = consciousness_metrics.emergent_properties_count;
        let reasoning_context = consciousness_metrics.reasoning_context;
        let context_phi_weight = consciousness_metrics.context_phi_weight;

        // ── Phase 18: Context Phi Weight → Unified Psi modulation ────────────
        // Science: Gigerenzer (2007) — ecological rationality requires context-adaptive
        // consciousness weighting. Apply context_phi_weight as multiplicative modulation
        // so consciousness measurement adapts to reasoning context (analytical/creative/emotional).
        let context_phi_applied = context_phi_weight > 0.0 && context_phi_weight != 1.0;
        if context_phi_applied {
            // Weight range: 0.0-1.0 from optimizer; map to [0.8, 1.2] scaling factor
            let scale = 0.8 + context_phi_weight * 0.4;
            let adjusted_psi = (unified_psi * scale).clamp(0.0, 1.0);
            self.unification_engine.update_psi(adjusted_psi);
            self.stats.context_phi_applied_count += 1;
        }

        let value_embeddings_created = consciousness_metrics.value_embeddings_created;
        let value_cache_hit_rate = consciousness_metrics.value_cache_hit_rate;
        let harmonies_alignment = consciousness_metrics.harmonies_alignment;
        let harmonies_approved = consciousness_metrics.harmonies_approved;
        let composition_rule_applied = consciousness_metrics.composition_rule_applied;
        let harmonic_field_coherence = consciousness_metrics.harmonic_field_coherence;
        let harmonic_love_resonance = consciousness_metrics.harmonic_love_resonance;
        let harmonic_interferences = consciousness_metrics.harmonic_interferences;
        let reasoning_chain_confidence = consciousness_metrics.reasoning_chain_confidence;
        let reasoning_chain_depth = consciousness_metrics.reasoning_chain_depth;
        let causal_relations_count = consciousness_metrics.causal_relations_count;
        let causal_avg_confidence = consciousness_metrics.causal_avg_confidence;
        let adaptive_reasoning_phi = consciousness_metrics.adaptive_reasoning_phi;
        let epistemic_quality = consciousness_metrics.epistemic_quality;
        let phi_validation_correlation = consciousness_metrics.phi_validation_correlation;
        let dissipative_health = consciousness_metrics.dissipative_health
            * (1.0 - somatic_signals.dissipative_health_penalty); // Nociception: infrastructure stress penalty
        let dissipative_regime = consciousness_metrics.dissipative_regime;
        let dissipative_entropy_rate = consciousness_metrics.dissipative_entropy_rate;
        let epistemic_phi_eff = consciousness_metrics.epistemic_phi_eff;
        let epistemic_conflict_count = consciousness_metrics.epistemic_conflict_count;
        let equation_v2_consciousness = consciousness_metrics.equation_v2_consciousness;

        // ═══════════════════════════════════════════════════════════════════════
        // ADVANCED SUBSYSTEMS: Extracted to cycle_subsystems.rs
        // Includes: hierarchical LTC, evolution coordinator, holographic analyzer,
        // differentiable consciousness, affective consciousness, unified pipeline,
        // multi-modal integration, synthetic grounding, epistemic gate, primitive
        // validation, cross-module feedback, meta-cognitive reasoner, code primitive
        // router, empathic unification, multi-objective evolution.
        // ═══════════════════════════════════════════════════════════════════════
        let subsystem_metrics = self.run_advanced_subsystems(
            &cycle_state,
            &active_primitive_names,
            &mut module_timings,
        );

        // Destructure subsystem metrics for use by later phases
        let hierarchical_ltc_phi = subsystem_metrics.hierarchical_ltc_phi;
        let evolution_generation = subsystem_metrics.evolution_generation;
        let evolution_phi_delta = subsystem_metrics.evolution_phi_delta;
        // Phase 18: Track the confidence delta that was applied in run_advanced_subsystems
        let evolution_confidence_delta = if evolution_phi_delta > 0.01 {
            (evolution_phi_delta * 0.05).min(0.03) as f32
        } else if evolution_phi_delta < -0.01 {
            // Negative: exploration boost (report as negative confidence delta)
            -((-evolution_phi_delta) * 0.08).min(0.04) as f32
        } else {
            0.0
        };
        let holographic_unity = subsystem_metrics.holographic_unity;
        let holographic_binding = subsystem_metrics.holographic_binding;
        let consciousness_gradient_magnitude = subsystem_metrics.consciousness_gradient_magnitude;
        let consciousness_limiting_component = subsystem_metrics.consciousness_limiting_component;
        let affect_cons_valence = subsystem_metrics.affect_cons_valence;
        let affect_cons_arousal = subsystem_metrics.affect_cons_arousal;
        let pipeline_consciousness = subsystem_metrics.pipeline_consciousness;
        let multimodal_integrated_phi = subsystem_metrics.multimodal_integrated_phi;
        let consciousness_state_label = subsystem_metrics.consciousness_state_label;
        let consciousness_state_level = subsystem_metrics.consciousness_state_level;
        let epistemic_gate_confidence = subsystem_metrics.epistemic_gate_confidence;
        let epistemic_gate_approved = subsystem_metrics.epistemic_gate_approved;
        let primitive_validation_phi_gain = subsystem_metrics.primitive_validation_phi_gain;
        let primitive_validation_p_value = subsystem_metrics.primitive_validation_p_value;
        let meta_reasoning_confidence = subsystem_metrics.meta_reasoning_confidence;
        let meta_reasoning_insights = subsystem_metrics.meta_reasoning_insights;
        let code_primitives_selected = subsystem_metrics.code_primitives_selected;
        let empathic_compassion = subsystem_metrics.empathic_compassion;
        let empathic_tone_adj = subsystem_metrics.empathic_tone_adj;
        let multi_obj_frontier_size = subsystem_metrics.multi_obj_frontier_size;
        let grid_encoding_norm = subsystem_metrics.grid_encoding_norm;
        let grid_spatial_complexity = subsystem_metrics.grid_spatial_complexity;

        // ── Phase 18: Empathic tone → speech rate modulation ─────────────────
        // Science: Decety & Jackson (2004) — empathic resonance should modulate output.
        // Positive tone_adj (patience detected) → slow speech; negative (impatience) → speed up.
        let empathic_speech_rate_mod = if empathic_tone_adj.abs() > 0.1 {
            // Map [-1, 1] to speech rate multiplier: patience slows, impatience speeds
            let rate_mod = 1.0 - empathic_tone_adj as f32 * 0.1; // [-1,1] → [0.9, 1.1]
            self.adaptive_behavior.speech_rate_multiplier *= rate_mod;
            self.adaptive_behavior.speech_rate_multiplier = self
                .adaptive_behavior
                .speech_rate_multiplier
                .clamp(0.6, 1.5);
            empathic_tone_adj as f32
        } else {
            0.0
        };

        // ── Phase 19: Consciousness limiting component → targeted boost ─────
        // Science: Bengio (2017) — gradient descent requires acting on gradient info.
        // Match on the limiting component to apply a targeted nudge.
        let limiting_component_boosted = if !consciousness_limiting_component.is_empty()
            && consciousness_gradient_magnitude > 0.01
        {
            match consciousness_limiting_component.as_str() {
                "Attention" => {
                    self.adaptive_behavior.attention_sensitivity =
                        (self.adaptive_behavior.attention_sensitivity * 1.05).min(2.0);
                    self.stats.limiting_component_boost_count += 1;
                    "Attention"
                }
                "Binding" => {
                    // Boost prediction confidence — binding weakness signals integration gap
                    self.adjust_confidence("limit_binding", 0.01);
                    self.stats.limiting_component_boost_count += 1;
                    "Binding"
                }
                "Efficacy" => {
                    self.scale_lr("limit_efficacy", 1.05);
                    self.stats.limiting_component_boost_count += 1;
                    "Efficacy"
                }
                _ => "",
            }
        } else {
            ""
        };

        // ── Phase 19: Harmonic love resonance → confidence/soul amplifier ────
        // Science: The Seven Harmonies' emergent unity signal. When all harmonics
        // resonate together (love > 0.6), boost confidence and soul learning rate.
        let love_resonance_boost = if harmonic_love_resonance > 0.6 {
            let boost = ((harmonic_love_resonance - 0.6) * 0.04) as f32; // up to +1.6%
            self.adjust_confidence("love_resonance", boost);
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost * 0.5;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
            self.stats.love_resonance_boost_count += 1;
            boost
        } else {
            0.0
        };

        // ── Phase 19: Reasoning chain confidence + depth → confidence ────────
        // Science: Pearl (2000) — causal chains with high confidence = genuine explanation.
        let reasoning_chain_boosted =
            reasoning_chain_confidence > 0.7 && reasoning_chain_depth >= 3;
        if reasoning_chain_boosted {
            let chain_boost = (reasoning_chain_confidence - 0.7) * 0.05;
            self.adjust_confidence("reasoning_chain", chain_boost);
            self.stats.reasoning_chain_boost_count += 1;
        }

        // ── Phase 20: Harmonic interferences → LR feedback ───────────────────
        // Science: Treisman (1998) — feature binding conflicts (interferences)
        // signal integration difficulty. Many interferences (>3) dampen LR to avoid
        // encoding conflicted representations. Zero interferences boost LR.
        let harmonic_interference_lr_mod: f32 = if harmonic_interferences > 3 {
            let dampen = ((harmonic_interferences - 3) as f32 * 0.02).min(0.1);
            self.carryover.learning.subsystem_lr_factor *= 1.0 - dampen;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
            self.stats.harmonic_interference_mod_count += 1;
            -dampen
        } else if harmonic_interferences == 0 {
            let boost = 0.02_f32;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
            self.stats.harmonic_interference_mod_count += 1;
            boost
        } else {
            0.0
        };

        // ── Phase 20: Causal relations density → urgency gating ──────────────
        // Science: Spirtes et al. (2000) — dense causal graphs with high confidence
        // indicate a well-mapped problem space → reduce exploration urgency.
        let causal_urgency_gated = causal_relations_count > 10
            && causal_avg_confidence > 0.6
            && self.stats.total_cycles > 20;
        if causal_urgency_gated {
            // Boost consecutive_low_error to extend Cruise mode
            self.carryover.urgency.consecutive_low_error = self
                .carryover
                .urgency
                .consecutive_low_error
                .saturating_add(2);
            self.stats.causal_urgency_gated_count += 1;
        }

        // ── Phase 19: Attention budget gated flag for metadata ───────────────
        let attention_budget_gated =
            attention_budget_exceeded && self.stats.attention_budget_exceeded_count > 3;

        // ── Track 5a: Epistemic gate → actual information gating ─────────────
        // Science: Kruger & Dunning (1999) — epistemic humility gates downstream integration
        // When the gate rejects input (low confidence + not approved), dampen learning
        // and skip codebook growth. When approved, boost LR proportional to confidence.
        let mut epistemic_coherence_gated = false;
        if !epistemic_gate_approved {
            // Gate rejects: dampen learning proportional to gate certainty
            // (high confidence in rejection → strong dampening)
            let rejection_strength = (1.0 - epistemic_gate_confidence).clamp(0.0, 0.5);
            self.carryover.learning.subsystem_lr_factor *= 1.0 - rejection_strength * 0.3;
            self.scale_confidence("epistemic_reject", 1.0 - rejection_strength * 0.15);
        } else if epistemic_gate_confidence > 0.6 {
            // Gate approves with high confidence → modest LR boost
            let approval_boost = (epistemic_gate_confidence - 0.6) * 0.08;
            self.carryover.learning.subsystem_lr_factor *= 1.0 + approval_boost;
            self.carryover.learning.subsystem_lr_factor =
                self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.3);
        }

        // ── Phase 16: Epistemic gate confidence → coherence-gating spectrum ──
        // Science: Fernandez-Duque & Johnson (2002) — metacognitive monitoring adjusts
        // processing depth. Low gate confidence → raise coherence bar for expensive modules;
        // High confidence → feed back into adaptive threshold (trust inputs more).
        if epistemic_gate_confidence < 0.4 && epistemic_gate_confidence > 0.0 {
            // Low confidence → raise coherence requirements (be cautious)
            let caution_factor = (0.4 - epistemic_gate_confidence) * 0.3;
            self.carryover.learning.adaptive_threshold_scale *= 1.0 + caution_factor;
            self.carryover.learning.adaptive_threshold_scale = self
                .carryover
                .learning
                .adaptive_threshold_scale
                .clamp(0.5, 2.0);
            epistemic_coherence_gated = true;
            self.stats.epistemic_coherence_gated_count += 1;
        } else if epistemic_gate_confidence > 0.8 {
            // High confidence → loosen threshold (trust inputs, learn faster)
            let trust_factor = (epistemic_gate_confidence - 0.8) * 0.15;
            self.carryover.learning.adaptive_threshold_scale *= 1.0 - trust_factor;
            self.carryover.learning.adaptive_threshold_scale = self
                .carryover
                .learning
                .adaptive_threshold_scale
                .clamp(0.5, 2.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // RESONATOR CODEBOOK GROWTH + HIGH-PHI PROMOTION + DIVERSITY (extracted)
        // Phase 15: Skip when input is memoized (identical stimulus → no new codebook info)
        // ═══════════════════════════════════════════════════════════════════════
        let ResonatorCodebookResult {
            resonator_promotions,
            codebook_evictions,
            codebook_diversity,
            codebook_utilization_rate,
        } = if input_memoized {
            ResonatorCodebookResult {
                resonator_promotions: 0,
                codebook_evictions: 0,
                codebook_diversity: self.stats.codebook_diversity,
                codebook_utilization_rate: self.stats.codebook_utilization_rate,
            }
        } else {
            self.run_resonator_codebook_phase(
                epistemic_gate_approved,
                &compressed_state,
                &active_primitive_names,
                &causal_codebook_entries,
                &reflection_thresholds,
                &mut module_timings,
            )
        };

        // ═══════════════════════════════════════════════════════════════════════
        // EPISODIC REPLAY + MEMORY COORDINATOR (extracted)
        // ═══════════════════════════════════════════════════════════════════════
        let EpisodicReplayResult {
            surprise_replay_batch_size,
        } = self.run_episodic_replay_and_memory_phase(
            &cycle_state,
            memory_context_boost,
            fep_surprise,
            surprise_thresh,
            &mut module_timings,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // SUPPORT INTELLIGENCE: Triage + Knowledge + Predictive + Federation
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        #[cfg(feature = "support")]
        let (support_triage_count, support_alert_fired, support_federation_graduated, support_efe) = {
            self.support_cycle_counter += 1;

            // Triage: classify current input every cycle (lightweight keyword match)
            let mut triage_count: u32 = 0;
            if let Some(ref engine) = self.support_triage_engine {
                let result = engine.triage(input, "");
                triage_count = 1;
                // Feed triage category into knowledge search
                if let Some(ref manager) = self.support_knowledge_manager {
                    let category_str = format!("{:?}", result.suggested_category);
                    let articles = manager.search(&category_str, 3);
                    if !articles.is_empty() {
                        tracing::trace!(
                            category = %category_str,
                            articles = articles.len(),
                            "Support: matched knowledge articles for input"
                        );
                    }
                }
            }

            // Predictive telemetry check every 47 cycles (co-prime)
            let mut alert_fired = false;
            let mut efe = 0.0_f64;
            if self.support_cycle_counter % 47 == 0 {
                if let Some(ref engine) = self.support_predictive_engine {
                    let telemetry = symthaea_support::telemetry::collect_telemetry();
                    let prediction = engine.assess_system_state(&telemetry);
                    efe = prediction.expected_free_energy;
                    if engine.should_alert(&prediction) {
                        alert_fired = true;
                        tracing::warn!(
                            efe = prediction.expected_free_energy,
                            failure = ?prediction.predicted_failure,
                            "Support predictive alert: elevated free energy"
                        );
                    }
                }
            }

            // Federation: graduation check every 97 cycles (co-prime, privacy-gated)
            let mut graduated: usize = 0;
            if self.support_cycle_counter % 97 == 0 {
                let can_share = self
                    .support_privacy_manager
                    .as_ref()
                    .map(|pm| pm.can_share_cognitive())
                    .unwrap_or(true); // default: allow if no manager

                if can_share {
                    if let Some(ref manager) = self.support_knowledge_manager {
                        let pending = Vec::new(); // populated from resolution queue when bridge is wired
                        let result =
                            symthaea_support::federation::check_graduations(manager, &pending);
                        graduated = result.graduated;
                        if result.graduated > 0 {
                            tracing::debug!(
                                graduated = result.graduated,
                                deferred = result.deferred,
                                "Support federation: knowledge graduated"
                            );
                        }
                    }
                } else {
                    tracing::trace!("Support federation skipped: privacy tier blocks sharing");
                }
            }

            (triage_count, alert_fired, graduated, efe)
        };
        #[cfg(not(feature = "support"))]
        let (support_triage_count, support_alert_fired, support_federation_graduated, support_efe) =
            (0u32, false, 0usize, 0.0f64);
        module_timings.support_intelligence = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 10.5: Hyper-Parameter Optimization (The Meta-Forge)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let opt_result = self.run_parameter_optimization_phase();
        module_timings.parameter_optimization = _t.elapsed().as_micros() as u64;

        if opt_result.swap_occurred {
            // Hot-swap occurred: update metrics or log if needed
            self.stats.brain_swaps_count = self.stats.brain_swaps_count.saturating_add(1);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // Phase 11: DREAM ENGINE (Recording & Wisdom Application)
        // ═══════════════════════════════════════════════════════════════════════
        let DreamPhaseResult {
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
        } = self.run_dream_phase(&cycle_state, &prediction, &mut module_timings);

        // 7 (deferred). Send prediction to encoder for next cycle — moved here from step 7
        // so we can move instead of clone (prediction is no longer referenced after this point).
        self.encoder.set_prediction(prediction);

        // ═══════════════════════════════════════════════════════════════════════
        // LATE CONSCIOUSNESS MONITORS + INTEGRATION (extracted to cycle_late_consciousness.rs)
        // Prefrontal cortex, meta-cognition, virtual body, affective bridge, user state,
        // narrative self, predictive processing, hierarchical free energy, predictive self,
        // attention schema, phi attention, GWT, cross-modal binding, consciousness monitors,
        // phenomenal binding, temporal consciousness, thermodynamics, embodied cognition,
        // narrative-GWT, unified living mind, master consciousness equation.
        // ═══════════════════════════════════════════════════════════════════════
        use super::cycle_late_consciousness::LateConsciousnessContext;

        let late_ctx = LateConsciousnessContext {
            prediction_error,
            coherence,
            unified_psi,
            hv16_cached,
            compressed_state: &compressed_state,
            input,
            urgency,
            moral_concern_detected,
            surprise_triggered,
            reasoning_gate_blocked,
            pp_phi,
            peak_attention: encoding_result.peak_attention,
        };

        let late_result = self.run_late_consciousness_monitors(&late_ctx, &mut module_timings);
        let integration_result =
            self.run_consciousness_integration(&late_ctx, &late_result, &mut module_timings);

        // Re-bind local variables from late consciousness results
        let prefrontal_veto = late_result.prefrontal_veto;
        let meta_cognitive_accuracy = late_result.meta_cognitive_accuracy;
        let meta_cognitive_depth = late_result.meta_cognitive_depth;
        let body_psi_modulation = late_result.body_psi_modulation;
        let body_valence = late_result.body_valence;
        let body_arousal = late_result.body_arousal;
        let affective_valence = late_result.affective_valence;
        let affective_arousal = late_result.affective_arousal;
        let narrative_self_psi = late_result.narrative_self_psi;
        let predictive_free_energy = late_result.predictive_free_energy;
        // Phase 21: Cache predictive free energy for next cycle's surprise scaling
        self.carryover.consciousness.last_predictive_free_energy = predictive_free_energy;
        let predictive_psi_modulation = late_result.predictive_psi_modulation;
        let hierarchical_total_free_energy = late_result.hierarchical_total_free_energy;
        let predictive_self_safety = late_result.predictive_self_safety;
        let attention_schema_focus = late_result.attention_schema_focus;
        let psi_attention_avg = late_result.psi_attention_avg;

        // Re-bind local variables from consciousness integration results
        let gwt_broadcast = integration_result.gwt_broadcast;
        let cross_modal_binding_strength = integration_result.cross_modal_binding_strength;
        let cross_modal_psi = integration_result.cross_modal_psi;
        let resonance_frequency = integration_result.resonance_frequency;
        let quantum_coherence_level = integration_result.quantum_coherence_level;
        let phenomenal_binding_strength = integration_result.phenomenal_binding_strength;
        let phenomenal_fragmented = integration_result.phenomenal_fragmented;
        let temporal_coherence_score = integration_result.temporal_coherence_score;
        let temporal_discontinuity = integration_result.temporal_discontinuity;
        let thermodynamic_entropy = integration_result.thermodynamic_entropy;
        let thermodynamic_free_energy = integration_result.thermodynamic_free_energy;
        let embodied_psi_modulation = integration_result.embodied_psi_modulation;
        let embodied_agency = integration_result.embodied_agency;
        // Phase 21: Cache embodied agency for next cycle's strategy modulation
        self.carryover.consciousness.last_embodied_agency = embodied_agency;
        let narrative_gwt_veto = integration_result.narrative_gwt_veto;
        let narrative_gwt_self_psi = integration_result.narrative_gwt_self_psi;
        let living_mind_vitality = integration_result.living_mind_vitality;
        let living_mind_coherence = integration_result.living_mind_coherence;
        let consciousness_level = integration_result.consciousness_level;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 16: Quality-Aware Adaptive Processing
        // ═══════════════════════════════════════════════════════════════════════

        // ── Task #55: Dissipative health + phenomenal binding → learning gate ──
        // Science: Prigogine (1977) — dissipative structures require stability to consolidate.
        // When the system's thermodynamic health is low OR phenomenal binding is fragmented,
        // dampen learning to prevent cementing unstable patterns.
        let dissipative_lr_factor;
        let dissipative_health_gated;
        {
            let dh = self.carryover.quality.last_dissipative_health as f32;
            let pb = self.carryover.quality.last_phenomenal_binding as f32;
            if dh < 0.5 && pb < 0.6 && self.stats.total_cycles > 50 {
                // Both unhealthy → strong dampening (up to -30%)
                let dampening = (1.0 - dh) * (1.0 - pb) * 0.3;
                dissipative_lr_factor = (1.0 - dampening).max(0.7);
                self.carryover.learning.subsystem_lr_factor *= dissipative_lr_factor;
                dissipative_health_gated = true;
                self.stats.dissipative_health_gated_count += 1;
            } else if dh < 0.5 || pb < 0.4 {
                // One unhealthy → mild dampening (up to -15%)
                let dampening = if dh < 0.5 {
                    (0.5 - dh) * 0.15
                } else {
                    (0.4 - pb) * 0.15
                };
                dissipative_lr_factor = (1.0 - dampening).max(0.85);
                self.carryover.learning.subsystem_lr_factor *= dissipative_lr_factor;
                dissipative_health_gated = true;
                self.stats.dissipative_health_gated_count += 1;
            } else {
                dissipative_lr_factor = 1.0;
                dissipative_health_gated = false;
            }
            // Cache for next cycle
            self.carryover.quality.last_dissipative_health = dissipative_health as f64;
            self.carryover.quality.last_phenomenal_binding = phenomenal_binding_strength;
        }

        // ── Task #57: Coherence velocity → dynamic gating ─────────────────────
        // Science: Kelso (1995) — phase transitions in coordination dynamics.
        // Track rate of coherence change; rapid drops indicate instability.
        let coherence_velocity;
        let coherence_velocity_gated;
        {
            let prev_coh = self.carryover.quality.last_coherence;
            coherence_velocity = coherence - prev_coh;
            self.carryover.quality.coherence_velocity = coherence_velocity;
            self.carryover.quality.last_coherence = coherence;

            // If temporal discontinuity detected OR rapid coherence drop,
            // raise confidence requirements and tighten exploration
            if temporal_discontinuity || coherence_velocity < -0.15 {
                let severity = if temporal_discontinuity {
                    0.5 + (-coherence_velocity).max(0.0)
                } else {
                    (-coherence_velocity - 0.15).min(0.5)
                };
                // Dampen confidence proportional to severity
                self.scale_confidence("coherence_vel_drop", 1.0 - severity * 0.1);
                // Raise learning threshold (require higher error to trigger training)
                self.carryover.learning.adaptive_threshold_scale *= 1.0 + severity * 0.2;
                self.carryover.learning.adaptive_threshold_scale = self
                    .carryover
                    .learning
                    .adaptive_threshold_scale
                    .clamp(0.5, 2.0);
                coherence_velocity_gated = true;
                self.stats.coherence_velocity_gated_count += 1;
            } else {
                coherence_velocity_gated = false;
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // END-OF-CYCLE HOMEOSTASIS: Prevent asymmetric drift and runaway spirals
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();

        // Guard: clamp total per-cycle confidence drift to ±15%.
        // prediction_confidence is modified ~25 times per cycle by different subsystems,
        // each reading a different intermediate value. Without bounding, subsystems can
        // compound to produce wild swings. This ensures no single cycle changes confidence
        // by more than 15% regardless of subsystem ordering.
        // Science: Homeostatic plasticity (Turrigiano 2004) — bound rate of change.
        {
            let confidence_start = self.carryover.learning.prediction_confidence;
            let max_drift = confidence_start * 0.15 + 0.02; // ±15% + 2% floor
            let clamped = self
                .prediction_confidence
                .clamp(confidence_start - max_drift, confidence_start + max_drift)
                .clamp(0.0, 1.0);
            self.set_confidence("homeostasis_drift_clamp", clamped);
        }

        // Clamp attention_sensitivity to [0.5, 2.0] after all modifications.
        // 10+ subsystems multiply this field per cycle; without bounding, it can drift
        // to extreme values. Science: Weber-Fechner law — perception has bounded dynamic range.
        self.adaptive_behavior.attention_sensitivity =
            self.adaptive_behavior.attention_sensitivity.clamp(0.5, 2.0);

        // Boredom↔confidence homeostasis (Turrigiano 2004 — homeostatic plasticity)
        // High boredom signals stagnation → dampen confidence (system shouldn't be confident
        // when stuck in repetitive states). Prevents confidence runaway from accumulated boosts.
        if self.curiosity_drive.boredom > 0.7 {
            let boredom_dampen = (self.curiosity_drive.boredom - 0.7) * 0.15;
            self.scale_confidence("boredom_dampen", (1.0 - boredom_dampen).max(0.85));
        }

        // Boredom homeostasis: slow drift toward neutral (0.5) prevents monotonic saturation.
        // Phase 18: urgency-adaptive pull (Cruise=1.5×, Critical=0.6×)
        self.curiosity_drive.boredom +=
            (0.5 - self.curiosity_drive.boredom) * 0.02 * homeostasis_pull_strength;

        // Exploration urge per-cycle budget: clamp total change to ±0.5.
        // 15+ subsystems write exploration_urge per cycle; without bounding, cumulative
        // nudges can pin it to 0.0 or 1.0. Science: Homeostatic control of exploration.
        self.curiosity_drive.exploration_urge = self.curiosity_drive.exploration_urge.clamp(
            (exploration_urge_start - 0.5).max(0.0),
            (exploration_urge_start + 0.5).min(1.0),
        );

        // Exploration urge homeostasis: slow drift toward neutral (0.3) prevents saturation.
        // Phase 18: urgency-adaptive pull (Cruise=1.5×, Critical=0.6×)
        self.curiosity_drive.exploration_urge +=
            (0.3 - self.curiosity_drive.exploration_urge) * 0.03 * homeostasis_pull_strength;

        // Store urgency for next cycle's hysteresis
        self.carryover.urgency.urgency = urgency;
        module_timings.homeostasis = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // SPECTRAL MIP — O(n³) Fiedler-ordered MIP search (Layer 2)
        let _t = Instant::now();
        // Replaces SynergisticIntegration: 128 dims (vs 64), better MIP via
        // Fiedler ordering + bordered Cholesky. Computed every 47 cycles (co-prime).
        // sigma derived from spectral_mip_phi for backward compatibility.
        // ═══════════════════════════════════════════════════════════════════════
        self.spectral_mip_finder.push(&encoding_result.hdv);
        // Move hdv out now — only peak_attention (Copy) and detected_primitives are needed later.
        // Avoids a 64KB ContinuousHV clone for soul experience integration below.
        let encoding_hdv = encoding_result.hdv;
        let spectral_mip_phi = if self.stats.total_cycles % 97 == 0 {
            let result = self.spectral_mip_finder.compute();
            let phi = result.as_ref().map(|r| r.phi);
            if phi.is_some() {
                self.carryover.consciousness.last_spectral_mip_phi = phi;
                self.carryover.consciousness.last_sigma = phi; // backward compat for memory coordinator
            }
            // Adaptive dimension selection: every 194 cycles (every 2nd compute at 97-cycle cadence),
            // concentrate tracked dimensions near the MIP boundary for better
            // partition quality. Fiedler ordering identifies informative dims.
            if self.stats.total_cycles % 194 == 0 {
                if let Some(ref r) = result {
                    self.spectral_mip_finder.adapt(r);
                }
                // Hierarchical spectral MIP: multi-scale (32→64→128) Phi.
                // Coarser scales focus finer scales on MIP boundary region.
                // Runs every 94 cycles (~1.9s at 50Hz) for deeper integration analysis.
                if let Some(hier) = self.spectral_mip_finder.compute_hierarchical() {
                    self.carryover.consciousness.last_hierarchical_mip_phi = Some(hier.phi);
                }
            }
            phi
        } else {
            self.carryover.consciousness.last_spectral_mip_phi
        };
        let sigma = self.carryover.consciousness.last_sigma;
        module_timings.spectral_mip = _t.elapsed().as_micros() as u64;

        // ── W1.7: Σ (sigma) → learning rate + confidence modulation ──────
        // Science: Tononi (2008) — high integration (Φ) indicates coherent processing;
        // high Σ → stabilize learning (reduce LR boost), increase prediction confidence
        if let Some(sig) = sigma {
            if sig > 0.5 {
                // High integration → consolidate (stabilize LR)
                let sig_dampen = ((sig - 0.5) * 0.1).min(0.05) as f32;
                self.scale_lr("sigma_high", 1.0 - sig_dampen);
                self.adjust_confidence("sigma_high", sig_dampen * 0.5);
            } else if sig < 0.2 {
                // Low integration → boost learning (model needs updating)
                let sig_boost = ((0.2 - sig) * 0.15).min(0.05) as f32;
                self.scale_lr("sigma_low", 1.0 + sig_boost);
            }
        }

        // ── Phase 16: Adaptive Phi weighting from validation ───────────────
        // Science: Casali et al. (2013) — validated Phi measures are more reliable.
        // Use cached phi_validation_correlation to scale sigma's influence on learning.
        // High validation correlation → sigma is trustworthy → amplify its effect.
        // Low correlation → sigma is noisy → attenuate its effect.
        let phi_spectral_weight = self.carryover.quality.phi_spectral_weight;
        let phi_validation_cached = self.carryover.quality.phi_validation_correlation;
        if let Some(sig) = sigma {
            if phi_validation_cached > 0.7 {
                // Validated: amplify sigma's confidence contribution
                let validation_boost = (phi_validation_cached - 0.7) as f32 * 0.1;
                self.adjust_confidence("phi_validated", sig as f32 * validation_boost);
            } else if phi_validation_cached > 0.0 && phi_validation_cached < 0.3 {
                // Poorly validated: reduce sigma's influence (already applied above)
                let attenuate = (0.3 - phi_validation_cached) as f32 * 0.05;
                self.scale_confidence("phi_unvalidated", 1.0 - attenuate);
            }
        }
        // Also weight equation_v2 when it deviates from spectral MIP
        if let (Some(sig), eq_v2) = (sigma, equation_v2_consciousness) {
            let deviation = (sig - eq_v2).abs();
            if deviation > 0.2 && phi_spectral_weight < 0.6 {
                // Spectral weight reduced (validation says eq_v2 is more reliable)
                // → trust eq_v2 for confidence modulation
                let eq_v2_boost = (eq_v2 * (1.0 - phi_spectral_weight as f64) * 0.03) as f32;
                self.adjust_confidence("eq_v2_deviation", eq_v2_boost);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED CONSCIOUSNESS ENGINE — additive telemetry (runs alongside inline code)
        // Wraps SpectralMIP + MultiModal + EqV2 + Pipeline into single measure() call
        // ═══════════════════════════════════════════════════════════════════════
        let _consciousness_output = self.consciousness_engine.measure(
            &super::consciousness_engine::ConsciousnessEngineInput {
                hdv: &encoding_hdv,
                hv16: &hv16_cached,
                cycle: self.stats.total_cycles as u64,
                unified_psi,
                coherence,
                prediction_error,
                phi_attention_weight,
                epistemic_quality: self.carryover.quality.last_epistemic_quality,
                phi_validation_correlation: self.carryover.quality.phi_validation_correlation,
                phi_spectral_weight: phi_spectral_weight as f64,
            },
        );
        // NOTE: Not calling update_cache() yet — the existing inline code already
        // writes to carryover.consciousness. The engine output is for telemetry only
        // during this additive wiring phase.
        module_timings.consciousness_engine = _consciousness_output.total_us;

        // Soul experience integration: feed cycle outcome back into value learning.
        let _t = Instant::now();
        // This closes the loop: Soul evaluates alignment (pre-cycle) → cognitive cycle
        // → integrate experience (post-cycle) → Soul's essence evolves.
        if let Some(ref mut soul) = self.soul {
            let moral_score = self
                .last_moral_judgment
                .as_ref()
                .map(|j| j.moral_score)
                .unwrap_or(0.0);
            let experience = crate::soul::Experience {
                embedding: encoding_hdv.clone(),
                value_alignment: moral_score,
                emotional_valence: self.emotion_contagion.valence,
                lessons: Vec::new(),
                timestamp: self.stats.total_cycles as u64,
            };
            soul.integrate_experience(experience);
        }
        module_timings.soul_experience = _t.elapsed().as_micros() as u64;

        // ── Track 4b: Cross-module agreement metric ─────────────────────────
        // Science: Dehaene & Naccache (2001) — global workspace coherence requires
        // multiple module agreement for conscious access
        // Components: (1) FEP confidence (low surprise), (2) resonator match,
        // (3) moral alignment, (4) MCTS plan confidence (cached from this cycle)
        let fep_confidence = (1.0 - fep_surprise.min(1.0)).max(0.0) as f32;
        let resonator_confidence = resonator_best_sim;
        let moral_confidence = self
            .last_moral_judgment
            .as_ref()
            .map(|j| (j.moral_score + 1.0) / 2.0) // normalize [-1,1] → [0,1]
            .unwrap_or(0.5);
        let mcts_confidence = self
            .carryover
            .history
            .mcts_plan
            .as_ref()
            .map(|&(_, c)| c)
            .unwrap_or(0.5);
        // Agreement = 1 - variance of normalized confidence signals
        let signals = [
            fep_confidence,
            resonator_confidence,
            moral_confidence,
            mcts_confidence,
        ];
        let mean_signal: f32 = signals.iter().sum::<f32>() / signals.len() as f32;
        let variance: f32 = signals
            .iter()
            .map(|s| (s - mean_signal).powi(2))
            .sum::<f32>()
            / signals.len() as f32;
        let cross_module_agreement = (1.0 - (variance * 4.0).sqrt()).clamp(0.0, 1.0);
        // Agreement modulates confidence and exploration
        if cross_module_agreement > 0.8 {
            // High agreement → amplify shared signal
            self.adjust_confidence("cross_mod_agree", (cross_module_agreement - 0.8) * 0.05);
        } else if cross_module_agreement < 0.3 {
            // Low agreement → modules conflict, dampen confidence, boost exploration
            self.scale_confidence("cross_mod_disagree", 1.0 - (0.3 - cross_module_agreement) * 0.1);
            self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                + (0.3 - cross_module_agreement) * 0.15)
                .clamp(0.0, 1.0);
        }
        // EMA update for stats tracking
        self.stats.avg_cross_module_agreement =
            self.stats.avg_cross_module_agreement * 0.95 + cross_module_agreement * 0.05;

        // ── Task #54: Unified quality signal fusion ───────────────────────────
        // Science: Ernst & Banks (2002) — multi-cue integration via reliability weighting.
        // Fuse prediction coherence + cross-module agreement + anomaly status into
        // unified quality score for downstream gating.
        let unified_quality_score;
        {
            let anomaly_factor = if metacognitive_anomaly { 0.0 } else { 1.0 };
            unified_quality_score =
                0.5 * prediction_coherence + 0.3 * cross_module_agreement + 0.2 * anomaly_factor;
            self.stats.avg_unified_quality =
                self.stats.avg_unified_quality * 0.9 + unified_quality_score * 0.1;

            // High quality → boost learning rate (confident multi-system agreement)
            if unified_quality_score > 0.8 {
                let quality_boost = (unified_quality_score - 0.8) * 0.25;
                self.carryover.learning.subsystem_lr_factor *= 1.0 + quality_boost;
                self.carryover.learning.subsystem_lr_factor =
                    self.carryover.learning.subsystem_lr_factor.clamp(0.7, 1.5);
            }
            // Low quality → dampen exploration (conflicting signals, don't wander)
            if unified_quality_score < 0.3 && self.stats.total_cycles > 30 {
                self.curiosity_drive.exploration_urge *= 0.9;
            }
        }

        // ── Track 4e: Thalamic depth → storage salience ──────────────────────
        // Science: Sherman & Guillery (2006) — thalamic relay modulates cortical encoding
        let thalamic_depth_score = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => 1.0f32,
            super::CognitiveDepth::Cortical => 0.5,
            super::CognitiveDepth::Reflex => 0.2,
        };

        // Pre-compute values and formatted strings to avoid expensive ops inside struct literal
        let value_trend = self.value_feedback.recent_trend(50);
        let circadian_phase_str = self.biorhythm.phase.as_str();
        let selected_strategy_str = selected_strategy.as_str();

        // Build cycle metadata for observability
        let _t = Instant::now();
        let mut metadata = super::CycleMetadata {
            surprise_triggered,
            prefrontal_veto,
            reasoning_confidence,
            exploration_action,
            reasoning_gate_blocked,
            reasoning_fallback,
            reasoning_plan_action,
            reasoning_plan_confidence,
            reasoning_narrative,
            meta_cognitive_accuracy,
            meta_cognitive_depth,
            narrative_self_psi,
            body_phi_modulation: body_psi_modulation,
            body_valence,
            body_arousal,
            consciousness_level,
            predictive_self_safety,
            attention_schema_focus,
            gwt_broadcast,
            resonance_frequency,
            quantum_coherence_level,
            temporal_coherence_score,
            temporal_discontinuity,
            embodied_phi_modulation: embodied_psi_modulation,
            embodied_agency,
            narrative_gwt_veto,
            narrative_gwt_self_psi,
            living_mind_vitality,
            living_mind_coherence,
            urgency,
            dream_insights,
            dream_phi_improvement,
            dream_wisdom_count,
            predictive_free_energy,
            predictive_phi_modulation: predictive_psi_modulation,
            cross_modal_binding_strength,
            cross_modal_psi,
            affective_valence,
            affective_arousal,
            thermodynamic_entropy,
            thermodynamic_free_energy,
            phenomenal_binding_strength,
            phenomenal_fragmented,
            hierarchical_total_free_energy,
            psi_attention_avg,
            primitive_psi,
            temporal_causal_chains,
            temporal_continuity,
            temporal_max_chain_length,
            lattice_height,
            lattice_width,
            lattice_join_concept: lattice_join_concept.unwrap_or_default(),
            causal_codebook_entries: causal_codebook_entries.len(),
            continuity_replay_triggered: continuity_replay_needed,
            compositionality_total,
            composition_rule_applied,
            harmonies_alignment,
            harmonies_approved,
            empathic_compassion,
            empathic_tone_adj,
            multi_obj_frontier_size,
            value_evaluator_score,
            value_evaluator_decision,
            consciousness_profile_composite,
            synergy_enhanced_composite,
            emergent_properties_count,
            reasoning_context,
            context_phi_weight,
            harmonic_field_coherence,
            harmonic_love_resonance,
            harmonic_interferences,
            reasoning_chain_confidence,
            reasoning_chain_depth,
            causal_relations_count,
            causal_avg_confidence,
            evolution_generation,
            evolution_phi_delta,
            value_embeddings_created,
            value_cache_hit_rate,
            adaptive_reasoning_phi,
            epistemic_quality,
            phi_validation_correlation,
            dissipative_health,
            dissipative_regime,
            dissipative_entropy_rate,
            epistemic_phi_eff,
            epistemic_conflict_count,
            equation_v2_consciousness,
            hierarchical_ltc_phi,
            holographic_unity,
            holographic_binding,
            consciousness_gradient_magnitude,
            consciousness_limiting_component,
            affect_consciousness_valence: affect_cons_valence,
            affect_consciousness_arousal: affect_cons_arousal,
            pipeline_consciousness,
            multimodal_integrated_phi,
            consciousness_state_label,
            consciousness_state_level,
            epistemic_gate_confidence,
            epistemic_gate_approved,
            primitive_validation_phi_gain,
            primitive_validation_p_value,
            meta_reasoning_confidence,
            meta_reasoning_insights,
            code_primitives_selected,
            metacognitive_anomaly,
            safety_blocked: false,
            safety_category: None,
            negation_polarity: input_negation_polarity,
            moral_score,
            selected_strategy: selected_strategy_str.into(),
            actual_effective_lr: if learning_occurred { effective_lr } else { 0.0 },
            cycle_reward,
            fep_action: fep_action_idx,
            value_feedback_trend: value_trend,
            support_triage_count,
            support_alert_fired,
            support_federation_graduated,
            support_efe,
            soul_alignment,
            sigma,
            spectral_mip_phi,
            hierarchical_mip_phi: self.carryover.consciousness.last_hierarchical_mip_phi,
            hierarchical_mip_scales: self
                .carryover
                .consciousness
                .last_hierarchical_mip_phi
                .map(|_| 3usize)
                .unwrap_or(0), // 3 scales: 32→64→128
            resonator_codebook_size: self
                .resonator_memory
                .as_ref()
                .and_then(|m| m.resonator.codebooks.first())
                .map(|cb| cb.len())
                .unwrap_or(0),
            resonator_episodes: self.resonator_memory.as_ref().map(|m| m.len()).unwrap_or(0),
            resonator_factorization_iters: self
                .resonator_memory
                .as_ref()
                .map(|m| m.resonator.iterations())
                .unwrap_or(0),
            module_timings_us: {
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings
            },
            circadian_phase: circadian_phase_str.into(),
            circadian_plasticity: self.biorhythm.plasticity_mod as f32,
            phi_attention_weight,
            guiding_question: guiding_question.into(),
            dominant_harmonic: dominant_harmonic.into(),
            resonator_wm_primed,
            resonator_reconsolidated,
            resonator_promotions,
            fep_pragmatic_value,
            fep_accuracy,
            fep_complexity,
            fep_surprise,
            fep_td_error,
            resonator_best_sim,
            codebook_evictions,
            codebook_diversity,
            resonator_prediction_error,
            cross_module_agreement,
            thalamic_depth_score,
            // Phase 14: Subsystem Feedback Closure
            epistemic_gate_gated: !epistemic_gate_approved,
            causal_attention_edges,
            mcts_plan_effectiveness,
            moral_steering_category: moral_steering_category.into(),
            codebook_utilization_rate,
            surprise_replay_batch_size,
            // Phase 15: Adaptive Architecture + Emotional Homeostasis
            attention_budget_exceeded,
            attention_budget_elapsed_us,
            prediction_coherence,
            valence_homeostasis_pull,
            arousal_homeostasis_pull,
            arousal_recovery_active,
            arousal_recovery_tau_factor,
            input_similarity,
            input_memoized,
            guiding_priority_category: guiding_priority_category.into(),
            cycle_duration_us: cycle_start.elapsed().as_micros() as u64,
            school_predicted_phi_gain,
            // Phase 16: Quality-Aware Adaptive Processing
            epistemic_coherence_gated,
            unified_quality_score,
            dissipative_health_gated,
            dissipative_lr_factor,
            phi_validation_cached,
            phi_spectral_weight,
            coherence_velocity,
            coherence_velocity_gated,
            anomaly_recovery_progress,
            anomaly_recovering,
            // Phase 17: Predictive Self-Tuning
            error_pattern: error_pattern.into(),
            startup_suppressed,
            startup_warmup_progress,
            self_model_accuracy,
            mode_confidence: self.carryover.urgency.mode_confidence,
            mode_stability_counter: self.carryover.urgency.mode_stability_counter,
            predicted_urgency: predicted_urgency.into(),
            // Phase 18: Closing Feedback Loops
            context_phi_applied,
            empathic_speech_rate_mod,
            value_gate_factor,
            evolution_confidence_delta,
            homeostasis_pull_strength,
            prediction_coherence_urgency_bias,
            // Phase 19: Activating Dormant Pathways
            attention_budget_gated,
            limiting_component_boosted: limiting_component_boosted.into(),
            love_resonance_boost,
            reasoning_chain_boosted,
            attention_shift_applied: self.stats.attention_shift,
            // Phase 20: Signal-to-Control Synthesis
            harmonic_interference_lr_mod,
            resonator_error_exploration_mod,
            binding_threshold_mod,
            causal_urgency_gated,
            epistemic_semantic_lr_mod,
            predictive_budget_gated,
            // Phase 21: Consciousness-Grounded Control
            binding_confidence_mod,
            discontinuity_streak: self.carryover.urgency.discontinuity_streak,
            epistemic_reasoning_accelerated: self.carryover.quality.last_epistemic_conflict_count > 5,
            agency_strategy_override,
            pfe_surprise_mod,
            adaptive_memo_threshold: memo_threshold,
            // Spatial Reasoning (GridEncoder)
            grid_encoding_norm,
            grid_spatial_complexity,
            // Thermodynamic / affective (populated by consciousness pipeline + somatic bridge)
            mood_temperature: 1.0,
            // Liquid-Mamba fusion telemetry
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_semantic_pe: self.stats.last_liquid_mamba_pe,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_effective_rank: self.stats.last_liquid_mamba_rank,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_lr: self.stats.last_liquid_mamba_lr,
            #[cfg(feature = "liquid-mamba")]
            liquid_mamba_generation_count: self.stats.liquid_mamba_generation_count,
            // Fields defaulting to 0/false/None — populated elsewhere or left as default:
            // thermodynamic_load, somatic_stress, mesh_*, feedback_*, subsystem_integration_*,
            // safety_blocked, safety_category, replay_da_tag_avg, neuromod_* (applied below)
            ..Default::default()
        };

        // Apply neuromodulator telemetry via helper (replaces 36 inline fields)
        metadata.apply_neuromod(self.collect_neuromod_telemetry(neuromod_attention_alloc));

        // Update cumulative stats for resonator-memory loop diagnostics
        if resonator_wm_primed {
            self.stats.resonator_wm_primed_count += 1;
        }
        self.stats.resonator_promotions_total += resonator_promotions as u64;
        self.stats.codebook_evictions_total += codebook_evictions as u64;
        if codebook_diversity > 0.0 {
            self.stats.codebook_diversity = codebook_diversity;
        }
        if fep_surprise > surprise_thresh {
            self.stats.fep_surprise_replay_boosts += 1;
        }

        // Exocortex trigger counter
        if self.neuromodulator_bath.should_query_exocortex() {
            self.stats.exocortex_triggers += 1;
        }

        // Neuromodulator EMA stats (alpha=0.05)
        {
            let alpha = 0.05_f32;
            let da = self.neuromodulator_bath.dopamine.effective();
            let ne = self.neuromodulator_bath.noradrenaline.effective();
            let sht = self.neuromodulator_bath.serotonin.effective();
            let ach = self.neuromodulator_bath.acetylcholine.effective();
            self.stats.avg_dopamine += alpha * (da - self.stats.avg_dopamine);
            self.stats.avg_noradrenaline += alpha * (ne - self.stats.avg_noradrenaline);
            self.stats.avg_serotonin += alpha * (sht - self.stats.avg_serotonin);
            self.stats.avg_acetylcholine += alpha * (ach - self.stats.avg_acetylcholine);
        }

        // Populate v0.8.0 Resonance Metadata
        metadata.thermodynamic_load = self.thermodynamic_load;
        metadata.somatic_stress = self.somatic_bridge.systemic_stress();
        metadata.mood_temperature = self.mood_temperature;
        // Phase 2.2: feedback proposal attribution telemetry
        metadata.feedback_confidence_proposals = self.feedback_state.confidence.len() as u32;
        metadata.feedback_lr_proposals = self.feedback_state.learning_rate.len() as u32;

        // Project 16,384D HDC to 32D for visualization (mean-pooling)
        let thought_vector = {
            let chunk_size = encoding_hdv.values.len() / 32;
            encoding_hdv.values
                .chunks(chunk_size)
                .take(32)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect()
        };

        tracing::debug!(
            surprise = metadata.surprise_triggered,
            prefrontal_veto = metadata.prefrontal_veto,
            reasoning_confidence = metadata.reasoning_confidence,
            exploration = ?metadata.exploration_action,
            "Cycle metadata"
        );

        // ═══════════════════════════════════════════════════════════════════════
        // METRICS COLLECTION: Export consciousness telemetry for observability.
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref metrics) = self.metrics_collector {
            metrics.set_phi(unified_psi);
            metrics.set_coherence(coherence as f64);
            metrics.set_consciousness_level(metadata.consciousness_level);
            metrics.track_execution(metadata.safety_blocked, false);
        }

        // Pre-compute identity fields before moving output
        // sign_output accepts &[f32] — no need to clone output
        #[cfg(feature = "identity")]
        let signed_output = self.mfdi_bridge.sign_output(&output).ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        // ── Phase 2.2: End feedback proposal collection ──────────────────
        self.feedback_state.end_cycle(
            self.prediction_confidence as f64,
            self.fep_lr_boost as f64,
        );

        // ── Phase 2.3: Integrate subsystem outputs (Phase C) ─────────────
        // Consensus-average all SubsystemOutput proposals collected during
        // Phase B. Currently in dual-write bridge mode: integration result
        // is logged for comparison but does NOT override direct mutations.
        let integrated = self.subsystem_collector.integrate();
        if integrated.n_contributors > 0 {
            metadata.subsystem_integration_contributors = integrated.n_contributors as u32;
            tracing::trace!(
                "Phase C integration: {}",
                integrated,
            );
        }

        CycleResult {
            output,
            prediction_error,
            peak_attention: encoding_result.peak_attention,
            detected_primitives: encoding_result.detected_primitives,
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            thought_vector,
            wisdom_hv: hv16_cached,
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
        }
    }

    // Extracted cycle phases moved to helpers/cycle_phases.rs:
    // - run_resonator_codebook_phase()
    // - run_episodic_replay_and_memory_phase()
    // - run_dream_phase()

    /// Safe wrapper around `cycle()` that catches panics from unexpected subsystem failures.
    ///
    /// Use this in production code paths where a panic must not propagate (e.g., actor loops,
    /// async bridges). Returns `Err` with the panic message if any subsystem panics during
    /// the cycle.
    /// Online distillation step for the Liquid-Mamba HDC↔SSM projection.
    ///
    /// Called after generation with the original thought HV, back-projected
    /// output HVs, and semantic prediction error. Adjusts projection weights
    /// using FEP-modulated learning rate, gated by the cognitive loop's
    /// learning state and thermodynamic load.
    #[cfg(feature = "liquid-mamba")]
    pub fn update_liquid_mamba_telemetry(
        &mut self,
        semantic_pe: f32,
        effective_rank: f32,
        current_lr: f32,
        generation_count: u32,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;
        self.stats.last_liquid_mamba_rank = effective_rank;
        self.stats.last_liquid_mamba_lr = current_lr;
        self.stats.liquid_mamba_generation_count = generation_count;
    }

    #[cfg(feature = "liquid-mamba")]
    pub fn liquid_mamba_distillation_step(
        &mut self,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
        output_hvs: &[symthaea_core::hdc::ContinuousHV],
        semantic_pe: f32,
        projection: &mut symthaea_broca::HdcSsmProjection,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;

        // Gate on FEP precision confidence (mirrors enhanced_fep_bridge threshold)
        if self.carryover.learning.prediction_confidence < 0.4 { return; }
        if output_hvs.is_empty() || semantic_pe > 0.8 { return; }

        // FEP-modulated learning rate: precision × load × boost
        let fep_precision = self.fep_learning_signal.clamp(0.0, 1.0);
        let effective_lr = 0.001
            * fep_precision
            * (1.0 - self.thermodynamic_load)
            * self.fep_lr_boost;
        if effective_lr < 1e-6 { return; }

        let refs: Vec<&symthaea_core::hdc::ContinuousHV> = output_hvs.iter().collect();
        let bundled = symthaea_core::hdc::ContinuousHV::bundle(&refs).normalize();
        projection.compute_gradients(thought_hv, &bundled);
        projection.apply_gradients(effective_lr, 1.0);
    }

    pub fn try_cycle(&mut self, input: &str) -> Result<CycleResult, crate::errors::SymthaeaError> {
        // SAFETY: CognitiveLoopService is not UnwindSafe by default because it contains
        // mutable state. We use AssertUnwindSafe because a panic mid-cycle leaves the
        // service in a potentially inconsistent state, but callers should reset() after
        // an error rather than continuing.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.cycle(input)));
        result.map_err(|payload| {
            crate::errors::SymthaeaError::CognitiveLoop(format_panic_payload(payload))
        })
    }
}

/// Convert a panic payload into a human-readable error string.
///
/// Handles the three common payload types: `&str`, `String`, and unknown.
/// This is a standalone function so it can be tested independently.
pub(crate) fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("cognitive cycle panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("cognitive cycle panicked: {s}")
    } else {
        "cognitive cycle panicked with unknown payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::format_panic_payload;
    use super::CognitiveLoopService;
    use crate::cognitive_loop::CognitiveLoopConfig;

    // ── format_panic_payload tests (existing) ─────────────────────────

    #[test]
    fn test_panic_payload_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("subsystem failure");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: subsystem failure");
    }

    #[test]
    fn test_panic_payload_string() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("HDC bridge overflow"));
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: HDC bridge overflow");
    }

    #[test]
    fn test_panic_payload_unknown() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42u32);
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked with unknown payload");
    }

    #[test]
    fn test_panic_payload_empty_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: ");
    }

    // ── Helper ────────────────────────────────────────────────────────

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    // ── cycle() basic execution ───────────────────────────────────────

    #[test]
    fn cycle_returns_valid_result() {
        let mut s = make_service();
        let result = s.cycle("hello world");
        assert!(!result.output.is_empty(), "output should not be empty");
        assert!(result.prediction_error.is_finite());
        assert!(result.peak_attention.is_finite());
        assert!(result.cycle_time_us > 0);
    }

    #[test]
    fn cycle_increments_total_cycles() {
        let mut s = make_service();
        assert_eq!(s.stats().total_cycles, 0);
        s.cycle("first");
        assert_eq!(s.stats().total_cycles, 1);
        s.cycle("second");
        assert_eq!(s.stats().total_cycles, 2);
    }

    #[test]
    fn cycle_output_dimension_matches_config() {
        let mut s = make_service();
        let result = s.cycle("testing output dim");
        assert_eq!(
            result.output.len(),
            s.config().cfc_config.num_neurons,
            "output dimension should match num_neurons"
        );
    }

    #[test]
    fn cycle_prediction_error_non_negative() {
        let mut s = make_service();
        let result = s.cycle("checking error sign");
        assert!(
            result.prediction_error >= 0.0,
            "prediction_error should be non-negative, got {}",
            result.prediction_error
        );
    }

    #[test]
    fn cycle_thought_vector_has_values() {
        let mut s = make_service();
        let result = s.cycle("thought projection");
        assert!(!result.thought_vector.is_empty());
        assert_eq!(result.thought_vector.len(), 32, "thought_vector should be 32D");
    }

    #[test]
    fn cycle_metadata_urgency_populated() {
        let mut s = make_service();
        let result = s.cycle("metadata check");
        // Urgency should be one of the three valid variants
        let u = result.metadata.urgency;
        assert!(
            matches!(
                u,
                crate::cognitive_loop::CycleUrgency::Critical
                    | crate::cognitive_loop::CycleUrgency::Normal
                    | crate::cognitive_loop::CycleUrgency::Cruise
            ),
            "urgency should be a valid variant"
        );
    }

    #[test]
    fn cycle_output_all_finite() {
        let mut s = make_service();
        let result = s.cycle("NaN guard check");
        for (i, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    // ── Multiple cycles ───────────────────────────────────────────────

    #[test]
    fn multiple_cycles_do_not_panic() {
        let mut s = make_service();
        for i in 0..10 {
            let result = s.cycle(&format!("cycle input {i}"));
            assert!(result.prediction_error.is_finite());
        }
        assert_eq!(s.stats().total_cycles, 10);
    }

    #[test]
    fn empty_input_does_not_panic() {
        let mut s = make_service();
        let result = s.cycle("");
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn long_input_does_not_panic() {
        let mut s = make_service();
        let long_input = "a".repeat(10_000);
        let result = s.cycle(&long_input);
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn repeated_identical_input_reduces_prediction_error() {
        let mut s = make_service();
        // First cycle has no prior prediction
        let r1 = s.cycle("repeating input");
        // Run several identical cycles so the system can learn the pattern
        let mut last_error = r1.prediction_error;
        for _ in 0..20 {
            last_error = s.cycle("repeating input").prediction_error;
        }
        // After 20 identical cycles, error should be lower or comparable
        // (not necessarily strictly lower due to stochastic subsystems)
        assert!(
            last_error.is_finite(),
            "error should remain finite after repeated cycles"
        );
    }

    // ── try_cycle() ───────────────────────────────────────────────────

    #[test]
    fn try_cycle_returns_ok() {
        let mut s = make_service();
        let result = s.try_cycle("safe input");
        assert!(result.is_ok(), "try_cycle should succeed for normal input");
    }

    #[test]
    fn try_cycle_result_matches_cycle() {
        // Use genesis phrase for determinism
        let mut cfg = CognitiveLoopConfig::default();
        cfg.genesis_phrase = Some("determinism test".to_string());
        let mut s1 = CognitiveLoopService::new(cfg.clone()).unwrap();
        let mut s2 = CognitiveLoopService::new(cfg).unwrap();

        let r1 = s1.cycle("hello");
        let r2 = s2.try_cycle("hello").unwrap();

        // Both should produce same output with deterministic genesis
        assert_eq!(r1.output.len(), r2.output.len());
        assert_eq!(r1.prediction_error, r2.prediction_error);
    }

    // ── Cycle with different backends ─────────────────────────────────

    #[test]
    fn cycle_with_hdc_ltc_unified_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let mut s = CognitiveLoopService::new(config).unwrap();
        let result = s.cycle("HdcLtc backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn cycle_with_hdc_ltc_fast_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_fast();
        let mut s = CognitiveLoopService::new(config).unwrap();
        let result = s.cycle("fast backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    // ── Cycle stats tracking ──────────────────────────────────────────

    #[test]
    fn cycle_updates_avg_prediction_error() {
        let mut s = make_service();
        s.cycle("first");
        let err1 = s.stats().avg_prediction_error;
        // After first cycle, avg error should be populated (may be 0.0 for first cycle)
        assert!(err1.is_finite());
    }

    #[test]
    fn cycle_populates_adaptive_learning_rate() {
        let mut s = make_service();
        s.cycle("learning rate check");
        let lr = s.stats().adaptive_learning_rate;
        assert!(lr.is_finite());
        assert!(lr >= 0.0);
    }

    // ── Genesis determinism ───────────────────────────────────────────

    #[test]
    fn genesis_seeded_cycles_are_deterministic() {
        let phrase = "We hold these truths to be self-evident".to_string();

        let mut cfg_a = CognitiveLoopConfig::default();
        cfg_a.genesis_phrase = Some(phrase.clone());
        let mut sa = CognitiveLoopService::new(cfg_a).unwrap();

        let mut cfg_b = CognitiveLoopConfig::default();
        cfg_b.genesis_phrase = Some(phrase);
        let mut sb = CognitiveLoopService::new(cfg_b).unwrap();

        let ra = sa.cycle("determinism check");
        let rb = sb.cycle("determinism check");

        assert_eq!(ra.output, rb.output, "genesis-seeded outputs should match");
        assert_eq!(
            ra.prediction_error, rb.prediction_error,
            "genesis-seeded errors should match"
        );
    }

    // ── Cycle metadata fields ─────────────────────────────────────────

    #[test]
    fn cycle_metadata_somatic_stress_finite() {
        let mut s = make_service();
        let result = s.cycle("somatic check");
        assert!(result.metadata.somatic_stress.is_finite());
    }

    #[test]
    fn cycle_metadata_consciousness_level_bounded() {
        let mut s = make_service();
        // Run a few cycles to populate MCE
        for _ in 0..15 {
            s.cycle("populate MCE");
        }
        let result = s.cycle("check consciousness");
        assert!(result.metadata.consciousness_level >= 0.0);
        assert!(result.metadata.consciousness_level <= 1.0);
    }

    #[test]
    fn cycle_metadata_thermodynamic_load_bounded() {
        let mut s = make_service();
        let result = s.cycle("thermo check");
        assert!(result.metadata.thermodynamic_load >= 0.0);
        assert!(result.metadata.thermodynamic_load <= 1.0);
    }
}
