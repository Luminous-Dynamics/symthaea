//! Core cognitive cycle implementation with parallel post-processing.
//!
//! Contains the main `cycle()` method which implements the bidirectional
//! HDC-CfC loop with rayon-parallelized subsystem updates.

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use ndarray::Array1;
use rayon::join as rayon_join;
use std::time::Instant;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;

// ═══════════════════════════════════════════════════════════════════════════════
// Tuning Constants: centralized for sweep-ability and self-documentation
// ═══════════════════════════════════════════════════════════════════════════════

// -- Amortization strategy --
// Subsystem intervals use co-prime (prime) values to prevent processing pileups.
// Old round-number intervals (10, 20, 50, 100, …) all aligned at LCM boundaries
// (e.g., cycle 100 fired ~15 subsystems simultaneously). Prime intervals ensure
// at most 2 subsystems coincide on any given cycle.
//   5→7, 10→11, 15→13, 20→19, 25→23, 50→47, 100→97, 200→199, 1000→997

// -- Moral evaluation (constants used only in cycle.rs; others moved to helpers.rs) --
const MORAL_CONCERN_THRESHOLD: f32 = -0.3; // score below this triggers concern
const MORAL_BENEFIT_THRESHOLD: f32 = 0.5; // score above this boosts confidence
const MORAL_CONCERN_EXPLORATION_DAMPEN: f32 = 0.5; // reduce exploration on moral concern
const MORAL_CONCERN_PAUSE_BOOST: f32 = 1.5; // slow down on moral concern
const MORAL_BENEFIT_CONFIDENCE_BOOST: f32 = 1.05; // confidence nudge for positive morality

// -- Surprise & exploration --
const QUANTUM_COHERENCE_THRESHOLD: f64 = 0.5; // coherence above this boosts exploration
const QUANTUM_COHERENCE_BOOST_SCALE: f32 = 0.2; // strength of coherence → exploration


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

// -- FEP tuning --
const FEP_SURPRISE_SCALE: f32 = 3.0; // free-energy divisor for surprise boost
const FEP_LR_DECAY: f32 = 0.95; // boost decay rate when not surprised


// -- Dominance estimation --
const DOMINANCE_FLOW_BASE: f64 = 0.6;
const DOMINANCE_FLOW_SCALE: f64 = 0.2;
const DOMINANCE_CONFIDENT: f64 = 0.4;
const DOMINANCE_DEFAULT: f64 = 0.2;

// -- Resonance tau modulation --
const RESONANCE_TAU_CENTER: f64 = 0.5; // neutral frequency
const RESONANCE_TAU_SCALE: f32 = 0.1; // ±5% CfC time-step modulation


// -- Policy agreement (KL gate) --
const POLICY_SOFT_THRESHOLD: f64 = 0.2; // FEP prob to accept MCTS choice
const POLICY_FULL_AGREEMENT_BOOST: f32 = 1.2; // confidence boost on full agreement
const POLICY_WINDOW_SIZE: usize = 20; // agreement tracking window
const POLICY_MIN_WINDOW: usize = 5; // minimum samples for temp adaptation
const POLICY_TEMP_BASE: f64 = 0.5; // min softmax temperature
const POLICY_TEMP_RANGE: f64 = 1.5; // temperature range [0.5, 2.0]

// -- GWT / broadcast --
const GWT_BROADCAST_CONFIDENCE_BOOST: f32 = 0.03;

// -- MCE consciousness --
const MCE_LR_BOOST_SCALE: f32 = 0.1; // up to +10% LR from consciousness
const MCE_BOOST_DECAY: f32 = 0.9; // decay when MCE doesn't fire

use super::helpers;
use super::helpers::MEMORY_RECALL_TOP_K;
use super::temporal_network::TemporalNetwork;
use super::training::TrainingSample;
use super::{
    ActionHint, AdaptiveBehavior, CognitiveLoopService, CycleLearningResult, CycleResult,
    ResponseStrategy, TrainingMethod,
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

        // Snapshot exploration_urge for end-of-cycle budget clamping (Task B)
        let exploration_urge_start = self.curiosity_drive.exploration_urge;

        // Snapshot confidence for end-of-cycle drift clamping (Task G)
        self.carryover.learning.prediction_confidence = self.prediction_confidence;

        // Chronobiology: refresh biorhythm every 97 cycles (co-prime amortization)
        self.biorhythm_refresh_counter += 1;
        if self.biorhythm_refresh_counter >= 97 {
            self.biorhythm = crate::chronobiology::Biorhythm::current();
            self.biorhythm_refresh_counter = 0;
        }
        // Apply circadian plasticity to learning rate (Night=high plasticity, Day=low)
        let circadian_lr = self.stats.adaptive_learning_rate * self.biorhythm.plasticity_mod as f32;
        self.stats.adaptive_learning_rate = circadian_lr.clamp(0.0001, 0.1);

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
        let selected_strategy = if moral_concern_detected {
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
            if let Some(&(plan_action, plan_confidence)) =
                self.carryover.history.mcts_plan.as_ref()
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

        // ═══════════════════════════════════════════════════════════════════════
        // 1.2 Adaptive Learning Threshold + Urgency
        // ═══════════════════════════════════════════════════════════════════════
        // Science: Friston (2010) — precision (inverse uncertainty) modulates PE weighting.
        // Low confidence → lower threshold (learn on smaller errors); high confidence → raise it.
        // Combined with temporal coherence scaling (adaptive_threshold_scale).
        // + Thelen & Smith (1994): exploration urge bidirectionally couples to threshold.
        let confidence_scale = 1.0 + (self.prediction_confidence - 0.5) * 0.4; // ±20% from confidence
        let exploration_scale =
            1.0 - (self.curiosity_drive.exploration_urge - 0.5) * 0.2; // high explore → lower threshold
        let effective_threshold = self.config.learning_threshold
            * self.carryover.learning.adaptive_threshold_scale
            * confidence_scale
            * exploration_scale;
        if prediction_error < effective_threshold {
            self.carryover.urgency.consecutive_low_error = self.carryover.urgency.consecutive_low_error.saturating_add(1);
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
        let hysteresis_threshold = match self.carryover.urgency.urgency {
            super::CycleUrgency::Cruise => effective_threshold * 1.2, // harder to leave Cruise
            super::CycleUrgency::Critical => effective_threshold * 0.8, // harder to leave Critical
            _ => effective_threshold,
        };
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
        let urgency = match self.cognitive_depth {
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

        // FEEDBACK: Quantum coherence boosts exploration (prev cycle)
        // Science: Lambert (2013) — quantum coherence enhances biological search
        if self.carryover.consciousness.quantum_coherence > QUANTUM_COHERENCE_THRESHOLD {
            let coherence_boost =
                (self.carryover.consciousness.quantum_coherence - QUANTUM_COHERENCE_THRESHOLD) as f32 * QUANTUM_COHERENCE_BOOST_SCALE;
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
        // Coherence gate: skip resonator recall during unstable CfC dynamics
        // Science: noisy priors during turbulent dynamics can destabilize predictions
        // Uses previous cycle's smoothed coherence (updated at line ~646)
        let reflection_thresholds = self.self_reflection.get_thresholds();
        let resonator_coherence_gate = self.coherence_bridge.smoothed_coherence()
            > reflection_thresholds.coherence_gate
            || self.stats.total_cycles < 10; // bypass gate during warmup
        if resonator_coherence_gate && urgency.should_run(self.stats.total_cycles, 1, 1, 4) {
        if let Some(ref mut res_mem) = self.resonator_memory {
            let res_start = Instant::now();

            // Dimension guard: skip if compressed_state doesn't match resonator codebook dim
            let res_dim_ok = compressed_state.len() == res_mem.resonator.config.dim;
            if res_dim_ok && !res_mem.is_empty() {
                // Retrieve resonator episodes similar to current content
                if let Ok(matches) = res_mem.retrieve(&[("content", &compressed_state)]) {
                    let top_matches: Vec<_> = matches.into_iter().take(MEMORY_RECALL_TOP_K).collect();

                    // Extract ALL owned data from borrowed episodes before releasing
                    // res_mem borrow. retrieve() returns Vec<&Episode>, so we must
                    // copy what we need before calling query_factorize(&mut res_mem).
                    let best_match_sim = top_matches.iter()
                        .map(|m| {
                            let dot: f32 = compressed_state.iter()
                                .zip(m.hv.iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            let na: f32 = compressed_state.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let nb: f32 = m.hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
                        })
                        .fold(0.0f32, f32::max);
                    let match_timestamps: Vec<u64> = top_matches.iter()
                        .map(|m| m.timestamp)
                        .collect();
                    resonator_best_sim = best_match_sim;

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
                        for v in &mut b { *v /= n; }
                        Some(b)
                    } else {
                        None
                    };

                    // Drop episode references — releases the &mut res_mem borrow
                    drop(top_matches);

                    // Now safe to call query_factorize (no outstanding borrows on res_mem)
                    if let Some(bundled) = bundled {
                        if let Ok(factors) = res_mem.query_factorize(
                            &bundled,
                            &[("content", &compressed_state)],
                        ) {
                            for (label, _hv) in &factors {
                                match label.as_str() {
                                    "positive" => {
                                        self.emotion_contagion.valence =
                                            (self.emotion_contagion.valence + 0.1).clamp(-1.0, 1.0);
                                    }
                                    "negative" => {
                                        self.emotion_contagion.valence =
                                            (self.emotion_contagion.valence - 0.1).clamp(-1.0, 1.0);
                                    }
                                    "high" => {
                                        self.prediction_confidence =
                                            (self.prediction_confidence + 0.03).clamp(0.0, 1.0);
                                    }
                                    _ => {} // neutral, medium, proto_N — no bias
                                }
                            }
                        }
                    }

                    // Track 3a: Resonator recall → confidence priming
                    // Science: Tulving (1983) — episodic retrieval primes processing
                    if best_match_sim > 0.3 {
                        self.prediction_confidence =
                            (self.prediction_confidence + best_match_sim * 0.02).clamp(0.0, 1.0);
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
            self.fep_agent.precision.prior_precision =
                (self.fep_agent.precision.prior_precision
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
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 + goal_lr_boost)).clamp(1.0, 2.0);
            }
            // Successful prediction (low error) during goal pursuit → exploration toward goal
            if prediction_error < self.config.learning_threshold && goal_priority > 0.3 {
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + goal_priority * 0.03).clamp(0.0, 1.0);
            }
        }

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
        module_timings.core_semantic_lookup = _t_core.elapsed().as_micros() as u64;

        // 3. Convert to ndarray for CfC (copy elements directly — avoids Vec clone)
        let input_array: Array1<f32> = compressed_state.iter().copied().collect();

        // 4. Step CfC forward with current input
        // FEEDBACK: Resonance frequency modulates CfC time constant (prev cycle)
        // Science: Buzsáki (2006) — neural oscillations modulate processing speed
        let resonance_tau_factor = if self.carryover.history.resonance_frequency > 0.0 {
            let deviation = (self.carryover.history.resonance_frequency as f32 - RESONANCE_TAU_CENTER as f32).clamp(-0.5, 0.5);
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
        let delta_t = self.config.cfc_config.delta_t
            * resonance_tau_factor
            * arousal_tau_factor
            * codebook_tau_factor;
        let _t_core = Instant::now();
        if let Err(e) = self.temporal_network.step(&input_array, delta_t) {
            tracing::warn!(error = %e, "CfC temporal step failed — continuing with stale state");
        }
        module_timings.core_cfc_step = _t_core.elapsed().as_micros() as u64;

        // 5. Get multi-scale predictions using CfC's O(1) predict_forward
        // This is the key advantage: instant prediction at any future time
        let _t_core = Instant::now();
        let prediction = self.get_multi_scale_prediction(&input_array);

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
                self.fep_lr_boost = (self.fep_lr_boost + stiffness_nudge).clamp(1.0, 2.0);
            } else if wm_stiffness < 0.2 {
                let spongy_dampen = (0.2 - wm_stiffness) * 0.15;
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 - spongy_dampen)).max(1.0);
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
        // Reuse tau_owned from above instead of calling flattened_tau() again
        let flattened_tau: Vec<f32> = tau_owned.iter().flat_map(|a| a.iter().copied()).collect();
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

        // ── Apply previous cycle's MCTS plan at reduced weight ──────────────
        // Science: MCTS plans (deliberative) complement FEP actions (habitual).
        // When the prior cycle's deliberative system produced a confident plan
        // (confidence > 0.7), apply its effect at 40% strength alongside the
        // current FEP action — "dual process" theory (Kahneman 2011).
        if let Some((plan_action, plan_confidence)) = self.carryover.history.mcts_plan.take() {
            if plan_confidence > 0.7 && plan_action != fep_action_idx {
                let plan_weight = plan_confidence * 0.4;
                match plan_action {
                    0 => {
                        // Plan said "exploit" — nudge LR down (floor at 1.0)
                        self.fep_lr_boost =
                            (self.fep_lr_boost * (1.0 - plan_weight * 0.1)).max(1.0);
                    }
                    1 => {
                        // Plan said "consolidate" — reinforce prediction confidence
                        self.prediction_confidence = (self.prediction_confidence
                            + plan_weight * 0.05)
                            .clamp(0.0, 1.0);
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
        let (fep_accuracy, fep_complexity, fep_surprise, fep_td_error) =
            if let Some(ref fe) = self.fep_agent.last_fe_components {
                // High accuracy → stabilize (model fits well)
                if fe.accuracy > 0.5 {
                    self.prediction_confidence =
                        (self.prediction_confidence + 0.01).clamp(0.0, 1.0);
                }
                // High complexity → reduce LR (Occam's razor: penalize overfitting)
                if fe.complexity > 1.0 {
                    self.fep_lr_boost =
                        (self.fep_lr_boost * (1.0 - ((fe.complexity - 1.0).min(0.5) * 0.1) as f32))
                            .max(1.0);
                }
                // High surprise → boost exploration (complement existing is_surprised gate)
                if fe.surprise > reflection_thresholds.surprise as f64 {
                    let s_explore = ((fe.surprise - reflection_thresholds.surprise as f64) * 0.1).min(0.05) as f32;
                    self.curiosity_drive.exploration_urge =
                        (self.curiosity_drive.exploration_urge + s_explore).clamp(0.0, 1.0);
                }
                (fe.accuracy, fe.complexity, fe.surprise, fe.prediction_error)
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
                    let _graph = enhancer.run_discovery();
                }
            }
        }

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
        } else if moral_score > MORAL_BENEFIT_THRESHOLD {
            // Positive moral alignment boosts confidence slightly
            self.prediction_confidence = (self.prediction_confidence * MORAL_BENEFIT_CONFIDENCE_BOOST).clamp(0.0, 1.0);
        }

        // Surprise-gated learning rate boost: when FEP detects surprise, accelerate adaptation
        if is_surprised {
            let surprise_boost =
                (self.fep_agent.current_free_energy() as f32 / FEP_SURPRISE_SCALE).clamp(0.1, 0.5);
            self.fep_lr_boost = (self.fep_lr_boost + surprise_boost).clamp(1.0, 2.0);
        } else {
            // Decay boost back toward 1.0 when not surprised
            self.fep_lr_boost = (self.fep_lr_boost * FEP_LR_DECAY).max(1.0);
        }

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
        } // end if let Some(enhanced_result)

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
                            self.fep_lr_boost = (self.fep_lr_boost * 0.9).max(1.0);
                        }
                        super::AdjustmentDirection::Increase => {
                            self.fep_lr_boost = (self.fep_lr_boost * 1.1).clamp(1.0, 2.0);
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

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.exp EXPERIENCE BUS: Update principled signals from cognitive state
        // Maps cycle values to 5 principled signals (Active Inference).
        // Science: Friston (2010) — principled signals drive behavior.
        // ═══════════════════════════════════════════════════════════════════════
        let guiding_question: String;
        let dominant_harmonic: String;
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
            guiding_question = bus.current_guiding_question().to_string();
            dominant_harmonic = format!("{:?}", bus.dominant_harmonic());
        } else {
            guiding_question = String::new();
            dominant_harmonic = String::new();
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
            let fep_prob_for_mcts = fep_action_probs
                .get(mcts_idx)
                .copied()
                .unwrap_or(0.0);
            if mcts_idx == fep_action_idx {
                // Full agreement: both systems chose the same action — boost confidence
                reasoning_plan_confidence = (reasoning_plan_confidence * POLICY_FULL_AGREEMENT_BOOST).min(1.0);
                policy_agreement = true;
            } else if fep_prob_for_mcts > POLICY_SOFT_THRESHOLD {
                // Soft agreement: FEP assigns reasonable probability to MCTS choice
                policy_agreement = true;
                reasoning_plan_confidence = (reasoning_plan_confidence
                    * (1.0 + fep_prob_for_mcts as f32 * 0.3))
                    .min(1.0);
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
                let agree_rate = self
                    .policy_agreement_window
                    .iter()
                    .filter(|&&a| a)
                    .count() as f64
                    / self.policy_agreement_window.len() as f64;
                let adaptive_temp = POLICY_TEMP_BASE + (1.0 - agree_rate) * POLICY_TEMP_RANGE;
                self.fep_agent.config.action_temperature = adaptive_temp;
            }
        }

        // Store current MCTS plan for next cycle's dual-process application
        self.carryover.history.mcts_plan = reasoning_plan_action
            .map(|a| (a, reasoning_plan_confidence));

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1.5 Metacognitive Monitoring (Phi trajectory anomaly detection)
        // ═══════════════════════════════════════════════════════════════════════
        // After reasoning, observe unified Phi for trajectory anomalies.
        // Anomalies (drops, plateaus, oscillations) indicate reasoning degradation
        // and dampen the learning rate to avoid consolidating bad patterns.
        let mut metacognitive_anomaly = false;
        if let Some(ref mut monitor) = self.metacognitive_monitor {
            if monitor.observe_phi(unified_psi) {
                metacognitive_anomaly = true;
                // Dampen learning rate when reasoning is degrading
                reasoning_lr_factor *= 0.5;
                tracing::debug!(
                    target: "cognitive_loop::metacognition",
                    unified_psi,
                    "Metacognitive anomaly detected — dampening learning rate"
                );
            }
        }

        // Compose effective LR from all modulation sources (flow, curiosity, FEP, MCE, subsystem)
        let effective_lr = self.compose_effective_lr(semantic_lr_factor, reasoning_lr_factor);

        // 11. Learn if error is significant AND we have a previous state AND not paused
        // FEEDBACK: Narrative-GWT veto suppresses learning (consciousness governance)
        // Science: Baars (2005) — global workspace vetoing prevents consolidation
        // FEEDBACK: Consciousness-gated learning — system must be "awake" to consolidate
        // Science: Dehaene (2014) — conscious access required for durable learning
        let _t_core = Instant::now();
        let consciousness_awake = self.carryover.history.consciousness_level > 0.0
            || self.stats.total_cycles < 20; // grace period for boot-up
        let (learning_occurred, training_loss) = if prediction_error
            > effective_threshold
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
                // First cycle: bootstrap with self-prediction
                let current_array: Array1<f32> = compressed_state.iter().copied().collect();
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
        let _t_core = Instant::now();

        // Pre-compute read-only values needed by parallel branches
        let pp_total_cycles = self.stats.total_cycles;
        let pp_in_flow = self.flow_state.in_flow;
        let pp_emotional_valence = self.emotion_contagion.prosody_valence();
        let pp_phi = self.unification_engine.psi as f32;
        let pp_smoothed_coh = self.coherence_bridge.smoothed_coherence() as f64;
        // World model error → resonator storage importance bias
        // Science: Rescorla-Wagner (1972) — surprising events deserve higher encoding priority
        let pp_wm_importance_boost = self.world_model.avg_error.clamp(0.0, 1.0) * 0.3;
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
                stability_regime, discovery_service, &hv16_cached,
                delta_t, pp_total_cycles, urgency,
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
                wm_importance_boost: pp_wm_importance_boost,
            };

            rayon_join(
                || helpers::parallel_semantic_causal(
                    semantic_memory, causal_enhancer, semantic_hdc,
                    &compressed_state, &output, prediction_error, pp_total_cycles,
                ),
                || helpers::parallel_episodic_learning(
                    episodic_memory, resonator_memory, prediction_confidence_ref,
                    primitive_belief_bridge, prev_primitive_state, fep_learning_signal,
                    closed_learning_loop, &episodic_ctx, cycle_learning_result,
                ),
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
        // CONSCIOUSNESS METRICS: Extracted to cycle_consciousness.rs
        // Includes: primitive consciousness, temporal primitives, lattice,
        // compositionality, value evaluator, consciousness profile, context-aware
        // evolution, semantic value embedder, harmonies, composition rules,
        // fiduciary harmonics, primitive reasoning, causal self-explanation,
        // adaptive reasoning, epistemic tiers, phi validation, dissipative
        // consciousness, epistemic conflict, consciousness equation v2.
        // ═══════════════════════════════════════════════════════════════════════
        let consciousness_metrics = self.compute_consciousness_metrics(
            hv16_cached,
            unified_psi,
            coherence,
            prediction_error,
            phi_attention_weight,
            &compressed_state,
            input,
            urgency,
            &mut module_timings,
        );

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
        let consciousness_profile_composite = consciousness_metrics.consciousness_profile_composite;
        let synergy_enhanced_composite = consciousness_metrics.synergy_enhanced_composite;
        let emergent_properties_count = consciousness_metrics.emergent_properties_count;
        let reasoning_context = consciousness_metrics.reasoning_context;
        let context_phi_weight = consciousness_metrics.context_phi_weight;
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
        let dissipative_health = consciousness_metrics.dissipative_health;
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
            hv16_cached,
            unified_psi,
            coherence,
            prediction_error,
            phi_attention_weight,
            &compressed_state,
            input,
            &active_primitive_names,
            &mut module_timings,
        );

        // Destructure subsystem metrics for use by later phases
        let hierarchical_ltc_phi = subsystem_metrics.hierarchical_ltc_phi;
        let evolution_generation = subsystem_metrics.evolution_generation;
        let evolution_phi_delta = subsystem_metrics.evolution_phi_delta;
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

        // (Inline code for HIERARCHICAL LTC through MULTI-OBJECTIVE EVOLUTION
        //  removed — now in run_advanced_subsystems() in cycle_subsystems.rs)


        // ═══════════════════════════════════════════════════════════════════════
        // RESONATOR CODEBOOK GROWTH: add novel patterns to semantic codebook
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        if let Some(ref mut res_mem) = self.resonator_memory {
            let res_dim_ok = compressed_state.len() == res_mem.resonator.config.dim;
            if res_dim_ok && self.stats.total_cycles % self.config.resonator_growth_interval == 0 {
                if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                    // Check novelty: max similarity to existing symbols
                    let max_sim = semantic_cb.symbols.iter()
                        .map(|(_, hv)| {
                            let dot: f32 = compressed_state.iter().zip(hv.iter()).map(|(a, b)| a * b).sum();
                            let na: f32 = compressed_state.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let nb: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
                        })
                        .fold(0.0f32, f32::max);

                    if max_sim < self.config.resonator_novelty_threshold
                        && semantic_cb.len() < self.config.resonator_max_symbols
                    {
                        semantic_cb.add(
                            &format!("learned_{}", self.stats.total_cycles),
                            compressed_state.clone(),
                        );

                        // Track B: Lattice meet for semantic grounding of learned symbol
                        if let Some(ref lattice) = self.primitive_lattice {
                            if active_primitive_names.len() >= 2 {
                                if let (Some(a), Some(b)) = (
                                    lattice.element_index_by_name(&active_primitive_names[0]),
                                    lattice.element_index_by_name(&active_primitive_names[1]),
                                ) {
                                    if let Some(meet_idx) = lattice.meet(a, b) {
                                        let last = semantic_cb.symbols.len() - 1;
                                        semantic_cb.symbols[last].0 =
                                            format!("learned_{}_{}", self.stats.total_cycles,
                                                    lattice.elements[meet_idx].name);
                                    }
                                }
                            }
                        }

                        tracing::trace!(
                            symbols = semantic_cb.len(),
                            max_sim,
                            cycle = self.stats.total_cycles,
                            "Resonator: novel pattern added to semantic codebook"
                        );
                    }
                }
            }
        }

        // Track A-2: Causal chain content → resonator codebook symbols
        if !causal_codebook_entries.is_empty() {
            if let Some(ref mut res_mem) = self.resonator_memory {
                for (label, hv) in &causal_codebook_entries {
                    if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                        if semantic_cb.len() < self.config.resonator_max_symbols
                            && hv.len() == res_mem.resonator.config.dim
                        {
                            semantic_cb.add(label, hv.clone());
                        }
                    }
                }
            }
        }

        module_timings.resonator_codebook = _t.elapsed().as_micros() as u64;

        // Track 3c: High-Phi episodes → resonator codebook promotion
        // Science: Dehaene (2014) — conscious access creates durable representations
        // Co-prime cadence (97 cycles) avoids interference with other periodic tasks
        let _t = Instant::now();
        let mut resonator_promotions: usize = 0;
        let mut codebook_evictions: usize = 0;
        if self.stats.total_cycles % 97 == 0 && self.stats.total_cycles > 0 {
            let top_eps = self
                .phi_episodic_replay
                .as_ref()
                .map(|replay| replay.get_top_episodes(3))
                .unwrap_or_default();

            if !top_eps.is_empty() {
                if let Some(ref mut res_mem) = self.resonator_memory {
                    let dim = res_mem.resonator.config.dim;
                    if let Some(ref mut semantic_cb) = res_mem.resonator.codebooks.get_mut(0) {
                        for ep in &top_eps {
                            if ep.psi > 0.5 {
                                let ep_vec = &ep.input.values;
                                if ep_vec.len() != dim {
                                    continue;
                                }

                                // Track 3c-evict: Prune most redundant entry when at capacity
                                // Science: competitive learning — maintain codebook diversity
                                if semantic_cb.len() >= self.config.resonator_max_symbols
                                    && semantic_cb.len() > 1
                                {
                                    let n = semantic_cb.symbols.len();
                                    let mut max_redundancy = f32::MIN;
                                    let mut evict_idx = 0;
                                    for i in 0..n {
                                        let avg_sim: f32 = (0..n)
                                            .filter(|&j| j != i)
                                            .map(|j| {
                                                let dot: f32 = semantic_cb.symbols[i]
                                                    .1
                                                    .iter()
                                                    .zip(semantic_cb.symbols[j].1.iter())
                                                    .map(|(a, b)| a * b)
                                                    .sum();
                                                let na: f32 = semantic_cb.symbols[i]
                                                    .1
                                                    .iter()
                                                    .map(|x| x * x)
                                                    .sum::<f32>()
                                                    .sqrt();
                                                let nb: f32 = semantic_cb.symbols[j]
                                                    .1
                                                    .iter()
                                                    .map(|x| x * x)
                                                    .sum::<f32>()
                                                    .sqrt();
                                                if na > 0.0 && nb > 0.0 {
                                                    dot / (na * nb)
                                                } else {
                                                    0.0
                                                }
                                            })
                                            .sum::<f32>()
                                            / (n - 1) as f32;
                                        if avg_sim > max_redundancy {
                                            max_redundancy = avg_sim;
                                            evict_idx = i;
                                        }
                                    }
                                    semantic_cb.symbols.remove(evict_idx);
                                    codebook_evictions += 1;
                                }

                                if semantic_cb.len() < self.config.resonator_max_symbols {
                                    semantic_cb.add(
                                        &format!(
                                            "phi_{:.0}_{}",
                                            ep.psi * 100.0,
                                            ep.timestamp
                                        ),
                                        ep_vec.clone(),
                                    );
                                    resonator_promotions += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        module_timings.high_phi_promotion = _t.elapsed().as_micros() as u64;

        // Track 3e: Codebook diversity metric
        // Science: competitive learning — low diversity = redundant representations
        // Compute average pairwise cosine distance (every 50 cycles to amortize cost)
        let codebook_diversity: f32 = if self.stats.total_cycles % 50 == 0 {
            if let Some(ref res_mem) = self.resonator_memory {
                if let Some(ref semantic_cb) = res_mem.resonator.codebooks.first() {
                    let n = semantic_cb.symbols.len();
                    if n >= 2 {
                        let mut total_dist = 0.0f32;
                        let mut pairs = 0u32;
                        for i in 0..n {
                            for j in (i + 1)..n {
                                let dot: f32 = semantic_cb.symbols[i]
                                    .1
                                    .iter()
                                    .zip(semantic_cb.symbols[j].1.iter())
                                    .map(|(a, b)| a * b)
                                    .sum();
                                let na: f32 = semantic_cb.symbols[i]
                                    .1
                                    .iter()
                                    .map(|x| x * x)
                                    .sum::<f32>()
                                    .sqrt();
                                let nb: f32 = semantic_cb.symbols[j]
                                    .1
                                    .iter()
                                    .map(|x| x * x)
                                    .sum::<f32>()
                                    .sqrt();
                                let sim = if na > 0.0 && nb > 0.0 {
                                    dot / (na * nb)
                                } else {
                                    0.0
                                };
                                total_dist += 1.0 - sim; // distance = 1 - similarity
                                pairs += 1;
                            }
                        }
                        if pairs > 0 {
                            total_dist / pairs as f32
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
            }
        } else {
            self.stats.codebook_diversity // carry forward cached value
        };

        // Track 3f: Codebook diversity → exploration governor
        // Science: competitive learning — low diversity signals representational collapse
        // Low diversity → boost exploration urge (seek novel inputs)
        // High diversity → allow exploitation (good codebook coverage)
        let div_low = reflection_thresholds.diversity_low;
        let div_high = reflection_thresholds.diversity_high;
        if codebook_diversity > 0.0 {
            if codebook_diversity < div_low {
                // Representational collapse risk — boost exploration
                let diversity_boost = (div_low - codebook_diversity) * 0.2;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + diversity_boost).clamp(0.0, 1.0);
            } else if codebook_diversity > div_high {
                // Good coverage — allow exploitation, dampen exploration slightly
                let exploit_dampen = (codebook_diversity - div_high) * 0.1;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge - exploit_dampen).clamp(0.0, 1.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // DEMAND-DRIVEN CONSOLIDATION TRIGGERS
        // ═══════════════════════════════════════════════════════════════════════
        // Trigger early episodic replay when:
        //   (a) prediction error spikes >2x the moving average, or
        //   (b) semantic memory returned zero hits (retrieval miss)
        // The periodic 100-cycle floor is still enforced by should_replay().
        let _t = Instant::now();
        if let Some(ref mut replay) = self.phi_episodic_replay {
            let avg_err = self.stats.avg_prediction_error;
            let error_spike = avg_err > 0.01 && prediction_error > avg_err * 2.0;
            let semantic_miss = self.semantic_memory.stats().semantic_misses > 0
                && memory_context_boost == 0.0 // no episodic memories recalled this cycle
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

        module_timings.demand_consolidation = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // SEQUENTIAL: Episodic replay + Memory coordinator
        // ═══════════════════════════════════════════════════════════════════════
        // These remain sequential because:
        // - Episodic replay needs &mut temporal_network for CfC retraining
        // - Memory coordinator needs &mut phi_episodic_replay after replay completes
        let _t = Instant::now();
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
                            avg_psi = result.average_psi,
                            "Episodic replay session completed"
                        );

                        // Track 3g: Dream consolidation — resonator factorization of replayed episodes
                        // Science: Stickgold (2005) — sleep replay extracts gist representations
                        // After episodic replay, factorize top episodes through the resonator to
                        // extract clean semantic components and strengthen codebook representations.
                        if let Some(ref mut res_mem) = self.resonator_memory {
                            if !res_mem.resonator.codebooks.is_empty() {
                                let res_dim = res_mem.resonator.config.dim;
                                let top_eps = replay.get_top_episodes(3);
                                for ep in &top_eps {
                                    // Project episode input down to resonator dim
                                    let ep_vals = &ep.input.values;
                                    if ep_vals.len() >= res_dim {
                                        let projected: Vec<f32> =
                                            ep_vals.iter().take(res_dim).copied().collect();
                                        if let Ok(factors) = res_mem.resonator.factorize(&projected) {
                                            // Each factor strengthens its codebook entry via re-exposure
                                            // This is the "gist extraction" — dreaming distills episodes
                                            // into their categorical components
                                            for (label, _factor_hv) in &factors {
                                                tracing::trace!(
                                                    label,
                                                    psi = ep.psi,
                                                    "Dream factorized episode component"
                                                );
                                            }
                                            let _ = factors.len(); // factorization itself updates resonator state
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Memory coordinator: broadcast signals and process graduations
        {
            let coord_phi = self.coherence_bridge.smoothed_coherence() as f64;
            let coord_coherence = coherence as f64;
            self.memory_coordinator
                .update_signals_with_sigma(coord_phi, coord_coherence, self.carryover.consciousness.last_sigma);

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

        module_timings.episodic_replay = _t.elapsed().as_micros() as u64;

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
        // DREAM ENGINE: Record surprise events + dream during Cruise
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // 1. Every cycle: record high-surprise events for later dreaming.
        // 2. During Cruise urgency: run a dream cycle to discover better actions.
        // 3. Apply accumulated wisdom to bias exploration toward Phi-optimal choices.
        let mut dream_insights: usize = 0;
        let mut dream_phi_improvement: f32 = 0.0;
        let mut dream_wisdom_count: usize = 0;
        if let Some(ref mut dream) = self.dream_engine {
            // Record: use compressed state as "state", output as "action",
            // and prediction as "outcome" — these align with the dream API dimensions
            let dream_state: Vec<f32> = compressed_state.iter().take(64).copied().collect();
            let dream_action: Vec<f32> = output.iter().take(32).copied().collect();
            let dream_outcome: Vec<f32> = prediction.iter().take(64).copied().collect();
            // Weight surprise by consciousness level and narrative self-coherence:
            // Science: Tononi (2015) — consciousness = integrated information = memory salience
            // Narrative→Dream coupling (Conway 2005): self-relevant memories encode preferentially.
            let narrative_salience = self
                .narrative_self
                .as_ref()
                .map(|n| 1.0 + n.self_phi() as f32 * 0.5) // 1.0 to 1.5x boost
                .unwrap_or(1.0);
            let phi_weighted_surprise =
                prediction_error * (1.0 + unified_psi as f32).clamp(1.0, 2.0) * narrative_salience;
            dream.record(
                &dream_state,
                &dream_action,
                &dream_outcome,
                phi_weighted_surprise,
            );

            // Dream during Cruise urgency (low-error steady state) or every 20th cycle
            if matches!(urgency, super::CycleUrgency::Cruise)
                || urgency.should_run(self.stats.total_cycles, 10, 20, 5)
            {
                if let Ok(result) = dream.dream() {
                    dream_insights = result.insights;
                    dream_phi_improvement = result.best_phi_improvement;

                    if result.insights > 0 {
                        tracing::debug!(
                            insights = result.insights,
                            phi_improvement = result.best_phi_improvement,
                            simulations = result.simulations_run,
                            cycle = self.stats.total_cycles,
                            "Dream replay generated insights"
                        );

                        // Dream→Narrative coupling: dream insights feed narrative self-model.
                        // Science: Revonsuo (2000) — dreaming enhances threat simulation
                        // and narrative integration of novel experiences.
                        if let Some(ref mut narrative) = self.narrative_self {
                            narrative.process_experience(
                                &hv16_cached,
                                &format!("dream_insight_{}", result.insights),
                                true, // counterfactual-validated
                                unified_psi,
                                result.best_phi_improvement as f64,
                            );
                        }
                    }
                }
            }

            dream_wisdom_count = dream.wisdom().len();

            // Feed dream wisdom into DreamFeedbackBridge for context-aware priors.
            // Bridge converts Wisdom → action priors + confidence adjustments keyed
            // by context hash, enabling future cycles to leverage dream discoveries.
            #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
            for wisdom in dream.wisdom().iter() {
                let context_hash =
                    crate::consciousness::recursive_improvement::hash_context(&wisdom.context_state);
                let insight =
                    crate::consciousness::recursive_improvement::DreamInsight::new(
                        context_hash,
                        wisdom.context_state.clone(),    // original action = context state
                        wisdom.better_action.clone(),     // alternative action
                        wisdom.phi_improvement as f64,
                    );
                self.dream_feedback_bridge.process_insight(insight);
            }

            // Apply wisdom: if we have accumulated wisdom, modulate exploration
            // toward states where dream counterfactuals found Phi improvements
            if !dream.wisdom().is_empty() {
                let avg_phi_improvement: f32 = dream
                    .wisdom()
                    .iter()
                    .map(|w| w.phi_improvement)
                    .sum::<f32>()
                    / dream.wisdom().len() as f32;
                // Dream wisdom boosts exploration when Phi improvements are found
                let wisdom_exploration_boost = (avg_phi_improvement * 0.5).clamp(0.0, 0.2);
                self.curiosity_drive.exploration_urge = (self.curiosity_drive.exploration_urge
                    + wisdom_exploration_boost)
                    .clamp(0.0, 1.0);

                // FEEDBACK: Dream Phi insights feed forward into waking prediction confidence
                // Science: Prospective consciousness — offline simulation prepares waking cognition.
                // Dream-discovered Phi improvements signal that exploration can yield better states,
                // boosting confidence that the system can navigate toward them.
                if avg_phi_improvement > 0.01 {
                    let dream_confidence_boost =
                        (avg_phi_improvement * 0.1).min(0.05);
                    self.prediction_confidence =
                        (self.prediction_confidence + dream_confidence_boost).clamp(0.0, 1.0);
                }
            }
        }

        // FEEDBACK: Current-cycle dream insights boost learning signal
        // (reinforces pathways that produced the insight)
        if dream_phi_improvement > 0.05 {
            self.fep_learning_signal *= 1.0 + (dream_phi_improvement * 0.2).min(0.15);
        }

        // Dream feedback bridge: adjust prediction confidence based on accumulated
        // dream priors. Context hash from compressed state enables context-specific
        // calibration — contexts where dreams found better alternatives get a boost.
        #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
        if self.dream_feedback_bridge.num_priors() > 0 {
            let context_hash = crate::consciousness::recursive_improvement::hash_context(
                &compressed_state[..64.min(compressed_state.len())],
            );
            let (adjusted, was_informed) = self
                .dream_feedback_bridge
                .adjust_confidence(self.prediction_confidence as f64, context_hash);
            if was_informed {
                self.prediction_confidence = (adjusted as f32).clamp(0.0, 1.0);
            }
            // Decay priors every 199 cycles to forget stale wisdom (co-prime)
            if self.stats.total_cycles % 199 == 0 {
                self.dream_feedback_bridge.decay_priors(0.95);
            }
        }

        module_timings.dream_replay = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREFRONTAL CORTEX: Executive control and working memory gating
        // Amortized: Critical=every cycle, Normal=every 2nd, Cruise=every 5th
        // PFC inhibition is a stable control policy (Miller & Cohen 2001);
        // caching the veto for 5 cycles (~100ms at 50Hz) is within the
        // temporal binding window for executive control.
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let prefrontal_veto = if urgency.should_run(self.stats.total_cycles, 1, 2, 5) {
            if let Some(ref mut pfc) = self.prefrontal {
                // Add current input as a working memory item
                let wm_item = crate::brain::prefrontal::WorkingMemoryItem::new(
                    format!("cycle_{}", self.stats.total_cycles),
                    symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(compressed_state.clone()),
                );
                pfc.add_to_memory(wm_item);

                // Advance time (decay activations, evict expired items)
                pfc.tick();

                // Check memory utilization — high utilization triggers inhibition
                let utilization = pfc.memory_contents().len() as f32 / 7.0; // default capacity
                let veto = utilization > self.config.learning_threshold.max(0.8);

                if veto {
                    tracing::debug!(
                        utilization,
                        cycle = self.stats.total_cycles,
                        "Prefrontal veto: working memory overloaded"
                    );
                }

                // Graduate evicted items to episodic memory
                // Track 3h: WM eviction → resonator routing pipeline
                // Science: Frankland & Bontempi (2005) — systems consolidation routes
                // memories through hippocampal replay for categorical integration
                let graduates = pfc.drain_graduates();
                if !graduates.is_empty() {
                    for grad in &graduates {
                        // Route graduate through resonator for importance scoring
                        let grad_importance = if let Some(ref mut res_mem) = self.resonator_memory {
                            let res_dim = res_mem.resonator.config.dim;
                            let grad_vals = &grad.embedding.values;
                            if grad_vals.len() >= res_dim && !res_mem.episodes.is_empty() {
                                // Project to resonator dim and find best episode match
                                let projected: Vec<f32> =
                                    grad_vals.iter().take(res_dim).copied().collect();
                                let best_sim = res_mem.episodes.iter()
                                    .map(|ep| {
                                        ep.hv.iter().zip(projected.iter())
                                            .map(|(a, b)| a * b).sum::<f32>()
                                            / (ep.hv.iter().map(|x| x * x).sum::<f32>().sqrt()
                                               * projected.iter().map(|x| x * x).sum::<f32>().sqrt())
                                                .max(1e-8)
                                    })
                                    .fold(0.0f32, f32::max);
                                // High resonator match → boost importance (consolidation-worthy)
                                // Low match → novel content, still store but with base importance
                                pp_phi + best_sim * 0.2
                            } else {
                                pp_phi
                            }
                        } else {
                            pp_phi
                        };
                        self.episodic_memory.encode(
                            &grad.id,
                            grad.embedding
                                .values
                                .iter()
                                .take(64)
                                .copied()
                                .collect::<Vec<_>>(),
                            0.0,
                            grad_importance,
                            self.stats.total_cycles,
                        );
                    }
                    tracing::debug!(
                        count = graduates.len(),
                        "Prefrontal graduated items to episodic memory (resonator-routed)"
                    );
                }

                self.carryover.quality.cached_prefrontal_veto = veto;
                veto
            } else {
                false
            }
        } else {
            // Reuse cached veto from last computed cycle
            self.carryover.quality.cached_prefrontal_veto
        };

        // FEEDBACK: Prefrontal veto suppresses exploration (executive control)
        // Science: Miller & Cohen (2001) — PFC inhibits impulsive exploration when WM overloaded
        if prefrontal_veto {
            self.curiosity_drive.exploration_urge = 0.0;

            // FEEDBACK: WM overload triggers emergency consolidation (Baddeley 2000)
            // Science: Working memory overflow should push items to long-term storage,
            // not just block exploration. Force episodic consolidation to free WM slots.
            self.episodic_memory.consolidate_recent();
        }

        // FEEDBACK: Dual-veto freeze detection and recovery (Fuchs 2008 multistability)
        // Science: When reasoning gate AND prefrontal veto both fire, system is paralyzed:
        // exploration=0, learning=0. Soften both to allow partial recovery.
        if reasoning_gate_blocked && prefrontal_veto {
            self.curiosity_drive.exploration_urge = 0.3;
            self.fep_lr_boost = self.fep_lr_boost.max(1.0); // enforce fep_lr_boost >= 1.0 invariant
            tracing::debug!(
                cycle = self.stats.total_cycles,
                "Dual-veto freeze detected: softening both gates for recovery"
            );
        }

        module_timings.prefrontal = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // META-COGNITION: Recursive self-modeling and learning rate modulation
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (meta_cognitive_accuracy, meta_cognitive_depth) =
            if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut meta) = self.meta_cognition {
                    meta.update_self_model(prediction_error);
                    meta.deepen_recursion();
                    let accuracy = meta.accuracy();
                    let depth = meta.depth();
                    if accuracy > 0.7 {
                        let boost = 1.0 + (accuracy - 0.7) * 0.5; // up to 1.15x
                        self.carryover.learning.subsystem_lr_factor *= boost;
                    }
                    (accuracy, depth)
                } else {
                    (0.0, 0)
                }
            } else {
                // Read cached accuracy/depth without updating (avoid 0.0 in telemetry on skip)
                self.meta_cognition.as_ref().map(|m| (m.accuracy(), m.depth())).unwrap_or((0.0, 0))
            };

        module_timings.meta_cognition = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // VIRTUAL BODY: Map cognitive signals to interoceptive states
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (body_psi_modulation, body_valence, body_arousal) =
            if urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut body) = self.virtual_body {
                    let signals = super::virtual_body::CognitiveSignals {
                        prediction_error,
                        coherence,
                        prediction_confidence: self.prediction_confidence,
                        unified_psi,
                        flow_intensity: self.flow_state.intensity,
                        in_flow: self.flow_state.in_flow,
                        curiosity_boredom: self.curiosity_drive.boredom,
                        fep_learning_signal: self.fep_learning_signal,
                        error_trend: self.stats.error_trend,
                        cycles_per_second: self.stats.cycles_per_second,
                        target_frequency: self.config.target_frequency,
                    };
                    let state = body.update(&signals);
                    self.carryover.consciousness.body_phi_modulation = state.phi_modulation;
                    self.carryover.history.body_arousal = state.arousal;
                    (state.phi_modulation, state.valence, state.arousal)
                } else {
                    (1.0, 0.0, 0.0)
                }
            } else {
                // Urgency-skipped: use carryover for phi_modulation and arousal; valence has no
                // carryover so use neutral 0.0 (lightweight — doesn't trigger somatic marker feedback).
                (self.carryover.consciousness.body_phi_modulation, 0.0, self.carryover.history.body_arousal)
            };

        module_timings.virtual_body = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Body valence modulates prediction confidence (Damasio somatic markers)
        // Science: Damasio (1999) — positive somatic state boosts cognitive coherence;
        // negative somatic state signals danger → dampen confidence
        if body_valence > 0.3 {
            self.prediction_confidence =
                (self.prediction_confidence + body_valence * 0.02).clamp(0.0, 1.0);
        } else if body_valence < -0.3 {
            self.prediction_confidence =
                (self.prediction_confidence + body_valence * 0.03).clamp(0.0, 1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // AFFECTIVE BRIDGE: Evaluate somatic markers from cognitive signals
        // Runs every cycle (lightweight: ~5 arithmetic ops + blend)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (affective_valence, affective_arousal) =
            if let Some(ref mut bridge) = self.affective_bridge {
                let moral_score = self
                    .last_moral_judgment
                    .as_ref()
                    .map(|j| j.moral_score)
                    .unwrap_or(0.0);
                // Social modulation: feed ToM signals into affect (Decety & Chaminade 2003)
                // Social trust/cooperation injected by Mind module via set_social_signals()
                let affect = bridge.evaluate_from_signals_with_social(
                    prediction_error,
                    surprise_triggered,
                    unified_psi,
                    moral_score,
                    self.social_trust,
                    self.social_cooperation_rate,
                    0.0, // peer_valence: future — aggregate from social inbox
                );
                (affect.valence, affect.arousal)
            } else {
                (0.0, 0.5)
            };

        // FEEDBACK: Positive affect broadens exploration (Fredrickson 2001 broaden-and-build)
        if affective_valence > 0.2 && self.affective_bridge.is_some() {
            self.curiosity_drive.boredom *= 1.05;
        }
        // FEEDBACK: Arousal gates learning consolidation (Russell 1980 VAD model)
        // Science: Steriade (1996) — high arousal (fight-or-flight) suppresses consolidation;
        // low arousal (rest) enhances memory consolidation (REM/slow-wave effect)
        if affective_arousal > 0.7 {
            let arousal_suppress = ((affective_arousal - 0.7) * 0.5).min(0.15);
            self.fep_lr_boost = (self.fep_lr_boost * (1.0 - arousal_suppress)).max(1.0);

            // Arousal trap detection (Yerkes-Dodson 1908 — inverted-U performance curve)
            // Science: Prolonged high arousal suppresses LR → error stays high → arousal stays
            // high → positive feedback trap. After 10 stuck cycles, force exploration escape.
            if affective_arousal > 0.8 {
                self.carryover.urgency.arousal_trap_counter = self.carryover.urgency.arousal_trap_counter.saturating_add(1);
            }
            if self.carryover.urgency.arousal_trap_counter > 10 {
                self.curiosity_drive.exploration_urge = 1.0; // forced escape attempt
                self.prediction_confidence *= 0.9; // reset confidence to allow re-learning
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
                let consolidation_boost = ((0.3 - affective_arousal) * 0.3).min(0.1);
                self.fep_lr_boost =
                    (self.fep_lr_boost * (1.0 + consolidation_boost)).clamp(1.0, 2.0);
            }
        }
        module_timings.affective_bridge = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // USER STATE INFERENCE: Infer cognitive load, frustration, engagement
        // Runs every cycle (lightweight: keyword detection + rolling averages)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut usi) = self.user_state {
            let had_error = prediction_error > 0.8;
            usi.process(input, had_error);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE SELF: Process experience and track self-Φ
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let narrative_self_psi = if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut narrative) = self.narrative_self {
                let significance = if moral_concern_detected {
                    0.8
                } else {
                    (prediction_error as f64).clamp(0.0, 1.0)
                };
                narrative.process_experience(
                    &hv16_cached,
                    input,
                    prediction_error < self.config.learning_threshold,
                    coherence as f64,
                    significance,
                );
                narrative.self_phi()
            } else {
                0.0
            }
        } else {
            // Read cached self_phi without processing (avoid 0.0 triggering weak-identity feedback)
            self.narrative_self.as_ref().map(|n| n.self_phi()).unwrap_or(0.0)
        };

        // FEEDBACK: Narrative self-Phi modulates prediction confidence (identity coherence)
        // Science: Gallagher (2000) — strong narrative identity stabilizes learning
        if narrative_self_psi > 0.5 {
            self.prediction_confidence = (self.prediction_confidence * 1.02).clamp(0.0, 1.0);
        } else if narrative_self_psi > 0.0 && narrative_self_psi < 0.2 {
            self.prediction_confidence = (self.prediction_confidence * 0.95).clamp(0.0, 1.0);
        }

        // FEEDBACK: Narrative self-Phi modulates moral sensitivity (Gallagher & Hutto 2007)
        // Science: Strong narrative identity constrains moral reasoning (values are stable);
        // weak/incoherent identity amplifies moral sensitivity (recalibration needed)
        if narrative_self_psi > 0.7 {
            // High self-coherence → stabilize moral score (dampen fluctuations)
            // Multiply moral learning signal toward 1.0 (neutral)
            self.fep_learning_signal *= 1.0 + (narrative_self_psi as f32 - 0.7) * 0.1;
        } else if narrative_self_psi > 0.0 && narrative_self_psi < 0.2 {
            // Low self-coherence → amplify moral concern sensitivity
            self.adaptive_behavior.attention_sensitivity *= 1.0 + (0.2 - narrative_self_psi as f32) * 0.15;
        }

        module_timings.narrative_self = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE PROCESSING: Hierarchical predictive coding + precision
        // Runs every cycle (lightweight: BinaryHV → prediction → free energy)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (predictive_free_energy, predictive_psi_modulation) = if let Some(ref mut mind) =
            self.predictive_mind
        {
            if self.affective_bridge.is_some() {
                mind.precision
                    .apply_affective_modulation(affective_arousal as f64, affective_valence as f64);
            }
            let state = mind.process(&hv16_cached);
            self.carryover.consciousness.predictive_phi_modulation = state.phi_modulation;
            (state.free_energy, state.phi_modulation)
        } else {
            (0.0, 1.0)
        };

        // FEEDBACK: Predictive phi modulation gates plasticity (Friston 2010)
        if predictive_psi_modulation > 1.0 {
            let boost = ((predictive_psi_modulation - 1.0) * 0.1) as f32; // up to +10%
            self.carryover.learning.subsystem_lr_factor *= 1.0 + boost;
        } else if predictive_psi_modulation < 0.8 {
            self.carryover.learning.subsystem_lr_factor *= 0.9; // reduce LR when free energy is low
        }
        module_timings.predictive_processing = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // HIERARCHICAL FREE ENERGY: Multi-level variational decomposition
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // Science: Friston (2008) — hierarchical predictive processing
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let hierarchical_total_free_energy = if urgency.should_run(self.stats.total_cycles, 1, 2, 4)
        {
            if let Some(ref mut hfe) = self.hierarchical_free_energy {
                // FEEDBACK: Phi→precision coupling — higher integrated information
                // sharpens lower-level precision (Feldman & Friston 2010, §7.4).
                // This creates a causal mechanism: consciousness improves perceptual accuracy.
                let psi_boost = (unified_psi * 0.5).clamp(0.0, 0.5);
                let base_decay = hfe.config.precision_decay;
                for level in &mut hfe.levels {
                    let base_precision = base_decay.powi(level.level as i32);
                    level.precision = base_precision * (1.0 + psi_boost);
                }

                // Build observation from compressed state (clamped to state_dim)
                let obs: Vec<f64> = compressed_state
                    .iter()
                    .take(hfe.config.state_dim)
                    .map(|&x| x as f64)
                    .collect();
                hfe.update_beliefs(&obs);
                hfe.total_free_energy()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // FEEDBACK: High hierarchical free energy suppresses exploration AND boosts learning
        // Science: Friston (2008) — poor model → focus on learning, not exploring
        if hierarchical_total_free_energy > 1.0 {
            let fe_factor = (1.0 / (1.0 + hierarchical_total_free_energy * 0.1)) as f32;
            self.curiosity_drive.boredom *= fe_factor; // suppress exploration urge
            // Boost LR proportional to free energy (poor model → learn harder)
            // Capped at +30% to prevent instability
            let hfe_lr_boost = (1.0 + (hierarchical_total_free_energy * 0.05).min(0.3)) as f32;
            self.fep_lr_boost = (self.fep_lr_boost * hfe_lr_boost).clamp(1.0, 2.0);
        }

        module_timings.hierarchical_free_energy = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PREDICTIVE SELF: Evaluate action safety via self-state prediction
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let predictive_self_safety = if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut pred_self) = self.predictive_self {
                if let Some(ref narrative) = self.narrative_self {
                    pred_self.observe(narrative);
                }
                pred_self.learn_from_outcome_raw(unified_psi, coherence as f64);
                pred_self.confidence() as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        // FEEDBACK: Low self-model confidence reduces learning rate (precision-weighting)
        // Science: Clark (2013) — low precision on self-model predictions should reduce LR
        if predictive_self_safety > 0.0 && predictive_self_safety < 0.4 {
            let safety_factor = 0.85 + predictive_self_safety * 0.375; // 0.85-1.0
            self.carryover.learning.subsystem_lr_factor *= safety_factor;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // ATTENTION SCHEMA: Track attention state and generate control signals
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let attention_schema_focus = if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut schema) = self.attention_schema {
                let salience = prediction_error.max(0.1);
                let update = schema.update(hv16_cached, salience);
                let gain = if update.control_signal > 0.3 {
                    ((update.control_signal - 0.3) * 0.6).min(0.3)
                } else if update.control_signal < 0.2 {
                    -0.1
                } else {
                    0.0
                };
                self.adaptive_behavior.attention_sensitivity *= 1.0 + gain;
                update.new_intensity
            } else {
                0.0
            }
        } else {
            0.0
        };

        module_timings.attention_schema = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Attention focus gates novelty-seeking vs focus-locking (Baars 1988 GWT)
        // Science: Low focus → attention is scattered, force novelty-seeking to re-engage;
        // high focus → deep attention, suppress context-switching to maintain flow
        if attention_schema_focus > 0.0 {
            if attention_schema_focus < 0.3 {
                let novelty_push = ((0.3 - attention_schema_focus) * 0.12) as f32;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + novelty_push).clamp(0.0, 1.0);
            } else if attention_schema_focus > 0.8 {
                let focus_lock = ((attention_schema_focus - 0.8) * 0.15) as f32;
                self.adaptive_behavior.exploration_factor *= (1.0 - focus_lock).max(0.7);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHI ATTENTION: Adaptive Φ-weighted attention routing
        // Observes current Phi and gates expensive actions by consciousness level.
        // Science: Dehaene (2014) — conscious access enables flexible routing
        // ═══════════════════════════════════════════════════════════════════════
        let psi_attention_avg = if let Some(ref mut phi_attn) = self.phi_attention {
            phi_attn.observe(unified_psi as f32);
            // Gate: only allow state-modifying actions when Phi is sufficient
            if !phi_attn.allows_action(
                crate::consciousness::phi_attention::ActionType::StateModifying,
                unified_psi as f32,
            ) {
                // Low consciousness → reduce exploration (don't take risky actions unconsciously)
                self.curiosity_drive.exploration_urge *= 0.7;
            }
            phi_attn.phi_average().unwrap_or(0.0)
        } else {
            0.0
        };

        // Attention visualization: record snapshot for debugging/introspection
        if let Some(ref mut viz) = self.attention_visualizer {
            let snapshot = crate::visualization::AttentionSnapshot::new(
                vec!["psi".into(), "coherence".into(), "body".into(), "attention".into()],
                vec![unified_psi, coherence as f64, body_psi_modulation, psi_attention_avg as f64],
                vec![
                    unified_psi as f32,
                    coherence,
                    body_psi_modulation as f32,
                    psi_attention_avg,
                ],
                1.0,
            );
            viz.record(snapshot);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // GWT INTEGRATION: Submit encoding to global workspace for broadcast
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let gwt_broadcast = if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
            if let Some(ref mut gwt) = self.gwt {
                let activation = (1.0 - prediction_error as f64).clamp(0.0, 1.0);
                gwt.submit_strategy(
                    "cognitive_loop",
                    activation,
                    vec![hv16_cached],
                    vec!["encoder".to_string()],
                );
                let result = gwt.process();
                result.broadcast_occurred
            } else {
                false
            }
        } else {
            false
        };

        // FEEDBACK: GWT broadcast boosts confidence (conscious access moment)
        // Science: Baars (1988) — broadcast = conscious access, should amplify integration
        if gwt_broadcast {
            self.prediction_confidence = (self.prediction_confidence + GWT_BROADCAST_CONFIDENCE_BOOST).clamp(0.0, 1.0);
        }

        module_timings.gwt = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CROSS-MODAL BINDING: Bind HDC encodings across modalities
        // Runs every cycle (lightweight: 2 HV ops + similarity)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (cross_modal_binding_strength, cross_modal_psi) =
            self.update_cross_modal_binding(&hv16_cached, affective_valence, predictive_free_energy);

        module_timings.cross_modal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS MONITORS: Resonance + Quantum coherence
        // Urgency-gated: skip in Cruise mode
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Pre-compute to avoid borrow conflict with mutable subsystem references below
        let wm_utilization = self.prefrontal_utilization();
        let resonance_frequency = if urgency.run_consciousness_monitors() {
            if let Some(ref mut resonance) = self.consciousness_resonance {
                let dims = [
                    unified_psi,
                    coherence as f64,
                    wm_utilization,
                    self.adaptive_behavior.attention_sensitivity as f64,
                    (self.stats.total_cycles.min(100) as f64 / 100.0),
                    body_psi_modulation,
                    self.prediction_confidence as f64,
                ];
                let state = resonance.analyze(dims);
                state.dominant_frequency
            } else {
                0.0
            }
        } else {
            self.carryover.history.resonance_frequency // use previous cycle's value instead of 0.0
        };

        // FEEDBACK: Resonance frequency modulates attention sensitivity (Engel 2001)
        // Stable resonance near 0.5 → sharp attention; deviant frequency → diffuse
        if resonance_frequency > 0.0 {
            let resonance_quality = 1.0 - (resonance_frequency - 0.5).abs() * 2.0; // peaks at 0.5
            let attention_mod = 1.0 + (resonance_quality as f32 - 0.5) * 0.1; // ±5%
            self.adaptive_behavior.attention_sensitivity *= attention_mod;
        }

        let quantum_coherence_level = if urgency.run_consciousness_monitors() {
            if let Some(ref mut qc) = self.quantum_coherence {
                qc.observe(&hv16_cached, unified_psi);
                qc.coherence()
            } else {
                0.0
            }
        } else {
            self.carryover.consciousness.quantum_coherence // use previous cycle's value instead of 0.0
        };

        // FEEDBACK: Quantum coherence modulates prediction confidence (Penrose & Hameroff 2014)
        // High coherence → quantum-enhanced processing → slightly boost confidence
        // Decoherence → noisy processing → reduce confidence
        if quantum_coherence_level > 0.6 {
            let qc_boost = (quantum_coherence_level - 0.6) as f32 * 0.05; // up to +2%
            self.prediction_confidence = (self.prediction_confidence + qc_boost).clamp(0.0, 1.0);
        } else if quantum_coherence_level > 0.0 && quantum_coherence_level < 0.2 {
            self.prediction_confidence *= 0.98; // slight reduction during decoherence
        }

        module_timings.consciousness_resonance = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // PHENOMENAL BINDING: Temporal synchronization across consciousness dims
        // Urgency-gated: same as consciousness monitors (skip in Cruise)
        // Science: Singer & Gray (1989) — temporal binding hypothesis
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (phenomenal_binding_strength, phenomenal_fragmented) =
            if urgency.run_consciousness_monitors() {
                if let Some(ref mut binding) = self.phenomenal_binding {
                    let dims = [
                        unified_psi,
                        coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        body_psi_modulation,
                        self.prediction_confidence as f64,
                    ];
                    binding.observe_all(&dims);
                    let strength = binding.phenomenal_binding_strength();
                    let fragmented = binding.detect_fragmentation().is_some();
                    (strength, fragmented)
                } else {
                    (0.0, false)
                }
            } else {
                (0.0, false)
            };

        // FEEDBACK: Fragmentation suppresses exploration (Singer 1989)
        // When consciousness is fragmented, focus on integration not exploration
        if phenomenal_fragmented {
            self.curiosity_drive.boredom *= 0.8;
            self.adaptive_behavior.exploration_factor *= 0.7;
        }

        // FEEDBACK: High binding strength (flow) boosts learning rate (Csikszentmihalyi 1990)
        if phenomenal_binding_strength > 0.8 {
            let binding_boost = ((phenomenal_binding_strength - 0.8) * 0.2) as f32; // up to +4%
            self.carryover.learning.subsystem_lr_factor *= 1.0 + binding_boost;
        }

        // FEEDBACK: Binding strength gates WM access via attention sensitivity (Tononi 2015 IIT)
        // Science: High integrated information → more can be held in working memory;
        // low binding → restrict input (WM fragmented, accept less)
        if phenomenal_binding_strength > 0.7 {
            let wm_boost = ((phenomenal_binding_strength - 0.7) * 0.1) as f32; // up to +3%
            self.adaptive_behavior.attention_sensitivity *= 1.0 + wm_boost;
        } else if phenomenal_binding_strength > 0.0 && phenomenal_binding_strength < 0.4 {
            let wm_restrict = ((0.4 - phenomenal_binding_strength) * 0.08) as f32;
            self.adaptive_behavior.attention_sensitivity *= (1.0 - wm_restrict).max(0.8);
        }

        module_timings.phenomenal_binding = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // TEMPORAL CONSCIOUSNESS: Track Phi trajectory, continuity, identity
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (temporal_coherence_score, temporal_discontinuity) =
            if urgency.run_consciousness_monitors() {
                if let Some(ref mut temporal) = self.temporal_consciousness {
                    temporal.observe(
                        &hv16_cached,
                        unified_psi,
                        self.narrative_self.as_ref(),
                        self.predictive_self.as_ref(),
                    );
                    let coherence = temporal.overall_temporal_coherence();
                    let healthy = temporal.is_temporally_healthy();
                    (coherence, !healthy)
                } else {
                    (0.0, false)
                }
            } else {
                (0.0, false)
            };

        // FEEDBACK: Temporal discontinuity resets adaptation (context shift re-calibration)
        // Science: Varela (1999) — temporal discontinuities require re-orientation
        if temporal_discontinuity {
            self.fep_lr_boost = 1.0;
            self.prediction_confidence *= 0.8;
            // Lower learning threshold to learn more aggressively after discontinuity
            self.carryover.learning.adaptive_threshold_scale =
                (self.carryover.learning.adaptive_threshold_scale * 0.8).clamp(0.6, 1.5);
        } else if temporal_coherence_score > 0.8 {
            // High temporal coherence → model is reliable, raise threshold (learn less often)
            self.carryover.learning.adaptive_threshold_scale =
                (self.carryover.learning.adaptive_threshold_scale * 1.01).clamp(0.6, 1.5);
        } else {
            // Slowly return toward baseline
            self.carryover.learning.adaptive_threshold_scale += (1.0 - self.carryover.learning.adaptive_threshold_scale) * 0.02;
        }

        // FEEDBACK: High temporal coherence strengthens narrative self engagement
        // Science: Damasio (2010) — temporal continuity is the substrate of selfhood
        if temporal_coherence_score > 0.6 {
            if let Some(ref mut narrative) = self.narrative_self {
                let continuity_boost = (temporal_coherence_score - 0.6) * 0.1; // up to +4%
                narrative.boost_coherence(continuity_boost);
            }
        }

        // FEEDBACK: Temporal coherence ↔ attention mutual coupling (Engel et al. 2001)
        // Science: Temporal binding via phase synchrony — attention must gate synchronization.
        // Low temporal coherence → attention is fragmenting the time-axis → penalize sensitivity
        // to prevent amplification of incoherent states. High coherence → attention is stable.
        if temporal_coherence_score > 0.0 && temporal_coherence_score < 0.4 {
            let coherence_penalty = ((0.4 - temporal_coherence_score) * 0.1) as f32;
            self.adaptive_behavior.attention_sensitivity *= (1.0 - coherence_penalty).max(0.85);
        }

        module_timings.temporal_consciousness = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS THERMODYNAMICS: Phase transitions & free energy
        // Urgency-gated: same as consciousness monitors (skip in Cruise)
        // Science: Friston (2010) — free energy, Kelso — critical fluctuations
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (thermodynamic_entropy, thermodynamic_free_energy) =
            if urgency.run_consciousness_monitors() {
                if let Some(ref mut thermo) = self.consciousness_thermodynamics {
                    let dims = [
                        unified_psi,
                        coherence as f64,
                        wm_utilization,
                        self.adaptive_behavior.attention_sensitivity as f64,
                        (self.stats.total_cycles.min(100) as f64 / 100.0),
                        body_psi_modulation,
                        self.prediction_confidence as f64,
                    ];
                    let state = thermo.analyze(dims);
                    // FEEDBACK: Phase-dependent exploration modulation (Kelso 1995)
                    use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
                    match state.phase {
                        ConsciousnessPhase::Critical => {
                            // Edge of chaos — maximum creativity, boost exploration
                            self.curiosity_drive.boredom *= 1.1;
                            self.adaptive_behavior.exploration_factor *= 1.15;
                        }
                        ConsciousnessPhase::Flow => {
                            // Superfluid state — boost learning rate
                            self.carryover.learning.subsystem_lr_factor *= 1.05;
                        }
                        ConsciousnessPhase::Chaotic => {
                            // Fragmented — suppress exploration, seek stability
                            self.curiosity_drive.boredom *= 0.7;
                            self.adaptive_behavior.exploration_factor *= 0.5;
                        }
                        ConsciousnessPhase::Frozen => {
                            // Rigid — nudge toward exploration to unfreeze
                            self.curiosity_drive.boredom *= 1.05;
                        }
                        _ => {} // Normal, Unified — no modulation needed
                    }
                    (state.entropy, state.free_energy)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

        module_timings.consciousness_thermodynamics = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Thermodynamic entropy magnitude modulates exploration intensity
        // Science: Ulanowicz (2009) — entropy quantifies degrees of freedom in the system.
        // High entropy → system has many accessible states, exploration is cheap → boost;
        // Low entropy → system is ordered, consolidation is productive → dampen exploration.
        // This complements the phase-based modulation above with continuous magnitude scaling.
        if thermodynamic_entropy > 0.0 {
            if thermodynamic_entropy > 0.7 {
                let entropy_boost = ((thermodynamic_entropy - 0.7) * 0.1).min(0.1) as f32;
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + entropy_boost).clamp(0.0, 1.0);
            } else if thermodynamic_entropy < 0.3 {
                let consolidation_bias = ((0.3 - thermodynamic_entropy) * 0.08).min(0.08) as f32;
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 + consolidation_bias)).clamp(1.0, 2.0);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // EMBODIED COGNITION: Bridge virtual body to full body schema
        // Urgency-gated: Critical=always, Normal=always, Cruise=every 2nd
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (embodied_psi_modulation, embodied_agency) =
            if urgency.should_run(self.stats.total_cycles, 1, 1, 2) {
                if let Some(ref mut embodied) = self.embodied_cognition {
                    if let Some(ref body) = self.virtual_body {
                        embodied.update_interoception(body.interoceptive_state().clone());
                    }
                    let response = embodied.process();
                    self.carryover.consciousness.embodied_phi_modulation = response.phi_modulation;

                    // Wire embodied signals into cognitive loop:
                    // 1. Homeostatic deviation increases urgency (survival takes priority)
                    // Science: Damasio (1999) — somatic markers guide decision-making
                    if response.homeostatic_deviation > 0.5 {
                        self.carryover.urgency.consecutive_low_error = 0; // prevent Cruise when body is stressed
                    }
                    // 2. Sensorimotor surprise blends into exploration urge
                    // Science: Friston (2010) — interoceptive surprise drives active inference
                    if response.sensorimotor_surprise > 0.3 {
                        let body_nudge = (response.sensorimotor_surprise * 0.1).min(0.15) as f32;
                        self.curiosity_drive.exploration_urge =
                            (self.curiosity_drive.exploration_urge + body_nudge).clamp(0.0, 1.0);
                    }
                    // 3. High allostatic load suppresses learning (conserve resources)
                    // Science: McEwen (2004) — allostatic overload impairs plasticity
                    if response.allostatic_load > 0.7 {
                        self.fep_lr_boost = (self.fep_lr_boost
                            * (1.0 - (response.allostatic_load - 0.7) as f32 * 0.5))
                            .max(1.0);
                    }

                    (response.phi_modulation, response.sense_of_agency)
                } else {
                    (1.0, 0.0)
                }
            } else {
                // Urgency-skipped: use carryover for phi_modulation; agency has no carryover.
                (self.carryover.consciousness.embodied_phi_modulation, 0.0)
            };

        module_timings.embodied_cognition = _t.elapsed().as_micros() as u64;

        // FEEDBACK: Embodied agency modulates exploration risk tolerance
        // Science: Friston, Stephan et al. (2015) — sense of agency enables bold action
        // High agency → allow riskier exploration; low agency → cautious retreat
        if embodied_agency > 0.7 {
            let agency_boost = ((embodied_agency - 0.7) * 0.15) as f32; // up to +4.5%
            self.adaptive_behavior.exploration_factor *= 1.0 + agency_boost;
        } else if embodied_agency > 0.0 && embodied_agency < 0.3 {
            let caution = ((0.3 - embodied_agency) * 0.1) as f32; // up to -3%
            self.curiosity_drive.exploration_urge *= (1.0 - caution).max(0.7);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // NARRATIVE-GWT INTEGRATION: Consciousness governance capstone
        // Urgency-gated: Critical=always, Normal=every 2nd, Cruise=every 4th
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        let (narrative_gwt_veto, narrative_gwt_self_psi) =
            if urgency.should_run(self.stats.total_cycles, 1, 2, 4) {
                if let Some(ref mut ngwt) = self.narrative_gwt {
                    let activation = (1.0 - prediction_error as f64).clamp(0.0, 1.0);
                    let veto = ngwt.submit_content(
                        "cognitive_loop",
                        vec![hv16_cached],
                        input,
                        vec!["encoder".to_string(), "temporal".to_string()],
                        activation,
                    );
                    let vetoed = veto.map(|v| v.vetoed).unwrap_or(false);
                    let _result = ngwt.process();
                    (vetoed, ngwt.self_phi())
                } else {
                    (false, 0.0)
                }
            } else {
                (false, 0.0)
            };

        // Store narrative-GWT veto for next cycle's learning gate
        self.carryover.quality.narrative_veto_active = narrative_gwt_veto;
        module_timings.narrative_gwt = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // UNIFIED LIVING MIND: life-mind continuity (full_consciousness only)
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Integrates autopoietic self-maintenance, enactive sense-making, and
        // predictive processing into a unified vitality/coherence measure.
        #[cfg(feature = "full_consciousness")]
        let (living_mind_vitality, living_mind_coherence) = {
            // Update autopoietic self-maintenance with current consciousness signals
            self.autopoietic
                .update(unified_psi, coherence as f64, prediction_error as f64);

            // Map cognitive loop action to enactive ActionType based on adaptive behavior
            let enactive_action = match self.adaptive_behavior.action_hint {
                super::ActionHint::Explore => {
                    crate::consciousness::enactive_cognition::ActionType::Explore
                }
                super::ActionHint::SeekInput => {
                    crate::consciousness::enactive_cognition::ActionType::Observe
                }
                super::ActionHint::SlowDown => {
                    crate::consciousness::enactive_cognition::ActionType::Reflect
                }
                super::ActionHint::SpeedUp => {
                    crate::consciousness::enactive_cognition::ActionType::Execute
                }
                _ => crate::consciousness::enactive_cognition::ActionType::Observe,
            };

            // Build perception summary from current cycle signals
            let perception = crate::consciousness::enactive_cognition::PerceptionSummary {
                features: {
                    let mut f = std::collections::HashMap::new();
                    f.insert("prediction_error".into(), prediction_error as f64);
                    f.insert("coherence".into(), coherence as f64);
                    f.insert("phi".into(), unified_psi);
                    f
                },
                surprise: prediction_error as f64,
                affordances: encoding_result
                    .detected_primitives
                    .iter()
                    .take(3)
                    .cloned()
                    .collect(),
            };

            // Run enactive sense-making cycle
            let enacted_meaning = self.enactive.cycle(enactive_action, perception, input);

            // Wire enactive meaning into cognitive loop:
            // 1. High relevance boosts attention sensitivity (salient = attend more)
            // Science: Thompson (2007) — enacted meaning modulates attention
            if enacted_meaning.meaning.relevance > 0.6 {
                let relevance_gain = (enacted_meaning.meaning.relevance * 0.1).min(0.15) as f32;
                self.adaptive_behavior.attention_sensitivity *= 1.0 + relevance_gain;
            }
            // 2. Negative valence strengthens narrative veto tendency (caution)
            // Science: Colombetti (2014) — affect and enaction are inseparable
            if enacted_meaning.meaning.valence < -0.5 {
                self.prediction_confidence *= (1.0 + enacted_meaning.meaning.valence * 0.1) as f32;
                self.prediction_confidence = self.prediction_confidence.clamp(0.0, 1.0);
            }

            // Integrate all subsystems into unified living state
            let free_energy = self.fep_agent.current_free_energy();
            let unified_state = self.unified_living_mind.integrate(
                &self.autopoietic,
                &self.enactive,
                unified_psi,
                free_energy,
            );

            (unified_state.vitality, unified_state.coherence)
        };

        #[cfg(not(feature = "full_consciousness"))]
        let (living_mind_vitality, living_mind_coherence) = (0.0, 0.0);
        module_timings.living_mind = _t.elapsed().as_micros() as u64;

        // ═══════════════════════════════════════════════════════════════════════
        // MASTER CONSCIOUSNESS EQUATION: comprehensive consciousness metric
        // ═══════════════════════════════════════════════════════════════════════
        let _t = Instant::now();
        // Run every 10th cycle to amortize cost. Maps cognitive loop signals to
        // the 8-factor ConsciousnessInputs: Phi, Broadcast, WorkingMemory,
        // Attention, Recurrence, Embodiment, Knowledge, Synchrony.
        // Urgency-adaptive: Critical=every 5th, Normal=every 10th, Cruise=every 20th
        let consciousness_level = if urgency.should_run(self.stats.total_cycles, 5, 10, 20) {
            let inputs = crate::consciousness::master_consciousness_equation::ConsciousnessInputs {
                phi: unified_psi,
                broadcast: coherence as f64, // coherence ~ global workspace broadcast
                working_memory: self.prefrontal_utilization(),
                attention: encoding_result.peak_attention as f64,
                recurrence: (self.stats.total_cycles.min(100) as f64 / 100.0), // ramp up over 100 cycles
                embodiment: body_psi_modulation, // virtual body provides embodiment
                knowledge: self.prediction_confidence as f64,
                synchrony: (0.3 + self.flow_state.intensity as f64 * 0.7).clamp(0.1, 1.0),
            };
            let level = self.master_equation.compute(&inputs).consciousness_level;

            // Track consciousness level for learning gating (Task C)
            self.carryover.history.consciousness_level = level;

            // FEEDBACK: MCE consciousness level boosts learning rate (decaying)
            // Science: Dehaene (2014) — conscious access improves encoding
            if level > 0.0 {
                self.carryover.learning.mce_lr_boost = (level * MCE_LR_BOOST_SCALE as f64) as f32;
            } else {
                self.carryover.learning.mce_lr_boost *= MCE_BOOST_DECAY;
            }

            // FEEDBACK: Consciousness gates consolidation intensity (Dehaene 2014)
            // Science: Only conscious moments produce durable memories. Scale episodic
            // consolidation by consciousness level — low consciousness → skip storage,
            // high consciousness → prioritize memory encoding.
            if level > 0.5 {
                // Trigger demand-driven consolidation at high consciousness
                self.episodic_memory.consolidate_recent();
            }
            // Scale learning signal by consciousness quality (gradual, not on/off)
            // This complements the binary consciousness_awake gate with continuous modulation
            self.fep_learning_signal *= (0.5 + level as f32 * 0.5).clamp(0.5, 1.0);

            level
        } else {
            // Decay MCE LR boost between MCE firings
            self.carryover.learning.mce_lr_boost *= MCE_BOOST_DECAY;
            0.0
        };

        // Store resonance frequency and quantum coherence for next cycle's feedback
        self.carryover.history.resonance_frequency = resonance_frequency;
        self.carryover.consciousness.quantum_coherence = quantum_coherence_level;
        module_timings.master_consciousness_equation = _t.elapsed().as_micros() as u64;

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
            self.prediction_confidence = self
                .prediction_confidence
                .clamp(confidence_start - max_drift, confidence_start + max_drift)
                .clamp(0.0, 1.0);
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
            self.prediction_confidence *= (1.0 - boredom_dampen).max(0.85);
        }

        // Boredom homeostasis: slow drift toward neutral (0.5) prevents monotonic saturation.
        // Without this, boredom accumulates asymmetrically toward 0 or 1.
        self.curiosity_drive.boredom += (0.5 - self.curiosity_drive.boredom) * 0.02;

        // Exploration urge per-cycle budget: clamp total change to ±0.5.
        // 15+ subsystems write exploration_urge per cycle; without bounding, cumulative
        // nudges can pin it to 0.0 or 1.0. Science: Homeostatic control of exploration.
        self.curiosity_drive.exploration_urge = self
            .curiosity_drive
            .exploration_urge
            .clamp(
                (exploration_urge_start - 0.5).max(0.0),
                (exploration_urge_start + 0.5).min(1.0),
            );

        // Exploration urge homeostasis: slow drift toward neutral (0.3) prevents saturation.
        self.curiosity_drive.exploration_urge +=
            (0.3 - self.curiosity_drive.exploration_urge) * 0.03;

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
        let spectral_mip_phi = if self.stats.total_cycles % 47 == 0 {
            let result = self.spectral_mip_finder.compute();
            let phi = result.as_ref().map(|r| r.phi);
            if phi.is_some() {
                self.carryover.consciousness.last_spectral_mip_phi = phi;
                self.carryover.consciousness.last_sigma = phi; // backward compat for memory coordinator
            }
            // Adaptive dimension selection: every 94 cycles (every 2nd compute at 47-cycle cadence),
            // concentrate tracked dimensions near the MIP boundary for better
            // partition quality. Fiedler ordering identifies informative dims.
            if self.stats.total_cycles % 94 == 0 {
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
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 - sig_dampen)).max(1.0);
                self.prediction_confidence =
                    (self.prediction_confidence + sig_dampen * 0.5).clamp(0.0, 1.0);
            } else if sig < 0.2 {
                // Low integration → boost learning (model needs updating)
                let sig_boost = ((0.2 - sig) * 0.15).min(0.05) as f32;
                self.fep_lr_boost = (self.fep_lr_boost * (1.0 + sig_boost)).clamp(1.0, 2.0);
            }
        }

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
                embedding: encoding_hdv,
                value_alignment: moral_score,
                emotional_valence: self.emotion_contagion.valence,
                lessons: Vec::new(),
                timestamp: self.stats.total_cycles as u64,
            };
            soul.integrate_experience(experience);
        }
        module_timings.soul_experience = _t.elapsed().as_micros() as u64;

        // Pre-compute values and formatted strings to avoid expensive ops inside struct literal
        let value_trend = self.value_feedback.recent_trend(50);
        let circadian_phase_str = format!("{:?}", self.biorhythm.phase);
        let selected_strategy_str = format!("{:?}", selected_strategy);

        // Build cycle metadata for observability
        let _t = Instant::now();
        let metadata = super::CycleMetadata {
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
            selected_strategy: selected_strategy_str,
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
            hierarchical_mip_scales: self.carryover.consciousness.last_hierarchical_mip_phi
                .map(|_| 3usize).unwrap_or(0), // 3 scales: 32→64→128
            resonator_codebook_size: self.resonator_memory.as_ref()
                .and_then(|m| m.resonator.codebooks.first())
                .map(|cb| cb.len())
                .unwrap_or(0),
            resonator_episodes: self.resonator_memory.as_ref()
                .map(|m| m.len())
                .unwrap_or(0),
            resonator_factorization_iters: self.resonator_memory.as_ref()
                .map(|m| m.resonator.iterations())
                .unwrap_or(0),
            module_timings_us: {
                module_timings.metadata_assembly = _t.elapsed().as_micros() as u64;
                module_timings
            },
            circadian_phase: circadian_phase_str,
            circadian_plasticity: self.biorhythm.plasticity_mod as f32,
            phi_attention_weight,
            guiding_question,
            dominant_harmonic,
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
        };

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
        #[cfg(feature = "identity")]
        let signed_output = self.mfdi_bridge.sign_output(&output).ok();
        #[cfg(feature = "identity")]
        let assurance_level = self.mfdi_bridge.assurance_level();

        CycleResult {
            output,
            prediction_error,
            peak_attention: encoding_result.peak_attention,
            detected_primitives: encoding_result.detected_primitives,
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            metadata,
            #[cfg(feature = "identity")]
            signed_output,
            #[cfg(feature = "identity")]
            assurance_level,
        }
    }

    /// Safe wrapper around `cycle()` that catches panics from unexpected subsystem failures.
    ///
    /// Use this in production code paths where a panic must not propagate (e.g., actor loops,
    /// async bridges). Returns `Err` with the panic message if any subsystem panics during
    /// the cycle.
    pub fn try_cycle(&mut self, input: &str) -> Result<CycleResult, String> {
        // SAFETY: CognitiveLoopService is not UnwindSafe by default because it contains
        // mutable state. We use AssertUnwindSafe because a panic mid-cycle leaves the
        // service in a potentially inconsistent state, but callers should reset() after
        // an error rather than continuing.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.cycle(input)));
        result.map_err(format_panic_payload)
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

    #[test]
    fn test_panic_payload_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("subsystem failure");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: subsystem failure");
    }

    #[test]
    fn test_panic_payload_string() {
        let payload: Box<dyn std::any::Any + Send> =
            Box::new(String::from("HDC bridge overflow"));
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
}
