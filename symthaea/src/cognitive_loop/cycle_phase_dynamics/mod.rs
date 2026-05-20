// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core dynamics phase of the cognitive cycle.
//!
//! `phase_dynamics()` orchestrates Phases A–11, delegating to three private helpers:
//! - `phase_dynamics_memory_binding()` — episodic recall, resonator, binding, goals
//! - `phase_dynamics_cfc_planning()` — semantic memory, CfC step, prediction, world model
//! - `phase_dynamics_training()` — training dispatch, Broca, parallel post-processing
//!
//! Ordering is load-bearing — do not reorder sections or helper calls.
//!
//! ## Section index (phase_dynamics outline)
//!
//! | Section | Description |
//! |---------|-------------|
//! | Phase A: OBSERVE | Build immutable CycleSnapshot |
//! | Phase B: COMPUTE | Run subsystem managers via CognitiveSubsystem trait |
//! | Self-model | Accuracy tracking (EMA) |
//! | Foveation | Vision-manifold coupling (cfg) |
//! | **→ memory_binding()** | Episodic recall + resonator + binding + goals |
//! | 1b+15+18: Emotion | Contagion + homeostasis |
//! | 1c: Emotion | Unified emotional bridge (VAD) |
//! | **→ cfc_planning()** | Semantic memory + CfC step + prediction + world model |
//! | 8–10c: Experience | Create experience, coherence bridge, adaptive behavior |
//! | 10d: Active inference | FEP, MCTS, moral modulation, math solver |
//! | Neuromod + Psi | Neuromodulator bath, unified Psi synthesis |
//! | 10d.6b: Enhanced FEP | Enhanced FEP bridge + motor commands |
//! | Attention | Budget check + active rest + moral attractor |
//! | Reasoning engine | Phi-gated multi-tier reasoning (cfg) |
//! | Metacognition | Anomaly detection + recovery |
//! | LR composition | Compose final effective learning rate |
//! | **→ training()** | Training + Broca + parallel post-processing |
//! | Result | Assemble DynamicsPhaseResult |

use crate::consciousness::fep_active_inference::{MotorCommandType, Observation};
use ndarray::Array1;
#[cfg(feature = "parallel")]
use rayon::join as rayon_join;
use std::borrow::Cow;
use std::time::Instant;

use super::feedback_state::Priority;
use super::helpers;
use super::phase_results::{
    DynAttention, DynCore, DynFep, DynGuidance, DynHomeostasis, DynMath, DynNeuromod, DynReasoning,
    DynResonator, DynamicsPhaseResult, PerceptionPhaseResult,
};
#[cfg(feature = "cpg")]
use super::thresholds::CPG_SYNC_TAU_FLOOR;
use super::thresholds::{
    ACTION_OUTCOME_COUPLING_RESET_THRESHOLD,
    ALEATORIC_UNCERTAINTY_DEFAULT,
    AROUSAL_RECOVERY_TAU_SCALE,
    AROUSAL_TAU_DEADZONE,
    AROUSAL_TAU_SENSITIVITY,
    // Phase 1a: dynamics startup & miscellaneous
    AROUSAL_TRAP_RECOVERY_MIN_CYCLES,
    AROUSAL_TRAP_RECOVERY_RAMP_CYCLES,
    ATTENTION_BUDGET_US,
    ATTENTION_SENSITIVITY_BOOST_FACTOR,
    BINDING_CONFIDENCE_THRESHOLD,
    BINDING_LOW_THRESHOLD,
    BINDING_STRONG_CONFIDENCE_SCALE,
    BINDING_STRONG_RELIEF_SCALE,
    BINDING_WEAK_CAUTION_SCALE,
    BINDING_WEAK_CONFIDENCE_SCALE,
    BROCA_CONSCIOUSNESS_THRESHOLD_DECREASE,
    BROCA_CONSCIOUSNESS_THRESHOLD_INCREASE,
    BROCA_CONSCIOUSNESS_THRESHOLD_MAX,
    BROCA_CONSCIOUSNESS_THRESHOLD_MIN,
    BROCA_LOW_QUALITY_THRESHOLD,
    BROCA_QUALITY_COHERENCE_WEIGHT,
    BROCA_QUALITY_HIGH_THRESHOLD,
    BROCA_QUALITY_LONG_COHERENCE_WEIGHT,
    BROCA_QUALITY_PE_WEIGHT,
    CAUSAL_ATTENTION_CONFIDENCE_SCALE,
    CAUSAL_ATTENTION_STRENGTH_THRESHOLD,
    CAUSAL_CONFIDENCE_DENSE_THRESHOLD,
    CAUSAL_CONFIDENCE_MODERATE_THRESHOLD,
    CAUSAL_DENSE_CONFIDENCE_SCALE,
    CAUSAL_MODERATE_CONFIDENCE_SCALE,
    CODEBOOK_FAMILIAR_TAU_SCALE,
    CODEBOOK_FAMILIAR_THRESHOLD,
    CODEBOOK_NOVEL_TAU_SCALE,
    CODEBOOK_NOVEL_THRESHOLD,
    COHERENCE_CONFIDENCE_BOOST,
    COHERENCE_HIGH_THRESHOLD,
    COHERENCE_LOW_DAMPEN_SCALE,
    COHERENCE_LOW_THRESHOLD,
    COHERENCE_PREDICTION_EMA,
    COHERENCE_VELOCITY_BUDGET_CONTRACT,
    COHERENCE_VELOCITY_BUDGET_EXPAND,
    COHERENCE_VELOCITY_BUDGET_THRESHOLD,
    COHERENCE_VELOCITY_TAU_BOOST,
    COHERENCE_VELOCITY_TAU_DAMPEN,
    COHERENCE_VELOCITY_TAU_THRESHOLD,
    CONFIDENCE_CRASH_EXPLORATION_BOOST,
    CONFIDENCE_CRASH_FLOW_MULTIPLIER,
    CONFIDENCE_CRASH_FREEZE_CYCLES,
    CONFIDENCE_VELOCITY_BOOST_SCALE,
    CONFIDENCE_VELOCITY_DAMPEN_SCALE,
    CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD,
    CONFIDENCE_VELOCITY_POSITIVE_THRESHOLD,
    CONSCIOUSNESS_RESIZE_CENTER,
    CONSCIOUSNESS_RESIZE_SCALE,
    CPG_TAU_CLAMP_MAX,
    CPG_TAU_CLAMP_MIN,
    DOMINANCE_CONFIDENCE_THRESHOLD,
    DOMINANCE_CONFIDENT,
    DOMINANCE_DEFAULT,
    DOMINANCE_FLOW_BASE,
    DOMINANCE_FLOW_SCALE,
    DYNAMICS_POST_BOOT_CYCLES,
    DYNAMICS_STARTUP_WARMUP_CYCLES,
    // Round 22: magic number extraction
    EPISTEMIC_BUDGET_CONTRACT_BASE,
    EPISTEMIC_BUDGET_CONTRACT_RAMP,
    EPISTEMIC_BUDGET_CONTRACT_THRESHOLD,
    EPISTEMIC_BUDGET_EXPAND_CAP,
    EPISTEMIC_BUDGET_EXPAND_THRESHOLD,
    EPISTEMIC_EXPLORE_SCALE,
    EPISTEMIC_EXPLORE_THRESHOLD,
    EPISTEMIC_LOW_DAMPEN,
    EPISTEMIC_LOW_THRESHOLD,
    EPISTEMIC_OSCILLATION_MULTIPLIER,
    EPISTEMIC_OSCILLATION_THRESHOLD,
    EPISTEMIC_SEMANTIC_BOOST_SCALE,
    EPISTEMIC_SEMANTIC_BOOST_THRESHOLD,
    EPISTEMIC_SEMANTIC_CAUTION_BASE,
    EPISTEMIC_SEMANTIC_CAUTION_SCALE,
    EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD,
    EPISTEMIC_UNCERTAINTY_DEFAULT,
    ETHICS_CAUTION_CONFIDENCE_CAP,
    FEP_ACCURACY_CONFIDENCE_THRESHOLD,
    FEP_COMPLEXITY_LR_SCALE,
    FEP_COMPLEXITY_PENALTY_CAP,
    FEP_COMPLEXITY_THRESHOLD,
    FEP_EFFICIENT_EXPLORATION_DAMPEN,
    FEP_LEARNING_PLASTICITY_THRESHOLD,
    FEP_PRAGMATIC_EXPLOIT_SCALE,
    FEP_PRAGMATIC_EXPLOIT_THRESHOLD,
    FEP_PRAGMATIC_EXPLORE_SCALE,
    FEP_PRAGMATIC_EXPLORE_THRESHOLD,
    FEP_SURPRISE_EXPLORE_CAP,
    FEP_SURPRISE_EXPLORE_SCALE,
    FEP_SURPRISE_EXPLORE_SECONDARY_CAP,
    FEP_SURPRISE_EXPLORE_SECONDARY_SCALE,
    FEP_SURPRISE_TAU_SCALE,
    FEP_TD_ERROR_DISCOVERY_THRESHOLD,
    GOAL_DELTA_BASE_STEP,
    GOAL_DELTA_CONFIDENCE_SCALE,
    GOAL_PRIORITY_EXPLORATION_THRESHOLD,
    GOAL_PRIORITY_LR_THRESHOLD,
    HARMONY_INDEX_SACRED_STILLNESS,
    HOMEOSTASIS_AROUSAL_TARGET,
    HOMEOSTASIS_EFFICIENCY_EMA,
    HOMEOSTASIS_EFFICIENCY_HIGH,
    HOMEOSTASIS_EFFICIENCY_LOW,
    HOMEOSTASIS_EMOTIONAL_INERTIA,
    HOMEOSTASIS_NEUROMOD_STEP,
    HOMEOSTASIS_PULL_AROUSAL_SCALE,
    HOMEOSTASIS_PULL_CRITICAL,
    HOMEOSTASIS_PULL_INCREASE,
    HOMEOSTASIS_PULL_NORMAL,
    HOMEOSTASIS_PULL_REDUCTION,
    HOMEOSTASIS_PULL_VELOCITY_SCALE,
    HORIZON_PE_CONTRACT_RATE,
    HORIZON_PE_CONTRACT_THRESHOLD,
    HORIZON_PE_EXPAND_RATE,
    HORIZON_PE_EXPAND_THRESHOLD,
    HORIZON_SLOPE_CONTRACT_CAP,
    HORIZON_SLOPE_CONTRACT_RATE,
    HORIZON_SLOPE_EXPAND_CAP,
    HORIZON_SLOPE_EXPAND_RATE,
    HORIZON_SLOPE_THRESHOLD,
    INFERENCE_MODE_INIT_CONFIDENCE,
    KNOWLEDGE_ALERT_EXPLORE_CAP,
    KNOWLEDGE_ATTENTION_CONTRADICTION_BOOST,
    KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD,
    KNOWLEDGE_CAUSAL_DEPTH_DA_NUDGE,
    KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD,
    KNOWLEDGE_CONTRADICTION_FLOOR,
    KNOWLEDGE_CONTRADICTION_NE_BOOST,
    KNOWLEDGE_CONTRADICTION_SHT_BOOST,
    KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT,
    KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT,
    KNOWLEDGE_GROUNDING_SHT_NUDGE,
    KNOWLEDGE_NOVELTY_EXPLORE_SCALE,
    KNOWLEDGE_UNCERTAINTY_NE_SCALE,
    MCTS_CONSOLIDATE_CONFIDENCE_SCALE,
    MCTS_EFFECTIVENESS_CONFIDENCE_SCALE,
    MCTS_EFFECTIVENESS_EMA,
    MCTS_EFFECTIVENESS_EXPLORE_SCALE,
    MCTS_EFFECTIVENESS_HIGH,
    MCTS_EFFECTIVENESS_LOW,
    MCTS_EXPLOIT_LR_SCALE,
    MCTS_EXPLORE_SCALE,
    MCTS_PLAN_CONFIDENCE_THRESHOLD,
    MCTS_PLAN_WEIGHT_SCALE,
    MEMORY_RECALL_TOP_K,
    MOTOR_ADAPTIVE_LR_ALPHA,
    MOTOR_ADAPTIVE_LR_MAX,
    MOTOR_ADAPTIVE_LR_MIN,
    MOTOR_ADAPTIVE_LR_MOMENTUM,
    MOTOR_ATTENTION_SENSITIVITY_MAX,
    MOTOR_ATTENTION_SENSITIVITY_MIN,
    MOTOR_ATTENTION_SENSITIVITY_SCALE,
    // Round 17: Motor modulation, Broca quality, neuroevo, homeostasis
    MOTOR_ATTENTION_SHIFT_SCALE,
    MOTOR_EXPLORATION_BOOST_MAX,
    MOTOR_EXPLORATION_EPISTEMIC_THRESHOLD,
    MOTOR_EXPLORATION_INTENSITY_SCALE,
    MOTOR_FAILURE_OBSERVATION_VALUE,
    MOTOR_SUCCESS_OBSERVATION_VALUE,
    NEUROEVO_BLEND_DEFAULT_WEIGHT,
    NEUROEVO_BLEND_EVOLVED_WEIGHT,
    NEUROEVO_DEFAULT_TAU_BASE,
    NEUROEVO_TAU_CLAMP_MAX,
    NEUROEVO_TAU_CLAMP_MIN,
    NEUROMOD_DELTA_THRESHOLD,
    PE_VARIANCE_DAMPEN_SCALE,
    PE_VARIANCE_MAX_EFFECT,
    PE_VARIANCE_THRESHOLD,
    POLICY_FULL_AGREEMENT_BOOST,
    POLICY_MIN_WINDOW,
    POLICY_SOFT_THRESHOLD,
    POLICY_TEMP_BASE,
    POLICY_TEMP_RANGE,
    POLICY_WINDOW_SIZE,
    PREDICTION_HORIZON_MAX_SCALE,
    PREDICTION_HORIZON_MIN_SCALE,
    PREDICTIVE_BUDGET_GATING_RATIO,
    QUANTUM_COHERENCE_BOOST_SCALE,
    QUANTUM_COHERENCE_THRESHOLD,
    RESONANCE_TAU_CENTER,
    RESONANCE_TAU_SCALE,
    RESONATOR_CONSOLIDATION_THRESHOLD,
    RESONATOR_ERROR_CONFIDENCE_DAMPEN,
    RESONATOR_ERROR_EXPLORATION_SCALE,
    RESONATOR_ERROR_EXPLORATION_THRESHOLD,
    RESONATOR_FAMILIAR_LR_SCALE,
    RESONATOR_LOW_ERROR_CONFIDENCE_SCALE,
    RESONATOR_LOW_ERROR_THRESHOLD,
    RESONATOR_NOVEL_LR_SCALE,
    RESONATOR_NOVEL_THRESHOLD,
    RESONATOR_SIMILARITY_PRIME_THRESHOLD,
    RESONATOR_STARTUP_CYCLES,
    SELF_MODEL_ACCURACY_EMA,
    SELF_MODEL_CONFIDENCE_WEIGHT,
    SELF_MODEL_HIGH_THRESHOLD,
    SELF_MODEL_HIGH_TRUST_BOOST,
    SELF_MODEL_LOW_CONFIDENCE_SCALE,
    SELF_MODEL_LOW_THRESHOLD,
    SELF_MODEL_URGENCY_WEIGHT,
    SELF_MODEL_WEIGHT_BONUS,
    SELF_MODEL_WEIGHT_LOW_THRESHOLD,
    SELF_MODEL_WEIGHT_PENALTY,
    SLEEP_PRESSURE_LR_DAMPEN_SCALE,
    SLEEP_PRESSURE_LR_FACTOR_MIN,
    SLEEP_PRESSURE_LR_THRESHOLD,
    STILLNESS_BUDGET_CONTRACT_CAP,
    STILLNESS_BUDGET_THRESHOLD,
    THALAMIC_DEEP_BUDGET_SCALE,
    THALAMIC_DEEP_LR_FACTOR,
    THALAMIC_DEEP_SALIENCE,
    THALAMIC_REFLEX_BUDGET_SCALE,
    THALAMIC_REFLEX_LR_FACTOR,
    THALAMIC_REFLEX_SALIENCE,
    TRAINING_BASE_IMPORTANCE,
    TRANSITION_COST_MAX_EFFECT,
    TRANSITION_COST_STRENGTH_SCALE,
    TRANSITION_COST_THRESHOLD,
    VALENCE_HOMEOSTASIS_ALPHA,
    VALENCE_HOMEOSTASIS_MOMENTUM,
    WM_MISMATCH_CONFIDENCE_SCALE,
    WM_MISMATCH_LR_SCALE,
    WORLD_MODEL_ERROR_IMPORTANCE_SCALE,
    WORLD_MODEL_SPONGINESS_THRESHOLD,
    WORLD_MODEL_SPONGY_LR_SCALE,
    WORLD_MODEL_STIFFNESS_LR_SCALE,
    WORLD_MODEL_STIFFNESS_THRESHOLD,
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

/// Results from memory recall, resonator matching, phenomenal binding, and goal attention.
struct MemoryBindingResult {
    memory_context_boost: f32,
    resonator_wm_primed: bool,
    resonator_reconsolidated: usize,
    resonator_best_sim: f32,
    resonator_prediction_error: f32,
    resonator_error_exploration_mod: f32,
    binding_threshold_mod: f32,
    binding_confidence_mod: f32,
    pre_update_coherence: f32,
    goal_attention_bias: f32,
}

/// Results from semantic memory, CfC temporal step, prediction, uncertainty, and world model.
struct CfcPlanningResult {
    /// Owned semantic HDC projection (consumed by parallel post-processing).
    semantic_hdc: Vec<f32>,
    /// Phi-weighted semantic memory LR factor.
    semantic_lr_factor: f32,
    /// Epistemic gate modulation of semantic LR.
    epistemic_semantic_lr_mod: f32,
    /// CfC temporal step size (tau-modulated).
    delta_t: f32,
    /// CfC hidden state output vector.
    output: Vec<f32>,
    /// Multi-scale prediction from CfC.
    prediction: Vec<f32>,
    /// Cross-horizon prediction coherence (EMA'd).
    prediction_coherence: f32,
    /// Model uncertainty (reducible by exploration).
    epistemic_uncertainty: f32,
    /// Data noise uncertainty (irreducible).
    aleatoric_uncertainty: f32,
    /// Sensory-abstract mismatch in world model hierarchy.
    wm_sensory_mismatch: bool,
    /// FEP surprise → CfC tau modulation factor.
    fep_tau_factor: f32,
    /// SpectralMIP Phi → CfC tau modulation factor.
    phi_tau_factor: f32,
    /// Prediction horizon → CfC integration depth factor.
    prediction_horizon_tau: f32,
    /// Whether arousal trap recovery is active.
    arousal_recovery_active: bool,
    /// Arousal recovery tau scaling factor.
    arousal_recovery_tau_factor: f32,
}

/// Results from training, stats update, and parallel post-processing.
struct TrainingPostResult {
    learning_occurred: bool,
    training_loss: Option<f32>,
    effective_lr: f32,
    cycle_reward: f32,
    had_semantic_eviction: bool,
    school_predicted_phi_gain: f32,
}

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
            self.language_comm
                .voice_coherence
                .bridge
                .smoothed_coherence(),
            self.stats.unified_psi as f64,
            phi_attention_weight,
            self.behavior.emotion_contagion.arousal,
            self.behavior.emotion_contagion.valence,
            self.thermodynamic_load,
            self.carryover.quality.last_dissipative_health,
            self.sensorimotor.somatic_bridge.systemic_stress(),
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

            // Macro for panic-isolated subsystem execution.
            // Inlines catch_unwind to enable split borrows: the subsystem manager field,
            // `self.subsystem_health`, and `self.subsystem_collector` are all separate
            // fields of CognitiveLoopService, so the borrow checker accepts this.
            macro_rules! run_subsystem {
                ($mgr:expr_2021, $name:literal, $snapshot:expr_2021) => {
                    if !self.subsystem_health.is_faulted($name) {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            $mgr.process($snapshot)
                        }));
                        match result {
                            Ok(output) => {
                                self.subsystem_health.record_success($name);
                                self.subsystem_collector.record($name, output);
                            }
                            Err(payload) => {
                                let msg = super::cycle::format_panic_payload(payload);
                                tracing::error!(
                                    subsystem = $name,
                                    panic_message = %msg,
                                    "Subsystem panicked — returning NEUTRAL output",
                                );
                                self.subsystem_health.record_panic($name);
                            }
                        }
                    }
                };
            }

            // Pre-encode input text for glyph projection (before snapshot borrow).
            // Uses 3-channel TextHdcEncoder for semantically meaningful modality coordinates
            // instead of the coarse BinaryHV→±1 conversion from the snapshot.
            #[cfg(feature = "glyph_codex")]
            self.glyph_manager.encode_input(input);

            if let Some(ref snapshot) = self.last_snapshot {
                let urgency_u8 = snapshot.urgency;
                let cycle_num = snapshot.cycle_number;

                if self.drive_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.drive_manager, "drive_manager", snapshot);
                }

                if self.memory_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.memory_manager, "memory_manager", snapshot);
                }

                if self.learning_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.learning_manager, "learning_manager", snapshot);
                }

                if self.multimodal_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.multimodal_manager, "multimodal_manager", snapshot);
                }

                if self.perception_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.perception_manager, "perception_manager", snapshot);
                }

                // ── Drain swarm event channel (non-blocking) ──────────
                // Any async P2P component (NetworkService, Hyperfeel,
                // FederatedAggregator) sends SwarmEvents through mpsc::Sender;
                // we drain them here before processing.
                if let Ok(rx_guard) = self.swarm_event_rx.lock() {
                    if let Some(ref rx) = *rx_guard {
                        use super::managers::swarm_manager::SwarmEvent;
                        for _ in 0..20 {
                            match rx.try_recv() {
                                Ok(event) => {
                                    // Sovereign Clock: forward TimeBeacons to TimeManager.
                                    #[cfg(feature = "mesh")]
                                    if let SwarmEvent::TimeBeaconReceived {
                                        source_id,
                                        timestamp_us,
                                        stratum,
                                        phi,
                                        drift_ppm,
                                    } = &event
                                    {
                                        let beacon = crate::swarm::mesh::time_beacon::TimeBeacon {
                                            timestamp_us: *timestamp_us,
                                            stratum: *stratum,
                                            counter: 0,
                                            phi: *phi,
                                            drift_ppm: *drift_ppm,
                                        };
                                        self.time_manager.inject_beacon(beacon, *source_id);
                                    }

                                    // Sovereign Trust: forward TrustVerified to TrustManager.
                                    #[cfg(feature = "mesh-trust")]
                                    if let SwarmEvent::TrustVerified {
                                        peer_id,
                                        trust_level,
                                        ..
                                    } = &event
                                    {
                                        self.trust_manager.inject_event(
                                            super::managers::trust_manager::TrustEvent::PeerVerified {
                                                peer_id: peer_id.clone(),
                                                trust_level: *trust_level,
                                                pq_verified: false,
                                            },
                                        );
                                    }

                                    // Sovereign Social: forward ContentAnnounced to SocialFabricManager.
                                    #[cfg(feature = "social-fabric")]
                                    if let SwarmEvent::ContentAnnounced {
                                        ref peer_id,
                                        content_hash,
                                        truncated_hdv: _,
                                        ref domain,
                                        created_at,
                                    } = &event
                                    {
                                        use symthaea_core::hdc::BinaryHV;
                                        self.social_fabric_manager.inject_event(
                                            super::managers::social_fabric_manager::SocialEvent::ContentReceived(
                                                crate::swarm::resonance_graph::ContentRef {
                                                    source_peer: peer_id.clone(),
                                                    content_hash: *content_hash,
                                                    hdv_embedding: BinaryHV::zero(),
                                                    domain: domain.clone(),
                                                    created_at: *created_at,
                                                },
                                            ),
                                        );
                                    }

                                    // Consciousness-Aware Router: extract peer consciousness from events.
                                    #[cfg(feature = "mesh")]
                                    if let SwarmEvent::PeerJoined {
                                        ref peer_id,
                                        trust_level: trust_score,
                                        ..
                                    } = &event
                                    {
                                        let peer_id_bytes = {
                                            let mut buf = [0u8; 8];
                                            let bytes = peer_id.as_bytes();
                                            let len = bytes.len().min(8);
                                            buf[..len].copy_from_slice(&bytes[..len]);
                                            buf
                                        };
                                        self.consciousness_router.update_peer(
                                            peer_id_bytes,
                                            0.5, // initial phi estimate
                                            0.5, // initial consciousness
                                            1,   // default participant tier
                                            cycle_num,
                                        );
                                    }

                                    // All events still go to SwarmManager for normal processing.
                                    self.swarm_manager.inject_event(event);
                                }
                                Err(_) => break,
                            }
                        }
                    }
                }

                // ── Swarm Manager (interval 41, co-prime) ─────────────
                if self.swarm_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.swarm_manager, "swarm_manager", snapshot);
                }

                // ── Muse Manager (interval 1, every cycle) ──────────
                // Streaming consciousness-driven music synthesis with
                // allostatic sonification and peer distress resonance.
                #[cfg(feature = "muse")]
                {
                    self.muse_manager.inject_neuromod(
                        self.neuromod.bath.dopamine.effective(),
                        self.neuromod.bath.serotonin.effective(),
                        self.neuromod.bath.noradrenaline.effective(),
                        self.neuromod.bath.allostatic_load,
                    );
                    let energy_ratio = if self.substrate_manager.consciousness_viable {
                        0.0
                    } else {
                        1.0
                    };
                    self.muse_manager
                        .inject_substrate(self.substrate_manager.feasibility, energy_ratio);
                    let safety_u8 = if snapshot.unified_psi > 0.6 {
                        0
                    } else if snapshot.unified_psi > 0.3 {
                        1
                    } else if snapshot.unified_psi > 0.1 {
                        2
                    } else {
                        3
                    };
                    self.muse_manager.inject_safety(safety_u8);

                    // Inject aesthetic identity: long-term harmony bias from the
                    // AestheticTracker into generation, so Symthaea develops a
                    // signature sound over hundreds of compositions.
                    #[cfg(feature = "creative")]
                    if let Some(ref cm) = self.sensorimotor.motor_rendering.creative_manager {
                        self.muse_manager.inject_harmony_bias(cm.harmony_bias());
                    }

                    if self.muse_manager.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(self.muse_manager, "muse_manager", snapshot);
                    }

                    // Poll for completed CompositionExports and hand off to the
                    // background MusicPublisher.  Runs every cycle (cheap: just
                    // a VecDeque::pop_front + mpsc::Sender::send).
                    #[cfg(feature = "muse")]
                    {
                        while let Some(export) = self.muse_manager.take_next_export() {
                            // ── Self-listening: score the composition and feed back ──
                            #[cfg(feature = "creative")]
                            if !export.notes.is_empty() {
                                use symthaea_muse::creative_bench::CreativeQualityScore;
                                let target_va = symthaea_aesthetic::ValenceArousal {
                                    valence: export.musical_state.valence,
                                    arousal: export.musical_state.arousal,
                                };
                                // Build a minimal Composition for the scorer
                                let comp = symthaea_muse::Composition {
                                    audio: symthaea_muse::AudioData::Stereo(Vec::new()),
                                    sample_rate: 44100,
                                    notes: export.notes.clone(),
                                    duration_secs: export.duration_secs,
                                    section: symthaea_muse::structure::SectionType::Development,
                                };
                                let quality = CreativeQualityScore::evaluate(&comp, target_va);
                                // Map creative quality → AestheticScore
                                let aesthetic_score = symthaea_aesthetic::AestheticScore {
                                    order: quality.rhythmic_regularity,
                                    complexity: 1.0 - quality.form_compliance, // high form = low complexity
                                    surprise: (1.0 - quality.melodic_coherence).max(0.0), // low coherence = surprise
                                    harmony: quality.emotional_alignment,
                                    birkhoff: quality.composite,
                                    composite: quality.composite,
                                };
                                if let Some(ref mut cm) =
                                    self.sensorimotor.motor_rendering.creative_manager
                                {
                                    let feedback = cm.process_external_score(
                                        &aesthetic_score,
                                        &export.musical_state.harmony_activations,
                                    );
                                    // Feed aesthetic dopamine back to muse injection
                                    self.muse_manager.inject_neuromod(
                                        (self.muse_manager.musical_state().dopamine
                                            + feedback.dopamine_delta)
                                            .clamp(0.0, 1.0),
                                        (self.muse_manager.musical_state().serotonin
                                            + feedback.serotonin_delta)
                                            .clamp(0.0, 1.0),
                                        (self.muse_manager.musical_state().noradrenaline
                                            + feedback.surprise_signal)
                                            .clamp(0.0, 1.0),
                                        0.0, // allostatic load unaffected
                                    );
                                    log::debug!(
                                        "[self-listen] quality={:.3} → dopamine_δ={:+.3}, harmony_bias={:.2?}",
                                        quality.composite,
                                        feedback.dopamine_delta,
                                        cm.last_telemetry(),
                                    );
                                }
                            }
                            self.music_publisher.submit(export);
                        }
                        // Drain upload results (best-effort logging; non-blocking).
                        for result in self.music_publisher.drain_results() {
                            if result.success {
                                log::debug!(
                                    "[music-publisher] uploaded {:?} → CID {:?}",
                                    result.title,
                                    result.cid
                                );
                            } else {
                                log::warn!(
                                    "[music-publisher] upload failed for {:?}: {:?}",
                                    result.title,
                                    result.error
                                );
                            }
                        }
                    }
                }

                // ── Thermodynamic Manager (interval 43, co-prime) ─────
                // Unified thermodynamics: cross-couples dissipative,
                // analyzer, HFE, physics bridge. Inputs set by cycle_consciousness,
                // integration, monitors, and cycle phases.
                {
                    if self.thermodynamic_mgr.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(self.thermodynamic_mgr, "thermodynamic_manager", snapshot);
                    }
                }

                // ── Holon Receiver (every cycle — low cost) ────────────
                // Drain HTTP channel into HolonReceiver, then process all queued messages.
                // Routes tasks, knowledge, and peer state into the existing managers.
                {
                    // Drain mpsc channel from HTTP handlers (HolonHttpState) into HolonReceiver.
                    if let Ok(guard) = self.holon_inbound_rx.lock() {
                        if let Some(ref rx) = *guard {
                            for _ in 0..20 {
                                match rx.try_recv() {
                                    Ok((device_id, msg)) => {
                                        self.holon_receiver.enqueue_message(device_id, msg);
                                    }
                                    Err(_) => break,
                                }
                            }
                        }
                    }
                    let processed = self.holon_receiver.process_inbound(cycle_num as u64);
                    if processed > 0 {
                        use super::managers::swarm_manager::SwarmEvent;

                        // Collect peer data into local vecs (avoid borrow conflicts with downstream managers)
                        let consciousness_updates: Vec<_> = self
                            .holon_receiver
                            .peers()
                            .map(|p| {
                                (
                                    p.device_id.clone(),
                                    p.phi as f64,
                                    p.valence as f64,
                                    p.arousal as f64,
                                )
                            })
                            .collect();
                        let navigation_updates: Vec<_> = self
                            .holon_receiver
                            .peers()
                            .filter_map(|p| {
                                Some((
                                    p.device_id.clone(),
                                    p.last_position_m?,
                                    p.last_position_sigma_m?,
                                    p.navigation_confidence.map(f64::from),
                                ))
                            })
                            .collect();
                        for (peer_id, phi, valence, arousal) in consciousness_updates {
                            self.swarm_manager
                                .inject_event(SwarmEvent::ConsciousnessUpdate {
                                    peer_id,
                                    phi,
                                    valence,
                                    arousal,
                                });
                        }
                        for (peer_id, position_m, position_sigma_m, confidence) in
                            navigation_updates
                        {
                            if self.network_service().is_some() {
                                self.receive_peer_navigation_estimate(
                                    &peer_id,
                                    positioning::GaussianEstimate3D::from_diagonal_sigma(
                                        position_m,
                                        position_sigma_m as f64,
                                    ),
                                    confidence,
                                );
                            } else {
                                self.swarm_manager
                                    .inject_event(SwarmEvent::PeerNavigationUpdate {
                                        peer_id,
                                        position_m,
                                        position_sigma_m: position_sigma_m as f64,
                                        confidence,
                                    });
                            }
                        }
                    }
                    self.holon_receiver.evict_stale(cycle_num as u64, 500);
                }

                // ── Soul Manager (interval 43, co-prime) ──────────────
                if let Some(ref mut soul_mgr) = self.soul_manager {
                    if soul_mgr.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(soul_mgr, "soul_manager", snapshot);
                    }
                }

                // ── Spectrum Manager (interval 53, co-prime) ────────────
                #[cfg(feature = "mesh")]
                {
                    // Swarm→Spectrum: inject synthetic observations from peer state.
                    // This feeds the waterfall model even without SDR hardware.
                    let swarm_telem = self.swarm_manager.telemetry();
                    self.spectrum_manager.ingest_swarm_state(
                        swarm_telem.connected_peers,
                        swarm_telem.mean_peer_phi,
                        swarm_telem.connectivity_ema,
                    );

                    // Consciousness-Aware Router: feed local state each cycle.
                    // Uses snapshot's unified_psi as the consciousness level,
                    // and swarm telemetry for peer Phi tracking.
                    self.consciousness_router.update_local(
                        snapshot.unified_psi as f32,
                        snapshot.unified_psi as f32,
                        0, // governance tier — updated from Mycelix bridge when available
                    );

                    // Store-and-Forward: detect offline/online transitions from network health.
                    let net_health = self.spectrum_manager.network_health();
                    if net_health == super::managers::radio_dispatcher::NetworkHealth::Blackout {
                        self.store_and_forward.go_offline(cycle_num);
                    } else if self.store_and_forward.is_offline() {
                        // Transition to online — may set consolidation_pending.
                        self.store_and_forward.go_online(cycle_num);
                    }

                    // Drain up to 5 buffered experiences per cycle to spread
                    // consolidation cost and avoid multi-second latency spikes.
                    if self.store_and_forward.needs_consolidation() {
                        let wisdom = self.store_and_forward.consolidate_batch(cycle_num, 5);
                        if wisdom.experiences_consolidated > 0 {
                            tracing::info!(
                                experiences = wisdom.experiences_consolidated,
                                duration = wisdom.offline_duration,
                                patterns = wisdom.patterns.len(),
                                "Dream consolidation batch — {} experiences, {} patterns",
                                wisdom.experiences_consolidated,
                                wisdom.patterns.len(),
                            );
                            // TODO(blocked:mesh-wisdom): Transmit consolidated wisdom via mesh bridge.
                            // Blocker: NetworkServiceBridge bidirectional message passing.
                            // Gate: #[cfg(feature = "mesh")]
                        }
                    }

                    if self.spectrum_manager.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(self.spectrum_manager, "spectrum_manager", snapshot);

                        // Cross-coupling: Spectrum → Swarm connectivity modifier
                        let connectivity_penalty = match net_health {
                            super::managers::radio_dispatcher::NetworkHealth::AllTiersUp => 1.0,
                            super::managers::radio_dispatcher::NetworkHealth::LocalDown => {
                                super::thresholds::RADIO_CONNECTIVITY_PENALTY_LOCAL_DOWN
                            }
                            super::managers::radio_dispatcher::NetworkHealth::MetroOnly => {
                                super::thresholds::RADIO_CONNECTIVITY_PENALTY_METRO_ONLY
                            }
                            super::managers::radio_dispatcher::NetworkHealth::Blackout => 0.0,
                        };
                        self.swarm_manager
                            .set_connectivity_modifier(connectivity_penalty);
                    }
                }

                // ── CPG Manager (interval 59, co-prime) ───────────────
                #[cfg(feature = "cpg")]
                {
                    // Cross-coupling: Substrate → CPG frequency scaling
                    self.cpg_manager
                        .set_tau_factor(self.substrate_manager.tau_factor as f64);

                    if self.cpg_manager.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(self.cpg_manager, "cpg_manager", snapshot);
                    }
                }

                // ── Spectral Twin Manager (interval 67, co-prime) ────────
                // Records state every tick for history continuity, but only
                // runs full spectral analysis at its interval.
                #[cfg(feature = "spectral_state")]
                {
                    self.spectral_manager
                        .record_state(&snapshot.compressed_state);
                    if self.spectral_manager.should_run(cycle_num, urgency_u8) {
                        run_subsystem!(self.spectral_manager, "spectral_twin", snapshot);
                    }
                }

                // ── Therapeutic Manager (interval 11, co-prime) ─────────
                #[cfg(feature = "therapeutic")]
                if self.config.enable_therapeutic
                    && self.therapeutic_manager.should_run(cycle_num, urgency_u8)
                {
                    run_subsystem!(self.therapeutic_manager, "therapeutic_manager", snapshot);

                    // ── Bidirectional bridge: neuromod bath → RDoC profile ──
                    // Reads actual transmitter levels and adjusts RDoC domains via EMA.
                    {
                        let bath = [
                            self.neuromod.bath.dopamine.effective(),
                            self.neuromod.bath.noradrenaline.effective(),
                            self.neuromod.bath.serotonin.effective(),
                            self.neuromod.bath.acetylcholine.effective(),
                            self.neuromod.bath.gaba.effective(),
                            self.neuromod.bath.oxytocin.effective(),
                            self.neuromod.bath.glutamate.effective(),
                            self.neuromod.bath.adenosine.effective(),
                        ];
                        self.therapeutic_manager.update_rdoc_from_bath(&bath);
                    }

                    // Inject neuromod deltas from regulation strategy into the bath.
                    // This bridges RDoC domains → the 8-transmitter neuromod system.
                    if let Some(delta) = self.therapeutic_manager.last_neuromod_delta {
                        let half_life = 30_u32; // ~30 cycles for therapeutic effects
                        if delta.serotonin.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod
                                .bath
                                .inject("serotonin", delta.serotonin, half_life);
                        }
                        if delta.dopamine.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod
                                .bath
                                .inject("dopamine", delta.dopamine, half_life);
                        }
                        if delta.noradrenaline.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod.bath.inject(
                                "noradrenaline",
                                delta.noradrenaline,
                                half_life,
                            );
                        }
                        if delta.oxytocin.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod
                                .bath
                                .inject("oxytocin", delta.oxytocin, half_life);
                        }
                        if delta.gaba.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod.bath.inject("gaba", delta.gaba, half_life);
                        }
                        if delta.acetylcholine.abs() > NEUROMOD_DELTA_THRESHOLD {
                            self.neuromod.bath.inject(
                                "acetylcholine",
                                delta.acetylcholine,
                                half_life,
                            );
                        }
                    }
                }

                // ── Fabrication Manager (interval 47, co-prime) ─────────
                #[cfg(feature = "advanced-manufacturing")]
                if self.fabrication_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.fabrication_manager, "fabrication_manager", snapshot);
                }

                // ── Language Manager (interval 61, co-prime) ────────────
                #[cfg(feature = "ssm_language")]
                if self.language_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.language_manager, "language_manager", snapshot);
                }

                // ── Neuroevolution Manager (interval 71, co-prime) ─────
                #[cfg(feature = "neuroevolution")]
                {
                    // Inject MetaCognitive mutation suggestions (Lamarckian loop)
                    if let Some(ref meta) = self.consciousness.self_model_tier.meta_cognition {
                        let suggestions = meta.suggest_mutations();
                        if !suggestions.is_empty() {
                            self.neuroevolution_manager.inject_mutations(suggestions);
                        }
                    }
                    run_subsystem!(self.neuroevolution_manager, "neuroevolution", snapshot);
                }

                // ── Hypervisor Manager (interval 71, co-prime) ──────
                #[cfg(feature = "hypervisor")]
                {
                    run_subsystem!(self.hypervisor_manager, "hypervisor", snapshot);
                }

                // ── Vision Manager (interval 17, co-prime) ─────────
                #[cfg(feature = "vision-manifold")]
                if self.vision_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.vision_manager, "vision_manager", snapshot);
                }

                // ── Reasoning Manager (interval 73, co-prime) ────────
                #[cfg(feature = "reasoning_engine")]
                if self.reasoning_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.reasoning_manager, "reasoning_manager", snapshot);
                }

                // ── Governance Manager (interval 37, co-prime) ──────────
                #[cfg(feature = "mycelix")]
                if self.governance_mgr.should_run(cycle_num, urgency_u8) {
                    // Derive local community mode from experience bus KosmicSong as
                    // single-agent fallback. External bridge can override via accessors.
                    if let Some(ref bus) = self.experience_bus {
                        use crate::experience::kosmic_state::KosmicMode;
                        use crate::mycelix::collective_identity::CommunityMode;

                        // Map local KosmicMode → CommunityMode
                        let mode = match bus.kosmic_state.dominant_mode {
                            KosmicMode::Playful | KosmicMode::Connecting => {
                                CommunityMode::Exploratory
                            }
                            KosmicMode::Nurturing | KosmicMode::Giving => CommunityMode::Protective,
                            KosmicMode::Growing => CommunityMode::Creative,
                            KosmicMode::Contemplative
                            | KosmicMode::Resting
                            | KosmicMode::Integrating => CommunityMode::Reflective,
                            KosmicMode::Balanced => CommunityMode::Reflective,
                        };
                        self.governance_mgr.set_community_mode(mode);

                        // Local epistemic summary from GIS dark spots
                        let ks = &bus.kosmic_state;
                        let blind_spots: Vec<String> = ks
                            .gis_state
                            .dark_spots
                            .iter()
                            .map(|ds| ds.topic_hash.clone())
                            .collect();
                        use crate::experience::kosmic_state::GisType;
                        use crate::mycelix::gis::IgnoranceType;
                        let gis_type = match ks.gis_state.current_type {
                            GisType::KnownKnown => IgnoranceType::Known,
                            GisType::KnownUnknown => IgnoranceType::KnownUnknown,
                            GisType::UnknownKnown => IgnoranceType::Known, // tacit knowledge
                            GisType::UnknownUnknown => IgnoranceType::Unknown,
                            GisType::StrategicIgnorance => IgnoranceType::KnownUnknown,
                        };
                        let summary = crate::mycelix::epistemic_mesh::EpistemicSummary {
                            agent_id: "local".to_string(),
                            dominant_ignorance: gis_type,
                            domain_expertise: vec![],
                            blind_spots,
                        };
                        let mesh =
                            crate::mycelix::epistemic_mesh::EpistemicMesh::new(vec![summary]);
                        // Use set_local — will NOT overwrite an externally-set multi-agent mesh.
                        self.governance_mgr.set_local_epistemic_mesh(mesh);
                    }

                    run_subsystem!(self.governance_mgr, "governance_manager", snapshot);

                    // Cross-coupling: Governance → Spectrum preferred tier
                    #[cfg(feature = "mesh")]
                    if let Some(best_tier) = self.spectrum_manager.best_tier_for_governance() {
                        self.governance_mgr.set_preferred_tier(best_tier);
                    }
                }

                // ── Glyph Manager: symbolic consciousness field tracking ──
                // Interval 43 (co-prime). Projects cognitive state onto 11 Field Modality
                // basis vectors, tracks nearest resonant glyph, developmental spiral position.
                // Science: Jung (1959) — archetypal symbolic fields; Graves (1970) — spiral dynamics.
                #[cfg(feature = "glyph_codex")]
                if self.glyph_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.glyph_manager, "glyph_manager", snapshot);
                }

                // ── Sovereign Inoculation Managers ──────────────────────────
                // Mesh infrastructure subsystems — time consensus, trust graph,
                // social fabric resonance, survival/resource monitoring.

                #[cfg(feature = "mesh")]
                if self.time_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.time_manager, "time_manager", snapshot);
                    // Emit time beacon to mesh peers via CLS→Mind outbound channel.
                    let beacon = self.time_manager.create_beacon();
                    let hv = beacon.encode();
                    let timestamp_s = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as u32;
                    let packet = crate::swarm::mesh::WisdomPacket {
                        source_id: [0; 8], // filled by mesh bridge
                        sequence: 0,
                        phi: snapshot.unified_psi as f32,
                        urgency: crate::swarm::mesh::MeshUrgency::Cruise,
                        timestamp_s,
                        payload_type: crate::swarm::mesh::PayloadType::TimeBeacon,
                        auth_mac: 0,
                        ttl: crate::swarm::mesh::MESH_DEFAULT_TTL,
                        wisdom: hv,
                    };
                    if let Err(e) = self
                        .mesh_outbound_tx
                        .send(crate::swarm::mesh::MeshOutbound { packet })
                    {
                        tracing::debug!(error = %e, "Mesh time beacon send failed — no receiver");
                    }
                }

                #[cfg(feature = "mesh-trust")]
                if self.trust_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.trust_manager, "trust_manager", snapshot);
                }

                #[cfg(feature = "social-fabric")]
                if self.social_fabric_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(
                        self.social_fabric_manager,
                        "social_fabric_manager",
                        snapshot
                    );
                }

                #[cfg(feature = "survival")]
                if self.survival_manager.should_run(cycle_num, urgency_u8) {
                    run_subsystem!(self.survival_manager, "survival_manager", snapshot);
                }

                // ── Knowledge Manager: per-cycle extraction + neuromod coupling ──
                // Not a CognitiveSubsystem — called directly with (input, cycle).
                // Science: Kanerva (2009) HDC, Pearl (2009) Causality.
                if let Some(ref mut km) = self.memory.knowledge_manager {
                    let (_telem, sigs) = km.process(input, cycle_num);

                    // Extract scalar fields to release the borrow on km,
                    // avoiding a full KnowledgeSignals clone per cycle.
                    let sigs_uncertainty = sigs.uncertainty;
                    let sigs_causal_depth = sigs.causal_depth;
                    let sigs_relevance = sigs.relevance;
                    let sigs_contradiction = sigs.contradiction_signal;

                    // ── Write carryover + telemetry (needs &mut km / &km) ──
                    let grounding = (sigs_relevance * KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT
                        + (1.0 - sigs_uncertainty) * KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT)
                        .clamp(0.0, 1.0);
                    self.carryover.quality.wm_knowledge_grounding = grounding;
                    self.carryover.quality.wm_knowledge_injection_count =
                        km.telemetry().facts_inserted.min(255) as u8;

                    // ── Neuromod coupling from knowledge signals ──────────
                    // High uncertainty → NE vigilance (Yu & Dayan 2005)
                    if sigs_uncertainty.is_finite() && sigs_uncertainty > 0.5 {
                        let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
                        let ne_nudge =
                            KNOWLEDGE_UNCERTAINTY_NE_SCALE * (sigs_uncertainty as f32 - 0.5);
                        self.neuromod
                            .bath
                            .noradrenaline
                            .set_baseline((ne_base + ne_nudge).clamp(0.0, 1.0));
                    }

                    // High causal depth → DA reward for deep reasoning (Schultz 1997)
                    if sigs_causal_depth.is_finite()
                        && sigs_causal_depth > KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD
                    {
                        let da_base = self.neuromod.bath.dopamine.baseline_val();
                        self.neuromod.bath.dopamine.set_baseline(
                            (da_base + KNOWLEDGE_CAUSAL_DEPTH_DA_NUDGE).clamp(0.0, 1.0),
                        );
                    }

                    // High relevance → 5-HT grounding confidence (Cools et al. 2008)
                    if sigs_relevance.is_finite()
                        && sigs_uncertainty.is_finite()
                        && sigs_relevance > 0.5
                        && sigs_uncertainty < 0.5
                    {
                        let sht_base = self.neuromod.bath.serotonin.baseline_val();
                        self.neuromod.bath.serotonin.set_baseline(
                            (sht_base + KNOWLEDGE_GROUNDING_SHT_NUDGE).clamp(0.0, 1.0),
                        );
                    }

                    // Contradiction → NE + 5-HT (cognitive dissonance, Festinger 1957)
                    if sigs_contradiction.is_finite() && sigs_contradiction > 0.0 {
                        let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
                        self.neuromod.bath.noradrenaline.set_baseline(
                            (ne_base + KNOWLEDGE_CONTRADICTION_NE_BOOST).clamp(0.0, 1.0),
                        );
                        let sht_base = self.neuromod.bath.serotonin.baseline_val();
                        self.neuromod.bath.serotonin.set_baseline(
                            (sht_base + KNOWLEDGE_CONTRADICTION_SHT_BOOST).clamp(0.0, 1.0),
                        );
                    }

                    // ── Drain contradiction alerts → exploration boost ─────
                    // Confidence = contradiction_signal strength: strong contradictions
                    // get full weight, weak ones are discounted.
                    // Science: Festinger (1957) — dissonance strength scales with confidence.
                    let alerts = km.drain_alerts();
                    if !alerts.is_empty() {
                        let boost = (alerts.len() as f32 * KNOWLEDGE_NOVELTY_EXPLORE_SCALE)
                            .min(KNOWLEDGE_ALERT_EXPLORE_CAP);
                        let contra_conf = sigs_contradiction.clamp(0.0, 1.0) as f32;
                        self.adjust_exploration_weighted(
                            "knowledge_contradictions",
                            boost,
                            Priority::Cognitive,
                            contra_conf.max(KNOWLEDGE_CONTRADICTION_FLOOR), // floor so alerts always have some weight
                        );
                    }

                    // ── Knowledge contradiction → attention reallocation ─────
                    // When contradictions exceed salience threshold, boost exploration
                    // to allocate more cognitive resources toward examining the conflict.
                    // Science: Clark (2013) — predictive processing allocates attention
                    // to prediction error sources; contradictions are high-PE events.
                    if sigs_contradiction.is_finite()
                        && sigs_contradiction > KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD
                    {
                        let intensity = (sigs_contradiction
                            - KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD)
                            .min(1.0);
                        let boost = (intensity * KNOWLEDGE_ATTENTION_CONTRADICTION_BOOST) as f32;
                        self.adjust_exploration("knowledge_attention_realloc", boost);
                    }
                }
            }

            // Knowledge→Ethics cross-coupling: deep causal understanding → moral confidence
            // (Pearl 2009). Applied outside the borrow of knowledge_manager.
            self.cross_couple_knowledge_ethics();
        }

        // ── Cross-couplings: applied after snapshot borrow is released ────
        // Drive→Learning: boredom boosts plasticity, flow dampens LR
        // (Berlyne 1960, Csikszentmihalyi 1990)
        self.cross_couple_drive_learning();

        // Swarm→Neuromod: peer connectivity → oxytocin, anomalies → NE
        // (Zak 2012, Arnsten 2009, Crockett 2009, Schultz 1997)
        self.apply_swarm_neuromod();

        // Swarm→Exploration: peer mean phi modulates local curiosity drive.
        // Low mesh phi → peers are struggling → increase exploration (diverge to find solutions).
        // High mesh phi → collective coherence → deepen focus (converge on shared understanding).
        // (Bak 1996 — SOC: systems at criticality explore more; Kauffman 1993 — fitness landscape search)
        if let Some(svc) = self.network_service() {
            let peer_count = svc.peer_count();
            if peer_count > 0 {
                let mean_phi = svc.network_mean_phi();
                // Neutral point: 0.5 phi. Below → boost exploration, above → dampen.
                let deviation = 0.5 - mean_phi; // positive when peers struggling
                let exploration_delta = (deviation * 0.15).clamp(-0.05, 0.10) as f32;
                if exploration_delta.abs() > 0.001 {
                    self.adjust_exploration("swarm_peer_phi", exploration_delta);
                }
            }
        }

        // Fabrication→Neuromod: Cincinnati anomaly → NE, print success → DA,
        // emergency → NE+5-HT, quality trend → 5-HT, PoGF → oxytocin
        // (Aston-Jones 2005, Schultz 1997, Sapolsky 2004, Crockett 2009, Zak 2012)
        #[cfg(feature = "advanced-manufacturing")]
        self.apply_fabrication_neuromod();

        // Swarm↔Governance: bidirectional peer Φ / governance confidence
        // (Woolley 2010)
        #[cfg(feature = "mycelix")]
        self.cross_couple_swarm_governance();

        // Memory→Learning: consolidation pressure → plasticity boost, low recall → dampening
        // (Born & Wilhelm 2012, Tulving 2002)
        self.cross_couple_memory_learning();

        // Perception→Drive: low coherence → exploration boost, high load → suppression
        // (Damasio 1994, Lavie 2005)
        self.cross_couple_perception_drive();

        // Vision→Attention: visual surprise → prediction confidence modulation
        // (Itti & Koch 2001)
        #[cfg(feature = "vision-manifold")]
        self.cross_couple_vision_attention();

        // Language→Confidence: generation quality → prediction confidence & LR
        // (Clark 2013)
        #[cfg(feature = "ssm_language")]
        self.cross_couple_language_confidence();

        // Reasoning→Exploration: falling reliability → LR boost, high reliability → LR dampen
        // (Carver & Scheier 1998)
        #[cfg(feature = "reasoning_engine")]
        self.cross_couple_reasoning_exploration();

        // ── Self-model tracking, crash detection, PE variance, foveation ──
        let (confidence_crash_detected, lr_frozen) =
            self.track_self_model_and_detect_crash(urgency, &perception);

        // ── Vision surprise → exploration: delegated to VisionManager (interval 17) ──

        // ═══════════════════════════════════════════════════════════════════════
        // 1a–1a.2: Memory recall, resonator, binding, goals
        // ═══════════════════════════════════════════════════════════════════════
        let mem_bind = self.phase_dynamics_memory_binding(
            perception,
            urgency,
            prediction_error,
            module_timings,
        );
        let memory_context_boost = mem_bind.memory_context_boost;
        let resonator_wm_primed = mem_bind.resonator_wm_primed;
        let resonator_reconsolidated = mem_bind.resonator_reconsolidated;
        let resonator_best_sim = mem_bind.resonator_best_sim;
        let resonator_prediction_error = mem_bind.resonator_prediction_error;
        let resonator_error_exploration_mod = mem_bind.resonator_error_exploration_mod;
        let binding_threshold_mod = mem_bind.binding_threshold_mod;
        let binding_confidence_mod = mem_bind.binding_confidence_mod;
        let pre_update_coherence = mem_bind.pre_update_coherence;
        let goal_attention_bias = mem_bind.goal_attention_bias;

        // Re-derive reflection thresholds (also used in FEP decomposition below)
        let reflection_thresholds = self
            .consciousness
            .self_model_tier
            .self_reflection
            .get_thresholds();

        // 1b. Analyze emotional content for simple contagion (keyword-based)
        self.behavior.emotion_contagion.analyze(input);

        // ── Phase 15+18: Emotional homeostasis ──
        // Session 9 Item 7: Track pre-pull valence distance for efficiency computation.
        let pre_pull_valence = self.behavior.emotion_contagion.valence;
        let (valence_homeostasis_pull, arousal_homeostasis_pull, mut homeostasis_pull_strength) =
            self.apply_emotional_homeostasis();

        // Compute homeostasis efficiency: ratio of post/pre distance to target (0.0).
        // <1.0 = pulls working, >1.0 = overcorrecting.
        // Cannon (1929)/Ashby (1960): homeostatic regulation must be monitored for overshoot.
        let post_pull_valence = self.behavior.emotion_contagion.valence;
        let pre_dist = pre_pull_valence.abs().max(0.01);
        let post_dist = post_pull_valence.abs();
        let cycle_efficiency = if pre_dist.is_finite() && post_dist.is_finite() {
            post_dist / pre_dist
        } else {
            1.0 // neutral efficiency on NaN input
        };
        // EMA smooth (alpha=0.2), clamped to [0.5, 1.5] to prevent unbounded drift.
        // Session 15 Item 6: Clamp homeostasis efficiency.
        // Science: Cannon (1929) — regulation has bounded operating range.
        self.carryover.quality.homeostasis_efficiency =
            (self.carryover.quality.homeostasis_efficiency * (1.0 - HOMEOSTASIS_EFFICIENCY_EMA)
                + cycle_efficiency * HOMEOSTASIS_EFFICIENCY_EMA)
                .clamp(0.5, 1.5);

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
        if eff > self.threshold_overrides.homeostasis_recalibrate_high()
            && self.stats.total_cycles > DYNAMICS_POST_BOOT_CYCLES
        {
            self.scale_lr_pri(
                "homeostasis_overcorrect",
                1.0 - HOMEOSTASIS_NEUROMOD_STEP,
                Priority::Homeostatic,
            );
            self.scale_exploration_pri(
                "homeostasis_overcorrect",
                1.0 + HOMEOSTASIS_NEUROMOD_STEP,
                Priority::Homeostatic,
            );
        } else if eff < self.threshold_overrides.homeostasis_recalibrate_low()
            && eff > 0.0
            && self.stats.total_cycles > DYNAMICS_POST_BOOT_CYCLES
        {
            self.scale_lr_pri(
                "homeostasis_sluggish",
                1.0 + HOMEOSTASIS_NEUROMOD_STEP,
                Priority::Homeostatic,
            );
        }

        // Session 10 Item 3: Coherence velocity → CfC tau modulation.
        // Rising coherence → slow down (stabilize); falling → speed up (explore corrections).
        // Science: Buzsáki (2006) — coherent oscillations modulate integration timescale.
        // (Applied below in delta_t product chain as coherence_velocity_tau_factor.)

        // ═══════════════════════════════════════════════════════════════════════
        // 1c. Update Unified Emotional Bridge (VAD-based)
        // ═══════════════════════════════════════════════════════════════════════
        // Feed raw text-derived affect into unified bridge (EmotionContagion
        // is a stateless preprocessor; smoothing happens in UnifiedEmotionalState).
        let simple_valence = self.behavior.emotion_contagion.valence as f64;
        let simple_arousal = self.behavior.emotion_contagion.arousal as f64;
        let dominance = if self.behavior.flow_state.in_flow {
            DOMINANCE_FLOW_BASE + DOMINANCE_FLOW_SCALE * self.behavior.flow_state.intensity as f64
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
        // 2a–6b: Semantic memory, CfC step, prediction, world model
        // ═══════════════════════════════════════════════════════════════════════
        let cfc_plan = self.phase_dynamics_cfc_planning(
            perception,
            pre_update_coherence,
            resonator_best_sim,
            module_timings,
        );
        let semantic_hdc = cfc_plan.semantic_hdc;
        let semantic_lr_factor = cfc_plan.semantic_lr_factor;
        let epistemic_semantic_lr_mod = cfc_plan.epistemic_semantic_lr_mod;
        let delta_t = cfc_plan.delta_t;
        let output = cfc_plan.output;
        let prediction = cfc_plan.prediction;
        let prediction_coherence = cfc_plan.prediction_coherence;
        let epistemic_uncertainty = cfc_plan.epistemic_uncertainty;
        let aleatoric_uncertainty = cfc_plan.aleatoric_uncertainty;
        let wm_sensory_mismatch = cfc_plan.wm_sensory_mismatch;
        let fep_tau_factor = cfc_plan.fep_tau_factor;
        let phi_tau_factor = cfc_plan.phi_tau_factor;
        let prediction_horizon_tau = cfc_plan.prediction_horizon_tau;
        let arousal_recovery_active = cfc_plan.arousal_recovery_active;
        let arousal_recovery_tau_factor = cfc_plan.arousal_recovery_tau_factor;

        // 8. Snapshot previous state into the reusable training buffer before
        //    create_experience moves last_state into the replay buffer.
        //    This replaces a per-cycle Vec<f32> clone (~1 KB alloc) with a
        //    memcpy into a pre-allocated buffer — zero allocator overhead.
        self.copy_last_state_to_training_buf();

        // 9. Create experience (moves last_state into Experience, then stores
        //    the new compressed_state for next cycle).
        self.create_experience(
            &perception.encoding.compressed_state,
            &prediction,
            prediction_error,
        );

        // 10. Update coherence bridge (zero-clone on CfC hot path)
        self.temporal_network.with_tau_refs(|refs| {
            self.language_comm.voice_coherence.bridge.update(refs);
        });

        // 10b. Update temporal signature encoder
        let flattened_tau = self.temporal_network.flattened_tau();
        self.language_comm
            .voice_coherence
            .temporal
            .record_batch(&flattened_tau);

        // 10c. Update adaptive behavior
        let (pattern, pattern_confidence) =
            self.language_comm.voice_coherence.temporal.classify_state();
        let coherence = self
            .language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence();
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
            speech_rate: super::thresholds::VOICE_HEARTBEAT_BASE_RATE
                * self.behavior.adaptive_behavior.speech_rate_multiplier,
            pitch_stability: pattern_confidence,
            coarticulation_smoothness: coherence.clamp(0.0, 1.0)
                * super::thresholds::VOICE_HEARTBEAT_COARTICULATION_WEIGHT,
            listener_prediction: if prediction_error < self.config.learning_threshold {
                super::thresholds::VOICE_HEARTBEAT_LISTENER_SUCCESS
            } else {
                super::thresholds::VOICE_HEARTBEAT_LISTENER_FAIL
            },
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
        };
        self.language_comm
            .voice_coherence
            .voice
            .update(voice_heartbeat);

        let voice_confidence = self
            .language_comm
            .voice_coherence
            .voice
            .summary()
            .voice_confidence;
        self.behavior.adaptive_behavior = AdaptiveBehavior::from_consciousness_state(
            pattern,
            pattern_confidence,
            coherence,
            voice_confidence,
        );

        self.reapply_strategy_modulation(selected_strategy);

        self.behavior.adaptive_behavior.attention_sensitivity *= goal_attention_bias;
        if wm_sensory_mismatch {
            self.behavior.adaptive_behavior.attention_sensitivity *=
                ATTENTION_SENSITIVITY_BOOST_FACTOR;
            // Sensory-abstract mismatch → slow consolidation + dampen confidence.
            // Hierarchical decomposition is breaking → protect abstract representations.
            // Science: Friston (2010) — hierarchical level misalignment = high free energy.
            self.scale_lr_pri(
                "wm_sensory_mismatch",
                WM_MISMATCH_LR_SCALE,
                Priority::Homeostatic,
            );
            self.scale_confidence_pri(
                "wm_sensory_mismatch",
                WM_MISMATCH_CONFIDENCE_SCALE,
                Priority::Homeostatic,
            );
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
        // NOTE: This is the PREVIOUS cycle's effective LR (from stats gathering in helpers/mod.rs).
        // The current cycle's composed LR is not available until line ~2354 (after FEP runs),
        // because FEP contributes semantic_lr_factor/reasoning_lr_factor to the composition.
        // This is a structural ordering dependency, not a bug — FEP sees last cycle's LR.
        let effective_lr = self.stats.adaptive_learning_rate;
        let (fep_action_idx, fep_action_probs, is_surprised, fep_pragmatic_value_raw) =
            self.step_fep_active_inference(prediction_error, coherence);

        // Store action index for JEPA and other downstream consumers
        self.fep.last_action_idx = fep_action_idx as u8;

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
                self.adjust_exploration_pri(
                    "cross_manifold_error",
                    excess * CROSS_MANIFOLD_EXPLORATION_SCALE,
                    Priority::Homeostatic,
                );
                self.scale_confidence_pri(
                    "cross_manifold_error",
                    CROSS_MANIFOLD_CONFIDENCE_DAMPEN,
                    Priority::Homeostatic,
                );
                self.scale_lr_pri(
                    "cross_manifold_error",
                    CROSS_MANIFOLD_LR_BOOST,
                    Priority::Homeostatic,
                );
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
                self.adjust_exploration_pri("vision_horizon_short", boost, Priority::Homeostatic);
            }

            // Long-term error (500ms+, index 2) → planning uncertainty
            if let Some(&long_err) = perception.vision_horizon_errors.get(2) {
                if long_err > VISION_LONG_HORIZON_CONFIDENCE_THRESHOLD {
                    self.scale_confidence_pri(
                        "vision_horizon_long",
                        VISION_HORIZON_CONFIDENCE_DAMPEN,
                        Priority::Homeostatic,
                    );
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
                let effectiveness = (raw_effectiveness
                    * super::thresholds::MCTS_EFFECTIVENESS_NORM_SCALE
                    + super::thresholds::MCTS_EFFECTIVENESS_NORM_OFFSET)
                    .clamp(0.0, 1.0);
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
                    // Confidence = FEP accuracy: models with high accuracy
                    // get full weight; marginal accuracy is discounted.
                    // Science: Friston (2010) — model evidence scales with accuracy.
                    self.adjust_confidence_weighted(
                        "fep_accuracy_high",
                        super::thresholds::FEP_ACCURACY_HIGH_CONFIDENCE,
                        Priority::Homeostatic,
                        acc.clamp(0.0, 1.0) as f32,
                    );
                }
                if comp > FEP_COMPLEXITY_THRESHOLD {
                    self.scale_lr(
                        "fep_complexity",
                        1.0 - ((comp - FEP_COMPLEXITY_THRESHOLD).min(FEP_COMPLEXITY_PENALTY_CAP)
                            * FEP_COMPLEXITY_LR_SCALE) as f32,
                    );
                }
                if surp > reflection_thresholds.surprise as f64 {
                    let s_explore = ((surp - reflection_thresholds.surprise as f64)
                        * FEP_SURPRISE_EXPLORE_SECONDARY_SCALE)
                        .min(FEP_SURPRISE_EXPLORE_SECONDARY_CAP)
                        as f32;
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
                (1.0 - (fep_pragmatic_value - FEP_PRAGMATIC_EXPLOIT_THRESHOLD)
                    * FEP_PRAGMATIC_EXPLOIT_SCALE) as f32,
            );
        } else if fep_pragmatic_value < FEP_PRAGMATIC_EXPLORE_THRESHOLD && fep_pragmatic_value > 0.0
        {
            let p_explore = ((FEP_PRAGMATIC_EXPLORE_THRESHOLD - fep_pragmatic_value)
                * FEP_PRAGMATIC_EXPLORE_SCALE)
                .min(0.05) as f32;
            self.adjust_exploration("fep_pragmatic_low", p_explore);
        }

        if fep_td_error.abs() > FEP_TD_ERROR_DISCOVERY_THRESHOLD {
            if let Some(ref mut enhancer) = self.memory.causal_enhancer {
                if enhancer.should_discover() {
                    let graph = enhancer.run_discovery();
                    tracing::trace!(edges = graph.edges.len(), "causal discovery completed");
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
            self.scale_exploration(
                "fep_td_converged",
                super::thresholds::FEP_TD_CONVERGE_EXPLORE_SCALE,
            );
        }

        // ── Track 5e: Causal graph → attention weighting ─────────────────
        let causal_attention_edges: usize = if let Some(ref enhancer) = self.memory.causal_enhancer
        {
            let graph = enhancer.current_graph();
            let edge_count = graph.edges.len();
            if edge_count > 0 {
                let avg_confidence = if edge_count > 0 {
                    graph.edges.iter().map(|e| e.confidence).sum::<f64>() / edge_count as f64
                } else {
                    0.0
                };
                if edge_count > 5 && avg_confidence > CAUSAL_CONFIDENCE_DENSE_THRESHOLD as f64 {
                    // Confidence = avg_confidence of causal edges: denser, more
                    // confident graphs carry more weight in the consensus.
                    // Science: Pearl (2000) — causal confidence scales with evidence.
                    self.adjust_confidence_weighted(
                        "causal_graph_dense",
                        (avg_confidence as f32 - CAUSAL_CONFIDENCE_DENSE_THRESHOLD)
                            * CAUSAL_DENSE_CONFIDENCE_SCALE,
                        Priority::Cognitive,
                        avg_confidence.clamp(0.0, 1.0) as f32,
                    );
                } else if edge_count >= 3
                    && avg_confidence > CAUSAL_CONFIDENCE_MODERATE_THRESHOLD as f64
                {
                    // Session 13 Item 1: Fill dead zone for moderate causal density.
                    // 3-5 edges with decent confidence = emerging structure → small boost.
                    // Science: Pearl (2000) — partial causal knowledge still informative.
                    self.adjust_confidence_weighted(
                        "causal_graph_emerging",
                        (avg_confidence as f32 - CAUSAL_CONFIDENCE_MODERATE_THRESHOLD)
                            * CAUSAL_MODERATE_CONFIDENCE_SCALE,
                        Priority::Cognitive,
                        avg_confidence.clamp(0.0, 1.0) as f32,
                    );
                }
                if edge_count < 2 && self.stats.total_cycles > 200 {
                    self.adjust_exploration(
                        "sparse_causal_graph",
                        super::thresholds::SPARSE_CAUSAL_EXPLORATION_BOOST,
                    );
                }
                self.stats.causal_attention_uses += 1;
            }
            edge_count
        } else {
            0
        };

        // ── FEP decomposition → adaptive behavior modulation ─────────────
        if fep_accuracy > 0.5 && fep_complexity < 0.5 {
            self.behavior.adaptive_behavior.learning_rate_multiplier =
                (self.behavior.adaptive_behavior.learning_rate_multiplier * 1.1).min(2.0);
            self.behavior.adaptive_behavior.exploration_factor *= FEP_EFFICIENT_EXPLORATION_DAMPEN;
            // Session 13 Item 6: Wire FEP efficiency into proposal system.
            // High accuracy + low complexity = efficient model → boost confidence.
            // Science: Friston (2010) — low complexity = good model evidence.
            self.adjust_confidence("fep_efficient", super::thresholds::FEP_EFFICIENT_CONFIDENCE);
        }
        let surprise_thresh = reflection_thresholds.surprise as f64;
        if fep_surprise > surprise_thresh {
            self.behavior.adaptive_behavior.exploration_factor =
                (self.behavior.adaptive_behavior.exploration_factor + 0.15).min(1.0);
            self.behavior.adaptive_behavior.action_hint = ActionHint::Explore;
        }
        if fep_complexity > FEP_COMPLEXITY_THRESHOLD {
            use super::thresholds::{
                FEP_COMPLEXITY_LR_DAMPEN, FEP_COMPLEXITY_LR_FLOOR, FEP_COMPLEXITY_PAUSE_MAX,
                FEP_COMPLEXITY_PAUSE_MULT,
            };
            self.behavior.adaptive_behavior.learning_rate_multiplier =
                (self.behavior.adaptive_behavior.learning_rate_multiplier
                    * FEP_COMPLEXITY_LR_DAMPEN)
                    .max(FEP_COMPLEXITY_LR_FLOOR);
            self.behavior.adaptive_behavior.pause_multiplier =
                (self.behavior.adaptive_behavior.pause_multiplier * FEP_COMPLEXITY_PAUSE_MULT)
                    .min(FEP_COMPLEXITY_PAUSE_MAX);
            self.behavior.adaptive_behavior.action_hint = ActionHint::SlowDown;
        }

        if fep_surprise > surprise_thresh {
            if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                let surprise_boost = (fep_surprise - surprise_thresh).min(FEP_SURPRISE_EXPLORE_CAP)
                    * FEP_SURPRISE_EXPLORE_SCALE;
                replay.boost_recent_consolidation(surprise_boost);
            }
        }

        if self.behavior.social_mgr.social.external_reward.abs() > f32::EPSILON {
            let outcome_obs = Observation::from_consciousness_state(
                self.behavior.social_mgr.social.external_reward as f64,
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
        // 10d.8 Math Solver Dispatch
        // When perception detected a math intent, dispatch to the appropriate
        // solver engine. The result's Phi feeds into consciousness coupling and
        // the confidence modulates epistemic state.
        // Science: Dehaene (2011) — number sense as a core cognitive module;
        //          Lakoff & Núñez (2000) — mathematical reasoning is embodied.
        // ═══════════════════════════════════════════════════════════════════════
        #[cfg(feature = "mathematics")]
        let math_result = if perception.math.math_detected {
            let _t = Instant::now();
            // ── Phase 7c: Memory recall — check for analogous past problems ──
            // Before solving, search mathematical memory for similar episodes.
            // Science: Hofstadter (2001) — analogy is the core of cognition.
            let _recalled = {
                let query_hv = &perception.encoding.hv16_cached;
                self.math_service
                    .recall_similar(query_hv, 3)
                    .first()
                    .map(|ep| (ep.problem_type, ep.phi, ep.description.clone()))
            };

            // ── Extract numbers from NL input for solver dispatch ──
            let numbers: Vec<f64> = input
                .split_whitespace()
                .filter_map(|w| {
                    w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                        .parse::<f64>()
                        .ok()
                })
                .collect();

            // ── Route to typed solver based on classified problem type ──
            use super::math_service::MathProblemType;
            let problem_type = perception
                .math
                .problem_type
                .unwrap_or(MathProblemType::Unknown);

            let response: Option<super::math_service::MathResponse> =
                match problem_type {
                    MathProblemType::Statistics => {
                        if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::LinearSystem => {
                        if numbers.len() >= 5 {
                            let n = (numbers.len() as f64).sqrt() as usize;
                            let n = n.max(2);
                            let matrix_size = n * n;
                            if numbers.len() >= matrix_size + n {
                                let a_data = &numbers[..matrix_size];
                                let b_data = &numbers[matrix_size..matrix_size + n];
                                Some(self.math_service.solve_linear_system(a_data, n, n, b_data))
                            } else if !numbers.is_empty() {
                                Some(self.math_service.compute_statistics(&numbers))
                            } else {
                                None
                            }
                        } else if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::RootFinding => {
                        if numbers.len() >= 2 {
                            let a = numbers[0];
                            let b = numbers[1];
                            if numbers.len() > 2 {
                                let coeffs = numbers[2..].to_vec();
                                Some(self.math_service.find_root_phi_guided(
                                    &|x| {
                                        coeffs
                                            .iter()
                                            .rev()
                                            .enumerate()
                                            .map(|(i, c)| c * x.powi(i as i32))
                                            .sum::<f64>()
                                    },
                                    a,
                                    b,
                                ))
                            } else {
                                Some(
                                    self.math_service
                                        .find_root_phi_guided(&|x| x * x - 1.0, a, b),
                                )
                            }
                        } else {
                            None
                        }
                    }
                    MathProblemType::Integration => {
                        if numbers.len() >= 2 {
                            let a = numbers[0];
                            let b = numbers[1];
                            if numbers.len() > 2 {
                                let coeffs = numbers[2..].to_vec();
                                Some(self.math_service.integrate(
                                    &|x| {
                                        coeffs
                                            .iter()
                                            .rev()
                                            .enumerate()
                                            .map(|(i, c)| c * x.powi(i as i32))
                                            .sum::<f64>()
                                    },
                                    a,
                                    b,
                                ))
                            } else {
                                Some(self.math_service.integrate(&|x| x * x, a, b))
                            }
                        } else {
                            None
                        }
                    }
                    MathProblemType::MatrixAnalysis => {
                        if numbers.len() >= 4 {
                            let n = (numbers.len() as f64).sqrt() as usize;
                            let n = n.max(2);
                            if numbers.len() >= n * n {
                                Some(self.math_service.matrix_determinant(&numbers[..n * n], n))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    MathProblemType::Optimization => {
                        if !numbers.is_empty() {
                            Some(self.math_service.optimize(
                                &|x: &[f64]| x.iter().map(|v| v * v).sum::<f64>(),
                                &numbers,
                            ))
                        } else {
                            None
                        }
                    }
                    MathProblemType::SignalAnalysis => {
                        if !numbers.is_empty() {
                            Some(self.math_service.compute_fft(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::Geometry => {
                        if numbers.len() >= 4 && numbers.len() % 2 == 0 {
                            let pairs: Vec<(f64, f64)> =
                                numbers.chunks(2).map(|c| (c[0], c[1])).collect();
                            Some(self.math_service.convex_hull(&pairs))
                        } else {
                            None
                        }
                    }
                    MathProblemType::GraphTheory => {
                        if numbers.len() >= 3 && numbers.len() % 3 == 0 {
                            // Guard against panic: negative values, NaN, Infinity, or
                            // excessively large node indices would cause usize overflow
                            // or unbounded allocation. Cap node count at 1024.
                            const MAX_GRAPH_NODES: usize = 1024;
                            let max_val = numbers.iter().cloned().fold(0.0f64, f64::max);
                            if max_val.is_finite()
                                && max_val >= 0.0
                                && (max_val as usize) < MAX_GRAPH_NODES
                                && numbers.iter().all(|v| v.is_finite())
                            {
                                let n = max_val as usize + 1;
                                let edges: Vec<(usize, usize, f64)> = numbers
                                    .chunks(3)
                                    .filter_map(|c| {
                                        let (a, b) = (c[0], c[1]);
                                        if a >= 0.0 && b >= 0.0 {
                                            Some((a as usize, b as usize, c[2]))
                                        } else {
                                            None // skip edges with negative node indices
                                        }
                                    })
                                    .collect();
                                Some(self.math_service.shortest_path(n, &edges, 0))
                            } else {
                                None // invalid graph input
                            }
                        } else {
                            None
                        }
                    }
                    MathProblemType::DifferentialEquation => {
                        if numbers.len() >= 3 {
                            // Interpret as: [t_start, t_end, y0_0, y0_1, ...]
                            let t_start = numbers[0];
                            let t_end = numbers[1];
                            let y0 = &numbers[2..];
                            Some(self.math_service.solve_ode(
                                |_t, y| y.to_vec(), // identity RHS as fallback
                                (t_start, t_end),
                                y0,
                                100,
                            ))
                        } else if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::Logic => {
                        // Route to tautology checker with a simple proposition
                        // (structured logic queries need richer input parsing)
                        if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::ConstraintSatisfaction => {
                        // CSP needs structured input (domains, constraints) — fall back
                        // to statistics for now; full CSP routing needs text→CSP parser
                        if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                    MathProblemType::Chemistry => Some(self.math_service.compute_chemistry(input)),
                    MathProblemType::Proof => Some(self.math_service.construct_proof(input)),
                    // Arithmetic, Unknown — safe fallback to statistics
                    _ => {
                        if !numbers.is_empty() {
                            Some(self.math_service.compute_statistics(&numbers))
                        } else {
                            None
                        }
                    }
                };

            let dm = if let Some(resp) = response {
                // Math Phi → consciousness coupling: boost prediction confidence
                // when math produces high-Phi verified results.
                if resp.multipath_verified && resp.phi > 0.3 {
                    self.adjust_confidence(
                        "math_verified",
                        super::thresholds::MATH_VERIFIED_CONFIDENCE,
                    );
                }
                // Epistemic caveat → dampen confidence (honest uncertainty).
                if resp.epistemic_caveat.is_some() {
                    self.scale_confidence(
                        "math_epistemic_caveat",
                        super::thresholds::MATH_CAVEAT_CONFIDENCE_SCALE,
                    );
                }
                // Math Phi → DA nudge (reward signal for successful problem solving).
                // Science: Schultz (1997) — unexpected reward prediction error → DA burst.
                if resp.phi > 0.5 {
                    let da_base = self.neuromod.bath.dopamine.baseline_val();
                    self.neuromod.bath.dopamine.set_baseline(da_base + 0.01);
                }

                // Move owned fields (answer, epistemic_caveat) instead of cloning.
                DynMath {
                    solved: true,
                    phi: resp.phi,
                    confidence: resp.confidence,
                    multipath_verified: resp.multipath_verified,
                    answer: resp.answer,
                    epistemic_caveat: resp.epistemic_caveat,
                    error_bound: resp.error_bound,
                }
            } else {
                DynMath::default()
            };
            module_timings.math_service += _t.elapsed().as_micros() as u64;
            dm
        } else {
            DynMath::default()
        };
        #[cfg(not(feature = "mathematics"))]
        let math_result = DynMath::default();

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
        if confidence_velocity > CONFIDENCE_VELOCITY_POSITIVE_THRESHOLD
            && self.stats.total_cycles > 15
        {
            let dampen = (1.0 - confidence_velocity * CONFIDENCE_VELOCITY_DAMPEN_SCALE).max(0.95);
            self.scale_exploration("confidence_rising", dampen);
        }
        // Falling confidence → speed up learning (model needs correction).
        // Confidence collapse signals prediction degradation → recalibrate faster.
        // Science: Cools et al. (2008) — rapid confidence decline triggers
        // serotonergic recalibration and increased learning rate.
        if confidence_velocity < CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD
            && self.stats.total_cycles > 15
        {
            let boost = (1.0
                + (-confidence_velocity - CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD.abs())
                    * CONFIDENCE_VELOCITY_BOOST_SCALE)
                .min(1.15);
            self.scale_lr("confidence_falling", boost);
        }
        let unified_psi = neuromod_result.unified_psi;
        let guiding_question = neuromod_result.guiding_question;
        let dominant_harmonic = neuromod_result.dominant_harmonic;
        let guiding_priority_category = neuromod_result.guiding_priority_category;

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6a Markov Blanket Permeability (Friston 2013; Kirchhoff et al. 2018)
        // ═══════════════════════════════════════════════════════════════════════
        // Feed neuromodulator bath + sentinel threat + flow state into the
        // boundary operator BEFORE the FEP cycle. This gates how much external
        // surprise enters (sensory permeability) and how much internal state
        // leaks outward (active permeability).
        {
            let blanket_inputs = crate::consciousness::fep_active_inference::PermeabilityInputs {
                acetylcholine: self.neuromod.bath.acetylcholine.effective() as f64,
                noradrenaline: self.neuromod.bath.noradrenaline.effective() as f64,
                serotonin: self.neuromod.bath.serotonin.effective() as f64,
                oxytocin: self.neuromod.bath.oxytocin.effective() as f64,
                threat_level: {
                    #[cfg(feature = "sentinel")]
                    {
                        self.sentinel_manager.threat_level() as f64
                    }
                    #[cfg(not(feature = "sentinel"))]
                    {
                        0.0
                    }
                },
                peer_trust: self.swarm_manager.telemetry().connectivity_ema,
                flow_state: self.behavior.flow_state.intensity as f64,
            };
            self.fep
                .enhanced_bridge
                .update_blanket_permeability(&blanket_inputs);
        }

        // Blanket → neuromodulator feedback (closed loop).
        // The blanket state feeds back into neuromod bath:
        // isolation → NE spike, coalescence → oxytocin, opening → 5-HT, closing → NE.
        {
            let perm = self.fep.enhanced_bridge.blanket.permeability();
            let trend = self.fep.enhanced_bridge.blanket.trend();
            if perm.effective < 0.2 {
                self.neuromod
                    .bath
                    .noradrenaline
                    .produce((0.2 - perm.effective) as f32 * 0.3);
            }
            if self.fep.enhanced_bridge.blanket.coalescence_ready(0.6) {
                self.neuromod.bath.oxytocin.produce(0.02);
            }
            if trend > 0.01 {
                self.neuromod
                    .bath
                    .serotonin
                    .produce((trend * 0.15).min(0.03) as f32);
            }
            if trend < -0.01 {
                self.neuromod
                    .bath
                    .noradrenaline
                    .produce((-trend * 0.1).min(0.02) as f32);
            }
        }

        // Topology → blanket constraint: coherence proxies boundary quality.
        {
            let boundary_thickness_proxy = (1.0 - coherence as f64).clamp(0.0, 1.0);
            let fiedler_proxy = self.prediction_confidence.clamp(0.0, 2.0);
            let boundary_components = if self.carryover.quality.subsystem_veto {
                3
            } else {
                1
            };
            self.fep.apply_topology_constraints(
                boundary_thickness_proxy,
                fiedler_proxy,
                boundary_components,
            );
        }

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

            // JEPA latent prediction error blend: redistribute learning signal
            // composition from (0.4*td + 0.3*motor + 0.3*fe) to include
            // latent-space PE for thermodynamically efficient prediction.
            #[cfg(feature = "jepa")]
            if let Some(ref jepa) = self.jepa_engine {
                let jepa_pe = jepa.latent_pe() as f32;
                // Blend: 80% original signal + 20% JEPA latent PE
                // This preserves the FEP bridge's existing precision weighting
                // while adding the latent prediction error contribution.
                self.fep.learning_signal = 0.8 * self.fep.learning_signal + 0.2 * jepa_pe;
            }

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
                    let shift_amount = enhanced_result.motor_command.intensity as f32
                        * MOTOR_ATTENTION_SHIFT_SCALE;
                    self.behavior.adaptive_behavior.attention_sensitivity =
                        (self.behavior.adaptive_behavior.attention_sensitivity
                            * (1.0 + shift_amount * MOTOR_ATTENTION_SENSITIVITY_SCALE))
                            .clamp(
                                MOTOR_ATTENTION_SENSITIVITY_MIN,
                                MOTOR_ATTENTION_SENSITIVITY_MAX,
                            );
                    self.stats.attention_shift = shift_amount;
                }
                MotorCommandType::LearningRateAdjust => {
                    if enhanced_result.should_learn {
                        let lr_mod = enhanced_result.fep_result.learning_rate_modulation as f32;
                        self.stats.adaptive_learning_rate = (self.stats.adaptive_learning_rate
                            * MOTOR_ADAPTIVE_LR_MOMENTUM
                            + lr_mod * MOTOR_ADAPTIVE_LR_ALPHA)
                            .clamp(MOTOR_ADAPTIVE_LR_MIN, MOTOR_ADAPTIVE_LR_MAX);
                    }
                }
                MotorCommandType::ExplorationTrigger => {
                    let intensity = enhanced_result.motor_command.intensity as f32;
                    if enhanced_result.fep_result.epistemic_value
                        > MOTOR_EXPLORATION_EPISTEMIC_THRESHOLD as f64
                    {
                        // Scale exploration boost by epistemic value
                        let boost = (intensity * MOTOR_EXPLORATION_INTENSITY_SCALE)
                            .min(MOTOR_EXPLORATION_BOOST_MAX);
                        self.adjust_exploration("motor_exploration_trigger", boost);
                    }
                    // High-intensity exploration → boost learning to absorb novelty
                    if intensity > 0.8 {
                        self.scale_lr(
                            "motor_explore_intense",
                            super::thresholds::MOTOR_EXPLORE_INTENSE_LR,
                        );
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    let intensity = enhanced_result.motor_command.intensity as f32;
                    if intensity > super::thresholds::MOTOR_REFLECTION_THRESHOLD {
                        self.consciousness
                            .self_model_tier
                            .self_reflection
                            .force_reflection();
                        // Boost meta-awareness proportional to intensity
                        self.adjust_confidence(
                            "motor_reflection",
                            (intensity - super::thresholds::MOTOR_REFLECTION_THRESHOLD)
                                * super::thresholds::MOTOR_REFLECTION_CONFIDENCE_SCALE,
                        );
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
                    if enhanced_result.action_outcome_coupling
                        < ACTION_OUTCOME_COUPLING_RESET_THRESHOLD as f64
                    {
                        self.last_prediction = None;
                        self.set_confidence("inference_mode_init", INFERENCE_MODE_INIT_CONFIDENCE);
                        // Reset world model levels to accept new patterns
                        self.fep.world_model.reset();
                    }
                }
                MotorCommandType::MotorOutput => {
                    // Ethics gate: Blocked verdict prevents motor execution
                    let effective_verdict = self
                        .ethics_verdict_override
                        .as_ref()
                        .unwrap_or(&self.last_ethics_verdict);
                    if *effective_verdict == super::ethics_engine::EthicalVerdict::Blocked {
                        let result = super::motor_output_bridge::MotorOutputResult {
                            success: false,
                            action_type: None,
                            prediction_error: 1.0,
                            outcome: None,
                            error: Some(
                                "Action blocked by ethics engine \
                                 — consent violation or value veto"
                                    .to_string(),
                            ),
                        };
                        self.sensorimotor.motor_rendering.last_result = Some(result);
                    } else if self.carryover.quality.safety_motor_halt {
                        // Safety enforcement Red: consciousness < 0.1 — hard motor halt.
                        // This is the physical boundary of the consciousness safety cascade.
                        // No motor output of any kind may execute at Red level.
                        tracing::warn!(
                            cycle = self.stats.total_cycles,
                            "Motor output HALTED by safety enforcement (Red level) \
                             — consciousness below critical threshold"
                        );
                        let result = super::motor_output_bridge::MotorOutputResult {
                            success: false,
                            action_type: None,
                            prediction_error: 1.0,
                            outcome: None,
                            error: Some(
                                "Motor output halted — safety level Red \
                                 (consciousness below 0.1)"
                                    .to_string(),
                            ),
                        };
                        self.sensorimotor.motor_rendering.last_result = Some(result);
                    } else if self.carryover.quality.safety_motor_readonly {
                        // Safety enforcement Orange: only read-only actions permitted.
                        // Write, execute, and destructive actions are blocked.
                        let is_readonly = enhanced_result
                            .motor_command
                            .parameters
                            .first()
                            .and_then(|&v| super::motor_output_bridge::ActionType::from_param(v))
                            .map_or(false, |at| {
                                matches!(
                                    at,
                                    super::motor_output_bridge::ActionType::Read
                                        | super::motor_output_bridge::ActionType::List
                                        | super::motor_output_bridge::ActionType::Parse
                                )
                            });
                        if is_readonly {
                            // Read-only action: permitted at Orange level
                            if let Some(ref mut bridge) =
                                self.sensorimotor.motor_rendering.output_bridge
                            {
                                let request = self
                                    .sensorimotor
                                    .motor_rendering
                                    .pending_request
                                    .take()
                                    .unwrap_or_default();
                                let motor_phi = if self.carryover.history.consciousness_level > 0.0
                                {
                                    self.carryover.history.consciousness_level
                                } else {
                                    coherence as f64
                                };
                                let result = bridge.execute(
                                    &enhanced_result.motor_command.parameters,
                                    enhanced_result.motor_command.confidence,
                                    motor_phi,
                                    &request,
                                );
                                self.sensorimotor.motor_rendering.last_phi = motor_phi;
                                self.sensorimotor.motor_rendering.last_result = Some(result);
                            }
                        } else {
                            tracing::warn!(
                                cycle = self.stats.total_cycles,
                                "Non-read motor output blocked by safety enforcement \
                                 (Orange level — read-only mode)"
                            );
                            let result = super::motor_output_bridge::MotorOutputResult {
                                success: false,
                                action_type: None,
                                prediction_error: 1.0,
                                outcome: None,
                                error: Some(
                                    "Motor output restricted to read-only — safety level Orange"
                                        .to_string(),
                                ),
                            };
                            self.sensorimotor.motor_rendering.last_result = Some(result);
                        }
                    } else if self.carryover.quality.subsystem_veto {
                        // Subsystem veto: sentinel/safety manager flagged this cycle unsafe
                        tracing::warn!(
                            cycle = self.stats.total_cycles,
                            "Motor output blocked by subsystem veto — safety or sentinel override"
                        );
                        let result = super::motor_output_bridge::MotorOutputResult {
                            success: false,
                            action_type: None,
                            prediction_error: 1.0,
                            outcome: None,
                            error: Some(
                                "Action blocked by subsystem veto \
                                 — safety or sentinel override"
                                    .to_string(),
                            ),
                        };
                        self.sensorimotor.motor_rendering.last_result = Some(result);
                    } else if let Some(ref mut bridge) =
                        self.sensorimotor.motor_rendering.output_bridge
                    {
                        let request = self
                            .sensorimotor
                            .motor_rendering
                            .pending_request
                            .take()
                            .unwrap_or_default();
                        // Use actual consciousness level for Phi gating,
                        // falling back to coherence if not yet computed.
                        // Coherence alone is a poor Phi proxy — it measures voice
                        // resonator stability, not integrated information (Tononi 2004).
                        let motor_phi = if self.carryover.history.consciousness_level > 0.0 {
                            self.carryover.history.consciousness_level
                        } else {
                            coherence as f64
                        };
                        // Caution verdict: cap motor confidence at 0.3
                        let effective_confidence = if *effective_verdict
                            == super::ethics_engine::EthicalVerdict::Caution
                        {
                            enhanced_result
                                .motor_command
                                .confidence
                                .min(ETHICS_CAUTION_CONFIDENCE_CAP as f64)
                        } else {
                            enhanced_result.motor_command.confidence
                        };
                        let result = bridge.execute(
                            &enhanced_result.motor_command.parameters,
                            effective_confidence,
                            motor_phi,
                            &request,
                        );

                        // Feed outcome back as FEP observation
                        let obs_value = if result.success {
                            MOTOR_SUCCESS_OBSERVATION_VALUE
                        } else {
                            MOTOR_FAILURE_OBSERVATION_VALUE
                        };
                        let motor_obs = symthaea_fep::Observation::from_consciousness_state(
                            obs_value,
                            result.prediction_error,
                            motor_phi,
                            effective_lr as f64,
                        );
                        self.fep.agent.perceive(&motor_obs);

                        self.sensorimotor.motor_rendering.last_phi = motor_phi;
                        self.sensorimotor.motor_rendering.last_result = Some(result);
                    }
                }
                MotorCommandType::NoOp => {}
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
            self.scale_lr(
                "coherence_degraded",
                super::thresholds::COHERENCE_DEGRADED_LR_BOOST,
            );
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

        // ── Attention budget check, active rest, moral attractor ──────────
        let (attention_budget_exceeded, predictive_budget_gated, epistemic_budget_scale) =
            self.compute_attention_budget(cycle_start, epistemic_uncertainty);

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.0 PsiAttestation record (every 10 cycles to reduce overhead)
        // ═══════════════════════════════════════════════════════════════════════
        if self.config.enable_psi_attestation
            && self.config.agent_did.is_some()
            && (self.stats.total_cycles % 10 == 0)
        {
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
            self.push_psi_attestation(record);
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
        // Reasoning engine internal diagnostics (populated when feature enabled).
        #[allow(unused_mut)]
        let mut re_phi_eff_raw: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_phi_eff: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_epistemic_mod: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_gamma: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_reliability: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_budget_consumed: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_wall_time_us: u64 = 0;
        #[allow(unused_mut)]
        let mut re_steps_taken: u32 = 0;
        #[allow(unused_mut)]
        let mut re_tier_reached: u32 = 0;
        #[allow(unused_mut)]
        let mut re_gate_checks: u32 = 0;
        #[allow(unused_mut)]
        let mut re_budget_exceeded: bool = false;
        #[allow(unused_mut)]
        let mut re_evs: f32 = 0.0;
        #[allow(unused_mut)]
        let mut re_mcts_iterations: u32 = 0;
        #[allow(unused_mut)]
        let mut re_did_simulate: bool = false;
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
                // Static actions are constructed once via OnceLock to avoid per-cycle
                // string allocations on the hot path.
                #[allow(unused_mut)]
                let mut actions: Vec<
                    crate::consciousness::temporal_planning::types::PlannedAction,
                > = Vec::new();

                #[cfg(feature = "code_generation")]
                {
                    use crate::consciousness::temporal_planning::types::PlannedAction;
                    static CODE_ACTIONS: std::sync::OnceLock<Vec<PlannedAction>> =
                        std::sync::OnceLock::new();
                    let cached = CODE_ACTIONS.get_or_init(|| {
                        vec![
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
                        ]
                    });
                    actions.extend(cached.iter().cloned());
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
                    epistemic_quality: 0.5, // default neutral; wired when epistemic tiers active
                    code_context: self.carryover.injected_code_context.take(),
                };

                let reasoning_result = reasoning_engine.reason(&reasoning_ctx);

                reasoning_confidence = reasoning_result.phi_eff as f32;
                reasoning_lr_factor = reasoning_result.reliability as f32;

                if let Some(ref gate) = reasoning_result.gate {
                    if !gate.is_allowed() {
                        reasoning_gate_blocked = true;
                        reasoning_fallback = gate.fallback.as_ref().map(|f| f.label().to_string());
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

                // Populate internal reasoning engine diagnostics (all Copy fields).
                re_phi_eff_raw = reasoning_result.phi_eff_raw as f32;
                re_phi_eff = reasoning_result.phi_eff as f32;
                re_epistemic_mod = reasoning_result.epistemic_mod as f32;
                re_gamma = reasoning_result.gamma as f32;
                re_reliability = reasoning_result.reliability as f32;
                re_wall_time_us = reasoning_result.wall_time_us;
                re_budget_exceeded = reasoning_result.budget_exceeded;
                re_gate_checks = reasoning_result.gate_checks;
                re_evs = reasoning_result.evs as f32;
                // Derive tier_reached from BudgetTier enum.
                use crate::consciousness::temporal_planning::types::BudgetTier as BT;
                re_tier_reached = match reasoning_result.tier {
                    BT::Tier0 => 0,
                    BT::Tier1 => 1,
                    BT::Tier2 => 2,
                };
                // Steps: Tier0=2(detect+assess), Tier1=+3(decide+plan+gate), Tier2=+2(analyze+narrative)
                re_steps_taken = match reasoning_result.tier {
                    BT::Tier0 => 3, // detect + assess + gate
                    BT::Tier1 => 5, // + decide + plan
                    BT::Tier2 => 7, // + analyze + narrative
                };

                // Move narrative last — avoids cloning an Option<String>.
                reasoning_narrative = reasoning_result.narrative;
                // Budget consumed: wall_time / available_budget.
                if available_us > 0 {
                    re_budget_consumed =
                        (reasoning_result.wall_time_us as f32 / available_us as f32).min(1.0);
                }
                // MCTS iterations from plan.
                if let Some(ref plan) = reasoning_result.plan {
                    re_mcts_iterations = plan.iterations;
                    re_did_simulate = plan.did_plan;
                }

                tracing::debug!(
                    tier = ?reasoning_result.tier,
                    phi_eff = reasoning_result.phi_eff,
                    phi_eff_raw = reasoning_result.phi_eff_raw,
                    epistemic_mod = reasoning_result.epistemic_mod,
                    reliability = reasoning_result.reliability,
                    gate_blocked = reasoning_gate_blocked,
                    plan_confidence = reasoning_plan_confidence,
                    wall_time_us = reasoning_result.wall_time_us,
                    budget_exceeded = reasoning_result.budget_exceeded,
                    gate_checks = reasoning_result.gate_checks,
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
                // Session 15 Item 8: Accelerate recovery when Phi is improving.
                // If unified Psi exceeds recent average, add bonus progress.
                // Science: Tononi (2004) — rising Phi signals integration recovery.
                let phi_bonus = if unified_psi > self.stats.avg_psi as f64 * 1.05 {
                    0.25 // 25% bonus progress when Phi above average
                } else {
                    0.0
                };
                let recovery = ((counter as f32 / 20.0) + phi_bonus).min(1.0);
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

        // Phi → LR stabilization: high Phi dampens learning to preserve integrated representations.
        // Only active when phi_tau_feedback is enabled — creates behavioral divergence between
        // conscious (selective learning) and zombie (indiscriminate learning) agents.
        // Science: Tononi (2008) — high Φ = stable integration, reduce plasticity.
        let effective_lr = if self.config.enable_phi_tau_feedback {
            if let Some(phi) = self.carryover.consciousness.last_spectral_mip_phi {
                let normalized = 1.0
                    / (1.0
                        + (-(phi - super::thresholds::PHI_TAU_REFERENCE as f64)
                            * super::thresholds::PHI_TAU_SIGMOID_STEEPNESS as f64)
                            .exp());
                let dampen = (normalized as f32 * super::thresholds::PHI_LR_STABILIZATION_SCALE)
                    .min(super::thresholds::PHI_LR_STABILIZATION_MAX);
                effective_lr * (1.0 - dampen)
            } else {
                effective_lr
            }
        } else {
            effective_lr
        };

        let effective_lr = effective_lr * self.neuromod.bath.gradient_scale_factor();
        let effective_lr = effective_lr * self.neuromod.bath.plasticity_gate();
        let effective_lr = if self.neuromod.bath.sleep_pressure() > SLEEP_PRESSURE_LR_THRESHOLD {
            let pressure_factor = 1.0
                - (self.neuromod.bath.sleep_pressure() - SLEEP_PRESSURE_LR_THRESHOLD)
                    * SLEEP_PRESSURE_LR_DAMPEN_SCALE;
            effective_lr * pressure_factor.clamp(SLEEP_PRESSURE_LR_FACTOR_MIN, 1.0)
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

        // ═══════════════════════════════════════════════════════════════════════
        // 11–12 + Broca + Parallel: Training, stats, Broca, post-processing
        // ═══════════════════════════════════════════════════════════════════════
        // Take the training buffer to break the borrow — phase_dynamics_training
        // only reads the slice to build an Array1 for the CfC update step.
        // After training we put the buffer back so it can be reused next cycle
        // (no allocation, just two Option pointer swaps).
        let training_buf = self.training_state_buf.take();
        let previous_state = training_buf.as_deref();
        let train_result = self.phase_dynamics_training(
            input,
            perception,
            prediction_error,
            effective_lr,
            delta_t,
            previous_state,
            &output,
            coherence,
            semantic_hdc,
            urgency,
            selected_strategy,
            surprise_triggered,
            memory_context_boost,
            cycle_start,
            &math_result,
            semantic_lr_factor,
            module_timings,
        );
        // Return the training buffer for reuse next cycle (no allocation).
        self.training_state_buf = training_buf;
        let learning_occurred = train_result.learning_occurred;
        let training_loss = train_result.training_loss;
        let effective_lr = train_result.effective_lr;
        let cycle_reward = train_result.cycle_reward;
        let had_semantic_eviction = train_result.had_semantic_eviction;
        let school_predicted_phi_gain = train_result.school_predicted_phi_gain;

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
                self_model_accuracy: self.carryover.learning.self_model_accuracy,
            },
            fep: DynFep {
                fep_action_idx,
                fep_pragmatic_value,
                fep_accuracy,
                fep_complexity,
                fep_surprise,
                fep_td_error,
                trajectory_efe: self.fep.trajectory_telemetry.best_efe,
                trajectory_best_action: self.fep.trajectory_telemetry.best_action,
                trajectory_surprise: self.fep.trajectory_telemetry.best_trajectory_surprise,
                trajectory_ode_steps: self.fep.trajectory_telemetry.total_ode_steps,
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
                re_phi_eff_raw,
                re_phi_eff,
                re_epistemic_mod,
                re_gamma,
                re_reliability,
                re_budget_consumed,
                re_wall_time_us,
                re_steps_taken,
                re_tier_reached,
                re_gate_checks,
                re_budget_exceeded,
                re_evs,
                re_mcts_iterations,
                re_did_simulate,
            },
            attention: DynAttention {
                attention_budget_exceeded,
                attention_budget_elapsed_us: cycle_start.elapsed().as_micros() as u64,
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
            math: math_result,
            neuromod: DynNeuromod {
                neuromod_attention_alloc: self.neuromod.bath.attention_budget_allocation(),
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
            phi_tau_factor,
            prediction_horizon_tau,
            causal_world_model_edges: if self
                .memory
                .causal_enhancer
                .as_ref()
                .map_or(false, |e| e.has_causal_structure())
            {
                self.memory
                    .causal_enhancer
                    .as_ref()
                    .map_or(0, |e| e.current_graph().edges.len())
            } else {
                0
            },
            epistemic_budget_scale,
            confidence_crash_detected,
            lr_frozen,
            semantic_evictions: if had_semantic_eviction { 1 } else { 0 },
        }
    }

    /// Attention budget computation, active rest mode, and moral attractor dampening.
    ///
    /// Returns `(attention_budget_exceeded, predictive_budget_gated, epistemic_budget_scale)`.
    fn compute_attention_budget(
        &mut self,
        cycle_start: std::time::Instant,
        epistemic_uncertainty: f32,
    ) -> (bool, bool, f32) {
        let neuromod_attention_alloc = self.neuromod.bath.attention_budget_allocation();
        let depth_budget_scale = match self.cognitive_depth {
            super::CognitiveDepth::DeepThought => THALAMIC_DEEP_BUDGET_SCALE,
            super::CognitiveDepth::Cortical => 1.0,
            super::CognitiveDepth::Reflex => THALAMIC_REFLEX_BUDGET_SCALE,
        };
        let epistemic_budget_scale = if epistemic_uncertainty > EPISTEMIC_BUDGET_EXPAND_THRESHOLD {
            1.0 + (epistemic_uncertainty - EPISTEMIC_BUDGET_EXPAND_THRESHOLD)
                .min(EPISTEMIC_BUDGET_EXPAND_CAP)
        } else if epistemic_uncertainty < EPISTEMIC_BUDGET_CONTRACT_THRESHOLD {
            EPISTEMIC_BUDGET_CONTRACT_BASE + epistemic_uncertainty * EPISTEMIC_BUDGET_CONTRACT_RAMP
        } else {
            1.0
        };
        let stillness_budget_scale = {
            let ss_coord =
                self.ethics_engine.last_harmony_coordinates()[HARMONY_INDEX_SACRED_STILLNESS];
            if ss_coord > STILLNESS_BUDGET_THRESHOLD {
                1.0 - (ss_coord - STILLNESS_BUDGET_THRESHOLD).min(STILLNESS_BUDGET_CONTRACT_CAP)
            } else {
                1.0
            }
        };
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
        let base_budget_us = self
            .config
            .attention_budget_override_us
            .unwrap_or(ATTENTION_BUDGET_US);
        let attention_budget_us = (base_budget_us as f64
            * neuromod_attention_alloc as f64
            * depth_budget_scale
            * epistemic_budget_scale as f64
            * stillness_budget_scale
            * coherence_velocity_budget_scale) as u64;

        // Active Rest Mode
        {
            let ss_coord =
                self.ethics_engine.last_harmony_coordinates()[HARMONY_INDEX_SACRED_STILLNESS];
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
                if let Some(ref mut replay) = self.memory.episodic_persistence.replay {
                    replay.boost_recent_consolidation(0.15);
                }
            }
        }

        // Moral attractor dampening
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

        (
            attention_budget_exceeded,
            predictive_budget_gated,
            epistemic_budget_scale,
        )
    }

    /// Self-model accuracy tracking, confidence crash detection, PE variance modulation,
    /// quantum coherence feedback, and foveation coupling.
    ///
    /// Returns `(confidence_crash_detected, lr_frozen)` for result assembly.
    fn track_self_model_and_detect_crash(
        &mut self,
        urgency: super::CycleUrgency,
        perception: &PerceptionPhaseResult,
    ) -> (bool, bool) {
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
        if self.stats.total_cycles % super::thresholds::SELF_MODEL_PREDICTION_INTERVAL == 0
            && self.carryover.history.self_model_prediction.is_none()
        {
            self.carryover.history.self_model_prediction =
                Some((self.stats.total_cycles, self.prediction_confidence, urgency));
            self.stats.self_model_predictions_made += 1;
        }

        // Self-model accuracy → proposal weighting (Friston 2010).
        if self_model_accuracy > self.threshold_overrides.self_model_weight_high() {
            self.scale_confidence(
                "self_model_weight_hi",
                (1.0 + SELF_MODEL_WEIGHT_BONUS) as f32,
            );
        } else if self_model_accuracy < SELF_MODEL_WEIGHT_LOW_THRESHOLD {
            self.scale_confidence("self_model_weight_lo", SELF_MODEL_WEIGHT_PENALTY as f32);
        }

        // Confidence crash detector → emergency stabilization (Cools et al. 2008).
        let confidence_crash_detected;
        let lr_frozen;
        {
            let prev_conf = self.carryover.quality.prev_confidence_for_crash;
            let current_conf = self.prediction_confidence;
            let drop = prev_conf - current_conf;
            let effective_crash_threshold = if self.behavior.flow_state.in_flow {
                self.threshold_overrides.confidence_crash_threshold()
                    * CONFIDENCE_CRASH_FLOW_MULTIPLIER
            } else {
                self.threshold_overrides.confidence_crash_threshold()
            };
            confidence_crash_detected = drop > prev_conf * effective_crash_threshold
                && prev_conf > super::thresholds::CONFIDENCE_CRASH_MIN_PRIOR
                && self.stats.total_cycles > DYNAMICS_STARTUP_WARMUP_CYCLES;
            if confidence_crash_detected {
                let freeze_duration = if self.carryover.urgency.mode_stability_counter
                    < super::thresholds::MODE_STABILITY_GRACE_THRESHOLD
                {
                    super::thresholds::CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES
                } else {
                    CONFIDENCE_CRASH_FREEZE_CYCLES
                };
                self.carryover.quality.crash_freeze_remaining = freeze_duration;
                self.adjust_exploration_pri(
                    "confidence_crash",
                    CONFIDENCE_CRASH_EXPLORATION_BOOST,
                    Priority::Safety,
                );
                tracing::debug!(
                    "Confidence crash detected: {prev_conf:.3} → {current_conf:.3} (drop={drop:.3}), \
                     freezing LR for {freeze_duration} cycles"
                );
            }
            lr_frozen = self.carryover.quality.crash_freeze_remaining > 0;
            if lr_frozen {
                let frozen_lr = self.feedback_state.cycle_start_lr() as f32;
                self.set_lr("crash_freeze", frozen_lr);
                self.carryover.quality.crash_freeze_remaining -= 1;
            }
            if self.carryover.quality.crash_freeze_remaining > 0 {
                self.scale_threshold("crash_binding_relax", 0.95);
            }
        }

        // PE variance → confidence modulation (Yu & Dayan 2005).
        let pe_variance = self.stats.avg_prediction_error_sq
            - self.stats.avg_prediction_error * self.stats.avg_prediction_error;
        let pe_variance = pe_variance.max(0.0);
        if pe_variance > PE_VARIANCE_THRESHOLD
            && self.stats.total_cycles > DYNAMICS_POST_BOOT_CYCLES
        {
            let variance_dampen = 1.0
                - (pe_variance - PE_VARIANCE_THRESHOLD).min(PE_VARIANCE_MAX_EFFECT)
                    * PE_VARIANCE_DAMPEN_SCALE;
            self.scale_confidence_pri("pe_variance", variance_dampen, Priority::Homeostatic);
        }

        // Quantum coherence → exploration boost.
        if self.carryover.consciousness.quantum_coherence > QUANTUM_COHERENCE_THRESHOLD {
            let coherence_boost = (self.carryover.consciousness.quantum_coherence
                - QUANTUM_COHERENCE_THRESHOLD) as f32
                * QUANTUM_COHERENCE_BOOST_SCALE;
            self.adjust_exploration_pri(
                "quantum_coherence",
                coherence_boost,
                Priority::Homeostatic,
            );
        }

        // Foveation → dynamics coupling (Corbetta & Shulman 2002).
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
                self.scale_exploration_pri(
                    "foveation_familiar",
                    FOVEATION_FAMILIAR_EXPLORATION_DAMPEN,
                    Priority::Homeostatic,
                );
                self.scale_confidence_pri(
                    "foveation_familiar",
                    FOVEATION_CONFIDENCE_BOOST,
                    Priority::Homeostatic,
                );
            } else if fov_count > 0 && fov_conf < FOVEATION_HIGH_CONFIDENCE_THRESHOLD {
                self.scale_lr_pri(
                    "foveation_novel",
                    FOVEATION_NOVEL_LR_BOOST,
                    Priority::Homeostatic,
                );
            }
        }
        #[cfg(not(feature = "foveation"))]
        let _ = perception; // suppress unused warning

        (confidence_crash_detected, lr_frozen)
    }

    /// Apply emotional homeostasis: pull valence toward neutral, arousal toward target.
    ///
    /// Incorporates emotional inertia from the previous cycle: rapid valence/arousal
    /// shifts are dampened by blending in the prior state, creating smoother
    /// emotional trajectories that resist oscillation.
    /// Science: Sokolov (1963) — habituation creates resistance to rapid shifts;
    /// Cannon (1929) — homeostatic regulation must be gradual to avoid overshoot.
    ///
    /// Returns (valence_pull, arousal_pull, pull_strength).
    fn apply_emotional_homeostasis(&mut self) -> (f32, f32, f32) {
        let unified_emo = self.unification_engine.emotional.state();
        let curr_v = unified_emo.valence as f32;
        let curr_a = unified_emo.arousal as f32;

        // Emotional inertia: resist rapid swings by blending toward previous state.
        // This closes the feedback loop — last_emotion_valence/arousal (written at
        // cycle end) are now read back to create momentum resistance.
        let prev_v = self.carryover.history.last_emotion_valence;
        let prev_a = self.carryover.history.last_emotion_arousal;
        let inertia = HOMEOSTASIS_EMOTIONAL_INERTIA;

        // Blend current emotion toward previous: dampens rapid changes.
        // When inertia=0.15: 85% current + 15% previous — mild smoothing.
        let smoothed_v = curr_v * (1.0 - inertia) + prev_v * inertia;
        let smoothed_a = curr_a * (1.0 - inertia) + prev_a * inertia;

        let pull_mult = match self.carryover.urgency.urgency {
            super::CycleUrgency::Cruise => self.threshold_overrides.homeostasis_pull_cruise(),
            super::CycleUrgency::Normal => HOMEOSTASIS_PULL_NORMAL,
            super::CycleUrgency::Critical => HOMEOSTASIS_PULL_CRITICAL,
        };

        let v_pull = -smoothed_v * HOMEOSTASIS_PULL_VELOCITY_SCALE * pull_mult;
        let a_pull =
            (HOMEOSTASIS_AROUSAL_TARGET - smoothed_a) * HOMEOSTASIS_PULL_AROUSAL_SCALE * pull_mult;
        self.behavior.emotion_contagion.valence = (smoothed_v + v_pull).clamp(-1.0, 1.0);

        self.stats.avg_valence_homeostasis = self.stats.avg_valence_homeostasis
            * VALENCE_HOMEOSTASIS_MOMENTUM
            + v_pull.abs() * VALENCE_HOMEOSTASIS_ALPHA;

        // Store for next cycle's inertia computation.
        self.carryover.history.last_emotion_valence = self.behavior.emotion_contagion.valence;
        self.carryover.history.last_emotion_arousal = smoothed_a;

        (v_pull, a_pull, pull_mult)
    }
}

mod memory_binding;
mod planning;
mod training;
