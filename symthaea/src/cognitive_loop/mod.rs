// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cognitive Loop Service - Emergent HDC↔CfC Integration
//!
//! This module implements the **bidirectional cognitive loop** that creates
//! emergent structure through prediction error minimization using Closed-form
//! Continuous-time (CfC) networks for O(1) temporal prediction.
//!
//! ## The Core Loop
//!
//! ```text
//! Cycle t:
//! 1. Input → HDC encode (with attention from t-1)
//! 2. CfC processes current HDC state (O(1) closed-form)
//! 3. CfC predicts next HDC state at multiple time horizons
//! 4. Prediction error computed (multi-scale)
//! 5. Error → CfC analytical gradient + HDC attention update
//! 6. Prediction sent to encoder for cycle t+1
//! 7. Post-processing: parallel subsystem updates via rayon::join
//!    Branch A: stability regime, semantic memory, causal enhancement
//!    Branch B: episodic memory, primitive-belief bridge, closed learning loop
//!    Sequential: episodic replay (needs CfC), memory coordinator
//! ```
//!
//! ## Why CfC (vs LTC)
//!
//! - **O(1) vs O(N)**: Closed-form solution avoids Euler integration
//! - **Multi-scale prediction**: Instant prediction at any future time
//! - **Analytical gradients**: No numerical approximation for training
//! - **Temporal "jumps"**: Can query t+10 without computing t+1..t+9
//!
//! ## Why This Creates Emergence
//!
//! - **Not hardcoded**: Structure emerges from prediction error, not rules
//! - **Biologically inspired**: Predictive coding in cortex
//! - **Self-organizing**: Attention weights evolve to minimize surprise
//! - **Continuous**: Service runs at 50Hz even without input
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::cognitive_loop::{CognitiveLoopService, CognitiveLoopConfig};
//!
//! let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default())?;
//!
//! // Process input
//! let result = service.cycle("cause leads to effect");
//!
//! // Check if learning is occurring
//! println!("Prediction error: {}", result.prediction_error);
//! println!("Attention variance: {}", service.stats().attention_variance);
//! ```
//!
//! ## Deterministic Reproducibility (Genesis Seeding)
//!
//! For full reproducibility, use the genesis phrase to seed all randomness:
//!
//! ```rust,ignore
//! use symthaea::cognitive_loop::CognitiveLoopBuilder;
//!
//! // Two instances with identical phrases produce identical outputs
//! let loop_a = CognitiveLoopBuilder::new()
//!     .with_genesis_phrase("We hold these truths...")
//!     .build()?;
//!
//! let loop_b = CognitiveLoopBuilder::new()
//!     .seeded("We hold these truths...")  // Alias for with_genesis_phrase
//!     .build()?;
//!
//! // loop_a and loop_b will produce identical outputs for identical inputs
//! ```
//!
//! ### What Genesis Seeds
//!
//! - **CfC network weights**: `cognitive_loop::cfc::cell_N`
//! - **HdcLtc bridge**: `cognitive_loop::hdc_ltc`
//! - **Exploration RNG**: `cognitive_loop::exploration`
//! - **Causal enhancer**: `causal_enhancer`
//!
//! All randomness flows through SHAKE-256 domain-separated streams, ensuring
//! identical phrase + domain produces identical values on any machine, forever.

// ── Public submodules (re-exported) ─────────────────────────────────────────
pub mod config;
pub use config::*;

pub mod drives;
pub(crate) use drives::*;

pub mod routing;
pub use routing::*;

pub mod snapshot;
pub use snapshot::*;

pub mod flow;
pub use flow::*;

pub mod learning;
pub use learning::*;

pub mod memory_bridge;
pub use memory_bridge::*;

pub mod goal_world;
pub use goal_world::*;

pub mod types;
pub(crate) use types::CycleState;
pub use types::*;

pub mod stats;
pub use stats::*;

pub mod builder;
pub use builder::*;

// ── Private submodules ──────────────────────────────────────────────────────
mod training;
use training::AsyncTrainerHandle;

mod temporal_network;

mod metrics_provider;

mod identity_integration;

// ── Impl-block submodules (split from this file) ────────────────────────────
mod accessors;
pub(crate) mod behavioral_synthesis;
pub(crate) mod biorhythm_manager;
#[cfg(feature = "mycelix")]
pub(crate) mod broca_factcheck;
pub(crate) mod cantor_dream_manager;
pub(crate) mod consciousness_engine;
pub(crate) mod consciousness_execution;
pub(crate) mod consciousness_monitor_tier;
pub(crate) mod consciousness_state_manager;
mod constructor;
mod cycle;
mod cycle_consciousness;
mod cycle_late_consciousness;
mod cycle_neuromod_phase;
mod cycle_phase_dynamics;
mod cycle_phase_feedback;
mod cycle_phase_output;
mod cycle_phase_perception;
mod cycle_quality;
mod cycle_strategy;
mod cycle_subsystems;
pub(crate) mod episodic_persistence_manager;
pub(crate) mod ethics_engine;
pub(crate) mod ethics_values_manager;
pub(crate) mod feature_integration_manager;
pub mod feedback_state;
pub(crate) mod fep_module;
pub(crate) mod gwt_manager;
mod helpers;
mod imagination;
pub(crate) mod language_comm_manager;

pub use imagination::ImagineFutureError;
pub(crate) mod managers;
pub(crate) mod memory_consolidation_manager;
pub(crate) mod memory_execution;
mod moral;
pub mod motor_output_bridge;
pub(crate) mod motor_rendering_manager;
pub(crate) mod neuromod_manager;
pub(crate) mod neuromodulators;
mod phase_results;
mod prediction;
pub(crate) mod primitive_tier;
pub mod radiation_environment;
pub(crate) mod self_model_tier;
pub(crate) mod sensorimotor_execution;
pub(crate) mod social_manager;
pub(crate) mod substrate_manager;
#[cfg(feature = "support")]
pub(crate) mod support_manager;
pub(crate) mod vision_sensory_manager;
pub use substrate_manager::SubstrateTransitionRecord;
pub(crate) mod subsystem_trait;
pub(crate) mod threshold_overrides;
pub(crate) mod thresholds;

#[cfg(feature = "epistemic_auditor")]
pub(crate) mod epistemic_auditor;
pub(crate) mod virtual_body;
pub(crate) mod voice_coherence_bridge;

pub mod broca_bridge;
pub mod broca_lite;
pub mod llm_language_channel;
pub mod voice_channel;

#[cfg(feature = "canvas")]
pub(crate) mod canvas_bridge;

#[cfg(feature = "creative")]
pub(crate) mod creative_bridge;

#[cfg(feature = "physics-bridge")]
pub(crate) mod physics_integration;
#[cfg(feature = "physics-bridge")]
pub use physics_integration::ParetoContext;

#[cfg(feature = "analogy-engine")]
pub(crate) mod analogy_integration;

#[cfg(feature = "ucl-frames")]
pub(crate) mod ucl_frame_integration;

pub(crate) mod thermodynamic_integration;
pub(crate) mod thermodynamic_physics_bridge;
pub(crate) mod thermodynamic_state;

#[cfg(feature = "mycelix")]
pub use broca_factcheck::{
    BrocaFactcheckBridge, BrocaModulation, FactCheckResult, FactCheckVerdict, FactcheckChannels,
    FactcheckTelemetry,
};
#[cfg(feature = "mycelix")]
pub use managers::governance_manager::{GovernanceEvent, GovernanceEventKind, GovernanceOutcome};

pub use ethics_engine::EthicalVerdict;
pub use managers::network_service_bridge::{
    forward_affective_state, forward_federated_round, FederatedCoordinatorHandle,
    NetworkServiceBridge, NetworkServiceBridgeHandle,
};
pub use managers::swarm_manager::{SwarmEvent, SwarmTelemetry};
pub use subsystem_trait::{output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput};

#[cfg(feature = "advanced-manufacturing")]
pub use managers::fabrication_manager::{
    FabricationEvent, FabricationEventKind, FabricationManager, FabricationTelemetry,
};

#[cfg(feature = "mesh")]
pub use managers::radio_dispatcher::{
    ConsciousRoutingDecision, ConsciousnessAwareRouter, DiscoveryBeacon, OfflineExperience,
    OfflineExperienceKind, StoreAndForward, ThreatObservation,
};
#[cfg(feature = "mesh")]
pub use managers::{
    CompressedDelta, NetworkHealth, PayloadClass, PayloadClassifier, RadioTier, RoutingDecision,
    SpectrumManager, SpectrumObservation, SpectrumTelemetry,
};

pub mod calibration;
#[cfg(feature = "mathematics")]
pub mod math_epistemic;
#[cfg(feature = "mathematics")]
pub mod math_service;

#[cfg(feature = "nurture")]
pub mod nurture_bridge;

#[cfg(any(
    feature = "humanoid",
    feature = "helicopter",
    feature = "flight",
    feature = "vehicle",
    feature = "auv",
    feature = "manipulator",
    feature = "exoskeleton",
    feature = "surgical",
    feature = "orbital",
    feature = "quadruped",
    feature = "subterranean",
    feature = "infrastructure",
    feature = "scavenger",
    feature = "agribot",
    feature = "biota",
    feature = "clime",
    feature = "phone"
))]
pub mod motor_bridge;

#[cfg(feature = "safety-agents")]
pub mod civic_crisis_detector;
#[cfg(feature = "sentinel")]
pub mod collective_immunity;
pub mod defense;
#[cfg(feature = "safety-agents")]
pub mod guardian;
pub mod safety_alert;
#[cfg(feature = "safety-agents")]
pub mod safety_enforcement;
#[cfg(feature = "sentinel")]
pub mod threat_memory;

pub mod life_support;
pub mod power_budget;

// ── Imports (only what the struct definitions below require) ─────────────────
// AffectiveBridge now owned by ConsciousnessStateManager
use crate::brain::prefrontal::PrefrontalCortex;
use crate::causal::CausalLoopEnhancer;
// AttentionSchema now owned by SelfModelTierManager
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::AutopoieticConsciousness;
// ConsciousnessMonitorTier now owns: ResonanceAnalyzer, ConsciousnessThermodynamicsAnalyzer,
// EmbodiedConsciousnessAnalyzer, HierarchicalFreeEnergy, QuantumCoherenceAnalyzer,
// TemporalConsciousnessAnalyzer, TemporalSynchronizationAnalyzer
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
// CrossModalBinder now owned by ConsciousnessStateManager
use crate::consciousness::dream::DreamEngine;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
// ActiveInferenceAgent, EnhancedFEPBridge moved to fep_module.rs
// UnifiedGlobalWorkspace now owned by GwtManager
// MasterConsciousnessEquation now owned by ConsciousnessExecution
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
// PredictiveMind now owned by ConsciousnessStateManager
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
// PrimitiveDiscoveryService + StabilityRegimeProcessor now owned by MemoryAndConsolidationManager
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
use crate::consciousness::recursive_improvement::DreamFeedbackBridge;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
// CfCCoherenceBridge + TemporalSignatureEncoder now owned by VoiceCoherenceBridge
// SurpriseExplorationBridge moved to fep_module.rs
// MoralAlgebra + MoralParser now solely owned by EthicsEngine
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
// MemoryCoordinator + SemanticMemory now owned by MemoryAndConsolidationManager
use crate::mycelix::KosmicSong;
// HumanPartnerModel + PhiDyadCalculator now owned by SocialManager
// NeuralBridge now owned by FeatureIntegrationManager
use crate::safety::SafetyGateway;
// VoiceFeedbackBridge now owned by VoiceCoherenceBridge
// MetaCognitiveLayer now owned by SelfModelTierManager
use std::collections::VecDeque;
use std::time::Instant;
use symthaea_core::hdc::predictive_encoder::PredictiveHdcEncoder;

use temporal_network::TemporalNetwork;

// ── Data types ──────────────────────────────────────────────────────────────

/// Experience for replay buffer
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for experience replay
struct Experience {
    /// Compressed HDC state
    state: Vec<f32>,
    /// LTC prediction
    prediction: Vec<f32>,
    /// Actual next state (for learning)
    next_state: Option<Vec<f32>>,
    /// Prediction error
    error: f32,
    /// Importance weight
    importance: f32,
}

// SocialState + SocialCoherenceState moved to social_manager.rs
pub(crate) use social_manager::SocialManager;

// ── Primary service struct ──────────────────────────────────────────────────

/// The Cognitive Loop Service
///
/// Orchestrates the bidirectional HDC↔CfC loop for emergent cognition.
/// Supports both CfC and HdcLtcUnified networks for O(1) temporal prediction.
///
/// Implementation is split across submodules:
/// - `constructor`: `new()` and backend selection
/// - `cycle`: the main `cycle()` method with rayon-parallel post-processing
/// - `prediction`: multi-scale prediction, primitive state building, consolidation
/// - `moral`: moral algebra evaluation
/// - `helpers`: experience creation, stats updates, error trends, reset
/// - `accessors`: read-only and mutable accessors for all subsystems
/// - `metrics_provider`: MetricsProvider trait impl
/// - `identity_integration`: MFDI identity integration
pub struct CognitiveLoopService {
    /// Configuration
    config: CognitiveLoopConfig,

    /// Predictive HDC encoder
    encoder: PredictiveHdcEncoder,

    /// Temporal network (CfC or HdcLtcUnified)
    temporal_network: TemporalNetwork,

    /// Experience buffer for replay
    buffer: VecDeque<Experience>,

    /// Statistics
    stats: LoopStats,

    /// Error history for trend detection.
    /// Capacity bound: 100 elements — evict before push via pop_front.
    error_history: VecDeque<f32>,

    /// Last compressed state (for creating experience)
    last_state: Option<Vec<f32>>,

    /// Pre-allocated buffer holding a copy of `last_state` for the training
    /// phase.  Populated each cycle by `copy_last_state_to_training_buf()`
    /// before `create_experience` moves `last_state` into the replay buffer.
    /// Eliminates a per-cycle `Vec<f32>` allocation on the CfC hot path.
    training_state_buf: Option<Vec<f32>>,

    /// Last prediction (for experience)
    last_prediction: Option<Vec<f32>>,

    /// Start time for cycles/second calculation
    start_time: Instant,

    /// Is currently consolidating (background learning)
    is_consolidating: bool,

    /// Language & communication: voice coherence, Broca, user state.
    pub(crate) language_comm: language_comm_manager::LanguageAndCommunicationManager,

    /// Async voice synthesis: sends text to background thread, retrieves audio.
    /// None when voice synthesis is not configured.
    pub(crate) voice_synthesis: Option<voice_channel::VoiceSynthesisChannel>,

    /// Async LLM language: sends consciousness state to Gemma 4 for translation.
    /// None when LLM language is not configured. BrocaLite fills in immediately;
    /// LLM response upgrades the output for subsequent cycles.
    pub(crate) llm_language: Option<llm_language_channel::LlmLanguageChannel>,

    /// Behavioral synthesis group: flow, emotion, curiosity, adaptive behavior,
    /// thalamic routing, and social cognition.
    /// Extracted from CognitiveLoopService to reduce field count (Phase 5, Stage 5).
    pub(crate) behavior: behavioral_synthesis::BehavioralSynthesis,

    /// Prediction confidence (0.0 to 1.0)
    /// Decays during uncertain states, grows with accurate predictions
    prediction_confidence: f64,

    // NOTE: self_reflection moved to self_model_tier.self_reflection (Phase 3, Step 5)
    /// Consciousness Unification Engine - integrates all consciousness subsystems
    /// Provides: EmotionalBridge (VAD emotions), CausalReasoning, DialoguePipeline
    /// This replaces simple EmotionContagion with full VAD emotional tracking
    unification_engine: ConsciousnessUnificationEngine,

    /// Current cognitive routing depth (from Thalamus)
    /// Determines how deep the cognitive processing should go
    cognitive_depth: CognitiveDepth,

    /// Consolidated FEP / Active Inference subsystem (10 fields -> 1).
    fep: fep_module::FepModule,

    /// Latest mental simulation (imagination) result.
    #[cfg(feature = "vision-manifold")]
    pub(crate) last_mental_movie: Option<types::MentalMovie>,

    /// Decoupled feedback state: attributed proposals for prediction_confidence
    /// and fep_lr_boost (Phase 2.2 Great Refactor).
    feedback_state: feedback_state::FeedbackState,

    /// Conversation coherence tracker for degradation detection
    coherence_tracker: ConversationCoherenceTracker,

    /// Memory & consolidation: semantic memory, coordinator, resonator, stability, discovery.
    pub(crate) memory: memory_execution::MemoryExecution,

    /// Feature integrations: neural bridge, semantic encoder, school, causal, physics.
    pub(crate) feature_integ: feature_integration_manager::FeatureIntegrationManager,

    /// Background training thread handle (when `config.async_training` is true
    /// and the backend is CfC).  `None` for synchronous training or HdcLtc backend.
    async_trainer: Option<AsyncTrainerHandle>,

    /// Causal loop enhancer for discovering causal structure in (input, output) pairs.
    /// When enabled via `config.causal_enhancement`, this:
    /// - Tracks recent (input, output) pairs
    // causal_enhancer moved to memory_execution::MemoryExecution

    /// Episodic persistence: replay, database, flush guard, reasoning context.
    // episodic_persistence moved to memory_execution::MemoryExecution

    /// Conscious Reasoning Engine: unified 7-step reasoning cycle
    /// Composes epistemic conflict, temporal planning, counterfactual reasoning,
    /// and tool gating with tiered degradation (Tier 0/1/2).
    #[cfg(feature = "reasoning_engine")]
    reasoning_engine: Option<crate::consciousness::reasoning_engine::ConsciousReasoningEngine>,

    /// MFDI Bridge for identity verification and signed outputs
    /// Provides: request verification, output signing, capability gating
    #[cfg(feature = "identity")]
    mfdi_bridge: crate::identity::MfdiBridge,

    // ═══════════════════════════════════════════════════════════════════════
    // SAFETY GATEWAY: Pre-cognitive fast veto (Amygdala + HDC guardrails)
    // ═══════════════════════════════════════════════════════════════════════
    /// Metacognitive monitor for Phi trajectory anomaly detection.
    /// When enabled, observes Phi after each reasoning step and detects
    /// drops, plateaus, and oscillations that indicate reasoning degradation.
    metacognitive_monitor:
        Option<crate::consciousness::metacognitive_monitoring::MetacognitiveMonitor>,

    /// Safety gateway for pre-cognitive safety veto.
    /// When enabled, scans input before expensive HDC encoding and short-circuits
    /// on dangerous patterns. Combines fast regex (AmygdalaActor) and HDC-based
    /// forbidden-subspace checking (SafetyGuardrails).
    safety_gateway: Option<SafetyGateway>,

    // ═══════════════════════════════════════════════════════════════════════
    // MORAL ALGEBRA: Compositional Ethical Reasoning
    // (MoralParser + MoralAlgebra now owned by EthicsEngine)
    // ═══════════════════════════════════════════════════════════════════════
    /// Ethics and values manager: groups moral judgment, contextual weights,
    /// phi-attention routing, negation detection, and soul alignment.
    pub(crate) ethics_values: ethics_values_manager::EthicsAndValuesManager,

    /// Primitive-Belief Bridge: maps 9-tier primitives to active inference beliefs
    /// Computes per-tier prediction errors and TD signals for learning
    primitive_belief_bridge: PrimitiveBeliefBridge,

    /// Previous cycle's primitive consciousness state for prediction error computation
    prev_primitive_state: Option<PrimitiveConsciousnessState>,

    // NOTE: surprise_bridge moved into fep_module::FepModule
    /// Prefrontal cortex for executive control and working memory.
    /// When enabled, maintains a working memory of recent inputs and
    /// gates learning/exploration when memory utilization is high.
    prefrontal: Option<PrefrontalCortex>,

    /// Consciousness execution group: consciousness engine, monitors, GWT, self-model, MCE.
    /// Extracted from CognitiveLoopService to reduce field count (Phase 5, Stage 1).
    pub(crate) consciousness: consciousness_execution::ConsciousnessExecution,

    /// Narrative-GWT integration (consciousness governance capstone).
    /// When enabled, provides coherence veto, value checking, goal alignment
    /// via a unified NarrativeSelf + GWT + PredictiveSelf integration.
    narrative_gwt: Option<NarrativeGWTIntegration>,

    /// Counterfactual Dream Engine for learning from surprise.
    /// When enabled, records high-surprise events during waking cycles and
    /// dreams during Cruise urgency to discover Phi-improving alternative actions.
    dream_engine: Option<DreamEngine>,

    /// Dream feedback bridge: converts dream insights into action priors and
    /// confidence adjustments. Connects the DreamEngine's wisdom to MAGI Loop
    /// calibration — dream-discovered Phi improvements bias future action selection.
    #[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
    dream_feedback_bridge: DreamFeedbackBridge,

    /// Consciousness state manager: groups cross-modal binding, predictive processing,
    /// phi-attention gating, and affective bridge into a single manager.
    /// Science: Treisman (1996), Friston (2010), Tononi (2004), Russell (1980).
    pub(crate) consciousness_state: consciousness_state_manager::ConsciousnessStateManager,

    // contextual_weights, phi_attention, negation_detector moved to ethics_values_manager
    /// All primitive-consciousness-gated subsystems, grouped into a single manager.
    /// See `primitive_tier::PrimitiveTierManager` for field list.
    primitive_tier: primitive_tier::PrimitiveTierManager,

    /// Unified thermodynamic manager (interval 43): cross-couples Dissipative,
    /// Analyzer, HFE, and physics bridge. See `managers/thermodynamic_manager.rs`.
    pub(crate) thermodynamic_mgr: managers::thermodynamic_manager::ThermodynamicManager,

    // ═══════════════════════════════════════════════════════════════════════
    // SUPPORT INTELLIGENCE: Predictive diagnostics + knowledge federation
    // ═══════════════════════════════════════════════════════════════════════
    /// Support intelligence: predictive engine, knowledge, triage, privacy, actions.
    #[cfg(feature = "support")]
    pub(crate) support: support_manager::SupportManager,

    /// State carried over between consecutive cycles (phi modulations, veto flags,
    /// urgency hysteresis, MCE boost, etc.). Reset via `CycleCarryover::default()`.
    /// Exposed for property testing; prefer accessors for production code.
    pub carryover: CycleCarryover,

    // soul moved to ethics_values_manager
    /// Attention visualizer for debugging attention flow.
    /// When present, captures attention snapshots each cycle for
    /// ASCII heatmaps, JSON export, and Graphviz flow graphs.
    attention_visualizer: Option<crate::visualization::AttentionVisualizer>,

    // social_mgr moved to behavioral_synthesis::BehavioralSynthesis

    // user_state moved to language_comm_manager
    /// Resonant speech generator: adapts response complexity to user cognitive load.
    /// Uses neuromod bath signals + USI to determine response profile each cycle.
    /// Science: Ritter et al. (2019) — adaptive complexity reduces cognitive overload.
    resonant_speech: crate::resonant_speech::ResonantSpeech,

    /// Streaming inference engine: CfC-based real-time inference with batching.
    /// Pushes perception encodings each cycle, polls outputs for downstream use.
    /// Config: batch_accumulation=1, max_latency=32ms (cycle-aligned).
    streaming_inference: Option<crate::inference::StreamingInference>,

    /// Sensorimotor execution group: vision/sensory, motor rendering, somatic bridge,
    /// pain/thermal channels, embodiment bridge.
    /// Extracted from CognitiveLoopService to reduce field count (Phase 5, Stage 4).
    pub(crate) sensorimotor: sensorimotor_execution::SensoriMotorExecution,

    /// Live STT capture handle. When `Some`, the perception phase polls the
    /// mic each cycle and binds the resulting auditory HV into the input
    /// encoding. Sibling to vision — both are sensory input modalities.
    /// Opt-in: call `start_stt_capture()` after construction.
    #[cfg(feature = "voice-stt-live")]
    pub(crate) stt_capture: Option<crate::perception::MicCaptureHandle>,

    /// Optional IMU fusion module. When `Some`, the perception phase
    /// fuses `latest_imu_reading` into an auxiliary sensory HV and bundles
    /// it into the input encoding alongside STT/radio. Opt-in.
    #[cfg(feature = "sensor-imu")]
    pub(crate) imu_fusion: Option<Box<dyn crate::perception::sensor_fusion::ImuFusion>>,

    /// Most recent IMU reading. Pushed from an external source (hardware
    /// driver, ADB bridge, MAVLink adapter) via `inject_imu_reading()`.
    /// The perception phase reads it each cycle.
    #[cfg(feature = "sensor-imu")]
    pub(crate) latest_imu_reading: Option<crate::perception::sensor_fusion::ImuReading>,

    /// Nurture/attachment bridge — Bowlby attachment -> neuromodulator modulation.
    /// When enabled, models caregiver presence/absence and modulates oxytocin, NE,
    /// 5-HT, DA, adenosine based on attachment dynamics each cycle.
    #[cfg(feature = "nurture")]
    pub(crate) nurture_attachment: Option<nurture_bridge::NurtureAttachmentBridge>,

    // vision_bridge, vision_frame_buffer, cross_manifold_predictor, foveation_manager moved to sensorimotor_execution

    // broca_manager, last_broca_text moved to language_comm_manager

    // canvas_manager, last_canvas_svg moved to motor_rendering manager
    /// Buffer of PsiAttestationRecords ready for governance bridge consumption.
    /// Populated when `config.enable_psi_attestation` is true.
    /// Capacity bound: attestation_buffer_capacity (max 256) — evict before push.
    psi_attestation_buffer: VecDeque<PsiAttestationRecord>,

    /// Sliding window of FEP↔MCTS policy agreement for adaptive temperature control.
    /// Science: Friston & Parr (2020) — policy agreement modulates exploration/exploitation.
    /// Capacity bound: POLICY_WINDOW_SIZE (20) — evict before push.
    policy_agreement_window: VecDeque<bool>,

    // master_equation moved to consciousness_execution::ConsciousnessExecution
    /// Unified Living Mind: life-mind continuity integration (full_consciousness feature).
    /// Integrates autopoietic self-maintenance, enactive sense-making, affect, and prediction
    /// into a unified vitality/coherence state that measures "aliveness" of the system.
    #[cfg(feature = "full_consciousness")]
    unified_living_mind: UnifiedLivingMind,

    /// Autopoietic consciousness for self-production/maintenance (full_consciousness feature).
    /// Tracks boundary maintenance, self-model updates, and component health.
    #[cfg(feature = "full_consciousness")]
    autopoietic: AutopoieticConsciousness,

    /// Enactive cognition for embodied sense-making (full_consciousness feature).
    /// Tracks action-perception coupling, meaning emergence, and enacted world.
    #[cfg(feature = "full_consciousness")]
    enactive: EnactiveCognition,

    /// Biorhythm manager: circadian/ultradian rhythm + refresh counter.
    biorhythm_mgr: biorhythm_manager::BiorhythmManager,

    // phi_attention_gate moved to consciousness_state_manager
    /// Metrics collector for Prometheus-compatible observability.
    /// When present, records per-cycle consciousness, performance, and safety metrics
    /// for external monitoring dashboards.
    metrics_collector: Option<crate::infrastructure::MetricsCollector>,

    /// Knowledge engine: general-purpose reasoning infrastructure.
    /// Extracts structured facts from input, encodes as HDC vectors, stores in a
    /// temporal knowledge graph, builds causal DAG edges, and grows adaptive ontology.
    /// Science: Kanerva (2009) HDC, Pearl (2009) Causality, Carey (2009) conceptual change.
    // knowledge_manager moved to memory_execution::MemoryExecution

    // last_reasoning_context moved to episodic_persistence manager
    /// Experience integration bus for principled signal tracking and harmonic reasoning.
    /// Bridges cognitive loop signals to Eight Harmonies wisdom system.
    experience_bus: Option<crate::experience::ExperienceBus>,

    // school_bridge + causal_consciousness moved to feature_integ manager
    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached).
    pub(crate) thermodynamic_load: f32,

    /// Affective bias: cognitive temperature (0.0 to 2.0).
    pub(crate) mood_temperature: f32,

    /// Neuromodulator manager: bath, calibration, phase tracking, drift monitoring.
    /// Groups 8 neuromod-related fields into a single manager.
    pub(crate) neuromod: neuromod_manager::NeuromodManager,

    // somatic_bridge, pain_tx, thermal_bridge, thermal_tx moved to sensorimotor_execution
    /// Subsystem output collector (Phase 2.3 staged computation model).
    /// Collects SubsystemOutput proposals during Phase B (COMPUTE),
    /// integrates them in Phase C for consensus-averaged state updates.
    /// Currently in dual-write bridge mode alongside direct mutations.
    subsystem_collector: subsystem_trait::OutputCollector,

    /// Tracks consecutive panics per subsystem and disables repeat offenders.
    /// Subsystems that panic 3 times in a row are skipped for the session.
    subsystem_health: subsystem_trait::SubsystemHealthTracker,

    /// Last cycle snapshot (Phase 2.3) for telemetry and debugging.
    /// Built at the start of each cycle (Phase A: OBSERVE).
    last_snapshot: Option<subsystem_trait::CycleSnapshot>,

    // consciousness_engine moved to consciousness_execution::ConsciousnessExecution
    /// Substrate independence manager: consolidates feasibility, validation overlay,
    /// speed/scale modulation, and telemetry into a single cohesive struct.
    pub(super) substrate_manager: substrate_manager::SubstrateManager,

    /// Metabolic conductor for Mk0 hardware coordination.
    pub(crate) metabolic_conductor: Option<crate::embodiment::MetabolicConductor>,

    pub(crate) threshold_overrides: threshold_overrides::ThresholdOverrides,
    #[cfg(feature = "jepa")]
    pub(super) jepa_engine: Option<symthaea_jepa::JepaEngine>,

    /// Rolling window of per-cycle cortical activation maps for temporal analysis.
    /// Capacity: ~1000 cycles (~32s at 31Hz). Used for HRF convolution and EEG comparison.
    #[cfg(feature = "neural_validation")]
    pub(super) cortical_history:
        std::collections::VecDeque<symthaea_core::hdc::cortical_activation::CorticalActivationMap>,

    // physics_integration moved to feature_integ manager
    /// Cycle at which consciousness weights first converged (0 = not yet).
    convergence_cycle: usize,

    /// Lagged consciousness for governance gating — excludes recent governance feedback.
    /// Ring buffer of last `GOVERNANCE_CONSCIOUSNESS_LAG_SIZE` consciousness values;
    /// governance gates use the oldest to decorrelate the feedback loop:
    /// consciousness → governance → neuromod → consciousness.
    /// Science: Granger (1969) — temporal decorrelation breaks circular causation.
    governance_consciousness_lag: VecDeque<f64>,

    /// Unified Ethics Engine: wraps MoralParser + MoralAlgebra + ValueEvaluator + Harmonies
    /// into a single `evaluate()` call per cycle with co-prime interval scheduling.
    /// Runs **alongside** existing inline code (additive wiring — old code not removed yet).
    ethics_engine: ethics_engine::EthicsEngine,

    /// Last unified ethical verdict from `ethics_engine.evaluate()`.
    /// Checked before motor execution: `Blocked` prevents action, `Caution` caps confidence.
    pub(crate) last_ethics_verdict: ethics_engine::EthicalVerdict,

    /// External override for the ethical verdict. When `Some`, takes precedence
    /// over the ethics engine's output each cycle. Used by safety systems and
    /// integration tests to force a specific verdict for deterministic testing.
    /// Cleared only by an explicit `clear_ethics_override()` call.
    pub(crate) ethics_verdict_override: Option<ethics_engine::EthicalVerdict>,

    /// KosmicSong: Unified identity synthesizing Phi + Eight Harmonies + Epistemic Humility.
    /// Computed every cycle after consciousness_engine + ethics_engine settle.
    /// Outputs coherence_score (0.0-1.0) that gates FEP learning rate and exploration.
    kosmic_song: KosmicSong,

    /// Drive Manager: consolidates curiosity, boredom, flow, and exploration drives.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    drive_manager: managers::DriveManager,

    /// Memory Manager: consolidation pressure, retrieval quality, episodic gating.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    memory_manager: managers::MemoryManager,

    /// Learning Manager: FEP plasticity, dream consolidation, error trend gating.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    learning_manager: managers::LearningManager,

    /// Multimodal Manager: MCE gating for external gen models, moral action gating.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    multimodal_manager: managers::MultimodalManager,

    /// Perception Manager: attention budget, coherence tracking, Yerkes-Dodson regulation.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    perception_manager: managers::PerceptionManager,

    /// Soul Manager: value alignment → confidence, exploration, LR modulation.
    /// Implements CognitiveSubsystem at interval 43. Active when soul is enabled.
    soul_manager: Option<managers::SoulManager>,

    /// Governance Manager: Mycelix governance events → neuromod contagion, confidence,
    /// exploration. Implements CognitiveSubsystem at interval 37. Feature-gated behind `mycelix`.
    #[cfg(feature = "mycelix")]
    governance_mgr: managers::GovernanceManager,

    /// Broca Factcheck Bridge: Verifies Broca language output against the Mycelix
    /// knowledge graph. Extracts claims, submits for verification, modulates
    /// confidence/cadence based on verdicts. Feature-gated behind `mycelix`.
    #[cfg(feature = "mycelix")]
    factcheck_bridge: broca_factcheck::BrocaFactcheckBridge,

    /// Sacred Stillness: tracks known unknowns for epistemic humility.
    #[cfg(feature = "epistemic")]
    known_unknowns: Option<crate::consciousness::sacred_stillness::KnownUnknowns>,

    /// Swarm Manager: Peer consciousness signals → social buffering, affective contagion,
    /// collective Φ modulation. Implements CognitiveSubsystem at interval 41.
    swarm_manager: managers::SwarmManager,

    /// Muse Manager: Streaming consciousness-driven music synthesis.
    #[cfg(feature = "muse")]
    pub(crate) muse_manager: managers::MuseManager,

    /// Music Publisher: Background uploader — drains CompositionExports from
    /// muse_manager and POSTs WAV files to the Mycelix-Music HTTP API.
    #[cfg(feature = "muse")]
    pub(crate) music_publisher: managers::MusicPublisher,

    /// Holon Receiver: Desktop-side bridge accepting Soma WebSocket connections.
    /// Processes inbound SomaMessages (heartbeats, CVs, tasks, knowledge) and
    /// routes them to SwarmManager (peers), ReasoningManager (tasks), KnowledgeManager (offers).
    holon_receiver: crate::consciousness::holon_receiver::HolonReceiver,

    /// Receiver for Holon inbound messages from the HTTP layer.
    /// Created eagerly at construction; clone `holon_inbound_tx` to feed from HTTP handlers.
    holon_inbound_rx: std::sync::Mutex<
        Option<
            std::sync::mpsc::Receiver<(String, crate::consciousness::holon_receiver::SomaMessage)>,
        >,
    >,

    /// Sender half of the Holon inbound channel. Clone via `holon_inbound_sender()` to
    /// feed messages from HTTP handlers (HolonHttpState).
    holon_inbound_tx:
        std::sync::mpsc::Sender<(String, crate::consciousness::holon_receiver::SomaMessage)>,

    /// Receiver for swarm events from external async P2P layer.
    /// Drained non-blocking in Phase B before `swarm_manager.process()`.
    /// Created eagerly at construction; clone `swarm_event_tx` to inject events.
    swarm_event_rx:
        std::sync::Mutex<Option<std::sync::mpsc::Receiver<managers::swarm_manager::SwarmEvent>>>,

    /// Sender half of the swarm event channel. Clone via `swarm_event_sender()` to
    /// inject events from async components (NetworkService, Hyperfeel, mesh layer).
    swarm_event_tx: std::sync::mpsc::Sender<managers::swarm_manager::SwarmEvent>,

    /// Sender for safety-critical alerts requiring human attention.
    /// Bounded channel (capacity 32). The host application drains via
    /// `drain_safety_alerts()` and forwards to desktop notifications, monitoring, etc.
    safety_alert_tx: std::sync::mpsc::SyncSender<safety_alert::SafetyAlert>,
    /// Receiver for safety alerts. Wrapped in Mutex<Option<>> so the host can
    /// take ownership via `take_safety_alert_receiver()`.
    safety_alert_rx: std::sync::Mutex<Option<std::sync::mpsc::Receiver<safety_alert::SafetyAlert>>>,

    /// Handle for the federated coordinator task (if enabled).
    federation_handle:
        Option<crate::cognitive_loop::managers::network_service_bridge::FederatedCoordinatorHandle>,

    /// Live swarm network service retained after runtime wiring.
    /// Keeping this handle prevents the async bridge subscriptions from
    /// outliving the underlying service.
    network_service: Option<std::sync::Arc<crate::swarm::NetworkService>>,

    /// Spectrum Manager: Multi-band radio dispatch, AIMD congestion, delta compression.
    /// Implements CognitiveSubsystem at interval 53. Feature-gated behind `mesh`.
    #[cfg(feature = "mesh")]
    pub(crate) spectrum_manager: SpectrumManager,

    /// Consciousness-Aware Router: Routes mesh traffic by Phi, moral urgency,
    /// governance tier. Adaptive sharing cadence based on collective coherence.
    #[cfg(feature = "mesh")]
    pub(crate) consciousness_router: managers::radio_dispatcher::ConsciousnessAwareRouter,

    /// Store-and-Forward: Dream-consolidated reconnection for intermittent mesh nodes.
    /// Buffers offline experiences, consolidates on reconnect.
    #[cfg(feature = "mesh")]
    pub(crate) store_and_forward: managers::radio_dispatcher::StoreAndForward,

    /// CPG Manager: Kuramoto coupled oscillators for rhythmic motor timing.
    /// Implements CognitiveSubsystem at interval 59. Feature-gated behind `cpg`.
    #[cfg(feature = "cpg")]
    cpg_manager: managers::CpgManager,

    /// Spectral Twin Manager: frequency-domain analysis of CfC hidden state.
    /// Maintains rolling state history, computes band power / PAC / entropy.
    /// Implements CognitiveSubsystem at interval 67. Feature-gated behind `spectral_state`.
    #[cfg(feature = "spectral_state")]
    spectral_manager: managers::SpectralManager,

    /// Therapeutic Manager: client model, alliance, crisis detection, regulation.
    /// Implements CognitiveSubsystem at interval 11. Feature-gated behind `therapeutic`.
    #[cfg(feature = "therapeutic")]
    therapeutic_manager: managers::TherapeuticManager,

    /// Fabrication Manager: Cincinnati quality monitoring, ManufacturingTwin readings, PoGF.
    /// Implements CognitiveSubsystem at interval 47. Feature-gated behind `advanced-manufacturing`.
    #[cfg(feature = "advanced-manufacturing")]
    pub(crate) fabrication_manager: managers::FabricationManager,

    /// Glyph Manager: symbolic consciousness field, 11 Field Modality basis vectors,
    /// 70-glyph registry, developmental spiral tracking.
    /// Implements CognitiveSubsystem at interval 43. Feature-gated behind `glyph_codex`.
    #[cfg(feature = "glyph_codex")]
    glyph_manager: managers::GlyphManager,

    /// Time Manager: Mesh-wide time consensus from peer beacons.
    /// Implements CognitiveSubsystem at interval 23. Feature-gated behind `mesh`.
    #[cfg(feature = "mesh")]
    time_manager: managers::TimeManager,

    /// Sender for mesh outbound packets (beacons, name responses, content announces).
    /// ContinuousMind drains the paired receiver each tick via `drain_mesh_outbound()`.
    #[cfg(feature = "mesh")]
    mesh_outbound_tx: std::sync::mpsc::Sender<crate::swarm::mesh::MeshOutbound>,
    /// Receiver held until ContinuousMind claims it via `take_mesh_outbound_rx()`.
    #[cfg(feature = "mesh")]
    mesh_outbound_rx:
        std::sync::Mutex<Option<std::sync::mpsc::Receiver<crate::swarm::mesh::MeshOutbound>>>,

    /// Trust Manager: Web-of-trust graph with decay and violation detection.
    /// Implements CognitiveSubsystem at interval 29. Feature-gated behind `mesh-trust`.
    #[cfg(feature = "mesh-trust")]
    trust_manager: managers::TrustManager,

    /// Social Fabric Manager: Resonance graph for content diversity and echo-chamber detection.
    /// Implements CognitiveSubsystem at interval 31. Feature-gated behind `social-fabric`.
    #[cfg(feature = "social-fabric")]
    social_fabric_manager: managers::SocialFabricManager,

    /// Survival Manager: IoT sensor monitoring, demand forecasting, emergency detection.
    /// Implements CognitiveSubsystem at interval 47. Feature-gated behind `survival`.
    #[cfg(feature = "survival")]
    survival_manager: managers::SurvivalManager,

    /// Integrity Manager: BLAKE3 attestation, temporal consistency, behavioral canaries.
    /// Runs tamper detection at co-prime intervals. Feature-gated behind `integrity`.
    #[cfg(feature = "integrity")]
    integrity_manager: crate::integrity::IntegrityManager,

    /// Neuroevolution Manager: evolves CfC-HDC neural organisms via FEP fitness.
    /// Interval 71 (co-prime). Feature-gated behind `neuroevolution`.
    #[cfg(feature = "neuroevolution")]
    pub(crate) neuroevolution_manager: managers::NeuroevolutionManager,
    #[cfg(feature = "hypervisor")]
    pub(crate) hypervisor_manager: managers::HypervisorManager,

    /// Reasoning Manager: reasoning reliability → LR modulation, confidence, trend affect.
    /// Implements CognitiveSubsystem at interval 73. Feature-gated behind `reasoning_engine`.
    #[cfg(feature = "reasoning_engine")]
    reasoning_manager: managers::ReasoningManager,

    /// Language Manager: Broca quality feedback → confidence, LR, consolidation.
    /// Implements CognitiveSubsystem at interval 61. Feature-gated behind `ssm_language`.
    #[cfg(feature = "ssm_language")]
    language_manager: managers::LanguageManager,

    /// Vision Manager: visual surprise → exploration, attention, habituation.
    /// Implements CognitiveSubsystem at interval 17. Feature-gated behind `vision-manifold`.
    #[cfg(feature = "vision-manifold")]
    vision_manager: managers::VisionManager,

    /// Cantor dream: broadcast buffer, cleanup engine, activation, surprise, resonance.
    pub(crate) cantor_dream: cantor_dream_manager::CantorDreamManager,

    // motor_rendering moved to sensorimotor_execution
    /// Hierarchical bundler for per-region BinaryHV aggregation.
    /// Accumulates HDC encodings per cortical region, enabling structured
    /// role-bound aggregation and per-region recovery via XOR unbinding.
    pub(crate) hierarchical_bundler:
        Option<symthaea_core::hdc::hierarchical_bundle::HierarchicalBundler>,

    /// Pre-allocated input buffer for CfC temporal network step.
    /// Sized to `config.cfc_config.input_dim` at construction, reused each cycle
    /// to avoid per-cycle `Array1<f32>` heap allocation in `phase_dynamics`.
    cfc_input_buffer: ndarray::Array1<f32>,

    /// Math Service: unified math dispatcher routing queries to Phase 1-3 solvers
    /// (linear algebra, root finding, quadrature, statistics, optimization, FFT,
    /// logic engine, constraint solver, geometry, graphs, differential equations).
    /// Tracks telemetry and stores solved-problem episodes for analogical retrieval.
    #[cfg(feature = "mathematics")]
    math_service: math_service::MathService,

    /// Conjecture Engine: automated mathematical conjecture generation via symbolic
    /// regression (Ramanujan Protocol). Discovers closed-form formulas from numerical
    /// sequences, verifies them by bounded induction, and self-corrects structural
    /// tautologies. Features: GP + Nelder-Mead/L-BFGS, Pareto-optimal selection,
    /// OEIS lookup, FFT periodicity, Bayesian confidence, sensitivity analysis.
    #[cfg(feature = "mathematics")]
    conjecture_engine: symthaea_core::hdc::conjecture_engine::ConjectureEngine,

    /// Epistemic Auditor: DuckDB-backed consciousness telemetry audit trail.
    #[cfg(feature = "epistemic_auditor")]
    pub(crate) epistemic_auditor: Option<epistemic_auditor::EpistemicAuditor>,

    /// NRC-style safety agent for consciousness monitoring and operational gating.
    #[cfg(feature = "safety-agents")]
    pub(crate) safety_agent: crate::safety::SafetyAgent,
    /// Physical guardian posture state machine.
    #[cfg(feature = "safety-agents")]
    pub(crate) guardian_state: guardian::GuardianState,
    #[cfg(feature = "sentinel")]
    pub(crate) sentinel_manager: managers::SentinelManager,
    #[cfg(feature = "sentinel")]
    pub(crate) threat_memory: threat_memory::ThreatMemory,
    #[cfg(feature = "sentinel")]
    pub(crate) collective_immune_state: collective_immunity::CollectiveImmuneState,

    /// Defense actions proposed this cycle (populated by defense cascade).
    #[cfg(feature = "safety-agents")]
    pub(crate) defense_actions_proposed: u32,
    /// Defense actions that passed moral filter this cycle.
    #[cfg(feature = "safety-agents")]
    pub(crate) defense_actions_approved: u32,

    /// Civic crisis detector: monitors PE, safety level, Phi, arousal for
    /// community-level emergencies. Produces CivicCrisisEvent for Mycelix.
    #[cfg(feature = "safety-agents")]
    pub(crate) civic_crisis_detector: civic_crisis_detector::CivicCrisisDetector,

    /// Crisis events waiting to be forwarded to Mycelix civic bridge.
    /// Drained by `drain_pending_crisis_events()` from the external host.
    #[cfg(feature = "safety-agents")]
    pub(crate) pending_crisis_events: Vec<civic_crisis_detector::CivicCrisisEvent>,

    /// Aggregate security telemetry for the crypto/swarm stack.
    pub(crate) security_telemetry: crate::swarm::SecurityTelemetry,

    /// Scientific-method engine: observe → hypothesize → predict → test → update_beliefs.
    /// Accumulates Bayesian belief updates across cycles. Feature-gated behind
    /// `scientific_method`. The engine is seeded with a standing "input is coherent"
    /// hypothesis (id 0) whose posterior drifts with prediction error.
    #[cfg(feature = "scientific_method")]
    pub(crate) scientific_method_engine: crate::scientific_method::ScientificMethodEngine,
    // embodiment_bridge, last_proprioceptive_hv, embodiment_telemetry moved to sensorimotor_execution
}

// MetricsProvider impl is in metrics_provider.rs
// MFDI identity impl is in identity_integration.rs

impl CognitiveLoopService {
    /// Inject explicit FEP priors (Passport Route).
    pub fn inject_priors(&mut self, mean: Vec<f64>, precision: Vec<f64>) {
        self.fep.agent.inject_priors(mean, precision);
    }
}

#[cfg(test)]
mod tests;
