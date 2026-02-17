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
pub use drives::*;

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
pub use types::*;

pub mod stats;
pub use stats::*;

pub mod builder;
pub use builder::*;

pub mod executor;
pub use executor::*;

// ── Private submodules ──────────────────────────────────────────────────────
mod training;
use training::AsyncTrainerHandle;

mod temporal_network;

mod metrics_provider;

mod identity_integration;

// ── Impl-block submodules (split from this file) ────────────────────────────
mod accessors;
mod constructor;
mod cycle;
mod helpers;
mod moral;
mod prediction;
pub mod virtual_body;

// ── Imports (only what the struct definitions below require) ─────────────────
use crate::causal::CausalLoopEnhancer;
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::consciousness::attention_schema::AttentionSchema;
use crate::consciousness::consciousness_resonance::ResonanceAnalyzer;
use crate::consciousness::gwt_integration::UnifiedGlobalWorkspace;
use crate::consciousness::narrative_self::NarrativeSelfModel;
use crate::consciousness::predictive_self::PredictiveSelfModel;
use crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer;
use crate::consciousness::temporal_consciousness::TemporalConsciousnessAnalyzer;
use crate::consciousness::embodied_cognition::EmbodiedConsciousnessAnalyzer;
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::AutopoieticConsciousness;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
use crate::consciousness::dream::DreamEngine;
use crate::exploration::SurpriseExplorationBridge;
use crate::wisdom::meta_cognition::MetaCognitiveLayer;
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use crate::consciousness::primitive_discovery::PrimitiveDiscoveryService;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::dynamics::cfc_coherence::CfCCoherenceBridge;
use crate::dynamics::temporal_signatures::TemporalSignatureEncoder;
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::MemoryCoordinator;
use crate::memory::semantic_memory::SemanticMemory;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
use crate::voice::voice_feedback::VoiceFeedbackBridge;
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

    /// Error history for trend detection
    error_history: VecDeque<f32>,

    /// Last compressed state (for creating experience)
    last_state: Option<Vec<f32>>,

    /// Last prediction (for experience)
    last_prediction: Option<Vec<f32>>,

    /// Start time for cycles/second calculation
    start_time: Instant,

    /// Is currently consolidating (background learning)
    is_consolidating: bool,

    /// Coherence bridge for bidirectional CfC↔consciousness feedback
    coherence_bridge: CfCCoherenceBridge,

    /// Voice feedback bridge for voice→CfC feedback
    voice_feedback_bridge: VoiceFeedbackBridge,

    /// Temporal signature encoder for consciousness pattern detection
    temporal_signature_encoder: TemporalSignatureEncoder,

    /// Current adaptive behavior based on consciousness state
    adaptive_behavior: AdaptiveBehavior,

    /// Prediction confidence (0.0 to 1.0)
    /// Decays during uncertain states, grows with accurate predictions
    prediction_confidence: f32,

    /// Flow state tracker
    /// Detects and maintains flow state for optimal cognitive engagement
    flow_state: FlowState,

    /// Emotion contagion tracker
    /// Emotional content influences consciousness patterns
    emotion_contagion: EmotionContagion,

    /// Curiosity drive for novelty seeking
    /// Triggers exploration when predictions are too accurate
    curiosity_drive: CuriosityDrive,

    /// Self-reflection for meta-learning
    /// Periodically analyzes and adjusts internal thresholds
    self_reflection: SelfReflection,

    // ═══════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE: Consciousness Unification Engine
    // ═══════════════════════════════════════════════════════════════════════
    /// Thalamic router for cognitive depth selection
    /// Routes inputs to Reflex/Cortical/DeepThought paths based on novelty and urgency
    thalamic_router: ThalamicRouter,

    /// Consciousness Unification Engine - integrates all consciousness subsystems
    /// Provides: EmotionalBridge (VAD emotions), CausalReasoning, DialoguePipeline
    /// This replaces simple EmotionContagion with full VAD emotional tracking
    unification_engine: ConsciousnessUnificationEngine,

    /// Current cognitive routing depth (from Thalamus)
    /// Determines how deep the cognitive processing should go
    cognitive_depth: CognitiveDepth,

    /// Active Inference Bridge for precision-weighted prediction
    /// Connects MAGI Loop calibration to control signals via PAC tracking
    active_inference_bridge: ActiveInferenceBridge,

    /// Closed Learning Loop for strategy-based behavioral adaptation
    /// Implements the paradigm: Learning -> Behavioral Change
    closed_learning_loop: ClosedLearningLoop,

    /// Episodic Memory Bridge for memory encoding and recall during cycles
    episodic_memory: EpisodicMemoryBridge,

    /// Goal System Bridge for goal-directed attention modulation
    goal_system: GoalSystemBridge,

    /// World Model Bridge for hierarchical grounded prediction
    world_model: WorldModelBridge,

    /// FEP Active Inference Agent for full perception-action loop
    fep_agent: ActiveInferenceAgent,

    /// Enhanced FEP Bridge with motor system integration
    /// Provides learning signals and motor command outputs
    enhanced_fep_bridge: EnhancedFEPBridge,

    /// Current learning signal from FEP (for downstream systems)
    fep_learning_signal: f32,

    /// FEP-driven learning rate boost (applied during CfC training step)
    fep_lr_boost: f32,

    /// Conversation coherence tracker for degradation detection
    coherence_tracker: ConversationCoherenceTracker,

    /// Stability regime processor: CfC dynamics for primitives
    /// Frequently-used primitives crystallize, rarely-used stay fluid
    stability_regime: StabilityRegimeProcessor,

    /// Discovery service for finding new primitives seeded by crystallization events
    discovery_service: PrimitiveDiscoveryService,

    /// Semantic Memory: HDC-based similarity lookup for CfC contextual learning
    /// Stores (HDC vector, prediction error) pairs and retrieves similar past inputs
    /// to modulate learning rate - high error on similar inputs -> boost learning
    semantic_memory: SemanticMemory,

    /// Memory Coordinator: cross-tier signal broadcaster
    /// Bridges episodic and semantic memory with shared consciousness signals,
    /// handles graduation from working memory to episodic storage.
    memory_coordinator: MemoryCoordinator,

    /// Neural bridge for projecting pre-computed embeddings (e.g. BGE-M3)
    /// directly into HDC space via a trained linear probe.
    /// Only available when the `neural-bridge` feature is enabled and
    /// probe weights exist on disk.
    #[cfg(feature = "neural-bridge")]
    neural_bridge: Option<NeuralBridge>,

    /// Background training thread handle (when `config.async_training` is true
    /// and the backend is CfC).  `None` for synchronous training or HdcLtc backend.
    async_trainer: Option<AsyncTrainerHandle>,

    /// Causal loop enhancer for discovering causal structure in (input, output) pairs.
    /// When enabled via `config.causal_enhancement`, this:
    /// - Tracks recent (input, output) pairs
    /// - Periodically runs causal discovery
    /// - Weights attention based on discovered causal parents
    /// - Suggests interventions for exploration
    causal_enhancer: Option<CausalLoopEnhancer>,

    /// Episodic memory replay for high-Phi moment consolidation.
    /// When enabled via `config.episodic_replay`, stores high-consciousness episodes
    /// and periodically replays them to reinforce important patterns.
    phi_episodic_replay: Option<crate::memory::episodic_replay::EpisodicMemory>,

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
    // MORAL ALGEBRA: Compositional Ethical Reasoning
    // ═══════════════════════════════════════════════════════════════════════
    /// Moral Algebra for compositional ethical reasoning using HDC
    /// Encodes moral primitives (AGENT, PATIENT, ACTION, INTENT, CONSENT, OBLIGATION, MAGNITUDE)
    /// and provides judgment operations for action evaluation
    moral_algebra: MoralAlgebra,

    /// Moral Parser for extracting ethical primitives from natural language input
    /// Detects consent, intent, magnitude, and negation from text
    moral_parser: MoralParser,

    /// Last moral evaluation result (for tracking and learning)
    last_moral_judgment: Option<MoralJudgmentSummary>,

    /// Primitive-Belief Bridge: maps 9-tier primitives to active inference beliefs
    /// Computes per-tier prediction errors and TD signals for learning
    primitive_belief_bridge: PrimitiveBeliefBridge,

    /// Previous cycle's primitive consciousness state for prediction error computation
    prev_primitive_state: Option<PrimitiveConsciousnessState>,

    /// Surprise-driven exploration bridge for FEP-based exploration.
    /// Tracks prediction errors and triggers exploration when surprise
    /// exceeds an adaptive threshold. Modulates curiosity drive.
    surprise_bridge: Option<SurpriseExplorationBridge>,

    /// Prefrontal cortex for executive control and working memory.
    /// When enabled, maintains a working memory of recent inputs and
    /// gates learning/exploration when memory utilization is high.
    prefrontal: Option<PrefrontalCortex>,

    /// Meta-cognitive self-model layer.
    /// When enabled, tracks prediction error tendencies and uses
    /// self-model accuracy to modulate learning rate.
    meta_cognition: Option<MetaCognitiveLayer>,

    /// Narrative self-model for autobiographical identity.
    /// When enabled, maintains a three-level self-model (proto/core/autobio)
    /// and tracks self-Φ (integrated information of the self-model).
    narrative_self: Option<NarrativeSelfModel>,

    /// Predictive self-model for action safety evaluation.
    /// When enabled, predicts future self-states and evaluates action safety.
    predictive_self: Option<PredictiveSelfModel>,

    /// Attention schema (AST) for self-modeling attention state.
    /// When enabled, tracks attention focus, shifts, and generates control signals.
    attention_schema: Option<AttentionSchema>,

    /// Global Workspace Theory integration.
    /// When enabled, submits encodings to workspace for conscious broadcast.
    gwt: Option<UnifiedGlobalWorkspace>,

    /// Consciousness resonance monitor.
    /// When enabled, extracts harmonic modes from Phi time-series.
    consciousness_resonance: Option<ResonanceAnalyzer>,

    /// Quantum coherence observer.
    /// When enabled, monitors CfC hidden states for superposition richness.
    quantum_coherence: Option<QuantumCoherenceAnalyzer>,

    /// Temporal consciousness analyzer.
    /// When enabled, tracks Phi trajectory, continuity, Husserlian time,
    /// and temporal identity coherence across cycles.
    temporal_consciousness: Option<TemporalConsciousnessAnalyzer>,

    /// Embodied cognition analyzer.
    /// When enabled, bridges virtual body interoceptive state to body schema,
    /// sensorimotor engine, and affordance detection.
    embodied_cognition: Option<EmbodiedConsciousnessAnalyzer>,

    /// Narrative-GWT integration (consciousness governance capstone).
    /// When enabled, provides coherence veto, value checking, goal alignment
    /// via a unified NarrativeSelf + GWT + PredictiveSelf integration.
    narrative_gwt: Option<NarrativeGWTIntegration>,

    /// Counterfactual Dream Engine for learning from surprise.
    /// When enabled, records high-surprise events during waking cycles and
    /// dreams during Cruise urgency to discover Phi-improving alternative actions.
    dream_engine: Option<DreamEngine>,

    /// Whether narrative-GWT vetoed the previous cycle (suppresses learning this cycle)
    narrative_veto_active: bool,

    /// Relational Phi from dyad computation (set externally by Symthaea facade).
    /// Blended into unified_phi at 15% weight when > 0.
    relational_phi: f64,

    /// External reward signal injected by environment (0.0 = none).
    /// Blended with internal prediction-error-based reward at 50% weight.
    /// Resets to 0.0 after consumption.
    external_reward: f32,

    /// Virtual body adapter for embodied cognition.
    /// When enabled, maps cognitive signals to interoceptive states and produces
    /// a phi_modulation factor that scales consciousness from somatic feedback.
    virtual_body: Option<virtual_body::VirtualBody>,

    /// Previous cycle's body phi modulation (fed back into next cycle's unified_phi)
    prev_body_phi_modulation: f64,

    /// Previous cycle's embodied cognition phi modulation (fed back into unified_phi)
    prev_embodied_phi_modulation: f64,

    /// Consecutive cycles with prediction error below learning threshold.
    /// Used by CycleUrgency to enter Cruise mode after 10+ stable cycles.
    consecutive_low_error: u32,

    /// Previous cycle's resonance frequency (fed back into CfC delta_t modulation)
    /// Science: Buzsáki (2006) — neural oscillations modulate processing speed
    prev_resonance_frequency: f64,

    /// Previous cycle's quantum coherence level (fed back into exploration boost)
    /// Science: Lambert (2013) — quantum coherence enhances biological search
    prev_quantum_coherence: f64,

    /// MCE consciousness-level LR boost (decays 10%/cycle between MCE firings)
    /// Science: Dehaene (2014) — conscious access improves encoding
    mce_lr_boost: f32,

    /// Master Consciousness Equation (MCE) — comprehensive consciousness metric.
    /// C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × weighted_sum × S × ρ(t) × M × N × Soc
    /// Runs every 10th cycle to provide richer consciousness measurement than Phi alone.
    master_equation: MasterConsciousnessEquation,

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
}

// MetricsProvider impl is in metrics_provider.rs
// MFDI identity impl is in identity_integration.rs

#[cfg(test)]
mod tests;
