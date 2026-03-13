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
pub(crate) mod biorhythm_manager;
pub(crate) mod consciousness_engine;
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
pub(crate) mod ethics_engine;
pub(crate) mod feedback_state;
pub(crate) mod fep_module;
pub(crate) mod gwt_manager;
mod helpers;
pub(crate) mod managers;
mod moral;
pub mod motor_output_bridge;
pub(crate) mod neuromod_manager;
pub(crate) mod neuromodulators;
mod phase_results;
mod prediction;
pub(crate) mod primitive_tier;
pub(crate) mod self_model_tier;
pub(crate) mod social_manager;
pub(crate) mod substrate_manager;
pub use substrate_manager::SubstrateTransitionRecord;
pub(crate) mod subsystem_trait;
#[allow(dead_code)] // Registry of tuning constants — many reserved for future wiring
pub(crate) mod thresholds;
pub(crate) mod virtual_body;
pub(crate) mod voice_coherence_bridge;

#[cfg(feature = "ssm_language")]
pub(crate) mod broca_bridge;

#[cfg(feature = "canvas")]
pub(crate) mod canvas_bridge;

#[cfg(feature = "physics-bridge")]
pub(crate) mod physics_integration;
#[cfg(feature = "physics-bridge")]
pub use physics_integration::ParetoContext;

#[cfg(feature = "mycelix")]
pub use managers::governance_manager::{GovernanceEvent, GovernanceEventKind, GovernanceOutcome};

pub mod calibration;
pub mod math_service;

#[cfg(feature = "nurture")]
pub mod nurture_bridge;

#[cfg(feature = "humanoid")]
pub mod motor_bridge;

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
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
// PredictiveMind now owned by ConsciousnessStateManager
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use crate::consciousness::primitive_discovery::PrimitiveDiscoveryService;
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
use crate::consciousness::recursive_improvement::DreamFeedbackBridge;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
// CfCCoherenceBridge + TemporalSignatureEncoder now owned by VoiceCoherenceBridge
// SurpriseExplorationBridge moved to fep_module.rs
// MoralAlgebra + MoralParser now solely owned by EthicsEngine
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::MemoryCoordinator;
use crate::memory::semantic_memory::SemanticMemory;
use crate::mycelix::KosmicSong;
// HumanPartnerModel + PhiDyadCalculator now owned by SocialManager
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
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

    /// Last prediction (for experience)
    last_prediction: Option<Vec<f32>>,

    /// Start time for cycles/second calculation
    start_time: Instant,

    /// Is currently consolidating (background learning)
    is_consolidating: bool,

    /// Voice-coherence bridge: CfC coherence + voice feedback + temporal signatures.
    voice_coherence: voice_coherence_bridge::VoiceCoherenceBridge,

    /// Current adaptive behavior based on consciousness state
    adaptive_behavior: AdaptiveBehavior,

    /// Prediction confidence (0.0 to 1.0)
    /// Decays during uncertain states, grows with accurate predictions
    prediction_confidence: f64,

    /// Flow state tracker
    /// Detects and maintains flow state for optimal cognitive engagement
    flow_state: FlowState,

    /// Emotion contagion tracker
    /// Emotional content influences consciousness patterns
    emotion_contagion: EmotionContagion,

    /// Curiosity drive for novelty seeking
    /// Triggers exploration when predictions are too accurate
    curiosity_drive: CuriosityDrive,

    // NOTE: self_reflection moved to self_model_tier.self_reflection (Phase 3, Step 5)

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

    /// Consolidated FEP / Active Inference subsystem (10 fields -> 1).
    fep: fep_module::FepModule,

    /// Decoupled feedback state: attributed proposals for prediction_confidence
    /// and fep_lr_boost (Phase 2.2 Great Refactor).
    feedback_state: feedback_state::FeedbackState,

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

    /// Resonator Memory for factorized episodic recall.
    /// Stores episodes as bound (content ⊗ valence ⊗ phi_level) hypervectors
    /// with growing semantic codebook. Factorization decomposes bundled recalls
    /// into clean content/valence/phi components for richer context priming.
    resonator_memory: Option<crate::dynamics::resonator::ResonatorMemory>,

    /// Neural bridge for projecting pre-computed embeddings (e.g. BGE-M3)
    /// directly into HDC space via a trained linear probe.
    /// Only available when the `neural-bridge` feature is enabled and
    /// probe weights exist on disk.
    #[cfg(feature = "neural-bridge")]
    neural_bridge: Option<NeuralBridge>,

    /// Background embedding channel for Qwen3 semantic encoding.
    /// Runs a dedicated thread that produces 1024D embeddings, projected
    /// to BinaryHV via HdcBridge. Non-blocking: submits current input,
    /// collects previous cycle's result.
    #[cfg(feature = "semantic-encoder")]
    semantic_embedding_channel: Option<symthaea_embeddings::channel::EmbeddingChannel>,

    /// JL projection bridge for semantic embeddings → BinaryHV.
    #[cfg(feature = "semantic-encoder")]
    semantic_hdc_bridge: Option<symthaea_embeddings::HdcBridge>,

    /// Pending response receiver from the semantic embedding channel.
    /// Swapped each cycle: previous cycle's rx is consumed, new rx installed.
    /// Wrapped in Mutex to satisfy Sync bound (MetricsProvider).
    #[cfg(feature = "semantic-encoder")]
    pending_semantic_rx: std::sync::Mutex<
        Option<std::sync::mpsc::Receiver<symthaea_embeddings::channel::EmbedResponse>>,
    >,

    /// Last semantic embedding projected to continuous HDC space (16,384D).
    /// Fed to the ethics engine for moral topology trajectory analysis,
    /// giving genuine semantic resolution vs N-gram fallback.
    #[cfg(feature = "semantic-encoder")]
    last_semantic_continuous: Option<Vec<f32>>,

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
    /// Last moral evaluation result (for tracking and learning)
    last_moral_judgment: Option<MoralJudgmentSummary>,

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

    /// Self-model subsystems: narrative, predictive, attention schema, meta-cognition.
    self_model_tier: self_model_tier::SelfModelTierManager,

    /// GWT manager: workspace + memory flag + perception counter.
    gwt_mgr: gwt_manager::GwtManager,

    /// Consciousness monitoring tier: resonance, quantum, temporal, embodied,
    /// thermodynamics, phenomenal binding, hierarchical free energy.
    consciousness_monitors: consciousness_monitor_tier::ConsciousnessMonitorTier,

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

    /// Contextual harmony weighting for domain-aware ethical reasoning.
    contextual_weights: Option<crate::consciousness::contextual_weights::ContextualWeights>,

    /// Phi-weighted attention routing with adaptive thresholds.
    phi_attention: Option<crate::consciousness::phi_attention::AdaptiveThresholds>,

    /// Negation detector for moral/value text preprocessing.
    negation_detector: Option<crate::consciousness::negation_detector::NegationDetector>,

    /// All primitive-consciousness-gated subsystems, grouped into a single manager.
    /// See `primitive_tier::PrimitiveTierManager` for field list.
    primitive_tier: primitive_tier::PrimitiveTierManager,

    // ═══════════════════════════════════════════════════════════════════════
    // SUPPORT INTELLIGENCE: Predictive diagnostics + knowledge federation
    // ═══════════════════════════════════════════════════════════════════════
    /// Predictive engine for zero-click proactive support (telemetry → free energy alerts).
    #[cfg(feature = "support")]
    support_predictive_engine: Option<symthaea_support::predictive::PredictiveEngine>,

    /// Knowledge manager for article graduation and cognitive update absorption.
    /// Drives federation graduation checks and knowledge search during triage.
    #[cfg(feature = "support")]
    support_knowledge_manager: Option<symthaea_support::knowledge::KnowledgeManager>,

    /// Triage engine for ticket classification and prioritization.
    /// Classifies current input every cycle (lightweight keyword match).
    #[cfg(feature = "support")]
    support_triage_engine: Option<symthaea_support::triage::TriageEngine>,

    /// Privacy manager for federation sharing tier enforcement.
    /// Gates outbound knowledge federation based on SharingTier.
    #[cfg(feature = "support")]
    support_privacy_manager: Option<symthaea_support::privacy::PrivacyManager>,

    /// Action engine for autonomous remediation proposals.
    /// Proposes and gates actions based on autonomy level.
    /// Consumed by bridge dispatch when conductor events are wired.
    #[cfg(feature = "support")]
    #[allow(dead_code)] // RESERVED(feature-support): autonomy-aware action engine
    support_action_engine: Option<symthaea_support::actions::ActionEngine>,

    /// Cycle counter for amortizing support subsystem updates.
    #[cfg(feature = "support")]
    support_cycle_counter: u64,

    /// State carried over between consecutive cycles (phi modulations, veto flags,
    /// urgency hysteresis, MCE boost, etc.). Reset via `CycleCarryover::default()`.
    carryover: CycleCarryover,

    /// Soul: Eight Harmonies value alignment for moral evaluation.
    /// When present, evaluates action alignment against core values
    /// and integrates experiences for long-term value learning.
    soul: Option<crate::soul::Soul>,

    /// Attention visualizer for debugging attention flow.
    /// When present, captures attention snapshots each cycle for
    /// ASCII heatmaps, JSON export, and Graphviz flow graphs.
    attention_visualizer: Option<crate::visualization::AttentionVisualizer>,

    /// Social manager: social signals + phi-dyad + partner model + ring buffers.
    pub(crate) social_mgr: SocialManager,

    /// User state inference for adaptive response generation.
    /// When enabled, infers user cognitive load, frustration, and engagement from input text.
    user_state: Option<crate::user_state_inference::UserStateInference>,

    /// Resonant speech generator: adapts response complexity to user cognitive load.
    /// Uses neuromod bath signals + USI to determine response profile each cycle.
    /// Science: Ritter et al. (2019) — adaptive complexity reduces cognitive overload.
    resonant_speech: crate::resonant_speech::ResonantSpeech,

    /// Physiology coherence field — consciousness integration via hormone modulation.
    /// Tracks coherence state, applies hormone effects from neuromod bath each cycle.
    coherence_field: Option<crate::physiology::CoherenceField>,

    /// Virtual body adapter for embodied cognition.
    /// When enabled, maps cognitive signals to interoceptive states and produces
    /// a phi_modulation factor that scales consciousness from somatic feedback.
    virtual_body: Option<virtual_body::VirtualBody>,

    /// Nurture/attachment bridge — Bowlby attachment -> neuromodulator modulation.
    /// When enabled, models caregiver presence/absence and modulates oxytocin, NE,
    /// 5-HT, DA, adenosine based on attachment dynamics each cycle.
    #[cfg(feature = "nurture")]
    pub(crate) nurture_attachment: Option<nurture_bridge::NurtureAttachmentBridge>,

    /// Vision bridge: frame → attention-boosted HDC encoding.
    #[cfg(feature = "vision-manifold")]
    pub(super) vision_bridge: Option<symthaea_vision_manifold::VisionBridge>,
    /// Latest frame buffer for vision processing (injected externally or from mock).
    #[cfg(feature = "vision-manifold")]
    pub(super) vision_frame_buffer: Option<Vec<u8>>,

    /// Cross-manifold predictor: vision→cognitive Hebbian mapping.
    #[cfg(feature = "vision-manifold")]
    pub(super) cross_manifold_predictor: Option<symthaea_vision_manifold::CrossManifoldPredictor>,

    /// Foveation bridge: dorsal surprise → ventral recognition dispatch.
    /// When enabled, receives salient patches from vision manifold and dispatches
    /// background ventral recognition, feeding results into GWT.
    #[cfg(feature = "foveation")]
    pub(super) foveation_manager: Option<std::sync::Mutex<symthaea_foveation::FoveationManager>>,

    /// Broca SSM language center: consciousness-gated thought-to-text.
    /// When enabled via `ssm_language` feature + `enable_broca_language` config,
    /// generates text from HDC-encoded thoughts with epistemic/emotional gating.
    #[cfg(feature = "ssm_language")]
    pub(crate) broca_manager: Option<broca_bridge::BrocaManager>,

    /// Most recent Broca-generated text, drained into `CycleResult.language_output`
    /// each cycle. `None` when Broca is disabled or gated by low consciousness.
    #[cfg(feature = "ssm_language")]
    pub(crate) last_broca_text: Option<String>,

    /// Canvas living topology: consciousness-driven SVG generation.
    /// When enabled via `canvas` feature, generates real-time topology SVGs
    /// from cognitive telemetry with EMA-smoothed aesthetic mapping.
    #[cfg(feature = "canvas")]
    pub(crate) canvas_manager: Option<canvas_bridge::CanvasManager>,

    /// Most recent canvas SVG, drained into `CycleResult.canvas_svg` each cycle.
    #[cfg(feature = "canvas")]
    pub(crate) last_canvas_svg: Option<String>,

    /// Buffer of PsiAttestationRecords ready for governance bridge consumption.
    /// Populated when `config.enable_psi_attestation` is true.
    /// Capacity bound: attestation_buffer_capacity (max 256) — evict before push.
    psi_attestation_buffer: VecDeque<PsiAttestationRecord>,

    /// Sliding window of FEP↔MCTS policy agreement for adaptive temperature control.
    /// Science: Friston & Parr (2020) — policy agreement modulates exploration/exploitation.
    /// Capacity bound: POLICY_WINDOW_SIZE (20) — evict before push.
    policy_agreement_window: VecDeque<bool>,

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
    knowledge_manager: Option<crate::knowledge::KnowledgeManager>,

    /// Experience integration bus for principled signal tracking and harmonic reasoning.
    /// Bridges cognitive loop signals to Eight Harmonies wisdom system.
    experience_bus: Option<crate::experience::ExperienceBus>,

    /// School bridge for curriculum-aware learning recommendations.
    /// When present (and `school_learning` feature enabled), recommends objectives
    /// with predicted Phi gain from CfC-powered O(1) lookahead.
    /// Co-gated with `school_learning` feature.
    #[cfg(feature = "school_learning")]
    school_bridge: Option<crate::school::School>,

    /// Causal consciousness: HSIC-based causal attention weighting.
    /// Provides causal-strength attention maps for encoding interpretation.
    /// Richer than CausalLoopEnhancer — uses HSIC independence testing.
    causal_consciousness: Option<crate::intelligence::CausalConsciousness>,

    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached).
    pub(crate) thermodynamic_load: f32,

    /// Affective bias: cognitive temperature (0.0 to 2.0).
    pub(crate) mood_temperature: f32,

    /// Neuromodulator manager: bath, calibration, phase tracking, drift monitoring.
    /// Groups 8 neuromod-related fields into a single manager.
    pub(crate) neuromod: neuromod_manager::NeuromodManager,

    /// Somatic error bridge: converts infrastructure failures into felt stress.
    /// Lock poisoning, task panics, DB errors → arousal, thermodynamic load, tau slowdown.
    pub(crate) somatic_bridge: crate::infrastructure::somatic_error_bridge::SomaticErrorBridge,

    /// Pain channel sender for distributing to subsystems.
    /// Subsystems clone this to report infrastructure errors.
    pub(crate) pain_tx: Option<crate::infrastructure::somatic_error_bridge::PainSender>,

    /// Subsystem output collector (Phase 2.3 staged computation model).
    /// Collects SubsystemOutput proposals during Phase B (COMPUTE),
    /// integrates them in Phase C for consensus-averaged state updates.
    /// Currently in dual-write bridge mode alongside direct mutations.
    subsystem_collector: subsystem_trait::OutputCollector,

    /// Last cycle snapshot (Phase 2.3) for telemetry and debugging.
    /// Built at the start of each cycle (Phase A: OBSERVE).
    last_snapshot: Option<subsystem_trait::CycleSnapshot>,

    /// Unified Consciousness Engine: wraps SpectralMIP + MultiModal + EquationV2 + Pipeline
    /// into a single `measure()` call per cycle with co-prime interval scheduling.
    /// Runs **alongside** existing inline code (additive wiring — old code not removed yet).
    consciousness_engine: consciousness_engine::ConsciousnessEngine,

    /// Substrate independence manager: consolidates feasibility, validation overlay,
    /// speed/scale modulation, and telemetry into a single cohesive struct.
    pub(super) substrate_manager: substrate_manager::SubstrateManager,

    /// Physics bridge integration: HDC semantic search for physics analogies.
    /// When enabled, blends physics-informed HDC vectors into CfC dynamics each cycle.
    #[cfg(feature = "physics-bridge")]
    physics_integration: Option<physics_integration::PhysicsIntegration>,

    /// Cycle at which consciousness weights first converged (0 = not yet).
    convergence_cycle: usize,

    /// Unified Ethics Engine: wraps MoralParser + MoralAlgebra + ValueEvaluator + Harmonies
    /// into a single `evaluate()` call per cycle with co-prime interval scheduling.
    /// Runs **alongside** existing inline code (additive wiring — old code not removed yet).
    ethics_engine: ethics_engine::EthicsEngine,

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

    /// Perception Manager: attention budget, coherence tracking, Yerkes-Dodson regulation.
    /// Implements CognitiveSubsystem — proposals fed into OutputCollector.
    perception_manager: managers::PerceptionManager,

    /// Governance Manager: Mycelix governance events → neuromod contagion, confidence,
    /// exploration. Implements CognitiveSubsystem at interval 37. Feature-gated behind `mycelix`.
    #[cfg(feature = "mycelix")]
    governance_mgr: managers::GovernanceManager,

    /// Integrity Manager: BLAKE3 attestation, temporal consistency, behavioral canaries.
    /// Runs tamper detection at co-prime intervals. Feature-gated behind `integrity`.
    #[cfg(feature = "integrity")]
    integrity_manager: crate::integrity::IntegrityManager,

    /// Cantor broadcast buffer: CRHVs created from GWT broadcasts for dream consolidation.
    /// When a thought becomes "conscious" (enters workspace and is broadcast), it gets
    /// wrapped as a Cantor Recursive Hypervector preserving multi-scale structure.
    /// During dream consolidation, the CantorCleanupEngine factorizes these through
    /// the resonator codebook, preventing metacognitive amnesia (loss of faint peripheral
    /// Cantor layers at shift/27, shift/81).
    /// Science: Baars (1988) + Stickgold (2005) — conscious broadcast → fractal dreaming.
    cantor_broadcast_buffer: Vec<symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV>,

    /// Persistent Cantor cleanup engine: codebook accumulates across dream cycles.
    /// Unlike the previous ephemeral approach (rebuild each dream), this engine retains
    /// learned representations so dream consolidation genuinely strengthens memories
    /// over the brain's lifetime.
    /// Science: Born & Wilhelm (2012) — sleep spindle replay strengthens stable traces;
    ///          Walker (2009) — offline consolidation requires persistent memory stores.
    cantor_cleanup_engine: symthaea_core::hdc::cantor_resonator_cleanup::CantorCleanupEngine,

    /// Last GWT activation strength, used for adaptive CRHV depth.
    /// Stronger activations (higher workspace competition score) produce deeper fractals.
    /// Science: Dehaene et al. (2006) — ignition strength varies with stimulus salience;
    ///          stronger ignition recruits more recurrent cortical layers.
    cantor_last_activation: f32,

    /// EMA of dream consolidation surprise (|pre_ss − post_ss|).
    /// High surprise signals the codebook is encountering novel fractal structure.
    /// Science: Friston (2010) — free-energy surprise drives plasticity updates;
    ///          unexpected outcomes signal model inadequacy requiring learning.
    cantor_dream_surprise: f32,

    /// Resonance boost from coherent CRHV pairs in the broadcast buffer.
    /// When multiple CRHVs share high similarity (>0.8), the resulting coalition
    /// amplifies workspace integration — a "fractal choir" effect.
    /// Science: Edelman & Tononi (2000) — reentrant cortical signaling
    ///          creates dynamic coalitions; Singer (1999) — binding by synchrony.
    cantor_resonance_boost: f32,

    /// Motor output bridge: translates FEP MotorOutput commands into real-world
    /// actions (file I/O, shell commands, tests) via SimpleExecutor.
    /// When `None`, MotorOutput commands are no-ops (default behavior).
    motor_output_bridge: Option<motor_output_bridge::MotorOutputBridge>,

    /// Pending motor action request (string data for the next MotorOutput dispatch).
    /// Set externally before a cycle to provide path/content/args for motor commands.
    pub(crate) pending_motor_request: Option<motor_output_bridge::MotorActionRequest>,

    /// Last motor output result for FEP feedback.
    pub(crate) last_motor_result: Option<motor_output_bridge::MotorOutputResult>,

    /// Phi value used for motor gating in the most recent motor execution (telemetry).
    pub(crate) last_motor_phi: f64,

    /// Math Service: unified math dispatcher routing queries to Phase 1-3 solvers
    /// (linear algebra, root finding, quadrature, statistics, optimization, FFT,
    /// logic engine, constraint solver, geometry, graphs, differential equations).
    /// Tracks telemetry and stores solved-problem episodes for analogical retrieval.
    math_service: math_service::MathService,
}

// MetricsProvider impl is in metrics_provider.rs
// MFDI identity impl is in identity_integration.rs

#[cfg(test)]
mod tests;
