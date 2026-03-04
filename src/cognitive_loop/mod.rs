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
pub(crate) mod consciousness_engine;
pub(crate) mod consciousness_monitor_tier;
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
mod helpers;
pub(crate) mod managers;
mod moral;
pub(crate) mod neuromod_manager;
pub(crate) mod neuromodulators;
mod prediction;
pub(crate) mod primitive_tier;
pub(crate) mod self_model_tier;
pub(crate) mod subsystem_trait;
pub(crate) mod thresholds;
pub(crate) mod virtual_body;
pub(crate) mod substrate_manager;

#[cfg(feature = "physics-bridge")]
pub(crate) mod physics_integration;

pub mod calibration;

#[cfg(feature = "nurture")]
pub mod nurture_bridge;

#[cfg(feature = "humanoid")]
pub mod motor_bridge;

// ── Imports (only what the struct definitions below require) ─────────────────
use crate::brain::affective_bridge::AffectiveBridge;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::causal::CausalLoopEnhancer;
// AttentionSchema now owned by SelfModelTierManager
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::AutopoieticConsciousness;
// ConsciousnessMonitorTier now owns: ResonanceAnalyzer, ConsciousnessThermodynamicsAnalyzer,
// EmbodiedConsciousnessAnalyzer, HierarchicalFreeEnergy, QuantumCoherenceAnalyzer,
// TemporalConsciousnessAnalyzer, TemporalSynchronizationAnalyzer
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
use crate::consciousness::cross_modal_binding::CrossModalBinder;
use crate::consciousness::dream::DreamEngine;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::consciousness::gwt_integration::UnifiedGlobalWorkspace;
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
use crate::consciousness::predictive_processing::PredictiveMind;
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use crate::consciousness::primitive_discovery::PrimitiveDiscoveryService;
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
use crate::consciousness::recursive_improvement::DreamFeedbackBridge;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
use crate::dynamics::cfc_coherence::CfCCoherenceBridge;
use crate::dynamics::temporal_signatures::TemporalSignatureEncoder;
use crate::exploration::SurpriseExplorationBridge;
// MoralAlgebra + MoralParser now solely owned by EthicsEngine
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::MemoryCoordinator;
use crate::memory::semantic_memory::SemanticMemory;
use crate::partnership::{HumanPartnerModel, PhiDyadCalculator};
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
use crate::safety::SafetyGateway;
use crate::voice::voice_feedback::VoiceFeedbackBridge;
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

/// Social/external signal state.
pub(crate) struct SocialState {
    /// Relational Psi from dyad computation (15% blend weight into unified_psi).
    pub relational_psi: f64,
    /// External reward signal injected by environment (0.0 = none).
    pub external_reward: f32,
    /// Social trust level injected by Mind module's SocialCoherence (0.0–1.0).
    pub social_trust: f32,
    /// Social cooperation rate injected by Mind module's SocialCoherence (0.0–1.0).
    pub social_cooperation_rate: f32,
}

impl Default for SocialState {
    fn default() -> Self {
        Self {
            relational_psi: 0.0,
            external_reward: 0.0,
            social_trust: 0.5,
            social_cooperation_rate: 0.0,
        }
    }
}

/// Wrapper around social coherence state, used by cycle phases that
/// reference `self.social_coherence.social.*`.
pub(crate) struct SocialCoherenceState {
    pub social: SocialState,
}

impl Default for SocialCoherenceState {
    fn default() -> Self {
        Self {
            social: SocialState::default(),
        }
    }
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
    fep_lr_boost: f64,

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
    pending_semantic_rx:
        std::sync::Mutex<Option<std::sync::mpsc::Receiver<symthaea_embeddings::channel::EmbedResponse>>>,

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

    /// Surprise-driven exploration bridge for FEP-based exploration.
    /// Tracks prediction errors and triggers exploration when surprise
    /// exceeds an adaptive threshold. Modulates curiosity drive.
    surprise_bridge: Option<SurpriseExplorationBridge>,

    /// Prefrontal cortex for executive control and working memory.
    /// When enabled, maintains a working memory of recent inputs and
    /// gates learning/exploration when memory utilization is high.
    prefrontal: Option<PrefrontalCortex>,

    /// Self-model subsystems: narrative, predictive, attention schema, meta-cognition.
    self_model_tier: self_model_tier::SelfModelTierManager,

    /// Global Workspace Theory integration.
    /// When enabled, submits encodings to workspace for conscious broadcast.
    gwt: Option<UnifiedGlobalWorkspace>,

    /// GWT handler flag: memory consolidation requested via broadcast.
    gwt_memory_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// GWT handler counter: perception broadcast events consumed.
    gwt_perception_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,

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

    /// Predictive processing mind (hierarchical predictive coding + precision dynamics).
    /// When enabled, provides phi_modulation from free energy minimization.
    predictive_mind: Option<PredictiveMind>,

    /// Cross-modal binder for multi-modal integration.
    /// When enabled, binds HDC encodings across modalities and computes cross-modal Phi.
    cross_modal_binder: Option<CrossModalBinder>,

    /// Affective bridge for emotion-cognition coupling.
    /// When enabled, evaluates somatic marker signals from cognitive loop state.
    affective_bridge: Option<AffectiveBridge>,

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

    /// Soul: Seven Harmonies value alignment for moral evaluation.
    /// When present, evaluates action alignment against core values
    /// and integrates experiences for long-term value learning.
    soul: Option<crate::soul::Soul>,

    /// Attention visualizer for debugging attention flow.
    /// When present, captures attention snapshots each cycle for
    /// ASCII heatmaps, JSON export, and Graphviz flow graphs.
    attention_visualizer: Option<crate::visualization::AttentionVisualizer>,

    /// Social and external signal state (relational psi, reward, trust, cooperation).
    pub(crate) social: SocialState,

    /// Social coherence wrapper — used by cycle phases that reference
    /// `self.social_coherence.social.*` for relational psi access.
    pub(crate) social_coherence: SocialCoherenceState,

    /// Phi-Dyad calculator for relational consciousness.
    /// Computes Φ_dyad from recent AI + input HVs each cycle.
    phi_dyad: Option<PhiDyadCalculator>,

    /// Human partner model for relational state tracking.
    partner_model: Option<HumanPartnerModel>,

    /// Ring buffer of recent AI HDC states (last 4, for dyad computation).
    recent_ai_hvs: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,

    /// Ring buffer of recent input HDC states (last 4, as human proxy).
    recent_input_hvs: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,

    /// User state inference for adaptive response generation.
    /// When enabled, infers user cognitive load, frustration, and engagement from input text.
    user_state: Option<crate::user_state_inference::UserStateInference>,

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

    /// Chronobiology: circadian/ultradian rhythm modulation.
    /// Modulates learning rate (plasticity) and exploration (creativity) based on local time.
    /// Refreshed every 100 cycles to avoid unnecessary chrono calls.
    biorhythm: crate::chronobiology::Biorhythm,

    /// Cycle counter for biorhythm refresh (refreshes every 100 cycles).
    biorhythm_refresh_counter: usize,

    /// Phi-guided attention gate for consciousness-aware perception weighting.
    /// When present, weights perception inputs by their integrated information
    /// contribution, focusing processing on high-Phi signals.
    phi_attention_gate: Option<crate::attention::PhiAttentionGate>,

    /// Metrics collector for Prometheus-compatible observability.
    /// When present, records per-cycle consciousness, performance, and safety metrics
    /// for external monitoring dashboards.
    metrics_collector: Option<crate::infrastructure::MetricsCollector>,

    /// Experience integration bus for principled signal tracking and harmonic reasoning.
    /// Bridges cognitive loop signals to Seven Harmonies wisdom system.
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

    /// Pre-computed substrate feasibility [0,1] from config.substrate_type.
    /// Scales Equation V2 consciousness to reflect substrate limitations.
    substrate_feasibility: f64,

    /// Pending substrate transition description for telemetry.
    /// Populated by `reconfigure_substrate()`/`reconfigure_composition()`,
    /// drained into `CycleMetadata.substrate_transition` once per cycle.
    pending_substrate_transition: Option<String>,

    /// Honest evidence confidence for the current substrate (0.0–0.95).
    /// From SubstrateValidationFramework: biological=0.95, silicon=0.10, etc.
    substrate_honest_confidence: f64,

    /// Effective feasibility after validation overlay blending.
    /// When overlay disabled: equals substrate_feasibility.
    /// When enabled: substrate_feasibility × (floor + (1 − floor) × honest_confidence).
    substrate_effective_feasibility: f64,

    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    /// 1.0 when speed modulation is disabled.
    substrate_tau_factor: f32,

    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    /// Telemetry-only. 0.0 when speed modulation is disabled.
    substrate_scale_pressure: f32,

    /// Cycle at which consciousness weights first converged (0 = not yet).
    convergence_cycle: usize,

    /// Unified Ethics Engine: wraps MoralParser + MoralAlgebra + ValueEvaluator + Harmonies
    /// into a single `evaluate()` call per cycle with co-prime interval scheduling.
    /// Runs **alongside** existing inline code (additive wiring — old code not removed yet).
    ethics_engine: ethics_engine::EthicsEngine,

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
}

// MetricsProvider impl is in metrics_provider.rs
// MFDI identity impl is in identity_integration.rs

#[cfg(test)]
mod tests;
