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

pub mod executor;

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
mod cycle_consciousness;
mod cycle_late_consciousness;
mod cycle_subsystems;
mod helpers;
mod moral;
mod prediction;
pub(crate) mod virtual_body;

#[cfg(feature = "humanoid")]
pub mod motor_bridge;

// ── Imports (only what the struct definitions below require) ─────────────────
use crate::brain::affective_bridge::AffectiveBridge;
use crate::brain::prefrontal::PrefrontalCortex;
use crate::causal::CausalLoopEnhancer;
use crate::consciousness::attention_schema::AttentionSchema;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::autopoietic_consciousness::AutopoieticConsciousness;
use crate::consciousness::consciousness_resonance::ResonanceAnalyzer;
use crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer;
use crate::consciousness::consciousness_unification::ConsciousnessUnificationEngine;
use crate::consciousness::cross_modal_binding::CrossModalBinder;
use crate::consciousness::dream::DreamEngine;
use crate::consciousness::embodied_cognition::EmbodiedConsciousnessAnalyzer;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::enactive_cognition::EnactiveCognition;
use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::consciousness::gwt_integration::UnifiedGlobalWorkspace;
use crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy;
use crate::consciousness::master_consciousness_equation::MasterConsciousnessEquation;
use crate::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;
use crate::consciousness::narrative_self::NarrativeSelfModel;
use crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer;
use crate::consciousness::predictive_processing::PredictiveMind;
use crate::consciousness::predictive_self::PredictiveSelfModel;
use crate::consciousness::primitive_belief_bridge::PrimitiveBeliefBridge;
use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use crate::consciousness::primitive_discovery::PrimitiveDiscoveryService;
use crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer;
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
use crate::consciousness::recursive_improvement::DreamFeedbackBridge;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::consciousness::temporal_consciousness::TemporalConsciousnessAnalyzer;
#[cfg(feature = "full_consciousness")]
use crate::consciousness::unified_living_mind::UnifiedLivingMind;
use crate::dynamics::cfc_coherence::CfCCoherenceBridge;
use crate::dynamics::temporal_signatures::TemporalSignatureEncoder;
use crate::exploration::SurpriseExplorationBridge;
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::memory_coordinator::MemoryCoordinator;
use crate::memory::semantic_memory::SemanticMemory;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;
use crate::safety::SafetyGateway;
use crate::voice::voice_feedback::VoiceFeedbackBridge;
use crate::wisdom::meta_cognition::MetaCognitiveLayer;
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

    /// Consciousness thermodynamics analyzer.
    /// When enabled, analyzes entropy, free energy, and phase transitions
    /// from the 7 consciousness dimensions [Φ, B, W, A, R, E, K].
    consciousness_thermodynamics: Option<ConsciousnessThermodynamicsAnalyzer>,

    /// Phenomenal binding analyzer (temporal synchronization).
    /// When enabled, tracks phase coherence across consciousness dimensions
    /// to measure unified experience quality.
    phenomenal_binding: Option<TemporalSynchronizationAnalyzer>,

    /// Hierarchical free energy engine.
    /// When enabled, maintains a multi-level variational free energy hierarchy
    /// with precision-weighted prediction errors at each level.
    hierarchical_free_energy: Option<HierarchicalFreeEnergy>,

    /// Contextual harmony weighting for domain-aware ethical reasoning.
    contextual_weights: Option<crate::consciousness::contextual_weights::ContextualWeights>,

    /// Phi-weighted attention routing with adaptive thresholds.
    phi_attention: Option<crate::consciousness::phi_attention::AdaptiveThresholds>,

    /// Negation detector for moral/value text preprocessing.
    negation_detector: Option<crate::consciousness::negation_detector::NegationDetector>,

    /// Primitive consciousness decomposition for explainable consciousness.
    primitive_processor:
        Option<crate::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor>,

    /// Temporal consciousness analyzer using Allen's Interval Algebra.
    /// Records conscious intervals each cycle; detects causal chains and continuity gaps.
    /// Co-gated with `enable_primitive_consciousness`.
    temporal_analyzer:
        Option<crate::consciousness::temporal_primitives::ConsciousnessTemporalAnalyzer>,

    /// Lattice structure over the 9-tier primitive system.
    /// Computed once at startup; provides O(1) join/meet and structural metrics.
    /// Co-gated with `enable_primitive_consciousness`.
    primitive_lattice: Option<crate::consciousness::primitive_lattice::PrimitiveLattice>,

    /// Compositionality engine: algebraic composition of primitives (sequential, parallel, etc.).
    /// Co-gated with `enable_primitive_consciousness`.
    compositionality_engine: Option<crate::consciousness::compositionality::CompositionalityEngine>,

    /// Unified value evaluator: Seven Harmonies alignment scoring.
    /// Co-gated with `enable_primitive_consciousness`.
    value_evaluator: Option<crate::consciousness::unified_value_evaluator::UnifiedValueEvaluator>,

    /// Harmonic field: tracks strength of each of the Seven Fiduciary Harmonics.
    /// Co-gated with `enable_primitive_consciousness`.
    harmonic_field: Option<crate::consciousness::harmonics::HarmonicField>,

    /// Harmonic resolver: resolves conflicts between harmonics.
    /// Co-gated with `enable_primitive_consciousness`.
    harmonic_resolver: Option<crate::consciousness::harmonics::HarmonicResolver>,

    /// Primitive reasoner: HDC-based analogical reasoning with concept binding.
    /// Co-gated with `enable_primitive_consciousness`.
    primitive_reasoner: Option<crate::consciousness::primitive_reasoning::PrimitiveReasoner>,

    /// Adaptive reasoner: Q-learning-guided primitive selection for reasoning chains.
    /// Co-gated with `enable_primitive_consciousness`.
    adaptive_reasoner: Option<crate::consciousness::adaptive_reasoning::AdaptiveReasoner>,

    /// Phi validation framework: empirical validation of Phi against synthetic states.
    /// Co-gated with `enable_primitive_consciousness`. Expensive — runs rarely.
    phi_validation: Option<crate::consciousness::phi_validation::PhiValidationFramework>,

    /// Causal self-explanation: builds causal model of primitive→Φ relationships.
    /// Co-gated with `enable_primitive_consciousness`.
    causal_explainer: Option<crate::consciousness::causal_explanation::CausalExplainer>,

    /// Context-aware optimizer: dynamic Φ/Harmonic/Epistemic weighting by reasoning context.
    /// Co-gated with `enable_primitive_consciousness`.
    context_optimizer: Option<crate::consciousness::context_aware_evolution::ContextAwareOptimizer>,

    /// Evolution coordinator: co-evolves primitives and architecture via Thompson sampling.
    /// Replaces one-shot PrimitiveEvolution with stateful cross-generation tracking.
    /// Co-gated with `enable_primitive_consciousness`.
    evolution_coordinator: Option<crate::consciousness::evolution_bridge::EvolutionCoordinator>,

    /// Harmonies integrator: per-action ethical alignment gate using Seven Harmonies.
    /// Evaluates cycle actions against harmony embeddings for approval/rejection.
    /// Co-gated with `enable_primitive_consciousness`.
    harmonies_integrator: Option<crate::consciousness::harmonies_integration::HarmoniesIntegrator>,

    /// Semantic value embedder: value-aligned embeddings grounded in primitive tiers.
    /// Maps input embeddings to harmony-scored value-aware representations.
    /// Co-gated with `enable_primitive_consciousness`.
    semantic_value_embedder:
        Option<crate::consciousness::semantic_value_embedder::SemanticValueEmbedder>,

    /// Dissipative consciousness: Prigogine thermodynamic model for consciousness.
    /// Tracks entropy production, order parameters, and criticality distance.
    /// Co-gated with `enable_primitive_consciousness`.
    dissipative_consciousness:
        Option<crate::consciousness::dissipative_consciousness::DissipativeConsciousness>,

    /// Epistemic conflict detector: multi-theory conflict analysis (IIT vs GWT vs AST vs PP vs RPT vs 4E).
    /// Co-gated with `enable_primitive_consciousness`.
    epistemic_conflict_detector: Option<crate::consciousness::epistemic_conflict::ConflictDetector>,

    /// Theory calibrator for Φ_eff = Φ × R^γ reliability weighting.
    /// Co-gated with `enable_primitive_consciousness`.
    theory_calibrator: Option<crate::consciousness::epistemic_conflict::TheoryCalibrator>,

    /// Master Consciousness Equation v2: unified 7-theory consciousness formula C(t).
    /// Co-gated with `enable_primitive_consciousness`.
    consciousness_equation_v2:
        Option<crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2>,

    /// Hierarchical LTC: local circuits + global integrator for distributed temporal processing.
    /// Co-gated with `enable_primitive_consciousness`.
    hierarchical_ltc: Option<crate::consciousness::hierarchical_ltc::HierarchicalLTC>,

    /// Holographic consciousness analyzer: interference-based binding and holographic recall.
    /// Co-gated with `enable_primitive_consciousness`.
    holographic_analyzer:
        Option<crate::consciousness::consciousness_holography::HolographicConsciousnessAnalyzer>,

    /// Differentiable consciousness: gradient-based optimization of consciousness components.
    /// Provides ∂C/∂component gradients for identifying bottlenecks.
    /// Co-gated with `enable_primitive_consciousness`.
    differentiable_consciousness:
        Option<crate::consciousness::differentiable::DifferentiableConsciousness>,

    /// Affective consciousness analyzer: valence-arousal-dominance affect tracking.
    /// Processes stimuli into core affect, decays over time, learns affective markers.
    /// Co-gated with `enable_primitive_consciousness`.
    affective_consciousness:
        Option<crate::consciousness::affective_consciousness::AffectiveConsciousnessAnalyzer>,

    /// Unified consciousness pipeline: end-to-end sensory→consciousness pipeline.
    /// Combines HDC encoding, hierarchical LTC, binding, and master equation.
    /// Co-gated with `enable_primitive_consciousness`.
    unified_consciousness_pipeline:
        Option<crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline>,

    /// Multi-modal integration: phi-guided cross-modal binding with convergence zones.
    /// Co-gated with `enable_primitive_consciousness`.
    multi_modal_integrator:
        Option<crate::consciousness::multi_modal_integration::MultiModalIntegrator>,

    /// Primitive composition rules: domain-specific HDC binding operators.
    /// Selects TemporalPhysical, Mathematical, Consciousness, or CrossTier rule
    /// based on operand tiers, yielding semantically structured bindings.
    /// Co-gated with `enable_primitive_consciousness`.
    composition_rule_engine:
        Option<crate::consciousness::primitive_composition_rules::CompositionRuleEngine>,

    /// Synthetic states NSM grounding: maps current BinaryHV to consciousness states
    /// (Deep Anesthesia → Alert/Focused) via Natural Semantic Metalanguage primitives.
    /// Co-gated with `enable_primitive_consciousness`.
    synthetic_grounding:
        Option<crate::consciousness::synthetic_states::SyntheticStatesNSMGrounding>,

    /// Epistemic decision gate: evaluates input through Graceful Ignorance System
    /// to determine confidence vs uncertainty before acting.
    /// Co-gated with `enable_primitive_consciousness`.
    epistemic_gate: Option<crate::consciousness::gis_integration::EpistemicDecisionGate>,

    /// Meta-cognitive reasoner: reflects on its own reasoning context, evaluates
    /// strategy effectiveness, and learns meta-patterns across reasoning episodes.
    /// Co-gated with `enable_primitive_consciousness`.
    meta_cognitive_reasoner: Option<crate::consciousness::meta_reasoning::MetaCognitiveReasoner>,

    /// Code primitive router: consciousness-aware code reasoning via HDC primitives.
    /// Selects and caches code-tier primitives for code-related cognitive tasks.
    /// Co-gated with `enable_primitive_consciousness`.
    code_primitive_router: Option<crate::consciousness::code_primitives::CodePrimitiveRouter>,

    /// Empathic unification: resonant empathy via user state inference + mirroring.
    /// Processes input text to infer user emotional state, generates compassionate
    /// response modulation (tone, support intent, patience scaling).
    /// Co-gated with `enable_primitive_consciousness`.
    empathic_unification: Option<crate::consciousness::empathic_unification::EmpathicUnification>,

    /// Multi-objective evolution: Pareto-frontier consciousness optimization.
    /// Evolves primitives across 5 consciousness dimensions (Φ, ∇Φ, Entropy,
    /// Complexity, Coherence) using NSGA-II-inspired selection.
    /// Co-gated with `enable_primitive_consciousness`. Very expensive — runs rarely.
    multi_objective_evolution:
        Option<crate::consciousness::multi_objective_evolution::MultiObjectiveEvolution>,

    /// Cached primitive validation results (one-shot at cycle 500).
    /// Validates whether mathematical primitives actually improve Φ.
    primitive_validation_result: Option<(f64, f64)>, // (mean_phi_gain, p_value)

    /// Value feedback loop for TD-learning on moral alignment.
    /// Records per-cycle moral assessments and provides a moving trend
    /// that modulates future moral scores as a self-correcting mechanism.
    value_feedback: crate::consciousness::value_feedback_loop::ValueFeedbackLoop,

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

    /// Spectral MIP Finder — O(n³) MIP search via Fiedler ordering (Layer 2).
    /// Replaces SynergisticIntegration: 128 dims, Fiedler + bordered Cholesky.
    /// Fed with HDC state snapshots each cycle; computed every 50 cycles.
    spectral_mip_finder: symthaea_core::consciousness_metrics::SpectralMIPFinder,

    /// Soul: Seven Harmonies value alignment for moral evaluation.
    /// When present, evaluates action alignment against core values
    /// and integrates experiences for long-term value learning.
    soul: Option<crate::soul::Soul>,

    /// Attention visualizer for debugging attention flow.
    /// When present, captures attention snapshots each cycle for
    /// ASCII heatmaps, JSON export, and Graphviz flow graphs.
    attention_visualizer: Option<crate::visualization::AttentionVisualizer>,

    /// Relational Psi from dyad computation (set externally by Symthaea facade).
    /// Blended into unified_psi at 15% weight when > 0.
    relational_psi: f64,

    /// External reward signal injected by environment (0.0 = none).
    /// Blended with internal prediction-error-based reward at 50% weight.
    /// Resets to 0.0 after consumption.
    external_reward: f32,

    /// Social trust level injected by Mind module's SocialCoherence (0.0–1.0, default 0.5).
    /// Fed into AffectiveBridge for social modulation of affect (Decety & Chaminade 2003).
    social_trust: f32,

    /// Social cooperation rate injected by Mind module's SocialCoherence (0.0–1.0, default 0.0).
    /// Fed into AffectiveBridge arousal modulation.
    social_cooperation_rate: f32,

    /// User state inference for adaptive response generation.
    /// When enabled, infers user cognitive load, frustration, and engagement from input text.
    user_state: Option<crate::user_state_inference::UserStateInference>,

    /// Virtual body adapter for embodied cognition.
    /// When enabled, maps cognitive signals to interoceptive states and produces
    /// a phi_modulation factor that scales consciousness from somatic feedback.
    virtual_body: Option<virtual_body::VirtualBody>,

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
}

// MetricsProvider impl is in metrics_provider.rs
// MFDI identity impl is in identity_integration.rs

#[cfg(test)]
mod tests;
