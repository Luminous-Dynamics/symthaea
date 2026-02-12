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


// ── Submodules ────────────────────────────────────────────────────────────────
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

// ── Imports ──────────────────────────────────────────────────────────────────
use anyhow::Result;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use ndarray::Array1;
use symthaea_core::hdc::predictive_encoder::PredictiveHdcEncoder;
use crate::cfc::CfCNetwork;
use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig, CoherenceSummary};
use crate::dynamics::temporal_signatures::{
    TemporalSignatureEncoder, SignatureConfig, ConsciousnessPattern, TemporalStateSummary
};
use crate::voice::voice_feedback::{VoiceFeedbackBridge, VoiceFeedbackConfig, VoiceOutputMetrics, VoiceQualitySummary};
use crate::consciousness::consciousness_unification::{
    ConsciousnessUnificationEngine, UnifiedEmotionalState, UnifiedEmotion, EmotionalPattern,
};
use crate::consciousness::fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation,
    EnhancedFEPBridge, MotorCommandType,
};
use crate::memory::coherence_tracker::ConversationCoherenceTracker;
use crate::memory::semantic_memory::SemanticMemory;
use crate::memory::memory_coordinator::{MemoryCoordinator, CoordinatorConfig};
use crate::hdc_ltc_bridge::HdcLtcBridge;
use crate::consciousness::stability_regime::{StabilityRegimeProcessor, RegimeTransition};
use crate::consciousness::primitive_discovery::{PrimitiveDiscoveryService, DiscoveryServiceConfig};
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use crate::causal::{CausalLoopEnhancer, CausalEnhancerConfig, CausalGraph, DiscoveredRelationship};
use crate::hdc::moral_algebra::{MoralAlgebra, MoralVerdict, DeontologicalVerdict};
use crate::hdc::moral_parser::MoralParser;
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;

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

// ═══════════════════════════════════════════════════════════════════════════════
// ASYNC TRAINING — Background thread for BPTT/SPSA so inference never blocks
// ═══════════════════════════════════════════════════════════════════════════════

/// A single training sample sent from the inference thread to the trainer.
struct TrainingSample {
    input: Array1<f32>,
    target: Array1<f32>,
    dt: f32,
    learning_rate: f32,
    method: TrainingMethod,
    avg_loss: f32,
}

/// Handle held by `CognitiveLoopService` to communicate with the background
/// training thread.  Dropping it causes the background thread to exit.
///
/// The `Mutex<Receiver>` makes this struct `Sync` so that `CognitiveLoopService`
/// can implement `MetricsProvider: Send + Sync`.  In practice the mutex is
/// uncontended because `cycle()` is the only reader.
struct AsyncTrainerHandle {
    sample_tx: mpsc::SyncSender<TrainingSample>,
    weights_rx: std::sync::Mutex<mpsc::Receiver<Vec<f32>>>,
    updates_applied: u64,
}

impl AsyncTrainerHandle {
    fn spawn(mut network: CfCNetwork) -> Self {
        let (sample_tx, sample_rx) = mpsc::sync_channel::<TrainingSample>(4);
        let (weights_tx, weights_rx) = mpsc::channel::<Vec<f32>>();

        std::thread::Builder::new()
            .name("symthaea-trainer".into())
            .spawn(move || {
                let mut steps_since_publish: u32 = 0;
                while let Ok(sample) = sample_rx.recv() {
                    let result = match sample.method {
                        TrainingMethod::Spsa => {
                            network.train_step_spsa(&sample.input, &sample.target, sample.dt, sample.learning_rate)
                        }
                        TrainingMethod::Bptt => {
                            network.train_step_bptt(&[sample.input], &[sample.target], &[sample.dt], sample.learning_rate)
                        }
                        TrainingMethod::BpttWithSpsaFallback => {
                            let bptt = network.train_step_bptt(
                                &[sample.input.clone()], &[sample.target.clone()],
                                &[sample.dt], sample.learning_rate,
                            );
                            match bptt {
                                Ok(loss) if loss.is_finite() && (sample.avg_loss <= 0.0 || loss < sample.avg_loss * 2.0) => Ok(loss),
                                _ => network.train_step_spsa(&sample.input, &sample.target, sample.dt, sample.learning_rate),
                            }
                        }
                    };
                    steps_since_publish += 1;
                    if steps_since_publish >= 4 && result.is_ok() {
                        let _ = weights_tx.send(network.get_weights());
                        steps_since_publish = 0;
                    }
                }
            })
            .expect("failed to spawn trainer thread");

        Self { sample_tx, weights_rx: std::sync::Mutex::new(weights_rx), updates_applied: 0 }
    }

    fn apply_latest_weights(&mut self, network: &mut CfCNetwork) -> bool {
        let mut latest: Option<Vec<f32>> = None;
        let rx = self.weights_rx.get_mut().expect("weights_rx mutex poisoned");
        while let Ok(w) = rx.try_recv() {
            latest = Some(w);
        }
        if let Some(w) = latest {
            network.set_weights(&w);
            self.updates_applied += 1;
            true
        } else {
            false
        }
    }

    fn send(&self, sample: TrainingSample) {
        let _ = self.sample_tx.try_send(sample);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL NETWORK WRAPPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Wrapper enum for temporal network backends
///
/// This allows the CognitiveLoopService to use either CfC or HdcLtcUnified
/// as the temporal prediction backend, selected at runtime.
#[allow(dead_code)]  // Some methods are provided for API completeness
enum TemporalNetwork {
    /// CfC (Closed-form Continuous-time) network
    CfC(CfCNetwork),
    /// HdcLtcUnified network via bridge
    HdcLtc(HdcLtcBridge),
}

#[allow(dead_code)]  // Methods provided for API completeness and future use
impl TemporalNetwork {
    /// Step the network forward
    fn step(&mut self, input: &Array1<f32>, dt: f32) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.step(input, dt),
            Self::HdcLtc(bridge) => bridge.step(input, dt),
        }
    }

    /// Read the current state
    fn read_state(&self) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.read_state(),
            Self::HdcLtc(bridge) => bridge.read_state(),
        }
    }

    /// Forward pass and return output
    fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        match self {
            Self::CfC(cfc) => cfc.forward(input, dt),
            Self::HdcLtc(bridge) => bridge.forward(input, dt),
        }
    }

    /// Train step (delegates to BPTT by default for CfC)
    fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => cfc.train_step(input, target, dt, learning_rate),
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using BPTT (analytical gradients).
    /// For HdcLtc this falls through to the default train_step.
    fn train_step_bptt(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => cfc.train_step_bptt(
                &[input.clone()], &[target.clone()], &[dt], learning_rate,
            ),
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using SPSA (perturbation-based gradients).
    /// For HdcLtc this falls through to the default train_step.
    fn train_step_spsa(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => cfc.train_step_spsa(input, target, dt, learning_rate),
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Predict forward at a specific time horizon
    fn predict_forward(&mut self, input: &Array1<f32>, horizon: f32) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.predict_forward(input, horizon),
            Self::HdcLtc(bridge) => bridge.predict_forward(input, horizon),
        }
    }

    /// Inject state
    fn inject(&mut self, state: &Array1<f32>) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.inject(state),
            Self::HdcLtc(bridge) => bridge.inject(state),
        }
    }

    /// Reset the network
    fn reset(&mut self) {
        match self {
            Self::CfC(cfc) => cfc.reset(),
            Self::HdcLtc(bridge) => bridge.reset(),
        }
    }

    /// Get state diversity metric
    fn state_diversity(&self) -> f32 {
        match self {
            Self::CfC(cfc) => cfc.state_diversity(),
            Self::HdcLtc(bridge) => bridge.state_diversity(),
        }
    }

    /// Get all tau values for coherence tracking
    fn all_tau(&self) -> Vec<&Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.all_tau(),
            Self::HdcLtc(_) => vec![], // HdcLtc returns owned, handled separately
        }
    }

    /// Get all tau values (owned version for HdcLtc compatibility)
    fn all_tau_owned(&self) -> Vec<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.all_tau().into_iter().cloned().collect(),
            Self::HdcLtc(bridge) => bridge.all_tau(),
        }
    }

    /// Get flattened tau values
    fn flattened_tau(&self) -> Vec<f32> {
        match self {
            Self::CfC(cfc) => cfc.flattened_tau(),
            Self::HdcLtc(bridge) => bridge.flattened_tau(),
        }
    }

    /// Adaptively resize HDC dimension based on prediction error (HdcLtc only)
    fn maybe_resize(&mut self, current_error: f32) {
        if let Self::HdcLtc(bridge) = self {
            bridge.maybe_resize(current_error);
        }
    }

    /// Check if using HdcLtc backend
    fn is_hdc_ltc(&self) -> bool {
        matches!(self, Self::HdcLtc(_))
    }

    /// Get backend type
    fn backend_type(&self) -> TemporalBackend {
        match self {
            Self::CfC(_) => TemporalBackend::CfC,
            Self::HdcLtc(_) => TemporalBackend::HdcLtcUnified,
        }
    }

    /// Project input directly to HDC space, bypassing CfC temporal dynamics.
    ///
    /// Returns `None` for CfC backend (no HDC projection available).
    /// Returns `Some(Vec<f32>)` for HdcLtc backend with the raw HDC vector.
    fn project_to_hdc_vec(&self, input: &[f32]) -> Option<Vec<f32>> {
        match self {
            Self::CfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.project_to_hdc_vec(input)),
        }
    }

    /// Get HDC dimension (returns None for CfC backend)
    fn hdc_dim(&self) -> Option<usize> {
        match self {
            Self::CfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.hdc_dim()),
        }
    }
}

/// The Cognitive Loop Service
///
/// Orchestrates the bidirectional HDC↔CfC loop for emergent cognition.
/// Supports both CfC and HdcLtcUnified networks for O(1) temporal prediction.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE: Consciousness Unification Engine
    // ═══════════════════════════════════════════════════════════════════════════

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
    /// Implements the paradigm: Learning → Behavioral Change
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
    /// to modulate learning rate - high error on similar inputs → boost learning
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

    // ═══════════════════════════════════════════════════════════════════════════
    // MORAL ALGEBRA: Compositional Ethical Reasoning
    // ═══════════════════════════════════════════════════════════════════════════

    /// Moral Algebra for compositional ethical reasoning using HDC
    /// Encodes moral primitives (AGENT, PATIENT, ACTION, INTENT, CONSENT, OBLIGATION, MAGNITUDE)
    /// and provides judgment operations for action evaluation
    moral_algebra: MoralAlgebra,

    /// Moral Parser for extracting ethical primitives from natural language input
    /// Detects consent, intent, magnitude, and negation from text
    moral_parser: MoralParser,

    /// Last moral evaluation result (for tracking and learning)
    last_moral_judgment: Option<MoralJudgmentSummary>,
}

impl CognitiveLoopService {
    /// Create a new cognitive loop service
    pub fn new(config: CognitiveLoopConfig) -> Result<Self> {
        let encoder = PredictiveHdcEncoder::new(config.encoder_config.clone());

        // Create temporal network based on selected backend
        let temporal_network = match config.temporal_backend {
            TemporalBackend::CfC => {
                // Create CfC network with input_dim and num_neurons
                let cfc = if let Some(ref phrase) = config.genesis_phrase {
                    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
                    let net_config = crate::dynamics::cfc::CfCNetworkConfig {
                        input_dim: config.cfc_config.input_dim,
                        hidden_dim: config.cfc_config.num_neurons,
                        ..Default::default()
                    };
                    CfCNetwork::from_genesis(net_config, &genesis, "cognitive_loop::cfc")
                } else {
                    CfCNetwork::new_with_input(
                        config.cfc_config.input_dim,
                        config.cfc_config.num_neurons,
                    )
                };
                TemporalNetwork::CfC(cfc)
            }
            TemporalBackend::HdcLtcUnified => {
                // Create HdcLtcBridge with appropriate config
                let mut bridge_config = config.hdc_ltc_config.clone();
                // Ensure dimensions match CfC config for compatibility
                bridge_config.input_dim = config.cfc_config.input_dim;
                bridge_config.output_dim = config.cfc_config.num_neurons;
                let bridge = if let Some(ref phrase) = config.genesis_phrase {
                    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
                    HdcLtcBridge::from_genesis(bridge_config, &genesis)
                } else {
                    HdcLtcBridge::new(bridge_config)
                };
                TemporalNetwork::HdcLtc(bridge)
            }
        };

        // Initialize coherence bridge with learning rate from config
        let coherence_config = CoherenceConfig {
            base_learning_rate: config.cfc_config.learning_rate,
            ..Default::default()
        };
        let coherence_bridge = CfCCoherenceBridge::new(coherence_config);

        // Initialize voice feedback bridge
        let voice_feedback_bridge = VoiceFeedbackBridge::new(VoiceFeedbackConfig::default());

        // Initialize temporal signature encoder for consciousness pattern detection
        let temporal_signature_encoder = TemporalSignatureEncoder::new(SignatureConfig::default());

        // Initialize adaptive behavior with defaults
        let adaptive_behavior = AdaptiveBehavior::default();

        // Initialize closed learning loop with genesis-seeded RNG when available
        let closed_learning_loop = if let Some(ref phrase) = config.genesis_phrase {
            let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
            ClosedLearningLoop::with_rng(genesis.domain("cognitive_loop::exploration"))
        } else {
            ClosedLearningLoop::default()
        };

        // Spawn background training thread when async_training is enabled and backend is CfC
        let async_trainer = if config.async_training {
            match &temporal_network {
                TemporalNetwork::CfC(cfc) => Some(AsyncTrainerHandle::spawn(cfc.clone())),
                _ => None,
            }
        } else {
            None
        };

        // Build optional causal enhancer (needs config fields before move)
        let causal_enhancer = if config.causal_enhancement {
            let causal_config = CausalEnhancerConfig {
                discovery_interval: config.causal_discovery_interval,
                seed: config.genesis_phrase.as_ref()
                    .map(|p| symthaea_core::genesis::GenesisSeed::from_phrase(p)
                        .domain("causal_enhancer")
                        .gen::<u64>())
                    .unwrap_or(42),
                ..Default::default()
            };
            Some(CausalLoopEnhancer::with_config(causal_config))
        } else {
            None
        };

        // Build optional episodic replay (needs config fields before move)
        let phi_episodic_replay = if config.episodic_replay {
            Some(crate::memory::episodic_replay::EpisodicMemory::new(
                config.episodic_replay_config.clone()
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            encoder,
            temporal_network,
            buffer: VecDeque::with_capacity(1000),
            stats: LoopStats::default(),
            error_history: VecDeque::with_capacity(100),
            last_state: None,
            last_prediction: None,
            start_time: Instant::now(),
            is_consolidating: false,
            coherence_bridge,
            voice_feedback_bridge,
            temporal_signature_encoder,
            adaptive_behavior,
            prediction_confidence: 0.5, // Start neutral
            flow_state: FlowState::default(),
            emotion_contagion: EmotionContagion::default(),
            curiosity_drive: CuriosityDrive::default(),
            self_reflection: SelfReflection::default(),
            // Mega-unified architecture components
            thalamic_router: ThalamicRouter::default(),
            unification_engine: ConsciousnessUnificationEngine::new(),
            cognitive_depth: CognitiveDepth::default(),
            active_inference_bridge: ActiveInferenceBridge::with_defaults(),
            closed_learning_loop,
            // Memory system bridges
            episodic_memory: EpisodicMemoryBridge::default(),
            goal_system: GoalSystemBridge::new(),
            world_model: WorldModelBridge::default(),
            // FEP Active Inference Agent
            fep_agent: ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 4,
                enable_td_learning: true,
                ..Default::default()
            }),
            // Enhanced FEP Bridge with motor system (8 motor command types, 4D proprioceptive state)
            enhanced_fep_bridge: EnhancedFEPBridge::new(
                ActiveInferenceAgentConfig {
                    state_dim: 8,
                    obs_dim: 4,
                    num_actions: 8,  // Matches MotorCommandType variants
                    enable_td_learning: true,
                    ..Default::default()
                },
                4,  // Motor state dimension
            ),
            fep_learning_signal: 0.0,
            fep_lr_boost: 1.0,
            coherence_tracker: ConversationCoherenceTracker::new(0.3),
            stability_regime: StabilityRegimeProcessor::new(),
            discovery_service: PrimitiveDiscoveryService::new(DiscoveryServiceConfig::default()),
            // Semantic memory: HDC-based similarity lookup for CfC context
            // 1000 entries, 0.3 similarity threshold
            semantic_memory: SemanticMemory::with_threshold(1000, 0.3),
            // Memory coordinator: cross-tier signal broadcaster
            memory_coordinator: MemoryCoordinator::new(CoordinatorConfig::default()),
            #[cfg(feature = "neural-bridge")]
            neural_bridge: {
                let probe_path = std::path::Path::new("models/neural_bridge/probe_weights.npy");
                if probe_path.exists() {
                    match NeuralBridge::load(probe_path) {
                        Ok(nb) => {
                            tracing::info!(
                                input_dim = nb.input_dim(),
                                "Neural bridge loaded from {}",
                                probe_path.display()
                            );
                            Some(nb)
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load neural bridge: {e}");
                            None
                        }
                    }
                } else {
                    tracing::debug!("No probe weights at {}, neural bridge disabled", probe_path.display());
                    None
                }
            },
            async_trainer,
            causal_enhancer,
            phi_episodic_replay,
            #[cfg(feature = "reasoning_engine")]
            reasoning_engine: Some(crate::consciousness::reasoning_engine::ConsciousReasoningEngine::new()),
            // MFDI Bridge for identity verification and signed outputs
            #[cfg(feature = "identity")]
            mfdi_bridge: crate::identity::MfdiBridge::new(crate::identity::MfdiConfig::default()),

            // Moral Algebra for compositional ethical reasoning
            moral_algebra: MoralAlgebra::default_dim(),
            moral_parser: MoralParser::new(),
            last_moral_judgment: None,
        })
    }

    /// Get the current temporal backend type
    pub fn temporal_backend(&self) -> TemporalBackend {
        self.temporal_network.backend_type()
    }

    /// Process a pre-computed text embedding through the neural bridge and
    /// cognitive loop.
    ///
    /// Pipeline: embedding (e.g. BGE-M3 768-d) → NeuralBridge linear probe
    /// → 16384-d HDC vector → compress → CfC temporal processing → CycleResult.
    ///
    /// This bypasses the text-based HDC encoder and instead uses a trained
    /// probe to project dense embeddings directly into HDC space, giving
    /// the cognitive loop access to rich semantic representations.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Pre-computed embedding vector (dimension must match
    ///   the probe's input_dim, e.g. 768 for BGE-M3 or 1024 for BGE-M3
    ///   dense).
    ///
    /// # Returns
    ///
    /// * `CycleResult` on success, or an error if the neural bridge is not
    ///   loaded or the embedding dimension is wrong.
    #[cfg(feature = "neural-bridge")]
    pub fn process_text_input(&mut self, embedding: &[f32]) -> Result<CycleResult> {
        use symthaea_core::hdc::ContinuousHV;

        let bridge = self.neural_bridge.as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "Neural bridge not loaded (no probe weights found)"
            ))?;

        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // 1. Project embedding → continuous HDC vector (16384-d)
        let hdc_continuous = bridge.project(embedding)?;

        // 2. Wrap as ContinuousHV so we can reuse compress_for_ltc
        let hdv = ContinuousHV::from_vec(hdc_continuous);

        // 3. Compress HDC → CfC input dimension via random projection
        let compressed_state = self.encoder.compress_for_ltc(
            &hdv,
            self.config.cfc_config.input_dim,
        );

        // 4. Convert to ndarray and step the temporal network
        let input_array = Array1::from_vec(compressed_state.clone());
        let delta_t = self.config.cfc_config.delta_t;
        let _ = self.temporal_network.step(&input_array, delta_t);

        // 5. Multi-scale prediction
        let prediction = self.get_multi_scale_prediction(&input_array);

        // 6. Read CfC output state
        let output = self.temporal_network.read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);

        // 7. Feed prediction back to encoder for next cycle
        self.encoder.set_prediction(prediction.clone());

        // 8. Compute prediction error against previous prediction
        let prediction_error = if let Some(ref prev) = self.last_prediction {
            let n = compressed_state.len().min(prev.len());
            if n == 0 {
                0.0
            } else {
                compressed_state[..n].iter()
                    .zip(prev[..n].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / n as f32
            }
        } else {
            0.0
        };

        // 9. Store experience
        self.create_experience(&compressed_state, &prediction, prediction_error);

        // 10. Learning step: consolidate periodically
        let mut learning_occurred = false;
        let mut training_loss = None;
        if self.config.enable_consolidation && self.stats.total_cycles % 50 == 0 {
            if let Ok(loss) = self.consolidate() {
                if loss > 0.0 {
                    learning_occurred = true;
                    training_loss = Some(loss);
                }
            }
        }

        // 11. Update error history
        self.error_history.push_back(prediction_error);
        if self.error_history.len() > 100 {
            self.error_history.pop_front();
        }
        self.stats.avg_prediction_error = self.error_history.iter().sum::<f32>()
            / self.error_history.len().max(1) as f32;

        Ok(CycleResult {
            output: output.clone(),
            prediction_error,
            attention_state: HashMap::new(), // No text-based attention for embedding input
            detected_primitives: Vec::new(), // No text primitives for embedding input
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            #[cfg(feature = "identity")]
            signed_output: self.mfdi_bridge.sign_output(&output).ok(),
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MORAL ALGEBRA INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════════

    /// Evaluate the moral alignment of an input text.
    ///
    /// Uses HDC-based moral algebra to:
    /// - Extract moral primitives (agent, patient, action, intent, consent, magnitude)
    /// - Check for consent violations
    /// - Check for deontological violations/satisfactions
    /// - Compute overall moral score
    ///
    /// Returns a summary of the moral evaluation.
    pub fn evaluate_moral_alignment(&mut self, input: &str) -> MoralJudgmentSummary {
        // Parse and encode the input
        let encoded = self.moral_parser.parse_and_encode(input, &self.moral_algebra);

        // Get basic judgment
        let (verdict_str, good_sim, bad_sim) = if let Some(judgment) = encoded.judge(&self.moral_algebra) {
            let v = match judgment.verdict {
                MoralVerdict::Good => "Good",
                MoralVerdict::Bad => "Bad",
                MoralVerdict::Neutral => "Neutral",
                MoralVerdict::ConsentViolation => "ConsentViolation",
            };
            (v.to_string(), judgment.good_similarity, judgment.bad_similarity)
        } else {
            ("Neutral".to_string(), 0.0, 0.0)
        };

        // Get deontological judgment
        let deont = self.moral_algebra.judge_deontological(input);
        let deont_verdict_str = match deont.verdict {
            DeontologicalVerdict::RightDutyFulfilled => "Permissible",
            DeontologicalVerdict::WrongPerfectDutyViolated => "Impermissible",
            DeontologicalVerdict::WrongImperfectDutyViolated => "Impermissible",
            DeontologicalVerdict::Neutral => "Neutral",
        }.to_string();

        // Extract violation and satisfaction names
        let violations: Vec<String> = deont.violations.iter()
            .map(|v| v.rule_name.clone())
            .collect();
        let satisfactions: Vec<String> = deont.satisfactions.iter()
            .map(|s| s.rule_name.clone())
            .collect();

        // Check consent violation
        let consent_violation = encoded.is_consent_violation();

        // Compute moral score (-1.0 to 1.0)
        let moral_score = if consent_violation {
            -0.8 // Strong penalty for consent violation
        } else {
            // Balance good/bad similarity and deontological score
            let base_score = (good_sim - bad_sim).clamp(-1.0, 1.0);
            let deont_factor = deont.score.clamp(-1.0, 1.0);
            (base_score * 0.6 + deont_factor * 0.4).clamp(-1.0, 1.0)
        };

        // Compute confidence based on parsing quality
        let confidence = encoded.parsed.confidence;

        let summary = MoralJudgmentSummary {
            input: input.to_string(),
            verdict: verdict_str,
            deontological_verdict: deont_verdict_str,
            violations,
            satisfactions,
            consent_violation,
            moral_score,
            confidence,
        };

        // Store for tracking
        self.last_moral_judgment = Some(summary.clone());
        summary
    }

    /// Get the last moral judgment (if any)
    pub fn last_moral_judgment(&self) -> Option<&MoralJudgmentSummary> {
        self.last_moral_judgment.as_ref()
    }

    /// Check if the last input had moral concerns
    pub fn has_moral_concerns(&self) -> bool {
        self.last_moral_judgment.as_ref()
            .map(|j| j.moral_score < -0.3 || j.consent_violation || !j.violations.is_empty())
            .unwrap_or(false)
    }

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
    /// - **Φ Updates**: Feeds consciousness level to the unification engine
    /// - **Moral Algebra**: Evaluates ethical alignment of inputs
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE -1: Ingest background-trained weights (non-blocking)
        // ═══════════════════════════════════════════════════════════════════════
        if let Some(ref mut trainer) = self.async_trainer {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                trainer.apply_latest_weights(cfc);
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0: Thalamic Routing (Cognitive Depth Selection)
        // ═══════════════════════════════════════════════════════════════════════
        // Determine how deep to process BEFORE encoding, based on prior state

        let prior_pattern = self.temporal_signature_encoder.classify_state().0;
        let prior_valence = self.emotion_contagion.prosody_valence();
        let prior_error = self.stats.avg_prediction_error;

        self.cognitive_depth = self.thalamic_router.route_from_cycle(
            prior_error,
            prior_pattern,
            prior_valence,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.4: Moral Evaluation
        // ═══════════════════════════════════════════════════════════════════════
        // Evaluate input for moral alignment using HDC-based moral algebra.
        // This informs downstream processing and can trigger ethical safeguards.

        let moral_judgment = self.evaluate_moral_alignment(input);
        let moral_concern_detected = moral_judgment.moral_score < -0.3
            || moral_judgment.consent_violation
            || !moral_judgment.violations.is_empty();

        // Update stats with moral evaluation
        self.stats.moral_evaluations += 1;
        if moral_concern_detected {
            self.stats.moral_concerns_detected += 1;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        // Select response strategy BEFORE processing, based on:
        // - Q-learning from past interactions
        // - Previous reward (stick with success, avoid failure)
        // - Φ-gating (high Φ → Exploratory, low Φ → Supportive)
        // - Moral concerns (bias toward Supportive for ethical guidance)

        let prior_phi = self.unification_engine.phi;
        let prior_reward = self.closed_learning_loop.last_result.as_ref().map(|r| r.reward);
        let selected_strategy = if moral_concern_detected {
            // Bias toward supportive strategy when moral concerns detected
            ResponseStrategy::Supportive
        } else {
            self.closed_learning_loop.select_strategy(prior_phi, prior_reward)
        };

        // Strategy influences adaptive behavior
        match selected_strategy {
            ResponseStrategy::Exploratory => {
                self.adaptive_behavior.exploration_factor = 0.8;
            }
            ResponseStrategy::Detailed => {
                self.adaptive_behavior.attention_sensitivity = 1.2;
            }
            ResponseStrategy::Concise => {
                self.adaptive_behavior.speech_rate_multiplier = 1.2;
            }
            ResponseStrategy::Clarifying => {
                self.adaptive_behavior.exploration_factor = 0.5;
            }
            ResponseStrategy::Supportive => {
                self.adaptive_behavior.pause_multiplier = 1.3;
            }
        }

        // 1. HDC encode with attention from previous prediction
        let encoding_result = self.encoder.encode(input);
        let prediction_error = encoding_result.prediction_error;

        // ═══════════════════════════════════════════════════════════════════════
        // 1a. Memory System Integration: Recall relevant episodic memories
        // ═══════════════════════════════════════════════════════════════════════
        // Use HDC embedding to query episodic memory for context

        let hdv_sample: Vec<f32> = encoding_result.hdv.as_slice()[..64.min(encoding_result.hdv.dim())].to_vec();
        let recalled_memories = self.episodic_memory.recall(&hdv_sample, 3, 0.3);
        let memory_context_boost = if !recalled_memories.is_empty() {
            // Recalled memories boost prediction confidence slightly (safe division with max(1))
            recalled_memories.iter().map(|(_, sim)| sim).sum::<f32>() / recalled_memories.len().max(1) as f32 * 0.1
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════════
        // 1a.2. Goal System: Apply attention bias from active goals
        // ═══════════════════════════════════════════════════════════════════════

        let goal_attention_bias = self.goal_system.attention_bias();
        self.adaptive_behavior.attention_sensitivity *= goal_attention_bias;

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
            0.6 + 0.2 * self.flow_state.intensity as f64
        } else if self.prediction_confidence > 0.6 {
            0.4
        } else {
            0.2
        };

        self.unification_engine.emotional.update_from_core_affect(
            simple_valence,
            simple_arousal,
            dominance,
        );

        // 2. Compress HDC state for CfC (using Random Projection)
        let compressed_state = self.encoder.compress_for_ltc(
            &encoding_result.hdv,
            self.config.cfc_config.input_dim
        );

        // ═══════════════════════════════════════════════════════════════════════
        // 2a. SEMANTIC MEMORY: HDC-based similarity lookup for CfC context
        // ═══════════════════════════════════════════════════════════════════════
        // Project to HDC space and find similar past inputs.
        // Use their prediction errors to modulate learning rate:
        // - High error on similar inputs → boost learning (we struggled before)
        // - Low error on similar inputs → reduce learning (familiar territory)
        //
        // For HdcLtc backend: use the native HDC projection
        // For CfC backend: use the compressed state as the semantic vector

        let semantic_hdc = self.temporal_network.project_to_hdc_vec(&compressed_state)
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

        // 3. Convert to ndarray for CfC
        let input_array = Array1::from_vec(compressed_state.clone());

        // 4. Step CfC forward with current input
        let delta_t = self.config.cfc_config.delta_t;
        let _ = self.temporal_network.step(&input_array, delta_t);

        // 5. Get multi-scale predictions using CfC's O(1) predict_forward
        // This is the key advantage: instant prediction at any future time
        let prediction = self.get_multi_scale_prediction(&input_array);

        // 6. Get current CfC state as output
        let output = self.temporal_network.read_state()
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.num_neurons]);

        // ═══════════════════════════════════════════════════════════════════════
        // 6b. World Model: Update hierarchical world model with sensory input
        // ═══════════════════════════════════════════════════════════════════════

        self.world_model.update_sensory(&compressed_state);

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
        let flattened_tau = self.temporal_network.flattened_tau();
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

        // 10d. Update prediction confidence with decay during uncertain states
        self.update_prediction_confidence(pattern, prediction_error, pattern_confidence);

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.5 Active Inference Bridge: Observe prediction resolution for PAC tracking
        // ═══════════════════════════════════════════════════════════════════════
        // Track prediction-outcome coupling via Phase-Amplitude Coupling (PAC)
        // This enables precision-weighted prediction errors

        // Consider prediction "successful" if error is below learning threshold
        let prediction_success = prediction_error < self.config.learning_threshold;
        self.active_inference_bridge.observe_resolution(
            self.prediction_confidence as f64,
            prediction_success,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6 FEP Active Inference: Full perception-action loop
        // ═══════════════════════════════════════════════════════════════════════
        let effective_lr = self.stats.adaptive_learning_rate;
        let fep_obs = Observation::from_consciousness_state(
            prediction_error as f64,
            coherence as f64,
            self.prediction_confidence as f64,
            effective_lr as f64,
        );
        let _perception = self.fep_agent.perceive(&fep_obs);
        let action_result = self.fep_agent.select_action();
        let _outcome = self.fep_agent.act(action_result.action);

        // Apply FEP-selected action to modulate cognitive parameters
        let is_surprised = self.fep_agent.is_surprised();
        match action_result.action {
            0 => {
                // Boost learning rate when free energy is high
                if let Some(ref fe) = self.fep_agent.last_fe_components {
                    let fe_boost = (fe.total.abs() as f32 / 2.0).clamp(0.0, 1.5);
                    self.fep_lr_boost =
                        (self.fep_lr_boost * (1.0 + fe_boost * 0.5)).clamp(1.0, 2.0);
                }
            }
            1 => {
                // Reset sensory precision toward 1.0 to trust new observations after shift
                let current = self.fep_agent.precision.sensory_precision;
                self.fep_agent.precision.sensory_precision =
                    current * 0.7 + 1.0 * 0.3;
            }
            2 => {
                // Boost exploration — stronger nudge when surprised
                let nudge = if is_surprised { 0.15 } else { 0.05 };
                self.curiosity_drive.exploration_urge =
                    (self.curiosity_drive.exploration_urge + nudge).clamp(0.0, 1.0);
            }
            3 => {
                // Tighten trust via precision
                if let Some(ref fe) = self.fep_agent.last_fe_components {
                    let precision_mod = (1.0 - fe.prediction_error).clamp(0.0, 1.0) as f32;
                    self.self_reflection.trust_threshold =
                        (self.self_reflection.trust_threshold * 0.9 + precision_mod * 0.1).clamp(0.1, 0.9);
                }
            }
            _ => {}
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Moral Modulation of Active Inference
        // ═══════════════════════════════════════════════════════════════════════
        // Apply moral constraints to FEP-selected actions:
        // - Negative moral score → reduce exploration, increase caution
        // - Consent violation → strong ethical override
        // - Deontological violations → trigger reflective pause

        if moral_concern_detected {
            // Reduce exploration when facing moral concerns
            self.curiosity_drive.exploration_urge *= 0.5;

            // Increase trust threshold (be more cautious)
            self.self_reflection.trust_threshold =
                (self.self_reflection.trust_threshold * 1.2).clamp(0.1, 0.95);

            // Boost reflective processing (take time to consider ethics)
            self.adaptive_behavior.pause_multiplier *= 1.5;

            // If severe moral violation (perfect duty or consent), flag for review
            if moral_judgment.consent_violation ||
               moral_judgment.violations.iter().any(|v| v.contains("perfect") || v.contains("harm")) {
                self.stats.moral_review_needed = true;
            }
        } else if moral_judgment.moral_score > 0.5 {
            // Positive moral alignment boosts confidence slightly
            self.prediction_confidence =
                (self.prediction_confidence * 1.05).clamp(0.0, 1.0);
        }

        // Surprise-gated learning rate boost: when FEP detects surprise, accelerate adaptation
        if is_surprised {
            let surprise_boost = (self.fep_agent.current_free_energy() as f32 / 3.0).clamp(0.1, 0.5);
            self.fep_lr_boost = (self.fep_lr_boost + surprise_boost).clamp(1.0, 2.0);
        } else {
            // Decay boost back toward 1.0 when not surprised
            self.fep_lr_boost = (self.fep_lr_boost * 0.95).max(1.0);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.6b Enhanced FEP Bridge: Motor commands and learning signals
        // ═══════════════════════════════════════════════════════════════════════
        // Run enhanced FEP cycle for motor system integration and learning signals
        let enhanced_result = self.enhanced_fep_bridge.cycle(
            prediction_error as f64,
            coherence as f64,
            self.prediction_confidence as f64,
            effective_lr as f64,
        );

        // Update learning signal for downstream systems
        self.fep_learning_signal = enhanced_result.learning_signal as f32;

        // Apply motor command-based modulations
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
                        (self.stats.adaptive_learning_rate * 0.9 + lr_mod * 0.1).clamp(0.01, 1.0);
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
            self.world_model.increase_plasticity(self.fep_learning_signal);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10d.7 Coherence tracking with degradation detection
        // ═══════════════════════════════════════════════════════════════════════
        let degraded = self.coherence_tracker.record_turn(coherence);
        if degraded {
            // Coherence degradation → boost learning rate to accelerate recovery
            self.fep_lr_boost = (self.fep_lr_boost * 1.3).clamp(1.0, 2.0);
            let urgency = self.coherence_tracker.correction_urgency();
            // Feed urgency as a high-error observation to drive FEP learning
            let urgent_obs = Observation::from_consciousness_state(
                urgency as f64, 0.1, 0.1, effective_lr as f64,
            );
            self.fep_agent.perceive(&urgent_obs);
            // Also signal enhanced bridge about degradation
            self.enhanced_fep_bridge.cycle(urgency as f64, 0.1, 0.1, effective_lr as f64);
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
        self.curiosity_drive.set_boredom_threshold(adapted_thresholds.boredom);
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
            let _recommendations = self.self_reflection.reflect();
            // Recommendations are stored in self_reflection.recommendations
            // and can be queried by external systems
        }

        // ═══════════════════════════════════════════════════════════════════════
        // 10h. Update Consciousness Unification Engine with current Φ
        // ═══════════════════════════════════════════════════════════════════════
        // Compute unified Φ from coherence, confidence, and flow state
        // This feeds the dialogue pipeline for consciousness-aware responses

        let coherence_phi = self.coherence_bridge.phi_contribution();
        let voice_phi = self.voice_feedback_bridge.summary().phi_adjustment;
        let flow_phi = if self.flow_state.in_flow {
            self.flow_state.intensity * 0.2
        } else {
            0.0
        };
        // Combine contributions: temporal coherence + voice quality + flow state
        let unified_phi = (coherence_phi + voice_phi + flow_phi).clamp(0.0, 1.0) as f64;
        self.unification_engine.update_phi(unified_phi);

        // ═══════════════════════════════════════════════════════════════════════
        // 10h.1 Conscious Reasoning Engine: unified 7-step reasoning cycle
        // ═══════════════════════════════════════════════════════════════════════
        // When the reasoning_engine feature is enabled, run the full conscious
        // reasoning cycle (conflict detection → Φ_eff → planning → gating →
        // counterfactual → telemetry) with tiered degradation.
        #[cfg(feature = "reasoning_engine")]
        if let Some(ref mut reasoning_engine) = self.reasoning_engine {
            use crate::consciousness::epistemic_conflict::MultiTheoryMetrics as ECMetrics;
            use crate::consciousness::reasoning_engine::ReasoningContext;

            // Build theory metrics from available consciousness signals
            let ec_metrics = ECMetrics {
                phi: unified_phi,
                gwt: coherence as f64,
                ast: self.prediction_confidence as f64,
                pp: (1.0 - prediction_error as f64).clamp(0.0, 1.0),
                rpt: pattern_confidence as f64,
                embodiment: self.fep_learning_signal as f64,
                unified: unified_phi,
            };

            // Compute available budget: 20ms target cycle minus time already spent
            let elapsed_us = cycle_start.elapsed().as_micros() as u64;
            let available_us = 20_000u64.saturating_sub(elapsed_us);

            let reasoning_ctx = ReasoningContext {
                theory_metrics: ec_metrics,
                phi: unified_phi,
                available_budget_us: available_us,
                available_actions: Vec::new(), // populated by external action providers
                tool: None, // populated by shell integration
                recent_utility: 0.5,
                cycle_id: self.stats.total_cycles as u64,
            };

            let _reasoning_result = reasoning_engine.reason(&reasoning_ctx);
        }

        // Get adaptive learning rate (respects pause_learning and all modulations)
        // Include flow state boost, curiosity novelty bonus, and semantic context
        let base_lr = self.combined_learning_rate();
        let adaptive_lr = self.adaptive_behavior.effective_learning_rate(base_lr);
        let flow_lr = self.flow_state.effective_learning_multiplier(adaptive_lr);
        // Apply semantic memory modulation: boost learning when similar inputs had high error
        let semantic_modulated_lr = flow_lr * semantic_lr_factor;
        let effective_lr = (self.curiosity_drive.effective_learning_rate(semantic_modulated_lr) * self.fep_lr_boost)
            .clamp(0.0, 0.01); // Hard cap: reduced from 0.05 to 0.01 to prevent oscillation with cyclic patterns

        // 11. Learn if error is significant AND we have a previous state AND not paused
        let (learning_occurred, training_loss) = if prediction_error > self.config.learning_threshold
            && !self.adaptive_behavior.pause_learning
        {
            self.stats.learning_cycles += 1;

            // Build training sample
            let (train_input, train_target, lr) = if let Some(ref prev) = previous_state {
                (
                    Array1::from_vec(prev.clone()),
                    Array1::from_vec(compressed_state.clone()),
                    effective_lr,
                )
            } else {
                // First cycle: bootstrap with self-prediction
                let current_array = Array1::from_vec(compressed_state.clone());
                (current_array.clone(), current_array, effective_lr * 0.1)
            };

            // ─── Async path: send sample to background thread (never blocks) ───
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
                // ─── Sync path: train inline (original behaviour) ───
                let result = match self.config.training_method {
                    TrainingMethod::Spsa => {
                        self.stats.spsa_fallback_steps += 1;
                        self.temporal_network.train_step_spsa(&train_input, &train_target, delta_t, lr)
                    }
                    TrainingMethod::Bptt => {
                        self.stats.bptt_steps += 1;
                        self.temporal_network.train_step_bptt(&train_input, &train_target, delta_t, lr)
                    }
                    TrainingMethod::BpttWithSpsaFallback => {
                        let old_loss = self.stats.avg_training_loss;
                        let bptt_result = self.temporal_network.train_step_bptt(
                            &train_input, &train_target, delta_t, lr,
                        );
                        match bptt_result {
                            Ok(loss) if loss.is_finite() && (old_loss <= 0.0 || loss < old_loss * 2.0) => {
                                self.stats.bptt_steps += 1;
                                Ok(loss)
                            }
                            _ => {
                                self.stats.spsa_fallback_steps += 1;
                                self.temporal_network.train_step_spsa(
                                    &train_input, &train_target, delta_t, lr,
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

        // 12. Update statistics
        self.update_stats(prediction_error, cycle_start.elapsed());

        // Update state diversity from CfC
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Adaptive HDC dimension: resize if error demands it
        self.temporal_network.maybe_resize(prediction_error);

        // Update coherence metrics in stats
        self.stats.temporal_coherence = self.coherence_bridge.smoothed_coherence();
        self.stats.effective_learning_rate = effective_lr;
        self.stats.coherence_phi_contribution = self.coherence_bridge.phi_contribution();

        // ═══════════════════════════════════════════════════════════════════════
        // EPISODIC MEMORY: Encode this cycle's experience
        // ═══════════════════════════════════════════════════════════════════════
        // Only encode if prediction error is significant (worth remembering)

        if prediction_error > 0.1 || self.flow_state.in_flow {
            let emotional_valence = self.emotion_contagion.prosody_valence();
            let phi = self.unification_engine.phi as f32;
            self.episodic_memory.encode(
                input,
                hdv_sample.clone(),
                emotional_valence,
                phi,
                self.stats.total_cycles,
            );
        }

        // Apply memory context boost to confidence
        self.prediction_confidence = (self.prediction_confidence + memory_context_boost).clamp(0.0, 1.0);

        // ═══════════════════════════════════════════════════════════════════════
        // STABILITY REGIME: Update primitive CfC dynamics
        // ═══════════════════════════════════════════════════════════════════════
        // Convert the HDC encoding to BinaryHV and run through stability regime processor.
        // Frequently-used primitives crystallize, rarely-used stay fluid.
        {
            let hv16_input = real_hv_to_hv16(&encoding_result.hdv);
            let timestamp = self.stats.total_cycles as f64 * delta_t as f64;
            let (_regime_state, transitions) = self.stability_regime.process_input(&hv16_input, delta_t, timestamp);

            // When primitives crystallize, seed the discovery system to explore neighbors
            for transition in &transitions {
                if let RegimeTransition::Crystallized { primitive_name, encoding } = transition {
                    self.discovery_service.seed_neighbor_exploration(primitive_name, encoding);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // CLOSED LEARNING LOOP: Update with cycle results
        // ═══════════════════════════════════════════════════════════════════════
        // This closes the loop: learning from this cycle influences next cycle's strategy

        let cycle_reward = if prediction_error < self.config.learning_threshold {
            // Good prediction → positive reward (scaled by confidence)
            0.5 + 0.5 * self.prediction_confidence
        } else if prediction_error > 0.5 {
            // Very poor prediction → negative reward
            -0.3 - 0.2 * (prediction_error - 0.5)
        } else {
            // Moderate prediction → neutral to slightly negative
            0.2 - 0.5 * prediction_error
        };

        let cycle_learning_result = CycleLearningResult {
            reward: cycle_reward.clamp(-1.0, 1.0),
            strategy_used: selected_strategy,
            successful: prediction_error < self.config.learning_threshold && self.flow_state.in_flow,
            prediction_error,
            coherence,
        };

        self.closed_learning_loop.update(cycle_learning_result);

        // ═══════════════════════════════════════════════════════════════════════
        // SEMANTIC MEMORY: Store this cycle's HDC vector + prediction error
        // ═══════════════════════════════════════════════════════════════════════
        // This enables future cycles to find semantically similar inputs and
        // use their prediction errors to modulate learning rate.
        self.semantic_memory.store_with_timestamp(
            semantic_hdc,
            prediction_error,
            None, // Category could be derived from detected_primitives if desired
            self.stats.total_cycles as u64,
        );

        // Update semantic memory stats in loop stats
        self.stats.semantic_hits = self.semantic_memory.stats().semantic_hits;
        self.stats.semantic_misses = self.semantic_memory.stats().semantic_misses;
        self.stats.semantic_lr_factor = semantic_lr_factor;
        self.stats.semantic_avg_retrieved_error = self.semantic_memory.stats().avg_retrieved_error;
        self.stats.semantic_entries_stored = self.semantic_memory.stats().total_stored;

        // ═══════════════════════════════════════════════════════════════════════
        // CAUSAL ENHANCEMENT: Track (input, output) pairs and discover causal structure
        // ═══════════════════════════════════════════════════════════════════════
        // When enabled, the causal enhancer:
        // - Records each (compressed_state, output) pair
        // - Periodically runs causal discovery to find structure
        // - Logs discovered causal relationships
        if let Some(ref mut enhancer) = self.causal_enhancer {
            // Record this cycle's (input, output) pair
            enhancer.record_cycle_from_f32(&compressed_state, &output);

            // Check if it's time to run causal discovery
            if enhancer.should_discover() {
                let causal_graph = enhancer.run_discovery();

                // Log discovered relationships
                if !causal_graph.is_empty() {
                    tracing::info!(
                        edges = causal_graph.edges.len(),
                        cycle = self.stats.total_cycles,
                        "Causal structure discovered in cognitive loop"
                    );
                    enhancer.log_discoveries();
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // EPISODIC REPLAY: Store high-Phi moments and periodically replay
        // ═══════════════════════════════════════════════════════════════════════
        // When enabled, the episodic replay system:
        // - Stores episodes that exceed the Phi threshold
        // - Periodically replays high-Phi episodes to reinforce important patterns
        // - Uses Phi-weighted sampling to prioritize most conscious moments
        if let Some(ref mut replay) = self.phi_episodic_replay {
            // Get coherence summary for Phi estimation and overall coherence
            let coherence_summary = self.coherence_bridge.summary();
            // Use smoothed coherence as a proxy for Phi (both measure integration)
            let current_phi = coherence_summary.smoothed_coherence as f64;

            // Create episode from this cycle
            let input_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(
                compressed_state.clone()
            );
            let output_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_vec(
                output.clone()
            );

            let episode = crate::memory::episodic_replay::Episode::with_metadata(
                input_hv,
                output_hv,
                current_phi,
                self.stats.total_cycles as u64,
                prediction_error,
                self.emotion_contagion.smoothed_valence(),
                coherence_summary.coherence,
            );

            // Store if Phi exceeds threshold
            let stored = replay.store_if_significant(episode);
            if stored {
                tracing::trace!(
                    phi = current_phi,
                    cycle = self.stats.total_cycles,
                    "High-Phi episode stored for replay"
                );
            }

            // Check if we should run a replay session
            if replay.should_replay() {
                // Get CfC network for training (only works with CfC backend)
                if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                    let learning_rate = self.config.cfc_config.learning_rate;
                    let result = replay.replay_session(cfc, learning_rate);

                    if !result.skipped {
                        tracing::debug!(
                            episodes = result.episodes_replayed,
                            avg_loss = result.average_loss,
                            avg_phi = result.average_phi,
                            "Episodic replay session completed"
                        );
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MEMORY COORDINATOR: Broadcast consciousness signals across memory tiers
        // ═══════════════════════════════════════════════════════════════════════
        {
            let coord_phi = self.coherence_bridge.smoothed_coherence() as f64;
            let coord_coherence = coherence as f64;
            self.memory_coordinator.update_signals(coord_phi, coord_coherence);

            // Process any queued graduations into episodic memory
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

        CycleResult {
            output: output.clone(),
            prediction_error,
            attention_state: encoding_result.attention_snapshot,
            detected_primitives: encoding_result.detected_primitives,
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
            #[cfg(feature = "identity")]
            signed_output: self.mfdi_bridge.sign_output(&output).ok(),
            #[cfg(feature = "identity")]
            assurance_level: self.mfdi_bridge.assurance_level(),
        }
    }

    /// Get multi-scale prediction by averaging predictions at different time horizons
    ///
    /// This uses CfC's O(1) predict_forward to instantly query multiple future times,
    /// forcing the network to learn temporal "rules" rather than just noise patterns.
    fn get_multi_scale_prediction(&mut self, input: &Array1<f32>) -> Vec<f32> {
        let horizons = &self.config.cfc_config.prediction_horizons;

        if horizons.is_empty() {
            // Fallback: single-step prediction
            return self.temporal_network.predict_forward(input, self.config.cfc_config.delta_t)
                .map(|arr| arr.to_vec())
                .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.input_dim]);
        }

        // Collect predictions at multiple time horizons
        let mut predictions: Vec<Array1<f32>> = Vec::with_capacity(horizons.len());

        for &horizon in horizons {
            if let Ok(pred) = self.temporal_network.predict_forward(input, horizon) {
                predictions.push(pred);
            }
        }

        if predictions.is_empty() {
            return vec![0.0; self.config.cfc_config.input_dim];
        }

        // Average the multi-scale predictions
        // This forces temporal consistency across different timescales
        // Safe division: use max(1) to prevent division by zero
        let n = predictions.len().max(1) as f32;
        let dim = predictions[0].len();
        let mut result = vec![0.0f32; dim];

        for pred in &predictions {
            for (i, val) in pred.iter().enumerate() {
                if i < dim {
                    result[i] += val / n;
                }
            }
        }

        result
    }

    /// Run a background consolidation cycle
    ///
    /// This replays important experiences to strengthen learning using CfC.
    pub fn consolidate(&mut self) -> Result<f32> {
        if self.buffer.len() < 10 {
            return Ok(0.0);
        }

        self.is_consolidating = true;

        // Sort by importance and replay top experiences
        let mut experiences: Vec<_> = self.buffer.iter().collect();
        experiences.sort_by(|a, b| b.importance.partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal));

        let mut total_loss = 0.0;
        let replay_count = experiences.len().min(10);
        let delta_t = self.config.cfc_config.delta_t;
        let lr = self.config.cfc_config.learning_rate;

        for exp in experiences.iter().take(replay_count) {
            if let Some(ref next_state) = exp.next_state {
                // Reset CfC state for clean replay by injecting zeros
                let zeros = Array1::from_vec(vec![0.0f32; self.config.cfc_config.input_dim]);
                let _ = self.temporal_network.inject(&zeros);

                // Train using CfC's analytical gradient
                let prev_array = Array1::from_vec(exp.state.clone());
                let target_array = Array1::from_vec(next_state.clone());
                if let Ok(loss) = self.temporal_network.train_step(&prev_array, &target_array, delta_t, lr) {
                    total_loss += loss;
                }
            }
        }

        self.is_consolidating = false;

        Ok(total_loss / replay_count as f32)
    }

    /// Get current statistics
    pub fn stats(&self) -> &LoopStats {
        &self.stats
    }

    /// Get the configuration used to create this service.
    ///
    /// Useful for verifying that genesis seeding is correctly configured.
    pub fn config(&self) -> &CognitiveLoopConfig {
        &self.config
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CAUSAL ENHANCEMENT ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if causal enhancement is enabled
    pub fn causal_enhancement_enabled(&self) -> bool {
        self.causal_enhancer.is_some()
    }

    /// Get the current causal graph (if causal enhancement is enabled)
    pub fn causal_graph(&self) -> Option<&CausalGraph> {
        self.causal_enhancer.as_ref().map(|e| e.current_graph())
    }

    /// Get discovered causal relationships history
    pub fn causal_discoveries(&self) -> Option<&[DiscoveredRelationship]> {
        self.causal_enhancer.as_ref().map(|e| e.discovered_relationships())
    }

    /// Get causal enhancer statistics
    pub fn causal_stats(&self) -> Option<crate::causal::CausalLoopStats> {
        self.causal_enhancer.as_ref().map(|e| e.stats().clone())
    }

    /// Check if any causal structure has been discovered
    pub fn has_causal_structure(&self) -> bool {
        self.causal_enhancer.as_ref()
            .map(|e| e.has_causal_structure())
            .unwrap_or(false)
    }

    /// Force a causal discovery run (useful for testing)
    pub fn force_causal_discovery(&mut self) -> Option<CausalGraph> {
        self.causal_enhancer.as_mut().map(|e| e.run_discovery())
    }

    /// Get causal attention weights for a target dimension
    ///
    /// Returns weights that give more attention to causal parents of the target.
    /// Returns None if causal enhancement is disabled.
    pub fn causal_attention_weights(&mut self, target_dim: usize) -> Option<Vec<f32>> {
        self.causal_enhancer.as_mut().map(|e| e.causal_attention_weights(target_dim))
    }

    /// Suggest an intervention based on discovered causal structure
    ///
    /// Returns (dimension_to_intervene, suggested_value) if exploration is triggered.
    pub fn suggest_causal_intervention(&mut self) -> Option<(usize, f64)> {
        self.causal_enhancer.as_mut().and_then(|e| e.suggest_intervention())
    }

    /// Get encoder statistics
    pub fn encoder_stats(&self) -> &symthaea_core::hdc::predictive_encoder::EncoderStats {
        self.encoder.stats()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EPISODIC REPLAY ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if episodic replay is enabled
    pub fn episodic_replay_enabled(&self) -> bool {
        self.phi_episodic_replay.is_some()
    }

    /// Get episodic replay statistics
    pub fn episodic_replay_stats(&self) -> Option<crate::memory::episodic_replay::EpisodicMemoryStats> {
        self.phi_episodic_replay.as_ref().map(|r| r.stats())
    }

    /// Get the number of stored episodes
    pub fn episodic_replay_count(&self) -> usize {
        self.phi_episodic_replay.as_ref().map(|r| r.len()).unwrap_or(0)
    }

    /// Get top N episodes by Phi (highest consciousness moments)
    pub fn top_phi_episodes(&self, n: usize) -> Vec<crate::memory::episodic_replay::Episode> {
        self.phi_episodic_replay.as_ref()
            .map(|r| r.get_top_episodes(n))
            .unwrap_or_default()
    }

    /// Force an episodic replay session (useful for testing or manual consolidation)
    pub fn force_episodic_replay(&mut self, learning_rate: f32) -> Option<crate::memory::episodic_replay::ReplaySessionResult> {
        if let Some(ref mut replay) = self.phi_episodic_replay {
            if let TemporalNetwork::CfC(ref mut cfc) = self.temporal_network {
                // Temporarily bypass should_replay check by manually running replay
                let batch = replay.sample_replay_batch(self.config.episodic_replay_config.batch_size);
                if batch.is_empty() {
                    return Some(crate::memory::episodic_replay::ReplaySessionResult {
                        episodes_replayed: 0,
                        average_loss: 0.0,
                        average_phi: 0.0,
                        skipped: true,
                    });
                }

                let mut total_loss = 0.0;
                let mut total_phi = 0.0;

                for episode in &batch {
                    let loss = replay.replay_training_step(
                        cfc,
                        episode,
                        learning_rate,
                        self.config.episodic_replay_config.replay_dt,
                    );
                    total_loss += loss;
                    total_phi += episode.phi;
                }

                let n = batch.len();
                return Some(crate::memory::episodic_replay::ReplaySessionResult {
                    episodes_replayed: n,
                    average_loss: total_loss / n as f32,
                    average_phi: total_phi / n as f64,
                    skipped: false,
                });
            }
        }
        None
    }

    /// Get CfC state diversity (activation variance across cells)
    pub fn cfc_state_diversity(&self) -> f32 {
        self.temporal_network.state_diversity()
    }

    /// Get CfC state dimension
    pub fn cfc_state_dim(&self) -> usize {
        self.config.cfc_config.num_neurons
    }

    /// Get HDC bridge dimension (returns None if using CfC backend)
    ///
    /// This is the dimension of HDC vectors used by the HdcLtcBridge.
    /// Typically 16384 (HDC_DIMENSION) but can be smaller for fast configs.
    pub fn hdc_bridge_dim(&self) -> Option<usize> {
        self.temporal_network.hdc_dim()
    }

    /// Project an embedding directly to HDC space, bypassing CfC temporal dynamics.
    ///
    /// This returns the pure semantic HDC representation before any temporal
    /// state accumulation occurs. Useful for:
    /// - Semantic similarity comparisons (cosine similarity of HDC vectors)
    /// - Debugging whether semantic structure is preserved
    /// - Comparing HDC-direct clustering vs CfC-output clustering
    ///
    /// # Arguments
    /// * `embedding` - The input embedding (e.g., from BGE-M3 or mock embeddings)
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - The HDC vector (before CfC processing)
    /// * `Err` - If using CfC backend (no HDC projection available)
    pub fn project_embedding_to_hdc(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        // The HdcLtcBridge expects input of size config.input_dim (default 256).
        // We need to compress the embedding to that dimension first.
        let input_dim = self.config.cfc_config.input_dim;

        // Simple downsampling: take evenly spaced values
        let compressed = if embedding.len() <= input_dim {
            // Pad if shorter
            let mut v = embedding.to_vec();
            v.resize(input_dim, 0.0);
            v
        } else {
            // Downsample by strided selection
            let step = embedding.len() / input_dim;
            embedding.iter()
                .step_by(step)
                .take(input_dim)
                .cloned()
                .collect::<Vec<_>>()
        };

        // Project to HDC space (bypasses CfC temporal processing)
        self.temporal_network.project_to_hdc_vec(&compressed)
            .ok_or_else(|| anyhow::anyhow!(
                "HDC projection not available (using CfC backend, not HdcLtcBridge)"
            ))
    }

    /// Check if loop is learning (error trend negative)
    pub fn is_learning(&self) -> bool {
        self.stats.error_trend < 0.0 && self.stats.learning_cycles > 0
    }

    /// Check if attention has emerged (variance > threshold)
    pub fn has_emerged_attention(&self) -> bool {
        self.stats.attention_variance > 0.01
    }

    /// Get coherence summary for external systems
    pub fn coherence_summary(&self) -> CoherenceSummary {
        self.coherence_bridge.summary()
    }

    /// Get temporal coherence value
    pub fn temporal_coherence(&self) -> f32 {
        self.coherence_bridge.smoothed_coherence()
    }

    // ========== Semantic Memory Accessors ==========

    /// Get semantic memory statistics
    ///
    /// Returns stats about the HDC-based content-addressable memory including:
    /// - Total entries stored
    /// - Hit/miss counts and rates
    /// - Average retrieved error
    pub fn semantic_memory_stats(&self) -> &crate::memory::semantic_memory::SemanticMemoryStats {
        self.semantic_memory.stats()
    }

    // ========== Stability Regime Accessors ==========

    /// Get reference to the stability regime processor
    ///
    /// Provides access to CfC dynamics for primitives:
    /// - Regime distribution (Crystallized/Plastic/Fluid)
    /// - Active primitive counts
    /// - Coherence bridge from stability regime
    pub fn stability_regime(&self) -> &StabilityRegimeProcessor {
        &self.stability_regime
    }

    // ========== Prediction Confidence Methods ==========

    /// Update prediction confidence based on consciousness state and prediction accuracy
    ///
    /// Confidence decays during uncertain/transitioning states and grows when
    /// predictions are accurate in stable states.
    fn update_prediction_confidence(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        pattern_confidence: f32,
    ) {
        use ConsciousnessPattern::*;

        // Base decay/growth parameters
        const DECAY_RATE_UNCERTAIN: f32 = 0.05;    // Fast decay when uncertain
        const DECAY_RATE_TRANSITION: f32 = 0.03;   // Moderate decay during transitions
        const GROWTH_RATE_ACCURATE: f32 = 0.02;    // Slow growth for stability
        const ERROR_THRESHOLD: f32 = 0.3;          // Below this = accurate prediction

        // Decay rate depends on consciousness state
        let decay_rate = match pattern {
            Uncertain => DECAY_RATE_UNCERTAIN,
            Transitioning => DECAY_RATE_TRANSITION,
            Resting => DECAY_RATE_TRANSITION * 0.5, // Slight decay in resting
            _ => 0.0, // No decay in stable states
        };

        // Growth when predictions are accurate in stable states
        let growth_rate = if prediction_error < ERROR_THRESHOLD {
            match pattern {
                Focused | Contemplative => GROWTH_RATE_ACCURATE * 1.5,
                Excited | Exploratory => GROWTH_RATE_ACCURATE,
                _ => GROWTH_RATE_ACCURATE * 0.5,
            }
        } else {
            0.0
        };

        // Apply decay and growth
        let confidence_delta = growth_rate - decay_rate;

        // Scale by pattern confidence (more confident = stronger effect)
        let scaled_delta = confidence_delta * pattern_confidence;

        // Update with bounds
        self.prediction_confidence = (self.prediction_confidence + scaled_delta).clamp(0.0, 1.0);

        // Additional penalty for very high prediction errors
        if prediction_error > 0.7 {
            self.prediction_confidence *= 0.95; // 5% penalty for bad predictions
        }
    }

    /// Get current prediction confidence
    pub fn prediction_confidence(&self) -> f32 {
        self.prediction_confidence
    }

    /// Check if predictions should be trusted
    /// Returns true if confidence is above threshold (0.4)
    pub fn predictions_trustworthy(&self) -> bool {
        self.prediction_confidence > 0.4
    }

    // ========== Flow State Methods ==========

    /// Check if currently in flow state
    /// Flow state = sustained focus + low error + high coherence
    pub fn in_flow(&self) -> bool {
        self.flow_state.in_flow
    }

    /// Get flow state intensity (0.0 to 1.0)
    /// Higher = deeper flow state with greater benefits
    pub fn flow_intensity(&self) -> f32 {
        self.flow_state.intensity
    }

    /// Get flow state streak (consecutive flow-compatible cycles)
    pub fn flow_streak(&self) -> u32 {
        self.flow_state.streak
    }

    /// Get current flow state reference
    pub fn flow_state(&self) -> &FlowState {
        &self.flow_state
    }

    /// Get flow learning boost multiplier
    /// 1.0 when not in flow, up to 1.5 at max flow intensity
    pub fn flow_learning_boost(&self) -> f32 {
        self.flow_state.learning_boost
    }

    // ========== Emotion Contagion Methods ==========

    /// Get current emotional valence from content analysis
    /// Positive = happy/exciting content, Negative = sad/angry content
    pub fn emotional_valence(&self) -> f32 {
        self.emotion_contagion.smoothed_valence()
    }

    /// Get current emotional arousal
    /// High = intense/urgent, Low = calm/peaceful
    pub fn emotional_arousal(&self) -> f32 {
        self.emotion_contagion.smoothed_arousal()
    }

    /// Get emotion-based pattern nudge suggestion
    /// Returns (suggested pattern, influence strength)
    pub fn emotion_pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) {
        self.emotion_contagion.pattern_nudge()
    }

    /// Get emotion contagion reference
    pub fn emotion_contagion(&self) -> &EmotionContagion {
        &self.emotion_contagion
    }

    /// Check if emotional content is significant
    pub fn has_emotional_content(&self) -> bool {
        self.emotion_contagion.smoothed_valence().abs() > 0.2
    }

    // ========== Curiosity Drive Methods ==========

    /// Get current boredom level (0.0 to 1.0)
    /// High when predictions are consistently too accurate
    pub fn boredom(&self) -> f32 {
        self.curiosity_drive.boredom
    }

    /// Get curiosity level (0.0 to 1.0)
    pub fn curiosity(&self) -> f32 {
        self.curiosity_drive.curiosity
    }

    /// Get exploration urge (0.0 to 1.0)
    /// High when boredom + curiosity trigger exploration
    pub fn exploration_urge(&self) -> f32 {
        self.curiosity_drive.exploration_urge
    }

    /// Check if curiosity-triggered exploration should occur
    pub fn curiosity_should_explore(&self) -> bool {
        self.curiosity_drive.should_explore()
    }

    /// Get curiosity drive reference
    pub fn curiosity_drive(&self) -> &CuriosityDrive {
        &self.curiosity_drive
    }

    /// Get novelty bonus for learning
    pub fn novelty_bonus(&self) -> f32 {
        self.curiosity_drive.novelty_bonus
    }

    /// Check if the system is bored (needs new stimuli)
    pub fn is_bored(&self) -> bool {
        self.curiosity_drive.boredom > 0.5
    }

    // ========== Self-Reflection Methods ==========

    /// Get current self-assessment
    pub fn self_assessment(&self) -> SelfAssessment {
        self.self_reflection.self_assessment
    }

    /// Get self-reflection summary
    pub fn reflection_summary(&self) -> ReflectionSummary {
        self.self_reflection.summary()
    }

    /// Get adapted thresholds from self-reflection
    pub fn adapted_thresholds(&self) -> ReflectionThresholds {
        self.self_reflection.get_thresholds()
    }

    /// Get current recommendations from self-reflection
    pub fn recommendations(&self) -> &[Recommendation] {
        &self.self_reflection.recommendations
    }

    /// Get number of reflections performed
    pub fn reflection_count(&self) -> u64 {
        self.self_reflection.reflection_count
    }

    /// Get learning effectiveness score
    pub fn learning_effectiveness(&self) -> f32 {
        self.self_reflection.learning_effectiveness()
    }

    /// Check if system needs calibration (based on self-reflection)
    pub fn needs_calibration(&self) -> bool {
        self.self_reflection.self_assessment == SelfAssessment::NeedsCalibration
    }

    /// Check if system is performing optimally (based on self-reflection)
    pub fn is_optimal(&self) -> bool {
        self.self_reflection.self_assessment == SelfAssessment::Optimal
    }

    /// Force an immediate reflection cycle
    pub fn force_reflect(&mut self) -> Vec<Recommendation> {
        self.self_reflection.reflect()
    }

    /// Get self-reflection reference
    pub fn self_reflection(&self) -> &SelfReflection {
        &self.self_reflection
    }

    // ========== Consciousness Snapshot ==========

    /// Get a complete snapshot of current consciousness state
    ///
    /// This aggregates all cognitive metrics into a single queryable view,
    /// useful for monitoring, logging, APIs, or external integrations.
    pub fn consciousness_snapshot(&self) -> ConsciousnessSnapshot {
        let (pattern, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        let temporal_summary = self.temporal_signature_encoder.summary();
        let reflection_summary = self.self_reflection.summary();
        let thresholds = self.self_reflection.get_thresholds();
        let (emotion_nudge, _) = self.emotion_contagion.pattern_nudge();

        let consciousness_level = ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence,
            self.coherence_bridge.smoothed_coherence(),
            self.flow_state.intensity,
            pattern_confidence,
        );

        ConsciousnessSnapshot {
            // Core metrics
            cycle: self.stats.total_cycles,
            consciousness_level,
            pattern,
            pattern_confidence,

            // Prediction & Learning
            prediction_error: self.stats.avg_prediction_error,
            prediction_confidence: self.prediction_confidence,
            predictions_trustworthy: self.predictions_trustworthy(),
            effective_learning_rate: self.stats.adaptive_learning_rate,
            learning_effectiveness: self.self_reflection.learning_effectiveness(),

            // Flow state
            in_flow: self.flow_state.in_flow,
            flow_intensity: self.flow_state.intensity,
            flow_streak: self.flow_state.streak,
            flow_learning_boost: self.flow_state.learning_boost,

            // Curiosity & Exploration
            boredom: self.curiosity_drive.boredom,
            curiosity: self.curiosity_drive.curiosity,
            exploration_urge: self.curiosity_drive.exploration_urge,
            exploring: self.curiosity_drive.should_explore(),
            novelty_bonus: self.curiosity_drive.novelty_bonus,

            // Emotional state
            emotional_valence: self.emotion_contagion.smoothed_valence(),
            emotional_arousal: self.emotion_contagion.smoothed_arousal(),
            has_emotional_content: self.has_emotional_content(),
            emotion_nudge,

            // Self-reflection
            self_assessment: self.self_reflection.self_assessment,
            reflection_count: reflection_summary.reflection_count,
            adjustments_made: reflection_summary.adjustments_made,
            next_reflection_in: reflection_summary.next_reflection_in,

            // Adaptive behavior
            action_hint: self.adaptive_behavior.action_hint,
            speech_rate_multiplier: self.adaptive_behavior.speech_rate_multiplier,
            pause_multiplier: self.adaptive_behavior.pause_multiplier,
            learning_paused: self.adaptive_behavior.pause_learning,

            // Adapted thresholds
            flow_threshold: thresholds.flow_error,
            boredom_threshold: thresholds.boredom,
            trust_threshold: thresholds.trust,

            // Temporal coherence
            temporal_coherence: self.coherence_bridge.smoothed_coherence(),
            tau_mean: temporal_summary.features.mean,
            tau_trend: temporal_summary.features.trend,

            // ═══════════════════════════════════════════════════════════════════
            // MEGA-UNIFIED ARCHITECTURE FIELDS
            // ═══════════════════════════════════════════════════════════════════

            // Cognitive depth from thalamic routing
            cognitive_depth: self.cognitive_depth,

            // Unified Φ from the unification engine
            unified_phi: self.unification_engine.phi as f32,

            // Unified emotional state (VAD-based)
            unified_valence: self.unification_engine.emotional.state().valence as f32,
            unified_arousal: self.unification_engine.emotional.state().arousal as f32,
            unified_dominance: self.unification_engine.emotional.state().dominance as f32,
            unified_discrete_emotion: self.unification_engine.emotional.state().discrete_emotion,

            // Emotional pattern from the bridge
            emotional_pattern: self.unification_engine.emotional.detect_pattern(),

            // Natural language description of emotional state
            emotional_description: self.unification_engine.emotional.state().describe(),

            // ═══════════════════════════════════════════════════════════════════
            // TEMPORAL ENCODING FIELDS
            // ═══════════════════════════════════════════════════════════════════

            // Snapshot timestamp (nanoseconds since start)
            snapshot_timestamp_nanos: self.start_time.elapsed().as_nanos() as u64,

            // Flow temporal statistics
            current_flow_duration_secs: self.flow_state.current_flow_duration_secs(),
            total_flow_time_secs: self.flow_state.total_flow_time_with_current(),
            flow_periods: self.flow_state.flow_periods,
            avg_flow_duration_secs: self.flow_state.avg_flow_duration_secs,

            // FEP Active Inference metrics
            fep_free_energy: self.fep_agent.last_fe_components.as_ref().map(|fe| fe.total).unwrap_or(0.0),
            fep_precision: self.fep_agent.precision.perceptual_precision(),
        }
    }

    /// Get a concise status line for logging/display
    pub fn status_line(&self) -> String {
        self.consciousness_snapshot().status()
    }

    /// Check if system needs attention (via snapshot)
    pub fn snapshot_needs_attention(&self) -> bool {
        self.consciousness_snapshot().needs_attention()
    }

    /// Get current consciousness level (0.0 to 1.0)
    pub fn consciousness_level(&self) -> f32 {
        let (_, pattern_confidence) = self.temporal_signature_encoder.classify_state();
        ConsciousnessSnapshot::compute_consciousness_level(
            self.prediction_confidence,
            self.coherence_bridge.smoothed_coherence(),
            self.flow_state.intensity,
            pattern_confidence,
        )
    }

    // ========== Voice Feedback Methods ==========

    /// Update voice feedback with synthesis output metrics
    ///
    /// Call this after voice synthesis to feed quality metrics back into
    /// the cognitive loop, enabling self-regulating improvement.
    pub fn update_voice_feedback(&mut self, metrics: VoiceOutputMetrics) {
        self.voice_feedback_bridge.update(metrics);
    }

    /// Update listener prediction feedback
    ///
    /// Call this when listener comprehension data is available.
    /// 0.0 = complete misunderstanding, 1.0 = perfect prediction
    pub fn update_listener_prediction(&mut self, success: f32) {
        self.voice_feedback_bridge.update_listener_prediction(success);
    }

    /// Get voice quality summary for external systems
    pub fn voice_feedback_summary(&self) -> VoiceQualitySummary {
        self.voice_feedback_bridge.summary()
    }

    /// Check if voice indicates uncertainty (poor articulation or unstable rate)
    pub fn voice_indicates_uncertainty(&self) -> bool {
        self.voice_feedback_bridge.is_uncertain()
    }

    /// Get combined phi contribution from all feedback sources
    ///
    /// This combines:
    /// - Coherence phi contribution (from CfC temporal coherence)
    /// - Voice phi adjustment (from voice synthesis quality)
    pub fn combined_phi_contribution(&self) -> f32 {
        self.coherence_bridge.phi_contribution() + self.voice_feedback_bridge.compute_phi_adjustment()
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Get current cognitive depth from thalamic routing
    ///
    /// Returns the current processing depth:
    /// - Reflex: Fast pattern matching (<10ms)
    /// - Cortical: Standard cognitive cycle (50-200ms)
    /// - DeepThought: Full deliberation with causal reasoning (200ms+)
    pub fn cognitive_depth(&self) -> CognitiveDepth {
        self.cognitive_depth
    }

    /// Get the thalamic router reference
    pub fn thalamic_router(&self) -> &ThalamicRouter {
        &self.thalamic_router
    }

    /// Get thalamic routing statistics (reflex_rate, cortical_rate, deep_rate)
    pub fn thalamic_stats(&self) -> (f32, f32, f32) {
        self.thalamic_router.routing_stats()
    }

    /// Get the unified Φ from the ConsciousnessUnificationEngine
    pub fn unified_phi(&self) -> f64 {
        self.unification_engine.phi
    }

    /// Get the ConsciousnessUnificationEngine reference
    ///
    /// Provides access to:
    /// - EmotionalBridge (VAD emotional state)
    /// - UnifiedCausalReasoning
    /// - ConsciousDialoguePipeline
    pub fn unification_engine(&self) -> &ConsciousnessUnificationEngine {
        &self.unification_engine
    }

    /// Get mutable reference to the unification engine
    pub fn unification_engine_mut(&mut self) -> &mut ConsciousnessUnificationEngine {
        &mut self.unification_engine
    }

    /// Get the unified emotional state (VAD-based)
    pub fn unified_emotional_state(&self) -> &UnifiedEmotionalState {
        self.unification_engine.emotional.state()
    }

    /// Get the emotional pattern (Stable/Escalating/Calming/Volatile)
    pub fn emotional_pattern(&self) -> EmotionalPattern {
        self.unification_engine.emotional.detect_pattern()
    }

    /// Get natural language description of current emotional state
    pub fn emotional_description(&self) -> String {
        self.unification_engine.emotional.state().describe()
    }

    /// Get the discrete unified emotion
    pub fn unified_emotion(&self) -> Option<UnifiedEmotion> {
        self.unification_engine.emotional.state().discrete_emotion
    }

    /// Process input through the unified dialogue pipeline
    ///
    /// This uses the consciousness-aware dialogue generation that
    /// adapts depth (Reactive/Reflective/Integrative) based on Φ.
    pub fn process_unified(&mut self, input: &str) -> crate::consciousness::consciousness_unification::UnifiedConsciousnessResult {
        self.unification_engine.process(input)
    }

    /// Get a description of the current consciousness state
    pub fn unified_state_description(&self) -> String {
        self.unification_engine.describe_state()
    }

    /// Get the Active Inference Bridge reference
    pub fn active_inference_bridge(&self) -> &ActiveInferenceBridge {
        &self.active_inference_bridge
    }

    /// Get the FEP Active Inference Agent reference
    pub fn fep_agent(&self) -> &ActiveInferenceAgent {
        &self.fep_agent
    }

    /// Get the current FEP free energy (if available)
    pub fn fep_free_energy(&self) -> Option<f64> {
        self.fep_agent.last_fe_components.as_ref().map(|fe| fe.total)
    }

    /// Get the conversation coherence tracker reference
    pub fn coherence_tracker(&self) -> &ConversationCoherenceTracker {
        &self.coherence_tracker
    }

    /// Get the prediction-outcome coupling Modulation Index
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = No coupling (predictions don't inform outcomes)
    /// - 1.0 = Perfect coupling (confidence perfectly predicts success)
    pub fn modulation_index(&self) -> Option<f64> {
        self.active_inference_bridge.modulation_index()
    }

    /// Get the coupling quality assessment
    pub fn coupling_quality(&self) -> CouplingQuality {
        self.active_inference_bridge.coupling_quality()
    }

    /// Check if prediction-outcome coupling is meaningful
    pub fn has_meaningful_coupling(&self) -> bool {
        self.active_inference_bridge.coupling_quality().is_meaningful()
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // CLOSED LEARNING LOOP: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Get the current response strategy
    pub fn current_strategy(&self) -> ResponseStrategy {
        self.closed_learning_loop.current_strategy
    }

    /// Get the best strategy according to Q-learning
    pub fn best_strategy(&self) -> ResponseStrategy {
        self.closed_learning_loop.best_strategy()
    }

    /// Get the closed learning loop reference
    pub fn closed_learning_loop(&self) -> &ClosedLearningLoop {
        &self.closed_learning_loop
    }

    /// Get average reward from the learning loop
    pub fn average_reward(&self) -> f32 {
        self.closed_learning_loop.average_reward()
    }

    /// Get Q-values for all strategies
    pub fn strategy_q_values(&self) -> &[f32; 5] {
        self.closed_learning_loop.q_values()
    }

    /// Get strategy usage counts
    pub fn strategy_usage_counts(&self) -> &[u64; 5] {
        self.closed_learning_loop.strategy_counts()
    }

    /// Get the last learning result
    pub fn last_learning_result(&self) -> Option<&CycleLearningResult> {
        self.closed_learning_loop.last_result.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // MEMORY SYSTEM: Accessor Methods
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Get the episodic memory bridge reference
    pub fn episodic_memory(&self) -> &EpisodicMemoryBridge {
        &self.episodic_memory
    }

    /// Get mutable reference to episodic memory for manual operations
    pub fn episodic_memory_mut(&mut self) -> &mut EpisodicMemoryBridge {
        &mut self.episodic_memory
    }

    /// Get memory counts (short_term, long_term)
    pub fn memory_counts(&self) -> (usize, usize) {
        self.episodic_memory.memory_count()
    }

    /// Recall memories similar to input
    pub fn recall_memories(&mut self, query: &[f32], top_k: usize) -> Vec<(EpisodicMemory, f32)> {
        self.episodic_memory.recall(query, top_k, 0.2)
    }

    /// Get the goal system bridge reference
    pub fn goal_system(&self) -> &GoalSystemBridge {
        &self.goal_system
    }

    /// Get mutable reference to goal system
    pub fn goal_system_mut(&mut self) -> &mut GoalSystemBridge {
        &mut self.goal_system
    }

    /// Add a goal to the system
    pub fn add_goal(&mut self, id: &str, description: &str, priority: f32) {
        self.goal_system.add_goal(CognitiveGoal::new(id, description, priority));
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&CognitiveGoal> {
        self.goal_system.active_goals()
    }

    /// Get the world model bridge reference
    pub fn world_model(&self) -> &WorldModelBridge {
        &self.world_model
    }

    /// Get abstract level state from world model (for planning)
    pub fn world_model_abstract_state(&self) -> &[f32] {
        self.world_model.abstract_state()
    }

    /// Get world model prediction errors at each level
    pub fn world_model_level_errors(&self) -> &[f32] {
        self.world_model.level_errors()
    }

    /// Get combined learning rate modifier
    ///
    /// Returns a modifier (0.25 to 2.0) based on:
    /// - CfC coherence (higher coherence = higher rate)
    /// - Voice quality (higher quality = higher rate)
    pub fn combined_learning_rate(&self) -> f32 {
        let coherence_lr = self.coherence_bridge.effective_learning_rate();
        let voice_modifier = self.voice_feedback_bridge.learning_rate_modifier();

        // coherence_lr already includes base_lr × coherence_factor
        // voice_modifier is 0.5 to 1.0
        coherence_lr * voice_modifier
    }

    // ========== Consciousness Pattern Methods ==========

    /// Get current consciousness pattern classification
    ///
    /// Returns (pattern, confidence) where pattern is one of:
    /// Contemplative, Excited, Focused, Exploratory, Resting, Transitioning, Uncertain
    pub fn consciousness_pattern(&self) -> (ConsciousnessPattern, f32) {
        self.temporal_signature_encoder.classify_state()
    }

    /// Get full temporal state summary
    pub fn temporal_state_summary(&self) -> TemporalStateSummary {
        self.temporal_signature_encoder.summary()
    }

    /// Check if current state matches a specific consciousness pattern
    pub fn is_consciousness_state(&self, pattern: ConsciousnessPattern) -> bool {
        self.temporal_signature_encoder.is_state(pattern)
    }

    /// Get similarity to a specific consciousness pattern
    pub fn consciousness_pattern_similarity(&self, pattern: ConsciousnessPattern) -> f32 {
        self.temporal_signature_encoder.similarity_to(pattern)
    }

    // ========== Adaptive Behavior Methods ==========

    /// Get current adaptive behavior
    pub fn adaptive_behavior(&self) -> &AdaptiveBehavior {
        &self.adaptive_behavior
    }

    /// Get current action hint
    pub fn action_hint(&self) -> ActionHint {
        self.adaptive_behavior.action_hint
    }

    /// Check if system should seek more input/clarification
    pub fn should_seek_input(&self) -> bool {
        self.adaptive_behavior.should_seek_input()
    }

    /// Check if system is in a confident state
    pub fn is_confident(&self) -> bool {
        self.adaptive_behavior.is_confident()
    }

    /// Get description of current adaptive state
    pub fn state_description(&self) -> &'static str {
        self.adaptive_behavior.description()
    }

    /// Get speech rate multiplier for voice synthesis
    pub fn speech_rate_multiplier(&self) -> f32 {
        self.adaptive_behavior.speech_rate_multiplier
    }

    /// Get pause duration multiplier for voice synthesis
    pub fn pause_multiplier(&self) -> f32 {
        self.adaptive_behavior.pause_multiplier
    }

    /// Get attention sensitivity for input processing
    pub fn attention_sensitivity(&self) -> f32 {
        self.adaptive_behavior.attention_sensitivity
    }

    /// Get exploration factor for decision making
    pub fn exploration_factor(&self) -> f32 {
        self.adaptive_behavior.exploration_factor
    }

    /// Reset all learning state
    pub fn reset(&mut self) {
        self.encoder.reset_attention();
        // Reset CfC state by injecting zeros
        let zeros = Array1::from_vec(vec![0.0f32; self.config.cfc_config.input_dim]);
        let _ = self.temporal_network.inject(&zeros);
        self.buffer.clear();
        self.error_history.clear();
        self.last_state = None;
        self.last_prediction = None;
        self.stats = LoopStats::default();
        self.start_time = Instant::now();
        self.coherence_bridge.reset();
        self.voice_feedback_bridge.reset();
        self.temporal_signature_encoder.reset();
        self.adaptive_behavior = AdaptiveBehavior::default();
        self.prediction_confidence = 0.5; // Reset to neutral confidence
        self.flow_state.reset();
        self.emotion_contagion.reset();
        self.curiosity_drive.reset();
        self.self_reflection.reset(); // Preserves learned thresholds
        self.fep_agent = ActiveInferenceAgent::new(self.fep_agent.config.clone());
        self.coherence_tracker.reset();
    }

    /// Get the compressed state dimension (input to CfC)
    pub fn state_dim(&self) -> usize {
        self.config.cfc_config.input_dim
    }

    /// Get the prediction dimension (CfC neurons)
    pub fn prediction_dim(&self) -> usize {
        self.config.cfc_config.num_neurons
    }

    // ========== Internal Methods ==========

    fn create_experience(&mut self, state: &[f32], prediction: &[f32], error: f32) {
        // Update last experience with next_state
        if let Some(ref last_state) = self.last_state.take() {
            if let Some(last_pred) = self.last_prediction.take() {
                // Calculate importance based on error
                let importance = error + 0.1; // Base importance

                let exp = Experience {
                    state: last_state.clone(),
                    prediction: last_pred,
                    next_state: Some(state.to_vec()),
                    error,
                    importance,
                };

                if self.buffer.len() >= self.config.buffer_size {
                    self.buffer.pop_front();
                }
                self.buffer.push_back(exp);
            }
        }

        // Store current state for next cycle
        self.last_state = Some(state.to_vec());
        self.last_prediction = Some(prediction.to_vec());
    }

    fn update_stats(&mut self, error: f32, cycle_time: Duration) {
        // EMA for error
        let alpha = 0.1;
        self.stats.avg_prediction_error =
            self.stats.avg_prediction_error * (1.0 - alpha) + error * alpha;

        // Error trend
        self.error_history.push_back(error);
        if self.error_history.len() > 100 {
            self.error_history.pop_front();
        }
        self.stats.error_trend = self.compute_error_trend();

        // Attention stats from encoder
        let encoder_stats = self.encoder.stats();
        self.stats.attention_variance = encoder_stats.attention_variance;
        self.stats.diverged_primitives = encoder_stats.diverged_primitives;

        // Buffer utilization
        self.stats.buffer_utilization =
            self.buffer.len() as f32 / self.config.buffer_size as f32;

        // Timing stats
        let cycle_us = cycle_time.as_micros() as f32;
        self.stats.avg_cycle_time_us =
            self.stats.avg_cycle_time_us * 0.99 + cycle_us * 0.01;

        // Cycles per second
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.stats.cycles_per_second = self.stats.total_cycles as f32 / elapsed;
        }

        // CfC state diversity (already updated in cycle(), but ensure consistency)
        self.stats.ltc_consciousness = self.temporal_network.state_diversity();

        // Voice feedback stats
        let voice_summary = self.voice_feedback_bridge.summary();
        self.stats.voice_articulation_quality = voice_summary.articulation_quality;
        self.stats.voice_rate_stability = voice_summary.rate_stability;
        self.stats.voice_phi_adjustment = voice_summary.phi_adjustment;

        // Combined phi = coherence contribution + voice adjustment
        self.stats.combined_phi_contribution =
            self.stats.coherence_phi_contribution + self.stats.voice_phi_adjustment;

        // Consciousness pattern from temporal signatures
        let temporal_summary = self.temporal_signature_encoder.summary();
        self.stats.consciousness_pattern = format!("{:?}", temporal_summary.pattern);
        self.stats.pattern_confidence = temporal_summary.confidence;
        self.stats.tau_mean = temporal_summary.features.mean;
        self.stats.tau_trend = temporal_summary.features.trend;

        // Adaptive behavior stats
        self.stats.adaptive_confidence = self.adaptive_behavior.confidence;
        self.stats.action_hint = format!("{:?}", self.adaptive_behavior.action_hint);
        self.stats.learning_paused = self.adaptive_behavior.pause_learning;
        self.stats.adaptive_learning_rate = self.adaptive_behavior.effective_learning_rate(
            self.combined_learning_rate()
        );
        self.stats.adaptive_speech_rate = self.adaptive_behavior.speech_rate_multiplier;

        // Prediction confidence stats
        self.stats.prediction_confidence = self.prediction_confidence;
        // Decay rate: higher when in uncertain states
        self.stats.confidence_decay_rate = match self.adaptive_behavior.action_hint {
            ActionHint::Stabilize | ActionHint::SeekInput => 0.05,
            ActionHint::SlowDown => 0.03,
            _ => 0.0,
        };

        // Flow state stats
        self.stats.in_flow = self.flow_state.in_flow;
        self.stats.flow_intensity = self.flow_state.intensity;
        self.stats.flow_streak = self.flow_state.streak;
        self.stats.flow_learning_boost = self.flow_state.learning_boost;

        // Emotion contagion stats
        self.stats.emotional_valence = self.emotion_contagion.smoothed_valence();
        self.stats.emotional_arousal = self.emotion_contagion.smoothed_arousal();
        let (nudge_pattern, nudge_strength) = self.emotion_contagion.pattern_nudge();
        self.stats.emotion_nudge_pattern = nudge_pattern
            .map(|p| format!("{:?}", p))
            .unwrap_or_else(|| "None".to_string());
        self.stats.emotion_nudge_strength = nudge_strength;

        // Curiosity drive stats
        self.stats.boredom = self.curiosity_drive.boredom;
        self.stats.curiosity = self.curiosity_drive.curiosity;
        self.stats.exploration_urge = self.curiosity_drive.exploration_urge;
        self.stats.curiosity_exploring = self.curiosity_drive.should_explore();
        self.stats.novelty_bonus = self.curiosity_drive.novelty_bonus;

        // Self-reflection stats
        self.stats.self_assessment = format!("{:?}", self.self_reflection.self_assessment);
        self.stats.reflection_count = self.self_reflection.reflection_count;
        self.stats.adjustments_made = self.self_reflection.adjustments_made;
        self.stats.learning_effectiveness = self.self_reflection.learning_effectiveness();
        let summary = self.self_reflection.summary();
        self.stats.next_reflection_in = summary.next_reflection_in;
        self.stats.adapted_flow_threshold = self.self_reflection.flow_error_threshold;
        self.stats.adapted_boredom_threshold = self.self_reflection.boredom_threshold;

        // ═══════════════════════════════════════════════════════════════════════
        // MEGA-UNIFIED ARCHITECTURE STATS
        // ═══════════════════════════════════════════════════════════════════════

        // Cognitive depth from thalamic routing
        self.stats.cognitive_depth = format!("{:?}", self.cognitive_depth);

        // Unified Φ from the unification engine
        self.stats.unified_phi = self.unification_engine.phi as f32;

        // Unified emotional state (VAD)
        let unified_state = self.unification_engine.emotional.state();
        self.stats.unified_emotional_valence = unified_state.valence as f32;
        self.stats.unified_emotional_arousal = unified_state.arousal as f32;
        self.stats.unified_emotional_dominance = unified_state.dominance as f32;
        self.stats.unified_emotion = unified_state.discrete_emotion
            .map(|e| format!("{:?}", e))
            .unwrap_or_else(|| "Neutral".to_string());

        // Emotional pattern from the bridge
        self.stats.emotional_pattern = format!("{:?}", self.unification_engine.emotional.detect_pattern());

        // Thalamic routing statistics
        let (reflex_rate, cortical_rate, deep_rate) = self.thalamic_router.routing_stats();
        self.stats.thalamic_reflex_rate = reflex_rate;
        self.stats.thalamic_cortical_rate = cortical_rate;
        self.stats.thalamic_deep_rate = deep_rate;

        // Active Inference Bridge statistics
        let ai_stats = self.active_inference_bridge.statistics();
        self.stats.active_inference_modulation_index = ai_stats.modulation_index
            .map(|mi| mi as f32)
            .unwrap_or(0.0);
        self.stats.active_inference_coupling_quality = format!("{:?}", ai_stats.coupling_quality);
        self.stats.active_inference_avg_error = ai_stats.average_prediction_error
            .map(|e| e as f32)
            .unwrap_or(0.5);

        // Enhanced FEP Bridge statistics
        self.stats.fep_learning_signal = self.fep_learning_signal;
        // attention_shift is updated during cycle processing
        self.stats.fep_action_outcome_coupling = 0.5;  // Will be updated during cycle

        // Closed Learning Loop statistics
        self.stats.current_strategy = format!("{:?}", self.closed_learning_loop.current_strategy);
        self.stats.best_strategy = format!("{:?}", self.closed_learning_loop.best_strategy());
        self.stats.average_reward = self.closed_learning_loop.average_reward();
        self.stats.exploration_rate = self.closed_learning_loop.exploration_rate();
        self.stats.learning_loop_interactions = self.closed_learning_loop.total_interactions();

        // Memory system statistics
        let (short_term, long_term) = self.episodic_memory.memory_count();
        self.stats.memory_short_term_count = short_term;
        self.stats.memory_long_term_count = long_term;
        self.stats.memory_total_encoded = self.episodic_memory.stats.total_encoded;
        self.stats.world_model_avg_error = self.world_model.avg_error;
        self.stats.active_goals_count = self.goal_system.active_goals().len();
    }

    fn update_loss_stats(&mut self, loss: f32) {
        let alpha = 0.1;
        self.stats.avg_training_loss =
            self.stats.avg_training_loss * (1.0 - alpha) + loss * alpha;
    }

    fn compute_error_trend(&self) -> f32 {
        if self.error_history.len() < 10 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.error_history.len() as f32;
        let errors: Vec<f32> = self.error_history.iter().cloned().collect();

        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f32 = errors.iter().sum::<f32>() / n;

        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;

        for (i, &y) in errors.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() > 0.0001 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC INTEGRATION: MetricsProvider Implementation
// ═══════════════════════════════════════════════════════════════════════════════

use crate::shell::ipc_server::MetricsProvider;
use crate::shell::ipc_client::MetricsSnapshot;

impl MetricsProvider for CognitiveLoopService {
    fn get_metrics(&self) -> MetricsSnapshot {
        let phi = self.unification_engine.phi;
        let coherence = self.coherence_bridge.smoothed_coherence() as f64;
        MetricsSnapshot {
            phi,
            coherence,
            is_conscious: phi > 0.3,
            cognitive_depth: format!("{:?}", self.cognitive_depth),
            strategy: format!("{:?}", self.closed_learning_loop.current_strategy),
            in_flow: self.flow_state.in_flow,
            prediction_error: self.stats.avg_prediction_error,
            emotional_valence: self.emotion_contagion.prosody_valence(),
            emotional_arousal: self.emotion_contagion.prosody_arousal(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_cycles: self.stats.total_cycles as u64,
            consciousness_level: (phi + coherence) / 2.0,
            latency_ms: 0, // Updated by IPC layer
        }
    }

    fn phi(&self) -> f64 {
        self.unification_engine.phi
    }

    fn coherence(&self) -> f64 {
        self.coherence_bridge.smoothed_coherence() as f64
    }

    fn is_conscious(&self) -> bool {
        self.unification_engine.phi > 0.3
    }

    fn cognitive_depth(&self) -> String {
        format!("{:?}", self.cognitive_depth)
    }

    fn current_strategy(&self) -> String {
        format!("{:?}", self.closed_learning_loop.current_strategy)
    }

    fn in_flow(&self) -> bool {
        self.flow_state.in_flow
    }

    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn total_cycles(&self) -> u64 {
        self.stats.total_cycles as u64
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MFDI Identity Integration
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current agent ID (if identity is set)
    #[cfg(feature = "identity")]
    pub fn agent_id(&self) -> Option<&str> {
        self.mfdi_bridge.agent_id()
    }

    /// Get current assurance level
    #[cfg(feature = "identity")]
    pub fn assurance_level(&self) -> crate::identity::AssuranceLevel {
        self.mfdi_bridge.assurance_level()
    }

    /// Set identity from external verification
    #[cfg(feature = "identity")]
    pub fn set_identity(&mut self, identity: crate::identity::MfdiIdentity) {
        self.mfdi_bridge.set_identity(identity);
    }

    /// Check if a cognitive capability is allowed at current assurance level
    #[cfg(feature = "identity")]
    pub fn check_capability(&self, capability: crate::identity::CognitiveCapability) -> Result<()> {
        self.mfdi_bridge.check_capability(capability)
            .map_err(|e| anyhow::anyhow!("MFDI capability denied: {:?}", e))
    }

    /// Sign a cycle output
    #[cfg(feature = "identity")]
    pub fn sign_output(&self, output: &[f32]) -> Result<crate::identity::SignedOutput> {
        self.mfdi_bridge.sign_output(output)
            .map_err(|e| anyhow::anyhow!("MFDI signing failed: {:?}", e))
    }

    /// Verify a signed request
    #[cfg(feature = "identity")]
    pub fn verify_request(&mut self, request: &crate::identity::SignedRequest) -> Result<()> {
        self.mfdi_bridge.verify_request(request)
            .map_err(|e| anyhow::anyhow!("MFDI verification failed: {:?}", e))
    }

    /// Get mutable access to MFDI bridge for advanced operations
    #[cfg(feature = "identity")]
    pub fn mfdi_bridge_mut(&mut self) -> &mut crate::identity::MfdiBridge {
        &mut self.mfdi_bridge
    }
}


#[cfg(test)]
mod tests;
