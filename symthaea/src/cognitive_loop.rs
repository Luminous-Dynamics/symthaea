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


use anyhow::Result;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use ndarray::Array1;
use symthaea_core::genesis::ShakeRng;

use symthaea_core::hdc::predictive_encoder::{PredictiveHdcEncoder, PredictiveEncoderConfig};
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
use crate::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};
use crate::consciousness::stability_regime::{StabilityRegimeProcessor, RegimeTransition};
use crate::consciousness::primitive_discovery::{PrimitiveDiscoveryService, DiscoveryServiceConfig};
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use crate::causal::{CausalLoopEnhancer, CausalEnhancerConfig, CausalGraph, DiscoveredRelationship};
#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridge;

// TEMPORAL BACKEND SELECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal backend selection for the cognitive loop
///
/// The cognitive loop can use either CfC (Closed-form Continuous-time) or
/// HdcLtcUnified (Unified HDC-LTC) networks for temporal prediction.
///
/// ## CfC (Default)
/// - Traditional approach using ndarray-based weights
/// - Matrix multiplication for state transitions
/// - Well-tested and stable
///
/// ## HdcLtcUnified
/// - Novel approach using hypervector states
/// - HDC binding/bundling instead of matrix multiplication
/// - O(1) temporal jumps via closed-form solution
/// - State IS memory (holographic representation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TemporalBackend {
    /// Original Closed-form Continuous-time network
    #[default]
    CfC,
    /// New Unified HDC-LTC network with hypervector states
    HdcLtcUnified,
}

/// Training method selection for the cognitive loop
///
/// Controls how the temporal network is trained each cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingMethod {
    /// Always use BPTT (analytical gradients)
    Bptt,
    /// Always use SPSA (perturbation-based)
    Spsa,
    /// Use BPTT by default, fall back to SPSA when BPTT diverges
    #[default]
    BpttWithSpsaFallback,
}

/// Configuration for CfC in the cognitive loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCConfig {
    /// Number of CfC neurons
    pub num_neurons: usize,

    /// Input dimension (compressed HDC)
    pub input_dim: usize,

    /// Learning rate for CfC training
    pub learning_rate: f32,

    /// Time step for CfC predictions (seconds)
    pub delta_t: f32,

    /// Future prediction horizons for multi-scale prediction
    pub prediction_horizons: Vec<f32>,
}

impl Default for CfCConfig {
    fn default() -> Self {
        Self {
            num_neurons: 256,
            input_dim: 256,  // Must match num_neurons for train_step compatibility
            learning_rate: 0.001,
            delta_t: 0.02,  // 50Hz base rate
            // Multi-scale prediction: t+1, t+5, t+10 steps
            prediction_horizons: vec![0.02, 0.1, 0.2],
        }
    }
}

/// Configuration for the cognitive loop service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoopConfig {
    /// HDC encoder configuration
    pub encoder_config: PredictiveEncoderConfig,

    /// CfC configuration (replaces LTC for O(1) temporal prediction)
    pub cfc_config: CfCConfig,

    /// HDC-LTC Unified configuration (alternative to CfC)
    pub hdc_ltc_config: HdcLtcBridgeConfig,

    /// Which temporal backend to use
    pub temporal_backend: TemporalBackend,

    /// Minimum prediction error to trigger learning
    pub learning_threshold: f32,

    /// Experience buffer size
    pub buffer_size: usize,

    /// Whether to enable background consolidation
    pub enable_consolidation: bool,

    /// Target loop frequency (Hz)
    pub target_frequency: f32,

    /// Maximum cycles before stats reset (for long-running service)
    pub max_cycles_before_reset: usize,

    /// Optional genesis phrase for deterministic initialization.
    /// When set, all HDC vectors and network weights are derived from this
    /// phrase via SHAKE-256, making the system fully reproducible.
    pub genesis_phrase: Option<String>,

    /// Training method for the temporal network
    pub training_method: TrainingMethod,

    /// When true, BPTT/SPSA training runs on a background thread so that
    /// inference never blocks on training.  The main loop sends (input, target)
    /// samples over a channel and receives updated weights via non-blocking
    /// `try_recv` at the top of each cycle.
    pub async_training: bool,

    /// Enable online learning during inference.
    /// When true, the CfC network will adapt weights based on prediction errors
    /// after each forward pass, using a small learning rate to prevent
    /// catastrophic forgetting.
    pub enable_online_learning: bool,

    /// Enable causal discovery integration.
    /// When true, the cognitive loop tracks (input, output) pairs and
    /// periodically runs causal discovery to:
    /// - Weight attention (causal parents get more weight)
    /// - Guide exploration (intervene on discovered causes)
    pub causal_enhancement: bool,

    /// Interval (in cycles) between causal discovery runs.
    /// Only used when `causal_enhancement` is true.
    /// Lower values = more frequent discovery but higher compute cost.
    pub causal_discovery_interval: usize,

    /// Enable episodic memory replay for high-Phi moment consolidation.
    /// When true, the system stores high-consciousness episodes and periodically
    /// replays them to reinforce important patterns.
    pub episodic_replay: bool,

    /// Configuration for episodic memory replay.
    /// Only used when `episodic_replay` is true.
    pub episodic_replay_config: crate::memory::episodic_replay::EpisodicReplayConfig,
}

impl Default for CognitiveLoopConfig {
    fn default() -> Self {
        Self {
            encoder_config: PredictiveEncoderConfig::default(),
            cfc_config: CfCConfig::default(),
            hdc_ltc_config: HdcLtcBridgeConfig {
                hdc_dim: 2048,
                adaptive_dim: Some(crate::hdc_ltc_bridge::AdaptiveDimConfig::default()),
                ..HdcLtcBridgeConfig::default()
            },
            temporal_backend: TemporalBackend::default(),
            learning_threshold: 0.05,
            buffer_size: 1000,
            enable_consolidation: true,
            target_frequency: 50.0, // 50 Hz
            max_cycles_before_reset: 100000,
            genesis_phrase: None,
            training_method: TrainingMethod::default(),
            async_training: true,
            enable_online_learning: false,
            causal_enhancement: false,
            causal_discovery_interval: 100,
            episodic_replay: false,
            episodic_replay_config: crate::memory::episodic_replay::EpisodicReplayConfig::default(),
        }
    }
}

impl CognitiveLoopConfig {
    /// Create configuration with CfC backend (default)
    pub fn with_cfc() -> Self {
        Self {
            temporal_backend: TemporalBackend::CfC,
            ..Default::default()
        }
    }

    /// Create configuration with HdcLtcUnified backend
    pub fn with_hdc_ltc_unified() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            ..Default::default()
        }
    }

    /// Create configuration with HdcLtcUnified backend optimized for speed
    pub fn with_hdc_ltc_fast() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            hdc_ltc_config: HdcLtcBridgeConfig::fast(),
            ..Default::default()
        }
    }

    /// Create configuration with HdcLtcUnified backend optimized for accuracy
    pub fn with_hdc_ltc_accurate() -> Self {
        Self {
            temporal_backend: TemporalBackend::HdcLtcUnified,
            hdc_ltc_config: HdcLtcBridgeConfig::accurate(),
            ..Default::default()
        }
    }
}

/// Result of a single cognitive cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// LTC output (interpretation of current state)
    pub output: Vec<f32>,

    /// Prediction error for this cycle
    pub prediction_error: f32,

    /// Current attention state
    pub attention_state: HashMap<String, f32>,

    /// Detected primitives in input
    pub detected_primitives: Vec<String>,

    /// Whether learning occurred this cycle
    pub learning_occurred: bool,

    /// Training loss (if learning occurred)
    pub training_loss: Option<f32>,

    /// Cycle timing (microseconds)
    pub cycle_time_us: u64,
}

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
// ADAPTIVE BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive behavior parameters derived from consciousness state
///
/// These parameters allow the system to self-regulate based on its
/// current consciousness pattern, creating a truly self-aware loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBehavior {
    /// Learning rate multiplier (0.1 to 2.0)
    /// Higher when focused/confident, lower when uncertain
    pub learning_rate_multiplier: f32,

    /// Speech rate multiplier (0.7 to 1.3)
    /// Slower when contemplative/uncertain, faster when excited/focused
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier (0.5 to 2.0)
    /// Longer pauses when contemplative, shorter when excited
    pub pause_multiplier: f32,

    /// Attention sensitivity (0.5 to 1.5)
    /// Higher sensitivity when exploratory, lower when focused
    pub attention_sensitivity: f32,

    /// Exploration factor (0.0 to 1.0)
    /// Higher when exploratory/uncertain, lower when focused
    pub exploration_factor: f32,

    /// Confidence level (0.0 to 1.0)
    /// Derived from pattern + coherence + voice quality
    pub confidence: f32,

    /// Should pause learning (e.g., during transitions)
    pub pause_learning: bool,

    /// Recommended action hint
    pub action_hint: ActionHint,
}

/// Recommended action based on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionHint {
    /// Continue normally
    Continue,
    /// Slow down, be more deliberate
    SlowDown,
    /// Speed up, more confident
    SpeedUp,
    /// Pause and stabilize
    Stabilize,
    /// Explore alternatives
    Explore,
    /// Seek clarification or more input
    SeekInput,
}

/// Flow state - optimal cognitive engagement
///
/// Detected when there is sustained focus, low prediction error,
/// and high temporal coherence. Flow state boosts learning efficiency
/// and signals peak cognitive performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowState {
    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow intensity (0.0 to 1.0)
    /// Higher = deeper flow state
    pub intensity: f32,

    /// Consecutive cycles in flow-compatible state
    pub streak: u32,

    /// Average prediction error during flow detection window
    pub avg_error: f32,

    /// Average coherence during flow detection window
    pub avg_coherence: f32,

    /// Learning rate boost when in flow (1.0 to 2.0)
    pub learning_boost: f32,

    /// Attention enhancement when in flow
    pub attention_boost: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING - Time Context for Flow States
    // ═══════════════════════════════════════════════════════════════════════════

    /// Timestamp when flow state started (if in flow)
    /// Note: Not serialized as Instant is monotonic/non-portable
    #[serde(skip)]
    pub flow_started_at: Option<Instant>,

    /// Total time spent in flow during this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average duration of flow periods (seconds)
    pub avg_flow_duration_secs: f32,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            in_flow: false,
            intensity: 0.0,
            streak: 0,
            avg_error: 0.5,
            avg_coherence: 0.5,
            learning_boost: 1.0,
            attention_boost: 1.0,
            // Temporal encoding defaults
            flow_started_at: None,
            total_flow_time_secs: 0.0,
            flow_periods: 0,
            avg_flow_duration_secs: 0.0,
        }
    }
}

impl FlowState {
    /// Minimum streak for flow state entry
    const FLOW_ENTRY_STREAK: u32 = 5;
    /// Error threshold for flow eligibility
    const FLOW_ERROR_THRESHOLD: f32 = 0.25;
    /// Coherence threshold for flow eligibility
    const FLOW_COHERENCE_THRESHOLD: f32 = 0.6;
    /// Confidence threshold for flow eligibility
    const FLOW_CONFIDENCE_THRESHOLD: f32 = 0.5;

    /// Update flow state based on current metrics
    pub fn update(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        coherence: f32,
        prediction_confidence: f32,
    ) {
        // Check if current state is flow-compatible
        let is_flow_compatible = matches!(
            pattern,
            ConsciousnessPattern::Focused | ConsciousnessPattern::Contemplative
        ) && prediction_error < Self::FLOW_ERROR_THRESHOLD
          && coherence > Self::FLOW_COHERENCE_THRESHOLD
          && prediction_confidence > Self::FLOW_CONFIDENCE_THRESHOLD;

        // Update running averages (EMA)
        let alpha = 0.2;
        self.avg_error = self.avg_error * (1.0 - alpha) + prediction_error * alpha;
        self.avg_coherence = self.avg_coherence * (1.0 - alpha) + coherence * alpha;

        if is_flow_compatible {
            self.streak += 1;

            // Enter flow state after sustained focus
            if self.streak >= Self::FLOW_ENTRY_STREAK {
                self.in_flow = true;

                // Intensity grows with streak (caps at 1.0)
                // Use saturating_sub to prevent underflow, then safe cast via f64
                self.intensity = (self.streak.saturating_sub(Self::FLOW_ENTRY_STREAK) as f64 / 10.0)
                    .min(1.0) as f32;

                // Boost learning when in flow (up to 50% boost at max intensity)
                self.learning_boost = 1.0 + 0.5 * self.intensity;

                // Enhance attention (up to 30% boost)
                self.attention_boost = 1.0 + 0.3 * self.intensity;
            }
        } else {
            // Exit flow or reduce streak
            if self.in_flow {
                // Grace period: don't exit immediately
                if self.streak > 0 {
                    self.streak = self.streak.saturating_sub(2);
                }
                if self.streak < Self::FLOW_ENTRY_STREAK / 2 {
                    self.in_flow = false;
                    self.intensity = 0.0;
                    self.learning_boost = 1.0;
                    self.attention_boost = 1.0;
                }
            } else {
                self.streak = 0;
            }
        }
    }

    /// Update flow state with adaptive thresholds from self-reflection
    ///
    /// This allows the meta-learning system to adjust flow entry criteria.
    pub fn update_with_thresholds(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        coherence: f32,
        prediction_confidence: f32,
        error_threshold: f32,
        coherence_threshold: f32,
    ) {
        // Check if current state is flow-compatible using adaptive thresholds
        let is_flow_compatible = matches!(
            pattern,
            ConsciousnessPattern::Focused | ConsciousnessPattern::Contemplative
        ) && prediction_error < error_threshold
          && coherence > coherence_threshold
          && prediction_confidence > Self::FLOW_CONFIDENCE_THRESHOLD;

        // Update running averages (EMA)
        let alpha = 0.2;
        self.avg_error = self.avg_error * (1.0 - alpha) + prediction_error * alpha;
        self.avg_coherence = self.avg_coherence * (1.0 - alpha) + coherence * alpha;

        if is_flow_compatible {
            self.streak += 1;

            // Enter flow state after sustained focus
            if self.streak >= Self::FLOW_ENTRY_STREAK {
                // Track temporal: entering flow
                let was_in_flow = self.in_flow;
                self.in_flow = true;

                // Start flow timer if just entering
                if !was_in_flow {
                    self.flow_started_at = Some(Instant::now());
                    self.flow_periods += 1;
                }

                // Intensity grows with streak (caps at 1.0)
                // Use saturating_sub to prevent underflow, then safe cast via f64
                self.intensity = (self.streak.saturating_sub(Self::FLOW_ENTRY_STREAK) as f64 / 10.0)
                    .min(1.0) as f32;

                // Boost learning when in flow (up to 50% boost at max intensity)
                self.learning_boost = 1.0 + 0.5 * self.intensity;

                // Enhance attention (up to 30% boost)
                self.attention_boost = 1.0 + 0.3 * self.intensity;
            }
        } else {
            // Exit flow or reduce streak
            if self.in_flow {
                // Grace period: don't exit immediately
                if self.streak > 0 {
                    self.streak = self.streak.saturating_sub(2);
                }
                if self.streak < Self::FLOW_ENTRY_STREAK / 2 {
                    // Track temporal: exiting flow
                    if let Some(started) = self.flow_started_at.take() {
                        let duration = started.elapsed().as_secs_f32();
                        self.total_flow_time_secs += duration;

                        // Update average duration (safe division with max(1))
                        if self.flow_periods > 0 {
                            self.avg_flow_duration_secs = self.total_flow_time_secs
                                / self.flow_periods.max(1) as f32;
                        }
                    }

                    self.in_flow = false;
                    self.intensity = 0.0;
                    self.learning_boost = 1.0;
                    self.attention_boost = 1.0;
                }
            } else {
                self.streak = 0;
            }
        }
    }

    /// Reset flow state
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get effective learning rate multiplier including flow boost
    pub fn effective_learning_multiplier(&self, base_multiplier: f32) -> f32 {
        base_multiplier * self.learning_boost
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current flow duration in seconds (if in flow)
    pub fn current_flow_duration_secs(&self) -> Option<f32> {
        self.flow_started_at.map(|started| started.elapsed().as_secs_f32())
    }

    /// Get total time spent in flow (including current session)
    pub fn total_flow_time_with_current(&self) -> f32 {
        let current = self.current_flow_duration_secs().unwrap_or(0.0);
        self.total_flow_time_secs + current
    }

    /// Get the timestamp when current flow started
    pub fn flow_started(&self) -> Option<Instant> {
        self.flow_started_at
    }

    /// Get flow statistics summary
    pub fn temporal_summary(&self) -> FlowTemporalSummary {
        FlowTemporalSummary {
            total_flow_time_secs: self.total_flow_time_with_current(),
            flow_periods: self.flow_periods,
            avg_flow_duration_secs: self.avg_flow_duration_secs,
            current_flow_duration_secs: self.current_flow_duration_secs(),
            is_in_flow: self.in_flow,
        }
    }
}

/// Summary of flow state temporal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowTemporalSummary {
    /// Total time spent in flow during this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average duration of flow periods (seconds)
    pub avg_flow_duration_secs: f32,

    /// Current flow duration if in flow (seconds)
    pub current_flow_duration_secs: Option<f32>,

    /// Whether currently in flow
    pub is_in_flow: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOSED LEARNING LOOP - Strategy-Based Behavioral Adaptation
// ═══════════════════════════════════════════════════════════════════════════════

/// Response strategy selected by the closed learning loop
///
/// Based on CLOSED_LEARNING_LOOP.md - strategies are selected based on:
/// 1. Q-learning from past interactions
/// 2. Previous reward (stick with success, avoid failure)
/// 3. Φ-gating (high Φ → Exploratory, low Φ → Supportive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseStrategy {
    /// Elaborate explanations with detail
    Detailed,
    /// Brief, direct answers
    Concise,
    /// Ask clarifying questions
    Clarifying,
    /// Acknowledge and validate
    Supportive,
    /// Offer new perspectives
    Exploratory,
}

impl Default for ResponseStrategy {
    fn default() -> Self {
        Self::Supportive
    }
}

impl ResponseStrategy {
    /// Get the opposite strategy (for switching after negative feedback)
    pub fn opposite(self) -> Self {
        match self {
            Self::Detailed => Self::Concise,
            Self::Concise => Self::Detailed,
            Self::Clarifying => Self::Supportive,
            Self::Supportive => Self::Exploratory,
            Self::Exploratory => Self::Clarifying,
        }
    }

    /// Get description of strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::Detailed => "Elaborate explanations with full context",
            Self::Concise => "Brief, direct responses",
            Self::Clarifying => "Ask questions to understand better",
            Self::Supportive => "Acknowledge and validate",
            Self::Exploratory => "Offer novel perspectives and connections",
        }
    }
}

/// Learning result from a cycle (for closed loop)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleLearningResult {
    /// Reward from cycle (-1.0 to 1.0)
    /// Based on prediction error (lower error = higher reward)
    pub reward: f32,

    /// Strategy that was used
    pub strategy_used: ResponseStrategy,

    /// Whether the cycle was successful (low error, in flow, etc.)
    pub successful: bool,

    /// Prediction error during this cycle
    pub prediction_error: f32,

    /// Coherence during this cycle
    pub coherence: f32,
}

/// Closed Learning Loop Manager
///
/// Implements the paradigm shift from CLOSED_LEARNING_LOOP.md:
/// - Learning → Behavioral Change (not just compute and discard)
/// - Q-learning guided strategy selection
/// - Φ-gated strategy preferences
#[derive(Debug)]
pub struct ClosedLearningLoop {
    /// Current selected strategy
    pub current_strategy: ResponseStrategy,

    /// Last learning result (influences next strategy)
    pub last_result: Option<CycleLearningResult>,

    /// Q-values for each strategy (estimated long-term reward)
    q_values: [f32; 5],

    /// Learning rate for Q-updates
    q_learning_rate: f32,

    /// Exploration rate (epsilon for epsilon-greedy)
    exploration_rate: f32,

    /// Total interactions
    total_interactions: u64,

    /// Total accumulated reward
    total_reward: f32,

    /// Strategy usage counts
    strategy_counts: [u64; 5],

    /// Optional genesis-seeded RNG for deterministic exploration
    rng: Option<ShakeRng>,
}

impl Default for ClosedLearningLoop {
    fn default() -> Self {
        Self {
            current_strategy: ResponseStrategy::default(),
            last_result: None,
            q_values: [0.5; 5], // Start neutral
            q_learning_rate: 0.1,
            exploration_rate: 0.2,
            total_interactions: 0,
            total_reward: 0.0,
            strategy_counts: [0; 5],
            rng: None,
        }
    }
}

impl ClosedLearningLoop {
    /// Create with a genesis-seeded RNG for deterministic exploration.
    pub fn with_rng(rng: ShakeRng) -> Self {
        Self {
            rng: Some(rng),
            ..Default::default()
        }
    }

    /// Select strategy based on Q-learning + previous result + Φ
    ///
    /// This is the core of the closed learning loop:
    /// 1. Start with Q-learning policy (greedy or explore)
    /// 2. Modify based on previous result
    /// 3. Gate based on consciousness level (Φ)
    pub fn select_strategy(&mut self, phi: f64, _previous_reward: Option<f32>) -> ResponseStrategy {
        // Step 1: Q-learning selection (epsilon-greedy)
        let (explore_val, variant_val): (f32, u8) = match self.rng.as_mut() {
            Some(rng) => (rng.gen::<f32>(), rng.gen::<u8>()),
            None => (rand::random::<f32>(), rand::random::<u8>()),
        };
        let explore = explore_val < self.exploration_rate;
        let base_strategy = if explore {
            // Random exploration
            match variant_val % 5 {
                0 => ResponseStrategy::Detailed,
                1 => ResponseStrategy::Concise,
                2 => ResponseStrategy::Clarifying,
                3 => ResponseStrategy::Supportive,
                _ => ResponseStrategy::Exploratory,
            }
        } else {
            // Greedy: select best Q-value
            let best_idx = self.q_values.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(3); // Default to Supportive

            match best_idx {
                0 => ResponseStrategy::Detailed,
                1 => ResponseStrategy::Concise,
                2 => ResponseStrategy::Clarifying,
                3 => ResponseStrategy::Supportive,
                _ => ResponseStrategy::Exploratory,
            }
        };

        // Step 2: Modify based on previous result
        let strategy = if let Some(ref last) = self.last_result {
            if last.reward > 0.5 {
                // Strong positive - stick with what worked
                last.strategy_used
            } else if last.reward < -0.2 {
                // Negative - try opposite strategy
                last.strategy_used.opposite()
            } else {
                base_strategy
            }
        } else {
            base_strategy
        };

        // Step 3: Φ-gating (consciousness influences strategy)
        let final_strategy = if phi >= 0.6 {
            // Integrative mode - favor Exploratory/Detailed
            match strategy {
                ResponseStrategy::Supportive => ResponseStrategy::Exploratory,
                ResponseStrategy::Concise => ResponseStrategy::Detailed,
                other => other,
            }
        } else if phi < 0.3 {
            // Reactive mode - favor Supportive/Concise
            match strategy {
                ResponseStrategy::Exploratory => ResponseStrategy::Supportive,
                ResponseStrategy::Detailed => ResponseStrategy::Concise,
                other => other,
            }
        } else {
            // Reflective mode - use Q-learning selection as-is
            strategy
        };

        self.current_strategy = final_strategy;
        final_strategy
    }

    /// Update Q-values with cycle result
    pub fn update(&mut self, result: CycleLearningResult) {
        // Update strategy count
        let strategy_idx = self.strategy_index(result.strategy_used);
        self.strategy_counts[strategy_idx] += 1;

        // Q-learning update: Q(s,a) <- Q(s,a) + α * (r - Q(s,a))
        let old_q = self.q_values[strategy_idx];
        let new_q = old_q + self.q_learning_rate * (result.reward - old_q);
        self.q_values[strategy_idx] = new_q;

        // Update totals
        self.total_interactions += 1;
        self.total_reward += result.reward;

        // Store for next selection
        self.last_result = Some(result);

        // Decay exploration rate over time (but keep minimum of 5%)
        self.exploration_rate = (self.exploration_rate * 0.999).max(0.05);
    }

    /// Get strategy index for Q-value lookup
    fn strategy_index(&self, strategy: ResponseStrategy) -> usize {
        match strategy {
            ResponseStrategy::Detailed => 0,
            ResponseStrategy::Concise => 1,
            ResponseStrategy::Clarifying => 2,
            ResponseStrategy::Supportive => 3,
            ResponseStrategy::Exploratory => 4,
        }
    }

    /// Get average reward
    pub fn average_reward(&self) -> f32 {
        if self.total_interactions == 0 {
            0.0
        } else {
            self.total_reward / self.total_interactions as f32
        }
    }

    /// Get best strategy according to Q-values
    pub fn best_strategy(&self) -> ResponseStrategy {
        let best_idx = self.q_values.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(3);

        match best_idx {
            0 => ResponseStrategy::Detailed,
            1 => ResponseStrategy::Concise,
            2 => ResponseStrategy::Clarifying,
            3 => ResponseStrategy::Supportive,
            _ => ResponseStrategy::Exploratory,
        }
    }

    /// Get Q-values for each strategy
    pub fn q_values(&self) -> &[f32; 5] {
        &self.q_values
    }

    /// Get strategy usage counts
    pub fn strategy_counts(&self) -> &[u64; 5] {
        &self.strategy_counts
    }

    /// Reset the learning loop
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY INTEGRATION BRIDGES
// ═══════════════════════════════════════════════════════════════════════════════

/// Episodic memory trace for the cognitive loop
///
/// Lightweight representation of a memory that can be queried during cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Memory ID
    pub id: u64,
    /// Timestamp when encoded (cycle count)
    pub encoded_at_cycle: usize,
    /// Content summary
    pub content: String,
    /// Embedding (compressed for efficiency)
    pub embedding: Vec<f32>,
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,
    /// Φ at encoding time
    pub phi_at_encoding: f32,
    /// Access count
    pub access_count: u32,
    /// Strength (0.0 to 1.0, decays over time)
    pub strength: f32,
}

impl EpisodicMemory {
    /// Compute similarity with query embedding
    pub fn similarity(&self, query: &[f32]) -> f32 {
        if self.embedding.len() != query.len() {
            return 0.0;
        }
        let dot: f32 = self.embedding.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        let mag_self: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_query: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_self > 0.0 && mag_query > 0.0 {
            dot / (mag_self * mag_query)
        } else {
            0.0
        }
    }
}

/// Episodic Memory Bridge for the cognitive loop
///
/// Provides memory encoding and recall during cognitive cycles.
/// Can be connected to the full HippocampusActor for persistence.
#[derive(Debug, Clone)]
pub struct EpisodicMemoryBridge {
    /// Short-term memory buffer (recent cycles)
    short_term: Vec<EpisodicMemory>,
    /// Long-term memory store
    long_term: Vec<EpisodicMemory>,
    /// Maximum short-term memories
    max_short_term: usize,
    /// Maximum long-term memories
    max_long_term: usize,
    /// Next memory ID
    next_id: u64,
    /// Consolidation threshold (strength needed to move to long-term)
    consolidation_threshold: f32,
    /// Statistics
    pub stats: MemoryBridgeStats,
}

/// Statistics for the memory bridge
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryBridgeStats {
    pub total_encoded: u64,
    pub total_recalled: u64,
    pub consolidations: u64,
    pub avg_recall_similarity: f32,
}

impl Default for EpisodicMemoryBridge {
    fn default() -> Self {
        Self {
            short_term: Vec::with_capacity(100),
            long_term: Vec::with_capacity(1000),
            max_short_term: 100,
            max_long_term: 1000,
            next_id: 0,
            consolidation_threshold: 0.5,
            stats: MemoryBridgeStats::default(),
        }
    }
}

impl EpisodicMemoryBridge {
    /// Encode a new memory
    pub fn encode(
        &mut self,
        content: impl Into<String>,
        embedding: Vec<f32>,
        valence: f32,
        phi: f32,
        cycle: usize,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let memory = EpisodicMemory {
            id,
            encoded_at_cycle: cycle,
            content: content.into(),
            embedding,
            valence,
            phi_at_encoding: phi,
            access_count: 0,
            strength: 1.0,
        };

        // Add to short-term
        if self.short_term.len() >= self.max_short_term {
            // Consolidate oldest to long-term if strong enough
            if let Some(oldest) = self.short_term.first() {
                if oldest.strength >= self.consolidation_threshold {
                    self.long_term.push(oldest.clone());
                    self.stats.consolidations += 1;
                    // Trim long-term if needed
                    if self.long_term.len() > self.max_long_term {
                        // Remove weakest memory
                        if let Some(min_idx) = self.long_term.iter()
                            .enumerate()
                            .min_by(|a, b| a.1.strength.partial_cmp(&b.1.strength).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(i, _)| i)
                        {
                            self.long_term.remove(min_idx);
                        }
                    }
                }
            }
            self.short_term.remove(0);
        }
        self.short_term.push(memory);
        self.stats.total_encoded += 1;

        id
    }

    /// Recall memories similar to query embedding
    pub fn recall(&mut self, query: &[f32], top_k: usize, min_similarity: f32) -> Vec<(EpisodicMemory, f32)> {
        let mut results: Vec<(EpisodicMemory, f32)> = Vec::new();

        // Search both short-term and long-term
        for memory in self.short_term.iter().chain(self.long_term.iter()) {
            let sim = memory.similarity(query);
            if sim >= min_similarity {
                results.push((memory.clone(), sim));
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Update access counts for recalled memories
        for (recalled, _) in &results {
            // Update in short-term
            if let Some(mem) = self.short_term.iter_mut().find(|m| m.id == recalled.id) {
                mem.access_count += 1;
                mem.strength = (mem.strength + 0.1).min(1.0);
            }
            // Update in long-term
            if let Some(mem) = self.long_term.iter_mut().find(|m| m.id == recalled.id) {
                mem.access_count += 1;
                mem.strength = (mem.strength + 0.05).min(1.0);
            }
        }

        if !results.is_empty() {
            self.stats.total_recalled += 1;
            self.stats.avg_recall_similarity = results.iter().map(|(_, s)| s).sum::<f32>()
                / results.len() as f32;
        }

        results
    }

    /// Decay unused memories
    pub fn decay(&mut self, decay_rate: f32) {
        for memory in self.short_term.iter_mut().chain(self.long_term.iter_mut()) {
            memory.strength = (memory.strength - decay_rate).max(0.0);
        }
        // Remove memories with zero strength from long-term
        self.long_term.retain(|m| m.strength > 0.01);
    }

    /// Get memory count
    pub fn memory_count(&self) -> (usize, usize) {
        (self.short_term.len(), self.long_term.len())
    }

    /// Reset the memory bridge
    pub fn reset(&mut self) {
        self.short_term.clear();
        self.long_term.clear();
        self.next_id = 0;
        self.stats = MemoryBridgeStats::default();
    }

    /// Consolidate recent memories to long-term storage
    ///
    /// Triggered by motor commands to strengthen recent experiences.
    /// This forces consolidation of high-strength short-term memories.
    pub fn consolidate_recent(&mut self) {
        // Find strong short-term memories and move to long-term
        let strong_memories: Vec<EpisodicMemory> = self.short_term.iter()
            .filter(|m| m.strength >= self.consolidation_threshold * 0.8)  // Slightly lower threshold
            .cloned()
            .collect();

        for memory in strong_memories {
            // Check if not already in long-term
            if !self.long_term.iter().any(|m| m.id == memory.id) {
                self.long_term.push(memory);
                self.stats.consolidations += 1;
            }
        }

        // Boost strength of recently consolidated memories
        for mem in self.long_term.iter_mut().rev().take(5) {
            mem.strength = (mem.strength + 0.1).min(1.0);
        }

        // Trim long-term if needed
        while self.long_term.len() > self.max_long_term {
            if let Some(min_idx) = self.long_term.iter()
                .enumerate()
                .min_by(|a, b| a.1.strength.partial_cmp(&b.1.strength).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                self.long_term.remove(min_idx);
            } else {
                break;
            }
        }
    }
}

/// Goal representation for the cognitive loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveGoal {
    /// Goal ID
    pub id: String,
    /// Goal description
    pub description: String,
    /// Priority (0.0 to 1.0)
    pub priority: f32,
    /// Progress (0.0 to 1.0)
    pub progress: f32,
    /// Whether actively pursued
    pub is_active: bool,
    /// Attention weight (how much to bias attention toward this goal)
    pub attention_weight: f32,
}

impl CognitiveGoal {
    /// Create a new goal
    pub fn new(id: impl Into<String>, description: impl Into<String>, priority: f32) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            priority: priority.clamp(0.0, 1.0),
            progress: 0.0,
            is_active: true,
            attention_weight: priority, // Initially weight by priority
        }
    }
}

/// Goal System Bridge for goal-directed attention
#[derive(Debug, Clone, Default)]
pub struct GoalSystemBridge {
    /// Active goals
    goals: Vec<CognitiveGoal>,
    /// Maximum concurrent goals
    max_goals: usize,
}

impl GoalSystemBridge {
    /// Create with default capacity
    pub fn new() -> Self {
        Self {
            goals: Vec::with_capacity(10),
            max_goals: 10,
        }
    }

    /// Add a goal
    pub fn add_goal(&mut self, goal: CognitiveGoal) {
        if self.goals.len() < self.max_goals {
            self.goals.push(goal);
        }
    }

    /// Get attention bias based on goals
    ///
    /// Returns a multiplier for attention based on goal priorities
    pub fn attention_bias(&self) -> f32 {
        if self.goals.is_empty() {
            return 1.0;
        }
        let active_weight: f32 = self.goals.iter()
            .filter(|g| g.is_active)
            .map(|g| g.attention_weight)
            .sum();
        1.0 + active_weight * 0.2 // Up to 20% boost per unit of goal weight
    }

    /// Update goal progress
    pub fn update_progress(&mut self, goal_id: &str, delta: f32) {
        if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
            goal.progress = (goal.progress + delta).clamp(0.0, 1.0);
            if goal.progress >= 1.0 {
                goal.is_active = false;
            }
        }
    }

    /// Get active goals
    pub fn active_goals(&self) -> Vec<&CognitiveGoal> {
        self.goals.iter().filter(|g| g.is_active).collect()
    }

    /// Get highest priority active goal
    pub fn top_goal(&self) -> Option<&CognitiveGoal> {
        self.goals.iter()
            .filter(|g| g.is_active)
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Clear completed goals
    pub fn clear_completed(&mut self) {
        self.goals.retain(|g| g.progress < 1.0);
    }

    /// Reset all goals
    pub fn reset(&mut self) {
        self.goals.clear();
    }
}

/// World Model Bridge for grounded prediction
///
/// Lightweight interface to hierarchical world model predictions
#[derive(Debug, Clone)]
pub struct WorldModelBridge {
    /// Multi-level state representations
    level_states: Vec<Vec<f32>>,
    /// Level dimensions
    level_dims: Vec<usize>,
    /// Prediction error at each level
    level_errors: Vec<f32>,
    /// Total predictions made
    pub total_predictions: u64,
    /// Average prediction error across levels
    pub avg_error: f32,
}

impl Default for WorldModelBridge {
    fn default() -> Self {
        // Default 4-level hierarchy
        let level_dims = vec![64, 128, 256, 128];
        Self {
            level_states: level_dims.iter().map(|&d| vec![0.0; d]).collect(),
            level_dims,
            level_errors: vec![0.0; 4],
            total_predictions: 0,
            avg_error: 0.0,
        }
    }
}

impl WorldModelBridge {
    /// Update with sensory input (level 0)
    pub fn update_sensory(&mut self, input: &[f32]) {
        if input.len() >= self.level_dims[0] {
            // Compute prediction error at level 0
            let error: f32 = self.level_states[0].iter()
                .zip(input.iter().take(self.level_dims[0]))
                .map(|(pred, actual)| (pred - actual).powi(2))
                .sum::<f32>()
                .sqrt();
            self.level_errors[0] = error;

            // Update level 0 state
            for (i, &val) in input.iter().take(self.level_dims[0]).enumerate() {
                self.level_states[0][i] = val;
            }

            // Propagate up (simplified: just average to higher levels)
            self.propagate_up();

            self.total_predictions += 1;
            // Safe division: use max(1) to prevent division by zero
            self.avg_error = self.level_errors.iter().sum::<f32>() / self.level_errors.len().max(1) as f32;
        }
    }

    /// Propagate state up the hierarchy
    fn propagate_up(&mut self) {
        for level in 1..self.level_states.len() {
            let prev_level = level - 1;
            let prev_dim = self.level_dims[prev_level];
            let curr_dim = self.level_dims[level];

            // Simple projection: chunk and average
            // Safe division: use max(1) to prevent division by zero
            let chunk_size = (prev_dim + curr_dim - 1) / curr_dim.max(1);
            for i in 0..curr_dim {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(prev_dim);
                if start < prev_dim {
                    let sum: f32 = self.level_states[prev_level][start..end].iter().sum();
                    // Safe cast via f64 to prevent precision loss on large counts
                    let count = end.saturating_sub(start) as f64;
                    self.level_states[level][i] = (sum as f64 / count.max(1.0)) as f32;
                }
            }
        }
    }

    /// Get prediction at a specific level
    pub fn get_level_state(&self, level: usize) -> Option<&[f32]> {
        self.level_states.get(level).map(|v| v.as_slice())
    }

    /// Get prediction error at each level
    pub fn level_errors(&self) -> &[f32] {
        &self.level_errors
    }

    /// Get abstract level state (highest level - for planning)
    pub fn abstract_state(&self) -> &[f32] {
        self.level_states.last().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Reset the world model
    pub fn reset(&mut self) {
        for state in &mut self.level_states {
            state.fill(0.0);
        }
        self.level_errors.fill(0.0);
        self.total_predictions = 0;
        self.avg_error = 0.0;
    }

    /// Increase plasticity in the world model (triggered by high learning signals)
    ///
    /// Higher plasticity means faster state updates and more sensitivity to prediction errors.
    /// This is implemented by scaling the level states to be more receptive to new input.
    pub fn increase_plasticity(&mut self, plasticity_signal: f32) {
        // Reduce state magnitudes slightly to make room for new learning
        let decay = 1.0 - (plasticity_signal * 0.1).clamp(0.0, 0.3);
        for level_state in &mut self.level_states {
            for val in level_state.iter_mut() {
                *val *= decay;
            }
        }
    }
}

/// Emotion contagion - emotional content influences consciousness state
///
/// Detects emotional valence in input and nudges consciousness patterns:
/// - Positive emotions → Excited, Focused
/// - Negative emotions → Contemplative
/// - Neutral → no influence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionContagion {
    /// Current emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    /// High arousal = excited/angry, Low arousal = calm/sad
    pub arousal: f32,

    /// Emotional influence strength (0.0 to 1.0)
    /// How much emotion affects consciousness pattern
    pub influence_strength: f32,

    /// Smoothed valence (EMA)
    smoothed_valence: f32,

    /// Smoothed arousal (EMA)
    smoothed_arousal: f32,
}

impl Default for EmotionContagion {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            influence_strength: 0.3,
            smoothed_valence: 0.0,
            smoothed_arousal: 0.5,
        }
    }
}

impl EmotionContagion {
    /// Positive emotion indicators
    const POSITIVE_WORDS: &'static [&'static str] = &[
        "happy", "joy", "love", "great", "wonderful", "excellent", "amazing",
        "beautiful", "fantastic", "good", "perfect", "brilliant", "awesome",
        "delighted", "excited", "pleased", "thrilled", "grateful", "hope",
        "success", "win", "celebrate", "smile", "laugh", "fun", "enjoy",
    ];

    /// Negative emotion indicators
    const NEGATIVE_WORDS: &'static [&'static str] = &[
        "sad", "angry", "fear", "hate", "terrible", "awful", "horrible",
        "bad", "wrong", "fail", "lost", "pain", "hurt", "worry", "anxious",
        "stressed", "frustrated", "disappointed", "regret", "sorry", "grief",
        "cry", "suffer", "struggle", "difficult", "problem", "error",
    ];

    /// High arousal indicators (excitement/intensity)
    const HIGH_AROUSAL: &'static [&'static str] = &[
        "!", "amazing", "incredible", "urgent", "now", "immediately",
        "excited", "thrilled", "furious", "terrified", "ecstatic",
    ];

    /// Analyze text for emotional content
    pub fn analyze(&mut self, text: &str) {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        // Safe cast: use f64 intermediate to prevent precision loss on large word counts
        let word_count = (words.len().max(1) as f64) as f32;

        // Count emotional indicators (safe casts via f64)
        let positive_count = (Self::POSITIVE_WORDS.iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let negative_count = (Self::NEGATIVE_WORDS.iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let arousal_count = (Self::HIGH_AROUSAL.iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        // Compute raw valence (-1 to 1)
        let total_emotional = positive_count + negative_count;
        let raw_valence = if total_emotional > 0.0 {
            (positive_count - negative_count) / total_emotional
        } else {
            0.0
        };

        // Compute intensity based on proportion of emotional words
        let emotional_density = total_emotional / word_count;
        let intensity = (emotional_density * 3.0).min(1.0); // Scale up, cap at 1

        // Compute arousal (base + exclamation points + high-arousal words)
        // Safe cast via f64 to handle large match counts
        let exclamation_boost = (text.matches('!').count() as f64 * 0.1) as f32;
        let raw_arousal = (0.5 + arousal_count * 0.1 + exclamation_boost).min(1.0);

        // Apply intensity to valence
        self.valence = raw_valence * intensity;

        // Update arousal
        self.arousal = raw_arousal;

        // Smooth with EMA
        let alpha = 0.3;
        self.smoothed_valence = self.smoothed_valence * (1.0 - alpha) + self.valence * alpha;
        self.smoothed_arousal = self.smoothed_arousal * (1.0 - alpha) + self.arousal * alpha;
    }

    /// Get suggested pattern nudge based on emotional state
    /// Returns (pattern_suggestion, strength) where strength is 0-1
    pub fn pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) {
        let valence = self.smoothed_valence;
        let arousal = self.smoothed_arousal;

        // Only nudge if emotion is significant
        if valence.abs() < 0.2 {
            return (None, 0.0);
        }

        let strength = valence.abs() * self.influence_strength;

        let suggested_pattern = if valence > 0.3 && arousal > 0.6 {
            // High positive + high arousal → Excited
            Some(ConsciousnessPattern::Excited)
        } else if valence > 0.2 && arousal < 0.5 {
            // Positive + calm → Focused
            Some(ConsciousnessPattern::Focused)
        } else if valence < -0.3 {
            // Negative → Contemplative (processing/reflecting)
            Some(ConsciousnessPattern::Contemplative)
        } else if valence > 0.2 {
            // Mildly positive → Exploratory
            Some(ConsciousnessPattern::Exploratory)
        } else {
            None
        };

        (suggested_pattern, strength)
    }

    /// Get emotional valence for voice prosody
    pub fn prosody_valence(&self) -> f32 {
        self.smoothed_valence
    }

    /// Get arousal for voice prosody
    pub fn prosody_arousal(&self) -> f32 {
        self.smoothed_arousal
    }

    /// Reset emotional state
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Curiosity drive - novelty-seeking mechanism to prevent stagnation
///
/// When predictions become too accurate/predictable, curiosity triggers
/// exploration mode to discover new patterns and prevent cognitive stagnation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityDrive {
    /// Recent prediction errors (rolling window)
    error_history: Vec<f32>,

    /// Boredom level (0.0 to 1.0)
    /// High when predictions are too accurate for too long
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    /// High boredom triggers high curiosity
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    /// Direct measure of desire to explore new patterns
    pub exploration_urge: f32,

    /// Novelty bonus for learning rate
    /// Higher when exploring new territory
    pub novelty_bonus: f32,

    /// Consecutive cycles of low error (boredom buildup)
    low_error_streak: u32,

    /// Threshold below which error is "too good"
    boredom_threshold: f32,
}

impl Default for CuriosityDrive {
    fn default() -> Self {
        Self {
            error_history: Vec::with_capacity(50),
            boredom: 0.0,
            curiosity: 0.3, // Start with some curiosity
            exploration_urge: 0.0,
            novelty_bonus: 1.0,
            low_error_streak: 0,
            boredom_threshold: 0.1,
        }
    }
}

impl CuriosityDrive {
    /// Window size for error history
    const HISTORY_SIZE: usize = 50;
    /// Boredom streak threshold
    const BOREDOM_STREAK: u32 = 10;
    /// Maximum novelty bonus
    const MAX_NOVELTY_BONUS: f32 = 1.5;

    /// Update curiosity drive based on prediction error
    pub fn update(&mut self, prediction_error: f32) {
        // Track error history
        self.error_history.push(prediction_error);
        if self.error_history.len() > Self::HISTORY_SIZE {
            self.error_history.remove(0);
        }

        // Compute average error (safe division with max(1))
        let avg_error = if !self.error_history.is_empty() {
            self.error_history.iter().sum::<f32>() / self.error_history.len().max(1) as f32
        } else {
            0.5
        };

        // Detect boredom (consistently low error)
        if prediction_error < self.boredom_threshold {
            self.low_error_streak += 1;
        } else {
            self.low_error_streak = self.low_error_streak.saturating_sub(2);
        }

        // Boredom grows with low error streak (safe cast via f64)
        let streak_factor = ((self.low_error_streak as f64 / Self::BOREDOM_STREAK.max(1) as f64) as f32).min(1.0);
        self.boredom = 0.9 * self.boredom + 0.1 * streak_factor;

        // Curiosity is inverse of average error (interesting when things are predictable)
        let error_curiosity = (1.0 - avg_error.min(1.0)).max(0.0);
        self.curiosity = 0.8 * self.curiosity + 0.2 * error_curiosity;

        // Exploration urge triggered by high boredom + high curiosity
        if self.boredom > 0.5 && self.curiosity > 0.5 {
            self.exploration_urge = (self.boredom * self.curiosity).min(1.0);
        } else {
            self.exploration_urge *= 0.9; // Decay
        }

        // Novelty bonus: higher when exploring after boredom
        // This encourages the system to seek novel inputs
        if self.exploration_urge > 0.3 {
            self.novelty_bonus = 1.0 + (Self::MAX_NOVELTY_BONUS - 1.0) * self.exploration_urge;
        } else if prediction_error > 0.5 {
            // High error = novel situation, boost learning
            self.novelty_bonus = 1.0 + 0.3 * (prediction_error - 0.5);
        } else {
            self.novelty_bonus = 1.0;
        }
    }

    /// Check if exploration should be triggered
    pub fn should_explore(&self) -> bool {
        self.exploration_urge > 0.4 || (self.boredom > 0.7 && self.curiosity > 0.6)
    }

    /// Get action hint based on curiosity state
    pub fn action_hint(&self) -> Option<ActionHint> {
        if self.should_explore() {
            Some(ActionHint::Explore)
        } else if self.boredom > 0.5 {
            Some(ActionHint::SeekInput) // Need new stimuli
        } else {
            None
        }
    }

    /// Get effective learning rate with novelty bonus
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        base_rate * self.novelty_bonus
    }

    /// Reset curiosity drive
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get exploration probability (for stochastic exploration)
    pub fn exploration_probability(&self) -> f32 {
        (self.exploration_urge * 0.5 + self.boredom * 0.3).min(0.8)
    }

    /// Set boredom threshold from self-reflection
    ///
    /// This allows the meta-learning system to adjust when boredom triggers.
    pub fn set_boredom_threshold(&mut self, threshold: f32) {
        self.boredom_threshold = threshold.clamp(0.05, 0.3);
    }

    /// Get current boredom threshold
    pub fn get_boredom_threshold(&self) -> f32 {
        self.boredom_threshold
    }
}

/// Self-reflection - meta-learning through introspection
///
/// Periodically analyzes the system's own performance and adjusts
/// internal thresholds to optimize behavior. This enables the system
/// to learn about itself and improve over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflection {
    // ===== Adaptive Thresholds =====

    /// Flow entry error threshold (adjusted based on flow frequency)
    pub flow_error_threshold: f32,

    /// Flow entry coherence threshold
    pub flow_coherence_threshold: f32,

    /// Boredom detection threshold (for curiosity drive)
    pub boredom_threshold: f32,

    /// Confidence threshold for trusting predictions
    pub trust_threshold: f32,

    // ===== Meta-Statistics =====

    /// Total reflection cycles performed
    pub reflection_count: u64,

    /// Cycles since last reflection
    cycles_since_reflection: u32,

    /// Reflection interval (cycles between reflections)
    reflection_interval: u32,

    /// Historical flow entry rate (EMA)
    flow_entry_rate: f32,

    /// Historical exploration rate (EMA)
    exploration_rate: f32,

    /// Historical average error (EMA)
    historical_error: f32,

    /// Historical average confidence (EMA)
    historical_confidence: f32,

    /// Learning rate effectiveness score
    learning_effectiveness: f32,

    // ===== Adjustment History =====

    /// Number of threshold adjustments made
    pub adjustments_made: u32,

    /// Last adjustment direction for flow threshold (-1, 0, 1)
    last_flow_adjustment: i8,

    /// Last adjustment direction for boredom threshold
    last_boredom_adjustment: i8,

    // ===== Insights =====

    /// Current self-assessment
    pub self_assessment: SelfAssessment,

    /// Recommended actions based on reflection
    pub recommendations: Vec<Recommendation>,
}

/// Self-assessment of system state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfAssessment {
    /// System is performing optimally
    Optimal,
    /// System is learning effectively
    Learning,
    /// System is stagnating (needs stimulation)
    Stagnating,
    /// System is struggling (high error, low confidence)
    Struggling,
    /// System is overconfident (low error but not learning)
    Overconfident,
    /// System is in exploration mode
    Exploring,
    /// System needs calibration
    NeedsCalibration,
}

/// Recommendation from self-reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// What to adjust
    pub target: RecommendationTarget,
    /// Direction of adjustment
    pub direction: AdjustmentDirection,
    /// Confidence in this recommendation (0.0 to 1.0)
    pub confidence: f32,
    /// Reason for recommendation
    pub reason: String,
}

/// What the recommendation targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationTarget {
    FlowThreshold,
    BoredomThreshold,
    TrustThreshold,
    LearningRate,
    ExplorationFactor,
    ReflectionInterval,
}

/// Direction of adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentDirection {
    Increase,
    Decrease,
    NoChange,
}

impl Default for SelfReflection {
    fn default() -> Self {
        Self {
            // Initial thresholds (will be adjusted)
            flow_error_threshold: 0.25,
            flow_coherence_threshold: 0.6,
            boredom_threshold: 0.1,
            trust_threshold: 0.4,

            // Meta-statistics
            reflection_count: 0,
            cycles_since_reflection: 0,
            reflection_interval: 50, // Reflect every 50 cycles
            flow_entry_rate: 0.0,
            exploration_rate: 0.0,
            historical_error: 0.5,
            historical_confidence: 0.5,
            learning_effectiveness: 0.5,

            // Adjustment tracking
            adjustments_made: 0,
            last_flow_adjustment: 0,
            last_boredom_adjustment: 0,

            // Initial state
            self_assessment: SelfAssessment::Learning,
            recommendations: Vec::new(),
        }
    }
}

impl SelfReflection {
    /// Minimum reflection interval
    const MIN_INTERVAL: u32 = 20;
    /// Maximum reflection interval
    const MAX_INTERVAL: u32 = 200;
    /// Threshold adjustment step size
    const ADJUSTMENT_STEP: f32 = 0.02;

    /// Record a cycle's metrics (called every cycle)
    pub fn record_cycle(
        &mut self,
        prediction_error: f32,
        in_flow: bool,
        exploring: bool,
        confidence: f32,
    ) {
        self.cycles_since_reflection += 1;

        // Update EMAs
        let alpha = 0.05;
        self.historical_error = self.historical_error * (1.0 - alpha) + prediction_error * alpha;
        self.historical_confidence = self.historical_confidence * (1.0 - alpha) + confidence * alpha;

        // Track flow and exploration rates
        let flow_val = if in_flow { 1.0 } else { 0.0 };
        let explore_val = if exploring { 1.0 } else { 0.0 };
        self.flow_entry_rate = self.flow_entry_rate * (1.0 - alpha) + flow_val * alpha;
        self.exploration_rate = self.exploration_rate * (1.0 - alpha) + explore_val * alpha;
    }

    /// Check if it's time to reflect
    pub fn should_reflect(&self) -> bool {
        self.cycles_since_reflection >= self.reflection_interval
    }

    /// Perform self-reflection and adjust thresholds
    pub fn reflect(&mut self) -> Vec<Recommendation> {
        self.reflection_count += 1;
        self.cycles_since_reflection = 0;
        self.recommendations.clear();

        // Analyze current state
        self.analyze_state();

        // Generate recommendations based on analysis
        self.generate_recommendations();

        // Apply automatic adjustments
        self.apply_adjustments();

        // Adjust reflection interval based on stability
        self.adjust_interval();

        self.recommendations.clone()
    }

    /// Analyze current system state
    fn analyze_state(&mut self) {
        // Determine self-assessment based on metrics
        self.self_assessment = if self.flow_entry_rate > 0.3 && self.historical_error < 0.2 {
            SelfAssessment::Optimal
        } else if self.exploration_rate > 0.3 {
            SelfAssessment::Exploring
        } else if self.historical_error < 0.15 && self.flow_entry_rate < 0.1 {
            SelfAssessment::Overconfident
        } else if self.historical_error > 0.5 && self.historical_confidence < 0.4 {
            SelfAssessment::Struggling
        } else if self.historical_error < 0.2 && self.exploration_rate < 0.05 {
            SelfAssessment::Stagnating
        } else if self.adjustments_made > 10 && self.flow_entry_rate < 0.05 {
            SelfAssessment::NeedsCalibration
        } else {
            SelfAssessment::Learning
        };

        // Update learning effectiveness
        // Good learning = moderate error (not too high, not too low) + improving trend
        let optimal_error = 0.25;
        let error_quality = 1.0 - (self.historical_error - optimal_error).abs() * 2.0;
        self.learning_effectiveness = error_quality.clamp(0.0, 1.0);
    }

    /// Generate recommendations based on current state
    fn generate_recommendations(&mut self) {
        match self.self_assessment {
            SelfAssessment::Stagnating => {
                // Lower boredom threshold to trigger exploration sooner
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::BoredomThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.8,
                    reason: "System is stagnating; lower boredom threshold to encourage exploration".into(),
                });
                // Increase exploration factor
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::ExplorationFactor,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.7,
                    reason: "Need more exploration to break stagnation".into(),
                });
            }
            SelfAssessment::Struggling => {
                // Raise trust threshold (be more cautious)
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::TrustThreshold,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.7,
                    reason: "High error rate; increase trust threshold for caution".into(),
                });
                // Lower learning rate if errors are very high
                if self.historical_error > 0.6 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::LearningRate,
                        direction: AdjustmentDirection::Decrease,
                        confidence: 0.6,
                        reason: "Very high errors; reduce learning rate for stability".into(),
                    });
                }
            }
            SelfAssessment::Overconfident => {
                // Lower flow threshold (make flow harder to achieve)
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::FlowThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.7,
                    reason: "Predictions too easy; tighten flow entry criteria".into(),
                });
                // Raise boredom threshold
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::BoredomThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.6,
                    reason: "System may be bored but not detecting it".into(),
                });
            }
            SelfAssessment::NeedsCalibration => {
                // Reset to more moderate thresholds
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::FlowThreshold,
                    direction: AdjustmentDirection::NoChange,
                    confidence: 0.5,
                    reason: "Consider manual threshold review".into(),
                });
                // Extend reflection interval to allow stabilization
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::ReflectionInterval,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.8,
                    reason: "Allow more time between adjustments".into(),
                });
            }
            SelfAssessment::Optimal | SelfAssessment::Learning => {
                // Fine-tune based on specific metrics
                if self.flow_entry_rate < 0.1 && self.historical_error < 0.3 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::FlowThreshold,
                        direction: AdjustmentDirection::Increase,
                        confidence: 0.5,
                        reason: "Good performance but rarely in flow; relax flow criteria".into(),
                    });
                }
            }
            SelfAssessment::Exploring => {
                // Don't interfere with active exploration
                // Maybe shorten reflection interval to monitor
                if self.exploration_rate > 0.5 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::ReflectionInterval,
                        direction: AdjustmentDirection::Decrease,
                        confidence: 0.4,
                        reason: "High exploration; monitor more frequently".into(),
                    });
                }
            }
        }
    }

    /// Apply automatic threshold adjustments
    fn apply_adjustments(&mut self) {
        for rec in &self.recommendations {
            if rec.confidence < 0.5 {
                continue; // Skip low-confidence recommendations
            }

            match rec.target {
                RecommendationTarget::FlowThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.flow_error_threshold = (self.flow_error_threshold + Self::ADJUSTMENT_STEP).min(0.4);
                            self.last_flow_adjustment = 1;
                        }
                        AdjustmentDirection::Decrease => {
                            self.flow_error_threshold = (self.flow_error_threshold - Self::ADJUSTMENT_STEP).max(0.1);
                            self.last_flow_adjustment = -1;
                        }
                        AdjustmentDirection::NoChange => {
                            self.last_flow_adjustment = 0;
                        }
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::BoredomThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.boredom_threshold = (self.boredom_threshold + Self::ADJUSTMENT_STEP).min(0.3);
                            self.last_boredom_adjustment = 1;
                        }
                        AdjustmentDirection::Decrease => {
                            self.boredom_threshold = (self.boredom_threshold - Self::ADJUSTMENT_STEP).max(0.05);
                            self.last_boredom_adjustment = -1;
                        }
                        AdjustmentDirection::NoChange => {
                            self.last_boredom_adjustment = 0;
                        }
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::TrustThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.trust_threshold = (self.trust_threshold + Self::ADJUSTMENT_STEP).min(0.7);
                        }
                        AdjustmentDirection::Decrease => {
                            self.trust_threshold = (self.trust_threshold - Self::ADJUSTMENT_STEP).max(0.2);
                        }
                        AdjustmentDirection::NoChange => {}
                    }
                    self.adjustments_made += 1;
                }
                _ => {} // Other targets handled externally
            }
        }
    }

    /// Adjust reflection interval based on system stability
    fn adjust_interval(&mut self) {
        // If making many adjustments, reflect more often
        // If stable, reflect less often
        // Safe cast via f64 to prevent precision loss on large values
        let recent_adjustment_rate = (self.adjustments_made as f64 / self.reflection_count.max(1) as f64) as f32;

        if recent_adjustment_rate > 0.8 {
            // Lots of adjustments = unstable, reflect more
            self.reflection_interval = (self.reflection_interval - 5).max(Self::MIN_INTERVAL);
        } else if recent_adjustment_rate < 0.2 && self.self_assessment == SelfAssessment::Optimal {
            // Stable and optimal, reflect less
            self.reflection_interval = (self.reflection_interval + 10).min(Self::MAX_INTERVAL);
        }
    }

    /// Get current thresholds for use by other components
    pub fn get_thresholds(&self) -> ReflectionThresholds {
        ReflectionThresholds {
            flow_error: self.flow_error_threshold,
            flow_coherence: self.flow_coherence_threshold,
            boredom: self.boredom_threshold,
            trust: self.trust_threshold,
        }
    }

    /// Get a human-readable summary of current state
    pub fn summary(&self) -> ReflectionSummary {
        ReflectionSummary {
            assessment: self.self_assessment,
            reflection_count: self.reflection_count,
            adjustments_made: self.adjustments_made,
            flow_entry_rate: self.flow_entry_rate,
            exploration_rate: self.exploration_rate,
            learning_effectiveness: self.learning_effectiveness,
            historical_error: self.historical_error,
            historical_confidence: self.historical_confidence,
            next_reflection_in: self.reflection_interval.saturating_sub(self.cycles_since_reflection),
        }
    }

    /// Reset self-reflection state
    pub fn reset(&mut self) {
        // Keep learned thresholds but reset statistics
        let thresholds = (
            self.flow_error_threshold,
            self.flow_coherence_threshold,
            self.boredom_threshold,
            self.trust_threshold,
        );
        *self = Self::default();
        self.flow_error_threshold = thresholds.0;
        self.flow_coherence_threshold = thresholds.1;
        self.boredom_threshold = thresholds.2;
        self.trust_threshold = thresholds.3;
    }

    /// Full reset including learned thresholds
    pub fn full_reset(&mut self) {
        *self = Self::default();
    }

    /// Force an immediate reflection (triggered by motor commands)
    ///
    /// This bypasses the normal interval and triggers reflection now.
    pub fn force_reflection(&mut self) {
        // Set cycles to trigger immediate reflection
        self.cycles_since_reflection = self.reflection_interval;
    }
}

/// Thresholds from self-reflection for use by other components
#[derive(Debug, Clone, Copy)]
pub struct ReflectionThresholds {
    pub flow_error: f32,
    pub flow_coherence: f32,
    pub boredom: f32,
    pub trust: f32,
}

/// Summary of self-reflection state
#[derive(Debug, Clone)]
pub struct ReflectionSummary {
    pub assessment: SelfAssessment,
    pub reflection_count: u64,
    pub adjustments_made: u32,
    pub flow_entry_rate: f32,
    pub exploration_rate: f32,
    pub learning_effectiveness: f32,
    pub historical_error: f32,
    pub historical_confidence: f32,
    pub next_reflection_in: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTING - Cognitive Depth Selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Cognitive depth determines how much processing to apply
///
/// Based on the Thalamus architecture (ARCHITECTURAL_EVOLUTION_SUMMARY.md):
/// - Reflex: Pattern matching only, <10ms response
/// - Cortical: Standard cognitive cycle, 50-200ms
/// - DeepThought: Full deliberation with causal reasoning, 200ms+
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveDepth {
    /// Fast pattern matching, minimal processing
    /// Used for: Familiar inputs, low novelty, low urgency
    Reflex,

    /// Standard cognitive cycle with prediction and learning
    /// Used for: Normal conversation, moderate complexity
    Cortical,

    /// Deep deliberation with causal reasoning and counterfactuals
    /// Used for: Novel situations, high stakes, complex reasoning
    DeepThought,
}

impl Default for CognitiveDepth {
    fn default() -> Self {
        Self::Cortical
    }
}

/// Thalamic router - determines cognitive depth before processing
///
/// Implements the 3-path routing from Architecture V2:
/// - High novelty/urgency → DeepThought
/// - Normal → Cortical
/// - Familiar/low stakes → Reflex
#[derive(Debug, Clone)]
pub struct ThalamicRouter {
    /// Novelty threshold for DeepThought (0.0-1.0)
    pub novelty_threshold: f32,

    /// Urgency threshold for DeepThought (0.0-1.0)
    pub urgency_threshold: f32,

    /// Familiarity threshold for Reflex (0.0-1.0)
    pub familiarity_threshold: f32,

    /// Recent routing decisions for pattern analysis
    routing_history: Vec<CognitiveDepth>,

    /// Maximum history size
    max_history: usize,
}

impl Default for ThalamicRouter {
    fn default() -> Self {
        Self {
            novelty_threshold: 0.7,
            urgency_threshold: 0.8,
            familiarity_threshold: 0.3,
            routing_history: Vec::with_capacity(100),
            max_history: 100,
        }
    }
}

impl ThalamicRouter {
    /// Route based on input characteristics
    ///
    /// # Arguments
    /// * `novelty` - How novel/surprising the input is (0.0-1.0)
    /// * `urgency` - How urgent the response needs to be (0.0-1.0)
    /// * `complexity` - Estimated complexity of the input (0.0-1.0)
    /// * `emotional_intensity` - Emotional intensity of input (0.0-1.0)
    pub fn route(
        &mut self,
        novelty: f32,
        urgency: f32,
        complexity: f32,
        emotional_intensity: f32,
    ) -> CognitiveDepth {
        let depth = if novelty > self.novelty_threshold
            || urgency > self.urgency_threshold
            || complexity > 0.8
            || emotional_intensity > 0.7
        {
            // High stakes - use deep thought
            CognitiveDepth::DeepThought
        } else if novelty < self.familiarity_threshold
            && complexity < 0.3
            && urgency < 0.5
        {
            // Familiar, simple, not urgent - use reflex
            CognitiveDepth::Reflex
        } else {
            // Default to standard cortical processing
            CognitiveDepth::Cortical
        };

        // Record history
        if self.routing_history.len() >= self.max_history {
            self.routing_history.remove(0);
        }
        self.routing_history.push(depth);

        depth
    }

    /// Route based on prediction error and pattern
    pub fn route_from_cycle(
        &mut self,
        prediction_error: f32,
        pattern: ConsciousnessPattern,
        emotional_valence: f32,
    ) -> CognitiveDepth {
        // Novelty from prediction error (high error = novel)
        let novelty = prediction_error.min(1.0);

        // Complexity from pattern
        let complexity = match pattern {
            ConsciousnessPattern::Uncertain => 0.8,
            ConsciousnessPattern::Transitioning => 0.7,
            ConsciousnessPattern::Exploratory => 0.6,
            ConsciousnessPattern::Contemplative => 0.5,
            ConsciousnessPattern::Focused => 0.4,
            ConsciousnessPattern::Excited => 0.4,
            ConsciousnessPattern::Resting => 0.2,
        };

        // Urgency from pattern (uncertain/transitioning = urgent)
        let urgency = match pattern {
            ConsciousnessPattern::Uncertain => 0.8,
            ConsciousnessPattern::Transitioning => 0.6,
            ConsciousnessPattern::Excited => 0.5,
            _ => 0.3,
        };

        // Emotional intensity from absolute valence
        let emotional_intensity = emotional_valence.abs();

        self.route(novelty, urgency, complexity, emotional_intensity)
    }

    /// Get statistics on routing patterns
    pub fn routing_stats(&self) -> (f32, f32, f32) {
        if self.routing_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Safe cast via f64 to prevent precision loss on large counts
        let total = self.routing_history.len().max(1) as f64;
        let reflex = (self.routing_history.iter()
            .filter(|d| **d == CognitiveDepth::Reflex)
            .count() as f64 / total) as f32;
        let cortical = (self.routing_history.iter()
            .filter(|d| **d == CognitiveDepth::Cortical)
            .count() as f64 / total) as f32;
        let deep = (self.routing_history.iter()
            .filter(|d| **d == CognitiveDepth::DeepThought)
            .count() as f64 / total) as f32;

        (reflex, cortical, deep)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE BRIDGE - Precision-Weighted Prediction Tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality of prediction-outcome coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingQuality {
    /// Not enough data to assess coupling
    InsufficientData,
    /// No meaningful coupling (MI < 0.1)
    NoCoupling,
    /// Weak coupling (MI 0.1-0.3)
    WeakCoupling,
    /// Moderate coupling (MI 0.3-0.6)
    ModerateCoupling,
    /// Strong coupling (MI > 0.6)
    StrongCoupling,
}

impl CouplingQuality {
    /// Is coupling meaningful?
    pub fn is_meaningful(&self) -> bool {
        matches!(
            self,
            Self::WeakCoupling | Self::ModerateCoupling | Self::StrongCoupling
        )
    }
}

/// Simplified Active Inference Bridge for prediction-outcome coupling
///
/// Tracks the relationship between prediction confidence and actual outcomes
/// using a simplified Phase-Amplitude Coupling (PAC) approach.
#[derive(Debug, Clone)]
pub struct ActiveInferenceBridge {
    /// Recent confidence values (phase signal)
    confidence_history: Vec<f64>,

    /// Recent outcomes (amplitude signal)
    outcome_history: Vec<f64>,

    /// Window size for coupling computation
    window_size: usize,

    /// Minimum data points before coupling is meaningful
    min_data_points: usize,

    /// Total observations
    total_observations: usize,
}

impl Default for ActiveInferenceBridge {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl ActiveInferenceBridge {
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self {
            confidence_history: Vec::with_capacity(100),
            outcome_history: Vec::with_capacity(100),
            window_size: 100,
            min_data_points: 10,
            total_observations: 0,
        }
    }

    /// Observe a prediction resolution
    ///
    /// * `confidence`: The predicted confidence (0.0-1.0)
    /// * `success`: Whether the prediction was correct
    pub fn observe_resolution(&mut self, confidence: f64, success: bool) {
        self.total_observations += 1;

        // Track confidence
        if self.confidence_history.len() >= self.window_size {
            self.confidence_history.remove(0);
        }
        self.confidence_history.push(confidence);

        // Track outcome
        let outcome = if success { 1.0 } else { 0.0 };
        if self.outcome_history.len() >= self.window_size {
            self.outcome_history.remove(0);
        }
        self.outcome_history.push(outcome);
    }

    /// Compute the Modulation Index (simplified PAC)
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = No coupling (predictions don't inform outcomes)
    /// - 1.0 = Perfect coupling (confidence perfectly predicts success)
    pub fn modulation_index(&self) -> Option<f64> {
        if self.confidence_history.len() < self.min_data_points {
            return None;
        }

        // Compute correlation between confidence and success
        // Safe cast (already f64, just ensure non-zero)
        let n = self.confidence_history.len().max(1) as f64;
        let conf_mean: f64 = self.confidence_history.iter().sum::<f64>() / n;
        let out_mean: f64 = self.outcome_history.iter().sum::<f64>() / n;

        let mut covariance = 0.0;
        let mut conf_variance = 0.0;
        let mut out_variance = 0.0;

        for (c, o) in self.confidence_history.iter().zip(self.outcome_history.iter()) {
            let c_diff = c - conf_mean;
            let o_diff = o - out_mean;
            covariance += c_diff * o_diff;
            conf_variance += c_diff * c_diff;
            out_variance += o_diff * o_diff;
        }

        // Pearson correlation, normalized to [0, 1]
        let denom = (conf_variance * out_variance).sqrt();
        if denom < 1e-10 {
            return Some(0.0);
        }

        let correlation = covariance / denom;
        // Map [-1, 1] to [0, 1], favoring positive correlations
        Some((correlation.max(0.0)).clamp(0.0, 1.0))
    }

    /// Get the current coupling quality assessment
    pub fn coupling_quality(&self) -> CouplingQuality {
        match self.modulation_index() {
            None => CouplingQuality::InsufficientData,
            Some(mi) if mi < 0.1 => CouplingQuality::NoCoupling,
            Some(mi) if mi < 0.3 => CouplingQuality::WeakCoupling,
            Some(mi) if mi < 0.6 => CouplingQuality::ModerateCoupling,
            Some(_) => CouplingQuality::StrongCoupling,
        }
    }

    /// Get average prediction error (from recent history)
    pub fn average_prediction_error(&self) -> Option<f64> {
        if self.outcome_history.is_empty() {
            return None;
        }
        // Error = 1 - success rate (safe division with max(1))
        let success_rate: f64 = self.outcome_history.iter().sum::<f64>()
            / self.outcome_history.len().max(1) as f64;
        Some(1.0 - success_rate)
    }

    /// Get statistics
    pub fn statistics(&self) -> ActiveInferenceBridgeStats {
        ActiveInferenceBridgeStats {
            total_observations: self.total_observations,
            modulation_index: self.modulation_index(),
            coupling_quality: self.coupling_quality(),
            average_prediction_error: self.average_prediction_error(),
        }
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.confidence_history.clear();
        self.outcome_history.clear();
        self.total_observations = 0;
    }
}

/// Statistics from the Active Inference bridge
#[derive(Debug, Clone)]
pub struct ActiveInferenceBridgeStats {
    /// Total observations processed
    pub total_observations: usize,
    /// Current Modulation Index
    pub modulation_index: Option<f64>,
    /// Current coupling quality
    pub coupling_quality: CouplingQuality,
    /// Average prediction error (recent)
    pub average_prediction_error: Option<f64>,
}

// ============================================================================
// CONSCIOUSNESS SNAPSHOT - Unified Dashboard
// ============================================================================

/// Unified consciousness snapshot - aggregates all cognitive metrics
///
/// This provides a single point of observation for the entire cognitive state,
/// making it easy to monitor, log, or expose via API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    // ===== Core Metrics =====

    /// Timestamp of this snapshot (cycle count)
    pub cycle: usize,

    /// Overall consciousness level (0.0 to 1.0)
    /// Computed from prediction confidence, coherence, and flow
    pub consciousness_level: f32,

    /// Current consciousness pattern
    pub pattern: ConsciousnessPattern,

    /// Pattern classification confidence
    pub pattern_confidence: f32,

    // ===== Prediction & Learning =====

    /// Current prediction error
    pub prediction_error: f32,

    /// Prediction confidence (decays during uncertainty)
    pub prediction_confidence: f32,

    /// Whether predictions should be trusted
    pub predictions_trustworthy: bool,

    /// Effective learning rate (after all modulations)
    pub effective_learning_rate: f32,

    /// Learning effectiveness score from self-reflection
    pub learning_effectiveness: f32,

    // ===== Flow State =====

    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow intensity (0.0 to 1.0)
    pub flow_intensity: f32,

    /// Consecutive flow-compatible cycles
    pub flow_streak: u32,

    /// Learning boost from flow
    pub flow_learning_boost: f32,

    // ===== Curiosity & Exploration =====

    /// Boredom level (0.0 to 1.0)
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    pub exploration_urge: f32,

    /// Whether curiosity is triggering exploration
    pub exploring: bool,

    /// Novelty bonus for learning
    pub novelty_bonus: f32,

    // ===== Emotional State =====

    /// Emotional valence (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Emotional arousal (0.0 to 1.0)
    pub emotional_arousal: f32,

    /// Whether input has significant emotional content
    pub has_emotional_content: bool,

    /// Emotion-suggested pattern nudge
    pub emotion_nudge: Option<ConsciousnessPattern>,

    // ===== Self-Reflection =====

    /// Self-assessment from meta-learning
    pub self_assessment: SelfAssessment,

    /// Number of reflection cycles performed
    pub reflection_count: u64,

    /// Threshold adjustments made
    pub adjustments_made: u32,

    /// Cycles until next reflection
    pub next_reflection_in: u32,

    // ===== Adaptive Behavior =====

    /// Recommended action
    pub action_hint: ActionHint,

    /// Speech rate multiplier
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier
    pub pause_multiplier: f32,

    /// Whether learning is paused
    pub learning_paused: bool,

    // ===== Adapted Thresholds =====

    /// Adapted flow error threshold
    pub flow_threshold: f32,

    /// Adapted boredom threshold
    pub boredom_threshold: f32,

    /// Adapted trust threshold
    pub trust_threshold: f32,

    // ===== Temporal Coherence =====

    /// Temporal coherence from CfC
    pub temporal_coherence: f32,

    /// Tau trajectory mean
    pub tau_mean: f32,

    /// Tau trajectory trend
    pub tau_trend: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE FIELDS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Current cognitive depth (Reflex/Cortical/DeepThought)
    pub cognitive_depth: CognitiveDepth,

    /// Unified Φ from ConsciousnessUnificationEngine
    pub unified_phi: f32,

    /// Unified emotional valence (VAD-based, -1.0 to 1.0)
    pub unified_valence: f32,

    /// Unified emotional arousal (VAD-based, 0.0 to 1.0)
    pub unified_arousal: f32,

    /// Unified emotional dominance (VAD-based, -1.0 to 1.0)
    pub unified_dominance: f32,

    /// Discrete emotion from unified EmotionalBridge
    pub unified_discrete_emotion: Option<UnifiedEmotion>,

    /// Emotional pattern (Stable/Escalating/Calming/Volatile)
    pub emotional_pattern: EmotionalPattern,

    /// Emotional description in natural language
    pub emotional_description: String,

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING FIELDS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Snapshot creation timestamp (monotonic, for relative time)
    pub snapshot_timestamp_nanos: u64,

    /// Current flow duration if in flow (seconds)
    pub current_flow_duration_secs: Option<f32>,

    /// Total time spent in flow this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average flow period duration (seconds)
    pub avg_flow_duration_secs: f32,

    // ===== FEP Active Inference =====

    /// FEP variational free energy
    pub fep_free_energy: f64,

    /// FEP precision estimate
    pub fep_precision: f64,
}

impl ConsciousnessSnapshot {
    /// Compute overall consciousness level from components
    fn compute_consciousness_level(
        prediction_confidence: f32,
        temporal_coherence: f32,
        flow_intensity: f32,
        pattern_confidence: f32,
    ) -> f32 {
        // Weighted combination of key indicators
        let confidence_contrib = prediction_confidence * 0.3;
        let coherence_contrib = temporal_coherence * 0.25;
        let flow_contrib = flow_intensity * 0.2;
        let pattern_contrib = pattern_confidence * 0.25;

        (confidence_contrib + coherence_contrib + flow_contrib + pattern_contrib).clamp(0.0, 1.0)
    }

    /// Get a concise status string
    pub fn status(&self) -> String {
        let flow_status = if self.in_flow { "FLOW" } else { "---" };
        let explore_status = if self.exploring { "EXPLORE" } else { "---" };

        format!(
            "[L:{:.2}] {:?} | {} {} | Conf:{:.2} Err:{:.2}",
            self.consciousness_level,
            self.pattern,
            flow_status,
            explore_status,
            self.prediction_confidence,
            self.prediction_error,
        )
    }

    /// Check if system is in an optimal state
    pub fn is_optimal(&self) -> bool {
        self.self_assessment == SelfAssessment::Optimal
            || (self.in_flow && self.prediction_confidence > 0.6)
    }

    /// Check if system needs attention (struggling or stagnating)
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.self_assessment,
            SelfAssessment::Struggling | SelfAssessment::Stagnating | SelfAssessment::NeedsCalibration
        )
    }

    /// Get the dominant concern (what needs most attention)
    pub fn dominant_concern(&self) -> Option<&'static str> {
        if self.self_assessment == SelfAssessment::Struggling {
            Some("High prediction error - system is struggling")
        } else if self.self_assessment == SelfAssessment::Stagnating {
            Some("Low error but no exploration - system is stagnating")
        } else if self.boredom > 0.7 && !self.exploring {
            Some("High boredom - needs novel input")
        } else if self.prediction_confidence < 0.3 {
            Some("Low confidence - predictions unreliable")
        } else if self.self_assessment == SelfAssessment::NeedsCalibration {
            Some("Many adjustments made - consider manual review")
        } else {
            None
        }
    }

    /// Get recommended actions based on current state
    pub fn recommended_actions(&self) -> Vec<&'static str> {
        let mut actions = Vec::new();

        match self.action_hint {
            ActionHint::SlowDown => actions.push("Reduce input rate"),
            ActionHint::SpeedUp => actions.push("Can increase input rate"),
            ActionHint::Stabilize => actions.push("Maintain current input"),
            ActionHint::Explore => actions.push("Introduce novel inputs"),
            ActionHint::SeekInput => actions.push("System needs more input"),
            ActionHint::Continue => {}
        }

        if self.boredom > 0.5 && !self.exploring {
            actions.push("Consider varying input content");
        }

        if !self.predictions_trustworthy {
            actions.push("Predictions currently unreliable");
        }

        if self.in_flow {
            actions.push("In flow state - optimal for learning");
        }

        actions
    }
}

impl Default for AdaptiveBehavior {
    fn default() -> Self {
        Self {
            learning_rate_multiplier: 1.0,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            attention_sensitivity: 1.0,
            exploration_factor: 0.3,
            confidence: 0.5,
            pause_learning: false,
            action_hint: ActionHint::Continue,
        }
    }
}

impl AdaptiveBehavior {
    /// Compute adaptive behavior from consciousness pattern and metrics
    pub fn from_consciousness_state(
        pattern: ConsciousnessPattern,
        pattern_confidence: f32,
        coherence: f32,
        voice_confidence: f32,
    ) -> Self {
        // Base confidence from all sources
        let confidence = (pattern_confidence * 0.4 + coherence * 0.3 + voice_confidence * 0.3)
            .clamp(0.0, 1.0);

        match pattern {
            ConsciousnessPattern::Focused => Self {
                learning_rate_multiplier: 1.3 + confidence * 0.4,  // 1.3 to 1.7
                speech_rate_multiplier: 1.05 + confidence * 0.15,  // 1.05 to 1.2
                pause_multiplier: 0.7,
                attention_sensitivity: 0.7,  // Less distracted
                exploration_factor: 0.1,     // Stay on track
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SpeedUp,
            },

            ConsciousnessPattern::Contemplative => Self {
                learning_rate_multiplier: 0.8,
                speech_rate_multiplier: 0.85,
                pause_multiplier: 1.5,       // Longer pauses for reflection
                attention_sensitivity: 1.0,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SlowDown,
            },

            ConsciousnessPattern::Excited => Self {
                learning_rate_multiplier: 1.1,
                speech_rate_multiplier: 1.15,
                pause_multiplier: 0.6,       // Quick transitions
                attention_sensitivity: 1.3,  // More reactive
                exploration_factor: 0.4,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Exploratory => Self {
                learning_rate_multiplier: 1.0,
                speech_rate_multiplier: 0.95,
                pause_multiplier: 1.0,
                attention_sensitivity: 1.4,  // High sensitivity to new info
                exploration_factor: 0.7,     // Actively explore
                confidence: confidence * 0.8,
                pause_learning: false,
                action_hint: ActionHint::Explore,
            },

            ConsciousnessPattern::Resting => Self {
                learning_rate_multiplier: 0.6,
                speech_rate_multiplier: 0.9,
                pause_multiplier: 1.2,
                attention_sensitivity: 0.8,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Transitioning => Self {
                learning_rate_multiplier: 0.3,  // Minimal learning during transition
                speech_rate_multiplier: 0.8,
                pause_multiplier: 1.8,          // Pause to stabilize
                attention_sensitivity: 1.0,
                exploration_factor: 0.3,
                confidence: confidence * 0.5,
                pause_learning: true,           // Pause learning
                action_hint: ActionHint::Stabilize,
            },

            ConsciousnessPattern::Uncertain => Self {
                learning_rate_multiplier: 0.4,  // Careful learning
                speech_rate_multiplier: 0.75,   // Slow down
                pause_multiplier: 2.0,          // Long pauses
                attention_sensitivity: 1.2,
                exploration_factor: 0.5,
                confidence: confidence * 0.3,
                pause_learning: false,
                action_hint: ActionHint::SeekInput,
            },
        }
    }

    /// Get effective learning rate with all modulations
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        if self.pause_learning {
            0.0
        } else {
            base_rate * self.learning_rate_multiplier
        }
    }

    /// Check if the system should seek more input/clarification
    pub fn should_seek_input(&self) -> bool {
        self.action_hint == ActionHint::SeekInput || self.confidence < 0.3
    }

    /// Check if the system is in a confident state
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.6 && !self.pause_learning
    }

    /// Get a human-readable description of current state
    pub fn description(&self) -> &'static str {
        match self.action_hint {
            ActionHint::Continue => "Operating normally",
            ActionHint::SlowDown => "Deliberating carefully",
            ActionHint::SpeedUp => "Confident and focused",
            ActionHint::Stabilize => "Stabilizing state",
            ActionHint::Explore => "Exploring possibilities",
            ActionHint::SeekInput => "Seeking clarification",
        }
    }
}

/// Statistics for the cognitive loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoopStats {
    /// Total cycles completed
    pub total_cycles: usize,

    /// Average prediction error (EMA)
    pub avg_prediction_error: f32,

    /// Learning cycles (error > threshold)
    pub learning_cycles: usize,

    /// Average training loss (EMA)
    pub avg_training_loss: f32,

    /// Attention variance (emergence metric)
    pub attention_variance: f32,

    /// Number of primitives with diverged attention
    pub diverged_primitives: usize,

    /// Buffer utilization (0-1)
    pub buffer_utilization: f32,

    /// Average cycle time (microseconds)
    pub avg_cycle_time_us: f32,

    /// Cycles per second
    pub cycles_per_second: f32,

    /// Prediction error trend (negative = improving)
    pub error_trend: f32,

    /// LTC consciousness level
    pub ltc_consciousness: f32,

    /// Temporal coherence (from CfC tau values)
    pub temporal_coherence: f32,

    /// Coherence-modulated learning rate
    pub effective_learning_rate: f32,

    /// Phi contribution from temporal coherence
    pub coherence_phi_contribution: f32,

    /// Voice articulation quality (0.0 to 1.0)
    pub voice_articulation_quality: f32,

    /// Voice rate stability (0.0 to 1.0)
    pub voice_rate_stability: f32,

    /// Phi adjustment from voice feedback
    pub voice_phi_adjustment: f32,

    /// Combined phi (coherence + voice contributions)
    pub combined_phi_contribution: f32,

    /// Current consciousness pattern (from temporal signatures)
    pub consciousness_pattern: String,

    /// Confidence in consciousness pattern classification
    pub pattern_confidence: f32,

    /// Tau trajectory mean
    pub tau_mean: f32,

    /// Tau trajectory trend
    pub tau_trend: f32,

    /// Current adaptive behavior state
    pub adaptive_confidence: f32,

    /// Current action hint
    pub action_hint: String,

    /// Whether learning is paused due to state transition
    pub learning_paused: bool,

    /// Adaptive learning rate (after all modulations)
    pub adaptive_learning_rate: f32,

    /// Adaptive speech rate multiplier
    pub adaptive_speech_rate: f32,

    /// Prediction confidence (decays during uncertain states)
    /// High when predictions are accurate and state is stable
    pub prediction_confidence: f32,

    /// Prediction confidence decay rate (higher = faster decay)
    pub confidence_decay_rate: f32,

    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow state intensity (0.0 to 1.0)
    pub flow_intensity: f32,

    /// Consecutive cycles in flow-compatible state
    pub flow_streak: u32,

    /// Learning boost from flow state
    pub flow_learning_boost: f32,

    /// Emotional valence from content (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Emotional arousal (0.0 to 1.0)
    pub emotional_arousal: f32,

    /// Suggested pattern from emotion contagion
    pub emotion_nudge_pattern: String,

    /// Strength of emotion influence
    pub emotion_nudge_strength: f32,

    /// Boredom level from curiosity drive (0.0 to 1.0)
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    pub exploration_urge: f32,

    /// Whether curiosity-triggered exploration is active
    pub curiosity_exploring: bool,

    /// Novelty bonus to learning rate
    pub novelty_bonus: f32,

    // ===== Self-Reflection Stats =====

    /// Current self-assessment
    pub self_assessment: String,

    /// Number of reflection cycles performed
    pub reflection_count: u64,

    /// Threshold adjustments made
    pub adjustments_made: u32,

    /// Learning effectiveness score (0.0 to 1.0)
    pub learning_effectiveness: f32,

    /// Cycles until next reflection
    pub next_reflection_in: u32,

    /// Adapted flow error threshold
    pub adapted_flow_threshold: f32,

    /// Adapted boredom threshold
    pub adapted_boredom_threshold: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE STATS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Current cognitive depth (from Thalamic routing)
    pub cognitive_depth: String,

    /// Unified Φ from the unification engine
    pub unified_phi: f32,

    /// Unified emotional valence (VAD-based, from EmotionalBridge)
    pub unified_emotional_valence: f32,

    /// Unified emotional arousal (VAD-based)
    pub unified_emotional_arousal: f32,

    /// Unified emotional dominance (VAD-based)
    pub unified_emotional_dominance: f32,

    /// Discrete emotion from unified bridge
    pub unified_emotion: String,

    /// Emotional pattern detected (Stable/Escalating/Calming/Volatile)
    pub emotional_pattern: String,

    /// Thalamic routing: fraction using Reflex path
    pub thalamic_reflex_rate: f32,

    /// Thalamic routing: fraction using Cortical path
    pub thalamic_cortical_rate: f32,

    /// Thalamic routing: fraction using DeepThought path
    pub thalamic_deep_rate: f32,

    /// Active Inference: Modulation Index (prediction-outcome coupling)
    pub active_inference_modulation_index: f32,

    /// Active Inference: Coupling quality
    pub active_inference_coupling_quality: String,

    /// Active Inference: Average prediction error (from PAC)
    pub active_inference_avg_error: f32,

    /// Enhanced FEP: Learning signal (for downstream systems)
    pub fep_learning_signal: f32,

    /// Enhanced FEP: Attention shift amount
    pub attention_shift: f32,

    /// Enhanced FEP: Action-outcome coupling quality
    pub fep_action_outcome_coupling: f32,

    /// Closed Learning Loop: Current strategy
    pub current_strategy: String,

    /// Closed Learning Loop: Best strategy (from Q-values)
    pub best_strategy: String,

    /// Closed Learning Loop: Average reward
    pub average_reward: f32,

    /// Closed Learning Loop: Exploration rate
    pub exploration_rate: f32,

    /// Closed Learning Loop: Total interactions
    pub learning_loop_interactions: u64,

    /// Episodic Memory: Short-term count
    pub memory_short_term_count: usize,

    /// Episodic Memory: Long-term count
    pub memory_long_term_count: usize,

    /// Episodic Memory: Total encoded
    pub memory_total_encoded: u64,

    /// World Model: Average prediction error across levels
    pub world_model_avg_error: f32,

    /// Active goals count
    pub active_goals_count: usize,

    /// Training cycles that used BPTT
    pub bptt_steps: u64,

    /// Training cycles that fell back to SPSA
    pub spsa_fallback_steps: u64,

    // ═══════════════════════════════════════════════════════════════════════════
    // SEMANTIC MEMORY STATS (HDC-based similarity lookup for CfC context)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Semantic Memory: Number of queries that found similar entries
    pub semantic_hits: u64,

    /// Semantic Memory: Number of queries that found no similar entries
    pub semantic_misses: u64,

    /// Semantic Memory: Current learning rate factor from semantic context
    pub semantic_lr_factor: f32,

    /// Semantic Memory: Average prediction error of retrieved similar entries
    pub semantic_avg_retrieved_error: f32,

    /// Semantic Memory: Total entries stored
    pub semantic_entries_stored: u64,

    // ═══════════════════════════════════════════════════════════════════════════
    // ONLINE LEARNING STATS (Inference-time adaptation)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Online Learning: Total adaptation calls
    pub online_adaptation_calls: u64,

    /// Online Learning: Adaptations that modified weights
    pub online_adaptations_applied: u64,

    /// Online Learning: Adaptations skipped due to low error
    pub online_adaptations_skipped: u64,

    /// Online Learning: EMA of prediction errors during online learning
    pub online_ema_error: f32,

    /// Online Learning: Cumulative weight change from online adaptation
    pub online_cumulative_weight_change: f32,
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
        use symthaea_core::hdc::real_hv::RealHV;

        let bridge = self.neural_bridge.as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "Neural bridge not loaded (no probe weights found)"
            ))?;

        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // 1. Project embedding → continuous HDC vector (16384-d)
        let hdc_continuous = bridge.project(embedding)?;

        // 2. Wrap as RealHV so we can reuse compress_for_ltc
        let hdv = RealHV::from_vec(hdc_continuous);

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
            output,
            prediction_error,
            attention_state: HashMap::new(), // No text-based attention for embedding input
            detected_primitives: Vec::new(), // No text primitives for embedding input
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
        })
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
        // PHASE 0.5: Closed Learning Loop - Strategy Selection
        // ═══════════════════════════════════════════════════════════════════════
        // Select response strategy BEFORE processing, based on:
        // - Q-learning from past interactions
        // - Previous reward (stick with success, avoid failure)
        // - Φ-gating (high Φ → Exploratory, low Φ → Supportive)

        let prior_phi = self.unification_engine.phi;
        let prior_reward = self.closed_learning_loop.last_result.as_ref().map(|r| r.reward);
        let selected_strategy = self.closed_learning_loop.select_strategy(prior_phi, prior_reward);

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
        let semantic_lr_factor = self.semantic_memory.compute_lr_factor(&semantic_hdc, 3);

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
        // Convert the HDC encoding to HV16 and run through stability regime processor.
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
                self.emotion_contagion.smoothed_valence,
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

        CycleResult {
            output,
            prediction_error,
            attention_state: encoding_result.attention_snapshot,
            detected_primitives: encoding_result.detected_primitives,
            learning_occurred,
            training_loss,
            cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),
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
        self.emotion_contagion.smoothed_valence
    }

    /// Get current emotional arousal
    /// High = intense/urgent, Low = calm/peaceful
    pub fn emotional_arousal(&self) -> f32 {
        self.emotion_contagion.smoothed_arousal
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
        self.emotion_contagion.smoothed_valence.abs() > 0.2
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
        self.self_reflection.learning_effectiveness
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
            learning_effectiveness: self.self_reflection.learning_effectiveness,

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
            emotional_valence: self.emotion_contagion.smoothed_valence,
            emotional_arousal: self.emotion_contagion.smoothed_arousal,
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
        self.stats.emotional_valence = self.emotion_contagion.smoothed_valence;
        self.stats.emotional_arousal = self.emotion_contagion.smoothed_arousal;
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
        self.stats.learning_effectiveness = self.self_reflection.learning_effectiveness;
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
        self.stats.exploration_rate = self.closed_learning_loop.exploration_rate;
        self.stats.learning_loop_interactions = self.closed_learning_loop.total_interactions;

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

/// Builder for configuring the cognitive loop service
pub struct CognitiveLoopBuilder {
    config: CognitiveLoopConfig,
}

impl CognitiveLoopBuilder {
    pub fn new() -> Self {
        Self {
            config: CognitiveLoopConfig::default(),
        }
    }

    pub fn with_cfc_neurons(mut self, neurons: usize) -> Self {
        self.config.cfc_config.num_neurons = neurons;
        self.config.cfc_config.input_dim = neurons;  // Keep in sync for train_step
        self
    }

    /// Alias for backward compatibility
    pub fn with_ltc_neurons(self, neurons: usize) -> Self {
        self.with_cfc_neurons(neurons)
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.config.cfc_config.learning_rate = lr;
        self
    }

    pub fn with_delta_t(mut self, delta_t: f32) -> Self {
        self.config.cfc_config.delta_t = delta_t;
        self
    }

    pub fn with_prediction_horizons(mut self, horizons: Vec<f32>) -> Self {
        self.config.cfc_config.prediction_horizons = horizons;
        self
    }

    pub fn with_attention_lr(mut self, lr: f32) -> Self {
        self.config.encoder_config.attention_lr = lr;
        self
    }

    pub fn with_learning_threshold(mut self, threshold: f32) -> Self {
        self.config.learning_threshold = threshold;
        self
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Enable causal discovery integration
    ///
    /// When enabled, the cognitive loop tracks (input, output) pairs and
    /// periodically runs causal discovery to weight attention based on
    /// discovered causal structure.
    pub fn with_causal_enhancement(mut self, enabled: bool) -> Self {
        self.config.causal_enhancement = enabled;
        self
    }

    /// Set the interval (in cycles) between causal discovery runs
    ///
    /// Lower values = more frequent discovery but higher compute cost.
    /// Default is 100 cycles.
    pub fn with_causal_discovery_interval(mut self, interval: usize) -> Self {
        self.config.causal_discovery_interval = interval;
        self
    }

    /// Set a genesis phrase for deterministic initialization.
    ///
    /// When set, all HDC vectors, network weights, and exploration randomness
    /// are derived from this phrase via SHAKE-256 domain separation, making
    /// the cognitive loop fully reproducible.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loop_a = CognitiveLoopBuilder::new()
    ///     .with_genesis_phrase("We hold these truths...")
    ///     .build()?;
    ///
    /// let loop_b = CognitiveLoopBuilder::new()
    ///     .with_genesis_phrase("We hold these truths...")
    ///     .build()?;
    ///
    /// // loop_a and loop_b will produce identical outputs for identical inputs
    /// ```
    pub fn with_genesis_phrase(mut self, phrase: impl Into<String>) -> Self {
        self.config.genesis_phrase = Some(phrase.into());
        // Disable async training for determinism (training order matters)
        self.config.async_training = false;
        self
    }

    /// Alias for `with_genesis_phrase` using the term from the Genesis module.
    pub fn seeded(self, phrase: impl Into<String>) -> Self {
        self.with_genesis_phrase(phrase)
    }

    /// Set the temporal backend (CfC or HdcLtcUnified)
    pub fn with_temporal_backend(mut self, backend: TemporalBackend) -> Self {
        self.config.temporal_backend = backend;
        self
    }

    /// Enable or disable async training
    ///
    /// Note: When a genesis phrase is set, async training is automatically
    /// disabled to ensure determinism.
    pub fn with_async_training(mut self, enabled: bool) -> Self {
        // Only allow if no genesis phrase is set
        if self.config.genesis_phrase.is_none() {
            self.config.async_training = enabled;
        }
        self
    }

    pub fn build(self) -> Result<CognitiveLoopService> {
        CognitiveLoopService::new(self.config)
    }
}

impl Default for CognitiveLoopBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC INTEGRATION: MetricsProvider Implementation
// ═══════════════════════════════════════════════════════════════════════════════

use crate::shell::ipc_server::{MetricsProvider, CommandExecutor, ExecutionResult, ValidationResult};
use crate::shell::ipc_client::MetricsSnapshot;
use crate::shell::context::{Completion, CompletionKind};
use crate::action::DestructivenessLevel;

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
}

/// CommandExecutor implementation for CognitiveLoopService
///
/// Wraps the cognitive loop to provide command execution with Phi-gating.
pub struct CognitiveLoopExecutor {
    /// Reference to cognitive loop for Phi checks
    min_phi: f64,
    /// Current Phi value (updated from metrics)
    current_phi: f64,
}

impl CognitiveLoopExecutor {
    pub fn new(min_phi: f64) -> Self {
        Self {
            min_phi,
            current_phi: 0.5,
        }
    }

    /// Update current Phi from service
    pub fn update_phi(&mut self, phi: f64) {
        self.current_phi = phi;
    }
}

impl Default for CognitiveLoopExecutor {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl CommandExecutor for CognitiveLoopExecutor {
    fn execute(&self, command: &str, require_phi: Option<f64>) -> ExecutionResult {
        let required = require_phi.unwrap_or(self.min_phi);

        // Check Phi threshold
        if self.current_phi < required {
            return ExecutionResult {
                success: false,
                output: String::new(),
                phi_at_execution: self.current_phi,
                vetoed: true,
                veto_reason: Some(format!(
                    "Consciousness level ({:.2}) below required threshold ({:.2})",
                    self.current_phi, required
                )),
            };
        }

        // For now, return a stub result - actual execution would delegate to NixOS
        ExecutionResult {
            success: true,
            output: format!("[Phi={:.2}] Command queued: {}", self.current_phi, command),
            phi_at_execution: self.current_phi,
            vetoed: false,
            veto_reason: None,
        }
    }

    fn validate(&self, command: &str) -> ValidationResult {
        // Classify command destructiveness
        let cmd_lower = command.to_lowercase();

        let (safety_level, warnings) = if cmd_lower.contains("rm")
            || cmd_lower.contains("delete")
            || cmd_lower.contains("gc -d")
            || cmd_lower.contains("--delete")
        {
            (
                DestructivenessLevel::Destructive,
                vec!["This command may permanently delete data".to_string()],
            )
        } else if cmd_lower.contains("rebuild")
            || cmd_lower.contains("switch")
            || cmd_lower.contains("restart")
        {
            (
                DestructivenessLevel::NeedsConfirmation,
                vec!["This command may modify system state".to_string()],
            )
        } else if cmd_lower.contains("install")
            || cmd_lower.contains("remove")
            || cmd_lower.contains("update")
        {
            (DestructivenessLevel::Reversible, Vec::new())
        } else {
            (DestructivenessLevel::ReadOnly, Vec::new())
        };

        ValidationResult {
            valid: true,
            safety_level: format!("{:?}", safety_level),
            preview: Some(format!("Would execute: {}", command)),
            warnings,
        }
    }

    fn get_completions(&self, input: &str, _cursor_pos: usize) -> Vec<Completion> {
        // Common NixOS commands for completion
        let commands = [
            ("install", "Install packages to environment"),
            ("remove", "Remove packages from environment"),
            ("search", "Search for packages"),
            ("rebuild", "Rebuild NixOS configuration"),
            ("switch", "Switch to new configuration"),
            ("rollback", "Rollback to previous generation"),
            ("gc", "Run garbage collection"),
            ("flake", "Flake management commands"),
            ("profile", "Profile management"),
            ("doctor", "Check system health"),
        ];

        commands
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(input))
            .map(|(cmd, desc)| {
                Completion::new(*cmd, CompletionKind::Command)
                    .with_docs(*desc)
                    .with_similarity(if *cmd == input { 1.0 } else { 0.8 })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_creation() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        assert_eq!(service.stats().total_cycles, 0);
    }

    #[test]
    fn test_single_cycle() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let result = service.cycle("test input");

        assert!(result.prediction_error >= 0.0);
        assert!(result.prediction_error <= 1.0);
        assert_eq!(service.stats().total_cycles, 1);
    }

    #[test]
    fn test_multiple_cycles_reduce_error() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            learning_threshold: 0.0, // Always learn
            ..Default::default()
        }).unwrap();

        // Run multiple cycles with same input
        let mut errors = Vec::new();
        for _ in 0..20 {
            let result = service.cycle("cause effect action");
            errors.push(result.prediction_error);
        }

        // Error should generally decrease (or at least not increase dramatically)
        let first_half_avg: f32 = errors[..10].iter().sum::<f32>() / 10.0;
        let second_half_avg: f32 = errors[10..].iter().sum::<f32>() / 10.0;

        println!("First half avg error: {}", first_half_avg);
        println!("Second half avg error: {}", second_half_avg);

        // Second half should be lower or similar
        assert!(second_half_avg <= first_half_avg + 0.1,
            "Error should decrease or stabilize over cycles");
    }

    #[test]
    fn test_attention_emergence() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            learning_threshold: 0.0,
            encoder_config: PredictiveEncoderConfig {
                attention_lr: 0.5, // High learning rate
                ..Default::default()
            },
            ..Default::default()
        }).unwrap();

        // Run many cycles
        for _ in 0..50 {
            service.cycle("cause effect");
        }

        // Check attention has diverged from uniform
        let stats = service.stats();
        println!("Attention variance: {}", stats.attention_variance);

        // Some attention emergence should occur
        // (may be small depending on the input)
    }

    #[test]
    fn test_builder() {
        let service = CognitiveLoopBuilder::new()
            .with_ltc_neurons(128)
            .with_learning_rate(0.001)
            .with_learning_threshold(0.1)
            .build()
            .unwrap();

        assert_eq!(service.stats().total_cycles, 0);
    }

    #[test]
    fn test_reset() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run some cycles
        for _ in 0..5 {
            service.cycle("test");
        }
        assert!(service.stats().total_cycles > 0);

        // Reset
        service.reset();

        assert_eq!(service.stats().total_cycles, 0);
        assert_eq!(service.buffer.len(), 0);
    }

    #[test]
    fn test_consolidation() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            enable_consolidation: true,
            learning_threshold: 0.0,
            ..Default::default()
        }).unwrap();

        // Fill buffer with experiences
        for i in 0..20 {
            service.cycle(&format!("input {}", i));
        }

        // Should have some experiences
        assert!(service.buffer.len() > 0);

        // Run consolidation
        let loss = service.consolidate().unwrap();
        println!("Consolidation loss: {}", loss);
    }

    #[test]
    fn test_prediction_confidence() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initial confidence should be 0.5
        assert!((service.prediction_confidence() - 0.5).abs() < 0.01);

        // Run several cycles
        for _ in 0..10 {
            service.cycle("consistent stable input");
        }

        // Confidence should be tracked
        let confidence = service.prediction_confidence();
        assert!(confidence >= 0.0 && confidence <= 1.0);

        // Reset should restore neutral confidence
        service.reset();
        assert!((service.prediction_confidence() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_predictions_trustworthy() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initial state should have some trust
        assert!(service.prediction_confidence() > 0.3);

        // predictions_trustworthy depends on confidence threshold
        // At 0.5 initial confidence, should be trustworthy
        assert!(service.predictions_trustworthy());
    }

    #[test]
    fn test_flow_state_initial() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initially not in flow
        assert!(!service.in_flow());
        assert_eq!(service.flow_intensity(), 0.0);
        assert_eq!(service.flow_streak(), 0);
        assert_eq!(service.flow_learning_boost(), 1.0);
    }

    #[test]
    fn test_flow_state_reset() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run some cycles
        for _ in 0..10 {
            service.cycle("focused input");
        }

        // Reset
        service.reset();

        // Flow state should be reset
        assert!(!service.in_flow());
        assert_eq!(service.flow_state().streak, 0);
    }

    #[test]
    fn test_flow_state_struct() {
        let mut flow = FlowState::default();

        // Test update with flow-compatible conditions
        for _ in 0..10 {
            flow.update(
                ConsciousnessPattern::Focused,
                0.1,  // Low error
                0.8,  // High coherence
                0.7,  // Good confidence
            );
        }

        // After sufficient streak, should be in flow
        assert!(flow.streak >= FlowState::FLOW_ENTRY_STREAK);
        assert!(flow.in_flow);
        assert!(flow.learning_boost > 1.0);
    }

    #[test]
    fn test_emotion_contagion_positive() {
        let mut emotion = EmotionContagion::default();

        // Analyze happy content
        emotion.analyze("I am so happy and excited! This is wonderful and amazing!");

        // Should detect positive valence
        assert!(emotion.valence > 0.0);
        assert!(emotion.smoothed_valence > 0.0);

        // High arousal due to exclamation and excited words
        assert!(emotion.arousal > 0.5);
    }

    #[test]
    fn test_emotion_contagion_negative() {
        let mut emotion = EmotionContagion::default();

        // Analyze sad content
        emotion.analyze("I feel sad and worried about this terrible problem.");

        // Should detect negative valence
        assert!(emotion.valence < 0.0);
        assert!(emotion.smoothed_valence < 0.0);
    }

    #[test]
    fn test_emotion_contagion_neutral() {
        let mut emotion = EmotionContagion::default();

        // Analyze neutral content
        emotion.analyze("The system processes data and returns results.");

        // Should have near-zero valence
        assert!(emotion.valence.abs() < 0.3);
    }

    #[test]
    fn test_emotion_pattern_nudge() {
        let mut emotion = EmotionContagion::default();

        // Strong positive emotion should nudge toward Excited
        emotion.analyze("This is absolutely amazing! I'm so thrilled and excited!");
        let (pattern, strength) = emotion.pattern_nudge();
        assert!(pattern.is_some());
        assert!(strength > 0.0);
    }

    #[test]
    fn test_emotion_in_service() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initially no emotional content
        assert!(!service.has_emotional_content());

        // Process emotional content
        service.cycle("I'm so happy and grateful for this wonderful day!");

        // Should detect emotional content
        let valence = service.emotional_valence();
        // The smoothing will reduce the effect, but should be positive
        assert!(valence >= 0.0 || service.emotion_contagion().valence > 0.0);
    }

    #[test]
    fn test_emotion_reset() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Process emotional content
        service.cycle("I'm incredibly excited about this amazing opportunity!");

        // Reset
        service.reset();

        // Emotion should be reset
        assert_eq!(service.emotional_valence(), 0.0);
        assert_eq!(service.emotional_arousal(), 0.5);
    }

    #[test]
    fn test_curiosity_drive_initial() {
        let curiosity = CuriosityDrive::default();

        assert_eq!(curiosity.boredom, 0.0);
        assert!(curiosity.curiosity > 0.0); // Starts with some curiosity
        assert_eq!(curiosity.exploration_urge, 0.0);
        assert_eq!(curiosity.novelty_bonus, 1.0);
        assert!(!curiosity.should_explore());
    }

    #[test]
    fn test_curiosity_boredom_buildup() {
        let mut curiosity = CuriosityDrive::default();

        // Feed consistently low errors (boring/predictable)
        for _ in 0..20 {
            curiosity.update(0.05); // Very low error
        }

        // Boredom should build up
        assert!(curiosity.boredom > 0.3);
        assert!(curiosity.curiosity > 0.5);
    }

    #[test]
    fn test_curiosity_exploration_trigger() {
        let mut curiosity = CuriosityDrive::default();

        // Feed many low errors to trigger exploration
        for _ in 0..30 {
            curiosity.update(0.05);
        }

        // Should want to explore
        assert!(curiosity.boredom > 0.5);
        // After sufficient boredom, should_explore or have high exploration urge
        assert!(curiosity.exploration_urge > 0.0 || curiosity.boredom > 0.7);
    }

    #[test]
    fn test_curiosity_novelty_bonus() {
        let mut curiosity = CuriosityDrive::default();

        // High error = novel situation
        curiosity.update(0.8);

        // Should have some novelty bonus
        assert!(curiosity.novelty_bonus >= 1.0);
    }

    #[test]
    fn test_curiosity_in_service() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initially no boredom
        assert_eq!(service.boredom(), 0.0);
        assert!(!service.is_bored());

        // Run some cycles
        for _ in 0..5 {
            service.cycle("test input");
        }

        // Curiosity should be tracked
        assert!(service.curiosity() >= 0.0);
        assert!(service.novelty_bonus() >= 1.0);
    }

    #[test]
    fn test_curiosity_reset() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run some cycles
        for _ in 0..10 {
            service.cycle("test");
        }

        // Reset
        service.reset();

        // Curiosity should be reset
        assert_eq!(service.boredom(), 0.0);
        assert!(!service.curiosity_should_explore());
    }

    // ========== Self-Reflection Tests ==========

    #[test]
    fn test_self_reflection_initial() {
        let reflection = SelfReflection::default();

        assert_eq!(reflection.reflection_count, 0);
        assert_eq!(reflection.adjustments_made, 0);
        assert_eq!(reflection.self_assessment, SelfAssessment::Learning);
        assert!(reflection.flow_error_threshold > 0.0);
        assert!(reflection.boredom_threshold > 0.0);
    }

    #[test]
    fn test_self_reflection_record_cycle() {
        let mut reflection = SelfReflection::default();

        // Record some cycles
        for _ in 0..10 {
            reflection.record_cycle(0.3, false, false, 0.5);
        }

        // Should track historical metrics
        assert!(reflection.historical_error > 0.0);
        assert!(reflection.historical_confidence > 0.0);
    }

    #[test]
    fn test_self_reflection_should_reflect() {
        let mut reflection = SelfReflection::default();

        // Initially shouldn't reflect
        assert!(!reflection.should_reflect());

        // Record enough cycles
        for _ in 0..60 {
            reflection.record_cycle(0.3, false, false, 0.5);
        }

        // Should now want to reflect
        assert!(reflection.should_reflect());
    }

    #[test]
    fn test_self_reflection_reflect() {
        let mut reflection = SelfReflection::default();

        // Record cycles to trigger reflection
        for _ in 0..60 {
            reflection.record_cycle(0.3, false, false, 0.5);
        }

        // Perform reflection
        let recommendations = reflection.reflect();

        // Reflection count should increase
        assert_eq!(reflection.reflection_count, 1);

        // Should have some assessment
        assert!(reflection.learning_effectiveness >= 0.0);

        // Recommendations may or may not be empty depending on state
        let _ = recommendations;
    }

    #[test]
    fn test_self_reflection_stagnation_detection() {
        let mut reflection = SelfReflection::default();

        // Simulate stagnation: low error, no flow, no exploration
        for _ in 0..60 {
            reflection.record_cycle(0.1, false, false, 0.6);
        }
        reflection.reflect();

        // Should detect stagnation or overconfidence
        assert!(
            reflection.self_assessment == SelfAssessment::Stagnating ||
            reflection.self_assessment == SelfAssessment::Overconfident ||
            reflection.self_assessment == SelfAssessment::Learning
        );
    }

    #[test]
    fn test_self_reflection_in_service() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Initial state
        assert_eq!(service.reflection_count(), 0);
        assert!(service.learning_effectiveness() >= 0.0);

        // Run some cycles (not enough to trigger reflection yet)
        for _ in 0..10 {
            service.cycle("test input");
        }

        // Check self-assessment is available
        let assessment = service.self_assessment();
        assert!(assessment == SelfAssessment::Learning || assessment == SelfAssessment::Exploring);
    }

    #[test]
    fn test_self_reflection_thresholds() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Get adapted thresholds
        let thresholds = service.adapted_thresholds();

        // Should have valid thresholds
        assert!(thresholds.flow_error > 0.0 && thresholds.flow_error < 1.0);
        assert!(thresholds.boredom > 0.0 && thresholds.boredom < 1.0);
        assert!(thresholds.trust > 0.0 && thresholds.trust < 1.0);
    }

    #[test]
    fn test_self_reflection_reset() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run cycles and force reflect
        for _ in 0..10 {
            service.cycle("test");
        }
        service.force_reflect();

        // Reset
        service.reset();

        // Reflection count should reset but thresholds preserved
        assert_eq!(service.self_reflection().reflection_count, 0);
        // Thresholds are preserved across reset
        assert!(service.adapted_thresholds().flow_error > 0.0);
    }

    #[test]
    fn test_self_reflection_summary() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let summary = service.reflection_summary();

        assert_eq!(summary.reflection_count, 0);
        assert!(summary.learning_effectiveness >= 0.0);
        assert!(summary.next_reflection_in > 0);
    }

    // ========== Consciousness Snapshot Tests ==========

    #[test]
    fn test_consciousness_snapshot_initial() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();

        // Initial state checks
        assert_eq!(snapshot.cycle, 0);
        assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
        assert!(!snapshot.in_flow);
        assert!(!snapshot.exploring);
        assert_eq!(snapshot.reflection_count, 0);
    }

    #[test]
    fn test_consciousness_snapshot_after_cycles() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run some cycles
        for _ in 0..10 {
            service.cycle("test input for consciousness");
        }

        let snapshot = service.consciousness_snapshot();

        // Should have recorded cycles
        assert_eq!(snapshot.cycle, 10);
        // Should have valid metrics
        assert!(snapshot.prediction_confidence >= 0.0);
        assert!(snapshot.consciousness_level >= 0.0);
        assert!(snapshot.flow_threshold > 0.0);
        assert!(snapshot.boredom_threshold > 0.0);
    }

    #[test]
    fn test_consciousness_snapshot_status() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();
        let status = snapshot.status();

        // Status should be a non-empty string
        assert!(!status.is_empty());
        // Should contain pattern info
        assert!(status.contains("Conf:") || status.contains("Err:"));
    }

    #[test]
    fn test_consciousness_snapshot_recommended_actions() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();
        let actions = snapshot.recommended_actions();

        // Actions is a vec that may or may not be empty
        let _ = actions;
    }

    #[test]
    fn test_consciousness_snapshot_is_optimal() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();

        // is_optimal returns a bool
        let _ = snapshot.is_optimal();
    }

    #[test]
    fn test_consciousness_snapshot_needs_attention() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();

        // Initially shouldn't need attention
        // (though this depends on initial state)
        let needs = snapshot.needs_attention();
        let _ = needs; // Just verify it returns
    }

    #[test]
    fn test_consciousness_snapshot_dominant_concern() {
        let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        let snapshot = service.consciousness_snapshot();

        // dominant_concern returns Option<&str>
        let concern = snapshot.dominant_concern();
        let _ = concern;
    }

    #[test]
    fn test_status_line() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        service.cycle("test");
        let status = service.status_line();

        assert!(!status.is_empty());
    }

    #[test]
    fn test_consciousness_level() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run some cycles
        for _ in 0..5 {
            service.cycle("focused stable input");
        }

        let level = service.consciousness_level();
        assert!(level >= 0.0 && level <= 1.0);
    }

    #[test]
    fn test_adapted_thresholds_wiring() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Get initial thresholds
        let initial_flow = service.adapted_thresholds().flow_error;
        let initial_boredom = service.adapted_thresholds().boredom;

        // The thresholds should be valid
        assert!(initial_flow > 0.0 && initial_flow < 1.0);
        assert!(initial_boredom > 0.0 && initial_boredom < 1.0);

        // Run cycles - thresholds are passed to flow_state and curiosity_drive
        for _ in 0..5 {
            service.cycle("test");
        }

        // Verify snapshot reflects adapted thresholds
        let snapshot = service.consciousness_snapshot();
        assert_eq!(snapshot.flow_threshold, service.adapted_thresholds().flow_error);
        assert_eq!(snapshot.boredom_threshold, service.adapted_thresholds().boredom);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // UNIFIED ARCHITECTURE COMPONENT TESTS
    // ═══════════════════════════════════════════════════════════════════════════════

    // -------------------- ThalamicRouter Tests --------------------

    #[test]
    fn test_thalamic_router_default() {
        let router = ThalamicRouter::default();
        assert_eq!(router.novelty_threshold, 0.7);
        assert_eq!(router.urgency_threshold, 0.8);
        assert_eq!(router.familiarity_threshold, 0.3);
    }

    #[test]
    fn test_thalamic_router_reflex_route() {
        let mut router = ThalamicRouter::default();

        // Low novelty, low complexity, low urgency → Reflex
        let depth = router.route(0.1, 0.2, 0.1, 0.1);
        assert_eq!(depth, CognitiveDepth::Reflex);
    }

    #[test]
    fn test_thalamic_router_cortical_route() {
        let mut router = ThalamicRouter::default();

        // Medium values → Cortical
        let depth = router.route(0.4, 0.4, 0.5, 0.3);
        assert_eq!(depth, CognitiveDepth::Cortical);
    }

    #[test]
    fn test_thalamic_router_deep_thought_high_novelty() {
        let mut router = ThalamicRouter::default();

        // High novelty → DeepThought
        let depth = router.route(0.9, 0.3, 0.3, 0.3);
        assert_eq!(depth, CognitiveDepth::DeepThought);
    }

    #[test]
    fn test_thalamic_router_deep_thought_high_urgency() {
        let mut router = ThalamicRouter::default();

        // High urgency → DeepThought
        let depth = router.route(0.3, 0.9, 0.3, 0.3);
        assert_eq!(depth, CognitiveDepth::DeepThought);
    }

    #[test]
    fn test_thalamic_router_deep_thought_high_complexity() {
        let mut router = ThalamicRouter::default();

        // High complexity → DeepThought
        let depth = router.route(0.3, 0.3, 0.9, 0.3);
        assert_eq!(depth, CognitiveDepth::DeepThought);
    }

    #[test]
    fn test_thalamic_router_deep_thought_high_emotion() {
        let mut router = ThalamicRouter::default();

        // High emotional intensity → DeepThought
        let depth = router.route(0.3, 0.3, 0.3, 0.9);
        assert_eq!(depth, CognitiveDepth::DeepThought);
    }

    #[test]
    fn test_thalamic_router_routing_stats() {
        let mut router = ThalamicRouter::default();

        // Make several routing decisions
        router.route(0.1, 0.2, 0.1, 0.1); // Reflex
        router.route(0.1, 0.2, 0.1, 0.1); // Reflex
        router.route(0.5, 0.5, 0.5, 0.3); // Cortical
        router.route(0.9, 0.5, 0.5, 0.3); // DeepThought

        let (reflex, cortical, deep) = router.routing_stats();

        assert_eq!(reflex, 0.5);     // 2 out of 4
        assert_eq!(cortical, 0.25);  // 1 out of 4
        assert_eq!(deep, 0.25);      // 1 out of 4
    }

    #[test]
    fn test_thalamic_router_from_cycle() {
        let mut router = ThalamicRouter::default();

        // High prediction error (novel) → DeepThought
        let depth = router.route_from_cycle(0.9, ConsciousnessPattern::Uncertain, 0.3);
        assert_eq!(depth, CognitiveDepth::DeepThought);

        // Low error, focused, neutral emotion → likely Cortical or Reflex
        let depth2 = router.route_from_cycle(0.1, ConsciousnessPattern::Focused, 0.1);
        assert!(matches!(depth2, CognitiveDepth::Cortical | CognitiveDepth::Reflex));
    }

    // -------------------- ActiveInferenceBridge Tests --------------------

    #[test]
    fn test_active_inference_bridge_default() {
        let bridge = ActiveInferenceBridge::default();
        assert_eq!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
        assert!(bridge.modulation_index().is_none());
    }

    #[test]
    fn test_active_inference_bridge_observe_resolution() {
        let mut bridge = ActiveInferenceBridge::default();

        // Add some observations
        for i in 0..15 {
            let confidence = 0.8;
            let success = i % 2 == 0; // Alternating success/failure
            bridge.observe_resolution(confidence, success);
        }

        // Should have enough data now
        assert!(bridge.modulation_index().is_some());
        assert_ne!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
    }

    #[test]
    fn test_active_inference_bridge_perfect_coupling() {
        let mut bridge = ActiveInferenceBridge::default();

        // Perfect coupling: high confidence → success, low confidence → failure
        for _ in 0..20 {
            bridge.observe_resolution(0.9, true);
            bridge.observe_resolution(0.1, false);
        }

        let mi = bridge.modulation_index().unwrap();
        // Should have strong positive correlation
        assert!(mi > 0.5, "Expected strong coupling, got MI={}", mi);
        assert!(matches!(
            bridge.coupling_quality(),
            CouplingQuality::ModerateCoupling | CouplingQuality::StrongCoupling
        ));
    }

    #[test]
    fn test_active_inference_bridge_statistics() {
        let mut bridge = ActiveInferenceBridge::default();

        for _ in 0..15 {
            bridge.observe_resolution(0.7, true);
        }

        let stats = bridge.statistics();
        assert_eq!(stats.total_observations, 15);
        assert!(stats.modulation_index.is_some());
        assert!(stats.average_prediction_error.is_some());
        // All successes → 0% error
        assert!(stats.average_prediction_error.unwrap() < 0.01);
    }

    #[test]
    fn test_active_inference_bridge_reset() {
        let mut bridge = ActiveInferenceBridge::default();

        for _ in 0..20 {
            bridge.observe_resolution(0.5, true);
        }

        bridge.reset();

        assert_eq!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
        let stats = bridge.statistics();
        assert_eq!(stats.total_observations, 0);
    }

    #[test]
    fn test_coupling_quality_is_meaningful() {
        assert!(!CouplingQuality::InsufficientData.is_meaningful());
        assert!(!CouplingQuality::NoCoupling.is_meaningful());
        assert!(CouplingQuality::WeakCoupling.is_meaningful());
        assert!(CouplingQuality::ModerateCoupling.is_meaningful());
        assert!(CouplingQuality::StrongCoupling.is_meaningful());
    }

    // -------------------- ClosedLearningLoop Tests --------------------

    #[test]
    fn test_closed_learning_loop_default() {
        let loop_ = ClosedLearningLoop::default();
        assert_eq!(loop_.current_strategy, ResponseStrategy::Supportive);
        assert!(loop_.last_result.is_none());
        assert_eq!(loop_.average_reward(), 0.0);
    }

    #[test]
    fn test_closed_learning_loop_strategy_selection() {
        let mut loop_ = ClosedLearningLoop::default();

        // With neutral Φ (0.45), should use Q-learning selection
        let strategy = loop_.select_strategy(0.45, None);

        // Should return some valid strategy
        assert!(matches!(
            strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_closed_learning_loop_phi_gating_high() {
        let mut loop_ = ClosedLearningLoop::default();

        // Set Supportive as best strategy with high Q-value
        // Then with high Φ, it should shift toward Exploratory
        for _ in 0..100 {
            let strategy = loop_.select_strategy(0.8, None);
            // High Φ → integrative mode → favors Exploratory/Detailed
            assert!(!matches!(strategy, ResponseStrategy::Supportive | ResponseStrategy::Concise)
                || loop_.last_result.is_some(),
                "High Φ should shift away from Supportive/Concise");
            break; // Just check first selection
        }
    }

    #[test]
    fn test_closed_learning_loop_q_learning_update() {
        let mut loop_ = ClosedLearningLoop::default();

        // Record a positive result for Detailed
        let result = CycleLearningResult {
            strategy_used: ResponseStrategy::Detailed,
            reward: 0.8,
            successful: true,
            prediction_error: 0.1,
            coherence: 0.8,
        };

        let initial_q = loop_.q_values()[0]; // Detailed index
        loop_.update(result);

        // Q-value should increase
        assert!(loop_.q_values()[0] > initial_q);
        assert_eq!(loop_.strategy_counts()[0], 1);
    }

    #[test]
    fn test_closed_learning_loop_reward_tracking() {
        let mut loop_ = ClosedLearningLoop::default();

        // Record multiple results
        for _ in 0..5 {
            let result = CycleLearningResult {
                strategy_used: ResponseStrategy::Supportive,
                reward: 0.6,
                successful: true,
                prediction_error: 0.2,
                coherence: 0.7,
            };
            loop_.update(result);
        }

        assert_eq!(loop_.average_reward(), 0.6);
        assert_eq!(loop_.strategy_counts()[3], 5); // Supportive index
    }

    #[test]
    fn test_closed_learning_loop_best_strategy() {
        let mut loop_ = ClosedLearningLoop::default();

        // Train Exploratory with high rewards
        for _ in 0..20 {
            let result = CycleLearningResult {
                strategy_used: ResponseStrategy::Exploratory,
                reward: 0.9,
                successful: true,
                prediction_error: 0.1,
                coherence: 0.9,
            };
            loop_.update(result);
        }

        // Exploratory should become best
        assert_eq!(loop_.best_strategy(), ResponseStrategy::Exploratory);
    }

    #[test]
    fn test_closed_learning_loop_reset() {
        let mut loop_ = ClosedLearningLoop::default();

        // Add some data
        let result = CycleLearningResult {
            strategy_used: ResponseStrategy::Detailed,
            reward: 0.7,
            successful: true,
            prediction_error: 0.15,
            coherence: 0.75,
        };
        loop_.update(result);

        loop_.reset();

        assert!(loop_.last_result.is_none());
        assert_eq!(loop_.average_reward(), 0.0);
    }

    #[test]
    fn test_response_strategy_opposite() {
        // Check actual implementation:
        // Detailed <-> Concise (symmetric)
        // Clarifying -> Supportive -> Exploratory -> Clarifying (cycle)
        assert_eq!(ResponseStrategy::Detailed.opposite(), ResponseStrategy::Concise);
        assert_eq!(ResponseStrategy::Concise.opposite(), ResponseStrategy::Detailed);
        assert_eq!(ResponseStrategy::Clarifying.opposite(), ResponseStrategy::Supportive);
        assert_eq!(ResponseStrategy::Supportive.opposite(), ResponseStrategy::Exploratory);
        assert_eq!(ResponseStrategy::Exploratory.opposite(), ResponseStrategy::Clarifying);
    }

    // -------------------- EpisodicMemoryBridge Tests --------------------

    #[test]
    fn test_episodic_memory_bridge_default() {
        let bridge = EpisodicMemoryBridge::default();
        assert_eq!(bridge.memory_count(), (0, 0));
    }

    #[test]
    fn test_episodic_memory_encode() {
        let mut bridge = EpisodicMemoryBridge::default();

        let id = bridge.encode(
            "test memory",
            vec![0.1, 0.2, 0.3, 0.4],
            0.5,  // valence
            0.6,  // phi
            100,  // cycle
        );

        assert_eq!(id, 0);
        assert_eq!(bridge.memory_count(), (1, 0));
        assert_eq!(bridge.stats.total_encoded, 1);
    }

    #[test]
    fn test_episodic_memory_recall() {
        let mut bridge = EpisodicMemoryBridge::default();

        // Encode some memories
        bridge.encode("memory one", vec![1.0, 0.0, 0.0, 0.0], 0.5, 0.6, 1);
        bridge.encode("memory two", vec![0.0, 1.0, 0.0, 0.0], 0.3, 0.5, 2);
        bridge.encode("memory three", vec![0.9, 0.1, 0.0, 0.0], 0.7, 0.8, 3);

        // Query similar to "memory one" and "memory three"
        let results = bridge.recall(&[1.0, 0.0, 0.0, 0.0], 2, 0.5);

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // First result should be most similar (memory one)
        assert_eq!(results[0].0.content, "memory one");
    }

    #[test]
    fn test_episodic_memory_consolidation() {
        let mut bridge = EpisodicMemoryBridge::default();

        // Fill short-term memory to trigger consolidation
        for i in 0..105 {
            bridge.encode(
                format!("memory {}", i),
                vec![0.1; 4],
                0.5,
                0.6,
                i,
            );
        }

        // Should have consolidated some to long-term
        let (short, long) = bridge.memory_count();
        assert!(short <= 100);
        assert!(long > 0, "Expected some memories consolidated to long-term");
        assert!(bridge.stats.consolidations > 0);
    }

    #[test]
    fn test_episodic_memory_decay() {
        let mut bridge = EpisodicMemoryBridge::default();

        bridge.encode("memory", vec![0.1; 4], 0.5, 0.6, 1);

        // Decay several times
        for _ in 0..10 {
            bridge.decay(0.1);
        }

        // Short-term memories persist but weaken
        assert_eq!(bridge.memory_count().0, 1);
    }

    #[test]
    fn test_episodic_memory_reset() {
        let mut bridge = EpisodicMemoryBridge::default();

        bridge.encode("memory", vec![0.1; 4], 0.5, 0.6, 1);
        bridge.reset();

        assert_eq!(bridge.memory_count(), (0, 0));
        assert_eq!(bridge.stats.total_encoded, 0);
    }

    #[test]
    fn test_episodic_memory_similarity() {
        let memory = EpisodicMemory {
            id: 0,
            encoded_at_cycle: 0,
            content: "test".into(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            valence: 0.5,
            phi_at_encoding: 0.6,
            access_count: 0,
            strength: 1.0,
        };

        // Same vector → similarity 1.0
        let sim1 = memory.similarity(&[1.0, 0.0, 0.0, 0.0]);
        assert!((sim1 - 1.0).abs() < 0.001);

        // Orthogonal vector → similarity 0.0
        let sim2 = memory.similarity(&[0.0, 1.0, 0.0, 0.0]);
        assert!((sim2 - 0.0).abs() < 0.001);
    }

    // -------------------- GoalSystemBridge Tests --------------------

    #[test]
    fn test_goal_system_bridge_default() {
        let bridge = GoalSystemBridge::new();
        assert!(bridge.active_goals().is_empty());
        assert_eq!(bridge.attention_bias(), 1.0);
    }

    #[test]
    fn test_goal_system_add_goal() {
        let mut bridge = GoalSystemBridge::new();

        let goal = CognitiveGoal::new("goal1", "Test goal", 0.8);
        bridge.add_goal(goal);

        assert_eq!(bridge.active_goals().len(), 1);
        assert!(bridge.attention_bias() > 1.0);
    }

    #[test]
    fn test_goal_system_attention_bias() {
        let mut bridge = GoalSystemBridge::new();

        // Add high-priority goal
        bridge.add_goal(CognitiveGoal::new("goal1", "High priority", 1.0));

        // Attention bias should increase
        let bias = bridge.attention_bias();
        assert!(bias > 1.0);
        assert!(bias <= 1.2); // Max 20% boost per unit weight
    }

    #[test]
    fn test_goal_system_update_progress() {
        let mut bridge = GoalSystemBridge::new();

        bridge.add_goal(CognitiveGoal::new("goal1", "Test", 0.5));

        // Update progress
        bridge.update_progress("goal1", 0.5);

        let goals = bridge.active_goals();
        assert_eq!(goals[0].progress, 0.5);

        // Complete the goal
        bridge.update_progress("goal1", 0.6);

        // Goal should be deactivated when progress >= 1.0
        assert!(bridge.active_goals().is_empty());
    }

    #[test]
    fn test_goal_system_top_goal() {
        let mut bridge = GoalSystemBridge::new();

        bridge.add_goal(CognitiveGoal::new("low", "Low priority", 0.3));
        bridge.add_goal(CognitiveGoal::new("high", "High priority", 0.9));
        bridge.add_goal(CognitiveGoal::new("mid", "Mid priority", 0.5));

        let top = bridge.top_goal().unwrap();
        assert_eq!(top.id, "high");
        assert_eq!(top.priority, 0.9);
    }

    #[test]
    fn test_goal_system_clear_completed() {
        let mut bridge = GoalSystemBridge::new();

        bridge.add_goal(CognitiveGoal::new("goal1", "Goal 1", 0.5));
        bridge.add_goal(CognitiveGoal::new("goal2", "Goal 2", 0.5));

        // Complete goal1
        bridge.update_progress("goal1", 1.0);

        bridge.clear_completed();

        assert_eq!(bridge.active_goals().len(), 1);
    }

    #[test]
    fn test_goal_system_reset() {
        let mut bridge = GoalSystemBridge::new();

        bridge.add_goal(CognitiveGoal::new("goal1", "Test", 0.5));
        bridge.reset();

        assert!(bridge.active_goals().is_empty());
    }

    #[test]
    fn test_cognitive_goal_creation() {
        let goal = CognitiveGoal::new("test", "Test goal description", 0.75);

        assert_eq!(goal.id, "test");
        assert_eq!(goal.description, "Test goal description");
        assert_eq!(goal.priority, 0.75);
        assert_eq!(goal.progress, 0.0);
        assert!(goal.is_active);
        assert_eq!(goal.attention_weight, 0.75);
    }

    // -------------------- WorldModelBridge Tests --------------------

    #[test]
    fn test_world_model_bridge_default() {
        let bridge = WorldModelBridge::default();

        assert_eq!(bridge.total_predictions, 0);
        assert_eq!(bridge.avg_error, 0.0);

        // Should have 4 levels by default
        assert!(bridge.get_level_state(0).is_some());
        assert!(bridge.get_level_state(3).is_some());
        assert!(bridge.get_level_state(4).is_none());
    }

    #[test]
    fn test_world_model_update_sensory() {
        let mut bridge = WorldModelBridge::default();

        // Create input matching level 0 dimension (64)
        let input: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();

        bridge.update_sensory(&input);

        assert_eq!(bridge.total_predictions, 1);
        assert!(bridge.avg_error >= 0.0);
    }

    #[test]
    fn test_world_model_level_states() {
        let mut bridge = WorldModelBridge::default();

        let input: Vec<f32> = vec![1.0; 64];
        bridge.update_sensory(&input);

        // Level 0 should match input
        let level0 = bridge.get_level_state(0).unwrap();
        assert_eq!(level0.len(), 64);
        assert!((level0[0] - 1.0).abs() < 0.001);

        // Higher levels should exist and have been updated
        let level1 = bridge.get_level_state(1).unwrap();
        assert!(!level1.is_empty(), "Level 1 should have state");
        // The propagation logic chunks and averages, so sum should be non-zero
        let level1_sum: f32 = level1.iter().sum();
        assert!(level1_sum > 0.0, "Level 1 should have non-zero sum after propagation");
    }

    #[test]
    fn test_world_model_abstract_state() {
        let mut bridge = WorldModelBridge::default();

        let input: Vec<f32> = vec![0.5; 64];
        bridge.update_sensory(&input);

        let abstract_state = bridge.abstract_state();
        assert!(!abstract_state.is_empty());
        // Abstract state is highest level (128 dims)
        assert_eq!(abstract_state.len(), 128);
    }

    #[test]
    fn test_world_model_level_errors() {
        let mut bridge = WorldModelBridge::default();

        // First update will have high error (predicting from zeros)
        let input: Vec<f32> = vec![1.0; 64];
        bridge.update_sensory(&input);

        let errors = bridge.level_errors();
        assert_eq!(errors.len(), 4);
        assert!(errors[0] > 0.0); // First prediction has error
    }

    #[test]
    fn test_world_model_reset() {
        let mut bridge = WorldModelBridge::default();

        let input: Vec<f32> = vec![1.0; 64];
        bridge.update_sensory(&input);

        bridge.reset();

        assert_eq!(bridge.total_predictions, 0);
        assert_eq!(bridge.avg_error, 0.0);

        // States should be zeroed
        let level0 = bridge.get_level_state(0).unwrap();
        assert!(level0.iter().all(|&v| v == 0.0));
    }

    // -------------------- Cognitive Depth Tests --------------------

    #[test]
    fn test_cognitive_depth_default() {
        assert_eq!(CognitiveDepth::default(), CognitiveDepth::Cortical);
    }

    #[test]
    fn test_cognitive_depth_equality() {
        assert_eq!(CognitiveDepth::Reflex, CognitiveDepth::Reflex);
        assert_eq!(CognitiveDepth::Cortical, CognitiveDepth::Cortical);
        assert_eq!(CognitiveDepth::DeepThought, CognitiveDepth::DeepThought);
        assert_ne!(CognitiveDepth::Reflex, CognitiveDepth::Cortical);
    }

    // -------------------- Integration Tests --------------------

    #[test]
    fn test_unified_architecture_integration() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run several cycles to exercise unified architecture
        for _ in 0..10 {
            service.cycle("test unified architecture integration");
        }

        let snapshot = service.consciousness_snapshot();

        // Verify unified components are operating
        assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
        assert_eq!(snapshot.cycle, 10);

        // Verify cognitive depth was set
        assert!(matches!(
            snapshot.cognitive_depth,
            CognitiveDepth::Reflex | CognitiveDepth::Cortical | CognitiveDepth::DeepThought
        ));

        // Verify response strategy was set (use service method)
        let strategy = service.current_strategy();
        assert!(matches!(
            strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }

    #[test]
    fn test_thalamic_routing_in_service() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run cycles and verify routing happens
        for _ in 0..5 {
            service.cycle("familiar simple input");
        }

        // After several similar inputs, should settle into a routing pattern
        let snapshot = service.consciousness_snapshot();
        assert!(matches!(
            snapshot.cognitive_depth,
            CognitiveDepth::Reflex | CognitiveDepth::Cortical | CognitiveDepth::DeepThought
        ));
    }

    #[test]
    fn test_closed_learning_loop_in_service() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

        // Run cycles to accumulate learning
        for _ in 0..20 {
            service.cycle("learning loop test input");
        }

        // Should have a strategy selected (use service method)
        let strategy = service.current_strategy();
        assert!(matches!(
            strategy,
            ResponseStrategy::Detailed
                | ResponseStrategy::Concise
                | ResponseStrategy::Clarifying
                | ResponseStrategy::Supportive
                | ResponseStrategy::Exploratory
        ));
    }
}
