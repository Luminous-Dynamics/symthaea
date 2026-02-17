//! Configuration types for the cognitive loop.
//!
//! Temporal backend selection (CfC vs HdcLtcUnified), training methods,
//! and the main `CognitiveLoopConfig` builder.

use crate::hdc_ltc_bridge::HdcLtcBridgeConfig;
use serde::{Deserialize, Serialize};
pub use symthaea_core::hdc::predictive_encoder::PredictiveEncoderConfig;

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
            input_dim: 256, // Must match num_neurons for train_step compatibility
            learning_rate: 0.001,
            delta_t: 0.02, // 50Hz base rate
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

    /// Enable surprise-driven exploration.
    /// When true, the cognitive loop tracks prediction surprise and triggers
    /// exploration when surprise exceeds an adaptive threshold. The exploration
    /// modulates the curiosity drive's boredom threshold to seek novel states.
    pub enable_surprise_exploration: bool,

    /// Enable prefrontal cortex executive control.
    /// When true, the cognitive loop maintains a working memory of recent inputs
    /// and uses prefrontal gating to modulate learning and exploration.
    /// High memory utilization triggers inhibition (prefrontal_veto in metadata).
    pub enable_prefrontal: bool,

    /// Enable meta-cognitive self-modeling.
    /// When true, the cognitive loop maintains a model of its own prediction error
    /// tendencies and uses self-model accuracy to modulate learning rate.
    /// High self-model accuracy → deeper recursion and faster learning.
    pub enable_meta_cognition: bool,

    /// Enable narrative self-model.
    /// When true, the cognitive loop maintains a three-level autobiographical
    /// self-model (proto/core/autobio). Each cycle's experience is processed
    /// and self-Φ is used to modulate reasoning confidence.
    pub enable_narrative_self: bool,

    /// Enable virtual body for embodied cognition.
    /// When true, the cognitive loop maintains a virtual interoceptive body
    /// that maps prediction error, flow state, curiosity, and other signals
    /// to somatic states (heart rate, breathing, fatigue, etc.). The resulting
    /// phi_modulation factor scales consciousness via somatic marker feedback.
    pub enable_virtual_body: bool,
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
            enable_surprise_exploration: false,
            enable_prefrontal: false,
            enable_meta_cognition: false,
            enable_narrative_self: false,
            enable_virtual_body: false,
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
