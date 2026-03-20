//! Broca language generation and voice heartbeat constants.

/// Broca coherence threshold for language generation quality gating.
pub const BROCA_COHERENT_THRESHOLD: f64 = 0.7;

/// Confidence boost scale when Broca coherence exceeds threshold.
/// Applied as: (coherence - BROCA_COHERENT_THRESHOLD) * SCALE → adjust_confidence.
/// Science: Pickering & Garrod (2013) — coherent language production reinforces cognitive confidence.
pub const BROCA_COHERENT_CONFIDENCE_SCALE: f64 = 0.1;

/// Exploration damping when Broca semantic veto fires.
/// A veto means the model rejected incoherent output → exploitation is safer than exploration.
/// Science: Pickering & Garrod (2013) — veto = self-correction → reduce drift.
pub const BROCA_VETO_EXPLORATION_SCALE: f64 = -0.02;
/// Voice heartbeat base speech rate multiplier.
/// Science: Liberman & Mattingly (1985) — synthetic proxy for vocal tract feedback.
pub const VOICE_HEARTBEAT_BASE_RATE: f64 = 4.0;

/// Voice heartbeat coarticulation smoothness weight.
pub const VOICE_HEARTBEAT_COARTICULATION_WEIGHT: f64 = 0.8;

/// Voice heartbeat listener prediction: success value.
pub const VOICE_HEARTBEAT_LISTENER_SUCCESS: f64 = 0.8;

/// Voice heartbeat listener prediction: failure value.
pub const VOICE_HEARTBEAT_LISTENER_FAIL: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// BROCA QUALITY TUNING
// Science: Pickering & Garrod (2013) — self-monitoring in language production.
// ═══════════════════════════════════════════════════════════════════════════════

/// Weight of final coherence in Broca quality composite.
pub const BROCA_QUALITY_COHERENCE_WEIGHT: f64 = 0.4;

/// Weight of semantic PE (inverted) in Broca quality composite.
pub const BROCA_QUALITY_PE_WEIGHT: f64 = 0.4;

/// Weight of long-range coherence in Broca quality composite.
pub const BROCA_QUALITY_LONG_COHERENCE_WEIGHT: f64 = 0.2;

/// Broca quality EMA momentum (weight of prior estimate).
pub const BROCA_QUALITY_EMA_MOMENTUM: f64 = 0.85;

/// Broca quality EMA alpha (weight of new sample).
pub const BROCA_QUALITY_EMA_ALPHA: f64 = 0.15;

/// Broca quality threshold below which low-quality streak increments.
pub const BROCA_LOW_QUALITY_THRESHOLD: f64 = 0.3;

/// Consciousness threshold increase when low-quality streak reaches 3.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_INCREASE: f64 = 0.05;

/// Maximum consciousness threshold for Broca generation.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_MAX: f64 = 0.5;

/// Consciousness threshold decrease when quality EMA is high.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_DECREASE: f64 = 0.02;

/// Minimum consciousness threshold for Broca generation.
pub const BROCA_CONSCIOUSNESS_THRESHOLD_MIN: f64 = 0.1;

/// Broca quality EMA threshold above which threshold can decrease.
pub const BROCA_QUALITY_HIGH_THRESHOLD: f64 = 0.7;

/// Broca incoherent output threshold (final_coherence below this).
pub const BROCA_INCOHERENT_THRESHOLD: f64 = 0.3;

/// Confidence dampening rate for incoherent Broca output.
pub const BROCA_INCOHERENT_DAMPEN_RATE: f64 = 0.05;

/// Broca quality threshold above which LR is boosted.
pub const BROCA_QUALITY_LR_THRESHOLD: f64 = 0.6;

/// Broca quality LR boost scale.
pub const BROCA_QUALITY_LR_SCALE: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS VELOCITY / VALENCE
// Science: Cannon (1929) — homeostatic regulation; Sokolov (1963) — habituation.
// ═══════════════════════════════════════════════════════════════════════════════

/// Homeostasis pull scale for velocity toward neutral.
pub const HOMEOSTASIS_PULL_VELOCITY_SCALE: f64 = 0.05;

/// Homeostasis pull scale for arousal toward target.
pub const HOMEOSTASIS_PULL_AROUSAL_SCALE: f64 = 0.05;

/// Valence homeostasis EMA momentum.
pub const VALENCE_HOMEOSTASIS_MOMENTUM: f64 = 0.95;

/// Valence homeostasis EMA alpha.
pub const VALENCE_HOMEOSTASIS_ALPHA: f64 = 0.05;

/// Consciousness resize center (consciousness level center for resize factor).
pub const CONSCIOUSNESS_RESIZE_CENTER: f64 = 0.5;

/// Consciousness resize scale (deviation from center × this = factor offset).
pub const CONSCIOUSNESS_RESIZE_SCALE: f64 = 0.3;

/// Goal progress base step per cycle.
pub const GOAL_DELTA_BASE_STEP: f64 = 0.01;

/// Goal progress confidence scaling factor.
pub const GOAL_DELTA_CONFIDENCE_SCALE: f64 = 0.5;

/// World model error importance scale.
pub const WORLD_MODEL_ERROR_IMPORTANCE_SCALE: f64 = 0.3;
