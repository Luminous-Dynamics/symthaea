//! Broca language generation and voice heartbeat constants.

/// Broca coherence threshold for language generation quality gating.
pub const BROCA_COHERENT_THRESHOLD: f32 = 0.7;

/// Confidence boost scale when Broca coherence exceeds threshold.
/// Applied as: (coherence - BROCA_COHERENT_THRESHOLD) * SCALE → adjust_confidence.
/// Science: Pickering & Garrod (2013) — coherent language production reinforces cognitive confidence.
pub const BROCA_COHERENT_CONFIDENCE_SCALE: f32 = 0.1;

/// Exploration damping when Broca semantic veto fires.
/// A veto means the model rejected incoherent output → exploitation is safer than exploration.
/// Science: Pickering & Garrod (2013) — veto = self-correction → reduce drift.
pub const BROCA_VETO_EXPLORATION_SCALE: f32 = -0.02;
/// Voice heartbeat base speech rate multiplier.
/// Science: Liberman & Mattingly (1985) — synthetic proxy for vocal tract feedback.
pub const VOICE_HEARTBEAT_BASE_RATE: f32 = 4.0;

/// Voice heartbeat coarticulation smoothness weight.
pub const VOICE_HEARTBEAT_COARTICULATION_WEIGHT: f32 = 0.8;

/// Voice heartbeat listener prediction: success value.
pub const VOICE_HEARTBEAT_LISTENER_SUCCESS: f32 = 0.8;

/// Voice heartbeat listener prediction: failure value.
pub const VOICE_HEARTBEAT_LISTENER_FAIL: f32 = 0.3;
