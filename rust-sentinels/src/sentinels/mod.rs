//! Consciousness state detection Sentinels
//!
//! This module contains the six neural Sentinels for consciousness detection:
//!
//! ## Consciousness Trilogy (Validated)
//! - [`EmotionSentinel`] - Proof of Joy (valence/arousal detection)
//! - [`SleepSentinel`] - Proof of Rest (sleep stage classification)
//! - [`MeditationSentinel`] - Proof of Focus (meditation depth)
//!
//! ## Extended Proofs
//! - [`AttentionSentinel`] - Proof of Attention (sustained/selective attention)
//! - [`FlowSentinel`] - Proof of Flow (optimal performance states)
//! - [`EngagementSentinel`] - Proof of Engagement (cognitive/emotional engagement)

// Consciousness Trilogy
mod emotion;
mod sleep;
mod meditation;

// Extended Proofs
mod attention;
mod flow;
mod engagement;

// Re-export Trilogy Sentinels
pub use emotion::EmotionSentinel;
pub use sleep::{SleepSentinel, SleepConfig, SpectralRatios, SpindleEvent, KComplexEvent};
pub use meditation::MeditationSentinel;

// Re-export Trilogy types from crate::types
pub use crate::types::{EmotionScore, EmotionQuadrant};
pub use crate::types::{SleepScore, SleepStage};
pub use crate::types::{MeditationScore, MeditationState};

// Re-export Extended Proofs
pub use attention::{AttentionSentinel, AttentionScore, AttentionState, AttentionConfig};
pub use flow::{FlowSentinel, FlowScore, FlowState, FlowConfig};
pub use engagement::{EngagementSentinel, EngagementScore, EngagementLevel, EngagementConfig};
