// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
mod meditation;
mod sleep;

// Extended Proofs
mod attention;
mod engagement;
mod flow;

// Re-export Trilogy Sentinels
pub use emotion::EmotionSentinel;
pub use meditation::MeditationSentinel;
pub use sleep::{KComplexEvent, SleepConfig, SleepSentinel, SpectralRatios, SpindleEvent};

// Re-export Trilogy types from crate::types
pub use crate::types::{EmotionQuadrant, EmotionScore};
pub use crate::types::{MeditationScore, MeditationState};
pub use crate::types::{SleepScore, SleepStage};

// Re-export Extended Proofs
pub use attention::{AttentionConfig, AttentionScore, AttentionSentinel, AttentionState};
pub use engagement::{EngagementConfig, EngagementLevel, EngagementScore, EngagementSentinel};
pub use flow::{FlowConfig, FlowScore, FlowSentinel, FlowState};