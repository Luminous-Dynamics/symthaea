// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Behavioral Synthesis — extracted from CognitiveLoopService
//!
//! Groups the 6 behavioral / social / drive subsystems that were
//! previously individual fields on `CognitiveLoopService`:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `flow_state` | `FlowState` | Flow/zone state tracking for optimal cognitive engagement |
//! | `emotion_contagion` | `EmotionContagion` | Emotional contagion tracking from content analysis |
//! | `curiosity_drive` | `CuriosityDrive` | Curiosity/exploration drive for novelty seeking |
//! | `adaptive_behavior` | `AdaptiveBehavior` | Adaptive behavior based on consciousness state |
//! | `thalamic_router` | `ThalamicRouter` | Sensory routing for cognitive depth selection |
//! | `social_mgr` | `SocialManager` | Social cognition: trust, cooperation, partner model |
//!
//! ## Design
//!
//! - **Zero behavior change**: Pure structural refactor, same field paths via delegation.
//! - **Public API preserved**: All accessor methods on `CognitiveLoopService` still work.
//! - **Field access**: Internal code accesses via `self.behavior.field_name`.

use super::social_manager::SocialManager;
use super::{AdaptiveBehavior, CuriosityDrive, EmotionContagion, FlowState, ThalamicRouter};

/// Groups 6 behavioral, drive, and social subsystems.
///
/// Extracted from `CognitiveLoopService` to reduce its field count
/// and provide a cohesive behavioral synthesis boundary.
pub(crate) struct BehavioralSynthesis {
    /// Flow state tracker.
    /// Detects and maintains flow state for optimal cognitive engagement.
    pub flow_state: FlowState,

    /// Emotion contagion tracker.
    /// Emotional content influences consciousness patterns.
    pub emotion_contagion: EmotionContagion,

    /// Curiosity drive for novelty seeking.
    /// Triggers exploration when predictions are too accurate.
    pub curiosity_drive: CuriosityDrive,

    /// Current adaptive behavior based on consciousness state.
    pub adaptive_behavior: AdaptiveBehavior,

    /// Thalamic router for cognitive depth selection.
    /// Routes inputs to Reflex/Cortical/DeepThought paths based on novelty and urgency.
    pub thalamic_router: ThalamicRouter,

    /// Social manager: social signals + phi-dyad + partner model + ring buffers.
    pub social_mgr: SocialManager,
}

impl BehavioralSynthesis {
    /// Construct from individually built components.
    ///
    /// Called from `CognitiveLoopService::new()` after each subsystem is created.
    pub fn new(
        flow_state: FlowState,
        emotion_contagion: EmotionContagion,
        curiosity_drive: CuriosityDrive,
        adaptive_behavior: AdaptiveBehavior,
        thalamic_router: ThalamicRouter,
        social_mgr: SocialManager,
    ) -> Self {
        Self {
            flow_state,
            emotion_contagion,
            curiosity_drive,
            adaptive_behavior,
            thalamic_router,
            social_mgr,
        }
    }
}
