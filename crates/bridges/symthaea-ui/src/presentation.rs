// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic presentation types for the Symthaea UI.
//!
//! This crate-local module deliberately contains only small, WASM-friendly
//! presentation semantics. It is not a mirror of the cognitive engine and
//! makes no claim that a display label is itself a scientific measurement.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresenceState {
    Disconnected,
    Available,
    Processing,
    Recovering,
    Degraded,
}

impl PresenceState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Disconnected => "disconnected",
            Self::Available => "available",
            Self::Processing => "processing",
            Self::Recovering => "recovering",
            Self::Degraded => "degraded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveMode {
    Resting,
    Exploring,
    Integrating,
    Responding,
    Uncertain,
}

impl CognitiveMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Resting => "resting",
            Self::Exploring => "exploring",
            Self::Integrating => "integrating",
            Self::Responding => "responding",
            Self::Uncertain => "uncertain",
        }
    }
}

/// Human-facing semantic state derived only from observable telemetry.
///
/// Thresholds here are intentionally conservative and descriptive. They are
/// presentation heuristics, not a new cognitive measurement or ontology.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitiveState {
    pub presence: PresenceState,
    pub mode: CognitiveMode,
    pub coherence: f64,
    pub thermodynamic_load: f64,
    pub confidence: f64,
}

impl CognitiveState {
    pub fn from_observation(
        connected: bool,
        processing: bool,
        coherence: f64,
        thermodynamic_load: f64,
        confidence: f64,
        prediction_error: f64,
    ) -> Self {
        let presence = if !connected {
            PresenceState::Disconnected
        } else if processing {
            PresenceState::Processing
        } else if coherence.is_nan()
            || thermodynamic_load.is_nan()
        {
            PresenceState::Degraded
        } else {
            PresenceState::Available
        };

        let mode = if matches!(presence, PresenceState::Disconnected | PresenceState::Degraded) {
            CognitiveMode::Uncertain
        } else if processing {
            CognitiveMode::Responding
        } else if thermodynamic_load < 0.12 {
            CognitiveMode::Resting
        } else if prediction_error > 0.5 {
            CognitiveMode::Exploring
        } else if coherence > 0.70 && thermodynamic_load > 0.45 {
            CognitiveMode::Integrating
        } else {
            CognitiveMode::Uncertain
        };

        Self {
            presence,
            mode,
            coherence: coherence.clamp(0.0, 1.0),
            thermodynamic_load: thermodynamic_load.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_state_is_explicit() {
        let state = CognitiveState::from_observation(false, false, 0.8, 0.2, 0.7, 0.1);
        assert_eq!(state.presence, PresenceState::Disconnected);
        assert_eq!(state.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn low_confidence_maps_to_exploration() {
        let state = CognitiveState::from_observation(true, false, 0.6, 0.3, 0.4, 0.2);
        assert_eq!(state.presence, PresenceState::Available);
        assert_eq!(state.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn stable_low_load_maps_to_resting() {
        let state = CognitiveState::from_observation(true, false, 0.8, 0.1, 0.05, 0.0);
        assert_eq!(state.mode, CognitiveMode::Resting);
    }

    #[test]
    fn high_prediction_error_maps_to_exploration() {
        let state = CognitiveState::from_observation(true, false, 0.6, 0.3, 0.8, 0.6);
        assert_eq!(state.mode, CognitiveMode::Exploring);
    }

    #[test]
    fn values_are_clamped_for_display() {
        let state = CognitiveState::from_observation(true, false, 2.0, 4.0, 3.0, 2.0);
        assert_eq!(state.coherence, 1.0);
        assert_eq!(state.thermodynamic_load, 1.0);
        assert_eq!(state.confidence, 1.0);
    }
}
