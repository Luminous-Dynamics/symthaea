// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Calibration provenance and sensor-health state for chemical observations.

use serde::{Deserialize, Serialize};

/// Stable identifier for the calibration applied to an observation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CalibrationId(pub String);

impl CalibrationId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Calibration state captured alongside each chemical observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Calibration profile/version used for compensation.
    pub id: CalibrationId,
    /// Estimated baseline offset for this sensor session.
    pub baseline: f32,
    /// Estimated multiplicative gain correction.
    pub gain: f32,
    /// Estimated normalized drift magnitude in [0, 1].
    pub drift: f32,
}

impl CalibrationState {
    pub fn identity(id: impl Into<String>) -> Self {
        Self {
            id: CalibrationId::new(id),
            baseline: 0.0,
            gain: 1.0,
            drift: 0.0,
        }
    }

    /// Apply baseline/gain calibration without mutating the raw observation.
    pub fn apply(&self, raw: f32) -> f32 {
        (raw - self.baseline) * self.gain
    }

    pub fn normalized_drift(&self) -> f32 {
        self.drift.clamp(0.0, 1.0)
    }
}

/// Health metadata for the transducer that produced an observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorHealth {
    /// Overall health score in [0, 1].
    pub score: f32,
    /// True when the channel is believed to be saturated.
    pub saturated: bool,
    /// True when contamination/poisoning is suspected.
    pub contaminated: bool,
}

impl Default for SensorHealth {
    fn default() -> Self {
        Self {
            score: 1.0,
            saturated: false,
            contaminated: false,
        }
    }
}

impl SensorHealth {
    pub fn confidence_factor(&self) -> f32 {
        let penalty = if self.saturated || self.contaminated {
            0.5
        } else {
            1.0
        };
        self.score.clamp(0.0, 1.0) * penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_calibration_is_noop() {
        let c = CalibrationState::identity("factory-v1");
        assert!((c.apply(2.5) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn calibration_applies_baseline_and_gain() {
        let c = CalibrationState {
            id: CalibrationId::new("session-a"),
            baseline: 1.0,
            gain: 2.0,
            drift: 0.2,
        };
        assert!((c.apply(2.5) - 3.0).abs() < 1e-6);
        assert!((c.normalized_drift() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn unhealthy_flags_reduce_confidence() {
        let h = SensorHealth {
            score: 0.8,
            saturated: true,
            contaminated: false,
        };
        assert!((h.confidence_factor() - 0.4).abs() < 1e-6);
    }
}
