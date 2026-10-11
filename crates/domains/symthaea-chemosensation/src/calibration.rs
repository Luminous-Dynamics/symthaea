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

    /// Whether the numeric calibration parameters are finite.
    pub fn is_valid(&self) -> bool {
        self.baseline.is_finite() && self.gain.is_finite() && self.drift.is_finite()
    }

    /// Apply baseline/gain calibration without mutating the raw observation.
    ///
    /// Invalid calibration or measurement values yield `None` rather than
    /// fabricating a plausible numeric result.
    pub fn apply(&self, raw: f32) -> Option<f32> {
        if !raw.is_finite() || !self.is_valid() {
            return None;
        }
        let calibrated = (raw - self.baseline) * self.gain;
        calibrated.is_finite().then_some(calibrated)
    }

    /// Drift normalized to [0, 1]. Invalid drift is treated as maximally
    /// uncertain rather than optimistically trusted.
    pub fn normalized_drift(&self) -> f32 {
        if self.drift.is_finite() {
            self.drift.clamp(0.0, 1.0)
        } else {
            1.0
        }
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
        let score = if self.score.is_finite() {
            self.score.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let penalty = if self.saturated || self.contaminated {
            0.5
        } else {
            1.0
        };
        score * penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_calibration_is_noop() {
        let c = CalibrationState::identity("factory-v1");
        assert!((c.apply(2.5).unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn calibration_applies_baseline_and_gain() {
        let c = CalibrationState {
            id: CalibrationId::new("session-a"),
            baseline: 1.0,
            gain: 2.0,
            drift: 0.2,
        };
        assert!((c.apply(2.5).unwrap() - 3.0).abs() < 1e-6);
        assert!((c.normalized_drift() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn non_finite_calibration_never_produces_fake_value() {
        let c = CalibrationState {
            id: CalibrationId::new("bad"),
            baseline: f32::NAN,
            gain: 1.0,
            drift: 0.0,
        };
        assert!(c.apply(2.5).is_none());
        assert!(CalibrationState::identity("ok").apply(f32::INFINITY).is_none());
    }

    #[test]
    fn invalid_drift_is_maximally_uncertain() {
        let mut c = CalibrationState::identity("bad-drift");
        c.drift = f32::NAN;
        assert_eq!(c.normalized_drift(), 1.0);
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

    #[test]
    fn non_finite_health_score_is_not_trusted() {
        let h = SensorHealth {
            score: f32::NAN,
            saturated: false,
            contaminated: false,
        };
        assert_eq!(h.confidence_factor(), 0.0);
    }
}
