// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sensor-confidence metadata for exoskeleton control and evidence.
//!
//! `symthaea-sensors` already models noisy readings. This module adds a small,
//! deterministic trust surface around those readings so downstream controllers
//! can distinguish fresh/calibrated data from stale, biased, saturated, or
//! missing observations without reading hidden simulator truth.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorConfidencePolicy {
    /// Sample age at which confidence reaches zero if no newer reading arrives.
    pub max_staleness_ticks: u64,
    /// Calibration age at which the calibration-age contribution reaches zero.
    pub max_calibration_age_ticks: u64,
    /// Absolute bias estimate corresponding to zero bias-confidence.
    pub max_bias_abs: f64,
    /// Multiplicative confidence retained for a saturated-but-present sample.
    pub saturation_multiplier: f64,
}

impl Default for SensorConfidencePolicy {
    fn default() -> Self {
        Self {
            max_staleness_ticks: 20,
            max_calibration_age_ticks: 100_000,
            max_bias_abs: 0.25,
            saturation_multiplier: 0.25,
        }
    }
}

impl SensorConfidencePolicy {
    pub fn is_valid(&self) -> bool {
        self.max_staleness_ticks > 0
            && self.max_calibration_age_ticks > 0
            && self.max_bias_abs.is_finite()
            && self.max_bias_abs > 0.0
            && self.saturation_multiplier.is_finite()
            && (0.0..=1.0).contains(&self.saturation_multiplier)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorEvidence {
    /// Stable device/channel label in the owning integration layer.
    pub sensor_id: String,
    /// Tick represented by this sample.
    pub sample_tick: u64,
    /// Tick of the calibration record currently applied to this channel.
    pub last_calibration_tick: u64,
    /// Nominal one-sigma measurement uncertainty in the sensor's native unit.
    pub nominal_sigma: f64,
    /// Current absolute bias estimate in the sensor's native unit.
    pub estimated_bias_abs: f64,
    /// Sensor reports that the sample hit its measurable range boundary.
    pub saturated: bool,
    /// Sensor did not return a usable observation at this sample tick.
    pub dropout: bool,
}

impl SensorEvidence {
    pub fn is_valid(&self) -> bool {
        !self.sensor_id.is_empty()
            && self.nominal_sigma.is_finite()
            && self.nominal_sigma >= 0.0
            && self.estimated_bias_abs.is_finite()
            && self.estimated_bias_abs >= 0.0
            && self.last_calibration_tick <= self.sample_tick
    }

    /// Deterministic bounded confidence at `current_tick`.
    ///
    /// A dropout is never silently promoted into evidence. Saturation retains only
    /// the configured bounded fraction because the sign/direction may still be
    /// informative while the magnitude is not.
    pub fn confidence(&self, current_tick: u64, policy: SensorConfidencePolicy) -> f64 {
        assert!(self.is_valid(), "SensorEvidence must be valid");
        assert!(policy.is_valid(), "SensorConfidencePolicy must be valid");
        if self.dropout || current_tick < self.sample_tick {
            return 0.0;
        }

        let sample_age = current_tick - self.sample_tick;
        let staleness = 1.0
            - (sample_age as f64 / policy.max_staleness_ticks as f64).clamp(0.0, 1.0);
        let calibration_age = self.sample_tick - self.last_calibration_tick;
        let calibration = 1.0
            - (calibration_age as f64 / policy.max_calibration_age_ticks as f64)
                .clamp(0.0, 1.0);
        let bias = 1.0 - (self.estimated_bias_abs / policy.max_bias_abs).clamp(0.0, 1.0);
        let saturation = if self.saturated {
            policy.saturation_multiplier
        } else {
            1.0
        };
        (staleness * calibration * bias * saturation).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorDisposition {
    Trusted,
    Degraded,
    Unavailable,
}

pub fn disposition(confidence: f64) -> SensorDisposition {
    assert!(confidence.is_finite() && (0.0..=1.0).contains(&confidence));
    if confidence == 0.0 {
        SensorDisposition::Unavailable
    } else if confidence >= 0.75 {
        SensorDisposition::Trusted
    } else {
        SensorDisposition::Degraded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> SensorEvidence {
        SensorEvidence {
            sensor_id: "left-knee-encoder".into(),
            sample_tick: 100,
            last_calibration_tick: 90,
            nominal_sigma: 0.01,
            estimated_bias_abs: 0.01,
            saturated: false,
            dropout: false,
        }
    }

    #[test]
    fn fresh_sample_is_more_trusted_than_stale_sample() {
        let sample = evidence();
        let policy = SensorConfidencePolicy::default();
        assert!(sample.confidence(100, policy) > sample.confidence(115, policy));
        assert_eq!(sample.confidence(120, policy), 0.0);
    }

    #[test]
    fn dropout_is_never_evidence() {
        let mut sample = evidence();
        sample.dropout = true;
        assert_eq!(sample.confidence(100, SensorConfidencePolicy::default()), 0.0);
        assert_eq!(
            disposition(sample.confidence(100, SensorConfidencePolicy::default())),
            SensorDisposition::Unavailable
        );
    }

    #[test]
    fn saturation_degrades_without_erasing_present_sample() {
        let mut sample = evidence();
        sample.saturated = true;
        let confidence = sample.confidence(100, SensorConfidencePolicy::default());
        assert!(confidence > 0.0 && confidence < 0.75);
        assert_eq!(disposition(confidence), SensorDisposition::Degraded);
    }

    #[test]
    fn bias_and_calibration_age_reduce_confidence() {
        let good = evidence();
        let mut degraded = evidence();
        degraded.estimated_bias_abs = 0.20;
        degraded.last_calibration_tick = 0;
        let policy = SensorConfidencePolicy::default();
        assert!(good.confidence(100, policy) > degraded.confidence(100, policy));
    }

    #[test]
    fn same_metadata_produces_same_confidence() {
        let sample = evidence();
        let policy = SensorConfidencePolicy::default();
        assert_eq!(sample.confidence(107, policy), sample.confidence(107, policy));
    }
}
