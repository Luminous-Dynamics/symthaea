// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-residual sensor reliability and observation precision.
//!
//! Finite values are not necessarily trustworthy. A biased, stuck, or drifting
//! sensor can remain inside its physical range while disagreeing persistently
//! with the command-conditioned plant model. This monitor converts per-channel
//! normalized residuals into bounded reliability estimates and an explicit FEP
//! observation precision.

use crate::encoder::normalized_channels;
use crate::types::{
    BATTERY_RATIO, CUTTER_TEMP_C, GAS_RISK, LOCALIZATION_CONFIDENCE, MOTOR_TEMP_C,
    NUM_STATE_CHANNELS, ROOF_STABILITY, SEAL_INTEGRITY, StateIntegrityReport, SubterraneanState,
    WATER_INGRESS_RATIO,
};
use serde::{Deserialize, Serialize};

pub const CRITICAL_SENSOR_CHANNELS: [usize; 9] = [
    CUTTER_TEMP_C,
    MOTOR_TEMP_C,
    BATTERY_RATIO,
    WATER_INGRESS_RATIO,
    GAS_RISK,
    ROOF_STABILITY,
    LOCALIZATION_CONFIDENCE,
    SEAL_INTEGRITY,
    crate::types::ABORT_RECOMMENDATION,
];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationQualityReport {
    pub aggregate_precision: f64,
    pub minimum_reliability: f64,
    pub maximum_residual: f64,
    pub degraded_channels: usize,
    pub critical_degraded_channels: usize,
}

impl ObservationQualityReport {
    pub const fn nominal() -> Self {
        Self {
            aggregate_precision: 1.0,
            minimum_reliability: 1.0,
            maximum_residual: 0.0,
            degraded_channels: 0,
            critical_degraded_channels: 0,
        }
    }

    pub fn requires_fail_closed(self) -> bool {
        self.critical_degraded_channels > 0 || self.aggregate_precision < 0.35
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelReliabilityMonitor {
    reliability: [f64; NUM_STATE_CHANNELS],
    residual_ewma: [f64; NUM_STATE_CHANNELS],
    learning_rate: f64,
    degraded_threshold: f64,
    last_report: ObservationQualityReport,
}

impl ChannelReliabilityMonitor {
    pub fn new(learning_rate: f64, degraded_threshold: f64) -> Self {
        Self {
            reliability: [1.0; NUM_STATE_CHANNELS],
            residual_ewma: [0.0; NUM_STATE_CHANNELS],
            learning_rate: if learning_rate.is_finite() {
                learning_rate.clamp(0.001, 1.0)
            } else {
                0.08
            },
            degraded_threshold: if degraded_threshold.is_finite() {
                degraded_threshold.clamp(0.05, 0.95)
            } else {
                0.55
            },
            last_report: ObservationQualityReport::nominal(),
        }
    }

    pub fn update(
        &mut self,
        predicted: &SubterraneanState,
        observed: &SubterraneanState,
    ) -> ObservationQualityReport {
        let predicted = normalized_channels(predicted);
        let observed = normalized_channels(observed);
        for index in 0..NUM_STATE_CHANNELS {
            let residual = (predicted[index] - observed[index]).abs().clamp(0.0, 1.0);
            self.residual_ewma[index] +=
                (residual - self.residual_ewma[index]) * self.learning_rate;
            let target_reliability = (1.0 - self.residual_ewma[index] * 2.4).clamp(0.0, 1.0);
            self.reliability[index] +=
                (target_reliability - self.reliability[index]) * self.learning_rate;
        }
        self.last_report = self.summarize();
        self.last_report
    }

    pub fn penalize_integrity_fault(
        &mut self,
        report: StateIntegrityReport,
    ) -> ObservationQualityReport {
        if report.invalid_count == 0 {
            return self.last_report;
        }
        if let Some(index) = report.first_invalid_channel {
            if index < NUM_STATE_CHANNELS {
                self.reliability[index] = 0.0;
                self.residual_ewma[index] = 1.0;
            }
        }
        if report.invalid_count > 1 {
            let global_penalty =
                (report.invalid_count as f64 / NUM_STATE_CHANNELS as f64).clamp(0.0, 1.0);
            for reliability in &mut self.reliability {
                *reliability = (*reliability - global_penalty * 0.5).clamp(0.0, 1.0);
            }
        }
        self.last_report = self.summarize();
        self.last_report
    }

    fn summarize(&self) -> ObservationQualityReport {
        let minimum_reliability = self.reliability.iter().copied().fold(1.0, f64::min);
        let maximum_residual = self.residual_ewma.iter().copied().fold(0.0, f64::max);
        let degraded_channels = self
            .reliability
            .iter()
            .filter(|reliability| **reliability < self.degraded_threshold)
            .count();
        let critical_degraded_channels = CRITICAL_SENSOR_CHANNELS
            .iter()
            .filter(|index| self.reliability[**index] < self.degraded_threshold)
            .count();
        let mean_reliability =
            self.reliability.iter().copied().sum::<f64>() / NUM_STATE_CHANNELS as f64;
        let critical_minimum = CRITICAL_SENSOR_CHANNELS
            .iter()
            .map(|index| self.reliability[*index])
            .fold(1.0, f64::min);
        let aggregate_precision =
            (mean_reliability * 0.65 + critical_minimum * 0.35).clamp(0.05, 1.0);
        ObservationQualityReport {
            aggregate_precision,
            minimum_reliability,
            maximum_residual,
            degraded_channels,
            critical_degraded_channels,
        }
    }

    pub fn reliability(&self, channel: usize) -> Option<f64> {
        self.reliability.get(channel).copied()
    }

    pub fn report(&self) -> ObservationQualityReport {
        self.last_report
    }

    pub fn reset(&mut self) {
        self.reliability = [1.0; NUM_STATE_CHANNELS];
        self.residual_ewma = [0.0; NUM_STATE_CHANNELS];
        self.last_report = ObservationQualityReport::nominal();
    }
}

impl Default for ChannelReliabilityMonitor {
    fn default() -> Self {
        Self::new(0.08, 0.55)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_critical_bias_reduces_precision_and_fails_closed() {
        let predicted = SubterraneanState::home();
        let mut observed = predicted.clone();
        observed.channels[CUTTER_TEMP_C] = 150.0;
        let mut monitor = ChannelReliabilityMonitor::new(0.2, 0.55);
        let mut report = ObservationQualityReport::nominal();
        for _ in 0..30 {
            report = monitor.update(&predicted, &observed);
        }
        assert!(report.aggregate_precision < 1.0);
        assert!(report.critical_degraded_channels > 0);
        assert!(report.requires_fail_closed());
    }

    #[test]
    fn matching_prediction_and_observation_remain_high_precision() {
        let state = SubterraneanState::home();
        let mut monitor = ChannelReliabilityMonitor::default();
        for _ in 0..50 {
            let report = monitor.update(&state, &state);
            assert!(report.aggregate_precision > 0.99);
        }
    }

    #[test]
    fn nonfinite_integrity_fault_immediately_penalizes_reported_channel() {
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = f64::NAN;
        let mut monitor = ChannelReliabilityMonitor::default();
        let report = monitor.penalize_integrity_fault(state.integrity_report());
        assert!(report.critical_degraded_channels > 0);
        assert_eq!(monitor.reliability(GAS_RISK), Some(0.0));
    }
}
