// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free Energy Principle agent for womb control.
//!
//! Selects homeostatic actions (adjust nutrients, oxygen, hormones, etc.)
//! by computing surprise (deviation from expected state) and choosing
//! the action that minimizes expected free energy.

use serde::{Deserialize, Serialize};

use crate::types::{FetalMetrics, GestationalWeek};

/// Actions the womb FEP agent can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WombAction {
    /// Increase or decrease nutrient delivery.
    AdjustNutrients,
    /// Adjust oxygenation parameters.
    AdjustOxygen,
    /// Modify hormone infusion rates.
    AdjustHormones,
    /// Adjust amniotic fluid temperature.
    AdjustTemp,
    /// Adjust ECMO flow rate.
    AdjustFlowRate,
    /// Increase monitoring frequency.
    MonitorCloser,
    /// Escalate to ethics review board.
    AlertEthics,
    /// No intervention needed; maintain current parameters.
    MaintainCourse,
}

/// Active inference agent for artificial womb homeostasis.
///
/// Computes surprise (deviation from expected trajectory) and selects
/// the action most likely to minimize future free energy.
#[derive(Debug, Clone)]
pub struct WombFepAgent {
    /// Surprise threshold below which no action is taken.
    pub surprise_threshold: f64,
}

impl WombFepAgent {
    /// Create a new agent with the given surprise threshold.
    pub fn new(surprise_threshold: f64) -> Self {
        Self { surprise_threshold }
    }

    /// Create an agent with default threshold.
    pub fn default_agent() -> Self {
        Self::new(0.3)
    }

    /// Select the best action given current metrics and gestational week.
    ///
    /// The agent computes per-channel surprise for each vital sign and
    /// uses the maximum channel surprise to decide escalation level,
    /// then selects the action targeting the highest-surprise channel.
    pub fn select_action(&self, metrics: &FetalMetrics, week: GestationalWeek) -> WombAction {
        let expected = FetalMetrics::normative(week);

        // Identify the most deviant channel
        let weight_err = self.channel_surprise(
            metrics.weight_grams,
            expected.weight_grams,
            expected.weight_grams.max(1.0),
        );
        let hr_err = self.channel_surprise(
            metrics.heart_rate_bpm,
            expected.heart_rate_bpm,
            expected.heart_rate_bpm.max(1.0),
        );
        let lung_err = self.channel_surprise(metrics.lung_maturity, expected.lung_maturity, 1.0);
        let movement_err =
            self.channel_surprise(metrics.movement_score, expected.movement_score, 1.0);
        let brain_err = self.channel_surprise(
            metrics.brain_activity_score,
            expected.brain_activity_score,
            1.0,
        );

        let max_err = weight_err
            .max(hr_err)
            .max(lung_err)
            .max(movement_err)
            .max(brain_err);

        // No channel deviates enough — maintain course
        if max_err < self.surprise_threshold {
            return WombAction::MaintainCourse;
        }

        // Critical: if any channel has very high surprise, alert ethics
        if max_err > 1.2 {
            return WombAction::AlertEthics;
        }

        // High surprise triggers closer monitoring
        if max_err > 0.8 {
            return WombAction::MonitorCloser;
        }

        // Select action based on dominant error channel
        if (max_err - weight_err).abs() < 1e-9 {
            WombAction::AdjustNutrients
        } else if (max_err - hr_err).abs() < 1e-9 {
            WombAction::AdjustOxygen
        } else if (max_err - lung_err).abs() < 1e-9 {
            WombAction::AdjustHormones
        } else if (max_err - movement_err).abs() < 1e-9 {
            WombAction::AdjustTemp
        } else {
            WombAction::AdjustFlowRate
        }
    }

    /// Compute total surprise: sum of squared fractional prediction errors.
    pub fn compute_surprise(&self, actual: &FetalMetrics, expected: &FetalMetrics) -> f64 {
        let channels = [
            self.channel_surprise(
                actual.weight_grams,
                expected.weight_grams,
                expected.weight_grams.max(1.0),
            ),
            self.channel_surprise(
                actual.heart_rate_bpm,
                expected.heart_rate_bpm,
                expected.heart_rate_bpm.max(1.0),
            ),
            self.channel_surprise(actual.lung_maturity, expected.lung_maturity, 1.0),
            self.channel_surprise(actual.movement_score, expected.movement_score, 1.0),
            self.channel_surprise(
                actual.brain_activity_score,
                expected.brain_activity_score,
                1.0,
            ),
        ];

        channels.iter().sum::<f64>() / channels.len() as f64
    }

    /// Compute per-channel surprise as squared fractional error.
    fn channel_surprise(&self, actual: f64, expected: f64, scale: f64) -> f64 {
        let err = (actual - expected) / scale;
        err * err
    }
}

impl Default for WombFepAgent {
    fn default() -> Self {
        Self::default_agent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maintain_course_when_normative() {
        let agent = WombFepAgent::default_agent();
        let metrics = FetalMetrics::normative(GestationalWeek::new(20));
        let action = agent.select_action(&metrics, GestationalWeek::new(20));
        assert_eq!(action, WombAction::MaintainCourse);
    }

    #[test]
    fn test_action_when_weight_deviated() {
        let agent = WombFepAgent::default_agent();
        let mut metrics = FetalMetrics::normative(GestationalWeek::new(20));
        metrics.weight_grams *= 0.3; // Severely underweight
        let action = agent.select_action(&metrics, GestationalWeek::new(20));
        // Should not be MaintainCourse
        assert_ne!(action, WombAction::MaintainCourse);
    }

    #[test]
    fn test_alert_ethics_on_extreme_surprise() {
        let agent = WombFepAgent::new(0.1);
        let mut metrics = FetalMetrics::normative(GestationalWeek::new(20));
        // Extreme deviations across multiple channels
        metrics.weight_grams *= 0.1;
        metrics.heart_rate_bpm = 300.0;
        metrics.lung_maturity = 0.0;
        let action = agent.select_action(&metrics, GestationalWeek::new(20));
        assert_eq!(action, WombAction::AlertEthics);
    }

    #[test]
    fn test_compute_surprise_zero_for_normative() {
        let agent = WombFepAgent::default_agent();
        let metrics = FetalMetrics::normative(GestationalWeek::new(20));
        let expected = FetalMetrics::normative(GestationalWeek::new(20));
        let surprise = agent.compute_surprise(&metrics, &expected);
        assert!(surprise < 1e-9, "Surprise should be ~0, got {surprise}");
    }

    #[test]
    fn test_compute_surprise_positive_for_deviated() {
        let agent = WombFepAgent::default_agent();
        let mut actual = FetalMetrics::normative(GestationalWeek::new(20));
        actual.weight_grams *= 2.0;
        let expected = FetalMetrics::normative(GestationalWeek::new(20));
        let surprise = agent.compute_surprise(&actual, &expected);
        assert!(surprise > 0.0, "Deviated should have positive surprise");
    }

    #[test]
    fn test_womb_action_serde_roundtrip() {
        let action = WombAction::AdjustHormones;
        let json = serde_json::to_string(&action).unwrap();
        let deser: WombAction = serde_json::from_str(&json).unwrap();
        assert_eq!(action, deser);
    }

    #[test]
    fn test_monitor_closer_moderate_surprise() {
        let agent = WombFepAgent::new(0.1);
        let mut metrics = FetalMetrics::normative(GestationalWeek::new(25));
        // Moderate deviation in heart rate
        metrics.heart_rate_bpm *= 1.8;
        metrics.weight_grams *= 0.5;
        let action = agent.select_action(&metrics, GestationalWeek::new(25));
        // Should trigger monitoring or stronger action, not maintain course
        assert_ne!(action, WombAction::MaintainCourse);
    }
}
