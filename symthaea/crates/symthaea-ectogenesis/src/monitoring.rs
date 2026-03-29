// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fetal monitoring, growth percentile tracking, and HDC-based anomaly detection.
//!
//! Maintains a history of fetal measurements and provides population-referenced
//! growth analysis plus holographic anomaly scoring.

use crate::gestational_encoder::{
    anomaly_score as hdc_anomaly, encode_fetal_state, encode_normative_trajectory,
};
use crate::types::FetalMetrics;
#[cfg(test)]
use crate::types::GestationalWeek;

/// Fetal monitoring system with growth tracking and anomaly detection.
#[derive(Debug, Clone)]
pub struct FetalMonitor {
    /// History of recorded fetal metrics.
    pub history: Vec<FetalMetrics>,
}

impl FetalMonitor {
    /// Create a new empty monitor.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    /// Record a new set of fetal metrics.
    pub fn record(&mut self, metrics: FetalMetrics) {
        self.history.push(metrics);
    }

    /// Compute a growth percentile based on the latest weight vs. expected.
    ///
    /// Returns a value ~0.5 for on-track, <0.1 for severely restricted,
    /// >0.9 for large-for-gestational-age.
    pub fn growth_percentile(&self) -> f64 {
        if let Some(latest) = self.history.last() {
            let expected = FetalMetrics::expected_weight(latest.week);
            if expected < 1e-6 {
                return 0.5;
            }
            let ratio = latest.weight_grams / expected;
            // Map ratio to approximate percentile via sigmoid
            // ratio 1.0 -> 0.5, ratio 0.5 -> ~0.1, ratio 1.5 -> ~0.9
            let x = (ratio - 1.0) * 5.0;
            1.0 / (1.0 + (-x).exp())
        } else {
            0.5
        }
    }

    /// Whether the fetus is growth-restricted (below 10th percentile).
    pub fn is_growth_restricted(&self) -> bool {
        self.growth_percentile() < 0.1
    }

    /// Whether the current heart rate is within normal range for gestational age.
    pub fn heart_rate_normal(&self) -> bool {
        if let Some(latest) = self.history.last() {
            let expected = FetalMetrics::expected_heart_rate(latest.week);
            if expected < 1.0 {
                // Pre-heartbeat phase
                return latest.heart_rate_bpm < 10.0;
            }
            let ratio = latest.heart_rate_bpm / expected;
            // Normal if within 20% of expected
            (0.8..=1.2).contains(&ratio)
        } else {
            true
        }
    }

    /// Compute HDC anomaly score for the latest measurement.
    ///
    /// Compares the holographic encoding of the actual state against
    /// the normative trajectory. Lower is better (0 = perfect match).
    pub fn anomaly_score(&self) -> f64 {
        if let Some(latest) = self.history.last() {
            let actual_hv = encode_fetal_state(latest);
            let normative_hv = encode_normative_trajectory(latest.week);
            hdc_anomaly(&actual_hv, &normative_hv)
        } else {
            0.0
        }
    }

    /// Compute growth velocity (grams per week) over the last N recordings.
    pub fn growth_velocity(&self, window: usize) -> f64 {
        if self.history.len() < 2 || window < 2 {
            return 0.0;
        }
        let start = self.history.len().saturating_sub(window);
        let first = &self.history[start];
        let Some(last) = self.history.last() else {
            return 0.0;
        };
        let weeks_elapsed = last.week.0 as f64 - first.week.0 as f64;
        if weeks_elapsed < 0.5 {
            return 0.0;
        }
        (last.weight_grams - first.weight_grams) / weeks_elapsed
    }
}

impl Default for FetalMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normative_metrics(week: u32) -> FetalMetrics {
        FetalMetrics::normative(GestationalWeek::new(week))
    }

    #[test]
    fn test_empty_monitor_defaults() {
        let m = FetalMonitor::new();
        assert_eq!(m.growth_percentile(), 0.5);
        assert!(!m.is_growth_restricted());
        assert!(m.heart_rate_normal());
        assert_eq!(m.anomaly_score(), 0.0);
    }

    #[test]
    fn test_normative_growth_percentile() {
        let mut m = FetalMonitor::new();
        m.record(normative_metrics(20));
        let pct = m.growth_percentile();
        assert!(
            (pct - 0.5).abs() < 0.15,
            "Normative should be near 50th percentile, got {pct}"
        );
    }

    #[test]
    fn test_restricted_growth() {
        let mut m = FetalMonitor::new();
        let mut metrics = normative_metrics(30);
        metrics.weight_grams *= 0.3; // Severely underweight
        m.record(metrics);
        assert!(m.is_growth_restricted(), "Should detect growth restriction");
    }

    #[test]
    fn test_heart_rate_normal_for_normative() {
        let mut m = FetalMonitor::new();
        m.record(normative_metrics(20));
        assert!(
            m.heart_rate_normal(),
            "Normative heart rate should be normal"
        );
    }

    #[test]
    fn test_heart_rate_abnormal_tachycardia() {
        let mut m = FetalMonitor::new();
        let mut metrics = normative_metrics(20);
        metrics.heart_rate_bpm = 250.0; // Severe tachycardia
        m.record(metrics);
        assert!(!m.heart_rate_normal(), "Tachycardia should be abnormal");
    }

    #[test]
    fn test_anomaly_score_low_for_normative() {
        let mut m = FetalMonitor::new();
        m.record(normative_metrics(20));
        let score = m.anomaly_score();
        // Normative vs normative should be very low
        assert!(
            score < 0.1,
            "Normative should have low anomaly score, got {score}"
        );
    }

    #[test]
    fn test_growth_velocity() {
        let mut m = FetalMonitor::new();
        m.record(normative_metrics(10));
        m.record(normative_metrics(20));
        let vel = m.growth_velocity(2);
        assert!(vel > 0.0, "Growth velocity should be positive");
    }

    #[test]
    fn test_growth_velocity_empty() {
        let m = FetalMonitor::new();
        assert_eq!(m.growth_velocity(5), 0.0);
    }

    #[test]
    fn test_monitor_record_history() {
        let mut m = FetalMonitor::new();
        for w in (10..=30).step_by(5) {
            m.record(normative_metrics(w));
        }
        assert_eq!(m.history.len(), 5);
    }
}
