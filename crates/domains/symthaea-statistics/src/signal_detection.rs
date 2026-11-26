// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binary signal-detection metrics.
//!
//! The raw 2x2 counts are preserved alongside a finite-sample loglinear
//! correction for Gaussian equal-variance signal-detection estimates. This
//! separates sensitivity (d-prime) from response criterion (c); hit rate or
//! percent-correct alone cannot make that distinction.
//!
//! The equal-variance Gaussian interpretation of d-prime is an explicit model
//! assumption. Report raw counts/rates and do not interpret d-prime as a
//! physical transmission measurement or as evidence for a causal mechanism.

use crate::distributions::normal_quantile;

/// Counts from a binary signal-detection task.
///
/// A "signal" is an event-present interval; a "noise" is an event-absent
/// interval. A "yes" response is a call/report that the event occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SignalDetectionCounts {
    pub hits: u64,
    pub misses: u64,
    pub false_alarms: u64,
    pub correct_rejections: u64,
}

/// Raw and corrected metrics from a binary signal-detection task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalDetectionMetrics {
    /// Number of event-present (signal) intervals.
    pub signal_trials: u64,
    /// Number of event-absent (noise) intervals.
    pub noise_trials: u64,
    /// Raw hit rate: hits / signal_trials.
    pub hit_rate: f64,
    /// Raw miss rate: misses / signal_trials.
    pub miss_rate: f64,
    /// Raw false-alarm rate: false_alarms / noise_trials.
    pub false_alarm_rate: f64,
    /// Raw correct-rejection rate: correct_rejections / noise_trials.
    pub correct_rejection_rate: f64,
    /// Loglinear-corrected hit rate: (hits + 0.5) / (signal_trials + 1).
    ///
    /// This finite-sample correction prevents infinite normal quantiles when
    /// every signal trial is a hit or every signal trial is a miss.
    pub corrected_hit_rate: f64,
    /// Loglinear-corrected false-alarm rate:
    /// (false_alarms + 0.5) / (noise_trials + 1).
    pub corrected_false_alarm_rate: f64,
    /// Equal-variance Gaussian sensitivity:
    /// z(corrected_hit_rate) - z(corrected_false_alarm_rate).
    pub d_prime: f64,
    /// Equal-variance Gaussian response criterion:
    /// -0.5 * (z(corrected_hit_rate) + z(corrected_false_alarm_rate)).
    ///
    /// Negative values indicate a liberal "signal present" response bias;
    /// positive values indicate a conservative response bias.
    pub criterion_c: f64,
}

impl SignalDetectionCounts {
    /// Total event-present trials, or None if the u64 sum overflows.
    pub fn signal_trials(self) -> Option<u64> {
        self.hits.checked_add(self.misses)
    }

    /// Total event-absent trials, or None if the u64 sum overflows.
    pub fn noise_trials(self) -> Option<u64> {
        self.false_alarms.checked_add(self.correct_rejections)
    }

    /// Calculate raw rates and loglinear-corrected d-prime / criterion.
    ///
    /// Returns None when either class has zero trials or a count total
    /// overflows. The 0.5/1 correction is applied to the d-prime/criterion
    /// estimates only; the raw observed rates remain unchanged.
    pub fn metrics(self) -> Option<SignalDetectionMetrics> {
        let signal_trials = self.signal_trials()?;
        let noise_trials = self.noise_trials()?;
        if signal_trials == 0 || noise_trials == 0 {
            return None;
        }

        let signal_n = signal_trials as f64;
        let noise_n = noise_trials as f64;
        let hit_rate = self.hits as f64 / signal_n;
        let miss_rate = self.misses as f64 / signal_n;
        let false_alarm_rate = self.false_alarms as f64 / noise_n;
        let correct_rejection_rate = self.correct_rejections as f64 / noise_n;

        let corrected_hit_rate = (self.hits as f64 + 0.5) / (signal_n + 1.0);
        let corrected_false_alarm_rate =
            (self.false_alarms as f64 + 0.5) / (noise_n + 1.0);

        // Corrected rates are strictly inside (0, 1), so both quantiles are
        // finite. Keep the correction and equal-variance assumption explicit.
        let z_hit = normal_quantile(corrected_hit_rate);
        let z_false_alarm = normal_quantile(corrected_false_alarm_rate);
        let d_prime = z_hit - z_false_alarm;
        let criterion_c = -0.5 * (z_hit + z_false_alarm);

        if !hit_rate.is_finite()
            || !miss_rate.is_finite()
            || !false_alarm_rate.is_finite()
            || !correct_rejection_rate.is_finite()
            || !corrected_hit_rate.is_finite()
            || !corrected_false_alarm_rate.is_finite()
            || !d_prime.is_finite()
            || !criterion_c.is_finite()
        {
            return None;
        }

        Some(SignalDetectionMetrics {
            signal_trials,
            noise_trials,
            hit_rate,
            miss_rate,
            false_alarm_rate,
            correct_rejection_rate,
            corrected_hit_rate,
            corrected_false_alarm_rate,
            d_prime,
            criterion_c,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_preserve_all_four_outcomes() {
        let metrics = SignalDetectionCounts {
            hits: 80,
            misses: 20,
            false_alarms: 30,
            correct_rejections: 70,
        }
        .metrics()
        .expect("both signal and noise trials are present");

        assert_eq!(metrics.signal_trials, 100);
        assert_eq!(metrics.noise_trials, 100);
        assert!((metrics.hit_rate - 0.8).abs() < 1e-12);
        assert!((metrics.miss_rate - 0.2).abs() < 1e-12);
        assert!((metrics.false_alarm_rate - 0.3).abs() < 1e-12);
        assert!((metrics.correct_rejection_rate - 0.7).abs() < 1e-12);
        assert!((metrics.hit_rate + metrics.miss_rate - 1.0).abs() < 1e-12);
        assert!(
            (metrics.false_alarm_rate + metrics.correct_rejection_rate - 1.0).abs() < 1e-12
        );
    }

    #[test]
    fn sensitivity_and_criterion_are_separate() {
        // These two observers have the same equal-variance d-prime but opposite
        // response criteria: one is liberal, the other conservative.
        let liberal = SignalDetectionCounts {
            hits: 90,
            misses: 10,
            false_alarms: 30,
            correct_rejections: 70,
        }
        .metrics()
        .unwrap();
        let conservative = SignalDetectionCounts {
            hits: 70,
            misses: 30,
            false_alarms: 10,
            correct_rejections: 90,
        }
        .metrics()
        .unwrap();

        assert!((liberal.d_prime - conservative.d_prime).abs() < 1e-10);
        assert!(liberal.criterion_c < 0.0);
        assert!(conservative.criterion_c > 0.0);
    }

    #[test]
    fn loglinear_correction_keeps_extreme_rates_finite() {
        let perfect = SignalDetectionCounts {
            hits: 100,
            misses: 0,
            false_alarms: 0,
            correct_rejections: 100,
        }
        .metrics()
        .expect("extreme counts have valid corrected rates");

        assert_eq!(perfect.hit_rate, 1.0);
        assert_eq!(perfect.false_alarm_rate, 0.0);
        assert!(perfect.corrected_hit_rate < 1.0);
        assert!(perfect.corrected_false_alarm_rate > 0.0);
        assert!(perfect.d_prime.is_finite() && perfect.d_prime > 0.0);
        assert!(perfect.criterion_c.is_finite());
    }

    #[test]
    fn zero_class_and_overflow_are_not_evaluable() {
        assert!(
            SignalDetectionCounts {
                hits: 0,
                misses: 0,
                false_alarms: 1,
                correct_rejections: 1,
            }
            .metrics()
            .is_none()
        );

        assert!(
            SignalDetectionCounts {
                hits: u64::MAX,
                misses: 1,
                false_alarms: 1,
                correct_rejections: 1,
            }
            .metrics()
            .is_none()
        );
    }

    #[test]
    fn chance_level_has_near_zero_sensitivity() {
        let metrics = SignalDetectionCounts {
            hits: 50,
            misses: 50,
            false_alarms: 50,
            correct_rejections: 50,
        }
        .metrics()
        .unwrap();

        assert!(metrics.d_prime.abs() < 1e-12);
        assert!(metrics.criterion_c.abs() < 1e-12);
    }
}
