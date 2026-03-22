// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Consistency Checks
//!
//! Cross-references wall-clock elapsed time against CfC temporal dynamics to
//! detect clock manipulation. The CfC neurons operate on `delta_t` (default 0.02s
//! = 50Hz), modulated by ~8 factors. If the wall clock duration of a cycle is
//! wildly inconsistent with the configured delta_t, something may be wrong.
//!
//! This catches:
//! - System clock manipulation (time jumps forward/backward)
//! - Process suspension attacks (freeze-then-resume with stale CfC state)
//! - Severe resource contention (cycles taking 100x longer than expected)

use std::collections::VecDeque;
use std::time::Duration;

/// Monitors temporal consistency between wall clock and CfC dynamics.
pub struct TemporalConsistencyMonitor {
    /// History of (wall_clock_duration, cfc_delta_t) pairs for drift detection.
    history: VecDeque<(Duration, f32)>,
    /// Maximum history length.
    max_history: usize,
    /// Minimum acceptable cycle duration (catches impossibly fast cycles).
    pub min_cycle_duration: Duration,
    /// Maximum acceptable cycle duration (catches frozen/resumed cycles).
    max_cycle_duration: Duration,
    /// Consecutive anomaly count.
    anomaly_streak: u32,
    /// Alert threshold for consecutive anomalies.
    anomaly_alert_threshold: u32,
}

impl TemporalConsistencyMonitor {
    /// Create a new monitor with default thresholds.
    ///
    /// Defaults:
    /// - min cycle: 100us (faster than any realistic cycle)
    /// - max cycle: 2s (50Hz target = 20ms; 2s = 100x slower)
    /// - alert after 5 consecutive anomalies
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(100),
            max_history: 100,
            min_cycle_duration: Duration::from_micros(100),
            max_cycle_duration: Duration::from_secs(2),
            anomaly_streak: 0,
            anomaly_alert_threshold: 5,
        }
    }

    /// Check a single cycle's timing for anomalies.
    ///
    /// `wall_elapsed`: actual wall-clock time for this cycle.
    /// `cfc_delta_t`: the CfC delta_t used this cycle (may be modulated).
    ///
    /// Returns `Some(description)` if an anomaly is detected, `None` if clean.
    pub fn check(&mut self, wall_elapsed: Duration, cfc_delta_t: f32) -> Option<String> {
        // Record in history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back((wall_elapsed, cfc_delta_t));

        // Check 1: Cycle duration bounds
        if wall_elapsed < self.min_cycle_duration {
            self.anomaly_streak += 1;
            if self.anomaly_streak >= self.anomaly_alert_threshold {
                return Some(format!(
                    "Cycle completed in {:?} (min {:?}). {} consecutive anomalies. \
                     Possible clock manipulation or impossibly fast execution.",
                    wall_elapsed, self.min_cycle_duration, self.anomaly_streak
                ));
            }
            return None; // Below alert threshold, just count
        }

        if wall_elapsed > self.max_cycle_duration {
            self.anomaly_streak += 1;
            if self.anomaly_streak >= self.anomaly_alert_threshold {
                return Some(format!(
                    "Cycle took {:?} (max {:?}). {} consecutive anomalies. \
                     Possible process suspension or severe resource contention.",
                    wall_elapsed, self.max_cycle_duration, self.anomaly_streak
                ));
            }
            return None;
        }

        // Check 2: Wall time vs CfC delta_t ratio consistency
        // The CfC delta_t is ~0.02s base, modulated up to ~10x.
        // Wall time should correlate loosely — we allow 200x ratio tolerance
        // to avoid false positives from modulation variance.
        if cfc_delta_t > 0.0 {
            let wall_secs = wall_elapsed.as_secs_f32();
            let ratio = wall_secs / cfc_delta_t;
            // Extreme ratios suggest clock manipulation
            if ratio > 200.0 || ratio < 0.005 {
                self.anomaly_streak += 1;
                if self.anomaly_streak >= self.anomaly_alert_threshold {
                    return Some(format!(
                        "Wall/CfC ratio {ratio:.1} (wall={wall_secs:.4}s, cfc_dt={cfc_delta_t:.4}). \
                         {} consecutive anomalies. Possible clock skew.",
                        self.anomaly_streak
                    ));
                }
                return None;
            }
        }

        // All checks passed — reset streak
        self.anomaly_streak = 0;
        None
    }

    /// Get the current anomaly streak count.
    pub fn anomaly_streak(&self) -> u32 {
        self.anomaly_streak
    }

    /// Get the history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Compute the mean wall-clock cycle duration over the history window.
    pub fn mean_cycle_duration(&self) -> Option<Duration> {
        if self.history.is_empty() {
            return None;
        }
        let total: Duration = self.history.iter().map(|(d, _)| *d).sum();
        Some(total / self.history.len() as u32)
    }
}

impl Default for TemporalConsistencyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cycle_passes() {
        let mut mon = TemporalConsistencyMonitor::new();
        let result = mon.check(Duration::from_millis(20), 0.02);
        assert!(result.is_none());
        assert_eq!(mon.anomaly_streak(), 0);
    }

    #[test]
    fn test_too_fast_cycle_increments_streak() {
        let mut mon = TemporalConsistencyMonitor::new();
        // Below min duration
        let result = mon.check(Duration::from_micros(10), 0.02);
        assert!(result.is_none()); // Below alert threshold
        assert_eq!(mon.anomaly_streak(), 1);
    }

    #[test]
    fn test_consecutive_anomalies_trigger_alert() {
        let mut mon = TemporalConsistencyMonitor::new();
        mon.anomaly_alert_threshold = 3;
        // 3 consecutive too-fast cycles
        assert!(mon.check(Duration::from_micros(10), 0.02).is_none());
        assert!(mon.check(Duration::from_micros(10), 0.02).is_none());
        let result = mon.check(Duration::from_micros(10), 0.02);
        assert!(result.is_some());
        assert!(result.unwrap().contains("clock manipulation"));
    }

    #[test]
    fn test_normal_cycle_resets_streak() {
        let mut mon = TemporalConsistencyMonitor::new();
        mon.check(Duration::from_micros(10), 0.02); // anomaly
        assert_eq!(mon.anomaly_streak(), 1);
        mon.check(Duration::from_millis(20), 0.02); // normal
        assert_eq!(mon.anomaly_streak(), 0);
    }

    #[test]
    fn test_too_slow_cycle() {
        let mut mon = TemporalConsistencyMonitor::new();
        mon.anomaly_alert_threshold = 1;
        let result = mon.check(Duration::from_secs(5), 0.02);
        assert!(result.is_some());
        assert!(result.unwrap().contains("suspension"));
    }

    #[test]
    fn test_extreme_ratio_detected() {
        let mut mon = TemporalConsistencyMonitor::new();
        mon.anomaly_alert_threshold = 1;
        // Wall time 1s but CfC delta_t 0.001 → ratio = 1000
        let result = mon.check(Duration::from_secs(1), 0.001);
        assert!(result.is_some());
        assert!(result.unwrap().contains("clock skew"));
    }

    #[test]
    fn test_mean_cycle_duration() {
        let mut mon = TemporalConsistencyMonitor::new();
        mon.check(Duration::from_millis(10), 0.02);
        mon.check(Duration::from_millis(30), 0.02);
        let mean = mon.mean_cycle_duration().unwrap();
        assert!((mean.as_millis() as i64 - 20).abs() <= 1);
    }

    #[test]
    fn test_zero_delta_t_doesnt_crash() {
        let mut mon = TemporalConsistencyMonitor::new();
        let result = mon.check(Duration::from_millis(20), 0.0);
        assert!(result.is_none()); // Skips ratio check when delta_t is 0
    }
}
