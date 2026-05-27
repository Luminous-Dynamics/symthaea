// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
use std::fmt;

use crate::result::IntegrationReport;

/// A single measurement point in a coherence trend.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrendPoint {
    /// Step or epoch number when this measurement was taken.
    pub step: usize,
    /// Integration index at this step.
    pub integration_index: f64,
    /// Normalized integration index (phi / total_mi, clamped to [0, 1]).
    pub normalized_index: f64,
    /// Total mutual information at this step.
    pub total_mi: f64,
    /// Sizes of the two MIP partition halves.
    pub partition_sizes: (usize, usize),
}

/// Tracks integration measurements over multiple windows to detect
/// strengthening, weakening, or structural changes in system coherence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoherenceTrend {
    points: VecDeque<TrendPoint>,
    capacity: usize,
    next_step: usize,
}

impl CoherenceTrend {
    /// Create a new trend tracker with the given capacity (ring buffer).
    /// Capacity is clamped to a minimum of 2.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        Self {
            points: VecDeque::with_capacity(capacity),
            capacity,
            next_step: 0,
        }
    }

    /// Record a measurement from an [`IntegrationReport`], auto-incrementing
    /// the step counter.
    pub fn record(&mut self, report: &IntegrationReport) {
        let (ref a, ref b) = report.minimum_information_partition;
        let point = TrendPoint {
            step: self.next_step,
            integration_index: report.integration_index,
            normalized_index: report.normalized_index,
            total_mi: report.total_mutual_information,
            partition_sizes: (a.len(), b.len()),
        };
        self.next_step += 1;
        self.push_point(point);
    }

    /// Record a pre-built [`TrendPoint`] directly.
    pub fn record_point(&mut self, point: TrendPoint) {
        self.push_point(point);
    }

    /// Number of recorded points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the trend buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Access the recorded points (oldest first).
    pub fn points(&self) -> &VecDeque<TrendPoint> {
        &self.points
    }

    /// OLS regression slope of `integration_index` vs `step`.
    /// Returns `None` if fewer than 2 points.
    pub fn trend_slope(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let n = self.points.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for p in &self.points {
            let x = p.step as f64;
            let y = p.integration_index;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return Some(0.0);
        }

        Some((n * sum_xy - sum_x * sum_y) / denom)
    }

    /// Whether integration is trending upward (positive slope).
    pub fn is_strengthening(&self) -> bool {
        self.trend_slope().map_or(false, |s| s > 0.0)
    }

    /// Whether integration is trending downward (negative slope).
    pub fn is_weakening(&self) -> bool {
        self.trend_slope().map_or(false, |s| s < 0.0)
    }

    /// Whether the MIP partition structure changed between the last two
    /// measurements.
    pub fn partition_changed(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }
        let last = &self.points[self.points.len() - 1];
        let prev = &self.points[self.points.len() - 2];
        last.partition_sizes != prev.partition_sizes
    }

    fn push_point(&mut self, point: TrendPoint) {
        if self.points.len() >= self.capacity {
            self.points.pop_front();
        }
        self.points.push_back(point);
    }
}

impl fmt::Display for CoherenceTrend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.is_strengthening() {
            "STRENGTHENING"
        } else if self.is_weakening() {
            "WEAKENING"
        } else {
            "STABLE"
        };

        write!(f, "Trend: {direction}")?;

        if let Some(slope) = self.trend_slope() {
            write!(f, " (slope: {slope:+.6})")?;
        }

        if let Some(last) = self.points.back() {
            write!(f, ", latest: {:.6}", last.integration_index)?;
        }

        if self.partition_changed() {
            write!(f, " [partition changed]")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(step: usize, idx: f64) -> TrendPoint {
        let total_mi = idx * 1.5;
        TrendPoint {
            step,
            integration_index: idx,
            normalized_index: if total_mi > 1e-12 {
                (idx / total_mi).clamp(0.0, 1.0)
            } else {
                0.0
            },
            total_mi,
            partition_sizes: (3, 3),
        }
    }

    #[test]
    fn test_increasing_slope() {
        let mut trend = CoherenceTrend::new(10);
        for i in 0..5 {
            trend.record_point(make_point(i, i as f64 * 0.1));
        }
        let slope = trend.trend_slope().unwrap();
        assert!(
            slope > 0.0,
            "increasing data should have positive slope, got {slope}"
        );
        assert!(trend.is_strengthening());
        assert!(!trend.is_weakening());
    }

    #[test]
    fn test_decreasing_slope() {
        let mut trend = CoherenceTrend::new(10);
        for i in 0..5 {
            trend.record_point(make_point(i, 1.0 - i as f64 * 0.1));
        }
        let slope = trend.trend_slope().unwrap();
        assert!(
            slope < 0.0,
            "decreasing data should have negative slope, got {slope}"
        );
        assert!(trend.is_weakening());
        assert!(!trend.is_strengthening());
    }

    #[test]
    fn test_constant_slope() {
        let mut trend = CoherenceTrend::new(10);
        for i in 0..5 {
            trend.record_point(make_point(i, 0.5));
        }
        let slope = trend.trend_slope().unwrap();
        assert!(slope.abs() < 1e-10, "constant data should have zero slope");
        assert!(!trend.is_strengthening());
        assert!(!trend.is_weakening());
    }

    #[test]
    fn test_capacity_eviction() {
        let mut trend = CoherenceTrend::new(3);
        for i in 0..5 {
            trend.record_point(make_point(i, i as f64));
        }
        assert_eq!(trend.len(), 3);
        // Should contain steps 2, 3, 4
        assert_eq!(trend.points()[0].step, 2);
        assert_eq!(trend.points()[2].step, 4);
    }

    #[test]
    fn test_partition_changed() {
        let mut trend = CoherenceTrend::new(10);
        trend.record_point(TrendPoint {
            step: 0,
            integration_index: 0.5,
            normalized_index: 0.5 / 0.7,
            total_mi: 0.7,
            partition_sizes: (3, 3),
        });
        assert!(!trend.partition_changed());

        trend.record_point(TrendPoint {
            step: 1,
            integration_index: 0.6,
            normalized_index: 0.6 / 0.8,
            total_mi: 0.8,
            partition_sizes: (2, 4),
        });
        assert!(trend.partition_changed());

        trend.record_point(TrendPoint {
            step: 2,
            integration_index: 0.7,
            normalized_index: 0.7 / 0.9,
            total_mi: 0.9,
            partition_sizes: (2, 4),
        });
        assert!(!trend.partition_changed());
    }

    #[test]
    fn test_insufficient_data() {
        let trend = CoherenceTrend::new(10);
        assert!(trend.trend_slope().is_none());
        assert!(!trend.is_strengthening());
        assert!(!trend.is_weakening());
    }

    #[test]
    fn test_display() {
        let mut trend = CoherenceTrend::new(10);
        trend.record_point(make_point(0, 0.1));
        trend.record_point(make_point(1, 0.3));
        let display = format!("{trend}");
        assert!(display.contains("STRENGTHENING"), "display: {display}");
    }

    #[test]
    fn test_record_from_report() {
        let report = IntegrationReport {
            betti_numbers: Default::default(),
            persistent_cycles: Default::default(),
            integration_index: 0.42,
            total_mutual_information: 0.68,
            minimum_information_partition: (vec![0, 1], vec![2, 3]),
            spectral_order: vec![0, 1, 2, 3],
            temporal_coherence: None,
            normalized_index: (0.42 / 0.68f64).clamp(0.0, 1.0),
            variable_contributions: vec![0.1, 0.2, 0.1, 0.0],
            num_observations: 100,
        };
        let mut trend = CoherenceTrend::new(10);
        trend.record(&report);
        assert_eq!(trend.len(), 1);
        assert_eq!(trend.points()[0].step, 0);
        assert!((trend.points()[0].integration_index - 0.42).abs() < 1e-10);
        assert_eq!(trend.points()[0].partition_sizes, (2, 2));
    }
}
