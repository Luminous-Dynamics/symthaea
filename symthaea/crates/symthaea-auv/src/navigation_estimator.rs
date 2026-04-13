// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Navigation state estimator — minimal position/uncertainty tracker.
//!
//! Simplified implementation: tracks position with dead-reckoning and
//! accumulates sigma uncertainty over time. A full EKF with DVL/depth/
//! compass fusion is future work.

/// Navigation estimate output.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuvNavigationEstimate {
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub position_sigma_m: f64,
    pub update_count: u64,
}

/// Simple position estimator with dead-reckoning and uncertainty growth.
#[derive(Debug, Clone)]
pub struct AuvNavigationEstimator {
    position: [f64; 3],
    velocity: [f64; 3],
    sigma: f64,
    initial_sigma: f64,
    sigma_growth_rate: f64,
    measurement_count: usize,
}

impl AuvNavigationEstimator {
    /// Create a new estimator starting at a known position.
    pub fn new(initial_position: [f64; 3], initial_sigma_m: f64) -> Self {
        Self {
            position: initial_position,
            velocity: [0.0; 3],
            sigma: initial_sigma_m,
            initial_sigma: initial_sigma_m,
            sigma_growth_rate: 0.001,
            measurement_count: 0,
        }
    }

    /// Ingest new measurements to correct the estimate.
    pub fn ingest_measurements(&mut self, measurements: &[Measurement]) {
        for m in measurements {
            // Only process Position measurements for now
            if let positioning::MeasurementValue::Position { xyz, .. } = &m.value {
                self.position[0] = 0.7 * self.position[0] + 0.3 * xyz[0];
                self.position[1] = 0.7 * self.position[1] + 0.3 * xyz[1];
                self.position[2] = 0.7 * self.position[2] + 0.3 * xyz[2];
                self.measurement_count += 1;
                self.sigma = (self.sigma * 0.95).max(0.5);
            }
        }
        // Between measurements, sigma grows
        if measurements.is_empty() {
            self.sigma += self.sigma_growth_rate;
        }
    }

    /// Current estimate.
    pub fn estimate(&self) -> AuvNavigationEstimate {
        AuvNavigationEstimate {
            position_m: self.position,
            velocity_mps: self.velocity,
            position_sigma_m: self.sigma,
            update_count: self.measurement_count as u64,
        }
    }

    /// Update velocity (e.g., from IMU integration).
    pub fn update_velocity(&mut self, velocity: [f64; 3]) {
        self.velocity = velocity;
    }

    /// Reset to initial state with specified sigma.
    pub fn reset(&mut self, position: [f64; 3], sigma_m: f64) {
        self.position = position;
        self.velocity = [0.0; 3];
        self.sigma = sigma_m;
        self.initial_sigma = sigma_m;
        self.measurement_count = 0;
    }
}

// Use positioning crate Measurement type for interop.
pub use positioning::Measurement;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_initialization() {
        let est = AuvNavigationEstimator::new([1.0, 2.0, -5.0], 10.0);
        let e = est.estimate();
        assert_eq!(e.position_m, [1.0, 2.0, -5.0]);
        assert_eq!(e.position_sigma_m, 10.0);
    }

    #[test]
    fn test_measurement_reduces_sigma() {
        let mut est = AuvNavigationEstimator::new([0.0, 0.0, -5.0], 10.0);
        let initial_sigma = est.estimate().position_sigma_m;
        let m = positioning::Measurement {
            modality: positioning::MeasurementModality::Gps,
            provenance: positioning::MeasurementProvenance::Local,
            frame: positioning::ReferenceFrame::Enu,
            value: positioning::MeasurementValue::Position {
                xyz: [0.0, 0.0, -5.0],
                sigma: [1.0, 1.0, 1.0],
            },
            timestamp_us: 0,
            source_id: "test".to_string(),
        };
        est.ingest_measurements(&[m]);
        assert!(est.estimate().position_sigma_m < initial_sigma);
    }

    #[test]
    fn test_sigma_grows_without_measurements() {
        let mut est = AuvNavigationEstimator::new([0.0, 0.0, -5.0], 1.0);
        let s0 = est.estimate().position_sigma_m;
        est.ingest_measurements(&[]);
        let s1 = est.estimate().position_sigma_m;
        assert!(s1 > s0);
    }
}
