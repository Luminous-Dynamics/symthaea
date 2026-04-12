// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Navigation state estimation for SAR helicopter.
//!
//! Fuses sensor data (GPS, IMU, barometer, compass) into a filtered
//! navigation estimate for position, velocity, and heading.

use serde::{Deserialize, Serialize};

/// Filtered navigation estimate from sensor fusion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HelicopterNavigationEstimate {
    /// Position [x, y, z] in meters (NED frame).
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz] in m/s.
    pub velocity: [f64; 3],
    /// Heading in radians (0 = North, clockwise).
    pub heading: f64,
    /// Estimated position uncertainty (meters, 1-sigma).
    pub position_uncertainty: f64,
    /// Whether GPS fix is available.
    pub gps_valid: bool,
}

/// Simple complementary-filter navigation estimator.
///
/// Blends high-frequency IMU integration with low-frequency GPS corrections.
pub struct HelicopterNavigationEstimator {
    estimate: HelicopterNavigationEstimate,
    /// Complementary filter alpha (0 = trust IMU, 1 = trust GPS).
    alpha: f64,
}

impl HelicopterNavigationEstimator {
    /// Create a new estimator with default alpha (0.02).
    pub fn new() -> Self {
        Self {
            estimate: HelicopterNavigationEstimate::default(),
            alpha: 0.02,
        }
    }

    /// Get the current filtered estimate.
    pub fn estimate(&self) -> &HelicopterNavigationEstimate {
        &self.estimate
    }

    /// Update with new sensor readings.
    ///
    /// `imu_accel`: body-frame acceleration [ax, ay, az] (m/s^2).
    /// `gps_pos`: GPS position if available [x, y, z] (NED, meters).
    /// `heading_mag`: magnetometer heading (radians).
    /// `dt`: time step (seconds).
    pub fn update(
        &mut self,
        imu_accel: [f64; 3],
        gps_pos: Option<[f64; 3]>,
        heading_mag: f64,
        dt: f64,
    ) {
        // Dead-reckoning from IMU
        for i in 0..3 {
            self.estimate.velocity[i] += imu_accel[i] * dt;
            self.estimate.position[i] += self.estimate.velocity[i] * dt;
        }

        // GPS correction (complementary filter)
        if let Some(gps) = gps_pos {
            self.estimate.gps_valid = true;
            for i in 0..3 {
                self.estimate.position[i] =
                    (1.0 - self.alpha) * self.estimate.position[i] + self.alpha * gps[i];
            }
            self.estimate.position_uncertainty *= 1.0 - self.alpha;
        } else {
            self.estimate.gps_valid = false;
            self.estimate.position_uncertainty += 0.1 * dt; // drift grows without GPS
        }

        // Heading from magnetometer
        self.estimate.heading = heading_mag;
    }

    /// Reset estimator to default state.
    pub fn reset(&mut self) {
        self.estimate = HelicopterNavigationEstimate::default();
    }
}

impl Default for HelicopterNavigationEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_estimate() {
        let est = HelicopterNavigationEstimate::default();
        assert_eq!(est.position, [0.0; 3]);
        assert!(!est.gps_valid);
    }

    #[test]
    fn test_imu_integration() {
        let mut nav = HelicopterNavigationEstimator::new();
        nav.update([1.0, 0.0, 0.0], None, 0.0, 1.0);
        assert!((nav.estimate().velocity[0] - 1.0).abs() < 1e-10);
        assert!((nav.estimate().position[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gps_correction() {
        let mut nav = HelicopterNavigationEstimator::new();
        nav.update([0.0; 3], Some([10.0, 20.0, 0.0]), 0.0, 1.0);
        assert!(nav.estimate().gps_valid);
        // Position should be pulled toward GPS
        assert!(nav.estimate().position[0] > 0.0);
    }
}
