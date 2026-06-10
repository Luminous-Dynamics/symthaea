// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Space navigation: orbital position estimation without GPS.

use serde::{Deserialize, Serialize};

/// Space navigation position estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceNavigationEstimate {
    /// Position in ECI frame [x, y, z] (km).
    pub position_eci_km: [f64; 3],
    /// Velocity in ECI frame [vx, vy, vz] (km/s).
    pub velocity_eci_km_s: [f64; 3],
    /// Position uncertainty 1-sigma (km).
    pub position_sigma_km: f64,
    /// Epoch (Julian date).
    pub epoch_jd: f64,
}

impl SpaceNavigationEstimate {
    pub fn from_state(pos: [f64; 3], vel: [f64; 3], sigma: f64, epoch: f64) -> Self {
        Self {
            position_eci_km: pos,
            velocity_eci_km_s: vel,
            position_sigma_km: sigma,
            epoch_jd: epoch,
        }
    }
    pub fn altitude_km(&self) -> f64 {
        let r = (self.position_eci_km.iter().map(|p| p * p).sum::<f64>()).sqrt();
        r - 6371.0 // Earth radius
    }
}

/// Space navigation estimator using two-body propagation + measurement updates.
pub struct SpaceNavigationEstimator {
    state: SpaceNavigationEstimate,
    mu: f64, // Gravitational parameter (km^3/s^2)
}

impl SpaceNavigationEstimator {
    /// Create for Earth orbit.
    pub fn earth_orbit() -> Self {
        Self {
            state: SpaceNavigationEstimate::from_state(
                [6771.0, 0.0, 0.0],
                [0.0, 7.67, 0.0],
                1.0,
                0.0,
            ),
            mu: 398600.4418,
        }
    }

    /// Propagate state forward by dt seconds using two-body dynamics.
    pub fn propagate(&mut self, dt_s: f64) {
        let r = (self
            .state
            .position_eci_km
            .iter()
            .map(|p| p * p)
            .sum::<f64>())
        .sqrt();
        if r < 1.0 {
            return;
        } // Prevent singularity
        let a = -self.mu / (r * r * r);
        for i in 0..3 {
            self.state.velocity_eci_km_s[i] += a * self.state.position_eci_km[i] * dt_s;
            self.state.position_eci_km[i] += self.state.velocity_eci_km_s[i] * dt_s;
        }
        self.state.position_sigma_km += 0.001 * dt_s; // Uncertainty grows
    }

    /// Get current estimate.
    pub fn estimate(&self) -> &SpaceNavigationEstimate {
        &self.state
    }

    /// Reset to default LEO orbit.
    pub fn reset(&mut self) {
        *self = Self::earth_orbit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_propagate_stays_finite() {
        let mut nav = SpaceNavigationEstimator::earth_orbit();
        for _ in 0..1000 {
            nav.propagate(1.0);
        }
        let alt = nav.estimate().altitude_km();
        assert!(alt.is_finite(), "Altitude diverged: {alt}");
        assert!(
            alt > 100.0 && alt < 2000.0,
            "LEO orbit should stay in range: {alt}"
        );
    }
}
