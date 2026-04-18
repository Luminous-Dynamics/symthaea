// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flight navigation state estimator — minimal position/uncertainty tracker.

/// Navigation estimate for flight platforms.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlightNavigationEstimate {
    pub position_m: [f64; 3],
    pub velocity_m_per_s: [f64; 3],
    pub altitude_m: f64,
    pub position_sigma_m: f64,
}

/// Simple flight navigation estimator.
#[derive(Debug, Clone)]
pub struct FlightNavigationEstimator {
    position: [f64; 3],
    velocity: [f64; 3],
    sigma: f64,
}

impl FlightNavigationEstimator {
    pub fn new(initial_position: [f64; 3], initial_sigma_m: f64) -> Self {
        Self {
            position: initial_position,
            velocity: [0.0; 3],
            sigma: initial_sigma_m,
        }
    }

    pub fn update(&mut self, position: [f64; 3], velocity: [f64; 3]) {
        self.position = position;
        self.velocity = velocity;
    }

    pub fn estimate(&self) -> FlightNavigationEstimate {
        FlightNavigationEstimate {
            position_m: self.position,
            velocity_m_per_s: self.velocity,
            altitude_m: self.position[2],
            position_sigma_m: self.sigma,
        }
    }
}

impl Default for FlightNavigationEstimator {
    fn default() -> Self {
        Self::new([0.0; 3], 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flight_nav_estimator() {
        let mut est = FlightNavigationEstimator::new([1.0, 2.0, 5.0], 0.5);
        est.update([1.1, 2.1, 5.1], [0.1, 0.1, 0.0]);
        let e = est.estimate();
        assert_eq!(e.altitude_m, 5.1);
        assert_eq!(e.position_sigma_m, 0.5);
    }
}
