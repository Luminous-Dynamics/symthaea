// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extended Kalman Filter for continuous position tracking.
//!
//! State: [x, y, z, vx, vy, vz] (position + velocity in body-fixed frame).
//! Predict: constant-velocity model with configurable process noise.
//! Update: range measurements from anchors (nonlinear observation model).
//!
//! Handles intermittent measurements (mesh radio packet loss) gracefully —
//! uncertainty grows during predict-only phases but doesn't diverge.
//!
//! Important: this filter assumes conditionally independent measurement noise.
//! Unknown cross-correlated peer state estimates should be fused with
//! covariance-bounding methods such as Covariance Intersection before or
//! instead of a direct EKF update.

use serde::{Deserialize, Serialize};

/// EKF state dimension (3D position + 3D velocity).
const STATE_DIM: usize = 6;

/// Filter state at a given time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterState {
    /// State vector [x, y, z, vx, vy, vz] in meters and m/s.
    pub state: [f64; STATE_DIM],
    /// 6×6 covariance matrix (row-major) stored as Vec for serde compat.
    pub covariance: Vec<f64>,
    /// Timestamp of last update (arbitrary units, caller manages).
    pub last_update_time: f64,
    /// Number of measurement updates applied.
    pub update_count: u64,
}

/// Configuration for the position filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Process noise for position (m²/s), controls how fast uncertainty grows.
    pub position_noise: f64,
    /// Process noise for velocity (m²/s³).
    pub velocity_noise: f64,
    /// Maximum covariance diagonal before reset (m²).
    pub max_covariance: f64,
    /// Reject updates whose innovation exceeds this many sigma.
    pub innovation_gate_sigma: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            position_noise: 0.1,  // 0.1 m²/s — walking speed uncertainty
            velocity_noise: 0.01, // 0.01 m²/s³
            max_covariance: 1e8,  // 10 km max uncertainty before reset
            innovation_gate_sigma: 4.0,
        }
    }
}

/// Extended Kalman Filter for position tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionFilter {
    pub state: FilterState,
    pub config: FilterConfig,
}

impl PositionFilter {
    /// Create a new filter with an initial position guess.
    pub fn new(initial_position: [f64; 3], initial_sigma_m: f64, config: FilterConfig) -> Self {
        let mut covariance = vec![0.0_f64; STATE_DIM * STATE_DIM];
        let pos_var = initial_sigma_m * initial_sigma_m;
        let vel_var = 10.0 * 10.0; // 10 m/s initial velocity uncertainty
        covariance[0] = pos_var; // xx
        covariance[STATE_DIM + 1] = pos_var; // yy
        covariance[2 * STATE_DIM + 2] = pos_var; // zz
        covariance[3 * STATE_DIM + 3] = vel_var; // vxvx
        covariance[4 * STATE_DIM + 4] = vel_var; // vyvy
        covariance[5 * STATE_DIM + 5] = vel_var; // vzvz

        Self {
            state: FilterState {
                state: [
                    initial_position[0],
                    initial_position[1],
                    initial_position[2],
                    0.0,
                    0.0,
                    0.0, // zero initial velocity
                ],
                covariance,
                last_update_time: 0.0,
                update_count: 0,
            },
            config,
        }
    }

    /// Predict step: propagate state forward by dt seconds.
    ///
    /// Uses constant-velocity model: x(t+dt) = x(t) + v(t) * dt
    pub fn predict(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }

        // State transition: position += velocity * dt
        self.state.state[0] += self.state.state[3] * dt;
        self.state.state[1] += self.state.state[4] * dt;
        self.state.state[2] += self.state.state[5] * dt;

        // Covariance propagation: P = F*P*F^T + Q
        // F = [[I, dt*I], [0, I]]
        // For simplicity, add process noise directly to diagonal
        let q_pos = self.config.position_noise * dt;
        let q_vel = self.config.velocity_noise * dt;

        // Position uncertainty grows from velocity uncertainty
        self.state.covariance[0] += q_pos + self.state.covariance[3 * STATE_DIM + 3] * dt * dt;
        self.state.covariance[STATE_DIM + 1] +=
            q_pos + self.state.covariance[4 * STATE_DIM + 4] * dt * dt;
        self.state.covariance[2 * STATE_DIM + 2] +=
            q_pos + self.state.covariance[5 * STATE_DIM + 5] * dt * dt;
        // Velocity uncertainty grows
        self.state.covariance[3 * STATE_DIM + 3] += q_vel;
        self.state.covariance[4 * STATE_DIM + 4] += q_vel;
        self.state.covariance[5 * STATE_DIM + 5] += q_vel;

        // Cap covariance
        for i in 0..STATE_DIM {
            let idx = i * STATE_DIM + i;
            self.state.covariance[idx] = self.state.covariance[idx].min(self.config.max_covariance);
        }

        self.state.last_update_time += dt;
    }

    fn update_linear_observation(
        &mut self,
        state_index: usize,
        observed_value: f64,
        variance: f64,
    ) {
        if state_index >= STATE_DIM || variance <= 0.0 || !observed_value.is_finite() {
            return;
        }

        let innovation = observed_value - self.state.state[state_index];
        let s = self.state.covariance[state_index * STATE_DIM + state_index] + variance;

        if s.abs() < 1e-30 {
            return;
        }
        if innovation.abs() > self.config.innovation_gate_sigma * s.sqrt() {
            return;
        }

        let mut k = [0.0_f64; STATE_DIM];
        for i in 0..STATE_DIM {
            k[i] = self.state.covariance[i * STATE_DIM + state_index] / s;
        }

        for i in 0..STATE_DIM {
            self.state.state[i] += k[i] * innovation;
        }

        let mut new_cov = self.state.covariance.clone();
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                new_cov[i * STATE_DIM + j] -=
                    k[i] * self.state.covariance[state_index * STATE_DIM + j];
            }
        }
        self.state.covariance = new_cov;
        self.state.update_count += 1;
    }

    /// Update step: incorporate a range measurement from an anchor.
    ///
    /// # Arguments
    /// - `anchor`: Known anchor position [x, y, z] in meters
    /// - `measured_range`: Measured distance to anchor (meters)
    /// - `range_sigma`: Measurement uncertainty (meters)
    /// - `trust_weight`: Trust in this anchor (0-1)
    pub fn update(
        &mut self,
        anchor: &[f64; 3],
        measured_range: f64,
        range_sigma: f64,
        trust_weight: f64,
    ) {
        if measured_range <= 0.0 || range_sigma <= 0.0 {
            return;
        }

        let pos = &self.state.state[0..3];
        let dx = pos[0] - anchor[0];
        let dy = pos[1] - anchor[1];
        let dz = pos[2] - anchor[2];
        let predicted_range = (dx * dx + dy * dy + dz * dz).sqrt();

        if predicted_range < 1e-10 {
            return;
        }

        // Innovation (measurement residual)
        let innovation = measured_range - predicted_range;

        // Observation Jacobian H: d(range)/d(state) = [dx/r, dy/r, dz/r, 0, 0, 0]
        let h = [
            dx / predicted_range,
            dy / predicted_range,
            dz / predicted_range,
            0.0,
            0.0,
            0.0,
        ];

        // Effective measurement variance (trust-adjusted)
        let effective_sigma = range_sigma / trust_weight.max(0.01).sqrt();
        let r = effective_sigma * effective_sigma;

        // Innovation covariance: S = H*P*H^T + R
        let mut hp = [0.0_f64; STATE_DIM]; // H*P (1×6)
        for j in 0..STATE_DIM {
            for i in 0..STATE_DIM {
                hp[j] += h[i] * self.state.covariance[i * STATE_DIM + j];
            }
        }
        let mut s = r;
        for i in 0..STATE_DIM {
            s += h[i] * hp[i];
        }

        if s.abs() < 1e-30 {
            return;
        }
        if innovation.abs() > self.config.innovation_gate_sigma * s.sqrt() {
            return;
        }

        // Kalman gain: K = P*H^T / S (6×1)
        let mut k = [0.0_f64; STATE_DIM];
        for i in 0..STATE_DIM {
            let mut ph_i = 0.0;
            for j in 0..STATE_DIM {
                ph_i += self.state.covariance[i * STATE_DIM + j] * h[j];
            }
            k[i] = ph_i / s;
        }

        // State update: x = x + K * innovation
        for i in 0..STATE_DIM {
            self.state.state[i] += k[i] * innovation;
        }

        // Covariance update: P = (I - K*H) * P (Joseph form for stability)
        let mut new_cov = vec![0.0_f64; STATE_DIM * STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut sum = 0.0;
                for m in 0..STATE_DIM {
                    let ikm = if i == m { 1.0 } else { 0.0 } - k[i] * h[m];
                    sum += ikm * self.state.covariance[m * STATE_DIM + j];
                }
                new_cov[i * STATE_DIM + j] = sum;
            }
        }
        // Add K*R*K^T for numerical stability
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                new_cov[i * STATE_DIM + j] += k[i] * r * k[j];
            }
        }
        self.state.covariance = new_cov;

        self.state.update_count += 1;
    }

    /// Update with an absolute position fix.
    pub fn update_absolute_position(&mut self, position: [f64; 3], covariance: [f64; 9]) {
        self.update_linear_observation(0, position[0], covariance[0]);
        self.update_linear_observation(1, position[1], covariance[4]);
        self.update_linear_observation(2, position[2], covariance[8]);
    }

    /// Update with an absolute velocity fix.
    pub fn update_absolute_velocity(&mut self, velocity: [f64; 3], covariance: [f64; 9]) {
        self.update_linear_observation(3, velocity[0], covariance[0]);
        self.update_linear_observation(4, velocity[1], covariance[4]);
        self.update_linear_observation(5, velocity[2], covariance[8]);
    }

    /// Update with a direct depth observation in a down-positive frame.
    pub fn update_depth(&mut self, depth_m: f64, sigma_m: f64) {
        self.update_linear_observation(2, depth_m, sigma_m * sigma_m);
    }

    /// Update with a relative offset against a known absolute reference.
    pub fn update_relative_position(
        &mut self,
        reference_position: [f64; 3],
        relative_position: [f64; 3],
        covariance: [f64; 9],
    ) {
        let absolute = [
            reference_position[0] + relative_position[0],
            reference_position[1] + relative_position[1],
            reference_position[2] + relative_position[2],
        ];
        self.update_absolute_position(absolute, covariance);
    }

    /// Get current position estimate.
    pub fn position(&self) -> [f64; 3] {
        [
            self.state.state[0],
            self.state.state[1],
            self.state.state[2],
        ]
    }

    /// Get current velocity estimate.
    pub fn velocity(&self) -> [f64; 3] {
        [
            self.state.state[3],
            self.state.state[4],
            self.state.state[5],
        ]
    }

    /// Update with barometric altitude measurement.
    ///
    /// # Arguments
    /// - `pressure_hpa`: Current barometric pressure
    /// - `reference_hpa`: Reference pressure at sea level (default 1013.25)
    /// - `sigma_m`: Altitude measurement uncertainty (typically 1.0m absolute, 0.3m relative)
    pub fn update_barometer(&mut self, pressure_hpa: f64, reference_hpa: f64, sigma_m: f64) {
        if pressure_hpa < 100.0 || sigma_m <= 0.0 {
            return;
        }
        let measured_alt = crate::dead_reckoning::barometric_altitude(pressure_hpa, reference_hpa);

        // Observation: z-component only. H = [0, 0, 1, 0, 0, 0]
        let predicted_z = self.state.state[2];
        let innovation = measured_alt - predicted_z;

        let r = sigma_m * sigma_m;
        let p_zz = self.state.covariance[2 * STATE_DIM + 2];
        let s = p_zz + r;
        if s.abs() < 1e-30 {
            return;
        }

        // Kalman gain for z-only observation
        let mut k = [0.0_f64; STATE_DIM];
        for i in 0..STATE_DIM {
            k[i] = self.state.covariance[i * STATE_DIM + 2] / s;
        }

        // State update
        for i in 0..STATE_DIM {
            self.state.state[i] += k[i] * innovation;
        }

        // Covariance update (Joseph form for z-only)
        let mut new_cov = self.state.covariance.clone();
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                new_cov[i * STATE_DIM + j] -= k[i] * self.state.covariance[2 * STATE_DIM + j];
            }
        }
        self.state.covariance = new_cov;
        self.state.update_count += 1;
    }

    /// Get 1-sigma position uncertainty (meters).
    pub fn position_sigma(&self) -> f64 {
        let var_x = self.state.covariance[0];
        let var_y = self.state.covariance[STATE_DIM + 1];
        let var_z = self.state.covariance[2 * STATE_DIM + 2];
        (var_x + var_y + var_z).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_moves_position() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 100.0, FilterConfig::default());
        filter.state.state[3] = 1.0; // 1 m/s in x
        filter.predict(10.0);
        assert!((filter.position()[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn predict_increases_uncertainty() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 10.0, FilterConfig::default());
        let sigma_before = filter.position_sigma();
        filter.predict(60.0); // 1 minute
        assert!(filter.position_sigma() > sigma_before);
    }

    #[test]
    fn update_reduces_uncertainty() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 1000.0, FilterConfig::default());
        let sigma_before = filter.position_sigma();
        // Anchor at (100, 0, 0), range 100m — consistent with origin
        filter.update(&[100.0, 0.0, 0.0], 100.0, 1.0, 1.0);
        assert!(filter.position_sigma() < sigma_before);
    }

    #[test]
    fn convergence_with_multiple_updates() {
        let true_pos = [50.0, 30.0, -20.0];
        let anchors: [[f64; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ];

        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 200.0, FilterConfig::default());

        // 20 rounds of measurements
        for _ in 0..20 {
            filter.predict(1.0);
            for anchor in &anchors {
                let dx: f64 = true_pos[0] - anchor[0];
                let dy: f64 = true_pos[1] - anchor[1];
                let dz: f64 = true_pos[2] - anchor[2];
                let range = (dx * dx + dy * dy + dz * dz).sqrt();
                filter.update(anchor, range, 5.0, 1.0);
            }
        }

        let pos = filter.position();
        let error = ((pos[0] - true_pos[0]).powi(2)
            + (pos[1] - true_pos[1]).powi(2)
            + (pos[2] - true_pos[2]).powi(2))
        .sqrt();
        assert!(
            error < 5.0,
            "EKF should converge to within 5m, got {}m error",
            error
        );
        assert!(
            filter.position_sigma() < 50.0,
            "Sigma should shrink: {}",
            filter.position_sigma()
        );
    }

    #[test]
    fn barometer_updates_altitude() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 100.0, FilterConfig::default());
        let sigma_z_before = filter.state.covariance[2 * 6 + 2];
        // Feed barometric reading at ~100m altitude
        filter.update_barometer(1001.3, 1013.25, 1.0);
        let sigma_z_after = filter.state.covariance[2 * 6 + 2];
        assert!(
            sigma_z_after < sigma_z_before,
            "Barometer should reduce z uncertainty"
        );
        // Z state should move toward ~100m
        assert!(
            filter.position()[2].abs() > 10.0,
            "Z should update: {}",
            filter.position()[2]
        );
    }

    #[test]
    fn trust_weighting_affects_convergence() {
        let good_anchor = [0.0, 0.0, 0.0]; // correct range = 50m
        let bad_anchor = [100.0, 0.0, 0.0]; // correct range = 50m, but we'll give wrong range

        let mut filter = PositionFilter::new([25.0, 0.0, 0.0], 100.0, FilterConfig::default());

        for _ in 0..10 {
            filter.predict(1.0);
            filter.update(&good_anchor, 50.0, 1.0, 1.0); // correct, high trust
            filter.update(&bad_anchor, 80.0, 1.0, 0.05); // wrong range, low trust
        }

        let pos = filter.position();
        // Should be closer to 50 than to where bad measurement would pull it
        assert!(
            (pos[0] - 50.0).abs() < 10.0,
            "Position {} should be near 50",
            pos[0]
        );
    }

    #[test]
    fn outlier_measurement_is_gated() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 25.0, FilterConfig::default());
        let sigma_before = filter.position_sigma();
        filter.update(&[100.0, 0.0, 0.0], 10_000.0, 1.0, 1.0);
        assert!(
            (filter.position()[0]).abs() < 1.0,
            "Outlier should be ignored"
        );
        assert!(filter.position_sigma() <= sigma_before + 1e-6);
    }

    #[test]
    fn absolute_position_update_pulls_state_toward_fix() {
        let mut filter = PositionFilter::new([50.0, -20.0, 10.0], 100.0, FilterConfig::default());
        filter.update_absolute_position(
            [5.0, 2.0, -3.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let pos = filter.position();
        assert!((pos[0] - 5.0).abs() < 1.0);
        assert!((pos[1] - 2.0).abs() < 1.0);
        assert!((pos[2] + 3.0).abs() < 1.0);
    }

    #[test]
    fn absolute_velocity_update_pulls_state_toward_fix() {
        let mut filter = PositionFilter::new([0.0, 0.0, 0.0], 25.0, FilterConfig::default());
        filter.update_absolute_velocity(
            [1.5, -0.5, 0.25],
            [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
        );
        let vel = filter.velocity();
        assert!((vel[0] - 1.5).abs() < 0.5);
        assert!((vel[1] + 0.5).abs() < 0.5);
        assert!((vel[2] - 0.25).abs() < 0.5);
    }

    #[test]
    fn depth_update_reduces_vertical_error() {
        let mut filter = PositionFilter::new([0.0, 0.0, 40.0], 50.0, FilterConfig::default());
        filter.update_depth(12.0, 0.5);
        assert!((filter.position()[2] - 12.0).abs() < 5.0);
    }
}
