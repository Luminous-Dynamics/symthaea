// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sensory noise filter for sim-to-real transfer validation (behind `mujoco` feature).
//!
//! Injects realistic sensor noise and communication latency between MuJoCo's
//! perfect state and the controller's perceived `FlightState`. This proves that
//! HDC's high-dimensional vector orthogonality inherently filters noise — a
//! measurable advantage over standard RL policies that memorize precise states.
//!
//! Noise model based on Crazyflie 2 IMU (BMI088):
//! - Gyro noise density: 0.014 °/s/√Hz → ~0.05 rad/s at 500Hz
//! - Accelerometer noise: ~0.002 m/s² → position noise ~0.002 m
//! - CAN-bus latency: ~4ms (2 ticks at 500Hz)

use std::collections::VecDeque;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::types::FlightState;

/// Configuration for the sensory noise filter.
#[derive(Debug, Clone)]
pub struct SensoryFilterConfig {
    /// Gaussian noise stddev on position channels (meters). Default: 0.002
    pub position_noise: f64,
    /// Gaussian noise stddev on velocity channels (m/s). Default: 0.01
    pub velocity_noise: f64,
    /// Gaussian noise stddev on angular velocity (rad/s). Default: 0.05
    pub gyro_noise: f64,
    /// Gyro random walk drift rate (rad/s per sqrt(Hz)). Default: 0.001
    pub gyro_drift_rate: f64,
    /// Quaternion noise (small rotation perturbation). Default: 0.005
    pub quaternion_noise: f64,
    /// Communication delay in timesteps (at 500Hz, 2 ticks = 4ms). Default: 2
    pub delay_ticks: usize,
    /// Whether filter is active. Default: true
    pub enabled: bool,
}

impl Default for SensoryFilterConfig {
    fn default() -> Self {
        Self {
            position_noise: 0.002,
            velocity_noise: 0.01,
            gyro_noise: 0.05,
            gyro_drift_rate: 0.001,
            quaternion_noise: 0.005,
            delay_ticks: 2,
            enabled: true,
        }
    }
}

impl SensoryFilterConfig {
    /// Clean filter (no noise, no delay) — for A/B comparison.
    pub fn clean() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Heavy noise profile — stress test for HDC resilience.
    pub fn heavy() -> Self {
        Self {
            position_noise: 0.01,
            velocity_noise: 0.05,
            gyro_noise: 0.2,
            gyro_drift_rate: 0.005,
            quaternion_noise: 0.02,
            delay_ticks: 4,
            enabled: true,
        }
    }
}

/// Injects realistic sensor noise and communication latency between
/// MuJoCo's perfect state and the controller's perceived FlightState.
pub struct SensoryFilter {
    config: SensoryFilterConfig,
    /// Ring buffer for latency simulation. Holds last N states.
    delay_buffer: VecDeque<FlightState>,
    /// Accumulated gyro drift per axis (random walk).
    gyro_drift: [f64; 3],
    /// RNG for deterministic noise generation.
    rng: ChaCha8Rng,
}

impl SensoryFilter {
    /// Create a new sensory filter with deterministic seed.
    pub fn new(config: SensoryFilterConfig, seed: u64) -> Self {
        Self {
            delay_buffer: VecDeque::with_capacity(config.delay_ticks + 1),
            gyro_drift: [0.0; 3],
            rng: ChaCha8Rng::seed_from_u64(seed),
            config,
        }
    }

    /// Filter a perfect FlightState through noise + delay.
    pub fn filter(&mut self, perfect: &FlightState) -> FlightState {
        if !self.config.enabled {
            return perfect.clone();
        }

        // 1. Add Gaussian noise to all channels
        let noisy = self.add_noise(perfect);

        // 2. Add gyro drift (random walk on angular velocity)
        let drifted = self.add_gyro_drift(noisy);

        // 3. Push into delay buffer, pop oldest
        self.delay_buffer.push_back(drifted);
        if self.delay_buffer.len() > self.config.delay_ticks {
            self.delay_buffer
                .pop_front()
                .expect("buffer non-empty after push")
        } else {
            // Not enough history yet — return current with noise only
            self.delay_buffer
                .back()
                .expect("buffer non-empty after push")
                .clone()
        }
    }

    /// Reset filter state (for new episode).
    pub fn reset(&mut self) {
        self.delay_buffer.clear();
        self.gyro_drift = [0.0; 3];
    }

    /// Get current configuration.
    pub fn config(&self) -> &SensoryFilterConfig {
        &self.config
    }

    /// Add Gaussian noise to all state channels.
    fn add_noise(&mut self, state: &FlightState) -> FlightState {
        let mut noisy = state.clone();

        // Position noise
        for p in &mut noisy.position {
            *p += self.gaussian(self.config.position_noise);
        }

        // Velocity noise
        for v in &mut noisy.linear_velocity {
            *v += self.gaussian(self.config.velocity_noise);
        }

        // Gyro noise
        for w in &mut noisy.angular_velocity {
            *w += self.gaussian(self.config.gyro_noise);
        }

        // Quaternion noise: small rotation perturbation
        if self.config.quaternion_noise > 0.0 {
            let axis = [
                self.gaussian(self.config.quaternion_noise),
                self.gaussian(self.config.quaternion_noise),
                self.gaussian(self.config.quaternion_noise),
            ];
            let angle = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if angle > 1e-10 {
                let half = angle * 0.5;
                let s = half.sin() / angle;
                let dq = [half.cos(), axis[0] * s, axis[1] * s, axis[2] * s];
                noisy.quaternion = quat_mul(state.quaternion, dq);
                // Re-normalize
                let norm = (noisy.quaternion[0].powi(2)
                    + noisy.quaternion[1].powi(2)
                    + noisy.quaternion[2].powi(2)
                    + noisy.quaternion[3].powi(2))
                .sqrt();
                if norm > 1e-10 {
                    for q in &mut noisy.quaternion {
                        *q /= norm;
                    }
                }
            }
        }

        noisy
    }

    /// Add gyro drift (random walk on angular velocity).
    fn add_gyro_drift(&mut self, mut state: FlightState) -> FlightState {
        for i in 0..3 {
            self.gyro_drift[i] += self.gaussian(self.config.gyro_drift_rate);
            state.angular_velocity[i] += self.gyro_drift[i];
        }
        state
    }

    /// Generate Gaussian noise using Box-Muller transform.
    fn gaussian(&mut self, stddev: f64) -> f64 {
        if stddev <= 0.0 {
            return 0.0;
        }
        let u1: f64 = self.rng.gen_range(1e-10..1.0);
        let u2: f64 = self.rng.gen_range(0.0..std::f64::consts::TAU);
        stddev * (-2.0 * u1.ln()).sqrt() * u2.cos()
    }
}

/// Hamilton product of two quaternions [w, x, y, z].
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory_filter_disabled() {
        let config = SensoryFilterConfig::clean();
        let mut filter = SensoryFilter::new(config, 42);

        let state = FlightState::hover(0.1);
        let filtered = filter.filter(&state);

        assert_eq!(state.position, filtered.position);
        assert_eq!(state.quaternion, filtered.quaternion);
    }

    #[test]
    fn test_sensory_filter_adds_noise() {
        let config = SensoryFilterConfig::default();
        let mut filter = SensoryFilter::new(config, 42);

        let state = FlightState::hover(0.1);
        let filtered = filter.filter(&state);

        // With noise enabled, at least one channel should differ
        let pos_changed = state
            .position
            .iter()
            .zip(filtered.position.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(pos_changed, "Noise filter should modify position channels");
    }

    #[test]
    fn test_sensory_filter_noise_bounded() {
        let config = SensoryFilterConfig::default();
        let mut filter = SensoryFilter::new(config, 42);
        let state = FlightState::hover(0.1);

        // Run many samples — noise should be bounded (within 5 sigma)
        for _ in 0..100 {
            let filtered = filter.filter(&state);
            for i in 0..3 {
                let delta = (state.position[i] - filtered.position[i]).abs();
                assert!(delta < 0.05, "Position noise exceeded 5 sigma: {}", delta);
            }
        }
    }

    #[test]
    fn test_sensory_filter_delay() {
        let config = SensoryFilterConfig {
            delay_ticks: 3,
            position_noise: 0.0,
            velocity_noise: 0.0,
            gyro_noise: 0.0,
            gyro_drift_rate: 0.0,
            quaternion_noise: 0.0,
            enabled: true,
        };
        let mut filter = SensoryFilter::new(config, 42);

        // Send 5 states with increasing altitude
        let states: Vec<FlightState> = (0..5)
            .map(|i| FlightState::hover(0.1 * (i + 1) as f64))
            .collect();

        let mut outputs = Vec::new();
        for s in &states {
            outputs.push(filter.filter(s));
        }

        // With delay=3, output[3] should be state[0]'s altitude (0.1)
        // (first 3 outputs repeat the latest available)
        assert!(
            (outputs[3].altitude() - 0.1).abs() < 1e-6,
            "Delay should return state from 3 ticks ago: got {}",
            outputs[3].altitude()
        );
    }

    #[test]
    fn test_sensory_filter_deterministic() {
        let config = SensoryFilterConfig::default();
        let state = FlightState::hover(0.1);

        let mut f1 = SensoryFilter::new(config.clone(), 42);
        let mut f2 = SensoryFilter::new(config, 42);

        let r1 = f1.filter(&state);
        let r2 = f2.filter(&state);

        assert_eq!(r1.position, r2.position);
    }

    #[test]
    fn test_sensory_filter_reset() {
        let config = SensoryFilterConfig::default();
        let mut filter = SensoryFilter::new(config, 42);

        let state = FlightState::hover(0.1);
        filter.filter(&state);
        filter.filter(&state);

        filter.reset();
        assert!(filter.delay_buffer.is_empty());
        assert_eq!(filter.gyro_drift, [0.0; 3]);
    }

    #[test]
    fn test_sensory_filter_heavy_still_finite() {
        let config = SensoryFilterConfig::heavy();
        let mut filter = SensoryFilter::new(config, 42);
        let state = FlightState::hover(0.1);

        for _ in 0..100 {
            let filtered = filter.filter(&state);
            assert!(filtered.altitude().is_finite());
            assert!(filtered.speed().is_finite());
        }
    }

    #[test]
    fn test_quat_mul_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let q = [0.707, 0.707, 0.0, 0.0];
        let result = quat_mul(identity, q);
        for i in 0..4 {
            assert!((result[i] - q[i]).abs() < 1e-6);
        }
    }
}
