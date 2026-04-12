// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wind and atmospheric model for SAR helicopter simulation.
//!
//! Models three wind phenomena critical for SAR operations:
//! - **Steady wind**: constant velocity vector (e.g., prevailing wind)
//! - **Gusts**: stochastic bursts with Dryden turbulence spectrum
//! - **Ground effect**: reduced power requirement near terrain
//!
//! Science:
//! - Dryden wind turbulence model (MIL-F-8785C, 1980)
//! - Ground effect: thrust augmentation ratio ≈ 1 / (1 - (R/4z)²)
//!   where R = rotor radius, z = altitude (Cheeseman & Bennett, 1955)

use serde::{Deserialize, Serialize};

/// Wind model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindConfig {
    /// Steady wind velocity [x, y, z] in m/s (world frame).
    pub steady_wind: [f64; 3],
    /// Gust intensity (standard deviation of velocity perturbation, m/s).
    pub gust_intensity: f64,
    /// Gust bandwidth (1/time constant, Hz). Higher = faster gusts.
    pub gust_bandwidth: f64,
    /// Enable ground effect modeling.
    pub enable_ground_effect: bool,
    /// Main rotor radius (meters). Used for ground effect calculation.
    pub rotor_radius: f64,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self {
            steady_wind: [0.0; 3],
            gust_intensity: 0.0,
            gust_bandwidth: 1.0,
            enable_ground_effect: true,
            rotor_radius: 5.3, // ~R44 class rotor radius
        }
    }
}

impl WindConfig {
    /// Light wind conditions (5 m/s from west, light gusts).
    pub fn light_wind() -> Self {
        Self {
            steady_wind: [5.0, 0.0, 0.0],
            gust_intensity: 1.0,
            gust_bandwidth: 0.5,
            ..Self::default()
        }
    }

    /// Moderate wind conditions (10 m/s, moderate gusts).
    pub fn moderate_wind() -> Self {
        Self {
            steady_wind: [8.0, 4.0, 0.0],
            gust_intensity: 3.0,
            gust_bandwidth: 1.0,
            ..Self::default()
        }
    }

    /// Severe wind conditions (20 m/s, strong gusts — near operational limit).
    pub fn severe_wind() -> Self {
        Self {
            steady_wind: [15.0, 10.0, -2.0],
            gust_intensity: 6.0,
            gust_bandwidth: 2.0,
            ..Self::default()
        }
    }
}

/// Wind model state (evolves each physics step).
#[derive(Debug, Clone)]
pub struct WindModel {
    config: WindConfig,
    /// Current gust velocity (filtered noise).
    gust_state: [f64; 3],
    /// Simple PRNG state for deterministic gusts.
    rng_state: u64,
}

impl WindModel {
    /// Create a new wind model from config.
    pub fn new(config: WindConfig) -> Self {
        Self {
            config,
            gust_state: [0.0; 3],
            rng_state: 12345,
        }
    }

    /// Create with default (no wind).
    pub fn calm() -> Self {
        Self::new(WindConfig::default())
    }

    /// Create with a specific seed for deterministic replay.
    pub fn with_seed(config: WindConfig, seed: u64) -> Self {
        Self {
            config,
            gust_state: [0.0; 3],
            rng_state: seed,
        }
    }

    /// Step the wind model forward by dt seconds.
    ///
    /// Returns the total wind force vector [Fx, Fy, Fz] in Newtons,
    /// given the current airspeed and a drag area (Cd × A).
    pub fn step(&mut self, dt: f64, airspeed: [f64; 3], drag_area: f64) -> WindForce {
        // 1. Update gust state (first-order Markov, Dryden-inspired)
        let alpha = (-self.config.gust_bandwidth * dt).exp();
        let noise_scale = self.config.gust_intensity * (1.0 - alpha * alpha).sqrt();

        for i in 0..3 {
            let noise = self.gaussian_noise() * noise_scale;
            self.gust_state[i] = alpha * self.gust_state[i] + noise;
        }

        // 2. Total wind velocity = steady + gust
        let wind_vel = [
            self.config.steady_wind[0] + self.gust_state[0],
            self.config.steady_wind[1] + self.gust_state[1],
            self.config.steady_wind[2] + self.gust_state[2],
        ];

        // 3. Relative airspeed (vehicle speed - wind speed)
        let relative = [
            airspeed[0] - wind_vel[0],
            airspeed[1] - wind_vel[1],
            airspeed[2] - wind_vel[2],
        ];

        // 4. Aerodynamic drag force = -0.5 × ρ × Cd×A × |v_rel| × v_rel
        let rho = 1.225; // kg/m³ at sea level
        let speed_sq =
            relative[0] * relative[0] + relative[1] * relative[1] + relative[2] * relative[2];
        let speed = speed_sq.sqrt();
        let drag_factor = -0.5 * rho * drag_area * speed;

        let force = [
            drag_factor * relative[0],
            drag_factor * relative[1],
            drag_factor * relative[2],
        ];

        WindForce {
            force,
            wind_velocity: wind_vel,
            gust_velocity: self.gust_state,
        }
    }

    /// Compute ground effect thrust augmentation ratio.
    ///
    /// At low altitudes, the rotor downwash is reflected off the ground,
    /// increasing effective thrust. The ratio approaches 1.0 at high altitude
    /// and can reach ~1.3 at one rotor radius above ground.
    ///
    /// Science: Cheeseman & Bennett (1955) — T_ge/T_oge = 1 / (1 - (R/4z)²)
    pub fn ground_effect_ratio(&self, altitude: f64) -> f64 {
        if !self.config.enable_ground_effect || altitude <= 0.0 {
            return 1.0;
        }

        let r = self.config.rotor_radius;
        let ratio_sq = (r / (4.0 * altitude)).powi(2);

        if ratio_sq >= 1.0 {
            // Very close to ground — cap at reasonable maximum
            1.3
        } else {
            1.0 / (1.0 - ratio_sq)
        }
    }

    /// Reset gust state (for new episodes).
    pub fn reset(&mut self) {
        self.gust_state = [0.0; 3];
    }

    /// Simple xorshift64 PRNG for deterministic gust generation.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate a Gaussian-distributed random number (Box-Muller).
    fn gaussian_noise(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1 = u1.max(1e-15); // Prevent log(0)
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Output from one wind model step.
#[derive(Debug, Clone, Copy)]
pub struct WindForce {
    /// Aerodynamic force vector [Fx, Fy, Fz] in Newtons (world frame).
    pub force: [f64; 3],
    /// Total wind velocity [vx, vy, vz] in m/s (steady + gust).
    pub wind_velocity: [f64; 3],
    /// Current gust component [vx, vy, vz] in m/s.
    pub gust_velocity: [f64; 3],
}

impl WindForce {
    /// Wind speed magnitude (m/s).
    pub fn wind_speed(&self) -> f64 {
        let [vx, vy, vz] = self.wind_velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    /// Force magnitude (Newtons).
    pub fn force_magnitude(&self) -> f64 {
        let [fx, fy, fz] = self.force;
        (fx * fx + fy * fy + fz * fz).sqrt()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calm_wind_no_force() {
        let mut wind = WindModel::calm();
        let force = wind.step(0.01, [0.0; 3], 2.0);
        assert_eq!(force.force, [0.0; 3]);
        assert_eq!(force.wind_velocity, [0.0; 3]);
    }

    #[test]
    fn test_steady_wind_produces_drag() {
        let config = WindConfig {
            steady_wind: [10.0, 0.0, 0.0],
            gust_intensity: 0.0,
            ..Default::default()
        };
        let mut wind = WindModel::new(config);
        // Vehicle stationary, wind blowing → drag force pushes vehicle
        let force = wind.step(0.01, [0.0; 3], 2.0);
        // Force should be in +x direction (wind pushes vehicle with the wind)
        // Actually: relative airspeed = 0 - 10 = -10, drag = -0.5*rho*CdA*|v|*v
        // drag_x = -0.5 * 1.225 * 2.0 * 10.0 * (-10.0) = +12.25 N
        assert!(
            force.force[0] > 0.0,
            "Wind should push vehicle: Fx={}",
            force.force[0]
        );
    }

    #[test]
    fn test_gusts_are_deterministic() {
        let config = WindConfig {
            gust_intensity: 5.0,
            gust_bandwidth: 1.0,
            ..Default::default()
        };
        let mut w1 = WindModel::with_seed(config.clone(), 42);
        let mut w2 = WindModel::with_seed(config, 42);

        let f1 = w1.step(0.01, [0.0; 3], 2.0);
        let f2 = w2.step(0.01, [0.0; 3], 2.0);

        assert_eq!(f1.gust_velocity, f2.gust_velocity);
    }

    #[test]
    fn test_gusts_are_stochastic() {
        let config = WindConfig {
            gust_intensity: 5.0,
            gust_bandwidth: 1.0,
            ..Default::default()
        };
        let mut wind = WindModel::with_seed(config, 42);

        let f1 = wind.step(0.01, [0.0; 3], 2.0);
        let f2 = wind.step(0.01, [0.0; 3], 2.0);

        // Consecutive gusts should differ (not constant)
        let changed = f1.gust_velocity[0] != f2.gust_velocity[0]
            || f1.gust_velocity[1] != f2.gust_velocity[1];
        assert!(changed, "Gusts should evolve over time");
    }

    #[test]
    fn test_ground_effect_increases_near_ground() {
        let wind = WindModel::calm();
        let ratio_high = wind.ground_effect_ratio(100.0); // High altitude
        let ratio_low = wind.ground_effect_ratio(5.0); // One rotor radius

        assert!(
            (ratio_high - 1.0).abs() < 0.01,
            "High altitude: no ground effect"
        );
        assert!(
            ratio_low > 1.0,
            "Low altitude: ground effect should augment thrust"
        );
        assert!(ratio_low > ratio_high);
    }

    #[test]
    fn test_ground_effect_capped() {
        let wind = WindModel::calm();
        let ratio = wind.ground_effect_ratio(0.5); // Very close to ground
        assert!(
            ratio <= 1.3,
            "Ground effect should be capped: ratio={ratio}"
        );
    }

    #[test]
    fn test_ground_effect_disabled() {
        let config = WindConfig {
            enable_ground_effect: false,
            ..Default::default()
        };
        let wind = WindModel::new(config);
        assert_eq!(wind.ground_effect_ratio(1.0), 1.0);
    }

    #[test]
    fn test_wind_presets() {
        let light = WindConfig::light_wind();
        assert!(light.steady_wind[0] > 0.0);
        assert!(light.gust_intensity > 0.0);

        let moderate = WindConfig::moderate_wind();
        assert!(moderate.gust_intensity > light.gust_intensity);

        let severe = WindConfig::severe_wind();
        assert!(severe.gust_intensity > moderate.gust_intensity);
    }

    #[test]
    fn test_reset_clears_gust_state() {
        let config = WindConfig {
            gust_intensity: 5.0,
            ..Default::default()
        };
        let mut wind = WindModel::new(config);
        wind.step(0.01, [0.0; 3], 2.0);
        assert!(wind.gust_state.iter().any(|&v| v != 0.0));
        wind.reset();
        assert_eq!(wind.gust_state, [0.0; 3]);
    }

    #[test]
    fn test_wind_force_magnitude() {
        let config = WindConfig::moderate_wind();
        let mut wind = WindModel::new(config);
        let force = wind.step(0.01, [0.0; 3], 2.0);
        assert!(force.wind_speed() > 0.0);
        assert!(force.force_magnitude() > 0.0);
    }
}
