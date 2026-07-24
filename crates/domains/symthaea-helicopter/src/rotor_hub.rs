// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order rotor-hub flapping, coning, and control-moment dynamics.
//!
//! The previous body model converted cyclic command directly into an
//! instantaneous moment. This module inserts a bounded first-order rotor-disk
//! state so cyclic authority, advancing-blade asymmetry, and thrust-dependent
//! coning are explicit and recordable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorHubError {
    InvalidConfiguration,
    NonFiniteInput,
    InvalidTimeStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorHubConfig {
    /// First-order disk-flapping response time, seconds.
    pub flapping_time_constant_s: f64,
    /// Maximum longitudinal/lateral disk flap magnitude, radians.
    pub maximum_flapping_angle_rad: f64,
    /// Steady disk flap per unit cyclic input, radians.
    pub cyclic_flapping_gain_rad: f64,
    /// Lateral flap contribution per unit advance ratio.
    pub advancing_blade_flapping_gain_rad: f64,
    /// Rotor-tip speed below which aerodynamic control authority fades.
    pub minimum_tip_speed_mps: f64,
    /// Effective hub moment per radian of disk flap, N·m/rad.
    pub hub_moment_gain_nm_per_rad: f64,
    /// Coning response time, seconds.
    pub coning_time_constant_s: f64,
    /// Coning gain from disk loading divided by tip-speed squared.
    pub coning_gain: f64,
    /// Maximum modeled coning angle, radians.
    pub maximum_coning_angle_rad: f64,
}

impl Default for RotorHubConfig {
    fn default() -> Self {
        Self {
            flapping_time_constant_s: 0.08,
            maximum_flapping_angle_rad: 0.14,
            cyclic_flapping_gain_rad: 0.10,
            advancing_blade_flapping_gain_rad: 0.025,
            minimum_tip_speed_mps: 25.0,
            hub_moment_gain_nm_per_rad: 5_000.0,
            coning_time_constant_s: 0.15,
            coning_gain: 16.0,
            maximum_coning_angle_rad: 0.12,
        }
    }
}

impl RotorHubConfig {
    pub fn validate(&self) -> bool {
        [
            self.flapping_time_constant_s,
            self.maximum_flapping_angle_rad,
            self.cyclic_flapping_gain_rad,
            self.minimum_tip_speed_mps,
            self.hub_moment_gain_nm_per_rad,
            self.coning_time_constant_s,
            self.coning_gain,
            self.maximum_coning_angle_rad,
        ]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
            && self.advancing_blade_flapping_gain_rad.is_finite()
            && self.advancing_blade_flapping_gain_rad >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorHubState {
    pub longitudinal_flap_rad: f64,
    pub lateral_flap_rad: f64,
    pub coning_angle_rad: f64,
}

impl RotorHubState {
    pub const ZERO: Self = Self {
        longitudinal_flap_rad: 0.0,
        lateral_flap_rad: 0.0,
        coning_angle_rad: 0.0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorHubOutput {
    pub longitudinal_flap_rad: f64,
    pub lateral_flap_rad: f64,
    pub coning_angle_rad: f64,
    pub roll_moment_nm: f64,
    pub pitch_moment_nm: f64,
    pub advance_ratio: f64,
    pub control_authority: f64,
}

#[derive(Debug, Clone)]
pub struct RotorHubDynamics {
    config: RotorHubConfig,
    state: RotorHubState,
}

impl RotorHubDynamics {
    pub fn new(config: RotorHubConfig) -> Result<Self, RotorHubError> {
        if !config.validate() {
            return Err(RotorHubError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            state: RotorHubState::ZERO,
        })
    }

    pub fn state(&self) -> RotorHubState {
        self.state
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        cyclic_lon: f64,
        cyclic_lat: f64,
        horizontal_airspeed_mps: f64,
        main_rpm: f64,
        rotor_radius_m: f64,
        thrust_force_n: f64,
        disk_area_m2: f64,
        dt: f64,
    ) -> Result<RotorHubOutput, RotorHubError> {
        let inputs = [
            cyclic_lon,
            cyclic_lat,
            horizontal_airspeed_mps,
            main_rpm,
            rotor_radius_m,
            thrust_force_n,
            disk_area_m2,
            dt,
        ];
        if !inputs.iter().all(|value| value.is_finite()) {
            return Err(RotorHubError::NonFiniteInput);
        }
        if dt <= 0.0 || rotor_radius_m <= 0.0 || disk_area_m2 <= 0.0 {
            return Err(RotorHubError::InvalidTimeStep);
        }

        let omega_rad_s = main_rpm.max(0.0) * std::f64::consts::TAU / 60.0;
        let tip_speed_mps = omega_rad_s * rotor_radius_m;
        let advance_ratio = horizontal_airspeed_mps.max(0.0) / tip_speed_mps.max(1.0);
        let control_authority = (tip_speed_mps / self.config.minimum_tip_speed_mps).clamp(0.0, 1.0);

        let target_longitudinal = (cyclic_lon.clamp(-1.0, 1.0)
            * self.config.cyclic_flapping_gain_rad
            * control_authority)
            .clamp(
                -self.config.maximum_flapping_angle_rad,
                self.config.maximum_flapping_angle_rad,
            );
        let target_lateral = (cyclic_lat.clamp(-1.0, 1.0) * self.config.cyclic_flapping_gain_rad
            + advance_ratio * self.config.advancing_blade_flapping_gain_rad)
            * control_authority;
        let target_lateral = target_lateral.clamp(
            -self.config.maximum_flapping_angle_rad,
            self.config.maximum_flapping_angle_rad,
        );
        let flap_alpha = 1.0 - (-dt / self.config.flapping_time_constant_s).exp();
        self.state.longitudinal_flap_rad +=
            flap_alpha * (target_longitudinal - self.state.longitudinal_flap_rad);
        self.state.lateral_flap_rad += flap_alpha * (target_lateral - self.state.lateral_flap_rad);

        let disk_loading_pa = thrust_force_n.max(0.0) / disk_area_m2;
        let target_coning = (self.config.coning_gain * disk_loading_pa
            / tip_speed_mps.powi(2).max(1.0))
        .clamp(0.0, self.config.maximum_coning_angle_rad);
        let coning_alpha = 1.0 - (-dt / self.config.coning_time_constant_s).exp();
        self.state.coning_angle_rad += coning_alpha * (target_coning - self.state.coning_angle_rad);

        Ok(RotorHubOutput {
            longitudinal_flap_rad: self.state.longitudinal_flap_rad,
            lateral_flap_rad: self.state.lateral_flap_rad,
            coning_angle_rad: self.state.coning_angle_rad,
            // Longitudinal disk flap produces pitch moment; lateral produces roll.
            roll_moment_nm: self.state.lateral_flap_rad * self.config.hub_moment_gain_nm_per_rad,
            pitch_moment_nm: self.state.longitudinal_flap_rad
                * self.config.hub_moment_gain_nm_per_rad,
            advance_ratio,
            control_authority,
        })
    }

    pub fn reset(&mut self) {
        self.state = RotorHubState::ZERO;
    }
}

impl Default for RotorHubDynamics {
    fn default() -> Self {
        Self::new(RotorHubConfig::default()).expect("default rotor-hub configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_response_is_lagged_and_bounded() {
        let mut hub = RotorHubDynamics::default();
        let first = hub
            .step(1.0, 0.0, 0.0, 3300.0, 5.3, 4905.0, 88.0, 0.01)
            .unwrap();
        assert!(first.longitudinal_flap_rad > 0.0);
        assert!(first.longitudinal_flap_rad < RotorHubConfig::default().cyclic_flapping_gain_rad);
        for _ in 0..1000 {
            hub.step(1.0, 0.0, 0.0, 3300.0, 5.3, 4905.0, 88.0, 0.01)
                .unwrap();
        }
        assert!(
            hub.state().longitudinal_flap_rad
                <= RotorHubConfig::default().maximum_flapping_angle_rad
        );
    }

    #[test]
    fn low_rpm_removes_cyclic_authority() {
        let mut hub = RotorHubDynamics::default();
        let output = hub.step(1.0, 1.0, 0.0, 0.0, 5.3, 0.0, 88.0, 0.01).unwrap();
        assert_eq!(output.control_authority, 0.0);
        assert_eq!(output.roll_moment_nm, 0.0);
        assert_eq!(output.pitch_moment_nm, 0.0);
    }

    #[test]
    fn forward_flight_exposes_advancing_blade_flap() {
        let mut hub = RotorHubDynamics::default();
        let output = hub
            .step(0.0, 0.0, 40.0, 3300.0, 5.3, 4905.0, 88.0, 0.2)
            .unwrap();
        assert!(output.advance_ratio > 0.0);
        assert!(output.lateral_flap_rad > 0.0);
    }
}
