// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic outer-loop guidance for target-conditioned flight.
//!
//! The HDC-LTC controller remains available as an adaptive residual, while
//! this module supplies the explicit position/velocity reference that mission
//! execution previously omitted.

use serde::{Deserialize, Serialize};

use crate::controller::pd_hover_baseline;
use crate::types::{HelicopterCommand, HelicopterState};

/// Desired translational and heading state in the simulator's local frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlightReference {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub yaw: f64,
}

impl FlightReference {
    pub fn hold(position: [f64; 3], yaw: f64) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            yaw,
        }
    }
}

/// Tunable gains and envelope limits for the classical guidance backbone.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GuidanceConfig {
    pub position_kp: f64,
    pub velocity_kd: f64,
    pub attitude_kp: f64,
    pub rate_kd: f64,
    pub yaw_kp: f64,
    pub yaw_rate_kd: f64,
    pub max_tilt_rad: f64,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            position_kp: 0.035,
            velocity_kd: 0.12,
            attitude_kp: 1.8,
            rate_kd: 0.55,
            yaw_kp: 0.8,
            yaw_rate_kd: 0.6,
            max_tilt_rad: 20.0_f64.to_radians(),
        }
    }
}

/// Position/velocity outer loop followed by attitude/rate stabilization.
pub fn position_hold_command(
    state: &HelicopterState,
    reference: &FlightReference,
    config: &GuidanceConfig,
) -> HelicopterCommand {
    let mut cmd = pd_hover_baseline(state, reference.position[2]);

    let ex = reference.position[0] - state.position[0];
    let ey = reference.position[1] - state.position[1];
    let evx = reference.velocity[0] - state.linear_velocity[0];
    let evy = reference.velocity[1] - state.linear_velocity[1];

    // Positive pitch tilts thrust toward +x. Positive roll tilts thrust toward
    // -y in the simulator convention, so north/+y requires negative roll.
    let desired_pitch = (config.position_kp * ex + config.velocity_kd * evx)
        .clamp(-config.max_tilt_rad, config.max_tilt_rad);
    let desired_roll = -(config.position_kp * ey + config.velocity_kd * evy)
        .clamp(-config.max_tilt_rad, config.max_tilt_rad);

    let (roll, pitch, yaw) = state.euler_angles();
    let [wx, wy, wz] = state.angular_velocity;
    cmd.cyclic_lon = (config.attitude_kp * (desired_pitch - pitch) - config.rate_kd * wy) as f32;
    cmd.cyclic_lat = (config.attitude_kp * (desired_roll - roll) - config.rate_kd * wx) as f32;

    let yaw_error = wrap_pi(reference.yaw - yaw);
    cmd.pedal = (config.yaw_kp * yaw_error - config.yaw_rate_kd * wz) as f32;
    cmd.clamped()
}

fn wrap_pi(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn east_reference_commands_positive_pitch() {
        let state = HelicopterState::hover(20.0);
        let reference = FlightReference::hold([50.0, 0.0, 20.0], 0.0);
        let cmd = position_hold_command(&state, &reference, &GuidanceConfig::default());
        assert!(cmd.cyclic_lon > 0.0);
        assert!(cmd.cyclic_lat.abs() < 1e-6);
    }

    #[test]
    fn north_reference_commands_negative_roll() {
        let state = HelicopterState::hover(20.0);
        let reference = FlightReference::hold([0.0, 50.0, 20.0], 0.0);
        let cmd = position_hold_command(&state, &reference, &GuidanceConfig::default());
        assert!(cmd.cyclic_lat < 0.0);
        assert!(cmd.cyclic_lon.abs() < 1e-6);
    }

    #[test]
    fn position_hold_preserves_altitude_trim_at_target() {
        let state = HelicopterState::hover(20.0);
        let reference = FlightReference::hold([0.0, 0.0, 20.0], 0.0);
        let cmd = position_hold_command(&state, &reference, &GuidanceConfig::default());
        assert!((cmd.collective - HelicopterCommand::HOVER_COLLECTIVE).abs() < 1e-6);
        assert!((cmd.thrust - HelicopterCommand::HOVER_THRUST).abs() < 1e-6);
    }
}
