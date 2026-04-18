// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Waypoint + cruise controller for the demo.
//!
//! - Lateral: Stanley-style heading + cross-track correction toward the current
//!   look-ahead waypoint.
//! - Longitudinal: PI on speed error; brake when commanded throttle would be
//!   negative and speed is above target.

use symthaea_vehicle::types::{VehicleCommand, VehicleState};

pub struct StanleyController {
    pub target: [f64; 2],
    pub target_speed: f64,
    pub lateral_gain: f64,
    pub heading_gain: f64,
    pub throttle_kp: f64,
    pub max_steering: f64,
}

impl Default for StanleyController {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0],
            target_speed: 8.0,
            lateral_gain: 0.35,
            heading_gain: 1.1,
            throttle_kp: 0.25,
            max_steering: 0.9,
        }
    }
}

impl StanleyController {
    /// Compute a motor command from current state + target waypoint.
    pub fn compute(&self, state: &VehicleState) -> VehicleCommand {
        // Heading error toward target (world frame)
        let dx = self.target[0] - state.position_x;
        let dy = self.target[1] - state.position_y;
        let desired_heading = dy.atan2(dx);
        let heading_err = wrap_angle(desired_heading - state.heading);

        // Cross-track error: signed perpendicular distance to the target heading line
        let forward = [state.heading.cos(), state.heading.sin()];
        let cross = dx * (-forward[1]) + dy * forward[0];
        let v = state.speed.max(1.0);
        let cross_err = (self.lateral_gain * cross / v).atan();

        let steering_raw = self.heading_gain * heading_err + cross_err;
        let steering = steering_raw.clamp(-self.max_steering, self.max_steering) as f32;

        // Longitudinal: throttle/brake split on speed error sign
        let speed_err = self.target_speed - state.speed;
        let ctrl = self.throttle_kp * speed_err;
        let (throttle, brake) = if ctrl >= 0.0 {
            (ctrl.min(1.0) as f32, 0.0_f32)
        } else {
            (0.0_f32, (-ctrl).min(1.0) as f32)
        };

        VehicleCommand {
            steering,
            throttle,
            brake,
        }
        .clamped()
    }
}

/// Scale a command by a consciousness motor gain. Steering is attenuated toward
/// zero; throttle is attenuated toward zero; brake is UN-attenuated (braking is
/// the safe action so we don't weaken it when Φ is low).
pub fn gain_scale(cmd: VehicleCommand, gain: f64) -> VehicleCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    VehicleCommand {
        steering: g * cmd.steering,
        throttle: g * cmd.throttle,
        brake: cmd.brake,
    }
}

/// Wrap angle to `[-π, π]`.
fn wrap_angle(a: f64) -> f64 {
    let mut x = a;
    while x > std::f64::consts::PI {
        x -= std::f64::consts::TAU;
    }
    while x < -std::f64::consts::PI {
        x += std::f64::consts::TAU;
    }
    x
}
