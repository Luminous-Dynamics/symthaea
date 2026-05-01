// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight waypoint PID controller for the demo.
//!
//! We deliberately do NOT use `symthaea_multirotor::FlightController` here — that one
//! wraps the full 16,384-D HDC-LTC network + learned output head and needs training
//! to converge. For an easy-to-run demo a classical cascade controller is clearer:
//! it reliably hovers and tracks waypoints so the consciousness side-channel's
//! effect (motor-gain modulation under gusts) is legible.

use symthaea_multirotor::types::{FlightState, QuadrotorCommand};

/// Cascade PID: outer-loop position → target attitude; inner-loop attitude → moments.
pub struct WaypointController {
    pub target: [f64; 3],
    pub kp_xy: f64,
    pub kp_z: f64,
    pub kd_z: f64,
    pub kp_att: f64,
    pub kd_att: f64,
    pub yaw_kp: f64,
}

impl Default for WaypointController {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 1.5],
            kp_xy: 0.35,
            kp_z: 0.40,
            kd_z: 0.25,
            kp_att: 0.0025,
            kd_att: 0.0008,
            yaw_kp: 0.0008,
        }
    }
}

impl WaypointController {
    /// Compute a motor command from current state + target waypoint.
    ///
    /// Returns a clamped `QuadrotorCommand`. The caller may scale this by a
    /// motor-gain value from the consciousness side-channel before stepping physics.
    pub fn compute(&self, state: &FlightState, mass: f64) -> QuadrotorCommand {
        let g = 9.81;

        let dx = self.target[0] - state.position[0];
        let dy = self.target[1] - state.position[1];
        let dz = self.target[2] - state.position[2];
        let vz = state.linear_velocity[2];

        let thrust = (mass * g + self.kp_z * mass * dz - self.kd_z * mass * vz) as f32;

        let desired_pitch = (self.kp_xy * dx).clamp(-0.30, 0.30);
        let desired_roll = (-self.kp_xy * dy).clamp(-0.30, 0.30);

        let (roll, pitch, _yaw) = state.euler_angles();
        let (wx, wy, wz) = (
            state.angular_velocity[0],
            state.angular_velocity[1],
            state.angular_velocity[2],
        );

        let roll_moment = (self.kp_att * (desired_roll - roll) - self.kd_att * wx) as f32;
        let pitch_moment = (self.kp_att * (desired_pitch - pitch) - self.kd_att * wy) as f32;
        let yaw_moment = (-self.yaw_kp * wz) as f32;

        QuadrotorCommand {
            thrust,
            roll_moment,
            pitch_moment,
            yaw_moment,
        }
        .clamped()
    }
}

/// Scale every channel of a command by a gain in `[0,1]`.
///
/// Thrust is handled specially: we don't want the gain to drop thrust below
/// hover, which would kill altitude instantly. Instead we interpolate between
/// hover thrust and the commanded thrust: `gain=0` → hover, `gain=1` → cmd.
pub fn gain_scale(cmd: QuadrotorCommand, gain: f64) -> QuadrotorCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    let hover = QuadrotorCommand::HOVER_THRUST;
    QuadrotorCommand {
        thrust: hover + g * (cmd.thrust - hover),
        roll_moment: g * cmd.roll_moment,
        pitch_moment: g * cmd.pitch_moment,
        yaw_moment: g * cmd.yaw_moment,
    }
}
