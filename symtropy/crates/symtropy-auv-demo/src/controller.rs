// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 8-thruster controller for AUV waypoint navigation.
//!
//! Thruster layout (matching `AuvCommand::forward` / `descend` conventions):
//!   0,1 = surge (horizontal, forward/reverse)
//!   2,3 = sway (lateral) / yaw differential
//!   4,5 = heave (vertical); negative value descends (+z in this world)
//!   6,7 = pitch differential (unused in this demo)
//!
//! The controller maps waypoint errors onto this vector:
//!   - surge ← forward distance to target (in body frame)
//!   - yaw   ← heading error toward target
//!   - heave ← depth error

use symthaea_auv::types::{AuvCommand, AuvState};

pub struct WaypointController {
    pub target: [f64; 3],
    pub kp_surge: f64,
    pub kp_yaw: f64,
    pub kp_depth: f64,
    pub kd_depth: f64,
}

impl Default for WaypointController {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 10.0],
            kp_surge: 0.18,
            kp_yaw: 0.55,
            kp_depth: 0.35,
            kd_depth: 0.25,
        }
    }
}

impl WaypointController {
    pub fn compute(&self, state: &AuvState) -> AuvCommand {
        // Horizontal plane errors (world frame)
        let dx = self.target[0] - state.position[0];
        let dy = self.target[1] - state.position[1];
        let horiz = (dx * dx + dy * dy).sqrt();

        // Current heading (yaw) from the body quaternion
        let [w, x, y, z] = state.quaternion;
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));

        let desired_heading = dy.atan2(dx);
        let heading_err = wrap_angle(desired_heading - yaw);

        // Body-frame surge = |horiz| projected onto forward (after heading correction)
        // A cleaner-than-necessary approximation: surge speed proportional to
        // horizontal distance, attenuated when we're pointed wrong.
        let align = heading_err.cos().max(0.0);
        let surge = (self.kp_surge * horiz * align).clamp(0.0, 0.8) as f32;

        // Yaw differential — positive value pushes the nose to port
        let yaw_cmd = (self.kp_yaw * heading_err).clamp(-0.8, 0.8) as f32;

        // Depth: positive = downward. Error is target_depth - current_depth.
        let depth_err = self.target[2] - state.depth;
        let depth_rate = state.linear_velocity[2]; // body-frame heave (approximates world dz/dt near level)
        let depth_cmd =
            (self.kp_depth * depth_err - self.kd_depth * depth_rate).clamp(-0.8, 0.8) as f32;

        // Map onto 8 thrusters. Sign convention for heave matches
        // `AuvCommand::descend`: negative values produce positive depth motion.
        let mut cmd = AuvCommand::zero();
        cmd.thrusters[0] = surge;
        cmd.thrusters[1] = surge;
        // yaw differential: thrusters 2 and 3 (opposing sign)
        cmd.thrusters[2] = yaw_cmd;
        cmd.thrusters[3] = -yaw_cmd;
        // heave (negate to match descend convention)
        cmd.thrusters[4] = -depth_cmd;
        cmd.thrusters[5] = -depth_cmd;
        cmd.clamped()
    }
}

/// Scale all thrusters by a consciousness motor gain.
///
/// Unlike the vehicle demo (where brake was pass-through), AUV thrusters are
/// bidirectional — every channel is under the gain. Gain < 1 shrinks
/// aggressive thrust changes under disturbance.
pub fn gain_scale(cmd: AuvCommand, gain: f64) -> AuvCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    let mut out = cmd;
    for t in &mut out.thrusters {
        *t *= g;
    }
    out
}

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
