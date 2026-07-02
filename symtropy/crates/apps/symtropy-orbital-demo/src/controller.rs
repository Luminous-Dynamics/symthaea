// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint-space PD controller driving toward a deployment pose.

use symthaea_orbital::types::{NUM_ACTUATORS, NUM_JOINTS, OrbitalCommand, OrbitalState};

pub struct DeploymentController {
    pub target_angles: [f64; NUM_JOINTS],
    pub kp: f64,
    pub kd: f64,
}

impl Default for DeploymentController {
    fn default() -> Self {
        // A "reach forward + slight pitch" deployment pose.
        // Joint 0 = shoulder yaw, 1 = shoulder pitch, 2 = elbow pitch,
        // 3 = wrist pitch, 4 = wrist yaw, 5 = gripper pitch, 6 = gripper roll.
        Self {
            target_angles: [0.6, 0.9, -1.4, 0.4, 0.0, 0.2, 0.0],
            kp: 4.0,
            kd: 1.5,
        }
    }
}

impl DeploymentController {
    pub fn compute(&self, state: &OrbitalState) -> OrbitalCommand {
        let mut torques = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            let raw = self.kp * err - self.kd * vel;
            // Normalize into [-1, 1]; simulator multiplies by config.max_joint_torques.
            torques[i] = (raw / 8.0).clamp(-1.0, 1.0) as f32;
        }
        OrbitalCommand {
            joint_torques: torques,
        }
    }
}

/// Scale every joint torque by the consciousness motor gain. This is the
/// simplest (and for orbital, most natural) Φ-gate: all motion slows when Φ
/// drops, letting reaction wheels catch up with accumulated angular momentum
/// before more arm motion is commanded.
pub fn gain_scale(cmd: OrbitalCommand, gain: f64) -> OrbitalCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    let mut out = cmd;
    for t in &mut out.joint_torques {
        *t *= g;
    }
    out
}
