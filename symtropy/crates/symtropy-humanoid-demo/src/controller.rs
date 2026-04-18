// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint-space balance controller: PD holding all joints near the neutral
//! standing pose. With 21 actuators driven by a single uniform gain it
//! isn't a real locomotion controller; the value of the demo is the
//! consciousness attenuation behavior under push, not task performance.

use symthaea_humanoid::types::{HumanoidCommand, HumanoidState};

pub struct BalanceController {
    pub kp: f64,
    pub kd: f64,
    /// Normalizing divisor used before clamping to [-1, 1].
    pub scale: f64,
}

impl Default for BalanceController {
    fn default() -> Self {
        Self {
            kp: 35.0,
            kd: 4.5,
            scale: 40.0,
        }
    }
}

impl BalanceController {
    /// Compute a raw (un-gated) balance command.
    pub fn compute(&self, state: &HumanoidState) -> HumanoidCommand {
        let n = state.joint_angles.len();
        let mut torques = Vec::with_capacity(n);
        for i in 0..n {
            let err = -state.joint_angles[i]; // target is 0
            let vel = state.joint_velocities[i];
            let raw = self.kp * err - self.kd * vel;
            torques.push((raw / self.scale).clamp(-1.0, 1.0) as f32);
        }
        HumanoidCommand { torques }
    }
}

/// Uniform scalar attenuation across all 21 torques.
///
/// Unlike helicopter or vehicle, there's no "safer action" carve-out here
/// — every joint is pure motion authority, so a single gain is the right
/// primitive.
pub fn gain_scale(cmd: HumanoidCommand, gain: f64) -> HumanoidCommand {
    let g = gain.clamp(0.0, 1.0) as f32;
    let torques: Vec<f32> = cmd.torques.iter().map(|t| t * g).collect();
    HumanoidCommand { torques }
}
