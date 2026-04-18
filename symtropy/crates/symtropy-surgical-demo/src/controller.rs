// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint-space controller toward a target tissue pose. Cautery is a simple
//! state-machine: OFF when retracting, ON (if allowed) when in contact.

use symthaea_surgical::types::{SurgicalCommand, SurgicalSafetyLevel, SurgicalState, NUM_JOINTS};

pub struct CauteryProcedureController {
    /// Joint-space target (close to tissue, slight pitch for cautery angle).
    pub target_angles: [f64; NUM_JOINTS],
    pub kp: f64,
    pub kd: f64,
}

impl Default for CauteryProcedureController {
    fn default() -> Self {
        // Joints 0..5 = shoulder, elbow, wrist pitch, wrist yaw, tool roll, jaw.
        // Target pose drops the tip into the tissue region for the procedure.
        Self {
            target_angles: [0.25, 0.55, -0.35, 0.20, 0.0, 0.0],
            kp: 2.5,
            kd: 1.0,
        }
    }
}

impl CauteryProcedureController {
    pub fn compute(&self, state: &SurgicalState, level: SurgicalSafetyLevel) -> SurgicalCommand {
        let mut torques = [0.0f32; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            let raw = self.kp * err - self.kd * vel;
            // Normalize into [-1, 1] — simulator multiplies by max_joint_torques.
            torques[i] = (raw / 3.0).clamp(-1.0, 1.0) as f32;
        }

        // Jaw closes slowly toward 0.6 once we're approximately in pose.
        let in_pose_err: f64 = self
            .target_angles
            .iter()
            .zip(state.joint_angles.iter())
            .map(|(t, q)| (t - q).abs())
            .sum();
        let jaw_target = if in_pose_err < 0.6 { 0.6 } else { 0.0 };

        // Request cautery when in pose; the interlock below decides whether it fires.
        let cautery_request = if in_pose_err < 0.3 { 1.0 } else { 0.0 };

        // Apply the platform's torque gain and cautery interlock directly.
        let gain = level.torque_gain();
        for t in &mut torques {
            *t *= gain;
        }
        let cautery = if level.cautery_allowed() {
            cautery_request
        } else {
            0.0
        };

        SurgicalCommand {
            joint_torques: torques,
            jaw: jaw_target as f32,
            cautery,
        }
    }
}
