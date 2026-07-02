// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stance-hold controller.
//!
//! The `SimpleQuadrupedSimulator` drives locomotion via a built-in
//! central pattern generator (CPG) at the gait's frequency; joint
//! torques just ride on top for small corrections. We emit small
//! torques toward a neutral standing pose. The real Φ-gate is the
//! `set_gait(GaitType::from_phi(phi))` call in the plugin, not the
//! torque magnitude — so `gain_from_gait` below is a modest
//! magnitude tint on top of the hard mode switch.

use symthaea_quadruped::types::{
    GaitType, NUM_ACTUATORS, NUM_JOINTS, QuadrupedCommand, QuadrupedState,
};

pub struct StanceController {
    /// Per-joint neutral target (matches `QuadrupedState::standing` layout:
    /// each leg = (hip_yaw, hip_pitch, knee_pitch) with the knee pre-bent).
    pub target_angles: [f64; NUM_JOINTS],
    pub kp: f64,
    pub kd: f64,
}

impl Default for StanceController {
    fn default() -> Self {
        Self {
            target_angles: [
                0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0,
            ],
            kp: 6.0,
            kd: 0.5,
        }
    }
}

impl StanceController {
    pub fn compute(&self, state: &QuadrupedState, gait: GaitType) -> QuadrupedCommand {
        let mut torques = [0.0f32; NUM_ACTUATORS];
        let gate = torque_gain_for_gait(gait);
        for i in 0..NUM_JOINTS {
            let err = self.target_angles[i] - state.joint_angles[i];
            let vel = state.joint_velocities[i];
            let raw = self.kp * err - self.kd * vel;
            let t = (raw / 8.0).clamp(-1.0, 1.0) as f32;
            torques[i] = t * gate;
        }
        QuadrupedCommand {
            joint_torques: torques,
        }
    }
}

/// How much to weight the stance-correction torques by gait mode. The
/// CPG already goes silent at Freeze/Collapse (frequency = 0), but we
/// cut the correction torques too so a low-Φ robot visibly relaxes.
pub fn torque_gain_for_gait(gait: GaitType) -> f32 {
    match gait {
        GaitType::Trot => 1.0,
        GaitType::Walk => 0.8,
        GaitType::Freeze => 0.3,
        GaitType::Collapse => 0.0,
    }
}
