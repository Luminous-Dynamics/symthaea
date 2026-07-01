// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Per-joint safety authority gating.

use symtropy_consciousness_physics::safety::SafetyTier;

/// Safety authority that operates at individual joint resolution.
pub struct JointSafetyAuthority {
    /// Per-joint safety tiers.
    pub joint_tiers: Vec<SafetyTier>,
}

impl JointSafetyAuthority {
    /// Create a new authority for the given number of actuators.
    pub fn new(num_actuators: usize) -> Self {
        Self {
            joint_tiers: vec![SafetyTier::Green; num_actuators],
        }
    }

    /// Update per-joint safety tiers from localized prediction errors.
    pub fn update_from_surprise(&mut self, surprises: &[f64]) {
        for (i, &surprise) in surprises.iter().enumerate() {
            if i >= self.joint_tiers.len() {
                break;
            }
            self.joint_tiers[i] = if surprise < 1.0 {
                SafetyTier::Green
            } else if surprise < 3.0 {
                SafetyTier::Yellow
            } else if surprise < 5.0 {
                SafetyTier::Orange
            } else {
                SafetyTier::Red
            };
        }
    }

    /// Apply per-joint safety gains to a command vector.
    pub fn apply_gains(&self, commands: &mut [f64]) {
        for (i, command) in commands.iter_mut().enumerate() {
            if i >= self.joint_tiers.len() {
                break;
            }
            *command *= self.joint_tiers[i].motor_gain();
        }
    }
}
