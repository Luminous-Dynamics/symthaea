// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Joints and motors: constrained motion with PD controllers.

use crate::body::BodyHandle;
use nalgebra::SVector;

/// PD (Proportional-Derivative) controller configuration.
#[derive(Clone, Debug)]
pub struct PdController {
    pub kp: f64,         // Proportional gain (stiffness)
    pub kd: f64,         // Derivative gain (damping)
    pub max_effort: f64, // Max torque or force
}

impl Default for PdController {
    fn default() -> Self {
        Self {
            kp: 100.0,
            kd: 10.0,
            max_effort: 1000.0,
        }
    }
}

impl PdController {
    /// Calculate control effort (torque or force).
    pub fn calculate(&self, error: f64, error_dot: f64) -> f64 {
        let effort = self.kp * error + self.kd * error_dot;
        effort.clamp(-self.max_effort, self.max_effort)
    }
}

/// Joint motor state and target.
#[derive(Clone, Debug, Default)]
pub struct Motor {
    pub target_pos: f64,
    pub target_vel: f64,
    pub current_effort: f64,
    pub enabled: bool,
}

/// A hinge joint: rotates around a single axis.
pub struct HingeJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub anchor_a: SVector<f64, D>, // Local anchor in A
    pub anchor_b: SVector<f64, D>, // Local anchor in B
    pub axis: SVector<f64, D>,     // Rotation axis
    pub motor: Motor,
    pub pd: PdController,
}

/// A prismatic joint: slides along a single axis.
pub struct PrismaticJoint<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub anchor_a: SVector<f64, D>,
    pub anchor_b: SVector<f64, D>,
    pub axis: SVector<f64, D>, // Sliding axis
    pub motor: Motor,
    pub pd: PdController,
}
