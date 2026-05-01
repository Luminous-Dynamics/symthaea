// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Joint types for articulated bodies.
//!
//! All joints implement the `Constraint<D>` trait, using the same iterative
//! position + velocity solve pipeline as other constraints. Joints are
//! dimension-agnostic (const-generic `<const D: usize>`).

pub mod ball;
pub mod fixed;
pub mod hinge;
pub mod prismatic;

pub use ball::BallJoint;
pub use fixed::FixedJoint;
pub use hinge::HingeJoint;
pub use prismatic::PrismaticJoint;

/// Motor drive for actuated joints (hinge + prismatic).
///
/// Uses a PD controller to reach a target angle/position:
/// `force = kp * (target - current) - kd * velocity`
#[derive(Clone, Debug)]
pub struct MotorDrive {
    /// Target angle (radians, for hinge) or position (units, for prismatic).
    pub target: f64,
    /// Maximum force/torque the motor can apply.
    pub max_force: f64,
    /// Damping coefficient (velocity damping, prevents overshoot).
    pub damping: f64,
}

impl MotorDrive {
    /// Create a motor with the given target and max force.
    pub fn new(target: f64, max_force: f64) -> Self {
        Self {
            target,
            max_force,
            damping: max_force * 0.1, // default 10% of max force
        }
    }
}
