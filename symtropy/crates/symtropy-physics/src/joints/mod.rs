// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Joint types for articulated bodies.
//!
//! All joints implement the `Constraint<D>` trait, using the same iterative
//! position + velocity solve pipeline as other constraints. Joints are
//! dimension-agnostic (const-generic `<const D: usize>`).

pub mod ball;
pub mod fixed;
pub mod hinge;

pub use ball::BallJoint;
pub use fixed::FixedJoint;
pub use hinge::HingeJoint;
