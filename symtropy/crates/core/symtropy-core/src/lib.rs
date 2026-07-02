// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # symtropy-core
//!
//! The permissive Symtropy distribution: core Bevy physics bundle without AGPL dependencies.

pub use symtropy_bevy_core as bevy_physics;
pub use symtropy_bevy_scene as scene;
pub use symtropy_devconsole as devconsole;
pub use symtropy_math as math;
pub use symtropy_physics as physics;

pub mod prelude {
    pub use crate::bevy_physics::{BevyPhysicsPlugin, NoCouplingResource, PhysicsBody};
    pub use crate::math::Point;
    pub use crate::physics::body::BodyHandle;
    pub use crate::physics::world::PhysicsWorld;
}
