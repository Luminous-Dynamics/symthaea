// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! N-dimensional rigid body physics engine.
//!
//! Provides dimension-agnostic rigid body dynamics, GJK collision detection,
//! and constraint solving. All types are `const D: usize` parameterized for
//! stack-allocated, SIMD-friendly physics at 2D/3D/4D.
//!
//! # Architecture
//! - `RigidBody<D>` — position, velocity, angular velocity (bivector), mass, collider
//! - `PhysicsWorld<D>` — owns bodies, steps simulation, resolves collisions
//! - `gjk::intersects()` — GJK intersection test for any `Shape<D>`
//! - `contact::ContactManifold<D>` — collision contact data
//! - `integrator` — semi-implicit Euler with bivector angular dynamics

pub mod body;
pub mod broadphase;
pub mod contact;
pub mod constraint;
pub mod epa;
pub mod gjk;
pub mod integrator;
pub mod world;

pub use body::{BodyHandle, BodyType, RigidBody};
pub use contact::{CollisionEvent, ContactManifold};
pub use constraint::Constraint;
pub use epa::EpaResult;
pub use broadphase::{Aabb, Lbvh, morton_encode, morton_prefix};
pub use world::{NoOpCallback, PhysicsCallback, PhysicsWorld};
