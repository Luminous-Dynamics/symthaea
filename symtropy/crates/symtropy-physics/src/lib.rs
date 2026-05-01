// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
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

pub mod articulation;
pub mod body;
pub mod broadphase;
pub mod ccd;
pub mod constraint;
pub mod contact;
pub mod epa;
pub mod gjk;
pub mod integrator;
pub mod island;
pub mod joints;
pub mod manifold_gen;
pub mod raycast;
pub mod replay;
pub mod world;

pub use articulation::{ArticulatedChain, ChainBuilder, LinkSpec};
pub use body::{BodyHandle, BodyType, NetId, RigidBody};
pub use broadphase::{morton_encode, morton_prefix, Aabb, Lbvh};
pub use constraint::Constraint;
pub use contact::{CollisionEvent, ContactCache, ContactManifold, SensorEvent};
pub use epa::EpaResult;
pub use joints::{BallJoint, FixedJoint, HingeJoint, MotorDrive, PrismaticJoint};
pub use replay::{apply_commands, ReplayTape, WorldCommand, WorldSnapshot};
pub use world::{NoOpCallback, PhysicsCallback, PhysicsWorld};
