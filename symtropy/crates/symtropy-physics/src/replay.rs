// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Determinism helpers: record/replay command streams and bitwise snapshots.
//!
//! This module is intentionally minimal: it provides
//! - a small `WorldCommand` vocabulary for driving a `PhysicsWorld`
//! - a `ReplayTape` of per-tick frames (dt + commands)
//! - `WorldSnapshot`/`BodySnapshot` that capture simulation state as raw `f64` bits
//!
//! The goal is to make it easy to build a replay harness that asserts
//! **bitwise-equal state per tick** across record/replay passes.

use nalgebra::SVector;
use symtropy_math::Bivector;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::integrator;
use crate::world::PhysicsWorld;

/// Commands that mutate a physics world at a tick boundary.
#[derive(Clone, Debug)]
pub enum WorldCommand<const D: usize> {
    ApplyForce {
        body: BodyHandle,
        force: SVector<f64, D>,
    },
    ApplyImpulse {
        body: BodyHandle,
        impulse: SVector<f64, D>,
    },
    SetLinearVelocity {
        body: BodyHandle,
        velocity: SVector<f64, D>,
    },
    SetAngularVelocity {
        body: BodyHandle,
        velocity: Bivector<D>,
    },
    Wake {
        body: BodyHandle,
    },
}

/// A single replay frame: `dt` + ordered list of commands to apply before stepping.
#[derive(Clone, Debug)]
pub struct ReplayFrame<const D: usize> {
    pub dt: f64,
    pub commands: Vec<WorldCommand<D>>,
}

/// A full replay tape: ordered frames.
#[derive(Clone, Debug, Default)]
pub struct ReplayTape<const D: usize> {
    pub frames: Vec<ReplayFrame<D>>,
}

impl<const D: usize> ReplayTape<D> {
    pub fn push_frame(&mut self, dt: f64, commands: Vec<WorldCommand<D>>) {
        self.frames.push(ReplayFrame { dt, commands });
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyCommandError {
    MissingBody(BodyHandle),
}

/// Apply a list of commands to a world.
pub fn apply_commands<const D: usize>(
    world: &mut PhysicsWorld<D>,
    commands: &[WorldCommand<D>],
) -> Result<(), ApplyCommandError> {
    for cmd in commands {
        match cmd {
            WorldCommand::ApplyForce { body, force } => {
                let Some(b) = world.body_mut(*body) else {
                    return Err(ApplyCommandError::MissingBody(*body));
                };
                b.apply_force(force.clone());
            }
            WorldCommand::ApplyImpulse { body, impulse } => {
                let Some(b) = world.body_mut(*body) else {
                    return Err(ApplyCommandError::MissingBody(*body));
                };
                integrator::apply_impulse(b, impulse);
            }
            WorldCommand::SetLinearVelocity { body, velocity } => {
                let Some(b) = world.body_mut(*body) else {
                    return Err(ApplyCommandError::MissingBody(*body));
                };
                b.linear_velocity = velocity.clone();
            }
            WorldCommand::SetAngularVelocity { body, velocity } => {
                let Some(b) = world.body_mut(*body) else {
                    return Err(ApplyCommandError::MissingBody(*body));
                };
                b.angular_velocity = *velocity;
            }
            WorldCommand::Wake { body } => {
                let Some(b) = world.body_mut(*body) else {
                    return Err(ApplyCommandError::MissingBody(*body));
                };
                b.wake();
            }
        }
    }
    Ok(())
}

/// Bitwise snapshot of a rigid body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodySnapshot<const D: usize> {
    pub handle: BodyHandle,
    pub body_type: BodyType,
    pub translation: [u64; D],
    pub rotation: [[u64; D]; D],
    pub linear_velocity: [u64; D],
    pub angular_velocity: [[u64; D]; D],
    pub sleeping: bool,
    pub sleep_counter: u32,
}

impl<const D: usize> BodySnapshot<D> {
    pub fn from_body(body: &RigidBody<D>) -> Self {
        let translation = std::array::from_fn(|i| body.transform.translation.0[i].to_bits());

        let rot = body.transform.rotation.to_matrix();
        let rotation = std::array::from_fn(|r| std::array::from_fn(|c| rot[(r, c)].to_bits()));

        let linear_velocity = std::array::from_fn(|i| body.linear_velocity[i].to_bits());

        let ang = body.angular_velocity.to_matrix();
        let angular_velocity =
            std::array::from_fn(|r| std::array::from_fn(|c| ang[(r, c)].to_bits()));

        Self {
            handle: body.handle,
            body_type: body.body_type,
            translation,
            rotation,
            linear_velocity,
            angular_velocity,
            sleeping: body.sleeping,
            sleep_counter: body.sleep_counter,
        }
    }
}

/// Bitwise snapshot of a collision event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollisionEventSnapshot<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub impulse: u64,
    pub normal: [u64; D],
    pub depth: u64,
}

impl<const D: usize> CollisionEventSnapshot<D> {
    pub fn from_event(event: &crate::contact::CollisionEvent<D>) -> Self {
        Self {
            body_a: event.body_a,
            body_b: event.body_b,
            impulse: event.impulse.to_bits(),
            normal: std::array::from_fn(|i| event.normal[i].to_bits()),
            depth: event.depth.to_bits(),
        }
    }
}

/// Bitwise snapshot of a physics world (bodies + last-step collision events).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldSnapshot<const D: usize> {
    pub bodies: Vec<BodySnapshot<D>>,
    pub collision_events: Vec<CollisionEventSnapshot<D>>,
}

impl<const D: usize> WorldSnapshot<D> {
    pub fn capture(world: &PhysicsWorld<D>) -> Self {
        let mut bodies: Vec<_> = world.bodies.iter().map(BodySnapshot::from_body).collect();
        bodies.sort_by_key(|b| b.handle);

        let mut collision_events: Vec<_> = world
            .collision_events
            .iter()
            .map(CollisionEventSnapshot::from_event)
            .collect();
        collision_events.sort_by_key(|e| (e.body_a, e.body_b, e.impulse, e.depth));

        Self {
            bodies,
            collision_events,
        }
    }
}
