// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm` — Tier 1 showcase: Phi-coupled physics on a 10x10 grid.
//!
//! See [`pendulum_swarm.md`](./pendulum_swarm.md) for the full design doc,
//! empirical constants, and step-by-step implementation plan. This file is
//! built incrementally per the plan's "Implementation steps" section.
//!
//! Current step: **2 — One pendulum.** Static pivot + dynamic bob hinged by a
//! `DistanceConstraint`, released from horizontal. Should swing under gravity.
//! Debug gizmo draws the constraint arm. Physics units = pixels (gravity scales
//! to -981 px/s² so 100 px = 1 m).

use bevy::prelude::*;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_math::{Point, Sphere};
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, RigidBody};

const ARM_LENGTH: f64 = 60.0;
const PIVOT_Y: f64 = 200.0;
const BOB_RADIUS: f32 = 10.0;
const PIVOT_RADIUS: f32 = 4.0;

#[derive(Component)]
struct Pendulum {
    #[allow(dead_code)] // used by Step 4 (neighborhood variance for Phi)
    bob: BodyHandle,
    pivot_pos: Vec2,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symtropy: Pendulum Swarm (Phi-coupled physics)".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.04, 0.06)))
        .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -981.0]))
        .add_systems(Startup, (setup_camera, spawn_one_pendulum))
        .add_systems(Update, draw_arm_gizmo)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn spawn_one_pendulum(mut commands: Commands, mut physics: ResMut<SymtropyPhysics<2>>) {
    let pivot_x = 0.0_f64;
    let pivot_y = PIVOT_Y;

    let pivot_handle = physics.world.add_body(RigidBody::<2>::static_body(
        BodyHandle(0),
        Point::new([pivot_x, pivot_y]),
        Box::new(Sphere::new(Point::origin(), 1.0)),
    ));
    if let Some(p) = physics.world.body_mut(pivot_handle) {
        p.collision_mask = 0;
    }

    let bob_handle =
        physics
            .world
            .add_sphere(Point::new([pivot_x + ARM_LENGTH, pivot_y]), 10.0, 1.0);
    if let Some(b) = physics.world.body_mut(bob_handle) {
        b.collision_mask = 0;
        b.linear_damping = 0.05;
    }

    physics
        .world
        .add_constraint(Box::new(DistanceConstraint::<2> {
            body_a: pivot_handle,
            body_b: bob_handle,
            rest_length: ARM_LENGTH,
            stiffness: 1.0,
        }));

    commands.spawn((
        Sprite::from_color(Color::srgb(0.6, 0.6, 0.65), Vec2::splat(PIVOT_RADIUS * 2.0)),
        Transform::from_xyz(pivot_x as f32, pivot_y as f32, 0.0),
    ));

    commands.spawn((
        Pendulum {
            bob: bob_handle,
            pivot_pos: Vec2::new(pivot_x as f32, pivot_y as f32),
        },
        PhysicsBody::new(bob_handle, BOB_RADIUS),
        Sprite::from_color(Color::srgb(0.7, 0.85, 1.0), Vec2::splat(BOB_RADIUS * 2.0)),
        Transform::from_xyz((pivot_x + ARM_LENGTH) as f32, pivot_y as f32, 1.0),
    ));
}

fn draw_arm_gizmo(mut gizmos: Gizmos, query: Query<(&Pendulum, &Transform)>) {
    for (p, t) in &query {
        gizmos.line_2d(
            p.pivot_pos,
            t.translation.truncate(),
            Color::srgb(0.4, 0.4, 0.45),
        );
    }
}
