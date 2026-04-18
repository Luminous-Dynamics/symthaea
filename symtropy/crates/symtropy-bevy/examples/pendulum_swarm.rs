// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm` — Tier 1 showcase: Phi-coupled physics on a 10x10 grid.
//!
//! See [`pendulum_swarm.md`](./pendulum_swarm.md) for the full design doc,
//! empirical constants, and step-by-step implementation plan. This file is
//! built incrementally per the plan's "Implementation steps" section.
//!
//! Current step: **4 — Phi from neighborhood variance.** Each bob is registered
//! with the `ConsciousnessField`. Each tick, a system reads the bob's velocity
//! and its 8 neighbors' velocities, computes the variance, and maps low variance
//! → high coherence → high Phi (via uniform `ConsciousnessInputs`). Damping is
//! still uniform (Step 5 wires Phi → damping). Phi is only printed at startup
//! diagnostic level for now; visual coupling lands in Step 6.

use std::collections::HashMap;

use bevy::prelude::*;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_math::{Point, Sphere};
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, RigidBody};

const ARM_LENGTH: f64 = 60.0;
const BOB_RADIUS: f32 = 10.0;
const PIVOT_RADIUS: f32 = 4.0;
const GRID: usize = 10;
const SPACING: f64 = 64.0;
const SANCTUARY_RADIUS: f64 = 32.0;
const MAX_ENERGY: f64 = 100.0;
// Variance scale: tuned so a swing of ~200 px/s vs neighbor at rest maps to
// coherence ≈ 0.5. With v ∈ [0, 400], var can reach ~10000; we want
// 1/(1 + var * scale) ≈ 0.5 there → scale ≈ 1e-4.
const VARIANCE_SCALE: f64 = 1.0e-4;

#[derive(Component)]
struct Pendulum {
    bob: BodyHandle,
    pivot_pos: Vec2,
    grid: (usize, usize),
}

#[derive(Resource, Default)]
struct GridHandles {
    map: HashMap<(usize, usize), BodyHandle>,
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
        .insert_resource(GridHandles::default())
        .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -981.0]))
        .add_systems(Startup, (setup_camera, spawn_swarm))
        .add_systems(FixedUpdate, update_phi_from_neighborhood)
        .add_systems(Update, draw_arm_gizmo)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn spawn_swarm(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<2>>,
    mut grid_handles: ResMut<GridHandles>,
) {
    let half = (GRID as f64 - 1.0) * SPACING * 0.5;
    for i in 0..GRID {
        for j in 0..GRID {
            let pivot_x = (i as f64) * SPACING - half;
            let pivot_y = (j as f64) * SPACING - half;
            let bob = spawn_pendulum(&mut commands, &mut physics, pivot_x, pivot_y, (i, j));
            grid_handles.map.insert((i, j), bob);
        }
    }
}

fn spawn_pendulum(
    commands: &mut Commands,
    physics: &mut SymtropyPhysics<2>,
    pivot_x: f64,
    pivot_y: f64,
    grid: (usize, usize),
) -> BodyHandle {
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

    physics
        .field
        .register(bob_handle, MAX_ENERGY, SANCTUARY_RADIUS);

    commands.spawn((
        Sprite::from_color(Color::srgb(0.6, 0.6, 0.65), Vec2::splat(PIVOT_RADIUS * 2.0)),
        Transform::from_xyz(pivot_x as f32, pivot_y as f32, 0.0),
    ));

    commands.spawn((
        Pendulum {
            bob: bob_handle,
            pivot_pos: Vec2::new(pivot_x as f32, pivot_y as f32),
            grid,
        },
        PhysicsBody::new(bob_handle, BOB_RADIUS),
        Sprite::from_color(Color::srgb(0.7, 0.85, 1.0), Vec2::splat(BOB_RADIUS * 2.0)),
        Transform::from_xyz((pivot_x + ARM_LENGTH) as f32, pivot_y as f32, 1.0),
    ));

    bob_handle
}

/// For each pendulum: read self+neighbor velocities, compute variance,
/// map low variance to high coherence, feed uniform `ConsciousnessInputs`
/// into the field. Low variance = neighborhood is moving together (or all at
/// rest) = coherent = high Phi.
fn update_phi_from_neighborhood(
    mut physics: ResMut<SymtropyPhysics<2>>,
    grid_handles: Res<GridHandles>,
    pendulums: Query<&Pendulum>,
) {
    for p in &pendulums {
        let (gi, gj) = p.grid;
        let mut sum_v = 0.0;
        let mut sum_v_sq = 0.0;
        let mut count = 0;
        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                let ni = gi as i32 + di;
                let nj = gj as i32 + dj;
                if ni < 0 || nj < 0 || ni >= GRID as i32 || nj >= GRID as i32 {
                    continue;
                }
                if let Some(&h) = grid_handles.map.get(&(ni as usize, nj as usize)) {
                    if let Some(body) = physics.world.body(h) {
                        let v = body.linear_velocity.norm();
                        sum_v += v;
                        sum_v_sq += v * v;
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            continue;
        }
        let mean = sum_v / count as f64;
        let var = (sum_v_sq / count as f64) - mean * mean;
        let coherence = (1.0 / (1.0 + var * VARIANCE_SCALE)).clamp(0.0, 1.0);
        let inputs = ConsciousnessInputs {
            phi: coherence,
            broadcast: coherence,
            working_memory: coherence,
            attention: coherence,
            recurrence: coherence,
            embodiment: coherence,
            knowledge: coherence,
            synchrony: coherence,
        };
        let pos = physics
            .world
            .body(p.bob)
            .map(|b| *b.position())
            .unwrap_or_else(Point::origin);
        physics.field.update_entity(p.bob, &inputs, pos);
    }
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
