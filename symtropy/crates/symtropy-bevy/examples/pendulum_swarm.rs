// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm` — Tier 1 showcase: Phi-coupled physics on a 10x10 grid.
//!
//! See [`pendulum_swarm.md`](./pendulum_swarm.md) for the full design doc,
//! empirical constants, and step-by-step implementation plan. This file is
//! built incrementally per the plan's "Implementation steps" section.
//!
//! Current step: **7 + jitter polish.** Demo is feature-complete (Steps 1-7
//! landed). Bobs now spawn at slightly perturbed angles (~horizontal ± 17°,
//! deterministic per-cell) instead of perfect lockstep — this gives
//! `update_phi_from_neighborhood` non-zero baseline variance from frame 1, so
//! the Phi values vary across the grid even before any click.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PrimaryWindow;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_devconsole::SymtropyDevConsolePlugin;
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
// Phi normalization (landmine 0): MasterConsciousnessEquation steady-state
// max under uniform unit inputs is ~0.314, NOT 1.0. Without normalizing here,
// the damping range collapses from 500× to ~1.5×.
const PHI_NORMALIZE: f64 = 0.314;
const LOW_DAMP: f64 = 0.001; // phi ≈ max → essentially conservative
const HIGH_DAMP: f64 = 0.5; // phi = 0 → PBD sleep within ~4 s
const SHOCK_RADIUS: f32 = 50.0;
const SHOCK_VELOCITY: f64 = 400.0;
// Per-cell phase jitter amplitude (radians). 0.30 ≈ ±17°. Breaks the all-synced
// equilibrium so neighborhood variance is non-zero from the first frame.
const PHASE_JITTER: f64 = 0.30;

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

/// Headless verification mode. Set `PENDULUM_CAPTURE_DIR=/some/dir` before
/// running to schedule screenshots at fixed real-time intervals and exit
/// the app afterward. Default behaviour (env unset) is unchanged.
#[derive(Resource)]
struct CaptureSchedule {
    dir: String,
    schedule: Vec<f32>, // seconds since startup, must be sorted
    fired: usize,
    exit_at: f32,
}

fn capture_schedule_from_env() -> Option<CaptureSchedule> {
    let dir = std::env::var("PENDULUM_CAPTURE_DIR").ok()?;
    Some(CaptureSchedule {
        dir,
        schedule: vec![1.5, 4.0, 7.0],
        fired: 0,
        exit_at: 8.5,
    })
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
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
    .add_plugins(SymtropyDevConsolePlugin)
    .add_systems(Startup, (setup_camera, spawn_swarm))
    .add_systems(
        FixedUpdate,
        (update_phi_from_neighborhood, phi_modulates_damping).chain(),
    )
    .add_systems(Update, (shock_on_click, color_by_phi, draw_arm_gizmo));

    if let Some(sched) = capture_schedule_from_env() {
        app.insert_resource(sched);
        app.add_systems(Update, headless_capture);
    }

    app.run();
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

    // Deterministic per-cell phase offset around horizontal (90° from vertical).
    // Hash-style index combination keeps adjacent cells visually distinct.
    let cell_hash = grid.0.wrapping_mul(7).wrapping_add(grid.1.wrapping_mul(13));
    let phase = (cell_hash as f64 * 0.37).sin() * PHASE_JITTER;
    let theta = std::f64::consts::FRAC_PI_2 + phase;
    let bob_x = pivot_x + ARM_LENGTH * theta.sin();
    let bob_y = pivot_y - ARM_LENGTH * theta.cos();

    let bob_handle = physics
        .world
        .add_sphere(Point::new([bob_x, bob_y]), 10.0, 1.0);
    if let Some(b) = physics.world.body_mut(bob_handle) {
        b.collision_mask = 0;
        // Initial damping; overwritten each FixedUpdate by `phi_modulates_damping`.
        b.linear_damping = HIGH_DAMP;
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
        Transform::from_xyz(bob_x as f32, bob_y as f32, 1.0),
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

/// Read each pendulum's current Phi, normalize, and write per-body damping.
/// Coherent neighborhoods (high Phi) → near-conservative damping; incoherent
/// neighborhoods → high damping → PBD sleep freezes the bob within seconds.
fn phi_modulates_damping(mut physics: ResMut<SymtropyPhysics<2>>, pendulums: Query<&Pendulum>) {
    for p in &pendulums {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0);
        let damping = HIGH_DAMP + (LOW_DAMP - HIGH_DAMP) * phi_norm;
        if let Some(body) = physics.world.body_mut(p.bob) {
            body.linear_damping = damping;
        }
    }
}

/// On left-click, find the nearest pendulum bob within `SHOCK_RADIUS` of the
/// cursor (in world coords) and add `SHOCK_VELOCITY` horizontal kick. Linear
/// scan over 100 entries — trivial.
fn shock_on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    grid_handles: Res<GridHandles>,
    mut physics: ResMut<SymtropyPhysics<2>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_xform)) = cameras.single() else {
        return;
    };
    let Ok(world_pos) = camera.viewport_to_world_2d(camera_xform, cursor_pos) else {
        return;
    };

    let mut nearest: Option<(BodyHandle, f32)> = None;
    for (_, &h) in grid_handles.map.iter() {
        if let Some(body) = physics.world.body(h) {
            let pos = body.position();
            let dx = pos.coord(0) as f32 - world_pos.x;
            let dy = pos.coord(1) as f32 - world_pos.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < SHOCK_RADIUS && nearest.map_or(true, |(_, nd)| d < nd) {
                nearest = Some((h, d));
            }
        }
    }
    if let Some((h, _)) = nearest {
        if let Some(body) = physics.world.body_mut(h) {
            body.linear_velocity[0] += SHOCK_VELOCITY;
        }
    }
}

/// Sprite color reflects current Phi: blue (cool, Phi=0) → red (warm, Phi=max).
/// Updates every frame via `Update` so visual stays smooth even between
/// FixedUpdate physics ticks.
fn color_by_phi(physics: Res<SymtropyPhysics<2>>, mut query: Query<(&Pendulum, &mut Sprite)>) {
    for (p, mut sprite) in &mut query {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0) as f32;
        sprite.color = Color::hsl(240.0 - phi_norm * 240.0, 1.0, 0.5);
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

/// Headless verification: fires screenshots at scheduled times, exits when done.
fn headless_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut sched: ResMut<CaptureSchedule>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    while sched.fired < sched.schedule.len() && now >= sched.schedule[sched.fired] {
        let label = sched.schedule[sched.fired];
        let path = format!("{}/pswarm_t{:.1}.png", sched.dir, label);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!("queued screenshot: {}", path);
        sched.fired += 1;
    }
    if now >= sched.exit_at {
        info!("headless capture done — exiting");
        exit.write(AppExit::Success);
    }
}
