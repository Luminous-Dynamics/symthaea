// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm_3d` — 3D port of the 2D pendulum_swarm Tier 1 showcase.
//!
//! 10×10 grid of pendulums hanging in a 3D scene. Same Phi-from-neighborhood-
//! variance + Phi-modulates-damping coupling as the 2D demo. Bobs swing freely
//! in the X/Z plane (3D motion) because per-cell jitter is applied as a 2D
//! direction in the swing plane, not a single axis.
//!
//! Visual differences from 2D: real spheres with PBR `StandardMaterial`,
//! `DirectionalLight`, `Camera3d` looking down at the grid from a tilt angle
//! (so depth is visible). Click-to-shock uses `Camera::viewport_to_world` to
//! cast a ray and shock the nearest bob to the ray.
//!
//! Headless capture mode: `PENDULUM_CAPTURE_DIR=/some/dir cargo run ...`
//! schedules PNG screenshots at t=1.5/4.0/7.0 s and AppExit at t=8.5 s.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PrimaryWindow;
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_bevy_scene::{fixed_camera, SymtropyScenePlugin};
use symtropy_math::{Point, Sphere as PhysicsSphere};
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, RigidBody};

const ARM_LENGTH: f64 = 1.0; // 1 metre
const BOB_RADIUS: f32 = 0.10;
const PIVOT_RADIUS: f32 = 0.04;
const GRID: usize = 10;
const SPACING: f64 = 0.6; // metres between adjacent pivots
const SANCTUARY_RADIUS: f64 = 0.5;
const MAX_ENERGY: f64 = 100.0;
// Pendulum at amplitude 0.5 rad reaches v ≈ 1.5 m/s; squared ≈ 2.25;
// neighbourhood variance sits in the same range. Coherence = 0.5 at var ≈ 1
// → scale ≈ 1.0. Tunable.
const VARIANCE_SCALE: f64 = 1.0;
// Phi normalisation (landmine 0): MasterConsciousnessEquation steady-state
// max under uniform unit inputs is ~0.314, NOT 1.0.
const PHI_NORMALIZE: f64 = 0.314;
const LOW_DAMP: f64 = 0.001;
const HIGH_DAMP: f64 = 0.5;
const SHOCK_RAY_RADIUS: f32 = 0.20; // bob within 20 cm of cursor ray
const SHOCK_VELOCITY: f64 = 3.0; // m/s horizontal kick
const PHASE_JITTER: f64 = 0.30; // ±17° around vertical-down

#[derive(Component)]
struct Pendulum {
    bob: BodyHandle,
    pivot_pos: Vec3,
    grid: (usize, usize),
}

#[derive(Resource, Default)]
struct GridHandles {
    map: HashMap<(usize, usize), BodyHandle>,
}

#[derive(Resource)]
struct CaptureSchedule {
    dir: String,
    schedule: Vec<f32>,
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
            title: "Symtropy: Pendulum Swarm 3D (Phi-coupled physics)".into(),
            resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(GridHandles::default())
    .add_plugins(SymtropyScenePlugin::default())
    .add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
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
    // Camera: positioned to see the grid hanging below the pivot plane (y=0)
    // with clear depth perception from the front. ClearColor + ambient + sun
    // are now handled by SymtropyScenePlugin.
    commands.spawn(fixed_camera(
        Vec3::new(0.0, 1.5, 6.0),
        Vec3::new(0.0, -1.0, 0.0),
    ));
}

fn spawn_swarm(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<3>>,
    mut grid_handles: ResMut<GridHandles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // One sphere mesh shared across all bobs (cheap), per-bob material.
    let bob_mesh = meshes.add(Sphere::new(BOB_RADIUS).mesh().uv(16, 16));
    let pivot_mesh = meshes.add(Sphere::new(PIVOT_RADIUS).mesh().uv(8, 8));
    let pivot_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.55),
        perceptual_roughness: 0.9,
        ..default()
    });

    let half = (GRID as f64 - 1.0) * SPACING * 0.5;
    for i in 0..GRID {
        for j in 0..GRID {
            let pivot_x = (i as f64) * SPACING - half;
            let pivot_z = (j as f64) * SPACING - half;
            let bob = spawn_pendulum(
                &mut commands,
                &mut physics,
                pivot_x,
                pivot_z,
                (i, j),
                bob_mesh.clone(),
                pivot_mesh.clone(),
                pivot_material.clone(),
                &mut materials,
            );
            grid_handles.map.insert((i, j), bob);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_pendulum(
    commands: &mut Commands,
    physics: &mut SymtropyPhysics<3>,
    pivot_x: f64,
    pivot_z: f64,
    grid: (usize, usize),
    bob_mesh: Handle<Mesh>,
    pivot_mesh: Handle<Mesh>,
    pivot_material: Handle<StandardMaterial>,
    materials: &mut Assets<StandardMaterial>,
) -> BodyHandle {
    let pivot_y = 0.0_f64;
    let pivot_handle = physics.world.add_body(RigidBody::<3>::static_body(
        BodyHandle(0),
        Point::new([pivot_x, pivot_y, pivot_z]),
        Box::new(PhysicsSphere::new(Point::origin(), 0.01)),
    ));
    if let Some(p) = physics.world.body_mut(pivot_handle) {
        p.collision_mask = 0;
    }

    // Per-cell jitter: random direction in the XZ plane × random magnitude up
    // to PHASE_JITTER. Bob hangs from pivot at angle (theta_x, theta_z) from
    // straight down.
    let cell_hash = grid.0.wrapping_mul(7).wrapping_add(grid.1.wrapping_mul(13));
    let theta = (cell_hash as f64 * 0.37).sin().abs() * PHASE_JITTER;
    let dir_phase = cell_hash as f64 * 0.83;
    let (dx, dz) = (dir_phase.sin(), dir_phase.cos());
    // Bob position: pivot + ARM_LENGTH * sin(theta) in (dx, dz) direction,
    // and -ARM_LENGTH * cos(theta) in y.
    let bob_x = pivot_x + ARM_LENGTH * theta.sin() * dx;
    let bob_z = pivot_z + ARM_LENGTH * theta.sin() * dz;
    let bob_y = pivot_y - ARM_LENGTH * theta.cos();

    let bob_handle =
        physics
            .world
            .add_sphere(Point::new([bob_x, bob_y, bob_z]), BOB_RADIUS as f64, 1.0);
    if let Some(b) = physics.world.body_mut(bob_handle) {
        b.collision_mask = 0;
        b.linear_damping = HIGH_DAMP;
    }

    physics
        .world
        .add_constraint(Box::new(DistanceConstraint::<3> {
            body_a: pivot_handle,
            body_b: bob_handle,
            rest_length: ARM_LENGTH,
            stiffness: 1.0,
        }));

    physics
        .field
        .register(bob_handle, MAX_ENERGY, SANCTUARY_RADIUS);

    let pivot_pos = Vec3::new(pivot_x as f32, pivot_y as f32, pivot_z as f32);

    // Pivot visual (small grey sphere, doesn't move)
    commands.spawn((
        Mesh3d(pivot_mesh),
        MeshMaterial3d(pivot_material),
        Transform::from_translation(pivot_pos),
    ));

    // Per-bob material so color_by_phi can mutate it without touching neighbours.
    let bob_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.0, 1.0),
        perceptual_roughness: 0.6,
        metallic: 0.1,
        ..default()
    });

    commands.spawn((
        Pendulum {
            bob: bob_handle,
            pivot_pos,
            grid,
        },
        PhysicsBody::new(bob_handle, BOB_RADIUS),
        Mesh3d(bob_mesh),
        MeshMaterial3d(bob_material),
        Transform::from_xyz(bob_x as f32, bob_y as f32, bob_z as f32),
    ));

    bob_handle
}

fn update_phi_from_neighborhood(
    mut physics: ResMut<SymtropyPhysics<3>>,
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

fn phi_modulates_damping(mut physics: ResMut<SymtropyPhysics<3>>, pendulums: Query<&Pendulum>) {
    for p in &pendulums {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0);
        let damping = HIGH_DAMP + (LOW_DAMP - HIGH_DAMP) * phi_norm;
        if let Some(body) = physics.world.body_mut(p.bob) {
            body.linear_damping = damping;
        }
    }
}

/// Cast a ray from the cursor; find the nearest bob within `SHOCK_RAY_RADIUS`
/// of that ray (point-line distance); kick it with `SHOCK_VELOCITY` along +X.
fn shock_on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    grid_handles: Res<GridHandles>,
    mut physics: ResMut<SymtropyPhysics<3>>,
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
    let Ok(ray) = camera.viewport_to_world(camera_xform, cursor_pos) else {
        return;
    };

    let origin = ray.origin;
    let dir = ray.direction.as_vec3();

    let mut nearest: Option<(BodyHandle, f32)> = None;
    for (_, &h) in grid_handles.map.iter() {
        if let Some(body) = physics.world.body(h) {
            let pos = body.position();
            let bob_pos = Vec3::new(
                pos.coord(0) as f32,
                pos.coord(1) as f32,
                pos.coord(2) as f32,
            );
            // Distance from bob_pos to the ray (origin + t*dir):
            // d = ||(bob - origin) - ((bob - origin) · dir) * dir||
            let to_bob = bob_pos - origin;
            let t = to_bob.dot(dir);
            if t < 0.0 {
                continue; // bob is behind camera
            }
            let closest = origin + dir * t;
            let d = (bob_pos - closest).length();
            if d < SHOCK_RAY_RADIUS && nearest.map_or(true, |(_, nd)| d < nd) {
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

/// Update each bob's `StandardMaterial` base + emissive color from current Phi.
/// Blue (cool, Phi=0) → red (warm, Phi=max), with a faint emissive boost at
/// high Phi so the warm cells visibly glow.
fn color_by_phi(
    physics: Res<SymtropyPhysics<3>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(&Pendulum, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (p, mat_handle) in &query {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0) as f32;
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.base_color = Color::hsl(240.0 - phi_norm * 240.0, 1.0, 0.5);
            // Faint emissive scaled by Phi — high-Phi cells visibly glow.
            let emissive_mag = phi_norm * 0.4;
            let rgb = Color::hsl(240.0 - phi_norm * 240.0, 1.0, 0.5).to_linear();
            mat.emissive = LinearRgba::rgb(
                rgb.red * emissive_mag,
                rgb.green * emissive_mag,
                rgb.blue * emissive_mag,
            );
        }
    }
}

fn draw_arm_gizmo(mut gizmos: Gizmos, query: Query<(&Pendulum, &Transform)>) {
    for (p, t) in &query {
        gizmos.line(p.pivot_pos, t.translation, Color::srgb(0.4, 0.4, 0.45));
    }
}

fn headless_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut sched: ResMut<CaptureSchedule>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    while sched.fired < sched.schedule.len() && now >= sched.schedule[sched.fired] {
        let label = sched.schedule[sched.fired];
        let path = format!("{}/pswarm3d_t{:.1}.png", sched.dir, label);
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
