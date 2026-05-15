// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm_4d` — 4D port with hyperplane slicing.
//!
//! Same Phi-coupled physics as the 2D and 3D variants, but the world is 4D.
//! Pendulums are arranged in 5×5 columns (XZ plane) × 3 W-layers, hanging from
//! pivots in -Y. Per-cell jitter perturbs the initial bob position in X, Z,
//! AND W directions — so bobs can swing across the W axis and drift between
//! layers.
//!
//! Visualisation: a hyperplane perpendicular to the W axis at `w_slice` is
//! the "viewport." Bobs near the slice are fully opaque; bobs further along
//! W fade to invisible. Press `[` and `]` to move the slice along W and watch
//! one layer fade in as another fades out.
//!
//! The 4D physics keeps running for ALL bobs, even ones outside the slice.
//! The full simulation lives in 4D; we just see one cross-section at a time.
//!
//! Headless capture: `PENDULUM_CAPTURE_DIR=/some/dir cargo run ...` schedules
//! PNG screenshots and AppExit. The slice is moved automatically across the
//! capture window so each PNG shows a different W cross-section.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_bevy::{PhysicsBody, SymtropyPhysics, SymtropyPhysicsPlugin};
use symtropy_bevy_scene::{fixed_camera, SymtropyScenePlugin};
use symtropy_devconsole::SymtropyDevConsolePlugin;
use symtropy_math::{Point, Sphere as PhysicsSphere};
use symtropy_physics::constraint::DistanceConstraint;
use symtropy_physics::{BodyHandle, RigidBody};
use symtropy_render_bridge::material::{
    NdSlicingExtension, NdSlicingMaterial, NdSlicingPlugin, NdSlicingSettings,
};
use symtropy_render_bridge::projection::Projector4D;

const ARM_LENGTH: f64 = 1.0;
const BOB_RADIUS: f32 = 0.10;
const PIVOT_RADIUS: f32 = 0.04;
const GRID_XZ: usize = 5;
const GRID_W: usize = 3;
const SPACING: f64 = 0.6;
const W_SPACING: f64 = 1.0; // distance between W-layers
const SANCTUARY_RADIUS: f64 = 0.5;
const MAX_ENERGY: f64 = 100.0;
const VARIANCE_SCALE: f64 = 1.0;
const PHI_NORMALIZE: f64 = 0.314;
const LOW_DAMP: f64 = 0.001;
const HIGH_DAMP: f64 = 0.5;
const PHASE_JITTER: f64 = 0.30;
const SLICE_THICKNESS: f64 = 0.45; // ±0.45m around w_slice fully visible
const W_SLICE_STEP: f64 = 0.1; // [/] keys move slice by this much per press

#[derive(Component)]
struct Pendulum {
    bob: BodyHandle,
    pivot_pos: Vec3, // Bevy world position (x, y, z) of the pivot — w dropped
    grid: (usize, usize, usize), // (i, j, w_layer)
}

#[derive(Component)]
struct WSliceText;

#[derive(Resource, Default)]
struct GridHandles {
    map: HashMap<(usize, usize, usize), BodyHandle>,
}

#[derive(Resource)]
struct CaptureSchedule {
    dir: String,
    schedule: Vec<(f32, f64)>, // (real time, w_slice override)
    fired: usize,
    exit_at: f32,
}

fn capture_schedule_from_env() -> Option<CaptureSchedule> {
    let dir = std::env::var("PENDULUM_CAPTURE_DIR").ok()?;
    Some(CaptureSchedule {
        dir,
        // Three captures: parked on each of the three W-layers so each PNG
        // shows a different cross-section. Times allow physics to settle
        // between slice moves.
        schedule: vec![(2.0, -W_SPACING), (4.5, 0.0), (7.0, W_SPACING)],
        fired: 0,
        exit_at: 8.5,
    })
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Symtropy: Pendulum Swarm 4D (hyperplane slicing)".into(),
            resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(GridHandles::default())
    .insert_resource(Projector4D::new(0.0, SLICE_THICKNESS, 1.0))
    .add_plugins(SymtropyScenePlugin::default())
    .add_plugins(SymtropyPhysicsPlugin::<4>::with_gravity([
        0.0, -9.81, 0.0, 0.0,
    ]))
    .add_plugins(SymtropyDevConsolePlugin)
    .add_plugins(NdSlicingPlugin)
    .add_systems(Startup, (setup_camera, spawn_swarm, setup_hud))
    .add_systems(
        FixedUpdate,
        (update_phi_from_neighborhood, phi_modulates_damping).chain(),
    )
    .add_systems(
        Update,
        (
            handle_w_slice_input,
            projector_to_shader_system,
            color_by_phi,
            draw_arm_gizmo,
            update_hud,
        ),
    );

    if let Some(sched) = capture_schedule_from_env() {
        app.insert_resource(sched);
        app.add_systems(Update, headless_capture);
    }

    app.run();
}

fn setup_camera(mut commands: Commands) {
    // ClearColor + ambient + sun handled by SymtropyScenePlugin.
    commands.spawn(fixed_camera(
        Vec3::new(0.0, 1.5, 5.0),
        Vec3::new(0.0, -1.0, 0.0),
    ));
}

fn setup_hud(mut commands: Commands) {
    commands.spawn((
        Text::new("w_slice = 0.00\n[ / ] to move slice"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgba(0.85, 0.9, 1.0, 0.9)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        WSliceText,
    ));
}

fn spawn_swarm(
    mut commands: Commands,
    mut physics: ResMut<SymtropyPhysics<4>>,
    mut grid_handles: ResMut<GridHandles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<NdSlicingMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    let bob_mesh = meshes.add(Sphere::new(BOB_RADIUS).mesh().uv(16, 16));
    let pivot_mesh = meshes.add(Sphere::new(PIVOT_RADIUS).mesh().uv(8, 8));
    let pivot_material = standard_materials.add(StandardMaterial {
        base_color: Color::srgba(0.5, 0.5, 0.55, 1.0),
        perceptual_roughness: 0.9,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let half_xz = (GRID_XZ as f64 - 1.0) * SPACING * 0.5;
    let half_w = (GRID_W as f64 - 1.0) * W_SPACING * 0.5;
    for i in 0..GRID_XZ {
        for j in 0..GRID_XZ {
            for k in 0..GRID_W {
                let pivot_x = (i as f64) * SPACING - half_xz;
                let pivot_z = (j as f64) * SPACING - half_xz;
                let pivot_w = (k as f64) * W_SPACING - half_w;
                let bob = spawn_pendulum(
                    &mut commands,
                    &mut physics,
                    pivot_x,
                    pivot_z,
                    pivot_w,
                    (i, j, k),
                    bob_mesh.clone(),
                    pivot_mesh.clone(),
                    pivot_material.clone(),
                    &mut materials,
                );
                grid_handles.map.insert((i, j, k), bob);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_pendulum(
    commands: &mut Commands,
    physics: &mut SymtropyPhysics<4>,
    pivot_x: f64,
    pivot_z: f64,
    pivot_w: f64,
    grid: (usize, usize, usize),
    bob_mesh: Handle<Mesh>,
    pivot_mesh: Handle<Mesh>,
    pivot_material: Handle<StandardMaterial>,
    materials: &mut Assets<NdSlicingMaterial>,
) -> BodyHandle {
    let pivot_y = 0.0_f64;
    let pivot_handle = physics.world.add_body(RigidBody::<4>::static_body(
        BodyHandle(0),
        Point::new([pivot_x, pivot_y, pivot_z, pivot_w]),
        Box::new(PhysicsSphere::new(Point::origin(), 0.01)),
    ));
    if let Some(p) = physics.world.body_mut(pivot_handle) {
        p.collision_mask = 0;
    }

    // Per-cell jitter: random unit vector in 3D (X, Z, W) × random magnitude.
    // Bob hangs from pivot at angle theta from straight down, perturbed in
    // (X, Z, W) directions. This means bobs CAN drift along the W axis as
    // they swing.
    let cell_hash = grid
        .0
        .wrapping_mul(7)
        .wrapping_add(grid.1.wrapping_mul(13))
        .wrapping_add(grid.2.wrapping_mul(31));
    let theta = (cell_hash as f64 * 0.37).sin().abs() * PHASE_JITTER;
    let phase_a = cell_hash as f64 * 0.83;
    let phase_b = cell_hash as f64 * 1.27;
    let (dx, daxis) = (phase_a.sin(), phase_a.cos());
    let (dz, dw) = (daxis * phase_b.sin(), daxis * phase_b.cos());
    // Normalise (dx, dz, dw) to unit length.
    let len = (dx * dx + dz * dz + dw * dw).sqrt().max(1e-9);
    let (ux, uz, uw) = (dx / len, dz / len, dw / len);
    let bob_x = pivot_x + ARM_LENGTH * theta.sin() * ux;
    let bob_z = pivot_z + ARM_LENGTH * theta.sin() * uz;
    let bob_w = pivot_w + ARM_LENGTH * theta.sin() * uw;
    let bob_y = pivot_y - ARM_LENGTH * theta.cos();

    let bob_handle = physics.world.add_sphere(
        Point::new([bob_x, bob_y, bob_z, bob_w]),
        BOB_RADIUS as f64,
        1.0,
    );
    if let Some(b) = physics.world.body_mut(bob_handle) {
        b.collision_mask = 0;
        b.linear_damping = HIGH_DAMP;
    }

    physics
        .world
        .add_constraint(Box::new(DistanceConstraint::<4> {
            body_a: pivot_handle,
            body_b: bob_handle,
            rest_length: ARM_LENGTH,
            stiffness: 1.0,
        }));

    physics
        .field
        .register(bob_handle, MAX_ENERGY, SANCTUARY_RADIUS);

    let pivot_pos = Vec3::new(pivot_x as f32, pivot_y as f32, pivot_z as f32);

    commands.spawn((
        Mesh3d(pivot_mesh),
        MeshMaterial3d(pivot_material),
        Transform::from_translation(pivot_pos),
    ));

    let bob_material = materials.add(NdSlicingMaterial {
        base: StandardMaterial {
            base_color: Color::srgba(0.0, 0.0, 1.0, 1.0),
            perceptual_roughness: 0.6,
            metallic: 0.1,
            alpha_mode: AlphaMode::Blend,
            ..default()
        },
        extension: NdSlicingExtension {
            settings: NdSlicingSettings {
                w_pos: bob_w as f32,
                w_slice: 0.0,
                slice_thickness: SLICE_THICKNESS as f32,
                edge_fade: 1.0,
                time: 0.0,
            },
        },
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
    mut physics: ResMut<SymtropyPhysics<4>>,
    grid_handles: Res<GridHandles>,
    pendulums: Query<&Pendulum>,
) {
    for p in &pendulums {
        let (gi, gj, gk) = p.grid;
        let mut sum_v = 0.0;
        let mut sum_v_sq = 0.0;
        let mut count = 0;
        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                for dk in -1i32..=1 {
                    let ni = gi as i32 + di;
                    let nj = gj as i32 + dj;
                    let nk = gk as i32 + dk;
                    if ni < 0 || nj < 0 || nk < 0 {
                        continue;
                    }
                    if ni >= GRID_XZ as i32 || nj >= GRID_XZ as i32 || nk >= GRID_W as i32 {
                        continue;
                    }
                    if let Some(&h) = grid_handles
                        .map
                        .get(&(ni as usize, nj as usize, nk as usize))
                    {
                        if let Some(body) = physics.world.body(h) {
                            let v = body.linear_velocity.norm();
                            sum_v += v;
                            sum_v_sq += v * v;
                            count += 1;
                        }
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

fn phi_modulates_damping(mut physics: ResMut<SymtropyPhysics<4>>, pendulums: Query<&Pendulum>) {
    for p in &pendulums {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0);
        let damping = HIGH_DAMP + (LOW_DAMP - HIGH_DAMP) * phi_norm;
        if let Some(body) = physics.world.body_mut(p.bob) {
            body.linear_damping = damping;
        }
    }
}

fn handle_w_slice_input(keys: Res<ButtonInput<KeyCode>>, mut projector: ResMut<Projector4D>) {
    if keys.just_pressed(KeyCode::BracketLeft) {
        projector.w_slice -= W_SLICE_STEP;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        projector.w_slice += W_SLICE_STEP;
    }
}

/// Read each bob's W coordinate from physics, compute alpha against the
/// current `w_slice`, and mutate the StandardMaterial's base_color.alpha.
/// Sync the 4D projector settings to all materials using the slicing shader.
fn projector_to_shader_system(
    physics: Res<SymtropyPhysics<4>>,
    projector: Res<Projector4D>,
    mut materials: ResMut<Assets<NdSlicingMaterial>>,
    query: Query<(&Pendulum, &MeshMaterial3d<NdSlicingMaterial>)>,
    time: Res<Time>,
) {
    for (p, mat_handle) in &query {
        let Some(body) = physics.world.body(p.bob) else {
            continue;
        };
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.extension.settings.w_pos = body.position().coord(3) as f32;
            mat.extension.settings.w_slice = projector.w_slice as f32;
            mat.extension.settings.slice_thickness = projector.slice_thickness as f32;
            mat.extension.settings.time = time.elapsed_secs();
        }
    }
}

fn color_by_phi(
    physics: Res<SymtropyPhysics<4>>,
    mut materials: ResMut<Assets<NdSlicingMaterial>>,
    query: Query<(&Pendulum, &MeshMaterial3d<NdSlicingMaterial>)>,
) {
    for (p, mat_handle) in &query {
        let phi = physics.field.phi(p.bob);
        let phi_norm = (phi / PHI_NORMALIZE).clamp(0.0, 1.0) as f32;
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let new_color = Color::hsl(240.0 - phi_norm * 240.0, 1.0, 0.5);
            mat.base.base_color = new_color;
        }
    }
}

fn draw_arm_gizmo(
    mut gizmos: Gizmos,
    physics: Res<SymtropyPhysics<4>>,
    projector: Res<Projector4D>,
    query: Query<(&Pendulum, &Transform)>,
) {
    for (p, t) in &query {
        let Some(body) = physics.world.body(p.bob) else {
            continue;
        };
        let w = body.position().coord(3);
        // Gizmos are still CPU-culled using the projector logic
        let d = (w - projector.w_slice).abs();
        if d >= projector.slice_thickness {
            continue;
        }
        let alpha = (1.0 - d / projector.slice_thickness) as f32;
        gizmos.line(
            p.pivot_pos,
            t.translation,
            Color::srgba(0.4, 0.4, 0.45, alpha),
        );
    }
}

fn update_hud(projector: Res<Projector4D>, mut text_query: Query<&mut Text, With<WSliceText>>) {
    if !projector.is_changed() {
        return;
    }
    for mut text in &mut text_query {
        text.0 = format!("w_slice = {:+.2}\n[ / ] to move slice", projector.w_slice);
    }
}

fn headless_capture(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut sched: ResMut<CaptureSchedule>,
    mut projector: ResMut<Projector4D>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    while sched.fired < sched.schedule.len() && now >= sched.schedule[sched.fired].0 {
        let (_label, w_override) = sched.schedule[sched.fired];
        // Move the slice plane just before capturing.
        projector.w_slice = w_override;
        let path = format!(
            "{}/pswarm4d_t{:.1}_w{:+.1}.png",
            sched.dir, _label, w_override
        );
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!("queued screenshot: {} (w_slice={})", path, w_override);
        sched.fired += 1;
    }
    if now >= sched.exit_at {
        info!("headless capture done — exiting");
        exit.write(AppExit::Success);
    }
}
