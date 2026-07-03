// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Walkable ground-level view — Step 3 of the "H3 hex Earth" plan.
//!
//! Only reachable by drilling all the way into a cell (see
//! `cell_entry::CellZoomTransition::arrived_at_floor`) — every other zoom
//! level stays the orbital globe view, by design.
//!
//! Honest scope: this is a placeholder platform (flat ground + a few
//! landmark props), not real terrain for the picked cell's actual Earth
//! coordinates — there's no heightmap/terrain data source wired up for
//! arbitrary lat/lon yet. Real terrain rendering is future work; this
//! proves the "arrive and walk around" loop end-to-end.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

/// Tags every entity spawned for the walkable scene, so it can all be
/// cleanly despawned on exit.
#[derive(Component)]
pub struct CellWalkEntity;

/// The walkable-view first-person camera.
#[derive(Component)]
pub struct CellWalkCamera;

/// The player's ground position (a separate entity from the camera so
/// movement and look can be updated independently, same pattern as
/// symtropy's own `rendering_3d.rs` FPS controller).
#[derive(Component)]
pub struct CellWalkPlayer;

/// Accumulated mouse-look yaw/pitch, mirroring symtropy's
/// `rendering_3d.rs::FirstPersonLook` pattern.
#[derive(Resource, Default)]
pub struct CellWalkLook {
    pub yaw: f32,
    pub pitch: f32,
}

const EYE_HEIGHT: f32 = 1.7;
const MOVE_SPEED: f32 = 4.0;
const MOUSE_SENSITIVITY: f32 = 0.0025;
const PITCH_LIMIT: f32 = 1.45;

/// Spawns the placeholder walkable scene: flat ground, a few box props so
/// it doesn't read as an empty void, a sun, and a first-person camera
/// starting at the origin.
pub fn setup_cell_walk(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(CellWalkLook::default());

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(40.0, 40.0));
    let ground_material = materials.add(StandardMaterial {
        // Placeholder tone — not sampled from the actual picked location's
        // real terrain color (no heightmap/terrain source wired up yet).
        base_color: Color::srgb(0.28, 0.36, 0.22),
        perceptual_roughness: 0.95,
        ..default()
    });
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_material),
        Transform::IDENTITY,
        CellWalkEntity,
    ));

    // A handful of simple landmark props purely so the space has visible
    // scale/orientation cues — not meant to represent anything real.
    for (i, (x, z)) in [(-6.0, -5.0), (7.0, -3.0), (-4.0, 8.0), (5.0, 6.0)]
        .into_iter()
        .enumerate()
    {
        let height = 1.6 + i as f32 * 0.4;
        let prop_mesh = meshes.add(Cuboid::new(1.0, height, 1.0));
        let prop_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.46, 0.4),
            perceptual_roughness: 0.8,
            ..default()
        });
        commands.spawn((
            Mesh3d(prop_mesh),
            MeshMaterial3d(prop_material),
            Transform::from_xyz(x, height * 0.5, z),
            CellWalkEntity,
        ));
    }

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.5, 0.0)),
        CellWalkEntity,
    ));
    commands.spawn((
        AmbientLight {
            color: Color::WHITE,
            brightness: 250.0,
            ..default()
        },
        CellWalkEntity,
    ));

    commands.spawn((Transform::IDENTITY, CellWalkPlayer, CellWalkEntity));

    commands.spawn((
        Camera3d::default(),
        // Same LUT-free tonemapping fix as the orbital camera and
        // symtropy's own 3D FPS camera — TonyMcMapFace washes the frame
        // out without the tonemapping_luts feature.
        Tonemapping::SomewhatBoringDisplayTransform,
        Transform::from_xyz(0.0, EYE_HEIGHT, 0.0),
        CellWalkCamera,
        CellWalkEntity,
    ));

    info!("[atlas] Entered walkable cell view (placeholder ground — real terrain is future work)");
}

/// Despawns everything tagged `CellWalkEntity` — called on exiting the
/// walkable view, symmetric with `setup_cell_walk`.
pub fn cleanup_cell_walk(mut commands: Commands, query: Query<Entity, With<CellWalkEntity>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

pub fn cell_walk_cursor_grab_system(mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    let Ok(mut cursor) = windows.single_mut() else {
        return;
    };
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
}

pub fn cell_walk_cursor_release_system(
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = windows.single_mut() else {
        return;
    };
    cursor.grab_mode = CursorGrabMode::None;
    cursor.visible = true;
}

pub fn cell_walk_mouse_look_system(
    mut motion: MessageReader<MouseMotion>,
    mut look: ResMut<CellWalkLook>,
) {
    for event in motion.read() {
        look.yaw -= event.delta.x * MOUSE_SENSITIVITY;
        look.pitch =
            (look.pitch - event.delta.y * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }
}

pub fn cell_walk_movement_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    look: Res<CellWalkLook>,
    mut player_q: Query<&mut Transform, (With<CellWalkPlayer>, Without<CellWalkCamera>)>,
) {
    let Ok(mut player_tf) = player_q.single_mut() else {
        return;
    };
    let forward = Vec3::new(look.yaw.sin(), 0.0, look.yaw.cos());
    let right = Vec3::new(look.yaw.cos(), 0.0, -look.yaw.sin());

    let mut direction = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        direction += forward;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        direction -= forward;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        direction += right;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        direction -= right;
    }

    if direction != Vec3::ZERO {
        player_tf.translation += direction.normalize() * MOVE_SPEED * time.delta_secs();
    }
}

pub fn cell_walk_camera_sync_system(
    look: Res<CellWalkLook>,
    player_q: Query<&Transform, (With<CellWalkPlayer>, Without<CellWalkCamera>)>,
    mut camera_q: Query<&mut Transform, (With<CellWalkCamera>, Without<CellWalkPlayer>)>,
) {
    let Ok(player_tf) = player_q.single() else {
        return;
    };
    let Ok(mut camera_tf) = camera_q.single_mut() else {
        return;
    };

    let eye = player_tf.translation + Vec3::new(0.0, EYE_HEIGHT, 0.0);
    let forward = Vec3::new(
        look.yaw.sin() * look.pitch.cos(),
        look.pitch.sin(),
        look.yaw.cos() * look.pitch.cos(),
    );
    *camera_tf = Transform::from_translation(eye).looking_at(eye + forward, Vec3::Y);
}
