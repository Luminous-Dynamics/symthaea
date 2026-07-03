// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: long corridor + quadruped base + 4 legs × 3 links + optional
//! rough-terrain cue tile.
//!
//! Leg-index convention (matches `QuadrupedState::standing` joint layout):
//!   Legs 0,1,2,3 each use 3 consecutive joints:
//!     [leg*3 + 0] = hip_yaw   (rotation about body-Z, side-swing)
//!     [leg*3 + 1] = hip_pitch (rotation about body-Y, forward swing)
//!     [leg*3 + 2] = knee_pitch (rotation about body-Y, bend)
//!
//! Legs in 3D are placed at body-frame (±0.25, ±0.18) from the base
//! center; leg 0 = FR, 1 = FL, 2 = BR, 3 = BL (any consistent mapping
//! works for the demo — the simulator doesn't assign body sides).

use bevy::prelude::*;
use symthaea_quadruped::simulator::QuadrupedPhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct BaseVisual;

/// Tagged by segment: `leg_index * 2 + segment_index` where segment 0 =
/// thigh (hip→knee), segment 1 = shin (knee→foot).
#[derive(Component)]
pub struct LegSegment {
    pub leg: usize,
    pub seg: usize,
}

#[derive(Component)]
pub struct TerrainCue;

const HIP_FWD: f32 = 0.25; // forward offset of front hips in body frame
const HIP_SIDE: f32 = 0.18; // lateral offset of hips
const THIGH_LEN: f32 = 0.20;
const SHIN_LEN: f32 = 0.20;

/// Hip attachment points in body frame: leg 0 = FR, 1 = FL, 2 = BR, 3 = BL.
fn hip_origin(leg: usize) -> Vec3 {
    let fwd = if leg < 2 { HIP_FWD } else { -HIP_FWD };
    let side = if leg % 2 == 0 { -HIP_SIDE } else { HIP_SIDE };
    Vec3::new(fwd, side, 0.0)
}

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground: long corridor so forward walking has somewhere to go
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(40.0, 4.0, 0.05))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.34, 0.32),
            ..default()
        })),
        Transform::from_xyz(15.0, 0.0, -0.025),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 9500.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, -5.0, 8.0).looking_at(Vec3::new(3.0, 0.0, 0.3), Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 290.0,
        affects_lightmapped_meshes: false,
    });

    // Body materials
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.88),
        metallic: 0.55,
        perceptual_roughness: 0.35,
        ..default()
    });
    let leg_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.50, 0.75),
        metallic: 0.45,
        perceptual_roughness: 0.40,
        ..default()
    });

    // Base (torso) — ~0.60 m × 0.30 m × 0.12 m. Parent transform is
    // authoritative; children (legs) are repositioned in world frame each
    // frame because the sim reports joint angles independently of base
    // orientation, so we compute leg world positions outside any hierarchy.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.60, 0.30, 0.12))),
        MeshMaterial3d(base_mat),
        Transform::from_xyz(0.0, 0.0, 0.35),
        BaseVisual,
    ));

    // Head marker (tiny sphere at the front of the base to show facing)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.06))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.65, 0.25),
            emissive: LinearRgba::new(0.25, 0.15, 0.05, 1.0),
            ..default()
        })),
        Transform::from_xyz(0.32, 0.0, 0.38),
    ));

    // Four legs × two visual segments (thigh + shin). Each segment is a
    // short cuboid the plugin repositions each frame.
    let thigh_mesh = meshes.add(Cuboid::new(0.05, 0.05, THIGH_LEN));
    let shin_mesh = meshes.add(Cuboid::new(0.04, 0.04, SHIN_LEN));
    for leg in 0..4 {
        commands.spawn((
            Mesh3d(thigh_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::from_xyz(0.0, 0.0, 0.2),
            LegSegment { leg, seg: 0 },
        ));
        commands.spawn((
            Mesh3d(shin_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::from_xyz(0.0, 0.0, 0.0),
            LegSegment { leg, seg: 1 },
        ));
    }

    // Terrain cue: translucent yellow tile whose alpha tracks roughness.
    // Sits on the ground just ahead of the base by default; moves with
    // `update_terrain_cue_visual`.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 2.0, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.75, 0.15, 0.0),
            emissive: LinearRgba::new(0.35, 0.25, 0.05, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(1.5, 0.0, 0.02),
        TerrainCue,
    ));
}

/// Place the base from simulator state.
pub fn update_base_visual(
    q: Res<QuadrupedResources>,
    mut query: Query<&mut Transform, (With<BaseVisual>, Without<LegSegment>)>,
) {
    let st = q.simulator.state();
    let [w, x, y, z] = st.base_quaternion;
    let rot = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);
    for mut t in &mut query {
        t.translation = Vec3::new(
            st.base_position[0] as f32,
            st.base_position[1] as f32,
            st.base_position[2] as f32,
        );
        t.rotation = rot;
    }
}

/// Place each leg segment via a simple sagittal-plane FK from its hip.
pub fn update_leg_visual(
    q: Res<QuadrupedResources>,
    mut query: Query<(&mut Transform, &LegSegment), Without<BaseVisual>>,
) {
    let st = q.simulator.state();
    let base_world = Vec3::new(
        st.base_position[0] as f32,
        st.base_position[1] as f32,
        st.base_position[2] as f32,
    );
    let [w, x, y, z] = st.base_quaternion;
    let base_rot = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);

    for (mut t, seg) in &mut query {
        let hip_body = hip_origin(seg.leg);
        let hip_world = base_world + base_rot * hip_body;

        let hip_pitch = st.joint_angles[seg.leg * 3 + 1] as f32;
        let knee = st.joint_angles[seg.leg * 3 + 2] as f32;

        // Thigh: downward along (sin(hip_pitch), 0, -cos(hip_pitch)) in
        // body frame, rotated by base_rot into world.
        let thigh_dir_body = Vec3::new(hip_pitch.sin(), 0.0, -hip_pitch.cos());
        let thigh_dir_world = base_rot * thigh_dir_body;
        let knee_world = hip_world + thigh_dir_world * THIGH_LEN;

        // Shin: downward along cumulative angle (hip_pitch + knee)
        let shin_angle = hip_pitch + knee;
        let shin_dir_body = Vec3::new(shin_angle.sin(), 0.0, -shin_angle.cos());
        let shin_dir_world = base_rot * shin_dir_body;

        match seg.seg {
            0 => {
                t.translation = hip_world + thigh_dir_world * (THIGH_LEN * 0.5);
                t.rotation = base_rot * Quat::from_rotation_y(hip_pitch);
            }
            1 => {
                t.translation = knee_world + shin_dir_world * (SHIN_LEN * 0.5);
                t.rotation = base_rot * Quat::from_rotation_y(shin_angle);
            }
            _ => {}
        }
    }
}

/// Fade the terrain cue alpha with current roughness.
pub fn update_terrain_cue_visual(
    q: Res<QuadrupedResources>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(&mut Transform, &MeshMaterial3d<StandardMaterial>), With<TerrainCue>>,
) {
    let r = q.last_terrain_roughness as f32;
    let base_x = q.simulator.state().base_position[0] as f32;
    for (mut transform, mat_handle) in &mut query {
        // Place the cue tile just ahead of the robot (1.5 m forward along X).
        transform.translation = Vec3::new(base_x + 1.5, 0.0, 0.02);
        if let Some(mut mat) = materials.get_mut(&mat_handle.0) {
            let alpha = (0.05 + 0.60 * r).clamp(0.0, 0.7);
            mat.base_color = Color::srgba(1.0, 0.75, 0.15, alpha);
        }
    }
}
