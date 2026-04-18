// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D visualization: ground + human torso stand-in + 6-DoF leg segments
//! rendered via forward-kinematics each frame.

use bevy::prelude::*;
use symthaea_exoskeleton::simulator::ExoskeletonPhysicsSimulator;

use crate::kinematics::{leg_chain, FOOT_LEN, HIP_WIDTH, SHIN_LEN, THIGH_LEN};
use crate::resources::*;

/// Marker tagging the segment index within one leg: `leg * 3 + joint`.
/// 0,1,2 = right thigh/shin/foot; 3,4,5 = left thigh/shin/foot.
#[derive(Component)]
pub struct LegSegment {
    pub index: usize,
}

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground (wide, gray-green)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 6.0, 0.04))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.40, 0.35),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.02),
    ));

    // Treadmill lane (darker strip)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.5, 1.2, 0.02))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.22, 0.22),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.001),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 9500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.0, -3.0, 4.0).looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Human torso (cuboid) above the hip line
    let hip_z = 1.0; // 1 m off the ground; leg chain hangs downward from here
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.25, 0.6))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.60, 0.70),
            metallic: 0.2,
            perceptual_roughness: 0.45,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, hip_z + 0.35),
    ));
    // Head
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.12))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.80, 0.70, 0.60),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, hip_z + 0.80),
    ));

    // Six leg segments: thigh/shin/foot × 2 legs. Each rendered as an
    // oriented cuboid whose transform is updated each frame.
    let exo_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.55, 0.85),
        metallic: 0.65,
        perceptual_roughness: 0.30,
        emissive: LinearRgba::new(0.02, 0.06, 0.12, 1.0),
        ..default()
    });

    let thigh_mesh = meshes.add(Cuboid::new(0.10, 0.10, THIGH_LEN));
    let shin_mesh = meshes.add(Cuboid::new(0.08, 0.08, SHIN_LEN));
    let foot_mesh = meshes.add(Cuboid::new(FOOT_LEN, 0.10, 0.05));

    // Right leg (y = +HIP_WIDTH / 2)
    commands.spawn((
        Mesh3d(thigh_mesh.clone()),
        MeshMaterial3d(exo_mat.clone()),
        Transform::from_xyz(0.0, HIP_WIDTH * 0.5, hip_z - THIGH_LEN * 0.5),
        LegSegment { index: 0 },
    ));
    commands.spawn((
        Mesh3d(shin_mesh.clone()),
        MeshMaterial3d(exo_mat.clone()),
        Transform::from_xyz(0.0, HIP_WIDTH * 0.5, hip_z - THIGH_LEN - SHIN_LEN * 0.5),
        LegSegment { index: 1 },
    ));
    commands.spawn((
        Mesh3d(foot_mesh.clone()),
        MeshMaterial3d(exo_mat.clone()),
        Transform::from_xyz(
            FOOT_LEN * 0.5,
            HIP_WIDTH * 0.5,
            hip_z - THIGH_LEN - SHIN_LEN,
        ),
        LegSegment { index: 2 },
    ));

    // Left leg (y = -HIP_WIDTH / 2)
    commands.spawn((
        Mesh3d(thigh_mesh),
        MeshMaterial3d(exo_mat.clone()),
        Transform::from_xyz(0.0, -HIP_WIDTH * 0.5, hip_z - THIGH_LEN * 0.5),
        LegSegment { index: 3 },
    ));
    commands.spawn((
        Mesh3d(shin_mesh),
        MeshMaterial3d(exo_mat.clone()),
        Transform::from_xyz(0.0, -HIP_WIDTH * 0.5, hip_z - THIGH_LEN - SHIN_LEN * 0.5),
        LegSegment { index: 4 },
    ));
    commands.spawn((
        Mesh3d(foot_mesh),
        MeshMaterial3d(exo_mat),
        Transform::from_xyz(
            FOOT_LEN * 0.5,
            -HIP_WIDTH * 0.5,
            hip_z - THIGH_LEN - SHIN_LEN,
        ),
        LegSegment { index: 5 },
    ));
}

/// Re-place + re-orient every leg segment from the current joint angles.
pub fn update_leg_visual(
    exo: Res<ExoskeletonResources>,
    mut query: Query<(&mut Transform, &LegSegment)>,
) {
    let st = exo.simulator.state();
    let hip_z = 1.0_f32;

    // Right leg (joints 0,1,2) hip at (0, +H/2, hip_z)
    let right = leg_chain(
        Vec3::new(0.0, HIP_WIDTH * 0.5, hip_z),
        1.0,
        st.joint_angles[0] as f32,
        st.joint_angles[1] as f32,
        st.joint_angles[2] as f32,
    );
    // Left leg (joints 3,4,5)
    let left = leg_chain(
        Vec3::new(0.0, -HIP_WIDTH * 0.5, hip_z),
        -1.0,
        st.joint_angles[3] as f32,
        st.joint_angles[4] as f32,
        st.joint_angles[5] as f32,
    );

    for (mut t, seg) in &mut query {
        let (pos, angle) = match seg.index {
            0 => right.thigh,
            1 => right.shin,
            2 => right.foot,
            3 => left.thigh,
            4 => left.shin,
            5 => left.foot,
            _ => continue,
        };
        t.translation = pos;
        // Rotate around world Y (sagittal-plane pitch)
        t.rotation = Quat::from_rotation_y(angle);
    }
}
