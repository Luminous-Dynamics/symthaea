// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal humanoid visualization. Instead of hand-rolling 21-DoF FK, we
//! place:
//!
//! - **Torso**: a cuboid at `root_height`, rotated by `root_quaternion`.
//! - **Head**: a sphere offset above the torso.
//! - **Four extremities** (right_hand, left_hand, right_foot, left_foot):
//!   spheres at the world-frame positions the simulator already computes
//!   into `state.extremities[0..12]`.
//!
//! No connector bones — the tetrahedron of hands+feet around the torso
//! reads as a human silhouette when the extremities move.

use bevy::prelude::*;
use symthaea_humanoid::simulator::HumanoidPhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct TorsoVisual;

#[derive(Component)]
pub struct HeadVisual;

#[derive(Component)]
pub struct Extremity {
    /// 0 = right hand, 1 = left hand, 2 = right foot, 3 = left foot.
    pub index: usize,
}

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 8.0, 0.05))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.40, 0.35),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.025),
    ));

    // Start circle (visual reference for "standing position")
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.6, 0.02))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.1, 0.9, 0.3, 0.25),
            emissive: LinearRgba::new(0.1, 0.4, 0.2, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.005),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 9500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(3.0, -3.0, 5.0).looking_at(Vec3::new(0.0, 0.0, 1.2), Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Torso: a humanoid-proportioned cuboid (~0.40 m × 0.22 m × 0.60 m)
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.70, 0.85),
        metallic: 0.25,
        perceptual_roughness: 0.45,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.40, 0.22, 0.60))),
        MeshMaterial3d(body_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 1.3),
        TorsoVisual,
    ));

    // Head: sphere above the torso (follows torso)
    let head_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.80, 0.70, 0.60),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.13))),
        MeshMaterial3d(head_mat),
        Transform::from_xyz(0.0, 0.0, 1.7),
        HeadVisual,
    ));

    // Four extremities — distinct colors help read motion
    let colors = [
        Color::srgb(0.95, 0.55, 0.35), // right hand (warm)
        Color::srgb(0.95, 0.75, 0.35), // left hand
        Color::srgb(0.40, 0.70, 0.95), // right foot
        Color::srgb(0.40, 0.90, 0.75), // left foot
    ];
    for (i, color) in colors.iter().enumerate() {
        let mat = materials.add(StandardMaterial {
            base_color: *color,
            emissive: LinearRgba::new(0.05, 0.05, 0.05, 1.0),
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.11))),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, 0.0, 0.5),
            Extremity { index: i },
        ));
    }
}

/// Re-place torso, head, and four extremities from simulator state.
pub fn update_humanoid_visual(
    h: Res<HumanoidResources>,
    mut torso_q: Query<
        &mut Transform,
        (With<TorsoVisual>, Without<HeadVisual>, Without<Extremity>),
    >,
    mut head_q: Query<&mut Transform, (With<HeadVisual>, Without<TorsoVisual>, Without<Extremity>)>,
    mut ext_q: Query<(&mut Transform, &Extremity), (Without<TorsoVisual>, Without<HeadVisual>)>,
) {
    let st = h.simulator.state();
    let [w, x, y, z] = st.root_quaternion;
    let rot = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);

    // Torso sits at (0, 0, root_height) — the simulator keeps the humanoid
    // roughly centered laterally, so we just render at its reported root_height.
    for mut t in &mut torso_q {
        t.translation = Vec3::new(0.0, 0.0, st.root_height as f32);
        t.rotation = rot;
    }
    // Head: `head_height` is authoritative from the simulator.
    for mut t in &mut head_q {
        t.translation = Vec3::new(0.0, 0.0, st.head_height as f32);
        t.rotation = rot;
    }
    // Extremities: state.extremities[0..12] = r_hand(3) + l_hand(3) + r_foot(3) + l_foot(3)
    for (mut t, ext) in &mut ext_q {
        if ext.index >= 4 {
            continue;
        }
        let base = ext.index * 3;
        if st.extremities.len() >= base + 3 {
            t.translation = Vec3::new(
                st.extremities[base] as f32,
                st.extremities[base + 1] as f32,
                st.extremities[base + 2] as f32,
            );
        }
    }
}
