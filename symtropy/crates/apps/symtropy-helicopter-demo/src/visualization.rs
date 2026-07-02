// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: ground, SAR survivor marker, helicopter body + main rotor + tail
//! rotor, station-center halo.

use bevy::prelude::*;
use symthaea_helicopter::simulator::HelicopterPhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct HelicopterRoot;

/// Main rotor — rotates about its local Z axis at RPM-derived speed.
#[derive(Component)]
pub struct MainRotor;

/// Station-center halo on the ground (SAR target).
#[derive(Component)]
pub struct StationHalo;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(80.0, 80.0, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.28, 0.36, 0.24),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.05),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, -10.0, 30.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 280.0,
        affects_lightmapped_meshes: false,
    });

    // SAR survivor marker on the ground (orange X)
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.5, 0.1),
        emissive: LinearRgba::new(0.4, 0.2, 0.05, 1.0),
        ..default()
    });
    let bar_mesh = meshes.add(Cuboid::new(3.0, 0.3, 0.05));
    commands.spawn((
        Mesh3d(bar_mesh.clone()),
        MeshMaterial3d(marker_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 0.01)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
    ));
    commands.spawn((
        Mesh3d(bar_mesh),
        MeshMaterial3d(marker_mat),
        Transform::from_xyz(0.0, 0.0, 0.01)
            .with_rotation(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_4)),
    ));

    // Vertical station line + halo — shows the target column
    let halo_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.1, 0.9, 0.3, 0.25),
        emissive: LinearRgba::new(0.1, 0.4, 0.2, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.15, 20.0))),
        MeshMaterial3d(halo_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 10.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        StationHalo,
    ));
    // Circle on the ground at the station
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(2.0, 0.08))),
        MeshMaterial3d(halo_mat),
        Transform::from_xyz(0.0, 0.0, 0.04)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    ));

    // Helicopter body: cabin + tail boom + main rotor + tail rotor
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.80, 0.20, 0.15),
        metallic: 0.55,
        perceptual_roughness: 0.35,
        ..default()
    });
    let skid_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.15),
        ..default()
    });
    let rotor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.20, 0.22),
        metallic: 0.7,
        ..default()
    });

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 20.0),
            Visibility::Visible,
            HelicopterRoot,
        ))
        .with_children(|parent| {
            // Cabin (3 m × 1.4 m × 1.6 m)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(3.0, 1.4, 1.6))),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
            // Tail boom
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(4.0, 0.25, 0.25))),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(-3.0, 0.0, 0.2),
            ));
            // Tail fin (vertical stabilizer)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.8, 0.05, 0.8))),
                MeshMaterial3d(body_mat),
                Transform::from_xyz(-5.0, 0.0, 0.6),
            ));
            // Landing skids
            let skid_mesh = meshes.add(Cuboid::new(2.4, 0.08, 0.08));
            parent.spawn((
                Mesh3d(skid_mesh.clone()),
                MeshMaterial3d(skid_mat.clone()),
                Transform::from_xyz(0.0, 0.75, -1.0),
            ));
            parent.spawn((
                Mesh3d(skid_mesh),
                MeshMaterial3d(skid_mat),
                Transform::from_xyz(0.0, -0.75, -1.0),
            ));
            // Main rotor — two-blade disc centered above the cabin
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(5.0, 0.06, 0.04))),
                MeshMaterial3d(rotor_mat.clone()),
                Transform::from_xyz(0.0, 0.0, 1.0),
                MainRotor,
            ));
            // Tail rotor — vertical two-blade disc on the tail
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.04, 1.5, 0.05))),
                MeshMaterial3d(rotor_mat),
                Transform::from_xyz(-5.0, -0.2, 0.6),
            ));
        });
}

/// Position + orient the helicopter from the simulator state.
pub fn update_helicopter_visual(
    heli: Res<HelicopterResources>,
    mut query: Query<&mut Transform, (With<HelicopterRoot>, Without<MainRotor>)>,
) {
    let st = heli.simulator.state();
    let [w, x, y, z] = st.quaternion;
    let rot = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);
    for mut t in &mut query {
        t.translation = Vec3::new(
            st.position[0] as f32,
            st.position[1] as f32,
            st.position[2] as f32,
        );
        t.rotation = rot;
    }
}

/// Spin the main rotor blade about its local Z axis at the RPM rate.
pub fn update_rotor_spin(
    heli: Res<HelicopterResources>,
    mut query: Query<&mut Transform, With<MainRotor>>,
) {
    for mut t in &mut query {
        // Keep rotor translation at +z 1.0 above the body, rotate about body-local Z
        t.rotation = Quat::from_rotation_z(heli.last_rotor_spin_angle);
    }
}
