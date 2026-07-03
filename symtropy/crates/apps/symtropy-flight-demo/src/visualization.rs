// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: ground, quadrotor body + rotors, and waypoint markers.

use bevy::prelude::*;
use symthaea_multirotor::simulator::PhysicsSimulator;

use crate::resources::*;

/// Root marker for the quadrotor visual (positioned + rotated each frame).
#[derive(Component)]
pub struct QuadrotorRoot;

/// Marker for each waypoint sphere.
#[derive(Component)]
pub struct WaypointMarker;

/// Marker for the "current" waypoint highlight.
#[derive(Component)]
pub struct ActiveWaypointHighlight;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    waypoints: Res<WaypointPath>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(10.0, 10.0, 0.05))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.22, 0.25),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.025),
    ));

    // Directional light + ambient
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(3.0, -3.0, 8.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 260.0,
        affects_lightmapped_meshes: false,
    });

    // Quadrotor: parent root with 4 rotors + a central body as children.
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.9),
        metallic: 0.6,
        perceptual_roughness: 0.4,
        ..default()
    });
    let arm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.15),
        ..default()
    });
    let rotor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.6, 1.0),
        emissive: LinearRgba::new(0.0, 0.15, 0.3, 1.0),
        ..default()
    });

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 1.5),
            Visibility::Visible,
            QuadrotorRoot,
        ))
        .with_children(|parent| {
            // Central body (small cuboid)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.10, 0.10, 0.04))),
                MeshMaterial3d(body_mat),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));

            // X-config arms + rotors — 4 booms at ±(0.12, 0.12)
            let arm_offset = 0.12_f32;
            let rotor_mesh = meshes.add(Cylinder::new(0.05, 0.008));
            let arm_mesh = meshes.add(Cuboid::new(0.17, 0.012, 0.012));

            for (i, (sx, sy)) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
                .iter()
                .enumerate()
            {
                let ax = arm_offset * 0.5 * sx;
                let ay = arm_offset * 0.5 * sy;
                // Arm (rotated 45° around Z)
                parent.spawn((
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(arm_mat.clone()),
                    Transform::from_xyz(ax, ay, 0.0).with_rotation(Quat::from_rotation_z(
                        std::f32::consts::FRAC_PI_4 * if (i == 0) || (i == 2) { 1.0 } else { -1.0 },
                    )),
                ));
                // Rotor disc at the arm tip
                parent.spawn((
                    Mesh3d(rotor_mesh.clone()),
                    MeshMaterial3d(rotor_mat.clone()),
                    Transform::from_xyz(arm_offset * sx, arm_offset * sy, 0.015),
                ));
            }
        });

    // Waypoints as faint translucent spheres, active one gets an emissive highlight.
    let wp_mat_idle = materials.add(StandardMaterial {
        base_color: Color::srgba(0.6, 0.6, 0.7, 0.55),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    for wp in waypoints.waypoints.iter() {
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.08))),
            MeshMaterial3d(wp_mat_idle.clone()),
            Transform::from_xyz(wp[0] as f32, wp[1] as f32, wp[2] as f32),
            WaypointMarker,
        ));
    }

    // Active-waypoint halo (larger, brighter; re-positioned each frame).
    let halo_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.1, 0.9, 0.3, 0.35),
        emissive: LinearRgba::new(0.1, 0.4, 0.2, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.20))),
        MeshMaterial3d(halo_mat),
        Transform::from_xyz(0.0, 0.0, 1.5),
        ActiveWaypointHighlight,
    ));
}

/// Position + orient the quadrotor root from the simulator state.
pub fn update_quadrotor_visual(
    quad: Res<QuadrotorState>,
    mut query: Query<&mut Transform, With<QuadrotorRoot>>,
) {
    let st = quad.simulator.state();
    let [w, x, y, z] = st.quaternion;
    // MuJoCo quaternion is [w,x,y,z]; bevy Quat::from_xyzw takes (x,y,z,w).
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

/// Move the green halo to the current waypoint.
pub fn update_waypoint_visual(
    waypoints: Res<WaypointPath>,
    mut query: Query<&mut Transform, With<ActiveWaypointHighlight>>,
) {
    let wp = waypoints.current();
    for mut t in &mut query {
        t.translation = Vec3::new(wp[0] as f32, wp[1] as f32, wp[2] as f32);
    }
}
