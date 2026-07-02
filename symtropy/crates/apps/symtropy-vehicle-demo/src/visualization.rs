// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: ground, car body + 4 wheels, ice patches, waypoint markers.

use bevy::prelude::*;
use symthaea_vehicle::simulator::VehiclePhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct VehicleRoot;

#[derive(Component)]
pub struct WaypointMarker;

#[derive(Component)]
pub struct ActiveWaypointHighlight;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    waypoints: Res<WaypointPath>,
    ice: Res<Ice>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(80.0, 80.0, 0.05))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.24, 0.26),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -0.025),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(6.0, -6.0, 20.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 260.0,
        affects_lightmapped_meshes: false,
    });

    // Ice patches as blue translucent discs at z=0.02
    let ice_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.35, 0.75, 1.0, 0.45),
        emissive: LinearRgba::new(0.05, 0.15, 0.25, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    for p in &ice.field.patches {
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(p.radius as f32, 0.02))),
            MeshMaterial3d(ice_mat.clone()),
            Transform::from_xyz(p.center[0] as f32, p.center[1] as f32, 0.02)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        ));
    }

    // Waypoints as faint spheres
    let wp_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.7, 0.7, 0.8, 0.55),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    for wp in waypoints.waypoints.iter() {
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.4))),
            MeshMaterial3d(wp_mat.clone()),
            Transform::from_xyz(wp[0] as f32, wp[1] as f32, 0.4),
            WaypointMarker,
        ));
    }

    // Active-waypoint halo
    let halo_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.1, 0.9, 0.3, 0.45),
        emissive: LinearRgba::new(0.1, 0.4, 0.2, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.0))),
        MeshMaterial3d(halo_mat),
        Transform::from_xyz(0.0, 0.0, 0.5),
        ActiveWaypointHighlight,
    ));

    // Car: body + 4 wheels as children of a parent transform
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.15, 0.20),
        metallic: 0.55,
        perceptual_roughness: 0.35,
        ..default()
    });
    let wheel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.08),
        ..default()
    });

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 0.4),
            Visibility::Visible,
            VehicleRoot,
        ))
        .with_children(|parent| {
            // Body (2.8m × 1.4m × 0.6m, approx sedan)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(2.8, 1.4, 0.6))),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(0.0, 0.0, 0.1),
            ));
            // Cabin (shorter + raised)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.4, 1.2, 0.4))),
                MeshMaterial3d(body_mat),
                Transform::from_xyz(0.0, 0.0, 0.55),
            ));
            // Four wheels as flat cylinders laid on their sides
            let wheel_mesh = meshes.add(Cylinder::new(0.35, 0.2));
            for (sx, sy) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
                parent.spawn((
                    Mesh3d(wheel_mesh.clone()),
                    MeshMaterial3d(wheel_mat.clone()),
                    Transform::from_xyz(1.2 * sx, 0.8 * sy, -0.25)
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                ));
            }
        });
}

/// Position + yaw the car from the simulator state.
pub fn update_vehicle_visual(
    vehicle: Res<VehicleResources>,
    mut query: Query<&mut Transform, With<VehicleRoot>>,
) {
    let st = vehicle.simulator.state();
    for mut t in &mut query {
        t.translation = Vec3::new(st.position_x as f32, st.position_y as f32, 0.4);
        t.rotation = Quat::from_rotation_z(st.heading as f32);
    }
}

/// Move the green halo to the current waypoint.
pub fn update_waypoint_visual(
    waypoints: Res<WaypointPath>,
    mut query: Query<&mut Transform, With<ActiveWaypointHighlight>>,
) {
    let wp = waypoints.current();
    for mut t in &mut query {
        t.translation = Vec3::new(wp[0] as f32, wp[1] as f32, 0.5);
    }
}
