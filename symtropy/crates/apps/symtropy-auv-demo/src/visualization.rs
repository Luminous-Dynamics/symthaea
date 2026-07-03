// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: water volume, seafloor, AUV body, waypoint beacons.

use bevy::prelude::*;
use symthaea_auv::simulator::AuvPhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct AuvRoot;

#[derive(Component)]
pub struct WaypointBeacon;

#[derive(Component)]
pub struct ActiveWaypointHighlight;

/// Visual world convention: z=0 is the surface, z decreases as the AUV
/// descends (simulator depth is positive-down, so visual z = -depth).
pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    waypoints: Res<WaypointPath>,
) {
    // Seafloor at 25 m (visual z = -25)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(50.0, 50.0, 0.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.30, 0.35),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -25.25),
    ));

    // Surface plane (translucent blue)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(50.0, 50.0, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.15, 0.45, 0.70, 0.35),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Thermocline disc at ~12 m (cue for the current's peak region)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(50.0, 50.0, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.10, 0.70, 0.85, 0.20),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -12.0),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 7500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(6.0, -6.0, 20.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::srgb(0.55, 0.75, 0.90),
        brightness: 320.0,
        affects_lightmapped_meshes: false,
    });

    // Waypoint beacons — vertical cylinders spanning a small range
    let beacon_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.90, 0.85, 0.40, 0.75),
        emissive: LinearRgba::new(0.25, 0.20, 0.05, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    for wp in waypoints.waypoints.iter() {
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.35, 1.5))),
            MeshMaterial3d(beacon_mat.clone()),
            Transform::from_xyz(wp[0] as f32, wp[1] as f32, -(wp[2] as f32)),
            WaypointBeacon,
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
        Transform::from_xyz(0.0, 0.0, -10.0),
        ActiveWaypointHighlight,
    ));

    // AUV body: torpedo = capsule rotated so long axis is body-forward (x)
    let hull_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.80, 0.20),
        metallic: 0.55,
        perceptual_roughness: 0.35,
        ..default()
    });
    let fin_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.20, 0.22),
        ..default()
    });

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, -10.0),
            Visibility::Visible,
            AuvRoot,
        ))
        .with_children(|parent| {
            // Torpedo hull (Capsule3d is long-axis = Y; rotate so long axis = X)
            parent.spawn((
                Mesh3d(meshes.add(Capsule3d::new(0.25, 1.6))),
                MeshMaterial3d(hull_mat),
                Transform::from_xyz(0.0, 0.0, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            ));
            // Four stabilizer fins at the tail
            let fin_mesh = meshes.add(Cuboid::new(0.05, 0.35, 0.01));
            for angle in [0.0, 90.0, 180.0, 270.0] {
                let rad = (angle as f32).to_radians();
                parent.spawn((
                    Mesh3d(fin_mesh.clone()),
                    MeshMaterial3d(fin_mat.clone()),
                    Transform::from_xyz(-0.95, 0.0, 0.0).with_rotation(Quat::from_rotation_x(rad)),
                ));
            }
        });
}

/// Position + yaw the AUV from the simulator state (depth → visual z = -depth).
pub fn update_auv_visual(auv: Res<AuvResources>, mut query: Query<&mut Transform, With<AuvRoot>>) {
    let st = auv.simulator.state();
    let [w, x, y, z] = st.quaternion;
    // MuJoCo quaternion [w,x,y,z] → bevy Quat::from_xyzw(x,y,z,w)
    let rot = Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32);
    for mut t in &mut query {
        t.translation = Vec3::new(
            st.position[0] as f32,
            st.position[1] as f32,
            -(st.depth as f32),
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
        t.translation = Vec3::new(wp[0] as f32, wp[1] as f32, -(wp[2] as f32));
    }
}
