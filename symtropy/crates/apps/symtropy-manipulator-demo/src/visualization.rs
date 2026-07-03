// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy visualization: arm rendering, workspace envelope, human avatar, HUD.

use bevy::prelude::*;
use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::simulator::ManipulatorPhysicsSimulator;

/// Marker for arm link meshes (ISO arm).
#[derive(Component)]
pub struct IsoArmLink {
    pub joint_index: usize,
}

/// Marker for the ISO arm end-effector.
#[derive(Component)]
pub struct IsoEndEffector;

/// Marker for the human avatar capsule.
#[derive(Component)]
pub struct HumanAvatar;

/// Marker for the human reach sphere.
#[derive(Component)]
pub struct HumanReachSphere;

/// Marker for pick/place position indicators.
#[derive(Component)]
pub struct TaskMarker;

/// Marker for the Phi arm workspace envelope (semi-transparent sphere).
#[derive(Component)]
pub struct WorkspaceEnvelope;

/// Marker for the table surface.
#[derive(Component)]
pub struct TableSurface;

/// Spawn the scene: table, task markers, human avatar, and ISO arm.
pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane / table
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 0.02))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        TableSurface,
    ));

    // Pick position marker (green dot)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.02))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.8, 0.0),
            emissive: LinearRgba::new(0.0, 0.3, 0.0, 1.0),
            ..default()
        })),
        Transform::from_xyz(0.4, -0.3, 0.02),
        TaskMarker,
    ));

    // Place position marker (blue dot)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.02))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.3, 0.8),
            emissive: LinearRgba::new(0.0, 0.0, 0.3, 1.0),
            ..default()
        })),
        Transform::from_xyz(0.4, 0.3, 0.02),
        TaskMarker,
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.0, -2.0, 4.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));

    // Global ambient light (Resource in Bevy 0.18 via bevy_light)
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        affects_lightmapped_meshes: false,
    });

    // Human avatar (capsule body)
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.15, 0.8))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.4, 0.8, 0.7),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, -1.2, 0.5),
        HumanAvatar,
    ));

    // Human reach sphere (transparent)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.7))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.5, 0.9, 0.08),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, -1.2, 0.5),
        HumanReachSphere,
    ));

    // Workspace envelope (Phi arm) — semi-transparent sphere
    // Color and radius update dynamically based on safety tier
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.855))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.0, 0.8, 0.0, 0.12),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.333), // Centered at arm base
        WorkspaceEnvelope,
    ));

    // ISO arm: 7 joint spheres + links (cylinders) + end-effector
    let joint_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 0.25),
        metallic: 0.8,
        ..default()
    });
    // Link mesh and material reserved for future inter-joint cylinder rendering
    let _link_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.65),
        metallic: 0.9,
        ..default()
    });

    let joint_mesh = meshes.add(Sphere::new(0.025));
    let _link_mesh = meshes.add(Cylinder::new(0.015, 0.1));

    // Spawn 7 joint markers
    for i in 0..7 {
        commands.spawn((
            Mesh3d(joint_mesh.clone()),
            MeshMaterial3d(joint_material.clone()),
            Transform::from_xyz(0.0, 0.0, 0.333 * (i as f32 / 6.0)), // Initial spread
            IsoArmLink { joint_index: i },
        ));
    }

    // End-effector (small box for gripper)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.04, 0.06, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.4, 0.1),
            metallic: 0.7,
            ..default()
        })),
        Transform::from_xyz(0.3, 0.0, 0.5),
        IsoEndEffector,
    ));
}

/// Update human avatar position from shared obstacle state.
pub fn update_human_visual(
    human: Res<crate::resources::SharedHuman>,
    mut query: Query<&mut Transform, With<HumanAvatar>>,
    mut reach_query: Query<&mut Transform, (With<HumanReachSphere>, Without<HumanAvatar>)>,
) {
    let pos = human.obstacle.position;
    for mut transform in &mut query {
        transform.translation = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
    }
    for mut transform in &mut reach_query {
        transform.translation = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
    }
}

/// Update ISO arm joint positions from FK.
pub fn update_iso_arm_visual(
    iso_arm: Res<crate::resources::IsoArmState>,
    mut joint_query: Query<(&IsoArmLink, &mut Transform)>,
    mut ee_query: Query<&mut Transform, (With<IsoEndEffector>, Without<IsoArmLink>)>,
) {
    let state = iso_arm.simulator.state();
    let kin = &iso_arm.kinematics;

    // Compute intermediate joint positions via partial FK chains
    for (link, mut transform) in &mut joint_query {
        let pos = compute_joint_position(kin, &state.joint_angles, link.joint_index);
        transform.translation = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
    }

    // End-effector position
    let ee_pos = state.end_effector_position;
    for mut transform in &mut ee_query {
        transform.translation = Vec3::new(ee_pos[0] as f32, ee_pos[1] as f32, ee_pos[2] as f32);
    }
}

/// Update workspace envelope: radius shrinks with safety level, color shifts Green→Red.
pub fn update_workspace_envelope(
    phi_arm: Res<crate::resources::PhiArmState>,
    mut query: Query<(&mut Transform, &MeshMaterial3d<StandardMaterial>), With<WorkspaceEnvelope>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use symtropy_consciousness_physics::safety::SafetyTier;

    let (radius_fraction, color) = match phi_arm.current_safety {
        SafetyTier::Green => (1.0, Color::srgba(0.0, 0.8, 0.0, 0.12)),
        SafetyTier::Yellow => (0.8, Color::srgba(0.8, 0.8, 0.0, 0.18)),
        SafetyTier::Orange => (0.5, Color::srgba(0.9, 0.4, 0.0, 0.22)),
        SafetyTier::Red => (0.3, Color::srgba(0.9, 0.0, 0.0, 0.28)),
    };

    for (mut transform, mat_handle) in &mut query {
        // Scale sphere to match workspace fraction
        let scale = 0.855 * radius_fraction as f32;
        transform.scale = Vec3::splat(scale / 0.855); // Normalize since mesh radius is 0.855

        // Update material color
        if let Some(mut mat) = materials.get_mut(&mat_handle.0) {
            mat.base_color = color;
        }
    }
}

/// Compute position of joint i by running FK up to joint i.
fn compute_joint_position(
    kin: &ManipulatorKinematics,
    angles: &[f64],
    joint_index: usize,
) -> [f64; 3] {
    // For joint 0, use base position
    if joint_index == 0 {
        return [0.0, 0.0, 0.333]; // Base height (DH d1)
    }

    // Compute partial FK: chain up to joint_index
    // This is an approximation using interpolated DH parameters
    // For a proper implementation, we'd compute partial transforms
    // For now, linearly interpolate between base and full EE
    let ee = kin.end_effector_position(angles);
    let base = [0.0, 0.0, 0.333];
    let t = joint_index as f64 / 7.0;
    [
        base[0] + (ee[0] - base[0]) * t,
        base[1] + (ee[1] - base[1]) * t,
        base[2] + (ee[2] - base[2]) * t,
    ]
}
