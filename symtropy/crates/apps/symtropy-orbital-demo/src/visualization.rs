// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: starfield backdrop, spacecraft bus + solar panels + 7-link arm,
//! payload target, Sun light whose intensity tracks `solar_exposure`, and a
//! ground-station comm beam that fades with `comm_window`.

use bevy::prelude::*;
use symthaea_orbital::simulator::OrbitalPhysicsSimulator;
use symthaea_orbital::types::NUM_JOINTS;

use crate::resources::*;

/// Root of the spacecraft (body + panels + arm base). Positioned/rotated by
/// `OrbitalResources::spacecraft_attitude` each frame.
#[derive(Component)]
pub struct SpacecraftRoot;

/// Each arm link, tagged by joint index (0..NUM_JOINTS).
#[derive(Component)]
pub struct ArmLink {
    pub index: usize,
}

/// Payload target "box" the arm is reaching for.
#[derive(Component)]
pub struct PayloadTarget;

/// Directional "sun" light whose illuminance tracks `solar_exposure`.
#[derive(Component)]
pub struct SunLight;

/// Comm-beam mesh (translucent cyan cylinder), alpha tracks `comm_window`.
#[derive(Component)]
pub struct CommBeam;

// Link lengths matching SimpleOrbitalSimulator's `links` field.
const LINK_LENGTHS: [f32; NUM_JOINTS] = [1.5, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2];

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Starfield backdrop: a large dark cuboid "below" (along -z, distant).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(100.0, 100.0, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.02, 0.02, 0.06),
            emissive: LinearRgba::new(0.01, 0.01, 0.04, 1.0),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -12.0),
    ));

    // Sun light (directional, starts bright)
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadow_maps_enabled: false,
            color: Color::srgb(1.0, 0.95, 0.85),
            ..default()
        },
        Transform::from_xyz(10.0, -10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Z),
        SunLight,
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::srgb(0.15, 0.18, 0.25),
        brightness: 110.0,
        affects_lightmapped_meshes: false,
    });

    // Comm-beam (cyan cylinder running from spacecraft origin toward a ground
    // station nominal direction); alpha updated per-frame.
    let beam_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.85, 1.0, 0.25),
        emissive: LinearRgba::new(0.05, 0.25, 0.4, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.08, 12.0))),
        MeshMaterial3d(beam_mat),
        Transform::from_xyz(0.0, 3.0, -5.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 + 0.3)),
        CommBeam,
    ));

    // Payload target — the arm's goal
    let target_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.5, 0.1, 0.75),
        emissive: LinearRgba::new(0.5, 0.2, 0.05, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.35, 0.35, 0.35))),
        MeshMaterial3d(target_mat),
        Transform::from_xyz(2.8, 0.0, 2.5),
        PayloadTarget,
    ));

    // Spacecraft root (bus + panels + arm links as children)
    let bus_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.88),
        metallic: 0.65,
        perceptual_roughness: 0.35,
        ..default()
    });
    let panel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.2, 0.55),
        emissive: LinearRgba::new(0.02, 0.04, 0.15, 1.0),
        metallic: 0.15,
        perceptual_roughness: 0.55,
        ..default()
    });
    let arm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.90, 0.35),
        metallic: 0.3,
        perceptual_roughness: 0.45,
        ..default()
    });

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 1.5),
            Visibility::Visible,
            SpacecraftRoot,
        ))
        .with_children(|parent| {
            // Bus (central cuboid)
            parent.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.2, 1.2, 1.0))),
                MeshMaterial3d(bus_mat),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
            // Solar panels (two thin panels extending ±Y)
            let panel_mesh = meshes.add(Cuboid::new(0.05, 2.8, 1.1));
            parent.spawn((
                Mesh3d(panel_mesh.clone()),
                MeshMaterial3d(panel_mat.clone()),
                Transform::from_xyz(0.0, 2.1, 0.0),
            ));
            parent.spawn((
                Mesh3d(panel_mesh),
                MeshMaterial3d(panel_mat),
                Transform::from_xyz(0.0, -2.1, 0.0),
            ));

            // Seven arm links, positioned/rotated each frame by update_arm_visual.
            for i in 0..NUM_JOINTS {
                let len = LINK_LENGTHS[i];
                let width = 0.15 - (i as f32) * 0.01;
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(len, width, width))),
                    MeshMaterial3d(arm_mat.clone()),
                    Transform::from_xyz(0.0, 0.0, 0.6 + i as f32 * 0.1),
                    ArmLink { index: i },
                ));
            }
        });
}

/// Rotate the spacecraft root by the integrated attitude from the simulator's
/// angular velocity (the crate doesn't keep attitude in state — we do).
pub fn update_spacecraft_visual(
    orbital: Res<OrbitalResources>,
    mut query: Query<&mut Transform, With<SpacecraftRoot>>,
) {
    for mut t in &mut query {
        t.rotation = orbital.spacecraft_attitude;
    }
}

/// Place each arm link in the spacecraft-local frame using a sagittal-plane
/// forward-kinematics chain along the first 4 joints (per
/// SimpleOrbitalSimulator's own FK), then append joints 4–6 as local twists.
pub fn update_arm_visual(
    orbital: Res<OrbitalResources>,
    mut query: Query<(&mut Transform, &ArmLink)>,
) {
    let st = orbital.simulator.state();

    // Base of the arm sits on top of the bus
    let mut pos = Vec3::new(0.0, 0.0, 0.6);
    let mut cumulative_angle = 0.0f32;

    let mut transforms: [(Vec3, f32); NUM_JOINTS] = [(Vec3::ZERO, 0.0); NUM_JOINTS];

    for i in 0..NUM_JOINTS {
        cumulative_angle += st.joint_angles[i] as f32;
        // Link direction in local (spacecraft) frame: tilted about Y axis by
        // cumulative angle; points "forward" (+X) when all angles zero.
        let dir = Vec3::new(cumulative_angle.cos(), 0.0, cumulative_angle.sin());
        let len = LINK_LENGTHS[i];
        let midpoint = pos + dir * (len * 0.5);
        transforms[i] = (midpoint, cumulative_angle);
        pos += dir * len;
    }

    for (mut t, link) in &mut query {
        let (midpoint, angle) = transforms[link.index];
        t.translation = midpoint;
        t.rotation = Quat::from_rotation_y(angle);
    }
}

/// Dim the sun light and fade the comm beam according to the orbital state.
pub fn update_lighting_cue(
    orbital: Res<OrbitalResources>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut sun_q: Query<&mut DirectionalLight, With<SunLight>>,
    beam_q: Query<&MeshMaterial3d<StandardMaterial>, With<CommBeam>>,
) {
    let st = orbital.simulator.state();
    for mut light in &mut sun_q {
        light.illuminance = (2500.0 + 12500.0 * st.solar_exposure as f32).max(500.0);
    }
    // Fade the comm beam alpha with comm_window
    for handle in &beam_q {
        if let Some(mut mat) = materials.get_mut(&handle.0) {
            let alpha = 0.10 + 0.55 * st.comm_window as f32;
            mat.base_color = Color::srgba(0.2, 0.85, 1.0, alpha);
        }
    }
}
