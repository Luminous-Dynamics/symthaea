// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D scene: surgical table, tissue target, critical structure, trocar port,
//! 3-link instrument reaching down to the tip. All distances in millimeters
//! (to match the simulator's FK output).

use bevy::prelude::*;
use symthaea_surgical::simulator::SurgicalPhysicsSimulator;

use crate::resources::*;

#[derive(Component)]
pub struct ArmSegment {
    pub index: usize,
}

#[derive(Component)]
pub struct ToolTip;

#[derive(Component)]
pub struct CriticalStructure;

#[derive(Component)]
pub struct CauteryGlow;

/// Segment lengths in millimeters (match simulator's FK: 150 / 100 / 30).
const L0: f32 = 150.0;
const L1: f32 = 100.0;
const L2: f32 = 30.0;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Surgical "table" / skin backdrop at z = -60 mm (a bit above the tissue layer).
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(400.0, 300.0, 4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.70, 0.55),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -60.0),
    ));

    // Tissue blob (translucent pink sphere) at the target area (z = -40)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(25.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.95, 0.40, 0.45, 0.75),
            emissive: LinearRgba::new(0.15, 0.05, 0.08, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -40.0),
    ));

    // Critical structure (red translucent cylinder, pulsing) about 15 mm lateral.
    let crit_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.1, 0.15, 0.65),
        emissive: LinearRgba::new(0.60, 0.05, 0.10, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(3.0, 80.0))),
        MeshMaterial3d(crit_mat),
        Transform::from_xyz(16.0, 0.0, -40.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        CriticalStructure,
    ));

    // Trocar port (dark collar at the skin surface, z = -58)
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(14.0, 8.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.15, 0.18),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -55.0),
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadow_maps_enabled: true,
            color: Color::srgb(1.0, 0.98, 0.95),
            ..default()
        },
        Transform::from_xyz(60.0, -60.0, 150.0).looking_at(Vec3::new(0.0, 0.0, -40.0), Vec3::Z),
    ));
    commands.insert_resource(bevy::prelude::GlobalAmbientLight {
        color: Color::srgb(1.0, 0.95, 0.90),
        brightness: 320.0,
        affects_lightmapped_meshes: false,
    });

    // Instrument arm: three serial link cuboids + tip. Orientation updated
    // each frame; the simulator's FK uses just 3 sagittal joints (0,1,2).
    let arm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.78, 0.85),
        metallic: 0.75,
        perceptual_roughness: 0.25,
        ..default()
    });
    let tip_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.65, 0.7),
        metallic: 0.8,
        perceptual_roughness: 0.20,
        ..default()
    });

    let link0 = meshes.add(Cuboid::new(L0, 8.0, 8.0));
    let link1 = meshes.add(Cuboid::new(L1, 6.0, 6.0));
    let link2 = meshes.add(Cuboid::new(L2, 5.0, 5.0));

    commands.spawn((
        Mesh3d(link0),
        MeshMaterial3d(arm_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 60.0),
        ArmSegment { index: 0 },
    ));
    commands.spawn((
        Mesh3d(link1),
        MeshMaterial3d(arm_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 30.0),
        ArmSegment { index: 1 },
    ));
    commands.spawn((
        Mesh3d(link2),
        MeshMaterial3d(arm_mat),
        Transform::from_xyz(0.0, 0.0, 0.0),
        ArmSegment { index: 2 },
    ));

    // Tool tip (cone-ish cuboid) — emissive glow toggled by cautery.
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(3.0))),
        MeshMaterial3d(tip_mat),
        Transform::from_xyz(0.0, 0.0, -40.0),
        ToolTip,
    ));

    // Cautery glow (larger translucent sphere around the tip, alpha driven by cautery power)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(6.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.5, 0.05, 0.0),
            emissive: LinearRgba::new(0.8, 0.3, 0.02, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, -40.0),
        CauteryGlow,
    ));
}

/// Sagittal-plane forward-kinematics visualization: 3 serial links hinged at
/// the trocar port, plus a tip marker placed at the simulator's `tip_position`.
pub fn update_arm_visual(
    surg: Res<SurgicalResources>,
    mut query: Query<(
        &mut Transform,
        Option<&ArmSegment>,
        Option<&ToolTip>,
        Option<&CauteryGlow>,
    )>,
) {
    let st = surg.simulator.state();
    // Trocar hinge location — sits at the port on the skin.
    let base = Vec3::new(0.0, 0.0, -55.0);

    let q0 = st.joint_angles[0] as f32;
    let q1 = (st.joint_angles[0] + st.joint_angles[1]) as f32;
    let q2 = (st.joint_angles[0] + st.joint_angles[1] + st.joint_angles[2]) as f32;

    // Each link's "down" vector (pointing toward the tissue) in world frame.
    // Sign convention here matches the simulator's FK: x = Σ L_i * sin(cum),
    // z descends = -Σ L_i * cos(cum).
    let dir0 = Vec3::new(q0.sin(), 0.0, -q0.cos());
    let dir1 = Vec3::new(q1.sin(), 0.0, -q1.cos());
    let dir2 = Vec3::new(q2.sin(), 0.0, -q2.cos());

    let mid0 = base + dir0 * (L0 * 0.5);
    let end0 = base + dir0 * L0;
    let mid1 = end0 + dir1 * (L1 * 0.5);
    let end1 = end0 + dir1 * L1;
    let mid2 = end1 + dir2 * (L2 * 0.5);

    // The tip is what the simulator's FK reports — authoritative.
    let tip = Vec3::new(
        st.tip_position[0] as f32,
        st.tip_position[1] as f32,
        st.tip_position[2] as f32,
    );

    for (mut t, arm, tip_tag, glow) in &mut query {
        if let Some(seg) = arm {
            let (mid, angle) = match seg.index {
                0 => (mid0, q0),
                1 => (mid1, q1),
                2 => (mid2, q2),
                _ => continue,
            };
            t.translation = mid;
            // Rotate link so its long axis aligns with the link direction.
            // Cuboids are long along X; a rotation of (π/2 - angle) about Y
            // gets X aligned with (sin θ, 0, -cos θ).
            t.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2 - angle);
        } else if tip_tag.is_some() {
            t.translation = tip;
        } else if glow.is_some() {
            t.translation = tip;
        }
    }
}

/// Make the cautery glow alpha follow the cautery power channel.
pub fn update_cautery_glow(
    surg: Res<SurgicalResources>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<&MeshMaterial3d<StandardMaterial>, With<CauteryGlow>>,
) {
    let cautery = surg.last_cautery.clamp(0.0, 1.0);
    for handle in &query {
        if let Some(mut mat) = materials.get_mut(&handle.0) {
            let alpha = 0.10 + 0.60 * cautery;
            mat.base_color = Color::srgba(1.0, 0.5, 0.05, alpha);
        }
    }
}

/// Pulse the critical structure's emissive brightness inversely with distance.
pub fn update_critical_structure_pulse(
    surg: Res<SurgicalResources>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<&MeshMaterial3d<StandardMaterial>, With<CriticalStructure>>,
) {
    let dist = surg.simulator.state().critical_structure_distance as f32;
    // Map distance 15 mm (safe) → dim, 0 mm → bright.
    let danger = (1.0 - (dist / 15.0)).clamp(0.0, 1.0);
    let glow = 0.30 + 1.20 * danger;
    for handle in &query {
        if let Some(mut mat) = materials.get_mut(&handle.0) {
            mat.emissive = LinearRgba::new(glow * 0.9, 0.05, 0.08 + glow * 0.15, 1.0);
        }
    }
}
