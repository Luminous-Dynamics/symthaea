// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Globe mesh spawning — textured sphere with atmosphere glow and directional light.

use bevy::prelude::*;
use sol_atlas_core::constants::*;

/// Marker component for the globe entity.
#[derive(Component)]
pub struct Globe;

/// Marker component for atmosphere shell.
#[derive(Component)]
pub struct Atmosphere;

/// Path to the earth texture (relative to the binary's `assets/` directory).
pub const EARTH_TEXTURE_PATH: &str = "textures/earth-8k.jpg";

/// Spawn the globe sphere, atmosphere shell, and sun light.
pub fn spawn_globe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Earth globe with blue marble texture
    let earth_mesh = meshes.add(Sphere::new(GLOBE_RADIUS).mesh().uv(128, 128));
    let earth_texture: Handle<Image> = asset_server.load(EARTH_TEXTURE_PATH);

    let earth_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(earth_texture),
        perceptual_roughness: 0.85,
        metallic: 0.0,
        ..default()
    });

    commands.spawn((
        Mesh3d(earth_mesh),
        MeshMaterial3d(earth_material),
        Transform::IDENTITY,
        Globe,
    ));

    // Atmosphere shell — slightly larger, emissive, transparent
    let atmo_mesh = meshes.add(Sphere::new(GLOBE_RADIUS * 1.02).mesh().uv(64, 64));

    let atmo_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(
            MYCELIX_CYAN[0] * 0.3,
            MYCELIX_CYAN[1] * 0.3,
            MYCELIX_CYAN[2] * 0.3,
            0.08,
        ),
        emissive: LinearRgba::new(
            MYCELIX_CYAN[0] * 0.15,
            MYCELIX_CYAN[1] * 0.15,
            MYCELIX_CYAN[2] * 0.15,
            1.0,
        ),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Mesh3d(atmo_mesh),
        MeshMaterial3d(atmo_material),
        Transform::IDENTITY,
        Atmosphere,
    ));

    // Sun — directional light from upper-right
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            color: Color::linear_rgb(1.0, 0.98, 0.95),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.4, 0.6, 0.0)),
    ));
}
