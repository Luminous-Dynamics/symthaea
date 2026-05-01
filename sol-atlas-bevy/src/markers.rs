// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Marker spawning — converts terra-atlas-core data into Bevy entities.

use bevy::prelude::*;
use sol_atlas_core::geo;
use sol_atlas_core::types::*;

/// Marker component for data point entities on the globe.
#[derive(Component, Clone, Debug)]
pub struct DataMarker {
    pub layer: Layer,
    pub name: String,
}

/// Spawn energy site markers as small emissive spheres on the globe.
pub fn spawn_site_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    sites: &[Site],
) {
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(8, 8));

    for site in sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.01);
        let size = geo::marker_size_from_capacity(site.capacity_mw);
        let c = site.energy_type.rgb();

        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            emissive: LinearRgba::new(c[0] * 2.0, c[1] * 2.0, c[2] * 2.0, 1.0),
            ..default()
        });

        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker {
                layer: Layer::Energy,
                name: site.name.clone(),
            },
        ));
    }
}

/// Spawn geothermal node markers.
pub fn spawn_geothermal_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    nodes: &[GeothermalNode],
) {
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(8, 8));
    let c = Layer::Geothermal.rgb();

    for node in nodes {
        let pos = geo::lat_lon_to_xyz(node.lat, node.lon, 1.01);
        let size = geo::marker_size_from_capacity(node.capacity_mw);

        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            emissive: LinearRgba::new(c[0] * 2.0, c[1] * 2.0, c[2] * 2.0, 1.0),
            ..default()
        });

        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker {
                layer: Layer::Geothermal,
                name: node.name.clone(),
            },
        ));
    }
}

/// Spawn Terra Lumina site markers (larger, crystal-like).
pub fn spawn_terra_lumina_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    sites: &[TerraLuminaSite],
) {
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(8, 8));
    let c = Layer::TerraLumina.rgb();

    for site in sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.015);

        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            emissive: LinearRgba::new(c[0] * 3.0, c[1] * 3.0, c[2] * 3.0, 1.0),
            ..default()
        });

        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.025)),
            DataMarker {
                layer: Layer::TerraLumina,
                name: site.name.clone(),
            },
        ));
    }
}

/// Spawn Resontia vault markers.
pub fn spawn_vault_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    vaults: &[ResontiaVault],
) {
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(8, 8));
    let c = Layer::ResontiaVaults.rgb();

    for vault in vaults {
        let pos = geo::lat_lon_to_xyz(vault.lat, vault.lon, 1.01);

        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            emissive: LinearRgba::new(c[0] * 2.0, c[1] * 2.0, c[2] * 2.0, 1.0),
            ..default()
        });

        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.018)),
            DataMarker {
                layer: Layer::ResontiaVaults,
                name: vault.name.clone(),
            },
        ));
    }
}
