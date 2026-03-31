// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Terra Atlas globe view integration.
//!
//! Press M during gameplay to open the planetary map.
//! Press Escape to return to the dungeon.

use bevy::prelude::*;
use terra_atlas_bevy::camera::OrbitalCamera;
use terra_atlas_bevy::globe::{Atmosphere, Globe};
use terra_atlas_bevy::markers::DataMarker;
use terra_atlas_bevy::timeline::{TimelineLayer, TimelineState};
use terra_atlas_core::geo;
use terra_atlas_core::types::Layer;

use crate::resources::GamePhase;

/// Marker for all atlas-spawned entities so we can despawn them on exit.
#[derive(Component)]
pub struct AtlasEntity;

/// Holds loaded data for arc rendering each frame (gizmos are immediate-mode).
#[derive(Resource)]
pub struct AtlasData {
    pub data: terra_atlas_core::types::LoadedData,
}

/// Toggle to globe view when M is pressed during gameplay.
pub fn atlas_toggle_system(
    kb: Res<ButtonInput<KeyCode>>,
    mut next: ResMut<NextState<GamePhase>>,
) {
    if kb.just_pressed(KeyCode::KeyM) {
        next.set(GamePhase::GlobeView);
    }
}

/// Set up the globe view — spawn globe, camera, lights, stars, and data markers.
pub fn setup_globe_view(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    dungeon_cameras: Query<Entity, (With<Camera2d>, Without<OrbitalCamera>)>,
) {
    // Hide the 2D dungeon camera
    for entity in dungeon_cameras.iter() {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    // Pure black space background
    commands.insert_resource(ClearColor(Color::BLACK));

    // Ambient light so the dark side of the globe isn't pitch black
    commands.spawn((
        AmbientLight {
            color: Color::linear_rgb(0.15, 0.18, 0.25),
            brightness: 50.0,
            ..default()
        },
        AtlasEntity,
    ));

    // Earth globe with blue marble texture
    let earth_mesh = meshes.add(Sphere::new(1.0).mesh().uv(128, 128));
    let earth_texture: Handle<Image> = asset_server.load(terra_atlas_bevy::globe::EARTH_TEXTURE_PATH);
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
        AtlasEntity,
    ));

    // Atmosphere shell
    let atmo_mesh = meshes.add(Sphere::new(1.02).mesh().uv(64, 64));
    let atmo_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.0, 0.261, 0.3, 0.15),
        emissive: LinearRgba::new(0.0, 0.25, 0.30, 1.0),
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
        AtlasEntity,
    ));

    // Sun light — moderate to avoid bloom oversaturation
    commands.spawn((
        DirectionalLight {
            illuminance: 5_000.0,
            color: Color::linear_rgb(1.0, 0.98, 0.95),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.4, 0.6, 0.0)),
        AtlasEntity,
    ));

    // 3D orbital camera with Reinhard tonemapping (avoids TonyMcMapFace LUT requirement)
    commands.spawn((
        Camera3d::default(),
        bevy::core_pipeline::tonemapping::Tonemapping::Reinhard,
        Transform::from_xyz(0.0, 0.0, 4.2).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitalCamera,
        AtlasEntity,
    ));

    // ─── Starfield ───────────────────────────────────────────────
    let star_data = terra_atlas_core::geometry::generate_starfield(300, 40.0);
    let star_mesh = meshes.add(Sphere::new(1.0).mesh().uv(4, 4));
    // 7 floats per star: pos.xyz, color.rgb, brightness
    for chunk in star_data.chunks_exact(7) {
        let brightness = chunk[6];
        // Only spawn the brighter stars as entities (top ~30%)
        if brightness < 0.45 {
            continue;
        }
        let size = 0.1 + brightness * 0.3;
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(
                chunk[3] * brightness * 1.5,
                chunk[4] * brightness * 1.5,
                chunk[5] * brightness * 1.5,
            ),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(star_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(chunk[0], chunk[1], chunk[2])
                .with_scale(Vec3::splat(size)),
            AtlasEntity,
        ));
    }

    // ─── Data markers ────────────────────────────────────────────
    let data = terra_atlas_bevy::data::load_all();
    let marker_mesh = meshes.add(Sphere::new(1.0).mesh().uv(6, 6));
    let mut marker_count = 0usize;

    // Energy sites — color-coded by type, unlit for consistent visibility
    for site in &data.sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.008);
        let size = geo::marker_size_from_capacity(site.capacity_mw);
        let c = site.energy_type.rgb();
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Energy, name: site.name.clone() },
            TimelineLayer::Renewable,
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Geothermal nodes — red
    let gc = Layer::Geothermal.rgb();
    for node in &data.geothermal_nodes {
        let pos = geo::lat_lon_to_xyz(node.lat, node.lon, 1.008);
        let size = geo::marker_size_from_capacity(node.capacity_mw);
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(gc[0], gc[1], gc[2]),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Geothermal, name: node.name.clone() },
            TimelineLayer::Renewable,
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Terra Lumina sites — purple, larger (flagship projects)
    let tc = Layer::TerraLumina.rgb();
    for site in &data.terra_lumina_sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.015);
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(tc[0] * 1.3, tc[1] * 1.3, tc[2] * 1.3),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.025)),
            DataMarker { layer: Layer::TerraLumina, name: site.name.clone() },
            TimelineLayer::Renewable,
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Resontia vaults — emerald
    let vc = Layer::ResontiaVaults.rgb();
    for (i, vault) in data.resontia_vaults.iter().enumerate() {
        let pos = geo::lat_lon_to_xyz(vault.lat, vault.lon, 1.01);
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(vc[0], vc[1], vc[2]),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.018)),
            DataMarker { layer: Layer::ResontiaVaults, name: vault.name.clone() },
            TimelineLayer::Vault(i),
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Fossil deposits — EROI-colored (green→amber→red), with carbon emission halos
    let halo_mesh = meshes.add(Sphere::new(1.0).mesh().uv(12, 12));
    for deposit in &data.fossil_deposits {
        let pos = geo::lat_lon_to_xyz(deposit.lat, deposit.lon, 1.006);
        let eroi = terra_atlas_core::economics::compute_eroi(deposit).unwrap_or(5.0);
        let c = terra_atlas_core::economics::eroi_color(eroi);
        let emissive = geo::fossil_emissive_factor(&deposit.status) * 0.5;
        let scale = geo::fossil_scale_factor(&deposit.status);
        let size = geo::marker_size_from_reserves(deposit.proven_reserves_mboe) * scale;
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0] * emissive, c[1] * emissive, c[2] * emissive),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::FossilDeposits, name: deposit.name.clone() },
            TimelineLayer::Fossil,
            AtlasEntity,
        ));

        // Carbon emission halo — translucent red, sized by CO2 output
        if deposit.annual_production_mboe > 0.0 {
            let halo_radius = geo::emission_halo_radius(
                deposit.annual_production_mboe,
                &deposit.fuel_type,
            );
            let halo_mat = materials.add(StandardMaterial {
                base_color: Color::linear_rgba(1.0, 0.15, 0.05, 0.10),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            });
            commands.spawn((
                Mesh3d(halo_mesh.clone()),
                MeshMaterial3d(halo_mat),
                Transform::from_xyz(pos[0], pos[1], pos[2])
                    .with_scale(Vec3::splat(halo_radius)),
                TimelineLayer::Fossil,
                AtlasEntity,
            ));
        }

        marker_count += 1;
    }

    // Nuclear sites — violet, SMR planned sites brighter
    let nc = Layer::Nuclear.rgb();
    for site in &data.nuclear_sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.01);
        let size = geo::marker_size_from_capacity(site.capacity_mw);
        let brightness = if site.reactor_type.is_smr() { 1.4 } else { 1.0 };
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(nc[0] * brightness, nc[1] * brightness, nc[2] * brightness),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Nuclear, name: site.name.clone() },
            TimelineLayer::Nuclear,
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Grid stress markers (FEP allostatic load visualization)
    let stress_mesh = meshes.add(Sphere::new(1.0).mesh().uv(10, 10));
    let stress_data = terra_atlas_core::energy_trading::simulate_grid_stress(0);
    for stress in &stress_data {
        let pos = geo::lat_lon_to_xyz(stress.lat, stress.lon, 1.02);
        let c = terra_atlas_core::energy_trading::stress_color(stress.allostatic_load);
        let size = 0.015 + stress.allostatic_load * 0.025;
        // Solid marker for the stress point
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(stress_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Energy, name: format!("{} (Φ={:.2})", stress.name, stress.phi) },
            AtlasEntity,
        ));
        // Translucent stress halo — larger when under more stress
        if stress.allostatic_load > 0.3 {
            let halo_size = 0.03 + stress.allostatic_load * 0.06;
            let alpha = stress.allostatic_load * 0.15;
            let halo_mat = materials.add(StandardMaterial {
                base_color: Color::linear_rgba(c[0], c[1] * 0.3, c[2] * 0.2, alpha),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            });
            commands.spawn((
                Mesh3d(stress_mesh.clone()),
                MeshMaterial3d(halo_mat),
                Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(halo_size)),
                AtlasEntity,
            ));
        }
    }

    // Store data for arc rendering (gizmos are immediate-mode)
    commands.insert_resource(AtlasData { data });

    info!("[atlas] Globe view: {marker_count} markers + {} grid stress nodes — press Esc to return", stress_data.len());
}

/// Draw maglev corridor arcs using gizmos (immediate-mode, redrawn each frame).
pub fn draw_arcs_system(
    atlas_data: Option<Res<AtlasData>>,
    mut gizmos: Gizmos,
    time: Res<Time>,
) {
    let Some(atlas_data) = atlas_data else { return };
    let t = time.elapsed_secs();

    // P2P energy trades — green animated arcs between renewable sites
    let trade_sites: Vec<(f64, f64, f64)> = atlas_data.data.sites.iter()
        .take(20) // top 20 sites for trade simulation
        .map(|s| (s.lat, s.lon, s.capacity_mw))
        .collect();
    let trades = terra_atlas_core::energy_trading::simulate_trades(&trade_sites, t as f64);
    let trade_color = Color::linear_rgba(0.0, 1.0, 0.5, 0.5); // bright green
    for trade in &trades {
        let from = geo::lat_lon_to_xyz(trade.seller_lat, trade.seller_lon, 1.0);
        let to = geo::lat_lon_to_xyz(trade.buyer_lat, trade.buyer_lon, 1.0);
        let dist = terra_atlas_core::geo::haversine_km(
            trade.seller_lat, trade.seller_lon,
            trade.buyer_lat, trade.buyer_lon,
        );
        let peak = geo::arc_peak_height(dist);
        let segments = 12u32;
        let arc = terra_atlas_core::geometry::generate_arc(from, to, peak, segments);
        for i in 0..segments as usize {
            let a = Vec3::new(arc[i * 3], arc[i * 3 + 1], arc[i * 3 + 2]);
            let b = Vec3::new(arc[(i + 1) * 3], arc[(i + 1) * 3 + 1], arc[(i + 1) * 3 + 2]);
            gizmos.line(a, b, trade_color);
        }
    }

    // Maglev corridors — amber arcs
    let maglev_color = Color::linear_rgba(0.984, 0.749, 0.141, 0.7);
    for corridor in &atlas_data.data.maglev_corridors {
        let from = geo::lat_lon_to_xyz(corridor.from_lat, corridor.from_lon, 1.0);
        let to = geo::lat_lon_to_xyz(corridor.to_lat, corridor.to_lon, 1.0);
        let peak = geo::arc_peak_height(corridor.distance_km);

        // Draw arc as connected line segments
        let segments = 24u32;
        let arc = terra_atlas_core::geometry::generate_arc(from, to, peak, segments);
        // arc is flat [x,y,z] * (segments+1)
        for i in 0..segments as usize {
            let a = Vec3::new(arc[i * 3], arc[i * 3 + 1], arc[i * 3 + 2]);
            let b = Vec3::new(arc[(i + 1) * 3], arc[(i + 1) * 3 + 1], arc[(i + 1) * 3 + 2]);
            gizmos.line(a, b, maglev_color);
        }
    }

    // Supply routes — cyan arcs, dimmer
    let supply_color = Color::linear_rgba(0.0, 0.87, 1.0, 0.3);
    for route in &atlas_data.data.supply_routes {
        let from = geo::lat_lon_to_xyz(route.from_lat, route.from_lon, 1.0);
        let to = geo::lat_lon_to_xyz(route.to_lat, route.to_lon, 1.0);
        let dist = terra_atlas_core::geo::haversine_km(
            route.from_lat, route.from_lon,
            route.to_lat, route.to_lon,
        );
        let peak = geo::arc_peak_height(dist);

        let segments = 16u32;
        let arc = terra_atlas_core::geometry::generate_arc(from, to, peak, segments);
        for i in 0..segments as usize {
            let a = Vec3::new(arc[i * 3], arc[i * 3 + 1], arc[i * 3 + 2]);
            let b = Vec3::new(arc[(i + 1) * 3], arc[(i + 1) * 3 + 1], arc[(i + 1) * 3 + 2]);
            gizmos.line(a, b, supply_color);
        }
    }
}

/// Update marker visibility based on timeline year.
pub fn timeline_visibility_system(
    state: Res<TimelineState>,
    mut markers: Query<(&TimelineLayer, &mut Visibility), With<AtlasEntity>>,
) {
    let year = state.year;
    for (layer, mut vis) in markers.iter_mut() {
        let opacity = match layer {
            TimelineLayer::Fossil => {
                // Use a generic fade for all fossils (per-deposit opacity needs stored data)
                let t = (year as f32 / 300.0).min(1.0);
                1.0 - t
            }
            TimelineLayer::Renewable => terra_atlas_core::timeline::renewable_opacity(year),
            TimelineLayer::Nuclear => terra_atlas_core::timeline::nuclear_opacity(year),
            TimelineLayer::Vault(i) => {
                if terra_atlas_core::timeline::vault_visible(*i, year) { 1.0 } else { 0.0 }
            }
            TimelineLayer::Corridor(i) => {
                if terra_atlas_core::timeline::corridor_visible(*i, year) { 1.0 } else { 0.0 }
            }
            TimelineLayer::Star => 1.0,
        };

        *vis = if opacity < 0.05 {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
}

/// Return to gameplay when Escape is pressed in globe view.
pub fn globe_input_system(
    kb: Res<ButtonInput<KeyCode>>,
    mut next: ResMut<NextState<GamePhase>>,
) {
    if kb.just_pressed(KeyCode::Escape) {
        next.set(GamePhase::Playing);
    }
}

/// Tear down globe view — despawn all atlas entities, restore 2D camera.
pub fn cleanup_globe_view(
    mut commands: Commands,
    atlas_entities: Query<Entity, With<AtlasEntity>>,
    hidden_cameras: Query<Entity, With<Camera2d>>,
) {
    for entity in atlas_entities.iter() {
        commands.entity(entity).despawn();
    }

    for entity in hidden_cameras.iter() {
        commands.entity(entity).insert(Visibility::Visible);
    }

    // Restore dungeon background color and remove atlas data
    commands.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)));
    commands.remove_resource::<AtlasData>();

    info!("[atlas] Returned to dungeon");
}
