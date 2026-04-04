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
use terra_atlas_core::lod::LodLevel;
use terra_atlas_core::types::Layer;

/// Tag for markers that are only visible at Surface LOD (close zoom).
#[derive(Component)]
pub struct SurfaceLod;

/// Tag for heat blob markers visible at Orbit LOD (far zoom).
#[derive(Component)]
pub struct OrbitLod;

/// 4D temporal position for a marker — W coordinate = year.
/// When 4D mode is active, markers fade based on distance from the timeline slice.
#[derive(Component)]
pub struct TemporalW {
    pub year: f64,
}

use crate::resources::GamePhase;

/// Marker for all atlas-spawned entities so we can despawn them on exit.
#[derive(Component)]
pub struct AtlasEntity;

/// Current aesthetic preset — cycle with number keys 1-5.
#[derive(Resource)]
pub struct CurrentAesthetic {
    pub aesthetic: terra_atlas_core::aesthetics::Aesthetic,
    pub changed: bool,
}

impl Default for CurrentAesthetic {
    fn default() -> Self {
        Self {
            aesthetic: terra_atlas_core::aesthetics::Aesthetic::Holographic,
            changed: false,
        }
    }
}

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
    mut holo_materials: ResMut<Assets<terra_atlas_bevy::holographic_material::HolographicMaterial>>,
    asset_server: Res<AssetServer>,
    dungeon_cameras: Query<Entity, (With<Camera2d>, Without<OrbitalCamera>)>,
    hud_texts: Query<Entity, With<crate::systems::rendering::HudText>>,
) {
    // Hide the 2D dungeon camera
    for entity in dungeon_cameras.iter() {
        commands.entity(entity).insert(Visibility::Hidden);
    }
    // Hide the dungeon HUD
    for entity in hud_texts.iter() {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    // Pure black space background
    commands.insert_resource(ClearColor(Color::BLACK));

    // Ambient light so the dark side of the globe isn't pitch black
    commands.spawn((
        AmbientLight {
            color: Color::linear_rgb(0.15, 0.18, 0.25),
            brightness: 300.0,
            ..default()
        },
        AtlasEntity,
    ));

    // ═══ HOLOGRAPHIC GLOBE ═══════════════════════════════════════

    // [1] Globe with custom holographic shader — Fresnel + scanlines
    let earth_mesh = meshes.add(Sphere::new(1.0).mesh().uv(64, 64)); // 64 = smooth at distance, 4x fewer verts
    let earth_texture: Handle<Image> = asset_server.load(terra_atlas_bevy::globe::EARTH_TEXTURE_PATH);
    let holo_globe = holo_materials.add(terra_atlas_bevy::holographic_material::HolographicMaterial {
        base: StandardMaterial {
            base_color: Color::linear_rgba(0.20, 0.28, 0.35, 0.7), // slightly brighter — coastlines visible
            base_color_texture: Some(earth_texture),
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            ..default()
        },
        extension: terra_atlas_bevy::holographic_material::HolographicExtension {
            fresnel_color: LinearRgba::new(0.0, 0.87, 1.0, 1.0),
            fresnel_power: 3.0,
            scanline_speed: 0.5,
            scanline_density: 20.0,
            hologram_alpha: 0.55,
            ..default()
        },
    });
    commands.spawn((
        Mesh3d(earth_mesh),
        MeshMaterial3d(holo_globe),
        Transform::IDENTITY,
        Globe,
        AtlasEntity,
    ));

    // [2] Sacred geometry wireframe grid — the hologram's skeleton
    // Inner sphere at 0.97 radius — far enough inside to avoid z-fighting
    let grid_mesh = meshes.add(Sphere::new(0.97).mesh().uv(24, 24)); // low-poly = visible edges
    let grid_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.0, 0.87, 1.0, 0.06), // Mycelix cyan, subtle
        emissive: LinearRgba::new(0.0, 0.2, 0.25, 1.0), // reduced — don't wash out texture
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(grid_mesh),
        MeshMaterial3d(grid_material),
        Transform::IDENTITY,
        AtlasEntity,
    ));

    // [5] Fresnel edge glow — outer atmosphere, brighter at grazing angles
    let fresnel_mesh = meshes.add(Sphere::new(1.03).mesh().uv(48, 48));
    let fresnel_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.0, 0.6, 0.8, 0.04),
        emissive: LinearRgba::new(0.0, 0.25, 0.35, 1.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(fresnel_mesh.clone()),
        MeshMaterial3d(fresnel_material),
        Transform::IDENTITY,
        Atmosphere,
        AtlasEntity,
    ));

    // Second Fresnel layer — wider, softer
    let fresnel2_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.0, 0.87, 1.0, 0.03),
        emissive: LinearRgba::new(0.0, 0.4, 0.55, 1.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.05).mesh().uv(32, 32))),
        MeshMaterial3d(fresnel2_material),
        Transform::IDENTITY,
        Atmosphere,
        AtlasEntity,
    ));

    // [7] Holographic projection base — subtle dark glass disc
    let base_mesh = meshes.add(Sphere::new(1.2).mesh().uv(32, 4));
    let base_material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.01, 0.03, 0.04, 0.4), // darker obsidian
        emissive: LinearRgba::new(0.0, 0.015, 0.02, 1.0), // barely visible glow
        perceptual_roughness: 0.1, // polished — catches reflections
        metallic: 0.8,
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(base_mesh),
        MeshMaterial3d(base_material),
        Transform::from_scale(Vec3::new(1.0, 0.005, 1.0)) // thinner disc
            .with_translation(Vec3::new(0.0, -1.15, 0.0)),
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

    // 3D orbital camera — holographic post-processing
    commands.spawn((
        Camera3d::default(),
        bevy::core_pipeline::tonemapping::Tonemapping::AcesFitted,
        // [6] Bloom — makes emissive markers glow through the hologram
        bevy::post_process::bloom::Bloom {
            intensity: 0.04, // low — crisp text, subtle marker glow
            ..default()
        },
        // [3] Chromatic aberration — holographic projection artifact
        bevy::post_process::effect_stack::ChromaticAberration {
            intensity: 0.002,  // minimal — preserves text clarity
            max_samples: 8,
            ..default()
        },
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
        if brightness < 0.55 { // only brightest stars — prevents edge strays
            continue;
        }
        let size = 0.06 + brightness * 0.12; // smaller — stars shouldn't be diamonds
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

    // [6] Energy sites — deep cybernetic palette, emissive glow
    for site in &data.sites {
        let pos = geo::lat_lon_to_xyz(site.lat, site.lon, 1.008);
        let size = geo::marker_size_from_capacity(site.capacity_mw);
        let c = site.energy_type.rgb();
        // Desaturate and deepen colors to match holographic aesthetic
        let depth = 0.6; // pull colors toward deeper tones
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0] * depth, c[1] * depth, c[2] * depth),
            emissive: LinearRgba::new(c[0] * 0.25, c[1] * 0.25, c[2] * 0.25, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Energy, name: site.name.clone() },
            TemporalW { year: 2010.0 }, // renewables: modern era
            SurfaceLod,
            TimelineLayer::Renewable,
            AtlasEntity,
        ));
        marker_count += 1;
    }

    // Heat blob clustering for ALL marker types (visible when zoomed out)
    {
        let mut all_markers: Vec<(f64, f64, f64, [f32; 3])> = Vec::new();
        // Energy
        for s in &data.sites {
            let c = s.energy_type.rgb();
            all_markers.push((s.lat, s.lon, s.capacity_mw, [c[0] * 0.5, c[1] * 0.5, c[2] * 0.5]));
        }
        // Geothermal
        let gc = Layer::Geothermal.rgb();
        for n in &data.geothermal_nodes {
            all_markers.push((n.lat, n.lon, n.capacity_mw, [gc[0] * 0.5, gc[1] * 0.5, gc[2] * 0.5]));
        }
        // Fossil
        for d in &data.fossil_deposits {
            let eroi = terra_atlas_core::economics::compute_eroi(d).unwrap_or(5.0);
            let c = terra_atlas_core::economics::eroi_color(eroi);
            all_markers.push((d.lat, d.lon, d.proven_reserves_mboe * 0.01, [c[0] * 0.5, c[1] * 0.5, c[2] * 0.5]));
        }
        // Nuclear
        let nc = Layer::Nuclear.rgb();
        for s in &data.nuclear_sites {
            all_markers.push((s.lat, s.lon, s.capacity_mw, [nc[0] * 0.5, nc[1] * 0.5, nc[2] * 0.5]));
        }

        let clusters = terra_atlas_core::lod::cluster_markers(&all_markers, 4, 8); // coarser = fewer blobs
        let blob_mesh = meshes.add(Sphere::new(1.0).mesh().uv(10, 10));
        for cell in &clusters {
            let pos = geo::lat_lon_to_xyz(cell.center_lat, cell.center_lon, 1.01);
            let size = terra_atlas_core::lod::heat_blob_size(cell.count);
            let c = cell.avg_color;
            let mat = materials.add(StandardMaterial {
                base_color: Color::linear_rgba(c[0] * 0.3, c[1] * 0.3, c[2] * 0.3, 0.08),
                emissive: LinearRgba::new(c[0] * 0.2, c[1] * 0.2, c[2] * 0.2, 1.0),
                // Additive blend — glows through instead of opaque disc
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            });
            commands.spawn((
                Mesh3d(blob_mesh.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
                OrbitLod,
                AtlasEntity,
            ));
        }
    }

    // Geothermal nodes — red
    let gc = Layer::Geothermal.rgb();
    for node in &data.geothermal_nodes {
        let pos = geo::lat_lon_to_xyz(node.lat, node.lon, 1.008);
        let size = geo::marker_size_from_capacity(node.capacity_mw);
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(gc[0], gc[1], gc[2]),
            emissive: LinearRgba::new(gc[0] * 0.4, gc[1] * 0.4, gc[2] * 0.4, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Geothermal, name: node.name.clone() },
            SurfaceLod,
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
            emissive: LinearRgba::new(tc[0] * 0.5, tc[1] * 0.5, tc[2] * 0.5, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.014)),
            DataMarker { layer: Layer::TerraLumina, name: site.name.clone() },
            SurfaceLod,
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
            emissive: LinearRgba::new(vc[0] * 0.3, vc[1] * 0.3, vc[2] * 0.3, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(0.010)),
            DataMarker { layer: Layer::ResontiaVaults, name: vault.name.clone() },
            SurfaceLod,
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
            TemporalW { year: deposit.discovery_year as f64 },
            SurfaceLod,
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
                SurfaceLod,
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
            emissive: LinearRgba::new(nc[0] * 0.4, nc[1] * 0.4, nc[2] * 0.4, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Nuclear, name: site.name.clone() },
            TemporalW { year: site.commission_year as f64 },
            SurfaceLod,
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
        let size = 0.015 + stress.allostatic_load * 0.020;
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
            SurfaceLod,
            AtlasEntity,
        ));
        // Translucent stress halo — larger when under more stress
        if stress.allostatic_load > 0.3 {
            let halo_size = 0.03 + stress.allostatic_load * 0.05;
            let alpha = stress.allostatic_load * 0.12;
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
                SurfaceLod,
                AtlasEntity,
            ));
        }
    }

    // ─── Solar System Bodies ────────────────────────────────────
    let bodies = terra_atlas_core::solar_system::solar_system_bodies();
    let body_mesh = meshes.add(Sphere::new(1.0).mesh().uv(32, 32)); // higher detail for textured planets
    for body in &bodies {
        let pos = terra_atlas_core::solar_system::body_position(body, 0.0);
        let texture: Handle<Image> = asset_server.load(format!("textures/{}", body.texture));
        let mat = if body.is_sun {
            materials.add(StandardMaterial {
                base_color: Color::linear_rgba(1.0, 0.85, 0.4, 0.5),
                base_color_texture: Some(texture),
                emissive: LinearRgba::new(1.0, 0.7, 0.2, 1.0),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            })
        } else {
            // Real planet textures with subtle emissive for visibility in space
            materials.add(StandardMaterial {
                base_color: Color::linear_rgba(0.9, 0.9, 0.9, 0.85),
                base_color_texture: Some(texture),
                emissive: LinearRgba::new(0.08, 0.08, 0.08, 1.0),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                ..default()
            })
        };
        commands.spawn((
            Mesh3d(body_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2])
                .with_scale(Vec3::splat(body.visual_radius)),
            AtlasEntity,
        ));
    }

    // ─── Governance participation markers ──────────────────────
    let gov_pulses = terra_atlas_core::mycelix_flows::simulate_governance_pulses();
    let gov_mesh = meshes.add(Sphere::new(1.0).mesh().uv(8, 8));
    for pulse in &gov_pulses {
        let pos = geo::lat_lon_to_xyz(pulse.lat, pulse.lon, 1.025);
        let c = terra_atlas_core::mycelix_flows::governance_color(pulse.participation);
        let size = 0.012 + pulse.participation * 0.018;
        let mat = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(c[0], c[1], c[2]),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(gov_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(pos[0], pos[1], pos[2]).with_scale(Vec3::splat(size)),
            DataMarker { layer: Layer::Regions, name: format!("{} ({}% participation)", pulse.name, (pulse.participation * 100.0) as u32) },
            SurfaceLod,
            AtlasEntity,
        ));
    }

    // Store data for arc rendering (gizmos are immediate-mode)
    commands.insert_resource(AtlasData { data });

    info!("[atlas] Globe view: {marker_count} markers + {} stress + {} bodies + {} governance — Esc to return",
        stress_data.len(), bodies.len(), gov_pulses.len());
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
        .take(8) // top 8 sites — fewer arcs = less pole convergence
        .map(|s| (s.lat, s.lon, s.capacity_mw))
        .collect();
    let trades = terra_atlas_core::energy_trading::simulate_trades(&trade_sites, t as f64);
    let trade_color = Color::linear_rgba(0.2, 1.0, 0.3, 0.8); // bright green, more opaque
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

    // TEND time-banking flows — Mycelix lime arcs with animated packets
    let tend_flows = terra_atlas_core::mycelix_flows::simulate_tend_flows();
    for (fi, flow) in tend_flows.iter().enumerate() {
        let from = geo::lat_lon_to_xyz(flow.from_lat, flow.from_lon, 1.0);
        let to = geo::lat_lon_to_xyz(flow.to_lat, flow.to_lon, 1.0);
        let dist = terra_atlas_core::geo::haversine_km(flow.from_lat, flow.from_lon, flow.to_lat, flow.to_lon);
        let peak = geo::arc_peak_height(dist) * 1.5;
        let segments = 16u32;
        let arc = terra_atlas_core::geometry::generate_arc(from, to, peak, segments);
        let packet_pos = ((t * 0.25 + fi as f32 * 0.5) % 1.0).abs();
        let packet_seg = (packet_pos * segments as f32) as usize;
        for i in 0..segments as usize {
            let a = Vec3::new(arc[i * 3], arc[i * 3 + 1], arc[i * 3 + 2]);
            let b = Vec3::new(arc[(i + 1) * 3], arc[(i + 1) * 3 + 1], arc[(i + 1) * 3 + 2]);
            let dist_to_packet = (i as f32 - packet_seg as f32).abs() / segments as f32;
            let brightness = 0.3 + 0.7 * (-dist_to_packet * 6.0).exp();
            gizmos.line(a, b, Color::linear_rgba(0.486 * brightness, 0.988 * brightness, 0.0, brightness * 0.7));
        }
    }

    // Maglev corridors — amber arcs with animated data packets
    for (ci, corridor) in atlas_data.data.maglev_corridors.iter().enumerate() {
        let from = geo::lat_lon_to_xyz(corridor.from_lat, corridor.from_lon, 1.0);
        let to = geo::lat_lon_to_xyz(corridor.to_lat, corridor.to_lon, 1.0);
        let peak = geo::arc_peak_height(corridor.distance_km);
        let segments = 24u32;
        let arc = terra_atlas_core::geometry::generate_arc(from, to, peak, segments);

        // Data packet position (0.0-1.0) travels along the arc
        let packet_pos = ((t * 0.3 + ci as f32 * 0.4) % 1.0).abs();
        let packet_seg = (packet_pos * segments as f32) as usize;

        for i in 0..segments as usize {
            let a = Vec3::new(arc[i * 3], arc[i * 3 + 1], arc[i * 3 + 2]);
            let b = Vec3::new(arc[(i + 1) * 3], arc[(i + 1) * 3 + 1], arc[(i + 1) * 3 + 2]);
            // Bright pulse at packet position, dim elsewhere
            let dist_to_packet = (i as f32 - packet_seg as f32).abs() / segments as f32;
            let brightness = 0.4 + 0.6 * (-dist_to_packet * 8.0).exp();
            gizmos.line(a, b, Color::linear_rgba(1.0 * brightness, 0.8 * brightness, 0.1 * brightness, brightness));
        }
    }

    // Supply routes — cyan arcs, dimmer
    let supply_color = Color::linear_rgba(0.0, 0.5, 0.8, 0.3); // cyan, dimmer — background layer
    for route in atlas_data.data.supply_routes.iter().take(5) { // limit to 5 routes
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

/// 4D temporal visibility — markers emerge from the 4th dimension based on timeline year.
/// Uses Projector4D hyperplane slicing: markers with TemporalW close to the current
/// timeline year are visible; those far from the slice fade out.
pub fn temporal_4d_system(
    timeline: Res<TimelineState>,
    mut markers: Query<(&TemporalW, &mut Visibility), With<AtlasEntity>>,
) {
    // Map timeline year (0-500) to absolute year (1900-2400)
    let current_year = 1900.0 + timeline.year as f64 * 1.0;

    // Create a Projector4D with the timeline as W-slice
    let projector = symtropy_render_bridge::Projector4D::new(
        current_year,
        100.0, // slice_thickness: markers within ±100 years are visible
        1.0,
    );

    for (temporal, mut vis) in markers.iter_mut() {
        let point = symtropy_math::Point::new([0.0, 0.0, 0.0, temporal.year]);
        let alpha = projector.alpha(&point);

        *vis = if alpha > 0.05 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Aesthetic switcher — number keys 1-5 cycle visual presets.
pub fn aesthetic_switch_system(
    kb: Res<ButtonInput<KeyCode>>,
    mut current: ResMut<CurrentAesthetic>,
) {
    use terra_atlas_core::aesthetics::Aesthetic;
    let new = if kb.just_pressed(KeyCode::Digit1) { Some(Aesthetic::Holographic) }
        else if kb.just_pressed(KeyCode::Digit2) { Some(Aesthetic::Satellite) }
        else if kb.just_pressed(KeyCode::Digit3) { Some(Aesthetic::Procedural) }
        else if kb.just_pressed(KeyCode::Digit4) { Some(Aesthetic::Minimal) }
        else if kb.just_pressed(KeyCode::Digit5) { Some(Aesthetic::Night) }
        else { None };

    if let Some(aesthetic) = new {
        if aesthetic != current.aesthetic {
            current.aesthetic = aesthetic;
            current.changed = true;
            info!("[atlas] Aesthetic: {} (press 1-5 to switch)", aesthetic.label());
        }
    }
}

/// Apply aesthetic changes to globe materials when preset changes.
pub fn aesthetic_apply_system(
    mut current: ResMut<CurrentAesthetic>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    globe_q: Query<&MeshMaterial3d<StandardMaterial>, With<Globe>>,
    atmo_q: Query<&MeshMaterial3d<StandardMaterial>, With<Atmosphere>>,
) {
    if !current.changed { return; }
    current.changed = false;

    let config = terra_atlas_core::aesthetics::config_for(current.aesthetic);

    // Update globe material
    for mat_handle in globe_q.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let c = config.globe.base_color;
            mat.base_color = Color::linear_rgba(c[0], c[1], c[2], c[3]);
            let e = config.globe.emissive;
            mat.emissive = LinearRgba::new(e[0], e[1], e[2], e[3]);
            mat.unlit = config.globe.unlit;
            if !config.globe.unlit {
                mat.perceptual_roughness = config.globe.roughness;
                mat.metallic = config.globe.metalness;
            }
            if config.globe.alpha_blend {
                mat.alpha_mode = AlphaMode::Blend;
            } else {
                mat.alpha_mode = AlphaMode::Opaque;
            }
        }
    }

    // Update atmosphere/fresnel materials
    for mat_handle in atmo_q.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let f = &config.fresnel;
            mat.base_color = Color::linear_rgba(f.color[0], f.color[1], f.color[2], f.color[3]);
            mat.emissive = LinearRgba::new(f.emissive[0], f.emissive[1], f.emissive[2], f.emissive[3]);
        }
    }
}

/// LOD visibility — toggle markers based on camera zoom distance.
/// Mutually exclusive: Orbit = heat blobs, Surface = markers, Atmosphere = neither.
pub fn lod_visibility_system(
    camera: Query<&Transform, With<OrbitalCamera>>,
    mut surface_markers: Query<&mut Visibility, (With<SurfaceLod>, Without<OrbitLod>)>,
    mut orbit_blobs: Query<&mut Visibility, (With<OrbitLod>, Without<SurfaceLod>)>,
) {
    let Ok(cam_tf) = camera.single() else { return };
    let distance = cam_tf.translation.length();
    let lod = LodLevel::from_camera_distance(distance);

    let show_surface = matches!(lod, LodLevel::Surface);
    let show_orbit = matches!(lod, LodLevel::Orbit);
    // Atmosphere = clean gap — only arcs + globe visible

    for mut vis in surface_markers.iter_mut() {
        *vis = if show_surface { Visibility::Visible } else { Visibility::Hidden };
    }
    for mut vis in orbit_blobs.iter_mut() {
        *vis = if show_orbit { Visibility::Visible } else { Visibility::Hidden };
    }
}

/// [8] Sacred Stillness breathing — atmosphere shells pulse on 8-second cycle.
/// Only pulses the atmosphere (not markers, to avoid scale drift).
pub fn holographic_pulse_system(
    time: Res<Time>,
    mut atmospheres: Query<&mut Transform, (With<Atmosphere>, With<AtlasEntity>)>,
) {
    let t = time.elapsed_secs();
    // 8-second Sacred Stillness breathing cycle
    let breath = 1.0 + 0.03 * (t * std::f32::consts::TAU / 8.0).sin();

    for mut tf in atmospheres.iter_mut() {
        let base = tf.scale.x.max(0.5); // avoid zero scale
        // Apply breathing to atmosphere shells (they started at ~1.03-1.05 scale)
        tf.scale = Vec3::splat(base.signum() * breath * 1.04);
    }
}

/// Animate celestial bodies along their orbits (drawn as gizmo orbit rings).
pub fn celestial_orbit_system(
    mut gizmos: Gizmos,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();
    let bodies = terra_atlas_core::solar_system::solar_system_bodies();

    for body in &bodies {
        // Draw faint orbit ring
        let segments = 64;
        let orbit_color = if body.is_sun {
            Color::linear_rgba(1.0, 0.8, 0.3, 0.03) // barely visible
        } else {
            Color::linear_rgba(0.3, 0.4, 0.5, 0.02) // ghost lines
        };

        for i in 0..segments {
            let a0 = i as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            let p0 = Vec3::new(
                body.orbit_radius * a0.cos(),
                body.y_offset,
                body.orbit_radius * a0.sin(),
            );
            let p1 = Vec3::new(
                body.orbit_radius * a1.cos(),
                body.y_offset,
                body.orbit_radius * a1.sin(),
            );
            gizmos.line(p0, p1, orbit_color);
        }
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
    hidden_huds: Query<Entity, With<crate::systems::rendering::HudText>>,
) {
    for entity in atlas_entities.iter() {
        commands.entity(entity).despawn();
    }

    for entity in hidden_cameras.iter() {
        commands.entity(entity).insert(Visibility::Visible);
    }
    // Restore dungeon HUD
    for entity in hidden_huds.iter() {
        commands.entity(entity).insert(Visibility::Visible);
    }

    // Restore dungeon background color and remove atlas data
    commands.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)));
    commands.remove_resource::<AtlasData>();

    info!("[atlas] Returned to dungeon");
}
