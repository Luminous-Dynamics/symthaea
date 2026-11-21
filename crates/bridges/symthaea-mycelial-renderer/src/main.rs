// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run a native 3D Mycelial Presence preview.
//!
//! Usage:
//! cargo run --manifest-path crates/bridges/symthaea-mycelial-renderer/Cargo.toml -- --mode ambient
//!
//! --mode accepts ambient, cognitive, or conversational.
//! Pass --reduced-motion to disable all renderer-local motion.

use bevy::{core_pipeline::tonemapping::Tonemapping, post_process::bloom::Bloom, prelude::*};
use std::f32::consts::TAU;
use symthaea_canvas::{
    EmbodimentSnapshot, MycelialFilament, MycelialTubeMesh, OrbitalCell, PresenceMode,
    TubeTessellationConfig, build_mycelial_model, tessellate_mycelial_model,
};
use symthaea_mycelial_renderer::into_bevy_mesh;

#[derive(Resource, Clone, Copy)]
struct RendererConfig {
    mode: PresenceMode,
    reduced_motion: bool,
}

#[derive(Component)]
struct CorePulse;

#[derive(Component)]
struct OrbitalMotion {
    phase: f32,
    radius_x: f32,
    radius_z: f32,
    height: f32,
    speed: f32,
}

fn main() {
    let config = parse_args();
    App::new()
        .insert_resource(config)
        .insert_resource(ClearColor(Color::srgb(0.004, 0.008, 0.025)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea — Mycelial Presence".to_string(),
                resolution: (1120, 900).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup_presence)
        .add_systems(Update, animate_presence)
        .run();
}

fn parse_args() -> RendererConfig {
    let mut mode = PresenceMode::Ambient;
    let mut reduced_motion = false;
    let args: Vec<String> = std::env::args().collect();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--reduced-motion" => reduced_motion = true,
            "--mode" if index + 1 < args.len() => {
                mode = match args[index + 1].as_str() {
                    "ambient" => PresenceMode::Ambient,
                    "cognitive" => PresenceMode::Cognitive,
                    "conversational" => PresenceMode::Conversational,
                    other => {
                        eprintln!("Unknown mode '{other}', using ambient");
                        PresenceMode::Ambient
                    }
                };
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    RendererConfig {
        mode,
        reduced_motion,
    }
}

fn setup_presence(
    mut commands: Commands,
    config: Res<RendererConfig>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Tonemapping::TonyMcMapface,
        Bloom::NATURAL,
        Transform::from_xyz(13.0, 9.0, 19.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 3_200.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-4.0, 7.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 1_250.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, -2.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let snapshot = EmbodimentSnapshot::default();
    let model = build_mycelial_model(&snapshot, config.mode);
    let cpu_mesh = tessellate_mycelial_model(&model, TubeTessellationConfig::default())
        .expect("bounded canonical model must tessellate");
    spawn_filament_canopy(&mut commands, &mut meshes, &mut materials, cpu_mesh);
    spawn_branch_junctions(&mut commands, &mut meshes, &mut materials, &model.filaments);
    spawn_terminal_spores(&mut commands, &mut meshes, &mut materials, &model.filaments);

    spawn_core(
        &mut commands,
        &mut meshes,
        &mut materials,
        config.reduced_motion,
    );
    spawn_orbital_cells(
        &mut commands,
        &mut meshes,
        &mut materials,
        &model.orbital_cells,
        config.reduced_motion,
    );
}

fn spawn_filament_canopy(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    cpu_mesh: MycelialTubeMesh,
) {
    let mesh = meshes.add(into_bevy_mesh(cpu_mesh));
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::rgb(0.012, 0.16, 0.21),
        metallic: 0.28,
        perceptual_roughness: 0.34,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_scale(Vec3::splat(0.035)),
    ));
}

fn spawn_branch_junctions(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    filaments: &[MycelialFilament],
) {
    let junction_mesh = meshes.add(Sphere::new(1.0).mesh().ico(2).expect("valid junction mesh"));
    let junction_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.76, 0.84),
        emissive: LinearRgba::rgb(0.028, 0.22, 0.30),
        metallic: 0.18,
        perceptual_roughness: 0.28,
        ..default()
    });

    for filament in filaments.iter().filter(|filament| !filament.terminal) {
        let position = filament.end;
        let radius = filament.radius_end * 0.035 * 1.55;
        commands.spawn((
            Mesh3d(junction_mesh.clone()),
            MeshMaterial3d(junction_material.clone()),
            Transform::from_xyz(position.x * 0.035, position.z * 0.035, position.y * 0.035)
                .with_scale(Vec3::splat(radius)),
        ));
    }
}

fn spawn_terminal_spores(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    filaments: &[MycelialFilament],
) {
    let spore_mesh = meshes.add(Sphere::new(0.058).mesh().ico(2).expect("valid spore mesh"));
    let spore_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.86, 0.90),
        emissive: LinearRgba::rgb(0.12, 0.70, 0.82),
        metallic: 0.08,
        perceptual_roughness: 0.22,
        ..default()
    });

    for filament in filaments.iter().filter(|filament| filament.terminal) {
        let position = filament.end;
        commands.spawn((
            Mesh3d(spore_mesh.clone()),
            MeshMaterial3d(spore_material.clone()),
            Transform::from_xyz(position.x * 0.035, position.z * 0.035, position.y * 0.035),
        ));
    }
}

fn spawn_core(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    reduced_motion: bool,
) {
    let sphere = meshes.add(Sphere::new(0.72).mesh().ico(5).expect("valid nucleus mesh"));
    let core_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.78, 0.80),
        emissive: LinearRgba::rgb(0.08, 1.4, 1.5),
        metallic: 0.36,
        perceptual_roughness: 0.20,
        ..default()
    });
    let mut core = commands.spawn((
        Mesh3d(sphere),
        MeshMaterial3d(core_material),
        Transform::default(),
    ));
    if !reduced_motion {
        core.insert(CorePulse);
    }

    let subcore_mesh = meshes.add(Sphere::new(0.34).mesh().ico(4).expect("valid subcore mesh"));
    let subcore_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.60, 0.20),
        emissive: LinearRgba::rgb(0.95, 0.34, 0.055),
        metallic: 0.16,
        perceptual_roughness: 0.25,
        ..default()
    });
    commands.spawn((
        Mesh3d(subcore_mesh),
        MeshMaterial3d(subcore_material),
        Transform::from_xyz(0.0, 0.02, 0.0),
    ));

    let membrane_mesh = meshes.add(
        Sphere::new(0.96)
            .mesh()
            .ico(4)
            .expect("valid membrane mesh"),
    );
    let membrane_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.12, 0.72, 0.82, 0.075),
        emissive: LinearRgba::rgb(0.015, 0.12, 0.16),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(membrane_mesh),
        MeshMaterial3d(membrane_material),
        Transform::from_scale(Vec3::new(1.05, 0.82, 1.14)),
    ));
}

fn spawn_orbital_cells(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    cells: &[OrbitalCell],
    reduced_motion: bool,
) {
    let cell_mesh = meshes.add(Sphere::new(0.085).mesh().ico(2).expect("valid cell mesh"));
    let cell_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.82, 0.88),
        emissive: LinearRgba::rgb(0.035, 0.30, 0.35),
        metallic: 0.12,
        perceptual_roughness: 0.28,
        ..default()
    });

    for (index, cell) in cells.iter().enumerate() {
        let translation = cell_position(cell);
        let mut entity = commands.spawn((
            Mesh3d(cell_mesh.clone()),
            MeshMaterial3d(cell_material.clone()),
            Transform::from_translation(translation)
                .with_scale(Vec3::splat(cell.radius * 0.035 / 0.085)),
        ));
        if !reduced_motion {
            let angle = index as f32 * TAU / cells.len().max(1) as f32;
            entity.insert(OrbitalMotion {
                phase: angle,
                radius_x: 1.7,
                radius_z: 1.2,
                height: translation.y,
                speed: if index % 2 == 0 { 0.09 } else { -0.065 },
            });
        }
    }
}

fn cell_position(cell: &OrbitalCell) -> Vec3 {
    Vec3::new(
        cell.position.x * 0.035,
        cell.position.z * 0.035,
        cell.position.y * 0.035,
    )
}

fn animate_presence(
    time: Res<Time>,
    mut pulse_query: Query<&mut Transform, With<CorePulse>>,
    mut orbit_query: Query<(&mut Transform, &OrbitalMotion)>,
) {
    let elapsed = time.elapsed_secs_wrapped();
    for mut transform in &mut pulse_query {
        let scale = 1.0 + 0.018 * (elapsed * 0.62).sin();
        transform.scale = Vec3::splat(scale);
    }

    for (mut transform, orbit) in &mut orbit_query {
        let angle = orbit.phase + elapsed * orbit.speed;
        transform.translation = Vec3::new(
            orbit.radius_x * angle.cos(),
            orbit.height + 0.08 * (angle * 1.7).sin(),
            orbit.radius_z * angle.sin(),
        );
    }
}
