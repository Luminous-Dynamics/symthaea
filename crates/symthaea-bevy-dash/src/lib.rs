// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-bevy-dash
//!
//! Visualizes the "Cognitive Twin" — real-time 3D rendering of the robot's
//! physical state, its internal consciousness topology, and the "AI Imagination" (geodesic movies).

use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use symthaea_core::hdc::ContinuousHV;
use symthaea_phi_oracle::IntegrationReport;

/// Shared mental movie type (mirrors cognitive_loop::types::MentalMovie)
#[derive(Debug, Clone)]
pub struct MentalMovie {
    pub frames: Vec<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub path_length: usize,
    pub semantic_coherence: f32,
}

/// Resource holding the latest consciousness metrics and imagination for the dashboard.
#[derive(Resource, Default, Debug)]
pub struct CognitiveStateResource {
    pub phi: f64,
    pub report: Option<IntegrationReport>,
    pub sensor_hv: Option<ContinuousHV>,
    /// The latest mental simulation result from the cognitive loop.
    pub last_imagination: Option<MentalMovie>,
}

/// Component for a node in the 3D consciousness graph.
#[derive(Component)]
pub struct ConsciousnessNode {
    pub id: usize,
    pub activation: f32,
}

/// Component/Marker for the imagination display texture.
#[derive(Resource)]
pub struct ImaginationTexture {
    pub image_handle: Handle<Image>,
    pub current_frame_idx: usize,
    pub last_update: f64,
}

/// Plugin to add Symthaea visualization capabilities to a Bevy app.
pub struct SymthaeaDashPlugin;

impl Plugin for SymthaeaDashPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .init_resource::<CognitiveStateResource>()
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    update_topology_viz,
                    animate_nodes,
                    update_imagination_texture,
                    render_imagination_ui,
                    render_topological_ui,
                ),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-5.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Placeholder for consciousness center
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.5))),
        MeshMaterial3d(materials.add(Color::srgb(0.2, 0.5, 1.0))),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Create a 64x64 texture placeholder for AI Imagination
    let initial_image = Image::new_fill(
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );
    let handle = images.add(initial_image);
    commands.insert_resource(ImaginationTexture {
        image_handle: handle,
        current_frame_idx: 0,
        last_update: 0.0,
    });
}

fn update_topology_viz(
    mut commands: Commands,
    state: Res<CognitiveStateResource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    node_query: Query<Entity, With<ConsciousnessNode>>,
    cycle_query: Query<Entity, With<BettiCycle>>,
) {
    if let Some(report) = &state.report {
        // 1. Update Nodes (Integration β0)
        let partition_count = report.spectral_order.len();
        if node_query.iter().count() != partition_count {
            for entity in &node_query {
                commands.entity(entity).despawn();
            }

            let radius = 3.0;
            for i in 0..partition_count {
                let angle = (i as f32 / partition_count as f32) * std::f32::consts::TAU;
                let pos = Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0);
                let color = Color::hsla(220.0, 0.8, 0.5 + (state.phi as f32 * 0.5), 1.0);

                commands.spawn((
                    Mesh3d(meshes.add(Sphere::new(0.2))),
                    MeshMaterial3d(materials.add(color)),
                    Transform::from_translation(pos),
                    ConsciousnessNode {
                        id: i,
                        activation: state.phi as f32,
                    },
                ));
            }
        }

        // 2. Update Persistent Cycles (β1 with Filtration)
        let persistent_count = report.persistent_cycles.len();
        if cycle_query.iter().count() != persistent_count {
            for entity in &cycle_query {
                commands.entity(entity).despawn();
            }

            for (i, cycle) in report.persistent_cycles.iter().enumerate() {
                let radius = 4.0 + (i as f32 * 0.5);
                // Map lifespan to Alpha and Emissive (Stronger persistence = Clearer signal)
                let alpha = (cycle.lifespan as f32 * 2.0).min(1.0).max(0.1);
                let emissive_mult = (cycle.lifespan as f32 * 10.0).min(5.0);

                commands.spawn((
                    Mesh3d(meshes.add(Torus::new(0.05, radius))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::hsla(300.0, 1.0, 0.5, alpha),
                        emissive: LinearRgba::from(Color::hsla(300.0, 1.0, 0.5, 1.0))
                            * emissive_mult,
                        alpha_mode: AlphaMode::Blend,
                        ..default()
                    })),
                    Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                    BettiCycle {
                        birth: cycle.birth as usize,
                        lifespan: cycle.lifespan as f32,
                    },
                ));
            }
        }
    }
}

/// System to render topological metrics (Betti Numbers) in an Egui window.

/// System to render topological metrics (Betti Numbers) in an Egui window.

#[derive(bevy::prelude::Component, Debug, Clone, Copy)]
pub struct BettiCycle {
    pub birth: usize,
    pub lifespan: f32,
}

/// System to render topological metrics (Betti Numbers) in an Egui window.

/// System to render topological metrics (Betti Numbers) in an Egui window.

/// System to render topological metrics (Betti Numbers) in an Egui window.
fn render_topological_ui(mut contexts: EguiContexts, state: Res<CognitiveStateResource>) {
    let ctx = contexts.ctx_mut();
    egui::Window::new("Topological Manifold")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            if let Some(report) = &state.report {
                ui.heading("Betti Numbers (Filtration Scan)");
                ui.label(format!("β₀ (Components): {}", report.betti_numbers[0]));
                ui.label(format!("β₁ (Cycles):     {}", report.betti_numbers[1]));
                ui.label(format!(
                    "Persistent Holes: {}",
                    report.persistent_cycles.len()
                ));

                ui.separator();
                ui.heading("Persistence Barcode");
                for (i, cycle) in report.persistent_cycles.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("C{}", i));
                        let rect = ui.available_rect_before_wrap();
                        let width = rect.width() * (cycle.lifespan as f32);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(rect.min, egui::vec2(width, 10.0)),
                            2.0,
                            egui::Color32::from_rgb(200, 50, 200),
                        );
                    });
                }
            } else {
                ui.label("Waiting for system integration...");
            }
        });
}

fn animate_nodes(time: Res<Time>, mut query: Query<&mut Transform, With<ConsciousnessNode>>) {
    for mut transform in &mut query {
        transform.translation.y += (time.elapsed_secs() + transform.translation.x).sin() * 0.01;
    }
}

/// System to animate the mental movie frames onto the texture.
fn update_imagination_texture(
    time: Res<Time>,
    state: Res<CognitiveStateResource>,
    imagination_tex: Res<ImaginationTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(movie) = &state.last_imagination else {
        return;
    };
    if movie.frames.is_empty() {
        return;
    }

    let frame_idx = (time.elapsed_secs() * 12.0) as usize % movie.frames.len();
    let current_frame = &movie.frames[frame_idx];

    let mut rgba_data = Vec::with_capacity(movie.width as usize * movie.height as usize * 4);
    if movie.channels == 3 {
        for chunk in current_frame.chunks_exact(3) {
            rgba_data.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
        }
    } else {
        rgba_data = current_frame.clone();
    }

    if let Some(image) = images.get_mut(&imagination_tex.image_handle) {
        if image.size().x != movie.width || image.size().y != movie.height {
            *image = Image::new(
                bevy::render::render_resource::Extent3d {
                    width: movie.width,
                    height: movie.height,
                    depth_or_array_layers: 1,
                },
                bevy::render::render_resource::TextureDimension::D2,
                rgba_data,
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                bevy::render::render_asset::RenderAssetUsages::default(),
            );
        } else {
            image.data = rgba_data;
        }
    }
}

/// System to render the egui window showing the AI imagination.
fn render_imagination_ui(
    mut contexts: EguiContexts,
    state: Res<CognitiveStateResource>,
    imagination_tex: Res<ImaginationTexture>,
) {
    let texture_id = contexts.add_image(imagination_tex.image_handle.clone_weak());
    let ctx = contexts.ctx_mut();
    egui::Window::new("AI Imagination (Mental Movie)")
        .default_size([300.0, 300.0])
        .show(ctx, |ui| {
            if let Some(movie) = &state.last_imagination {
                ui.label(format!("Path Coherence: {:.3}", movie.semantic_coherence));
                ui.label(format!("Horizon: {} steps", movie.path_length));
                ui.label(format!(
                    "Res: {}x{} ({}ch)",
                    movie.width, movie.height, movie.channels
                ));
                ui.image(egui::load::SizedTexture::new(texture_id, [256.0, 256.0]));
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Waiting for geodesic surprise...");
                });
            }
        });
}
