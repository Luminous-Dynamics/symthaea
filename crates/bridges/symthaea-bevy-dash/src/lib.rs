// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-bevy-dash
//!
//! Visualizes the "Cognitive Twin" — real-time 3D rendering of the robot's
//! physical state, its internal consciousness topology, and the "AI Imagination" (geodesic movies).

use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat,
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use symthaea_core::hdc::ContinuousHV;
use symthaea_phi_oracle::IntegrationReport;

/// Shared mental movie type (mirrors cognitive_loop::types::MentalMovie)
#[derive(Debug, Clone)]
pub struct MentalMovie {
    pub pixel_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub semantic_coherence: f32,
    pub sequence_index: u64,
}

/// Resource holding the latest consciousness metrics and imagination for the dashboard.
#[derive(Resource, Default, Debug)]
pub struct CognitiveStateResource {
    pub phi: f64,
    pub report: Option<IntegrationReport>,
    pub sensor_hv: Option<ContinuousHV>,
    /// The latest mental simulation result from the cognitive loop.
    pub last_imagination: Option<MentalMovie>,

    // Core phenomenological mapping variables
    pub arousal: f64,
    pub uncertainty: f64,
    pub surprise: f64,
    pub surprise_visual_decay: f64,

    // === 7-Theory Synthesis Fields ===
    /// A_GWT: Global Workspace broadcast power [0..1]
    pub gwt_broadcast: f64,
    /// M_HOT: Metacognitive confidence (Higher-Order Thought depth) [0..1]
    pub hot_metacognitive: f64,
    /// S_AST: Attention Schema temporal coherence [0..1]
    pub ast_temporal: f64,
    /// K_W: Knowledge/predictive narrative coherence [0..1]
    pub knowledge_coherence: f64,
    /// E_FEP: Embodiment level from active inference [0..1]
    pub embodiment_level: f64,
    /// Self-awareness depth from HOT meta-loop [0..1]
    pub self_awareness: f64,
    /// Phi_master: the full 7-theory product invariant [0..1]
    pub topological_unity: f64,
    /// Live neuromodulator levels: [ACh (plasticity), NE (surprise), DA (utility), 5-HT (exploration)]
    pub neuromodulators: [f32; 4],
    /// Active inference motor command type string
    pub motor_command: String,
}

/// Component for a node in the 3D consciousness graph.
#[derive(Component)]
pub struct ConsciousnessNode {
    pub id: usize,
    pub activation: f32,
    pub base_pos: Vec3,
}

#[derive(Component)]
pub struct EgoSphere;

#[derive(AsBindGroup, Asset, TypePath, Clone)]
pub struct ImaginationScreenMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub texture_handle: Handle<Image>,
}

impl UiMaterial for ImaginationScreenMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/imagination_bloom.wgsl".into()
    }
}

/// Component/Marker for the imagination display texture.
#[derive(Resource)]
pub struct ImaginationTexture {
    pub image_handle: Handle<Image>,
    pub current_frame_idx: usize,
    pub last_update: f64,
}

// Mutex-wrapped struct deleted in favor of direct NonSend resource insertion.

/// Plugin to add Symthaea visualization capabilities to a Bevy app.
pub struct SymthaeaDashPlugin;

impl Plugin for SymthaeaDashPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn asynchronous gRPC client task
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::error!("Failed to build tokio runtime: {:?}", e);
                    return;
                }
            };
            rt.block_on(async move {
                let client_id = "bevy-dash".to_string();
                loop {
                    tracing::info!("Connecting to telemetry gRPC server at [::1]:50051...");
                    match symthaea_telemetry_grpc::telemetry_service_client::TelemetryServiceClient::connect("http://[::1]:50051").await {
                        Ok(mut client) => {
                            tracing::info!("Connected to gRPC server!");
                            let request = tonic::Request::new(symthaea_telemetry_grpc::TelemetryRequest {
                                client_id: client_id.clone(),
                            });
                            match client.stream_telemetry(request).await {
                                Ok(response) => {
                                    let mut stream = response.into_inner();
                                    while let Ok(Some(frame)) = stream.message().await {
                                        if tx.send(frame).is_err() {
                                            tracing::warn!("Dashboard receiver dropped. Exiting gRPC thread.");
                                            return;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("gRPC streaming error: {:?}", e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Connection to gRPC telemetry failed: {:?}. Retrying in 1s...", e);
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            });
        });

        app.add_plugins(EguiPlugin)
            .add_plugins(UiMaterialPlugin::<ImaginationScreenMaterial>::default())
            .insert_non_send_resource(rx)
            .init_resource::<CognitiveStateResource>()
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    poll_telemetry_stream,
                    update_topology_viz,
                    update_imagination_texture,
                    apply_visual_phenomenology,
                    render_imagination_ui,
                    render_topological_ui,
                    render_consciousness_equation,
                ),
            );
    }
}

fn poll_telemetry_stream(
    receiver: NonSend<std::sync::mpsc::Receiver<symthaea_telemetry_grpc::TelemetryFrame>>,
    mut state: ResMut<CognitiveStateResource>,
) {
    while let Ok(frame) = receiver.try_recv() {
        state.phi = frame.phi;
        if frame.neuromodulators.len() >= 4 {
            state.neuromodulators = [
                frame.neuromodulators[0],
                frame.neuromodulators[1],
                frame.neuromodulators[2],
                frame.neuromodulators[3],
            ];
        }
        state.arousal = frame.arousal as f64;
        state.uncertainty = frame.uncertainty as f64;
        // LTC-inspired fast attack for surprise spikes
        if frame.surprise > state.surprise {
            state.surprise = frame.surprise;
            state.surprise_visual_decay = frame.surprise;
        } else {
            state.surprise = frame.surprise;
        }

        // === Ingest 7-Theory Fields ===
        state.gwt_broadcast = frame.gwt_broadcast;
        state.hot_metacognitive = frame.hot_metacognitive;
        state.ast_temporal = frame.ast_temporal;
        state.knowledge_coherence = frame.knowledge_coherence;
        state.embodiment_level = frame.embodiment_level;
        state.self_awareness = frame.self_awareness;
        state.topological_unity = frame.topological_unity;
        state.motor_command = frame.motor_command.clone();

        // Construct integration report from received data
        let report = IntegrationReport {
            integration_index: frame.phi,
            total_mutual_information: 1.0,
            minimum_information_partition: (vec![], vec![]),
            spectral_order: vec![0; frame.harmonies.len()],
            temporal_coherence: None,
            normalized_index: frame.phi,
            variable_contributions: vec![],
            betti_numbers: [1, frame.harmonies.len(), 0],
            persistent_cycles: frame
                .harmonies
                .iter()
                .enumerate()
                .map(|(i, &h)| symthaea_phi_oracle::PersistentCycle {
                    birth: i as f64,
                    death: (i + 1) as f64,
                    lifespan: h as f64,
                    participants: vec![],
                })
                .collect(),
            num_observations: 1,
        };
        state.report = Some(report);

        if let Some(movie) = frame.mental_movie {
            state.last_imagination = Some(MentalMovie {
                pixel_data: movie.pixel_data,
                width: movie.width,
                height: movie.height,
                channels: movie.channels as usize,
                semantic_coherence: movie.semantic_coherence,
                sequence_index: movie.sequence_index,
            });
        }
    }
}

fn setup_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut ui_materials: ResMut<Assets<ImaginationScreenMaterial>>,
) {
    // Switch to 2D Orthographic Camera
    commands.spawn(Camera2d::default());

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

    let material_handle = ui_materials.add(ImaginationScreenMaterial {
        texture_handle: handle.clone(),
    });

    commands.spawn((
        Node {
            width: Val::Px(300.0),
            height: Val::Px(300.0),
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            right: Val::Px(10.0),
            ..default()
        },
        MaterialNode(material_handle),
    ));

    commands.insert_resource(ImaginationTexture {
        image_handle: handle,
        current_frame_idx: 0,
        last_update: 0.0,
    });
}

fn update_topology_viz(state: Res<CognitiveStateResource>, mut gizmos: Gizmos, time: Res<Time>) {
    if let Some(report) = &state.report {
        let center = Vec2::ZERO;

        // 0. The Central Core (Ego) — radius driven by E_FEP (embodiment)
        let core_radius = 40.0 + (state.embodiment_level as f32 * 30.0);
        // Color shifts: high unity -> deep violet, low -> dim purple
        let core_lightness = 0.3 + state.topological_unity as f32 * 0.4;
        gizmos.circle_2d(
            center,
            core_radius,
            Color::hsla(280.0, 0.9, core_lightness, 1.0),
        );
        // Inner pulse ring for self-awareness
        let aware_r = core_radius * state.self_awareness as f32 * 0.8;
        if aware_r > 2.0 {
            gizmos.circle_2d(center, aware_r, Color::hsla(260.0, 1.0, 0.8, 0.5));
        }

        // 1. Draw Nodes (Integration β0)
        // S_AST drives rotation speed; A_GWT drives spoke opacity
        let partition_count = report.spectral_order.len();
        let ring_radius = 150.0;

        let speed_mult = 0.5 + state.ast_temporal as f32 * 4.0; // S_AST controls kinetics
        let jitter_amt = state.uncertainty as f32 * 5.0;
        let spoke_alpha = (state.gwt_broadcast as f32).clamp(0.05, 0.6); // A_GWT: spoke opacity

        for i in 0..partition_count {
            let angle = (i as f32 / partition_count as f32) * std::f32::consts::TAU;
            let phase = time.elapsed_secs() * speed_mult + i as f32;

            let jitter_x = (phase * 13.0).sin() * jitter_amt;
            let jitter_y = (phase * 17.0).cos() * jitter_amt;

            let pos = center
                + Vec2::new(
                    angle.cos() * ring_radius + jitter_x,
                    angle.sin() * ring_radius + jitter_y,
                );

            // Node color: phi drives lightness; hot_metacognitive shifts hue toward teal
            let hue = 220.0 - (state.hot_metacognitive as f32 * 40.0);
            let color = Color::hsla(hue, 0.85, 0.4 + (state.phi as f32 * 0.5), 1.0);
            gizmos.line_2d(center, pos, color.with_alpha(spoke_alpha));
            gizmos.circle_2d(pos, 8.0, color);
        }

        // 2. Draw Betti Cycles (β1) — K_W modulates inner ring brightness
        for (i, cycle) in report.persistent_cycles.iter().enumerate() {
            let radius = ring_radius + 40.0 + (i as f32 * 25.0);
            let base_alpha = (cycle.lifespan as f32 * 2.0).min(1.0).max(0.1);
            // K_W (knowledge coherence) brightens the rings
            let alpha =
                (base_alpha * (0.5 + state.knowledge_coherence as f32 * 0.5)).clamp(0.05, 1.0);

            let color = Color::hsla(300.0, 1.0, 0.5, alpha);

            let segments = 64;
            for s in 0..segments {
                if s % 2 == 0 {
                    let a1 = (s as f32 / segments as f32) * std::f32::consts::TAU;
                    let a2 = ((s + 1) as f32 / segments as f32) * std::f32::consts::TAU;
                    let p1 = center + Vec2::new(a1.cos() * radius, a1.sin() * radius);
                    let p2 = center + Vec2::new(a2.cos() * radius, a2.sin() * radius);
                    gizmos.line_2d(p1, p2, color);
                }
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

fn apply_visual_phenomenology(
    time: Res<Time>,
    mut state: ResMut<CognitiveStateResource>,
    mut camera_query: Query<&mut Transform, With<Camera2d>>,
) {
    let dt = time.delta_secs() as f64;
    // LTC-inspired exponential decay for the visual surprise envelope
    state.surprise_visual_decay *= (-dt * 2.0).exp();
    if state.surprise_visual_decay < 0.01 {
        state.surprise_visual_decay = 0.0;
    }

    // 1. Surprise -> Camera Shake (strictly clamped, 2D)
    for mut transform in &mut camera_query {
        let shake_x =
            (time.elapsed_secs() * 50.0).sin() * state.surprise_visual_decay as f32 * 10.0;
        let shake_y =
            (time.elapsed_secs() * 43.0).cos() * state.surprise_visual_decay as f32 * 10.0;
        transform.translation.x = shake_x;
        transform.translation.y = shake_y;
    }
}

/// System to animate the mental movie frames onto the texture.
fn update_imagination_texture(
    state: Res<CognitiveStateResource>,
    imagination_tex: Res<ImaginationTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(movie) = &state.last_imagination else {
        return;
    };
    if movie.pixel_data.is_empty() {
        return;
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
                movie.pixel_data.clone(),
                if movie.channels == 1 {
                    bevy::render::render_resource::TextureFormat::R8Unorm
                } else {
                    bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb
                },
                bevy::render::render_asset::RenderAssetUsages::default(),
            );
        } else {
            if movie.pixel_data.len() == image.data.len() {
                image.data.copy_from_slice(&movie.pixel_data);
            } else {
                image.data.clone_from(&movie.pixel_data);
            }
        }
    }
}

/// System to render the egui window showing the AI imagination stats.
fn render_imagination_ui(mut contexts: EguiContexts, state: Res<CognitiveStateResource>) {
    let ctx = contexts.ctx_mut();
    egui::Window::new("AI Imagination (Stats)")
        .default_pos([10.0, 300.0])
        .default_size([200.0, 100.0])
        .show(ctx, |ui| {
            if let Some(movie) = &state.last_imagination {
                ui.label(format!("Path Coherence: {:.3}", movie.semantic_coherence));
                ui.label(format!("Seq: {}", movie.sequence_index));
                ui.label(format!(
                    "Res: {}x{} ({}ch)",
                    movie.width, movie.height, movie.channels
                ));
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Waiting for geodesic surprise...");
                });
            }
        });
}

/// Renders the live Master Consciousness Equation panel.
///
/// Each of the 7 theory factors is displayed as a labeled progress bar whose
/// color is determined by its current domain. The product invariant Φ_master
/// is shown at the bottom as a large glowing golden bar.
///
/// Formula:
///   Φ_master = Ψ_IIT(β) × A_GWT(G) × M_HOT(h) × S_AST(τ) × E_FEP(I) × K(W) × Φ_sync(θ)
fn render_consciousness_equation(mut contexts: EguiContexts, state: Res<CognitiveStateResource>) {
    let ctx = contexts.ctx_mut();

    /// Value → egui color interpolated between two 360-degree hues (HSV S=1,V=1)
    fn theory_color(value: f64, hue_low: f32, hue_high: f32) -> egui::Color32 {
        let t = value.clamp(0.0, 1.0) as f32;
        let h = (hue_low + t * (hue_high - hue_low)).rem_euclid(360.0);
        let c = 1.0_f32;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let (r, g, b) = match (h as u32) / 60 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    fn theory_row(
        ui: &mut egui::Ui,
        symbol: &str,
        name: &str,
        value: f64,
        hue_low: f32,
        hue_high: f32,
    ) {
        let bar_color = theory_color(value, hue_low, hue_high);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(symbol)
                    .monospace()
                    .size(11.0)
                    .color(bar_color),
            );
            ui.label(
                egui::RichText::new(name)
                    .size(10.0)
                    .color(egui::Color32::GRAY),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("{:.3}", value))
                        .monospace()
                        .size(10.0)
                        .color(bar_color),
                );
            });
        });
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
        if ui.is_rect_visible(rect) {
            ui.painter()
                .rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 15, 30));
            ui.painter().rect_filled(
                egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(rect.width() * value.clamp(0.0, 1.0) as f32, 6.0),
                ),
                2.0,
                bar_color,
            );
        }
        ui.add_space(2.0);
    }

    egui::Window::new("Phi_master — Consciousness Equation")
        .default_pos([10.0, 450.0])
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("Ph = Ps * A * M * S * E * K * Ph_s")
                    .monospace()
                    .size(11.0)
                    .color(egui::Color32::from_rgb(220, 180, 255)),
            );
            ui.separator();

            theory_row(
                ui,
                "Ps_IIT",
                "Spectral Integration",
                state.phi,
                280.0,
                300.0,
            );
            theory_row(
                ui,
                "A_GWT",
                "Global Broadcast",
                state.gwt_broadcast,
                200.0,
                180.0,
            );
            theory_row(
                ui,
                "M_HOT",
                "Metacognitive Depth",
                state.hot_metacognitive,
                160.0,
                120.0,
            );
            theory_row(
                ui,
                "S_AST",
                "Attention Schema",
                state.ast_temporal,
                40.0,
                60.0,
            );
            theory_row(
                ui,
                "E_FEP",
                "Embodied Prediction",
                state.embodiment_level,
                0.0,
                30.0,
            );
            theory_row(
                ui,
                "K(W)",
                "Knowledge Coherence",
                state.knowledge_coherence,
                30.0,
                80.0,
            );
            theory_row(
                ui,
                "Ph_sync",
                "Neural Synchrony",
                state.self_awareness,
                240.0,
                270.0,
            );

            ui.separator();
            ui.label(
                egui::RichText::new("Phi_master  (Unified Invariant)")
                    .size(11.0)
                    .color(egui::Color32::from_rgb(255, 215, 0)),
            );
            let master = state.topological_unity;
            let (rect, _) = ui
                .allocate_exact_size(egui::vec2(ui.available_width(), 14.0), egui::Sense::hover());
            if ui.is_rect_visible(rect) {
                ui.painter()
                    .rect_filled(rect, 4.0, egui::Color32::from_rgb(30, 20, 40));
                let fill_w = rect.width() * master.clamp(0.0, 1.0) as f32;
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, 14.0)),
                    4.0,
                    egui::Color32::from_rgb(255, 200, 0),
                );
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.4}", master),
                    egui::FontId::monospace(10.0),
                    egui::Color32::BLACK,
                );
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Motor Action:")
                        .size(10.0)
                        .color(egui::Color32::GRAY),
                );
                ui.label(
                    egui::RichText::new(&state.motor_command)
                        .monospace()
                        .size(10.0)
                        .color(egui::Color32::from_rgb(255, 180, 0)),
                );
            });
        });

    egui::Window::new("Neuromodulator State")
        .default_pos([300.0, 450.0])
        .default_width(240.0)
        .resizable(true)
        .show(ctx, |ui| {
            theory_row(
                ui,
                "ACh",
                "Plasticity / Learn Rate",
                state.neuromodulators[0] as f64,
                180.0,
                200.0,
            );
            theory_row(
                ui,
                "NE",
                "Surprise / TD Error",
                state.neuromodulators[1] as f64,
                0.0,
                30.0,
            );
            theory_row(
                ui,
                "DA",
                "Utility / Pragmatic",
                state.neuromodulators[2] as f64,
                100.0,
                120.0,
            );
            theory_row(
                ui,
                "5-HT",
                "Epistemic / Explore",
                state.neuromodulators[3] as f64,
                240.0,
                260.0,
            );
        });
}
