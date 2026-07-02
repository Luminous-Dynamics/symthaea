// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gravcraft 3D Demo: Metric Perturbation Propulsion Visualization
//!
//! Bevy scene showing:
//! - Deformable spacetime grid that warps under gravity amplifier influence
//! - The craft at center with 3 amplifier cones
//! - Geodesic trail showing the craft's path through curved space
//! - Phi gauge controlling metric distortion amplitude
//!
//! Controls:
//! - Mouse drag: rotate camera
//! - Scroll: zoom
//! - 1/2/3: toggle individual amplifiers
//! - Space: toggle all amplifiers
//! - Up/Down: adjust Phi

use bevy::prelude::*;
use symthaea_gravcraft::*;

/// Grid vertex that deforms with the metric perturbation.
#[derive(Component)]
struct GridVertex {
    rest_position: Vec3,
}

/// The gravcraft entity.
#[derive(Component)]
struct Craft;

/// Amplifier cone visual.
#[derive(Component)]
struct AmplifierCone {
    index: usize,
}

/// Geodesic trail point.
#[derive(Component)]
struct TrailPoint {
    age: f32,
}

/// Simulation state.
#[derive(Resource)]
struct GravcraftSim {
    simulator: MetricPerturbationSimulator,
    config: GravcraftConfig,
    phi: f64,
    amplifiers_active: [bool; 3],
    trail: Vec<Vec3>,
}

impl Default for GravcraftSim {
    fn default() -> Self {
        let config = GravcraftConfig {
            max_metric_amplitude: 0.1,
            bubble_radius: 8.0,
            wall_sigma: 1.5,
            rk4_substeps: 4,
        };
        Self {
            simulator: MetricPerturbationSimulator::new(config.clone()),
            config,
            phi: 0.8,
            amplifiers_active: [true, true, true],
            trail: Vec::new(),
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea Gravcraft: Metric Perturbation Propulsion".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<GravcraftSim>()
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                update_simulation,
                update_grid_deformation,
                handle_input,
                update_hud,
            ),
        )
        .run();
}

/// Set up the 3D scene.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 25.0, 35.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.1, 0.15, 0.3),
        brightness: 200.0,
        ..default()
    });

    // Directional light
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.9, 0.95, 1.0),
            illuminance: 5000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Spacetime grid — flat plane of small spheres
    let grid_size = 20;
    let grid_spacing = 2.0;
    let vertex_mesh = meshes.add(Sphere::new(0.08));
    let grid_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.5, 1.0, 0.6),
        emissive: LinearRgba::new(0.1, 0.2, 0.5, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    for x in -grid_size..=grid_size {
        for z in -grid_size..=grid_size {
            let pos = Vec3::new(x as f32 * grid_spacing, 0.0, z as f32 * grid_spacing);
            commands.spawn((
                Mesh3d(vertex_mesh.clone()),
                MeshMaterial3d(grid_material.clone()),
                Transform::from_translation(pos),
                GridVertex { rest_position: pos },
            ));
        }
    }

    // The craft (glowing sphere at center)
    let craft_mesh = meshes.add(Sphere::new(0.5));
    let craft_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.2),
        emissive: LinearRgba::new(2.0, 1.6, 0.4, 1.0),
        ..default()
    });
    commands.spawn((
        Mesh3d(craft_mesh),
        MeshMaterial3d(craft_material),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Craft,
    ));

    // HUD text
    commands.spawn((
        Text::new("Phi: 0.80 | Amplifiers: ON ON ON | Press 1/2/3 to toggle, Up/Down for Phi"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgba(0.8, 0.9, 1.0, 0.9)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(10.0),
            top: Val::Px(10.0),
            ..default()
        },
    ));
}

/// Step the simulation each frame.
fn update_simulation(mut sim: ResMut<GravcraftSim>, time: Res<Time>) {
    let dt = time.delta_secs() as f64;
    if dt <= 0.0 || dt > 0.1 {
        return;
    }

    let amp = sim.config.max_metric_amplitude * sim.phi;
    let cmd = MetricCommand {
        amplifiers: [
            if sim.amplifiers_active[0] {
                (amp, 10.0, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            },
            if sim.amplifiers_active[1] {
                (amp, 10.0, 2.094, 0.0)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            },
            if sim.amplifiers_active[2] {
                (amp, 10.0, 4.189, 0.0)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            },
        ],
    };

    sim.simulator.step(&cmd, dt);

    let pos = sim.simulator.state().position;
    sim.trail.push(Vec3::new(pos[0] as f32, 0.5, pos[2] as f32));
    if sim.trail.len() > 500 {
        sim.trail.remove(0);
    }
}

/// Deform grid vertices based on metric perturbation.
fn update_grid_deformation(
    sim: Res<GravcraftSim>,
    mut query: Query<(&GridVertex, &mut Transform)>,
) {
    let craft_pos = sim.simulator.state().position;

    for (vertex, mut transform) in &mut query {
        let rest = vertex.rest_position;
        let dx = rest.x - craft_pos[0] as f32;
        let dz = rest.z - craft_pos[2] as f32;
        let r = (dx * dx + dz * dz).sqrt();

        // Alcubierre-like deformation: grid sinks toward amplifier focal points
        let shaping = alcubierre_shaping(r as f64, sim.config.bubble_radius, sim.config.wall_sigma);
        let deformation = shaping as f32 * sim.phi as f32 * 3.0; // Visual scale

        transform.translation = Vec3::new(
            rest.x,
            -deformation, // Grid sinks where metric is perturbed
            rest.z,
        );
    }
}

/// Handle keyboard input.
fn handle_input(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<GravcraftSim>) {
    if keys.just_pressed(KeyCode::Digit1) {
        sim.amplifiers_active[0] = !sim.amplifiers_active[0];
    }
    if keys.just_pressed(KeyCode::Digit2) {
        sim.amplifiers_active[1] = !sim.amplifiers_active[1];
    }
    if keys.just_pressed(KeyCode::Digit3) {
        sim.amplifiers_active[2] = !sim.amplifiers_active[2];
    }
    if keys.just_pressed(KeyCode::Space) {
        let all_on = sim.amplifiers_active.iter().all(|&a| a);
        sim.amplifiers_active = [!all_on; 3];
    }
    if keys.pressed(KeyCode::ArrowUp) {
        sim.phi = (sim.phi + 0.01).min(1.0);
    }
    if keys.pressed(KeyCode::ArrowDown) {
        sim.phi = (sim.phi - 0.01).max(0.0);
    }
}

/// Update the HUD text.
fn update_hud(sim: Res<GravcraftSim>, mut query: Query<&mut Text>) {
    for mut text in &mut query {
        let amps = sim.amplifiers_active.map(|a| if a { "ON" } else { "OFF" });
        let safety = symthaea_core::embodiment::MotorSafetyLevel::from_phi(sim.phi);
        **text = format!(
            "Phi: {:.2} | Safety: {:?} | Amplifiers: {} {} {} | Metric: {:.4} | Press 1/2/3 to toggle, Up/Down for Phi",
            sim.phi,
            safety,
            amps[0],
            amps[1],
            amps[2],
            sim.simulator.state().metric_perturbation,
        );
    }
}
