// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OLD WATERWORKS MICRO-SLICE — Ticket 1, 2, 5
//!
//! This is the first concrete playable foundation for Symtropy.
//! - Floor, walls, and greybox machinery (pump, tank, console).
//! - Basic first-person movement (WASD + Arrows).
//! - Proximity-based interaction with the console (Press E).
//! - Field Deck: Amber interface frame (Press F to toggle).
//! - Dead authority lock inspection state (inside Field Deck).
//! - Panic Drop: Esc/Shift exits instantly.
//! - Procedural Lighting: Oscillating room intensity.
//!
//! Hard Scope:
//! - No Device Bus yet
//! - No WASM yet
//! - No Lightyear yet
//! - No Mycelix yet
//! - No Holochain yet
//! - No SymLogic yet
//! - No full Field Deck UI yet
//! - No faction evolution yet
//! - No Chronicle yet
//! - just the first playable room

use bevy::prelude::*;
use symtropy_basin::{BasinIntervention, OldWaterworksScenario};
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Console;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
enum InterfaceState {
    #[default]
    None,
    FieldDeck,
    Console,
}

#[derive(Component)]
struct OscillatingLight;

#[derive(Resource)]
struct ScenarioRuntime {
    scenario: OldWaterworksScenario,
    timer: Timer,
}

#[derive(Component)]
struct EcologyMeter {
    kind: EcologyMeterKind,
}

#[derive(Debug, Clone, Copy)]
enum EcologyMeterKind {
    BasinToxin,
    ColonyStress,
    MyceliumBiomass,
}

fn main() {
    let mut scenario = OldWaterworksScenario::new(16, 9);
    scenario.apply(BasinIntervention::PipeLeak);
    scenario.apply(BasinIntervention::NullGreenwash);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Old Waterworks Micro-Slice".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymtropyScenePlugin::default())
        .insert_resource(ScenarioRuntime {
            scenario,
            timer: Timer::from_seconds(0.20, TimerMode::Repeating),
        })
        .init_state::<InterfaceState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                scenario_step_system,
                player_move_system,
                interaction_system,
                ecological_meter_system,
                ui_visibility_system,
                oscillating_light_system,
            ),
        )
        .run();
}

fn scenario_step_system(time: Res<Time>, mut runtime: ResMut<ScenarioRuntime>) {
    if runtime.timer.tick(time.delta()).just_finished() {
        runtime.scenario.step();
    }
}

fn oscillating_light_system(
    time: Res<Time>,
    mut query: Query<&mut PointLight, With<OscillatingLight>>,
) {
    let elapsed = time.elapsed_secs();
    for mut light in query.iter_mut() {
        // Oscillation between 80k and 120k intensity
        light.intensity = 100_000.0 + (elapsed * 2.0).sin() * 20_000.0;
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Materials
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 0.22),
        ..default()
    });
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.35),
        ..default()
    });
    let tank_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.4),
        ..default()
    });
    let console_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.1),
        ..default()
    });

    // Meshes
    let cube_mesh = meshes.add(Cuboid::default());

    // Floor
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(0.0, -0.1, 0.0).with_scale(Vec3::new(20.0, 0.2, 20.0)),
    ));

    // Walls
    let wall_height = 4.0;
    // North
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, -10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // South
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, 10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // East
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));
    // West
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(-10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));

    // Pump
    let pump_texture = asset_server.load("symtropy/old_waterworks/materials/pump_rust.jpg");
    let pump_mat = materials.add(StandardMaterial {
        base_color_texture: Some(pump_texture),
        ..default()
    });
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(pump_mat),
        Transform::from_xyz(-2.0, 1.0, -2.0).with_scale(Vec3::new(2.0, 2.0, 2.0)),
    ));

    // Tank / Pipe
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(tank_mat),
        Transform::from_xyz(-5.0, 1.5, 5.0).with_scale(Vec3::new(1.5, 3.0, 1.5)),
    ));

    // Console
    commands.spawn((
        Console,
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(console_mat),
        Transform::from_xyz(8.0, 1.2, 0.0).with_scale(Vec3::new(0.5, 1.0, 1.5)),
    ));

    // Living Basin OS meters. These are deliberately simple greybox columns:
    // red = toxin pressure, amber = colony stress, green = mycelial biomass.
    let meter_specs = [
        (
            EcologyMeterKind::BasinToxin,
            Vec3::new(5.7, 0.35, -2.0),
            Color::srgb(0.8, 0.1, 0.1),
        ),
        (
            EcologyMeterKind::ColonyStress,
            Vec3::new(6.5, 0.35, -2.0),
            Color::srgb(1.0, 0.65, 0.1),
        ),
        (
            EcologyMeterKind::MyceliumBiomass,
            Vec3::new(7.3, 0.35, -2.0),
            Color::srgb(0.1, 0.7, 0.25),
        ),
    ];
    for (kind, position, color) in meter_specs {
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive: color.to_linear() * 0.08,
            ..default()
        });
        commands.spawn((
            EcologyMeter { kind },
            Mesh3d(cube_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position).with_scale(Vec3::new(0.35, 0.7, 0.35)),
        ));
    }

    // Player/Camera
    commands.spawn((
        Player,
        fixed_camera(Vec3::new(0.0, 1.7, 5.0), Vec3::new(0.0, 1.7, 0.0)),
    ));

    // Light source (additional to the sun)
    commands.spawn((
        PointLight {
            intensity: 100_000.0,
            range: 20.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 3.5, 0.0),
        OscillatingLight,
    ));

    // UI Root for interaction hint
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: 30.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                InteractionHint,
            ));
        });

    // UI Root for Field Deck (Amber Frame)
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                border: UiRect::all(Val::Px(20.0)),
                display: Display::None,
                ..default()
            },
            BorderColor::all(Color::srgb(1.0, 0.8, 0.2)), // Amber
            FieldDeckUI,
        ))
        .with_children(|parent| {
            // Label for the Field Deck itself
            parent
                .spawn((Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(25.0),
                    left: Val::Px(25.0),
                    ..default()
                },))
                .with_children(|inner| {
                    inner.spawn((
                        Text::new("FIELD DECK MK0 - OFFLINE LINK"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.8, 0.2)),
                    ));
                });

            // Container for inner content (Console info)
            parent
                .spawn((Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },))
                .with_children(|inner| {
                    inner.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: 40.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.8, 0.2)),
                        InspectionText,
                    ));
                });

            // Panic Drop hint
            parent
                .spawn((Node {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(25.0),
                    right: Val::Px(25.0),
                    ..default()
                },))
                .with_children(|inner| {
                    inner.spawn((
                        Text::new("Press Esc/Shift to Panic Drop"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.5, 0.0)), // More orange/warning
                    ));
                });
        });
}

#[derive(Component)]
struct InteractionHint;

#[derive(Component)]
struct FieldDeckUI;

#[derive(Component)]
struct InspectionText;

fn player_move_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    state: Res<State<InterfaceState>>,
    mut query: Query<&mut Transform, With<Player>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    // Movement speed depends on state
    let speed = match *state.get() {
        InterfaceState::None => 5.0,
        InterfaceState::FieldDeck => 1.0, // Slowed movement while Deck is up
        InterfaceState::Console => 0.0,   // No movement while inspecting console
    };

    if speed == 0.0 {
        return;
    }

    let mut direction = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        let forward = transform.forward();
        direction += *forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        let back = transform.back();
        direction += *back;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        let left = transform.left();
        direction += *left;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        let right = transform.right();
        direction += *right;
    }

    direction.y = 0.0;
    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
        transform.translation += direction * speed * time.delta_secs();
    }

    // Simple rotation with Arrows
    let mut rotation = 0.0;
    if keyboard.pressed(KeyCode::ArrowLeft) {
        rotation += 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        rotation -= 1.0;
    }
    transform.rotate_y(rotation * 2.0 * time.delta_secs());
}

fn interaction_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    runtime: Res<ScenarioRuntime>,
    player_query: Query<&Transform, With<Player>>,
    console_query: Query<&Transform, With<Console>>,
    mut interaction_hint_query: Query<(&mut Text, &mut Visibility), With<InteractionHint>>,
    mut inspection_text_query: Query<&mut Text, (With<InspectionText>, Without<InteractionHint>)>,
    state: Res<State<InterfaceState>>,
    mut next_state: ResMut<NextState<InterfaceState>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok(console_tf) = console_query.single() else {
        return;
    };
    let Ok((mut hint_text, mut hint_visibility)) = interaction_hint_query.single_mut() else {
        return;
    };
    let Ok(mut inspect_text) = inspection_text_query.single_mut() else {
        return;
    };

    let dist = player_tf.translation.distance(console_tf.translation);
    let near_console = dist < 2.5;
    let current_state = *state.get();

    // Interaction Hint
    if current_state != InterfaceState::Console && near_console {
        **hint_text = "Press E to inspect pump console".to_string();
        *hint_visibility = Visibility::Visible;
    } else {
        *hint_visibility = Visibility::Hidden;
    }

    // Input Handling
    if keyboard.just_pressed(KeyCode::Escape)
        || keyboard.just_pressed(KeyCode::ShiftLeft)
        || keyboard.just_pressed(KeyCode::ShiftRight)
    {
        // Panic Drop
        next_state.set(InterfaceState::None);
    } else if keyboard.just_pressed(KeyCode::KeyF) {
        // Toggle Field Deck
        match current_state {
            InterfaceState::None => next_state.set(InterfaceState::FieldDeck),
            InterfaceState::FieldDeck => next_state.set(InterfaceState::None),
            InterfaceState::Console => next_state.set(InterfaceState::None), // Panic drop also works here
        }
    } else if keyboard.just_pressed(KeyCode::KeyE) && near_console {
        // Inspect Console
        next_state.set(InterfaceState::Console);
    }

    // Update Inspection Text
    if current_state == InterfaceState::Console {
        **inspect_text =
            "OLD WATERWORKS CONSOLE\nPUMP_1: LOCKED\nTANK_0: 12%\nAUTHORITY: DEAD_AUTHORITY_LOCK"
                .to_string();
    } else if current_state == InterfaceState::FieldDeck {
        let Some(record) = runtime.scenario.records().last() else {
            **inspect_text = "FIELD DECK ECOLOGY LINK\nAwaiting basin signal...".to_string();
            return;
        };
        **inspect_text = format!(
            "FIELD DECK ECOLOGY LINK\n\
             Basin viability: {:.2}\n\
             Toxin load: {:.2}\n\
             Signal corruption: {:.2}\n\
             Colony stress: {:.2}\n\
             Mycelium toxin buffered: {:.3}\n\
             Machine status: GREEN / BIOLOGICAL STATUS: CONFLICT",
            record.basin.viability,
            record.basin.toxin_load,
            record.basin.signal_corruption,
            record.colony.stress,
            record.mycelium_exchange.toxin_buffered,
        );
    } else {
        **inspect_text = "".to_string();
    }
}

fn ecological_meter_system(
    runtime: Res<ScenarioRuntime>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(
        &EcologyMeter,
        &MeshMaterial3d<StandardMaterial>,
        &mut Transform,
    )>,
) {
    let Some(record) = runtime.scenario.records().last() else {
        return;
    };

    for (meter, material, mut transform) in query.iter_mut() {
        let (value, color) = match meter.kind {
            EcologyMeterKind::BasinToxin => (
                (record.basin.toxin_load / 1.0).clamp(0.0, 1.0),
                Color::srgb(0.9, 0.1, 0.1),
            ),
            EcologyMeterKind::ColonyStress => (
                (record.colony.stress / 2.0).clamp(0.0, 1.0),
                Color::srgb(1.0, 0.65, 0.1),
            ),
            EcologyMeterKind::MyceliumBiomass => (
                (record.mycelium.total_biomass / 80.0).clamp(0.0, 1.0),
                Color::srgb(0.1, 0.8, 0.25),
            ),
        };
        let height = 0.25 + value * 2.2;
        transform.scale.y = height;
        transform.translation.y = height * 0.5;

        if let Some(mat) = materials.get_mut(&material.0) {
            mat.base_color = color.with_alpha(0.35 + value * 0.65);
            mat.emissive = color.to_linear() * (0.02 + value * 0.18);
        }
    }
}

fn ui_visibility_system(
    state: Res<State<InterfaceState>>,
    mut query: Query<&mut Node, With<FieldDeckUI>>,
) {
    let Ok(mut node) = query.single_mut() else {
        return;
    };
    match *state.get() {
        InterfaceState::None => node.display = Display::None,
        InterfaceState::FieldDeck | InterfaceState::Console => node.display = Display::Flex,
    }
}
