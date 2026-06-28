// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OLD WATERWORKS MICRO-SLICE — Ticket 1, 2, 5
//!
//! This is the first concrete playable foundation for Symtropy.
//! - Floor, walls, and greybox machinery (pump, tank, console).
//! - Intent-based first-person movement (WASD + mouse look).
//! - Proximity-based interaction with the console (Press E).
//! - Field Deck: Amber interface frame (Press Tab to toggle).
//! - Dead authority lock inspection state (inside Field Deck).
//! - Panic Drop: Esc releases the mouse and exits tools instantly.
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
use symtropy_bevy_core::{
    ControlMode, ControlsState, FirstPersonController, FirstPersonInputPlugin, InputIntent,
    IntentFrame,
};
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

const INTERACTION_DISTANCE: f32 = 2.5;

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Console;

#[derive(Component, Debug, Clone, Copy)]
struct InteractableTarget {
    kind: InteractableKind,
    label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractableKind {
    PumpConsole,
    WillowRoots,
    AntTrail,
    MyceliumPatch,
}

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
    paused: bool,
    step_once: bool,
    last_outcome: Option<symtropy_basin::OldWaterworksChoiceOutcome>,
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
        .add_plugins(FirstPersonInputPlugin)
        .insert_resource(ScenarioRuntime {
            scenario,
            timer: Timer::from_seconds(0.20, TimerMode::Repeating),
            paused: false,
            step_once: false,
            last_outcome: None,
        })
        .init_state::<InterfaceState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                scenario_dev_controls_system,
                scenario_step_system,
                interaction_system,
                ecological_meter_system,
                ui_visibility_system,
                controls_overlay_system,
                dev_panel_overlay_system,
                oscillating_light_system,
            )
                .chain(),
        )
        .run();
}

fn scenario_step_system(time: Res<Time>, mut runtime: ResMut<ScenarioRuntime>) {
    let auto_step = !runtime.paused && runtime.timer.tick(time.delta()).just_finished();
    if auto_step || runtime.step_once {
        runtime.scenario.step();
        runtime.step_once = false;
    }
}

fn scenario_dev_controls_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    intents: Res<IntentFrame>,
    controls: Res<ControlsState>,
    mut runtime: ResMut<ScenarioRuntime>,
    mut next_state: ResMut<NextState<InterfaceState>>,
) {
    if intents.just_pressed(InputIntent::PauseOrRelease) {
        next_state.set(InterfaceState::None);
    }

    if intents.just_pressed(InputIntent::OpenDevScenarioPanel) {
        next_state.set(InterfaceState::None);
    }

    if intents.just_pressed(InputIntent::OpenFieldDeck) {
        match controls.mode {
            ControlMode::FieldDeck => next_state.set(InterfaceState::FieldDeck),
            _ => next_state.set(InterfaceState::None),
        }
    }

    if !controls.show_dev_panel {
        return;
    }

    for (key, intervention) in [
        (KeyCode::F2, BasinIntervention::PipeLeak),
        (KeyCode::F3, BasinIntervention::FastMechanicalRepair),
        (KeyCode::F4, BasinIntervention::EcologicalReroute),
        (KeyCode::F5, BasinIntervention::WillowPlanting),
        (KeyCode::F6, BasinIntervention::NullGreenwash),
        (KeyCode::F7, BasinIntervention::DecomposerAid),
        (KeyCode::F11, BasinIntervention::CutWillowRoots),
        (KeyCode::F12, BasinIntervention::DelayRepair),
    ] {
        if keyboard.just_pressed(key) {
            runtime.last_outcome = Some(runtime.scenario.apply_choice_and_step(intervention, 3));
        }
    }

    if intents.just_pressed(InputIntent::PauseSimulation) {
        runtime.paused = !runtime.paused;
    }
    if intents.just_pressed(InputIntent::StepSimulation) {
        runtime.step_once = true;
    }
    if intents.just_pressed(InputIntent::ResetScenario) {
        runtime.scenario = OldWaterworksScenario::new(16, 9);
        runtime.scenario.apply(BasinIntervention::PipeLeak);
        runtime.scenario.apply(BasinIntervention::NullGreenwash);
        runtime.paused = false;
        runtime.step_once = false;
        runtime.last_outcome = None;
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
        InteractableTarget {
            kind: InteractableKind::PumpConsole,
            label: "Pump console",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(pump_mat),
        Transform::from_xyz(-2.0, 1.0, -2.0).with_scale(Vec3::new(2.0, 2.0, 2.0)),
    ));

    // Tank / Pipe
    commands.spawn((
        InteractableTarget {
            kind: InteractableKind::MyceliumPatch,
            label: "Toxin-buffering mycelium",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(tank_mat),
        Transform::from_xyz(-5.0, 1.5, 5.0).with_scale(Vec3::new(1.5, 3.0, 1.5)),
    ));

    // Console
    commands.spawn((
        Console,
        InteractableTarget {
            kind: InteractableKind::PumpConsole,
            label: "Dead-authority pump console",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(console_mat),
        Transform::from_xyz(8.0, 1.2, 0.0).with_scale(Vec3::new(0.5, 1.0, 1.5)),
    ));

    let willow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.36, 0.12),
        emissive: Color::srgb(0.12, 0.20, 0.06).to_linear() * 0.08,
        ..default()
    });
    commands.spawn((
        InteractableTarget {
            kind: InteractableKind::WillowRoots,
            label: "Willow root filtration",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(willow_mat),
        Transform::from_xyz(1.5, 0.25, -2.5).with_scale(Vec3::new(3.0, 0.2, 0.35)),
    ));

    let ant_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.65, 0.18),
        emissive: Color::srgb(0.9, 0.45, 0.10).to_linear() * 0.12,
        ..default()
    });
    commands.spawn((
        InteractableTarget {
            kind: InteractableKind::AntTrail,
            label: "Rerouting ant trail",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(ant_mat),
        Transform::from_xyz(2.2, 0.08, 1.8).with_scale(Vec3::new(4.0, 0.08, 0.18)),
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
        FirstPersonController::default(),
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

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(20.0),
            top: Val::Px(20.0),
            width: Val::Px(520.0),
            padding: UiRect::all(Val::Px(16.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.88)),
        ControlsOverlay,
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.86, 0.9, 1.0)),
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(20.0),
            top: Val::Px(20.0),
            width: Val::Px(520.0),
            padding: UiRect::all(Val::Px(16.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.04, 0.025, 0.015, 0.9)),
        DevPanelOverlay,
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.76, 0.46)),
    ));
}

#[derive(Component)]
struct InteractionHint;

#[derive(Component)]
struct FieldDeckUI;

#[derive(Component)]
struct InspectionText;

#[derive(Component)]
struct ControlsOverlay;

#[derive(Component)]
struct DevPanelOverlay;

fn interaction_system(
    intents: Res<IntentFrame>,
    runtime: Res<ScenarioRuntime>,
    player_query: Query<&Transform, With<Player>>,
    target_query: Query<(&Transform, &InteractableTarget)>,
    mut interaction_hint_query: Query<(&mut Text, &mut Visibility), With<InteractionHint>>,
    mut inspection_text_query: Query<&mut Text, (With<InspectionText>, Without<InteractionHint>)>,
    mut next_state: ResMut<NextState<InterfaceState>>,
    mut controls_state: ResMut<ControlsState>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok((mut hint_text, mut hint_visibility)) = interaction_hint_query.single_mut() else {
        return;
    };
    let Ok(mut inspect_text) = inspection_text_query.single_mut() else {
        return;
    };

    let nearest = nearest_target(player_tf.translation, &target_query);
    let near_target = nearest
        .as_ref()
        .map(|(_, distance)| *distance < INTERACTION_DISTANCE)
        .unwrap_or(false);
    let target = nearest.map(|(target, _)| target);
    let current_mode = controls_state.mode;

    // Interaction Hint
    if current_mode != ControlMode::Console && near_target {
        let target = target.expect("near_target implies nearest target");
        **hint_text = format!(
            "E: inspect {} | Tab: Field Deck | F1: controls",
            target.label
        );
        *hint_visibility = Visibility::Visible;
    } else {
        *hint_visibility = Visibility::Hidden;
    }

    if intents.just_pressed(InputIntent::Interact)
        && near_target
        && target
            .map(|target| target.kind == InteractableKind::PumpConsole)
            .unwrap_or(false)
    {
        controls_state.mode = ControlMode::Console;
        controls_state.mouse_captured = false;
        next_state.set(InterfaceState::Console);
    }

    // Update Inspection Text
    if current_mode == ControlMode::Console {
        **inspect_text = format!(
            "OLD WATERWORKS CONSOLE\n\
             PUMP_1: LOCKED\n\
             TANK_0: 12%\n\
             AUTHORITY: DEAD_AUTHORITY_LOCK\n\
             Active tool slot: {}\n\
             Esc: release | Tab: Field Deck",
            controls_state.selected_tool_slot
        );
    } else if current_mode == ControlMode::FieldDeck {
        let Some(record) = runtime.scenario.records().last() else {
            **inspect_text = "FIELD DECK ECOLOGY LINK\nAwaiting basin signal...".to_string();
            return;
        };
        let testimony = if record.testimony.is_empty() {
            "Life testimony: no stable interpretation yet.".to_string()
        } else {
            record
                .testimony
                .iter()
                .take(3)
                .map(|entry| format!("{:?}: {}", entry.channel, entry.summary))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let events = if record.events.is_empty() {
            "Events: none".to_string()
        } else {
            format!(
                "Events: {}",
                record
                    .events
                    .iter()
                    .map(|event| format!("{event:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let current_target = target
            .filter(|_| near_target)
            .map(|target| format!("Focused target: {} ({:?})", target.label, target.kind))
            .unwrap_or_else(|| "Focused target: none in range".to_string());
        let outcome = runtime
            .last_outcome
            .as_ref()
            .map(format_outcome)
            .unwrap_or_else(|| "Last choice outcome: none yet".to_string());
        **inspect_text = format!(
            "FIELD DECK ECOLOGY LINK\n\
             Mode: {} ([ / ] to cycle)\n\
             Tool slot: {}\n\
             {}\n\
             Basin viability: {:.2}\n\
             Toxin load: {:.2}\n\
             Signal corruption: {:.2}\n\
             Colony stress: {:.2}\n\
             Mycelium toxin buffered: {:.3}\n\
             Machine status: GREEN / BIOLOGICAL STATUS: CONFLICT\n\
             {}\n\
             {}\n\
             {}\n\
             C: Chronicle evidence | V: scan overlay | F: focus reading",
            controls_state.scan_mode.label(),
            controls_state.selected_tool_slot,
            current_target,
            record.basin.viability,
            record.basin.toxin_load,
            record.basin.signal_corruption,
            record.colony.stress,
            record.mycelium_exchange.toxin_buffered,
            events,
            testimony,
            outcome,
        );
    } else {
        **inspect_text = "".to_string();
    }
}

fn nearest_target<'a>(
    player_position: Vec3,
    query: &'a Query<(&Transform, &InteractableTarget)>,
) -> Option<(&'a InteractableTarget, f32)> {
    query
        .iter()
        .map(|(transform, target)| (target, player_position.distance(transform.translation)))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}

fn format_outcome(outcome: &symtropy_basin::OldWaterworksChoiceOutcome) -> String {
    let chronicle = outcome
        .chronicle
        .first()
        .map(|event| event.summary.as_str())
        .unwrap_or("Chronicle: no durable event yet.");
    let faction = outcome
        .faction_reactions
        .first()
        .map(|reaction| {
            format!(
                "{:?} {:?}: {}",
                reaction.faction, reaction.stance, reaction.summary
            )
        })
        .unwrap_or_else(|| "No faction reaction yet.".to_string());
    format!(
        "Last choice: {:?} at tick {}\n{}\n{}",
        outcome.intervention, outcome.tick, chronicle, faction
    )
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

fn controls_overlay_system(
    controls: Res<ControlsState>,
    runtime: Res<ScenarioRuntime>,
    mut query: Query<(&mut Node, &mut Text), With<ControlsOverlay>>,
) {
    let Ok((mut node, mut text)) = query.single_mut() else {
        return;
    };
    node.display = if controls.show_controls {
        Display::Flex
    } else {
        Display::None
    };
    if !controls.show_controls {
        return;
    }

    **text = format!(
        "FIRST-PERSON CONTROL SPINE\n\
         WASD move | Mouse look | Shift sprint | Ctrl crouch | Space jump later\n\
         E interact/use focused object | F focus inspect\n\
         Tab Field Deck | [ / ] scan mode | V scan visualization\n\
         1-6 toolbelt slots | Q quick tool | R repair tool | B build mode\n\
         C Chronicle/evidence | M basin map | Ctrl+K command palette\n\
         F10 dev scenario panel | Esc release mouse / pause\n\n\
         Mode: {:?}\n\
         Mouse captured: {}\n\
         Tool slot: {}\n\
         Field Deck mode: {}\n\
         Scenario paused: {}\n\
         Tick records: {}",
        controls.mode,
        controls.mouse_captured,
        controls.selected_tool_slot,
        controls.scan_mode.label(),
        runtime.paused,
        runtime.scenario.records().len()
    );
}

fn dev_panel_overlay_system(
    controls: Res<ControlsState>,
    runtime: Res<ScenarioRuntime>,
    mut query: Query<(&mut Node, &mut Text), With<DevPanelOverlay>>,
) {
    let Ok((mut node, mut text)) = query.single_mut() else {
        return;
    };
    node.display = if controls.show_dev_panel {
        Display::Flex
    } else {
        Display::None
    };
    if !controls.show_dev_panel {
        return;
    }

    let latest = runtime.scenario.records().last();
    let metrics = latest
        .map(|record| {
            format!(
                "Latest basin: viability {:.2}, toxin {:.2}, corruption {:.2}\n\
                 Colony stress {:.2} | Mycelium biomass {:.2}",
                record.basin.viability,
                record.basin.toxin_load,
                record.basin.signal_corruption,
                record.colony.stress,
                record.mycelium.total_biomass
            )
        })
        .unwrap_or_else(|| "Latest basin: awaiting first tick".to_string());

    **text = format!(
        "DEV SCENARIO PANEL (F10)\n\
         F2 pipe leak\n\
         F3 fast mechanical repair\n\
         F4 ecological reroute\n\
         F5 willow planting\n\
         F6 Null greenwash\n\
         F7 decomposer aid\n\
         F8 reset scenario\n\
         F9 capture replay marker (intent only)\n\
         F11 cut willow roots\n\
         F12 delay repair for evidence review\n\
         P pause/resume simulation\n\
         . step one tick\n\n\
         Simulation paused: {}\n\
         Tick records: {}\n\
         {}",
        runtime.paused,
        runtime.scenario.records().len(),
        metrics
    );
}
