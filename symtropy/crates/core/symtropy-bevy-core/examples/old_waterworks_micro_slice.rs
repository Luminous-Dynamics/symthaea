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

use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};
use symtropy_basin::{BasinIntervention, OldWaterworksScenario};
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

const MOUSE_SENSITIVITY: Vec2 = Vec2::new(0.0022, 0.0018);
const INTERACTION_DISTANCE: f32 = 2.5;

#[derive(Component)]
struct Player;

#[derive(Component)]
struct PlayerLook {
    sensitivity: Vec2,
}

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
    paused: bool,
    step_once: bool,
}

#[derive(Resource, Debug, Clone)]
struct InputBindings {
    move_forward: KeyCode,
    move_back: KeyCode,
    move_left: KeyCode,
    move_right: KeyCode,
    sprint: KeyCode,
    crouch: KeyCode,
    jump: KeyCode,
    interact: KeyCode,
    focus_inspect: KeyCode,
    toggle_field_deck: KeyCode,
    quick_tool: KeyCode,
    repair_tool: KeyCode,
    build_mode: KeyCode,
    chronicle_panel: KeyCode,
    basin_map: KeyCode,
    scan_visualization: KeyCode,
    previous_scan_mode: KeyCode,
    next_scan_mode: KeyCode,
    pause_or_release: KeyCode,
    controls_overlay: KeyCode,
    command_palette: KeyCode,
    dev_scenario_panel: KeyCode,
}

impl Default for InputBindings {
    fn default() -> Self {
        Self {
            move_forward: KeyCode::KeyW,
            move_back: KeyCode::KeyS,
            move_left: KeyCode::KeyA,
            move_right: KeyCode::KeyD,
            sprint: KeyCode::ShiftLeft,
            crouch: KeyCode::ControlLeft,
            jump: KeyCode::Space,
            interact: KeyCode::KeyE,
            focus_inspect: KeyCode::KeyF,
            toggle_field_deck: KeyCode::Tab,
            quick_tool: KeyCode::KeyQ,
            repair_tool: KeyCode::KeyR,
            build_mode: KeyCode::KeyB,
            chronicle_panel: KeyCode::KeyC,
            basin_map: KeyCode::KeyM,
            scan_visualization: KeyCode::KeyV,
            previous_scan_mode: KeyCode::BracketLeft,
            next_scan_mode: KeyCode::BracketRight,
            pause_or_release: KeyCode::Escape,
            controls_overlay: KeyCode::F1,
            command_palette: KeyCode::KeyK,
            dev_scenario_panel: KeyCode::F10,
        }
    }
}

#[derive(Resource, Default, Debug, Clone)]
struct IntentFrame {
    movement: Vec2,
    look_delta: Vec2,
    pressed: Vec<InputIntent>,
    just_pressed: Vec<InputIntent>,
}

impl IntentFrame {
    fn clear(&mut self) {
        self.movement = Vec2::ZERO;
        self.look_delta = Vec2::ZERO;
        self.pressed.clear();
        self.just_pressed.clear();
    }

    fn pressed(&self, intent: InputIntent) -> bool {
        self.pressed.contains(&intent)
    }

    fn just_pressed(&self, intent: InputIntent) -> bool {
        self.just_pressed.contains(&intent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputIntent {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Sprint,
    Crouch,
    Jump,
    Interact,
    FocusInspect,
    OpenFieldDeck,
    CycleFieldDeckModePrev,
    CycleFieldDeckModeNext,
    EquipToolSlot(u8),
    QuickTool,
    RepairTool,
    BuildMode,
    OpenChroniclePanel,
    OpenBasinMap,
    ToggleScanVisualization,
    PauseOrRelease,
    OpenControlsOverlay,
    OpenCommandPalette,
    OpenDevScenarioPanel,
    TriggerScenarioIntervention(BasinIntervention),
    PauseSimulation,
    StepSimulation,
    ResetScenario,
    CaptureReplay,
}

#[derive(Resource, Debug, Clone)]
struct ControlsState {
    mode: ControlMode,
    selected_tool_slot: u8,
    scan_mode: ScanMode,
    show_controls: bool,
    show_dev_panel: bool,
    mouse_captured: bool,
}

impl Default for ControlsState {
    fn default() -> Self {
        Self {
            mode: ControlMode::FirstPerson,
            selected_tool_slot: 1,
            scan_mode: ScanMode::Ecology,
            show_controls: false,
            show_dev_panel: false,
            mouse_captured: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    FirstPerson,
    FieldDeck,
    Console,
    DevScenario,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    Infrastructure,
    Ecology,
    MachineDiagnostics,
    CivicClaims,
    NullSignalCorruption,
    RepairPreview,
    ChronicleEvidence,
}

impl ScanMode {
    const ALL: [ScanMode; 7] = [
        ScanMode::Infrastructure,
        ScanMode::Ecology,
        ScanMode::MachineDiagnostics,
        ScanMode::CivicClaims,
        ScanMode::NullSignalCorruption,
        ScanMode::RepairPreview,
        ScanMode::ChronicleEvidence,
    ];

    fn label(self) -> &'static str {
        match self {
            ScanMode::Infrastructure => "Infrastructure",
            ScanMode::Ecology => "Ecology",
            ScanMode::MachineDiagnostics => "Machine Diagnostics",
            ScanMode::CivicClaims => "Civic Claims",
            ScanMode::NullSignalCorruption => "Null / Signal Corruption",
            ScanMode::RepairPreview => "Repair Preview",
            ScanMode::ChronicleEvidence => "Chronicle Evidence",
        }
    }

    fn offset(self, delta: isize) -> Self {
        let current = Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0) as isize;
        let len = Self::ALL.len() as isize;
        Self::ALL[((current + delta).rem_euclid(len)) as usize]
    }
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
            paused: false,
            step_once: false,
        })
        .insert_resource(InputBindings::default())
        .insert_resource(IntentFrame::default())
        .insert_resource(ControlsState::default())
        .init_state::<InterfaceState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                input_intent_system,
                control_mode_system,
                cursor_capture_system,
                scenario_step_system,
                player_move_system,
                mouse_look_system,
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

fn input_intent_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    bindings: Res<InputBindings>,
    mut intents: ResMut<IntentFrame>,
) {
    intents.clear();
    intents.look_delta = mouse_motion.delta;

    if keyboard.pressed(bindings.move_forward) {
        intents.movement.y += 1.0;
        intents.pressed.push(InputIntent::MoveForward);
    }
    if keyboard.pressed(bindings.move_back) {
        intents.movement.y -= 1.0;
        intents.pressed.push(InputIntent::MoveBack);
    }
    if keyboard.pressed(bindings.move_left) {
        intents.movement.x -= 1.0;
        intents.pressed.push(InputIntent::MoveLeft);
    }
    if keyboard.pressed(bindings.move_right) {
        intents.movement.x += 1.0;
        intents.pressed.push(InputIntent::MoveRight);
    }
    if keyboard.pressed(bindings.sprint) || keyboard.pressed(KeyCode::ShiftRight) {
        intents.pressed.push(InputIntent::Sprint);
    }
    if keyboard.pressed(bindings.crouch) || keyboard.pressed(KeyCode::ControlRight) {
        intents.pressed.push(InputIntent::Crouch);
    }

    macro_rules! add_just {
        ($key:expr, $intent:expr) => {
            if keyboard.just_pressed($key) {
                intents.just_pressed.push($intent);
            }
        };
    }
    macro_rules! add_just_ctrl {
        ($key:expr, $intent:expr) => {
            if (keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight))
                && keyboard.just_pressed($key)
            {
                intents.just_pressed.push($intent);
            }
        };
    }

    add_just!(bindings.jump, InputIntent::Jump);
    add_just!(bindings.interact, InputIntent::Interact);
    add_just!(bindings.focus_inspect, InputIntent::FocusInspect);
    add_just!(bindings.toggle_field_deck, InputIntent::OpenFieldDeck);
    add_just!(bindings.quick_tool, InputIntent::QuickTool);
    add_just!(bindings.repair_tool, InputIntent::RepairTool);
    add_just!(bindings.build_mode, InputIntent::BuildMode);
    add_just!(bindings.chronicle_panel, InputIntent::OpenChroniclePanel);
    add_just!(bindings.basin_map, InputIntent::OpenBasinMap);
    add_just!(
        bindings.scan_visualization,
        InputIntent::ToggleScanVisualization
    );
    add_just!(
        bindings.previous_scan_mode,
        InputIntent::CycleFieldDeckModePrev
    );
    add_just!(bindings.next_scan_mode, InputIntent::CycleFieldDeckModeNext);
    add_just!(bindings.pause_or_release, InputIntent::PauseOrRelease);
    add_just!(bindings.controls_overlay, InputIntent::OpenControlsOverlay);
    add_just_ctrl!(bindings.command_palette, InputIntent::OpenCommandPalette);
    add_just!(
        bindings.dev_scenario_panel,
        InputIntent::OpenDevScenarioPanel
    );

    for (key, slot) in [
        (KeyCode::Digit1, 1),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
        (KeyCode::Digit6, 6),
    ] {
        if keyboard.just_pressed(key) {
            intents.just_pressed.push(InputIntent::EquipToolSlot(slot));
        }
    }

    for (key, intervention) in [
        (KeyCode::F2, BasinIntervention::PipeLeak),
        (KeyCode::F3, BasinIntervention::FastMechanicalRepair),
        (KeyCode::F4, BasinIntervention::EcologicalReroute),
        (KeyCode::F5, BasinIntervention::WillowPlanting),
        (KeyCode::F6, BasinIntervention::NullGreenwash),
        (KeyCode::F7, BasinIntervention::DecomposerAid),
    ] {
        if keyboard.just_pressed(key) {
            intents
                .just_pressed
                .push(InputIntent::TriggerScenarioIntervention(intervention));
        }
    }
    add_just!(KeyCode::KeyP, InputIntent::PauseSimulation);
    add_just!(KeyCode::Period, InputIntent::StepSimulation);
    add_just!(KeyCode::F8, InputIntent::ResetScenario);
    add_just!(KeyCode::F9, InputIntent::CaptureReplay);

    if mouse.just_pressed(MouseButton::Left) {
        intents.just_pressed.push(InputIntent::FocusInspect);
    }
}

fn control_mode_system(
    intents: Res<IntentFrame>,
    mut controls: ResMut<ControlsState>,
    mut runtime: ResMut<ScenarioRuntime>,
    mut next_state: ResMut<NextState<InterfaceState>>,
) {
    if intents.just_pressed(InputIntent::PauseOrRelease) {
        controls.mode = ControlMode::Pause;
        controls.mouse_captured = false;
        controls.show_dev_panel = false;
        next_state.set(InterfaceState::None);
    }

    if intents.just_pressed(InputIntent::OpenControlsOverlay) {
        controls.show_controls = !controls.show_controls;
    }

    if intents.just_pressed(InputIntent::OpenDevScenarioPanel) {
        controls.show_dev_panel = !controls.show_dev_panel;
        controls.mode = if controls.show_dev_panel {
            controls.mouse_captured = false;
            ControlMode::DevScenario
        } else {
            controls.mouse_captured = true;
            ControlMode::FirstPerson
        };
        next_state.set(InterfaceState::None);
    }

    if intents.just_pressed(InputIntent::OpenFieldDeck) {
        controls.show_dev_panel = false;
        match controls.mode {
            ControlMode::FieldDeck => {
                controls.mode = ControlMode::FirstPerson;
                controls.mouse_captured = true;
                next_state.set(InterfaceState::None);
            }
            _ => {
                controls.mode = ControlMode::FieldDeck;
                controls.mouse_captured = false;
                next_state.set(InterfaceState::FieldDeck);
            }
        }
    }

    if intents.just_pressed(InputIntent::CycleFieldDeckModePrev) {
        controls.scan_mode = controls.scan_mode.offset(-1);
    }
    if intents.just_pressed(InputIntent::CycleFieldDeckModeNext) {
        controls.scan_mode = controls.scan_mode.offset(1);
    }

    for intent in &intents.just_pressed {
        match *intent {
            InputIntent::EquipToolSlot(slot) => controls.selected_tool_slot = slot,
            InputIntent::TriggerScenarioIntervention(intervention) if controls.show_dev_panel => {
                runtime.scenario.apply(intervention);
            }
            InputIntent::PauseSimulation if controls.show_dev_panel => {
                runtime.paused = !runtime.paused;
            }
            InputIntent::StepSimulation if controls.show_dev_panel => {
                runtime.step_once = true;
            }
            InputIntent::ResetScenario if controls.show_dev_panel => {
                runtime.scenario = OldWaterworksScenario::new(16, 9);
                runtime.scenario.apply(BasinIntervention::PipeLeak);
                runtime.scenario.apply(BasinIntervention::NullGreenwash);
                runtime.paused = false;
                runtime.step_once = false;
            }
            InputIntent::FocusInspect if controls.mode == ControlMode::Pause => {
                controls.mode = ControlMode::FirstPerson;
                controls.mouse_captured = true;
            }
            _ => {}
        }
    }
}

fn cursor_capture_system(
    controls: Res<ControlsState>,
    mut cursor_options: Single<&mut CursorOptions>,
) {
    if controls.is_changed() {
        cursor_options.visible = !controls.mouse_captured;
        cursor_options.grab_mode = if controls.mouse_captured {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
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
        PlayerLook {
            sensitivity: MOUSE_SENSITIVITY,
        },
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

fn player_move_system(
    intents: Res<IntentFrame>,
    time: Res<Time>,
    controls: Res<ControlsState>,
    mut query: Query<&mut Transform, With<Player>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    let base_speed = match controls.mode {
        ControlMode::FirstPerson => 5.0,
        ControlMode::FieldDeck => 1.0,
        ControlMode::Console | ControlMode::DevScenario | ControlMode::Pause => 0.0,
    };

    if base_speed == 0.0 {
        return;
    }

    let mut speed = base_speed;
    if intents.pressed(InputIntent::Sprint) && controls.mode == ControlMode::FirstPerson {
        speed *= 1.65;
    }
    if intents.pressed(InputIntent::Crouch) {
        speed *= 0.45;
    }

    let mut direction =
        *transform.forward() * intents.movement.y + *transform.right() * intents.movement.x;
    direction.y = 0.0;
    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
        transform.translation += direction * speed * time.delta_secs();
    }
}

fn mouse_look_system(
    intents: Res<IntentFrame>,
    controls: Res<ControlsState>,
    mut query: Query<(&mut Transform, &PlayerLook), With<Player>>,
) {
    if !controls.mouse_captured || controls.mode != ControlMode::FirstPerson {
        return;
    }

    let Ok((mut transform, look)) = query.single_mut() else {
        return;
    };
    let delta = intents.look_delta;
    if delta == Vec2::ZERO {
        return;
    }

    let delta_yaw = -delta.x * look.sensitivity.x;
    let delta_pitch = -delta.y * look.sensitivity.y;
    let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
    let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
    transform.rotation = Quat::from_euler(
        EulerRot::YXZ,
        yaw + delta_yaw,
        (pitch + delta_pitch).clamp(-pitch_limit, pitch_limit),
        roll,
    );
}

fn interaction_system(
    intents: Res<IntentFrame>,
    runtime: Res<ScenarioRuntime>,
    player_query: Query<&Transform, With<Player>>,
    console_query: Query<&Transform, With<Console>>,
    mut interaction_hint_query: Query<(&mut Text, &mut Visibility), With<InteractionHint>>,
    mut inspection_text_query: Query<&mut Text, (With<InspectionText>, Without<InteractionHint>)>,
    mut next_state: ResMut<NextState<InterfaceState>>,
    mut controls_state: ResMut<ControlsState>,
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
    let near_console = dist < INTERACTION_DISTANCE;
    let current_mode = controls_state.mode;

    // Interaction Hint
    if current_mode != ControlMode::Console && near_console {
        **hint_text = "E: inspect pump console | Tab: Field Deck | F1: controls".to_string();
        *hint_visibility = Visibility::Visible;
    } else {
        *hint_visibility = Visibility::Hidden;
    }

    if intents.just_pressed(InputIntent::Interact) && near_console {
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
        **inspect_text = format!(
            "FIELD DECK ECOLOGY LINK\n\
             Mode: {} ([ / ] to cycle)\n\
             Tool slot: {}\n\
             Basin viability: {:.2}\n\
             Toxin load: {:.2}\n\
             Signal corruption: {:.2}\n\
             Colony stress: {:.2}\n\
             Mycelium toxin buffered: {:.3}\n\
             Machine status: GREEN / BIOLOGICAL STATUS: CONFLICT\n\
             C: Chronicle evidence | V: scan overlay | F: focus reading",
            controls_state.scan_mode.label(),
            controls_state.selected_tool_slot,
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
