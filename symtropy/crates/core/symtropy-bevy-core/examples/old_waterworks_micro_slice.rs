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
//! - Local Chronicle JSONL consequence log (Press C).
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
//! - just the first playable room

use bevy::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};
use symtropy_basin::{BasinIntervention, OldWaterworksScenario};
use symtropy_bevy_core::{
    ControlMode, ControlsState, FirstPersonController, FirstPersonInputPlugin, InputIntent,
    IntentFrame, ScanMode,
};
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

const INTERACTION_DISTANCE: f32 = 2.5;
#[cfg(test)]
const OLD_WATERWORKS_VISIBLE_PROP_COUNT: usize = 9;
#[cfg(test)]
const OLD_WATERWORKS_MATERIAL_PALETTE: &[&str] = &[
    "dry_concrete",
    "wet_concrete",
    "algae_stained_concrete",
    "rusted_metal",
    "old_painted_steel",
    "dirty_water",
    "warning_stripe",
    "chronicle_seal",
    "field_deck_scanline",
];

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

#[derive(Resource, Debug, Clone)]
struct SliceRuntime {
    origin: PlayerOrigin,
    pending_repair: Option<PendingRepair>,
    ending_card_open: bool,
}

impl Default for SliceRuntime {
    fn default() -> Self {
        Self {
            origin: PlayerOrigin::BasinBornTechnician,
            pending_repair: None,
            ending_card_open: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayerOrigin {
    BasinBornTechnician,
    ArchiveApprentice,
    CorporateUtilityDefector,
}

impl PlayerOrigin {
    const ALL: [Self; 3] = [
        Self::BasinBornTechnician,
        Self::ArchiveApprentice,
        Self::CorporateUtilityDefector,
    ];

    fn id(self) -> &'static str {
        match self {
            Self::BasinBornTechnician => "basin_born_technician",
            Self::ArchiveApprentice => "archive_apprentice",
            Self::CorporateUtilityDefector => "corporate_utility_defector",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::BasinBornTechnician => "Basin-Born Technician",
            Self::ArchiveApprentice => "Archive Apprentice",
            Self::CorporateUtilityDefector => "Corporate Utility Defector",
        }
    }

    fn field_deck_note(self) -> &'static str {
        match self {
            Self::BasinBornTechnician => {
                "Origin note: repair scars and water-pressure risk stand out first."
            }
            Self::ArchiveApprentice => {
                "Origin note: authority chain damage and witness gaps stand out first."
            }
            Self::CorporateUtilityDefector => {
                "Origin note: private firmware residue and ownership locks stand out first."
            }
        }
    }

    fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|profile| *profile == self)
            .unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingRepair {
    target_kind: InteractableKind,
    intervention: BasinIntervention,
}

#[derive(Resource)]
struct ChronicleRuntime {
    writer: Option<ChronicleWriter>,
    field_deck_raised_written: bool,
    lock_inspected_written: bool,
    last_preview_key: Option<String>,
    last_event: Option<String>,
    last_error: Option<String>,
    panel_open: bool,
    recent_events: Vec<ChronicleEventSummary>,
}

impl ChronicleRuntime {
    fn open_default() -> Self {
        match ChronicleWriter::open("chronicle") {
            Ok(writer) => {
                let recent_events = writer.recent_events(6).unwrap_or_default();
                Self {
                    writer: Some(writer),
                    field_deck_raised_written: false,
                    lock_inspected_written: false,
                    last_preview_key: None,
                    last_event: None,
                    last_error: None,
                    panel_open: false,
                    recent_events,
                }
            }
            Err(error) => Self {
                writer: None,
                field_deck_raised_written: false,
                lock_inspected_written: false,
                last_preview_key: None,
                last_event: None,
                last_error: Some(error.to_string()),
                panel_open: false,
                recent_events: Vec::new(),
            },
        }
    }

    fn append(&mut self, event_type: &str, payload: Value) {
        let Some(writer) = &mut self.writer else {
            return;
        };
        match writer.append(event_type, payload) {
            Ok(hash) => {
                self.last_event = Some(format!("{event_type} [{hash}]"));
                self.last_error = None;
                self.refresh_recent_events();
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
            }
        }
    }

    fn refresh_recent_events(&mut self) {
        let Some(writer) = &self.writer else {
            return;
        };
        match writer.recent_events(6) {
            Ok(events) => {
                self.recent_events = events;
                self.last_error = None;
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ChronicleEventSummary {
    logical_time: u64,
    event_type: String,
    hash_prefix: String,
    text: String,
}

#[derive(Component)]
struct EcologyMeter {
    kind: EcologyMeterKind,
}

#[derive(Component)]
struct SceneConsequenceVisual {
    kind: SceneVisualKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SceneVisualKind {
    Pump,
    Console,
    WillowRoots,
    AntTrail,
    MyceliumPatch,
    DirtyWater,
    EvidenceTag,
    Light,
}

#[derive(Debug, Clone, Copy)]
enum EcologyMeterKind {
    BasinToxin,
    ColonyStress,
    MyceliumBiomass,
}

struct ChronicleWriter {
    events_path: PathBuf,
    logical_time: u64,
    last_hash: String,
}

impl ChronicleWriter {
    fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(dir.join("snapshots"))?;
        fs::create_dir_all(dir.join("indexes"))?;
        fs::create_dir_all(dir.join("signatures"))?;

        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            fs::write(
                &manifest_path,
                concat!(
                    "{\n",
                    "  \"schema_version\": \"chronicle.manifest.v0\",\n",
                    "  \"worldline_id\": \"seed_age_firstlight\",\n",
                    "  \"site_id\": \"old_waterworks\",\n",
                    "  \"writer_version\": \"old_waterworks_micro_slice.v0\",\n",
                    "  \"min_reader_version\": \"chronicle.reader.v0\"\n",
                    "}\n"
                ),
            )?;
        }

        let events_path = dir.join("events.jsonl");
        if !events_path.exists() {
            fs::write(&events_path, "")?;
        }

        let (logical_time, last_hash) = verify_chronicle_file(&events_path)?;
        Ok(Self {
            events_path,
            logical_time,
            last_hash,
        })
    }

    fn append(&mut self, event_type: &str, payload: Value) -> io::Result<String> {
        self.logical_time += 1;
        let prev_hash = if self.logical_time == 1 {
            "GENESIS".to_string()
        } else {
            self.last_hash.clone()
        };

        let mut event = json!({
            "actor_id": "player_local",
            "event_id": format!("evt_{:08}", self.logical_time),
            "event_type": event_type,
            "logical_time": self.logical_time,
            "payload": payload,
            "prev_hash": prev_hash,
            "schema_version": "chronicle.event.v0",
            "site_id": "old_waterworks",
            "worldline_id": "seed_age_firstlight"
        });
        let hash = hash_event_without_signature(&event)?;
        event["hash"] = Value::String(hash.clone());
        event["signature"] = Value::String("placeholder".to_string());

        let line = serde_json::to_string(&event).map_err(json_error)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.events_path)?;
        writeln!(file, "{line}")?;
        self.last_hash = hash.clone();
        Ok(hash)
    }

    fn events_path(&self) -> &Path {
        &self.events_path
    }

    fn recent_events(&self, limit: usize) -> io::Result<Vec<ChronicleEventSummary>> {
        read_recent_chronicle_events(&self.events_path, limit)
    }
}

fn read_recent_chronicle_events(
    path: &Path,
    limit: usize,
) -> io::Result<Vec<ChronicleEventSummary>> {
    let content = fs::read_to_string(path)?;
    let mut summaries = Vec::new();
    for line in content.lines().filter(|line| !line.trim().is_empty()) {
        let event: Value = serde_json::from_str(line).map_err(json_error)?;
        let logical_time = event
            .get("logical_time")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let event_type = event
            .get("event_type")
            .and_then(Value::as_str)
            .unwrap_or("UnknownEvent")
            .to_string();
        let hash_prefix = event
            .get("hash")
            .and_then(Value::as_str)
            .map(|hash| hash.chars().take(12).collect())
            .unwrap_or_else(|| "missing-hash".to_string());
        let text = event
            .get("payload")
            .and_then(|payload| payload.get("chronicle_text"))
            .and_then(Value::as_str)
            .or_else(|| {
                event
                    .get("payload")
                    .and_then(|payload| payload.get("repair_path"))
                    .and_then(Value::as_str)
            })
            .or_else(|| {
                event
                    .get("payload")
                    .and_then(|payload| payload.get("authority_state"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("event recorded")
            .to_string();
        summaries.push(ChronicleEventSummary {
            logical_time,
            event_type,
            hash_prefix,
            text,
        });
    }
    let start = summaries.len().saturating_sub(limit);
    Ok(summaries.split_off(start))
}

fn verify_chronicle_file(path: &Path) -> io::Result<(u64, String)> {
    let content = fs::read_to_string(path)?;
    let mut expected_prev = "GENESIS".to_string();
    let mut last_hash = "GENESIS".to_string();
    let mut last_time = 0;

    for (line_index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(line).map_err(json_error)?;
        let logical_time = event
            .get("logical_time")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid_chronicle(line_index, "missing logical_time"))?;
        if logical_time != last_time + 1 {
            return Err(invalid_chronicle(
                line_index,
                "logical_time is not strictly increasing",
            ));
        }

        let prev_hash = event
            .get("prev_hash")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_chronicle(line_index, "missing prev_hash"))?;
        if prev_hash != expected_prev {
            return Err(invalid_chronicle(line_index, "prev_hash does not match"));
        }

        let stored_hash = event
            .get("hash")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_chronicle(line_index, "missing hash"))?;
        let computed_hash = hash_event_without_signature(&event)?;
        if stored_hash != computed_hash {
            return Err(invalid_chronicle(line_index, "hash mismatch"));
        }

        last_time = logical_time;
        last_hash = stored_hash.to_string();
        expected_prev = last_hash.clone();
    }

    Ok((last_time, last_hash))
}

fn hash_event_without_signature(event: &Value) -> io::Result<String> {
    let mut canonical = event.clone();
    if let Some(object) = canonical.as_object_mut() {
        object.remove("hash");
        object.remove("signature");
    }
    let bytes = serde_json::to_string(&canonical).map_err(json_error)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn json_error(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn invalid_chronicle(line_index: usize, reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("corrupt Chronicle at line {}: {reason}", line_index + 1),
    )
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
        .insert_resource(SliceRuntime::default())
        .insert_resource(ChronicleRuntime::open_default())
        .init_state::<InterfaceState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                scenario_dev_controls_system,
                slice_control_system,
                chronicle_panel_toggle_system,
                scenario_step_system,
                interaction_system,
                ecological_meter_system,
                consequence_visual_system,
                ui_visibility_system,
                objective_overlay_system,
                controls_overlay_system,
                dev_panel_overlay_system,
                chronicle_panel_system,
                ending_card_system,
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
    mut slice: ResMut<SliceRuntime>,
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
        slice.pending_repair = None;
        slice.ending_card_open = false;
    }
}

fn slice_control_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    intents: Res<IntentFrame>,
    mut slice: ResMut<SliceRuntime>,
) {
    if keyboard.just_pressed(KeyCode::KeyO) {
        slice.origin = slice.origin.next();
        slice.pending_repair = None;
    }

    if intents.just_pressed(InputIntent::PauseOrRelease) {
        slice.pending_repair = None;
        slice.ending_card_open = false;
    }
}

fn chronicle_panel_toggle_system(
    intents: Res<IntentFrame>,
    mut chronicle: ResMut<ChronicleRuntime>,
) {
    if intents.just_pressed(InputIntent::OpenChroniclePanel) {
        chronicle.panel_open = !chronicle.panel_open;
        chronicle.refresh_recent_events();
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Deliberate procedural material palette. The playable slice must not depend
    // on missing texture fallback colors to communicate the room.
    let dry_concrete_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.31, 0.30),
        perceptual_roughness: 0.92,
        ..default()
    });
    let wet_concrete_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.13, 0.15, 0.16),
        perceptual_roughness: 0.78,
        reflectance: 0.22,
        ..default()
    });
    let stained_concrete_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.23, 0.18),
        perceptual_roughness: 0.96,
        ..default()
    });
    let rusted_metal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.54, 0.20, 0.09),
        metallic: 0.35,
        perceptual_roughness: 0.88,
        ..default()
    });
    let painted_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.23, 0.34, 0.38),
        metallic: 0.45,
        perceptual_roughness: 0.65,
        ..default()
    });
    let console_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.27, 0.24),
        emissive: Color::srgb(0.02, 0.10, 0.08).to_linear() * 0.08,
        metallic: 0.2,
        perceptual_roughness: 0.7,
        ..default()
    });
    let warning_stripe_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.68, 0.10),
        emissive: Color::srgb(0.24, 0.14, 0.01).to_linear() * 0.08,
        ..default()
    });
    let dirty_water_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.05, 0.12, 0.13, 0.72),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.42,
        reflectance: 0.35,
        ..default()
    });
    let chronicle_seal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.74, 0.18),
        emissive: Color::srgb(0.95, 0.55, 0.12).to_linear() * 0.45,
        ..default()
    });
    let scanline_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.18, 0.72, 1.0, 0.58),
        emissive: Color::srgb(0.08, 0.45, 1.0).to_linear() * 0.35,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // Meshes
    let cube_mesh = meshes.add(Cuboid::default());

    // Floor
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wet_concrete_mat.clone()),
        Transform::from_xyz(0.0, -0.1, 0.0).with_scale(Vec3::new(20.0, 0.2, 20.0)),
    ));

    // Walls
    let wall_height = 4.0;
    // North
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(stained_concrete_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, -10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // South
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(dry_concrete_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, 10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // East
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(dry_concrete_mat.clone()),
        Transform::from_xyz(10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));
    // West
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(stained_concrete_mat.clone()),
        Transform::from_xyz(-10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));

    // Pump
    commands.spawn((
        SceneConsequenceVisual {
            kind: SceneVisualKind::Pump,
        },
        InteractableTarget {
            kind: InteractableKind::PumpConsole,
            label: "Pump console",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(rusted_metal_mat.clone()),
        Transform::from_xyz(-2.0, 1.0, -2.0).with_scale(Vec3::new(2.0, 2.0, 2.0)),
    ));

    // Tank / Pipe
    commands.spawn((
        SceneConsequenceVisual {
            kind: SceneVisualKind::MyceliumPatch,
        },
        InteractableTarget {
            kind: InteractableKind::MyceliumPatch,
            label: "Toxin-buffering mycelium",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(painted_steel_mat.clone()),
        Transform::from_xyz(-5.0, 1.5, 5.0).with_scale(Vec3::new(1.5, 3.0, 1.5)),
    ));

    // Console
    commands.spawn((
        Console,
        SceneConsequenceVisual {
            kind: SceneVisualKind::Console,
        },
        InteractableTarget {
            kind: InteractableKind::PumpConsole,
            label: "Dead-authority pump console",
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(console_mat.clone()),
        Transform::from_xyz(8.0, 1.2, 0.0).with_scale(Vec3::new(0.5, 1.0, 1.5)),
    ));

    // Low-cost silhouettes and decals for the material truth pass.
    let prop_specs = [
        (
            "overhead pipe",
            Vec3::new(-1.0, 3.05, -8.8),
            Vec3::new(14.0, 0.18, 0.18),
            painted_steel_mat.clone(),
        ),
        (
            "rust leak pipe",
            Vec3::new(-8.8, 1.9, -1.0),
            Vec3::new(0.18, 0.18, 10.5),
            rusted_metal_mat.clone(),
        ),
        (
            "valve wheel crossbar",
            Vec3::new(-6.2, 1.65, -4.5),
            Vec3::new(0.10, 0.9, 0.10),
            warning_stripe_mat.clone(),
        ),
        (
            "walkway grate",
            Vec3::new(2.5, 0.03, -5.8),
            Vec3::new(5.5, 0.05, 1.2),
            painted_steel_mat.clone(),
        ),
        (
            "warning sign",
            Vec3::new(5.5, 1.8, -9.85),
            Vec3::new(2.2, 0.75, 0.05),
            warning_stripe_mat.clone(),
        ),
        (
            "waterline stain",
            Vec3::new(-9.86, 0.85, 1.0),
            Vec3::new(0.04, 0.24, 12.0),
            stained_concrete_mat.clone(),
        ),
        (
            "repair crate",
            Vec3::new(4.2, 0.35, 3.8),
            Vec3::new(1.2, 0.7, 0.9),
            painted_steel_mat.clone(),
        ),
        (
            "exposed conduit",
            Vec3::new(8.9, 2.35, 2.8),
            Vec3::new(0.12, 0.12, 4.0),
            rusted_metal_mat.clone(),
        ),
        (
            "field deck scan plane",
            Vec3::new(-2.0, 2.25, -0.85),
            Vec3::new(2.6, 0.04, 1.7),
            scanline_mat.clone(),
        ),
    ];
    for (_label, position, scale, material) in prop_specs {
        commands.spawn((
            Mesh3d(cube_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position).with_scale(scale),
        ));
    }

    commands.spawn((
        SceneConsequenceVisual {
            kind: SceneVisualKind::DirtyWater,
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(dirty_water_mat),
        Transform::from_xyz(0.5, 0.02, 2.8).with_scale(Vec3::new(7.5, 0.04, 3.2)),
    ));

    commands.spawn((
        SceneConsequenceVisual {
            kind: SceneVisualKind::EvidenceTag,
        },
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(chronicle_seal_mat),
        Transform::from_xyz(-0.65, 2.18, -2.95).with_scale(Vec3::new(0.55, 0.08, 0.35)),
    ));

    let willow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.36, 0.12),
        emissive: Color::srgb(0.12, 0.20, 0.06).to_linear() * 0.08,
        ..default()
    });
    commands.spawn((
        SceneConsequenceVisual {
            kind: SceneVisualKind::WillowRoots,
        },
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
        SceneConsequenceVisual {
            kind: SceneVisualKind::AntTrail,
        },
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
        SceneConsequenceVisual {
            kind: SceneVisualKind::Light,
        },
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
            left: Val::Px(24.0),
            top: Val::Px(88.0),
            width: Val::Px(560.0),
            padding: UiRect::all(Val::Px(12.0)),
            display: Display::Flex,
            ..default()
        },
        BackgroundColor(Color::srgba(0.015, 0.02, 0.025, 0.72)),
        ObjectiveOverlay,
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.96, 1.0)),
    ));

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

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(32.0),
            bottom: Val::Px(32.0),
            width: Val::Px(680.0),
            padding: UiRect::all(Val::Px(16.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.015, 0.018, 0.028, 0.94)),
        ChroniclePanelOverlay,
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.94, 1.0)),
    ));

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(18.0),
            right: Val::Percent(18.0),
            top: Val::Percent(18.0),
            padding: UiRect::all(Val::Px(24.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.02, 0.018, 0.012, 0.96)),
        EndingCardOverlay,
        Text::new(""),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.90, 0.66)),
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

#[derive(Component)]
struct ChroniclePanelOverlay;

#[derive(Component)]
struct ObjectiveOverlay;

#[derive(Component)]
struct EndingCardOverlay;

fn interaction_system(
    intents: Res<IntentFrame>,
    mut runtime: ResMut<ScenarioRuntime>,
    mut slice: ResMut<SliceRuntime>,
    mut chronicle: ResMut<ChronicleRuntime>,
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

    if current_mode == ControlMode::FieldDeck && !chronicle.field_deck_raised_written {
        chronicle.append(
            "FieldDeckRaised",
            json!({
                "deck_id": "field_deck_mk0",
                "origin_profile": slice.origin.id(),
                "visor_assist_enabled": false,
                "location_hint": "old_waterworks"
            }),
        );
        chronicle.field_deck_raised_written = true;
    }

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
        if !chronicle.lock_inspected_written {
            chronicle.append("DeadAuthorityLockInspected", dead_authority_payload());
            chronicle.lock_inspected_written = true;
        }
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
        if near_target
            && target
                .map(|target| target.kind == InteractableKind::PumpConsole)
                .unwrap_or(false)
            && matches!(
                controls_state.scan_mode,
                ScanMode::MachineDiagnostics
                    | ScanMode::CivicClaims
                    | ScanMode::NullSignalCorruption
                    | ScanMode::RepairPreview
            )
            && !chronicle.lock_inspected_written
        {
            chronicle.append("DeadAuthorityLockInspected", dead_authority_payload());
            chronicle.lock_inspected_written = true;
        }

        if controls_state.scan_mode == ScanMode::RepairPreview {
            if let Some(target) = target.filter(|_| near_target) {
                if let Some(intervention) =
                    selected_repair_intervention(controls_state.selected_tool_slot, target.kind)
                {
                    let preview_key = format!("{:?}:{:?}", target.kind, intervention);
                    if chronicle.last_preview_key.as_deref() != Some(preview_key.as_str()) {
                        chronicle.append(
                            "RepairPathPreviewed",
                            repair_preview_payload(target, intervention, slice.origin),
                        );
                        chronicle.last_preview_key = Some(preview_key);
                    }
                    if intents.just_pressed(InputIntent::RepairTool) {
                        let pending = PendingRepair {
                            target_kind: target.kind,
                            intervention,
                        };
                        if slice.pending_repair == Some(pending) {
                            let outcome = runtime.scenario.apply_choice_and_step(intervention, 6);
                            chronicle.append(
                                "RepairPathCommitted",
                                repair_commit_payload(target, intervention, slice.origin),
                            );
                            chronicle.append(
                                "WaterworksOutcomeRecorded",
                                waterworks_outcome_payload(&outcome, slice.origin),
                            );
                            runtime.last_outcome = Some(outcome);
                            slice.pending_repair = None;
                            slice.ending_card_open = true;
                        } else {
                            slice.pending_repair = Some(pending);
                        }
                    }
                }
            }
        }

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
        let reading = format_field_deck_reading(
            controls_state.scan_mode,
            target.filter(|_| near_target),
            controls_state.selected_tool_slot,
            record,
            &slice,
        );
        let chronicle_status = format_chronicle_status(&chronicle);
        **inspect_text = format!(
            "FIELD DECK ECOLOGY LINK\n\
             Origin: {}\n\
             Mode: {} ([ / ] to cycle)\n\
             Tool slot: {}\n\
             {}\n\
             {}\n\
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
             {}\n\
             C: Chronicle evidence | V: scan overlay | F: focus reading",
            slice.origin.label(),
            controls_state.scan_mode.label(),
            controls_state.selected_tool_slot,
            current_target,
            slice.origin.field_deck_note(),
            reading,
            record.basin.viability,
            record.basin.toxin_load,
            record.basin.signal_corruption,
            record.colony.stress,
            record.mycelium_exchange.toxin_buffered,
            events,
            testimony,
            outcome,
            chronicle_status,
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

fn selected_repair_intervention(
    tool_slot: u8,
    target: InteractableKind,
) -> Option<BasinIntervention> {
    match (tool_slot, target) {
        (2, InteractableKind::PumpConsole) => Some(BasinIntervention::FastMechanicalRepair),
        (3, InteractableKind::PumpConsole | InteractableKind::AntTrail) => {
            Some(BasinIntervention::EcologicalReroute)
        }
        (4, InteractableKind::WillowRoots) => Some(BasinIntervention::CutWillowRoots),
        (5, InteractableKind::MyceliumPatch) => Some(BasinIntervention::DecomposerAid),
        (6, _) => Some(BasinIntervention::DelayRepair),
        _ => None,
    }
}

fn format_field_deck_reading(
    mode: ScanMode,
    target: Option<&InteractableTarget>,
    tool_slot: u8,
    record: &symtropy_basin::OldWaterworksTickRecord,
    slice: &SliceRuntime,
) -> String {
    let Some(target) = target else {
        return "Reading: no focused system in range.".to_string();
    };

    match mode {
        ScanMode::Infrastructure => match target.kind {
            InteractableKind::PumpConsole => format!(
                "SCAN/PUMP: pressure {:.2}, integrity {:.2}; physical repair possible.",
                record.basin.water_trust, record.basin.infrastructure_integrity
            ),
            InteractableKind::WillowRoots => {
                "SCAN/ROOTS: living root mass crosses pipe route; obstruction and filtration overlap."
                    .to_string()
            }
            InteractableKind::AntTrail => {
                "SCAN/TRAIL: traffic line is rerouting around wet floor and toxin pressure."
                    .to_string()
            }
            InteractableKind::MyceliumPatch => format!(
                "SCAN/MYCELIUM: biomass {:.2}; toxin buffer {:.3}.",
                record.mycelium.total_biomass, record.mycelium_exchange.toxin_buffered
            ),
        },
        ScanMode::Ecology => format!(
            "ECOLOGY: viability {:.2}, toxin {:.2}, recovery momentum {:.2}.",
            record.basin.viability, record.basin.toxin_load, record.basin.recovery_momentum
        ),
        ScanMode::MachineDiagnostics => {
            "DIAG: machine reports GREEN; Field Deck flags biological contradiction.".to_string()
        }
        ScanMode::CivicClaims => match target.kind {
            InteractableKind::PumpConsole => {
                "CIVIC: public water charter conflicts with dead emergency authority.".to_string()
            }
            InteractableKind::WillowRoots => {
                "CIVIC: Utility Sovereign sees obstruction; Watershed Commons sees living filtration."
                    .to_string()
            }
            InteractableKind::AntTrail => {
                "CIVIC: colony route is ecological evidence, not decorative wildlife.".to_string()
            }
            InteractableKind::MyceliumPatch => {
                "CIVIC: decomposer aid can become a public repair argument.".to_string()
            }
        },
        ScanMode::NullSignalCorruption => format!(
            "NULL: diagnostic coherence is suspect; signal corruption {:.2}.",
            record.basin.signal_corruption
        ),
        ScanMode::RepairPreview => match selected_repair_intervention(tool_slot, target.kind) {
            Some(intervention) => {
                let pending = PendingRepair {
                    target_kind: target.kind,
                    intervention,
                };
                if slice.pending_repair == Some(pending) {
                    format!(
                        "REPAIR CONFIRM: {:?} on {} is armed. Press R again to commit Chronicle precedent.",
                        intervention, target.label
                    )
                } else {
                    format!(
                        "REPAIR PREVIEW: slot {} selects {:?} on {}. Press R to arm confirmation.",
                        tool_slot, intervention, target.label
                    )
                }
            }
            None => format!(
                "REPAIR PREVIEW: slot {} has no valid repair action for {}.",
                tool_slot, target.label
            ),
        },
        ScanMode::ChronicleEvidence => {
            let latest = record
                .testimony
                .first()
                .map(|entry| entry.summary.as_str())
                .unwrap_or("No citable ecological testimony yet.");
            format!("CHRONICLE EVIDENCE: {latest}")
        }
    }
}

fn format_chronicle_status(chronicle: &ChronicleRuntime) -> String {
    if let Some(error) = &chronicle.last_error {
        return format!("Chronicle writer: blocked ({error})");
    }
    let Some(writer) = &chronicle.writer else {
        return "Chronicle writer: unavailable".to_string();
    };
    let last_event = chronicle
        .last_event
        .as_deref()
        .unwrap_or("no event written this session");
    format!(
        "Chronicle writer: {} | {}",
        writer.events_path().display(),
        last_event
    )
}

fn dead_authority_payload() -> Value {
    json!({
        "archive_trace": [
            "Built 2048: Municipal drought adaptation works.",
            "Modified 2087: Emergency Water Act automation.",
            "Authority chain failed approximately 2113."
        ],
        "authority_state": "DEAD_AUTHORITY_LOCK",
        "null_loop_detected": true,
        "null_loop_duration": "55 years",
        "pump_id": "PUMP_1",
        "tank_level_percent": 12,
        "target_id": "pump_1"
    })
}

fn repair_preview_payload(
    target: &InteractableTarget,
    intervention: BasinIntervention,
    origin: PlayerOrigin,
) -> Value {
    let warnings: Vec<&str> = match intervention {
        BasinIntervention::FastMechanicalRepair => vec![
            "Fast repair may leave ecological toxin memory unresolved.",
            "Public legitimacy may lag behind mechanical pressure.",
        ],
        BasinIntervention::EcologicalReroute => vec![
            "Repair timeline is slower.",
            "Living filtration and colony reroute evidence are preserved.",
        ],
        BasinIntervention::CutWillowRoots => vec![
            "Root obstruction falls.",
            "Living toxin filtration may be damaged.",
        ],
        BasinIntervention::DecomposerAid => vec![
            "Decomposer aid may improve recovery momentum.",
            "Outcome depends on toxin pressure after settling.",
        ],
        BasinIntervention::DelayRepair => vec![
            "Evidence quality improves.",
            "Settlement water trust may degrade during delay.",
        ],
        BasinIntervention::PipeLeak
        | BasinIntervention::WillowPlanting
        | BasinIntervention::NullGreenwash => {
            vec!["Scenario intervention is normally dev-controlled."]
        }
    };
    json!({
        "predicted_outcome_class": format!("{intervention:?}"),
        "repair_path": format!("{intervention:?}"),
        "origin_profile": origin.id(),
        "origin_note": origin.field_deck_note(),
        "target_id": format!("{:?}", target.kind),
        "target_label": target.label,
        "visible_warnings": warnings
    })
}

fn repair_commit_payload(
    target: &InteractableTarget,
    intervention: BasinIntervention,
    origin: PlayerOrigin,
) -> Value {
    json!({
        "charter_context": "Firstlight Public Repair Charter",
        "origin_profile": origin.id(),
        "origin_label": origin.label(),
        "repair_path": format!("{intervention:?}"),
        "target_id": format!("{:?}", target.kind),
        "target_label": target.label,
        "witness_mode_used": true
    })
}

fn waterworks_outcome_payload(
    outcome: &symtropy_basin::OldWaterworksChoiceOutcome,
    origin: PlayerOrigin,
) -> Value {
    let chronicle_text = outcome
        .chronicle
        .first()
        .map(|event| event.summary.clone())
        .unwrap_or_else(|| format!("Chronicle: {:?} committed.", outcome.intervention));
    let faction_memory_flags = faction_memory_flags(outcome.intervention);
    json!({
        "authority_state_after": authority_state_after(outcome.intervention),
        "basin_viability": outcome.basin.viability,
        "chronicle_text": chronicle_text,
        "origin_profile": origin.id(),
        "origin_label": origin.label(),
        "origin_interpretation": origin.field_deck_note(),
        "event_summaries": outcome
            .chronicle
            .iter()
            .map(|event| event.summary.clone())
            .collect::<Vec<_>>(),
        "faction_memory_flags": faction_memory_flags,
        "faction_reactions": outcome
            .faction_reactions
            .iter()
            .map(|reaction| {
                json!({
                    "confidence": reaction.confidence,
                    "faction": format!("{:?}", reaction.faction),
                    "stance": format!("{:?}", reaction.stance),
                    "summary": reaction.summary
                })
            })
            .collect::<Vec<_>>(),
        "legitimacy_effect": legitimacy_effect(outcome.intervention),
        "null_drift_effect": null_drift_effect(outcome.intervention),
        "outcome_class": outcome_class(outcome.intervention),
        "repair_path": format!("{:?}", outcome.intervention),
        "tick": outcome.tick,
        "water_flow_state": water_flow_state(outcome.intervention)
    })
}

#[cfg(test)]
fn write_waterworks_chronicle_sequence(
    writer: &mut ChronicleWriter,
    target: &InteractableTarget,
    intervention: BasinIntervention,
    outcome: &symtropy_basin::OldWaterworksChoiceOutcome,
    origin: PlayerOrigin,
) -> io::Result<()> {
    writer.append(
        "FieldDeckRaised",
        json!({
            "deck_id": "field_deck_mk0",
            "origin_profile": origin.id(),
            "visor_assist_enabled": false,
            "location_hint": "old_waterworks"
        }),
    )?;
    writer.append("DeadAuthorityLockInspected", dead_authority_payload())?;
    writer.append(
        "RepairPathPreviewed",
        repair_preview_payload(target, intervention, origin),
    )?;
    writer.append(
        "WaterworksOutcomeRecorded",
        waterworks_outcome_payload(outcome, origin),
    )?;
    Ok(())
}

fn outcome_class(intervention: BasinIntervention) -> &'static str {
    match intervention {
        BasinIntervention::FastMechanicalRepair => "FastMechanicalRepair",
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => {
            "EcologicalRepair"
        }
        BasinIntervention::CutWillowRoots => "AccessStabilizedFiltrationDamaged",
        BasinIntervention::DelayRepair => "EvidenceReviewDelay",
        BasinIntervention::PipeLeak => "DamageRecorded",
        BasinIntervention::WillowPlanting => "LivingFiltrationExpanded",
        BasinIntervention::NullGreenwash => "FalseLegibilityDetected",
    }
}

fn water_flow_state(intervention: BasinIntervention) -> &'static str {
    match intervention {
        BasinIntervention::FastMechanicalRepair | BasinIntervention::CutWillowRoots => {
            "RESTORED_FAST"
        }
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => {
            "STABILIZING_SLOW"
        }
        BasinIntervention::DelayRepair => "DEFERRED",
        BasinIntervention::PipeLeak => "FAILING",
        BasinIntervention::WillowPlanting => "FILTERING",
        BasinIntervention::NullGreenwash => "DIAGNOSTIC_GREEN_ONLY",
    }
}

fn authority_state_after(intervention: BasinIntervention) -> &'static str {
    match intervention {
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => {
            "PUBLIC_REPAIR_UNDER_ECOLOGICAL_EVIDENCE"
        }
        BasinIntervention::DelayRepair => "ARCHIVE_REVIEW_OPEN",
        BasinIntervention::FastMechanicalRepair | BasinIntervention::CutWillowRoots => "UNRESOLVED",
        _ => "UNCHANGED",
    }
}

fn legitimacy_effect(intervention: BasinIntervention) -> &'static str {
    match intervention {
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => {
            "LEGITIMACY_INCREASED"
        }
        BasinIntervention::DelayRepair => "LEGITIMACY_UNRESOLVED",
        BasinIntervention::FastMechanicalRepair | BasinIntervention::CutWillowRoots => {
            "LEGITIMACY_DEBT_INCREASED"
        }
        _ => "LEGITIMACY_UNCHANGED",
    }
}

fn null_drift_effect(intervention: BasinIntervention) -> &'static str {
    match intervention {
        BasinIntervention::NullGreenwash => "NULL_REINFORCEMENT_CONTINUES",
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => {
            "NULL_DRIFT_REDUCED_BY_CONTRADICTORY_EVIDENCE"
        }
        _ => "NULL_DRIFT_UNRESOLVED",
    }
}

fn faction_memory_flags(intervention: BasinIntervention) -> Vec<&'static str> {
    match intervention {
        BasinIntervention::FastMechanicalRepair => vec![
            "old_waterworks_repaired_fast",
            "archive_witness_bypassed",
            "ecological_memory_unresolved",
        ],
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => vec![
            "old_waterworks_repaired_legitimately",
            "living_infrastructure_respected",
            "dead_authority_overturned_by_evidence",
        ],
        BasinIntervention::CutWillowRoots => vec![
            "old_waterworks_access_restored",
            "living_filtration_damaged",
            "utility_precedent_created",
        ],
        BasinIntervention::DelayRepair => vec![
            "archive_witness_respected",
            "water_repair_delayed_for_evidence",
            "continuance_precedent_created",
        ],
        BasinIntervention::PipeLeak => vec!["pipe_leak_recorded"],
        BasinIntervention::WillowPlanting => vec!["living_filtration_expanded"],
        BasinIntervention::NullGreenwash => vec!["false_legibility_detected"],
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

fn consequence_visual_system(
    runtime: Res<ScenarioRuntime>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut material_query: Query<(
        &SceneConsequenceVisual,
        &MeshMaterial3d<StandardMaterial>,
        &mut Transform,
    )>,
    mut light_query: Query<(&SceneConsequenceVisual, &mut PointLight)>,
) {
    let Some(outcome) = runtime.last_outcome.as_ref() else {
        return;
    };

    for (visual, material, mut transform) in &mut material_query {
        let color = consequence_color(outcome.intervention, visual.kind);
        if let Some(mat) = materials.get_mut(&material.0) {
            mat.base_color = color;
            mat.emissive =
                color.to_linear() * consequence_emissive(outcome.intervention, visual.kind);
        }
        transform.scale = consequence_scale(outcome.intervention, visual.kind, transform.scale);
    }

    for (visual, mut light) in &mut light_query {
        if visual.kind == SceneVisualKind::Light {
            light.color = consequence_color(outcome.intervention, SceneVisualKind::Light);
        }
    }
}

fn consequence_color(intervention: BasinIntervention, visual: SceneVisualKind) -> Color {
    match intervention {
        BasinIntervention::FastMechanicalRepair => match visual {
            SceneVisualKind::Pump
            | SceneVisualKind::Console
            | SceneVisualKind::EvidenceTag
            | SceneVisualKind::Light => Color::srgb(0.35, 0.85, 1.0),
            SceneVisualKind::WillowRoots | SceneVisualKind::MyceliumPatch => {
                Color::srgb(0.35, 0.28, 0.16)
            }
            SceneVisualKind::AntTrail => Color::srgb(0.85, 0.55, 0.12),
            SceneVisualKind::DirtyWater => Color::srgba(0.08, 0.18, 0.20, 0.68),
        },
        BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid => match visual {
            SceneVisualKind::Pump | SceneVisualKind::Console => Color::srgb(0.45, 0.62, 0.70),
            SceneVisualKind::WillowRoots
            | SceneVisualKind::MyceliumPatch
            | SceneVisualKind::EvidenceTag
            | SceneVisualKind::Light => Color::srgb(0.18, 0.85, 0.32),
            SceneVisualKind::AntTrail => Color::srgb(1.0, 0.78, 0.18),
            SceneVisualKind::DirtyWater => Color::srgba(0.04, 0.16, 0.12, 0.60),
        },
        BasinIntervention::CutWillowRoots => match visual {
            SceneVisualKind::Pump
            | SceneVisualKind::Console
            | SceneVisualKind::EvidenceTag
            | SceneVisualKind::Light => Color::srgb(0.92, 0.82, 0.35),
            SceneVisualKind::WillowRoots | SceneVisualKind::MyceliumPatch => {
                Color::srgb(0.55, 0.12, 0.08)
            }
            SceneVisualKind::AntTrail => Color::srgb(0.75, 0.45, 0.10),
            SceneVisualKind::DirtyWater => Color::srgba(0.18, 0.10, 0.07, 0.72),
        },
        BasinIntervention::DelayRepair => match visual {
            SceneVisualKind::Light => Color::srgb(1.0, 0.55, 0.18),
            SceneVisualKind::EvidenceTag => Color::srgb(1.0, 0.72, 0.20),
            SceneVisualKind::DirtyWater => Color::srgba(0.12, 0.13, 0.12, 0.78),
            _ => Color::srgb(0.55, 0.50, 0.42),
        },
        BasinIntervention::NullGreenwash => Color::srgb(0.08, 1.0, 0.42),
        BasinIntervention::PipeLeak => Color::srgb(0.85, 0.12, 0.10),
        BasinIntervention::WillowPlanting => Color::srgb(0.16, 0.75, 0.26),
    }
}

fn consequence_emissive(intervention: BasinIntervention, visual: SceneVisualKind) -> f32 {
    match (intervention, visual) {
        (
            BasinIntervention::FastMechanicalRepair,
            SceneVisualKind::Pump | SceneVisualKind::Console | SceneVisualKind::EvidenceTag,
        ) => 0.25,
        (
            BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid,
            SceneVisualKind::WillowRoots
            | SceneVisualKind::MyceliumPatch
            | SceneVisualKind::AntTrail
            | SceneVisualKind::EvidenceTag,
        ) => 0.30,
        (
            BasinIntervention::CutWillowRoots,
            SceneVisualKind::WillowRoots | SceneVisualKind::MyceliumPatch,
        ) => 0.10,
        (BasinIntervention::NullGreenwash, _) => 0.35,
        _ => 0.08,
    }
}

fn consequence_scale(
    intervention: BasinIntervention,
    visual: SceneVisualKind,
    current: Vec3,
) -> Vec3 {
    match (intervention, visual) {
        (BasinIntervention::CutWillowRoots, SceneVisualKind::WillowRoots) => {
            Vec3::new(current.x, 0.08, current.z)
        }
        (
            BasinIntervention::EcologicalReroute | BasinIntervention::DecomposerAid,
            SceneVisualKind::MyceliumPatch,
        ) => Vec3::new(current.x, current.y.max(3.4), current.z),
        (BasinIntervention::EcologicalReroute, SceneVisualKind::AntTrail) => {
            Vec3::new(current.x.max(4.8), current.y, current.z.max(0.24))
        }
        _ => current,
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
         R in Repair Preview: arm, then confirm repair consequence\n\
         C Chronicle/evidence | O cycle origin | M basin map | Ctrl+K command palette\n\
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

fn objective_overlay_system(
    controls: Res<ControlsState>,
    runtime: Res<ScenarioRuntime>,
    slice: Res<SliceRuntime>,
    chronicle: Res<ChronicleRuntime>,
    mut query: Query<&mut Text, With<ObjectiveOverlay>>,
) {
    let Ok(mut text) = query.single_mut() else {
        return;
    };
    let objective = current_objective(&runtime, &slice, &chronicle, &controls);
    **text = format!(
        "OLD WATERWORKS OBJECTIVE\n\
         Origin: {} (press O to cycle prototype origin)\n\
         {}",
        slice.origin.label(),
        objective
    );
}

fn current_objective(
    runtime: &ScenarioRuntime,
    slice: &SliceRuntime,
    chronicle: &ChronicleRuntime,
    controls: &ControlsState,
) -> String {
    if runtime.last_outcome.is_some() {
        return "Outcome recorded. Press C for Chronicle evidence or Esc to close ending card."
            .to_string();
    }
    if slice.pending_repair.is_some() {
        return "Repair path armed. Press R again in Repair Preview to commit public precedent."
            .to_string();
    }
    if chronicle
        .recent_events
        .iter()
        .any(|event| event.event_type == "RepairPathPreviewed")
    {
        return "Repair path previewed. Choose the consequence you are willing to make public."
            .to_string();
    }
    if chronicle.lock_inspected_written {
        return "Dead authority inspected. Open Field Deck Repair Preview and select a tool slot."
            .to_string();
    }
    if controls.mode == ControlMode::FieldDeck {
        return "Field Deck raised. Inspect the dead-authority pump in DIAG, CIVIC, NULL, or REPAIR."
            .to_string();
    }
    "Inspect the dead-authority pump. Use Tab for Field Deck, E near the console, F1 for controls."
        .to_string()
}

fn ending_card_system(
    slice: Res<SliceRuntime>,
    runtime: Res<ScenarioRuntime>,
    chronicle: Res<ChronicleRuntime>,
    mut query: Query<(&mut Node, &mut Text), With<EndingCardOverlay>>,
) {
    let Ok((mut node, mut text)) = query.single_mut() else {
        return;
    };
    node.display = if slice.ending_card_open {
        Display::Flex
    } else {
        Display::None
    };
    if !slice.ending_card_open {
        return;
    }

    let Some(outcome) = runtime.last_outcome.as_ref() else {
        **text = "No outcome recorded yet.".to_string();
        return;
    };
    let hash = chronicle
        .recent_events
        .iter()
        .rev()
        .find(|event| event.event_type == "WaterworksOutcomeRecorded")
        .map(|event| event.hash_prefix.as_str())
        .unwrap_or("pending");
    let faction = outcome
        .faction_reactions
        .iter()
        .map(|reaction| {
            format!(
                "{:?} {:?}: {}",
                reaction.faction, reaction.stance, reaction.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    **text = format!(
        "OLD WATERWORKS OUTCOME RECORDED\n\
         Origin: {}\n\
         Repair path: {:?}\n\
         Water state: {}\n\
         Legitimacy: {}\n\
         Authority: {}\n\
         Chronicle hash: {}\n\n\
         Faction reactions:\n{}\n\n\
         Press C for full Chronicle panel. Press Esc to return.",
        slice.origin.label(),
        outcome.intervention,
        water_flow_state(outcome.intervention),
        legitimacy_effect(outcome.intervention),
        authority_state_after(outcome.intervention),
        hash,
        faction
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

fn chronicle_panel_system(
    chronicle: Res<ChronicleRuntime>,
    runtime: Res<ScenarioRuntime>,
    mut query: Query<(&mut Node, &mut Text), With<ChroniclePanelOverlay>>,
) {
    let Ok((mut node, mut text)) = query.single_mut() else {
        return;
    };
    node.display = if chronicle.panel_open {
        Display::Flex
    } else {
        Display::None
    };
    if !chronicle.panel_open {
        return;
    }

    let event_lines = if chronicle.recent_events.is_empty() {
        "No Chronicle events recorded yet.".to_string()
    } else {
        chronicle
            .recent_events
            .iter()
            .rev()
            .map(|event| {
                format!(
                    "#{:03} {} [{}]\n  {}",
                    event.logical_time, event.event_type, event.hash_prefix, event.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let outcome = runtime
        .last_outcome
        .as_ref()
        .map(format_outcome)
        .unwrap_or_else(|| "No committed repair outcome yet.".to_string());
    let writer = chronicle
        .writer
        .as_ref()
        .map(|writer| writer.events_path().display().to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    let error = chronicle.last_error.as_deref().unwrap_or("none");

    **text = format!(
        "CHRONICLE / CIVIC EVIDENCE PANEL (C)\n\
         Source: {writer}\n\
         Last writer error: {error}\n\n\
         Latest outcome:\n{outcome}\n\n\
         Recent events:\n{event_lines}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn chronicle_writer_appends_hash_chain() {
        let dir = temp_chronicle_dir("hash_chain");
        let mut writer = ChronicleWriter::open(&dir).expect("open Chronicle writer");

        let first = writer
            .append("FieldDeckRaised", json!({"deck_id": "field_deck_mk0"}))
            .expect("append first event");
        let second = writer
            .append("DeadAuthorityLockInspected", dead_authority_payload())
            .expect("append second event");

        let lines = fs::read_to_string(dir.join("events.jsonl")).expect("read events");
        let events: Vec<Value> = lines
            .lines()
            .map(|line| serde_json::from_str(line).expect("parse event"))
            .collect();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["prev_hash"], "GENESIS");
        assert_eq!(events[0]["hash"], first);
        assert_eq!(events[1]["prev_hash"], first);
        assert_eq!(events[1]["hash"], second);

        let (logical_time, last_hash) =
            verify_chronicle_file(&dir.join("events.jsonl")).expect("verify Chronicle");
        assert_eq!(logical_time, 2);
        assert_eq!(last_hash, second);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn chronicle_verifier_rejects_tampering() {
        let dir = temp_chronicle_dir("tamper");
        let mut writer = ChronicleWriter::open(&dir).expect("open Chronicle writer");
        writer
            .append("FieldDeckRaised", json!({"deck_id": "field_deck_mk0"}))
            .expect("append event");

        let path = dir.join("events.jsonl");
        let tampered = fs::read_to_string(&path)
            .expect("read events")
            .replace("field_deck_mk0", "field_deck_mk9");
        fs::write(&path, tampered).expect("write tampered events");

        let error = verify_chronicle_file(&path).expect_err("tampering must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("hash mismatch"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn repair_preview_payload_names_target_and_warning() {
        let target = InteractableTarget {
            kind: InteractableKind::WillowRoots,
            label: "Willow root filtration",
        };
        let payload = repair_preview_payload(
            &target,
            BasinIntervention::CutWillowRoots,
            PlayerOrigin::ArchiveApprentice,
        );

        assert_eq!(payload["repair_path"], "CutWillowRoots");
        assert_eq!(payload["origin_profile"], "archive_apprentice");
        assert_eq!(payload["target_id"], "WillowRoots");
        assert!(
            payload["visible_warnings"]
                .as_array()
                .expect("warnings")
                .iter()
                .any(|warning| warning.as_str().unwrap_or("").contains("filtration"))
        );
    }

    #[test]
    fn golden_playable_loop_writes_required_chronicle_sequence() {
        let dir = temp_chronicle_dir("golden_loop");
        let mut writer = ChronicleWriter::open(&dir).expect("open Chronicle writer");
        let mut scenario = OldWaterworksScenario::new(16, 9);
        scenario.apply(BasinIntervention::PipeLeak);
        scenario.apply(BasinIntervention::NullGreenwash);
        let outcome = scenario.apply_choice_and_step(BasinIntervention::EcologicalReroute, 6);
        let target = InteractableTarget {
            kind: InteractableKind::PumpConsole,
            label: "Dead-authority pump console",
        };

        write_waterworks_chronicle_sequence(
            &mut writer,
            &target,
            BasinIntervention::EcologicalReroute,
            &outcome,
            PlayerOrigin::BasinBornTechnician,
        )
        .expect("write golden sequence");

        let path = dir.join("events.jsonl");
        let (logical_time, _) = verify_chronicle_file(&path).expect("valid hash chain");
        assert_eq!(logical_time, 4);

        let lines = fs::read_to_string(&path).expect("read events");
        let event_types: Vec<String> = lines
            .lines()
            .map(|line| {
                let event: Value = serde_json::from_str(line).expect("parse event");
                event["event_type"]
                    .as_str()
                    .expect("event_type")
                    .to_string()
            })
            .collect();
        assert_eq!(
            event_types,
            [
                "FieldDeckRaised",
                "DeadAuthorityLockInspected",
                "RepairPathPreviewed",
                "WaterworksOutcomeRecorded"
            ]
        );

        let recent = read_recent_chronicle_events(&path, 2).expect("recent events");
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[1].event_type, "WaterworksOutcomeRecorded");
        assert!(
            recent[1].text.contains("ecological repair trajectory")
                || recent[1].text.contains("Chronicle:")
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn old_waterworks_material_truth_gate_has_required_placeholders() {
        assert!(OLD_WATERWORKS_VISIBLE_PROP_COUNT >= 5);
        assert!(OLD_WATERWORKS_MATERIAL_PALETTE.contains(&"wet_concrete"));
        assert!(OLD_WATERWORKS_MATERIAL_PALETTE.contains(&"rusted_metal"));
        assert!(OLD_WATERWORKS_MATERIAL_PALETTE.contains(&"dirty_water"));
        assert!(OLD_WATERWORKS_MATERIAL_PALETTE.contains(&"chronicle_seal"));
        assert!(
            !OLD_WATERWORKS_MATERIAL_PALETTE
                .iter()
                .any(|material| material.contains("magenta") || material.contains("missing"))
        );
    }

    fn temp_chronicle_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("symtropy_old_waterworks_{name}_{unique}"))
    }
}
