// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global ECS resources for Symtropy.

use bevy::prelude::*;
use std::collections::HashMap;
use symthaea_biometrics::input_telemetry::InputTelemetryEncoder;
use symthaea_biometrics::stress_model::PlayerStressModel;
// TODO: re-enable when symthaea-muse compiles with muse-live
// use symthaea_muse::live_output::LiveMuseOutput;

/// Player behavioral biometrics state.
#[derive(Resource)]
pub struct BiometricsCtx {
    pub encoder: InputTelemetryEncoder,
    pub model: PlayerStressModel,
}

impl Default for BiometricsCtx {
    fn default() -> Self {
        Self {
            encoder: InputTelemetryEncoder::new(),
            model: PlayerStressModel::new(),
        }
    }
}

/// Leviathan sleep phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepPhase {
    /// Safe — no audio effect, free exploration.
    Dormant,
    /// Warning — sub-bass hum, lights flicker.
    Stirring,
    /// Danger — full horror audio, doors seal.
    Awake,
    /// Lethal — Leviathan hunts the player.
    Hunting,
}

/// Global Leviathan state.
#[derive(Resource)]
pub struct LeviathanState {
    /// Current sleep phase.
    pub phase: SleepPhase,
    /// Accumulated noise from all sources [0, ∞).
    pub noise_accumulator: f32,
    /// Noise threshold to transition Dormant → Stirring.
    pub threshold: f32,
    /// Seconds the Leviathan has been in Stirring phase.
    pub stirring_duration: f32,
    /// Seconds of quiet since last noise.
    pub quiet_duration: f32,
    /// Grace period (seconds) at game start — noise ignored while > 0.
    pub grace_timer: f32,
}

/// Grace period before the Leviathan starts listening (seconds).
const LEVIATHAN_GRACE_SECS: f32 = 20.0;

impl Default for LeviathanState {
    fn default() -> Self {
        Self {
            phase: SleepPhase::Dormant,
            noise_accumulator: 0.0,
            threshold: 10.0, // high threshold — player must actively make noise to wake it
            stirring_duration: 0.0,
            quiet_duration: 0.0,
            grace_timer: LEVIATHAN_GRACE_SECS,
        }
    }
}

/// Dungeon seed for reproducible levels.
#[derive(Resource)]
pub struct DungeonSeed(pub u64);

impl Default for DungeonSeed {
    fn default() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(42),
        )
    }
}

/// Game state for phase management.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Hash, States)]
pub enum GamePhase {
    /// Title screen with menu.
    #[default]
    MainMenu,
    /// Generating the dungeon.
    Loading,
    /// Active gameplay.
    Playing,
    /// Governance council session — gameplay paused for voting.
    Council,
    /// Leviathan caught the player.
    GameOver,
    /// Player escaped with the core.
    Victory,
    /// Sol Atlas globe view — planetary coordination layer.
    #[cfg(feature = "atlas")]
    GlobeView,
    /// Phase 11: City-Scale Governance Demonstration.
    CityScale,
    /// Muse: Thermodynamic Visualizer.
    Muse,
}

// ============================================================================
// Governance / Economy event log (Phase A0)
// ============================================================================

/// Log of governance events for the HUD and post-game analysis.
#[derive(Resource, Default)]
pub struct GovernanceLog {
    /// Recent governance event messages (ring buffer, max 20).
    pub messages: Vec<GovernanceMessage>,
}

/// A timestamped governance message.
pub struct GovernanceMessage {
    /// Game time when the event occurred.
    pub time_secs: f32,
    /// Human-readable event description.
    pub text: String,
    /// Severity: 0=info, 1=warning (oppression), 2=critical (crisis).
    pub severity: u8,
}

impl GovernanceLog {
    /// Push a message, keeping max 20 entries.
    pub fn push(&mut self, time_secs: f32, text: String, severity: u8) {
        self.messages.push(GovernanceMessage {
            time_secs,
            text,
            severity,
        });
        if self.messages.len() > 20 {
            self.messages.remove(0);
        }
    }
}

// ============================================================================
// Tile Grid (H1.1 — wall collision)
// ============================================================================

/// Spatial lookup for tile walkability. O(1) collision checks.
#[derive(Resource, Default)]
pub struct TileGrid {
    /// Map from (grid_x, grid_y) → walkable.
    pub cells: HashMap<(i32, i32), bool>,
    /// Tile size in pixels.
    pub tile_size: f32,
    /// Grid origin offset (half-width, half-height in tiles).
    pub origin_col: i32,
    pub origin_row: i32,
    pub cols: i32,
    pub rows: i32,
}

impl TileGrid {
    /// Check if a world-space position is walkable.
    pub fn is_walkable(&self, world_x: f32, world_y: f32) -> bool {
        let col = ((world_x / self.tile_size) + self.origin_col as f32).round() as i32;
        let row = (self.origin_row as f32 - (world_y / self.tile_size)).round() as i32;
        self.cells.get(&(col, row)).copied().unwrap_or(false)
    }
}

// ============================================================================
// Physics Engine (Symtropy consciousness-physics runtime)
// ============================================================================

/// 2D physics world resource — wraps the ND physics engine AND consciousness field.
///
/// The consciousness field is wired INTO the physics step via `step_with_callback`,
/// making Φ a real physical force that modulates impulses, friction, and energy.
#[derive(Resource)]
pub struct PhysicsWorldRes {
    pub world: symtropy_physics::PhysicsWorld<2>,
    pub consciousness: symtropy_consciousness_physics::ConsciousnessField<2>,
}

impl Default for PhysicsWorldRes {
    fn default() -> Self {
        Self {
            world: symtropy_physics::PhysicsWorld::new(nalgebra::SVector::from([0.0, 0.0])),
            consciousness: symtropy_consciousness_physics::ConsciousnessField::new(),
        }
    }
}

/// Buffered player input — written in Update, consumed in FixedUpdate.
///
/// This bridges the gap between Bevy's Update (where `just_pressed` works)
/// and FixedUpdate (where physics must run at a consistent timestep).
#[derive(Resource, Default)]
pub struct PlayerInput {
    /// Desired movement direction (not normalized — magnitude encodes intent).
    pub direction: Vec2,
    /// Whether the player is sprinting.
    pub sprinting: bool,
}

// ============================================================================
// Energy Wells (thermodynamic life sources)
// ============================================================================

/// Spatial energy source in the dungeon. Entities within radius regenerate energy.
/// Wells have finite capacity — they deplete over time (forcing migration).
#[derive(Component)]
pub struct EnergyWell {
    /// Joules per tick for entities in range.
    pub regen_rate: f64,
    /// Effect radius in world units.
    pub radius: f32,
    /// Remaining Joules in this well.
    pub remaining: f64,
    /// Maximum capacity (for display).
    pub max_capacity: f64,
}

impl EnergyWell {
    pub fn new(regen_rate: f64, radius: f32, capacity: f64) -> Self {
        Self {
            regen_rate,
            radius,
            remaining: capacity,
            max_capacity: capacity,
        }
    }

    /// Fraction of energy remaining [0, 1].
    pub fn fraction_remaining(&self) -> f64 {
        if self.max_capacity < 1e-10 {
            return 0.0;
        }
        self.remaining / self.max_capacity
    }

    /// Whether this well still has energy.
    pub fn is_active(&self) -> bool {
        self.remaining > 1e-10
    }
}

// ============================================================================
// Live Audio Output (H1.2)
// ============================================================================

// AudioOutput moved to systems/audio.rs as AudioState
