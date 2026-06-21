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
    /// Active 3D gameplay (Milestone H1.5).
    Playing3D,
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
    pub messages: std::collections::VecDeque<GovernanceMessage>,
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
        self.messages.push_back(GovernanceMessage {
            time_secs,
            text,
            severity,
        });
        if self.messages.len() > 20 {
            self.messages.pop_front();
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
// Settlement Metrics (Firstlight Basin Vertical Slice)
// ============================================================================

/// Global metrics for the Firstlight Basin settlement.
/// These metrics drive NPC behavior and the First Public Vote.
#[derive(Resource, Debug, Clone)]
pub struct SettlementMetrics {
    /// Power stability [0, 1].
    pub power: f32,
    /// Water availability [0, 1].
    pub water: f32,
    /// Food reserves [0, 1].
    pub food: f32,
    /// Infrastructure repair quality [0, 1].
    pub repair: f32,
    /// Collective trust in leadership/player [0, 1].
    pub trust: f32,
    /// Institutional legitimacy [0, 1].
    pub legitimacy: f32,
    /// Physical safety [0, 1].
    pub safety: f32,
    /// Systemic entropy [0, 1].
    pub entropy: f32,
}

impl Default for SettlementMetrics {
    fn default() -> Self {
        Self {
            power: 0.2,      // unstable
            water: 0.1,      // critical
            food: 0.3,       // low
            repair: 0.2,     // poor
            trust: 0.4,      // fragile
            legitimacy: 0.3, // provisional
            safety: 0.3,     // weak
            entropy: 0.6,    // rising
        }
    }
}

// ============================================================================
// Governance Vote (Firstlight Basin Vertical Slice)
// ============================================================================

/// A strategic decision for the settlement.
#[derive(Debug, Clone)]
pub struct VoteOption {
    pub label: String,
    pub description: String,
    pub effect_text: String,
}

/// Active governance vote state.
#[derive(Resource, Debug, Clone, Default)]
pub struct GovernanceVote {
    pub is_active: bool,
    pub question: String,
    pub options: Vec<VoteOption>,
    pub selected_index: usize,
}

impl GovernanceVote {
    pub fn new_water_crisis_vote() -> Self {
        Self {
            is_active: true,
            question: "What should Seedworks become after surviving the water crisis?".to_string(),
            options: vec![
                VoteOption {
                    label: "Public Repair".to_string(),
                    description: "Reinforce shared infrastructure and housing.".to_string(),
                    effect_text: "Trust rises, repair costs decrease.".to_string(),
                },
                VoteOption {
                    label: "Factory Overdrive".to_string(),
                    description: "Prioritize fabrication and machine expansion.".to_string(),
                    effect_text: "Production rises, pollution risk begins.".to_string(),
                },
                VoteOption {
                    label: "Perimeter Defense".to_string(),
                    description: "Build walls and floodlights.".to_string(),
                    effect_text: "Safety improves, trust may split.".to_string(),
                },
                VoteOption {
                    label: "Archive Recovery".to_string(),
                    description: "Investigate the Ghost Civic Center.".to_string(),
                    effect_text: "Knowledge increases, old systems awaken.".to_string(),
                },
            ],
            selected_index: 0,
        }
    }
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

// ============================================================================
// Shared Presentation / Simulation Site Layout
// ============================================================================

/// Global site layout describing dungeon room centers and tiles.
#[derive(Resource, Clone, Debug, Default)]
pub struct SiteLayout {
    pub site_id: String,
    pub width: usize,
    pub height: usize,
    /// 0=wall, 1=floor, 2=core_room, 3=player_start
    pub tiles: Vec<Vec<u8>>,
    /// Center of rooms
    pub room_centers: Vec<(usize, usize)>,
    /// Coordinates of special points (player spawn, core, etc.) in world space
    pub player_start: Vec2,
    pub core_pos: Vec2,
}
