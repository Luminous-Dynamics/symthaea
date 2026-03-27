// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ECS components for Symtropy entities.

use bevy::prelude::*;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

/// Marker for the player entity.
#[derive(Component)]
pub struct Player;

/// FEP-driven crew NPC.
#[derive(Component)]
pub struct CrewNpc {
    /// Active inference agent running the FEP perception-action cycle.
    pub fep: ActiveInferenceAgent,
    /// Name for UI display.
    pub name: String,
    /// Caution level [0, 1] — modulated by LearningRateAdjust.
    pub caution: f32,
}

impl CrewNpc {
    pub fn new(name: &str, seed: u64) -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 8,
            ..ActiveInferenceAgentConfig::default()
        };
        Self {
            fep: ActiveInferenceAgent::new(config),
            name: name.to_string(),
            caution: 0.5,
        }
    }
}

/// Target position for NPC movement.
#[derive(Component, Default)]
pub struct MoveTarget {
    pub target: Option<Vec2>,
    pub speed: f32,
}

/// Player's flashlight.
#[derive(Component)]
pub struct Flashlight {
    /// Base radius in pixels.
    pub base_radius: f32,
    /// Current flicker amount [0, 1] — driven by stress.
    pub flicker: f32,
}

impl Default for Flashlight {
    fn default() -> Self {
        Self {
            base_radius: 150.0,
            flicker: 0.0,
        }
    }
}

/// Noise source component — anything that makes sound alerts the Leviathan.
#[derive(Component, Default)]
pub struct NoiseEmitter {
    /// Current noise level [0, 1].
    pub level: f32,
}

/// Tile map marker.
#[derive(Component)]
pub struct Tile {
    pub grid_x: i32,
    pub grid_y: i32,
    pub walkable: bool,
}

/// Fusion core — the extraction objective.
#[derive(Component)]
pub struct FusionCore {
    /// Whether the player is currently extracting.
    pub being_extracted: bool,
    /// Extraction progress [0, 1].
    pub extraction_progress: f32,
}
