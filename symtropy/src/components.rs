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

// ============================================================================
// Governance / Economy / Faction components (Phase A0)
// ============================================================================

/// Consciousness profile for an agent (player or NPC).
/// Maps to `mycelix-multiworld-sim::agent::ConsciousnessState`.
#[derive(Component, Debug, Clone)]
pub struct ConsciousnessProfile {
    /// Composite Phi score [0, 1].
    pub phi: f64,
    /// Consciousness tier: 0=Observer, 1=Participant, 2=Contributor, 3=Steward, 4=Guardian.
    pub tier: u8,
    /// 6D consciousness dimensions: [level, meta_awareness, coherence, care, harmony, epistemic].
    pub dimensions: [f64; 6],
}

impl Default for ConsciousnessProfile {
    fn default() -> Self {
        Self {
            phi: 0.5,
            tier: 2, // Contributor by default
            dimensions: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    }
}

impl ConsciousnessProfile {
    /// Compute Phi from dimensions (same weights as multiworld-sim).
    pub fn compute_phi(&mut self) {
        self.phi = 0.25 * self.dimensions[0]  // level
            + 0.20 * self.dimensions[1]       // meta_awareness
            + 0.15 * self.dimensions[2]       // coherence
            + 0.15 * self.dimensions[3]       // care_activation
            + 0.15 * self.dimensions[4]       // harmonic_alignment
            + 0.10 * self.dimensions[5];      // epistemic_confidence
        self.tier = match self.phi {
            p if p >= 0.8 => 4,
            p if p >= 0.6 => 3,
            p if p >= 0.4 => 2,
            p if p >= 0.2 => 1,
            _ => 0,
        };
    }
}

/// TEND mutual credit balance for an agent.
#[derive(Component, Debug, Clone, Default)]
pub struct TendBalance {
    /// Current balance (can be negative — mutual credit).
    pub balance: i64,
    /// Credit limit (how far negative they can go).
    pub credit_limit: i64,
}

impl TendBalance {
    pub fn new(credit_limit: i64) -> Self {
        Self {
            balance: 0,
            credit_limit,
        }
    }

    /// Attempt a transfer. Returns false if it would exceed credit limit.
    pub fn can_spend(&self, amount: i64) -> bool {
        (self.balance - amount) >= -self.credit_limit
    }
}

/// Faction affiliation for an agent.
#[derive(Component, Debug, Clone, Default)]
pub struct FactionAffiliation {
    /// Faction ID (None = unaffiliated).
    pub faction_id: Option<u32>,
    /// 4D ideology vector: [economic, authority, tradition, individual].
    pub ideology: [f64; 4],
}

/// NPC trust toward the player [0, 1]. Decays on coercion, grows on TEND exchanges.
#[derive(Component, Debug, Clone)]
pub struct NpcTrust {
    pub trust: f64,
}

impl Default for NpcTrust {
    fn default() -> Self {
        Self { trust: 0.6 }
    }
}

/// Epistemic tag on a scavenged item.
/// E0=unverified, E1=single-witness, E2=tested, E3=replicated, E4=consensus.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicTag(pub u8);

impl EpistemicTag {
    pub fn label(&self) -> &'static str {
        match self.0 {
            0 => "E0:Unverified",
            1 => "E1:Preliminary",
            2 => "E2:Tested",
            3 => "E3:Replicated",
            _ => "E4:Consensus",
        }
    }

    /// Degrade by one level (coercion penalty). Floors at E0.
    pub fn degrade(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }
}
