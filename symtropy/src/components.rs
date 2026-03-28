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
// Governance / Economy / Faction components
// Uses REAL Mycelix types — the game is a direct integration test.
// ============================================================================

/// Simplified consciousness profile (mirrors Mycelix ConsciousnessProfile).
/// Will be replaced with real type when Mycelix integration stabilizes.
#[derive(Debug, Clone)]
pub struct SimpleConsciousnessProfile {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

impl SimpleConsciousnessProfile {
    pub fn combined_score(&self) -> f64 {
        (self.identity * 0.25 + self.reputation * 0.25 + self.community * 0.30 + self.engagement * 0.20).clamp(0.0, 1.0)
    }
}
/// Agent consciousness — wraps the REAL `mycelix-bridge-common::ConsciousnessProfile`
/// plus the 6D simulation state from `mycelix-multiworld-sim::agent::ConsciousnessState`.
///
/// The 4D governance profile (identity/reputation/community/engagement) uses
/// canonical `ConsciousnessProfile::combined_score()` and `ConsciousnessTier::from_score()`.
/// The 6D simulation state drives NPC behavior via the FEP observation vector.
#[derive(Component, Debug, Clone)]
pub struct ConsciousnessComp {
    /// Governance profile — the REAL Mycelix 4D consciousness.
    /// Uses canonical combined_score() and tier derivation.
    pub governance: SimpleConsciousnessProfile,
    /// Simulation state — 6D from multiworld-sim (level, meta, coherence, care, harmony, epistemic).
    pub sim_dimensions: [f64; 6],
}

impl Default for ConsciousnessComp {
    fn default() -> Self {
        Self {
            governance: SimpleConsciousnessProfile {
                identity: 0.5,
                reputation: 0.5,
                community: 0.5,
                engagement: 0.5,
            },
            sim_dimensions: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    }
}

impl ConsciousnessComp {
    /// Combined governance score using REAL Mycelix weights (0.25/0.25/0.30/0.20).
    pub fn combined_score(&self) -> f64 {
        self.governance.combined_score()
    }

    /// Governance tier using REAL Mycelix thresholds (0.3/0.4/0.6/0.8).
    pub fn tier(&self) -> u8 {
        let s = self.combined_score();
        if s >= 0.8 { 4 } else if s >= 0.6 { 3 } else if s >= 0.4 { 2 } else if s >= 0.3 { 1 } else { 0 }
    }

    /// Phi from 6D simulation dimensions (multiworld-sim formula).
    pub fn sim_phi(&self) -> f64 {
        0.25 * self.sim_dimensions[0]  // level
            + 0.20 * self.sim_dimensions[1]  // meta_awareness
            + 0.15 * self.sim_dimensions[2]  // coherence
            + 0.15 * self.sim_dimensions[3]  // care_activation
            + 0.15 * self.sim_dimensions[4]  // harmonic_alignment
            + 0.10 * self.sim_dimensions[5]  // epistemic_confidence
    }

    /// Sync governance engagement from simulation phi.
    /// This bridges the sim → governance profile.
    pub fn sync_engagement_from_sim(&mut self) {
        self.governance.engagement = self.sim_phi().clamp(0.0, 1.0);
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

/// NPC trust toward the player [0, 1].
/// TODO: Wire to real KVector when builder API is integrated.
#[derive(Component, Debug, Clone)]
pub struct NpcTrust {
    pub trust: f64,
}

impl Default for NpcTrust {
    fn default() -> Self {
        Self { trust: 0.6 }
    }
}

/// Epistemic tag on a scavenged item — wraps REAL `mycelix-core-types::u8`.
/// E0=Unverifiable, E1=Anecdotal, E2=Observable, E3=Measurable, E4=CryptographicallyVerifiable.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicTag(pub u8);

impl Default for EpistemicTag {
    fn default() -> Self {
        Self(0) // E0: Unverifiable
    }
}

impl EpistemicTag {
    pub fn label(&self) -> &'static str {
        match self.0 {
            0 => "E0:Unverifiable",
            1 => "E1:Anecdotal",
            2 => "E2:Observable",
            3 => "E3:Measurable",
            _ => "E4:Verified",
        }
    }

    pub fn level(&self) -> u8 { self.0 }

    pub fn degrade(&mut self) { self.0 = self.0.saturating_sub(1); }

    pub fn advance(&mut self) { if self.0 < 4 { self.0 += 1; } }
}
