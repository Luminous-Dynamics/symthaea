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

// Re-export real Mycelix types for use across game systems
pub use mycelix_bridge_common::{
    ConsciousnessProfile as MycelixConsciousnessProfile,
    ConsciousnessTier as MycelixTier,
    consciousness_thresholds::ConsciousnessThresholds,
};
pub use mycelix_core_types::epistemic::EmpiricalLevel;
// TODO: wire KVector when builder API integration completes
// pub use mycelix_core_types::k_vector::KVector;

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
    pub governance: MycelixConsciousnessProfile,
    /// Simulation state — 6D from multiworld-sim (level, meta, coherence, care, harmony, epistemic).
    pub sim_dimensions: [f64; 6],
}

impl Default for ConsciousnessComp {
    fn default() -> Self {
        Self {
            governance: MycelixConsciousnessProfile {
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
    pub fn tier(&self) -> MycelixTier {
        MycelixTier::from_score(self.combined_score())
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

/// Epistemic tag on a scavenged item — wraps REAL `mycelix-core-types::EmpiricalLevel`.
/// E0=Unverifiable, E1=Anecdotal, E2=Observable, E3=Measurable, E4=CryptographicallyVerifiable.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicTag(pub EmpiricalLevel);

impl Default for EpistemicTag {
    fn default() -> Self {
        Self(EmpiricalLevel::Unverifiable)
    }
}

impl EpistemicTag {
    pub fn label(&self) -> &'static str {
        match self.0 {
            EmpiricalLevel::Unverifiable => "E0:Unverifiable",
            EmpiricalLevel::Anecdotal => "E1:Anecdotal",
            EmpiricalLevel::Observable => "E2:Observable",
            EmpiricalLevel::Measurable => "E3:Measurable",
            EmpiricalLevel::CryptographicallyVerifiable => "E4:Verified",
        }
    }

    /// Numeric level (0-4) for flashlight radius computation.
    pub fn level(&self) -> u8 {
        self.0.value()
    }

    /// Degrade by one level (coercion penalty). Floors at E0.
    pub fn degrade(&mut self) {
        let current = self.0.value();
        if current > 0 {
            self.0 = EmpiricalLevel::from_value(current - 1)
                .unwrap_or(EmpiricalLevel::Unverifiable);
        }
    }

    /// Advance by one level (verification reward). Caps at E4.
    pub fn advance(&mut self) {
        let current = self.0.value();
        if current < 4 {
            if let Some(next) = EmpiricalLevel::from_value(current + 1) {
                self.0 = next;
            }
        }
    }
}
