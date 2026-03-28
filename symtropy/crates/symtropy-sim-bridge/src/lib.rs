// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between `mycelix-multiworld-sim` governance/economy models and Bevy ECS.
//!
//! Wraps the pure-Rust simulation types as Bevy `Resource`s so the game can run
//! governance ticks, economy ticks, and faction dynamics without any Holochain
//! dependency. The multiworld-sim models become the "physics engine" for the
//! game's social dynamics.

use bevy::prelude::*;
use mycelix_multiworld_sim::{
    config::PolicyConfig,
    economy::WorldEconomy,
    events::CivEvent,
    factions::{Faction, FactionConflict, FactionEngine},
    governance::{GovernanceAuthority, WorldGovernance},
    harmony::HarmonyTracker,
    knowledge::WorldKnowledge,
    stochastic::StochasticEngine,
    world::{CulturalProfile, ResourceStock, World, WorldResources},
};

// ============================================================================
// Bevy Resources wrapping multiworld-sim state
// ============================================================================

/// Governance state for the game's single "world" (the ship/station).
#[derive(Resource)]
pub struct GovernanceState {
    /// The core governance model from multiworld-sim.
    pub governance: WorldGovernance,
    /// Simulated world used as context for governance ticks.
    pub world: World,
    /// Random engine for stochastic governance decisions.
    pub rng: StochasticEngine,
    /// Policy configuration knobs.
    pub policy: PolicyConfig,
    /// Current simulation tick (incremented each governance update).
    pub sim_tick: u32,
    /// Events generated during the last tick.
    pub last_events: Vec<CivEvent>,
}

impl Default for GovernanceState {
    fn default() -> Self {
        Self {
            governance: WorldGovernance::new(),
            world: create_ship_world(),
            rng: StochasticEngine::new(42),
            policy: PolicyConfig::default(),
            sim_tick: 0,
            last_events: Vec::new(),
        }
    }
}

impl GovernanceState {
    /// Create with a specific seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StochasticEngine::new(seed),
            ..Default::default()
        }
    }

    /// Run one governance tick. Returns events generated.
    pub fn tick(&mut self) -> Vec<CivEvent> {
        self.sim_tick += 1;
        let events = self.governance.tick_governance(
            &self.world,
            self.sim_tick,
            &mut self.rng,
        );
        self.last_events = events.clone();
        events
    }

    /// Current oppression index (0.0 = fair, 1.0 = oppressive).
    pub fn oppression_index(&self) -> f64 {
        self.governance.oppression_index
    }

    /// Current governance stability (0.0 = unstable, 1.0 = stable).
    pub fn stability(&self) -> f64 {
        self.governance.stability_score
    }

    /// Current governance authority level.
    pub fn authority(&self) -> GovernanceAuthority {
        self.governance.authority_level
    }

    /// Whether emergency powers are active.
    pub fn emergency_active(&self) -> bool {
        self.governance.emergency_powers_active
    }

    /// Anti-tyranny stats: (vetoes, overrides, deterred).
    pub fn veto_stats(&self) -> (u32, u32, u32) {
        (
            self.governance.veto_count,
            self.governance.veto_override_count,
            self.governance.veto_deterred_count,
        )
    }
}

/// Economy state for the game world.
#[derive(Resource)]
pub struct EconomyState {
    /// The core economy model.
    pub economy: WorldEconomy,
}

impl Default for EconomyState {
    fn default() -> Self {
        Self {
            economy: WorldEconomy::new(),
        }
    }
}

impl EconomyState {
    /// Gini coefficient (0.0 = perfect equality, 1.0 = maximum inequality).
    pub fn gini(&self) -> f64 {
        self.economy.gini_coefficient
    }

    /// Total production this tick.
    pub fn total_production(&self) -> f64 {
        self.economy.total_production
    }

    /// Currency supply after demurrage.
    pub fn currency_supply(&self) -> f64 {
        self.economy.currency_supply
    }
}

/// Faction dynamics state.
#[derive(Resource)]
pub struct FactionState {
    /// The faction engine from multiworld-sim.
    pub engine: FactionEngine,
}

impl Default for FactionState {
    fn default() -> Self {
        Self {
            engine: FactionEngine::new(),
        }
    }
}

impl FactionState {
    /// All active factions.
    pub fn factions(&self) -> &[Faction] {
        &self.engine.factions
    }

    /// Active conflicts.
    pub fn active_conflicts(&self) -> Vec<&FactionConflict> {
        self.engine
            .conflicts
            .iter()
            .filter(|c| c.resolved_tick.is_none())
            .collect()
    }

    /// Civil unrest for a given world (default 0.0).
    pub fn civil_unrest(&self, world_id: u32) -> f64 {
        self.engine.civil_unrest.get(&world_id).copied().unwrap_or(0.0)
    }
}

/// An active governance proposal that players and NPCs can vote on.
#[derive(Resource, Default)]
pub struct ActiveProposal {
    /// Whether there is currently a proposal open for voting.
    pub active: bool,
    /// Proposal description.
    pub description: String,
    /// Proposal tier: 0=Basic, 1=Major, 2=Constitutional.
    pub tier: u8,
    /// Minimum Phi required to vote on this proposal.
    pub min_phi: f64,
    /// Votes cast: (agent_name, vote_weight, approve).
    pub votes: Vec<(String, f64, bool)>,
    /// Tick when proposal was submitted.
    pub submitted_tick: u32,
    /// Whether a Guardian has vetoed this proposal.
    pub vetoed: bool,
    /// Override votes (if vetoed): (agent_name, vote_weight, approve_override).
    pub override_votes: Vec<(String, f64, bool)>,
    /// Ticks remaining before proposal expires.
    pub ticks_remaining: u32,
}

impl ActiveProposal {
    /// Create a new proposal.
    pub fn new(description: &str, tier: u8, submitted_tick: u32) -> Self {
        let min_phi = match tier {
            0 => 0.3,
            1 => 0.4,
            _ => 0.6,
        };
        let ticks_remaining = match tier {
            0 => 5,  // Basic: 5 game-ticks (~25 seconds)
            1 => 10, // Major: 10 game-ticks (~50 seconds)
            _ => 20, // Constitutional: 20 game-ticks (~100 seconds)
        };
        Self {
            active: true,
            description: description.to_string(),
            tier,
            min_phi,
            votes: Vec::new(),
            submitted_tick,
            vetoed: false,
            override_votes: Vec::new(),
            ticks_remaining,
        }
    }

    /// Cast a vote. Returns false if agent's phi is too low.
    pub fn cast_vote(&mut self, name: &str, phi: f64, approve: bool) -> bool {
        if phi < self.min_phi {
            return false;
        }
        // Phi-weighted vote: weight = phi (higher consciousness = stronger vote)
        self.votes.push((name.to_string(), phi, approve));
        true
    }

    /// Veto the proposal (only Guardians with phi >= 0.8).
    pub fn veto(&mut self, _name: &str, phi: f64) -> bool {
        if phi < 0.8 {
            return false;
        }
        self.vetoed = true;
        self.ticks_remaining = 10; // 80% override window
        true
    }

    /// Cast an override vote against a veto. Requires 80% weighted approval.
    pub fn cast_override_vote(&mut self, name: &str, phi: f64, approve: bool) -> bool {
        if !self.vetoed || phi < self.min_phi {
            return false;
        }
        self.override_votes.push((name.to_string(), phi, approve));
        true
    }

    /// Compute weighted approval ratio [0.0, 1.0].
    pub fn approval_ratio(&self) -> f64 {
        if self.votes.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = self.votes.iter().map(|(_, w, _)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let approve_weight: f64 = self
            .votes
            .iter()
            .filter(|(_, _, a)| *a)
            .map(|(_, w, _)| w)
            .sum();
        approve_weight / total_weight
    }

    /// Whether the veto override has reached 80% threshold.
    pub fn override_succeeded(&self) -> bool {
        if !self.vetoed || self.override_votes.is_empty() {
            return false;
        }
        let total_weight: f64 = self.override_votes.iter().map(|(_, w, _)| w).sum();
        if total_weight == 0.0 {
            return false;
        }
        let approve_weight: f64 = self
            .override_votes
            .iter()
            .filter(|(_, _, a)| *a)
            .map(|(_, w, _)| w)
            .sum();
        (approve_weight / total_weight) >= 0.80
    }

    /// Whether the proposal passed (majority approval, not vetoed or override succeeded).
    pub fn passed(&self) -> bool {
        if !self.active {
            return false;
        }
        let min_approval = match self.tier {
            0 => 0.50,
            1 => 0.60,
            _ => 0.67,
        };
        let approved = self.approval_ratio() >= min_approval;
        if self.vetoed {
            approved && self.override_succeeded()
        } else {
            approved
        }
    }

    /// Tick the proposal countdown. Returns true if expired.
    pub fn tick(&mut self) -> bool {
        if self.ticks_remaining > 0 {
            self.ticks_remaining -= 1;
        }
        self.ticks_remaining == 0
    }

    /// Clear the proposal after resolution.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

// ============================================================================
// Sim tick schedule
// ============================================================================

/// Timer resource for controlling sim tick rate.
#[derive(Resource)]
pub struct SimTickTimer {
    pub timer: Timer,
}

impl Default for SimTickTimer {
    fn default() -> Self {
        // 1 sim tick every 5 seconds of game time
        Self {
            timer: Timer::from_seconds(5.0, TimerMode::Repeating),
        }
    }
}

/// Bevy system: advance the governance simulation by one tick.
pub fn governance_tick_system(
    time: Res<Time>,
    mut tick_timer: ResMut<SimTickTimer>,
    mut gov: ResMut<GovernanceState>,
) {
    tick_timer.timer.tick(time.delta());
    if tick_timer.timer.just_finished() {
        let events = gov.tick();
        if !events.is_empty() {
            eprintln!(
                "[sim] Governance tick {} — {} events, oppression={:.2}, stability={:.2}",
                gov.sim_tick,
                events.len(),
                gov.oppression_index(),
                gov.stability(),
            );
        }
    }
}

// ============================================================================
// Plugin
// ============================================================================

/// Bevy plugin that registers all sim-bridge resources and systems.
pub struct SimBridgePlugin;

impl Plugin for SimBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GovernanceState>()
            .init_resource::<EconomyState>()
            .init_resource::<FactionState>()
            .init_resource::<ActiveProposal>()
            .init_resource::<SimTickTimer>();
        // Note: governance_tick_system should be added by the main plugin
        // so it can be gated on GamePhase::Playing.
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Create a minimal World representing the game's ship/station.
/// This provides the context that `WorldGovernance::tick_governance` needs.
fn create_ship_world() -> World {
    let mut resources = WorldResources::new();
    resources.set(
        "energy",
        ResourceStock {
            current: 500.0,
            capacity: 1000.0,
            production_rate: 10.0,
            consumption_rate: 8.0,
            critical_threshold: 0.2,
        },
    );
    resources.set(
        "oxygen",
        ResourceStock {
            current: 800.0,
            capacity: 1000.0,
            production_rate: 12.0,
            consumption_rate: 10.0,
            critical_threshold: 0.15,
        },
    );
    resources.set(
        "food",
        ResourceStock {
            current: 300.0,
            capacity: 500.0,
            production_rate: 5.0,
            consumption_rate: 6.0,
            critical_threshold: 0.2,
        },
    );

    World {
        id: 0,
        name: "The First Room".into(),
        location: "Ship".into(),
        founded_tick: 0,
        parent_world_id: None,
        agents: Vec::new(),
        next_agent_id: 100,
        resources,
        culture: CulturalProfile {
            harmony_weights: [0.125; 8], // Equal emphasis on all Eight Harmonies
            individualism: 0.4,
            risk_tolerance: 0.6, // Crew are risk-takers
            xenophilia: 0.5,
            traditionalism: 0.3, // Forward-looking
        },
        infrastructure_level: 0.3,
        max_population: 20,
        habitable_area_m2: 200.0,
        founding_harmony_emphasis: [0.125; 8],
        epidemics: Vec::new(),
        knowledge: WorldKnowledge::default(),
        economy: WorldEconomy::new(),
        harmony: HarmonyTracker::default(),
        governance: WorldGovernance::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governance_state_default_is_stable() {
        let gov = GovernanceState::default();
        assert_eq!(gov.stability(), 1.0);
        assert_eq!(gov.oppression_index(), 0.0);
        assert!(!gov.emergency_active());
        assert_eq!(gov.authority(), GovernanceAuthority::MissionControl);
    }

    #[test]
    fn governance_tick_advances() {
        let mut gov = GovernanceState::default();
        assert_eq!(gov.sim_tick, 0);
        gov.tick();
        assert_eq!(gov.sim_tick, 1);
    }

    #[test]
    fn proposal_voting_respects_phi_threshold() {
        let mut prop = ActiveProposal::new("Open airlock", 0, 1);
        // Phi too low
        assert!(!prop.cast_vote("Low", 0.1, true));
        assert!(prop.votes.is_empty());
        // Phi sufficient
        assert!(prop.cast_vote("High", 0.5, true));
        assert_eq!(prop.votes.len(), 1);
    }

    #[test]
    fn proposal_approval_ratio() {
        let mut prop = ActiveProposal::new("Redistribute", 0, 1);
        prop.cast_vote("A", 0.6, true);
        prop.cast_vote("B", 0.4, false);
        // Weighted: 0.6 approve, 0.4 reject = 0.6 ratio
        assert!((prop.approval_ratio() - 0.6).abs() < 0.01);
    }

    #[test]
    fn veto_requires_guardian_phi() {
        let mut prop = ActiveProposal::new("Change constitution", 2, 1);
        assert!(!prop.veto("NotGuardian", 0.7));
        assert!(!prop.vetoed);
        assert!(prop.veto("Guardian", 0.85));
        assert!(prop.vetoed);
    }

    #[test]
    fn override_requires_80_percent() {
        let mut prop = ActiveProposal::new("Change rules", 0, 1);
        prop.cast_vote("A", 0.8, true);
        prop.veto("G", 0.85);
        // 3 approve override, 1 reject — 75% < 80%, fails
        prop.cast_override_vote("A", 0.8, true);
        prop.cast_override_vote("B", 0.7, true);
        prop.cast_override_vote("C", 0.6, true);
        prop.cast_override_vote("D", 0.5, false);
        // total=2.6, approve=2.1, ratio=0.808 — just over 80%
        assert!(prop.override_succeeded());
    }

    #[test]
    fn proposal_tier_thresholds() {
        let basic = ActiveProposal::new("Basic", 0, 1);
        assert!((basic.min_phi - 0.3).abs() < 0.01);

        let major = ActiveProposal::new("Major", 1, 1);
        assert!((major.min_phi - 0.4).abs() < 0.01);

        let constitutional = ActiveProposal::new("Constitutional", 2, 1);
        assert!((constitutional.min_phi - 0.6).abs() < 0.01);
    }

    #[test]
    fn economy_state_defaults() {
        let econ = EconomyState::default();
        assert_eq!(econ.gini(), 0.0);
    }

    #[test]
    fn faction_state_empty() {
        let factions = FactionState::default();
        assert!(factions.factions().is_empty());
        assert!(factions.active_conflicts().is_empty());
        assert_eq!(factions.civil_unrest(0), 0.0);
    }
}
