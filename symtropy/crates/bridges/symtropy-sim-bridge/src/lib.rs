// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between `mycelix-multiworld-sim` governance/economy models and Bevy ECS.
//!
//! Wraps the pure-Rust simulation types as Bevy `Resource`s so the game can run
//! governance ticks, economy ticks, and faction dynamics without any Holochain
//! dependency. The multiworld-sim models become the "physics engine" for the
//! game's social dynamics.

use bevy::prelude::*;

mod mk0;
pub use mk0::*;

// Re-export Mycelix types for game systems to use via symtropy_sim_bridge::
pub use feldman_dkg::participant::ParticipantId;
pub use feldman_dkg::{CeremonyPhase, DkgCeremony, DkgConfig};
pub use mycelix_fl::fl_core::{
    AggregatedGradient as FlAggregatedGradient, AggregationMethod as FlAggregationMethod,
    GradientUpdate as FlGradientUpdate, aggregation::trimmed_mean as fl_trimmed_mean,
};

/// Veto override threshold — the fraction of weighted votes needed to override a Guardian veto.
///
/// The canonical governance zome (`mycelix-governance/execution/integrity`) uses 0.67
/// with adaptive decay (0.67→0.62→0.60 on repeated failures). The multiworld-sim
/// uses 0.80 for conservative 300-year modeling.
///
/// The game uses the governance zome value (0.67) because we're testing the real
/// architecture, not the conservative simulation. This ensures that the override
/// behavior players experience matches what they'd encounter in production Mycelix.
pub const VETO_OVERRIDE_THRESHOLD: f64 = 0.67;

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
        let events = self
            .governance
            .tick_governance(&self.world, self.sim_tick, &mut self.rng);
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
        self.engine
            .civil_unrest
            .get(&world_id)
            .copied()
            .unwrap_or(0.0)
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
        (approve_weight / total_weight) >= VETO_OVERRIDE_THRESHOLD
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
            ..CulturalProfile::earth_default()
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
        // Infrastructure and narrative
        power_generation_kw: 50.0,
        power_demand_kw: 40.0,
        narrative_identity: Default::default(),
        maintenance_hours_required: 10.0,
        maintenance_hours_available: 15.0,
        bus_factor_critical: 2,
        pathogen_pressure: 0.0,
        civilizational_phi: 0.5,
        trust_level: 0.7,
        earth_funding: 1.0,
        // Mortality parameters
        mortality_alpha_mult: 1.0,
        mortality_beta_mult: 1.0,
        mortality_lambda_mult: 1.0,
        // Ecosystem
        ecosystem_balance: 0.5,
        reproduction_viable: true,
        ..Default::default()
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
    fn override_requires_67_percent() {
        let mut prop = ActiveProposal::new("Change rules", 0, 1);
        prop.cast_vote("A", 0.8, true);
        prop.veto("G", 0.85);
        // 2 approve, 1 reject — weighted: 1.5/2.0 = 75% > 67%, succeeds
        prop.cast_override_vote("A", 0.8, true);
        prop.cast_override_vote("B", 0.7, true);
        prop.cast_override_vote("C", 0.5, false);
        // total=2.0, approve=1.5, ratio=0.75 > 0.67
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

    // ========================================================================
    // Integration boundary tests — validate real Mycelix types
    // ========================================================================

    #[test]
    fn consciousness_tier_boundaries_match_canonical() {
        use mycelix_bridge_common::{CivicTier, ConsciousnessProfile};

        // The canonical tier boundaries from mycelix-bridge-common
        assert_eq!(CivicTier::from_score(0.0), CivicTier::Observer);
        assert_eq!(CivicTier::from_score(0.29), CivicTier::Observer);
        assert_eq!(CivicTier::from_score(0.30), CivicTier::Participant);
        assert_eq!(CivicTier::from_score(0.39), CivicTier::Participant);
        assert_eq!(CivicTier::from_score(0.40), CivicTier::Citizen);
        assert_eq!(CivicTier::from_score(0.59), CivicTier::Citizen);
        assert_eq!(CivicTier::from_score(0.60), CivicTier::Steward);
        assert_eq!(CivicTier::from_score(0.79), CivicTier::Steward);
        assert_eq!(CivicTier::from_score(0.80), CivicTier::Guardian);
        assert_eq!(CivicTier::from_score(1.0), CivicTier::Guardian);
    }

    #[test]
    fn consciousness_profile_combined_score_weights() {
        use mycelix_bridge_common::ConsciousnessProfile;

        // Canonical weights: identity=0.25, reputation=0.25, community=0.30, engagement=0.20
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
        };
        let score = profile.combined_score();
        assert!(
            (score - 0.25).abs() < 0.01,
            "identity weight should be 0.25, got {}",
            score
        );

        let profile2 = ConsciousnessProfile {
            identity: 0.0,
            reputation: 0.0,
            community: 1.0,
            engagement: 0.0,
        };
        let score2 = profile2.combined_score();
        assert!(
            (score2 - 0.30).abs() < 0.01,
            "community weight should be 0.30, got {}",
            score2
        );
    }

    #[test]
    fn consciousness_thresholds_canonical_values() {
        use mycelix_bridge_common::consciousness_thresholds::ConsciousnessThresholds;

        let t = ConsciousnessThresholds::default();

        // Governance gates
        assert!((t.consciousness_gate_basic - 0.2).abs() < 0.01);
        assert!((t.consciousness_gate_proposal - 0.3).abs() < 0.01);
        assert!((t.consciousness_gate_voting - 0.4).abs() < 0.01);
        assert!((t.consciousness_gate_constitutional - 0.6).abs() < 0.01);
    }

    #[test]
    fn epistemic_level_transitions() {
        use mycelix_core_types::epistemic::EmpiricalLevel;

        // E0 → E1 → E2 → E3 → E4
        let e0 = EmpiricalLevel::Unverifiable;
        assert_eq!(e0.value(), 0);

        let e1 = EmpiricalLevel::from_value(1).unwrap();
        assert_eq!(e1, EmpiricalLevel::Anecdotal);

        let e4 = EmpiricalLevel::from_value(4).unwrap();
        assert_eq!(e4, EmpiricalLevel::CryptographicallyVerifiable);

        // Out of range returns None
        assert!(EmpiricalLevel::from_value(5).is_none());
    }

    #[test]
    fn real_fl_trimmed_mean_filters_byzantine() {
        use mycelix_fl::fl_core::{GradientUpdate, aggregation::trimmed_mean};

        // 4 honest nodes + 1 poisoned node
        let gradients = vec![
            GradientUpdate::new("honest_1".into(), 1, vec![1.0, 2.0, 3.0], 1, 0.0),
            GradientUpdate::new("honest_2".into(), 1, vec![1.1, 1.9, 3.1], 1, 0.0),
            GradientUpdate::new("honest_3".into(), 1, vec![0.9, 2.1, 2.9], 1, 0.0),
            GradientUpdate::new("honest_4".into(), 1, vec![1.0, 2.0, 3.0], 1, 0.0),
            GradientUpdate::new("byzantine".into(), 1, vec![100.0, -50.0, 999.0], 1, 0.0),
        ];

        let result = trimmed_mean(&gradients, 0.2).unwrap();

        // After trimming: the 3 middle values should average close to [1.0, 2.0, 3.0]
        assert!((result[0] - 1.0).abs() < 0.15, "dim 0: {}", result[0]);
        assert!((result[1] - 2.0).abs() < 0.15, "dim 1: {}", result[1]);
        assert!((result[2] - 3.0).abs() < 0.15, "dim 2: {}", result[2]);
    }

    #[test]
    fn veto_override_uses_governance_zome_threshold() {
        // The game uses 0.67 (governance zome), not 0.80 (multiworld-sim)
        assert!(
            (VETO_OVERRIDE_THRESHOLD - 0.67).abs() < f64::EPSILON,
            "Game should use governance zome threshold (0.67), got {}",
            VETO_OVERRIDE_THRESHOLD
        );

        let mut prop = ActiveProposal::new("Test", 0, 1);
        prop.cast_vote("A", 0.9, true);
        prop.veto("G", 0.85);

        // 66% should NOT override at 67% threshold
        prop.cast_override_vote("A", 0.66, true);
        prop.cast_override_vote("B", 0.34, false);
        assert!(
            !prop.override_succeeded(),
            "66% should not override at 67% threshold"
        );

        // 68% SHOULD override
        prop.override_votes.clear();
        prop.cast_override_vote("X", 0.68, true);
        prop.cast_override_vote("Y", 0.32, false);
        assert!(
            prop.override_succeeded(),
            "68% should override at 67% threshold"
        );
    }

    #[test]
    fn governance_anti_tyranny_invariants() {
        // These constants must match the multiworld-sim hardcoded values
        use mycelix_multiworld_sim::governance::WorldGovernance;

        let gov = WorldGovernance::new();
        assert_eq!(gov.stability_score, 1.0);
        assert_eq!(gov.oppression_index, 0.0);
        assert!(!gov.emergency_powers_active);
        assert_eq!(gov.consecutive_emergency_sessions, 0);
        assert_eq!(gov.veto_count, 0);
        assert_eq!(gov.veto_override_count, 0);
        assert_eq!(gov.veto_deterred_count, 0);
    }
}
