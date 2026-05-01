// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Proposal-Based Governance — Real Policy Decisions
//!
//! Factions author proposals that change real economic parameters.
//! Citizens vote weighted by governance tier and ideology alignment.
//! Winning proposals execute, changing the economy for all citizens.
//!
//! This closes the feedback loop: governance decisions → resource
//! allocation → citizen outcomes → governance preferences.
//!
//! ## Proposal Lifecycle
//!
//! 1. Factions generate proposals from ideology (1-2% probability/tick)
//! 2. All adult agents vote (weighted by tier, directed by ideology)
//! 3. Proposals pass if votes_for > votes_against AND quorum > 30%
//! 4. Guardian veto can block
//! 5. Passed proposals modify PolicyState
//! 6. Economy reads PolicyState next tick → different outcomes

use crate::agent::CivAgent;
use crate::config::{BirthPolicy, ResourcePriority};
use crate::factions::Faction;
use crate::stochastic::StochasticEngine;
use serde::{Deserialize, Serialize};

// ============================================================================
// Policy State — Mutable per-world policy parameters
// ============================================================================

/// Mutable policy parameters that governance can change via proposals.
///
/// Initialized from PolicyConfig, then mutated by passed proposals.
/// Each field corresponds to a real economic lever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyState {
    /// Trade openness [0.0, 1.0] — scales interworld trade volume.
    pub trade_openness: f64,
    /// Defense spending [0.0, 0.3] — fraction of labor diverted from production.
    pub defense_spending: f64,
    /// Exploration investment [0.0, 0.2] — accelerates technology discovery.
    pub exploration_investment: f64,
    /// Birth policy — population growth strategy.
    pub birth_policy: BirthPolicy,
    /// Resource priority — which sectors get investment priority.
    pub resource_priority: ResourcePriority,
    /// Care effectiveness [0.0, 1.0] — how much care workers reduce allostatic load.
    pub care_effectiveness: f64,
    /// Infrastructure investment rate per tick.
    pub infrastructure_rate: f64,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self {
            trade_openness: 0.5,
            defense_spending: 0.0,
            exploration_investment: 0.05,
            birth_policy: BirthPolicy::Natural,
            resource_priority: ResourcePriority::Balanced,
            care_effectiveness: 1.0,
            infrastructure_rate: 0.01,
        }
    }
}

impl PolicyState {
    /// Initialize from the simulation's PolicyConfig.
    pub fn from_config(config: &crate::config::PolicyConfig) -> Self {
        Self {
            trade_openness: config.trade_openness,
            defense_spending: config.defense_spending,
            exploration_investment: config.exploration_investment,
            birth_policy: config.birth_policy,
            resource_priority: config.resource_priority,
            care_effectiveness: config.care_effectiveness,
            infrastructure_rate: 0.01,
        }
    }
}

// ============================================================================
// Proposal Types
// ============================================================================

/// A single policy change proposed by a faction or agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyChange {
    TradeOpenness(f64),
    DefenseSpending(f64),
    ExplorationInvestment(f64),
    BirthPolicy(BirthPolicy),
    ResourcePriority(ResourcePriority),
    CareEffectiveness(f64),
    InfrastructureRate(f64),
}

impl PolicyChange {
    /// Apply this change to a PolicyState.
    pub fn apply(&self, state: &mut PolicyState) {
        match self {
            Self::TradeOpenness(v) => state.trade_openness = v.clamp(0.0, 1.0),
            Self::DefenseSpending(v) => state.defense_spending = v.clamp(0.0, 0.3),
            Self::ExplorationInvestment(v) => state.exploration_investment = v.clamp(0.0, 0.2),
            Self::BirthPolicy(p) => state.birth_policy = *p,
            Self::ResourcePriority(p) => state.resource_priority = *p,
            Self::CareEffectiveness(v) => state.care_effectiveness = v.clamp(0.0, 1.0),
            Self::InfrastructureRate(v) => state.infrastructure_rate = v.clamp(0.0, 0.05),
        }
    }
}

/// Status of a proposal in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Collecting votes this tick.
    Voting,
    /// Passed: votes_for > votes_against AND quorum met.
    Passed,
    /// Failed vote or quorum.
    Rejected,
    /// Policy applied to world.
    Executed,
    /// Guardian blocked.
    Vetoed,
}

/// A governance proposal to change one or more policy parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u32,
    pub author_faction_id: Option<u32>,
    pub proposed_at: u32,
    pub changes: Vec<PolicyChange>,
    pub votes_for: f64,
    pub votes_against: f64,
    pub votes_abstain: u32,
    pub status: ProposalStatus,
}

// ============================================================================
// Proposal Generation
// ============================================================================

/// Generate proposals from active factions based on ideology.
///
/// Each faction with strength > 0.1 has a 2% chance per tick of
/// authoring a proposal. The proposal direction maps from ideology.
pub fn generate_proposals(
    factions: &[Faction],
    current_policy: &PolicyState,
    current_tick: u32,
    next_id: &mut u32,
    rng: &mut StochasticEngine,
) -> Vec<Proposal> {
    let mut proposals = Vec::new();

    for faction in factions {
        if faction.strength < 0.1 {
            continue;
        }

        // 2% per tick per faction
        if !rng.bernoulli(0.05 * faction.strength) {
            continue;
        }

        let changes = ideology_to_changes(&faction.ideology, current_policy, rng);
        if changes.is_empty() {
            continue;
        }

        let proposal = Proposal {
            id: *next_id,
            author_faction_id: Some(faction.id),
            proposed_at: current_tick,
            changes,
            votes_for: 0.0,
            votes_against: 0.0,
            votes_abstain: 0,
            status: ProposalStatus::Voting,
        };

        *next_id += 1;
        proposals.push(proposal);
    }

    proposals
}

/// Map faction ideology to policy changes.
///
/// Ideology is a 4D vector: [economic, authoritarian, traditional, collective]
/// Each dimension influences different policy parameters.
fn ideology_to_changes(
    ideology: &[f64; 4],
    current: &PolicyState,
    rng: &mut StochasticEngine,
) -> Vec<PolicyChange> {
    let mut changes = Vec::new();

    // Economic dimension → trade openness
    // High economic (> 0.6) → increase trade; Low (< 0.4) → decrease
    if ideology[0] > 0.6 {
        let new_val = (current.trade_openness + 0.2).min(1.0);
        changes.push(PolicyChange::TradeOpenness(new_val));
    } else if ideology[0] < 0.4 {
        let new_val = (current.trade_openness - 0.2).max(0.0);
        changes.push(PolicyChange::TradeOpenness(new_val));
    }

    // Authoritarian dimension → defense spending
    if ideology[1] > 0.6 {
        let new_val = (current.defense_spending + 0.1).min(0.3);
        changes.push(PolicyChange::DefenseSpending(new_val));
    } else if ideology[1] < 0.3 {
        let new_val = (current.defense_spending - 0.1).max(0.0);
        changes.push(PolicyChange::DefenseSpending(new_val));
    }

    // Traditional dimension → birth policy
    if ideology[2] > 0.7 && current.birth_policy != BirthPolicy::ProNatal {
        changes.push(PolicyChange::BirthPolicy(BirthPolicy::ProNatal));
    } else if ideology[2] < 0.3 && current.birth_policy != BirthPolicy::PopulationControl {
        changes.push(PolicyChange::BirthPolicy(BirthPolicy::PopulationControl));
    }

    // Collective dimension → care effectiveness
    if ideology[3] > 0.6 {
        let new_val = (current.care_effectiveness + 0.2).min(1.0);
        changes.push(PolicyChange::CareEffectiveness(new_val));
    }

    // Random exploration investment proposal (10% chance)
    if rng.bernoulli(0.1) {
        let delta = if rng.bernoulli(0.5) { 0.02 } else { -0.02 };
        let new_val = (current.exploration_investment + delta).clamp(0.0, 0.2);
        changes.push(PolicyChange::ExplorationInvestment(new_val));
    }

    changes
}

// ============================================================================
// Voting
// ============================================================================

/// Tally votes on a proposal from all living adult agents.
///
/// Vote weight = governance tier weight (from mycel_score tier mapping).
/// Vote direction = ideology alignment with proposing faction.
/// Deliberation cost = +0.005 allostatic load per vote cast.
pub fn tally_votes(
    proposal: &mut Proposal,
    agents: &mut [CivAgent],
    factions: &[Faction],
    current_tick: u32,
    rng: &mut StochasticEngine,
) {
    let author_ideology = proposal
        .author_faction_id
        .and_then(|id| factions.iter().find(|f| f.id == id))
        .map(|f| f.ideology);

    for agent in agents.iter_mut().filter(|a| a.is_alive()) {
        // Must be adult
        let age = agent.age_years(current_tick);
        if age < 15.0 {
            continue;
        }

        // Low engagement → abstain
        if agent.needs.engagement < 0.2 {
            proposal.votes_abstain += 1;
            continue;
        }

        // Compute vote weight from MYCEL score (maps to tier → weight)
        let weight = mycel_to_weight(agent.mycel_score);

        // Compute vote direction from ideology alignment
        let direction = if let Some(author_ideo) = author_ideology {
            let agent_ideo = agent
                .faction_id
                .and_then(|id| factions.iter().find(|f| f.id == id))
                .map(|f| f.ideology)
                .unwrap_or([0.5; 4]);

            let distance = ideology_distance(&agent_ideo, &author_ideo);
            (1.0 - 2.0 * distance) * agent.coordination_understanding
        } else {
            // No faction authorship → random direction weighted by cu
            (rng.next_f64() - 0.5) * agent.coordination_understanding
        };

        if direction > 0.0 {
            proposal.votes_for += weight * direction;
        } else {
            proposal.votes_against += weight * direction.abs();
        }

        // Deliberation cost
        agent.needs.allostatic_load = (agent.needs.allostatic_load + 0.005).min(1.0);
    }
}

/// Resolve a proposal after voting.
///
/// Passes if votes_for > votes_against AND quorum met (30% of adults voted).
pub fn resolve_proposal(proposal: &mut Proposal, total_adults: usize) -> ProposalStatus {
    let total_votes = proposal.votes_for + proposal.votes_against;
    let quorum_fraction = total_votes / (total_adults.max(1) as f64);

    if quorum_fraction < 0.3 {
        proposal.status = ProposalStatus::Rejected;
    } else if proposal.votes_for > proposal.votes_against {
        proposal.status = ProposalStatus::Passed;
    } else {
        proposal.status = ProposalStatus::Rejected;
    }

    proposal.status
}

/// Execute a passed proposal, modifying the world's policy state.
pub fn execute_proposal(proposal: &Proposal, state: &mut PolicyState) {
    for change in &proposal.changes {
        change.apply(state);
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Map MYCEL score to voting weight [0.0, 1.0].
fn mycel_to_weight(mycel: f64) -> f64 {
    if mycel < 0.3 {
        0.0 // Observer — no vote
    } else if mycel < 0.4 {
        0.3 // Participant
    } else if mycel < 0.6 {
        0.6 // Citizen
    } else if mycel < 0.8 {
        0.85 // Steward
    } else {
        1.0 // Guardian
    }
}

/// Euclidean distance between two 4D ideology vectors, normalized to [0, 1].
fn ideology_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / 4.0).sqrt() // max distance = 1.0 when all dims differ by 1.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_change_applies_correctly() {
        let mut state = PolicyState::default();
        PolicyChange::TradeOpenness(0.8).apply(&mut state);
        assert_eq!(state.trade_openness, 0.8);

        PolicyChange::DefenseSpending(0.2).apply(&mut state);
        assert_eq!(state.defense_spending, 0.2);

        PolicyChange::BirthPolicy(BirthPolicy::ProNatal).apply(&mut state);
        assert_eq!(state.birth_policy, BirthPolicy::ProNatal);
    }

    #[test]
    fn policy_change_clamps_values() {
        let mut state = PolicyState::default();
        PolicyChange::TradeOpenness(1.5).apply(&mut state);
        assert_eq!(state.trade_openness, 1.0);

        PolicyChange::DefenseSpending(-0.1).apply(&mut state);
        assert_eq!(state.defense_spending, 0.0);
    }

    #[test]
    fn ideology_distance_zero_for_identical() {
        let a = [0.5, 0.5, 0.5, 0.5];
        assert_eq!(ideology_distance(&a, &a), 0.0);
    }

    #[test]
    fn ideology_distance_max_for_opposite() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        assert!((ideology_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mycel_to_weight_tiers() {
        assert_eq!(mycel_to_weight(0.1), 0.0); // Observer
        assert_eq!(mycel_to_weight(0.35), 0.3); // Participant
        assert_eq!(mycel_to_weight(0.5), 0.6); // Citizen
        assert_eq!(mycel_to_weight(0.7), 0.85); // Steward
        assert_eq!(mycel_to_weight(0.9), 1.0); // Guardian
    }

    #[test]
    fn resolve_proposal_quorum_failure() {
        let mut proposal = Proposal {
            id: 1,
            author_faction_id: None,
            proposed_at: 0,
            changes: vec![PolicyChange::TradeOpenness(0.8)],
            votes_for: 1.0,
            votes_against: 0.0,
            votes_abstain: 0,
            status: ProposalStatus::Voting,
        };
        // 1 vote out of 100 adults = 1% < 30% quorum
        let status = resolve_proposal(&mut proposal, 100);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn resolve_proposal_passes_with_quorum() {
        let mut proposal = Proposal {
            id: 1,
            author_faction_id: None,
            proposed_at: 0,
            changes: vec![PolicyChange::TradeOpenness(0.8)],
            votes_for: 20.0,
            votes_against: 15.0,
            votes_abstain: 0,
            status: ProposalStatus::Voting,
        };
        // 35 votes out of 100 = 35% > 30% quorum, for > against
        let status = resolve_proposal(&mut proposal, 100);
        assert_eq!(status, ProposalStatus::Passed);
    }

    #[test]
    fn resolve_proposal_rejected_by_vote() {
        let mut proposal = Proposal {
            id: 1,
            author_faction_id: None,
            proposed_at: 0,
            changes: vec![PolicyChange::TradeOpenness(0.8)],
            votes_for: 10.0,
            votes_against: 25.0,
            votes_abstain: 0,
            status: ProposalStatus::Voting,
        };
        let status = resolve_proposal(&mut proposal, 100);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn execute_proposal_changes_state() {
        let proposal = Proposal {
            id: 1,
            author_faction_id: None,
            proposed_at: 0,
            changes: vec![
                PolicyChange::TradeOpenness(0.9),
                PolicyChange::CareEffectiveness(0.8),
            ],
            votes_for: 30.0,
            votes_against: 10.0,
            votes_abstain: 0,
            status: ProposalStatus::Passed,
        };
        let mut state = PolicyState::default();
        execute_proposal(&proposal, &mut state);
        assert_eq!(state.trade_openness, 0.9);
        assert_eq!(state.care_effectiveness, 0.8);
    }

    #[test]
    fn default_policy_state_sensible() {
        let state = PolicyState::default();
        assert!(state.trade_openness > 0.0 && state.trade_openness <= 1.0);
        assert!(state.defense_spending >= 0.0 && state.defense_spending <= 0.3);
        assert!(state.care_effectiveness > 0.0);
    }
}
