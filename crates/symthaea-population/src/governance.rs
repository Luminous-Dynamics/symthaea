// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-compatible governance types (no HDK dependency).
//!
//! Population decisions map to governance tiers with escalating approval thresholds.
//! This module provides the governance framework without depending on Holochain,
//! making it usable in both on-chain and off-chain contexts.

use serde::{Deserialize, Serialize};

use crate::types::{BreedingStrategy, GeneticRescuePlan, MatingPair};

/// Governance tier determining the approval threshold for decisions.
///
/// Tiers escalate from simple majority to near-unanimous consensus,
/// reflecting the gravity and irreversibility of population decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GovernanceTier {
    /// Simple majority (51%). Routine decisions.
    Citizen,
    /// Supermajority (67%). Significant policy changes.
    Steward,
    /// High consensus (75%). Conservation-critical decisions.
    Guardian,
    /// Near-unanimous (80%). Constitutional-level changes.
    Constitutional,
    /// Extraordinary consensus (90%). Emergency overrides.
    Override,
}

impl GovernanceTier {
    /// Get the approval fraction threshold for this tier.
    pub fn threshold(&self) -> f64 {
        match self {
            GovernanceTier::Citizen => 0.51,
            GovernanceTier::Steward => 0.67,
            GovernanceTier::Guardian => 0.75,
            GovernanceTier::Constitutional => 0.80,
            GovernanceTier::Override => 0.90,
        }
    }

    /// Human-readable name for this tier.
    pub fn name(&self) -> &'static str {
        match self {
            GovernanceTier::Citizen => "Citizen",
            GovernanceTier::Steward => "Steward",
            GovernanceTier::Guardian => "Guardian",
            GovernanceTier::Constitutional => "Constitutional",
            GovernanceTier::Override => "Override",
        }
    }

    /// Description of what this tier governs.
    pub fn description(&self) -> &'static str {
        match self {
            GovernanceTier::Citizen => "Routine population management decisions",
            GovernanceTier::Steward => "Significant breeding policy changes",
            GovernanceTier::Guardian => "Conservation-critical genetic decisions",
            GovernanceTier::Constitutional => "Fundamental governance changes",
            GovernanceTier::Override => "Emergency override of AI recommendations",
        }
    }

    /// All governance tiers in ascending order.
    pub fn all() -> [GovernanceTier; 5] {
        [
            GovernanceTier::Citizen,
            GovernanceTier::Steward,
            GovernanceTier::Guardian,
            GovernanceTier::Constitutional,
            GovernanceTier::Override,
        ]
    }
}

/// A population governance decision requiring community approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PopulationDecision {
    /// Approve a specific mating pair.
    ApprovePairing(MatingPair),
    /// Set a growth target (population size by a target generation).
    SetGrowthTarget {
        /// Target population size.
        size: usize,
        /// Target generation.
        generation: u32,
    },
    /// Approve a genetic rescue plan (introducing migrants).
    ApproveGeneticRescue(GeneticRescuePlan),
    /// Modify the breeding strategy.
    ModifyStrategy(BreedingStrategy),
    /// Override an AI recommendation with human reasoning.
    OverrideAiRecommendation(String),
}

impl PopulationDecision {
    /// Get the governance tier required for this decision.
    pub fn required_tier(&self) -> GovernanceTier {
        match self {
            PopulationDecision::ApprovePairing(_) => GovernanceTier::Citizen,
            PopulationDecision::SetGrowthTarget { .. } => GovernanceTier::Steward,
            PopulationDecision::ApproveGeneticRescue(_) => GovernanceTier::Guardian,
            PopulationDecision::ModifyStrategy(_) => GovernanceTier::Constitutional,
            PopulationDecision::OverrideAiRecommendation(_) => GovernanceTier::Override,
        }
    }

    /// Human-readable description of this decision.
    pub fn description(&self) -> String {
        match self {
            PopulationDecision::ApprovePairing(pair) => {
                format!(
                    "Approve pairing of individual {} with {}",
                    pair.parent_a, pair.parent_b
                )
            }
            PopulationDecision::SetGrowthTarget { size, generation } => {
                format!(
                    "Set growth target: {} individuals by generation {}",
                    size, generation
                )
            }
            PopulationDecision::ApproveGeneticRescue(plan) => {
                format!(
                    "Approve genetic rescue: {} migrants from {} at generation {}",
                    plan.num_migrants, plan.source_population, plan.target_generation
                )
            }
            PopulationDecision::ModifyStrategy(strategy) => {
                format!("Modify breeding strategy to {:?}", strategy)
            }
            PopulationDecision::OverrideAiRecommendation(reason) => {
                format!("Override AI recommendation: {}", reason)
            }
        }
    }
}

/// Check whether a decision passes given the approval fraction.
///
/// Returns true if approval_fraction >= the decision's required tier threshold.
pub fn decision_passes(decision: &PopulationDecision, approval_fraction: f64) -> bool {
    approval_fraction >= decision.required_tier().threshold()
}

/// A vote record for governance decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRecord {
    /// The decision being voted on.
    pub decision: PopulationDecision,
    /// Number of votes in favor.
    pub votes_for: u64,
    /// Number of votes against.
    pub votes_against: u64,
    /// Number of abstentions.
    pub abstentions: u64,
}

impl VoteRecord {
    /// Create a new vote record.
    pub fn new(decision: PopulationDecision) -> Self {
        Self {
            decision,
            votes_for: 0,
            votes_against: 0,
            abstentions: 0,
        }
    }

    /// Total votes cast (excluding abstentions).
    pub fn total_votes(&self) -> u64 {
        self.votes_for + self.votes_against
    }

    /// Approval fraction (votes_for / total_votes).
    pub fn approval_fraction(&self) -> f64 {
        let total = self.total_votes();
        if total == 0 {
            return 0.0;
        }
        self.votes_for as f64 / total as f64
    }

    /// Whether the decision passes.
    pub fn passes(&self) -> bool {
        decision_passes(&self.decision, self.approval_fraction())
    }

    /// Add a vote.
    pub fn add_vote(&mut self, in_favor: Option<bool>) {
        match in_favor {
            Some(true) => self.votes_for += 1,
            Some(false) => self.votes_against += 1,
            None => self.abstentions += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pairing_decision() -> PopulationDecision {
        PopulationDecision::ApprovePairing(MatingPair {
            parent_a: 1,
            parent_b: 2,
            kinship: 0.05,
            hla_complementarity: 0.8,
        })
    }

    fn make_rescue_decision() -> PopulationDecision {
        PopulationDecision::ApproveGeneticRescue(GeneticRescuePlan {
            source_population: "zoo_pop".to_string(),
            num_migrants: 3,
            target_generation: 5,
            expected_heterozygosity_gain: 0.05,
        })
    }

    #[test]
    fn test_governance_tier_thresholds() {
        assert!((GovernanceTier::Citizen.threshold() - 0.51).abs() < 1e-10);
        assert!((GovernanceTier::Steward.threshold() - 0.67).abs() < 1e-10);
        assert!((GovernanceTier::Guardian.threshold() - 0.75).abs() < 1e-10);
        assert!((GovernanceTier::Constitutional.threshold() - 0.80).abs() < 1e-10);
        assert!((GovernanceTier::Override.threshold() - 0.90).abs() < 1e-10);
    }

    #[test]
    fn test_governance_tiers_ascending() {
        let tiers = GovernanceTier::all();
        for i in 1..tiers.len() {
            assert!(
                tiers[i].threshold() > tiers[i - 1].threshold(),
                "tiers should have ascending thresholds"
            );
        }
    }

    #[test]
    fn test_governance_tier_ordering() {
        assert!(GovernanceTier::Citizen < GovernanceTier::Steward);
        assert!(GovernanceTier::Steward < GovernanceTier::Guardian);
        assert!(GovernanceTier::Guardian < GovernanceTier::Constitutional);
        assert!(GovernanceTier::Constitutional < GovernanceTier::Override);
    }

    #[test]
    fn test_citizen_decision_passes_at_51() {
        let decision = make_pairing_decision();
        assert!(decision_passes(&decision, 0.51));
        assert!(decision_passes(&decision, 0.99));
        assert!(!decision_passes(&decision, 0.50));
    }

    #[test]
    fn test_guardian_decision_passes_at_75() {
        let decision = make_rescue_decision();
        assert!(decision_passes(&decision, 0.75));
        assert!(decision_passes(&decision, 0.80));
        assert!(!decision_passes(&decision, 0.74));
    }

    #[test]
    fn test_override_decision_passes_at_90() {
        let decision = PopulationDecision::OverrideAiRecommendation("emergency".to_string());
        assert!(decision_passes(&decision, 0.90));
        assert!(decision_passes(&decision, 1.0));
        assert!(!decision_passes(&decision, 0.89));
    }

    #[test]
    fn test_steward_decision() {
        let decision = PopulationDecision::SetGrowthTarget {
            size: 200,
            generation: 10,
        };
        assert_eq!(decision.required_tier(), GovernanceTier::Steward);
        assert!(decision_passes(&decision, 0.67));
        assert!(!decision_passes(&decision, 0.66));
    }

    #[test]
    fn test_constitutional_decision() {
        let decision = PopulationDecision::ModifyStrategy(BreedingStrategy::MinimumKinship);
        assert_eq!(decision.required_tier(), GovernanceTier::Constitutional);
        assert!(decision_passes(&decision, 0.80));
        assert!(!decision_passes(&decision, 0.79));
    }

    #[test]
    fn test_decision_required_tiers() {
        assert_eq!(
            make_pairing_decision().required_tier(),
            GovernanceTier::Citizen
        );
        assert_eq!(
            PopulationDecision::SetGrowthTarget {
                size: 100,
                generation: 5
            }
            .required_tier(),
            GovernanceTier::Steward
        );
        assert_eq!(
            make_rescue_decision().required_tier(),
            GovernanceTier::Guardian
        );
        assert_eq!(
            PopulationDecision::ModifyStrategy(BreedingStrategy::Random).required_tier(),
            GovernanceTier::Constitutional
        );
        assert_eq!(
            PopulationDecision::OverrideAiRecommendation("".to_string()).required_tier(),
            GovernanceTier::Override
        );
    }

    #[test]
    fn test_decision_description() {
        let desc = make_pairing_decision().description();
        assert!(desc.contains("1"), "should mention parent IDs");
        assert!(desc.contains("2"), "should mention parent IDs");
    }

    #[test]
    fn test_vote_record_creation() {
        let vote = VoteRecord::new(make_pairing_decision());
        assert_eq!(vote.votes_for, 0);
        assert_eq!(vote.votes_against, 0);
        assert_eq!(vote.abstentions, 0);
        assert_eq!(vote.total_votes(), 0);
    }

    #[test]
    fn test_vote_record_approval() {
        let mut vote = VoteRecord::new(make_pairing_decision());
        vote.add_vote(Some(true));
        vote.add_vote(Some(true));
        vote.add_vote(Some(false));
        assert_eq!(vote.total_votes(), 3);
        assert!((vote.approval_fraction() - 2.0 / 3.0).abs() < 1e-10);
        assert!(vote.passes(), "2/3 > 0.51");
    }

    #[test]
    fn test_vote_record_abstentions_not_counted() {
        let mut vote = VoteRecord::new(make_pairing_decision());
        vote.add_vote(Some(true));
        vote.add_vote(None); // abstention
        vote.add_vote(None); // abstention
        assert_eq!(vote.total_votes(), 1);
        assert!((vote.approval_fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vote_record_guardian_fails() {
        let mut vote = VoteRecord::new(make_rescue_decision());
        // 7 for, 3 against = 70% < 75%
        for _ in 0..7 {
            vote.add_vote(Some(true));
        }
        for _ in 0..3 {
            vote.add_vote(Some(false));
        }
        assert!(!vote.passes(), "70% < 75% guardian threshold");
    }

    #[test]
    fn test_vote_record_guardian_passes() {
        let mut vote = VoteRecord::new(make_rescue_decision());
        // 8 for, 2 against = 80% > 75%
        for _ in 0..8 {
            vote.add_vote(Some(true));
        }
        for _ in 0..2 {
            vote.add_vote(Some(false));
        }
        assert!(vote.passes(), "80% > 75% guardian threshold");
    }

    #[test]
    fn test_vote_record_empty_does_not_pass() {
        let vote = VoteRecord::new(make_pairing_decision());
        assert!(!vote.passes(), "no votes should not pass");
    }

    #[test]
    fn test_governance_tier_names() {
        for tier in GovernanceTier::all() {
            assert!(!tier.name().is_empty());
            assert!(!tier.description().is_empty());
        }
    }

    #[test]
    fn test_governance_tier_serialization() {
        for tier in GovernanceTier::all() {
            let json = serde_json::to_string(&tier).unwrap();
            let back: GovernanceTier = serde_json::from_str(&json).unwrap();
            assert_eq!(tier, back);
        }
    }
}
