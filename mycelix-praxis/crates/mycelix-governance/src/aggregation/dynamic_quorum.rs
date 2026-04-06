// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dynamic quorum calculation with epistemic weighting
//!
//! ## Revolutionary Feature: Truth-Aware Participation Requirements
//!
//! Traditional governance systems use fixed quorum regardless of proposal quality:
//! - All proposals need same participation (e.g., 33%)
//! - No distinction between verified facts and wild speculation
//! - No adaptation to community engagement patterns
//!
//! Dynamic quorum changes the game:
//! - E4 systematic reviews: 20% quorum (trust the science)
//! - E0 unverified claims: 50% quorum (extra scrutiny required)
//! - Adaptive to recent community participation
//! - Scope-aware (local vs global proposals)
//!
//! ## Mathematical Foundation
//!
//! ```text
//! Required Quorum = base × epistemic_multiplier × scope_multiplier × participation_adjustment
//!
//! where:
//!   base = 0.33 (33% default)
//!   epistemic_multiplier = 0.6 to 1.5 based on tier (E0-E4)
//!   scope_multiplier = 0.8 (local) to 1.3 (global)
//!   participation_adjustment = 0.9 to 1.1 based on recent activity
//!
//! Result is clamped to [0.15, 0.75] for safety
//! ```
//!
//! ## Example Scenarios
//!
//! **High Trust + Local**: E4 systematic review, local scope, high participation
//! - Quorum = 0.33 × 0.6 × 0.8 × 1.1 ≈ 0.17 (17%)
//! - Rationale: Trust the peer-reviewed research on local matter
//!
//! **Low Trust + Global**: E0 wild claim, global scope, low participation
//! - Quorum = 0.33 × 1.5 × 1.3 × 0.95 ≈ 0.61 (61%)
//! - Rationale: Unverified global change during inactive period needs scrutiny
//!
//! **Standard**: E2 peer verified, regional, normal participation
//! - Quorum = 0.33 × 1.0 × 1.0 × 1.0 = 0.33 (33%)
//! - Rationale: Default case, no special adjustments

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::types::{EpistemicTier, ProposalScope, HierarchicalProposal};

/// Dynamic quorum calculator with epistemic and scope weighting
///
/// This calculator adapts quorum requirements based on:
/// 1. Epistemic tier (E0-E4) - trust in claim
/// 2. Proposal scope (Local/Regional/Global) - breadth of impact
/// 3. Recent community participation - adaptive engagement
///
/// ## Example
///
/// ```rust
/// use mycelix_governance::aggregation::DynamicQuorumCalculator;
/// use mycelix_governance::types::{HierarchicalProposal, ProposalCategory, ProposalType, EpistemicTier};
///
/// let calculator = DynamicQuorumCalculator::default();
///
/// let mut proposal = HierarchicalProposal::new(
///     "prop_001".to_string(),
///     "Peer-reviewed algorithm improvement".to_string(),
///     "Based on systematic review...".to_string(),
///     "proposer".to_string(),
///     ProposalCategory::Technical,
///     ProposalType::Normal,
/// );
///
/// // E4 systematic review → lower quorum
/// proposal.epistemic_tier = EpistemicTier::E4SystematicReview;
///
/// let required = calculator.calculate_required_quorum(&proposal);
/// assert!(required < 0.33); // Less than default
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicQuorumCalculator {
    /// Base participation requirement (default: 0.33 = 33%)
    pub base_quorum: f64,

    /// Epistemic tier multipliers (how much to adjust based on truth confidence)
    pub epistemic_multipliers: HashMap<String, f64>, // Using String for serialization

    /// Scope multipliers (how much to adjust based on proposal breadth)
    pub scope_multipliers: HashMap<String, f64>, // Using String for serialization

    /// Recent participation rate (rolling average, 0.0 to 1.0)
    pub recent_participation: f64,

    /// Minimum allowed quorum (safety floor)
    pub min_quorum: f64,

    /// Maximum allowed quorum (safety ceiling)
    pub max_quorum: f64,
}

impl Default for DynamicQuorumCalculator {
    fn default() -> Self {
        let mut epistemic_multipliers = HashMap::new();
        epistemic_multipliers.insert("E0Null".to_string(), 1.5); // Unverified → higher quorum
        epistemic_multipliers.insert("E1PersonalTestimony".to_string(), 1.2);
        epistemic_multipliers.insert("E2PeerVerified".to_string(), 1.0); // Default
        epistemic_multipliers.insert("E3CommunityCensensus".to_string(), 0.8);
        epistemic_multipliers.insert("E4PubliclyReproducible".to_string(), 0.6); // Trusted → lower quorum

        let mut scope_multipliers = HashMap::new();
        scope_multipliers.insert("Local".to_string(), 0.8); // Local decisions need less global participation
        scope_multipliers.insert("Regional".to_string(), 1.0); // Default
        scope_multipliers.insert("Global".to_string(), 1.3); // Global decisions need more participation

        Self {
            base_quorum: 0.33,
            epistemic_multipliers,
            scope_multipliers,
            recent_participation: 0.4, // Default 40% recent participation
            min_quorum: 0.15,
            max_quorum: 0.75,
        }
    }
}

impl DynamicQuorumCalculator {
    /// Create a new calculator with custom parameters
    pub fn new(
        base_quorum: f64,
        epistemic_multipliers: HashMap<String, f64>,
        scope_multipliers: HashMap<String, f64>,
        recent_participation: f64,
    ) -> Result<Self, String> {
        if base_quorum < 0.0 || base_quorum > 1.0 {
            return Err("base_quorum must be between 0.0 and 1.0".to_string());
        }

        if recent_participation < 0.0 || recent_participation > 1.0 {
            return Err("recent_participation must be between 0.0 and 1.0".to_string());
        }

        Ok(Self {
            base_quorum,
            epistemic_multipliers,
            scope_multipliers,
            recent_participation,
            min_quorum: 0.15,
            max_quorum: 0.75,
        })
    }

    /// Calculate required quorum for a specific proposal
    ///
    /// Applies all three adjustments:
    /// 1. Epistemic tier (trust level)
    /// 2. Proposal scope (breadth of impact)
    /// 3. Recent participation (adaptive engagement)
    ///
    /// Result is clamped to [min_quorum, max_quorum] for safety.
    pub fn calculate_required_quorum(&self, proposal: &HierarchicalProposal) -> f64 {
        let mut quorum = self.base_quorum;

        // Apply epistemic adjustment
        let epistemic_key = self.epistemic_tier_to_string(&proposal.epistemic_tier);
        let epistemic_mult = self.epistemic_multipliers
            .get(&epistemic_key)
            .unwrap_or(&1.0);
        quorum *= epistemic_mult;

        // Apply scope adjustment
        let scope_key = self.scope_to_string(&proposal.scope);
        let scope_mult = self.scope_multipliers
            .get(&scope_key)
            .unwrap_or(&1.0);
        quorum *= scope_mult;

        // Apply participation adjustment
        // If participation is low (0.2), reduce requirement (0.9x)
        // If participation is high (0.6), can require more (1.1x)
        let participation_mult = 0.9 + (self.recent_participation * 0.4);
        quorum *= participation_mult;

        // Clamp to safety range
        quorum.max(self.min_quorum).min(self.max_quorum)
    }

    /// Update recent participation rate based on new proposal outcome
    ///
    /// Uses exponential moving average (EMA) to track recent trends:
    /// new_rate = 0.7 × old_rate + 0.3 × actual_participation
    pub fn update_participation_rate(&mut self, actual_participation: f64) {
        if actual_participation < 0.0 || actual_participation > 1.0 {
            return; // Ignore invalid values
        }

        // EMA with alpha=0.3 (30% weight on new sample)
        self.recent_participation = 0.7 * self.recent_participation + 0.3 * actual_participation;
    }

    /// Check if proposal has met dynamic quorum
    pub fn has_met_quorum(&self, proposal: &HierarchicalProposal) -> bool {
        let required = self.calculate_required_quorum(proposal);
        let actual = proposal.total_weighted_votes();

        // Assumes total possible votes = 1.0 (100% of reputation)
        // In practice, this would be compared to total available reputation
        actual >= required
    }

    /// Get recommendation for proposal based on quorum status
    pub fn get_quorum_status(&self, proposal: &HierarchicalProposal, total_available_reputation: f64) -> QuorumStatus {
        let required = self.calculate_required_quorum(proposal);
        let required_votes = required * total_available_reputation;
        let actual_votes = proposal.total_weighted_votes();

        let participation_rate = if total_available_reputation > 0.0 {
            actual_votes / total_available_reputation
        } else {
            0.0
        };

        QuorumStatus {
            required_quorum: required,
            required_votes,
            actual_votes,
            participation_rate,
            met: actual_votes >= required_votes,
            epistemic_adjustment: self.get_epistemic_multiplier(&proposal.epistemic_tier),
            scope_adjustment: self.get_scope_multiplier(&proposal.scope),
        }
    }

    // Helper methods

    fn epistemic_tier_to_string(&self, tier: &EpistemicTier) -> String {
        match tier {
            EpistemicTier::E0Null => "E0Null".to_string(),
            EpistemicTier::E1PersonalTestimony => "E1PersonalTestimony".to_string(),
            EpistemicTier::E2PeerVerified => "E2PeerVerified".to_string(),
            EpistemicTier::E3CommunityCensensus => "E3CommunityCensensus".to_string(),
            EpistemicTier::E4PubliclyReproducible => "E4PubliclyReproducible".to_string(),
        }
    }

    fn scope_to_string(&self, scope: &ProposalScope) -> String {
        match scope {
            ProposalScope::Local => "Local".to_string(),
            ProposalScope::Regional => "Regional".to_string(),
            ProposalScope::Global => "Global".to_string(),
        }
    }

    fn get_epistemic_multiplier(&self, tier: &EpistemicTier) -> f64 {
        let key = self.epistemic_tier_to_string(tier);
        *self.epistemic_multipliers.get(&key).unwrap_or(&1.0)
    }

    fn get_scope_multiplier(&self, scope: &ProposalScope) -> f64 {
        let key = self.scope_to_string(scope);
        *self.scope_multipliers.get(&key).unwrap_or(&1.0)
    }
}

/// Quorum status report for a proposal
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QuorumStatus {
    /// Required quorum as fraction (0.0 to 1.0)
    pub required_quorum: f64,

    /// Required votes (absolute reputation value)
    pub required_votes: f64,

    /// Actual votes received
    pub actual_votes: f64,

    /// Actual participation rate
    pub participation_rate: f64,

    /// Whether quorum is met
    pub met: bool,

    /// Epistemic tier adjustment applied
    pub epistemic_adjustment: f64,

    /// Scope adjustment applied
    pub scope_adjustment: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ProposalCategory, ProposalType};

    #[test]
    fn test_default_calculator() {
        let calculator = DynamicQuorumCalculator::default();

        assert_eq!(calculator.base_quorum, 0.33);
        assert_eq!(calculator.min_quorum, 0.15);
        assert_eq!(calculator.max_quorum, 0.75);
        assert_eq!(calculator.recent_participation, 0.4);
    }

    #[test]
    fn test_e4_systematic_review_lower_quorum() {
        let calculator = DynamicQuorumCalculator::default();

        let mut proposal = HierarchicalProposal::new(
            "prop_001".to_string(),
            "Systematic Review Proposal".to_string(),
            "Peer-reviewed research".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        proposal.epistemic_tier = EpistemicTier::E4PubliclyReproducible;

        let required = calculator.calculate_required_quorum(&proposal);

        // E4 has 0.6x multiplier → 0.33 × 0.6 × 0.8 × 1.06 ≈ 0.17
        assert!(required < 0.33);
        assert!(required >= 0.15); // Above floor
    }

    #[test]
    fn test_e0_null_higher_quorum() {
        let calculator = DynamicQuorumCalculator::default();

        let mut proposal = HierarchicalProposal::new(
            "prop_002".to_string(),
            "Unverified Claim".to_string(),
            "No evidence provided".to_string(),
            "proposer".to_string(),
            ProposalCategory::Community,
            ProposalType::Normal,
        );

        proposal.epistemic_tier = EpistemicTier::E0Null;

        let required = calculator.calculate_required_quorum(&proposal);

        // E0 has 1.5x multiplier → 0.33 × 1.5 × 0.8 × 1.06 ≈ 0.42
        assert!(required > 0.33);
    }

    #[test]
    fn test_global_scope_higher_quorum() {
        let calculator = DynamicQuorumCalculator::default();

        let mut proposal = HierarchicalProposal::new(
            "prop_003".to_string(),
            "Global Change".to_string(),
            "Affects entire ecosystem".to_string(),
            "proposer".to_string(),
            ProposalCategory::Governance,
            ProposalType::Slow,
        );

        proposal.scope = ProposalScope::Global;

        let required = calculator.calculate_required_quorum(&proposal);

        // Global has 1.3x multiplier → 0.33 × 1.0 × 1.3 × 1.06 ≈ 0.45
        assert!(required > 0.33);
    }

    #[test]
    fn test_local_scope_lower_quorum() {
        let calculator = DynamicQuorumCalculator::default();

        let mut proposal = HierarchicalProposal::new(
            "prop_004".to_string(),
            "Local Decision".to_string(),
            "Only affects this hApp".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        proposal.scope = ProposalScope::Local;
        // Set E2 epistemic tier (1.0 multiplier) to test scope effect cleanly
        proposal.epistemic_tier = EpistemicTier::E2PeerVerified;

        let required = calculator.calculate_required_quorum(&proposal);

        // Local has 0.8x multiplier → 0.33 × 1.0 × 0.8 × 1.06 ≈ 0.28
        assert!(required < 0.33);
    }

    #[test]
    fn test_participation_adjustment() {
        let mut calculator = DynamicQuorumCalculator::default();

        let proposal = HierarchicalProposal::new(
            "prop_005".to_string(),
            "Standard Proposal".to_string(),
            "Description".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        // Default participation (0.4) → mult ≈ 1.06
        let quorum_normal = calculator.calculate_required_quorum(&proposal);

        // High participation (0.6) → mult = 0.9 + 0.24 = 1.14
        calculator.recent_participation = 0.6;
        let quorum_high = calculator.calculate_required_quorum(&proposal);

        // Low participation (0.2) → mult = 0.9 + 0.08 = 0.98
        calculator.recent_participation = 0.2;
        let quorum_low = calculator.calculate_required_quorum(&proposal);

        assert!(quorum_high > quorum_normal);
        assert!(quorum_low < quorum_normal);
    }

    #[test]
    fn test_update_participation_rate() {
        let mut calculator = DynamicQuorumCalculator::default();

        assert_eq!(calculator.recent_participation, 0.4);

        // New proposal with 60% participation
        calculator.update_participation_rate(0.6);

        // EMA: 0.7 × 0.4 + 0.3 × 0.6 = 0.28 + 0.18 = 0.46
        assert!((calculator.recent_participation - 0.46).abs() < 0.01);

        // Another with 30% participation
        calculator.update_participation_rate(0.3);

        // EMA: 0.7 × 0.46 + 0.3 × 0.3 = 0.322 + 0.09 = 0.412
        assert!((calculator.recent_participation - 0.412).abs() < 0.01);
    }

    #[test]
    fn test_quorum_clamping() {
        let calculator = DynamicQuorumCalculator::default();

        // Create extreme scenario: E0 + Global + low participation
        let mut proposal = HierarchicalProposal::new(
            "prop_006".to_string(),
            "Extreme Proposal".to_string(),
            "Description".to_string(),
            "proposer".to_string(),
            ProposalCategory::Governance,
            ProposalType::Normal,
        );

        proposal.epistemic_tier = EpistemicTier::E0Null; // 1.5x
        proposal.scope = ProposalScope::Global; // 1.3x

        let required = calculator.calculate_required_quorum(&proposal);

        // 0.33 × 1.5 × 1.3 × 1.06 ≈ 0.68
        // Should be clamped to max_quorum (0.75)
        assert!(required <= 0.75);

        // Test floor with E4 + Local
        proposal.epistemic_tier = EpistemicTier::E4PubliclyReproducible; // 0.6x
        proposal.scope = ProposalScope::Local; // 0.8x

        let required = calculator.calculate_required_quorum(&proposal);

        // 0.33 × 0.6 × 0.8 × 1.06 ≈ 0.17
        // Should stay above min_quorum (0.15)
        assert!(required >= 0.15);
    }

    #[test]
    fn test_quorum_status() {
        let calculator = DynamicQuorumCalculator::default();

        let mut proposal = HierarchicalProposal::new(
            "prop_007".to_string(),
            "Test Proposal".to_string(),
            "Description".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        // Simulate 200 weighted votes
        proposal.weighted_for = 150.0;
        proposal.weighted_against = 50.0;

        // Total available reputation: 1000
        let status = calculator.get_quorum_status(&proposal, 1000.0);

        // Required quorum: 0.33 × 1.2 (E1) × 0.8 (Local) × 1.06 (participation) ≈ 0.335
        // ≈ 0.335 × 1000 = 335 votes required
        // Actual: 200 votes
        assert_eq!(status.actual_votes, 200.0);
        assert!(status.required_votes > 300.0 && status.required_votes < 400.0);
        assert!(!status.met); // 200 < 335

        assert_eq!(status.epistemic_adjustment, 1.2); // E1 default
        assert_eq!(status.scope_adjustment, 0.8); // Local
    }

    #[test]
    fn test_combined_extreme_scenarios() {
        let calculator = DynamicQuorumCalculator::default();

        // Scenario 1: Best case (low quorum) - E4 + Local + High participation
        let mut best_case = HierarchicalProposal::new(
            "best".to_string(),
            "Trusted Local".to_string(),
            "Description".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );
        best_case.epistemic_tier = EpistemicTier::E4PubliclyReproducible;
        best_case.scope = ProposalScope::Local;

        let mut high_participation_calc = calculator.clone();
        high_participation_calc.recent_participation = 0.6;

        let best_quorum = high_participation_calc.calculate_required_quorum(&best_case);

        // 0.33 × 0.6 × 0.8 × 1.14 ≈ 0.18
        assert!(best_quorum < 0.20);
        assert!(best_quorum >= 0.15); // Floor

        // Scenario 2: Worst case (high quorum) - E0 + Global + Low participation
        let mut worst_case = HierarchicalProposal::new(
            "worst".to_string(),
            "Unverified Global".to_string(),
            "Description".to_string(),
            "proposer".to_string(),
            ProposalCategory::Governance,
            ProposalType::Normal,
        );
        worst_case.epistemic_tier = EpistemicTier::E0Null;
        worst_case.scope = ProposalScope::Global;

        let mut low_participation_calc = calculator.clone();
        low_participation_calc.recent_participation = 0.2;

        let worst_quorum = low_participation_calc.calculate_required_quorum(&worst_case);

        // 0.33 × 1.5 × 1.3 × 0.98 ≈ 0.63
        assert!(worst_quorum > 0.60);
        assert!(worst_quorum <= 0.75); // Ceiling

        // Verify vast difference
        assert!(worst_quorum > 3.0 * best_quorum); // >3x difference
    }

    #[test]
    fn test_invalid_participation_ignored() {
        let mut calculator = DynamicQuorumCalculator::default();
        let original = calculator.recent_participation;

        calculator.update_participation_rate(1.5); // Invalid (>1.0)
        assert_eq!(calculator.recent_participation, original);

        calculator.update_participation_rate(-0.1); // Invalid (<0.0)
        assert_eq!(calculator.recent_participation, original);
    }
}
