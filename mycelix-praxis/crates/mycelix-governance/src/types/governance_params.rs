// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Governance parameters for configuring DAO behavior
//!
//! This module defines tunable parameters that control how a DAO operates,
//! including quorum requirements, voting periods, reputation thresholds, and more.

use serde::{Deserialize, Serialize};

use super::proposal::ProposalType;
use super::epistemic::EpistemicTier;

/// Governance parameters - configures DAO behavior
///
/// These parameters define the rules and thresholds for governance operations.
/// They can be modified through governance proposals, creating a self-evolving DAO.
///
/// ## Parameter Categories
///
/// - **Quorum**: Minimum participation requirements
/// - **Voting**: Duration and approval thresholds
/// - **Reputation**: Minimum scores needed for participation
/// - **Escalation**: Thresholds for hierarchical DAO coordination
///
/// ## Example
///
/// ```rust
/// use mycelix_governance::types::GovernanceParams;
///
/// let params = GovernanceParams::default();
/// assert_eq!(params.base_quorum, 0.33); // 33% default
/// assert_eq!(params.fast_voting_hours, 48);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GovernanceParams {
    // === Quorum Parameters ===

    /// Base quorum for standard proposals (0.0 to 1.0)
    ///
    /// Default: 0.33 (33% of eligible voters must participate)
    pub base_quorum: f64,

    /// Minimum quorum for any proposal (0.0 to 1.0)
    ///
    /// Even if epistemic tier or other factors reduce quorum,
    /// it cannot go below this threshold.
    ///
    /// Default: 0.20 (20% absolute minimum)
    pub min_quorum: f64,

    /// Maximum quorum for any proposal (0.0 to 1.0)
    ///
    /// Even if factors increase quorum, it cannot exceed this.
    ///
    /// Default: 0.60 (60% absolute maximum)
    pub max_quorum: f64,

    // === Voting Period Parameters ===

    /// Voting period for fast proposals (in hours)
    ///
    /// Used for urgent matters like security patches.
    ///
    /// Default: 48 hours (2 days)
    pub fast_voting_hours: i64,

    /// Voting period for normal proposals (in hours)
    ///
    /// Used for standard governance decisions.
    ///
    /// Default: 168 hours (7 days)
    pub normal_voting_hours: i64,

    /// Voting period for slow proposals (in hours)
    ///
    /// Used for major changes like constitution amendments.
    ///
    /// Default: 336 hours (14 days)
    pub slow_voting_hours: i64,

    // === Approval Thresholds ===

    /// Approval ratio needed to pass (0.0 to 1.0)
    ///
    /// Calculated as: for_votes / (for_votes + against_votes)
    ///
    /// Default: 0.51 (simple majority)
    pub approval_threshold: f64,

    /// Super-majority threshold for constitutional changes (0.0 to 1.0)
    ///
    /// Default: 0.67 (two-thirds majority)
    pub supermajority_threshold: f64,

    // === Reputation Requirements ===

    /// Minimum reputation to propose (composite trust score)
    ///
    /// Prevents spam by requiring proposers to have established reputation.
    ///
    /// Default: 10.0
    pub min_proposer_reputation: f64,

    /// Minimum reputation to vote (composite trust score)
    ///
    /// Allows voting with minimal participation history.
    ///
    /// Default: 1.0
    pub min_voter_reputation: f64,

    /// Reputation decay rate per day (0.0 to 1.0)
    ///
    /// Inactive users gradually lose reputation to prevent
    /// long-absent members from dominating votes.
    ///
    /// Default: 0.01 (1% per day, ~30% per month)
    pub reputation_decay_rate: f64,

    // === Hierarchical Governance Parameters ===

    /// Approval ratio needed to escalate to parent DAO (0.0 to 1.0)
    ///
    /// Default: 0.80 (80% support needed to escalate)
    pub escalation_threshold: f64,

    /// Whether to allow automatic escalation
    ///
    /// If false, escalation requires manual action.
    ///
    /// Default: true
    pub auto_escalation_enabled: bool,

    // === Epistemic Tier Integration ===

    /// Whether to use epistemic tiers for quorum adjustment
    ///
    /// If true, higher tiers reduce quorum requirements.
    ///
    /// Default: true
    pub epistemic_quorum_adjustment: bool,

    /// Multiplier for epistemic tier quorum reduction (0.0 to 1.0)
    ///
    /// Controls how much epistemic tiers affect quorum.
    /// 1.0 = full effect, 0.0 = no effect
    ///
    /// Default: 1.0
    pub epistemic_adjustment_strength: f64,

    // === Economic Parameters ===

    /// Minimum CIV stake to propose (in CIV tokens)
    ///
    /// Economic barrier to prevent spam proposals.
    ///
    /// Default: 100.0 CIV
    pub min_proposal_stake: f64,

    /// Slash percentage for rejected proposals (0.0 to 1.0)
    ///
    /// Proposers lose this percentage of stake if proposal rejected.
    /// Incentivizes quality proposals.
    ///
    /// Default: 0.10 (10%)
    pub proposal_slash_percentage: f64,

    // === Dynamic Quorum Parameters ===

    /// Whether to enable dynamic quorum based on participation history
    ///
    /// If true, quorum adjusts based on recent voting activity.
    ///
    /// Default: true
    pub dynamic_quorum_enabled: bool,

    /// Window for calculating average participation (in days)
    ///
    /// Used to determine recent participation trends.
    ///
    /// Default: 30 days
    pub participation_window_days: i64,

    /// Maximum quorum adjustment from dynamic calculation (0.0 to 1.0)
    ///
    /// Limits how much dynamic quorum can change from base.
    ///
    /// Default: 0.15 (±15% adjustment)
    pub max_dynamic_adjustment: f64,
}

impl GovernanceParams {
    /// Get effective quorum for a proposal based on its epistemic tier
    ///
    /// Combines base quorum with epistemic tier adjustments.
    pub fn effective_quorum(&self, tier: EpistemicTier) -> f64 {
        let base = self.base_quorum;

        if !self.epistemic_quorum_adjustment {
            return base.max(self.min_quorum).min(self.max_quorum);
        }

        // Apply epistemic tier adjustment
        let tier_quorum = tier.min_quorum();
        let adjustment_strength = self.epistemic_adjustment_strength;

        // Weighted average between base and tier quorum
        let effective = base * (1.0 - adjustment_strength) + tier_quorum * adjustment_strength;

        // Clamp to min/max bounds
        effective.max(self.min_quorum).min(self.max_quorum)
    }

    /// Get voting period in hours for a proposal type
    pub fn voting_period_hours(&self, proposal_type: ProposalType) -> i64 {
        match proposal_type {
            ProposalType::Fast => self.fast_voting_hours,
            ProposalType::Normal => self.normal_voting_hours,
            ProposalType::Slow => self.slow_voting_hours,
        }
    }

    /// Calculate deadline timestamp for a proposal
    ///
    /// Returns Unix timestamp when voting should close.
    pub fn calculate_deadline(&self, proposal_type: ProposalType, start_time: i64) -> i64 {
        let hours = self.voting_period_hours(proposal_type);
        start_time + (hours * 3600)
    }

    /// Check if a composite reputation meets minimum for proposing
    pub fn can_propose(&self, reputation: f64) -> bool {
        reputation >= self.min_proposer_reputation
    }

    /// Check if a composite reputation meets minimum for voting
    pub fn can_vote(&self, reputation: f64) -> bool {
        reputation >= self.min_voter_reputation
    }

    /// Apply reputation decay for inactive period
    ///
    /// Returns new reputation after applying decay for given number of days.
    pub fn apply_reputation_decay(&self, current_reputation: f64, days_inactive: i64) -> f64 {
        let decay_factor = (1.0 - self.reputation_decay_rate).powi(days_inactive as i32);
        current_reputation * decay_factor
    }

    /// Validate governance parameters for consistency
    ///
    /// Returns error message if parameters are invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Quorum checks
        if self.min_quorum > self.max_quorum {
            return Err("min_quorum cannot exceed max_quorum".to_string());
        }
        if self.base_quorum < self.min_quorum || self.base_quorum > self.max_quorum {
            return Err("base_quorum must be between min_quorum and max_quorum".to_string());
        }
        if self.min_quorum < 0.0 || self.max_quorum > 1.0 {
            return Err("quorum values must be between 0.0 and 1.0".to_string());
        }

        // Voting period checks
        if self.fast_voting_hours <= 0 || self.normal_voting_hours <= 0 || self.slow_voting_hours <= 0 {
            return Err("voting periods must be positive".to_string());
        }
        if self.fast_voting_hours > self.normal_voting_hours || self.normal_voting_hours > self.slow_voting_hours {
            return Err("fast <= normal <= slow voting periods required".to_string());
        }

        // Approval threshold checks
        if self.approval_threshold < 0.0 || self.approval_threshold > 1.0 {
            return Err("approval_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.supermajority_threshold < self.approval_threshold {
            return Err("supermajority_threshold must be >= approval_threshold".to_string());
        }

        // Reputation checks
        if self.min_proposer_reputation < self.min_voter_reputation {
            return Err("proposer reputation must be >= voter reputation".to_string());
        }
        if self.reputation_decay_rate < 0.0 || self.reputation_decay_rate > 1.0 {
            return Err("reputation_decay_rate must be between 0.0 and 1.0".to_string());
        }

        // Economic checks
        if self.min_proposal_stake < 0.0 {
            return Err("min_proposal_stake cannot be negative".to_string());
        }
        if self.proposal_slash_percentage < 0.0 || self.proposal_slash_percentage > 1.0 {
            return Err("proposal_slash_percentage must be between 0.0 and 1.0".to_string());
        }

        // Dynamic quorum checks
        if self.participation_window_days <= 0 {
            return Err("participation_window_days must be positive".to_string());
        }
        if self.max_dynamic_adjustment < 0.0 || self.max_dynamic_adjustment > 1.0 {
            return Err("max_dynamic_adjustment must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

impl Default for GovernanceParams {
    /// Default governance parameters - conservative and secure
    fn default() -> Self {
        Self {
            // Quorum parameters
            base_quorum: 0.33,
            min_quorum: 0.20,
            max_quorum: 0.60,

            // Voting periods (in hours)
            fast_voting_hours: 48,
            normal_voting_hours: 168,
            slow_voting_hours: 336,

            // Approval thresholds
            approval_threshold: 0.51,
            supermajority_threshold: 0.67,

            // Reputation requirements
            min_proposer_reputation: 10.0,
            min_voter_reputation: 1.0,
            reputation_decay_rate: 0.01,

            // Hierarchical governance
            escalation_threshold: 0.80,
            auto_escalation_enabled: true,

            // Epistemic tier integration
            epistemic_quorum_adjustment: true,
            epistemic_adjustment_strength: 1.0,

            // Economic parameters
            min_proposal_stake: 100.0,
            proposal_slash_percentage: 0.10,

            // Dynamic quorum
            dynamic_quorum_enabled: true,
            participation_window_days: 30,
            max_dynamic_adjustment: 0.15,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = GovernanceParams::default();

        assert_eq!(params.base_quorum, 0.33);
        assert_eq!(params.min_quorum, 0.20);
        assert_eq!(params.max_quorum, 0.60);

        // Validation should pass
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_effective_quorum() {
        let params = GovernanceParams::default();

        // E4 should have lower quorum than E0
        let e4_quorum = params.effective_quorum(EpistemicTier::E4PubliclyReproducible);
        let e0_quorum = params.effective_quorum(EpistemicTier::E0Null);

        assert!(e4_quorum < e0_quorum);

        // All should be within bounds
        assert!(e4_quorum >= params.min_quorum);
        assert!(e0_quorum <= params.max_quorum);
    }

    #[test]
    fn test_voting_periods() {
        let params = GovernanceParams::default();

        assert_eq!(params.voting_period_hours(ProposalType::Fast), 48);
        assert_eq!(params.voting_period_hours(ProposalType::Normal), 168);
        assert_eq!(params.voting_period_hours(ProposalType::Slow), 336);

        // Periods should be ascending
        assert!(params.fast_voting_hours < params.normal_voting_hours);
        assert!(params.normal_voting_hours < params.slow_voting_hours);
    }

    #[test]
    fn test_calculate_deadline() {
        let params = GovernanceParams::default();
        let start_time = 1000000;

        let deadline = params.calculate_deadline(ProposalType::Fast, start_time);
        assert_eq!(deadline, start_time + (48 * 3600));
    }

    #[test]
    fn test_reputation_checks() {
        let params = GovernanceParams::default();

        assert!(params.can_propose(10.0));
        assert!(!params.can_propose(9.9));

        assert!(params.can_vote(1.0));
        assert!(!params.can_vote(0.9));
    }

    #[test]
    fn test_reputation_decay() {
        let params = GovernanceParams::default();

        let initial_reputation = 100.0;
        let after_30_days = params.apply_reputation_decay(initial_reputation, 30);

        // With 1% daily decay, 30 days should reduce by ~26%
        // (1 - 0.01)^30 ≈ 0.74
        assert!(after_30_days < initial_reputation);
        assert!(after_30_days > initial_reputation * 0.70);
        assert!(after_30_days < initial_reputation * 0.75);
    }

    #[test]
    fn test_validation() {
        let mut params = GovernanceParams::default();

        // Valid params should pass
        assert!(params.validate().is_ok());

        // Invalid: min > max quorum
        params.min_quorum = 0.70;
        params.max_quorum = 0.60;
        assert!(params.validate().is_err());
        params.min_quorum = 0.20; // Reset

        // Invalid: base outside bounds
        params.base_quorum = 0.10;
        assert!(params.validate().is_err());
        params.base_quorum = 0.33; // Reset

        // Invalid: voting period order
        params.fast_voting_hours = 200;
        assert!(params.validate().is_err());
        params.fast_voting_hours = 48; // Reset

        // Invalid: proposer < voter reputation
        params.min_proposer_reputation = 0.5;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_disabled_epistemic_adjustment() {
        let mut params = GovernanceParams::default();
        params.epistemic_quorum_adjustment = false;

        // All tiers should have same quorum
        let e0_quorum = params.effective_quorum(EpistemicTier::E0Null);
        let e4_quorum = params.effective_quorum(EpistemicTier::E4PubliclyReproducible);

        assert_eq!(e0_quorum, e4_quorum);
        assert_eq!(e0_quorum, params.base_quorum);
    }

    #[test]
    fn test_partial_epistemic_adjustment() {
        let mut params = GovernanceParams::default();
        params.epistemic_adjustment_strength = 0.5; // 50% effect

        let full_strength_params = GovernanceParams::default();

        let e4_quorum_half = params.effective_quorum(EpistemicTier::E4PubliclyReproducible);
        let e4_quorum_full = full_strength_params.effective_quorum(EpistemicTier::E4PubliclyReproducible);

        // Half strength should be between base and full strength
        assert!(e4_quorum_half > e4_quorum_full);
        assert!(e4_quorum_half < params.base_quorum);
    }
}
