// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical proposal types with reputation-weighted governance

use serde::{Deserialize, Serialize};
use hdi::prelude::*;

use super::epistemic::EpistemicTier;

/// Hierarchical governance proposal with reputation weighting and cross-hApp escalation
///
/// ## Revolutionary Features
///
/// - **Reputation-Weighted Tallies**: Vote weight = f(PoGQ, TCDM, Entropy, Stake)
/// - **Hierarchical Scoping**: Local → Regional → Global automatic escalation
/// - **Epistemic Validation**: E0-E4 truth confidence scoring
/// - **Dynamic Quorum**: Adaptive participation requirements
///
/// ## Example
///
/// ```rust
/// use mycelix_governance::types::{HierarchicalProposal, ProposalScope, ProposalCategory, ProposalType};
///
/// let mut proposal = HierarchicalProposal::new(
///     "prop_001".to_string(),
///     "Upgrade FL aggregation algorithm".to_string(),
///     "Implement FedProx instead of FedAvg...".to_string(),
///     "proposer_agent_pubkey".to_string(),
///     ProposalCategory::Technical,
///     ProposalType::Normal,
/// );
///
/// // Customize as needed
/// proposal.scope = ProposalScope::Local;
/// proposal.escalation_threshold = 0.8; // 80% vote to escalate
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HierarchicalProposal {
    // Core identification
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposer: String, // AgentPubKey as string

    // Hierarchical governance
    pub scope: ProposalScope,
    pub parent_dao: Option<DnaHash>,
    pub escalation_threshold: f64, // Vote % needed to escalate to next level

    // Proposal classification
    pub category: ProposalCategory,
    pub proposal_type: ProposalType,

    // Reputation-weighted voting
    pub weighted_for: f64,
    pub weighted_against: f64,
    pub weighted_abstain: f64,

    // Dynamic quorum
    pub min_participation: f64, // 0.0 to 1.0
    pub quorum_met: bool,

    // Epistemic validation
    pub epistemic_tier: EpistemicTier,
    pub truth_confidence: f64, // 0.0 to 1.0

    // State management
    pub status: ProposalStatus,
    pub voting_deadline: i64, // Unix timestamp
    pub created_at: i64,
    pub executed_at: Option<i64>,

    // Execution
    pub actions_json: String, // JSON-serialized Vec<ProposalAction>
}

/// Proposal scope - determines governance level
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalScope {
    /// Single hApp only (e.g., EduNet-specific)
    Local,
    /// Multiple hApps in same category (e.g., all education hApps)
    Regional,
    /// Entire Mycelix ecosystem
    Global,
}

/// Proposal category - used for domain expertise weighting
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProposalCategory {
    /// Technical improvements (weighted by tech expertise)
    Technical,
    /// Economic/tokenomics changes
    Economic,
    /// Governance process changes
    Governance,
    /// Community management
    Community,
    /// Security patches/audits
    Security,
    /// Integration with other systems
    Integration,
}

/// Proposal type - determines voting timeline
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalType {
    /// 48-hour voting period (urgent matters)
    Fast,
    /// 7-day voting period (standard)
    Normal,
    /// 14-day voting period (major changes)
    Slow,
}

/// Proposal lifecycle status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalStatus {
    /// Voting in progress
    Active,
    /// Passed and awaiting execution
    Approved,
    /// Did not pass
    Rejected,
    /// Successfully executed
    Executed,
    /// Escalated to parent DAO
    Escalated,
    /// Cancelled by proposer
    Cancelled,
}

impl HierarchicalProposal {
    /// Create a new proposal with default values
    pub fn new(
        proposal_id: String,
        title: String,
        description: String,
        proposer: String,
        category: ProposalCategory,
        proposal_type: ProposalType,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();

        let deadline_hours = match proposal_type {
            ProposalType::Fast => 48,
            ProposalType::Normal => 168,
            ProposalType::Slow => 336,
        };

        Self {
            proposal_id,
            title,
            description,
            proposer,
            scope: ProposalScope::Local,
            parent_dao: None,
            escalation_threshold: 0.8, // Default 80% to escalate
            category,
            proposal_type,
            weighted_for: 0.0,
            weighted_against: 0.0,
            weighted_abstain: 0.0,
            min_participation: 0.33, // Default 33% quorum
            quorum_met: false,
            epistemic_tier: EpistemicTier::E1PersonalTestimony,
            truth_confidence: 0.0,
            status: ProposalStatus::Active,
            voting_deadline: now + (deadline_hours * 3600),
            created_at: now,
            executed_at: None,
            actions_json: "[]".to_string(),
        }
    }

    /// Calculate total weighted votes cast
    pub fn total_weighted_votes(&self) -> f64 {
        self.weighted_for + self.weighted_against + self.weighted_abstain
    }

    /// Calculate approval ratio (for / (for + against))
    pub fn approval_ratio(&self) -> f64 {
        let total = self.weighted_for + self.weighted_against;
        if total == 0.0 {
            return 0.0;
        }
        self.weighted_for / total
    }

    /// Check if voting deadline has passed
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.voting_deadline
    }

    /// Check if proposal should escalate to parent DAO
    pub fn should_escalate(&self) -> bool {
        self.approval_ratio() >= self.escalation_threshold
            && self.quorum_met
            && self.parent_dao.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_creation() {
        let proposal = HierarchicalProposal::new(
            "test_001".to_string(),
            "Test Proposal".to_string(),
            "Description".to_string(),
            "proposer_pubkey".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        assert_eq!(proposal.proposal_id, "test_001");
        assert_eq!(proposal.status, ProposalStatus::Active);
        assert_eq!(proposal.scope, ProposalScope::Local);
    }

    #[test]
    fn test_approval_ratio() {
        let mut proposal = HierarchicalProposal::new(
            "test_002".to_string(),
            "Test".to_string(),
            "Desc".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        proposal.weighted_for = 70.0;
        proposal.weighted_against = 30.0;

        assert_eq!(proposal.approval_ratio(), 0.7);
    }

    #[test]
    fn test_escalation_check() {
        let mut proposal = HierarchicalProposal::new(
            "test_003".to_string(),
            "Test".to_string(),
            "Desc".to_string(),
            "proposer".to_string(),
            ProposalCategory::Technical,
            ProposalType::Normal,
        );

        // Set up for escalation
        proposal.weighted_for = 85.0;
        proposal.weighted_against = 15.0;
        proposal.quorum_met = true;
        proposal.parent_dao = Some(DnaHash::from_raw_36(vec![0u8; 36]));

        assert!(proposal.should_escalate());
    }
}
