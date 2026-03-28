// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # DAO Integrity Zome
//!
//! Defines entry types and validation for decentralized governance.
//! This zome is immutable - entry type definitions should never change.

use hdi::prelude::*;

/// Governance proposal entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Proposal {
    /// Unique identifier
    pub proposal_id: String,

    /// Proposal title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Proposer agent (string representation)
    pub proposer: String,

    /// Proposal type (fast, normal, slow)
    pub proposal_type: ProposalType,

    /// Proposal category
    pub category: ProposalCategory,

    /// Current status
    pub status: ProposalStatus,

    /// Vote tallies (flattened, simple mode)
    pub for_votes: u32,
    pub against_votes: u32,
    pub abstain_votes: u32,

    /// Weighted vote tallies (permille, for Quadratic/Conviction modes)
    pub weighted_for: u64,
    pub weighted_against: u64,

    /// Voting mode for this proposal
    pub voting_mode: VotingMode,

    /// Voting deadline
    pub voting_deadline: i64,

    /// Creation timestamp
    pub created_at: i64,

    /// Execution timestamp (if approved and executed)
    pub executed_at: Option<i64>,

    /// Associated actions (serialized JSON string)
    pub actions_json: String,
}

/// Individual vote entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// Proposal being voted on
    pub proposal_id: String,

    /// Voter agent (string representation)
    pub voter: String,

    /// Vote choice
    pub choice: VoteChoice,

    /// Optional justification
    pub justification: Option<String>,

    /// Vote timestamp
    pub timestamp: i64,

    /// Reputation staked on this vote (permille, for weighted voting modes)
    pub reputation_allocated: Option<u64>,

    /// Computed vote weight (permille, for auditing)
    pub vote_weight: Option<u64>,
}

// ============================================================================
// Enums - These don't use #[hdk_entry_helper], just standard derives
// ============================================================================

/// Type of governance proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum ProposalType {
    /// Fast path (24-48 hours, emergency fixes)
    Fast,
    /// Normal path (3-14 days, features and updates)
    Normal,
    /// Slow path (14+ days, protocol changes)
    Slow,
}

/// Category of proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum ProposalCategory {
    /// Course content and curriculum
    Curriculum,
    /// Protocol parameters (aggregation, privacy)
    Protocol,
    /// Credential standards and rubrics
    Credentials,
    /// Treasury and resource allocation
    Treasury,
    /// Governance rules themselves
    Governance,
    /// Emergency actions
    Emergency,
}

/// Status of a proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum ProposalStatus {
    /// Open for voting
    Active,
    /// Approved, pending execution
    Approved,
    /// Approved and executed
    Executed,
    /// Rejected by vote
    Rejected,
    /// Cancelled by proposer
    Cancelled,
    /// Vetoed by maintainer
    Vetoed,
}

/// Vote choice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

/// Voting mode for proposals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SerializedBytes)]
pub enum VotingMode {
    /// 1 person = 1 vote (default, backward compatible)
    Simple,
    /// sqrt(reputation) voting — prevents plutocracy
    Quadratic,
    /// Time-locked voting (future implementation)
    Conviction,
}

impl Default for VotingMode {
    fn default() -> Self {
        VotingMode::Simple
    }
}

// ============================================================================
// Entry types
// ============================================================================

#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    Proposal(Proposal),
    Vote(Vote),
}

// ============================================================================
// Link types
// ============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from proposal to votes
    ProposalToVotes,
    /// Link from agent to proposals they created
    AgentToProposals,
    /// Link from agent to votes they cast
    AgentToVotes,
    /// Link from category to proposals
    CategoryToProposals,
    /// All proposals (for listing)
    AllProposals,
}

// ============================================================================
// Validation
// ============================================================================

pub fn validate_create_proposal(proposal: &Proposal) -> ExternResult<ValidateCallbackResult> {
    // Validate title
    if proposal.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal title cannot be empty".to_string(),
        ));
    }

    if proposal.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal title cannot exceed 200 characters".to_string(),
        ));
    }

    // Validate description
    if proposal.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal description cannot be empty".to_string(),
        ));
    }

    if proposal.description.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal description cannot exceed 10000 characters".to_string(),
        ));
    }

    // Validate voting deadline is after creation
    if proposal.voting_deadline <= proposal.created_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Voting deadline must be after creation time".to_string(),
        ));
    }

    // Validate voting period based on proposal type
    let voting_period_hours = (proposal.voting_deadline - proposal.created_at) / 3600;

    let (min_hours, max_hours, type_name) = match proposal.proposal_type {
        ProposalType::Fast => (24, 72, "Fast"),        // 1-3 days
        ProposalType::Normal => (72, 336, "Normal"),   // 3-14 days
        ProposalType::Slow => (336, 720, "Slow"),      // 14-30 days
    };

    if voting_period_hours < min_hours || voting_period_hours > max_hours {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "{} proposals must have voting period between {} and {} hours (got {} hours)",
                type_name, min_hours, max_hours, voting_period_hours
            ),
        ));
    }

    // Validate proposal ID is not empty
    if proposal.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID cannot be empty".to_string(),
        ));
    }

    // Validate actions JSON is valid JSON
    if serde_json::from_str::<serde_json::Value>(&proposal.actions_json).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Actions JSON is not valid JSON".to_string(),
        ));
    }

    // Validate vote tallies start at zero for new proposals
    if proposal.for_votes != 0 || proposal.against_votes != 0 || proposal.abstain_votes != 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "New proposals must have zero votes".to_string(),
        ));
    }

    // Validate weighted tallies start at zero
    if proposal.weighted_for != 0 || proposal.weighted_against != 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "New proposals must have zero weighted votes".to_string(),
        ));
    }

    // Validate status is Active for new proposals
    if proposal.status != ProposalStatus::Active {
        return Ok(ValidateCallbackResult::Invalid(
            "New proposals must have Active status".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_create_vote(vote: &Vote) -> ExternResult<ValidateCallbackResult> {
    // Validate proposal ID is not empty
    if vote.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote proposal_id cannot be empty".to_string(),
        ));
    }

    // Validate voter is not empty
    if vote.voter.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter cannot be empty".to_string(),
        ));
    }

    // Validate justification length if provided
    if let Some(ref justification) = vote.justification {
        if justification.len() > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote justification cannot exceed 1000 characters".to_string(),
            ));
        }
    }

    // Note: Additional validation (proposal exists, deadline hasn't passed, no duplicate votes)
    // should be done in the coordinator zome before creating the vote entry

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::Proposal(proposal)) => {
                        validate_create_proposal(&proposal)
                    }
                    Some(EntryTypes::Vote(vote)) => {
                        validate_create_vote(&vote)
                    }
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // Test Helpers
    // =============================================================================

    fn create_valid_proposal() -> Proposal {
        let now = chrono::Utc::now().timestamp();
        Proposal {
            proposal_id: "prop_001".to_string(),
            title: "Test Proposal".to_string(),
            description: "This is a test proposal description".to_string(),
            proposer: "did:example:proposer123".to_string(),
            proposal_type: ProposalType::Normal,
            category: ProposalCategory::Curriculum,
            status: ProposalStatus::Active,
            for_votes: 0,
            against_votes: 0,
            abstain_votes: 0,
            weighted_for: 0,
            weighted_against: 0,
            voting_mode: VotingMode::Simple,
            voting_deadline: now + (7 * 24 * 3600), // 7 days from now
            created_at: now,
            executed_at: None,
            actions_json: "[]".to_string(),
        }
    }

    fn create_valid_vote() -> Vote {
        let now = chrono::Utc::now().timestamp();
        Vote {
            proposal_id: "prop_001".to_string(),
            voter: "did:example:voter456".to_string(),
            choice: VoteChoice::For,
            justification: Some("I support this proposal".to_string()),
            timestamp: now,
            reputation_allocated: None,
            vote_weight: None,
        }
    }

    // =============================================================================
    // Basic Enum Tests
    // =============================================================================

    #[test]
    fn test_proposal_types() {
        let fast = ProposalType::Fast;
        let normal = ProposalType::Normal;
        let slow = ProposalType::Slow;

        assert_ne!(fast, normal);
        assert_ne!(normal, slow);
        assert_ne!(fast, slow);
    }

    #[test]
    fn test_vote_choice() {
        let for_vote = VoteChoice::For;
        let against = VoteChoice::Against;
        let abstain = VoteChoice::Abstain;

        assert_ne!(for_vote, against);
        assert_ne!(against, abstain);
        assert_ne!(for_vote, abstain);
    }

    #[test]
    fn test_proposal_status() {
        let active = ProposalStatus::Active;
        let approved = ProposalStatus::Approved;
        let rejected = ProposalStatus::Rejected;

        assert_ne!(active, approved);
        assert_ne!(approved, rejected);
    }

    // =============================================================================
    // Proposal Validation Tests
    // =============================================================================

    #[test]
    fn test_valid_proposal() {
        let proposal = create_valid_proposal();
        let result = validate_create_proposal(&proposal).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_proposal_empty_title() {
        let mut proposal = create_valid_proposal();
        proposal.title = "".to_string();
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_title_too_long() {
        let mut proposal = create_valid_proposal();
        proposal.title = "x".repeat(201);
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_empty_description() {
        let mut proposal = create_valid_proposal();
        proposal.description = "".to_string();
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_description_too_long() {
        let mut proposal = create_valid_proposal();
        proposal.description = "x".repeat(10001);
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_deadline_in_past() {
        let mut proposal = create_valid_proposal();
        proposal.voting_deadline = chrono::Utc::now().timestamp() - 3600; // 1 hour ago
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fast_proposal_valid_period() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_type = ProposalType::Fast;
        let now = chrono::Utc::now().timestamp();
        proposal.created_at = now;
        proposal.voting_deadline = now + (48 * 3600); // 2 days
        let result = validate_create_proposal(&proposal).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_fast_proposal_too_short() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_type = ProposalType::Fast;
        let now = chrono::Utc::now().timestamp();
        proposal.created_at = now;
        proposal.voting_deadline = now + (12 * 3600); // 12 hours (< 24 minimum)
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_fast_proposal_too_long() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_type = ProposalType::Fast;
        let now = chrono::Utc::now().timestamp();
        proposal.created_at = now;
        proposal.voting_deadline = now + (96 * 3600); // 4 days (> 72 maximum)
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_normal_proposal_valid_period() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_type = ProposalType::Normal;
        let now = chrono::Utc::now().timestamp();
        proposal.created_at = now;
        proposal.voting_deadline = now + (7 * 24 * 3600); // 7 days
        let result = validate_create_proposal(&proposal).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_slow_proposal_valid_period() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_type = ProposalType::Slow;
        let now = chrono::Utc::now().timestamp();
        proposal.created_at = now;
        proposal.voting_deadline = now + (14 * 24 * 3600); // 14 days
        let result = validate_create_proposal(&proposal).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_proposal_empty_id() {
        let mut proposal = create_valid_proposal();
        proposal.proposal_id = "".to_string();
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_invalid_json() {
        let mut proposal = create_valid_proposal();
        proposal.actions_json = "not valid json {".to_string();
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_nonzero_votes() {
        let mut proposal = create_valid_proposal();
        proposal.for_votes = 5;
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_proposal_not_active_status() {
        let mut proposal = create_valid_proposal();
        proposal.status = ProposalStatus::Approved;
        let result = validate_create_proposal(&proposal).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // =============================================================================
    // Vote Validation Tests
    // =============================================================================

    #[test]
    fn test_valid_vote() {
        let vote = create_valid_vote();
        let result = validate_create_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_vote_empty_proposal_id() {
        let mut vote = create_valid_vote();
        vote.proposal_id = "".to_string();
        let result = validate_create_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_vote_empty_voter() {
        let mut vote = create_valid_vote();
        vote.voter = "".to_string();
        let result = validate_create_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_vote_justification_too_long() {
        let mut vote = create_valid_vote();
        vote.justification = Some("x".repeat(1001));
        let result = validate_create_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_vote_no_justification() {
        let mut vote = create_valid_vote();
        vote.justification = None;
        let result = validate_create_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_vote_timestamp_in_future() {
        let mut vote = create_valid_vote();
        vote.timestamp = chrono::Utc::now().timestamp() + 3600; // Future timestamps are allowed at validation layer
        let result = validate_create_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_vote_all_choices() {
        let base_vote = create_valid_vote();

        let mut vote_for = base_vote.clone();
        vote_for.choice = VoteChoice::For;
        assert_eq!(validate_create_vote(&vote_for).unwrap(), ValidateCallbackResult::Valid);

        let mut vote_against = base_vote.clone();
        vote_against.choice = VoteChoice::Against;
        assert_eq!(validate_create_vote(&vote_against).unwrap(), ValidateCallbackResult::Valid);

        let mut vote_abstain = base_vote.clone();
        vote_abstain.choice = VoteChoice::Abstain;
        assert_eq!(validate_create_vote(&vote_abstain).unwrap(), ValidateCallbackResult::Valid);
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    #[test]
    fn test_proposal_all_categories() {
        let now = chrono::Utc::now().timestamp();
        let categories = vec![
            ProposalCategory::Curriculum,
            ProposalCategory::Protocol,
            ProposalCategory::Credentials,
            ProposalCategory::Treasury,
            ProposalCategory::Governance,
            ProposalCategory::Emergency,
        ];

        for category in categories {
            let mut proposal = create_valid_proposal();
            proposal.category = category;
            proposal.created_at = now;
            proposal.voting_deadline = now + (7 * 24 * 3600);
            let result = validate_create_proposal(&proposal).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_proposal_complex_actions_json() {
        let mut proposal = create_valid_proposal();
        proposal.actions_json = r#"[{"type":"update_param","param":"max_learners","value":100}]"#.to_string();
        let result = validate_create_proposal(&proposal).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
