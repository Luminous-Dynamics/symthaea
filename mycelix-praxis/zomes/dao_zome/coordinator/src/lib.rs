// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # DAO Coordinator Zome
//!
//! Implements business logic for decentralized governance.
//! This zome is upgradeable - business logic can change without breaking data.
//!
//! ## Core Functions:
//! - Create and manage proposals
//! - Cast and tally votes
//! - Execute approved proposals
//! - Query proposal status and history

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use dao_integrity::{
    Proposal, Vote, ProposalType, ProposalCategory, ProposalStatus, VoteChoice,
    VotingMode, EntryTypes, LinkTypes
};

/// Create a new governance proposal
#[hdk_extern]
pub fn create_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Participant tier with identity >= 0.25
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_proposal(),
        "create_proposal",
    )?;

    // Get proposer agent info
    let agent_info = agent_info()?;
    let proposer_pubkey = agent_info.agent_initial_pubkey;

    let now = chrono::Utc::now().timestamp();
    let voting_deadline = voting_deadline(now, &input.proposal_type);

    // Serialize actions to JSON string
    let actions_json = serde_json::to_string(&input.actions)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to serialize actions: {}", e))))?;

    // Create proposal entry
    let proposal = Proposal {
        proposal_id: input.proposal_id.clone(),
        title: input.title,
        description: input.description,
        proposer: proposer_pubkey.to_string(),
        proposal_type: input.proposal_type,
        category: input.category,
        status: ProposalStatus::Active,
        for_votes: 0,
        against_votes: 0,
        abstain_votes: 0,
        weighted_for: 0,
        weighted_against: 0,
        voting_mode: input.voting_mode.unwrap_or_default(),
        voting_deadline,
        created_at: now,
        executed_at: None,
        actions_json,
    };

    // Store proposal entry
    let action_hash = create_entry(EntryTypes::Proposal(proposal.clone()))?;

    // Create links for easy lookup
    // Link from category to proposal
    let category_anchor = Path::from(format!("category_{:?}", proposal.category));
    let category_entry_hash = ensure_path(category_anchor, LinkTypes::CategoryToProposals)?;

    create_link(
        category_entry_hash,
        action_hash.clone(),
        LinkTypes::CategoryToProposals,
        vec![],
    )?;

    // Link from agent to proposal
    let agent_entry_hash: AnyDhtHash = proposer_pubkey.into();

    create_link(
        agent_entry_hash,
        action_hash.clone(),
        LinkTypes::AgentToProposals,
        vec![],
    )?;

    // Add to all proposals list
    let all_proposals_anchor = Path::from("all_proposals");
    let all_proposals_hash = ensure_path(all_proposals_anchor, LinkTypes::AllProposals)?;

    create_link(
        all_proposals_hash,
        action_hash.clone(),
        LinkTypes::AllProposals,
        vec![],
    )?;

    Ok(action_hash)
}

/// Cast a vote on a proposal
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<ActionHash> {
    // Trust tier gate: requires Citizen tier with identity >= 0.5
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
        "cast_vote",
    )?;

    // Get voter agent info
    let agent_info = agent_info()?;
    let voter_pubkey = agent_info.agent_initial_pubkey;

    // Verify proposal exists and is active (single fetch to avoid races)
    let proposal_record = get(input.proposal_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Proposal not found".to_string())))?;

    let proposal: Proposal = match proposal_record.entry().as_option() {
        Some(Entry::App(bytes)) => Proposal::try_from(SerializedBytes::from(UnsafeBytes::from(
            bytes.bytes().to_vec(),
        )))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to deserialize proposal: {}", e))))?,
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Record was not a Proposal entry".to_string()
            )))
        }
    };

    if proposal.status != ProposalStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal is not active".to_string()
        )));
    }

    // Ensure the caller's proposal_id matches the stored proposal entry
    if proposal.proposal_id != input.proposal_id {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal id mismatch".to_string()
        )));
    }

    // Verify voting deadline hasn't passed
    if chrono::Utc::now().timestamp() > proposal.voting_deadline {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voting deadline has passed".to_string()
        )));
    }

    // Verify voter hasn't already voted (proposal-scoped check)
    if has_existing_vote(&input.proposal_hash, &voter_pubkey)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voter has already cast a vote for this proposal".to_string()
        )));
    }

    // Compute vote weight for weighted voting modes
    let (reputation_allocated, vote_weight) = match proposal.voting_mode {
        VotingMode::Quadratic => {
            let rep = input.reputation_allocated.unwrap_or(1000); // default 1000 permille
            let weight = isqrt(rep);
            (Some(rep), Some(weight))
        }
        VotingMode::Conviction => {
            // Conviction voting: for now, treat like quadratic (future: time-lock)
            let rep = input.reputation_allocated.unwrap_or(1000);
            let weight = isqrt(rep);
            (Some(rep), Some(weight))
        }
        VotingMode::Simple => {
            (None, None)
        }
    };

    // Create vote entry
    let vote = Vote {
        proposal_id: input.proposal_id.clone(),
        voter: voter_pubkey.to_string(),
        choice: input.choice,
        justification: input.justification,
        timestamp: chrono::Utc::now().timestamp(),
        reputation_allocated,
        vote_weight,
    };

    // Store vote entry
    let vote_hash = create_entry(EntryTypes::Vote(vote.clone()))?;

    // Link vote to proposal
    create_link(
        input.proposal_hash.clone(),
        vote_hash.clone(),
        LinkTypes::ProposalToVotes,
        vec![],
    )?;

    // Link vote to voter
    let voter_entry_hash: AnyDhtHash = voter_pubkey.into();

    create_link(
        voter_entry_hash,
        vote_hash.clone(),
        LinkTypes::AgentToVotes,
        vec![],
    )?;

    // Update vote tallies on proposal
    let updated_proposal = update_proposal_vote_tallies(
        input.proposal_hash.clone(),
        proposal,
        &vote.choice,
        vote_weight,
    )?;

    // Update the proposal entry with new tallies
    update_entry(input.proposal_hash, &updated_proposal)?;

    Ok(vote_hash)
}

/// Get a specific proposal by its hash
#[hdk_extern]
pub fn get_proposal(proposal_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(proposal_hash, GetOptions::default())
}

/// Get all proposals in a specific category
#[hdk_extern]
pub fn get_proposals_by_category(category: ProposalCategory) -> ExternResult<Vec<Record>> {
    let category_anchor = format!("category_{:?}", category);
    let category_anchor = Path::from(category_anchor);
    let category_entry_hash = ensure_path(category_anchor, LinkTypes::CategoryToProposals)?;

    let links = get_links(
        LinkQuery::new(
            category_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::CategoryToProposals as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                proposals.push(record);
            }
        }
    }

    Ok(proposals)
}

/// Get all proposals created by a specific agent
#[hdk_extern]
pub fn get_agent_proposals(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let agent_entry_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::new(
            agent_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToProposals as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                proposals.push(record);
            }
        }
    }

    Ok(proposals)
}

/// Get all votes cast by a specific agent
#[hdk_extern]
pub fn get_agent_votes(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let agent_entry_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::new(
            agent_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                votes.push(record);
            }
        }
    }

    Ok(votes)
}

/// Get all votes for a specific proposal
#[hdk_extern]
pub fn get_proposal_votes(proposal_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::new(
            proposal_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::ProposalToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                votes.push(record);
            }
        }
    }

    Ok(votes)
}

/// Get all proposals (for admin/overview)
#[hdk_extern]
pub fn get_all_proposals(_: ()) -> ExternResult<Vec<Record>> {
    let all_proposals_anchor = Path::from("all_proposals");
    let all_proposals_hash = ensure_path(all_proposals_anchor, LinkTypes::AllProposals)?;

    let links = get_links(
        LinkQuery::new(
            all_proposals_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AllProposals as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                proposals.push(record);
            }
        }
    }

    Ok(proposals)
}

// ============================================================================
// Pure business logic (HDK-free, unit-testable)
// ============================================================================

/// Determine proposal outcome from vote counts.
///
/// Requires strictly more than `threshold_pct`% of non-abstain votes to pass.
/// If there are zero non-abstain votes, the proposal is rejected.
pub fn determine_proposal_outcome(
    for_votes: u32,
    against_votes: u32,
    _abstain: u32,
    threshold_pct: u32,
) -> ProposalStatus {
    let total_deciding = for_votes + against_votes;
    if total_deciding == 0 {
        return ProposalStatus::Rejected;
    }
    let for_percentage = (for_votes as f64 / total_deciding as f64) * 100.0;
    if for_percentage > threshold_pct as f64 {
        ProposalStatus::Approved
    } else {
        ProposalStatus::Rejected
    }
}

/// Calculate voting deadline timestamp from creation time and proposal type.
///
/// Returns deadline as Unix timestamp (seconds).
/// - Fast: 48 hours
/// - Normal: 168 hours (7 days)
/// - Slow: 336 hours (14 days)
pub fn voting_deadline(created_at: i64, proposal_type: &ProposalType) -> i64 {
    let deadline_hours: i64 = match proposal_type {
        ProposalType::Fast => 48,
        ProposalType::Normal => 168,
        ProposalType::Slow => 336,
    };
    created_at + deadline_hours * 3600
}

/// Increment vote tallies on a proposal (pure mutation, no HDK).
///
/// Returns the updated (for_votes, against_votes, abstain_votes).
pub fn apply_vote(
    for_votes: u32,
    against_votes: u32,
    abstain_votes: u32,
    choice: &VoteChoice,
) -> (u32, u32, u32) {
    match choice {
        VoteChoice::For => (for_votes.saturating_add(1), against_votes, abstain_votes),
        VoteChoice::Against => (for_votes, against_votes.saturating_add(1), abstain_votes),
        VoteChoice::Abstain => (for_votes, against_votes, abstain_votes.saturating_add(1)),
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Integer square root for u64 (Newton's method)
///
/// Used for quadratic voting: vote_weight = isqrt(reputation_allocated)
pub fn isqrt(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    // Newton's method with overflow-safe initial guess
    let mut x = n;
    let mut y = n / 2 + 1;
    // Ensure y < x for the first iteration (y = n/2+1 can equal n when n=2)
    if y >= x {
        y = x - 1;
    }
    loop {
        if y >= x {
            return x;
        }
        x = y;
        y = (x + n / x) / 2;
    }
}

/// Update vote tallies on a proposal based on a new vote.
/// Delegates to pure helper functions for testable logic.
fn update_proposal_vote_tallies(
    _proposal_hash: ActionHash,
    mut proposal: Proposal,
    vote_choice: &VoteChoice,
    vote_weight: Option<u64>,
) -> ExternResult<Proposal> {
    // Always update simple tallies (for backward compatibility and audit)
    let (f, a, ab) = apply_vote(
        proposal.for_votes,
        proposal.against_votes,
        proposal.abstain_votes,
        vote_choice,
    );
    proposal.for_votes = f;
    proposal.against_votes = a;
    proposal.abstain_votes = ab;

    // Update weighted tallies for non-Simple modes
    if let Some(weight) = vote_weight {
        match vote_choice {
            VoteChoice::For => {
                proposal.weighted_for = proposal.weighted_for.saturating_add(weight);
            }
            VoteChoice::Against => {
                proposal.weighted_against = proposal.weighted_against.saturating_add(weight);
            }
            VoteChoice::Abstain => {
                // Abstain doesn't affect weighted tallies
            }
        }
    }

    // Check if voting deadline has passed and update status if needed
    let now = chrono::Utc::now().timestamp();
    if now > proposal.voting_deadline && proposal.status == ProposalStatus::Active {
        proposal.status = determine_proposal_outcome(
            proposal.for_votes,
            proposal.against_votes,
            proposal.abstain_votes,
            50, // >50% threshold
        );
    }

    Ok(proposal)
}

/// Check if the given voter already cast a vote for a proposal
fn has_existing_vote(proposal_hash: &ActionHash, voter: &AgentPubKey) -> ExternResult<bool> {
    let links = get_links(
        LinkQuery::new(
            proposal_hash.clone(),
            LinkTypeFilter::single_type(0.into(), (LinkTypes::ProposalToVotes as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(vote) =
                        Vote::try_from(SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())))
                    {
                        if vote.voter == voter.to_string() {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

// ============================================================================
// Input/Output structures
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProposalInput {
    pub proposal_id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub category: ProposalCategory,
    pub actions: Vec<serde_json::Value>, // Serialized to JSON string in entry
    /// Voting mode (defaults to Simple for backward compatibility)
    pub voting_mode: Option<VotingMode>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CastVoteInput {
    pub proposal_id: String,
    pub proposal_hash: ActionHash,
    pub choice: VoteChoice,
    pub justification: Option<String>,
    /// Reputation to stake on this vote (permille, required for Quadratic mode)
    pub reputation_allocated: Option<u64>,
}

// ============================================================================
// Tests -- pure business logic only (no HDK required)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- determine_proposal_outcome ----

    #[test]
    fn test_proposal_passes_at_majority() {
        let status = determine_proposal_outcome(51, 49, 0, 50);
        assert_eq!(status, ProposalStatus::Approved);
    }

    #[test]
    fn test_proposal_fails_at_exactly_50_percent() {
        let status = determine_proposal_outcome(50, 50, 0, 50);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn test_proposal_passes_with_abstains() {
        let status = determine_proposal_outcome(6, 4, 100, 50);
        assert_eq!(status, ProposalStatus::Approved);
    }

    #[test]
    fn test_proposal_rejected_zero_votes() {
        let status = determine_proposal_outcome(0, 0, 0, 50);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn test_proposal_rejected_only_abstains() {
        let status = determine_proposal_outcome(0, 0, 10, 50);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn test_proposal_single_for_vote() {
        let status = determine_proposal_outcome(1, 0, 0, 50);
        assert_eq!(status, ProposalStatus::Approved);
    }

    #[test]
    fn test_proposal_single_against_vote() {
        let status = determine_proposal_outcome(0, 1, 0, 50);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    #[test]
    fn test_proposal_supermajority_threshold() {
        let status = determine_proposal_outcome(67, 33, 0, 66);
        assert_eq!(status, ProposalStatus::Approved);

        let status = determine_proposal_outcome(66, 34, 0, 66);
        assert_eq!(status, ProposalStatus::Rejected);
    }

    // ---- voting_deadline ----

    #[test]
    fn test_fast_deadline() {
        let created = 1_000_000i64;
        let deadline = voting_deadline(created, &ProposalType::Fast);
        assert_eq!(deadline, created + 48 * 3600);
    }

    #[test]
    fn test_normal_deadline() {
        let created = 1_000_000i64;
        let deadline = voting_deadline(created, &ProposalType::Normal);
        assert_eq!(deadline, created + 168 * 3600);
    }

    #[test]
    fn test_slow_deadline() {
        let created = 1_000_000i64;
        let deadline = voting_deadline(created, &ProposalType::Slow);
        assert_eq!(deadline, created + 336 * 3600);
    }

    #[test]
    fn test_deadline_from_zero() {
        let deadline = voting_deadline(0, &ProposalType::Fast);
        assert_eq!(deadline, 48 * 3600);
    }

    // ---- apply_vote ----

    #[test]
    fn test_apply_vote_for() {
        let (f, a, ab) = apply_vote(5, 3, 2, &VoteChoice::For);
        assert_eq!((f, a, ab), (6, 3, 2));
    }

    #[test]
    fn test_apply_vote_against() {
        let (f, a, ab) = apply_vote(5, 3, 2, &VoteChoice::Against);
        assert_eq!((f, a, ab), (5, 4, 2));
    }

    #[test]
    fn test_apply_vote_abstain() {
        let (f, a, ab) = apply_vote(5, 3, 2, &VoteChoice::Abstain);
        assert_eq!((f, a, ab), (5, 3, 3));
    }

    #[test]
    fn test_apply_vote_saturating() {
        let (f, _, _) = apply_vote(u32::MAX, 0, 0, &VoteChoice::For);
        assert_eq!(f, u32::MAX);
    }

    // ---- isqrt ----

    #[test]
    fn test_isqrt_known_values() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(400), 20);
        assert_eq!(isqrt(10000), 100);
    }

    #[test]
    fn test_isqrt_non_perfect_squares() {
        // isqrt floors to nearest integer
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(5), 2);
        assert_eq!(isqrt(99), 9);
        assert_eq!(isqrt(101), 10);
    }

    #[test]
    fn test_isqrt_large_values() {
        assert_eq!(isqrt(1_000_000), 1000);
        assert_eq!(isqrt(u64::MAX), 4294967295); // floor(sqrt(2^64 - 1))
    }

    // ---- quadratic voting weight ----

    #[test]
    fn test_quadratic_vote_weight() {
        // isqrt(100) = 10, isqrt(400) = 20
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(400), 20);
    }

    #[test]
    fn test_weighted_tallying() {
        // Two voters: rep 100 and 900
        // Weights: isqrt(100) = 10, isqrt(900) = 30
        let weight_a = isqrt(100);
        let weight_b = isqrt(900);
        assert_eq!(weight_a, 10);
        assert_eq!(weight_b, 30);

        // Total weighted_for if both vote For
        let total = weight_a + weight_b;
        assert_eq!(total, 40);
    }

    #[test]
    fn test_simple_mode_backward_compatible() {
        // In Simple mode, apply_vote works as before
        let (f, a, ab) = apply_vote(0, 0, 0, &VoteChoice::For);
        assert_eq!((f, a, ab), (1, 0, 0));
    }

    #[test]
    fn test_voting_mode_default_is_simple() {
        assert_eq!(VotingMode::default(), VotingMode::Simple);
    }

    #[test]
    fn test_voting_mode_enum_variants() {
        let simple = VotingMode::Simple;
        let quadratic = VotingMode::Quadratic;
        let conviction = VotingMode::Conviction;
        assert_ne!(simple, quadratic);
        assert_ne!(quadratic, conviction);
        assert_ne!(simple, conviction);
    }
}
