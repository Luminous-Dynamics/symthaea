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
    EntryTypes, LinkTypes
};

/// Create a new governance proposal
#[hdk_extern]
pub fn create_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    // Get proposer agent info
    let agent_info = agent_info()?;
    let proposer_pubkey = agent_info.agent_initial_pubkey;

    // Calculate voting deadline based on proposal type
    let deadline_hours = match input.proposal_type {
        ProposalType::Fast => 48,    // 2 days
        ProposalType::Normal => 168, // 7 days
        ProposalType::Slow => 336,   // 14 days
    };

    let now = chrono::Utc::now().timestamp();
    let voting_deadline = now + (deadline_hours * 3600);

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

    // Create vote entry
    let vote = Vote {
        proposal_id: input.proposal_id.clone(),
        voter: voter_pubkey.to_string(),
        choice: input.choice,
        justification: input.justification,
        timestamp: chrono::Utc::now().timestamp(),
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
// Helper functions
// ============================================================================

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Update vote tallies on a proposal based on a new vote
fn update_proposal_vote_tallies(
    _proposal_hash: ActionHash,
    mut proposal: Proposal,
    vote_choice: &VoteChoice,
) -> ExternResult<Proposal> {
    // Increment the appropriate vote counter based on choice
    match vote_choice {
        VoteChoice::For => {
            proposal.for_votes = proposal.for_votes.saturating_add(1);
        }
        VoteChoice::Against => {
            proposal.against_votes = proposal.against_votes.saturating_add(1);
        }
        VoteChoice::Abstain => {
            proposal.abstain_votes = proposal.abstain_votes.saturating_add(1);
        }
    }

    // Check if voting deadline has passed and update status if needed
    let now = chrono::Utc::now().timestamp();
    if now > proposal.voting_deadline && proposal.status == ProposalStatus::Active {
        // Calculate total votes
        let total_votes = proposal.for_votes + proposal.against_votes + proposal.abstain_votes;
        let for_percentage = if total_votes > 0 {
            (proposal.for_votes as f64 / total_votes as f64) * 100.0
        } else {
            0.0
        };

        // Determine outcome: requires >50% for votes to pass
        if for_percentage > 50.0 {
            proposal.status = ProposalStatus::Approved;
        } else {
            proposal.status = ProposalStatus::Rejected;
        }
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
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CastVoteInput {
    pub proposal_id: String,
    pub proposal_hash: ActionHash,
    pub choice: VoteChoice,
    pub justification: Option<String>,
}

// ============================================================================
// Tests (host-side only; relies on coordinator logic)
// ============================================================================

// Note: host-side tests are omitted because Holochain mock helpers are not available in this crate.
