// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
 * Governance Record Zome
 *
 * Stores governance records on Holochain DHT:
 * - Proposals (MIPs, parameter changes, emergency actions)
 * - Votes (identity-weighted, quadratic voting)
 * - Execution records (audit trail)
 * - Guardian authorization requests and approvals
 *
 * Week 7-8 Phase 1: Governance Integration
 */

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ========================================
// Entry Types
// ========================================

/// Governance proposal entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Proposal {
    pub proposal_id: String,
    pub proposal_type: String,  // "PARAMETER_CHANGE", "PARTICIPANT_MANAGEMENT", "CAPABILITY_UPDATE", "ECONOMIC", "EMERGENCY_ACTION"
    pub title: String,
    pub description: String,
    pub proposer_did: String,
    pub proposer_participant_id: String,

    // Voting parameters
    pub voting_start: i64,      // Microseconds timestamp
    pub voting_end: i64,        // Microseconds timestamp
    pub quorum: f64,            // Minimum participation (0.0-1.0)
    pub approval_threshold: f64, // Required for approval (0.5-1.0)

    // Current status
    pub status: String,         // "DRAFT", "SUBMITTED", "VOTING", "APPROVED", "REJECTED", "EXECUTED"
    pub total_votes_for: f64,
    pub total_votes_against: f64,
    pub total_votes_abstain: f64,
    pub total_voting_power: f64,

    // Execution
    pub execution_params: String, // JSON-encoded parameters
    pub executed_at: Option<i64>,
    pub execution_result: String,  // JSON-encoded result

    // Metadata
    pub created_at: i64,
    pub updated_at: i64,
    pub tags: String,           // JSON-encoded array
}

/// Vote entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    pub vote_id: String,
    pub proposal_id: String,
    pub voter_did: String,
    pub voter_participant_id: String,
    pub choice: String,         // "FOR", "AGAINST", "ABSTAIN"
    pub credits_spent: i32,
    pub vote_weight: f64,       // Calculated at vote time
    pub effective_votes: f64,   // sqrt(credits_spent) for quadratic
    pub timestamp: i64,
    pub signature: String,      // Cryptographic signature
}

/// Execution record entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ExecutionRecord {
    pub execution_id: String,
    pub proposal_id: String,
    pub executed_by: String,    // DID of executor
    pub execution_type: String, // "AUTOMATIC", "MANUAL", "EMERGENCY"
    pub execution_params: String, // JSON-encoded parameters used
    pub execution_result: String, // JSON-encoded result
    pub success: bool,
    pub error_message: String,  // Empty if success
    pub executed_at: i64,
    pub metadata: String,       // JSON-encoded additional metadata
}

/// Guardian authorization request entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianAuthorizationRequest {
    pub request_id: String,
    pub subject_participant_id: String, // Who is requesting authorization
    pub action: String,                 // "EMERGENCY_STOP", "BAN_PARTICIPANT", etc.
    pub action_params: String,          // JSON-encoded action parameters
    pub required_threshold: f64,        // 0.0-1.0
    pub expires_at: i64,                // Microseconds timestamp
    pub status: String,                 // "PENDING", "APPROVED", "REJECTED", "EXPIRED"
    pub created_at: i64,
    pub updated_at: i64,
}

/// Guardian approval entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianApproval {
    pub approval_id: String,
    pub request_id: String,
    pub guardian_did: String,
    pub guardian_participant_id: String,
    pub approved: bool,         // true = approve, false = reject
    pub reasoning: String,      // Optional explanation
    pub timestamp: i64,
    pub signature: String,      // Cryptographic signature
}

// ========================================
// Link Types
// ========================================

#[hdk_link_types]
pub enum LinkTypes {
    ProposalToVotes,
    RequestToApprovals,
    ProposalsByStatus,
    ProposalsByType,
    VotesByVoter,
}

// ========================================
// Zome Functions: Proposal Management
// ========================================

/// Store a new proposal on DHT
#[hdk_extern]
pub fn store_proposal(proposal: Proposal) -> ExternResult<ActionHash> {
    let proposal_hash = create_entry(&EntryTypes::Proposal(proposal.clone()))?;

    // Create path-based link for proposal lookup by ID
    let proposal_path = Path::from(format!("proposal.{}", proposal.proposal_id));
    create_link(
        proposal_path.path_entry_hash()?,
        proposal_hash.clone(),
        LinkTypes::ProposalsByStatus,
        LinkTag::new(proposal.status.as_bytes()),
    )?;

    // Create link by proposal type
    let type_path = Path::from(format!("proposal_type.{}", proposal.proposal_type));
    create_link(
        type_path.path_entry_hash()?,
        proposal_hash.clone(),
        LinkTypes::ProposalsByType,
        LinkTag::new(proposal.proposal_type.as_bytes()),
    )?;

    Ok(proposal_hash)
}

/// Get a proposal by ID
#[hdk_extern]
pub fn get_proposal(proposal_id: String) -> ExternResult<Option<Proposal>> {
    let proposal_path = Path::from(format!("proposal.{}", proposal_id));
    let links = get_links(
        LinkQuery::try_new(
            proposal_path.path_entry_hash()?, LinkTypes::ProposalsByStatus,
        )?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the most recent link (should only be one)
    let link = &links[0];
    let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
    let proposal_record = get(target_hash, GetOptions::default())?;

    match proposal_record {
        Some(record) => {
            let proposal: Proposal = record.entry().to_app_option().map_err(|e| wasm_error!(e))?.ok_or(wasm_error!(
                WasmErrorInner::Guest("Failed to deserialize Proposal".into())
            ))?;
            Ok(Some(proposal))
        }
        None => Ok(None),
    }
}

/// List proposals by status
#[hdk_extern]
pub fn list_proposals_by_status(input: ListProposalsInput) -> ExternResult<Vec<Proposal>> {
    let status_path = Path::from(format!("proposal_status.{}", input.status));
    let links = get_links(
        LinkQuery::try_new(status_path.path_entry_hash()?, LinkTypes::ProposalsByStatus)?,
        GetStrategy::default()
    )?;

    let mut proposals = Vec::new();

    for link in links.iter().take(input.limit.unwrap_or(100) as usize) {
        let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
        if let Some(record) = get(target_hash, GetOptions::default())? {
            if let Some(proposal) = record.entry().to_app_option::<Proposal>().map_err(|e| wasm_error!(e))? {
                proposals.push(proposal);
            }
        }
    }

    Ok(proposals)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ListProposalsInput {
    pub status: String,
    pub limit: Option<i32>,
}

/// Update proposal status (used after vote tallying)
#[hdk_extern]
pub fn update_proposal_status(input: UpdateProposalStatusInput) -> ExternResult<ActionHash> {
    // Get existing proposal
    let existing_proposal = get_proposal(input.proposal_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Proposal not found".into())))?;

    // Create updated proposal
    let updated_proposal = Proposal {
        status: input.new_status.clone(),
        total_votes_for: input.total_votes_for,
        total_votes_against: input.total_votes_against,
        total_votes_abstain: input.total_votes_abstain,
        total_voting_power: input.total_voting_power,
        executed_at: input.executed_at,
        execution_result: input.execution_result.unwrap_or_default(),
        updated_at: input.updated_at,
        ..existing_proposal
    };

    // Store updated proposal
    let updated_hash = create_entry(&EntryTypes::Proposal(updated_proposal.clone()))?;

    // Update path links
    let proposal_path = Path::from(format!("proposal.{}", updated_proposal.proposal_id));
    create_link(
        proposal_path.path_entry_hash()?,
        updated_hash.clone(),
        LinkTypes::ProposalsByStatus,
        LinkTag::new(updated_proposal.status.as_bytes()),
    )?;

    Ok(updated_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateProposalStatusInput {
    pub proposal_id: String,
    pub new_status: String,
    pub total_votes_for: f64,
    pub total_votes_against: f64,
    pub total_votes_abstain: f64,
    pub total_voting_power: f64,
    pub executed_at: Option<i64>,
    pub execution_result: Option<String>,
    pub updated_at: i64,
}

// ========================================
// Zome Functions: Vote Management
// ========================================

/// Store a vote on DHT
#[hdk_extern]
pub fn store_vote(vote: Vote) -> ExternResult<ActionHash> {
    let vote_hash = create_entry(&EntryTypes::Vote(vote.clone()))?;

    // Create path-based link for votes by proposal
    let proposal_path = Path::from(format!("proposal.{}", vote.proposal_id));
    create_link(
        proposal_path.path_entry_hash()?,
        vote_hash.clone(),
        LinkTypes::ProposalToVotes,
        LinkTag::new(vote.voter_did.as_bytes()),
    )?;

    // Create link for votes by voter
    let voter_path = Path::from(format!("voter.{}", vote.voter_did));
    create_link(
        voter_path.path_entry_hash()?,
        vote_hash.clone(),
        LinkTypes::VotesByVoter,
        LinkTag::new(vote.proposal_id.as_bytes()),
    )?;

    Ok(vote_hash)
}

/// Get all votes for a proposal
#[hdk_extern]
pub fn get_votes(proposal_id: String) -> ExternResult<Vec<Vote>> {
    let proposal_path = Path::from(format!("proposal.{}", proposal_id));
    let links = get_links(
        LinkQuery::try_new(proposal_path.path_entry_hash()?, LinkTypes::ProposalToVotes)?,
        GetStrategy::default()
    )?;

    let mut votes = Vec::new();

    for link in links {
        let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
        if let Some(record) = get(target_hash, GetOptions::default())? {
            if let Some(vote) = record.entry().to_app_option::<Vote>().map_err(|e| wasm_error!(e))? {
                votes.push(vote);
            }
        }
    }

    Ok(votes)
}

/// Get votes by a specific voter
#[hdk_extern]
pub fn get_votes_by_voter(voter_did: String) -> ExternResult<Vec<Vote>> {
    let voter_path = Path::from(format!("voter.{}", voter_did));
    let links = get_links(
        LinkQuery::try_new(voter_path.path_entry_hash()?, LinkTypes::VotesByVoter)?,
        GetStrategy::default()
    )?;

    let mut votes = Vec::new();

    for link in links {
        let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
        if let Some(record) = get(target_hash, GetOptions::default())? {
            if let Some(vote) = record.entry().to_app_option::<Vote>().map_err(|e| wasm_error!(e))? {
                votes.push(vote);
            }
        }
    }

    Ok(votes)
}

// ========================================
// Zome Functions: Execution Records
// ========================================

/// Store an execution record
#[hdk_extern]
pub fn store_execution_record(record: ExecutionRecord) -> ExternResult<ActionHash> {
    let record_hash = create_entry(&EntryTypes::ExecutionRecord(record.clone()))?;

    // Create path-based link for execution record by proposal
    let proposal_path = Path::from(format!("proposal.{}.execution", record.proposal_id));
    create_link(
        proposal_path.path_entry_hash()?,
        record_hash.clone(),
        LinkTypes::ProposalToVotes,  // Reusing link type
        LinkTag::new(record.execution_id.as_bytes()),
    )?;

    Ok(record_hash)
}

/// Get execution record for a proposal
#[hdk_extern]
pub fn get_execution_record(proposal_id: String) -> ExternResult<Option<ExecutionRecord>> {
    let proposal_path = Path::from(format!("proposal.{}.execution", proposal_id));
    let links = get_links(
        LinkQuery::try_new(proposal_path.path_entry_hash()?, LinkTypes::ProposalToVotes)?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the most recent execution record
    let link = &links[0];
    let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
    let record_data = get(target_hash, GetOptions::default())?;

    match record_data {
        Some(record) => {
            let execution_record: ExecutionRecord = record.entry().to_app_option().map_err(|e| wasm_error!(e))?.ok_or(
                wasm_error!(WasmErrorInner::Guest(
                    "Failed to deserialize ExecutionRecord".into()
                )),
            )?;
            Ok(Some(execution_record))
        }
        None => Ok(None),
    }
}

// ========================================
// Zome Functions: Guardian Authorization
// ========================================

/// Store a guardian authorization request
#[hdk_extern]
pub fn store_authorization_request(
    request: GuardianAuthorizationRequest,
) -> ExternResult<ActionHash> {
    let request_hash = create_entry(&EntryTypes::GuardianAuthorizationRequest(request.clone()))?;

    // Create path-based link for request lookup by ID
    let request_path = Path::from(format!("auth_request.{}", request.request_id));
    create_link(
        request_path.path_entry_hash()?,
        request_hash.clone(),
        LinkTypes::RequestToApprovals,
        LinkTag::new(request.status.as_bytes()),
    )?;

    Ok(request_hash)
}

/// Get an authorization request by ID
#[hdk_extern]
pub fn get_authorization_request(
    request_id: String,
) -> ExternResult<Option<GuardianAuthorizationRequest>> {
    let request_path = Path::from(format!("auth_request.{}", request_id));
    let links = get_links(
        LinkQuery::try_new(request_path.path_entry_hash()?, LinkTypes::RequestToApprovals)?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let link = &links[0];
    let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
    let request_record = get(target_hash, GetOptions::default())?;

    match request_record {
        Some(record) => {
            let request: GuardianAuthorizationRequest =
                record.entry().to_app_option().map_err(|e| wasm_error!(e))?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Failed to deserialize GuardianAuthorizationRequest".into())
                ))?;
            Ok(Some(request))
        }
        None => Ok(None),
    }
}

/// Store a guardian approval
#[hdk_extern]
pub fn store_guardian_approval(approval: GuardianApproval) -> ExternResult<ActionHash> {
    let approval_hash = create_entry(&EntryTypes::GuardianApproval(approval.clone()))?;

    // Create link from request to approval
    let request_path = Path::from(format!("auth_request.{}", approval.request_id));
    create_link(
        request_path.path_entry_hash()?,
        approval_hash.clone(),
        LinkTypes::RequestToApprovals,
        LinkTag::new(approval.guardian_did.as_bytes()),
    )?;

    Ok(approval_hash)
}

/// Get all guardian approvals for a request
#[hdk_extern]
pub fn get_guardian_approvals(request_id: String) -> ExternResult<Vec<GuardianApproval>> {
    let request_path = Path::from(format!("auth_request.{}", request_id));
    let links = get_links(
        LinkQuery::try_new(request_path.path_entry_hash()?, LinkTypes::RequestToApprovals)?,
        GetStrategy::default()
    )?;

    let mut approvals = Vec::new();

    for link in links {
        let target_hash = link.target.clone().into_any_dht_hash().ok_or(wasm_error!(WasmErrorInner::Guest("Failed to convert link target".into())))?;
        if let Some(record) = get(target_hash, GetOptions::default())? {
            if let Some(approval) = record.entry().to_app_option::<GuardianApproval>().map_err(|e| wasm_error!(e))? {
                approvals.push(approval);
            }
        }
    }

    Ok(approvals)
}

/// Update authorization request status
#[hdk_extern]
pub fn update_authorization_status(input: UpdateAuthorizationStatusInput) -> ExternResult<ActionHash> {
    // Get existing request
    let existing_request = get_authorization_request(input.request_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Authorization request not found".into())))?;

    // Create updated request
    let updated_request = GuardianAuthorizationRequest {
        status: input.new_status.clone(),
        updated_at: input.updated_at,
        ..existing_request
    };

    // Store updated request
    let updated_hash = create_entry(&EntryTypes::GuardianAuthorizationRequest(updated_request.clone()))?;

    // Update path link
    let request_path = Path::from(format!("auth_request.{}", updated_request.request_id));
    create_link(
        request_path.path_entry_hash()?,
        updated_hash.clone(),
        LinkTypes::RequestToApprovals,
        LinkTag::new(updated_request.status.as_bytes()),
    )?;

    Ok(updated_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateAuthorizationStatusInput {
    pub request_id: String,
    pub new_status: String,
    pub updated_at: i64,
}

// ========================================
// Entry Type Definitions
// ========================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    Proposal(Proposal),
    #[entry_type]
    Vote(Vote),
    #[entry_type]
    ExecutionRecord(ExecutionRecord),
    #[entry_type]
    GuardianAuthorizationRequest(GuardianAuthorizationRequest),
    #[entry_type]
    GuardianApproval(GuardianApproval),
}
