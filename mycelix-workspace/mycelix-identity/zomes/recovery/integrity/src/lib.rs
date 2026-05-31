// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Social Recovery Integrity Zome
//! Defines entry types and validation for DID social recovery
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// Recovery configuration for a DID
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryConfig {
    /// The DID being protected
    pub did: String,
    /// Owner's agent pub key
    pub owner: AgentPubKey,
    /// List of trustee DIDs
    pub trustees: Vec<String>,
    /// Minimum trustees required (threshold)
    pub threshold: u32,
    /// Time lock in seconds before recovery executes
    pub time_lock: u64,
    /// Whether recovery is currently active
    pub active: bool,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
}

/// A recovery request initiated by trustees
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryRequest {
    /// Request identifier
    pub id: String,
    /// DID being recovered
    pub did: String,
    /// New agent pub key to recover to
    pub new_agent: AgentPubKey,
    /// Initiating trustee's DID
    pub initiated_by: String,
    /// Reason for recovery
    pub reason: String,
    /// Current status
    pub status: RecoveryStatus,
    /// When the request was created
    pub created: Timestamp,
    /// When time lock expires (if approved)
    pub time_lock_expires: Option<Timestamp>,
}

/// Trustee vote on a recovery request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryVote {
    /// Recovery request ID
    pub request_id: String,
    /// Voting trustee's DID
    pub trustee: String,
    /// Vote decision
    pub vote: VoteDecision,
    /// Optional comment
    pub comment: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

/// Status of a recovery request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RecoveryStatus {
    /// Waiting for trustee votes
    Pending,
    /// Threshold reached, in time lock period
    Approved,
    /// Time lock expired, can execute
    ReadyToExecute,
    /// Recovery completed
    Completed,
    /// Recovery was rejected
    Rejected,
    /// Recovery was cancelled by owner
    Cancelled,
}

/// Trustee vote decision
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VoteDecision {
    Approve,
    Reject,
    Abstain,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    RecoveryConfig(RecoveryConfig),
    RecoveryRequest(RecoveryRequest),
    RecoveryVote(RecoveryVote),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// DID to recovery config
    DidToRecoveryConfig,
    /// DID to recovery requests
    DidToRecoveryRequest,
    /// Recovery request to votes
    RequestToVotes,
    /// Trustee to their responsibilities
    TrusteeToConfig,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::RecoveryConfig(config) => {
                    validate_create_recovery_config(EntryCreationAction::Create(action), config)
                }
                EntryTypes::RecoveryRequest(request) => {
                    validate_create_recovery_request(EntryCreationAction::Create(action), request)
                }
                EntryTypes::RecoveryVote(vote) => {
                    validate_create_recovery_vote(EntryCreationAction::Create(action), vote)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::RecoveryConfig(config) => {
                    validate_update_recovery_config(action, config)
                }
                EntryTypes::RecoveryRequest(request) => {
                    validate_update_recovery_request(action, request)
                }
                EntryTypes::RecoveryVote(_) => Ok(ValidateCallbackResult::Invalid(
                    "Recovery votes cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::DidToRecoveryConfig => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToRecoveryRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequestToVotes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TrusteeToConfig => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate recovery config creation
fn validate_create_recovery_config(
    action: EntryCreationAction,
    config: RecoveryConfig,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !config.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate owner is author
    if config.owner != *action.author() {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be the author".into(),
        ));
    }

    // Validate trustee count (3-7)
    if config.trustees.len() < 3 || config.trustees.len() > 7 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have 3-7 trustees".into(),
        ));
    }

    // Validate threshold
    let min_threshold = (config.trustees.len() as f64 * 0.5).ceil() as u32;
    if config.threshold < min_threshold {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Threshold must be at least {} (majority)",
            min_threshold
        )));
    }

    if config.threshold as usize > config.trustees.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold cannot exceed trustee count".into(),
        ));
    }

    // Validate time lock (minimum 24 hours)
    if config.time_lock < 86400 {
        return Ok(ValidateCallbackResult::Invalid(
            "Time lock must be at least 24 hours (86400 seconds)".into(),
        ));
    }

    // Validate all trustees are valid DIDs
    for trustee in &config.trustees {
        if !trustee.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid trustee DID: {}",
                trustee
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery config update
fn validate_update_recovery_config(
    _action: Update,
    config: RecoveryConfig,
) -> ExternResult<ValidateCallbackResult> {
    // Validate trustee count (3-7)
    if config.trustees.len() < 3 || config.trustees.len() > 7 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have 3-7 trustees".into(),
        ));
    }

    // Validate threshold
    let min_threshold = (config.trustees.len() as f64 * 0.5).ceil() as u32;
    if config.threshold < min_threshold {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Threshold must be at least {} (majority)",
            min_threshold
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery request creation
fn validate_create_recovery_request(
    _action: EntryCreationAction,
    request: RecoveryRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !request.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate initiator is a DID
    if !request.initiated_by.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Initiator must be a valid DID".into(),
        ));
    }

    // Validate reason provided
    if request.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery reason is required".into(),
        ));
    }

    // Validate initial status is Pending
    if request.status != RecoveryStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial status must be Pending".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery request update
fn validate_update_recovery_request(
    _action: Update,
    request: RecoveryRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !request.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery vote creation
fn validate_create_recovery_vote(
    _action: EntryCreationAction,
    vote: RecoveryVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate trustee is a DID
    if !vote.trustee.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Trustee must be a valid DID".into(),
        ));
    }

    // Validate request ID not empty
    if vote.request_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
