// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! HEARTH (Commons Pool) Integrity Zome
//!
//! Implements Commons Charter Article II, Section 2 - Commons Hearths
//!
//! Core Principles:
//! - Custodial resource governance: community-owned pools
//! - Democratic allocation: members vote on how resources flow
//! - Transparency: all contributions and allocations are public
//! - No individual ownership: resources belong to the commons
//!
//! Cultural Aliases: HEARTH, WELL, CAMPFIRE, COMMONS
//!
//! Constitutional Reference: Commons Charter v1.0, Article II, Section 2

use hdi::prelude::*;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Minimum contributors required to activate a pool
pub const MIN_CONTRIBUTORS_TO_ACTIVATE: u32 = 3;

/// Maximum allocation per request (as percentage of pool)
pub const MAX_ALLOCATION_PERCENT: f32 = 0.25; // 25% max per request

/// Voting period for allocation requests (in hours)
pub const DEFAULT_VOTING_PERIOD_HOURS: u32 = 72;

// =============================================================================
// ENTRY TYPES
// =============================================================================

/// A Commons Pool (HEARTH)
///
/// A shared resource pool owned by no individual but governed by the community.
/// Members contribute resources; the community decides allocations.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommonsPool {
    /// Unique identifier
    pub id: String,

    /// Name of the pool
    pub name: String,

    /// Description and purpose
    pub description: String,

    /// The DAO that governs this pool
    pub dao_did: String,

    /// Type of resources in this pool
    pub resource_type: ResourceType,

    /// Cultural alias (HEARTH, WELL, etc.)
    pub cultural_alias: Option<String>,

    /// Current total in the pool
    pub total_resources: f64,

    /// Total ever contributed (lifetime)
    pub lifetime_contributions: f64,

    /// Total ever allocated (lifetime)
    pub lifetime_allocations: f64,

    /// Number of contributors
    pub contributor_count: u32,

    /// Whether the pool is active
    pub status: PoolStatus,

    /// Governance rules for this pool
    pub governance: PoolGovernance,

    /// When created
    pub created: Timestamp,
}

/// Type of resources in the pool
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// FLOW tokens
    FlowTokens,
    /// Time credits (TEND)
    TimeCredits,
    /// Compute resources
    ComputeCredits,
    /// Storage resources
    StorageCredits,
    /// External currency (stablecoins)
    ExternalCurrency(String),
    /// Mixed/general purpose
    General,
    /// Custom resource type
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PoolStatus {
    /// Pool is proposed, gathering initial contributors
    Proposed,
    /// Pool is active and accepting contributions/requests
    Active,
    /// Pool is paused (no new activity)
    Paused,
    /// Pool is being wound down
    WindingDown,
    /// Pool is closed
    Closed,
}

/// Governance rules for a pool
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PoolGovernance {
    /// Voting threshold for allocations (0.0-1.0)
    pub allocation_threshold: f32,

    /// Voting period in hours
    pub voting_period_hours: u32,

    /// Whether contributors get weighted votes
    pub contribution_weighted_voting: bool,

    /// Minimum CIV required to request allocation
    pub min_civ_to_request: Option<f32>,

    /// Maximum single allocation as percent of pool
    pub max_allocation_percent: f32,

    /// Whether allocations require reason/justification
    pub require_justification: bool,

    /// Cool-down period between requests (hours)
    pub request_cooldown_hours: u32,
}

/// A contribution to a commons pool
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PoolContribution {
    /// Unique identifier
    pub id: String,

    /// Pool being contributed to
    pub pool_id: String,

    /// DID of the contributor
    pub contributor_did: String,

    /// Amount contributed
    pub amount: f64,

    /// Optional message/dedication
    pub message: Option<String>,

    /// Whether this is anonymous (contributor_did still recorded but hidden in UI)
    pub anonymous: bool,

    /// When contributed
    pub timestamp: Timestamp,
}

/// A request to allocate from a commons pool
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AllocationRequest {
    /// Unique identifier
    pub id: String,

    /// Pool to allocate from
    pub pool_id: String,

    /// DID of the requester
    pub requester_did: String,

    /// DID of the recipient (may be different from requester)
    pub recipient_did: String,

    /// Amount requested
    pub amount: f64,

    /// Justification for the request
    pub justification: String,

    /// Category of need
    pub need_category: NeedCategory,

    /// Status of the request
    pub status: RequestStatus,

    /// Voting deadline
    pub voting_deadline: Timestamp,

    /// Vote tally
    pub votes_for: u32,
    pub votes_against: u32,
    pub total_vote_weight: f64,

    /// When requested
    pub created: Timestamp,

    /// When resolved (if resolved)
    pub resolved: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NeedCategory {
    /// Emergency/urgent need
    Emergency,
    /// Medical/health
    Medical,
    /// Housing/shelter
    Housing,
    /// Food security
    Food,
    /// Education
    Education,
    /// Community project
    CommunityProject,
    /// Infrastructure
    Infrastructure,
    /// Care work support
    CareSupport,
    /// Creative/cultural
    Cultural,
    /// General need
    General,
    /// Custom category
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestStatus {
    /// Open for voting
    Voting,
    /// Approved by community
    Approved,
    /// Rejected by community
    Rejected,
    /// Withdrawn by requester
    Withdrawn,
    /// Funds disbursed
    Disbursed,
    /// Expired (voting period ended without quorum)
    Expired,
}

/// A vote on an allocation request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AllocationVote {
    /// The request being voted on
    pub request_id: String,

    /// DID of the voter
    pub voter_did: String,

    /// Vote (true = for, false = against)
    pub vote: bool,

    /// Vote weight (based on contribution if weighted)
    pub weight: f64,

    /// Optional comment
    pub comment: Option<String>,

    /// When voted
    pub timestamp: Timestamp,
}

/// Record of an allocation being disbursed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Disbursement {
    /// The request that was approved
    pub request_id: String,

    /// Pool it came from
    pub pool_id: String,

    /// Recipient DID
    pub recipient_did: String,

    /// Amount disbursed
    pub amount: f64,

    /// Transaction/proof hash
    pub transaction_hash: Option<String>,

    /// When disbursed
    pub timestamp: Timestamp,
}

// =============================================================================
// ENTRY & LINK TYPE ENUMS
// =============================================================================

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CommonsPool(CommonsPool),
    PoolContribution(PoolContribution),
    AllocationRequest(AllocationRequest),
    AllocationVote(AllocationVote),
    Disbursement(Disbursement),
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from DAO to its pools
    DaoToPools,
    /// Link from pool to its contributions
    PoolToContributions,
    /// Link from pool to allocation requests
    PoolToRequests,
    /// Link from request to votes
    RequestToVotes,
    /// Link from contributor to their contributions
    ContributorToContributions,
    /// Link from member to their requests
    MemberToRequests,
    /// Link from pool to disbursements
    PoolToDisbursements,
    /// Link type for anchor/path infrastructure
    AnchorLinks,
}

// =============================================================================
// VALIDATION
// =============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::CommonsPool(pool) => {
                        validate_create_pool(EntryCreationAction::Create(action), pool)
                    }
                    EntryTypes::PoolContribution(contribution) => validate_create_contribution(
                        EntryCreationAction::Create(action),
                        contribution,
                    ),
                    EntryTypes::AllocationRequest(request) => {
                        validate_create_request(EntryCreationAction::Create(action), request)
                    }
                    EntryTypes::AllocationVote(vote) => {
                        validate_create_vote(EntryCreationAction::Create(action), vote)
                    }
                    EntryTypes::Disbursement(disbursement) => validate_create_disbursement(
                        EntryCreationAction::Create(action),
                        disbursement,
                    ),
                    // Anchors are always valid (just hash placeholders)
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => {
                match app_entry {
                    EntryTypes::CommonsPool(pool) => validate_update_pool(action, pool),
                    EntryTypes::PoolContribution(_) => {
                        // Contributions are immutable
                        Ok(ValidateCallbackResult::Invalid(
                            "Contributions cannot be updated".into(),
                        ))
                    }
                    EntryTypes::AllocationRequest(request) => {
                        validate_update_request(action, request)
                    }
                    EntryTypes::AllocationVote(_) => {
                        // Votes are immutable (can't change your vote)
                        Ok(ValidateCallbackResult::Invalid(
                            "Votes cannot be changed".into(),
                        ))
                    }
                    EntryTypes::Disbursement(_) => Ok(ValidateCallbackResult::Invalid(
                        "Disbursements cannot be updated".into(),
                    )),
                    // Anchors cannot be updated
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                        "Anchors cannot be updated".into(),
                    )),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::DaoToPools => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PoolToContributions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PoolToRequests => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequestToVotes => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributorToContributions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToRequests => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PoolToDisbursements => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AnchorLinks => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_pool(
    _action: EntryCreationAction,
    pool: CommonsPool,
) -> ExternResult<ValidateCallbackResult> {
    if !pool.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DAO must be a valid DID".into(),
        ));
    }
    if pool.name.is_empty() || pool.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool name must be 1-100 chars".into(),
        ));
    }
    if pool.description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 2000 chars)".into(),
        ));
    }

    // Validate governance rules
    if pool.governance.allocation_threshold < 0.0 || pool.governance.allocation_threshold > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Allocation threshold must be 0.0-1.0".into(),
        ));
    }
    if pool.governance.max_allocation_percent <= 0.0 || pool.governance.max_allocation_percent > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Max allocation percent must be 0.0-1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_pool(
    _action: Update,
    pool: CommonsPool,
) -> ExternResult<ValidateCallbackResult> {
    // Pool can be updated (status changes, governance updates via DAO vote)
    if pool.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Pool name required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_contribution(
    _action: EntryCreationAction,
    contribution: PoolContribution,
) -> ExternResult<ValidateCallbackResult> {
    if !contribution.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }
    if contribution.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution must be positive".into(),
        ));
    }
    if let Some(msg) = &contribution.message {
        if msg.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid(
                "Message too long (max 500 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_request(
    _action: EntryCreationAction,
    request: AllocationRequest,
) -> ExternResult<ValidateCallbackResult> {
    if !request.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }
    if !request.recipient_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient must be a valid DID".into(),
        ));
    }
    if request.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Request amount must be positive".into(),
        ));
    }
    if request.justification.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Justification is required".into(),
        ));
    }
    if request.justification.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Justification too long (max 2000 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(
    _action: Update,
    request: AllocationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Request can be updated (status, vote tallies)
    if request.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_vote(
    _action: EntryCreationAction,
    vote: AllocationVote,
) -> ExternResult<ValidateCallbackResult> {
    if !vote.voter_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }
    if vote.weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote weight cannot be negative".into(),
        ));
    }
    if let Some(comment) = &vote.comment {
        if comment.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid(
                "Comment too long (max 500 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_disbursement(
    _action: EntryCreationAction,
    disbursement: Disbursement,
) -> ExternResult<ValidateCallbackResult> {
    if !disbursement.recipient_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient must be a valid DID".into(),
        ));
    }
    if disbursement.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Disbursement must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
