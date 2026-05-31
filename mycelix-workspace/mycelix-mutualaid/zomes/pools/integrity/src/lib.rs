// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Pools Integrity Zome - Community mutual aid pools
//!
//! This zome defines the data structures and validation rules for mutual aid
//! pools, contributions, and disbursements within the Mycelix network.

use hdi::prelude::*;

/// Anchor entry type for string-based link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Anchor(pub String);

impl Anchor {
    pub fn new(value: impl Into<String>) -> Self {
        Anchor(value.into())
    }
}

/// Rules for pool contributions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContributionRule {
    /// Minimum monthly contribution required (in smallest currency unit)
    pub min_monthly: u64,
    /// Maximum amount that can be withdrawn per request
    pub max_withdrawal: u64,
    /// Cooldown period in days between withdrawals
    pub cooldown_days: u32,
}

impl Default for ContributionRule {
    fn default() -> Self {
        ContributionRule {
            min_monthly: 0,
            max_withdrawal: u64::MAX,
            cooldown_days: 0,
        }
    }
}

/// Rules for pool disbursements
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisbursementRule {
    /// Minimum number of approvals required
    pub min_approvals: u32,
    /// Percentage of members that must approve (0-100)
    pub approval_threshold_percent: u8,
    /// Maximum disbursement per request
    pub max_disbursement: u64,
    /// Whether emergency disbursements bypass approvals
    pub allow_emergency_bypass: bool,
}

impl Default for DisbursementRule {
    fn default() -> Self {
        DisbursementRule {
            min_approvals: 1,
            approval_threshold_percent: 50,
            max_disbursement: u64::MAX,
            allow_emergency_bypass: false,
        }
    }
}

/// Pool status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolStatus {
    Active,
    Paused,
    Closed,
}

/// A mutual aid pool managed by a community
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MutualAidPool {
    /// Unique identifier for this pool
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of the pool's purpose
    pub description: String,
    /// List of member DIDs
    pub members: Vec<String>,
    /// Rules for contributions
    pub contribution_rules: ContributionRule,
    /// Rules for disbursements
    pub disbursement_rules: DisbursementRule,
    /// Current balance (in smallest currency unit)
    pub balance: u64,
    /// Pool status
    pub status: PoolStatus,
    /// Timestamp when pool was created
    pub created_at: Timestamp,
    /// Timestamp when pool was last updated
    pub updated_at: Timestamp,
}

/// A contribution to a mutual aid pool
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contribution {
    /// Unique identifier for this contribution
    pub id: String,
    /// Reference to the pool
    pub pool_id: String,
    /// DID of the contributing member
    pub member_did: String,
    /// Amount contributed (in smallest currency unit)
    pub amount: u64,
    /// Optional note from the contributor
    pub note: Option<String>,
    /// Timestamp of the contribution
    pub timestamp: Timestamp,
}

/// Status of a disbursement request
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisbursementStatus {
    Pending,
    Approved,
    Rejected,
    Completed,
    Cancelled,
}

/// A disbursement from a mutual aid pool
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Disbursement {
    /// Unique identifier for this disbursement
    pub id: String,
    /// Reference to the pool
    pub pool_id: String,
    /// DID of the recipient
    pub recipient_did: String,
    /// Amount requested/disbursed (in smallest currency unit)
    pub amount: u64,
    /// Reason for the disbursement
    pub reason: String,
    /// DIDs of members who approved
    pub approved_by: Vec<String>,
    /// DIDs of members who rejected
    pub rejected_by: Vec<String>,
    /// Current status
    pub status: DisbursementStatus,
    /// Whether this is an emergency request
    pub is_emergency: bool,
    /// Timestamp of the request
    pub requested_at: Timestamp,
    /// Timestamp when processed (if applicable)
    pub processed_at: Option<Timestamp>,
}

/// Pool membership record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PoolMembership {
    /// Pool ID
    pub pool_id: String,
    /// Member DID
    pub member_did: String,
    /// Role in the pool
    pub role: MemberRole,
    /// Timestamp of joining
    pub joined_at: Timestamp,
    /// Total contributed by this member
    pub total_contributed: u64,
    /// Total received by this member
    pub total_received: u64,
    /// Last contribution timestamp
    pub last_contribution: Option<Timestamp>,
    /// Last disbursement timestamp
    pub last_disbursement: Option<Timestamp>,
}

/// Member role within a pool
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberRole {
    Admin,
    Member,
    Observer,
}

/// All entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    MutualAidPool(MutualAidPool),
    #[entry_type(visibility = "public")]
    Contribution(Contribution),
    #[entry_type(visibility = "public")]
    Disbursement(Disbursement),
    #[entry_type(visibility = "public")]
    PoolMembership(PoolMembership),
}

/// Link types for connecting entries
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all pools
    AnchorToPool,
    /// Pool to its contributions
    PoolToContribution,
    /// Pool to its disbursements
    PoolToDisbursement,
    /// Pool to its memberships
    PoolToMembership,
    /// Member DID to their pool memberships
    MemberToMembership,
    /// Member DID to their contributions
    MemberToContribution,
    /// Member DID to their received disbursements
    MemberToDisbursement,
    /// Pool to pending disbursements
    PoolToPendingDisbursement,
}

/// Validation errors for pools zome
#[derive(Debug)]
pub enum PoolsError {
    InvalidDid(String),
    InvalidId(String),
    ZeroAmount,
    EmptyName,
    EmptyReason,
    InvalidApprovalThreshold(u8),
    InsufficientBalance,
    NotAMember,
}

impl std::fmt::Display for PoolsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDid(s) => write!(f, "Invalid DID format: {}", s),
            Self::InvalidId(s) => write!(f, "Invalid ID format: {}", s),
            Self::ZeroAmount => write!(f, "Amount cannot be zero"),
            Self::EmptyName => write!(f, "Pool name cannot be empty"),
            Self::EmptyReason => write!(f, "Reason cannot be empty"),
            Self::InvalidApprovalThreshold(t) => write!(f, "Invalid approval threshold: {}", t),
            Self::InsufficientBalance => write!(f, "Disbursement exceeds pool balance"),
            Self::NotAMember => write!(f, "Member not in pool"),
        }
    }
}

/// Validate that a DID has a valid format
fn validate_did(did: &str) -> ExternResult<()> {
    if did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            PoolsError::InvalidDid("DID cannot be empty".to_string()).to_string()
        )));
    }
    // Basic DID format check: did:method:identifier
    if !did.starts_with("did:") || did.split(':').count() < 3 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            PoolsError::InvalidDid(format!("Invalid DID format: {}", did)).to_string()
        )));
    }
    Ok(())
}

/// Validate that an ID is non-empty
fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
    if id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            PoolsError::InvalidId(format!("{} cannot be empty", field_name)).to_string()
        )));
    }
    Ok(())
}

/// Validate a MutualAidPool entry
fn validate_mutual_aid_pool(pool: &MutualAidPool) -> ExternResult<ValidateCallbackResult> {
    // Validate ID
    validate_id(&pool.id, "Pool ID")?;

    // Validate name is not empty
    if pool.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::EmptyName.to_string(),
        ));
    }

    // Validate all member DIDs
    for member in &pool.members {
        validate_did(member)?;
    }

    // Validate disbursement approval threshold
    if pool.disbursement_rules.approval_threshold_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::InvalidApprovalThreshold(
                pool.disbursement_rules.approval_threshold_percent,
            )
            .to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Contribution entry
fn validate_contribution(contribution: &Contribution) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&contribution.id, "Contribution ID")?;
    validate_id(&contribution.pool_id, "Pool ID")?;

    // Validate member DID
    validate_did(&contribution.member_did)?;

    // Validate amount is positive
    if contribution.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::ZeroAmount.to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Disbursement entry
fn validate_disbursement(disbursement: &Disbursement) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&disbursement.id, "Disbursement ID")?;
    validate_id(&disbursement.pool_id, "Pool ID")?;

    // Validate recipient DID
    validate_did(&disbursement.recipient_did)?;

    // Validate amount is positive
    if disbursement.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::ZeroAmount.to_string(),
        ));
    }

    // Validate reason is not empty
    if disbursement.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::EmptyReason.to_string(),
        ));
    }

    // Validate approver and rejector DIDs
    for approver in &disbursement.approved_by {
        validate_did(approver)?;
    }
    for rejector in &disbursement.rejected_by {
        validate_did(rejector)?;
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a PoolMembership entry
fn validate_pool_membership(membership: &PoolMembership) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&membership.pool_id, "Pool ID")?;

    // Validate member DID
    validate_did(&membership.member_did)?;

    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check callback
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::MutualAidPool(pool) => validate_mutual_aid_pool(&pool),
                    EntryTypes::Contribution(contribution) => validate_contribution(&contribution),
                    EntryTypes::Disbursement(disbursement) => validate_disbursement(&disbursement),
                    EntryTypes::PoolMembership(membership) => validate_pool_membership(&membership),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToPool
            | LinkTypes::PoolToContribution
            | LinkTypes::PoolToDisbursement
            | LinkTypes::PoolToMembership
            | LinkTypes::MemberToMembership
            | LinkTypes::MemberToContribution
            | LinkTypes::MemberToDisbursement
            | LinkTypes::PoolToPendingDisbursement => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToPool
            | LinkTypes::PoolToContribution
            | LinkTypes::PoolToDisbursement
            | LinkTypes::PoolToMembership
            | LinkTypes::MemberToMembership
            | LinkTypes::MemberToContribution
            | LinkTypes::MemberToDisbursement
            | LinkTypes::PoolToPendingDisbursement => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
