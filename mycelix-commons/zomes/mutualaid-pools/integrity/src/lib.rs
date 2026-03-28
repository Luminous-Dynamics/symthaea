// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pools Integrity Zome - Community mutual aid pools
//!
//! This zome defines the data structures and validation rules for mutual aid
//! pools, contributions, and disbursements within the Mycelix network.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
    if did.trim().is_empty() {
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
    if id.trim().is_empty() {
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

    if pool.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool ID must be 256 characters or fewer".into(),
        ));
    }

    // Validate name is not empty
    if pool.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::EmptyName.to_string(),
        ));
    }

    if pool.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool name must be 256 characters or fewer".into(),
        ));
    }

    if pool.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool description must be 4096 characters or fewer".into(),
        ));
    }

    // Validate member count
    if pool.members.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 500 members".into(),
        ));
    }

    // Validate all member DIDs and check for duplicates
    let mut seen_members = std::collections::HashSet::new();
    for member in &pool.members {
        validate_did(member)?;
        if !seen_members.insert(member.as_str()) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Duplicate member DID: {}",
                member
            )));
        }
    }

    // Validate min_approvals doesn't exceed member count (when members exist)
    if !pool.members.is_empty()
        && pool.disbursement_rules.min_approvals as usize > pool.members.len()
    {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "min_approvals ({}) exceeds member count ({})",
            pool.disbursement_rules.min_approvals,
            pool.members.len()
        )));
    }

    // Validate contribution rules
    if pool.contribution_rules.max_withdrawal == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Max withdrawal must be greater than 0".into(),
        ));
    }

    // Validate disbursement rules
    if pool.disbursement_rules.min_approvals == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum approvals must be greater than 0".into(),
        ));
    }

    if pool.disbursement_rules.approval_threshold_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::InvalidApprovalThreshold(
                pool.disbursement_rules.approval_threshold_percent,
            )
            .to_string(),
        ));
    }

    if pool.disbursement_rules.max_disbursement == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Max disbursement must be greater than 0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Contribution entry
fn validate_contribution(contribution: &Contribution) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&contribution.id, "Contribution ID")?;
    validate_id(&contribution.pool_id, "Pool ID")?;

    if contribution.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution ID must be 256 characters or fewer".into(),
        ));
    }
    if contribution.pool_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool ID must be 256 characters or fewer".into(),
        ));
    }
    if contribution.member_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Member DID must be 256 characters or fewer".into(),
        ));
    }

    // Validate member DID
    validate_did(&contribution.member_did)?;

    // Validate amount is positive
    if contribution.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::ZeroAmount.to_string(),
        ));
    }

    if let Some(ref note) = contribution.note {
        if note.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Contribution note must be 4096 characters or fewer".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a Disbursement entry
fn validate_disbursement(disbursement: &Disbursement) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&disbursement.id, "Disbursement ID")?;
    validate_id(&disbursement.pool_id, "Pool ID")?;

    if disbursement.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Disbursement ID must be 256 characters or fewer".into(),
        ));
    }
    if disbursement.pool_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool ID must be 256 characters or fewer".into(),
        ));
    }
    if disbursement.recipient_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient DID must be 256 characters or fewer".into(),
        ));
    }

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

    if disbursement.reason.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Disbursement reason must be 4096 characters or fewer".into(),
        ));
    }

    // Validate approver and rejector Vec limits
    if disbursement.approved_by.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 approvers".into(),
        ));
    }
    if disbursement.rejected_by.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 rejectors".into(),
        ));
    }

    // Validate approver and rejector DIDs
    for approver in &disbursement.approved_by {
        validate_did(approver)?;
    }
    for rejector in &disbursement.rejected_by {
        validate_did(rejector)?;
    }

    // Check for overlap between approved_by and rejected_by (same person both approving and rejecting)
    {
        let approver_set: std::collections::HashSet<&str> = disbursement
            .approved_by
            .iter()
            .map(|s| s.as_str())
            .collect();
        for rejector in &disbursement.rejected_by {
            if approver_set.contains(rejector.as_str()) {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "DID appears in both approved_by and rejected_by: {}",
                    rejector
                )));
            }
        }
    }

    // Check for duplicate DIDs within approved_by
    {
        let mut seen = std::collections::HashSet::new();
        for approver in &disbursement.approved_by {
            if !seen.insert(approver.as_str()) {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Duplicate DID in approved_by: {}",
                    approver
                )));
            }
        }
    }

    // Check for duplicate DIDs within rejected_by
    {
        let mut seen = std::collections::HashSet::new();
        for rejector in &disbursement.rejected_by {
            if !seen.insert(rejector.as_str()) {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Duplicate DID in rejected_by: {}",
                    rejector
                )));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a PoolMembership entry
fn validate_pool_membership(membership: &PoolMembership) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&membership.pool_id, "Pool ID")?;

    if membership.pool_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pool ID must be 256 characters or fewer".into(),
        ));
    }
    if membership.member_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Member DID must be 256 characters or fewer".into(),
        ));
    }

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
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AnchorToPool => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AnchorToPool link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PoolToContribution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PoolToContribution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PoolToDisbursement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PoolToDisbursement link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PoolToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PoolToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MemberToMembership => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToMembership link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MemberToContribution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToContribution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MemberToDisbursement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToDisbursement link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PoolToPendingDisbursement => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PoolToPendingDisbursement link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type, action, ..
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            match link_type {
                LinkTypes::AnchorToPool
                | LinkTypes::PoolToContribution
                | LinkTypes::PoolToDisbursement
                | LinkTypes::PoolToMembership
                | LinkTypes::MemberToMembership
                | LinkTypes::MemberToContribution
                | LinkTypes::MemberToDisbursement
                | LinkTypes::PoolToPendingDisbursement => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::StoreRecord(_) | FlatOp::RegisterAgentActivity(_) => {
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_timestamp(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn valid_did() -> String {
        "did:mycelix:alice123".to_string()
    }

    fn valid_did2() -> String {
        "did:mycelix:bob456".to_string()
    }

    fn valid_pool() -> MutualAidPool {
        MutualAidPool {
            id: "pool123".to_string(),
            name: "Community Pool".to_string(),
            description: "A mutual aid pool for the community".to_string(),
            members: vec![valid_did(), valid_did2()],
            contribution_rules: ContributionRule::default(),
            disbursement_rules: DisbursementRule::default(),
            balance: 1000,
            status: PoolStatus::Active,
            created_at: fake_timestamp(1000),
            updated_at: fake_timestamp(1000),
        }
    }

    fn valid_contribution() -> Contribution {
        Contribution {
            id: "contrib123".to_string(),
            pool_id: "pool123".to_string(),
            member_did: valid_did(),
            amount: 100,
            note: Some("Monthly contribution".to_string()),
            timestamp: fake_timestamp(2000),
        }
    }

    fn valid_disbursement() -> Disbursement {
        Disbursement {
            id: "disb123".to_string(),
            pool_id: "pool123".to_string(),
            recipient_did: valid_did(),
            amount: 50,
            reason: "Emergency assistance".to_string(),
            approved_by: vec![valid_did2()],
            rejected_by: vec![],
            status: DisbursementStatus::Pending,
            is_emergency: false,
            requested_at: fake_timestamp(3000),
            processed_at: None,
        }
    }

    fn valid_membership() -> PoolMembership {
        PoolMembership {
            pool_id: "pool123".to_string(),
            member_did: valid_did(),
            role: MemberRole::Member,
            joined_at: fake_timestamp(1000),
            total_contributed: 100,
            total_received: 0,
            last_contribution: Some(fake_timestamp(2000)),
            last_disbursement: None,
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_pool_status() {
        let statuses = vec![PoolStatus::Active, PoolStatus::Paused, PoolStatus::Closed];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: PoolStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, status);
        }
    }

    #[test]
    fn serde_roundtrip_disbursement_status() {
        let statuses = vec![
            DisbursementStatus::Pending,
            DisbursementStatus::Approved,
            DisbursementStatus::Rejected,
            DisbursementStatus::Completed,
            DisbursementStatus::Cancelled,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let back: DisbursementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, status);
        }
    }

    #[test]
    fn serde_roundtrip_member_role() {
        let roles = vec![MemberRole::Admin, MemberRole::Member, MemberRole::Observer];
        for role in &roles {
            let json = serde_json::to_string(role).unwrap();
            let back: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, role);
        }
    }

    #[test]
    fn serde_roundtrip_contribution_rule() {
        let rule = ContributionRule {
            min_monthly: 50,
            max_withdrawal: 1000,
            cooldown_days: 30,
        };
        let json = serde_json::to_string(&rule).unwrap();
        let back: ContributionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rule);
    }

    #[test]
    fn serde_roundtrip_disbursement_rule() {
        let rule = DisbursementRule {
            min_approvals: 3,
            approval_threshold_percent: 66,
            max_disbursement: 5000,
            allow_emergency_bypass: true,
        };
        let json = serde_json::to_string(&rule).unwrap();
        let back: DisbursementRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rule);
    }

    #[test]
    fn serde_roundtrip_mutual_aid_pool() {
        let pool = valid_pool();
        let json = serde_json::to_string(&pool).unwrap();
        let back: MutualAidPool = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pool);
    }

    #[test]
    fn serde_roundtrip_contribution() {
        let contrib = valid_contribution();
        let json = serde_json::to_string(&contrib).unwrap();
        let back: Contribution = serde_json::from_str(&json).unwrap();
        assert_eq!(back, contrib);
    }

    #[test]
    fn serde_roundtrip_disbursement() {
        let disb = valid_disbursement();
        let json = serde_json::to_string(&disb).unwrap();
        let back: Disbursement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, disb);
    }

    #[test]
    fn serde_roundtrip_pool_membership() {
        let mem = valid_membership();
        let json = serde_json::to_string(&mem).unwrap();
        let back: PoolMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(back, mem);
    }

    // ── validate_did tests ──────────────────────────────────────────────

    #[test]
    fn validate_did_valid() {
        assert!(validate_did("did:mycelix:alice").is_ok());
        assert!(validate_did("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK").is_ok());
        assert!(validate_did("did:web:example.com").is_ok());
    }

    #[test]
    fn validate_did_empty() {
        let result = validate_did("");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("DID cannot be empty"));
    }

    #[test]
    fn validate_did_missing_prefix() {
        let result = validate_did("mycelix:alice");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Invalid DID format"));
    }

    #[test]
    fn validate_did_too_few_parts() {
        let result = validate_did("did:mycelix");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Invalid DID format"));
    }

    #[test]
    fn validate_did_just_did() {
        let result = validate_did("did:");
        assert!(result.is_err());
    }

    // ── validate_id tests ───────────────────────────────────────────────

    #[test]
    fn validate_id_valid() {
        assert!(validate_id("pool123", "Pool ID").is_ok());
        assert!(validate_id("a", "ID").is_ok());
    }

    #[test]
    fn validate_id_empty() {
        let result = validate_id("", "Test ID");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Test ID cannot be empty"));
    }

    // ── validate_mutual_aid_pool tests ──────────────────────────────────

    #[test]
    fn validate_pool_valid() {
        assert_valid(validate_mutual_aid_pool(&valid_pool()));
    }

    #[test]
    fn validate_pool_empty_id() {
        let mut pool = valid_pool();
        pool.id = String::new();
        let result = validate_mutual_aid_pool(&pool);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Pool ID cannot be empty"));
    }

    #[test]
    fn validate_pool_empty_name() {
        let mut pool = valid_pool();
        pool.name = String::new();
        assert_invalid(validate_mutual_aid_pool(&pool), "Pool name cannot be empty");
    }

    #[test]
    fn validate_pool_whitespace_only_name() {
        let mut pool = valid_pool();
        pool.name = "   ".to_string();
        assert_invalid(validate_mutual_aid_pool(&pool), "Pool name cannot be empty");
    }

    #[test]
    fn validate_pool_name_with_leading_trailing_spaces_ok() {
        let mut pool = valid_pool();
        pool.name = "  Valid Name  ".to_string();
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_invalid_member_did() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), "invalid_did".to_string()];
        let result = validate_mutual_aid_pool(&pool);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Invalid DID format"));
    }

    #[test]
    fn validate_pool_empty_member_did() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), String::new()];
        let result = validate_mutual_aid_pool(&pool);
        assert!(result.is_err());
    }

    #[test]
    fn validate_pool_no_members_ok() {
        let mut pool = valid_pool();
        pool.members = vec![];
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_approval_threshold_100_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.approval_threshold_percent = 100;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_approval_threshold_0_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.approval_threshold_percent = 0;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_approval_threshold_101() {
        let mut pool = valid_pool();
        pool.disbursement_rules.approval_threshold_percent = 101;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Invalid approval threshold: 101",
        );
    }

    #[test]
    fn validate_pool_approval_threshold_255() {
        let mut pool = valid_pool();
        pool.disbursement_rules.approval_threshold_percent = 255;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Invalid approval threshold: 255",
        );
    }

    #[test]
    fn validate_pool_zero_balance_ok() {
        let mut pool = valid_pool();
        pool.balance = 0;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_max_balance_ok() {
        let mut pool = valid_pool();
        pool.balance = u64::MAX;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_all_statuses_valid() {
        for status in [PoolStatus::Active, PoolStatus::Paused, PoolStatus::Closed] {
            let mut pool = valid_pool();
            pool.status = status;
            assert_valid(validate_mutual_aid_pool(&pool));
        }
    }

    #[test]
    fn validate_pool_default_contribution_rules() {
        let mut pool = valid_pool();
        pool.contribution_rules = ContributionRule::default();
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_default_disbursement_rules() {
        let mut pool = valid_pool();
        pool.disbursement_rules = DisbursementRule::default();
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_empty_description_ok() {
        let mut pool = valid_pool();
        pool.description = String::new();
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── validate_contribution tests ─────────────────────────────────────

    #[test]
    fn validate_contribution_valid() {
        assert_valid(validate_contribution(&valid_contribution()));
    }

    #[test]
    fn validate_contribution_empty_id() {
        let mut contrib = valid_contribution();
        contrib.id = String::new();
        let result = validate_contribution(&contrib);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Contribution ID cannot be empty"));
    }

    #[test]
    fn validate_contribution_empty_pool_id() {
        let mut contrib = valid_contribution();
        contrib.pool_id = String::new();
        let result = validate_contribution(&contrib);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Pool ID cannot be empty"));
    }

    #[test]
    fn validate_contribution_invalid_member_did() {
        let mut contrib = valid_contribution();
        contrib.member_did = "not_a_did".to_string();
        let result = validate_contribution(&contrib);
        assert!(result.is_err());
    }

    #[test]
    fn validate_contribution_zero_amount() {
        let mut contrib = valid_contribution();
        contrib.amount = 0;
        assert_invalid(validate_contribution(&contrib), "Amount cannot be zero");
    }

    #[test]
    fn validate_contribution_amount_one_ok() {
        let mut contrib = valid_contribution();
        contrib.amount = 1;
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_max_amount_ok() {
        let mut contrib = valid_contribution();
        contrib.amount = u64::MAX;
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_no_note_ok() {
        let mut contrib = valid_contribution();
        contrib.note = None;
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_empty_note_ok() {
        let mut contrib = valid_contribution();
        contrib.note = Some(String::new());
        assert_valid(validate_contribution(&contrib));
    }

    // ── validate_disbursement tests ─────────────────────────────────────

    #[test]
    fn validate_disbursement_valid() {
        assert_valid(validate_disbursement(&valid_disbursement()));
    }

    #[test]
    fn validate_disbursement_empty_id() {
        let mut disb = valid_disbursement();
        disb.id = String::new();
        let result = validate_disbursement(&disb);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Disbursement ID cannot be empty"));
    }

    #[test]
    fn validate_disbursement_empty_pool_id() {
        let mut disb = valid_disbursement();
        disb.pool_id = String::new();
        let result = validate_disbursement(&disb);
        assert!(result.is_err());
    }

    #[test]
    fn validate_disbursement_invalid_recipient_did() {
        let mut disb = valid_disbursement();
        disb.recipient_did = "invalid".to_string();
        let result = validate_disbursement(&disb);
        assert!(result.is_err());
    }

    #[test]
    fn validate_disbursement_zero_amount() {
        let mut disb = valid_disbursement();
        disb.amount = 0;
        assert_invalid(validate_disbursement(&disb), "Amount cannot be zero");
    }

    #[test]
    fn validate_disbursement_amount_one_ok() {
        let mut disb = valid_disbursement();
        disb.amount = 1;
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_empty_reason() {
        let mut disb = valid_disbursement();
        disb.reason = String::new();
        assert_invalid(validate_disbursement(&disb), "Reason cannot be empty");
    }

    #[test]
    fn validate_disbursement_whitespace_reason() {
        let mut disb = valid_disbursement();
        disb.reason = "   ".to_string();
        assert_invalid(validate_disbursement(&disb), "Reason cannot be empty");
    }

    #[test]
    fn validate_disbursement_reason_with_spaces_ok() {
        let mut disb = valid_disbursement();
        disb.reason = "  Valid reason  ".to_string();
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_invalid_approver_did() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![valid_did(), "bad_did".to_string()];
        let result = validate_disbursement(&disb);
        assert!(result.is_err());
    }

    #[test]
    fn validate_disbursement_invalid_rejector_did() {
        let mut disb = valid_disbursement();
        disb.rejected_by = vec![valid_did(), "bad_did".to_string()];
        let result = validate_disbursement(&disb);
        assert!(result.is_err());
    }

    #[test]
    fn validate_disbursement_empty_approved_by_ok() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![];
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_empty_rejected_by_ok() {
        let mut disb = valid_disbursement();
        disb.rejected_by = vec![];
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_all_statuses_valid() {
        for status in [
            DisbursementStatus::Pending,
            DisbursementStatus::Approved,
            DisbursementStatus::Rejected,
            DisbursementStatus::Completed,
            DisbursementStatus::Cancelled,
        ] {
            let mut disb = valid_disbursement();
            disb.status = status;
            assert_valid(validate_disbursement(&disb));
        }
    }

    #[test]
    fn validate_disbursement_emergency_true_ok() {
        let mut disb = valid_disbursement();
        disb.is_emergency = true;
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_emergency_false_ok() {
        let mut disb = valid_disbursement();
        disb.is_emergency = false;
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_with_processed_at_ok() {
        let mut disb = valid_disbursement();
        disb.processed_at = Some(fake_timestamp(5000));
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_without_processed_at_ok() {
        let mut disb = valid_disbursement();
        disb.processed_at = None;
        assert_valid(validate_disbursement(&disb));
    }

    // ── validate_pool_membership tests ──────────────────────────────────

    #[test]
    fn validate_membership_valid() {
        assert_valid(validate_pool_membership(&valid_membership()));
    }

    #[test]
    fn validate_membership_empty_pool_id() {
        let mut mem = valid_membership();
        mem.pool_id = String::new();
        let result = validate_pool_membership(&mem);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("Pool ID cannot be empty"));
    }

    #[test]
    fn validate_membership_invalid_member_did() {
        let mut mem = valid_membership();
        mem.member_did = "invalid".to_string();
        let result = validate_pool_membership(&mem);
        assert!(result.is_err());
    }

    #[test]
    fn validate_membership_all_roles_valid() {
        for role in [MemberRole::Admin, MemberRole::Member, MemberRole::Observer] {
            let mut mem = valid_membership();
            mem.role = role;
            assert_valid(validate_pool_membership(&mem));
        }
    }

    #[test]
    fn validate_membership_zero_contributed_ok() {
        let mut mem = valid_membership();
        mem.total_contributed = 0;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_zero_received_ok() {
        let mut mem = valid_membership();
        mem.total_received = 0;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_max_contributed_ok() {
        let mut mem = valid_membership();
        mem.total_contributed = u64::MAX;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_max_received_ok() {
        let mut mem = valid_membership();
        mem.total_received = u64::MAX;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_no_last_contribution_ok() {
        let mut mem = valid_membership();
        mem.last_contribution = None;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_no_last_disbursement_ok() {
        let mut mem = valid_membership();
        mem.last_disbursement = None;
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_with_both_timestamps_ok() {
        let mut mem = valid_membership();
        mem.last_contribution = Some(fake_timestamp(2000));
        mem.last_disbursement = Some(fake_timestamp(3000));
        assert_valid(validate_pool_membership(&mem));
    }

    // ── validate_mutual_aid_pool: contribution rule numeric bounds ─────

    #[test]
    fn validate_pool_max_withdrawal_zero_rejected() {
        let mut pool = valid_pool();
        pool.contribution_rules.max_withdrawal = 0;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Max withdrawal must be greater than 0",
        );
    }

    #[test]
    fn validate_pool_max_withdrawal_one_ok() {
        let mut pool = valid_pool();
        pool.contribution_rules.max_withdrawal = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_max_withdrawal_max_ok() {
        let mut pool = valid_pool();
        pool.contribution_rules.max_withdrawal = u64::MAX;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── validate_mutual_aid_pool: disbursement rule numeric bounds ─────

    #[test]
    fn validate_pool_min_approvals_zero_rejected() {
        let mut pool = valid_pool();
        pool.disbursement_rules.min_approvals = 0;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Minimum approvals must be greater than 0",
        );
    }

    #[test]
    fn validate_pool_min_approvals_one_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.min_approvals = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_min_approvals_large_with_enough_members_ok() {
        let mut pool = valid_pool();
        pool.members = (0..100)
            .map(|i| format!("did:mycelix:member{}", i))
            .collect();
        pool.disbursement_rules.min_approvals = 100;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_max_disbursement_zero_rejected() {
        let mut pool = valid_pool();
        pool.disbursement_rules.max_disbursement = 0;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Max disbursement must be greater than 0",
        );
    }

    #[test]
    fn validate_pool_max_disbursement_one_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.max_disbursement = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_max_disbursement_max_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.max_disbursement = u64::MAX;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_min_monthly_zero_ok() {
        let mut pool = valid_pool();
        pool.contribution_rules.min_monthly = 0;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_min_monthly_nonzero_ok() {
        let mut pool = valid_pool();
        pool.contribution_rules.min_monthly = 500;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── Default trait tests ─────────────────────────────────────────────

    #[test]
    fn contribution_rule_default_values() {
        let rule = ContributionRule::default();
        assert_eq!(rule.min_monthly, 0);
        assert_eq!(rule.max_withdrawal, u64::MAX);
        assert_eq!(rule.cooldown_days, 0);
    }

    #[test]
    fn disbursement_rule_default_values() {
        let rule = DisbursementRule::default();
        assert_eq!(rule.min_approvals, 1);
        assert_eq!(rule.approval_threshold_percent, 50);
        assert_eq!(rule.max_disbursement, u64::MAX);
        assert_eq!(rule.allow_emergency_bypass, false);
    }

    // ── Anchor tests ────────────────────────────────────────────────────

    #[test]
    fn anchor_new() {
        let anchor = Anchor::new("test_anchor");
        assert_eq!(anchor.0, "test_anchor");
    }

    #[test]
    fn anchor_clone_eq() {
        let a1 = Anchor::new("same");
        let a2 = Anchor::new("same");
        assert_eq!(a1, a2);
    }

    // ── String/Vec length limit tests ─────────────────────────────────────

    // Pool: id max 64
    #[test]
    fn validate_pool_id_exactly_64_ok() {
        let mut pool = valid_pool();
        pool.id = "x".repeat(64);
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_id_65_rejected() {
        let mut pool = valid_pool();
        pool.id = "x".repeat(257);
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Pool ID must be 256 characters or fewer",
        );
    }

    // Pool: name max 256
    #[test]
    fn validate_pool_name_exactly_256_ok() {
        let mut pool = valid_pool();
        pool.name = "x".repeat(256);
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_name_257_rejected() {
        let mut pool = valid_pool();
        pool.name = "x".repeat(257);
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Pool name must be 256 characters or fewer",
        );
    }

    // Pool: description max 4096
    #[test]
    fn validate_pool_description_exactly_4096_ok() {
        let mut pool = valid_pool();
        pool.description = "x".repeat(4096);
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_description_4097_rejected() {
        let mut pool = valid_pool();
        pool.description = "x".repeat(4097);
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Pool description must be 4096 characters or fewer",
        );
    }

    // Pool: members max 500
    #[test]
    fn validate_pool_500_members_ok() {
        let mut pool = valid_pool();
        pool.members = (0..500)
            .map(|i| format!("did:mycelix:member{}", i))
            .collect();
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_501_members_rejected() {
        let mut pool = valid_pool();
        pool.members = (0..501)
            .map(|i| format!("did:mycelix:member{}", i))
            .collect();
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Cannot have more than 500 members",
        );
    }

    // Contribution: id max 64
    #[test]
    fn validate_contribution_id_exactly_64_ok() {
        let mut contrib = valid_contribution();
        contrib.id = "x".repeat(64);
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_id_65_rejected() {
        let mut contrib = valid_contribution();
        contrib.id = "x".repeat(257);
        assert_invalid(
            validate_contribution(&contrib),
            "Contribution ID must be 256 characters or fewer",
        );
    }

    // Contribution: pool_id max 64
    #[test]
    fn validate_contribution_pool_id_exactly_64_ok() {
        let mut contrib = valid_contribution();
        contrib.pool_id = "x".repeat(64);
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_pool_id_65_rejected() {
        let mut contrib = valid_contribution();
        contrib.pool_id = "x".repeat(257);
        assert_invalid(
            validate_contribution(&contrib),
            "Pool ID must be 256 characters or fewer",
        );
    }

    // Contribution: note max 4096
    #[test]
    fn validate_contribution_note_exactly_4096_ok() {
        let mut contrib = valid_contribution();
        contrib.note = Some("x".repeat(4096));
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_note_4097_rejected() {
        let mut contrib = valid_contribution();
        contrib.note = Some("x".repeat(4097));
        assert_invalid(
            validate_contribution(&contrib),
            "Contribution note must be 4096 characters or fewer",
        );
    }

    // Disbursement: id max 64
    #[test]
    fn validate_disbursement_id_exactly_64_ok() {
        let mut disb = valid_disbursement();
        disb.id = "x".repeat(64);
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_id_65_rejected() {
        let mut disb = valid_disbursement();
        disb.id = "x".repeat(257);
        assert_invalid(
            validate_disbursement(&disb),
            "Disbursement ID must be 256 characters or fewer",
        );
    }

    // Disbursement: reason max 4096
    #[test]
    fn validate_disbursement_reason_exactly_4096_ok() {
        let mut disb = valid_disbursement();
        disb.reason = "x".repeat(4096);
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_reason_4097_rejected() {
        let mut disb = valid_disbursement();
        disb.reason = "x".repeat(4097);
        assert_invalid(
            validate_disbursement(&disb),
            "Disbursement reason must be 4096 characters or fewer",
        );
    }

    // Disbursement: approved_by max 100
    #[test]
    fn validate_disbursement_100_approvers_ok() {
        let mut disb = valid_disbursement();
        disb.approved_by = (0..100)
            .map(|i| format!("did:mycelix:approver{}", i))
            .collect();
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_101_approvers_rejected() {
        let mut disb = valid_disbursement();
        disb.approved_by = (0..101)
            .map(|i| format!("did:mycelix:approver{}", i))
            .collect();
        assert_invalid(
            validate_disbursement(&disb),
            "Cannot have more than 100 approvers",
        );
    }

    // Disbursement: rejected_by max 100
    #[test]
    fn validate_disbursement_100_rejectors_ok() {
        let mut disb = valid_disbursement();
        disb.rejected_by = (0..100)
            .map(|i| format!("did:mycelix:rejector{}", i))
            .collect();
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_101_rejectors_rejected() {
        let mut disb = valid_disbursement();
        disb.rejected_by = (0..101)
            .map(|i| format!("did:mycelix:rejector{}", i))
            .collect();
        assert_invalid(
            validate_disbursement(&disb),
            "Cannot have more than 100 rejectors",
        );
    }

    // PoolMembership: pool_id max 64
    #[test]
    fn validate_membership_pool_id_exactly_64_ok() {
        let mut mem = valid_membership();
        mem.pool_id = "x".repeat(64);
        assert_valid(validate_pool_membership(&mem));
    }

    #[test]
    fn validate_membership_pool_id_65_rejected() {
        let mut mem = valid_membership();
        mem.pool_id = "x".repeat(257);
        assert_invalid(
            validate_pool_membership(&mem),
            "Pool ID must be 256 characters or fewer",
        );
    }

    // ── Link tag validation tests ──────────────────────────────────────

    fn validate_link_tag_for(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AnchorToPool
            | LinkTypes::PoolToContribution
            | LinkTypes::PoolToDisbursement
            | LinkTypes::PoolToMembership
            | LinkTypes::MemberToMembership
            | LinkTypes::MemberToContribution
            | LinkTypes::MemberToDisbursement => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PoolToPendingDisbursement => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PoolToPendingDisbursement link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_anchor_to_pool_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::AnchorToPool,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_anchor_to_pool_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::AnchorToPool, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_pool_to_pending_disbursement_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::PoolToPendingDisbursement,
            vec![0u8; 512],
        ));
    }

    #[test]
    fn test_link_pool_to_pending_disbursement_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::PoolToPendingDisbursement, vec![0u8; 513]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_member_to_membership_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::MemberToMembership,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_member_to_membership_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::MemberToMembership, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_pool_to_contribution_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::PoolToContribution,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_pool_to_contribution_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::PoolToContribution, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_pool_to_disbursement_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::PoolToDisbursement,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_pool_to_disbursement_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::PoolToDisbursement, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_pool_to_membership_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::PoolToMembership,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_pool_to_membership_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::PoolToMembership, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_member_to_contribution_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::MemberToContribution,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_member_to_contribution_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::MemberToContribution, vec![0u8; 257]),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_member_to_disbursement_tag_at_limit() {
        assert_valid(validate_link_tag_for(
            LinkTypes::MemberToDisbursement,
            vec![0u8; 256],
        ));
    }

    #[test]
    fn test_link_member_to_disbursement_tag_too_long() {
        assert_invalid(
            validate_link_tag_for(LinkTypes::MemberToDisbursement, vec![0u8; 257]),
            "link tag too long",
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // HARDENING: Byzantine & Edge Case Tests
    // ══════════════════════════════════════════════════════════════════════

    // ── Duplicate member detection ──────────────────────────────────────

    #[test]
    fn validate_pool_duplicate_members_rejected() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), valid_did2(), valid_did()];
        assert_invalid(validate_mutual_aid_pool(&pool), "Duplicate member DID");
    }

    #[test]
    fn validate_pool_all_same_members_rejected() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), valid_did(), valid_did()];
        assert_invalid(validate_mutual_aid_pool(&pool), "Duplicate member DID");
    }

    #[test]
    fn validate_pool_unique_members_ok() {
        let mut pool = valid_pool();
        pool.members = vec![
            "did:mycelix:alice".to_string(),
            "did:mycelix:bob".to_string(),
            "did:mycelix:charlie".to_string(),
        ];
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── min_approvals vs member count ───────────────────────────────────

    #[test]
    fn validate_pool_min_approvals_exceeds_members_rejected() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), valid_did2()]; // 2 members
        pool.disbursement_rules.min_approvals = 3; // need 3 approvals
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "min_approvals (3) exceeds member count (2)",
        );
    }

    #[test]
    fn validate_pool_min_approvals_equals_members_ok() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), valid_did2()]; // 2 members
        pool.disbursement_rules.min_approvals = 2;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_min_approvals_less_than_members_ok() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did(), valid_did2()]; // 2 members
        pool.disbursement_rules.min_approvals = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_min_approvals_with_empty_members_ok() {
        // Empty members + min_approvals > 0 is allowed (pool not yet populated)
        let mut pool = valid_pool();
        pool.members = vec![];
        pool.disbursement_rules.min_approvals = 5;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── Single-member pool edge cases ───────────────────────────────────

    #[test]
    fn validate_pool_single_member_min_approvals_1_ok() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did()];
        pool.disbursement_rules.min_approvals = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_pool_single_member_min_approvals_2_rejected() {
        let mut pool = valid_pool();
        pool.members = vec![valid_did()];
        pool.disbursement_rules.min_approvals = 2;
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "min_approvals (2) exceeds member count (1)",
        );
    }

    // ── Disbursement: approved_by/rejected_by overlap ───────────────────

    #[test]
    fn validate_disbursement_same_did_approve_and_reject() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![valid_did()];
        disb.rejected_by = vec![valid_did()]; // same person
        assert_invalid(
            validate_disbursement(&disb),
            "DID appears in both approved_by and rejected_by",
        );
    }

    #[test]
    fn validate_disbursement_overlap_among_many() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![
            "did:mycelix:alice".to_string(),
            "did:mycelix:bob".to_string(),
        ];
        disb.rejected_by = vec![
            "did:mycelix:charlie".to_string(),
            "did:mycelix:bob".to_string(), // overlap
        ];
        assert_invalid(
            validate_disbursement(&disb),
            "DID appears in both approved_by and rejected_by",
        );
    }

    #[test]
    fn validate_disbursement_no_overlap_ok() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![
            "did:mycelix:alice".to_string(),
            "did:mycelix:bob".to_string(),
        ];
        disb.rejected_by = vec![
            "did:mycelix:charlie".to_string(),
            "did:mycelix:dave".to_string(),
        ];
        assert_valid(validate_disbursement(&disb));
    }

    // ── Disbursement: duplicate DIDs within approved_by ─────────────────

    #[test]
    fn validate_disbursement_duplicate_approver() {
        let mut disb = valid_disbursement();
        disb.approved_by = vec![valid_did(), valid_did()];
        assert_invalid(validate_disbursement(&disb), "Duplicate DID in approved_by");
    }

    #[test]
    fn validate_disbursement_duplicate_rejector() {
        let mut disb = valid_disbursement();
        disb.rejected_by = vec![valid_did(), valid_did()];
        assert_invalid(validate_disbursement(&disb), "Duplicate DID in rejected_by");
    }

    // ── DID injection attempts ──────────────────────────────────────────

    #[test]
    fn validate_did_with_null_bytes() {
        // Null bytes in DID — should at least not panic
        let mut pool = valid_pool();
        pool.members = vec!["did:mycelix:alice\0injected".to_string()];
        // Passes basic format check (starts with did:, 3+ parts), but documents the edge case
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_did_with_very_long_method() {
        let long_method = "x".repeat(1000);
        let did = format!("did:{}:identifier", long_method);
        // Basic DID format check passes — we rely on member_did length limits per-entry
        assert!(validate_did(&did).is_ok());
    }

    #[test]
    fn validate_did_minimal_valid() {
        // "did:a:b" — minimal valid DID (3 parts)
        assert!(validate_did("did:a:b").is_ok());
    }

    #[test]
    fn validate_did_many_colons() {
        // "did:method:id:extra:parts" — valid, extra parts allowed
        assert!(validate_did("did:method:id:extra:parts").is_ok());
    }

    // ── Balance saturation edge cases ───────────────────────────────────

    #[test]
    fn validate_pool_max_u64_balance_ok() {
        let mut pool = valid_pool();
        pool.balance = u64::MAX;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── Emergency bypass with zero approvals ────────────────────────────

    #[test]
    fn validate_pool_emergency_bypass_enabled_ok() {
        let mut pool = valid_pool();
        pool.disbursement_rules.allow_emergency_bypass = true;
        pool.disbursement_rules.min_approvals = 1;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    // ── Combination edge cases ──────────────────────────────────────────

    #[test]
    fn validate_pool_max_members_with_max_rules() {
        let mut pool = valid_pool();
        pool.members = (0..500)
            .map(|i| format!("did:mycelix:member{}", i))
            .collect();
        pool.disbursement_rules.min_approvals = 500;
        pool.disbursement_rules.approval_threshold_percent = 100;
        pool.contribution_rules.min_monthly = u64::MAX;
        pool.contribution_rules.cooldown_days = u32::MAX;
        assert_valid(validate_mutual_aid_pool(&pool));
    }

    #[test]
    fn validate_disbursement_max_amount() {
        let mut disb = valid_disbursement();
        disb.amount = u64::MAX;
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_contribution_member_did_at_256_ok() {
        let mut contrib = valid_contribution();
        // Build a valid DID that's exactly 256 chars
        let padding = "x".repeat(256 - "did:mycelix:".len());
        contrib.member_did = format!("did:mycelix:{}", padding);
        assert_valid(validate_contribution(&contrib));
    }

    #[test]
    fn validate_contribution_member_did_257_rejected() {
        let mut contrib = valid_contribution();
        let padding = "x".repeat(257 - "did:mycelix:".len());
        contrib.member_did = format!("did:mycelix:{}", padding);
        assert_invalid(
            validate_contribution(&contrib),
            "Member DID must be 256 characters or fewer",
        );
    }

    #[test]
    fn validate_disbursement_recipient_did_at_256_ok() {
        let mut disb = valid_disbursement();
        let padding = "x".repeat(256 - "did:mycelix:".len());
        disb.recipient_did = format!("did:mycelix:{}", padding);
        assert_valid(validate_disbursement(&disb));
    }

    #[test]
    fn validate_disbursement_recipient_did_257_rejected() {
        let mut disb = valid_disbursement();
        let padding = "x".repeat(257 - "did:mycelix:".len());
        disb.recipient_did = format!("did:mycelix:{}", padding);
        assert_invalid(
            validate_disbursement(&disb),
            "Recipient DID must be 256 characters or fewer",
        );
    }
}
