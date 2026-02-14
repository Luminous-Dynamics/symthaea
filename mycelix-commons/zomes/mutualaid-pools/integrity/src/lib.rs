//! Pools Integrity Zome - Community mutual aid pools
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

    // Validate name is not empty
    if pool.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::EmptyName.to_string()
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
                pool.disbursement_rules.approval_threshold_percent
            ).to_string()
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
            PoolsError::ZeroAmount.to_string()
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
            PoolsError::ZeroAmount.to_string()
        ));
    }

    // Validate reason is not empty
    if disbursement.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            PoolsError::EmptyReason.to_string()
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
        let statuses = vec![
            PoolStatus::Active,
            PoolStatus::Paused,
            PoolStatus::Closed,
        ];
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
        let roles = vec![
            MemberRole::Admin,
            MemberRole::Member,
            MemberRole::Observer,
        ];
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
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Pool name cannot be empty",
        );
    }

    #[test]
    fn validate_pool_whitespace_only_name() {
        let mut pool = valid_pool();
        pool.name = "   ".to_string();
        assert_invalid(
            validate_mutual_aid_pool(&pool),
            "Pool name cannot be empty",
        );
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
        assert_invalid(
            validate_contribution(&contrib),
            "Amount cannot be zero",
        );
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
        assert_invalid(
            validate_disbursement(&disb),
            "Amount cannot be zero",
        );
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
        assert_invalid(
            validate_disbursement(&disb),
            "Reason cannot be empty",
        );
    }

    #[test]
    fn validate_disbursement_whitespace_reason() {
        let mut disb = valid_disbursement();
        disb.reason = "   ".to_string();
        assert_invalid(
            validate_disbursement(&disb),
            "Reason cannot be empty",
        );
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
}
