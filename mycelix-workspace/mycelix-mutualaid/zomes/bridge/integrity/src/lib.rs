// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bridge Integrity Zome - Cross-hApp verification for mutual aid
//!
//! This zome defines the data structures and validation rules for cross-hApp
//! queries and MATL (Multi-Agent Trust Layer) integration within the Mycelix
//! mutual aid network.

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

/// Purpose of a cross-hApp query
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPurpose {
    /// Verify membership status for pool operations
    MemberVerification,
    /// Query contribution history for trust scoring
    ContributionHistory,
    /// Query request/fulfillment history
    RequestHistory,
    /// Query reputation score
    ReputationQuery,
    /// Query compliance status
    ComplianceCheck,
}

/// Query status tracking
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Expired,
}

/// A cross-hApp aid query
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AidQuery {
    /// Unique identifier for this query
    pub id: String,
    /// DID of the subject being queried
    pub subject_did: String,
    /// DID of the requesting party
    pub requester_did: String,
    /// Purpose of the query
    pub purpose: QueryPurpose,
    /// Source hApp identifier
    pub source_happ: String,
    /// Target hApp identifier
    pub target_happ: String,
    /// Optional parameters for the query
    pub parameters: Option<String>,
    /// Query status
    pub status: QueryStatus,
    /// Timestamp when query was created
    pub created_at: Timestamp,
    /// Timestamp when query expires
    pub expires_at: Timestamp,
}

/// Result of a cross-hApp query
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AidResult {
    /// Reference to the query this is responding to
    pub query_id: String,
    /// Whether the query was successful
    pub success: bool,
    /// Result data (JSON encoded)
    pub data: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// MATL trust score of the result provider
    pub provider_trust_score: f64,
    /// Timestamp when result was created
    pub created_at: Timestamp,
}

/// MATL reputation score for a member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MatlReputation {
    /// DID of the member
    pub member_did: String,
    /// Overall trust score (0.0 - 1.0)
    pub trust_score: f64,
    /// Contribution consistency score
    pub contribution_score: f64,
    /// Request fulfillment score (as helper)
    pub fulfillment_score: f64,
    /// Community engagement score
    pub engagement_score: f64,
    /// Number of pools participated in
    pub pools_count: u32,
    /// Total value contributed
    pub total_contributed: u64,
    /// Total value received
    pub total_received: u64,
    /// Number of successful aid offers
    pub successful_offers: u32,
    /// Number of requests made
    pub requests_made: u32,
    /// Last activity timestamp
    pub last_activity: Timestamp,
    /// Score calculation timestamp
    pub calculated_at: Timestamp,
}

/// Cross-hApp verification record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerificationRecord {
    /// Unique identifier
    pub id: String,
    /// DID being verified
    pub subject_did: String,
    /// Type of verification
    pub verification_type: VerificationType,
    /// Verifying hApp
    pub verifier_happ: String,
    /// Verification result
    pub result: VerificationResult,
    /// Evidence supporting the verification
    pub evidence: Option<String>,
    /// Timestamp of verification
    pub verified_at: Timestamp,
    /// Validity period in seconds
    pub valid_for_seconds: u64,
}

/// Type of verification
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationType {
    Identity,
    PoolMembership,
    ContributionHistory,
    GoodStanding,
    SkillCredential,
    LocationProximity,
}

/// Verification result
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationResult {
    Verified,
    NotVerified,
    Inconclusive,
    Expired,
}

/// Bridge configuration for cross-hApp communication
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BridgeConfig {
    /// Configuration ID
    pub id: String,
    /// List of trusted hApps (by DNA hash or identifier)
    pub trusted_happs: Vec<String>,
    /// Minimum trust score to accept queries
    pub min_trust_score: f64,
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,
    /// Maximum concurrent queries
    pub max_concurrent_queries: u32,
    /// Whether to allow anonymous queries
    pub allow_anonymous: bool,
    /// Last updated timestamp
    pub updated_at: Timestamp,
}

/// All entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    AidQuery(AidQuery),
    #[entry_type(visibility = "public")]
    AidResult(AidResult),
    #[entry_type(visibility = "public")]
    MatlReputation(MatlReputation),
    #[entry_type(visibility = "public")]
    VerificationRecord(VerificationRecord),
    #[entry_type(visibility = "private")]
    BridgeConfig(BridgeConfig),
}

/// Link types for connecting entries
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all queries
    AnchorToQuery,
    /// Query to its result
    QueryToResult,
    /// Member DID to their reputation
    MemberToReputation,
    /// Member DID to verification records
    MemberToVerification,
    /// Anchor to pending queries
    AnchorToPendingQuery,
    /// Source hApp to queries
    SourceHappToQuery,
    /// Target hApp to queries
    TargetHappToQuery,
}

/// Validation errors for bridge zome
#[derive(Debug)]
pub enum BridgeError {
    InvalidDid(String),
    InvalidId(String),
    InvalidTrustScore,
    InvalidTimeout,
    QueryExpired,
    InvalidHappId,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDid(s) => write!(f, "Invalid DID format: {}", s),
            Self::InvalidId(s) => write!(f, "Invalid ID format: {}", s),
            Self::InvalidTrustScore => {
                write!(f, "Invalid trust score: must be between 0.0 and 1.0")
            }
            Self::InvalidTimeout => write!(f, "Invalid timeout: must be positive"),
            Self::QueryExpired => write!(f, "Query expired"),
            Self::InvalidHappId => write!(f, "Invalid hApp identifier"),
        }
    }
}

/// Validate that a DID has a valid format
fn validate_did(did: &str) -> ExternResult<()> {
    if did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            BridgeError::InvalidDid("DID cannot be empty".to_string()).to_string()
        )));
    }
    // Basic DID format check: did:method:identifier
    if !did.starts_with("did:") || did.split(':').count() < 3 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            BridgeError::InvalidDid(format!("Invalid DID format: {}", did)).to_string()
        )));
    }
    Ok(())
}

/// Validate that an ID is non-empty
fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
    if id.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            BridgeError::InvalidId(format!("{} cannot be empty", field_name)).to_string()
        )));
    }
    Ok(())
}

/// Validate a trust score is in valid range
fn validate_trust_score(score: f64) -> ExternResult<()> {
    if score < 0.0 || score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            BridgeError::InvalidTrustScore.to_string()
        )));
    }
    Ok(())
}

/// Validate an AidQuery entry
fn validate_aid_query(query: &AidQuery) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&query.id, "Query ID")?;

    // Validate DIDs
    validate_did(&query.subject_did)?;
    validate_did(&query.requester_did)?;

    // Validate hApp identifiers
    if query.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            BridgeError::InvalidHappId.to_string(),
        ));
    }
    if query.target_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            BridgeError::InvalidHappId.to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an AidResult entry
fn validate_aid_result(result: &AidResult) -> ExternResult<ValidateCallbackResult> {
    // Validate query ID
    validate_id(&result.query_id, "Query ID")?;

    // Validate trust score
    validate_trust_score(result.provider_trust_score)?;

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a MatlReputation entry
fn validate_matl_reputation(reputation: &MatlReputation) -> ExternResult<ValidateCallbackResult> {
    // Validate DID
    validate_did(&reputation.member_did)?;

    // Validate all scores
    validate_trust_score(reputation.trust_score)?;
    validate_trust_score(reputation.contribution_score)?;
    validate_trust_score(reputation.fulfillment_score)?;
    validate_trust_score(reputation.engagement_score)?;

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a VerificationRecord entry
fn validate_verification_record(
    record: &VerificationRecord,
) -> ExternResult<ValidateCallbackResult> {
    // Validate IDs
    validate_id(&record.id, "Verification ID")?;

    // Validate DID
    validate_did(&record.subject_did)?;

    // Validate hApp identifier
    if record.verifier_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            BridgeError::InvalidHappId.to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a BridgeConfig entry
fn validate_bridge_config(config: &BridgeConfig) -> ExternResult<ValidateCallbackResult> {
    // Validate ID
    validate_id(&config.id, "Config ID")?;

    // Validate trust score threshold
    validate_trust_score(config.min_trust_score)?;

    // Validate timeout
    if config.query_timeout_seconds == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            BridgeError::InvalidTimeout.to_string(),
        ));
    }

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
                    EntryTypes::AidQuery(query) => validate_aid_query(&query),
                    EntryTypes::AidResult(result) => validate_aid_result(&result),
                    EntryTypes::MatlReputation(reputation) => validate_matl_reputation(&reputation),
                    EntryTypes::VerificationRecord(record) => validate_verification_record(&record),
                    EntryTypes::BridgeConfig(config) => validate_bridge_config(&config),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToQuery
            | LinkTypes::QueryToResult
            | LinkTypes::MemberToReputation
            | LinkTypes::MemberToVerification
            | LinkTypes::AnchorToPendingQuery
            | LinkTypes::SourceHappToQuery
            | LinkTypes::TargetHappToQuery => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToQuery
            | LinkTypes::QueryToResult
            | LinkTypes::MemberToReputation
            | LinkTypes::MemberToVerification
            | LinkTypes::AnchorToPendingQuery
            | LinkTypes::SourceHappToQuery
            | LinkTypes::TargetHappToQuery => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
