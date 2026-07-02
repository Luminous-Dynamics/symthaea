// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bridge Integrity Zome
//!
//! Defines entry types for inter-hApp communication in the Mycelix ecosystem.
//! This includes hApp registrations, reputation records, bridge events, and
//! cross-hApp credential verification.
//!
//! ## Security Considerations
//!
//! - All string fields are validated for length to prevent DoS attacks
//! - Score values must be within [0.0, 1.0] range
//! - Event payloads have maximum size limits
//! - Credential verification has proper status validation

use hdi::prelude::*;

// =============================================================================
// Constants for Validation
// =============================================================================

/// Maximum length for string identifiers (hApp IDs, agent IDs, etc.)
pub const MAX_ID_LENGTH: usize = 256;

/// Maximum length for names
pub const MAX_NAME_LENGTH: usize = 512;

/// Maximum payload size for bridge events (64KB)
pub const MAX_EVENT_PAYLOAD_SIZE: usize = 65536;

/// Maximum number of capabilities per hApp
pub const MAX_CAPABILITIES: usize = 50;

/// Maximum number of target hApps for events
pub const MAX_EVENT_TARGETS: usize = 100;

/// Valid credential verification statuses
pub const VALID_CREDENTIAL_STATUSES: &[&str] = &["pending", "verified", "failed", "expired"];

/// Valid event types
pub const STANDARD_EVENT_TYPES: &[&str] = &[
    "reputation_update",
    "credential_issued",
    "credential_revoked",
    "happ_registered",
    "happ_deregistered",
    "trust_alert",
    "custom",
];

// =============================================================================
// Entry Types
// =============================================================================

/// hApp registration entry
///
/// Registers a hApp with the bridge to enable cross-hApp communication.
/// Each hApp must register before it can participate in reputation sharing
/// or credential verification.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HappRegistration {
    /// Unique hApp identifier (DNA hash or chosen ID)
    pub happ_id: String,

    /// DNA hash of the hApp (for cryptographic verification)
    pub dna_hash: Option<String>,

    /// Human-readable hApp name
    pub happ_name: String,

    /// Capabilities this hApp supports
    /// e.g., ["reputation", "credentials", "messaging"]
    pub capabilities: Vec<String>,

    /// Registration timestamp (Unix seconds)
    pub registered_at: u64,

    /// Registering agent (AgentPubKey as string)
    pub registrant: String,

    /// Whether this hApp is currently active
    pub active: bool,
}

/// Reputation record entry
///
/// Stores an agent's reputation score from a specific hApp.
/// Multiple records can exist per agent-hApp pair (historical).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationRecord {
    /// Agent identifier (AgentPubKey as string)
    pub agent: String,

    /// Source hApp ID
    pub happ_id: String,

    /// Source hApp name
    pub happ_name: String,

    /// Reputation score [0.0, 1.0]
    pub score: f64,

    /// Number of positive interactions
    pub interactions: u64,

    /// Number of negative interactions
    pub negative_interactions: u64,

    /// Last update timestamp
    pub updated_at: u64,

    /// Evidence hash (optional proof of reputation calculation)
    pub evidence_hash: Option<String>,
}

/// Aggregated cross-hApp reputation entry
///
/// Stores the computed aggregate reputation for an agent across multiple hApps.
/// This is stored to enable efficient queries and historical tracking.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossHappReputationRecord {
    /// Agent identifier
    pub agent: String,

    /// Map of hApp ID to score (serialized as JSON)
    pub scores_json: String,

    /// Weighted aggregate score [0.0, 1.0]
    pub aggregated_score: f64,

    /// Total interactions across all hApps
    pub total_interactions: u64,

    /// Number of hApps contributing to this score
    pub happ_count: u32,

    /// When this aggregate was computed
    pub computed_at: u64,
}

/// Bridge event entry
///
/// Published events for cross-hApp notifications.
/// Events are stored in the DHT and can be queried by type.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BridgeEventRecord {
    /// Event type (e.g., "reputation_update", "credential_issued")
    pub event_type: String,

    /// Source hApp ID
    pub source_happ: String,

    /// Event payload (serialized)
    pub payload: Vec<u8>,

    /// Event timestamp
    pub timestamp: u64,

    /// Target hApps (empty = broadcast to all)
    pub targets: Vec<String>,

    /// Priority level (0=normal, 1=high, 2=critical)
    pub priority: u8,
}

/// Cross-hApp credential verification request
///
/// Requests verification of a credential from another hApp.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialVerificationRequest {
    /// Unique request ID
    pub request_id: String,

    /// Credential hash (from issuing hApp)
    pub credential_hash: String,

    /// Credential type being verified
    pub credential_type: String,

    /// Issuing hApp ID
    pub issuer_happ: String,

    /// Agent the credential belongs to
    pub subject_agent: String,

    /// Requesting agent
    pub requester: String,

    /// Request timestamp
    pub requested_at: u64,

    /// Verification status: pending, verified, failed, expired
    pub status: String,
}

/// Credential verification response
///
/// Response to a credential verification request.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialVerificationResponse {
    /// The request ID this responds to
    pub request_id: String,

    /// Whether the credential is valid
    pub is_valid: bool,

    /// Credential issuer (verified)
    pub issuer: String,

    /// Credential type
    pub credential_type: String,

    /// Claims included in the credential (JSON array)
    pub claims_json: String,

    /// When the credential was issued
    pub issued_at: u64,

    /// When the credential expires (0 = no expiry)
    pub expires_at: u64,

    /// Verification timestamp
    pub verified_at: u64,

    /// Verifier agent
    pub verifier: String,

    /// Cryptographic proof of verification (optional)
    pub proof: Option<Vec<u8>>,
}

// =============================================================================
// Ethereum Bridge Entry Types
// =============================================================================

/// Ethereum payment distribution intent
///
/// Records the intent to distribute payment via Ethereum.
/// The WASM zome cannot make HTTP calls directly - it stores this intent
/// and emits a signal for the native Host (FL Aggregator) to process.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EthereumPaymentIntent {
    /// Unique intent ID
    pub intent_id: String,

    /// Model ID for payment routing
    pub model_id: String,

    /// FL round number
    pub round: u64,

    /// Total amount to distribute (in wei as string)
    pub total_amount_wei: String,

    /// Payment splits as JSON: [{ "address": "0x...", "basis_points": 2500, "node_id": "..." }]
    pub splits_json: String,

    /// Platform fee in basis points (e.g., 500 = 5%)
    pub platform_fee_bps: u64,

    /// Intent status: pending, submitted, confirmed, failed
    pub status: String,

    /// Ethereum transaction hash (set when submitted)
    pub tx_hash: Option<String>,

    /// Block number where confirmed
    pub block_number: Option<u64>,

    /// Request timestamp
    pub created_at: u64,

    /// Last status update timestamp
    pub updated_at: u64,

    /// Requesting agent
    pub requester: String,

    /// Error message if failed
    pub error: Option<String>,
}

/// Ethereum anchor intent
///
/// Records the intent to anchor data to Ethereum.
/// Used for reputation anchoring, contribution proofs, etc.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EthereumAnchorIntent {
    /// Unique intent ID
    pub intent_id: String,

    /// Type of anchor: "reputation", "contribution", "proof"
    pub anchor_type: String,

    /// Data to anchor (hash or commitment)
    pub data_hash: String,

    /// Optional metadata JSON
    pub metadata_json: Option<String>,

    /// Intent status: pending, submitted, confirmed, failed
    pub status: String,

    /// Ethereum transaction hash (set when submitted)
    pub tx_hash: Option<String>,

    /// Block number where confirmed
    pub block_number: Option<u64>,

    /// Block hash for finality verification
    pub block_hash: Option<String>,

    /// On-chain commitment ID (contract-assigned)
    pub commitment_id: Option<String>,

    /// Request timestamp
    pub created_at: u64,

    /// Last status update timestamp
    pub updated_at: u64,

    /// Requesting agent
    pub requester: String,

    /// Error message if failed
    pub error: Option<String>,
}

/// All entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    HappRegistration(HappRegistration),
    ReputationRecord(ReputationRecord),
    CrossHappReputationRecord(CrossHappReputationRecord),
    BridgeEventRecord(BridgeEventRecord),
    CredentialVerificationRequest(CredentialVerificationRequest),
    CredentialVerificationResponse(CredentialVerificationResponse),
    EthereumPaymentIntent(EthereumPaymentIntent),
    EthereumAnchorIntent(EthereumAnchorIntent),
}

/// All link types for this integrity zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from hApp ID to registration
    HappIdToRegistration,
    /// Links from agent to their reputation records
    AgentToReputations,
    /// Links from hApp to reputation records
    HappToReputations,
    /// Links from event type to events
    EventTypeToEvents,
    /// Links from hApp to credential verification requests
    HappToCredentialRequests,
    /// Links from request to verification response
    RequestToResponse,
    /// Links from agent to cross-hApp reputation aggregates
    AgentToAggregateReputation,
    /// Links to all registered hApps
    AllHapps,
    /// Links from model to payment intents
    ModelToPaymentIntents,
    /// Links from intent ID to Ethereum payment intent
    IntentIdToPaymentIntent,
    /// Links from intent ID to Ethereum anchor intent
    IntentIdToAnchorIntent,
    /// Links to all pending Ethereum intents (for Host polling)
    PendingEthereumIntents,
}

// =============================================================================
// Validation Functions
// =============================================================================

/// Helper: Validate string field for length and emptiness
fn validate_string_field(
    value: &str,
    field_name: &str,
    max_length: usize,
    allow_empty: bool,
) -> Result<(), String> {
    let trimmed = value.trim();

    if !allow_empty && trimmed.is_empty() {
        return Err(format!("{} cannot be empty", field_name));
    }

    if trimmed.len() > max_length {
        return Err(format!(
            "{} exceeds maximum length of {} characters",
            field_name, max_length
        ));
    }

    Ok(())
}

/// Helper: Validate score range [0.0, 1.0]
fn validate_score(score: f64, field_name: &str) -> Result<(), String> {
    if !score.is_finite() {
        return Err(format!("{} must be a finite number", field_name));
    }

    if score < 0.0 || score > 1.0 {
        return Err(format!("{} must be between 0.0 and 1.0", field_name));
    }

    Ok(())
}

/// Validation for HappRegistration
fn validate_happ_registration(registration: HappRegistration) -> ExternResult<ValidateCallbackResult> {
    // Validate hApp ID
    if let Err(e) = validate_string_field(&registration.happ_id, "hApp ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate hApp name
    if let Err(e) = validate_string_field(&registration.happ_name, "hApp name", MAX_NAME_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate registrant
    if let Err(e) = validate_string_field(&registration.registrant, "Registrant", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate capabilities count
    if registration.capabilities.len() > MAX_CAPABILITIES {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Capabilities list exceeds maximum of {} items", MAX_CAPABILITIES),
        ));
    }

    // Validate each capability
    for cap in &registration.capabilities {
        if let Err(e) = validate_string_field(cap, "Capability", MAX_ID_LENGTH, false) {
            return Ok(ValidateCallbackResult::Invalid(e));
        }
    }

    // Validate optional DNA hash if provided
    if let Some(ref dna_hash) = registration.dna_hash {
        if let Err(e) = validate_string_field(dna_hash, "DNA hash", MAX_ID_LENGTH, true) {
            return Ok(ValidateCallbackResult::Invalid(e));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for ReputationRecord
fn validate_reputation_record(record: ReputationRecord) -> ExternResult<ValidateCallbackResult> {
    // Validate agent
    if let Err(e) = validate_string_field(&record.agent, "Agent", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate hApp ID
    if let Err(e) = validate_string_field(&record.happ_id, "hApp ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate hApp name
    if let Err(e) = validate_string_field(&record.happ_name, "hApp name", MAX_NAME_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate score
    if let Err(e) = validate_score(record.score, "Reputation score") {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate evidence hash if provided
    if let Some(ref evidence_hash) = record.evidence_hash {
        if let Err(e) = validate_string_field(evidence_hash, "Evidence hash", 128, true) {
            return Ok(ValidateCallbackResult::Invalid(e));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CrossHappReputationRecord
fn validate_cross_happ_reputation_record(record: CrossHappReputationRecord) -> ExternResult<ValidateCallbackResult> {
    // Validate agent
    if let Err(e) = validate_string_field(&record.agent, "Agent", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate aggregated score
    if let Err(e) = validate_score(record.aggregated_score, "Aggregated score") {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate scores JSON (must be valid JSON)
    if serde_json::from_str::<serde_json::Value>(&record.scores_json).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "scores_json must be valid JSON".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for BridgeEventRecord
fn validate_bridge_event_record(event: BridgeEventRecord) -> ExternResult<ValidateCallbackResult> {
    // Validate event type
    if let Err(e) = validate_string_field(&event.event_type, "Event type", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate source hApp
    if let Err(e) = validate_string_field(&event.source_happ, "Source hApp", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate payload size
    if event.payload.len() > MAX_EVENT_PAYLOAD_SIZE {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Event payload exceeds maximum size of {} bytes", MAX_EVENT_PAYLOAD_SIZE),
        ));
    }

    // Validate targets count
    if event.targets.len() > MAX_EVENT_TARGETS {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Event targets exceed maximum of {} hApps", MAX_EVENT_TARGETS),
        ));
    }

    // Validate priority
    if event.priority > 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Priority must be 0 (normal), 1 (high), or 2 (critical)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CredentialVerificationRequest
fn validate_credential_verification_request(request: CredentialVerificationRequest) -> ExternResult<ValidateCallbackResult> {
    // Validate request ID
    if let Err(e) = validate_string_field(&request.request_id, "Request ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate credential hash
    if let Err(e) = validate_string_field(&request.credential_hash, "Credential hash", 128, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate credential type
    if let Err(e) = validate_string_field(&request.credential_type, "Credential type", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate issuer hApp
    if let Err(e) = validate_string_field(&request.issuer_happ, "Issuer hApp", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate subject agent
    if let Err(e) = validate_string_field(&request.subject_agent, "Subject agent", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate requester
    if let Err(e) = validate_string_field(&request.requester, "Requester", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate status
    if !VALID_CREDENTIAL_STATUSES.contains(&request.status.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Status must be one of: {}",
                VALID_CREDENTIAL_STATUSES.join(", ")
            ),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CredentialVerificationResponse
fn validate_credential_verification_response(response: CredentialVerificationResponse) -> ExternResult<ValidateCallbackResult> {
    // Validate request ID
    if let Err(e) = validate_string_field(&response.request_id, "Request ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate issuer
    if let Err(e) = validate_string_field(&response.issuer, "Issuer", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate credential type
    if let Err(e) = validate_string_field(&response.credential_type, "Credential type", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate verifier
    if let Err(e) = validate_string_field(&response.verifier, "Verifier", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate claims JSON
    if serde_json::from_str::<serde_json::Value>(&response.claims_json).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "claims_json must be valid JSON".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Valid Ethereum intent statuses
const VALID_ETH_INTENT_STATUSES: &[&str] = &["pending", "submitted", "confirmed", "failed"];

/// Validation for EthereumPaymentIntent
fn validate_ethereum_payment_intent(intent: EthereumPaymentIntent) -> ExternResult<ValidateCallbackResult> {
    // Validate intent ID
    if let Err(e) = validate_string_field(&intent.intent_id, "Intent ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate model ID
    if let Err(e) = validate_string_field(&intent.model_id, "Model ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate amount (must be non-empty)
    if let Err(e) = validate_string_field(&intent.total_amount_wei, "Total amount", 78, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate splits JSON
    if serde_json::from_str::<serde_json::Value>(&intent.splits_json).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "splits_json must be valid JSON".to_string(),
        ));
    }

    // Validate platform fee (max 100% = 10000 bps)
    if intent.platform_fee_bps > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Platform fee cannot exceed 10000 basis points (100%)".to_string(),
        ));
    }

    // Validate status
    if !VALID_ETH_INTENT_STATUSES.contains(&intent.status.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Status must be one of: {}",
                VALID_ETH_INTENT_STATUSES.join(", ")
            ),
        ));
    }

    // Validate requester
    if let Err(e) = validate_string_field(&intent.requester, "Requester", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for EthereumAnchorIntent
fn validate_ethereum_anchor_intent(intent: EthereumAnchorIntent) -> ExternResult<ValidateCallbackResult> {
    // Validate intent ID
    if let Err(e) = validate_string_field(&intent.intent_id, "Intent ID", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate anchor type
    let valid_anchor_types = ["reputation", "contribution", "proof", "model", "round"];
    if !valid_anchor_types.contains(&intent.anchor_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Anchor type must be one of: {}",
                valid_anchor_types.join(", ")
            ),
        ));
    }

    // Validate data hash
    if let Err(e) = validate_string_field(&intent.data_hash, "Data hash", 128, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

    // Validate metadata JSON if provided
    if let Some(ref metadata) = intent.metadata_json {
        if serde_json::from_str::<serde_json::Value>(metadata).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "metadata_json must be valid JSON".to_string(),
            ));
        }
    }

    // Validate status
    if !VALID_ETH_INTENT_STATUSES.contains(&intent.status.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Status must be one of: {}",
                VALID_ETH_INTENT_STATUSES.join(", ")
            ),
        ));
    }

    // Validate requester
    if let Err(e) = validate_string_field(&intent.requester, "Requester", MAX_ID_LENGTH, false) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }

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
                    Some(EntryTypes::HappRegistration(reg)) => validate_happ_registration(reg),
                    Some(EntryTypes::ReputationRecord(rec)) => validate_reputation_record(rec),
                    Some(EntryTypes::CrossHappReputationRecord(rec)) => validate_cross_happ_reputation_record(rec),
                    Some(EntryTypes::BridgeEventRecord(event)) => validate_bridge_event_record(event),
                    Some(EntryTypes::CredentialVerificationRequest(req)) => validate_credential_verification_request(req),
                    Some(EntryTypes::CredentialVerificationResponse(resp)) => validate_credential_verification_response(resp),
                    Some(EntryTypes::EthereumPaymentIntent(intent)) => validate_ethereum_payment_intent(intent),
                    Some(EntryTypes::EthereumAnchorIntent(intent)) => validate_ethereum_anchor_intent(intent),
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
