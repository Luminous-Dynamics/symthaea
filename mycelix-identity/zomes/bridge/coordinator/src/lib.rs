//! Identity Bridge Coordinator Zome
//!
//! Cross-hApp communication for identity verification, DID queries,
//! and reputation aggregation across the Mycelix ecosystem.
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use identity_bridge_integrity::*;
use mycelix_bridge_common::consciousness_profile::{
    ConsciousnessCredential, ConsciousnessProfile, ConsciousnessTier,
};

// Local type definition for DID document data we receive from cross-zome calls
// This mirrors the did_registry_integrity::DidDocument but avoids importing that crate
// which would cause duplicate symbol errors
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DidDocumentData {
    pub id: String,
    pub controller: AgentPubKey,
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Vec<VerificationMethodData>,
    pub authentication: Vec<String>,
    #[serde(
        rename = "keyAgreement",
        alias = "key_agreement",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub key_agreement: Vec<String>,
    pub service: Vec<ServiceEndpointData>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

// Manual implementation to avoid getrandom dependency from SerializedBytes derive
impl TryFrom<SerializedBytes> for DidDocumentData {
    type Error = SerializedBytesError;
    fn try_from(sb: SerializedBytes) -> Result<Self, Self::Error> {
        holochain_serialized_bytes::decode(sb.bytes())
    }
}

impl TryFrom<DidDocumentData> for SerializedBytes {
    type Error = SerializedBytesError;
    fn try_from(t: DidDocumentData) -> Result<Self, Self::Error> {
        Ok(SerializedBytes::from(UnsafeBytes::from(
            holochain_serialized_bytes::encode(&t)?,
        )))
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct VerificationMethodData {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    pub controller: String,
    #[serde(rename = "publicKeyMultibase", alias = "public_key_multibase")]
    pub public_key_multibase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ServiceEndpointData {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    #[serde(rename = "serviceEndpoint", alias = "service_endpoint")]
    pub service_endpoint: String,
}

// ==================== CONSTANTS ====================

const IDENTITY_HAPP_ID: &str = "mycelix-identity";
const REGISTERED_HAPPS_ANCHOR: &str = "registered_happs";
const RECENT_EVENTS_ANCHOR: &str = "recent_events";

/// API version for this coordinator zome.
/// Callers can probe this via `get_api_version` to detect schema mismatches
/// before deserializing cross-zome responses.
/// Increment when making breaking changes to extern function signatures or types.
const API_VERSION: u16 = 1;

// MFA weight in combined MATL score (40% MFA, 60% reputation)
const MFA_WEIGHT: f64 = 0.4;
const REPUTATION_WEIGHT: f64 = 0.6;

// ==================== MFA MIRROR TYPES ====================
// These mirror types from mfa_coordinator to avoid import cycles

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum AssuranceLevel {
    Anonymous,
    Basic,
    Verified,
    HighlyAssured,
    ConstitutionallyCritical,
}

impl AssuranceLevel {
    /// Convert assurance level to a numeric score (0.0 - 1.0)
    pub fn to_score(&self) -> f64 {
        match self {
            AssuranceLevel::Anonymous => 0.0,
            AssuranceLevel::Basic => 0.25,
            AssuranceLevel::Verified => 0.5,
            AssuranceLevel::HighlyAssured => 0.75,
            AssuranceLevel::ConstitutionallyCritical => 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MfaSummary {
    pub did: String,
    pub assurance_level: AssuranceLevel,
    pub assurance_score: f64,
    pub factor_count: usize,
    pub category_count: u8,
    pub has_external_verification: bool,
    pub fl_eligible: bool,
}

// ==================== HELPER FUNCTIONS ====================

/// Create a deterministic entry hash from a string identifier
/// This is used for link bases when we need to link from string IDs
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>(),
    )
}

// ==================== MFA CROSS-ZOME CALLS ====================

/// Get MFA summary for a DID via cross-zome call
fn get_mfa_summary_for_did(did: &str) -> ExternResult<Option<MfaSummary>> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("get_mfa_summary"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let summary: Option<MfaSummary> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode get_mfa_summary response: {:?}",
                    e
                )))
            })?;
            Ok(summary)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // MFA zome not accessible, return None
            Ok(None)
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling mfa zome: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            // Authentication failed, return None
            Ok(None)
        }
    }
}

/// Get MFA assurance score for a DID (0.0 - 1.0)
fn get_mfa_assurance_for_did(did: &str) -> ExternResult<f64> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("get_mfa_assurance_score"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let score: f64 = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode get_mfa_assurance_score response: {:?}",
                    e
                )))
            })?;
            Ok(score)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // MFA zome not accessible, return 0 (Anonymous level)
            Ok(0.0)
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling mfa zome: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            // Authentication failed, return 0 (Anonymous level)
            Ok(0.0)
        }
    }
}

/// Check if DID has MFA state enrolled
#[allow(dead_code)] // Reserved for cross-zome MFA checks
fn has_mfa_enrolled(did: &str) -> ExternResult<bool> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("has_mfa_state"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let has_mfa: bool = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode has_mfa_state response: {:?}",
                    e
                )))
            })?;
            Ok(has_mfa)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => Ok(false),
        ZomeCallResponse::NetworkError(_) => Ok(false),
        ZomeCallResponse::CountersigningSession(_) => Ok(false),
        ZomeCallResponse::AuthenticationFailed(_, _) => Ok(false),
    }
}

// ==================== API VERSION ====================

/// Returns the API version of this coordinator zome.
///
/// Callers should probe this before making cross-zome calls to detect
/// incompatible schema changes. If the returned version is higher than
/// the caller's expected version, the caller should log a warning or
/// fail gracefully rather than risk silent deserialization failures.
#[hdk_extern]
pub fn get_api_version(_: ()) -> ExternResult<u16> {
    Ok(API_VERSION)
}

// ==================== HAPP REGISTRATION ====================

/// Register a hApp with the identity bridge
#[hdk_extern]
pub fn register_happ(input: RegisterHappInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let registration = HappRegistration {
        happ_id: input.happ_id.clone(),
        happ_name: input.happ_name.clone(),
        capabilities: input.capabilities.clone(),
        matl_score: 0.5, // Default starting MATL score
        registered_at: now,
    };

    let action_hash = create_entry(&EntryTypes::HappRegistration(registration))?;

    // Link from anchor to registration
    let anchor = string_to_entry_hash(REGISTERED_HAPPS_ANCHOR);
    create_link(anchor, action_hash.clone(), LinkTypes::RegisteredHapps, ())?;

    // Broadcast registration event
    broadcast_event(BroadcastEventInput {
        event_type: BridgeEventType::HappRegistered,
        subject: input.happ_id.clone(),
        payload: serde_json::json!({
            "happ_id": input.happ_id,
            "happ_name": input.happ_name,
            "capabilities": input.capabilities,
        })
        .to_string(),
    })?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Registration not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterHappInput {
    pub happ_id: String,
    pub happ_name: String,
    pub capabilities: Vec<String>,
}

/// Get all registered hApps
#[hdk_extern]
pub fn get_registered_happs(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = string_to_entry_hash(REGISTERED_HAPPS_ANCHOR);
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RegisteredHapps)?,
        GetStrategy::default(),
    )?;

    let mut happs = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            happs.push(record);
        }
    }

    Ok(happs)
}

// ==================== IDENTITY QUERIES ====================

/// Minimum interval between audit trail entries for the same DID query (60 seconds).
const QUERY_RATE_LIMIT_MICROS: i64 = 60 * 1_000_000;

/// Query a DID from another hApp (cross-hApp identity lookup)
#[hdk_extern]
pub fn query_identity(input: QueryIdentityInput) -> ExternResult<IdentityVerificationResult> {
    let now = sys_time()?;

    // Rate-limit audit trail entries: skip if the same DID was queried
    // within QUERY_RATE_LIMIT_MICROS to prevent DHT bloat from rapid
    // repeated lookups (e.g., enumeration attacks or polling loops).
    let did_hash = string_to_entry_hash(&input.did);
    let recent_queries = get_links(
        LinkQuery::try_new(did_hash.clone(), LinkTypes::DidToQueries)?,
        GetStrategy::default(),
    )?;

    let should_audit = !recent_queries.iter().any(|link| {
        let age = now.as_micros() as i64 - link.timestamp.as_micros();
        (0..QUERY_RATE_LIMIT_MICROS).contains(&age)
    });

    if should_audit {
        let query = IdentityQuery {
            id: format!("query:{}:{}", input.did, now.as_micros()),
            did: input.did.clone(),
            source_happ: input.source_happ.clone(),
            requested_fields: input.requested_fields.clone(),
            queried_at: now,
        };
        let query_hash = create_entry(&EntryTypes::IdentityQuery(query))?;

        create_link(did_hash, query_hash, LinkTypes::DidToQueries, ())?;
    }

    // Get DID details from did_registry (includes creation timestamp)
    let did_details = get_did_details(&input.did)?;
    let did_valid = did_details.is_some();
    let did_created = did_details.as_ref().map(|doc| doc.created);

    // Check if DID is deactivated
    let is_deactivated = is_did_deactivated(&input.did)?;

    // Get aggregated reputation from hApps
    let reputation_score = get_aggregated_reputation(&input.did)?;

    // Get MFA summary for enhanced verification
    let mfa_summary = get_mfa_summary_for_did(&input.did)?;
    let mfa_enrolled = mfa_summary.is_some();
    let mfa_assurance_score = mfa_summary
        .as_ref()
        .map(|s| s.assurance_score)
        .unwrap_or(0.0);
    let mfa_assurance_level = mfa_summary.as_ref().map(|s| s.assurance_level.clone());
    let mfa_factor_count = mfa_summary.as_ref().map(|s| s.factor_count).unwrap_or(0);
    let fl_eligible = mfa_summary.as_ref().map(|s| s.fl_eligible).unwrap_or(false);

    // Calculate combined MATL score (weighted average of reputation and MFA)
    // If MFA is enrolled, use weighted combination; otherwise use reputation only
    let matl_score = if mfa_enrolled {
        (reputation_score * REPUTATION_WEIGHT) + (mfa_assurance_score * MFA_WEIGHT)
    } else {
        reputation_score
    };

    // Count credentials from credential_schema zome
    let credential_count = count_credentials_for_did(&input.did)?;

    // Create verification result
    let verification = IdentityVerification {
        id: format!("verify:{}:{}", input.did, now.as_micros()),
        did: input.did.clone(),
        is_valid: did_valid && !is_deactivated,
        is_deactivated,
        matl_score,
        credential_count,
        did_created,
        verified_at: now,
    };

    let verification_hash = create_entry(&EntryTypes::IdentityVerification(verification.clone()))?;

    Ok(IdentityVerificationResult {
        verification_hash,
        did: verification.did,
        is_valid: verification.is_valid,
        matl_score: verification.matl_score,
        credential_count: verification.credential_count,
        mfa_enrolled,
        mfa_assurance_level,
        mfa_assurance_score,
        mfa_factor_count,
        fl_eligible,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryIdentityInput {
    pub did: String,
    pub source_happ: String,
    pub requested_fields: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IdentityVerificationResult {
    pub verification_hash: ActionHash,
    pub did: String,
    pub is_valid: bool,
    pub matl_score: f64,
    pub credential_count: u32,
    // MFA-related fields
    pub mfa_enrolled: bool,
    pub mfa_assurance_level: Option<AssuranceLevel>,
    pub mfa_assurance_score: f64,
    pub mfa_factor_count: usize,
    pub fl_eligible: bool,
}

/// Verify if a DID exists (calls did_registry zome)
fn verify_did_exists(did: &str) -> ExternResult<bool> {
    // Validate format first
    if !did.starts_with("did:mycelix:") {
        return Ok(false);
    }

    // Call did_registry zome to resolve the DID
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("did_registry"),
        FunctionName::new("resolve_did"),
        None,
        did.to_string(),
    )?;

    // Check if the DID exists (resolve_did returns Option<Record>)
    match response {
        ZomeCallResponse::Ok(result) => {
            let record: Option<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode resolve_did response: {:?}",
                    e
                )))
            })?;
            Ok(record.is_some())
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => Err(wasm_error!(WasmErrorInner::Guest(
            "Unauthorized to call did_registry zome".into()
        ))),
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling did_registry: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => Err(wasm_error!(WasmErrorInner::Guest(
            "Authentication failed calling did_registry zome".into()
        ))),
    }
}

/// Count credentials issued by a DID (calls verifiable_credential zome)
fn count_credentials_for_did(did: &str) -> ExternResult<u32> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("verifiable_credential"),
        FunctionName::new("get_credentials_issued_by"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let credentials: Vec<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode get_credentials_issued_by response: {:?}",
                    e
                )))
            })?;
            Ok(credentials.len() as u32)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            // VC zome not accessible — return 0 rather than failing
            Ok(0)
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling verifiable_credential: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
    }
}

/// Get DID document details (calls did_registry zome)
/// Returns the DID document if it exists, along with creation timestamp
fn get_did_details(did: &str) -> ExternResult<Option<DidDocumentData>> {
    // Validate format first
    if !did.starts_with("did:mycelix:") {
        return Ok(None);
    }

    // Call did_registry zome to resolve the DID
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("did_registry"),
        FunctionName::new("resolve_did"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let record: Option<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode resolve_did response: {:?}",
                    e
                )))
            })?;

            if let Some(rec) = record {
                let did_doc: Option<DidDocumentData> =
                    rec.entry().to_app_option().map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "Failed to deserialize DID document: {:?}",
                            e
                        )))
                    })?;
                Ok(did_doc)
            } else {
                Ok(None)
            }
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => Ok(None),
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling did_registry: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => Ok(None),
    }
}

/// Check if a DID is deactivated (calls did_registry zome)
fn is_did_deactivated(did: &str) -> ExternResult<bool> {
    // Validate format first
    if !did.starts_with("did:mycelix:") {
        return Ok(false);
    }

    // Call did_registry zome to check if DID is active
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("did_registry"),
        FunctionName::new("is_did_active"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let is_active: bool = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode is_did_active response: {:?}",
                    e
                )))
            })?;
            // Deactivated is the inverse of active
            Ok(!is_active)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // If we can't check, assume not deactivated
            Ok(false)
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling did_registry: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            // If authentication failed, assume not deactivated
            Ok(false)
        }
    }
}

// ==================== REPUTATION ====================

/// Report reputation for a DID from another hApp
#[hdk_extern]
pub fn report_reputation(input: ReportReputationInput) -> ExternResult<Record> {
    // Validate DID format
    if !input.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format — must start with did:mycelix:".into()
        )));
    }

    // Validate score bounds
    if input.score < 0.0 || input.score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Reputation score must be between 0.0 and 1.0, got {}",
            input.score
        ))));
    }

    let now = sys_time()?;

    let reputation = IdentityReputation {
        did: input.did.clone(),
        source_happ: input.source_happ.clone(),
        score: input.score,
        interactions: input.interactions,
        last_updated: now,
    };

    let action_hash = create_entry(&EntryTypes::IdentityReputation(reputation))?;

    // Link from DID to this reputation
    create_link(
        string_to_entry_hash(&input.did),
        action_hash.clone(),
        LinkTypes::DidToReputations,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Reputation not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReportReputationInput {
    pub did: String,
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

/// Get aggregated reputation for a DID across all hApps
#[hdk_extern]
pub fn get_reputation(did: String) -> ExternResult<AggregatedReputation> {
    let score = get_aggregated_reputation(&did)?;

    let did_hash = string_to_entry_hash(&did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToReputations)?,
        GetStrategy::default(),
    )?;

    let mut sources = Vec::new();
    let mut total_interactions = 0u64;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(rep) = record
                .entry()
                .to_app_option::<IdentityReputation>()
                .ok()
                .flatten()
            {
                total_interactions += rep.interactions;
                sources.push(ReputationSource {
                    source_happ: rep.source_happ,
                    score: rep.score,
                    interactions: rep.interactions,
                });
            }
        }
    }

    Ok(AggregatedReputation {
        did,
        aggregate_score: score,
        sources,
        total_interactions,
    })
}

/// Exponential decay half-life for reputation scores (30 days in microseconds).
/// After 30 days, a reputation entry's weight is halved.
const REPUTATION_HALF_LIFE_MICROS: f64 = 30.0 * 24.0 * 3600.0 * 1_000_000.0;

/// Internal function to compute aggregated reputation with exponential decay.
///
/// Recent reputation entries are weighted more heavily than old ones. Each
/// entry's interaction count is multiplied by `2^(-age / half_life)` so
/// entries older than 30 days contribute roughly half as much, 60 days
/// contributes a quarter, etc.
fn get_aggregated_reputation(did: &str) -> ExternResult<f64> {
    let did_hash = string_to_entry_hash(did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToReputations)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(0.5); // Default for unknown agents
    }

    let now = sys_time()?;
    let now_micros = now.as_micros() as f64;

    let mut total_weight = 0.0f64;
    let mut weighted_sum = 0.0f64;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(rep) = record
                .entry()
                .to_app_option::<IdentityReputation>()
                .ok()
                .flatten()
            {
                // Exponential decay: weight = interactions * 2^(-age/half_life)
                let age_micros = (now_micros - rep.last_updated.as_micros() as f64).max(0.0);
                let decay =
                    (-age_micros / REPUTATION_HALF_LIFE_MICROS * core::f64::consts::LN_2).exp();
                let effective_weight = rep.interactions as f64 * decay;

                weighted_sum += rep.score * effective_weight;
                total_weight += effective_weight;
            }
        }
    }

    if total_weight < f64::EPSILON {
        Ok(0.5)
    } else {
        Ok(weighted_sum / total_weight)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AggregatedReputation {
    pub did: String,
    pub aggregate_score: f64,
    pub sources: Vec<ReputationSource>,
    pub total_interactions: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReputationSource {
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

// ==================== EVENT BROADCASTING ====================

/// Broadcast an event to all listening hApps
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = BridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type.clone(),
        subject: input.subject.clone(),
        payload: input.payload.clone(),
        source_happ: IDENTITY_HAPP_ID.to_string(),
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::BridgeEvent(event))?;

    // Link from recent events anchor
    let anchor = string_to_entry_hash(RECENT_EVENTS_ANCHOR);
    create_link(anchor, action_hash.clone(), LinkTypes::RecentEvents, ())?;

    // Link by event type for filtering
    let event_type_anchor = string_to_entry_hash(&format!("event_type:{:?}", input.event_type));
    create_link(
        event_type_anchor,
        action_hash.clone(),
        LinkTypes::EventTypeToEvents,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastEventInput {
    pub event_type: BridgeEventType,
    pub subject: String,
    pub payload: String,
}

/// Get recent events (for polling by other hApps)
#[hdk_extern]
pub fn get_recent_events(input: GetEventsInput) -> ExternResult<Vec<Record>> {
    let anchor = if let Some(ref event_type) = input.event_type {
        string_to_entry_hash(&format!("event_type:{:?}", event_type))
    } else {
        string_to_entry_hash(RECENT_EVENTS_ANCHOR)
    };

    let link_type = if input.event_type.is_some() {
        LinkTypes::EventTypeToEvents
    } else {
        LinkTypes::RecentEvents
    };

    let links = get_links(
        LinkQuery::try_new(anchor, link_type)?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    let since_micros = input.since.unwrap_or(0);

    for link in links {
        // Filter by timestamp if provided
        if link.timestamp.as_micros() < since_micros as i64 {
            continue;
        }

        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            events.push(record);
        }

        // Limit results
        if events.len() >= input.limit.unwrap_or(50) as usize {
            break;
        }
    }

    Ok(events)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetEventsInput {
    pub event_type: Option<BridgeEventType>,
    pub since: Option<u64>,
    pub limit: Option<u32>,
}

// ==================== BRIDGE NOTIFICATION ENDPOINTS ====================
// Called by other zomes (did_registry, mfa) via cross-zome calls

/// Notify bridge of DID deactivation
#[hdk_extern]
pub fn notify_did_deactivated(input: DidDeactivatedInput) -> ExternResult<Record> {
    broadcast_event(BroadcastEventInput {
        event_type: BridgeEventType::DidDeactivated,
        subject: input.did.clone(),
        payload: serde_json::json!({
            "did": input.did,
            "reason": input.reason,
            "deactivated_at": input.deactivated_at,
        })
        .to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DidDeactivatedInput {
    pub did: String,
    pub reason: String,
    pub deactivated_at: String,
}

/// Notify bridge of MFA assurance level change
#[hdk_extern]
pub fn notify_mfa_assurance_changed(input: MfaAssuranceChangedInput) -> ExternResult<Record> {
    broadcast_event(BroadcastEventInput {
        event_type: BridgeEventType::MfaAssuranceChanged,
        subject: input.did.clone(),
        payload: serde_json::json!({
            "did": input.did,
            "old_level": input.old_level,
            "new_level": input.new_level,
            "new_score": input.new_score,
        })
        .to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MfaAssuranceChangedInput {
    pub did: String,
    pub old_level: String,
    pub new_level: String,
    pub new_score: f64,
}

// ==================== CONVENIENCE FUNCTIONS ====================

// ==================== SELECTIVE DISCLOSURE QUERIES ====================

/// Query identity with selective disclosure — only returns fields the caller requests.
/// This respects privacy by limiting data exposure to what the requesting hApp actually needs.
#[hdk_extern]
pub fn query_identity_selective(
    input: SelectiveQueryInput,
) -> ExternResult<SelectiveIdentityResult> {
    let now = sys_time()?;

    if input.did.is_empty() || !input.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format — must start with did:mycelix:".into()
        )));
    }
    if input.requested_fields.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Must request at least one field".into()
        )));
    }

    // Create audit trail
    let query = IdentityQuery {
        id: format!("selective:{}:{}", input.did, now.as_micros()),
        did: input.did.clone(),
        source_happ: input.source_happ.clone(),
        requested_fields: input.requested_fields.clone(),
        queried_at: now,
    };
    let query_hash = create_entry(&EntryTypes::IdentityQuery(query))?;
    create_link(
        string_to_entry_hash(&input.did),
        query_hash,
        LinkTypes::DidToQueries,
        (),
    )?;

    let fields = &input.requested_fields;

    // Only resolve what's needed
    let mut result = SelectiveIdentityResult {
        did: input.did.clone(),
        queried_at: now,
        disclosed_fields: input.requested_fields.clone(),
        is_valid: None,
        is_deactivated: None,
        matl_score: None,
        reputation_score: None,
        credential_count: None,
        mfa_enrolled: None,
        mfa_assurance_level: None,
        mfa_assurance_score: None,
        mfa_factor_count: None,
        fl_eligible: None,
        did_created: None,
        services: None,
        verification_methods: None,
    };

    // Resolve DID details if any DID-related field is requested
    let needs_did = fields.iter().any(|f| {
        matches!(
            f.as_str(),
            "is_valid" | "is_deactivated" | "did_created" | "services" | "verification_methods"
        )
    });
    let did_details = if needs_did {
        get_did_details(&input.did)?
    } else {
        None
    };
    let did_valid = did_details.is_some();

    if fields.iter().any(|f| f == "is_valid") {
        let deactivated = is_did_deactivated(&input.did)?;
        result.is_valid = Some(did_valid && !deactivated);
    }

    if fields.iter().any(|f| f == "is_deactivated") {
        result.is_deactivated = Some(is_did_deactivated(&input.did)?);
    }

    if fields.iter().any(|f| f == "did_created") {
        result.did_created = did_details.as_ref().map(|doc| doc.created);
    }

    if fields.iter().any(|f| f == "services") {
        result.services = did_details.as_ref().map(|doc| {
            doc.service
                .iter()
                .map(|s| ServiceInfo {
                    id: s.id.clone(),
                    type_: s.type_.clone(),
                    endpoint: s.service_endpoint.clone(),
                })
                .collect()
        });
    }

    if fields.iter().any(|f| f == "verification_methods") {
        result.verification_methods = did_details.as_ref().map(|doc| {
            doc.verification_method
                .iter()
                .map(|vm| VerificationMethodInfo {
                    id: vm.id.clone(),
                    type_: vm.type_.clone(),
                })
                .collect()
        });
    }

    // MFA fields — only call MFA zome if any MFA field is requested
    let needs_mfa = fields.iter().any(|f| {
        matches!(
            f.as_str(),
            "mfa_enrolled"
                | "mfa_assurance_level"
                | "mfa_assurance_score"
                | "mfa_factor_count"
                | "fl_eligible"
                | "matl_score"
        )
    });
    let mfa_summary = if needs_mfa {
        get_mfa_summary_for_did(&input.did)?
    } else {
        None
    };

    if fields.iter().any(|f| f == "mfa_enrolled") {
        result.mfa_enrolled = Some(mfa_summary.is_some());
    }
    if fields.iter().any(|f| f == "mfa_assurance_level") {
        result.mfa_assurance_level = mfa_summary
            .as_ref()
            .map(|s| format!("{:?}", s.assurance_level));
    }
    if fields.iter().any(|f| f == "mfa_assurance_score") {
        result.mfa_assurance_score = mfa_summary.as_ref().map(|s| s.assurance_score);
    }
    if fields.iter().any(|f| f == "mfa_factor_count") {
        result.mfa_factor_count = mfa_summary.as_ref().map(|s| s.factor_count as u32);
    }
    if fields.iter().any(|f| f == "fl_eligible") {
        result.fl_eligible = mfa_summary.as_ref().map(|s| s.fl_eligible);
    }

    // Reputation-based fields
    let needs_reputation = fields
        .iter()
        .any(|f| matches!(f.as_str(), "reputation_score" | "matl_score"));
    let reputation = if needs_reputation {
        get_aggregated_reputation(&input.did)?
    } else {
        0.0
    };

    if fields.iter().any(|f| f == "reputation_score") {
        result.reputation_score = Some(reputation);
    }

    if fields.iter().any(|f| f == "matl_score") {
        let mfa_enrolled = mfa_summary.is_some();
        let mfa_score = mfa_summary
            .as_ref()
            .map(|s| s.assurance_score)
            .unwrap_or(0.0);
        let matl = if mfa_enrolled {
            (reputation * REPUTATION_WEIGHT) + (mfa_score * MFA_WEIGHT)
        } else {
            reputation
        };
        result.matl_score = Some(matl);
    }

    if fields.iter().any(|f| f == "credential_count") {
        result.credential_count = Some(count_credentials_for_did(&input.did)?);
    }

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelectiveQueryInput {
    pub did: String,
    pub source_happ: String,
    pub requested_fields: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelectiveIdentityResult {
    pub did: String,
    pub queried_at: Timestamp,
    /// Which fields were disclosed in this response
    pub disclosed_fields: Vec<String>,
    // All fields are Option — None means not disclosed
    pub is_valid: Option<bool>,
    pub is_deactivated: Option<bool>,
    pub matl_score: Option<f64>,
    pub reputation_score: Option<f64>,
    pub credential_count: Option<u32>,
    pub mfa_enrolled: Option<bool>,
    pub mfa_assurance_level: Option<String>,
    pub mfa_assurance_score: Option<f64>,
    pub mfa_factor_count: Option<u32>,
    pub fl_eligible: Option<bool>,
    pub did_created: Option<Timestamp>,
    pub services: Option<Vec<ServiceInfo>>,
    pub verification_methods: Option<Vec<VerificationMethodInfo>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServiceInfo {
    pub id: String,
    pub type_: String,
    pub endpoint: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VerificationMethodInfo {
    pub id: String,
    pub type_: String,
}

// ==================== CONVENIENCE FUNCTIONS ====================

/// Quick DID verification for other hApps
#[hdk_extern]
pub fn verify_did(did: String) -> ExternResult<bool> {
    verify_did_exists(&did)
}

/// Get MATL score for a DID (combines reputation + MFA assurance)
#[hdk_extern]
pub fn get_matl_score(did: String) -> ExternResult<f64> {
    let reputation_score = get_aggregated_reputation(&did)?;
    let mfa_score = get_mfa_assurance_for_did(&did)?;

    // If MFA enrolled (score > 0), use weighted combination
    if mfa_score > 0.0 {
        Ok((reputation_score * REPUTATION_WEIGHT) + (mfa_score * MFA_WEIGHT))
    } else {
        Ok(reputation_score)
    }
}

/// Get reputation-only score (without MFA factor)
#[hdk_extern]
pub fn get_reputation_score(did: String) -> ExternResult<f64> {
    get_aggregated_reputation(&did)
}

/// Check if DID meets trust threshold
#[hdk_extern]
pub fn is_trustworthy(input: TrustCheckInput) -> ExternResult<bool> {
    let matl_score = get_matl_score(input.did.clone())?;
    Ok(matl_score >= input.threshold)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrustCheckInput {
    pub did: String,
    pub threshold: f64,
}

/// Check if DID is eligible for FL participation (requires MFA)
#[hdk_extern]
pub fn is_fl_eligible(did: String) -> ExternResult<bool> {
    let mfa_summary = get_mfa_summary_for_did(&did)?;
    Ok(mfa_summary.map(|s| s.fl_eligible).unwrap_or(false))
}

/// Get MFA summary for a DID (exposed for other hApps)
#[hdk_extern]
pub fn get_identity_mfa_summary(did: String) -> ExternResult<Option<MfaSummary>> {
    get_mfa_summary_for_did(&did)
}

/// Enhanced trust check that considers both MATL score and MFA requirements
#[hdk_extern]
pub fn check_enhanced_trust(input: EnhancedTrustCheckInput) -> ExternResult<EnhancedTrustResult> {
    let reputation_score = get_aggregated_reputation(&input.did)?;
    let mfa_summary = get_mfa_summary_for_did(&input.did)?;

    let mfa_enrolled = mfa_summary.is_some();
    let mfa_score = mfa_summary
        .as_ref()
        .map(|s| s.assurance_score)
        .unwrap_or(0.0);
    let mfa_level = mfa_summary.as_ref().map(|s| s.assurance_level.clone());
    let fl_eligible = mfa_summary.as_ref().map(|s| s.fl_eligible).unwrap_or(false);

    // Calculate combined MATL score
    let matl_score = if mfa_enrolled {
        (reputation_score * REPUTATION_WEIGHT) + (mfa_score * MFA_WEIGHT)
    } else {
        reputation_score
    };

    // Check if meets threshold
    let meets_threshold = matl_score >= input.threshold;

    // Check if meets MFA requirement
    let meets_mfa_requirement = if input.require_mfa {
        mfa_enrolled && mfa_score >= input.min_mfa_score.unwrap_or(0.0)
    } else {
        true
    };

    // Overall trust result
    let is_trusted = meets_threshold && meets_mfa_requirement;

    Ok(EnhancedTrustResult {
        did: input.did,
        is_trusted,
        matl_score,
        reputation_score,
        mfa_score,
        mfa_level,
        mfa_enrolled,
        fl_eligible,
        meets_threshold,
        meets_mfa_requirement,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EnhancedTrustCheckInput {
    pub did: String,
    pub threshold: f64,
    pub require_mfa: bool,
    pub min_mfa_score: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EnhancedTrustResult {
    pub did: String,
    pub is_trusted: bool,
    pub matl_score: f64,
    pub reputation_score: f64,
    pub mfa_score: f64,
    pub mfa_level: Option<AssuranceLevel>,
    pub mfa_enrolled: bool,
    pub fl_eligible: bool,
    pub meets_threshold: bool,
    pub meets_mfa_requirement: bool,
}

// ==================== LIGHTWEIGHT BRIDGE QUERIES ====================
// These externs are designed for other hApps to call without creating
// audit trail entries (unlike query_identity which creates IdentityQuery
// and IdentityVerification records). Use these for frequent, low-overhead
// identity checks during normal operation.

/// Get combined DID verification status without creating audit entries.
///
/// Returns validity, deactivation status, and MFA assurance in one call.
/// Other hApps should use this instead of making separate cross-zome calls
/// to did_registry and mfa zomes directly.
#[hdk_extern]
pub fn get_did_verification_status(did: String) -> ExternResult<DidVerificationStatus> {
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format — must start with did:mycelix:".into()
        )));
    }

    let exists = verify_did_exists(&did)?;
    let deactivated = if exists {
        is_did_deactivated(&did)?
    } else {
        false
    };
    let mfa_summary = get_mfa_summary_for_did(&did)?;

    Ok(DidVerificationStatus {
        did,
        exists,
        active: exists && !deactivated,
        mfa_enrolled: mfa_summary.is_some(),
        mfa_assurance_level: mfa_summary
            .as_ref()
            .map(|s| format!("{:?}", s.assurance_level)),
        mfa_assurance_score: mfa_summary
            .as_ref()
            .map(|s| s.assurance_score)
            .unwrap_or(0.0),
    })
}

/// Lightweight DID verification status (no audit trail)
#[derive(Serialize, Deserialize, Debug)]
pub struct DidVerificationStatus {
    pub did: String,
    /// Whether the DID document exists on the DHT
    pub exists: bool,
    /// Whether the DID is active (exists and not deactivated)
    pub active: bool,
    /// Whether the DID has MFA factors enrolled
    pub mfa_enrolled: bool,
    /// MFA assurance level as string (e.g. "Basic", "Verified", "HighlyAssured")
    pub mfa_assurance_level: Option<String>,
    /// Numeric MFA assurance score (0.0 - 1.0)
    pub mfa_assurance_score: f64,
}

/// Get just the MFA assurance level for a DID (no audit trail).
///
/// Returns the assurance level string and score. Useful for other hApps
/// that need to gate features based on MFA level (e.g. FL participation).
#[hdk_extern]
pub fn get_mfa_assurance_level(did: String) -> ExternResult<MfaAssuranceLevelResult> {
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format — must start with did:mycelix:".into()
        )));
    }

    let mfa_summary = get_mfa_summary_for_did(&did)?;

    Ok(MfaAssuranceLevelResult {
        did,
        enrolled: mfa_summary.is_some(),
        assurance_level: mfa_summary
            .as_ref()
            .map(|s| format!("{:?}", s.assurance_level)),
        assurance_score: mfa_summary
            .as_ref()
            .map(|s| s.assurance_score)
            .unwrap_or(0.0),
        factor_count: mfa_summary
            .as_ref()
            .map(|s| s.factor_count as u32)
            .unwrap_or(0),
        fl_eligible: mfa_summary.as_ref().map(|s| s.fl_eligible).unwrap_or(false),
    })
}

/// MFA assurance level result (no audit trail)
#[derive(Serialize, Deserialize, Debug)]
pub struct MfaAssuranceLevelResult {
    pub did: String,
    pub enrolled: bool,
    pub assurance_level: Option<String>,
    pub assurance_score: f64,
    pub factor_count: u32,
    pub fl_eligible: bool,
}

// ==================== CONSCIOUSNESS CREDENTIAL ISSUANCE ====================

/// Issue a consciousness credential for a DID.
///
/// Aggregates identity (MFA), reputation, and community trust dimensions
/// into a `ConsciousnessProfile`. The engagement dimension is left at 0.0 —
/// the calling cluster bridge fills it in locally from its own DHT data.
#[hdk_extern]
pub fn issue_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format — must start with did:mycelix:".into()
        )));
    }

    // 1. Identity dimension — from MFA assurance level
    let identity_score = get_mfa_assurance_for_did(&did)?;

    // 2. Reputation dimension — existing aggregated reputation
    let reputation_score = get_aggregated_reputation(&did)?;

    // 3. Community dimension — aggregated peer trust credentials
    let community_score = get_community_trust_score(&did)?;

    // 4. Engagement is 0.0 here — caller's bridge fills it in locally
    let profile = ConsciousnessProfile {
        identity: identity_score,
        reputation: reputation_score,
        community: community_score,
        engagement: 0.0,
    };

    let tier = ConsciousnessTier::from_score(profile.combined_score());
    let now = sys_time()?;
    let now_us = now.as_micros() as u64;

    Ok(ConsciousnessCredential {
        did,
        profile,
        tier,
        issued_at: now_us,
        expires_at: now_us + ConsciousnessCredential::DEFAULT_TTL_US,
        issuer: format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey),
    })
}

/// Refresh a consciousness credential for a DID.
///
/// Called by cluster bridges when a credential is nearing expiry (within 2 hours).
/// Re-issues a fresh credential with updated scores from the same dimensions.
/// This avoids the 24-hour credential cliff where agents experience cold-start
/// latency on their next governance action after full expiry.
#[hdk_extern]
pub fn refresh_consciousness_credential(did: String) -> ExternResult<ConsciousnessCredential> {
    issue_consciousness_credential(did)
}

/// Compute a community trust score for a DID from peer trust credentials.
///
/// Queries all non-revoked, non-expired trust credentials where
/// `subject_did == did`, weighted by attestor tier (higher-consciousness
/// peers count more). Returns a 0.0–1.0 score.
fn get_community_trust_score(did: &str) -> ExternResult<f64> {
    // Call trust_credential zome to get credentials for this subject
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("trust_credential"),
        FunctionName::new("get_subject_credentials"),
        None,
        did.to_string(),
    )?;

    let credential_records: Vec<Record> = match response {
        ZomeCallResponse::Ok(result) => result.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode get_subject_credentials response: {:?}",
                e
            )))
        })?,
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            return Ok(0.0);
        }
        ZomeCallResponse::NetworkError(err) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Network error calling trust_credential zome: {}",
                err
            ))));
        }
        ZomeCallResponse::CountersigningSession(err) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Countersigning error: {}",
                err
            ))));
        }
    };

    if credential_records.is_empty() {
        return Ok(0.0);
    }

    let now = sys_time()?;
    let now_micros = now.as_micros() as f64;
    let mut total_weight = 0.0f64;
    let mut weighted_sum = 0.0f64;

    for record in &credential_records {
        // Decode TrustCredential from the record — uses the integrity types
        // imported via trust_credential_integrity (available in this DNA)
        let cred_data: Option<serde_json::Value> = record.entry().as_option().and_then(|entry| {
            serde_json::from_slice(&holochain_serialized_bytes::encode(entry).unwrap_or_default())
                .ok()
        });

        if let Some(cred) = cred_data {
            // Skip revoked or self-attested (issuer == subject)
            let revoked = cred
                .get("revoked")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if revoked {
                continue;
            }

            let subject = cred
                .get("subject_did")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let issuer = cred
                .get("issuer_did")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if subject == issuer {
                // Self-attestation: weight at 0.1 (low trust)
                // but still count it
            }

            // Check expiry
            if let Some(expires) = cred.get("expires_at") {
                if let Some(exp_micros) = expires.get("0").and_then(|v| v.as_i64()) {
                    if (exp_micros as f64) < now_micros {
                        continue; // Expired
                    }
                }
            }

            // Get trust score from range midpoint
            let score = cred
                .get("trust_score_range")
                .map(|r| {
                    let lower = r.get("lower").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let upper = r.get("upper").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    (lower + upper) / 2.0
                })
                .unwrap_or(0.0);

            // Weight by attestor tier — higher tiers count more
            let is_self_attest = subject == issuer;
            let attestor_weight = if is_self_attest {
                0.1 // Self-attestation has low weight
            } else {
                // Weight based on the credential's trust tier
                match cred.get("trust_tier").and_then(|v| v.as_str()) {
                    Some("Guardian") => 2.0,
                    Some("Elevated") => 1.5,
                    Some("Standard") => 1.0,
                    Some("Basic") => 0.5,
                    _ => 0.3,
                }
            };

            // Apply time decay (same 30-day half-life as reputation)
            let issued_micros = cred
                .get("issued_at")
                .and_then(|v| v.get("0").and_then(|v| v.as_i64()))
                .unwrap_or(0) as f64;
            let age_micros = (now_micros - issued_micros).max(0.0);
            let decay = (-age_micros / REPUTATION_HALF_LIFE_MICROS * core::f64::consts::LN_2).exp();

            let effective_weight = attestor_weight * decay;
            weighted_sum += score * effective_weight;
            total_weight += effective_weight;
        }
    }

    if total_weight < f64::EPSILON {
        Ok(0.0)
    } else {
        Ok((weighted_sum / total_weight).clamp(0.0, 1.0))
    }
}

// ==================== HEALTH CHECK ====================

/// Identity bridge health check — returns basic operational status.
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<IdentityBridgeHealth> {
    let agent = agent_info()?.agent_initial_pubkey;
    Ok(IdentityBridgeHealth {
        healthy: true,
        agent: format!("{}", agent),
        api_version: API_VERSION,
    })
}

/// Identity bridge health status.
#[derive(Serialize, Deserialize, Debug)]
pub struct IdentityBridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub api_version: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AssuranceLevel::to_score ---

    #[test]
    fn assurance_level_scores() {
        assert_eq!(AssuranceLevel::Anonymous.to_score(), 0.0);
        assert_eq!(AssuranceLevel::Basic.to_score(), 0.25);
        assert_eq!(AssuranceLevel::Verified.to_score(), 0.5);
        assert_eq!(AssuranceLevel::HighlyAssured.to_score(), 0.75);
        assert_eq!(AssuranceLevel::ConstitutionallyCritical.to_score(), 1.0);
    }

    #[test]
    fn assurance_scores_monotonically_increasing() {
        let levels = [
            AssuranceLevel::Anonymous,
            AssuranceLevel::Basic,
            AssuranceLevel::Verified,
            AssuranceLevel::HighlyAssured,
            AssuranceLevel::ConstitutionallyCritical,
        ];
        for i in 1..levels.len() {
            assert!(
                levels[i].to_score() > levels[i - 1].to_score(),
                "{:?} score should be > {:?} score",
                levels[i],
                levels[i - 1]
            );
        }
    }

    // --- MFA / Reputation weight constants ---

    #[test]
    fn weights_sum_to_one() {
        assert!((MFA_WEIGHT + REPUTATION_WEIGHT - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mfa_weight_is_40_percent() {
        assert_eq!(MFA_WEIGHT, 0.4);
    }

    // --- MfaSummary serde ---

    #[test]
    fn mfa_summary_json_round_trip() {
        let summary = MfaSummary {
            did: "did:mycelix:test".into(),
            assurance_level: AssuranceLevel::Verified,
            assurance_score: 0.5,
            factor_count: 3,
            category_count: 2,
            has_external_verification: true,
            fl_eligible: false,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let restored: MfaSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.did, "did:mycelix:test");
        assert_eq!(restored.assurance_level, AssuranceLevel::Verified);
        assert_eq!(restored.factor_count, 3);
        assert!(restored.has_external_verification);
        assert!(!restored.fl_eligible);
    }

    // --- EnhancedTrustResult serde ---

    #[test]
    fn enhanced_trust_result_serde() {
        let result = EnhancedTrustResult {
            did: "did:mycelix:test".into(),
            is_trusted: true,
            matl_score: 0.85,
            reputation_score: 0.9,
            mfa_score: 0.75,
            mfa_level: Some(AssuranceLevel::HighlyAssured),
            mfa_enrolled: true,
            fl_eligible: true,
            meets_threshold: true,
            meets_mfa_requirement: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: EnhancedTrustResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.matl_score, 0.85);
        assert_eq!(restored.mfa_level, Some(AssuranceLevel::HighlyAssured));
    }
}
