// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Identity Bridge Coordinator Zome
//!
//! Cross-hApp communication for identity verification, DID queries,
//! and reputation aggregation across the Mycelix ecosystem.
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use identity_bridge_integrity::*;

// Local type definition for DID document data we receive from cross-zome calls
// This mirrors the did_registry_integrity::DidDocument but avoids importing that crate
// which would cause duplicate symbol errors
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DidDocumentData {
    pub id: String,
    pub controller: AgentPubKey,
    pub verification_method: Vec<VerificationMethodData>,
    pub authentication: Vec<String>,
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
    pub type_: String,
    pub controller: String,
    pub public_key_multibase: String,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ServiceEndpointData {
    pub id: String,
    pub type_: String,
    pub service_endpoint: String,
}

// ==================== CONSTANTS ====================

const IDENTITY_HAPP_ID: &str = "mycelix-identity";
const REGISTERED_HAPPS_ANCHOR: &str = "registered_happs";
const RECENT_EVENTS_ANCHOR: &str = "recent_events";

// ==================== HELPER FUNCTIONS ====================

/// Create a deterministic entry hash from a string identifier
/// This is used for link bases when we need to link from string IDs
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>()
            .try_into()
            .expect("36 bytes"),
    )
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

/// Query a DID from another hApp (cross-hApp identity lookup)
#[hdk_extern]
pub fn query_identity(input: QueryIdentityInput) -> ExternResult<IdentityVerificationResult> {
    let now = sys_time()?;

    // Create query record for audit trail
    let query = IdentityQuery {
        id: format!("query:{}:{}", input.did, now.as_micros()),
        did: input.did.clone(),
        source_happ: input.source_happ.clone(),
        requested_fields: input.requested_fields.clone(),
        queried_at: now,
    };
    let query_hash = create_entry(&EntryTypes::IdentityQuery(query))?;

    // Link query to DID for tracking
    create_link(
        string_to_entry_hash(&input.did),
        query_hash,
        LinkTypes::DidToQueries,
        (),
    )?;

    // Get DID details from did_registry (includes creation timestamp)
    let did_details = get_did_details(&input.did)?;
    let did_valid = did_details.is_some();
    let did_created = did_details.as_ref().map(|doc| doc.created);

    // Check if DID is deactivated
    let is_deactivated = is_did_deactivated(&input.did)?;

    // Get aggregated reputation
    let matl_score = get_aggregated_reputation(&input.did)?;

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

/// Count credentials for a DID (calls credential_schema zome)
/// Note: Currently counts schemas authored by this DID as a proxy metric.
/// In a complete system, this would query a credentials/issuance zome for actual issued credentials.
fn count_credentials_for_did(did: &str) -> ExternResult<u32> {
    // Call credential_schema zome to get schemas by author
    // This counts schemas authored by the DID, which serves as a participation metric
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("credential_schema"),
        FunctionName::new("get_schemas_by_author"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let schemas: Vec<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode get_schemas_by_author response: {:?}",
                    e
                )))
            })?;
            Ok(schemas.len() as u32)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // If unauthorized, return 0 rather than failing
            // This allows queries to succeed even with limited permissions
            Ok(0)
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling credential_schema: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            // If authentication failed, return 0 rather than failing
            Ok(0)
        }
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

/// Internal function to compute aggregated reputation
fn get_aggregated_reputation(did: &str) -> ExternResult<f64> {
    let did_hash = string_to_entry_hash(did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToReputations)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(0.5); // Default for unknown agents
    }

    let mut total_weight = 0u64;
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
                weighted_sum += rep.score * rep.interactions as f64;
                total_weight += rep.interactions;
            }
        }
    }

    if total_weight == 0 {
        Ok(0.5)
    } else {
        Ok(weighted_sum / total_weight as f64)
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

// ==================== CONVENIENCE FUNCTIONS ====================

/// Quick DID verification for other hApps
#[hdk_extern]
pub fn verify_did(did: String) -> ExternResult<bool> {
    verify_did_exists(&did)
}

/// Get MATL score for a DID
#[hdk_extern]
pub fn get_matl_score(did: String) -> ExternResult<f64> {
    get_aggregated_reputation(&did)
}

/// Check if DID meets trust threshold
#[hdk_extern]
pub fn is_trustworthy(input: TrustCheckInput) -> ExternResult<bool> {
    let score = get_aggregated_reputation(&input.did)?;
    Ok(score >= input.threshold)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrustCheckInput {
    pub did: String,
    pub threshold: f64,
}
