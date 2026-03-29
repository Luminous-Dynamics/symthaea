// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bridge Coordinator Zome
//!
//! Implements inter-hApp communication for the Mycelix ecosystem.
//! Enables cross-hApp reputation queries, credential verification,
//! and event broadcasting.
//!
//! ## API Overview
//!
//! ### hApp Registration
//! - `register_happ` - Register a hApp with the bridge
//! - `deregister_happ` - Deactivate a hApp registration
//! - `get_registered_happs` - List all registered hApps
//! - `get_happ_registration` - Get specific hApp details
//!
//! ### Reputation Management
//! - `record_reputation` - Record reputation for an agent from a hApp
//! - `query_reputation` - Query reputation with optional hApp filter
//! - `aggregate_cross_happ_reputation` - Compute aggregate across all hApps
//! - `is_agent_trustworthy` - Check if agent meets trust threshold
//!
//! ### Credential Verification
//! - `request_credential_verification` - Request verification from another hApp
//! - `verify_credential` - Provide verification response
//! - `get_verification_requests` - Get pending verification requests
//! - `get_verification_response` - Get response for a request
//!
//! ### Event Broadcasting
//! - `broadcast_event` - Broadcast event to registered hApps
//! - `get_events` - Query events by type and time range

use hdk::prelude::*;
use bridge_integrity::{
    HappRegistration, ReputationRecord, CrossHappReputationRecord,
    BridgeEventRecord, CredentialVerificationRequest, CredentialVerificationResponse,
    EthereumPaymentIntent, EthereumAnchorIntent,
    EntryTypes, LinkTypes,
    MAX_ID_LENGTH, MAX_EVENT_PAYLOAD_SIZE,
};
use std::collections::HashMap;

// SDK Bridge types
use mycelix_sdk::bridge::{HappReputationScore, CrossHappReputation};

// =============================================================================
// SECURITY: Authorization & Rate Limiting
// =============================================================================

/// Rate limit for reputation updates per agent per minute
const MAX_REPUTATION_UPDATES_PER_MINUTE: u32 = 30;

/// Rate limit for event broadcasts per minute
const MAX_BROADCASTS_PER_MINUTE: u32 = 100;

/// Validate string field length and characters
fn validate_string_field(field: &str, name: &str) -> ExternResult<()> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("{} cannot be empty", name)
        )));
    }
    if trimmed.len() > MAX_ID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("{} exceeds maximum length of {} characters", name, MAX_ID_LENGTH)
        )));
    }
    Ok(())
}

/// Check if caller is the registered owner of a hApp
fn require_happ_owner(happ_id: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    let happ_path = Path::from(format!("happs.{}", happ_id));
    let happ_hash = match happ_path.clone().typed(LinkTypes::HappIdToRegistration) {
        Ok(typed) => {
            if !typed.exists()? {
                // hApp not registered, allow first registration
                return Ok(());
            }
            typed.path_entry_hash()?
        }
        Err(e) => return Err(e),
    };

    let links = get_links(
        LinkQuery::new(
            happ_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappIdToRegistration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reg) = HappRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if reg.registrant == agent {
                            return Ok(());
                        }
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Only the hApp owner can perform this action".to_string()
                        )));
                    }
                }
            }
        }
    }

    Ok(()) // No registration found, allow
}

/// Check rate limit for operations
fn check_rate_limit(action: &str, max_per_minute: u32) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;
    let window_start = now - 60;

    let rate_path = Path::from(format!("rate.{}.{}", action, &agent[..16.min(agent.len())]));
    let rate_hash = match rate_path.clone().typed(LinkTypes::HappIdToRegistration) {
        Ok(typed) => typed.path_entry_hash()?,
        Err(_) => return Ok(()),
    };

    let links = get_links(
        LinkQuery::new(
            rate_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappIdToRegistration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let recent_count = links.iter()
        .filter(|l| l.timestamp.0 as i64 / 1_000_000 > window_start)
        .count() as u32;

    if recent_count >= max_per_minute {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Rate limit exceeded: {} per minute maximum for {}", max_per_minute, action)
        )));
    }

    Ok(())
}

/// Generate a unique request ID
fn generate_request_id() -> ExternResult<String> {
    let timestamp = sys_time()?.0;
    let agent = agent_info()?.agent_initial_pubkey;
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    hasher.update(timestamp.to_le_bytes());
    hasher.update(agent.get_raw_39());
    Ok(format!("{:x}", hasher.finalize())[..32].to_string())
}

// =============================================================================
// hApp Registration
// =============================================================================

/// Input for registering a hApp
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterHappInput {
    /// Human-readable hApp name
    pub name: String,
    /// DNA hash for cryptographic verification
    pub dna_hash: Option<String>,
    /// Capabilities this hApp supports
    pub capabilities: Vec<String>,
}

/// Register a hApp with the bridge
///
/// This registers a hApp to enable cross-hApp reputation sharing and
/// credential verification. The registering agent becomes the owner.
#[hdk_extern]
pub fn register_happ(input: RegisterHappInput) -> ExternResult<ActionHash> {
    // Input validation
    validate_string_field(&input.name, "hApp name")?;

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as u64 / 1_000_000;

    // Generate hApp ID from name and registrant
    let happ_id = if let Some(ref dna) = input.dna_hash {
        dna.clone()
    } else {
        // Generate from name + agent
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        hasher.update(input.name.as_bytes());
        hasher.update(agent_info.agent_initial_pubkey.get_raw_39());
        format!("{:x}", hasher.finalize())[..32].to_string()
    };

    let registration = HappRegistration {
        happ_id: happ_id.clone(),
        dna_hash: input.dna_hash,
        happ_name: input.name,
        capabilities: input.capabilities,
        registered_at: now,
        registrant: agent_info.agent_initial_pubkey.to_string(),
        active: true,
    };

    let action_hash = create_entry(EntryTypes::HappRegistration(registration))?;

    // Link from hApp ID path to registration
    let happ_path = Path::from(format!("happs.{}", happ_id));
    let happ_hash = ensure_path(happ_path, LinkTypes::HappIdToRegistration)?;

    create_link(
        happ_hash,
        action_hash.clone(),
        LinkTypes::HappIdToRegistration,
        vec![],
    )?;

    // Link to all happs collection
    let all_happs_path = Path::from("all_happs");
    let all_happs_hash = ensure_path(all_happs_path, LinkTypes::AllHapps)?;

    create_link(
        all_happs_hash,
        action_hash.clone(),
        LinkTypes::AllHapps,
        vec![],
    )?;

    // Broadcast registration event
    let _ = broadcast_event(BroadcastEventInput {
        event_type: "happ_registered".to_string(),
        payload: serde_json::to_vec(&serde_json::json!({
            "happ_id": happ_id,
        })).unwrap_or_default(),
        targets: vec![],
        priority: 0,
    });

    Ok(action_hash)
}

/// Deactivate a hApp registration
#[hdk_extern]
pub fn deregister_happ(happ_id: String) -> ExternResult<ActionHash> {
    require_happ_owner(&happ_id)?;

    // Get current registration
    let happ_path = Path::from(format!("happs.{}", happ_id));
    let happ_hash = ensure_path(happ_path, LinkTypes::HappIdToRegistration)?;

    let links = get_links(
        LinkQuery::new(
            happ_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappIdToRegistration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut reg) = HappRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        reg.active = false;
                        let new_hash = update_entry(action_hash, &EntryTypes::HappRegistration(reg))?;

                        // Broadcast deregistration event
                        let _ = broadcast_event(BroadcastEventInput {
                            event_type: "happ_deregistered".to_string(),
                            payload: serde_json::to_vec(&serde_json::json!({
                                "happ_id": happ_id,
                            })).unwrap_or_default(),
                            targets: vec![],
                            priority: 0,
                        });

                        return Ok(new_hash);
                    }
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "hApp not found".to_string()
    )))
}

/// Get all registered hApps
#[hdk_extern]
pub fn get_registered_happs(_: ()) -> ExternResult<Vec<HappRegistration>> {
    let all_happs_path = Path::from("all_happs");
    let all_happs_hash = match ensure_path(all_happs_path, LinkTypes::AllHapps) {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            all_happs_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AllHapps as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut registrations = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reg) = HappRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if reg.active {
                            registrations.push(reg);
                        }
                    }
                }
            }
        }
    }

    Ok(registrations)
}

/// Get a specific hApp registration
#[hdk_extern]
pub fn get_happ_registration(happ_id: String) -> ExternResult<Option<HappRegistration>> {
    let happ_path = Path::from(format!("happs.{}", happ_id));
    let happ_hash = match ensure_path(happ_path, LinkTypes::HappIdToRegistration) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            happ_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappIdToRegistration as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(reg) = HappRegistration::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(reg));
                    }
                }
            }
        }
    }

    Ok(None)
}

// =============================================================================
// Reputation Management
// =============================================================================

/// Input for recording reputation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordReputationInput {
    /// Agent to record reputation for
    pub agent: String,
    /// hApp ID recording the reputation
    pub happ_id: String,
    /// hApp name
    pub happ_name: String,
    /// Reputation score [0.0, 1.0]
    pub score: f64,
    /// Number of positive interactions
    pub interactions: u64,
    /// Number of negative interactions
    pub negative_interactions: u64,
    /// Evidence hash (optional)
    pub evidence_hash: Option<String>,
}

/// Record reputation for an agent from a specific hApp
///
/// SECURITY: Only hApp owners can record reputation for their hApp
#[hdk_extern]
pub fn record_reputation(input: RecordReputationInput) -> ExternResult<ActionHash> {
    // Input validation
    validate_string_field(&input.agent, "Agent ID")?;
    validate_string_field(&input.happ_id, "hApp ID")?;
    validate_string_field(&input.happ_name, "hApp name")?;

    // Authorization: Must be hApp owner
    require_happ_owner(&input.happ_id)?;

    // Rate limiting
    check_rate_limit("reputation", MAX_REPUTATION_UPDATES_PER_MINUTE)?;

    // Validate score range
    if input.score < 0.0 || input.score > 1.0 || !input.score.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Score must be between 0.0 and 1.0".to_string()
        )));
    }

    let now = sys_time()?.0 as u64 / 1_000_000;

    let record = ReputationRecord {
        agent: input.agent.clone(),
        happ_id: input.happ_id.clone(),
        happ_name: input.happ_name,
        score: input.score,
        interactions: input.interactions,
        negative_interactions: input.negative_interactions,
        updated_at: now,
        evidence_hash: input.evidence_hash,
    };

    let action_hash = create_entry(EntryTypes::ReputationRecord(record))?;

    // Link from agent to reputation
    let agent_path = Path::from(format!("agent_rep.{}", input.agent));
    let agent_hash = ensure_path(agent_path, LinkTypes::AgentToReputations)?;

    create_link(
        agent_hash,
        action_hash.clone(),
        LinkTypes::AgentToReputations,
        input.happ_id.as_bytes().to_vec(), // Store happ_id in tag for filtering
    )?;

    // Link from hApp to reputation
    let happ_path = Path::from(format!("happ_rep.{}", input.happ_id));
    let happ_hash = ensure_path(happ_path, LinkTypes::HappToReputations)?;

    create_link(
        happ_hash,
        action_hash.clone(),
        LinkTypes::HappToReputations,
        vec![],
    )?;

    // Broadcast reputation update event
    let _ = broadcast_event(BroadcastEventInput {
        event_type: "reputation_update".to_string(),
        payload: serde_json::to_vec(&serde_json::json!({
            "agent": input.agent,
            "happ_id": input.happ_id,
            "score": input.score,
        })).unwrap_or_default(),
        targets: vec![],
        priority: 0,
    });

    Ok(action_hash)
}

/// Input for reputation query
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryReputationInput {
    /// Agent to query
    pub agent: String,
    /// Optional hApp filter (None = all hApps)
    pub happ: Option<String>,
}

/// Query reputation for an agent with optional hApp filter
#[hdk_extern]
pub fn query_reputation(input: QueryReputationInput) -> ExternResult<Vec<ReputationRecord>> {
    let agent_path = Path::from(format!("agent_rep.{}", input.agent));
    let agent_hash = match ensure_path(agent_path, LinkTypes::AgentToReputations) {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            agent_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToReputations as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        // Filter by hApp if specified
        if let Some(ref happ_filter) = input.happ {
            let tag_str = String::from_utf8_lossy(link.tag.as_ref());
            if tag_str != *happ_filter {
                continue;
            }
        }

        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(rep) = ReputationRecord::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        records.push(rep);
                    }
                }
            }
        }
    }

    // Sort by updated_at descending (most recent first)
    records.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    Ok(records)
}

/// Aggregate cross-hApp reputation for an agent
///
/// Computes a weighted aggregate score based on interaction counts across all hApps.
/// Stores the result for efficient future queries.
#[hdk_extern]
pub fn aggregate_cross_happ_reputation(agent: String) -> ExternResult<CrossHappReputation> {
    // Get all reputation records for this agent
    let records = query_reputation(QueryReputationInput {
        agent: agent.clone(),
        happ: None,
    })?;

    // Group by hApp and get latest record for each
    let mut latest_by_happ: HashMap<String, ReputationRecord> = HashMap::new();
    for record in records {
        let existing = latest_by_happ.get(&record.happ_id);
        if existing.map_or(true, |e| e.updated_at < record.updated_at) {
            latest_by_happ.insert(record.happ_id.clone(), record);
        }
    }

    // Convert to SDK format
    let scores: Vec<HappReputationScore> = latest_by_happ
        .values()
        .map(|r| HappReputationScore {
            happ_id: r.happ_id.clone(),
            happ_name: r.happ_name.clone(),
            score: r.score,
            interactions: r.interactions,
            last_updated: r.updated_at,
        })
        .collect();

    let cross_happ = CrossHappReputation::from_scores(&agent, scores.clone());

    // Store the aggregate for efficient future queries
    let now = sys_time()?.0 as u64 / 1_000_000;

    // Serialize scores to JSON
    let scores_map: HashMap<String, f64> = scores.iter()
        .map(|s| (s.happ_id.clone(), s.score))
        .collect();

    let aggregate_record = CrossHappReputationRecord {
        agent: agent.clone(),
        scores_json: serde_json::to_string(&scores_map).unwrap_or_default(),
        aggregated_score: cross_happ.aggregate,
        total_interactions: cross_happ.total_interactions(),
        happ_count: scores.len() as u32,
        computed_at: now,
    };

    let action_hash = create_entry(EntryTypes::CrossHappReputationRecord(aggregate_record))?;

    // Link from agent to aggregate
    let agg_path = Path::from(format!("agg_rep.{}", agent));
    let agg_hash = ensure_path(agg_path, LinkTypes::AgentToAggregateReputation)?;

    create_link(
        agg_hash,
        action_hash,
        LinkTypes::AgentToAggregateReputation,
        vec![],
    )?;

    Ok(cross_happ)
}

/// Check if an agent is trustworthy based on cross-hApp reputation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustCheckInput {
    pub agent: String,
    pub threshold: f64,
}

#[hdk_extern]
pub fn is_agent_trustworthy(input: TrustCheckInput) -> ExternResult<bool> {
    let reputation = aggregate_cross_happ_reputation(input.agent)?;
    Ok(reputation.is_trustworthy(input.threshold))
}

// =============================================================================
// Credential Verification
// =============================================================================

/// Input for requesting credential verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CredentialVerifyInput {
    /// Hash of the credential to verify
    pub credential_hash: String,
    /// Type of credential
    pub credential_type: String,
    /// Issuing hApp ID
    pub issuer_happ: String,
    /// Agent the credential belongs to
    pub subject_agent: String,
}

/// Request verification of a credential from another hApp
#[hdk_extern]
pub fn request_credential_verification(input: CredentialVerifyInput) -> ExternResult<ActionHash> {
    validate_string_field(&input.credential_hash, "Credential hash")?;
    validate_string_field(&input.credential_type, "Credential type")?;
    validate_string_field(&input.issuer_happ, "Issuer hApp")?;
    validate_string_field(&input.subject_agent, "Subject agent")?;

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as u64 / 1_000_000;
    let request_id = generate_request_id()?;

    let request = CredentialVerificationRequest {
        request_id: request_id.clone(),
        credential_hash: input.credential_hash,
        credential_type: input.credential_type,
        issuer_happ: input.issuer_happ.clone(),
        subject_agent: input.subject_agent,
        requester: agent_info.agent_initial_pubkey.to_string(),
        requested_at: now,
        status: "pending".to_string(),
    };

    let action_hash = create_entry(EntryTypes::CredentialVerificationRequest(request))?;

    // Link from hApp to request
    let happ_path = Path::from(format!("cred_verify.{}", input.issuer_happ));
    let happ_hash = ensure_path(happ_path, LinkTypes::HappToCredentialRequests)?;

    create_link(
        happ_hash,
        action_hash.clone(),
        LinkTypes::HappToCredentialRequests,
        request_id.as_bytes().to_vec(),
    )?;

    // Create global index for request lookup by request_id
    // This enables verify_credential to look up the original request
    let index_path = Path::from(format!("cred_request_index.{}", request_id));
    let index_hash = ensure_path(index_path, LinkTypes::HappToCredentialRequests)?;

    create_link(
        index_hash,
        action_hash.clone(),
        LinkTypes::HappToCredentialRequests,
        vec![],
    )?;

    Ok(action_hash)
}

/// Input for providing verification response
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VerifyCredentialInput {
    /// Request ID to respond to
    pub request_id: String,
    /// Whether the credential is valid
    pub is_valid: bool,
    /// Issuer identifier
    pub issuer: String,
    /// Claims in the credential (JSON array)
    pub claims: Vec<String>,
    /// When the credential was issued
    pub issued_at: u64,
    /// When the credential expires (0 = no expiry)
    pub expires_at: u64,
    /// Optional cryptographic proof
    pub proof: Option<Vec<u8>>,
}

/// Look up a credential verification request by its request_id
fn lookup_credential_request(request_id: &str) -> ExternResult<Option<CredentialVerificationRequest>> {
    // Requests are linked from hApp paths with request_id as link tag
    // We need to search across all hApp paths or use a global request index

    // First, try the global request index path
    let request_path = Path::from(format!("cred_request_index.{}", request_id));
    let request_hash = match request_path.clone().typed(LinkTypes::HappToCredentialRequests) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(None);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            request_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappToCredentialRequests as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(req) = CredentialVerificationRequest::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if req.request_id == request_id {
                            return Ok(Some(req));
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Verify a credential (respond to verification request)
///
/// This should be called by the issuing hApp to provide verification.
#[hdk_extern]
pub fn verify_credential(input: VerifyCredentialInput) -> ExternResult<ActionHash> {
    validate_string_field(&input.request_id, "Request ID")?;
    validate_string_field(&input.issuer, "Issuer")?;

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as u64 / 1_000_000;

    // Look up the original request to get credential_type
    let credential_type = match lookup_credential_request(&input.request_id)? {
        Some(request) => request.credential_type,
        None => {
            // Fallback: if we can't find the request, require credential_type in input
            // This maintains backwards compatibility while encouraging proper lookup
            debug!("WARNING: Could not find original request for {}, using 'unverified' type", input.request_id);
            "unverified".to_string()
        }
    };

    let response = CredentialVerificationResponse {
        request_id: input.request_id.clone(),
        is_valid: input.is_valid,
        issuer: input.issuer,
        credential_type,
        claims_json: serde_json::to_string(&input.claims).unwrap_or_default(),
        issued_at: input.issued_at,
        expires_at: input.expires_at,
        verified_at: now,
        verifier: agent_info.agent_initial_pubkey.to_string(),
        proof: input.proof,
    };

    let action_hash = create_entry(EntryTypes::CredentialVerificationResponse(response))?;

    // Link from request ID to response
    let req_path = Path::from(format!("cred_response.{}", input.request_id));
    let req_hash = ensure_path(req_path, LinkTypes::RequestToResponse)?;

    create_link(
        req_hash,
        action_hash.clone(),
        LinkTypes::RequestToResponse,
        vec![],
    )?;

    Ok(action_hash)
}

/// Get pending credential verification requests for a hApp
#[hdk_extern]
pub fn get_verification_requests(happ_id: String) -> ExternResult<Vec<CredentialVerificationRequest>> {
    let happ_path = Path::from(format!("cred_verify.{}", happ_id));
    let happ_hash = match ensure_path(happ_path, LinkTypes::HappToCredentialRequests) {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            happ_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::HappToCredentialRequests as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(req) = CredentialVerificationRequest::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if req.status == "pending" {
                            requests.push(req);
                        }
                    }
                }
            }
        }
    }

    Ok(requests)
}

/// Get verification response for a request
#[hdk_extern]
pub fn get_verification_response(request_id: String) -> ExternResult<Option<CredentialVerificationResponse>> {
    let req_path = Path::from(format!("cred_response.{}", request_id));
    let req_hash = match ensure_path(req_path, LinkTypes::RequestToResponse) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            req_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RequestToResponse as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(resp) = CredentialVerificationResponse::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(resp));
                    }
                }
            }
        }
    }

    Ok(None)
}

// =============================================================================
// Event Broadcasting
// =============================================================================

/// Input for broadcasting an event
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BroadcastEventInput {
    /// Event type
    pub event_type: String,
    /// Event payload
    pub payload: Vec<u8>,
    /// Target hApps (empty = all)
    pub targets: Vec<String>,
    /// Priority (0=normal, 1=high, 2=critical)
    pub priority: u8,
}

/// Broadcast an event to other hApps
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<ActionHash> {
    validate_string_field(&input.event_type, "Event type")?;

    // Rate limiting
    check_rate_limit("broadcast", MAX_BROADCASTS_PER_MINUTE)?;

    // Validate payload size
    if input.payload.len() > MAX_EVENT_PAYLOAD_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Event payload exceeds maximum size of {} bytes", MAX_EVENT_PAYLOAD_SIZE)
        )));
    }

    let now = sys_time()?.0 as u64 / 1_000_000;
    let agent_info = agent_info()?;

    let event_record = BridgeEventRecord {
        event_type: input.event_type.clone(),
        source_happ: agent_info.agent_initial_pubkey.to_string(),
        payload: input.payload,
        timestamp: now,
        targets: input.targets,
        priority: input.priority.min(2),
    };

    let action_hash = create_entry(EntryTypes::BridgeEventRecord(event_record))?;

    // Link from event type to event
    let event_path = Path::from(format!("events.{}", input.event_type));
    let event_hash = ensure_path(event_path, LinkTypes::EventTypeToEvents)?;

    create_link(
        event_hash,
        action_hash.clone(),
        LinkTypes::EventTypeToEvents,
        vec![],
    )?;

    Ok(action_hash)
}

/// Input for querying events
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetEventsInput {
    /// Event type to query
    pub event_type: String,
    /// Only return events after this timestamp
    pub since: u64,
    /// Maximum number of events to return
    pub limit: Option<usize>,
}

/// Get events of a specific type since a timestamp
#[hdk_extern]
pub fn get_events(input: GetEventsInput) -> ExternResult<Vec<BridgeEventRecord>> {
    let event_path = Path::from(format!("events.{}", input.event_type));
    let event_hash = match ensure_path(event_path, LinkTypes::EventTypeToEvents) {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            event_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::EventTypeToEvents as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(event) = BridgeEventRecord::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if event.timestamp >= input.since {
                            events.push(event);
                        }
                    }
                }
            }
        }
    }

    // Sort by timestamp
    events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    // Apply limit if specified
    if let Some(limit) = input.limit {
        events.truncate(limit);
    }

    Ok(events)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Ensure path exists and return its hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.clone().typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// =============================================================================
// Tests
// =============================================================================

// =============================================================================
// Ethereum Bridge Integration
// =============================================================================
// These functions enable Holochain↔Ethereum communication via the Bridge.
// IMPORTANT: WASM zomes cannot make HTTP calls directly.
// Instead, they create intents and emit signals for the native Host to process.

/// Signal types for Ethereum bridge operations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EthereumBridgeSignal {
    /// Request to distribute payment on Ethereum
    PaymentRequest {
        intent_id: String,
        model_id: String,
        round: u64,
        total_amount_wei: String,
        splits_json: String,
        platform_fee_bps: u64,
    },
    /// Request to anchor data on Ethereum
    AnchorRequest {
        intent_id: String,
        anchor_type: String,
        data_hash: String,
        metadata_json: Option<String>,
    },
}

/// Input for distributing payment on Ethereum
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DistributePaymentInput {
    /// Model ID for payment routing
    pub model_id: String,
    /// FL round number
    pub round: u64,
    /// Total amount to distribute (in wei as string)
    pub total_amount_wei: String,
    /// Payment splits as JSON array
    pub splits_json: String,
    /// Platform fee in basis points (e.g., 500 = 5%)
    pub platform_fee_bps: u64,
}

/// Result of payment distribution request
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DistributePaymentResult {
    /// Intent ID for tracking
    pub intent_id: String,
    /// DHT action hash
    pub action_hash: ActionHash,
    /// Current status
    pub status: String,
}

/// Distribute payment on Ethereum chain
///
/// Creates an intent record and emits a signal for the native Host (FL Aggregator)
/// to process. The Host listens for these signals and executes the actual
/// Ethereum transaction via the EthereumClient.
///
/// # Arguments
/// * `input` - Payment distribution parameters
///
/// # Returns
/// * `DistributePaymentResult` with intent ID for tracking
#[hdk_extern]
pub fn distribute_payment_on_chain(input: DistributePaymentInput) -> ExternResult<DistributePaymentResult> {
    // Validate input
    validate_string_field(&input.model_id, "Model ID")?;
    if input.total_amount_wei.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Total amount cannot be empty".to_string()
        )));
    }
    if input.platform_fee_bps > 10000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Platform fee cannot exceed 10000 basis points (100%)".to_string()
        )));
    }

    // Validate splits JSON
    if serde_json::from_str::<serde_json::Value>(&input.splits_json).is_err() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "splits_json must be valid JSON".to_string()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as u64 / 1_000_000;
    let intent_id = generate_request_id()?;

    // Create the payment intent entry
    let intent = EthereumPaymentIntent {
        intent_id: intent_id.clone(),
        model_id: input.model_id.clone(),
        round: input.round,
        total_amount_wei: input.total_amount_wei.clone(),
        splits_json: input.splits_json.clone(),
        platform_fee_bps: input.platform_fee_bps,
        status: "pending".to_string(),
        tx_hash: None,
        block_number: None,
        created_at: now,
        updated_at: now,
        requester: agent_info.agent_initial_pubkey.to_string(),
        error: None,
    };

    let action_hash = create_entry(EntryTypes::EthereumPaymentIntent(intent))?;

    // Link to pending intents (for Host polling)
    let pending_path = Path::from("pending_ethereum_intents");
    let pending_hash = ensure_path(pending_path, LinkTypes::PendingEthereumIntents)?;

    create_link(
        pending_hash,
        action_hash.clone(),
        LinkTypes::PendingEthereumIntents,
        intent_id.as_bytes().to_vec(),
    )?;

    // Link by intent ID for lookup
    let intent_path = Path::from(format!("eth_intent.{}", intent_id));
    let intent_hash = ensure_path(intent_path, LinkTypes::IntentIdToPaymentIntent)?;

    create_link(
        intent_hash,
        action_hash.clone(),
        LinkTypes::IntentIdToPaymentIntent,
        vec![],
    )?;

    // Link by model ID
    let model_path = Path::from(format!("model_payments.{}", input.model_id));
    let model_hash = ensure_path(model_path, LinkTypes::ModelToPaymentIntents)?;

    create_link(
        model_hash,
        action_hash.clone(),
        LinkTypes::ModelToPaymentIntents,
        vec![],
    )?;

    // Emit signal for Host to process
    // The native FL Aggregator listens for this signal and executes the Ethereum transaction
    emit_signal(EthereumBridgeSignal::PaymentRequest {
        intent_id: intent_id.clone(),
        model_id: input.model_id,
        round: input.round,
        total_amount_wei: input.total_amount_wei,
        splits_json: input.splits_json,
        platform_fee_bps: input.platform_fee_bps,
    })?;

    // Also broadcast via Bridge events for observability
    let _ = broadcast_event(BroadcastEventInput {
        event_type: "ethereum_payment_requested".to_string(),
        payload: serde_json::to_vec(&serde_json::json!({
            "intent_id": intent_id,
            "round": input.round,
        })).unwrap_or_default(),
        targets: vec![],
        priority: 1, // High priority
    });

    Ok(DistributePaymentResult {
        intent_id,
        action_hash,
        status: "pending".to_string(),
    })
}

/// Input for anchoring data on Ethereum
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnchorToEthereumInput {
    /// Type of anchor: "reputation", "contribution", "proof", "model", "round"
    pub anchor_type: String,
    /// Data to anchor (hash or commitment)
    pub data_hash: String,
    /// Optional metadata JSON
    pub metadata_json: Option<String>,
}

/// Result of anchor request
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnchorResult {
    /// Intent ID for tracking
    pub intent_id: String,
    /// DHT action hash
    pub action_hash: ActionHash,
    /// Current status
    pub status: String,
}

/// Anchor data to Ethereum chain
///
/// Creates an anchor intent and emits a signal for the native Host to process.
///
/// # Arguments
/// * `input` - Anchor parameters
///
/// # Returns
/// * `AnchorResult` with intent ID for tracking
#[hdk_extern]
pub fn anchor_to_ethereum(input: AnchorToEthereumInput) -> ExternResult<AnchorResult> {
    // Validate input
    let valid_anchor_types = ["reputation", "contribution", "proof", "model", "round"];
    if !valid_anchor_types.contains(&input.anchor_type.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Anchor type must be one of: {}",
                valid_anchor_types.join(", ")
            )
        )));
    }

    validate_string_field(&input.data_hash, "Data hash")?;

    // Validate metadata JSON if provided
    if let Some(ref metadata) = input.metadata_json {
        if serde_json::from_str::<serde_json::Value>(metadata).is_err() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "metadata_json must be valid JSON".to_string()
            )));
        }
    }

    let agent_info = agent_info()?;
    let now = sys_time()?.0 as u64 / 1_000_000;
    let intent_id = generate_request_id()?;

    // Create the anchor intent entry
    let intent = EthereumAnchorIntent {
        intent_id: intent_id.clone(),
        anchor_type: input.anchor_type.clone(),
        data_hash: input.data_hash.clone(),
        metadata_json: input.metadata_json.clone(),
        status: "pending".to_string(),
        tx_hash: None,
        block_number: None,
        block_hash: None,
        commitment_id: None,
        created_at: now,
        updated_at: now,
        requester: agent_info.agent_initial_pubkey.to_string(),
        error: None,
    };

    let action_hash = create_entry(EntryTypes::EthereumAnchorIntent(intent))?;

    // Link to pending intents
    let pending_path = Path::from("pending_ethereum_intents");
    let pending_hash = ensure_path(pending_path, LinkTypes::PendingEthereumIntents)?;

    create_link(
        pending_hash,
        action_hash.clone(),
        LinkTypes::PendingEthereumIntents,
        intent_id.as_bytes().to_vec(),
    )?;

    // Link by intent ID
    let intent_path = Path::from(format!("eth_anchor.{}", intent_id));
    let intent_hash = ensure_path(intent_path, LinkTypes::IntentIdToAnchorIntent)?;

    create_link(
        intent_hash,
        action_hash.clone(),
        LinkTypes::IntentIdToAnchorIntent,
        vec![],
    )?;

    // Emit signal for Host to process
    emit_signal(EthereumBridgeSignal::AnchorRequest {
        intent_id: intent_id.clone(),
        anchor_type: input.anchor_type,
        data_hash: input.data_hash,
        metadata_json: input.metadata_json,
    })?;

    // Broadcast event
    let _ = broadcast_event(BroadcastEventInput {
        event_type: "ethereum_anchor_requested".to_string(),
        payload: serde_json::to_vec(&serde_json::json!({
            "intent_id": intent_id,
        })).unwrap_or_default(),
        targets: vec![],
        priority: 0,
    });

    Ok(AnchorResult {
        intent_id,
        action_hash,
        status: "pending".to_string(),
    })
}

/// Get all pending Ethereum intents
///
/// Used by the native Host to poll for new intents to process.
#[hdk_extern]
pub fn get_pending_ethereum_intents(_: ()) -> ExternResult<Vec<ActionHash>> {
    let pending_path = Path::from("pending_ethereum_intents");
    let pending_hash = match ensure_path(pending_path, LinkTypes::PendingEthereumIntents) {
        Ok(h) => h,
        Err(_) => return Ok(Vec::new()),
    };

    let links = get_links(
        LinkQuery::new(
            pending_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::PendingEthereumIntents as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let hashes: Vec<ActionHash> = links
        .into_iter()
        .filter_map(|l| l.target.into_action_hash())
        .collect();

    Ok(hashes)
}

/// Input for updating payment intent status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdatePaymentStatusInput {
    /// Intent ID
    pub intent_id: String,
    /// New status
    pub status: String,
    /// Transaction hash
    pub tx_hash: Option<String>,
    /// Block number
    pub block_number: Option<u64>,
    /// Error message
    pub error: Option<String>,
}

/// Update payment intent status (called by Host after Ethereum tx)
#[hdk_extern]
pub fn update_payment_intent_status(input: UpdatePaymentStatusInput) -> ExternResult<ActionHash> {
    // Find the intent by ID
    let intent_path = Path::from(format!("eth_intent.{}", input.intent_id));
    let intent_hash = ensure_path(intent_path, LinkTypes::IntentIdToPaymentIntent)?;

    let links = get_links(
        LinkQuery::new(
            intent_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::IntentIdToPaymentIntent as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut intent) = EthereumPaymentIntent::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        let now = sys_time()?.0 as u64 / 1_000_000;

                        // Update fields
                        intent.status = input.status.clone();
                        intent.tx_hash = input.tx_hash;
                        intent.block_number = input.block_number;
                        intent.error = input.error;
                        intent.updated_at = now;

                        let new_hash = update_entry(action_hash, &EntryTypes::EthereumPaymentIntent(intent))?;

                        // If status is no longer pending, remove from pending list
                        if input.status != "pending" {
                            // Note: In production, would delete the link
                            // For now, just broadcast the status update
                            let _ = broadcast_event(BroadcastEventInput {
                                event_type: "ethereum_payment_status".to_string(),
                                payload: serde_json::to_vec(&serde_json::json!({
                                    "intent_id": input.intent_id,
                                    "status": input.status,
                                })).unwrap_or_default(),
                                targets: vec![],
                                priority: 1,
                            });
                        }

                        return Ok(new_hash);
                    }
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        format!("Payment intent not found: {}", input.intent_id)
    )))
}

/// Input for updating anchor intent status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateAnchorStatusInput {
    /// Intent ID
    pub intent_id: String,
    /// New status
    pub status: String,
    /// Transaction hash
    pub tx_hash: Option<String>,
    /// Block number
    pub block_number: Option<u64>,
    /// Block hash
    pub block_hash: Option<String>,
    /// Commitment ID from contract
    pub commitment_id: Option<String>,
    /// Error message
    pub error: Option<String>,
}

/// Update anchor intent status (called by Host after Ethereum tx)
#[hdk_extern]
pub fn update_anchor_intent_status(input: UpdateAnchorStatusInput) -> ExternResult<ActionHash> {
    // Find the intent by ID
    let intent_path = Path::from(format!("eth_anchor.{}", input.intent_id));
    let intent_hash = ensure_path(intent_path, LinkTypes::IntentIdToAnchorIntent)?;

    let links = get_links(
        LinkQuery::new(
            intent_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::IntentIdToAnchorIntent as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(mut intent) = EthereumAnchorIntent::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        let now = sys_time()?.0 as u64 / 1_000_000;

                        // Update fields
                        intent.status = input.status.clone();
                        intent.tx_hash = input.tx_hash;
                        intent.block_number = input.block_number;
                        intent.block_hash = input.block_hash;
                        intent.commitment_id = input.commitment_id;
                        intent.error = input.error;
                        intent.updated_at = now;

                        let new_hash = update_entry(action_hash, &EntryTypes::EthereumAnchorIntent(intent))?;

                        // Broadcast status update
                        if input.status != "pending" {
                            let _ = broadcast_event(BroadcastEventInput {
                                event_type: "ethereum_anchor_status".to_string(),
                                payload: serde_json::to_vec(&serde_json::json!({
                                    "intent_id": input.intent_id,
                                    "status": input.status,
                                })).unwrap_or_default(),
                                targets: vec![],
                                priority: 0,
                            });
                        }

                        return Ok(new_hash);
                    }
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        format!("Anchor intent not found: {}", input.intent_id)
    )))
}

/// Get payment intent by ID
#[hdk_extern]
pub fn get_payment_intent(intent_id: String) -> ExternResult<Option<EthereumPaymentIntent>> {
    let intent_path = Path::from(format!("eth_intent.{}", intent_id));
    let intent_hash = match ensure_path(intent_path, LinkTypes::IntentIdToPaymentIntent) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            intent_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::IntentIdToPaymentIntent as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(intent) = EthereumPaymentIntent::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(intent));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get anchor intent by ID
#[hdk_extern]
pub fn get_anchor_intent(intent_id: String) -> ExternResult<Option<EthereumAnchorIntent>> {
    let intent_path = Path::from(format!("eth_anchor.{}", intent_id));
    let intent_hash = match ensure_path(intent_path, LinkTypes::IntentIdToAnchorIntent) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            intent_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::IntentIdToAnchorIntent as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(intent) = EthereumAnchorIntent::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(Some(intent));
                    }
                }
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_happ_reputation_calculation() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Weighted average: (0.9 * 100 + 0.8 * 50) / 150 = 0.867
        assert!((rep.aggregate - 0.867).abs() < 0.01);
        assert!(rep.is_trustworthy(0.5));
        assert!(!rep.is_trustworthy(0.9));
    }
}
