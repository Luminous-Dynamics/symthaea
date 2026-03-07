//! Personal Bridge Coordinator Zome
//!
//! Selective disclosure gateway for the Sovereign (Personal) cluster.
//! Provides three integration patterns (matching civic/commons bridges):
//!
//! 1. **dispatch_call** — synchronous RPC to any personal domain zome
//! 2. **query_personal** — audited async query/response
//! 3. **broadcast_event** — event distribution to connected clients
//!
//! Cross-cluster callers (Civic, Commons) reach this bridge via
//! `CallTargetCell::OtherRole("personal")`.

use personal_bridge_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    self as bridge,
    DispatchInput, DispatchResult, CrossClusterDispatchInput,
    ResolveQueryInput, EventTypeQuery, BridgeHealth,
    RATE_LIMIT_WINDOW_SECS, check_rate_limit_count,
};
use personal_types::{
    CredentialType, CredentialPresentation, DisclosureScope,
};
#[cfg(test)]
use personal_types::PresentationRequest;

// ============================================================================
// Allowed zome names — security boundary for dispatch
// ============================================================================

const ALLOWED_ZOMES: &[&str] = &[
    "identity_vault",
    "health_vault",
    "credential_wallet",
];

// Cross-cluster dispatch targets (what personal bridge can call in other clusters)
const ALLOWED_COMMONS_ZOMES: &[&str] = &[
    "commons_bridge",
];

const ALLOWED_CIVIC_ZOMES: &[&str] = &[
    "civic_bridge",
];

const ALLOWED_GOVERNANCE_ZOMES: &[&str] = &[
    "governance_bridge",
];

const ALLOWED_IDENTITY_ZOMES: &[&str] = &[
    "identity_bridge",
    "did_registry",
    "verifiable_credential",
];

// ============================================================================
// Helpers
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

// ============================================================================
// Rate Limiting
// ============================================================================

fn enforce_rate_limit(target_zome: &str) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let anchor = ensure_anchor("dispatch_rate_limit")?;

    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::DispatchRateLimit)?,
        GetStrategy::Local,
    )?;

    let now = sys_time()?;
    let window_start_micros = now.as_micros() - (RATE_LIMIT_WINDOW_SECS * 1_000_000);
    let window_start = Timestamp::from_micros(window_start_micros);

    let recent_count = links.iter()
        .filter(|l| l.timestamp >= window_start)
        .count();

    check_rate_limit_count(recent_count)
        .map_err(|msg| wasm_error!(WasmErrorInner::Guest(msg)))?;

    create_link(
        agent,
        anchor,
        LinkTypes::DispatchRateLimit,
        target_zome.as_bytes().to_vec(),
    )?;

    Ok(())
}

// ============================================================================
// Cross-Domain Dispatch (synchronous RPC)
// ============================================================================

/// Dispatch a synchronous call to any domain zome within the Personal DNA.
#[hdk_extern]
pub fn dispatch_call(input: DispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&input.zome)?;
    bridge::dispatch_call_checked(&input, ALLOWED_ZOMES)
}

// ============================================================================
// Audited Query/Response
// ============================================================================

/// Submit a cross-domain personal query.
#[hdk_extern]
pub fn query_personal(query: PersonalQueryEntry) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Query(query.clone()))?;

    let all_anchor = ensure_anchor("all_personal_queries")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllQueries, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_queries:{}", query.requester))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToQuery, ())?;

    let domain_anchor = ensure_anchor(&format!("domain_queries:{}", query.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToQuery, ())?;

    // Attempt auto-dispatch
    if let Some(zome_name) = resolve_domain_zome(&query.domain) {
        let payload_bytes = ExternIO::encode(query.params.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0;
        let dispatch = DispatchInput {
            zome: zome_name,
            fn_name: query.query_type.clone(),
            payload: payload_bytes,
        };
        if let Ok(result) = dispatch_call(dispatch) {
            if result.success {
                let result_str = result.response
                    .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                    .unwrap_or_else(|| "null".to_string());
                let _ = resolve_query(ResolveQueryInput {
                    query_hash: action_hash.clone(),
                    result: result_str,
                    success: true,
                });
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created query".into()
    )))
}

fn resolve_domain_zome(domain: &str) -> Option<String> {
    match domain {
        "identity" => Some("identity_vault".to_string()),
        "health" => Some("health_vault".to_string()),
        "credential" => Some("credential_wallet".to_string()),
        _ => None,
    }
}

// ============================================================================
// Query Resolution
// ============================================================================

#[hdk_extern]
pub fn resolve_query(input: ResolveQueryInput) -> ExternResult<Record> {
    let record = get(input.query_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Query not found".into())))?;

    let mut query: PersonalQueryEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid query entry".into())))?;

    let now = sys_time()?;
    query.result = Some(input.result);
    query.resolved_at = Some(now);
    query.success = Some(input.success);

    let updated_hash = update_entry(input.query_hash, &EntryTypes::Query(query))?;
    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated query".into()
    )))
}

// ============================================================================
// Event Broadcasting
// ============================================================================

/// Signal payload emitted to connected UI clients.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub signal_type: String,
    pub domain: String,
    pub event_type: String,
    pub payload: String,
    pub action_hash: ActionHash,
}

/// Broadcast a cross-domain event and emit a signal to connected clients.
#[hdk_extern]
pub fn broadcast_event(event: PersonalEventEntry) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Event(event.clone()))?;

    let all_anchor = ensure_anchor("all_personal_events")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllEvents, ())?;

    let type_anchor = ensure_anchor(&format!("event_type:{}:{}", event.domain, event.event_type))?;
    create_link(type_anchor, action_hash.clone(), LinkTypes::EventTypeToEvent, ())?;

    let agent_anchor = ensure_anchor(&format!("agent_events:{}", event.source_agent))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToEvent, ())?;

    let domain_anchor = ensure_anchor(&format!("domain_events:{}", event.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainToEvent, ())?;

    let signal = BridgeEventSignal {
        signal_type: "personal_bridge_event".to_string(),
        domain: event.domain.clone(),
        event_type: event.event_type.clone(),
        payload: event.payload.clone(),
        action_hash: action_hash.clone(),
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created event".into()
    )))
}

// ============================================================================
// Query helpers
// ============================================================================

#[hdk_extern]
pub fn get_domain_events(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = anchor_hash(&format!("domain_events:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

#[hdk_extern]
pub fn get_all_events(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_personal_events")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllEvents)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

#[hdk_extern]
pub fn get_events_by_type(query: EventTypeQuery) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("event_type:{}:{}", query.domain, query.event_type))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::EventTypeToEvent)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

#[hdk_extern]
pub fn get_my_queries(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let agent_anchor = anchor_hash(&format!("agent_queries:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToQuery)?,
        GetStrategy::default(),
    )?;
    bridge::records_from_links(links)
}

// ============================================================================
// Credential Presentation (cross-cluster API)
// ============================================================================

/// Present a Phi credential for governance voting.
///
/// Called by governance via `CallTargetCell::OtherRole("personal")`.
/// Returns a signed attestation of the agent's Phi score from FL participation.
#[hdk_extern]
pub fn present_phi_credential(_: ()) -> ExternResult<CredentialPresentation> {
    // Dispatch to credential_wallet to get FL credential
    let dispatch = DispatchInput {
        zome: "credential_wallet".into(),
        fn_name: "present_credential".into(),
        payload: ExternIO::encode(CredentialType::FederatedLearning)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    let now = sys_time()?;

    if result.success {
        let data = result.response
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|| "{}".to_string());
        Ok(CredentialPresentation {
            credential_type: CredentialType::FederatedLearning,
            disclosed_data: data,
            scope: DisclosureScope::Full,
            presented_at: now,
        })
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "No FL credential available for Phi attestation".into()
        )))
    }
}

/// Present an identity proof for cross-cluster verification.
///
/// Discloses only the requested fields from the identity vault.
#[hdk_extern]
pub fn present_identity_proof(fields: Vec<String>) -> ExternResult<CredentialPresentation> {
    let dispatch = DispatchInput {
        zome: "identity_vault".into(),
        fn_name: "disclose_profile".into(),
        payload: ExternIO::encode(fields.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    let now = sys_time()?;

    if result.success {
        let data = result.response
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|| "{}".to_string());
        Ok(CredentialPresentation {
            credential_type: CredentialType::Identity,
            disclosed_data: data,
            scope: DisclosureScope::SelectedFields(fields),
            presented_at: now,
        })
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "Failed to disclose identity fields".into()
        )))
    }
}

/// Present a K-vector credential for governance trust scoring.
#[hdk_extern]
pub fn present_k_vector(_: ()) -> ExternResult<CredentialPresentation> {
    let dispatch = DispatchInput {
        zome: "credential_wallet".into(),
        fn_name: "present_credential".into(),
        payload: ExternIO::encode(CredentialType::FederatedLearning)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .0,
    };

    let result = bridge::dispatch_call_checked(&dispatch, ALLOWED_ZOMES)?;
    let now = sys_time()?;

    if result.success {
        let data = result.response
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|| "{}".to_string());
        Ok(CredentialPresentation {
            credential_type: CredentialType::FederatedLearning,
            disclosed_data: data,
            scope: DisclosureScope::Full,
            presented_at: now,
        })
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "No K-vector credential available".into()
        )))
    }
}

// ============================================================================
// Phi Attestation Submission (Personal → Governance)
// ============================================================================

/// Input for submitting a Phi attestation to the governance cluster.
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitPhiAttestationInput {
    /// Phi value from Symthaea cognitive cycle, in [0.0, 1.0].
    pub phi: f64,
    /// Symthaea cognitive cycle number (must be > 0).
    pub cycle_id: u64,
}

/// Submit a signed Phi attestation to the governance bridge.
///
/// Signs the attestation with the agent's Holochain key and dispatches
/// it to the governance cluster's `record_phi_attestation` function.
/// The governance bridge verifies the Ed25519 signature before committing.
///
/// Message format (must match governance bridge verification):
///   `symthaea-phi-attestation:v1:{agent_did}:{phi:.6}:{cycle_id}:{captured_at_us}`
#[hdk_extern]
pub fn submit_phi_attestation(input: SubmitPhiAttestationInput) -> ExternResult<DispatchResult> {
    // Validate inputs
    if input.phi < 0.0 || input.phi > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Phi must be between 0.0 and 1.0".into()
        )));
    }
    if input.cycle_id == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cycle ID must be > 0".into()
        )));
    }

    let info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", info.agent_initial_pubkey);
    let captured_at_us = sys_time()?.as_micros() as u64;

    // Construct the message in the exact format that governance bridge verifies.
    let message = format!(
        "symthaea-phi-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, input.phi, input.cycle_id, captured_at_us,
    );

    // Sign with the agent's Holochain Ed25519 key.
    let signature = sign_raw(
        info.agent_initial_pubkey,
        message.into_bytes(),
    )?;

    // Build the payload matching governance bridge's RecordPhiAttestationInput.
    #[derive(Serialize, Debug)]
    struct RecordPhiAttestationInput {
        phi: f64,
        cycle_id: u64,
        captured_at_us: u64,
        signature: Vec<u8>,
    }

    let attestation_input = RecordPhiAttestationInput {
        phi: input.phi,
        cycle_id: input.cycle_id,
        captured_at_us,
        signature: signature.0.to_vec(),
    };

    let payload = ExternIO::encode(attestation_input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;

    // Dispatch to governance bridge via cross-cluster call.
    enforce_rate_limit("governance_bridge")?;
    let dispatch = CrossClusterDispatchInput {
        role: GOVERNANCE_ROLE.to_string(),
        zome: "governance_bridge".to_string(),
        fn_name: "record_phi_attestation".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_GOVERNANCE_ZOMES)
}

// ============================================================================
// Cross-Cluster Dispatch
// ============================================================================

const COMMONS_ROLE: &str = "commons";
const CIVIC_ROLE: &str = "civic";
const GOVERNANCE_ROLE: &str = "governance";
const IDENTITY_ROLE: &str = "identity";

/// Dispatch a call to the Commons cluster.
#[hdk_extern]
pub fn dispatch_commons_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("commons:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: COMMONS_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_COMMONS_ZOMES)
}

/// Dispatch a call to the Civic cluster.
#[hdk_extern]
pub fn dispatch_civic_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("civic:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: CIVIC_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_CIVIC_ZOMES)
}

/// Dispatch a call to the Governance cluster.
#[hdk_extern]
pub fn dispatch_governance_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("governance:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: GOVERNANCE_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_GOVERNANCE_ZOMES)
}

/// Dispatch a call to the Identity cluster.
#[hdk_extern]
pub fn dispatch_identity_call(input: CrossClusterDispatchInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit(&format!("identity:{}", input.zome))?;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: input.zome,
        fn_name: input.fn_name,
        payload: input.payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

// ============================================================================
// Identity Cluster Proxy — DID Registry (Task #88)
// ============================================================================

/// Resolve a DID document from the identity cluster's DID registry.
///
/// Proxies to `identity:did_registry:resolve_did`. Returns the DID document
/// record if found, or an error if the DID doesn't exist.
#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:did_registry")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "did_registry".to_string(),
        fn_name: "resolve_did".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Check whether a DID is currently active (not deactivated).
#[hdk_extern]
pub fn is_did_active(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:did_registry")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "did_registry".to_string(),
        fn_name: "is_did_active".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Query identity verification via the identity bridge.
///
/// Performs a full DID lookup including MATL score, credential count,
/// and deactivation status. The identity bridge audits this query.
#[hdk_extern]
pub fn query_identity(input: QueryIdentityInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:identity_bridge")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "query_identity".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Check MATL trust score for a DID.
#[hdk_extern]
pub fn get_matl_score(did: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:identity_bridge")?;
    let payload = ExternIO::encode(did)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "identity_bridge".to_string(),
        fn_name: "get_matl_score".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Input for querying identity verification from the identity bridge.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryIdentityInput {
    pub did: String,
    pub source_happ: String,
    pub requested_fields: Vec<String>,
}

// ============================================================================
// Identity Cluster Proxy — Verifiable Credentials (Task #89)
// ============================================================================

/// Verify a credential by ID via the identity cluster.
///
/// Proxies to `identity:verifiable_credential:verify_credential`.
/// Returns the full verification result including revocation status.
#[hdk_extern]
pub fn verify_credential(credential_id: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:verifiable_credential")?;
    let payload = ExternIO::encode(credential_id)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "verifiable_credential".to_string(),
        fn_name: "verify_credential".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Get a credential record by ID.
#[hdk_extern]
pub fn get_credential(credential_id: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:verifiable_credential")?;
    let payload = ExternIO::encode(credential_id)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "verifiable_credential".to_string(),
        fn_name: "get_credential".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Check whether a credential has been revoked.
#[hdk_extern]
pub fn is_credential_revoked(credential_id: String) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:verifiable_credential")?;
    let payload = ExternIO::encode(credential_id)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "verifiable_credential".to_string(),
        fn_name: "is_credential_revoked".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Get all credentials issued to the calling agent's DID.
#[hdk_extern]
pub fn get_my_credentials_from_identity(_: ()) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:verifiable_credential")?;
    let payload = ExternIO::encode(())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "verifiable_credential".to_string(),
        fn_name: "get_my_credentials".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Create a verifiable presentation from credentials held in the identity cluster.
#[hdk_extern]
pub fn create_presentation(input: CreatePresentationInput) -> ExternResult<DispatchResult> {
    enforce_rate_limit("identity:verifiable_credential")?;
    let payload = ExternIO::encode(&input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .0;
    let dispatch = CrossClusterDispatchInput {
        role: IDENTITY_ROLE.to_string(),
        zome: "verifiable_credential".to_string(),
        fn_name: "create_presentation".to_string(),
        payload,
    };
    bridge::dispatch_call_cross_cluster(&dispatch, ALLOWED_IDENTITY_ZOMES)
}

/// Input for creating a verifiable presentation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreatePresentationInput {
    pub credential_hashes: Vec<ActionHash>,
    pub challenge: Option<String>,
    pub domain: Option<String>,
    pub verifier_did: Option<String>,
}

// ============================================================================
// Health Check
// ============================================================================

#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<BridgeHealth> {
    let caller = agent_info()?.agent_initial_pubkey;
    let events = get_all_events(())?;
    let queries = get_my_queries(())?;

    Ok(BridgeHealth {
        healthy: true,
        agent: caller.to_string(),
        total_events: events.len() as u32,
        total_queries: queries.len() as u32,
        domains: vec![
            "identity".to_string(),
            "health".to_string(),
            "credential".to_string(),
        ],
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Allowlist validation ----

    #[test]
    fn local_allowlist_covers_all_domains() {
        assert!(ALLOWED_ZOMES.contains(&"identity_vault"));
        assert!(ALLOWED_ZOMES.contains(&"health_vault"));
        assert!(ALLOWED_ZOMES.contains(&"credential_wallet"));
    }

    #[test]
    fn local_allowlist_has_expected_count() {
        assert_eq!(ALLOWED_ZOMES.len(), 3);
    }

    #[test]
    fn allowed_zomes_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_ZOMES {
            assert!(seen.insert(zome), "Duplicate in ALLOWED_ZOMES: '{}'", zome);
        }
    }

    #[test]
    fn allowed_zomes_entries_are_non_empty() {
        for zome in ALLOWED_ZOMES {
            assert!(!zome.is_empty());
            assert!(!zome.contains(' '));
        }
    }

    // ---- Cross-cluster allowlists ----

    #[test]
    fn commons_allowlist_contains_bridge() {
        assert!(ALLOWED_COMMONS_ZOMES.contains(&"commons_bridge"));
    }

    #[test]
    fn civic_allowlist_contains_bridge() {
        assert!(ALLOWED_CIVIC_ZOMES.contains(&"civic_bridge"));
    }

    #[test]
    fn governance_allowlist_contains_bridge() {
        assert!(ALLOWED_GOVERNANCE_ZOMES.contains(&"governance_bridge"));
    }

    #[test]
    fn role_constants_correct() {
        assert_eq!(COMMONS_ROLE, "commons");
        assert_eq!(CIVIC_ROLE, "civic");
        assert_eq!(GOVERNANCE_ROLE, "governance");
    }

    // ---- Domain resolution ----

    #[test]
    fn resolve_identity_domain() {
        assert_eq!(resolve_domain_zome("identity").unwrap(), "identity_vault");
    }

    #[test]
    fn resolve_health_domain() {
        assert_eq!(resolve_domain_zome("health").unwrap(), "health_vault");
    }

    #[test]
    fn resolve_credential_domain() {
        assert_eq!(resolve_domain_zome("credential").unwrap(), "credential_wallet");
    }

    #[test]
    fn resolve_unknown_domain_returns_none() {
        assert!(resolve_domain_zome("justice").is_none());
        assert!(resolve_domain_zome("property").is_none());
        assert!(resolve_domain_zome("").is_none());
    }

    #[test]
    fn resolve_outputs_are_in_allowlist() {
        let domains = ["identity", "health", "credential"];
        for domain in &domains {
            let resolved = resolve_domain_zome(domain).unwrap();
            assert!(
                ALLOWED_ZOMES.contains(&resolved.as_str()),
                "resolve('{}') = '{}' not in ALLOWED_ZOMES",
                domain, resolved
            );
        }
    }

    // ---- Signal type ----

    #[test]
    fn bridge_event_signal_type_is_personal() {
        let signal = BridgeEventSignal {
            signal_type: "personal_bridge_event".to_string(),
            domain: "identity".to_string(),
            event_type: "profile_updated".to_string(),
            payload: "{}".to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(signal.signal_type, "personal_bridge_event");
    }

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            signal_type: "personal_bridge_event".to_string(),
            domain: "health".to_string(),
            event_type: "biometric_recorded".to_string(),
            payload: r#"{"type":"heart_rate"}"#.to_string(),
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.domain, "health");
        assert_eq!(back.event_type, "biometric_recorded");
    }

    // ---- Rate limit ----

    #[test]
    fn rate_limit_under_threshold_passes() {
        assert!(check_rate_limit_count(0).is_ok());
        assert!(check_rate_limit_count(99).is_ok());
    }

    #[test]
    fn rate_limit_at_threshold_rejects() {
        let err = check_rate_limit_count(100).unwrap_err();
        assert!(err.contains("Rate limit exceeded"));
    }

    // ---- Presentation types serde ----

    #[test]
    fn presentation_request_serde_roundtrip() {
        let req = PresentationRequest {
            credential_type: CredentialType::Governance,
            scope: DisclosureScope::ExistenceOnly,
            context: Some("vote:prop_7".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: PresentationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_type, CredentialType::Governance);
    }

    #[test]
    fn credential_presentation_serde_roundtrip() {
        let pres = CredentialPresentation {
            credential_type: CredentialType::FederatedLearning,
            disclosed_data: r#"{"phi":0.42}"#.into(),
            scope: DisclosureScope::Full,
            presented_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&pres).unwrap();
        let back: CredentialPresentation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_type, CredentialType::FederatedLearning);
        assert!(back.disclosed_data.contains("0.42"));
    }

    // ---- Health check domain list ----

    #[test]
    fn health_check_domains_match_resolution() {
        let domains = ["identity", "health", "credential"];
        for domain in &domains {
            assert!(resolve_domain_zome(domain).is_some());
        }
    }

    // ---- Phi attestation ----

    #[test]
    fn submit_phi_attestation_input_serde_roundtrip() {
        let input = SubmitPhiAttestationInput {
            phi: 0.72,
            cycle_id: 42,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SubmitPhiAttestationInput = serde_json::from_str(&json).unwrap();
        assert!((back.phi - 0.72).abs() < f64::EPSILON);
        assert_eq!(back.cycle_id, 42);
    }

    #[test]
    fn attestation_message_format_matches_governance() {
        // Verify our message format exactly matches what governance bridge expects.
        let agent_did = "did:mycelix:uhCAktest123";
        let phi: f64 = 0.654321;
        let cycle_id: u64 = 100;
        let captured_at_us: u64 = 1_700_000_000_000_000;

        let message = format!(
            "symthaea-phi-attestation:v1:{}:{:.6}:{}:{}",
            agent_did, phi, cycle_id, captured_at_us,
        );

        assert!(message.starts_with("symthaea-phi-attestation:v1:"));
        assert!(message.contains("did:mycelix:"));
        assert!(message.contains("0.654321"));
        assert!(message.contains(":100:"));
        assert!(message.ends_with("1700000000000000"));
    }

    #[test]
    fn attestation_message_phi_precision() {
        // Governance bridge uses {:.6} format — verify precision.
        let message = format!(
            "symthaea-phi-attestation:v1:did:mycelix:test:{:.6}:1:0",
            0.1_f64,
        );
        assert!(message.contains("0.100000"));
    }

    #[test]
    fn governance_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_GOVERNANCE_ZOMES {
            assert!(seen.insert(zome), "Duplicate in ALLOWED_GOVERNANCE_ZOMES: '{}'", zome);
        }
    }

    // ---- Identity cluster allowlist ----

    #[test]
    fn identity_allowlist_contains_bridge() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"identity_bridge"));
    }

    #[test]
    fn identity_allowlist_contains_did_registry() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"did_registry"));
    }

    #[test]
    fn identity_allowlist_contains_verifiable_credential() {
        assert!(ALLOWED_IDENTITY_ZOMES.contains(&"verifiable_credential"));
    }

    #[test]
    fn identity_allowlist_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for zome in ALLOWED_IDENTITY_ZOMES {
            assert!(seen.insert(zome), "Duplicate in ALLOWED_IDENTITY_ZOMES: '{}'", zome);
        }
    }

    #[test]
    fn identity_role_constant_correct() {
        assert_eq!(IDENTITY_ROLE, "identity");
    }

    // ---- QueryIdentityInput serde ----

    #[test]
    fn query_identity_input_serde_roundtrip() {
        let input = QueryIdentityInput {
            did: "did:mycelix:uhCAktest".into(),
            source_happ: "personal".into(),
            requested_fields: vec!["matl_score".into(), "credential_count".into()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: QueryIdentityInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.did, "did:mycelix:uhCAktest");
        assert_eq!(back.requested_fields.len(), 2);
    }

    // ---- CreatePresentationInput serde ----

    #[test]
    fn create_presentation_input_serde_roundtrip() {
        let input = CreatePresentationInput {
            credential_hashes: vec![ActionHash::from_raw_36(vec![0u8; 36])],
            challenge: Some("nonce-123".into()),
            domain: Some("voting.mycelix.net".into()),
            verifier_did: Some("did:mycelix:verifier".into()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreatePresentationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.challenge, Some("nonce-123".into()));
        assert_eq!(back.credential_hashes.len(), 1);
    }

    #[test]
    fn create_presentation_input_minimal() {
        let input = CreatePresentationInput {
            credential_hashes: vec![],
            challenge: None,
            domain: None,
            verifier_did: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreatePresentationInput = serde_json::from_str(&json).unwrap();
        assert!(back.credential_hashes.is_empty());
        assert!(back.challenge.is_none());
    }
}
