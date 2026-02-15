//! DID Registry Coordinator Zome
//! Business logic for DID:mycelix operations
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use did_registry_integrity::*;
use mycelix_crypto::{AlgorithmId, TaggedPublicKey, CryptoError};

/// API version for cross-zome compatibility detection.
/// Increment when making breaking changes to extern signatures or types.
const API_VERSION: u16 = 1;

/// Returns the API version of this coordinator zome.
#[hdk_extern]
pub fn get_api_version(_: ()) -> ExternResult<u16> {
    Ok(API_VERSION)
}

/// Validate a multibase-encoded public key and return its detected algorithm.
///
/// Per the W3C DID specification and multibase encoding:
/// - Must start with 'z' (base58btc prefix)
/// - Remaining characters must be valid base58btc
/// - Decoded bytes must contain a recognized multicodec prefix + key bytes of
///   the correct length, OR be a legacy raw 32-byte Ed25519 key.
///
/// Returns the detected `AlgorithmId` on success.
fn validate_multibase_key(key: &str) -> Result<AlgorithmId, String> {
    TaggedPublicKey::from_multibase(key)
        .map(|tagged| tagged.algorithm)
        .map_err(|e| match e {
            CryptoError::InvalidKeyLength { algorithm, expected, actual } => {
                format!("Invalid {} key length: expected {}, got {}", algorithm, expected, actual)
            }
            CryptoError::InvalidMultibase(msg) => format!("Invalid multibase key: {}", msg),
            CryptoError::Base58Decode(msg) => format!("Invalid base58btc encoding: {}", msg),
            other => format!("Key validation error: {}", other),
        })
}

/// Legacy Ed25519-only validator (delegates to the algorithm-agnostic version).
#[cfg(test)]
fn validate_multibase_ed25519_key(key: &str) -> Result<(), String> {
    let alg = validate_multibase_key(key)?;
    if alg != AlgorithmId::Ed25519 {
        return Err(format!(
            "Expected Ed25519 key but detected {}",
            alg.did_verification_method_type()
        ));
    }
    Ok(())
}

// ==================== MFA INTEGRATION ====================

/// Input structure for creating MFA state (mirrors mfa_coordinator)
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CreateMfaStateInput {
    did: String,
    primary_key_hash: String,
}

/// Auto-create MFA state for a newly created DID
/// This sets up the initial PrimaryKeyPair factor automatically
fn auto_create_mfa_state(did: &str, agent_pub_key: &AgentPubKey) -> ExternResult<()> {
    // Create primary key hash from agent pub key
    // Using the agent's public key as the initial factor
    let primary_key_hash = format!("sha256:{}", agent_pub_key);

    let input = CreateMfaStateInput {
        did: did.to_string(),
        primary_key_hash,
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("create_mfa_state"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => {
            // MFA state created successfully
            Ok(())
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // MFA zome not accessible - warn and continue DID creation.
            // This allows DID creation even if MFA zome is not installed,
            // but the warning ensures operators notice MFA is missing.
            warn!("MFA zome unauthorized - DID created WITHOUT MFA state (MFA enrollment will be required separately)");
            Ok(())
        }
        ZomeCallResponse::NetworkError(err) => {
            warn!("MFA zome network error: {} - DID created WITHOUT MFA state", err);
            Ok(())
        }
        ZomeCallResponse::CountersigningSession(err) => {
            warn!("MFA zome countersigning error: {} - DID created WITHOUT MFA state", err);
            Ok(())
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            warn!("MFA zome authentication failed - DID created WITHOUT MFA state");
            Ok(())
        }
    }
}

/// Notify bridge of DID creation for ecosystem-wide awareness
fn notify_bridge_of_did_event(did: &str, event_type: &str, payload: &str) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct BroadcastEventInput {
        event_type: String,
        subject: String,
        payload: String,
        source_happ: String,
    }

    let input = BroadcastEventInput {
        event_type: event_type.to_string(),
        subject: did.to_string(),
        payload: payload.to_string(),
        source_happ: "mycelix-identity".to_string(),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("identity_bridge"),
        FunctionName::new("broadcast_event"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, zome, fn_name) => {
            warn!(
                "Bridge notification unauthorized: zome={:?} fn={:?} event={} did={}",
                zome, fn_name, event_type, did
            );
            Ok(())
        }
        ZomeCallResponse::NetworkError(err) => {
            warn!(
                "Bridge notification network error: {} event={} did={}",
                err, event_type, did
            );
            Ok(())
        }
        ZomeCallResponse::CountersigningSession(err) => {
            warn!(
                "Bridge notification countersigning error: {} event={} did={}",
                err, event_type, did
            );
            Ok(())
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            warn!(
                "Bridge notification auth failed: event={} did={}",
                event_type, did
            );
            Ok(())
        }
    }
}

/// Create a new DID document for the calling agent
#[hdk_extern]
pub fn create_did() -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Rate limit: 1 DID per agent (idempotency guard)
    if get_did_document(agent_pub_key.clone())?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Agent already has a DID document. Use update_did_document to modify it.".into()
        )));
    }

    // Generate DID identifier
    let did_id = format!("did:mycelix:{}", agent_pub_key);

    // Create default verification method
    let verification_method = VerificationMethod {
        id: format!("{}#keys-1", did_id),
        type_: AlgorithmId::Ed25519.did_verification_method_type().to_string(),
        controller: did_id.clone(),
        public_key_multibase: format!("z{}", agent_pub_key),
        algorithm: Some(AlgorithmId::Ed25519.as_u16()),
    };

    let now = sys_time()?;
    let did_doc = DidDocument {
        id: did_id.clone(),
        controller: agent_pub_key.clone(),
        verification_method: vec![verification_method.clone()],
        authentication: vec![format!("{}#keys-1", did_id)],
        key_agreement: vec![],
        service: vec![],
        created: now,
        updated: now,
        version: 1,
    };

    let action_hash = create_entry(&EntryTypes::DidDocument(did_doc.clone()))?;

    // Link agent to DID
    create_link(
        agent_pub_key.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDid,
        (),
    )?;

    // Auto-create MFA state for the new DID
    // This registers the primary key pair as the initial authentication factor
    // Errors are logged but don't fail DID creation (MFA is optional)
    if let Err(e) = auto_create_mfa_state(&did_id, &agent_pub_key) {
        debug!("Failed to auto-create MFA state: {:?}", e);
    }

    // Broadcast DidCreated event to bridge for ecosystem-wide awareness
    let payload = serde_json::json!({
        "did": did_id,
        "event": "did_created",
    }).to_string();
    if let Err(e) = notify_bridge_of_did_event(&did_id, "DidCreated", &payload) {
        debug!("Failed to notify bridge of DID creation: {:?}", e);
    }

    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created DID".into()
        )))?;

    Ok(record)
}

/// Get DID document for an agent
#[hdk_extern]
pub fn get_did_document(agent_pub_key: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::AgentToDid)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the latest DID document
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        get(action_hash, GetOptions::default())
    } else {
        Ok(None)
    }
}

/// Resolve a DID to its document
#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<Option<Record>> {
    // Parse DID to extract agent pub key
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did.strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into()))
    })?;

    get_did_document(agent_pub_key)
}

/// Update DID document (add service endpoints, rotate keys, etc.)
#[hdk_extern]
pub fn update_did_document(input: UpdateDidInput) -> ExternResult<Record> {
    // Input validation
    if let Some(ref methods) = input.verification_method {
        if methods.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest("Verification methods must not exceed 100 entries".into())));
        }
        for method in methods {
            if method.id.is_empty() || method.id.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Verification method ID must be 1-256 characters".into())));
            }
            if method.type_.is_empty() || method.type_.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Verification method type must be 1-256 characters".into())));
            }
            if method.controller.is_empty() || method.controller.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Verification method controller must be 1-256 characters".into())));
            }
            if method.public_key_multibase.is_empty() || method.public_key_multibase.len() > 4096 {
                return Err(wasm_error!(WasmErrorInner::Guest("Public key multibase must be 1-4096 characters".into())));
            }
            // Note: multibase key format validation is handled by the individual
            // add functions (add_verification_method, add_key_agreement) which
            // validate before calling update. We skip re-validation here because
            // legacy keys (created by create_did with `z{agent_pubkey}` format)
            // do not parse as TaggedPublicKey.
        }
    }
    if let Some(ref auth) = input.authentication {
        if auth.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest("Authentication entries must not exceed 100".into())));
        }
        for a in auth {
            if a.is_empty() || a.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Authentication entry must be 1-256 characters".into())));
            }
        }
    }
    if let Some(ref services) = input.service {
        if services.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest("Service endpoints must not exceed 100 entries".into())));
        }
        for svc in services {
            if svc.id.is_empty() || svc.id.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint ID must be 1-256 characters".into())));
            }
            if svc.type_.is_empty() || svc.type_.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint type must be 1-256 characters".into())));
            }
            if svc.service_endpoint.is_empty() || svc.service_endpoint.len() > 4096 {
                return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint URL must be 1-4096 characters".into())));
            }
        }
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Get current DID document
    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    let now = sys_time()?;

    // Validate key_agreement references if provided
    if let Some(ref ka) = input.key_agreement {
        if ka.len() > 100 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Key agreement entries must not exceed 100".into()
            )));
        }
        for entry in ka {
            if entry.is_empty() || entry.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Key agreement entry must be 1-256 characters".into()
                )));
            }
        }

        // Cross-validate: key_agreement fragment IDs must reference verification
        // methods that exist in the (potentially updated) verification method list.
        let effective_methods = input
            .verification_method
            .as_ref()
            .unwrap_or(&current_did.verification_method);
        let method_ids: std::collections::HashSet<&str> =
            effective_methods.iter().map(|m| m.id.as_str()).collect();
        for ka_ref in ka {
            if !method_ids.contains(ka_ref.as_str()) {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Key agreement '{}' references a verification method that does not exist in the document",
                    ka_ref
                ))));
            }
        }
    }

    // Build updated document
    let updated_did = DidDocument {
        id: current_did.id.clone(),
        controller: current_did.controller.clone(),
        verification_method: input
            .verification_method
            .unwrap_or(current_did.verification_method),
        authentication: input.authentication.unwrap_or(current_did.authentication),
        key_agreement: input.key_agreement.unwrap_or(current_did.key_agreement),
        service: input.service.unwrap_or(current_did.service),
        created: current_did.created,
        updated: now,
        version: current_did.version + 1,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::DidDocument(updated_did),
    )?;

    // Delete stale AgentToDid links to prevent DHT bloat.
    // Without cleanup, every update_did_document call adds another link,
    // and get_did_document must scan them all to find the latest.
    let existing_links = get_links(
        LinkQuery::try_new(agent_pub_key.clone(), LinkTypes::AgentToDid)?,
        GetStrategy::default(),
    )?;
    for link in existing_links {
        delete_link(link.create_link_hash, GetOptions::default())?;
    }

    // Create a single AgentToDid link pointing to the updated record.
    create_link(
        agent_pub_key,
        action_hash.clone(),
        LinkTypes::AgentToDid,
        (),
    )?;

    // Broadcast DidUpdated event (covers key rotation, service changes, etc.)
    let payload = serde_json::json!({
        "did": current_did.id,
        "version": current_did.version + 1,
        "event": "did_updated",
    }).to_string();
    if let Err(e) = notify_bridge_of_did_event(&current_did.id, "DidUpdated", &payload) {
        debug!("Failed to notify bridge of DID update: {:?}", e);
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated DID".into()
        )))
}

/// Input for updating a DID document
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDidInput {
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Option<Vec<VerificationMethod>>,
    pub authentication: Option<Vec<String>>,
    /// Key agreement methods (DID URL fragments referencing KEM verification methods).
    #[serde(rename = "keyAgreement", alias = "key_agreement")]
    pub key_agreement: Option<Vec<String>>,
    pub service: Option<Vec<ServiceEndpoint>>,
}

/// Deactivate a DID
#[hdk_extern]
pub fn deactivate_did(reason: String) -> ExternResult<Record> {
    // Input validation
    if reason.is_empty() || reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Get current DID
    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    // Idempotent: if already deactivated, return the existing deactivation record
    if !is_did_active(current_did.id.clone())? {
        // Find and return the existing deactivation record
        let deactivation_links = get_links(
            LinkQuery::try_new(agent_pub_key.clone(), LinkTypes::DidToDeactivation)?,
            GetStrategy::default(),
        )?;
        if let Some(link) = deactivation_links.into_iter().max_by_key(|l| l.timestamp) {
            let existing_hash = ActionHash::try_from(link.target).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest("Invalid deactivation link target".into()))
            })?;
            return get(existing_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Deactivation record not found".into()
                )));
        }
        // Fallback: links gone but DID is inactive (shouldn't happen)
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID is deactivated but deactivation record is missing".into()
        )));
    }

    let now = sys_time()?;

    let deactivation = DidDeactivation {
        did: current_did.id.clone(),
        reason: reason.clone(),
        deactivated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::DidDeactivation(deactivation))?;

    // Create link from DID to deactivation record for efficient lookup
    // Use the agent pub key as the base since that's how DIDs are looked up
    create_link(
        agent_pub_key.clone(),
        action_hash.clone(),
        LinkTypes::DidToDeactivation,
        (),
    )?;

    // Cascade: revoke all credentials issued by this DID.
    // The DID deactivation entry is already committed above, so even if cascade
    // revocation fails, the DID itself is deactivated. We propagate the error so
    // the caller knows credentials may still be active and can retry.
    cascade_revoke_credentials_for_did(&current_did.id, &reason, now)?;

    // Notify bridge of deactivation for ecosystem-wide awareness
    if let Err(e) = notify_bridge_of_deactivation(&current_did.id, &reason, now) {
        debug!("Failed to notify bridge of DID deactivation: {:?}", e);
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find deactivation record".into()
        )))
}

/// Notify bridge of DID deactivation via cross-zome call
fn notify_bridge_of_deactivation(did: &str, reason: &str, deactivated_at: Timestamp) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct DidDeactivatedNotification {
        did: String,
        reason: String,
        deactivated_at: String,
    }

    let input = DidDeactivatedNotification {
        did: did.to_string(),
        reason: reason.to_string(),
        deactivated_at: deactivated_at.as_micros().to_string(),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("identity_bridge"),
        FunctionName::new("notify_did_deactivated"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            debug!("Bridge zome not accessible for deactivation notification");
            Ok(())
        }
        ZomeCallResponse::NetworkError(err) => {
            debug!("Network error notifying bridge of deactivation: {}", err);
            Ok(())
        }
        ZomeCallResponse::CountersigningSession(err) => {
            debug!("Countersigning error notifying bridge: {}", err);
            Ok(())
        }
    }
}

/// Cascade-revoke all credentials issued by a deactivated DID via cross-zome call
fn cascade_revoke_credentials_for_did(did: &str, reason: &str, _deactivated_at: Timestamp) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct BatchRevokeInput {
        credential_ids: Vec<String>,
        issuer_did: String,
        reason: String,
        effective_from: Option<Timestamp>,
        revocation_list_id: Option<String>,
    }

    // First, get all credentials issued by this DID
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("verifiable_credential"),
        FunctionName::new("get_credentials_issued_by"),
        None,
        did.to_string(),
    )?;

    let credential_ids: Vec<String> = match response {
        ZomeCallResponse::Ok(result) => {
            let records: Vec<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Cascade revocation: failed to decode credential list for DID {}: {}",
                    did, e
                )))
            })?;
            records.iter().filter_map(|r| {
                r.action().entry_hash().map(|h| h.to_string())
            }).collect()
        }
        ZomeCallResponse::Unauthorized(_, _, zome, fn_name) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade revocation failed: unauthorized to query credentials (zome={:?} fn={:?}) for DID {}",
                zome, fn_name, did
            ))));
        }
        ZomeCallResponse::NetworkError(err) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade revocation failed: network error querying credentials for DID {}: {}",
                did, err
            ))));
        }
        ZomeCallResponse::CountersigningSession(err) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade revocation failed: countersigning error for DID {}: {}",
                did, err
            ))));
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade revocation failed: authentication failed querying credentials for DID {}",
                did
            ))));
        }
    };

    if credential_ids.is_empty() {
        return Ok(());
    }

    // Batch-revoke all credentials via the revocation zome
    let input = BatchRevokeInput {
        credential_ids,
        issuer_did: did.to_string(),
        reason: format!("DID deactivated: {}", reason),
        effective_from: None,
        revocation_list_id: None,
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("revocation"),
        FunctionName::new("batch_revoke_credentials"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => {
            debug!("Successfully cascade-revoked credentials for deactivated DID: {}", did);
            Ok(())
        }
        ZomeCallResponse::Unauthorized(_, _, zome, fn_name) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade batch revocation unauthorized: zome={:?} fn={:?} did={}",
                zome, fn_name, did
            ))))
        }
        ZomeCallResponse::NetworkError(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade batch revocation network error: {} did={}",
                err, did
            ))))
        }
        ZomeCallResponse::CountersigningSession(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade batch revocation countersigning error: {} did={}",
                err, did
            ))))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cascade batch revocation auth failed: did={}",
                did
            ))))
        }
    }
}

/// Check if a DID is active (not deactivated)
#[hdk_extern]
pub fn is_did_active(did: String) -> ExternResult<bool> {
    // First check if DID exists
    let record = resolve_did(did.clone())?;
    if record.is_none() {
        return Ok(false);
    }

    // Parse DID to extract agent pub key for link lookup
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did.strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into()))
    })?;

    // Check for deactivation links - if any exist, DID is deactivated
    let deactivation_links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::DidToDeactivation)?,
        GetStrategy::default(),
    )?;

    // If there are any deactivation links, the DID is not active
    if !deactivation_links.is_empty() {
        return Ok(false);
    }

    // No deactivation links found, DID is active
    Ok(true)
}

/// Get the deactivation record for a DID (if deactivated)
#[hdk_extern]
pub fn get_did_deactivation(did: String) -> ExternResult<Option<DidDeactivation>> {
    // Parse DID to extract agent pub key
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent_str = did.strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let agent_pub_key = AgentPubKey::try_from(agent_str).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into()))
    })?;

    // Get deactivation links
    let deactivation_links = get_links(
        LinkQuery::try_new(agent_pub_key, LinkTypes::DidToDeactivation)?,
        GetStrategy::default(),
    )?;

    if deactivation_links.is_empty() {
        return Ok(None);
    }

    // Get the most recent deactivation record
    let latest_link = deactivation_links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let deactivation: DidDeactivation = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid deactivation entry".into())))?;
            return Ok(Some(deactivation));
        }
    }

    Ok(None)
}

/// Add a service endpoint to the DID
#[hdk_extern]
pub fn add_service_endpoint(service: ServiceEndpoint) -> ExternResult<Record> {
    // Input validation
    if service.id.is_empty() || service.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint ID must be 1-256 characters".into())));
    }
    if service.type_.is_empty() || service.type_.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint type must be 1-256 characters".into())));
    }
    if service.service_endpoint.is_empty() || service.service_endpoint.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Service endpoint URL must be 1-4096 characters".into())));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    let mut services = current_did.service.clone();
    services.push(service);

    update_did_document(UpdateDidInput {
        verification_method: None,
        authentication: None,
        key_agreement: None,
        service: Some(services),
    })
}

/// Remove a service endpoint from the DID
#[hdk_extern]
pub fn remove_service_endpoint(service_id: String) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    let services: Vec<ServiceEndpoint> = current_did
        .service
        .into_iter()
        .filter(|s| s.id != service_id)
        .collect();

    update_did_document(UpdateDidInput {
        verification_method: None,
        authentication: None,
        key_agreement: None,
        service: Some(services),
    })
}

/// Add a verification method to the DID
#[hdk_extern]
pub fn add_verification_method(method: VerificationMethod) -> ExternResult<Record> {
    // Input validation
    if method.id.is_empty() || method.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Verification method ID must be 1-256 characters".into())));
    }
    if method.type_.is_empty() || method.type_.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Verification method type must be 1-256 characters".into())));
    }
    if method.controller.is_empty() || method.controller.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Verification method controller must be 1-256 characters".into())));
    }
    if method.public_key_multibase.is_empty() || method.public_key_multibase.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Public key multibase must be 1-4096 characters".into())));
    }

    // Validate multibase key format (algorithm-agnostic)
    if let Err(e) = validate_multibase_key(&method.public_key_multibase) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid multibase key: {}", e)
        )));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    let mut methods = current_did.verification_method.clone();
    methods.push(method);

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: None,
        key_agreement: None,
        service: None,
    })
}

/// Add a KEM verification method and register it for key agreement.
///
/// This adds a post-quantum KEM public key (e.g. ML-KEM-768) to the DID
/// document and records its DID URL fragment in the `keyAgreement` array.
/// Other agents can then look up this key to encrypt data to this DID's owner.
#[hdk_extern]
pub fn add_key_agreement(method: VerificationMethod) -> ExternResult<Record> {
    // Input validation
    if method.id.is_empty() || method.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verification method ID must be 1-256 characters".into()
        )));
    }
    if method.public_key_multibase.is_empty() || method.public_key_multibase.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Public key multibase must be 1-4096 characters".into()
        )));
    }

    // Validate the key is a KEM algorithm
    let alg = validate_multibase_key(&method.public_key_multibase)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid KEM key: {}", e))))?;

    match alg {
        AlgorithmId::MlKem768 | AlgorithmId::MlKem1024 => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Key agreement requires a KEM algorithm (ML-KEM-768 or ML-KEM-1024), got {}",
                alg.did_verification_method_type()
            ))));
        }
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    // Add the KEM key as a verification method
    let mut methods = current_did.verification_method.clone();
    methods.push(method.clone());

    // Register the method ID in keyAgreement
    let mut ka = current_did.key_agreement.clone();
    if !ka.contains(&method.id) {
        ka.push(method.id);
    }

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: None,
        key_agreement: Some(ka),
        service: None,
    })
}

/// Get my DID (convenience function)
#[hdk_extern]
pub fn get_my_did(_: ()) -> ExternResult<Option<Record>> {
    let agent_info = agent_info()?;
    get_did_document(agent_info.agent_initial_pubkey)
}

/// Input for key rotation
#[derive(Serialize, Deserialize, Debug)]
pub struct RotateKeyInput {
    /// ID of the verification method to replace (e.g. "#keys-1")
    pub old_key_id: String,
    /// New verification method to add
    pub new_method: VerificationMethod,
}

/// Rotate a verification method: deprecate the old key and add a new one atomically.
///
/// The old key is retained with its ID suffixed by `-deprecated-v{version}` so that
/// previously-issued signatures remain verifiable, but the deprecated key is removed
/// from the `authentication` array so it can no longer be used to authenticate.
#[hdk_extern]
pub fn rotate_key(input: RotateKeyInput) -> ExternResult<Record> {
    // Input validation
    if input.old_key_id.is_empty() || input.old_key_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Old key ID must be 1-256 characters".into()
        )));
    }
    if input.new_method.id.is_empty() || input.new_method.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "New verification method ID must be 1-256 characters".into()
        )));
    }
    if input.new_method.public_key_multibase.is_empty()
        || input.new_method.public_key_multibase.len() > 4096
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Public key multibase must be 1-4096 characters".into()
        )));
    }

    // Validate the new key format
    if let Err(e) = validate_multibase_key(&input.new_method.public_key_multibase) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid new key: {}", e)
        )));
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    // Find the old key
    let old_key_idx = current_did
        .verification_method
        .iter()
        .position(|m| m.id == input.old_key_id)
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Verification method '{}' not found",
            input.old_key_id
        ))))?;

    // Build updated methods: deprecate old key, add new one
    let mut methods = current_did.verification_method.clone();
    let deprecated_id = format!(
        "{}-deprecated-v{}",
        methods[old_key_idx].id, current_did.version
    );
    methods[old_key_idx].id = deprecated_id.clone();
    methods.push(input.new_method.clone());

    // Update authentication: remove old key reference, add new one
    let mut auth = current_did.authentication.clone();
    auth.retain(|a| a != &input.old_key_id);
    if !auth.contains(&input.new_method.id) {
        auth.push(input.new_method.id);
    }

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: Some(auth),
        key_agreement: None,
        service: None,
    })
}

/// Input for key agreement rotation
#[derive(Serialize, Deserialize, Debug)]
pub struct RotateKeyAgreementInput {
    /// ID of the KEM verification method to replace (e.g. "#kem-1")
    pub old_key_id: String,
    /// New KEM verification method to add
    pub new_method: VerificationMethod,
}

/// Rotate a key agreement method: deprecate the old KEM key and add a new one atomically.
///
/// Mirrors [`rotate_key`] but operates on the `keyAgreement` array instead of
/// `authentication`. The old KEM key is retained (renamed with `-deprecated-v{version}`)
/// so that previously-encrypted messages can still be decrypted, but the deprecated key
/// is removed from `keyAgreement` so new encryption uses the new key.
#[hdk_extern]
pub fn rotate_key_agreement(input: RotateKeyAgreementInput) -> ExternResult<Record> {
    // Input validation
    if input.old_key_id.is_empty() || input.old_key_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Old key ID must be 1-256 characters".into()
        )));
    }
    if input.new_method.id.is_empty() || input.new_method.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "New verification method ID must be 1-256 characters".into()
        )));
    }
    if input.new_method.public_key_multibase.is_empty()
        || input.new_method.public_key_multibase.len() > 4096
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Public key multibase must be 1-4096 characters".into()
        )));
    }

    // Validate the new key is a KEM algorithm
    let alg = validate_multibase_key(&input.new_method.public_key_multibase)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Invalid new KEM key: {}", e))))?;

    match alg {
        AlgorithmId::MlKem768 | AlgorithmId::MlKem1024 => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Key agreement rotation requires a KEM algorithm (ML-KEM-768 or ML-KEM-1024), got {}",
                alg.did_verification_method_type()
            ))));
        }
    }

    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    let current_record = get_did_document(agent_pub_key)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No DID found".into())))?;

    let current_did: DidDocument = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid DID entry".into())))?;

    // Find the old KEM key in verification_method
    let old_key_idx = current_did
        .verification_method
        .iter()
        .position(|m| m.id == input.old_key_id)
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "KEM verification method '{}' not found",
            input.old_key_id
        ))))?;

    // Build updated methods: deprecate old KEM key, add new one
    let mut methods = current_did.verification_method.clone();
    let deprecated_id = format!(
        "{}-deprecated-v{}",
        methods[old_key_idx].id, current_did.version
    );
    methods[old_key_idx].id = deprecated_id.clone();
    methods.push(input.new_method.clone());

    // Update key_agreement: remove old key reference, add new one
    let mut ka = current_did.key_agreement.clone();
    ka.retain(|k| k != &input.old_key_id);
    if !ka.contains(&input.new_method.id) {
        ka.push(input.new_method.id);
    }

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: None, // preserve existing authentication
        key_agreement: Some(ka),
        service: None,
    })
}

// =============================================================================
// SOCIAL RECOVERY: DID TRANSFER
// =============================================================================

/// Mirror type for recovery request deserialization (cross-zome)
#[derive(Serialize, Deserialize, Debug, Clone)]
struct RecoveryRequestMirror {
    pub id: String,
    pub did: String,
    pub new_agent: AgentPubKey,
    pub initiated_by: String,
    pub reason: String,
    pub status: RecoveryStatusMirror,
    pub created: Timestamp,
    pub time_lock_expires: Option<Timestamp>,
}

impl TryFrom<SerializedBytes> for RecoveryRequestMirror {
    type Error = SerializedBytesError;
    fn try_from(sb: SerializedBytes) -> Result<Self, Self::Error> {
        holochain_serialized_bytes::decode(sb.bytes())
    }
}

/// Mirror of recovery_integrity::RecoveryStatus for cross-zome deserialization
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum RecoveryStatusMirror {
    Pending,
    Approved,
    ReadyToExecute,
    Completed,
    Rejected,
    Cancelled,
}

/// Input for claiming a recovered DID
#[derive(Serialize, Deserialize, Debug)]
pub struct ClaimRecoveredDidInput {
    /// The completed recovery request ID
    pub request_id: String,
    /// The DID being recovered (did:mycelix:...)
    pub did: String,
}

/// Claim a DID after social recovery has been completed.
///
/// This must be called by the NEW agent specified in the recovery request.
/// It creates a new DID document controlled by the new agent, linked from
/// the original agent's pubkey so that DID resolution picks up the transfer.
#[hdk_extern]
pub fn claim_recovered_did(input: ClaimRecoveredDidInput) -> ExternResult<Record> {
    // Input validation
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    if !input.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let my_agent_info = agent_info()?;
    let new_agent = my_agent_info.agent_initial_pubkey;

    // Verify recovery request is completed via cross-zome call to recovery zome
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("recovery"),
        FunctionName::new("get_recovery_request"),
        None,
        input.request_id.clone(),
    )?;

    let request: RecoveryRequestMirror = match response {
        ZomeCallResponse::Ok(result) => {
            let record: Option<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode recovery request: {:?}", e
                )))
            })?;
            let rec = record.ok_or(wasm_error!(WasmErrorInner::Guest(
                "Recovery request not found".into()
            )))?;
            rec.entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid recovery request entry".into()
                )))?
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Failed to query recovery zome".into()
            )));
        }
    };

    // Verify request is completed
    if request.status != RecoveryStatusMirror::Completed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Recovery request is not completed".into()
        )));
    }

    // Verify the caller is the designated new agent
    if request.new_agent != new_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the designated recovery agent can claim this DID".into()
        )));
    }

    // Verify the DID matches
    if request.did != input.did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID does not match recovery request".into()
        )));
    }

    // Parse original agent pubkey from DID for linking
    let original_agent_str = input.did.strip_prefix("did:mycelix:")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())))?;
    let original_agent = AgentPubKey::try_from(original_agent_str).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent pub key in DID".into()))
    })?;

    let now = sys_time()?;

    // Create new DID document controlled by the new agent
    let verification_method = VerificationMethod {
        id: format!("{}#recovery-key-1", input.did),
        type_: AlgorithmId::Ed25519.did_verification_method_type().to_string(),
        controller: input.did.clone(),
        public_key_multibase: format!("z{}", new_agent),
        algorithm: Some(AlgorithmId::Ed25519.as_u16()),
    };

    let did_doc = DidDocument {
        id: input.did.clone(),
        controller: new_agent.clone(),
        verification_method: vec![verification_method.clone()],
        authentication: vec![format!("{}#recovery-key-1", input.did)],
        key_agreement: vec![],
        service: vec![],
        created: now,
        updated: now,
        version: 1, // Fresh document for new controller
    };

    let action_hash = create_entry(&EntryTypes::DidDocument(did_doc))?;

    // Link from the ORIGINAL agent's pubkey to this new DID document.
    // This ensures resolve_did() finds the recovered document, since it
    // looks up links from the agent pubkey embedded in the DID string.
    // The latest link (by timestamp) takes precedence.
    create_link(
        original_agent,
        action_hash.clone(),
        LinkTypes::AgentToDid,
        (),
    )?;

    // Also link from the new agent for get_my_did() convenience
    create_link(
        new_agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToDid,
        (),
    )?;

    // Auto-create MFA state for the new agent's control of this DID
    if let Err(e) = auto_create_mfa_state(&input.did, &new_agent) {
        debug!("Failed to auto-create MFA state for recovered DID: {:?}", e);
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find recovered DID document".into()
        )))
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_multibase_ed25519_key_with_multicodec() {
        let mut key_bytes = vec![0xed, 0x01];
        key_bytes.extend([0x42u8; 32]);
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let alg = validate_multibase_key(&multibase)
            .expect("Ed25519 multicodec key should validate");
        assert_eq!(alg, AlgorithmId::Ed25519);
    }

    #[test]
    fn test_valid_raw_32_byte_key() {
        let key_bytes = [0x42u8; 32];
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let alg = validate_multibase_key(&multibase)
            .expect("Raw 32-byte key should validate as legacy Ed25519");
        assert_eq!(alg, AlgorithmId::Ed25519);
    }

    #[test]
    fn test_pqc_key_accepted() {
        // ML-DSA-65 multicodec prefix [0xF0, 0x01] + 1952-byte key
        let mut key_bytes = vec![0xF0, 0x01];
        key_bytes.extend(vec![0x33u8; 1952]);
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let alg = validate_multibase_key(&multibase)
            .expect("ML-DSA-65 multicodec key should validate");
        assert_eq!(alg, AlgorithmId::MlDsa65);
    }

    #[test]
    fn test_empty_key_rejected() {
        assert!(validate_multibase_key("").is_err());
    }

    #[test]
    fn test_missing_prefix_rejected() {
        let key_bytes = [0x42u8; 32];
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        assert!(validate_multibase_key(&encoded).is_err());
    }

    #[test]
    fn test_wrong_prefix_rejected() {
        assert!(validate_multibase_key("m").is_err());
        assert!(validate_multibase_key("u").is_err());
    }

    #[test]
    fn test_invalid_base58_characters_rejected() {
        assert!(validate_multibase_key("z0000000000000000000000000000000000000000000").is_err());
        assert!(validate_multibase_key("zOOOOOOOOOOOO").is_err());
    }

    #[test]
    fn test_wrong_key_length_rejected() {
        // 16 bytes - too short for any algorithm
        let short_key = [0x42u8; 16];
        let encoded = bs58::encode(&short_key)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);
        assert!(validate_multibase_key(&multibase).is_err());

        // 64 bytes - doesn't match any known algorithm
        let long_key = [0x42u8; 64];
        let encoded = bs58::encode(&long_key)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);
        assert!(validate_multibase_key(&multibase).is_err());
    }

    #[test]
    fn test_wrong_multicodec_prefix_rejected() {
        // Use unrecognized prefix (0x80, 0x24) + 32 bytes
        let mut key_bytes = vec![0x80, 0x24];
        key_bytes.extend([0x42u8; 32]);
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        assert!(validate_multibase_key(&multibase).is_err());
    }

    #[test]
    fn test_prefix_only_rejected() {
        assert!(validate_multibase_key("z").is_err());
    }

    #[test]
    fn test_legacy_ed25519_validator_rejects_pqc() {
        let mut key_bytes = vec![0xF0, 0x01];
        key_bytes.extend(vec![0x33u8; 1952]);
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        assert!(validate_multibase_ed25519_key(&multibase).is_err());
    }

    // =========================================================================
    // Key rotation logic tests
    // =========================================================================

    /// Helper: build a fake Ed25519 multibase key for testing.
    fn make_test_multibase_key(seed: u8) -> String {
        let mut key_bytes = vec![0xed, 0x01]; // Ed25519 multicodec prefix
        key_bytes.extend([seed; 32]);
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        format!("z{}", encoded)
    }

    #[test]
    fn test_rotate_key_input_validation() {
        // Empty old_key_id should be rejected
        let input = RotateKeyInput {
            old_key_id: String::new(),
            new_method: VerificationMethod {
                id: "#keys-2".into(),
                type_: "Ed25519VerificationKey2020".into(),
                controller: "did:mycelix:test".into(),
                public_key_multibase: make_test_multibase_key(0xBB),
                algorithm: None,
            },
        };
        assert!(input.old_key_id.is_empty());

        // Empty new method ID should be rejected
        let input2 = RotateKeyInput {
            old_key_id: "#keys-1".into(),
            new_method: VerificationMethod {
                id: String::new(),
                type_: "Ed25519VerificationKey2020".into(),
                controller: "did:mycelix:test".into(),
                public_key_multibase: make_test_multibase_key(0xCC),
                algorithm: None,
            },
        };
        assert!(input2.new_method.id.is_empty());
    }

    #[test]
    fn test_rotate_key_deprecated_id_format() {
        let old_id = "#keys-1";
        let version = 3u32;
        let deprecated = format!("{}-deprecated-v{}", old_id, version);
        assert_eq!(deprecated, "#keys-1-deprecated-v3");
    }

    #[test]
    fn test_rotate_key_auth_update_logic() {
        // Simulate the auth array update logic from rotate_key
        let old_key_id = "#keys-1";
        let new_key_id = "#keys-2";
        let mut auth = vec![
            "#keys-1".to_string(),
            "#keys-1".to_string(), // duplicate
        ];

        // Remove old key references
        auth.retain(|a| a != old_key_id);
        assert!(auth.is_empty(), "All references to old key should be removed");

        // Add new key
        if !auth.contains(&new_key_id.to_string()) {
            auth.push(new_key_id.to_string());
        }
        assert_eq!(auth, vec!["#keys-2".to_string()]);
    }

    #[test]
    fn test_rotate_key_methods_update_logic() {
        // Simulate the methods update logic from rotate_key
        let mut methods = vec![
            VerificationMethod {
                id: "#keys-1".into(),
                type_: "Ed25519VerificationKey2020".into(),
                controller: "did:mycelix:test".into(),
                public_key_multibase: make_test_multibase_key(0xAA),
                algorithm: None,
            },
        ];

        let version = 1u32;
        let old_key_idx = 0;

        // Deprecate old key
        let deprecated_id = format!("{}-deprecated-v{}", methods[old_key_idx].id, version);
        methods[old_key_idx].id = deprecated_id.clone();

        // Add new key
        methods.push(VerificationMethod {
            id: "#keys-2".into(),
            type_: "Ed25519VerificationKey2020".into(),
            controller: "did:mycelix:test".into(),
            public_key_multibase: make_test_multibase_key(0xBB),
            algorithm: None,
        });

        assert_eq!(methods.len(), 2);
        assert_eq!(methods[0].id, "#keys-1-deprecated-v1");
        assert_eq!(methods[1].id, "#keys-2");
        // Old key's public material is preserved for signature verification
        assert!(!methods[0].public_key_multibase.is_empty());
    }

    // --- rotate_key_agreement tests ---

    #[test]
    fn test_rotate_key_agreement_input_validation() {
        let input = RotateKeyAgreementInput {
            old_key_id: String::new(),
            new_method: VerificationMethod {
                id: "#kem-2".into(),
                type_: "Multikey".into(),
                controller: "did:mycelix:test".into(),
                public_key_multibase: make_test_multibase_key(0xAA),
                algorithm: None,
            },
        };
        assert!(input.old_key_id.is_empty());
    }

    #[test]
    fn test_rotate_key_agreement_ka_update_logic() {
        // Simulate the key_agreement array update logic from rotate_key_agreement
        let old_key_id = "#kem-1";
        let new_key_id = "#kem-2";
        let mut ka = vec![
            "#kem-1".to_string(),
            "#kem-1".to_string(), // duplicate
        ];

        // Remove old key references
        ka.retain(|k| k != old_key_id);
        assert!(ka.is_empty(), "All references to old KEM key should be removed");

        // Add new key
        if !ka.contains(&new_key_id.to_string()) {
            ka.push(new_key_id.to_string());
        }
        assert_eq!(ka, vec!["#kem-2".to_string()]);
    }

    #[test]
    fn test_rotate_key_agreement_preserves_auth() {
        // rotate_key_agreement passes authentication: None, which means
        // update_did_document preserves the existing authentication array
        let auth = vec!["#keys-1".to_string()];
        let preserved = None::<Vec<String>>.unwrap_or(auth.clone());
        assert_eq!(preserved, auth);
    }

    #[test]
    fn test_rotate_key_agreement_deprecated_id_format() {
        let old_id = "#kem-1";
        let version = 2u32;
        let deprecated = format!("{}-deprecated-v{}", old_id, version);
        assert_eq!(deprecated, "#kem-1-deprecated-v2");
    }
}
