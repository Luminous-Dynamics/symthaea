//! DID Registry Coordinator Zome
//! Business logic for DID:mycelix operations
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use did_registry_integrity::*;
use bs58;

/// Validate a multibase-encoded Ed25519 public key.
///
/// Per the W3C DID specification and multibase encoding:
/// - Must start with 'z' (base58btc prefix)
/// - Remaining characters must be valid base58btc (no 0, O, I, l)
/// - Decoded bytes must be exactly 34 bytes (2-byte multicodec prefix + 32-byte Ed25519 key)
///   or 32 bytes (raw Ed25519 key without multicodec prefix)
fn validate_multibase_ed25519_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("Public key multibase is empty".to_string());
    }

    // Check multibase prefix (z = base58btc)
    if !key.starts_with('z') {
        return Err(format!(
            "Invalid multibase prefix '{}': expected 'z' (base58btc)",
            key.chars().next().unwrap_or('?')
        ));
    }

    let encoded = &key[1..]; // Strip 'z' prefix
    if encoded.is_empty() {
        return Err("Public key multibase has no data after prefix".to_string());
    }

    // Decode base58btc to verify character set and get actual key bytes
    let decoded = bs58::decode(encoded)
        .with_alphabet(bs58::Alphabet::BITCOIN)
        .into_vec()
        .map_err(|e| format!("Invalid base58btc encoding: {}", e))?;

    // Accept either:
    // - 34 bytes: multicodec prefix (0xed, 0x01 for Ed25519) + 32-byte key
    // - 32 bytes: raw Ed25519 key (no multicodec prefix)
    match decoded.len() {
        34 => {
            // Verify Ed25519 multicodec prefix (varint 0xed01)
            if decoded[0] != 0xed || decoded[1] != 0x01 {
                return Err(format!(
                    "Invalid multicodec prefix: expected [0xed, 0x01] (Ed25519), got [{:#04x}, {:#04x}]",
                    decoded[0], decoded[1]
                ));
            }
            Ok(())
        }
        32 => {
            // Raw Ed25519 key without multicodec prefix - acceptable
            Ok(())
        }
        n => {
            Err(format!(
                "Invalid Ed25519 key length: decoded to {} bytes, expected 34 (with multicodec prefix) or 32 (raw)",
                n
            ))
        }
    }
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
            // MFA zome not accessible - log but don't fail DID creation
            // This allows DID creation even if MFA zome is not installed
            debug!("MFA zome unauthorized - DID created without MFA state");
            Ok(())
        }
        ZomeCallResponse::NetworkError(err) => {
            // Network error - log but don't fail DID creation
            debug!("MFA zome network error: {} - DID created without MFA state", err);
            Ok(())
        }
        ZomeCallResponse::CountersigningSession(err) => {
            debug!("MFA zome countersigning error: {} - DID created without MFA state", err);
            Ok(())
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            debug!("MFA zome authentication failed - DID created without MFA state");
            Ok(())
        }
    }
}

/// Create a new DID document for the calling agent
#[hdk_extern]
pub fn create_did() -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_pub_key = agent_info.agent_initial_pubkey;

    // Generate DID identifier
    let did_id = format!("did:mycelix:{}", agent_pub_key);

    // Create default verification method
    let verification_method = VerificationMethod {
        id: format!("{}#keys-1", did_id),
        type_: "Ed25519VerificationKey2020".to_string(),
        controller: did_id.clone(),
        public_key_multibase: format!("z{}", agent_pub_key),
    };

    let now = sys_time()?;
    let did_doc = DidDocument {
        id: did_id.clone(),
        controller: agent_pub_key.clone(),
        verification_method: vec![verification_method.clone()],
        authentication: vec![format!("{}#keys-1", did_id)],
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
            // Validate multibase Ed25519 key format
            if method.type_ == "Ed25519VerificationKey2020" || method.type_ == "Ed25519VerificationKey2018" {
                if let Err(e) = validate_multibase_ed25519_key(&method.public_key_multibase) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        format!("Invalid Ed25519 multibase key in update: {}", e)
                    )));
                }
            }
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

    // Build updated document
    let updated_did = DidDocument {
        id: current_did.id.clone(),
        controller: current_did.controller.clone(),
        verification_method: input
            .verification_method
            .unwrap_or(current_did.verification_method),
        authentication: input.authentication.unwrap_or(current_did.authentication),
        service: input.service.unwrap_or(current_did.service),
        created: current_did.created,
        updated: now,
        version: current_did.version + 1,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::DidDocument(updated_did),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated DID".into()
        )))
}

/// Input for updating a DID document
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDidInput {
    pub verification_method: Option<Vec<VerificationMethod>>,
    pub authentication: Option<Vec<String>>,
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

    // Check if already deactivated
    if is_did_active(current_did.id.clone())? == false {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID is already deactivated".into()
        )));
    }

    let now = sys_time()?;

    let deactivation = DidDeactivation {
        did: current_did.id.clone(),
        reason,
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find deactivation record".into()
        )))
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

    // Validate multibase encoding format for Ed25519 keys
    if method.type_ == "Ed25519VerificationKey2020" || method.type_ == "Ed25519VerificationKey2018" {
        if let Err(e) = validate_multibase_ed25519_key(&method.public_key_multibase) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Invalid Ed25519 multibase key: {}", e)
            )));
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

    let mut methods = current_did.verification_method.clone();
    methods.push(method);

    update_did_document(UpdateDidInput {
        verification_method: Some(methods),
        authentication: None,
        service: None,
    })
}

/// Get my DID (convenience function)
#[hdk_extern]
pub fn get_my_did(_: ()) -> ExternResult<Option<Record>> {
    let agent_info = agent_info()?;
    get_did_document(agent_info.agent_initial_pubkey)
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_multibase_ed25519_key_with_multicodec() {
        // Ed25519 multicodec prefix (0xed, 0x01) + 32 bytes = 34 bytes total
        let mut key_bytes = vec![0xed, 0x01];
        key_bytes.extend([0x42u8; 32]); // 32-byte dummy Ed25519 key
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        assert!(validate_multibase_ed25519_key(&multibase).is_ok());
    }

    #[test]
    fn test_valid_raw_32_byte_key() {
        // Raw 32-byte Ed25519 key (no multicodec prefix)
        let key_bytes = [0x42u8; 32];
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        assert!(validate_multibase_ed25519_key(&multibase).is_ok());
    }

    #[test]
    fn test_empty_key_rejected() {
        assert!(validate_multibase_ed25519_key("").is_err());
    }

    #[test]
    fn test_missing_prefix_rejected() {
        let key_bytes = [0x42u8; 32];
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        // Missing 'z' prefix
        assert!(validate_multibase_ed25519_key(&encoded).is_err());
    }

    #[test]
    fn test_wrong_prefix_rejected() {
        assert!(validate_multibase_ed25519_key("m" ).is_err());
        assert!(validate_multibase_ed25519_key("u" ).is_err());
    }

    #[test]
    fn test_invalid_base58_characters_rejected() {
        // '0', 'O', 'I', 'l' are not in base58btc alphabet
        assert!(validate_multibase_ed25519_key("z0000000000000000000000000000000000000000000").is_err());
        assert!(validate_multibase_ed25519_key("zOOOOOOOOOOOO").is_err());
    }

    #[test]
    fn test_wrong_key_length_rejected() {
        // 16 bytes - too short for Ed25519
        let short_key = [0x42u8; 16];
        let encoded = bs58::encode(&short_key)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);
        assert!(validate_multibase_ed25519_key(&multibase).is_err());

        // 64 bytes - too long for Ed25519
        let long_key = [0x42u8; 64];
        let encoded = bs58::encode(&long_key)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);
        assert!(validate_multibase_ed25519_key(&multibase).is_err());
    }

    #[test]
    fn test_wrong_multicodec_prefix_rejected() {
        // Use P-256 multicodec prefix (0x80, 0x24) instead of Ed25519
        let mut key_bytes = vec![0x80, 0x24];
        key_bytes.extend([0x42u8; 32]); // 32-byte key body
        let encoded = bs58::encode(&key_bytes)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        assert!(validate_multibase_ed25519_key(&multibase).is_err());
    }

    #[test]
    fn test_prefix_only_rejected() {
        assert!(validate_multibase_ed25519_key("z").is_err());
    }
}
