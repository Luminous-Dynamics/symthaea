// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use std::collections::{HashMap, HashSet};
use ed25519_dalek::{Signature, VerifyingKey, Verifier};

//================================
// Entry Types
//================================

/// Identity factors for multi-factor authentication
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityFactors {
    pub did: String,                   // Associated DID
    pub factors: Vec<IdentityFactor>,  // All factors
    pub created: i64,                  // Microseconds since UNIX epoch
    pub updated: i64,                  // Last update timestamp
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct IdentityFactor {
    pub factor_id: String,
    pub factor_type: String,           // "CryptoKey", "GitcoinPassport", etc.
    pub category: String,              // "PRIMARY", "REPUTATION", "SOCIAL", "BACKUP"
    pub status: String,                // "ACTIVE", "SUSPENDED", "REVOKED"
    pub metadata: String,              // JSON-encoded metadata
    pub added: i64,
    pub last_verified: i64,
}

/// Verifiable credentials
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    pub id: String,                    // Credential ID
    pub issuer_did: String,            // Who issued
    pub subject_did: String,           // Who it's about
    pub vc_type: String,               // "VerifiedHuman", "DomainExpert", etc.
    pub claims: String,                // JSON-encoded claims
    pub proof: Vec<u8>,                // Ed25519 signature
    pub issued_at: i64,
    pub expires_at: Option<i64>,
}

/// Computed identity trust signals
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentitySignals {
    pub did: String,
    pub assurance_level: String,       // "E0", "E1", "E2", "E3", "E4"
    pub sybil_resistance: f64,         // 0.0-1.0
    pub risk_level: String,            // "LOW", "MEDIUM", "HIGH"
    pub guardian_graph_diversity: f64, // 0.0-1.0 (from guardian_graph zome)
    pub verified_human: bool,
    pub credential_count: u32,
    pub computed_at: i64,
}

//================================
// Link Types
//================================

#[hdk_link_types]
pub enum LinkTypes {
    IdentityFactorsLink,     // DID -> IdentityFactors
    CredentialLink,          // DID -> VerifiableCredential
    CredentialTypeLink,      // VC Type -> VerifiableCredential
    IdentitySignalsLink,     // DID -> IdentitySignals
    RevocationLink,          // Credential ID -> RevocationEntry
}

/// Credential revocation entry stored on DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialRevocation {
    pub credential_id: String,           // ID of revoked credential
    pub issuer_did: String,              // DID that issued the credential
    pub revoked_by: String,              // DID that performed revocation
    pub reason: String,                  // Revocation reason
    pub revoked_at: i64,                 // Timestamp of revocation
    pub proof: Vec<u8>,                  // Ed25519 signature proving authority to revoke
}

//================================
// Entry Definitions
//================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    IdentityFactors(IdentityFactors),
    #[entry_type]
    VerifiableCredential(VerifiableCredential),
    #[entry_type]
    IdentitySignals(IdentitySignals),
    #[entry_type]
    CredentialRevocation(CredentialRevocation),
}

//================================
// Zome Functions - Identity Factors
//================================

/// Store identity factors for a DID
#[hdk_extern]
pub fn store_identity_factors(input: StoreFactorsInput) -> ExternResult<ActionHash> {
    // 1. Verify caller controls the DID
    verify_did_controller(&input.did)?;

    // 2. Validate factors
    validate_factors(&input.factors)?;

    // 3. Create entry
    let identity_factors = IdentityFactors {
        did: input.did.clone(),
        factors: input.factors.clone(),
        created: sys_time()?.as_micros(),
        updated: sys_time()?.as_micros(),
    };

    let action_hash = create_entry(&EntryTypes::IdentityFactors(identity_factors))?;

    // 4. Create link from DID
    let path = Path::from(format!("identity_factors.{}", input.did)).typed(LinkTypes::IdentityFactorsLink)?;
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::IdentityFactorsLink,
        LinkTag::new("factors")
    )?;

    debug!("Identity factors stored for DID: {}", input.did);
    Ok(action_hash)
}

/// Get identity factors for a DID
#[hdk_extern]
pub fn get_identity_factors(did: String) -> ExternResult<Option<IdentityFactors>> {
    let path = Path::from(format!("identity_factors.{}", did));
    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::IdentityFactorsLink
        )?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent (last link)
    let action_hash = links.last().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let factors = IdentityFactors::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(factors));
            }
        }
    }

    Ok(None)
}

/// Update identity factors (add/remove/modify)
#[hdk_extern]
pub fn update_identity_factors(input: UpdateFactorsInput) -> ExternResult<ActionHash> {
    // 1. Verify caller controls DID
    verify_did_controller(&input.did)?;

    // 2. Get existing factors
    let existing = get_identity_factors(input.did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No existing factors found".into())))?;

    // 3. Merge with new factors
    let mut updated_factors = existing.factors.clone();

    for new_factor in input.factors {
        // Check if factor exists
        if let Some(pos) = updated_factors.iter().position(|f| f.factor_id == new_factor.factor_id) {
            // Update existing
            updated_factors[pos] = new_factor;
        } else {
            // Add new
            updated_factors.push(new_factor);
        }
    }

    // 4. Validate updated factors
    validate_factors(&updated_factors)?;

    // 5. Create new entry
    let identity_factors = IdentityFactors {
        did: input.did.clone(),
        factors: updated_factors,
        created: existing.created,
        updated: sys_time()?.as_micros(),
    };

    let action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::IdentityFactors(identity_factors)
    )?;

    // 6. Create new link
    let path = Path::from(format!("identity_factors.{}", input.did));
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::IdentityFactorsLink,
        LinkTag::new("factors")
    )?;

    debug!("Identity factors updated for DID: {}", input.did);
    Ok(action_hash)
}

//================================
// Zome Functions - Verifiable Credentials
//================================

/// Store a verifiable credential
#[hdk_extern]
pub fn store_verifiable_credential(input: StoreCredentialInput) -> ExternResult<ActionHash> {
    // 1. Verify issuer signature
    verify_credential_proof(&input.credential)?;

    // 2. Check credential is not expired
    if let Some(expires_at) = input.credential.expires_at {
        if sys_time()?.as_micros() > expires_at {
            return Err(wasm_error!(WasmErrorInner::Guest("Credential expired".into())));
        }
    }

    // 3. Create entry
    let action_hash = create_entry(&EntryTypes::VerifiableCredential(input.credential.clone()))?;

    // 4. Create link from subject DID
    let subject_path = Path::from(format!("credentials.{}", input.credential.subject_did)).typed(LinkTypes::CredentialLink)?;
    subject_path.ensure()?;
    create_link(
        subject_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CredentialLink,
        LinkTag::new(input.credential.vc_type.as_bytes())
    )?;

    // 5. Create link by credential type (for querying by type)
    let type_path = Path::from(format!("credential_type.{}", input.credential.vc_type)).typed(LinkTypes::CredentialTypeLink)?;
    type_path.ensure()?;
    create_link(
        type_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CredentialTypeLink,
        LinkTag::new(input.credential.subject_did.as_bytes())
    )?;

    debug!(
        "Credential stored: {} for subject: {}",
        input.credential.id,
        input.credential.subject_did
    );
    Ok(action_hash)
}

/// Get all credentials for a DID
#[hdk_extern]
pub fn get_credentials(did: String) -> ExternResult<Vec<VerifiableCredential>> {
    let path = Path::from(format!("credentials.{}", did));
    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::CredentialLink
        )?,
        GetStrategy::default()
    )?;

    let mut credentials = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    if let Ok(cred) = VerifiableCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ) {
                        // Check expiration
                        if let Some(expires_at) = cred.expires_at {
                            if sys_time()?.as_micros() > expires_at {
                                continue; // Skip expired
                            }
                        }
                        credentials.push(cred);
                    }
                }
            }
        }
    }

    Ok(credentials)
}

/// Get credentials by type
#[hdk_extern]
pub fn get_credentials_by_type(vc_type: String) -> ExternResult<Vec<VerifiableCredential>> {
    let path = Path::from(format!("credential_type.{}", vc_type));
    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::CredentialTypeLink
        )?,
        GetStrategy::default()
    )?;

    let mut credentials = Vec::new();

    for link in links {
        let action_hash = link.target.into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    if let Ok(cred) = VerifiableCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ) {
                        // Check expiration
                        if let Some(expires_at) = cred.expires_at {
                            if sys_time()?.as_micros() > expires_at {
                                continue; // Skip expired
                            }
                        }
                        credentials.push(cred);
                    }
                }
            }
        }
    }

    Ok(credentials)
}

/// Revoke a verifiable credential with DHT-stored revocation entry
#[hdk_extern]
pub fn revoke_credential(input: RevokeCredentialInput) -> ExternResult<ActionHash> {
    // 1. Find the credential to verify it exists and get issuer info
    let credential = find_credential_by_id(&input.credential_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            format!("Credential not found: {}", input.credential_id)
        )))?;

    // 2. Verify caller has authority to revoke (must be issuer or subject)
    let caller_did = get_caller_did()?;
    if caller_did != credential.issuer_did && caller_did != credential.subject_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only issuer or subject can revoke a credential".into()
        )));
    }

    // 3. Check if credential is already revoked
    if is_credential_revoked(input.credential_id.clone())? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential is already revoked".into()
        )));
    }

    // 4. Create revocation entry
    let revocation = CredentialRevocation {
        credential_id: input.credential_id.clone(),
        issuer_did: credential.issuer_did.clone(),
        revoked_by: caller_did,
        reason: input.reason,
        revoked_at: sys_time()?.as_micros(),
        proof: input.proof,
    };

    // 5. Store revocation entry on DHT
    let action_hash = create_entry(&EntryTypes::CredentialRevocation(revocation))?;

    // 6. Create link from credential ID for efficient lookup
    let revocation_path = Path::from(format!("revocation.{}", input.credential_id)).typed(LinkTypes::RevocationLink)?;
    revocation_path.ensure()?;
    create_link(
        revocation_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::RevocationLink,
        LinkTag::new("revoked")
    )?;

    // 7. Delete credential links to make it non-discoverable via normal queries
    // Note: The credential entry remains on DHT (immutable), but links are removed
    delete_credential_links(&credential)?;

    debug!("Credential revoked: {} by issuer: {}", input.credential_id, credential.issuer_did);
    Ok(action_hash)
}

/// Check if a credential has been revoked
#[hdk_extern]
pub fn is_credential_revoked(credential_id: String) -> ExternResult<bool> {
    let revocation_path = Path::from(format!("revocation.{}", credential_id));
    let links = get_links(
        LinkQuery::try_new(revocation_path.path_entry_hash()?, LinkTypes::RevocationLink)?,
        GetStrategy::default()
    )?;

    Ok(!links.is_empty())
}

/// Get revocation details for a credential
#[hdk_extern]
pub fn get_credential_revocation(credential_id: String) -> ExternResult<Option<CredentialRevocation>> {
    let revocation_path = Path::from(format!("revocation.{}", credential_id));
    let links = get_links(
        LinkQuery::try_new(revocation_path.path_entry_hash()?, LinkTypes::RevocationLink)?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let action_hash = links.first().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    if let Some(record) = get(action_hash, GetOptions::default())? {
        if let Some(entry) = record.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let revocation = CredentialRevocation::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(revocation));
            }
        }
    }

    Ok(None)
}

/// Get all revoked credential IDs for a DID (as issuer or subject)
#[hdk_extern]
pub fn get_revoked_credentials(did: String) -> ExternResult<Vec<String>> {
    // Get all credentials for this DID first
    let credentials = get_credentials(did.clone())?;
    let mut revoked = Vec::new();

    for cred in credentials {
        if is_credential_revoked(cred.id.clone())? {
            revoked.push(cred.id);
        }
    }

    Ok(revoked)
}

/// Helper function to find a credential by its ID
fn find_credential_by_id(credential_id: &str) -> ExternResult<Option<VerifiableCredential>> {
    // Search through all credential type links to find the credential
    // This is a simple linear search - in production, consider adding an ID->credential link
    let type_paths = vec!["VerifiedHuman", "DomainExpert", "GitcoinPassport", "SocialRecovery"];

    for vc_type in type_paths {
        let creds = get_credentials_by_type(vc_type.to_string())?;
        for cred in creds {
            if cred.id == credential_id {
                return Ok(Some(cred));
            }
        }
    }

    Ok(None)
}

/// Helper function to get caller's DID
fn get_caller_did() -> ExternResult<String> {
    let agent_info = agent_info()?;
    let caller_pubkey = agent_info.agent_initial_pubkey;
    Ok(format!("did:mycelix:{}", holo_hash::HoloHash::to_b64(&caller_pubkey)))
}

/// Helper function to delete credential links
fn delete_credential_links(credential: &VerifiableCredential) -> ExternResult<()> {
    // Delete link from subject DID
    let subject_path = Path::from(format!("credentials.{}", credential.subject_did));
    let subject_links = get_links(
        LinkQuery::try_new(subject_path.path_entry_hash()?, LinkTypes::CredentialLink)?,
        GetStrategy::default()
    )?;

    for link in subject_links {
        if let Some(tag_str) = std::str::from_utf8(&link.tag.0).ok() {
            if tag_str == credential.vc_type {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    // Delete link from credential type
    let type_path = Path::from(format!("credential_type.{}", credential.vc_type));
    let type_links = get_links(
        LinkQuery::try_new(type_path.path_entry_hash()?, LinkTypes::CredentialTypeLink)?,
        GetStrategy::default()
    )?;

    for link in type_links {
        if let Some(tag_str) = std::str::from_utf8(&link.tag.0).ok() {
            if tag_str == credential.subject_did {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    Ok(())
}

//================================
// Zome Functions - Identity Signals
//================================

/// Compute identity signals from factors and credentials
#[hdk_extern]
pub fn compute_identity_signals(did: String) -> ExternResult<IdentitySignals> {
    // 1. Get identity factors
    let factors = get_identity_factors(did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No identity factors found".into())))?;

    // 2. Get credentials
    let credentials = get_credentials(did.clone())?;

    // 3. Compute assurance level
    let assurance_level = compute_assurance_level(&factors.factors, &credentials)?;

    // 4. Compute Sybil resistance
    let sybil_resistance = compute_sybil_resistance(&factors.factors, &credentials)?;

    // 5. Compute risk level
    let risk_level = compute_risk_level(sybil_resistance);

    // 6. Check for VerifiedHuman credential
    let verified_human = credentials.iter().any(|c| c.vc_type == "VerifiedHuman");

    // 7. Query guardian_graph zome for diversity score
    let guardian_graph_diversity = query_guardian_graph_diversity(&did)?;

    // 8. Create signals entry
    let signals = IdentitySignals {
        did: did.clone(),
        assurance_level,
        sybil_resistance,
        risk_level,
        guardian_graph_diversity,
        verified_human,
        credential_count: credentials.len() as u32,
        computed_at: sys_time()?.as_micros(),
    };

    // 9. Store signals
    let action_hash = create_entry(&EntryTypes::IdentitySignals(signals.clone()))?;

    // 10. Create link from DID
    let path = Path::from(format!("identity_signals.{}", did)).typed(LinkTypes::IdentitySignalsLink)?;
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash,
        LinkTypes::IdentitySignalsLink,
        LinkTag::new("signals")
    )?;

    debug!("Identity signals computed for DID: {}", did);
    Ok(signals)
}

/// Query the guardian_graph zome to get diversity score for a DID
///
/// This cross-zome call retrieves the guardian network diversity score
/// which measures how well-distributed the guardian relationships are.
fn query_guardian_graph_diversity(did: &str) -> ExternResult<f64> {
    // Call the guardian_graph zome's get_guardian_diversity_score function
    let diversity_result: ExternResult<f64> = call(
        CallTargetCell::Local,
        "guardian_graph",
        "get_guardian_diversity_score".into(),
        None,
        did.to_string(),
    );

    match diversity_result {
        Ok(diversity) => Ok(diversity),
        Err(e) => {
            // Log the error but don't fail - return 0.0 if guardian_graph unavailable
            debug!("Failed to query guardian_graph diversity for {}: {:?}", did, e);
            Ok(0.0)
        }
    }
}

/// Get identity signals for a DID
#[hdk_extern]
pub fn get_identity_signals(did: String) -> ExternResult<Option<IdentitySignals>> {
    let path = Path::from(format!("identity_signals.{}", did));
    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::IdentitySignalsLink
        )?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent signals
    let action_hash = links.last().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let signals = IdentitySignals::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(signals));
            }
        }
    }

    Ok(None)
}

//================================
// Helper Functions
//================================

/// Verify that the caller controls the given DID
///
/// This function performs full authorization checking:
/// 1. Validates DID format
/// 2. Calls the did_registry zome to resolve the DID document
/// 3. Extracts the controller(s) from the DID document
/// 4. Verifies that the caller's agent public key matches a controller
fn verify_did_controller(did: &str) -> ExternResult<()> {
    // Step 1: Validate DID format
    if !did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Invalid DID format".into())));
    }

    // Step 2: Get the caller's agent public key
    let agent_info = agent_info()?;
    let caller_pubkey = agent_info.agent_initial_pubkey;

    // Step 3: Call did_registry zome to resolve the DID
    let did_document: Option<DIDDocumentExternal> = call(
        CallTargetCell::Local,
        "did_registry",
        "resolve_did".into(),
        None,
        did.to_string(),
    )?;

    let doc = did_document.ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
        format!("DID not found: {}", did)
    )))?;

    // Step 4: Extract the controller(s) from the DID document
    // The DID controller can be:
    // - The DID itself (self-controlled)
    // - A list of controller DIDs
    // - A verification method ID

    // Check if DID ID contains the caller's pubkey (common pattern: did:mycelix:<pubkey>)
    let caller_pubkey_b64 = holo_hash::HoloHash::to_b64(&caller_pubkey);
    let caller_pubkey_hex = hex::encode(caller_pubkey.get_raw_39());

    // Check direct DID ownership (did:mycelix:<agent_pubkey>)
    if doc.id.contains(&caller_pubkey_b64) || doc.id.contains(&caller_pubkey_hex) {
        debug!("DID controller verified: caller pubkey matches DID {}", did);
        return Ok(());
    }

    // Step 5: Check verification methods for caller's key
    for vm in &doc.verification_methods {
        // Check if the verification method's public key matches caller
        if let Ok(vm_key_bytes) = decode_multibase_public_key(&vm.public_key_multibase) {
            let vm_key_raw = vm_key_bytes.to_bytes();
            let caller_raw = caller_pubkey.get_raw_32();

            if vm_key_raw == caller_raw {
                debug!("DID controller verified via verification method: {}", vm.id);
                return Ok(());
            }
        }
    }

    // Step 6: If none of the above matched, caller does not control this DID
    Err(wasm_error!(WasmErrorInner::Guest(
        format!(
            "Caller does not control DID {}. The caller's agent public key must match a controller or verification method.",
            did
        )
    )))
}

fn validate_factors(factors: &[IdentityFactor]) -> ExternResult<()> {
    // 1. Check at least one factor
    if factors.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("At least one factor required".into())));
    }

    // 2. Check for duplicate factor IDs
    let ids: HashSet<_> = factors.iter().map(|f| &f.factor_id).collect();
    if ids.len() != factors.len() {
        return Err(wasm_error!(WasmErrorInner::Guest("Duplicate factor IDs".into())));
    }

    // 3. Validate each factor
    for factor in factors {
        validate_factor(factor)?;
    }

    Ok(())
}

fn validate_factor(factor: &IdentityFactor) -> ExternResult<()> {
    // 1. Check factor type
    match factor.factor_type.as_str() {
        "CryptoKey" | "GitcoinPassport" | "SocialRecovery" | "HardwareKey" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Unsupported factor type: {}", factor.factor_type)
        )))
    }

    // 2. Check category
    match factor.category.as_str() {
        "PRIMARY" | "REPUTATION" | "SOCIAL" | "BACKUP" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid category: {}", factor.category)
        )))
    }

    // 3. Check status
    match factor.status.as_str() {
        "ACTIVE" | "SUSPENDED" | "REVOKED" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid status: {}", factor.status)
        )))
    }

    Ok(())
}

/// Verify the Ed25519 signature on a verifiable credential
///
/// This function performs full cryptographic verification:
/// 1. Validates proof structure (must be 64 bytes for Ed25519)
/// 2. Resolves the issuer DID to get the verification key
/// 3. Constructs the canonical message (credential without proof)
/// 4. Verifies the Ed25519 signature
fn verify_credential_proof(credential: &VerifiableCredential) -> ExternResult<()> {
    // 1. Basic structure checks
    if credential.proof.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Proof cannot be empty".into())));
    }

    if credential.proof.len() != 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid proof length: expected 64 bytes (Ed25519), got {}", credential.proof.len())
        )));
    }

    if credential.issued_at > sys_time()?.as_micros() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issued timestamp is in the future".into()
        )));
    }

    // 2. Resolve issuer DID to get public key
    let issuer_public_key = resolve_issuer_key(&credential.issuer_did)?;

    // 3. Construct canonical message (credential data without proof)
    let canonical_message = create_canonical_credential_message(credential)?;

    // 4. Parse the Ed25519 signature
    let sig_bytes: [u8; 64] = credential.proof.as_slice().try_into()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Invalid signature format".into()
        )))?;
    let signature = Signature::from_bytes(&sig_bytes);

    // 5. Verify the signature
    issuer_public_key.verify(canonical_message.as_bytes(), &signature)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Ed25519 signature verification failed: credential was not signed by issuer".into()
        )))?;

    debug!(
        "Credential proof verified: issuer={}, subject={}",
        credential.issuer_did, credential.subject_did
    );
    Ok(())
}

/// Resolve issuer DID to get Ed25519 public key
///
/// This calls the did_registry zome to resolve the DID and extract
/// the signing key from the verification methods.
fn resolve_issuer_key(issuer_did: &str) -> ExternResult<VerifyingKey> {
    // Call the did_registry zome to resolve the DID
    // The did_registry is in the same DNA, so we can call it directly
    let did_document: Option<DIDDocumentExternal> = call(
        CallTargetCell::Local,
        "did_registry",
        "resolve_did".into(),
        None,
        issuer_did.to_string(),
    )?;

    let doc = did_document.ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
        format!("Issuer DID not found: {}", issuer_did)
    )))?;

    // Find the first Ed25519 verification method
    let vm = doc.verification_methods.iter()
        .find(|v| v.method_type == "Ed25519VerificationKey2020" ||
                  v.method_type == "Ed25519VerificationKey2018")
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
            format!("No Ed25519 verification method found for issuer: {}", issuer_did)
        )))?;

    // Decode the public key from multibase
    decode_multibase_public_key(&vm.public_key_multibase)
}

/// External DID document structure for cross-zome calls
#[derive(Serialize, Deserialize, Debug)]
struct DIDDocumentExternal {
    pub id: String,
    pub verification_methods: Vec<VerificationMethodExternal>,
}

#[derive(Serialize, Deserialize, Debug)]
struct VerificationMethodExternal {
    pub id: String,
    pub method_type: String,
    pub controller: String,
    pub public_key_multibase: String,
}

/// Decode a public key from multibase format (base58btc prefixed with 'z')
fn decode_multibase_public_key(multibase: &str) -> ExternResult<VerifyingKey> {
    // Multibase base58btc is prefixed with 'z'
    if !multibase.starts_with('z') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid multibase prefix: expected 'z' (base58btc), got '{}'",
                multibase.chars().next().unwrap_or('?'))
        )));
    }

    // Decode base58btc (skip the 'z' prefix)
    let decoded = bs58::decode(&multibase[1..])
        .into_vec()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to decode base58btc: {}", e)
        )))?;

    // For Ed25519 keys in multicodec format:
    // First 2 bytes are multicodec prefix (0xed 0x01 for Ed25519)
    // Followed by 32 bytes of public key
    let key_bytes = if decoded.len() == 34 && decoded[0] == 0xed && decoded[1] == 0x01 {
        // Multicodec format: skip 2-byte prefix
        &decoded[2..]
    } else if decoded.len() == 32 {
        // Raw 32-byte key
        &decoded
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid public key length: expected 32 or 34 bytes, got {}", decoded.len())
        )));
    };

    let key_array: [u8; 32] = key_bytes.try_into()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Failed to convert public key bytes to array".into()
        )))?;

    VerifyingKey::from_bytes(&key_array)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Invalid Ed25519 public key".into()
        )))
}

/// Create canonical message for credential signature verification
///
/// This creates a deterministic JSON representation of the credential
/// without the proof field, which is what was originally signed.
fn create_canonical_credential_message(credential: &VerifiableCredential) -> ExternResult<String> {
    #[derive(Serialize)]
    struct CredentialForSigning<'a> {
        pub id: &'a str,
        pub issuer_did: &'a str,
        pub subject_did: &'a str,
        pub vc_type: &'a str,
        pub claims: &'a str,
        pub issued_at: i64,
        pub expires_at: Option<i64>,
    }

    let cred_for_signing = CredentialForSigning {
        id: &credential.id,
        issuer_did: &credential.issuer_did,
        subject_did: &credential.subject_did,
        vc_type: &credential.vc_type,
        claims: &credential.claims,
        issued_at: credential.issued_at,
        expires_at: credential.expires_at,
    };

    serde_json::to_string(&cred_for_signing)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to serialize credential: {}", e)
        )))
}

fn compute_assurance_level(factors: &[IdentityFactor], credentials: &[VerifiableCredential]) -> ExternResult<String> {
    // Filter active factors
    let active_factors: Vec<_> = factors.iter()
        .filter(|f| f.status == "ACTIVE")
        .collect();

    // Count categories
    let categories: HashSet<_> = active_factors.iter()
        .map(|f| f.category.as_str())
        .collect();

    // Check for Gitcoin Passport with high score
    let has_high_gitcoin = active_factors.iter().any(|f| {
        f.factor_type == "GitcoinPassport" && {
            // Parse metadata JSON to check score
            if let Ok(metadata) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&f.metadata) {
                if let Some(score) = metadata.get("score") {
                    if let Some(score_val) = score.as_f64() {
                        return score_val >= 50.0;
                    }
                }
            }
            false
        }
    });

    // Check for social recovery
    let has_social_recovery = active_factors.iter()
        .any(|f| f.factor_type == "SocialRecovery");

    // Determine assurance level
    match active_factors.len() {
        0 => Ok("E0".to_string()), // Anonymous
        1 => Ok("E1".to_string()), // Testimonial
        2 => Ok("E2".to_string()), // Privately Verifiable
        3 => {
            if has_high_gitcoin && has_social_recovery {
                Ok("E3".to_string()) // Cryptographically Proven
            } else {
                Ok("E2".to_string())
            }
        }
        _ => {
            // E4 requires 4+ factors across 4 categories + Gitcoin ≥50
            if categories.len() >= 4 && has_high_gitcoin {
                Ok("E4".to_string()) // Constitutionally Critical
            } else if has_high_gitcoin && has_social_recovery {
                Ok("E3".to_string())
            } else {
                Ok("E2".to_string())
            }
        }
    }
}

fn compute_sybil_resistance(factors: &[IdentityFactor], credentials: &[VerifiableCredential]) -> ExternResult<f64> {
    let active_count = factors.iter().filter(|f| f.status == "ACTIVE").count();
    let verified_human = credentials.iter().any(|c| c.vc_type == "VerifiedHuman");

    // Base score from factor count
    let base_score: f64 = match active_count {
        0 => 0.0,
        1 => 0.2,
        2 => 0.4,
        3 => 0.6,
        _ => 0.8,
    };

    // Bonus for VerifiedHuman credential
    let bonus: f64 = if verified_human { 0.2 } else { 0.0 };

    Ok((base_score + bonus).min(1.0_f64))
}

fn compute_risk_level(sybil_resistance: f64) -> String {
    if sybil_resistance >= 0.7 {
        "LOW".to_string()
    } else if sybil_resistance >= 0.4 {
        "MEDIUM".to_string()
    } else {
        "HIGH".to_string()
    }
}

//================================
// Validation Callback
//================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    // Try to deserialize as each entry type

                    // IdentityFactors
                    if let Ok(factors) = IdentityFactors::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if let Err(e) = validate_factors(&factors.factors) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid identity factors: {:?}", e)
                            ));
                        }
                        return Ok(ValidateCallbackResult::Valid);
                    }

                    // VerifiableCredential
                    if let Ok(credential) = VerifiableCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if let Err(e) = verify_credential_proof(&credential) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid credential proof: {:?}", e)
                            ));
                        }
                        return Ok(ValidateCallbackResult::Valid);
                    }

                    // IdentitySignals - no special validation needed
                    if let Ok(_signals) = IdentitySignals::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        return Ok(ValidateCallbackResult::Valid);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(ValidateCallbackResult::Valid)
}

//================================
// Input Types
//================================

#[derive(Serialize, Deserialize, Debug)]
pub struct StoreFactorsInput {
    pub did: String,
    pub factors: Vec<IdentityFactor>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateFactorsInput {
    pub did: String,
    pub original_action_hash: ActionHash,
    pub factors: Vec<IdentityFactor>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StoreCredentialInput {
    pub credential: VerifiableCredential,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeCredentialInput {
    pub credential_id: String,
    pub reason: String,
    pub proof: Vec<u8>,  // Signature proving authority to revoke
}
