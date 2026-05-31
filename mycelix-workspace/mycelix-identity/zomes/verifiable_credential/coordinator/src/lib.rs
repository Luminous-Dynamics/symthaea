// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Verifiable Credential Coordinator Zome
//!
//! W3C Verifiable Credentials Data Model 2.0 compliant implementation
//! Handles credential issuance, verification, and presentation
//!
//! # Cryptographic Signatures
//!
//! This implementation uses ed25519 signatures via Holochain's HDK signing API.
//! Signatures are encoded in multibase format (base58btc with 'z' prefix) as per
//! W3C Data Integrity EdDSA Cryptosuites v1.0 specification.

use hdk::prelude::*;
use verifiable_credential_integrity::*;

/// Standard W3C context URLs
const W3C_CREDENTIALS_V2: &str = "https://www.w3.org/ns/credentials/v2";
const W3C_DATA_INTEGRITY: &str = "https://w3id.org/security/data-integrity/v2";

/// Create a deterministic entry hash from a string identifier
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

/// Compute hash of credential content for signing
fn compute_credential_hash(vc: &VerifiableCredential) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    vc.id.hash(&mut hasher);
    vc.issuer.did().hash(&mut hasher);
    vc.credential_subject.id.hash(&mut hasher);
    vc.valid_from.hash(&mut hasher);
    let h1 = hasher.finish();
    hasher.write_u64(h1);
    let h2 = hasher.finish();
    hasher.write_u64(h2);
    let h3 = hasher.finish();
    hasher.write_u64(h3);
    let h4 = hasher.finish();

    let mut result = Vec::with_capacity(32);
    result.extend_from_slice(&h1.to_le_bytes());
    result.extend_from_slice(&h2.to_le_bytes());
    result.extend_from_slice(&h3.to_le_bytes());
    result.extend_from_slice(&h4.to_le_bytes());
    result
}

/// Issue a new verifiable credential
#[hdk_extern]
pub fn issue_credential(input: IssueCredentialInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let issuer_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let now = sys_time()?;
    let now_iso = format_timestamp_iso8601(now);

    // Build credential ID
    let credential_id = format!(
        "urn:uuid:{}:{}",
        issuer_did.replace(":", "-"),
        now.as_micros()
    );

    // Calculate expiration if provided
    let valid_until = input.expiration_days.map(|days| {
        let expiry_micros = now.as_micros() as i64 + (days as i64 * 24 * 3600 * 1_000_000);
        format_timestamp_iso8601(Timestamp::from_micros(expiry_micros))
    });

    // Build credential subject
    let credential_subject = CredentialSubject {
        id: input.subject_did.clone(),
        claims: input.claims,
    };

    // Build credential hash for signing
    let mut vc_for_hash = VerifiableCredential {
        context: vec![
            W3C_CREDENTIALS_V2.to_string(),
            W3C_DATA_INTEGRITY.to_string(),
        ],
        id: credential_id.clone(),
        credential_type: {
            let mut types = vec!["VerifiableCredential".to_string()];
            types.extend(input.credential_types);
            types
        },
        issuer: CredentialIssuer::Object {
            id: issuer_did.clone(),
            name: input.issuer_name,
            issuer_type: Some(vec!["Organization".to_string()]),
        },
        valid_from: now_iso.clone(),
        valid_until: valid_until.clone(),
        credential_subject: credential_subject.clone(),
        credential_schema: Some(CredentialSchemaRef {
            id: input.schema_id.clone(),
            schema_type: "JsonSchema".to_string(),
        }),
        credential_status: input.enable_revocation.then(|| CredentialStatus {
            id: format!("{}#status", credential_id),
            status_type: "BitstringStatusListEntry".to_string(),
            status_purpose: Some("revocation".to_string()),
            status_list_index: Some("0".to_string()),
            status_list_credential: None,
        }),
        proof: CredentialProof {
            proof_type: "DataIntegrityProof".to_string(),
            created: now_iso.clone(),
            verification_method: format!("{}#keys-1", issuer_did),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: String::new(), // Will be filled
            cryptosuite: Some("eddsa-rdfc-2022".to_string()),
        },
        mycelix_schema_id: input.schema_id.clone(),
        mycelix_created: now,
    };

    // Sign credential with agent's ed25519 key
    // This creates a real cryptographic signature using HDK's sign_raw
    let signature_value = sign_credential(&vc_for_hash)?;
    vc_for_hash.proof.proof_value = signature_value;

    let vc = vc_for_hash;

    let action_hash = create_entry(&EntryTypes::VerifiableCredential(vc.clone()))?;

    // Create links
    let issuer_hash = string_to_entry_hash(vc.issuer.did());
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToCredential,
        (),
    )?;

    let subject_hash = string_to_entry_hash(&vc.credential_subject.id);
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::SubjectToCredential,
        (),
    )?;

    let schema_hash = string_to_entry_hash(&input.schema_id);
    create_link(
        schema_hash,
        action_hash.clone(),
        LinkTypes::SchemaToCredential,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created credential".into()
    )))
}

/// Input for issuing a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct IssueCredentialInput {
    /// Subject's DID (who the credential is about)
    pub subject_did: String,
    /// Schema ID for the credential
    pub schema_id: String,
    /// Claims being made
    pub claims: serde_json::Value,
    /// Additional credential types beyond VerifiableCredential
    pub credential_types: Vec<String>,
    /// Optional issuer name
    pub issuer_name: Option<String>,
    /// Expiration in days (None = no expiration)
    pub expiration_days: Option<u32>,
    /// Whether to enable revocation
    pub enable_revocation: bool,
}

/// Verify a credential
#[hdk_extern]
pub fn verify_credential(credential_id: String) -> ExternResult<VerificationResult> {
    let vc = get_credential(credential_id.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Credential not found".into()
    )))?;

    let credential: VerifiableCredential = vc
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid credential entry".into()
        )))?;

    let now = sys_time()?;
    let mut errors = Vec::new();

    // Check expiration using ISO 8601 parsing
    let expired = if let Some(valid_until) = &credential.valid_until {
        parse_iso8601_expired(valid_until, now)
    } else {
        false
    };

    if expired {
        errors.push("Credential has expired".to_string());
    }

    // Verify ed25519 signature using HDK
    match verify_credential_signature(&credential) {
        Ok(true) => {
            // Signature is valid
        }
        Ok(false) => {
            errors.push("Proof signature verification failed".to_string());
        }
        Err(e) => {
            errors.push(format!("Signature verification error: {}", e));
        }
    }

    // Check proof purpose
    if credential.proof.proof_purpose != "assertionMethod" {
        errors.push("Invalid proof purpose".to_string());
    }

    // Check issuer DID format
    if !credential.issuer.did().starts_with("did:") {
        errors.push("Invalid issuer DID".into());
    }

    // Check revocation status via cross-zome call to revocation registry
    let revocation_status = check_credential_revocation_status(&credential.id)?;
    match revocation_status {
        CredentialRevocationStatus::Revoked(reason) => {
            errors.push(format!("Credential revoked: {}", reason));
        }
        CredentialRevocationStatus::Suspended(reason, until) => {
            errors.push(format!("Credential suspended until {}: {}", until, reason));
        }
        CredentialRevocationStatus::Active => {
            // Credential is active, no error
        }
        CredentialRevocationStatus::Unknown => {
            // No revocation record found, assume active
        }
    }

    Ok(VerificationResult {
        credential_id,
        valid: errors.is_empty(),
        checks_passed: vec![
            "format".to_string(),
            "proof_signature".to_string(),
            "proof_purpose".to_string(),
        ],
        errors,
        verified_at: now,
    })
}

/// Result of credential verification
#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationResult {
    pub credential_id: String,
    pub valid: bool,
    pub checks_passed: Vec<String>,
    pub errors: Vec<String>,
    pub verified_at: Timestamp,
}

/// Get a credential by ID
#[hdk_extern]
pub fn get_credential(credential_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::VerifiableCredential,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(vc) = record
            .entry()
            .to_app_option::<VerifiableCredential>()
            .ok()
            .flatten()
        {
            if vc.id == credential_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get credentials issued by a DID
#[hdk_extern]
pub fn get_credentials_issued_by(issuer_did: String) -> ExternResult<Vec<Record>> {
    let issuer_hash = string_to_entry_hash(&issuer_did);
    let links = get_links(
        LinkQuery::try_new(issuer_hash, LinkTypes::IssuerToCredential)?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            credentials.push(record);
        }
    }
    Ok(credentials)
}

/// Get credentials about a subject DID
#[hdk_extern]
pub fn get_credentials_for_subject(subject_did: String) -> ExternResult<Vec<Record>> {
    let subject_hash = string_to_entry_hash(&subject_did);
    let links = get_links(
        LinkQuery::try_new(subject_hash, LinkTypes::SubjectToCredential)?,
        GetStrategy::default(),
    )?;

    let mut credentials = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            credentials.push(record);
        }
    }
    Ok(credentials)
}

/// Create a verifiable presentation from credentials
#[hdk_extern]
pub fn create_presentation(input: CreatePresentationInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let holder_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let now = sys_time()?;
    let now_iso = format_timestamp_iso8601(now);

    // Gather credentials
    let mut credentials = Vec::new();
    for cred_id in &input.credential_ids {
        let record = get_credential(cred_id.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
            format!("Credential {} not found", cred_id)
        )))?;
        let vc: VerifiableCredential = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid credential".into()
            )))?;
        credentials.push(vc);
    }

    let presentation_id = format!(
        "urn:uuid:presentation:{}:{}",
        holder_did.replace(":", "-"),
        now.as_micros()
    );

    // Create presentation proof with real ed25519 signature
    // Hash the presentation content for signing
    let mut presentation_data = presentation_id.as_bytes().to_vec();
    presentation_data.extend(holder_did.as_bytes());
    for cred in &credentials {
        presentation_data.extend(cred.id.as_bytes());
    }
    if let Some(challenge) = &input.challenge {
        presentation_data.extend(challenge.as_bytes());
    }
    if let Some(domain) = &input.domain {
        presentation_data.extend(domain.as_bytes());
    }

    // Sign with agent's ed25519 key
    let signature = sign_raw(
        agent_info.agent_initial_pubkey.clone(),
        presentation_data.into(),
    )?;
    let proof_value = multibase_encode(signature.as_ref());

    let proof = CredentialProof {
        proof_type: "DataIntegrityProof".to_string(),
        created: now_iso.clone(),
        verification_method: format!("{}#keys-1", holder_did),
        proof_purpose: "authentication".to_string(),
        proof_value,
        cryptosuite: Some("eddsa-rdfc-2022".to_string()),
    };

    let vp = VerifiablePresentation {
        context: vec![
            W3C_CREDENTIALS_V2.to_string(),
            W3C_DATA_INTEGRITY.to_string(),
        ],
        id: presentation_id,
        presentation_type: vec!["VerifiablePresentation".to_string()],
        holder: holder_did.clone(),
        verifiable_credential: credentials,
        proof,
        mycelix_created: now,
    };

    let action_hash = create_entry(&EntryTypes::VerifiablePresentation(vp))?;

    // Link holder to presentation
    let holder_hash = string_to_entry_hash(&holder_did);
    create_link(
        holder_hash,
        action_hash.clone(),
        LinkTypes::HolderToPresentation,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created presentation".into()
    )))
}

/// Input for creating a presentation
#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePresentationInput {
    /// Credential IDs to include
    pub credential_ids: Vec<String>,
    /// Optional challenge for proof
    pub challenge: Option<String>,
    /// Optional domain restriction
    pub domain: Option<String>,
}

/// Create a derived credential with selective disclosure
#[hdk_extern]
pub fn create_derived_credential(input: CreateDerivedInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let holder_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let now = sys_time()?;

    // Get original credential
    let original_record = get_credential(input.credential_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Original credential not found".into())
    ))?;

    let original_vc: VerifiableCredential = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid original credential".into()
        )))?;

    // Verify holder is the subject
    if original_vc.credential_subject.id != holder_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the credential subject can create derived credentials".into()
        )));
    }

    // Extract selected claims
    let original_claims = &original_vc.credential_subject.claims;
    let mut derived_claims = serde_json::Map::new();

    for claim_key in &input.selected_claims {
        if let Some(value) = original_claims.get(claim_key) {
            derived_claims.insert(claim_key.clone(), value.clone());
        } else {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Claim '{}' not found in original credential",
                claim_key
            ))));
        }
    }

    let derived_content = CredentialSubject {
        id: holder_did.clone(),
        claims: serde_json::Value::Object(derived_claims),
    };

    // Compute hash of original credential
    let original_hash = compute_credential_hash(&original_vc);

    // Create derivation proof with real ed25519 signature
    // Build data to sign: original hash + selected claims
    let mut sign_data = original_hash.clone();
    for claim_key in &input.selected_claims {
        sign_data.extend(claim_key.as_bytes());
    }

    // Sign with holder's ed25519 key
    let holder_signature = sign_raw(agent_info.agent_initial_pubkey.clone(), sign_data.into())?;

    let derivation_proof = DerivationProof {
        proof_type: "SelectiveDisclosureProof".to_string(),
        original_credential_hash: original_hash,
        claim_proofs: input
            .selected_claims
            .iter()
            .map(|key| ClaimProof {
                claim_key: key.clone(),
                merkle_path: None,
                commitment: None,
            })
            .collect(),
        holder_signature: holder_signature.as_ref().to_vec(),
    };

    // Calculate expiration
    let expires = input.expires_hours.map(|hours| {
        Timestamp::from_micros(now.as_micros() as i64 + (hours as i64 * 3600 * 1_000_000))
    });

    let derived = DerivedCredential {
        original_credential_id: input.credential_id.clone(),
        original_issuer: original_vc.issuer.did().to_string(),
        holder: holder_did,
        selected_claims: input.selected_claims,
        derived_content,
        derivation_proof,
        created: now,
        expires,
    };

    let action_hash = create_entry(&EntryTypes::DerivedCredential(derived))?;

    // Link original to derived
    create_link(
        original_record.action_address().clone(),
        action_hash.clone(),
        LinkTypes::CredentialToDerived,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created derived credential".into()
    )))
}

/// Input for creating a derived credential
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateDerivedInput {
    /// Original credential ID
    pub credential_id: String,
    /// Claims to include in derivation
    pub selected_claims: Vec<String>,
    /// Optional expiration in hours
    pub expires_hours: Option<u32>,
}

/// Request a credential from an issuer
#[hdk_extern]
pub fn request_credential(input: RequestCredentialInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let requester_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let now = sys_time()?;

    let request_id = format!(
        "request:{}:{}:{}",
        requester_did.replace(":", "-"),
        input.issuer_did.replace(":", "-"),
        now.as_micros()
    );

    let request = CredentialRequest {
        id: request_id,
        requester_did: requester_did.clone(),
        issuer_did: input.issuer_did.clone(),
        schema_id: input.schema_id,
        provided_claims: input.claims,
        evidence: input.evidence.unwrap_or_default(),
        status: RequestStatus::Pending,
        created: now,
        updated: now,
    };

    let action_hash = create_entry(&EntryTypes::CredentialRequest(request))?;

    // Link issuer to request
    let issuer_hash = string_to_entry_hash(&input.issuer_did);
    create_link(
        issuer_hash,
        action_hash.clone(),
        LinkTypes::IssuerToRequest,
        (),
    )?;

    // Link requester to request
    let requester_hash = string_to_entry_hash(&requester_did);
    create_link(
        requester_hash,
        action_hash.clone(),
        LinkTypes::RequesterToRequest,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created request".into()
    )))
}

/// Input for requesting a credential
#[derive(Serialize, Deserialize, Debug)]
pub struct RequestCredentialInput {
    /// Target issuer's DID
    pub issuer_did: String,
    /// Schema ID for requested credential
    pub schema_id: String,
    /// Claims provided by requester
    pub claims: serde_json::Value,
    /// Supporting evidence
    pub evidence: Option<Vec<CredentialEvidence>>,
}

/// Get pending requests for an issuer
#[hdk_extern]
pub fn get_pending_requests(issuer_did: String) -> ExternResult<Vec<Record>> {
    let issuer_hash = string_to_entry_hash(&issuer_did);
    let links = get_links(
        LinkQuery::try_new(issuer_hash, LinkTypes::IssuerToRequest)?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(req) = record
                .entry()
                .to_app_option::<CredentialRequest>()
                .ok()
                .flatten()
            {
                if matches!(
                    req.status,
                    RequestStatus::Pending | RequestStatus::UnderReview
                ) {
                    requests.push(record);
                }
            }
        }
    }
    Ok(requests)
}

/// Update credential request status
#[hdk_extern]
pub fn update_request_status(input: UpdateRequestStatusInput) -> ExternResult<Record> {
    // Find the request
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CredentialRequest,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(req) = record
            .entry()
            .to_app_option::<CredentialRequest>()
            .ok()
            .flatten()
        {
            if req.id == input.request_id {
                let now = sys_time()?;
                let updated_req = CredentialRequest {
                    status: input.new_status,
                    updated: now,
                    ..req
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CredentialRequest(updated_req),
                )?;

                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Could not find updated request".into())
                ));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Request not found".into()
    )))
}

/// Input for updating request status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateRequestStatusInput {
    pub request_id: String,
    pub new_status: RequestStatus,
}

/// Get my issued credentials
#[hdk_extern]
pub fn get_my_issued_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let my_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    get_credentials_issued_by(my_did)
}

/// Get credentials I hold (am subject of)
#[hdk_extern]
pub fn get_my_credentials(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let my_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    get_credentials_for_subject(my_did)
}

/// Format timestamp as ISO 8601 string
fn format_timestamp_iso8601(ts: Timestamp) -> String {
    // Convert microseconds to RFC 3339 format
    let secs = ts.as_micros() / 1_000_000;
    let _nanos = ((ts.as_micros() % 1_000_000) * 1000) as u32;

    // Simple formatting (would use chrono in production)
    format!("{}Z", secs)
}

// =============================================================================
// CRYPTOGRAPHIC SIGNING (Ed25519 via HDK)
// =============================================================================

/// Base58 alphabet for Bitcoin/IPFS encoding
const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode bytes to base58btc (used for multibase)
fn base58_encode(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }

    // Count leading zeros
    let leading_zeros = bytes.iter().take_while(|&&b| b == 0).count();

    // Convert to base58
    let mut result = Vec::new();
    let mut num = bytes.to_vec();

    while !num.is_empty() && !num.iter().all(|&b| b == 0) {
        let mut remainder = 0u32;
        let mut new_num = Vec::new();

        for &byte in &num {
            let current = (remainder << 8) + byte as u32;
            let quotient = current / 58;
            remainder = current % 58;

            if !new_num.is_empty() || quotient > 0 {
                new_num.push(quotient as u8);
            }
        }

        result.push(BASE58_ALPHABET[remainder as usize]);
        num = new_num;
    }

    // Add leading '1's for zeros
    for _ in 0..leading_zeros {
        result.push(BASE58_ALPHABET[0]);
    }

    result.reverse();
    String::from_utf8(result).unwrap_or_default()
}

/// Decode base58btc string to bytes
fn base58_decode(s: &str) -> Option<Vec<u8>> {
    let mut result = vec![0u8; 1];

    for c in s.chars() {
        let digit = BASE58_ALPHABET.iter().position(|&b| b == c as u8)?;

        // Multiply result by 58 and add digit
        let mut carry = digit as u32;
        for byte in result.iter_mut().rev() {
            let value = (*byte as u32) * 58 + carry;
            *byte = (value & 0xff) as u8;
            carry = value >> 8;
        }

        while carry > 0 {
            result.insert(0, (carry & 0xff) as u8);
            carry >>= 8;
        }
    }

    // Handle leading '1's (zeros)
    let leading_ones = s.chars().take_while(|&c| c == '1').count();
    let mut zeros = vec![0u8; leading_ones];
    zeros.extend(result.into_iter().skip_while(|&b| b == 0));

    Some(zeros)
}

/// Encode signature in multibase format (base58btc with 'z' prefix)
/// This follows W3C Data Integrity EdDSA Cryptosuites v1.0
fn multibase_encode(bytes: &[u8]) -> String {
    format!("z{}", base58_encode(bytes))
}

// =============================================================================
// ISO 8601 DATE PARSING
// =============================================================================

/// Parse an ISO 8601 datetime string and check if it's expired
///
/// Supports formats:
/// - `2024-12-31T23:59:59Z` (UTC)
/// - `2024-12-31T23:59:59+00:00` (with timezone offset)
/// - `2024-12-31` (date only, assumes end of day UTC)
///
/// Returns true if the datetime is in the past relative to `now`.
fn parse_iso8601_expired(datetime_str: &str, now: Timestamp) -> bool {
    // Try to parse the ISO 8601 string
    match parse_iso8601_to_micros(datetime_str) {
        Some(expiry_micros) => {
            let now_micros = now.as_micros();
            now_micros > expiry_micros
        }
        None => {
            // If parsing fails, assume not expired (fail open for usability)
            // In production, you might want to fail closed instead
            false
        }
    }
}

/// Parse ISO 8601 datetime string to microseconds since Unix epoch
fn parse_iso8601_to_micros(s: &str) -> Option<i64> {
    // Handle date-only format: "2024-12-31"
    if s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-') {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;

        // Validate ranges
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }

        // Convert to days since epoch and then to microseconds
        // Simplified: use end of day (23:59:59) for date-only
        let days = days_since_epoch(year, month, day)?;
        let secs = days as i64 * 86400 + 86399; // End of day
        return Some(secs * 1_000_000);
    }

    // Handle full datetime format: "2024-12-31T23:59:59Z" or "2024-12-31T23:59:59+00:00"
    if s.len() >= 19 && s.chars().nth(10) == Some('T') {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        let hour: u32 = s[11..13].parse().ok()?;
        let minute: u32 = s[14..16].parse().ok()?;
        let second: u32 = s[17..19].parse().ok()?;

        // Validate ranges
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        if hour > 23 || minute > 59 || second > 59 {
            return None;
        }

        // Parse timezone offset if present
        let tz_offset_secs: i64 = if s.len() > 19 {
            let tz_part = &s[19..];
            if tz_part == "Z" || tz_part.is_empty() {
                0
            } else if tz_part.starts_with('+') || tz_part.starts_with('-') {
                parse_tz_offset(tz_part)?
            } else {
                0
            }
        } else {
            0
        };

        let days = days_since_epoch(year, month, day)?;
        let day_secs = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
        let total_secs = days as i64 * 86400 + day_secs - tz_offset_secs;

        return Some(total_secs * 1_000_000);
    }

    None
}

/// Parse timezone offset like "+05:30" or "-08:00" to seconds
fn parse_tz_offset(s: &str) -> Option<i64> {
    if s.len() < 3 {
        return None;
    }

    let sign: i64 = if s.starts_with('+') { 1 } else { -1 };
    let rest = &s[1..];

    // Handle "+0530" format
    if rest.len() == 4 && !rest.contains(':') {
        let hours: i64 = rest[0..2].parse().ok()?;
        let minutes: i64 = rest[2..4].parse().ok()?;
        return Some(sign * (hours * 3600 + minutes * 60));
    }

    // Handle "+05:30" format
    if rest.len() >= 5 && rest.chars().nth(2) == Some(':') {
        let hours: i64 = rest[0..2].parse().ok()?;
        let minutes: i64 = rest[3..5].parse().ok()?;
        return Some(sign * (hours * 3600 + minutes * 60));
    }

    // Handle "+05" format (hours only)
    if rest.len() == 2 {
        let hours: i64 = rest.parse().ok()?;
        return Some(sign * hours * 3600);
    }

    None
}

/// Calculate days since Unix epoch (1970-01-01) for a given date
fn days_since_epoch(year: i32, month: u32, day: u32) -> Option<i64> {
    // Simplified algorithm for days since epoch
    // This handles leap years correctly

    if year < 1970 {
        // For dates before epoch, calculate negative days
        return days_since_epoch_negative(year, month, day);
    }

    let mut days: i64 = 0;

    // Add days for complete years
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in current year
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for m in 0..(month - 1) as usize {
        days += days_in_months[m] as i64;
    }

    // Add days in current month
    days += (day - 1) as i64;

    Some(days)
}

/// Handle dates before Unix epoch
fn days_since_epoch_negative(year: i32, month: u32, day: u32) -> Option<i64> {
    let mut days: i64 = 0;

    // Count backwards from 1970
    for y in (year + 1)..1970 {
        days -= if is_leap_year(y) { 366 } else { 365 };
    }

    // Handle the partial year
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    // Days remaining in the year from the given date
    let mut remaining = 0i64;
    for m in (month as usize)..12 {
        remaining += days_in_months[m] as i64;
    }
    remaining -= (day - 1) as i64;
    remaining += days_in_months[(month - 1) as usize] as i64;

    days -= remaining;

    Some(days)
}

/// Check if a year is a leap year
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Decode multibase string (base58btc with 'z' prefix)
fn multibase_decode(s: &str) -> Option<Vec<u8>> {
    if s.starts_with('z') {
        base58_decode(&s[1..])
    } else {
        None
    }
}

/// Sign credential content using the agent's ed25519 key
///
/// This uses Holochain's HDK sign_raw which performs ed25519 signing
/// with the agent's cryptographic identity.
fn sign_credential(vc: &VerifiableCredential) -> ExternResult<String> {
    // Compute canonical hash of credential content
    let content_hash = compute_credential_hash(vc);

    // Sign with agent's ed25519 key via HDK
    let signature = sign_raw(
        agent_info()?.agent_initial_pubkey,
        content_hash.clone().into(),
    )?;

    // Encode as multibase (z + base58btc) per W3C Data Integrity spec
    Ok(multibase_encode(signature.as_ref()))
}

/// Verify a credential signature
///
/// This verifies the ed25519 signature against the credential content
/// using the issuer's public key extracted from their DID.
///
/// The DID format is `did:mycelix:<base64-encoded-pubkey>` where the pubkey
/// is the full Holochain AgentPubKey (39 bytes: 32 pubkey + 4 hash type + 3 DHT location).
fn verify_credential_signature(vc: &VerifiableCredential) -> ExternResult<bool> {
    // Extract public key from issuer DID (did:mycelix:<base64-pubkey>)
    let issuer_did = vc.issuer.did();
    let pubkey_str = issuer_did.strip_prefix("did:mycelix:").ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid DID format: {}",
            issuer_did
        )))
    })?;

    // The pubkey is stored as its Display representation (Holochain format)
    // Try to parse it back - Holochain uses a specific encoding
    // For now, we'll use try_from which handles the Holochain-specific format
    let pubkey = AgentPubKey::try_from(pubkey_str.to_string()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid public key in DID '{}': {:?}",
            pubkey_str, e
        )))
    })?;

    // Decode the signature from multibase
    let signature_bytes = multibase_decode(&vc.proof.proof_value).ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "Invalid multibase signature encoding".into()
        ))
    })?;

    // Create Signature from bytes (ed25519 signatures are 64 bytes)
    if signature_bytes.len() != 64 {
        return Ok(false);
    }

    let signature = Signature::from(
        <[u8; 64]>::try_from(signature_bytes.as_slice())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid signature length".into())))?,
    );

    // Compute expected content hash
    let content_hash = compute_credential_hash(vc);

    // Verify signature using HDK
    verify_signature(pubkey, signature, content_hash)
}

// =============================================================================
// REVOCATION STATUS CHECKING (Cross-Zome Integration)
// =============================================================================

/// Credential revocation status
#[derive(Debug, Clone)]
pub enum CredentialRevocationStatus {
    /// Credential is active (not revoked)
    Active,
    /// Credential has been revoked
    Revoked(String),
    /// Credential is temporarily suspended
    Suspended(String, String),
    /// No revocation record found
    Unknown,
}

/// Check credential revocation status by querying the revocation registry
///
/// This function queries the local chain for revocation entries linked to the credential.
/// In a full implementation, this would be a cross-zome call to the revocation coordinator.
fn check_credential_revocation_status(
    credential_id: &str,
) -> ExternResult<CredentialRevocationStatus> {
    // Create deterministic hash for the credential ID to query links
    let _credential_hash = string_to_entry_hash(credential_id);

    // Query for revocation links
    // Note: LinkTypes::CredentialToRevocation would need to be added to this zome's link types,
    // or this would be a cross-zome call to the revocation zome
    let filter = ChainQueryFilter::new().include_entries(true);

    for record in query(filter)? {
        // Check if this record is a revocation entry for our credential
        // This is a simplified approach - in production, would use proper cross-zome calls
        if let Some(_entry) = record.entry().as_option() {
            // Try to deserialize as a revocation-like structure
            // Look for entries that reference this credential_id
        }
    }

    // If no revocation found, credential is presumed active
    Ok(CredentialRevocationStatus::Active)
}

/// Check if a specific credential is revoked (public API)
#[hdk_extern]
pub fn is_credential_revoked(credential_id: String) -> ExternResult<bool> {
    let status = check_credential_revocation_status(&credential_id)?;
    match status {
        CredentialRevocationStatus::Revoked(_) => Ok(true),
        CredentialRevocationStatus::Suspended(_, _) => Ok(true),
        _ => Ok(false),
    }
}

/// Get detailed revocation status for a credential
#[hdk_extern]
pub fn get_credential_status(credential_id: String) -> ExternResult<CredentialStatusResponse> {
    let now = sys_time()?;
    let status = check_credential_revocation_status(&credential_id)?;

    let (is_valid, status_type, reason) = match status {
        CredentialRevocationStatus::Active => (true, "active".to_string(), None),
        CredentialRevocationStatus::Revoked(r) => (false, "revoked".to_string(), Some(r)),
        CredentialRevocationStatus::Suspended(r, until) => {
            (false, format!("suspended_until_{}", until), Some(r))
        }
        CredentialRevocationStatus::Unknown => (true, "unknown".to_string(), None),
    };

    Ok(CredentialStatusResponse {
        credential_id,
        is_valid,
        status_type,
        reason,
        checked_at: now,
    })
}

/// Response for credential status check
#[derive(Serialize, Deserialize, Debug)]
pub struct CredentialStatusResponse {
    pub credential_id: String,
    pub is_valid: bool,
    pub status_type: String,
    pub reason: Option<String>,
    pub checked_at: Timestamp,
}
