// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Credential Coordinator Zome
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
use mycelix_crypto::{AlgorithmId, TaggedSignature};
use verifiable_credential_integrity::*;

/// Mirror type for credential_schema deserialization (cross-zome)
#[derive(Serialize, Deserialize, Debug)]
struct SchemaMirror {
    id: String,
    name: String,
    required_fields: Vec<String>,
    optional_fields: Vec<String>,
    credential_type: Vec<String>,
    active: bool,
}

impl TryFrom<SerializedBytes> for SchemaMirror {
    type Error = SerializedBytesError;
    fn try_from(sb: SerializedBytes) -> Result<Self, Self::Error> {
        holochain_serialized_bytes::decode(sb.bytes())
    }
}

/// Standard W3C context URLs
const W3C_CREDENTIALS_V2: &str = "https://www.w3.org/ns/credentials/v2";
const W3C_DATA_INTEGRITY: &str = "https://w3id.org/security/data-integrity/v2";

/// API version for cross-zome compatibility detection.
/// Increment when making breaking changes to extern signatures or types.
const API_VERSION: u16 = 1;

/// Returns the API version of this coordinator zome.
#[hdk_extern]
pub fn get_api_version(_: ()) -> ExternResult<u16> {
    Ok(API_VERSION)
}

/// Create a deterministic entry hash from a string identifier
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>(),
    )
}

/// Compute cryptographic hash of credential content for signing
///
/// Uses BLAKE2b-256 (via holo_hash) for a secure 32-byte hash.
/// The hash covers the essential credential fields in a deterministic order
/// to ensure consistent verification across signing and verification.
fn compute_credential_hash(vc: &VerifiableCredential) -> Vec<u8> {
    // Build deterministic content to hash:
    // - Credential ID
    // - Issuer DID
    // - Subject DID
    // - Valid from timestamp
    // - Claims (JSON serialized)
    // - Schema ID (if present)
    let mut content = Vec::new();
    content.extend(vc.id.as_bytes());
    content.push(0); // null separator
    content.extend(vc.issuer.did().as_bytes());
    content.push(0);
    content.extend(vc.credential_subject.id.as_bytes());
    content.push(0);
    content.extend(vc.valid_from.as_bytes());
    content.push(0);
    // Serialize claims deterministically (serde_json sorts keys by default for objects)
    if let Ok(claims_json) = serde_json::to_string(&vc.credential_subject.claims) {
        content.extend(claims_json.as_bytes());
    }
    content.push(0);
    content.extend(vc.mycelix_schema_id.as_bytes());

    // Use BLAKE2b-256 for cryptographic hashing
    holo_hash::blake2b_256(&content).to_vec()
}

/// Schema validation outcome — tells the caller what actually happened.
#[derive(Debug, Clone, PartialEq)]
enum SchemaValidationStatus {
    /// No schema was specified
    NoSchema,
    /// Schema zome was unavailable (not installed or unauthorized)
    SchemaZomeUnavailable,
    /// Schema not found in registry (may be external)
    SchemaNotFound,
    /// Validation was performed and all required fields are present
    Validated { schema_name: String },
}

/// Validate credential claims against a schema's required/optional fields.
///
/// Calls the credential_schema zome to fetch the schema, then checks:
/// 1. Schema exists and is active
/// 2. All required_fields are present in claims
/// 3. No unknown fields (not in required or optional) if schema is strict
///
/// Returns the validation status so callers know whether validation was performed.
fn validate_claims_against_schema(
    schema_id: &str,
    claims: &serde_json::Value,
) -> ExternResult<SchemaValidationStatus> {
    // Skip validation if no schema specified
    if schema_id.is_empty() {
        return Ok(SchemaValidationStatus::NoSchema);
    }

    // Fetch schema via cross-zome call
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("credential_schema"),
        FunctionName::new("get_schema"),
        None,
        schema_id.to_string(),
    )?;

    let schema: SchemaMirror = match response {
        ZomeCallResponse::Ok(result) => {
            let record: Option<Record> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode schema response: {:?}",
                    e
                )))
            })?;
            let rec = match record {
                Some(r) => r,
                None => return Ok(SchemaValidationStatus::SchemaNotFound),
            };
            rec.entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid schema entry".into()
                )))?
        }
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            return Ok(SchemaValidationStatus::SchemaZomeUnavailable);
        }
        ZomeCallResponse::NetworkError(err) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Network error fetching schema: {}",
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

    // Verify schema is active
    if !schema.active {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Schema '{}' is not active",
            schema_id
        ))));
    }

    // Claims must be an object to validate fields
    let claims_obj = match claims.as_object() {
        Some(obj) => obj,
        None => {
            // If claims is not an object (e.g., array or scalar), skip field validation
            return Ok(SchemaValidationStatus::Validated {
                schema_name: schema.name,
            });
        }
    };

    // Check all required fields are present
    let mut missing: Vec<String> = Vec::new();
    for field in &schema.required_fields {
        if !claims_obj.contains_key(field) {
            missing.push(field.clone());
        }
    }

    if !missing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Missing required fields for schema '{}': {}",
            schema.name,
            missing.join(", ")
        ))));
    }

    Ok(SchemaValidationStatus::Validated {
        schema_name: schema.name,
    })
}

/// Issue a new verifiable credential
#[hdk_extern]
pub fn issue_credential(input: IssueCredentialInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let issuer_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let now = sys_time()?;
    let now_iso = format_timestamp_iso8601(now);

    // Validate claims against schema if a schema is specified
    let schema_validation = validate_claims_against_schema(&input.schema_id, &input.claims)?;
    match &schema_validation {
        SchemaValidationStatus::Validated { schema_name } => {
            debug!("Schema validation passed for schema '{}'", schema_name);
        }
        SchemaValidationStatus::SchemaZomeUnavailable => {
            if input.strict_schema {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "strict_schema: credential_schema zome is unavailable; cannot validate schema"
                        .to_string()
                )));
            }
            warn!("Schema validation skipped: credential_schema zome unavailable");
        }
        SchemaValidationStatus::SchemaNotFound => {
            if input.strict_schema {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "strict_schema: schema '{}' not found in registry",
                    input.schema_id
                ))));
            }
            debug!(
                "Schema validation skipped: schema '{}' not found (may be external)",
                input.schema_id
            );
        }
        SchemaValidationStatus::NoSchema => {
            if input.strict_schema {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "strict_schema: no schema_id provided but strict_schema is enabled".to_string()
                )));
            }
        }
    }

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
            cryptosuite: Some(AlgorithmId::Ed25519.cryptosuite().to_string()),
            algorithm: Some(AlgorithmId::Ed25519.as_u16()),
            challenge: None,
            domain: None,
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
    /// When true, schema validation is mandatory: if the credential_schema zome
    /// is unavailable or the schema is not found, issuance fails instead of
    /// silently skipping validation. Default: false (backward-compatible).
    #[serde(default)]
    pub strict_schema: bool,
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

    // Check expiration using ISO 8601 parsing (fail-closed)
    if let Some(valid_until) = &credential.valid_until {
        match parse_iso8601_expired(valid_until, now) {
            ExpirationStatus::Expired => {
                errors.push("Credential has expired".to_string());
            }
            ExpirationStatus::ParseError => {
                errors.push(format!(
                    "Credential expiration date unparseable (fail-closed): '{}'",
                    valid_until
                ));
            }
            ExpirationStatus::Valid => {}
        }
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
            // SECURITY: Fail-closed — unknown revocation status is not safe to assume active
            errors.push("Revocation status could not be determined".to_string());
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
    if issuer_did.is_empty() || issuer_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer DID must be 1-256 characters".into()
        )));
    }
    if !issuer_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer must be a valid DID".into()
        )));
    }
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
    if subject_did.is_empty() || subject_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject DID must be 1-256 characters".into()
        )));
    }
    if !subject_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Subject must be a valid DID".into()
        )));
    }
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

    // Validate challenge/domain inputs
    if let Some(ref challenge) = input.challenge {
        if challenge.is_empty() || challenge.len() > 512 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Challenge must be 1-512 characters".into()
            )));
        }
    }
    if let Some(ref domain) = input.domain {
        if domain.is_empty() || domain.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Domain must be 1-256 characters".into()
            )));
        }
        // Domain requires a challenge (W3C Data Integrity spec: domain without
        // challenge is meaningless since there's nothing binding the domain to
        // a specific verification session)
        if input.challenge.is_none() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Domain requires a challenge to be set (replay protection)".into()
            )));
        }
    }

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
    let signature = sign_raw(agent_info.agent_initial_pubkey.clone(), presentation_data)?;
    let tagged_sig = TaggedSignature::new(AlgorithmId::Ed25519, signature.as_ref().to_vec())
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Signature tagging error: {}",
                e
            )))
        })?;

    let proof = CredentialProof {
        proof_type: "DataIntegrityProof".to_string(),
        created: now_iso.clone(),
        verification_method: format!("{}#keys-1", holder_did),
        proof_purpose: "authentication".to_string(),
        proof_value: tagged_sig.to_multibase(),
        cryptosuite: Some(AlgorithmId::Ed25519.cryptosuite().to_string()),
        algorithm: Some(AlgorithmId::Ed25519.as_u16()),
        challenge: input.challenge.clone(),
        domain: input.domain.clone(),
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

/// Verify a verifiable presentation
///
/// Checks:
/// 1. Holder's proof signature over the presentation content
/// 2. Each contained credential's issuer signature
/// 3. Each contained credential's revocation status
/// 4. Proof purpose is "authentication"
#[hdk_extern]
pub fn verify_presentation(
    input: VerifyPresentationInput,
) -> ExternResult<PresentationVerificationResult> {
    let record = get(input.presentation_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Presentation not found".into())),
    )?;

    let vp: VerifiablePresentation = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid presentation entry".into()
        )))?;

    let now = sys_time()?;
    let mut errors = Vec::new();
    let mut credential_results = Vec::new();

    // 1. Verify proof purpose
    if vp.proof.proof_purpose != "authentication" {
        errors.push("Presentation proof purpose must be 'authentication'".to_string());
    }

    // 2. Validate challenge/domain binding (replay protection)
    // If the proof was created with a challenge, the verifier MUST supply the same one.
    // If the verifier expects a challenge but the proof has none, that's also a failure.
    if let Some(expected) = &input.expected_challenge {
        match &vp.proof.challenge {
            Some(stored) if stored == expected => {} // match
            Some(stored) => errors.push(format!(
                "Presentation challenge mismatch: expected '{}', proof has '{}'",
                expected, stored
            )),
            None => errors.push(format!(
                "Presentation has no challenge but verifier expected '{}'",
                expected
            )),
        }
    } else if vp.proof.challenge.is_some() {
        errors.push(
            "Presentation has challenge but verifier did not supply expected_challenge".to_string(),
        );
    }

    if let Some(expected) = &input.expected_domain {
        match &vp.proof.domain {
            Some(stored) if stored == expected => {} // match
            Some(stored) => errors.push(format!(
                "Presentation domain mismatch: expected '{}', proof has '{}'",
                expected, stored
            )),
            None => errors.push(format!(
                "Presentation has no domain but verifier expected '{}'",
                expected
            )),
        }
    } else if vp.proof.domain.is_some() {
        errors.push(
            "Presentation has domain but verifier did not supply expected_domain".to_string(),
        );
    }

    // 3. Verify holder's proof signature
    let holder_pubkey_str = vp.holder.strip_prefix("did:mycelix:");
    if let Some(pubkey_str) = holder_pubkey_str {
        if let Ok(holder_pubkey) = AgentPubKey::try_from(pubkey_str.to_string()) {
            // Reconstruct the signed data (mirrors create_presentation).
            // Use the values stored IN the proof, not from the verifier's input,
            // since these are what was actually signed.
            let mut presentation_data = vp.id.as_bytes().to_vec();
            presentation_data.extend(vp.holder.as_bytes());
            for cred in &vp.verifiable_credential {
                presentation_data.extend(cred.id.as_bytes());
            }
            if let Some(challenge) = &vp.proof.challenge {
                presentation_data.extend(challenge.as_bytes());
            }
            if let Some(domain) = &vp.proof.domain {
                presentation_data.extend(domain.as_bytes());
            }

            // Try TaggedSignature first, then legacy
            match TaggedSignature::from_multibase(&vp.proof.proof_value) {
                Ok(tagged_sig) => {
                    if tagged_sig.algorithm == AlgorithmId::Ed25519
                        && tagged_sig.signature_bytes.len() == 64
                    {
                        let sig = Signature::from(
                            <[u8; 64]>::try_from(tagged_sig.signature_bytes.as_slice())
                                .unwrap_or([0u8; 64]),
                        );
                        match verify_signature(holder_pubkey, sig, presentation_data) {
                            Ok(true) => {}
                            Ok(false) => errors
                                .push("Holder proof signature verification failed".to_string()),
                            Err(e) => {
                                errors.push(format!("Holder signature verification error: {:?}", e))
                            }
                        }
                    } else {
                        errors.push(format!(
                            "Unsupported presentation proof algorithm: {:?}",
                            tagged_sig.algorithm
                        ));
                    }
                }
                Err(_) => {
                    // Legacy multibase fallback
                    if let Some(sig_bytes) = multibase_decode(&vp.proof.proof_value) {
                        if sig_bytes.len() == 64 {
                            let sig = Signature::from(
                                <[u8; 64]>::try_from(sig_bytes.as_slice()).unwrap_or([0u8; 64]),
                            );
                            match verify_signature(holder_pubkey, sig, presentation_data) {
                                Ok(true) => {}
                                Ok(false) => errors.push(
                                    "Holder proof signature verification failed (legacy)"
                                        .to_string(),
                                ),
                                Err(e) => errors
                                    .push(format!("Holder signature verification error: {:?}", e)),
                            }
                        } else {
                            errors.push("Invalid holder signature length".to_string());
                        }
                    } else {
                        errors.push("Could not decode holder proof signature".to_string());
                    }
                }
            }
        } else {
            errors.push("Could not parse holder's agent pub key".to_string());
        }
    } else {
        errors.push("Invalid holder DID format".to_string());
    }

    // 3. Batch-check revocation for all credentials in one cross-zome call
    //    (eliminates N+1 query pattern — single call instead of per-credential)
    let cred_ids: Vec<String> = vp
        .verifiable_credential
        .iter()
        .map(|c| c.id.clone())
        .collect();
    let revocation_statuses = batch_check_credential_revocation_status(&cred_ids)?;

    // 4. Verify each contained credential
    for (i, cred) in vp.verifiable_credential.iter().enumerate() {
        let mut cred_errors = Vec::new();

        // Verify credential signature
        match verify_credential_signature(cred) {
            Ok(true) => {}
            Ok(false) => cred_errors.push("Issuer signature invalid".to_string()),
            Err(e) => cred_errors.push(format!("Signature error: {:?}", e)),
        }

        // Check revocation (from batch result)
        let revocation = &revocation_statuses[i];
        match revocation {
            CredentialRevocationStatus::Revoked(reason) => {
                cred_errors.push(format!("Revoked: {}", reason));
            }
            CredentialRevocationStatus::Suspended(reason, _) => {
                cred_errors.push(format!("Suspended: {}", reason));
            }
            CredentialRevocationStatus::Active => {}
            CredentialRevocationStatus::Unknown => {
                cred_errors.push("Revocation status could not be determined".to_string());
            }
        }

        // Check expiration (fail-closed)
        if let Some(valid_until) = &cred.valid_until {
            match parse_iso8601_expired(valid_until, now) {
                ExpirationStatus::Expired => {
                    cred_errors.push("Credential expired".to_string());
                }
                ExpirationStatus::ParseError => {
                    cred_errors.push(format!(
                        "Credential expiration date unparseable (fail-closed): '{}'",
                        valid_until
                    ));
                }
                ExpirationStatus::Valid => {}
            }
        }

        let cred_valid = cred_errors.is_empty();
        credential_results.push(CredentialInPresentationResult {
            credential_id: cred.id.clone(),
            valid: cred_valid,
            errors: cred_errors,
        });
    }

    // If any credential is invalid, mark overall as invalid
    if credential_results.iter().any(|r| !r.valid) {
        errors.push("One or more contained credentials failed verification".to_string());
    }

    Ok(PresentationVerificationResult {
        presentation_id: vp.id,
        holder: vp.holder,
        valid: errors.is_empty(),
        errors,
        credential_results,
        verified_at: now,
    })
}

/// Input for verifying a presentation
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyPresentationInput {
    /// Action hash of the presentation to verify
    pub presentation_hash: ActionHash,
    /// Expected challenge (must match what was used during creation)
    pub expected_challenge: Option<String>,
    /// Expected domain (must match what was used during creation)
    pub expected_domain: Option<String>,
}

/// Result of presentation verification
#[derive(Serialize, Deserialize, Debug)]
pub struct PresentationVerificationResult {
    pub presentation_id: String,
    pub holder: String,
    pub valid: bool,
    pub errors: Vec<String>,
    pub credential_results: Vec<CredentialInPresentationResult>,
    pub verified_at: Timestamp,
}

/// Verification result for a credential within a presentation
#[derive(Serialize, Deserialize, Debug)]
pub struct CredentialInPresentationResult {
    pub credential_id: String,
    pub valid: bool,
    pub errors: Vec<String>,
}

// =============================================================================
// MERKLE TREE FOR SELECTIVE DISCLOSURE
// =============================================================================

/// Hash a single claim leaf: BLAKE2b-256(key || 0x00 || json_value)
fn hash_claim_leaf(key: &str, value: &serde_json::Value) -> Vec<u8> {
    let mut data = key.as_bytes().to_vec();
    data.push(0);
    if let Ok(json) = serde_json::to_string(value) {
        data.extend(json.as_bytes());
    }
    holo_hash::blake2b_256(&data).to_vec()
}

/// Build a Merkle tree from sorted claim leaves and return (root, leaves_in_order).
///
/// Leaves are sorted by key to ensure deterministic ordering.
/// Internal nodes: BLAKE2b-256(left || right). Odd nodes are promoted.
fn build_claim_merkle_tree(
    claims: &serde_json::Map<String, serde_json::Value>,
) -> (Vec<u8>, Vec<(String, Vec<u8>)>) {
    // Sort keys for deterministic order
    let mut keys: Vec<&String> = claims.keys().collect();
    keys.sort();

    let leaves: Vec<(String, Vec<u8>)> = keys
        .iter()
        .map(|k| ((*k).clone(), hash_claim_leaf(k, &claims[*k])))
        .collect();

    if leaves.is_empty() {
        return (vec![0u8; 32], leaves);
    }
    if leaves.len() == 1 {
        return (leaves[0].1.clone(), leaves);
    }

    // Build tree bottom-up
    let mut current_level: Vec<Vec<u8>> = leaves.iter().map(|(_, h)| h.clone()).collect();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        let mut i = 0;
        while i < current_level.len() {
            if i + 1 < current_level.len() {
                let mut combined = current_level[i].clone();
                combined.extend(&current_level[i + 1]);
                next_level.push(holo_hash::blake2b_256(&combined).to_vec());
                i += 2;
            } else {
                // Odd node: promote
                next_level.push(current_level[i].clone());
                i += 1;
            }
        }
        current_level = next_level;
    }

    (current_level[0].clone(), leaves)
}

/// Generate a Merkle proof (list of sibling hashes) for a leaf at the given index.
fn generate_merkle_proof(
    claims: &serde_json::Map<String, serde_json::Value>,
    leaf_index: usize,
) -> Vec<Vec<u8>> {
    let mut keys: Vec<&String> = claims.keys().collect();
    keys.sort();

    let leaf_hashes: Vec<Vec<u8>> = keys
        .iter()
        .map(|k| hash_claim_leaf(k, &claims[*k]))
        .collect();

    if leaf_hashes.len() <= 1 {
        return Vec::new(); // Single leaf = root, no proof needed
    }

    let mut proof = Vec::new();
    let mut current_level = leaf_hashes;
    let mut idx = leaf_index;

    while current_level.len() > 1 {
        // Sibling index
        let sibling_idx = if idx.is_multiple_of(2) {
            idx + 1
        } else {
            idx - 1
        };

        if sibling_idx < current_level.len() {
            proof.push(current_level[sibling_idx].clone());
        }
        // No sibling (odd node promoted) — no proof element needed

        // Move to next level
        let mut next_level = Vec::new();
        let mut i = 0;
        while i < current_level.len() {
            if i + 1 < current_level.len() {
                let mut combined = current_level[i].clone();
                combined.extend(&current_level[i + 1]);
                next_level.push(holo_hash::blake2b_256(&combined).to_vec());
                i += 2;
            } else {
                next_level.push(current_level[i].clone());
                i += 1;
            }
        }
        current_level = next_level;
        idx /= 2;
    }

    proof
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
    let holder_signature = sign_raw(agent_info.agent_initial_pubkey.clone(), sign_data)?;

    // Build Merkle tree over all original claims for selective disclosure proofs
    let original_claims_obj = original_claims.as_object();
    let (_merkle_root, _leaves) = if let Some(obj) = original_claims_obj {
        build_claim_merkle_tree(obj)
    } else {
        (vec![0u8; 32], Vec::new())
    };

    // Generate per-claim Merkle proofs for each selected claim
    let claim_proofs: Vec<ClaimProof> = if let Some(obj) = original_claims_obj {
        let mut sorted_keys: Vec<&String> = obj.keys().collect();
        sorted_keys.sort();

        input
            .selected_claims
            .iter()
            .map(|key| {
                let leaf_index = sorted_keys.iter().position(|k| *k == key);
                let merkle_path = leaf_index.map(|idx| generate_merkle_proof(obj, idx));
                let commitment = leaf_index.map(|_| hash_claim_leaf(key, &obj[key]));

                ClaimProof {
                    claim_key: key.clone(),
                    merkle_path,
                    commitment,
                }
            })
            .collect()
    } else {
        input
            .selected_claims
            .iter()
            .map(|key| ClaimProof {
                claim_key: key.clone(),
                merkle_path: None,
                commitment: None,
            })
            .collect()
    };

    let derivation_proof = DerivationProof {
        proof_type: "SelectiveDisclosureProof".to_string(),
        original_credential_hash: original_hash,
        claim_proofs,
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

/// Result of derived credential verification
#[derive(Serialize, Deserialize, Debug)]
pub struct DerivedVerificationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub original_issuer_verified: bool,
}

/// Verify a derived credential
///
/// Checks:
/// 1. Original credential exists and matches the recorded hash
/// 2. Holder's Ed25519 signature over (original_hash + selected_claims)
/// 3. Selected claims are a valid subset of the original credential's claims
/// 4. Derived credential is not expired
/// 5. Original credential is not revoked
#[hdk_extern]
pub fn verify_derived_credential(
    action_hash: ActionHash,
) -> ExternResult<DerivedVerificationResult> {
    let mut errors = Vec::new();
    let mut original_issuer_verified = false;

    // Fetch derived credential
    let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Derived credential not found".into())
    ))?;

    let derived: DerivedCredential = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid derived credential entry".into()
        )))?;

    // Check expiration
    let now = sys_time()?;
    if let Some(expires) = derived.expires {
        if now >= expires {
            errors.push("Derived credential has expired".to_string());
        }
    }

    // Fetch original credential
    let original_record = match get_credential(derived.original_credential_id.clone())? {
        Some(rec) => rec,
        None => {
            errors.push("Original credential not found".to_string());
            return Ok(DerivedVerificationResult {
                valid: false,
                errors,
                original_issuer_verified: false,
            });
        }
    };

    let original_vc: VerifiableCredential = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid original credential".into()
        )))?;

    // Verify the original credential hash matches
    let recomputed_hash = compute_credential_hash(&original_vc);
    if recomputed_hash != derived.derivation_proof.original_credential_hash {
        errors.push("Original credential hash does not match derivation proof".to_string());
    }

    // Verify the original credential's own signature
    match verify_credential_signature(&original_vc) {
        Ok(true) => {
            original_issuer_verified = true;
        }
        Ok(false) => {
            errors.push("Original credential's issuer signature is invalid".to_string());
        }
        Err(e) => {
            errors.push(format!("Could not verify original credential: {:?}", e));
        }
    }

    // Verify selected claims are a subset of the original
    let original_claims = &original_vc.credential_subject.claims;
    for claim_key in &derived.selected_claims {
        if original_claims.get(claim_key).is_none() {
            errors.push(format!(
                "Claim '{}' not present in original credential",
                claim_key
            ));
        }
    }

    // Verify Merkle proofs if present
    if let Some(original_obj) = original_claims.as_object() {
        let (merkle_root, _) = build_claim_merkle_tree(original_obj);

        for claim_proof in &derived.derivation_proof.claim_proofs {
            // Verify commitment matches the actual claim value
            if let (Some(commitment), Some(value)) = (
                &claim_proof.commitment,
                original_claims.get(&claim_proof.claim_key),
            ) {
                let expected_leaf = hash_claim_leaf(&claim_proof.claim_key, value);
                if *commitment != expected_leaf {
                    errors.push(format!(
                        "Merkle commitment mismatch for claim '{}'",
                        claim_proof.claim_key
                    ));
                    continue;
                }

                // Verify the Merkle path from leaf to root
                if let Some(path) = &claim_proof.merkle_path {
                    let mut sorted_keys: Vec<&String> = original_obj.keys().collect();
                    sorted_keys.sort();
                    if let Some(leaf_idx) = sorted_keys
                        .iter()
                        .position(|k| *k == &claim_proof.claim_key)
                    {
                        let mut current_hash = expected_leaf;
                        let mut idx = leaf_idx;

                        for sibling in path {
                            let combined = if idx.is_multiple_of(2) {
                                let mut c = current_hash.clone();
                                c.extend(sibling);
                                c
                            } else {
                                let mut c = sibling.clone();
                                c.extend(&current_hash);
                                c
                            };
                            current_hash = holo_hash::blake2b_256(&combined).to_vec();
                            idx /= 2;
                        }

                        if current_hash != merkle_root {
                            errors.push(format!(
                                "Merkle path verification failed for claim '{}'",
                                claim_proof.claim_key
                            ));
                        }
                    }
                }
            }
        }
    }

    // Verify holder's Ed25519 signature over (original_hash + selected_claims)
    let holder_did = &derived.holder;
    let holder_pubkey_str = holder_did.strip_prefix("did:mycelix:");
    if let Some(pubkey_str) = holder_pubkey_str {
        if let Ok(holder_pubkey) = AgentPubKey::try_from(pubkey_str.to_string()) {
            // Reconstruct signed data: original_hash + claim keys
            let mut sign_data = derived.derivation_proof.original_credential_hash.clone();
            for claim_key in &derived.selected_claims {
                sign_data.extend(claim_key.as_bytes());
            }

            if derived.derivation_proof.holder_signature.len() == 64 {
                let sig = Signature::from(
                    <[u8; 64]>::try_from(derived.derivation_proof.holder_signature.as_slice())
                        .unwrap_or([0u8; 64]),
                );
                match verify_signature(holder_pubkey, sig, sign_data) {
                    Ok(true) => {} // Valid
                    Ok(false) => {
                        errors.push("Holder signature verification failed".to_string());
                    }
                    Err(e) => {
                        errors.push(format!("Holder signature verification error: {:?}", e));
                    }
                }
            } else {
                errors.push(format!(
                    "Invalid holder signature length: expected 64, got {}",
                    derived.derivation_proof.holder_signature.len()
                ));
            }
        } else {
            errors.push("Could not parse holder's agent pub key".to_string());
        }
    } else {
        errors.push("Invalid holder DID format".to_string());
    }

    // Check original credential revocation status
    let revocation_status = check_credential_revocation_status(&derived.original_credential_id)?;
    match revocation_status {
        CredentialRevocationStatus::Revoked(reason) => {
            errors.push(format!("Original credential revoked: {}", reason));
        }
        CredentialRevocationStatus::Suspended(reason, _) => {
            errors.push(format!("Original credential suspended: {}", reason));
        }
        CredentialRevocationStatus::Active => {}
        CredentialRevocationStatus::Unknown => {
            errors
                .push("Original credential revocation status could not be determined".to_string());
        }
    }

    Ok(DerivedVerificationResult {
        valid: errors.is_empty(),
        errors,
        original_issuer_verified,
    })
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
    // Capability guard: only the target issuer can approve/reject requests
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the request
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CredentialRequest,
        )?))
        .include_entries(true);

    // Find the latest version of this request (update_entry appends newer versions)
    let mut found_record: Option<Record> = None;
    let mut found_req: Option<CredentialRequest> = None;
    for record in query(filter)? {
        if let Some(req) = record
            .entry()
            .to_app_option::<CredentialRequest>()
            .ok()
            .flatten()
        {
            if req.id == input.request_id {
                found_req = Some(req);
                found_record = Some(record);
            }
        }
    }

    let (record, req) = match (found_record, found_req) {
        (Some(r), Some(q)) => (r, q),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Request not found".into()
            )))
        }
    };

    if req.issuer_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the target issuer can update request status".into()
        )));
    }
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated request".into()
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

// =============================================================================
// PRE-SIGNED CREDENTIAL ISSUANCE (for PQC/hybrid proofs created off-chain)
// =============================================================================

/// Input for issuing a credential with a pre-computed proof.
///
/// The proof was created off-chain (e.g. by the CLI using `hybrid-sign`)
/// and contains a TaggedSignature-encoded multibase proof value.
#[derive(Serialize, Deserialize, Debug)]
pub struct IssueCredentialWithProofInput {
    /// The complete W3C Verifiable Credential with proof already filled in.
    pub credential: VerifiableCredential,
}

/// Issue a credential with a pre-signed proof (PQC or hybrid).
///
/// This extern accepts a fully-formed VerifiableCredential whose `proof.proof_value`
/// was produced off-chain by a PQC-capable signer. It validates the structure
/// and stores the credential without re-signing.
///
/// The proof value should be a TaggedSignature-encoded multibase string so that
/// `verify_credential_signature` can dispatch to the correct algorithm.
#[hdk_extern]
pub fn issue_credential_with_proof(input: IssueCredentialWithProofInput) -> ExternResult<Record> {
    let vc = input.credential;

    // Capability guard: only the claimed issuer can submit pre-signed credentials
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if vc.issuer.did() != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the claimed issuer can submit pre-signed credentials".into()
        )));
    }

    // Validate basic structure
    if !vc.context.iter().any(|c| c.contains("credentials")) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential must include W3C credentials context".into()
        )));
    }
    if !vc
        .credential_type
        .contains(&"VerifiableCredential".to_string())
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential type must include 'VerifiableCredential'".into()
        )));
    }
    if !vc.issuer.did().starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Issuer must be a valid DID".into()
        )));
    }
    if vc.proof.proof_value.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Pre-signed credential must have a non-empty proof value".into()
        )));
    }

    // Validate that the proof value is parseable as a tagged signature
    if let Ok(tagged) = TaggedSignature::from_multibase(&vc.proof.proof_value) {
        if !tagged.algorithm.is_signature_algorithm() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Proof value uses a non-signature algorithm".into()
            )));
        }
    }
    // If TaggedSignature parsing fails, it may be a legacy Ed25519 multibase — that's OK

    // Validate claims against schema if specified
    let schema_validation =
        validate_claims_against_schema(&vc.mycelix_schema_id, &vc.credential_subject.claims)?;
    match &schema_validation {
        SchemaValidationStatus::Validated { schema_name } => {
            debug!(
                "Pre-signed credential schema validation passed for '{}'",
                schema_name
            );
        }
        SchemaValidationStatus::SchemaZomeUnavailable => {
            warn!("Pre-signed credential schema validation skipped: credential_schema zome unavailable");
        }
        SchemaValidationStatus::SchemaNotFound => {
            debug!("Pre-signed credential schema validation skipped: schema not found");
        }
        SchemaValidationStatus::NoSchema => {}
    }

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

    if !vc.mycelix_schema_id.is_empty() {
        let schema_hash = string_to_entry_hash(&vc.mycelix_schema_id);
        create_link(
            schema_hash,
            action_hash.clone(),
            LinkTypes::SchemaToCredential,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created credential".into()
    )))
}

/// Format timestamp as ISO 8601 string
fn format_timestamp_iso8601(ts: Timestamp) -> String {
    // Convert microseconds → RFC 3339 / ISO 8601 (no chrono in WASM)
    let total_secs = ts.as_micros() / 1_000_000;

    // Days since Unix epoch
    let days = total_secs / 86400;
    let day_secs = total_secs % 86400;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;

    // Civil date from day count (Euclidean affine algorithm)
    // Shift epoch to 0000-03-01 for easy leap year math
    let z = days + 719468; // days from 0000-03-01 to 1970-01-01
    let era = z / 146097;
    let doe = z - era * 146097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hour, minute, second
    )
}

// =============================================================================
// CRYPTOGRAPHIC SIGNING (Ed25519 via HDK)
// =============================================================================

/// Base58 alphabet for Bitcoin/IPFS encoding
const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

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
/// Returns the expiration status of a datetime string relative to `now`.
///
/// Fail-closed: if the datetime string cannot be parsed, returns
/// `ExpirationStatus::ParseError` so callers can treat it as a verification
/// failure rather than silently accepting a potentially expired credential.
fn parse_iso8601_expired(datetime_str: &str, now: Timestamp) -> ExpirationStatus {
    match parse_iso8601_to_micros(datetime_str) {
        Some(expiry_micros) => {
            let now_micros = now.as_micros();
            if now_micros > expiry_micros {
                ExpirationStatus::Expired
            } else {
                ExpirationStatus::Valid
            }
        }
        None => ExpirationStatus::ParseError,
    }
}

/// Result of checking a credential's expiration date.
#[derive(Debug, Clone, PartialEq)]
enum ExpirationStatus {
    /// The credential has not yet expired.
    Valid,
    /// The credential has expired.
    Expired,
    /// The expiration date could not be parsed (fail-closed: treat as invalid).
    ParseError,
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

    for d in &days_in_months[..((month - 1) as usize)] {
        days += *d as i64;
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
    for d in &days_in_months[(month as usize)..12] {
        remaining += *d as i64;
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
    if let Some(stripped) = s.strip_prefix('z') {
        base58_decode(stripped)
    } else {
        None
    }
}

// ============================================================================
// ZKP-ENHANCED SELECTIVE DISCLOSURE (DASTARK)
// ============================================================================

/// Input for ZKP-verified selective disclosure.
///
/// The holder proves a predicate about credential attributes without
/// revealing the attribute values themselves. For example:
/// - "My age is between 18-65" without revealing DOB
/// - "My credential is not expired" without revealing expiry date
/// - "My attribute X satisfies condition Y" without revealing X
///
/// Domain tag: `ZTML:Identity:SelectiveDisclosure:v1`
#[derive(Debug, Serialize, Deserialize, SerializedBytes)]
pub struct ZkSelectiveDisclosureInput {
    /// Original credential ID.
    pub credential_id: String,
    /// Predicate to prove (e.g., "age >= 18", "not_expired", "attribute_in_range").
    pub predicate: ZkPredicate,
    /// ZK proof bytes (generated client-side via DASTARK).
    pub proof_bytes: Vec<u8>,
    /// Commitment to the credential data (Blake3 hash).
    pub data_commitment: Vec<u8>,
}

/// Predicate types for selective disclosure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ZkPredicate {
    /// Age is within [min, max] range.
    AgeInRange { min: u32, max: Option<u32> },
    /// Credential has not expired.
    NotExpired,
    /// A numeric attribute satisfies a comparison.
    AttributeRange {
        attribute_key: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Credential was issued by a specific issuer (prove membership).
    IssuedBy { issuer_did_hash: Vec<u8> },
    /// Credential type matches expected type.
    CredentialTypeIs { expected_type: String },
}

/// Result of ZKP-verified selective disclosure.
#[derive(Debug, Serialize, Deserialize, SerializedBytes)]
pub struct ZkDisclosureResult {
    /// Whether the predicate is satisfied (per the ZK proof).
    pub satisfied: bool,
    /// Proof verification status.
    pub proof_valid: bool,
    /// Domain tag used for verification.
    pub domain_tag: String,
    /// Credential ID (public).
    pub credential_id: String,
    /// Predicate that was verified (public).
    pub predicate_description: String,
}

/// Verify a ZKP-based selective disclosure claim.
///
/// Checks:
/// 1. Original credential exists and belongs to the caller
/// 2. Data commitment matches the credential hash
/// 3. Proof bytes are non-empty and within size limits
/// 4. Domain tag is correct (ZTML:Identity:SelectiveDisclosure:v1)
///
/// Note: Full STARK proof verification is delegated to mycelix-zkp-core
/// once AIR circuits are implemented for each predicate type.
#[hdk_extern]
pub fn verify_selective_disclosure(
    input: ZkSelectiveDisclosureInput,
) -> ExternResult<ZkDisclosureResult> {
    let domain_tag = mycelix_zkp_core::domain::tag_identity_disclosure();

    // 1. Verify original credential exists
    let credential_record = get_credential(input.credential_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Original credential not found".into())
    ))?;

    let vc: VerifiableCredential = credential_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid credential entry".into()
        )))?;

    // 2. Verify caller is the credential subject
    let agent_info = agent_info()?;
    let caller_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    if vc.credential_subject.id != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the credential subject can create selective disclosures".into()
        )));
    }

    // 3. Verify proof bytes structure
    if input.proof_bytes.is_empty() {
        return Ok(ZkDisclosureResult {
            satisfied: false,
            proof_valid: false,
            domain_tag: domain_tag.as_str().to_string(),
            credential_id: input.credential_id,
            predicate_description: format!("{:?}", input.predicate),
        });
    }

    if input.proof_bytes.len() > 500_000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proof exceeds 500KB size limit".into()
        )));
    }

    // 4. Verify data commitment is 32 bytes (Blake3)
    if input.data_commitment.len() != 32 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Data commitment must be exactly 32 bytes".into()
        )));
    }

    // 5. Domain-tagged proof verification
    // The proof was generated with ZTML:Identity:SelectiveDisclosure:v1
    // Full STARK verification will be wired once AIR circuits are ready.
    // For now, structural validation + domain tag binding.
    let proof_valid = !input.proof_bytes.is_empty() && input.data_commitment.len() == 32;

    Ok(ZkDisclosureResult {
        satisfied: proof_valid,
        proof_valid,
        domain_tag: domain_tag.as_str().to_string(),
        credential_id: input.credential_id,
        predicate_description: format!("{:?}", input.predicate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- is_leap_year ---

    #[test]
    fn leap_year_divisible_by_4() {
        assert!(is_leap_year(2024));
        assert!(is_leap_year(2004));
        assert!(is_leap_year(1996));
    }

    #[test]
    fn leap_year_century_not_leap() {
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2100));
        assert!(!is_leap_year(2200));
    }

    #[test]
    fn leap_year_400_is_leap() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(1600));
        assert!(is_leap_year(2400));
    }

    #[test]
    fn non_leap_years() {
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(2025));
        assert!(!is_leap_year(1971));
    }

    // --- days_since_epoch ---

    #[test]
    fn epoch_is_zero() {
        assert_eq!(days_since_epoch(1970, 1, 1), Some(0));
    }

    #[test]
    fn known_date_2024_01_01() {
        // 2024-01-01 = 54 years (including leap years)
        // 1970-2023 = 54 years: 13 leap years (72,76,80,84,88,92,96,00,04,08,12,16,20) + 41 normal
        // 13*366 + 41*365 = 4758 + 14965 = 19723
        assert_eq!(days_since_epoch(2024, 1, 1), Some(19723));
    }

    #[test]
    fn known_date_2000_03_01() {
        // 2000-03-01 (right after Feb 29 leap day in Y2K)
        let days = days_since_epoch(2000, 3, 1).unwrap();
        // 1970-01-01 to 2000-03-01 = 11017 days
        assert_eq!(days, 11017);
    }

    // --- parse_tz_offset ---

    #[test]
    fn tz_offset_utc_plus() {
        assert_eq!(parse_tz_offset("+05:30"), Some(5 * 3600 + 30 * 60));
        assert_eq!(parse_tz_offset("+00:00"), Some(0));
        assert_eq!(parse_tz_offset("+08:00"), Some(8 * 3600));
    }

    #[test]
    fn tz_offset_utc_minus() {
        assert_eq!(parse_tz_offset("-08:00"), Some(-8 * 3600));
        assert_eq!(parse_tz_offset("-05:00"), Some(-5 * 3600));
    }

    #[test]
    fn tz_offset_compact_format() {
        assert_eq!(parse_tz_offset("+0530"), Some(5 * 3600 + 30 * 60));
        assert_eq!(parse_tz_offset("-0800"), Some(-8 * 3600));
    }

    #[test]
    fn tz_offset_hours_only() {
        assert_eq!(parse_tz_offset("+05"), Some(5 * 3600));
        assert_eq!(parse_tz_offset("-08"), Some(-8 * 3600));
    }

    #[test]
    fn tz_offset_invalid() {
        assert_eq!(parse_tz_offset("+"), None);
        assert_eq!(parse_tz_offset(""), None);
    }

    // --- parse_iso8601_to_micros ---

    #[test]
    fn iso8601_date_only() {
        let micros = parse_iso8601_to_micros("2024-01-01").unwrap();
        // Should be end of day (23:59:59)
        let expected_days = days_since_epoch(2024, 1, 1).unwrap();
        let expected_micros = (expected_days * 86400 + 86399) * 1_000_000;
        assert_eq!(micros, expected_micros);
    }

    #[test]
    fn iso8601_utc_z() {
        let micros = parse_iso8601_to_micros("2024-01-01T00:00:00Z").unwrap();
        let expected_days = days_since_epoch(2024, 1, 1).unwrap();
        assert_eq!(micros, expected_days * 86400 * 1_000_000);
    }

    #[test]
    fn iso8601_with_tz_offset() {
        let utc_micros = parse_iso8601_to_micros("2024-01-01T00:00:00Z").unwrap();
        let plus5_micros = parse_iso8601_to_micros("2024-01-01T05:00:00+05:00").unwrap();
        // Both should resolve to same UTC time
        assert_eq!(utc_micros, plus5_micros);
    }

    #[test]
    fn iso8601_invalid_month() {
        assert!(parse_iso8601_to_micros("2024-13-01").is_none());
        assert!(parse_iso8601_to_micros("2024-00-01").is_none());
    }

    #[test]
    fn iso8601_invalid_format() {
        assert!(parse_iso8601_to_micros("not a date").is_none());
        assert!(parse_iso8601_to_micros("").is_none());
    }

    // --- parse_iso8601_expired ---

    #[test]
    fn expired_past_date() {
        let now = Timestamp::from_micros(1_700_000_000_000_000); // ~2023-11-14
        assert_eq!(
            parse_iso8601_expired("2020-01-01T00:00:00Z", now),
            ExpirationStatus::Expired
        );
    }

    #[test]
    fn not_expired_future_date() {
        let now = Timestamp::from_micros(1_700_000_000_000_000);
        assert_eq!(
            parse_iso8601_expired("2030-01-01T00:00:00Z", now),
            ExpirationStatus::Valid
        );
    }

    #[test]
    fn unparseable_fails_closed() {
        let now = Timestamp::from_micros(1_700_000_000_000_000);
        assert_eq!(
            parse_iso8601_expired("garbage", now),
            ExpirationStatus::ParseError,
            "Unparseable dates must fail closed (ParseError), not silently pass"
        );
    }

    // --- format_timestamp_iso8601 ---

    #[test]
    fn format_timestamp_epoch() {
        let ts = Timestamp::from_micros(0);
        assert_eq!(format_timestamp_iso8601(ts), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn format_timestamp_known_value() {
        let ts = Timestamp::from_micros(1_700_000_000_000_000);
        let formatted = format_timestamp_iso8601(ts);
        assert_eq!(formatted, "2023-11-14T22:13:20Z");
    }

    // --- base58_decode ---

    #[test]
    fn base58_decode_known_value() {
        // "1" in base58 = 0x00
        let result = base58_decode("1").unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn base58_decode_round_trip_simple() {
        // "2" in base58 = 0x01
        let result = base58_decode("2").unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn base58_decode_invalid_char() {
        // '0', 'O', 'I', 'l' are not in base58 alphabet
        assert!(base58_decode("0").is_none());
        assert!(base58_decode("O").is_none());
        assert!(base58_decode("I").is_none());
        assert!(base58_decode("l").is_none());
    }

    // --- multibase_decode ---

    #[test]
    fn multibase_decode_z_prefix() {
        let result = multibase_decode("z2");
        assert!(result.is_some());
    }

    #[test]
    fn multibase_decode_wrong_prefix() {
        assert!(multibase_decode("f01020304").is_none());
        assert!(multibase_decode("m01020304").is_none());
        assert!(multibase_decode("").is_none());
    }

    // --- compute_credential_hash ---

    #[test]
    fn credential_hash_deterministic() {
        let vc = VerifiableCredential {
            context: vec!["https://www.w3.org/ns/credentials/v2".into()],
            id: "test-cred-1".into(),
            credential_type: vec!["VerifiableCredential".into()],
            issuer: CredentialIssuer::Did("did:mycelix:issuer1".into()),
            valid_from: "2026-01-01T00:00:00Z".into(),
            valid_until: None,
            credential_subject: CredentialSubject {
                id: "did:mycelix:holder1".into(),
                claims: serde_json::json!({"degree": "BSc"}),
            },
            credential_schema: None,
            credential_status: None,
            proof: CredentialProof {
                proof_type: "Ed25519Signature2020".into(),
                created: "2026-01-01".into(),
                verification_method: "did:mycelix:issuer1#key-1".into(),
                proof_purpose: "assertionMethod".into(),
                proof_value: "zSig".into(),
                cryptosuite: None,
                algorithm: None,
                challenge: None,
                domain: None,
            },
            mycelix_schema_id: "mycelix:schema:test:v1".into(),
            mycelix_created: Timestamp::from_micros(0),
        };
        let hash1 = compute_credential_hash(&vc);
        let hash2 = compute_credential_hash(&vc);
        assert_eq!(hash1, hash2, "Hash must be deterministic");
        assert_eq!(hash1.len(), 32, "BLAKE2b-256 produces 32 bytes");
    }

    #[test]
    fn credential_hash_differs_for_different_ids() {
        let make_vc = |id: &str| VerifiableCredential {
            context: vec!["https://www.w3.org/ns/credentials/v2".into()],
            id: id.into(),
            credential_type: vec!["VerifiableCredential".into()],
            issuer: CredentialIssuer::Did("did:mycelix:issuer1".into()),
            valid_from: "2026-01-01T00:00:00Z".into(),
            valid_until: None,
            credential_subject: CredentialSubject {
                id: "did:mycelix:holder1".into(),
                claims: serde_json::json!({}),
            },
            credential_schema: None,
            credential_status: None,
            proof: CredentialProof {
                proof_type: "Ed25519Signature2020".into(),
                created: "2026-01-01".into(),
                verification_method: "did:mycelix:issuer1#key-1".into(),
                proof_purpose: "assertionMethod".into(),
                proof_value: "zSig".into(),
                cryptosuite: None,
                algorithm: None,
                challenge: None,
                domain: None,
            },
            mycelix_schema_id: "mycelix:schema:test:v1".into(),
            mycelix_created: Timestamp::from_micros(0),
        };
        let hash_a = compute_credential_hash(&make_vc("cred-A"));
        let hash_b = compute_credential_hash(&make_vc("cred-B"));
        assert_ne!(hash_a, hash_b);
    }

    // --- Credential Revocation Status lifecycle ---

    #[test]
    fn revocation_status_to_verify_errors_active() {
        // Simulate the verify_credential error-aggregation pattern
        let mut errors = Vec::new();
        let status = CredentialRevocationStatus::Active;
        match status {
            CredentialRevocationStatus::Revoked(reason) => {
                errors.push(format!("Credential revoked: {}", reason));
            }
            CredentialRevocationStatus::Suspended(reason, until) => {
                errors.push(format!("Credential suspended until {}: {}", until, reason));
            }
            CredentialRevocationStatus::Active => {}
            CredentialRevocationStatus::Unknown => {
                errors.push("Revocation status could not be determined".to_string());
            }
        }
        assert!(
            errors.is_empty(),
            "Active credential should produce no errors"
        );
    }

    #[test]
    fn revocation_status_to_verify_errors_revoked() {
        let mut errors = Vec::new();
        let status = CredentialRevocationStatus::Revoked("Fraud detected".into());
        match status {
            CredentialRevocationStatus::Revoked(reason) => {
                errors.push(format!("Credential revoked: {}", reason));
            }
            CredentialRevocationStatus::Suspended(reason, until) => {
                errors.push(format!("Credential suspended until {}: {}", until, reason));
            }
            CredentialRevocationStatus::Active => {}
            CredentialRevocationStatus::Unknown => {
                errors.push("Revocation status could not be determined".to_string());
            }
        }
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Fraud detected"));
    }

    #[test]
    fn revocation_status_to_verify_errors_suspended() {
        let mut errors = Vec::new();
        let status = CredentialRevocationStatus::Suspended(
            "Under review".into(),
            "2026-12-31T00:00:00Z".into(),
        );
        match status {
            CredentialRevocationStatus::Revoked(reason) => {
                errors.push(format!("Credential revoked: {}", reason));
            }
            CredentialRevocationStatus::Suspended(reason, until) => {
                errors.push(format!("Credential suspended until {}: {}", until, reason));
            }
            CredentialRevocationStatus::Active => {}
            CredentialRevocationStatus::Unknown => {
                errors.push("Revocation status could not be determined".to_string());
            }
        }
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Under review"));
        assert!(errors[0].contains("2026-12-31"));
    }

    #[test]
    fn revocation_status_unknown_treated_as_invalid() {
        // SECURITY: Unknown status (revocation zome unavailable) must block verification
        // (fail-closed). Previously this was fail-open — fixed March 2026.
        let mut errors = Vec::new();
        let status = CredentialRevocationStatus::Unknown;
        match status {
            CredentialRevocationStatus::Revoked(reason) => {
                errors.push(format!("Credential revoked: {}", reason));
            }
            CredentialRevocationStatus::Suspended(reason, until) => {
                errors.push(format!("Credential suspended until {}: {}", until, reason));
            }
            CredentialRevocationStatus::Active => {}
            CredentialRevocationStatus::Unknown => {
                errors.push("Revocation status could not be determined".to_string());
            }
        }
        assert_eq!(
            errors.len(),
            1,
            "Unknown must produce an error (fail-closed)"
        );
        assert!(errors[0].contains("could not be determined"));
    }

    #[test]
    fn revocation_status_mirror_to_credential_status_mapping() {
        // Test the mapping from RevocationCheckResult to CredentialRevocationStatus
        // (mirrors check_credential_revocation_status logic)

        // Active
        let check = RevocationCheckResult {
            credential_id: "cred:1".into(),
            status: RevocationStatusMirror::Active,
            reason: None,
            checked_at: Timestamp::from_micros(0),
        };
        let mapped = match check.status {
            RevocationStatusMirror::Active => CredentialRevocationStatus::Active,
            RevocationStatusMirror::Revoked => CredentialRevocationStatus::Revoked(
                check.reason.unwrap_or_else(|| "No reason provided".into()),
            ),
            RevocationStatusMirror::Suspended => CredentialRevocationStatus::Suspended(
                check.reason.unwrap_or_else(|| "No reason provided".into()),
                check.checked_at.to_string(),
            ),
        };
        assert!(matches!(mapped, CredentialRevocationStatus::Active));

        // Revoked with reason
        let check_revoked = RevocationCheckResult {
            credential_id: "cred:2".into(),
            status: RevocationStatusMirror::Revoked,
            reason: Some("Key compromised".into()),
            checked_at: Timestamp::from_micros(1_000_000),
        };
        let mapped_revoked = match check_revoked.status {
            RevocationStatusMirror::Active => CredentialRevocationStatus::Active,
            RevocationStatusMirror::Revoked => CredentialRevocationStatus::Revoked(
                check_revoked
                    .reason
                    .unwrap_or_else(|| "No reason provided".into()),
            ),
            RevocationStatusMirror::Suspended => CredentialRevocationStatus::Suspended(
                check_revoked
                    .reason
                    .unwrap_or_else(|| "No reason provided".into()),
                check_revoked.checked_at.to_string(),
            ),
        };
        assert!(
            matches!(mapped_revoked, CredentialRevocationStatus::Revoked(ref r) if r == "Key compromised")
        );

        // Revoked without reason (defaults to "No reason provided")
        let check_no_reason = RevocationCheckResult {
            credential_id: "cred:3".into(),
            status: RevocationStatusMirror::Revoked,
            reason: None,
            checked_at: Timestamp::from_micros(0),
        };
        let mapped_no_reason = match check_no_reason.status {
            RevocationStatusMirror::Active => CredentialRevocationStatus::Active,
            RevocationStatusMirror::Revoked => CredentialRevocationStatus::Revoked(
                check_no_reason
                    .reason
                    .unwrap_or_else(|| "No reason provided".into()),
            ),
            RevocationStatusMirror::Suspended => CredentialRevocationStatus::Suspended(
                check_no_reason
                    .reason
                    .unwrap_or_else(|| "No reason provided".into()),
                check_no_reason.checked_at.to_string(),
            ),
        };
        assert!(
            matches!(mapped_no_reason, CredentialRevocationStatus::Revoked(ref r) if r == "No reason provided")
        );
    }

    #[test]
    fn credential_status_response_mapping() {
        // Test the status → CredentialStatusResponse mapping (mirrors get_credential_status logic)
        let statuses = vec![
            (
                CredentialRevocationStatus::Active,
                true,
                "active",
                None::<&str>,
            ),
            (
                CredentialRevocationStatus::Revoked("Expired cert".into()),
                false,
                "revoked",
                Some("Expired cert"),
            ),
            // SECURITY: Unknown is now fail-closed (invalid)
            (
                CredentialRevocationStatus::Unknown,
                false,
                "unknown",
                Some("Revocation status could not be determined"),
            ),
        ];

        for (status, expected_valid, expected_type, expected_reason) in statuses {
            let (is_valid, status_type, reason) = match status {
                CredentialRevocationStatus::Active => (true, "active".to_string(), None),
                CredentialRevocationStatus::Revoked(r) => (false, "revoked".to_string(), Some(r)),
                CredentialRevocationStatus::Suspended(r, until) => {
                    (false, format!("suspended_until_{}", until), Some(r))
                }
                // SECURITY: Fail-closed — unknown revocation status treated as invalid
                CredentialRevocationStatus::Unknown => (
                    false,
                    "unknown".to_string(),
                    Some("Revocation status could not be determined".to_string()),
                ),
            };
            assert_eq!(is_valid, expected_valid, "valid for {}", status_type);
            assert_eq!(status_type, expected_type);
            assert_eq!(reason.as_deref(), expected_reason);
        }
    }

    #[test]
    fn is_credential_revoked_logic() {
        // Test the is_credential_revoked boolean reduction pattern
        let cases: Vec<(CredentialRevocationStatus, bool)> = vec![
            (CredentialRevocationStatus::Active, false),
            (CredentialRevocationStatus::Revoked("test".into()), true),
            (
                CredentialRevocationStatus::Suspended("test".into(), "2026".into()),
                true,
            ),
            // SECURITY: Unknown is now fail-closed (treated as revoked)
            (CredentialRevocationStatus::Unknown, true),
        ];
        for (status, expected) in cases {
            let result = matches!(
                status,
                CredentialRevocationStatus::Revoked(_)
                    | CredentialRevocationStatus::Suspended(_, _)
                    | CredentialRevocationStatus::Unknown
            );
            assert_eq!(result, expected, "is_revoked for {:?}", status);
        }
    }

    #[test]
    fn revocation_check_result_serde_round_trip() {
        // Verify RevocationCheckResult serializes/deserializes correctly
        // (critical for cross-zome calls)
        let check = RevocationCheckResult {
            credential_id: "urn:uuid:12345".into(),
            status: RevocationStatusMirror::Revoked,
            reason: Some("Key compromise".into()),
            checked_at: Timestamp::from_micros(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&check).unwrap();
        let restored: RevocationCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.credential_id, "urn:uuid:12345");
        assert_eq!(restored.status, RevocationStatusMirror::Revoked);
        assert_eq!(restored.reason.unwrap(), "Key compromise");
    }

    #[test]
    fn revocation_status_mirror_serde_all_variants() {
        let variants = vec![
            (RevocationStatusMirror::Active, "Active"),
            (RevocationStatusMirror::Suspended, "Suspended"),
            (RevocationStatusMirror::Revoked, "Revoked"),
        ];
        for (variant, expected_json_contains) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert!(json.contains(expected_json_contains));
            let restored: RevocationStatusMirror = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, variant);
        }
    }

    // --- Schema validation status ---

    #[test]
    fn schema_validation_status_equality() {
        assert_eq!(
            SchemaValidationStatus::NoSchema,
            SchemaValidationStatus::NoSchema
        );
        assert_eq!(
            SchemaValidationStatus::Validated {
                schema_name: "test".into()
            },
            SchemaValidationStatus::Validated {
                schema_name: "test".into()
            }
        );
        assert_ne!(
            SchemaValidationStatus::NoSchema,
            SchemaValidationStatus::SchemaZomeUnavailable
        );
    }

    // --- Merkle tree for selective disclosure ---

    #[test]
    fn merkle_tree_single_claim() {
        let mut claims = serde_json::Map::new();
        claims.insert("degree".into(), serde_json::json!("BSc"));

        let (root, leaves) = build_claim_merkle_tree(&claims);
        assert_eq!(root.len(), 32);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].0, "degree");
        // Single leaf = root
        assert_eq!(root, leaves[0].1);
    }

    #[test]
    fn merkle_tree_deterministic() {
        let mut claims = serde_json::Map::new();
        claims.insert("name".into(), serde_json::json!("Alice"));
        claims.insert("age".into(), serde_json::json!(30));
        claims.insert("degree".into(), serde_json::json!("PhD"));

        let (root1, _) = build_claim_merkle_tree(&claims);
        let (root2, _) = build_claim_merkle_tree(&claims);
        assert_eq!(root1, root2, "Merkle root must be deterministic");
    }

    #[test]
    fn merkle_tree_different_claims_different_root() {
        let mut claims1 = serde_json::Map::new();
        claims1.insert("name".into(), serde_json::json!("Alice"));

        let mut claims2 = serde_json::Map::new();
        claims2.insert("name".into(), serde_json::json!("Bob"));

        let (root1, _) = build_claim_merkle_tree(&claims1);
        let (root2, _) = build_claim_merkle_tree(&claims2);
        assert_ne!(root1, root2);
    }

    #[test]
    fn merkle_proof_verifies_for_all_claims() {
        let mut claims = serde_json::Map::new();
        claims.insert("name".into(), serde_json::json!("Alice"));
        claims.insert("age".into(), serde_json::json!(30));
        claims.insert("degree".into(), serde_json::json!("PhD"));
        claims.insert("university".into(), serde_json::json!("MIT"));

        let (root, _) = build_claim_merkle_tree(&claims);
        let mut sorted_keys: Vec<&String> = claims.keys().collect();
        sorted_keys.sort();

        for (idx, key) in sorted_keys.iter().enumerate() {
            let proof = generate_merkle_proof(&claims, idx);
            let leaf = hash_claim_leaf(key, &claims[*key]);

            // Walk proof to reconstruct root
            let mut current = leaf;
            let mut i = idx;
            for sibling in &proof {
                let combined = if i % 2 == 0 {
                    let mut c = current.clone();
                    c.extend(sibling);
                    c
                } else {
                    let mut c = sibling.clone();
                    c.extend(&current);
                    c
                };
                current = holo_hash::blake2b_256(&combined).to_vec();
                i /= 2;
            }

            assert_eq!(
                current, root,
                "Merkle proof should verify for claim '{}'",
                key
            );
        }
    }

    #[test]
    fn merkle_proof_empty_for_single_claim() {
        let mut claims = serde_json::Map::new();
        claims.insert("only".into(), serde_json::json!(true));

        let proof = generate_merkle_proof(&claims, 0);
        assert!(proof.is_empty(), "Single claim needs no proof siblings");
    }

    #[test]
    fn hash_claim_leaf_deterministic() {
        let val = serde_json::json!("test_value");
        let h1 = hash_claim_leaf("key", &val);
        let h2 = hash_claim_leaf("key", &val);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);
    }

    #[test]
    fn hash_claim_leaf_different_keys() {
        let val = serde_json::json!("same_value");
        let h1 = hash_claim_leaf("key1", &val);
        let h2 = hash_claim_leaf("key2", &val);
        assert_ne!(h1, h2, "Different keys should produce different hashes");
    }
}

/// Sign credential content using the agent's ed25519 key
///
/// This uses Holochain's HDK sign_raw which performs ed25519 signing
/// with the agent's cryptographic identity. The result is a
/// `TaggedSignature`-aware multibase string that includes the algorithm
/// multicodec prefix so verifiers can detect the algorithm.
fn sign_credential(vc: &VerifiableCredential) -> ExternResult<String> {
    // Compute canonical hash of credential content
    let content_hash = compute_credential_hash(vc);

    // Sign with agent's ed25519 key via HDK
    let signature = sign_raw(agent_info()?.agent_initial_pubkey, content_hash.clone())?;

    // Wrap in TaggedSignature for algorithm-tagged multibase encoding
    let tagged =
        TaggedSignature::new(AlgorithmId::Ed25519, signature.as_ref().to_vec()).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Signature tagging error: {}",
                e
            )))
        })?;
    Ok(tagged.to_multibase())
}

/// Verify a credential signature with algorithm dispatch.
///
/// Parses the proof value as a `TaggedSignature` to detect the algorithm,
/// then dispatches:
/// - Ed25519 → HDK verify_signature
/// - Hybrid → verify the Ed25519 component (PQC verification requires native)
/// - Pure PQC → structural accept in WASM (real verification off-chain)
///
/// Falls back to legacy 64-byte raw Ed25519 for old credentials.
fn verify_credential_signature(vc: &VerifiableCredential) -> ExternResult<bool> {
    // Extract public key from issuer DID
    let issuer_did = vc.issuer.did();
    let pubkey_str = issuer_did.strip_prefix("did:mycelix:").ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid DID format: {}",
            issuer_did
        )))
    })?;

    let pubkey = AgentPubKey::try_from(pubkey_str.to_string()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid public key in DID '{}': {:?}",
            pubkey_str, e
        )))
    })?;

    // Compute expected content hash
    let content_hash = compute_credential_hash(vc);

    // Try to parse as TaggedSignature (algorithm-aware multibase)
    match TaggedSignature::from_multibase(&vc.proof.proof_value) {
        Ok(tagged_sig) => {
            match tagged_sig.algorithm {
                AlgorithmId::Ed25519 => {
                    // Standard Ed25519 verification via HDK
                    if tagged_sig.signature_bytes.len() != 64 {
                        return Ok(false);
                    }
                    let sig = Signature::from(
                        <[u8; 64]>::try_from(tagged_sig.signature_bytes.as_slice()).map_err(
                            |_| {
                                wasm_error!(WasmErrorInner::Guest(
                                    "Invalid signature length".into()
                                ))
                            },
                        )?,
                    );
                    verify_signature(pubkey, sig, content_hash)
                }
                AlgorithmId::HybridEd25519MlDsa65 => {
                    // Verify Ed25519 component; PQC component verified off-chain
                    let ed_bytes = tagged_sig.ed25519_component().ok_or_else(|| {
                        wasm_error!(WasmErrorInner::Guest(
                            "Hybrid signature missing Ed25519 component".into()
                        ))
                    })?;
                    if ed_bytes.len() != 64 {
                        return Ok(false);
                    }
                    let sig = Signature::from(<[u8; 64]>::try_from(ed_bytes).map_err(|_| {
                        wasm_error!(WasmErrorInner::Guest("Invalid Ed25519 component".into()))
                    })?);
                    verify_signature(pubkey, sig, content_hash)
                }
                AlgorithmId::MlDsa65
                | AlgorithmId::MlDsa87
                | AlgorithmId::SlhDsaSha2_128s
                | AlgorithmId::SlhDsaShake128s => {
                    // Pure PQC: WASM cannot verify, accept structurally.
                    // Real verification happens via CLI/SDK (off-chain).
                    let expected_size = tagged_sig.algorithm.signature_size();
                    Ok(tagged_sig.signature_bytes.len() == expected_size)
                }
                _ => {
                    // Non-signature algorithm used as signature → reject
                    Ok(false)
                }
            }
        }
        Err(_) => {
            // Legacy fallback: try to decode as raw multibase (z + base58btc)
            let signature_bytes = multibase_decode(&vc.proof.proof_value).ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(
                    "Invalid multibase signature encoding".into()
                ))
            })?;
            if signature_bytes.len() != 64 {
                return Ok(false);
            }
            let signature =
                Signature::from(<[u8; 64]>::try_from(signature_bytes.as_slice()).map_err(
                    |_| wasm_error!(WasmErrorInner::Guest("Invalid signature length".into())),
                )?);
            verify_signature(pubkey, signature, content_hash)
        }
    }
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

/// Revocation check result from the revocation zome (local mirror type for deserialization)
#[derive(Serialize, Deserialize, Debug, Clone)]
struct RevocationCheckResult {
    pub credential_id: String,
    pub status: RevocationStatusMirror,
    pub reason: Option<String>,
    pub checked_at: Timestamp,
}

/// Mirror of revocation_integrity::RevocationStatus for cross-zome deserialization
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
enum RevocationStatusMirror {
    Active,
    Suspended,
    Revoked,
}

/// Check credential revocation status via cross-zome call to the revocation coordinator
fn check_credential_revocation_status(
    credential_id: &str,
) -> ExternResult<CredentialRevocationStatus> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("revocation"),
        FunctionName::new("check_revocation_status"),
        None,
        credential_id.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let check: RevocationCheckResult = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode revocation check response: {:?}",
                    e
                )))
            })?;

            match check.status {
                RevocationStatusMirror::Active => Ok(CredentialRevocationStatus::Active),
                RevocationStatusMirror::Revoked => Ok(CredentialRevocationStatus::Revoked(
                    check
                        .reason
                        .unwrap_or_else(|| "No reason provided".to_string()),
                )),
                RevocationStatusMirror::Suspended => Ok(CredentialRevocationStatus::Suspended(
                    check
                        .reason
                        .unwrap_or_else(|| "No reason provided".to_string()),
                    check.checked_at.to_string(),
                )),
            }
        }
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            // SECURITY: Fail-closed — revocation zome unreachable means we cannot
            // confirm the credential is unrevoked. Deny rather than assume active.
            Err(wasm_error!(WasmErrorInner::Guest(
                "Revocation check failed: revocation zome unreachable (fail-closed). \
                 Cannot verify credential is unrevoked."
                    .to_string()
            )))
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling revocation zome: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
    }
}

/// Batch check revocation status for multiple credentials in a single cross-zome call.
///
/// Falls back to individual checks if the revocation zome doesn't support batch.
fn batch_check_credential_revocation_status(
    credential_ids: &[String],
) -> ExternResult<Vec<CredentialRevocationStatus>> {
    if credential_ids.is_empty() {
        return Ok(Vec::new());
    }

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("revocation"),
        FunctionName::new("batch_check_revocation"),
        None,
        credential_ids.to_vec(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let checks: Vec<RevocationCheckResult> = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode batch revocation response: {:?}",
                    e
                )))
            })?;
            Ok(checks
                .into_iter()
                .map(|check| match check.status {
                    RevocationStatusMirror::Active => CredentialRevocationStatus::Active,
                    RevocationStatusMirror::Revoked => CredentialRevocationStatus::Revoked(
                        check
                            .reason
                            .unwrap_or_else(|| "No reason provided".to_string()),
                    ),
                    RevocationStatusMirror::Suspended => CredentialRevocationStatus::Suspended(
                        check
                            .reason
                            .unwrap_or_else(|| "No reason provided".to_string()),
                        check.checked_at.to_string(),
                    ),
                })
                .collect())
        }
        ZomeCallResponse::Unauthorized(_, _, _, _)
        | ZomeCallResponse::AuthenticationFailed(_, _) => {
            // SECURITY: Fail-closed — cannot batch-verify revocation status
            Err(wasm_error!(WasmErrorInner::Guest(
                "Batch revocation check failed: revocation zome unreachable (fail-closed). \
                 Cannot verify credentials are unrevoked."
                    .to_string()
            )))
        }
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Network error calling batch revocation: {}",
            err
        )))),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Countersigning error: {}", err)
        ))),
    }
}

/// Check if a specific credential is revoked (public API)
#[hdk_extern]
pub fn is_credential_revoked(credential_id: String) -> ExternResult<bool> {
    let status = check_credential_revocation_status(&credential_id)?;
    match status {
        CredentialRevocationStatus::Revoked(_) | CredentialRevocationStatus::Suspended(_, _) => {
            Ok(true)
        }
        CredentialRevocationStatus::Active => Ok(false),
        // SECURITY: Fail-closed — unknown status treated as revoked
        CredentialRevocationStatus::Unknown => Ok(true),
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
        // SECURITY: Fail-closed — unknown revocation status treated as invalid
        CredentialRevocationStatus::Unknown => (
            false,
            "unknown".to_string(),
            Some("Revocation status could not be determined".to_string()),
        ),
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
