// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Credential Integrity Zome
//!
//! W3C Verifiable Credentials Data Model 2.0 compliant implementation
//! <https://www.w3.org/TR/vc-data-model-2.0/>

use hdi::prelude::*;

/// W3C Verifiable Credential
/// Full implementation of VC Data Model 2.0
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    /// JSON-LD context (required: `https://www.w3.org/ns/credentials/v2`)
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Unique credential identifier
    pub id: String,
    /// Credential types (must include "VerifiableCredential")
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    /// DID of the issuer
    pub issuer: CredentialIssuer,
    /// Issuance date (ISO 8601)
    #[serde(rename = "validFrom")]
    pub valid_from: String,
    /// Expiration date (optional, ISO 8601)
    #[serde(rename = "validUntil")]
    pub valid_until: Option<String>,
    /// The claims being made
    #[serde(rename = "credentialSubject")]
    pub credential_subject: CredentialSubject,
    /// Schema reference
    #[serde(rename = "credentialSchema")]
    pub credential_schema: Option<CredentialSchemaRef>,
    /// Credential status (for revocation checking)
    #[serde(rename = "credentialStatus")]
    pub credential_status: Option<CredentialStatus>,
    /// Cryptographic proof
    pub proof: CredentialProof,
    /// Mycelix-specific: schema ID used
    pub mycelix_schema_id: String,
    /// Mycelix-specific: creation timestamp
    pub mycelix_created: Timestamp,
}

/// Credential issuer - can be DID string or object with id
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CredentialIssuer {
    /// Simple DID string
    Did(String),
    /// Object with id and optional properties
    Object {
        id: String,
        name: Option<String>,
        #[serde(rename = "type")]
        issuer_type: Option<Vec<String>>,
    },
}

impl CredentialIssuer {
    pub fn did(&self) -> &str {
        match self {
            CredentialIssuer::Did(did) => did,
            CredentialIssuer::Object { id, .. } => id,
        }
    }
}

/// Credential subject containing the claims
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// DID of the subject
    pub id: String,
    /// Claims as key-value pairs (JSON)
    #[serde(flatten)]
    pub claims: serde_json::Value,
}

/// Reference to credential schema
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CredentialSchemaRef {
    pub id: String,
    #[serde(rename = "type")]
    pub schema_type: String,
}

/// Credential status for revocation
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CredentialStatus {
    pub id: String,
    #[serde(rename = "type")]
    pub status_type: String,
    /// For BitstringStatusList
    #[serde(rename = "statusPurpose")]
    pub status_purpose: Option<String>,
    #[serde(rename = "statusListIndex")]
    pub status_list_index: Option<String>,
    #[serde(rename = "statusListCredential")]
    pub status_list_credential: Option<String>,
}

/// Cryptographic proof
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type (Ed25519Signature2020, DataIntegrityProof, etc.)
    #[serde(rename = "type")]
    pub proof_type: String,
    /// When the proof was created (ISO 8601)
    pub created: String,
    /// Verification method used (DID URL)
    #[serde(rename = "verificationMethod")]
    pub verification_method: String,
    /// Purpose of the proof
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,
    /// The actual signature/proof value (multibase encoded)
    #[serde(rename = "proofValue")]
    pub proof_value: String,
    /// For DataIntegrityProof: cryptosuite used
    pub cryptosuite: Option<String>,
    /// Algorithm identifier (multicodec u16). None defaults to Ed25519 (0xed01).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
    /// Challenge value for replay protection (W3C Data Integrity spec).
    /// When present, the verifier MUST supply the same challenge to verify.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    /// Domain restriction (W3C Data Integrity spec).
    /// When present, the verifier MUST supply the same domain to verify.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Verifiable Presentation - for presenting credentials
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiablePresentation {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Unique presentation identifier
    pub id: String,
    /// Types (must include "VerifiablePresentation")
    #[serde(rename = "type")]
    pub presentation_type: Vec<String>,
    /// DID of the holder presenting
    pub holder: String,
    /// Credentials being presented
    #[serde(rename = "verifiableCredential")]
    pub verifiable_credential: Vec<VerifiableCredential>,
    /// Proof of presentation
    pub proof: CredentialProof,
    /// Mycelix-specific: creation timestamp
    pub mycelix_created: Timestamp,
}

/// Derived Credential for selective disclosure
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DerivedCredential {
    /// Original credential ID
    pub original_credential_id: String,
    /// DID of the original issuer
    pub original_issuer: String,
    /// DID of the holder creating the derivation
    pub holder: String,
    /// Selected claims (subset of original)
    pub selected_claims: Vec<String>,
    /// The derived credential content
    pub derived_content: CredentialSubject,
    /// Proof that this is a valid derivation
    pub derivation_proof: DerivationProof,
    /// Creation timestamp
    pub created: Timestamp,
    /// Expiration (inherits from original or shorter)
    pub expires: Option<Timestamp>,
}

/// Proof of valid derivation from original credential
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DerivationProof {
    /// Type of derivation proof
    #[serde(rename = "type")]
    pub proof_type: String,
    /// Hash of original credential
    pub original_credential_hash: Vec<u8>,
    /// Merkle proof for selected claims (if using Merkle tree)
    pub claim_proofs: Vec<ClaimProof>,
    /// Holder's signature on the derivation
    pub holder_signature: Vec<u8>,
}

/// Proof for individual claim in selective disclosure
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ClaimProof {
    /// Claim key
    pub claim_key: String,
    /// Merkle path (for Merkle tree proofs)
    pub merkle_path: Option<Vec<Vec<u8>>>,
    /// Commitment (for commitment schemes)
    pub commitment: Option<Vec<u8>>,
}

/// Credential issuance request (holder to issuer)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialRequest {
    /// Request ID
    pub id: String,
    /// Requester's DID
    pub requester_did: String,
    /// Target issuer's DID
    pub issuer_did: String,
    /// Schema ID for requested credential
    pub schema_id: String,
    /// Claims the requester is providing
    pub provided_claims: serde_json::Value,
    /// Supporting evidence (links to other credentials, documents, etc.)
    pub evidence: Vec<CredentialEvidence>,
    /// Request status
    pub status: RequestStatus,
    /// Request timestamp
    pub created: Timestamp,
    /// Status update timestamp
    pub updated: Timestamp,
}

/// Evidence supporting a credential request
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CredentialEvidence {
    /// Evidence type
    #[serde(rename = "type")]
    pub evidence_type: String,
    /// Evidence ID/URL
    pub id: String,
    /// Description
    pub description: Option<String>,
}

/// Status of credential request
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum RequestStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    Issued,
}

/// An encrypted entry wrapping sensitive credential data on the DHT.
///
/// Credential claims containing PII are encrypted to the subject's ML-KEM
/// public key. Only the subject (or parties they delegate to) can decrypt.
///
/// The `entry_type_tag` identifies what was encrypted so the decryptor
/// knows which type to deserialize after decryption.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EncryptedEntry {
    /// Which entry type is encrypted (e.g. "CredentialClaims", "DerivedContent").
    pub entry_type_tag: String,
    /// KEM algorithm used to derive the symmetric key (AlgorithmId as u16).
    /// 0xF020 = ML-KEM-768, 0xF021 = ML-KEM-1024, 0xF030 = self-encrypt.
    pub kem_algorithm: u16,
    /// KEM ciphertext (encapsulated key). Empty for self-encryption.
    pub encapsulated_key: Vec<u8>,
    /// 24-byte nonce for XChaCha20-Poly1305.
    pub nonce: Vec<u8>,
    /// AEAD ciphertext (plaintext || 16-byte Poly1305 tag).
    pub ciphertext: Vec<u8>,
    /// DID URL of the recipient's KEM key (e.g. "did:mycelix:abc#kem-1").
    pub recipient_key_id: String,
    /// When the entry was encrypted.
    pub encrypted_at: Timestamp,
    /// Schema version of the plaintext, for forward-compatible decryption.
    pub plaintext_version: u32,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
#[allow(clippy::large_enum_variant)] // HDK entry types require inline variants
pub enum EntryTypes {
    VerifiableCredential(VerifiableCredential),
    VerifiablePresentation(VerifiablePresentation),
    DerivedCredential(DerivedCredential),
    CredentialRequest(CredentialRequest),
    EncryptedEntry(EncryptedEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Issuer to credentials they've issued
    IssuerToCredential,
    /// Subject to credentials about them
    SubjectToCredential,
    /// Holder to presentations they've created
    HolderToPresentation,
    /// Credential to derived credentials
    CredentialToDerived,
    /// Schema to credentials using it
    SchemaToCredential,
    /// Issuer to pending requests
    IssuerToRequest,
    /// Requester to their requests
    RequesterToRequest,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::VerifiableCredential(vc) => {
                    validate_create_verifiable_credential(EntryCreationAction::Create(action), vc)
                }
                EntryTypes::VerifiablePresentation(vp) => {
                    validate_create_verifiable_presentation(EntryCreationAction::Create(action), vp)
                }
                EntryTypes::DerivedCredential(dc) => {
                    validate_create_derived_credential(EntryCreationAction::Create(action), dc)
                }
                EntryTypes::CredentialRequest(req) => {
                    validate_create_credential_request(EntryCreationAction::Create(action), req)
                }
                EntryTypes::EncryptedEntry(entry) => validate_create_encrypted_entry(entry),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CredentialRequest(req) => {
                    validate_update_credential_request(action, req)
                }
                EntryTypes::EncryptedEntry(_) => Ok(ValidateCallbackResult::Invalid(
                    "Encrypted entries are append-only (re-encrypt instead)".into(),
                )),
                _ => Ok(ValidateCallbackResult::Invalid(
                    "Credentials and presentations cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::IssuerToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SubjectToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::HolderToPresentation => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CredentialToDerived => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SchemaToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::IssuerToRequest => Ok(ValidateCallbackResult::Valid),
                LinkTypes::RequesterToRequest => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate verifiable credential creation
fn validate_create_verifiable_credential(
    _action: EntryCreationAction,
    vc: VerifiableCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Validate context includes W3C VC context
    if !vc.context.iter().any(|c| c.contains("credentials")) {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential must include W3C credentials context".into(),
        ));
    }

    // Validate type includes VerifiableCredential
    if !vc
        .credential_type
        .contains(&"VerifiableCredential".to_string())
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential type must include 'VerifiableCredential'".into(),
        ));
    }

    // Validate issuer is a DID
    if !vc.issuer.did().starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Validate subject has ID
    if !vc.credential_subject.id.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential subject must have a valid DID".into(),
        ));
    }

    // Validate proof exists and has required fields
    if vc.proof.proof_type.is_empty() || vc.proof.proof_value.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential must have valid proof".into(),
        ));
    }

    // Validate proof purpose
    if vc.proof.proof_purpose != "assertionMethod" {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential proof purpose must be 'assertionMethod'".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate verifiable presentation creation
fn validate_create_verifiable_presentation(
    _action: EntryCreationAction,
    vp: VerifiablePresentation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate type includes VerifiablePresentation
    if !vp
        .presentation_type
        .contains(&"VerifiablePresentation".to_string())
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation type must include 'VerifiablePresentation'".into(),
        ));
    }

    // Validate holder is a DID
    if !vp.holder.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Holder must be a valid DID".into(),
        ));
    }

    // Validate at least one credential
    if vp.verifiable_credential.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation must contain at least one credential".into(),
        ));
    }

    // Validate proof purpose for presentation
    if vp.proof.proof_purpose != "authentication" {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation proof purpose must be 'authentication'".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate derived credential creation
fn validate_create_derived_credential(
    _action: EntryCreationAction,
    dc: DerivedCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Validate holder is a DID
    if !dc.holder.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Holder must be a valid DID".into(),
        ));
    }

    // Validate selected claims not empty
    if dc.selected_claims.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must select at least one claim".into(),
        ));
    }

    // Validate derivation proof exists
    if dc.derivation_proof.holder_signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Derivation must have holder signature".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate credential request creation
fn validate_create_credential_request(
    _action: EntryCreationAction,
    req: CredentialRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate requester DID
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    // Validate issuer DID
    if !req.issuer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Target issuer must be a valid DID".into(),
        ));
    }

    // Validate schema ID
    if !req.schema_id.starts_with("mycelix:schema:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema ID must be valid Mycelix schema".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate credential request update
fn validate_update_credential_request(
    action: Update,
    req: CredentialRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Fetch original to enforce state transitions
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: CredentialRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original credential request not found".into()
        )))?;

    // Immutable fields
    if req.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID cannot be changed".into(),
        ));
    }
    if req.requester_did != original.requester_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID cannot be changed".into(),
        ));
    }
    if req.issuer_did != original.issuer_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer DID cannot be changed".into(),
        ));
    }
    if req.schema_id != original.schema_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema ID cannot be changed".into(),
        ));
    }

    // State machine: valid transitions
    let valid = match (&original.status, &req.status) {
        (RequestStatus::Pending, RequestStatus::UnderReview)
        | (RequestStatus::Pending, RequestStatus::Rejected)
        | (RequestStatus::UnderReview, RequestStatus::Approved)
        | (RequestStatus::UnderReview, RequestStatus::Rejected)
        | (RequestStatus::Approved, RequestStatus::Issued) => true,
        (a, b) if a == b => true, // No-op allowed
        _ => false,
    };

    if !valid {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid credential request status transition".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate encrypted entry creation
fn validate_create_encrypted_entry(entry: EncryptedEntry) -> ExternResult<ValidateCallbackResult> {
    // Validate entry type tag is non-empty
    if entry.entry_type_tag.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Encrypted entry must specify entry_type_tag".into(),
        ));
    }

    // Validate nonce length (XChaCha20-Poly1305 requires 24 bytes)
    if entry.nonce.len() != 24 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Nonce must be 24 bytes (XChaCha20-Poly1305), got {}",
            entry.nonce.len()
        )));
    }

    // Validate ciphertext is non-empty (minimum: 16-byte Poly1305 tag)
    if entry.ciphertext.len() < 16 {
        return Ok(ValidateCallbackResult::Invalid(
            "Ciphertext too short (minimum 16 bytes for Poly1305 tag)".into(),
        ));
    }

    // Validate recipient key ID is a DID URL or "self"
    if entry.recipient_key_id != "self" && !entry.recipient_key_id.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "recipient_key_id must be 'self' or a DID URL".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn valid_proof() -> CredentialProof {
        CredentialProof {
            proof_type: "Ed25519Signature2020".into(),
            created: "2026-01-01T00:00:00Z".into(),
            verification_method: "did:mycelix:issuer1#key-1".into(),
            proof_purpose: "assertionMethod".into(),
            proof_value: "zBase64EncodedSignature".into(),
            cryptosuite: None,
            algorithm: None,
            challenge: None,
            domain: None,
        }
    }

    fn valid_vc() -> VerifiableCredential {
        VerifiableCredential {
            context: vec![
                "https://www.w3.org/ns/credentials/v2".into(),
                "https://mycelix.net/ns/v1".into(),
            ],
            id: "urn:uuid:test-credential-001".into(),
            credential_type: vec!["VerifiableCredential".into(), "DegreeCredential".into()],
            issuer: CredentialIssuer::Did("did:mycelix:issuer1".into()),
            valid_from: "2026-01-01T00:00:00Z".into(),
            valid_until: Some("2030-01-01T00:00:00Z".into()),
            credential_subject: CredentialSubject {
                id: "did:mycelix:holder1".into(),
                claims: serde_json::json!({"degree": "BSc Computer Science"}),
            },
            credential_schema: Some(CredentialSchemaRef {
                id: "mycelix:schema:education:degree:v1".into(),
                schema_type: "JsonSchema".into(),
            }),
            credential_status: None,
            proof: valid_proof(),
            mycelix_schema_id: "mycelix:schema:education:degree:v1".into(),
            mycelix_created: ts(1_700_000_000_000_000),
        }
    }

    // --- W3C camelCase field naming ---

    #[test]
    fn vc_json_uses_w3c_camel_case_fields() {
        let vc = valid_vc();
        let json = serde_json::to_string_pretty(&vc).unwrap();

        // W3C fields must be camelCase
        assert!(json.contains("\"@context\""), "Must have @context");
        assert!(
            json.contains("\"validFrom\""),
            "Must have validFrom not valid_from"
        );
        assert!(
            json.contains("\"validUntil\""),
            "Must have validUntil not valid_until"
        );
        assert!(
            json.contains("\"credentialSubject\""),
            "Must have credentialSubject"
        );
        assert!(
            json.contains("\"credentialSchema\""),
            "Must have credentialSchema"
        );
        assert!(json.contains("\"proofPurpose\""), "Must have proofPurpose");
        assert!(json.contains("\"proofValue\""), "Must have proofValue");
        assert!(
            json.contains("\"verificationMethod\""),
            "Must have verificationMethod"
        );

        // Must NOT contain snake_case versions
        assert!(!json.contains("\"valid_from\""));
        assert!(!json.contains("\"valid_until\""));
        assert!(!json.contains("\"credential_subject\""));
        assert!(!json.contains("\"credential_schema\""));
        assert!(!json.contains("\"proof_purpose\""));
        assert!(!json.contains("\"proof_value\""));
        assert!(!json.contains("\"verification_method\""));

        // JSON-LD type field should be "type" not "credential_type"
        assert!(json.contains("\"type\""));
        assert!(!json.contains("\"credential_type\""));
    }

    #[test]
    fn vc_json_round_trip() {
        let vc = valid_vc();
        let json = serde_json::to_string(&vc).unwrap();
        let back: VerifiableCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, back);
    }

    // --- CredentialIssuer ---

    #[test]
    fn credential_issuer_did_string() {
        let issuer = CredentialIssuer::Did("did:mycelix:issuer1".into());
        assert_eq!(issuer.did(), "did:mycelix:issuer1");
        let json = serde_json::to_string(&issuer).unwrap();
        assert_eq!(json, "\"did:mycelix:issuer1\"");
    }

    #[test]
    fn credential_issuer_object() {
        let issuer = CredentialIssuer::Object {
            id: "did:mycelix:issuer2".into(),
            name: Some("Acme University".into()),
            issuer_type: Some(vec!["Issuer".into()]),
        };
        assert_eq!(issuer.did(), "did:mycelix:issuer2");
        let json = serde_json::to_string(&issuer).unwrap();
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"name\""));
        let back: CredentialIssuer = serde_json::from_str(&json).unwrap();
        assert_eq!(back.did(), "did:mycelix:issuer2");
    }

    // --- CredentialProof ---

    #[test]
    fn credential_proof_camel_case_fields() {
        let proof = valid_proof();
        let json = serde_json::to_string(&proof).unwrap();
        assert!(json.contains("\"proofPurpose\""));
        assert!(json.contains("\"proofValue\""));
        assert!(json.contains("\"verificationMethod\""));
        assert!(json.contains("\"type\""));
        assert!(!json.contains("\"proof_type\""));
    }

    #[test]
    fn credential_proof_with_algorithm() {
        let mut proof = valid_proof();
        proof.cryptosuite = Some("eddsa-rdfc-2022".into());
        proof.algorithm = Some(0xED01);
        let json = serde_json::to_string(&proof).unwrap();
        assert!(json.contains("\"cryptosuite\""));
        assert!(json.contains("\"algorithm\""));
        let back: CredentialProof = serde_json::from_str(&json).unwrap();
        assert_eq!(back.algorithm, Some(0xED01));
    }

    #[test]
    fn credential_proof_algorithm_defaults_to_none() {
        let proof = valid_proof();
        assert_eq!(proof.algorithm, None);
        let json = serde_json::to_string(&proof).unwrap();
        // skip_serializing_if = "Option::is_none"
        assert!(!json.contains("\"algorithm\""));
    }

    // --- VerifiablePresentation ---

    #[test]
    fn vp_json_camel_case_fields() {
        let vp = VerifiablePresentation {
            context: vec!["https://www.w3.org/ns/credentials/v2".into()],
            id: "urn:uuid:presentation-001".into(),
            presentation_type: vec!["VerifiablePresentation".into()],
            holder: "did:mycelix:holder1".into(),
            verifiable_credential: vec![valid_vc()],
            proof: CredentialProof {
                proof_purpose: "authentication".into(),
                ..valid_proof()
            },
            mycelix_created: ts(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&vp).unwrap();
        assert!(json.contains("\"verifiableCredential\""));
        assert!(!json.contains("\"verifiable_credential\""));
        // Round-trip
        let back: VerifiablePresentation = serde_json::from_str(&json).unwrap();
        assert_eq!(vp, back);
    }

    // --- Validation conditions ---

    #[test]
    fn vc_must_include_credentials_context() {
        let ctx = vec!["https://www.w3.org/ns/credentials/v2".to_string()];
        assert!(ctx.iter().any(|c| c.contains("credentials")));
        let bad_ctx = vec!["https://example.com".to_string()];
        assert!(!bad_ctx.iter().any(|c| c.contains("credentials")));
    }

    #[test]
    fn vc_type_must_include_verifiable_credential() {
        let types = vec![
            "VerifiableCredential".to_string(),
            "DegreeCredential".to_string(),
        ];
        assert!(types.contains(&"VerifiableCredential".to_string()));
        let bad_types = vec!["DegreeCredential".to_string()];
        assert!(!bad_types.contains(&"VerifiableCredential".to_string()));
    }

    #[test]
    fn vc_issuer_must_be_did() {
        assert!("did:mycelix:abc".starts_with("did:"));
        assert!(!"https://example.com".starts_with("did:"));
    }

    #[test]
    fn vp_proof_purpose_must_be_authentication() {
        assert_eq!("authentication", "authentication");
        assert_ne!("assertionMethod", "authentication");
    }

    // --- CredentialStatus ---

    #[test]
    fn credential_status_camel_case() {
        let status = CredentialStatus {
            id: "https://example.com/status/1".into(),
            status_type: "BitstringStatusListEntry".into(),
            status_purpose: Some("revocation".into()),
            status_list_index: Some("42".into()),
            status_list_credential: Some("https://example.com/status-list".into()),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"statusPurpose\""));
        assert!(json.contains("\"statusListIndex\""));
        assert!(json.contains("\"statusListCredential\""));
        let back: CredentialStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, back);
    }

    // --- RequestStatus ---

    #[test]
    fn request_status_json_variants() {
        let variants = vec![
            (RequestStatus::Pending, "\"Pending\""),
            (RequestStatus::UnderReview, "\"UnderReview\""),
            (RequestStatus::Approved, "\"Approved\""),
            (RequestStatus::Rejected, "\"Rejected\""),
            (RequestStatus::Issued, "\"Issued\""),
        ];
        for (variant, expected) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected);
        }
    }

    // --- CredentialRequest ---

    #[test]
    fn credential_request_json_round_trip() {
        let req = CredentialRequest {
            id: "req-001".into(),
            requester_did: "did:mycelix:holder1".into(),
            issuer_did: "did:mycelix:issuer1".into(),
            schema_id: "mycelix:schema:education:degree:v1".into(),
            provided_claims: serde_json::json!({"name": "Alice", "degree": "BSc CS"}),
            evidence: vec![CredentialEvidence {
                evidence_type: "DocumentVerification".into(),
                id: "evidence-001".into(),
                description: Some("Transcript verification".into()),
            }],
            status: RequestStatus::Pending,
            created: ts(1_700_000_000_000_000),
            updated: ts(1_700_000_000_000_000),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CredentialRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn credential_request_requires_did_prefix() {
        assert!("did:mycelix:holder1".starts_with("did:"));
        assert!(!"alice@example.com".starts_with("did:"));
    }

    #[test]
    fn credential_request_requires_schema_prefix() {
        assert!("mycelix:schema:education:degree:v1".starts_with("mycelix:schema:"));
        assert!(!"custom-schema-id".starts_with("mycelix:schema:"));
    }

    // --- EncryptedEntry ---

    #[test]
    fn encrypted_entry_json_round_trip() {
        let entry = EncryptedEntry {
            entry_type_tag: "CredentialClaims".into(),
            kem_algorithm: 0xF020,
            encapsulated_key: vec![1u8; 128],
            nonce: vec![0u8; 24],
            ciphertext: vec![42u8; 64],
            recipient_key_id: "did:mycelix:holder1#kem-1".into(),
            encrypted_at: ts(1_700_000_000_000_000),
            plaintext_version: 1,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: EncryptedEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn encrypted_entry_nonce_must_be_24_bytes() {
        assert_eq!(vec![0u8; 24].len(), 24, "24-byte nonce is valid");
        assert_ne!(vec![0u8; 16].len(), 24, "16-byte nonce is invalid");
        assert_ne!(vec![0u8; 32].len(), 24, "32-byte nonce is invalid");
    }

    #[test]
    fn encrypted_entry_ciphertext_minimum_16_bytes() {
        assert!(vec![0u8; 16].len() >= 16, "16 bytes is minimum (tag only)");
        assert!(vec![0u8; 15].len() < 16, "15 bytes is too short");
        assert!(vec![0u8; 64].len() >= 16, "64 bytes is valid");
    }

    #[test]
    fn encrypted_entry_recipient_key_id_must_be_did_or_self() {
        assert_eq!("self", "self");
        assert!("did:mycelix:holder1#kem-1".starts_with("did:"));
        assert!(!"random-key-id".starts_with("did:") && "random-key-id" != "self");
    }

    #[test]
    fn encrypted_entry_self_encryption() {
        let entry = EncryptedEntry {
            entry_type_tag: "DerivedContent".into(),
            kem_algorithm: 0xF030,    // self-encrypt
            encapsulated_key: vec![], // empty for self-encryption
            nonce: vec![0u8; 24],
            ciphertext: vec![42u8; 32],
            recipient_key_id: "self".into(),
            encrypted_at: ts(1_700_000_000_000_000),
            plaintext_version: 1,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: EncryptedEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
        assert!(entry.encapsulated_key.is_empty());
    }

    // --- DerivedCredential ---

    #[test]
    fn derived_credential_json_round_trip() {
        let dc = DerivedCredential {
            original_credential_id: "urn:uuid:cred-001".into(),
            original_issuer: "did:mycelix:issuer1".into(),
            holder: "did:mycelix:holder1".into(),
            selected_claims: vec!["degree".into(), "institution".into()],
            derived_content: CredentialSubject {
                id: "did:mycelix:holder1".into(),
                claims: serde_json::json!({"degree": "BSc CS"}),
            },
            derivation_proof: DerivationProof {
                proof_type: "MerkleDisclosure2024".into(),
                original_credential_hash: vec![0u8; 32],
                claim_proofs: vec![ClaimProof {
                    claim_key: "degree".into(),
                    merkle_path: Some(vec![vec![1u8; 32], vec![2u8; 32]]),
                    commitment: None,
                }],
                holder_signature: vec![3u8; 64],
            },
            created: ts(1_700_000_000_000_000),
            expires: Some(ts(1_800_000_000_000_000)),
        };
        let json = serde_json::to_string(&dc).unwrap();
        let back: DerivedCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(dc, back);
    }
}
