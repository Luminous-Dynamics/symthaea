// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Verifiable Credential Integrity Zome
//!
//! W3C Verifiable Credentials Data Model 2.0 compliant implementation
//! https://www.w3.org/TR/vc-data-model-2.0/

use hdi::prelude::*;

/// W3C Verifiable Credential
/// Full implementation of VC Data Model 2.0
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    /// JSON-LD context (required: "https://www.w3.org/ns/credentials/v2")
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

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    VerifiableCredential(VerifiableCredential),
    VerifiablePresentation(VerifiablePresentation),
    DerivedCredential(DerivedCredential),
    CredentialRequest(CredentialRequest),
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
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CredentialRequest(req) => {
                    validate_update_credential_request(action, req)
                }
                _ => Ok(ValidateCallbackResult::Invalid(
                    "Credentials and presentations cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::IssuerToCredential => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SubjectToCredential => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HolderToPresentation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CredentialToDerived => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SchemaToCredential => Ok(ValidateCallbackResult::Valid),
            LinkTypes::IssuerToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RequesterToRequest => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
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
    _action: Update,
    _req: CredentialRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Request updates are allowed (status changes)
    Ok(ValidateCallbackResult::Valid)
}
