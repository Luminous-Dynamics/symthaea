#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// legal-did integrity — isolated DID namespace for state-facing identities.
// MUST NOT share linkage with the primary `did:mycelix:*` namespace.
// See ../../docs/THREAT_MODEL.md for the vectors this isolation addresses.

use hdi::prelude::*;

/// A state-facing DID. Lives in a distinct DHT partition from the primary
/// consciousness-gated identity. Holds nothing except what the state needs.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LegalDid {
    /// The DID string, always of form `did:mycelix:legal:<opaque-id>`.
    /// The opaque-id MUST be unrelated to any primary DID controlled by
    /// the same agent (enforced at coordinator creation time).
    pub did: String,

    /// ISO 8601 creation timestamp.
    pub created_at: String,

    /// Optional human-readable label, for the user's own organization only.
    /// Never disclosed to verifiers. Never indexed.
    pub label: Option<String>,
}

/// A sovereign-issued or regulated-intermediary-issued credential held under
/// the legal DID. The credential body is an eIDAS-compliant VC (see
/// `mycelix-identity/crates/eidas-zkp/`). Stored here only as an opaque blob;
/// unpacking happens client-side.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LegalCredentialRecord {
    /// Hash commitment to the underlying eIDAS credential.
    pub credential_hash: String,

    /// Issuer DID (`did:web:state.gov`, `did:web:jumio.com`, etc.).
    pub issuer_did: String,

    /// Credential type (e.g., `"PassportCredential"`, `"MobileDriversLicense"`).
    pub credential_type: String,

    /// ISO 8601 issuance date.
    pub issued_at: String,

    /// ISO 8601 expiry date, if any.
    pub expires_at: Option<String>,

    /// Revocation status URL (from credentialStatus field).
    pub revocation_check_url: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    LegalDid(LegalDid),
    LegalCredentialRecord(LegalCredentialRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent → their LegalDid entries.
    AgentToLegalDid,
    /// LegalDid → credentials held under it.
    LegalDidToCredential,
}

// ============================================================================
// Validation
// ============================================================================

/// Genesis self-check — called when the app is installed. We have no
/// membership gate; anyone running the cluster can create legal DIDs
/// under their own agent key.
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate an `Op` against the integrity rules. The legal-did zome
/// enforces a minimal ruleset: DID strings must start with the legal
/// prefix, and credentials must reference a non-empty issuer DID.
/// Richer validation (credential-signature verification, issuer-tier
/// lookup) happens in the coordinator layer so it can call out to
/// other zomes.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::LegalDid(did_entry) => validate_legal_did(&did_entry),
            EntryTypes::LegalCredentialRecord(cred) => validate_credential_record(&cred),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_legal_did(entry: &LegalDid) -> ExternResult<ValidateCallbackResult> {
    const PREFIX: &str = "did:mycelix:legal:";
    if !entry.did.starts_with(PREFIX) {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "legal DID must start with \"{PREFIX}\""
        )));
    }
    let suffix = &entry.did[PREFIX.len()..];
    // Opaque id is 32 bytes = 64 hex chars.
    if suffix.len() != 64 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "legal DID opaque id must be 64 hex chars, got {}",
            suffix.len()
        )));
    }
    if !suffix.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "legal DID opaque id must be hex".to_string(),
        ));
    }
    if entry.created_at.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "created_at missing".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_credential_record(
    entry: &LegalCredentialRecord,
) -> ExternResult<ValidateCallbackResult> {
    if entry.credential_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "credential_hash missing".to_string(),
        ));
    }
    if entry.issuer_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "issuer_did missing".to_string(),
        ));
    }
    if entry.credential_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "credential_type missing".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
