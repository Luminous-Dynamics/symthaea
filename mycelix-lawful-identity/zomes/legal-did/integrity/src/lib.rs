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
