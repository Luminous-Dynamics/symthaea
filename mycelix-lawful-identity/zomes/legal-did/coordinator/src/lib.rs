#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// legal-did coordinator — manages state-facing DID lifecycle.
//
// A legal DID is the address under which a user holds government-issued
// credentials (passport, mDL, SSN-derived attestation). Each agent may
// control any number of legal DIDs; they are addressable only via links
// from the creating agent's pubkey, never indexed by any attribute that
// could cross-link to the primary consciousness-gated identity.
//
// The DID string is of form `did:mycelix:legal:<32-byte-hex-opaque-id>`.
// The opaque id is generated from cryptographic randomness inside the
// zome and is the only thing distinguishing one legal DID from another;
// there is no deterministic derivation from the agent key (which would
// leak linkability).
//
// See mycelix-lawful-identity/docs/THREAT_MODEL.md for the threat
// model this isolation addresses.

use hdk::prelude::*;
use legal_did_integrity::{EntryTypes, LegalCredentialRecord, LegalDid, LinkTypes};

// ============================================================================
// Constants
// ============================================================================

const DID_PREFIX: &str = "did:mycelix:legal:";
const OPAQUE_ID_BYTES: usize = 32;

// ============================================================================
// Helpers
// ============================================================================

/// Generate an opaque identifier for a new legal DID. Uses the HDK's
/// random source so an agent's legal DIDs cannot be deterministically
/// derived from its pubkey (which would make them linkable).
fn generate_opaque_id() -> ExternResult<String> {
    let bytes = random_bytes(OPAQUE_ID_BYTES as u32)?;
    let mut out = String::with_capacity(OPAQUE_ID_BYTES * 2);
    for byte in bytes.into_vec() {
        use std::fmt::Write;
        let _ = write!(out, "{:02x}", byte);
    }
    Ok(out)
}

/// Get the calling agent's pubkey, used as the anchor for per-agent links.
fn caller() -> ExternResult<AgentPubKey> {
    agent_info().map(|info| info.agent_initial_pubkey)
}

/// Current wall-clock time as ISO 8601 (best effort; Holochain's Timestamp
/// is micros since epoch, we emit milliseconds truncated).
fn now_iso_8601() -> ExternResult<String> {
    let ts = sys_time()?;
    let secs = ts.as_seconds_and_nanos().0;
    let nanos = ts.as_seconds_and_nanos().1;
    // Just emit the epoch millis as a canonical string — a browser/CLI
    // caller can render whatever locale it likes.
    Ok(format!("{}.{:09}Z", secs, nanos))
}

// ============================================================================
// Create
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateLegalDidInput {
    pub label: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateLegalDidOutput {
    pub did: String,
    pub action_hash: ActionHash,
}

#[hdk_extern]
pub fn create_legal_did(input: CreateLegalDidInput) -> ExternResult<CreateLegalDidOutput> {
    let opaque = generate_opaque_id()?;
    let did = format!("{DID_PREFIX}{opaque}");

    let entry = LegalDid {
        did: did.clone(),
        created_at: now_iso_8601()?,
        label: input.label,
    };

    let action_hash = create_entry(&EntryTypes::LegalDid(entry.clone()))?;

    // Link from agent → this DID so we can list our own DIDs later.
    let agent_pk = caller()?;
    create_link(
        agent_pk,
        action_hash.clone(),
        LinkTypes::AgentToLegalDid,
        // No link tag — we don't want a searchable index on the DID string
        // (that would defeat isolation).
        LinkTag::new(Vec::<u8>::new()),
    )?;

    Ok(CreateLegalDidOutput { did, action_hash })
}

// ============================================================================
// List
// ============================================================================

#[hdk_extern]
pub fn list_my_legal_dids(_: ()) -> ExternResult<Vec<LegalDid>> {
    let agent_pk = caller()?;
    let links = get_links(
        LinkQuery::try_new(agent_pk, LinkTypes::AgentToLegalDid)?,
        GetStrategy::Local,
    )?;

    let mut out = Vec::with_capacity(links.len());
    for link in links {
        let ah: ActionHash = link.target.try_into().map_err(|_| {
            wasm_error!(WasmErrorInner::Guest(
                "link target was not an ActionHash".to_string()
            ))
        })?;
        if let Some(record) = get(ah, GetOptions::default())? {
            let entry = record
                .entry()
                .to_app_option::<LegalDid>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
            if let Some(did) = entry {
                out.push(did);
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Import credential
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ImportCredentialInput {
    pub legal_did: String,
    pub credential_hash: String,
    pub issuer_did: String,
    pub credential_type: String,
    pub issued_at: String,
    pub expires_at: Option<String>,
    pub revocation_check_url: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ImportCredentialOutput {
    pub credential_hash: String,
    pub record_action_hash: ActionHash,
}

#[hdk_extern]
pub fn import_credential(input: ImportCredentialInput) -> ExternResult<ImportCredentialOutput> {
    // Validate caller owns the legal DID they're attaching this credential to.
    let owned_dids = list_my_legal_dids(())?;
    if !owned_dids.iter().any(|d| d.did == input.legal_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "caller does not own legal DID {}",
            input.legal_did
        ))));
    }

    // Find the LegalDid's action hash to link from.
    let agent_pk = caller()?;
    let links = get_links(
        LinkQuery::try_new(agent_pk, LinkTypes::AgentToLegalDid)?,
        GetStrategy::Local,
    )?;
    let mut did_ah: Option<ActionHash> = None;
    for link in links {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Ok(Some(did)) = record.entry().to_app_option::<LegalDid>() {
                if did.did == input.legal_did {
                    did_ah = Some(ah);
                    break;
                }
            }
        }
    }
    let did_ah = did_ah.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "legal DID action hash not found (consistency error)".to_string()
        ))
    })?;

    let record = LegalCredentialRecord {
        credential_hash: input.credential_hash.clone(),
        issuer_did: input.issuer_did,
        credential_type: input.credential_type,
        issued_at: input.issued_at,
        expires_at: input.expires_at,
        revocation_check_url: input.revocation_check_url,
    };

    let record_ah = create_entry(&EntryTypes::LegalCredentialRecord(record))?;
    create_link(
        did_ah,
        record_ah.clone(),
        LinkTypes::LegalDidToCredential,
        LinkTag::new(Vec::<u8>::new()),
    )?;

    Ok(ImportCredentialOutput {
        credential_hash: input.credential_hash,
        record_action_hash: record_ah,
    })
}

// ============================================================================
// List credentials for a DID
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetCredentialsForDidInput {
    pub legal_did: String,
}

#[hdk_extern]
pub fn get_credentials_for_did(
    input: GetCredentialsForDidInput,
) -> ExternResult<Vec<LegalCredentialRecord>> {
    // Caller must own the DID — matching the create path's isolation rule.
    let owned = list_my_legal_dids(())?;
    let did = owned
        .iter()
        .find(|d| d.did == input.legal_did)
        .ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "caller does not own this legal DID or it does not exist".to_string()
            ))
        })?;

    let agent_pk = caller()?;
    let links = get_links(
        LinkQuery::try_new(agent_pk.clone(), LinkTypes::AgentToLegalDid)?,
        GetStrategy::Local,
    )?;

    // Find the action hash matching `did`.
    let mut did_ah: Option<ActionHash> = None;
    for link in links {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Ok(Some(entry)) = record.entry().to_app_option::<LegalDid>() {
                if entry.did == did.did {
                    did_ah = Some(ah);
                    break;
                }
            }
        }
    }
    let did_ah = did_ah.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(
            "legal DID action hash not found".to_string()
        ))
    })?;

    let cred_links = get_links(
        LinkQuery::try_new(did_ah, LinkTypes::LegalDidToCredential)?,
        GetStrategy::Local,
    )?;
    let mut out = Vec::with_capacity(cred_links.len());
    for link in cred_links {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Ok(Some(cred)) = record.entry().to_app_option::<LegalCredentialRecord>() {
                out.push(cred);
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Ping (retained for compatibility with the scaffold smoke path)
// ============================================================================

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("legal_did:pong".to_string())
}
