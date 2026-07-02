#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// cross-did-zkp coordinator — nonce issuance + proof submission + replay check.
//
// The zome stores the state a verifier needs to detect replay
// (VerifierToNonce links anchor the nonces a given verifier has
// issued; NonceHashToProof links anchor the proofs consumed by a
// given nonce). The actual cryptographic verification of STARK proof
// bytes happens client-side — the zome is only the coordination
// surface.
//
// Replay detection: when a verifier sees a proof, it (a) checks the
// proof's nonce_hash does not already have a CrossDidProof linked
// from it, and (b) checks the underlying nonce was actually issued
// by this verifier (otherwise someone replayed a proof from a
// different session). Both checks are cheap DHT queries.

use cross_did_zkp_integrity::{CrossDidProof, EntryTypes, LinkTypes, NonceRequest};
use hdk::prelude::*;

// ============================================================================
// Helpers
// ============================================================================

const NONCE_BYTES: usize = 32;

fn now_iso_8601() -> ExternResult<String> {
    let ts = sys_time()?;
    let (secs, nanos) = ts.as_seconds_and_nanos();
    Ok(format!("{}.{:09}Z", secs, nanos))
}

/// Anchor for a verifier's issued nonces.
fn verifier_path(verifier_did: &str) -> Path {
    Path::from(format!("verifier/{}", verifier_did))
}

/// Anchor for a nonce-hash's consumed proofs.
fn nonce_hash_path(nonce_hash: &str) -> Path {
    Path::from(format!("nonce_hash/{}", nonce_hash))
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};
    STANDARD.encode(bytes)
}

// ============================================================================
// Request nonce
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestNonceInput {
    pub verifier_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestNonceOutput {
    /// 32-byte random nonce, base64-encoded. Prover uses this exactly once.
    pub nonce_b64: String,
    /// Action hash of the stored NonceRequest entry.
    pub action_hash: ActionHash,
}

#[hdk_extern]
pub fn request_nonce(input: RequestNonceInput) -> ExternResult<RequestNonceOutput> {
    if input.verifier_did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "verifier_did must not be empty".to_string()
        )));
    }

    let random = random_bytes(NONCE_BYTES as u32)?;
    let nonce_b64 = base64_encode(&random.into_vec());

    let entry = NonceRequest {
        verifier_did: input.verifier_did.clone(),
        nonce_b64: nonce_b64.clone(),
        created_at: now_iso_8601()?,
    };
    let ah = create_entry(&EntryTypes::NonceRequest(entry))?;

    let anchor = verifier_path(&input.verifier_did).path_entry_hash()?;
    create_link(
        anchor,
        ah.clone(),
        LinkTypes::VerifierToNonce,
        LinkTag::new(nonce_b64.as_bytes().to_vec()),
    )?;

    Ok(RequestNonceOutput {
        nonce_b64,
        action_hash: ah,
    })
}

// ============================================================================
// Submit proof
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitProofInput {
    pub issuer_pk_hash: String,
    pub claim_predicate_hash: String,
    pub nonce_hash: String,
    pub proof_value: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitProofOutput {
    pub action_hash: ActionHash,
    /// True if this nonce_hash had never been consumed before. If false,
    /// verifier MUST reject the proof (replay attempt).
    pub nonce_was_fresh: bool,
}

#[hdk_extern]
pub fn submit_proof(input: SubmitProofInput) -> ExternResult<SubmitProofOutput> {
    if input.nonce_hash.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "nonce_hash must not be empty".to_string()
        )));
    }

    // Check if any CrossDidProof already claims this nonce_hash.
    let nonce_anchor = nonce_hash_path(&input.nonce_hash).path_entry_hash()?;
    let existing = get_links(
        LinkQuery::try_new(nonce_anchor.clone(), LinkTypes::NonceHashToProof)?,
        GetStrategy::Local,
    )?;
    let nonce_was_fresh = existing.is_empty();

    let entry = CrossDidProof {
        issuer_pk_hash: input.issuer_pk_hash,
        claim_predicate_hash: input.claim_predicate_hash,
        nonce_hash: input.nonce_hash.clone(),
        proof_value: input.proof_value,
        generated_at: now_iso_8601()?,
    };
    let ah = create_entry(&EntryTypes::CrossDidProof(entry))?;

    // Link from the nonce_hash anchor so future submits see this consumed.
    create_link(
        nonce_anchor,
        ah.clone(),
        LinkTypes::NonceHashToProof,
        LinkTag::new(Vec::<u8>::new()),
    )?;

    Ok(SubmitProofOutput {
        action_hash: ah,
        nonce_was_fresh,
    })
}

// ============================================================================
// Check nonce freshness (verifier-side lookup helper)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckNonceFreshInput {
    pub nonce_hash: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckNonceFreshOutput {
    pub fresh: bool,
    pub consumed_proof_count: usize,
}

#[hdk_extern]
pub fn check_nonce_fresh(input: CheckNonceFreshInput) -> ExternResult<CheckNonceFreshOutput> {
    let anchor = nonce_hash_path(&input.nonce_hash).path_entry_hash()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::NonceHashToProof)?,
        GetStrategy::Local,
    )?;
    let count = links.len();
    Ok(CheckNonceFreshOutput {
        fresh: count == 0,
        consumed_proof_count: count,
    })
}

// ============================================================================
// List nonces issued by a verifier
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListVerifierNoncesInput {
    pub verifier_did: String,
    /// Cap on how many to return; the DHT link query itself may return
    /// more and the coordinator truncates after hydration.
    pub limit: Option<u32>,
}

#[hdk_extern]
pub fn list_verifier_nonces(input: ListVerifierNoncesInput) -> ExternResult<Vec<NonceRequest>> {
    let anchor = verifier_path(&input.verifier_did).path_entry_hash()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::VerifierToNonce)?,
        GetStrategy::Local,
    )?;

    let limit = input.limit.unwrap_or(1024) as usize;
    let mut out = Vec::with_capacity(links.len().min(limit));
    for link in links.into_iter().take(limit) {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Ok(Some(entry)) = record.entry().to_app_option::<NonceRequest>() {
                out.push(entry);
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Ping
// ============================================================================

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("cross_did_zkp:pong".to_string())
}
