#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// cross-did-zkp integrity — the bridge between primary and legal DIDs.
// The primary DID can prove "I control a legal DID that holds credential X
// from issuer Y" without revealing which legal DID. See
// ../../docs/THREAT_MODEL.md vectors 1 and 2.

use hdi::prelude::*;

/// A verifier's nonce request. The verifier publishes a fresh nonce; the prover
/// uses it exactly once. Reuse detection happens at verifier side (LRU window
/// of 65,536 most-recent nonces per verifier).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NonceRequest {
    /// Verifier DID (may be primary, legal, or external).
    pub verifier_did: String,
    /// 32-byte random nonce, base64-encoded.
    pub nonce_b64: String,
    /// ISO 8601 creation timestamp. Nonces older than 5 minutes SHOULD be rejected.
    pub created_at: String,
}

/// The cross-DID proof as presented to the verifier. Crucially, this structure
/// contains ONLY the fields the verifier needs — no legal DID string, no
/// pubkey of the legal DID, no deterministic hash of it.
///
/// Public inputs to the underlying STARK are exactly:
///   { issuer_public_key_hash, claim_predicate_hash, nonce_hash }.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossDidProof {
    /// Hash of the issuer's public key (permits verifier to look up issuer trust tier).
    pub issuer_pk_hash: String,
    /// Hash of the claim predicate being proven (e.g., "age>=21", "nationality=USA").
    pub claim_predicate_hash: String,
    /// Hash of the verifier-supplied nonce (prevents replay).
    pub nonce_hash: String,
    /// STARK proof bytes, multibase-encoded.
    pub proof_value: String,
    /// ISO 8601 generation timestamp.
    pub generated_at: String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    NonceRequest(NonceRequest),
    CrossDidProof(CrossDidProof),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Verifier → nonces they've issued (for reuse detection).
    VerifierToNonce,
}
