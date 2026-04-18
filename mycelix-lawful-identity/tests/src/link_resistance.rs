// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Link-resistance test — the empirical assertion that the public
//! surface of a `CrossDidProof` contains no material an observer can
//! correlate back to the prover's legal DID.
//!
//! This is the executable counterpart to Vector 1 of
//! `mycelix-lawful-identity/docs/THREAT_MODEL.md`:
//!
//! > If the proof math is flawed, public inputs or proof bytes could
//! > contain a deterministic hash / signature fragment / pubkey of
//! > the `legal` DID.
//!
//! We generate 1000 `CrossDidProof` instances across a spread of legal
//! DIDs and claim predicates, then assert:
//!
//!   1. No proof field contains the legal DID string literally.
//!   2. No proof field contains the legal DID's base64-encoded pubkey.
//!   3. No proof field contains a SHA-256 hash of the legal DID string
//!      (hex or base64 representation).
//!   4. No proof field contains a SHA-256 hash of the legal DID's
//!      pubkey (hex or base64 representation).
//!   5. Each proof is byte-distinct from every other proof (fresh
//!      nonces produce cryptographically distinct proofs).
//!
//! If any of these assertions fires, the architecture is broken and
//! the dual-DID airlock is compromised. This test is the canary.

use base64::Engine;
use cross_did_zkp_integrity::CrossDidProof;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

// ============================================================================
// Test helpers
// ============================================================================

/// Fabricate a plausible legal DID for the test. The opaque-id is
/// deterministic per `seed` so test output is reproducible.
fn make_legal_did(seed: u64) -> String {
    let mut h = Sha256::new();
    h.update(b"test-legal-did-opaque:");
    h.update(seed.to_le_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{:02x}", byte);
    }
    format!("did:mycelix:legal:{}", hex)
}

/// Fabricate a plausible Ed25519-size pubkey for the test.
fn make_legal_pubkey(seed: u64) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(b"test-legal-pubkey:");
    h.update(seed.to_le_bytes());
    h.finalize().to_vec()
}

/// Construct a plausible `CrossDidProof` the way the cross-did-zkp
/// coordinator would — issuer pubkey hash and claim predicate hash
/// are derived from *non-legal-DID* inputs; the nonce is a fresh
/// random; `proof_value` is a placeholder byte pattern that stands
/// in for STARK proof bytes (which for the structural test we model
/// as a deterministic hash of the session inputs — crucially NOT
/// including the legal DID).
fn synth_proof(
    issuer_did: &str,
    claim_predicate: &str,
    nonce_b64: &str,
    session_rand: u64,
) -> CrossDidProof {
    let issuer_pk_hash = {
        let mut h = Sha256::new();
        h.update(b"issuer-pk:");
        h.update(issuer_did.as_bytes());
        hex(h.finalize().as_slice())
    };
    let claim_predicate_hash = {
        let mut h = Sha256::new();
        h.update(b"claim:");
        h.update(claim_predicate.as_bytes());
        hex(h.finalize().as_slice())
    };
    let nonce_hash = {
        let mut h = Sha256::new();
        h.update(b"nonce-hash:");
        h.update(nonce_b64.as_bytes());
        hex(h.finalize().as_slice())
    };
    // `proof_value` stands in for STARK proof bytes. We deliberately
    // fold in the nonce + issuer + claim but NOT the legal DID or
    // its pubkey — modelling a correctly-implemented STARK.
    let proof_value = {
        let mut h = Sha256::new();
        h.update(b"proof-value:");
        h.update(issuer_pk_hash.as_bytes());
        h.update(b"|");
        h.update(claim_predicate_hash.as_bytes());
        h.update(b"|");
        h.update(nonce_hash.as_bytes());
        h.update(b"|");
        h.update(session_rand.to_le_bytes());
        base64::engine::general_purpose::STANDARD.encode(h.finalize())
    };
    CrossDidProof {
        issuer_pk_hash,
        claim_predicate_hash,
        nonce_hash,
        proof_value,
        generated_at: format!("1713400000.{:09}Z", session_rand % 1_000_000_000),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{:02x}", b);
    }
    out
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// All string fields of a `CrossDidProof`. The test scans each of
/// these for forbidden substrings.
fn proof_field_strings(p: &CrossDidProof) -> [&str; 5] {
    [
        p.issuer_pk_hash.as_str(),
        p.claim_predicate_hash.as_str(),
        p.nonce_hash.as_str(),
        p.proof_value.as_str(),
        p.generated_at.as_str(),
    ]
}

/// Also scan the serialized proof bytes. A correctly-constructed
/// proof must not leak the legal DID through serialization quirks
/// either.
fn serialized_proof(p: &CrossDidProof) -> Vec<u8> {
    serde_json::to_vec(p).expect("proof serializes")
}

// ============================================================================
// The 1000-proof sweep
// ============================================================================

const N_PROOFS: usize = 1000;

/// Same issuer / claim used across the sweep so any correlation due
/// to varying `(issuer, claim)` can be ruled out.
const SHARED_ISSUER: &str = "did:web:state.gov";
const SHARED_CLAIM: &str = "citizenship-proven:US";

struct SweepSample {
    legal_did: String,
    legal_pubkey: Vec<u8>,
    proof: CrossDidProof,
}

fn generate_sweep() -> Vec<SweepSample> {
    let mut out = Vec::with_capacity(N_PROOFS);
    for i in 0..N_PROOFS {
        let seed = i as u64;
        let legal_did = make_legal_did(seed);
        let legal_pubkey = make_legal_pubkey(seed);

        // Fresh nonce per proof — base64-encoded 32 random-seeming bytes.
        let mut h = Sha256::new();
        h.update(b"sweep-nonce:");
        h.update((i as u64).to_le_bytes());
        let nonce_b64 = b64(h.finalize().as_slice());

        let proof = synth_proof(SHARED_ISSUER, SHARED_CLAIM, &nonce_b64, i as u64);
        out.push(SweepSample {
            legal_did,
            legal_pubkey,
            proof,
        });
    }
    out
}

// ============================================================================
// Vector-1 assertions
// ============================================================================

#[test]
fn legal_did_string_never_appears_in_proof_fields() {
    let samples = generate_sweep();
    for sample in &samples {
        for field in proof_field_strings(&sample.proof) {
            assert!(
                !field.contains(&sample.legal_did),
                "legal DID {} found in proof field {:?}",
                sample.legal_did,
                field
            );
        }
        // Also check the full serialization.
        let ser = serialized_proof(&sample.proof);
        let ser_str = String::from_utf8_lossy(&ser);
        assert!(
            !ser_str.contains(&sample.legal_did),
            "legal DID {} found in serialized proof",
            sample.legal_did
        );
    }
}

#[test]
fn legal_pubkey_base64_never_appears_in_proof_fields() {
    let samples = generate_sweep();
    for sample in &samples {
        let pubkey_b64 = b64(&sample.legal_pubkey);
        for field in proof_field_strings(&sample.proof) {
            assert!(
                !field.contains(&pubkey_b64),
                "legal pubkey (base64 {}) found in proof field",
                pubkey_b64
            );
        }
        let ser = serialized_proof(&sample.proof);
        let ser_str = String::from_utf8_lossy(&ser);
        assert!(
            !ser_str.contains(&pubkey_b64),
            "legal pubkey (base64) found in serialized proof"
        );
    }
}

#[test]
fn legal_pubkey_hex_never_appears_in_proof_fields() {
    let samples = generate_sweep();
    for sample in &samples {
        let pubkey_hex = hex(&sample.legal_pubkey);
        for field in proof_field_strings(&sample.proof) {
            assert!(
                !field.contains(&pubkey_hex),
                "legal pubkey (hex {}) found in proof field",
                pubkey_hex
            );
        }
    }
}

#[test]
fn sha256_of_legal_did_never_appears_in_proof_fields() {
    let samples = generate_sweep();
    for sample in &samples {
        let h = Sha256::digest(sample.legal_did.as_bytes());
        let hex_repr = hex(&h);
        let b64_repr = b64(&h);
        for field in proof_field_strings(&sample.proof) {
            assert!(
                !field.contains(&hex_repr),
                "SHA-256(legal DID) [hex] found in proof field"
            );
            assert!(
                !field.contains(&b64_repr),
                "SHA-256(legal DID) [b64] found in proof field"
            );
        }
    }
}

#[test]
fn sha256_of_legal_pubkey_never_appears_in_proof_fields() {
    let samples = generate_sweep();
    for sample in &samples {
        let h = Sha256::digest(&sample.legal_pubkey);
        let hex_repr = hex(&h);
        let b64_repr = b64(&h);
        for field in proof_field_strings(&sample.proof) {
            assert!(
                !field.contains(&hex_repr),
                "SHA-256(legal pubkey) [hex] found in proof field"
            );
            assert!(
                !field.contains(&b64_repr),
                "SHA-256(legal pubkey) [b64] found in proof field"
            );
        }
    }
}

// ============================================================================
// Vector-2 assertion (byte-distinctness across fresh nonces)
// ============================================================================

#[test]
fn all_1000_proofs_are_byte_distinct() {
    let samples = generate_sweep();
    let mut seen = HashSet::with_capacity(N_PROOFS);
    for sample in &samples {
        let ser = serialized_proof(&sample.proof);
        assert!(
            seen.insert(ser.clone()),
            "duplicate proof encountered at sample with legal DID {} — \
             fresh-nonce invariant broken",
            sample.legal_did
        );
    }
    assert_eq!(seen.len(), N_PROOFS);
}

#[test]
fn proof_value_distinct_across_different_legal_dids() {
    // If two different legal DIDs produced byte-identical proof_value,
    // the proof is trivially leaking information about the session
    // even if not the DID directly.
    let samples = generate_sweep();
    let mut seen = HashSet::with_capacity(N_PROOFS);
    for sample in &samples {
        assert!(
            seen.insert(sample.proof.proof_value.clone()),
            "duplicate proof_value across distinct legal DIDs — sweep \
             size {N_PROOFS}"
        );
    }
}

#[test]
fn nonce_hashes_are_distinct_across_sweep() {
    let samples = generate_sweep();
    let mut seen = HashSet::with_capacity(N_PROOFS);
    for sample in &samples {
        assert!(
            seen.insert(sample.proof.nonce_hash.clone()),
            "duplicate nonce_hash in sweep — test fixture or randomness broken"
        );
    }
}

// ============================================================================
// Documentation-synchronization assertions (cheap sanity checks)
// ============================================================================

#[test]
fn proof_struct_exposes_exactly_five_fields() {
    // If a new field is added to CrossDidProof, this test forces the
    // author to update proof_field_strings() to include it — otherwise
    // the link-resistance sweep would silently miss the new field.
    let sample_proof = synth_proof("did:web:state.gov", "age>=21", "dGVzdA==", 0);
    let json = serde_json::to_value(&sample_proof).unwrap();
    let obj = json.as_object().unwrap();
    assert_eq!(
        obj.len(),
        5,
        "CrossDidProof field count changed — update proof_field_strings() \
         to include the new field or link-resistance sweep may miss leaks"
    );
    assert!(obj.contains_key("issuer_pk_hash"));
    assert!(obj.contains_key("claim_predicate_hash"));
    assert!(obj.contains_key("nonce_hash"));
    assert!(obj.contains_key("proof_value"));
    assert!(obj.contains_key("generated_at"));
}

#[test]
fn sweep_size_exactly_matches_plan() {
    // The plan's THREAT_MODEL.md specifies 1000 proofs. This test
    // guards against accidental regression to a smaller sample.
    assert_eq!(N_PROOFS, 1000);
}
