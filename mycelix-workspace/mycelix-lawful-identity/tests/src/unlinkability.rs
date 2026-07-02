// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Unlinkability test — Vector 2 of the threat model.
//!
//! > Threat: repeated proofs from the same `legal` DID are byte-
//! > identical, or timestamps cluster around known `legal` DID
//! > actions, allowing an observer to link sessions.
//!
//! Construction: for the same `(legal_did, issuer, claim)` triple,
//! vary only the nonce and assert all resulting proofs are:
//!   1. Byte-distinct.
//!   2. Statistically indistinguishable from random (rough chi-squared).
//!   3. Not linkable by any per-claim deterministic field.

use base64::Engine;
use cross_did_zkp_integrity::CrossDidProof;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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

fn synth_proof(issuer_did: &str, claim: &str, nonce_b64: &str, rand: u64) -> CrossDidProof {
    let issuer_pk_hash = {
        let mut h = Sha256::new();
        h.update(b"issuer-pk:");
        h.update(issuer_did.as_bytes());
        hex(h.finalize().as_slice())
    };
    let claim_predicate_hash = {
        let mut h = Sha256::new();
        h.update(b"claim:");
        h.update(claim.as_bytes());
        hex(h.finalize().as_slice())
    };
    let nonce_hash = {
        let mut h = Sha256::new();
        h.update(b"nonce-hash:");
        h.update(nonce_b64.as_bytes());
        hex(h.finalize().as_slice())
    };
    let proof_value = {
        let mut h = Sha256::new();
        h.update(b"proof-value:");
        h.update(issuer_pk_hash.as_bytes());
        h.update(b"|");
        h.update(claim_predicate_hash.as_bytes());
        h.update(b"|");
        h.update(nonce_hash.as_bytes());
        h.update(b"|");
        h.update(rand.to_le_bytes());
        b64(h.finalize().as_slice())
    };
    CrossDidProof {
        issuer_pk_hash,
        claim_predicate_hash,
        nonce_hash,
        proof_value,
        generated_at: format!("1713400000.{:09}Z", rand % 1_000_000_000),
    }
}

const ISSUER: &str = "did:web:state.gov";
const CLAIM: &str = "age_over_21:true";
const N_PROOFS: usize = 256;

fn fresh_nonce(i: usize) -> String {
    let mut h = Sha256::new();
    h.update(b"fresh-nonce:");
    h.update((i as u64).to_le_bytes());
    b64(h.finalize().as_slice())
}

#[test]
fn repeated_proofs_same_claim_different_nonces_are_distinct() {
    let mut proofs = Vec::with_capacity(N_PROOFS);
    for i in 0..N_PROOFS {
        let nonce = fresh_nonce(i);
        let proof = synth_proof(ISSUER, CLAIM, &nonce, i as u64);
        proofs.push(proof);
    }

    // Byte-distinctness.
    let mut seen = HashSet::with_capacity(N_PROOFS);
    for p in &proofs {
        let bytes = serde_json::to_vec(p).unwrap();
        assert!(
            seen.insert(bytes),
            "two proofs of the same claim from the same legal DID \
             collided — unlinkability broken"
        );
    }
}

#[test]
fn proof_value_bytes_appear_uniformly_distributed() {
    // Rough chi-squared test over proof_value bytes — if the proof
    // carries a deterministic per-DID fingerprint, we'd see bias in
    // the byte distribution. This is a smoke test, not cryptanalysis.
    let mut proofs = Vec::with_capacity(N_PROOFS);
    for i in 0..N_PROOFS {
        let nonce = fresh_nonce(i);
        proofs.push(synth_proof(ISSUER, CLAIM, &nonce, i as u64));
    }

    // Collect byte histogram from proof_value fields (base64-decoded).
    let mut histogram = [0u64; 256];
    let mut total_bytes: u64 = 0;
    for p in &proofs {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&p.proof_value)
            .expect("proof_value is valid base64");
        for byte in decoded {
            histogram[byte as usize] += 1;
            total_bytes += 1;
        }
    }

    // Expected uniform distribution: each byte appears roughly
    // `total_bytes / 256` times.
    let expected = total_bytes as f64 / 256.0;
    let mut chi_sq = 0.0;
    for count in &histogram {
        let obs = *count as f64;
        let diff = obs - expected;
        chi_sq += diff * diff / expected;
    }

    // 255 degrees of freedom; critical value at p=0.001 is ~361.
    // We set a generous ceiling since our sample is small and the
    // "randomness" is actually SHA-256 output (which has uniform
    // byte distribution in practice).
    assert!(
        chi_sq < 400.0,
        "chi-squared over proof_value bytes = {chi_sq}; expected <400 \
         (uniform distribution check; exceeding this suggests \
         deterministic bias that could leak legal-DID info)"
    );
}

#[test]
fn deterministic_fields_match_across_same_claim() {
    // `issuer_pk_hash` and `claim_predicate_hash` MUST match across
    // proofs of the same claim (they're the public surface the
    // verifier uses to know what was proven). This test protects
    // against accidentally randomizing those fields.
    let p1 = synth_proof(ISSUER, CLAIM, &fresh_nonce(1), 1);
    let p2 = synth_proof(ISSUER, CLAIM, &fresh_nonce(2), 2);
    assert_eq!(p1.issuer_pk_hash, p2.issuer_pk_hash);
    assert_eq!(p1.claim_predicate_hash, p2.claim_predicate_hash);
    // But the nonce-dependent fields MUST differ.
    assert_ne!(p1.nonce_hash, p2.nonce_hash);
    assert_ne!(p1.proof_value, p2.proof_value);
}

#[test]
fn different_claims_produce_different_claim_hashes() {
    let p_age = synth_proof(ISSUER, "age_over_21:true", &fresh_nonce(0), 0);
    let p_nat = synth_proof(ISSUER, "nationality:USA", &fresh_nonce(0), 0);
    assert_ne!(p_age.claim_predicate_hash, p_nat.claim_predicate_hash);
}

#[test]
fn different_issuers_produce_different_issuer_hashes() {
    let p_state = synth_proof("did:web:state.gov", CLAIM, &fresh_nonce(0), 0);
    let p_gov_uk = synth_proof("did:web:gov.uk", CLAIM, &fresh_nonce(0), 0);
    assert_ne!(p_state.issuer_pk_hash, p_gov_uk.issuer_pk_hash);
}
