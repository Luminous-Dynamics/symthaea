#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// cross-did-zkp coordinator — nonce issuance + proof verification.
// Stub scaffold; real implementation lands under task #13.

use hdk::prelude::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestNonceInput {
    pub verifier_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitProofInput {
    pub issuer_pk_hash: String,
    pub claim_predicate_hash: String,
    pub nonce_hash: String,
    pub proof_value: String,
}

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("cross_did_zkp:pong".to_string())
}

// Real surface (request_nonce, submit_proof, verify_proof, check_nonce_reuse)
// lands under task #13. Nonce-reuse detection must reject any proof whose
// nonce_hash has appeared in the last 65,536 verifications (LRU window).
