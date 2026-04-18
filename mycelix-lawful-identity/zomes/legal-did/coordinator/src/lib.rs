#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// legal-did coordinator — manages state-facing DID lifecycle.
// Stub scaffold; real implementation lands under task #10.

use hdk::prelude::*;
use legal_did_integrity::{LegalCredentialRecord, LegalDid};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateLegalDidInput {
    pub label: Option<String>,
}

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

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("legal_did:pong".to_string())
}

// Full coordinator surface (create_legal_did, import_credential,
// list_my_legal_dids, revoke_legal_did, etc.) is implemented under task #10.
// See the plan at /home/tstoltz/.claude-account2/plans/would-you-like-to-parallel-fox.md
