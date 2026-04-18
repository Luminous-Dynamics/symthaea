#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// issuer-trust-tier coordinator — classify credential issuers into tiers.
// Stub scaffold; real implementation lands under task #11.

use hdk::prelude::*;
use issuer_trust_tier_integrity::{IssuerClassification, IssuerTier};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassifyIssuerInput {
    pub issuer_did: String,
    pub tier: IssuerTier,
    pub rationale: Option<String>,
}

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("issuer_trust_tier:pong".to_string())
}

// Real surface (classify_issuer, lookup_tier, list_by_tier, revoke_classification)
// lands under task #11.
