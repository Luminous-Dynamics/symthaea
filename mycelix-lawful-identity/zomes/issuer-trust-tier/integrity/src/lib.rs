#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// issuer-trust-tier integrity — three-tier classification for credential issuers.
// No tier ever influences Mycelix governance weight.

use hdi::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IssuerTier {
    /// State-backed identity issuers: `did:web:state.gov`, `did:web:gov.uk`, etc.
    /// User-configurable list; never canonical, never hardcoded.
    Sovereign,
    /// Regulated KYC/AML providers: `did:web:jumio.com`, `did:web:onfido.com`.
    /// User-configurable; requires explicit opt-in.
    RegulatedIntermediary,
    /// Default tier. All issuers start here.
    Peer,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IssuerClassification {
    /// Issuer DID string.
    pub issuer_did: String,
    /// Assigned tier.
    pub tier: IssuerTier,
    /// ISO 8601 classification timestamp.
    pub classified_at: String,
    /// Optional rationale (freeform, for audit trail).
    pub rationale: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    IssuerClassification(IssuerClassification),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Per-tier anchor → classifications at that tier.
    TierAnchor,
}
