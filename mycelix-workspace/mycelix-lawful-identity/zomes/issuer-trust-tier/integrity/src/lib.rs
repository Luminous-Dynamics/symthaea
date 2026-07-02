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

impl IssuerTier {
    pub fn as_str(self) -> &'static str {
        match self {
            IssuerTier::Sovereign => "sovereign",
            IssuerTier::RegulatedIntermediary => "regulated_intermediary",
            IssuerTier::Peer => "peer",
        }
    }
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
    /// The anchor is a Path keyed by tier string (e.g., "tier/sovereign").
    TierAnchor,
    /// Per-issuer anchor → classification history for that issuer.
    /// The anchor is a Path keyed by issuer DID (e.g., "issuer/did:web:state.gov").
    IssuerAnchor,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::IssuerClassification(entry) => validate_classification(&entry),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_classification(entry: &IssuerClassification) -> ExternResult<ValidateCallbackResult> {
    if entry.issuer_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "issuer_did empty".to_string(),
        ));
    }
    if entry.classified_at.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "classified_at empty".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
