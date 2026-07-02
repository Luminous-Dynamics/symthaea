// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim Verification Integrity Zome
//!
//! Defines entry types and validation rules for claim verification.

use hdi::prelude::*;

/// Entry types for the verification zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A claim verification
    #[entry_type(visibility = "public")]
    ClaimVerification(ClaimVerification),
}

/// Link types for the verification zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from verifier to verifications
    VerifierToVerifications,
    /// Link for all verifications
    AllVerifications,
}

/// Verification status enum
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    Pending,
    Verified,
    Rejected,
}

/// A claim verification entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ClaimVerification {
    /// Hash of the claim being verified
    pub claim_hash: ActionHash,
    /// Verifier DID or public key
    pub verifier: String,
    /// Verification status
    pub status: VerificationStatus,
    /// Verification timestamp (microseconds since UNIX epoch)
    pub verified_at: u64,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::ClaimVerification(verification) => validate_verification(verification),
    }
}

/// Validate a claim verification
fn validate_verification(verification: ClaimVerification) -> ExternResult<ValidateCallbackResult> {
    // Verifier must not be empty
    if verification.verifier.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::VerifierToVerifications => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllVerifications => Ok(ValidateCallbackResult::Valid),
    }
}
