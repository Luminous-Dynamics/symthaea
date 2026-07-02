// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supply Chain Claims Integrity Zome
//!
//! Defines entry types and validation rules for supply chain provenance claims.

use hdi::prelude::*;

/// Entry types for the claims zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A supply chain claim
    #[entry_type(visibility = "public")]
    SupplyChainClaim(SupplyChainClaim),
    /// A provider profile
    #[entry_type(visibility = "public")]
    ProviderProfile(ProviderProfile),
}

/// Link types for the claims zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from provider to their claims
    ProviderToClaims,
    /// Link from item_id to claims
    ItemToClaims,
    /// Link for all claims discovery
    AllClaims,
    /// Link from claim to verifications
    ClaimToVerifications,
}

/// A supply chain claim entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SupplyChainClaim {
    /// Unique item identifier
    pub item_id: String,
    /// Type of claim (e.g., "PRODUCED", "SHIPPED", "CERTIFIED")
    pub claim_type: String,
    /// Claim data as JSON string
    pub data: String,
    /// Issuer DID or public key
    pub issuer: String,
    /// Creation timestamp (microseconds since UNIX epoch)
    pub timestamp: u64,
}

/// A provider profile entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderProfile {
    /// Provider DID
    pub provider_did: String,
    /// Provider name
    pub name: String,
    /// Certifications
    pub certifications: Vec<String>,
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
        EntryTypes::SupplyChainClaim(claim) => validate_claim(claim),
        EntryTypes::ProviderProfile(profile) => validate_profile(profile),
    }
}

/// Validate a supply chain claim
fn validate_claim(claim: SupplyChainClaim) -> ExternResult<ValidateCallbackResult> {
    // Item ID must not be empty
    if claim.item_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Item ID cannot be empty".to_string(),
        ));
    }

    // Item ID length limit
    if claim.item_id.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Item ID cannot exceed 200 characters".to_string(),
        ));
    }

    // Claim type must not be empty
    if claim.claim_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim type cannot be empty".to_string(),
        ));
    }

    // Claim type length limit
    if claim.claim_type.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim type cannot exceed 100 characters".to_string(),
        ));
    }

    // Data length limit (10KB)
    if claim.data.len() > 10_000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim data cannot exceed 10KB".to_string(),
        ));
    }

    // Issuer must not be empty
    if claim.issuer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a provider profile
fn validate_profile(profile: ProviderProfile) -> ExternResult<ValidateCallbackResult> {
    // Provider DID must not be empty
    if profile.provider_did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider DID cannot be empty".to_string(),
        ));
    }

    // Name must not be empty
    if profile.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if profile.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Provider name cannot exceed 200 characters".to_string(),
        ));
    }

    // Certifications limit
    if profile.certifications.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many certifications (max 100)".to_string(),
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
        LinkTypes::ProviderToClaims => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ItemToClaims => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllClaims => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ClaimToVerifications => Ok(ValidateCallbackResult::Valid),
    }
}
