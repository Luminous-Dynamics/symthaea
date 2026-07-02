// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge Integrity Zome
//!
//! Defines entry types for cross-hApp bridge records.

use hdi::prelude::*;

/// Entry types for the bridge zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A bridge record
    #[entry_type(visibility = "public")]
    BridgeRecord(BridgeRecord),
}

/// Link types for the bridge zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from source hApp to bridge records
    SourceToBridge,
    /// Link from target hApp to bridge records
    TargetToBridge,
    /// Link for all bridge records
    AllBridges,
}

/// A bridge record entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BridgeRecord {
    /// Source hApp identifier
    pub source_happ: String,
    /// Target hApp identifier
    pub target_happ: String,
    /// Record hash in the target hApp
    pub record_hash: String,
    /// Type of bridge (e.g., "identity", "reputation", "marketplace")
    pub bridge_type: String,
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
        EntryTypes::BridgeRecord(bridge) => validate_bridge(bridge),
    }
}

/// Validate a bridge record
fn validate_bridge(bridge: BridgeRecord) -> ExternResult<ValidateCallbackResult> {
    // Source hApp must not be empty
    if bridge.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp cannot be empty".to_string(),
        ));
    }

    // Target hApp must not be empty
    if bridge.target_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Target hApp cannot be empty".to_string(),
        ));
    }

    // Record hash must not be empty
    if bridge.record_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Record hash cannot be empty".to_string(),
        ));
    }

    // Bridge type must not be empty
    if bridge.bridge_type.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Bridge type cannot be empty".to_string(),
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
        LinkTypes::SourceToBridge => Ok(ValidateCallbackResult::Valid),
        LinkTypes::TargetToBridge => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllBridges => Ok(ValidateCallbackResult::Valid),
    }
}
