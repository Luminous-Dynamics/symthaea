// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anchor registry integrity zome: validates anchor nodes and certifications.

use hdi::prelude::*;
use mycelix_position_shared::{
    validate_geodetic, validate_node_id, AnchorCertification, AnchorNode,
};

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AnchorNode(AnchorNode),
    AnchorCertification(AnchorCertification),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Index anchors by geohash prefix for spatial queries.
    AnchorsByRegion,
    /// Link anchor to its certifications.
    AnchorToCertifications,
    /// Global index of all anchors.
    AllAnchors,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::AnchorNode(node) => validate_anchor_node(&node),
        EntryTypes::AnchorCertification(cert) => validate_certification(&cert),
    }
}

fn validate_anchor_node(node: &AnchorNode) -> ExternResult<ValidateCallbackResult> {
    if let Err(e) = validate_node_id(&node.node_id) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }
    if let Err(e) = validate_geodetic(node.latitude_deg, node.longitude_deg, node.altitude_m) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }
    if node.accuracy_m <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Accuracy must be positive, got {}",
            node.accuracy_m
        )));
    }
    if node.accuracy_m > 100_000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Accuracy > 100km is not useful for positioning".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_certification(cert: &AnchorCertification) -> ExternResult<ValidateCallbackResult> {
    if let Err(e) = validate_node_id(&cert.anchor_node_id) {
        return Ok(ValidateCallbackResult::Invalid(e));
    }
    if cert.verified_accuracy_m <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Verified accuracy must be positive".to_string(),
        ));
    }
    if cert.certification_method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Certification method cannot be empty".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
