// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Position estimates integrity zome: validates fused position entries.

use hdi::prelude::*;
use mycelix_position_shared::{validate_geodetic, validate_node_id, PositionEstimateEntry};

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PositionEstimate(PositionEstimateEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    EstimatesByNode,
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
        EntryTypes::PositionEstimate(est) => {
            if let Err(e) = validate_node_id(&est.node_id) {
                return Ok(ValidateCallbackResult::Invalid(e));
            }
            if let Err(e) = validate_geodetic(est.latitude_deg, est.longitude_deg, est.altitude_m) {
                return Ok(ValidateCallbackResult::Invalid(e));
            }
            if est.covariance.len() != 9 {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Covariance must have 9 elements (3×3), got {}",
                    est.covariance.len()
                )));
            }
            // Check diagonal is positive (valid covariance)
            if est.covariance[0] < 0.0 || est.covariance[4] < 0.0 || est.covariance[8] < 0.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Covariance diagonal must be non-negative".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}
