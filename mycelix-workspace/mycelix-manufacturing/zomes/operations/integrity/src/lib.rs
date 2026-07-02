// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Operations Integrity Zome
//!
//! Entry types and validation for manufacturing operations and routing sequences.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OperationEntry {
    pub name: String,
    pub description: String,
    pub machine_type: String,
    pub setup_time_min: u32,
    pub cycle_time_min: u32,
    pub tooling: Option<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoutingEntry {
    pub design_id: String,
    pub revision: String,
    pub steps: Vec<RoutingStepEntry>,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, SerializedBytes)]
pub struct RoutingStepEntry {
    pub sequence: u32,
    pub operation_name: String,
    pub machine_type: String,
    pub setup_time_min: u32,
    pub cycle_time_min: u32,
    pub description: String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Operation(OperationEntry),
    Routing(RoutingEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllOperations,
    DesignToRouting,
    RoutingToOperations,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Operation(op_entry) => {
                if op_entry.name.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "operation name is required".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            EntryTypes::Routing(routing) => {
                if routing.design_id.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "design_id is required".into(),
                    ));
                }
                if routing.steps.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "routing must have at least one step".into(),
                    ));
                }
                // Verify steps are in ascending sequence order
                for w in routing.steps.windows(2) {
                    if w[0].sequence >= w[1].sequence {
                        return Ok(ValidateCallbackResult::Invalid(
                            "routing steps must be in ascending sequence order".into(),
                        ));
                    }
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
