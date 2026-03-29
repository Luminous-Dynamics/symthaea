// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machines Integrity Zome
//!
//! Entry types and validation for the machine registry.

use hdi::prelude::*;
use manufacturing_common::{MachineStatus, MachineType};

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MachineEntry {
    pub name: String,
    pub machine_type: MachineType,
    pub capabilities: Vec<String>,
    pub location: String,
    pub max_throughput_per_hour: u32,
    pub status: MachineStatus,
    pub current_work_order: Option<ActionHash>,
    pub registered_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MachineStatusLog {
    pub machine_hash: ActionHash,
    pub previous_status: MachineStatus,
    pub new_status: MachineStatus,
    pub work_order_hash: Option<ActionHash>,
    pub changed_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Machine(MachineEntry),
    StatusLog(MachineStatusLog),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllMachines,
    TypeToMachines,
    MachineToStatusLog,
    LocationToMachines,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Machine(m) => {
                if m.name.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "machine name is required".into(),
                    ));
                }
                if m.location.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "machine location is required".into(),
                    ));
                }
                if m.max_throughput_per_hour == 0 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "max_throughput_per_hour must be > 0".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            EntryTypes::StatusLog(log) => {
                if !log.previous_status.can_transition_to(&log.new_status) {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "Invalid machine transition: {:?} -> {:?}",
                        log.previous_status, log.new_status
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
