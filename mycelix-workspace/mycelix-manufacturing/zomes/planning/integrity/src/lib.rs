// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Planning Integrity Zome
//!
//! Entry types for MRP planning runs and results.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MrpRunEntry {
    pub work_order_hashes: Vec<ActionHash>,
    pub horizon_days: u32,
    pub run_at: Timestamp,
    pub feasible: bool,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PlannedOrderEntry {
    pub mrp_run_hash: ActionHash,
    pub part_id: String,
    pub quantity_needed: u64,
    pub quantity_available: u64,
    pub quantity_to_order: u64,
    pub due_date: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MaterialShortageEntry {
    pub mrp_run_hash: ActionHash,
    pub part_id: String,
    pub quantity_needed: u64,
    pub quantity_available: u64,
    pub short_quantity: u64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    MrpRun(MrpRunEntry),
    PlannedOrder(PlannedOrderEntry),
    MaterialShortage(MaterialShortageEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllMrpRuns,
    MrpRunToPlannedOrders,
    MrpRunToShortages,
    WorkOrderToMrpRuns,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::MrpRun(run) => {
                if run.work_order_hashes.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MRP run must include at least one work order".into(),
                    ));
                }
                if run.horizon_days == 0 || run.horizon_days > 365 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "horizon_days must be 1-365".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            EntryTypes::PlannedOrder(po) => {
                if po.part_id.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "part_id is required".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
