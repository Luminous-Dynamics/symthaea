// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Work Orders Integrity Zome
//!
//! Entry types and validation for manufacturing work orders.

use hdi::prelude::*;
use manufacturing_common::{WorkOrderPriority, WorkOrderStatus};

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WorkOrderEntry {
    pub product_id: String,
    pub quantity: u64,
    pub due_date: Timestamp,
    pub status: WorkOrderStatus,
    pub priority: WorkOrderPriority,
    pub notes: Option<String>,
    pub bom_hash: Option<ActionHash>,
    pub routing_hash: Option<ActionHash>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WorkOrderStatusUpdate {
    pub work_order_hash: ActionHash,
    pub previous_status: WorkOrderStatus,
    pub new_status: WorkOrderStatus,
    pub updated_at: Timestamp,
    pub reason: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    WorkOrder(WorkOrderEntry),
    StatusUpdate(WorkOrderStatusUpdate),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllWorkOrders,
    ProductToWorkOrders,
    WorkOrderToStatusUpdates,
    StatusToWorkOrders,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::WorkOrder(wo) => {
                if wo.product_id.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "product_id is required".into(),
                    ));
                }
                if wo.quantity == 0 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "quantity must be > 0".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            EntryTypes::StatusUpdate(su) => {
                if !su.previous_status.can_transition_to(&su.new_status) {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "Invalid transition: {:?} -> {:?}",
                        su.previous_status, su.new_status
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
