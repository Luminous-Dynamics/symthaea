// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manufacturing Bridge Integrity Zome
//!
//! Entry types for cross-cluster bridge records between manufacturing
//! and other clusters (supplychain, marketplace, etc.).

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BridgeEventEntry {
    pub event_type: String,
    pub source_cluster: String,
    pub target_cluster: String,
    pub payload: String,
    pub reference_hash: Option<String>,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProductionNotification {
    pub work_order_id: String,
    pub product_id: String,
    pub quantity_completed: u64,
    pub completed_at: Timestamp,
    pub target_cluster: String,
    pub acknowledged: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    BridgeEvent(BridgeEventEntry),
    ProductionNotification(ProductionNotification),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllBridgeEvents,
    ClusterToBridgeEvents,
    WorkOrderToNotifications,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::BridgeEvent(evt) => {
                if evt.event_type.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "event_type is required".into(),
                    ));
                }
                if evt.source_cluster.is_empty() || evt.target_cluster.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "source and target cluster are required".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            EntryTypes::ProductionNotification(notif) => {
                if notif.product_id.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "product_id is required".into(),
                    ));
                }
                if notif.quantity_completed == 0 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "quantity_completed must be > 0".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
