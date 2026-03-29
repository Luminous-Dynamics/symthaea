// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inventory Integrity Zome - Entry types for inventory management
use hdi::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MovementType { Inbound, Outbound, Transfer, Adjustment }

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct InventoryItem {
    pub sku: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub unit: String,
    pub reorder_point: u64,
    pub reorder_quantity: u64,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StockLevel {
    pub item_hash: ActionHash,
    pub location: String,
    pub quantity: u64,
    pub reserved: u64,
    pub updated_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StockMovement {
    pub item_hash: ActionHash,
    pub movement_type: MovementType,
    pub quantity: u64,
    pub from_location: Option<String>,
    pub to_location: Option<String>,
    pub reference: Option<String>,
    pub notes: Option<String>,
    pub created_at: Timestamp,
    pub created_by: AgentPubKey,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    InventoryItem(InventoryItem),
    StockLevel(StockLevel),
    StockMovement(StockMovement),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllItems, CategoryToItems, ItemToStockLevels, ItemToMovements, LocationToStock, SkuToItem,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => {
            match app_entry {
                EntryTypes::InventoryItem(item) => {
                    if item.sku.is_empty() { return Ok(ValidateCallbackResult::Invalid("SKU required".into())); }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
