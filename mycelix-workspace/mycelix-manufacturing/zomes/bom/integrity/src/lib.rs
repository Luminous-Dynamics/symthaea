// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bill of Materials Integrity Zome
//!
//! Entry types and validation for BOMs.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BomEntry {
    pub design_id: String,
    pub revision: String,
    pub items: Vec<BomItemEntry>,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, SerializedBytes)]
pub struct BomItemEntry {
    pub part_id: String,
    pub quantity_per: u64,
    pub unit: String,
    pub sub_assembly_bom_hash: Option<ActionHash>,
    pub notes: Option<String>,
}

/// Flat material line after BOM explosion.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BomExplosionResult {
    pub source_bom_hash: ActionHash,
    pub lines: Vec<ExplosionLine>,
    pub total_quantity: u64,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, SerializedBytes)]
pub struct ExplosionLine {
    pub part_id: String,
    pub total_quantity: u64,
    pub unit: String,
    pub depth: u32,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Bom(BomEntry),
    BomExplosion(BomExplosionResult),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllBoms,
    DesignToBom,
    BomToSubAssembly,
    BomToExplosion,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Bom(bom) => {
                if bom.design_id.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "design_id is required".into(),
                    ));
                }
                if bom.revision.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "revision is required".into(),
                    ));
                }
                if bom.items.is_empty() {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BOM must have at least one item".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
