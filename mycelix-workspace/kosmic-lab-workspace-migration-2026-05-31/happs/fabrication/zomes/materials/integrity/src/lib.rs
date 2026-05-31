// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Materials Integrity Zome
//!
//! Defines entry types for 3D printing materials with certifications,
//! properties, and Supply Chain integration for circular economy tracking.

use fabrication_common::*;
use hdi::prelude::*;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Material(Material),
    #[entry_type(visibility = "public")]
    MaterialBatch(MaterialBatch),
}

#[hdk_link_types]
pub enum LinkTypes {
    MaterialTypeToBatches,
    SupplierToMaterials,
    AllMaterials,
    FoodSafeMaterials,
    CertifiedMaterials,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Material {
    pub id: String,
    pub name: String,
    pub material_type: MaterialType,
    pub properties: MaterialProperties,
    pub certifications: Vec<Certification>,
    pub suppliers: Vec<ActionHash>,
    pub safety_data_sheet: Option<String>,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MaterialBatch {
    pub material_hash: ActionHash,
    pub batch_id: String,
    pub origin: MaterialOrigin,
    pub recycled_content_percent: f32,
    pub supply_chain_hash: Option<ActionHash>,
    pub certifications: Vec<String>,
    pub end_of_life: EndOfLifeStrategy,
    pub quantity_kg: f32,
    pub received_at: Timestamp,
}

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Material(m) => {
                    if m.name.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("Name required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::MaterialBatch(b) => {
                    if b.recycled_content_percent < 0.0 || b.recycled_content_percent > 100.0 {
                        return Ok(ValidateCallbackResult::Invalid("Invalid recycled %".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
