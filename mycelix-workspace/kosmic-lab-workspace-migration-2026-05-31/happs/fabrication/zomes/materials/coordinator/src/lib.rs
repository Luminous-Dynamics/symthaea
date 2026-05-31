// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Materials Coordinator Zome
//!
//! CRUD operations and discovery for 3D printing materials.

use fabrication_common::*;
use hdk::prelude::*;
use materials_integrity::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMaterialInput {
    pub name: String,
    pub material_type: MaterialType,
    pub properties: MaterialProperties,
    pub certifications: Vec<Certification>,
    pub safety_data_sheet: Option<String>,
}

#[hdk_extern]
pub fn create_material(input: CreateMaterialInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let material = Material {
        id: format!("mat_{}", now.as_micros()),
        name: input.name,
        material_type: input.material_type.clone(),
        properties: input.properties,
        certifications: input.certifications,
        suppliers: vec![],
        safety_data_sheet: input.safety_data_sheet,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::Material(material))?;

    let anchor = all_materials_anchor()?;
    create_link(anchor, hash.clone(), LinkTypes::AllMaterials, ())?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_material(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_materials_by_type(material_type: MaterialType) -> ExternResult<Vec<Record>> {
    let anchor = all_materials_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllMaterials)?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(mat) = record.entry().to_app_option::<Material>().ok().flatten() {
                    if mat.material_type == material_type {
                        results.push(record);
                    }
                }
            }
        }
    }
    Ok(results)
}

#[hdk_extern]
pub fn get_food_safe_materials(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = all_materials_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllMaterials)?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(mat) = record.entry().to_app_option::<Material>().ok().flatten() {
                    if mat.properties.food_safe {
                        results.push(record);
                    }
                }
            }
        }
    }
    Ok(results)
}

/// Simple anchor helper - creates deterministic hash from string
fn all_materials_anchor() -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        "anchor:all_materials".as_bytes().to_vec(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}
