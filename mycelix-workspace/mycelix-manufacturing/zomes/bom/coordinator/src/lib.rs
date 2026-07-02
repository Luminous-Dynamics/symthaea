// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bill of Materials Coordinator Zome
//!
//! CRUD operations and BOM explosion for manufacturing BOMs.

use hdk::prelude::*;
use bom_integrity::*;

// ============================================================================
// Input types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateBomInput {
    pub design_id: String,
    pub revision: String,
    pub items: Vec<BomItemInput>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BomItemInput {
    pub part_id: String,
    pub quantity_per: u64,
    pub unit: String,
    pub sub_assembly_bom_hash: Option<ActionHash>,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExplodeBomInput {
    pub bom_hash: ActionHash,
    pub quantity: u64,
    pub flatten: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExplodeBomOutput {
    pub lines: Vec<ExplosionLine>,
    pub total_parts: usize,
}

// ============================================================================
// Extern functions
// ============================================================================

/// Create a new bill of materials.
#[hdk_extern]
pub fn create_bom(input: CreateBomInput) -> ExternResult<ActionHash> {
    if input.design_id.is_empty() || input.design_id.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "design_id must be 1-200 characters".to_string()
        )));
    }
    if input.revision.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "revision is required".to_string()
        )));
    }
    if input.items.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "BOM must have at least one item".to_string()
        )));
    }

    let items: Vec<BomItemEntry> = input
        .items
        .into_iter()
        .map(|i| BomItemEntry {
            part_id: i.part_id,
            quantity_per: i.quantity_per,
            unit: i.unit,
            sub_assembly_bom_hash: i.sub_assembly_bom_hash,
            notes: i.notes,
        })
        .collect();

    let now = sys_time()?;
    let entry = BomEntry {
        design_id: input.design_id.clone(),
        revision: input.revision,
        items,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::Bom(entry))?;

    // Link from "all_boms" anchor
    let all_path = Path::from("all_boms").typed(LinkTypes::AllBoms)?;
    all_path.ensure()?;
    create_link(
        all_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllBoms,
        (),
    )?;

    // Link from design
    let design_path =
        Path::from(format!("design/{}", input.design_id)).typed(LinkTypes::DesignToBom)?;
    design_path.ensure()?;
    create_link(
        design_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::DesignToBom,
        (),
    )?;

    Ok(action_hash)
}

/// Retrieve a BOM by its action hash.
#[hdk_extern]
pub fn get_bom(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Explode a BOM into a flat list of material requirements.
///
/// If `flatten` is true and items reference sub-assembly BOMs, those are
/// recursively resolved and merged into a single flat list.
#[hdk_extern]
pub fn explode_bom(input: ExplodeBomInput) -> ExternResult<ExplodeBomOutput> {
    let record = get(input.bom_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("BOM not found".to_string())),
    )?;

    let bom: BomEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize BOM".to_string()
        )))?;

    let mut lines = Vec::new();
    explode_recursive(&bom, input.quantity, input.flatten, 0, &mut lines)?;

    let total_parts = lines.len();
    Ok(ExplodeBomOutput { lines, total_parts })
}

/// Recursively explode a BOM, following sub-assembly references.
fn explode_recursive(
    bom: &BomEntry,
    multiplier: u64,
    flatten: bool,
    depth: u32,
    lines: &mut Vec<ExplosionLine>,
) -> ExternResult<()> {
    const MAX_DEPTH: u32 = 10;
    if depth > MAX_DEPTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "BOM explosion exceeded max depth (circular reference?)".to_string()
        )));
    }

    for item in &bom.items {
        let total = item.quantity_per * multiplier;

        if flatten && item.sub_assembly_bom_hash.is_some() {
            // Resolve sub-assembly
            let sub_hash = item.sub_assembly_bom_hash.clone().unwrap();
            let sub_record = get(sub_hash, GetOptions::default())?.ok_or(
                wasm_error!(WasmErrorInner::Guest(
                    "Sub-assembly BOM not found".to_string()
                )),
            )?;
            let sub_bom: BomEntry = sub_record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Could not deserialize sub-assembly BOM".to_string()
                )))?;
            explode_recursive(&sub_bom, total, flatten, depth + 1, lines)?;
        } else {
            lines.push(ExplosionLine {
                part_id: item.part_id.clone(),
                total_quantity: total,
                unit: item.unit.clone(),
                depth,
            });
        }
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_bom_input_serde() {
        let input = CreateBomInput {
            design_id: "D-001".to_string(),
            revision: "A".to_string(),
            items: vec![BomItemInput {
                part_id: "BOLT-M6".to_string(),
                quantity_per: 4,
                unit: "each".to_string(),
                sub_assembly_bom_hash: None,
                notes: None,
            }],
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreateBomInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.design_id, "D-001");
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.items[0].quantity_per, 4);
    }

    #[test]
    fn test_explode_input_serde() {
        let input = ExplodeBomInput {
            bom_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            quantity: 10,
            flatten: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: ExplodeBomInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.quantity, 10);
        assert!(back.flatten);
    }

    #[test]
    fn test_explosion_line_serde() {
        let line = ExplosionLine {
            part_id: "BOLT-M6".to_string(),
            total_quantity: 40,
            unit: "each".to_string(),
            depth: 0,
        };
        let json = serde_json::to_string(&line).unwrap();
        let back: ExplosionLine = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_quantity, 40);
    }

    #[test]
    fn test_bom_item_with_sub_assembly() {
        let item = BomItemInput {
            part_id: String::new(),
            quantity_per: 2,
            unit: "each".to_string(),
            sub_assembly_bom_hash: Some(ActionHash::from_raw_36(vec![1u8; 36])),
            notes: Some("Sub-assembly reference".to_string()),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("sub_assembly_bom_hash"));
    }
}
