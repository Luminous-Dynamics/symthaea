// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Food Preservation Coordinator Zome
//! Business logic for preservation batches, methods, and storage management.

use food_preservation_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// ============================================================================
// BATCH MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn start_batch(batch: PreservationBatch) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "start_batch")?;
    let agent = agent_info()?.agent_initial_pubkey;
    let action_hash = create_entry(&EntryTypes::PreservationBatch(batch.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_batches".to_string())))?;
    create_link(
        anchor_hash("all_batches")?,
        action_hash.clone(),
        LinkTypes::AllBatches,
        (),
    )?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToBatch, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created batch".into()
    )))
}

#[hdk_extern]
pub fn complete_batch(batch_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "complete_batch")?;
    let record = get(batch_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let mut batch: PreservationBatch = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid batch entry".into()
        )))?;

    batch.status = BatchStatus::Completed;
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::PreservationBatch(batch),
    )?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated batch".into()
    )))
}

#[hdk_extern]
pub fn get_batch(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_agent_batches(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToBatch)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// METHOD REGISTRY
// ============================================================================

#[hdk_extern]
pub fn register_method(method: PreservationMethod) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "register_method")?;
    let action_hash = create_entry(&EntryTypes::PreservationMethod(method))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_methods".to_string())))?;
    create_link(
        anchor_hash("all_methods")?,
        action_hash.clone(),
        LinkTypes::AllMethods,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created method".into()
    )))
}

#[hdk_extern]
pub fn get_all_methods(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_methods")?, LinkTypes::AllMethods)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// STORAGE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn register_storage(storage: StorageUnit) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "register_storage")?;
    let action_hash = create_entry(&EntryTypes::StorageUnit(storage))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_storage".to_string())))?;
    create_link(
        anchor_hash("all_storage")?,
        action_hash.clone(),
        LinkTypes::AllStorage,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created storage".into()
    )))
}

#[hdk_extern]
pub fn get_storage_inventory(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_storage")?, LinkTypes::AllStorage)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    // ========================================================================
    // BatchStatus enum serde tests
    // ========================================================================

    #[test]
    fn batch_status_all_variants_serde() {
        let variants = vec![
            BatchStatus::InProgress,
            BatchStatus::Completed,
            BatchStatus::Failed,
            BatchStatus::Consumed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: BatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // SkillLevel enum serde tests
    // ========================================================================

    #[test]
    fn skill_level_all_variants_serde() {
        let variants = vec![
            SkillLevel::Beginner,
            SkillLevel::Intermediate,
            SkillLevel::Advanced,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: SkillLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // StorageType enum serde tests
    // ========================================================================

    #[test]
    fn storage_type_all_variants_serde() {
        let variants = vec![
            StorageType::RootCellar,
            StorageType::Cellar,
            StorageType::Freezer,
            StorageType::Dehydrator,
            StorageType::Fermenter,
            StorageType::Pantry,
            StorageType::Composter,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: StorageType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // PreservationBatch serde roundtrip tests
    // ========================================================================

    #[test]
    fn preservation_batch_serde_roundtrip() {
        let batch = PreservationBatch {
            id: "batch-1".to_string(),
            source_crop_hash: None,
            method: "Lacto-fermentation".to_string(),
            quantity_kg: 5.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: Some("First fermentation attempt".to_string()),
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "batch-1");
        assert_eq!(decoded.method, "Lacto-fermentation");
        assert_eq!(decoded.quantity_kg, 5.0);
        assert_eq!(decoded.status, BatchStatus::InProgress);
        assert_eq!(
            decoded.notes,
            Some("First fermentation attempt".to_string())
        );
    }

    #[test]
    fn preservation_batch_none_fields_roundtrip() {
        let batch = PreservationBatch {
            id: "batch-2".to_string(),
            source_crop_hash: None,
            method: "Drying".to_string(),
            quantity_kg: 2.0,
            started_at: 1700000000,
            expected_ready: 1700500000,
            status: BatchStatus::Completed,
            notes: None,
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.source_crop_hash, None);
        assert_eq!(decoded.notes, None);
        assert_eq!(decoded.status, BatchStatus::Completed);
    }

    // ========================================================================
    // PreservationMethod serde roundtrip tests
    // ========================================================================

    #[test]
    fn preservation_method_serde_roundtrip() {
        let method = PreservationMethod {
            name: "Water Bath Canning".to_string(),
            description: "High-acid food preservation using boiling water".to_string(),
            shelf_life_days: 365,
            equipment_needed: vec!["Canning pot".to_string(), "Jars".to_string()],
            skill_level: SkillLevel::Intermediate,
        };
        let json = serde_json::to_string(&method).unwrap();
        let decoded: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Water Bath Canning");
        assert_eq!(decoded.shelf_life_days, 365);
        assert_eq!(decoded.skill_level, SkillLevel::Intermediate);
        assert_eq!(decoded.equipment_needed.len(), 2);
    }

    // ========================================================================
    // StorageUnit serde roundtrip tests
    // ========================================================================

    #[test]
    fn storage_unit_serde_roundtrip() {
        let unit = StorageUnit {
            id: "store-1".to_string(),
            name: "Community Root Cellar".to_string(),
            capacity_kg: 500.0,
            storage_type: StorageType::RootCellar,
            steward: fake_agent(),
        };
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: StorageUnit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "store-1");
        assert_eq!(decoded.name, "Community Root Cellar");
        assert_eq!(decoded.capacity_kg, 500.0);
        assert_eq!(decoded.storage_type, StorageType::RootCellar);
    }

    // ========================================================================
    // Clone/equality tests
    // ========================================================================

    #[test]
    fn preservation_batch_clone_equals_original() {
        let batch = PreservationBatch {
            id: "batch-clone".to_string(),
            source_crop_hash: Some(ActionHash::from_raw_36(vec![1u8; 36])),
            method: "Smoking".to_string(),
            quantity_kg: 3.5,
            started_at: 1700000000,
            expected_ready: 1700500000,
            status: BatchStatus::InProgress,
            notes: Some("Hickory chips".to_string()),
            allergen_flags: vec![],
        };
        assert_eq!(batch, batch.clone());
    }

    #[test]
    fn preservation_batch_ne_different_status() {
        let a = PreservationBatch {
            id: "batch-ne".to_string(),
            source_crop_hash: None,
            method: "Drying".to_string(),
            quantity_kg: 1.0,
            started_at: 0,
            expected_ready: 1,
            status: BatchStatus::InProgress,
            notes: None,
            allergen_flags: vec![],
        };
        let mut b = a.clone();
        b.status = BatchStatus::Failed;
        assert_ne!(a, b);
    }

    #[test]
    fn preservation_method_clone_equals_original() {
        let method = PreservationMethod {
            name: "Canning".to_string(),
            description: "Seal in jars".to_string(),
            shelf_life_days: 730,
            equipment_needed: vec!["Jars".to_string(), "Lids".to_string()],
            skill_level: SkillLevel::Advanced,
        };
        assert_eq!(method, method.clone());
    }

    #[test]
    fn storage_unit_clone_equals_original() {
        let unit = StorageUnit {
            id: "s-clone".to_string(),
            name: "Pantry".to_string(),
            capacity_kg: 50.0,
            storage_type: StorageType::Pantry,
            steward: fake_agent(),
        };
        assert_eq!(unit, unit.clone());
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn preservation_batch_with_source_crop_hash() {
        let batch = PreservationBatch {
            id: "batch-crop".to_string(),
            source_crop_hash: Some(ActionHash::from_raw_36(vec![99u8; 36])),
            method: "Pickling".to_string(),
            quantity_kg: 10.0,
            started_at: 1700000000,
            expected_ready: 1702000000,
            status: BatchStatus::Completed,
            notes: Some("Pickled cucumbers".to_string()),
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert!(decoded.source_crop_hash.is_some());
        assert_eq!(decoded.status, BatchStatus::Completed);
    }

    #[test]
    fn preservation_batch_zero_quantity_serde() {
        let batch = PreservationBatch {
            id: "batch-zero".to_string(),
            source_crop_hash: None,
            method: "Freeze-drying".to_string(),
            quantity_kg: 0.0,
            started_at: 0,
            expected_ready: 0,
            status: BatchStatus::Failed,
            notes: None,
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.quantity_kg, 0.0);
        assert_eq!(decoded.status, BatchStatus::Failed);
    }

    #[test]
    fn preservation_batch_consumed_status_serde() {
        let batch = PreservationBatch {
            id: "batch-consumed".to_string(),
            source_crop_hash: None,
            method: "Fermenting".to_string(),
            quantity_kg: 2.5,
            started_at: 1700000000,
            expected_ready: 1700500000,
            status: BatchStatus::Consumed,
            notes: Some("Kimchi consumed by community".to_string()),
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, BatchStatus::Consumed);
    }

    #[test]
    fn preservation_method_empty_equipment_serde() {
        let method = PreservationMethod {
            name: "Sun drying".to_string(),
            description: "Dry in direct sunlight".to_string(),
            shelf_life_days: 90,
            equipment_needed: vec![],
            skill_level: SkillLevel::Beginner,
        };
        let json = serde_json::to_string(&method).unwrap();
        let decoded: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert!(decoded.equipment_needed.is_empty());
    }

    #[test]
    fn preservation_method_u32_max_shelf_life() {
        let method = PreservationMethod {
            name: "Ancient method".to_string(),
            description: "Lasts forever".to_string(),
            shelf_life_days: u32::MAX,
            equipment_needed: vec!["Magic jar".to_string()],
            skill_level: SkillLevel::Advanced,
        };
        let json = serde_json::to_string(&method).unwrap();
        let decoded: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.shelf_life_days, u32::MAX);
    }

    #[test]
    fn storage_unit_all_types_serde() {
        for st in [
            StorageType::RootCellar,
            StorageType::Cellar,
            StorageType::Freezer,
            StorageType::Dehydrator,
            StorageType::Fermenter,
            StorageType::Pantry,
            StorageType::Composter,
        ] {
            let unit = StorageUnit {
                id: "s".to_string(),
                name: "T".to_string(),
                capacity_kg: 1.0,
                storage_type: st.clone(),
                steward: fake_agent(),
            };
            let json = serde_json::to_string(&unit).unwrap();
            let decoded: StorageUnit = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.storage_type, st);
        }
    }

    #[test]
    fn preservation_method_unicode_name_serde() {
        let method = PreservationMethod {
            name: "\u{767D}\u{83DC}\u{306E}\u{6F2C}\u{7269}".to_string(),
            description: "Japanese pickling".to_string(),
            shelf_life_days: 30,
            equipment_needed: vec!["\u{6F2C}\u{7269}\u{77F3}".to_string()],
            skill_level: SkillLevel::Intermediate,
        };
        let json = serde_json::to_string(&method).unwrap();
        let decoded: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "\u{767D}\u{83DC}\u{306E}\u{6F2C}\u{7269}");
    }

    #[test]
    fn storage_unit_large_capacity_serde() {
        let unit = StorageUnit {
            id: "s-big".to_string(),
            name: "Warehouse".to_string(),
            capacity_kg: f64::MAX,
            storage_type: StorageType::Freezer,
            steward: fake_agent(),
        };
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: StorageUnit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.capacity_kg, f64::MAX);
    }

    // ========================================================================
    // Additional boundary and edge case tests
    // ========================================================================

    #[test]
    fn preservation_method_many_equipment_items_serde() {
        let equipment: Vec<String> = (0..100).map(|i| format!("Equipment item #{}", i)).collect();
        let method = PreservationMethod {
            name: "Complex method".to_string(),
            description: "Requires many tools and supplies".to_string(),
            shelf_life_days: 180,
            equipment_needed: equipment.clone(),
            skill_level: SkillLevel::Advanced,
        };
        let json = serde_json::to_string(&method).unwrap();
        let decoded: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.equipment_needed.len(), 100);
        assert_eq!(decoded.equipment_needed[0], "Equipment item #0");
        assert_eq!(decoded.equipment_needed[99], "Equipment item #99");
    }

    #[test]
    fn preservation_batch_ne_different_id() {
        let a = PreservationBatch {
            id: "batch-a".to_string(),
            source_crop_hash: None,
            method: "Drying".to_string(),
            quantity_kg: 1.0,
            started_at: 0,
            expected_ready: 1,
            status: BatchStatus::InProgress,
            notes: None,
            allergen_flags: vec![],
        };
        let mut b = a.clone();
        b.id = "batch-b".to_string();
        assert_ne!(a, b);
    }

    #[test]
    fn storage_unit_ne_different_type() {
        let a = StorageUnit {
            id: "s-1".to_string(),
            name: "Unit A".to_string(),
            capacity_kg: 100.0,
            storage_type: StorageType::Freezer,
            steward: fake_agent(),
        };
        let mut b = a.clone();
        b.storage_type = StorageType::Pantry;
        assert_ne!(a, b);
    }

    #[test]
    fn preservation_batch_long_notes_serde() {
        let long_notes = "A".repeat(10_000);
        let batch = PreservationBatch {
            id: "batch-long".to_string(),
            source_crop_hash: None,
            method: "Salt curing".to_string(),
            quantity_kg: 25.0,
            started_at: 1700000000,
            expected_ready: 1702000000,
            status: BatchStatus::InProgress,
            notes: Some(long_notes.clone()),
            allergen_flags: vec![],
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.notes.unwrap().len(), 10_000);
    }
}
