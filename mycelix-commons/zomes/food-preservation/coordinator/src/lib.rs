//! Food Preservation Coordinator Zome
//! Business logic for preservation batches, methods, and storage management.

use food_preservation_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// BATCH MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn start_batch(batch: PreservationBatch) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let action_hash = create_entry(&EntryTypes::PreservationBatch(batch.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_batches".to_string())))?;
    create_link(anchor_hash("all_batches")?, action_hash.clone(), LinkTypes::AllBatches, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToBatch, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created batch".into())))
}

#[hdk_extern]
pub fn complete_batch(batch_hash: ActionHash) -> ExternResult<Record> {
    let record = get(batch_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Batch not found".into())))?;
    let mut batch: PreservationBatch = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid batch entry".into())))?;

    batch.status = BatchStatus::Completed;
    let new_hash = update_entry(record.action_address().clone(), &EntryTypes::PreservationBatch(batch))?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated batch".into())))
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
    let action_hash = create_entry(&EntryTypes::PreservationMethod(method))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_methods".to_string())))?;
    create_link(anchor_hash("all_methods")?, action_hash.clone(), LinkTypes::AllMethods, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created method".into())))
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
    let action_hash = create_entry(&EntryTypes::StorageUnit(storage))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_storage".to_string())))?;
    create_link(anchor_hash("all_storage")?, action_hash.clone(), LinkTypes::AllStorage, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created storage".into())))
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
        };
        let json = serde_json::to_string(&batch).unwrap();
        let decoded: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "batch-1");
        assert_eq!(decoded.method, "Lacto-fermentation");
        assert_eq!(decoded.quantity_kg, 5.0);
        assert_eq!(decoded.status, BatchStatus::InProgress);
        assert_eq!(decoded.notes, Some("First fermentation attempt".to_string()));
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
}
