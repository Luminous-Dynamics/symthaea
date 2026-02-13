//! Food Preservation Integrity Zome
//! Entry types and validation for food preservation batches, methods, and storage.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// PRESERVATION BATCH
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BatchStatus {
    InProgress,
    Completed,
    Failed,
    Consumed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PreservationBatch {
    pub id: String,
    pub source_crop_hash: Option<ActionHash>,
    pub method: String,
    pub quantity_kg: f64,
    pub started_at: u64,
    pub expected_ready: u64,
    pub status: BatchStatus,
    pub notes: Option<String>,
}

// ============================================================================
// PRESERVATION METHOD
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PreservationMethod {
    pub name: String,
    pub description: String,
    pub shelf_life_days: u32,
    pub equipment_needed: Vec<String>,
    pub skill_level: SkillLevel,
}

// ============================================================================
// STORAGE UNIT
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StorageType {
    RootCellar,
    Cellar,
    Freezer,
    Dehydrator,
    Fermenter,
    Pantry,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StorageUnit {
    pub id: String,
    pub name: String,
    pub capacity_kg: f64,
    pub storage_type: StorageType,
    pub steward: AgentPubKey,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    PreservationBatch(PreservationBatch),
    PreservationMethod(PreservationMethod),
    StorageUnit(StorageUnit),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllBatches,
    AllMethods,
    AllStorage,
    MethodToBatch,
    StorageToBatch,
    AgentToBatch,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::PreservationBatch(b) => validate_batch(b),
                EntryTypes::PreservationMethod(m) => validate_method(m),
                EntryTypes::StorageUnit(s) => validate_storage(s),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::PreservationBatch(b) => validate_batch(b),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_batch(b: PreservationBatch) -> ExternResult<ValidateCallbackResult> {
    if b.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Batch ID cannot be empty".into()));
    }
    if b.method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Method cannot be empty".into()));
    }
    if b.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Quantity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_method(m: PreservationMethod) -> ExternResult<ValidateCallbackResult> {
    if m.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Method name cannot be empty".into()));
    }
    if m.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Description cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_storage(s: StorageUnit) -> ExternResult<ValidateCallbackResult> {
    if s.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Storage ID cannot be empty".into()));
    }
    if s.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Storage name cannot be empty".into()));
    }
    if s.capacity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Capacity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    #[test]
    fn valid_batch_passes() {
        let b = PreservationBatch {
            id: "batch-1".into(),
            source_crop_hash: None,
            method: "Lacto-fermentation".into(),
            quantity_kg: 5.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: None,
        };
        assert_eq!(validate_batch(b).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn batch_empty_id_rejected() {
        let b = PreservationBatch {
            id: String::new(),
            source_crop_hash: None,
            method: "Drying".into(),
            quantity_kg: 2.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: None,
        };
        assert!(matches!(validate_batch(b).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn batch_empty_method_rejected() {
        let b = PreservationBatch {
            id: "batch-2".into(),
            source_crop_hash: None,
            method: String::new(),
            quantity_kg: 2.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: None,
        };
        assert!(matches!(validate_batch(b).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn batch_zero_quantity_rejected() {
        let b = PreservationBatch {
            id: "batch-3".into(),
            source_crop_hash: None,
            method: "Canning".into(),
            quantity_kg: 0.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: None,
        };
        assert!(matches!(validate_batch(b).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_method_passes() {
        let m = PreservationMethod {
            name: "Water Bath Canning".into(),
            description: "High-acid food preservation using boiling water".into(),
            shelf_life_days: 365,
            equipment_needed: vec!["Canning pot".into(), "Jars".into()],
            skill_level: SkillLevel::Intermediate,
        };
        assert_eq!(validate_method(m).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn method_empty_name_rejected() {
        let m = PreservationMethod {
            name: String::new(),
            description: "Some method".into(),
            shelf_life_days: 30,
            equipment_needed: vec![],
            skill_level: SkillLevel::Beginner,
        };
        assert!(matches!(validate_method(m).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn method_empty_description_rejected() {
        let m = PreservationMethod {
            name: "Drying".into(),
            description: String::new(),
            shelf_life_days: 180,
            equipment_needed: vec!["Dehydrator".into()],
            skill_level: SkillLevel::Beginner,
        };
        assert!(matches!(validate_method(m).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_storage_passes() {
        let s = StorageUnit {
            id: "store-1".into(),
            name: "Community Root Cellar".into(),
            capacity_kg: 500.0,
            storage_type: StorageType::RootCellar,
            steward: fake_agent(),
        };
        assert_eq!(validate_storage(s).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn storage_empty_id_rejected() {
        let s = StorageUnit {
            id: String::new(),
            name: "Cellar".into(),
            capacity_kg: 100.0,
            storage_type: StorageType::Cellar,
            steward: fake_agent(),
        };
        assert!(matches!(validate_storage(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn storage_zero_capacity_rejected() {
        let s = StorageUnit {
            id: "store-2".into(),
            name: "Tiny Pantry".into(),
            capacity_kg: 0.0,
            storage_type: StorageType::Pantry,
            steward: fake_agent(),
        };
        assert!(matches!(validate_storage(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn all_storage_types_valid() {
        for st in [StorageType::RootCellar, StorageType::Cellar, StorageType::Freezer,
                    StorageType::Dehydrator, StorageType::Fermenter, StorageType::Pantry] {
            let s = StorageUnit {
                id: "s".into(),
                name: "Test".into(),
                capacity_kg: 10.0,
                storage_type: st,
                steward: fake_agent(),
            };
            assert_eq!(validate_storage(s).unwrap(), ValidateCallbackResult::Valid);
        }
    }
}
