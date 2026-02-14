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
    if b.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Batch ID cannot be empty".into()));
    }
    if b.method.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Method cannot be empty".into()));
    }
    if b.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Quantity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_method(m: PreservationMethod) -> ExternResult<ValidateCallbackResult> {
    if m.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Method name cannot be empty".into()));
    }
    if m.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Description cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_storage(s: StorageUnit) -> ExternResult<ValidateCallbackResult> {
    if s.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Storage ID cannot be empty".into()));
    }
    if s.name.trim().is_empty() {
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

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn valid_batch() -> PreservationBatch {
        PreservationBatch {
            id: "batch-1".into(),
            source_crop_hash: None,
            method: "Lacto-fermentation".into(),
            quantity_kg: 5.0,
            started_at: 1700000000,
            expected_ready: 1701000000,
            status: BatchStatus::InProgress,
            notes: None,
        }
    }

    fn valid_method() -> PreservationMethod {
        PreservationMethod {
            name: "Water Bath Canning".into(),
            description: "High-acid food preservation using boiling water".into(),
            shelf_life_days: 365,
            equipment_needed: vec!["Canning pot".into(), "Jars".into()],
            skill_level: SkillLevel::Intermediate,
        }
    }

    fn valid_storage() -> StorageUnit {
        StorageUnit {
            id: "store-1".into(),
            name: "Community Root Cellar".into(),
            capacity_kg: 500.0,
            storage_type: StorageType::RootCellar,
            steward: fake_agent(),
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_batch_status() {
        let statuses = vec![
            BatchStatus::InProgress, BatchStatus::Completed,
            BatchStatus::Failed, BatchStatus::Consumed,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: BatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s);
        }
    }

    #[test]
    fn serde_roundtrip_skill_level() {
        let levels = vec![SkillLevel::Beginner, SkillLevel::Intermediate, SkillLevel::Advanced];
        for l in &levels {
            let json = serde_json::to_string(l).unwrap();
            let back: SkillLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, l);
        }
    }

    #[test]
    fn serde_roundtrip_storage_type() {
        let types = vec![
            StorageType::RootCellar, StorageType::Cellar, StorageType::Freezer,
            StorageType::Dehydrator, StorageType::Fermenter, StorageType::Pantry,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: StorageType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, t);
        }
    }

    #[test]
    fn serde_roundtrip_preservation_batch() {
        let b = valid_batch();
        let json = serde_json::to_string(&b).unwrap();
        let back: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn serde_roundtrip_batch_with_optionals() {
        let mut b = valid_batch();
        b.source_crop_hash = Some(ActionHash::from_raw_36(vec![5u8; 36]));
        b.notes = Some("Good fermentation".into());
        let json = serde_json::to_string(&b).unwrap();
        let back: PreservationBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn serde_roundtrip_preservation_method() {
        let m = valid_method();
        let json = serde_json::to_string(&m).unwrap();
        let back: PreservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn serde_roundtrip_storage_unit() {
        let s = valid_storage();
        let json = serde_json::to_string(&s).unwrap();
        let back: StorageUnit = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    // ── validate_batch: id ──────────────────────────────────────────────

    #[test]
    fn valid_batch_passes() {
        assert_valid(validate_batch(valid_batch()));
    }

    #[test]
    fn batch_empty_id_rejected() {
        let mut b = valid_batch();
        b.id = String::new();
        assert_invalid(validate_batch(b), "Batch ID cannot be empty");
    }

    #[test]
    fn batch_whitespace_id_accepted() {
        // The validator uses is_empty(), not trim().is_empty()
        let mut b = valid_batch();
        b.id = "   ".into();
        assert_valid(validate_batch(b));
    }

    // ── validate_batch: method ──────────────────────────────────────────

    #[test]
    fn batch_empty_method_rejected() {
        let mut b = valid_batch();
        b.method = String::new();
        assert_invalid(validate_batch(b), "Method cannot be empty");
    }

    #[test]
    fn batch_whitespace_method_accepted() {
        let mut b = valid_batch();
        b.method = " ".into();
        assert_valid(validate_batch(b));
    }

    // ── validate_batch: quantity_kg ─────────────────────────────────────

    #[test]
    fn batch_zero_quantity_rejected() {
        let mut b = valid_batch();
        b.quantity_kg = 0.0;
        assert_invalid(validate_batch(b), "Quantity must be positive");
    }

    #[test]
    fn batch_negative_quantity_rejected() {
        let mut b = valid_batch();
        b.quantity_kg = -5.0;
        assert_invalid(validate_batch(b), "Quantity must be positive");
    }

    #[test]
    fn batch_barely_positive_quantity_valid() {
        let mut b = valid_batch();
        b.quantity_kg = 0.001;
        assert_valid(validate_batch(b));
    }

    #[test]
    fn batch_large_quantity_valid() {
        let mut b = valid_batch();
        b.quantity_kg = 99999.0;
        assert_valid(validate_batch(b));
    }

    #[test]
    fn batch_barely_negative_quantity_rejected() {
        let mut b = valid_batch();
        b.quantity_kg = -0.001;
        assert_invalid(validate_batch(b), "Quantity must be positive");
    }

    // ── validate_batch: status variants ─────────────────────────────────

    #[test]
    fn batch_all_statuses_valid() {
        for status in [BatchStatus::InProgress, BatchStatus::Completed,
                       BatchStatus::Failed, BatchStatus::Consumed] {
            let mut b = valid_batch();
            b.status = status;
            assert_valid(validate_batch(b));
        }
    }

    // ── validate_batch: optional fields ─────────────────────────────────

    #[test]
    fn batch_with_source_crop_hash_valid() {
        let mut b = valid_batch();
        b.source_crop_hash = Some(ActionHash::from_raw_36(vec![10u8; 36]));
        assert_valid(validate_batch(b));
    }

    #[test]
    fn batch_with_notes_valid() {
        let mut b = valid_batch();
        b.notes = Some("Batch notes here".into());
        assert_valid(validate_batch(b));
    }

    #[test]
    fn batch_empty_id_with_valid_rest_rejected() {
        let mut b = valid_batch();
        b.id = String::new();
        b.notes = Some("notes".into());
        b.source_crop_hash = Some(ActionHash::from_raw_36(vec![1u8; 36]));
        assert_invalid(validate_batch(b), "Batch ID cannot be empty");
    }

    // ── validate_method: name ───────────────────────────────────────────

    #[test]
    fn valid_method_passes() {
        assert_valid(validate_method(valid_method()));
    }

    #[test]
    fn method_empty_name_rejected() {
        let mut m = valid_method();
        m.name = String::new();
        assert_invalid(validate_method(m), "Method name cannot be empty");
    }

    #[test]
    fn method_whitespace_name_accepted() {
        let mut m = valid_method();
        m.name = "  ".into();
        assert_valid(validate_method(m));
    }

    // ── validate_method: description ────────────────────────────────────

    #[test]
    fn method_empty_description_rejected() {
        let mut m = valid_method();
        m.description = String::new();
        assert_invalid(validate_method(m), "Description cannot be empty");
    }

    #[test]
    fn method_whitespace_description_accepted() {
        let mut m = valid_method();
        m.description = " ".into();
        assert_valid(validate_method(m));
    }

    // ── validate_method: skill levels ───────────────────────────────────

    #[test]
    fn method_all_skill_levels_valid() {
        for level in [SkillLevel::Beginner, SkillLevel::Intermediate, SkillLevel::Advanced] {
            let mut m = valid_method();
            m.skill_level = level;
            assert_valid(validate_method(m));
        }
    }

    // ── validate_method: optional fields ────────────────────────────────

    #[test]
    fn method_zero_shelf_life_valid() {
        let mut m = valid_method();
        m.shelf_life_days = 0;
        assert_valid(validate_method(m));
    }

    #[test]
    fn method_empty_equipment_valid() {
        let mut m = valid_method();
        m.equipment_needed = vec![];
        assert_valid(validate_method(m));
    }

    #[test]
    fn method_large_shelf_life_valid() {
        let mut m = valid_method();
        m.shelf_life_days = 36500;
        assert_valid(validate_method(m));
    }

    // ── validate_storage: id ────────────────────────────────────────────

    #[test]
    fn valid_storage_passes() {
        assert_valid(validate_storage(valid_storage()));
    }

    #[test]
    fn storage_empty_id_rejected() {
        let mut s = valid_storage();
        s.id = String::new();
        assert_invalid(validate_storage(s), "Storage ID cannot be empty");
    }

    #[test]
    fn storage_whitespace_id_accepted() {
        let mut s = valid_storage();
        s.id = " ".into();
        assert_valid(validate_storage(s));
    }

    // ── validate_storage: name ──────────────────────────────────────────

    #[test]
    fn storage_empty_name_rejected() {
        let mut s = valid_storage();
        s.name = String::new();
        assert_invalid(validate_storage(s), "Storage name cannot be empty");
    }

    #[test]
    fn storage_whitespace_name_accepted() {
        let mut s = valid_storage();
        s.name = "  ".into();
        assert_valid(validate_storage(s));
    }

    // ── validate_storage: capacity_kg ───────────────────────────────────

    #[test]
    fn storage_zero_capacity_rejected() {
        let mut s = valid_storage();
        s.capacity_kg = 0.0;
        assert_invalid(validate_storage(s), "Capacity must be positive");
    }

    #[test]
    fn storage_negative_capacity_rejected() {
        let mut s = valid_storage();
        s.capacity_kg = -10.0;
        assert_invalid(validate_storage(s), "Capacity must be positive");
    }

    #[test]
    fn storage_barely_positive_capacity_valid() {
        let mut s = valid_storage();
        s.capacity_kg = 0.01;
        assert_valid(validate_storage(s));
    }

    #[test]
    fn storage_large_capacity_valid() {
        let mut s = valid_storage();
        s.capacity_kg = 100000.0;
        assert_valid(validate_storage(s));
    }

    #[test]
    fn storage_barely_negative_capacity_rejected() {
        let mut s = valid_storage();
        s.capacity_kg = -0.001;
        assert_invalid(validate_storage(s), "Capacity must be positive");
    }

    // ── validate_storage: type variants ─────────────────────────────────

    #[test]
    fn all_storage_types_valid() {
        for st in [StorageType::RootCellar, StorageType::Cellar, StorageType::Freezer,
                    StorageType::Dehydrator, StorageType::Fermenter, StorageType::Pantry] {
            let mut s = valid_storage();
            s.storage_type = st;
            assert_valid(validate_storage(s));
        }
    }

    // ── validate_storage: empty_id takes priority ───────────────────────

    #[test]
    fn storage_empty_id_rejects_before_empty_name() {
        let mut s = valid_storage();
        s.id = String::new();
        s.name = String::new();
        // id check comes first
        assert_invalid(validate_storage(s), "Storage ID cannot be empty");
    }

    #[test]
    fn storage_empty_name_with_zero_capacity_rejects_name_first() {
        let mut s = valid_storage();
        s.name = String::new();
        s.capacity_kg = 0.0;
        // name check comes before capacity check
        assert_invalid(validate_storage(s), "Storage name cannot be empty");
    }

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_batches".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }
}
