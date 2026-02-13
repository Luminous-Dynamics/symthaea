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
