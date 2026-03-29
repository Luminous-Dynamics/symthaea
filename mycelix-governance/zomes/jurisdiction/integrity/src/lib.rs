// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Jurisdiction Integrity Zome
//! Defines entry types and validation for governance jurisdiction constraints
//!
//! Re-uses `JurisdictionConstraintEntry` and `validate_jurisdiction_constraint`
//! from `mycelix-bridge-entry-types` for shared cross-cluster validation.

use hdi::prelude::*;
pub use mycelix_bridge_entry_types::{validate_jurisdiction_constraint, JurisdictionConstraintEntry};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Versioned wrapper around `JurisdictionConstraintEntry`.
///
/// The bridge entry type does not carry a version field, so we wrap it here
/// to support optimistic-concurrency version checks on DHT updates.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JurisdictionRecord {
    /// The underlying jurisdiction constraint data.
    pub entry: JurisdictionConstraintEntry,
    /// Monotonically increasing version counter (starts at 1).
    pub version: u32,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Jurisdiction(JurisdictionRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// O(1) lookup: zone_id anchor -> jurisdiction record
    ZoneById,
    /// Regulatory tag anchor -> jurisdiction records with that tag
    TagToZone,
    /// "active_zones" anchor -> all active jurisdiction records
    ActiveZones,
    /// Authority DID anchor -> jurisdiction records asserted by that authority
    AuthorityToZone,
}

// ============================================================================
// PURE VALIDATION FUNCTIONS (testable without HDI host)
// ============================================================================

/// Validate a new jurisdiction record on creation.
///
/// Delegates field-level validation to the bridge-entry-types crate, then
/// checks version invariants.
pub fn check_create_jurisdiction(record: &JurisdictionRecord) -> Result<(), String> {
    validate_jurisdiction_constraint(&record.entry)?;
    if record.version != 1 {
        return Err("Initial jurisdiction version must be 1".into());
    }
    Ok(())
}

/// Validate a jurisdiction record update.
///
/// Ensures immutable fields are preserved and version increments correctly.
pub fn check_update_jurisdiction(
    original: &JurisdictionRecord,
    updated: &JurisdictionRecord,
) -> Result<(), String> {
    // Delegate field-level validation on the new entry
    validate_jurisdiction_constraint(&updated.entry)?;

    // zone_id is the primary key — must never change
    if updated.entry.zone_id != original.entry.zone_id {
        return Err("Cannot change zone_id on update".into());
    }

    // Version must increment by exactly 1
    if updated.version != original.version + 1 {
        return Err("Version must be incremented by 1".into());
    }

    Ok(())
}

// ============================================================================
// VALIDATION CALLBACK
// ============================================================================

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Jurisdiction(record) => {
                    validate_create_jurisdiction(action, record)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Jurisdiction(record) => {
                    validate_update_jurisdiction(action, record, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ZoneById => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TagToZone => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveZones => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AuthorityToZone => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ZoneById => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TagToZone => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveZones => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AuthorityToZone => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate jurisdiction creation
fn validate_create_jurisdiction(
    _action: Create,
    record: JurisdictionRecord,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_jurisdiction(&record) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate jurisdiction update
fn validate_update_jurisdiction(
    _action: Update,
    record: JurisdictionRecord,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original: JurisdictionRecord = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original jurisdiction record not found".into()
        )))?;

    match check_update_jurisdiction(&original, &record) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_jurisdiction() -> JurisdictionRecord {
        JurisdictionRecord {
            entry: JurisdictionConstraintEntry {
                zone_id: "eu-gdpr-zone-1".into(),
                zone_polygon: vec![
                    (35.0, -10.0),
                    (71.0, -10.0),
                    (71.0, 40.0),
                    (35.0, 40.0),
                ],
                regulatory_tags: vec!["gdpr".into(), "right_to_erasure".into()],
                fiat_zone: Some("EUR".into()),
                enforcement_risk: 0.85,
                authority_did: Some("did:mycelix:eu-dpa-collective".into()),
                description: "European Union GDPR enforcement zone".into(),
                created_at: ts(1_000_000),
                last_validated: ts(1_000_000),
            },
            version: 1,
        }
    }

    // --- Create validation ---

    #[test]
    fn test_valid_jurisdiction_accepted() {
        assert!(check_create_jurisdiction(&make_jurisdiction()).is_ok());
    }

    #[test]
    fn test_initial_version_must_be_1() {
        let mut j = make_jurisdiction();
        j.version = 5;
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("version"));
    }

    #[test]
    fn test_empty_zone_id_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_id = "".into();
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("Zone ID"));
    }

    #[test]
    fn test_long_zone_id_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_id = "z".repeat(257);
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("256"));
    }

    #[test]
    fn test_too_many_polygon_vertices_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_polygon = (0..1001).map(|i| (i as f64 * 0.01, 0.0)).collect();
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("1000"));
    }

    #[test]
    fn test_empty_polygon_accepted() {
        let mut j = make_jurisdiction();
        j.entry.zone_polygon = vec![];
        assert!(check_create_jurisdiction(&j).is_ok());
    }

    #[test]
    fn test_invalid_latitude_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_polygon = vec![(91.0, 0.0)];
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("Latitude"));
    }

    #[test]
    fn test_invalid_longitude_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_polygon = vec![(0.0, 181.0)];
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("Longitude"));
    }

    #[test]
    fn test_nan_coordinates_rejected() {
        let mut j = make_jurisdiction();
        j.entry.zone_polygon = vec![(f64::NAN, 0.0)];
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("finite"));
    }

    #[test]
    fn test_no_regulatory_tags_rejected() {
        let mut j = make_jurisdiction();
        j.entry.regulatory_tags = vec![];
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("At least one"));
    }

    #[test]
    fn test_empty_tag_rejected() {
        let mut j = make_jurisdiction();
        j.entry.regulatory_tags = vec!["gdpr".into(), "".into()];
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("empty strings"));
    }

    #[test]
    fn test_enforcement_nan_rejected() {
        let mut j = make_jurisdiction();
        j.entry.enforcement_risk = f64::NAN;
        assert!(check_create_jurisdiction(&j).is_err());
    }

    #[test]
    fn test_enforcement_out_of_range_rejected() {
        let mut j = make_jurisdiction();
        j.entry.enforcement_risk = 1.01;
        assert!(check_create_jurisdiction(&j).is_err());

        j.entry.enforcement_risk = -0.01;
        assert!(check_create_jurisdiction(&j).is_err());
    }

    #[test]
    fn test_enforcement_boundaries_accepted() {
        let mut j = make_jurisdiction();
        j.entry.enforcement_risk = 0.0;
        assert!(check_create_jurisdiction(&j).is_ok());

        j.entry.enforcement_risk = 1.0;
        assert!(check_create_jurisdiction(&j).is_ok());
    }

    #[test]
    fn test_long_description_rejected() {
        let mut j = make_jurisdiction();
        j.entry.description = "d".repeat(4097);
        let err = check_create_jurisdiction(&j).unwrap_err();
        assert!(err.contains("4096"));
    }

    #[test]
    fn test_optional_fields_none_accepted() {
        let mut j = make_jurisdiction();
        j.entry.fiat_zone = None;
        j.entry.authority_did = None;
        assert!(check_create_jurisdiction(&j).is_ok());
    }

    // --- Update validation ---

    #[test]
    fn test_valid_update_accepted() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.enforcement_risk = 0.90;
        assert!(check_update_jurisdiction(&orig, &updated).is_ok());
    }

    #[test]
    fn test_cannot_change_zone_id() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.zone_id = "different-zone".into();
        let err = check_update_jurisdiction(&orig, &updated).unwrap_err();
        assert!(err.contains("zone_id"));
    }

    #[test]
    fn test_version_must_increment_by_1() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 1; // same version
        let err = check_update_jurisdiction(&orig, &updated).unwrap_err();
        assert!(err.contains("Version"));

        let mut updated2 = orig.clone();
        updated2.version = 3; // skipped
        let err2 = check_update_jurisdiction(&orig, &updated2).unwrap_err();
        assert!(err2.contains("Version"));
    }

    #[test]
    fn test_update_validates_entry_fields() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.enforcement_risk = f64::NAN;
        assert!(check_update_jurisdiction(&orig, &updated).is_err());
    }

    #[test]
    fn test_update_can_change_tags() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.regulatory_tags = vec!["hipaa".into()];
        assert!(check_update_jurisdiction(&orig, &updated).is_ok());
    }

    #[test]
    fn test_update_can_change_polygon() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.zone_polygon = vec![(10.0, 20.0), (30.0, 40.0), (50.0, 20.0)];
        assert!(check_update_jurisdiction(&orig, &updated).is_ok());
    }

    #[test]
    fn test_update_can_change_enforcement_risk() {
        let orig = make_jurisdiction();
        let mut updated = orig.clone();
        updated.version = 2;
        updated.entry.enforcement_risk = 0.10;
        assert!(check_update_jurisdiction(&orig, &updated).is_ok());
    }
}
