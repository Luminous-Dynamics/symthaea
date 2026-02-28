//! Materials Integrity Zome
//!
//! Defines entry types for 3D printing materials with certifications,
//! properties, and Supply Chain integration for circular economy tracking.

use hdi::prelude::*;
use fabrication_common::*;

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
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Dispatch entry validation to type-specific validators.
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::Material(m) => validate_material(m),
        EntryTypes::MaterialBatch(b) => validate_material_batch(b),
    }
}

/// Validate a Material entry with full security hardening.
fn validate_material(m: Material) -> ExternResult<ValidateCallbackResult> {
    // id: non-empty, max 64 chars
    check!(validation::require_non_empty(&m.id, "id"));
    check!(validation::require_max_len(&m.id, 64, "id"));

    // name: non-empty (trim), max 256 chars
    check!(validation::require_non_empty(&m.name, "name"));
    check!(validation::require_max_len(&m.name, 256, "name"));

    // density_g_cm3: must be in range 0.001..100.0
    check!(validation::require_in_range(
        m.properties.density_g_cm3,
        0.001,
        100.0,
        "density_g_cm3"
    ));

    // tensile_strength_mpa: if Some, require finite and >= 0
    if let Some(ts) = m.properties.tensile_strength_mpa {
        check!(validation::require_finite(ts, "tensile_strength_mpa"));
        if ts < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "tensile_strength_mpa must be >= 0".to_string(),
            ));
        }
    }

    // elongation_percent: if Some, require in range 0.0..1000.0
    if let Some(ep) = m.properties.elongation_percent {
        check!(validation::require_in_range(
            ep,
            0.0,
            1000.0,
            "elongation_percent"
        ));
    }

    // certifications: max 32 items
    check!(validation::require_max_vec_len(
        &m.certifications,
        32,
        "certifications"
    ));

    // suppliers: max 64 items
    check!(validation::require_max_vec_len(
        &m.suppliers,
        64,
        "suppliers"
    ));

    // safety_data_sheet: if Some, max 256 chars
    if let Some(ref sds) = m.safety_data_sheet {
        check!(validation::require_max_len(sds, 256, "safety_data_sheet"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a MaterialBatch entry with full security hardening.
fn validate_material_batch(b: MaterialBatch) -> ExternResult<ValidateCallbackResult> {
    // batch_id: non-empty, max 64 chars
    check!(validation::require_non_empty(&b.batch_id, "batch_id"));
    check!(validation::require_max_len(&b.batch_id, 64, "batch_id"));

    // recycled_content_percent: require in range 0.0..100.0
    check!(validation::require_in_range(
        b.recycled_content_percent,
        0.0,
        100.0,
        "recycled_content_percent"
    ));

    // quantity_kg: require finite and > 0
    check!(validation::require_finite(b.quantity_kg, "quantity_kg"));
    if b.quantity_kg <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "quantity_kg must be > 0".to_string(),
        ));
    }

    // certifications: max 32 items, each max 256 chars
    check!(validation::require_max_vec_len(
        &b.certifications,
        32,
        "certifications"
    ));
    for (i, cert) in b.certifications.iter().enumerate() {
        check!(validation::require_max_len(
            cert,
            256,
            &format!("certifications[{}]", i)
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a valid Material with sensible defaults.
    fn valid_material() -> Material {
        Material {
            id: "mat-001".to_string(),
            name: "PLA Filament".to_string(),
            material_type: MaterialType::PLA,
            properties: MaterialProperties {
                print_temp_min: 190,
                print_temp_max: 220,
                bed_temp_min: Some(50),
                bed_temp_max: Some(70),
                density_g_cm3: 1.24,
                tensile_strength_mpa: Some(50.0),
                elongation_percent: Some(6.0),
                food_safe: false,
                uv_resistant: false,
                water_resistant: false,
                recyclable: true,
            },
            certifications: vec![],
            suppliers: vec![],
            safety_data_sheet: None,
            created_at: Timestamp::from_micros(0),
        }
    }

    /// Helper: build a valid MaterialBatch with sensible defaults.
    fn valid_batch() -> MaterialBatch {
        MaterialBatch {
            material_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            batch_id: "batch-001".to_string(),
            origin: MaterialOrigin::Virgin,
            recycled_content_percent: 0.0,
            supply_chain_hash: None,
            certifications: vec![],
            end_of_life: EndOfLifeStrategy::MechanicalRecycling,
            quantity_kg: 25.0,
            received_at: Timestamp::from_micros(0),
        }
    }

    // =========================================================================
    // Material tests
    // =========================================================================

    #[test]
    fn valid_material_passes() {
        let result = validate_material(valid_material()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn empty_name_rejected() {
        let mut m = valid_material();
        m.name = "   ".to_string();
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("name")));
    }

    #[test]
    fn nan_density_rejected() {
        let mut m = valid_material();
        m.properties.density_g_cm3 = f32::NAN;
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("density_g_cm3")));
    }

    #[test]
    fn density_out_of_range_rejected() {
        let mut m = valid_material();
        m.properties.density_g_cm3 = 0.0001;
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("density_g_cm3")));

        let mut m2 = valid_material();
        m2.properties.density_g_cm3 = 200.0;
        let result2 = validate_material(m2).unwrap();
        assert!(matches!(result2, ValidateCallbackResult::Invalid(msg) if msg.contains("density_g_cm3")));
    }

    #[test]
    fn negative_tensile_strength_rejected() {
        let mut m = valid_material();
        m.properties.tensile_strength_mpa = Some(-1.0);
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("tensile_strength_mpa")));
    }

    #[test]
    fn elongation_nan_rejected() {
        let mut m = valid_material();
        m.properties.elongation_percent = Some(f32::NAN);
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("elongation_percent")));
    }

    #[test]
    fn too_many_certifications_rejected() {
        let mut m = valid_material();
        m.certifications = (0..33)
            .map(|_| Certification {
                cert_type: CertificationType::ISO,
                issuer: "Test".to_string(),
                valid_until: None,
                document_cid: None,
            })
            .collect();
        let result = validate_material(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("certifications")));
    }

    // =========================================================================
    // MaterialBatch tests
    // =========================================================================

    #[test]
    fn valid_batch_passes() {
        let result = validate_material_batch(valid_batch()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn recycled_percent_nan_rejected() {
        let mut b = valid_batch();
        b.recycled_content_percent = f32::NAN;
        let result = validate_material_batch(b).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("recycled_content_percent")));
    }

    #[test]
    fn recycled_percent_over_100_rejected() {
        let mut b = valid_batch();
        b.recycled_content_percent = 101.0;
        let result = validate_material_batch(b).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("recycled_content_percent")));
    }

    #[test]
    fn quantity_zero_rejected() {
        let mut b = valid_batch();
        b.quantity_kg = 0.0;
        let result = validate_material_batch(b).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("quantity_kg")));
    }

    #[test]
    fn batch_id_empty_rejected() {
        let mut b = valid_batch();
        b.batch_id = "".to_string();
        let result = validate_material_batch(b).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("batch_id")));
    }
}
