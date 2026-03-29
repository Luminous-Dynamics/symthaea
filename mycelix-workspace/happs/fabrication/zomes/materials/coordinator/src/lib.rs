// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Materials Coordinator Zome
//!
//! CRUD operations and discovery for 3D printing materials.

use hdk::prelude::*;
use materials_integrity::*;
use fabrication_common::*;
use std::cell::RefCell;

thread_local! {
    static CONFIG: RefCell<Option<FabricationConfig>> = const { RefCell::new(None) };
}

fn get_config() -> FabricationConfig {
    CONFIG.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                dna_info()
                    .map(|info| FabricationConfig::from_properties_or_default(info.modifiers.properties.bytes()))
                    .unwrap_or_default()
            })
            .clone()
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMaterialInput {
    pub name: String,
    pub material_type: MaterialType,
    pub properties: MaterialProperties,
    pub certifications: Vec<Certification>,
    pub safety_data_sheet: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMaterialsByTypeInput {
    pub material_type: MaterialType,
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetFoodSafeMaterialsInput {
    pub pagination: Option<PaginationInput>,
}

// =============================================================================
// RATE LIMITING
// =============================================================================

fn rate_limit_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("rate_limit:{}", agent).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn enforce_rate_limit(caller: &AgentPubKey) -> ExternResult<()> {
    let cfg = get_config();
    let max_ops = cfg.rate_limit_max_ops as usize;
    let window_micros = cfg.rate_limit_window_secs as i64 * 1_000_000;

    let anchor = rate_limit_anchor(caller)?;
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::RateLimitBucket)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let window_start = now.as_micros() - window_micros;

    let recent_count = links
        .iter()
        .filter(|l| l.timestamp.as_micros() >= window_start)
        .count();

    if recent_count >= max_ops {
        return Err(FabricationError::RateLimited {
            max_ops: cfg.rate_limit_max_ops,
            window_secs: cfg.rate_limit_window_secs,
        }.to_wasm_error());
    }

    create_link(anchor.clone(), anchor, LinkTypes::RateLimitBucket, ())?;
    Ok(())
}

fn rate_limit_caller() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    enforce_rate_limit(&agent)
}

// =============================================================================
// CRUD
// =============================================================================

#[hdk_extern]
pub fn create_material(input: CreateMaterialInput) -> ExternResult<Record> {
    rate_limit_caller()?;
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Material,
        event_type: FabricationEventType::MaterialCreated,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    let anchor = all_materials_anchor()?;
    create_link(anchor, hash.clone(), LinkTypes::AllMaterials, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("Material", &hash))
}

#[hdk_extern]
pub fn get_material(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_materials_by_type(input: GetMaterialsByTypeInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = all_materials_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllMaterials)?, GetStrategy::default())?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(mat) = record.entry().to_app_option::<Material>().ok().flatten() {
                    if mat.material_type == input.material_type {
                        results.push(record);
                    }
                }
            }
        }
    }
    Ok(paginate(results, input.pagination.as_ref()))
}

#[hdk_extern]
pub fn get_food_safe_materials(input: GetFoodSafeMaterialsInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = all_materials_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllMaterials)?, GetStrategy::default())?;

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
    Ok(paginate(results, input.pagination.as_ref()))
}

// =============================================================================
// UPDATE / DELETE
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMaterialInput {
    pub original_action_hash: ActionHash,
    pub name: Option<String>,
    pub material_type: Option<MaterialType>,
    pub properties: Option<MaterialProperties>,
    pub certifications: Option<Vec<Certification>>,
    pub safety_data_sheet: Option<String>,
}

#[hdk_extern]
pub fn update_material(input: UpdateMaterialInput) -> ExternResult<Record> {
    let original = get(input.original_action_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Material", &input.original_action_hash))?;

    let mut mat: Material = original
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {e}"))))?
        .ok_or(FabricationError::not_found("Material entry", &input.original_action_hash))?;

    // Apply partial updates
    if let Some(name) = input.name { mat.name = name; }
    if let Some(material_type) = input.material_type { mat.material_type = material_type; }
    if let Some(properties) = input.properties { mat.properties = properties; }
    if let Some(certifications) = input.certifications { mat.certifications = certifications; }
    if let Some(sds) = input.safety_data_sheet { mat.safety_data_sheet = Some(sds); }

    let hash = update_entry(input.original_action_hash, mat)?;
    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("Material", &hash))
}

#[hdk_extern]
pub fn delete_material(hash: ActionHash) -> ExternResult<ActionHash> {
    let _ = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Material", &hash))?;
    delete_entry(hash)
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple anchor helper - creates deterministic hash from string
fn all_materials_anchor() -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        "anchor:all_materials".as_bytes().to_vec()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_properties_full() -> MaterialProperties {
        MaterialProperties {
            print_temp_min: 190,
            print_temp_max: 220,
            bed_temp_min: Some(60),
            bed_temp_max: Some(80),
            density_g_cm3: 1.24,
            tensile_strength_mpa: Some(50.0),
            elongation_percent: Some(6.5),
            food_safe: true,
            uv_resistant: false,
            water_resistant: true,
            recyclable: true,
        }
    }

    fn make_properties_minimal() -> MaterialProperties {
        MaterialProperties {
            print_temp_min: 200,
            print_temp_max: 240,
            bed_temp_min: None,
            bed_temp_max: None,
            density_g_cm3: 1.27,
            tensile_strength_mpa: None,
            elongation_percent: None,
            food_safe: false,
            uv_resistant: false,
            water_resistant: false,
            recyclable: false,
        }
    }

    // 1. CreateMaterialInput serde roundtrip
    #[test]
    fn test_create_material_input_serde() {
        let input = CreateMaterialInput {
            name: "Eco PLA".to_string(),
            material_type: MaterialType::PLA,
            properties: make_properties_full(),
            certifications: vec![Certification {
                cert_type: CertificationType::FoodSafe,
                issuer: "NSF International".to_string(),
                valid_until: None,
                document_cid: Some("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".to_string()),
            }],
            safety_data_sheet: Some("ipfs://bafkreih4algyjuxoq7dqm7n5oy3s4n7p6e5x7ygqvnfzn7".to_string()),
        };

        let json = serde_json::to_string(&input).expect("serialize CreateMaterialInput");
        let decoded: CreateMaterialInput =
            serde_json::from_str(&json).expect("deserialize CreateMaterialInput");

        assert_eq!(decoded.name, "Eco PLA");
        assert_eq!(decoded.material_type, MaterialType::PLA);
        assert_eq!(decoded.properties.print_temp_min, 190);
        assert_eq!(decoded.properties.food_safe, true);
        assert_eq!(decoded.certifications.len(), 1);
        assert_eq!(decoded.certifications[0].cert_type, CertificationType::FoodSafe);
        assert_eq!(decoded.certifications[0].issuer, "NSF International");
        assert_eq!(
            decoded.safety_data_sheet.as_deref(),
            Some("ipfs://bafkreih4algyjuxoq7dqm7n5oy3s4n7p6e5x7ygqvnfzn7")
        );
    }

    // 2. All MaterialType variants round-trip
    #[test]
    fn test_material_type_all_variants_serde() {
        let variants: Vec<MaterialType> = vec![
            MaterialType::PLA,
            MaterialType::PETG,
            MaterialType::ABS,
            MaterialType::ASA,
            MaterialType::TPU,
            MaterialType::Nylon,
            MaterialType::PC,
            MaterialType::PEEK,
            MaterialType::PVA,
            MaterialType::HIPS,
            MaterialType::StandardResin,
            MaterialType::ToughResin,
            MaterialType::FlexibleResin,
            MaterialType::CastableResin,
            MaterialType::DentalResin,
            MaterialType::NylonPowder,
            MaterialType::MetalPowder,
            MaterialType::Custom("BioFilament-X".to_string()),
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant)
                .unwrap_or_else(|e| panic!("serialize {:?} failed: {}", variant, e));
            let decoded: MaterialType = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("deserialize {:?} failed: {}", variant, e));
            assert_eq!(
                variant, &decoded,
                "roundtrip mismatch for variant {:?}",
                variant
            );
        }

        // Verify the Custom payload survives intact
        let custom = MaterialType::Custom("BioFilament-X".to_string());
        let json = serde_json::to_string(&custom).unwrap();
        let decoded: MaterialType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, MaterialType::Custom("BioFilament-X".to_string()));
    }

    // 3. MaterialProperties with all optional fields populated
    #[test]
    fn test_material_properties_with_optionals_serde() {
        let props = make_properties_full();

        let json = serde_json::to_string(&props).expect("serialize MaterialProperties (full)");
        let decoded: MaterialProperties =
            serde_json::from_str(&json).expect("deserialize MaterialProperties (full)");

        assert_eq!(decoded.print_temp_min, 190);
        assert_eq!(decoded.print_temp_max, 220);
        assert_eq!(decoded.bed_temp_min, Some(60));
        assert_eq!(decoded.bed_temp_max, Some(80));
        assert!((decoded.density_g_cm3 - 1.24_f32).abs() < 1e-6);
        assert_eq!(decoded.tensile_strength_mpa, Some(50.0_f32));
        assert_eq!(decoded.elongation_percent, Some(6.5_f32));
        assert_eq!(decoded.food_safe, true);
        assert_eq!(decoded.uv_resistant, false);
        assert_eq!(decoded.water_resistant, true);
        assert_eq!(decoded.recyclable, true);
    }

    // 4. MaterialProperties with all Option fields as None
    #[test]
    fn test_material_properties_minimal_serde() {
        let props = make_properties_minimal();

        let json = serde_json::to_string(&props).expect("serialize MaterialProperties (minimal)");
        let decoded: MaterialProperties =
            serde_json::from_str(&json).expect("deserialize MaterialProperties (minimal)");

        assert_eq!(decoded.print_temp_min, 200);
        assert_eq!(decoded.print_temp_max, 240);
        assert_eq!(decoded.bed_temp_min, None);
        assert_eq!(decoded.bed_temp_max, None);
        assert!((decoded.density_g_cm3 - 1.27_f32).abs() < 1e-6);
        assert_eq!(decoded.tensile_strength_mpa, None);
        assert_eq!(decoded.elongation_percent, None);
        assert_eq!(decoded.food_safe, false);
        assert_eq!(decoded.uv_resistant, false);
        assert_eq!(decoded.water_resistant, false);
        assert_eq!(decoded.recyclable, false);
    }

    // 5. GetMaterialsByTypeInput with pagination round-trip
    #[test]
    fn test_get_materials_by_type_input_serde() {
        // With pagination present
        let with_page = GetMaterialsByTypeInput {
            material_type: MaterialType::PEEK,
            pagination: Some(PaginationInput { offset: 10, limit: 25 }),
        };
        let json = serde_json::to_string(&with_page)
            .expect("serialize GetMaterialsByTypeInput (with pagination)");
        let decoded: GetMaterialsByTypeInput = serde_json::from_str(&json)
            .expect("deserialize GetMaterialsByTypeInput (with pagination)");
        assert_eq!(decoded.material_type, MaterialType::PEEK);
        let page = decoded.pagination.expect("pagination should be Some");
        assert_eq!(page.offset, 10);
        assert_eq!(page.limit, 25);

        // With pagination absent
        let no_page = GetMaterialsByTypeInput {
            material_type: MaterialType::Custom("PolyFlex".to_string()),
            pagination: None,
        };
        let json2 = serde_json::to_string(&no_page)
            .expect("serialize GetMaterialsByTypeInput (no pagination)");
        let decoded2: GetMaterialsByTypeInput = serde_json::from_str(&json2)
            .expect("deserialize GetMaterialsByTypeInput (no pagination)");
        assert_eq!(
            decoded2.material_type,
            MaterialType::Custom("PolyFlex".to_string())
        );
        assert!(decoded2.pagination.is_none());
    }
}
