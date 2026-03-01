//! # Fabrication — Materials Sweettest
//!
//! Integration tests for material creation, retrieval, type-based
//! discovery, and food-safe filtering.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_materials -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types — materials coordinator
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateMaterialInput {
    pub name: String,
    pub material_type: String,
    pub properties: MaterialProperties,
    pub certifications: Vec<Certification>,
    pub safety_data_sheet: Option<String>,
}

/// Mirror of fabrication_common's `MaterialProperties`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MaterialProperties {
    pub print_temp_min: u16,
    pub print_temp_max: u16,
    pub bed_temp_min: Option<u16>,
    pub bed_temp_max: Option<u16>,
    pub density_g_cm3: f32,
    pub tensile_strength_mpa: Option<f32>,
    pub elongation_percent: Option<f32>,
    pub food_safe: bool,
    pub uv_resistant: bool,
    pub water_resistant: bool,
    pub recyclable: bool,
}

/// Mirror of fabrication_common's `Certification`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Certification {
    pub cert_type: String,
    pub issuer: String,
    pub valid_until: Option<Timestamp>,
    pub document_cid: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetMaterialsByTypeInput {
    pub material_type: String,
    pub pagination: Option<PaginationInput>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetFoodSafeMaterialsInput {
    pub pagination: Option<PaginationInput>,
}

// ============================================================================
// Material CRUD Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_material_create_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = CreateMaterialInput {
        name: "Prusament PETG".to_string(),
        material_type: "PETG".to_string(),
        properties: MaterialProperties {
            print_temp_min: 220,
            print_temp_max: 250,
            bed_temp_min: Some(70),
            bed_temp_max: Some(85),
            density_g_cm3: 1.27,
            tensile_strength_mpa: Some(50.0),
            elongation_percent: Some(7.6),
            food_safe: true,
            uv_resistant: false,
            water_resistant: true,
            recyclable: true,
        },
        certifications: vec![Certification {
            cert_type: "FoodSafe".to_string(),
            issuer: "EU Regulation 10/2011".to_string(),
            valid_until: None,
            document_cid: None,
        }],
        safety_data_sheet: Some("https://example.com/sds/petg.pdf".to_string()),
    };

    let record: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "create_material",
            input,
        )
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Retrieve it
    let material_hash = record.action_address().clone();
    let retrieved: Option<Record> = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "get_material",
            material_hash,
        )
        .await;

    assert!(retrieved.is_some());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_materials_by_type_paginated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create 3 PLA materials
    for i in 0..3 {
        let input = CreateMaterialInput {
            name: format!("PLA Brand #{}", i),
            material_type: "PLA".to_string(),
            properties: MaterialProperties {
                print_temp_min: 190,
                print_temp_max: 220,
                bed_temp_min: Some(50),
                bed_temp_max: Some(70),
                density_g_cm3: 1.24,
                tensile_strength_mpa: Some(37.0 + i as f32),
                elongation_percent: Some(6.0),
                food_safe: false,
                uv_resistant: false,
                water_resistant: false,
                recyclable: true,
            },
            certifications: vec![],
            safety_data_sheet: None,
        };

        let _: Record = conductor
            .call(
                &alice.zome("materials_coordinator"),
                "create_material",
                input,
            )
            .await;
    }

    // Query by type
    let result: serde_json::Value = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "get_materials_by_type",
            GetMaterialsByTypeInput {
                material_type: "PLA".to_string(),
                pagination: None,
            },
        )
        .await;

    let total = result.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(total >= 3, "Should have at least 3 PLA materials, got {}", total);

    // Paginated query
    let page: serde_json::Value = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "get_materials_by_type",
            GetMaterialsByTypeInput {
                material_type: "PLA".to_string(),
                pagination: Some(PaginationInput {
                    offset: 0,
                    limit: 2,
                }),
            },
        )
        .await;

    let items = page
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(items, 2, "Page should have 2 items");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_food_safe_materials() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a food-safe material
    let food_safe_input = CreateMaterialInput {
        name: "Food-Safe PETG".to_string(),
        material_type: "PETG".to_string(),
        properties: MaterialProperties {
            print_temp_min: 220,
            print_temp_max: 250,
            bed_temp_min: Some(70),
            bed_temp_max: Some(85),
            density_g_cm3: 1.27,
            tensile_strength_mpa: Some(50.0),
            elongation_percent: Some(7.6),
            food_safe: true,
            uv_resistant: false,
            water_resistant: true,
            recyclable: true,
        },
        certifications: vec![Certification {
            cert_type: "FoodSafe".to_string(),
            issuer: "FDA".to_string(),
            valid_until: None,
            document_cid: None,
        }],
        safety_data_sheet: None,
    };

    let _: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "create_material",
            food_safe_input,
        )
        .await;

    // Create a non-food-safe material
    let regular_input = CreateMaterialInput {
        name: "Regular ABS".to_string(),
        material_type: "ABS".to_string(),
        properties: MaterialProperties {
            print_temp_min: 230,
            print_temp_max: 260,
            bed_temp_min: Some(90),
            bed_temp_max: Some(110),
            density_g_cm3: 1.04,
            tensile_strength_mpa: Some(40.0),
            elongation_percent: Some(3.5),
            food_safe: false,
            uv_resistant: false,
            water_resistant: false,
            recyclable: false,
        },
        certifications: vec![],
        safety_data_sheet: None,
    };

    let _: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "create_material",
            regular_input,
        )
        .await;

    // Query food-safe
    let result: serde_json::Value = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "get_food_safe_materials",
            GetFoodSafeMaterialsInput { pagination: None },
        )
        .await;

    let items = result
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        items >= 1,
        "Should have at least 1 food-safe material, got {}",
        items
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Update + Delete Tests
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateMaterialInput {
    pub original_action_hash: ActionHash,
    pub name: Option<String>,
    pub material_type: Option<String>,
    pub properties: Option<MaterialProperties>,
    pub certifications: Option<Vec<Certification>>,
    pub safety_data_sheet: Option<String>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_material_update() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create
    let input = CreateMaterialInput {
        name: "Original Name".to_string(),
        material_type: "PLA".to_string(),
        properties: MaterialProperties {
            print_temp_min: 190,
            print_temp_max: 220,
            bed_temp_min: Some(50),
            bed_temp_max: Some(70),
            density_g_cm3: 1.24,
            tensile_strength_mpa: None,
            elongation_percent: None,
            food_safe: false,
            uv_resistant: false,
            water_resistant: false,
            recyclable: true,
        },
        certifications: vec![],
        safety_data_sheet: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "create_material",
            input,
        )
        .await;

    let original_hash = record.action_address().clone();

    // Update name only
    let update_input = UpdateMaterialInput {
        original_action_hash: original_hash.clone(),
        name: Some("Updated Name".to_string()),
        material_type: None,
        properties: None,
        certifications: None,
        safety_data_sheet: None,
    };

    let updated: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "update_material",
            update_input,
        )
        .await;

    // Updated record should have a different action hash
    assert_ne!(updated.action_address(), &original_hash);
    assert_eq!(updated.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_material_delete() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create
    let input = CreateMaterialInput {
        name: "To Be Deleted".to_string(),
        material_type: "ABS".to_string(),
        properties: MaterialProperties {
            print_temp_min: 230,
            print_temp_max: 260,
            bed_temp_min: Some(90),
            bed_temp_max: Some(110),
            density_g_cm3: 1.04,
            tensile_strength_mpa: None,
            elongation_percent: None,
            food_safe: false,
            uv_resistant: false,
            water_resistant: false,
            recyclable: false,
        },
        certifications: vec![],
        safety_data_sheet: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "create_material",
            input,
        )
        .await;

    let material_hash = record.action_address().clone();

    // Delete
    let delete_hash: ActionHash = conductor
        .call(
            &alice.zome("materials_coordinator"),
            "delete_material",
            material_hash,
        )
        .await;

    // Delete should return an action hash
    assert_ne!(delete_hash, ActionHash::from_raw_36(vec![0u8; 36]));

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
