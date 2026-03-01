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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MaterialProperties {
    pub tensile_strength_mpa: Option<f32>,
    pub elongation_at_break: Option<f32>,
    pub heat_deflection_temp_c: Option<f32>,
    pub density_g_cm3: Option<f32>,
    pub food_safe: bool,
    pub uv_resistant: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Certification {
    pub cert_type: String,
    pub issuer: String,
    pub valid_until: Option<i64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginationInput {
    pub offset: u32,
    pub limit: u32,
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
// DNA setup helper
// ============================================================================

fn fabrication_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // sweettest/ -> tests/
    path.pop(); // tests/ -> fabrication/
    path.push("workdir");
    path.push("fabrication.dna");
    path
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
            tensile_strength_mpa: Some(50.0),
            elongation_at_break: Some(7.6),
            heat_deflection_temp_c: Some(78.0),
            density_g_cm3: Some(1.27),
            food_safe: true,
            uv_resistant: false,
        },
        certifications: vec![Certification {
            cert_type: "FoodContact".to_string(),
            issuer: "EU Regulation 10/2011".to_string(),
            valid_until: None,
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
                tensile_strength_mpa: Some(37.0 + i as f32),
                elongation_at_break: Some(6.0),
                heat_deflection_temp_c: Some(56.0),
                density_g_cm3: Some(1.24),
                food_safe: false,
                uv_resistant: false,
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
            tensile_strength_mpa: Some(50.0),
            elongation_at_break: Some(7.6),
            heat_deflection_temp_c: Some(78.0),
            density_g_cm3: Some(1.27),
            food_safe: true,
            uv_resistant: false,
        },
        certifications: vec![Certification {
            cert_type: "FoodContact".to_string(),
            issuer: "FDA".to_string(),
            valid_until: None,
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
            tensile_strength_mpa: Some(40.0),
            elongation_at_break: Some(3.5),
            heat_deflection_temp_c: Some(98.0),
            density_g_cm3: Some(1.04),
            food_safe: false,
            uv_resistant: false,
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
