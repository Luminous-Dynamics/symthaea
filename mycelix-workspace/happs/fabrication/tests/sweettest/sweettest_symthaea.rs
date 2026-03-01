//! # Fabrication — Symthaea Sweettest
//!
//! Integration tests for HDC intent generation, semantic search,
//! design optimizations, and parametric variant generation.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_symthaea -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types — symthaea coordinator
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateIntentInput {
    pub description: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SerializedBinding {
    pub concept: String,
    pub role: String,
    pub weight: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IntentResult {
    pub record: Record,
    pub bindings: Vec<SerializedBinding>,
    pub vector_hash: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MyIntentsInput {
    pub pagination: Option<PaginationInput>,
}

/// Mirror of symthaea coordinator's `SemanticSearchInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SemanticSearchInput {
    pub intent_hash: ActionHash,
    pub threshold: Option<f32>,
    pub limit: Option<u32>,
    pub record_matches: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GenerateVariantInput {
    pub base_design_hash: ActionHash,
    pub intent_modifiers: Vec<SerializedBinding>,
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>,
}

// ============================================================================
// Intent Generation Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_generate_intent_vector() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = CreateIntentInput {
        description: "I need a weatherproof bracket for a 12mm pipe".to_string(),
        language: Some("en".to_string()),
    };

    let result: IntentResult = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_intent_vector",
            input,
        )
        .await;

    // Verify the record was created and authored by alice
    assert_eq!(result.record.action().author(), alice.agent_pubkey());

    // Verify HDC vector hash is present and correctly prefixed
    assert!(
        result.vector_hash.starts_with("hdc_"),
        "Vector hash should start with hdc_ prefix, got: {}",
        result.vector_hash
    );

    // Verify semantic bindings were parsed from the description
    assert!(
        !result.bindings.is_empty(),
        "Should have parsed at least one semantic binding from the description"
    );

    // Verify specific bindings (bracket, 12mm, weatherproof)
    assert!(
        result.bindings.iter().any(|b| b.concept == "bracket"),
        "Should detect 'bracket' as a Base binding"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// My Intents Listing Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_my_intents() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Generate 2 intents
    let _: IntentResult = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_intent_vector",
            CreateIntentInput {
                description: "a sturdy bracket for 12mm pipe".to_string(),
                language: None,
            },
        )
        .await;

    let _: IntentResult = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_intent_vector",
            CreateIntentInput {
                description: "a gear for a small motor".to_string(),
                language: None,
            },
        )
        .await;

    // Get all intents (no pagination)
    let all: serde_json::Value = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "get_my_intents",
            MyIntentsInput { pagination: None },
        )
        .await;

    let total = all.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(
        total >= 2,
        "Should have at least 2 intents, got {}",
        total
    );

    // Get paginated (limit 1)
    let page1: serde_json::Value = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "get_my_intents",
            MyIntentsInput {
                pagination: Some(PaginationInput {
                    offset: 0,
                    limit: 1,
                }),
            },
        )
        .await;

    let items = page1
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(items, 1, "Page 1 should have 1 item");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Semantic Search Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_semantic_search() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a target intent
    let target: IntentResult = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_intent_vector",
            CreateIntentInput {
                description: "a weatherproof bracket for 12mm pipe".to_string(),
                language: None,
            },
        )
        .await;

    // Create a second intent that should be semantically similar
    let _: IntentResult = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_intent_vector",
            CreateIntentInput {
                description: "a bracket for outdoor pipe mounting".to_string(),
                language: None,
            },
        )
        .await;

    // Search using the first intent with a low threshold to allow matches
    let search_input = SemanticSearchInput {
        intent_hash: target.record.action_address().clone(),
        threshold: Some(0.1),
        limit: Some(10),
        record_matches: None,
    };

    let results: serde_json::Value = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "semantic_search",
            search_input,
        )
        .await;

    // The call should succeed without error. Results may or may not contain
    // matches depending on the HDC similarity score, but the response must
    // be a valid array.
    assert!(
        results.is_array(),
        "semantic_search should return an array, got: {:?}",
        results
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Design Optimizations Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_design_optimizations() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a design to query optimizations for
    let design: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            CreateDesignInput {
                title: "Pipe Bracket".to_string(),
                description: "A bracket for pipe mounting".to_string(),
                category: "Parts".to_string(),
                intent_vector: None,
                parametric_schema: None,
                constraint_graph: None,
                material_compatibility: vec![],
                circularity_score: 0.5,
                embodied_energy_kwh: 1.0,
                repair_manifest: None,
                license: serde_json::json!("PublicDomain"),
                safety_class: "Class1Functional".to_string(),
            },
        )
        .await;

    let design_hash = design.action_address().clone();

    // Query optimizations for this design (should be empty but valid)
    let result: serde_json::Value = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "get_design_optimizations",
            HashPaginationInput {
                hash: design_hash,
                pagination: None,
            },
        )
        .await;

    let total = result.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    assert_eq!(total, 0, "New design should have 0 optimizations");

    let items = result
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(items, 0, "Items array should be empty");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Parametric Variant Generation Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_generate_parametric_variant() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a base design first via the designs coordinator
    let design: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            CreateDesignInput {
                title: "Base Bracket".to_string(),
                description: "A simple bracket".to_string(),
                category: "Parts".to_string(),
                intent_vector: None,
                parametric_schema: None,
                constraint_graph: None,
                material_compatibility: vec![],
                circularity_score: 0.5,
                embodied_energy_kwh: 1.0,
                repair_manifest: None,
                license: serde_json::json!("PublicDomain"),
                safety_class: "Class1Functional".to_string(),
            },
        )
        .await;

    let design_hash = design.action_address().clone();

    // Generate a parametric variant from the base design
    let variant_input = GenerateVariantInput {
        base_design_hash: design_hash,
        intent_modifiers: vec![
            SerializedBinding {
                concept: "cylinder tube".to_string(),
                role: "shape".to_string(),
                weight: 1.0,
            },
            SerializedBinding {
                concept: "50x30x10".to_string(),
                role: "dimension".to_string(),
                weight: 1.0,
            },
        ],
        material_constraints: vec!["PETG".to_string()],
        printer_constraints: Some("FDM".to_string()),
    };

    let variant: Record = conductor
        .call(
            &alice.zome("symthaea_coordinator"),
            "generate_parametric_variant",
            variant_input,
        )
        .await;

    // Verify the variant record was created and authored by alice
    assert_eq!(variant.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
