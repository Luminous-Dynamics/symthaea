// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Circular Economy Sweettest — Commons Cluster
//!
//! Integration tests for the waste/circular economy zomes:
//! waste-registry, waste-collection, compost-control, circular-marketplace.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_circular_economy -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — waste-registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WasteCategory {
    Organic,
    Recyclable,
    Hazardous,
    Electronic,
    Mixed,
    Inert,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContaminationLevel {
    Clean,
    LightlyContaminated,
    Contaminated,
    Hazardous,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EndOfLifeStrategy {
    MechanicalRecycling,
    ChemicalRecycling,
    Biodegradable,
    IndustrialCompost,
    Downcycle,
    Landfill,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WasteStreamStatus {
    Registered,
    Classified,
    Routed,
    Collected,
    Processing,
    Completed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ClassificationMethod {
    Manual,
    SensorBased,
    VisionAI,
    MaterialPassport,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WasteStream {
    pub id: String,
    pub source_did: String,
    pub material_passport_hash: Option<ActionHash>,
    pub waste_category: WasteCategory,
    pub subcategory: String,
    pub quantity_kg: f64,
    pub contamination_level: ContaminationLevel,
    pub recommended_eol: EndOfLifeStrategy,
    pub location_lat: f64,
    pub location_lon: f64,
    pub status: WasteStreamStatus,
    pub created_at: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WasteClassification {
    pub stream_hash: ActionHash,
    pub classifier_did: String,
    pub category: WasteCategory,
    pub subcategory: String,
    pub contamination_level: ContaminationLevel,
    pub confidence: f32,
    pub method: ClassificationMethod,
    pub classified_at: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FacilityType {
    MRF,
    Composter,
    AnaerobicDigester,
    ChemicalRecycler,
    Landfill,
    TransferStation,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FacilityStatus {
    Active,
    Maintenance,
    AtCapacity,
    Closed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WasteFacility {
    pub id: String,
    pub name: String,
    pub facility_type: FacilityType,
    pub accepts: Vec<WasteCategory>,
    pub capacity_kg_per_day: f64,
    pub current_load_kg: f64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub operator_did: String,
    pub status: FacilityStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RouteWasteStreamInput {
    pub stream_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiversionRateInput {
    pub period_start_us: u64,
    pub period_end_us: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiversionRateResult {
    pub total_kg: f64,
    pub composted_kg: f64,
    pub recycled_kg: f64,
    pub landfilled_kg: f64,
    pub diversion_rate_pct: f64,
}

// ============================================================================
// Mirror types — compost-control
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompostMethod {
    Windrow,
    StaticAerated,
    InVessel,
    Vermicompost,
    Bokashi,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompostBatchStatus {
    Mixing,
    ActiveComposting,
    Curing,
    Screening,
    Complete,
    Failed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompostInput {
    pub stream_hash: ActionHash,
    pub quantity_kg: f64,
    pub resource_type: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PhaseTransition {
    pub from: CompostBatchStatus,
    pub to: CompostBatchStatus,
    pub timestamp_us: u64,
    pub trigger: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompostBatch {
    pub id: String,
    pub facility_hash: ActionHash,
    pub method: CompostMethod,
    pub inputs: Vec<CompostInput>,
    pub carbon_nitrogen_ratio: f64,
    pub status: CompostBatchStatus,
    pub started_at: u64,
    pub target_complete: u64,
    pub phase_history: Vec<PhaseTransition>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompostReading {
    pub batch_hash: ActionHash,
    pub sensor_id: String,
    pub temperature_c: f64,
    pub moisture_pct: f64,
    pub oxygen_pct: f64,
    pub ph: f64,
    pub timestamp_us: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchEvaluation {
    pub current_phase: String,
    pub recommended_actions: Vec<RecommendedAction>,
    pub phase_transition: Option<String>,
    pub temperature_status: String,
    pub moisture_status: String,
    pub oxygen_status: String,
    pub ph_status: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecommendedAction {
    pub action_type: CompostActionType,
    pub reason: String,
    pub urgency: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompostActionType {
    Turn,
    AddWater,
    AddBulking,
    AdjustAeration,
    HarvestScreen,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CarbonAttribution {
    pub waste_diverted_kg: f64,
    pub co2e_avoided_tonnes: f64,
    pub methodology: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NutrientEstimate {
    pub total_kg: f64,
    pub nitrogen_pct: f64,
    pub phosphorus_pct: f64,
    pub potassium_pct: f64,
}

// ============================================================================
// Mirror types — circular-marketplace
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SecondaryMaterialType {
    Compost,
    Vermicompost,
    Digestate,
    Biochar,
    RecycledPlastic,
    RecycledMetal,
    RecycledGlass,
    RecycledFiber,
    Mulch,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MaterialQualityGrade {
    Premium,
    Standard,
    Industrial,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Withdrawn,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NutrientInfo {
    pub nitrogen_pct: f64,
    pub phosphorus_pct: f64,
    pub potassium_pct: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SecondaryMaterialListing {
    pub facility_hash: ActionHash,
    pub batch_hash: Option<ActionHash>,
    pub material_type: SecondaryMaterialType,
    pub quantity_kg: f64,
    pub quality_grade: MaterialQualityGrade,
    pub nutrient_info: Option<NutrientInfo>,
    pub price_per_kg: f64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub available_from: u64,
    pub status: ListingStatus,
    pub seller_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemandMatchInput {
    pub material_type: SecondaryMaterialType,
    pub min_quantity_kg: f64,
    pub min_quality: MaterialQualityGrade,
    pub buyer_lat: f64,
    pub buyer_lon: f64,
    pub max_distance_km: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListingMatch {
    pub listing_hash: ActionHash,
    pub material_type: String,
    pub quantity_kg: f64,
    pub quality_grade: String,
    pub price_per_kg: f64,
    pub distance_km: f64,
    pub match_score: f32,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("COMMONS_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ → mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Waste Registry Tests
// ============================================================================

/// Register a waste stream and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_waste_stream_register_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let stream = WasteStream {
        id: "WS-TEST-001".to_string(),
        source_did: "did:example:household1".to_string(),
        material_passport_hash: None,
        waste_category: WasteCategory::Organic,
        subcategory: "food scraps".to_string(),
        quantity_kg: 25.0,
        contamination_level: ContaminationLevel::Clean,
        recommended_eol: EndOfLifeStrategy::IndustrialCompost,
        location_lat: 32.9483,
        location_lon: -96.7299,
        status: WasteStreamStatus::Registered,
        created_at: 1711100000_000000,
    };

    let record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_waste_stream", stream)
        .await;

    assert!(
        record.entry().as_option().is_some(),
        "Should get back the created waste stream"
    );
}

/// Register a facility and verify it appears in all_facilities.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_facility_register_and_list() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let facility = WasteFacility {
        id: "FAC-TEST-001".to_string(),
        name: "Test Composting Hub".to_string(),
        facility_type: FacilityType::Composter,
        accepts: vec![WasteCategory::Organic],
        capacity_kg_per_day: 500.0,
        current_load_kg: 100.0,
        location_lat: 32.9500,
        location_lon: -96.7300,
        operator_did: "did:example:operator1".to_string(),
        status: FacilityStatus::Active,
    };

    let _record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_facility", facility)
        .await;

    let all: Vec<Record> = conductor
        .call(&alice.zome("waste_registry"), "get_all_facilities", ())
        .await;

    assert!(
        !all.is_empty(),
        "Should have at least one facility after registration"
    );
}

/// Register stream + facility, then route stream to facility.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_waste_stream_routing() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Register facility first
    let facility = WasteFacility {
        id: "FAC-ROUTE-001".to_string(),
        name: "Composting Facility".to_string(),
        facility_type: FacilityType::Composter,
        accepts: vec![WasteCategory::Organic],
        capacity_kg_per_day: 1000.0,
        current_load_kg: 200.0,
        location_lat: 32.9500,
        location_lon: -96.7300,
        operator_did: "did:example:operator1".to_string(),
        status: FacilityStatus::Active,
    };

    let _fac_record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_facility", facility)
        .await;

    // Register waste stream nearby
    let stream = WasteStream {
        id: "WS-ROUTE-001".to_string(),
        source_did: "did:example:household1".to_string(),
        material_passport_hash: None,
        waste_category: WasteCategory::Organic,
        subcategory: "yard waste".to_string(),
        quantity_kg: 50.0,
        contamination_level: ContaminationLevel::Clean,
        recommended_eol: EndOfLifeStrategy::IndustrialCompost,
        location_lat: 32.9490,
        location_lon: -96.7290,
        status: WasteStreamStatus::Registered,
        created_at: 1711100000_000000,
    };

    let stream_record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_waste_stream", stream)
        .await;

    let stream_hash = stream_record.action_address().clone();

    // Route the stream
    let route_input = RouteWasteStreamInput { stream_hash };

    let route_result: Option<Record> = conductor
        .call(&alice.zome("waste_registry"), "route_waste_stream", route_input)
        .await;

    assert!(
        route_result.is_some(),
        "Should find a matching facility for organic waste"
    );
}

/// Classify a waste stream with AI confidence.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_waste_classification() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Register a stream
    let stream = WasteStream {
        id: "WS-CLASS-001".to_string(),
        source_did: "did:example:source1".to_string(),
        material_passport_hash: None,
        waste_category: WasteCategory::Mixed,
        subcategory: "unsorted".to_string(),
        quantity_kg: 30.0,
        contamination_level: ContaminationLevel::LightlyContaminated,
        recommended_eol: EndOfLifeStrategy::Downcycle,
        location_lat: 32.9483,
        location_lon: -96.7299,
        status: WasteStreamStatus::Registered,
        created_at: 1711100000_000000,
    };

    let stream_record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_waste_stream", stream)
        .await;

    // Classify it
    let classification = WasteClassification {
        stream_hash: stream_record.action_address().clone(),
        classifier_did: "did:example:symthaea_ai".to_string(),
        category: WasteCategory::Recyclable,
        subcategory: "PET plastic".to_string(),
        contamination_level: ContaminationLevel::Clean,
        confidence: 0.92,
        method: ClassificationMethod::VisionAI,
        classified_at: 1711200000_000000,
    };

    let class_record: Record = conductor
        .call(
            &alice.zome("waste_registry"),
            "classify_waste_stream",
            classification,
        )
        .await;

    assert!(class_record.entry().as_option().is_some());

    // Retrieve classifications
    let classifications: Vec<Record> = conductor
        .call(
            &alice.zome("waste_registry"),
            "get_classifications",
            stream_record.action_address().clone(),
        )
        .await;

    assert_eq!(
        classifications.len(),
        1,
        "Should have exactly one classification"
    );
}

// ============================================================================
// Compost Control Tests
// ============================================================================

/// Create a compost batch, record sensor readings, evaluate status.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_compost_batch_lifecycle() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);

    let batch = CompostBatch {
        id: "CB-TEST-001".to_string(),
        facility_hash: dummy_hash.clone(),
        method: CompostMethod::Windrow,
        inputs: vec![CompostInput {
            stream_hash: dummy_hash.clone(),
            quantity_kg: 100.0,
            resource_type: "KitchenWaste".to_string(),
        }],
        carbon_nitrogen_ratio: 28.0,
        status: CompostBatchStatus::ActiveComposting,
        started_at: 1711100000_000000,
        target_complete: 1714100000_000000,
        phase_history: vec![],
    };

    let batch_record: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "create_compost_batch",
            batch,
        )
        .await;

    let batch_hash = batch_record.action_address().clone();

    // Record a thermophilic reading
    let reading = CompostReading {
        batch_hash: batch_hash.clone(),
        sensor_id: "SENSOR-001".to_string(),
        temperature_c: 58.0,
        moisture_pct: 55.0,
        oxygen_pct: 15.0,
        ph: 7.0,
        timestamp_us: 1711200000_000000,
    };

    let _: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "record_compost_reading",
            reading,
        )
        .await;

    // Evaluate batch status
    let evaluation: BatchEvaluation = conductor
        .call(
            &alice.zome("compost_control"),
            "evaluate_batch_status",
            batch_hash.clone(),
        )
        .await;

    assert_eq!(evaluation.current_phase, "ActiveComposting");
    assert!(
        evaluation.temperature_status.contains("OK"),
        "58C should be OK for active composting: {}",
        evaluation.temperature_status
    );
}

/// Low moisture triggers AddWater recommendation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_compost_low_moisture_recommendation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);

    let batch = CompostBatch {
        id: "CB-DRY-001".to_string(),
        facility_hash: dummy_hash.clone(),
        method: CompostMethod::Windrow,
        inputs: vec![CompostInput {
            stream_hash: dummy_hash.clone(),
            quantity_kg: 200.0,
            resource_type: "GreenWaste".to_string(),
        }],
        carbon_nitrogen_ratio: 25.0,
        status: CompostBatchStatus::ActiveComposting,
        started_at: 1711100000_000000,
        target_complete: 1714100000_000000,
        phase_history: vec![],
    };

    let batch_record: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "create_compost_batch",
            batch,
        )
        .await;

    // Record a dry reading (moisture 25% < 40% minimum)
    let reading = CompostReading {
        batch_hash: batch_record.action_address().clone(),
        sensor_id: "SENSOR-002".to_string(),
        temperature_c: 55.0,
        moisture_pct: 25.0, // Way too dry
        oxygen_pct: 15.0,
        ph: 7.0,
        timestamp_us: 1711200000_000000,
    };

    let _: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "record_compost_reading",
            reading,
        )
        .await;

    let evaluation: BatchEvaluation = conductor
        .call(
            &alice.zome("compost_control"),
            "evaluate_batch_status",
            batch_record.action_address().clone(),
        )
        .await;

    assert!(
        !evaluation.recommended_actions.is_empty(),
        "Low moisture should trigger action recommendations"
    );
    assert!(
        evaluation.moisture_status.contains("Low"),
        "Moisture status should indicate Low: {}",
        evaluation.moisture_status
    );
}

/// Carbon attribution for a completed batch.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_carbon_attribution() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);

    let batch = CompostBatch {
        id: "CB-CARBON-001".to_string(),
        facility_hash: dummy_hash.clone(),
        method: CompostMethod::StaticAerated,
        inputs: vec![
            CompostInput {
                stream_hash: dummy_hash.clone(),
                quantity_kg: 500.0,
                resource_type: "KitchenWaste".to_string(),
            },
            CompostInput {
                stream_hash: dummy_hash.clone(),
                quantity_kg: 500.0,
                resource_type: "GreenWaste".to_string(),
            },
        ],
        carbon_nitrogen_ratio: 27.0,
        status: CompostBatchStatus::Complete,
        started_at: 1711100000_000000,
        target_complete: 1714100000_000000,
        phase_history: vec![],
    };

    let batch_record: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "create_compost_batch",
            batch,
        )
        .await;

    let attribution: CarbonAttribution = conductor
        .call(
            &alice.zome("compost_control"),
            "calculate_carbon_attribution",
            batch_record.action_address().clone(),
        )
        .await;

    // 1000 kg = 1 tonne → 0.06 tCO2e
    assert!(
        (attribution.waste_diverted_kg - 1000.0).abs() < 0.01,
        "Should divert 1000kg, got {}",
        attribution.waste_diverted_kg
    );
    assert!(
        (attribution.co2e_avoided_tonnes - 0.06).abs() < 0.001,
        "Should avoid 0.06 tCO2e, got {}",
        attribution.co2e_avoided_tonnes
    );
    assert!(attribution.methodology.contains("EPA"));
}

/// Nutrient estimation from batch inputs.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_nutrient_estimation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);

    let batch = CompostBatch {
        id: "CB-NPK-001".to_string(),
        facility_hash: dummy_hash.clone(),
        method: CompostMethod::Vermicompost,
        inputs: vec![CompostInput {
            stream_hash: dummy_hash.clone(),
            quantity_kg: 100.0,
            resource_type: "KitchenWaste".to_string(),
        }],
        carbon_nitrogen_ratio: 28.0,
        status: CompostBatchStatus::Mixing,
        started_at: 1711100000_000000,
        target_complete: 1714100000_000000,
        phase_history: vec![],
    };

    let batch_record: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "create_compost_batch",
            batch,
        )
        .await;

    let nutrients: NutrientEstimate = conductor
        .call(
            &alice.zome("compost_control"),
            "get_batch_nutrient_estimate",
            batch_record.action_address().clone(),
        )
        .await;

    assert!((nutrients.total_kg - 100.0).abs() < 0.01);
    assert!(
        nutrients.nitrogen_pct > 0.0,
        "Kitchen waste should have positive nitrogen"
    );
    assert!(
        nutrients.phosphorus_pct > 0.0,
        "Kitchen waste should have positive phosphorus"
    );
    assert!(
        nutrients.potassium_pct > 0.0,
        "Kitchen waste should have positive potassium"
    );
}

// ============================================================================
// Circular Marketplace Tests
// ============================================================================

/// List a secondary material and find it via demand matching.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_marketplace_listing_and_matching() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);

    let listing = SecondaryMaterialListing {
        facility_hash: dummy_hash.clone(),
        batch_hash: None,
        material_type: SecondaryMaterialType::Compost,
        quantity_kg: 500.0,
        quality_grade: MaterialQualityGrade::Standard,
        nutrient_info: Some(NutrientInfo {
            nitrogen_pct: 1.8,
            phosphorus_pct: 0.7,
            potassium_pct: 1.2,
        }),
        price_per_kg: 0.05,
        location_lat: 32.9500,
        location_lon: -96.7300,
        available_from: 1711100000_000000,
        status: ListingStatus::Available,
        seller_did: "did:example:composting_facility".to_string(),
    };

    let listing_record: Record = conductor
        .call(
            &alice.zome("circular_marketplace"),
            "list_secondary_material",
            listing,
        )
        .await;

    assert!(listing_record.entry().as_option().is_some());

    // Search for compost near our location
    let demand = DemandMatchInput {
        material_type: SecondaryMaterialType::Compost,
        min_quantity_kg: 100.0,
        min_quality: MaterialQualityGrade::Industrial,
        buyer_lat: 32.9600,
        buyer_lon: -96.7200,
        max_distance_km: 50.0,
    };

    let matches: Vec<ListingMatch> = conductor
        .call(
            &alice.zome("circular_marketplace"),
            "find_matching_listings",
            demand,
        )
        .await;

    assert!(
        !matches.is_empty(),
        "Should find matching compost listing"
    );
    assert!(
        matches[0].match_score > 0.0,
        "Match score should be positive"
    );
}

/// Full circular loop: waste → compost → marketplace listing.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_full_circular_loop() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Step 1: Register waste stream
    let stream = WasteStream {
        id: "WS-LOOP-001".to_string(),
        source_did: "did:example:farm1".to_string(),
        material_passport_hash: None,
        waste_category: WasteCategory::Organic,
        subcategory: "crop residue".to_string(),
        quantity_kg: 200.0,
        contamination_level: ContaminationLevel::Clean,
        recommended_eol: EndOfLifeStrategy::IndustrialCompost,
        location_lat: 32.9483,
        location_lon: -96.7299,
        status: WasteStreamStatus::Registered,
        created_at: 1711100000_000000,
    };

    let stream_record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_waste_stream", stream)
        .await;
    let stream_hash = stream_record.action_address().clone();

    // Step 2: Create compost batch from waste stream
    let dummy_hash = ActionHash::from_raw_36(vec![0u8; 36]);
    let batch = CompostBatch {
        id: "CB-LOOP-001".to_string(),
        facility_hash: dummy_hash.clone(),
        method: CompostMethod::Windrow,
        inputs: vec![CompostInput {
            stream_hash: stream_hash.clone(),
            quantity_kg: 200.0,
            resource_type: "GreenWaste".to_string(),
        }],
        carbon_nitrogen_ratio: 26.0,
        status: CompostBatchStatus::Complete,
        started_at: 1711100000_000000,
        target_complete: 1714100000_000000,
        phase_history: vec![],
    };

    let batch_record: Record = conductor
        .call(
            &alice.zome("compost_control"),
            "create_compost_batch",
            batch,
        )
        .await;
    let batch_hash = batch_record.action_address().clone();

    // Step 3: Calculate carbon credit
    let carbon: CarbonAttribution = conductor
        .call(
            &alice.zome("compost_control"),
            "calculate_carbon_attribution",
            batch_hash.clone(),
        )
        .await;
    assert!(carbon.co2e_avoided_tonnes > 0.0);

    // Step 4: List finished compost on marketplace
    let listing = SecondaryMaterialListing {
        facility_hash: dummy_hash.clone(),
        batch_hash: Some(batch_hash),
        material_type: SecondaryMaterialType::Compost,
        quantity_kg: 150.0, // Compost is lighter than input
        quality_grade: MaterialQualityGrade::Standard,
        nutrient_info: Some(NutrientInfo {
            nitrogen_pct: 2.0,
            phosphorus_pct: 0.3,
            potassium_pct: 1.5,
        }),
        price_per_kg: 0.0, // Free for community members
        location_lat: 32.9500,
        location_lon: -96.7300,
        available_from: 1714100000_000000,
        status: ListingStatus::Available,
        seller_did: "did:example:composting_facility".to_string(),
    };

    let listing_record: Record = conductor
        .call(
            &alice.zome("circular_marketplace"),
            "list_secondary_material",
            listing,
        )
        .await;

    assert!(
        listing_record.entry().as_option().is_some(),
        "Full circular loop: waste → compost → marketplace complete"
    );
}

// ============================================================================
// E2E Demo: Complete Narrative
// ============================================================================

/// Comprehensive end-to-end demo exercising every function in the circular economy.
///
/// This is the definitive "it works" test — if this passes on a real conductor,
/// the entire circular economy infrastructure is operational.
///
/// Narrative: A community farm generates crop residue, it gets classified by AI,
/// routed to a composting facility, processed with sensor monitoring, recipe
/// optimized, carbon credited, listed on the marketplace, and matched back to
/// farms needing nutrients.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop) — full e2e demo"]
async fn test_e2e_complete_circular_economy_demo() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // ── Step 1: Register composting facility ────────────────────────────
    let facility = WasteFacility {
        id: "FAC-DEMO-001".to_string(),
        name: "Community Composting Hub".to_string(),
        facility_type: FacilityType::Composter,
        accepts: vec![WasteCategory::Organic],
        capacity_kg_per_day: 1000.0,
        current_load_kg: 100.0,
        location_lat: 32.9500,
        location_lon: -96.7300,
        operator_did: "did:example:operator".to_string(),
        status: FacilityStatus::Active,
    };
    let fac_record: Record = conductor
        .call(&alice.zome("waste_registry"), "register_facility", facility)
        .await;
    let _fac_hash = fac_record.action_address().clone();

    // ── Step 2: Register waste streams from 3 sources ───────────────────
    let sources = vec![
        ("WS-DEMO-001", "did:example:farm1", 200.0, "crop residue"),
        ("WS-DEMO-002", "did:example:farm2", 150.0, "food scraps"),
        ("WS-DEMO-003", "did:example:restaurant1", 80.0, "kitchen waste"),
    ];

    let mut stream_hashes = Vec::new();
    for (id, did, kg, subcat) in &sources {
        let stream = WasteStream {
            id: id.to_string(),
            source_did: did.to_string(),
            material_passport_hash: None,
            waste_category: WasteCategory::Organic,
            subcategory: subcat.to_string(),
            quantity_kg: *kg,
            contamination_level: ContaminationLevel::Clean,
            recommended_eol: EndOfLifeStrategy::IndustrialCompost,
            location_lat: 32.9483 + stream_hashes.len() as f64 * 0.005,
            location_lon: -96.7299 + stream_hashes.len() as f64 * 0.003,
            status: WasteStreamStatus::Registered,
            created_at: 1711100000_000000 + stream_hashes.len() as u64 * 1000,
        };
        let record: Record = conductor
            .call(&alice.zome("waste_registry"), "register_waste_stream", stream)
            .await;
        stream_hashes.push(record.action_address().clone());
    }
    assert_eq!(stream_hashes.len(), 3, "Should have 3 registered streams");

    // ── Step 3: AI classifies each stream ───────────────────────────────
    for (i, hash) in stream_hashes.iter().enumerate() {
        let classification = WasteClassification {
            stream_hash: hash.clone(),
            classifier_did: "did:example:symthaea_ai".to_string(),
            category: WasteCategory::Organic,
            subcategory: sources[i].3.to_string(),
            contamination_level: ContaminationLevel::Clean,
            confidence: 0.95,
            method: ClassificationMethod::VisionAI,
            classified_at: 1711200000_000000 + i as u64 * 1000,
        };
        let _: Record = conductor
            .call(&alice.zome("waste_registry"), "classify_waste_stream", classification)
            .await;
    }

    // ── Step 4: Route each stream to the facility ───────────────────────
    for hash in &stream_hashes {
        let route_input = RouteWasteStreamInput {
            stream_hash: hash.clone(),
        };
        let route: Option<Record> = conductor
            .call(&alice.zome("waste_registry"), "route_waste_stream", route_input)
            .await;
        assert!(route.is_some(), "Each stream should route to the composting facility");
    }

    // ── Step 5: Check analytics ─────────────────────────────────────────
    let diversion_input = DiversionRateInput {
        period_start_us: 1711000000_000000,
        period_end_us: 1712000000_000000,
    };
    let _diversion: DiversionRateResult = conductor
        .call(&alice.zome("waste_registry"), "calculate_diversion_rate", diversion_input)
        .await;

    // ── Step 6: Create compost batch with all 3 streams ─────────────────
    let dummy_fac = ActionHash::from_raw_36(vec![0u8; 36]);
    let batch = CompostBatch {
        id: "CB-DEMO-001".to_string(),
        facility_hash: dummy_fac.clone(),
        method: CompostMethod::Windrow,
        inputs: vec![
            CompostInput {
                stream_hash: stream_hashes[0].clone(),
                quantity_kg: 200.0,
                resource_type: "GreenWaste".to_string(),
            },
            CompostInput {
                stream_hash: stream_hashes[1].clone(),
                quantity_kg: 150.0,
                resource_type: "KitchenWaste".to_string(),
            },
            CompostInput {
                stream_hash: stream_hashes[2].clone(),
                quantity_kg: 80.0,
                resource_type: "KitchenWaste".to_string(),
            },
        ],
        carbon_nitrogen_ratio: 18.0, // Nitrogen-heavy — needs amendment
        status: CompostBatchStatus::Mixing,
        started_at: 1711300000_000000,
        target_complete: 1714300000_000000,
        phase_history: vec![],
    };
    let batch_record: Record = conductor
        .call(&alice.zome("compost_control"), "create_compost_batch", batch)
        .await;
    let batch_hash = batch_record.action_address().clone();

    // ── Step 7: Recipe optimizer suggests amendments ─────────────────────
    let recipe_input = OptimizeRecipeInput {
        batch_hash: batch_hash.clone(),
        target_cn_ratio: 27.5,
    };
    let recipe: RecipeOptimization = conductor
        .call(&alice.zome("compost_control"), "optimize_recipe", recipe_input)
        .await;

    assert!(
        !recipe.is_optimal,
        "C:N of 18 should not be optimal"
    );
    assert_eq!(
        recipe.diagnosis, "nitrogen_heavy",
        "C:N 18 < 25 should be nitrogen_heavy"
    );
    assert!(
        !recipe.amendments.is_empty(),
        "Should suggest carbon-rich amendments"
    );

    // ── Step 8: Simulate sensor readings ────────────────────────────────
    // Thermophilic phase reading
    let reading = CompostReading {
        batch_hash: batch_hash.clone(),
        sensor_id: "TEMP-001".to_string(),
        temperature_c: 58.0,
        moisture_pct: 52.0,
        oxygen_pct: 12.0,
        ph: 7.2,
        timestamp_us: 1711400000_000000,
    };
    let _: Record = conductor
        .call(&alice.zome("compost_control"), "record_compost_reading", reading)
        .await;

    // ── Step 9: Evaluate batch status ───────────────────────────────────
    let eval: BatchEvaluation = conductor
        .call(&alice.zome("compost_control"), "evaluate_batch_status", batch_hash.clone())
        .await;

    assert!(
        eval.temperature_status.contains("58"),
        "Should report actual temperature"
    );

    // ── Step 10: Nutrient estimation ────────────────────────────────────
    let nutrients: NutrientEstimate = conductor
        .call(&alice.zome("compost_control"), "get_batch_nutrient_estimate", batch_hash.clone())
        .await;

    assert!(
        (nutrients.total_kg - 430.0).abs() < 0.1,
        "Total should be 200+150+80=430kg, got {}",
        nutrients.total_kg
    );
    assert!(nutrients.nitrogen_pct > 0.0, "Should have positive nitrogen");

    // ── Step 11: Carbon attribution ─────────────────────────────────────
    let carbon: CarbonAttribution = conductor
        .call(&alice.zome("compost_control"), "calculate_carbon_attribution", batch_hash.clone())
        .await;

    assert!(
        carbon.co2e_avoided_tonnes > 0.0,
        "Should avoid positive CO2e"
    );
    assert!(
        (carbon.waste_diverted_kg - 430.0).abs() < 0.1,
        "Should divert 430kg"
    );

    // ── Step 12: List finished compost on marketplace ────────────────────
    let listing = SecondaryMaterialListing {
        facility_hash: dummy_fac.clone(),
        batch_hash: Some(batch_hash.clone()),
        material_type: SecondaryMaterialType::Compost,
        quantity_kg: 320.0, // Compost weighs less than inputs
        quality_grade: MaterialQualityGrade::Standard,
        nutrient_info: Some(NutrientInfo {
            nitrogen_pct: nutrients.nitrogen_pct,
            phosphorus_pct: nutrients.phosphorus_pct,
            potassium_pct: nutrients.potassium_pct,
        }),
        price_per_kg: 0.0, // Free for community
        location_lat: 32.9500,
        location_lon: -96.7300,
        available_from: 1714300000_000000,
        status: ListingStatus::Available,
        seller_did: "did:example:composting_facility".to_string(),
    };
    let listing_record: Record = conductor
        .call(&alice.zome("circular_marketplace"), "list_secondary_material", listing)
        .await;

    // ── Step 13: Demand matching ────────────────────────────────────────
    let demand = DemandMatchInput {
        material_type: SecondaryMaterialType::Compost,
        min_quantity_kg: 50.0,
        min_quality: MaterialQualityGrade::Industrial,
        buyer_lat: 32.9600,
        buyer_lon: -96.7200,
        max_distance_km: 50.0,
    };
    let matches: Vec<ListingMatch> = conductor
        .call(&alice.zome("circular_marketplace"), "find_matching_listings", demand)
        .await;

    assert!(
        !matches.is_empty(),
        "Should find the compost listing"
    );
    assert!(
        matches[0].match_score > 0.0,
        "Match score should be positive"
    );

    // ── Step 14: Place order ────────────────────────────────────────────
    let order = SecondaryMaterialOrder {
        listing_hash: listing_record.action_address().clone(),
        buyer: alice.agent_pubkey().clone(),
        quantity_kg: 100.0,
        delivery_lat: 32.9483,
        delivery_lon: -96.7299,
        delivery_route_hash: None,
        status: OrderStatus::Placed,
        placed_at: 1714400000_000000,
    };
    let order_record: Record = conductor
        .call(&alice.zome("circular_marketplace"), "place_order", order)
        .await;

    assert!(
        order_record.entry().as_option().is_some(),
        "E2E COMPLETE: waste → classify → route → compost → recipe → sensors → carbon → marketplace → order"
    );
}

// ── Additional types needed for e2e ─────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizeRecipeInput {
    pub batch_hash: ActionHash,
    pub target_cn_ratio: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecipeOptimization {
    pub current_cn_ratio: f64,
    pub target_cn_ratio: f64,
    pub is_optimal: bool,
    pub diagnosis: String,
    pub amendments: Vec<RecipeAmendment>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecipeAmendment {
    pub resource_type: String,
    pub quantity_kg: f64,
    pub material_cn_ratio: f64,
    pub resulting_cn_ratio: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OrderStatus {
    Placed,
    Confirmed,
    InTransit,
    Delivered,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SecondaryMaterialOrder {
    pub listing_hash: ActionHash,
    pub buyer: AgentPubKey,
    pub quantity_kg: f64,
    pub delivery_lat: f64,
    pub delivery_lon: f64,
    pub delivery_route_hash: Option<ActionHash>,
    pub status: OrderStatus,
    pub placed_at: u64,
}
