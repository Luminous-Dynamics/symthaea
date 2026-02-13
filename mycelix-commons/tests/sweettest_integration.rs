//! # Mycelix Commons — Sweettest Integration Tests
//!
//! Tests the unified Commons cluster DNA: property, housing, care,
//! mutualaid, water domain zomes + commons-bridge.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cargo test -p commons-tests --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- property-registry ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPropertyInput {
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    Residential,
    Commercial,
    Agricultural,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_percentage: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state: String,
    pub country: String,
    pub postal_code: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyMetadata {
    pub area_sqm: Option<f64>,
    pub year_built: Option<u32>,
    pub zoning: Option<String>,
    pub assessed_value: Option<u64>,
    pub currency: Option<String>,
}

// --- commons-bridge ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsQueryInput {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsEventInput {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ → mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Property Registry Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_property_register_and_get() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let input = RegisterPropertyInput {
        property_type: PropertyType::Residential,
        title: "Test Property".to_string(),
        description: "A test property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
        }),
        address: Some(Address {
            street: "123 Main St".to_string(),
            city: "Richardson".to_string(),
            state: "TX".to_string(),
            country: "US".to_string(),
            postal_code: "75080".to_string(),
        }),
        metadata: PropertyMetadata {
            area_sqm: Some(150.0),
            year_built: Some(2020),
            zoning: Some("R-1".to_string()),
            assessed_value: Some(250000),
            currency: Some("USD".to_string()),
        },
    };

    let record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());
}

// ============================================================================
// Commons Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_query_and_resolve() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Submit a query via the bridge
    let query = CommonsQueryInput {
        domain: "property".to_string(),
        query_type: "ownership_check".to_string(),
        requester: agent.clone(),
        params: r#"{"property_id":"test-1"}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    let query_hash = record.action_address().clone();

    // Resolve the query
    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"owner":"did:key:abc"}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("commons_bridge"), "resolve_query", resolve)
        .await;

    assert!(resolved.action().author() == alice.agent_pubkey());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_broadcast_event() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "unit_created".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"unit_id":"unit-1","name":"Apt 101"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    // Verify event appears in domain query
    let events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert_eq!(events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_check() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 7);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
    assert!(health.domains.contains(&"food".to_string()));
    assert!(health.domains.contains(&"transport".to_string()));
}

// ============================================================================
// Cross-Domain Tests — the real value of cluster consolidation
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_queries_property() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a property
    let prop_input = RegisterPropertyInput {
        property_type: PropertyType::Residential,
        title: "CLT Community Housing".to_string(),
        description: "Community land trust property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: PropertyMetadata {
            area_sqm: Some(500.0),
            year_built: Some(2024),
            zoning: None,
            assessed_value: None,
            currency: None,
        },
    };

    let prop_record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", prop_input)
        .await;

    // 2. Bridge event: housing references the property
    let event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "clt_property_linked".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": prop_record.action_address().to_string(),
            "clt_name": "Richardson Community Trust"
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![prop_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    // 3. Verify cross-domain event is retrievable by housing domain
    let housing_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert!(!housing_events.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_care_checks_mutualaid_resources() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Care zome queries mutualaid resources via bridge
    let query = CommonsQueryInput {
        domain: "mutualaid".to_string(),
        query_type: "available_resources".to_string(),
        requester: agent.clone(),
        params: r#"{"resource_type":"MeetingRoom","location":"Remote"}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    // The query is stored — in production the bridge would dispatch to
    // mutualaid_resources zome. For now, verify it's recorded.
    let my_queries: Vec<Record> = conductor
        .call(&alice.zome("commons_bridge"), "get_my_queries", ())
        .await;

    assert_eq!(my_queries.len(), 1);
}

// ============================================================================
// Food Domain Tests
// ============================================================================

// Mirror types — food domain

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SoilType { Clay, Sandy, Loam, Silt, Peat, Chalk, Mixed }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PlotStatus { Active, Fallow, Preparing, Retired }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Plot {
    pub id: String,
    pub name: String,
    pub area_sqm: f64,
    pub soil_type: SoilType,
    pub location_lat: f64,
    pub location_lon: f64,
    pub steward: AgentPubKey,
    pub status: PlotStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CropStatus { Planned, Planted, Growing, Ready, Harvested, Failed }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Crop {
    pub plot_hash: ActionHash,
    pub name: String,
    pub variety: String,
    pub planted_at: u64,
    pub expected_harvest: u64,
    pub status: CropStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum QualityGrade { Premium, Standard, Processing, Compost }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct YieldRecord {
    pub crop_hash: ActionHash,
    pub quantity_kg: f64,
    pub quality_grade: QualityGrade,
    pub harvested_at: u64,
    pub notes: Option<String>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_food_register_plot_and_plant_crop() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Register a plot
    let plot = Plot {
        id: "plot-001".to_string(),
        name: "Community Garden A".to_string(),
        area_sqm: 200.0,
        soil_type: SoilType::Loam,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: agent.clone(),
        status: PlotStatus::Active,
    };

    let plot_record: Record = conductor
        .call(&alice.zome("food_production"), "register_plot", plot)
        .await;

    assert!(plot_record.action().author() == alice.agent_pubkey());
    let plot_hash = plot_record.action_address().clone();

    // Plant a crop in the plot
    let crop = Crop {
        plot_hash: plot_hash.clone(),
        name: "Tomato".to_string(),
        variety: "Roma".to_string(),
        planted_at: 1700000000,
        expected_harvest: 1707000000,
        status: CropStatus::Planted,
    };

    let crop_record: Record = conductor
        .call(&alice.zome("food_production"), "plant_crop", crop)
        .await;

    assert!(crop_record.action().author() == alice.agent_pubkey());

    // Verify the crop is linked to the plot
    let crops: Vec<Record> = conductor
        .call(&alice.zome("food_production"), "get_plot_crops", plot_hash)
        .await;

    assert_eq!(crops.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_food_harvest_and_yield_record() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Register plot → plant crop → record harvest
    let plot = Plot {
        id: "plot-002".to_string(),
        name: "Herb Garden".to_string(),
        area_sqm: 50.0,
        soil_type: SoilType::Sandy,
        location_lat: 33.0,
        location_lon: -96.8,
        steward: agent.clone(),
        status: PlotStatus::Active,
    };

    let plot_record: Record = conductor
        .call(&alice.zome("food_production"), "register_plot", plot)
        .await;

    let crop = Crop {
        plot_hash: plot_record.action_address().clone(),
        name: "Basil".to_string(),
        variety: "Genovese".to_string(),
        planted_at: 1700000000,
        expected_harvest: 1703000000,
        status: CropStatus::Growing,
    };

    let crop_record: Record = conductor
        .call(&alice.zome("food_production"), "plant_crop", crop)
        .await;

    let yield_rec = YieldRecord {
        crop_hash: crop_record.action_address().clone(),
        quantity_kg: 5.5,
        quality_grade: QualityGrade::Premium,
        harvested_at: 1703000000,
        notes: Some("Excellent first harvest".to_string()),
    };

    let yield_record: Record = conductor
        .call(&alice.zome("food_production"), "record_harvest", yield_rec)
        .await;

    assert!(yield_record.action().author() == alice.agent_pubkey());

    // Get yields for the crop
    let yields: Vec<Record> = conductor
        .call(
            &alice.zome("food_production"),
            "get_crop_yields",
            crop_record.action_address().clone(),
        )
        .await;

    assert_eq!(yields.len(), 1);
}

// ============================================================================
// Transport Domain Tests
// ============================================================================

// Mirror types — transport domain

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VehicleType { Car, Van, Bike, Bus, Cargo, ElectricScooter }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VehicleStatus { Available, InUse, Maintenance, Retired }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vehicle {
    pub id: String,
    pub owner: AgentPubKey,
    pub vehicle_type: VehicleType,
    pub capacity_kg: f64,
    pub capacity_passengers: u32,
    pub status: VehicleStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TransportMode { Driving, Cycling, Walking, Transit, Mixed }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Waypoint {
    pub lat: f64,
    pub lon: f64,
    pub label: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Route {
    pub id: String,
    pub name: String,
    pub waypoints: Vec<Waypoint>,
    pub distance_km: f64,
    pub estimated_minutes: u32,
    pub mode: TransportMode,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TripMode { Driving, Cycling, Walking, Transit, Carpool, ElectricVehicle }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TripLog {
    pub vehicle_hash: Option<ActionHash>,
    pub route_hash: Option<ActionHash>,
    pub distance_km: f64,
    pub mode: TripMode,
    pub passengers: u32,
    pub cargo_kg: f64,
    pub emissions_kg_co2: f64,
    pub logged_at: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmissionsCalcInput {
    pub distance_km: f64,
    pub mode: TripMode,
    pub passengers: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmissionsCalcResult {
    pub emissions_kg_co2: f64,
    pub baseline_emissions: f64,
    pub savings_kg_co2: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommunityImpactSummary {
    pub total_trips: u32,
    pub total_distance_km: f64,
    pub total_emissions_kg_co2: f64,
    pub total_credits_earned: f64,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_register_vehicle_and_route() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Register a vehicle
    let vehicle = Vehicle {
        id: "veh-001".to_string(),
        owner: agent.clone(),
        vehicle_type: VehicleType::Car,
        capacity_kg: 500.0,
        capacity_passengers: 4,
        status: VehicleStatus::Available,
    };

    let veh_record: Record = conductor
        .call(&alice.zome("transport_routes"), "register_vehicle", vehicle)
        .await;

    assert!(veh_record.action().author() == alice.agent_pubkey());

    // Create a route
    let route = Route {
        id: "route-001".to_string(),
        name: "Downtown Loop".to_string(),
        waypoints: vec![
            Waypoint { lat: 32.948, lon: -96.730, label: Some("Start".to_string()) },
            Waypoint { lat: 32.955, lon: -96.725, label: Some("Mid".to_string()) },
            Waypoint { lat: 32.960, lon: -96.720, label: Some("End".to_string()) },
        ],
        distance_km: 5.2,
        estimated_minutes: 15,
        mode: TransportMode::Driving,
    };

    let route_record: Record = conductor
        .call(&alice.zome("transport_routes"), "create_route", route)
        .await;

    assert!(route_record.action().author() == alice.agent_pubkey());

    // Verify routes are retrievable
    let routes: Vec<Record> = conductor
        .call(&alice.zome("transport_routes"), "get_all_routes", ())
        .await;

    assert!(!routes.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_trip_logging_and_carbon_credits() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    // Log a cycling trip (should earn carbon credits)
    let trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 10.0,
        mode: TripMode::Cycling,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0, // auto-calculated by zome
        logged_at: 1700000000,
    };

    let trip_record: Record = conductor
        .call(&alice.zome("transport_impact"), "log_trip", trip)
        .await;

    assert!(trip_record.action().author() == alice.agent_pubkey());

    // Verify trip is linked to agent
    let my_trips: Vec<Record> = conductor
        .call(&alice.zome("transport_impact"), "get_my_trips", ())
        .await;

    assert_eq!(my_trips.len(), 1);

    // Cycling should earn carbon credits (baseline 10km * 0.21 = 2.1 kg CO2 saved)
    let credits: Vec<Record> = conductor
        .call(&alice.zome("transport_impact"), "get_my_carbon_credits", ())
        .await;

    assert_eq!(credits.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_emissions_calculator() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let input = EmissionsCalcInput {
        distance_km: 20.0,
        mode: TripMode::Carpool,
        passengers: 4,
    };

    let result: EmissionsCalcResult = conductor
        .call(&alice.zome("transport_impact"), "calculate_emissions", input)
        .await;

    // Carpool 20km with 4 passengers: base = 20 * 0.07 = 1.4, / 4 = 0.35
    assert!(result.emissions_kg_co2 < result.baseline_emissions);
    assert!(result.savings_kg_co2 > 0.0);
    // Baseline: 20km * 0.21 = 4.2
    assert!((result.baseline_emissions - 4.2).abs() < 0.01);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_community_impact_summary() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    // Log two trips
    let trip1 = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 5.0,
        mode: TripMode::Walking,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0,
        logged_at: 1700000000,
    };

    let trip2 = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 15.0,
        mode: TripMode::Transit,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0,
        logged_at: 1700001000,
    };

    let _: Record = conductor
        .call(&alice.zome("transport_impact"), "log_trip", trip1)
        .await;
    let _: Record = conductor
        .call(&alice.zome("transport_impact"), "log_trip", trip2)
        .await;

    let summary: CommunityImpactSummary = conductor
        .call(
            &alice.zome("transport_impact"),
            "get_community_impact_summary",
            (),
        )
        .await;

    assert_eq!(summary.total_trips, 2);
    assert!((summary.total_distance_km - 20.0).abs() < 0.01);
    assert!(summary.total_credits_earned > 0.0);
}

// ============================================================================
// Cross-Domain: Food + Transport Integration
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_event_via_bridge() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Broadcast a food domain event through the bridge
    let event = CommonsEventInput {
        domain: "food".to_string(),
        event_type: "harvest_recorded".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"crop":"Tomato","quantity_kg":50.0}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    // Verify food events are retrievable
    let food_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "food".to_string(),
        )
        .await;

    assert_eq!(food_events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_transport_query_via_bridge() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[&dna_file])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Query transport domain through bridge
    let query = CommonsQueryInput {
        domain: "transport".to_string(),
        query_type: "available_rides".to_string(),
        requester: agent.clone(),
        params: r#"{"origin_lat":32.9,"origin_lon":-96.7}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    let my_queries: Vec<Record> = conductor
        .call(&alice.zome("commons_bridge"), "get_my_queries", ())
        .await;

    assert_eq!(my_queries.len(), 1);
}
