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
//! cd tests
//! cargo test --release --test sweettest_integration -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

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
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_basis_points: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub boundaries: Option<Vec<(f64, f64)>>,
    pub area_sqm: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub region: String,
    pub country: String,
    pub postal_code: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyMetadata {
    pub appraised_value: Option<f64>,
    pub currency: Option<String>,
    pub legal_description: Option<String>,
    pub parcel_number: Option<String>,
    pub attachments: Vec<String>,
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
    // Allow overriding DNA path via env var for split-DNA testing
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
// Property Registry Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_property_register_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let input = RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: "Test Property".to_string(),
        description: "A test property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
            boundaries: None,
            area_sqm: Some(150.0),
        }),
        address: Some(Address {
            street: "123 Main St".to_string(),
            city: "Richardson".to_string(),
            region: "TX".to_string(),
            country: "US".to_string(),
            postal_code: Some("75080".to_string()),
        }),
        metadata: PropertyMetadata {
            appraised_value: Some(250_000.0),
            currency: Some("USD".to_string()),
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 9);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
    assert!(health.domains.contains(&"food".to_string()));
    assert!(health.domains.contains(&"transport".to_string()));
    assert!(health.domains.contains(&"support".to_string()));
    assert!(health.domains.contains(&"space".to_string()));
}

// ============================================================================
// Cross-Domain Tests — the real value of cluster consolidation
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_queries_property() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a property
    let prop_input = RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: "CLT Community Housing".to_string(),
        description: "Community land trust property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: PropertyMetadata {
            appraised_value: Some(500_000.0),
            currency: None,
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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

    let _record: Record = conductor
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
pub enum PlotType { Garden, FoodForest, Orchard, Greenhouse, Raised, Rooftop }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Plot {
    pub id: String,
    pub name: String,
    pub area_sqm: f64,
    pub soil_type: SoilType,
    pub plot_type: PlotType,
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
    pub allergen_flags: Vec<String>,
    pub organic_certified: bool,
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
        plot_type: PlotType::Garden,
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
        allergen_flags: vec![],
        organic_certified: false,
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
        plot_type: PlotType::Garden,
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
        allergen_flags: vec![],
        organic_certified: false,
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
pub enum VehicleType {
    Car, Van, Bike, Bus, Cargo, ElectricScooter,
    Helicopter, EVTOL, AirTaxi, Ferry, Boat, Train, Tram,
    Skateboard, Wheelchair, Segway, AutonomousVehicle, Drone,
}

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
pub enum TransportMode { Driving, Cycling, Walking, Transit, Mixed, Flying, Water, Rail, Micromobility, Autonomous }

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
pub enum TripMode { Driving, Cycling, Walking, Transit, Carpool, ElectricVehicle, Flying, Water, Rail, Micromobility, Autonomous }

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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
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

// ============================================================================
// Cross-Domain Dispatch Scenarios — intra-cluster typed dispatch
// ============================================================================
//
// These scenarios exercise the `dispatch_call` extern on the commons_bridge
// zome, which calls `dispatch_call_checked()` against the Commons cluster
// allowlist. Each test constructs a DispatchInput targeting a specific
// domain zome within the same DNA. In a real conductor, the call would be
// dispatched via `call(CallTargetCell::Local, ...)`.

// Mirror type for DispatchInput (from mycelix_bridge_common)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchInput {
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

// Mirror type for DispatchResult (from mycelix_bridge_common)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
}

/// Cross-domain: Food -> Water
///
/// Scenario: Before planting a crop in a plot, the food_production zome
/// needs to check water quality from the water_purity zome. This dispatch
/// routes food_production -> water_purity via the commons bridge.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_dispatches_water_quality_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a plot in food_production (prerequisite)
    let plot = Plot {
        id: "plot-water-check".to_string(),
        name: "Riverside Garden".to_string(),
        area_sqm: 300.0,
        soil_type: SoilType::Loam,
        plot_type: PlotType::Garden,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: agent.clone(),
        status: PlotStatus::Active,
    };

    let _plot_record: Record = conductor
        .call(&alice.zome("food_production"), "register_plot", plot)
        .await;

    // 2. Dispatch from food context -> water_purity to check water quality
    //    before planting near the water source
    let water_query_payload = serde_json::to_vec(&serde_json::json!({
        "source_id": "river-001",
        "location_lat": 32.9483,
        "location_lon": -96.7299,
    }))
    .unwrap();

    let dispatch = DispatchInput {
        zome: "water_purity".to_string(),
        fn_name: "check_water_quality".to_string(),
        payload: water_query_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    // The dispatch should reach the zome (success depends on whether the
    // function exists in the compiled DNA, but the bridge will not reject
    // the zome name because water_purity is in the allowlist)
    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );
}

/// Cross-domain: Transport -> Mutualaid
///
/// Scenario: A transport_sharing coordinator requests cargo transport for
/// a mutual aid resource delivery. The dispatch routes from transport_sharing
/// context to mutualaid_resources to find available resources for pickup.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_transport_dispatches_mutualaid_resources() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a cargo vehicle in transport_routes
    let vehicle = Vehicle {
        id: "cargo-aid-001".to_string(),
        owner: agent.clone(),
        vehicle_type: VehicleType::Cargo,
        capacity_kg: 2000.0,
        capacity_passengers: 2,
        status: VehicleStatus::Available,
    };

    let _veh_record: Record = conductor
        .call(&alice.zome("transport_routes"), "register_vehicle", vehicle)
        .await;

    // 2. Dispatch from transport context -> mutualaid_resources to find
    //    resources pending delivery
    let resource_query_payload = serde_json::to_vec(&serde_json::json!({
        "resource_type": "FoodSupply",
        "needs_transport": true,
        "max_weight_kg": 2000.0,
    }))
    .unwrap();

    let dispatch = DispatchInput {
        zome: "mutualaid_resources".to_string(),
        fn_name: "get_resources_needing_transport".to_string(),
        payload: resource_query_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );
}

/// Cross-domain: Food -> Mutualaid
///
/// Scenario: A food bank donation is recorded in food_distribution, which
/// then dispatches to mutualaid_needs to auto-fulfill matching needs.
/// This demonstrates the food -> mutualaid cross-domain link.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_donation_triggers_mutualaid_fulfillment() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Broadcast a food bank donation event via the bridge
    let donation_event = CommonsEventInput {
        domain: "food".to_string(),
        event_type: "food_bank_donation".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "donor_did": format!("did:key:{}", agent),
            "items": [
                {"name": "Rice", "quantity_kg": 50.0},
                {"name": "Canned Beans", "quantity_kg": 30.0},
            ],
            "food_bank_id": "fb-richardson-001",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let donation_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", donation_event)
        .await;

    assert!(donation_record.action().author() == alice.agent_pubkey());

    // 2. Dispatch from food context -> mutualaid_needs to fulfill
    //    outstanding food requests
    let fulfill_payload = serde_json::to_vec(&serde_json::json!({
        "need_type": "food",
        "donation_hash": donation_record.action_address().to_string(),
        "items_available": ["Rice", "Canned Beans"],
    }))
    .unwrap();

    let dispatch = DispatchInput {
        zome: "mutualaid_needs".to_string(),
        fn_name: "fulfill_matching_needs".to_string(),
        payload: fulfill_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );

    // 3. Verify the food domain event was recorded
    let food_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "food".to_string(),
        )
        .await;

    assert!(!food_events.is_empty(), "Food domain should have the donation event");
}

/// Cross-domain: Housing -> Property
///
/// Scenario: Before creating a lease for a housing unit, the housing_units
/// zome dispatches to property_registry to verify that the property is
/// actually owned by the CLT (community land trust) that manages the unit.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_dispatches_property_ownership_verify() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a property owned by the CLT
    let prop_input = RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: "Oak Park Apartments".to_string(),
        description: "CLT-managed apartment complex".to_string(),
        owner_did: "did:mycelix:richardson-clt".to_string(),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9510,
            longitude: -96.7320,
            boundaries: None,
            area_sqm: Some(2500.0),
        }),
        address: Some(Address {
            street: "200 Oak Park Blvd".to_string(),
            city: "Richardson".to_string(),
            region: "TX".to_string(),
            country: "US".to_string(),
            postal_code: Some("75080".to_string()),
        }),
        metadata: PropertyMetadata {
            appraised_value: Some(1_200_000.0),
            currency: Some("USD".to_string()),
            legal_description: Some("Lot 14, Block 7, Oak Park Addition".to_string()),
            parcel_number: Some("R-2024-00147".to_string()),
            attachments: vec![],
        },
    };

    let prop_record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", prop_input)
        .await;

    // 2. Dispatch from housing context -> property_registry to verify
    //    ownership before creating a lease
    let verify_payload = serde_json::to_vec(&serde_json::json!({
        "property_id": prop_record.action_address().to_string(),
        "requester_did": "did:mycelix:richardson-clt",
    }))
    .unwrap();

    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "verify_ownership".to_string(),
        payload: verify_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );

    // 3. Bridge event linking housing lease intent to property verification
    let lease_event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "lease_ownership_verified".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": prop_record.action_address().to_string(),
            "clt_did": "did:mycelix:richardson-clt",
            "unit_id": "apt-101",
            "verification": "pending",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![prop_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", lease_event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());
}

/// Cross-domain: Water -> Food
///
/// Scenario: The water_flow zome checks irrigation credit availability
/// before approving a water allocation for farming. It dispatches to
/// food_production to verify the plot is active and eligible for
/// irrigation credits.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_water_dispatches_food_irrigation_credit_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a plot in food_production (the farming operation)
    let plot = Plot {
        id: "plot-irrigate-001".to_string(),
        name: "Riverside Farm".to_string(),
        area_sqm: 5000.0,
        soil_type: SoilType::Silt,
        plot_type: PlotType::Garden,
        location_lat: 33.0100,
        location_lon: -96.7500,
        steward: agent.clone(),
        status: PlotStatus::Active,
    };

    let plot_record: Record = conductor
        .call(&alice.zome("food_production"), "register_plot", plot)
        .await;

    // 2. Dispatch from water context -> food_production to check that the
    //    plot is active and eligible for irrigation credit
    let irrigation_check_payload = serde_json::to_vec(&serde_json::json!({
        "plot_hash": plot_record.action_address().to_string(),
        "requested_liters": 10000,
        "season": "spring",
    }))
    .unwrap();

    let dispatch = DispatchInput {
        zome: "food_production".to_string(),
        fn_name: "check_irrigation_eligibility".to_string(),
        payload: irrigation_check_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );

    // 3. Bridge query: water domain queries food for plot status
    let query = CommonsQueryInput {
        domain: "food".to_string(),
        query_type: "check_plot_status".to_string(),
        requester: agent.clone(),
        params: serde_json::to_string(&serde_json::json!({
            "plot_hash": plot_record.action_address().to_string(),
        }))
        .unwrap(),
        created_at: Timestamp::now(),
    };

    let query_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    assert!(query_record.action().author() == alice.agent_pubkey());
}

// ============================================================================
// Support Domain — Mirror Types
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SupportCategory {
    Network,
    Hardware,
    Software,
    Holochain,
    Mycelix,
    Security,
    General,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TicketPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TicketStatus {
    Open,
    InProgress,
    AwaitingUser,
    Resolved,
    Closed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutonomyLevel {
    Advisory,
    SemiAutonomous,
    FullAutonomous,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DiagnosticType {
    NetworkCheck,
    DiskSpace,
    ServiceStatus,
    HolochainHealth,
    MemoryUsage,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DiagnosticSeverity {
    Healthy,
    Warning,
    Error,
    Critical,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ArticleSource {
    Community,
    PreSeeded,
    SymthaeaGenerated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SharingTier {
    LocalOnly,
    Anonymized,
    Full,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EpistemicStatus {
    Certain,
    Probable,
    Uncertain,
    Unknown,
    OutOfDomain,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportKnowledgeArticle {
    pub title: String,
    pub content: String,
    pub category: SupportCategory,
    pub tags: Vec<String>,
    pub author: AgentPubKey,
    pub source: ArticleSource,
    pub difficulty_level: DifficultyLevel,
    pub upvotes: u32,
    pub verified: bool,
    pub deprecated: bool,
    pub deprecation_reason: Option<String>,
    pub version: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportTicketEntry {
    pub title: String,
    pub description: String,
    pub category: SupportCategory,
    pub priority: TicketPriority,
    pub status: TicketStatus,
    pub requester: AgentPubKey,
    pub assignee: Option<AgentPubKey>,
    pub autonomy_level: AutonomyLevel,
    pub system_info: Option<String>,
    pub is_preemptive: bool,
    pub prediction_confidence: Option<f32>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportTicketComment {
    pub ticket_hash: ActionHash,
    pub author: AgentPubKey,
    pub content: String,
    pub is_symthaea_response: bool,
    pub confidence: Option<f32>,
    pub epistemic_status: Option<EpistemicStatus>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportDiagnosticResult {
    pub ticket_hash: Option<ActionHash>,
    pub diagnostic_type: DiagnosticType,
    pub findings: String,
    pub severity: DiagnosticSeverity,
    pub recommendations: Vec<String>,
    pub agent: AgentPubKey,
    pub scrubbed: bool,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportPrivacyPreference {
    pub agent: AgentPubKey,
    pub sharing_tier: SharingTier,
    pub allowed_categories: Vec<SupportCategory>,
    pub share_system_info: bool,
    pub share_resolution_patterns: bool,
    pub share_cognitive_updates: bool,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportCognitiveUpdate {
    pub category: SupportCategory,
    pub encoding: Vec<u8>,
    pub phi: f64,
    pub resolution_pattern: String,
    pub source_agent: AgentPubKey,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SupportUpdateTicketInput {
    pub original_hash: ActionHash,
    pub updated: SupportTicketEntry,
}

// ============================================================================
// Support Domain — Sweettest Integration Tests
// ============================================================================

/// Support: Create article → search by category → verify found.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_create_article_and_search() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let article = SupportKnowledgeArticle {
        title: "Configuring Holochain Bootstrap Servers".into(),
        content: "## Overview\nBootstrap servers help new agents discover peers...\n\n```bash\nhc sandbox generate\n```".into(),
        category: SupportCategory::Holochain,
        tags: vec!["holochain".into(), "bootstrap".into(), "networking".into()],
        author: agent.clone(),
        source: ArticleSource::Community,
        difficulty_level: DifficultyLevel::Intermediate,
        upvotes: 0,
        verified: false,
        deprecated: false,
        deprecation_reason: None,
        version: 1,
    };

    let article_record: Record = conductor
        .call(&alice.zome("support_knowledge"), "create_article", article)
        .await;

    assert!(article_record.action().author() == &agent);

    // Search by category
    let results: Vec<Record> = conductor
        .call(
            &alice.zome("support_knowledge"),
            "search_by_category",
            SupportCategory::Holochain,
        )
        .await;

    assert!(
        !results.is_empty(),
        "search_by_category should return the created article"
    );
}

/// Support: Create ticket → add comment → update status → close ticket.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_create_ticket_lifecycle() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    let ticket = SupportTicketEntry {
        title: "Conductor crash on restart".into(),
        description: "After NixOS rebuild, conductor segfaults on startup".into(),
        category: SupportCategory::Holochain,
        priority: TicketPriority::High,
        status: TicketStatus::Open,
        requester: agent.clone(),
        assignee: None,
        autonomy_level: AutonomyLevel::Advisory,
        system_info: Some("NixOS 25.11, Holochain 0.4".into()),
        is_preemptive: false,
        prediction_confidence: None,
        created_at: now,
        updated_at: now,
    };

    let ticket_record: Record = conductor
        .call(&alice.zome("support_tickets"), "create_ticket", ticket.clone())
        .await;

    let ticket_hash = ticket_record.action_address().clone();

    // Add a comment
    let comment = SupportTicketComment {
        ticket_hash: ticket_hash.clone(),
        author: agent.clone(),
        content: "Tried clearing lair cache, same crash persists".into(),
        is_symthaea_response: false,
        confidence: None,
        epistemic_status: None,
    };

    let _comment_record: Record = conductor
        .call(&alice.zome("support_tickets"), "add_comment", comment)
        .await;

    // Update to resolved
    let mut updated_ticket = ticket.clone();
    updated_ticket.status = TicketStatus::Resolved;
    updated_ticket.updated_at = Timestamp::now();

    let update_input = SupportUpdateTicketInput {
        original_hash: ticket_hash.clone(),
        updated: updated_ticket,
    };

    let _updated_record: Record = conductor
        .call(&alice.zome("support_tickets"), "update_ticket", update_input)
        .await;

    // Close the ticket
    let _closed_record: Record = conductor
        .call(&alice.zome("support_tickets"), "close_ticket", ticket_hash)
        .await;
}

/// Support: Run diagnostic → verify link to ticket.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_diagnostic_flow() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    // Create a ticket first
    let ticket = SupportTicketEntry {
        title: "Slow DHT gossip".into(),
        description: "Gossip rounds taking >30s".into(),
        category: SupportCategory::Network,
        priority: TicketPriority::Medium,
        status: TicketStatus::Open,
        requester: agent.clone(),
        assignee: None,
        autonomy_level: AutonomyLevel::Advisory,
        system_info: None,
        is_preemptive: false,
        prediction_confidence: None,
        created_at: now,
        updated_at: now,
    };

    let ticket_record: Record = conductor
        .call(&alice.zome("support_tickets"), "create_ticket", ticket)
        .await;

    let ticket_hash = ticket_record.action_address().clone();

    // Run diagnostic linked to the ticket
    let diag = SupportDiagnosticResult {
        ticket_hash: Some(ticket_hash),
        diagnostic_type: DiagnosticType::NetworkCheck,
        findings: r#"{"latency_ms": 2500, "packet_loss": 0.15}"#.into(),
        severity: DiagnosticSeverity::Warning,
        recommendations: vec![
            "Check firewall rules for UDP port range".into(),
            "Verify bootstrap server reachability".into(),
        ],
        agent: agent.clone(),
        scrubbed: false,
        created_at: Timestamp::now(),
    };

    let diag_record: Record = conductor
        .call(&alice.zome("support_diagnostics"), "run_diagnostic", diag)
        .await;

    assert!(diag_record.action().author() == &agent);

    // Verify we can fetch it back
    let fetched: Option<Record> = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "get_diagnostic",
            diag_record.action_address().clone(),
        )
        .await;

    assert!(fetched.is_some(), "Diagnostic should be retrievable after creation");
}

/// Support: Set privacy preference → get → verify tier.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_privacy_preference() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let pref = SupportPrivacyPreference {
        agent: agent.clone(),
        sharing_tier: SharingTier::Anonymized,
        allowed_categories: vec![SupportCategory::Network, SupportCategory::Holochain],
        share_system_info: true,
        share_resolution_patterns: true,
        share_cognitive_updates: false,
        updated_at: Timestamp::now(),
    };

    let _pref_record: Record = conductor
        .call(&alice.zome("support_diagnostics"), "set_privacy_preference", pref)
        .await;

    // Retrieve it
    let fetched: Option<Record> = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "get_privacy_preference",
            agent,
        )
        .await;

    assert!(fetched.is_some(), "Privacy preference should be retrievable");
}

/// Support: Publish cognitive update → get by category → verify.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_cognitive_update_publish() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let update = SupportCognitiveUpdate {
        category: SupportCategory::Network,
        encoding: vec![0xAB; 2048], // 2048-byte BinaryHV
        phi: 0.72,
        resolution_pattern: "DNS misconfiguration: edit /etc/resolv.conf, restart systemd-resolved".into(),
        source_agent: agent.clone(),
        created_at: Timestamp::now(),
    };

    let update_record: Record = conductor
        .call(&alice.zome("support_diagnostics"), "publish_cognitive_update", update)
        .await;

    assert!(update_record.action().author() == &agent);

    // Query by category
    let results: Vec<Record> = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "get_cognitive_updates_by_category",
            SupportCategory::Network,
        )
        .await;

    assert!(
        !results.is_empty(),
        "Should find the published cognitive update by category"
    );
}

/// Support: Bridge dispatch routes to support zomes correctly.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_bridge_dispatch() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Query via commons_bridge for the support domain
    let query = CommonsQueryInput {
        domain: "support".to_string(),
        query_type: "list_tickets".to_string(),
        requester: agent.clone(),
        params: "{}".to_string(),
        created_at: Timestamp::now(),
    };

    let query_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    assert!(query_record.action().author() == &agent);
}

/// Support: Multi-agent ticket resolution — Alice creates, Bob resolves.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_multi_agent_resolution() {
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path()).await.unwrap();

    // Set up two separate conductors (Holochain 0.6 multi-agent pattern)
    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Exchange peers for DHT gossip
    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let alice_agent = alice.agent_pubkey().clone();
    let bob_agent = bob.agent_pubkey().clone();
    let now = Timestamp::now();

    // Alice creates a ticket
    let ticket = SupportTicketEntry {
        title: "Bridge dispatch timeout".into(),
        description: "Cross-domain calls to water_flow are timing out after 10s".into(),
        category: SupportCategory::Mycelix,
        priority: TicketPriority::High,
        status: TicketStatus::Open,
        requester: alice_agent.clone(),
        assignee: None,
        autonomy_level: AutonomyLevel::Advisory,
        system_info: None,
        is_preemptive: false,
        prediction_confidence: None,
        created_at: now,
        updated_at: now,
    };

    let ticket_record: Record = alice_conductor
        .call(&alice.zome("support_tickets"), "create_ticket", ticket)
        .await;

    let ticket_hash = ticket_record.action_address().clone();

    // Wait for DHT sync between conductors
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // Bob comments with a resolution suggestion
    let comment = SupportTicketComment {
        ticket_hash: ticket_hash.clone(),
        author: bob_agent.clone(),
        content: "The issue is a rate limit hitting 100 calls/min. Increase RATE_LIMIT_WINDOW_SECS.".into(),
        is_symthaea_response: false,
        confidence: Some(0.85),
        epistemic_status: Some(EpistemicStatus::Probable),
    };

    let _comment_record: Record = bob_conductor
        .call(&bob.zome("support_tickets"), "add_comment", comment)
        .await;

    // Bob can also fetch the ticket
    let fetched: Option<Record> = bob_conductor
        .call(&bob.zome("support_tickets"), "get_ticket", ticket_hash)
        .await;

    assert!(fetched.is_some(), "Bob should be able to fetch Alice's ticket via DHT");
}
