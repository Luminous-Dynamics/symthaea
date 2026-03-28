// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Commons — Food & Transport Sweettest
//!
//! Batch 2 of sweettest integration tests: food domain CRUD, transport domain
//! CRUD, and cross-domain food/transport bridge scenarios.
//!
//! Split from sweettest_integration.rs to reduce per-process conductor
//! memory pressure (~1-2 GB per conductor). Each [[test]] binary runs
//! as a separate OS process, so memory is fully reclaimed between batches.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons/tests
//! cargo test --release --test sweettest_food_transport -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — commons-bridge (needed for cross-domain tests)
// ============================================================================

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

// ============================================================================
// Mirror types — food domain
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SoilType {
    Clay,
    Sandy,
    Loam,
    Silt,
    Peat,
    Chalk,
    Mixed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PlotStatus {
    Active,
    Fallow,
    Preparing,
    Retired,
}

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
pub enum CropStatus {
    Planned,
    Planted,
    Growing,
    Ready,
    Harvested,
    Failed,
}

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
pub enum QualityGrade {
    Premium,
    Standard,
    Processing,
    Compost,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct YieldRecord {
    pub crop_hash: ActionHash,
    pub quantity_kg: f64,
    pub quality_grade: QualityGrade,
    pub harvested_at: u64,
    pub notes: Option<String>,
}

// ============================================================================
// Mirror types — transport domain
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VehicleType {
    Car, Van, Bike, Bus, Cargo, ElectricScooter,
    Helicopter, EVTOL, AirTaxi, Ferry, Boat, Train, Tram,
    Skateboard, Wheelchair, Segway, AutonomousVehicle, Drone,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VehicleStatus {
    Available,
    InUse,
    Maintenance,
    Retired,
}

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
pub enum TransportMode {
    Driving, Cycling, Walking, Transit, Mixed, Flying, Water, Rail, Micromobility, Autonomous,
}

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
pub enum TripMode {
    Driving, Cycling, Walking, Transit, Carpool, ElectricVehicle,
    Flying, Water, Rail, Micromobility, Autonomous,
}

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

// ============================================================================
// DNA setup helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("COMMONS_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Food Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_food_register_plot_and_plant_crop() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

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

    let crops: Vec<Record> = conductor
        .call(&alice.zome("food_production"), "get_plot_crops", plot_hash)
        .await;

    assert_eq!(crops.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_food_harvest_and_yield_record() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

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

    let yields: Vec<Record> = conductor
        .call(
            &alice.zome("food_production"),
            "get_crop_yields",
            crop_record.action_address().clone(),
        )
        .await;

    assert_eq!(yields.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Transport Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_register_vehicle_and_route() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

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

    let route = Route {
        id: "route-001".to_string(),
        name: "Downtown Loop".to_string(),
        waypoints: vec![
            Waypoint {
                lat: 32.948,
                lon: -96.730,
                label: Some("Start".to_string()),
            },
            Waypoint {
                lat: 32.955,
                lon: -96.725,
                label: Some("Mid".to_string()),
            },
            Waypoint {
                lat: 32.960,
                lon: -96.720,
                label: Some("End".to_string()),
            },
        ],
        distance_km: 5.2,
        estimated_minutes: 15,
        mode: TransportMode::Driving,
    };

    let route_record: Record = conductor
        .call(&alice.zome("transport_routes"), "create_route", route)
        .await;

    assert!(route_record.action().author() == alice.agent_pubkey());

    let routes: Vec<Record> = conductor
        .call(&alice.zome("transport_routes"), "get_all_routes", ())
        .await;

    assert!(!routes.is_empty());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_trip_logging_and_carbon_credits() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 10.0,
        mode: TripMode::Cycling,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0,
        logged_at: 1700000000,
    };

    let trip_record: Record = conductor
        .call(&alice.zome("transport_impact"), "log_trip", trip)
        .await;

    assert!(trip_record.action().author() == alice.agent_pubkey());

    let my_trips: Vec<Record> = conductor
        .call(&alice.zome("transport_impact"), "get_my_trips", ())
        .await;

    assert_eq!(my_trips.len(), 1);

    let credits: Vec<Record> = conductor
        .call(&alice.zome("transport_impact"), "get_my_carbon_credits", ())
        .await;

    assert_eq!(credits.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_emissions_calculator() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
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
        .call(
            &alice.zome("transport_impact"),
            "calculate_emissions",
            input,
        )
        .await;

    assert!(result.emissions_kg_co2 < result.baseline_emissions);
    assert!(result.savings_kg_co2 > 0.0);
    assert!((result.baseline_emissions - 4.2).abs() < 0.01);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_transport_community_impact_summary() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Cross-Domain: Food + Transport Integration via Bridge
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_event_via_bridge() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

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

    let food_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "food".to_string(),
        )
        .await;

    assert_eq!(food_events.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_transport_query_via_bridge() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
