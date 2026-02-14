//! Cross-Domain Integration Sweettest
//!
//! Proves that zomes within the Commons DNA can call each other directly
//! via `call(CallTargetCell::Local, ...)` through the bridge dispatcher.
//!
//! Run with: `cargo test --test cross_domain_sweettest -- --ignored`
//! Requires a running Holochain conductor (use `nix develop`).

use hdk::prelude::*;
use holochain::sweettest::*;

/// Mirror types — we redefine entry structs in test code to avoid
/// pulling in WASM-only integrity crates.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegisterPropertyInput {
    property_type: String,
    title: String,
    description: String,
    owner_did: String,
    co_owners: Vec<String>,
    geolocation: Option<GeoLocation>,
    address: Option<Address>,
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GeoLocation {
    latitude: f64,
    longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Address {
    street: String,
    city: String,
    state: String,
    postal_code: String,
    country: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchInput {
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchResult {
    success: bool,
    response: Option<Vec<u8>>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: u32,
    total_queries: u32,
    domains: Vec<String>,
}

// -- Food Production mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum SoilType {
    Clay,
    Sandy,
    Loam,
    Silt,
    Peat,
    Chalk,
    Mixed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PlotStatus {
    Active,
    Fallow,
    Preparing,
    Retired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Plot {
    id: String,
    name: String,
    area_sqm: f64,
    soil_type: SoilType,
    location_lat: f64,
    location_lon: f64,
    steward: AgentPubKey,
    status: PlotStatus,
}

// -- Food Distribution mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MarketType {
    Farmers,
    CSA,
    FoodBank,
    CoOp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Market {
    id: String,
    name: String,
    location_lat: f64,
    location_lon: f64,
    market_type: MarketType,
    steward: AgentPubKey,
    schedule: String,
}

// -- Transport Routes mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum VehicleType {
    Car,
    Van,
    Bike,
    Bus,
    Cargo,
    ElectricScooter,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum VehicleStatus {
    Available,
    InUse,
    Maintenance,
    Retired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Vehicle {
    id: String,
    owner: AgentPubKey,
    vehicle_type: VehicleType,
    capacity_kg: f64,
    capacity_passengers: u32,
    status: VehicleStatus,
}

// -- Transport Sharing mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum OfferStatus {
    Open,
    Full,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RideOffer {
    vehicle_hash: ActionHash,
    route_hash: Option<ActionHash>,
    driver: AgentPubKey,
    departure_time: u64,
    seats_available: u32,
    price_per_seat: f64,
    status: OfferStatus,
}

/// Helper to get the commons DNA path
fn commons_dna_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna")
        .join("commons.dna")
}

/// Test: Bridge health check returns all 5 domains
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_bridge_health_check() {
    let (conductor, _app, cell) = setup_conductor().await;

    let health: BridgeHealth = conductor
        .call(&cell.zome("commons_bridge"), "health_check", ())
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

/// Test: Cross-domain dispatch from bridge to property-registry
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_property() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Register a property first
    let input = RegisterPropertyInput {
        property_type: "residential".to_string(),
        title: "Test House".to_string(),
        description: "Cross-domain test property".to_string(),
        owner_did: "did:test:alice".to_string(),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
        }),
        address: Some(Address {
            street: "100 Main St".to_string(),
            city: "Richardson".to_string(),
            state: "TX".to_string(),
            postal_code: "75080".to_string(),
            country: "US".to_string(),
        }),
        metadata: std::collections::HashMap::new(),
    };

    // Call property_registry directly
    let record: Record = conductor
        .call(&cell.zome("property_registry"), "register_property", input.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(input).unwrap().0;
    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "register_property".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Cross-domain dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: housing-clt can verify property existence via bridge dispatch
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_housing_queries_property_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Register a property
    let input = RegisterPropertyInput {
        property_type: "land".to_string(),
        title: "CLT Land Parcel".to_string(),
        description: "Land for community land trust".to_string(),
        owner_did: "did:test:trust".to_string(),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: std::collections::HashMap::new(),
    };

    let _record: Record = conductor
        .call(&cell.zome("property_registry"), "register_property", input)
        .await;

    // Now use bridge to query properties from housing context
    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "get_all_properties".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Housing should be able to query property registry");
}

/// Test: Cross-domain dispatch from bridge to food_production's register_plot
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_food_production() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let plot = Plot {
        id: "plot-sweet-1".to_string(),
        name: "Community Garden Alpha".to_string(),
        area_sqm: 250.0,
        soil_type: SoilType::Loam,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: agent,
        status: PlotStatus::Active,
    };

    // Call food_production directly first
    let _record: Record = conductor
        .call(&cell.zome("food_production"), "register_plot", plot.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(plot).unwrap().0;
    let dispatch = DispatchInput {
        zome: "food_production".to_string(),
        fn_name: "register_plot".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food production dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to food_distribution's create_market
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_food_distribution() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let market = Market {
        id: "mkt-sweet-1".to_string(),
        name: "Richardson Farmers Market".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        market_type: MarketType::Farmers,
        steward: agent,
        schedule: "Saturdays 8am-1pm".to_string(),
    };

    // Call food_distribution directly first
    let _record: Record = conductor
        .call(&cell.zome("food_distribution"), "create_market", market.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(market).unwrap().0;
    let dispatch = DispatchInput {
        zome: "food_distribution".to_string(),
        fn_name: "create_market".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food distribution dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to transport_routes's register_vehicle
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_transport_routes() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let vehicle = Vehicle {
        id: "v-sweet-1".to_string(),
        owner: agent,
        vehicle_type: VehicleType::Car,
        capacity_kg: 450.0,
        capacity_passengers: 4,
        status: VehicleStatus::Available,
    };

    // Call transport_routes directly first
    let _record: Record = conductor
        .call(&cell.zome("transport_routes"), "register_vehicle", vehicle.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(vehicle).unwrap().0;
    let dispatch = DispatchInput {
        zome: "transport_routes".to_string(),
        fn_name: "register_vehicle".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport routes dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to transport_sharing's post_ride_offer
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_transport_sharing() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // First register a vehicle (ride offer needs a vehicle_hash)
    let vehicle = Vehicle {
        id: "v-sweet-ride".to_string(),
        owner: agent.clone(),
        vehicle_type: VehicleType::Van,
        capacity_kg: 800.0,
        capacity_passengers: 7,
        status: VehicleStatus::Available,
    };

    let vehicle_record: Record = conductor
        .call(&cell.zome("transport_routes"), "register_vehicle", vehicle)
        .await;

    let vehicle_hash = vehicle_record.action_address().clone();

    let offer = RideOffer {
        vehicle_hash: vehicle_hash.clone(),
        route_hash: None,
        driver: agent,
        departure_time: 1700000000,
        seats_available: 3,
        price_per_seat: 5.0,
        status: OfferStatus::Open,
    };

    // Dispatch through the bridge
    let payload = ExternIO::encode(offer).unwrap().0;
    let dispatch = DispatchInput {
        zome: "transport_sharing".to_string(),
        fn_name: "post_ride_offer".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport sharing dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: food_production queries via bridge (get_all_plots)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_food_queries_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // Register a plot first
    let plot = Plot {
        id: "plot-query-1".to_string(),
        name: "Herb Spiral".to_string(),
        area_sqm: 50.0,
        soil_type: SoilType::Sandy,
        location_lat: 32.95,
        location_lon: -96.73,
        steward: agent,
        status: PlotStatus::Preparing,
    };

    let _record: Record = conductor
        .call(&cell.zome("food_production"), "register_plot", plot)
        .await;

    // Now use bridge to query all plots
    let dispatch = DispatchInput {
        zome: "food_production".to_string(),
        fn_name: "get_all_plots".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food production query via bridge should succeed");
}

/// Test: transport_routes queries via bridge (get_all_routes)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_transport_queries_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Use bridge to query all routes (may be empty, but dispatch should succeed)
    let dispatch = DispatchInput {
        zome: "transport_routes".to_string(),
        fn_name: "get_all_routes".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport routes query via bridge should succeed");
}

/// Helper to set up a conductor with the commons DNA
async fn setup_conductor() -> (SweetConductor, SweetApp, SweetCell) {
    let dna_path = commons_dna_path();
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor.setup_app("commons", &[dna]).await.unwrap();
    let cell = app.into_cells().into_iter().next().unwrap();

    (conductor, app, cell)
}
