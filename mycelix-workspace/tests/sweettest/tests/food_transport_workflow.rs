// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Food + Transport Domain Sweettest Integration Tests
//!
//! Covers all 7 zomes across the food sovereignty and community transport
//! domains in the Mycelix Commons cluster:
//!
//! Food (4 zomes):
//!   - `food_production`   — plots, crops, yield records, season plans
//!   - `food_distribution` — markets, produce listings, orders
//!   - `food_preservation` — preservation batches, methods, storage units
//!   - `food_knowledge`    — seed catalog, traditional practices, recipes
//!
//! Transport (3 zomes):
//!   - `transport_routes`  — vehicles, community routes, stops
//!   - `transport_sharing` — ride offers, ride requests, ride matching
//!   - `transport_impact`  — trip logs, carbon credit tracking
//!
//! ## Prerequisites
//!
//! ```bash
//! cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-commons/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test food_transport_workflow -- --ignored --test-threads=2
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — food_production
// ============================================================================

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CropStatus {
    Planned,
    Planted,
    Growing,
    Ready,
    Harvested,
    Failed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Crop {
    plot_hash: ActionHash,
    name: String,
    variety: String,
    planted_at: u64,
    expected_harvest: u64,
    status: CropStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum QualityGrade {
    Premium,
    Standard,
    Processing,
    Compost,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct YieldRecord {
    crop_hash: ActionHash,
    quantity_kg: f64,
    quality_grade: QualityGrade,
    harvested_at: u64,
    notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SeasonPlan {
    plot_hash: ActionHash,
    year: u32,
    season: String,
    planned_crops: Vec<String>,
    rotation_notes: Option<String>,
}

// ============================================================================
// Mirror types — food_distribution
// ============================================================================

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Listing {
    market_hash: ActionHash,
    producer: AgentPubKey,
    product_name: String,
    quantity_kg: f64,
    price_per_kg: f64,
    available_from: u64,
    status: ListingStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum OrderStatus {
    Pending,
    Confirmed,
    Fulfilled,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Order {
    listing_hash: ActionHash,
    buyer: AgentPubKey,
    quantity_kg: f64,
    status: OrderStatus,
}

// ============================================================================
// Mirror types — food_preservation
// (matches food_preservation_integrity entry types exactly)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum BatchStatus {
    InProgress,
    Completed,
    Failed,
    Consumed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PreservationBatch {
    id: String,
    source_crop_hash: Option<ActionHash>,
    method: String,
    quantity_kg: f64,
    started_at: u64,
    expected_ready: u64,
    status: BatchStatus,
    notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PreservationMethod {
    name: String,
    description: String,
    shelf_life_days: u32,
    equipment_needed: Vec<String>,
    skill_level: SkillLevel,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum StorageType {
    RootCellar,
    Cellar,
    Freezer,
    Dehydrator,
    Fermenter,
    Pantry,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StorageUnit {
    id: String,
    name: String,
    capacity_kg: f64,
    storage_type: StorageType,
    steward: AgentPubKey,
}

// ============================================================================
// Mirror types — food_knowledge
// (matches food_knowledge_integrity entry types exactly)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SeedVariety {
    name: String,
    species: String,
    origin: Option<String>,
    days_to_maturity: u32,
    companion_plants: Vec<String>,
    avoid_plants: Vec<String>,
    seed_saving_notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PracticeCategory {
    Planting,
    Harvest,
    Soil,
    Pest,
    Water,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TraditionalPractice {
    name: String,
    description: String,
    region: Option<String>,
    season: Option<String>,
    category: PracticeCategory,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: String,
    servings: u32,
    prep_time_min: u32,
    tags: Vec<String>,
    source_attribution: Option<String>,
}

// ============================================================================
// Mirror types — transport_routes
// (matches transport_routes_integrity entry types exactly)
// ============================================================================

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum TransportMode {
    Driving,
    Cycling,
    Walking,
    Transit,
    Mixed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Waypoint {
    lat: f64,
    lon: f64,
    label: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Route {
    id: String,
    name: String,
    waypoints: Vec<Waypoint>,
    distance_km: f64,
    estimated_minutes: u32,
    mode: TransportMode,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum StopType {
    Pickup,
    Dropoff,
    Transfer,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Stop {
    route_hash: ActionHash,
    name: String,
    location_lat: f64,
    location_lon: f64,
    scheduled_time: Option<u64>,
    stop_type: StopType,
}

// ============================================================================
// Mirror types — transport_sharing
// (matches transport_sharing_integrity entry types exactly)
// ============================================================================

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum RequestStatus {
    Open,
    Matched,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RideRequest {
    requester: AgentPubKey,
    origin_lat: f64,
    origin_lon: f64,
    destination_lat: f64,
    destination_lon: f64,
    requested_time: u64,
    passengers: u32,
    status: RequestStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MatchStatus {
    Pending,
    Confirmed,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RideMatch {
    offer_hash: ActionHash,
    request_hash: ActionHash,
    confirmed_at: Option<u64>,
    status: MatchStatus,
}

// ============================================================================
// Mirror types — transport_impact
// (matches transport_impact_integrity entry types exactly)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum TripMode {
    Driving,
    Cycling,
    Walking,
    Transit,
    Carpool,
    ElectricVehicle,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TripLog {
    vehicle_hash: Option<ActionHash>,
    route_hash: Option<ActionHash>,
    distance_km: f64,
    mode: TripMode,
    passengers: u32,
    cargo_kg: f64,
    emissions_kg_co2: f64,
    logged_at: u64,
}

// ============================================================================
// Helper: single-agent commons setup
// ============================================================================

async fn setup_commons() -> TestAgent {
    let agents = setup_test_agents(&DnaPaths::commons(), "commons_app", 1).await;
    agents.into_iter().next().unwrap()
}

// ============================================================================
// food_production tests
// ============================================================================

/// Test: Register a growing plot, plant a crop, and record a harvest.
///
/// Exercises the full food_production lifecycle:
///   `register_plot` → `plant_crop` → `record_harvest` → `get_plot_crops` → `get_crop_yields`
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_production_plot_crop_harvest_lifecycle() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // 1. Register a community growing plot
    let plot = Plot {
        id: "plot-riverside-001".to_string(),
        name: "Riverside Community Garden".to_string(),
        area_sqm: 450.0,
        soil_type: SoilType::Loam,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: agent_key.clone(),
        status: PlotStatus::Active,
    };

    let plot_record: Record = agent
        .call_zome_fn("food_production", "register_plot", plot)
        .await;

    assert!(
        !plot_record.action_hashed().hash.as_ref().is_empty(),
        "Plot should be registered on DHT"
    );
    let plot_hash = plot_record.action_hashed().hash.clone();

    // 2. Verify the plot appears in the global listing
    let all_plots: Vec<Record> = agent
        .call_zome_fn("food_production", "get_all_plots", ())
        .await;

    assert!(!all_plots.is_empty(), "get_all_plots should return at least 1 plot");

    // 3. Plant a Cherokee Purple tomato crop on this plot
    let crop = Crop {
        plot_hash: plot_hash.clone(),
        name: "Tomato".to_string(),
        variety: "Cherokee Purple".to_string(),
        planted_at: 1_740_000_000,
        expected_harvest: 1_746_000_000,
        status: CropStatus::Planted,
    };

    let crop_record: Record = agent
        .call_zome_fn("food_production", "plant_crop", crop)
        .await;

    assert!(
        !crop_record.action_hashed().hash.as_ref().is_empty(),
        "Crop should be created"
    );
    let crop_hash = crop_record.action_hashed().hash.clone();

    // 4. Verify the crop is linked to the plot
    let plot_crops: Vec<Record> = agent
        .call_zome_fn("food_production", "get_plot_crops", plot_hash)
        .await;

    assert_eq!(plot_crops.len(), 1, "Plot should have exactly 1 crop");

    // 5. Record a premium harvest
    let harvest = YieldRecord {
        crop_hash: crop_hash.clone(),
        quantity_kg: 38.5,
        quality_grade: QualityGrade::Premium,
        harvested_at: 1_746_200_000,
        notes: Some("Excellent season — minimal pests, good rainfall".to_string()),
    };

    let harvest_record: Record = agent
        .call_zome_fn("food_production", "record_harvest", harvest)
        .await;

    assert!(
        !harvest_record.action_hashed().hash.as_ref().is_empty(),
        "Harvest record should be created"
    );

    // 6. Verify the yield record is linked to the crop
    let crop_yields: Vec<Record> = agent
        .call_zome_fn("food_production", "get_crop_yields", crop_hash)
        .await;

    assert_eq!(crop_yields.len(), 1, "Crop should have exactly 1 yield record");
}

/// Test: Create a season plan for a plot and retrieve it.
///
/// Verifies `create_season_plan` and `get_season_plans`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_production_season_plan_crud() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // Register a plot to plan for
    let plot = Plot {
        id: "plot-south-field-001".to_string(),
        name: "South Field".to_string(),
        area_sqm: 800.0,
        soil_type: SoilType::Clay,
        location_lat: 32.9100,
        location_lon: -96.6900,
        steward: agent_key,
        status: PlotStatus::Preparing,
    };

    let plot_record: Record = agent
        .call_zome_fn("food_production", "register_plot", plot)
        .await;

    let plot_hash = plot_record.action_hashed().hash.clone();

    // Create a spring season plan with crop rotation notes
    let plan = SeasonPlan {
        plot_hash: plot_hash.clone(),
        year: 2026,
        season: "Spring".to_string(),
        planned_crops: vec![
            "Tomato".to_string(),
            "Basil".to_string(),
            "Pepper".to_string(),
            "Squash".to_string(),
        ],
        rotation_notes: Some(
            "Follow last year's legumes with heavy feeders — tomatoes and squash".to_string(),
        ),
    };

    let plan_record: Record = agent
        .call_zome_fn("food_production", "create_season_plan", plan)
        .await;

    assert!(
        !plan_record.action_hashed().hash.as_ref().is_empty(),
        "Season plan should be created"
    );

    // Retrieve season plans linked to this plot
    let plans: Vec<Record> = agent
        .call_zome_fn("food_production", "get_season_plans", plot_hash)
        .await;

    assert_eq!(plans.len(), 1, "Plot should have exactly 1 season plan");
}

// ============================================================================
// food_distribution tests
// ============================================================================

/// Test: Create a farmers market, list produce, and place an order.
///
/// Exercises `create_market` → `list_product` → `place_order` → `get_my_orders`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_distribution_market_listing_order_workflow() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // 1. Create a community farmers market
    let market = Market {
        id: "mkt-richardson-001".to_string(),
        name: "Richardson Farmers Market".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        market_type: MarketType::Farmers,
        steward: agent_key.clone(),
        schedule: "Saturdays 8am–1pm".to_string(),
    };

    let market_record: Record = agent
        .call_zome_fn("food_distribution", "create_market", market)
        .await;

    assert!(
        !market_record.action_hashed().hash.as_ref().is_empty(),
        "Market should be created"
    );
    let market_hash = market_record.action_hashed().hash.clone();

    // 2. Verify market appears in the global market listing
    let all_markets: Vec<Record> = agent
        .call_zome_fn("food_distribution", "get_all_markets", ())
        .await;

    assert!(!all_markets.is_empty(), "Should have at least 1 market");

    // 3. List heirloom tomatoes at this market
    let listing = Listing {
        market_hash: market_hash.clone(),
        producer: agent_key.clone(),
        product_name: "Heirloom Tomatoes".to_string(),
        quantity_kg: 25.0,
        price_per_kg: 6.50,
        available_from: 1_740_028_800,
        status: ListingStatus::Available,
    };

    let listing_record: Record = agent
        .call_zome_fn("food_distribution", "list_product", listing)
        .await;

    assert!(
        !listing_record.action_hashed().hash.as_ref().is_empty(),
        "Listing should be created"
    );
    let listing_hash = listing_record.action_hashed().hash.clone();

    // 4. Verify listing is linked to the market
    let market_listings: Vec<Record> = agent
        .call_zome_fn("food_distribution", "get_market_listings", market_hash)
        .await;

    assert_eq!(market_listings.len(), 1, "Market should have 1 listing");

    // 5. Place an order for 5 kg of tomatoes
    let order = Order {
        listing_hash,
        buyer: agent_key.clone(),
        quantity_kg: 5.0,
        status: OrderStatus::Pending,
    };

    let order_record: Record = agent
        .call_zome_fn("food_distribution", "place_order", order)
        .await;

    assert!(
        !order_record.action_hashed().hash.as_ref().is_empty(),
        "Order should be placed"
    );

    // 6. Retrieve all orders for this buyer
    let my_orders: Vec<Record> = agent
        .call_zome_fn("food_distribution", "get_my_orders", ())
        .await;

    assert!(!my_orders.is_empty(), "Buyer should have at least 1 order");
}

/// Test: List produce as a producer and retrieve producer listings.
///
/// Verifies `list_product` and `get_producer_listings`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_distribution_producer_listings() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // Create a food bank market for donations
    let market = Market {
        id: "mkt-foodbank-001".to_string(),
        name: "Community Food Bank Distribution".to_string(),
        location_lat: 32.9600,
        location_lon: -96.7100,
        market_type: MarketType::FoodBank,
        steward: agent_key.clone(),
        schedule: "Wednesdays 10am–2pm".to_string(),
    };

    let market_record: Record = agent
        .call_zome_fn("food_distribution", "create_market", market)
        .await;

    let market_hash = market_record.action_hashed().hash.clone();

    // List two surplus produce items as donations (price = 0)
    for (name, qty) in [("Donated Zucchini", 40.0f64), ("Surplus Kale", 15.0f64)] {
        let listing = Listing {
            market_hash: market_hash.clone(),
            producer: agent_key.clone(),
            product_name: name.to_string(),
            quantity_kg: qty,
            price_per_kg: 0.0,
            available_from: 1_740_000_000,
            status: ListingStatus::Available,
        };
        let _: Record = agent
            .call_zome_fn("food_distribution", "list_product", listing)
            .await;
    }

    // Retrieve listings attributed to this producer
    let producer_listings: Vec<Record> = agent
        .call_zome_fn("food_distribution", "get_producer_listings", ())
        .await;

    assert_eq!(
        producer_listings.len(),
        2,
        "Producer should have exactly 2 listings"
    );
}

// ============================================================================
// food_preservation tests
// ============================================================================

/// Test: Start a preservation batch, complete it, and verify retrieval.
///
/// Exercises `start_batch`, `get_batch`, `complete_batch`, and `get_agent_batches`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_preservation_batch_lifecycle() {
    let agent = setup_commons().await;

    // 1. Start a lacto-fermentation batch
    let batch = PreservationBatch {
        id: "batch-sauerkraut-001".to_string(),
        source_crop_hash: None,
        method: "Lacto-fermentation".to_string(),
        quantity_kg: 12.5,
        started_at: 1_740_000_000,
        expected_ready: 1_740_604_800, // +7 days
        status: BatchStatus::InProgress,
        notes: Some("2% salt by weight; submerge cabbage fully below brine".to_string()),
    };

    let batch_record: Record = agent
        .call_zome_fn("food_preservation", "start_batch", batch)
        .await;

    assert!(
        !batch_record.action_hashed().hash.as_ref().is_empty(),
        "Batch should be started"
    );
    let batch_hash = batch_record.action_hashed().hash.clone();

    // 2. Retrieve the batch directly by hash
    let retrieved: Option<Record> = agent
        .call_zome_fn("food_preservation", "get_batch", batch_hash.clone())
        .await;

    assert!(retrieved.is_some(), "Batch should be retrievable by hash");

    // 3. Retrieve all batches started by this agent
    let my_batches: Vec<Record> = agent
        .call_zome_fn("food_preservation", "get_agent_batches", ())
        .await;

    assert!(!my_batches.is_empty(), "Agent should have at least 1 batch");

    // 4. Mark the batch as complete
    let completed_record: Record = agent
        .call_zome_fn("food_preservation", "complete_batch", batch_hash)
        .await;

    assert!(
        !completed_record.action_hashed().hash.as_ref().is_empty(),
        "Batch completion should return an updated record"
    );
}

/// Test: Register preservation methods and retrieve all methods.
///
/// Verifies `register_method` and `get_all_methods`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_preservation_method_registry() {
    let agent = setup_commons().await;

    // Register two community-approved preservation methods
    let methods = vec![
        PreservationMethod {
            name: "Water Bath Canning".to_string(),
            description: "Submerge sealed jars in boiling water for the specified processing \
                time. Safe for high-acid foods (pH < 4.6): tomatoes, fruits, pickles."
                .to_string(),
            shelf_life_days: 365,
            equipment_needed: vec![
                "Large canning pot".to_string(),
                "Jar rack".to_string(),
                "Mason jars with lids".to_string(),
                "Jar lifter".to_string(),
            ],
            skill_level: SkillLevel::Beginner,
        },
        PreservationMethod {
            name: "Solar Dehydration".to_string(),
            description: "Dry produce on mesh screens in direct sun for 3–5 days. \
                Ideal for tomatoes, herbs, and mushrooms in low-humidity climates."
                .to_string(),
            shelf_life_days: 180,
            equipment_needed: vec![
                "Mesh drying screens".to_string(),
                "Fine mesh cover (insect barrier)".to_string(),
                "Food-safe desiccant packs".to_string(),
            ],
            skill_level: SkillLevel::Beginner,
        },
    ];

    for method in methods {
        let _: Record = agent
            .call_zome_fn("food_preservation", "register_method", method)
            .await;
    }

    // Retrieve all registered methods
    let all_methods: Vec<Record> = agent
        .call_zome_fn("food_preservation", "get_all_methods", ())
        .await;

    assert!(
        all_methods.len() >= 2,
        "Should have at least 2 preservation methods, got {}",
        all_methods.len()
    );
}

// ============================================================================
// food_knowledge tests
// ============================================================================

/// Test: Catalog seed varieties and retrieve by species.
///
/// Verifies `catalog_seed`, `get_seed`, and `get_seeds_by_species`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_knowledge_seed_catalog_crud() {
    let agent = setup_commons().await;

    // Catalog two heirloom tomato varieties
    let seeds = vec![
        SeedVariety {
            name: "Cherokee Purple".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: Some("Cherokee Nation, Tennessee — pre-1890s heirloom".to_string()),
            days_to_maturity: 80,
            companion_plants: vec!["Basil".to_string(), "Carrot".to_string()],
            avoid_plants: vec!["Fennel".to_string(), "Brassica".to_string()],
            seed_saving_notes: Some(
                "Ferment seeds 2–3 days to remove germination inhibitors; dry at room temp"
                    .to_string(),
            ),
        },
        SeedVariety {
            name: "Brandywine".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: Some("Amish community, Ohio — 1885".to_string()),
            days_to_maturity: 85,
            companion_plants: vec!["Parsley".to_string(), "Marigold".to_string()],
            avoid_plants: vec!["Corn".to_string()],
            seed_saving_notes: Some(
                "Potato-leaf type; collect from fully ripe fruit; dry for 2 weeks before storing"
                    .to_string(),
            ),
        },
    ];

    let mut seed_hashes = Vec::new();
    for seed in seeds {
        let record: Record = agent
            .call_zome_fn("food_knowledge", "catalog_seed", seed)
            .await;
        seed_hashes.push(record.action_hashed().hash.clone());
    }

    // Retrieve the first seed by hash
    let retrieved: Option<Record> = agent
        .call_zome_fn("food_knowledge", "get_seed", seed_hashes[0].clone())
        .await;

    assert!(retrieved.is_some(), "Seed should be retrievable by hash");

    // Query both tomato varieties by species name
    let tomato_seeds: Vec<Record> = agent
        .call_zome_fn(
            "food_knowledge",
            "get_seeds_by_species",
            "Solanum lycopersicum".to_string(),
        )
        .await;

    assert_eq!(
        tomato_seeds.len(),
        2,
        "Should find both tomato varieties by species"
    );
}

/// Test: Share a traditional planting practice and a recipe; retrieve each.
///
/// Verifies `share_practice`, `get_practices_by_category`,
/// `share_recipe`, and `get_recipes_by_tag`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_food_knowledge_practices_and_recipes() {
    let agent = setup_commons().await;

    // Share a traditional soil-care practice
    let practice = TraditionalPractice {
        name: "Three Sisters Companion Planting".to_string(),
        description: "Plant corn, beans, and squash together: corn provides a trellis, \
            beans fix nitrogen, squash shades the soil to retain moisture and suppress weeds. \
            A pre-Columbian polyculture still practiced by many Indigenous communities."
            .to_string(),
        region: Some("Americas — Haudenosaunee (Iroquois) tradition".to_string()),
        season: Some("Spring — plant after last frost".to_string()),
        category: PracticeCategory::Planting,
    };

    let practice_record: Record = agent
        .call_zome_fn("food_knowledge", "share_practice", practice)
        .await;

    assert!(
        !practice_record.action_hashed().hash.as_ref().is_empty(),
        "Practice should be shared on DHT"
    );

    // Retrieve practices in the Planting category
    let planting_practices: Vec<Record> = agent
        .call_zome_fn(
            "food_knowledge",
            "get_practices_by_category",
            "Planting".to_string(),
        )
        .await;

    assert!(
        !planting_practices.is_empty(),
        "Should find at least 1 planting practice"
    );

    // Share a seasonal preservation recipe
    let recipe = Recipe {
        name: "Summer Tomato Sauce".to_string(),
        ingredients: vec![
            "2 kg ripe heirloom tomatoes".to_string(),
            "4 cloves garlic".to_string(),
            "1 bunch fresh basil".to_string(),
            "3 tbsp olive oil".to_string(),
            "Salt and pepper to taste".to_string(),
        ],
        instructions: "Blanch and peel tomatoes. Sauté garlic in olive oil 2 min. \
            Add tomatoes, simmer 45 min until thickened. Add basil, season. \
            Water-bath can in sterilised jars (35 min processing time)."
            .to_string(),
        servings: 8,
        prep_time_min: 30,
        tags: vec![
            "tomato".to_string(),
            "canning".to_string(),
            "preservation".to_string(),
            "sauce".to_string(),
        ],
        source_attribution: Some("Community cookbook — Riverside Garden collective, 2024".to_string()),
    };

    let recipe_record: Record = agent
        .call_zome_fn("food_knowledge", "share_recipe", recipe)
        .await;

    assert!(
        !recipe_record.action_hashed().hash.as_ref().is_empty(),
        "Recipe should be shared on DHT"
    );

    // Retrieve recipes tagged with "tomato"
    let tomato_recipes: Vec<Record> = agent
        .call_zome_fn(
            "food_knowledge",
            "get_recipes_by_tag",
            "tomato".to_string(),
        )
        .await;

    assert!(
        !tomato_recipes.is_empty(),
        "Should find at least 1 tomato-tagged recipe"
    );
}

// ============================================================================
// transport_routes tests
// ============================================================================

/// Test: Register a vehicle, create a route, and add stops to it.
///
/// Exercises `register_vehicle`, `get_my_vehicles`, `create_route`,
/// `get_all_routes`, `add_stop`, and `get_route_stops`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_transport_routes_vehicle_route_stops_workflow() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // 1. Register a community cargo van for produce transport
    let vehicle = Vehicle {
        id: "veh-community-van-001".to_string(),
        owner: agent_key.clone(),
        vehicle_type: VehicleType::Van,
        capacity_kg: 800.0,
        capacity_passengers: 7,
        status: VehicleStatus::Available,
    };

    let vehicle_record: Record = agent
        .call_zome_fn("transport_routes", "register_vehicle", vehicle)
        .await;

    assert!(
        !vehicle_record.action_hashed().hash.as_ref().is_empty(),
        "Vehicle should be registered"
    );
    let vehicle_hash = vehicle_record.action_hashed().hash.clone();

    // 2. Verify the vehicle appears in this agent's vehicle list
    let my_vehicles: Vec<Record> = agent
        .call_zome_fn("transport_routes", "get_my_vehicles", ())
        .await;

    assert!(!my_vehicles.is_empty(), "Agent should have at least 1 vehicle");

    // 3. Retrieve the vehicle directly by hash
    let retrieved_vehicle: Option<Record> = agent
        .call_zome_fn("transport_routes", "get_vehicle", vehicle_hash)
        .await;

    assert!(retrieved_vehicle.is_some(), "Vehicle should be retrievable by hash");

    // 4. Create a weekly produce delivery route
    let route = Route {
        id: "rt-produce-loop-001".to_string(),
        name: "Weekly Produce Delivery Loop".to_string(),
        waypoints: vec![
            Waypoint {
                lat: 32.9483,
                lon: -96.7299,
                label: Some("Riverside Garden (start)".to_string()),
            },
            Waypoint {
                lat: 32.9550,
                lon: -96.7200,
                label: Some("Community Center".to_string()),
            },
            Waypoint {
                lat: 32.9400,
                lon: -96.7400,
                label: Some("Food Bank (end)".to_string()),
            },
        ],
        distance_km: 4.8,
        estimated_minutes: 18,
        mode: TransportMode::Driving,
    };

    let route_record: Record = agent
        .call_zome_fn("transport_routes", "create_route", route)
        .await;

    assert!(
        !route_record.action_hashed().hash.as_ref().is_empty(),
        "Route should be created"
    );
    let route_hash = route_record.action_hashed().hash.clone();

    // 5. Verify the route appears in global route listing
    let all_routes: Vec<Record> = agent
        .call_zome_fn("transport_routes", "get_all_routes", ())
        .await;

    assert!(!all_routes.is_empty(), "Should have at least 1 route");

    // 6. Add a pickup stop at the community garden
    let garden_stop = Stop {
        route_hash: route_hash.clone(),
        name: "Riverside Community Garden".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        scheduled_time: Some(1_740_028_800), // 8:00 AM Saturday
        stop_type: StopType::Pickup,
    };

    let _: Record = agent
        .call_zome_fn("transport_routes", "add_stop", garden_stop)
        .await;

    // 7. Add a dropoff stop at the food bank
    let foodbank_stop = Stop {
        route_hash: route_hash.clone(),
        name: "Community Food Bank".to_string(),
        location_lat: 32.9400,
        location_lon: -96.7400,
        scheduled_time: Some(1_740_030_600), // 8:30 AM Saturday
        stop_type: StopType::Dropoff,
    };

    let _: Record = agent
        .call_zome_fn("transport_routes", "add_stop", foodbank_stop)
        .await;

    // 8. Verify both stops are linked to the route
    let route_stops: Vec<Record> = agent
        .call_zome_fn("transport_routes", "get_route_stops", route_hash)
        .await;

    assert_eq!(route_stops.len(), 2, "Route should have exactly 2 stops");
}

// ============================================================================
// transport_sharing tests
// ============================================================================

/// Test: Post a ride offer, submit a request, and create a match.
///
/// Exercises `post_ride_offer`, `get_available_rides`, `request_ride`,
/// and `match_ride`.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_transport_sharing_offer_request_match_workflow() {
    let agent = setup_commons().await;
    let agent_key = agent.cell.agent_pubkey().clone();

    // 1. Register a vehicle to reference in the offer
    let vehicle = Vehicle {
        id: "veh-sharing-001".to_string(),
        owner: agent_key.clone(),
        vehicle_type: VehicleType::Car,
        capacity_kg: 200.0,
        capacity_passengers: 4,
        status: VehicleStatus::Available,
    };

    let vehicle_record: Record = agent
        .call_zome_fn("transport_routes", "register_vehicle", vehicle)
        .await;

    let vehicle_hash = vehicle_record.action_hashed().hash.clone();

    // 2. Post a Saturday farmers market carpool offer
    let offer = RideOffer {
        vehicle_hash: vehicle_hash.clone(),
        route_hash: None,
        driver: agent_key.clone(),
        departure_time: 1_740_028_800, // Saturday 8:00 AM
        seats_available: 3,
        price_per_seat: 0.0, // Community volunteer — no charge
        status: OfferStatus::Open,
    };

    let offer_record: Record = agent
        .call_zome_fn("transport_sharing", "post_ride_offer", offer)
        .await;

    assert!(
        !offer_record.action_hashed().hash.as_ref().is_empty(),
        "Ride offer should be posted"
    );
    let offer_hash = offer_record.action_hashed().hash.clone();

    // 3. Verify the offer appears in available rides
    let available_rides: Vec<Record> = agent
        .call_zome_fn("transport_sharing", "get_available_rides", ())
        .await;

    assert!(
        !available_rides.is_empty(),
        "Should have at least 1 available ride"
    );

    // 4. Submit a ride request from a nearby neighborhood
    let request = RideRequest {
        requester: agent_key.clone(),
        origin_lat: 32.9510,
        origin_lon: -96.7350,
        destination_lat: 32.9483,
        destination_lon: -96.7299,
        requested_time: 1_740_027_600, // 7:40 AM (arrive by 8:00 AM)
        passengers: 2,
        status: RequestStatus::Open,
    };

    let request_record: Record = agent
        .call_zome_fn("transport_sharing", "request_ride", request)
        .await;

    assert!(
        !request_record.action_hashed().hash.as_ref().is_empty(),
        "Ride request should be created"
    );
    let request_hash = request_record.action_hashed().hash.clone();

    // 5. Create a match between the offer and the request
    let ride_match = RideMatch {
        offer_hash,
        request_hash,
        confirmed_at: None, // Pending confirmation
        status: MatchStatus::Pending,
    };

    let match_record: Record = agent
        .call_zome_fn("transport_sharing", "match_ride", ride_match)
        .await;

    assert!(
        !match_record.action_hashed().hash.as_ref().is_empty(),
        "Ride match should be recorded"
    );
}

// ============================================================================
// transport_impact tests
// ============================================================================

/// Test: Log multiple trips with different modes and verify DHT retrieval.
///
/// Exercises `log_trip` and `get_my_trips` across cycling, carpool, and
/// electric vehicle modes.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_transport_impact_trip_logging_and_retrieval() {
    let agent = setup_commons().await;

    // 1. Log a zero-emission cycling trip to the community garden
    let cycling_trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 3.2,
        mode: TripMode::Cycling,
        passengers: 1,
        cargo_kg: 5.0, // Carrying seed exchange haul
        emissions_kg_co2: 0.0,
        logged_at: 1_740_000_000,
    };

    let cycling_record: Record = agent
        .call_zome_fn("transport_impact", "log_trip", cycling_trip)
        .await;

    assert!(
        !cycling_record.action_hashed().hash.as_ref().is_empty(),
        "Cycling trip should be logged"
    );

    // 2. Log a 4-person carpool to the farmers market
    let carpool_trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 8.5,
        mode: TripMode::Carpool,
        passengers: 4,
        cargo_kg: 30.0, // Produce boxes
        emissions_kg_co2: 0.0, // Let zome auto-calculate (0.0 = recalculate)
        logged_at: 1_740_003_600,
    };

    let _: Record = agent
        .call_zome_fn("transport_impact", "log_trip", carpool_trip)
        .await;

    // 3. Log a transit trip on the commuter rail
    let transit_trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 12.0,
        mode: TripMode::Transit,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0,
        logged_at: 1_740_007_200,
    };

    let _: Record = agent
        .call_zome_fn("transport_impact", "log_trip", transit_trip)
        .await;

    // 4. Retrieve all trips logged by this agent
    let my_trips: Vec<Record> = agent
        .call_zome_fn("transport_impact", "get_my_trips", ())
        .await;

    assert!(
        my_trips.len() >= 3,
        "Agent should have at least 3 logged trips, got {}",
        my_trips.len()
    );
}

/// Test: Log a green trip and verify carbon credits are awarded.
///
/// Walking and cycling trips save emissions vs. the solo-car baseline.
/// The `log_trip` function auto-awards `CarbonCredit` entries for saved kg CO2.
/// Verifies `get_my_carbon_credits` returns at least 1 credit.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires commons DNA packed"]
async fn test_transport_impact_carbon_credits_for_green_trip() {
    let agent = setup_commons().await;

    // Log a walking trip — baseline is 0.21 kg CO2/km solo car,
    // so 2.0 km walking saves 0.42 kg CO2 → 1 CarbonCredit record is created
    let walking_trip = TripLog {
        vehicle_hash: None,
        route_hash: None,
        distance_km: 2.0,
        mode: TripMode::Walking,
        passengers: 1,
        cargo_kg: 0.0,
        emissions_kg_co2: 0.0, // Walking emits nothing
        logged_at: 1_740_010_800,
    };

    let trip_record: Record = agent
        .call_zome_fn("transport_impact", "log_trip", walking_trip)
        .await;

    assert!(
        !trip_record.action_hashed().hash.as_ref().is_empty(),
        "Walking trip should be logged"
    );

    // Verify carbon credits were awarded for the emissions saved
    let my_credits: Vec<Record> = agent
        .call_zome_fn("transport_impact", "get_my_carbon_credits", ())
        .await;

    assert!(
        !my_credits.is_empty(),
        "Walking trip (2.0 km) should generate at least 1 carbon credit record \
         (saves 0.42 kg CO2 vs. solo car baseline)"
    );
}
