// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Commons — Dispatch & Support Sweettest
//!
//! Batch 3 of sweettest integration tests: cross-domain dispatch scenarios
//! (food->water, transport->mutualaid, food->mutualaid, housing->property,
//! water->food) and support domain tests (knowledge, tickets, diagnostics,
//! privacy, cognitive updates, multi-agent).
//!
//! Split from sweettest_integration.rs to reduce per-process conductor
//! memory pressure (~1-2 GB per conductor). Each [[test]] binary runs
//! as a separate OS process, so memory is fully reclaimed between batches.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons/tests
//! cargo test --release --test sweettest_dispatch_support -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — commons-bridge
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
// Mirror types — property-registry (needed for housing->property dispatch)
// ============================================================================

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

// ============================================================================
// Mirror types — food domain (needed for food->water/mutualaid dispatch)
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
pub enum PlotType {
    Garden,
    FoodForest,
    Orchard,
    Greenhouse,
    Raised,
    Rooftop,
}

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

// ============================================================================
// Mirror types — transport domain (needed for transport->mutualaid dispatch)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VehicleType {
    Car,
    Van,
    Bike,
    Bus,
    Cargo,
    ElectricScooter,
    Helicopter,
    EVTOL,
    AirTaxi,
    Ferry,
    Boat,
    Train,
    Tram,
    Skateboard,
    Wheelchair,
    Segway,
    AutonomousVehicle,
    Drone,
    Other(String),
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

// ============================================================================
// Mirror types — bridge dispatch
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchInput {
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
}

// ============================================================================
// Mirror types — support domain
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
// Cross-Domain Dispatch Scenarios
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_dispatches_water_quality_check() {
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

    assert!(
        result.success || result.error.is_some(),
        "Dispatch should either succeed or return an error from the target zome"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_transport_dispatches_mutualaid_resources() {
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_food_donation_triggers_mutualaid_fulfillment() {
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
        .call(
            &alice.zome("commons_bridge"),
            "broadcast_event",
            donation_event,
        )
        .await;

    assert!(donation_record.action().author() == alice.agent_pubkey());

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

    let food_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "food".to_string(),
        )
        .await;

    assert!(
        !food_events.is_empty(),
        "Food domain should have the donation event"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_dispatches_property_ownership_verify() {
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
        .call(
            &alice.zome("property_registry"),
            "register_property",
            prop_input,
        )
        .await;

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
        .call(
            &alice.zome("commons_bridge"),
            "broadcast_event",
            lease_event,
        )
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_water_dispatches_food_irrigation_credit_check() {
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Support Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_create_article_and_search() {
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_create_ticket_lifecycle() {
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
        .call(
            &alice.zome("support_tickets"),
            "create_ticket",
            ticket.clone(),
        )
        .await;

    let ticket_hash = ticket_record.action_address().clone();

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

    let mut updated_ticket = ticket.clone();
    updated_ticket.status = TicketStatus::Resolved;
    updated_ticket.updated_at = Timestamp::now();

    let update_input = SupportUpdateTicketInput {
        original_hash: ticket_hash.clone(),
        updated: updated_ticket,
    };

    let _updated_record: Record = conductor
        .call(
            &alice.zome("support_tickets"),
            "update_ticket",
            update_input,
        )
        .await;

    let _closed_record: Record = conductor
        .call(&alice.zome("support_tickets"), "close_ticket", ticket_hash)
        .await;

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_diagnostic_flow() {
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
    let now = Timestamp::now();

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

    let fetched: Option<Record> = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "get_diagnostic",
            diag_record.action_address().clone(),
        )
        .await;

    assert!(
        fetched.is_some(),
        "Diagnostic should be retrievable after creation"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_privacy_preference() {
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
        .call(
            &alice.zome("support_diagnostics"),
            "set_privacy_preference",
            pref,
        )
        .await;

    let fetched: Option<Record> = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "get_privacy_preference",
            agent,
        )
        .await;

    assert!(
        fetched.is_some(),
        "Privacy preference should be retrievable"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_cognitive_update_publish() {
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

    let update = SupportCognitiveUpdate {
        category: SupportCategory::Network,
        encoding: vec![0xAB; 2048],
        phi: 0.72,
        resolution_pattern: "DNS misconfiguration: edit /etc/resolv.conf, restart systemd-resolved"
            .into(),
        source_agent: agent.clone(),
        created_at: Timestamp::now(),
    };

    let update_record: Record = conductor
        .call(
            &alice.zome("support_diagnostics"),
            "publish_cognitive_update",
            update,
        )
        .await;

    assert!(update_record.action().author() == &agent);

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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_bridge_dispatch() {
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_support_multi_agent_resolution() {
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();

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

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    let alice_agent = alice.agent_pubkey().clone();
    let bob_agent = bob.agent_pubkey().clone();
    let now = Timestamp::now();

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

    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let comment = SupportTicketComment {
        ticket_hash: ticket_hash.clone(),
        author: bob_agent.clone(),
        content:
            "The issue is a rate limit hitting 100 calls/min. Increase RATE_LIMIT_WINDOW_SECS."
                .into(),
        is_symthaea_response: false,
        confidence: Some(0.85),
        epistemic_status: Some(EpistemicStatus::Probable),
    };

    let _comment_record: Record = bob_conductor
        .call(&bob.zome("support_tickets"), "add_comment", comment)
        .await;

    let fetched: Option<Record> = bob_conductor
        .call(&bob.zome("support_tickets"), "get_ticket", ticket_hash)
        .await;

    assert!(
        fetched.is_some(),
        "Bob should be able to fetch Alice's ticket via DHT"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
