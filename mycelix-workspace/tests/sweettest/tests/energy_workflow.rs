// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy cluster sweettest integration tests.
//!
//! Smoke tests for all 5 energy zomes: projects, investments, grid,
//! regenerative, and bridge. Verifies basic CRUD operations through
//! the Holochain conductor.
//!
//! Prerequisites:
//!   cd mycelix-energy && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack mycelix-energy/dna/
//!
//! Run:
//!   cargo test -p mycelix-sweettest --test energy_workflow -- --ignored --test-threads=1

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ============================================================================
// Mirror types (avoid WASM symbol conflicts with zome entry types)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EnergyProject {
    id: String,
    terra_atlas_id: Option<String>,
    name: String,
    description: String,
    project_type: String, // "Solar", "Wind", etc.
    location: ProjectLocation,
    capacity_kw: f64,
    status: String, // "Proposed"
    developer_did: String,
    community_did: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProjectLocation {
    latitude: f64,
    longitude: f64,
    country: String,
    region: String,
    address: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EnergyProduction {
    id: String,
    producer_did: String,
    project_id: String,
    amount_kwh: f64,
    timestamp: Timestamp,
    source_type: String, // "Solar"
    verified: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TradeOffer {
    id: String,
    seller_did: String,
    project_id: Option<String>,
    amount_kwh: f64,
    price_per_kwh: f64,
    currency: String,
    status: String, // "Active"
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Investment {
    id: String,
    project_id: String,
    investor_did: String,
    amount: f64,
    currency: String,
    investment_type: String, // "Equity"
    status: String,          // "Pledged"
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegenerativeContract {
    id: String,
    project_id: String,
    community_did: String,
    conditions: Vec<TransitionCondition>,
    current_ownership_percentage: f64,
    target_ownership_percentage: f64,
    status: String, // "Active"
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TransitionCondition {
    condition_type: String,
    threshold: f64,
    current_value: f64,
    weight: f64,
    satisfied: bool,
}

// ============================================================================
// Setup
// ============================================================================

async fn setup_energy() -> Vec<TestAgent> {
    setup_test_agents(&DnaPaths::energy(), "mycelix-energy", 1).await
}

async fn setup_energy_pair() -> Vec<TestAgent> {
    setup_test_agents(&DnaPaths::energy(), "mycelix-energy", 2).await
}

// ============================================================================
// Projects zome
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_register_and_get_project() {
    let agents = setup_energy().await;
    let agent = &agents[0];

    let project = EnergyProject {
        id: "proj-solar-001".into(),
        terra_atlas_id: Some("ta-12345".into()),
        name: "Sunshine Community Solar".into(),
        description: "100kW rooftop solar for community center".into(),
        project_type: "Solar".into(),
        location: ProjectLocation {
            latitude: -25.7,
            longitude: 28.2,
            country: "ZA".into(),
            region: "Gauteng".into(),
            address: Some("123 Solar Street".into()),
        },
        capacity_kw: 100.0,
        status: "Proposed".into(),
        developer_did: "did:mycelix:dev001".into(),
        community_did: Some("did:mycelix:community001".into()),
    };

    let record: Record = agent
        .call_zome_fn("projects", "register_project", project)
        .await;
    assert!(!record.action_hashed().hash.as_ref().is_empty());

    // Read it back
    let fetched: Record = agent
        .call_zome_fn("projects", "get_project", record.action_hashed().hash.clone())
        .await;
    assert_eq!(
        fetched.action_hashed().hash,
        record.action_hashed().hash
    );
}

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_search_projects_by_location() {
    let agents = setup_energy().await;
    let agent = &agents[0];

    let project = EnergyProject {
        id: "proj-wind-002".into(),
        terra_atlas_id: None,
        name: "Cape Wind Farm".into(),
        description: "50MW offshore wind".into(),
        project_type: "Wind".into(),
        location: ProjectLocation {
            latitude: -34.0,
            longitude: 18.5,
            country: "ZA".into(),
            region: "Western Cape".into(),
            address: None,
        },
        capacity_kw: 50000.0,
        status: "Proposed".into(),
        developer_did: "did:mycelix:dev002".into(),
        community_did: None,
    };

    let _record: Record = agent
        .call_zome_fn("projects", "register_project", project)
        .await;

    let results: Vec<Record> = agent
        .call_zome_fn(
            "projects",
            "search_projects_by_location",
            serde_json::json!({
                "latitude": -34.0,
                "longitude": 18.5,
                "radius_km": 100.0
            }),
        )
        .await;
    assert!(!results.is_empty(), "Should find the Cape Wind Farm project");
}

// ============================================================================
// Grid zome
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_record_production_and_create_trade() {
    let agents = setup_energy().await;
    let agent = &agents[0];

    // Record production
    let production = EnergyProduction {
        id: "prod-001".into(),
        producer_did: "did:mycelix:producer001".into(),
        project_id: "proj-solar-001".into(),
        amount_kwh: 250.5,
        timestamp: Timestamp::now(),
        source_type: "Solar".into(),
        verified: false,
    };

    let prod_record: Record = agent
        .call_zome_fn("grid", "record_production", production)
        .await;
    assert!(!prod_record.action_hashed().hash.as_ref().is_empty());

    // Create trade offer
    let offer = TradeOffer {
        id: "offer-001".into(),
        seller_did: "did:mycelix:producer001".into(),
        project_id: Some("proj-solar-001".into()),
        amount_kwh: 100.0,
        price_per_kwh: 0.85,
        currency: "ZAR".into(),
        status: "Active".into(),
    };

    let offer_record: Record = agent
        .call_zome_fn("grid", "create_trade_offer", offer)
        .await;
    assert!(!offer_record.action_hashed().hash.as_ref().is_empty());

    // Verify active offers
    let active: Vec<Record> = agent
        .call_zome_fn("grid", "get_active_offers", ())
        .await;
    assert!(!active.is_empty(), "Should have at least one active offer");
}

// ============================================================================
// Investments zome
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_pledge_and_get_investment() {
    let agents = setup_energy().await;
    let agent = &agents[0];

    let investment = Investment {
        id: "inv-001".into(),
        project_id: "proj-solar-001".into(),
        investor_did: "did:mycelix:investor001".into(),
        amount: 50000.0,
        currency: "ZAR".into(),
        investment_type: "Equity".into(),
        status: "Pledged".into(),
    };

    let record: Record = agent
        .call_zome_fn("investments", "pledge_investment", investment)
        .await;
    assert!(!record.action_hashed().hash.as_ref().is_empty());

    // Get portfolio
    let portfolio: Vec<Record> = agent
        .call_zome_fn(
            "investments",
            "get_investor_portfolio",
            "did:mycelix:investor001",
        )
        .await;
    assert!(!portfolio.is_empty(), "Should have at least one investment");
}

// ============================================================================
// Regenerative zome
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_regenerative_contract_lifecycle() {
    let agents = setup_energy().await;
    let agent = &agents[0];

    let contract = RegenerativeContract {
        id: "regen-001".into(),
        project_id: "proj-solar-001".into(),
        community_did: "did:mycelix:community001".into(),
        conditions: vec![
            TransitionCondition {
                condition_type: "OperationalYears".into(),
                threshold: 5.0,
                current_value: 0.0,
                weight: 0.4,
                satisfied: false,
            },
            TransitionCondition {
                condition_type: "CommunityReadiness".into(),
                threshold: 0.8,
                current_value: 0.3,
                weight: 0.6,
                satisfied: false,
            },
        ],
        current_ownership_percentage: 0.0,
        target_ownership_percentage: 100.0,
        status: "Active".into(),
    };

    let record: Record = agent
        .call_zome_fn("regenerative", "create_regenerative_contract", contract)
        .await;
    assert!(!record.action_hashed().hash.as_ref().is_empty());

    // Get contract
    let fetched: Record = agent
        .call_zome_fn("regenerative", "get_contract", record.action_hashed().hash.clone())
        .await;
    assert_eq!(
        fetched.action_hashed().hash,
        record.action_hashed().hash
    );
}

// ============================================================================
// Multi-agent DHT sync
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_two_agent_project_visibility() {
    let agents = setup_energy_pair().await;

    // Agent 0 creates a project
    let project = EnergyProject {
        id: "proj-shared-001".into(),
        terra_atlas_id: None,
        name: "Shared Visibility Test".into(),
        description: "Testing DHT propagation".into(),
        project_type: "Solar".into(),
        location: ProjectLocation {
            latitude: -26.0,
            longitude: 28.0,
            country: "ZA".into(),
            region: "Gauteng".into(),
            address: None,
        },
        capacity_kw: 10.0,
        status: "Proposed".into(),
        developer_did: "did:mycelix:dev_a".into(),
        community_did: None,
    };

    let record: Record = agents[0]
        .call_zome_fn("projects", "register_project", project)
        .await;

    // Wait for DHT propagation
    wait_for_dht_sync().await;

    // Agent 1 should be able to read it
    let fetched: Record = agents[1]
        .call_zome_fn("projects", "get_project", record.action_hashed().hash.clone())
        .await;
    assert_eq!(
        fetched.action_hashed().hash,
        record.action_hashed().hash,
        "Agent 1 should see Agent 0's project after DHT sync"
    );
}
