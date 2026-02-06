//! Climate hApp sweettest integration tests.
//!
//! Tests carbon credit lifecycle, footprint calculation, and credit transfers
//! using the Holochain sweettest framework.
//!
//! Prerequisites:
//!   cd mycelix-climate && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dnas/climate/workdir/ -o dnas/climate/workdir/climate.dna
//!
//! Run: cargo test --release -p mycelix-sweettest -- --ignored climate
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// =============================================================================
// Carbon Credit Creation Tests
// =============================================================================

/// Test: Create a carbon credit and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_get_carbon_credit() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let issuer = &agents[0];

    // Create a carbon credit
    let credit_input = serde_json::json!({
        "project_id": "solar-farm-2024-001",
        "credit_type": "renewable_energy",
        "amount_tonnes": 100.0,
        "vintage_year": 2024,
        "verification_standard": "gold_standard",
        "issuer": format!("did:mycelix:{}", issuer.agent_pubkey),
        "metadata": {
            "location": "Arizona, USA",
            "project_name": "Desert Sun Solar Farm",
            "methodology": "AMS-I.D"
        },
        "created_at": Timestamp::now().as_micros()
    });

    let credit_record: Record = issuer
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    let action_hash = credit_record.action_hashed().hash.clone();
    assert!(!action_hash.as_ref().is_empty(), "Credit should be created");

    // Retrieve the credit
    let retrieved: Option<Record> = issuer
        .call_zome_fn("carbon", "get_carbon_credit", action_hash)
        .await;

    assert!(retrieved.is_some(), "Credit should be retrievable");
}

/// Test: List credits by owner.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_credits_by_owner() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let owner = &agents[0];

    // Create multiple credits
    for i in 1..=3 {
        let credit_input = serde_json::json!({
            "project_id": format!("test-project-{}", i),
            "credit_type": "forestry",
            "amount_tonnes": 50.0 * i as f64,
            "vintage_year": 2024,
            "verification_standard": "verra",
            "issuer": format!("did:mycelix:{}", owner.agent_pubkey),
            "metadata": {
                "location": "Brazil",
                "project_name": format!("Forest Conservation Project {}", i)
            },
            "created_at": Timestamp::now().as_micros()
        });

        let _: Record = owner
            .call_zome_fn("carbon", "create_carbon_credit", credit_input)
            .await;
    }

    // Get all credits owned by this agent
    let credits: Vec<Record> = owner
        .call_zome_fn("carbon", "get_credits_by_owner", owner.agent_pubkey.to_string())
        .await;

    assert!(credits.len() >= 3, "Owner should have at least 3 credits");
}

// =============================================================================
// Credit Transfer Tests
// =============================================================================

/// Test: Transfer carbon credit to another agent.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_transfer_credit() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Seller creates credit
    let credit_input = serde_json::json!({
        "project_id": "transfer-test-001",
        "credit_type": "wind_energy",
        "amount_tonnes": 25.0,
        "vintage_year": 2024,
        "verification_standard": "gold_standard",
        "issuer": format!("did:mycelix:{}", seller.agent_pubkey),
        "metadata": {
            "location": "Texas, USA",
            "project_name": "Prairie Wind Farm"
        },
        "created_at": Timestamp::now().as_micros()
    });

    let credit_record: Record = seller
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    let credit_hash = credit_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Transfer to buyer
    let transfer_input = serde_json::json!({
        "credit_hash": credit_hash,
        "new_owner": buyer.agent_pubkey.to_string(),
        "transfer_note": "Sale of carbon credits",
        "price": {
            "amount": 500.00,
            "currency": "USD"
        },
        "transferred_at": Timestamp::now().as_micros()
    });

    let transfer_record: Record = seller
        .call_zome_fn("carbon", "transfer_credit", transfer_input)
        .await;

    assert!(
        !transfer_record.action_hashed().hash.as_ref().is_empty(),
        "Transfer should be recorded"
    );

    wait_for_dht_sync().await;

    // Verify buyer now owns the credit
    let buyer_credits: Vec<Record> = buyer
        .call_zome_fn("carbon", "get_credits_by_owner", buyer.agent_pubkey.to_string())
        .await;

    assert!(!buyer_credits.is_empty(), "Buyer should own the transferred credit");
}

/// Test: Multi-agent credit visibility via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_multi_agent_credit_visibility() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        2,
    )
    .await;

    let issuer = &agents[0];
    let verifier = &agents[1];

    // Issuer creates credit
    let credit_input = serde_json::json!({
        "project_id": "visibility-test-001",
        "credit_type": "methane_capture",
        "amount_tonnes": 75.0,
        "vintage_year": 2024,
        "verification_standard": "verra",
        "issuer": format!("did:mycelix:{}", issuer.agent_pubkey),
        "metadata": {
            "location": "California, USA",
            "project_name": "Landfill Methane Recovery"
        },
        "created_at": Timestamp::now().as_micros()
    });

    let credit_record: Record = issuer
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    let credit_hash = credit_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Verifier should be able to see the credit
    let retrieved: Option<Record> = verifier
        .call_zome_fn("carbon", "get_carbon_credit", credit_hash)
        .await;

    assert!(retrieved.is_some(), "Credit should be visible to verifier via DHT");
}

// =============================================================================
// Credit Retirement Tests
// =============================================================================

/// Test: Retire carbon credit (permanent offset).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_retire_credit() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let owner = &agents[0];

    // Create credit to retire
    let credit_input = serde_json::json!({
        "project_id": "retirement-test-001",
        "credit_type": "blue_carbon",
        "amount_tonnes": 10.0,
        "vintage_year": 2024,
        "verification_standard": "gold_standard",
        "issuer": format!("did:mycelix:{}", owner.agent_pubkey),
        "metadata": {
            "location": "Florida, USA",
            "project_name": "Mangrove Restoration"
        },
        "created_at": Timestamp::now().as_micros()
    });

    let credit_record: Record = owner
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    let credit_hash = credit_record.action_hashed().hash.clone();

    // Retire the credit
    let retire_input = serde_json::json!({
        "credit_hash": credit_hash,
        "beneficiary": "Luminous Dynamics Carbon Neutral Initiative",
        "retirement_reason": "Annual carbon offset commitment",
        "retired_at": Timestamp::now().as_micros()
    });

    let retired_record: Record = owner
        .call_zome_fn("carbon", "retire_credit", retire_input)
        .await;

    assert!(
        !retired_record.action_hashed().hash.as_ref().is_empty(),
        "Retirement should be recorded"
    );
}

/// Test: Get credits by status (Active, Transferred, Retired).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_credits_by_status() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let owner = &agents[0];

    // Create active credit
    let credit_input = serde_json::json!({
        "project_id": "status-test-active",
        "credit_type": "solar",
        "amount_tonnes": 20.0,
        "vintage_year": 2024,
        "verification_standard": "verra",
        "issuer": format!("did:mycelix:{}", owner.agent_pubkey),
        "metadata": {},
        "created_at": Timestamp::now().as_micros()
    });

    let _: Record = owner
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    // Get active credits
    let active_credits: Vec<Record> = owner
        .call_zome_fn("carbon", "get_credits_by_status", "Active")
        .await;

    assert!(!active_credits.is_empty(), "Should have active credits");
}

// =============================================================================
// Carbon Footprint Tests
// =============================================================================

/// Test: Calculate carbon footprint.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_calculate_footprint() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let calculator = &agents[0];

    // Calculate footprint for activities
    let footprint_input = serde_json::json!({
        "entity_id": format!("did:mycelix:{}", calculator.agent_pubkey),
        "period": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        "activities": [
            {
                "category": "electricity",
                "amount": 12000.0,
                "unit": "kWh",
                "emission_factor": 0.42
            },
            {
                "category": "natural_gas",
                "amount": 500.0,
                "unit": "therms",
                "emission_factor": 5.3
            },
            {
                "category": "vehicle_fuel",
                "amount": 800.0,
                "unit": "gallons",
                "emission_factor": 8.89
            }
        ],
        "calculated_at": Timestamp::now().as_micros()
    });

    let footprint_record: Record = calculator
        .call_zome_fn("carbon", "calculate_footprint", footprint_input)
        .await;

    assert!(
        !footprint_record.action_hashed().hash.as_ref().is_empty(),
        "Footprint should be calculated"
    );
}

/// Test: Get footprint for entity.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_footprint() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let entity = &agents[0];

    // First calculate a footprint
    let footprint_input = serde_json::json!({
        "entity_id": format!("did:mycelix:{}", entity.agent_pubkey),
        "period": {
            "start": "2024-01-01",
            "end": "2024-06-30"
        },
        "activities": [
            {
                "category": "electricity",
                "amount": 6000.0,
                "unit": "kWh",
                "emission_factor": 0.42
            }
        ],
        "calculated_at": Timestamp::now().as_micros()
    });

    let footprint_record: Record = entity
        .call_zome_fn("carbon", "calculate_footprint", footprint_input)
        .await;

    let footprint_hash = footprint_record.action_hashed().hash.clone();

    // Retrieve the footprint
    let retrieved: Option<Record> = entity
        .call_zome_fn("carbon", "get_footprint", footprint_hash)
        .await;

    assert!(retrieved.is_some(), "Footprint should be retrievable");
}

// =============================================================================
// Project Management Tests
// =============================================================================

/// Test: Create climate project.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_create_project() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let developer = &agents[0];

    // Create a climate project
    let project_input = serde_json::json!({
        "project_id": "project-reforestation-2024",
        "name": "Amazon Reforestation Initiative",
        "description": "Large-scale reforestation of degraded lands",
        "project_type": "forestry",
        "location": {
            "country": "Brazil",
            "region": "Amazonas",
            "coordinates": [-3.4653, -62.2159]
        },
        "developer": format!("did:mycelix:{}", developer.agent_pubkey),
        "start_date": "2024-01-01",
        "expected_credits_per_year": 50000.0,
        "verification_standard": "verra",
        "sdg_goals": [13, 15, 6],
        "created_at": Timestamp::now().as_micros()
    });

    let project_record: Record = developer
        .call_zome_fn("projects", "create_project", project_input)
        .await;

    assert!(
        !project_record.action_hashed().hash.as_ref().is_empty(),
        "Project should be created"
    );
}

/// Test: Link credit to project.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_link_credit_to_project() {
    let agents = setup_test_agents(
        &DnaPaths::climate(),
        "mycelix-climate",
        1,
    )
    .await;

    let developer = &agents[0];

    // Create project
    let project_input = serde_json::json!({
        "project_id": "link-test-project",
        "name": "Credit Link Test Project",
        "description": "Testing credit-project linkage",
        "project_type": "renewable_energy",
        "location": {
            "country": "USA",
            "region": "Nevada"
        },
        "developer": format!("did:mycelix:{}", developer.agent_pubkey),
        "start_date": "2024-01-01",
        "expected_credits_per_year": 10000.0,
        "verification_standard": "gold_standard",
        "sdg_goals": [7, 13],
        "created_at": Timestamp::now().as_micros()
    });

    let project_record: Record = developer
        .call_zome_fn("projects", "create_project", project_input)
        .await;

    let project_hash = project_record.action_hashed().hash.clone();

    // Create credit linked to project
    let credit_input = serde_json::json!({
        "project_id": "link-test-project",
        "project_hash": project_hash,
        "credit_type": "solar",
        "amount_tonnes": 100.0,
        "vintage_year": 2024,
        "verification_standard": "gold_standard",
        "issuer": format!("did:mycelix:{}", developer.agent_pubkey),
        "metadata": {},
        "created_at": Timestamp::now().as_micros()
    });

    let credit_record: Record = developer
        .call_zome_fn("carbon", "create_carbon_credit", credit_input)
        .await;

    assert!(
        !credit_record.action_hashed().hash.as_ref().is_empty(),
        "Credit should be created and linked to project"
    );
}
