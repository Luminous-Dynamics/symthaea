//! Resilience hApp sweettest integration tests.
//!
//! Tests workflow operations across hearth, knowledge, supplychain,
//! commons, and care domains using the Holochain sweettest framework
//! with real conductors.
//!
//! Prerequisites:
//!   Build the relevant cluster DNAs (hearth, commons_care, supplychain, knowledge)
//!   and pack them with `hc dna pack`.
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored resilience
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

/// Decode a Record's entry bytes into a serde_json::Value via msgpack.
/// HC 0.6 stores entries as msgpack; `to_app_option::<serde_json::Value>()` doesn't work
/// because `serde_json::Value` doesn't implement `TryFrom<SerializedBytes>`.
fn decode_record_entry(record: &Record) -> serde_json::Value {
    let entry = record
        .entry()
        .as_option()
        .expect("Record should have an entry");
    let bytes = match entry {
        Entry::App(app_entry) => app_entry.bytes().to_vec(),
        _ => panic!("Expected App entry"),
    };
    let value: serde_json::Value =
        rmp_serde::from_slice(&bytes).expect("Should decode msgpack entry");
    value
}

// ============================================================================
// HEARTH EMERGENCY PLAN TESTS
// ============================================================================

/// Test: Create a hearth, create an emergency plan with contacts, verify retrieval.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires conductor + 8-DNA resilience hApp
async fn test_hearth_emergency_plan_workflow() {
    let agents = setup_test_agents(
        &DnaPaths::hearth(),
        "mycelix-hearth",
        1,
    )
    .await;

    let agent = &agents[0];
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let now = Timestamp::now();

    // Create a hearth (household unit)
    let hearth_input = serde_json::json!({
        "name": "Resilience Test Hearth",
        "description": "A hearth for testing emergency preparedness workflows",
        "creator_did": agent_did,
        "created": now
    });

    let hearth_record: Record = agent
        .call_zome_fn("hearth_emergency", "create_hearth", hearth_input)
        .await;

    let hearth_entry = decode_record_entry(&hearth_record);
    assert!(
        !hearth_record.action_hashed().hash.as_ref().is_empty(),
        "Hearth should be created"
    );

    wait_for_dht_sync().await;

    // Create an emergency plan with contacts
    let plan_input = serde_json::json!({
        "hearth_id": hearth_entry.get("id").and_then(|v| v.as_str()).unwrap_or("hearth-1"),
        "title": "Wildfire Evacuation Plan",
        "description": "Emergency procedures for wildfire evacuation",
        "contacts": [
            {
                "name": "Alice",
                "role": "Primary Contact",
                "phone": "+1-555-0101"
            },
            {
                "name": "Bob",
                "role": "Emergency Services Liaison",
                "phone": "+1-555-0102"
            }
        ],
        "meeting_point": "Community center parking lot",
        "created": now,
        "updated": now
    });

    let plan_record: Record = agent
        .call_zome_fn("hearth_emergency", "create_emergency_plan", plan_input)
        .await;

    let plan_entry = decode_record_entry(&plan_record);
    assert!(
        !plan_record.action_hashed().hash.as_ref().is_empty(),
        "Emergency plan should be created"
    );

    wait_for_dht_sync().await;

    // Retrieve the emergency plan
    let plan_id = plan_entry
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("plan-1")
        .to_string();

    let retrieved: Option<Record> = agent
        .call_zome_fn("hearth_emergency", "get_emergency_plan", plan_id)
        .await;

    assert!(
        retrieved.is_some(),
        "Emergency plan should be retrievable after creation"
    );
}

// ============================================================================
// KNOWLEDGE CLAIM TESTS
// ============================================================================

/// Test: Submit a knowledge claim with tags, search by tag, verify it appears.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires conductor + 8-DNA resilience hApp
async fn test_knowledge_claim_submit_and_search() {
    let agents = setup_test_agents(
        &DnaPaths::commons_care(),
        "mycelix-commons-care",
        1,
    )
    .await;

    let agent = &agents[0];
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let now = Timestamp::now();

    // Submit a knowledge claim with tags
    let claim_input = serde_json::json!({
        "title": "Rainwater Harvesting Best Practices",
        "content": "Rooftop collection systems in semi-arid regions can yield 50-80 liters per square meter annually when properly maintained.",
        "author_did": agent_did,
        "tags": ["water", "harvesting", "resilience", "semi-arid"],
        "epistemic_level": "E2",
        "created": now
    });

    let claim_record: Record = agent
        .call_zome_fn("claims", "submit_claim", claim_input)
        .await;

    assert!(
        !claim_record.action_hashed().hash.as_ref().is_empty(),
        "Knowledge claim should be submitted"
    );

    wait_for_dht_sync().await;

    // Search by tag
    let search_input = serde_json::json!({
        "tag": "water"
    });

    let results: Vec<Record> = agent
        .call_zome_fn("claims", "search_by_tag", search_input)
        .await;

    assert!(
        !results.is_empty(),
        "Search by tag 'water' should return at least one result"
    );
}

// ============================================================================
// SUPPLY CHAIN INVENTORY TESTS
// ============================================================================

/// Test: Add an inventory item, record a stock level, check low stock detection.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires conductor + 8-DNA resilience hApp
async fn test_supply_inventory_workflow() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let agent = &agents[0];
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let now = Timestamp::now();

    // Add an inventory item
    let item_input = serde_json::json!({
        "name": "Emergency Water Filters",
        "description": "Portable water filtration units for emergency deployment",
        "category": "water-safety",
        "unit": "units",
        "minimum_stock": 50,
        "owner_did": agent_did,
        "created": now
    });

    let item_record: Record = agent
        .call_zome_fn("inventory_coordinator", "add_item", item_input)
        .await;

    let item_entry = decode_record_entry(&item_record);
    assert!(
        !item_record.action_hashed().hash.as_ref().is_empty(),
        "Inventory item should be created"
    );

    wait_for_dht_sync().await;

    // Record a stock level (below minimum to trigger low stock)
    let item_id = item_entry
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("item-1")
        .to_string();

    let stock_input = serde_json::json!({
        "item_id": item_id,
        "quantity": 15,
        "recorded_by": agent_did,
        "timestamp": now
    });

    let stock_record: Record = agent
        .call_zome_fn("inventory_coordinator", "record_stock_level", stock_input)
        .await;

    assert!(
        !stock_record.action_hashed().hash.as_ref().is_empty(),
        "Stock level should be recorded"
    );

    wait_for_dht_sync().await;

    // Check low stock detection
    let low_stock: Vec<Record> = agent
        .call_zome_fn("inventory_coordinator", "get_low_stock_items", ())
        .await;

    assert!(
        !low_stock.is_empty(),
        "Low stock detection should flag items below minimum (15 < 50)"
    );
}

// ============================================================================
// WATER SYSTEM REGISTRATION TESTS
// ============================================================================

/// Test: Register a water system, submit a reading, verify it's stored.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires conductor + 8-DNA resilience hApp
async fn test_water_system_registration() {
    let agents = setup_test_agents(
        &DnaPaths::commons_care(),
        "mycelix-commons-care",
        1,
    )
    .await;

    let agent = &agents[0];
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let now = Timestamp::now();

    // Register a water harvest system
    let system_input = serde_json::json!({
        "name": "Community Rainwater Cistern Alpha",
        "system_type": "RainwaterHarvest",
        "location": {
            "latitude": 32.7767,
            "longitude": -96.7970
        },
        "capacity_liters": 5000.0,
        "owner_did": agent_did,
        "created": now
    });

    let system_record: Record = agent
        .call_zome_fn("water_capture", "register_harvest_system", system_input)
        .await;

    let system_entry = decode_record_entry(&system_record);
    assert!(
        !system_record.action_hashed().hash.as_ref().is_empty(),
        "Water system should be registered"
    );

    wait_for_dht_sync().await;

    // Submit a reading
    let system_id = system_entry
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("system-1")
        .to_string();

    let reading_input = serde_json::json!({
        "system_id": system_id,
        "level_liters": 3200.0,
        "quality_ph": 7.1,
        "recorded_by": agent_did,
        "timestamp": now
    });

    let reading_record: Record = agent
        .call_zome_fn("water_capture", "submit_reading", reading_input)
        .await;

    assert!(
        !reading_record.action_hashed().hash.as_ref().is_empty(),
        "Water reading should be recorded"
    );

    wait_for_dht_sync().await;

    // Verify the reading is stored by retrieving system readings
    let readings: Vec<Record> = agent
        .call_zome_fn("water_capture", "get_system_readings", system_id.clone())
        .await;

    assert!(
        !readings.is_empty(),
        "System should have at least one reading after submission"
    );
}

// ============================================================================
// CARE CIRCLE MEMBERSHIP TESTS
// ============================================================================

/// Test: Get care circles, join one, verify membership.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires conductor + 8-DNA resilience hApp
async fn test_care_circle_join_workflow() {
    let agents = setup_test_agents(
        &DnaPaths::commons_care(),
        "mycelix-commons-care",
        2,
    )
    .await;

    let creator = &agents[0];
    let joiner = &agents[1];
    let creator_did = format!("did:mycelix:{}", creator.agent_pubkey);
    let joiner_did = format!("did:mycelix:{}", joiner.agent_pubkey);
    let now = Timestamp::now();

    // Creator creates a care circle
    let circle_input = serde_json::json!({
        "name": "Neighborhood Mutual Aid Circle",
        "description": "A care circle for coordinating neighborhood mutual aid and emergency support",
        "creator_did": creator_did,
        "max_members": 20,
        "created": now
    });

    let circle_record: Record = creator
        .call_zome_fn("care_circles", "create_circle", circle_input)
        .await;

    let circle_entry = decode_record_entry(&circle_record);
    assert!(
        !circle_record.action_hashed().hash.as_ref().is_empty(),
        "Care circle should be created"
    );

    wait_for_dht_sync().await;

    // List available care circles
    let circles: Vec<Record> = joiner
        .call_zome_fn("care_circles", "get_circles", ())
        .await;

    assert!(
        !circles.is_empty(),
        "At least one care circle should be visible"
    );

    // Joiner joins the care circle
    let circle_id = circle_entry
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("circle-1")
        .to_string();

    let join_input = serde_json::json!({
        "circle_id": circle_id,
        "member_did": joiner_did,
        "role": "Member"
    });

    let join_record: Record = joiner
        .call_zome_fn("care_circles", "join_circle", join_input)
        .await;

    assert!(
        !join_record.action_hashed().hash.as_ref().is_empty(),
        "Join operation should succeed"
    );

    wait_for_dht_sync().await;

    // Verify membership
    let members: Vec<Record> = creator
        .call_zome_fn("care_circles", "get_circle_members", circle_id.clone())
        .await;

    assert!(
        !members.is_empty(),
        "Care circle should have at least one member after join"
    );
}
