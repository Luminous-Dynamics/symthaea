// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Supply Chain hApp sweettest integration tests.
//!
//! Tests inventory management, logistics/shipment tracking, and provenance
//! claims using the Holochain sweettest framework.
//!
//! Prerequisites:
//!   cd mycelix-supplychain/holochain && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/supplychain.dna
//!
//! Run: cargo test --release -p mycelix-sweettest -- --ignored supplychain
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// =============================================================================
// Inventory Management Tests
// =============================================================================

/// Test: Create an inventory item and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_get_item() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let warehouse = &agents[0];

    // Create an inventory item
    let item_input = serde_json::json!({
        "sku": "WIDGET-001",
        "name": "Industrial Widget",
        "description": "High-grade industrial widget for manufacturing",
        "category": "components",
        "unit": "pieces",
        "reorder_point": 100,
        "reorder_quantity": 500
    });

    let item_hash: ActionHash = warehouse
        .call_zome_fn("inventory", "create_item", item_input)
        .await;

    assert!(!item_hash.as_ref().is_empty(), "Item should be created");

    // Retrieve the item
    let retrieved: Option<serde_json::Value> = warehouse
        .call_zome_fn("inventory", "get_item", item_hash)
        .await;

    assert!(retrieved.is_some(), "Item should be retrievable");
    let item = retrieved.unwrap();
    assert_eq!(item["sku"], "WIDGET-001");
    assert_eq!(item["name"], "Industrial Widget");
}

/// Test: List all inventory items.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_list_inventory_items() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let warehouse = &agents[0];

    // Create multiple items
    for i in 1..=3 {
        let item_input = serde_json::json!({
            "sku": format!("TEST-ITEM-{}", i),
            "name": format!("Test Item {}", i),
            "description": null,
            "category": "test",
            "unit": "units",
            "reorder_point": 10,
            "reorder_quantity": 50
        });

        let _: ActionHash = warehouse
            .call_zome_fn("inventory", "create_item", item_input)
            .await;
    }

    // List all items
    let items: Vec<serde_json::Value> = warehouse
        .call_zome_fn("inventory", "get_all_items", ())
        .await;

    assert!(items.len() >= 3, "Should have at least 3 items");
}

/// Test: Update stock levels and retrieve them.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_stock_level_management() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let warehouse = &agents[0];

    // Create an item
    let item_input = serde_json::json!({
        "sku": "STOCK-TEST-001",
        "name": "Stock Test Item",
        "description": null,
        "category": "test",
        "unit": "units",
        "reorder_point": 50,
        "reorder_quantity": 200
    });

    let item_hash: ActionHash = warehouse
        .call_zome_fn("inventory", "create_item", item_input)
        .await;

    // Update stock at two locations
    let stock_input_1 = serde_json::json!({
        "item_hash": item_hash,
        "location": "warehouse-a",
        "quantity": 150,
        "reserved": 20
    });

    let _: ActionHash = warehouse
        .call_zome_fn("inventory", "update_stock", stock_input_1)
        .await;

    let stock_input_2 = serde_json::json!({
        "item_hash": item_hash,
        "location": "warehouse-b",
        "quantity": 75,
        "reserved": 0
    });

    let _: ActionHash = warehouse
        .call_zome_fn("inventory", "update_stock", stock_input_2)
        .await;

    // Get total stock
    let total: u64 = warehouse
        .call_zome_fn("inventory", "get_total_stock", item_hash.clone())
        .await;

    assert_eq!(total, 225, "Total stock should be 225");

    // Get available stock
    let available: u64 = warehouse
        .call_zome_fn("inventory", "get_available_stock", item_hash)
        .await;

    assert_eq!(available, 205, "Available stock should be 205 (225 - 20 reserved)");
}

/// Test: Record stock movement.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_stock_movement() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let warehouse = &agents[0];

    // Create an item
    let item_input = serde_json::json!({
        "sku": "MOVEMENT-TEST-001",
        "name": "Movement Test Item",
        "description": null,
        "category": "test",
        "unit": "units",
        "reorder_point": 10,
        "reorder_quantity": 50
    });

    let item_hash: ActionHash = warehouse
        .call_zome_fn("inventory", "create_item", item_input)
        .await;

    // Record inbound movement
    let movement_input = serde_json::json!({
        "item_hash": item_hash,
        "movement_type": "Inbound",
        "quantity": 100,
        "from_location": null,
        "to_location": "warehouse-main",
        "reference": "PO-12345",
        "notes": "Initial stock receipt"
    });

    let movement_hash: ActionHash = warehouse
        .call_zome_fn("inventory", "record_movement", movement_input)
        .await;

    assert!(!movement_hash.as_ref().is_empty(), "Movement should be recorded");

    // Get item movements
    let movements: Vec<serde_json::Value> = warehouse
        .call_zome_fn("inventory", "get_item_movements", item_hash)
        .await;

    assert!(movements.len() >= 1, "Should have at least 1 movement");
}

// =============================================================================
// Logistics / Shipment Tests
// =============================================================================

/// Test: Create shipment and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_create_and_get_shipment() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let shipper = &agents[0];

    // Create a shipment
    let shipment_input = serde_json::json!({
        "tracking_number": "TRK-2024-001",
        "po_hash": null,
        "carrier": "FedEx",
        "origin": {
            "street": "123 Factory Lane",
            "city": "Manufacturing City",
            "country": "US"
        },
        "destination": {
            "street": "456 Retail Blvd",
            "city": "Commerce Town",
            "country": "US"
        },
        "items": [
            {
                "sku": "WIDGET-001",
                "quantity": 50,
                "description": "Industrial Widgets"
            }
        ],
        "estimated_delivery": null
    });

    let shipment_hash: ActionHash = shipper
        .call_zome_fn("logistics", "create_shipment", shipment_input)
        .await;

    assert!(!shipment_hash.as_ref().is_empty(), "Shipment should be created");

    // Retrieve the shipment
    let retrieved: Option<serde_json::Value> = shipper
        .call_zome_fn("logistics", "get_shipment", shipment_hash)
        .await;

    assert!(retrieved.is_some(), "Shipment should be retrievable");
    let shipment = retrieved.unwrap();
    assert_eq!(shipment["tracking_number"], "TRK-2024-001");
    assert_eq!(shipment["carrier"], "FedEx");
}

/// Test: Add tracking events to shipment.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_shipment_tracking_events() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        2,
    )
    .await;

    let shipper = &agents[0];
    let carrier = &agents[1];

    // Create a shipment
    let shipment_input = serde_json::json!({
        "tracking_number": "TRK-2024-002",
        "po_hash": null,
        "carrier": "UPS",
        "origin": {
            "street": "100 Warehouse Dr",
            "city": "Origin City",
            "country": "US"
        },
        "destination": {
            "street": "200 Customer Ave",
            "city": "Destination City",
            "country": "US"
        },
        "items": [
            {
                "sku": "GADGET-001",
                "quantity": 25,
                "description": "Electronic Gadgets"
            }
        ],
        "estimated_delivery": null
    });

    let shipment_hash: ActionHash = shipper
        .call_zome_fn("logistics", "create_shipment", shipment_input)
        .await;

    wait_for_dht_sync().await;

    // Carrier adds tracking events
    let pickup_event = serde_json::json!({
        "shipment_hash": shipment_hash,
        "status": "PickedUp",
        "location": "Origin City Hub",
        "description": "Package picked up by carrier"
    });

    let _: ActionHash = carrier
        .call_zome_fn("logistics", "add_tracking_event", pickup_event)
        .await;

    let transit_event = serde_json::json!({
        "shipment_hash": shipment_hash,
        "status": "InTransit",
        "location": "Regional Sort Facility",
        "description": "In transit to destination"
    });

    let _: ActionHash = carrier
        .call_zome_fn("logistics", "add_tracking_event", transit_event)
        .await;

    wait_for_dht_sync().await;

    // Get tracking events
    let events: Vec<serde_json::Value> = shipper
        .call_zome_fn("logistics", "get_tracking_events", shipment_hash)
        .await;

    assert!(events.len() >= 2, "Should have at least 2 tracking events");
}

// =============================================================================
// Claims / Provenance Tests
// =============================================================================

/// Test: Create provenance claim and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_create_and_get_claim() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let supplier = &agents[0];

    // Create a claim
    let claim_input = serde_json::json!({
        "item_id": "PRODUCT-001",
        "claim_type": "origin",
        "data": "{\"farm\": \"Organic Valley Farm\", \"harvest_date\": \"2024-01-15\"}",
        "issuer": format!("did:mycelix:{}", supplier.agent_pubkey),
        "previous_claim": null
    });

    let claim_record: Record = supplier
        .call_zome_fn("claims", "create_claim", claim_input)
        .await;

    let claim_hash = claim_record.action_hashed().hash.clone();
    assert!(!claim_hash.as_ref().is_empty(), "Claim should be created");

    // Retrieve the claim
    let retrieved: Option<Record> = supplier
        .call_zome_fn("claims", "get_claim", claim_hash)
        .await;

    assert!(retrieved.is_some(), "Claim should be retrievable");
}

/// Test: Build provenance chain for an item.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_provenance_chain() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        3,
    )
    .await;

    let farmer = &agents[0];
    let processor = &agents[1];
    let distributor = &agents[2];

    let item_id = "ORGANIC-COFFEE-001";

    // Farmer creates origin claim
    let origin_claim = serde_json::json!({
        "item_id": item_id,
        "claim_type": "origin",
        "data": "{\"farm\": \"Costa Rica Highland Farm\", \"altitude\": \"1500m\"}",
        "issuer": format!("did:mycelix:{}", farmer.agent_pubkey),
        "previous_claim": null
    });

    let origin_record: Record = farmer
        .call_zome_fn("claims", "create_claim", origin_claim)
        .await;

    let origin_hash = origin_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Processor creates processing claim linked to origin
    let processing_claim = serde_json::json!({
        "item_id": item_id,
        "claim_type": "processing",
        "data": "{\"roast_level\": \"medium\", \"processing_date\": \"2024-02-01\"}",
        "issuer": format!("did:mycelix:{}", processor.agent_pubkey),
        "previous_claim": origin_hash
    });

    let processing_record: Record = processor
        .call_zome_fn("claims", "create_claim", processing_claim)
        .await;

    let processing_hash = processing_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Distributor creates distribution claim linked to processing
    let distribution_claim = serde_json::json!({
        "item_id": item_id,
        "claim_type": "distribution",
        "data": "{\"warehouse\": \"Seattle Distribution Center\", \"lot_number\": \"LOT-2024-0215\"}",
        "issuer": format!("did:mycelix:{}", distributor.agent_pubkey),
        "previous_claim": processing_hash
    });

    let _: Record = distributor
        .call_zome_fn("claims", "create_claim", distribution_claim)
        .await;

    wait_for_dht_sync().await;
    wait_for_dht_sync().await; // Extra time for 3 conductors

    // Get full provenance chain
    let chain: Vec<Record> = farmer
        .call_zome_fn("claims", "get_item_provenance_chain", item_id.to_string())
        .await;

    assert!(chain.len() >= 3, "Provenance chain should have at least 3 claims");
}

/// Test: Multi-agent claim visibility via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_multi_agent_claim_visibility() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        2,
    )
    .await;

    let supplier = &agents[0];
    let verifier = &agents[1];

    // Supplier creates a claim
    let claim_input = serde_json::json!({
        "item_id": "CERTIFIED-ITEM-001",
        "claim_type": "certification",
        "data": "{\"certification\": \"ISO-9001\", \"valid_until\": \"2025-12-31\"}",
        "issuer": format!("did:mycelix:{}", supplier.agent_pubkey),
        "previous_claim": null
    });

    let claim_record: Record = supplier
        .call_zome_fn("claims", "create_claim", claim_input)
        .await;

    let claim_hash = claim_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Verifier retrieves and verifies the claim
    let retrieved: Option<Record> = verifier
        .call_zome_fn("claims", "get_claim", claim_hash.clone())
        .await;

    assert!(retrieved.is_some(), "Verifier should see claim via DHT");

    // Verify authenticity
    let is_authentic: bool = verifier
        .call_zome_fn("claims", "verify_claim_authenticity", claim_hash)
        .await;

    assert!(is_authentic, "Claim should be verified as authentic");
}

/// Test: Create and retrieve provider profile.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_provider_profile() {
    let agents = setup_test_agents(
        &DnaPaths::supplychain(),
        "mycelix-supplychain",
        1,
    )
    .await;

    let provider = &agents[0];

    // Create provider profile
    let profile_input = serde_json::json!({
        "provider_did": format!("did:mycelix:{}", provider.agent_pubkey),
        "name": "Sustainable Farms Co.",
        "certifications": ["USDA Organic", "Fair Trade", "Rainforest Alliance"]
    });

    let profile_record: Record = provider
        .call_zome_fn("claims", "create_provider_profile", profile_input)
        .await;

    let profile_hash = profile_record.action_hashed().hash.clone();
    assert!(!profile_hash.as_ref().is_empty(), "Profile should be created");

    // Retrieve provider profile
    let retrieved: Option<Record> = provider
        .call_zome_fn("claims", "get_provider_profile", profile_hash)
        .await;

    assert!(retrieved.is_some(), "Provider profile should be retrievable");
}
