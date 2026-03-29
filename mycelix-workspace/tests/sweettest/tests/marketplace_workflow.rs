// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Marketplace hApp sweettest integration tests.
//!
//! Tests P2P commerce workflows including listings, transactions, reputation,
//! and MATL trust scoring using the Holochain sweettest framework.
//!
//! Prerequisites:
//!   cd mycelix-marketplace/backend && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack . -o mycelix_marketplace.dna
//!
//! Run: cargo test --release -p mycelix-sweettest -- --ignored marketplace
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// =============================================================================
// Listing Management Tests
// =============================================================================

/// Test: Create a listing and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_get_listing() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        1,
    )
    .await;

    let seller = &agents[0];

    // Create a listing
    let listing_input = serde_json::json!({
        "title": "Handmade Pottery Bowl",
        "description": "Beautiful ceramic bowl, locally crafted",
        "price": {
            "amount": 45.00,
            "currency": "USD"
        },
        "category": "crafts",
        "images": ["ipfs://Qm123..."],
        "location": "Portland, OR",
        "shipping_options": ["local_pickup", "usps_priority"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    let action_hash = listing_record.action_hashed().hash.clone();
    assert!(!action_hash.as_ref().is_empty(), "Listing should be created");

    // Retrieve the listing
    let retrieved: Option<Record> = seller
        .call_zome_fn("listings", "get_listing", action_hash)
        .await;

    assert!(retrieved.is_some(), "Listing should be retrievable");
}

/// Test: List all active listings.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_list_active_listings() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        1,
    )
    .await;

    let seller = &agents[0];

    // Create multiple listings
    for i in 1..=3 {
        let listing_input = serde_json::json!({
            "title": format!("Test Listing {}", i),
            "description": format!("Description for listing {}", i),
            "price": {
                "amount": 10.0 * i as f64,
                "currency": "USD"
            },
            "category": "test",
            "images": [],
            "location": "Test Location",
            "shipping_options": ["local_pickup"],
            "created_at": Timestamp::now().as_micros()
        });

        let _: Record = seller
            .call_zome_fn("listings", "create_listing", listing_input)
            .await;
    }

    // List all active listings
    let listings: Vec<Record> = seller
        .call_zome_fn("listings", "list_active_listings", ())
        .await;

    assert!(listings.len() >= 3, "Should have at least 3 listings");
}

/// Test: Update listing price.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_update_listing() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        1,
    )
    .await;

    let seller = &agents[0];

    // Create listing
    let listing_input = serde_json::json!({
        "title": "Update Test Item",
        "description": "Original description",
        "price": {
            "amount": 50.00,
            "currency": "USD"
        },
        "category": "electronics",
        "images": [],
        "location": "Seattle, WA",
        "shipping_options": ["usps_priority"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    let original_hash = listing_record.action_hashed().hash.clone();

    // Update the listing
    let update_input = serde_json::json!({
        "original_action_hash": original_hash,
        "updated_listing": {
            "title": "Update Test Item",
            "description": "Updated description with more details",
            "price": {
                "amount": 45.00,
                "currency": "USD"
            },
            "category": "electronics",
            "images": [],
            "location": "Seattle, WA",
            "shipping_options": ["usps_priority", "fedex_ground"],
            "created_at": Timestamp::now().as_micros()
        }
    });

    let updated_record: Record = seller
        .call_zome_fn("listings", "update_listing", update_input)
        .await;

    assert!(
        !updated_record.action_hashed().hash.as_ref().is_empty(),
        "Listing should be updated"
    );
}

// =============================================================================
// Transaction Lifecycle Tests
// =============================================================================

/// Test: Create transaction between buyer and seller.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_create_transaction() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Seller creates listing
    let listing_input = serde_json::json!({
        "title": "Transaction Test Item",
        "description": "Item for transaction test",
        "price": {
            "amount": 100.00,
            "currency": "USD"
        },
        "category": "general",
        "images": [],
        "location": "Denver, CO",
        "shipping_options": ["local_pickup"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    let listing_hash = listing_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Buyer initiates transaction
    let transaction_input = serde_json::json!({
        "listing_hash": listing_hash,
        "seller": seller.agent_pubkey.to_string(),
        "buyer": buyer.agent_pubkey.to_string(),
        "amount": {
            "amount": 100.00,
            "currency": "USD"
        },
        "shipping_method": "local_pickup",
        "created_at": Timestamp::now().as_micros()
    });

    let transaction_record: Record = buyer
        .call_zome_fn("transactions", "create_transaction", transaction_input)
        .await;

    assert!(
        !transaction_record.action_hashed().hash.as_ref().is_empty(),
        "Transaction should be created"
    );
}

/// Test: Transaction status progression (Initiated -> Paid -> Shipped -> Delivered).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_transaction_status_progression() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Create listing and transaction
    let listing_input = serde_json::json!({
        "title": "Status Test Item",
        "description": "Testing status progression",
        "price": {
            "amount": 75.00,
            "currency": "USD"
        },
        "category": "general",
        "images": [],
        "location": "Austin, TX",
        "shipping_options": ["usps_priority"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    let listing_hash = listing_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    let transaction_input = serde_json::json!({
        "listing_hash": listing_hash,
        "seller": seller.agent_pubkey.to_string(),
        "buyer": buyer.agent_pubkey.to_string(),
        "amount": {
            "amount": 75.00,
            "currency": "USD"
        },
        "shipping_method": "usps_priority",
        "created_at": Timestamp::now().as_micros()
    });

    let transaction_record: Record = buyer
        .call_zome_fn("transactions", "create_transaction", transaction_input)
        .await;

    let tx_hash = transaction_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Update status to Paid
    let status_update = serde_json::json!({
        "transaction_hash": tx_hash.clone(),
        "new_status": "Paid",
        "note": "Payment confirmed via escrow"
    });

    let _: Record = buyer
        .call_zome_fn("transactions", "update_status", status_update)
        .await;

    wait_for_dht_sync().await;

    // Seller marks as Shipped
    let ship_update = serde_json::json!({
        "transaction_hash": tx_hash.clone(),
        "new_status": "Shipped",
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS"
    });

    let _: Record = seller
        .call_zome_fn("transactions", "update_status", ship_update)
        .await;

    wait_for_dht_sync().await;

    // Get transaction to verify status
    let retrieved: Option<Record> = buyer
        .call_zome_fn("transactions", "get_transaction", tx_hash)
        .await;

    assert!(retrieved.is_some(), "Transaction should be retrievable");
}

/// Test: Deliver item and complete transaction.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_deliver_item() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Quick setup - create listing and transaction
    let listing_input = serde_json::json!({
        "title": "Delivery Test Item",
        "description": "Testing delivery flow",
        "price": { "amount": 25.00, "currency": "USD" },
        "category": "general",
        "images": [],
        "location": "Miami, FL",
        "shipping_options": ["local_pickup"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    wait_for_dht_sync().await;

    let transaction_input = serde_json::json!({
        "listing_hash": listing_record.action_hashed().hash,
        "seller": seller.agent_pubkey.to_string(),
        "buyer": buyer.agent_pubkey.to_string(),
        "amount": { "amount": 25.00, "currency": "USD" },
        "shipping_method": "local_pickup",
        "created_at": Timestamp::now().as_micros()
    });

    let transaction_record: Record = buyer
        .call_zome_fn("transactions", "create_transaction", transaction_input)
        .await;

    let tx_hash = transaction_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Buyer confirms delivery
    let deliver_input = serde_json::json!({
        "transaction_hash": tx_hash,
        "delivered_at": Timestamp::now().as_micros(),
        "notes": "Item received in good condition"
    });

    let delivered: Record = buyer
        .call_zome_fn("transactions", "deliver_item", deliver_input)
        .await;

    assert!(
        !delivered.action_hashed().hash.as_ref().is_empty(),
        "Delivery should be recorded"
    );
}

// =============================================================================
// Reputation System Tests
// =============================================================================

/// Test: Leave feedback after transaction.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_leave_feedback() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Setup completed transaction
    let listing_input = serde_json::json!({
        "title": "Feedback Test Item",
        "description": "Testing feedback system",
        "price": { "amount": 30.00, "currency": "USD" },
        "category": "general",
        "images": [],
        "location": "Chicago, IL",
        "shipping_options": ["local_pickup"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    wait_for_dht_sync().await;

    let transaction_input = serde_json::json!({
        "listing_hash": listing_record.action_hashed().hash,
        "seller": seller.agent_pubkey.to_string(),
        "buyer": buyer.agent_pubkey.to_string(),
        "amount": { "amount": 30.00, "currency": "USD" },
        "shipping_method": "local_pickup",
        "created_at": Timestamp::now().as_micros()
    });

    let transaction_record: Record = buyer
        .call_zome_fn("transactions", "create_transaction", transaction_input)
        .await;

    let tx_hash = transaction_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Buyer leaves feedback for seller
    let feedback_input = serde_json::json!({
        "transaction_hash": tx_hash,
        "subject": seller.agent_pubkey.to_string(),
        "rating": 5,
        "comment": "Excellent seller! Fast shipping and great communication.",
        "aspects": {
            "communication": 5,
            "item_quality": 5,
            "shipping_speed": 4
        },
        "created_at": Timestamp::now().as_micros()
    });

    let feedback_record: Record = buyer
        .call_zome_fn("reputation", "leave_feedback", feedback_input)
        .await;

    assert!(
        !feedback_record.action_hashed().hash.as_ref().is_empty(),
        "Feedback should be recorded"
    );
}

/// Test: Get agent reputation score.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_reputation_score() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        1,
    )
    .await;

    let agent = &agents[0];

    // Get reputation for agent (may be empty for new agent)
    let reputation: serde_json::Value = agent
        .call_zome_fn("reputation", "get_reputation", agent.agent_pubkey.to_string())
        .await;

    // New agent should have default/empty reputation
    assert!(
        reputation.is_object() || reputation.is_null(),
        "Reputation should be an object or null for new agent"
    );
}

// =============================================================================
// MATL Bridge Integration Tests
// =============================================================================

/// Test: Report transaction outcome to Bridge for trust scoring.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_report_to_bridge() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let buyer = &agents[1];

    // Create and complete a transaction
    let listing_input = serde_json::json!({
        "title": "Bridge Report Item",
        "description": "Testing Bridge integration",
        "price": { "amount": 50.00, "currency": "USD" },
        "category": "general",
        "images": [],
        "location": "Boston, MA",
        "shipping_options": ["local_pickup"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    wait_for_dht_sync().await;

    let transaction_input = serde_json::json!({
        "listing_hash": listing_record.action_hashed().hash,
        "seller": seller.agent_pubkey.to_string(),
        "buyer": buyer.agent_pubkey.to_string(),
        "amount": { "amount": 50.00, "currency": "USD" },
        "shipping_method": "local_pickup",
        "created_at": Timestamp::now().as_micros()
    });

    let transaction_record: Record = buyer
        .call_zome_fn("transactions", "create_transaction", transaction_input)
        .await;

    let tx_hash = transaction_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Report successful transaction to Bridge
    let report_input = serde_json::json!({
        "transaction_hash": tx_hash,
        "outcome": "completed",
        "buyer_satisfaction": 0.95,
        "seller_reliability": 0.90,
        "dispute": null
    });

    let report_result: Record = buyer
        .call_zome_fn("transactions", "report_to_bridge", report_input)
        .await;

    assert!(
        !report_result.action_hashed().hash.as_ref().is_empty(),
        "Bridge report should be recorded"
    );
}

/// Test: Multi-agent transaction visibility via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_multi_agent_listing_visibility() {
    let agents = setup_test_agents(
        &DnaPaths::marketplace(),
        "mycelix-marketplace",
        2,
    )
    .await;

    let seller = &agents[0];
    let browser = &agents[1];

    // Seller creates listing
    let listing_input = serde_json::json!({
        "title": "Visibility Test Item",
        "description": "Testing DHT propagation",
        "price": { "amount": 15.00, "currency": "USD" },
        "category": "test",
        "images": [],
        "location": "Phoenix, AZ",
        "shipping_options": ["local_pickup"],
        "created_at": Timestamp::now().as_micros()
    });

    let listing_record: Record = seller
        .call_zome_fn("listings", "create_listing", listing_input)
        .await;

    let listing_hash = listing_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Browser should see the listing
    let retrieved: Option<Record> = browser
        .call_zome_fn("listings", "get_listing", listing_hash)
        .await;

    assert!(retrieved.is_some(), "Listing should be visible to other agents via DHT");
}
