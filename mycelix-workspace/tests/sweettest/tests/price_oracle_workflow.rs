// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Price Oracle Sweettest Integration Tests
//!
//! Tests the on-chain price reporting, consensus computation, accuracy
//! tracking, and volatility-triggered TEND escalation flow.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build finance WASM + pack DNA
//! cd mycelix-finance && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-finance/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test price_oracle_workflow -- --ignored --test-threads=1
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — must match coordinator struct layout (NOT linking the crate
// to avoid __num_entry_types symbol conflicts in sweettest)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReportPriceInput {
    item: String,
    price_tend: f64,
    evidence: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GetConsensusInput {
    item: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsensusResult {
    item: String,
    median_price: f64,
    reporter_count: u32,
    std_dev: f64,
    signal_integrity: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReporterAccuracyResult {
    reporter_did: String,
    accuracy_score: f64,
    report_count: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GetTopReportersInput {
    item: String,
    limit: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DefineBasketInput {
    name: String,
    items: Vec<BasketItemInput>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BasketItemInput {
    item: String,
    weight: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GetBasketIndexInput {
    basket_name: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BasketIndexResult {
    basket_name: String,
    index: f64,
    item_prices: Vec<ItemPriceResult>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ItemPriceResult {
    item: String,
    price: f64,
    weight: f64,
    weighted_price: f64,
}

// ============================================================================
// Test 1: Two agents report prices → consensus = accuracy-weighted median
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_price_report_and_consensus() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Alice reports bread at 0.15 TEND
    let _: Record = alice
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.15,
                evidence: "Local Market, 2026-03-14".into(),
            },
        )
        .await;

    // Wait for DHT propagation
    wait_for_dht_sync().await;

    // Bob reports bread at 0.17 TEND
    let _: Record = bob
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.17,
                evidence: "Community Store, 2026-03-14".into(),
            },
        )
        .await;

    wait_for_dht_sync().await;

    // Alice computes consensus
    let consensus: ConsensusResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_consensus_price",
            GetConsensusInput {
                item: "bread_750g".into(),
            },
        )
        .await;

    assert_eq!(consensus.item, "bread_750g");
    assert_eq!(consensus.reporter_count, 2);
    // With 2 reporters and equal accuracy (both start at 1.0),
    // weighted median of [0.15, 0.17] should be the midpoint
    assert!(
        consensus.median_price >= 0.15 && consensus.median_price <= 0.17,
        "Expected median in [0.15, 0.17], got {}",
        consensus.median_price
    );
    assert!(
        consensus.signal_integrity > 0.0,
        "Signal integrity should be positive"
    );
}

// ============================================================================
// Test 2: Consensus requires at least 2 reporters
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_consensus_requires_min_reporters() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 1).await;
    let alice = &agents[0];

    // Alice reports diesel
    let _: Record = alice
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "diesel_1l".into(),
                price_tend: 1.5,
                evidence: "Fuel Station".into(),
            },
        )
        .await;

    // Consensus should fail with only 1 reporter
    let result: Result<ConsensusResult, _> = alice
        .call_zome_fn_fallible(
            "price_oracle",
            "get_consensus_price",
            GetConsensusInput {
                item: "diesel_1l".into(),
            },
        )
        .await;

    assert!(
        result.is_err(),
        "Consensus should fail with only 1 reporter"
    );
}

// ============================================================================
// Test 3: Accuracy tracking — reporter scores update after consensus
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_accuracy_updates_after_consensus() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 3).await;
    let alice = &agents[0];
    let bob = &agents[1];
    let charlie = &agents[2];

    let alice_did = format!("did:holo:{}", alice.agent_pubkey);
    let bob_did = format!("did:holo:{}", bob.agent_pubkey);
    let charlie_did = format!("did:holo:{}", charlie.agent_pubkey);

    // Alice: 0.15 (accurate)
    let _: Record = alice
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "eggs_dozen".into(),
                price_tend: 0.50,
                evidence: "Checkers".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Bob: 0.55 (close to Alice)
    let _: Record = bob
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "eggs_dozen".into(),
                price_tend: 0.55,
                evidence: "Pick n Pay".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Charlie: 2.00 (way off — should lose accuracy)
    let _: Record = charlie
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "eggs_dozen".into(),
                price_tend: 2.00,
                evidence: "Organic market, premium".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Compute consensus — this should update accuracy scores
    let _consensus: ConsensusResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_consensus_price",
            GetConsensusInput {
                item: "eggs_dozen".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Check accuracy scores
    let alice_acc: ReporterAccuracyResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_reporter_accuracy",
            alice_did,
        )
        .await;

    let charlie_acc: ReporterAccuracyResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_reporter_accuracy",
            charlie_did,
        )
        .await;

    // Alice should have higher accuracy than Charlie
    assert!(
        alice_acc.accuracy_score > charlie_acc.accuracy_score,
        "Accurate reporter (Alice: {}) should score higher than inaccurate (Charlie: {})",
        alice_acc.accuracy_score,
        charlie_acc.accuracy_score
    );
}

// ============================================================================
// Test 4: Top reporters query
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_top_reporters_ranked_by_accuracy() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Both report maize
    let _: Record = alice
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "maize_2kg".into(),
                price_tend: 0.30,
                evidence: "Local market".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    let _: Record = bob
        .call_zome_fn(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "maize_2kg".into(),
                price_tend: 0.32,
                evidence: "Shoprite".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Compute consensus to establish accuracy
    let _: ConsensusResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_consensus_price",
            GetConsensusInput {
                item: "maize_2kg".into(),
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Get top reporters
    let top: Vec<ReporterAccuracyResult> = alice
        .call_zome_fn(
            "price_oracle",
            "get_top_reporters",
            GetTopReportersInput {
                item: "maize_2kg".into(),
                limit: 10,
            },
        )
        .await;

    assert_eq!(top.len(), 2, "Should have 2 reporters");
    // First reporter should have accuracy >= second
    assert!(
        top[0].accuracy_score >= top[1].accuracy_score,
        "Reporters should be sorted by accuracy descending"
    );
}

// ============================================================================
// Test 5: Basket definition and index computation
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_basket_definition_and_index() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Both report bread and diesel prices
    for (item, price_a, price_b) in [
        ("bread_750g", 0.15, 0.17),
        ("diesel_1l", 1.50, 1.60),
    ] {
        let _: Record = alice
            .call_zome_fn(
                "price_oracle",
                "report_price",
                ReportPriceInput {
                    item: item.into(),
                    price_tend: price_a,
                    evidence: "Alice source".into(),
                },
            )
            .await;
        wait_for_dht_sync().await;

        let _: Record = bob
            .call_zome_fn(
                "price_oracle",
                "report_price",
                ReportPriceInput {
                    item: item.into(),
                    price_tend: price_b,
                    evidence: "Bob source".into(),
                },
            )
            .await;
        wait_for_dht_sync().await;

        // Compute consensus for each item
        let _: ConsensusResult = alice
            .call_zome_fn(
                "price_oracle",
                "get_consensus_price",
                GetConsensusInput { item: item.into() },
            )
            .await;
        wait_for_dht_sync().await;
    }

    // Define a basket (50% bread, 50% diesel)
    let _: Record = alice
        .call_zome_fn(
            "price_oracle",
            "define_basket",
            DefineBasketInput {
                name: "Community Essentials".into(),
                items: vec![
                    BasketItemInput {
                        item: "bread_750g".into(),
                        weight: 0.5,
                    },
                    BasketItemInput {
                        item: "diesel_1l".into(),
                        weight: 0.5,
                    },
                ],
            },
        )
        .await;
    wait_for_dht_sync().await;

    // Compute basket index
    let index: BasketIndexResult = alice
        .call_zome_fn(
            "price_oracle",
            "get_basket_index",
            GetBasketIndexInput {
                basket_name: "Community Essentials".into(),
            },
        )
        .await;

    assert_eq!(index.basket_name, "Community Essentials");
    assert_eq!(index.item_prices.len(), 2);
    assert!(
        index.index > 0.0,
        "Basket index should be positive, got {}",
        index.index
    );

    // Index should be ~(0.16*0.5 + 1.55*0.5) = ~0.855
    // (rough, since weighted median may not be exact average)
    assert!(
        index.index > 0.5 && index.index < 1.5,
        "Basket index should be reasonable, got {}",
        index.index
    );
}

// ============================================================================
// Test 6: Invalid price report rejected
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_invalid_price_rejected() {
    let agents = setup_test_agents(&DnaPaths::finance(), "mycelix-finance", 1).await;
    let alice = &agents[0];

    // Negative price should fail
    let result: Result<Record, _> = alice
        .call_zome_fn_fallible(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: -1.0,
                evidence: "test".into(),
            },
        )
        .await;

    assert!(result.is_err(), "Negative price should be rejected");

    // Zero price should fail
    let result2: Result<Record, _> = alice
        .call_zome_fn_fallible(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.0,
                evidence: "test".into(),
            },
        )
        .await;

    assert!(result2.is_err(), "Zero price should be rejected");

    // Empty item name should fail
    let result3: Result<Record, _> = alice
        .call_zome_fn_fallible(
            "price_oracle",
            "report_price",
            ReportPriceInput {
                item: "".into(),
                price_tend: 1.0,
                evidence: "test".into(),
            },
        )
        .await;

    assert!(result3.is_err(), "Empty item name should be rejected");
}
