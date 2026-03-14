//! # Price Oracle Accuracy-Weighted Consensus Sweettest
//!
//! Verifies the full accuracy-weighted consensus flow:
//! 1. Multiple agents report prices for the same item
//! 2. Consensus is computed with accuracy-weighted median
//! 3. Reporter accuracy scores are updated based on how close
//!    their reports were to the consensus
//! 4. Signal integrity reflects average reporter accuracy
//!
//! ## Running
//! ```bash
//! cd mycelix-finance
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_price_oracle_accuracy -- --ignored --test-threads=1
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types for price oracle zome calls
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportPriceInput {
    pub item: String,
    pub price_tend: f64,
    pub evidence: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetConsensusInput {
    pub item: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusResult {
    pub item: String,
    pub median_price: f64,
    pub reporter_count: u32,
    pub std_dev: f64,
    pub signal_integrity: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReporterAccuracyResult {
    pub reporter_did: String,
    pub accuracy_score: f64,
    pub report_count: u32,
}

// ============================================================================
// Helper: Get DNA path
// ============================================================================

fn dna_path() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest).join("dna/finance.dna")
}

// ============================================================================
// Tests
// ============================================================================

/// Two agents report similar prices → consensus should be close to both.
/// Both agents should retain high accuracy scores.
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires conductor + packed DNA
async fn test_two_accurate_reporters_consensus() {
    let (conductors, _app, cells) = setup_two_agents().await;

    let alice = &cells[0];
    let bob = &cells[1];

    // Alice reports bread at 0.15 TEND
    let _: holochain::prelude::Record = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.15,
                evidence: "Pick n Pay, 2026-03-14".into(),
            },
        )
        .await;

    // Bob reports bread at 0.16 TEND (close to Alice)
    let _: holochain::prelude::Record = conductors[1]
        .call(
            &bob.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.16,
                evidence: "Shoprite, 2026-03-14".into(),
            },
        )
        .await;

    // Wait for DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Compute consensus (either agent can call this)
    let consensus: ConsensusResult = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "bread_750g".into(),
            },
        )
        .await;

    // Consensus should be between 0.15 and 0.16
    assert!(consensus.median_price >= 0.15 && consensus.median_price <= 0.16,
        "Consensus should be between reports: got {}", consensus.median_price);
    assert_eq!(consensus.reporter_count, 2);
    // Signal integrity should be high (both reporters accurate)
    assert!(consensus.signal_integrity > 0.9,
        "Signal integrity should be high: got {}", consensus.signal_integrity);
}

/// One accurate reporter + one wildly inaccurate reporter.
/// After consensus, the accurate reporter should have higher accuracy
/// than the inaccurate one.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_accuracy_divergence() {
    let (conductors, _app, cells) = setup_two_agents().await;

    let alice = &cells[0];
    let bob = &cells[1];

    // Alice: accurate (0.50 TEND for diesel)
    let _: holochain::prelude::Record = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "diesel_1l".into(),
                price_tend: 0.50,
                evidence: "Shell garage".into(),
            },
        )
        .await;

    // Bob: wildly inaccurate (5.00 TEND for diesel — 10x off)
    let _: holochain::prelude::Record = conductors[1]
        .call(
            &bob.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "diesel_1l".into(),
                price_tend: 5.00,
                evidence: "Guess".into(),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Compute consensus
    let consensus: ConsensusResult = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "diesel_1l".into(),
            },
        )
        .await;

    // With 2 reporters, no trimming occurs.
    // Weighted median with both at accuracy 1.0 (initial) → midpoint or one side
    assert!(consensus.reporter_count == 2);

    // Check accuracy scores — Alice's DID should have higher accuracy
    let alice_did = format!("did:holo:{}", alice.agent_pubkey());
    let bob_did = format!("did:holo:{}", bob.agent_pubkey());

    let alice_acc: ReporterAccuracyResult = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "get_reporter_accuracy",
            alice_did,
        )
        .await;

    let bob_acc: ReporterAccuracyResult = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "get_reporter_accuracy",
            bob_did,
        )
        .await;

    // Both should have report_count = 1
    assert_eq!(alice_acc.report_count, 1);
    assert_eq!(bob_acc.report_count, 1);

    // Alice should be more accurate than Bob
    assert!(alice_acc.accuracy_score > bob_acc.accuracy_score,
        "Alice ({}) should be more accurate than Bob ({})",
        alice_acc.accuracy_score, bob_acc.accuracy_score);
}

/// Single reporter should fail — need at least 2 for consensus.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_single_reporter_fails() {
    let (conductors, _app, cells) = setup_two_agents().await;

    let alice = &cells[0];

    let _: holochain::prelude::Record = conductors[0]
        .call(
            &alice.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "eggs_6".into(),
                price_tend: 0.25,
                evidence: "Local shop".into(),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Consensus should fail with only 1 reporter
    let result: Result<ConsensusResult, _> = conductors[0]
        .call_fallible(
            &alice.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "eggs_6".into(),
            },
        )
        .await;

    assert!(result.is_err(), "Consensus should fail with only 1 reporter");
}

// ============================================================================
// Setup helpers
// ============================================================================

async fn setup_two_agents() -> (
    SweetConductorBatch,
    SweetApp,
    Vec<SweetCell>,
) {
    let dna = SweetDnaFile::from_bundle(dna_path()).await.unwrap();

    let mut conductors = SweetConductorBatch::from_standard_config(2).await;

    let apps = conductors
        .setup_app("mycelix-finance", &[dna])
        .await
        .unwrap();

    let cells: Vec<SweetCell> = apps
        .into_inner()
        .into_iter()
        .map(|a| a.into_cells().into_iter().next().unwrap())
        .collect();

    conductors.exchange_peer_info().await;

    // Return a dummy SweetApp (we don't need it)
    let dummy_app = conductors[0]
        .setup_app("dummy", &[])
        .await
        .unwrap()
        .into_inner()
        .into_iter()
        .next()
        .unwrap();

    (conductors, dummy_app, cells)
}
