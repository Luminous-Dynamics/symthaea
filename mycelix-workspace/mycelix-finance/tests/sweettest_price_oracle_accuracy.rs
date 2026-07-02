// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Price Oracle Accuracy-Weighted Consensus Sweettest
//!
//! Verifies the full accuracy-weighted consensus flow against a real conductor:
//! 1. Two agents report prices for the same item
//! 2. Consensus is computed with accuracy-weighted median
//! 3. Reporter accuracy scores are updated
//! 4. Signal integrity reflects average reporter accuracy
//!
//! ## Running
//! ```bash
//! cd mycelix-finance/tests
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
    PathBuf::from(manifest)
        .parent()
        .unwrap()
        .join("dna/mycelix_finance.dna")
}

// ============================================================================
// Tests
// ============================================================================

/// Two agents report similar prices → consensus should be close to both.
/// Both agents should retain high accuracy scores.
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires conductor + packed DNA
async fn test_two_accurate_reporters_consensus() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(&dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("alice-app", &[dna.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (bob,) = conductor
        .setup_app("bob-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Alice reports bread at 0.15 TEND
    let _: holochain::prelude::Record = conductor
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

    // Bob reports bread at 0.16 TEND
    let _: holochain::prelude::Record = conductor
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

    // Compute consensus
    let consensus: ConsensusResult = conductor
        .call(
            &alice.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "bread_750g".into(),
            },
        )
        .await;

    assert!(
        consensus.median_price >= 0.14 && consensus.median_price <= 0.17,
        "Consensus should be between reports: got {}",
        consensus.median_price
    );
    assert_eq!(consensus.reporter_count, 2);
    assert!(
        consensus.signal_integrity > 0.9,
        "Signal integrity should be high: got {}",
        consensus.signal_integrity
    );
}

/// One accurate reporter + one wildly inaccurate reporter.
/// After multiple consensus rounds, the accurate reporter should gain higher accuracy.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_accuracy_divergence_over_rounds() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(&dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("alice-app", &[dna.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (bob,) = conductor
        .setup_app("bob-app", &[dna.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (carol,) = conductor
        .setup_app("carol-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Run 3 consensus rounds with Alice+Carol accurate, Bob wildly off
    for round in 0..3 {
        let item = format!("diesel_round_{round}");

        // Alice: accurate
        let _: holochain::prelude::Record = conductor
            .call(
                &alice.zome("price_oracle"),
                "report_price",
                ReportPriceInput {
                    item: item.clone(),
                    price_tend: 0.50,
                    evidence: "Shell garage".into(),
                },
            )
            .await;

        // Carol: also accurate
        let _: holochain::prelude::Record = conductor
            .call(
                &carol.zome("price_oracle"),
                "report_price",
                ReportPriceInput {
                    item: item.clone(),
                    price_tend: 0.52,
                    evidence: "Engen garage".into(),
                },
            )
            .await;

        // Bob: wildly off
        let _: holochain::prelude::Record = conductor
            .call(
                &bob.zome("price_oracle"),
                "report_price",
                ReportPriceInput {
                    item: item.clone(),
                    price_tend: 5.00,
                    evidence: "Guess".into(),
                },
            )
            .await;

        // Trigger consensus (updates accuracy scores)
        let _: ConsensusResult = conductor
            .call(
                &alice.zome("price_oracle"),
                "get_consensus_price",
                GetConsensusInput { item },
            )
            .await;
    }

    // Check accuracy scores
    let alice_did = format!("did:holo:{}", alice.agent_pubkey());
    let bob_did = format!("did:holo:{}", bob.agent_pubkey());

    let alice_acc: ReporterAccuracyResult = conductor
        .call(
            &alice.zome("price_oracle"),
            "get_reporter_accuracy",
            alice_did,
        )
        .await;

    let bob_acc: ReporterAccuracyResult = conductor
        .call(
            &alice.zome("price_oracle"),
            "get_reporter_accuracy",
            bob_did,
        )
        .await;

    assert!(
        alice_acc.accuracy_score > bob_acc.accuracy_score,
        "Alice ({}) should be more accurate than Bob ({})",
        alice_acc.accuracy_score,
        bob_acc.accuracy_score
    );
    // Report count may vary by ±1 due to DHT eventual consistency
    assert!(
        alice_acc.report_count >= 2,
        "alice report count: {}",
        alice_acc.report_count
    );
    assert!(
        bob_acc.report_count >= 2,
        "bob report count: {}",
        bob_acc.report_count
    );
}

/// Single reporter should fail — need at least 2 for consensus.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_single_reporter_fails() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(&dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("alice-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    let _: holochain::prelude::Record = conductor
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

    // Consensus should fail with only 1 reporter
    let result: Result<ConsensusResult, _> = conductor
        .call_fallible(
            &alice.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "eggs_6".into(),
            },
        )
        .await;

    assert!(
        result.is_err(),
        "Consensus should fail with only 1 reporter"
    );
}
