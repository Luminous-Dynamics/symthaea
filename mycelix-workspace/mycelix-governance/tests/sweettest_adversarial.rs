// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Governance — Adversarial Sweettest Suite
//!
//! Proves that governance functions fail-closed when dependent zomes or
//! clusters are unavailable. Uses single-DNA deployment (no identity/personal
//! roles) to simulate cross-cluster unavailability.
//!
//! ## Scenarios
//!
//! 1. Cross-cluster identity calls fail when identity role is absent
//! 2. Consciousness gate blocks operations when credentials unavailable
//! 3. Bridge health is maintained after gate failures
//! 4. Non-existent zome function calls are rejected
//!
//! ## Running
//!
//! ```bash
//! cd mycelix-governance
//! nix develop
//! hc dna pack dna/
//! cd tests
//! cargo test --release --test sweettest_adversarial -- --ignored --test-threads=1
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Setup helpers
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_governance_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load governance DNA bundle — run `hc dna pack dna/` first")
}

// ============================================================================
// Test 1: Consciousness gate returns failed when no snapshot exists
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_consciousness_gate_fails_without_snapshot() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;

    // Single-DNA setup — NO identity role, no consciousness snapshots recorded
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // verify_consciousness_gate should return { passed: false } because
    // no consciousness snapshot has been recorded for this agent
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("governance_bridge"),
            "verify_consciousness_gate",
            serde_json::json!({
                "action_type": "Constitutional",
                "action_id": "adversarial-test-1"
            }),
        )
        .await;

    match result {
        Ok(val) => {
            let passed = val
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            assert!(
                !passed,
                "Consciousness gate MUST NOT pass without a recorded snapshot"
            );
        }
        Err(_) => {
            // Error is also acceptable — fail-closed
        }
    }
}

// ============================================================================
// Test 2: Non-existent function call is rejected
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_non_existent_function_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("proposals"),
            "this_function_does_not_exist_anywhere",
            serde_json::json!({}),
        )
        .await;

    assert!(
        result.is_err(),
        "Calling a non-existent function MUST be rejected"
    );
}

// ============================================================================
// Test 3: Consciousness thresholds are queryable (smoke test)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_consciousness_thresholds_queryable() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // get_consciousness_thresholds should return valid thresholds
    // even without any recorded snapshots — it's a config query
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("governance_bridge"),
            "get_consciousness_thresholds",
            (),
        )
        .await;

    assert!(
        result.is_ok(),
        "Consciousness thresholds should be queryable without conductor state"
    );

    let thresholds = result.unwrap();
    // Verify basic structure exists
    assert!(
        thresholds.get("basic").is_some()
            || thresholds.get("consciousness_gate_basic").is_some()
            || thresholds.is_object(),
        "Thresholds should contain governance configuration"
    );
}

// ============================================================================
// Test 4: Consciousness gate blocks without credentials
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_consciousness_gate_blocks_without_credentials() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Try to verify consciousness gate — should fail because no
    // consciousness credential has been issued (identity cluster absent)
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("governance_bridge"),
            "verify_consciousness_gate",
            serde_json::json!({
                "action_type": "Voting",
                "action_id": "test-proposal-1"
            }),
        )
        .await;

    // The gate should either:
    // (a) return { passed: false } because no credential exists, OR
    // (b) return an error because the credential lookup failed
    // Both are acceptable fail-closed behaviors.
    match result {
        Err(_) => {
            // Error propagated — fail-closed ✅
        }
        Ok(val) => {
            // Check that passed = false
            let passed = val
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(true); // default true to catch bugs
            assert!(
                !passed,
                "Consciousness gate MUST NOT pass without credentials"
            );
        }
    }
}

// ============================================================================
// Test 5: Multiple gate failures don't corrupt bridge state
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_repeated_gate_failures_dont_corrupt_state() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Trigger multiple gate failures (no snapshots exist)
    for i in 0..5 {
        let _ = conductor
            .call_fallible::<_, serde_json::Value>(
                &alice.zome("governance_bridge"),
                "verify_consciousness_gate",
                serde_json::json!({
                    "action_type": "Voting",
                    "action_id": format!("adversarial-repeat-{}", i)
                }),
            )
            .await;
    }

    // After repeated failures, the bridge should still respond normally
    // to a valid query (consciousness thresholds are static config)
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("governance_bridge"),
            "get_consciousness_thresholds",
            (),
        )
        .await;

    assert!(
        result.is_ok(),
        "Bridge should remain functional after repeated gate failures"
    );
}
