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

// Mirror types to avoid WASM symbol collisions
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: u32,
    total_queries: u32,
    domains: Vec<String>,
}

// ============================================================================
// Test 1: Cross-cluster identity call fails when role is absent
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_cross_cluster_identity_fails_when_role_absent() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;

    // Single-DNA setup — NO identity or personal roles available
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Try to dispatch a cross-cluster call to identity (OtherRole)
    // This should fail because no identity DNA is installed
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("governance_bridge"),
            "dispatch_cross_cluster",
            serde_json::json!({
                "target_role": "identity",
                "zome_name": "did_registry",
                "fn_name": "verify_did",
                "payload": "did:mycelix:test"
            }),
        )
        .await;

    assert!(
        result.is_err(),
        "Cross-cluster call to absent identity role MUST fail (fail-closed)"
    );

    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("OtherRole")
            || err.contains("identity")
            || err.contains("role")
            || err.contains("unreachable")
            || err.contains("Circuit breaker"),
        "Error should indicate identity cluster unavailability, got: {}",
        err
    );
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
// Test 3: Bridge health is maintained after gate failures
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_bridge_health_maintained_after_failures() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Check health before — bridge should start healthy
    let health: BridgeHealth = conductor
        .call(&alice.zome("governance_bridge"), "health_check", ())
        .await;
    assert!(health.healthy, "Bridge should start healthy");

    // Trigger a cross-cluster failure (identity role absent)
    let _ = conductor
        .call_fallible::<_, serde_json::Value>(
            &alice.zome("governance_bridge"),
            "dispatch_cross_cluster",
            serde_json::json!({
                "target_role": "identity",
                "zome_name": "did_registry",
                "fn_name": "verify_did",
                "payload": "did:mycelix:test"
            }),
        )
        .await;

    // Check health after — bridge should remain healthy despite the failure
    let health_after: BridgeHealth = conductor
        .call(&alice.zome("governance_bridge"), "health_check", ())
        .await;
    assert!(
        health_after.healthy,
        "Bridge should remain healthy after cross-cluster failures"
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
// Test 5: Metrics survive gate failures (no corruption)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + packed DNA (nix develop)"]
async fn test_metrics_survive_gate_failures() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let (alice,) = conductor
        .setup_app("test-app", &[dna])
        .await
        .unwrap()
        .into_tuple();

    // Trigger some failures
    for _ in 0..3 {
        let _ = conductor
            .call_fallible::<_, serde_json::Value>(
                &alice.zome("governance_bridge"),
                "dispatch_cross_cluster",
                serde_json::json!({
                    "target_role": "identity",
                    "zome_name": "did_registry",
                    "fn_name": "verify_did",
                    "payload": "did:mycelix:test"
                }),
            )
            .await;
    }

    // Metrics should still be retrievable and consistent
    let metrics: String = conductor
        .call(&alice.zome("governance_bridge"), "get_bridge_metrics", ())
        .await;

    let snapshot: serde_json::Value =
        serde_json::from_str(&metrics).expect("Metrics should be valid JSON");

    // Verify structure is intact
    assert!(
        snapshot.get("total_success").is_some(),
        "Metrics snapshot should contain total_success"
    );
    assert!(
        snapshot.get("total_errors").is_some(),
        "Metrics snapshot should contain total_errors"
    );
}
