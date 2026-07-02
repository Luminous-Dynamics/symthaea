#![allow(deprecated)] // Tests use legacy ConsciousnessCredential/Tier for backward-compat bridge testing
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Security Hardening Sweettest — Commons Cluster
//!
//! Validates the critical security improvements made during the March 22, 2026
//! comprehensive review:
//!
//! 1. **Fail-closed credential fallback** — Identity partition → error, not
//!    silent Citizen-tier grant (commit e9e2889fd)
//! 2. **Consciousness gating on bridge writes** — query_commons, resolve_query,
//!    broadcast_event now require tier checks (commit 137a37c20)
//! 3. **Metrics export** — get_bridge_metrics returns valid JSON (commit 7f2eeffc0)
//! 4. **Structured error codes** — dispatch errors carry BridgeErrorCode (commit 7da145417)
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! cd tests
//! cargo test --release --test sweettest_security_hardening -- --ignored --test-threads=1
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — must match bridge coordinator structs
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsQuery {
    pub domain: String,
    pub query_type: String,
    pub requester: String,
    pub params: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsEvent {
    pub domain: String,
    pub event_type: String,
    pub payload: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchInput {
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
    pub error_code: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("COMMONS_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Test 1: Fail-Closed Credential Fallback
//
// When running with a single DNA (no identity cluster), the bridge should
// fail-closed instead of issuing permissive fallback credentials.
// Before the fix: returned ConsciousnessCredential with 0.5/0.5/0.5/0.5
// After the fix: returns Err("Identity cluster unreachable")
// ============================================================================

/// Verify that get_consciousness_credential fails when identity is unreachable.
///
/// This is the CRITICAL security test — prior to commit e9e2889fd, this
/// function silently returned Citizen-tier credentials (0.5 across all
/// dimensions), allowing unverified agents to vote and perform treasury ops.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_fail_closed_when_identity_unreachable() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    // Single-DNA setup — NO identity role available
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Call get_consciousness_credential directly on the bridge
    let did = "did:mycelix:test-alice".to_string();
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("commons_bridge"),
            "get_consciousness_credential",
            did,
        )
        .await;

    // MUST fail — not silently succeed with fallback credentials
    assert!(
        result.is_err(),
        "get_consciousness_credential MUST fail when identity cluster is \
         unreachable. If this passes, the fail-closed fix has been reverted!"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("Identity cluster unreachable")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole"),
        "Error should indicate identity cluster unavailability, got: {}",
        err_msg,
    );
}

/// Verify that refresh_consciousness_credential also fails closed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_refresh_fails_closed_when_identity_unreachable() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let did = "did:mycelix:test-alice".to_string();
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(
            &alice.zome("commons_bridge"),
            "refresh_consciousness_credential",
            did,
        )
        .await;

    assert!(
        result.is_err(),
        "refresh_consciousness_credential MUST fail when identity is unreachable"
    );
}

// ============================================================================
// Test 2: Consciousness Gating on Bridge Write Operations
//
// query_commons, resolve_query, and broadcast_event now require consciousness
// credentials. Without an identity cluster, these should all be rejected.
// ============================================================================

/// query_commons requires Participant tier — should fail without identity.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_query_commons_gated_by_consciousness() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let query = CommonsQuery {
        domain: "property".to_string(),
        query_type: "get_all".to_string(),
        requester: "did:mycelix:test-alice".to_string(),
        params: "{}".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    assert!(
        result.is_err(),
        "query_commons should be rejected without consciousness credentials"
    );
}

/// broadcast_event requires Participant tier — should fail without identity.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_broadcast_event_gated_by_consciousness() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let event = CommonsEvent {
        domain: "property".to_string(),
        event_type: "test_event".to_string(),
        payload: r#"{"test": true}"#.to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(
        result.is_err(),
        "broadcast_event should be rejected without consciousness credentials"
    );
}

// ============================================================================
// Test 3: Metrics Export
//
// get_bridge_metrics should return valid JSON with the BridgeMetricsSnapshot
// schema, even when no dispatches have occurred.
// ============================================================================

/// Metrics endpoint returns valid JSON snapshot.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_metrics_returns_valid_json() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Call metrics endpoint
    let json_str: String = conductor
        .call(&alice.zome("commons_bridge"), "get_bridge_metrics", ())
        .await;

    // Must be valid JSON
    let snapshot: serde_json::Value = serde_json::from_str(&json_str)
        .expect("get_bridge_metrics must return valid JSON");

    // Verify schema has expected fields
    assert!(
        snapshot.get("total_success").is_some(),
        "Snapshot must have total_success field"
    );
    assert!(
        snapshot.get("total_errors").is_some(),
        "Snapshot must have total_errors field"
    );
    assert!(
        snapshot.get("total_cross_cluster").is_some(),
        "Snapshot must have total_cross_cluster field"
    );
    assert!(
        snapshot.get("rate_limit_hits").is_some(),
        "Snapshot must have rate_limit_hits field"
    );
    assert!(
        snapshot.get("call_counts").is_some(),
        "Snapshot must have call_counts field"
    );

    // Initial state: all counters should be zero or near-zero
    let total_success = snapshot["total_success"].as_u64().unwrap_or(999);
    assert!(
        total_success < 100,
        "Fresh metrics should have few successes, got {}",
        total_success
    );
}

/// Metrics endpoint records dispatch errors after failed calls.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_metrics_record_dispatch_errors() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Dispatch to a non-existent zome (should be rejected by allowlist)
    let bad_dispatch = DispatchInput {
        zome: "nonexistent_zome".to_string(),
        fn_name: "fake_function".to_string(),
        payload: vec![],
    };

    let result: DispatchResult = conductor
        .call(
            &alice.zome("commons_bridge"),
            "dispatch_call",
            bad_dispatch,
        )
        .await;

    assert!(!result.success, "Dispatch to nonexistent zome should fail");
    assert!(
        result.error.as_deref().unwrap_or("").contains("not in the commons allowlist"),
        "Error should mention allowlist rejection"
    );

    // Now check metrics — should show at least 1 error
    let json_str: String = conductor
        .call(&alice.zome("commons_bridge"), "get_bridge_metrics", ())
        .await;

    let snapshot: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let total_errors = snapshot["total_errors"].as_u64().unwrap_or(0);
    assert!(
        total_errors >= 1,
        "Metrics should record at least 1 error after failed dispatch, got {}",
        total_errors
    );
}

// ============================================================================
// Test 4: Structured Error Codes
//
// Dispatch failures should carry structured BridgeErrorCode for programmatic
// handling, not just string messages.
// ============================================================================

/// Allowlist rejection returns BridgeErrorCode::AllowlistRejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_dispatch_error_code_on_allowlist_rejection() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let bad_dispatch = DispatchInput {
        zome: "unauthorized_zome".to_string(),
        fn_name: "evil_function".to_string(),
        payload: vec![],
    };

    let result: DispatchResult = conductor
        .call(
            &alice.zome("commons_bridge"),
            "dispatch_call",
            bad_dispatch,
        )
        .await;

    assert!(!result.success);
    assert!(
        result.error_code.is_some(),
        "Failed dispatch should carry structured error_code"
    );
    assert_eq!(
        result.error_code.as_deref(),
        Some("AllowlistRejected"),
        "Allowlist rejection should return AllowlistRejected code"
    );
}

// ============================================================================
// Test 5: Bridge Health After Security Operations
//
// The bridge should remain healthy after gate failures and dispatch rejections.
// This validates that our security hardening doesn't destabilize the system.
// ============================================================================

/// Bridge remains healthy after multiple security rejections.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_healthy_after_security_rejections() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Check initial health
    let health_before: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;
    assert!(health_before.healthy);

    // Trigger multiple rejection scenarios
    for _ in 0..5 {
        // 1. Allowlist rejection
        let _: DispatchResult = conductor
            .call(
                &alice.zome("commons_bridge"),
                "dispatch_call",
                DispatchInput {
                    zome: "bad_zome".into(),
                    fn_name: "bad_fn".into(),
                    payload: vec![],
                },
            )
            .await;

        // 2. Consciousness gate rejection (query_commons)
        let _: Result<::holochain::prelude::Record, _> = conductor
            .call_fallible(
                &alice.zome("commons_bridge"),
                "query_commons",
                CommonsQuery {
                    domain: "property".into(),
                    query_type: "get_all".into(),
                    requester: "did:test".into(),
                    params: "{}".into(),
                },
            )
            .await;
    }

    // Bridge should STILL be healthy
    let health_after: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;
    assert!(
        health_after.healthy,
        "Bridge must remain healthy after multiple security rejections"
    );
}
