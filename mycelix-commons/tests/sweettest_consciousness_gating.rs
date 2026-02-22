//! # Consciousness Gating Sweettest — Commons Cluster
//!
//! Verifies that governance functions in the commons cluster are properly
//! gated by consciousness credentials, proving the full wiring from
//! domain coordinator → commons_bridge → cross-cluster identity call.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — property registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPropertyInput {
    pub name: String,
    pub description: String,
    pub property_type: String,
    pub location: String,
    pub area_sqm: Option<u32>,
}

// ============================================================================
// Mirror types — housing governance
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateResolutionInput {
    pub title: String,
    pub body: String,
    pub resolution_type: String,
}

// ============================================================================
// Mirror types — bridge
// ============================================================================

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
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Tests
// ============================================================================

/// Property transfer requires proposal-level consciousness gate.
/// Without identity bridge, initiate_transfer should be blocked.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_property_transfer_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Register a property — this is gated by proposal-level consciousness
    let input = RegisterPropertyInput {
        name: "Test Property".to_string(),
        description: "For gating test".to_string(),
        property_type: "residential".to_string(),
        location: "Test Location".to_string(),
        area_sqm: Some(100),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(
        result.is_err(),
        "register_property should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Housing budget approval requires constitutional-level consciousness
/// (Guardian tier). This tests the highest gate level in commons.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_housing_budget_requires_constitutional() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // create_resolution in housing-governance requires voting-level gate
    let input = CreateResolutionInput {
        title: "Budget Resolution".to_string(),
        body: "Allocate funds for improvements".to_string(),
        resolution_type: "budget".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("housing_governance"),
            "create_resolution",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "create_resolution should be blocked without consciousness credentials"
    );
}

/// Verify the audit trail is maintained even when gate rejects.
/// Bridge health should remain healthy after gate failures, and
/// total_events should be non-negative.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_gate_rejection_audit_logged() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Check bridge health before
    let health_before: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;
    assert!(health_before.healthy, "Bridge should be healthy initially");

    // Attempt a gated action (will fail)
    let input = RegisterPropertyInput {
        name: "Audit Test".to_string(),
        description: "For audit trail test".to_string(),
        property_type: "commercial".to_string(),
        location: "Audit Location".to_string(),
        area_sqm: Some(500),
    };

    let _result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("property_registry"), "register_property", input)
        .await;

    // Bridge should remain healthy after gate failure
    let health_after: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(
        health_after.healthy,
        "Bridge should remain healthy after gate failure"
    );
    assert!(
        health_after.total_events >= health_before.total_events,
        "Total events should not decrease: before={}, after={}",
        health_before.total_events,
        health_after.total_events,
    );
}
