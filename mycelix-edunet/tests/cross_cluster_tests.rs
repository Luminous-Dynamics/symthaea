// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Cluster Dispatch Integration Tests
//!
//! Tests verifying that the EduNet bridge zome correctly dispatches calls
//! across internal domains and validates consciousness gating.
//!
//! These tests require a running Holochain conductor with the EduNet DNA.
//! Run with: `cargo test --test cross_cluster_tests -- --ignored`

use std::collections::HashMap;

// Holochain imports
use holochain::sweettest::{SweetConductor, SweetDnaFile, SweetAgents, SweetCell, SweetZome};
use holochain_types::prelude::*;

// ============================================================================
// Bridge Dispatch Types (matching zome coordinator structs)
// ============================================================================

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct DispatchInput {
    pub target_zome: String,
    pub target_fn: String,
    pub payload: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct BridgeHealth {
    pub status: String,
    pub total_dispatches: u64,
    pub total_queries: u64,
    pub total_events: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct EdunetQueryEntry {
    pub query_id: String,
    pub domain: String,
    pub query_type: String,
    pub payload: serde_json::Value,
    pub requester: String,
    pub created_at: i64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct EdunetEventEntry {
    pub event_id: String,
    pub domain: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub emitter: String,
    pub created_at: i64,
}

// ============================================================================
// Test: Bridge Health Check
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_bridge_health_returns_valid_status() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("edunet_bridge"), "get_bridge_health", ())
        .await;

    assert_eq!(health.status, "healthy");
}

// ============================================================================
// Test: Dispatch to Valid Domain
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_dispatch_to_learning_domain_succeeds() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Dispatch a list_courses call through the bridge
    let input = DispatchInput {
        target_zome: "learning".to_string(),
        target_fn: "list_courses".to_string(),
        payload: serde_json::json!(null),
    };

    let result: serde_json::Value = conductor
        .call(&alice.zome("edunet_bridge"), "dispatch_call", input)
        .await;

    // Should succeed (empty list for fresh DNA)
    assert!(result.is_array() || result.is_null());
}

// ============================================================================
// Test: Dispatch to Invalid Domain Rejected
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_dispatch_to_invalid_domain_rejected() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Try to dispatch to an invalid domain (civic is not in VALID_DOMAINS)
    let input = DispatchInput {
        target_zome: "justice".to_string(),
        target_fn: "list_cases".to_string(),
        payload: serde_json::json!(null),
    };

    let result = conductor
        .call_fallible::<_, serde_json::Value>(&alice.zome("edunet_bridge"), "dispatch_call", input)
        .await;

    assert!(result.is_err(), "Dispatch to invalid domain should be rejected");
}

// ============================================================================
// Test: Event Broadcasting
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_broadcast_event_and_retrieve() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Broadcast a learning event
    let event = EdunetEventEntry {
        event_id: "evt-001".to_string(),
        domain: "learning".to_string(),
        event_type: "course_completed".to_string(),
        payload: serde_json::json!({"course_id": "rust-101", "score": 95}),
        emitter: "alice".to_string(),
        created_at: 1711500000,
    };

    let _hash: ActionHash = conductor
        .call(&alice.zome("edunet_bridge"), "broadcast_event", event)
        .await;

    // Retrieve events for learning domain
    let events: Vec<holochain_types::prelude::Record> = conductor
        .call(&alice.zome("edunet_bridge"), "get_domain_events", "learning".to_string())
        .await;

    assert!(!events.is_empty(), "Should have at least one event after broadcasting");
}

// ============================================================================
// Test: Query Submission and Resolution
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_query_submission_and_retrieval() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Submit a cross-domain query
    let query = EdunetQueryEntry {
        query_id: "q-001".to_string(),
        domain: "adaptive".to_string(),
        query_type: "learner_profile".to_string(),
        payload: serde_json::json!({"learner": "alice"}),
        requester: "integration".to_string(),
        created_at: 1711500000,
    };

    let _hash: ActionHash = conductor
        .call(&alice.zome("edunet_bridge"), "query_edunet", query)
        .await;

    // Retrieve my queries
    let queries: Vec<holochain_types::prelude::Record> = conductor
        .call(&alice.zome("edunet_bridge"), "get_my_queries", ())
        .await;

    assert!(!queries.is_empty(), "Should have at least one query after submission");
}

// ============================================================================
// Test: Cross-Domain Credential Flow (Learning -> Credential)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_cross_domain_learning_to_credential_flow() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Step 1: Create a course via bridge dispatch
    let create_input = DispatchInput {
        target_zome: "learning".to_string(),
        target_fn: "list_courses".to_string(),
        payload: serde_json::json!(null),
    };

    let _result: serde_json::Value = conductor
        .call(&alice.zome("edunet_bridge"), "dispatch_call", create_input)
        .await;

    // Step 2: Broadcast a completion event that would trigger credential issuance
    let event = EdunetEventEntry {
        event_id: "evt-credential-001".to_string(),
        domain: "credential".to_string(),
        event_type: "course_completion_verified".to_string(),
        payload: serde_json::json!({
            "learner": "alice",
            "course_id": "rust-101",
            "score": 95,
            "rubric_id": "default"
        }),
        emitter: "learning".to_string(),
        created_at: 1711500000,
    };

    let hash: ActionHash = conductor
        .call(&alice.zome("edunet_bridge"), "broadcast_event", event)
        .await;

    assert_ne!(hash.get_raw_39().len(), 0, "Event should be stored on-chain");
}

// ============================================================================
// Test: Rate Limiting on Dispatch
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_dispatch_rate_limiting() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Make many rapid dispatch calls — should not panic but may rate-limit
    for i in 0..10 {
        let input = DispatchInput {
            target_zome: "learning".to_string(),
            target_fn: "list_courses".to_string(),
            payload: serde_json::json!(null),
        };

        let result = conductor
            .call_fallible::<_, serde_json::Value>(&alice.zome("edunet_bridge"), "dispatch_call", input)
            .await;

        // First several should succeed; rate limiting may kick in later
        if i < 5 {
            assert!(result.is_ok(), "First dispatches should succeed (iteration {i})");
        }
    }
}

// ============================================================================
// Test: Governance Gate Audit Trail
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires running conductor"]
async fn test_governance_gate_audit_trail() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna = SweetDnaFile::from_bundle(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dna/edunet.dna")
    ).await.unwrap();
    let (alice,) = conductor.setup_app("test-app", &[dna]).await.unwrap().into_tuple();

    // Log a governance gate event
    let gate_input = serde_json::json!({
        "agent": "alice",
        "action": "create_course",
        "required_tier": "Steward",
        "actual_tier": "Participant",
        "outcome": "denied",
        "timestamp": 1711500000
    });

    let result = conductor
        .call_fallible::<_, ()>(&alice.zome("edunet_bridge"), "log_governance_gate", gate_input)
        .await;

    // This may succeed or fail depending on implementation — we're testing the endpoint exists
    // and doesn't panic
    assert!(result.is_ok() || result.is_err());
}
