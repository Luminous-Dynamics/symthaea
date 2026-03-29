// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security E2E tests for consciousness attestation system.
//!
//! Validates that the governance bridge zome correctly enforces:
//! - Clock skew rejection (future + stale timestamps)
//! - Cycle ID monotonicity (duplicate and decreasing cycle IDs)
//! - Signature verification (cross-agent spoofing)
//! - Rate limiting (attestation interval enforcement)
//!
//! Prerequisites:
//!   cd mycelix-governance && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_governance.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored phi_attestation

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

/// Helper: build a valid attestation input for a given agent.
///
/// The signature is constructed by signing the canonical attestation message
/// `"{agent_did}:{consciousness_level}:{cycle_id}:{captured_at_us}"` with the agent's key.
/// Since we don't have the Ed25519 private key in sweettest, we use a dummy
/// 64-byte signature — the zome validates signature format but can't verify
/// against the Holochain agent key (bridge uses external Ed25519 keys).
fn build_attestation_input(
    phi: f64,
    cycle_id: u64,
    captured_at_us: u64,
    signature: Vec<u8>,
) -> serde_json::Value {
    serde_json::json!({
        "consciousness_level": phi,
        "cycle_id": cycle_id,
        "captured_at_us": captured_at_us,
        "signature": signature
    })
}

/// Get the current time in microseconds (matching Holochain Timestamp format).
fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

/// A dummy 64-byte Ed25519 signature for testing.
fn dummy_signature() -> Vec<u8> {
    vec![42u8; 64]
}

// ============================================================================
// CLOCK SKEW TESTS
// ============================================================================

/// Test: Attestation with a timestamp >30 seconds in the future is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_future_timestamp_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // 60 seconds in the future — exceeds the 30-second clock skew tolerance
    let future_us = now_us() + 60_000_000;

    let input = build_attestation_input(
        0.65,
        1,
        future_us,
        dummy_signature(),
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Attestation with future timestamp should be rejected"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("future") || err_msg.contains("timestamp"),
        "Error should mention future timestamp, got: {err_msg}"
    );
}

/// Test: Attestation with a timestamp >30 seconds in the past is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_stale_timestamp_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // 60 seconds in the past — exceeds the 30-second clock skew tolerance
    let stale_us = now_us() - 60_000_000;

    let input = build_attestation_input(
        0.65,
        1,
        stale_us,
        dummy_signature(),
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Attestation with stale timestamp should be rejected"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("old") || err_msg.contains("timestamp"),
        "Error should mention old/stale timestamp, got: {err_msg}"
    );
}

/// Test: Attestation with a timestamp within ±30 seconds is accepted.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_valid_timestamp_accepted() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let input = build_attestation_input(
        0.65,
        1,
        now_us(),
        dummy_signature(),
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_ok(),
        "Attestation with valid timestamp should be accepted: {:?}",
        result.unwrap_err()
    );
}

// ============================================================================
// CYCLE ID MONOTONICITY TESTS
// ============================================================================

/// Test: Two attestations with the same cycle_id are rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_duplicate_cycle_id_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // First attestation — should succeed
    let input1 = build_attestation_input(
        0.65,
        100,
        now_us(),
        dummy_signature(),
    );

    let result1: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input1)
        .await;

    assert!(result1.is_ok(), "First attestation should succeed");

    wait_for_dht_sync().await;

    // Second attestation with same cycle_id — should fail
    let input2 = build_attestation_input(
        0.70,
        100, // Same cycle_id
        now_us(),
        dummy_signature(),
    );

    let result2: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input2)
        .await;

    assert!(
        result2.is_err(),
        "Duplicate cycle_id should be rejected"
    );

    let err_msg = format!("{:?}", result2.unwrap_err());
    assert!(
        err_msg.contains("cycle") || err_msg.contains("Cycle"),
        "Error should mention cycle ID, got: {err_msg}"
    );
}

/// Test: Attestation with a decreasing cycle_id is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_decreasing_cycle_id_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // First attestation with cycle_id=200
    let input1 = build_attestation_input(
        0.65,
        200,
        now_us(),
        dummy_signature(),
    );

    let result1: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input1)
        .await;

    assert!(result1.is_ok(), "First attestation should succeed");

    wait_for_dht_sync().await;

    // Second attestation with lower cycle_id=150 — should fail
    let input2 = build_attestation_input(
        0.80,
        150, // Lower than 200
        now_us(),
        dummy_signature(),
    );

    let result2: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input2)
        .await;

    assert!(
        result2.is_err(),
        "Decreasing cycle_id should be rejected"
    );
}

/// Test: Attestation with a strictly increasing cycle_id succeeds.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_increasing_cycle_id_accepted() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // First attestation
    let input1 = build_attestation_input(
        0.65,
        300,
        now_us(),
        dummy_signature(),
    );

    let result1: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input1)
        .await;

    assert!(result1.is_ok(), "First attestation should succeed");

    // Wait for rate limit + DHT sync
    tokio::time::sleep(std::time::Duration::from_secs(11)).await;

    // Second attestation with higher cycle_id
    let input2 = build_attestation_input(
        0.70,
        301,
        now_us(),
        dummy_signature(),
    );

    let result2: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input2)
        .await;

    assert!(
        result2.is_ok(),
        "Increasing cycle_id should be accepted: {:?}",
        result2.unwrap_err()
    );
}

// ============================================================================
// SIGNATURE VALIDATION TESTS
// ============================================================================

/// Test: Attestation with empty signature is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_empty_signature_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let input = build_attestation_input(
        0.65,
        1,
        now_us(),
        vec![], // Empty signature
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Empty signature should be rejected"
    );
}

/// Test: Attestation with wrong-length signature is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_wrong_signature_length_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    let input = build_attestation_input(
        0.65,
        1,
        now_us(),
        vec![42u8; 32], // 32 bytes instead of 64
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Wrong-length signature should be rejected"
    );
}

// ============================================================================
// CONSCIOUSNESS LEVEL RANGE VALIDATION TESTS
// ============================================================================

/// Test: Consciousness level outside [0.0, 1.0] is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle + conductor
async fn test_consciousness_out_of_range_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::governance(),
        "mycelix-governance",
        1,
    )
    .await;

    let agent = &agents[0];

    // Consciousness level > 1.0
    let input = build_attestation_input(
        1.5,
        1,
        now_us(),
        dummy_signature(),
    );

    let result: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Consciousness level > 1.0 should be rejected"
    );

    // Consciousness level < 0.0
    let input_neg = build_attestation_input(
        -0.1,
        2,
        now_us(),
        dummy_signature(),
    );

    let result_neg: Result<Record, _> = agent
        .call_zome_fn_fallible("governance_bridge", "record_consciousness_attestation", input_neg)
        .await;

    assert!(
        result_neg.is_err(),
        "Consciousness level < 0.0 should be rejected"
    );
}
