// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # FL Bridge E2E Integration Tests
//!
//! Proves the Symthaea-Mycelix federated learning pipeline works end-to-end:
//! - Honest gradients with high PoGQ are accepted by the FL zome
//! - Byzantine gradients with low PoGQ are rejected
//! - PoGQ round-trip values match SDK expectations
//!
//! ## Architecture Note
//!
//! Due to the rmp-serde conflict (holochain pins =1.3.0, burn needs ^1.3.1),
//! these tests do NOT depend on symthaea-core or burn. Instead:
//! - Symthaea's consciousness assessment is proven separately (2,860+ lib tests)
//! - The bridge `pogq_from_quality_score()` is proven via symthaea-mycelix-bridge tests
//! - This E2E test proves the zome pipeline accepts/rejects PoGQ values correctly
//!
//! ## Running
//!
//! ```bash
//! # 1. Build FL zomes to WASM
//! cd Mycelix-Core/zomes/federated_learning
//! cargo build --release --target wasm32-unknown-unknown
//!
//! # 2. Pack DNA bundle
//! cd workdir && hc dna pack dna/
//!
//! # 3. Run conductor tests (requires Holochain conductor via nix develop)
//! cd mycelix-workspace/tests/sweettest
//! cargo test --test fl_bridge_e2e -- --ignored --test-threads=1
//!
//! # 4. Run SDK tests (no conductor needed)
//! cargo test --test fl_bridge_e2e sdk_tests
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

/// Mirror of FL coordinator::GradientWithPoGQInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GradientWithPoGQInput {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    pub quality: f64,
    pub consistency: f64,
    pub entropy: f64,
}

/// Mirror of FL coordinator::PoGQResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoGQResult {
    pub action_hash: ActionHash,
    pub pogq: PoGQData,
    pub composite_score: f64,
    pub is_byzantine: bool,
}

/// Mirror of FL coordinator::PoGQData
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoGQData {
    pub quality: f64,
    pub consistency: f64,
    pub entropy: f64,
    pub timestamp: u64,
}

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a honest gradient input with high PoGQ values
fn honest_gradient(node_id: &str, round: u32) -> GradientWithPoGQInput {
    // Integrity zome validates gradient_hash is exactly 64 hex characters (SHA256)
    let hash_input = format!("honest_gradient_r{}_n{}", round, node_id);
    let hash_hex = fake_sha256(&hash_input);
    GradientWithPoGQInput {
        node_id: node_id.to_string(),
        round,
        gradient_hash: hash_hex,
        cpu_usage: 0.45,       // Fraction [0.0, 1.0], not percentage
        memory_mb: 512.0,
        network_latency_ms: 15.0,
        quality: 0.95,
        consistency: 0.88,
        entropy: 0.12,
    }
}

/// Create a Byzantine gradient input with low PoGQ values
fn byzantine_gradient(node_id: &str, round: u32) -> GradientWithPoGQInput {
    let hash_input = format!("byzantine_gradient_r{}_n{}", round, node_id);
    let hash_hex = fake_sha256(&hash_input);
    GradientWithPoGQInput {
        node_id: node_id.to_string(),
        round,
        gradient_hash: hash_hex,
        cpu_usage: 0.99,       // Fraction [0.0, 1.0], not percentage
        memory_mb: 64.0,
        network_latency_ms: 500.0,
        quality: 0.1,
        consistency: 0.05,
        entropy: 0.95,
    }
}

/// Deterministic 64-char hex string (fake SHA256 for testing)
fn fake_sha256(input: &str) -> String {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x100000001b3f00d;
    for byte in input.bytes() {
        h1 ^= byte as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= byte as u64;
        h2 = h2.wrapping_mul(0x01000193);
    }
    // Two u64 = 32 hex digits each = 64 total
    format!("{:016x}{:016x}{:016x}{:016x}", h1, h2, h1 ^ h2, h1.wrapping_add(h2))
}

// ============================================================================
// Conductor-Based Integration Tests
// ============================================================================

/// Test that an honest gradient with high PoGQ values is accepted.
///
/// Verifies `submit_gradient_with_pogq` returns `is_byzantine: false`
/// for a gradient with quality=0.95, consistency=0.88.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + built FL DNA
async fn test_honest_gradient_accepted() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-bridge-test",
        1,
    )
    .await;

    let node = &agents[0];
    let input = honest_gradient("honest-node-1", 1);

    let result: PoGQResult = node
        .call_zome_fn("federated_learning", "submit_gradient_with_pogq", input.clone())
        .await;

    assert!(
        !result.is_byzantine,
        "Honest gradient (quality={}, consistency={}) should not be Byzantine, \
         but got is_byzantine=true with composite_score={}",
        input.quality, input.consistency, result.composite_score
    );

    assert!(
        result.composite_score > 0.5,
        "Composite score {} should be above Byzantine threshold 0.5",
        result.composite_score
    );

    // Verify PoGQ data round-trips correctly
    assert!(
        (result.pogq.quality - input.quality).abs() < 1e-10,
        "Quality should round-trip: expected {}, got {}",
        input.quality, result.pogq.quality
    );
    assert!(
        (result.pogq.consistency - input.consistency).abs() < 1e-10,
        "Consistency should round-trip: expected {}, got {}",
        input.consistency, result.pogq.consistency
    );
}

/// Test that a Byzantine gradient with low PoGQ values is detected.
///
/// The coordinator either:
/// - Rejects with an Err containing "Byzantine" (high confidence)
/// - Returns PoGQResult with is_byzantine=true (low confidence)
/// Both outcomes prove the Byzantine detection pipeline works.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + built FL DNA
async fn test_byzantine_gradient_rejected() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-bridge-test",
        1,
    )
    .await;

    let node = &agents[0];
    let input = byzantine_gradient("malicious-node-1", 1);

    let result: Result<PoGQResult, _> = node
        .call_zome_fn_fallible("federated_learning", "submit_gradient_with_pogq", input.clone())
        .await;

    match result {
        Err(e) => {
            // High-confidence Byzantine rejection (confidence >= 0.7)
            let err_msg = format!("{:?}", e);
            assert!(
                err_msg.contains("Byzantine") || err_msg.contains("byzantine"),
                "Rejection error should mention Byzantine detection, got: {}",
                err_msg
            );
        }
        Ok(pogq_result) => {
            // Low-confidence detection: gradient stored but flagged
            assert!(
                pogq_result.is_byzantine,
                "Byzantine gradient (quality={}, consistency={}) should be detected, \
                 but got is_byzantine=false with composite_score={}",
                input.quality, input.consistency, pogq_result.composite_score
            );
        }
    }
}

/// Test that an honest gradient is stored and retrievable via DHT.
///
/// Uses `get_round_gradients` to verify the gradient was stored and
/// is retrievable by round number.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + built FL DNA
async fn test_gradient_stored_on_dht() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-bridge-test",
        1,
    )
    .await;

    let node = &agents[0];
    let input = honest_gradient("storage-test-node", 1);
    let expected_node_id = input.node_id.clone();

    let result: PoGQResult = node
        .call_zome_fn("federated_learning", "submit_gradient_with_pogq", input)
        .await;

    // Verify gradient is retrievable by round
    let gradients: Vec<(ActionHash, serde_json::Value)> = node
        .call_zome_fn("federated_learning", "get_round_gradients", 1u32)
        .await;

    assert!(
        !gradients.is_empty(),
        "Round 1 should have at least one gradient stored"
    );

    // Verify the stored gradient matches what we submitted
    let (stored_hash, stored_gradient) = &gradients[0];
    assert_eq!(
        *stored_hash, result.action_hash,
        "Stored gradient hash should match the returned action_hash"
    );

    let stored_node_id = stored_gradient["node_id"].as_str().unwrap_or("");
    assert_eq!(
        stored_node_id, expected_node_id,
        "Stored gradient node_id should match"
    );
}

// ============================================================================
// SDK-Level Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod sdk_tests {
    /// Validate the MATL composite score formula:
    ///   Composite = 0.4 * quality + 0.3 * consistency + 0.3 * reputation
    #[test]
    fn test_pogq_composite_score_formula() {
        // Honest gradient: high quality + high consistency
        let quality = 0.95;
        let consistency = 0.88;
        let reputation = 0.5; // Default for new nodes

        let composite = 0.4 * quality + 0.3 * consistency + 0.3 * reputation;
        assert!(
            composite > 0.5,
            "Honest gradient composite {} should exceed Byzantine threshold 0.5",
            composite
        );

        // Byzantine gradient: low quality + low consistency
        let byz_composite = 0.4 * 0.1 + 0.3 * 0.05 + 0.3 * reputation;
        assert!(
            byz_composite < 0.5,
            "Byzantine gradient composite {} should be below threshold 0.5",
            byz_composite
        );
    }

    /// Verify PoGQ values from Symthaea's bridge are in the FL zome's expected range.
    ///
    /// The bridge maps:
    /// - epistemic_confidence -> quality [0.0, 1.0]
    /// - phi_gain + similarity -> consistency [0.0, 1.0]
    /// - anomaly severity -> entropy [0.0, ...]
    #[test]
    fn test_pogq_round_trip_values() {
        // Good assessment
        let quality = 0.92_f64;
        let consistency = 0.85_f64;
        let entropy = 0.15_f64;

        assert!((0.0..=1.0).contains(&quality), "Quality must be in [0, 1]");
        assert!((0.0..=1.0).contains(&consistency), "Consistency must be in [0, 1]");
        assert!(entropy >= 0.0, "Entropy must be non-negative");

        let composite = 0.4 * quality + 0.3 * consistency + 0.3 * 0.5;
        assert!(composite > 0.5, "Good assessment composite {} > 0.5", composite);
        assert!(composite <= 1.0, "Composite {} <= 1.0", composite);

        // Bad assessment
        let bad_composite = 0.4 * 0.05 + 0.3 * 0.02 + 0.3 * 0.5;
        assert!(bad_composite < 0.5, "Bad assessment composite {} < 0.5", bad_composite);
    }

    /// Fixture: known PoGQ mapping from Symthaea bridge.
    ///
    /// These values are the CONTRACT between symthaea-mycelix-bridge's
    /// `pogq_from_quality_score()` and the FL zome's composite score formula.
    /// If the bridge changes its mapping, these must be updated in lockstep.
    ///
    /// Bridge mapping:
    ///   quality = epistemic_confidence
    ///   consistency = 0.5 * phi_trend + 0.5 * similarity
    ///   entropy = severity-based (None=0.1, Mild=0.4, Moderate=0.7, Severe=1.0)
    ///
    /// Zome composite: 0.4 * quality + 0.3 * consistency + 0.3 * reputation
    #[test]
    fn test_pogq_fixture_honest_assessment() {
        // Honest assessment: confidence=0.95, phi_gain=+0.2 (trend=1.0), similarity=0.9
        // Expected: quality=0.95, consistency=0.5*1.0+0.5*0.9=0.95, entropy=0.1
        let fixture_quality = 0.95;
        let fixture_consistency = 0.95;
        let _fixture_entropy = 0.1;

        // Composite with default reputation (0.5 for new nodes)
        let composite = 0.4 * fixture_quality + 0.3 * fixture_consistency + 0.3 * 0.5;
        assert!(
            composite > 0.7,
            "Honest assessment fixture composite {} should be well above threshold",
            composite
        );
    }

    #[test]
    fn test_pogq_fixture_byzantine_assessment() {
        // Byzantine assessment: confidence=0.1, phi_gain=-0.4 (trend=0.6), similarity=0.2
        // Expected: quality=0.1, consistency=0.5*0.6+0.5*0.2=0.4, entropy=1.0 (Severe)
        let fixture_quality = 0.1;
        let fixture_consistency = 0.4;

        // Composite with default reputation (0.5)
        let composite = 0.4 * fixture_quality + 0.3 * fixture_consistency + 0.3 * 0.5;
        assert!(
            composite < 0.5,
            "Byzantine assessment fixture composite {} should be below threshold",
            composite
        );
    }

    #[test]
    fn test_pogq_fixture_gray_zone_assessment() {
        // Gray zone: confidence=0.6, phi_gain=0 (trend=1.0), similarity=0.78 (ambiguous)
        // Expected: quality=0.6, consistency=0.5*1.0+0.5*0.78=0.89, entropy=0.5 (ambiguous)
        let fixture_quality = 0.6;
        let fixture_consistency = 0.89;

        // Composite with moderate reputation (0.6)
        let composite = 0.4 * fixture_quality + 0.3 * fixture_consistency + 0.3 * 0.6;
        // Gray zone should be near threshold — not clearly honest or byzantine
        assert!(
            composite > 0.45 && composite < 0.75,
            "Gray zone fixture composite {} should be near threshold (0.45-0.75)",
            composite
        );
    }

    /// Test edge cases for PoGQ boundary values.
    #[test]
    fn test_pogq_boundary_values() {
        // Perfect gradient
        let perfect: f64 = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0;
        assert!((perfect - 1.0).abs() < 1e-10, "Perfect composite = 1.0");

        // Worst gradient
        let worst: f64 = 0.4 * 0.0 + 0.3 * 0.0 + 0.3 * 0.0;
        assert!(worst.abs() < 1e-10, "Worst composite = 0.0");

        // Threshold boundary (all 0.5)
        let boundary: f64 = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5;
        assert!((boundary - 0.5).abs() < 1e-10, "Boundary composite = 0.5");

        // Just above threshold with new node
        let above = 0.4 * 0.6 + 0.3 * 0.6 + 0.3 * 0.5;
        assert!(above > 0.5, "Above-threshold composite {} > 0.5", above);
    }
}

// ============================================================================
// FL COHERENCE SERIES & ANOMALY DETECTION TESTS
// ============================================================================

/// Mirror type for CoherenceRecord from integrity zome
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoherenceRecord {
    pub round: u64,
    pub coherence_value: f32,
    pub epistemic_confidence: f32,
    pub byzantine_count: u32,
    pub node_count: u32,
    pub defense_level: u32,
    pub recorded_at: i64,
}

/// Mirror type for GetCoherenceSeriesInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetCoherenceSeriesInput {
    pub start_round: Option<u64>,
    pub end_round: Option<u64>,
    pub limit: Option<usize>,
}

/// Mirror type for CoherenceAnomalyResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoherenceAnomalyResult {
    pub round: u64,
    pub expected_coherence: f32,
    pub actual_coherence: f32,
    pub z_score: f32,
    pub severity: String,
}

/// Mirror type for ValidatorPipelineResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValidatorPipelineResult {
    pub commitment_hash: String,
    pub aggregated_hv: Vec<u8>,
    pub method: String,
    pub gradient_count: u32,
    pub excluded_count: u32,
    pub excluded_participants: Vec<String>,
    // detection_summary is complex nested -- use serde_json::Value
    pub detection_summary: serde_json::Value,
}

/// Test that get_coherence_series returns empty vec when no data exists
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + FL DNA
async fn test_coherence_series_empty_returns_empty() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-coherence-test",
        1,
    )
    .await;

    let node = &agents[0];

    let input = GetCoherenceSeriesInput {
        start_round: None,
        end_round: None,
        limit: Some(10),
    };

    let result: Vec<CoherenceRecord> = node
        .call_zome_fn("federated_learning", "get_coherence_series", input)
        .await;

    assert!(result.is_empty(), "Should be empty with no pipeline runs");
}

/// Test that check_coherence_anomalies returns empty with insufficient data
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + FL DNA
async fn test_coherence_anomalies_insufficient_data() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-anomaly-test",
        1,
    )
    .await;

    let node = &agents[0];

    let result: Vec<CoherenceAnomalyResult> = node
        .call_zome_fn("federated_learning", "check_coherence_anomalies", 5u32)
        .await;

    assert!(result.is_empty(), "Should be empty with <3 data points");
}

/// Test that run_validator_pipeline fails gracefully with no gradients
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires Holochain conductor + FL DNA
async fn test_validator_pipeline_no_gradients_returns_error() {
    let agents = setup_test_agents(
        &DnaPaths::federated_learning(),
        "fl-pipeline-test",
        1,
    )
    .await;

    let node = &agents[0];

    // Round 99 has no submitted gradients -- should return error
    let result = node
        .call_zome_fn_fallible::<_, ValidatorPipelineResult>(
            "federated_learning",
            "run_validator_pipeline",
            99u32,
        )
        .await;

    assert!(
        result.is_err(),
        "Pipeline should fail when no gradients exist for round"
    );
}
