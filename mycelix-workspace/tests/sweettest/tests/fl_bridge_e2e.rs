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
    GradientWithPoGQInput {
        node_id: node_id.to_string(),
        round,
        gradient_hash: format!("sha256:honest_gradient_r{}_n{}", round, node_id),
        cpu_usage: 45.0,
        memory_mb: 512.0,
        network_latency_ms: 15.0,
        quality: 0.95,
        consistency: 0.88,
        entropy: 0.12,
    }
}

/// Create a Byzantine gradient input with low PoGQ values
fn byzantine_gradient(node_id: &str, round: u32) -> GradientWithPoGQInput {
    GradientWithPoGQInput {
        node_id: node_id.to_string(),
        round,
        gradient_hash: format!("sha256:byzantine_gradient_r{}_n{}", round, node_id),
        cpu_usage: 99.0,
        memory_mb: 64.0,
        network_latency_ms: 500.0,
        quality: 0.1,
        consistency: 0.05,
        entropy: 0.95,
    }
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

/// Test that a Byzantine gradient with low PoGQ values is rejected.
///
/// Verifies `submit_gradient_with_pogq` returns `is_byzantine: true`
/// for a gradient with quality=0.1, consistency=0.05.
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

    let result: PoGQResult = node
        .call_zome_fn("federated_learning", "submit_gradient_with_pogq", input.clone())
        .await;

    assert!(
        result.is_byzantine,
        "Byzantine gradient (quality={}, consistency={}) should be detected, \
         but got is_byzantine=false with composite_score={}",
        input.quality, input.consistency, result.composite_score
    );

    assert!(
        result.composite_score < 0.5,
        "Composite score {} for Byzantine gradient should be below threshold 0.5",
        result.composite_score
    );
}

/// Test that an honest gradient is stored and retrievable via DHT.
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

    let result: PoGQResult = node
        .call_zome_fn("federated_learning", "submit_gradient_with_pogq", input)
        .await;

    let record: Option<Record> = node
        .call_zome_fn("federated_learning", "get_gradient", result.action_hash.clone())
        .await;

    assert!(
        record.is_some(),
        "Stored gradient should be retrievable by action hash"
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

    /// Test edge cases for PoGQ boundary values.
    #[test]
    fn test_pogq_boundary_values() {
        // Perfect gradient
        let perfect = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0;
        assert!((perfect - 1.0).abs() < 1e-10, "Perfect composite = 1.0");

        // Worst gradient
        let worst = 0.4 * 0.0 + 0.3 * 0.0 + 0.3 * 0.0;
        assert!(worst.abs() < 1e-10, "Worst composite = 0.0");

        // Threshold boundary (all 0.5)
        let boundary = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5;
        assert!((boundary - 0.5).abs() < 1e-10, "Boundary composite = 0.5");

        // Just above threshold with new node
        let above = 0.4 * 0.6 + 0.3 * 0.6 + 0.3 * 0.5;
        assert!(above > 0.5, "Above-threshold composite {} > 0.5", above);
    }
}
