//! Error Path Tests (Phase 4.2)
//!
//! Tests for edge cases and error paths that should not panic:
//! - Phi with empty nodes
//! - Phi with single node
//! - Cosine similarity of zero vectors
//! - HV operations on mismatched dimensions

use symthaea::phi_engine::PhiEngine;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::hdc::tiered_phi::{TieredPhi, ApproximationTier};
use symthaea::hdc::binary_hv::HV16;

// =============================================================================
// PHI WITH EDGE-CASE INPUTS
// =============================================================================

#[test]
fn test_phi_engine_empty_nodes() {
    let engine = PhiEngine::auto();
    let empty: Vec<ContinuousHV> = vec![];
    let result = engine.compute(&empty);
    // Should return Phi=0 for empty input, not panic
    assert_eq!(result.phi, 0.0, "Empty nodes should give Phi=0");
}

#[test]
fn test_phi_engine_single_node() {
    let engine = PhiEngine::auto();
    let single = vec![ContinuousHV::random(512, 42)];
    let result = engine.compute(&single);
    // Single node cannot integrate — Phi should be 0
    assert_eq!(result.phi, 0.0, "Single node should give Phi=0");
}

#[test]
fn test_phi_engine_two_identical_nodes() {
    let engine = PhiEngine::auto();
    let hv = ContinuousHV::random(512, 42);
    let nodes = vec![hv.clone(), hv];
    let result = engine.compute(&nodes);
    assert!(result.phi.is_finite(), "Two identical nodes should give finite Phi");
    assert!(result.phi >= 0.0, "Phi should be non-negative");
}

// =============================================================================
// TIERED PHI EDGE CASES
// =============================================================================

#[test]
fn test_tiered_phi_empty_components() {
    let mut calc = TieredPhi::new(ApproximationTier::SpectralConnectivity);
    let empty: Vec<HV16> = vec![];
    let phi = calc.compute(&empty);
    assert_eq!(phi, 0.0, "Empty components should give 0");
}

#[test]
fn test_tiered_phi_single_component() {
    let mut calc = TieredPhi::new(ApproximationTier::SpectralConnectivity);
    let single = vec![HV16::random(42)];
    let phi = calc.compute(&single);
    assert_eq!(phi, 0.0, "Single component should give 0");
}

#[test]
fn test_tiered_phi_all_tiers_handle_two_components() {
    let components = vec![HV16::random(1), HV16::random(2)];

    for tier in [
        ApproximationTier::RandomBaseline,
        ApproximationTier::SampledPartition,
        ApproximationTier::SpectralConnectivity,
        ApproximationTier::ExhaustivePartition,
    ] {
        let mut calc = TieredPhi::new(tier);
        let phi = calc.compute(&components);
        assert!(phi.is_finite(), "Tier {:?} returned non-finite for 2 components", tier);
        assert!(phi >= 0.0 && phi <= 1.0, "Tier {:?} returned out-of-range: {}", tier, phi);
    }
}

// =============================================================================
// COSINE SIMILARITY EDGE CASES
// =============================================================================

#[test]
fn test_cosine_similarity_zero_norm() {
    let zero = ContinuousHV::zero(512);
    let random = ContinuousHV::random(512, 42);
    let sim = zero.similarity(&random);
    assert!(
        !sim.is_nan(),
        "Similarity of zero vector should not be NaN (got {})",
        sim
    );
}

#[test]
fn test_cosine_similarity_both_zero() {
    let z1 = ContinuousHV::zero(512);
    let z2 = ContinuousHV::zero(512);
    let sim = z1.similarity(&z2);
    assert!(
        !sim.is_nan(),
        "Similarity of two zero vectors should not be NaN (got {})",
        sim
    );
}

#[test]
fn test_cosine_similarity_identical() {
    let hv = ContinuousHV::random(512, 99);
    let sim = hv.similarity(&hv);
    assert!(
        (sim - 1.0).abs() < 0.01,
        "Self-similarity should be ~1.0 (got {})",
        sim
    );
}

#[test]
fn test_cosine_similarity_range() {
    let a = ContinuousHV::random(512, 1);
    let b = ContinuousHV::random(512, 2);
    let sim = a.similarity(&b);
    assert!(
        sim >= -1.0 && sim <= 1.0,
        "Cosine similarity must be in [-1, 1] (got {})",
        sim
    );
}

// =============================================================================
// BINARY HV EDGE CASES
// =============================================================================

#[test]
fn test_hv16_bundle_empty() {
    let empty: Vec<HV16> = vec![];
    // HV16::bundle requires non-empty input; verify it doesn't panic
    // or returns a zero vector
    let result = HV16::bundle(&empty);
    // Should return zero or some default, not crash
    let _ = result;
}

#[test]
fn test_hv16_bundle_single() {
    let hv = HV16::random(42);
    let bundled = HV16::bundle(&[hv.clone()]);
    // Bundle of single element should be similar to itself
    let sim = hv.similarity(&bundled);
    assert!(sim > 0.9, "Bundle of single HV should be very similar to original");
}

#[test]
fn test_hv16_similarity_with_self() {
    let hv = HV16::random(42);
    let sim = hv.similarity(&hv);
    assert!(
        (sim - 1.0).abs() < 0.01,
        "Self-similarity should be ~1.0 for HV16"
    );
}
