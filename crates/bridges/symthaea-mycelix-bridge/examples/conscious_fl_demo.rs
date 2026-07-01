// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea-guided federated learning update demo.
//!
//! This example shows how to:
//! - take a HyperFeel-encoded `HyperGradient`,
//! - evaluate it with `SymthaeaBackend` to obtain a `QualityScore`,
//! - map that into MATL `ProofOfGradientQuality`, and
//! - feed it into `MatlEngine` for a trust decision.

use symthaea_mycelix_bridge::{
    ConsciousnessBackend, ModelSnapshot, SymthaeaBackend, pogq_from_quality_score,
};

use mycelix_sdk::{
    hyperfeel::{HV16_BYTES, HyperGradient},
    matl::{MatlEngine, ProofOfGradientQuality},
};

/// Minimal snapshot that pretends to evaluate a model; in a real system this
/// would wrap a concrete model + validation dataset.
struct DemoSnapshot;

impl ModelSnapshot for DemoSnapshot {
    fn apply_update_clone(
        &self,
        _update: &HyperGradient,
    ) -> symthaea_mycelix_bridge::Result<Box<dyn ModelSnapshot>> {
        Ok(Box::new(DemoSnapshot))
    }

    fn evaluate(&self) -> symthaea_mycelix_bridge::Result<(f32, f32)> {
        // Toy accuracy/loss numbers.
        Ok((0.82, 0.4))
    }

    fn current_accuracy(&self) -> Option<f32> {
        Some(0.78)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic HyperGradient: in a real system this would come from
    // `HyperFeelEncoder::encode_gradient`.
    let hypergradient = HyperGradient::new(
        "demo-node".into(),
        1,
        vec![128u8; HV16_BYTES],
        0.5,
        4_000_000,
        5.0,
        [0u8; 32],
    );

    // Symthaea backend to compute Φ and a QualityScore.
    let mut backend = SymthaeaBackend::new();
    let snapshot = DemoSnapshot;

    let quality = backend.assess_update(&snapshot, &hypergradient)?;

    println!("Symthaea quality assessment:");
    println!("  accuracy: {:.3}", quality.accuracy);
    println!("  loss: {:.3}", quality.loss);
    println!(
        "  connectivity before/after/gain: {:.3} → {:.3} (Δ = {:.3})",
        quality.spectral.connectivity_before,
        quality.spectral.connectivity_after,
        quality.spectral.connectivity_gain
    );
    println!(
        "  consciousness composite: {:.3}",
        quality.consciousness_vector.composite()
    );
    println!("  best phi: {:.3}", quality.consciousness_vector.best_phi());
    println!(
        "  epistemic confidence: {:.3}",
        quality.epistemic_confidence
    );
    println!("  is anomalous? {}", quality.is_anomalous);
    if !quality.causes.is_empty() {
        println!("  causes: {:?}", quality.causes);
    }

    // Map Symthaea QualityScore into MATL PoGQ.
    let pogq: ProofOfGradientQuality = pogq_from_quality_score(&quality);

    println!();
    println!("Mapped PoGQ:");
    println!("  quality: {:.3}", pogq.quality);
    println!("  consistency: {:.3}", pogq.consistency);
    println!("  entropy: {:.3}", pogq.entropy);

    // Feed into MATL engine for a network-level decision.
    let mut matl = MatlEngine::new(20, 3, 2);
    let reputation = 0.8;

    let (node_eval, net_eval) = matl.evaluate_node(&hypergradient.node_id, &pogq, reputation);

    println!();
    println!("MATL evaluation:");
    println!("  node: {}", node_eval.node_id);
    println!("  composite score: {:.3}", node_eval.composite_score);
    println!("  node threshold: {:.3}", node_eval.node_threshold);
    println!("  anomalous? {}", node_eval.is_node_anomalous);
    println!(
        "  in suspicious cluster? {}",
        node_eval.is_in_suspicious_cluster
    );

    println!();
    println!("Network summary:");
    println!(
        "  estimated Byzantine fraction: {:.3}",
        net_eval.estimated_byzantine_fraction
    );
    println!(
        "  adaptive Byzantine threshold: {:.3}",
        net_eval.adaptive_byzantine_threshold
    );
    println!("  status: {:?}", net_eval.status);

    Ok(())
}
