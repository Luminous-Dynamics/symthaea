//! Minimal MATL + HyperFeel demo.
//!
//! This example shows how to:
//! - encode a synthetic gradient with HyperFeel, and
//! - feed a PoGQ measurement into the high-level MatlEngine.
//!
//! It does **not** talk to any Holochain conductor; it is a pure Rust
//! library demo intended for quick experimentation.

use mycelix_sdk::{
    hyperfeel::{EncodingConfig, HyperFeelEncoder},
    matl::{MatlEngine, ProofOfGradientQuality},
};

fn main() {
    // Create a small synthetic gradient.
    let gradient: Vec<f32> = (0..2_000).map(|i| (i as f32 * 0.002).sin()).collect();

    // HyperFeel encoder with default HV16 configuration.
    let config = EncodingConfig::default();
    let mut encoder = HyperFeelEncoder::new(config);

    let hyper = encoder.encode_gradient(&gradient, 1, "demo-node");

    println!("Encoded hypervector size: {:.1} KB", hyper.size_kb());
    println!(
        "Compression ratio: {:.1}x ({} bytes → {} bytes)",
        hyper.compression_ratio,
        hyper.original_size,
        hyper.hypervector.len()
    );

    // Construct a simple PoGQ signal for this contribution. In a real system,
    // quality/consistency/entropy would come from the FL coordinator.
    let pogq = ProofOfGradientQuality::new(
        0.9,  // quality
        0.85, // consistency
        0.1,  // entropy
    );

    // High-level MATL orchestrator with modest history window and hierarchy.
    let mut engine = MatlEngine::new(20, 3, 2);
    let reputation = 0.8;

    let (node_eval, net_eval) = engine.evaluate_node(&hyper.node_id, &pogq, reputation);

    println!();
    println!("=== MATL Evaluation ===");
    println!("Node: {}", node_eval.node_id);
    println!("Composite score: {:.3}", node_eval.composite_score);
    println!("Node threshold: {:.3}", node_eval.node_threshold);
    println!("Is anomalous for node? {}", node_eval.is_node_anomalous);
    println!(
        "In suspicious cluster? {}",
        node_eval.is_in_suspicious_cluster
    );
    println!();
    println!(
        "Estimated Byzantine fraction: {:.3}",
        net_eval.estimated_byzantine_fraction
    );
    println!(
        "Adaptive Byzantine threshold: {:.3}",
        net_eval.adaptive_byzantine_threshold
    );
    println!("Network status: {:?}", net_eval.status);
}
