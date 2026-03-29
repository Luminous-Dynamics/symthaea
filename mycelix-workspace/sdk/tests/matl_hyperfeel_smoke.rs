// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL + HyperFeel smoke tests.
//!
//!"Does everything still fit together?" rather than deep statistical
//! guarantees. These tests are intentionally light-weight so they can
//! run in CI by default.

use mycelix_sdk::{
    hyperfeel::{EncodingConfig, HyperFeelEncoder},
    matl::{MatlEngine, ProofOfGradientQuality},
};

/// Simple end-to-end check that HyperFeel + MATL can evaluate a node
/// contribution without panicking and that honest vs obviously bad
/// contributions are distinguished.
#[test]
fn matl_and_hyperfeel_basic_flow() {
    // HyperFeel setup.
    let config = EncodingConfig::default();
    let mut encoder = HyperFeelEncoder::new(config);

    // Two similar gradients and one clearly different gradient.
    let grad_honest: Vec<f32> = (0..2_000).map(|i| (i as f32 * 0.001).sin()).collect();
    let grad_honest_perturbed: Vec<f32> = (0..2_000)
        .map(|i| (i as f32 * 0.001).sin() + 0.01)
        .collect();
    let grad_bad: Vec<f32> = (0..2_000)
        .map(|i| {
            let x = i as f32;
            ((x * 0.07).sin() * (x * 0.13).cos()).powi(3) + (x / 2_000.0).exp() - 1.0
        })
        .collect();

    let hg1 = encoder.encode_gradient(&grad_honest, 1, "node-h1");
    let hg2 = encoder.encode_gradient(&grad_honest_perturbed, 1, "node-h2");
    let hg_bad = encoder.encode_gradient(&grad_bad, 1, "node-bad");

    let sim_hh = HyperFeelEncoder::cosine_similarity(&hg1.hypervector, &hg2.hypervector);
    let sim_hb = HyperFeelEncoder::cosine_similarity(&hg1.hypervector, &hg_bad.hypervector);

    assert!(
        sim_hh > sim_hb,
        "similar gradients should have higher similarity (hh={} vs hb={})",
        sim_hh,
        sim_hb
    );

    // MATL setup – use the high-level orchestrator.
    let mut engine = MatlEngine::new(20, 3, 2);

    // Honest nodes get high PoGQ; bad node gets very low.
    let pogq_honest = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
    let pogq_bad = ProofOfGradientQuality::new(0.1, 0.2, 0.9);

    let (eval_h1, _) = engine.evaluate_node("node-h1", &pogq_honest, 0.8);
    let (eval_h2, _) = engine.evaluate_node("node-h2", &pogq_honest, 0.8);
    let (eval_bad, net_eval) = engine.evaluate_node("node-bad", &pogq_bad, 0.2);

    // Honest nodes should not be anomalous; the bad node should be.
    assert!(
        !eval_h1.is_node_anomalous && !eval_h2.is_node_anomalous,
        "honest nodes should not be anomalous"
    );
    assert!(
        eval_bad.is_node_anomalous,
        "clearly bad node should be treated as anomalous"
    );

    // Network evaluation should always be within configured bounds.
    assert!(
        net_eval.adaptive_byzantine_threshold >= mycelix_sdk::matl::MIN_BYZANTINE_TOLERANCE
            && net_eval.adaptive_byzantine_threshold <= mycelix_sdk::matl::MAX_BYZANTINE_TOLERANCE,
        "adaptive threshold must remain within configured bounds"
    );
}
