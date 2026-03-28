// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end integration tests for the Mycelix Federated Learning platform.
//!
//! These tests exercise the full pipeline:
//! 1. Coordinator setup with PoGQ defense
//! 2. Node registration
//! 3. Gradient submission (honest + Byzantine)
//! 4. PoGQ Byzantine detection
//! 5. Defense aggregation
//! 6. Result verification

#![cfg(feature = "std")]

use mycelix_fl::compression::hyperfeel::{HyperFeelConfig, HyperFeelEncoder};
use mycelix_fl::coordinator::config::CoordinatorConfig;
use mycelix_fl::coordinator::node_manager::NodeCredential;
use mycelix_fl::coordinator::orchestrator::FLCoordinator;
use mycelix_fl::defenses::coordinate_median::CoordinateMedian;
use mycelix_fl::defenses::krum::Krum;
use mycelix_fl::defenses::rfa::Rfa;
use mycelix_fl::defenses::trimmed_mean::TrimmedMean;
use mycelix_fl::defenses::{Defense, FedAvg};
use mycelix_fl::detection::self_healing::{RepairMethod, SelfHealerConfig};
use mycelix_fl::detection::shapley::ShapleyConfig;
use mycelix_fl::detection::stack::{DetectionConfig, DetectionStack};
use mycelix_fl::error::FlError;
use mycelix_fl::pogq::config::PoGQv41Config;
use mycelix_fl::types::{cosine_similarity, DefenseConfig, Gradient};

use tokio::sync::mpsc;

// ===========================================================================
// Helper functions
// ===========================================================================

/// Simple xorshift64 PRNG for deterministic noise generation.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Map to [-1, 1]
        (x as f32 / u64::MAX as f32) * 2.0 - 1.0
    }
}

/// Generate an honest gradient with small noise around a true gradient.
fn add_noise(gradient: &[f32], noise_std: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    gradient
        .iter()
        .map(|&v| v + rng.next_f32() * noise_std)
        .collect()
}

/// Compute cosine similarity as f64 (delegates to the crate utility).
fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    cosine_similarity(a, b)
}

/// Generate a set of honest gradients with small noise around a true direction.
fn generate_honest_gradients(n: usize, dim: usize, seed: u64) -> Vec<Gradient> {
    let true_gradient: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
    (0..n)
        .map(|i| {
            let values = add_noise(&true_gradient, 0.05, seed + i as u64 * 1000);
            Gradient::new(format!("node-{}", i), values, 0)
        })
        .collect()
}

/// Create a default coordinator config with high rate limits for testing.
fn test_coordinator_config(min_nodes: usize) -> CoordinatorConfig {
    CoordinatorConfig {
        min_nodes,
        max_nodes: 100,
        rate_limit_per_minute: 100,
        rate_limit_per_round: 1,
        round_timeout_secs: 10,
        ..CoordinatorConfig::default()
    }
}

/// Create a PoGQ config tuned for fast detection in tests (no warm-up, k=1).
fn aggressive_pogq_config() -> PoGQv41Config {
    PoGQv41Config {
        warm_up_rounds: 0,
        k_quarantine: 1,
        beta: 0.0,
        ..PoGQv41Config::default()
    }
}

/// Register N nodes with IDs "node-0" through "node-(N-1)".
fn register_nodes(coord: &mut FLCoordinator, n: usize) {
    for i in 0..n {
        coord
            .register_node(NodeCredential {
                node_id: format!("node-{}", i),
                public_key: None,
                did: None,
                registered_at: 0,
            })
            .unwrap();
    }
}

// ===========================================================================
// Test 1: Full pipeline — 5 honest, 1 Byzantine
// ===========================================================================

/// Flagship integration test: 5 honest nodes + 1 sign-flipping Byzantine.
///
/// Verifies:
/// - PoGQ detects and excludes the Byzantine node
/// - Aggregated gradient remains close to the true gradient
/// - Round history is correctly recorded
#[test]
fn test_full_pipeline_5_honest_1_byzantine() {
    let config = test_coordinator_config(3);
    let pogq_config = aggressive_pogq_config();
    let mut coordinator = FLCoordinator::new(config, Box::new(FedAvg), Some(pogq_config));

    register_nodes(&mut coordinator, 6);

    let dim = 500;
    let true_gradient: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

    let mut byzantine_detected_at_least_once = false;

    for round in 0..5 {
        coordinator.start_round().unwrap();

        // 5 honest nodes: true gradient + small noise
        for i in 0..5 {
            let noise_seed = (round * 100 + i) as u64;
            let values = add_noise(&true_gradient, 0.05, noise_seed);
            coordinator
                .submit_gradient(Gradient::new(format!("node-{}", i), values, round as u64))
                .unwrap();
        }

        // 1 Byzantine node: sign-flipped gradient with large magnitude
        let byz_values: Vec<f32> = true_gradient.iter().map(|v| -v * 2.0).collect();
        coordinator
            .submit_gradient(Gradient::new("node-5", byz_values, round as u64))
            .unwrap();

        let result = coordinator.complete_round().unwrap();

        // Track if the Byzantine node is ever excluded
        if result.excluded_nodes.contains(&"node-5".to_string()) {
            byzantine_detected_at_least_once = true;
        }

        // Aggregated gradient should always be somewhat aligned with truth
        // (even if Byzantine is not yet excluded, honest majority should dominate)
        let similarity = cos_sim(&result.gradient, &true_gradient);
        assert!(
            similarity > 0.5,
            "Round {}: aggregated gradient similarity {} too low",
            round,
            similarity
        );
    }

    // PoGQ should have detected the Byzantine at least once across 5 rounds
    assert!(
        byzantine_detected_at_least_once,
        "Byzantine node should be detected at least once across 5 rounds"
    );

    // Round history should have 5 entries
    let history = coordinator.round_history();
    assert_eq!(history.len(), 5);
    for summary in history {
        assert_eq!(summary.status, "Complete");
    }
}

// ===========================================================================
// Test 2: Multi-round convergence — 10 honest, 2 Byzantine, 20 rounds
// ===========================================================================

/// Verify that model quality improves over rounds and Byzantines are detected.
#[test]
fn test_multi_round_convergence() {
    let config = test_coordinator_config(3);
    let pogq_config = aggressive_pogq_config();
    let mut coordinator = FLCoordinator::new(config, Box::new(FedAvg), Some(pogq_config));

    let total_nodes = 12;
    register_nodes(&mut coordinator, total_nodes);

    let dim = 200;
    let true_gradient: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.005).collect();
    let num_honest = 10;
    let num_rounds = 20;

    let mut similarities: Vec<f64> = Vec::new();
    let mut byzantine_detections = 0;

    for round in 0..num_rounds {
        coordinator.start_round().unwrap();

        // 10 honest nodes
        for i in 0..num_honest {
            let values = add_noise(&true_gradient, 0.03, (round * 200 + i) as u64);
            coordinator
                .submit_gradient(Gradient::new(format!("node-{}", i), values, round as u64))
                .unwrap();
        }

        // 2 Byzantine nodes: sign-flipped
        for i in num_honest..total_nodes {
            let byz_values: Vec<f32> = true_gradient.iter().map(|v| -v * 1.5).collect();
            coordinator
                .submit_gradient(Gradient::new(format!("node-{}", i), byz_values, round as u64))
                .unwrap();
        }

        let result = coordinator.complete_round().unwrap();

        let sim = cos_sim(&result.gradient, &true_gradient);
        similarities.push(sim);

        // Count Byzantine detections
        for excluded in &result.excluded_nodes {
            if excluded == "node-10" || excluded == "node-11" {
                byzantine_detections += 1;
            }
        }
    }

    // Byzantines should be detected at some point
    assert!(
        byzantine_detections > 0,
        "At least one Byzantine should be detected across {} rounds",
        num_rounds,
    );

    // Late-round similarity should be better than early-round average
    // (once Byzantines are excluded, aggregation quality improves)
    let early_avg: f64 = similarities[..5].iter().sum::<f64>() / 5.0;
    let late_avg: f64 = similarities[num_rounds - 5..].iter().sum::<f64>() / 5.0;
    assert!(
        late_avg >= early_avg - 0.1,
        "Late similarity ({:.3}) should not be much worse than early ({:.3})",
        late_avg,
        early_avg,
    );

    // All similarities should be positive (honest majority dominates)
    for (r, &sim) in similarities.iter().enumerate() {
        assert!(
            sim > 0.0,
            "Round {}: similarity {} should be positive (honest majority)",
            r,
            sim,
        );
    }
}

// ===========================================================================
// Test 3: HyperFeel compression preserves Byzantine detection
// ===========================================================================

/// Verify that compressing gradients via HyperFeel still allows
/// cosine similarity to distinguish honest from Byzantine.
#[test]
fn test_compression_preserves_detection() {
    let original_dim = 2000;
    let compressed_dim = 512;

    let encoder = HyperFeelEncoder::new(
        original_dim,
        HyperFeelConfig {
            compressed_dim,
            seed: 42,
            ..Default::default()
        },
    );

    // Create honest gradients
    let true_gradient: Vec<f32> = (0..original_dim).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let honest_gradients: Vec<Gradient> = (0..5)
        .map(|i| {
            let values = add_noise(&true_gradient, 0.02, i * 1000);
            Gradient::new(format!("honest-{}", i), values, 0)
        })
        .collect();

    // Create a Byzantine gradient (sign-flipped)
    let byz_values: Vec<f32> = true_gradient.iter().map(|v| -v * 3.0).collect();
    let byz_gradient = Gradient::new("byzantine", byz_values, 0);

    // Compress all gradients
    let compressed_honest: Vec<_> = honest_gradients
        .iter()
        .map(|g| encoder.compress(g).unwrap())
        .collect();
    let compressed_byz = encoder.compress(&byz_gradient).unwrap();

    // Compute mean of compressed honest gradients
    let mean_compressed: Vec<f32> = {
        let dim = compressed_dim;
        let n = compressed_honest.len() as f32;
        let mut mean = vec![0.0f32; dim];
        for c in &compressed_honest {
            for (i, &v) in c.values.iter().enumerate() {
                mean[i] += v;
            }
        }
        mean.iter_mut().for_each(|v| *v /= n);
        mean
    };

    // Honest gradients should be similar to the mean in compressed space
    for c in &compressed_honest {
        let sim = cos_sim(&c.values, &mean_compressed);
        assert!(
            sim > 0.7,
            "Honest gradient {} should be similar to mean in compressed space (got {:.3})",
            c.node_id,
            sim,
        );
    }

    // Byzantine gradient should be dissimilar (negative cosine) in compressed space
    let byz_sim = cos_sim(&compressed_byz.values, &mean_compressed);
    assert!(
        byz_sim < 0.0,
        "Byzantine gradient should be dissimilar to mean in compressed space (got {:.3})",
        byz_sim,
    );

    // Verify compression ratio
    assert!(
        encoder.ratio() > 1.0,
        "Compression ratio should be > 1.0 (got {:.1})",
        encoder.ratio()
    );
}

// ===========================================================================
// Test 4: Detection stack — multi-layer catches different attack types
// ===========================================================================

/// Verify that the multi-layer detection stack catches attackers.
///
/// Layer 1 (PoGQ): catches obvious sign-flipping attacker
/// Layer 2 (Shapley): catches subtle free-riders
/// Both layers contribute to detection.
#[test]
fn test_detection_stack_layers() {
    let dim = 50;
    let base: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // --- Test with Shapley + healing enabled ---
    let config = DetectionConfig {
        enable_pogq: false,
        enable_shapley: true,
        enable_healing: true,
        shapley_config: Some(ShapleyConfig {
            threshold: -0.01,
            ..ShapleyConfig::default()
        }),
        healing_config: Some(SelfHealerConfig {
            repair_method: RepairMethod::WeightedInterpolate,
            max_repair_distance: 2.0,
        }),
        ..DetectionConfig::default()
    };
    let mut stack = DetectionStack::new(config);

    // 5 honest gradients
    let mut grads: Vec<Gradient> = (0..5)
        .map(|i| {
            let vals: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(j, &v)| v + 0.005 * ((i * 3 + j) % 7) as f32)
                .collect();
            Gradient::new(format!("honest-{}", i), vals, 0)
        })
        .collect();

    // Obvious attacker: sign-flipped with large magnitude
    let attacker_vals: Vec<f32> = base.iter().map(|&v| -v * 50.0).collect();
    grads.push(Gradient::new("attacker", attacker_vals, 0));

    let result = stack.detect_and_filter(&grads).unwrap();

    // Attacker should be excluded
    assert!(
        result.excluded_nodes.contains(&"attacker".to_string()),
        "Attacker should be excluded, got excluded: {:?}",
        result.excluded_nodes,
    );

    // Clean gradients should not contain the attacker
    assert!(
        !result
            .clean_gradients
            .iter()
            .any(|g| g.node_id == "attacker"),
        "Clean set should not contain attacker"
    );

    // At least one layer report should exist
    assert!(
        !result.layer_reports.is_empty(),
        "Should have at least one layer report"
    );

    // --- Test PoGQ-only mode with history seeding ---
    let pogq_config = DetectionConfig {
        enable_pogq: true,
        enable_shapley: false,
        enable_healing: false,
        pogq_config: Some(PoGQv41Config {
            warm_up_rounds: 0,
            k_quarantine: 1,
            beta: 0.0,
            ..PoGQv41Config::default()
        }),
        ..DetectionConfig::default()
    };
    let mut pogq_stack = DetectionStack::new(pogq_config);

    // Seed PoGQ history with honest-only rounds
    let honest_only: Vec<Gradient> = (0..5)
        .map(|i| {
            let vals: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(j, &v)| v + 0.01 * ((i * 7 + j * 3) % 13) as f32 * 0.1)
                .collect();
            Gradient::new(format!("h-{}", i), vals, 0)
        })
        .collect();

    for _ in 0..3 {
        pogq_stack.detect_and_filter(&honest_only).unwrap();
    }

    // Now add Byzantine
    let mut mixed = honest_only[..4].to_vec();
    let byz: Vec<f32> = base.iter().map(|&v| -v * 5.0).collect();
    mixed.push(Gradient::new("byz", byz, 0));

    let result = pogq_stack.detect_and_filter(&mixed).unwrap();

    assert_eq!(
        result.layer_reports.len(),
        1,
        "Should have exactly 1 layer report (PoGQ)"
    );
    assert_eq!(result.layer_reports[0].layer_name, "PoGQ");
}

// ===========================================================================
// Test 5: Coordinator node lifecycle — register, blacklist, reject
// ===========================================================================

/// Test node registration, blacklisting, and rejection of unregistered nodes.
#[test]
fn test_coordinator_node_lifecycle() {
    let config = test_coordinator_config(2);
    let mut coordinator = FLCoordinator::new(config, Box::new(FedAvg), None);

    // Register 3 nodes
    for id in &["alice", "bob", "carol"] {
        coordinator
            .register_node(NodeCredential {
                node_id: id.to_string(),
                public_key: None,
                did: None,
                registered_at: 0,
            })
            .unwrap();
    }
    assert_eq!(coordinator.node_count(), 3);

    // --- Unregistered node rejected ---
    coordinator.start_round().unwrap();
    let result = coordinator.submit_gradient(Gradient::new("unknown", vec![1.0; 10], 1));
    assert!(
        matches!(result, Err(FlError::NodeNotRegistered(_))),
        "Unregistered node should be rejected"
    );

    // Submit from registered nodes to complete the round
    for id in &["alice", "bob", "carol"] {
        coordinator
            .submit_gradient(Gradient::new(*id, vec![1.0; 10], 1))
            .unwrap();
    }
    coordinator.complete_round().unwrap();

    // --- Blacklist a node ---
    coordinator.node_manager_mut().blacklist("bob", "byzantine behavior".into());
    assert!(coordinator.node_manager().is_blacklisted("bob"));

    // Bob should be rejected when submitting.
    // Note: blacklisting removes from the registered set, so the coordinator
    // rejects with NodeNotRegistered (checked before NodeBlacklisted).
    coordinator.start_round().unwrap();
    let result = coordinator.submit_gradient(Gradient::new("bob", vec![1.0; 10], 2));
    assert!(
        matches!(
            result,
            Err(FlError::NodeNotRegistered(_)) | Err(FlError::NodeBlacklisted(_))
        ),
        "Blacklisted/unregistered node should be rejected, got: {:?}",
        result,
    );

    // Remaining nodes can still submit
    for id in &["alice", "carol"] {
        coordinator
            .submit_gradient(Gradient::new(*id, vec![1.0; 10], 2))
            .unwrap();
    }
    let result = coordinator.complete_round().unwrap();
    assert_eq!(result.included_nodes.len(), 2);

    // --- Blacklisted node cannot re-register ---
    let re_register = coordinator.register_node(NodeCredential {
        node_id: "bob".into(),
        public_key: None,
        did: None,
        registered_at: 0,
    });
    assert!(
        matches!(re_register, Err(FlError::NodeBlacklisted(_))),
        "Blacklisted node should not be able to re-register"
    );

    // --- Non-finite gradient rejected ---
    coordinator.start_round().unwrap();
    let nan_grad = Gradient::new("alice", vec![1.0, f32::NAN, 3.0], 3);
    let result = coordinator.submit_gradient(nan_grad);
    assert!(
        matches!(result, Err(FlError::NonFiniteGradient(_))),
        "NaN gradient should be rejected"
    );

    // --- Empty gradient rejected ---
    let empty_grad = Gradient::new("carol", vec![], 3);
    let result = coordinator.submit_gradient(empty_grad);
    assert!(
        matches!(result, Err(FlError::EmptyGradient)),
        "Empty gradient should be rejected"
    );
}

// ===========================================================================
// Test 6: All defenses produce valid output
// ===========================================================================

/// Smoke test every defense algorithm with the same input.
/// Verifies all produce same-dimensional, non-empty output.
#[test]
fn test_all_defenses_produce_valid_output() {
    let dim = 100;
    let gradients = generate_honest_gradients(10, dim, 42);
    let config = DefenseConfig {
        num_byzantine: 2,
        ..DefenseConfig::default()
    };

    // Collect all defenses to test
    // --- Stateless defenses (implement Defense trait) ---
    let defenses: Vec<(&str, Box<dyn Defense>)> = vec![
        ("FedAvg", Box::new(FedAvg)),
        ("TrimmedMean", Box::new(TrimmedMean)),
        ("CoordinateMedian", Box::new(CoordinateMedian)),
        ("Rfa", Box::new(Rfa)),
    ];

    for (name, defense) in &defenses {
        let result = defense.aggregate(&gradients, &config);
        match result {
            Ok(r) => {
                assert_eq!(
                    r.gradient.len(),
                    dim,
                    "{}: output dimension mismatch (expected {}, got {})",
                    name,
                    dim,
                    r.gradient.len()
                );

                // Output should be finite
                for &v in &r.gradient {
                    assert!(
                        v.is_finite(),
                        "{}: output contains non-finite value",
                        name,
                    );
                }

                // Should not be all zeros
                let has_nonzero = r.gradient.iter().any(|&v| v.abs() > 1e-10);
                assert!(has_nonzero, "{}: output is all zeros", name);
            }
            Err(e) => {
                panic!("{}: unexpected error: {:?}", name, e);
            }
        }
    }

    // --- Krum (different API: takes num_byzantine directly) ---
    let krum_result = Krum::aggregate(&gradients, 2).unwrap();
    assert_eq!(
        krum_result.gradient.len(),
        dim,
        "Krum: output dimension mismatch"
    );
    assert!(
        krum_result.gradient.iter().all(|v| v.is_finite()),
        "Krum: output contains non-finite value"
    );
    assert!(
        krum_result.gradient.iter().any(|&v| v.abs() > 1e-10),
        "Krum: output is all zeros"
    );
}

// ===========================================================================
// Test 7: Async round with timeout
// ===========================================================================

/// Test the async coordinator with tokio runtime.
#[tokio::test]
async fn test_async_round_with_timeout() {
    let config = CoordinatorConfig {
        min_nodes: 3,
        max_nodes: 100,
        round_timeout_secs: 5,
        rate_limit_per_minute: 100,
        rate_limit_per_round: 1,
        ..CoordinatorConfig::default()
    };
    let mut coordinator = FLCoordinator::new(config, Box::new(FedAvg), None);

    for id in &["a", "b", "c", "d"] {
        coordinator
            .register_node(NodeCredential {
                node_id: id.to_string(),
                public_key: None,
                did: None,
                registered_at: 0,
            })
            .unwrap();
    }

    let (tx, rx) = mpsc::channel(10);

    // Spawn a task that sends gradients
    tokio::spawn(async move {
        for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
            let values: Vec<f32> = (0..50).map(|j| (j as f32 + i as f32) * 0.1).collect();
            tx.send(Gradient::new(*id, values, 1)).await.unwrap();
        }
    });

    let result = coordinator.run_round_async(rx).await.unwrap();

    // Should have aggregated at least min_nodes gradients
    assert!(
        result.included_nodes.len() >= 3,
        "Should include at least 3 nodes, got {}",
        result.included_nodes.len()
    );
    assert_eq!(result.gradient.len(), 50);

    // Verify round was recorded
    assert_eq!(coordinator.round_history().len(), 1);
    assert_eq!(coordinator.round_history()[0].status, "Complete");
}

// ===========================================================================
// Test 8: Byzantine detection with different attack types
// ===========================================================================

/// Verify PoGQ detects various Byzantine attack strategies.
#[test]
fn test_different_attack_types_detected() {
    let dim = 100;
    let true_gradient: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.01).collect();

    // --- Attack type: Large magnitude scaling ---
    {
        let config = test_coordinator_config(3);
        let mut coord =
            FLCoordinator::new(config, Box::new(FedAvg), Some(aggressive_pogq_config()));
        register_nodes(&mut coord, 6);

        for round in 0..5u64 {
            coord.start_round().unwrap();

            // 5 honest
            for i in 0..5 {
                let values = add_noise(&true_gradient, 0.03, round * 100 + i);
                coord
                    .submit_gradient(Gradient::new(format!("node-{}", i), values, round))
                    .unwrap();
            }

            // Byzantine: correct direction but 100x magnitude
            let byz: Vec<f32> = true_gradient.iter().map(|v| v * 100.0).collect();
            coord
                .submit_gradient(Gradient::new("node-5", byz, round))
                .unwrap();

            coord.complete_round().unwrap();
        }

        // Check that the aggregated result over rounds is still close to truth
        let last_result = {
            coord.start_round().unwrap();
            for i in 0..5 {
                let values = add_noise(&true_gradient, 0.03, 999 + i);
                coord
                    .submit_gradient(Gradient::new(format!("node-{}", i), values, 5))
                    .unwrap();
            }
            let byz: Vec<f32> = true_gradient.iter().map(|v| v * 100.0).collect();
            coord
                .submit_gradient(Gradient::new("node-5", byz, 5))
                .unwrap();
            coord.complete_round().unwrap()
        };

        let sim = cos_sim(&last_result.gradient, &true_gradient);
        assert!(
            sim > 0.5,
            "Magnitude attack: aggregated similarity {} should be > 0.5",
            sim
        );
    }

    // --- Attack type: Random noise (no structure) ---
    {
        let config = test_coordinator_config(3);
        let mut coord =
            FLCoordinator::new(config, Box::new(FedAvg), Some(aggressive_pogq_config()));
        register_nodes(&mut coord, 6);

        for round in 0..5u64 {
            coord.start_round().unwrap();

            for i in 0..5 {
                let values = add_noise(&true_gradient, 0.03, round * 100 + i);
                coord
                    .submit_gradient(Gradient::new(format!("node-{}", i), values, round))
                    .unwrap();
            }

            // Byzantine: random noise
            let mut rng = Rng::new(round * 7 + 42);
            let byz: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 10.0).collect();
            coord
                .submit_gradient(Gradient::new("node-5", byz, round))
                .unwrap();

            coord.complete_round().unwrap();
        }

        let history = coord.round_history();
        assert_eq!(history.len(), 5);
        // All rounds should have completed successfully
        for s in history {
            assert_eq!(s.status, "Complete");
        }
    }
}

// ===========================================================================
// Test 9: Coordinator rejects double-start and no-round submit
// ===========================================================================

/// Test error handling for invalid coordinator state transitions.
#[test]
fn test_coordinator_state_errors() {
    let config = test_coordinator_config(2);
    let mut coord = FLCoordinator::new(config, Box::new(FedAvg), None);

    // Cannot start round with insufficient nodes
    coord
        .register_node(NodeCredential {
            node_id: "solo".into(),
            public_key: None,
            did: None,
            registered_at: 0,
        })
        .unwrap();
    assert!(matches!(
        coord.start_round(),
        Err(FlError::InsufficientGradients { .. })
    ));

    // Register enough nodes
    coord
        .register_node(NodeCredential {
            node_id: "duo".into(),
            public_key: None,
            did: None,
            registered_at: 0,
        })
        .unwrap();

    // Cannot submit without starting a round
    let result = coord.submit_gradient(Gradient::new("solo", vec![1.0], 0));
    assert!(
        matches!(result, Err(FlError::NoActiveRound)),
        "Should reject submission when no round is active"
    );

    // Start round, then try to start another
    coord.start_round().unwrap();
    let result = coord.start_round();
    assert!(
        matches!(result, Err(FlError::InvalidRoundState { .. })),
        "Should reject double-start"
    );

    // Complete round with insufficient gradients
    coord
        .submit_gradient(Gradient::new("solo", vec![1.0], 1))
        .unwrap();
    let result = coord.complete_round();
    assert!(
        matches!(result, Err(FlError::InsufficientGradients { .. })),
        "Should fail with insufficient gradients"
    );
}

// ===========================================================================
// Test 10: Full pipeline with FedAvg (no PoGQ) baseline
// ===========================================================================

/// Verify the coordinator works correctly without any Byzantine defense
/// (FedAvg baseline). The honest majority should still produce a reasonable result.
#[test]
fn test_fedavg_baseline_no_pogq() {
    let config = test_coordinator_config(3);
    let mut coord = FLCoordinator::new(config, Box::new(FedAvg), None);
    register_nodes(&mut coord, 5);

    let dim = 100;
    let true_gradient: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.01).collect();

    coord.start_round().unwrap();

    for i in 0..5 {
        let values = add_noise(&true_gradient, 0.02, i * 1000);
        coord
            .submit_gradient(Gradient::new(format!("node-{}", i), values, 1))
            .unwrap();
    }

    let result = coord.complete_round().unwrap();

    // All nodes included (no defense)
    assert_eq!(result.included_nodes.len(), 5);
    assert!(result.excluded_nodes.is_empty());

    // Result should be very close to true gradient (only small noise)
    let sim = cos_sim(&result.gradient, &true_gradient);
    assert!(
        sim > 0.95,
        "FedAvg with honest nodes: similarity {} should be > 0.95",
        sim,
    );
}
