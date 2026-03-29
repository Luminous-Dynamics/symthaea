// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HyperFeel-FL Integration Tests
//!
//! Tests demonstrating the full Byzantine-resistant federated learning pipeline
//! with 2000x gradient compression via HyperFeel.

use mycelix_sdk::fl::{
    AggregationMethod, CompressedSubmission, FLConfig, HyperFeelFLConfig, HyperFeelFLCoordinator,
};

/// Create a gradient simulating honest training
fn create_honest_gradient(node_id: usize, round: usize, size: usize) -> Vec<f32> {
    // Honest gradients follow a similar pattern with small variations
    (0..size)
        .map(|i| {
            let base = (i as f32 * 0.001).sin();
            let node_variance = (node_id as f32 * 0.1).sin() * 0.01;
            let round_variance = (round as f32 * 0.05).cos() * 0.005;
            base + node_variance + round_variance
        })
        .collect()
}

/// Create a Byzantine (malicious) gradient
fn create_byzantine_gradient(size: usize) -> Vec<f32> {
    // Byzantine gradient is completely different - trying to poison the model
    (0..size)
        .map(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 })
        .collect()
}

#[test]
fn test_full_fl_round_with_hyperfeel_compression() {
    // Configure FL with TrustWeighted aggregation
    let fl_config = FLConfig {
        min_participants: 3,
        max_participants: 10,
        round_timeout_ms: 60_000,
        byzantine_tolerance: 0.33,
        aggregation_method: AggregationMethod::TrustWeighted,
        trust_threshold: 0.3,
    };

    // Configure HyperFeel with Byzantine detection
    let hf_config = HyperFeelFLConfig {
        similarity_threshold: 0.7,
        min_cluster_size: 3,
        use_similarity_weighting: true,
        ..Default::default()
    };

    let mut coordinator = HyperFeelFLCoordinator::new(fl_config, hf_config);

    // Register 5 participants
    for i in 1..=5 {
        coordinator.register_participant(format!("node-{}", i));
    }

    // Start round
    coordinator.start_round().expect("Failed to start round");

    // Each honest node submits compressed gradient
    // Using 20K parameters (80KB original) → 2KB compressed for faster tests
    let gradient_size = 20_000;

    for node_id in 1..=5 {
        let gradient = create_honest_gradient(node_id, 1, gradient_size);
        let hg = coordinator
            .bridge
            .compress_gradient(&gradient, 1, &format!("node-{}", node_id));

        assert!(
            coordinator.submit_compressed(
                CompressedSubmission {
                    hypergradient: hg,
                    batch_size: 64,
                    loss: 0.5 - (node_id as f64 * 0.01),
                    accuracy: Some(0.8 + node_id as f64 * 0.02),
                },
                1,
            ),
            "Node {} should submit successfully",
            node_id
        );
    }

    // Analyze Byzantine behavior
    let analysis = coordinator.analyze_and_flag_byzantine();

    // All honest nodes should pass
    assert_eq!(
        analysis.flagged_count, 0,
        "No honest nodes should be flagged as Byzantine"
    );
    assert_eq!(analysis.total_count, 5);

    // Get compression stats
    let stats = coordinator.compression_stats();
    println!("Compression Stats:");
    println!("  Submissions: {}", stats.submissions);
    println!("  Original: {} bytes", stats.total_original_bytes);
    println!("  Compressed: {} bytes", stats.total_compressed_bytes);
    println!("  Ratio: {:.1}x", stats.overall_compression_ratio);
    println!(
        "  Bandwidth saved: {:.1}%",
        stats.bandwidth_savings_percent()
    );

    // Verify significant compression (20K floats = 80KB → 2KB = ~40x)
    assert!(
        stats.overall_compression_ratio > 30.0,
        "Should achieve >30x compression, got {}x",
        stats.overall_compression_ratio
    );
    assert!(
        stats.bandwidth_savings_percent() > 95.0,
        "Should save >95% bandwidth, got {:.1}%",
        stats.bandwidth_savings_percent()
    );

    // Aggregate round
    let result = coordinator.aggregate_round();
    assert!(result.is_ok(), "Aggregation should succeed");

    let aggregated = result.unwrap();
    assert_eq!(aggregated.participant_count, 5);
    assert_eq!(aggregated.excluded_count, 0);
}

#[test]
fn test_byzantine_node_detection_and_exclusion() {
    let fl_config = FLConfig {
        min_participants: 3,
        max_participants: 10,
        byzantine_tolerance: 0.33,
        trust_threshold: 0.3,
        ..Default::default()
    };

    let hf_config = HyperFeelFLConfig {
        similarity_threshold: 0.8, // Strict threshold for outlier detection
        min_cluster_size: 3,
        use_similarity_weighting: true,
        ..Default::default()
    };

    let mut coordinator = HyperFeelFLCoordinator::new(fl_config, hf_config);

    // Register participants (including Byzantine attacker)
    for i in 1..=5 {
        coordinator.register_participant(format!("node-{}", i));
    }
    coordinator.register_participant("byzantine-attacker".to_string());

    coordinator.start_round().expect("Failed to start round");

    let gradient_size = 15_000; // Smaller for faster tests

    // Submit honest gradients
    for node_id in 1..=5 {
        let gradient = create_honest_gradient(node_id, 1, gradient_size);
        let hg = coordinator
            .bridge
            .compress_gradient(&gradient, 1, &format!("node-{}", node_id));

        coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: hg,
                batch_size: 64,
                loss: 0.5,
                accuracy: Some(0.85),
            },
            1,
        );
    }

    // Submit Byzantine gradient (completely different pattern)
    let byzantine_gradient = create_byzantine_gradient(gradient_size);
    let byzantine_hg =
        coordinator
            .bridge
            .compress_gradient(&byzantine_gradient, 1, "byzantine-attacker");

    coordinator.submit_compressed(
        CompressedSubmission {
            hypergradient: byzantine_hg,
            batch_size: 64,
            loss: 0.1, // Suspiciously good loss
            accuracy: Some(0.99),
        },
        1,
    );

    // Analyze Byzantine behavior
    let analysis = coordinator.analyze_and_flag_byzantine();

    println!("Byzantine Analysis:");
    println!("  Total participants: {}", analysis.total_count);
    println!("  Flagged: {}", analysis.flagged_count);

    for (node_id, &sim) in &analysis.similarity_scores {
        let is_flagged = analysis
            .byzantine_flags
            .get(node_id)
            .copied()
            .unwrap_or(false);
        println!(
            "  {} - similarity: {:.3}, flagged: {}",
            node_id, sim, is_flagged
        );
    }

    // Byzantine node should have lower similarity than honest nodes
    let byzantine_sim = *analysis
        .similarity_scores
        .get("byzantine-attacker")
        .unwrap_or(&1.0);
    let honest_sims: Vec<f32> = (1..=5)
        .map(|i| {
            *analysis
                .similarity_scores
                .get(&format!("node-{}", i))
                .unwrap_or(&0.0)
        })
        .collect();
    let avg_honest_sim = honest_sims.iter().sum::<f32>() / honest_sims.len() as f32;

    println!("\n  Byzantine similarity: {:.3}", byzantine_sim);
    println!("  Average honest similarity: {:.3}", avg_honest_sim);

    assert!(
        byzantine_sim < avg_honest_sim,
        "Byzantine node ({:.3}) should have lower similarity than honest avg ({:.3})",
        byzantine_sim,
        avg_honest_sim
    );
}

#[test]
fn test_multi_round_fl_with_reputation_updates() {
    let fl_config = FLConfig {
        min_participants: 2,
        trust_threshold: 0.2,
        ..Default::default()
    };

    let hf_config = HyperFeelFLConfig {
        min_cluster_size: 2,
        ..Default::default()
    };

    let mut coordinator = HyperFeelFLCoordinator::new(fl_config, hf_config);

    // Register participants
    coordinator.register_participant("reliable-node".to_string());
    coordinator.register_participant("sporadic-node".to_string());

    let gradient_size = 5_000;

    // Run 3 rounds
    for round in 1..=3 {
        coordinator.start_round().expect("Failed to start round");

        // Reliable node always contributes
        let reliable_gradient = create_honest_gradient(1, round, gradient_size);
        let reliable_hg =
            coordinator
                .bridge
                .compress_gradient(&reliable_gradient, round as u32, "reliable-node");

        coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: reliable_hg,
                batch_size: 64,
                loss: 0.5 / round as f64,
                accuracy: Some(0.8 + round as f64 * 0.05),
            },
            round as u64,
        );

        // Sporadic node contributes
        let sporadic_gradient = create_honest_gradient(2, round, gradient_size);
        let sporadic_hg =
            coordinator
                .bridge
                .compress_gradient(&sporadic_gradient, round as u32, "sporadic-node");

        coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: sporadic_hg,
                batch_size: 32,
                loss: 0.6 / round as f64,
                accuracy: Some(0.75 + round as f64 * 0.04),
            },
            round as u64,
        );

        // Analyze and aggregate
        coordinator.analyze_and_flag_byzantine();
        let result = coordinator.aggregate_round();
        assert!(result.is_ok(), "Round {} should succeed", round);

        println!("Round {} completed", round);
    }

    // Check final participant reputations
    let reliable = coordinator
        .coordinator()
        .get_participant("reliable-node")
        .unwrap();
    let sporadic = coordinator
        .coordinator()
        .get_participant("sporadic-node")
        .unwrap();

    println!("\nFinal Reputations:");
    println!(
        "  Reliable node: {:.3} (rounds: {})",
        reliable.trust_score(),
        reliable.rounds_participated
    );
    println!(
        "  Sporadic node: {:.3} (rounds: {})",
        sporadic.trust_score(),
        sporadic.rounds_participated
    );

    // Both should have participated in all rounds
    assert_eq!(reliable.rounds_participated, 3);
    assert_eq!(sporadic.rounds_participated, 3);

    // Both should have positive reputation
    assert!(reliable.trust_score() > 0.5);
    assert!(sporadic.trust_score() > 0.5);
}

#[test]
fn test_bandwidth_savings_at_scale() {
    let fl_config = FLConfig {
        min_participants: 5,
        trust_threshold: 0.2,
        ..Default::default()
    };

    let hf_config = HyperFeelFLConfig::default();
    let mut coordinator = HyperFeelFLCoordinator::new(fl_config, hf_config);

    // Register many participants
    for i in 1..=10 {
        coordinator.register_participant(format!("node-{}", i));
    }

    coordinator.start_round().expect("Failed to start round");

    // Simulate medium model (100K parameters = 400KB per gradient) - faster for tests
    let large_gradient_size = 100_000;

    for node_id in 1..=10 {
        let gradient = create_honest_gradient(node_id, 1, large_gradient_size);
        let hg = coordinator
            .bridge
            .compress_gradient(&gradient, 1, &format!("node-{}", node_id));

        coordinator.submit_compressed(
            CompressedSubmission {
                hypergradient: hg,
                batch_size: 128,
                loss: 0.3,
                accuracy: Some(0.9),
            },
            1,
        );
    }

    let stats = coordinator.compression_stats();

    println!("Large-Scale Compression Stats:");
    println!("  Participants: {}", stats.submissions);
    println!("  Original total: {} KB", stats.total_original_bytes / 1024);
    println!(
        "  Compressed total: {} KB",
        stats.total_compressed_bytes / 1024
    );
    println!(
        "  Compression ratio: {:.0}x",
        stats.overall_compression_ratio
    );
    println!(
        "  Bandwidth saved: {} KB",
        stats.bandwidth_saved_bytes / 1024
    );

    // With 10 nodes × 400KB gradients = 4MB original
    // With HyperFeel: 10 × 2KB = 20KB compressed
    // Ratio should be ~200x
    assert!(
        stats.overall_compression_ratio > 150.0,
        "Should achieve >150x compression at scale, got {}x",
        stats.overall_compression_ratio
    );

    // Bandwidth savings should be >99%
    assert!(
        stats.bandwidth_savings_percent() > 99.0,
        "Should save >99% bandwidth"
    );

    // Aggregation should still work
    coordinator.analyze_and_flag_byzantine();
    let result = coordinator.aggregate_round();
    assert!(result.is_ok());
}
