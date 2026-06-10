// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy Budget Tracking Demo
//!
//! Demonstrates Rényi Differential Privacy (RDP) composition across FL rounds:
//! - Shows epsilon growth under different noise levels (high/moderate/low privacy)
//! - Tracks cumulative privacy loss via RDP → (ε, δ)-DP conversion
//! - Demonstrates budget exhaustion detection
//! - Measures accuracy impact of DP noise on aggregation quality
//!
//! Run: `cargo run --example benchmark_privacy_budget --release`

use mycelix_fl_core::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

fn main() {
    println!("=== Privacy Budget Tracking Demo ===\n");

    // === Part 1: Epsilon Growth Across Rounds ===
    println!("--- Part 1: Epsilon Growth (delta = 1e-5) ---\n");

    let presets = [
        ("high_privacy", DifferentialPrivacyConfig::high_privacy()),
        ("moderate", DifferentialPrivacyConfig::moderate_privacy()),
        ("low_privacy", DifferentialPrivacyConfig::low_privacy()),
    ];

    let target_delta = 1e-5;
    let max_rounds = 100;

    println!(
        "{:<16} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Preset", "Sigma", "ε@10", "ε@25", "ε@50", "ε@100"
    );
    println!("{:-<80}", "");

    for (name, config) in &presets {
        let sigma = config.sigma();
        let mut tracker = RdpBudgetTracker::new(target_delta);

        let mut eps_at = [0.0f64; 4];
        let checkpoints = [10, 25, 50, 100];

        for round in 1..=max_rounds {
            tracker.record_round(sigma);
            for (idx, &cp) in checkpoints.iter().enumerate() {
                if round == cp {
                    eps_at[idx] = tracker.epsilon();
                }
            }
        }

        println!(
            "{:<16} {:>8.3} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            name, sigma, eps_at[0], eps_at[1], eps_at[2], eps_at[3]
        );
    }

    // === Part 2: Round-by-Round Epsilon Curve ===
    println!("\n--- Part 2: Round-by-Round Epsilon (moderate privacy) ---\n");

    let moderate = DifferentialPrivacyConfig::moderate_privacy();
    let sigma = moderate.sigma();
    let mut tracker = RdpBudgetTracker::new(target_delta);

    println!("{:<8} {:>12} {:>15}", "Round", "Epsilon", "Budget Status");
    println!("{:-<40}", "");

    let target_epsilon = 10.0;
    let mut exhaustion_round = None;

    for round in 1..=max_rounds {
        tracker.record_round(sigma);
        let eps = tracker.epsilon();

        if round <= 10 || round % 10 == 0 || (exhaustion_round.is_none() && eps > target_epsilon) {
            let status = if eps > target_epsilon {
                "EXHAUSTED"
            } else {
                "OK"
            };
            println!("{:<8} {:>12.4} {:>15}", round, eps, status);
        }

        if exhaustion_round.is_none() && eps > target_epsilon {
            exhaustion_round = Some(round);
        }
    }

    // === Part 3: Aggregation Quality Under DP ===
    println!("\n--- Part 3: Aggregation Quality Under DP Noise ---\n");

    let mut rng = StdRng::seed_from_u64(42);
    let dim = 100;

    // Generate 10 honest updates with known mean
    let mean_val = 0.5;
    let mut base_updates = Vec::new();
    for i in 0..10 {
        let gradients: Vec<f32> = (0..dim)
            .map(|_| mean_val + rng.gen_range(-0.1..0.1))
            .collect();
        base_updates.push(GradientUpdate::new(
            format!("node_{}", i),
            1,
            gradients,
            100,
            0.5,
        ));
    }

    let reputations: HashMap<String, f32> = (0..10).map(|i| (format!("node_{}", i), 0.8)).collect();

    // Aggregate without DP (baseline)
    let no_dp_config = PipelineConfig {
        dp_config: None,
        ..PipelineConfig::performance()
    };
    let mut no_dp_pipeline = UnifiedPipeline::new(no_dp_config);
    let baseline = no_dp_pipeline
        .aggregate(&base_updates, &reputations)
        .unwrap();
    let baseline_mean: f32 = baseline.aggregated.gradients.iter().sum::<f32>() / dim as f32;

    println!(
        "{:<16} {:>12} {:>12} {:>12} {:>12}",
        "Privacy Level", "Mean", "MSE", "Cosine Sim", "Max Drift"
    );
    println!("{:-<60}", "");

    // No DP baseline
    println!(
        "{:<16} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "none (baseline)", baseline_mean, 0.0, 1.0, 0.0
    );

    for (name, dp_config) in &presets {
        let config = PipelineConfig {
            dp_config: Some(*dp_config),
            ..PipelineConfig::performance()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        // Average over 5 trials (DP is stochastic)
        let mut total_mse = 0.0f64;
        let mut total_cos = 0.0f64;
        let mut total_drift = 0.0f64;
        let n_trials = 5;

        for _ in 0..n_trials {
            let result = pipeline.aggregate(&base_updates, &reputations).unwrap();
            let dp_grads = &result.aggregated.gradients;

            // MSE vs baseline
            let mse: f64 = dp_grads
                .iter()
                .zip(baseline.aggregated.gradients.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / dim as f64;
            total_mse += mse;

            // Cosine similarity vs baseline
            let dot: f64 = dp_grads
                .iter()
                .zip(baseline.aggregated.gradients.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let norm_a: f64 = dp_grads
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm_b: f64 = baseline
                .aggregated
                .gradients
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            };
            total_cos += cos;

            // Max absolute drift
            let max_d: f64 = dp_grads
                .iter()
                .zip(baseline.aggregated.gradients.iter())
                .map(|(a, b)| ((*a - *b) as f64).abs())
                .fold(0.0, f64::max);
            total_drift += max_d;
        }

        let avg_mse = total_mse / n_trials as f64;
        let avg_cos = total_cos / n_trials as f64;
        let avg_drift = total_drift / n_trials as f64;
        let dp_mean: f32 = {
            let result = pipeline.aggregate(&base_updates, &reputations).unwrap();
            result.aggregated.gradients.iter().sum::<f32>() / dim as f32
        };

        println!(
            "{:<16} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            name, dp_mean, avg_mse, avg_cos, avg_drift
        );
    }

    // === Part 4: Budget Exhaustion Comparison ===
    println!(
        "\n--- Part 4: Rounds Until Budget Exhaustion (target ε={}) ---\n",
        target_epsilon
    );

    println!(
        "{:<16} {:>8} {:>15} {:>15}",
        "Preset", "Sigma", "Rounds to ε=10", "Rounds to ε=50"
    );
    println!("{:-<58}", "");

    for (name, config) in &presets {
        let sigma = config.sigma();
        let mut tracker10 = RdpBudgetTracker::new(target_delta);
        let mut tracker50 = RdpBudgetTracker::new(target_delta);
        let mut rounds_10 = None;
        let mut rounds_50 = None;

        for round in 1..=10000 {
            tracker10.record_round(sigma);
            tracker50.record_round(sigma);
            if rounds_10.is_none() && tracker10.epsilon() > 10.0 {
                rounds_10 = Some(round);
            }
            if rounds_50.is_none() && tracker50.epsilon() > 50.0 {
                rounds_50 = Some(round);
            }
            if rounds_10.is_some() && rounds_50.is_some() {
                break;
            }
        }

        let r10 = rounds_10
            .map(|r| format!("{}", r))
            .unwrap_or_else(|| ">10000".into());
        let r50 = rounds_50
            .map(|r| format!("{}", r))
            .unwrap_or_else(|| ">10000".into());
        println!("{:<16} {:>8.3} {:>15} {:>15}", name, sigma, r10, r50);
    }

    // === Verification ===
    println!("\n=== Verification ===");
    let mut passed = 0;
    let total_tests = 5;

    // Test 1: Epsilon increases with rounds
    let mut tracker = RdpBudgetTracker::new(target_delta);
    tracker.record_round(1.1);
    let eps1 = tracker.epsilon();
    for _ in 0..9 {
        tracker.record_round(1.1);
    }
    let eps10 = tracker.epsilon();
    if eps10 > eps1 && eps1 > 0.0 {
        println!(
            "[PASS] Test 1: Epsilon increases monotonically ({:.4} → {:.4})",
            eps1, eps10
        );
        passed += 1;
    } else {
        println!("[FAIL] Test 1: Epsilon not monotonically increasing");
    }

    // Test 2: Higher sigma → slower epsilon growth (use explicit sigma values)
    let mut track_high_sigma = RdpBudgetTracker::new(target_delta);
    let mut track_low_sigma = RdpBudgetTracker::new(target_delta);
    for _ in 0..50 {
        track_high_sigma.record_round(2.0); // High noise
        track_low_sigma.record_round(0.5); // Low noise
    }
    if track_high_sigma.epsilon() < track_low_sigma.epsilon() {
        println!(
            "[PASS] Test 2: Higher sigma → slower ε growth (σ=2.0: {:.4} < σ=0.5: {:.4})",
            track_high_sigma.epsilon(),
            track_low_sigma.epsilon()
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 2: Expected σ=2.0 ε < σ=0.5 ε ({:.4} vs {:.4})",
            track_high_sigma.epsilon(),
            track_low_sigma.epsilon()
        );
    }

    // Test 3: Budget exhaustion detection works
    let mut tracker = RdpBudgetTracker::new(target_delta);
    assert!(!tracker.is_exhausted(10.0));
    for _ in 0..1000 {
        tracker.record_round(DifferentialPrivacyConfig::low_privacy().sigma());
    }
    if tracker.is_exhausted(10.0) {
        println!("[PASS] Test 3: Budget exhaustion detected after 1000 low-privacy rounds");
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 3: Should be exhausted after 1000 rounds, ε={:.4}",
            tracker.epsilon()
        );
    }

    // Test 4: DP noise degrades but doesn't destroy aggregation quality
    let high_dp_config = PipelineConfig {
        dp_config: Some(DifferentialPrivacyConfig::high_privacy()),
        ..PipelineConfig::performance()
    };
    let mut high_dp_pipeline = UnifiedPipeline::new(high_dp_config);
    let dp_result = high_dp_pipeline
        .aggregate(&base_updates, &reputations)
        .unwrap();
    let dp_mean: f32 = dp_result.aggregated.gradients.iter().sum::<f32>() / dim as f32;
    // With high privacy, mean should still be within 2.0 of baseline (very loose bound)
    if (dp_mean - baseline_mean).abs() < 2.0 {
        println!(
            "[PASS] Test 4: High-privacy mean {:.4} within 2.0 of baseline {:.4}",
            dp_mean, baseline_mean
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 4: High-privacy mean {:.4} too far from baseline {:.4}",
            dp_mean, baseline_mean
        );
    }

    // Test 5: Tiny noise preserves aggregation quality closely
    // Use very small noise (sigma=0.01) so the signal clearly dominates
    let tiny_dp_config = PipelineConfig {
        dp_config: Some(DifferentialPrivacyConfig::new(10.0, 0.01)), // clip=10.0, noise=0.01
        ..PipelineConfig::performance()
    };
    let mut tiny_dp_pipeline = UnifiedPipeline::new(tiny_dp_config);
    let tiny_result = tiny_dp_pipeline
        .aggregate(&base_updates, &reputations)
        .unwrap();
    let cos: f64 = {
        let a = &tiny_result.aggregated.gradients;
        let b = &baseline.aggregated.gradients;
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        }
    };
    if cos > 0.9 {
        println!(
            "[PASS] Test 5: Tiny-noise cosine similarity {:.4} > 0.9",
            cos
        );
        passed += 1;
    } else {
        println!(
            "[FAIL] Test 5: Tiny-noise cosine similarity too low: {:.4}",
            cos
        );
    }

    println!("\n=== RESULTS: {}/{} passed ===", passed, total_tests);
    if passed < total_tests {
        std::process::exit(1);
    }
}
