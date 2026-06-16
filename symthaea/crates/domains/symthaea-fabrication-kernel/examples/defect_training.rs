// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Training demo and benchmark for the DefectPredictor.
//!
//! Generates 200 synthetic training samples with a known quality function,
//! trains the predictor online, then evaluates on 5 test cases and prints
//! feature importance and adjustment suggestions.
//!
//! Run with:
//! ```sh
//! cargo run -p symthaea-fabrication-kernel --example defect_training
//! ```

use symthaea_fabrication_kernel::defect_prediction::{DefectPredictor, PredictionInput};

/// Deterministic pseudo-random noise for sample `i`.
fn noise(i: usize) -> f64 {
    ((i * 7 + 13) % 100) as f64 / 1000.0 - 0.05
}

/// Ground-truth quality function used to generate training labels.
fn true_quality(input: &PredictionInput, i: usize) -> f64 {
    let anomaly_free = 1.0 / (1.0 + input.anomaly_count as f64);
    let layer_density = (input.layer_count.min(500) as f64) / 500.0;

    let q = 0.30 * input.tolerance
        + 0.25 * input.surface_quality
        + 0.20 * input.throughput
        + 0.15 * (1.0 - input.energy_cost)
        + 0.05 * anomaly_free
        + 0.05 * layer_density
        + noise(i);

    q.clamp(0.0, 1.0)
}

/// Generate a synthetic training sample at index `i`.
fn generate_sample(i: usize) -> PredictionInput {
    // Sweep parameter space deterministically
    let t = (i as f64) / 200.0;
    let phase = (i as f64) * 0.1;

    PredictionInput {
        tolerance: (0.5 + 0.5 * (phase).sin()).clamp(0.0, 1.0),
        surface_quality: (0.3 + 0.7 * t).clamp(0.0, 1.0),
        throughput: (0.2 + 0.6 * (phase * 0.7).cos().abs()).clamp(0.0, 1.0),
        energy_cost: (0.8 - 0.6 * t).clamp(0.0, 1.0),
        anomaly_count: ((10.0 * (1.0 - t)).round() as u32).min(20),
        layer_count: (50.0 + 400.0 * t) as u32,
    }
}

fn main() {
    println!("=== DefectPredictor Training Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create predictor
    // -----------------------------------------------------------------------
    let mut predictor = DefectPredictor::new();
    println!("Created new DefectPredictor (6 features, equal initial weights)\n");

    // -----------------------------------------------------------------------
    // 2. Generate 200 synthetic samples and train
    // -----------------------------------------------------------------------
    println!("Training on 200 synthetic samples...\n");

    for i in 0..200 {
        let input = generate_sample(i);
        let quality = true_quality(&input, i);
        predictor.train(&input, quality);

        if (i + 1) % 50 == 0 {
            // Evaluate current accuracy on this sample
            let pred = predictor.predict(&input);
            let error = (pred.predicted_quality - quality).abs();
            println!(
                "  [{:>3}/200] confidence={:.2}  last_error={:.4}  training_count={}",
                i + 1,
                pred.confidence,
                error,
                predictor.training_count,
            );
        }
    }

    println!("\nTraining complete.\n");

    // -----------------------------------------------------------------------
    // 3. Predict on 5 test cases
    // -----------------------------------------------------------------------
    let test_cases: Vec<(&str, PredictionInput)> = vec![
        (
            "Perfect conditions (all high)",
            PredictionInput {
                tolerance: 0.95,
                surface_quality: 0.95,
                throughput: 0.90,
                energy_cost: 0.05,
                anomaly_count: 0,
                layer_count: 400,
            },
        ),
        (
            "Poor tolerance only",
            PredictionInput {
                tolerance: 0.10,
                surface_quality: 0.90,
                throughput: 0.85,
                energy_cost: 0.10,
                anomaly_count: 0,
                layer_count: 350,
            },
        ),
        (
            "High anomaly count",
            PredictionInput {
                tolerance: 0.80,
                surface_quality: 0.80,
                throughput: 0.75,
                energy_cost: 0.20,
                anomaly_count: 15,
                layer_count: 300,
            },
        ),
        (
            "Low throughput + high energy",
            PredictionInput {
                tolerance: 0.70,
                surface_quality: 0.70,
                throughput: 0.10,
                energy_cost: 0.95,
                anomaly_count: 2,
                layer_count: 200,
            },
        ),
        (
            "Mixed conditions",
            PredictionInput {
                tolerance: 0.50,
                surface_quality: 0.60,
                throughput: 0.50,
                energy_cost: 0.50,
                anomaly_count: 3,
                layer_count: 150,
            },
        ),
    ];

    println!("--- Predictions on 5 Test Cases ---\n");

    let mut worst_idx = 0;
    let mut worst_quality = f64::MAX;

    for (idx, (label, input)) in test_cases.iter().enumerate() {
        let output = predictor.predict(input);
        println!("Test {}: {}", idx + 1, label);
        println!(
            "  Predicted quality: {:.4}  Confidence: {:.2}",
            output.predicted_quality, output.confidence
        );

        if output.risk_factors.is_empty() {
            println!("  Risk factors: none");
        } else {
            println!("  Risk factors:");
            for rf in &output.risk_factors {
                println!(
                    "    - {} (contribution={:.4}, severity={:.2})",
                    rf.name, rf.contribution, rf.severity
                );
            }
        }
        println!();

        if output.predicted_quality < worst_quality {
            worst_quality = output.predicted_quality;
            worst_idx = idx;
        }
    }

    // -----------------------------------------------------------------------
    // 4. Feature importance
    // -----------------------------------------------------------------------
    println!("--- Feature Importance Ranking ---\n");

    let importance = predictor.feature_importance();
    for (rank, (name, weight)) in importance.iter().enumerate() {
        let bar_len = (weight.abs() * 40.0).round() as usize;
        let bar: String = "#".repeat(bar_len.min(40));
        println!(
            "  {:>2}. {:<20} weight={:+.4}  |{}",
            rank + 1,
            name,
            weight,
            bar
        );
    }

    // -----------------------------------------------------------------------
    // 5. Adjustments for worst prediction
    // -----------------------------------------------------------------------
    let (worst_label, worst_input) = &test_cases[worst_idx];
    let target = 0.8;

    println!(
        "\n--- Suggested Adjustments for Worst Case ---\n\
         Test: {}\n\
         Predicted quality: {:.4}  Target: {:.2}\n",
        worst_label, worst_quality, target
    );

    let adjustments = predictor.suggest_adjustments(worst_input, target);
    if adjustments.is_empty() {
        println!("  (no adjustments needed — already at or above target)");
    } else {
        for adj in &adjustments {
            println!(
                "  {:<20} {:.3} -> {:.3}  (expected improvement: {:.4})",
                adj.parameter, adj.current_value, adj.suggested_value, adj.expected_improvement
            );
        }
    }

    println!("\n=== Demo complete ===");
}
