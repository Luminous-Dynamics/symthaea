// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learning curve analysis for N-back and WCST.
//!
//! Runs multiple blocks with different seeds to simulate session progression
//! and measures whether performance improves over time.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example learning_curves
//!   cargo run -p symthaea-psych-bench --example learning_curves -- --csv

use symthaea_psych_bench::benchmarks::executive::WisconsinCardSortingBenchmark;
use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

/// Simple OLS slope: returns the slope of the least-squares line through `values`.
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = values.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean) * (x - x_mean);
    }

    if den.abs() < 1e-15 { 0.0 } else { num / den }
}

/// Ratio of final block value to initial block value.
fn final_vs_initial(values: &[f64]) -> f64 {
    if values.is_empty() || values[0].abs() < 1e-15 {
        return 0.0;
    }
    values[values.len() - 1] / values[0]
}

fn run_nback_learning_curve(
    base_config: &BenchmarkConfig,
    num_blocks: usize,
    trials_per_block: usize,
) -> Vec<f64> {
    let bench = NBackBenchmark;
    let mut per_block = Vec::with_capacity(num_blocks);

    for block in 0..num_blocks {
        let mut config = base_config.clone();
        config.seed = base_config.seed.wrapping_add(block as u64 * 1000);
        config.trials_per_condition = trials_per_block;
        let result = bench.run(&config);
        // Use nback_2 accuracy as the primary metric
        let acc = result
            .metrics
            .get("nback_2::accuracy")
            .map(|m| m.mean)
            .unwrap_or(0.0);
        per_block.push(acc);
    }

    per_block
}

fn run_wcst_learning_curve(base_config: &BenchmarkConfig, num_blocks: usize) -> Vec<f64> {
    let bench = WisconsinCardSortingBenchmark;
    let mut per_block = Vec::with_capacity(num_blocks);

    for block in 0..num_blocks {
        let mut config = base_config.clone();
        config.seed = base_config.seed.wrapping_add(block as u64 * 1000);
        let result = bench.run(&config);
        let cats = result
            .metrics
            .get("categories_completed")
            .map(|m| m.mean)
            .unwrap_or(0.0);
        per_block.push(cats);
    }

    per_block
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_csv = args.iter().any(|a| a == "--csv");

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 10,
        ..Default::default()
    };

    eprintln!("Running N-back learning curve (10 blocks)...");
    let nback_curve = run_nback_learning_curve(&config, 10, 10);

    eprintln!("Running WCST learning curve (5 blocks)...");
    let wcst_curve = run_wcst_learning_curve(&config, 5);

    if output_csv {
        println!("benchmark,block,value");
        for (i, v) in nback_curve.iter().enumerate() {
            println!("NBack_2,{},{:.4}", i + 1, v);
        }
        for (i, v) in wcst_curve.iter().enumerate() {
            println!("WCST_categories,{},{:.4}", i + 1, v);
        }
    } else {
        println!("\nN-back (n=2) Learning Curve");
        println!("{:<8} {:<10}", "Block", "Accuracy");
        println!("{}", "-".repeat(20));
        for (i, v) in nback_curve.iter().enumerate() {
            println!("{:<8} {:.4}", i + 1, v);
        }
        let nback_slope = linear_slope(&nback_curve);
        let nback_ratio = final_vs_initial(&nback_curve);
        println!("Learning slope: {:.6}", nback_slope);
        println!("Final/Initial:  {:.3}", nback_ratio);

        println!("\nWCST Learning Curve");
        println!("{:<8} {:<10}", "Block", "Categories");
        println!("{}", "-".repeat(20));
        for (i, v) in wcst_curve.iter().enumerate() {
            println!("{:<8} {:.4}", i + 1, v);
        }
        let wcst_slope = linear_slope(&wcst_curve);
        let wcst_ratio = final_vs_initial(&wcst_curve);
        println!("Learning slope: {:.6}", wcst_slope);
        println!("Final/Initial:  {:.3}", wcst_ratio);
    }
}
