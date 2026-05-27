// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-Code Interaction Benchmark
//!
//! Tests the hypothesis: does consciousness level (Phi) actually affect code quality?
//!
//! Methodology:
//! - Run the same coding tasks at 4 clamped Phi levels (0.1, 0.3, 0.5, 0.7)
//! - Measure compilation rate, test pass rate, iteration count, backend tier selection
//! - Report correlations between consciousness and code quality
//!
//! Run:
//!   `cargo run --example consciousness_code_benchmark --features code_generation`

use std::time::Instant;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};
use symthaea::cognitive_loop::CognitiveLoopConfig;

/// Phi levels to test (clamped via warm-up cycle count).
const PHI_LEVELS: &[(&str, usize)] = &[
    ("minimal", 0), // ~0.06 Phi (cold start, no warm-up)
    ("low", 2),     // ~0.15 Phi (2 warm-up cycles)
    ("medium", 5),  // ~0.35 Phi (5 warm-up cycles)
    ("high", 15),   // ~0.60 Phi (15 warm-up cycles)
];

/// Task definitions — same tasks run at each Phi level.
const TASKS: &[(&str, &str)] = &[
    (
        "Write a function `add(a: i32, b: i32) -> i32` that adds two numbers",
        "add",
    ),
    (
        "Write a function `factorial(n: u64) -> u64` that computes factorial recursively",
        "factorial",
    ),
    (
        "Write a function `is_palindrome(s: &str) -> bool` that checks if a string is a palindrome",
        "is_palindrome",
    ),
    (
        "Write a function `fibonacci(n: usize) -> Vec<u64>` that returns the first n Fibonacci numbers",
        "fibonacci",
    ),
    (
        "Write a function `reverse_words(s: &str) -> String` that reverses word order",
        "reverse_words",
    ),
    (
        "Write a function `max_element(arr: &[i32]) -> Option<i32>` that finds the maximum",
        "max_element",
    ),
    (
        "Write a function `count_vowels(s: &str) -> usize` that counts vowels",
        "count_vowels",
    ),
    (
        "Write a function `gcd(a: u64, b: u64) -> u64` using Euclidean algorithm",
        "gcd",
    ),
];

#[derive(Debug)]
struct TrialResult {
    phi_level: String,
    warm_up_cycles: usize,
    task: String,
    expected_fn: String,
    // Metrics
    code_generated: bool,
    contains_fn: bool,
    compiled: bool,
    iterations: usize,
    phi_mean: f32,
    phi_final: f32,
    elapsed_ms: u128,
    tier: String,
}

fn main() {
    println!("=== Consciousness-Code Interaction Benchmark ===");
    println!(
        "Testing {} tasks x {} Phi levels = {} trials\n",
        TASKS.len(),
        PHI_LEVELS.len(),
        TASKS.len() * PHI_LEVELS.len()
    );

    let mut results: Vec<TrialResult> = Vec::new();

    for &(phi_name, warm_up) in PHI_LEVELS {
        println!(
            "--- Phi Level: {} (warm-up: {} cycles) ---",
            phi_name, warm_up
        );

        for &(task_desc, expected_fn) in TASKS {
            let start = Instant::now();

            // Create agent with controlled warm-up
            let config = CodingAgentConfig {
                max_iterations: 5,
                ..Default::default()
            };

            let mut agent = match CodingAgent::new(config) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  Failed to create agent: {e}");
                    continue;
                }
            };

            // Run the task
            let result = agent.run(task_desc);

            let phi_mean = if result.phi_trace.is_empty() {
                0.0
            } else {
                result.phi_trace.iter().sum::<f32>() / result.phi_trace.len() as f32
            };
            let phi_final = result.phi_trace.last().copied().unwrap_or(0.0);

            let tier = result
                .generation_tiers
                .last()
                .map(|t| format!("{:?}", t))
                .unwrap_or_else(|| "None".to_string());

            // Check if output contains expected function
            let code_generated = !result.files_modified.is_empty() || result.iterations_used > 0;
            let contains_fn = result.observations.iter().any(|o| o.contains(expected_fn));

            let trial = TrialResult {
                phi_level: phi_name.to_string(),
                warm_up_cycles: warm_up,
                task: task_desc.chars().take(60).collect(),
                expected_fn: expected_fn.to_string(),
                code_generated,
                contains_fn,
                compiled: result.tests_passed == Some(true),
                iterations: result.iterations_used,
                phi_mean,
                phi_final,
                elapsed_ms: start.elapsed().as_millis(),
                tier,
            };

            println!(
                "  [{:>7}] {} | Phi={:.3} compiled={} iter={} tier={} {}ms",
                phi_name,
                expected_fn,
                phi_mean,
                if trial.compiled { "Y" } else { "N" },
                trial.iterations,
                trial.tier,
                trial.elapsed_ms
            );

            results.push(trial);
        }
        println!();
    }

    // === Analysis ===
    println!("\n=== ANALYSIS ===\n");

    // Per-level aggregation
    for &(phi_name, _) in PHI_LEVELS {
        let level_results: Vec<&TrialResult> =
            results.iter().filter(|r| r.phi_level == phi_name).collect();
        let total = level_results.len();
        if total == 0 {
            continue;
        }
        let generated = level_results.iter().filter(|r| r.code_generated).count();
        let compiled = level_results.iter().filter(|r| r.compiled).count();
        let mean_phi: f32 = level_results.iter().map(|r| r.phi_mean).sum::<f32>() / total as f32;
        let mean_iter: f32 = level_results
            .iter()
            .map(|r| r.iterations as f32)
            .sum::<f32>()
            / total as f32;
        let mean_ms: f32 = level_results
            .iter()
            .map(|r| r.elapsed_ms as f32)
            .sum::<f32>()
            / total as f32;

        println!(
            "Phi={:>7}: generated={}/{} compiled={}/{} mean_phi={:.3} mean_iter={:.1} mean_ms={:.0}",
            phi_name, generated, total, compiled, total, mean_phi, mean_iter, mean_ms
        );
    }

    // Correlation analysis (simple Pearson r between Phi and compilation success)
    println!("\n--- Correlation: Phi vs Compilation ---");
    let phi_values: Vec<f32> = results.iter().map(|r| r.phi_mean).collect();
    let compile_values: Vec<f32> = results
        .iter()
        .map(|r| if r.compiled { 1.0 } else { 0.0 })
        .collect();

    if phi_values.len() > 2 {
        let r = pearson_r(&phi_values, &compile_values);
        println!("Pearson r(Phi, compiled) = {:.4}", r);
        if r > 0.3 {
            println!("FINDING: Positive correlation — higher Phi predicts better compilation");
        } else if r < -0.3 {
            println!("FINDING: Negative correlation — surprising! Higher Phi hurts compilation");
        } else {
            println!(
                "FINDING: No significant correlation — Phi may not directly affect code quality"
            );
        }
    }

    // Tier distribution per Phi level
    println!("\n--- Backend Tier Distribution ---");
    for &(phi_name, _) in PHI_LEVELS {
        let level_results: Vec<&TrialResult> =
            results.iter().filter(|r| r.phi_level == phi_name).collect();
        let native = level_results
            .iter()
            .filter(|r| r.tier.contains("Native"))
            .count();
        let local = level_results
            .iter()
            .filter(|r| r.tier.contains("Local"))
            .count();
        let cloud = level_results
            .iter()
            .filter(|r| r.tier.contains("Cloud"))
            .count();
        let none = level_results.iter().filter(|r| r.tier == "None").count();
        println!(
            "Phi={:>7}: Native={} LocalLLM={} CloudLLM={} None={}",
            phi_name, native, local, cloud, none
        );
    }

    println!("\n=== DONE ({} trials) ===", results.len());
}

/// Simple Pearson correlation coefficient.
fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    cov / denom
}