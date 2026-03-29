// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Primitive Learning Gate Experiment
//!
//! Tests whether the HDC+LTC combined system can learn to complete
//! semantic reasoning patterns like:
//! - "CAUSE → ? → EFFECT"
//! - "WANT → ? → HAVE"
//! - "BEFORE → ? → AFTER"
//!
//! This bridges HDC's instant pattern matching with LTC's temporal learning.

use symthaea::benchmarks::{
    PrimitiveLearningConfig, PrimitiveLearningGate, PrimitiveLearningResults, PrimitiveTask,
};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   PRIMITIVE LEARNING GATE - HDC+LTC Combined Learning        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Hypothesis: Semantic primitives + temporal learning =        ║");
    println!("║             better reasoning than raw patterns alone         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let config = PrimitiveLearningConfig::default();
    println!("Configuration:");
    println!("  Episodes: {}", config.num_episodes);
    println!("  Examples/episode: {}", config.examples_per_episode);
    println!("  Test size: {}", config.test_size);
    println!("  LTC neurons: {}", config.ltc_config.num_neurons);
    println!();

    println!("Running 5 primitive reasoning tasks...\n");

    let mut gate = PrimitiveLearningGate::new(config)?;
    let results = gate.run_all_tasks()?;

    // Print results
    print_results(&results);

    Ok(())
}

fn print_results(results: &[PrimitiveLearningResults]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        PRIMITIVE LEARNING GATE RESULTS                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Can the system learn to complete semantic reasoning chains?  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut passed_count = 0;
    let mut total_improvement = 0.0;

    for result in results {
        let status = if result.passed {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };

        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Task: {:<40} {:>8} │", result.task_name, status);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ {}  │", result.reasoning);
        println!(
            "│ Time: {}ms                                              │",
            result.total_time_ms
        );
        println!("└─────────────────────────────────────────────────────────────┘");
        println!();

        if result.passed {
            passed_count += 1;
        }
        total_improvement += result.accuracy_improvement;
    }

    let avg_improvement = total_improvement / results.len() as f32;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║ OVERALL: {}/5 tasks passed ({:.1}%)                          ║",
        passed_count,
        passed_count as f32 / 5.0 * 100.0
    );
    println!(
        "║ Average Accuracy Improvement: {:.1}%                          ║",
        avg_improvement
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Print learning curves for best performing task
    if let Some(best) = results.iter().max_by(|a, b| {
        a.accuracy_improvement
            .partial_cmp(&b.accuracy_improvement)
            .unwrap()
    }) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║  LEARNING CURVE: {}                          ║",
            best.task_name
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("Episode | Train Loss | Test Loss  | Accuracy   | Tau Mean");
        println!("--------|------------|------------|------------|----------");

        for ep in &best.episodes {
            println!(
                "  {:>3}   |   {:.4}   |   {:.4}   |  {:.1}%     |  {:.2}",
                ep.episode, ep.train_loss, ep.test_loss, ep.completion_accuracy, ep.tau_mean
            );
        }
    }
}
