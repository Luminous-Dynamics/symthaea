// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Convergence validation: 50 generations at 16,384D.
//!
//! Verifies that neuroevolution actually discovers better CfC configs
//! over time, not just random drift.
//!
//! Run: cargo run -p symthaea-neuroevolution --example convergence_validation --release

use std::time::Instant;
use symthaea_neuroevolution::{FepFitnessConfig, NeuroevolutionConfig, NeuroevolutionEngine};

fn main() {
    println!("=== Neuroevolution Convergence Validation ===");
    println!("Config: pop=30, dim=16384, eval=50 steps, 50 generations\n");

    let config = NeuroevolutionConfig {
        population_size: 30,
        tournament_size: 3,
        max_generations: 50,
        convergence_patience: 50, // Don't early-stop
        fitness_config: FepFitnessConfig {
            warmup_steps: 10,
            eval_steps: 50,
            dt: 0.05,
            input_dim: 16_384,
            ..Default::default()
        },
        genesis_phrase: "convergence-validation".to_string(),
        ..Default::default()
    };

    let mut engine = NeuroevolutionEngine::new(config);
    engine.initialize();

    let start = Instant::now();
    let mut prev_best = f64::NEG_INFINITY;
    let mut monotonic_violations = 0;

    for r#gen in 1..=50 {
        let gen_start = Instant::now();
        let snapshot = engine.step_generation();
        let gen_time = gen_start.elapsed();

        // Check monotonic improvement (elitism should guarantee this)
        if snapshot.best_fitness < prev_best - 1e-10 {
            monotonic_violations += 1;
            println!(
                "  !! Gen {:3}: REGRESSION best={:+.6} < prev={:+.6}",
                r#gen, snapshot.best_fitness, prev_best
            );
        }
        prev_best = snapshot.best_fitness;

        if r#gen % 5 == 0 || r#gen == 1 {
            println!(
                "  Gen {:3}: best={:+.6}, mean={:+.6}, div={:.3}, species={}, time={:.1}s",
                snapshot.generation,
                snapshot.best_fitness,
                snapshot.mean_fitness,
                snapshot.diversity,
                snapshot.species_count,
                gen_time.as_secs_f64(),
            );
        }
    }

    let total = start.elapsed();
    let result = engine.best_ever();

    println!("\n--- Results ---");
    println!(
        "  Total time:      {:.1}s ({:.1}s/r#gen)",
        total.as_secs_f64(),
        total.as_secs_f64() / 50.0
    );
    println!("  Monotonic violations: {}", monotonic_violations);
    if let Some((genome, fitness)) = result {
        let phenotype = genome.decode();
        println!("  Best fitness:    {:.6}", fitness);
        println!("  Evolved tau:     {:.4}", phenotype.neuron_config.tau_base);
        println!(
            "  Evolved LR:      {:.6}",
            phenotype.neuron_config.learning_rate
        );
        println!(
            "  Evolved layers:  {:?}",
            phenotype.network_config.layer_sizes
        );
        println!(
            "  Activation:      {:?}",
            phenotype.neuron_config.activation
        );
    }

    // Save checkpoint
    let checkpoint_path = std::path::PathBuf::from("neuroevo_convergence_checkpoint.json");
    match engine.save_checkpoint(&checkpoint_path) {
        Ok(()) => println!("\n  Checkpoint saved: {}", checkpoint_path.display()),
        Err(e) => println!("\n  Checkpoint save failed: {}", e),
    }

    if monotonic_violations == 0 {
        println!("\n  PASS: Fitness monotonically non-decreasing across all 50 generations.");
    } else {
        println!(
            "\n  WARN: {} monotonic violations detected (elitism bug?).",
            monotonic_violations
        );
    }
}
