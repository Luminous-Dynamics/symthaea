// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark: measures neuroevolution generation latency.
//!
//! Run with: cargo run -p symthaea-neuroevolution --example benchmark

use std::time::Instant;
use symthaea_neuroevolution::{FepFitnessConfig, NeuroevolutionConfig, NeuroevolutionEngine};

fn main() {
    println!("=== CfC-HDC Neuroevolution Benchmark ===\n");

    // Small config (256D) for fast benchmark
    let config_256 = NeuroevolutionConfig {
        population_size: 20,
        tournament_size: 3,
        max_generations: 10,
        convergence_patience: 20,
        fitness_config: FepFitnessConfig {
            warmup_steps: 5,
            eval_steps: 20,
            dt: 0.05,
            input_dim: 256,
            ..Default::default()
        },
        genesis_phrase: "benchmark-256d".to_string(),
        ..Default::default()
    };

    println!(
        "Config: pop={}, eval_steps={}, dim=256",
        config_256.population_size, config_256.fitness_config.eval_steps
    );
    let mut engine = NeuroevolutionEngine::new(config_256);
    engine.initialize();

    let start = Instant::now();
    let mut gen_times = Vec::new();
    for _ in 0..10 {
        let gen_start = Instant::now();
        let snapshot = engine.step_generation();
        let gen_elapsed = gen_start.elapsed();
        gen_times.push(gen_elapsed);
        println!(
            "  Gen {:3}: best={:+.4}, mean={:+.4}, div={:.3}, species={}, time={:.0}ms",
            snapshot.generation,
            snapshot.best_fitness,
            snapshot.mean_fitness,
            snapshot.diversity,
            snapshot.species_count,
            gen_elapsed.as_millis(),
        );
    }
    let total = start.elapsed();

    let mean_ms =
        gen_times.iter().map(|d| d.as_millis()).sum::<u128>() as f64 / gen_times.len() as f64;
    let max_ms = gen_times.iter().map(|d| d.as_millis()).max().unwrap_or(0);
    let genome_kb = engine.population().len() * 2;

    println!("\n--- Summary ---");
    println!("  Total:     {:.1}s", total.as_secs_f64());
    println!("  Mean/r#gen:  {:.0}ms", mean_ms);
    println!("  Max/r#gen:   {}ms", max_ms);
    println!(
        "  Genome mem: {}KB ({} organisms x 2KB)",
        genome_kb,
        engine.population().len()
    );

    if let Some((genome, fitness)) = engine.best_ever() {
        let phenotype = genome.decode();
        println!("\n--- Best Evolved Organism ---");
        println!("  Fitness:    {:.4}", fitness);
        println!("  Tau base:   {:.4}", phenotype.neuron_config.tau_base);
        println!("  LR:         {:.6}", phenotype.neuron_config.learning_rate);
        println!("  Layers:     {:?}", phenotype.network_config.layer_sizes);
        println!(
            "  Skip conn:  {}",
            phenotype.network_config.skip_connections
        );
        println!("  Activation: {:?}", phenotype.neuron_config.activation);
    }
}
