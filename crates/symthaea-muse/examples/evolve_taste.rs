// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolve parameters using the TASTE BENCHMARK as fitness.
//!
//! Much faster than evolve_params (10s render vs 15s + audio analysis).
//! Directly optimizes toward Tristan's taste profile.
//!
//! Run: cargo run --release -p symthaea-muse --example evolve_taste

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Taste-Driven Evolution                                    ║");
    println!("║  Fitness = Taste Benchmark (Tristan's profile)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Smaller population, more generations for faster iteration
    let config = symthaea_muse::param_tuner::TunerConfig {
        population_size: 15,
        max_generations: 30,
        mutation_rate: 0.20,
        mutation_sigma: 0.10,
        elitism_count: 2,
        tournament_size: 3,
        crossover_rate: 0.7,
    };

    println!(
        "Population: {} | Generations: {} | 10s render per eval\n",
        config.population_size, config.max_generations
    );

    let result = symthaea_muse::param_tuner::evolve_taste(&config);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  Best taste score: {:.1}/100                                ║",
        result.best_fitness
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Optimal parameters:");
    for (name, value) in symthaea_muse::param_tuner::decode_all(&result.best_genome) {
        println!("  {name:25}: {value:.4}");
    }
}
