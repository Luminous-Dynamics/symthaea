// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolve optimal music parameters via evolutionary optimization.
//!
//! Renders 15-second clips for each genome in a population of 30,
//! scores with the music critic + aesthetic listener, evolves for 50
//! generations. Prints the best parameter set found.
//!
//! Run: cargo run -p symthaea-muse --example evolve_params

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Evolutionary Parameter Tuner                           ║");
    println!("║     Population: 30 | Generations: 50 | Clip: 15s          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let config = symthaea_muse::param_tuner::TunerConfig::default();

    println!(
        "Evolving {} parameters across {} individuals for {} generations...\n",
        symthaea_muse::param_tuner::default_params().len(),
        config.population_size,
        config.max_generations
    );

    let result = symthaea_muse::param_tuner::evolve(&config);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Evolution Complete                                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Best fitness: {:.4}                                       ║",
        result.best_fitness
    );
    println!(
        "║  Generations:  {}                                          ║",
        result.generations
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Optimal parameters:");
    for (name, value) in symthaea_muse::param_tuner::decode_all(&result.best_genome) {
        println!("  {name:25}: {value:.4}");
    }

    println!("\nFitness history (best, mean):");
    for (i, (best, mean)) in result.history.iter().enumerate() {
        if i % 10 == 0 || i == result.history.len() - 1 {
            println!("  Gen {i:3}: best={best:.4} mean={mean:.4}");
        }
    }
}
