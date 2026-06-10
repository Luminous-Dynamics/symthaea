// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolutionary Consciousness Dynamics Demo (Direction G)
//!
//! Evolves neural architectures using Phi (integrated information) as the
//! fitness function, tracking consciousness emergence across generations.
//!
//! Run: cargo run -p symthaea-neuroevolution --example evolve_consciousness_demo

use symthaea_neuroevolution::phi_fitness::{
    ConsciousnessGenome, ConsciousnessSpecies, EvolutionConfig, EvolutionaryLandscape,
    run_evolution,
};

fn main() {
    // ── Banner ──────────────────────────────────────────────────────────
    println!(
        "\n\
         ╔══════════════════════════════════════════════════════╗\n\
         ║  Evolutionary Consciousness Dynamics Demo            ║\n\
         ╠══════════════════════════════════════════════════════╣\n\
         ║  Evolving neural architectures to maximize Phi       ║\n\
         ║  Population: 100 | Generations: 50                   ║\n\
         ╚══════════════════════════════════════════════════════╝\n"
    );

    // ── Configure evolution ─────────────────────────────────────────────
    let config = EvolutionConfig {
        population_size: 100,
        generations: 50,
        input_size: 8,
        output_size: 4,
        seed: 2026,
        ..EvolutionConfig::default()
    };

    // ── Run evolution ───────────────────────────────────────────────────
    let start = std::time::Instant::now();
    let landscape: EvolutionaryLandscape = run_evolution(config.clone());
    let elapsed = start.elapsed();

    // ── Per-generation table ────────────────────────────────────────────
    println!(
        " Gen | Mean Phi | Max Phi  | Min Phi  | Species | Entropy\n\
         -----|----------|----------|----------|---------|--------"
    );
    for stats in &landscape.generation_stats {
        println!(
            " {:>3} | {:.4}   | {:.4}   | {:.4}   |   {:>3}   | {:.2}",
            stats.generation,
            stats.mean_phi,
            stats.max_phi,
            stats.min_phi,
            stats.num_species,
            stats.landscape_entropy,
        );
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("\nEvolution completed in {:.2?}", elapsed);

    if let Some(final_stats) = landscape.generation_stats.last() {
        println!(
            "Final generation: mean Phi = {:.4}, max Phi = {:.4}",
            final_stats.mean_phi, final_stats.max_phi
        );
    }

    // ── Best organism's consciousness profile ───────────────────────────
    // Reconstruct the best organism from the final generation to show its profile.
    // We re-run a mini evolution to get the actual population at the end.
    let best_profile = find_best_profile(&config);

    println!(
        "\n\
         ═══ Best Organism (Generation {}) ═══\n\
         Consciousness Profile:\n\
         \x20 Phi:        {:.3}\n\
         \x20 Complexity: {:.3}\n\
         \x20 Modularity: {:.3}\n\
         \x20 Recurrence: {:.3}\n\
         \x20 Hierarchy:  {:.3}\n\
         \x20 Diversity:  {:.3}",
        config.generations,
        best_profile.phi,
        best_profile.complexity,
        best_profile.modularity,
        best_profile.recurrence,
        best_profile.hierarchy,
        best_profile.diversity,
    );

    // ── Species distribution from final generation ──────────────────────
    if let Some(final_species) = landscape.species_history.last() {
        println!("\n═══ Species Distribution ═══");
        for sp in final_species {
            println!(
                "Species {:>2}: {:>3} members, centroid Phi = {:.3}",
                sp.id,
                sp.members.len(),
                sp.centroid.phi,
            );
        }
        println!();
    }

    // ── Phi trajectory sparkline ────────────────────────────────────────
    print_trajectory(&landscape);
}

/// Re-run evolution with identical config to extract the final population's best profile.
///
/// This is deterministic (same seed) so reproduces identical results.
fn find_best_profile(
    config: &EvolutionConfig,
) -> symthaea_neuroevolution::phi_fitness::ConsciousnessProfile {
    use rand::{Rng, SeedableRng};
    use symthaea_neuroevolution::phi_fitness::PhiFitnessEvaluator;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut innovation_counter = 0u64;
    let mut evaluator = PhiFitnessEvaluator::default();

    let mut population: Vec<ConsciousnessGenome> = (0..config.population_size)
        .map(|i| {
            let g = ConsciousnessGenome::random(
                config.input_size,
                config.output_size,
                config.seed + i as u64,
            );
            if let Some(mx) = g.network_topology.iter().map(|c| c.innovation).max() {
                if mx >= innovation_counter {
                    innovation_counter = mx + 1;
                }
            }
            g
        })
        .collect();

    for _ in 0..config.generations {
        let mut fitnesses: Vec<f64> = Vec::with_capacity(population.len());
        for genome in &mut population {
            let fitness = evaluator.evaluate(genome, 0.5, Some(0.0));
            fitnesses.push(fitness);
        }

        let species = ConsciousnessSpecies::speciate(&mut population, config.speciation_threshold);

        let mut next_gen: Vec<ConsciousnessGenome> = Vec::with_capacity(config.population_size);

        // Elitism: keep best from each species
        for sp in &species {
            if !sp.members.is_empty() {
                let best_idx = *sp
                    .members
                    .iter()
                    .max_by(|&&a, &&b| {
                        fitnesses[a]
                            .partial_cmp(&fitnesses[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                next_gen.push(population[best_idx].clone());
            }
        }

        // Fill rest via tournament selection + mutation
        while next_gen.len() < config.population_size {
            let a = rng.gen_range(0..population.len());
            let b = rng.gen_range(0..population.len());
            let winner = if fitnesses[a] >= fitnesses[b] { a } else { b };
            let mut child = population[winner].clone();
            child.mutate(config.mutation_rate, &mut rng, &mut innovation_counter);
            next_gen.push(child);
        }

        population = next_gen;
    }

    // Find the organism with the highest Phi in the final population
    let mut best_phi = 0.0_f64;
    let mut best_profile = symthaea_neuroevolution::phi_fitness::ConsciousnessProfile::default();
    for genome in &mut population {
        let profile = genome.compute_consciousness_profile();
        if profile.phi > best_phi {
            best_phi = profile.phi;
            best_profile = profile;
        }
    }
    best_profile
}

/// Print a simple ASCII trajectory of mean Phi over generations.
fn print_trajectory(landscape: &EvolutionaryLandscape) {
    let phis: Vec<f64> = landscape
        .generation_stats
        .iter()
        .map(|s| s.mean_phi)
        .collect();
    if phis.is_empty() {
        return;
    }

    let max_phi = phis.iter().cloned().fold(0.0_f64, f64::max);
    let min_phi = phis.iter().cloned().fold(f64::MAX, f64::min);
    let range = (max_phi - min_phi).max(1e-6);
    let height = 10usize;

    println!("═══ Mean Phi Trajectory ═══");
    for row in (0..height).rev() {
        let threshold = min_phi + range * (row as f64 / height as f64);
        let label = if row == height - 1 {
            format!("{:.3} |", max_phi)
        } else if row == 0 {
            format!("{:.3} |", min_phi)
        } else {
            "      |".to_string()
        };
        print!("{}", label);
        for &phi in &phis {
            if phi >= threshold {
                print!("*");
            } else {
                print!(" ");
            }
        }
        println!();
    }
    println!("       {}", "-".repeat(phis.len().min(50)));
    println!(
        "       Gen 0{:>width$}Gen {}",
        "",
        phis.len() - 1,
        width = phis.len().saturating_sub(6)
    );
}
