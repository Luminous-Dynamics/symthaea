// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold Evolution Runner
//!
//! Runs N generations of cognitive threshold evolution using the neuroevolution
//! crate's ThresholdGenome + consistency fitness evaluator. Demonstrates the
//! "honest Darwin-Gödel" concept: parameter evolution within Holochain's
//! immutability boundary.
//!
//! Usage: cargo run --example evolve_thresholds --features neuroevolution
//!
//! Output: Per-generation statistics and final Pareto front.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_neuroevolution::{
    decode_thresholds, encode_thresholds,
    governance::{gate_threshold_proposal, GovernanceResult},
    threshold_genome::evaluate_threshold_fitness,
    NeuralGenome, ThresholdPhenotype,
};

/// Population size per generation.
const POP_SIZE: usize = 30;

/// Number of generations to evolve.
const NUM_GENERATIONS: usize = 20;

/// Mutation rate: fraction of bytes to flip per offspring.
const MUTATION_RATE: f64 = 0.005;

/// Tournament selection size.
const TOURNAMENT_SIZE: usize = 3;

/// Simple deterministic PRNG for reproducibility.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }
}

/// Individual in the population.
#[derive(Clone)]
struct Individual {
    genome: NeuralGenome,
    fitness: f64,
    phenotype: ThresholdPhenotype,
}

/// Create a random individual.
fn random_individual(rng: &mut Rng) -> Individual {
    let genome = NeuralGenome::random(rng.next());
    let phenotype = genome.decode_thresholds();
    let fitness = evaluate_threshold_fitness(&phenotype);
    Individual {
        genome,
        fitness,
        phenotype,
    }
}

/// Mutate a genome by flipping random bytes in the threshold region (bits 400-634).
fn mutate(genome: &mut NeuralGenome, rng: &mut Rng) {
    // Only mutate bytes 50-80 (bits 400-640, the threshold region)
    for byte_idx in 50..80 {
        if rng.next_f64() < MUTATION_RATE {
            let flip = rng.next() as u8;
            genome.hv.0[byte_idx] ^= flip;
        }
    }
}

/// Tournament selection: pick TOURNAMENT_SIZE random individuals, return the best.
fn tournament_select<'a>(pop: &'a [Individual], rng: &mut Rng) -> &'a Individual {
    let mut best = &pop[rng.next_usize(pop.len())];
    for _ in 1..TOURNAMENT_SIZE {
        let candidate = &pop[rng.next_usize(pop.len())];
        if candidate.fitness > best.fitness {
            best = candidate;
        }
    }
    best
}

/// Uniform crossover in the threshold region.
fn crossover(a: &NeuralGenome, b: &NeuralGenome, rng: &mut Rng) -> NeuralGenome {
    let mut child = a.clone();
    for byte_idx in 50..80 {
        if rng.next() % 2 == 0 {
            child.hv.0[byte_idx] = b.hv.0[byte_idx];
        }
    }
    child
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Threshold Evolution Runner");
    println!(
        "  {} individuals, {} generations",
        POP_SIZE, NUM_GENERATIONS
    );
    println!("  18 evolvable thresholds (safety/moral excluded)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut rng = Rng::new(42);

    // Initialize population
    let mut population: Vec<Individual> =
        (0..POP_SIZE).map(|_| random_individual(&mut rng)).collect();

    println!(
        "{:>5}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}",
        "Gen", "Best", "Mean", "Worst", "FEP_Surp", "Dream_Int"
    );
    println!("{}", "-".repeat(65));

    for r#gen in 0..NUM_GENERATIONS {
        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = population[0].fitness;
        let worst = population[population.len() - 1].fitness;
        let mean: f64 = population.iter().map(|i| i.fitness).sum::<f64>() / POP_SIZE as f64;
        let best_pheno = &population[0].phenotype;

        println!(
            "{:>5}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.2}  {:>10}",
            r#gen, best, mean, worst, best_pheno.fep_surprise_scale, best_pheno.dream_base_interval,
        );

        // Elitism: keep top 20%
        let elite_count = POP_SIZE / 5;
        let mut next_gen: Vec<Individual> = population[..elite_count].to_vec();

        // Fill rest with offspring
        while next_gen.len() < POP_SIZE {
            let parent_a = tournament_select(&population, &mut rng);
            let parent_b = tournament_select(&population, &mut rng);
            let mut child_genome = crossover(&parent_a.genome, &parent_b.genome, &mut rng);
            mutate(&mut child_genome, &mut rng);
            let phenotype = child_genome.decode_thresholds();
            let fitness = evaluate_threshold_fitness(&phenotype);
            next_gen.push(Individual {
                genome: child_genome,
                fitness,
                phenotype,
            });
        }

        population = next_gen;
    }

    // Final sort
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    println!("\n═══════════════════════════════════════════════════════════════");
    println!(
        "  Final Best Individual (fitness: {:.4})",
        population[0].fitness
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    let best = &population[0].phenotype;
    let defaults = ThresholdPhenotype::default();

    println!(
        "{:<35} {:>10} {:>10} {:>8}",
        "Threshold", "Evolved", "Default", "Delta%"
    );
    println!("{}", "-".repeat(68));

    print_row(
        "fep_surprise_scale",
        best.fep_surprise_scale,
        defaults.fep_surprise_scale,
    );
    print_row("fep_lr_decay", best.fep_lr_decay, defaults.fep_lr_decay);
    print_row_u64(
        "dream_base_interval",
        best.dream_base_interval,
        defaults.dream_base_interval,
    );
    print_row_u64(
        "dream_min_interval",
        best.dream_min_interval,
        defaults.dream_min_interval,
    );
    print_row(
        "neuromod_d2_baseline",
        best.neuromod_d2_baseline as f32,
        defaults.neuromod_d2_baseline as f32,
    );
    print_row(
        "neuromod_ne_phasic_threshold",
        best.neuromod_ne_phasic_threshold,
        defaults.neuromod_ne_phasic_threshold,
    );
    print_row(
        "neuromod_arousal_ema_decay",
        best.neuromod_arousal_ema_decay,
        defaults.neuromod_arousal_ema_decay,
    );
    print_row(
        "homeostasis_recalibrate_high",
        best.homeostasis_recalibrate_high,
        defaults.homeostasis_recalibrate_high,
    );
    print_row(
        "homeostasis_recalibrate_low",
        best.homeostasis_recalibrate_low,
        defaults.homeostasis_recalibrate_low,
    );
    print_row(
        "neuromod_ema_alpha",
        best.neuromod_ema_alpha,
        defaults.neuromod_ema_alpha,
    );
    print_row(
        "frustration_dampen_threshold",
        best.frustration_dampen_threshold as f32,
        defaults.frustration_dampen_threshold as f32,
    );
    print_row(
        "engagement_low_threshold",
        best.engagement_low_threshold as f32,
        defaults.engagement_low_threshold as f32,
    );
    print_row(
        "flow_exploration_increment",
        best.flow_exploration_increment,
        defaults.flow_exploration_increment,
    );
    print_row("coherence_low", best.coherence_low, defaults.coherence_low);
    print_row(
        "arousal_trap_threshold",
        best.arousal_trap_threshold,
        defaults.arousal_trap_threshold,
    );
    print_row(
        "self_model_weight_high",
        best.self_model_weight_high,
        defaults.self_model_weight_high,
    );
    print_row(
        "homeostasis_pull_cruise",
        best.homeostasis_pull_cruise,
        defaults.homeostasis_pull_cruise,
    );
    print_row(
        "confidence_crash_threshold",
        best.confidence_crash_threshold as f32,
        defaults.confidence_crash_threshold as f32,
    );

    // Governance gate check
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Governance Gate Check");
    println!("═══════════════════════════════════════════════════════════════\n");

    for phi in [0.3, 0.5, 0.7] {
        let result = gate_threshold_proposal(&defaults, best, phi, population[0].fitness);
        match result {
            GovernanceResult::Approved { confidence, .. } => {
                println!("  Phi={:.1}: APPROVED (confidence: {:.3})", phi, confidence);
            }
            GovernanceResult::Rejected { reason } => {
                println!("  Phi={:.1}: REJECTED — {}", phi, reason);
            }
        }
    }
}

fn print_row(name: &str, evolved: f32, default: f32) {
    let delta = if default.abs() > 1e-6 {
        ((evolved - default) / default * 100.0) as i32
    } else {
        0
    };
    let sign = if delta >= 0 { "+" } else { "" };
    println!(
        "{:<35} {:>10.4} {:>10.4} {:>7}%",
        name,
        evolved,
        default,
        format!("{sign}{delta}")
    );
}

fn print_row_u64(name: &str, evolved: u64, default: u64) {
    let delta = if default > 0 {
        ((evolved as f64 - default as f64) / default as f64 * 100.0) as i32
    } else {
        0
    };
    let sign = if delta >= 0 { "+" } else { "" };
    println!(
        "{:<35} {:>10} {:>10} {:>7}%",
        name,
        evolved,
        default,
        format!("{sign}{delta}")
    );
}