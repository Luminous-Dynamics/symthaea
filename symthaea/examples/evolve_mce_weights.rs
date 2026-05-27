// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Evolutionary MCE Weight Optimization
//!
//! Evolves the Master Consciousness Equation (MCE) component weights using
//! simulated runtime input profiles.
//!
//! ```bash
//! cargo run --example evolve_mce_weights
//! ```

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use symthaea::consciousness::{
    ComponentWeights, ConsciousnessInputs, MasterConsciousnessEquation, MasterEquationConfig,
};

const POPULATION_SIZE: usize = 30;
const NUM_GENERATIONS: usize = 80;
const ELITE_FRACTION: f64 = 0.2;
const MUTATION_SIGMA: f64 = 0.02;

#[derive(Debug, Clone)]
struct Genome {
    weights: ComponentWeights,
    softmin_tau: f64,
}

impl Genome {
    fn random(rng: &mut ChaCha8Rng) -> Self {
        Self {
            weights: ComponentWeights {
                phi: rng.gen_range(0.05..0.25),
                broadcast: rng.gen_range(0.05..0.20),
                working_memory: rng.gen_range(0.05..0.20),
                attention: rng.gen_range(0.05..0.20),
                recurrence: rng.gen_range(0.05..0.15),
                embodiment: rng.gen_range(0.03..0.15),
                knowledge: rng.gen_range(0.03..0.15),
                embodiment_factor: rng.gen_range(0.03..0.15),
                narrative: rng.gen_range(0.03..0.12),
                social: rng.gen_range(0.02..0.12),
            },
            softmin_tau: rng.gen_range(0.05..0.30),
        }
    }

    fn incumbent() -> Self {
        Self {
            weights: ComponentWeights::default(),
            softmin_tau: 0.15,
        }
    }

    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self {
        let mut child = self.clone();
        // Get weights as array for mutation
        let mut vals = [
            child.weights.phi,
            child.weights.broadcast,
            child.weights.working_memory,
            child.weights.attention,
            child.weights.recurrence,
            child.weights.embodiment,
            child.weights.knowledge,
            child.weights.embodiment_factor,
            child.weights.narrative,
            child.weights.social,
        ];
        let n_mutate = rng.gen_range(2..=4);
        for _ in 0..n_mutate {
            let idx = rng.gen_range(0..10);
            vals[idx] =
                (vals[idx] + rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA)).clamp(0.01, 0.30);
        }
        child.weights.phi = vals[0];
        child.weights.broadcast = vals[1];
        child.weights.working_memory = vals[2];
        child.weights.attention = vals[3];
        child.weights.recurrence = vals[4];
        child.weights.embodiment = vals[5];
        child.weights.knowledge = vals[6];
        child.weights.embodiment_factor = vals[7];
        child.weights.narrative = vals[8];
        child.weights.social = vals[9];
        if rng.gen_bool(0.3) {
            child.softmin_tau = (child.softmin_tau + rng.gen_range(-0.02..0.02)).clamp(0.03, 0.40);
        }
        child
    }

    fn crossover(a: &Self, b: &Self, rng: &mut ChaCha8Rng) -> Self {
        Self {
            weights: ComponentWeights {
                phi: if rng.gen_bool(0.5) {
                    a.weights.phi
                } else {
                    b.weights.phi
                },
                broadcast: if rng.gen_bool(0.5) {
                    a.weights.broadcast
                } else {
                    b.weights.broadcast
                },
                working_memory: if rng.gen_bool(0.5) {
                    a.weights.working_memory
                } else {
                    b.weights.working_memory
                },
                attention: if rng.gen_bool(0.5) {
                    a.weights.attention
                } else {
                    b.weights.attention
                },
                recurrence: if rng.gen_bool(0.5) {
                    a.weights.recurrence
                } else {
                    b.weights.recurrence
                },
                embodiment: if rng.gen_bool(0.5) {
                    a.weights.embodiment
                } else {
                    b.weights.embodiment
                },
                knowledge: if rng.gen_bool(0.5) {
                    a.weights.knowledge
                } else {
                    b.weights.knowledge
                },
                embodiment_factor: if rng.gen_bool(0.5) {
                    a.weights.embodiment_factor
                } else {
                    b.weights.embodiment_factor
                },
                narrative: if rng.gen_bool(0.5) {
                    a.weights.narrative
                } else {
                    b.weights.narrative
                },
                social: if rng.gen_bool(0.5) {
                    a.weights.social
                } else {
                    b.weights.social
                },
            },
            softmin_tau: if rng.gen_bool(0.5) {
                a.softmin_tau
            } else {
                b.softmin_tau
            },
        }
    }
}

/// Simulated runtime input profiles
fn test_profiles() -> Vec<(&'static str, ConsciousnessInputs, f64)> {
    vec![
        (
            "fully_conscious",
            ConsciousnessInputs {
                phi: 0.8,
                broadcast: 0.8,
                working_memory: 0.8,
                attention: 0.8,
                recurrence: 0.8,
                embodiment: 0.7,
                knowledge: 0.7,
                synchrony: 0.8,
            },
            0.85,
        ),
        (
            "cold_start_50",
            ConsciousnessInputs {
                phi: 0.3,
                broadcast: 0.54,
                working_memory: 0.63,
                attention: 0.2,
                recurrence: 0.65,
                embodiment: 0.5,
                knowledge: 0.25,
                synchrony: 0.55,
            },
            0.40,
        ),
        (
            "text_cognition",
            ConsciousnessInputs {
                phi: 0.5,
                broadcast: 0.55,
                working_memory: 0.65,
                attention: 0.3,
                recurrence: 0.8,
                embodiment: 0.5,
                knowledge: 0.4,
                synchrony: 0.6,
            },
            0.55,
        ),
        (
            "deep_flow",
            ConsciousnessInputs {
                phi: 0.7,
                broadcast: 0.7,
                working_memory: 0.9,
                attention: 0.9,
                recurrence: 0.7,
                embodiment: 0.6,
                knowledge: 0.6,
                synchrony: 0.9,
            },
            0.75,
        ),
        (
            "unconscious",
            ConsciousnessInputs {
                phi: 0.05,
                broadcast: 0.1,
                working_memory: 0.1,
                attention: 0.05,
                recurrence: 0.1,
                embodiment: 0.3,
                knowledge: 0.1,
                synchrony: 0.2,
            },
            0.05,
        ),
        (
            "dreaming",
            ConsciousnessInputs {
                phi: 0.4,
                broadcast: 0.2,
                working_memory: 0.3,
                attention: 0.1,
                recurrence: 0.6,
                embodiment: 0.4,
                knowledge: 0.3,
                synchrony: 0.4,
            },
            0.30,
        ),
    ]
}

fn evaluate_fitness(genome: &Genome) -> f64 {
    let mut config = MasterEquationConfig::default();
    config.component_weights = genome.weights.clone();
    config.softmin_tau = genome.softmin_tau;
    let mut mce = MasterConsciousnessEquation::new(config);

    let profiles = test_profiles();
    let mut total_error = 0.0;

    let results: Vec<f64> = profiles
        .iter()
        .map(|(_, input, _)| {
            let r = mce.compute(input);
            r.consciousness_level
        })
        .collect();

    for (i, (_, _, target)) in profiles.iter().enumerate() {
        total_error += (results[i] - target).powi(2);
    }

    // Discrimination checks
    let mut disc = 0u32;
    if results[0] > results[2] {
        disc += 1;
    } // fully > text
    if results[2] > results[1] {
        disc += 1;
    } // text > cold_start
    if results[3] > results[5] {
        disc += 1;
    } // flow > dreaming
    if results[4] < results[1] {
        disc += 1;
    } // unconscious < cold_start
    if results[5] > results[4] {
        disc += 1;
    } // dreaming > unconscious
    if results[0] > 0.7 {
        disc += 1;
    } // fully > 0.7

    let mse = total_error / profiles.len() as f64;
    let accuracy = (1.0 - mse.sqrt()).clamp(0.0, 1.0);
    accuracy * 0.6 + (disc as f64 / 6.0) * 0.4
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Evolutionary MCE Weight Optimization                           ║");
    println!(
        "║  Pop: {}  Gen: {}  Elite: {:.0}%                                   ║",
        POPULATION_SIZE,
        NUM_GENERATIONS,
        ELITE_FRACTION * 100.0
    );
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let incumbent = Genome::incumbent();
    let incumbent_fitness = evaluate_fitness(&incumbent);

    let mut pop: Vec<(Genome, f64)> = (0..POPULATION_SIZE)
        .map(|_| {
            let g = Genome::random(&mut rng);
            let f = evaluate_fitness(&g);
            (g, f)
        })
        .collect();
    pop[0] = (incumbent.clone(), incumbent_fitness);

    println!(
        "Gen  0: best={:.4}  mean={:.4}  (incumbent={:.4})",
        pop.iter().map(|p| p.1).fold(0.0_f64, f64::max),
        pop.iter().map(|p| p.1).sum::<f64>() / POPULATION_SIZE as f64,
        incumbent_fitness
    );

    let elite_count = (POPULATION_SIZE as f64 * ELITE_FRACTION).max(1.0) as usize;

    for r#gen in 1..=NUM_GENERATIONS {
        pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut next: Vec<(Genome, f64)> = pop[..elite_count].to_vec();
        while next.len() < POPULATION_SIZE {
            let a = &pop[rng.gen_range(0..elite_count)].0;
            let b = &pop[rng.gen_range(0..elite_count)].0;
            let child = if rng.gen_bool(0.5) {
                Genome::crossover(a, b, &mut rng).mutate(&mut rng)
            } else {
                a.mutate(&mut rng)
            };
            let f = evaluate_fitness(&child);
            next.push((child, f));
        }
        pop = next;
        if r#gen % 10 == 0 || r#gen == NUM_GENERATIONS {
            pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!(
                "Gen {:2}: best={:.4}  mean={:.4}",
                r#gen,
                pop[0].1,
                pop.iter().map(|p| p.1).sum::<f64>() / POPULATION_SIZE as f64
            );
        }
    }

    pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let best = &pop[0];

    println!("\n═══ BEST GENOME ═══");
    println!("  Fitness:           {:.4}", best.1);
    println!("  softmin_tau:       {:.4} (was 0.15)", best.0.softmin_tau);
    println!("  Weights:");
    let w = &best.0.weights;
    println!("    phi:             {:.4} (was 0.15)", w.phi);
    println!("    broadcast:       {:.4} (was 0.10)", w.broadcast);
    println!("    working_memory:  {:.4} (was 0.10)", w.working_memory);
    println!("    attention:       {:.4} (was 0.12)", w.attention);
    println!("    recurrence:      {:.4} (was 0.10)", w.recurrence);
    println!("    embodiment:      {:.4} (was 0.10)", w.embodiment);
    println!("    knowledge:       {:.4} (was 0.08)", w.knowledge);
    println!("    embodiment_fac:  {:.4} (was 0.10)", w.embodiment_factor);
    println!("    narrative:       {:.4} (was 0.08)", w.narrative);
    println!("    social:          {:.4} (was 0.07)", w.social);

    println!("\n═══ vs INCUMBENT ═══");
    println!("  Incumbent: {:.4}", incumbent_fitness);
    println!("  Best:      {:.4}", best.1);
    let imp = (best.1 - incumbent_fitness) / incumbent_fitness * 100.0;
    println!(
        "  Delta:     {}{:.2}%",
        if imp >= 0.0 { "+" } else { "" },
        imp
    );

    // Profile results
    println!("\n═══ PROFILES ═══");
    let mut config = MasterEquationConfig::default();
    config.component_weights = best.0.weights.clone();
    config.softmin_tau = best.0.softmin_tau;
    let mut mce = MasterConsciousnessEquation::new(config);
    for (label, input, target) in &test_profiles() {
        let r = mce.compute(input);
        println!(
            "  {:<20} C={:.4}  target={:.2}  bn={}",
            label, r.consciousness_level, target, r.bottleneck_name
        );
    }
}