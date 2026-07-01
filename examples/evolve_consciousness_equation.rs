// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Evolutionary Consciousness Architecture Search
//!
//! Uses evolutionary strategies to discover optimal consciousness equation
//! configurations. Each genome encodes equation structure parameters and is
//! evaluated against theoretical consciousness tests.
//!
//! ```bash
//! cargo run --example evolve_consciousness_equation
//! ```

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use symthaea::consciousness::consciousness_equation_v2::{
    ConsciousnessEquationV2, ConsciousnessStateV2, CoreComponent,
};

const POPULATION_SIZE: usize = 20;
const NUM_GENERATIONS: usize = 50;
const ELITE_FRACTION: f64 = 0.25;
const MUTATION_RATE: f64 = 0.3;
const MUTATION_SIGMA: f64 = 0.1;

/// Genome encoding consciousness equation parameters
#[derive(Debug, Clone)]
struct Genome {
    /// Softmin exponent for necessary components (higher = harder min)
    softmin_necessary: f64,
    /// Softmin exponent for amplifiers
    softmin_amplifier: f64,
    /// Weight of necessary vs amplifier contribution
    necessary_weight: f64,
    /// Amplifier ceiling (max boost from amplifiers)
    amplifier_ceiling: f64,
    /// First-order consciousness floor
    first_order_floor: f64,
    /// Whether HOT depth is necessary (true) or amplifier (false)
    hot_is_necessary: bool,
    /// Whether attention is necessary (true) or amplifier (false)
    attention_is_necessary: bool,
}

impl Genome {
    fn random(rng: &mut ChaCha8Rng) -> Self {
        Self {
            softmin_necessary: rng.gen_range(1.0..10.0),
            softmin_amplifier: rng.gen_range(0.5..5.0),
            necessary_weight: rng.gen_range(0.3..0.9),
            amplifier_ceiling: rng.gen_range(1.1..2.5),
            first_order_floor: rng.gen_range(0.0..0.5),
            hot_is_necessary: rng.gen_bool(0.3),
            attention_is_necessary: rng.gen_bool(0.3),
        }
    }

    fn mutate(&self, rng: &mut ChaCha8Rng) -> Self {
        let mut child = self.clone();
        if rng.gen_bool(MUTATION_RATE) {
            child.softmin_necessary += rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA) * 3.0;
            child.softmin_necessary = child.softmin_necessary.clamp(0.5, 20.0);
        }
        if rng.gen_bool(MUTATION_RATE) {
            child.softmin_amplifier += rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA) * 2.0;
            child.softmin_amplifier = child.softmin_amplifier.clamp(0.1, 10.0);
        }
        if rng.gen_bool(MUTATION_RATE) {
            child.necessary_weight += rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA);
            child.necessary_weight = child.necessary_weight.clamp(0.1, 0.95);
        }
        if rng.gen_bool(MUTATION_RATE) {
            child.amplifier_ceiling += rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA) * 0.5;
            child.amplifier_ceiling = child.amplifier_ceiling.clamp(1.0, 3.0);
        }
        if rng.gen_bool(MUTATION_RATE) {
            child.first_order_floor += rng.gen_range(-MUTATION_SIGMA..MUTATION_SIGMA) * 0.2;
            child.first_order_floor = child.first_order_floor.clamp(0.0, 0.8);
        }
        if rng.gen_bool(MUTATION_RATE * 0.5) {
            child.hot_is_necessary = !child.hot_is_necessary;
        }
        if rng.gen_bool(MUTATION_RATE * 0.5) {
            child.attention_is_necessary = !child.attention_is_necessary;
        }
        child
    }

    fn crossover(a: &Self, b: &Self, rng: &mut ChaCha8Rng) -> Self {
        Self {
            softmin_necessary: if rng.gen_bool(0.5) {
                a.softmin_necessary
            } else {
                b.softmin_necessary
            },
            softmin_amplifier: if rng.gen_bool(0.5) {
                a.softmin_amplifier
            } else {
                b.softmin_amplifier
            },
            necessary_weight: if rng.gen_bool(0.5) {
                a.necessary_weight
            } else {
                b.necessary_weight
            },
            amplifier_ceiling: if rng.gen_bool(0.5) {
                a.amplifier_ceiling
            } else {
                b.amplifier_ceiling
            },
            first_order_floor: if rng.gen_bool(0.5) {
                a.first_order_floor
            } else {
                b.first_order_floor
            },
            hot_is_necessary: if rng.gen_bool(0.5) {
                a.hot_is_necessary
            } else {
                b.hot_is_necessary
            },
            attention_is_necessary: if rng.gen_bool(0.5) {
                a.attention_is_necessary
            } else {
                b.attention_is_necessary
            },
        }
    }
}

/// Create a state with the given component values
fn make_state(
    phi: f64,
    binding: f64,
    workspace: f64,
    hot: f64,
    attention: f64,
    recurrence: f64,
    efficacy: f64,
    knowledge: f64,
) -> ConsciousnessStateV2 {
    let mut core_values = HashMap::new();
    core_values.insert(CoreComponent::Integration, phi);
    core_values.insert(CoreComponent::Binding, binding);
    core_values.insert(CoreComponent::Workspace, workspace);
    core_values.insert(CoreComponent::Recursion, hot);
    core_values.insert(CoreComponent::Attention, attention);
    core_values.insert(CoreComponent::Efficacy, efficacy);
    core_values.insert(CoreComponent::Knowledge, knowledge);
    ConsciousnessStateV2 {
        core_values,
        extended_values: HashMap::new(),
        phase_coherence: HashMap::new(),
        substrate_feasibility: 1.0,
        timestamp: 0,
        context: String::new(),
    }
}

/// Compute consciousness using the genome's parameterized equation
fn compute_c(genome: &Genome, s: &ConsciousnessStateV2) -> f64 {
    let get = |c: CoreComponent| *s.core_values.get(&c).unwrap_or(&0.0);
    let phi = get(CoreComponent::Integration);
    let binding = get(CoreComponent::Binding);
    let workspace = get(CoreComponent::Workspace);
    let hot = get(CoreComponent::Recursion);
    let attention = get(CoreComponent::Attention);
    let efficacy = get(CoreComponent::Efficacy);
    let knowledge = get(CoreComponent::Knowledge);

    let mut necessary: Vec<f64> = vec![phi, binding, workspace];
    let mut amplifiers: Vec<f64> = vec![efficacy, knowledge];

    if genome.hot_is_necessary {
        necessary.push(hot);
    } else {
        amplifiers.push(hot);
    }
    if genome.attention_is_necessary {
        necessary.push(attention);
    } else {
        amplifiers.push(attention);
    }

    // Softmin of necessary
    let exp_n = genome.softmin_necessary;
    let nec = if necessary.is_empty() {
        0.0
    } else {
        let sum: f64 = necessary.iter().map(|&x| (-(x * exp_n)).exp()).sum();
        -(sum / necessary.len() as f64).ln() / exp_n
    };

    // Amplifier mean
    let amp = if amplifiers.is_empty() {
        1.0
    } else {
        let mean: f64 = amplifiers.iter().sum::<f64>() / amplifiers.len() as f64;
        1.0 + (mean * (genome.amplifier_ceiling - 1.0))
    };

    let base = nec * genome.necessary_weight + nec * amp * (1.0 - genome.necessary_weight);

    // First-order floor
    let first_order = if !genome.hot_is_necessary && hot < 0.1 {
        genome.first_order_floor * phi.min(binding).min(workspace)
    } else {
        0.0
    };

    (base + first_order).clamp(0.0, 1.0)
}

/// Evaluate fitness against 8 theoretical consciousness tests
fn evaluate_fitness(genome: &Genome) -> f64 {
    let mut tests_passed = 0u32;

    // 1. IIT: High Phi > Low Phi
    let hi_phi = compute_c(genome, &make_state(0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5));
    let lo_phi = compute_c(genome, &make_state(0.05, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5));
    if hi_phi > lo_phi {
        tests_passed += 1;
    }
    let sensitivity = (hi_phi - lo_phi).abs();

    // 2. GWT: No workspace → low C
    let no_ws = compute_c(genome, &make_state(0.5, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5));
    if no_ws < 0.3 {
        tests_passed += 1;
    }

    // 3. Binding: No binding → low C
    let no_b = compute_c(genome, &make_state(0.5, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5));
    if no_b < 0.3 {
        tests_passed += 1;
    }

    // 4. HOT: High recursion > Low recursion
    let hi_hot = compute_c(genome, &make_state(0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5));
    let lo_hot = compute_c(genome, &make_state(0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5));
    if hi_hot > lo_hot {
        tests_passed += 1;
    }

    // 5. First-order: High Phi+B+W, low HOT → still conscious (Block 2007)
    let first_order = compute_c(genome, &make_state(0.9, 0.9, 0.9, 0.05, 0.5, 0.5, 0.5, 0.5));
    if first_order > 0.2 {
        tests_passed += 1;
    }

    // 6. AST: Zero attention doesn't kill consciousness
    let no_att = compute_c(genome, &make_state(0.7, 0.7, 0.7, 0.6, 0.0, 0.5, 0.5, 0.5));
    if no_att >= 0.1 {
        tests_passed += 1;
    }

    // 7. Low everything → low C
    let min_c = compute_c(genome, &make_state(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1));
    if min_c < 0.3 {
        tests_passed += 1;
    }

    // 8. Dynamic range
    let max_c = compute_c(genome, &make_state(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
    if max_c - min_c > 0.5 {
        tests_passed += 1;
    }

    // Weighted fitness
    let theoretical = tests_passed as f64 / 8.0;
    let range = (max_c - min_c).clamp(0.0, 1.0);
    let discrim = [hi_phi > lo_phi, no_ws < 0.3, no_b < 0.3, hi_hot > lo_hot]
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 4.0;

    theoretical * 0.40
        + sensitivity.clamp(0.0, 1.0) * 0.20
        + range * 0.20
        + first_order.clamp(0.0, 1.0) * 0.10
        + discrim * 0.10
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Evolutionary Consciousness Architecture Search                 ║");
    println!(
        "║  Pop: {}  Gen: {}  Elite: {:.0}%  Mutation: {:.0}%                      ║",
        POPULATION_SIZE,
        NUM_GENERATIONS,
        ELITE_FRACTION * 100.0,
        MUTATION_RATE * 100.0
    );
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Incumbent: current structured_bottleneck config
    let incumbent = Genome {
        softmin_necessary: 5.0,
        softmin_amplifier: 2.0,
        necessary_weight: 0.7,
        amplifier_ceiling: 1.5,
        first_order_floor: 0.3,
        hot_is_necessary: false,
        attention_is_necessary: false,
    };
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
    println!("  Fitness:              {:.4}", best.1);
    println!("  softmin_necessary:    {:.3}", best.0.softmin_necessary);
    println!("  softmin_amplifier:    {:.3}", best.0.softmin_amplifier);
    println!("  necessary_weight:     {:.3}", best.0.necessary_weight);
    println!("  amplifier_ceiling:    {:.3}", best.0.amplifier_ceiling);
    println!("  first_order_floor:    {:.3}", best.0.first_order_floor);
    println!("  HOT necessary:        {}", best.0.hot_is_necessary);
    println!("  Attention necessary:  {}", best.0.attention_is_necessary);

    println!("\n═══ vs INCUMBENT ═══");
    println!("  Incumbent: {:.4}", incumbent_fitness);
    println!("  Best:      {:.4}", best.1);
    let imp = (best.1 - incumbent_fitness) / incumbent_fitness * 100.0;
    println!(
        "  Delta:     {}{:.2}%",
        if imp >= 0.0 { "+" } else { "" },
        imp
    );

    println!("\n═══ TOP 5 ═══");
    for (i, (g, f)) in pop.iter().take(5).enumerate() {
        println!(
            "  #{}: f={:.4} nec={:.1} amp={:.1} w={:.2} ceil={:.2} floor={:.2} HOT={} ATT={}",
            i + 1,
            f,
            g.softmin_necessary,
            g.softmin_amplifier,
            g.necessary_weight,
            g.amplifier_ceiling,
            g.first_order_floor,
            if g.hot_is_necessary { "N" } else { "A" },
            if g.attention_is_necessary { "N" } else { "A" }
        );
    }

    // Validate best genome with the actual equation for comparison
    println!("\n═══ VALIDATION: Best genome vs actual ConsciousnessEquationV2 ═══");
    let mut eq = ConsciousnessEquationV2::new();
    let test_states = [
        (
            "High all",
            make_state(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        ),
        (
            "Low Phi",
            make_state(0.05, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        ),
        (
            "No workspace",
            make_state(0.8, 0.8, 0.01, 0.8, 0.8, 0.8, 0.8, 0.8),
        ),
        (
            "First-order",
            make_state(0.9, 0.9, 0.9, 0.05, 0.5, 0.5, 0.5, 0.5),
        ),
    ];

    for (label, state) in &test_states {
        let evolved_c = compute_c(&best.0, state);
        let actual = eq.compute(state);
        println!(
            "  {:<15} evolved={:.4}  actual={:.4}",
            label, evolved_c, actual.consciousness
        );
    }
}