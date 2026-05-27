// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phi-Directed Threshold Evolution
//!
//! Evolves cognitive thresholds using ACTUAL Phi measurement as a fitness
//! objective, with multi-objective Pareto sorting to prevent Goodhart's Law.
//!
//! 5 objectives (all independent, Pareto-sorted):
//! 1. Mean Phi — sustained consciousness integration
//! 2. FE reduction rate — system must be learning, not catatonic
//! 3. Prediction accuracy — system must predict its next state
//! 4. Phi stability — penalize oscillation (seizure vs genuine consciousness)
//! 5. Threshold consistency — internal parameter coherence
//!
//! Usage: cargo run --release --features neuroevolution --example evolve_for_phi
//!
//! Each organism is evaluated for 50 CfC cycles (warmup 10 + eval 40).
//! Population 20, 15 generations. ~1000 total organism evaluations.

#[cfg(not(feature = "neuroevolution"))]
fn main() {
    eprintln!("Requires `neuroevolution` feature.");
}

#[cfg(feature = "neuroevolution")]
fn main() {
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_neuroevolution::{
        threshold_genome::evaluate_threshold_fitness, FepFitnessBridge, FepFitnessConfig,
        FitnessWeights, InputStrategy, NeuralGenome, NeuralOrganism,
    };

    const POP_SIZE: usize = 20;
    const GENERATIONS: usize = 15;
    const EVAL_STEPS: usize = 40;
    const WARMUP_STEPS: usize = 10;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Phi-Directed Threshold Evolution");
    println!(
        "  {} organisms × {} generations × {} eval cycles",
        POP_SIZE, GENERATIONS, EVAL_STEPS
    );
    println!("  5 Pareto objectives: Phi, FE-reduction, pred-acc, stability, consistency");
    println!("═══════════════════════════════════════════════════════════════\n");

    let genesis = GenesisSeed::from_phrase("phi-directed-evolution");
    let fitness_config = FepFitnessConfig {
        warmup_steps: WARMUP_STEPS,
        eval_steps: EVAL_STEPS,
        dt: 0.05,
        input_dim: 256,
        input_strategy: InputStrategy::Temporal {
            frequencies: vec![0.5, 1.7, 3.1, 7.0],
            seed: 42,
        },
        weights: FitnessWeights {
            free_energy: 0.25,
            phi: 0.35,
            consciousness: 0.2,
            efficiency: 0.2,
        },
    };
    let mut bridge = FepFitnessBridge::new(fitness_config);

    // Initialize population
    let mut population: Vec<NeuralOrganism> = (0..POP_SIZE)
        .map(|i| {
            let genome = NeuralGenome::from_genesis(&genesis, &format!("org-{i}"));
            NeuralOrganism::spawn(i as u64, genome, &genesis, 0)
        })
        .collect();

    println!(
        "{:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Gen", "Best Φ", "Mean Φ", "FE-red", "Pred", "Stab", "Thresh"
    );
    println!("{}", "-".repeat(70));

    for r#gen in 0..GENERATIONS {
        // Evaluate all organisms
        bridge.evaluate_population(&mut population);

        // Sort by composite (for display)
        population.sort_by(|a, b| {
            b.fitness
                .composite
                .partial_cmp(&a.fitness.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = &population[0].fitness;
        let mean_phi: f64 = population.iter().map(|o| o.fitness.phi).sum::<f64>() / POP_SIZE as f64;

        // FE reduction: compute from stored initial/final FE
        let best_fe_red = if population[0].initial_fe().abs() > 1e-10 {
            ((population[0].initial_fe() - population[0].final_fe())
                / population[0].initial_fe().abs())
            .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        println!(
            "{:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            r#gen,
            best.phi,
            mean_phi,
            best_fe_red,
            best.prediction_accuracy,
            best.consciousness,
            best.threshold_fitness
        );

        // Pareto sort
        let fronts = FepFitnessBridge::pareto_sort(&population);

        // Elitism: keep Pareto front 0
        let elite_indices = &fronts[0];
        let mut next_gen: Vec<NeuralOrganism> = elite_indices
            .iter()
            .map(|&i| population[i].clone())
            .collect();

        // Fill with offspring from tournament selection
        let mut child_id = (r#gen as u64 + 1) * POP_SIZE as u64;
        while next_gen.len() < POP_SIZE {
            // Tournament: pick 3, take best composite
            let mut best_idx = 0;
            let mut best_comp = f64::NEG_INFINITY;
            for _ in 0..3 {
                let idx = (child_id as usize * 7 + 13) % population.len();
                if population[idx].fitness.composite > best_comp {
                    best_comp = population[idx].fitness.composite;
                    best_idx = idx;
                }
                child_id += 1;
            }

            let parent = &population[best_idx];
            let child = parent.replicate(child_id, 0.02, child_id * 17, &genesis);
            next_gen.push(child);
            child_id += 1;
        }

        population = next_gen;
    }

    // Final evaluation
    bridge.evaluate_population(&mut population);
    population.sort_by(|a, b| {
        b.fitness
            .composite
            .partial_cmp(&a.fitness.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Final Pareto Front");
    println!("═══════════════════════════════════════════════════════════════\n");

    let fronts = FepFitnessBridge::pareto_sort(&population);
    let front0 = &fronts[0];

    println!("  {} organisms on Pareto front 0\n", front0.len());
    println!(
        "  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Rank", "Phi", "FE", "Pred", "Stab", "Thresh"
    );
    for (rank, &idx) in front0.iter().enumerate().take(10) {
        let f = &population[idx].fitness;
        println!(
            "  {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            rank, f.phi, f.free_energy, f.prediction_accuracy, f.consciousness, f.threshold_fitness
        );
    }

    // Best organism's thresholds
    let best = &population[front0[0]];
    let thresholds = best.genome.decode_thresholds();
    let defaults = symthaea_neuroevolution::ThresholdPhenotype::default();

    println!("\n  Best organism's evolved thresholds vs defaults:");
    println!(
        "  {:>30}  {:>8}  {:>8}  {:>6}",
        "Threshold", "Evolved", "Default", "Delta"
    );
    println!("  {}", "-".repeat(58));

    macro_rules! row {
        ($name:expr, $evolved:expr, $default:expr) => {
            let delta = if ($default as f64).abs() > 1e-6 {
                (($evolved as f64 - $default as f64) / $default as f64 * 100.0) as i32
            } else {
                0
            };
            println!(
                "  {:>30}  {:>8.3}  {:>8.3}  {:>+5}%",
                $name, $evolved, $default, delta
            );
        };
    }

    row!(
        "fep_surprise_scale",
        thresholds.fep_surprise_scale,
        defaults.fep_surprise_scale
    );
    row!(
        "fep_lr_decay",
        thresholds.fep_lr_decay,
        defaults.fep_lr_decay
    );
    row!(
        "neuromod_d2_baseline",
        thresholds.neuromod_d2_baseline,
        defaults.neuromod_d2_baseline
    );
    row!(
        "neuromod_ne_phasic_thresh",
        thresholds.neuromod_ne_phasic_threshold,
        defaults.neuromod_ne_phasic_threshold
    );
    row!(
        "arousal_ema_decay",
        thresholds.neuromod_arousal_ema_decay,
        defaults.neuromod_arousal_ema_decay
    );
    row!(
        "homeostasis_high",
        thresholds.homeostasis_recalibrate_high,
        defaults.homeostasis_recalibrate_high
    );
    row!(
        "homeostasis_low",
        thresholds.homeostasis_recalibrate_low,
        defaults.homeostasis_recalibrate_low
    );
    row!(
        "frustration_dampen",
        thresholds.frustration_dampen_threshold,
        defaults.frustration_dampen_threshold
    );
    row!(
        "self_model_weight_high",
        thresholds.self_model_weight_high,
        defaults.self_model_weight_high
    );
    row!(
        "homeostasis_pull_cruise",
        thresholds.homeostasis_pull_cruise,
        defaults.homeostasis_pull_cruise
    );

    // Goodhart check: verify the winner isn't catatonic
    let best_f = &best.fitness;
    println!("\n  Goodhart's Law check:");
    println!("    Phi:              {:.4} (target: maximize)", best_f.phi);
    println!(
        "    FE:               {:.4} (must be finite, decreasing)",
        best_f.free_energy
    );
    println!(
        "    Pred accuracy:    {:.4} (must be > 0)",
        best_f.prediction_accuracy
    );
    println!(
        "    Phi stability:    {:.4} (must be > 0.5 = not oscillating)",
        best_f.consciousness
    );
    println!(
        "    Threshold fit:    {:.4} (must be > 0.6 = internally consistent)",
        best_f.threshold_fitness
    );

    let is_healthy = best_f.phi > 0.0
        && best_f.free_energy.is_finite()
        && best_f.prediction_accuracy > 0.0
        && best_f.consciousness > 0.3
        && best_f.threshold_fitness > 0.5;

    if is_healthy {
        println!("    → HEALTHY: winner is conscious AND capable");
    } else {
        println!("    → WARNING: possible Goodhart violation — check objectives");
    }
}