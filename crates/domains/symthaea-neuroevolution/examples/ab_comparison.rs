// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! A/B comparison: default CfC config vs. evolved champion.
//!
//! Evolves for 20 generations, then compares the champion's decoded config
//! against the hand-tuned default on the same temporal prediction task.
//!
//! Run: cargo run -p symthaea-neuroevolution --example ab_comparison --release

use std::time::Instant;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{UnifiedConfig, UnifiedNetworkConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_neuroevolution::{
    FepFitnessBridge, FepFitnessConfig, FitnessWeights, InputStrategy, NeuralGenome,
    NeuralOrganism, NeuroevolutionConfig, NeuroevolutionEngine,
};

fn main() {
    println!("=== A/B Comparison: Default vs. Evolved CfC Config ===\n");

    // Phase 1: Evolve for 20 generations
    println!("Phase 1: Evolution (20 generations, 256D for speed)...");
    let evo_start = Instant::now();

    let config = NeuroevolutionConfig {
        population_size: 20,
        tournament_size: 3,
        max_generations: 20,
        convergence_patience: 20,
        fitness_config: FepFitnessConfig {
            warmup_steps: 5,
            eval_steps: 30,
            dt: 0.05,
            input_dim: 512,
            input_strategy: InputStrategy::Temporal {
                frequencies: vec![1.0, 5.0, 10.0, 20.0],
                seed: 42,
            },
            weights: FitnessWeights::default(),
        },
        genesis_phrase: "ab-comparison".to_string(),
        ..Default::default()
    };

    let mut engine = NeuroevolutionEngine::new(config);
    let result = engine.evolve();
    let evo_time = evo_start.elapsed();

    println!(
        "  Evolved {} generations in {:.1}s",
        result.generations_completed,
        evo_time.as_secs_f64()
    );
    println!("  Best fitness: {:.6}", result.best_fitness);

    let evolved_phenotype = result.best_genome.decode();
    println!("  Evolved config:");
    println!(
        "    tau_base:     {:.4}",
        evolved_phenotype.neuron_config.tau_base
    );
    println!(
        "    backbone_tau: {:.4}",
        evolved_phenotype.neuron_config.backbone_tau
    );
    println!(
        "    LR:           {:.6}",
        evolved_phenotype.neuron_config.learning_rate
    );
    println!(
        "    momentum:     {:.4}",
        evolved_phenotype.neuron_config.momentum
    );
    println!(
        "    gating:       {:.4}",
        evolved_phenotype.neuron_config.gating_steepness
    );
    println!(
        "    layers:       {:?}",
        evolved_phenotype.network_config.layer_sizes
    );
    println!(
        "    activation:   {:?}",
        evolved_phenotype.neuron_config.activation
    );

    // Phase 2: A/B comparison
    println!("\nPhase 2: Comparing default vs. evolved on temporal prediction task...");

    let eval_config = FepFitnessConfig {
        warmup_steps: 10,
        eval_steps: 100,
        dt: 0.05,
        input_dim: 512,
        input_strategy: InputStrategy::Temporal {
            frequencies: vec![1.0, 5.0, 10.0, 20.0],
            seed: 999, // Different seed from training
        },
        weights: FitnessWeights::default(),
    };

    let genesis = GenesisSeed::from_phrase("ab-eval");

    // A: Default config organism
    let default_genome = NeuralGenome::random(0);
    let default_phenotype = default_genome.decode();
    let mut org_default = NeuralOrganism::spawn_with_dim(0, default_genome, &genesis, 0, 512);

    // B: Evolved config organism
    let evolved_genome = result.best_genome.clone();
    let mut org_evolved = NeuralOrganism::spawn_with_dim(1, evolved_genome, &genesis, 0, 512);

    // Evaluate both
    let mut bridge = FepFitnessBridge::new(eval_config);
    let fitness_default = bridge.evaluate(&mut org_default);
    let fitness_evolved = bridge.evaluate(&mut org_evolved);

    println!("\n--- Results ---");
    println!("                    Default     Evolved     Delta");
    println!(
        "  Composite:       {:+.6}   {:+.6}   {:+.6}",
        fitness_default.composite,
        fitness_evolved.composite,
        fitness_evolved.composite - fitness_default.composite
    );
    println!(
        "  Free Energy:     {:+.6}   {:+.6}   {:+.6}",
        fitness_default.free_energy,
        fitness_evolved.free_energy,
        fitness_evolved.free_energy - fitness_default.free_energy
    );
    println!(
        "  Phi:             {:+.6}   {:+.6}   {:+.6}",
        fitness_default.phi,
        fitness_evolved.phi,
        fitness_evolved.phi - fitness_default.phi
    );
    println!(
        "  Pred Accuracy:   {:+.6}   {:+.6}   {:+.6}",
        fitness_default.prediction_accuracy,
        fitness_evolved.prediction_accuracy,
        fitness_evolved.prediction_accuracy - fitness_default.prediction_accuracy
    );
    println!(
        "  Efficiency:      {:+.6}   {:+.6}   {:+.6}",
        fitness_default.energy_efficiency,
        fitness_evolved.energy_efficiency,
        fitness_evolved.energy_efficiency - fitness_default.energy_efficiency
    );

    let delta = fitness_evolved.composite - fitness_default.composite;
    if delta > 0.01 {
        println!(
            "\n  RESULT: Evolved config OUTPERFORMS default by {:.4}",
            delta
        );
    } else if delta < -0.01 {
        println!(
            "\n  RESULT: Default config outperforms evolved by {:.4} (evolution needs more generations or better fitness signal)",
            -delta
        );
    } else {
        println!("\n  RESULT: Configs are EQUIVALENT (delta < 0.01)");
    }

    // Print config comparison table
    println!("\n--- Config Comparison ---");
    println!("  Parameter        Default       Evolved");
    println!(
        "  tau_base:        {:.4}         {:.4}",
        default_phenotype.neuron_config.tau_base, evolved_phenotype.neuron_config.tau_base
    );
    println!(
        "  backbone_tau:    {:.4}         {:.4}",
        default_phenotype.neuron_config.backbone_tau, evolved_phenotype.neuron_config.backbone_tau
    );
    println!(
        "  learning_rate:   {:.6}       {:.6}",
        default_phenotype.neuron_config.learning_rate,
        evolved_phenotype.neuron_config.learning_rate
    );
    println!(
        "  gating:          {:.4}         {:.4}",
        default_phenotype.neuron_config.gating_steepness,
        evolved_phenotype.neuron_config.gating_steepness
    );
    println!(
        "  layers:          {:?}  {:?}",
        default_phenotype.network_config.layer_sizes, evolved_phenotype.network_config.layer_sizes
    );
}
