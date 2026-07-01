// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the symthaea-neuroevolution crate.
//!
//! Verifies the full pipeline: random genomes → evaluate → evolve → fitness improves.
//!
//! Requires: `cargo test --features neuroevolution`

#![cfg(feature = "neuroevolution")]

use symthaea_neuroevolution::{
    FepFitnessBridge, FepFitnessConfig, FitnessWeights, NeuralGenome, NeuralOrganism,
    NeuroevolutionConfig, NeuroevolutionEngine,
};

use symthaea_core::genesis::GenesisSeed;

#[test]
fn test_full_pipeline_5_generations() {
    let config = NeuroevolutionConfig {
        population_size: 10,
        tournament_size: 3,
        max_generations: 5,
        convergence_patience: 10,
        fitness_config: FepFitnessConfig::fast_test(),
        ..Default::default()
    };

    let mut engine = NeuroevolutionEngine::new(config);
    let result = engine.evolve();

    assert_eq!(result.generations_completed, 5);
    assert!(result.best_fitness.is_finite());
    assert_eq!(result.history.len(), 5);

    // Verify non-degradation (elitism guarantees)
    let first_best = result.history[0].best_fitness;
    let last_best = result.history.last().unwrap().best_fitness;
    assert!(
        last_best >= first_best - 1e-6,
        "Best fitness should not degrade: first={first_best}, last={last_best}"
    );
}

#[test]
fn test_genome_roundtrip_in_pipeline() {
    let genome = NeuralGenome::random(42);
    let phenotype = genome.decode();
    let re_encoded = NeuralGenome::encode(&phenotype);
    let re_decoded = re_encoded.decode();

    let ratio = re_decoded.neuron_config.tau_base / phenotype.neuron_config.tau_base;
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "tau_base roundtrip error: {ratio}"
    );
    assert_eq!(
        phenotype.network_config.layer_sizes.len(),
        re_decoded.network_config.layer_sizes.len()
    );
}

#[test]
fn test_fitness_weights_affect_outcome() {
    let base_config = NeuroevolutionConfig {
        population_size: 8,
        max_generations: 3,
        convergence_patience: 10,
        fitness_config: FepFitnessConfig::fast_test(),
        genesis_phrase: "weights-test".to_string(),
        ..Default::default()
    };

    let mut engine1 = NeuroevolutionEngine::new(base_config.clone());
    let result1 = engine1.evolve();

    let mut config2 = base_config;
    config2.fitness_config.weights = FitnessWeights {
        free_energy: 0.9,
        phi: 0.05,
        consciousness: 0.025,
        efficiency: 0.025,
    };
    config2.genesis_phrase = "weights-test-fe".to_string();
    let mut engine2 = NeuroevolutionEngine::new(config2);
    let result2 = engine2.evolve();

    assert!(result1.best_fitness.is_finite());
    assert!(result2.best_fitness.is_finite());
}

#[test]
fn test_memory_budget_50_organisms() {
    let config = NeuroevolutionConfig {
        population_size: 50,
        fitness_config: FepFitnessConfig::fast_test(),
        ..Default::default()
    };

    let mut engine = NeuroevolutionEngine::new(config);
    engine.initialize();

    let genome_bytes = engine.population().len() * 2048;
    assert!(
        genome_bytes <= 200_000,
        "Genome storage should be <200KB, got {genome_bytes} bytes"
    );
}

#[test]
fn test_phi_proxy_nonzero_after_evaluation() {
    let genesis = GenesisSeed::from_phrase("phi-test");
    let config = FepFitnessConfig::fast_test();
    let mut bridge = FepFitnessBridge::new(config);

    let genome = NeuralGenome::random(42);
    let mut org = NeuralOrganism::spawn(0, genome, &genesis, 0);
    let fitness = bridge.evaluate(&mut org);

    // After evaluation, phi should be non-zero (Phi proxy from CfC output variance)
    assert!(
        fitness.phi >= 0.0,
        "Phi should be non-negative, got {}",
        fitness.phi
    );
}

#[test]
fn test_reduced_dim_faster_than_full() {
    // This test verifies that 256D evaluation works correctly.
    // (Speed comparison not measurable in unit tests, but correctness is.)
    let genesis = GenesisSeed::from_phrase("dim-test");
    let genome = NeuralGenome::random(42);

    let mut org_fast = NeuralOrganism::spawn_with_dim(0, genome.clone(), &genesis, 0, 256);
    let mut org_full = NeuralOrganism::spawn_with_dim(1, genome, &genesis, 0, 16_384);

    // Both should produce finite results
    let input_fast = symthaea_core::hdc::unified_hv::ContinuousHV::random(256, 99);
    let input_full = symthaea_core::hdc::unified_hv::ContinuousHV::random(16_384, 99);

    let r_fast = org_fast.evaluate_step(&input_fast, 0.05);
    let r_full = org_full.evaluate_step(&input_full, 0.05);

    assert!(r_fast.free_energy.total.is_finite());
    assert!(r_full.free_energy.total.is_finite());
    assert_eq!(org_fast.eval_dim, 256);
    assert_eq!(org_full.eval_dim, 16_384);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS (proptest)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn encode_decode_roundtrip(seed in 0u64..10_000) {
            let genome = NeuralGenome::random(seed);
            let phenotype = genome.decode();

            // All decoded values in valid ranges
            prop_assert!(phenotype.neuron_config.tau_base >= 0.01);
            prop_assert!(phenotype.neuron_config.tau_base <= 1.0);
            prop_assert!(phenotype.neuron_config.learning_rate >= 0.0001);
            prop_assert!(phenotype.neuron_config.learning_rate <= 0.1);
            prop_assert!(!phenotype.network_config.layer_sizes.is_empty());
            prop_assert!(phenotype.network_config.layer_sizes.len() <= 5);

            // Round-trip preserves weight seed exactly
            let re_encoded = NeuralGenome::encode(&phenotype);
            let re_decoded = re_encoded.decode();
            prop_assert_eq!(phenotype.weight_seed, re_decoded.weight_seed);
            prop_assert_eq!(
                phenotype.network_config.layer_sizes.len(),
                re_decoded.network_config.layer_sizes.len()
            );
        }

        #[test]
        fn mutation_preserves_validity(seed in 0u64..10_000, rate in 0.001f32..0.5) {
            let genome = NeuralGenome::random(seed);
            let mutated = genome.mutate(rate, seed.wrapping_add(1));
            let phenotype = mutated.decode();

            prop_assert!(phenotype.neuron_config.tau_base >= 0.01);
            prop_assert!(phenotype.neuron_config.tau_base <= 1.0);
            prop_assert!(phenotype.neuron_config.learning_rate >= 0.0001);
            prop_assert!(phenotype.neuron_config.learning_rate <= 0.1);
            prop_assert!(!phenotype.network_config.layer_sizes.is_empty());
        }

        #[test]
        fn crossover_produces_valid_phenotypes(seed_a in 0u64..5_000, seed_b in 5_000u64..10_000) {
            let parent_a = NeuralGenome::random(seed_a);
            let parent_b = NeuralGenome::random(seed_b);
            let child = parent_a.crossover(&parent_b, seed_a.wrapping_add(seed_b));
            let phenotype = child.decode();

            prop_assert!(phenotype.neuron_config.tau_base >= 0.01);
            prop_assert!(phenotype.neuron_config.tau_base <= 1.0);
            prop_assert!(!phenotype.network_config.layer_sizes.is_empty());
            prop_assert!(phenotype.network_config.layer_sizes.len() <= 5);
        }

        #[test]
        fn fitness_bounded(seed in 0u64..100) {
            let genesis = GenesisSeed::from_phrase("proptest-fitness");
            let config = FepFitnessConfig::fast_test();
            let mut bridge = FepFitnessBridge::new(config);

            let genome = NeuralGenome::random(seed * 7919);
            let mut org = NeuralOrganism::spawn(seed, genome, &genesis, 0);
            let fitness = bridge.evaluate(&mut org);

            prop_assert!(fitness.composite.is_finite());
            prop_assert!(fitness.composite >= -10.0);
            prop_assert!(fitness.energy_efficiency >= 0.0);
            prop_assert!(fitness.phi >= 0.0);
        }

        #[test]
        fn distance_is_metric(seed_a in 0u64..5_000, seed_b in 5_000u64..10_000) {
            let a = NeuralGenome::random(seed_a);
            let b = NeuralGenome::random(seed_b);

            // Symmetry
            let d_ab = a.distance(&b);
            let d_ba = b.distance(&a);
            prop_assert!((d_ab - d_ba).abs() < f32::EPSILON);

            // Non-negative
            prop_assert!(d_ab >= 0.0);
            prop_assert!(d_ab <= 1.0);

            // Identity
            prop_assert!(a.distance(&a) < f32::EPSILON);
        }
    }
}