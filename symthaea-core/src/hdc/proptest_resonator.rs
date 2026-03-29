// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for the Resonator Network module.
//! Tests algebraic properties of constraint satisfaction via coupled oscillators.

use super::resonator::{Constraint, ResonatorConfig, ResonatorNetwork};
use proptest::prelude::*;

// Use small dimensions for speed
const TEST_DIM: usize = 128;

/// Deterministic random vector generation using xorshift64.
/// Applies the seed-0 fix (XOR with golden ratio constant).
fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f32) / (u64::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

/// Generate a normalized random vector (unit length).
fn normalized_random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = random_vec(dim, seed);
    normalize(&mut v);
    v
}

/// Normalize a vector to unit length (local helper, mirrors resonator.rs private fn).
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

use crate::math::cosine_similarity_f32 as cosine_similarity;

/// Element-wise multiply (local helper, mirrors resonator.rs unbind).
fn unbind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    // -----------------------------------------------------------------------
    // 1. Cosine similarity is symmetric: cos(a, b) == cos(b, a)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_cosine_similarity_symmetric(seed_a in 1u64..100_000, seed_b in 100_001u64..200_000) {
        let a = random_vec(TEST_DIM, seed_a);
        let b = random_vec(TEST_DIM, seed_b);
        let ab = cosine_similarity(&a, &b);
        let ba = cosine_similarity(&b, &a);
        prop_assert!((ab - ba).abs() < 1e-6,
            "cos(a,b)={} != cos(b,a)={}", ab, ba);
    }

    // -----------------------------------------------------------------------
    // 2. Cosine similarity of a normalized vector with itself is ~1.0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_cosine_similarity_self_is_one(seed in 1u64..100_000) {
        let a = normalized_random_vec(TEST_DIM, seed);
        let sim = cosine_similarity(&a, &a);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "cos(a,a) should be ~1.0, got {}", sim);
    }

    // -----------------------------------------------------------------------
    // 3. Cosine similarity is bounded in [-1, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_cosine_similarity_bounded(seed_a in 1u64..100_000, seed_b in 100_001u64..200_000) {
        let a = random_vec(TEST_DIM, seed_a);
        let b = random_vec(TEST_DIM, seed_b);
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim >= -1.0 - 1e-6 && sim <= 1.0 + 1e-6,
            "cosine similarity out of bounds: {}", sim);
    }

    // -----------------------------------------------------------------------
    // 4. Normalizing twice gives the same result (idempotent)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_normalize_idempotent(seed in 1u64..100_000) {
        let mut v1 = random_vec(TEST_DIM, seed);
        normalize(&mut v1);
        let after_first = v1.clone();
        normalize(&mut v1);
        // v1 should be unchanged after second normalize
        for (a, b) in v1.iter().zip(after_first.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "normalize is not idempotent: {} vs {}", a, b);
        }
    }

    // -----------------------------------------------------------------------
    // 5. Normalized vector has norm ~1.0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_normalize_unit_length(seed in 1u64..100_000) {
        let v = normalized_random_vec(TEST_DIM, seed);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-5,
            "normalized vector should have unit length, got {}", norm);
    }

    // -----------------------------------------------------------------------
    // 6. unbind(a, b) == element-wise multiply (same as bind for continuous HDC)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_unbind_is_elementwise_multiply(seed_a in 1u64..100_000, seed_b in 100_001u64..200_000) {
        let a = normalized_random_vec(TEST_DIM, seed_a);
        let b = normalized_random_vec(TEST_DIM, seed_b);
        let result = unbind(&a, &b);
        for i in 0..TEST_DIM {
            let expected = a[i] * b[i];
            prop_assert!((result[i] - expected).abs() < 1e-7,
                "unbind[{}]={} != a*b={}", i, result[i], expected);
        }
    }

    // -----------------------------------------------------------------------
    // 7. solve produces a vector of correct dimension with all finite values
    // -----------------------------------------------------------------------
    #[test]
    fn prop_solve_produces_valid_output(seed_a in 1u64..100_000, seed_b in 100_001u64..200_000) {
        let left = normalized_random_vec(TEST_DIM, seed_a);
        let right = normalized_random_vec(TEST_DIM, seed_b);
        let constraint = Constraint::new(left, right);

        let mut network = ResonatorNetwork::new(TEST_DIM).unwrap();
        let solution = network.solve(&[constraint], Some(30)).unwrap();

        prop_assert_eq!(solution.vector.len(), TEST_DIM,
            "solution dimension mismatch");
        for (i, val) in solution.vector.iter().enumerate() {
            prop_assert!(val.is_finite(),
                "solution[{}] is not finite: {}", i, val);
        }
    }

    // -----------------------------------------------------------------------
    // 8. All energy values in history are finite
    // -----------------------------------------------------------------------
    #[test]
    fn prop_solve_energy_finite(seed_a in 1u64..100_000, seed_b in 100_001u64..200_000) {
        let left = normalized_random_vec(TEST_DIM, seed_a);
        let right = normalized_random_vec(TEST_DIM, seed_b);
        let constraint = Constraint::new(left, right);

        let mut network = ResonatorNetwork::new(TEST_DIM).unwrap();
        let _solution = network.solve(&[constraint], Some(30)).unwrap();

        for (i, &energy) in network.energy_history().iter().enumerate() {
            prop_assert!(energy.is_finite(),
                "energy_history[{}] is not finite: {}", i, energy);
        }
    }

    // -----------------------------------------------------------------------
    // 9. iterations <= max_iterations
    // -----------------------------------------------------------------------
    #[test]
    fn prop_solve_iterations_bounded(
        seed_a in 1u64..100_000,
        seed_b in 100_001u64..200_000,
        max_iter in 5usize..50
    ) {
        let left = normalized_random_vec(TEST_DIM, seed_a);
        let right = normalized_random_vec(TEST_DIM, seed_b);
        let constraint = Constraint::new(left, right);

        let mut network = ResonatorNetwork::new(TEST_DIM).unwrap();
        let solution = network.solve(&[constraint], Some(max_iter)).unwrap();

        prop_assert!(solution.iterations <= max_iter,
            "iterations {} exceeds max {}", solution.iterations, max_iter);
    }

    // -----------------------------------------------------------------------
    // 10. cleanup(noisy_symbol) has higher similarity to nearest codebook
    //     entry than a random vector does
    // -----------------------------------------------------------------------
    #[test]
    fn prop_cleanup_increases_codebook_similarity(
        seed_sym in 1u64..100_000,
        seed_noise in 100_001u64..200_000,
        seed_rand in 200_001u64..300_000
    ) {
        let dim = 256; // slightly larger for more stable statistics
        let sym = normalized_random_vec(dim, seed_sym);
        let random_vec_val = normalized_random_vec(dim, seed_rand);

        // Build network with one codebook entry
        let config = ResonatorConfig {
            temperature: 0.1,
            ..Default::default()
        };
        let mut network = ResonatorNetwork::with_config(dim, config).unwrap();
        network.add_symbol("target", sym.clone()).unwrap();

        // Create noisy version of the symbol (additive noise, then renormalize)
        let noise = random_vec(dim, seed_noise);
        let mut noisy: Vec<f32> = sym.iter().zip(noise.iter())
            .map(|(&s, &n)| s + 0.3 * n)
            .collect();
        normalize(&mut noisy);

        // Cleanup the noisy vector
        let cleaned = network.cleanup(&noisy);

        let sim_cleaned = cosine_similarity(&cleaned, &sym);
        let sim_random = cosine_similarity(&random_vec_val, &sym);

        prop_assert!(sim_cleaned > sim_random,
            "cleanup should produce higher similarity to codebook than random: \
             cleaned={}, random={}", sim_cleaned, sim_random);
    }
}
