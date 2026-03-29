// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic stochastic engine (xoshiro256++) for reproducible simulations.

use serde::{Deserialize, Serialize};

/// Deterministic PRNG based on xoshiro256++.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticEngine {
    state: [u64; 4],
}

impl StochasticEngine {
    /// Create a new engine from a seed. Initializes state via SplitMix64.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Next u64 (xoshiro256++).
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);

        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussian via Box-Muller transform.
    pub fn next_gaussian(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }

    /// Uniform float in [0, 1). Alias for next_f64().
    pub fn uniform_f64(&mut self) -> f64 {
        self.next_f64()
    }

    /// Bernoulli trial: returns true with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Weighted choice: returns index selected proportional to weights.
    /// Panics if weights is empty or all zero.
    pub fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty(), "weighted_choice: empty weights");
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "weighted_choice: all weights zero");
        let mut r = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_with_same_seed() {
        let mut a = StochasticEngine::new(12345);
        let mut b = StochasticEngine::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_f64_in_range() {
        let mut rng = StochasticEngine::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_bernoulli_extremes() {
        let mut rng = StochasticEngine::new(42);
        // p=0 should always be false
        for _ in 0..100 {
            assert!(!rng.bernoulli(0.0));
        }
        // p=1 should always be true
        for _ in 0..100 {
            assert!(rng.bernoulli(1.0));
        }
    }

    #[test]
    fn test_weighted_choice_single() {
        let mut rng = StochasticEngine::new(42);
        assert_eq!(rng.weighted_choice(&[1.0]), 0);
    }

    #[test]
    fn test_gaussian_mean_approximate() {
        let mut rng = StochasticEngine::new(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_gaussian(5.0, 1.0)).sum();
        let mean = sum / n as f64;
        assert!((mean - 5.0).abs() < 0.1, "mean was {mean}");
    }
}
