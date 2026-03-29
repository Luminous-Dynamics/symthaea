// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differential privacy mechanisms for federated learning.
//!
//! Implements DP-SGD gradient clipping and Gaussian noise injection to provide
//! (ε, δ)-differential privacy guarantees for gradient submissions.
//!
//! # Privacy Model
//! - Each participant clips their gradient to L2 norm ≤ `clip_norm`
//! - Gaussian noise N(0, σ²) is added where σ = `clip_norm * noise_multiplier`
//! - Privacy budget is tracked across rounds
//!
//! # Status
//! Stub implementation — clipping and noise injection are functional.
//! Formal (ε, δ) accounting (Rényi DP / moments accountant) is scaffolded.

use serde::{Deserialize, Serialize};

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpConfig {
    /// L2 norm clipping threshold (default: 1.0)
    pub clip_norm: f32,
    /// Noise multiplier σ/C (default: 1.1 for moderate privacy)
    pub noise_multiplier: f32,
    /// Maximum total epsilon budget (default: 10.0)
    pub epsilon_budget: f32,
    /// Delta parameter — probability of privacy failure (default: 1e-5)
    pub delta: f64,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            clip_norm: 1.0,
            noise_multiplier: 1.1,
            epsilon_budget: 10.0,
            delta: 1e-5,
        }
    }
}

/// Privacy budget consumption per round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Total epsilon consumed so far
    pub epsilon_spent: f32,
    /// Configured budget cap
    pub epsilon_budget: f32,
    /// Number of rounds that consumed budget
    pub rounds_counted: u32,
}

impl PrivacyBudget {
    pub fn new(budget: f32) -> Self {
        Self {
            epsilon_spent: 0.0,
            epsilon_budget: budget,
            rounds_counted: 0,
        }
    }

    /// Remaining epsilon budget
    pub fn remaining(&self) -> f32 {
        (self.epsilon_budget - self.epsilon_spent).max(0.0)
    }

    /// Check if the budget allows another round
    pub fn has_budget(&self) -> bool {
        self.remaining() > 0.0
    }

    /// Record epsilon spent this round (simple composition — conservative)
    pub fn consume(&mut self, epsilon: f32) {
        self.epsilon_spent += epsilon;
        self.rounds_counted += 1;
    }
}

/// DP-SGD gradient privatizer.
///
/// Applies L2 clipping followed by Gaussian noise injection.
pub struct GradientPrivatizer {
    config: DpConfig,
}

impl GradientPrivatizer {
    pub fn new(config: DpConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DpConfig::default())
    }

    /// Clip a gradient to L2 norm ≤ `clip_norm`.
    pub fn clip(&self, gradient: &[f32]) -> Vec<f32> {
        let l2_sq: f32 = gradient.iter().map(|&x| x * x).sum();
        let l2 = l2_sq.sqrt();
        if l2 <= self.config.clip_norm || l2 < 1e-8 {
            return gradient.to_vec();
        }
        let scale = self.config.clip_norm / l2;
        gradient.iter().map(|&x| x * scale).collect()
    }

    /// Add calibrated Gaussian noise to a clipped gradient.
    ///
    /// Noise std = clip_norm * noise_multiplier.
    /// Uses a seeded PRNG for WASM compatibility (no std::time).
    pub fn add_noise(&self, gradient: &[f32], seed: u64) -> Vec<f32> {
        let sigma = self.config.clip_norm * self.config.noise_multiplier;
        let mut state = seed;
        gradient
            .iter()
            .map(|&x| {
                let noise = gaussian_sample(&mut state, sigma);
                x + noise
            })
            .collect()
    }

    /// Apply full DP-SGD privatization: clip then add noise.
    pub fn privatize(&self, gradient: &[f32], seed: u64) -> Vec<f32> {
        let clipped = self.clip(gradient);
        self.add_noise(&clipped, seed)
    }

    /// Estimate epsilon consumed for one round (simple Gaussian mechanism).
    ///
    /// Conservative upper bound: ε ≈ sqrt(2 ln(1.25/δ)) / σ
    pub fn epsilon_per_round(&self, delta: f64) -> f32 {
        let sigma = self.config.noise_multiplier;
        let factor = (2.0 * (1.25_f64 / delta).ln()).sqrt() as f32;
        factor / sigma
    }
}

/// Sample from N(0, sigma^2) using Box-Muller transform with xorshift PRNG.
fn gaussian_sample(state: &mut u64, sigma: f32) -> f32 {
    // Generate two uniform [0,1] samples
    let u1 = next_uniform(state);
    let u2 = next_uniform(state);

    // Box-Muller transform
    let mag = sigma * (-2.0 * u1.ln()).sqrt();
    mag * (2.0 * core::f32::consts::PI * u2).cos()
}

/// xorshift64 PRNG — produces uniform [ε, 1] sample
fn next_uniform(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    // Map to (0, 1] — avoid exact 0 for ln()
    (*state as f32 / u64::MAX as f32).max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_within_norm_unchanged() {
        let priv_ = GradientPrivatizer::with_defaults();
        let g = vec![0.5_f32, 0.5]; // L2 = sqrt(0.5) ≈ 0.707 < 1.0
        let clipped = priv_.clip(&g);
        assert!((clipped[0] - g[0]).abs() < 0.001);
        assert!((clipped[1] - g[1]).abs() < 0.001);
    }

    #[test]
    fn test_clip_exceeds_norm_scaled() {
        let priv_ = GradientPrivatizer::with_defaults();
        let g = vec![3.0_f32, 4.0]; // L2 = 5.0
        let clipped = priv_.clip(&g);
        let l2_sq: f32 = clipped.iter().map(|&x| x * x).sum();
        let l2 = l2_sq.sqrt();
        assert!((l2 - 1.0).abs() < 0.001, "Expected L2=1.0, got {}", l2);
    }

    #[test]
    fn test_clip_preserves_direction() {
        let priv_ = GradientPrivatizer::with_defaults();
        let g = vec![3.0_f32, 4.0];
        let clipped = priv_.clip(&g);
        // Direction preserved: ratio should match original
        let ratio = clipped[0] / clipped[1];
        let expected = g[0] / g[1];
        assert!((ratio - expected).abs() < 0.001);
    }

    #[test]
    fn test_noise_changes_gradient() {
        let priv_ = GradientPrivatizer::with_defaults();
        let g = vec![1.0_f32; 100];
        let noisy = priv_.add_noise(&g, 42);
        // With noise, the gradient should differ (extremely unlikely to be identical)
        let total_diff: f32 = g.iter().zip(noisy.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(total_diff > 0.001, "Noise had no effect");
    }

    #[test]
    fn test_epsilon_positive() {
        let priv_ = GradientPrivatizer::with_defaults();
        let epsilon = priv_.epsilon_per_round(1e-5);
        assert!(epsilon > 0.0);
    }

    #[test]
    fn test_budget_tracking() {
        let mut budget = PrivacyBudget::new(10.0);
        assert!(budget.has_budget());
        budget.consume(3.0);
        budget.consume(3.0);
        budget.consume(3.0);
        budget.consume(3.0);
        assert!(!budget.has_budget());
        assert_eq!(budget.rounds_counted, 4);
    }
}
