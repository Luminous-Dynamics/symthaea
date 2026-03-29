// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Composition for Differential Privacy
//!
//! Implements tighter privacy accounting methods:
//! - Moments Accountant (Abadi et al., 2016)
//! - Advanced Composition Theorem
//! - Rényi Differential Privacy

use crate::{DpError, DpResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Advanced composition theorem for (ε,δ)-DP
///
/// For k compositions of (ε,δ)-DP mechanisms, the total is:
/// (√(2k ln(1/δ')) ε + k ε(e^ε - 1), k δ + δ')
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedComposition {
    /// Base epsilon per query
    epsilon_per_query: f64,
    /// Base delta per query
    delta_per_query: f64,
    /// Number of compositions
    k: usize,
    /// Additional failure probability for composition
    delta_prime: f64,
}

impl AdvancedComposition {
    /// Create a new advanced composition calculator
    pub fn new(epsilon: f64, delta: f64, k: usize, delta_prime: f64) -> DpResult<Self> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon(epsilon));
        }
        if delta < 0.0 || delta >= 1.0 {
            return Err(DpError::InvalidDelta(delta));
        }
        if delta_prime <= 0.0 || delta_prime >= 1.0 {
            return Err(DpError::InvalidDelta(delta_prime));
        }

        Ok(Self {
            epsilon_per_query: epsilon,
            delta_per_query: delta,
            k,
            delta_prime,
        })
    }

    /// Compute total epsilon using advanced composition
    pub fn total_epsilon(&self) -> f64 {
        let k = self.k as f64;
        let e = self.epsilon_per_query;

        // √(2k ln(1/δ')) ε + k ε(e^ε - 1)
        (2.0 * k * (1.0 / self.delta_prime).ln()).sqrt() * e
            + k * e * (e.exp() - 1.0)
    }

    /// Compute total delta using advanced composition
    pub fn total_delta(&self) -> f64 {
        self.k as f64 * self.delta_per_query + self.delta_prime
    }

    /// Compare with naive (linear) composition
    pub fn naive_epsilon(&self) -> f64 {
        self.k as f64 * self.epsilon_per_query
    }

    /// Compute the savings from advanced composition
    pub fn epsilon_savings(&self) -> f64 {
        self.naive_epsilon() - self.total_epsilon()
    }
}

/// Moments Accountant for tight privacy accounting
///
/// Based on "Deep Learning with Differential Privacy" (Abadi et al., 2016)
/// Tracks privacy loss using the moments of the privacy loss random variable.
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Sampling probability (q = batch_size / dataset_size)
    sampling_probability: f64,
    /// Noise multiplier (σ/Δ)
    noise_multiplier: f64,
    /// Maximum moment order to compute
    max_moment_order: usize,
    /// Log moments for each order
    log_moments: Vec<f64>,
    /// Number of steps taken
    steps: usize,
    /// Target delta for converting to (ε,δ)-DP
    target_delta: f64,
}

impl MomentsAccountant {
    /// Create a new moments accountant
    ///
    /// # Arguments
    /// * `sampling_probability` - q = batch_size / dataset_size
    /// * `noise_multiplier` - σ (Gaussian noise stddev relative to sensitivity)
    /// * `target_delta` - Target δ for computing ε
    pub fn new(
        sampling_probability: f64,
        noise_multiplier: f64,
        target_delta: f64,
    ) -> DpResult<Self> {
        if sampling_probability <= 0.0 || sampling_probability > 1.0 {
            return Err(DpError::InvalidDelta(sampling_probability));
        }
        if noise_multiplier <= 0.0 {
            return Err(DpError::InvalidSensitivity(noise_multiplier));
        }
        if target_delta <= 0.0 || target_delta >= 1.0 {
            return Err(DpError::InvalidDelta(target_delta));
        }

        let max_moment_order = 32;

        info!(
            q = sampling_probability,
            sigma = noise_multiplier,
            delta = target_delta,
            "Created moments accountant"
        );

        Ok(Self {
            sampling_probability,
            noise_multiplier,
            max_moment_order,
            log_moments: vec![0.0; max_moment_order],
            steps: 0,
            target_delta,
        })
    }

    /// Record a training step (mini-batch gradient descent)
    pub fn record_step(&mut self) {
        self.steps += 1;

        // Pre-compute all moment bounds to avoid borrow issues
        let bounds: Vec<f64> = (0..self.max_moment_order)
            .map(|i| {
                let lambda = (i + 1) as f64;
                self.compute_moment_bound(lambda)
            })
            .collect();

        // Update log moments
        for (log_moment, bound) in self.log_moments.iter_mut().zip(bounds.iter()) {
            *log_moment += bound;
        }

        debug!(
            steps = self.steps,
            epsilon = self.current_epsilon(),
            "Recorded training step"
        );
    }

    /// Record multiple steps at once
    pub fn record_steps(&mut self, n: usize) {
        for _ in 0..n {
            self.record_step();
        }
    }

    /// Compute the log moment bound for order λ
    ///
    /// Uses the bound from Theorem 1 of Abadi et al.
    fn compute_moment_bound(&self, lambda: f64) -> f64 {
        let q = self.sampling_probability;
        let sigma = self.noise_multiplier;

        // For subsampled Gaussian mechanism, use the moment generating function bound
        // log E[e^(λ * privacy_loss)] ≤ q² λ (λ+1) / (1-q) / σ² + log(1 + q(e^(1/σ²) - 1))
        if sigma >= 1.0 {
            // Tight bound for large sigma
            q * q * lambda * (lambda + 1.0) / (2.0 * sigma * sigma)
        } else {
            // More conservative bound for small sigma
            let term1 = q * q * lambda * (lambda + 1.0) / (2.0 * sigma * sigma);
            let term2 = q * ((1.0 / (sigma * sigma)).exp() - 1.0);
            term1 + (1.0 + term2).ln()
        }
    }

    /// Get current epsilon using optimal moment order
    pub fn current_epsilon(&self) -> f64 {
        self.compute_epsilon(self.target_delta)
    }

    /// Compute epsilon for a given delta using optimal moment
    pub fn compute_epsilon(&self, delta: f64) -> f64 {
        let log_delta = delta.ln();

        // ε = min_λ (log_moment(λ) - log(δ)) / λ
        let mut min_epsilon = f64::INFINITY;

        for (i, &log_moment) in self.log_moments.iter().enumerate() {
            let lambda = (i + 1) as f64;
            let epsilon = (log_moment - log_delta) / lambda;

            if epsilon > 0.0 && epsilon < min_epsilon {
                min_epsilon = epsilon;
            }
        }

        if min_epsilon.is_infinite() {
            // Fallback to simple bound
            self.steps as f64 * self.sampling_probability / (self.noise_multiplier * self.noise_multiplier)
        } else {
            min_epsilon
        }
    }

    /// Get the number of steps recorded
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Get sampling probability
    pub fn sampling_probability(&self) -> f64 {
        self.sampling_probability
    }

    /// Get noise multiplier
    pub fn noise_multiplier(&self) -> f64 {
        self.noise_multiplier
    }

    /// Check if we can afford more steps given an epsilon budget
    pub fn can_afford_steps(&self, additional_steps: usize, max_epsilon: f64) -> bool {
        // Create a temporary copy to test
        let mut temp = self.clone();
        temp.record_steps(additional_steps);
        temp.current_epsilon() <= max_epsilon
    }

    /// Compute how many more steps we can take given a budget
    pub fn remaining_steps(&self, max_epsilon: f64) -> usize {
        let mut low = 0;
        let mut high = 1_000_000;

        while low < high {
            let mid = (low + high + 1) / 2;
            if self.can_afford_steps(mid, max_epsilon) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        low
    }

    /// Reset the accountant
    pub fn reset(&mut self) {
        self.log_moments.fill(0.0);
        self.steps = 0;
    }
}

/// Rényi Differential Privacy accountant
///
/// Uses Rényi divergence for privacy accounting, which composes nicely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenyiAccountant {
    /// Order α for Rényi divergence
    alpha: f64,
    /// Cumulative Rényi divergence
    rdp: f64,
    /// Number of compositions
    compositions: usize,
}

impl RenyiAccountant {
    /// Create a new Rényi accountant
    pub fn new(alpha: f64) -> DpResult<Self> {
        if alpha <= 1.0 {
            return Err(DpError::InvalidEpsilon(alpha));
        }

        Ok(Self {
            alpha,
            rdp: 0.0,
            compositions: 0,
        })
    }

    /// Add RDP guarantee from a Gaussian mechanism
    ///
    /// For Gaussian with sensitivity Δ and noise σ:
    /// ε_α = α Δ² / (2σ²)
    pub fn add_gaussian(&mut self, sensitivity: f64, sigma: f64) {
        let rdp_step = self.alpha * sensitivity * sensitivity / (2.0 * sigma * sigma);
        self.rdp += rdp_step;
        self.compositions += 1;

        debug!(
            rdp_step = rdp_step,
            total_rdp = self.rdp,
            compositions = self.compositions,
            "Added Gaussian mechanism to RDP"
        );
    }

    /// Add RDP guarantee from a subsampled Gaussian mechanism
    ///
    /// Uses the subsampling lemma for RDP
    pub fn add_subsampled_gaussian(&mut self, sensitivity: f64, sigma: f64, q: f64) {
        // Simplified bound for subsampling
        let base_rdp = self.alpha * sensitivity * sensitivity / (2.0 * sigma * sigma);
        let rdp_step = q * q * base_rdp; // Loose but simple bound

        self.rdp += rdp_step;
        self.compositions += 1;
    }

    /// Convert RDP to (ε, δ)-DP
    ///
    /// ε = rdp - ln(δ) / (α - 1)
    pub fn to_epsilon_delta(&self, delta: f64) -> (f64, f64) {
        let epsilon = self.rdp - delta.ln() / (self.alpha - 1.0);
        (epsilon.max(0.0), delta)
    }

    /// Get current RDP value
    pub fn rdp(&self) -> f64 {
        self.rdp
    }

    /// Get number of compositions
    pub fn compositions(&self) -> usize {
        self.compositions
    }

    /// Reset the accountant
    pub fn reset(&mut self) {
        self.rdp = 0.0;
        self.compositions = 0;
    }
}

/// Zero-Concentrated Differential Privacy (zCDP) accountant
///
/// zCDP provides clean composition: ρ_total = Σ ρ_i
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZcdpAccountant {
    /// Cumulative zCDP parameter ρ
    rho: f64,
    /// Number of compositions
    compositions: usize,
}

impl ZcdpAccountant {
    /// Create a new zCDP accountant
    pub fn new() -> Self {
        Self {
            rho: 0.0,
            compositions: 0,
        }
    }

    /// Add zCDP from a Gaussian mechanism
    ///
    /// For Gaussian with L2 sensitivity Δ and noise σ:
    /// ρ = Δ² / (2σ²)
    pub fn add_gaussian(&mut self, sensitivity: f64, sigma: f64) {
        let rho_step = sensitivity * sensitivity / (2.0 * sigma * sigma);
        self.rho += rho_step;
        self.compositions += 1;
    }

    /// Convert zCDP to (ε, δ)-DP
    ///
    /// ε = ρ + 2√(ρ ln(1/δ))
    pub fn to_epsilon_delta(&self, delta: f64) -> (f64, f64) {
        let log_inv_delta = (1.0 / delta).ln();
        let epsilon = self.rho + 2.0 * (self.rho * log_inv_delta).sqrt();
        (epsilon, delta)
    }

    /// Get current rho value
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Get number of compositions
    pub fn compositions(&self) -> usize {
        self.compositions
    }

    /// Reset the accountant
    pub fn reset(&mut self) {
        self.rho = 0.0;
        self.compositions = 0;
    }
}

impl Default for ZcdpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_composition() {
        let comp = AdvancedComposition::new(0.1, 1e-6, 100, 1e-5).unwrap();

        // Advanced composition should be tighter than naive
        assert!(comp.total_epsilon() < comp.naive_epsilon());

        // Should have positive savings
        assert!(comp.epsilon_savings() > 0.0);
    }

    #[test]
    fn test_moments_accountant_creation() {
        let acc = MomentsAccountant::new(0.01, 1.0, 1e-5).unwrap();
        assert_eq!(acc.steps(), 0);
        assert_eq!(acc.sampling_probability(), 0.01);
        assert_eq!(acc.noise_multiplier(), 1.0);
    }

    #[test]
    fn test_moments_accountant_steps() {
        let mut acc = MomentsAccountant::new(0.01, 1.0, 1e-5).unwrap();

        acc.record_step();
        assert_eq!(acc.steps(), 1);
        let eps1 = acc.current_epsilon();

        acc.record_step();
        assert_eq!(acc.steps(), 2);
        let eps2 = acc.current_epsilon();

        // Epsilon should increase with more steps
        assert!(eps2 > eps1);
    }

    #[test]
    fn test_moments_accountant_bulk() {
        let mut acc = MomentsAccountant::new(0.01, 1.0, 1e-5).unwrap();
        acc.record_steps(100);
        assert_eq!(acc.steps(), 100);

        // Should have non-trivial epsilon
        assert!(acc.current_epsilon() > 0.0);
    }

    #[test]
    fn test_renyi_accountant() {
        let mut acc = RenyiAccountant::new(2.0).unwrap();

        acc.add_gaussian(1.0, 1.0);
        assert_eq!(acc.compositions(), 1);
        assert!(acc.rdp() > 0.0);

        let (eps, delta) = acc.to_epsilon_delta(1e-5);
        assert!(eps > 0.0);
        assert_eq!(delta, 1e-5);
    }

    #[test]
    fn test_zcdp_accountant() {
        let mut acc = ZcdpAccountant::new();

        acc.add_gaussian(1.0, 1.0);
        assert!((acc.rho() - 0.5).abs() < 1e-10); // ρ = 1/(2*1) = 0.5

        acc.add_gaussian(1.0, 1.0);
        assert!((acc.rho() - 1.0).abs() < 1e-10); // Composition: 0.5 + 0.5 = 1.0

        let (eps, _delta) = acc.to_epsilon_delta(1e-5);
        assert!(eps > 0.0);
    }

    #[test]
    fn test_invalid_params() {
        assert!(MomentsAccountant::new(0.0, 1.0, 1e-5).is_err());
        assert!(MomentsAccountant::new(0.01, 0.0, 1e-5).is_err());
        assert!(MomentsAccountant::new(0.01, 1.0, 0.0).is_err());
    }
}
