// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differential Privacy Mechanisms
//!
//! Implements noise-adding mechanisms for differential privacy:
//! - Gaussian mechanism: (ε,δ)-DP, optimal for L2 sensitivity
//! - Laplace mechanism: ε-DP, optimal for L1 sensitivity

use crate::{DpError, DpResult};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Laplace distribution for sampling
/// Implements inverse CDF method: X = μ - b * sign(U) * ln(1 - 2|U|)
struct LaplaceDistribution {
    location: f64,
    scale: f64,
}

impl LaplaceDistribution {
    fn new(location: f64, scale: f64) -> Option<Self> {
        if scale <= 0.0 {
            None
        } else {
            Some(Self { location, scale })
        }
    }
}

impl Distribution<f64> for LaplaceDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Sample U ~ Uniform(-0.5, 0.5)
        let u: f64 = rng.random::<f64>() - 0.5;
        // Inverse CDF: X = μ - b * sign(U) * ln(1 - 2|U|)
        self.location - self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

/// Trait for DP mechanisms that add noise to data
pub trait Mechanism: Send + Sync {
    /// Add noise to a single value
    fn add_noise(&self, value: f64) -> f64;

    /// Add noise to a vector of values
    fn apply(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.add_noise(v)).collect()
    }

    /// Add noise to a 2D array (e.g., gradient matrix)
    fn apply_2d(&self, values: &[Vec<f64>]) -> Vec<Vec<f64>> {
        values.iter().map(|row| self.apply(row)).collect()
    }

    /// Get the epsilon for this mechanism
    fn epsilon(&self) -> f64;

    /// Get the delta for this mechanism (0 for pure DP)
    fn delta(&self) -> f64;

    /// Get the noise scale (sigma for Gaussian, b for Laplace)
    fn noise_scale(&self) -> f64;
}

/// Gaussian Mechanism for (ε,δ)-differential privacy
///
/// Adds noise sampled from N(0, σ²) where σ = Δ * √(2 ln(1.25/δ)) / ε
/// Optimal for queries with L2 sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMechanism {
    /// Privacy parameter epsilon
    epsilon: f64,
    /// Privacy parameter delta
    delta: f64,
    /// L2 sensitivity of the query
    sensitivity: f64,
    /// Standard deviation of the Gaussian noise
    sigma: f64,
}

impl GaussianMechanism {
    /// Create a new Gaussian mechanism
    ///
    /// # Arguments
    /// * `sensitivity` - L2 sensitivity of the query
    /// * `epsilon` - Privacy parameter (lower = more privacy)
    /// * `delta` - Probability of privacy violation
    ///
    /// # Returns
    /// A new GaussianMechanism or error if parameters are invalid
    #[instrument(skip_all, fields(eps = epsilon, delta = delta, sens = sensitivity))]
    pub fn new(sensitivity: f64, epsilon: f64, delta: f64) -> DpResult<Self> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon(epsilon));
        }
        if delta <= 0.0 || delta >= 1.0 {
            return Err(DpError::InvalidDelta(delta));
        }
        if sensitivity <= 0.0 {
            return Err(DpError::InvalidSensitivity(sensitivity));
        }

        // σ = Δ * √(2 ln(1.25/δ)) / ε
        let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;

        debug!(sigma = sigma, "Created Gaussian mechanism");

        Ok(Self {
            epsilon,
            delta,
            sensitivity,
            sigma,
        })
    }

    /// Create from a specific sigma value (advanced usage)
    pub fn from_sigma(sigma: f64, epsilon: f64, delta: f64) -> DpResult<Self> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon(epsilon));
        }
        if delta <= 0.0 || delta >= 1.0 {
            return Err(DpError::InvalidDelta(delta));
        }
        if sigma <= 0.0 {
            return Err(DpError::InvalidSensitivity(sigma));
        }

        // Back-calculate sensitivity
        let sensitivity = sigma * epsilon / (2.0 * (1.25_f64 / delta).ln()).sqrt();

        Ok(Self {
            epsilon,
            delta,
            sensitivity,
            sigma,
        })
    }

    /// Get the standard deviation of the noise
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get the sensitivity
    pub fn sensitivity(&self) -> f64 {
        self.sensitivity
    }
}

impl Mechanism for GaussianMechanism {
    fn add_noise(&self, value: f64) -> f64 {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, self.sigma).expect("Invalid sigma");
        value + normal.sample(&mut rng)
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }

    fn delta(&self) -> f64 {
        self.delta
    }

    fn noise_scale(&self) -> f64 {
        self.sigma
    }
}

/// Laplace Mechanism for ε-differential privacy
///
/// Adds noise sampled from Lap(0, Δ/ε)
/// Optimal for queries with L1 sensitivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceMechanism {
    /// Privacy parameter epsilon
    epsilon: f64,
    /// L1 sensitivity of the query
    sensitivity: f64,
    /// Scale parameter b = Δ/ε
    scale: f64,
}

impl LaplaceMechanism {
    /// Create a new Laplace mechanism
    ///
    /// # Arguments
    /// * `sensitivity` - L1 sensitivity of the query
    /// * `epsilon` - Privacy parameter (lower = more privacy)
    ///
    /// # Returns
    /// A new LaplaceMechanism or error if parameters are invalid
    #[instrument(skip_all, fields(eps = epsilon, sens = sensitivity))]
    pub fn new(sensitivity: f64, epsilon: f64) -> DpResult<Self> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(DpError::InvalidSensitivity(sensitivity));
        }

        let scale = sensitivity / epsilon;

        debug!(scale = scale, "Created Laplace mechanism");

        Ok(Self {
            epsilon,
            sensitivity,
            scale,
        })
    }

    /// Get the scale parameter
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the sensitivity
    pub fn sensitivity(&self) -> f64 {
        self.sensitivity
    }
}

impl Mechanism for LaplaceMechanism {
    fn add_noise(&self, value: f64) -> f64 {
        let mut rng = rand::rng();
        let laplace = LaplaceDistribution::new(0.0, self.scale).expect("Invalid scale");
        value + laplace.sample(&mut rng)
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }

    fn delta(&self) -> f64 {
        0.0 // Pure DP
    }

    fn noise_scale(&self) -> f64 {
        self.scale
    }
}

/// Exponential Mechanism for discrete outputs
///
/// Used for selecting from a set of candidates based on quality scores.
#[derive(Debug, Clone)]
pub struct ExponentialMechanism {
    epsilon: f64,
    sensitivity: f64,
}

impl ExponentialMechanism {
    /// Create a new Exponential mechanism
    pub fn new(sensitivity: f64, epsilon: f64) -> DpResult<Self> {
        if epsilon <= 0.0 {
            return Err(DpError::InvalidEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(DpError::InvalidSensitivity(sensitivity));
        }

        Ok(Self { epsilon, sensitivity })
    }

    /// Select from candidates based on quality scores
    ///
    /// Higher quality scores are more likely to be selected.
    ///
    /// # Arguments
    /// * `scores` - Quality scores for each candidate
    ///
    /// # Returns
    /// Index of the selected candidate
    pub fn select(&self, scores: &[f64]) -> usize {
        if scores.is_empty() {
            return 0;
        }

        let mut rng = rand::rng();

        // Compute unnormalized probabilities: exp(ε * score / (2Δ))
        let factor = self.epsilon / (2.0 * self.sensitivity);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let weights: Vec<f64> = scores
            .iter()
            .map(|&s| ((s - max_score) * factor).exp())
            .collect();

        let total: f64 = weights.iter().sum();

        // Sample from the distribution
        let mut cumulative = 0.0;
        let threshold: f64 = rng.random::<f64>() * total;

        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return i;
            }
        }

        scores.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mechanism_creation() {
        let mech = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();
        assert!(mech.sigma() > 0.0);
        assert_eq!(mech.epsilon(), 1.0);
        assert_eq!(mech.delta(), 1e-5);
    }

    #[test]
    fn test_gaussian_invalid_params() {
        assert!(GaussianMechanism::new(1.0, -1.0, 1e-5).is_err());
        assert!(GaussianMechanism::new(1.0, 1.0, 0.0).is_err());
        assert!(GaussianMechanism::new(1.0, 1.0, 1.0).is_err());
        assert!(GaussianMechanism::new(-1.0, 1.0, 1e-5).is_err());
    }

    #[test]
    fn test_gaussian_noise_is_random() {
        let mech = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();
        let v1 = mech.add_noise(0.0);
        let v2 = mech.add_noise(0.0);
        // Very unlikely to be exactly equal
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_gaussian_apply_vector() {
        let mech = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();
        let values = vec![1.0, 2.0, 3.0];
        let noisy = mech.apply(&values);
        assert_eq!(noisy.len(), 3);
        // Values should be different due to noise
        assert_ne!(noisy[0], 1.0);
    }

    #[test]
    fn test_laplace_mechanism_creation() {
        let mech = LaplaceMechanism::new(1.0, 1.0).unwrap();
        assert_eq!(mech.scale(), 1.0);
        assert_eq!(mech.epsilon(), 1.0);
        assert_eq!(mech.delta(), 0.0);
    }

    #[test]
    fn test_laplace_invalid_params() {
        assert!(LaplaceMechanism::new(1.0, -1.0).is_err());
        assert!(LaplaceMechanism::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_exponential_mechanism() {
        let mech = ExponentialMechanism::new(1.0, 10.0).unwrap();
        let scores = vec![1.0, 5.0, 10.0];

        // With high epsilon, should almost always select highest score
        let mut count_best = 0;
        for _ in 0..100 {
            if mech.select(&scores) == 2 {
                count_best += 1;
            }
        }
        assert!(count_best > 80, "Exponential mechanism should prefer higher scores");
    }
}
