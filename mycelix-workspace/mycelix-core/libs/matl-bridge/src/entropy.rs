// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Entropy calculation for trust diversity
//!
//! Entropy measures the diversity of trust sources. Higher entropy indicates
//! that trust is derived from multiple independent dimensions, making it
//! more robust against manipulation.

use serde::{Deserialize, Serialize};
use mycelix_core_types::KVector;

/// Calculate Shannon entropy of a probability distribution
///
/// H(p) = -Σ p_i * log(p_i)
pub fn shannon_entropy(probabilities: &[f32]) -> f32 {
    let sum: f32 = probabilities.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }

    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let normalized = p / sum;
            -normalized * normalized.ln()
        })
        .sum()
}

/// Calculate normalized entropy (0-1 range)
///
/// Normalized by maximum possible entropy (uniform distribution)
pub fn normalized_entropy(probabilities: &[f32]) -> f32 {
    let n = probabilities.len();
    if n <= 1 {
        return 0.0;
    }

    let max_entropy = (n as f32).ln();
    if max_entropy == 0.0 {
        return 0.0;
    }

    let entropy = shannon_entropy(probabilities);
    (entropy / max_entropy).min(1.0)
}

/// Calculate entropy score from K-Vector
///
/// Measures how evenly distributed the trust components are.
/// Higher score = trust comes from multiple dimensions.
pub fn k_vector_entropy(k_vector: &KVector) -> f32 {
    normalized_entropy(&k_vector.to_array())
}

/// Entropy-based trust score component for MATL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyScore {
    /// Raw Shannon entropy
    pub raw_entropy: f32,
    /// Normalized entropy (0-1)
    pub normalized: f32,
    /// Dominant dimension index (0-7)
    pub dominant_dimension: usize,
    /// How dominant the leading dimension is (0-1)
    pub dominance_ratio: f32,
}

impl EntropyScore {
    /// Calculate entropy score from K-Vector
    pub fn from_k_vector(k_vector: &KVector) -> Self {
        let values = k_vector.to_array();

        // Calculate entropy
        let raw_entropy = shannon_entropy(&values);
        let normalized = normalized_entropy(&values);

        // Find dominant dimension
        let (dominant_dimension, &max_value) = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Calculate dominance ratio
        let sum: f32 = values.iter().sum();
        let dominance_ratio = if sum > 0.0 {
            max_value / sum
        } else {
            0.0
        };

        Self {
            raw_entropy,
            normalized,
            dominant_dimension,
            dominance_ratio,
        }
    }

    /// Get the MATL-compatible entropy score (0-1)
    pub fn score(&self) -> f32 {
        // Combine normalized entropy with anti-dominance factor
        // Penalize when one dimension dominates too much
        let anti_dominance = 1.0 - self.dominance_ratio.powf(2.0);
        (self.normalized * 0.7 + anti_dominance * 0.3).min(1.0)
    }

    /// Check if entropy is healthy (diverse trust sources)
    pub fn is_healthy(&self) -> bool {
        self.normalized > 0.5 && self.dominance_ratio < 0.5
    }
}

/// Calculate Gini coefficient (inequality measure)
///
/// G = 0 means perfect equality, G = 1 means perfect inequality
pub fn gini_coefficient(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let sum: f32 = values.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }

    // Sort values for Gini calculation
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate Gini using cumulative sums
    let mut cumsum = 0.0;
    let mut weighted_sum = 0.0;

    for (i, &v) in sorted.iter().enumerate() {
        cumsum += v;
        weighted_sum += (i + 1) as f32 * v;
    }

    if cumsum == 0.0 {
        return 0.0;
    }

    (2.0 * weighted_sum) / (n as f32 * cumsum) - (n as f32 + 1.0) / n as f32
}

/// Network entropy - measures diversity across a set of agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEntropy {
    /// Average entropy across agents
    pub avg_entropy: f32,
    /// Minimum entropy (least diverse agent)
    pub min_entropy: f32,
    /// Maximum entropy (most diverse agent)
    pub max_entropy: f32,
    /// Gini coefficient of trust distribution
    pub trust_gini: f32,
    /// Number of agents with healthy entropy
    pub healthy_count: usize,
    /// Total agents
    pub total_agents: usize,
}

impl NetworkEntropy {
    /// Calculate network entropy from a set of K-Vectors
    pub fn from_k_vectors(k_vectors: &[KVector]) -> Self {
        if k_vectors.is_empty() {
            return Self {
                avg_entropy: 0.0,
                min_entropy: 0.0,
                max_entropy: 0.0,
                trust_gini: 0.0,
                healthy_count: 0,
                total_agents: 0,
            };
        }

        let scores: Vec<EntropyScore> = k_vectors
            .iter()
            .map(EntropyScore::from_k_vector)
            .collect();

        let entropies: Vec<f32> = scores.iter().map(|s| s.normalized).collect();

        let avg_entropy = entropies.iter().sum::<f32>() / entropies.len() as f32;
        let min_entropy = entropies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_entropy = entropies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Calculate Gini of trust scores
        let trust_scores: Vec<f32> = k_vectors.iter().map(|k| k.trust_score()).collect();
        let trust_gini = gini_coefficient(&trust_scores);

        let healthy_count = scores.iter().filter(|s| s.is_healthy()).count();

        Self {
            avg_entropy,
            min_entropy,
            max_entropy,
            trust_gini,
            healthy_count,
            total_agents: k_vectors.len(),
        }
    }

    /// Check if network entropy is healthy
    pub fn is_healthy(&self) -> bool {
        self.avg_entropy > 0.4 && self.trust_gini < 0.6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_entropy_is_maximum() {
        // Uniform distribution should have maximum normalized entropy
        let uniform = vec![0.125; 8]; // 1/8 each
        let entropy = normalized_entropy(&uniform);
        assert!((entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_concentrated_entropy_is_low() {
        // One dominant value should have low entropy
        let concentrated = vec![0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04];
        let entropy = normalized_entropy(&concentrated);
        assert!(entropy < 0.5);
    }

    #[test]
    fn test_k_vector_entropy() {
        let k = KVector::neutral(); // All 0.5
        let entropy = k_vector_entropy(&k);
        assert!((entropy - 1.0).abs() < 0.01); // Should be maximum
    }

    #[test]
    fn test_entropy_score_healthy() {
        let k = KVector::neutral();
        let score = EntropyScore::from_k_vector(&k);
        assert!(score.is_healthy());
    }

    #[test]
    fn test_gini_coefficient_uniform() {
        let uniform = vec![1.0; 5];
        let gini = gini_coefficient(&uniform);
        assert!(gini.abs() < 0.01); // Should be ~0 for uniform
    }

    #[test]
    fn test_gini_coefficient_concentrated() {
        let concentrated = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let gini = gini_coefficient(&concentrated);
        assert!(gini > 0.5); // Should be high for concentrated
    }
}
