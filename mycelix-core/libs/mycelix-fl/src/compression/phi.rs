// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Approximate integrated information (Phi) for gradient sets.
//!
//! Ported from the Python `phi_measurement.py` implementation. Uses pairwise
//! cosine similarity as a proxy for information integration:
//!
//! - **Phi = total_integration - parts_integration**
//! - `total_integration` = average pairwise cosine similarity across all gradients
//! - `parts_integration` = 0 (individual gradients cannot integrate with themselves)
//!
//! Key insight: honest gradients increase system Phi (coherent learning),
//! while Byzantine gradients decrease Phi (information destruction).
//!
//! # Example
//!
//! ```
//! use mycelix_fl::compression::phi::PhiApproximator;
//! use mycelix_fl::Gradient;
//!
//! let gradients = vec![
//!     Gradient::new("a", vec![1.0, 2.0, 3.0], 1),
//!     Gradient::new("b", vec![1.1, 2.1, 3.1], 1),
//!     Gradient::new("c", vec![1.0, 1.9, 2.8], 1),
//! ];
//! let phi = PhiApproximator::approximate_phi(&gradients);
//! assert!(phi > 0.5, "Correlated gradients should have high Phi");
//! ```

use crate::types::{cosine_similarity, Gradient};

/// Approximate integrated information measurement for gradient sets.
pub struct PhiApproximator;

impl PhiApproximator {
    /// Compute approximate Phi for a gradient set.
    ///
    /// Higher Phi means the gradients are more integrated/coordinated across
    /// nodes (all learning in a coherent direction). Lower Phi means the
    /// gradient set contains conflicting or random information.
    ///
    /// Returns 0.0 for 0 or 1 gradients (nothing to integrate).
    pub fn approximate_phi(gradients: &[Gradient]) -> f64 {
        let n = gradients.len();
        if n <= 1 {
            return 0.0;
        }

        // Total integration = average pairwise cosine similarity (positive only).
        let mut total_sim = 0.0;
        let mut pairs = 0u64;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&gradients[i].values, &gradients[j].values);
                if sim > 0.0 {
                    total_sim += sim;
                }
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 0.0;
        }

        // Phi = total_integration (parts_integration is 0 by construction).
        total_sim / pairs as f64
    }

    /// Detect potential Byzantine gradients via Phi degradation.
    ///
    /// For each gradient, computes the system Phi with and without it.
    /// If removing a gradient **increases** Phi by more than `threshold_ratio`
    /// (e.g., 1.1 = 10%), that gradient is likely Byzantine.
    ///
    /// Returns a list of node IDs flagged as Byzantine and the system Phi.
    pub fn detect_byzantine(
        gradients: &[Gradient],
        threshold_ratio: f64,
    ) -> (Vec<String>, f64) {
        let system_phi = Self::approximate_phi(gradients);
        if system_phi <= 0.0 || gradients.len() <= 2 {
            return (Vec::new(), system_phi);
        }

        let mut flagged = Vec::new();

        for i in 0..gradients.len() {
            let without: Vec<Gradient> = gradients
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, g)| g.clone())
                .collect();
            let phi_without = Self::approximate_phi(&without);

            if phi_without > system_phi * threshold_ratio {
                flagged.push(gradients[i].node_id.clone());
            }
        }

        (flagged, system_phi)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(node: &str, values: Vec<f32>) -> Gradient {
        Gradient::new(node, values, 1)
    }

    #[test]
    fn test_identical_gradients_high_phi() {
        let gradients = vec![
            make_gradient("a", vec![1.0, 2.0, 3.0]),
            make_gradient("b", vec![1.0, 2.0, 3.0]),
            make_gradient("c", vec![1.0, 2.0, 3.0]),
        ];
        let phi = PhiApproximator::approximate_phi(&gradients);
        // Identical vectors => cosine sim = 1.0 => Phi ~ 1.0
        assert!(
            phi > 0.99,
            "Identical gradients should give Phi ~1.0, got {phi}"
        );
    }

    #[test]
    fn test_random_independent_gradients_low_phi() {
        // Use orthogonal-ish unit vectors: each gradient is non-zero in
        // a disjoint block of dimensions, so cosine similarity is exactly 0.
        let block = 400;
        let dim = block * 5;
        let gradients: Vec<Gradient> = (0..5)
            .map(|i| {
                let mut values = vec![0.0f32; dim];
                for j in (i * block)..((i + 1) * block) {
                    // Fill block with deterministic non-zero values.
                    values[j] = ((j * 7 + 3) % 100) as f32 / 50.0 - 1.0;
                }
                make_gradient(&format!("n{i}"), values)
            })
            .collect();
        let phi = PhiApproximator::approximate_phi(&gradients);
        // Disjoint support => cosine similarity = 0 => Phi = 0.
        assert!(
            phi < 0.01,
            "Orthogonal gradients should give Phi ~0, got {phi}"
        );
    }

    #[test]
    fn test_byzantine_outlier_drops_phi() {
        // 4 honest (correlated) + 1 byzantine (opposite direction).
        let honest = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let byzantine = vec![-5.0, -4.0, -3.0, -2.0, -1.0];

        let all = vec![
            make_gradient("h1", honest.clone()),
            make_gradient("h2", honest.iter().map(|x| x + 0.1).collect()),
            make_gradient("h3", honest.iter().map(|x| x - 0.1).collect()),
            make_gradient("h4", honest.iter().map(|x| x + 0.05).collect()),
            make_gradient("byz", byzantine),
        ];

        let phi_all = PhiApproximator::approximate_phi(&all);

        // Phi without byzantine should be higher.
        let honest_only = vec![
            make_gradient("h1", honest.clone()),
            make_gradient("h2", honest.iter().map(|x| x + 0.1).collect()),
            make_gradient("h3", honest.iter().map(|x| x - 0.1).collect()),
            make_gradient("h4", honest.iter().map(|x| x + 0.05).collect()),
        ];
        let phi_honest = PhiApproximator::approximate_phi(&honest_only);

        assert!(
            phi_honest > phi_all,
            "Removing Byzantine should increase Phi: honest={phi_honest}, all={phi_all}"
        );
    }

    #[test]
    fn test_detect_byzantine() {
        let honest = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let byzantine = vec![-10.0, -10.0, -10.0, -10.0, -10.0];

        let all = vec![
            make_gradient("h1", honest.clone()),
            make_gradient("h2", honest.iter().map(|x| x + 0.05).collect()),
            make_gradient("h3", honest.iter().map(|x| x - 0.05).collect()),
            make_gradient("byz", byzantine),
        ];

        let (flagged, _phi) = PhiApproximator::detect_byzantine(&all, 1.1);
        assert!(
            flagged.contains(&"byz".to_string()),
            "Should detect byzantine node, flagged: {flagged:?}"
        );
    }

    #[test]
    fn test_empty_and_single_gradient() {
        assert_eq!(PhiApproximator::approximate_phi(&[]), 0.0);
        let single = vec![make_gradient("a", vec![1.0, 2.0])];
        assert_eq!(PhiApproximator::approximate_phi(&single), 0.0);
    }
}
