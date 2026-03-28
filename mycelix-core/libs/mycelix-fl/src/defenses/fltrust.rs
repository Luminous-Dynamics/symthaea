// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! FLTrust: direction-based Byzantine defense using a trusted server gradient.
//!
//! FLTrust (Cao et al. 2021) bootstraps trust from a small clean dataset held
//! by the server. Each client gradient is scored by its cosine similarity with
//! the server reference:
//!
//! 1. Compute cosine similarity between each client gradient and the server
//!    gradient.
//! 2. Clip negative similarities to zero (opposite direction implies Byzantine
//!    behavior).
//! 3. Normalize each client gradient to have the same L2 norm as the server
//!    gradient (prevents magnitude attacks).
//! 4. Compute the weighted average using the clipped similarities as weights.
//!
//! # Reference
//!
//! Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2021).
//! "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping."
//! NDSS 2021.

use crate::error::FlError;
use crate::types::{AggregationResult, Gradient};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2 norm of an `f32` slice, computed in `f64` for precision.
fn l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// Dot product of two `f32` slices, accumulated in `f64`.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum()
}

/// Cosine similarity between two vectors. Returns 0.0 if either vector has
/// near-zero norm (< 1e-12).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

// ---------------------------------------------------------------------------
// FLTrust
// ---------------------------------------------------------------------------

/// FLTrust defense: direction-based trust scoring against a server reference.
pub struct FLTrust;

impl FLTrust {
    /// Aggregate client gradients weighted by directional trust with respect
    /// to a trusted server gradient.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Client gradient submissions.
    /// * `server_gradient` - Gradient computed by the server on a small trusted
    ///   dataset. Must have the same dimensionality as the client gradients.
    ///
    /// # Algorithm
    ///
    /// For each client gradient `g_i`:
    ///   - `trust_i = max(0, cos(g_i, g_server))`
    ///   - `normalized_i = g_i * (||g_server|| / ||g_i||)`
    ///
    /// Result = `sum(trust_i * normalized_i) / sum(trust_i)`, then scaled to
    /// `||g_server||`.
    ///
    /// If all trust scores are zero (all clients are adversarial), the server
    /// gradient is returned as the fallback.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::EmptyGradients`] when `gradients` is empty.
    /// Returns [`FlError::ZeroServerGradient`] when the server gradient has
    /// near-zero norm (< 1e-12), making direction comparison meaningless.
    /// Returns [`FlError::DimensionMismatch`] when a client gradient has a
    /// different length than the server gradient.
    pub fn aggregate(
        gradients: &[Gradient],
        server_gradient: &[f32],
    ) -> Result<AggregationResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }

        let server_norm = l2_norm(server_gradient);
        if server_norm < 1e-12 {
            return Err(FlError::InvalidInput(
                "server gradient has near-zero norm".into(),
            ));
        }

        let dim = server_gradient.len();

        // Validate dimensions and compute trust scores.
        let mut trust_scores = Vec::with_capacity(gradients.len());
        for (_i, g) in gradients.iter().enumerate() {
            if g.values.len() != dim {
                return Err(FlError::DimensionMismatch {
                    expected: dim,
                    got: g.values.len(),
                });
            }

            let cos_sim = cosine_similarity(&g.values, server_gradient);
            // ReLU: only positive alignment contributes.
            trust_scores.push(cos_sim.max(0.0));
        }

        let total_trust: f64 = trust_scores.iter().sum();

        // If all clients are adversarial (all trust scores zero), fall back to
        // the server gradient itself.
        if total_trust < 1e-12 {
            let excluded_nodes: Vec<String> =
                gradients.iter().map(|g| g.node_id.clone()).collect();
            let score_pairs: Vec<(String, f64)> = gradients
                .iter()
                .zip(trust_scores.iter())
                .map(|(g, &t)| (g.node_id.clone(), t))
                .collect();
            return Ok(AggregationResult {
                gradient: server_gradient.to_vec(),
                included_nodes: vec![],
                excluded_nodes,
                scores: score_pairs,
            });
        }

        // Weighted aggregation with magnitude normalization.
        let mut aggregated = vec![0.0_f64; dim];
        for (g, &trust) in gradients.iter().zip(trust_scores.iter()) {
            if trust < 1e-15 {
                continue; // skip zero-trust clients entirely
            }

            let g_norm = l2_norm(&g.values);
            // Normalize client gradient to server gradient magnitude.
            let scale = if g_norm < 1e-12 {
                0.0
            } else {
                server_norm / g_norm
            };

            let weight = trust / total_trust;
            for (a, &v) in aggregated.iter_mut().zip(g.values.iter()) {
                *a += weight * scale * (v as f64);
            }
        }

        // Final normalization: scale the aggregate to have the server's norm.
        let agg_norm: f64 = aggregated.iter().map(|x| x * x).sum::<f64>().sqrt();
        if agg_norm > 1e-12 {
            let final_scale = server_norm / agg_norm;
            for v in &mut aggregated {
                *v *= final_scale;
            }
        }

        let result: Vec<f32> = aggregated.iter().map(|&x| x as f32).collect();

        let included_nodes: Vec<String> = gradients
            .iter()
            .zip(trust_scores.iter())
            .filter(|(_, &t)| t > 0.0)
            .map(|(g, _)| g.node_id.clone())
            .collect();
        let excluded_nodes: Vec<String> = gradients
            .iter()
            .zip(trust_scores.iter())
            .filter(|(_, &t)| t == 0.0)
            .map(|(g, _)| g.node_id.clone())
            .collect();
        let score_pairs: Vec<(String, f64)> = gradients
            .iter()
            .zip(trust_scores.iter())
            .map(|(g, &t)| (g.node_id.clone(), t))
            .collect();

        Ok(AggregationResult {
            gradient: result,
            included_nodes,
            excluded_nodes,
            scores: score_pairs,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn grad(values: &[f32]) -> Gradient {
        Gradient::new("node", values.to_vec(), 0)
    }

    // -----------------------------------------------------------------------
    // FLTrust core behavior
    // -----------------------------------------------------------------------

    #[test]
    fn fltrust_honest_aligned_gradients() {
        // All clients aligned with server: all should get high trust.
        let server = vec![1.0, 1.0, 1.0];
        let gradients = vec![
            grad(&[1.0, 1.0, 1.0]),
            grad(&[2.0, 2.0, 2.0]), // same direction, different magnitude
            grad(&[0.5, 0.5, 0.5]),
        ];

        let result = FLTrust::aggregate(&gradients, &server).unwrap();

        // All gradients point in the same direction as server; the result
        // should also point in that direction with server-magnitude norm.
        let server_norm = l2_norm(&server);
        let result_norm = l2_norm(&result.gradient);
        assert!(
            (result_norm - server_norm).abs() < 0.1,
            "Result norm ({:.4}) should be close to server norm ({:.4})",
            result_norm,
            server_norm
        );

        // Direction should match: all components roughly equal.
        let ratio = result.gradient[0] / result.gradient[1];
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Components should be roughly equal, ratio = {}",
            ratio
        );
    }

    #[test]
    fn fltrust_byzantine_opposite_direction_excluded() {
        // Server points in +x direction. Byzantine points in -x.
        let server = vec![1.0, 0.0, 0.0];
        let gradients = vec![
            grad(&[1.0, 0.1, 0.0]),    // honest, aligned
            grad(&[0.9, -0.1, 0.0]),   // honest, aligned
            grad(&[-10.0, 0.0, 0.0]),  // Byzantine: opposite direction
        ];

        let result = FLTrust::aggregate(&gradients, &server).unwrap();

        // The Byzantine node should be in excluded_nodes.
        assert!(
            result.excluded_nodes.iter().any(|id| id == "node"),
            "Byzantine node should be excluded"
        );

        // Result should point in +x direction (not dragged negative).
        assert!(
            result.gradient[0] > 0.5,
            "Result should point in +x direction, got {}",
            result.gradient[0]
        );
    }

    #[test]
    fn fltrust_all_orthogonal_near_zero() {
        // If all clients are orthogonal to server, trust scores are ~0.
        let server = vec![1.0, 0.0];
        let gradients = vec![
            grad(&[0.0, 1.0]),  // orthogonal
            grad(&[0.0, -1.0]), // orthogonal
        ];

        let result = FLTrust::aggregate(&gradients, &server).unwrap();

        // All trust scores zero => fallback to server gradient.
        assert!(result.included_nodes.is_empty(), "all nodes should be excluded");
        assert_eq!(result.gradient, vec![1.0, 0.0]);
    }

    #[test]
    fn fltrust_zero_server_gradient_error() {
        let server = vec![0.0, 0.0, 0.0];
        let gradients = vec![grad(&[1.0, 2.0, 3.0])];
        let err = FLTrust::aggregate(&gradients, &server).unwrap_err();
        assert!(matches!(err, FlError::InvalidInput(_)));
    }

    #[test]
    fn fltrust_empty_gradients_error() {
        let server = vec![1.0, 0.0];
        let err = FLTrust::aggregate(&[], &server).unwrap_err();
        assert!(matches!(err, FlError::EmptyGradients));
    }

    #[test]
    fn fltrust_dimension_mismatch_error() {
        let server = vec![1.0, 0.0, 0.0];
        let gradients = vec![grad(&[1.0, 0.0])]; // 2D vs 3D server
        let err = FLTrust::aggregate(&gradients, &server).unwrap_err();
        match err {
            FlError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("Expected DimensionMismatch, got: {:?}", other),
        }
    }

    #[test]
    fn fltrust_result_norm_matches_server() {
        // Regardless of client magnitudes, result should have server's norm.
        let server = vec![3.0, 4.0]; // norm = 5
        let gradients = vec![
            grad(&[30.0, 40.0]),   // 10x magnitude
            grad(&[0.3, 0.4]),     // 0.1x magnitude
            grad(&[6.0, 8.0]),     // 2x magnitude
        ];

        let result = FLTrust::aggregate(&gradients, &server).unwrap();
        let server_norm = l2_norm(&server);
        let result_norm = l2_norm(&result.gradient);

        assert!(
            (result_norm - server_norm).abs() < 0.1,
            "Result norm ({:.4}) should match server norm ({:.4})",
            result_norm,
            server_norm
        );
    }

    #[test]
    fn fltrust_mixed_honest_and_byzantine() {
        // 3 honest (pointing roughly +x) + 2 Byzantine (pointing -x).
        let server = vec![1.0, 0.0, 0.0];
        let gradients = vec![
            grad(&[1.0, 0.1, 0.0]),
            grad(&[1.0, -0.1, 0.0]),
            grad(&[1.0, 0.0, 0.1]),
            grad(&[-5.0, 0.0, 0.0]),  // Byzantine
            grad(&[-3.0, -1.0, 0.0]), // Byzantine
        ];

        let result = FLTrust::aggregate(&gradients, &server).unwrap();

        assert_eq!(result.included_nodes.len(), 3);
        assert_eq!(result.excluded_nodes.len(), 2);

        // Result should still point in +x.
        assert!(
            result.gradient[0] > 0.5,
            "Result x-component should be positive, got {}",
            result.gradient[0]
        );
    }

    // -----------------------------------------------------------------------
    // Helper unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_identical() {
        let a = [1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = [0.0_f32, 0.0];
        let b = [1.0_f32, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn l2_norm_unit() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
    }
}
