// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Distance-based Byzantine defense algorithms: Krum, Multi-Krum, and Bulyan.
//!
//! # Krum (Blanchard et al. 2017)
//!
//! Selects the single gradient whose sum of squared distances to its nearest
//! `(n - f - 2)` neighbors is minimal. Byzantine-robust for `f < n/2 - 1`.
//!
//! # Multi-Krum
//!
//! Extends Krum by selecting the top-k gradients (ranked by Krum score) and
//! averaging them. Provides a smoother aggregate than single-selection Krum.
//!
//! # Bulyan (Mhamdi et al. 2018)
//!
//! Combines Multi-Krum selection with coordinate-wise trimmed mean. First
//! selects `n - 2f` gradients via Multi-Krum, then applies trimmed mean
//! (trimming the `f` most extreme values from each coordinate). Requires
//! `n >= 4f + 3` participants.
//!
//! # References
//!
//! - Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017).
//!   "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent."
//!   NeurIPS 2017.
//! - El Mhamdi, E. M., Guerraoui, R., & Rouault, S. (2018).
//!   "The Hidden Vulnerability of Distributed Learning in Byzantium."
//!   ICML 2018.

use crate::error::FlError;
use crate::types::{AggregationResult, Gradient};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute pairwise squared L2 distances between all gradient pairs.
///
/// Returns an `n x n` symmetric matrix where entry `[i][j]` is
/// `|| g_i - g_j ||^2`.  Diagonal entries are zero.
///
/// Complexity: O(n^2 * d) where `n` = number of gradients, `d` = dimension.
fn pairwise_squared_l2(gradients: &[Gradient]) -> Vec<Vec<f64>> {
    let n = gradients.len();
    let mut distances = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist: f64 = gradients[i]
                .values
                .iter()
                .zip(gradients[j].values.iter())
                .map(|(&a, &b)| {
                    let d = (a - b) as f64;
                    d * d
                })
                .sum();
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    distances
}

/// Compute the Krum score for every gradient.
///
/// The score for gradient `i` is the sum of squared distances to its
/// `m = n - f - 2` closest neighbors (excluding itself).
///
/// Returns `(scores, distances)` so callers can reuse the distance matrix.
fn krum_scores(
    gradients: &[Gradient],
    num_byzantine: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), FlError> {
    let n = gradients.len();

    // Krum requires n >= 2f + 3 so that m = n - f - 2 >= f + 1 > 0.
    let required = 2 * num_byzantine + 3;
    if n < required {
        return Err(FlError::InsufficientGradients {
            got: n,
            need: required,
        });
    }

    let m = n - num_byzantine - 2; // guaranteed > 0 by the check above
    let distances = pairwise_squared_l2(gradients);

    let scores: Vec<f64> = (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = distances[i].clone();
            // Partial sort: we only need the m+1 smallest (index 0 is self = 0.0).
            dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Sum indices 1..=m (skip the zero-distance self entry).
            dists[1..=m].iter().sum()
        })
        .collect();

    Ok((scores, distances))
}

/// Average a slice of gradients element-wise.
fn average_gradients(selected: &[&Gradient]) -> Vec<f32> {
    if selected.is_empty() {
        return Vec::new();
    }
    let dim = selected[0].values.len();
    let n = selected.len() as f32;
    let mut result = vec![0.0_f32; dim];
    for g in selected {
        for (r, &v) in result.iter_mut().zip(g.values.iter()) {
            *r += v;
        }
    }
    for r in &mut result {
        *r /= n;
    }
    result
}

// ---------------------------------------------------------------------------
// Krum
// ---------------------------------------------------------------------------

/// Krum defense: selects the single most central gradient.
pub struct Krum;

impl Krum {
    /// Aggregate gradients by selecting the one with the smallest Krum score.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Slice of participant gradients.
    /// * `num_byzantine` - Upper bound `f` on the number of Byzantine participants.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::InsufficientGradients`] when `n < 2f + 3`.
    /// Returns [`FlError::EmptyGradients`] when the slice is empty.
    pub fn aggregate(
        gradients: &[Gradient],
        num_byzantine: usize,
    ) -> Result<AggregationResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }
        if gradients.len() == 1 {
            return Ok(AggregationResult {
                gradient: gradients[0].values.clone(),
                included_nodes: vec![gradients[0].node_id.clone()],
                excluded_nodes: vec![],
                scores: vec![(gradients[0].node_id.clone(), 0.0)],
            });
        }

        let (scores, _distances) = krum_scores(gradients, num_byzantine)?;

        let selected_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap(); // safe: scores is non-empty

        let included_nodes = vec![gradients[selected_idx].node_id.clone()];
        let excluded_nodes: Vec<String> = gradients
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != selected_idx)
            .map(|(_, g)| g.node_id.clone())
            .collect();
        let score_pairs: Vec<(String, f64)> = gradients
            .iter()
            .zip(scores.iter())
            .map(|(g, &s)| (g.node_id.clone(), s))
            .collect();

        Ok(AggregationResult {
            gradient: gradients[selected_idx].values.clone(),
            included_nodes,
            excluded_nodes,
            scores: score_pairs,
        })
    }
}

// ---------------------------------------------------------------------------
// Multi-Krum
// ---------------------------------------------------------------------------

/// Multi-Krum defense: selects the top-k most central gradients and averages.
pub struct MultiKrum;

impl MultiKrum {
    /// Select the indices of the top-k gradients ranked by Krum score.
    ///
    /// `k` defaults to `n - num_byzantine` when `None`.
    fn select_indices(
        gradients: &[Gradient],
        num_byzantine: usize,
        k: Option<usize>,
    ) -> Result<Vec<usize>, FlError> {
        let n = gradients.len();
        let (scores, _) = krum_scores(gradients, num_byzantine)?;

        let k = k.unwrap_or(n.saturating_sub(num_byzantine)).min(n);

        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(indexed.into_iter().take(k).map(|(i, _)| i).collect())
    }

    /// Aggregate by selecting the top-k most central gradients and averaging.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Slice of participant gradients.
    /// * `num_byzantine` - Upper bound `f` on the number of Byzantine participants.
    /// * `k` - Number of gradients to select. Defaults to `n - f` when `None`.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::InsufficientGradients`] when `n < 2f + 3`.
    /// Returns [`FlError::EmptyGradients`] when the slice is empty.
    pub fn aggregate(
        gradients: &[Gradient],
        num_byzantine: usize,
        k: Option<usize>,
    ) -> Result<AggregationResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }
        if gradients.len() == 1 {
            return Ok(AggregationResult {
                gradient: gradients[0].values.clone(),
                included_nodes: vec![gradients[0].node_id.clone()],
                excluded_nodes: vec![],
                scores: vec![(gradients[0].node_id.clone(), 0.0)],
            });
        }

        let selected_indices = Self::select_indices(gradients, num_byzantine, k)?;
        let selected: Vec<&Gradient> = selected_indices.iter().map(|&i| &gradients[i]).collect();
        let averaged = average_gradients(&selected);

        let selected_set: std::collections::HashSet<usize> =
            selected_indices.iter().copied().collect();
        let included_nodes: Vec<String> = selected_indices
            .iter()
            .map(|&i| gradients[i].node_id.clone())
            .collect();
        let excluded_nodes: Vec<String> = gradients
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected_set.contains(i))
            .map(|(_, g)| g.node_id.clone())
            .collect();

        Ok(AggregationResult {
            gradient: averaged,
            included_nodes,
            excluded_nodes,
            scores: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Bulyan
// ---------------------------------------------------------------------------

/// Bulyan defense: Multi-Krum selection followed by coordinate-wise trimmed mean.
pub struct Bulyan;

impl Bulyan {
    /// Aggregate via Bulyan: Multi-Krum selection + trimmed mean.
    ///
    /// 1. Select `n - 2f` gradients using Multi-Krum.
    /// 2. For each coordinate, sort the selected values and trim `f` from each
    ///    end, then average the remaining middle values.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Slice of participant gradients.
    /// * `num_byzantine` - Upper bound `f` on the number of Byzantine participants.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::InsufficientGradients`] when `n < 4f + 3`.
    /// Returns [`FlError::EmptyGradients`] when the slice is empty.
    pub fn aggregate(
        gradients: &[Gradient],
        num_byzantine: usize,
    ) -> Result<AggregationResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }
        if gradients.len() == 1 {
            return Ok(AggregationResult {
                gradient: gradients[0].values.clone(),
                included_nodes: vec![gradients[0].node_id.clone()],
                excluded_nodes: vec![],
                scores: vec![],
            });
        }

        let n = gradients.len();
        let f = num_byzantine;

        // Bulyan requires n >= 4f + 3 for the trimmed-mean step to have
        // enough values after trimming f from each end.
        let required = 4 * f + 3;
        if n < required {
            return Err(FlError::InsufficientGradients {
                got: n,
                need: required,
            });
        }

        // Step 1: Multi-Krum selection of (n - 2f) gradients.
        let selection_k = n - 2 * f;
        let selected_indices = MultiKrum::select_indices(gradients, num_byzantine, Some(selection_k))?;
        let selected: Vec<&Gradient> = selected_indices.iter().map(|&i| &gradients[i]).collect();

        // Step 2: Coordinate-wise trimmed mean.
        let dim = selected[0].values.len();
        let m = selected.len(); // == n - 2f
        let trim = f; // trim f values from each end

        let mut result = vec![0.0_f32; dim];
        let mut col = Vec::with_capacity(m);

        for d in 0..dim {
            col.clear();
            for g in &selected {
                col.push(g.values[d]);
            }
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Average the middle values after trimming.
            let middle = &col[trim..m - trim];
            if middle.is_empty() {
                // Should not happen given the n >= 4f+3 constraint, but be safe.
                result[d] = col[m / 2];
            } else {
                let sum: f32 = middle.iter().sum();
                result[d] = sum / middle.len() as f32;
            }
        }

        let selected_set: std::collections::HashSet<usize> =
            selected_indices.iter().copied().collect();
        let included_nodes: Vec<String> = selected_indices
            .iter()
            .map(|&i| gradients[i].node_id.clone())
            .collect();
        let excluded_nodes: Vec<String> = gradients
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected_set.contains(i))
            .map(|(_, g)| g.node_id.clone())
            .collect();

        Ok(AggregationResult {
            gradient: result,
            included_nodes,
            excluded_nodes,
            scores: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a gradient from a slice.
    fn grad(values: &[f32]) -> Gradient {
        Gradient::new("node", values.to_vec(), 0)
    }

    /// Build a set of clustered "honest" gradients plus one distant Byzantine.
    fn honest_plus_byzantine() -> Vec<Gradient> {
        vec![
            grad(&[1.0, 1.0, 1.0]),  // honest 0
            grad(&[1.1, 0.9, 1.0]),  // honest 1
            grad(&[0.9, 1.1, 1.0]),  // honest 2
            grad(&[1.0, 1.0, 0.9]),  // honest 3
            grad(&[1.0, 1.0, 1.1]),  // honest 4
            grad(&[100.0, 100.0, 100.0]), // Byzantine
        ]
    }

    // -----------------------------------------------------------------------
    // Krum
    // -----------------------------------------------------------------------

    #[test]
    fn krum_selects_honest_node() {
        let gradients = honest_plus_byzantine();
        // f=1 Byzantine, n=6, requires 2*1+3=5 <= 6
        let result = Krum::aggregate(&gradients, 1).unwrap();

        // The selected gradient should be one of the honest ones (indices 0-4).
        // It must NOT be the Byzantine gradient [100, 100, 100].
        assert!(
            result.gradient[0] < 2.0,
            "Krum should select an honest gradient, got {:?}",
            result.gradient
        );
    }

    #[test]
    fn krum_identical_gradients() {
        let gradients = vec![
            grad(&[1.0, 2.0, 3.0]),
            grad(&[1.0, 2.0, 3.0]),
            grad(&[1.0, 2.0, 3.0]),
            grad(&[1.0, 2.0, 3.0]),
            grad(&[1.0, 2.0, 3.0]),
        ];
        let result = Krum::aggregate(&gradients, 1).unwrap();
        assert_eq!(result.gradient, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn krum_insufficient_gradients() {
        // f=2 requires 2*2+3 = 7, but we only have 4
        let gradients = vec![
            grad(&[1.0]),
            grad(&[2.0]),
            grad(&[3.0]),
            grad(&[4.0]),
        ];
        let err = Krum::aggregate(&gradients, 2).unwrap_err();
        match err {
            FlError::InsufficientGradients { got, need } => {
                assert_eq!(got, 4);
                assert_eq!(need, 7);
            }
            other => panic!("Expected InsufficientGradients, got: {:?}", other),
        }
    }

    #[test]
    fn krum_empty_gradients() {
        let err = Krum::aggregate(&[], 0).unwrap_err();
        assert!(matches!(err, FlError::EmptyGradients));
    }

    #[test]
    fn krum_single_gradient() {
        let gradients = vec![grad(&[5.0, 6.0])];
        let result = Krum::aggregate(&gradients, 0).unwrap();
        assert_eq!(result.gradient, vec![5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // Multi-Krum
    // -----------------------------------------------------------------------

    #[test]
    fn multikrum_excludes_byzantine() {
        let gradients = honest_plus_byzantine();
        // f=1, k defaults to n-f = 5: select 5 out of 6
        let result = MultiKrum::aggregate(&gradients, 1, None).unwrap();

        // The averaged result should be close to the honest centroid (~1.0).
        for &v in &result.gradient {
            assert!(
                (v - 1.0).abs() < 0.5,
                "Multi-Krum result should be near honest centroid, got {}",
                v
            );
        }
    }

    #[test]
    fn multikrum_closer_to_honest_than_fedavg() {
        let gradients = honest_plus_byzantine();
        let mk_result = MultiKrum::aggregate(&gradients, 1, None).unwrap();

        // FedAvg would naively include the Byzantine gradient.
        let all_refs: Vec<&Gradient> = gradients.iter().collect();
        let fedavg = average_gradients(&all_refs);

        // Honest centroid is approximately [1.0, 1.0, 1.0].
        let honest_centroid = [1.0_f32, 1.0, 1.0];

        let mk_dist: f32 = mk_result
            .gradient
            .iter()
            .zip(honest_centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let fedavg_dist: f32 = fedavg
            .iter()
            .zip(honest_centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        assert!(
            mk_dist < fedavg_dist,
            "Multi-Krum ({:.4}) should be closer to honest centroid than FedAvg ({:.4})",
            mk_dist,
            fedavg_dist
        );
    }

    #[test]
    fn multikrum_explicit_k() {
        let gradients = honest_plus_byzantine();
        // Select only 3 of 6
        let result = MultiKrum::aggregate(&gradients, 1, Some(3)).unwrap();
        assert_eq!(result.included_nodes.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Bulyan
    // -----------------------------------------------------------------------

    #[test]
    fn bulyan_requires_4f_plus_3() {
        // f=1 needs 4*1+3 = 7, providing only 6
        let gradients = honest_plus_byzantine(); // 6 gradients
        let err = Bulyan::aggregate(&gradients, 1).unwrap_err();
        match err {
            FlError::InsufficientGradients { got, need } => {
                assert_eq!(got, 6);
                assert_eq!(need, 7);
            }
            other => panic!("Expected InsufficientGradients, got: {:?}", other),
        }
    }

    #[test]
    fn bulyan_excludes_byzantine() {
        // f=1, need 7. 7 honest + 1 Byzantine = 8.
        let mut gradients = vec![
            grad(&[1.0, 1.0, 1.0]),
            grad(&[1.1, 0.9, 1.0]),
            grad(&[0.9, 1.1, 1.0]),
            grad(&[1.0, 1.0, 0.9]),
            grad(&[1.0, 1.0, 1.1]),
            grad(&[0.95, 1.05, 1.0]),
            grad(&[1.05, 0.95, 1.0]),
        ];
        gradients.push(grad(&[100.0, 100.0, 100.0])); // Byzantine

        let result = Bulyan::aggregate(&gradients, 1).unwrap();

        // Trimmed mean of honest gradients should be near [1.0, 1.0, 1.0].
        for &v in &result.gradient {
            assert!(
                (v - 1.0).abs() < 0.5,
                "Bulyan result should be near honest centroid, got {}",
                v
            );
        }
    }

    #[test]
    fn bulyan_more_robust_than_multikrum_against_coordinate_attack() {
        // A coordinate-wise attack: Byzantine gradient has one extreme
        // coordinate that might survive Multi-Krum but not Bulyan's trim.
        // f=1, need 7 for Bulyan. 7 honest + 2 Byzantine = 9 total (f=2 needs 11; use f=1).
        let mut gradients: Vec<Gradient> = (0..7)
            .map(|i| {
                let offset = (i as f32 - 3.0) * 0.05;
                grad(&[1.0 + offset, 1.0 - offset, 1.0])
            })
            .collect();
        // One Byzantine with a single extreme coordinate.
        gradients.push(grad(&[1.0, 1.0, 50.0]));

        // Multi-Krum (with default k = n - f = 7): might include the
        // Byzantine or at least its coordinate influence differs.
        let mk = MultiKrum::aggregate(&gradients, 1, None).unwrap();

        // Bulyan (f=1, n=8 >= 4*1+3=7): trimmed mean should eliminate
        // the extreme coordinate more aggressively.
        let bul = Bulyan::aggregate(&gradients, 1).unwrap();

        // The third coordinate should be closer to 1.0 for Bulyan.
        let mk_err = (mk.gradient[2] - 1.0).abs();
        let bul_err = (bul.gradient[2] - 1.0).abs();
        assert!(
            bul_err <= mk_err + 0.01, // allow tiny floating-point slack
            "Bulyan ({:.4}) should be at least as robust as Multi-Krum ({:.4}) on coordinate attack",
            bul_err,
            mk_err
        );
    }

    // -----------------------------------------------------------------------
    // Helper unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn pairwise_distances_symmetric_and_zero_diagonal() {
        let gradients = vec![grad(&[0.0, 0.0]), grad(&[3.0, 4.0])];
        let dists = pairwise_squared_l2(&gradients);
        assert_eq!(dists[0][0], 0.0);
        assert_eq!(dists[1][1], 0.0);
        assert!((dists[0][1] - 25.0).abs() < 1e-6); // 3^2 + 4^2
        assert_eq!(dists[0][1], dists[1][0]);
    }

    #[test]
    fn krum_scores_order() {
        // Two clusters: {0,1,2} near origin, {3} far away.
        // With f=0, m = 4-0-2 = 2.
        let gradients = vec![
            grad(&[0.0, 0.0]),
            grad(&[0.1, 0.0]),
            grad(&[0.0, 0.1]),
            grad(&[10.0, 10.0]),
        ];
        let (scores, _) = krum_scores(&gradients, 0).unwrap();
        // The outlier (index 3) should have the highest score.
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 3, "Outlier should have highest Krum score");
    }
}
