//! Clustering + purity evaluation.
//!
//! Phase 1 kill criterion: if cluster purity on a labeled Evtx corpus is below
//! 50%, the thesis dies. This module provides the purity metric and a thin
//! cluster-assignment helper.
//!
//! We deliberately keep the clusterer itself abstract — the spike's benchmark
//! example (`examples/cluster_evtx.rs`, to be added) can plug HDBSCAN,
//! k-means, or agglomerative clustering behind this interface without
//! changing the metric code.

use crate::encoder::{Hdv, cosine};
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};
use std::collections::HashMap;

/// A cluster assignment: one label per hypervector in input order.
/// `-1` is the HDBSCAN convention for "noise / unassigned".
pub type ClusterLabels = Vec<i32>;

/// Purity of a clustering against ground-truth labels.
///
/// For each cluster, take the mode of the ground-truth labels within it.
/// Purity = (sum over clusters of mode count) / total points.
/// Noise points (cluster == -1) are excluded from both numerator and
/// denominator — matching the standard HDBSCAN-purity convention.
///
/// Returns NaN if all points are noise.
pub fn purity(cluster_labels: &[i32], ground_truth: &[&str]) -> f32 {
    assert_eq!(cluster_labels.len(), ground_truth.len());

    // cluster_id -> (ground_truth_label -> count)
    let mut buckets: HashMap<i32, HashMap<&str, usize>> = HashMap::new();
    for (&cid, &gt) in cluster_labels.iter().zip(ground_truth.iter()) {
        if cid == -1 {
            continue;
        }
        *buckets.entry(cid).or_default().entry(gt).or_insert(0) += 1;
    }

    let mut correct = 0usize;
    let mut total = 0usize;
    for counts in buckets.values() {
        let cluster_total: usize = counts.values().sum();
        let mode: usize = *counts.values().max().unwrap_or(&0);
        correct += mode;
        total += cluster_total;
    }

    if total == 0 {
        return f32::NAN;
    }
    correct as f32 / total as f32
}

/// Run HDBSCAN over a set of bipolar hypervectors.
///
/// **Distance metric**: Euclidean on the i8→f64 conversion. For bipolar
/// {-1,+1} vectors of fixed dimension D:
///
///   ‖a − b‖² = 2·(D − dot(a,b)) = 2·D·(1 − cos(a,b))
///
/// so Euclidean distance is a monotonic function of cosine distance, and
/// HDBSCAN's density estimates transfer without modification.
///
/// Returns per-point cluster labels. `-1` is the HDBSCAN convention for
/// noise/unassigned points — `purity()` already handles these correctly.
///
/// **Parameter choice**: `min_cluster_size` defaults to `max(5, n/100)` —
/// this is a first-pass heuristic. Phase 1 benchmark should sweep it and
/// report the best-purity configuration.
pub fn hdbscan_cluster(
    points: &[Hdv],
    min_cluster_size: Option<usize>,
) -> Result<ClusterLabels, String> {
    if points.is_empty() {
        return Ok(Vec::new());
    }
    let data: Vec<Vec<f64>> = points
        .iter()
        .map(|hv| hv.iter().map(|&x| x as f64).collect())
        .collect();

    let n = points.len();
    let min_cluster = min_cluster_size.unwrap_or_else(|| (n / 100).max(5));

    let params = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster)
        .dist_metric(DistanceMetric::Euclidean)
        .build();

    let clusterer = Hdbscan::new(&data, params);
    clusterer.cluster().map_err(|e| format!("hdbscan: {e:?}"))
}

/// Brute-force nearest-centroid assignment.
///
/// Not a real clustering algorithm — used as a sanity baseline to confirm
/// the encoder separates obviously-different events before HDBSCAN is plugged
/// in. Given pre-computed centroids (one per known class), assigns each point
/// to its nearest centroid by cosine similarity.
pub fn nearest_centroid(points: &[Hdv], centroids: &[Hdv]) -> ClusterLabels {
    points
        .iter()
        .map(|p| {
            let mut best_idx = -1i32;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, c) in centroids.iter().enumerate() {
                let s = cosine(p, c);
                if s > best_sim {
                    best_sim = s;
                    best_idx = i as i32;
                }
            }
            best_idx
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purity_perfect() {
        let clusters = vec![0, 0, 1, 1];
        let gt = vec!["a", "a", "b", "b"];
        assert_eq!(purity(&clusters, &gt), 1.0);
    }

    #[test]
    fn purity_mixed() {
        // Cluster 0 has 2a + 1b → mode count 2
        // Cluster 1 has 1a + 1b → mode count 1
        // Total = 5, correct = 3, purity = 0.6
        let clusters = vec![0, 0, 0, 1, 1];
        let gt = vec!["a", "a", "b", "a", "b"];
        assert!((purity(&clusters, &gt) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn purity_ignores_noise() {
        let clusters = vec![-1, 0, 0];
        let gt = vec!["a", "b", "b"];
        assert_eq!(purity(&clusters, &gt), 1.0);
    }

    #[test]
    fn purity_all_noise_is_nan() {
        let clusters = vec![-1, -1];
        let gt = vec!["a", "b"];
        assert!(purity(&clusters, &gt).is_nan());
    }
}
