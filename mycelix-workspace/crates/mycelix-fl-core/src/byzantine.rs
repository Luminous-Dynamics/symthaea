//! Multi-Signal Byzantine Detection
//!
//! Combines four detection strategies to catch adaptive adversaries:
//! 1. Magnitude anomaly (z-score and robust z-score)
//! 2. Direction anomaly (cosine similarity to mean gradient)
//! 3. Cross-validation (Krum-like neighbor agreement)
//! 4. Coordinate-wise anomaly (per-dimension z-scores)
//!
//! **Threshold disambiguation**: The thresholds in this module (`direction_threshold`,
//! `confidence_threshold`, `SignalWeights`) are gradient-space detection parameters —
//! they are NOT IIT Phi thresholds and NOT governance consciousness gates. For canonical
//! consciousness thresholds, see `mycelix_bridge_common::consciousness_thresholds`.

use crate::aggregation::euclidean_distance;
use crate::types::GradientUpdate;

/// Simple norm-based Byzantine detection result
#[derive(Debug, Clone)]
pub struct ByzantineDetectionResult {
    /// Indices of detected Byzantine participants
    pub byzantine_indices: Vec<usize>,
    /// Detection confidence scores
    pub confidence_scores: Vec<f32>,
    /// Whether early termination was triggered
    pub early_terminated: bool,
    /// Detection method used
    pub method: String,
}

impl ByzantineDetectionResult {
    pub fn has_byzantine(&self) -> bool {
        !self.byzantine_indices.is_empty()
    }

    pub fn byzantine_fraction(&self, total: usize) -> f32 {
        if total == 0 {
            0.0
        } else {
            self.byzantine_indices.len() as f32 / total as f32
        }
    }
}

/// Early norm-based Byzantine detector
///
/// Fast O(n) check for obvious outliers before running expensive detection.
pub struct EarlyByzantineDetector {
    max_norm: f32,
    min_norm: f32,
    z_threshold: f32,
}

impl EarlyByzantineDetector {
    pub fn new() -> Self {
        Self {
            max_norm: 1000.0,
            min_norm: 1e-10,
            z_threshold: 3.0,
        }
    }

    pub fn with_thresholds(max_norm: f32, min_norm: f32, z_threshold: f32) -> Self {
        Self {
            max_norm,
            min_norm,
            z_threshold,
        }
    }

    pub fn detect(&self, updates: &[GradientUpdate]) -> ByzantineDetectionResult {
        if updates.is_empty() {
            return ByzantineDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                early_terminated: false,
                method: "early_norm_check".to_string(),
            };
        }

        let mut byzantine_indices = Vec::new();
        let mut confidence_scores = Vec::new();

        let norms: Vec<f32> = updates.iter().map(|u| u.l2_norm()).collect();
        let mean_norm: f32 = norms.iter().sum::<f32>() / norms.len() as f32;
        let variance: f32 =
            norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f32>() / norms.len() as f32;
        let std_dev = variance.sqrt();

        for (i, &norm) in norms.iter().enumerate() {
            let mut confidence = 0.0f32;

            if norm > self.max_norm || norm < self.min_norm {
                confidence = 1.0;
            } else if std_dev > 0.0 {
                let z_score = (norm - mean_norm).abs() / std_dev;
                if z_score > self.z_threshold {
                    confidence = ((z_score - self.z_threshold) / self.z_threshold).min(1.0);
                }
            }

            if confidence > 0.5 {
                byzantine_indices.push(i);
                confidence_scores.push(confidence);
            }
        }

        let byzantine_fraction = byzantine_indices.len() as f32 / updates.len() as f32;
        let early_terminated = byzantine_fraction > 0.5;

        ByzantineDetectionResult {
            byzantine_indices,
            confidence_scores,
            early_terminated,
            method: "early_norm_check".to_string(),
        }
    }
}

impl Default for EarlyByzantineDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-Signal Byzantine Detector
// ============================================================================

/// Weights for combining detection signals
#[derive(Debug, Clone)]
pub struct SignalWeights {
    pub magnitude: f32,
    pub direction: f32,
    pub cross_validation: f32,
    pub coordinate: f32,
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            magnitude: 0.25,
            direction: 0.35,
            cross_validation: 0.25,
            coordinate: 0.15,
        }
    }
}

/// Per-participant signal breakdown
#[derive(Debug, Clone)]
pub struct SignalBreakdown {
    pub participant_idx: usize,
    pub magnitude_score: f32,
    pub direction_score: f32,
    pub cross_validation_score: f32,
    pub coordinate_score: f32,
    pub combined_score: f32,
}

/// Detection statistics
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    pub mean_norm: f32,
    pub std_norm: f32,
    pub mean_cosine_sim: f32,
    pub participants_analyzed: usize,
    pub signals_triggered: usize,
}

/// Multi-signal detection result
#[derive(Debug, Clone)]
pub struct MultiSignalDetectionResult {
    pub byzantine_indices: Vec<usize>,
    pub confidence_scores: Vec<f32>,
    pub signal_breakdown: Vec<SignalBreakdown>,
    pub early_terminated: bool,
    pub method: String,
    pub stats: DetectionStats,
}

/// Multi-signal Byzantine detector combining 4 detection strategies
pub struct MultiSignalByzantineDetector {
    magnitude_z_threshold: f32,
    direction_threshold: f32,
    cross_validation_k: f32,
    coordinate_z_threshold: f32,
    confidence_threshold: f32,
    signal_weights: SignalWeights,
}

impl Default for MultiSignalByzantineDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiSignalByzantineDetector {
    pub fn new() -> Self {
        Self {
            magnitude_z_threshold: 2.5,
            direction_threshold: 0.3,
            cross_validation_k: 0.6,
            coordinate_z_threshold: 3.0,
            confidence_threshold: 0.5,
            signal_weights: SignalWeights::default(),
        }
    }

    pub fn high_security() -> Self {
        Self {
            magnitude_z_threshold: 2.0,
            direction_threshold: 0.4,
            cross_validation_k: 0.7,
            coordinate_z_threshold: 2.5,
            confidence_threshold: 0.4,
            signal_weights: SignalWeights {
                magnitude: 0.2,
                direction: 0.4,
                cross_validation: 0.3,
                coordinate: 0.1,
            },
        }
    }

    pub fn relaxed() -> Self {
        Self {
            magnitude_z_threshold: 3.5,
            direction_threshold: 0.2,
            cross_validation_k: 0.5,
            coordinate_z_threshold: 4.0,
            confidence_threshold: 0.6,
            signal_weights: SignalWeights::default(),
        }
    }

    pub fn detect(&self, updates: &[GradientUpdate]) -> MultiSignalDetectionResult {
        if updates.is_empty() {
            return MultiSignalDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                signal_breakdown: vec![],
                early_terminated: false,
                method: "multi_signal".to_string(),
                stats: DetectionStats::default(),
            };
        }

        let n = updates.len();
        if n < 3 {
            return MultiSignalDetectionResult {
                byzantine_indices: vec![],
                confidence_scores: vec![],
                signal_breakdown: vec![],
                early_terminated: false,
                method: "multi_signal_insufficient".to_string(),
                stats: DetectionStats {
                    participants_analyzed: n,
                    ..Default::default()
                },
            };
        }

        let gradient_dim = updates[0].gradients.len();
        let mean_gradient = self.compute_mean_gradient(updates);

        // Compute norms and statistics
        let norms: Vec<f32> = updates.iter().map(|u| u.l2_norm()).collect();
        let mean_norm: f32 = norms.iter().sum::<f32>() / n as f32;
        let std_norm: f32 = {
            let var = norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f32>() / n as f32;
            var.sqrt()
        };

        // Robust statistics (median-based)
        let mut sorted_norms = norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_norm = if n.is_multiple_of(2) {
            (sorted_norms[n / 2 - 1] + sorted_norms[n / 2]) / 2.0
        } else {
            sorted_norms[n / 2]
        };
        let mut deviations: Vec<f32> = norms.iter().map(|&n| (n - median_norm).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if n.is_multiple_of(2) {
            (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
        } else {
            deviations[n / 2]
        };
        let normalized_mad = mad * 1.4826;

        // Cosine similarities
        let cosine_sims: Vec<f32> = updates
            .iter()
            .map(|u| cosine_similarity(&u.gradients, &mean_gradient))
            .collect();
        let mean_cosine: f32 = cosine_sims.iter().sum::<f32>() / n as f32;

        // Pairwise distances
        let distances = compute_distance_matrix(updates);

        let mut signal_breakdown = Vec::with_capacity(n);
        let mut byzantine_indices = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut signals_triggered = 0;

        for i in 0..n {
            // Signal 1: Magnitude anomaly (standard + robust z-score)
            let (magnitude_score, extreme_outlier) = {
                let std_z = if std_norm > 1e-10 {
                    (norms[i] - mean_norm).abs() / std_norm
                } else {
                    0.0
                };
                let robust_z = if normalized_mad > 1e-10 {
                    (norms[i] - median_norm).abs() / normalized_mad
                } else {
                    0.0
                };
                let z = std_z.max(robust_z);
                let extreme = robust_z > 4.0 || norms[i] > 100.0 * median_norm.max(1.0);
                let score = (z / self.magnitude_z_threshold).min(1.0);
                (score, extreme)
            };

            // Signal 2: Direction anomaly
            let direction_score = if cosine_sims[i] < self.direction_threshold {
                1.0 - (cosine_sims[i] / self.direction_threshold).max(0.0)
            } else {
                0.0
            };

            // Signal 3: Cross-validation (Krum-like)
            let cross_validation_score = self.compute_cross_validation_score(i, &distances, n);

            // Signal 4: Coordinate-wise outlier
            let coordinate_score =
                self.compute_coordinate_anomaly(&updates[i].gradients, updates, gradient_dim);

            let combined = self.signal_weights.magnitude * magnitude_score
                + self.signal_weights.direction * direction_score
                + self.signal_weights.cross_validation * cross_validation_score
                + self.signal_weights.coordinate * coordinate_score;

            signal_breakdown.push(SignalBreakdown {
                participant_idx: i,
                magnitude_score,
                direction_score,
                cross_validation_score,
                coordinate_score,
                combined_score: combined,
            });

            let is_byzantine = combined >= self.confidence_threshold || extreme_outlier;

            if is_byzantine {
                byzantine_indices.push(i);
                let confidence = if extreme_outlier { 1.0 } else { combined };
                confidence_scores.push(confidence);
                if magnitude_score > 0.5 {
                    signals_triggered += 1;
                }
                if direction_score > 0.5 {
                    signals_triggered += 1;
                }
                if cross_validation_score > 0.5 {
                    signals_triggered += 1;
                }
                if coordinate_score > 0.5 {
                    signals_triggered += 1;
                }
            }
        }

        let byzantine_fraction = byzantine_indices.len() as f32 / n as f32;
        let early_terminated = byzantine_fraction > 0.5;

        MultiSignalDetectionResult {
            byzantine_indices,
            confidence_scores,
            signal_breakdown,
            early_terminated,
            method: "multi_signal".to_string(),
            stats: DetectionStats {
                mean_norm,
                std_norm,
                mean_cosine_sim: mean_cosine,
                participants_analyzed: n,
                signals_triggered,
            },
        }
    }

    fn compute_mean_gradient(&self, updates: &[GradientUpdate]) -> Vec<f32> {
        if updates.is_empty() {
            return vec![];
        }
        let dim = updates[0].gradients.len();
        let n = updates.len() as f32;
        let mut mean = vec![0.0f32; dim];
        for update in updates {
            for (i, &g) in update.gradients.iter().enumerate() {
                mean[i] += g / n;
            }
        }
        mean
    }

    fn compute_cross_validation_score(&self, idx: usize, distances: &[Vec<f32>], n: usize) -> f32 {
        if n < 3 {
            return 0.0;
        }

        let k = ((n - 1) as f32 * self.cross_validation_k) as usize;
        let k = k.max(1).min(n - 1);

        let mut dists: Vec<f32> = distances[idx]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, &d)| d)
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let score_i: f32 = dists.iter().take(k).sum();

        let mut all_scores: Vec<f32> = (0..n)
            .map(|i| {
                let mut d: Vec<f32> = distances[i]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &d)| d)
                    .collect();
                d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                d.iter().take(k).sum()
            })
            .collect();

        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_score = all_scores.first().copied().unwrap_or(0.0);
        let max_score = all_scores.last().copied().unwrap_or(1.0);

        if (max_score - min_score).abs() < 1e-10 {
            return 0.0;
        }

        (score_i - min_score) / (max_score - min_score)
    }

    fn compute_coordinate_anomaly(
        &self,
        gradient: &[f32],
        updates: &[GradientUpdate],
        dim: usize,
    ) -> f32 {
        if updates.len() < 3 || dim == 0 {
            return 0.0;
        }

        let n = updates.len() as f32;
        let mut anomaly_count = 0;
        let sample_coords = dim.min(100);
        let step = (dim / sample_coords).max(1);

        for coord in (0..dim).step_by(step) {
            let values: Vec<f32> = updates.iter().map(|u| u.gradients[coord]).collect();
            let mean: f32 = values.iter().sum::<f32>() / n;
            let std: f32 = {
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
                var.sqrt()
            };

            if std > 1e-10 {
                let z = (gradient[coord] - mean).abs() / std;
                if z > self.coordinate_z_threshold {
                    anomaly_count += 1;
                }
            }
        }

        anomaly_count as f32 / sample_coords as f32
    }
}

/// Cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute pairwise distance matrix
fn compute_distance_matrix(updates: &[GradientUpdate]) -> Vec<Vec<f32>> {
    let n = updates.len();
    let mut distances = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(&updates[i].gradients, &updates[j].gradients);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_detection_empty() {
        let detector = EarlyByzantineDetector::new();
        let result = detector.detect(&[]);
        assert!(!result.has_byzantine());
    }

    #[test]
    fn test_early_detection_honest() {
        let detector = EarlyByzantineDetector::new();
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.12, 0.18], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.11, 0.22], 100, 0.5),
        ];
        let result = detector.detect(&updates);
        assert!(!result.has_byzantine());
    }

    #[test]
    fn test_early_detection_outlier() {
        let detector = EarlyByzantineDetector::new();
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.12, 0.18], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.11, 0.22], 100, 0.5),
            GradientUpdate::new("byz".into(), 1, vec![1000.0, -500.0], 100, 0.5),
        ];
        let result = detector.detect(&updates);
        assert!(result.has_byzantine());
        assert!(result.byzantine_indices.contains(&3));
    }

    #[test]
    fn test_multi_signal_empty() {
        let detector = MultiSignalByzantineDetector::new();
        let result = detector.detect(&[]);
        assert!(!result.early_terminated);
    }

    #[test]
    fn test_multi_signal_honest() {
        let detector = MultiSignalByzantineDetector::new();
        let updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.12, 0.18, 0.28], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.11, 0.22, 0.32], 100, 0.5),
            GradientUpdate::new("p4".into(), 1, vec![0.09, 0.21, 0.29], 100, 0.5),
        ];
        let result = detector.detect(&updates);
        assert!(result.byzantine_indices.is_empty());
    }

    #[test]
    fn test_multi_signal_detects_outlier() {
        let detector = MultiSignalByzantineDetector::new();
        let mut updates = vec![
            GradientUpdate::new("p1".into(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".into(), 1, vec![0.12, 0.18, 0.28], 100, 0.5),
            GradientUpdate::new("p3".into(), 1, vec![0.11, 0.22, 0.32], 100, 0.5),
            GradientUpdate::new("p4".into(), 1, vec![0.09, 0.21, 0.29], 100, 0.5),
        ];
        updates.push(GradientUpdate::new(
            "byz".into(),
            1,
            vec![100.0, -50.0, 200.0],
            100,
            0.5,
        ));
        let result = detector.detect(&updates);
        assert!(result.byzantine_indices.contains(&4));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - (-1.0)).abs() < 0.001);
    }
}
