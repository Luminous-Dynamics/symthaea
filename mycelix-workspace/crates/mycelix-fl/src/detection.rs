//! Multi-layer Byzantine detection for federated learning.
//!
//! Provides a 9-layer defense stack that identifies Byzantine participants
//! using gradient statistics, cosine similarity, historical behavior, and
//! MATL-based composite trust scoring.
//!
//! # Status
//! Stub implementation — layers 1-3 (statistics, cosine filter, reputation)
//! are functional. Layers 4-9 (cartel detection, temporal analysis, etc.) are
//! scaffolded for future implementation.

use crate::types::{CompressedGradient, DetectionResult, DetectionSignal, DetectionStats};

/// Configuration for multi-layer Byzantine detection
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Cosine similarity threshold for HV-space filtering (default: 0.1)
    pub cosine_threshold: f32,
    /// Reputation threshold below which a participant is flagged (default: 0.3)
    pub reputation_threshold: f32,
    /// Confidence threshold for flagging as Byzantine (default: 0.7)
    pub confidence_threshold: f32,
    /// Maximum fraction of participants that can be flagged (default: 0.45)
    pub max_byzantine_fraction: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            cosine_threshold: 0.1,
            reputation_threshold: 0.3,
            confidence_threshold: 0.7,
            max_byzantine_fraction: 0.45,
        }
    }
}

/// Multi-layer Byzantine detector operating in HV16 space.
///
/// Each layer independently evaluates gradients and produces a confidence
/// score. The final decision is a weighted combination of all layer signals.
pub struct MultiLayerDetector {
    config: DetectionConfig,
}

impl MultiLayerDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DetectionConfig::default())
    }

    /// Run all detection layers on the given gradients with reputation scores.
    ///
    /// Returns a `DetectionResult` containing which participants are flagged
    /// and the per-participant confidence breakdown.
    pub fn detect(
        &self,
        gradients: &[CompressedGradient],
        reputations: &std::collections::HashMap<String, f32>,
    ) -> DetectionResult {
        let n = gradients.len();
        let mut confidence_scores = vec![0.0_f32; n];
        let mut signal_breakdown = Vec::with_capacity(n);
        let mut layers_used = vec!["CosineFilter".to_string(), "ReputationGate".to_string()];

        // Layer 1: Cosine similarity filter in HV-space
        let reference_hv = compute_majority_hv(gradients);
        for (i, g) in gradients.iter().enumerate() {
            let sim = cosine_similarity_hv(&g.hv_data, &reference_hv);
            if sim < self.config.cosine_threshold {
                confidence_scores[i] += 0.4;
            }
        }

        // Layer 2: Reputation gate
        for (i, g) in gradients.iter().enumerate() {
            let rep = reputations.get(&g.participant_id).copied().unwrap_or(0.5);
            if rep < self.config.reputation_threshold {
                confidence_scores[i] += 0.3;
            }
        }

        // Layer 3: Quality score outlier detection
        if n >= 3 {
            let mean_quality: f32 = gradients.iter().map(|g| g.quality_score).sum::<f32>() / n as f32;
            let variance: f32 = gradients
                .iter()
                .map(|g| (g.quality_score - mean_quality).powi(2))
                .sum::<f32>()
                / n as f32;
            let std_dev = variance.sqrt();
            layers_used.push("QualityOutlier".to_string());

            for (i, g) in gradients.iter().enumerate() {
                if std_dev > 1e-6 {
                    let z_score = (g.quality_score - mean_quality).abs() / std_dev;
                    if z_score > 2.5 {
                        confidence_scores[i] += 0.3;
                    }
                }
            }
        }

        // Build per-participant signal breakdown
        for (i, g) in gradients.iter().enumerate() {
            let rep = reputations.get(&g.participant_id).copied().unwrap_or(0.5);
            let sim = cosine_similarity_hv(&g.hv_data, &reference_hv);
            signal_breakdown.push(DetectionSignal {
                participant_id: g.participant_id.clone(),
                is_byzantine: confidence_scores[i] >= self.config.confidence_threshold,
                confidence: confidence_scores[i].min(1.0),
                signals: vec![
                    ("cosine_similarity".to_string(), sim),
                    ("reputation".to_string(), rep),
                    ("quality_score".to_string(), g.quality_score),
                ],
            });
        }

        // Determine Byzantine set (respecting max fraction)
        let max_byzantine = (n as f32 * self.config.max_byzantine_fraction) as usize;
        let mut indexed_scores: Vec<(usize, f32)> = confidence_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut byzantine_indices = Vec::new();
        for (idx, score) in indexed_scores.iter().take(max_byzantine) {
            if *score >= self.config.confidence_threshold {
                byzantine_indices.push(*idx);
            }
        }

        let flagged_count = byzantine_indices.len();
        DetectionResult {
            byzantine_indices,
            confidence_scores,
            signal_breakdown,
            stats: DetectionStats {
                total_participants: n,
                flagged_count,
                detection_layers_used: layers_used,
            },
        }
    }
}

/// Compute majority (bundle) HV from a set of compressed gradients
fn compute_majority_hv(gradients: &[CompressedGradient]) -> Vec<u8> {
    if gradients.is_empty() {
        return vec![0u8; crate::types::HV16_BYTES];
    }

    let n = gradients.len();
    let threshold = n / 2;
    let mut result = vec![0u8; crate::types::HV16_BYTES];

    for byte_idx in 0..crate::types::HV16_BYTES {
        let mut result_byte = 0u8;
        for bit_idx in 0..8u8 {
            let mask = 1u8 << (7 - bit_idx);
            let ones: usize = gradients
                .iter()
                .map(|g| {
                    if g.hv_data.get(byte_idx).map_or(false, |&b| b & mask != 0) {
                        1
                    } else {
                        0
                    }
                })
                .sum();
            if ones > threshold {
                result_byte |= mask;
            }
        }
        result[byte_idx] = result_byte;
    }

    result
}

/// Cosine similarity between two binary HVs (bipolar encoding: 0→-1, 1→+1)
fn cosine_similarity_hv(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let total_bits = a.len() * 8;
    let mut dot: i32 = 0;
    for (&ba, &bb) in a.iter().zip(b.iter()) {
        let xnor = !(ba ^ bb); // 1 where bits agree
        dot += 2 * xnor.count_ones() as i32 - 8; // convert agree count to dot product
    }
    dot as f32 / total_bits as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GradientMetadata, HV16_BYTES};
    use std::collections::HashMap;

    fn make_gradient(id: &str, hv: Vec<u8>, quality: f32) -> CompressedGradient {
        CompressedGradient {
            participant_id: id.to_string(),
            hv_data: hv,
            original_dimension: 1000,
            quality_score: quality,
            metadata: GradientMetadata::new(1, quality),
        }
    }

    #[test]
    fn test_detect_no_gradients() {
        let detector = MultiLayerDetector::with_defaults();
        let result = detector.detect(&[], &HashMap::new());
        assert_eq!(result.byzantine_indices.len(), 0);
        assert_eq!(result.stats.total_participants, 0);
    }

    #[test]
    fn test_detect_honest_gradients() {
        let detector = MultiLayerDetector::with_defaults();
        let hv = vec![0xAA; HV16_BYTES];
        let gradients = vec![
            make_gradient("p1", hv.clone(), 0.9),
            make_gradient("p2", hv.clone(), 0.85),
            make_gradient("p3", hv.clone(), 0.88),
        ];
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9_f32);
        reps.insert("p2".to_string(), 0.85_f32);
        reps.insert("p3".to_string(), 0.88_f32);
        let result = detector.detect(&gradients, &reps);
        // All similar gradients with good reputations should not be flagged
        assert_eq!(result.byzantine_indices.len(), 0);
    }

    #[test]
    fn test_detect_low_reputation_flagged() {
        let detector = MultiLayerDetector::with_defaults();
        let honest_hv = vec![0xAA; HV16_BYTES];
        let bad_hv = vec![0x00; HV16_BYTES]; // Opposite of honest
        let gradients = vec![
            make_gradient("p1", honest_hv.clone(), 0.9),
            make_gradient("p2", honest_hv.clone(), 0.85),
            make_gradient("p3", honest_hv.clone(), 0.88),
            make_gradient("bad", bad_hv, 0.1), // Low quality + opposite HV
        ];
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9_f32);
        reps.insert("p2".to_string(), 0.85_f32);
        reps.insert("p3".to_string(), 0.88_f32);
        reps.insert("bad".to_string(), 0.1_f32); // Low reputation
        let result = detector.detect(&gradients, &reps);
        // The bad participant should be detected
        assert!(!result.byzantine_indices.is_empty());
    }
}
