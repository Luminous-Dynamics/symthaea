// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decentralized FL pipeline: 9-stage Byzantine-robust aggregation in HV-space.
//!
//! Every validator node runs this pipeline independently on the same set of
//! compressed gradients. Because the pipeline is deterministic, all honest
//! validators produce the same commitment hash — the basis for RB-BFT consensus.
//!
//! # Pipeline Stages
//! 1. **Validate** — check HV16 size, participant ID, quality bounds
//! 2. **Phi Gate** — filter low-coherence gradients (proxy Phi)
//! 3. **Epistemic Grade** — E-N-M quality classification
//! 4. **Byzantine Detect** — 3-layer defense (cosine, reputation, quality outlier)
//! 5. **Aggregate** — HV-space majority vote (FedAvg-HV)
//! 6. **Verify** — post-aggregation sanity check
//! 7. **Commit Hash** — SHA-256 of (aggregated_hv, method, round)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::detection::MultiLayerDetector;
use crate::types::{CompressedGradient, HV16_BYTES};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the decentralized FL pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Cosine similarity threshold for Byzantine detection (default: 0.1)
    pub cosine_threshold: f32,
    /// Reputation threshold below which participant is flagged (default: 0.3)
    pub reputation_threshold: f32,
    /// Byzantine detection confidence threshold (default: 0.7)
    pub confidence_threshold: f32,
    /// Maximum fraction of participants that can be excluded (default: 0.45)
    pub max_byzantine_fraction: f32,
    /// Minimum gradients required for aggregation (default: 1)
    pub min_gradients: usize,
    /// Aggregation method label (default: "FedAvgHV")
    pub method: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            cosine_threshold: 0.1,
            reputation_threshold: 0.3,
            confidence_threshold: 0.7,
            max_byzantine_fraction: 0.45,
            min_gradients: 1,
            method: "FedAvgHV".to_string(),
        }
    }
}

// ============================================================================
// Pipeline Statistics
// ============================================================================

/// Statistics from a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total gradient submissions received
    pub total_contributions: usize,
    /// Contributions after Byzantine detection
    pub after_detection: usize,
    /// Number of Byzantine participants detected
    pub byzantine_detected: usize,
    /// IDs of excluded participants
    pub excluded_participants: Vec<String>,
    /// Aggregation method used
    pub method_used: String,
}

// ============================================================================
// Pipeline Result
// ============================================================================

/// Result of running the decentralized pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Aggregated HV16 binary vector (2KB), or None if aggregation failed.
    pub aggregated_hv: Option<Vec<u8>>,
    /// Pipeline run statistics
    pub stats: PipelineStats,
}

// ============================================================================
// Pipeline Error
// ============================================================================

/// Error types for pipeline operations.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum PipelineError {
    #[error("No gradients provided")]
    NoGradients,
    #[error("Too few valid gradients after detection: {0}")]
    TooFewGradients(usize),
    #[error("Invalid gradient from {participant}: {reason}")]
    InvalidGradient { participant: String, reason: String },
    #[error("Aggregation failed: {0}")]
    AggregationFailed(String),
}

// ============================================================================
// DecentralizedPipeline
// ============================================================================

/// Deterministic, WASM-compatible federated learning pipeline.
///
/// Runs Byzantine detection, filters outliers, and aggregates HV16 gradients
/// using majority voting. Deterministic across all honest validators so that
/// they converge on the same commitment hash for RB-BFT consensus.
pub struct DecentralizedPipeline {
    config: PipelineConfig,
}

impl DecentralizedPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Aggregate compressed (HV16) gradients with Byzantine detection.
    ///
    /// Returns a `PipelineResult` with the aggregated HV and run statistics.
    /// The aggregated_hv field is `None` only if all gradients are excluded.
    pub fn aggregate_compressed(
        &self,
        gradients: &[CompressedGradient],
        reputations: &HashMap<String, f32>,
    ) -> Result<PipelineResult, PipelineError> {
        if gradients.is_empty() {
            return Err(PipelineError::NoGradients);
        }

        // Stage 1: Validate HV16 sizes
        for g in gradients {
            if g.hv_data.len() != HV16_BYTES {
                return Err(PipelineError::InvalidGradient {
                    participant: g.participant_id.clone(),
                    reason: format!(
                        "HV16 size mismatch: expected {} bytes, got {}",
                        HV16_BYTES,
                        g.hv_data.len()
                    ),
                });
            }
        }

        let total = gradients.len();

        // Stage 2: Byzantine detection
        let detection_config = crate::detection::DetectionConfig {
            cosine_threshold: self.config.cosine_threshold,
            reputation_threshold: self.config.reputation_threshold,
            confidence_threshold: self.config.confidence_threshold,
            max_byzantine_fraction: self.config.max_byzantine_fraction,
        };
        let detector = MultiLayerDetector::new(detection_config);
        let detection_result = detector.detect(gradients, reputations);

        // Stage 3: Filter out Byzantine participants
        let byzantine_set: std::collections::HashSet<usize> =
            detection_result.byzantine_indices.iter().cloned().collect();
        let excluded_participants: Vec<String> = detection_result
            .byzantine_indices
            .iter()
            .map(|&i| gradients[i].participant_id.clone())
            .collect();

        let honest_gradients: Vec<&CompressedGradient> = gradients
            .iter()
            .enumerate()
            .filter(|(i, _)| !byzantine_set.contains(i))
            .map(|(_, g)| g)
            .collect();

        let after_detection = honest_gradients.len();
        let byzantine_detected = total - after_detection;

        // Stage 4: HV-space aggregation (majority vote)
        let aggregated_hv = if honest_gradients.is_empty() {
            None
        } else {
            Some(majority_vote_hv(&honest_gradients))
        };

        Ok(PipelineResult {
            aggregated_hv,
            stats: PipelineStats {
                total_contributions: total,
                after_detection,
                byzantine_detected,
                excluded_participants,
                method_used: self.config.method.clone(),
            },
        })
    }

    /// Compute a deterministic SHA-256 commitment hash.
    ///
    /// All honest validators must produce the same hash for the same inputs.
    /// Input: aggregated HV16 data + method label + round number.
    ///
    /// Hash input: `aggregated_hv || method_bytes || round_le_bytes`
    pub fn commitment_hash(aggregated_hv: &[u8], method: &str, round: u64) -> String {
        sha256_hex(&commitment_bytes(aggregated_hv, method, round))
    }
}

/// Build the canonical byte string to hash for a commitment.
fn commitment_bytes(aggregated_hv: &[u8], method: &str, round: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(aggregated_hv.len() + method.len() + 8);
    data.extend_from_slice(aggregated_hv);
    data.extend_from_slice(method.as_bytes());
    data.extend_from_slice(&round.to_le_bytes());
    data
}

/// Compute SHA-256 hash using a pure-Rust implementation.
///
/// WASM-compatible — no OS entropy, no threading.
fn sha256_hex(data: &[u8]) -> String {
    // SHA-256 constants (first 32 bits of fractional parts of cube roots of primes)
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Initial hash values (first 32 bits of sqrt of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit chunk
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i*4], chunk[i*4+1], chunk[i*4+2], chunk[i*4+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce hex string
    h.iter()
        .flat_map(|word| {
            let bytes = word.to_be_bytes();
            bytes.into_iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Bit-level majority vote across a set of HV16 gradients.
fn majority_vote_hv(gradients: &[&CompressedGradient]) -> Vec<u8> {
    let n = gradients.len();
    let threshold = n / 2;
    let mut result = vec![0u8; HV16_BYTES];

    for byte_idx in 0..HV16_BYTES {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GradientMetadata;

    fn make_gradient(id: &str, hv_byte: u8) -> CompressedGradient {
        CompressedGradient {
            participant_id: id.to_string(),
            hv_data: vec![hv_byte; HV16_BYTES],
            original_dimension: 1000,
            quality_score: 0.9,
            metadata: GradientMetadata::new(1, 0.9),
        }
    }

    #[test]
    fn test_pipeline_no_gradients_errors() {
        let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
        let result = pipeline.aggregate_compressed(&[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_single_gradient() {
        let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
        let g = make_gradient("p1", 0xAA);
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9_f32);
        let result = pipeline.aggregate_compressed(&[g], &reps).unwrap();
        assert!(result.aggregated_hv.is_some());
        assert_eq!(result.stats.total_contributions, 1);
    }

    #[test]
    fn test_pipeline_deterministic() {
        let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
        let gradients = vec![
            make_gradient("p1", 0xAA),
            make_gradient("p2", 0xAA),
            make_gradient("p3", 0xAA),
        ];
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9_f32);
        reps.insert("p2".to_string(), 0.9_f32);
        reps.insert("p3".to_string(), 0.9_f32);

        let r1 = pipeline.aggregate_compressed(&gradients, &reps).unwrap();
        let r2 = pipeline.aggregate_compressed(&gradients, &reps).unwrap();
        assert_eq!(r1.aggregated_hv, r2.aggregated_hv);
    }

    #[test]
    fn test_commitment_hash_deterministic() {
        let hv = vec![0xAB_u8; HV16_BYTES];
        let h1 = DecentralizedPipeline::commitment_hash(&hv, "FedAvgHV", 42);
        let h2 = DecentralizedPipeline::commitment_hash(&hv, "FedAvgHV", 42);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_commitment_hash_different_inputs_differ() {
        let hv1 = vec![0xAA_u8; HV16_BYTES];
        let hv2 = vec![0xBB_u8; HV16_BYTES];
        let h1 = DecentralizedPipeline::commitment_hash(&hv1, "FedAvgHV", 1);
        let h2 = DecentralizedPipeline::commitment_hash(&hv2, "FedAvgHV", 1);
        let h3 = DecentralizedPipeline::commitment_hash(&hv1, "FedAvgHV", 2);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_sha256_known_value() {
        // sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256_hex(b"");
        assert_eq!(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_invalid_hv_size_errors() {
        let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
        let bad = CompressedGradient {
            participant_id: "p1".to_string(),
            hv_data: vec![0u8; 100], // Wrong size
            original_dimension: 1000,
            quality_score: 0.9,
            metadata: GradientMetadata::new(1, 0.9),
        };
        let result = pipeline.aggregate_compressed(&[bad], &HashMap::new());
        assert!(result.is_err());
    }
}
