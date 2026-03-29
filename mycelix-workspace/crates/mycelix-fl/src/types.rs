// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core FL types — re-exported from mycelix-fl-core with HV16 extensions
//!
//! Canonical data structures for federated learning using f32 precision.
//! Extended with HV16 binary hypervector support for A2 aggregation.

use serde::{Deserialize, Serialize};

// ============================================================================
// Re-export canonical types from mycelix-fl-core (single source of truth)
// ============================================================================
pub use mycelix_fl_core::types::{
    AggregatedGradient, AggregationMethod, GradientMetadata, GradientUpdate, Participant,
    MAX_BYZANTINE_TOLERANCE,
};

// ============================================================================
// HV16 extensions (mycelix-fl only — not in fl-core)
// ============================================================================

/// HV16 dimension in bits
pub const HV16_DIMENSION: usize = 16_384;

/// HV16 size in bytes (2048 = 16384 / 8)
pub const HV16_BYTES: usize = 2048;

/// HV-space aggregation methods (A2 — operates on 2KB HV16 vectors)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HvAggregationMethod {
    /// FedAvg-HV: coordinate-wise majority vote
    FedAvgHv,
    /// Byzantine-Filtered-HV: cosine similarity threshold filtering
    ByzantineFilteredHv,
    /// Krum-HV: adapted for Hamming distance
    KrumHv,
    /// Multi-Krum-HV: top-m selection + bundling
    MultiKrumHv,
    /// Median-HV: coordinate-wise median (bit-level majority)
    MedianHv,
    /// Trimmed-Mean-HV: outlier removal + majority
    TrimmedMeanHv,
}

/// Compressed gradient in HV16 binary format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedGradient {
    pub participant_id: String,
    pub hv_data: Vec<u8>,
    pub original_dimension: usize,
    pub quality_score: f32,
    pub metadata: GradientMetadata,
}

impl CompressedGradient {
    pub fn is_valid(&self) -> bool {
        self.hv_data.len() == HV16_BYTES
            && self.metadata.is_valid()
            && self.quality_score.is_finite()
    }

    /// Compute Hamming distance to another compressed gradient
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.hv_data
            .iter()
            .zip(other.hv_data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Compute cosine similarity approximation via normalized Hamming
    /// For binary HVs: cosine ≈ 1 - 2 * hamming_distance / dimension
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other) as f32;
        1.0 - 2.0 * hamming / HV16_DIMENSION as f32
    }
}

// ============================================================================
// Aggregation error types (mycelix-fl specific — includes pipeline variants)
// ============================================================================

/// Aggregation error types
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum AggregationError {
    #[error("No updates provided")]
    NoUpdates,
    #[error("Empty gradients for participant {0}")]
    EmptyGradients(String),
    #[error("Gradient size mismatch: expected {expected}, got {got} from {participant}")]
    GradientSizeMismatch {
        expected: usize,
        got: usize,
        participant: String,
    },
    #[error("Invalid trim percentage: {0}")]
    InvalidTrimPercentage(f32),
    #[error("Not enough updates for Krum: need at least 3, got {0}")]
    NotEnoughForKrum(usize),
    #[error("Too many Byzantine participants detected")]
    TooManyByzantine,
    #[error("No valid gradients after filtering")]
    NoValidGradients,
    #[error("Pipeline error: {0}")]
    PipelineError(String),
}

/// Convert fl-core AggregationError to mycelix-fl AggregationError
impl From<mycelix_fl_core::aggregation::AggregationError> for AggregationError {
    fn from(e: mycelix_fl_core::aggregation::AggregationError) -> Self {
        use mycelix_fl_core::aggregation::AggregationError as CoreErr;
        match e {
            CoreErr::NoUpdates => AggregationError::NoUpdates,
            CoreErr::EmptyGradients(p) => AggregationError::EmptyGradients(p),
            CoreErr::GradientSizeMismatch {
                participant_id,
                expected,
                actual,
            } => AggregationError::GradientSizeMismatch {
                expected,
                got: actual,
                participant: participant_id,
            },
            CoreErr::InvalidTrimPercentage(p) => AggregationError::InvalidTrimPercentage(p),
            CoreErr::NotEnoughForKrum(n) => AggregationError::NotEnoughForKrum(n),
            CoreErr::InvalidKrumSelect(n) => AggregationError::NotEnoughForKrum(n),
            CoreErr::InvalidBatchSize(_) => {
                AggregationError::PipelineError("Invalid batch size".to_string())
            }
            CoreErr::InvalidLoss(p) => {
                AggregationError::PipelineError(format!("Invalid loss for {p}"))
            }
            CoreErr::NoTrustedParticipants => AggregationError::NoValidGradients,
            CoreErr::TooManyByzantine => AggregationError::TooManyByzantine,
        }
    }
}

// ============================================================================
// Detection types (mycelix-fl specific — different from fl-core)
// ============================================================================

/// Detection result for a single participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSignal {
    pub participant_id: String,
    pub is_byzantine: bool,
    pub confidence: f32,
    pub signals: Vec<(String, f32)>,
}

/// Result of multi-signal Byzantine detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub byzantine_indices: Vec<usize>,
    pub confidence_scores: Vec<f32>,
    pub signal_breakdown: Vec<DetectionSignal>,
    pub stats: DetectionStats,
}

/// Detection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectionStats {
    pub total_participants: usize,
    pub flagged_count: usize,
    pub detection_layers_used: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_update() {
        let update = GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5);
        assert_eq!(update.dimension(), 3);
        assert!(update.is_valid());
        assert!(update.l2_norm() > 0.0);
    }

    #[test]
    fn test_compressed_gradient_similarity() {
        let mut a = CompressedGradient {
            participant_id: "a".to_string(),
            hv_data: vec![0xFF; HV16_BYTES],
            original_dimension: 10000,
            quality_score: 0.9,
            metadata: GradientMetadata::new(100, 0.5),
        };
        let b = a.clone();
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 0.001);

        // Different gradient
        a.hv_data = vec![0x00; HV16_BYTES];
        assert!((a.cosine_similarity(&b) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_hv16_constants() {
        assert_eq!(HV16_BYTES, HV16_DIMENSION / 8);
        assert_eq!(HV16_BYTES, 2048);
    }

    #[test]
    fn test_core_error_conversion() {
        use mycelix_fl_core::aggregation::AggregationError as CoreErr;
        let core_err = CoreErr::GradientSizeMismatch {
            participant_id: "p1".to_string(),
            expected: 10,
            actual: 5,
        };
        let converted: AggregationError = core_err.into();
        match converted {
            AggregationError::GradientSizeMismatch {
                expected,
                got,
                participant,
            } => {
                assert_eq!(expected, 10);
                assert_eq!(got, 5);
                assert_eq!(participant, "p1");
            }
            _ => panic!("Wrong variant"),
        }
    }
}
