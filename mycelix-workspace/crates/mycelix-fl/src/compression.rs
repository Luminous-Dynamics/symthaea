//! HyperFeel compression: J-L random projection from high-dimensional gradients
//! to compact HV16 binary hypervectors (2KB per gradient).
//!
//! Reduces bandwidth by ~2000x (10M params × 4B → 2KB) while preserving
//! enough geometric structure for Byzantine-robust aggregation in HV-space.
//!
//! # Status
//! Stub implementation — full J-L projection to be implemented.

use crate::types::{CompressedGradient, GradientMetadata, HV16_BYTES};

/// Error types for compression operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompressionError {
    #[error("Gradient is empty")]
    EmptyGradient,
    #[error("Participant ID is empty")]
    EmptyParticipantId,
    #[error("Projection matrix not initialized")]
    ProjectionNotInitialized,
}

/// HyperFeel compressor: projects f32 gradients to binary HV16 hypervectors.
///
/// Uses Johnson-Lindenstrauss random projection to reduce dimensionality.
/// The binary projection preserves cosine similarity with high probability.
pub struct HyperFeelCompressor {
    /// Random seed for reproducible projection matrices
    seed: u64,
}

impl HyperFeelCompressor {
    /// Create a new compressor with the given random seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Create a compressor with the default seed.
    pub fn default_seed() -> Self {
        Self::new(42)
    }

    /// Compress a gradient vector to a HV16 binary hypervector.
    ///
    /// Projects `gradient` (arbitrary dimension) to a `HV16_BYTES`-byte binary
    /// vector using a seeded random projection matrix.
    ///
    /// # Stub
    /// This implementation produces a deterministic output based on gradient
    /// statistics. A production implementation uses a full J-L random matrix.
    pub fn compress(
        &self,
        participant_id: &str,
        round: u32,
        gradient: &[f32],
        quality_score: f32,
    ) -> Result<CompressedGradient, CompressionError> {
        if gradient.is_empty() {
            return Err(CompressionError::EmptyGradient);
        }
        if participant_id.is_empty() {
            return Err(CompressionError::EmptyParticipantId);
        }

        let original_dimension = gradient.len();
        let hv_data = project_to_hv16(gradient, self.seed);

        Ok(CompressedGradient {
            participant_id: participant_id.to_string(),
            hv_data,
            original_dimension,
            quality_score,
            metadata: GradientMetadata::new(round, quality_score),
        })
    }
}

/// Project a gradient to a HV16 binary hypervector using a seeded random projection.
///
/// # Stub
/// This stub uses a simple hash-based projection for structural correctness.
/// A production implementation uses a stored random matrix for J-L guarantees.
fn project_to_hv16(gradient: &[f32], seed: u64) -> Vec<u8> {
    let mut hv = vec![0u8; HV16_BYTES];

    // Compute gradient statistics for projection seed
    let n = gradient.len() as f64;
    let mean: f64 = gradient.iter().map(|&x| x as f64).sum::<f64>() / n;
    let variance: f64 = gradient.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;

    // Generate HV16 bytes using a simple PRNG seeded with gradient stats + user seed
    let mut state = seed
        .wrapping_add(mean.to_bits())
        .wrapping_add(variance.to_bits())
        .wrapping_add(gradient.len() as u64);

    for byte in hv.iter_mut() {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state & 0xFF) as u8;
    }

    hv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_produces_hv16_bytes() {
        let compressor = HyperFeelCompressor::new(42);
        let gradient = vec![0.1_f32; 1000];
        let result = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        assert_eq!(result.hv_data.len(), HV16_BYTES);
        assert_eq!(result.participant_id, "p1");
        assert_eq!(result.original_dimension, 1000);
    }

    #[test]
    fn test_compress_empty_gradient_errors() {
        let compressor = HyperFeelCompressor::new(42);
        let result = compressor.compress("p1", 1, &[], 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_compress_empty_participant_errors() {
        let compressor = HyperFeelCompressor::new(42);
        let result = compressor.compress("", 1, &[0.1, 0.2], 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_same_gradient_same_seed_deterministic() {
        let compressor = HyperFeelCompressor::new(42);
        let gradient = vec![0.5_f32; 100];
        let r1 = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        let r2 = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        assert_eq!(r1.hv_data, r2.hv_data);
    }
}
