// Ported from fl-aggregator/src/hyperfeel.rs — adapted for mycelix-fl types
//! HyperFeel compression: J-L random projection from dense gradients
//! to compact HV16 binary hypervectors (2KB per gradient).
//!
//! Reduces bandwidth by ~2000x (10M params x 4B → 2KB) while preserving
//! enough geometric structure for Byzantine-robust aggregation in HV-space.
//!
//! # Encoding Algorithm
//! 1. **Quantize**: Map f32 → i8 via `(value * 127.0).round().clamp(-127, 127)`
//! 2. **Fold**: Project to HV dimension via seeded index map or modular arithmetic
//! 3. **Accumulate**: Sum folded values in i16 with saturation to i8
//!
//! # Decoding Algorithm
//! 1. **Unfold**: Distribute HV components back to original dimension
//! 2. **Dequantize**: Map i8 → f32 via `component / 127.0 / fold_factor`

use crate::types::{CompressedGradient, GradientMetadata, HV16_BYTES};

/// Error types for compression operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompressionError {
    #[error("Gradient is empty")]
    EmptyGradient,
    #[error("Participant ID is empty")]
    EmptyParticipantId,
    #[error("HV data has wrong size: expected {expected}, got {actual}")]
    InvalidHvSize { expected: usize, actual: usize },
}

/// Configuration for HyperFeel encoding.
#[derive(Clone, Debug)]
pub struct EncodingConfig {
    /// Enable causal encoding (phase-shift by position).
    pub use_causal: bool,
    /// Enable temporal weighting (recency bias).
    pub use_temporal: bool,
    /// Optional seed for deterministic random projections.
    pub projection_seed: Option<u64>,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            use_causal: false,
            use_temporal: false,
            projection_seed: None,
        }
    }
}

impl EncodingConfig {
    pub fn with_causal(mut self) -> Self {
        self.use_causal = true;
        self
    }

    pub fn with_temporal(mut self) -> Self {
        self.use_temporal = true;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.projection_seed = Some(seed);
        self
    }
}

/// HyperFeel compressor: projects f32 gradients to HV16 binary hypervectors.
pub struct HyperFeelCompressor {
    config: EncodingConfig,
}

impl HyperFeelCompressor {
    /// Create a new compressor with the given config.
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }

    /// Create a compressor with default config and a seed.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(EncodingConfig::default().with_seed(seed))
    }

    /// Create a compressor with the default seed (42).
    pub fn default_seed() -> Self {
        Self::with_seed(42)
    }

    /// Compress a gradient to a `CompressedGradient` (HV16).
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
        let hv_data = self.encode(gradient);

        Ok(CompressedGradient {
            participant_id: participant_id.to_string(),
            hv_data,
            original_dimension,
            quality_score,
            metadata: GradientMetadata::new(round, quality_score),
        })
    }

    /// Decode a HV16 back to an approximate dense gradient.
    pub fn decompress(
        &self,
        hv_data: &[u8],
        original_dim: usize,
    ) -> Result<Vec<f32>, CompressionError> {
        if hv_data.len() != HV16_BYTES {
            return Err(CompressionError::InvalidHvSize {
                expected: HV16_BYTES,
                actual: hv_data.len(),
            });
        }

        let dim = HV16_BYTES;
        let fold_factor = (original_dim + dim - 1) / dim; // ceil division

        // Interpret HV bytes as i8 components
        let components: Vec<i8> = hv_data.iter().map(|&b| b as i8).collect();

        // Optionally reconstruct projection indices
        let proj = self
            .config
            .projection_seed
            .map(|seed| generate_projection_indices(original_dim, dim, seed));

        let mut values = vec![0.0_f32; original_dim];

        for idx in 0..original_dim {
            let pos = if let Some(ref p) = proj {
                p[idx % p.len()]
            } else {
                idx % dim
            };

            let final_pos = if self.config.use_causal {
                (pos + (idx / dim)) % dim
            } else {
                pos
            };

            let component = components[final_pos];
            let dequantized = component as f32 / 127.0;
            let distributed = dequantized / fold_factor as f32;

            let final_value = if self.config.use_temporal {
                let weight = 1.0 + (idx as f32 / original_dim as f32) * 0.1;
                distributed / weight
            } else {
                distributed
            };

            values[idx] = final_value;
        }

        Ok(values)
    }

    /// Encode gradient to HV16 bytes (internal).
    ///
    /// Uses `HV16_BYTES` (2048) dense i8 accumulators, yielding exactly 2048 bytes.
    fn encode(&self, gradient: &[f32]) -> Vec<u8> {
        let dim = HV16_BYTES;
        let mut accumulators: Vec<i16> = vec![0; dim];

        // Optionally generate projection indices
        let proj = self
            .config
            .projection_seed
            .map(|seed| generate_projection_indices(gradient.len(), dim, seed));

        for (idx, &value) in gradient.iter().enumerate() {
            // Quantize to i8 range
            let quantized = (value * 127.0).round().clamp(-127.0, 127.0) as i8;

            // Determine target position
            let pos = if let Some(ref p) = proj {
                p[idx % p.len()]
            } else {
                idx % dim
            };

            // Causal phase-shift
            let final_pos = if self.config.use_causal {
                (pos + (idx / dim)) % dim
            } else {
                pos
            };

            // Temporal weighting
            let weighted = if self.config.use_temporal {
                let weight = 1.0 + (idx as f32 / gradient.len() as f32) * 0.1;
                ((quantized as f32) * weight).round().clamp(-127.0, 127.0) as i8
            } else {
                quantized
            };

            accumulators[final_pos] += weighted as i16;
        }

        // Saturate to i8 and convert to u8
        accumulators
            .iter()
            .map(|&acc| acc.clamp(i8::MIN as i16, i8::MAX as i16) as i8 as u8)
            .collect()
    }
}

/// Generate deterministic projection indices using an LCG.
fn generate_projection_indices(input_dim: usize, output_dim: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(input_dim);
    let mut state = seed;

    // LCG parameters (same as glibc)
    const A: u64 = 1103515245;
    const C: u64 = 12345;
    const M: u64 = 1 << 31;

    for _ in 0..input_dim {
        state = (A.wrapping_mul(state).wrapping_add(C)) % M;
        indices.push((state as usize) % output_dim);
    }

    indices
}

/// Calculate cosine similarity between original and decoded gradients.
pub fn encoding_cosine_similarity(original: &[f32], decoded: &[f32]) -> f32 {
    if original.len() != decoded.len() || original.is_empty() {
        return 0.0;
    }

    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;

    for (&a, &b) in original.iter().zip(decoded.iter()) {
        dot += (a as f64) * (b as f64);
        na += (a as f64) * (a as f64);
        nb += (b as f64) * (b as f64);
    }

    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }

    (dot / (na.sqrt() * nb.sqrt())) as f32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_produces_hv16_bytes() {
        let compressor = HyperFeelCompressor::default_seed();
        let gradient = vec![0.1_f32; 1000];
        let result = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        assert_eq!(result.hv_data.len(), HV16_BYTES);
        assert_eq!(result.participant_id, "p1");
        assert_eq!(result.original_dimension, 1000);
    }

    #[test]
    fn test_compress_empty_gradient_errors() {
        let compressor = HyperFeelCompressor::default_seed();
        assert!(compressor.compress("p1", 1, &[], 0.9).is_err());
    }

    #[test]
    fn test_compress_empty_participant_errors() {
        let compressor = HyperFeelCompressor::default_seed();
        assert!(compressor.compress("", 1, &[0.1, 0.2], 0.9).is_err());
    }

    #[test]
    fn test_same_gradient_same_seed_deterministic() {
        let compressor = HyperFeelCompressor::with_seed(42);
        let gradient = vec![0.5_f32; 100];
        let r1 = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        let r2 = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        assert_eq!(r1.hv_data, r2.hv_data);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let compressor = HyperFeelCompressor::new(EncodingConfig::default());
        let gradient = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.9, -0.5, 0.2];

        let compressed = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        let decoded = compressor
            .decompress(&compressed.hv_data, gradient.len())
            .unwrap();

        assert_eq!(decoded.len(), gradient.len());
        for &v in &decoded {
            assert!(v.is_finite());
        }

        let sim = encoding_cosine_similarity(&gradient, &decoded);
        assert!(
            sim > 0.5,
            "Cosine similarity too low: {}",
            sim
        );
    }

    #[test]
    fn test_encode_decode_with_causal() {
        let compressor = HyperFeelCompressor::new(EncodingConfig::default().with_causal());
        let gradient = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let compressed = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        let decoded = compressor
            .decompress(&compressed.hv_data, gradient.len())
            .unwrap();

        assert_eq!(decoded.len(), gradient.len());
    }

    #[test]
    fn test_encode_decode_with_temporal() {
        let compressor = HyperFeelCompressor::new(EncodingConfig::default().with_temporal());
        let gradient = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let compressed = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        let decoded = compressor
            .decompress(&compressed.hv_data, gradient.len())
            .unwrap();

        assert_eq!(decoded.len(), gradient.len());
    }

    #[test]
    fn test_large_gradient_compression_ratio() {
        let compressor = HyperFeelCompressor::new(EncodingConfig::default());
        let gradient: Vec<f32> = (0..100_000).map(|i| (i as f32) / 100_000.0).collect();

        let compressed = compressor.compress("p1", 1, &gradient, 0.9).unwrap();
        assert_eq!(compressed.hv_data.len(), HV16_BYTES);

        // 100K f32 = 400KB → 2KB HV16 ≈ 200x
        let ratio = (gradient.len() * 4) as f32 / compressed.hv_data.len() as f32;
        assert!(ratio > 100.0, "Expected >100x ratio, got {}", ratio);
    }

    #[test]
    fn test_different_seeds_produce_different_encodings() {
        let gradient = vec![0.5_f32; 100];

        let c1 = HyperFeelCompressor::with_seed(42);
        let c2 = HyperFeelCompressor::with_seed(99);

        let r1 = c1.compress("p1", 1, &gradient, 0.9).unwrap();
        let r2 = c2.compress("p1", 1, &gradient, 0.9).unwrap();

        assert_ne!(r1.hv_data, r2.hv_data);
    }

    #[test]
    fn test_decompress_wrong_size_errors() {
        let compressor = HyperFeelCompressor::default_seed();
        let bad_hv = vec![0u8; 100]; // Not HV16_BYTES
        assert!(compressor.decompress(&bad_hv, 50).is_err());
    }

    #[test]
    fn test_projection_indices_deterministic() {
        let i1 = generate_projection_indices(100, 16, 12345);
        let i2 = generate_projection_indices(100, 16, 12345);
        let i3 = generate_projection_indices(100, 16, 54321);

        assert_eq!(i1, i2);
        assert_ne!(i1, i3);

        for &idx in &i1 {
            assert!(idx < 16);
        }
    }
}
