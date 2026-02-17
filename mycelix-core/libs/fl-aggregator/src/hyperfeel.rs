//! HyperFeel integration layer for hypervector-based gradient encoding.
//!
//! This module provides the bridge between traditional dense gradients and
//! hyperdimensional computing (HDC) representations, enabling efficient
//! compression and aggregation of federated learning updates.
//!
//! The canonical HyperFeel type definitions (e.g. `HyperGradient`) live in
//! the `mycelix-sdk` crate (`mycelix-workspace/sdk/src/hyperfeel`). This
//! module focuses on the FL engine side (encoding/decoding against
//! `DenseGradient` / `Hypervector`) and should remain semantically aligned
//! with the SDK representation.
//!
//! # Algorithm Overview
//!
//! ## Encoding
//! 1. **Quantize**: Map f32 gradient values to i8 range: `(value * 127.0).round().clamp(-127, 127)`
//! 2. **Fold**: Project to hypervector dimension via modular arithmetic: `idx % HYPERVECTOR_DIM`
//! 3. **Accumulate**: Sum folded values with saturation arithmetic
//!
//! ## Decoding
//! 1. **Unfold**: Distribute hypervector components back to original dimension
//! 2. **Dequantize**: Map i8 back to f32: `component as f32 / 127.0`
//!
//! # Example
//!
//! ```rust
//! use fl_aggregator::hyperfeel::{encode_gradient, decode_gradient, EncodingConfig};
//! use fl_aggregator::payload::{DenseGradient, Payload};
//!
//! let gradient = DenseGradient::from_vec(vec![0.5, -0.3, 0.8, 0.1]);
//! let config = EncodingConfig::default();
//!
//! let encoded = encode_gradient(&gradient, &config);
//! let decoded = decode_gradient(&encoded, gradient.dimension());
//!
//! // The roundtrip preserves the general structure, with some quantization loss
//! ```

use crate::payload::{DenseGradient, Hypervector, HypervectorMetadata, Payload, HYPERVECTOR_DIM};
use serde::{Deserialize, Serialize};

/// Configuration for gradient encoding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Enable causal encoding (preserves temporal ordering).
    pub use_causal: bool,

    /// Enable temporal encoding (time-weighted components).
    pub use_temporal: bool,

    /// Random projection seed for reproducibility.
    /// When set, enables deterministic random projections.
    pub projection_seed: Option<i64>,
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
    /// Create a new encoding config with causal encoding enabled.
    pub fn with_causal(mut self) -> Self {
        self.use_causal = true;
        self
    }

    /// Create a new encoding config with temporal encoding enabled.
    pub fn with_temporal(mut self) -> Self {
        self.use_temporal = true;
        self
    }

    /// Create a new encoding config with a specific projection seed.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.projection_seed = Some(seed);
        self
    }
}

/// Encode a dense gradient into a hypervector.
///
/// # Algorithm
/// 1. Quantize each f32 value to i8: `(value * 127.0).round().clamp(-127, 127)`
/// 2. Fold into hypervector dimension: `idx % HYPERVECTOR_DIM`
/// 3. Accumulate with saturation arithmetic
///
/// # Arguments
/// * `gradient` - The dense gradient to encode
/// * `config` - Encoding configuration
///
/// # Returns
/// A hypervector with metadata populated for decoding
pub fn encode_gradient(gradient: &DenseGradient, config: &EncodingConfig) -> Hypervector {
    let dim = HYPERVECTOR_DIM;
    let mut components = vec![0i8; dim];

    // Track accumulation in i16 to avoid overflow during folding
    let mut accumulators: Vec<i16> = vec![0; dim];

    // Optionally generate projection indices from seed
    let projection_indices: Option<Vec<usize>> = config.projection_seed.map(|seed| {
        generate_projection_indices(gradient.dimension(), dim, seed)
    });

    for (idx, &value) in gradient.values.iter().enumerate() {
        // Quantize to i8 range
        let quantized = (value * 127.0).round().clamp(-127.0, 127.0) as i8;

        // Determine target position (fold or project)
        let pos = if let Some(ref proj) = projection_indices {
            proj[idx % proj.len()]
        } else {
            idx % dim
        };

        // Apply causal encoding: phase-shift based on position
        let final_pos = if config.use_causal {
            (pos + (idx / dim)) % dim
        } else {
            pos
        };

        // Apply temporal weighting if enabled
        let weighted = if config.use_temporal {
            // Later indices get slightly higher weight (recency bias)
            let weight = 1.0 + (idx as f32 / gradient.dimension() as f32) * 0.1;
            ((quantized as f32) * weight).round().clamp(-127.0, 127.0) as i8
        } else {
            quantized
        };

        // Accumulate with overflow tracking
        accumulators[final_pos] += weighted as i16;
    }

    // Saturate accumulators to i8 range
    for (i, &acc) in accumulators.iter().enumerate() {
        components[i] = acc.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
    }

    // Calculate compression ratio
    let compression_ratio = calculate_compression_ratio(gradient, dim);

    // Build metadata
    let metadata = HypervectorMetadata {
        encoder_version: "hyperfeel-v1".to_string(),
        original_dimension: gradient.dimension(),
        compression_ratio,
        projection_seed: config.projection_seed,
        use_causal: config.use_causal,
        use_temporal: config.use_temporal,
    };

    Hypervector::with_metadata(components, metadata)
}

/// Decode a hypervector back to an approximate dense gradient.
///
/// # Algorithm
/// 1. Unfold components back to original dimension
/// 2. Dequantize: `component as f32 / 127.0`
///
/// # Arguments
/// * `hv` - The hypervector to decode
/// * `original_dim` - The original gradient dimension
///
/// # Returns
/// An approximate reconstruction of the original gradient
///
/// # Note
/// This is a lossy operation. The decoded gradient will differ from the
/// original due to:
/// - Quantization error (~1/127 resolution)
/// - Folding collisions (when original_dim > HYPERVECTOR_DIM)
/// - Saturation clipping
pub fn decode_gradient(hv: &Hypervector, original_dim: usize) -> DenseGradient {
    let dim = hv.components.len();
    let mut values = vec![0.0f32; original_dim];

    // Optionally reconstruct projection indices from metadata
    let projection_indices: Option<Vec<usize>> = hv.metadata.projection_seed.map(|seed| {
        generate_projection_indices(original_dim, dim, seed)
    });

    // Calculate fold factor for distribution
    let fold_factor = (original_dim + dim - 1) / dim; // ceil division

    for idx in 0..original_dim {
        // Determine source position
        let pos = if let Some(ref proj) = projection_indices {
            proj[idx % proj.len()]
        } else {
            idx % dim
        };

        // Reverse causal encoding if used
        let final_pos = if hv.metadata.use_causal {
            (pos + (idx / dim)) % dim
        } else {
            pos
        };

        // Dequantize component
        let component = hv.components[final_pos];
        let dequantized = component as f32 / 127.0;

        // Distribute evenly among folded positions
        // This averages out the collision artifacts
        let distributed = dequantized / fold_factor as f32;

        // Reverse temporal weighting if used
        let final_value = if hv.metadata.use_temporal {
            let weight = 1.0 + (idx as f32 / original_dim as f32) * 0.1;
            distributed / weight
        } else {
            distributed
        };

        values[idx] = final_value;
    }

    DenseGradient::from_vec(values)
}

/// Generate deterministic projection indices from a seed.
///
/// Uses a simple linear congruential generator for reproducibility.
fn generate_projection_indices(input_dim: usize, output_dim: usize, seed: i64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(input_dim);
    let mut state = seed as u64;

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

/// Calculate the compression ratio achieved by encoding.
///
/// # Arguments
/// * `original` - The original dense gradient
/// * `encoded_dim` - The encoded hypervector dimension
///
/// # Returns
/// The compression ratio (original_size / encoded_size)
pub fn calculate_compression_ratio(original: &DenseGradient, encoded_dim: usize) -> f32 {
    let original_size = original.size_bytes();
    let encoded_size = encoded_dim * std::mem::size_of::<i8>();

    if encoded_size == 0 {
        return 0.0;
    }

    original_size as f32 / encoded_size as f32
}

/// Estimate the quality of encoding by comparing original and decoded gradients.
///
/// # Arguments
/// * `original` - The original dense gradient
/// * `decoded` - The decoded gradient after roundtrip
///
/// # Returns
/// A quality score in [0, 1], where 1.0 indicates perfect reconstruction
///
/// # Metrics
/// Uses normalized mean squared error (NMSE) converted to a quality score:
/// `quality = 1.0 / (1.0 + NMSE)`
pub fn estimate_encoding_quality(original: &DenseGradient, decoded: &DenseGradient) -> f32 {
    if original.dimension() != decoded.dimension() {
        return 0.0;
    }

    if original.dimension() == 0 {
        return 1.0;
    }

    let mut mse: f64 = 0.0;
    let mut variance: f64 = 0.0;

    // Calculate mean for variance computation
    let mean: f64 = original.values.iter().map(|&x| x as f64).sum::<f64>()
        / original.dimension() as f64;

    for (a, b) in original.values.iter().zip(decoded.values.iter()) {
        let error = (*a as f64) - (*b as f64);
        mse += error * error;
        let deviation = (*a as f64) - mean;
        variance += deviation * deviation;
    }

    mse /= original.dimension() as f64;
    variance /= original.dimension() as f64;

    // Normalized MSE (avoid division by zero)
    let nmse = if variance.abs() < 1e-10 {
        mse // If no variance, just use raw MSE
    } else {
        mse / variance
    };

    // Convert to quality score in [0, 1]
    (1.0 / (1.0 + nmse)) as f32
}

/// Calculate cosine similarity between original and decoded gradients.
///
/// This measures directional preservation, which is often more important
/// than absolute value reconstruction in gradient descent.
///
/// # Returns
/// Cosine similarity in [-1, 1], where 1.0 indicates perfect alignment
pub fn calculate_cosine_similarity(original: &DenseGradient, decoded: &DenseGradient) -> f32 {
    if original.dimension() != decoded.dimension() || original.dimension() == 0 {
        return 0.0;
    }

    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for (&a, &b) in original.values.iter().zip(decoded.values.iter()) {
        dot += (a as f64) * (b as f64);
        norm_a += (a as f64) * (a as f64);
        norm_b += (b as f64) * (b as f64);
    }

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

/// Encoding quality report with multiple metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingQualityReport {
    /// Compression ratio (original_size / encoded_size).
    pub compression_ratio: f32,

    /// Quality score based on NMSE in [0, 1].
    pub quality_score: f32,

    /// Cosine similarity between original and decoded in [-1, 1].
    pub cosine_similarity: f32,

    /// Mean absolute error.
    pub mean_absolute_error: f32,

    /// Peak signal-to-noise ratio (dB).
    pub psnr_db: f32,
}

impl EncodingQualityReport {
    /// Generate a comprehensive quality report.
    pub fn generate(
        original: &DenseGradient,
        encoded: &Hypervector,
        decoded: &DenseGradient,
    ) -> Self {
        let compression_ratio = calculate_compression_ratio(original, encoded.dimension());
        let quality_score = estimate_encoding_quality(original, decoded);
        let cosine_similarity = calculate_cosine_similarity(original, decoded);

        // Calculate MAE
        let mae: f32 = if original.dimension() > 0 {
            original
                .values
                .iter()
                .zip(decoded.values.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f32>()
                / original.dimension() as f32
        } else {
            0.0
        };

        // Calculate PSNR
        let mse: f32 = if original.dimension() > 0 {
            original
                .values
                .iter()
                .zip(decoded.values.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                / original.dimension() as f32
        } else {
            0.0
        };

        // For normalized gradients, max value is typically around 1.0
        let max_val = original
            .values
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);

        let psnr_db = if mse > 0.0 {
            10.0 * (max_val * max_val / mse).log10()
        } else {
            f32::INFINITY
        };

        Self {
            compression_ratio,
            quality_score,
            cosine_similarity,
            mean_absolute_error: mae,
            psnr_db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create a test gradient
        let gradient = DenseGradient::from_vec(vec![
            0.5, -0.3, 0.8, -0.1, 0.0, 0.9, -0.5, 0.2,
        ]);
        let config = EncodingConfig::default();

        // Encode
        let encoded = encode_gradient(&gradient, &config);

        // Verify metadata
        assert_eq!(encoded.metadata.encoder_version, "hyperfeel-v1");
        assert_eq!(encoded.metadata.original_dimension, 8);
        assert!(!encoded.metadata.use_causal);
        assert!(!encoded.metadata.use_temporal);

        // Decode
        let decoded = decode_gradient(&encoded, gradient.dimension());

        // Verify roundtrip preserves structure
        assert_eq!(decoded.dimension(), gradient.dimension());

        // Check that values are in reasonable range
        for &val in decoded.values.iter() {
            assert!(val.is_finite());
            assert!(val.abs() <= 1.1); // Allow small overshoot from reconstruction
        }

        // Verify cosine similarity is high (direction preserved)
        let similarity = calculate_cosine_similarity(&gradient, &decoded);
        assert!(
            similarity > 0.5,
            "Cosine similarity too low: {}",
            similarity
        );
    }

    #[test]
    fn test_encode_decode_with_causal() {
        let gradient = DenseGradient::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let config = EncodingConfig::default().with_causal();

        let encoded = encode_gradient(&gradient, &config);
        assert!(encoded.metadata.use_causal);

        let decoded = decode_gradient(&encoded, gradient.dimension());
        assert_eq!(decoded.dimension(), gradient.dimension());
    }

    #[test]
    fn test_encode_decode_with_temporal() {
        let gradient = DenseGradient::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let config = EncodingConfig::default().with_temporal();

        let encoded = encode_gradient(&gradient, &config);
        assert!(encoded.metadata.use_temporal);

        let decoded = decode_gradient(&encoded, gradient.dimension());
        assert_eq!(decoded.dimension(), gradient.dimension());
    }

    #[test]
    fn test_encode_decode_with_seed() {
        let gradient = DenseGradient::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let config = EncodingConfig::default().with_seed(42);

        let encoded1 = encode_gradient(&gradient, &config);
        let encoded2 = encode_gradient(&gradient, &config);

        // Same seed should produce identical results
        assert_eq!(encoded1.components, encoded2.components);
        assert_eq!(encoded1.metadata.projection_seed, Some(42));
    }

    #[test]
    fn test_compression_ratio() {
        // Large gradient to see real compression
        let large_gradient: Vec<f32> = (0..100_000).map(|i| (i as f32) / 100_000.0).collect();
        let gradient = DenseGradient::from_vec(large_gradient);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);
        let ratio = calculate_compression_ratio(&gradient, encoded.dimension());

        // 100,000 f32 values (400KB) -> 16,384 i8 values (16KB)
        // Expected ratio ~= 400,000 / 16,384 ~= 24.4
        assert!(ratio > 20.0, "Compression ratio too low: {}", ratio);
        assert!(ratio < 30.0, "Compression ratio too high: {}", ratio);

        // Metadata should also have the ratio
        assert!((encoded.metadata.compression_ratio - ratio).abs() < 0.01);
    }

    #[test]
    fn test_encoding_preserves_similarity() {
        // Create two similar gradients
        let gradient1 = DenseGradient::from_vec(vec![0.5, 0.3, 0.8, 0.1]);
        let gradient2 = DenseGradient::from_vec(vec![0.52, 0.31, 0.79, 0.11]); // slightly perturbed

        let config = EncodingConfig::default();

        let encoded1 = encode_gradient(&gradient1, &config);
        let encoded2 = encode_gradient(&gradient2, &config);

        // Calculate cosine similarity between encoded hypervectors
        let hv_similarity = encoded1.cosine_similarity(&encoded2);

        // Calculate cosine similarity between original gradients
        let original_similarity = calculate_cosine_similarity(&gradient1, &gradient2);

        // The HDC encoding should preserve the similarity relationship
        // (i.e., similar gradients should encode to similar hypervectors)
        assert!(
            hv_similarity > 0.9,
            "Encoded similarity too low: {}",
            hv_similarity
        );
        assert!(
            (hv_similarity - original_similarity).abs() < 0.2,
            "Similarity not preserved: hv={}, orig={}",
            hv_similarity,
            original_similarity
        );
    }

    #[test]
    fn test_encoding_quality_metrics() {
        let gradient = DenseGradient::from_vec(vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.9, -0.5, 0.2]);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);
        let decoded = decode_gradient(&encoded, gradient.dimension());

        let quality = estimate_encoding_quality(&gradient, &decoded);

        // Quality should be reasonable for this small gradient
        assert!(quality > 0.3, "Quality too low: {}", quality);
        assert!(quality <= 1.0, "Quality out of range: {}", quality);
    }

    #[test]
    fn test_quality_report() {
        let gradient = DenseGradient::from_vec(vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.9, -0.5, 0.2]);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);
        let decoded = decode_gradient(&encoded, gradient.dimension());

        let report = EncodingQualityReport::generate(&gradient, &encoded, &decoded);

        // Verify all metrics are computed
        assert!(report.compression_ratio > 0.0);
        assert!(report.quality_score >= 0.0 && report.quality_score <= 1.0);
        assert!(report.cosine_similarity >= -1.0 && report.cosine_similarity <= 1.0);
        assert!(report.mean_absolute_error >= 0.0);
        // PSNR can be infinity for perfect reconstruction, so just check it's positive
        assert!(report.psnr_db > 0.0 || report.psnr_db.is_infinite());
    }

    #[test]
    fn test_empty_gradient() {
        let gradient = DenseGradient::from_vec(vec![]);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);
        assert_eq!(encoded.metadata.original_dimension, 0);

        let decoded = decode_gradient(&encoded, 0);
        assert_eq!(decoded.dimension(), 0);
    }

    #[test]
    fn test_large_gradient_folding() {
        // Gradient larger than HYPERVECTOR_DIM should fold correctly
        let large_gradient: Vec<f32> = (0..HYPERVECTOR_DIM * 3)
            .map(|i| ((i % 10) as f32 - 5.0) / 10.0)
            .collect();
        let gradient = DenseGradient::from_vec(large_gradient);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);

        // Should still be HYPERVECTOR_DIM size
        assert_eq!(encoded.dimension(), HYPERVECTOR_DIM);

        // All components should be valid i8
        for &c in &encoded.components {
            assert!(c >= i8::MIN && c <= i8::MAX);
        }
    }

    #[test]
    fn test_extreme_values() {
        // Test with values at the quantization boundaries
        let gradient = DenseGradient::from_vec(vec![
            1.0,   // max positive
            -1.0,  // max negative
            0.0,   // zero
            10.0,  // above range (should clamp)
            -10.0, // below range (should clamp)
        ]);
        let config = EncodingConfig::default();

        let encoded = encode_gradient(&gradient, &config);

        // All components should be valid
        assert!(encoded.is_valid());

        // Check clamping worked (i8 is always in -128..127, but we clamp to -127..127)
        for &c in &encoded.components {
            assert!((c as i16) >= -127 && (c as i16) <= 127);
        }
    }

    #[test]
    fn test_projection_indices_deterministic() {
        let indices1 = generate_projection_indices(100, 16, 12345);
        let indices2 = generate_projection_indices(100, 16, 12345);
        let indices3 = generate_projection_indices(100, 16, 54321);

        // Same seed = same indices
        assert_eq!(indices1, indices2);

        // Different seed = different indices
        assert_ne!(indices1, indices3);

        // All indices should be in valid range
        for &idx in &indices1 {
            assert!(idx < 16);
        }
    }
}
