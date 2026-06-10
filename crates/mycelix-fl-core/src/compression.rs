// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/compression.rs
//! Top-k sparsification and 8-bit quantization for gradient compression.
//!
//! Compresses gradient vectors by keeping only the k largest-magnitude
//! components (Top-k sparsification) and quantizing them to i8. Supports
//! error feedback to recover lost information over successive rounds.
//!
//! Achieves 10x–40x compression depending on configuration.

use serde::{Deserialize, Serialize};

use crate::aggregation::AggregationError;
use crate::types::GradientUpdate;

/// Configuration for gradient compression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (e.g., 10.0 = keep 1/10 of values).
    pub compression_ratio: f32,
    /// Bits for quantization (only 8-bit i8 is currently supported).
    pub quantization_bits: u8,
    /// Accumulate residuals across rounds for error feedback.
    pub error_feedback: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 10.0,
            quantization_bits: 8,
            error_feedback: true,
        }
    }
}

impl CompressionConfig {
    /// High compression preset (40x).
    pub fn high_compression() -> Self {
        Self {
            compression_ratio: 40.0,
            quantization_bits: 8,
            error_feedback: true,
        }
    }

    /// Low latency preset (2x, no error feedback).
    pub fn low_latency() -> Self {
        Self {
            compression_ratio: 2.0,
            quantization_bits: 8,
            error_feedback: false,
        }
    }
}

/// A compressed gradient: sparse Top-k indices + quantized i8 values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseCompressedGradient {
    /// Original gradient dimension.
    pub dimension: usize,
    /// Indices of kept components (Top-k by magnitude).
    pub indices: Vec<u32>,
    /// Quantized values (i8).
    pub values: Vec<i8>,
    /// Scale factor for dequantization: `original ≈ value * scale`.
    pub scale: f32,
    /// Achieved compression ratio.
    pub compression_ratio: f32,
}

impl SparseCompressedGradient {
    /// Compressed size in bytes (indices + values + metadata overhead).
    pub fn size_bytes(&self) -> usize {
        self.indices.len() * 4 + self.values.len() + 16
    }

    /// Original size in bytes (f32 per element).
    pub fn original_size_bytes(&self) -> usize {
        self.dimension * 4
    }
}

/// Top-k compressor with optional error feedback.
pub struct GradientCompressor {
    config: CompressionConfig,
    error_accumulator: Option<Vec<f32>>,
}

impl GradientCompressor {
    /// Create a new compressor.
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            error_accumulator: None,
        }
    }

    /// Compress a gradient slice to a sparse representation.
    pub fn compress(&mut self, gradient: &[f32]) -> SparseCompressedGradient {
        let dim = gradient.len();

        // Apply error feedback if enabled
        let effective: Vec<f32> = if self.config.error_feedback {
            match &self.error_accumulator {
                Some(err) if err.len() == dim => gradient
                    .iter()
                    .zip(err.iter())
                    .map(|(g, e)| g + e)
                    .collect(),
                _ => gradient.to_vec(),
            }
        } else {
            gradient.to_vec()
        };

        // Determine k
        let k = ((dim as f32) / self.config.compression_ratio).ceil() as usize;
        let k = k.max(1).min(dim);

        // Select Top-k by absolute value
        let mut indexed: Vec<(usize, f32)> =
            effective.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|(_, a), (_, b)| b.abs().total_cmp(&a.abs()));

        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        // Quantization scale
        let max_abs = top_k
            .iter()
            .map(|(_, v)| v.abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0);

        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0
        };

        // Quantize
        let indices: Vec<u32> = top_k.iter().map(|(i, _)| *i as u32).collect();
        let values: Vec<i8> = top_k
            .iter()
            .map(|(_, v)| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        // Compression ratio
        let compressed_size = indices.len() * 4 + values.len();
        let original_size = dim * 4;
        let compression_ratio = if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            0.0
        };

        // Update error accumulator
        if self.config.error_feedback {
            let mut error = effective;
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                error[idx as usize] -= val as f32 * scale;
            }
            self.error_accumulator = Some(error);
        }

        SparseCompressedGradient {
            dimension: dim,
            indices,
            values,
            scale,
            compression_ratio,
        }
    }

    /// Decompress back to a full-size gradient vector.
    pub fn decompress(&self, compressed: &SparseCompressedGradient) -> Vec<f32> {
        let mut gradient = vec![0.0_f32; compressed.dimension];
        for (&idx, &val) in compressed.indices.iter().zip(compressed.values.iter()) {
            let i = idx as usize;
            if i < compressed.dimension {
                gradient[i] = val as f32 * compressed.scale;
            }
        }
        gradient
    }

    /// Reset the error accumulator (start of new training run).
    pub fn reset_error(&mut self) {
        self.error_accumulator = None;
    }

    /// Current error accumulator norm (if error feedback is on).
    pub fn error_norm(&self) -> Option<f32> {
        self.error_accumulator
            .as_ref()
            .map(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt())
    }
}

/// Compress and aggregate multiple gradients in sparse-space.
///
/// Compresses each gradient, decompresses, and averages non-zero components.
pub fn compressed_aggregation(
    compressor: &mut GradientCompressor,
    updates: &[GradientUpdate],
) -> Result<Vec<f32>, AggregationError> {
    if updates.is_empty() {
        return Err(AggregationError::NoUpdates);
    }

    let dim = updates[0].gradients.len();
    let mut sum = vec![0.0_f32; dim];
    let mut count = vec![0.0_f32; dim];

    for update in updates {
        let compressed = compressor.compress(&update.gradients);
        let decompressed = compressor.decompress(&compressed);
        for (i, &v) in decompressed.iter().enumerate() {
            if v.abs() > 1e-10 {
                sum[i] += v;
                count[i] += 1.0;
            }
        }
    }

    for i in 0..dim {
        if count[i] > 0.0 {
            sum[i] /= count[i];
        }
    }

    Ok(sum)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let config = CompressionConfig {
            compression_ratio: 2.0,
            quantization_bits: 8,
            error_feedback: false,
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let compressed = compressor.compress(&gradient);

        assert!(compressed.compression_ratio > 1.0);
        assert!(compressed.indices.len() <= 3);

        let decompressed = compressor.decompress(&compressed);
        assert_eq!(decompressed.len(), 5);
    }

    #[test]
    fn test_top_k_selection() {
        let config = CompressionConfig {
            compression_ratio: 5.0,
            quantization_bits: 8,
            error_feedback: false,
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = vec![0.1, 0.2, 10.0, 0.3, 0.4];
        let compressed = compressor.compress(&gradient);

        // Largest by magnitude (10.0) should be kept
        assert!(compressed.indices.contains(&2));
    }

    #[test]
    fn test_error_feedback() {
        let config = CompressionConfig {
            compression_ratio: 10.0,
            quantization_bits: 8,
            error_feedback: true,
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = vec![1.0; 100];
        compressor.compress(&gradient);

        let err1 = compressor.error_norm();
        assert!(err1.is_some());
        assert!(err1.unwrap() > 0.0);

        compressor.compress(&gradient);
        let err2 = compressor.error_norm();
        assert!(err2.is_some());

        compressor.reset_error();
        assert!(compressor.error_norm().is_none());
    }

    #[test]
    fn test_high_compression() {
        let config = CompressionConfig::high_compression();
        let mut compressor = GradientCompressor::new(config);

        let gradient = vec![1.0; 1000];
        let compressed = compressor.compress(&gradient);

        assert!(
            compressed.compression_ratio > 20.0,
            "ratio: {}",
            compressed.compression_ratio
        );
    }

    #[test]
    fn test_config_presets() {
        let default = CompressionConfig::default();
        assert_eq!(default.compression_ratio, 10.0);
        assert!(default.error_feedback);

        let high = CompressionConfig::high_compression();
        assert_eq!(high.compression_ratio, 40.0);

        let low = CompressionConfig::low_latency();
        assert_eq!(low.compression_ratio, 2.0);
        assert!(!low.error_feedback);
    }

    #[test]
    fn test_compressed_size_bytes() {
        let mut compressor = GradientCompressor::new(CompressionConfig::default());
        let gradient = vec![1.0; 100];
        let compressed = compressor.compress(&gradient);

        assert!(compressed.size_bytes() < compressed.original_size_bytes());
    }
}
