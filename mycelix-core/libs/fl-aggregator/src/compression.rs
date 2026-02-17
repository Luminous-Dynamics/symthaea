//! Gradient compression for bandwidth-efficient federated learning.
//!
//! Implements Top-k sparsification and quantization for up to 40x compression.

use crate::error::{AggregatorError, Result};
use crate::Gradient;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Configuration for gradient compression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (e.g., 10.0 = 10x compression).
    pub compression_ratio: f32,

    /// Bits for quantization (8 = int8, 16 = int16).
    pub quantization_bits: u8,

    /// Whether to use error feedback (accumulate residuals).
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
    /// Create config for high compression (40x).
    pub fn high_compression() -> Self {
        Self {
            compression_ratio: 40.0,
            quantization_bits: 8,
            error_feedback: true,
        }
    }

    /// Create config for low latency (minimal compression).
    pub fn low_latency() -> Self {
        Self {
            compression_ratio: 2.0,
            quantization_bits: 16,
            error_feedback: false,
        }
    }
}

/// Compressed gradient representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Original dimension.
    pub dimension: usize,

    /// Indices of non-zero values (Top-k).
    pub indices: Vec<u32>,

    /// Quantized values.
    pub values: Vec<i8>,

    /// Scale factor for dequantization.
    pub scale: f32,

    /// Compression ratio achieved.
    pub compression_ratio: f32,
}

impl CompressedGradient {
    /// Get compressed size in bytes.
    pub fn size_bytes(&self) -> usize {
        // indices (4 bytes each) + values (1 byte each) + metadata
        self.indices.len() * 4 + self.values.len() + 16
    }

    /// Get original size in bytes.
    pub fn original_size_bytes(&self) -> usize {
        self.dimension * 4
    }
}

/// Gradient compressor using Top-k sparsification and quantization.
pub struct GradientCompressor {
    config: CompressionConfig,
    /// Accumulated error from previous compressions (for error feedback).
    error_accumulator: Option<Gradient>,
}

impl GradientCompressor {
    /// Create a new compressor.
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            error_accumulator: None,
        }
    }

    /// Compress a gradient.
    pub fn compress(&mut self, gradient: &Gradient) -> Result<CompressedGradient> {
        let dim = gradient.len();

        // Apply error feedback if enabled
        let effective_gradient = if self.config.error_feedback {
            match &self.error_accumulator {
                Some(error) => {
                    if error.len() == dim {
                        gradient + error
                    } else {
                        gradient.clone()
                    }
                }
                None => gradient.clone(),
            }
        } else {
            gradient.clone()
        };

        // Determine k (number of values to keep)
        let k = (dim as f32 / self.config.compression_ratio).ceil() as usize;
        let k = k.max(1).min(dim);

        // Select Top-k by absolute value
        let mut indexed: Vec<(usize, f32)> = effective_gradient
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed.sort_by(|(_, a), (_, b)| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(Ordering::Equal)
        });

        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        // Find scale for quantization
        let max_abs = top_k
            .iter()
            .map(|(_, v)| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(1.0);

        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0
        };

        // Quantize values
        let indices: Vec<u32> = top_k.iter().map(|(i, _)| *i as u32).collect();
        let values: Vec<i8> = top_k
            .iter()
            .map(|(_, v)| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        // Calculate compression ratio achieved
        let compressed_size = indices.len() * 4 + values.len();
        let original_size = dim * 4;
        let compression_ratio = original_size as f32 / compressed_size as f32;

        // Update error accumulator
        if self.config.error_feedback {
            let mut error = effective_gradient;
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                error[idx as usize] -= val as f32 * scale;
            }
            self.error_accumulator = Some(error);
        }

        tracing::debug!(
            "Compressed gradient: dim={}, k={}, ratio={:.1}x",
            dim,
            k,
            compression_ratio
        );

        Ok(CompressedGradient {
            dimension: dim,
            indices,
            values,
            scale,
            compression_ratio,
        })
    }

    /// Decompress a gradient back to full dimension.
    pub fn decompress(&self, compressed: &CompressedGradient) -> Gradient {
        let mut gradient = Array1::zeros(compressed.dimension);

        for (&idx, &val) in compressed.indices.iter().zip(compressed.values.iter()) {
            if (idx as usize) < compressed.dimension {
                gradient[idx as usize] = val as f32 * compressed.scale;
            }
        }

        gradient
    }

    /// Reset error accumulator.
    pub fn reset_error(&mut self) {
        self.error_accumulator = None;
    }

    /// Get current error norm (if error feedback is enabled).
    pub fn error_norm(&self) -> Option<f32> {
        self.error_accumulator
            .as_ref()
            .map(|e| e.iter().map(|x| x.powi(2)).sum::<f32>().sqrt())
    }
}

/// Compress multiple gradients and aggregate in compressed space.
pub fn compressed_aggregation(
    compressor: &mut GradientCompressor,
    gradients: &[Gradient],
) -> Result<Gradient> {
    if gradients.is_empty() {
        return Err(AggregatorError::Internal(
            "No gradients to aggregate".to_string(),
        ));
    }

    let dim = gradients[0].len();
    let mut sum: Array1<f32> = Array1::zeros(dim);
    let mut count_per_dim: Array1<f32> = Array1::zeros(dim);

    for gradient in gradients {
        let compressed = compressor.compress(gradient)?;
        let decompressed = compressor.decompress(&compressed);

        for i in 0..dim {
            if decompressed[i].abs() > 1e-10 {
                sum[i] += decompressed[i];
                count_per_dim[i] += 1.0;
            }
        }
    }

    // Average where we have values
    for i in 0..dim {
        if count_per_dim[i] > 0.0 {
            sum[i] /= count_per_dim[i];
        }
    }

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compress_decompress() {
        let config = CompressionConfig {
            compression_ratio: 2.0,
            quantization_bits: 8,
            error_feedback: false,
        };
        let mut compressor = GradientCompressor::new(config);

        let gradient = array![1.0, -2.0, 3.0, -4.0, 5.0];
        let compressed = compressor.compress(&gradient).unwrap();

        assert!(compressed.compression_ratio > 1.0);
        assert!(compressed.indices.len() <= 3); // ~2x compression

        let decompressed = compressor.decompress(&compressed);
        assert_eq!(decompressed.len(), 5);
    }

    #[test]
    fn test_top_k_selection() {
        let config = CompressionConfig {
            compression_ratio: 5.0, // Keep 1/5 of values
            quantization_bits: 8,
            error_feedback: false,
        };
        let mut compressor = GradientCompressor::new(config);

        // Gradient with one large value
        let gradient = array![0.1, 0.2, 10.0, 0.3, 0.4];
        let compressed = compressor.compress(&gradient).unwrap();

        // Should keep the largest value (10.0)
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

        // Compress gradient
        let gradient = Array1::from(vec![1.0; 100]);

        compressor.compress(&gradient).unwrap();

        // Error feedback should track residuals
        let error1 = compressor.error_norm();
        assert!(error1.is_some());
        assert!(error1.unwrap() > 0.0);

        // Compress again - error accumulator should be updated
        compressor.compress(&gradient).unwrap();
        let error2 = compressor.error_norm();
        assert!(error2.is_some());

        // Reset should clear error
        compressor.reset_error();
        assert!(compressor.error_norm().is_none());
    }

    #[test]
    fn test_high_compression() {
        let config = CompressionConfig::high_compression();
        let mut compressor = GradientCompressor::new(config);

        let gradient = Array1::from(vec![1.0; 1000]);
        let compressed = compressor.compress(&gradient).unwrap();

        // Should achieve close to 40x compression
        assert!(compressed.compression_ratio > 20.0);
    }
}
