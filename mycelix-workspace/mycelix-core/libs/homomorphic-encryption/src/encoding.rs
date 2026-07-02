// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fixed-Point Encoding for Floating-Point Values
//!
//! Homomorphic encryption operates on integers, so we need to encode
//! floating-point gradient values as fixed-point integers.

use crate::{HeError, HeResult};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

/// Parameters for fixed-point encoding
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EncodingParams {
    /// Number of decimal places of precision
    pub precision: u32,
    /// Scale factor (10^precision)
    pub scale: i64,
    /// Maximum encodable value
    pub max_value: f64,
    /// Minimum encodable value
    pub min_value: f64,
}

impl EncodingParams {
    /// Create new encoding parameters
    pub fn new(precision: u32) -> Self {
        let scale = 10i64.pow(precision);
        // Limit to avoid overflow in typical operations
        let max_value = (i64::MAX / scale / 1000) as f64;

        Self {
            precision,
            scale,
            max_value,
            min_value: -max_value,
        }
    }

    /// Default parameters for gradient encoding (6 decimal places)
    pub fn for_gradients() -> Self {
        Self::new(6)
    }

    /// High precision parameters (9 decimal places)
    pub fn high_precision() -> Self {
        Self::new(9)
    }

    /// Low precision parameters (3 decimal places)
    pub fn low_precision() -> Self {
        Self::new(3)
    }
}

impl Default for EncodingParams {
    fn default() -> Self {
        Self::for_gradients()
    }
}

/// Fixed-point encoder for converting floats to/from integers
#[derive(Debug, Clone)]
pub struct FixedPointEncoder {
    params: EncodingParams,
}

impl FixedPointEncoder {
    /// Create a new encoder with given parameters
    pub fn new(params: EncodingParams) -> Self {
        Self { params }
    }

    /// Create with default gradient parameters
    pub fn for_gradients() -> Self {
        Self::new(EncodingParams::for_gradients())
    }

    /// Get the encoding parameters
    pub fn params(&self) -> &EncodingParams {
        &self.params
    }

    /// Encode a float as a fixed-point integer
    pub fn encode(&self, value: f64) -> HeResult<i64> {
        if value > self.params.max_value || value < self.params.min_value {
            return Err(HeError::EncodingError(format!(
                "Value {} out of range [{}, {}]",
                value, self.params.min_value, self.params.max_value
            )));
        }

        Ok((value * self.params.scale as f64).round() as i64)
    }

    /// Decode a fixed-point integer back to float
    pub fn decode(&self, encoded: i64) -> f64 {
        encoded as f64 / self.params.scale as f64
    }

    /// Encode a BigInt result back to float
    pub fn decode_bigint(&self, encoded: &BigInt) -> HeResult<f64> {
        let value = encoded
            .to_i64()
            .ok_or_else(|| HeError::Overflow)?;
        Ok(self.decode(value))
    }

    /// Encode a vector of floats
    pub fn encode_vec(&self, values: &[f64]) -> HeResult<Vec<i64>> {
        values.iter().map(|&v| self.encode(v)).collect()
    }

    /// Decode a vector of integers
    pub fn decode_vec(&self, encoded: &[i64]) -> Vec<f64> {
        encoded.iter().map(|&v| self.decode(v)).collect()
    }

    /// Encode with clipping (saturate to bounds instead of error)
    pub fn encode_clipped(&self, value: f64) -> i64 {
        let clamped = value.clamp(self.params.min_value, self.params.max_value);
        (clamped * self.params.scale as f64).round() as i64
    }

    /// Encode vector with clipping
    pub fn encode_vec_clipped(&self, values: &[f64]) -> Vec<i64> {
        values.iter().map(|&v| self.encode_clipped(v)).collect()
    }
}

/// Quantization utilities for gradient compression
pub mod quantization {
    /// Quantize a value to a specified number of bits
    pub fn quantize(value: f64, bits: u32, min_val: f64, max_val: f64) -> i64 {
        let range = max_val - min_val;
        let levels = (1u64 << bits) - 1;

        let normalized = (value - min_val) / range;
        let clamped = normalized.clamp(0.0, 1.0);

        (clamped * levels as f64).round() as i64
    }

    /// Dequantize back to float
    pub fn dequantize(quantized: i64, bits: u32, min_val: f64, max_val: f64) -> f64 {
        let range = max_val - min_val;
        let levels = (1u64 << bits) - 1;

        let normalized = quantized as f64 / levels as f64;
        min_val + normalized * range
    }

    /// Stochastic quantization with randomized rounding
    pub fn stochastic_quantize<R: rand::Rng>(
        value: f64,
        bits: u32,
        min_val: f64,
        max_val: f64,
        rng: &mut R,
    ) -> i64 {
        let range = max_val - min_val;
        let levels = (1u64 << bits) - 1;

        let normalized = (value - min_val) / range;
        let clamped = normalized.clamp(0.0, 1.0);
        let scaled = clamped * levels as f64;

        let floor = scaled.floor();
        let prob = scaled - floor;

        if rng.gen::<f64>() < prob {
            (floor + 1.0) as i64
        } else {
            floor as i64
        }
    }

    /// Compute optimal quantization range from data
    pub fn compute_range(values: &[f64], percentile: f64) -> (f64, f64) {
        if values.is_empty() {
            return (-1.0, 1.0);
        }

        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let low_idx = ((1.0 - percentile) / 2.0 * sorted.len() as f64) as usize;
        let high_idx = ((1.0 + percentile) / 2.0 * sorted.len() as f64) as usize;

        let min_val = sorted[low_idx.min(sorted.len() - 1)];
        let max_val = sorted[high_idx.min(sorted.len() - 1)];

        // Add small margin
        let margin = (max_val - min_val) * 0.01;
        (min_val - margin, max_val + margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_params() {
        let params = EncodingParams::new(6);
        assert_eq!(params.scale, 1_000_000);
        assert_eq!(params.precision, 6);
    }

    #[test]
    fn test_encode_decode() {
        let encoder = FixedPointEncoder::for_gradients();

        let original = 0.123456;
        let encoded = encoder.encode(original).unwrap();
        let decoded = encoder.decode(encoded);

        assert!((original - decoded).abs() < 1e-6);
    }

    #[test]
    fn test_encode_negative() {
        let encoder = FixedPointEncoder::for_gradients();

        let original = -0.789012;
        let encoded = encoder.encode(original).unwrap();
        let decoded = encoder.decode(encoded);

        assert!((original - decoded).abs() < 1e-6);
    }

    #[test]
    fn test_encode_vec() {
        let encoder = FixedPointEncoder::for_gradients();

        let values = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let encoded = encoder.encode_vec(&values).unwrap();
        let decoded = encoder.decode_vec(&encoded);

        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1e-6);
        }
    }

    #[test]
    fn test_encode_clipped() {
        let encoder = FixedPointEncoder::new(EncodingParams::new(3));

        // Value within range
        let in_range = encoder.encode_clipped(100.5);
        assert_eq!(encoder.decode(in_range), 100.5);

        // Value at boundary should be clipped
        let huge = encoder.encode_clipped(f64::MAX);
        assert!(huge <= i64::MAX);
    }

    #[test]
    fn test_quantization() {
        use quantization::*;

        let value = 0.5;
        let quantized = quantize(value, 8, 0.0, 1.0);
        let dequantized = dequantize(quantized, 8, 0.0, 1.0);

        assert!((value - dequantized).abs() < 0.01);
    }

    #[test]
    fn test_compute_range() {
        use quantization::*;

        let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let (min_val, max_val) = compute_range(&values, 0.95);

        assert!(min_val <= -0.9);
        assert!(max_val >= 1.9);
    }
}
