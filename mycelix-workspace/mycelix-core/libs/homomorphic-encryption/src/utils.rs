// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Utility functions for homomorphic encryption operations
//!
//! Provides statistical helpers, parallel processing utilities,
//! and common operations used across the library.

use num_bigint::BigUint;
use tracing::debug;

/// Estimate the bit length needed for a sum of N values
///
/// When adding N encrypted values, the result can be up to N times larger.
/// This helps choose appropriate key sizes.
pub fn estimate_sum_bits(value_bits: usize, count: usize) -> usize {
    value_bits + (count as f64).log2().ceil() as usize + 1
}

/// Compute mean of a slice of f64 values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute variance of a slice of f64 values
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let sum_sq: f64 = values.iter().map(|x| (x - m).powi(2)).sum();
    sum_sq / (values.len() - 1) as f64
}

/// Compute standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Clip values to a range
pub fn clip_values(values: &[f64], min: f64, max: f64) -> Vec<f64> {
    values.iter().map(|&v| v.clamp(min, max)).collect()
}

/// Normalize values to [0, 1] range
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range.abs() < f64::EPSILON {
        return vec![0.5; values.len()];
    }

    values.iter().map(|&v| (v - min) / range).collect()
}

/// Compute L2 norm of a vector
pub fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute L-infinity norm (max absolute value)
pub fn linf_norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

/// Clip gradient by L2 norm
pub fn clip_by_l2_norm(gradient: &[f64], max_norm: f64) -> Vec<f64> {
    let norm = l2_norm(gradient);
    if norm <= max_norm {
        gradient.to_vec()
    } else {
        let scale = max_norm / norm;
        gradient.iter().map(|&v| v * scale).collect()
    }
}

/// Chunk a vector into batches
pub fn chunk_vec<T: Clone>(data: &[T], chunk_size: usize) -> Vec<Vec<T>> {
    data.chunks(chunk_size).map(|c| c.to_vec()).collect()
}

/// Compute the number of bytes needed to store a BigUint
pub fn biguint_byte_size(value: &BigUint) -> usize {
    (value.bits() as usize + 7) / 8
}

/// Log operation timing for debugging
#[inline]
pub fn log_timing(operation: &str, duration_ms: u64) {
    debug!(operation = operation, duration_ms = duration_ms, "HE operation timing");
}

/// Measure operation time (for benchmarking)
pub fn measure_time<F, T>(f: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Format a duration for display
pub fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}μs", duration.as_micros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_sum_bits() {
        // 64-bit values, 100 of them
        let bits = estimate_sum_bits(64, 100);
        assert!(bits >= 64 + 7); // log2(100) ≈ 7
    }

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&values) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&values);
        assert!((var - 4.571).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = normalize(&values);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        let values = vec![3.0, 4.0];
        assert!((l2_norm(&values) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_by_l2_norm() {
        let gradient = vec![3.0, 4.0];
        let clipped = clip_by_l2_norm(&gradient, 2.5);

        // Should be scaled down
        let clipped_norm = l2_norm(&clipped);
        assert!((clipped_norm - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_chunk_vec() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks = chunk_vec(&data, 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7]);
    }

    #[test]
    fn test_format_duration() {
        use std::time::Duration;

        assert!(format_duration(Duration::from_secs(2)).contains("s"));
        assert!(format_duration(Duration::from_millis(500)).contains("ms"));
        assert!(format_duration(Duration::from_micros(100)).contains("μs"));
    }
}
