// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Continuous Hypervector
//!
//! A portable, self-contained implementation of continuous-valued hypervectors
//! for Hyperdimensional Computing (HDC).
//!
//! Operations:
//! - **Bind** (element-wise multiply): creates associations, result dissimilar to inputs
//! - **Bundle** (element-wise average): creates superpositions, result similar to all inputs
//! - **Permute** (cyclic shift): encodes sequence/position information
//! - **Similarity** (cosine): measures relatedness in [-1, 1]

use serde::{Deserialize, Serialize};

/// Standard HDC dimension (2^14 = 16,384).
pub const HDC_DIMENSION: usize = 16_384;

/// A continuous-valued hypervector using f32 components.
///
/// Each component typically ranges in [-1, 1] though operations may
/// produce values outside this range before normalization.
///
/// # Memory
///
/// 64 KB per vector at the default 16,384 dimensions (16,384 x 4 bytes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuousHV {
    /// Vector components.
    pub values: Vec<f32>,
}

impl ContinuousHV {
    /// Create a zero-initialized hypervector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Create a deterministic random hypervector with values in [-1, 1].
    ///
    /// Uses xorshift64 seeded with a golden-ratio constant mix to avoid
    /// the fixed point at seed=0.
    pub fn new_random(dim: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(dim);
        let mut state = seed ^ 0x9E3779B97F4A7C15;

        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Map top 24 bits to [-1, 1] for full f32 mantissa precision.
            let normalized = ((state >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            values.push(normalized);
        }

        Self { values }
    }

    /// Create from an existing vector of values.
    pub fn from_values(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Create from a slice.
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            values: slice.to_vec(),
        }
    }

    /// Get the dimension (number of components).
    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Binding operation (element-wise multiplication).
    ///
    /// Creates an association between two vectors. The result is
    /// dissimilar to both inputs (for random HVs).
    ///
    /// # Panics
    /// Panics if dimensions do not match.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { values }
    }

    /// Bundling operation (element-wise average).
    ///
    /// Creates a superposition of vectors. The result is similar to all inputs.
    ///
    /// Returns a zero vector of `HDC_DIMENSION` if the slice is empty.
    #[inline]
    pub fn bundle(hvs: &[&Self]) -> Self {
        if hvs.is_empty() {
            return Self::new(HDC_DIMENSION);
        }
        let dim = hvs[0].values.len();
        let inv_n = 1.0 / hvs.len() as f32;
        let mut values = vec![0.0f32; dim];
        for hv in hvs {
            for (acc, &v) in values.iter_mut().zip(hv.values.iter()) {
                *acc += v;
            }
        }
        for v in values.iter_mut() {
            *v *= inv_n;
        }
        Self { values }
    }

    /// Cosine similarity in [-1, 1].
    ///
    /// For random vectors: similarity is approximately 0.
    /// For identical vectors: similarity = 1.
    /// For opposite vectors: similarity = -1.
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let mut dot = 0.0f32;
        let mut norm_a_sq = 0.0f32;
        let mut norm_b_sq = 0.0f32;

        for i in 0..self.values.len() {
            let a = self.values[i];
            let b = other.values[i];
            dot += a * b;
            norm_a_sq += a * a;
            norm_b_sq += b * b;
        }

        let denom = (norm_a_sq * norm_b_sq).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (dot / denom).clamp(-1.0, 1.0)
    }

    /// Cyclic right shift by `positions` elements.
    ///
    /// Permutation is the standard HDC mechanism for encoding position
    /// or sequence information. `permute(1)` followed by `bind` gives
    /// non-commutative temporal binding.
    pub fn permute(&self, positions: usize) -> Self {
        let dim = self.values.len();
        if dim == 0 {
            return self.clone();
        }
        let shift = positions % dim;
        if shift == 0 {
            return self.clone();
        }
        let mut values = vec![0.0f32; dim];
        for i in 0..dim {
            values[(i + shift) % dim] = self.values[i];
        }
        Self { values }
    }

    /// L2 norm.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length.
    ///
    /// Returns a clone of self if the norm is near zero.
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return self.clone();
        }
        Self {
            values: self.values.iter().map(|x| x / n).collect(),
        }
    }

    /// Scale by a constant factor.
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            values: self.values.iter().map(|x| x * factor).collect(),
        }
    }

    /// In-place scale.
    #[inline]
    pub fn scale_in_place(&mut self, factor: f32) {
        for v in self.values.iter_mut() {
            *v *= factor;
        }
    }

    /// Add `other * scale` to self in-place.
    #[inline]
    pub fn add_scaled(&mut self, other: &Self, scale: f32) {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (s, &o) in self.values.iter_mut().zip(other.values.iter()) {
            *s += o * scale;
        }
    }

    /// In-place linear interpolation: `self = (1 - alpha) * self + alpha * target`.
    #[inline]
    pub fn lerp_in_place(&mut self, target: &Self, alpha: f32) {
        debug_assert_eq!(self.values.len(), target.values.len());
        let one_minus = 1.0 - alpha;
        for (s, &t) in self.values.iter_mut().zip(target.values.iter()) {
            *s = one_minus * *s + alpha * t;
        }
    }

    /// Element-wise addition, returning a new vector.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction, returning a new vector.
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Raw dot product (inner product).
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Generate approximately orthogonal unit hypervectors via modified Gram-Schmidt.
    ///
    /// Each output vector has unit L2 norm and near-zero dot product with all others.
    pub fn orthogonal_set(dim: usize, count: usize, seed: u64) -> Vec<Self> {
        if count == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let mut v = Self::new_random(dim, seed.wrapping_add(i as u64 * 7919));
            for prev in &result {
                let proj_coeff = v.dot(prev);
                for (vi, pi) in v.values.iter_mut().zip(prev.values.iter()) {
                    *vi -= proj_coeff * pi;
                }
            }
            let n = v.norm();
            if n > 1e-10 {
                for vi in v.values.iter_mut() {
                    *vi /= n;
                }
            }
            result.push(v);
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_zero() {
        let hv = ContinuousHV::new(128);
        assert_eq!(hv.dim(), 128);
        assert!(hv.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_creation_random_deterministic() {
        let a = ContinuousHV::new_random(1024, 42);
        let b = ContinuousHV::new_random(1024, 42);
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn test_random_values_in_range() {
        let hv = ContinuousHV::new_random(HDC_DIMENSION, 99);
        for &v in &hv.values {
            assert!(v >= -1.0 && v <= 1.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_bind_dissimilarity() {
        let a = ContinuousHV::new_random(1024, 1);
        let b = ContinuousHV::new_random(1024, 2);
        let bound = a.bind(&b);
        // Binding two random vectors should produce a vector dissimilar to both
        assert!(bound.similarity(&a).abs() < 0.15);
        assert!(bound.similarity(&b).abs() < 0.15);
    }

    #[test]
    fn test_bundle_similarity() {
        let a = ContinuousHV::new_random(1024, 10);
        let b = ContinuousHV::new_random(1024, 20);
        let bundled = ContinuousHV::bundle(&[&a, &b]);
        // Bundle should be similar to both inputs
        assert!(bundled.similarity(&a) > 0.4);
        assert!(bundled.similarity(&b) > 0.4);
    }

    #[test]
    fn test_permute_changes_vector() {
        let hv = ContinuousHV::new_random(128, 5);
        let perm = hv.permute(1);
        assert_ne!(hv.values, perm.values);
    }

    #[test]
    fn test_permute_roundtrip() {
        let hv = ContinuousHV::new_random(128, 5);
        let roundtrip = hv.permute(128);
        assert_eq!(hv.values, roundtrip.values);
    }

    #[test]
    fn test_normalize() {
        let hv = ContinuousHV::new_random(256, 7);
        let normed = hv.normalize();
        let n = normed.norm();
        assert!((n - 1.0).abs() < 1e-5, "Norm after normalize: {}", n);
    }

    #[test]
    fn test_similarity_range() {
        let a = ContinuousHV::new_random(512, 1);
        let b = ContinuousHV::new_random(512, 2);
        let sim = a.similarity(&b);
        assert!(sim >= -1.0 && sim <= 1.0);
        // Self-similarity should be 1.0
        let self_sim = a.similarity(&a);
        assert!((self_sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_in_place() {
        let mut a = ContinuousHV::new_random(64, 1);
        let b = ContinuousHV::new_random(64, 2);
        let original = a.clone();
        a.lerp_in_place(&b, 0.5);
        // Result should be between a and b
        assert!(a.similarity(&original) > 0.4);
        assert!(a.similarity(&b) > 0.4);
    }

    #[test]
    fn test_add_scaled() {
        let mut a = ContinuousHV::new(4);
        a.values = vec![1.0, 2.0, 3.0, 4.0];
        let b = ContinuousHV::from_values(vec![10.0, 20.0, 30.0, 40.0]);
        a.add_scaled(&b, 0.1);
        assert!((a.values[0] - 2.0).abs() < 1e-6);
        assert!((a.values[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_set() {
        let set = ContinuousHV::orthogonal_set(1024, 5, 42);
        assert_eq!(set.len(), 5);
        // Each vector should be unit norm
        for v in &set {
            assert!((v.norm() - 1.0).abs() < 1e-4);
        }
        // Pairwise dot products should be near zero
        for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                let d = set[i].dot(&set[j]).abs();
                assert!(d < 0.05, "Dot product {} between {} and {}", d, i, j);
            }
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let hv = ContinuousHV::new_random(64, 123);
        let json = serde_json::to_string(&hv).unwrap();
        let restored: ContinuousHV = serde_json::from_str(&json).unwrap();
        assert_eq!(hv.values, restored.values);
    }
}
