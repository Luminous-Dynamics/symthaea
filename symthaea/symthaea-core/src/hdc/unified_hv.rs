// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Hypervector Types
//!
//! This module provides a unified interface for hypervector operations across
//! both continuous (f32) and binary (bipolar) representations.
//!
//! ## Why Unified HV?
//!
//! Previously, Symthaea had two separate types:
//! - `ContinuousHV` - f32 continuous values in [-1, 1]
//! - `BinaryHV` - 16,384 bits bipolar {-1, +1}
//!
//! The name "ContinuousHV" was confusing (suggests "actual" vs "fake").
//! This module provides:
//! - `ContinuousHV` - Clear name for f32 representation
//! - `HV` enum - Unified type wrapping ContinuousHV and BinaryHV
//!
//! ## Theoretical Basis
//!
//! Hyperdimensional Computing (HDC) uses high-dimensional vectors where:
//! - Random vectors are nearly orthogonal (similarity ≈ 0)
//! - Binding (⊗) creates associations: A⊗B is dissimilar to both A and B
//! - Bundling (⊕) creates superpositions: A⊕B is similar to both A and B
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::unified_hv::{ContinuousHV, HV};
//! use symthaea::hdc::binary_hv::BinaryHV;
//!
//! // Continuous operations
//! let a = ContinuousHV::random(16384, 42);
//! let b = ContinuousHV::random(16384, 43);
//! let bound = a.bind(&b);
//! let bundled = ContinuousHV::bundle(&[&a, &b]);
//! let sim = a.similarity(&b);  // ≈ 0 for random vectors
//!
//! // Binary operations (more efficient)
//! let x = BinaryHV::random(42);
//! let y = BinaryHV::random(43);
//! let bound = x.bind(&y);  // XOR
//! let bundled = BinaryHV::bundle(&[x, y]);  // Majority vote
//!
//! // Unified interface
//! let hv_a = HV::Continuous(a);
//! let hv_x = HV::Binary(x);
//! ```

pub use super::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global Cognitive Stride (Adaptive SSM Principle).
/// 1 = Full resolution (High Power)
/// 4 = Standard (Balanced)
/// 8+ = Low resolution (Power Saving)
pub static STRIDE: AtomicUsize = AtomicUsize::new(4);

/// Set the global cognitive stride for similarity calculations.
pub fn set_cognitive_stride(stride: usize) {
    STRIDE.store(stride.max(1), Ordering::Relaxed);
}

/// Get the current cognitive stride.
pub fn get_cognitive_stride() -> usize {
    STRIDE.load(Ordering::Relaxed)
}

/// Standard HDC dimension (2^14 = 16,384)
/// This is SIMD-optimized and matches research consensus.
pub const HDC_DIMENSION: usize = 16_384;

/// Number of bytes for binary representation (16,384 bits / 8)
pub const BINARY_BYTES: usize = HDC_DIMENSION / 8;

// ═══════════════════════════════════════════════════════════════════════════════
// CONTINUOUS HYPERVECTOR (formerly ContinuousHV)
// ═══════════════════════════════════════════════════════════════════════════════

/// Continuous-valued hypervector using f32 components in range [-1, 1].
///
/// Use this when:
/// - You need fine-grained similarity measurements
/// - Gradient-based optimization is required
/// - Precision matters more than memory/speed
///
/// # Memory: 64KB per vector (16,384 × 4 bytes)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuousHV {
    /// Vector components in [-1, 1]
    pub values: Vec<f32>,
}

impl ContinuousHV {
    /// Default dimension (uses HDC_DIMENSION constant)
    pub const DEFAULT_DIM: usize = HDC_DIMENSION;

    /// Create from existing values
    pub fn from_vec(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Create a zero vector
    pub fn zero(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Create a vector of ones
    pub fn ones(dim: usize) -> Self {
        Self {
            values: vec![1.0; dim],
        }
    }

    /// Create a random hypervector with deterministic seed
    ///
    /// Uses simple but fast PRNG for reproducibility.
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(dim);
        // Avoid xorshift64 fixed point: seed=0 produces all zeros.
        // Mixing with a constant ensures every seed yields a non-degenerate sequence.
        let mut state = seed ^ 0x9E3779B97F4A7C15; // golden-ratio constant

        for _ in 0..dim {
            // Simple xorshift64 for speed and reproducibility
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Map to [-1, 1] using top 24 bits for full f32 precision.
            // (state as f32 / u64::MAX as f32) loses precision since u64 >> f32 mantissa.
            let normalized = ((state >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            values.push(normalized);
        }

        Self { values }
    }

    /// Create a deterministic HV from a genesis seed with domain label.
    ///
    /// Squeezes `dim * 4` bytes from SHAKE-256, maps each 4-byte chunk
    /// to an f32 in \[-1, 1\]. Same phrase + label + dim → identical vector.
    pub fn from_genesis(genesis: &crate::genesis::GenesisSeed, label: &str, dim: usize) -> Self {
        genesis.hv(label, dim)
    }

    /// Create a random hypervector with default dimension
    pub fn random_default(seed: u64) -> Self {
        Self::random(Self::DEFAULT_DIM, seed)
    }

    /// Create a basis vector (one-hot with noise)
    ///
    /// Useful for creating orthogonal identity vectors.
    pub fn basis(index: usize, dim: usize) -> Self {
        let mut values = vec![0.0; dim];
        if index < dim {
            values[index] = 1.0;
        }
        Self { values }
    }

    /// Create from raw values
    pub fn from_values(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Get dimension
    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Binding operation (element-wise multiplication)
    ///
    /// Creates an association between two vectors.
    /// Result is dissimilar to both inputs.
    ///
    /// # Properties
    /// - Commutative: A⊗B = B⊗A
    /// - Self-inverse: A⊗A ≈ 1
    /// - Preserves similarity: sim(A⊗C, B⊗C) = sim(A, B)
    ///
    /// Non-commutative temporal binding: ρ(self) ⊙ other.
    ///
    /// Cyclic permutation + Hadamard product for continuous domain.
    /// `a.bind_temporal(&b) ≠ b.bind_temporal(&a)`, encoding temporal order.
    #[inline]
    pub fn bind_temporal(&self, other: &Self) -> Self {
        self.permute(1).bind(other)
    }

    /// # Performance
    /// Uses SIMD acceleration when the `simd` feature is enabled (4x+ speedup).
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");

        #[cfg(feature = "simd")]
        {
            let values = crate::hdc::simd_continuous::bind_simd(&self.values, &other.values);
            Self { values }
        }

        #[cfg(not(feature = "simd"))]
        {
            let values: Vec<f32> = self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a * b)
                .collect();
            Self { values }
        }
    }

    /// Precision-weighted binding (Eq. 17 in HAI paper)
    ///
    /// `bind_π(h₁, h₂, π) = (h₁ ⊗ h₂) ⊙ σ(π·1)`
    ///
    /// Computes the standard Hadamard bind, then scales element-wise by the
    /// sigmoid of the precision parameter. High precision (π >> 0) approximates
    /// a standard bind; low precision (π << 0) attenuates the binding signal
    /// toward zero, reflecting uncertainty in the association.
    ///
    /// This implements the Active Inference precision-weighting mechanism:
    /// associations formed under high precision are strongly bound, while
    /// associations formed under low precision are attenuated.
    ///
    /// # Arguments
    /// * `other` - The vector to bind with
    /// * `precision` - Precision parameter (log-scale; 0.0 → sigmoid = 0.5)
    #[inline]
    pub fn bind_precision(&self, other: &Self, precision: f32) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let sigmoid = 1.0 / (1.0 + (-precision).exp());
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b * sigmoid)
            .collect();
        Self { values }
    }

    /// Bundling operation (element-wise average)
    ///
    /// Creates a superposition of vectors.
    /// Result is similar to all inputs.
    ///
    /// # Properties
    /// - Commutative and associative
    /// - sim(bundle(A,B), A) > 0
    /// - sim(bundle(A,B), B) > 0
    #[inline]
    pub fn bundle(hvs: &[&Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].values.len();

        #[cfg(feature = "simd")]
        {
            // SIMD path requires uniform dimensions; fall through to scalar if mismatched
            let uniform = hvs.iter().all(|hv| hv.values.len() == dim);
            if uniform {
                let slices: Vec<&[f32]> = hvs.iter().map(|hv| hv.values.as_slice()).collect();
                let weights = vec![1.0f32; hvs.len()];
                let values = crate::hdc::simd_continuous::bundle_simd(&slices, &weights);
                return Self { values };
            }
        }

        let inv_n = 1.0 / hvs.len() as f32;

        // Scalar fallback: accumulate row-by-row for better cache locality
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

    /// Weighted bundling
    ///
    /// # Performance
    /// Uses SIMD acceleration when the `simd` feature is enabled (4x+ speedup).
    pub fn weighted_bundle(hvs: &[&Self], weights: &[f32]) -> Self {
        if hvs.is_empty() || weights.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].values.len();

        #[cfg(feature = "simd")]
        {
            let uniform = hvs.iter().all(|hv| hv.values.len() == dim);
            if uniform {
                let slices: Vec<&[f32]> = hvs.iter().map(|hv| hv.values.as_slice()).collect();
                let values = crate::hdc::simd_continuous::bundle_simd(&slices, weights);
                return Self { values };
            }
        }

        // Scalar fallback (also used when dimensions are non-uniform)
        let weight_sum: f32 = weights.iter().sum();

        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let weighted_sum: f32 = hvs
                    .iter()
                    .zip(weights.iter())
                    .filter(|(hv, _)| i < hv.values.len())
                    .map(|(hv, w)| hv.values[i] * w)
                    .sum();
                weighted_sum / weight_sum
            })
            .collect();

        Self { values }
    }

    /// Cosine similarity in range [-1, 1]
    ///
    /// For random vectors: similarity ≈ 0
    /// For identical vectors: similarity = 1
    /// For opposite vectors: similarity = -1
    ///
    /// # Performance
    /// Uses SIMD acceleration when the `simd` feature is enabled (4x+ speedup).
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Dimension mismatch in similarity(): {} vs {}",
            self.values.len(),
            other.values.len()
        );

        #[cfg(feature = "simd")]
        {
            crate::hdc::simd_continuous::similarity_simd(&self.values, &other.values)
        }

        #[cfg(not(feature = "simd"))]
        {
            let mut dot = 0.0f32;
            let mut norm_a_sq = 0.0f32;
            let mut norm_b_sq = 0.0f32;

            // SSM Selective Scan: We use the global dynamic STRIDE for
            // biological homeostasis (Cognitive Throttle).
            let stride = STRIDE.load(Ordering::Relaxed);

            for i in (0..self.values.len()).step_by(stride) {
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
    }

    /// Raw dot product (inner product) of two hypervectors.
    ///
    /// Unlike `similarity()`, this is NOT normalized by vector norms.
    /// Useful for Gram-Schmidt orthogonalization and projection computations.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Dimension mismatch in dot(): {} vs {}",
            self.values.len(),
            other.values.len()
        );
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Generate a set of approximately orthogonal unit hypervectors via modified Gram-Schmidt.
    ///
    /// For `count` HVs at dimension `dim`, this is O(count² × dim) — negligible for
    /// typical usage (count=5 at 16,384D ≈ 400K flops).
    ///
    /// Each output vector has unit L2 norm and near-zero dot product with all others.
    /// Deterministic given the same seed.
    pub fn orthogonal_set(dim: usize, count: usize, seed: u64) -> Vec<Self> {
        if count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            // Generate a random HV with offset seed
            let mut v = Self::random(dim, seed.wrapping_add(i as u64 * 7919));

            // Modified Gram-Schmidt: project out all previous vectors
            for prev in &result {
                let proj_coeff = v.dot(prev); // prev is unit norm, so proj = dot * prev
                                              // v = v - proj_coeff * prev
                for (vi, pi) in v.values.iter_mut().zip(prev.values.iter()) {
                    *vi -= proj_coeff * pi;
                }
            }

            // Normalize to unit length
            let norm = v.norm();
            if norm > 1e-10 {
                for vi in v.values.iter_mut() {
                    *vi /= norm;
                }
            }

            result.push(v);
        }

        result
    }

    /// L2 norm
    ///
    /// # Performance
    /// Uses SIMD acceleration when the `simd` feature is enabled (4x+ speedup).
    #[inline]
    pub fn norm(&self) -> f32 {
        #[cfg(feature = "simd")]
        {
            crate::hdc::simd_continuous::norm_simd(&self.values)
        }

        #[cfg(not(feature = "simd"))]
        {
            self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
        }
    }

    /// Normalize to unit length
    #[inline]
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm < 1e-10 {
            return self.clone();
        }

        Self {
            values: self.values.iter().map(|x| x / norm).collect(),
        }
    }

    /// Scale by constant
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            values: self.values.iter().map(|x| x * factor).collect(),
        }
    }

    /// Inverse for unbinding operations
    ///
    /// For continuous hypervectors with binding as element-wise multiplication,
    /// the inverse is the element-wise reciprocal: inv(a_i) = 1/a_i.
    /// Near-zero elements are mapped to zero to avoid division by zero.
    ///
    /// Property: `A.bind(&A.inverse())` should yield a vector near the identity
    /// (all ones) for non-zero elements.
    pub fn inverse(&self) -> Self {
        const EPSILON: f32 = 1e-7;
        Self {
            values: self
                .values
                .iter()
                .map(|&v| if v.abs() < EPSILON { 0.0 } else { 1.0 / v })
                .collect(),
        }
    }

    /// Element-wise addition
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Dimension mismatch in add(): {} vs {}",
            self.values.len(),
            other.values.len()
        );

        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "Dimension mismatch in subtract(): {} vs {}",
            self.values.len(),
            other.values.len()
        );

        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// In-place linear interpolation: self = a * x + b * self
    /// Avoids two allocations compared to `x.scale(a).add(&self.scale(b))`.
    #[inline]
    pub fn lerp_in_place(&mut self, other: &Self, self_weight: f32, other_weight: f32) {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (s, &o) in self.values.iter_mut().zip(other.values.iter()) {
            *s = self_weight * *s + other_weight * o;
        }
    }

    /// In-place scale
    #[inline]
    pub fn scale_in_place(&mut self, factor: f32) {
        for v in self.values.iter_mut() {
            *v *= factor;
        }
    }

    /// In-place add
    #[inline]
    pub fn add_in_place(&mut self, other: &Self) {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (s, &o) in self.values.iter_mut().zip(other.values.iter()) {
            *s += o;
        }
    }

    /// Element-wise multiplication (Hadamard product)
    ///
    /// Alias for bind() - useful in numerical contexts like optimizer updates.
    pub fn hadamard_product(&self, other: &Self) -> Self {
        self.bind(other)
    }

    /// Apply a function to each element
    ///
    /// Useful for element-wise transformations like sqrt, reciprocal, etc.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self {
            values: self.values.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Element-wise tanh activation
    pub fn tanh(&self) -> Self {
        Self {
            values: self.values.iter().map(|x| x.tanh()).collect(),
        }
    }

    /// Element-wise sigmoid activation
    pub fn sigmoid(&self) -> Self {
        Self {
            values: self
                .values
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect(),
        }
    }

    /// Encode a set of base vectors weighted by corresponding scalar values.
    ///
    /// This is a common HDC encoding pattern: scale each base vector by its
    /// corresponding weight, then bundle (average) the results.
    /// Equivalent to `bundle` of individually-scaled vectors but avoids
    /// intermediate allocations.
    ///
    /// # Panics
    /// - If `bases` and `weights` have different lengths.
    /// - If `bases` is empty.
    pub fn encode_weighted(bases: &[ContinuousHV], weights: &[f32]) -> Self {
        assert_eq!(
            bases.len(),
            weights.len(),
            "bases and weights must have equal length"
        );
        assert!(!bases.is_empty(), "cannot encode empty set");
        let dim = bases[0].dim();
        let inv_n = 1.0 / bases.len() as f32;
        let mut values = vec![0.0f32; dim];
        for (base, &w) in bases.iter().zip(weights.iter()) {
            for (acc, &v) in values.iter_mut().zip(base.values.iter()) {
                *acc += v * w;
            }
        }
        for v in values.iter_mut() {
            *v *= inv_n;
        }
        Self { values }
    }

    /// Bundle multiple vectors (owned version)
    ///
    /// Takes owned values instead of references. Useful when you have a `Vec<ContinuousHV>`
    /// and don't want to build a reference slice.
    pub fn bundle_owned(hvs: &[Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].values.len();

        #[cfg(feature = "simd")]
        {
            // SIMD path requires uniform dimensions; fall through to scalar if mismatched
            let uniform = hvs.iter().all(|hv| hv.values.len() == dim);
            if uniform {
                let slices: Vec<&[f32]> = hvs.iter().map(|hv| hv.values.as_slice()).collect();
                let weights = vec![1.0f32; hvs.len()];
                let values = crate::hdc::simd_continuous::bundle_simd(&slices, &weights);
                return Self { values };
            }
        }

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

    /// Get a slice view of the internal values
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Create a ContinuousHV from a slice
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            values: slice.to_vec(),
        }
    }

    /// Perturb the hypervector by adding random noise
    ///
    /// The rate controls the magnitude of perturbation (0.0 = no change, 1.0 = significant change)
    pub fn perturb(&self, rate: f32) -> Self {
        use blake3::Hasher;
        use std::sync::atomic::{AtomicU64, Ordering};
        static PERTURB_COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut hasher = Hasher::new();
        for v in &self.values {
            hasher.update(&v.to_le_bytes());
        }
        // Use an atomic counter for entropy instead of SystemTime (which panics on wasm32)
        hasher.update(
            &PERTURB_COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .to_le_bytes(),
        );

        let mut bytes = vec![0u8; self.values.len() * 4];
        let mut xof = hasher.finalize_xof();
        xof.fill(&mut bytes);

        let values: Vec<f32> = self
            .values
            .iter()
            .zip(bytes.chunks_exact(4))
            .map(|(&val, chunk)| {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let noise = (bits as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                val + noise * rate
            })
            .collect();

        Self { values }
    }

    /// Permute elements (rotation)
    ///
    /// Useful for sequence encoding: P^n(A) represents A at position n.
    pub fn permute(&self, shift: usize) -> Self {
        let dim = self.values.len();
        let shift = shift % dim;

        let mut values = vec![0.0; dim];
        for i in 0..dim {
            values[(i + shift) % dim] = self.values[i];
        }

        Self { values }
    }

    /// Inverse permutation
    pub fn inverse_permute(&self, shift: usize) -> Self {
        let dim = self.values.len();
        self.permute(dim - (shift % dim))
    }

    /// Perform 'Holographic Dilation' - scale dimensionality while preserving semantics.
    ///
    /// Scaling is performed using a fractal folding/unfolding technique
    /// that maintains semantic alignment across different resolutions.
    ///
    /// * **UPSAMPLING (Unfolding)**: Expands the vector by tiling it with permutations.
    ///   This preserves the original signal while providing more orthogonal space.
    /// * **DOWNSAMPLING (Folding)**: Bundles constituent segments after applying
    ///   inverse permutations. This allows near-perfect recovery of the original signal
    ///   (similarity > 0.99 for perfectly aligned target_dim multiples).
    pub fn dilate(&self, target_dim: usize) -> Self {
        let current_dim = self.values.len();

        if current_dim == target_dim {
            return self.clone();
        }

        if target_dim > current_dim {
            // UPSAMPLING (Unfolding)
            // We expand the vector by tiling it with permutations.
            // This maintains the original semantic signal while
            // providing more orthogonal space for new associations.
            let mut new_values = Vec::with_capacity(target_dim);
            let mut current_vec = self.clone();

            while new_values.len() < target_dim {
                let chunk_size = current_vec.values.len().min(target_dim - new_values.len());
                new_values.extend_from_slice(&current_vec.values[..chunk_size]);
                // Permute for the next chunk to avoid trivial repetition
                current_vec = current_vec.permute(1);
            }
            Self { values: new_values }
        } else {
            // DOWNSAMPLING (Folding)
            // We fold the vector back by bundling its constituent segments.
            // We apply the inverse permutation to each segment before bundling
            // to recover the original semantic alignment.
            let chunk_count = (self.values.len() + target_dim - 1) / target_dim;
            let mut folded = vec![0.0f32; target_dim];

            for k in 0..chunk_count {
                let start = k * target_dim;
                let end = (start + target_dim).min(self.values.len());
                let mut chunk_values = self.values[start..end].to_vec();

                // If chunk is smaller than target (last chunk), pad with zeros for inverse_permute
                if chunk_values.len() < target_dim {
                    chunk_values.resize(target_dim, 0.0);
                }

                let chunk_hv = Self {
                    values: chunk_values,
                };
                // Un-rotate: Chunk k was rotated by k steps during upsampling
                let unrotated = chunk_hv.inverse_permute(k);

                for (i, &val) in unrotated.values.iter().enumerate() {
                    folded[i] += val / chunk_count as f32;
                }
            }
            Self { values: folded }
        }
    }

    /// Convert to binary representation using threshold
    pub fn to_binary(&self, threshold: f32) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        let len = self.values.len().min(16_384);

        for i in 0..len {
            if self.values[i] > threshold {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }

        BinaryHV(bytes)
    }

    /// Convert to binary using probabilistic binarization
    ///
    /// Preserves heterogeneity better than threshold.
    pub fn to_binary_probabilistic(&self, seed: u64) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        let mut state = seed ^ 0x9E3779B97F4A7C15; // avoid xorshift64 fixed-point at 0
        let len = self.values.len().min(16_384);

        for i in 0..len {
            // Sigmoid to probability
            let prob = 1.0 / (1.0 + (-self.values[i] * 5.0).exp());

            // Random threshold
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let rand = (state as f32) / (u64::MAX as f32);

            if rand < prob {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }

        BinaryHV(bytes)
    }
}

impl Default for ContinuousHV {
    fn default() -> Self {
        Self::zero(HDC_DIMENSION)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED HV ENUM
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified hypervector type for mixed operations.
///
/// Use this when you need to work with both representations
/// or when the representation might change at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum HV {
    /// Continuous f32 representation
    Continuous(ContinuousHV),

    /// Binary bipolar representation
    Binary(BinaryHV),
}

impl HV {
    /// Get dimension
    pub fn dim(&self) -> usize {
        match self {
            HV::Continuous(hv) => hv.dim(),
            HV::Binary(hv) => BinaryHV::DIM,
        }
    }

    /// Similarity between any two HVs (converts if needed)
    pub fn similarity(&self, other: &HV) -> f32 {
        match (self, other) {
            (HV::Continuous(a), HV::Continuous(b)) => a.similarity(b),
            (HV::Binary(a), HV::Binary(b)) => {
                // BinaryHV::similarity returns [0, 1], convert to bipolar [-1, 1]
                a.similarity(b) * 2.0 - 1.0
            }
            (HV::Continuous(a), HV::Binary(b)) => {
                let b_cont = ContinuousHV::from_vec(b.to_bipolar());
                a.similarity(&b_cont)
            }
            (HV::Binary(a), HV::Continuous(b)) => {
                let a_cont = ContinuousHV::from_vec(a.to_bipolar());
                a_cont.similarity(b)
            }
        }
    }

    /// Convert to continuous (no-op if already continuous)
    pub fn to_continuous(&self) -> ContinuousHV {
        match self {
            HV::Continuous(hv) => hv.clone(),
            HV::Binary(hv) => ContinuousHV::from_vec(hv.to_bipolar()),
        }
    }

    /// Convert to binary
    pub fn to_binary(&self, threshold: f32) -> BinaryHV {
        match self {
            HV::Continuous(hv) => hv.to_binary(threshold),
            HV::Binary(hv) => *hv,
        }
    }

    /// Check if continuous
    pub fn is_continuous(&self) -> bool {
        matches!(self, HV::Continuous(_))
    }

    /// Check if binary
    pub fn is_binary(&self) -> bool {
        matches!(self, HV::Binary(_))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_random_orthogonality() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let b = ContinuousHV::random(HDC_DIMENSION, 43);

        let sim = a.similarity(&b);

        // Random vectors should be nearly orthogonal
        assert!(
            sim.abs() < 0.1,
            "Expected near-zero similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_continuous_self_similarity() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);

        let sim = a.similarity(&a);

        // Self-similarity should be 1.0
        assert!((sim - 1.0).abs() < 0.001, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_continuous_binding_dissimilar() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let b = ContinuousHV::random(HDC_DIMENSION, 43);
        let bound = a.bind(&b);

        // Bound should be dissimilar to both inputs
        assert!(a.similarity(&bound).abs() < 0.2);
        assert!(b.similarity(&bound).abs() < 0.2);
    }

    #[test]
    fn test_continuous_bundling_similar() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let b = ContinuousHV::random(HDC_DIMENSION, 43);
        let bundled = ContinuousHV::bundle(&[&a, &b]);

        // Bundled should be similar to both inputs
        assert!(a.similarity(&bundled) > 0.4);
        assert!(b.similarity(&bundled) > 0.4);
    }

    #[test]
    fn test_binary_random_orthogonality() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);

        let sim = a.similarity(&b);

        // Random binary vectors: ~50% bits match
        assert!((sim - 0.5).abs() < 0.1, "Expected ~0.5, got {}", sim);
    }

    #[test]
    fn test_binary_binding_xor() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(43);
        let bound = a.bind(&b);

        // XOR with self gives zero
        let self_bound = a.bind(&a);
        assert_eq!(self_bound.popcount(), 0);

        // Bound should be different from inputs
        assert!(a.hamming_distance(&bound) > HDC_DIMENSION as u32 / 4);
    }

    #[test]
    fn test_binary_bundling_majority() {
        let a = BinaryHV::ones();
        let b = BinaryHV::ones();
        let c = BinaryHV::zero();

        let bundled = BinaryHV::bundle(&[a, b, c]);

        // Majority of ones -> all ones
        assert_eq!(bundled.popcount(), HDC_DIMENSION as u32);
    }

    #[test]
    fn test_continuous_to_binary_conversion() {
        let continuous = ContinuousHV::random(HDC_DIMENSION, 42);
        let binary = continuous.to_binary(0.0);
        let back = ContinuousHV::from_vec(binary.to_bipolar());

        // Should preserve structure (not exact values)
        let sim = continuous.similarity(&back);
        assert!(
            sim > 0.5,
            "Expected high similarity after conversion, got {}",
            sim
        );
    }

    #[test]
    fn test_unified_hv_similarity() {
        let a = HV::Continuous(ContinuousHV::random(HDC_DIMENSION, 42));
        let b = HV::Binary(BinaryHV::random(42));

        // Cross-type similarity should work
        let sim = a.similarity(&b);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_permutation_invertible() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let shift = 1000;

        let permuted = a.permute(shift);
        let restored = permuted.inverse_permute(shift);

        let sim = a.similarity(&restored);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Permutation should be invertible"
        );
    }

    #[test]
    fn test_bind_inverse_is_near_identity() {
        // For non-zero elements, A.bind(A.inverse()) should yield ~1.0
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let inv = a.inverse();
        let result = a.bind(&inv);

        // Each element a_i * (1/a_i) should be ~1.0 for non-zero a_i
        let near_one_count = result
            .values
            .iter()
            .filter(|&&v| (v - 1.0).abs() < 0.01)
            .count();

        // Most elements should be near 1.0 (only near-zero inputs deviate)
        assert!(
            near_one_count > HDC_DIMENSION / 2,
            "Expected most elements near 1.0, got {}/{}",
            near_one_count,
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_inverse_of_zero_is_zero() {
        let zero = ContinuousHV::zero(100);
        let inv = zero.inverse();
        assert!(
            inv.values.iter().all(|&v| v == 0.0),
            "Inverse of zero should be zero"
        );
    }

    #[test]
    fn test_dimension_constant() {
        assert_eq!(HDC_DIMENSION, 16_384);
        assert_eq!(BINARY_BYTES, 2048);
        assert_eq!(ContinuousHV::DEFAULT_DIM, HDC_DIMENSION);
    }

    #[test]
    fn test_bind_precision_high_approximates_standard_bind() {
        let a = ContinuousHV::random(HDC_DIMENSION, 100);
        let b = ContinuousHV::random(HDC_DIMENSION, 101);

        let standard = a.bind(&b);
        let precision_high = a.bind_precision(&b, 10.0); // sigmoid(10) ≈ 0.99995

        // High precision should closely approximate standard bind
        let sim = standard.similarity(&precision_high);
        assert!(
            sim > 0.999,
            "High precision should approximate standard bind, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_bind_precision_low_attenuates() {
        let a = ContinuousHV::random(HDC_DIMENSION, 200);
        let b = ContinuousHV::random(HDC_DIMENSION, 201);

        let standard = a.bind(&b);
        let precision_low = a.bind_precision(&b, -5.0); // sigmoid(-5) ≈ 0.0067

        // Low precision should heavily attenuate: norm much smaller than standard
        let norm_standard: f32 = standard.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_low: f32 = precision_low
            .values
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();

        assert!(
            norm_low < norm_standard * 0.1,
            "Low precision should attenuate, norm_low={} vs norm_standard={}",
            norm_low,
            norm_standard
        );
    }

    #[test]
    fn test_bind_precision_monotonic_gradient() {
        let a = ContinuousHV::random(HDC_DIMENSION, 300);
        let b = ContinuousHV::random(HDC_DIMENSION, 301);

        let precisions = [-5.0, -2.0, 0.0, 2.0, 5.0, 10.0];
        let norms: Vec<f32> = precisions
            .iter()
            .map(|&p| {
                let bound = a.bind_precision(&b, p);
                bound.values.iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();

        // Norms should be monotonically increasing with precision
        for i in 1..norms.len() {
            assert!(
                norms[i] >= norms[i - 1] - 1e-6,
                "Norms should increase with precision: norm[{}]={} < norm[{}]={}",
                i,
                norms[i],
                i - 1,
                norms[i - 1]
            );
        }
    }

    #[test]
    fn test_encode_weighted_matches_manual() {
        let bases = [
            ContinuousHV::random(HDC_DIMENSION, 100),
            ContinuousHV::random(HDC_DIMENSION, 101),
            ContinuousHV::random(HDC_DIMENSION, 102),
        ];
        let weights = [0.5f32, 0.8, 0.3];

        // Manual: scale each base, then bundle
        let scaled: Vec<ContinuousHV> = bases
            .iter()
            .zip(weights.iter())
            .map(|(b, &w)| {
                let mut hv = b.clone();
                for x in hv.values.iter_mut() {
                    *x *= w;
                }
                hv
            })
            .collect();
        let refs: Vec<&ContinuousHV> = scaled.iter().collect();
        let manual = ContinuousHV::bundle(&refs);

        // encode_weighted
        let encoded = ContinuousHV::encode_weighted(&bases, &weights);

        let sim = manual.similarity(&encoded);
        assert!(
            sim > 0.9999,
            "encode_weighted should match manual encode, got similarity {}",
            sim
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: dot() and orthogonal_set() tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_dot_product_self() {
        let a = ContinuousHV::random(HDC_DIMENSION, 42);
        let norm = a.norm();
        let dot = a.dot(&a);
        assert!(
            (dot - norm * norm).abs() < 1e-2,
            "dot(a, a) should equal norm²: dot={}, norm²={}",
            dot,
            norm * norm
        );
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let ortho = ContinuousHV::orthogonal_set(HDC_DIMENSION, 2, 42);
        let dot = ortho[0].dot(&ortho[1]);
        assert!(
            dot.abs() < 0.01,
            "Orthogonal vectors should have near-zero dot product, got {}",
            dot
        );
    }

    #[test]
    fn test_orthogonal_set_empty() {
        let result = ContinuousHV::orthogonal_set(HDC_DIMENSION, 0, 42);
        assert!(result.is_empty());
    }

    #[test]
    fn test_orthogonal_set_single() {
        let result = ContinuousHV::orthogonal_set(HDC_DIMENSION, 1, 42);
        assert_eq!(result.len(), 1);
        let norm = result[0].norm();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Single vector should be unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_orthogonal_set_pairwise_orthogonal() {
        let hvs = ContinuousHV::orthogonal_set(HDC_DIMENSION, 5, 42);
        assert_eq!(hvs.len(), 5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                let sim = hvs[i].similarity(&hvs[j]);
                assert!(
                    sim.abs() < 0.01,
                    "HVs {} and {} should be orthogonal, got similarity {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_set_unit_norm() {
        let hvs = ContinuousHV::orthogonal_set(HDC_DIMENSION, 5, 42);
        for (i, hv) in hvs.iter().enumerate() {
            let norm = hv.norm();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "HV {} should have unit norm, got {}",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_orthogonal_set_deterministic() {
        let a = ContinuousHV::orthogonal_set(HDC_DIMENSION, 3, 42);
        let b = ContinuousHV::orthogonal_set(HDC_DIMENSION, 3, 42);
        for i in 0..3 {
            let sim = a[i].similarity(&b[i]);
            assert!(
                (sim - 1.0).abs() < 0.001,
                "Same seed should produce same HVs, got similarity {} for index {}",
                sim,
                i
            );
        }
    }

    #[test]
    fn test_encode_weighted_deterministic() {
        let bases = [
            ContinuousHV::random(HDC_DIMENSION, 200),
            ContinuousHV::random(HDC_DIMENSION, 201),
        ];
        let weights = [0.7f32, 0.3];
        let a = ContinuousHV::encode_weighted(&bases, &weights);
        let b = ContinuousHV::encode_weighted(&bases, &weights);
        assert!(
            a.similarity(&b) > 0.9999,
            "encode_weighted should be deterministic"
        );
    }
}
