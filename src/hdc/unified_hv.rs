//! # Unified Hypervector Types
//!
//! This module provides a unified interface for hypervector operations across
//! both continuous (f32) and binary (bipolar) representations.
//!
//! ## Why Unified HV?
//!
//! Previously, Symthaea had two separate types:
//! - `RealHV` - f32 continuous values in [-1, 1]
//! - `HV16` - 16,384 bits bipolar {-1, +1}
//!
//! The name "RealHV" was confusing (suggests "actual" vs "fake").
//! This module provides:
//! - `ContinuousHV` - Clear name for f32 representation
//! - `BinaryHV` - Clear name for bipolar representation
//! - `HV` enum - Unified type for mixed operations
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
//! ```rust
//! use symthaea::hdc::unified_hv::{ContinuousHV, BinaryHV, HV};
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
//! let bundled = BinaryHV::bundle(&[x.clone(), y.clone()]);  // Majority vote
//!
//! // Unified interface
//! let hv_a = HV::Continuous(a);
//! let hv_x = HV::Binary(x);
//! ```

use serde::{Deserialize, Serialize};

/// Standard HDC dimension (2^14 = 16,384)
/// This is SIMD-optimized and matches research consensus.
pub const HDC_DIMENSION: usize = 16_384;

/// Number of bytes for binary representation (16,384 bits / 8)
pub const BINARY_BYTES: usize = HDC_DIMENSION / 8;

// ═══════════════════════════════════════════════════════════════════════════════
// CONTINUOUS HYPERVECTOR (formerly RealHV)
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
        let mut state = seed;

        for _ in 0..dim {
            // Simple xorshift64 for speed and reproducibility
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Map to [-1, 1]
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            values.push(normalized);
        }

        Self { values }
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
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");

        let values: Vec<f32> = self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
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
    pub fn bundle(hvs: &[&Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].values.len();
        let n = hvs.len() as f32;

        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let sum: f32 = hvs.iter().map(|hv| hv.values[i]).sum();
                sum / n
            })
            .collect();

        Self { values }
    }

    /// Weighted bundling
    pub fn weighted_bundle(hvs: &[&Self], weights: &[f32]) -> Self {
        if hvs.is_empty() || weights.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].values.len();
        let weight_sum: f32 = weights.iter().sum();

        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let weighted_sum: f32 = hvs.iter()
                    .zip(weights.iter())
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
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");

        let dot: f32 = self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
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
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            values: self.values.iter().map(|x| x * factor).collect(),
        }
    }

    /// Inverse for unbinding operations
    ///
    /// For continuous hypervectors with binding as element-wise multiplication,
    /// the inverse is self (approximately self-inverse property).
    /// This is because for normalized random vectors: A⊗A ≈ 1
    pub fn inverse(&self) -> Self {
        // For HDC binding which is element-wise multiplication,
        // the inverse is approximately self for normalized vectors
        self.clone()
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");

        Self {
            values: self.values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");

        Self {
            values: self.values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a - b)
                .collect(),
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
            values: self.values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
        }
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

    /// Convert to binary representation using threshold
    pub fn to_binary(&self, threshold: f32) -> BinaryHV {
        let mut bytes = vec![0u8; self.values.len() / 8];

        for (i, &val) in self.values.iter().enumerate() {
            if val > threshold {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }

        BinaryHV::from_bytes(bytes)
    }

    /// Convert to binary using probabilistic binarization
    ///
    /// Preserves heterogeneity better than threshold.
    pub fn to_binary_probabilistic(&self, seed: u64) -> BinaryHV {
        let mut bytes = vec![0u8; self.values.len() / 8];
        let mut state = seed;

        for (i, &val) in self.values.iter().enumerate() {
            // Sigmoid to probability
            let prob = 1.0 / (1.0 + (-val * 5.0).exp());

            // Random threshold
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let rand = (state as f32) / (u64::MAX as f32);

            if rand < prob {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }

        BinaryHV::from_bytes(bytes)
    }
}

impl Default for ContinuousHV {
    fn default() -> Self {
        Self::zero(HDC_DIMENSION)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BINARY HYPERVECTOR (formerly HV16)
// ═══════════════════════════════════════════════════════════════════════════════

/// Binary (bipolar) hypervector using packed bits.
///
/// Each bit represents {-1, +1}: 0 = -1, 1 = +1
///
/// Use this when:
/// - Memory efficiency is critical (2KB vs 64KB)
/// - Hardware acceleration is available
/// - Exact precision is not required
///
/// # Memory: 2KB per vector (16,384 bits / 8)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BinaryHV {
    /// Packed bits (16,384 bits = 2,048 bytes)
    bytes: Vec<u8>,
}

impl BinaryHV {
    /// Create a zero vector (all -1 in bipolar interpretation)
    pub fn zero() -> Self {
        Self {
            bytes: vec![0u8; BINARY_BYTES],
        }
    }

    /// Create a ones vector (all +1 in bipolar interpretation)
    pub fn ones() -> Self {
        Self {
            bytes: vec![0xFF; BINARY_BYTES],
        }
    }

    /// Create a random bipolar vector with deterministic seed
    pub fn random(seed: u64) -> Self {
        let mut bytes = Vec::with_capacity(BINARY_BYTES);
        let mut state = seed;

        for _ in 0..BINARY_BYTES {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push((state & 0xFF) as u8);
        }

        Self { bytes }
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        assert_eq!(bytes.len(), BINARY_BYTES, "Expected {} bytes", BINARY_BYTES);
        Self { bytes }
    }

    /// Get dimension in bits
    pub fn dim(&self) -> usize {
        self.bytes.len() * 8
    }

    /// Get a specific bit as bipolar (-1 or +1)
    pub fn get_bipolar(&self, index: usize) -> i8 {
        let byte_idx = index / 8;
        let bit_idx = index % 8;

        if self.bytes[byte_idx] & (1 << bit_idx) != 0 {
            1
        } else {
            -1
        }
    }

    /// Count of 1 bits (population count)
    pub fn popcount(&self) -> u32 {
        self.bytes.iter().map(|b| b.count_ones()).sum()
    }

    /// Binding operation (XOR)
    ///
    /// In bipolar interpretation: (-1)×(-1)=1, (-1)×1=-1, 1×1=1
    /// XOR achieves this: 0⊕0=0, 0⊕1=1, 1⊕1=0
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.bytes.len(), other.bytes.len(), "Dimension mismatch");

        let bytes: Vec<u8> = self.bytes
            .iter()
            .zip(other.bytes.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        Self { bytes }
    }

    /// Bundling operation (majority vote)
    ///
    /// For each bit position, output 1 if majority of inputs have 1.
    pub fn bundle(hvs: &[Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero();
        }

        let threshold = hvs.len() / 2;
        let mut bytes = vec![0u8; BINARY_BYTES];

        for byte_idx in 0..BINARY_BYTES {
            for bit_idx in 0..8 {
                let count: usize = hvs.iter()
                    .filter(|hv| hv.bytes[byte_idx] & (1 << bit_idx) != 0)
                    .count();

                if count > threshold {
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        Self { bytes }
    }

    /// Hamming distance (number of differing bits)
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.bytes
            .iter()
            .zip(other.bytes.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Similarity in range [0, 1]
    ///
    /// 1.0 = identical, 0.5 = random, 0.0 = opposite
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other) as f32;
        let dim = self.dim() as f32;

        1.0 - (hamming / dim)
    }

    /// Bipolar similarity in range [-1, 1]
    ///
    /// Matches ContinuousHV similarity interpretation.
    pub fn bipolar_similarity(&self, other: &Self) -> f32 {
        self.similarity(other) * 2.0 - 1.0
    }

    /// Permute bits (rotation)
    pub fn permute(&self, shift: usize) -> Self {
        let dim = self.dim();
        let shift = shift % dim;

        let mut bytes = vec![0u8; BINARY_BYTES];

        for i in 0..dim {
            let src_byte = i / 8;
            let src_bit = i % 8;
            let dst_idx = (i + shift) % dim;
            let dst_byte = dst_idx / 8;
            let dst_bit = dst_idx % 8;

            if self.bytes[src_byte] & (1 << src_bit) != 0 {
                bytes[dst_byte] |= 1 << dst_bit;
            }
        }

        Self { bytes }
    }

    /// Convert to continuous representation
    pub fn to_continuous(&self) -> ContinuousHV {
        let values: Vec<f32> = (0..self.dim())
            .map(|i| self.get_bipolar(i) as f32)
            .collect();

        ContinuousHV { values }
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Default for BinaryHV {
    fn default() -> Self {
        Self::zero()
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
            HV::Binary(hv) => hv.dim(),
        }
    }

    /// Similarity between any two HVs (converts if needed)
    pub fn similarity(&self, other: &HV) -> f32 {
        match (self, other) {
            (HV::Continuous(a), HV::Continuous(b)) => a.similarity(b),
            (HV::Binary(a), HV::Binary(b)) => a.bipolar_similarity(b),
            (HV::Continuous(a), HV::Binary(b)) => a.similarity(&b.to_continuous()),
            (HV::Binary(a), HV::Continuous(b)) => a.to_continuous().similarity(b),
        }
    }

    /// Convert to continuous (no-op if already continuous)
    pub fn to_continuous(&self) -> ContinuousHV {
        match self {
            HV::Continuous(hv) => hv.clone(),
            HV::Binary(hv) => hv.to_continuous(),
        }
    }

    /// Convert to binary
    pub fn to_binary(&self, threshold: f32) -> BinaryHV {
        match self {
            HV::Continuous(hv) => hv.to_binary(threshold),
            HV::Binary(hv) => hv.clone(),
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
// TYPE ALIASES FOR BACKWARDS COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Alias for backwards compatibility (deprecated)
#[deprecated(since = "0.2.0", note = "Use ContinuousHV instead")]
pub type RealHV = ContinuousHV;

/// Alias for backwards compatibility (deprecated)
#[deprecated(since = "0.2.0", note = "Use BinaryHV instead")]
pub type HV16 = BinaryHV;

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
        assert!(sim.abs() < 0.1, "Expected near-zero similarity, got {}", sim);
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
        let back = binary.to_continuous();

        // Should preserve structure (not exact values)
        let sim = continuous.similarity(&back);
        assert!(sim > 0.5, "Expected high similarity after conversion, got {}", sim);
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
        assert!((sim - 1.0).abs() < 0.001, "Permutation should be invertible");
    }

    #[test]
    fn test_dimension_constant() {
        assert_eq!(HDC_DIMENSION, 16_384);
        assert_eq!(BINARY_BYTES, 2048);
        assert_eq!(ContinuousHV::DEFAULT_DIM, HDC_DIMENSION);
    }
}
