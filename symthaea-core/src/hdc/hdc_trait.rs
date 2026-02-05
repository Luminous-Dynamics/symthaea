//! Unified Hyperdimensional Computing Trait
//!
//! This module provides a common interface for all hypervector implementations:
//! - `HV16`: Binary hypervectors (byte-packed)
//! - `SimdHV16`: SIMD-optimized binary hypervectors (u64-packed)
//! - `RealHV`: Real-valued hypervectors (f32)
//!
//! # Example
//! ```ignore
//! use symthaea::hdc::{HyperdimensionalVector, HV16, SimdHV16};
//!
//! fn process_any_hv<H: HyperdimensionalVector>(hv: &H) -> f32 {
//!     hv.density()
//! }
//!
//! let binary = HV16::random(42);
//! let simd = SimdHV16::random(42);
//!
//! println!("Binary density: {}", process_any_hv(&binary));
//! println!("SIMD density: {}", process_any_hv(&simd));
//! ```

use super::HDC_DIMENSION;

/// Core trait for hyperdimensional vector operations
///
/// This trait abstracts over different HV implementations, allowing
/// generic algorithms to work with any hypervector type.
pub trait HyperdimensionalVector: Clone + Sized {
    /// Dimension of the hypervector (typically 16,384)
    const DIMENSION: usize = HDC_DIMENSION;

    /// Create a random hypervector from a seed (deterministic)
    fn random(seed: u64) -> Self;

    /// Create a zero vector (all bits/values = 0)
    fn zero() -> Self;

    /// Binding operation (XOR for binary, element-wise multiply for real)
    ///
    /// Binding creates associations and preserves similarity relationships
    fn bind(&self, other: &Self) -> Self;

    /// Similarity measure (normalized to [0, 1])
    ///
    /// - Binary: Hamming similarity (matching bits / total bits)
    /// - Real: Cosine similarity mapped to [0, 1]
    fn similarity(&self, other: &Self) -> f32;

    /// Population count / density
    ///
    /// Returns the fraction of "active" elements (set bits for binary, non-zero for real)
    fn density(&self) -> f32;

    /// Hamming distance (number of differing positions)
    fn hamming_distance(&self, other: &Self) -> u32;
}

/// Extended trait for hypervectors that support bundling (majority vote / averaging)
pub trait Bundleable: HyperdimensionalVector {
    /// Bundle multiple vectors into one using majority vote (binary) or averaging (real)
    ///
    /// The result is similar to all inputs - this is the "superposition" operation
    fn bundle(vectors: &[Self]) -> Self;
}

/// Extended trait for hypervectors that support permutation (rotation)
pub trait Permutable: HyperdimensionalVector {
    /// Circular rotation by `shift` positions
    ///
    /// Useful for encoding sequences: `encode(a, b, c) = a ⊕ ρ(b) ⊕ ρ²(c)`
    fn permute(&self, shift: usize) -> Self;
}

/// Implement the traits for SimdHV16
impl HyperdimensionalVector for super::simd_hv16::SimdHV16 {
    fn random(seed: u64) -> Self {
        Self::random(seed)
    }

    fn zero() -> Self {
        Self::zero()
    }

    fn bind(&self, other: &Self) -> Self {
        Self::bind(self, other)
    }

    fn similarity(&self, other: &Self) -> f32 {
        Self::similarity(self, other)
    }

    fn density(&self) -> f32 {
        self.popcount() as f32 / Self::DIM as f32
    }

    fn hamming_distance(&self, other: &Self) -> u32 {
        Self::hamming_distance(self, other)
    }
}

impl Bundleable for super::simd_hv16::SimdHV16 {
    fn bundle(vectors: &[Self]) -> Self {
        Self::bundle(vectors)
    }
}

impl Permutable for super::simd_hv16::SimdHV16 {
    fn permute(&self, shift: usize) -> Self {
        Self::permute(self, shift)
    }
}

/// Implement the traits for HV16
impl HyperdimensionalVector for super::binary_hv::HV16 {
    fn random(seed: u64) -> Self {
        Self::random(seed)
    }

    fn zero() -> Self {
        Self::zero()
    }

    fn bind(&self, other: &Self) -> Self {
        Self::bind(self, other)
    }

    fn similarity(&self, other: &Self) -> f32 {
        Self::similarity(self, other)
    }

    fn density(&self) -> f32 {
        self.popcount() as f32 / Self::DIM as f32
    }

    fn hamming_distance(&self, other: &Self) -> u32 {
        Self::hamming_distance(self, other)
    }
}

impl Bundleable for super::binary_hv::HV16 {
    fn bundle(vectors: &[Self]) -> Self {
        Self::bundle(vectors)
    }
}

impl Permutable for super::binary_hv::HV16 {
    fn permute(&self, shift: usize) -> Self {
        Self::permute(self, shift)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERIC HDC ALGORITHMS
// ═══════════════════════════════════════════════════════════════════════════

/// Encode a sequence of items into a single hypervector
///
/// Uses permutation to encode position: `encode([a, b, c]) = a ⊕ ρ(b) ⊕ ρ²(c)`
pub fn encode_sequence<H: HyperdimensionalVector + Permutable>(items: &[H]) -> H {
    if items.is_empty() {
        return H::zero();
    }

    let mut result = items[0].clone();
    for (i, item) in items.iter().enumerate().skip(1) {
        let permuted = item.permute(i);
        result = result.bind(&permuted);
    }
    result
}

/// Find the most similar vector in a set
pub fn nearest_neighbor<H: HyperdimensionalVector>(query: &H, candidates: &[H]) -> Option<(usize, f32)> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, query.similarity(c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Find all vectors above a similarity threshold
pub fn similar_vectors<H: HyperdimensionalVector>(
    query: &H,
    candidates: &[H],
    threshold: f32,
) -> Vec<(usize, f32)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let sim = query.similarity(c);
            if sim >= threshold {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::{binary_hv::HV16, simd_hv16::SimdHV16};

    #[test]
    fn test_generic_algorithm_with_hv16() {
        let a = HV16::random(1);
        let b = HV16::random(2);
        let c = HV16::random(3);

        let encoded = encode_sequence(&[a.clone(), b.clone(), c.clone()]);

        // Encoded should be a valid HV16
        assert!(encoded.density() > 0.4 && encoded.density() < 0.6);
    }

    #[test]
    fn test_generic_algorithm_with_simd() {
        let a = SimdHV16::random(1);
        let b = SimdHV16::random(2);
        let c = SimdHV16::random(3);

        let encoded = encode_sequence(&[a.clone(), b.clone(), c.clone()]);

        // Encoded should be a valid SimdHV16
        assert!(encoded.density() > 0.4 && encoded.density() < 0.6);
    }

    #[test]
    fn test_nearest_neighbor() {
        let query = SimdHV16::random(0);
        let candidates: Vec<_> = (1..10).map(|i| SimdHV16::random(i)).collect();

        let result = nearest_neighbor(&query, &candidates);
        assert!(result.is_some());

        let (idx, sim) = result.unwrap();
        assert!(idx < candidates.len());
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_trait_consistency() {
        // Same seed should give same random vector properties
        let hv = HV16::random(42);
        let simd = SimdHV16::random(42);

        // Both should have ~0.5 density (statistical property)
        let hv_density = <HV16 as HyperdimensionalVector>::density(&hv);
        let simd_density = <SimdHV16 as HyperdimensionalVector>::density(&simd);

        assert!((hv_density - 0.5).abs() < 0.05);
        assert!((simd_density - 0.5).abs() < 0.05);
    }
}
