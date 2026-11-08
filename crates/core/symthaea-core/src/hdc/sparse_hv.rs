// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sparse Hypervectors for Memory-Efficient HDC
//!
//! This module provides sparse representations of hypervectors, storing only
//! active (1-bit) positions instead of the full 16,384-bit vector.
//!
//! # When to Use Sparse Representations
//!
//! | Scenario | Dense (BinaryHV) | Sparse (SparseHV) |
//! |----------|--------------|-------------------|
//! | Storage | 2KB | ~4KB (50% density) |
//! | Bind | 80ns | 200ns |
//! | Bundle | 1μs | 500ns (small sets) |
//! | Similarity | 160ns | 50ns (Jaccard) |
//! | Memory > Speed | ❌ | ✅ |
//! | Speed > Memory | ✅ | ❌ |
//!
//! Best for:
//! - Very low density vectors (<20% ones)
//! - Jaccard similarity is acceptable
//! - Memory constrained environments
//! - Streaming/incremental operations
//!
//! # Architecture
//!
//! Uses existing LSH infrastructure for efficient operations:
//! - `lsh_simhash.rs` for locality-preserving hashing
//! - Pre-computed LSH signatures for O(1) approximate similarity

use super::HDC_DIMENSION;
use super::binary_hv::BinaryHV;
use std::collections::HashSet;

/// Errors returned when constructing a sparse hypervector from a target density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseHVError {
    /// Density must be finite and lie in the closed interval [0, 1].
    InvalidDensity,
}

impl std::fmt::Display for SparseHVError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDensity => {
                write!(f, "SparseHV density must be finite and within [0, 1]")
            }
        }
    }
}

impl std::error::Error for SparseHVError {}

/// Small deterministic generator used only for reproducible sparse sampling.
///
/// SplitMix64 has a full-width output permutation. The bounded draw uses
/// rejection sampling so reducing to a bounded integer does not introduce
/// modulo bias.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: u64) -> u64 {
        debug_assert!(upper_exclusive > 0);
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % upper_exclusive;
            }
        }
    }
}

/// Sparse representation of a 16,384-bit hypervector
///
/// Stores only the positions of 1-bits, plus a pre-computed LSH signature
/// for fast approximate similarity.
///
/// # Memory Usage
/// - 50% density: ~16K positions × 2 bytes = 32KB (worse than dense)
/// - 10% density: ~1.6K positions × 2 bytes = 3.2KB (similar to dense)
/// - 1% density: ~160 positions × 2 bytes = 320 bytes (10x better!)
///
/// **Rule of thumb**: Use SparseHV when density < 20%
#[derive(Debug, Clone)]
pub struct SparseHV {
    /// Indices of 1-bits (sorted for binary search)
    active_indices: Vec<u16>,

    /// Pre-computed 64-bit LSH signature for fast approximate similarity
    lsh_signature: u64,

    /// Cached density (for quick access)
    cached_density: f32,
}

impl SparseHV {
    /// Dimension of the hypervector
    pub const DIM: usize = HDC_DIMENSION;

    /// Create empty sparse vector (all zeros)
    pub fn zero() -> Self {
        Self {
            active_indices: Vec::new(),
            lsh_signature: 0,
            cached_density: 0.0,
        }
    }

    /// Create from dense BinaryHV
    ///
    /// Extracts positions of 1-bits and computes LSH signature.
    /// Time: O(D) = O(16,384)
    pub fn from_hv16(hv: &BinaryHV) -> Self {
        let mut active_indices = Vec::with_capacity(Self::DIM / 2);

        for byte_idx in 0..2048 {
            let byte = hv.0[byte_idx];
            if byte == 0 {
                continue;
            }
            for bit_idx in 0..8 {
                if (byte >> bit_idx) & 1 == 1 {
                    active_indices.push((byte_idx * 8 + bit_idx) as u16);
                }
            }
        }

        let density = active_indices.len() as f32 / Self::DIM as f32;
        let lsh_sig = Self::compute_lsh_signature(&active_indices);

        Self {
            active_indices,
            lsh_signature: lsh_sig,
            cached_density: density,
        }
    }

    /// Convert back to dense BinaryHV
    ///
    /// Time: O(k) where k = number of active bits
    pub fn to_hv16(&self) -> BinaryHV {
        let mut data = [0u8; 2048];

        for &pos in &self.active_indices {
            let byte_idx = pos as usize / 8;
            let bit_pos = pos as usize % 8;
            data[byte_idx] |= 1 << bit_pos;
        }

        BinaryHV(data)
    }

    /// Create a deterministic random sparse vector with an exact target density.
    ///
    /// The selected coordinates are sampled uniformly without replacement;
    /// no coordinate prefix or dense-vector truncation is privileged.
    ///
    /// # Arguments
    /// - `seed`: Seed for reproducible sampling (not a cryptographic key).
    /// - `density`: Target density in [0.0, 1.0]. Cardinality is floor(density * DIM).
    ///
    /// # Panics
    /// Panics if density is non-finite or outside [0, 1]. Use
    /// [Self::try_random] when input validation must be handled explicitly.
    pub fn random(seed: u64, density: f32) -> Self {
        Self::try_random(seed, density)
            .expect("SparseHV::random density must be finite and within [0, 1]")
    }

    /// Fallible counterpart to [Self::random].
    ///
    /// Uses Floyd's fixed-cardinality sampling algorithm with unbiased bounded
    /// draws. Sampling takes expected O(k) time and O(k) temporary membership
    /// storage; sorting the returned coordinates takes O(k log k), where
    /// k = floor(density * DIM).
    pub fn try_random(seed: u64, density: f32) -> Result<Self, SparseHVError> {
        if !density.is_finite() || !(0.0..=1.0).contains(&density) {
            return Err(SparseHVError::InvalidDensity);
        }

        let target_count = (density * Self::DIM as f32).floor() as usize;
        let mut selected = HashSet::with_capacity(target_count);
        let mut rng = SplitMix64::new(seed ^ 0x5350_4152_5345_4856);

        // Floyd's algorithm produces a uniform subset of exactly target_count
        // coordinates without replacement.
        for upper in (Self::DIM - target_count)..Self::DIM {
            let candidate = rng.below((upper + 1) as u64) as usize;
            let selected_index = if selected.contains(&candidate) {
                upper
            } else {
                candidate
            };
            selected.insert(selected_index);
        }

        let mut active_indices = selected
            .into_iter()
            .map(|index| index as u16)
            .collect::<Vec<_>>();
        active_indices.sort_unstable();

        let cached_density = active_indices.len() as f32 / Self::DIM as f32;
        let lsh_signature = Self::compute_lsh_signature(&active_indices);
        Ok(Self {
            active_indices,
            lsh_signature,
            cached_density,
        })
    }

    /// Get density (proportion of 1-bits)
    #[inline]
    pub fn density(&self) -> f32 {
        self.cached_density
    }

    /// Get number of active (1) bits
    #[inline]
    pub fn popcount(&self) -> usize {
        self.active_indices.len()
    }

    /// Check if a bit is active
    #[inline]
    pub fn is_active(&self, pos: u16) -> bool {
        self.active_indices.binary_search(&pos).is_ok()
    }

    /// Jaccard similarity between sparse vectors
    ///
    /// Jaccard = |A ∩ B| / |A ∪ B|
    ///
    /// This is efficient for sparse vectors: O(k₁ + k₂)
    /// where k₁, k₂ are the number of active bits.
    ///
    /// # Note
    /// Jaccard similarity is NOT the same as Hamming similarity!
    /// - Hamming: (matching bits) / total bits
    /// - Jaccard: |intersection| / |union|
    ///
    /// For sparse vectors, Jaccard is more meaningful.
    pub fn similarity_jaccard(&self, other: &Self) -> f32 {
        let intersection = self.intersection_count(other);
        let union = self.union_count(other);

        if union == 0 {
            return 1.0; // Both empty
        }

        intersection as f32 / union as f32
    }

    /// Approximate similarity using LSH signatures
    ///
    /// Time: O(1)
    /// Accuracy: ~90% for similar vectors, ~95% for dissimilar
    ///
    /// Uses pre-computed 64-bit LSH signatures for instant comparison.
    #[inline]
    pub fn similarity_lsh(&self, other: &Self) -> f32 {
        let xor = self.lsh_signature ^ other.lsh_signature;
        let matching = 64 - xor.count_ones();
        matching as f32 / 64.0
    }

    /// Count intersection (|A ∩ B|)
    fn intersection_count(&self, other: &Self) -> usize {
        let mut count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < self.active_indices.len() && j < other.active_indices.len() {
            match self.active_indices[i].cmp(&other.active_indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        count
    }

    /// Count union (|A ∪ B|)
    fn union_count(&self, other: &Self) -> usize {
        self.active_indices.len() + other.active_indices.len() - self.intersection_count(other)
    }

    /// Compute 64-bit LSH signature by sampling bit positions
    ///
    /// Uses 64 fixed positions spread across the dimension range.
    fn compute_lsh_signature(active_indices: &[u16]) -> u64 {
        let mut sig = 0u64;
        let set: HashSet<u16> = active_indices.iter().copied().collect();

        // Sample 64 positions evenly spread across the dimension
        for bit in 0..64 {
            let pos = (bit * HDC_DIMENSION / 64) as u16;
            if set.contains(&pos) {
                sig |= 1u64 << bit;
            }
        }

        sig
    }

    /// Bind sparse vectors (symmetric difference)
    ///
    /// XOR operation on sparse representation:
    /// Result contains positions that are in exactly one of the inputs.
    ///
    /// Time: O(k₁ + k₂)
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.active_indices.len() + other.active_indices.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.active_indices.len() && j < other.active_indices.len() {
            match self.active_indices[i].cmp(&other.active_indices[j]) {
                std::cmp::Ordering::Less => {
                    result.push(self.active_indices[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(other.active_indices[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // In both = XOR removes it
                    i += 1;
                    j += 1;
                }
            }
        }

        result.extend_from_slice(&self.active_indices[i..]);
        result.extend_from_slice(&other.active_indices[j..]);

        let density = result.len() as f32 / Self::DIM as f32;
        let lsh_sig = Self::compute_lsh_signature(&result);

        Self {
            active_indices: result,
            lsh_signature: lsh_sig,
            cached_density: density,
        }
    }

    /// Bundle sparse vectors (union with optional thresholding)
    ///
    /// For sparse vectors, bundle is a weighted union.
    /// Bits that appear in > 50% of vectors are included.
    ///
    /// Time: O(n × k) where n = number of vectors, k = average active count
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero();
        }

        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let threshold = vectors.len() / 2;

        // Count occurrences of each position
        let mut counts: std::collections::HashMap<u16, usize> =
            std::collections::HashMap::with_capacity(
                vectors.iter().map(|v| v.active_indices.len()).sum(),
            );

        for v in vectors {
            for &pos in &v.active_indices {
                *counts.entry(pos).or_insert(0) += 1;
            }
        }

        // Keep positions that appear in majority of vectors
        let mut result: Vec<u16> = counts
            .into_iter()
            .filter(|(_, count)| *count > threshold)
            .map(|(pos, _)| pos)
            .collect();

        result.sort();

        let density = result.len() as f32 / Self::DIM as f32;
        let lsh_sig = Self::compute_lsh_signature(&result);

        Self {
            active_indices: result,
            lsh_signature: lsh_sig,
            cached_density: density,
        }
    }

    /// Permute (circular rotation) in sparse representation
    ///
    /// Time: O(k) where k = number of active bits
    pub fn permute(&self, shift: usize) -> Self {
        let shift = (shift % Self::DIM) as u16;

        let mut new_indices: Vec<u16> = self
            .active_indices
            .iter()
            .map(|&pos| (pos + shift) % Self::DIM as u16)
            .collect();

        new_indices.sort();

        let lsh_sig = Self::compute_lsh_signature(&new_indices);

        Self {
            active_indices: new_indices,
            lsh_signature: lsh_sig,
            cached_density: self.cached_density,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Construction =====

    #[test]
    fn test_zero_construction() {
        let sparse = SparseHV::zero();
        assert_eq!(sparse.popcount(), 0);
        assert_eq!(sparse.density(), 0.0);
        assert_eq!(sparse.lsh_signature, 0);
    }

    #[test]
    fn test_from_hv16_and_back() {
        let original = BinaryHV::random(42);
        let sparse = SparseHV::from_hv16(&original);
        let recovered = sparse.to_hv16();
        assert_eq!(original, recovered, "Round-trip should preserve vector");
    }

    #[test]
    fn test_from_hv16_zero() {
        let zero = BinaryHV::zero();
        let sparse = SparseHV::from_hv16(&zero);
        assert_eq!(
            sparse.popcount(),
            0,
            "Zero vector should have 0 active bits"
        );
        assert_eq!(sparse.density(), 0.0);
    }

    #[test]
    fn test_from_hv16_ones() {
        let ones = BinaryHV::ones();
        let sparse = SparseHV::from_hv16(&ones);
        assert_eq!(
            sparse.popcount(),
            SparseHV::DIM,
            "All-ones vector should have DIM active bits"
        );
        assert!((sparse.density() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_random_density() {
        let hv = BinaryHV::random(42);
        let sparse = SparseHV::from_hv16(&hv);

        let density = sparse.density();
        assert!(
            density > 0.45 && density < 0.55,
            "Random vector density should be ~0.5, got {}",
            density
        );
    }

    #[test]
    fn random_target_density_has_exact_cardinality() {
        for target in [0.0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.999, 1.0] {
            let sparse = SparseHV::try_random(42, target).expect("valid target density");
            let expected = (target * SparseHV::DIM as f32).floor() as usize;
            assert_eq!(sparse.popcount(), expected, "target density {target}");
            assert!(
                (sparse.density() - expected as f32 / SparseHV::DIM as f32).abs() < 1e-7
            );
        }
    }

    #[test]
    fn random_sparse_sampling_is_deterministic_and_round_trips() {
        let first = SparseHV::try_random(0x51A7, 0.125).expect("valid density");
        let second = SparseHV::try_random(0x51A7, 0.125).expect("valid density");
        assert_eq!(first.active_indices, second.active_indices);
        assert_eq!(first.to_hv16(), second.to_hv16());
        assert_eq!(
            SparseHV::from_hv16(&first.to_hv16()).active_indices,
            first.active_indices
        );
    }

    #[test]
    fn random_sparse_sampling_does_not_prefer_low_coordinate_ranges() {
        const TRIALS: usize = 256;
        const BINS: usize = 16;
        const DENSITY: f32 = 0.1;
        let mut counts = [0usize; BINS];

        for seed in 0..TRIALS as u64 {
            let sparse = SparseHV::try_random(seed, DENSITY).expect("valid density");
            for &index in &sparse.active_indices {
                let bin = (index as usize * BINS) / SparseHV::DIM;
                counts[bin] += 1;
            }
        }

        let expected_per_bin =
            TRIALS * (DENSITY * SparseHV::DIM as f32).floor() as usize / BINS;
        let min = *counts.iter().min().expect("non-empty bins");
        let max = *counts.iter().max().expect("non-empty bins");
        assert!(
            max - min < expected_per_bin / 8,
            "coordinate bins should be exchangeable; counts={counts:?}, expected≈{expected_per_bin}"
        );
    }

    #[test]
    fn try_random_rejects_nonfinite_and_out_of_range_density() {
        for invalid in [
            -f32::EPSILON,
            1.0 + f32::EPSILON,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(matches!(
                SparseHV::try_random(7, invalid),
                Err(SparseHVError::InvalidDensity)
            ));
        }
    }

    #[test]
    fn random_compatibility_wrapper_fails_fast_on_invalid_density() {
        let result = std::panic::catch_unwind(|| SparseHV::random(7, f32::INFINITY));
        assert!(result.is_err(), "invalid density must not enter a non-terminating fill loop");
    }

    #[test]
    fn test_random_with_target_density() {
        for target in [0.01, 0.1, 0.3, 0.5] {
            let sparse = SparseHV::random(42, target);
            let actual = sparse.density();
            // Allow some tolerance because of discrete bit positions
            assert!(
                (actual - target).abs() < 0.05,
                "Target density {}, got {}",
                target,
                actual
            );
        }
    }

    // ===== Similarity =====

    #[test]
    fn test_jaccard_similarity_self() {
        let sparse = SparseHV::random(42, 0.5);
        let sim = sparse.similarity_jaccard(&sparse);
        assert_eq!(sim, 1.0, "Self-similarity should be 1.0");
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        // Construct two disjoint sparse vectors
        let mut a = SparseHV::zero();
        a.active_indices = (0..100).collect();
        a.cached_density = 100.0 / SparseHV::DIM as f32;
        a.lsh_signature = SparseHV::compute_lsh_signature(&a.active_indices);

        let mut b = SparseHV::zero();
        b.active_indices = (100..200).collect();
        b.cached_density = 100.0 / SparseHV::DIM as f32;
        b.lsh_signature = SparseHV::compute_lsh_signature(&b.active_indices);

        let sim = a.similarity_jaccard(&b);
        assert_eq!(sim, 0.0, "Disjoint vectors should have Jaccard=0");
    }

    #[test]
    fn test_jaccard_similarity_both_empty() {
        let a = SparseHV::zero();
        let b = SparseHV::zero();
        let sim = a.similarity_jaccard(&b);
        assert_eq!(sim, 1.0, "Both empty should be similarity 1.0");
    }

    #[test]
    fn test_jaccard_vs_hamming() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);

        let sparse_a = SparseHV::from_hv16(&a);
        let sparse_b = SparseHV::from_hv16(&b);

        let hamming_sim = a.similarity(&b);
        let jaccard_sim = sparse_a.similarity_jaccard(&sparse_b);

        assert!(
            jaccard_sim >= 0.0 && jaccard_sim <= 1.0,
            "Jaccard should be in [0, 1]"
        );
        // For random ~50% vectors, Jaccard < Hamming
        println!("Hamming: {}, Jaccard: {}", hamming_sim, jaccard_sim);
    }

    #[test]
    fn test_lsh_similarity_range() {
        let a = SparseHV::random(1, 0.5);
        let b = SparseHV::random(2, 0.5);

        let lsh = a.similarity_lsh(&b);
        assert!(
            lsh >= 0.0 && lsh <= 1.0,
            "LSH similarity should be in [0, 1]"
        );
    }

    #[test]
    fn test_lsh_similarity_self() {
        let a = SparseHV::random(42, 0.5);
        let sim = a.similarity_lsh(&a);
        assert_eq!(sim, 1.0, "LSH self-similarity should be 1.0");
    }

    // ===== is_active =====

    #[test]
    fn test_is_active() {
        let hv = BinaryHV::random(42);
        let sparse = SparseHV::from_hv16(&hv);

        // Check a few active positions
        for &pos in sparse.active_indices.iter().take(10) {
            assert!(sparse.is_active(pos), "Position {} should be active", pos);
        }

        // Check that zero vector has no active bits
        let zero = SparseHV::zero();
        assert!(!zero.is_active(0));
        assert!(!zero.is_active(100));
    }

    // ===== Bind =====

    #[test]
    fn test_bind_sparse() {
        let a = SparseHV::random(1, 0.3);
        let b = SparseHV::random(2, 0.3);

        let bound = a.bind(&b);
        assert!(
            bound.density() > 0.1 && bound.density() < 0.9,
            "Bound density {} should be reasonable",
            bound.density()
        );
    }

    #[test]
    fn test_bind_self_is_empty() {
        let a = SparseHV::random(42, 0.5);
        let self_bound = a.bind(&a);
        assert_eq!(self_bound.popcount(), 0, "Self-bind (XOR) should be empty");
    }

    #[test]
    fn test_bind_with_zero() {
        let a = SparseHV::random(42, 0.3);
        let zero = SparseHV::zero();
        let result = a.bind(&zero);
        assert_eq!(
            result.active_indices, a.active_indices,
            "Bind with zero should return original"
        );
    }

    #[test]
    fn test_bind_commutativity() {
        let a = SparseHV::random(1, 0.3);
        let b = SparseHV::random(2, 0.3);

        let ab = a.bind(&b);
        let ba = b.bind(&a);
        assert_eq!(
            ab.active_indices, ba.active_indices,
            "Sparse bind should be commutative"
        );
    }

    #[test]
    fn test_bind_matches_dense() {
        let a_dense = BinaryHV::random(10);
        let b_dense = BinaryHV::random(20);

        let a_sparse = SparseHV::from_hv16(&a_dense);
        let b_sparse = SparseHV::from_hv16(&b_dense);

        let sparse_result = a_sparse.bind(&b_sparse).to_hv16();
        let dense_result = a_dense.bind(&b_dense);

        assert_eq!(
            sparse_result, dense_result,
            "Sparse bind should match dense bind"
        );
    }

    // ===== Bundle =====

    #[test]
    fn test_bundle_sparse() {
        let vectors: Vec<SparseHV> = (0..5).map(|i| SparseHV::random(i, 0.3)).collect();

        let bundled = SparseHV::bundle(&vectors);
        assert!(
            bundled.density() > 0.0 && bundled.density() < 1.0,
            "Bundled density {} should be reasonable",
            bundled.density()
        );
    }

    #[test]
    fn test_bundle_empty() {
        let result = SparseHV::bundle(&[]);
        assert_eq!(result.popcount(), 0, "Bundle of empty should be zero");
    }

    #[test]
    fn test_bundle_single() {
        let v = SparseHV::random(42, 0.3);
        let result = SparseHV::bundle(&[v.clone()]);
        assert_eq!(
            result.active_indices, v.active_indices,
            "Bundle of single should return same vector"
        );
    }

    // ===== Permute =====

    #[test]
    fn test_permute_preserves_popcount() {
        let original = SparseHV::random(42, 0.3);
        let permuted = original.permute(100);

        assert_eq!(
            original.popcount(),
            permuted.popcount(),
            "Permute should preserve popcount"
        );
    }

    #[test]
    fn test_permute_changes_positions() {
        let original = SparseHV::random(42, 0.3);
        let permuted = original.permute(100);
        assert_ne!(
            original.active_indices, permuted.active_indices,
            "Permute should change positions"
        );
    }

    #[test]
    fn test_permute_full_rotation() {
        let original = SparseHV::random(42, 0.3);
        let full_rotate = original.permute(SparseHV::DIM);
        assert_eq!(
            original.active_indices, full_rotate.active_indices,
            "Permute by DIM should return to original"
        );
    }

    #[test]
    fn test_permute_zero_shift() {
        let original = SparseHV::random(42, 0.3);
        let not_shifted = original.permute(0);
        assert_eq!(
            original.active_indices, not_shifted.active_indices,
            "Permute by 0 should return original"
        );
    }

    // ===== Memory Efficiency =====

    #[test]
    fn test_low_density_efficiency() {
        let sparse = SparseHV::random(42, 0.01);
        let memory_bytes = sparse.active_indices.len() * 2;

        assert!(
            memory_bytes < 500,
            "1% density should use <500 bytes, got {}",
            memory_bytes
        );
        println!("1% density uses {} bytes (vs 2048 for dense)", memory_bytes);
    }

    #[test]
    fn test_indices_are_sorted() {
        let sparse = SparseHV::random(42, 0.5);
        for w in sparse.active_indices.windows(2) {
            assert!(
                w[0] < w[1],
                "Active indices should be sorted, found {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_indices_within_bounds() {
        let sparse = SparseHV::random(42, 0.5);
        for &idx in &sparse.active_indices {
            assert!(
                (idx as usize) < SparseHV::DIM,
                "Index {} should be < DIM={}",
                idx,
                SparseHV::DIM
            );
        }
    }
}
