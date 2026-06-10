// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hyperdimensional Computing (HDC) primitives
//!
//! This module implements 2048-bit binary hypervectors with:
//! - BLAKE3-based random projection for deterministic encoding
//! - XOR binding for associative memory
//! - Majority voting for bundling/superposition

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Hypervector dimension (bits)
pub const HDC_DIM: usize = 2048;

/// Number of u128 words to store the hypervector
pub const HDC_WORDS: usize = HDC_DIM / 128;

/// A 2048-bit binary hypervector stored as 16 x u128
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HV16 {
    /// The 16 u128 words comprising the 2048-bit vector
    pub words: [u128; HDC_WORDS],
}

impl Default for HV16 {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Debug for HV16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ones = self.popcount();
        write!(f, "HV16[popcount={ones}/{HDC_DIM}]")
    }
}

impl HV16 {
    /// Create a zero hypervector
    pub fn zero() -> Self {
        Self {
            words: [0u128; HDC_WORDS],
        }
    }

    /// Create a random hypervector from a seed string using BLAKE3
    pub fn random(seed: &str) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(seed.as_bytes());
        let mut output = [0u8; HDC_WORDS * 16];
        hasher.finalize_xof().fill(&mut output);

        let mut words = [0u128; HDC_WORDS];
        for (i, chunk) in output.chunks_exact(16).enumerate() {
            words[i] = u128::from_le_bytes(chunk.try_into().unwrap_or([0u8; 16]));
        }

        Self { words }
    }

    /// Create a hypervector from raw bytes (256 bytes = 2048 bits)
    pub fn from_bytes(bytes: &[u8; 256]) -> Self {
        let mut words = [0u128; HDC_WORDS];
        for (i, chunk) in bytes.chunks_exact(16).enumerate() {
            words[i] = u128::from_le_bytes(chunk.try_into().unwrap_or([0u8; 16]));
        }
        Self { words }
    }

    /// Convert to raw bytes
    pub fn to_bytes(&self) -> [u8; 256] {
        let mut bytes = [0u8; 256];
        for (i, &word) in self.words.iter().enumerate() {
            bytes[i * 16..(i + 1) * 16].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// XOR binding: associates two hypervectors
    /// Binding is its own inverse: bind(bind(A, B), B) = A
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..HDC_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// XOR in-place
    #[inline]
    pub fn bind_mut(&mut self, other: &Self) {
        for i in 0..HDC_WORDS {
            self.words[i] ^= other.words[i];
        }
    }

    /// Population count (number of 1-bits)
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Hamming distance between two hypervectors
    #[inline]
    pub fn hamming(&self, other: &Self) -> u32 {
        self.bind(other).popcount()
    }

    /// Normalized similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
    /// Maps Hamming distance to [-1, 1] range
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming(other) as f32;
        1.0 - (2.0 * hamming / HDC_DIM as f32)
    }

    /// Cosine-like similarity in [0, 1] range
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        (self.similarity(other) + 1.0) / 2.0
    }

    /// Circular shift (permutation) - used for sequence encoding
    pub fn rotate(&self, amount: i32) -> Self {
        if amount == 0 {
            return *self;
        }

        let mut result = Self::zero();
        let shift = amount.rem_euclid(HDC_DIM as i32) as usize;

        // For each bit position, compute its source
        for dst_bit in 0..HDC_DIM {
            let src_bit = (dst_bit + HDC_DIM - shift) % HDC_DIM;

            let src_word = src_bit / 128;
            let src_offset = src_bit % 128;
            let dst_word = dst_bit / 128;
            let dst_offset = dst_bit % 128;

            if (self.words[src_word] >> src_offset) & 1 == 1 {
                result.words[dst_word] |= 1u128 << dst_offset;
            }
        }

        result
    }

    /// Flip all bits (NOT operation)
    pub fn flip(&self) -> Self {
        let mut result = Self::zero();
        for i in 0..HDC_WORDS {
            result.words[i] = !self.words[i];
        }
        result
    }

    /// Check if approximately equal (similarity above threshold)
    pub fn approx_eq(&self, other: &Self, threshold: f32) -> bool {
        self.similarity(other) >= threshold
    }
}

/// Accumulator for majority-vote bundling
#[derive(Clone, Debug)]
pub struct BundleAccumulator {
    /// Per-bit counters
    counts: [i32; HDC_DIM],
    /// Number of vectors accumulated
    n: usize,
}

impl Default for BundleAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl BundleAccumulator {
    /// Create a new empty accumulator
    pub fn new() -> Self {
        Self {
            counts: [0; HDC_DIM],
            n: 0,
        }
    }

    /// Add a hypervector to the accumulator
    pub fn add(&mut self, hv: &HV16) {
        for i in 0..HDC_DIM {
            let word_idx = i / 128;
            let bit_idx = i % 128;
            let bit = ((hv.words[word_idx] >> bit_idx) & 1) as i32;
            self.counts[i] += if bit == 1 { 1 } else { -1 };
        }
        self.n += 1;
    }

    /// Add a weighted hypervector
    pub fn add_weighted(&mut self, hv: &HV16, weight: f32) {
        let w = weight as i32;
        for i in 0..HDC_DIM {
            let word_idx = i / 128;
            let bit_idx = i % 128;
            let bit = ((hv.words[word_idx] >> bit_idx) & 1) as i32;
            self.counts[i] += if bit == 1 { w } else { -w };
        }
        self.n += 1;
    }

    /// Finalize to a hypervector using majority voting
    pub fn finalize(&self) -> HV16 {
        let mut result = HV16::zero();
        for i in 0..HDC_DIM {
            if self.counts[i] > 0 {
                let word_idx = i / 128;
                let bit_idx = i % 128;
                result.words[word_idx] |= 1u128 << bit_idx;
            }
        }
        result
    }

    /// Number of vectors accumulated
    pub fn count(&self) -> usize {
        self.n
    }

    /// Clear the accumulator
    pub fn clear(&mut self) {
        self.counts = [0; HDC_DIM];
        self.n = 0;
    }
}

/// Bundle multiple hypervectors using majority voting
pub fn bundle(hvs: &[HV16]) -> HV16 {
    if hvs.is_empty() {
        return HV16::zero();
    }
    if hvs.len() == 1 {
        return hvs[0];
    }

    let mut acc = BundleAccumulator::new();
    for hv in hvs {
        acc.add(hv);
    }
    acc.finalize()
}

/// Bundle with weights
pub fn weighted_bundle(hvs: &[(HV16, f32)]) -> HV16 {
    if hvs.is_empty() {
        return HV16::zero();
    }

    let mut acc = BundleAccumulator::new();
    for (hv, weight) in hvs {
        acc.add_weighted(hv, *weight);
    }
    acc.finalize()
}

/// Encode a sequence of hypervectors using rotation + binding
/// Position 0 is unrotated, position 1 is rotated by 1, etc.
pub fn encode_sequence(hvs: &[HV16]) -> HV16 {
    if hvs.is_empty() {
        return HV16::zero();
    }

    let mut acc = BundleAccumulator::new();
    for (i, hv) in hvs.iter().enumerate() {
        acc.add(&hv.rotate(i as i32));
    }
    acc.finalize()
}

/// N-gram encoder: binds rotated hypervectors
pub fn encode_ngram(hvs: &[HV16]) -> HV16 {
    if hvs.is_empty() {
        return HV16::zero();
    }

    let mut result = hvs[0];
    for (i, hv) in hvs.iter().enumerate().skip(1) {
        result = result.bind(&hv.rotate(i as i32));
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYMTHAEA-CORE BRIDGE: Interoperability with main Symthaea HDC types
// ═══════════════════════════════════════════════════════════════════════════════

/// symthaea-core uses 16,384-bit vectors, we use 2,048-bit.
/// This constant defines the expansion factor.
pub const CORE_HDC_DIM: usize = 16_384;
pub const EXPANSION_FACTOR: usize = CORE_HDC_DIM / HDC_DIM; // 8x

impl HV16 {
    /// Convert to continuous f32 vector (bipolar interpretation)
    ///
    /// Each bit becomes +1.0 or -1.0 in the output.
    /// Returns a 2048-element f32 vector.
    pub fn to_continuous(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(HDC_DIM);
        for i in 0..HDC_DIM {
            let word_idx = i / 128;
            let bit_idx = i % 128;
            let bit = (self.words[word_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// Create from continuous f32 vector (threshold at 0)
    ///
    /// Values > 0 become 1, values <= 0 become 0.
    pub fn from_continuous(values: &[f32]) -> Self {
        assert!(values.len() >= HDC_DIM, "Need at least {HDC_DIM} values");

        let mut result = Self::zero();
        for i in 0..HDC_DIM {
            if values[i] > 0.0 {
                let word_idx = i / 128;
                let bit_idx = i % 128;
                result.words[word_idx] |= 1u128 << bit_idx;
            }
        }
        result
    }

    /// Expand to 16,384-bit representation for symthaea-core compatibility
    ///
    /// Uses deterministic expansion where each bit generates 8 bits
    /// via BLAKE3 hashing, preserving similarity relationships.
    pub fn expand_to_core(&self) -> Vec<u8> {
        let mut result = vec![0u8; CORE_HDC_DIM / 8];

        // Each of our 2048 bits expands to 8 bits in the output
        for src_bit in 0..HDC_DIM {
            let src_word = src_bit / 128;
            let src_offset = src_bit % 128;
            let bit_value = (self.words[src_word] >> src_offset) & 1;

            // If bit is 1, set 8 corresponding bits in output
            // Use a deterministic pattern based on position
            if bit_value == 1 {
                let dst_byte_start = src_bit; // Each src bit maps to 8 bits (1 byte)
                for offset in 0..EXPANSION_FACTOR {
                    let dst_idx = dst_byte_start * EXPANSION_FACTOR + offset;
                    let dst_byte = dst_idx / 8;
                    let dst_bit = dst_idx % 8;
                    result[dst_byte] |= 1 << dst_bit;
                }
            }
        }
        result
    }

    /// Compress from 16,384-bit representation back to HV16
    ///
    /// Uses majority voting: each group of 8 bits votes for 1 output bit.
    pub fn from_core_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= CORE_HDC_DIM / 8,
            "Need at least {} bytes",
            CORE_HDC_DIM / 8
        );

        let mut result = Self::zero();

        // Each group of 8 bits votes for one output bit
        for dst_bit in 0..HDC_DIM {
            let mut count = 0i32;

            for offset in 0..EXPANSION_FACTOR {
                let src_idx = dst_bit * EXPANSION_FACTOR + offset;
                let src_byte = src_idx / 8;
                let src_bit_offset = src_idx % 8;
                if (bytes[src_byte] >> src_bit_offset) & 1 == 1 {
                    count += 1;
                }
            }

            // Majority vote: more than half -> 1
            if count > (EXPANSION_FACTOR / 2) as i32 {
                let word_idx = dst_bit / 128;
                let bit_idx = dst_bit % 128;
                result.words[word_idx] |= 1u128 << bit_idx;
            }
        }
        result
    }

    /// Expand to continuous f32 representation at core dimension (16,384)
    ///
    /// This allows direct use with symthaea-core's ContinuousHV.
    pub fn to_core_continuous(&self) -> Vec<f32> {
        let binary = self.expand_to_core();
        let mut result = Vec::with_capacity(CORE_HDC_DIM);

        for i in 0..CORE_HDC_DIM {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (binary[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// Create from continuous f32 vector at core dimension (16,384)
    ///
    /// Compresses back to 2048-bit HV16 using majority voting.
    pub fn from_core_continuous(values: &[f32]) -> Self {
        assert!(
            values.len() >= CORE_HDC_DIM,
            "Need at least {CORE_HDC_DIM} values"
        );

        let mut result = Self::zero();

        // Each group of 8 values votes for one output bit
        for dst_bit in 0..HDC_DIM {
            let mut sum = 0.0f32;

            for offset in 0..EXPANSION_FACTOR {
                let src_idx = dst_bit * EXPANSION_FACTOR + offset;
                sum += values[src_idx];
            }

            // Positive sum -> 1
            if sum > 0.0 {
                let word_idx = dst_bit / 128;
                let bit_idx = dst_bit % 128;
                result.words[word_idx] |= 1u128 << bit_idx;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_deterministic() {
        let a = HV16::random("test");
        let b = HV16::random("test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_orthogonal() {
        let a = HV16::random("apple");
        let b = HV16::random("banana");
        // Random vectors should be approximately orthogonal (sim ~ 0)
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.1, "sim = {}", sim);
    }

    #[test]
    fn test_bind_inverse() {
        let a = HV16::random("A");
        let b = HV16::random("B");
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_bundle_similarity() {
        let a = HV16::random("A");
        let b = HV16::random("B");
        let c = HV16::random("C");
        let bundled = bundle(&[a, b, c]);

        // Bundle should be similar to all constituents
        assert!(bundled.similarity(&a) > 0.3);
        assert!(bundled.similarity(&b) > 0.3);
        assert!(bundled.similarity(&c) > 0.3);
    }

    #[test]
    fn test_popcount() {
        let zero = HV16::zero();
        assert_eq!(zero.popcount(), 0);

        let random = HV16::random("test");
        // Random vector should have ~50% ones
        let pop = random.popcount();
        assert!(pop > 900 && pop < 1150, "popcount = {}", pop);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // SYMTHAEA-CORE BRIDGE TESTS
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_continuous_roundtrip() {
        let original = HV16::random("continuous_test");
        let continuous = original.to_continuous();

        assert_eq!(continuous.len(), HDC_DIM);

        // All values should be +1 or -1
        for &val in &continuous {
            assert!(val == 1.0 || val == -1.0);
        }

        // Roundtrip should be exact
        let recovered = HV16::from_continuous(&continuous);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_expand_compress_roundtrip() {
        let original = HV16::random("expand_test");
        let expanded = original.expand_to_core();

        assert_eq!(expanded.len(), CORE_HDC_DIM / 8); // 2048 bytes

        let compressed = HV16::from_core_bytes(&expanded);
        assert_eq!(original, compressed, "Expand/compress roundtrip failed");
    }

    #[test]
    fn test_core_continuous_roundtrip() {
        let original = HV16::random("core_continuous_test");
        let expanded = original.to_core_continuous();

        assert_eq!(expanded.len(), CORE_HDC_DIM);

        let compressed = HV16::from_core_continuous(&expanded);
        assert_eq!(original, compressed, "Core continuous roundtrip failed");
    }

    #[test]
    fn test_expansion_preserves_similarity() {
        let a = HV16::random("similarity_A");
        let b = HV16::random("similarity_B");
        let c = bundle(&[a, b]); // c is similar to both a and b

        // Original similarities
        let sim_a_b = a.similarity(&b);
        let sim_a_c = a.similarity(&c);
        let sim_b_c = b.similarity(&c);

        // Expand all to core dimension
        let a_exp = a.expand_to_core();
        let b_exp = b.expand_to_core();
        let c_exp = c.expand_to_core();

        // Calculate expanded similarities (Hamming-based)
        let hamming_ab: u32 = a_exp
            .iter()
            .zip(b_exp.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        let exp_sim_a_b = 1.0 - (2.0 * hamming_ab as f32 / CORE_HDC_DIM as f32);

        let hamming_ac: u32 = a_exp
            .iter()
            .zip(c_exp.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        let exp_sim_a_c = 1.0 - (2.0 * hamming_ac as f32 / CORE_HDC_DIM as f32);

        let hamming_bc: u32 = b_exp
            .iter()
            .zip(c_exp.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        let exp_sim_b_c = 1.0 - (2.0 * hamming_bc as f32 / CORE_HDC_DIM as f32);

        // Expanded similarities should match original (within tolerance due to expansion)
        assert!(
            (sim_a_b - exp_sim_a_b).abs() < 0.01,
            "A-B: {} vs {}",
            sim_a_b,
            exp_sim_a_b
        );
        assert!(
            (sim_a_c - exp_sim_a_c).abs() < 0.01,
            "A-C: {} vs {}",
            sim_a_c,
            exp_sim_a_c
        );
        assert!(
            (sim_b_c - exp_sim_b_c).abs() < 0.01,
            "B-C: {} vs {}",
            sim_b_c,
            exp_sim_b_c
        );
    }
}
