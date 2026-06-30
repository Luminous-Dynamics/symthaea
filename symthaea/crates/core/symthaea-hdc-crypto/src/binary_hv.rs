// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Minimal BinaryHV implementation for cryptographic operations.
//!
//! This is a standalone extraction of the core BinaryHV type from symthaea-core,
//! containing only the operations needed for HDC cryptography:
//! bind (XOR), bundle (majority vote), similarity (Hamming), and permute (cyclic shift).
//!
//! For the full HDC toolkit (weighted bundle, SIMD acceleration, density normalization,
//! noise injection, bipolar conversion, etc.), use `symthaea-core` directly.

use serde::{Deserialize, Serialize};

/// Number of bits in each hypervector.
pub const HDC_DIMENSION: usize = 16_384;

/// Number of bytes in each hypervector (16,384 / 8).
pub const HDC_BYTES: usize = 2048;

/// 16,384-bit binary hypervector (2KB).
///
/// Each bit represents one dimension: bit=1 means +1, bit=0 means -1 (bipolar encoding).
///
/// # Memory Layout
///
/// 2048 bytes = 16,384 bits (2^14). 32-byte aligned for potential SIMD use.
///
/// # Examples
///
/// ```
/// use symthaea_hdc_crypto::BinaryHV;
///
/// let a = BinaryHV::new_random(42);
/// let b = BinaryHV::new_random(43);
///
/// // Binding (XOR)
/// let c = a.bind(&b);
///
/// // Similarity (Hamming)
/// let sim = a.similarity(&b);
/// assert!(sim > 0.45 && sim < 0.55); // random vectors ~ 0.5
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(align(32))]
pub struct BinaryHV(#[serde(with = "serde_bytes_2048")] pub [u8; HDC_BYTES]);

impl BinaryHV {
    /// Dimension of the hypervector (16,384 bits).
    pub const DIM: usize = HDC_DIMENSION;

    /// Number of bytes (2048).
    pub const BYTES: usize = HDC_BYTES;

    /// Create zero vector (all bits 0).
    pub const fn zero() -> Self {
        Self([0u8; HDC_BYTES])
    }

    /// Create a deterministic random hypervector from a seed.
    ///
    /// Uses BLAKE3 XOF (extendable output function) to fill the 2048 bytes
    /// deterministically from the seed. Same seed always produces the same vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symthaea_hdc_crypto::BinaryHV;
    ///
    /// let v1 = BinaryHV::new_random(42);
    /// let v2 = BinaryHV::new_random(42);
    /// assert_eq!(v1, v2); // Same seed = same vector
    /// ```
    pub fn new_random(seed: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        let mut bytes = [0u8; HDC_BYTES];
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut bytes);
        Self(bytes)
    }

    /// Bind two vectors (XOR operation).
    ///
    /// Binding combines concepts: `cat.bind(&orange)` = "orange cat".
    ///
    /// Properties:
    /// - Commutative: A ^ B = B ^ A
    /// - Associative: (A ^ B) ^ C = A ^ (B ^ C)
    /// - Self-inverse: A ^ A = 0
    /// - Identity: A ^ 0 = A
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u8; HDC_BYTES];
        for i in 0..HDC_BYTES {
            result[i] = self.0[i] ^ other.0[i];
        }
        Self(result)
    }

    /// Bundle multiple vectors via majority vote.
    ///
    /// For each bit position, the output bit is 1 if more than half the input
    /// vectors have a 1 at that position.
    ///
    /// # Examples
    ///
    /// ```
    /// use symthaea_hdc_crypto::BinaryHV;
    ///
    /// let a = BinaryHV::new_random(1);
    /// let b = BinaryHV::new_random(2);
    /// let c = BinaryHV::new_random(3);
    /// let prototype = BinaryHV::bundle(&[a, b, c]);
    /// assert!(prototype.similarity(&a) > 0.5);
    /// ```
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero();
        }

        let threshold = vectors.len() as i32 / 2;
        let mut result = [0u8; HDC_BYTES];

        // Count bits at each position using byte-level iteration
        for byte_idx in 0..HDC_BYTES {
            let mut byte_result = 0u8;
            for bit in 0..8u8 {
                let mut count = 0i32;
                let mask = 1u8 << bit;
                for v in vectors {
                    if v.0[byte_idx] & mask != 0 {
                        count += 1;
                    }
                }
                if count > threshold {
                    byte_result |= mask;
                }
            }
            result[byte_idx] = byte_result;
        }

        Self(result)
    }

    /// Hamming similarity: fraction of matching bits, in [0.0, 1.0].
    ///
    /// - 1.0 = identical vectors
    /// - 0.5 = unrelated (expected for random vectors)
    /// - 0.0 = complementary vectors
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        let matching: u32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / Self::DIM as f32
    }

    /// Cyclic permutation (bit rotation).
    ///
    /// Shifts all bits left by `shift` positions, wrapping around.
    /// Essential for encoding order: `a.bind(&b.permute(1))` != `b.bind(&a.permute(1))`.
    pub fn permute(&self, shift: usize) -> Self {
        if shift == 0 {
            return *self;
        }

        let shift = shift % Self::DIM;

        // Convert bytes to u64 words for efficient rotation
        let mut words = [0u64; 256];
        for (i, chunk) in self.0.chunks_exact(8).enumerate() {
            words[i] = u64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
        }

        let mut result_words = [0u64; 256];
        let word_shift = shift / 64;
        let bit_shift = shift % 64;

        if bit_shift == 0 {
            for i in 0..256 {
                let new_pos = (i + word_shift) % 256;
                result_words[new_pos] = words[i];
            }
        } else {
            let inv_shift = 64 - bit_shift;
            for i in 0..256 {
                let new_pos = (i + word_shift) % 256;
                let prev = if i == 0 { 255 } else { i - 1 };
                result_words[new_pos] = (words[i] << bit_shift) | (words[prev] >> inv_shift);
            }
        }

        let mut result = [0u8; HDC_BYTES];
        for (i, word) in result_words.iter().enumerate() {
            let bytes = word.to_ne_bytes();
            result[i * 8..i * 8 + 8].copy_from_slice(&bytes);
        }

        Self(result)
    }

    /// Density: fraction of bits that are 1, in [0.0, 1.0].
    #[inline]
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / Self::DIM as f32
    }

    /// Population count: number of 1-bits.
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.0.iter().map(|byte| byte.count_ones()).sum()
    }
}

impl std::fmt::Debug for BinaryHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BinaryHV(density={:.3}, first_8={:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...)",
            self.density(),
            self.0[0],
            self.0[1],
            self.0[2],
            self.0[3],
            self.0[4],
            self.0[5],
            self.0[6],
            self.0[7]
        )
    }
}

impl Default for BinaryHV {
    fn default() -> Self {
        Self::zero()
    }
}

// Custom serde for [u8; 2048] (serde derive doesn't support arrays > 32)
mod serde_bytes_2048 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(data: &[u8; 2048], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        data[..].serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 2048], D::Error>
    where
        D: Deserializer<'de>,
    {
        let slice: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if slice.len() != 2048 {
            return Err(serde::de::Error::custom(format!(
                "Expected 2048 bytes, got {}",
                slice.len()
            )));
        }
        let mut array = [0u8; 2048];
        array.copy_from_slice(&slice);
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_deterministic() {
        let a = BinaryHV::new_random(42);
        let b = BinaryHV::new_random(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_balanced_density() {
        let v = BinaryHV::new_random(42);
        let d = v.density();
        assert!(
            d > 0.45 && d < 0.55,
            "Random density should be ~0.5, got {d}"
        );
    }

    #[test]
    fn test_zero() {
        let z = BinaryHV::zero();
        assert_eq!(z.density(), 0.0);
    }

    #[test]
    fn test_bind_xor() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let c = a.bind(&b);

        // XOR is self-inverse
        let recovered = c.bind(&b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_bind_self_inverse() {
        let a = BinaryHV::new_random(42);
        let z = a.bind(&a);
        assert_eq!(z, BinaryHV::zero());
    }

    #[test]
    fn test_similarity_self() {
        let a = BinaryHV::new_random(42);
        assert_eq!(a.similarity(&a), 1.0);
    }

    #[test]
    fn test_similarity_random() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let sim = a.similarity(&b);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Random vectors should be ~0.5 similar, got {sim}"
        );
    }

    #[test]
    fn test_bundle_majority_vote() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let c = BinaryHV::new_random(3);
        let bundle = BinaryHV::bundle(&[a, b, c]);

        // Bundle should be more similar to each input than random pairs
        assert!(bundle.similarity(&a) > 0.5);
        assert!(bundle.similarity(&b) > 0.5);
        assert!(bundle.similarity(&c) > 0.5);
    }

    #[test]
    fn test_bundle_empty() {
        let b = BinaryHV::bundle(&[]);
        assert_eq!(b, BinaryHV::zero());
    }

    #[test]
    fn test_permute_identity() {
        let a = BinaryHV::new_random(42);
        assert_eq!(a.permute(0), a);
    }

    #[test]
    fn test_permute_full_cycle() {
        let a = BinaryHV::new_random(42);
        assert_eq!(a.permute(HDC_DIMENSION), a);
    }

    #[test]
    fn test_permute_changes_vector() {
        let a = BinaryHV::new_random(42);
        let p = a.permute(7);
        assert_ne!(a, p);
        // Permuted should be ~0.5 similar (quasi-independent)
        let sim = a.similarity(&p);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Permuted should be ~0.5 similar, got {sim}"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let v = BinaryHV::new_random(42);
        let json = serde_json::to_string(&v).unwrap();
        let recovered: BinaryHV = serde_json::from_str(&json).unwrap();
        assert_eq!(v, recovered);
    }
}
