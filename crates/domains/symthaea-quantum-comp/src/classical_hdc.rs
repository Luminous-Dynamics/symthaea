//! Classical binary hyperdimensional computing baseline.

use crate::errors::{QuantumCompError, Result};
use crate::rng::XorShift64;

/// Packed binary hypervector using `u64` words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryHypervector {
    dimension: usize,
    words: Vec<u64>,
}

impl BinaryHypervector {
    /// Creates a zero-valued hypervector with the requested bit dimension.
    pub fn zeros(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        let words_len = dimension.div_ceil(64);
        Ok(Self {
            dimension,
            words: vec![0; words_len],
        })
    }

    /// Creates a hypervector from packed words, masking unused high bits.
    pub fn from_words(dimension: usize, words: Vec<u64>) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        let expected = dimension.div_ceil(64);
        if words.len() != expected {
            return Err(QuantumCompError::DimensionMismatch {
                expected,
                actual: words.len(),
            });
        }
        let mut out = Self { dimension, words };
        out.mask_unused_bits();
        Ok(out)
    }

    /// Creates a random binary hypervector with deterministic seed.
    pub fn random(dimension: usize, seed: u64) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        let words_len = dimension.div_ceil(64);
        let mut rng = XorShift64::new(seed);
        let mut words = Vec::with_capacity(words_len);
        for _ in 0..words_len {
            words.push(rng.next_u64());
        }
        let mut hv = Self { dimension, words };
        hv.mask_unused_bits();
        Ok(hv)
    }

    /// Returns the bit dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of packed words.
    pub fn word_len(&self) -> usize {
        self.words.len()
    }

    /// Returns packed words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// XOR binding operation.
    pub fn bind_xor(&self, other: &Self) -> Result<Self> {
        self.check_same_dimension(other)?;
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| a ^ b)
            .collect();
        Ok(Self {
            dimension: self.dimension,
            words,
        })
    }

    /// XOR unbinding. For binary XOR binding, unbinding is the same operation.
    pub fn unbind_xor(&self, key: &Self) -> Result<Self> {
        self.bind_xor(key)
    }

    /// Permutes bits by deterministic cyclic rotation. This is a simple HDC role operation.
    pub fn rotate_bits(&self, shift: usize) -> Self {
        let shift = shift % self.dimension;
        if shift == 0 {
            return self.clone();
        }
        let mut out = Self::zeros(self.dimension).expect("nonzero dimension already known");
        for i in 0..self.dimension {
            let target = (i + shift) % self.dimension;
            if self.bit(i).unwrap_or(false) {
                let _ = out.set_bit(target, true);
            }
        }
        out
    }

    /// Hamming distance between two hypervectors.
    pub fn hamming_distance(&self, other: &Self) -> Result<usize> {
        self.check_same_dimension(other)?;
        let dist = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum();
        Ok(dist)
    }

    /// Similarity in `[0, 1]`, where `1` is identical.
    pub fn similarity(&self, other: &Self) -> Result<f32> {
        let dist = self.hamming_distance(other)? as f32;
        Ok(1.0 - dist / self.dimension as f32)
    }

    /// Returns a noisy copy with independent bit flips.
    pub fn with_bitflip_noise(&self, probability: f32, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let mut out = self.clone();
        for bit in 0..self.dimension {
            if rng.chance(probability) {
                out.flip_bit(bit);
            }
        }
        out
    }

    /// Encodes a scalar in `[0, 1]` as a thermometer/unary code: the first
    /// `round(value * dimension)` bits are set to `1`, the rest `0`.
    ///
    /// Thermometer coding is the fair, graceful-degradation baseline for
    /// representing a continuous quantity in a binary vector: decode is a
    /// plain popcount, so a bit flip *anywhere* changes the decoded estimate
    /// by exactly `1/dimension`, regardless of which bit flipped. A
    /// positional (even gray-coded) binary encoding of the same value does
    /// not have this property — see
    /// `docs/RESEARCH_NOTES.md`'s continuous-value comparison for why this
    /// is the representation classical HDC gets to be tested with, not a
    /// weaker strawman.
    pub fn thermometer_encode(value: f32, dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        let clamped = value.clamp(0.0, 1.0);
        let ones = ((clamped * dimension as f32).round() as usize).min(dimension);
        let mut out = Self::zeros(dimension)?;
        for i in 0..ones {
            out.set_bit(i, true)?;
        }
        Ok(out)
    }

    /// Decodes a thermometer-coded hypervector back to a scalar estimate in `[0, 1]`.
    ///
    /// The estimate is the fraction of set bits — `mask_unused_bits` already
    /// zeroes any bits beyond `dimension`, so summing popcount across all
    /// words is safe.
    pub fn thermometer_decode(&self) -> f32 {
        let ones: usize = self.words.iter().map(|w| w.count_ones() as usize).sum();
        ones as f32 / self.dimension as f32
    }

    /// Majority-bundles a nonempty slice of hypervectors.
    pub fn majority_bundle(vectors: &[Self], tie_seed: u64) -> Result<Self> {
        if vectors.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "bundle requires at least one vector",
            ));
        }
        let dimension = vectors[0].dimension;
        for v in vectors {
            if v.dimension != dimension {
                return Err(QuantumCompError::DimensionMismatch {
                    expected: dimension,
                    actual: v.dimension,
                });
            }
        }
        let mut rng = XorShift64::new(tie_seed);
        let mut out = Self::zeros(dimension)?;
        for bit in 0..dimension {
            let ones = vectors
                .iter()
                .filter(|v| v.bit(bit).unwrap_or(false))
                .count();
            let zeros = vectors.len() - ones;
            let value = if ones > zeros {
                true
            } else if zeros > ones {
                false
            } else {
                rng.chance(0.5)
            };
            out.set_bit(bit, value)?;
        }
        Ok(out)
    }

    /// Gets one bit.
    pub fn bit(&self, index: usize) -> Option<bool> {
        if index >= self.dimension {
            return None;
        }
        let word = self.words[index / 64];
        Some(((word >> (index % 64)) & 1) == 1)
    }

    /// Sets one bit.
    pub fn set_bit(&mut self, index: usize, value: bool) -> Result<()> {
        if index >= self.dimension {
            return Err(QuantumCompError::DimensionMismatch {
                expected: self.dimension,
                actual: index + 1,
            });
        }
        let mask = 1u64 << (index % 64);
        let word = &mut self.words[index / 64];
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
        Ok(())
    }

    fn flip_bit(&mut self, index: usize) {
        let mask = 1u64 << (index % 64);
        self.words[index / 64] ^= mask;
    }

    fn check_same_dimension(&self, other: &Self) -> Result<()> {
        if self.dimension != other.dimension {
            return Err(QuantumCompError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }
        Ok(())
    }

    fn mask_unused_bits(&mut self) {
        let rem = self.dimension % 64;
        if rem == 0 {
            return;
        }
        let mask = (1u64 << rem) - 1;
        if let Some(last) = self.words.last_mut() {
            *last &= mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_binding_round_trips() {
        let a = BinaryHypervector::random(1024, 1).unwrap();
        let key = BinaryHypervector::random(1024, 2).unwrap();
        let bound = a.bind_xor(&key).unwrap();
        let recovered = bound.unbind_xor(&key).unwrap();
        assert_eq!(a, recovered);
    }

    #[test]
    fn thermometer_round_trips_within_one_quantization_step() {
        for x in [0.0f32, 0.1, 0.37, 0.5, 0.83, 1.0] {
            let encoded = BinaryHypervector::thermometer_encode(x, 1024).unwrap();
            let decoded = encoded.thermometer_decode();
            assert!(
                (decoded - x).abs() <= 1.0 / 1024.0,
                "x={x} decoded={decoded}"
            );
        }
    }

    #[test]
    fn thermometer_clamps_out_of_range_inputs() {
        assert_eq!(
            BinaryHypervector::thermometer_encode(-1.0, 64)
                .unwrap()
                .thermometer_decode(),
            0.0
        );
        assert_eq!(
            BinaryHypervector::thermometer_encode(2.0, 64)
                .unwrap()
                .thermometer_decode(),
            1.0
        );
    }

    #[test]
    fn thermometer_decode_moves_by_exactly_one_step_per_bit_flip() {
        let encoded = BinaryHypervector::thermometer_encode(0.5, 1024).unwrap();
        let before = encoded.thermometer_decode();
        let mut flipped = encoded.clone();
        flipped.flip_bit(10); // arbitrary index, anywhere works by design
        let after = flipped.thermometer_decode();
        assert!(((after - before).abs() - 1.0 / 1024.0).abs() < 1e-6);
    }

    #[test]
    fn majority_bundle_is_deterministic() {
        let a = BinaryHypervector::random(128, 1).unwrap();
        let b = BinaryHypervector::random(128, 2).unwrap();
        let c = BinaryHypervector::random(128, 3).unwrap();
        let x = BinaryHypervector::majority_bundle(&[a.clone(), b.clone(), c.clone()], 9).unwrap();
        let y = BinaryHypervector::majority_bundle(&[a, b, c], 9).unwrap();
        assert_eq!(x, y);
    }
}
