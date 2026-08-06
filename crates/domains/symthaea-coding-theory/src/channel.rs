// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic channel models for reproducible coding experiments.
//!
//! Probabilities are represented as exact rational counts rather than floating
//! point thresholds. Sampling uses rejection to avoid modulo bias.

use std::{fmt, num::NonZeroU64};

use crate::hamming::InvalidBit;

/// Invalid exact probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityError {
    /// The denominator must be non-zero.
    ZeroDenominator,
    /// A probability numerator cannot exceed its denominator.
    NumeratorExceedsDenominator { numerator: u64, denominator: u64 },
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => write!(f, "probability denominator must be non-zero"),
            Self::NumeratorExceedsDenominator {
                numerator,
                denominator,
            } => write!(
                f,
                "probability numerator {numerator} exceeds denominator {denominator}"
            ),
        }
    }
}

impl std::error::Error for ProbabilityError {}

/// Exact probability `numerator / denominator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probability {
    numerator: u64,
    denominator: NonZeroU64,
}

impl Probability {
    /// Construct an exact probability in the closed interval `[0, 1]`.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, ProbabilityError> {
        let denominator = NonZeroU64::new(denominator).ok_or(ProbabilityError::ZeroDenominator)?;
        if numerator > denominator.get() {
            return Err(ProbabilityError::NumeratorExceedsDenominator {
                numerator,
                denominator: denominator.get(),
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// Numerator of the exact probability.
    #[must_use]
    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    /// Denominator of the exact probability.
    #[must_use]
    pub const fn denominator(self) -> u64 {
        self.denominator.get()
    }

    /// Floating representation for reporting only.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator.get() as f64
    }

    fn sample(self, rng: &mut DeterministicRng) -> bool {
        if self.numerator == 0 {
            return false;
        }
        if self.numerator == self.denominator.get() {
            return true;
        }
        rng.next_below(self.denominator) < self.numerator
    }
}

/// Stable SplitMix64 stream for evidence-reproducible simulations.
///
/// This is a simulation PRNG, not a cryptographic random-number generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Start a deterministic stream from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Emit the next 64-bit sample.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Emit the high byte of the next 64-bit sample.
    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }

    fn next_below(&mut self, upper: NonZeroU64) -> u64 {
        let upper = upper.get();
        let rejection_zone = u64::MAX - (u64::MAX % upper);
        loop {
            let candidate = self.next_u64();
            if candidate < rejection_zone {
                return candidate % upper;
            }
        }
    }
}

/// Output of a channel application plus exact corruption locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transmission<T> {
    /// Channel output.
    pub received: T,
    /// Zero-based locations changed or erased by the channel.
    pub corrupted_positions: Vec<usize>,
}

/// Output of a channel that distinguishes unknown errors from known erasures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataTransmission<T> {
    pub received: T,
    pub error_positions: Vec<usize>,
    pub erasure_positions: Vec<usize>,
}

impl<T> ErrataTransmission<T> {
    #[must_use]
    pub fn total_errata(&self) -> usize {
        self.error_positions.len() + self.erasure_positions.len()
    }
}

/// Invalid channel input or configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    /// A binary channel received a byte outside `{0, 1}`.
    InvalidBit(InvalidBit),
    /// A fixed-count erasure channel requested more positions than exist.
    TooManyRequestedErasures { requested: usize, symbols: usize },
    /// Fixed mixed errata requested more distinct locations than exist.
    TooManyRequestedErrata {
        errors: usize,
        erasures: usize,
        symbols: usize,
    },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBit(error) => error.fmt(f),
            Self::TooManyRequestedErasures { requested, symbols } => write!(
                f,
                "requested {requested} erasures from a {symbols}-symbol word"
            ),
            Self::TooManyRequestedErrata {
                errors,
                erasures,
                symbols,
            } => write!(
                f,
                "requested {errors} errors plus {erasures} erasures from a {symbols}-symbol word"
            ),
        }
    }
}

impl std::error::Error for ChannelError {}

impl From<InvalidBit> for ChannelError {
    fn from(value: InvalidBit) -> Self {
        Self::InvalidBit(value)
    }
}

/// Independent bit flips with an exact per-bit probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinarySymmetricChannel {
    flip_probability: Probability,
}

impl BinarySymmetricChannel {
    /// Construct a binary symmetric channel.
    #[must_use]
    pub const fn new(flip_probability: Probability) -> Self {
        Self { flip_probability }
    }

    /// Exact configured bit-flip probability.
    #[must_use]
    pub const fn flip_probability(self) -> Probability {
        self.flip_probability
    }

    /// Transmit validated bits and record every flipped position.
    pub fn transmit(
        self,
        bits: &[u8],
        rng: &mut DeterministicRng,
    ) -> Result<Transmission<Vec<u8>>, ChannelError> {
        validate_bits(bits)?;
        let mut received = bits.to_vec();
        let mut corrupted_positions = Vec::new();
        for (position, bit) in received.iter_mut().enumerate() {
            if self.flip_probability.sample(rng) {
                *bit ^= 1;
                corrupted_positions.push(position);
            }
        }
        Ok(Transmission {
            received,
            corrupted_positions,
        })
    }
}

/// Independent byte-symbol corruption by a uniformly generated non-zero XOR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolErrorChannel {
    error_probability: Probability,
}

impl SymbolErrorChannel {
    /// Construct an independent symbol-error channel.
    #[must_use]
    pub const fn new(error_probability: Probability) -> Self {
        Self { error_probability }
    }

    /// Transmit bytes and record every changed symbol.
    #[must_use]
    pub fn transmit(self, symbols: &[u8], rng: &mut DeterministicRng) -> Transmission<Vec<u8>> {
        let mut received = symbols.to_vec();
        let mut corrupted_positions = Vec::new();
        for (position, symbol) in received.iter_mut().enumerate() {
            if self.error_probability.sample(rng) {
                let mut magnitude = rng.next_u8();
                if magnitude == 0 {
                    magnitude = 1;
                }
                *symbol ^= magnitude;
                corrupted_positions.push(position);
            }
        }
        Transmission {
            received,
            corrupted_positions,
        }
    }
}

/// Independent known symbol erasures represented by a placeholder byte plus
/// an out-of-band position list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolErasureChannel {
    erasure_probability: Probability,
    placeholder: u8,
}

impl SymbolErasureChannel {
    #[must_use]
    pub const fn new(erasure_probability: Probability, placeholder: u8) -> Self {
        Self {
            erasure_probability,
            placeholder,
        }
    }

    #[must_use]
    pub const fn erasure_probability(self) -> Probability {
        self.erasure_probability
    }

    #[must_use]
    pub const fn placeholder(self) -> u8 {
        self.placeholder
    }

    /// Erase symbols independently and record all erased positions, including
    /// positions whose original byte happened to equal the placeholder.
    #[must_use]
    pub fn transmit(self, symbols: &[u8], rng: &mut DeterministicRng) -> Transmission<Vec<u8>> {
        let mut received = symbols.to_vec();
        let mut corrupted_positions = Vec::new();
        for (position, symbol) in received.iter_mut().enumerate() {
            if self.erasure_probability.sample(rng) {
                *symbol = self.placeholder;
                corrupted_positions.push(position);
            }
        }
        Transmission {
            received,
            corrupted_positions,
        }
    }
}

/// Exact-weight erasure channel for preregistered capacity experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedCountErasureChannel {
    erasure_count: usize,
    placeholder: u8,
}

impl FixedCountErasureChannel {
    #[must_use]
    pub const fn new(erasure_count: usize, placeholder: u8) -> Self {
        Self {
            erasure_count,
            placeholder,
        }
    }

    #[must_use]
    pub const fn erasure_count(self) -> usize {
        self.erasure_count
    }

    #[must_use]
    pub const fn placeholder(self) -> u8 {
        self.placeholder
    }

    /// Select exactly `erasure_count` distinct positions without replacement.
    pub fn transmit(
        self,
        symbols: &[u8],
        rng: &mut DeterministicRng,
    ) -> Result<Transmission<Vec<u8>>, ChannelError> {
        if self.erasure_count > symbols.len() {
            return Err(ChannelError::TooManyRequestedErasures {
                requested: self.erasure_count,
                symbols: symbols.len(),
            });
        }

        let mut candidates = (0..symbols.len()).collect::<Vec<_>>();
        for index in 0..self.erasure_count {
            let remaining = NonZeroU64::new((symbols.len() - index) as u64)
                .expect("a requested erasure always has a remaining candidate");
            let selected = index + rng.next_below(remaining) as usize;
            candidates.swap(index, selected);
        }

        let mut corrupted_positions = candidates[..self.erasure_count].to_vec();
        corrupted_positions.sort_unstable();
        let mut received = symbols.to_vec();
        for &position in &corrupted_positions {
            received[position] = self.placeholder;
        }
        Ok(Transmission {
            received,
            corrupted_positions,
        })
    }
}

/// Exact-count mixed unknown errors and known erasures.
///
/// Error and erasure locations are sampled without replacement and are always
/// disjoint. Unknown errors use a non-zero XOR magnitude; erasures use the
/// configured placeholder and retain their locations out of band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedCountErrataChannel {
    error_count: usize,
    erasure_count: usize,
    erasure_placeholder: u8,
}

impl FixedCountErrataChannel {
    #[must_use]
    pub const fn new(error_count: usize, erasure_count: usize, erasure_placeholder: u8) -> Self {
        Self {
            error_count,
            erasure_count,
            erasure_placeholder,
        }
    }

    #[must_use]
    pub const fn error_count(self) -> usize {
        self.error_count
    }

    #[must_use]
    pub const fn erasure_count(self) -> usize {
        self.erasure_count
    }

    #[must_use]
    pub const fn erasure_placeholder(self) -> u8 {
        self.erasure_placeholder
    }

    pub fn transmit(
        self,
        symbols: &[u8],
        rng: &mut DeterministicRng,
    ) -> Result<ErrataTransmission<Vec<u8>>, ChannelError> {
        let requested = self
            .error_count
            .checked_add(self.erasure_count)
            .filter(|&count| count <= symbols.len())
            .ok_or(ChannelError::TooManyRequestedErrata {
                errors: self.error_count,
                erasures: self.erasure_count,
                symbols: symbols.len(),
            })?;

        let mut candidates = (0..symbols.len()).collect::<Vec<_>>();
        for index in 0..requested {
            let remaining = NonZeroU64::new((symbols.len() - index) as u64)
                .expect("a requested erratum always has a remaining candidate");
            let selected = index + rng.next_below(remaining) as usize;
            candidates.swap(index, selected);
        }

        let mut error_positions = candidates[..self.error_count].to_vec();
        let mut erasure_positions = candidates[self.error_count..requested].to_vec();
        error_positions.sort_unstable();
        erasure_positions.sort_unstable();

        let mut received = symbols.to_vec();
        for &position in &error_positions {
            let mut magnitude = rng.next_u8();
            if magnitude == 0 {
                magnitude = 1;
            }
            received[position] ^= magnitude;
        }
        for &position in &erasure_positions {
            received[position] = self.erasure_placeholder;
        }

        Ok(ErrataTransmission {
            received,
            error_positions,
            erasure_positions,
        })
    }
}

/// A bounded contiguous XOR burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurstXorChannel {
    /// Maximum number of contiguous symbols changed.
    pub burst_len: usize,
    /// Non-zero XOR applied to each symbol in the burst.
    pub magnitude: u8,
}

impl BurstXorChannel {
    /// Apply one uniformly positioned bounded burst.
    #[must_use]
    pub fn transmit(self, symbols: &[u8], rng: &mut DeterministicRng) -> Transmission<Vec<u8>> {
        let mut received = symbols.to_vec();
        let width = self.burst_len.min(received.len());
        if width == 0 || self.magnitude == 0 {
            return Transmission {
                received,
                corrupted_positions: Vec::new(),
            };
        }

        let choices = NonZeroU64::new((received.len() - width + 1) as u64)
            .expect("a non-empty bounded burst always has a start position");
        let start = rng.next_below(choices) as usize;
        let corrupted_positions = (start..start + width).collect::<Vec<_>>();
        for &position in &corrupted_positions {
            received[position] ^= self.magnitude;
        }
        Transmission {
            received,
            corrupted_positions,
        }
    }
}

fn validate_bits(bits: &[u8]) -> Result<(), InvalidBit> {
    for (index, &value) in bits.iter().enumerate() {
        if value > 1 {
            return Err(InvalidBit { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_rejects_invalid_ratios() {
        assert_eq!(
            Probability::new(1, 0),
            Err(ProbabilityError::ZeroDenominator)
        );
        assert_eq!(
            Probability::new(3, 2),
            Err(ProbabilityError::NumeratorExceedsDenominator {
                numerator: 3,
                denominator: 2,
            })
        );
    }

    #[test]
    fn seeded_transmissions_are_exactly_reproducible() {
        let probability = Probability::new(1, 4).unwrap();
        let channel = BinarySymmetricChannel::new(probability);
        let bits = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        let mut first_rng = DeterministicRng::new(42);
        let mut second_rng = DeterministicRng::new(42);
        assert_eq!(
            channel.transmit(&bits, &mut first_rng),
            channel.transmit(&bits, &mut second_rng)
        );
    }

    #[test]
    fn probability_extremes_are_exact() {
        let bits = [0, 1, 1, 0];
        let never = BinarySymmetricChannel::new(Probability::new(0, 1).unwrap());
        let always = BinarySymmetricChannel::new(Probability::new(1, 1).unwrap());
        let mut rng = DeterministicRng::new(7);
        assert_eq!(never.transmit(&bits, &mut rng).unwrap().received, bits);
        let flipped = always.transmit(&bits, &mut rng).unwrap();
        assert_eq!(flipped.received, [1, 0, 0, 1]);
        assert_eq!(flipped.corrupted_positions, [0, 1, 2, 3]);
    }

    #[test]
    fn binary_channel_rejects_non_bits() {
        let channel = BinarySymmetricChannel::new(Probability::new(1, 2).unwrap());
        let mut rng = DeterministicRng::new(1);
        assert_eq!(
            channel.transmit(&[0, 2], &mut rng),
            Err(ChannelError::InvalidBit(InvalidBit { index: 1, value: 2 }))
        );
    }

    #[test]
    fn symbol_channel_never_records_a_zero_magnitude_change() {
        let channel = SymbolErrorChannel::new(Probability::new(1, 1).unwrap());
        let source = [0u8; 512];
        let mut rng = DeterministicRng::new(0x5EED);
        let transmission = channel.transmit(&source, &mut rng);
        assert_eq!(transmission.corrupted_positions.len(), source.len());
        assert!(transmission.received.iter().all(|&symbol| symbol != 0));
    }

    #[test]
    fn independent_erasure_channel_records_placeholder_collisions() {
        let channel = SymbolErasureChannel::new(Probability::new(1, 1).unwrap(), 0x00);
        let source = [0x00, 0x11, 0x00, 0x22];
        let mut rng = DeterministicRng::new(7);
        let transmission = channel.transmit(&source, &mut rng);
        assert_eq!(transmission.received, [0; 4]);
        assert_eq!(transmission.corrupted_positions, [0, 1, 2, 3]);
    }

    #[test]
    fn fixed_count_erasure_channel_is_unique_bounded_and_reproducible() {
        let channel = FixedCountErasureChannel::new(7, 0xEE);
        let source = [0u8; 20];
        let mut first_rng = DeterministicRng::new(0xE2A5);
        let mut second_rng = DeterministicRng::new(0xE2A5);
        let first = channel.transmit(&source, &mut first_rng).unwrap();
        let second = channel.transmit(&source, &mut second_rng).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.corrupted_positions.len(), 7);
        assert!(
            first
                .corrupted_positions
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(
            first
                .corrupted_positions
                .iter()
                .all(|&position| first.received[position] == 0xEE)
        );
        assert_eq!(
            FixedCountErasureChannel::new(21, 0).transmit(&source, &mut first_rng),
            Err(ChannelError::TooManyRequestedErasures {
                requested: 21,
                symbols: 20,
            })
        );
    }

    #[test]
    fn fixed_errata_are_disjoint_exact_and_reproducible() {
        let channel = FixedCountErrataChannel::new(4, 7, 0xEE);
        let source = [0u8; 32];
        let mut first_rng = DeterministicRng::new(0xE22A_7A);
        let mut second_rng = DeterministicRng::new(0xE22A_7A);
        let first = channel.transmit(&source, &mut first_rng).unwrap();
        let second = channel.transmit(&source, &mut second_rng).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.error_positions.len(), 4);
        assert_eq!(first.erasure_positions.len(), 7);
        assert_eq!(first.total_errata(), 11);
        assert!(
            first
                .error_positions
                .iter()
                .all(|position| first.erasure_positions.binary_search(position).is_err())
        );
        assert!(
            first
                .error_positions
                .iter()
                .all(|&position| first.received[position] != source[position])
        );
        assert!(
            first
                .erasure_positions
                .iter()
                .all(|&position| first.received[position] == 0xEE)
        );
    }

    #[test]
    fn fixed_errata_reject_impossible_distinct_counts() {
        let source = [0u8; 10];
        let mut rng = DeterministicRng::new(1);
        assert_eq!(
            FixedCountErrataChannel::new(6, 5, 0).transmit(&source, &mut rng),
            Err(ChannelError::TooManyRequestedErrata {
                errors: 6,
                erasures: 5,
                symbols: 10,
            })
        );
    }

    #[test]
    fn burst_is_contiguous_and_bounded() {
        let channel = BurstXorChannel {
            burst_len: 6,
            magnitude: 0xA5,
        };
        let source = [0u8; 10];
        let mut rng = DeterministicRng::new(9);
        let transmission = channel.transmit(&source, &mut rng);
        assert_eq!(transmission.corrupted_positions.len(), 6);
        assert!(
            transmission
                .corrupted_positions
                .windows(2)
                .all(|pair| pair[1] == pair[0] + 1)
        );
        for &position in &transmission.corrupted_positions {
            assert_eq!(transmission.received[position], 0xA5);
        }
    }
}
