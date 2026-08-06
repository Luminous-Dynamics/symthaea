// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The repetition code: send each bit an odd number of times and decode by
//! strict majority vote.

use std::{fmt, num::NonZeroUsize};

use crate::parameters::{BlockCodeParameters, CodeFamily, SymbolKind};

/// Invalid repetition-code parameters or input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepetitionError {
    /// A repetition count of zero cannot form codeword groups.
    ZeroRepetitions,
    /// Even repetition counts admit ties and are rejected by the strict API.
    EvenRepetitions { repetitions: usize },
    /// An input symbol was not binary.
    InvalidBit { index: usize, value: u8 },
    /// The encoded allocation size overflowed `usize`.
    CapacityOverflow,
    /// A received word did not contain complete repetition groups.
    LengthNotMultiple { len: usize, repetitions: usize },
    /// A caller-provided output slice had the wrong length.
    OutputLengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for RepetitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRepetitions => write!(f, "repetition count must be non-zero"),
            Self::EvenRepetitions { repetitions } => write!(
                f,
                "strict majority decoding requires an odd repetition count, got {repetitions}"
            ),
            Self::InvalidBit { index, value } => {
                write!(f, "expected bit 0 or 1 at index {index}, got {value}")
            }
            Self::CapacityOverflow => write!(f, "encoded repetition-code length overflowed usize"),
            Self::LengthNotMultiple { len, repetitions } => write!(
                f,
                "received length {len} is not a multiple of repetition count {repetitions}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "output length {actual} does not match required length {expected}"
            ),
        }
    }
}

impl std::error::Error for RepetitionError {}

/// Validated odd repetition code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepetitionCode {
    repetitions: NonZeroUsize,
}

impl RepetitionCode {
    /// Construct a code with a non-zero odd repetition count.
    pub fn new(repetitions: usize) -> Result<Self, RepetitionError> {
        let repetitions = NonZeroUsize::new(repetitions).ok_or(RepetitionError::ZeroRepetitions)?;
        if repetitions.get() % 2 == 0 {
            return Err(RepetitionError::EvenRepetitions {
                repetitions: repetitions.get(),
            });
        }
        Ok(Self { repetitions })
    }

    /// Number of transmitted copies of each source bit.
    #[must_use]
    pub const fn repetitions(self) -> usize {
        self.repetitions.get()
    }

    /// Maximum number of errors correctable independently in each group.
    #[must_use]
    pub const fn correctable_errors_per_group(self) -> usize {
        (self.repetitions.get() - 1) / 2
    }

    /// Algebraic parameters for a fixed number of source bits.
    pub fn parameters(self, source_bits: usize) -> Result<BlockCodeParameters, RepetitionError> {
        let codeword_symbols = self.encoded_len(source_bits)?;
        let minimum_distance = self.repetitions.get();
        Ok(BlockCodeParameters {
            family: CodeFamily::Repetition,
            symbol_kind: SymbolKind::Bit,
            message_symbols: source_bits,
            parity_symbols: codeword_symbols.saturating_sub(source_bits),
            codeword_symbols,
            minimum_distance,
            unknown_error_correction_radius: (minimum_distance - 1) / 2,
            known_erasure_correction_radius: minimum_distance - 1,
        })
    }

    /// Required encoded output length.
    pub fn encoded_len(self, source_bits: usize) -> Result<usize, RepetitionError> {
        source_bits
            .checked_mul(self.repetitions.get())
            .ok_or(RepetitionError::CapacityOverflow)
    }

    /// Required decoded output length.
    pub fn decoded_len(self, received_len: usize) -> Result<usize, RepetitionError> {
        if received_len % self.repetitions.get() != 0 {
            return Err(RepetitionError::LengthNotMultiple {
                len: received_len,
                repetitions: self.repetitions.get(),
            });
        }
        Ok(received_len / self.repetitions.get())
    }

    /// Encode validated binary symbols.
    pub fn encode(self, bits: &[u8]) -> Result<Vec<u8>, RepetitionError> {
        let mut output = vec![0; self.encoded_len(bits.len())?];
        self.encode_into(bits, &mut output)?;
        Ok(output)
    }

    /// Encode into an exactly-sized caller-owned buffer.
    pub fn encode_into(self, bits: &[u8], output: &mut [u8]) -> Result<(), RepetitionError> {
        validate_bits(bits)?;
        let expected = self.encoded_len(bits.len())?;
        if output.len() != expected {
            return Err(RepetitionError::OutputLengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        for (group, &bit) in output.chunks_exact_mut(self.repetitions.get()).zip(bits) {
            group.fill(bit);
        }
        Ok(())
    }

    /// Decode complete, validated groups by strict majority.
    pub fn decode(self, received: &[u8]) -> Result<Vec<u8>, RepetitionError> {
        let mut output = vec![0; self.decoded_len(received.len())?];
        self.decode_into(received, &mut output)?;
        Ok(output)
    }

    /// Decode into an exactly-sized caller-owned buffer.
    pub fn decode_into(self, received: &[u8], output: &mut [u8]) -> Result<(), RepetitionError> {
        let expected = self.decoded_len(received.len())?;
        if output.len() != expected {
            return Err(RepetitionError::OutputLengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        validate_bits(received)?;

        for (decoded, group) in output
            .iter_mut()
            .zip(received.chunks_exact(self.repetitions.get()))
        {
            let ones = group.iter().map(|&bit| usize::from(bit)).sum::<usize>();
            *decoded = u8::from(ones > self.repetitions.get() / 2);
        }
        Ok(())
    }
}

/// Checked repetition encoding with a non-zero odd repetition count.
pub fn encode_checked(bits: &[u8], repetitions: usize) -> Result<Vec<u8>, RepetitionError> {
    RepetitionCode::new(repetitions)?.encode(bits)
}

/// Checked repetition decoding with complete groups and no tie policy.
pub fn decode_checked(received: &[u8], repetitions: usize) -> Result<Vec<u8>, RepetitionError> {
    RepetitionCode::new(repetitions)?.decode(received)
}

/// Lossy compatibility encoder.
///
/// Each byte is reduced modulo two. New code should use [`encode_checked`].
#[must_use]
pub fn encode(bits: &[u8], n: usize) -> Vec<u8> {
    let capacity = bits
        .len()
        .checked_mul(n)
        .expect("encoded repetition-code length overflowed usize");
    let mut output = Vec::with_capacity(capacity);
    for &bit in bits {
        output.extend(std::iter::repeat_n(bit & 1, n));
    }
    output
}

/// Lossy compatibility decoder.
///
/// Trailing partial groups are ignored, arbitrary bytes are reduced modulo two,
/// and even-sized ties decode to zero. New code should use [`decode_checked`].
#[must_use]
pub fn decode(received: &[u8], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    received
        .chunks_exact(n)
        .map(|group| {
            let ones = group.iter().map(|&bit| usize::from(bit & 1)).sum::<usize>();
            u8::from(ones > n / 2)
        })
        .collect()
}

fn validate_bits(bits: &[u8]) -> Result<(), RepetitionError> {
    for (index, &value) in bits.iter().enumerate() {
        if value > 1 {
            return Err(RepetitionError::InvalidBit { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_encode_then_decode_is_identity() {
        let data = [1, 0, 1, 1, 0];
        let code = RepetitionCode::new(5).unwrap();
        assert_eq!(code.decode(&code.encode(&data).unwrap()).unwrap(), data);
        assert_eq!(code.correctable_errors_per_group(), 2);
    }

    #[test]
    fn parameters_name_repetition_distance_and_rate() {
        let parameters = RepetitionCode::new(5).unwrap().parameters(3).unwrap();
        assert_eq!(parameters.message_symbols, 3);
        assert_eq!(parameters.codeword_symbols, 15);
        assert_eq!(parameters.minimum_distance, 5);
        assert_eq!(parameters.unknown_error_correction_radius, 2);
        assert_eq!(parameters.known_erasure_correction_radius, 4);
        assert_eq!(parameters.rate(), 0.2);
    }

    #[test]
    fn caller_owned_buffers_are_exact_and_reusable() {
        let code = RepetitionCode::new(5).unwrap();
        let source = [1, 0, 1, 1];
        let mut encoded = vec![0xAA; code.encoded_len(source.len()).unwrap()];
        code.encode_into(&source, &mut encoded).unwrap();

        let mut decoded = vec![0xAA; code.decoded_len(encoded.len()).unwrap()];
        code.decode_into(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded, source);

        assert_eq!(
            code.encode_into(&source, &mut [0; 3]),
            Err(RepetitionError::OutputLengthMismatch {
                expected: 20,
                actual: 3,
            })
        );
        assert_eq!(
            code.decode_into(&encoded, &mut [0; 3]),
            Err(RepetitionError::OutputLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn strict_api_rejects_ambiguous_or_malformed_inputs() {
        assert_eq!(
            RepetitionCode::new(0),
            Err(RepetitionError::ZeroRepetitions)
        );
        assert_eq!(
            RepetitionCode::new(4),
            Err(RepetitionError::EvenRepetitions { repetitions: 4 })
        );
        let code = RepetitionCode::new(3).unwrap();
        assert_eq!(
            code.encode(&[0, 2]),
            Err(RepetitionError::InvalidBit { index: 1, value: 2 })
        );
        assert_eq!(
            code.decode(&[1, 1]),
            Err(RepetitionError::LengthNotMultiple {
                len: 2,
                repetitions: 3,
            })
        );
    }

    #[test]
    fn exhaustively_corrects_every_pattern_within_radius() {
        for repetitions in [1usize, 3, 5, 7] {
            let code = RepetitionCode::new(repetitions).unwrap();
            for source in [0u8, 1] {
                let clean = code.encode(&[source]).unwrap();
                for errors in 0..=code.correctable_errors_per_group() {
                    for_each_combination(repetitions, errors, &mut |positions| {
                        let mut corrupted = clean.clone();
                        for &position in positions {
                            corrupted[position] ^= 1;
                        }
                        assert_eq!(
                            code.decode(&corrupted).unwrap(),
                            [source],
                            "n={repetitions}, source={source}, positions={positions:?}"
                        );
                    });
                }
            }
        }
    }

    #[test]
    fn compatibility_api_retains_documented_lossy_behavior() {
        assert_eq!(encode(&[3], 2), [1, 1]);
        assert_eq!(decode(&[1, 1, 0, 0, 1], 2), [1, 0]);
        assert!(decode(&[1, 1], 0).is_empty());
    }

    fn for_each_combination(population: usize, choose: usize, visit: &mut impl FnMut(&[usize])) {
        fn recurse(
            start: usize,
            population: usize,
            remaining: usize,
            selected: &mut Vec<usize>,
            visit: &mut impl FnMut(&[usize]),
        ) {
            if remaining == 0 {
                visit(selected);
                return;
            }
            for value in start..=population - remaining {
                selected.push(value);
                recurse(value + 1, population, remaining - 1, selected, visit);
                selected.pop();
            }
        }

        let mut selected = Vec::with_capacity(choose);
        recurse(0, population, choose, &mut selected, visit);
    }
}
