//! Separate, explicitly lossy bounded-error i16 coding experiment.
//!
//! Unlike the exact numeric codec, this format does not preserve source values.
//! It guarantees |source[i] - reconstruction[i]| <= tolerance for every sample
//! at encode time, using a decoder-synchronous predictor and quantized residual
//! indices. The decoded-sample checksum protects deterministic reconstruction;
//! CRC-32 is not an authentication mechanism.

use crate::{compress, crc32, decompress, CodecError, ModelKind};
use std::fmt;

const MAGIC: &[u8; 4] = b"BE16";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 36;
const MAX_INDEX_BYTES: usize = 10;
// |source - linear predictor| is at most 131,070 for two i16 history values.
const MAX_ABS_RESIDUAL_INDEX: u64 = 131_070;

/// Decoder-synchronous integer predictors for bounded-error coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundedPredictor {
    /// Predict the previous reconstructed sample (zero at stream start).
    Previous = 0,
    /// Predict 2*previous - previous_previous (reduced to Previous for the first sample).
    Linear = 1,
}

impl BoundedPredictor {
    fn from_id(id: u8) -> Result<Self, BoundedError> {
        match id {
            0 => Ok(Self::Previous),
            1 => Ok(Self::Linear),
            _ => Err(BoundedError::UnsupportedPredictor(id)),
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }
}

/// Bounded-error codec failures.
#[derive(Debug, PartialEq, Eq)]
pub enum BoundedError {
    Codec(CodecError),
    TruncatedHeader,
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedPredictor(u8),
    UnsupportedModel(u8),
    UnsupportedFlags(u16),
    LengthOverflow,
    LengthMismatch,
    OutputLimitExceeded { declared: u64, limit: usize },
    ChecksumMismatch,
    TruncatedVarint,
    VarintOverflow,
    NonCanonicalVarint,
    InvalidTolerance(u16),
    NumericOverflow,
    ResidualOutOfRange,
}

impl fmt::Display for BoundedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(f, "inner CDL codec error: {error}"),
            Self::TruncatedHeader => write!(f, "truncated bounded-error frame header"),
            Self::BadMagic => write!(f, "invalid bounded-error frame magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported bounded-error version {v}"),
            Self::UnsupportedPredictor(v) => write!(f, "unsupported predictor id {v}"),
            Self::UnsupportedModel(v) => write!(f, "unsupported entropy model id {v}"),
            Self::UnsupportedFlags(v) => write!(f, "unsupported bounded-error flags {v:#06x}"),
            Self::LengthOverflow => write!(f, "bounded-error frame length overflow"),
            Self::LengthMismatch => write!(f, "bounded-error frame length mismatch"),
            Self::OutputLimitExceeded { declared, limit } => {
                write!(f, "declared sample count {declared} exceeds limit {limit}")
            }
            Self::ChecksumMismatch => write!(f, "bounded-error reconstruction or payload checksum mismatch"),
            Self::TruncatedVarint => write!(f, "truncated bounded-error residual varint"),
            Self::VarintOverflow => write!(f, "bounded-error residual varint exceeds u64"),
            Self::NonCanonicalVarint => write!(f, "non-canonical bounded-error residual varint"),
            Self::InvalidTolerance(v) => write!(f, "bounded-error tolerance {v} exceeds the supported i16 range"),
            Self::NumericOverflow => write!(f, "bounded-error residual reconstruction overflowed i64"),
            Self::ResidualOutOfRange => write!(f, "bounded-error residual index is outside the range reachable from i16 input and predictor"),
        }
    }
}

impl std::error::Error for BoundedError {}

impl From<CodecError> for BoundedError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

fn zigzag_encode(value: i64) -> u64 {
    ((value as u64) << 1) ^ ((value >> 63) as u64)
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn write_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_varint(input: &[u8], cursor: &mut usize) -> Result<u64, BoundedError> {
    let mut value = 0u64;
    for group in 0..MAX_INDEX_BYTES {
        let byte = *input.get(*cursor).ok_or(BoundedError::TruncatedVarint)?;
        *cursor += 1;

        if group == 9 && (byte & 0xfe) != 0 {
            return Err(BoundedError::VarintOverflow);
        }
        value |= ((byte & 0x7f) as u64) << (group * 7);

        if byte & 0x80 == 0 {
            if group > 0 && byte & 0x7f == 0 {
                return Err(BoundedError::NonCanonicalVarint);
            }
            return Ok(value);
        }
    }
    Err(BoundedError::VarintOverflow)
}

fn predict(history: &[i16], predictor: BoundedPredictor) -> i64 {
    match predictor {
        BoundedPredictor::Previous => history.last().copied().unwrap_or(0) as i64,
        BoundedPredictor::Linear => match history {
            [.., previous_previous, previous] => {
                2 * (*previous as i64) - *previous_previous as i64
            }
            _ => history.last().copied().unwrap_or(0) as i64,
        },
    }
}

fn nearest_integer_quotient(residual: i64, step: i64) -> i64 {
    debug_assert!(step > 0);
    let half = step / 2;
    if residual >= 0 {
        (residual + half) / step
    } else {
        -((-residual + half) / step)
    }
}

fn reconstruct(prediction: i64, index: i64, step: i64) -> Result<i16, BoundedError> {
    if index.unsigned_abs() > MAX_ABS_RESIDUAL_INDEX {
        return Err(BoundedError::ResidualOutOfRange);
    }
    let residual = index.checked_mul(step).ok_or(BoundedError::NumericOverflow)?;
    let value = prediction
        .checked_add(residual)
        .ok_or(BoundedError::NumericOverflow)?;
    Ok(value.clamp(i16::MIN as i64, i16::MAX as i64) as i16)
}

/// Encode an i16 sequence with a strict per-sample absolute-error tolerance.
///
/// The predictor only consumes previously reconstructed values. This ensures
/// encoder and decoder follow exactly the same state trajectory.
pub fn compress_bounded_i16(
    values: &[i16],
    tolerance: u16,
    predictor: BoundedPredictor,
    model: ModelKind,
) -> Result<Vec<u8>, BoundedError> {
    if tolerance > i16::MAX as u16 {
        return Err(BoundedError::InvalidTolerance(tolerance));
    }
    let step = 2 * tolerance as i64 + 1;
    let capacity = values
        .len()
        .checked_mul(MAX_INDEX_BYTES)
        .ok_or(BoundedError::LengthOverflow)?;
    let mut indices = Vec::with_capacity(capacity);
    let mut reconstructed = Vec::with_capacity(values.len());

    for &value in values {
        let prediction = predict(&reconstructed, predictor);
        let index = nearest_integer_quotient(value as i64 - prediction, step);
        let decoded = reconstruct(prediction, index, step)?;
        let error = (value as i32 - decoded as i32).unsigned_abs();
        if error > tolerance as u32 {
            return Err(BoundedError::LengthMismatch);
        }
        write_varint(zigzag_encode(index), &mut indices);
        reconstructed.push(decoded);
    }

    let inner = compress(&indices, model)?;
    let count = u64::try_from(values.len()).map_err(|_| BoundedError::LengthOverflow)?;
    let inner_len = u64::try_from(inner.len()).map_err(|_| BoundedError::LengthOverflow)?;
    let mut raw_reconstructed = Vec::with_capacity(
        reconstructed
            .len()
            .checked_mul(2)
            .ok_or(BoundedError::LengthOverflow)?,
    );
    for value in &reconstructed {
        raw_reconstructed.extend_from_slice(&value.to_le_bytes());
    }

    let mut frame = Vec::with_capacity(
        HEADER_LEN.checked_add(inner.len()).ok_or(BoundedError::LengthOverflow)?,
    );
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION.to_le_bytes());
    frame.push(predictor.id());
    frame.push(model.id());
    frame.extend_from_slice(&tolerance.to_le_bytes());
    frame.extend_from_slice(&0u16.to_le_bytes());
    frame.extend_from_slice(&count.to_le_bytes());
    frame.extend_from_slice(&inner_len.to_le_bytes());
    frame.extend_from_slice(&crc32(&raw_reconstructed).to_le_bytes());
    frame.extend_from_slice(&crc32(&inner).to_le_bytes());
    debug_assert_eq!(frame.len(), HEADER_LEN);
    frame.extend_from_slice(&inner);
    Ok(frame)
}

/// Decode a bounded-error frame. The result is an approximation, never the source.
pub fn decompress_bounded_i16(
    frame: &[u8],
    max_values: usize,
) -> Result<Vec<i16>, BoundedError> {
    if frame.len() < HEADER_LEN {
        return Err(BoundedError::TruncatedHeader);
    }
    if &frame[0..4] != MAGIC {
        return Err(BoundedError::BadMagic);
    }

    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(BoundedError::UnsupportedVersion(version));
    }
    let predictor = BoundedPredictor::from_id(frame[6])?;
    let model_id = frame[7];
    if model_id > 1 {
        return Err(BoundedError::UnsupportedModel(model_id));
    }
    let tolerance = u16::from_le_bytes([frame[8], frame[9]]);
    if tolerance > i16::MAX as u16 {
        return Err(BoundedError::InvalidTolerance(tolerance));
    }
    let flags = u16::from_le_bytes([frame[10], frame[11]]);
    if flags != 0 {
        return Err(BoundedError::UnsupportedFlags(flags));
    }
    let count = u64::from_le_bytes(
        frame[12..20]
            .try_into()
            .map_err(|_| BoundedError::LengthMismatch)?,
    );
    let inner_len = u64::from_le_bytes(
        frame[20..28]
            .try_into()
            .map_err(|_| BoundedError::LengthMismatch)?,
    );
    let reconstructed_crc = u32::from_le_bytes(
        frame[28..32]
            .try_into()
            .map_err(|_| BoundedError::LengthMismatch)?,
    );
    let inner_crc = u32::from_le_bytes(
        frame[32..36]
            .try_into()
            .map_err(|_| BoundedError::LengthMismatch)?,
    );

    if count > max_values as u64 {
        return Err(BoundedError::OutputLimitExceeded {
            declared: count,
            limit: max_values,
        });
    }
    let count = usize::try_from(count).map_err(|_| BoundedError::LengthOverflow)?;
    let inner_len = usize::try_from(inner_len).map_err(|_| BoundedError::LengthOverflow)?;
    let expected_len = HEADER_LEN
        .checked_add(inner_len)
        .ok_or(BoundedError::LengthOverflow)?;
    if expected_len != frame.len() {
        return Err(BoundedError::LengthMismatch);
    }

    let inner = &frame[HEADER_LEN..];
    if crc32(inner) != inner_crc {
        return Err(BoundedError::ChecksumMismatch);
    }
    if inner.get(6).copied() != Some(model_id) {
        return Err(BoundedError::UnsupportedModel(model_id));
    }

    let max_index_bytes = count
        .checked_mul(MAX_INDEX_BYTES)
        .ok_or(BoundedError::LengthOverflow)?;
    let indices = decompress(inner, max_index_bytes)?;
    let step = 2 * tolerance as i64 + 1;
    let mut values = Vec::with_capacity(count);
    let mut cursor = 0usize;

    for _ in 0..count {
        let index = zigzag_decode(read_varint(&indices, &mut cursor)?);
        let prediction = predict(&values, predictor);
        values.push(reconstruct(prediction, index, step)?);
    }
    if cursor != indices.len() {
        return Err(BoundedError::LengthMismatch);
    }

    let mut raw = Vec::with_capacity(
        values
            .len()
            .checked_mul(2)
            .ok_or(BoundedError::LengthOverflow)?,
    );
    for value in &values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    if crc32(&raw) != reconstructed_crc {
        return Err(BoundedError::ChecksumMismatch);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_e0, E0Kind};

    fn fixtures() -> Vec<Vec<i16>> {
        let random_bytes = generate_e0(E0Kind::Incompressible, 20_000, 0xB0_0D);
        let random = random_bytes
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        vec![
            Vec::new(),
            vec![0],
            vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX],
            vec![1000; 2048],
            (0..2048).map(|x| (x as i16) * 7).collect(),
            (0..2048).map(|x| ((x % 31) as i16) * 9 - 120).collect(),
            random,
        ]
    }

    #[test]
    fn every_reconstruction_respects_tolerance() {
        for tolerance in [0u16, 1, 5, 32, 127, 1024, 32767] {
            for predictor in [BoundedPredictor::Previous, BoundedPredictor::Linear] {
                for model in [ModelKind::Order0, ModelKind::Order1] {
                    for source in fixtures() {
                        let frame = compress_bounded_i16(
                            &source,
                            tolerance,
                            predictor,
                            model,
                        )
                        .unwrap();
                        let decoded = decompress_bounded_i16(&frame, source.len()).unwrap();
                        assert_eq!(decoded.len(), source.len());
                        for (&actual, &reconstructed) in source.iter().zip(&decoded) {
                            assert!(
                                (actual as i32 - reconstructed as i32).unsigned_abs()
                                    <= tolerance as u32
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_tolerance_is_exact() {
        for predictor in [BoundedPredictor::Previous, BoundedPredictor::Linear] {
            for source in fixtures() {
                let frame =
                    compress_bounded_i16(&source, 0, predictor, ModelKind::Order1).unwrap();
                assert_eq!(decompress_bounded_i16(&frame, source.len()).unwrap(), source);
            }
        }
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress_bounded_i16(
            &[0, 1, 2, 4, 8, 16, -16, -8],
            2,
            BoundedPredictor::Linear,
            ModelKind::Order0,
        )
        .unwrap();
        for end in 0..frame.len() {
            assert!(decompress_bounded_i16(&frame[..end], 1024).is_err());
        }
    }

    #[test]
    fn every_single_bit_mutation_is_rejected() {
        let frame = compress_bounded_i16(
            &[0, 1, 2, 4, 8, 16, -16, -8],
            3,
            BoundedPredictor::Previous,
            ModelKind::Order1,
        )
        .unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut mutated = frame.clone();
                mutated[index] ^= 1 << bit;
                assert!(decompress_bounded_i16(&mutated, 1024).is_err());
            }
        }
    }

    #[test]
    fn unsupported_tolerance_and_reconstruction_overflow_reject() {
        assert_eq!(
            compress_bounded_i16(
                &[0, 1],
                u16::MAX,
                BoundedPredictor::Previous,
                ModelKind::Order0
            )
            .unwrap_err(),
            BoundedError::InvalidTolerance(u16::MAX)
        );
        assert_eq!(
            reconstruct(0, i64::MAX, 3),
            Err(BoundedError::ResidualOutOfRange)
        );
        assert_eq!(
            reconstruct(-2, i64::MIN, 1),
            Err(BoundedError::ResidualOutOfRange)
        );
        assert_eq!(
            reconstruct(0, 131_071, 1),
            Err(BoundedError::ResidualOutOfRange)
        );
    }

    #[test]
    fn noncanonical_and_truncated_indices_are_rejected() {
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0x80], &mut cursor),
            Err(BoundedError::TruncatedVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0x80, 0x00], &mut cursor),
            Err(BoundedError::NonCanonicalVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02], &mut cursor),
            Err(BoundedError::VarintOverflow)
        );
    }
}
