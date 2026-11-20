//! Typed, lossless numerical transforms layered over the CDL byte coder.
//!
//! v1 supports signed i16 streams and four exact transforms. Transform IDs and
//! element counts are framed explicitly; every path reconstructs the original
//! little-endian i16 bit patterns exactly.

use crate::{compress, crc32, decompress, CodecError, ModelKind};
use std::fmt;

const MAGIC: &[u8; 4] = b"N16C";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
const MAX_TRANSFORMED_BYTES_PER_VALUE: usize = 5;

/// Reversible transform applied before entropy coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NumericTransform {
    /// Store each i16 as its original two little-endian bytes.
    Raw = 0,
    /// Encode first differences from the previous value (initial previous = 0).
    Delta = 1,
    /// Encode differences of first differences (both initial values = 0).
    DeltaOfDelta = 2,
    /// Encode XOR of the current i16 bit pattern with the previous pattern.
    XorPrevious = 3,
}

impl NumericTransform {
    fn from_id(value: u8) -> Result<Self, NumericError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Delta),
            2 => Ok(Self::DeltaOfDelta),
            3 => Ok(Self::XorPrevious),
            _ => Err(NumericError::UnsupportedTransform(value)),
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }
}

/// Errors in the typed numerical frame or reversible transform.
#[derive(Debug, PartialEq, Eq)]
pub enum NumericError {
    Codec(CodecError),
    TruncatedHeader,
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedTransform(u8),
    UnsupportedModel(u8),
    LengthOverflow,
    LengthMismatch,
    OutputLimitExceeded { declared: u64, limit: usize },
    ChecksumMismatch,
    TruncatedVarint,
    VarintOverflow,
    NonCanonicalVarint,
    NumericOverflow,
    InvalidSegmentSize,
    UnsupportedFlags(u8),
    SegmentPolicyMismatch { expected: u8, actual: u8 },
    InvalidProbeInterval,
    UnsupportedPolicy(u8),
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(f, "inner CDL codec error: {error}"),
            Self::TruncatedHeader => write!(f, "truncated typed numerical frame header"),
            Self::BadMagic => write!(f, "invalid typed numerical frame magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported numerical frame version {v}"),
            Self::UnsupportedTransform(v) => write!(f, "unsupported numerical transform id {v}"),
            Self::UnsupportedModel(v) => write!(f, "unsupported numerical entropy model id {v}"),
            Self::LengthOverflow => write!(f, "typed numerical frame length overflow"),
            Self::LengthMismatch => write!(f, "typed numerical frame length mismatch"),
            Self::OutputLimitExceeded { declared, limit } => {
                write!(f, "declared element count {declared} exceeds limit {limit}")
            }
            Self::ChecksumMismatch => write!(f, "typed numerical frame checksum mismatch"),
            Self::TruncatedVarint => write!(f, "truncated unsigned varint"),
            Self::VarintOverflow => write!(f, "unsigned varint exceeds u64"),
            Self::NonCanonicalVarint => write!(f, "non-canonical overlong unsigned varint"),
            Self::NumericOverflow => write!(f, "decoded numeric value exceeds the supported range"),
            Self::InvalidSegmentSize => write!(f, "segment size must be positive"),
            Self::UnsupportedFlags(v) => write!(f, "unsupported numeric frame flags {v}"),
            Self::SegmentPolicyMismatch { expected, actual } => write!(f, "segment policy mismatch: expected candidate {expected}, got {actual}"),
            Self::InvalidProbeInterval => write!(f, "probe interval must be a power of two from 1 through 128"),
            Self::UnsupportedPolicy(v) => write!(f, "unsupported F64S segment policy id {v}"),
        }
    }
}

impl std::error::Error for NumericError {}

impl From<CodecError> for NumericError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

fn zigzag_encode(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

fn zigzag_decode(value: u32) -> i64 {
    (((value >> 1) as i32) ^ (-((value & 1) as i32))) as i64
}

fn write_varint(mut value: u32, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push(((value & 0x7f) as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_varint(input: &[u8], cursor: &mut usize) -> Result<u32, NumericError> {
    let mut value = 0u32;
    for shift in (0..=28).step_by(7) {
        let byte = *input.get(*cursor).ok_or(NumericError::TruncatedVarint)?;
        *cursor += 1;

        if shift == 28 && (byte & 0xf0) != 0 {
            return Err(NumericError::VarintOverflow);
        }
        value |= ((byte & 0x7f) as u32) << shift;

        if (byte & 0x80) == 0 {
            if shift != 0 && (byte & 0x7f) == 0 {
                return Err(NumericError::NonCanonicalVarint);
            }
            return Ok(value);
        }
    }
    Err(NumericError::VarintOverflow)
}

fn encode_transform(values: &[i16], transform: NumericTransform) -> Result<Vec<u8>, NumericError> {
    let capacity = match transform {
        NumericTransform::Raw => values.len().checked_mul(2).ok_or(NumericError::LengthOverflow)?,
        _ => values
            .len()
            .checked_mul(MAX_TRANSFORMED_BYTES_PER_VALUE)
            .ok_or(NumericError::LengthOverflow)?,
    };
    let mut output = Vec::with_capacity(capacity);

    match transform {
        NumericTransform::Raw => {
            for value in values {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        NumericTransform::Delta => {
            let mut previous = 0i32;
            for &value in values {
                let current = value as i32;
                write_varint(zigzag_encode(current - previous), &mut output);
                previous = current;
            }
        }
        NumericTransform::DeltaOfDelta => {
            let mut previous = 0i32;
            let mut previous_delta = 0i32;
            for &value in values {
                let current = value as i32;
                let delta = current - previous;
                let delta_delta = delta - previous_delta;
                write_varint(zigzag_encode(delta_delta), &mut output);
                previous = current;
                previous_delta = delta;
            }
        }
        NumericTransform::XorPrevious => {
            let mut previous = 0u16;
            for &value in values {
                let current = value as u16;
                write_varint((current ^ previous) as u32, &mut output);
                previous = current;
            }
        }
    }
    Ok(output)
}

fn decode_transform(
    bytes: &[u8],
    transform: NumericTransform,
    count: usize,
) -> Result<Vec<i16>, NumericError> {
    let mut output = Vec::with_capacity(count);
    let mut cursor = 0usize;

    match transform {
        NumericTransform::Raw => {
            let expected = count.checked_mul(2).ok_or(NumericError::LengthOverflow)?;
            if bytes.len() != expected {
                return Err(NumericError::LengthMismatch);
            }
            for chunk in bytes.chunks_exact(2) {
                output.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        NumericTransform::Delta => {
            let mut previous = 0i64;
            for _ in 0..count {
                let delta = zigzag_decode(read_varint(bytes, &mut cursor)?);
                let current = previous + delta;
                if current < i16::MIN as i64 || current > i16::MAX as i64 {
                    return Err(NumericError::NumericOverflow);
                }
                output.push(current as i16);
                previous = current;
            }
        }
        NumericTransform::DeltaOfDelta => {
            let mut previous = 0i64;
            let mut previous_delta = 0i64;
            for _ in 0..count {
                let delta_delta = zigzag_decode(read_varint(bytes, &mut cursor)?);
                let delta = previous_delta + delta_delta;
                let current = previous + delta;
                if current < i16::MIN as i64 || current > i16::MAX as i64 {
                    return Err(NumericError::NumericOverflow);
                }
                output.push(current as i16);
                previous = current;
                previous_delta = delta;
            }
        }
        NumericTransform::XorPrevious => {
            let mut previous = 0u16;
            for _ in 0..count {
                let residual = read_varint(bytes, &mut cursor)?;
                if residual > u16::MAX as u32 {
                    return Err(NumericError::NumericOverflow);
                }
                let current = previous ^ residual as u16;
                output.push(current as i16);
                previous = current;
            }
        }
    }

    if transform != NumericTransform::Raw && cursor != bytes.len() {
        return Err(NumericError::LengthMismatch);
    }
    Ok(output)
}

/// Encode an i16 slice using a reversible transform and the shared CDL entropy coder.
pub fn compress_i16(
    values: &[i16],
    transform: NumericTransform,
    model: ModelKind,
) -> Result<Vec<u8>, NumericError> {
    let transformed = encode_transform(values, transform)?;
    let inner = compress(&transformed, model)?;

    let count = u64::try_from(values.len()).map_err(|_| NumericError::LengthOverflow)?;
    let inner_len = u64::try_from(inner.len()).map_err(|_| NumericError::LengthOverflow)?;
    let raw_len = values.len().checked_mul(2).ok_or(NumericError::LengthOverflow)?;
    let mut raw = Vec::with_capacity(raw_len);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }

    let mut frame = Vec::with_capacity(
        HEADER_LEN.checked_add(inner.len()).ok_or(NumericError::LengthOverflow)?,
    );
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION.to_le_bytes());
    frame.push(transform.id());
    frame.push(model.id());
    frame.extend_from_slice(&count.to_le_bytes());
    frame.extend_from_slice(&inner_len.to_le_bytes());
    frame.extend_from_slice(&crc32(&raw).to_le_bytes());
    frame.extend_from_slice(&crc32(&inner).to_le_bytes());
    debug_assert_eq!(frame.len(), HEADER_LEN);
    frame.extend_from_slice(&inner);
    Ok(frame)
}

/// Decode a typed i16 frame, enforcing an element-count limit before allocation.
pub fn decompress_i16(frame: &[u8], max_values: usize) -> Result<Vec<i16>, NumericError> {
    if frame.len() < HEADER_LEN {
        return Err(NumericError::TruncatedHeader);
    }
    if &frame[0..4] != MAGIC {
        return Err(NumericError::BadMagic);
    }

    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(NumericError::UnsupportedVersion(version));
    }
    let transform = NumericTransform::from_id(frame[6])?;
    let model_id = frame[7];
    if model_id > 1 {
        return Err(NumericError::UnsupportedModel(model_id));
    }

    let count = u64::from_le_bytes(
        frame[8..16]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );
    let inner_len = u64::from_le_bytes(
        frame[16..24]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );
    let raw_crc = u32::from_le_bytes(
        frame[24..28]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );
    let inner_crc = u32::from_le_bytes(
        frame[28..32]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );

    if count > max_values as u64 {
        return Err(NumericError::OutputLimitExceeded {
            declared: count,
            limit: max_values,
        });
    }
    let count = usize::try_from(count).map_err(|_| NumericError::LengthOverflow)?;
    let inner_len = usize::try_from(inner_len).map_err(|_| NumericError::LengthOverflow)?;
    let expected_len = HEADER_LEN
        .checked_add(inner_len)
        .ok_or(NumericError::LengthOverflow)?;
    if expected_len != frame.len() {
        return Err(NumericError::LengthMismatch);
    }

    let inner = &frame[HEADER_LEN..];
    if crc32(inner) != inner_crc {
        return Err(NumericError::ChecksumMismatch);
    }
    if inner.get(6).copied() != Some(model_id) {
        return Err(NumericError::UnsupportedModel(model_id));
    }

    let max_transformed_bytes = count
        .checked_mul(MAX_TRANSFORMED_BYTES_PER_VALUE)
        .ok_or(NumericError::LengthOverflow)?;
    let transformed = decompress(inner, max_transformed_bytes)?;
    let values = decode_transform(&transformed, transform, count)?;

    let raw_len = count.checked_mul(2).ok_or(NumericError::LengthOverflow)?;
    let mut raw = Vec::with_capacity(raw_len);
    for value in &values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    if crc32(&raw) != raw_crc {
        return Err(NumericError::ChecksumMismatch);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_e0;
    use crate::E0Kind;

    fn fixtures() -> Vec<Vec<i16>> {
        let random_bytes = generate_e0(E0Kind::Incompressible, 20_000, 0x16_16);
        let random: Vec<i16> = random_bytes
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let ramp: Vec<i16> = (0..4096)
            .map(|index| ((index * 13) % 60_000) as i32 - 30_000)
            .map(|value| value as i16)
            .collect();
        vec![
            Vec::new(),
            vec![0],
            vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX],
            vec![1234; 4096],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![30000, 29999, 29998, -30000, -29999, -29998],
            ramp,
            random,
        ]
    }

    #[test]
    fn every_transform_and_model_round_trips_all_fixtures() {
        for transform in [
            NumericTransform::Raw,
            NumericTransform::Delta,
            NumericTransform::DeltaOfDelta,
            NumericTransform::XorPrevious,
        ] {
            for model in [ModelKind::Order0, ModelKind::Order1] {
                for input in fixtures() {
                    let frame = compress_i16(&input, transform, model).unwrap();
                    let output = decompress_i16(&frame, input.len()).unwrap();
                    assert_eq!(output, input, "{transform:?} / {model:?}");
                }
            }
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let input = fixtures().pop().unwrap();
        assert_eq!(
            compress_i16(&input, NumericTransform::DeltaOfDelta, ModelKind::Order1).unwrap(),
            compress_i16(&input, NumericTransform::DeltaOfDelta, ModelKind::Order1).unwrap()
        );
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress_i16(
            &[0, 1, 2, 3, 4, 5, -1, -2, 30000, i16::MIN],
            NumericTransform::Delta,
            ModelKind::Order0,
        )
        .unwrap();
        for end in 0..frame.len() {
            assert!(decompress_i16(&frame[..end], 100).is_err());
        }
    }

    #[test]
    fn every_single_bit_mutation_is_rejected() {
        let frame = compress_i16(
            &[0, 1, 2, 3, 4, 5, -1, -2, 30000, i16::MIN],
            NumericTransform::DeltaOfDelta,
            ModelKind::Order1,
        )
        .unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut mutated = frame.clone();
                mutated[index] ^= 1 << bit;
                assert!(decompress_i16(&mutated, 100).is_err());
            }
        }
    }

    #[test]
    fn malformed_varints_are_rejected() {
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0x80], &mut cursor),
            Err(NumericError::TruncatedVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0x80, 0x00], &mut cursor),
            Err(NumericError::NonCanonicalVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint(&[0xff, 0xff, 0xff, 0xff, 0x1f], &mut cursor),
            Err(NumericError::VarintOverflow)
        );
    }

    #[test]
    fn decoded_integer_overflow_is_rejected() {
        let payload = [0xff, 0xff, 0xff, 0xff, 0x0f];
        assert_eq!(
            decode_transform(&payload, NumericTransform::Delta, 1),
            Err(NumericError::NumericOverflow)
        );
    }

    #[test]
    fn output_limit_is_enforced_before_decoding() {
        let frame = compress_i16(&[1, 2, 3, 4], NumericTransform::Raw, ModelKind::Order0).unwrap();
        assert!(matches!(
            decompress_i16(&frame, 2),
            Err(NumericError::OutputLimitExceeded { .. })
        ));
    }
}
