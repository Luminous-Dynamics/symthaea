//! Exact IEEE-754 f64 stream framing and reversible transforms.
//!
//! This is a reference baseline for floating-point time-series experiments,
//! not an implementation of Gorilla, Chimp, Rabbit, Dolphin or DeXOR. All
//! values are preserved by their u64 bit patterns, including NaN payloads,
//! infinities, signed zero and subnormals.

use crate::numeric::NumericError;
use crate::{compress, crc32, decompress, CodecError, ModelKind};

const MAGIC: &[u8; 4] = b"F64C";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
const MAX_TRANSFORMED_BYTES_PER_VALUE: usize = 10;

/// Reversible f64 transform. All operations are on IEEE-754 bits, not floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Float64Transform {
    /// Store the original u64 IEEE-754 bit pattern as eight little-endian bytes.
    Raw = 0,
    /// Store the XOR of each bit pattern with the preceding bit pattern as a u64 varint.
    XorPrevious = 1,
}

impl Float64Transform {
    fn from_id(value: u8) -> Result<Self, NumericError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::XorPrevious),
            _ => Err(NumericError::UnsupportedTransform(value)),
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }
}

fn write_varint_u64(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_varint_u64(input: &[u8], cursor: &mut usize) -> Result<u64, NumericError> {
    let mut value = 0u64;
    for byte_index in 0..10usize {
        let byte = *input.get(*cursor).ok_or(NumericError::TruncatedVarint)?;
        *cursor += 1;

        if byte_index == 9 && (byte & 0xfe) != 0 {
            return Err(NumericError::VarintOverflow);
        }
        let shift = (byte_index * 7) as u32;
        value |= ((byte & 0x7f) as u64) << shift;

        if byte & 0x80 == 0 {
            if byte_index > 0 && (byte & 0x7f) == 0 {
                return Err(NumericError::NonCanonicalVarint);
            }
            return Ok(value);
        }
    }
    Err(NumericError::VarintOverflow)
}

fn raw_bits_bytes(values: &[f64]) -> Result<Vec<u8>, NumericError> {
    let capacity = values
        .len()
        .checked_mul(8)
        .ok_or(NumericError::LengthOverflow)?;
    let mut raw = Vec::with_capacity(capacity);
    for value in values {
        raw.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(raw)
}

fn encode_transform(values: &[f64], transform: Float64Transform) -> Result<Vec<u8>, NumericError> {
    let capacity = match transform {
        Float64Transform::Raw => values.len().checked_mul(8).ok_or(NumericError::LengthOverflow)?,
        Float64Transform::XorPrevious => values
            .len()
            .checked_mul(MAX_TRANSFORMED_BYTES_PER_VALUE)
            .ok_or(NumericError::LengthOverflow)?,
    };
    let mut output = Vec::with_capacity(capacity);
    match transform {
        Float64Transform::Raw => {
            for value in values {
                output.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        Float64Transform::XorPrevious => {
            let mut previous = 0u64;
            for value in values {
                let current = value.to_bits();
                write_varint_u64(current ^ previous, &mut output);
                previous = current;
            }
        }
    }
    Ok(output)
}

fn decode_transform(
    bytes: &[u8],
    transform: Float64Transform,
    count: usize,
) -> Result<Vec<f64>, NumericError> {
    let mut values = Vec::with_capacity(count);
    let mut cursor = 0usize;
    match transform {
        Float64Transform::Raw => {
            let expected = count.checked_mul(8).ok_or(NumericError::LengthOverflow)?;
            if bytes.len() != expected {
                return Err(NumericError::LengthMismatch);
            }
            for chunk in bytes.chunks_exact(8) {
                let bits = u64::from_le_bytes(
                    chunk.try_into().map_err(|_| NumericError::LengthMismatch)?,
                );
                values.push(f64::from_bits(bits));
            }
        }
        Float64Transform::XorPrevious => {
            let mut previous = 0u64;
            for _ in 0..count {
                let residual = read_varint_u64(bytes, &mut cursor)?;
                let current = previous ^ residual;
                values.push(f64::from_bits(current));
                previous = current;
            }
            if cursor != bytes.len() {
                return Err(NumericError::LengthMismatch);
            }
        }
    }
    Ok(values)
}

/// Encode f64 values with exact bit preservation and a shared byte entropy coder.
pub fn compress_f64(
    values: &[f64],
    transform: Float64Transform,
    model: ModelKind,
) -> Result<Vec<u8>, NumericError> {
    let transformed = encode_transform(values, transform)?;
    let inner = compress(&transformed, model)?;
    let count = u64::try_from(values.len()).map_err(|_| NumericError::LengthOverflow)?;
    let inner_len = u64::try_from(inner.len()).map_err(|_| NumericError::LengthOverflow)?;
    let raw = raw_bits_bytes(values)?;

    let frame_capacity = HEADER_LEN
        .checked_add(inner.len())
        .ok_or(NumericError::LengthOverflow)?;
    let mut frame = Vec::with_capacity(frame_capacity);
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

/// Decode an f64 frame, enforcing its element-count limit before allocation.
pub fn decompress_f64(frame: &[u8], max_values: usize) -> Result<Vec<f64>, NumericError> {
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
    let transform = Float64Transform::from_id(frame[6])?;
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
    if crc32(&raw_bits_bytes(&values)?) != raw_crc {
        return Err(NumericError::ChecksumMismatch);
    }
    Ok(values)
}

const SEGMENTED_MAGIC: &[u8; 4] = b"F64S";
const SEGMENTED_VERSION: u16 = 1;
const SEGMENTED_HEADER_LEN: usize = 32;
const SEGMENT_SELECTION_POLICY_BEST_OF_FOUR: u8 = 1;

/// Encode fixed-size f64 chunks by selecting the smallest complete subframe
/// among Raw/XorPrevious x Order0/Order1. Every selected F64C frame (including
/// its own header and CRCs) is retained in the output, so selection overhead is
/// included. This is an offline segment-local size oracle, not an online or
/// WZS-equivalent algorithm.
pub fn compress_f64_segmented(
    values: &[f64],
    segment_size: usize,
) -> Result<Vec<u8>, NumericError> {
    if segment_size == 0 {
        return Err(NumericError::InvalidSegmentSize);
    }
    let segment_count = if values.is_empty() {
        0
    } else {
        (values.len() - 1) / segment_size + 1
    };
    let segment_size_u32 =
        u32::try_from(segment_size).map_err(|_| NumericError::LengthOverflow)?;
    let segment_count_u32 =
        u32::try_from(segment_count).map_err(|_| NumericError::LengthOverflow)?;
    let count = u64::try_from(values.len()).map_err(|_| NumericError::LengthOverflow)?;
    let raw = raw_bits_bytes(values)?;

    let mut frame = Vec::new();
    frame.extend_from_slice(SEGMENTED_MAGIC);
    frame.extend_from_slice(&SEGMENTED_VERSION.to_le_bytes());
    frame.push(0); // flags
    frame.push(SEGMENT_SELECTION_POLICY_BEST_OF_FOUR);
    frame.extend_from_slice(&count.to_le_bytes());
    frame.extend_from_slice(&segment_size_u32.to_le_bytes());
    frame.extend_from_slice(&segment_count_u32.to_le_bytes());
    frame.extend_from_slice(&crc32(&raw).to_le_bytes());
    let header_crc = crc32(&frame);
    frame.extend_from_slice(&header_crc.to_le_bytes());
    debug_assert_eq!(frame.len(), SEGMENTED_HEADER_LEN);

    for segment in values.chunks(segment_size) {
        let mut best: Option<Vec<u8>> = None;
        for transform in [Float64Transform::Raw, Float64Transform::XorPrevious] {
            for model in [ModelKind::Order0, ModelKind::Order1] {
                let candidate = compress_f64(segment, transform, model)?;
                if best
                    .as_ref()
                    .map_or(true, |current| candidate.len() < current.len())
                {
                    best = Some(candidate);
                }
            }
        }
        let best = best.ok_or(NumericError::LengthMismatch)?;
        frame.extend_from_slice(&best);
    }
    Ok(frame)
}

/// Decode a segmented F64S frame. Segment sizes are fixed by the outer header;
/// each self-describing F64C child frame identifies its chosen transform/model.
pub fn decompress_f64_segmented(
    frame: &[u8],
    max_values: usize,
) -> Result<Vec<f64>, NumericError> {
    if frame.len() < SEGMENTED_HEADER_LEN {
        return Err(NumericError::TruncatedHeader);
    }
    if &frame[0..4] != SEGMENTED_MAGIC {
        return Err(NumericError::BadMagic);
    }
    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != SEGMENTED_VERSION {
        return Err(NumericError::UnsupportedVersion(version));
    }
    if frame[6] != 0 {
        return Err(NumericError::UnsupportedFlags(frame[6]));
    }
    if frame[7] != SEGMENT_SELECTION_POLICY_BEST_OF_FOUR {
        return Err(NumericError::UnsupportedTransform(frame[7]));
    }

    let count = u64::from_le_bytes(
        frame[8..16]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );
    let segment_size = u32::from_le_bytes(
        frame[16..20]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    ) as usize;
    let segment_count = u32::from_le_bytes(
        frame[20..24]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    ) as usize;
    let raw_crc = u32::from_le_bytes(
        frame[24..28]
            .try_into()
            .map_err(|_| NumericError::LengthMismatch)?,
    );
    let header_crc = u32::from_le_bytes(
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
    if segment_size == 0 {
        return Err(NumericError::InvalidSegmentSize);
    }
    if crc32(&frame[..28]) != header_crc {
        return Err(NumericError::ChecksumMismatch);
    }
    let count = usize::try_from(count).map_err(|_| NumericError::LengthOverflow)?;
    let expected_segments = if count == 0 {
        0
    } else {
        (count - 1) / segment_size + 1
    };
    if segment_count != expected_segments {
        return Err(NumericError::LengthMismatch);
    }

    let mut output = Vec::with_capacity(count);
    let mut cursor = SEGMENTED_HEADER_LEN;
    for segment_index in 0..segment_count {
        let header_end = cursor
            .checked_add(HEADER_LEN)
            .ok_or(NumericError::LengthOverflow)?;
        if header_end > frame.len() {
            return Err(NumericError::TruncatedHeader);
        }
        if &frame[cursor..cursor + 4] != MAGIC {
            return Err(NumericError::BadMagic);
        }
        let inner_len = u64::from_le_bytes(
            frame[cursor + 16..cursor + 24]
                .try_into()
                .map_err(|_| NumericError::LengthMismatch)?,
        );
        let inner_len = usize::try_from(inner_len).map_err(|_| NumericError::LengthOverflow)?;
        let child_len = HEADER_LEN
            .checked_add(inner_len)
            .ok_or(NumericError::LengthOverflow)?;
        let child_end = cursor
            .checked_add(child_len)
            .ok_or(NumericError::LengthOverflow)?;
        if child_end > frame.len() {
            return Err(NumericError::LengthMismatch);
        }

        let expected_values = if segment_index + 1 == segment_count {
            count - output.len()
        } else {
            segment_size
        };
        let decoded = decompress_f64(&frame[cursor..child_end], expected_values)?;
        if decoded.len() != expected_values {
            return Err(NumericError::LengthMismatch);
        }
        output.extend_from_slice(&decoded);
        cursor = child_end;
    }

    if cursor != frame.len() || output.len() != count {
        return Err(NumericError::LengthMismatch);
    }
    if crc32(&raw_bits_bytes(&output)?) != raw_crc {
        return Err(NumericError::ChecksumMismatch);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_e0;
    use crate::E0Kind;

    fn bit_fixtures() -> Vec<Vec<f64>> {
        let edge_bits = [
            0x0000_0000_0000_0000u64, // +0
            0x8000_0000_0000_0000u64, // -0
            0x7ff0_0000_0000_0000u64, // +infinity
            0xfff0_0000_0000_0000u64, // -infinity
            0x0000_0000_0000_0001u64, // minimum positive subnormal
            0x0010_0000_0000_0000u64, // minimum positive normal
            0x7ff8_0000_0000_0001u64, // NaN with payload
            0xfff8_1234_5678_9abcu64, // negative NaN with payload
            0x3ff0_0000_0000_0000u64, // 1.0
        ];
        let edges = edge_bits.into_iter().map(f64::from_bits).collect::<Vec<_>>();
        let constant = vec![f64::from_bits(0x405e_0000_0000_0000); 2048];
        let smooth = (0..2048)
            .map(|index| f64::from_bits(0x3ff0_0000_0000_0000 + (index as u64 * 17)))
            .collect::<Vec<_>>();
        let oscillating = (0..2048)
            .map(|index| {
                let magnitude = 0x3ff0_0000_0000_0000u64 + ((index as u64 * 7919) & 0x000f_ffff);
                f64::from_bits(magnitude | if index % 2 == 0 { 0 } else { 1u64 << 63 })
            })
            .collect::<Vec<_>>();
        let random_bytes = generate_e0(E0Kind::Incompressible, 4096 * 8, 0xf64c);
        let random_bits = random_bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .map(f64::from_bits)
            .collect::<Vec<_>>();
        vec![edges, constant, smooth, oscillating, random_bits]
    }

    fn assert_same_bits(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (index, (a, b)) in left.iter().zip(right).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "f64 bits differ at index {index}");
        }
    }

    #[test]
    fn raw_and_xor_round_trip_all_bit_patterns() {
        for transform in [Float64Transform::Raw, Float64Transform::XorPrevious] {
            for model in [ModelKind::Order0, ModelKind::Order1] {
                for input in bit_fixtures() {
                    let frame = compress_f64(&input, transform, model).unwrap();
                    let output = decompress_f64(&frame, input.len()).unwrap();
                    assert_same_bits(&output, &input);
                }
            }
        }
    }

    #[test]
    fn f64_encoding_is_deterministic() {
        let input = bit_fixtures().pop().unwrap();
        let first = compress_f64(&input, Float64Transform::XorPrevious, ModelKind::Order1).unwrap();
        let second = compress_f64(&input, Float64Transform::XorPrevious, ModelKind::Order1).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress_f64(
            &[0.0, -0.0, f64::INFINITY, f64::from_bits(0x7ff8_0000_0000_0001)],
            Float64Transform::XorPrevious,
            ModelKind::Order0,
        )
        .unwrap();
        for end in 0..frame.len() {
            assert!(decompress_f64(&frame[..end], 100).is_err());
        }
    }

    #[test]
    fn every_single_bit_mutation_is_rejected() {
        let input = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(1),
        ];
        let frame = compress_f64(&input, Float64Transform::XorPrevious, ModelKind::Order1).unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut changed = frame.clone();
                changed[index] ^= 1 << bit;
                assert!(decompress_f64(&changed, input.len()).is_err());
            }
        }
    }

    #[test]
    fn output_limit_is_enforced_before_decoding() {
        let frame = compress_f64(&[1.0, 2.0, 3.0], Float64Transform::Raw, ModelKind::Order0).unwrap();
        assert!(matches!(
            decompress_f64(&frame, 2),
            Err(NumericError::OutputLimitExceeded { .. })
        ));
    }

    #[test]
    fn segmented_portfolio_round_trips_with_all_overhead_counted() {
        for input in bit_fixtures() {
            for segment_size in [1, 64, 256, 1024] {
                let frame = compress_f64_segmented(&input, segment_size).unwrap();
                let decoded = decompress_f64_segmented(&frame, input.len()).unwrap();
                assert_same_bits(&decoded, &input);
            }
        }
    }

    #[test]
    fn segmented_frame_rejects_truncation_and_bad_segment_size() {
        let input = [0.0, -0.0, 1.0, f64::from_bits(0x7ff8_0000_0000_0001)];
        let frame = compress_f64_segmented(&input, 2).unwrap();
        for end in 0..frame.len() {
            assert!(decompress_f64_segmented(&frame[..end], 100).is_err());
        }
        assert_eq!(
            compress_f64_segmented(&input, 0),
            Err(NumericError::InvalidSegmentSize)
        );
    }

    #[test]
    fn segmented_frame_rejects_header_flags_and_single_bit_mutations() {
        let input = [1.0, -0.0, f64::INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
        let frame = compress_f64_segmented(&input, 2).unwrap();

        let mut bad_flags = frame.clone();
        bad_flags[6] = 1;
        assert_eq!(
            decompress_f64_segmented(&bad_flags, input.len()),
            Err(NumericError::UnsupportedFlags(1))
        );

        let mut bad_policy = frame.clone();
        bad_policy[7] = 99;
        assert_eq!(
            decompress_f64_segmented(&bad_policy, input.len()),
            Err(NumericError::UnsupportedTransform(99))
        );

        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut changed = frame.clone();
                changed[index] ^= 1 << bit;
                assert!(
                    decompress_f64_segmented(&changed, input.len()).is_err(),
                    "segmented mutation at byte {index}, bit {bit} was accepted"
                );
            }
        }
    }

    #[test]
    fn segmented_output_limit_is_enforced() {
        let frame = compress_f64_segmented(&[1.0, 2.0, 3.0], 2).unwrap();
        assert!(matches!(
            decompress_f64_segmented(&frame, 2),
            Err(NumericError::OutputLimitExceeded { .. })
        ));
    }

    #[test]
    fn malformed_u64_varints_are_rejected() {
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0x80], &mut cursor),
            Err(NumericError::TruncatedVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0x80, 0x00], &mut cursor),
            Err(NumericError::NonCanonicalVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02], &mut cursor),
            Err(NumericError::VarintOverflow)
        );
    }
}
