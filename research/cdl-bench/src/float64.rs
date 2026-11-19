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
const MAX_TRANSFORMED_BYTES_PER_VALUE: usize = 11;

/// Reversible f64 transform. All operations are on IEEE-754 bits, not floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Float64Transform {
    /// Store the original u64 IEEE-754 bit pattern as eight little-endian bytes.
    Raw = 0,
    /// Store the XOR of each bit pattern with the preceding bit pattern as a u64 varint.
    XorPrevious = 1,
    /// Store XOR residual leading/trailing zero counts plus packed meaningful bits.
    XorLeadingTrailing = 2,
}

impl Float64Transform {
    fn from_id(value: u8) -> Result<Self, NumericError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::XorPrevious),
            2 => Ok(Self::XorLeadingTrailing),
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
        Float64Transform::XorLeadingTrailing => {
            let mut previous = 0u64;
            let mut window: Option<(u32, u32)> = None;
            for value in values {
                let current = value.to_bits();
                let residual = current ^ previous;
                if residual == 0 {
                    output.push(0); // zero XOR residual
                } else {
                    let actual_leading = residual.leading_zeros();
                    let actual_trailing = residual.trailing_zeros();
                    let reuse = window
                        .map(|(leading, trailing)| {
                            actual_leading >= leading && actual_trailing >= trailing
                        })
                        .unwrap_or(false);

                    let (leading, trailing) = if reuse {
                        output.push(1); // reuse prior window
                        window.expect("reuse requires an initialized window")
                    } else {
                        output.push(2); // define a new window
                        let next = (actual_leading, actual_trailing);
                        output.push(next.0 as u8);
                        output.push(next.1 as u8);
                        window = Some(next);
                        next
                    };

                    let width = 64 - leading - trailing;
                    let meaningful = residual >> trailing;
                    let payload_bytes = (width as usize + 7) / 8;
                    for byte in 0..payload_bytes {
                        output.push((meaningful >> (byte * 8)) as u8);
                    }
                }
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
        Float64Transform::XorLeadingTrailing => {
            let mut previous = 0u64;
            let mut window: Option<(u32, u32)> = None;
            for _ in 0..count {
                let marker = *bytes.get(cursor).ok_or(NumericError::TruncatedVarint)?;
                cursor += 1;
                let residual = match marker {
                    0 => 0,
                    1 | 2 => {
                        let (leading, trailing) = if marker == 2 {
                            let leading =
                                *bytes.get(cursor).ok_or(NumericError::TruncatedVarint)? as u32;
                            cursor += 1;
                            let trailing =
                                *bytes.get(cursor).ok_or(NumericError::TruncatedVarint)? as u32;
                            cursor += 1;
                            if leading >= 64 || trailing >= 64 || leading + trailing >= 64 {
                                return Err(NumericError::NumericOverflow);
                            }
                            window = Some((leading, trailing));
                            (leading, trailing)
                        } else {
                            window.ok_or(NumericError::NonCanonicalVarint)?
                        };

                        let width = 64 - leading - trailing;
                        let payload_bytes = (width as usize + 7) / 8;
                        let end = cursor
                            .checked_add(payload_bytes)
                            .ok_or(NumericError::LengthOverflow)?;
                        if end > bytes.len() {
                            return Err(NumericError::TruncatedVarint);
                        }
                        let mut meaningful = 0u64;
                        for byte in 0..payload_bytes {
                            meaningful |= (bytes[cursor + byte] as u64) << (byte * 8);
                        }
                        cursor = end;

                        if width % 8 != 0 {
                            let used = width % 8;
                            if (bytes[end - 1] >> used) != 0 {
                                return Err(NumericError::NonCanonicalVarint);
                            }
                        }
                        if meaningful == 0 {
                            return Err(NumericError::NonCanonicalVarint);
                        }

                        let residual = meaningful << trailing;
                        if marker == 2 {
                            if (meaningful & 1) == 0
                                || ((meaningful >> (width - 1)) & 1) == 0
                                || residual.leading_zeros() != leading
                                || residual.trailing_zeros() != trailing
                            {
                                return Err(NumericError::NonCanonicalVarint);
                            }
                        } else if residual.leading_zeros() < leading
                            || residual.trailing_zeros() < trailing
                        {
                            return Err(NumericError::NonCanonicalVarint);
                        }
                        residual
                    }
                    other => return Err(NumericError::UnsupportedTransform(other)),
                };
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
const SEGMENT_SELECTION_POLICY_BEST_OF_SIX: u8 = 1;

const SEGMENT_SELECTION_POLICY_HISTORICAL_COST: u8 = 2;
const SEGMENT_CANDIDATES: usize = 6;

fn encode_segment_candidate(
    values: &[f64],
    candidate: u8,
) -> Result<Vec<u8>, NumericError> {
    let (transform, model) = match candidate {
        0 => (Float64Transform::Raw, ModelKind::Order0),
        1 => (Float64Transform::Raw, ModelKind::Order1),
        2 => (Float64Transform::XorPrevious, ModelKind::Order0),
        3 => (Float64Transform::XorPrevious, ModelKind::Order1),
        4 => (Float64Transform::XorLeadingTrailing, ModelKind::Order0),
        5 => (Float64Transform::XorLeadingTrailing, ModelKind::Order1),
        _ => return Err(NumericError::UnsupportedTransform(candidate)),
    };
    compress_f64(values, transform, model)
}

fn segment_candidate_from_child_header(child: &[u8]) -> Result<u8, NumericError> {
    if child.len() < HEADER_LEN {
        return Err(NumericError::TruncatedHeader);
    }
    if &child[0..4] != MAGIC {
        return Err(NumericError::BadMagic);
    }
    let version = u16::from_le_bytes([child[4], child[5]]);
    if version != VERSION {
        return Err(NumericError::UnsupportedVersion(version));
    }
    let transform = child[6];
    let model = child[7];
    if model > 1 {
        return Err(NumericError::UnsupportedModel(model));
    }
    match (transform, model) {
        (0, 0) => Ok(0),
        (0, 1) => Ok(1),
        (1, 0) => Ok(2),
        (1, 1) => Ok(3),
        (2, 0) => Ok(4),
        (2, 1) => Ok(5),
        _ => Err(NumericError::UnsupportedTransform(transform)),
    }
}

#[derive(Default)]
struct HistoricalSegmentCosts {
    totals: [u64; SEGMENT_CANDIDATES],
    segments: u64,
}

impl HistoricalSegmentCosts {
    fn choose(&self) -> u8 {
        if self.segments == 0 {
            return 0;
        }
        let mut best = 0usize;
        for candidate in 1..SEGMENT_CANDIDATES {
            let candidate_average = self.totals[candidate] / self.segments;
            let best_average = self.totals[best] / self.segments;
            if candidate_average < best_average {
                best = candidate;
            }
        }
        best as u8
    }

    fn observe_sizes(
        &mut self,
        sizes: [u64; SEGMENT_CANDIDATES],
    ) -> Result<(), NumericError> {
        for (candidate, size) in sizes.into_iter().enumerate() {
            self.totals[candidate] = self.totals[candidate]
                .checked_add(size)
                .ok_or(NumericError::LengthOverflow)?;
        }
        self.segments = self.segments
            .checked_add(1)
            .ok_or(NumericError::LengthOverflow)?;
        Ok(())
    }

    fn observe(&mut self, values: &[f64]) -> Result<(), NumericError> {
        let mut sizes = [0u64; SEGMENT_CANDIDATES];
        for candidate in 0..SEGMENT_CANDIDATES {
            let frame = encode_segment_candidate(values, candidate as u8)?;
            sizes[candidate] =
                u64::try_from(frame.len()).map_err(|_| NumericError::LengthOverflow)?;
        }
        self.observe_sizes(sizes)
    }
}

/// Encode fixed-size f64 chunks using a deterministic historical-cost policy.
///
/// The first chunk uses Raw/Order0 (candidate 0). Before each later chunk, the
/// selector chooses the candidate with the lowest *mean complete child-frame
/// size on previously decoded chunks*. It never uses the current chunk to
/// choose its encoding. After encoding a chunk, all four candidate sizes are
/// measured to update the policy state for the next chunk.
///
/// The decoder repeats those four measurements from the reconstructed chunk,
/// derives the same next policy state, and rejects a child whose transform/model
/// ID does not match the policy's expected choice. The selected child header
/// already carries the transform/model ID; no additional selector message is
/// sent. This is an online policy with high diagnostic compute cost, not an
/// optimized implementation.
pub fn compress_f64_online_selector(
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
    frame.push(0);
    frame.push(SEGMENT_SELECTION_POLICY_HISTORICAL_COST);
    frame.extend_from_slice(&count.to_le_bytes());
    frame.extend_from_slice(&segment_size_u32.to_le_bytes());
    frame.extend_from_slice(&segment_count_u32.to_le_bytes());
    frame.extend_from_slice(&crc32(&raw).to_le_bytes());
    let header_crc = crc32(&frame);
    frame.extend_from_slice(&header_crc.to_le_bytes());
    debug_assert_eq!(frame.len(), SEGMENTED_HEADER_LEN);

    let mut history = HistoricalSegmentCosts::default();
    for segment in values.chunks(segment_size) {
        let selected = history.choose();
        let child = encode_segment_candidate(segment, selected)?;
        frame.extend_from_slice(&child);
        history.observe(segment)?;
    }
    Ok(frame)
}

/// Decode the F64S historical-cost online selector frame. It reproduces the
/// policy from prior decoded chunks and rejects a child with an unexpected
/// transform/model ID before decoding its payload.
pub fn decompress_f64_online_selector(
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
    if frame[7] != SEGMENT_SELECTION_POLICY_HISTORICAL_COST {
        return Err(NumericError::UnsupportedTransform(frame[7]));
    }

    let count = u64::from_le_bytes(
        frame[8..16].try_into().map_err(|_| NumericError::LengthMismatch)?,
    );
    let segment_size = u32::from_le_bytes(
        frame[16..20].try_into().map_err(|_| NumericError::LengthMismatch)?,
    ) as usize;
    let segment_count = u32::from_le_bytes(
        frame[20..24].try_into().map_err(|_| NumericError::LengthMismatch)?,
    ) as usize;
    let raw_crc = u32::from_le_bytes(
        frame[24..28].try_into().map_err(|_| NumericError::LengthMismatch)?,
    );
    let header_crc = u32::from_le_bytes(
        frame[28..32].try_into().map_err(|_| NumericError::LengthMismatch)?,
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
    let expected_segments = if count == 0 { 0 } else { (count - 1) / segment_size + 1 };
    if segment_count != expected_segments {
        return Err(NumericError::LengthMismatch);
    }

    let mut output = Vec::with_capacity(count);
    let mut cursor = SEGMENTED_HEADER_LEN;
    let mut history = HistoricalSegmentCosts::default();

    for segment_index in 0..segment_count {
        let header_end = cursor.checked_add(HEADER_LEN).ok_or(NumericError::LengthOverflow)?;
        if header_end > frame.len() {
            return Err(NumericError::TruncatedHeader);
        }
        let inner_len = u64::from_le_bytes(
            frame[cursor + 16..cursor + 24]
                .try_into().map_err(|_| NumericError::LengthMismatch)?,
        );
        let inner_len = usize::try_from(inner_len).map_err(|_| NumericError::LengthOverflow)?;
        let child_len = HEADER_LEN.checked_add(inner_len).ok_or(NumericError::LengthOverflow)?;
        let child_end = cursor.checked_add(child_len).ok_or(NumericError::LengthOverflow)?;
        if child_end > frame.len() {
            return Err(NumericError::LengthMismatch);
        }

        let expected_candidate = history.choose();
        let child = &frame[cursor..child_end];
        let actual_candidate = segment_candidate_from_child_header(child)?;
        if actual_candidate != expected_candidate {
            return Err(NumericError::SegmentPolicyMismatch {
                expected: expected_candidate,
                actual: actual_candidate,
            });
        }

        let expected_values = if segment_index + 1 == segment_count {
            count - output.len()
        } else {
            segment_size
        };
        let decoded = decompress_f64(child, expected_values)?;
        if decoded.len() != expected_values {
            return Err(NumericError::LengthMismatch);
        }
        history.observe(&decoded)?;
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


fn best_segment_candidate_frame(
    values: &[f64],
) -> Result<(u8, Vec<u8>), NumericError> {
    let mut best_candidate = 0u8;
    let mut best_frame: Option<Vec<u8>> = None;
    for candidate in 0..SEGMENT_CANDIDATES as u8 {
        let frame = encode_segment_candidate(values, candidate)?;
        if best_frame
            .as_ref()
            .map_or(true, |current| frame.len() < current.len())
        {
            best_candidate = candidate;
            best_frame = Some(frame);
        }
    }
    Ok((
        best_candidate,
        best_frame.ok_or(NumericError::LengthMismatch)?,
    ))
}

/// Encode fixed-size f64 chunks by selecting the smallest complete subframe
/// among Raw/XorPrevious/XorLeadingTrailing x Order0/Order1. Every selected F64C frame (including
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
    frame.push(SEGMENT_SELECTION_POLICY_BEST_OF_SIX);
    frame.extend_from_slice(&count.to_le_bytes());
    frame.extend_from_slice(&segment_size_u32.to_le_bytes());
    frame.extend_from_slice(&segment_count_u32.to_le_bytes());
    frame.extend_from_slice(&crc32(&raw).to_le_bytes());
    let header_crc = crc32(&frame);
    frame.extend_from_slice(&header_crc.to_le_bytes());
    debug_assert_eq!(frame.len(), SEGMENTED_HEADER_LEN);

    for segment in values.chunks(segment_size) {
        let (_, best) = best_segment_candidate_frame(segment)?;
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
    if frame[7] != SEGMENT_SELECTION_POLICY_BEST_OF_SIX {
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
        let (expected_candidate, canonical_child) =
            best_segment_candidate_frame(&decoded)?;
        let actual_candidate = segment_candidate_from_child_header(&frame[cursor..child_end])?;
        if actual_candidate != expected_candidate
            || frame[cursor..child_end] != canonical_child.as_slice()
        {
            return Err(NumericError::SegmentPolicyMismatch {
                expected: expected_candidate,
                actual: actual_candidate,
            });
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
        for transform in [
            Float64Transform::Raw,
            Float64Transform::XorPrevious,
            Float64Transform::XorLeadingTrailing,
        ] {
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
        let first = compress_f64(&input, Float64Transform::XorLeadingTrailing, ModelKind::Order1).unwrap();
        let second = compress_f64(&input, Float64Transform::XorLeadingTrailing, ModelKind::Order1).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress_f64(
            &[0.0, -0.0, f64::INFINITY, f64::from_bits(0x7ff8_0000_0000_0001)],
            Float64Transform::XorLeadingTrailing,
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
        let frame = compress_f64(&input, Float64Transform::XorLeadingTrailing, ModelKind::Order1).unwrap();
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
    fn historical_cost_selector_round_trips_and_uses_past_segments_only() {
        for input in bit_fixtures() {
            for segment_size in [1, 64, 256, 1024] {
                let frame = compress_f64_online_selector(&input, segment_size).unwrap();
                let decoded = decompress_f64_online_selector(&frame, input.len()).unwrap();
                assert_same_bits(&decoded, &input);
            }
        }
    }

    #[test]
    fn historical_cost_policy_uses_prior_means_and_stable_ties() {
        let mut policy = HistoricalSegmentCosts::default();
        assert_eq!(policy.choose(), 0, "the first segment must use Raw/Order0");

        policy.observe_sizes([100, 80, 60, 40, 90, 70]).unwrap();
        assert_eq!(policy.choose(), 3);

        // A later chunk contributes to future choices, not its own selection.
        // The mean costs after two observations are [60, 55, 50, 45].
        policy.observe_sizes([20, 30, 40, 50, 45, 60]).unwrap();
        assert_eq!(policy.choose(), 3);

        let mut tied = HistoricalSegmentCosts::default();
        tied.observe_sizes([10, 20, 20, 20, 20, 20]).unwrap();
        assert_eq!(tied.choose(), 0);
    }

    #[test]
    fn historical_cost_selector_rejects_wrong_child_choice() {
        let values = [0.0f64];
        let wrong_child =
            compress_f64(&values, Float64Transform::XorPrevious, ModelKind::Order1).unwrap();
        let raw = raw_bits_bytes(&values).unwrap();

        let mut malicious = Vec::new();
        malicious.extend_from_slice(SEGMENTED_MAGIC);
        malicious.extend_from_slice(&SEGMENTED_VERSION.to_le_bytes());
        malicious.push(0);
        malicious.push(SEGMENT_SELECTION_POLICY_HISTORICAL_COST);
        malicious.extend_from_slice(&(1u64).to_le_bytes());
        malicious.extend_from_slice(&(1u32).to_le_bytes());
        malicious.extend_from_slice(&(1u32).to_le_bytes());
        malicious.extend_from_slice(&crc32(&raw).to_le_bytes());
        let header_crc = crc32(&malicious);
        malicious.extend_from_slice(&header_crc.to_le_bytes());
        malicious.extend_from_slice(&wrong_child);

        assert_eq!(
            decompress_f64_online_selector(&malicious, 1),
            Err(NumericError::SegmentPolicyMismatch {
                expected: 0,
                actual: 3
            })
        );
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
    fn xor_leading_trailing_reuses_a_window_when_residual_fits() {
        let residual = 0x0000_0000_0000_1000u64;
        let input = [
            f64::from_bits(0),
            f64::from_bits(residual),
            f64::from_bits(0),
            f64::from_bits(residual),
        ];
        let encoded =
            encode_transform(&input, Float64Transform::XorLeadingTrailing).unwrap();

        assert_eq!(encoded[0], 0); // unchanged first value
        assert_eq!(encoded[1], 2); // define initial significant-bit window
        let leading = encoded[2] as u32;
        let trailing = encoded[3] as u32;
        let width = 64 - leading - trailing;
        let payload_bytes = (width as usize + 7) / 8;
        let reuse_first = 4 + payload_bytes;
        assert_eq!(encoded[reuse_first], 1); // reuse prior window
        let reuse_second = reuse_first + 1 + payload_bytes;
        assert_eq!(encoded[reuse_second], 1); // reuse it again

        let decoded =
            decode_transform(&encoded, Float64Transform::XorLeadingTrailing, input.len()).unwrap();
        assert_same_bits(&decoded, &input);
    }

    #[test]
    fn xor_leading_trailing_rejects_malformed_markers_and_spans() {
        for payload in [
            vec![3],                         // unsupported marker
            vec![2, 64, 0],                   // leading span out of range
            vec![2, 0, 64],                   // trailing span out of range
            vec![2, 32, 32],                  // empty meaningful window
            vec![2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // zero meaningful residual
            vec![2, 0, 58, 0b1100_0001],      // nonzero unused upper bits
            vec![1, 0, 0, 0],                 // reuse marker before any window
        ] {
            assert!(
                decode_transform(&payload, Float64Transform::XorLeadingTrailing, 1).is_err(),
                "malformed XOR leading/trailing payload was accepted: {payload:?}"
            );
        }
    }

    #[test]
    fn xor_leading_trailing_zero_residual_has_single_byte_code() {
        let input = [0.0f64, 0.0, 0.0, 0.0];
        let encoded =
            encode_transform(&input, Float64Transform::XorLeadingTrailing).unwrap();
        assert_eq!(encoded, vec![0, 0, 0, 0]);
        let decoded =
            decode_transform(&encoded, Float64Transform::XorLeadingTrailing, input.len()).unwrap();
        assert_same_bits(&decoded, &input);
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
