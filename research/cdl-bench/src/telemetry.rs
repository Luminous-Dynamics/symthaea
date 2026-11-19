//! Typed, bounded-state telemetry records with decoder-synchronous delta coding.
//!
//! Version 1 supports eight 64-bit typed fields, stable entity/series keys,
//! explicit presence/null semantics, per-key LRU state bounded by the frame,
//! deterministic segment checkpoints, and the shared CDL arithmetic coder.
//! This is a research reference format, not yet a production telemetry container.

use crate::{compress, crc32, decompress, CodecError, ModelKind};
use std::collections::BTreeMap;
use std::fmt;

const MAGIC: &[u8; 4] = b"TRD1";
const VERSION: u16 = 1;
pub const TELEMETRY_FIELD_COUNT: usize = 8;
const FIELD_COUNT: usize = TELEMETRY_FIELD_COUNT;
const HEADER_LEN: usize = 52;
const HEADER_BODY_LEN: usize = 48;
const MAX_STATE_CAPACITY: u32 = 1_000_000;
const MAX_SEGMENT_RECORDS: u32 = 1_000_000;
const MAX_TRANSFORMED_BYTES_PER_RECORD: usize = 88;
const RAW_BYTES_PER_RECORD: usize = 81;

/// The physical interpretation of one 64-bit field lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TelemetryFieldKind {
    /// Unsigned 64-bit integer.
    U64 = 0,
    /// Signed 64-bit integer, stored as its two's-complement bit pattern.
    I64 = 1,
    /// IEEE-754 binary64 bits; NaN payloads and signed zero are preserved.
    F64Bits = 2,
    /// Unsigned 32-bit integer stored in the low 32 bits.
    U32 = 3,
    /// Boolean stored as exactly 0 or 1.
    Boolean = 4,
    /// Opaque 64-bit application value.
    Opaque64 = 5,
}

impl TelemetryFieldKind {
    fn from_id(value: u8) -> Result<Self, TelemetryError> {
        match value {
            0 => Ok(Self::U64),
            1 => Ok(Self::I64),
            2 => Ok(Self::F64Bits),
            3 => Ok(Self::U32),
            4 => Ok(Self::Boolean),
            5 => Ok(Self::Opaque64),
            other => Err(TelemetryError::InvalidFieldKind(other)),
        }
    }

    fn id(self) -> u8 {
        self as u8
    }
}

/// Fixed eight-lane schema for the first typed-record prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelemetrySchema {
    pub fields: [TelemetryFieldKind; TELEMETRY_FIELD_COUNT],
}

impl TelemetrySchema {
    pub const fn new(fields: [TelemetryFieldKind; TELEMETRY_FIELD_COUNT]) -> Self {
        Self { fields }
    }
}

/// One observation from a stable entity/series key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelemetryRecord {
    pub timestamp: u64,
    pub entity_id: u32,
    pub series_id: u32,
    /// Bit i is set when field i is present. A clear bit represents null.
    pub present_mask: u8,
    /// Exact 64-bit payloads. Absent fields must contain zero.
    pub values: [u64; TELEMETRY_FIELD_COUNT],
}

impl TelemetryRecord {
    pub const fn new(
        timestamp: u64,
        entity_id: u32,
        series_id: u32,
        present_mask: u8,
        values: [u64; TELEMETRY_FIELD_COUNT],
    ) -> Self {
        Self {
            timestamp,
            entity_id,
            series_id,
            present_mask,
            values,
        }
    }

    fn key(self) -> SeriesKey {
        (self.entity_id, self.series_id)
    }
}

/// Decoded telemetry stream and the resource policy declared by its frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryStream {
    pub schema: TelemetrySchema,
    pub state_capacity: u32,
    pub segment_records: u32,
    pub records: Vec<TelemetryRecord>,
}

/// Errors in the typed telemetry frame, schema, or state-delta grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryError {
    Codec(CodecError),
    TruncatedHeader,
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedFieldCount(u8),
    UnsupportedModel(u8),
    UnsupportedFlags(u8),
    InvalidFieldKind(u8),
    InvalidStateCapacity(u32),
    InvalidSegmentSize(u32),
    OutputLimitExceeded { declared: u64, limit: usize },
    StateLimitExceeded { declared: u32, limit: usize },
    LengthOverflow,
    LengthMismatch,
    ChecksumMismatch,
    TruncatedVarint,
    VarintOverflow,
    NonCanonicalVarint,
    InvalidTimestampEncoding,
    InvalidStateEncoding,
    NonCanonicalUpdate,
    InvalidRecord(&'static str),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(f, "inner CDL codec error: {error}"),
            Self::TruncatedHeader => write!(f, "truncated typed telemetry frame header"),
            Self::BadMagic => write!(f, "invalid typed telemetry frame magic"),
            Self::UnsupportedVersion(value) => write!(f, "unsupported telemetry version {value}"),
            Self::UnsupportedFieldCount(value) => write!(f, "unsupported telemetry field count {value}"),
            Self::UnsupportedModel(value) => write!(f, "unsupported telemetry entropy model {value}"),
            Self::UnsupportedFlags(value) => write!(f, "unsupported telemetry flags {value}"),
            Self::InvalidFieldKind(value) => write!(f, "invalid telemetry field kind {value}"),
            Self::InvalidStateCapacity(value) => write!(f, "invalid telemetry state capacity {value}"),
            Self::InvalidSegmentSize(value) => write!(f, "invalid telemetry segment size {value}"),
            Self::OutputLimitExceeded { declared, limit } => {
                write!(f, "declared record count {declared} exceeds limit {limit}")
            }
            Self::StateLimitExceeded { declared, limit } => {
                write!(f, "declared state capacity {declared} exceeds limit {limit}")
            }
            Self::LengthOverflow => write!(f, "typed telemetry frame length overflow"),
            Self::LengthMismatch => write!(f, "typed telemetry frame length mismatch"),
            Self::ChecksumMismatch => write!(f, "typed telemetry checksum mismatch"),
            Self::TruncatedVarint => write!(f, "truncated telemetry varint"),
            Self::VarintOverflow => write!(f, "telemetry varint exceeds u64"),
            Self::NonCanonicalVarint => write!(f, "non-canonical telemetry varint"),
            Self::InvalidTimestampEncoding => write!(f, "invalid or non-canonical timestamp encoding"),
            Self::InvalidStateEncoding => write!(f, "invalid per-series state encoding"),
            Self::NonCanonicalUpdate => write!(f, "non-canonical changed-field bitmap"),
            Self::InvalidRecord(reason) => write!(f, "invalid telemetry record: {reason}"),
        }
    }
}

impl std::error::Error for TelemetryError {}

impl From<CodecError> for TelemetryError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

type SeriesKey = (u32, u32);

#[derive(Clone, Copy)]
struct StateEntry {
    record: TelemetryRecord,
    last_seen: u64,
}

fn write_varint_u64(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_varint_u64(input: &[u8], cursor: &mut usize) -> Result<u64, TelemetryError> {
    let mut value = 0u64;
    for byte_index in 0..10usize {
        let byte = *input.get(*cursor).ok_or(TelemetryError::TruncatedVarint)?;
        *cursor += 1;
        if byte_index == 9 && (byte & 0xfe) != 0 {
            return Err(TelemetryError::VarintOverflow);
        }
        value |= ((byte & 0x7f) as u64) << (byte_index * 7);
        if byte & 0x80 == 0 {
            if byte_index > 0 && (byte & 0x7f) == 0 {
                return Err(TelemetryError::NonCanonicalVarint);
            }
            return Ok(value);
        }
    }
    Err(TelemetryError::VarintOverflow)
}

fn validate_record(
    record: &TelemetryRecord,
    schema: &TelemetrySchema,
) -> Result<(), TelemetryError> {
    for index in 0..FIELD_COUNT {
        let bit = 1u8 << index;
        let present = record.present_mask & bit != 0;
        let value = record.values[index];
        if !present && value != 0 {
            return Err(TelemetryError::InvalidRecord(
                "absent field payloads must be canonical zero",
            ));
        }
        if present {
            match schema.fields[index] {
                TelemetryFieldKind::U32 if value > u32::MAX as u64 => {
                    return Err(TelemetryError::InvalidRecord(
                        "U32 field exceeds the 32-bit range",
                    ));
                }
                TelemetryFieldKind::Boolean if value > 1 => {
                    return Err(TelemetryError::InvalidRecord(
                        "boolean field must be zero or one",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn changed_mask(previous: &TelemetryRecord, current: &TelemetryRecord) -> u8 {
    let mut changed = 0u8;
    for index in 0..FIELD_COUNT {
        let bit = 1u8 << index;
        let was_present = previous.present_mask & bit != 0;
        let is_present = current.present_mask & bit != 0;
        if was_present != is_present
            || (is_present && previous.values[index] != current.values[index])
        {
            changed |= bit;
        }
    }
    changed
}

fn evict_oldest(states: &mut BTreeMap<SeriesKey, StateEntry>) {
    let mut victim: Option<(SeriesKey, u64)> = None;
    for (key, state) in states {
        let replace = match victim {
            None => true,
            Some((best_key, best_seen)) => {
                state.last_seen < best_seen
                    || (state.last_seen == best_seen && *key < best_key)
            }
        };
        if replace {
            victim = Some((*key, state.last_seen));
        }
    }
    if let Some((key, _)) = victim {
        states.remove(&key);
    }
}

fn raw_records_bytes(
    records: &[TelemetryRecord],
    schema: &TelemetrySchema,
) -> Result<Vec<u8>, TelemetryError> {
    let capacity = records
        .len()
        .checked_mul(RAW_BYTES_PER_RECORD)
        .and_then(|length| length.checked_add(FIELD_COUNT))
        .ok_or(TelemetryError::LengthOverflow)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| TelemetryError::LengthOverflow)?;
    for kind in schema.fields {
        output.push(kind.id());
    }
    for record in records {
        output.extend_from_slice(&record.timestamp.to_le_bytes());
        output.extend_from_slice(&record.entity_id.to_le_bytes());
        output.extend_from_slice(&record.series_id.to_le_bytes());
        output.push(record.present_mask);
        for value in record.values {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(output)
}

fn encode_state_delta(
    records: &[TelemetryRecord],
    schema: &TelemetrySchema,
    state_capacity: u32,
    segment_records: u32,
) -> Result<Vec<u8>, TelemetryError> {
    let capacity = records
        .len()
        .checked_mul(MAX_TRANSFORMED_BYTES_PER_RECORD)
        .ok_or(TelemetryError::LengthOverflow)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| TelemetryError::LengthOverflow)?;

    let mut states = BTreeMap::<SeriesKey, StateEntry>::new();
    let mut previous_timestamp: Option<u64> = None;

    for (index, record) in records.iter().enumerate() {
        validate_record(record, schema)?;
        let segment_offset = (index as u64) % (segment_records as u64);
        if segment_offset == 0 {
            states.clear();
            previous_timestamp = None;
        }

        match previous_timestamp {
            Some(previous) if record.timestamp >= previous => {
                output.push(0); // canonical nonnegative timestamp delta
                write_varint_u64(record.timestamp - previous, &mut output);
            }
            _ => {
                output.push(1); // absolute timestamp at segment start or on regression
                write_varint_u64(record.timestamp, &mut output);
            }
        }
        previous_timestamp = Some(record.timestamp);

        write_varint_u64(record.entity_id as u64, &mut output);
        write_varint_u64(record.series_id as u64, &mut output);

        let key = record.key();
        let previous = states.get(&key).map(|entry| entry.record);
        let (state_mode, changed) = if let Some(previous) = previous {
            (0u8, changed_mask(&previous, record))
        } else {
            if states.len() >= state_capacity as usize {
                evict_oldest(&mut states);
            }
            (1u8, u8::MAX)
        };
        output.push(state_mode);
        output.push(changed);
        output.push(record.present_mask);
        for field in 0..FIELD_COUNT {
            let bit = 1u8 << field;
            if changed & bit != 0 && record.present_mask & bit != 0 {
                output.extend_from_slice(&record.values[field].to_le_bytes());
            }
        }

        states.insert(
            key,
            StateEntry {
                record: *record,
                last_seen: segment_offset,
            },
        );
    }

    Ok(output)
}

/// Encode typed records using bounded per-series state and deterministic checkpoints.
///
/// A segment boundary clears the state table and makes the first timestamp absolute.
/// New or evicted keys are encoded as full record checkpoints. Within a segment,
/// the least-recently-seen key is evicted when the declared state capacity is full;
/// ties are broken lexicographically by (entity_id, series_id).
pub fn compress_telemetry_records(
    records: &[TelemetryRecord],
    schema: TelemetrySchema,
    state_capacity: u32,
    segment_records: u32,
    model: ModelKind,
) -> Result<Vec<u8>, TelemetryError> {
    if state_capacity == 0 || state_capacity > MAX_STATE_CAPACITY {
        return Err(TelemetryError::InvalidStateCapacity(state_capacity));
    }
    if segment_records == 0 || segment_records > MAX_SEGMENT_RECORDS {
        return Err(TelemetryError::InvalidSegmentSize(segment_records));
    }
    let transformed = encode_state_delta(records, &schema, state_capacity, segment_records)?;
    let inner = compress(&transformed, model)?;
    let raw = raw_records_bytes(records, &schema)?;

    let count = u64::try_from(records.len()).map_err(|_| TelemetryError::LengthOverflow)?;
    let inner_len = u64::try_from(inner.len()).map_err(|_| TelemetryError::LengthOverflow)?;
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.push(FIELD_COUNT as u8);
    header.push(model.id());
    header.extend_from_slice(&state_capacity.to_le_bytes());
    header.extend_from_slice(&segment_records.to_le_bytes());
    header.extend_from_slice(&count.to_le_bytes());
    header.extend_from_slice(&inner_len.to_le_bytes());
    header.extend_from_slice(&crc32(&raw).to_le_bytes());
    header.extend_from_slice(&crc32(&inner).to_le_bytes());
    for kind in schema.fields {
        header.push(kind.id());
    }
    debug_assert_eq!(header.len(), HEADER_BODY_LEN);
    let header_crc = crc32(&header);
    header.extend_from_slice(&header_crc.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_LEN);
    header.extend_from_slice(&inner);
    Ok(header)
}

/// Decode a typed telemetry frame while enforcing both record and state limits.
///
/// The embedded segment interval resets per-key predictor state and limits the
/// decoder's replay distance to one segment for future indexed-container designs.
/// This initial frame still requires a sequential pass to reconstruct its full stream.
pub fn decompress_telemetry_records(
    frame: &[u8],
    max_records: usize,
    max_state_entries: usize,
) -> Result<TelemetryStream, TelemetryError> {
    if frame.len() < HEADER_LEN {
        return Err(TelemetryError::TruncatedHeader);
    }
    if &frame[0..4] != MAGIC {
        return Err(TelemetryError::BadMagic);
    }
    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(TelemetryError::UnsupportedVersion(version));
    }
    if frame[6] as usize != FIELD_COUNT {
        return Err(TelemetryError::UnsupportedFieldCount(frame[6]));
    }
    let model_id = frame[7];
    if model_id > 1 {
        return Err(TelemetryError::UnsupportedModel(model_id));
    }
    let state_capacity =
        u32::from_le_bytes(frame[8..12].try_into().map_err(|_| TelemetryError::LengthMismatch)?);
    if state_capacity == 0 || state_capacity > MAX_STATE_CAPACITY {
        return Err(TelemetryError::InvalidStateCapacity(state_capacity));
    }
    if state_capacity as usize > max_state_entries {
        return Err(TelemetryError::StateLimitExceeded {
            declared: state_capacity,
            limit: max_state_entries,
        });
    }
    let segment_records =
        u32::from_le_bytes(frame[12..16].try_into().map_err(|_| TelemetryError::LengthMismatch)?);
    if segment_records == 0 || segment_records > MAX_SEGMENT_RECORDS {
        return Err(TelemetryError::InvalidSegmentSize(segment_records));
    }
    let count = u64::from_le_bytes(
        frame[16..24]
            .try_into()
            .map_err(|_| TelemetryError::LengthMismatch)?,
    );
    if count > max_records as u64 {
        return Err(TelemetryError::OutputLimitExceeded {
            declared: count,
            limit: max_records,
        });
    }
    let inner_len = u64::from_le_bytes(
        frame[24..32]
            .try_into()
            .map_err(|_| TelemetryError::LengthMismatch)?,
    );
    let raw_crc = u32::from_le_bytes(
        frame[32..36]
            .try_into()
            .map_err(|_| TelemetryError::LengthMismatch)?,
    );
    let inner_crc = u32::from_le_bytes(
        frame[36..40]
            .try_into()
            .map_err(|_| TelemetryError::LengthMismatch)?,
    );
    let mut fields = [TelemetryFieldKind::Opaque64; FIELD_COUNT];
    for (index, field) in fields.iter_mut().enumerate() {
        *field = TelemetryFieldKind::from_id(frame[40 + index])?;
    }
    let schema = TelemetrySchema::new(fields);

    let stored_header_crc = u32::from_le_bytes(
        frame[HEADER_BODY_LEN..HEADER_LEN]
            .try_into()
            .map_err(|_| TelemetryError::LengthMismatch)?,
    );
    if crc32(&frame[..HEADER_BODY_LEN]) != stored_header_crc {
        return Err(TelemetryError::ChecksumMismatch);
    }

    let count = usize::try_from(count).map_err(|_| TelemetryError::LengthOverflow)?;
    let inner_len = usize::try_from(inner_len).map_err(|_| TelemetryError::LengthOverflow)?;
    let expected_len = HEADER_LEN
        .checked_add(inner_len)
        .ok_or(TelemetryError::LengthOverflow)?;
    if expected_len != frame.len() {
        return Err(TelemetryError::LengthMismatch);
    }
    let inner = &frame[HEADER_LEN..];
    if crc32(inner) != inner_crc {
        return Err(TelemetryError::ChecksumMismatch);
    }
    if inner.get(6).copied() != Some(model_id) {
        return Err(TelemetryError::UnsupportedModel(model_id));
    }

    let max_transformed = count
        .checked_mul(MAX_TRANSFORMED_BYTES_PER_RECORD)
        .ok_or(TelemetryError::LengthOverflow)?;
    let transformed = decompress(inner, max_transformed)?;
    let records = decode_state_delta(&transformed, &schema, count, state_capacity, segment_records)?;
    let raw = raw_records_bytes(&records, &schema)?;
    if crc32(&raw) != raw_crc {
        return Err(TelemetryError::ChecksumMismatch);
    }

    Ok(TelemetryStream {
        schema,
        state_capacity,
        segment_records,
        records,
    })
}

fn decode_state_delta(
    input: &[u8],
    schema: &TelemetrySchema,
    count: usize,
    state_capacity: u32,
    segment_records: u32,
) -> Result<Vec<TelemetryRecord>, TelemetryError> {
    let _raw_capacity = count
        .checked_mul(RAW_BYTES_PER_RECORD)
        .ok_or(TelemetryError::LengthOverflow)?;
    let mut records = Vec::new();
    records
        .try_reserve_exact(count)
        .map_err(|_| TelemetryError::LengthOverflow)?;
    let mut states = BTreeMap::<SeriesKey, StateEntry>::new();
    let mut previous_timestamp: Option<u64> = None;
    let mut cursor = 0usize;

    for index in 0..count {
        let segment_offset = (index as u64) % (segment_records as u64);
        if segment_offset == 0 {
            states.clear();
            previous_timestamp = None;
        }

        let time_mode = *input.get(cursor).ok_or(TelemetryError::LengthMismatch)?;
        cursor += 1;
        if time_mode > 1 {
            return Err(TelemetryError::InvalidTimestampEncoding);
        }
        let time_value = read_varint_u64(input, &mut cursor)?;
        let timestamp = match previous_timestamp {
            None if time_mode == 1 => time_value,
            None => return Err(TelemetryError::InvalidTimestampEncoding),
            Some(previous) if time_mode == 0 => {
                let decoded = previous
                    .checked_add(time_value)
                    .ok_or(TelemetryError::InvalidTimestampEncoding)?;
                if decoded < previous {
                    return Err(TelemetryError::InvalidTimestampEncoding);
                }
                decoded
            }
            Some(previous) if time_mode == 1 => {
                if time_value >= previous {
                    return Err(TelemetryError::InvalidTimestampEncoding);
                }
                time_value
            }
            Some(_) => return Err(TelemetryError::InvalidTimestampEncoding),
        };
        previous_timestamp = Some(timestamp);

        let entity = read_varint_u64(input, &mut cursor)?;
        let series = read_varint_u64(input, &mut cursor)?;
        if entity > u32::MAX as u64 || series > u32::MAX as u64 {
            return Err(TelemetryError::InvalidStateEncoding);
        }
        let key = (entity as u32, series as u32);
        let previous = states.get(&key).map(|entry| entry.record);
        if previous.is_none() && states.len() >= state_capacity as usize {
            evict_oldest(&mut states);
        }

        let state_mode = *input.get(cursor).ok_or(TelemetryError::LengthMismatch)?;
        cursor += 1;
        let changed = *input.get(cursor).ok_or(TelemetryError::LengthMismatch)?;
        cursor += 1;
        let present_mask = *input.get(cursor).ok_or(TelemetryError::LengthMismatch)?;
        cursor += 1;

        let (mut values, expected_state_mode) = match previous {
            Some(previous) => (previous.values, 0u8),
            None => ([0u64; FIELD_COUNT], 1u8),
        };
        if state_mode != expected_state_mode {
            return Err(TelemetryError::InvalidStateEncoding);
        }
        if state_mode == 1 && changed != u8::MAX {
            return Err(TelemetryError::InvalidStateEncoding);
        }

        for field in 0..FIELD_COUNT {
            let bit = 1u8 << field;
            if changed & bit != 0 {
                if present_mask & bit != 0 {
                    let end = cursor
                        .checked_add(8)
                        .ok_or(TelemetryError::LengthOverflow)?;
                    let bytes = input.get(cursor..end).ok_or(TelemetryError::LengthMismatch)?;
                    values[field] = u64::from_le_bytes(
                        bytes.try_into().map_err(|_| TelemetryError::LengthMismatch)?,
                    );
                    cursor = end;
                } else {
                    values[field] = 0;
                }
            } else if previous.is_none()
                || (previous.unwrap().present_mask & bit != 0) != (present_mask & bit != 0)
            {
                return Err(TelemetryError::NonCanonicalUpdate);
            }
        }

        let record = TelemetryRecord::new(
            timestamp,
            key.0,
            key.1,
            present_mask,
            values,
        );
        validate_record(&record, schema)?;
        let expected_changed = previous
            .map(|old| changed_mask(&old, &record))
            .unwrap_or(u8::MAX);
        if changed != expected_changed {
            return Err(TelemetryError::NonCanonicalUpdate);
        }
        states.insert(
            key,
            StateEntry {
                record,
                last_seen: segment_offset,
            },
        );
        records.push(record);
    }

    if cursor != input.len() {
        return Err(TelemetryError::LengthMismatch);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> TelemetrySchema {
        TelemetrySchema::new([
            TelemetryFieldKind::F64Bits,
            TelemetryFieldKind::F64Bits,
            TelemetryFieldKind::U32,
            TelemetryFieldKind::Boolean,
            TelemetryFieldKind::I64,
            TelemetryFieldKind::U64,
            TelemetryFieldKind::Opaque64,
            TelemetryFieldKind::F64Bits,
        ])
    }

    fn fixtures() -> Vec<TelemetryRecord> {
        let nan_payload = 0x7ff8_1234_5678_9abc;
        vec![
            TelemetryRecord::new(100, 1, 10, 0xff, [0x3ff0_0000_0000_0000, 0x8000_0000_0000_0000, 4, 1, u64::MAX, 0, 0x55, nan_payload]),
            TelemetryRecord::new(101, 1, 10, 0xff, [0x3ff0_0000_0000_0000, 0x8000_0000_0000_0000, 4, 1, u64::MAX, 0, 0x55, nan_payload]),
            TelemetryRecord::new(103, 1, 10, 0xf7, [0x4000_0000_0000_0000, 0x8000_0000_0000_0000, 4, 0, 0, 0, 0x55, nan_payload]),
            TelemetryRecord::new(102, 2, 20, 0xff, [0, 0x3ff0_0000_0000_0000, 8, 1, 12, 0, 0x55, 0x7ff8_0000_0000_0001]),
            TelemetryRecord::new(104, 1, 10, 0xff, [0x4000_0000_0000_0000, 0x8000_0000_0000_0000, 4, 1, u64::MAX, 0, 0x55, nan_payload]),
            TelemetryRecord::new(105, 3, 30, 0xff, [0x3ff0_0000_0000_0000, 0, 1, 1, 0, 0, 0, 0]),
            TelemetryRecord::new(106, 2, 20, 0xff, [0, 0x3ff0_0000_0000_0000, 8, 1, 12, 0, 0x55, 0x7ff8_0000_0000_0001]),
            TelemetryRecord::new(107, 1, 10, 0xff, [0x4000_0000_0000_0000, 0x8000_0000_0000_0000, 4, 1, u64::MAX, 0, 0x55, nan_payload]),
        ]
    }

    #[test]
    fn exact_round_trip_across_state_limits_and_segment_checkpoints() {
        let input = fixtures();
        for model in [ModelKind::Order0, ModelKind::Order1] {
            for (capacity, segment_records) in [(1, 2), (2, 4), (8, 64)] {
                let frame = compress_telemetry_records(
                    &input,
                    schema(),
                    capacity,
                    segment_records,
                    model,
                )
                .unwrap();
                let decoded = decompress_telemetry_records(&frame, input.len(), 8).unwrap();
                assert_eq!(decoded.records, input);
                assert_eq!(decoded.schema, schema());
                assert_eq!(decoded.state_capacity, capacity);
                assert_eq!(decoded.segment_records, segment_records);
            }
        }
    }

    #[test]
    fn deterministic_encoding() {
        let input = fixtures();
        let first = compress_telemetry_records(&input, schema(), 2, 4, ModelKind::Order1).unwrap();
        let second = compress_telemetry_records(&input, schema(), 2, 4, ModelKind::Order1).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn empty_stream_is_supported() {
        let frame = compress_telemetry_records(&[], schema(), 1, 16, ModelKind::Order0).unwrap();
        let decoded = decompress_telemetry_records(&frame, 0, 1).unwrap();
        assert!(decoded.records.is_empty());
        assert_eq!(decoded.schema, schema());
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress_telemetry_records(&fixtures(), schema(), 2, 4, ModelKind::Order0).unwrap();
        for end in 0..frame.len() {
            assert!(decompress_telemetry_records(&frame[..end], 100, 10).is_err());
        }
    }

    #[test]
    fn every_single_bit_mutation_is_rejected() {
        let frame = compress_telemetry_records(&fixtures(), schema(), 2, 4, ModelKind::Order1).unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut mutated = frame.clone();
                mutated[index] ^= 1 << bit;
                assert!(decompress_telemetry_records(&mutated, 100, 10).is_err());
            }
        }
    }

    #[test]
    fn output_and_state_limits_are_enforced() {
        let frame = compress_telemetry_records(&fixtures(), schema(), 2, 4, ModelKind::Order0).unwrap();
        assert!(matches!(
            decompress_telemetry_records(&frame, 2, 10),
            Err(TelemetryError::OutputLimitExceeded { .. })
        ));
        assert!(matches!(
            decompress_telemetry_records(&frame, 100, 1),
            Err(TelemetryError::StateLimitExceeded { .. })
        ));
    }

    #[test]
    fn noncanonical_absent_fields_and_invalid_types_are_rejected() {
        let mut invalid = fixtures();
        invalid[0].values[2] = 0x1_0000_0000;
        assert!(matches!(
            compress_telemetry_records(&invalid, schema(), 2, 4, ModelKind::Order0),
            Err(TelemetryError::InvalidRecord(_))
        ));
        let mut absent_nonzero = fixtures();
        absent_nonzero[2].values[3] = 1;
        assert!(matches!(
            compress_telemetry_records(&absent_nonzero, schema(), 2, 4, ModelKind::Order0),
            Err(TelemetryError::InvalidRecord(_))
        ));
    }

    #[test]
    fn malformed_varints_are_rejected() {
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0x80], &mut cursor),
            Err(TelemetryError::TruncatedVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0x80, 0x00], &mut cursor),
            Err(TelemetryError::NonCanonicalVarint)
        );
        let mut cursor = 0;
        assert_eq!(
            read_varint_u64(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02], &mut cursor),
            Err(TelemetryError::VarintOverflow)
        );
    }
}
