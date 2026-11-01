//! Strict CSV adapter for Time-IMM-style numeric and text tables.
//!
//! The source files commonly provide event timestamps but not actual system
//! arrival times. This adapter therefore requires an explicit policy:
//! either name an availability column, or opt into the labeled
//! EventTimeAssumedImmediate approximation. It never silently infers arrival
//! time from event time.
//!
//! Numeric values are parsed as fixed-point millionths without float
//! round-tripping. Text payloads are stored as content-addressed opaque
//! references; bytes remain outside the EventRecord.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use sha2::{Digest, Sha256};

use crate::{
    validate_event_batch, ContractError, EventRecord, Measurement, MissingReason,
    TimestampUs,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CsvAvailabilityPolicy {
    /// Explicitly assume the source made each observation available at event
    /// time. The returned manifest records this approximation.
    EventTimeAssumedImmediate,
    /// Use a source column with actual arrival/availability timestamps.
    ExplicitColumn(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsvEventConfig {
    pub entity_id: u64,
    pub source_id: u64,
    pub event_time_column: String,
    pub record_id_column: Option<String>,
    pub expected_record_id: Option<String>,
    pub availability_policy: CsvAvailabilityPolicy,
    /// Time-IMM-style naive date_time values have no timezone in the CSV.
    /// Set this only after verifying the dataset's timezone convention.
    pub allow_naive_utc: bool,
    pub unit_code: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AvailabilityProvenance {
    AssumedImmediateAtEventTime,
    ExplicitColumn { column: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsvAdapterManifest {
    pub entity_id: u64,
    pub source_id: u64,
    pub availability_provenance: AvailabilityProvenance,
    pub timestamp_timezone: String,
    pub numeric_scale: i8,
    pub row_count: usize,
    pub event_count: usize,
    pub modality_names: BTreeMap<u32, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsvAdapterOutput {
    pub events: Vec<EventRecord>,
    pub manifest: CsvAdapterManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CsvAdapterError {
    CsvParse { row: usize, message: String },
    MissingColumn { name: String },
    InvalidConfiguration,
    EmptyDataset,
    EmptyFeatureSet,
    RecordIdMismatch { row: usize, expected: String, actual: String },
    InvalidTimestamp { row: usize, column: String, value: String },
    InvalidNumber { row: usize, column: String, value: String },
    ExcessDecimalPrecision { row: usize, column: String, value: String },
    ModalityHashCollision { id: u32, first: String, second: String },
    EventIdCollision { id: u64 },
    ContractInvariant { message: String },
}

impl std::fmt::Display for CsvAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CsvParse { row, message } => write!(f, "CSV row {row}: {message}"),
            Self::MissingColumn { name } => write!(f, "required CSV column {name:?} is missing"),
            Self::InvalidConfiguration => write!(f, "CSV adapter configuration contains an empty or zero identity"),
            Self::EmptyDataset => write!(f, "CSV dataset contains no records"),
            Self::EmptyFeatureSet => write!(f, "CSV dataset contains no measurement columns"),
            Self::RecordIdMismatch { row, expected, actual } => write!(f, "CSV row {row} record ID {actual:?} differs from expected {expected:?}"),
            Self::InvalidTimestamp { row, column, value } => write!(f, "CSV row {row}, column {column:?}: invalid timestamp {value:?}"),
            Self::InvalidNumber { row, column, value } => write!(f, "CSV row {row}, column {column:?}: invalid decimal value {value:?}"),
            Self::ExcessDecimalPrecision { row, column, value } => write!(f, "CSV row {row}, column {column:?}: value {value:?} exceeds six decimal places"),
            Self::ModalityHashCollision { id, first, second } => write!(f, "modality ID collision {id} between {first:?} and {second:?}"),
            Self::EventIdCollision { id } => write!(f, "deterministic event ID collision {id}"),
            Self::ContractInvariant { message } => write!(f, "event contract rejected adapter output: {message}"),
        }
    }
}

impl std::error::Error for CsvAdapterError {}

#[derive(Clone, Debug)]
enum PayloadMode {
    NumericColumns,
    OpaqueTextColumn(String),
}

pub fn parse_time_series_csv<R: Read>(
    reader: R,
    config: &CsvEventConfig,
) -> Result<CsvAdapterOutput, CsvAdapterError> {
    parse_csv(reader, config, PayloadMode::NumericColumns)
}

pub fn parse_text_csv<R: Read>(
    reader: R,
    config: &CsvEventConfig,
    content_column: &str,
) -> Result<CsvAdapterOutput, CsvAdapterError> {
    if content_column.trim().is_empty() {
        return Err(CsvAdapterError::InvalidConfiguration);
    }
    parse_csv(reader, config, PayloadMode::OpaqueTextColumn(content_column.to_owned()))
}

fn parse_csv<R: Read>(
    reader: R,
    config: &CsvEventConfig,
    mode: PayloadMode,
) -> Result<CsvAdapterOutput, CsvAdapterError> {
    if config.entity_id == 0
        || config.source_id == 0
        || config.event_time_column.trim().is_empty()
        || config.record_id_column.as_ref().is_some_and(|name| name.trim().is_empty())
        || config.expected_record_id.as_ref().is_some_and(|name| name.trim().is_empty())
        || (config.expected_record_id.is_some() && config.record_id_column.is_none())
    {
        return Err(CsvAdapterError::InvalidConfiguration);
    }

    let mut csv = csv::ReaderBuilder::new().flexible(false).from_reader(reader);
    let headers = csv.headers()
        .map_err(|error| CsvAdapterError::CsvParse { row: 1, message: error.to_string() })?
        .clone();
    let header_names: Vec<String> = headers.iter().map(str::to_owned).collect();
    let find_column = |name: &str| -> Result<usize, CsvAdapterError> {
        header_names.iter().position(|item| item == name)
            .ok_or_else(|| CsvAdapterError::MissingColumn { name: name.to_owned() })
    };

    let time_index = find_column(&config.event_time_column)?;
    let record_index = config.record_id_column.as_deref().map(find_column).transpose()?;
    let availability_column = match &config.availability_policy {
        CsvAvailabilityPolicy::EventTimeAssumedImmediate => None,
        CsvAvailabilityPolicy::ExplicitColumn(name) => Some((name.clone(), find_column(name)?)),
    };

    let mut reserved = BTreeSet::from([time_index]);
    if let Some(index) = record_index { reserved.insert(index); }
    if let Some((_, index)) = &availability_column { reserved.insert(*index); }

    let (feature_columns, modality_names) = match &mode {
        PayloadMode::NumericColumns => {
            let mut names = BTreeMap::new();
            let mut columns = Vec::new();
            let mut by_id = BTreeMap::<u32, String>::new();
            for (index, name) in header_names.iter().enumerate() {
                if reserved.contains(&index) { continue; }
                let modality_id = modality_id_for(name);
                if let Some(existing) = by_id.insert(modality_id, name.clone()) {
                    if existing != *name {
                        return Err(CsvAdapterError::ModalityHashCollision {
                            id: modality_id,
                            first: existing,
                            second: name.clone(),
                        });
                    }
                }
                names.insert(modality_id, name.clone());
                columns.push((index, name.clone(), modality_id));
            }
            if columns.is_empty() { return Err(CsvAdapterError::EmptyFeatureSet); }
            (columns, names)
        }
        PayloadMode::OpaqueTextColumn(name) => {
            let index = find_column(name)?;
            if reserved.contains(&index) {
                return Err(CsvAdapterError::InvalidConfiguration);
            }
            let id = modality_id_for(&name);
            (vec![(index, name.clone(), id)], BTreeMap::from([(id, name.clone())]))
        }
    };

    let mut modality_sequences = BTreeMap::<u32, u64>::new();
    let mut event_ids = BTreeSet::<u64>::new();
    let mut events = Vec::new();
    let mut row_count = 0_usize;

    for (row_offset, result) in csv.records().enumerate() {
        let row_number = row_offset + 2;
        let row = result.map_err(|error| CsvAdapterError::CsvParse {
            row: row_number,
            message: error.to_string(),
        })?;
        row_count = row_count.checked_add(1).ok_or(CsvAdapterError::InvalidConfiguration)?;

        if let (Some(index), Some(expected)) = (record_index, config.expected_record_id.as_ref()) {
            let actual = row.get(index).unwrap_or_default().to_owned();
            if actual != *expected {
                return Err(CsvAdapterError::RecordIdMismatch {
                    row: row_number,
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        let event_time_text = row.get(time_index).unwrap_or_default();
        let event_time = parse_timestamp(event_time_text, config.allow_naive_utc)
            .ok_or_else(|| CsvAdapterError::InvalidTimestamp {
                row: row_number,
                column: config.event_time_column.clone(),
                value: event_time_text.to_owned(),
            })?;
        let available_at = if let Some((name, index)) = &availability_column {
            let value = row.get(*index).unwrap_or_default();
            parse_timestamp(value, config.allow_naive_utc)
                .ok_or_else(|| CsvAdapterError::InvalidTimestamp {
                    row: row_number,
                    column: name.clone(),
                    value: value.to_owned(),
                })?
        } else {
            event_time
        };

        for (column_index, column_name, modality_id) in &feature_columns {
            let raw = row.get(*column_index).unwrap_or_default();
            let sequence = modality_sequences.entry(*modality_id).or_insert(0);
            let source_sequence = *sequence;
            *sequence = sequence.checked_add(1).ok_or(CsvAdapterError::InvalidConfiguration)?;

            let event_id = event_id_for(
                config.entity_id,
                config.source_id,
                *modality_id,
                source_sequence,
            );
            if event_id == 0 || !event_ids.insert(event_id) {
                return Err(CsvAdapterError::EventIdCollision { id: event_id });
            }

            let measurement = match &mode {
                PayloadMode::NumericColumns => {
                    if raw.trim().is_empty() {
                        Measurement::Missing { reason: MissingReason::NotMeasured }
                    } else {
                        Measurement::Observed {
                            scaled_value: parse_fixed_six(raw).ok_or_else(|| {
                                if raw.trim().split_once('.').is_some_and(|(_, fraction)| fraction.len() > 6) {
                                    CsvAdapterError::ExcessDecimalPrecision {
                                        row: row_number,
                                        column: column_name.clone(),
                                        value: raw.to_owned(),
                                    }
                                } else {
                                    CsvAdapterError::InvalidNumber {
                                        row: row_number,
                                        column: column_name.clone(),
                                        value: raw.to_owned(),
                                    }
                                }
                            })?,
                            decimal_scale: -6,
                        }
                    }
                }
                PayloadMode::OpaqueTextColumn(_) => {
                    if raw.is_empty() {
                        Measurement::Missing { reason: MissingReason::NotMeasured }
                    } else {
                        Measurement::OpaquePayload {
                            payload_sha256: Sha256::digest(raw.as_bytes()).into(),
                            payload_bytes: u64::try_from(raw.len()).map_err(|_| CsvAdapterError::InvalidConfiguration)?,
                            encoding_id: 1, // UTF-8
                        }
                    }
                }
            };

            let mut hasher = Sha256::new();
            hasher.update(b"SYMTHAEA-IRREGULAR-CSV-ROW-V1\0");
            hasher.update(config.entity_id.to_le_bytes());
            hasher.update(config.source_id.to_le_bytes());
            hasher.update(modality_id.to_le_bytes());
            hasher.update(source_sequence.to_le_bytes());
            for field in row.iter() {
                let bytes = field.as_bytes();
                hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
                hasher.update(bytes);
            }
            hasher.update(column_name.as_bytes());
            let source_sha256 = hasher.finalize().into();

            events.push(EventRecord {
                event_id,
                entity_id: config.entity_id,
                modality_id: *modality_id,
                source_id: config.source_id,
                event_time_us: event_time,
                available_at_us: available_at,
                source_sequence,
                unit_code: config.unit_code,
                measurement,
                source_sha256,
            });
        }
    }

    if row_count == 0 { return Err(CsvAdapterError::EmptyDataset); }
    validate_event_batch(&events)
        .map_err(|error: ContractError| CsvAdapterError::ContractInvariant { message: error.to_string() })?;

    let availability_provenance = match &config.availability_policy {
        CsvAvailabilityPolicy::EventTimeAssumedImmediate => AvailabilityProvenance::AssumedImmediateAtEventTime,
        CsvAvailabilityPolicy::ExplicitColumn(name) => AvailabilityProvenance::ExplicitColumn { column: name.clone() },
    };
    Ok(CsvAdapterOutput {
        manifest: CsvAdapterManifest {
            entity_id: config.entity_id,
            source_id: config.source_id,
            availability_provenance,
            timestamp_timezone: if config.allow_naive_utc { "RFC3339 or naive UTC by explicit config".into() } else { "RFC3339 only".into() },
            numeric_scale: -6,
            row_count,
            event_count: events.len(),
            modality_names,
        },
        events,
    })
}

fn parse_timestamp(value: &str, allow_naive_utc: bool) -> Option<TimestampUs> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Some(timestamp.timestamp_micros());
    }
    if !allow_naive_utc { return None; }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").ok()
        .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").ok())?;
    Some(Utc.from_utc_datetime(&naive).timestamp_micros())
}

fn modality_id_for(name: &str) -> u32 {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMTHAEA-CSV-MODALITY-V1\0");
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();
    u32::from_le_bytes(digest[..4].try_into().expect("fixed digest length")) | 1
}

fn event_id_for(entity_id: u64, source_id: u64, modality_id: u32, sequence: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMTHAEA-CSV-EVENT-ID-V1\0");
    hasher.update(entity_id.to_le_bytes());
    hasher.update(source_id.to_le_bytes());
    hasher.update(modality_id.to_le_bytes());
    hasher.update(sequence.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("fixed digest length")) | 1
}

/// Parse base-10 decimal input into exact signed millionths. Scientific
/// notation and precision beyond six decimal places are rejected.
fn parse_fixed_six(value: &str) -> Option<i64> {
    let value = value.trim();
    if value.is_empty() { return None; }
    let (negative, unsigned) = if let Some(rest) = value.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = value.strip_prefix('+') {
        (false, rest)
    } else {
        (false, value)
    };
    let mut parts = unsigned.split('.');
    let whole_text = parts.next()?;
    let fraction_text = parts.next().unwrap_or("");
    if parts.next().is_some() || fraction_text.len() > 6 { return None; }
    if whole_text.is_empty() && fraction_text.is_empty() { return None; }
    if !whole_text.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let whole = if whole_text.is_empty() { 0_i128 } else { whole_text.parse::<i128>().ok()? };
    let mut fraction = if fraction_text.is_empty() { 0_i128 } else { fraction_text.parse::<i128>().ok()? };
    for _ in fraction_text.len()..6 { fraction = fraction.checked_mul(10)?; }
    let magnitude = whole.checked_mul(1_000_000)?.checked_add(fraction)?;
    let signed = if negative { magnitude.checked_neg()? } else { magnitude };
    i64::try_from(signed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn config(policy: CsvAvailabilityPolicy) -> CsvEventConfig {
        CsvEventConfig {
            entity_id: 7,
            source_id: 11,
            event_time_column: "date_time".into(),
            record_id_column: Some("record_id".into()),
            expected_record_id: Some("site-A".into()),
            availability_policy: policy,
            allow_naive_utc: true,
            unit_code: 2,
        }
    }

    #[test]
    fn numeric_csv_preserves_irregular_times_and_feature_modalities() {
        let csv = "date_time,record_id,temp,pressure\n2026-01-01 00:00:00,site-A,1.25,101.125\n2026-01-01 00:00:03,site-A,1.5,100.75\n";
        let parsed = parse_time_series_csv(Cursor::new(csv), &config(CsvAvailabilityPolicy::EventTimeAssumedImmediate)).unwrap();
        assert_eq!(parsed.manifest.row_count, 2);
        assert_eq!(parsed.events.len(), 4);
        assert_eq!(parsed.manifest.availability_provenance, AvailabilityProvenance::AssumedImmediateAtEventTime);
        assert_eq!(parsed.events[0].event_time_us, parsed.events[0].available_at_us);
        assert_ne!(parsed.events[0].modality_id, parsed.events[1].modality_id);
    }

    #[test]
    fn explicit_arrival_column_is_not_replaced_by_event_time() {
        let csv = "date_time,available_at,record_id,temp\n2026-01-01 00:00:00,2026-01-01 00:00:05,site-A,2.5\n";
        let cfg = config(CsvAvailabilityPolicy::ExplicitColumn("available_at".into()));
        let parsed = parse_time_series_csv(Cursor::new(csv), &cfg).unwrap();
        assert_eq!(parsed.events[0].available_at_us - parsed.events[0].event_time_us, 5_000_000);
        assert_eq!(parsed.manifest.availability_provenance, AvailabilityProvenance::ExplicitColumn { column: "available_at".into() });
    }

    #[test]
    fn late_arrival_remains_hidden_until_declared_arrival_time() {
        let csv = "date_time,available_at,record_id,temp\n2026-01-01 00:00:00,2026-01-01 00:00:05,site-A,2.5\n";
        let cfg = config(CsvAvailabilityPolicy::ExplicitColumn("available_at".into()));
        let parsed = parse_time_series_csv(Cursor::new(csv), &cfg).unwrap();
        assert_eq!(crate::observations_available_at(&parsed.events, 7, parsed.events[0].event_time_us).unwrap().len(), 0);
        assert_eq!(crate::observations_available_at(&parsed.events, 7, parsed.events[0].available_at_us).unwrap().len(), 1);
    }

    #[test]
    fn missing_numeric_values_remain_missing_not_zero() {
        let csv = "date_time,record_id,temp\n2026-01-01 00:00:00,site-A,\n";
        let parsed = parse_time_series_csv(Cursor::new(csv), &config(CsvAvailabilityPolicy::EventTimeAssumedImmediate)).unwrap();
        assert_eq!(parsed.events[0].measurement, Measurement::Missing { reason: MissingReason::NotMeasured });
    }

    #[test]
    fn decimal_conversion_is_exact_and_rejects_unrepresented_precision() {
        assert_eq!(parse_fixed_six("-1.250001"), Some(-1_250_001));
        assert_eq!(parse_fixed_six(".5"), Some(500_000));
        assert_eq!(parse_fixed_six("1e-3"), None);
        assert_eq!(parse_fixed_six("0.0000001"), None);
    }

    #[test]
    fn text_csv_uses_opaque_content_addressed_payloads() {
        let csv = "date_time,record_id,text\n2026-01-01 00:00:00,site-A,pressure rose rapidly\n";
        let parsed = parse_text_csv(Cursor::new(csv), &config(CsvAvailabilityPolicy::EventTimeAssumedImmediate), "text").unwrap();
        match parsed.events[0].measurement {
            Measurement::OpaquePayload { payload_sha256, payload_bytes, encoding_id } => {
                assert_eq!(payload_sha256, Sha256::digest(b"pressure rose rapidly").into());
                assert_eq!(payload_bytes, 21);
                assert_eq!(encoding_id, 1);
            }
            other => panic!("expected opaque text payload, got {other:?}"),
        }
    }

    #[test]
    fn record_id_mismatch_and_missing_columns_fail_closed() {
        let csv = "date_time,record_id,temp\n2026-01-01 00:00:00,site-B,1.0\n";
        assert!(matches!(
            parse_time_series_csv(Cursor::new(csv), &config(CsvAvailabilityPolicy::EventTimeAssumedImmediate)),
            Err(CsvAdapterError::RecordIdMismatch { .. })
        ));
        let missing = "date_time,record_id\n2026-01-01 00:00:00,site-A\n";
        assert_eq!(
            parse_time_series_csv(Cursor::new(missing), &config(CsvAvailabilityPolicy::EventTimeAssumedImmediate)),
            Err(CsvAdapterError::EmptyFeatureSet)
        );
    }

    #[test]
    fn naive_timestamp_requires_explicit_timezone_assumption() {
        let csv = "date_time,record_id,temp\n2026-01-01 00:00:00,site-A,1.0\n";
        let mut cfg = config(CsvAvailabilityPolicy::EventTimeAssumedImmediate);
        cfg.allow_naive_utc = false;
        assert!(matches!(
            parse_time_series_csv(Cursor::new(csv), &cfg),
            Err(CsvAdapterError::InvalidTimestamp { .. })
        ));
    }
}
