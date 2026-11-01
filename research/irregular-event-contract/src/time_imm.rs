//! Time-IMM per-entity CSV adapter with explicit timezone and availability semantics.
//!
//! The public CSV schema exposes date_time but not actual system-arrival time.
//! Naive wall times therefore require an explicit fixed offset or IANA zone;
//! ambiguous/nonexistent local times fail closed. An explicit availability
//! policy is recorded separately and is never described as observed arrival.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Read;
use std::str::FromStr;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, TimeZone, LocalResult};
use chrono_tz::Tz;
use csv::{ReaderBuilder, StringRecord};
use sha2::{Digest, Sha256};

use super::{EventRecord, Measurement, MissingReason, TimestampUs};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NaiveTimestampZone {
    FixedOffsetSeconds(i32),
    Iana(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AvailabilityPolicy {
    /// Explicit offline-evaluation assumption: available immediately at event
    /// time. This is not evidence of physical acquisition/arrival time.
    SynchronousAssumption { policy_id: u32 },
    /// Explicit synthetic/reconstruction delay, not observed arrival time.
    FixedDelay { policy_id: u32, delay_us: u64 },
}

impl AvailabilityPolicy {
    fn resolve(self, event_time_us: TimestampUs) -> Result<TimestampUs, TimeImmAdapterError> {
        let (policy_id, delay_us) = match self {
            Self::SynchronousAssumption { policy_id } => (policy_id, 0),
            Self::FixedDelay { policy_id, delay_us } => (policy_id, delay_us),
        };
        if policy_id == 0 {
            return Err(TimeImmAdapterError::ZeroPolicyId);
        }
        let delay = i64::try_from(delay_us).map_err(|_| TimeImmAdapterError::TimestampOverflow)?;
        event_time_us.checked_add(delay).ok_or(TimeImmAdapterError::TimestampOverflow)
    }

    pub fn sha256(self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"SYMTHAEA-TIME-IMM-AVAILABILITY-POLICY-V1\0");
        match self {
            Self::SynchronousAssumption { policy_id } => {
                h.update([1]);
                h.update(policy_id.to_le_bytes());
            }
            Self::FixedDelay { policy_id, delay_us } => {
                h.update([2]);
                h.update(policy_id.to_le_bytes());
                h.update(delay_us.to_le_bytes());
            }
        }
        h.finalize().into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumericFeatureSpec {
    pub column_name: String,
    pub modality_id: u32,
    pub unit_code: u16,
    /// physical value = scaled_value * 10^decimal_scale; accepted range [-9, 0].
    pub decimal_scale: i8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeImmAdapterConfig {
    pub dataset_id: String,
    pub expected_record_id: String,
    pub naive_timestamp_zone: Option<NaiveTimestampZone>,
    pub availability_policy: AvailabilityPolicy,
    pub features: Vec<NumericFeatureSpec>,
    pub text_modality_id: u32,
    pub text_unit_code: u16,
    pub text_encoding_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimeImmParsedTable {
    pub entity_id: u64,
    pub source_file_sha256: [u8; 32],
    pub availability_policy_sha256: [u8; 32],
    pub timestamp_zone_sha256: [u8; 32],
    pub row_count: u64,
    pub events: Vec<EventRecord>,
    /// Maximum quantization error bound per numeric feature.
    pub max_rounding_error_by_feature: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimeImmAdapterError {
    Io(String),
    Csv(String),
    EmptyDatasetId,
    EmptyExpectedRecordId,
    ZeroPolicyId,
    MissingNaiveTimezone,
    InvalidNaiveTimezone { zone: String },
    AmbiguousLocalTimestamp { value: String, zone: String },
    NonexistentLocalTimestamp { value: String, zone: String },
    InvalidTimestamp { value: String },
    TimestampOverflow,
    MissingColumn { name: String },
    DuplicateColumn { name: String },
    UnexpectedColumns { expected: Vec<String>, actual: Vec<String> },
    UnexpectedRecordId { expected: String, actual: String },
    NoNumericFeatures,
    DuplicateFeatureName { name: String },
    DuplicateModalityId { modality_id: u32 },
    InvalidFeatureScale { name: String, scale: i8 },
    InvalidTextModality,
    InvalidTextEncoding,
    InvalidDecimal { column: String, value: String },
    DecimalOverflow { column: String, value: String },
    InvalidRowWidth { row: u64 },
    EmptyTextHeader,
    MultipleTextColumns { columns: Vec<String> },
    EmptyInputTable,
    EventIdCollision { event_id: u64 },
    SourceSequenceOverflow,
    InvalidEventBatch(String),
}

impl fmt::Display for TimeImmAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "Time-IMM input read failed: {message}"),
            Self::Csv(message) => write!(f, "Time-IMM CSV parsing failed: {message}"),
            Self::EmptyDatasetId => write!(f, "dataset_id must be nonempty"),
            Self::EmptyExpectedRecordId => write!(f, "expected_record_id must be nonempty"),
            Self::ZeroPolicyId => write!(f, "availability policy ID must be nonzero"),
            Self::MissingNaiveTimezone => write!(f, "naive date_time requires an explicit timezone"),
            Self::InvalidNaiveTimezone { zone } => write!(f, "invalid fixed offset or IANA timezone {zone:?}"),
            Self::AmbiguousLocalTimestamp { value, zone } => write!(f, "local timestamp {value:?} is ambiguous in timezone {zone:?}"),
            Self::NonexistentLocalTimestamp { value, zone } => write!(f, "local timestamp {value:?} does not exist in timezone {zone:?}"),
            Self::InvalidTimestamp { value } => write!(f, "unsupported or invalid date_time value {value:?}"),
            Self::TimestampOverflow => write!(f, "timestamp arithmetic overflow"),
            Self::MissingColumn { name } => write!(f, "required CSV column {name:?} is missing"),
            Self::DuplicateColumn { name } => write!(f, "CSV column {name:?} is duplicated"),
            Self::UnexpectedColumns { expected, actual } => write!(f, "CSV feature columns differ: expected {expected:?}, actual {actual:?}"),
            Self::UnexpectedRecordId { expected, actual } => write!(f, "expected record_id {expected:?}, got {actual:?}"),
            Self::NoNumericFeatures => write!(f, "at least one numeric feature spec is required"),
            Self::DuplicateFeatureName { name } => write!(f, "duplicate numeric feature name {name:?}"),
            Self::DuplicateModalityId { modality_id } => write!(f, "duplicate feature modality ID {modality_id}"),
            Self::InvalidFeatureScale { name, scale } => write!(f, "feature {name:?} has unsupported decimal scale {scale}; use [-9, 0]"),
            Self::InvalidTextModality => write!(f, "text modality ID must be nonzero and distinct from numeric modalities"),
            Self::InvalidTextEncoding => write!(f, "text encoding ID must be nonzero"),
            Self::InvalidDecimal { column, value } => write!(f, "invalid decimal in column {column:?}: {value:?}"),
            Self::DecimalOverflow { column, value } => write!(f, "scaled decimal overflows i64 in column {column:?}: {value:?}"),
            Self::InvalidRowWidth { row } => write!(f, "CSV row {row} has a different width than the header"),
            Self::EmptyTextHeader => write!(f, "text.csv must contain one text payload column"),
            Self::MultipleTextColumns { columns } => write!(f, "expected exactly one text payload column, got {columns:?}"),
            Self::EmptyInputTable => write!(f, "input CSV has no data rows"),
            Self::EventIdCollision { event_id } => write!(f, "stable event ID collision: {event_id}"),
            Self::SourceSequenceOverflow => write!(f, "source sequence exceeds u64"),
            Self::InvalidEventBatch(message) => write!(f, "constructed Time-IMM events failed contract validation: {message}"),
        }
    }
}
impl std::error::Error for TimeImmAdapterError {}

#[derive(Debug)]
pub struct TimeImmCsvAdapter {
    config: TimeImmAdapterConfig,
}

impl TimeImmCsvAdapter {
    pub fn new(config: TimeImmAdapterConfig) -> Result<Self, TimeImmAdapterError> {
        if config.dataset_id.trim().is_empty() {
            return Err(TimeImmAdapterError::EmptyDatasetId);
        }
        if config.expected_record_id.trim().is_empty() {
            return Err(TimeImmAdapterError::EmptyExpectedRecordId);
        }
        config.availability_policy.resolve(0)?;
        if config.features.is_empty() {
            return Err(TimeImmAdapterError::NoNumericFeatures);
        }
        let mut feature_names = BTreeSet::new();
        let mut modality_ids = BTreeSet::new();
        for feature in &config.features {
            if feature.column_name.trim().is_empty() || !feature_names.insert(feature.column_name.clone()) {
                return Err(TimeImmAdapterError::DuplicateFeatureName { name: feature.column_name.clone() });
            }
            if feature.modality_id == 0 || !modality_ids.insert(feature.modality_id) {
                return Err(TimeImmAdapterError::DuplicateModalityId { modality_id: feature.modality_id });
            }
            if !(-9..=0).contains(&feature.decimal_scale) {
                return Err(TimeImmAdapterError::InvalidFeatureScale {
                    name: feature.column_name.clone(),
                    scale: feature.decimal_scale,
                });
            }
        }
        if config.text_modality_id == 0 || modality_ids.contains(&config.text_modality_id) {
            return Err(TimeImmAdapterError::InvalidTextModality);
        }
        if config.text_encoding_id == 0 {
            return Err(TimeImmAdapterError::InvalidTextEncoding);
        }
        if let Some(zone) = &config.naive_timestamp_zone {
            match zone {
                NaiveTimestampZone::FixedOffsetSeconds(seconds) => {
                    if FixedOffset::east_opt(*seconds).is_none() {
                        return Err(TimeImmAdapterError::InvalidNaiveTimezone { zone: format!("UTC offset {seconds}") });
                    }
                }
                NaiveTimestampZone::Iana(name) => {
                    Tz::from_str(name).map_err(|_| TimeImmAdapterError::InvalidNaiveTimezone { zone: name.clone() })?;
                }
            }
        }
        Ok(Self { config })
    }

    /// Parse one per-entity time_series.csv. Source file digest covers exact
    /// input bytes; each event also binds a canonical parsed-row digest.
    pub fn parse_time_series<R: Read>(&self, mut input: R) -> Result<TimeImmParsedTable, TimeImmAdapterError> {
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).map_err(|e| TimeImmAdapterError::Io(e.to_string()))?;
        let source_file_sha256: [u8; 32] = Sha256::digest(&bytes).into();

        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(bytes.as_slice());
        let headers = reader.headers().map_err(|e| TimeImmAdapterError::Csv(e.to_string()))?.clone();
        let indexes = header_indexes(&headers)?;
        let date_index = *indexes.get("date_time").ok_or_else(|| TimeImmAdapterError::MissingColumn { name: "date_time".into() })?;
        let record_index = *indexes.get("record_id").ok_or_else(|| TimeImmAdapterError::MissingColumn { name: "record_id".into() })?;
        let actual_features: Vec<String> = headers.iter()
            .filter(|name| *name != "date_time" && *name != "record_id")
            .map(str::to_owned)
            .collect();
        let mut expected_features: Vec<String> = self.config.features.iter().map(|feature| feature.column_name.clone()).collect();
        expected_features.sort();
        let mut sorted_actual = actual_features.clone();
        sorted_actual.sort();
        if sorted_actual != expected_features {
            return Err(TimeImmAdapterError::UnexpectedColumns { expected: expected_features, actual: actual_features });
        }
        let feature_indexes: Vec<usize> = self.config.features.iter().map(|feature| {
            indexes.get(feature.column_name.as_str()).copied()
                .ok_or_else(|| TimeImmAdapterError::MissingColumn { name: feature.column_name.clone() })
        }).collect::<Result<_, _>>()?;

        let entity_id = stable_id(
            b"TIME-IMM-ENTITY-V1\0",
            &[self.config.dataset_id.as_bytes(), self.config.expected_record_id.as_bytes()],
        );
        let source_ids: Vec<u64> = self.config.features.iter().map(|feature| {
            stable_id(b"TIME-IMM-SOURCE-V1\0", &[self.config.dataset_id.as_bytes(), feature.column_name.as_bytes()])
        }).collect();

        let mut events = Vec::new();
        let mut seen_event_ids = BTreeSet::new();
        let mut max_rounding_error_by_feature: BTreeMap<String, f64> = self.config.features.iter()
            .map(|feature| (feature.column_name.clone(), 0.5 * 10_f64.powi(feature.decimal_scale as i32)))
            .collect();
        let mut row_count = 0_u64;
        for (row_index, row_result) in reader.records().enumerate() {
            let row_number = u64::try_from(row_index).map_err(|_| TimeImmAdapterError::SourceSequenceOverflow)?;
            let row = row_result.map_err(|e| TimeImmAdapterError::Csv(e.to_string()))?;
            if row.len() != headers.len() {
                return Err(TimeImmAdapterError::InvalidRowWidth { row: row_number });
            }
            let record_id = row.get(record_index).unwrap_or_default().trim();
            if record_id != self.config.expected_record_id {
                return Err(TimeImmAdapterError::UnexpectedRecordId {
                    expected: self.config.expected_record_id.clone(),
                    actual: record_id.to_owned(),
                });
            }
            let event_time_us = parse_time_us(row.get(date_index).unwrap_or_default().trim(), self.config.naive_timestamp_zone.as_ref())?;
            let available_at_us = self.config.availability_policy.resolve(event_time_us)?;
            let row_digest = canonical_row_digest(b"TIME-IMM-NUMERIC-ROW-V1\0", &headers, &row);

            for ((feature, column_index), source_id) in self.config.features.iter().zip(&feature_indexes).zip(&source_ids) {
                let raw_value = row.get(*column_index).unwrap_or_default().trim();
                let measurement = if is_missing(raw_value) {
                    Measurement::Missing { reason: MissingReason::NotMeasured }
                } else {
                    let (scaled_value, rounding_error) = decimal_to_scaled(raw_value, feature.decimal_scale)
                        .map_err(|error| match error {
                            DecimalParseError::Invalid => TimeImmAdapterError::InvalidDecimal { column: feature.column_name.clone(), value: raw_value.to_owned() },
                            DecimalParseError::Overflow => TimeImmAdapterError::DecimalOverflow { column: feature.column_name.clone(), value: raw_value.to_owned() },
                        })?;
                    max_rounding_error_by_feature.entry(feature.column_name.clone())
                        .and_modify(|current: &mut f64| *current = current.max(rounding_error))
                        .or_insert(rounding_error);
                    Measurement::Observed { scaled_value, decimal_scale: feature.decimal_scale }
                };
                let event_id = stable_id(
                    b"TIME-IMM-EVENT-V1\0",
                    &[self.config.dataset_id.as_bytes(), record_id.as_bytes(), feature.column_name.as_bytes(), &row_number.to_le_bytes()],
                );
                if !seen_event_ids.insert(event_id) {
                    return Err(TimeImmAdapterError::EventIdCollision { event_id });
                }
                let mut source_hash = Sha256::new();
                source_hash.update(b"TIME-IMM-SOURCE-RECORD-V1\0");
                source_hash.update(row_digest);
                source_hash.update(feature.column_name.as_bytes());
                events.push(EventRecord {
                    event_id,
                    entity_id,
                    modality_id: feature.modality_id,
                    source_id: *source_id,
                    event_time_us,
                    available_at_us,
                    source_sequence: row_number,
                    unit_code: feature.unit_code,
                    measurement,
                    source_sha256: source_hash.finalize().into(),
                });
            }
            row_count = row_count.checked_add(1).ok_or(TimeImmAdapterError::SourceSequenceOverflow)?;
        }
        if row_count == 0 {
            return Err(TimeImmAdapterError::EmptyInputTable);
        }
        super::validate_event_batch(&events)
            .map_err(|error| TimeImmAdapterError::InvalidEventBatch(error.to_string()))?;
        Ok(TimeImmParsedTable {
            entity_id,
            source_file_sha256,
            availability_policy_sha256: self.config.availability_policy.sha256(),
            timestamp_zone_sha256: timestamp_zone_sha256(self.config.naive_timestamp_zone.as_ref()),
            row_count,
            events,
            max_rounding_error_by_feature,
        })
    }

    /// Parse one per-entity text.csv. Payload fields are not whitespace-trimmed;
    /// payload bytes are hashed exactly as returned by the CSV parser.
    pub fn parse_text<R: Read>(&self, mut input: R) -> Result<TimeImmParsedTable, TimeImmAdapterError> {
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).map_err(|e| TimeImmAdapterError::Io(e.to_string()))?;
        let source_file_sha256: [u8; 32] = Sha256::digest(&bytes).into();

        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(bytes.as_slice());
        let headers = reader.headers().map_err(|e| TimeImmAdapterError::Csv(e.to_string()))?.clone();
        let indexes = header_indexes(&headers)?;
        let date_index = *indexes.get("date_time").ok_or_else(|| TimeImmAdapterError::MissingColumn { name: "date_time".into() })?;
        let record_index = *indexes.get("record_id").ok_or_else(|| TimeImmAdapterError::MissingColumn { name: "record_id".into() })?;
        let text_columns: Vec<usize> = headers.iter().enumerate()
            .filter_map(|(index, name)| if name != "date_time" && name != "record_id" { Some(index) } else { None })
            .collect();
        if text_columns.is_empty() {
            return Err(TimeImmAdapterError::EmptyTextHeader);
        }
        if text_columns.len() != 1 {
            return Err(TimeImmAdapterError::MultipleTextColumns {
                columns: text_columns.iter().map(|index| headers[*index].to_owned()).collect(),
            });
        }
        let text_index = text_columns[0];
        let entity_id = stable_id(
            b"TIME-IMM-ENTITY-V1\0",
            &[self.config.dataset_id.as_bytes(), self.config.expected_record_id.as_bytes()],
        );
        let source_id = stable_id(b"TIME-IMM-SOURCE-V1\0", &[self.config.dataset_id.as_bytes(), b"text"]);
        let mut events = Vec::new();
        let mut seen_event_ids = BTreeSet::new();
        let mut row_count = 0_u64;
        for (row_index, row_result) in reader.records().enumerate() {
            let row_number = u64::try_from(row_index).map_err(|_| TimeImmAdapterError::SourceSequenceOverflow)?;
            let row = row_result.map_err(|e| TimeImmAdapterError::Csv(e.to_string()))?;
            if row.len() != headers.len() {
                return Err(TimeImmAdapterError::InvalidRowWidth { row: row_number });
            }
            let record_id = row.get(record_index).unwrap_or_default().trim();
            if record_id != self.config.expected_record_id {
                return Err(TimeImmAdapterError::UnexpectedRecordId {
                    expected: self.config.expected_record_id.clone(),
                    actual: record_id.to_owned(),
                });
            }
            let event_time_us = parse_time_us(row.get(date_index).unwrap_or_default().trim(), self.config.naive_timestamp_zone.as_ref())?;
            let available_at_us = self.config.availability_policy.resolve(event_time_us)?;
            let text = row.get(text_index).unwrap_or_default();
            let measurement = if text.is_empty() {
                Measurement::Missing { reason: MissingReason::NotMeasured }
            } else {
                Measurement::OpaquePayload {
                    payload_sha256: Sha256::digest(text.as_bytes()).into(),
                    payload_bytes: u64::try_from(text.len()).map_err(|_| TimeImmAdapterError::SourceSequenceOverflow)?,
                    encoding_id: self.config.text_encoding_id,
                }
            };
            let event_id = stable_id(
                b"TIME-IMM-TEXT-EVENT-V1\0",
                &[self.config.dataset_id.as_bytes(), record_id.as_bytes(), &row_number.to_le_bytes()],
            );
            if !seen_event_ids.insert(event_id) {
                return Err(TimeImmAdapterError::EventIdCollision { event_id });
            }
            let source_sha256 = canonical_row_digest(b"TIME-IMM-TEXT-ROW-V1\0", &headers, &row);
            events.push(EventRecord {
                event_id,
                entity_id,
                modality_id: self.config.text_modality_id,
                source_id,
                event_time_us,
                available_at_us,
                source_sequence: row_number,
                unit_code: self.config.text_unit_code,
                measurement,
                source_sha256,
            });
            row_count = row_count.checked_add(1).ok_or(TimeImmAdapterError::SourceSequenceOverflow)?;
        }
        if row_count == 0 {
            return Err(TimeImmAdapterError::EmptyInputTable);
        }
        super::validate_event_batch(&events)
            .map_err(|error| TimeImmAdapterError::InvalidEventBatch(error.to_string()))?;
        Ok(TimeImmParsedTable {
            entity_id,
            source_file_sha256,
            availability_policy_sha256: self.config.availability_policy.sha256(),
            row_count,
            events,
            max_rounding_error_by_feature: BTreeMap::new(),
        })
    }
}

fn timestamp_zone_sha256(zone: Option<&NaiveTimestampZone>) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"SYMTHAEA-TIME-IMM-TIMESTAMP-ZONE-V1\\0");
    match zone {
        None => h.update([0]),
        Some(NaiveTimestampZone::FixedOffsetSeconds(seconds)) => {
            h.update([1]);
            h.update(seconds.to_le_bytes());
        }
        Some(NaiveTimestampZone::Iana(name)) => {
            h.update([2]);
            h.update(u64::try_from(name.len()).unwrap_or(u64::MAX).to_le_bytes());
            h.update(name.as_bytes());
        }
    }
    h.finalize().into()
}

fn header_indexes(headers: &StringRecord) -> Result<BTreeMap<String, usize>, TimeImmAdapterError> {
    let mut indexes = BTreeMap::new();
    for (index, name) in headers.iter().enumerate() {
        let name = name.trim().to_owned();
        if indexes.insert(name.clone(), index).is_some() {
            return Err(TimeImmAdapterError::DuplicateColumn { name });
        }
    }
    Ok(indexes)
}

fn parse_time_us(value: &str, zone: Option<&NaiveTimestampZone>) -> Result<TimestampUs, TimeImmAdapterError> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Ok(timestamp.timestamp_micros());
    }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M"))
        .or_else(|_| NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .map(|date| date.and_hms_opt(0, 0, 0).expect("midnight is valid")))
        .map_err(|_| TimeImmAdapterError::InvalidTimestamp { value: value.to_owned() })?;
    let zone = zone.ok_or(TimeImmAdapterError::MissingNaiveTimezone)?;
    let resolved = match zone {
        NaiveTimestampZone::FixedOffsetSeconds(seconds) => {
            let offset = FixedOffset::east_opt(*seconds)
                .ok_or_else(|| TimeImmAdapterError::InvalidNaiveTimezone { zone: format!("UTC offset {seconds}") })?;
            match offset.from_local_datetime(&naive) {
                LocalResult::Single(datetime) => datetime.timestamp_micros(),
                LocalResult::Ambiguous(_, _) => return Err(TimeImmAdapterError::AmbiguousLocalTimestamp { value: value.to_owned(), zone: format!("UTC offset {seconds}") }),
                LocalResult::None => return Err(TimeImmAdapterError::NonexistentLocalTimestamp { value: value.to_owned(), zone: format!("UTC offset {seconds}") }),
            }
        }
        NaiveTimestampZone::Iana(name) => {
            let tz = Tz::from_str(name)
                .map_err(|_| TimeImmAdapterError::InvalidNaiveTimezone { zone: name.clone() })?;
            match tz.from_local_datetime(&naive) {
                LocalResult::Single(datetime) => datetime.timestamp_micros(),
                LocalResult::Ambiguous(_, _) => return Err(TimeImmAdapterError::AmbiguousLocalTimestamp { value: value.to_owned(), zone: name.clone() }),
                LocalResult::None => return Err(TimeImmAdapterError::NonexistentLocalTimestamp { value: value.to_owned(), zone: name.clone() }),
            }
        }
    };
    Ok(resolved)
}

fn is_missing(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "" | "na" | "n/a" | "nan" | "null")
}

#[derive(Clone, Copy)]
enum DecimalParseError { Invalid, Overflow }

fn decimal_to_scaled(value: &str, decimal_scale: i8) -> Result<(i64, f64), DecimalParseError> {
    if !(-9..=0).contains(&decimal_scale) {
        return Err(DecimalParseError::Invalid);
    }
    let value = value.trim();
    let (negative, unsigned) = if let Some(rest) = value.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = value.strip_prefix('+') {
        (false, rest)
    } else {
        (false, value)
    };
    let mut pieces = unsigned.split('.');
    let whole_text_raw = pieces.next().unwrap_or_default();
    let fraction_text = pieces.next().unwrap_or_default();
    if pieces.next().is_some()
        || (!whole_text_raw.is_empty() && !whole_text_raw.bytes().all(|byte| byte.is_ascii_digit()))
        || !fraction_text.bytes().all(|byte| byte.is_ascii_digit())
        || (whole_text_raw.is_empty() && fraction_text.is_empty())
    {
        return Err(DecimalParseError::Invalid);
    }
    let whole_text = if whole_text_raw.is_empty() { "0" } else { whole_text_raw };
    let places = usize::try_from(-decimal_scale).map_err(|_| DecimalParseError::Invalid)?;
    let factor = 10_u128.checked_pow(u32::try_from(places).map_err(|_| DecimalParseError::Invalid)?)
        .ok_or(DecimalParseError::Overflow)?;
    let whole = whole_text.parse::<u128>().map_err(|_| DecimalParseError::Overflow)?;
    let mut magnitude = whole.checked_mul(factor).ok_or(DecimalParseError::Overflow)?;
    let kept: String = fraction_text.chars().take(places).collect();
    let kept_value = if kept.is_empty() { 0_u128 } else { kept.parse::<u128>().map_err(|_| DecimalParseError::Overflow)? };
    let pad_places = places.saturating_sub(kept.len());
    let padded = kept_value.checked_mul(10_u128.checked_pow(u32::try_from(pad_places).map_err(|_| DecimalParseError::Invalid)?)
        .ok_or(DecimalParseError::Overflow)?).ok_or(DecimalParseError::Overflow)?;
    magnitude = magnitude.checked_add(padded).ok_or(DecimalParseError::Overflow)?;
    if fraction_text.as_bytes().get(places).is_some_and(|digit| *digit >= b'5') {
        magnitude = magnitude.checked_add(1).ok_or(DecimalParseError::Overflow)?;
    }
    let signed = i128::try_from(magnitude).map_err(|_| DecimalParseError::Overflow)?;
    let signed = if negative { -signed } else { signed };
    let scaled = i64::try_from(signed).map_err(|_| DecimalParseError::Overflow)?;
    let rounding_error_bound = 0.5 * 10_f64.powi(decimal_scale as i32);
    Ok((scaled, rounding_error_bound))
}

fn stable_id(domain: &[u8], fields: &[&[u8]]) -> u64 {
    let mut h = Sha256::new();
    h.update(domain);
    for field in fields {
        h.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
        h.update(field);
    }
    let digest = h.finalize();
    let mut first = [0_u8; 8];
    first.copy_from_slice(&digest[..8]);
    let id = u64::from_le_bytes(first);
    if id == 0 { 1 } else { id }
}

fn canonical_row_digest(domain: &[u8], headers: &StringRecord, row: &StringRecord) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    for (header, value) in headers.iter().zip(row.iter()) {
        h.update(u64::try_from(header.len()).unwrap_or(u64::MAX).to_le_bytes());
        h.update(header.as_bytes());
        h.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        h.update(value.as_bytes());
    }
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TimeImmAdapterConfig {
        TimeImmAdapterConfig {
            dataset_id: "EPA-Air".into(),
            expected_record_id: "Bexar".into(),
            naive_timestamp_zone: Some(NaiveTimestampZone::Iana("America/Chicago".into())),
            availability_policy: AvailabilityPolicy::SynchronousAssumption { policy_id: 1 },
            features: vec![
                NumericFeatureSpec { column_name: "temp".into(), modality_id: 1, unit_code: 1, decimal_scale: -6 },
                NumericFeatureSpec { column_name: "pm2_5".into(), modality_id: 2, unit_code: 2, decimal_scale: -6 },
                NumericFeatureSpec { column_name: "aqi".into(), modality_id: 3, unit_code: 3, decimal_scale: 0 },
                NumericFeatureSpec { column_name: "ozone".into(), modality_id: 4, unit_code: 4, decimal_scale: -9 },
            ],
            text_modality_id: 5,
            text_unit_code: 10,
            text_encoding_id: 1,
        }
    }

    #[test]
    fn naive_timestamps_require_explicit_timezone() {
        assert_eq!(parse_time_us("2024-01-01 00:00:00", None), Err(TimeImmAdapterError::MissingNaiveTimezone));
    }

    #[test]
    fn iana_timezone_resolves_dst_and_rejects_ambiguous_local_time() {
        let january = parse_time_us("2024-01-01 00:00:00", Some(&NaiveTimestampZone::Iana("America/Chicago".into()))).unwrap();
        let july = parse_time_us("2024-07-01 00:00:00", Some(&NaiveTimestampZone::Iana("America/Chicago".into()))).unwrap();
        assert_eq!(january, parse_time_us("2024-01-01T06:00:00Z", None).unwrap());
        assert_eq!(july, parse_time_us("2024-07-01T05:00:00Z", None).unwrap());
        assert!(matches!(
            parse_time_us("2024-11-03 01:30:00", Some(&NaiveTimestampZone::Iana("America/Chicago".into()))),
            Err(TimeImmAdapterError::AmbiguousLocalTimestamp { .. })
        ));
        assert!(matches!(
            parse_time_us("2024-03-10 02:30:00", Some(&NaiveTimestampZone::Iana("America/Chicago".into()))),
            Err(TimeImmAdapterError::NonexistentLocalTimestamp { .. })
        ));
    }

    #[test]
    fn numeric_csv_preserves_missingness_and_explicit_availability_policy() {
        let adapter = TimeImmCsvAdapter::new(config()).unwrap();
        let csv = "date_time,record_id,temp,pm2_5,aqi,ozone\n2024-01-01 00:00:00,Bexar,53.82,45.24,199,0.021143574\n2024-01-01 01:00:00,Bexar,53.9,,,0.022\n";
        let parsed = adapter.parse_time_series(csv.as_bytes()).unwrap();
        assert_eq!(parsed.row_count, 2);
        assert_eq!(parsed.events.len(), 8);
        let event = parsed.events.iter().find(|event| event.source_sequence == 1 && event.modality_id == 2).unwrap();
        assert_eq!(event.measurement, Measurement::Missing { reason: MissingReason::NotMeasured });
        assert_eq!(event.available_at_us, event.event_time_us);
        assert_ne!(parsed.availability_policy_sha256, [0; 32]);
    }

    #[test]
    fn fixed_delay_is_explicit_and_does_not_rewrite_event_time() {
        let mut cfg = config();
        cfg.availability_policy = AvailabilityPolicy::FixedDelay { policy_id: 2, delay_us: 5_000_000 };
        let parsed = TimeImmCsvAdapter::new(cfg).unwrap()
            .parse_time_series(b"date_time,record_id,temp,pm2_5,aqi,ozone\n2024-01-01 00:00:00,Bexar,10.0,2,3,0.1\n".as_slice())
            .unwrap();
        let event = &parsed.events[0];
        assert_eq!(event.available_at_us - event.event_time_us, 5_000_000);
    }

    #[test]
    fn text_csv_is_content_addressed_and_preserves_payload_whitespace() {
        let adapter = TimeImmCsvAdapter::new(config()).unwrap();
        let parsed = adapter.parse_text(b"date_time,record_id,text\n2024-01-16 01:51:45,Bexar,\" hello, world \"\n".as_slice()).unwrap();
        let event = &parsed.events[0];
        match event.measurement {
            Measurement::OpaquePayload { payload_sha256, payload_bytes, encoding_id } => {
                assert_eq!(payload_sha256, Sha256::digest(b" hello, world ").into());
                assert_eq!(payload_bytes, 14);
                assert_eq!(encoding_id, 1);
            }
            other => panic!("expected opaque text payload, got {other:?}"),
        }
    }

    #[test]
    fn feature_and_timestamp_specs_fail_closed() {
        let mut cfg = config();
        cfg.features[0].modality_id = cfg.text_modality_id;
        assert_eq!(TimeImmCsvAdapter::new(cfg).unwrap_err(), TimeImmAdapterError::DuplicateModalityId { modality_id: 5 });
        let mut cfg = config();
        cfg.naive_timestamp_zone = Some(NaiveTimestampZone::Iana("not/a_timezone".into()));
        assert!(matches!(TimeImmCsvAdapter::new(cfg), Err(TimeImmAdapterError::InvalidNaiveTimezone { .. })));
    }

    #[test]
    fn decimal_quantization_is_bounded_and_rejects_exponents() {
        let (scaled, bound) = decimal_to_scaled("53.8200000001", -6).unwrap();
        assert_eq!(scaled, 53_820_000);
        assert_eq!(bound, 0.5e-6);
        assert!(matches!(decimal_to_scaled("1e-3", -6), Err(DecimalParseError::Invalid)));
    }

    #[test]
    fn source_file_and_row_digests_bind_input_contents() {
        let adapter = TimeImmCsvAdapter::new(config()).unwrap();
        let left = adapter.parse_time_series(b"date_time,record_id,temp,pm2_5,aqi,ozone\n2024-01-01 00:00:00,Bexar,10.0,2,3,0.1\n".as_slice()).unwrap();
        let right = adapter.parse_time_series(b"date_time,record_id,temp,pm2_5,aqi,ozone\n2024-01-01 00:00:00,Bexar,10.00,2,3,0.1\n".as_slice()).unwrap();
        assert_ne!(left.source_file_sha256, right.source_file_sha256);
        assert_ne!(left.events[0].source_sha256, right.events[0].source_sha256);
    }

    #[test]
    fn wrong_record_id_unknown_columns_and_empty_inputs_fail_closed() {
        let adapter = TimeImmCsvAdapter::new(config()).unwrap();
        assert!(matches!(
            adapter.parse_time_series(b"date_time,record_id,temp,pm2_5,aqi,ozone\n2024-01-01 00:00:00,Travis,10.0,2,3,0.1\n".as_slice()),
            Err(TimeImmAdapterError::UnexpectedRecordId { .. })
        ));
        assert!(matches!(
            adapter.parse_time_series(b"date_time,record_id,temp,extra\n2024-01-01 00:00:00,Bexar,10.0,3\n".as_slice()),
            Err(TimeImmAdapterError::UnexpectedColumns { .. })
        ));
        assert_eq!(
            adapter.parse_time_series(b"date_time,record_id,temp,pm2_5,aqi,ozone\n".as_slice()),
            Err(TimeImmAdapterError::EmptyInputTable)
        );
    }
}
