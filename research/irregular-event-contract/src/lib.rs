//! Causality-preserving input contract for irregular and asynchronous data.
//!
//! Event time is not availability time. A late-arriving observation may refer
//! to an event that occurred before the forecast origin, but it must remain
//! hidden until its availability timestamp. Legitimately known future
//! covariates use a separate type and must have been published by the forecast
//! origin. This candidate-side crate contains no future targets or evaluator
//! labels.

use std::collections::{BTreeMap, BTreeSet};
use sha2::{Digest, Sha256};

pub mod splits;
pub use splits::{purged_chronological_split, EntitySplitBoundaries, ForecastWindow, PurgedSplitPlan, SplitError};
use std::fmt;

pub type TimestampUs = i64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingReason {
    NotMeasured,
    SensorOffline,
    BelowDetectionLimit,
    Withheld,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Measurement {
    /// Fixed-point value: physical_value = scaled_value * 10^decimal_scale.
    Observed {
        scaled_value: i64,
        decimal_scale: i8,
    },
    Missing {
        reason: MissingReason,
    },
    /// A censored interval; either bound may be open.
    Censored {
        lower_scaled: Option<i64>,
        upper_scaled: Option<i64>,
        decimal_scale: i8,
    },
    /// Content-addressed nonnumeric modality (text/audio/image/embedding).
    /// The bytes live in an external immutable object store.
    OpaquePayload {
        payload_sha256: [u8; 32],
        payload_bytes: u64,
        encoding_id: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventRecord {
    pub event_id: u64,
    pub entity_id: u64,
    pub modality_id: u32,
    pub source_id: u64,
    /// When the measured phenomenon occurred, not when the model learned it.
    pub event_time_us: TimestampUs,
    /// Earliest time the candidate could legally access this record.
    pub available_at_us: TimestampUs,
    /// Source-local sequence number. It is not inferred from event time.
    pub source_sequence: u64,
    pub unit_code: u16,
    pub measurement: Measurement,
    pub source_sha256: [u8; 32],
}

/// A future covariate is separate from an observation/target. It is usable only
/// if it was published by the forecast origin and its validity interval covers
/// the target timestamp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnownFutureCovariate {
    pub covariate_id: u64,
    pub entity_id: u64,
    pub modality_id: u32,
    pub valid_from_us: TimestampUs,
    pub valid_until_us: TimestampUs,
    pub published_at_us: TimestampUs,
    pub unit_code: u16,
    pub scaled_value: i64,
    pub decimal_scale: i8,
    pub source_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractError {
    ZeroEventId,
    ZeroEntityId,
    ZeroModalityId,
    ZeroSourceId,
    AvailabilityPrecedesEvent { event_id: u64 },
    ZeroSourceDigest,
    DuplicateEventId { event_id: u64 },
    DuplicateSourceSequence {
        entity_id: u64,
        modality_id: u32,
        source_id: u64,
        source_sequence: u64,
    },
    InvalidDecimalScale { scale: i8 },
    EmptyCensoredInterval,
    InvertedCensoredInterval,
    InvalidOpaquePayload,
    ZeroCovariateId,
    InvalidCovariateValidityInterval,
    DuplicateCovariateId { covariate_id: u64 },
    ZeroForecastEntity,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEventId => write!(f, "event ID must be nonzero"),
            Self::ZeroEntityId => write!(f, "entity ID must be nonzero"),
            Self::ZeroModalityId => write!(f, "modality ID must be nonzero"),
            Self::ZeroSourceId => write!(f, "source ID must be nonzero"),
            Self::AvailabilityPrecedesEvent { event_id } => write!(f, "observation {event_id} is available before its event time; use the known-future covariate type for pre-published information"),
            Self::ZeroSourceDigest => write!(f, "source digest must be nonzero"),
            Self::DuplicateEventId { event_id } => write!(f, "duplicate event ID {event_id}"),
            Self::DuplicateSourceSequence { entity_id, modality_id, source_sequence } => {
                write!(f, "duplicate source sequence {source_sequence} for entity {entity_id}, modality {modality_id}, source {source_id}")
            }
            Self::InvalidDecimalScale { scale } => write!(f, "decimal scale {scale} is outside the supported range [-9, 9]"),
            Self::EmptyCensoredInterval => write!(f, "censored interval must have at least one bound"),
            Self::InvertedCensoredInterval => write!(f, "censored interval lower bound exceeds upper bound"),
            Self::InvalidOpaquePayload => write!(f, "opaque multimodal payload requires nonzero digest, byte length and encoding ID"),
            Self::ZeroCovariateId => write!(f, "covariate ID must be nonzero"),
            Self::InvalidCovariateValidityInterval => write!(f, "known-future covariate validity interval is inverted"),
            Self::DuplicateCovariateId { covariate_id } => write!(f, "duplicate future covariate ID {covariate_id}"),
            Self::ZeroForecastEntity => write!(f, "forecast entity ID must be nonzero"),
        }
    }
}

impl std::error::Error for ContractError {}

fn valid_scale(scale: i8) -> bool {
    (-9..=9).contains(&scale)
}

fn nonzero_digest(digest: &[u8; 32]) -> bool {
    *digest != [0; 32]
}

impl EventRecord {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.event_id == 0 {
            return Err(ContractError::ZeroEventId);
        }
        if self.entity_id == 0 {
            return Err(ContractError::ZeroEntityId);
        }
        if self.modality_id == 0 {
            return Err(ContractError::ZeroModalityId);
        }
        if self.source_id == 0 {
            return Err(ContractError::ZeroSourceId);
        }
        if self.available_at_us < self.event_time_us {
            return Err(ContractError::AvailabilityPrecedesEvent { event_id: self.event_id });
        }
        if !nonzero_digest(&self.source_sha256) {
            return Err(ContractError::ZeroSourceDigest);
        }
        match self.measurement {
            Measurement::Observed { decimal_scale, .. }
            | Measurement::Censored { decimal_scale, .. } => {
                if !valid_scale(decimal_scale) {
                    return Err(ContractError::InvalidDecimalScale { scale: decimal_scale });
                }
            }
            Measurement::Missing { .. } | Measurement::OpaquePayload { .. } => {}
        }
        if let Measurement::OpaquePayload { payload_sha256, payload_bytes, encoding_id } = self.measurement {
            if !nonzero_digest(&payload_sha256) || payload_bytes == 0 || encoding_id == 0 {
                return Err(ContractError::InvalidOpaquePayload);
            }
        }
        if let Measurement::Censored { lower_scaled, upper_scaled, .. } = self.measurement {
            if lower_scaled.is_none() && upper_scaled.is_none() {
                return Err(ContractError::EmptyCensoredInterval);
            }
            if let (Some(lower), Some(upper)) = (lower_scaled, upper_scaled) {
                if lower > upper {
                    return Err(ContractError::InvertedCensoredInterval);
                }
            }
        }
        Ok(())
    }
}

impl KnownFutureCovariate {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.covariate_id == 0 {
            return Err(ContractError::ZeroCovariateId);
        }
        if self.entity_id == 0 {
            return Err(ContractError::ZeroEntityId);
        }
        if self.modality_id == 0 {
            return Err(ContractError::ZeroModalityId);
        }
        if self.valid_from_us > self.valid_until_us {
            return Err(ContractError::InvalidCovariateValidityInterval);
        }
        if !valid_scale(self.decimal_scale) {
            return Err(ContractError::InvalidDecimalScale { scale: self.decimal_scale });
        }
        if !nonzero_digest(&self.source_sha256) {
            return Err(ContractError::ZeroSourceDigest);
        }
        Ok(())
    }

    pub fn is_available_for(&self, entity_id: u64, forecast_origin_us: TimestampUs, target_time_us: TimestampUs) -> bool {
        self.entity_id == entity_id
            && self.published_at_us <= forecast_origin_us
            && self.valid_from_us <= target_time_us
            && target_time_us <= self.valid_until_us
    }
}

/// Validate a dataset batch without sorting or regularizing it. Event-time
/// order may differ from arrival/availability order; the original vector order
/// and source-local sequence values are preserved.
pub fn validate_event_batch(events: &[EventRecord]) -> Result<(), ContractError> {
    let mut event_ids = BTreeSet::new();
    let mut source_sequences = BTreeSet::new();

    for event in events {
        event.validate()?;
        if !event_ids.insert(event.event_id) {
            return Err(ContractError::DuplicateEventId { event_id: event.event_id });
        }
        if !source_sequences.insert((event.entity_id, event.modality_id, event.source_id, event.source_sequence)) {
            return Err(ContractError::DuplicateSourceSequence {
                entity_id: event.entity_id,
                modality_id: event.modality_id,
                source_sequence: event.source_sequence,
            });
        }
    }
    Ok(())
}

/// Return only observations legally available at the forecast origin. Preserve
/// caller order; do not sort by event time, interpolate or fill missing values.
pub fn observations_available_at<'a>(
    events: &'a [EventRecord],
    entity_id: u64,
    forecast_origin_us: TimestampUs,
) -> Result<Vec<&'a EventRecord>, ContractError> {
    if entity_id == 0 {
        return Err(ContractError::ZeroForecastEntity);
    }
    validate_event_batch(events)?;
    Ok(events.iter()
        .filter(|event| event.entity_id == entity_id && event.available_at_us <= forecast_origin_us)
        .collect())
}

/// Return separately typed covariates legitimately known at forecast time.
/// These are not observations and are not target labels.
pub fn known_covariates_for_target<'a>(
    covariates: &'a [KnownFutureCovariate],
    entity_id: u64,
    forecast_origin_us: TimestampUs,
    target_time_us: TimestampUs,
) -> Result<Vec<&'a KnownFutureCovariate>, ContractError> {
    if entity_id == 0 {
        return Err(ContractError::ZeroForecastEntity);
    }
    let mut ids = BTreeSet::new();
    for covariate in covariates {
        covariate.validate()?;
        if !ids.insert(covariate.covariate_id) {
            return Err(ContractError::DuplicateCovariateId { covariate_id: covariate.covariate_id });
        }
    }
    Ok(covariates.iter()
        .filter(|covariate| covariate.is_available_for(entity_id, forecast_origin_us, target_time_us))
        .collect())
}

/// Append-only microsecond-resolution revisions for irregular source events.
///
/// Event time is the valid/world-time coordinate. recorded_at_us is the
/// transaction/knowledge-time coordinate. Supersede and Retract append new
/// revisions; neither mutates or deletes the earlier assertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventRevisionOperation {
    Assert { record: EventRecord },
    Supersede { target_revision_id: u64, record: EventRecord },
    Retract { target_revision_id: u64, reason_code: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventRevision {
    pub revision_id: u64,
    pub event_id: u64,
    /// Per-event revision sequence; first revision is 1.
    pub revision_sequence: u64,
    pub recorded_at_us: TimestampUs,
    pub operation: EventRevisionOperation,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BitemporalEventLedger {
    /// Append order is transaction commit order and is digest-bound.
    revisions: Vec<EventRevision>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitemporalEventError {
    EmptyBatch,
    BatchTimestampMismatch,
    NonIncreasingTransactionTime { previous: TimestampUs, actual: TimestampUs },
    ZeroRevisionId,
    ZeroEventId,
    ZeroRevisionSequence,
    DuplicateRevisionId { revision_id: u64 },
    AlreadyAsserted { event_id: u64 },
    UnknownTargetRevision { revision_id: u64 },
    TargetNotLatestActiveRevision { revision_id: u64 },
    TargetEventMismatch,
    RevisionIdentityChanged { event_id: u64 },
    RevisionSequenceMismatch { expected: u64, actual: u64 },
    RevisionSequenceOverflow { event_id: u64 },
    RecordUnavailableAtTransactionTime { available_at_us: TimestampUs, recorded_at_us: TimestampUs },
    ZeroRetractionReason,
    InvalidHistory,
    InvalidQueryEntity,
}

impl fmt::Display for BitemporalEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "bitemporal event transaction must not be empty"),
            Self::BatchTimestampMismatch => write!(f, "all revisions in one transaction must share recorded_at_us"),
            Self::NonIncreasingTransactionTime { previous, actual } => write!(f, "transaction time {actual} must be later than prior commit {previous}"),
            Self::ZeroRevisionId => write!(f, "revision ID must be nonzero"),
            Self::ZeroEventId => write!(f, "event ID must be nonzero"),
            Self::ZeroRevisionSequence => write!(f, "revision sequence must be nonzero"),
            Self::DuplicateRevisionId { revision_id } => write!(f, "duplicate revision ID {revision_id}"),
            Self::AlreadyAsserted { event_id } => write!(f, "event {event_id} already has an initial assertion"),
            Self::UnknownTargetRevision { revision_id } => write!(f, "unknown target revision {revision_id}"),
            Self::TargetNotLatestActiveRevision { revision_id } => write!(f, "target revision {revision_id} is not the latest active revision"),
            Self::TargetEventMismatch => write!(f, "target revision belongs to another event ID"),
            Self::RevisionIdentityChanged { event_id } => write!(f, "revision for event {event_id} changes entity, modality, or source identity"),
            Self::RevisionSequenceMismatch { expected, actual } => write!(f, "expected revision sequence {expected}, got {actual}"),
            Self::RevisionSequenceOverflow { event_id } => write!(f, "revision sequence overflow for event {event_id}"),
            Self::RecordUnavailableAtTransactionTime { available_at_us, recorded_at_us } => write!(f, "record availability {available_at_us} is after transaction time {recorded_at_us}"),
            Self::ZeroRetractionReason => write!(f, "retraction reason code must be nonzero"),
            Self::InvalidHistory => write!(f, "bitemporal event history is inconsistent"),
            Self::InvalidQueryEntity => write!(f, "query entity ID must be nonzero"),
        }
    }
}
impl std::error::Error for BitemporalEventError {}

impl EventRevision {
    fn record(&self) -> Option<&EventRecord> {
        match &self.operation {
            EventRevisionOperation::Assert { record }
            | EventRevisionOperation::Supersede { record, .. } => Some(record),
            EventRevisionOperation::Retract { .. } => None,
        }
    }

    fn target_revision_id(&self) -> Option<u64> {
        match self.operation {
            EventRevisionOperation::Assert { .. } => None,
            EventRevisionOperation::Supersede { target_revision_id, .. }
            | EventRevisionOperation::Retract { target_revision_id, .. } => Some(target_revision_id),
        }
    }
}

impl BitemporalEventLedger {
    pub fn revisions(&self) -> &[EventRevision] {
        &self.revisions
    }

    pub fn append(&mut self, revision: EventRevision) -> Result<(), BitemporalEventError> {
        self.append_batch([revision])
    }

    /// Append an atomic transaction. All entries share one knowledge-time
    /// timestamp; a new transaction must be strictly later than the previous.
    pub fn append_batch(
        &mut self,
        revisions: impl IntoIterator<Item = EventRevision>,
    ) -> Result<(), BitemporalEventError> {
        let incoming: Vec<EventRevision> = revisions.into_iter().collect();
        if incoming.is_empty() {
            return Err(BitemporalEventError::EmptyBatch);
        }
        let transaction_time = incoming[0].recorded_at_us;
        if incoming.iter().any(|revision| revision.recorded_at_us != transaction_time) {
            return Err(BitemporalEventError::BatchTimestampMismatch);
        }
        if let Some(previous) = self.revisions.last().map(|revision| revision.recorded_at_us) {
            if transaction_time <= previous {
                return Err(BitemporalEventError::NonIncreasingTransactionTime {
                    previous,
                    actual: transaction_time,
                });
            }
        }

        let mut candidate = self.clone();
        for revision in incoming {
            candidate.append_unchecked(revision)?;
        }
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    fn append_unchecked(&mut self, revision: EventRevision) -> Result<(), BitemporalEventError> {
        if revision.revision_id == 0 {
            return Err(BitemporalEventError::ZeroRevisionId);
        }
        if revision.event_id == 0 {
            return Err(BitemporalEventError::ZeroEventId);
        }
        if revision.revision_sequence == 0 {
            return Err(BitemporalEventError::ZeroRevisionSequence);
        }
        if self.revisions.iter().any(|existing| existing.revision_id == revision.revision_id) {
            return Err(BitemporalEventError::DuplicateRevisionId {
                revision_id: revision.revision_id,
            });
        }

        let prior: Vec<&EventRevision> = self.revisions.iter()
            .filter(|existing| existing.event_id == revision.event_id)
            .collect();
        let latest = prior.iter()
            .max_by_key(|existing| existing.revision_sequence)
            .copied();
        let expected_sequence = match latest {
            None => 1,
            Some(existing) => existing.revision_sequence.checked_add(1)
                .ok_or(BitemporalEventError::RevisionSequenceOverflow { event_id: revision.event_id })?,
        };
        if revision.revision_sequence != expected_sequence {
            return Err(BitemporalEventError::RevisionSequenceMismatch {
                expected: expected_sequence,
                actual: revision.revision_sequence,
            });
        }

        match revision.operation {
            EventRevisionOperation::Assert { record } => {
                if latest.is_some() {
                    return Err(BitemporalEventError::AlreadyAsserted { event_id: revision.event_id });
                }
                validate_revision_record(&revision, &record)?;
            }
            EventRevisionOperation::Supersede { target_revision_id, record } => {
                validate_revision_record(&revision, &record)?;
                let target = self.revisions.iter()
                    .find(|existing| existing.revision_id == target_revision_id)
                    .ok_or(BitemporalEventError::UnknownTargetRevision {
                        revision_id: target_revision_id,
                    })?;
                if target.event_id != revision.event_id {
                    return Err(BitemporalEventError::TargetEventMismatch);
                }
                if latest.map(|existing| existing.revision_id) != Some(target_revision_id)
                    || target.record().is_none()
                {
                    return Err(BitemporalEventError::TargetNotLatestActiveRevision {
                        revision_id: target_revision_id,
                    });
                }
                let old_record = target.record().expect("active target was checked");
                if old_record.entity_id != record.entity_id
                    || old_record.modality_id != record.modality_id
                    || old_record.source_id != record.source_id
                {
                    return Err(BitemporalEventError::RevisionIdentityChanged {
                        event_id: revision.event_id,
                    });
                }
            }
            EventRevisionOperation::Retract { target_revision_id, reason_code } => {
                if reason_code == 0 {
                    return Err(BitemporalEventError::ZeroRetractionReason);
                }
                let target = self.revisions.iter()
                    .find(|existing| existing.revision_id == target_revision_id)
                    .ok_or(BitemporalEventError::UnknownTargetRevision {
                        revision_id: target_revision_id,
                    })?;
                if target.event_id != revision.event_id {
                    return Err(BitemporalEventError::TargetEventMismatch);
                }
                if latest.map(|existing| existing.revision_id) != Some(target_revision_id)
                    || target.record().is_none()
                {
                    return Err(BitemporalEventError::TargetNotLatestActiveRevision {
                        revision_id: target_revision_id,
                    });
                }
            }
        }

        self.revisions.push(revision);
        Ok(())
    }

    /// Revalidate histories assembled through deserialization or other means.
    pub fn verify_consistency(&self) -> Result<(), BitemporalEventError> {
        let mut candidate = Self::default();
        let mut prior_transaction_time = None;
        for revision in &self.revisions {
            if let Some(previous) = prior_transaction_time {
                if revision.recorded_at_us < previous {
                    return Err(BitemporalEventError::InvalidHistory);
                }
            }
            prior_transaction_time = Some(revision.recorded_at_us);
            candidate.append_unchecked(*revision)?;
        }
        if candidate.revisions != self.revisions {
            return Err(BitemporalEventError::InvalidHistory);
        }

        let mut latest_by_event = BTreeMap::<u64, &EventRevision>::new();
        for revision in &self.revisions {
            let replace = latest_by_event.get(&revision.event_id)
                .is_none_or(|existing| existing.revision_sequence < revision.revision_sequence);
            if replace {
                latest_by_event.insert(revision.event_id, revision);
            }
        }
        let active_records: Vec<EventRecord> = latest_by_event.values()
            .filter_map(|revision| revision.record().copied())
            .collect();
        validate_event_batch(&active_records)
            .map_err(|_| BitemporalEventError::InvalidHistory)?;
        Ok(())
    }

    /// Return the latest revision for each event that was known by known_at_us
    /// and valid by valid_at_us. Corrections/retractions recorded later cannot
    /// change this historical view.
    pub fn events_as_known_at(
        &self,
        entity_id: u64,
        valid_at_us: TimestampUs,
        known_at_us: TimestampUs,
    ) -> Result<Vec<&EventRecord>, BitemporalEventError> {
        if entity_id == 0 {
            return Err(BitemporalEventError::InvalidQueryEntity);
        }
        self.verify_consistency()?;

        let mut latest_by_event = BTreeMap::<u64, &EventRevision>::new();
        for revision in self.revisions.iter().filter(|revision| {
            revision.recorded_at_us <= known_at_us
        }) {
            let replace = latest_by_event.get(&revision.event_id)
                .is_none_or(|existing| existing.revision_sequence < revision.revision_sequence);
            if replace {
                latest_by_event.insert(revision.event_id, revision);
            }
        }

        let mut visible = Vec::new();
        for revision in latest_by_event.values() {
            let Some(record) = revision.record() else {
                continue;
            };
            if record.entity_id == entity_id
                && record.event_time_us <= valid_at_us
                && record.available_at_us <= known_at_us
            {
                visible.push(record);
            }
        }
        visible.sort_by_key(|record| (
            record.available_at_us,
            record.source_id,
            record.source_sequence,
            record.event_id,
        ));
        validate_event_batch(&visible.iter().map(|record| **record).collect::<Vec<_>>())
            .map_err(|_| BitemporalEventError::InvalidHistory)?;
        Ok(visible)
    }

    /// Domain-separated digest over the append-only ordered revision history.
    pub fn state_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-BITEMPORAL-EVENT-LEDGER-V1\0");
        for revision in &self.revisions {
            hasher.update(revision.revision_id.to_le_bytes());
            hasher.update(revision.event_id.to_le_bytes());
            hasher.update(revision.revision_sequence.to_le_bytes());
            hasher.update(revision.recorded_at_us.to_le_bytes());
            match revision.operation {
                EventRevisionOperation::Assert { record } => {
                    hasher.update([1]);
                    hash_event_record(&mut hasher, &record);
                }
                EventRevisionOperation::Supersede { target_revision_id, record } => {
                    hasher.update([2]);
                    hasher.update(target_revision_id.to_le_bytes());
                    hash_event_record(&mut hasher, &record);
                }
                EventRevisionOperation::Retract { target_revision_id, reason_code } => {
                    hasher.update([3]);
                    hasher.update(target_revision_id.to_le_bytes());
                    hasher.update(reason_code.to_le_bytes());
                }
            }
        }
        hasher.finalize().into()
    }
}

fn validate_revision_record(
    revision: &EventRevision,
    record: &EventRecord,
) -> Result<(), BitemporalEventError> {
    record.validate().map_err(|_| BitemporalEventError::InvalidHistory)?;
    if record.event_id != revision.event_id {
        return Err(BitemporalEventError::TargetEventMismatch);
    }
    if record.available_at_us > revision.recorded_at_us {
        return Err(BitemporalEventError::RecordUnavailableAtTransactionTime {
            available_at_us: record.available_at_us,
            recorded_at_us: revision.recorded_at_us,
        });
    }
    Ok(())
}

fn hash_event_record(hasher: &mut Sha256, record: &EventRecord) {
    hasher.update(record.event_id.to_le_bytes());
    hasher.update(record.entity_id.to_le_bytes());
    hasher.update(record.modality_id.to_le_bytes());
    hasher.update(record.source_id.to_le_bytes());
    hasher.update(record.event_time_us.to_le_bytes());
    hasher.update(record.available_at_us.to_le_bytes());
    hasher.update(record.source_sequence.to_le_bytes());
    hasher.update(record.unit_code.to_le_bytes());
    hasher.update(record.source_sha256);
    match record.measurement {
        Measurement::Observed { scaled_value, decimal_scale } => {
            hasher.update([1]);
            hasher.update(scaled_value.to_le_bytes());
            hasher.update([decimal_scale as u8]);
        }
        Measurement::Missing { reason } => {
            hasher.update([2]);
            let reason_tag = match reason {
                MissingReason::NotMeasured => 1,
                MissingReason::SensorOffline => 2,
                MissingReason::BelowDetectionLimit => 3,
                MissingReason::Withheld => 4,
                MissingReason::Unknown => 5,
            };
            hasher.update([reason_tag]);
        }
        Measurement::Censored { lower_scaled, upper_scaled, decimal_scale } => {
            hasher.update([3]);
            match lower_scaled {
                Some(value) => { hasher.update([1]); hasher.update(value.to_le_bytes()); }
                None => hasher.update([0]),
            }
            match upper_scaled {
                Some(value) => { hasher.update([1]); hasher.update(value.to_le_bytes()); }
                None => hasher.update([0]),
            }
            hasher.update([decimal_scale as u8]);
        }
        Measurement::OpaquePayload { payload_sha256, payload_bytes, encoding_id } => {
            hasher.update([4]);
            hasher.update(payload_sha256);
            hasher.update(payload_bytes.to_le_bytes());
            hasher.update(encoding_id.to_le_bytes());
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> [u8; 32] {
        let mut value = [0_u8; 32];
        value[0] = seed;
        value
    }

    fn observed(id: u64, modality: u32, event_time: i64, available: i64, seq: u64) -> EventRecord {
        EventRecord {
            event_id: id,
            entity_id: 1,
            modality_id: modality,
            source_id: modality as u64,
            event_time_us: event_time,
            available_at_us: available,
            source_sequence: seq,
            unit_code: 1,
            measurement: Measurement::Observed { scaled_value: 1234, decimal_scale: -2 },
            source_sha256: digest(id as u8),
        }
    }

    fn revision(
        revision_id: u64,
        event_id: u64,
        revision_sequence: u64,
        recorded_at_us: i64,
        operation: EventRevisionOperation,
    ) -> EventRevision {
        EventRevision { revision_id, event_id, revision_sequence, recorded_at_us, operation }
    }

    fn revision_record(event_id: u64, event_time_us: i64, available_at_us: i64, value: i64) -> EventRecord {
        let mut record = observed(event_id, 1, event_time_us, available_at_us, 0);
        record.measurement = Measurement::Observed { scaled_value: value, decimal_scale: 0 };
        record.source_sha256 = digest(value as u8);
        record
    }


    #[test]
    fn microsecond_bitemporal_query_hides_late_correction_from_old_knowledge_time() {
        let mut ledger = BitemporalEventLedger::default();
        let original = revision_record(1, 1_000, 1_000, 10);
        ledger.append(revision(1, 1, 1, 1_000, EventRevisionOperation::Assert { record: original })).unwrap();
        let corrected = revision_record(1, 1_000, 3_000, 20);
        ledger.append(revision(2, 1, 2, 3_000, EventRevisionOperation::Supersede {
            target_revision_id: 1,
            record: corrected,
        })).unwrap();
        let before = ledger.events_as_known_at(1, 2_000, 2_000).unwrap();
        assert_eq!(before[0].measurement, Measurement::Observed { scaled_value: 10, decimal_scale: 0 });
        let after = ledger.events_as_known_at(1, 2_000, 4_000).unwrap();
        assert_eq!(after[0].measurement, Measurement::Observed { scaled_value: 20, decimal_scale: 0 });
    }

    #[test]
    fn delayed_retraction_does_not_rewrite_prior_as_known_state() {
        let mut ledger = BitemporalEventLedger::default();
        let record = revision_record(1, 1_000, 1_000, 10);
        ledger.append(revision(1, 1, 1, 1_000, EventRevisionOperation::Assert { record })).unwrap();
        ledger.append(revision(2, 1, 2, 3_000, EventRevisionOperation::Retract {
            target_revision_id: 1,
            reason_code: 7,
        })).unwrap();
        assert_eq!(ledger.events_as_known_at(1, 2_000, 2_000).unwrap().len(), 1);
        assert_eq!(ledger.events_as_known_at(1, 2_000, 4_000).unwrap().len(), 0);
    }

    #[test]
    fn valid_time_and_knowledge_time_are_independent_microsecond_coordinates() {
        let mut ledger = BitemporalEventLedger::default();
        let record = revision_record(1, 10_000, 12_000, 10);
        ledger.append(revision(1, 1, 1, 12_000, EventRevisionOperation::Assert { record })).unwrap();
        assert_eq!(ledger.events_as_known_at(1, 9_999, 20_000).unwrap().len(), 0);
        assert_eq!(ledger.events_as_known_at(1, 10_000, 11_999).unwrap().len(), 0);
        assert_eq!(ledger.events_as_known_at(1, 10_000, 12_000).unwrap().len(), 1);
    }

    #[test]
    fn correction_cannot_change_entity_modality_or_source_identity() {
        let mut ledger = BitemporalEventLedger::default();
        let original = revision_record(1, 10, 10, 1);
        ledger.append(revision(1, 1, 1, 10, EventRevisionOperation::Assert { record: original })).unwrap();
        let mut corrected = revision_record(1, 10, 20, 2);
        corrected.entity_id = 2;
        assert_eq!(
            ledger.append(revision(2, 1, 2, 20, EventRevisionOperation::Supersede {
                target_revision_id: 1,
                record: corrected,
            })),
            Err(BitemporalEventError::RevisionIdentityChanged { event_id: 1 })
        );
    }

    #[test]
    fn revision_transaction_batches_are_atomic_and_share_one_timestamp() {
        let mut ledger = BitemporalEventLedger::default();
        let before = ledger.state_sha256();
        let first = revision_record(1, 10, 10, 1);
        assert_eq!(
            ledger.append_batch([
                revision(1, 1, 1, 20, EventRevisionOperation::Assert { record: first }),
                revision(2, 2, 1, 20, EventRevisionOperation::Retract {
                    target_revision_id: 999,
                    reason_code: 1,
                }),
            ]),
            Err(BitemporalEventError::UnknownTargetRevision { revision_id: 999 })
        );
        assert!(ledger.revisions().is_empty());
        assert_eq!(ledger.state_sha256(), before);
    }

    #[test]
    fn invalid_revision_sequence_and_duplicate_revision_ids_fail_closed() {
        let mut ledger = BitemporalEventLedger::default();
        let record = revision_record(1, 10, 10, 1);
        assert_eq!(
            ledger.append(revision(1, 1, 2, 10, EventRevisionOperation::Assert { record })),
            Err(BitemporalEventError::RevisionSequenceMismatch { expected: 1, actual: 2 })
        );
        ledger.append(revision(1, 1, 1, 10, EventRevisionOperation::Assert { record })).unwrap();
        assert_eq!(
            ledger.append(revision(1, 1, 2, 20, EventRevisionOperation::Retract {
                target_revision_id: 1,
                reason_code: 1,
            })),
            Err(BitemporalEventError::DuplicateRevisionId { revision_id: 1 })
        );
    }

    #[test]
    fn supersede_must_target_latest_and_retraction_reason_must_be_nonzero() {
        let mut ledger = BitemporalEventLedger::default();
        let first = revision_record(1, 10, 10, 1);
        ledger.append(revision(1, 1, 1, 10, EventRevisionOperation::Assert { record: first })).unwrap();
        let second = revision_record(1, 10, 20, 2);
        ledger.append(revision(2, 1, 2, 20, EventRevisionOperation::Supersede {
            target_revision_id: 1,
            record: second,
        })).unwrap();
        assert_eq!(
            ledger.append(revision(3, 1, 3, 30, EventRevisionOperation::Retract {
                target_revision_id: 1,
                reason_code: 1,
            })),
            Err(BitemporalEventError::TargetNotLatestActiveRevision { revision_id: 1 })
        );
        assert_eq!(
            ledger.append(revision(3, 1, 3, 30, EventRevisionOperation::Retract {
                target_revision_id: 2,
                reason_code: 0,
            })),
            Err(BitemporalEventError::ZeroRetractionReason)
        );
    }

    #[test]
    fn transaction_time_cannot_precede_record_availability() {
        let mut ledger = BitemporalEventLedger::default();
        let record = revision_record(1, 10, 20, 1);
        assert_eq!(
            ledger.append(revision(1, 1, 1, 19, EventRevisionOperation::Assert { record })),
            Err(BitemporalEventError::RecordUnavailableAtTransactionTime {
                available_at_us: 20,
                recorded_at_us: 19,
            })
        );
        assert!(ledger.revisions().is_empty());
    }

    #[test]
    fn bitemporal_state_digest_changes_on_append_and_is_replay_deterministic() {
        let mut left = BitemporalEventLedger::default();
        let mut right = BitemporalEventLedger::default();
        let record = revision_record(1, 10, 10, 1);
        let item = revision(1, 1, 1, 10, EventRevisionOperation::Assert { record });
        left.append(item).unwrap();
        right.append(item).unwrap();
        assert_eq!(left.state_sha256(), right.state_sha256());
        let before = left.state_sha256();
        let corrected = revision_record(1, 10, 20, 2);
        left.append(revision(2, 1, 2, 20, EventRevisionOperation::Supersede {
            target_revision_id: 1,
            record: corrected,
        })).unwrap();
        assert_ne!(left.state_sha256(), before);
    }

    #[test]
    fn active_source_sequence_collision_is_rejected_at_commit() {
        let mut ledger = BitemporalEventLedger::default();
        let first = revision_record(1, 10, 10, 1);
        let mut second = revision_record(2, 10, 10, 2);
        second.source_sequence = first.source_sequence;
        assert_eq!(
            ledger.append_batch([
                revision(1, 1, 1, 10, EventRevisionOperation::Assert { record: first }),
                revision(2, 2, 1, 10, EventRevisionOperation::Assert { record: second }),
            ]),
            Err(BitemporalEventError::InvalidHistory)
        );
        assert!(ledger.revisions().is_empty());
    }

    #[test]
    fn transaction_clock_must_advance_between_batches() {
        let mut ledger = BitemporalEventLedger::default();
        let first = revision_record(1, 10, 10, 1);
        ledger.append(revision(1, 1, 1, 10, EventRevisionOperation::Assert { record: first })).unwrap();
        let second = revision_record(2, 11, 11, 2);
        assert_eq!(
            ledger.append(revision(2, 2, 1, 10, EventRevisionOperation::Assert { record: second })),
            Err(BitemporalEventError::NonIncreasingTransactionTime { previous: 10, actual: 10 })
        );
        assert_eq!(ledger.revisions().len(), 1);
    }

    #[test]
    fn event_batch_preserves_irregular_asynchronous_records() {
        let events = vec![
            observed(1, 1, 10, 10, 0),
            observed(2, 2, 13, 15, 0),
            observed(3, 1, 11, 19, 1),
        ];
        assert_eq!(validate_event_batch(&events), Ok(()));
        assert_eq!(events[1].event_time_us, 13);
        assert_eq!(events[1].available_at_us, 15);
    }

    #[test]
    fn late_arriving_past_event_is_not_visible_before_arrival() {
        let events = vec![
            observed(1, 1, 10, 10, 0),
            observed(2, 1, 12, 30, 1),
        ];
        let at_twenty = observations_available_at(&events, 1, 20).unwrap();
        assert_eq!(at_twenty.iter().map(|e| e.event_id).collect::<Vec<_>>(), vec![1]);
        let at_thirty = observations_available_at(&events, 1, 30).unwrap();
        assert_eq!(at_thirty.iter().map(|e| e.event_id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn event_time_order_is_not_used_to_retroactively_reorder_arrivals() {
        let events = vec![
            observed(1, 1, 100, 100, 0),
            observed(2, 1, 50, 120, 1),
            observed(3, 1, 90, 130, 2),
        ];
        let visible = observations_available_at(&events, 1, 130).unwrap();
        assert_eq!(visible.iter().map(|e| e.event_id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn missingness_is_preserved_as_a_first_class_record() {
        let mut item = observed(1, 1, 10, 10, 0);
        item.measurement = Measurement::Missing { reason: MissingReason::SensorOffline };
        assert_eq!(item.validate(), Ok(()));
        assert_eq!(item.measurement, Measurement::Missing { reason: MissingReason::SensorOffline });
    }

    #[test]
    fn censored_interval_requires_valid_bounds() {
        let mut item = observed(1, 1, 10, 10, 0);
        item.measurement = Measurement::Censored { lower_scaled: Some(5), upper_scaled: Some(4), decimal_scale: 0 };
        assert_eq!(item.validate(), Err(ContractError::InvertedCensoredInterval));
        item.measurement = Measurement::Censored { lower_scaled: None, upper_scaled: None, decimal_scale: 0 };
        assert_eq!(item.validate(), Err(ContractError::EmptyCensoredInterval));
    }

    #[test]
    fn opaque_multimodal_payload_is_content_addressed_and_validated() {
        let mut event = observed(1, 7, 10, 10, 0);
        event.measurement = Measurement::OpaquePayload {
            payload_sha256: digest(9),
            payload_bytes: 128,
            encoding_id: 3,
        };
        assert_eq!(event.validate(), Ok(()));
        event.measurement = Measurement::OpaquePayload {
            payload_sha256: [0; 32],
            payload_bytes: 128,
            encoding_id: 3,
        };
        assert_eq!(event.validate(), Err(ContractError::InvalidOpaquePayload));
        event.measurement = Measurement::OpaquePayload {
            payload_sha256: digest(9),
            payload_bytes: 0,
            encoding_id: 3,
        };
        assert_eq!(event.validate(), Err(ContractError::InvalidOpaquePayload));
    }

    #[test]
    fn duplicate_event_ids_are_rejected() {
        let events = vec![observed(1, 1, 10, 10, 0), observed(1, 2, 11, 11, 0)];
        assert_eq!(validate_event_batch(&events), Err(ContractError::DuplicateEventId { event_id: 1 }));
    }

    #[test]
    fn duplicate_source_sequences_are_scoped_per_entity_and_modality() {
        let events = vec![observed(1, 1, 10, 10, 0), observed(2, 1, 11, 12, 0)];
        assert_eq!(validate_event_batch(&events), Err(ContractError::DuplicateSourceSequence {
            entity_id: 1, modality_id: 1, source_id: 1, source_sequence: 0,
        }));
        let valid = vec![observed(1, 1, 10, 10, 0), observed(2, 2, 11, 12, 0)];
        assert_eq!(validate_event_batch(&valid), Ok(()));
    }

    #[test]
    fn future_covariates_require_publication_before_origin_and_validity_at_target() {
        let covariate = KnownFutureCovariate {
            covariate_id: 1,
            entity_id: 1,
            modality_id: 2,
            valid_from_us: 20,
            valid_until_us: 40,
            published_at_us: 15,
            unit_code: 1,
            scaled_value: 7,
            decimal_scale: 0,
            source_sha256: digest(8),
        };
        assert!(covariate.is_available_for(1, 18, 25));
        assert!(!covariate.is_available_for(1, 10, 25));
        assert!(!covariate.is_available_for(1, 18, 41));
        assert!(!covariate.is_available_for(2, 18, 25));
    }

    #[test]
    fn future_covariate_batch_rejects_duplicate_ids() {
        let covariate = KnownFutureCovariate {
            covariate_id: 1, entity_id: 1, modality_id: 1,
            valid_from_us: 1, valid_until_us: 2, published_at_us: 1,
            unit_code: 1, scaled_value: 1, decimal_scale: 0, source_sha256: digest(1),
        };
        assert_eq!(
            known_covariates_for_target(&[covariate, covariate], 1, 1, 2),
            Err(ContractError::DuplicateCovariateId { covariate_id: 1 })
        );
    }

    #[test]
    fn zero_identity_and_source_digest_are_rejected() {
        let mut item = observed(1, 1, 1, 1, 0);
        item.event_id = 0;
        assert_eq!(item.validate(), Err(ContractError::ZeroEventId));
        item = observed(1, 1, 1, 1, 0);
        item.entity_id = 0;
        assert_eq!(item.validate(), Err(ContractError::ZeroEntityId));
        item = observed(1, 1, 1, 1, 0);
        item.source_sha256 = [0; 32];
        assert_eq!(item.validate(), Err(ContractError::ZeroSourceDigest));
    }

    #[test]
    fn decimal_scale_is_bounded() {
        let mut item = observed(1, 1, 1, 1, 0);
        item.measurement = Measurement::Observed { scaled_value: 1, decimal_scale: 10 };
        assert_eq!(item.validate(), Err(ContractError::InvalidDecimalScale { scale: 10 }));
    }

    #[test]
    fn an_observation_cannot_be_available_before_its_event_time() {
        let mut event = observed(1, 1, 20, 19, 0);
        event.source_id = 1;
        assert_eq!(
            event.validate(),
            Err(ContractError::AvailabilityPrecedesEvent { event_id: 1 })
        );
    }

    #[test]
    fn equal_source_sequences_from_distinct_sources_do_not_collide() {
        let mut left = observed(1, 1, 10, 10, 0);
        left.source_id = 100;
        let mut right = observed(2, 1, 11, 11, 0);
        right.source_id = 200;
        assert_eq!(validate_event_batch(&[left, right]), Ok(()));
    }

    #[test]
    fn entity_filter_and_forecast_entity_validation_are_explicit() {
        let mut other = observed(2, 1, 11, 11, 0);
        other.entity_id = 2;
        let events = vec![observed(1, 1, 10, 10, 0), other];
        let visible = observations_available_at(&events, 1, 20).unwrap();
        assert_eq!(visible.len(), 1);
        assert_eq!(observations_available_at(&events, 0, 20), Err(ContractError::ZeroForecastEntity));
    }

    #[test]
    fn availability_tie_is_included_without_rewriting_source_order() {
        let events = vec![
            observed(1, 1, 10, 20, 0),
            observed(2, 2, 11, 20, 0),
            observed(3, 3, 12, 21, 0),
        ];
        let visible = observations_available_at(&events, 1, 20).unwrap();
        assert_eq!(visible.iter().map(|e| e.event_id).collect::<Vec<_>>(), vec![1, 2]);
    }
}
