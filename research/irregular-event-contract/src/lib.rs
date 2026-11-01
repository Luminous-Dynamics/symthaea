//! Causality-preserving input contract for irregular and asynchronous data.
//!
//! Event time is not availability time. A late-arriving observation may refer
//! to an event that occurred before the forecast origin, but it must remain
//! hidden until its availability timestamp. Legitimately known future
//! covariates use a separate type and must have been published by the forecast
//! origin. This candidate-side crate contains no future targets or evaluator
//! labels.

use std::collections::BTreeSet;
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
            Measurement::Missing { .. } => {}
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
