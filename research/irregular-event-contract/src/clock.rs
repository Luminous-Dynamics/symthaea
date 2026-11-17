//! Uncertainty-bounded mapping between source-local event clocks.
//!
//! A microsecond is a unit, not a shared clock. Two devices can report values
//! in microseconds while using different epochs, offsets, or oscillator rates.
//! This module maps source-local instants and exclusive watermarks into a
//! declared reference clock using an explicit, knowledge-time-qualified
//! calibration interval. Completeness uses the earliest possible mapped
//! watermark, never its midpoint or optimistic upper bound.
//!
//! This is a calibration contract, not source authentication. A caller still
//! needs an independently governed calibration/authority registry.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const DRIFT_DENOMINATOR: i128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClockDomainId(pub u32);

impl ClockDomainId {
    pub fn validate(self) -> Result<(), ClockCalibrationError> {
        if self.0 == 0 {
            Err(ClockCalibrationError::ZeroClockDomain)
        } else {
            Ok(())
        }
    }
}

/// An interval containing the possible reference-clock coordinate of one
/// source-local timestamp after applying a calibration known at the requested
/// knowledge-time cutoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedClockTime {
    source_domain: ClockDomainId,
    reference_domain: ClockDomainId,
    source_time_us: i64,
    earliest_reference_us: i64,
    latest_reference_us: i64,
    calibration_id: u64,
    calibration_sha256: [u8; 32],
}

impl MappedClockTime {
    pub fn source_domain(&self) -> ClockDomainId {
        self.source_domain
    }

    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn source_time_us(&self) -> i64 {
        self.source_time_us
    }

    pub fn earliest_reference_us(&self) -> i64 {
        self.earliest_reference_us
    }

    pub fn latest_reference_us(&self) -> i64 {
        self.latest_reference_us
    }

    pub fn calibration_id(&self) -> u64 {
        self.calibration_id
    }

    pub fn calibration_sha256(&self) -> [u8; 32] {
        self.calibration_sha256
    }
}

/// An exclusive source watermark mapped to a common reference clock.
///
/// The lower endpoint is the conservative bound suitable for deciding
/// whether a half-open interval is closed. The upper endpoint is diagnostic
/// only; it must not be used to make an absence claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedExclusiveWatermark {
    source_domain: ClockDomainId,
    reference_domain: ClockDomainId,
    source_exclusive_watermark_us: i64,
    earliest_reference_bound_us: i64,
    latest_reference_bound_us: i64,
    calibration_id: u64,
    calibration_sha256: [u8; 32],
}

impl MappedExclusiveWatermark {
    pub fn source_domain(&self) -> ClockDomainId {
        self.source_domain
    }

    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn source_exclusive_watermark_us(&self) -> i64 {
        self.source_exclusive_watermark_us
    }

    /// Conservative lower endpoint in the reference clock.
    pub fn earliest_reference_bound_us(&self) -> i64 {
        self.earliest_reference_bound_us
    }

    /// Optimistic upper endpoint for diagnostics only.
    pub fn latest_reference_bound_us(&self) -> i64 {
        self.latest_reference_bound_us
    }

    pub fn calibration_id(&self) -> u64 {
        self.calibration_id
    }

    pub fn calibration_sha256(&self) -> [u8; 32] {
        self.calibration_sha256
    }

    /// Whether this source alone conservatively closes [start, end).
    pub fn closes_half_open_interval(&self, end_exclusive_us: i64) -> bool {
        self.earliest_reference_bound_us >= end_exclusive_us
    }
}

/// Relationship between a mapped event-time interval and a mapped exclusive
/// watermark interval. Ambiguity is explicit: interval overlap is never forced
/// into an on-time/late binary classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventWatermarkOrdering {
    /// Every possible event time is strictly before every possible watermark.
    DefinitelyLate,
    /// Every possible event time is at or after every possible watermark.
    DefinitelyNotLate,
    /// The uncertainty intervals overlap and do not establish an ordering.
    Ambiguous,
}

/// Non-forgeable comparison preserving both mapped intervals and their
/// calibration identities. Consumers must not discard an Ambiguous outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventWatermarkComparisonV1 {
    ordering: EventWatermarkOrdering,
    event_time: MappedClockTime,
    watermark: MappedExclusiveWatermark,
}

impl EventWatermarkComparisonV1 {
    pub fn ordering(&self) -> EventWatermarkOrdering {
        self.ordering
    }

    pub fn event_time(&self) -> MappedClockTime {
        self.event_time
    }

    pub fn watermark(&self) -> MappedExclusiveWatermark {
        self.watermark
    }
}

/// Compare an uncertain mapped event timestamp against an uncertain mapped
/// exclusive watermark.
///
/// If both values use the same source clock and exact calibration revision,
/// monotonicity lets us compare their source-local coordinates directly. For
/// different source clocks or calibration revisions, only disjoint mapped
/// intervals establish an ordering. Overlap returns Ambiguous, because
/// choosing either endpoint or midpoint could invent lateness/timeliness.
pub fn compare_event_time_to_exclusive_watermark(
    event_time: MappedClockTime,
    watermark: MappedExclusiveWatermark,
) -> Result<EventWatermarkComparisonV1, ClockCalibrationError> {
    if event_time.reference_domain != watermark.reference_domain {
        return Err(ClockCalibrationError::ReferenceDomainMismatch {
            expected: watermark.reference_domain,
            actual: event_time.reference_domain,
        });
    }

    let ordering = if event_time.source_domain == watermark.source_domain
        && event_time.calibration_id == watermark.calibration_id
        && event_time.calibration_sha256 == watermark.calibration_sha256
    {
        if event_time.source_time_us < watermark.source_exclusive_watermark_us {
            EventWatermarkOrdering::DefinitelyLate
        } else {
            EventWatermarkOrdering::DefinitelyNotLate
        }
    } else if event_time.latest_reference_us < watermark.earliest_reference_bound_us {
        EventWatermarkOrdering::DefinitelyLate
    } else if event_time.earliest_reference_us >= watermark.latest_reference_bound_us {
        EventWatermarkOrdering::DefinitelyNotLate
    } else {
        EventWatermarkOrdering::Ambiguous
    };

    Ok(EventWatermarkComparisonV1 {
        ordering,
        event_time,
        watermark,
    })
}

/// Conservative meet of mapped exclusive watermarks from all required
/// sources, expressed in exactly one reference clock domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConservativeExclusiveWatermark {
    reference_domain: ClockDomainId,
    earliest_guaranteed_exclusive_us: i64,
    source_count: usize,
}

impl ConservativeExclusiveWatermark {
    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn earliest_guaranteed_exclusive_us(&self) -> i64 {
        self.earliest_guaranteed_exclusive_us
    }

    pub fn source_count(&self) -> usize {
        self.source_count
    }

    pub fn closes_half_open_interval(&self, end_exclusive_us: i64) -> bool {
        self.earliest_guaranteed_exclusive_us >= end_exclusive_us
    }
}

/// A policy-bound affine clock calibration with explicit offset and drift
/// uncertainty. The valid source-time range is inclusive at both ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClockCalibrationV1 {
    pub calibration_id: u64,
    pub source_domain: ClockDomainId,
    pub reference_domain: ClockDomainId,
    pub valid_from_source_us: i64,
    pub valid_through_source_us: i64,
    pub anchor_source_us: i64,
    pub anchor_reference_us: i64,
    /// Lower/upper offset error in microseconds at the calibration anchor.
    pub offset_lower_us: i64,
    pub offset_upper_us: i64,
    /// Maximum absolute relative clock-rate error, in parts per billion.
    pub max_abs_drift_ppb: u64,
    /// Knowledge-time coordinate at which this calibration became available.
    pub recorded_at_knowledge_us: i64,
    pub policy_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClockCalibrationError {
    ZeroClockDomain,
    ZeroCalibrationId,
    ZeroPolicyDigest,
    SameClockDomain,
    InvalidValidityRange,
    AnchorOutsideValidityRange,
    InvertedOffsetInterval,
    FutureCalibration {
        recorded_at_knowledge_us: i64,
        known_at_knowledge_us: i64,
    },
    SourceDomainMismatch {
        expected: ClockDomainId,
        actual: ClockDomainId,
    },
    OutsideCalibrationRange {
        source_time_us: i64,
        valid_from_source_us: i64,
        valid_through_source_us: i64,
    },
    EmptyWatermarkSet,
    EmptyCalibrationBatch,
    CalibrationBatchTimestampMismatch,
    NonIncreasingCalibrationTransactionTime {
        previous: i64,
        actual: i64,
    },
    DuplicateCalibrationId {
        calibration_id: u64,
    },
    NoCalibrationKnownForSourceTime {
        source_domain: ClockDomainId,
        reference_domain: ClockDomainId,
        source_time_us: i64,
        known_at_knowledge_us: i64,
    },
    InvalidCalibrationHistory,
    ReferenceDomainMismatch {
        expected: ClockDomainId,
        actual: ClockDomainId,
    },
    ArithmeticOverflow,
}

impl fmt::Display for ClockCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroClockDomain => write!(f, "clock domain ID must be nonzero"),
            Self::ZeroCalibrationId => write!(f, "clock calibration ID must be nonzero"),
            Self::ZeroPolicyDigest => write!(f, "clock calibration policy digest must be nonzero"),
            Self::SameClockDomain => write!(f, "source and reference clock domains must differ"),
            Self::InvalidValidityRange => {
                write!(f, "clock calibration validity range must be non-empty")
            }
            Self::AnchorOutsideValidityRange => write!(
                f,
                "clock calibration anchor must lie inside its validity range"
            ),
            Self::InvertedOffsetInterval => {
                write!(f, "clock calibration lower offset exceeds upper offset")
            }
            Self::FutureCalibration {
                recorded_at_knowledge_us,
                known_at_knowledge_us,
            } => write!(
                f,
                "clock calibration recorded at {recorded_at_knowledge_us} is not known at cutoff {known_at_knowledge_us}"
            ),
            Self::SourceDomainMismatch { expected, actual } => write!(
                f,
                "clock calibration expects source domain {:?}, received {:?}",
                expected, actual
            ),
            Self::OutsideCalibrationRange {
                source_time_us,
                valid_from_source_us,
                valid_through_source_us,
            } => write!(
                f,
                "source time {source_time_us} lies outside calibration validity [{valid_from_source_us}, {valid_through_source_us}]"
            ),
            Self::EmptyWatermarkSet => {
                write!(f, "at least one mapped source watermark is required")
            }
            Self::EmptyCalibrationBatch => {
                write!(f, "clock calibration transaction batch must not be empty")
            }
            Self::CalibrationBatchTimestampMismatch => write!(
                f,
                "all clock calibration revisions in a transaction must share recorded_at_knowledge_us"
            ),
            Self::NonIncreasingCalibrationTransactionTime { previous, actual } => write!(
                f,
                "clock calibration transaction time {actual} must be later than previous commit {previous}"
            ),
            Self::DuplicateCalibrationId { calibration_id } => {
                write!(f, "duplicate clock calibration ID {calibration_id}")
            }
            Self::NoCalibrationKnownForSourceTime {
                source_domain,
                reference_domain,
                source_time_us,
                known_at_knowledge_us,
            } => write!(
                f,
                "no calibration from {:?} to {:?} covers source time {source_time_us} and is known at {known_at_knowledge_us}",
                source_domain, reference_domain
            ),
            Self::InvalidCalibrationHistory => {
                write!(
                    f,
                    "clock calibration ledger history or indexes are inconsistent"
                )
            }
            Self::ReferenceDomainMismatch { expected, actual } => write!(
                f,
                "mapped watermark reference domain {:?} does not match {:?}",
                actual, expected
            ),
            Self::ArithmeticOverflow => {
                write!(f, "clock conversion or uncertainty arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for ClockCalibrationError {}

impl ClockCalibrationV1 {
    pub fn validate(&self) -> Result<(), ClockCalibrationError> {
        self.source_domain.validate()?;
        self.reference_domain.validate()?;
        if self.calibration_id == 0 {
            return Err(ClockCalibrationError::ZeroCalibrationId);
        }
        if self.source_domain == self.reference_domain {
            return Err(ClockCalibrationError::SameClockDomain);
        }
        if self.valid_from_source_us > self.valid_through_source_us {
            return Err(ClockCalibrationError::InvalidValidityRange);
        }
        if self.anchor_source_us < self.valid_from_source_us
            || self.anchor_source_us > self.valid_through_source_us
        {
            return Err(ClockCalibrationError::AnchorOutsideValidityRange);
        }
        if self.offset_lower_us > self.offset_upper_us {
            return Err(ClockCalibrationError::InvertedOffsetInterval);
        }
        if self.policy_sha256 == [0; 32] {
            return Err(ClockCalibrationError::ZeroPolicyDigest);
        }
        Ok(())
    }

    /// Stable digest binding the calibration parameters and their policy ID.
    pub fn calibration_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-CLOCK-CALIBRATION-V1\0");
        hasher.update(self.calibration_id.to_le_bytes());
        hasher.update(self.source_domain.0.to_le_bytes());
        hasher.update(self.reference_domain.0.to_le_bytes());
        hasher.update(self.valid_from_source_us.to_le_bytes());
        hasher.update(self.valid_through_source_us.to_le_bytes());
        hasher.update(self.anchor_source_us.to_le_bytes());
        hasher.update(self.anchor_reference_us.to_le_bytes());
        hasher.update(self.offset_lower_us.to_le_bytes());
        hasher.update(self.offset_upper_us.to_le_bytes());
        hasher.update(self.max_abs_drift_ppb.to_le_bytes());
        hasher.update(self.recorded_at_knowledge_us.to_le_bytes());
        hasher.update(self.policy_sha256);
        hasher.finalize().into()
    }

    /// Map one source-local timestamp to a conservative interval in the
    /// reference domain. Future knowledge, wrong domains, expired validity and
    /// arithmetic overflow all reject rather than guessing.
    pub fn map_time(
        &self,
        actual_source_domain: ClockDomainId,
        source_time_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedClockTime, ClockCalibrationError> {
        self.validate()?;
        if actual_source_domain != self.source_domain {
            return Err(ClockCalibrationError::SourceDomainMismatch {
                expected: self.source_domain,
                actual: actual_source_domain,
            });
        }
        if self.recorded_at_knowledge_us > known_at_knowledge_us {
            return Err(ClockCalibrationError::FutureCalibration {
                recorded_at_knowledge_us: self.recorded_at_knowledge_us,
                known_at_knowledge_us,
            });
        }
        if source_time_us < self.valid_from_source_us
            || source_time_us > self.valid_through_source_us
        {
            return Err(ClockCalibrationError::OutsideCalibrationRange {
                source_time_us,
                valid_from_source_us: self.valid_from_source_us,
                valid_through_source_us: self.valid_through_source_us,
            });
        }

        let delta = i128::from(source_time_us)
            .checked_sub(i128::from(self.anchor_source_us))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let drift_numerator = delta
            .unsigned_abs()
            .checked_mul(u128::from(self.max_abs_drift_ppb))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let denominator = DRIFT_DENOMINATOR as u128;
        let drift_floor = drift_numerator / denominator;
        let round_up = if drift_numerator % denominator == 0 {
            0
        } else {
            1
        };
        let drift_ceil = drift_floor
            .checked_add(round_up)
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let drift_uncertainty =
            i128::try_from(drift_ceil).map_err(|_| ClockCalibrationError::ArithmeticOverflow)?;

        let nominal_reference = i128::from(self.anchor_reference_us)
            .checked_add(delta)
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let earliest = nominal_reference
            .checked_add(i128::from(self.offset_lower_us))
            .and_then(|value| value.checked_sub(drift_uncertainty))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let latest = nominal_reference
            .checked_add(i128::from(self.offset_upper_us))
            .and_then(|value| value.checked_add(drift_uncertainty))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;

        Ok(MappedClockTime {
            source_domain: self.source_domain,
            reference_domain: self.reference_domain,
            source_time_us,
            earliest_reference_us: i64::try_from(earliest)
                .map_err(|_| ClockCalibrationError::ArithmeticOverflow)?,
            latest_reference_us: i64::try_from(latest)
                .map_err(|_| ClockCalibrationError::ArithmeticOverflow)?,
            calibration_id: self.calibration_id,
            calibration_sha256: self.calibration_sha256(),
        })
    }

    /// Map a source-local exclusive event-time watermark. Only the earliest
    /// possible mapped bound is safe for interval-completeness qualification.
    pub fn map_exclusive_watermark(
        &self,
        source_exclusive_watermark_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedExclusiveWatermark, ClockCalibrationError> {
        let mapped = self.map_time(
            self.source_domain,
            source_exclusive_watermark_us,
            known_at_knowledge_us,
        )?;
        Ok(MappedExclusiveWatermark {
            source_domain: mapped.source_domain,
            reference_domain: mapped.reference_domain,
            source_exclusive_watermark_us,
            earliest_reference_bound_us: mapped.earliest_reference_us,
            latest_reference_bound_us: mapped.latest_reference_us,
            calibration_id: mapped.calibration_id,
            calibration_sha256: mapped.calibration_sha256,
        })
    }
}

/// Append-only, as-known calibration history. For one source/reference
/// domain pair, the selected calibration is the newest revision visible at the
/// decision cutoff whose source-time validity covers the requested timestamp.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClockCalibrationLedger {
    revisions: Vec<ClockCalibrationV1>,
    revisions_by_domain_pair: BTreeMap<(ClockDomainId, ClockDomainId), Vec<usize>>,
    calibration_ids: BTreeSet<u64>,
    last_transaction_time_knowledge_us: Option<i64>,
}

impl ClockCalibrationLedger {
    pub fn revisions(&self) -> &[ClockCalibrationV1] {
        &self.revisions
    }

    pub fn append(&mut self, calibration: ClockCalibrationV1) -> Result<(), ClockCalibrationError> {
        if let Some(previous) = self.last_transaction_time_knowledge_us {
            if calibration.recorded_at_knowledge_us <= previous {
                return Err(
                    ClockCalibrationError::NonIncreasingCalibrationTransactionTime {
                        previous,
                        actual: calibration.recorded_at_knowledge_us,
                    },
                );
            }
        }
        self.append_unchecked(calibration)?;
        self.last_transaction_time_knowledge_us = Some(calibration.recorded_at_knowledge_us);
        Ok(())
    }

    /// Atomically append several calibrations recorded at one knowledge time.
    pub fn append_batch(
        &mut self,
        calibrations: impl IntoIterator<Item = ClockCalibrationV1>,
    ) -> Result<(), ClockCalibrationError> {
        let incoming: Vec<_> = calibrations.into_iter().collect();
        if incoming.is_empty() {
            return Err(ClockCalibrationError::EmptyCalibrationBatch);
        }
        let transaction_time = incoming[0].recorded_at_knowledge_us;
        if incoming
            .iter()
            .any(|calibration| calibration.recorded_at_knowledge_us != transaction_time)
        {
            return Err(ClockCalibrationError::CalibrationBatchTimestampMismatch);
        }
        if let Some(previous) = self.last_transaction_time_knowledge_us {
            if transaction_time <= previous {
                return Err(
                    ClockCalibrationError::NonIncreasingCalibrationTransactionTime {
                        previous,
                        actual: transaction_time,
                    },
                );
            }
        }

        let mut candidate = self.clone();
        for calibration in incoming {
            candidate.append_unchecked(calibration)?;
        }
        candidate.last_transaction_time_knowledge_us = Some(transaction_time);
        *self = candidate;
        Ok(())
    }

    fn append_unchecked(
        &mut self,
        calibration: ClockCalibrationV1,
    ) -> Result<(), ClockCalibrationError> {
        calibration.validate()?;
        if !self.calibration_ids.insert(calibration.calibration_id) {
            return Err(ClockCalibrationError::DuplicateCalibrationId {
                calibration_id: calibration.calibration_id,
            });
        }
        let index = self.revisions.len();
        self.revisions.push(calibration);
        self.revisions_by_domain_pair
            .entry((calibration.source_domain, calibration.reference_domain))
            .or_default()
            .push(index);
        Ok(())
    }

    fn latest_known_for_source_time(
        &self,
        source_domain: ClockDomainId,
        reference_domain: ClockDomainId,
        source_time_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<&ClockCalibrationV1, ClockCalibrationError> {
        let key = (source_domain, reference_domain);
        let chain = self.revisions_by_domain_pair.get(&key).ok_or(
            ClockCalibrationError::NoCalibrationKnownForSourceTime {
                source_domain,
                reference_domain,
                source_time_us,
                known_at_knowledge_us,
            },
        )?;
        let visible_count = chain.partition_point(|index| {
            self.revisions[*index].recorded_at_knowledge_us <= known_at_knowledge_us
        });
        chain[..visible_count]
            .iter()
            .rev()
            .map(|index| &self.revisions[*index])
            .find(|calibration| {
                source_time_us >= calibration.valid_from_source_us
                    && source_time_us <= calibration.valid_through_source_us
            })
            .ok_or(ClockCalibrationError::NoCalibrationKnownForSourceTime {
                source_domain,
                reference_domain,
                source_time_us,
                known_at_knowledge_us,
            })
    }

    pub fn map_time(
        &self,
        source_domain: ClockDomainId,
        reference_domain: ClockDomainId,
        source_time_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedClockTime, ClockCalibrationError> {
        self.latest_known_for_source_time(
            source_domain,
            reference_domain,
            source_time_us,
            known_at_knowledge_us,
        )?
        .map_time(source_domain, source_time_us, known_at_knowledge_us)
    }

    pub fn map_exclusive_watermark(
        &self,
        source_domain: ClockDomainId,
        reference_domain: ClockDomainId,
        source_exclusive_watermark_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedExclusiveWatermark, ClockCalibrationError> {
        self.latest_known_for_source_time(
            source_domain,
            reference_domain,
            source_exclusive_watermark_us,
            known_at_knowledge_us,
        )?
        .map_exclusive_watermark(source_exclusive_watermark_us, known_at_knowledge_us)
    }

    /// Rebuild the index from append-only records and compare the exact state.
    pub fn verify_consistency(&self) -> Result<(), ClockCalibrationError> {
        let mut rebuilt = Self::default();
        let mut offset = 0;
        while offset < self.revisions.len() {
            let transaction_time = self.revisions[offset].recorded_at_knowledge_us;
            let mut end = offset + 1;
            while end < self.revisions.len()
                && self.revisions[end].recorded_at_knowledge_us == transaction_time
            {
                end += 1;
            }
            rebuilt.append_batch(self.revisions[offset..end].iter().copied())?;
            offset = end;
        }
        if rebuilt != *self {
            return Err(ClockCalibrationError::InvalidCalibrationHistory);
        }
        Ok(())
    }

    pub fn state_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-CLOCK-CALIBRATION-LEDGER-V1\0");
        for calibration in &self.revisions {
            hasher.update(calibration.calibration_id.to_le_bytes());
            hasher.update(calibration.calibration_sha256());
        }
        hasher.finalize().into()
    }
}

/// Combine watermarks only after mapping every source into the same reference
/// clock. The minimum lower endpoint is the only conservative global bound.
pub fn conservative_min_exclusive_watermark(
    reference_domain: ClockDomainId,
    watermarks: &[MappedExclusiveWatermark],
) -> Result<ConservativeExclusiveWatermark, ClockCalibrationError> {
    reference_domain.validate()?;
    if watermarks.is_empty() {
        return Err(ClockCalibrationError::EmptyWatermarkSet);
    }
    let mut earliest = i64::MAX;
    for watermark in watermarks {
        if watermark.reference_domain != reference_domain {
            return Err(ClockCalibrationError::ReferenceDomainMismatch {
                expected: reference_domain,
                actual: watermark.reference_domain,
            });
        }
        earliest = earliest.min(watermark.earliest_reference_bound_us);
    }
    Ok(ConservativeExclusiveWatermark {
        reference_domain,
        earliest_guaranteed_exclusive_us: earliest,
        source_count: watermarks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> ClockCalibrationV1 {
        ClockCalibrationV1 {
            calibration_id: 7,
            source_domain: ClockDomainId(1),
            reference_domain: ClockDomainId(2),
            valid_from_source_us: 0,
            valid_through_source_us: 10_000_000,
            anchor_source_us: 0,
            anchor_reference_us: 10_000,
            offset_lower_us: 2,
            offset_upper_us: 4,
            max_abs_drift_ppb: 0,
            recorded_at_knowledge_us: 100,
            policy_sha256: [0xA5; 32],
        }
    }

    #[test]
    fn same_calibration_preserves_exclusive_watermark_equality_semantics() {
        let clock = calibration();
        let equal_event = clock.map_time(ClockDomainId(1), 1_000, 100).unwrap();
        let watermark = clock.map_exclusive_watermark(1_000, 100).unwrap();
        let equal = compare_event_time_to_exclusive_watermark(equal_event, watermark).unwrap();
        assert_eq!(equal.ordering(), EventWatermarkOrdering::DefinitelyNotLate);
        assert_eq!(
            equal.event_time().calibration_sha256(),
            watermark.calibration_sha256()
        );

        let earlier_event = clock.map_time(ClockDomainId(1), 999, 100).unwrap();
        let earlier = compare_event_time_to_exclusive_watermark(earlier_event, watermark).unwrap();
        assert_eq!(earlier.ordering(), EventWatermarkOrdering::DefinitelyLate);
    }

    #[test]
    fn different_calibration_intervals_produce_definite_or_ambiguous_ordering() {
        let first = calibration();
        let watermark = first.map_exclusive_watermark(1_000, 100).unwrap();

        let earlier_event = first.map_time(ClockDomainId(1), 900, 100).unwrap();
        assert_eq!(
            compare_event_time_to_exclusive_watermark(earlier_event, watermark)
                .unwrap()
                .ordering(),
            EventWatermarkOrdering::DefinitelyLate
        );

        let later_event = first.map_time(ClockDomainId(1), 1_100, 100).unwrap();
        assert_eq!(
            compare_event_time_to_exclusive_watermark(later_event, watermark)
                .unwrap()
                .ordering(),
            EventWatermarkOrdering::DefinitelyNotLate
        );

        let mut revised = calibration();
        revised.calibration_id = 8;
        revised.offset_lower_us = -5;
        revised.offset_upper_us = 5;
        let uncertain_event = revised.map_time(ClockDomainId(1), 1_000, 100).unwrap();
        let comparison =
            compare_event_time_to_exclusive_watermark(uncertain_event, watermark).unwrap();
        assert_eq!(comparison.ordering(), EventWatermarkOrdering::Ambiguous);
        assert_eq!(comparison.event_time().calibration_id(), 8);
        assert_eq!(comparison.watermark().calibration_id(), 7);
    }

    #[test]
    fn mapped_event_and_watermark_must_share_reference_domain() {
        let event = calibration()
            .map_time(ClockDomainId(1), 1_000, 100)
            .unwrap();
        let mut other_clock = calibration();
        other_clock.reference_domain = ClockDomainId(3);
        let watermark = other_clock.map_exclusive_watermark(1_000, 100).unwrap();

        assert_eq!(
            compare_event_time_to_exclusive_watermark(event, watermark),
            Err(ClockCalibrationError::ReferenceDomainMismatch {
                expected: ClockDomainId(3),
                actual: ClockDomainId(2),
            })
        );
    }

    #[test]
    fn exact_mapping_preserves_offset_uncertainty_interval() {
        let mapped = calibration()
            .map_time(ClockDomainId(1), 1_000, 100)
            .unwrap();

        assert_eq!(mapped.earliest_reference_us(), 11_002);
        assert_eq!(mapped.latest_reference_us(), 11_004);
        assert_eq!(mapped.calibration_id(), 7);
    }

    #[test]
    fn drift_uncertainty_expands_outward_and_rounds_up() {
        let mut clock = calibration();
        clock.anchor_reference_us = 0;
        clock.offset_lower_us = -1;
        clock.offset_upper_us = 1;
        clock.max_abs_drift_ppb = 1_000_000;

        let mapped = clock.map_time(ClockDomainId(1), 1_001, 100).unwrap();
        assert_eq!(mapped.earliest_reference_us(), 998);
        assert_eq!(mapped.latest_reference_us(), 1_004);
    }

    #[test]
    fn calibration_is_not_usable_before_its_knowledge_time() {
        let clock = calibration();
        assert_eq!(
            clock.map_time(ClockDomainId(1), 1_000, 99),
            Err(ClockCalibrationError::FutureCalibration {
                recorded_at_knowledge_us: 100,
                known_at_knowledge_us: 99,
            })
        );
    }

    #[test]
    fn wrong_clock_domain_and_expired_calibration_reject() {
        let clock = calibration();
        assert_eq!(
            clock.map_time(ClockDomainId(3), 1_000, 100),
            Err(ClockCalibrationError::SourceDomainMismatch {
                expected: ClockDomainId(1),
                actual: ClockDomainId(3),
            })
        );
        assert!(matches!(
            clock.map_time(ClockDomainId(1), 10_000_001, 100),
            Err(ClockCalibrationError::OutsideCalibrationRange { .. })
        ));
    }

    #[test]
    fn invalid_calibration_parameters_fail_closed() {
        let mut clock = calibration();
        clock.offset_lower_us = 5;
        clock.offset_upper_us = 4;
        assert_eq!(
            clock.validate(),
            Err(ClockCalibrationError::InvertedOffsetInterval)
        );

        let mut clock = calibration();
        clock.policy_sha256 = [0; 32];
        assert_eq!(
            clock.validate(),
            Err(ClockCalibrationError::ZeroPolicyDigest)
        );
    }

    #[test]
    fn watermark_uses_the_earliest_not_latest_mapped_bound() {
        let clock = calibration();
        let mapped = clock.map_exclusive_watermark(1_000, 100).unwrap();
        assert_eq!(mapped.earliest_reference_bound_us(), 11_002);
        assert_eq!(mapped.latest_reference_bound_us(), 11_004);
        assert!(mapped.closes_half_open_interval(11_002));
        assert!(!mapped.closes_half_open_interval(11_003));
    }

    #[test]
    fn cross_source_meet_uses_the_least_conservative_bound() {
        let first = calibration().map_exclusive_watermark(1_000, 100).unwrap();
        let mut second_clock = calibration();
        second_clock.calibration_id = 8;
        second_clock.anchor_reference_us = 9_500;
        let second = second_clock.map_exclusive_watermark(1_000, 100).unwrap();

        let combined =
            conservative_min_exclusive_watermark(ClockDomainId(2), &[first, second]).unwrap();
        assert_eq!(combined.earliest_guaranteed_exclusive_us(), 10_502);
        assert_eq!(combined.source_count(), 2);
        assert!(combined.closes_half_open_interval(10_502));
        assert!(!combined.closes_half_open_interval(10_503));
    }

    #[test]
    fn unmapped_reference_domains_cannot_be_aggregated() {
        let first = calibration().map_exclusive_watermark(1_000, 100).unwrap();
        let mut second_clock = calibration();
        second_clock.calibration_id = 8;
        second_clock.reference_domain = ClockDomainId(3);
        let second = second_clock.map_exclusive_watermark(1_000, 100).unwrap();

        assert_eq!(
            conservative_min_exclusive_watermark(ClockDomainId(2), &[first, second]),
            Err(ClockCalibrationError::ReferenceDomainMismatch {
                expected: ClockDomainId(2),
                actual: ClockDomainId(3),
            })
        );
    }

    #[test]
    fn empty_watermark_set_cannot_claim_completeness() {
        assert_eq!(
            conservative_min_exclusive_watermark(ClockDomainId(2), &[]),
            Err(ClockCalibrationError::EmptyWatermarkSet)
        );
    }

    #[test]
    fn calibration_ledger_selects_latest_revision_as_known() {
        let mut ledger = ClockCalibrationLedger::default();
        let first = calibration();
        ledger.append(first).unwrap();

        let mut second = calibration();
        second.calibration_id = 8;
        second.recorded_at_knowledge_us = 200;
        second.offset_lower_us = 10;
        second.offset_upper_us = 12;
        ledger.append(second).unwrap();

        let as_known_first = ledger
            .map_time(ClockDomainId(1), ClockDomainId(2), 1_000, 150)
            .unwrap();
        let as_known_latest = ledger
            .map_time(ClockDomainId(1), ClockDomainId(2), 1_000, 250)
            .unwrap();

        assert_eq!(as_known_first.calibration_id(), 7);
        assert_eq!(as_known_first.earliest_reference_us(), 11_002);
        assert_eq!(as_known_latest.calibration_id(), 8);
        assert_eq!(as_known_latest.earliest_reference_us(), 11_010);
    }

    #[test]
    fn calibration_ledger_does_not_select_a_future_revision() {
        let mut ledger = ClockCalibrationLedger::default();
        ledger.append(calibration()).unwrap();
        assert_eq!(
            ledger.map_time(ClockDomainId(1), ClockDomainId(2), 1_000, 99),
            Err(ClockCalibrationError::NoCalibrationKnownForSourceTime {
                source_domain: ClockDomainId(1),
                reference_domain: ClockDomainId(2),
                source_time_us: 1_000,
                known_at_knowledge_us: 99,
            })
        );
    }

    #[test]
    fn calibration_ledger_batch_is_atomic_and_enforces_transaction_order() {
        let mut ledger = ClockCalibrationLedger::default();
        let before = ledger.state_sha256();
        let first = calibration();
        let mut duplicate = calibration();
        duplicate.recorded_at_knowledge_us = 100;
        assert_eq!(
            ledger.append_batch([first, duplicate]),
            Err(ClockCalibrationError::DuplicateCalibrationId { calibration_id: 7 })
        );
        assert_eq!(ledger.revisions().len(), 0);
        assert_eq!(ledger.state_sha256(), before);

        ledger.append(first).unwrap();
        let mut earlier = calibration();
        earlier.calibration_id = 8;
        earlier.recorded_at_knowledge_us = 99;
        assert_eq!(
            ledger.append(earlier),
            Err(
                ClockCalibrationError::NonIncreasingCalibrationTransactionTime {
                    previous: 100,
                    actual: 99,
                }
            )
        );
    }

    #[test]
    fn calibration_ledger_replay_and_digest_are_deterministic() {
        let first = calibration();
        let mut second = calibration();
        second.calibration_id = 8;
        second.recorded_at_knowledge_us = 200;
        second.offset_lower_us = 5;
        second.offset_upper_us = 6;

        let mut left = ClockCalibrationLedger::default();
        let mut right = ClockCalibrationLedger::default();
        left.append(first).unwrap();
        right.append(first).unwrap();
        left.append(second).unwrap();
        right.append(second).unwrap();

        assert_eq!(left.state_sha256(), right.state_sha256());
        assert_eq!(left.verify_consistency(), Ok(()));
        assert_eq!(right.verify_consistency(), Ok(()));
    }

    #[test]
    fn arithmetic_overflow_rejects_unrepresentable_clock_mapping() {
        let mut clock = calibration();
        clock.anchor_reference_us = i64::MAX;
        assert_eq!(
            clock.map_time(ClockDomainId(1), 1, 100),
            Err(ClockCalibrationError::ArithmeticOverflow)
        );
    }
}
