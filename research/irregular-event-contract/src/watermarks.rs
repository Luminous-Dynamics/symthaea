//! Source progress watermarks and qualified interval completeness.
//!
//! A missing observation is not evidence of an absent event until every
//! required source has advanced an explicit exclusive event-time watermark
//! beyond the interval. Watermarks are source claims bound to a policy digest;
//! they are never inferred from the maximum event timestamp seen.
//!
//! A late event below a previously declared watermark is preserved as a
//! progress-contract breach. The caller must still append the raw EventRecord
//! to the bitemporal event ledger. This module records the breach receipt and
//! reopens affected completeness queries at knowledge times after arrival.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

use super::{EventRecord, SourceKeyPlaceholder};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SourceKey {
    pub entity_id: u64,
    pub modality_id: u32,
    pub source_id: u64,
}

impl SourceKey {
    pub fn validate(self) -> Result<(), WatermarkError> {
        if self.entity_id == 0 || self.modality_id == 0 || self.source_id == 0 {
            return Err(WatermarkError::ZeroSourceIdentity);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceProgressOperation {
    /// No on-time event with event_time_us < exclusive_event_time_us is
    /// expected under the named source policy.
    Advance {
        exclusive_event_time_us: i64,
    },
    /// The source has ended. No further events of any event time are expected.
    EndOfStream {
        final_exclusive_event_time_us: i64,
    },
}

impl SourceProgressOperation {
    fn exclusive_bound(self) -> i64 {
        match self {
            Self::Advance { exclusive_event_time_us } => exclusive_event_time_us,
            Self::EndOfStream { final_exclusive_event_time_us } => final_exclusive_event_time_us,
        }
    }

    fn is_terminal(self) -> bool {
        matches!(self, Self::EndOfStream { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceProgressRevision {
    pub progress_id: u64,
    pub source: SourceKey,
    /// Monotonic source incarnation. A new generation is explicit and cannot
    /// reset its event-time watermark.
    pub stream_generation: u64,
    /// Starts at one for each new generation and advances by exactly one.
    pub progress_sequence: u64,
    /// Transaction/knowledge time at which the progress claim was received.
    pub recorded_at_us: i64,
    /// Digest of the source contract or adapter policy supporting this claim.
    pub policy_sha256: [u8; 32],
    pub operation: SourceProgressOperation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LateEventKind {
    BelowExclusiveWatermark {
        exclusive_watermark_us: i64,
    },
    AfterEndOfStream {
        final_exclusive_watermark_us: i64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LateEventBreach {
    pub event: EventRecord,
    pub recorded_at_us: i64,
    pub kind: LateEventKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventArrivalClassification {
    OnTime,
    Late(LateEventKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceCompleteness {
    NoProgress,
    Incomplete {
        exclusive_watermark_us: i64,
    },
    Complete {
        exclusive_watermark_us: Option<i64>,
        terminal: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCompletenessEntry {
    pub source: SourceKey,
    pub status: SourceCompleteness,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletenessVerdict {
    Complete,
    Incomplete,
    Contested,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletenessReport {
    pub interval_start_us: i64,
    pub interval_end_exclusive_us: i64,
    pub known_at_us: i64,
    pub verdict: CompletenessVerdict,
    pub sources: Vec<SourceCompletenessEntry>,
    pub blocked_sources: Vec<SourceKey>,
    pub late_breaches: Vec<LateEventBreach>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WatermarkError {
    ZeroSourceIdentity,
    ZeroProgressId,
    ZeroStreamGeneration,
    ZeroProgressSequence,
    ZeroPolicyDigest,
    DuplicateProgressId { progress_id: u64 },
    EmptyBatch,
    BatchTimestampMismatch,
    NonIncreasingTransactionTime {
        previous: i64,
        actual: i64,
    },
    UnknownSourceGeneration {
        source: SourceKey,
    },
    StreamGenerationMismatch {
        expected: u64,
        actual: u64,
    },
    ProgressSequenceMismatch {
        expected: u64,
        actual: u64,
    },
    WatermarkRegression {
        previous: i64,
        actual: i64,
    },
    SourceAlreadyEnded {
        source: SourceKey,
    },
    DuplicateRequiredSource {
        source: SourceKey,
    },
    NoRequiredSources,
    InvalidCompletenessInterval,
    InvalidEventRecord,
    EventUnavailableAtReceipt {
        available_at_us: i64,
        recorded_at_us: i64,
    },
    DuplicateLateEventReceipt {
        event_id: u64,
    },
    ArithmeticOverflow,
    InvalidHistory,
}

impl fmt::Display for WatermarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSourceIdentity => write!(f, "watermark source entity/modality/source IDs must be nonzero"),
            Self::ZeroProgressId => write!(f, "source progress ID must be nonzero"),
            Self::ZeroStreamGeneration => write!(f, "source stream generation must be nonzero"),
            Self::ZeroProgressSequence => write!(f, "source progress sequence must be nonzero"),
            Self::ZeroPolicyDigest => write!(f, "source progress policy digest must be nonzero"),
            Self::DuplicateProgressId { progress_id } => write!(f, "duplicate source progress ID {progress_id}"),
            Self::EmptyBatch => write!(f, "source progress transaction batch must not be empty"),
            Self::BatchTimestampMismatch => write!(f, "all source progress records in a transaction must share recorded_at_us"),
            Self::NonIncreasingTransactionTime { previous, actual } => write!(f, "source progress transaction time {actual} must be later than previous commit {previous}"),
            Self::UnknownSourceGeneration { source } => write!(f, "source {:?} has no prior generation", source),
            Self::StreamGenerationMismatch { expected, actual } => write!(f, "expected source stream generation {expected}, received {actual}"),
            Self::ProgressSequenceMismatch { expected, actual } => write!(f, "expected progress sequence {expected}, received {actual}"),
            Self::WatermarkRegression { previous, actual } => write!(f, "exclusive source watermark regressed from {previous} to {actual}"),
            Self::SourceAlreadyEnded { source } => write!(f, "source {:?} already declared end-of-stream", source),
            Self::DuplicateRequiredSource { source } => write!(f, "required source {:?} appears more than once", source),
            Self::NoRequiredSources => write!(f, "completeness requires at least one source"),
            Self::InvalidCompletenessInterval => write!(f, "completeness interval must be non-empty and half-open"),
            Self::InvalidEventRecord => write!(f, "event record failed its event contract"),
            Self::EventUnavailableAtReceipt { available_at_us, recorded_at_us } => write!(f, "event availability {available_at_us} is after arrival receipt time {recorded_at_us}"),
            Self::DuplicateLateEventReceipt { event_id } => write!(f, "late event {event_id} already has a breach receipt"),
            Self::ArithmeticOverflow => write!(f, "watermark counter or digest length arithmetic overflowed"),
            Self::InvalidHistory => write!(f, "source watermark history or indexes are inconsistent"),
        }
    }
}
impl std::error::Error for WatermarkError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SourceWatermarkLedger {
    revisions: Vec<SourceProgressRevision>,
    revision_by_id: BTreeMap<u64, usize>,
    revisions_by_source: BTreeMap<SourceKey, Vec<usize>>,
    late_breaches: Vec<LateEventBreach>,
    late_event_ids: BTreeSet<u64>,
}

impl SourceWatermarkLedger {
    pub fn revisions(&self) -> &[SourceProgressRevision] {
        &self.revisions
    }

    pub fn late_breaches(&self) -> &[LateEventBreach] {
        &self.late_breaches
    }

    /// Append one source-progress revision after preflight, without cloning the log.
    pub fn append(&mut self, revision: SourceProgressRevision) -> Result<(), WatermarkError> {
        if let Some(previous) = self.revisions.last().map(|item| item.recorded_at_us) {
            if revision.recorded_at_us <= previous {
                return Err(WatermarkError::NonIncreasingTransactionTime {
                    previous,
                    actual: revision.recorded_at_us,
                });
            }
        }
        self.append_unchecked(revision)
    }

    /// Commit multiple sources' progress at one transaction time, atomically.
    pub fn append_batch(
        &mut self,
        revisions: impl IntoIterator<Item = SourceProgressRevision>,
    ) -> Result<(), WatermarkError> {
        let incoming: Vec<_> = revisions.into_iter().collect();
        if incoming.is_empty() {
            return Err(WatermarkError::EmptyBatch);
        }
        let transaction_time = incoming[0].recorded_at_us;
        if incoming.iter().any(|revision| revision.recorded_at_us != transaction_time) {
            return Err(WatermarkError::BatchTimestampMismatch);
        }
        if let Some(previous) = self.revisions.last().map(|item| item.recorded_at_us) {
            if transaction_time <= previous {
                return Err(WatermarkError::NonIncreasingTransactionTime {
                    previous,
                    actual: transaction_time,
                });
            }
        }

        let mut candidate = self.clone();
        for revision in incoming {
            candidate.append_unchecked(revision)?;
        }
        *self = candidate;
        Ok(())
    }

    fn append_unchecked(
        &mut self,
        revision: SourceProgressRevision,
    ) -> Result<(), WatermarkError> {
        revision.source.validate()?;
        if revision.progress_id == 0 {
            return Err(WatermarkError::ZeroProgressId);
        }
        if revision.stream_generation == 0 {
            return Err(WatermarkError::ZeroStreamGeneration);
        }
        if revision.progress_sequence == 0 {
            return Err(WatermarkError::ZeroProgressSequence);
        }
        if revision.policy_sha256 == [0; 32] {
            return Err(WatermarkError::ZeroPolicyDigest);
        }
        if self.revision_by_id.contains_key(&revision.progress_id) {
            return Err(WatermarkError::DuplicateProgressId {
                progress_id: revision.progress_id,
            });
        }

        let prior_index = self.revisions_by_source.get(&revision.source)
            .and_then(|chain| chain.last().copied());
        match prior_index {
            None => {
                if revision.stream_generation != 1 {
                    return Err(WatermarkError::StreamGenerationMismatch {
                        expected: 1,
                        actual: revision.stream_generation,
                    });
                }
                if revision.progress_sequence != 1 {
                    return Err(WatermarkError::ProgressSequenceMismatch {
                        expected: 1,
                        actual: revision.progress_sequence,
                    });
                }
            }
            Some(index) => {
                let previous = self.revisions[index];
                if previous.operation.is_terminal() {
                    return Err(WatermarkError::SourceAlreadyEnded {
                        source: revision.source,
                    });
                }
                let previous_bound = previous.operation.exclusive_bound();
                if revision.operation.exclusive_bound() < previous_bound {
                    return Err(WatermarkError::WatermarkRegression {
                        previous: previous_bound,
                        actual: revision.operation.exclusive_bound(),
                    });
                }
                if revision.stream_generation == previous.stream_generation {
                    let expected = previous.progress_sequence.checked_add(1)
                        .ok_or(WatermarkError::ArithmeticOverflow)?;
                    if revision.progress_sequence != expected {
                        return Err(WatermarkError::ProgressSequenceMismatch {
                            expected,
                            actual: revision.progress_sequence,
                        });
                    }
                } else {
                    let expected = previous.stream_generation.checked_add(1)
                        .ok_or(WatermarkError::ArithmeticOverflow)?;
                    if revision.stream_generation != expected {
                        return Err(WatermarkError::StreamGenerationMismatch {
                            expected,
                            actual: revision.stream_generation,
                        });
                    }
                    if revision.progress_sequence != 1 {
                        return Err(WatermarkError::ProgressSequenceMismatch {
                            expected: 1,
                            actual: revision.progress_sequence,
                        });
                    }
                }
            }
        }

        let index = self.revisions.len();
        self.revisions.push(revision);
        self.revision_by_id.insert(revision.progress_id, index);
        self.revisions_by_source.entry(revision.source).or_default().push(index);
        Ok(())
    }

    fn latest_before(
        &self,
        source: SourceKey,
        knowledge_time_us: i64,
        inclusive: bool,
    ) -> Option<&SourceProgressRevision> {
        let chain = self.revisions_by_source.get(&source)?;
        let count = chain.partition_point(|index| {
            let recorded = self.revisions[*index].recorded_at_us;
            if inclusive {
                recorded <= knowledge_time_us
            } else {
                recorded < knowledge_time_us
            }
        });
        count.checked_sub(1).map(|position| &self.revisions[chain[position]])
    }

    /// Assess whether every required source has closed the half-open
    /// event-time interval by the requested knowledge-time coordinate.
    pub fn completeness(
        &self,
        entity_id: u64,
        required_sources: &[SourceKey],
        interval_start_us: i64,
        interval_end_exclusive_us: i64,
        known_at_us: i64,
    ) -> Result<CompletenessReport, WatermarkError> {
        if entity_id == 0 {
            return Err(WatermarkError::ZeroSourceIdentity);
        }
        if interval_start_us >= interval_end_exclusive_us {
            return Err(WatermarkError::InvalidCompletenessInterval);
        }
        if required_sources.is_empty() {
            return Err(WatermarkError::NoRequiredSources);
        }
        self.verify_consistency()?;

        let mut unique = BTreeSet::new();
        let mut sources = Vec::with_capacity(required_sources.len());
        let mut blocked_sources = Vec::new();

        for source in required_sources {
            source.validate()?;
            if source.entity_id != entity_id {
                return Err(WatermarkError::ZeroSourceIdentity);
            }
            if !unique.insert(*source) {
                return Err(WatermarkError::DuplicateRequiredSource { source: *source });
            }
            let status = match self.latest_before(*source, known_at_us, true) {
                None => {
                    blocked_sources.push(*source);
                    SourceCompleteness::NoProgress
                }
                Some(progress) => {
                    let bound = progress.operation.exclusive_bound();
                    if progress.operation.is_terminal() || bound >= interval_end_exclusive_us {
                        SourceCompleteness::Complete {
                            exclusive_watermark_us: Some(bound),
                            terminal: progress.operation.is_terminal(),
                        }
                    } else {
                        blocked_sources.push(*source);
                        SourceCompleteness::Incomplete {
                            exclusive_watermark_us: bound,
                        }
                    }
                }
            };
            sources.push(SourceCompletenessEntry {
                source: *source,
                status,
            });
        }

        let late_breaches: Vec<_> = self.late_breaches.iter()
            .filter(|breach| {
                breach.recorded_at_us <= known_at_us
                    && breach.event.entity_id == entity_id
                    && breach.event.event_time_us >= interval_start_us
                    && breach.event.event_time_us < interval_end_exclusive_us
                    && unique.contains(&SourceKey {
                        entity_id: breach.event.entity_id,
                        modality_id: breach.event.modality_id,
                        source_id: breach.event.source_id,
                    })
            })
            .copied()
            .collect();

        let verdict = if !late_breaches.is_empty() {
            CompletenessVerdict::Contested
        } else if blocked_sources.is_empty() {
            CompletenessVerdict::Complete
        } else {
            CompletenessVerdict::Incomplete
        };

        Ok(CompletenessReport {
            interval_start_us,
            interval_end_exclusive_us,
            known_at_us,
            verdict,
            sources,
            blocked_sources,
            late_breaches,
        })
    }

    /// Observe the arrival of an event. On-time events are not stored here;
    /// their authoritative record belongs in BitemporalEventLedger. A late
    /// event emits a durable-in-this-reference-model breach receipt and
    /// reopens matching completeness queries after its recorded_at_us.
    pub fn record_event_arrival(
        &mut self,
        event: EventRecord,
        recorded_at_us: i64,
    ) -> Result<EventArrivalClassification, WatermarkError> {
        event.validate().map_err(|_| WatermarkError::InvalidEventRecord)?;
        if event.available_at_us > recorded_at_us {
            return Err(WatermarkError::EventUnavailableAtReceipt {
                available_at_us: event.available_at_us,
                recorded_at_us,
            });
        }
        if self.late_event_ids.contains(&event.event_id) {
            return Err(WatermarkError::DuplicateLateEventReceipt {
                event_id: event.event_id,
            });
        }

        let source = SourceKey {
            entity_id: event.entity_id,
            modality_id: event.modality_id,
            source_id: event.source_id,
        };
        let classification = match self.latest_before(source, recorded_at_us, false) {
            None => EventArrivalClassification::OnTime,
            Some(progress) => match progress.operation {
                SourceProgressOperation::EndOfStream { final_exclusive_event_time_us } => {
                    EventArrivalClassification::Late(LateEventKind::AfterEndOfStream {
                        final_exclusive_watermark_us: final_exclusive_event_time_us,
                    })
                }
                SourceProgressOperation::Advance { exclusive_event_time_us }
                    if event.event_time_us < exclusive_event_time_us =>
                {
                    EventArrivalClassification::Late(LateEventKind::BelowExclusiveWatermark {
                        exclusive_watermark_us: exclusive_event_time_us,
                    })
                }
                SourceProgressOperation::Advance { .. } => EventArrivalClassification::OnTime,
            },
        };

        if let EventArrivalClassification::Late(kind) = classification {
            self.late_event_ids.insert(event.event_id);
            self.late_breaches.push(LateEventBreach {
                event,
                recorded_at_us,
                kind,
            });
        }
        Ok(classification)
    }

    pub fn verify_consistency(&self) -> Result<(), WatermarkError> {
        let mut rebuilt = Self::default();
        let mut offset = 0;
        while offset < self.revisions.len() {
            let recorded_at = self.revisions[offset].recorded_at_us;
            let mut end = offset + 1;
            while end < self.revisions.len()
                && self.revisions[end].recorded_at_us == recorded_at
            {
                end += 1;
            }
            rebuilt.append_batch(self.revisions[offset..end].iter().copied())?;
            offset = end;
        }
        for breach in &self.late_breaches {
            match rebuilt.record_event_arrival(breach.event, breach.recorded_at_us)? {
                EventArrivalClassification::Late(kind) if kind == breach.kind => {}
                _ => return Err(WatermarkError::InvalidHistory),
            }
        }
        if rebuilt != *self {
            return Err(WatermarkError::InvalidHistory);
        }
        Ok(())
    }

    pub fn state_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-SOURCE-WATERMARK-LEDGER-V1\0");
        for revision in &self.revisions {
            hasher.update(revision.progress_id.to_le_bytes());
            hasher.update(revision.source.entity_id.to_le_bytes());
            hasher.update(revision.source.modality_id.to_le_bytes());
            hasher.update(revision.source.source_id.to_le_bytes());
            hasher.update(revision.stream_generation.to_le_bytes());
            hasher.update(revision.progress_sequence.to_le_bytes());
            hasher.update(revision.recorded_at_us.to_le_bytes());
            hasher.update(revision.policy_sha256);
            match revision.operation {
                SourceProgressOperation::Advance { exclusive_event_time_us } => {
                    hasher.update([1]);
                    hasher.update(exclusive_event_time_us.to_le_bytes());
                }
                SourceProgressOperation::EndOfStream { final_exclusive_event_time_us } => {
                    hasher.update([2]);
                    hasher.update(final_exclusive_event_time_us.to_le_bytes());
                }
            }
        }
        for breach in &self.late_breaches {
            hasher.update(breach.event.event_id.to_le_bytes());
            hasher.update(breach.recorded_at_us.to_le_bytes());
            super::hash_event_record(&mut hasher, &breach.event);
            match breach.kind {
                LateEventKind::BelowExclusiveWatermark { exclusive_watermark_us } => {
                    hasher.update([1]);
                    hasher.update(exclusive_watermark_us.to_le_bytes());
                }
                LateEventKind::AfterEndOfStream { final_exclusive_watermark_us } => {
                    hasher.update([2]);
                    hasher.update(final_exclusive_watermark_us.to_le_bytes());
                }
            }
        }
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(modality_id: u32) -> SourceKey {
        SourceKey { entity_id: 1, modality_id, source_id: u64::from(modality_id) }
    }

    fn progress(
        progress_id: u64,
        modality_id: u32,
        generation: u64,
        sequence: u64,
        recorded_at_us: i64,
        watermark_us: i64,
        terminal: bool,
    ) -> SourceProgressRevision {
        SourceProgressRevision {
            progress_id,
            source: source(modality_id),
            stream_generation: generation,
            progress_sequence: sequence,
            recorded_at_us,
            policy_sha256: [7; 32],
            operation: if terminal {
                SourceProgressOperation::EndOfStream {
                    final_exclusive_event_time_us: watermark_us,
                }
            } else {
                SourceProgressOperation::Advance {
                    exclusive_event_time_us: watermark_us,
                }
            },
        }
    }

    fn event(event_id: u64, modality_id: u32, event_time_us: i64, available_at_us: i64) -> EventRecord {
        EventRecord {
            event_id,
            entity_id: 1,
            modality_id,
            source_id: u64::from(modality_id),
            event_time_us,
            available_at_us,
            source_sequence: event_id,
            unit_code: 1,
            measurement: Measurement::Observed { scaled_value: 1, decimal_scale: 0 },
            source_sha256: [event_id as u8; 32],
        }
    }

    #[test]
    fn no_watermark_is_incomplete_not_absent() {
        let ledger = SourceWatermarkLedger::default();
        let report = ledger.completeness(1, &[source(1)], 0, 100, 100).unwrap();
        assert_eq!(report.verdict, CompletenessVerdict::Incomplete);
        assert_eq!(report.sources[0].status, SourceCompleteness::NoProgress);
    }

    #[test]
    fn interval_is_bounded_by_least_advanced_required_source() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append_batch([
            progress(1, 1, 1, 1, 10, 120, false),
            progress(2, 2, 1, 1, 10, 80, false),
        ]).unwrap();
        let report = ledger.completeness(1, &[source(1), source(2)], 0, 100, 10).unwrap();
        assert_eq!(report.verdict, CompletenessVerdict::Incomplete);
        assert_eq!(report.blocked_sources, vec![source(2)]);
    }

    #[test]
    fn all_required_sources_must_advance_to_the_exclusive_end() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append_batch([
            progress(1, 1, 1, 1, 10, 100, false),
            progress(2, 2, 1, 1, 10, 100, false),
        ]).unwrap();
        let report = ledger.completeness(1, &[source(1), source(2)], 0, 100, 10).unwrap();
        assert_eq!(report.verdict, CompletenessVerdict::Complete);
        assert!(report.blocked_sources.is_empty());
    }

    #[test]
    fn earlier_knowledge_time_does_not_see_later_watermark() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 50, false)).unwrap();
        ledger.append(progress(2, 1, 1, 2, 30, 100, false)).unwrap();
        assert_eq!(ledger.completeness(1, &[source(1)], 0, 100, 20).unwrap().verdict, CompletenessVerdict::Incomplete);
        assert_eq!(ledger.completeness(1, &[source(1)], 0, 100, 30).unwrap().verdict, CompletenessVerdict::Complete);
    }

    #[test]
    fn event_below_prior_exclusive_watermark_is_preserved_as_breach() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 100, false)).unwrap();
        let event = event(1, 1, 50, 20);
        assert_eq!(
            ledger.record_event_arrival(event, 20),
            Ok(EventArrivalClassification::Late(LateEventKind::BelowExclusiveWatermark {
                exclusive_watermark_us: 100,
            }))
        );
        let report = ledger.completeness(1, &[source(1)], 0, 100, 20).unwrap();
        assert_eq!(report.verdict, CompletenessVerdict::Contested);
        assert_eq!(report.late_breaches.len(), 1);
    }

    #[test]
    fn event_exactly_at_exclusive_watermark_is_not_late() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 100, false)).unwrap();
        assert_eq!(
            ledger.record_event_arrival(event(1, 1, 100, 20), 20),
            Ok(EventArrivalClassification::OnTime)
        );
        assert!(ledger.late_breaches().is_empty());
    }

    #[test]
    fn terminal_source_closes_future_intervals_without_inventing_a_watermark() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 50, true)).unwrap();
        let report = ledger.completeness(1, &[source(1)], 1_000, 2_000, 10).unwrap();
        assert_eq!(report.verdict, CompletenessVerdict::Complete);
        assert_eq!(report.sources[0].status, SourceCompleteness::Complete {
            exclusive_watermark_us: Some(50),
            terminal: true,
        });
    }

    #[test]
    fn watermark_regression_is_rejected_without_mutation() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 100, false)).unwrap();
        let before = ledger.state_sha256();
        assert_eq!(
            ledger.append(progress(2, 1, 1, 2, 20, 90, false)),
            Err(WatermarkError::WatermarkRegression { previous: 100, actual: 90 })
        );
        assert_eq!(ledger.state_sha256(), before);
    }

    #[test]
    fn new_generation_is_explicit_and_cannot_reset_watermark() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 100, false)).unwrap();
        assert_eq!(
            ledger.append(progress(2, 1, 2, 1, 20, 90, false)),
            Err(WatermarkError::WatermarkRegression { previous: 100, actual: 90 })
        );
        ledger.append(progress(3, 1, 2, 1, 21, 120, false)).unwrap();
        assert_eq!(ledger.revisions().last().unwrap().stream_generation, 2);
    }

    #[test]
    fn zero_policy_duplicate_sequences_and_terminal_append_fail_closed() {
        let mut ledger = SourceWatermarkLedger::default();
        let mut invalid = progress(1, 1, 1, 1, 10, 100, false);
        invalid.policy_sha256 = [0; 32];
        assert_eq!(ledger.append(invalid), Err(WatermarkError::ZeroPolicyDigest));
        ledger.append(progress(1, 1, 1, 1, 10, 100, true)).unwrap();
        assert_eq!(
            ledger.append(progress(2, 1, 1, 2, 20, 120, false)),
            Err(WatermarkError::SourceAlreadyEnded { source: source(1) })
        );
    }

    #[test]
    fn failed_multi_source_batch_is_atomic() {
        let mut ledger = SourceWatermarkLedger::default();
        let before = ledger.state_sha256();
        assert_eq!(
            ledger.append_batch([
                progress(1, 1, 1, 1, 10, 100, false),
                progress(2, 2, 1, 2, 10, 100, false),
            ]),
            Err(WatermarkError::ProgressSequenceMismatch { expected: 1, actual: 2 })
        );
        assert!(ledger.revisions().is_empty());
        assert_eq!(ledger.state_sha256(), before);
    }

    #[test]
    fn historical_completeness_is_not_rewritten_by_late_arrival() {
        let mut ledger = SourceWatermarkLedger::default();
        ledger.append(progress(1, 1, 1, 1, 10, 100, false)).unwrap();
        ledger.record_event_arrival(event(1, 1, 50, 20), 20).unwrap();
        assert_eq!(ledger.completeness(1, &[source(1)], 0, 100, 19).unwrap().verdict, CompletenessVerdict::Complete);
        assert_eq!(ledger.completeness(1, &[source(1)], 0, 100, 20).unwrap().verdict, CompletenessVerdict::Contested);
    }

    #[test]
    fn replay_rebuild_and_state_digest_are_deterministic() {
        let mut left = SourceWatermarkLedger::default();
        let mut right = SourceWatermarkLedger::default();
        let revisions = [
            progress(1, 1, 1, 1, 10, 100, false),
            progress(2, 2, 1, 1, 10, 120, false),
        ];
        left.append_batch(revisions).unwrap();
        right.append_batch(revisions).unwrap();
        left.record_event_arrival(event(1, 1, 50, 20), 20).unwrap();
        right.record_event_arrival(event(1, 1, 50, 20), 20).unwrap();
        assert_eq!(left.state_sha256(), right.state_sha256());
        assert_eq!(left.verify_consistency(), Ok(()));
    }
}
