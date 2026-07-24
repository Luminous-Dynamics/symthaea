// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-chain-preserving operational evidence compaction.
//!
//! Compaction may remove old payloads from the hot path, but it must not erase
//! their cryptographic history. A compacted checkpoint retains the exact prefix
//! chain head, a bounded tail of complete records, and the final head. The
//! prefix head still needs an external trusted anchor; this module never claims
//! that a digest alone proves the unavailable historical payloads.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EVIDENCE_RECORD_SCHEMA: &str = "symthaea.fabrication.evidence-record.v1";
pub const COMPACTED_EVIDENCE_SCHEMA: &str = "symthaea.fabrication.compacted-evidence.v1";
pub const MAX_EVIDENCE_KIND_BYTES: usize = 128;
pub const MAX_EVIDENCE_RECORDS: usize = 1_000_000;
pub const MAX_RETAINED_TAIL_RECORDS: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub schema_version: String,
    pub sequence: u64,
    pub recorded_at_unix_ms: u64,
    pub kind: String,
    pub subject_digest: Sha256Digest,
    pub previous_head: Sha256Digest,
    pub record_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceJournal {
    pub records: Vec<EvidenceRecord>,
}

impl Default for EvidenceJournal {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactedEvidence {
    pub schema_version: String,
    pub total_count: u64,
    pub prefix_count: u64,
    pub prefix_head: Sha256Digest,
    pub prefix_last_recorded_at_unix_ms: Option<u64>,
    pub retained_tail: Vec<EvidenceRecord>,
    pub final_head: Sha256Digest,
    pub predecessor_checkpoint_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCompactionPolicy {
    pub minimum_retained_tail: usize,
    pub maximum_retained_tail: usize,
}

impl Default for EvidenceCompactionPolicy {
    fn default() -> Self {
        Self {
            minimum_retained_tail: 64,
            maximum_retained_tail: 4_096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceCompactionError {
    CapacityExceeded,
    UnsupportedSchema,
    InvalidSequence { expected: u64, actual: u64 },
    InvalidKind,
    TimeRegressed { previous: u64, current: u64 },
    PreviousHeadMismatch { sequence: u64 },
    RecordDigestMismatch { sequence: u64 },
    InvalidPolicy,
    RetainedTailTooSmall { actual: usize, minimum: usize },
    RetainedTailTooLarge { actual: usize, maximum: usize },
    InvalidCounts,
    PrefixHeadMismatch,
    FinalHeadMismatch,
    DuplicateRecordDigest,
    Encoding(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCompactionTracker {
    latest_total_count: Option<u64>,
    latest_final_head: Option<Sha256Digest>,
    latest_checkpoint_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceCompactionTrackingError {
    InvalidTrackerState,
    Encoding(String),
    Invalid(EvidenceCompactionError),
    CountRollback { latest: u64, proposed: u64 },
    SameCountSubstitution,
    MissingPredecessor,
}

impl EvidenceJournal {
    pub fn append(
        &mut self,
        recorded_at_unix_ms: u64,
        kind: impl Into<String>,
        subject_digest: Sha256Digest,
    ) -> Result<Sha256Digest, EvidenceCompactionError> {
        self.validate()?;
        if self.records.len() >= MAX_EVIDENCE_RECORDS {
            return Err(EvidenceCompactionError::CapacityExceeded);
        }
        if self
            .records
            .last()
            .is_some_and(|record| recorded_at_unix_ms < record.recorded_at_unix_ms)
        {
            return Err(EvidenceCompactionError::TimeRegressed {
                previous: self
                    .records
                    .last()
                    .map_or(0, |record| record.recorded_at_unix_ms),
                current: recorded_at_unix_ms,
            });
        }
        let kind = kind.into();
        validate_kind(&kind)?;
        let previous_head = self.head();
        let sequence = self.records.len() as u64 + 1;
        let record_digest = digest_record_fields(
            sequence,
            recorded_at_unix_ms,
            &kind,
            subject_digest,
            previous_head,
        )?;
        self.records.push(EvidenceRecord {
            schema_version: EVIDENCE_RECORD_SCHEMA.into(),
            sequence,
            recorded_at_unix_ms,
            kind,
            subject_digest,
            previous_head,
            record_digest,
        });
        Ok(record_digest)
    }

    pub fn validate(&self) -> Result<(), EvidenceCompactionError> {
        if self.records.len() > MAX_EVIDENCE_RECORDS {
            return Err(EvidenceCompactionError::CapacityExceeded);
        }
        let mut previous_head = empty_evidence_head();
        let mut previous_time = None;
        let mut digests = BTreeSet::new();
        for (index, record) in self.records.iter().enumerate() {
            if record.schema_version != EVIDENCE_RECORD_SCHEMA {
                return Err(EvidenceCompactionError::UnsupportedSchema);
            }
            let expected_sequence = index as u64 + 1;
            if record.sequence != expected_sequence {
                return Err(EvidenceCompactionError::InvalidSequence {
                    expected: expected_sequence,
                    actual: record.sequence,
                });
            }
            validate_kind(&record.kind)?;
            if previous_time.is_some_and(|time| record.recorded_at_unix_ms < time) {
                return Err(EvidenceCompactionError::TimeRegressed {
                    previous: previous_time.unwrap_or_default(),
                    current: record.recorded_at_unix_ms,
                });
            }
            if record.previous_head != previous_head {
                return Err(EvidenceCompactionError::PreviousHeadMismatch {
                    sequence: record.sequence,
                });
            }
            let expected_digest = digest_record_fields(
                record.sequence,
                record.recorded_at_unix_ms,
                &record.kind,
                record.subject_digest,
                record.previous_head,
            )?;
            if record.record_digest != expected_digest {
                return Err(EvidenceCompactionError::RecordDigestMismatch {
                    sequence: record.sequence,
                });
            }
            if !digests.insert(record.record_digest) {
                return Err(EvidenceCompactionError::DuplicateRecordDigest);
            }
            previous_head = record.record_digest;
            previous_time = Some(record.recorded_at_unix_ms);
        }
        Ok(())
    }

    pub fn head(&self) -> Sha256Digest {
        self.records
            .last()
            .map_or_else(empty_evidence_head, |record| record.record_digest)
    }
}

impl CompactedEvidence {
    pub fn validate(
        &self,
        policy: &EvidenceCompactionPolicy,
    ) -> Result<(), EvidenceCompactionError> {
        validate_policy(policy)?;
        if self.schema_version != COMPACTED_EVIDENCE_SCHEMA {
            return Err(EvidenceCompactionError::UnsupportedSchema);
        }
        if self.retained_tail.len() < policy.minimum_retained_tail
            && self.total_count as usize > self.retained_tail.len()
        {
            return Err(EvidenceCompactionError::RetainedTailTooSmall {
                actual: self.retained_tail.len(),
                minimum: policy.minimum_retained_tail,
            });
        }
        if self.retained_tail.len() > policy.maximum_retained_tail {
            return Err(EvidenceCompactionError::RetainedTailTooLarge {
                actual: self.retained_tail.len(),
                maximum: policy.maximum_retained_tail,
            });
        }
        if self.total_count != self.prefix_count + self.retained_tail.len() as u64 {
            return Err(EvidenceCompactionError::InvalidCounts);
        }
        if (self.prefix_count == 0
            && (self.prefix_head != empty_evidence_head()
                || self.prefix_last_recorded_at_unix_ms.is_some()))
            || (self.prefix_count > 0
                && (self.prefix_head == empty_evidence_head()
                    || self.prefix_last_recorded_at_unix_ms.is_none()))
        {
            return Err(EvidenceCompactionError::PrefixHeadMismatch);
        }
        if self.total_count == 0 {
            if self.prefix_count != 0
                || !self.retained_tail.is_empty()
                || self.prefix_head != empty_evidence_head()
                || self.prefix_last_recorded_at_unix_ms.is_some()
                || self.final_head != empty_evidence_head()
            {
                return Err(EvidenceCompactionError::InvalidCounts);
            }
            return Ok(());
        }
        let mut head = self.prefix_head;
        let mut previous_time = self.prefix_last_recorded_at_unix_ms;
        let mut digests = BTreeSet::new();
        for (offset, record) in self.retained_tail.iter().enumerate() {
            if record.schema_version != EVIDENCE_RECORD_SCHEMA {
                return Err(EvidenceCompactionError::UnsupportedSchema);
            }
            let expected_sequence = self.prefix_count + offset as u64 + 1;
            if record.sequence != expected_sequence {
                return Err(EvidenceCompactionError::InvalidSequence {
                    expected: expected_sequence,
                    actual: record.sequence,
                });
            }
            validate_kind(&record.kind)?;
            if record.previous_head != head {
                return Err(EvidenceCompactionError::PreviousHeadMismatch {
                    sequence: record.sequence,
                });
            }
            if previous_time.is_some_and(|time| record.recorded_at_unix_ms < time) {
                return Err(EvidenceCompactionError::TimeRegressed {
                    previous: previous_time.unwrap_or_default(),
                    current: record.recorded_at_unix_ms,
                });
            }
            let expected_digest = digest_record_fields(
                record.sequence,
                record.recorded_at_unix_ms,
                &record.kind,
                record.subject_digest,
                record.previous_head,
            )?;
            if record.record_digest != expected_digest {
                return Err(EvidenceCompactionError::RecordDigestMismatch {
                    sequence: record.sequence,
                });
            }
            if !digests.insert(record.record_digest) {
                return Err(EvidenceCompactionError::DuplicateRecordDigest);
            }
            head = record.record_digest;
            previous_time = Some(record.recorded_at_unix_ms);
        }
        if head != self.final_head {
            return Err(EvidenceCompactionError::FinalHeadMismatch);
        }
        Ok(())
    }
}

impl EvidenceCompactionTracker {
    pub fn validate(&self) -> Result<(), EvidenceCompactionTrackingError> {
        match (
            self.latest_total_count,
            self.latest_final_head,
            self.latest_checkpoint_digest,
        ) {
            (None, None, None) => Ok(()),
            (Some(_), Some(final_head), Some(checkpoint))
                if final_head.0 != [0; 32] && checkpoint.0 != [0; 32] =>
            {
                Ok(())
            }
            _ => Err(EvidenceCompactionTrackingError::InvalidTrackerState),
        }
    }

    pub fn accept(
        &mut self,
        compacted: &CompactedEvidence,
        policy: &EvidenceCompactionPolicy,
    ) -> Result<Sha256Digest, EvidenceCompactionTrackingError> {
        self.validate()?;
        compacted
            .validate(policy)
            .map_err(EvidenceCompactionTrackingError::Invalid)?;
        let digest = digest_compacted_evidence(compacted, policy)
            .map_err(EvidenceCompactionTrackingError::Invalid)?;
        if let Some(latest_count) = self.latest_total_count {
            if compacted.total_count < latest_count {
                return Err(EvidenceCompactionTrackingError::CountRollback {
                    latest: latest_count,
                    proposed: compacted.total_count,
                });
            }
            if compacted.total_count == latest_count {
                if self.latest_checkpoint_digest == Some(digest) {
                    return Ok(digest);
                }
                return Err(EvidenceCompactionTrackingError::SameCountSubstitution);
            }
            if compacted.predecessor_checkpoint_digest != self.latest_checkpoint_digest {
                return Err(EvidenceCompactionTrackingError::MissingPredecessor);
            }
        } else if compacted.predecessor_checkpoint_digest.is_some() {
            return Err(EvidenceCompactionTrackingError::MissingPredecessor);
        }
        self.latest_total_count = Some(compacted.total_count);
        self.latest_final_head = Some(compacted.final_head);
        self.latest_checkpoint_digest = Some(digest);
        Ok(digest)
    }

    pub fn latest_checkpoint_digest(&self) -> Option<Sha256Digest> {
        self.latest_checkpoint_digest
    }
    pub fn latest_final_head(&self) -> Option<Sha256Digest> {
        self.latest_final_head
    }
}

pub fn digest_evidence_compaction_tracker(
    tracker: &EvidenceCompactionTracker,
) -> Result<Sha256Digest, EvidenceCompactionTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| EvidenceCompactionTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-compaction-tracker.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn compact_evidence(
    journal: &EvidenceJournal,
    retained_tail_count: usize,
    predecessor_checkpoint_digest: Option<Sha256Digest>,
    policy: &EvidenceCompactionPolicy,
) -> Result<CompactedEvidence, EvidenceCompactionError> {
    journal.validate()?;
    validate_policy(policy)?;
    if retained_tail_count > policy.maximum_retained_tail {
        return Err(EvidenceCompactionError::RetainedTailTooLarge {
            actual: retained_tail_count,
            maximum: policy.maximum_retained_tail,
        });
    }
    if retained_tail_count < policy.minimum_retained_tail
        && journal.records.len() > retained_tail_count
    {
        return Err(EvidenceCompactionError::RetainedTailTooSmall {
            actual: retained_tail_count,
            minimum: policy.minimum_retained_tail,
        });
    }
    let split = journal.records.len().saturating_sub(retained_tail_count);
    let prefix_head = if split == 0 {
        empty_evidence_head()
    } else {
        journal.records[split - 1].record_digest
    };
    let prefix_last_recorded_at_unix_ms = if split == 0 {
        None
    } else {
        Some(journal.records[split - 1].recorded_at_unix_ms)
    };
    let compacted = CompactedEvidence {
        schema_version: COMPACTED_EVIDENCE_SCHEMA.into(),
        total_count: journal.records.len() as u64,
        prefix_count: split as u64,
        prefix_head,
        prefix_last_recorded_at_unix_ms,
        retained_tail: journal.records[split..].to_vec(),
        final_head: journal.head(),
        predecessor_checkpoint_digest,
    };
    compacted.validate(policy)?;
    Ok(compacted)
}

pub fn digest_compacted_evidence(
    compacted: &CompactedEvidence,
    policy: &EvidenceCompactionPolicy,
) -> Result<Sha256Digest, EvidenceCompactionError> {
    compacted.validate(policy)?;
    let bytes = serde_json::to_vec(compacted)
        .map_err(|error| EvidenceCompactionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.compacted-evidence-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn empty_evidence_head() -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-empty.v1\0");
    hasher.finalize()
}

fn digest_record_fields(
    sequence: u64,
    recorded_at_unix_ms: u64,
    kind: &str,
    subject_digest: Sha256Digest,
    previous_head: Sha256Digest,
) -> Result<Sha256Digest, EvidenceCompactionError> {
    validate_kind(kind)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-record-digest.v1\0");
    hasher.update(&sequence.to_le_bytes());
    hasher.update(&recorded_at_unix_ms.to_le_bytes());
    hasher.update(&(kind.len() as u64).to_le_bytes());
    hasher.update(kind.as_bytes());
    hasher.update(&subject_digest.0);
    hasher.update(&previous_head.0);
    Ok(hasher.finalize())
}

fn validate_policy(policy: &EvidenceCompactionPolicy) -> Result<(), EvidenceCompactionError> {
    if policy.maximum_retained_tail == 0
        || policy.minimum_retained_tail > policy.maximum_retained_tail
        || policy.maximum_retained_tail > MAX_RETAINED_TAIL_RECORDS
    {
        return Err(EvidenceCompactionError::InvalidPolicy);
    }
    Ok(())
}

fn validate_kind(kind: &str) -> Result<(), EvidenceCompactionError> {
    if kind.trim().is_empty()
        || kind != kind.trim()
        || kind.len() > MAX_EVIDENCE_KIND_BYTES
        || kind.chars().any(char::is_control)
    {
        return Err(EvidenceCompactionError::InvalidKind);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn policy() -> EvidenceCompactionPolicy {
        EvidenceCompactionPolicy {
            minimum_retained_tail: 2,
            maximum_retained_tail: 4,
        }
    }

    #[test]
    fn compacted_tail_reconstructs_the_exact_final_chain_head() {
        let mut journal = EvidenceJournal::default();
        for index in 0..6 {
            journal
                .append(100 + index, "gateway-event", sha256(&[index as u8]))
                .unwrap();
        }
        let compacted = compact_evidence(&journal, 2, None, &policy()).unwrap();
        assert_eq!(compacted.prefix_count, 4);
        assert_eq!(compacted.final_head, journal.head());
        compacted.validate(&policy()).unwrap();

        let mut altered = compacted.clone();
        altered.retained_tail[0].subject_digest = sha256(b"altered");
        assert!(matches!(
            altered.validate(&policy()),
            Err(EvidenceCompactionError::RecordDigestMismatch { .. })
        ));
    }

    #[test]
    fn tracker_requires_checkpoint_linkage_for_later_compaction() {
        let mut first_journal = EvidenceJournal::default();
        for index in 0..4 {
            first_journal
                .append(index, "event", sha256(&[index as u8]))
                .unwrap();
        }
        let first = compact_evidence(&first_journal, 2, None, &policy()).unwrap();
        let mut tracker = EvidenceCompactionTracker::default();
        let first_digest = tracker.accept(&first, &policy()).unwrap();

        first_journal.append(5, "event", sha256(b"5")).unwrap();
        let unlinked = compact_evidence(&first_journal, 2, None, &policy()).unwrap();
        assert_eq!(
            tracker.accept(&unlinked, &policy()),
            Err(EvidenceCompactionTrackingError::MissingPredecessor)
        );
        let linked = compact_evidence(&first_journal, 2, Some(first_digest), &policy()).unwrap();
        tracker.accept(&linked, &policy()).unwrap();
    }
}
