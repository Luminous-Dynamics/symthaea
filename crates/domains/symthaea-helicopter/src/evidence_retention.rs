// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded evidence retention with explicit loss accounting.
//!
//! Flight evidence buffers are finite. A recorder must therefore expose every
//! rejection and eviction rather than silently dropping data under pressure.
//! This module provides deterministic priority-aware retention while protecting
//! safety-critical records from lower-priority traffic.

use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidencePriority {
    Routine,
    Diagnostic,
    SafetyCritical,
}

impl EvidencePriority {
    fn index(self) -> usize {
        match self {
            Self::Routine => 0,
            Self::Diagnostic => 1,
            Self::SafetyCritical => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecordMetadata {
    pub record_id: String,
    pub timestamp_ns: u64,
    pub priority: EvidencePriority,
    pub encoded_size_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionPolicy {
    pub maximum_records: usize,
    pub maximum_bytes: usize,
    pub maximum_evictions_per_insert: usize,
}

impl Default for EvidenceRetentionPolicy {
    fn default() -> Self {
        Self {
            maximum_records: 8_192,
            maximum_bytes: 32 * 1024 * 1024,
            maximum_evictions_per_insert: 128,
        }
    }
}

impl EvidenceRetentionPolicy {
    pub fn validate(&self) -> Result<(), EvidenceRetentionError> {
        if self.maximum_records == 0
            || self.maximum_bytes == 0
            || self.maximum_evictions_per_insert == 0
        {
            return Err(EvidenceRetentionError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceRetentionDisposition {
    Stored,
    StoredAfterEviction,
    RejectedCapacity,
    RejectedOversize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionResult {
    pub disposition: EvidenceRetentionDisposition,
    pub evicted_record_ids: Vec<String>,
    pub retained_records: usize,
    pub retained_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRetentionEvidence {
    pub retained_records: usize,
    pub retained_bytes: usize,
    pub accepted_records: u64,
    pub evicted_records_by_priority: [u64; 3],
    pub rejected_records_by_priority: [u64; 3],
    pub safety_critical_loss_events: u64,
    pub lossless_safety_critical: bool,
    pub oldest_timestamp_ns: Option<u64>,
    pub newest_timestamp_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceRetentionError {
    InvalidPolicy,
    EmptyRecordId,
    DuplicateRecordId,
    ZeroSizeRecord,
    TimestampRegression,
}

#[derive(Debug, Clone)]
pub struct EvidenceRetentionBuffer {
    policy: EvidenceRetentionPolicy,
    records: VecDeque<EvidenceRecordMetadata>,
    record_ids: BTreeSet<String>,
    retained_bytes: usize,
    accepted_records: u64,
    evicted_records_by_priority: [u64; 3],
    rejected_records_by_priority: [u64; 3],
    safety_critical_loss_events: u64,
    newest_timestamp_ns: Option<u64>,
}

impl EvidenceRetentionBuffer {
    pub fn new(policy: EvidenceRetentionPolicy) -> Result<Self, EvidenceRetentionError> {
        policy.validate()?;
        Ok(Self {
            policy,
            records: VecDeque::new(),
            record_ids: BTreeSet::new(),
            retained_bytes: 0,
            accepted_records: 0,
            evicted_records_by_priority: [0; 3],
            rejected_records_by_priority: [0; 3],
            safety_critical_loss_events: 0,
            newest_timestamp_ns: None,
        })
    }

    pub fn push(
        &mut self,
        record: EvidenceRecordMetadata,
    ) -> Result<EvidenceRetentionResult, EvidenceRetentionError> {
        if record.record_id.trim().is_empty() {
            return Err(EvidenceRetentionError::EmptyRecordId);
        }
        if record.encoded_size_bytes == 0 {
            return Err(EvidenceRetentionError::ZeroSizeRecord);
        }
        if self.record_ids.contains(&record.record_id) {
            return Err(EvidenceRetentionError::DuplicateRecordId);
        }
        if self
            .newest_timestamp_ns
            .is_some_and(|latest| record.timestamp_ns < latest)
        {
            return Err(EvidenceRetentionError::TimestampRegression);
        }
        if record.encoded_size_bytes > self.policy.maximum_bytes {
            self.account_rejection(record.priority);
            return Ok(self.result(EvidenceRetentionDisposition::RejectedOversize, vec![]));
        }

        let must_free_records = self.records.len() + 1 > self.policy.maximum_records;
        let must_free_bytes =
            self.retained_bytes + record.encoded_size_bytes > self.policy.maximum_bytes;
        let mut selected_indices = Vec::new();
        if must_free_records || must_free_bytes {
            let mut candidates: Vec<_> = self
                .records
                .iter()
                .enumerate()
                .filter(|(_, retained)| retained.priority <= record.priority)
                .map(|(index, retained)| (index, retained.priority, retained.encoded_size_bytes))
                .collect();
            candidates.sort_by_key(|(index, priority, _)| (*priority, *index));

            let mut simulated_records = self.records.len();
            let mut simulated_bytes = self.retained_bytes;
            for (index, _, size_bytes) in candidates {
                if simulated_records + 1 <= self.policy.maximum_records
                    && simulated_bytes + record.encoded_size_bytes <= self.policy.maximum_bytes
                {
                    break;
                }
                if selected_indices.len() >= self.policy.maximum_evictions_per_insert {
                    break;
                }
                selected_indices.push(index);
                simulated_records = simulated_records.saturating_sub(1);
                simulated_bytes = simulated_bytes.saturating_sub(size_bytes);
            }
            if simulated_records + 1 > self.policy.maximum_records
                || simulated_bytes + record.encoded_size_bytes > self.policy.maximum_bytes
            {
                self.account_rejection(record.priority);
                return Ok(self.result(EvidenceRetentionDisposition::RejectedCapacity, Vec::new()));
            }
        }

        let evicted_record_ids: Vec<_> = selected_indices
            .iter()
            .map(|index| self.records[*index].record_id.clone())
            .collect();
        selected_indices.sort_unstable_by(|left, right| right.cmp(left));
        for index in selected_indices {
            let evicted = self
                .records
                .remove(index)
                .expect("planned index originated from the same deque");
            self.record_ids.remove(&evicted.record_id);
            self.retained_bytes = self
                .retained_bytes
                .saturating_sub(evicted.encoded_size_bytes);
            self.evicted_records_by_priority[evicted.priority.index()] =
                self.evicted_records_by_priority[evicted.priority.index()].saturating_add(1);
            if evicted.priority == EvidencePriority::SafetyCritical {
                self.safety_critical_loss_events =
                    self.safety_critical_loss_events.saturating_add(1);
            }
        }

        self.retained_bytes += record.encoded_size_bytes;
        self.newest_timestamp_ns = Some(record.timestamp_ns);
        self.record_ids.insert(record.record_id.clone());
        self.records.push_back(record);
        self.accepted_records = self.accepted_records.saturating_add(1);
        let disposition = if evicted_record_ids.is_empty() {
            EvidenceRetentionDisposition::Stored
        } else {
            EvidenceRetentionDisposition::StoredAfterEviction
        };
        Ok(self.result(disposition, evicted_record_ids))
    }

    pub fn evidence(&self) -> EvidenceRetentionEvidence {
        EvidenceRetentionEvidence {
            retained_records: self.records.len(),
            retained_bytes: self.retained_bytes,
            accepted_records: self.accepted_records,
            evicted_records_by_priority: self.evicted_records_by_priority,
            rejected_records_by_priority: self.rejected_records_by_priority,
            safety_critical_loss_events: self.safety_critical_loss_events,
            lossless_safety_critical: self.safety_critical_loss_events == 0,
            oldest_timestamp_ns: self.records.front().map(|record| record.timestamp_ns),
            newest_timestamp_ns: self.records.back().map(|record| record.timestamp_ns),
        }
    }

    pub fn records(&self) -> impl Iterator<Item = &EvidenceRecordMetadata> {
        self.records.iter()
    }

    fn account_rejection(&mut self, priority: EvidencePriority) {
        self.rejected_records_by_priority[priority.index()] =
            self.rejected_records_by_priority[priority.index()].saturating_add(1);
        if priority == EvidencePriority::SafetyCritical {
            self.safety_critical_loss_events = self.safety_critical_loss_events.saturating_add(1);
        }
    }

    fn result(
        &self,
        disposition: EvidenceRetentionDisposition,
        evicted_record_ids: Vec<String>,
    ) -> EvidenceRetentionResult {
        EvidenceRetentionResult {
            disposition,
            evicted_record_ids,
            retained_records: self.records.len(),
            retained_bytes: self.retained_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, timestamp_ns: u64, priority: EvidencePriority) -> EvidenceRecordMetadata {
        EvidenceRecordMetadata {
            record_id: id.into(),
            timestamp_ns,
            priority,
            encoded_size_bytes: 10,
        }
    }

    #[test]
    fn safety_record_evicts_old_routine_record() {
        let mut buffer = EvidenceRetentionBuffer::new(EvidenceRetentionPolicy {
            maximum_records: 2,
            maximum_bytes: 20,
            maximum_evictions_per_insert: 2,
        })
        .unwrap();
        buffer
            .push(record("routine-1", 1, EvidencePriority::Routine))
            .unwrap();
        buffer
            .push(record("routine-2", 2, EvidencePriority::Routine))
            .unwrap();
        let result = buffer
            .push(record("critical", 3, EvidencePriority::SafetyCritical))
            .unwrap();
        assert_eq!(
            result.disposition,
            EvidenceRetentionDisposition::StoredAfterEviction
        );
        assert_eq!(result.evicted_record_ids, vec!["routine-1"]);
        assert!(buffer.evidence().lossless_safety_critical);
    }

    #[test]
    fn routine_record_cannot_evict_safety_record() {
        let mut buffer = EvidenceRetentionBuffer::new(EvidenceRetentionPolicy {
            maximum_records: 1,
            maximum_bytes: 10,
            maximum_evictions_per_insert: 1,
        })
        .unwrap();
        buffer
            .push(record("critical", 1, EvidencePriority::SafetyCritical))
            .unwrap();
        let result = buffer
            .push(record("routine", 2, EvidencePriority::Routine))
            .unwrap();
        assert_eq!(
            result.disposition,
            EvidenceRetentionDisposition::RejectedCapacity
        );
        assert_eq!(buffer.records().next().unwrap().record_id, "critical");
        assert_eq!(buffer.evidence().rejected_records_by_priority[0], 1);
    }

    #[test]
    fn rejected_critical_record_breaks_lossless_claim() {
        let mut buffer = EvidenceRetentionBuffer::new(EvidenceRetentionPolicy {
            maximum_records: 1,
            maximum_bytes: 10,
            maximum_evictions_per_insert: 1,
        })
        .unwrap();
        let mut oversized = record("critical", 1, EvidencePriority::SafetyCritical);
        oversized.encoded_size_bytes = 11;
        let result = buffer.push(oversized).unwrap();
        assert_eq!(
            result.disposition,
            EvidenceRetentionDisposition::RejectedOversize
        );
        assert!(!buffer.evidence().lossless_safety_critical);
    }

    #[test]
    fn timestamp_regression_is_rejected() {
        let mut buffer = EvidenceRetentionBuffer::new(EvidenceRetentionPolicy::default()).unwrap();
        buffer
            .push(record("a", 10, EvidencePriority::Routine))
            .unwrap();
        assert_eq!(
            buffer.push(record("b", 9, EvidencePriority::Routine)),
            Err(EvidenceRetentionError::TimestampRegression)
        );
    }
}
