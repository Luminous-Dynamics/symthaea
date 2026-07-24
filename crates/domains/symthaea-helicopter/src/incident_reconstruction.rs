// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic incident timeline reconstruction.
//!
//! Records from commands, sensors, faults, alerts, updates, and maintenance are
//! normalized into one ordered timeline. Explicit parent links generate
//! candidate causal chains, but the module never upgrades correlation or an
//! operator annotation into proof of causation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncidentRecordKind {
    Command,
    Sensor,
    Fault,
    Alert,
    ModeChange,
    SoftwareUpdate,
    Maintenance,
    EvidenceGap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentRecord {
    pub record_id: String,
    pub timestamp_ms: u64,
    pub source_id: String,
    pub source_sequence: u64,
    pub kind: IncidentRecordKind,
    pub summary: String,
    pub payload_digest: String,
    pub causal_parent_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncidentReconstructionStatus {
    CompleteCandidateTimeline,
    Incomplete,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncidentReconstructionIssue {
    EmptyIdentity(String),
    DuplicateRecord(String),
    InvalidDigest(String),
    MissingEvidence(String),
    NonMonotonicSourceSequence {
        source_id: String,
        previous: u64,
        observed: u64,
    },
    NonMonotonicSourceTime {
        source_id: String,
        previous_ms: u64,
        observed_ms: u64,
    },
    DuplicateParent {
        record_id: String,
        parent_id: String,
    },
    MissingParent {
        record_id: String,
        parent_id: String,
    },
    ParentNotEarlier {
        record_id: String,
        parent_id: String,
    },
    ExplicitEvidenceGap(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateCausalLink {
    pub parent_id: String,
    pub child_id: String,
    pub explicitly_declared: bool,
    pub causation_proven: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentTimelineEntry {
    pub ordinal: usize,
    pub record: IncidentRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentReconstructionReport {
    pub schema_version: String,
    pub incident_id: String,
    pub status: IncidentReconstructionStatus,
    pub timeline: Vec<IncidentTimelineEntry>,
    pub candidate_links: Vec<CandidateCausalLink>,
    pub root_record_ids: Vec<String>,
    pub issues: Vec<IncidentReconstructionIssue>,
    pub causation_proven: bool,
}

impl IncidentReconstructionReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, IncidentReconstructionError> {
        let mut canonical = self.clone();
        canonical.root_record_ids.sort();
        canonical
            .candidate_links
            .sort_by_key(|link| (link.parent_id.clone(), link.child_id.clone()));
        canonical.issues.sort_by_key(|issue| format!("{issue:?}"));
        serde_json::to_vec(&canonical).map_err(|_| IncidentReconstructionError::SerializationFailed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentReconstructionError {
    InvalidIncidentId,
    SerializationFailed,
}

#[derive(Debug, Clone, Default)]
pub struct IncidentReconstructor;

impl IncidentReconstructor {
    pub fn reconstruct(
        &self,
        incident_id: &str,
        records: &[IncidentRecord],
    ) -> Result<IncidentReconstructionReport, IncidentReconstructionError> {
        if incident_id.trim().is_empty() {
            return Err(IncidentReconstructionError::InvalidIncidentId);
        }
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut by_id = BTreeMap::new();
        for record in records {
            if record.record_id.trim().is_empty()
                || record.source_id.trim().is_empty()
                || record.summary.trim().is_empty()
            {
                issues.push(IncidentReconstructionIssue::EmptyIdentity(
                    record.record_id.clone(),
                ));
            }
            if !ids.insert(record.record_id.clone()) {
                issues.push(IncidentReconstructionIssue::DuplicateRecord(
                    record.record_id.clone(),
                ));
            }
            if !valid_digest(&record.payload_digest) {
                issues.push(IncidentReconstructionIssue::InvalidDigest(
                    record.record_id.clone(),
                ));
            }
            if record.evidence_ids.is_empty()
                || record.evidence_ids.iter().any(|id| id.trim().is_empty())
            {
                issues.push(IncidentReconstructionIssue::MissingEvidence(
                    record.record_id.clone(),
                ));
            }
            if record.kind == IncidentRecordKind::EvidenceGap {
                issues.push(IncidentReconstructionIssue::ExplicitEvidenceGap(
                    record.record_id.clone(),
                ));
            }
            by_id.entry(record.record_id.clone()).or_insert(record);
        }

        let mut ordered: Vec<_> = records.iter().collect();
        ordered.sort_by(|left, right| {
            left.timestamp_ms
                .cmp(&right.timestamp_ms)
                .then_with(|| left.source_id.cmp(&right.source_id))
                .then_with(|| left.source_sequence.cmp(&right.source_sequence))
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        let mut last_by_source = BTreeMap::<&str, (u64, u64)>::new();
        for record in &ordered {
            if let Some((previous_sequence, previous_ms)) =
                last_by_source.get(record.source_id.as_str())
            {
                if record.source_sequence <= *previous_sequence {
                    issues.push(IncidentReconstructionIssue::NonMonotonicSourceSequence {
                        source_id: record.source_id.clone(),
                        previous: *previous_sequence,
                        observed: record.source_sequence,
                    });
                }
                if record.timestamp_ms < *previous_ms {
                    issues.push(IncidentReconstructionIssue::NonMonotonicSourceTime {
                        source_id: record.source_id.clone(),
                        previous_ms: *previous_ms,
                        observed_ms: record.timestamp_ms,
                    });
                }
            }
            last_by_source.insert(
                record.source_id.as_str(),
                (record.source_sequence, record.timestamp_ms),
            );
        }

        let ordinal_by_id: BTreeMap<_, _> = ordered
            .iter()
            .enumerate()
            .map(|(ordinal, record)| (record.record_id.as_str(), ordinal))
            .collect();
        let mut candidate_links = Vec::new();
        let mut child_ids = BTreeSet::new();
        for record in &ordered {
            let mut parents = BTreeSet::new();
            for parent_id in &record.causal_parent_ids {
                if !parents.insert(parent_id) {
                    issues.push(IncidentReconstructionIssue::DuplicateParent {
                        record_id: record.record_id.clone(),
                        parent_id: parent_id.clone(),
                    });
                    continue;
                }
                let Some(parent) = by_id.get(parent_id) else {
                    issues.push(IncidentReconstructionIssue::MissingParent {
                        record_id: record.record_id.clone(),
                        parent_id: parent_id.clone(),
                    });
                    continue;
                };
                let parent_ordinal = ordinal_by_id
                    .get(parent.record_id.as_str())
                    .copied()
                    .unwrap_or(usize::MAX);
                let child_ordinal = ordinal_by_id
                    .get(record.record_id.as_str())
                    .copied()
                    .unwrap_or(usize::MAX);
                if parent.timestamp_ms > record.timestamp_ms || parent_ordinal >= child_ordinal {
                    issues.push(IncidentReconstructionIssue::ParentNotEarlier {
                        record_id: record.record_id.clone(),
                        parent_id: parent_id.clone(),
                    });
                    continue;
                }
                child_ids.insert(record.record_id.clone());
                candidate_links.push(CandidateCausalLink {
                    parent_id: parent_id.clone(),
                    child_id: record.record_id.clone(),
                    explicitly_declared: true,
                    causation_proven: false,
                });
            }
        }

        let timeline = ordered
            .into_iter()
            .enumerate()
            .map(|(ordinal, record)| IncidentTimelineEntry {
                ordinal,
                record: record.clone(),
            })
            .collect::<Vec<_>>();
        let root_record_ids = timeline
            .iter()
            .filter(|entry| !child_ids.contains(&entry.record.record_id))
            .map(|entry| entry.record.record_id.clone())
            .collect();

        let rejected = issues.iter().any(|issue| {
            matches!(
                issue,
                IncidentReconstructionIssue::DuplicateRecord(_)
                    | IncidentReconstructionIssue::ParentNotEarlier { .. }
                    | IncidentReconstructionIssue::NonMonotonicSourceSequence { .. }
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                IncidentReconstructionIssue::EmptyIdentity(_)
                    | IncidentReconstructionIssue::InvalidDigest(_)
                    | IncidentReconstructionIssue::MissingEvidence(_)
                    | IncidentReconstructionIssue::MissingParent { .. }
                    | IncidentReconstructionIssue::DuplicateParent { .. }
                    | IncidentReconstructionIssue::ExplicitEvidenceGap(_)
                    | IncidentReconstructionIssue::NonMonotonicSourceTime { .. }
            )
        });
        let status = if rejected {
            IncidentReconstructionStatus::Rejected
        } else if incomplete {
            IncidentReconstructionStatus::Incomplete
        } else {
            IncidentReconstructionStatus::CompleteCandidateTimeline
        };

        Ok(IncidentReconstructionReport {
            schema_version: "1".into(),
            incident_id: incident_id.to_string(),
            status,
            timeline,
            candidate_links,
            root_record_ids,
            issues,
            causation_proven: false,
        })
    }
}

fn valid_digest(digest: &str) -> bool {
    let digest = digest.trim();
    digest.starts_with("sha256:") && digest.len() > "sha256:".len()
        || digest.starts_with("fnv1a64:") && digest.len() == "fnv1a64:".len() + 16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, timestamp_ms: u64, source: &str, sequence: u64) -> IncidentRecord {
        IncidentRecord {
            record_id: id.into(),
            timestamp_ms,
            source_id: source.into(),
            source_sequence: sequence,
            kind: IncidentRecordKind::Fault,
            summary: format!("record {id}"),
            payload_digest: format!("sha256:{id}"),
            causal_parent_ids: Vec::new(),
            evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    #[test]
    fn timeline_is_deterministically_ordered() {
        let a = record("a", 100, "sensor", 1);
        let b = record("b", 90, "command", 1);
        let report = IncidentReconstructor
            .reconstruct("incident-1", &[a, b])
            .unwrap();
        assert_eq!(report.timeline[0].record.record_id, "b");
        assert_eq!(
            report.status,
            IncidentReconstructionStatus::CompleteCandidateTimeline
        );
    }

    #[test]
    fn declared_link_remains_candidate_only() {
        let a = record("a", 100, "sensor", 1);
        let mut b = record("b", 110, "fault", 1);
        b.causal_parent_ids.push("a".into());
        let report = IncidentReconstructor
            .reconstruct("incident-1", &[a, b])
            .unwrap();
        assert_eq!(report.candidate_links.len(), 1);
        assert!(!report.candidate_links[0].causation_proven);
        assert!(!report.causation_proven);
    }

    #[test]
    fn missing_parent_is_incomplete() {
        let mut a = record("a", 100, "sensor", 1);
        a.causal_parent_ids.push("missing".into());
        let report = IncidentReconstructor
            .reconstruct("incident-1", &[a])
            .unwrap();
        assert_eq!(report.status, IncidentReconstructionStatus::Incomplete);
    }

    #[test]
    fn backward_parent_is_rejected() {
        let mut a = record("a", 100, "sensor", 1);
        a.causal_parent_ids.push("b".into());
        let b = record("b", 110, "fault", 1);
        let report = IncidentReconstructor
            .reconstruct("incident-1", &[a, b])
            .unwrap();
        assert_eq!(report.status, IncidentReconstructionStatus::Rejected);
    }
}
